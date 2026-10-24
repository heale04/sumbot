"""PyTorch port of `solver.cfr`: the same abstract game and the same regret minimizers
(CFR+ and MI-SPRM+ with the paper's global adaptive R), compiled to static level-wise
dataflow in the spirit of GPU-CFR (Li & Huang): every iteration is a fixed sequence of
batched gathers, products and scatters, so on CUDA a half-iteration is captured once as a
CUDA graph and replayed.

Layout: node vectors (reach, values) live in one flat tensor with the same offsets as the
Numba solver (`voff`). Infoset state is stored per level group (street, depth) as dense
[n_nodes, K, A_max] tensors padded with masked actions, and converted to the Numba flat
layout (`ioff`, index = b * A + a) for export, checkpoints and warm starts, so the two
solvers are interchangeable.

Exploitability and the game value are evaluated in float64.
"""

import numpy as np
import torch

from solver.cfr import MI_SPRM_PLUS, PRM_PLUS, RM_PLUS, SPRM_PLUS, match_state
from solver.tree import CHANCE, DECISION, FOLD, SHOWDOWN

NEG = -1e30


class TorchSolver:
    def __init__(self, tree, abstraction, mode=MI_SPRM_PLUS, fixed_R=1.0, avg_weight="quad", use_r_weight=True,
                 ard="global", device="cpu", dtype=torch.float32, cuda_graph=None):
        if ard != "global":
            raise ValueError("TorchSolver supports the paper's global adaptive R only (ard='global')")
        self.tree = tree
        self.mode = mode
        self.device = torch.device(device)
        self.dtype = dtype
        self.avg_weight = {"const": 0, "lin": 1, "quad": 2}[avg_weight]
        self.use_r_weight = bool(use_r_weight and mode >= 2)
        self.cuda_graph = (self.device.type == "cuda") if cuda_graph is None else bool(cuda_graph)
        arrs = tree.arrays()
        self.node_type = arrs["node_type"]
        self.player = arrs["player"]
        self.street = arrs["street"]
        self.child = arrs["child"]
        self.n_actions = arrs["n_actions"]
        self.term_amount = arrs["term_amount"]
        self.term_winner = arrs["term_winner"]
        K = int(abstraction["K"])
        self.K_s = np.array([len(abstraction["p_class"]), K, K, K], dtype=np.int64)
        n = self.n = len(self.node_type)
        self.voff = np.zeros(n + 1, dtype=np.int64)
        self.ioff = np.zeros(n + 1, dtype=np.int64)
        for i in range(n):
            k = self.K_s[self.street[i]]
            self.voff[i + 1] = self.voff[i] + k
            self.ioff[i + 1] = self.ioff[i] + (k * self.n_actions[i] if self.node_type[i] == DECISION else 0)
        self.V, self.I = int(self.voff[-1]), int(self.ioff[-1])
        self.A_max = int(self.n_actions.max())
        # depth within the street (decision nodes); the acting player alternates with depth
        depth = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if self.node_type[i] != DECISION:
                continue
            for a in range(self.n_actions[i]):
                c = self.child[i, a]
                if self.node_type[c] == DECISION:
                    depth[c] = depth[i] + 1
        self.depth = depth
        dev = self.device
        Kmax = int(self.K_s.max())
        self.dummy = self.V  # one zero row of length Kmax after the real vectors
        Vp = self.V + Kmax

        # ---- level groups: (street, depth) -> decision nodes, processed in order
        self.groups = []  # list of dicts in forward order
        for s in range(4):
            ds = depth[(self.node_type == DECISION) & (self.street == s)]
            for d in range(int(ds.max()) + 1 if len(ds) else 0):
                nodes = np.nonzero((self.node_type == DECISION) & (self.street == s) & (depth == d))[0]
                if len(nodes) == 0:
                    continue
                Ks = int(self.K_s[s])
                ng = len(nodes)
                pl = int(self.player[nodes[0]])
                assert (self.player[nodes] == pl).all()
                vidx = self.voff[nodes][:, None] + np.arange(Ks)[None, :]  # [ng, K]
                cidx = np.full((ng, self.A_max, Ks), 0, dtype=np.int64)
                mask = np.zeros((ng, self.A_max), dtype=bool)
                for r, i in enumerate(nodes):
                    for a in range(self.A_max):
                        if a < self.n_actions[i]:
                            cidx[r, a] = self.voff[self.child[i, a]] + np.arange(Ks)
                            mask[r, a] = True
                        else:
                            cidx[r, a] = self.dummy + np.arange(Ks)
                # positions of real (node, action, bucket) entries in the [ng, A_max, K] child tensor
                sel = np.nonzero(np.repeat(mask[:, :, None], Ks, axis=2).ravel())[0]
                cflat = cidx.ravel()[sel]
                # flat (Numba-layout) infoset index for each [ng, K, A_max] entry (pads -> -1)
                fidx = np.full((ng, Ks, self.A_max), -1, dtype=np.int64)
                for r, i in enumerate(nodes):
                    A = int(self.n_actions[i])
                    fidx[r, :, :A] = self.ioff[i] + np.arange(Ks)[:, None] * A + np.arange(A)[None, :]
                g = dict(
                    street=s, depth=d, player=pl, n=ng, K=Ks, nodes=nodes,
                    vidx=torch.as_tensor(vidx, device=dev), cidx=torch.as_tensor(cidx, device=dev),
                    csel=torch.as_tensor(sel, device=dev), cflat=torch.as_tensor(cflat, device=dev),
                    mask=torch.as_tensor(mask[:, None, :], device=dev),  # [ng, 1, A_max]
                    fidx=fidx, n_act=torch.as_tensor(self.n_actions[nodes], device=dev)[:, None, None].to(dtype),
                )
                self.groups.append(g)
        self.groups_by_street = [[g for g in self.groups if g["street"] == s] for s in range(4)]
        # ---- chance nodes per street
        self.chance = []
        for s in range(3):
            nodes = np.nonzero((self.node_type == CHANCE) & (self.street == s))[0]
            if len(nodes) == 0:
                self.chance.append(None)
                continue
            Kf, Kt = int(self.K_s[s]), int(self.K_s[s + 1])
            src = self.voff[nodes][:, None] + np.arange(Kf)[None, :]
            dst = self.voff[self.child[nodes, 0]][:, None] + np.arange(Kt)[None, :]
            self.chance.append(dict(src=torch.as_tensor(src, device=dev), dst=torch.as_tensor(dst, device=dev)))
        # ---- terminals per street
        self.terms = []
        for s in range(4):
            Ks = int(self.K_s[s])
            sd = np.nonzero((self.node_type == SHOWDOWN) & (self.street == s))[0]
            fo = np.nonzero((self.node_type == FOLD) & (self.street == s))[0]
            t = dict(K=Ks)
            if len(sd):
                t["sd_idx"] = torch.as_tensor(self.voff[sd][:, None] + np.arange(Ks)[None, :], device=dev)
                t["sd_amt"] = self.term_amount[sd][:, None]
            if len(fo):
                t["fo_idx"] = torch.as_tensor(self.voff[fo][:, None] + np.arange(Ks)[None, :], device=dev)
                amt = self.term_amount[fo][:, None]
                win = self.term_winner[fo][:, None]
                t["fo_amt"] = [np.where(win == p, amt, -amt) for p in (0, 1)]  # signed for player p
            self.terms.append(t)
        # ---- tables in both precisions
        self.tab = {}
        for want in (dtype, torch.float64):
            tb = dict(
                p_class=torch.as_tensor(np.asarray(abstraction["p_class"], dtype=np.float64), device=dev).to(want),
                T=[torch.as_tensor(np.asarray(abstraction[k], dtype=np.float64), device=dev).to(want)
                   for k in ("T_pre", "T_flop", "T_turn")],
            )
            W0 = torch.as_tensor(np.asarray(abstraction["W0"], dtype=np.float64), device=dev).to(want)
            Wp = torch.as_tensor(np.asarray(abstraction["W_post"], dtype=np.float64), device=dev).to(want)
            tb["WT"] = [W0.T.contiguous()] + [Wp[s].T.contiguous() for s in range(3)]
            tb["TT"] = [T.T.contiguous() for T in tb["T"]]
            for s, t in enumerate(self.terms):
                if "sd_idx" in t:
                    t[("sd_amt", want)] = torch.as_tensor(t["sd_amt"], device=dev).to(want)
                if "fo_idx" in t:
                    t[("fo_amt", want)] = [torch.as_tensor(x, device=dev).to(want) for x in t["fo_amt"]]
            self.tab[want] = tb
        # ---- state
        z = lambda shape, dt=dtype: torch.zeros(shape, device=dev, dtype=dt)
        self.reach = [z(Vp), z(Vp)]
        self.val = z(Vp)
        self.reach64 = [z(Vp, torch.float64), z(Vp, torch.float64)]
        self.val64 = z(Vp, torch.float64)
        for g in self.groups:
            shape = (g["n"], g["K"], self.A_max)
            maskf = g["mask"].to(dtype)
            init = (maskf / g["n_act"]).expand(shape).clone()
            g["strat"] = init.clone()
            g["theta_hat"] = init.clone() if mode >= 1 else z(shape)
            g["theta_prev"] = init.clone() if mode >= 1 else z(shape)
            g["F_prev"] = z(shape)
            g["strat_sum"] = z(shape)
            g["Bsum"] = z((g["n"], g["K"]))
        self.R_global = torch.full((2,), float(fixed_R), device=dev, dtype=dtype)
        self.t = torch.zeros(2, device=dev, dtype=torch.float64)
        self.graphs = None

    # ------------------------------------------------------------------ pieces
    def _project(self, vec, mask, R):
        """Projection of each row of vec [.., A] onto {x >= 0, sum x >= R}; padded entries are
        masked out. R broadcasts to vec[..., :1]."""
        v = torch.where(mask, vec, torch.full_like(vec, NEG))
        clipped = torch.clamp(v, min=0.0)
        s = clipped.sum(-1, keepdim=True)
        srt, _ = torch.sort(v, dim=-1, descending=True)
        css = torch.cumsum(srt, dim=-1)
        k = torch.arange(1, vec.shape[-1] + 1, device=vec.device, dtype=vec.dtype)
        cand = (css - R) / k
        valid = (srt - cand) > 0
        # tau = cand at the last valid k (the Numba kernel keeps the last one that passes)
        idx = (valid.to(torch.int64) * torch.arange(1, vec.shape[-1] + 1, device=vec.device)).max(-1, keepdim=True).values
        tau = torch.gather(cand, -1, torch.clamp(idx - 1, min=0)) * (idx > 0).to(vec.dtype)
        proj = torch.clamp(v - tau, min=0.0)
        return torch.where(s >= R, clipped, proj)

    def _strategy_step(self, p):
        R = self.R_global[p]
        for g in self.groups:
            if g["player"] != p:
                continue
            mask = g["mask"]
            if self.mode == RM_PLUS:
                th = g["theta_hat"]
            else:
                tmp = g["theta_hat"] + g["F_prev"]
                th = torch.clamp(tmp, min=0.0) * mask if self.mode == PRM_PLUS else self._project(tmp, mask, R)
            s = th.sum(-1, keepdim=True)
            uniform = mask.to(th.dtype) / g["n_act"]
            # all state updates are in place: the CUDA graphs reference these buffers
            g["strat"].copy_(torch.where(s > 0, th / torch.where(s > 0, s, torch.ones_like(s)), uniform))
            if self.mode != RM_PLUS:
                d1 = g["theta_hat"] - g["theta_prev"]
                d2 = g["theta_hat"] - th
                g["Bsum"].copy_(0.5 * (d1 * d1).sum(-1) + 0.5 * (d2 * d2).sum(-1))
                g["theta_prev"].copy_(th)

    def _forward(self, reach, strat_of, tb):
        K0 = int(self.K_s[0])
        for q in (0, 1):
            reach[q][:K0] = tb["p_class"]
        for s in range(4):
            for g in self.groups_by_street[s]:
                pl = g["player"]
                st = strat_of(g)  # [n, K, A]
                for q in (0, 1):
                    par = torch.take(reach[q], g["vidx"])  # [n, K]
                    if q == pl:
                        ch = (par[:, :, None] * st).permute(0, 2, 1)  # [n, A, K]
                    else:
                        ch = par[:, None, :].expand(-1, self.A_max, -1)
                    reach[q].index_put_((g["cflat"],), torch.take(ch.contiguous(), g["csel"]))
            if s < 3 and self.chance[s] is not None:
                c = self.chance[s]
                for q in (0, 1):
                    src = torch.take(reach[q], c["src"])
                    reach[q].index_put_((c["dst"].reshape(-1),), (src @ tb["T"][s]).reshape(-1))

    def _terminals(self, p, reach_q, val, tb, want):
        for s, t in enumerate(self.terms):
            if "sd_idx" in t:
                rq = torch.take(reach_q, t["sd_idx"])
                val.index_put_((t["sd_idx"].reshape(-1),), (t[("sd_amt", want)] * (rq @ tb["WT"][s])).reshape(-1))
            if "fo_idx" in t:
                rq = torch.take(reach_q, t["fo_idx"])
                v = t[("fo_amt", want)][p] * rq.sum(1, keepdim=True)
                val.index_put_((t["fo_idx"].reshape(-1),), v.expand(-1, t["K"]).reshape(-1))

    def _backward(self, p):
        reach_q = self.reach[1 - p]
        reach_p = self.reach[p]
        val = self.val
        tb = self.tab[self.dtype]
        self._terminals(p, reach_q, val, tb, self.dtype)
        w = (self.t[p] ** self.avg_weight).to(self.dtype)
        R = self.R_global[p]
        wR = w * R if self.use_r_weight else w
        stat = torch.zeros((), device=self.device, dtype=self.dtype)
        for s in range(3, -1, -1):
            for g in reversed(self.groups_by_street[s]):
                u = torch.take(val, g["cidx"]).permute(0, 2, 1)  # [n, K, A]
                if g["player"] != p:
                    val.index_put_((g["vidx"].reshape(-1),), u.sum(-1).reshape(-1))
                    continue
                st = g["strat"]
                mask = g["mask"]
                node_v = (st * u).sum(-1, keepdim=True)
                F = (u - node_v) * mask
                wt = wR * torch.take(reach_p, g["vidx"])[:, :, None]
                g["strat_sum"].add_(wt * st)
                if self.mode == RM_PLUS or self.mode == PRM_PLUS:
                    g["theta_hat"].copy_(torch.clamp(g["theta_hat"] + F, min=0.0))
                else:
                    if self.mode == MI_SPRM_PLUS:
                        d = F - g["F_prev"]
                        stat = stat + ((d * d).sum(-1) - 0.5 * g["Bsum"]).sum()
                    g["theta_hat"].copy_(self._project(g["theta_hat"] + F, mask, R))
                g["F_prev"].copy_(F)
                val.index_put_((g["vidx"].reshape(-1),), node_v.reshape(-1))
            if s > 0 and self.chance[s - 1] is not None:
                c = self.chance[s - 1]
                vc = torch.take(val, c["dst"])
                val.index_put_((c["src"].reshape(-1),), (vc @ tb["TT"][s - 1]).reshape(-1))
        if self.mode == MI_SPRM_PLUS:
            self.R_global[p] += (stat > 0).to(self.dtype)

    def _half(self, p):
        self.t[p] += 1
        self._strategy_step(p)
        self._forward(self.reach, lambda g: g["strat"], self.tab[self.dtype])
        self._backward(p)

    # ------------------------------------------------------------------ iteration
    def iterate(self, n_iter=1):
        if self.cuda_graph and self.graphs is None:
            self._capture()
        for _ in range(n_iter):
            if self.graphs is not None:
                self.graphs[0].replay()
                self.graphs[1].replay()
            else:
                self._half(0)
                self._half(1)

    def _capture(self):
        """Record one half-iteration per player as a CUDA graph. The three warm-up iterations
        are ordinary iterations (they advance the state)."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._half(0)
                self._half(1)
        torch.cuda.current_stream().wait_stream(s)
        graphs = []
        for p in (0, 1):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._half(p)
            graphs.append(g)
        self.graphs = graphs

    # ------------------------------------------------------------------ export / evaluation
    def _flat(self, name, out_dtype=np.float64):
        """Infoset tensor `name` of every group in the Numba flat layout."""
        out = np.zeros(self.I, dtype=out_dtype)
        for g in self.groups:
            x = g[name].detach().cpu().numpy()
            m = g["fidx"] >= 0
            out[g["fidx"][m]] = x[m]
        return out

    def _set_flat(self, name, arr):
        for g in self.groups:
            m = g["fidx"] >= 0
            x = np.zeros(g["fidx"].shape, dtype=np.float64)
            x[m] = arr[g["fidx"][m]]
            g[name] = torch.as_tensor(x, device=self.device).to(g[name].dtype)

    @property
    def strat(self):
        return self._flat("strat")

    def average_strategy(self):
        out = np.zeros(self.I)
        for g in self.groups:
            ss = g["strat_sum"].to(torch.float64)
            tot = ss.sum(-1, keepdim=True)
            uniform = g["mask"].to(torch.float64) / g["n_act"].to(torch.float64)
            avg = torch.where(tot > 0, ss / torch.where(tot > 0, tot, torch.ones_like(tot)), uniform).cpu().numpy()
            m = g["fidx"] >= 0
            out[g["fidx"][m]] = avg[m]
        return out

    def _groups_from_flat(self, sigma):
        res = {}
        for gi, g in enumerate(self.groups):
            m = g["fidx"] >= 0
            x = np.zeros(g["fidx"].shape)
            x[m] = sigma[g["fidx"][m]]
            res[gi] = torch.as_tensor(x, device=self.device, dtype=torch.float64)
        return res

    def _evaluate(self, sigma, p, best):
        tb = self.tab[torch.float64]
        sg = self._groups_from_flat(sigma)
        self._forward(self.reach64, lambda g: sg[self.groups.index(g)], tb)
        val = self.val64
        self._terminals(p, self.reach64[1 - p], val, tb, torch.float64)
        for s in range(3, -1, -1):
            for g in reversed(self.groups_by_street[s]):
                u = torch.take(val, g["cidx"]).permute(0, 2, 1)
                if g["player"] != p:
                    v = u.sum(-1)
                elif best:
                    v = torch.where(g["mask"], u, torch.full_like(u, NEG)).max(-1).values
                else:
                    v = (sg[self.groups.index(g)] * u).sum(-1)
                val.index_put_((g["vidx"].reshape(-1),), v.reshape(-1))
            if s > 0 and self.chance[s - 1] is not None:
                c = self.chance[s - 1]
                vc = torch.take(val, c["dst"])
                val.index_put_((c["src"].reshape(-1),), (vc @ tb["TT"][s - 1]).reshape(-1))
        K0 = int(self.K_s[0])
        return float((tb["p_class"] * val[:K0]).sum().item())

    def exploitability(self, which="avg"):
        sigma = self.average_strategy() if which == "avg" else self.strat
        b0 = self._evaluate(sigma, 0, True)
        b1 = self._evaluate(sigma, 1, True)
        return (b0 + b1) / 2, b0, b1

    def game_value(self):
        return self._evaluate(self.average_strategy(), 0, False)

    # ------------------------------------------------------------------ state / warm start
    def state_dict(self):
        R = np.zeros(self.V, dtype=np.float32)
        Rg = self.R_global.detach().cpu().numpy().astype(np.float64)
        for g in self.groups:
            R[g["vidx"].cpu().numpy().ravel()] = Rg[g["player"]]
        return dict(theta_hat=self._flat("theta_hat", np.float32), theta_prev=self._flat("theta_prev", np.float32),
                    F_prev=self._flat("F_prev", np.float32), strat_sum=self._flat("strat_sum", np.float32),
                    R=R, R_global=Rg, t=self.t.cpu().numpy().astype(np.int64),
                    keys=np.array(self.tree.key, dtype=str), tags=np.array([" ".join(t) for t in self.tree.tags], dtype=str),
                    n_actions=self.n_actions, ioff=self.ioff, voff=self.voff, node_type=self.node_type,
                    street=self.street, K_s=self.K_s, mode=self.mode)

    def load_state(self, st):
        for name in ("theta_hat", "theta_prev", "F_prev", "strat_sum"):
            self._set_flat(name, st[name].astype(np.float64))
        self.R_global = torch.as_tensor(np.asarray(st["R_global"], dtype=np.float64), device=self.device).to(self.dtype)
        self.t = torch.as_tensor(np.asarray(st["t"], dtype=np.float64), device=self.device)
        self.graphs = None  # tensors were replaced; re-capture

    def warm_start(self, st, copy_average=False, strategy=None, scale=None):
        m = match_state(self.tree, self.node_type, self.street, self.n_actions, self.ioff, self.voff, self.K_s, st)
        if m is None:
            return 0.0
        dst_i, src_i, _, _ = m
        cur = {name: self._flat(name) for name in ("theta_hat", "theta_prev", "F_prev", "strat_sum")}
        if strategy is not None:
            sc = float(scale if scale is not None else max(st["R_global"]))
            cur["theta_hat"][dst_i] = sc * strategy[src_i]
            cur["theta_prev"][dst_i] = cur["theta_hat"][dst_i]
            cur["F_prev"][dst_i] = 0.0
        else:
            for name in ("theta_hat", "theta_prev", "F_prev"):
                cur[name][dst_i] = st[name][src_i]
            if copy_average:
                cur["strat_sum"][dst_i] = st["strat_sum"][src_i]
                self.t = torch.as_tensor(np.asarray(st["t"], dtype=np.float64), device=self.device)
        for name, arr in cur.items():
            self._set_flat(name, arr)
        self.R_global = torch.as_tensor(np.asarray(st["R_global"], dtype=np.float64), device=self.device).to(self.dtype)
        self.graphs = None
        return len(dst_i) / self.I
