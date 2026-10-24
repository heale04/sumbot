"""Vectorized CFR over the abstract game with pluggable per-infoset regret minimizers.

Modes: 0 RM+ (CFR+), 1 PRM+ (PCFR+), 2 SPRM+ (fixed R), 3 MI-SPRM+ (adaptive R, Alg. 1 of
Meng et al., ICLR 2026). Alternating updates; the average strategy is weighted by
w(t) * R^t (R only for modes 2/3 when `use_r_weight`).

Reach vectors carry chance probabilities (initial class distribution and Markov transitions),
so terminal values are linear in the opponent's reach and chance nodes are matrix products.

Speed features (all exact with respect to the regret updates):
- `dtype=np.float32` keeps the per-node state in single precision; exploitability and the
  game value are always evaluated in float64.
- Terminal values are computed for all terminals of a street in one BLAS product instead of
  one small product per terminal.
- `prune=True` skips, for the traversing player, the value gathers in every subtree the
  opponent reaches with probability zero under the current strategy (partial pruning). The
  counterfactual values there are exactly zero, so the F = 0 update is applied in closed
  form; the trajectory is identical to the unpruned one.
- `state_dict()` / `load_state()` and `warm_start()` transfer regrets between trees whose
  nodes share action paths (e.g. neighbouring stacks).
"""

import numpy as np
from numba import njit

from solver.tree import CHANCE, DECISION, FOLD, SHOWDOWN

RM_PLUS, PRM_PLUS, SPRM_PLUS, MI_SPRM_PLUS = 0, 1, 2, 3
MODE_NAMES = {0: "CFR+", 1: "PCFR+", 2: "SPCFR+", 3: "MI-SPCFR+"}


class Solver:
    def __init__(self, tree, abstraction, mode=MI_SPRM_PLUS, fixed_R=1.0, avg_weight="quad",
                 use_r_weight=True, qre_tau=0.0, ard="global", dtype=np.float32, prune=True):
        arrs = tree.arrays()
        self.tree = tree
        self.dtype = np.dtype(dtype)
        self.node_type = arrs["node_type"]
        self.player = arrs["player"]
        self.street = arrs["street"]
        self.child = arrs["child"]
        self.n_actions = arrs["n_actions"]
        self.term_amount = arrs["term_amount"]
        self.term_winner = arrs["term_winner"]
        K = int(abstraction["K"])
        self.K_s = np.array([len(abstraction["p_class"]), K, K, K], dtype=np.int64)
        n = len(self.node_type)
        self.n = n
        self.voff = np.zeros(n + 1, dtype=np.int64)
        self.ioff = np.zeros(n + 1, dtype=np.int64)
        for i in range(n):
            k = self.K_s[self.street[i]]
            self.voff[i + 1] = self.voff[i] + k
            self.ioff[i + 1] = self.ioff[i] + (k * self.n_actions[i] if self.node_type[i] == DECISION else 0)
        V, I = self.voff[-1], self.ioff[-1]
        self.V, self.I = V, I
        dt = self.dtype
        self.theta_hat = np.zeros(I, dt)
        self.theta_prev = np.zeros(I, dt)
        self.F_prev = np.zeros(I, dt)
        self.strat = np.zeros(I, dt)
        self.strat_sum = np.zeros(I, dt)
        self.R = np.full(V, float(fixed_R), dt)
        self.Bsum = np.zeros(V, dt)
        self.reach0 = np.zeros(V, dt)
        self.reach1 = np.zeros(V, dt)
        self.val = np.zeros(V, dt)
        self.alive0 = np.ones(n, dtype=np.uint8)
        self.alive1 = np.ones(n, dtype=np.uint8)
        self.prune = bool(prune)
        self.mode = mode
        self.avg_weight = {"const": 0, "lin": 1, "quad": 2}[avg_weight]
        self.use_r_weight = bool(use_r_weight and mode >= 2)
        self.qre_tau = float(qre_tau)
        self.ard_global = ard == "global"
        self.R_global = np.full(2, float(fixed_R))
        self.ard_stat = np.zeros(1)
        self.t = np.zeros(2, dtype=np.int64)
        # chance / showdown tables, in the state dtype for iteration and float64 for evaluation
        self.tables = {}
        for want in (dt, np.dtype(np.float64)):
            W_post = np.ascontiguousarray(abstraction["W_post"], dtype=want)
            self.tables[want] = dict(
                p_class=np.ascontiguousarray(abstraction["p_class"], dtype=want),
                T_pre=np.ascontiguousarray(abstraction["T_pre"], dtype=want),
                T_flop=np.ascontiguousarray(abstraction["T_flop"], dtype=want),
                T_turn=np.ascontiguousarray(abstraction["T_turn"], dtype=want),
                W0=np.ascontiguousarray(abstraction["W0"], dtype=want),
                W_post=W_post,
                W0T=np.ascontiguousarray(abstraction["W0"].T, dtype=want),
                W_postT=np.ascontiguousarray(np.transpose(abstraction["W_post"], (0, 2, 1)), dtype=want),
            )
        tb = self.tables[dt]
        self.p_class, self.T_pre, self.T_flop, self.T_turn = tb["p_class"], tb["T_pre"], tb["T_flop"], tb["T_turn"]
        self.W0, self.W_post, self.W0T, self.W_postT = tb["W0"], tb["W_post"], tb["W0T"], tb["W_postT"]
        # showdown terminals grouped by street (for the batched terminal product)
        sd = [np.nonzero((self.node_type == SHOWDOWN) & (self.street == s))[0] for s in range(4)]
        self.sd_nodes = np.concatenate(sd).astype(np.int64)
        self.sd_off = np.cumsum([0] + [len(x) for x in sd]).astype(np.int64)
        self.fold_nodes = np.nonzero(self.node_type == FOLD)[0].astype(np.int64)
        self._eval = None  # float64 buffers for exploitability
        # initial theta as in the paper: 1/|A| everywhere (modes 1-3)
        init_theta(self.node_type, self.n_actions, self.ioff, self.K_s, self.street, self.theta_hat, self.theta_prev, self.strat, mode)

    # ------------------------------------------------------------------ iteration
    def iterate(self, n_iter=1):
        for _ in range(n_iter):
            for p in (0, 1):
                self.t[p] += 1
                strategy_step(p, self.mode, self.node_type, self.player, self.n_actions, self.ioff, self.voff,
                              self.K_s, self.street, self.theta_hat, self.theta_prev, self.F_prev, self.strat,
                              self.R, self.Bsum)
                forward(self.node_type, self.player, self.street, self.child, self.n_actions, self.voff, self.ioff,
                        self.K_s, self.strat, self.reach0, self.reach1, self.p_class, self.T_pre, self.T_flop, self.T_turn,
                        self.alive0, self.alive1)
                w = float(self.t[p]) ** self.avg_weight
                self.ard_stat[0] = 0.0
                reach_q = self.reach1 if p == 0 else self.reach0
                terminal_batch(p, self.sd_nodes, self.sd_off, self.fold_nodes, self.street, self.K_s, self.voff,
                               self.term_amount, self.term_winner, reach_q, self.val, self.W0T, self.W_postT)
                backward(p, self.mode, w, self.use_r_weight, self.qre_tau, self.ard_global, self.ard_stat,
                         self.prune, self.alive1 if p == 0 else self.alive0,
                         self.node_type, self.player, self.street,
                         self.child, self.n_actions, self.voff, self.ioff, self.K_s,
                         self.strat, self.reach0, self.reach1, self.val, self.theta_hat, self.F_prev, self.strat_sum,
                         self.R, self.Bsum, self.T_pre, self.T_flop, self.T_turn)
                if self.ard_global and self.mode == MI_SPRM_PLUS and self.ard_stat[0] > 0.0:
                    self.R_global[p] += 1.0
                    set_player_r(p, self.node_type, self.player, self.street, self.voff, self.K_s, self.R, self.R_global[p])

    # ------------------------------------------------------------------ evaluation (float64)
    def _eval_buffers(self):
        if self._eval is None:
            self._eval = dict(reach0=np.zeros(self.V), reach1=np.zeros(self.V), val=np.zeros(self.V),
                              alive0=np.ones(self.n, dtype=np.uint8), alive1=np.ones(self.n, dtype=np.uint8))
        return self._eval

    def average_strategy(self):
        out = np.zeros(self.I)
        normalize_strategy(self.node_type, self.n_actions, self.ioff, self.K_s, self.street, self.strat_sum, out)
        return out

    def _forward64(self, sigma):
        e = self._eval_buffers()
        tb = self.tables[np.dtype(np.float64)]
        forward(self.node_type, self.player, self.street, self.child, self.n_actions, self.voff, self.ioff,
                self.K_s, sigma, e["reach0"], e["reach1"], tb["p_class"], tb["T_pre"], tb["T_flop"], tb["T_turn"],
                e["alive0"], e["alive1"])
        return e, tb

    def exploitability(self, which="avg"):
        """Best-response values of both players vs the given strategy in units (1 bb = 2).
        Returns (exploitability, br0, br1), exploitability = (br0 + br1) / 2."""
        sigma = self.average_strategy() if which == "avg" else self.strat.astype(np.float64)
        e, tb = self._forward64(sigma)
        brs = []
        for p in (0, 1):
            reach_q = e["reach1"] if p == 0 else e["reach0"]
            terminal_batch(p, self.sd_nodes, self.sd_off, self.fold_nodes, self.street, self.K_s, self.voff,
                           self.term_amount, self.term_winner, reach_q, e["val"], tb["W0T"], tb["W_postT"])
            v = best_response(p, self.node_type, self.player, self.street, self.child, self.n_actions, self.voff,
                              self.K_s, e["val"], tb["T_pre"], tb["T_flop"], tb["T_turn"])
            brs.append(float(np.dot(tb["p_class"], v)))
        return (brs[0] + brs[1]) / 2, brs[0], brs[1]

    def game_value(self):
        """Expected value for player 0 (units) when both play the average strategy."""
        sigma = self.average_strategy()
        e, tb = self._forward64(sigma)
        terminal_batch(0, self.sd_nodes, self.sd_off, self.fold_nodes, self.street, self.K_s, self.voff,
                       self.term_amount, self.term_winner, e["reach1"], e["val"], tb["W0T"], tb["W_postT"])
        v = expected_value(0, self.node_type, self.player, self.street, self.child, self.n_actions, self.voff,
                           self.ioff, self.K_s, sigma, e["val"], tb["T_pre"], tb["T_flop"], tb["T_turn"])
        return float(np.dot(tb["p_class"], v))

    # ------------------------------------------------------------------ state / warm start
    def state_dict(self):
        return dict(theta_hat=self.theta_hat, theta_prev=self.theta_prev, F_prev=self.F_prev,
                    strat_sum=self.strat_sum, R=self.R, R_global=self.R_global, t=self.t,
                    keys=np.array(self.tree.key, dtype=str), tags=np.array([" ".join(t) for t in self.tree.tags], dtype=str),
                    n_actions=self.n_actions, ioff=self.ioff, voff=self.voff, node_type=self.node_type,
                    street=self.street, K_s=self.K_s, mode=self.mode)

    def load_state(self, st):
        """Restore a checkpoint of this same tree."""
        for name in ("theta_hat", "theta_prev", "F_prev", "strat_sum", "R"):
            getattr(self, name)[:] = st[name]
        self.R_global[:] = st["R_global"]
        self.t[:] = st["t"]

    def warm_start(self, st, copy_average=False, strategy=None, scale=None):
        """Initialise from the state of another tree (another stack or bet-size set).

        Nodes are matched by action path (`tree.key`) and actions by tag; unmatched nodes and
        actions keep their cold initialisation. `strategy` (a normalised strategy of the other
        tree) warm-starts from a strategy instead: theta = scale * sigma, which under RM+ /
        SPRM+ reproduces sigma on the first iteration (scale must be >= R for SPRM+).
        Returns the fraction of this tree's infoset entries that were initialised."""
        m = match_state(self.tree, self.node_type, self.street, self.n_actions, self.ioff, self.voff, self.K_s, st)
        if m is None:
            return 0.0
        dst_i, src_i, dst_v, src_v = m
        if strategy is not None:
            sc = float(scale if scale is not None else max(st["R_global"]))
            self.theta_hat[dst_i] = sc * strategy[src_i]
            self.theta_prev[dst_i] = self.theta_hat[dst_i]
            self.F_prev[dst_i] = 0.0
        else:
            for name in ("theta_hat", "theta_prev", "F_prev"):
                getattr(self, name)[dst_i] = st[name][src_i]
            if copy_average:
                self.strat_sum[dst_i] = st["strat_sum"][src_i]
                self.t[:] = st["t"]
        if "R" in st:
            self.R[dst_v] = st["R"][src_v]
        self.R_global[:] = st["R_global"]
        if self.ard_global:
            for p in (0, 1):
                set_player_r(p, self.node_type, self.player, self.street, self.voff, self.K_s, self.R, self.R_global[p])
        return len(dst_i) / self.I


def tree_layout(tree, K_s):
    """The layout part of a state dict (keys, tags, offsets) for `tree` without a solver."""
    arrs = tree.arrays()
    n = len(arrs["node_type"])
    voff = np.zeros(n + 1, dtype=np.int64)
    ioff = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        k = K_s[arrs["street"][i]]
        voff[i + 1] = voff[i] + k
        ioff[i + 1] = ioff[i] + (k * arrs["n_actions"][i] if arrs["node_type"][i] == DECISION else 0)
    return dict(keys=np.array(tree.key, dtype=str), tags=np.array([" ".join(t) for t in tree.tags], dtype=str),
                n_actions=arrs["n_actions"], ioff=ioff, voff=voff, node_type=arrs["node_type"],
                street=arrs["street"], K_s=np.asarray(K_s, dtype=np.int64))


def match_state(tree, node_type, street, n_actions, ioff, voff, K_s, st):
    """Index arrays (dst_i, src_i, dst_v, src_v) pairing this tree's infoset entries and
    (node, bucket) slots with those of the checkpoint `st` (another tree), matched by action
    path and action tag. None when nothing matches."""
    src_key = {k: j for j, k in enumerate(st["keys"])}
    src_tags = [t.split(" ") if t else [] for t in st["tags"]]
    src_ioff, src_voff, src_na, src_K = st["ioff"], st["voff"], st["n_actions"], st["K_s"]
    src_nt, src_st = st["node_type"], st["street"]
    dst_i, src_i, dst_v, src_v = [], [], [], []
    for i in range(len(node_type)):
        if node_type[i] != DECISION:
            continue
        j = src_key.get(tree.key[i])
        if j is None or src_nt[j] != DECISION:
            continue
        K = int(K_s[street[i]])
        if int(src_K[src_st[j]]) != K:
            continue
        A, Ao = int(n_actions[i]), int(src_na[j])
        tag_pos = {tg: a for a, tg in enumerate(src_tags[j])}
        b = np.arange(K)
        for a, tg in enumerate(tree.tags[i]):
            ao = tag_pos.get(tg)
            if ao is None:
                continue
            dst_i.append(ioff[i] + b * A + a)
            src_i.append(src_ioff[j] + b * Ao + ao)
        dst_v.append(voff[i] + b)
        src_v.append(src_voff[j] + b)
    if not dst_i:
        return None
    return np.concatenate(dst_i), np.concatenate(src_i), np.concatenate(dst_v), np.concatenate(src_v)


# ====================================================================== kernels
@njit(cache=False)
def project_r(vec, A, R, out):
    s = 0.0
    for a in range(A):
        out[a] = vec[a] if vec[a] > 0.0 else 0.0
        s += out[a]
    if s >= R:
        return
    # projection onto {x >= 0, sum x = R}: sort descending (A is tiny)
    srt = np.empty(A)
    for a in range(A):
        srt[a] = vec[a]
    for i in range(1, A):
        key = srt[i]
        j = i - 1
        while j >= 0 and srt[j] < key:
            srt[j + 1] = srt[j]
            j -= 1
        srt[j + 1] = key
    css = 0.0
    tau = 0.0
    for k in range(A):
        css += srt[k]
        cand = (css - R) / (k + 1)
        if srt[k] - cand > 0.0:
            tau = cand
    for a in range(A):
        v = vec[a] - tau
        out[a] = v if v > 0.0 else 0.0


@njit(cache=False)
def init_theta(node_type, n_actions, ioff, K_s, street, theta_hat, theta_prev, strat, mode):
    n = len(node_type)
    for i in range(n):
        if node_type[i] != DECISION:
            continue
        A = n_actions[i]
        K = K_s[street[i]]
        base = ioff[i]
        for b in range(K):
            for a in range(A):
                idx = base + b * A + a
                strat[idx] = 1.0 / A
                if mode >= 1:
                    theta_hat[idx] = 1.0 / A
                    theta_prev[idx] = 1.0 / A


@njit(cache=False)
def strategy_step(p, mode, node_type, player, n_actions, ioff, voff, K_s, street, theta_hat, theta_prev, F_prev,
                  strat, R, Bsum):
    n = len(node_type)
    tmp = np.empty(16)
    th = np.empty(16)
    for i in range(n):
        if node_type[i] != DECISION or player[i] != p:
            continue
        A = n_actions[i]
        K = K_s[street[i]]
        base = ioff[i]
        vb = voff[i]
        for b in range(K):
            idx0 = base + b * A
            if mode == RM_PLUS:
                s = 0.0
                for a in range(A):
                    s += theta_hat[idx0 + a]
                for a in range(A):
                    strat[idx0 + a] = theta_hat[idx0 + a] / s if s > 0.0 else 1.0 / A
                continue
            for a in range(A):
                tmp[a] = theta_hat[idx0 + a] + F_prev[idx0 + a]
            if mode == PRM_PLUS:
                for a in range(A):
                    th[a] = tmp[a] if tmp[a] > 0.0 else 0.0
            else:
                project_r(tmp, A, R[vb + b], th)
            s = 0.0
            for a in range(A):
                s += th[a]
            b1 = 0.0
            b2 = 0.0
            for a in range(A):
                strat[idx0 + a] = th[a] / s if s > 0.0 else 1.0 / A
                d1 = theta_hat[idx0 + a] - theta_prev[idx0 + a]
                d2 = theta_hat[idx0 + a] - th[a]
                b1 += d1 * d1
                b2 += d2 * d2
                theta_prev[idx0 + a] = th[a]
            Bsum[vb + b] = 0.5 * b1 + 0.5 * b2


@njit(cache=False)
def chance_forward(street_from, src, dst, T_pre, T_flop, T_turn, Kf, Kt):
    # dst[b'] = sum_b src[b] T[b, b']
    for j in range(Kt):
        dst[j] = 0.0
    if street_from == 0:
        for b in range(Kf):
            s = src[b]
            if s != 0.0:
                for j in range(Kt):
                    dst[j] += s * T_pre[b, j]
    elif street_from == 1:
        for b in range(Kf):
            s = src[b]
            if s != 0.0:
                for j in range(Kt):
                    dst[j] += s * T_flop[b, j]
    else:
        for b in range(Kf):
            s = src[b]
            if s != 0.0:
                for j in range(Kt):
                    dst[j] += s * T_turn[b, j]


@njit(cache=False)
def chance_backward(street_from, vchild, vout, T_pre, T_flop, T_turn, Kf, Kt):
    # vout[b] = sum_b' T[b, b'] vchild[b']
    for b in range(Kf):
        s = 0.0
        if street_from == 0:
            for j in range(Kt):
                s += T_pre[b, j] * vchild[j]
        elif street_from == 1:
            for j in range(Kt):
                s += T_flop[b, j] * vchild[j]
        else:
            for j in range(Kt):
                s += T_turn[b, j] * vchild[j]
        vout[b] = s


@njit(cache=False)
def forward(node_type, player, street, child, n_actions, voff, ioff, K_s, strat, reach0, reach1, p_class,
            T_pre, T_flop, T_turn, alive0, alive1):
    """Reach probabilities of both players at every node; alive_p[i] = any reach_p[i, b] > 0."""
    n = len(node_type)
    K0 = K_s[0]
    for b in range(K0):
        reach0[b] = p_class[b]
        reach1[b] = p_class[b]
    for i in range(n):
        nt = node_type[i]
        s = street[i]
        K = K_s[s]
        vb = voff[i]
        a0 = 0
        a1 = 0
        for b in range(K):
            if reach0[vb + b] > 0.0:
                a0 = 1
            if reach1[vb + b] > 0.0:
                a1 = 1
        alive0[i] = a0
        alive1[i] = a1
        if nt == DECISION:
            A = n_actions[i]
            p = player[i]
            base = ioff[i]
            for a in range(A):
                c = child[i, a]
                vc = voff[c]
                if p == 0:
                    for b in range(K):
                        reach0[vc + b] = reach0[vb + b] * strat[base + b * A + a]
                        reach1[vc + b] = reach1[vb + b]
                else:
                    for b in range(K):
                        reach1[vc + b] = reach1[vb + b] * strat[base + b * A + a]
                        reach0[vc + b] = reach0[vb + b]
        elif nt == CHANCE:
            c = child[i, 0]
            vc = voff[c]
            Kt = K_s[s + 1]
            chance_forward(s, reach0[vb:vb + K], reach0[vc:vc + Kt], T_pre, T_flop, T_turn, K, Kt)
            chance_forward(s, reach1[vb:vb + K], reach1[vc:vc + Kt], T_pre, T_flop, T_turn, K, Kt)


@njit(cache=False)
def terminal_batch(p, sd_nodes, sd_off, fold_nodes, street, K_s, voff, term_amount, term_winner, reach_q, val,
                   W0T, W_postT):
    """Values of all terminals for player p: one matrix product per street for showdowns."""
    for j in range(len(fold_nodes)):
        i = fold_nodes[j]
        K = K_s[street[i]]
        vb = voff[i]
        tot = 0.0
        for b in range(K):
            tot += reach_q[vb + b]
        amt = term_amount[i]
        v = amt * tot if term_winner[i] == p else -amt * tot
        for b in range(K):
            val[vb + b] = v
    for s in range(4):
        lo = sd_off[s]
        hi = sd_off[s + 1]
        m = hi - lo
        if m == 0:
            continue
        K = K_s[s]
        M = np.empty((m, K), dtype=reach_q.dtype)
        for j in range(m):
            vb = voff[sd_nodes[lo + j]]
            for b in range(K):
                M[j, b] = reach_q[vb + b]
        if s == 0:
            P = np.dot(M, W0T)  # P[j, b] = sum_b' W0[b, b'] reach_q[j, b']
        else:
            P = np.dot(M, W_postT[s - 1])
        for j in range(m):
            i = sd_nodes[lo + j]
            vb = voff[i]
            amt = term_amount[i]
            for b in range(K):
                val[vb + b] = amt * P[j, b]


@njit(cache=False)
def backward(p, mode, w, use_r_weight, qre_tau, ard_global, ard_stat, prune, alive_q,
             node_type, player, street, child, n_actions,
             voff, ioff, K_s, strat, reach0, reach1, val, theta_hat, F_prev, strat_sum, R, Bsum,
             T_pre, T_flop, T_turn):
    """Regret and average-strategy update for player p. Terminal values must already be in `val`
    (see `terminal_batch`)."""
    n = len(node_type)
    reach_q = reach1 if p == 0 else reach0
    reach_p = reach0 if p == 0 else reach1
    u = np.empty(16)
    F = np.empty(16)
    tmp = np.empty(16)
    out = np.empty(16)
    for i in range(n - 1, -1, -1):
        nt = node_type[i]
        if nt == FOLD or nt == SHOWDOWN:
            continue
        s = street[i]
        K = K_s[s]
        vb = voff[i]
        if prune and alive_q[i] == 0:
            # opponent never reaches this node: every counterfactual value below is exactly
            # zero, so skip the child gathers and apply the F = 0 update in closed form
            for b in range(K):
                val[vb + b] = 0.0
            if nt == DECISION and player[i] == p:
                A = n_actions[i]
                base = ioff[i]
                for b in range(K):
                    idx0 = base + b * A
                    wt = w * (R[vb + b] if use_r_weight else 1.0) * reach_p[vb + b]
                    for a in range(A):
                        strat_sum[idx0 + a] += wt * strat[idx0 + a]
                    if mode >= SPRM_PLUS:
                        for a in range(A):
                            tmp[a] = theta_hat[idx0 + a]
                        project_r(tmp, A, R[vb + b], out)  # identity unless R grew
                        for a in range(A):
                            theta_hat[idx0 + a] = out[a]
                        if mode == MI_SPRM_PLUS:
                            diff = 0.0
                            for a in range(A):
                                diff += F_prev[idx0 + a] * F_prev[idx0 + a]
                            if ard_global:
                                ard_stat[0] += diff - 0.5 * Bsum[vb + b]
                            elif diff - 0.5 * Bsum[vb + b] > 0.0:
                                R[vb + b] += 1.0
                    for a in range(A):
                        F_prev[idx0 + a] = 0.0
            continue
        if nt == CHANCE:
            c = child[i, 0]
            vc = voff[c]
            Kt = K_s[s + 1]
            chance_backward(s, val[vc:vc + Kt], val[vb:vb + K], T_pre, T_flop, T_turn, K, Kt)
        else:
            A = n_actions[i]
            base = ioff[i]
            if player[i] != p:
                for b in range(K):
                    acc = 0.0
                    for a in range(A):
                        acc += val[voff[child[i, a]] + b]
                    val[vb + b] = acc
                continue
            for b in range(K):
                idx0 = base + b * A
                node_v = 0.0
                for a in range(A):
                    u[a] = val[voff[child[i, a]] + b]
                    node_v += strat[idx0 + a] * u[a]
                true_v = node_v
                if qre_tau > 0.0:
                    mass = 0.0
                    for j in range(K):
                        mass += reach_q[vb + j]
                    node_v = 0.0
                    for a in range(A):
                        sg = strat[idx0 + a]
                        if sg < 1e-12:
                            sg = 1e-12
                        u[a] -= qre_tau * mass * (np.log(sg) + 1.0)
                        node_v += strat[idx0 + a] * u[a]
                for a in range(A):
                    F[a] = u[a] - node_v
                # average strategy
                wt = w * (R[vb + b] if use_r_weight else 1.0) * reach_p[vb + b]
                for a in range(A):
                    strat_sum[idx0 + a] += wt * strat[idx0 + a]
                # regret update
                if mode == RM_PLUS or mode == PRM_PLUS:
                    for a in range(A):
                        v = theta_hat[idx0 + a] + F[a]
                        theta_hat[idx0 + a] = v if v > 0.0 else 0.0
                else:
                    for a in range(A):
                        tmp[a] = theta_hat[idx0 + a] + F[a]
                    project_r(tmp, A, R[vb + b], out)
                    for a in range(A):
                        theta_hat[idx0 + a] = out[a]
                    if mode == MI_SPRM_PLUS:
                        diff = 0.0
                        for a in range(A):
                            d = F[a] - F_prev[idx0 + a]
                            diff += d * d
                        if ard_global:
                            ard_stat[0] += diff - 0.5 * Bsum[vb + b]
                        elif diff - 0.5 * Bsum[vb + b] > 0.0:
                            R[vb + b] += 1.0
                for a in range(A):
                    F_prev[idx0 + a] = F[a]
                val[vb + b] = true_v


@njit(cache=False)
def set_player_r(p, node_type, player, street, voff, K_s, R, value):
    n = len(node_type)
    for i in range(n):
        if node_type[i] == DECISION and player[i] == p:
            K = K_s[street[i]]
            vb = voff[i]
            for b in range(K):
                R[vb + b] = value


@njit(cache=False)
def best_response(p, node_type, player, street, child, n_actions, voff, K_s, val, T_pre, T_flop, T_turn):
    """Best-response value vector of p at the root (terminal values must be in `val`)."""
    n = len(node_type)
    for i in range(n - 1, -1, -1):
        nt = node_type[i]
        if nt == FOLD or nt == SHOWDOWN:
            continue
        s = street[i]
        K = K_s[s]
        vb = voff[i]
        if nt == CHANCE:
            c = child[i, 0]
            vc = voff[c]
            Kt = K_s[s + 1]
            chance_backward(s, val[vc:vc + Kt], val[vb:vb + K], T_pre, T_flop, T_turn, K, Kt)
        else:
            A = n_actions[i]
            if player[i] != p:
                for b in range(K):
                    acc = 0.0
                    for a in range(A):
                        acc += val[voff[child[i, a]] + b]
                    val[vb + b] = acc
            else:
                for b in range(K):
                    best = -1e300
                    for a in range(A):
                        v = val[voff[child[i, a]] + b]
                        if v > best:
                            best = v
                    val[vb + b] = best
    return val[0:K_s[0]].copy()


@njit(cache=False)
def expected_value(p, node_type, player, street, child, n_actions, voff, ioff, K_s, strat, val,
                   T_pre, T_flop, T_turn):
    n = len(node_type)
    for i in range(n - 1, -1, -1):
        nt = node_type[i]
        if nt == FOLD or nt == SHOWDOWN:
            continue
        s = street[i]
        K = K_s[s]
        vb = voff[i]
        if nt == CHANCE:
            c = child[i, 0]
            vc = voff[c]
            Kt = K_s[s + 1]
            chance_backward(s, val[vc:vc + Kt], val[vb:vb + K], T_pre, T_flop, T_turn, K, Kt)
        else:
            A = n_actions[i]
            base = ioff[i]
            if player[i] != p:
                for b in range(K):
                    acc = 0.0
                    for a in range(A):
                        acc += val[voff[child[i, a]] + b]
                    val[vb + b] = acc
            else:
                for b in range(K):
                    acc = 0.0
                    for a in range(A):
                        acc += strat[base + b * A + a] * val[voff[child[i, a]] + b]
                    val[vb + b] = acc
    return val[0:K_s[0]].copy()


@njit(cache=False)
def normalize_strategy(node_type, n_actions, ioff, K_s, street, strat_sum, out):
    n = len(node_type)
    for i in range(n):
        if node_type[i] != DECISION:
            continue
        A = n_actions[i]
        K = K_s[street[i]]
        base = ioff[i]
        for b in range(K):
            s = 0.0
            for a in range(A):
                s += strat_sum[base + b * A + a]
            for a in range(A):
                out[base + b * A + a] = strat_sum[base + b * A + a] / s if s > 0.0 else 1.0 / A
