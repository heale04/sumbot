"""N-player preflop betting tree (equal stacks) for the multiway blueprint.

Positions: 0 = small blind, 1 = big blind, 2.. = UTG onward, N-1 = button. Preflop action
starts at position 2 (or at the small blind heads-up). Betting after the preflop round is not
modelled: a terminal with two or more live players is settled by multiway equity.
Units: 1 big blind = 2 units.
"""

from dataclasses import dataclass, field

import numpy as np

DECISION, FOLD, SHOWDOWN = 0, 2, 3


@dataclass
class MultiConfig:
    players: int = 8
    stack_bb: float = 30.0
    sizes: tuple = (0.5, 1.0)  # pot fractions for non-all-in raises
    raise_cap: int = 2  # non-all-in raises per hand

    @property
    def stack(self):
        return int(round(self.stack_bb * 2))


@dataclass
class MultiTree:
    config: MultiConfig
    node_type: list = field(default_factory=list)
    player: list = field(default_factory=list)
    children: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    term_pot: list = field(default_factory=list)
    term_contrib: list = field(default_factory=list)  # per player
    term_live: list = field(default_factory=list)  # bitmask
    term_aggressor: list = field(default_factory=list)  # last raiser or -1
    label: list = field(default_factory=list)

    def add(self, ntype, player, label):
        self.node_type.append(ntype)
        self.player.append(player)
        self.children.append([])
        self.actions.append([])
        self.term_pot.append(0)
        self.term_contrib.append(None)
        self.term_live.append(0)
        self.term_aggressor.append(-1)
        self.label.append(label)
        return len(self.node_type) - 1

    @property
    def n_nodes(self):
        return len(self.node_type)

    def arrays(self):
        n = self.n_nodes
        N = self.config.players
        max_a = max(len(a) for a in self.actions)
        child = np.full((n, max_a), -1, dtype=np.int64)
        n_actions = np.zeros(n, dtype=np.int64)
        kind = np.full((n, max_a), -1, dtype=np.int8)
        amount = np.zeros((n, max_a), dtype=np.int32)
        contrib = np.zeros((n, N), dtype=np.float64)
        for i in range(n):
            n_actions[i] = len(self.children[i])
            for a, ((k, amt), c) in enumerate(zip(self.actions[i], self.children[i])):
                child[i, a] = c
                kind[i, a] = {"fold": 0, "check": 1, "call": 1, "raise": 2}[k]
                amount[i, a] = amt
            if self.term_contrib[i] is not None:
                contrib[i] = self.term_contrib[i]
        return {
            "node_type": np.array(self.node_type, dtype=np.int64),
            "player": np.array(self.player, dtype=np.int64),
            "child": child,
            "n_actions": n_actions,
            "kind": kind,
            "amount": amount,
            "term_pot": np.array(self.term_pot, dtype=np.float64),
            "term_contrib": contrib,
            "term_live": np.array(self.term_live, dtype=np.int64),
            "term_aggressor": np.array(self.term_aggressor, dtype=np.int64),
        }


def build_multi_tree(cfg: MultiConfig) -> MultiTree:
    N = cfg.players
    S = cfg.stack
    tree = MultiTree(cfg)

    def next_actor(start, folded, allin):
        for k in range(1, N + 1):
            q = (start + k) % N
            if not folded[q] and not allin[q]:
                return q
        return -1

    def terminal(c, folded, label, aggressor):
        live = [q for q in range(N) if not folded[q]]
        pot = sum(c)
        if len(live) == 1:
            t = tree.add(FOLD, -1, label)
        else:
            t = tree.add(SHOWDOWN, -1, label)
        tree.term_pot[t] = pot
        tree.term_contrib[t] = list(c)
        tree.term_live[t] = sum(1 << q for q in live)
        tree.term_aggressor[t] = aggressor
        return t

    def recurse(c, folded, allin, to_act, need_act, n_raises, last_inc, label, aggressor=-1):
        """need_act: set of players who still must act before the round can end."""
        bet_max = max(c)
        pot = sum(c)
        to_call = bet_max - c[to_act]
        node = tree.add(DECISION, to_act, label)
        acts = []

        def after(c2, folded2, allin2, need2, n_raises2, last_inc2, lab, agg):
            live = [q for q in range(N) if not folded2[q]]
            if len(live) == 1:
                return terminal(c2, folded2, lab, agg)
            nxt = next_actor(to_act, folded2, allin2)
            pending = [q for q in need2 if not folded2[q] and not allin2[q]]
            if nxt < 0 or not pending:
                return terminal(c2, folded2, lab, agg)
            # next actor must be one who still needs to act
            k = 1
            while True:
                q = (to_act + k) % N
                if not folded2[q] and not allin2[q] and q in need2:
                    nxt = q
                    break
                k += 1
                if k > N:
                    return terminal(c2, folded2, lab, agg)
            return recurse(c2, folded2, allin2, nxt, need2, n_raises2, last_inc2, lab, agg)

        # fold
        if to_call > 0:
            f2 = list(folded)
            f2[to_act] = True
            need2 = need_act - {to_act}
            acts.append((("fold", 0), after(c, f2, allin, need2, n_raises, last_inc, label + "f", aggressor)))
        # call / check
        c2 = list(c)
        c2[to_act] += to_call
        a2 = list(allin)
        if c2[to_act] >= S:
            a2[to_act] = True
        need2 = need_act - {to_act}
        kind = "call" if to_call > 0 else "check"
        acts.append(((kind, 0), after(c2, folded, a2, need2, n_raises, last_inc, label + ("c" if to_call else "k"), aggressor)))
        # raises
        remaining = S - c[to_act]
        if remaining > to_call:
            targets = []
            if n_raises < cfg.raise_cap:
                for f in cfg.sizes:
                    raise_to = bet_max + int(round(f * (pot + to_call)))
                    inc = raise_to - bet_max
                    min_inc = max(last_inc, 2)
                    if inc < min_inc:
                        raise_to = bet_max + min_inc
                    if raise_to >= S:
                        continue
                    targets.append(raise_to)
            targets.append(S)
            seen = set()
            for raise_to in targets:
                if raise_to in seen:
                    continue
                seen.add(raise_to)
                inc = raise_to - bet_max
                c3 = list(c)
                c3[to_act] = raise_to
                a3 = list(allin)
                if raise_to >= S:
                    a3[to_act] = True
                need3 = {q for q in range(N) if not folded[q] and not a3[q] and q != to_act}
                lab = label + ("A" if raise_to >= S else f"r{raise_to}")
                acts.append((("raise", raise_to), after(c3, folded, a3, need3, n_raises + 1, max(inc, last_inc), lab, to_act)))
        tree.actions[node] = [a for a, _ in acts]
        tree.children[node] = [k for _, k in acts]
        return node

    c = [0] * N
    c[0] = min(1, S)
    c[1] = min(2, S)
    allin = [c[q] >= S for q in range(N)]
    folded = [False] * N
    first = 2 % N if N > 2 else 0
    need = {q for q in range(N) if not allin[q]}
    recurse(c, folded, allin, first, need, 0, 2, "")
    return tree


if __name__ == "__main__":
    import sys, time
    for N in (3, 4, 6, 8):
        for cap in (1, 2):
            for sizes in ((0.5,), (0.5, 1.0)):
                t0 = time.time()
                tree = build_multi_tree(MultiConfig(players=N, stack_bb=30, sizes=sizes, raise_cap=cap))
                nt = np.array(tree.node_type)
                print(f"N={N} cap={cap} sizes={sizes}: nodes={tree.n_nodes:,} decision={(nt==0).sum():,} showdown={(nt==3).sum():,} fold={(nt==2).sum():,} ({time.time()-t0:.1f}s)", flush=True)
