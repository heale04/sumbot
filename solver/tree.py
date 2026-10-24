"""Public betting tree for heads-up no-limit play with a fixed bet-size abstraction.

Units: 1 big blind = 2 units (so the small blind is 1 unit). Player 0 is the small blind
(button) and acts first preflop; player 1 is the big blind and acts first postflop.
Nodes: 0 decision, 1 chance (next street), 2 fold terminal, 3 showdown terminal.
"""

from dataclasses import dataclass, field

import numpy as np

DECISION, CHANCE, FOLD, SHOWDOWN = 0, 1, 2, 3
STREET_NAMES = ["preflop", "flop", "turn", "river"]


@dataclass
class TreeConfig:
    stack_bb: float = 100.0
    sizes: tuple = ((0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.5, 1.0))  # pot fractions per street
    raise_cap: int = 3  # non-all-in raises per street
    allin: bool = True

    @property
    def stack(self) -> int:
        return int(round(self.stack_bb * 2))


@dataclass
class Tree:
    config: TreeConfig
    node_type: list = field(default_factory=list)
    player: list = field(default_factory=list)
    street: list = field(default_factory=list)
    children: list = field(default_factory=list)  # list of child ids per node
    actions: list = field(default_factory=list)  # list of (kind, raise_to_units) per node
    term_amount: list = field(default_factory=list)  # fold: chips lost by folder; showdown: matched
    term_winner: list = field(default_factory=list)  # fold terminals: winner player
    pot: list = field(default_factory=list)  # c0 + c1 at the node
    label: list = field(default_factory=list)  # action string for debugging
    key: list = field(default_factory=list)  # stack-independent action path (for warm starts)
    tags: list = field(default_factory=list)  # per-action tags: f, c, r<fraction>, A

    def add(self, ntype, player, street, pot, label, key=""):
        self.node_type.append(ntype)
        self.player.append(player)
        self.street.append(street)
        self.children.append([])
        self.actions.append([])
        self.tags.append([])
        self.term_amount.append(0)
        self.term_winner.append(-1)
        self.pot.append(pot)
        self.label.append(label)
        self.key.append(key)
        return len(self.node_type) - 1

    @property
    def n_nodes(self):
        return len(self.node_type)

    def arrays(self):
        n = self.n_nodes
        max_a = max((len(a) for a in self.actions), default=1)
        child = np.full((n, max_a), -1, dtype=np.int64)
        n_actions = np.zeros(n, dtype=np.int64)
        for i, kids in enumerate(self.children):
            n_actions[i] = len(kids)
            child[i, : len(kids)] = kids
        return {
            "node_type": np.array(self.node_type, dtype=np.int64),
            "player": np.array(self.player, dtype=np.int64),
            "street": np.array(self.street, dtype=np.int64),
            "child": child,
            "n_actions": n_actions,
            "term_amount": np.array(self.term_amount, dtype=np.float64),
            "term_winner": np.array(self.term_winner, dtype=np.int64),
        }


def build_tree(cfg: TreeConfig) -> Tree:
    tree = Tree(cfg)
    S = cfg.stack

    def recurse(street, c, bet, to_act, n_raises, last_inc, acted, label, key):
        """c: total committed per player; bet: committed this street; returns node id."""
        pot = c[0] + c[1]
        to_call = bet[1 - to_act] - bet[to_act]
        node = tree.add(DECISION, to_act, street, pot, label, key)
        acts = []
        # fold
        if to_call > 0:
            t = tree.add(FOLD, -1, street, pot, label + "f", key + "f")
            tree.term_amount[t] = c[to_act]
            tree.term_winner[t] = 1 - to_act
            acts.append((("fold", 0), "f", t))
        # check / call
        c2 = list(c)
        b2 = list(bet)
        c2[to_act] += to_call
        b2[to_act] += to_call
        acted2 = acted | (1 << to_act)
        kind = "call" if to_call > 0 else "check"
        both_acted = acted2 == 3
        allin_now = c2[0] >= S or c2[1] >= S
        if both_acted:
            nxt = street_end(street, c2, allin_now, label + ("c" if to_call > 0 else "k"), key + "c")
        else:
            nxt = recurse(street, c2, b2, 1 - to_act, n_raises, last_inc, acted2,
                          label + ("c" if to_call > 0 else "k"), key + "c")
        acts.append(((kind, 0), "c", nxt))
        # raises
        remaining = S - c[to_act]
        if remaining > to_call:
            bet_max = max(bet)
            pot_after_call = pot + to_call
            targets = []
            if n_raises < cfg.raise_cap:
                for f in cfg.sizes[street]:
                    raise_to = bet_max + int(round(f * pot_after_call))
                    inc = raise_to - bet_max
                    min_inc = max(last_inc, 2)
                    if inc < min_inc:
                        raise_to = bet_max + min_inc
                    cap = bet[to_act] + remaining  # all-in total for this street
                    if raise_to >= cap:
                        continue  # becomes all-in below
                    targets.append((raise_to, f"r{f:g}"))
            if cfg.allin:
                targets.append((bet[to_act] + remaining, "A"))
            seen = set()
            for raise_to, tag in targets:
                if raise_to in seen:
                    continue
                seen.add(raise_to)
                inc = raise_to - bet_max
                c3 = list(c)
                b3 = list(bet)
                c3[to_act] += raise_to - bet[to_act]
                b3[to_act] = raise_to
                is_allin = c3[to_act] >= S
                nxt = recurse(
                    street, c3, b3, 1 - to_act, n_raises + 1, max(inc, last_inc), 1 << to_act,
                    label + ("A" if is_allin else f"r{raise_to}"), key + tag + ".",
                )
                acts.append((("raise", raise_to), tag, nxt))
        tree.actions[node] = [a for a, _, _ in acts]
        tree.tags[node] = [t for _, t, _ in acts]
        tree.children[node] = [k for _, _, k in acts]
        return node

    def street_end(street, c, allin_now, label, key):
        pot = c[0] + c[1]
        if allin_now or street == 3:
            t = tree.add(SHOWDOWN, -1, street, pot, label + "|sd", key + "|sd")
            tree.term_amount[t] = min(c[0], c[1])
            return t
        ch = tree.add(CHANCE, -1, street, pot, label + "|", key + "|ch")
        nxt = recurse(street + 1, c, [0, 0], 1, 0, 0, 0, label + "|", key + "|")
        tree.children[ch] = [nxt]
        return ch

    # preflop: SB posts 1, BB posts 2; SB (player 0) to act
    recurse(0, [min(1, S), min(2, S)], [min(1, S), min(2, S)], 0, 0, 2, 0, "", "")
    return tree


def summarize(tree: Tree) -> str:
    nt = np.array(tree.node_type)
    st = np.array(tree.street)
    parts = [f"nodes={tree.n_nodes}"]
    for s in range(4):
        parts.append(f"{STREET_NAMES[s]}: dec={int(((nt == DECISION) & (st == s)).sum())}")
    parts.append(f"fold={int((nt == FOLD).sum())} showdown={int((nt == SHOWDOWN).sum())}")
    return " ".join(parts)
