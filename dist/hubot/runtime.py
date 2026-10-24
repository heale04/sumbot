"""Map a GameState onto the blueprint and pick an action. Everything here must stay well
inside 1 ms per call; the heavy parts are one Numba equity estimate and a short tree walk."""

import math
import random

import numpy as np

from hubot.cards import card_from_str
from hubot.equity import equity

DECISION, CHANCE, FOLD, SHOWDOWN = 0, 1, 2, 3
SAMPLES = (0, 400, 400, 600)  # equity samples per street (preflop uses exact classes)


class Blueprint:
    def __init__(self, path):
        z = np.load(path)
        self.stacks = z["stacks"].astype(np.float64)
        self.node_off = z["node_off"]
        self.node_type = z["node_type"]
        self.player = z["player"]
        self.street = z["street"]
        self.pot = z["pot"]
        self.child = z["child"]
        self.kind = z["kind"]
        self.amount = z["amount"]
        self.n_actions = z["n_actions"]
        self.ioff = z["ioff"]
        self.strategy = z["strategy"]
        self.K = int(z["K"])
        self.edges = z["edges"]
        self.class_of = z["class_of"]
        self.n_classes = int(self.class_of.max()) + 1
        self.K_s = (self.n_classes, self.K, self.K, self.K)
        # postflop entry points per stack: (chance node from preflop, pot units)
        self.entries = []
        for i in range(len(self.stacks)):
            lo, hi = self.node_off[i], self.node_off[i + 1]
            idx = [n for n in range(lo, hi) if self.node_type[n] == CHANCE and self.street[n] == 0]
            self.entries.append([(n, int(self.pot[n])) for n in idx])
        self.entry_node = np.array([n for e in self.entries for n, _ in e], dtype=np.int64)
        self.entry_pot = np.array([float(p) for e in self.entries for _, p in e])
        self.entry_rem = np.array([max(self.stacks[i] * 2 - p / 2, 1.0) for i, e in enumerate(self.entries) for _, p in e])

    def root(self, stack_idx):
        return int(self.node_off[stack_idx])

    def stack_index(self, eff_bb):
        d = np.abs(np.log(self.stacks) - math.log(max(eff_bb, 1.0)))
        return int(np.argmin(d))

    def probs(self, node, bucket):
        a = int(self.n_actions[node])
        base = int(self.ioff[node]) + bucket * a
        p = self.strategy[base:base + a].astype(np.float64)
        s = p.sum()
        if s <= 0:
            return np.full(a, 1.0 / a)
        return p / s


class HandInfo:
    """Parsed view of the current hand from state.history."""

    __slots__ = ("sb_seat", "bb_seat", "dealt", "committed", "street_bet", "street", "actions",
                 "unit", "bb", "folded", "allin_seats")

    def __init__(self, state):
        self.bb = float(state.blinds.big_blind)
        self.unit = self.bb / 2.0
        self.dealt = [s.seat for s in state.seats if s.status != "busted"]
        self.committed = {s: 0.0 for s in self.dealt}
        self.street_bet = {s: 0.0 for s in self.dealt}
        self.folded = set()
        self.allin_seats = set()
        self.sb_seat = None
        self.bb_seat = None
        self.street = 0
        self.actions = [[], [], [], []]  # per street: (seat, kind, amount_units)
        blinds = []
        for e in state.history:
            k = e.kind
            if k == "blind":
                blinds.append((e.amount, e.seat))
                self.street_bet[e.seat] = float(e.amount)
            elif k == "ante":
                self.committed[e.seat] += float(e.amount)
            elif k == "board":
                for s in self.dealt:
                    self.committed[s] += self.street_bet[s]
                    self.street_bet[s] = 0.0
                self.street += 1
            elif k == "fold":
                self.folded.add(e.seat)
                self.actions[self.street].append((e.seat, 0, 0.0))
            elif k == "check":
                self.actions[self.street].append((e.seat, 1, 0.0))
            elif k == "call":
                self.street_bet[e.seat] += float(e.amount)
                self.actions[self.street].append((e.seat, 1, 0.0))
            elif k == "raise":
                self.street_bet[e.seat] = float(e.amount)
                self.actions[self.street].append((e.seat, 2, float(e.amount) / self.unit))
        if blinds:
            blinds.sort()
            self.sb_seat = blinds[0][1]
            self.bb_seat = blinds[-1][1]
        for s in state.seats:
            if s.status == "all_in":
                self.allin_seats.add(s.seat)
        self.street = min(self.street, 3)

    def start_stack(self, state, seat):
        st = next(s for s in state.seats if s.seat == seat)
        return st.stack + self.committed[seat] + self.street_bet[seat]


def walk(bp, node, actions, role_of_seat, start_units):
    """Advance `node` through the actions of one street. Returns node or -1 if off-tree."""
    for seat, kind, amt in actions:
        if bp.node_type[node] != DECISION:
            return -1
        role = role_of_seat.get(seat)
        if role is None or bp.player[node] != role:
            return -1
        a = int(bp.n_actions[node])
        kinds = bp.kind[node, :a]
        chosen = -1
        if kind == 2:
            best = None
            raise_idx = [i for i in range(a) if kinds[i] == 2]
            if not raise_idx:
                return -1
            top = max(raise_idx, key=lambda i: bp.amount[node, i])
            if amt >= start_units.get(seat, 1e18) - 1e-9:
                chosen = top
            else:
                for i in raise_idx:
                    d = abs(math.log(max(bp.amount[node, i], 1)) - math.log(max(amt, 1)))
                    if best is None or d < best:
                        best, chosen = d, i
        else:
            for i in range(a):
                if kinds[i] == kind:
                    chosen = i
                    break
            if chosen < 0:
                return -1
        node = int(bp.child[node, chosen])
    return node


def bucket_for(bp, street, hole, board):
    if street == 0:
        a, b, c = sorted(hole)
        return int(bp.class_of[a, b, c])
    e = equity(hole, board, SAMPLES[street])
    return int(np.searchsorted(bp.edges[street - 1], e, side="right"))


def choose(bp, node, bucket, rng, temperature=1.0):
    p = bp.probs(node, bucket)
    if temperature != 1.0:
        p = p ** (1.0 / temperature)
        p /= p.sum()
    r = rng.random()
    acc = 0.0
    for i, q in enumerate(p):
        acc += q
        if r < acc:
            return i
    return len(p) - 1


def translate(state, bp, node, idx, info):
    kind = int(bp.kind[node, idx])
    if kind == 0:
        return "fold", 0
    if kind == 1:
        return "call", 0
    a = int(bp.n_actions[node])
    top = max(range(a), key=lambda i: bp.amount[node, i] if bp.kind[node, i] == 2 else -1)
    if idx == top:
        return "raise", float("inf")  # the abstract all-in
    return "raise", float(bp.amount[node, idx]) * info.unit


class MultiBlueprint:
    """Preflop strategies for 3+ players: one tree per (players, stack)."""

    def __init__(self, path):
        z = np.load(path)
        self.players = z["players"].astype(np.int64)
        self.stacks = z["stacks"].astype(np.float64)
        self.node_off = z["node_off"]
        self.node_type = z["node_type"]
        self.player = z["player"]
        self.child = z["child"]
        self.kind = z["kind"]
        self.amount = z["amount"]
        self.n_actions = z["n_actions"]
        self.ioff = z["ioff"]
        self.strategy = z["strategy"]
        self.K = int(z["K"])
        self.bucket_of_class = z["bucket_of_class"]
        self.by_players = {}
        for i, n in enumerate(self.players):
            self.by_players.setdefault(int(n), []).append(i)

    def tree_index(self, n_players, eff_bb):
        idx = self.by_players.get(int(n_players))
        if not idx:
            return -1
        stacks = self.stacks[idx]
        d = np.abs(np.log(stacks) - math.log(max(eff_bb, 1.0)))
        return int(idx[int(np.argmin(d))])

    def root(self, ti):
        return int(self.node_off[ti])

    def probs(self, node, bucket):
        a = int(self.n_actions[node])
        base = int(self.ioff[node]) + bucket * a
        p = self.strategy[base:base + a].astype(np.float64)
        s = p.sum()
        if s <= 0:
            return np.full(a, 1.0 / a)
        return p / s
