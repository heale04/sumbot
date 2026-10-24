"""HUQT three-hole hold'em bot: CFR blueprint (heads-up) plus multiway heuristics.

Loads the blueprint and JIT-compiles the equity estimator at import time; each decision is
one tree walk, one Monte Carlo equity estimate, and a table lookup.
"""

import gc
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import poker_agent_sdk
from poker_agent_sdk import Action, GameState

from hubot import runtime as rt
from hubot.equity import equity, seed_rng
from hubot.icm import icm

BLUEPRINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hubot", "blueprint_hu.npz")
MULTI_BLUEPRINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hubot", "blueprint_multi.npz")
BP = rt.Blueprint(BLUEPRINT)
MBP = rt.MultiBlueprint(MULTI_BLUEPRINT) if os.path.exists(MULTI_BLUEPRINT) else None
EQ1 = None  # preflop class equity vs one random hand, filled from the blueprint if present
try:
    EQ1 = np.load(BLUEPRINT)["eq1"]
except KeyError:
    EQ1 = None
seed_rng(int(time.time_ns() % (2**31)))
# warm-up: compile every Numba path once
for _nb in (0, 3, 4, 5):
    equity([0, 5, 10], list(range(20, 20 + _nb)), 50)
RNG = random.Random()


def _warm_up():
    """Run every decision path once so no first-call cost lands inside a real decision."""
    from hubot.warmup import states

    for st in states():
        for _ in range(3):
            try:
                decide(st)
            except Exception:  # a warm-up failure must not break the import
                pass


class Agent(poker_agent_sdk.Agent):
    def act(self, state: GameState, memory: bytes) -> tuple[Action, bytes]:
        try:
            return decide(state), memory
        except Exception:  # never fail a decision: fall back to the passive legal action
            if state.can("check"):
                return Action.check(), memory
            if state.can("fold"):
                return Action.fold(), memory
            return Action(state.legal_actions[0].kind, state.legal_actions[0].min_amount), memory


def legal_raise(state, chips):
    r = state.can("raise")
    if r is None:
        return Action.call() if state.can("call") else Action.check()
    amount = int(min(max(chips, r.min_amount), r.max_amount))
    return Action.raise_to(amount)


def finish(state, kind, chips):
    if kind == "fold":
        if state.can("check"):
            return Action.check()
        return Action.fold()
    if kind == "call":
        return Action.call() if state.can("call") else Action.check()
    return legal_raise(state, chips)


def decide(state: GameState) -> Action:
    if state.phase != "betting" or state.can("show") or state.can("muck"):
        # the engine lists the automatic (sensible) showdown action first
        return Action(state.legal_actions[0].kind)
    info = rt.HandInfo(state)
    hole = [rt.card_from_str(c) for c in state.cards]
    board = [rt.card_from_str(c) for c in state.board]
    me = state.seat
    live = [s for s in info.dealt if s not in info.folded]
    action = blueprint_action(state, info, hole, board, me, live)
    return guards(state, info, hole, board, me, live, action)


def guards(state, info, hole, board, me, live, action):
    """Tournament layer on top of the chip-EV blueprint: never fold to a tiny bet, and do not
    risk the stack in a multiway table when ICM says folding keeps more tournament equity."""
    if action.kind == "fold" and state.to_call > 0:
        need = state.to_call / (state.pot + state.to_call)
        if need < 0.08:
            return Action.call()
    if action.kind in ("call", "raise") and state.players_remaining >= 3:
        my = next(s for s in state.seats if s.seat == me)
        risk = state.to_call if action.kind == "call" else max(action.amount - my.bet, state.to_call)
        if risk >= 0.5 * my.stack and state.to_call > 0:
            if not icm_approves(state, info, hole, board, me, live, my, min(risk, my.stack)):
                return Action.fold() if state.can("fold") else Action.check()
    return action


def icm_approves(state, info, hole, board, me, live, my, risk):
    payouts = list(state.config.payouts)
    stacks = [s.stack + s.bet for s in state.seats]
    n_opp = max(len([s for s in live if s != me]), 1)
    e1 = equity(hole, board, 300) if board else float(EQ1[BP.class_of[tuple(sorted(hole))]]) if EQ1 is not None else 0.5
    p_win = e1 ** n_opp if n_opp > 1 else e1
    aggressor = max((s for s in state.seats if s.seat != me and s.status != "busted"), key=lambda s: s.bet)
    pot_if_called = state.pot + risk
    fold_stacks = list(stacks)
    fold_stacks[me] = my.stack
    win_stacks = list(stacks)
    win_stacks[me] = my.stack - risk + pot_if_called
    win_stacks[aggressor.seat] = max(aggressor.stack + aggressor.bet - (pot_if_called - my.bet - risk), 0)
    lose_stacks = list(stacks)
    lose_stacks[me] = my.stack - risk
    lose_stacks[aggressor.seat] = aggressor.stack + aggressor.bet + my.bet + risk
    n = state.players_remaining
    ev_fold = icm(fold_stacks, payouts[:n])[me]
    ev_call = p_win * icm(win_stacks, payouts[:n])[me] + (1 - p_win) * icm(lose_stacks, payouts[:n])[me]
    return ev_call >= ev_fold


def blueprint_action(state, info, hole, board, me, live):
    if len(info.dealt) == 2:
        return heads_up(state, info, hole, board, me)
    if info.street == 0:
        if MBP is not None:
            action = multiway_preflop_blueprint(state, info, hole, me, live)
            if action is not None:
                return action
        return multiway_preflop(state, info, hole, me, live)
    return multiway_postflop(state, info, hole, board, me, live)


def multiway_preflop_blueprint(state, info, hole, me, live):
    """Solved N-player preflop model. Returns None when this spot is not covered."""
    order = order_after_button(state, info)
    n = len(order)
    if me not in order:
        return None
    my_start = info.start_stack(state, me)
    opp_max = max((info.start_stack(state, s) for s in live if s != me), default=my_start)
    eff = min(my_start, opp_max) / info.bb
    ti = MBP.tree_index(n, eff)
    if ti < 0:
        return None
    role = {seat: i for i, seat in enumerate(order)}
    start_units = {s: info.start_stack(state, s) / info.unit for s in order}
    node = rt.walk(MBP, MBP.root(ti), info.actions[0], role, start_units)
    if node < 0 or MBP.node_type[node] != rt.DECISION or MBP.player[node] != role[me]:
        return None
    a, b, c = sorted(hole)
    bucket = int(MBP.bucket_of_class[int(BP.class_of[a, b, c])])
    idx = rt.choose(MBP, node, bucket, RNG)
    kind, chips = rt.translate(state, MBP, node, idx, info)
    return finish(state, kind, chips)


def heads_up(state, info, hole, board, me):
    opp = next(s for s in info.dealt if s != me)
    eff = min(info.start_stack(state, me), info.start_stack(state, opp)) / info.bb
    si = BP.stack_index(eff)
    role = {info.sb_seat: 0, info.bb_seat: 1}
    start_units = {s: info.start_stack(state, s) / info.unit for s in (me, opp)}
    node = BP.root(si)
    for st in range(info.street + 1):
        if st > 0:
            if BP.node_type[node] != rt.CHANCE:
                node = -1
                break
            node = int(BP.child[node, 0])
        node = rt.walk(BP, node, info.actions[st], role, start_units)
        if node < 0:
            break
    if node < 0 or BP.node_type[node] != rt.DECISION or BP.player[node] != role.get(me):
        return fallback(state, info, hole, board)
    bucket = rt.bucket_for(BP, info.street, hole, board)
    idx = rt.choose(BP, node, bucket, RNG)
    kind, chips = rt.translate(state, BP, node, idx, info)
    return finish(state, kind, chips)


def multiway_postflop(state, info, hole, board, me, live):
    """Play the heads-up postflop tree against 'the field': pick the stack and preflop entry
    whose pot and stack-to-pot ratio match, then replay this hand's postflop actions with
    the opponents' actions merged into one opponent."""
    pot_units = state.pot / info.unit
    my_stack = next(s.stack for s in state.seats if s.seat == me)
    my_remaining = my_stack / info.unit + info.street_bet[me] / info.unit
    d = np.abs(np.log(BP.entry_pot / max(pot_units, 1.0))) + np.abs(np.log(BP.entry_rem / max(my_remaining, 1.0)))
    node = int(BP.entry_node[int(np.argmin(d))])
    # role: in position (player 0) if we act after every live opponent this street
    order = order_after_button(state, info)
    live_order = [s for s in order if s in live]
    role_me = 0 if live_order and live_order[-1] == me else 1
    role = {me: role_me}
    field = 1 - role_me
    start_units = {me: info.start_stack(state, me) / info.unit}
    node = int(BP.child[node, 0])  # flop decision
    for st in range(1, info.street + 1):
        if st > 1:
            if BP.node_type[node] != rt.CHANCE:
                return fallback(state, info, hole, board)
            node = int(BP.child[node, 0])
        merged = merge_field(info.actions[st], me, field, role_me)
        node = rt.walk(BP, node, merged, role, start_units)
        if node < 0:
            return fallback(state, info, hole, board)
    if BP.node_type[node] != rt.DECISION or BP.player[node] != role_me:
        return fallback(state, info, hole, board)
    bucket = rt.bucket_for(BP, info.street, hole, board)
    idx = rt.choose(BP, node, bucket, RNG)
    kind, chips = rt.translate(state, BP, node, idx, info)
    if kind == "fold" and state.can("check"):
        kind = "call"
    return finish(state, kind, chips)


def merge_field(actions, me, field_role, my_role):
    """Rewrite a street's actions as an alternating heads-up sequence. Opponents' actions
    between my actions collapse to one action: the largest raise if any, else check/call."""
    out = []
    pending = []
    def flush():
        if not pending:
            return
        raises = [a for a in pending if a[1] == 2]
        if raises:
            out.append((-1, 2, max(a[2] for a in raises)))
        else:
            out.append((-1, 1, 0.0))
        pending.clear()
    for seat, kind, amt in actions:
        if seat == me:
            flush()
            out.append((me, kind, amt))
        elif kind != 0:
            pending.append((seat, kind, amt))
    flush()
    # heads-up order: player 1 (BB) acts first postflop; drop a leading field check when we are player 1
    seq = []
    expect = 1
    for seat, kind, amt in out:
        role = my_role if seat == me else field_role
        if role != expect:
            if role == my_role:
                seq.append((-1, 1, 0.0))  # insert a field check before our action
            else:
                continue  # field acted twice in a row (several opponents): already merged
        seq.append((seat if seat == me else -1, kind, amt))
        expect = 1 - role
    return [(s if s == me else -1, k, a) for s, k, a in seq]


def order_after_button(state, info):
    n = len(state.seats)
    return [(state.button + i) % n for i in range(1, n + 1) if (state.button + i) % n in info.dealt]


def multiway_preflop(state, info, hole, me, live):
    """Equity-threshold preflop policy for 3+ players (replaced by a solved model later)."""
    a, b, c = sorted(hole)
    cls = int(BP.class_of[a, b, c])
    e = float(EQ1[cls]) if EQ1 is not None else equity(hole, [], 300)
    n_opp = len(live) - 1
    pot = state.pot / info.bb
    to_call = state.to_call / info.bb
    my_stack_bb = next(s.stack for s in state.seats if s.seat == me) / info.bb
    raised = any(k == 2 for _, k, _ in info.actions[0])
    # thresholds rise with the number of opponents
    open_thr = 0.56 + 0.02 * max(n_opp - 1, 0)
    call_thr = 0.60 + 0.02 * max(n_opp - 1, 0)
    reraise_thr = 0.68 + 0.015 * max(n_opp - 1, 0)
    if to_call == 0 or (not raised and to_call <= 0.5 + 1e-9):
        if e >= open_thr:
            size = 2.5 * info.bb if my_stack_bb > 12 else my_stack_bb * info.bb + state.to_call
            return legal_raise(state, size)
        if state.can("check"):
            return Action.check()
        return finish(state, "fold" if e < open_thr - 0.08 else "call", 0)
    # facing a raise: pot odds with a margin
    need = to_call / (pot + to_call)
    if e >= reraise_thr:
        if my_stack_bb <= 20:
            return legal_raise(state, 10**9)
        return legal_raise(state, 3.0 * state.to_call + state.pot)
    if e >= call_thr or e >= need + 0.12:
        if to_call >= my_stack_bb * 0.5 and e < call_thr:
            return finish(state, "fold", 0)
        return finish(state, "call", 0)
    return finish(state, "fold", 0)


def fallback(state, info, hole, board):
    """Pot-odds policy when the hand walked off the abstraction."""
    e = equity(hole, board, 300) if board else (float(EQ1[BP.class_of[tuple(sorted(hole))]]) if EQ1 is not None else 0.5)
    if state.to_call == 0:
        if e > 0.72:
            return legal_raise(state, 0.75 * state.pot)
        return finish(state, "call", 0)
    need = state.to_call / (state.pot + state.to_call)
    if e > 0.8:
        return legal_raise(state, state.pot + 2 * state.to_call)
    if e >= need + 0.05:
        return finish(state, "call", 0)
    return finish(state, "fold", 0)


import math  # noqa: E402  (used by multiway_postflop)

_warm_up()
# the import-time objects never die: keep the collector away from decisions
gc.collect()
gc.freeze()
gc.set_threshold(50000, 20, 20)