"""Decision-logging wrapper around the repo's agent.py, for measuring blueprint coverage.

Play it like any bot directory:
    .venv/bin/poker-agent play tools/logbot examples/min_raiser --tables 40
Every decision is appended as one JSON line to blueprint/decisions/dec_<pid>.jsonl (the guest
process does not receive environment variables, so the path is fixed). Summarise with
    python tools/coverage.py blueprint/decisions/dec_*.jsonl
The `path` field lists the functions a decision went through (heads_up, fallback,
multiway_preflop_blueprint, multiway_preflop, multiway_postflop, icm_approves, guard:*), every
tree walk with its inputs and result, and a `walkfail:` line explaining an off-tree walk.
"""
import importlib.util, json, math, os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SRC = os.path.join(ROOT, "agent.py")
spec = importlib.util.spec_from_file_location("sumbot_agent", SRC)
sb = importlib.util.module_from_spec(spec)
sys.modules["sumbot_agent"] = sb
spec.loader.exec_module(sb)
import poker_agent_sdk
from poker_agent_sdk import Action

LOG_DIR = os.path.join(ROOT, "blueprint", "decisions")
os.makedirs(LOG_DIR, exist_ok=True)
LOG = open(os.path.join(LOG_DIR, f"dec_{os.getpid()}.jsonl"), "a")
PATH = []


def wrap(name):
    f = getattr(sb, name)

    def g(*a, **k):
        PATH.append(name)
        return f(*a, **k)

    setattr(sb, name, g)


for n in ("heads_up", "fallback", "multiway_preflop_blueprint", "multiway_preflop", "multiway_postflop", "icm_approves"):
    wrap(n)
_guards = sb.guards


def guards(state, info, hole, board, me, live, action):
    out = _guards(state, info, hole, board, me, live, action)
    if out is not action:
        PATH.append(f"guard:{action.kind}->{out.kind}")
    return out


sb.guards = guards
_walk = sb.rt.walk


def walk(bp, node, actions, role_of_seat, start_units):
    out = _walk(bp, node, actions, role_of_seat, start_units)
    PATH.append(f"walk(node={node} type={int(bp.node_type[node]) if node >= 0 else None} actions={actions} "
                f"roles={role_of_seat}) -> {out} type={int(bp.node_type[out]) if out >= 0 else None}")
    if out < 0:
        n = node
        why = "node was -1 on entry" if node < 0 else "?"
        for i, (seat, kind, amt) in enumerate(actions):
            if n < 0:
                break
            if bp.node_type[n] != sb.rt.DECISION:
                why = f"step{i}: node type {int(bp.node_type[n])} not decision"
                break
            role = role_of_seat.get(seat)
            if role is None or bp.player[n] != role:
                why = f"step{i}: role {role} vs node player {int(bp.player[n])} action {(seat, kind, amt)}"
                break
            a = int(bp.n_actions[n])
            kinds = [int(x) for x in bp.kind[n, :a]]
            if kind == 2:
                ri = [j for j in range(a) if kinds[j] == 2]
                if not ri:
                    why = f"step{i}: no raise option, kinds {kinds}"
                    break
                top = max(ri, key=lambda j: bp.amount[n, j])
                if amt >= start_units.get(seat, 1e18) - 1e-9:
                    ch = top
                else:
                    ch = min(ri, key=lambda j: abs(math.log(max(bp.amount[n, j], 1)) - math.log(max(amt, 1))))
            else:
                ch = next((j for j in range(a) if kinds[j] == kind), -1)
                if ch < 0:
                    why = f"step{i}: no option kind {kind}, kinds {kinds}"
                    break
            n = int(bp.child[n, ch])
        PATH.append("walkfail:" + why)
    return out


sb.rt.walk = walk


class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        PATH.clear()
        t0 = time.perf_counter_ns()
        try:
            action = sb.decide(state)
            err = None
        except Exception as e:  # log it, then fail safe like the real agent
            err = repr(e)
            action = Action.check() if state.can("check") else (Action.fold() if state.can("fold") else Action(state.legal_actions[0].kind))
        dt = (time.perf_counter_ns() - t0) / 1000
        me = next(s for s in state.seats if s.seat == state.seat)
        rec = dict(tbl=state.config.table_id, hand=state.hand_number, phase=state.phase,
                   n=sum(1 for s in state.seats if s.status != "busted"), seat=state.seat, btn=state.button,
                   bb=state.blinds.big_blind, cards=state.cards, board=state.board, pot=state.pot, to_call=state.to_call,
                   stack=me.stack, bet=me.bet, stacks=[s.stack for s in state.seats],
                   hist=[(e.kind, e.seat, e.amount) for e in state.history if e.kind in ("fold", "check", "call", "raise", "board", "blind")],
                   legal=[(l.kind, l.min_amount, l.max_amount) for l in state.legal_actions],
                   path=list(PATH), action=(action.kind, action.amount), us=round(dt), err=err)
        LOG.write(json.dumps(rec) + "\n")
        LOG.flush()
        return action, memory
