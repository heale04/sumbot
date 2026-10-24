"""Play tables in-process with the agent, checking every decision for legality and latency.

  python tools/selfcheck.py --tables 20 --players 8
Opponents are the example bots; reports illegal actions (with traceback context) and the
latency distribution of agent.decide.
"""

import argparse
import importlib.util
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_agent_protocol.game import is_legal, passive_action
from poker_engine.table import SitAndGo, TableRules
from poker_engine.variant import load_variant
from poker_engine.scoring import table_points, table_places


def load_bot(path):
    spec = importlib.util.spec_from_file_location(f"bot_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod.Agent()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=int, default=10)
    ap.add_argument("--players", type=int, default=8)
    ap.add_argument("--seed", default="selfcheck")
    ap.add_argument("--copies", type=int, default=2, help="how many seats the agent takes")
    args = ap.parse_args()
    me = load_bot(ROOT / "agent.py")
    examples = [load_bot(ROOT / "examples" / n / "agent.py") for n in ("min_raiser", "check_or_fold")]
    rules = TableRules(variant="three_hole_holdem", players=args.players)
    variant = load_variant(rules.variant)
    lat = []
    slow = []
    illegal = 0
    errors = 0
    pts = {"me": [], "min_raiser": [], "check_or_fold": []}
    for t in range(args.tables):
        seats = []
        for i in range(args.players):
            if i < args.copies:
                seats.append(("me", me))
            else:
                seats.append((["min_raiser", "check_or_fold"][i % 2], examples[i % 2]))
        rng = random.Random(f"{args.seed}:{t}")
        rng.shuffle(seats)
        table = SitAndGo(rules=rules, variant=variant, table_id=f"sc-{t}", seed=f"{args.seed}:{t}")
        memory = [b""] * args.players
        failures = [0] * args.players
        while (state := table.decision()) is not None:
            seat = state.seat
            name, bot = seats[seat]
            t0 = time.perf_counter_ns()
            try:
                action, memory[seat] = bot.act(state, memory[seat])
            except Exception as e:
                errors += 1
                print("EXCEPTION", name, e)
                action = None
            dt = (time.perf_counter_ns() - t0) / 1000
            if name == "me":
                lat.append(dt)
                slow.append((dt, len(state.board), sum(1 for x in state.seats if x.status != "busted"), state.to_call, str(action)))
                if action is not None and not is_legal(action, state.legal_actions):
                    illegal += 1
                    if illegal <= 5:
                        print(f"ILLEGAL: {action} legal={state.legal_actions} street={len(state.board)} to_call={state.to_call} phase={state.phase}")
            ok = table.apply(action)
            if not ok:
                failures[seat] += 1
        result = table.result(tuple(failures))
        for seat, (name, _) in enumerate(seats):
            pts[name].append(table_points(result, table.payouts)[seat])
    lat.sort()
    n = len(lat)
    print(f"decisions={n} illegal={illegal} exceptions={errors}")
    print(f"latency us: median={lat[n//2]:.0f} p90={lat[int(n*0.9)]:.0f} p99={lat[int(n*0.99)]:.0f} max={lat[-1]:.0f}")
    slow.sort(reverse=True)
    for dt, nb, np_, tc, act in slow[:5]:
        print(f"  slow: {dt:.0f} us board={nb} players={np_} to_call={tc} -> {act}")
    for k, v in pts.items():
        if v:
            print(f"{k}: avg points {sum(v)/len(v):.2f} over {len(v)} seats")


if __name__ == "__main__":
    main()
