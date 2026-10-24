"""Summarise decision logs written by tools/logbot: blueprint coverage, walk failures, action mix.

    python tools/coverage.py blueprint/decisions/dec_*.jsonl
"""
import collections
import json
import re
import sys

recs = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
bet = [r for r in recs if r["phase"] == "betting"]
print(f"decisions {len(recs)}  betting {len(bet)}  errors {sum(1 for r in recs if r['err'])}")
us = sorted(r["us"] for r in recs)
print(f"latency us: max {us[-1]}  p99 {us[int(0.99 * len(us))]}  median {us[len(us) // 2]}")

print("\n== where decisions came from (street, source) ==")
c = collections.Counter()
for r in bet:
    p = r["path"]
    src = "fallback" if "fallback" in p else ("heuristic" if "multiway_preflop" in p else ("blueprint" if p else "none"))
    c[("pre" if not r["board"] else "post", "hu" if r["n"] == 2 else "multi", src)] += 1
for k in sorted(c):
    print(f"  {k}: {c[k]}")

print("\n== guard overrides ==")
g = collections.Counter(p for r in bet for p in r["path"] if p.startswith("guard:"))
for k, v in g.most_common():
    print(f"  {v} {k}")

print("\n== walk failures ==")
w = collections.Counter(re.sub(r"\(.*?\)", "", p.split(" action ")[0]) for r in bet for p in r["path"] if p.startswith("walkfail"))
for k, v in w.most_common(10):
    print(f"  {v} {k}")

print("\n== action mix (players<=3 capped, board cards, facing a bet?) ==")
m = collections.defaultdict(collections.Counter)
for r in bet:
    m[(min(r["n"], 3), len(r["board"]), "facing" if r["to_call"] > 0 else "open")][r["action"][0]] += 1
for k in sorted(m):
    t = sum(m[k].values())
    print(f"  {k} n={t} " + " ".join(f"{a}={v / t:.2f}" for a, v in m[k].most_common()))

print("\n== postflop folds facing small bets (fixed blueprint folded 32% to need<0.2 vs min_raiser) ==")
f = collections.defaultdict(collections.Counter)
for r in bet:
    if r["board"] and r["to_call"] > 0:
        need = r["to_call"] / (r["pot"] + r["to_call"])
        f["need<0.1" if need < 0.1 else "need<0.2" if need < 0.2 else "need<0.33" if need < 0.33 else "need>=0.33"][r["action"][0]] += 1
for k in sorted(f):
    t = sum(f[k].values())
    print(f"  {k} n={t} " + " ".join(f"{a}={v / t:.2f}" for a, v in f[k].most_common()))
