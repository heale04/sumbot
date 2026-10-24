"""Pack solved stacks into one compact blueprint file for the bot.

  python -m solver.export --runs blueprint/runs/night1 --abstraction blueprint/abs_K50.npz \
      --out hubot/blueprint_hu.npz
Strategies are stored as uint8 (probability * 255) per (node, bucket, action).
"""

import argparse
import glob
import json
import os

import numpy as np

from solver.iso import CLASS_OF
from solver.tree import DECISION, TreeConfig, build_tree


def tree_tables(tree):
    n = tree.n_nodes
    max_a = max(len(a) for a in tree.actions)
    child = np.full((n, max_a), -1, dtype=np.int32)
    kind = np.full((n, max_a), -1, dtype=np.int8)  # 0 fold, 1 check/call, 2 raise
    amount = np.zeros((n, max_a), dtype=np.int32)  # raise-to in units
    n_actions = np.zeros(n, dtype=np.int8)
    for i in range(n):
        acts = tree.actions[i]
        n_actions[i] = len(acts)
        for a, c in enumerate(tree.children[i]):  # chance nodes have a child but no action
            child[i, a] = c
        for a, (k, amt) in enumerate(acts):
            kind[i, a] = {"fold": 0, "check": 1, "call": 1, "raise": 2}[k]
            amount[i, a] = amt
    return {
        "node_type": np.array(tree.node_type, dtype=np.int8),
        "player": np.array(tree.player, dtype=np.int8),
        "street": np.array(tree.street, dtype=np.int8),
        "pot": np.array(tree.pot, dtype=np.int32),
        "child": child,
        "kind": kind,
        "amount": amount,
        "n_actions": n_actions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="one or more run directories, comma separated; best per stack wins")
    ap.add_argument("--abstraction", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ab = np.load(args.abstraction)
    K = int(ab["K"])
    best = {}
    for run in args.runs.split(","):
        for f in sorted(glob.glob(os.path.join(run, "hu_*bb.npz"))):
            z = np.load(f)
            stack = float(z["stack_bb"])
            ex = float(z["exploitability_mbb"])
            if stack not in best or ex < best[stack][0]:
                best[stack] = (ex, f)
    files = [best[s][1] for s in sorted(best)]
    stacks, tables, strategies, info = [], [], [], []
    max_a_all = 0
    for f in files:
        z = np.load(f)
        stack = float(z["stack_bb"])
        cfg = TreeConfig(stack_bb=stack, sizes=tuple(tuple(s) for s in json.loads(str(z["sizes"]))), raise_cap=int(z["raise_cap"]))
        tree = build_tree(cfg)
        tab = tree_tables(tree)
        strat = z["strategy"]
        q = np.clip(np.rint(strat * 255.0), 0, 255).astype(np.uint8)
        stacks.append(stack)
        tables.append(tab)
        strategies.append(q)
        max_a_all = max(max_a_all, tab["child"].shape[1])
        info.append((stack, tree.n_nodes, int(z["iters"]), float(z["exploitability_mbb"])))
        print(f"stack {stack:g}bb: {os.path.basename(os.path.dirname(f))} nodes={tree.n_nodes} iters={int(z['iters'])} expl={float(z['exploitability_mbb']):.2f} mbb strategy bytes={q.nbytes:,}")
    order = np.argsort(stacks)
    stacks = [stacks[i] for i in order]
    tables = [tables[i] for i in order]
    strategies = [strategies[i] for i in order]
    # concatenate trees with per-stack node offsets
    def cat2(key):
        parts = []
        for tab in tables:
            arr = tab[key]
            pad = np.full((arr.shape[0], max_a_all), -1 if key != "amount" else 0, dtype=arr.dtype)
            pad[:, : arr.shape[1]] = arr
            parts.append(pad)
        return np.concatenate(parts)
    node_off = np.zeros(len(tables) + 1, dtype=np.int64)
    for i, tab in enumerate(tables):
        node_off[i + 1] = node_off[i] + len(tab["node_type"])
    child = cat2("child")
    for i in range(len(tables)):
        seg = child[node_off[i]:node_off[i + 1]]
        seg[seg >= 0] += node_off[i]
    # infoset offsets per node (global)
    node_type = np.concatenate([t["node_type"] for t in tables])
    street = np.concatenate([t["street"] for t in tables])
    n_actions = np.concatenate([t["n_actions"] for t in tables])
    K_s = np.array([CLASS_OF.max() + 1, K, K, K], dtype=np.int64)
    ioff = np.zeros(len(node_type) + 1, dtype=np.int64)
    for i in range(len(node_type)):
        ioff[i + 1] = ioff[i] + (K_s[street[i]] * n_actions[i] if node_type[i] == DECISION else 0)
    strategy = np.concatenate(strategies)
    assert len(strategy) == ioff[-1], (len(strategy), ioff[-1])
    np.savez_compressed(
        args.out,
        stacks=np.array(stacks, dtype=np.float32),
        node_off=node_off,
        node_type=node_type,
        player=np.concatenate([t["player"] for t in tables]),
        street=street,
        pot=np.concatenate([t["pot"] for t in tables]),
        child=child.astype(np.int32),
        kind=cat2("kind"),
        amount=cat2("amount"),
        n_actions=n_actions,
        ioff=ioff,
        strategy=strategy,
        K=K,
        edges=ab["edges"].astype(np.float32),
        class_of=CLASS_OF,
        eq1=((1.0 + ab["W0"] @ ab["p_class"]) / 2.0).astype(np.float32),
        info=np.array(info, dtype=np.float64),
    )
    print(f"wrote {args.out}: {os.path.getsize(args.out) / 1e6:.1f} MB, {len(stacks)} stacks, {len(node_type):,} nodes, {len(strategy):,} strategy bytes")


if __name__ == "__main__":
    main()
