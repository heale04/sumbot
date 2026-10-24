"""Pack the multiway preflop solutions into hubot/blueprint_multi.npz."""

import argparse
import glob
import json
import os

import numpy as np

from solver.multi_tree import DECISION, MultiConfig, build_multi_tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--buckets", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    z = np.load(args.buckets)
    K = len(z["p_bucket"])
    jobs = []
    for f in sorted(glob.glob(os.path.join(args.runs, "multi_N*_*bb.npz"))):
        r = np.load(f)
        jobs.append((int(r["players"]), float(r["stack_bb"]), f))
    jobs.sort()
    players, stacks, node_off, parts, strategies, info = [], [], [0], [], [], []
    max_a = 0
    for n, s, f in jobs:
        r = np.load(f)
        cfg = MultiConfig(players=n, stack_bb=s, sizes=tuple(json.loads(str(r["sizes"]))), raise_cap=int(r["raise_cap"]))
        tree = build_multi_tree(cfg)
        arr = tree.arrays()
        strat = np.clip(np.rint(r["strategy"] * 255.0), 0, 255).astype(np.uint8)
        expected = sum(K * int(arr["n_actions"][i]) for i in range(tree.n_nodes) if arr["node_type"][i] == DECISION)
        assert len(strat) == expected, (f, len(strat), expected)
        players.append(n)
        stacks.append(s)
        parts.append(arr)
        strategies.append(strat)
        max_a = max(max_a, arr["child"].shape[1])
        node_off.append(node_off[-1] + tree.n_nodes)
        info.append((n, s, int(r["iters"]), float(r["nashconv_mbb"])))
        print(f"N={n} {s:g}bb: nodes={tree.n_nodes:,} iters={int(r['iters'])} nashconv={float(r['nashconv_mbb']):.2f} mbb bytes={strat.nbytes:,}")

    def cat(key, fill):
        out = []
        for arr in parts:
            a = arr[key]
            pad = np.full((a.shape[0], max_a), fill, dtype=a.dtype)
            pad[:, : a.shape[1]] = a
            out.append(pad)
        return np.concatenate(out)

    child = cat("child", -1).astype(np.int32)
    for i in range(len(parts)):
        seg = child[node_off[i]:node_off[i + 1]]
        seg[seg >= 0] += node_off[i]
    node_type = np.concatenate([a["node_type"] for a in parts]).astype(np.int8)
    n_actions = np.concatenate([a["n_actions"] for a in parts]).astype(np.int8)
    ioff = np.zeros(len(node_type) + 1, dtype=np.int64)
    for i in range(len(node_type)):
        ioff[i + 1] = ioff[i] + (K * int(n_actions[i]) if node_type[i] == DECISION else 0)
    strategy = np.concatenate(strategies)
    assert len(strategy) == ioff[-1]
    np.savez_compressed(
        args.out,
        players=np.array(players, dtype=np.int8),
        stacks=np.array(stacks, dtype=np.float32),
        node_off=np.array(node_off, dtype=np.int64),
        node_type=node_type,
        player=np.concatenate([a["player"] for a in parts]).astype(np.int8),
        child=child,
        kind=cat("kind", -1),
        amount=cat("amount", 0),
        n_actions=n_actions,
        ioff=ioff,
        strategy=strategy,
        K=K,
        bucket_of_class=z["labels"].astype(np.int16),
        info=np.array(info, dtype=np.float64),
    )
    print(f"wrote {args.out}: {os.path.getsize(args.out) / 1e6:.1f} MB, {len(jobs)} trees, {len(strategy):,} strategy bytes")


if __name__ == "__main__":
    main()
