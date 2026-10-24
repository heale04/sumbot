"""Solve the N-player preflop model over (players, stack) jobs in parallel.

  python -m solver.multi_train --buckets blueprint/multi_buckets_K100.npz --out blueprint/runs/multi \
      --players 3,4,5,6,7,8 --stacks 5,8,12,16,20,25,32,40,50,65,100 --minutes 60 --workers 6
"""

import argparse
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np

from solver.multi_cfr import MultiSolver
from solver.multi_tree import MultiConfig, build_multi_tree


def solve_job(players, stack, args):
    t_start = time.time()
    z = np.load(args.buckets)
    cfg = MultiConfig(players=players, stack_bb=stack, sizes=tuple(float(x) for x in args.sizes.split(",")), raise_cap=args.cap)
    tree = build_multi_tree(cfg)
    tag = f"multi_N{players}_{stack:g}bb"
    log_path = os.path.join(args.out, tag + ".log")
    out_path = os.path.join(args.out, tag + ".npz")

    def log(msg):
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    sol = MultiSolver(tree, z["Pw"], z["p_bucket"], pos_alpha=args.alpha, agg_beta=args.beta)
    log(f"N={players} stack={stack}bb nodes={tree.n_nodes:,} infoset entries={sol.I:,}")
    t0 = time.time()
    sol.iterate(1)
    log(f"compiled+1 iter in {time.time() - t0:.1f}s")
    history = []
    deadline = t_start + args.minutes * 60
    step = args.log_every
    while sol.t < args.iters and time.time() < deadline:
        sol.iterate(step)
        nc, per = sol.nash_conv()
        elapsed = time.time() - t0
        history.append((sol.t, nc / 2 * 1000, elapsed))
        log(f"t={sol.t:6d} nashconv={nc / 2 * 1000:9.3f} mbb/hand  {elapsed / sol.t * 1000:.0f} ms/iter  elapsed={elapsed / 60:.1f} min")
        np.savez(out_path, strategy=sol.average_strategy().astype(np.float32), history=np.array(history),
                 players=players, stack_bb=stack, sizes=json.dumps(cfg.sizes), raise_cap=cfg.raise_cap,
                 alpha=args.alpha, beta=args.beta, iters=sol.t, nashconv_mbb=nc / 2 * 1000)
        if nc / 2 * 1000 < args.target:
            break
    log(f"finished {sol.t} iterations in {(time.time() - t_start) / 60:.1f} min")
    return players, stack, sol.t, history[-1][1] if history else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--players", default="3,4,5,6,7,8")
    ap.add_argument("--stacks", default="5,8,12,16,20,25,32,40,50,65,100")
    ap.add_argument("--sizes", default="0.6")
    ap.add_argument("--cap", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--beta", type=float, default=0.12)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--minutes", type=float, default=60.0, help="time limit per job")
    ap.add_argument("--target", type=float, default=0.3, help="stop when nashconv (mbb/hand) is below this")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    jobs = [(int(n), float(s)) for n in args.players.split(",") for s in args.stacks.split(",")]
    jobs.sort(key=lambda j: (-j[0], -j[1]))  # biggest first
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results = pool.starmap(solve_job, [(n, s, args) for n, s in jobs])
    for n, s, t, nc in sorted(results):
        print(f"N={n} {s:g}bb: {t} iters nashconv={nc}")


if __name__ == "__main__":
    main()
