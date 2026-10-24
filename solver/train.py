"""Solve the heads-up abstract game over a grid of effective stacks, in parallel CPU processes.

Example:
  python -m solver.train --abstraction blueprint/abs_K50.npz --out blueprint/runs/night2 \\
      --stacks 3,4,5,6,8,10,12,15,20,25,30,40,50,65,80,100 --iters 30000 --workers 7
Each stack writes hu_{stack}bb.npz (average strategy, exploitability history) and
state_{stack}bb.npz (full solver state) every log interval. A job warm-starts from the
nearest stack's state checkpoint that already exists (`--warm auto`, the default), or resumes
its own (`--resume`). For one GPU use `solver.torch_train` instead.
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

from solver.cfr import MODE_NAMES, Solver
from solver.driver import parse_sizes, solve_loop, warm_start
from solver.tree import TreeConfig, build_tree, summarize


def solve_stack(stack, args):
    t_start = time.time()
    ab = dict(np.load(args.abstraction))
    cfg = TreeConfig(stack_bb=stack, sizes=parse_sizes(args.sizes), raise_cap=args.cap)
    tree = build_tree(cfg)
    log_path = os.path.join(args.out, f"hu_{stack:g}bb.log")

    def log(msg):
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    dtype = np.float32 if args.dtype == "f32" else np.float64
    log(f"stack={stack}bb mode={MODE_NAMES[args.mode]} avg={args.avg} R0={args.R0} qre={args.qre} "
        f"dtype={args.dtype} prune={args.prune} {summarize(tree)}")
    sol = Solver(tree, ab, mode=args.mode, fixed_R=args.R0, avg_weight=args.avg, qre_tau=args.qre, ard=args.ard,
                 dtype=dtype, prune=bool(args.prune))
    warm_start(sol, stack, args, args.out, log)
    t0 = time.time()
    sol.iterate(1)
    log(f"compiled in {time.time() - t0:.1f}s; infoset entries={sol.I:,} vector entries={sol.V:,}")
    done, ex = solve_loop(sol, stack, cfg, args, args.out, log, t_start)
    return stack, done, ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abstraction", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stacks", default="3,4,5,6,8,10,12,15,20,25,30,40,50,65,80,100")
    ap.add_argument("--sizes", default="0.5,1|0.5,1|0.5,1|0.5,1")
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--mode", type=int, default=3)
    ap.add_argument("--avg", default="quad")
    ap.add_argument("--R0", type=float, default=1.0)
    ap.add_argument("--qre", type=float, default=0.0)
    ap.add_argument("--ard", default="global", help="MI-SPCFR+ regret-domain scope: global (paper) or local")
    ap.add_argument("--dtype", default="f32", choices=["f32", "f64"])
    ap.add_argument("--prune", type=int, default=1, help="exact partial pruning of subtrees the opponent never reaches")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--time-limit", type=float, default=0.0, help="minutes per stack, 0 = none")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--warm", default="auto", choices=["auto", "none"],
                    help="auto: initialise from the nearest stack's state checkpoint if one exists")
    ap.add_argument("--warm-dirs", default="", help="extra directories searched for state checkpoints")
    ap.add_argument("--warm-strategy", default="", help="run dir whose hu_*bb.npz last iterates seed theta when no state exists")
    ap.add_argument("--warm-scale", type=float, default=0.0, help="theta = scale * sigma for strategy warm starts (0: use 2R)")
    ap.add_argument("--resume", action="store_true", help="continue from this stack's own state checkpoint")
    ap.add_argument("--checkpoint", type=int, default=1, help="write state_{stack}bb.npz every log interval")
    ap.add_argument("--order", default="desc", choices=["desc", "asc"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stacks = sorted((float(s) for s in args.stacks.split(",")), reverse=args.order == "desc")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.workers, len(stacks))) as pool:
        results = pool.starmap(solve_stack, [(s, args) for s in stacks])
    for stack, done, ex in sorted(results):
        print(f"stack {stack:g}bb: {done} iters, exploitability {ex} mbb/hand")


if __name__ == "__main__":
    main()
