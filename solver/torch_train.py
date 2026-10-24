"""Solve the heads-up abstract game stack by stack on one device with the PyTorch solver
(`solver.torch_cfr`), warm-starting every stack from the previous one.

GPU example (CUDA graphs are captured automatically on a CUDA device):
  python -m solver.torch_train --abstraction blueprint/abs_K50.npz --out blueprint/runs/gpu1 \\
      --device cuda --iters 30000 --log-every 500
Outputs are the same files as `solver.train`, so `solver.export` and `tools/build_blueprint.sh`
work unchanged. Stacks are solved in ascending order by default so the expensive deep
stacks start from a neighbour's regrets.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from solver.cfr import MODE_NAMES
from solver.driver import parse_sizes, solve_loop, warm_start
from solver.torch_cfr import TorchSolver
from solver.tree import TreeConfig, build_tree, summarize


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
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--time-limit", type=float, default=0.0, help="minutes per stack, 0 = none")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="f32", choices=["f32", "f64"])
    ap.add_argument("--no-graph", action="store_true", help="eager execution even on CUDA")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0: torch default)")
    ap.add_argument("--warm", default="auto", choices=["auto", "none"])
    ap.add_argument("--warm-dirs", default="")
    ap.add_argument("--warm-strategy", default="")
    ap.add_argument("--warm-scale", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint", type=int, default=1)
    ap.add_argument("--order", default="asc", choices=["asc", "desc"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    ab = dict(np.load(args.abstraction))
    dtype = torch.float32 if args.dtype == "f32" else torch.float64
    stacks = sorted((float(s) for s in args.stacks.split(",")), reverse=args.order == "desc")
    results = []
    for stack in stacks:
        t_start = time.time()
        cfg = TreeConfig(stack_bb=stack, sizes=parse_sizes(args.sizes), raise_cap=args.cap)
        tree = build_tree(cfg)
        log_path = os.path.join(args.out, f"hu_{stack:g}bb.log")

        def log(msg, _p=log_path):
            with open(_p, "a") as f:
                f.write(msg + "\n")
            print(f"[{stack:g}bb] {msg}", flush=True)

        log(f"stack={stack}bb mode={MODE_NAMES[args.mode]} avg={args.avg} R0={args.R0} device={args.device} "
            f"dtype={args.dtype} graph={not args.no_graph} {summarize(tree)}")
        sol = TorchSolver(tree, ab, mode=args.mode, fixed_R=args.R0, avg_weight=args.avg, device=args.device,
                          dtype=dtype, cuda_graph=False if args.no_graph else None)
        warm_start(sol, stack, args, args.out, log)
        t0 = time.time()
        sol.iterate(1)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        log(f"first iteration (incl. graph capture) in {time.time() - t0:.1f}s; infoset entries={sol.I:,} "
            f"vector entries={sol.V:,} level groups={len(sol.groups)}")
        done, ex = solve_loop(sol, stack, cfg, args, args.out, log, t_start)
        results.append((stack, done, ex))
        del sol
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    for stack, done, ex in results:
        print(f"stack {stack:g}bb: {done} iters, exploitability {ex} mbb/hand")


if __name__ == "__main__":
    main()
