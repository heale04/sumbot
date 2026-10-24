"""Smoke test for the PyTorch solver on a CUDA machine (run this first on the GPU box):

    PYTHONPATH=. python tools/gpu_smoke.py [--stack 20] [--abstraction blueprint/abs_K50.npz]

It checks that (1) CUDA is visible, (2) the CUDA-graph path reproduces the eager path and the
Numba CPU solver on the same iterations, and (3) reports ms/iteration on the GPU for the
chosen stack, so you know what --iters to give solver.torch_train.
"""

import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from solver.cfr import Solver  # noqa: E402
from solver.torch_cfr import TorchSolver  # noqa: E402
from solver.tree import TreeConfig, build_tree  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abstraction", default="blueprint/abs_K50.npz")
    ap.add_argument("--stack", type=float, default=20)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    ab = dict(np.load(args.abstraction))
    tree = build_tree(TreeConfig(stack_bb=args.stack))
    print(f"stack {args.stack:g}bb: {tree.n_nodes} nodes")

    print("numba float64 reference ...", flush=True)
    N = Solver(tree, ab, mode=3, dtype=np.float64, prune=False)
    N.iterate(args.iters)
    ref = N.exploitability()[0]
    print(f"  expl after {args.iters} iters: {ref / 2 * 1000:.4f} mbb/hand")

    for dev, graph, dt in (("cpu", False, torch.float64), ("cuda", False, torch.float32), ("cuda", True, torch.float32)):
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        T = TorchSolver(tree, ab, mode=3, device=dev, dtype=dt, cuda_graph=graph)
        T.iterate(1)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        T.iterate(args.iters - 1)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.time() - t0) / (args.iters - 1) * 1000
        ex = T.exploitability()[0]
        print(f"  torch {dev} graph={graph} {str(dt).split('.')[-1]}: expl={ex / 2 * 1000:.4f} mbb/hand "
              f"(rel diff vs numba {abs(ex - ref) / max(abs(ref), 1e-12):.1e})  {dt_ms:.2f} ms/iter  "
              f"R={T.R_global.tolist()} vs numba {N.R_global.tolist()}")
        del T
    if torch.cuda.is_available():
        for stack in (50, 100):
            tree = build_tree(TreeConfig(stack_bb=stack))
            T = TorchSolver(tree, ab, mode=3, device="cuda", dtype=torch.float32)
            T.iterate(1)
            torch.cuda.synchronize()
            t0 = time.time()
            T.iterate(50)
            torch.cuda.synchronize()
            print(f"  {stack}bb ({tree.n_nodes} nodes): {(time.time() - t0) / 50 * 1000:.2f} ms/iter on GPU, "
                  f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
            del T
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
