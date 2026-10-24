"""Shared pieces of the heads-up training drivers (`solver.train` on CPU processes,
`solver.torch_train` on one GPU): warm-start lookup, checkpoints, logging and the solve loop.

Files per stack in the output directory:
  hu_{stack}bb.npz    average strategy, last iterate, history (what `solver.export` reads)
  hu_{stack}bb.log    progress log
  state_{stack}bb.npz full solver state (regrets, predictions, averages) for warm starts and resume
"""

import glob
import json
import math
import os
import re
import time

import numpy as np


def parse_sizes(text):
    # "0.5,1|0.5,1|0.5,1|0.5,1" -> tuple of tuples per street
    return tuple(tuple(float(x) for x in street.split(",")) for street in text.split("|"))


def state_path(out_dir, stack):
    return os.path.join(out_dir, f"state_{stack:g}bb.npz")


def result_path(out_dir, stack):
    return os.path.join(out_dir, f"hu_{stack:g}bb.npz")


def _stack_of(path):
    m = re.search(r"_(\d+(?:\.\d+)?)bb\.npz$", os.path.basename(path))
    return float(m.group(1)) if m else None


def nearest_state(dirs, stack, exclude_self=True):
    """Path and stack of the state checkpoint nearest in log(stack) across `dirs`."""
    best = None
    for d in dirs:
        for f in glob.glob(os.path.join(d, "state_*bb.npz")):
            s = _stack_of(f)
            if s is None or (exclude_self and abs(s - stack) < 1e-9):
                continue
            dist = abs(math.log(s) - math.log(stack))
            if best is None or dist < best[0]:
                best = (dist, f, s)
    return (best[1], best[2]) if best else None


def nearest_result(dirs, stack):
    """Path and stack of the nearest `hu_*bb.npz` result (for strategy-based warm starts)."""
    best = None
    for d in dirs:
        for f in glob.glob(os.path.join(d, "hu_*bb.npz")):
            s = _stack_of(f)
            if s is None:
                continue
            dist = abs(math.log(s) - math.log(stack))
            if best is None or dist < best[0]:
                best = (dist, f, s)
    return (best[1], best[2]) if best else None


def load_state(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def save_state(path, sol):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **sol.state_dict())
    os.replace(tmp, path)


def warm_start(sol, stack, args, out_dir, log):
    """Apply the warm-start policy from `args` (resume > state of a neighbour > strategy)."""
    own = state_path(out_dir, stack)
    if getattr(args, "resume", False) and os.path.exists(own):
        sol.load_state(load_state(own))
        log(f"resumed from {own} at t={int(sol.t[0])}")
        return "resume"
    policy = getattr(args, "warm", "auto")
    if policy == "none":
        return None
    dirs = [out_dir] + [d for d in getattr(args, "warm_dirs", "").split(",") if d]
    found = nearest_state(dirs, stack)
    if found:
        path, donor = found
        frac = sol.warm_start(load_state(path))
        log(f"warm start from {path} ({donor:g}bb): {frac:.1%} of infoset entries initialised")
        return path
    sdir = getattr(args, "warm_strategy", "")
    if sdir:
        found = nearest_result([sdir], stack)
        if found:
            path, donor = found
            with np.load(path) as z:
                sigma = z["current"].astype(np.float64)
                R = float(z["R"].max()) if "R" in z.files else 1.0
                sizes, cap = json.loads(str(z["sizes"])), int(z["raise_cap"])
            from solver.tree import TreeConfig, build_tree
            from solver.cfr import tree_layout
            donor_tree = build_tree(TreeConfig(stack_bb=donor, sizes=tuple(tuple(s) for s in sizes), raise_cap=cap))
            st = tree_layout(donor_tree, sol.K_s)
            st["R_global"] = np.array([R, R])
            scale = args.warm_scale if getattr(args, "warm_scale", 0) > 0 else 2.0 * max(R, 1.0)
            frac = sol.warm_start(st, strategy=sigma, scale=scale)
            log(f"warm start from strategy {path} ({donor:g}bb), scale={scale:g}: {frac:.1%} initialised")
            return path
    log("cold start")
    return None


def solve_loop(sol, stack, cfg, args, out_dir, log, t_start, extra=None):
    """Iterate, evaluate every `log_every`, write results and the state checkpoint."""
    out_path = result_path(out_dir, stack)
    st_path = state_path(out_dir, stack)
    history = []
    done = int(sol.t[0])
    if done > 0 and os.path.exists(out_path):
        with np.load(out_path) as z:
            history = [tuple(r) for r in z["history"]]
    t0 = time.time()
    t_done0 = done
    deadline = t_start + args.time_limit * 60 if args.time_limit > 0 else float("inf")
    ex = None
    while done < args.iters and time.time() < deadline:
        step = min(args.log_every, args.iters - done)
        sol.iterate(step)
        done += step
        ex, b0, b1 = sol.exploitability()
        exc, _, _ = sol.exploitability("cur")
        val = sol.game_value()
        elapsed = time.time() - t0
        history.append((done, ex / 2 * 1000, exc / 2 * 1000, val / 2 * 1000, elapsed))
        Rg = np.asarray(sol.R_global.tolist() if hasattr(sol.R_global, "tolist") else sol.R_global, dtype=float)
        log(f"t={done:6d} expl(avg)={ex / 2 * 1000:9.3f} expl(cur)={exc / 2 * 1000:9.3f} mbb/hand "
            f"value(SB)={val / 2 * 1000:8.2f} mbb  R={Rg[0]:.0f}/{Rg[1]:.0f}  "
            f"{elapsed / max(done - t_done0, 1) * 1000:.0f} ms/iter  elapsed={elapsed / 60:.1f} min")
        R_flat = sol.state_dict()["R"] if not hasattr(sol, "R") else sol.R
        np.savez(
            out_path,
            strategy=sol.average_strategy().astype(np.float32),
            current=np.asarray(sol.strat, dtype=np.float32),
            R=np.asarray(R_flat, dtype=np.float32),
            history=np.array(history, dtype=np.float64),
            stack_bb=stack,
            sizes=json.dumps(cfg.sizes),
            raise_cap=cfg.raise_cap,
            K=int(sol.K_s[1]),
            mode=args.mode,
            iters=done,
            exploitability_mbb=ex / 2 * 1000,
            **(extra or {}),
        )
        if getattr(args, "checkpoint", True):
            save_state(st_path, sol)
    log(f"finished {done} iterations in {(time.time() - t_start) / 60:.1f} min")
    return done, (ex / 2 * 1000 if ex is not None else None)
