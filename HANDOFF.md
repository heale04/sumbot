# Handoff: GPU training run and open problems (written 2026-10-24)

Read this before touching `solver/`, `agent.py` or `hubot/`. It records what was found in
the shipped bot, what is fixed in source but not rebuilt, how to run the solver on the GPU,
and which abstractions to relax overnight, in priority order. `SOLVER.md` has the design and
the measured numbers from the first (CPU) night; `AGENTS.md` has the platform rules.

## 1. State of the repo

- `sumbot.zip` and `hubot/blueprint_hu.npz` are the artifacts from the night of 2026-10-23.
  They contain the export bug below. Do not upload them again.
- `solver/export.py` and `agent.py` carry the fixes for bugs 1 and 2 below. Nothing has been
  rebuilt or re-exported since. `tools/build_blueprint.sh blueprint/runs/misp` would ship the
  fixes on top of the old solve.
- Nothing is committed to git; every file is untracked.
- `blueprint/runs/misp/hu_*bb.npz` are the first night's heads-up solves (MI-SPCFR+),
  `blueprint/runs/multi/` the multiway preflop solves, `blueprint/abs_K50.npz` the card
  abstraction they were solved on. Only `solver/`, `tools/` and `blueprint/abs_K50.npz` are
  needed on the GPU machine to train; the run directory comes back here for export.

## 2. Why the bot played badly (diagnosed with `tools/logbot`)

Measured with the decision-logging wrapper `tools/logbot` (see section 6). Every row is a
betting decision; "fallback" is the pot-odds heuristic `fallback()` in `agent.py`.

| setting, shipped bot | postflop decisions on the blueprint | on the fallback |
|---|---|---|
| heads-up self-play | 0 | 2,125 |
| 8-handed, multiway postflop | 0 | 794 |

1. **Export dropped the children of chance nodes (fixed in `solver/export.py`).**
   `tree_tables` zipped `tree.actions[i]` with `tree.children[i]`; chance nodes have one child
   and no actions, so their child stayed -1. The heads-up walk died at every street change and
   the bot never used a postflop blueprint node, in any game. The 100 bb tree has 15,264
   chance nodes and the next node id is always the next street's root, which is how the patched
   copy was built for testing.
2. **Merged field opponent had no role (fixed in `agent.py`, `multiway_postflop`).** Opponents'
   actions are merged into a pseudo-seat -1, but the role map passed to `rt.walk` was
   `{me: role_me}`. Now `{me: role_me, -1: 1 - role_me}`. With both fixes, coverage is 99%
   heads-up and 95% in 8-handed postflop play.
3. **ICM guard vetoes correct raises (open).** `guards()` in `agent.py` blocks any call or raise
   that risks half the stack unless `icm_approves`, which estimates the win probability as
   equity against one random hand raised to the number of live opponents, with no fold equity.
   It turned 13% of all 8-handed preflop decisions from a raise into a fold, e.g. AAK facing a
   1.5 bb open with 197 bb behind. Since the multiway tree's only re-raise is all-in, the bot in
   practice never 3-bets and folds premiums at full tables. Median stack at a veto was 27 bb, so
   it also kills the push-or-fold end game.
4. **Off-tree preflop heuristic is a calling station (open).** `multiway_preflop()` takes over
   when the hand leaves the multiway tree and calls any raise where equity exceeds pot odds
   plus 0.12. In min-raise wars it called 80 chips at a time until all in with 865 and busted on
   hand 1 in 7 of 24 tables against four min_raisers.
5. **Smallest bet size is half pot (open, needs re-solve).** A quarter-pot bet maps to the
   half-pot node. With bugs 1 and 2 fixed, the blueprint folds 32% of the time to bets of a
   quarter pot or less (`need < 0.2`) against min_raiser, which is why the fixed bot beats
   min_raiser only 65% instead of the fallback's 85%.

Results of the fixed bot (patched blueprint, scratch copy) against the shipped bot: heads-up
89 to 71 over 160 tables; 8-handed against seven shipped copies 13.9 points vs 12.3 field
average over 32 tables. Real but modest, because 3 to 5 are still there.

## 3. Fixes to make in the agent (independent of the GPU run)

Do these first; they cost more points than the abstraction does.

- **ICM in the multiway solve, not a guard.** `solver/multi_cfr.py` settles terminals in chips
  (`m_terminal`). Replace the terminal payoff with the change in ICM equity of the resulting
  stacks (`poker_engine.scoring.icm`, equal starting stacks, the table's payouts; payouts vary
  per table so solve for 50/30/20 and let the bot pick the nearest or ignore the difference).
  The trees are tiny; the solve takes minutes per tree. Then delete the `icm_approves` veto for
  multiway preflop decisions and follow the blueprint.
- **Weaker guard for what remains** (heads-up blueprint used at 3+ player tables, off-tree
  spots): equity against the aggressor's likely calling range, fold equity included, applied
  only to marginal hands. Or drop it.
- **Cap the heuristic.** `multiway_preflop()` must not call more than a fraction of its stack
  per hand with a hand it would not shove with; treat the second and later raises in a hand as
  a shove decision.
- **Keep the tiny-bet rule** (`need < 0.08` means call) until the quarter-pot size exists.
- Re-run `tools/logbot` after every agent change (section 6) and check the coverage table
  before trusting match results.

## 4. GPU workflow (RTX 5090)

Untested on a GPU; the smoke test exists for that reason. From the repo root:

```sh
tools/setup_gpu.sh                      # .venv-gpu with cu128 PyTorch (Blackwell needs cu128)
source .venv-gpu/bin/activate
PYTHONPATH=. python tools/gpu_smoke.py  # CUDA visible; graph == eager == Numba to ~1e-5; ms/iter at 20/50/100 bb
```

If graph capture fails on that driver and torch version, every command below takes `--no-graph`
and runs the same code eagerly. Reference numbers for the current 100 bb tree (100k nodes,
50 buckets): Numba CPU 190 ms/iter, torch CPU 4 threads 125 ms/iter. The paper this port
follows (GPU-CFR, arXiv 2609.11923) reports 80x over CPU; expect 5 to 15 ms/iter here.
Record the smoke test number before choosing the abstraction size below.

Training, same outputs as the CPU driver (`hu_{stack}bb.npz`, `.log`, `state_{stack}bb.npz`):

```sh
PYTHONPATH=. python -m solver.torch_train --abstraction blueprint/abs_K50.npz \
    --out blueprint/runs/gpu1 --device cuda --iters 30000 --log-every 500 \
    --stacks 3,4,5,6,8,10,12,15,20,25,30,40,50,65,80,100 \
    --sizes "0.5,1|0.5,1|0.5,1|0.5,1" --cap 3
```

- Stacks run sequentially (ascending by default) and warm-start from the nearest finished
  stack's `state_*bb.npz` (`--warm auto`); `--resume` continues a stack from its own state file.
  `--warm-strategy blueprint/runs/misp` seeds from the first night's last iterates when the
  tree and abstraction are unchanged; with a new tree or new buckets the keys will not match and
  it falls back to a cold start, which is correct.
- `--log-every` controls how often exact exploitability is computed (float64, CPU-cost
  comparable to a few iterations) and the checkpoint written. The logged `expl(avg)` is the
  exploitability of the exported average strategy inside the abstract game, in mbb/hand.
  Last night: 0.01 at 3 to 8 bb, 4 at 100 bb after 30k iterations.
- `--dtype f32` is the default; evaluation is always float64.
- When done, copy `blueprint/runs/gpu1` (and the abstraction file if it changed) back and run
  `tools/build_blueprint.sh blueprint/runs/gpu1`, then the checks in section 6.

## 5. Overnight plan: which abstractions to relax, in order

Budget logic: last night needed 30k iterations for 4 mbb at 100 bb and more iterations on the
current abstraction are pointless, since 4 mbb is far below the abstraction error. If the smoke
test gives ~10 ms/iter, a 10 hour run has about 30x the current compute to spend and the
32 GB card holds about 30x the current state (state is five arrays of nodes x buckets x
actions, 500 MB at 100 bb today). Per-iteration cost scales with decision nodes x buckets x
actions plus showdown nodes x buckets squared (the batched GEMM, cheap on the GPU).

1. **Bet sizes (biggest real-world gain).** Postflop: add a third-pot and a 1.5x or 2x pot
   size, keep all-in. Preflop: 2x, 2.5x and 3x opens and a non-all-in 3-bet size. Raise cap
   stays 3. Pass them with `--sizes "0.33,0.5,1,2|0.33,0.5,1,2|0.33,0.5,1,2|0.33,0.5,1,2"`
   (preflop sizes are pot fractions too: 0.5 of the pot after calling is a min-raise to 2 bb,
   1.0 is 3 bb, so 2x/2.5x/3x opens are roughly 0.5/0.75/1.0). Expect the 100 bb tree to grow
   4x to 6x; check `summarize(tree)` in `solver/tree.py` first. This fixes bug 5 and the
   translation error on the bot's own bets. Runtime mapping is nearest size in log space
   (`rt.walk`), nothing to change there.
2. **Finer, draw-aware postflop buckets.** 50 scalar-equity buckets cannot separate a flush
   draw from a middle pair; three hole cards make draws common. Target 150 to 200 buckets on
   flop and turn from k-means over equity histograms across runouts, and about 100 river
   buckets on equity alone. `solver/buckets.py` `build_abstraction(n_buckets, n_samples,
   s_flop, s_turn, s_river, preflop_samples, seed)` currently buckets by equity quantiles; the
   histogram features and k-means are new code. The abstraction build is CPU (Numba), hours
   for 3M deals, and can run on this Mac in parallel with the GPU solve.
   **Hard constraint:** the runtime computes the bucket inside 1 ms with a 400/400/600-sample
   Monte Carlo (`hubot/runtime.py` `SAMPLES`, `bucket_for`). Histogram features need a second
   sampling level; budget about 0.5 ms and verify with `tools/selfcheck.py` before training on
   them. A bucket the bot cannot compute at the table is worthless. Finer scalar buckets also
   need more samples: with 200 buckets each is ~0.5% of equity wide and 400 samples have a
   standard error of 2.5%.
3. **Multiway preflop tree.** `solver/multi_train.py` defaults are `--sizes 0.6 --cap 1`, i.e.
   one 2.5x-ish open and all-in re-raises only. Use `--cap 3` with sizes `0.6,1.0` so 3-bets and
   4-bets exist, and consider the 1,755 exact classes instead of 100 k-means buckets
   (`cluster_preflop` in `solver/multi_cfr.py`). The trees are tiny; this costs minutes. Do this
   together with the ICM payoffs from section 3.
4. **Deeper stacks.** Chip leaders sit at 150 to 400 bb while blinds are still 10/20 and
   everything above 100 bb maps to the 100 bb tree. Add `150,200,300` to `--stacks`; warm
   starts from 100 bb make them cheap. `rt.translate` already sends the abstract all-in as a
   real all-in, so no runtime change.
5. **Not worth it:** raise cap 4, more preflop classes, more iterations on the current
   abstraction.

**Cannot be relaxed in this solver:** the two players' bucket chains are independent given the
board (imperfect recall with Markov transitions `T_pre/T_flop/T_turn`), and multiway postflop
play is the heads-up tree against a merged field (`multiway_postflop` in `agent.py`). Both are
structural; do not budget a night for them.

Suggested night: build the new abstraction on the CPU while the GPU solves the new bet-size
tree on `abs_K50.npz` for all stacks (a complete, shippable result on its own), then re-solve
the big stacks on the new abstraction with whatever time is left. Keep every run directory;
`solver.export` takes the best run per stack by logged exploitability.

## 6. Verification before any upload

```sh
.venv/bin/python tools/selfcheck.py --tables 20 --players 8   # legality and latency, in-process tables
.venv/bin/poker-agent check sumbot.zip
.venv/bin/poker-agent play tools/logbot examples/min_raiser --tables 40
.venv/bin/poker-agent play tools/logbot . . . . . . . --tables 24      # 8-handed vs itself
.venv/bin/python tools/coverage.py blueprint/decisions/dec_*.jsonl
```

`tools/logbot` wraps the repo's `agent.py`, logs every decision to
`blueprint/decisions/dec_<pid>.jsonl` with the functions it went through, every tree walk and
why a walk failed. `tools/coverage.py` prints the (street, source) table, guard overrides, walk
failure reasons, action mix and fold rate to small bets. Expect: postflop "blueprint" rows
near 95%+, no `walkfail: node was -1 on entry`, max latency well under 1 ms, zero errors.
Delete old `dec_*.jsonl` files between runs; the guest processes cannot be given a path.

The platform runs `agent.py` with a 1 ms decision limit and 100 ms time bank per table, no
GPU, Python 3.12 with the libraries in `requirements.txt` only. Nothing under `solver/` or
`blueprint/` ships; `tools/build_blueprint.sh` stages `agent.py` and `hubot/` into `dist/` and
packs `sumbot.zip` (limit 50 MB, last build 45.8 MB).
