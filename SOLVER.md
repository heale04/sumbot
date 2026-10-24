# CFR solver for three-hole hold'em

Offline solver (`solver/`) that produces the blueprint strategies the bot (`agent.py` +
`hubot/`) looks up at play time. Nothing under `solver/` or `blueprint/` is shipped.

## What is solved

**Heads-up (the part with guarantees).** A two-player zero-sum abstract game:

- Cards: preflop hands are exact suit-isomorphism classes (1,755 of the 22,100 three-card
  combos). Flop, turn and river use 50 buckets each, defined by quantiles of equity vs a random
  hand (Monte Carlo, `solver/buckets.py`). A player's buckets form a Markov chain
  (`T_pre`, `T_flop`, `T_turn`) and the two players' chains are independent, which is the
  imperfect-recall abstraction. Showdown payoffs between buckets on each street (`W0`,
  `W_post`) are estimated on shared boards, so board correlation is kept where it matters.
- Betting: pot-fraction sizes 0.5 and 1.0 plus all-in on every street, up to three non-all-in
  raises per street (`solver/tree.py`); one tree per effective stack in
  {3,4,5,6,8,10,12,15,20,25,30,40,50,65,80,100} bb. 100 bb is 100k nodes.
- Solver (`solver/cfr.py`): vectorized CFR with alternating updates and quadratic averaging,
  with pluggable per-infoset regret minimizers: CFR+, PCFR+, SPCFR+ and **MI-SPCFR+**
  (Meng et al., ICLR 2026, Algorithm 1). `ard=global` keeps one adaptive regret domain
  lower bound R per player as in the paper; `ard=local` keeps one per infoset. The 3x3 game
  of the paper is reproduced by `solver/nfg_check.py` (gap 2.0e-5 vs their 2.6e-5 at 1e5
  iterations). On a 15 bb test abstraction MI-SPCFR+ (global R) beat CFR+ 8.8 vs 11.0
  mbb/hand after 3,000 iterations, and its last iterate was 3x less exploitable.
- Guarantee: `Solver.exploitability()` computes the exact best response of both players
  inside the abstract game (abstract-recall best responder). The number logged per stack
  (`blueprint/runs/*/hu_*bb.log`) is the exploitability of the shipped average strategy in
  that abstract game. Real-game exploitability is larger by the abstraction error, which is
  not bounded.
- QRE: `qre_tau > 0` adds entropy-regularized counterfactual utilities (agent QRE at every
  infoset, rationality 1/tau in chip units). Off by default; the bot samples from the
  mixed Nash average strategy instead.

**Multiway (3 to 8 players, preflop).** `solver/multi_tree.py` + `solver/multi_cfr.py`:
an N-player preflop tree (2.5x open, re-raises are all-in only), 100 preflop buckets
(k-means over matchup profiles), and a terminal model that settles called pots by multiway
equity with position and aggressor realization factors (`alpha`, `beta`). Solved with CFR+;
`nash_conv()` reports the sum of best-response gains, which is a diagnostic, not a guarantee
(there is none for 3+ players). Postflop multiway play reuses the heads-up postflop tree
against the field, entering at the preflop line whose pot and stack-to-pot ratio match.

## Running

CPU (one process per stack, MI-SPCFR+ only; the CFR+ control grid of the first night is no
longer run by default):

```sh
PYTHONPATH=. .venv/bin/python -m solver.train --abstraction blueprint/abs_K50.npz \
    --out blueprint/runs/misp2 --iters 30000 --workers 7
PYTHONPATH=. .venv/bin/python -m solver.multi_train --buckets blueprint/multi_buckets_K100.npz \
    --out blueprint/runs/multi --minutes 75 --workers 6
tools/build_blueprint.sh      # export best-per-stack (default: blueprint/runs/misp), stage dist/, pack
python tools/selfcheck.py --tables 40 --players 8   # legality + latency + points
```

GPU (one device, stacks solved in sequence, each warm-started from its neighbour):

```sh
PYTHONPATH=. python tools/gpu_smoke.py                      # CUDA visible, graph path == eager == Numba, ms/iter
PYTHONPATH=. python -m solver.torch_train --abstraction blueprint/abs_K50.npz \
    --out blueprint/runs/gpu1 --device cuda --iters 30000 --log-every 500
tools/build_blueprint.sh blueprint/runs/gpu1
```

`tools/setup_gpu.sh` creates the solver environment on a CUDA machine (PyTorch cu128 wheels,
needed for Blackwell cards such as the RTX 5090). Only `solver/`, `tools/`, and
`blueprint/abs_K50.npz` are needed there; copy the resulting `blueprint/runs/gpu1` back and
run `tools/build_blueprint.sh blueprint/runs/gpu1` here to export and pack.

`tools/finalize.sh` waits for the solver processes, then runs the export, pack and checks and
writes `blueprint/REPORT.txt`. Abstraction and bucket files are rebuilt with
`solver.buckets.build_abstraction` and `solver.multi_cfr.cluster_preflop`.

## Speed features (added after the first overnight run)

Measured on the idle M4 Pro at 100 bb (100,501 nodes, 5.3 M infoset entries), one iteration:

| path | ms/iter |
|---|---|
| first-night Numba kernels (float64, per-terminal products) | 250 |
| Numba now (float32, batched terminals, exact pruning) | 190 |
| PyTorch CPU, 4 threads (level-wise dataflow, `solver/torch_cfr.py`) | 125 |
| PyTorch CUDA with graph replay | run `tools/gpu_smoke.py` on the GPU box |

- **Batched terminals.** All showdown terminals of a street are one BLAS product against the
  street's showdown matrix instead of one 50x50 product per terminal. Bit-identical results.
- **float32 state** (`--dtype f32`, default). Exploitability and the game value are always
  evaluated in float64 buffers; the 20 bb trajectory agrees with float64 to 1e-8 relative.
- **Exact partial pruning** (`--prune 1`, default). For the traversing player, subtrees the
  opponent reaches with probability zero under the current strategy skip their value
  gathers; the F = 0 update (including MI-SPRM+'s adaptive-R statistic) is applied in closed
  form, so the trajectory is identical to the unpruned one. Under the night's final
  strategies 9 to 20 percent of nodes are dead; the saving is a few percent because the
  per-infoset regret bookkeeping cannot be skipped.
- **Checkpoints and warm starts.** Every log interval writes `state_{stack}bb.npz` (regrets,
  predictions, average). A new stack initialises from the nearest stack's checkpoint by
  matching nodes on their stack-independent action path (`tree.key`, raise sizes as pot
  fractions, all-in as `A`); 60 to 100 percent of entries match between neighbouring stacks.
  Test on the K50 abstraction, 25 bb warm-started from 20 bb after 3,000 iterations:
  exploitability 143 vs 331 mbb at 500 iterations, 68 vs 128 at 1,000, 29 vs 36 at 2,000 and
  19 vs 22 at 3,000. `--resume` continues a stack from its own checkpoint. Runs without a
  state file (the first night's) are seeded from their last iterate with
  `--warm-strategy DIR` (theta = scale * sigma, default scale 2R; scale below R is useless
  because the projection then flattens the strategy): 25 bb seeded from the night's 25 bb
  last iterate is at 1.7 mbb after 500 iterations and 1.1 after 3,000, against 331 and 22
  cold. The final exploitability is still measured exactly, so warm starts do not weaken the
  reported guarantee.
- **PyTorch solver** (`solver/torch_cfr.py`). Same game, same regret minimizers (CFR+ and
  MI-SPRM+ with the paper's global R), expressed as a fixed sequence of batched gathers,
  products and scatters over level groups (street, depth), 24 groups at 100 bb. State is
  updated in place so each half-iteration can be captured as a CUDA graph and replayed.
  Checkpoints use the Numba flat layout, so the two solvers warm-start each other and
  `solver.export` reads either. Agreement with the Numba solver: 1e-13 relative in float64,
  1e-5 in float32, after 200 iterations on 6 and 12 bb.

## Runtime (`hubot/`)

`runtime.py` parses `state.history` into per-street action lists, picks the stack tree,
replays the hand through the tree (off-tree raise sizes map to the nearest size in log
space, all-ins to the all-in branch), computes the bucket (class lookup preflop, 400 to 600
sample Monte Carlo equity postflop) and samples an action from the stored uint8
probabilities. Decisions take 10 to 100 µs after the import-time warm-up.

## Overnight results (Oct 23 2026, 30,000 iterations per heads-up stack)

Full report: `blueprint/REPORT.txt`. Exploitability is measured exactly in the abstract
heads-up game (mbb/hand); the shipped blueprint takes the lower of CFR+ and MI-SPCFR+ per stack.

| stack (bb) | 3-8 | 10 | 12 | 15 | 20 | 25 | 30 | 40 | 50 | 65 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| expl (mbb) | ≤0.01 | 0.02 | 0.26 | 0.83 | 0.26 | 0.51 | 0.95 | 1.18 | 2.45 | 3.81 | 3.50 | 3.95 |
| solver | CFR+ | MI | MI | MI | MI | MI | MI | MI | MI | MI | MI | MI |

MI-SPCFR+ (global R) beat CFR+ at every stack from 10bb up (e.g. 100bb: 3.95 vs 4.85).
Multiway preflop trees (N=3..8, 11 stacks each) reached NashConv 0.06-1.7 mbb/hand.
QRE sweep (`qre_tau` 0.05/0.2/0.5 on the K10 test abstraction) raised exploitability to
840/1370/1490 mbb/hand versus 8.7 for tau=0, so the shipped bot samples from the Nash average
strategy; the flag stays available in `solver/cfr.py`.
Final `sumbot.zip`: 45.8 MB (limit 50 MB), 0 illegal actions, max act latency 193 us idle.
