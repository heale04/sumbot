#!/bin/bash
# Wait for every solver process to finish, then build the submission and run the checks.
cd "$(dirname "$0")/.."
LOG=blueprint/REPORT.txt
echo "finalize started $(date)" > $LOG
while pgrep -f "solver.train|solver.multi_train" >/dev/null; do sleep 120; done
echo "all solver processes finished $(date)" >> $LOG
tools/build_blueprint.sh >> $LOG 2>&1
echo "== self-check heads-up (40 tables) ==" >> $LOG
.venv/bin/python tools/selfcheck.py --tables 40 --players 2 --copies 1 >> $LOG 2>&1
echo "== self-check 8 players (40 tables, 2 copies) ==" >> $LOG
.venv/bin/python tools/selfcheck.py --tables 40 --players 8 --copies 2 >> $LOG 2>&1
echo "== simulator: 100 tables heads-up vs min_raiser ==" >> $LOG
(cd dist && ../.venv/bin/poker-agent play . ../examples/min_raiser --tables 100 --seed final1 2>/dev/null | tail -25) >> $LOG
echo "== simulator: 100 tables 8-handed (2 copies) ==" >> $LOG
(cd dist && ../.venv/bin/poker-agent play . ../examples/min_raiser ../examples/check_or_fold . ../examples/min_raiser ../examples/check_or_fold ../examples/min_raiser ../examples/check_or_fold --tables 100 --seed final2 2>/dev/null | grep -A 12 '"dist#' ) >> $LOG
echo "== exploitability summary (mbb/hand, abstract heads-up game) ==" >> $LOG
for f in blueprint/runs/*/hu_*bb.log; do echo "$f: $(tail -n 2 $f | head -n 1 | cut -c1-80)"; done >> $LOG
echo "== multiway nashconv (mbb/hand) ==" >> $LOG
for f in blueprint/runs/multi/*.log; do echo "$f: $(tail -n 2 $f | head -n 1 | cut -c1-60)"; done >> $LOG
echo "finalize done $(date)" >> $LOG
