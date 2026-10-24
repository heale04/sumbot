#!/bin/bash
# Export the solved strategies into the runtime package, stage a clean submission, pack and check it.
#   tools/build_blueprint.sh [hu_run_dirs_comma_separated] [multi_run_dir]
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
HU_RUNS=${1:-${HU_RUNS:-blueprint/runs/misp}}
MULTI_RUNS=${2:-blueprint/runs/multi}
export PYTHONPATH=$PWD
echo "== heads-up export (best solver per stack) =="
$PY -m solver.export --runs "$HU_RUNS" --abstraction blueprint/abs_K50.npz --out hubot/blueprint_hu.npz
if ls "$MULTI_RUNS"/multi_N*_*bb.npz >/dev/null 2>&1; then
  echo "== multiway export =="
  $PY -m solver.multi_export --runs "$MULTI_RUNS" --buckets blueprint/multi_buckets_K100.npz --out hubot/blueprint_multi.npz
fi
echo "== staging dist/ =="
rm -rf dist && mkdir -p dist/hubot
cp agent.py dist/
cp hubot/*.py hubot/*.npz dist/hubot/
rm -f sumbot.zip
.venv/bin/poker-agent pack dist sumbot.zip
.venv/bin/poker-agent check sumbot.zip | head -5
ls -la sumbot.zip
