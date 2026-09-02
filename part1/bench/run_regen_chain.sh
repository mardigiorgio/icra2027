#!/bin/bash
# Regeneration chain, 2026-09-01: randomized no-overlap per-world initial
# conditions + MuJoCo arms on Newton contacts. Runs the three GPU benches
# the design doc consumes, strictly sequentially, alone on the GPU.
# Launch detached from the icra2027 root:
#   setsid nohup bash part1/bench/run_regen_chain.sh > runlogs/regen_chain.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export VIRTUAL_ENV="$HOME/Documents/code/icra2027/.venv"
export PYTHONPATH="$PWD"
PY=.venv/bin/python
log() { echo "[chain $(date '+%m-%d %H:%M:%S')] $*"; }

log "stiffness grid (Exp 1, Newton contacts)"
$PY -m part1.bench.benchmarks.part1_stiffness_sweep || { log "FAIL stiffness"; exit 1; }

for scene in hard-clutter soft-clutter; do
  for n in 1 1024; do
    log "workprecision $scene N=$n (Exp 2)"
    $PY -m part1.bench.benchmarks.part1_workprecision --scene "$scene" --n "$n" \
      || { log "FAIL wp $scene $n"; exit 1; }
  done
done

log "scaling hard-clutter (Exp 3)"
$PY -m part1.bench.benchmarks.part1_scaling --scene hard-clutter || { log "FAIL scaling"; exit 1; }

log "figures"
$PY -c "from part1.bench import part1_plots as p; p.stiffness_sweep(); p.workprecision(); p.scaling(); p.cenic_scaling()" \
  || { log "FAIL figures"; exit 1; }
log "CHAIN COMPLETE"
