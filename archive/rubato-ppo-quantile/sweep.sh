#!/usr/bin/env bash
# rubato-ppo-quantile: the full Puget-scale sweep with the quantile boundary stop ON.
#
# A thin launcher over the stock framework (../rubato-ppo-sweep/sweep.sh): same tasks,
# stock env counts and iterations, same seeds, deferral, RESUME, cloud ledger and
# instance lock -- pointed at a FRESH W&B project with its own state directory, so the
# historical rubato-ppo data (adaptive WITHOUT the quantile stop) stays a clean control.
# The treatment itself arrives via the repo update the stock script performs: IsaacLab
# develop now defaults adaptive_landed_fraction=0.95.
#
# Launch (detached):
#   cd experiments/rubato-ppo-quantile && nohup bash sweep.sh > sweep.out 2>&1 &
#
# All stock knobs pass through (SEEDS, TASKS, SOLVERS, RESUME, SKIP_UPDATE, ...).
# The fixed arm is physically identical to rubato-ppo's r1 fixed runs (the quantile
# stop touches only the adaptive solver); to save ~half the compute and compare
# against the historical fixed groups instead:
#   SOLVERS="mujoco-adaptive" nohup bash sweep.sh > sweep.out 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT=${PROJECT:-rubato-ppo-quantile}
export RUN_TAG=${RUN_TAG:-q1}
export SWEEP_DIR=${SWEEP_DIR:-$HERE}   # own status/, joblogs/, summary.tsv, .lock
exec bash "$HERE/../rubato-ppo-sweep/sweep.sh"
