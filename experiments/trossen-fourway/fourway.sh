#!/usr/bin/env bash
# trossen-fourway: the PhysX / MuJoCo-fixed / SAP-fixed / SAP-adaptive campaign.
#
# A thin launcher over tools/rubato_sweep, in the same shape as
# experiments/trossen-sap-scale/sweep.sh. This directory owns its configs'
# outputs and never edits the harness or another campaign's directory.
#
# THE STAIRCASE. Run the stages in order and stop at each gate; the gates are
# where a human decides, not where the script does.
#
#   0. fourway.sh screen3 plan          # CPU only, no GPU touched
#      fourway.sh screen3               # 3 engines, 3 seeds, 200 iters  (~7 h)
#      GATE: all three arms run; s/iter measured under ONE window protocol for
#      the first time; failure-mode census in hand.
#
#   1. Marco's task-side PhysX changes land (see the pass-35 ledger entry).
#      fourway.sh screen                # adds the PhysX arm                (+?)
#      GATE: PhysX constructs, steps, and its s/iter is finally a number.
#
#   2. Read the finished main-sap-adaptive-1024x4000-s42 reward curve and pick
#      the smallest horizon whose slope is inside the seed-spread band. Edit
#      that horizon into p35_fourway_full.yaml. Then:
#      fourway.sh full plan             # re-price at the chosen horizon
#      fourway.sh full                  # the confirmation runs
#
#   3. fourway.sh cross plan            # resolves checkpoints, no GPU
#      fourway.sh cross                 # the 4x4 matrix + refinement column
#      fourway.sh judge                 # CPU-only re-judgement of the traces
#
# ALWAYS launch detached so an SSH drop cannot kill a multi-day sweep:
#   cd experiments/trossen-fourway && nohup bash fourway.sh screen3 >> fourway.out 2>&1 &
# Watch:  tail -f fourway.out
# Stop it AND its children:  fuser -k -TERM runs-screen3/.lock
#
# Usage: fourway.sh <stage> [action]
set -uo pipefail

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$_ROOT/.venv/bin/python}"
SWEEP="$_ROOT/tools/sweep.py"

declare -A CONFIGS=(
  [screen3]="p35_threeway_screen.yaml"
  [screen]="p35_fourway_screen.yaml"
  [full]="p35_fourway_full.yaml"
  [cross]="p35_fourway_crossplay.yaml"
  [judge]="p35_fourway_crossplay.yaml"
)
# Which sweep verb each stage takes by default, and which it takes for "plan".
declare -A DEFAULT_ACTION=( [screen3]="run" [screen]="run" [full]="run" [cross]="crossplay" [judge]="artifact" )
declare -A PLAN_ACTION=(    [screen3]="plan" [screen]="plan" [full]="plan" [cross]="crossplay" [judge]="artifact" )

stage="${1:-screen3}"
action="${2:-}"
cfg="${CONFIGS[$stage]:-}"
[[ -n "$cfg" ]] || { echo "[FATAL] unknown stage '$stage'; known: ${!CONFIGS[*]}" >&2; exit 2; }
cfg="$_ROOT/tools/rubato_sweep/configs/$cfg"
[[ -f "$cfg" ]] || { echo "[FATAL] missing config $cfg" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "[FATAL] no interpreter at $PY" >&2; exit 2; }

extra=()
if [[ -z "$action" ]]; then
  action="${DEFAULT_ACTION[$stage]}"
elif [[ "$action" == "plan" ]]; then
  action="${PLAN_ACTION[$stage]}"
  # crossplay's dry run is a flag, not a verb.
  [[ "$action" == "crossplay" ]] && extra=(--plan)
fi

echo "==== $(date +%F_%T) $action $cfg ${extra[*]} ===="
exec "$PY" "$SWEEP" "$action" "$cfg" "${extra[@]}"
