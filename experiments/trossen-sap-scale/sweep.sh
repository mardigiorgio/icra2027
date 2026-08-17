#!/usr/bin/env bash
# trossen-sap-scale: the SAP fixed-vs-adaptive campaign's committed driver.
#
# A thin launcher over tools/rubato_sweep (the harness), in the same shape as
# experiments/rubato-ppo-quantile/sweep.sh: a campaign directory owns its
# configs and its run artifacts, and never edits the harness or another
# campaign's directory.
#
# Everything that used to live in throwaway scratchpad bash -- skip-if-complete,
# the interleaved arm order, the per-run timeout, the nvidia-smi census at every
# boundary, the one-process-at-a-time rule -- is in the harness now. What is
# left here is which config to run and how to launch it detached.
#
# ALWAYS launch detached so an SSH drop cannot kill the sweep:
#   cd experiments/trossen-sap-scale && nohup bash sweep.sh d7 >> sweep.out 2>&1 &
# Watch:  tail -f sweep.out
# Stop it AND its children:  fuser -k -TERM runs/.lock
#
# Usage: sweep.sh <config-stem> [plan|run|analyze|preflight]
#   sweep.sh d7             -> run  tools/rubato_sweep/configs/trossen_sap_d7.yaml
#   sweep.sh scale plan     -> dry-run the queued scale protocol (no GPU)
set -uo pipefail

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$_ROOT/.venv/bin/python}"
SWEEP="$_ROOT/tools/sweep.py"

declare -A CONFIGS=(
  [d7]="trossen_sap_d7.yaml"
  [scale]="p33_scale_confirm.yaml"
  [pack]="p33_packing_probe.yaml"
)

stem="${1:-d7}"
action="${2:-run}"
cfg="${CONFIGS[$stem]:-}"
[[ -n "$cfg" ]] || { echo "[FATAL] unknown config '$stem'; known: ${!CONFIGS[*]}" >&2; exit 2; }
cfg="$_ROOT/tools/rubato_sweep/configs/$cfg"
[[ -f "$cfg" ]] || { echo "[FATAL] missing config $cfg" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "[FATAL] no interpreter at $PY" >&2; exit 2; }

echo "==== $(date +%F_%T) $action $cfg ===="
exec "$PY" "$SWEEP" "$action" "$cfg"
