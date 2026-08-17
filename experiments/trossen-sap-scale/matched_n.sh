#!/usr/bin/env bash
# THE MATCHED-N PROTOCOL. One command, one argument: N, the plateau iteration.
#
# WHAT IT DOES, in order, and why that order:
#   1. PARITY PREFLIGHT on both arms, live. The arms' tolerance triple, contact
#      capacity, contact law, determinism and containment wiring were equalized
#      in pass 29; this re-verifies it on the stack that is about to spend GPU
#      hours, and ABORTS on any difference outside the intended ones. Nothing is
#      trained until it passes.
#   2. FIXED-ARM TRAINING at exactly N iterations, one run per seed, with the
#      adaptive arm's env count, seed, config and video cadence.
#   3. ADAPTIVE FILL. The adaptive policy at iteration N for the seed already
#      running as the main 4000-iteration run is a CHECKPOINT of that run and
#      costs nothing. Any OTHER seed has to be trained, and $FILL_SEEDS says
#      which. Empty by default.
#   4. THE 2x2 CROSS-PLAY. Each policy replayed under BOTH physics arms, with a
#      per-step trace and a video per cell.
#   5. THE ARTIFACT ANALYSIS. Signatures, thresholds, and the verdict.
#
# THE DEFAULT ACTION IS `plan`, WHICH TOUCHES NO GPU. A run that is currently
# training must not be disturbed by a fat-fingered invocation of this script.
#
# Usage:
#   matched_n.sh <N> [SEEDS] [FILL_SEEDS] [plan|run|crossplay|analyze]
#     matched_n.sh 1200                        plan, no GPU
#     matched_n.sh 1200 42,7 7 run             the whole protocol, n=2 per arm
#     matched_n.sh 1200 42 "" run              n=1: runs, but the verdict will be
#                                              UNRESOLVED by design
#     matched_n.sh 1200 42,7 7 crossplay       skip training, evaluate what exists
#     matched_n.sh 1200 42,7 7 analyze         CPU only: re-judge existing traces
#
# Launch detached so an SSH drop cannot kill it:
#   cd experiments/trossen-sap-scale && nohup bash matched_n.sh 1200 42,7 7 run >> matched_n.out 2>&1 &
# Watch:  tail -f matched_n.out
# Stop it AND its children:  fuser -k -TERM runs-p34-n<N>/.lock
set -uo pipefail

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$_ROOT/.venv/bin/python}"
SWEEP="$_ROOT/tools/sweep.py"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

N="${1:-}"
SEEDS="${2:-42,7}"
FILL_SEEDS="${3:-}"
ACTION="${4:-plan}"

[[ -n "$N" ]] || { echo "[FATAL] N (the matched iteration count) is required" >&2; exit 2; }
[[ "$N" =~ ^[0-9]+$ ]] || { echo "[FATAL] N must be an integer, got '$N'" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "[FATAL] no interpreter at $PY" >&2; exit 2; }

yaml_list () { printf '[%s]' "$(echo "$1" | tr -d ' ')"; }

# N IS SNAPPED so both arms land on the SAME checkpoint index.
#
# rsl_rl saves every save_interval=50 iterations AND once at the end. A run
# stopped at N therefore ends on model_$((N-1)).pt, while the long adaptive run
# passing through N only has model_{k*50}.pt. Those indices coincide exactly
# when N-1 is a multiple of 50, so N is snapped DOWN to the nearest k*50+1 and
# the checkpoint tolerance is 0: the two policies are then trained for exactly
# the same number of iterations, which is the whole premise of the comparison.
# Snapping down costs at most 49 iterations of a horizon in the thousands.
# Set NOSNAP=1 to keep N verbatim and accept a mismatch the resolver will report.
N_REQ="$N"
if [[ "${NOSNAP:-0}" != "1" ]]; then
  N=$(( ((N - 1) / 50) * 50 + 1 ))
  (( N < 1 )) && N=1
  if [[ "$N" != "$N_REQ" ]]; then
    echo "[snap] N $N_REQ -> $N so both arms share checkpoint index $((N-1)) exactly"
  fi
fi

# Named AFTER the snap, so the directory and the configs inside it agree.
OUT="$HERE/runs-p34-n$N"
CFG_DIR="$OUT/configs"
mkdir -p "$CFG_DIR"

cat > "$CFG_DIR/p34_parity.yaml" <<YAML
# Parity preflight ONLY. Both arms are declared so their resolved identity can be
# dumped and diffed; no cell from this file is ever executed by the driver.
name: p34-parity
task: IsaacContrib-Lift-Spatula-Trossen-v0
project: rubato-trossen
out_dir: $OUT
num_steps_per_env: 24
decimation: 4
timing_sensitive: true
video: {enabled: false}
arms:
  fixed: {solver: sap}
  adaptive:
    solver: sap-adaptive
    extra_args: ["physics=newton_mjwarp_adaptive"]
axes:
  num_envs: [1024]
  iterations: [$N]
  seeds: [42]
preflight:
  enabled: true
  num_envs: 8
  abort_on_diff: true
  expected_differences: []
YAML

cat > "$CFG_DIR/p34_fixed.yaml" <<YAML
# The fixed arm at exactly N iterations, matched to the adaptive run in env
# count, seed, horizon, video cadence and every solver-adjacent setting the
# parity preflight checks. Preflight is disabled HERE because this file declares
# one arm, so its diff would be trivially empty; the two-arm check is run by the
# driver immediately before this and aborts the whole protocol on a difference.
name: p34-matched
task: IsaacContrib-Lift-Spatula-Trossen-v0
project: rubato-trossen
out_dir: $OUT
num_steps_per_env: 24
decimation: 4
timing_sensitive: true
timeout_s: 86400
startup_abort_s: 900
video:
  enabled: true
  every_iterations: 50
  length: 200
arms:
  fixed: {solver: sap}
axes:
  num_envs: [1024]
  iterations: [$N]
  seeds: $(yaml_list "$SEEDS")
preflight: {enabled: false}
YAML

if [[ -n "$FILL_SEEDS" ]]; then
cat > "$CFG_DIR/p34_fill_adaptive.yaml" <<YAML
# Adaptive replicates that do NOT already exist as a checkpoint of the long run.
# The main 4000-iteration run supplies seed 42 for free; every other seed has to
# be paid for, and this file is what pays for it.
name: p34-matched
task: IsaacContrib-Lift-Spatula-Trossen-v0
project: rubato-trossen
out_dir: $OUT
num_steps_per_env: 24
decimation: 4
timing_sensitive: true
timeout_s: 172800
startup_abort_s: 900
video:
  enabled: true
  every_iterations: 50
  length: 200
arms:
  adaptive:
    solver: sap-adaptive
    extra_args: ["physics=newton_mjwarp_adaptive"]
axes:
  num_envs: [1024]
  iterations: [$N]
  seeds: $(yaml_list "$FILL_SEEDS")
preflight: {enabled: false}
YAML
fi

cat > "$CFG_DIR/p34_crossplay.yaml" <<YAML
# The 2x2. Rows are the arm a policy TRAINED under, columns the arm it is
# REPLAYED under. The evaluation seed is held fixed across all cells so the
# comparison is paired; the replicate axis is the TRAINING seed, because the
# claim is about training.
name: p34-crossplay-n$N
task: IsaacContrib-Lift-Spatula-Trossen-v0
out_dir: $OUT
num_envs: 256
steps: 600
eval_seed: 1234
seeds: $(yaml_list "$SEEDS")
physics: [fixed, adaptive]
video: true
video_length: 300
checkpoint_tolerance: 0
policies:
  fixed:
    target_iter: $N
    log_globs: ["*p34-matched-fixed-1024x${N}-s{seed}"]
  adaptive:
    target_iter: $N
    log_globs:
      - "*main-sap-adaptive-1024x4000-s{seed}"
      - "*p34-matched-adaptive-1024x${N}-s{seed}"
YAML

echo "==== $(date +%F_%T) matched-N protocol: N=$N seeds=$SEEDS fill=$FILL_SEEDS action=$ACTION"
echo "==== configs rendered into $CFG_DIR"

nseeds="$(echo "$SEEDS" | tr ',' '\n' | grep -c .)"
"$PY" "$SWEEP" cost "$N" --seeds "$nseeds"
if [[ -n "$FILL_SEEDS" ]]; then
  nfill="$(echo "$FILL_SEEDS" | tr ',' '\n' | grep -c .)"
  echo "  adaptive fill       $(awk -v n="$nfill" -v it="$N" 'BEGIN{printf "%6.2f", n*it*18.207/3600}') h   ($nfill seed(s) x $N x 18.207 s/iter, MEASURED s/iter)"
fi
if (( nseeds < 2 )); then
  echo "[WARN] $nseeds training seed(s) per arm. The aggregator will report UNRESOLVED for every"
  echo "[WARN] between-arm difference: pass 30's same-seed restart diverged 2.4x in reward by"
  echo "[WARN] iteration 9, so one seed cannot separate a difference from the spread."
fi

case "$ACTION" in
  plan)
    "$PY" "$SWEEP" plan "$CFG_DIR/p34_fixed.yaml"
    [[ -n "$FILL_SEEDS" ]] && "$PY" "$SWEEP" plan "$CFG_DIR/p34_fill_adaptive.yaml"
    "$PY" "$SWEEP" crossplay "$CFG_DIR/p34_crossplay.yaml" --plan
    ;;
  run)
    echo "---- 1/5 parity preflight (both arms, live)"
    "$PY" "$SWEEP" preflight "$CFG_DIR/p34_parity.yaml" || { echo "[FATAL] parity preflight failed; nothing trained" >&2; exit 3; }
    echo "---- 2/5 fixed arm at $N iterations"
    "$PY" "$SWEEP" run "$CFG_DIR/p34_fixed.yaml" || { echo "[FATAL] fixed-arm training failed" >&2; exit 4; }
    if [[ -n "$FILL_SEEDS" ]]; then
      echo "---- 3/5 adaptive fill seeds $FILL_SEEDS at $N iterations"
      "$PY" "$SWEEP" run "$CFG_DIR/p34_fill_adaptive.yaml" || { echo "[FATAL] adaptive fill failed" >&2; exit 5; }
    else
      echo "---- 3/5 adaptive fill skipped (no FILL_SEEDS)"
    fi
    echo "---- 4/5 2x2 cross-play with video"
    "$PY" "$SWEEP" crossplay "$CFG_DIR/p34_crossplay.yaml" || { echo "[FATAL] cross-play failed" >&2; exit 6; }
    echo "---- 5/5 artifact analysis"
    "$PY" "$SWEEP" artifact "$OUT"
    ;;
  crossplay)
    "$PY" "$SWEEP" crossplay "$CFG_DIR/p34_crossplay.yaml"
    ;;
  analyze)
    "$PY" "$SWEEP" artifact "$OUT"
    ;;
  *)
    echo "[FATAL] unknown action '$ACTION'; use plan|run|crossplay|analyze" >&2
    exit 2
    ;;
esac
