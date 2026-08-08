#!/usr/bin/env bash
# sweep-4070: scaled-down fixed-vs-adaptive PPO sweep for a single 16 GB workstation.
#
# A separate framework from experiments/rubato-ppo-sweep (the full-scale Puget sweep):
# that script stays stock-config and untouched; this one exists precisely because a
# 4070 Ti SUPER cannot afford stock env counts (the dexterous tasks alone would be
# ~20 h/run). Results here are NOT comparable to the stock sweep -- different env
# counts change PPO's effective batch size -- so they live in their own W&B project,
# and only fixed-vs-adaptive WITHIN this project is a fair comparison.
#
# Per task x seed it trains BOTH solvers back-to-back, so partial sweeps still yield
# paired comparisons.
#
# CROSS-MACHINE RESUME: W&B is the completion ledger. A run is "done" when a FINISHED
# W&B run by its name exists in $PROJECT (wandb_done.py), so any machine that pulls
# this repo and launches will skip work any other machine already finished -- there is
# no state directory to sync. The local status/ dir is only a cache so re-runs on the
# same box skip without a network round-trip. Checkpoint resuming is deliberately not
# supported at this scale: runs are minutes, an interrupted run just retrains.
# (Offline mode falls back to the local cache alone.)
#
# Robustness, learned from the stock sweep's first run on this box:
#   * W&B preflight: if api.wandb.ai is unreachable from THIS process (e.g. launched
#     from a network-sandboxed shell), fall back to WANDB_MODE=offline instead of the
#     uploader retrying forever with training already finished. Sync later with:
#       wandb sync experiments/sweep-4070/wandb/offline-run-*
#   * Per-run timeout (RUN_TIMEOUT, default 3600 s): a hung run fails loudly with
#     rc=124 instead of silently blocking every run behind it.
#   * No repo auto-update: this box runs the local checkouts as-is.
#   * VRAM auto-skip: envs run at STOCK configs; a task+solver that OOMs is marked
#     unsupported in status/ (per-machine) and skipped for all remaining seeds.
#
# Launch detached so an SSH/terminal drop cannot kill it:
#   cd experiments/sweep-4070 && nohup bash sweep.sh >> sweep.out 2>&1 &
# Watch:   tail -f sweep.out          Kill:   pkill -TERM -f sweep-4070/sweep.sh
#
# Knobs (env vars): PROJECT, RUN_TAG, SEEDS, TASKS, SOLVERS, NUM_ENVS, MAX_ITERATIONS,
# RUN_TIMEOUT, WANDB_MODE, ADAPTIVE_LOG_EVERY.
set -uo pipefail

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUBATO_DIR=${RUBATO_DIR:-$_ROOT}
ISAACLAB_DIR=${ISAACLAB_DIR:-$(dirname "$_ROOT")/IsaacLab}
SWEEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}

PROJECT=${PROJECT:-rubato-4070}
RUN_TAG=${RUN_TAG:-s2}
SEEDS=${SEEDS:-"42 43 44"}
SOLVERS=${SOLVERS:-"mujoco mujoco-adaptive"}
# Stock env counts, truncated iterations: batch size and per-task hyperparameters stay
# faithful (rsl_rl is tuned for stock num_envs), so early curves are trustworthy and the
# fixed-vs-adaptive pairing is meaningful; the s1 attempt at NUM_ENVS=1024 starved the
# harder tasks of 4-8x their tuned batch and produced garbage curves. Empty = stock.
NUM_ENVS=${NUM_ENVS:-}
MAX_ITERATIONS=${MAX_ITERATIONS:-300}
RUN_TIMEOUT=${RUN_TIMEOUT:-3600}
ADAPTIVE_LOG_EVERY=${ADAPTIVE_LOG_EVERY:-120}

# Cheap -> expensive so early data lands fast; the dexterous pair last (reorient@8192
# adaptive is the dominant cost, ~75 min/run at 300 iters).
TASKS=${TASKS:-"
Isaac-Cartpole-Direct
Isaac-Cartpole
Isaac-Reach-UR10
Isaac-Ant-Direct
Isaac-Ant
Isaac-Humanoid-Direct
Isaac-Humanoid
Isaac-Velocity-Flat-AnymalD
IsaacContrib-Velocity-Flat-AnymalB
IsaacContrib-Velocity-Flat-AnymalC
IsaacContrib-Velocity-Flat-UnitreeA1
IsaacContrib-Velocity-Flat-UnitreeGo1
Isaac-Velocity-Flat-UnitreeGo2
Isaac-Velocity-Flat-G1-v0
Isaac-Velocity-Flat-H1
"}
# Dexterous tasks removed from this box's default roster (2026-08-08): at 300 iterations
# they cannot produce learning curves (stock budgets are 5000/15000), and their
# throughput + stability signal has been banked -- including reorient-adaptive's 3/3
# NaN failure at stock scale, which graduated to a dedicated diagnostic. Re-add via
# TASKS= if a cell is ever needed.

die() { echo "[FATAL] $*" >&2; exit 1; }

[[ -x "$ISAACLAB_DIR/isaaclab.sh" ]] || die "isaaclab.sh not found at $ISAACLAB_DIR"
[[ -f "$RUBATO_DIR/.venv/bin/activate" ]] || die "rubato venv not found at $RUBATO_DIR/.venv"
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "[WARN] nvidia-smi not found"
# shellcheck disable=SC1091
source "$RUBATO_DIR/.venv/bin/activate" || die "venv activation failed"

# W&B preflight: training must never hang on an uploader that cannot reach the API.
if [[ "${WANDB_MODE:-online}" == "online" ]]; then
  if ! curl -sI --max-time 5 https://api.wandb.ai >/dev/null 2>&1; then
    echo "[WARN] api.wandb.ai unreachable from this process -- forcing WANDB_MODE=offline."
    echo "[WARN] Sync later with: wandb sync $SWEEP_DIR/wandb/offline-run-*"
    export WANDB_MODE=offline
  elif [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    echo "[WARN] no W&B credentials -- forcing WANDB_MODE=offline (wandb login to fix)."
    export WANDB_MODE=offline
  fi
fi
# The cross-machine ledger needs the API; offline mode falls back to the local cache.
WANDB_LEDGER=0
[[ "${WANDB_MODE:-online}" == "online" ]] && WANDB_LEDGER=1

# Single-instance guard: a second launch in the same sweep dir raced the first one's
# in-flight run (the W&B ledger only knows FINISHED runs), doubling it on one GPU.
exec 9>"$SWEEP_DIR/.lock"
flock -n 9 || die "another sweep instance holds $SWEEP_DIR/.lock -- watch: tail -f $SWEEP_DIR/sweep.out ; stop it AND its children: fuser -k -TERM $SWEEP_DIR/.lock"

mkdir -p "$SWEEP_DIR"/{status,joblogs}
cd "$SWEEP_DIR" || die "cannot cd to $SWEEP_DIR"
summary="$SWEEP_DIR/summary.tsv"
[[ -f "$summary" ]] || printf "run\ttask\tsolver\tseed\trc\tminutes\n" > "$summary"

n_pass=0 n_fail=0 n_skip=0

run_one() {
  local task=$1 solver=$2 seed=$3
  local slug=${task#IsaacContrib-}; slug=${slug#Isaac-}; slug=${slug,,}
  local run_name="${slug}-${solver}-s${seed}-${RUN_TAG}"
  local status_f="status/${run_name}.exit"
  local unsupported_f="status/UNSUPPORTED.${slug}-${solver}"
  local t0 rc mins adaptive_env=() scale_args=()
  [[ -n "$NUM_ENVS" ]] && scale_args+=( --num_envs "$NUM_ENVS" )
  [[ -n "$MAX_ITERATIONS" ]] && scale_args+=( --max_iterations "$MAX_ITERATIONS" )

  # Skip tier 0: this task+solver already OOM'd at stock config on THIS GPU. The marker
  # is per-machine (status/ is gitignored) because VRAM capability is per-machine.
  if [[ -f "$unsupported_f" ]]; then
    echo "[SKIP] $run_name (unsupported on this GPU: $(cat "$unsupported_f"))"
    n_skip=$((n_skip+1)); return 0
  fi

  # The cloud ledger is authoritative when reachable: a run DELETED from W&B must
  # re-run even if a local status file says done -- the cache must never outlive the
  # ledger. Local status is consulted only when the ledger is offline/unreachable.
  if [[ "$WANDB_LEDGER" == 1 ]]; then
    "$RUBATO_DIR/.venv/bin/python" "$RUBATO_DIR/tools/wandb_done.py" "$PROJECT" "$run_name" >/dev/null 2>&1
    case $? in
      0) echo "[SKIP] $run_name (done: W&B ledger)"; echo 0 > "$status_f"
         n_skip=$((n_skip+1)); return 0 ;;
      1) if [[ -f "$status_f" && "$(cat "$status_f")" == 0 ]]; then
           echo "[STALE] $run_name: local marker says done but no finished run in W&B (deleted?) -- re-running"
           rm -f "$status_f"
         fi ;;
      2) echo "[WARN] $run_name: W&B ledger query failed; trusting local status"
         if [[ -f "$status_f" && "$(cat "$status_f")" == 0 ]]; then
           echo "[SKIP] $run_name (done: local cache)"; n_skip=$((n_skip+1)); return 0
         fi ;;
    esac
  elif [[ -f "$status_f" && "$(cat "$status_f")" == 0 ]]; then
    echo "[SKIP] $run_name (done: local cache, offline mode)"; n_skip=$((n_skip+1)); return 0
  fi

  if [[ "$solver" == *adaptive* ]]; then
    adaptive_env=( "NEWTON_ADAPTIVE_LOG=$SWEEP_DIR/joblogs/${run_name}.dt.log"
                   "NEWTON_ADAPTIVE_LOG_EVERY=$ADAPTIVE_LOG_EVERY" )
  fi

  echo "[RUN ] $(date +%F_%T) $run_name"
  t0=$(date +%s)
  echo "==== $(date +%F_%T) launch $run_name [envs=${NUM_ENVS:-stock} iters=${MAX_ITERATIONS:-stock} wandb=${WANDB_MODE:-online}] ====" \
    >> "joblogs/${run_name}.log"
  env "${adaptive_env[@]}" timeout --signal=TERM --kill-after=60 "$RUN_TIMEOUT" \
      "$ISAACLAB_DIR/isaaclab.sh" train --rl_library rsl_rl \
      --task "$task" --solver "$solver" --seed "$seed" "${scale_args[@]}" \
      --logger wandb --log_project_name "$PROJECT" \
      --run_name "$run_name" --run_group "${slug}-${solver}" \
      physics=newton_mjwarp >> "joblogs/${run_name}.log" 2>&1 9>&-
  rc=$?
  mins=$(( ($(date +%s) - t0) / 60 ))
  echo "$rc" > "$status_f"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$run_name" "$task" "$solver" "$seed" "$rc" "$mins" >> "$summary"
  if [[ $rc == 0 ]]; then
    echo "[PASS] $run_name (${mins}m)"; n_pass=$((n_pass+1))
  else
    [[ $rc == 124 ]] && echo "[HUNG] $run_name exceeded RUN_TIMEOUT=${RUN_TIMEOUT}s"
    if tail -n 200 "joblogs/${run_name}.log" \
       | grep -qiE "out of memory|cudaErrorMemoryAllocation|CUDA error 2|Failed to allocate [0-9]+ bytes"; then
      echo "GPU OOM at stock config, $(date +%F_%T)" > "$unsupported_f"
      echo "[VRAM] $run_name: stock config exceeds this GPU -- skipping ${slug}-${solver} for remaining seeds"
    fi
    echo "[FAIL] $run_name rc=$rc (${mins}m) -- tail of joblogs/${run_name}.log:"
    tail -n 12 "joblogs/${run_name}.log" | sed 's/^/    /'
    n_fail=$((n_fail+1))
  fi
}

# Seed-major: a full pass over every task at one seed before the next seed, so a
# partial sweep maximizes task diversity rather than seed depth.
for seed in $SEEDS; do
  for task in $TASKS; do
    for solver in $SOLVERS; do
      run_one "$task" "$solver" "$seed"
    done
  done
done

echo
echo "==== sweep done: $n_pass passed, $n_fail failed, $n_skip skipped ===="
column -t -s $'\t' "$summary"
