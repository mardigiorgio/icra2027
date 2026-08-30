#!/usr/bin/env bash
# rubato-ppo sweep: fixed vs adaptive MuJoCo-Warp PPO across the Newton-supported tasks.
#
# For every task x seed it trains BOTH solvers back-to-back (mujoco, mujoco-adaptive),
# so even a partially finished sweep yields paired comparisons. Logs to W&B:
#   project: rubato-ppo
#   group:   <task-slug>-<solver>          (group-mean = seed-averaged curve per solver)
#   run:     <task-slug>-<solver>-s<seed>-<tag>   (rsl_rl prefixes its timestamp)
# Videos are OFF by default (VIDEO=1 re-enables): the 2026-07-24 nvidia driver upgrade
# broke the GL recorder mid-sweep and killed 18 runs at first frame capture. Adaptive
# runs write per-run dt telemetry to joblogs/<run>.dt.log (survives reboots, unlike the
# old /tmp default), and ADAPTIVE_FLAGS passes NEWTON_ADAPTIVE_* switches through, e.g.
#   ADAPTIVE_FLAGS="NEWTON_ADAPTIVE_NAN_GUARD=1"
# Flags are echoed into each joblog launch header for provenance (they are env-only and
# would otherwise be invisible in the run's dumped params).
#
# Re-running the script skips runs that already finished with exit 0 (status/ dir); any
# run that is NOT complete is retrained fresh from iteration 0 -- no checkpoint resuming.
# Opt back into auto-resume (newest model_*.pt, budget capped to the stock remainder,
# continuing the same W&B run) with RESUME=1. ALWAYS launch detached so an SSH drop
# cannot kill it:
#   nohup bash sweep.sh >> sweep.out 2>&1 &
#
# Every run uses the task's stock config (num_envs, iterations, PPO params) untouched.
# Knobs (env vars): SEEDS, TASKS, SOLVERS, DEFER_TASKS, FINISH_NOW_TASKS, RUN_TAG,
# PROJECT, SKIP_UPDATE=1, RESUME=1, WANDB_MODE=offline, VIDEO=1, ADAPTIVE_FLAGS,
# ADAPTIVE_LOG_EVERY.
set -uo pipefail

# This script lives at <repo>/experiments/rubato-ppo-sweep/, so the repo root is
# derivable from its own location -- the repo may be named or moved freely.
_SWEEP_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUBATO_DIR=${RUBATO_DIR:-$_SWEEP_REPO_ROOT}
CODE_DIR=${CODE_DIR:-$(dirname "$_SWEEP_REPO_ROOT")}
ISAACLAB_DIR=${ISAACLAB_DIR:-$CODE_DIR/IsaacLab}
NEWTON_DIR=${NEWTON_DIR:-$CODE_DIR/newton-adaptive}

# Isaac Sim's one-time EULA prompt would stall a detached (nohup) sweep.
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}

PROJECT=${PROJECT:-rubato-ppo}
RUN_TAG=${RUN_TAG:-r1}
SEEDS=${SEEDS:-"42 43 44 45 46"}
SOLVERS=${SOLVERS:-"mujoco mujoco-adaptive"}
SKIP_UPDATE=${SKIP_UPDATE:-0}
SWEEP_DIR=${SWEEP_DIR:-$RUBATO_DIR/experiments/rubato-ppo-sweep}
VIDEO=${VIDEO:-0}
ADAPTIVE_FLAGS=${ADAPTIVE_FLAGS:-}
ADAPTIVE_LOG_EVERY=${ADAPTIVE_LOG_EVERY:-120}

# High-iteration scenes: excluded from the seed-major main passes and run in a final end
# pass (all seeds x both solvers) only after every seed of the other tasks has finished.
# Exception ("finish what's started"): a deferred task listed in FINISH_NOW_TASKS that
# already has a checkpoint resumes in the main pass instead of waiting -- currently just
# repose, so the half-done s42 run completes first. DEFER_TASKS="" disables deferral.
DEFER_TASKS=${DEFER_TASKS:-"Isaac-Reorient-Cube-Allegro-Direct Isaac-Lift-KukaAllegro"}
FINISH_NOW_TASKS=${FINISH_NOW_TASKS:-"Isaac-Reorient-Cube-Allegro-Direct"}

# Newton-supported task list, current origin/develop ids (old-doc names in comments).
# Ordered cheap -> expensive so early data lands fast; the two dexterous tasks dominate
# wall time (Reorient-Allegro 5k iters, Lift-KukaAllegro 15k iters by default).
# Not run: Isaac-Cartpole-Camera-Direct (vision PPO, same physics as state cartpole),
# G1-v1 sim-to-real, the *-Warp-* experimental envs (no newton_mjwarp preset), and
# Isaac-Reach-Franka (dropped 2026-07-18: USD asset fetch failing on the box).
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
Isaac-Reorient-Cube-Allegro-Direct
Isaac-Lift-KukaAllegro
"}

die() { echo "[FATAL] $*" >&2; exit 1; }

[[ -x "$ISAACLAB_DIR/isaaclab.sh" ]] || die "isaaclab.sh not found at $ISAACLAB_DIR"
[[ -f "$RUBATO_DIR/.venv/bin/activate" ]] || die "rubato venv not found at $RUBATO_DIR/.venv"
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "[WARN] nvidia-smi not found"

# This is the FULL-SCALE sweep (stock env counts; the dexterous tasks are ~20 h/run on
# a 16 GB card). On the 4070 workstation use experiments/sweep-4070 instead; both
# wrong-directory launches so far happened there. Override with ALLOW_SMALL_GPU=1.
if [[ "${ALLOW_SMALL_GPU:-0}" != 1 ]] && nvidia-smi -L 2>/dev/null | grep -q "4070"; then
  die "refusing to run the full-scale sweep on a 4070 -- use experiments/sweep-4070 (or ALLOW_SMALL_GPU=1)"
fi

if [[ "$SKIP_UPDATE" != 1 ]]; then
  echo "[INFO] updating repos to latest origin"
  git -C "$ISAACLAB_DIR" fetch origin || die "fetch IsaacLab failed"
  git -C "$ISAACLAB_DIR" checkout develop || die "checkout IsaacLab develop failed"
  git -C "$ISAACLAB_DIR" merge --ff-only origin/develop \
    || die "IsaacLab develop diverged from origin/develop; reconcile manually or SKIP_UPDATE=1"
  git -C "$NEWTON_DIR" fetch origin || die "fetch newton-adaptive failed"
  git -C "$NEWTON_DIR" checkout dev || die "checkout newton-adaptive dev failed"
  git -C "$NEWTON_DIR" merge --ff-only origin/dev \
    || die "newton-adaptive dev diverged from origin/dev; reconcile manually or SKIP_UPDATE=1"
fi

# shellcheck disable=SC1091
source "$RUBATO_DIR/.venv/bin/activate" || die "venv activation failed"

if [[ "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]] \
   && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
  echo "[WARN] no W&B credentials found (WANDB_API_KEY / wandb login); runs will stall at login."
  echo "[WARN] either 'wandb login' first or rerun with WANDB_MODE=offline."
fi

# Cross-machine skip ledger: a run is done when a FINISHED W&B run by its name exists in
# $PROJECT (tools/wandb_done.py) -- so a sweep continued on another machine skips work
# this one already finished, with no status/ syncing. Offline mode: status/ alone.
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
  local prev_dir last_ckpt ckpt_it total_it wdir t0 rc mins
  local resume_args=() wandb_env=() cmd=() video_args=() adaptive_env=()

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

  # Default: never resume -- an incomplete run retrains from iteration 0. Opt into
  # auto-resume with RESUME=1: newest checkpoint, iteration budget capped to the stock
  # remainder (read from the crashed run's own dumped params), same W&B run when the
  # local wandb files identify it.
  if [[ "${RESUME:-0}" == 1 ]]; then
    prev_dir=$(ls -dt logs/rsl_rl/*/*_"$run_name" 2>/dev/null | head -n1)
    last_ckpt=""
    [[ -n "$prev_dir" ]] && last_ckpt=$(ls -v "$prev_dir"/model_*.pt 2>/dev/null | tail -n1)
    if [[ -n "$last_ckpt" ]]; then
      ckpt_it=$(basename "$last_ckpt"); ckpt_it=${ckpt_it//[^0-9]/}
      total_it=$(grep -rho 'max_iterations: *[0-9]*' "$prev_dir/params" 2>/dev/null \
                 | head -n1 | tr -dc 0-9)
      if [[ -n "$total_it" && "$ckpt_it" -ge "$total_it" ]]; then
        echo "[DONE] $run_name (checkpoint $ckpt_it >= $total_it; marking complete)"
        echo 0 > "$status_f"; n_skip=$((n_skip+1)); return 0
      fi
      resume_args=( --resume --load_run "$(basename "$prev_dir")"
                    --checkpoint "$(basename "$last_ckpt")" )
      if [[ -n "$total_it" ]]; then
        resume_args+=( --max_iterations $(( total_it - ckpt_it )) )
      else
        echo "[WARN] $run_name: stock max_iterations not found under $prev_dir/params; resuming uncapped"
      fi
      for wdir in $(ls -dt "$prev_dir"/wandb/run-* wandb/run-* 2>/dev/null); do
        if grep -qsF -- "$run_name" "$wdir/files/wandb-metadata.json"; then
          wandb_env=( WANDB_RESUME=allow "WANDB_RUN_ID=${wdir##*-}" ); break
        fi
      done
      echo "[RES ] $run_name <- $(basename "$prev_dir")/$(basename "$last_ckpt") (iter $ckpt_it${total_it:+/$total_it})${wandb_env[1]:+, W&B run ${wandb_env[1]#WANDB_RUN_ID=}}"
    fi
  fi

  [[ "$VIDEO" == 1 ]] && video_args=( --video )
  if [[ "$solver" == *adaptive* ]]; then
    adaptive_env=( "NEWTON_ADAPTIVE_LOG=$SWEEP_DIR/joblogs/${run_name}.dt.log"
                   "NEWTON_ADAPTIVE_LOG_EVERY=$ADAPTIVE_LOG_EVERY" )
    local kv
    for kv in $ADAPTIVE_FLAGS; do adaptive_env+=( "$kv" ); done
  fi

  cmd=( "$ISAACLAB_DIR/isaaclab.sh" train --rl_library rsl_rl
        --task "$task" --solver "$solver" --seed "$seed" "${video_args[@]}"
        --logger wandb --log_project_name "$PROJECT"
        --run_name "$run_name" --run_group "${slug}-${solver}"
        "${resume_args[@]}"
        physics=newton_mjwarp )

  echo "[RUN ] $(date +%F_%T) $run_name"
  t0=$(date +%s)
  echo "==== $(date +%F_%T) launch $run_name [video=$VIDEO adaptive_env: ${adaptive_env[*]:-none}] ====" >> "joblogs/${run_name}.log"
  env "${wandb_env[@]}" "${adaptive_env[@]}" "${cmd[@]}" >> "joblogs/${run_name}.log" 2>&1 9>&-
  rc=$?
  mins=$(( ($(date +%s) - t0) / 60 ))
  echo "$rc" > "$status_f"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$run_name" "$task" "$solver" "$seed" "$rc" "$mins" >> "$summary"
  if [[ $rc == 0 ]]; then
    echo "[PASS] $run_name (${mins}m)"; n_pass=$((n_pass+1))
  else
    echo "[FAIL] $run_name rc=$rc (${mins}m) -- tail of joblogs/${run_name}.log:"
    tail -n 15 "joblogs/${run_name}.log" | sed 's/^/    /'
    n_fail=$((n_fail+1))
  fi
}

defer_pad=" ${DEFER_TASKS//$'\n'/ } "
finish_pad=" ${FINISH_NOW_TASKS//$'\n'/ } "

# Main pass, seed-major: a full pass over every env at one seed before the next seed,
# so partial sweeps maximize env diversity rather than seed depth. Deferred tasks are
# pushed to the end pass unless they already have a checkpoint to finish off.
for seed in $SEEDS; do
  for task in $TASKS; do
    slug=${task#IsaacContrib-}; slug=${slug#Isaac-}; slug=${slug,,}
    for solver in $SOLVERS; do
      run_name="${slug}-${solver}-s${seed}-${RUN_TAG}"
      if [[ "$defer_pad" == *" $task "* ]]; then
        if [[ "$finish_pad" != *" $task "* ]] \
           || ! compgen -G "logs/rsl_rl/*/*_${run_name}/model_*.pt" >/dev/null; then
          echo "[LATER] $run_name (deferred to end pass)"
          continue
        fi
      fi
      run_one "$task" "$solver" "$seed"
    done
  done
done

if [[ -n "${DEFER_TASKS// /}" ]]; then
  echo
  echo "==== end pass: deferred tasks ===="
  for seed in $SEEDS; do
    for task in $DEFER_TASKS; do
      for solver in $SOLVERS; do
        run_one "$task" "$solver" "$seed"
      done
    done
  done
fi

echo
echo "==== sweep done: $n_pass passed, $n_fail failed, $n_skip skipped ===="
column -t -s $'\t' "$summary"
