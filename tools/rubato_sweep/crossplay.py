"""The 2x2 cross-play evaluation: every policy replayed under every physics.

THE EXPERIMENT. Train one policy under each timestepper for the SAME number of
iterations, then evaluate each policy under BOTH timesteppers. Four cells per
replicate:

                        replayed under
                    fixed          adaptive
    trained  fixed   R_ff            R_fa
    under  adaptive  R_af            R_aa

A policy that learned to exploit a fixed-step artifact scores well at R_ff and
collapses at R_fa, because the states it was farming do not survive an error-
controlled step. A policy that learned the task holds up in both. That
ASYMMETRY -- not the training curves, which the campaign has repeatedly found
to be within their own seed spread -- is what the claim rests on, and it does
not require the fixed arm to visibly fail during training.

WHY THE PRIOR NULL DOES NOT SETTLE IT. Pass 30 ran exactly this cross at 300
iterations and both policies transferred without degrading. At 300 iterations
neither policy had learned much (pass 30's own videos show a crude tip-and-
hoist on both arms), so there was nothing for an exploit to be made of. That is
a null at a horizon too short to matter. This protocol runs at the horizon where
the adaptive arm's reward has stopped improving, which is the earliest point at
which "the policy had time to find the exploit" is true.

WHAT EACH CELL PRODUCES: a per-step trace (the artifact signatures), a mean
episodic return under one identical protocol, and a video. The video is not a
courtesy -- the exploit is usually visible, and the campaign's standing rule is
that a rendered rollout is watched before a run is judged.

REPLICATES. The claim is about TRAINING, so a replicate is a training SEED, and
the evaluation seed is held fixed across all cells so the comparison is paired.
:func:`rubato_sweep.artifact.crossplay_verdict` refuses to report a difference
from fewer than two training seeds per arm.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from . import artifact, gpu, runner
from .config import ConfigError, load_mapping

PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifact_probe.py")


@dataclass
class PolicySpec:
    """Where one arm's trained policies live, and which iteration to take.

    ``log_globs`` is a LIST because an arm's replicates do not have to come from
    one sweep: the adaptive arm's seed-42 policy at iteration N is a checkpoint of
    the long run already in flight, while its second seed comes from a run made
    for this protocol. Every pattern is searched and the best match wins.
    """

    arm: str
    log_globs: tuple[str, ...]
    target_iter: int


@dataclass
class CrossplayCfg:
    name: str
    task: str
    out_dir: str
    policies: dict[str, PolicySpec]
    physics: tuple[str, ...] = ("fixed", "adaptive")
    seeds: tuple[int, ...] = (42, 7)
    num_envs: int = 256
    steps: int = 600
    eval_seed: int = 1234
    video: bool = True
    video_length: int = 300
    # If rendering at ``num_envs`` proves too heavy, lower num_envs for the WHOLE
    # cross-play rather than rendering a separate, differently-sized rollout: the
    # video has to be the run the numbers came from, or it cannot be used to
    # check them.
    checkpoint_tolerance: int = 50
    isaaclab_dir: str = os.path.expanduser("~/Documents/code/IsaacLab")
    venv_dir: str = os.path.expanduser("~/Documents/code/IsaacLabRubato/.venv")
    experiment_name: str = "trossen_spatula_lift"
    timeout_s: int = 3600
    source: str = "<memory>"
    extra_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str) -> CrossplayCfg:
        data = load_mapping(path)
        pol = data.get("policies")
        if not isinstance(pol, dict) or not pol:
            raise ConfigError(f"{path}: 'policies' must map arm -> {{log_glob, target_iter}}")
        policies = {}
        for arm, spec in pol.items():
            globs = spec.get("log_globs", spec.get("log_glob"))
            if isinstance(globs, str):
                globs = [globs]
            if not globs:
                raise ConfigError(f"{path}: policies.{arm} needs log_globs")
            policies[arm] = PolicySpec(
                arm=arm, log_globs=tuple(str(g) for g in globs), target_iter=int(spec["target_iter"])
            )
        known = {f for f in cls.__dataclass_fields__} - {"policies", "source"}
        kwargs: dict[str, Any] = {k: v for k, v in data.items() if k in known}
        for key in ("physics", "seeds"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        if "out_dir" in kwargs:
            kwargs["out_dir"] = os.path.expanduser(str(kwargs["out_dir"]))
        for key in ("isaaclab_dir", "venv_dir"):
            if key in kwargs:
                kwargs[key] = os.path.expanduser(str(kwargs[key]))
        return cls(policies=policies, source=os.path.abspath(path), **kwargs)

    def log_root(self) -> str:
        return os.path.join(self.isaaclab_dir, "logs", "rsl_rl", self.experiment_name)

    def cells(self) -> list[tuple[str, str, int]]:
        """(policy_arm, physics_arm, seed) in execution order.

        Physics is the innermost axis so a kill lands with a policy's BOTH
        physics cells done -- the retention ratio is per-policy and per-seed, and
        half of one is worth nothing.
        """
        return [
            (pol, phys, seed)
            for seed in self.seeds
            for pol in sorted(self.policies)
            for phys in self.physics
        ]


# --------------------------------------------------------------------------
# checkpoint resolution
# --------------------------------------------------------------------------


def resolve_checkpoint(cfg: CrossplayCfg, arm: str, seed: int) -> dict[str, Any]:
    """Find the checkpoint closest to (and not past) the matched iteration.

    Two facts make this fiddly and both are recorded rather than smoothed over:
      * rsl_rl saves periodically at ``save_interval`` AND once at the end, so a
        run stopped at N has ``model_{N-1}.pt`` while a longer run passing
        through N has ``model_{k*save_interval}.pt``. The two arms therefore do
        not in general carry the same index.
      * taking a LATER checkpoint for one arm would hand it extra training. The
        resolver never does: it takes the largest index <= target, and reports
        the shortfall so the mismatch is in the record.
    """
    spec = cfg.policies[arm]
    patterns = [
        os.path.join(cfg.log_root(), g.replace("{seed}", str(seed)).replace("{N}", str(spec.target_iter)))
        for g in spec.log_globs
    ]
    dirs = sorted({d for p in patterns for d in glob.glob(p)}, key=os.path.getmtime, reverse=True)
    target = spec.target_iter
    best: dict[str, Any] | None = None
    for d in dirs:
        for path in glob.glob(os.path.join(d, "model_*.pt")):
            m = re.search(r"model_(\d+)\.pt$", path)
            if not m:
                continue
            it = int(m.group(1))
            if it > target - 1:
                continue
            if best is None or it > best["iteration"]:
                best = {"path": path, "iteration": it, "run_dir": d}
    if best is None:
        raise FileNotFoundError(
            f"no checkpoint at or below iteration {target - 1} for arm '{arm}' seed {seed} "
            f"under any of {patterns}. Nothing to evaluate."
        )
    best["target_iteration"] = target
    best["shortfall"] = target - 1 - best["iteration"]
    if best["shortfall"] > cfg.checkpoint_tolerance:
        raise RuntimeError(
            f"arm '{arm}' seed {seed}: nearest checkpoint is iteration {best['iteration']}, "
            f"{best['shortfall']} short of the matched horizon {target - 1}. That is more than the "
            f"declared tolerance of {cfg.checkpoint_tolerance}; the arms would not be matched."
        )
    return best


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


def _env(cfg: CrossplayCfg, extra: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env["VIRTUAL_ENV"] = cfg.venv_dir
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env.update(cfg.extra_env)
    env.update(extra)
    return env


def _launch(cfg: CrossplayCfg, tag: str, extra: dict[str, str], journal: runner.Journal) -> int:
    log_path = os.path.join(cfg.out_dir, f"{tag}.log")
    cmd = [os.path.join(cfg.isaaclab_dir, "isaaclab.sh"), "-p", PROBE]
    if extra.get("RA_VIDEO") == "1":
        cmd += ["--viz", "newton"]
    journal.boundary("cell_start", cell=tag, cmd=shlex.join(cmd), env={k: v for k, v in extra.items()})
    gpu.wait_for_capacity(slots=1, own_pids=set(), per_run_gb=12.0, headroom_gb=4.0, timeout_s=cfg.timeout_s)
    started = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(
            ["timeout", "--signal=TERM", "--kill-after=60", str(cfg.timeout_s), *cmd],
            cwd=cfg.isaaclab_dir,
            env=_env(cfg, extra),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    journal.boundary("cell_exit", cell=tag, rc=proc.returncode, wall_s=round(time.time() - started, 1))
    gpu.wait_until_clear(own_pids=set(), timeout_s=180)
    return proc.returncode


def run_baselines(cfg: CrossplayCfg, journal: runner.Journal, echo=print) -> dict[str, Any]:
    """Measure, per physics arm, every reference the signatures are judged against.

    This runs FIRST and its failure is fatal: a signature with no baseline cannot
    produce a verdict, and discovering that after the expensive cells is how a
    pass ends with traces nobody can judge.
    """
    out: dict[str, Any] = {}
    for phys in cfg.physics:
        stem = os.path.join(cfg.out_dir, f"baseline_{phys}")
        if os.path.exists(f"{stem}.json"):
            echo(f"baseline {phys}: already present, skipping")
            with open(f"{stem}.json") as fh:
                out[phys] = json.load(fh)
            continue
        rc = _launch(
            cfg,
            f"baseline_{phys}",
            {
                "RA_MODE": "baseline",
                "RA_PHYSICS": phys,
                "RA_TASK": cfg.task,
                "RA_NENV": str(min(cfg.num_envs, 64)),
                "RA_SEED": str(cfg.eval_seed),
                "RA_OUT": stem,
            },
            journal,
        )
        if rc != 0 or not os.path.exists(f"{stem}.json"):
            raise RuntimeError(f"baseline probe for physics '{phys}' failed (rc={rc}); see {stem}.log")
        with open(f"{stem}.json") as fh:
            data = json.load(fh)
        if not data.get("ok"):
            raise RuntimeError(f"baseline probe for '{phys}' reported an error: {data.get('error')}")
        out[phys] = data
        echo(
            f"baseline {phys}: rest overlap {data.get('rest_penetration_m')} m, "
            f"free-fall |a+g| {data.get('freefall_accel_residual')} m/s^2, "
            f"wall {data.get('wall_thickness_m')} m"
        )
    return out


def run(cfg: CrossplayCfg, dry_run: bool = False, echo=print) -> dict[str, Any]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    journal = runner.Journal(os.path.join(cfg.out_dir, "journal.jsonl"), echo=echo)

    ckpts: dict[tuple[str, int], dict[str, Any]] = {}
    missing: list[str] = []
    for seed in cfg.seeds:
        for arm in sorted(cfg.policies):
            try:
                ckpts[(arm, seed)] = resolve_checkpoint(cfg, arm, seed)
            except (FileNotFoundError, RuntimeError) as exc:
                # A PLAN must never crash: its whole job is to say what is not
                # ready yet, and "the fixed arm has not been trained" is the
                # expected state the first time this is run.
                if not dry_run:
                    raise
                missing.append(f"  {arm:>9} s{seed:<4} NOT READY: {exc}")
    echo("\nresolved policies (largest checkpoint at or below the matched horizon):")
    for (arm, seed), c in sorted(ckpts.items()):
        echo(f"  {arm:>9} s{seed:<4} iter {c['iteration']:>5} (target {c['target_iteration'] - 1}, "
             f"shortfall {c['shortfall']})  {c['path']}")
    for line in missing:
        echo(line)

    cells = cfg.cells()
    echo(f"\n{len(cells)} cell(s) + {len(cfg.physics)} baseline(s); eval seed {cfg.eval_seed} "
         f"held fixed so cells are paired")
    if dry_run:
        for pol, phys, seed in cells:
            echo(f"  {pol}_s{seed}_on_{phys}")
        return {"cells": cells, "checkpoints": {f"{k[0]}_s{k[1]}": v for k, v in ckpts.items()}}

    journal.write("crossplay_start", name=cfg.name, config=cfg.source, cells=len(cells),
                  stack=runner.stack_fingerprint(_sweep_shim(cfg)))
    run_baselines(cfg, journal, echo=echo)

    for pol, phys, seed in cells:
        tag = f"{pol}_s{seed}_on_{phys}"
        stem = os.path.join(cfg.out_dir, tag)
        if os.path.exists(f"{stem}.trace.npz"):
            journal.write("skip_complete", cell=tag)
            continue
        extra = {
            "RA_MODE": "rollout",
            "RA_PHYSICS": phys,
            "RA_POLICY": pol,
            "RA_CKPT": ckpts[(pol, seed)]["path"],
            "RA_TASK": cfg.task,
            "RA_NENV": str(cfg.num_envs),
            "RA_STEPS": str(cfg.steps),
            "RA_SEED": str(cfg.eval_seed),
            "RA_OUT": stem,
        }
        if cfg.video:
            extra.update({
                "RA_VIDEO": "1",
                "RA_VIDEO_DIR": os.path.join(cfg.out_dir, "videos"),
                "RA_VIDEO_LEN": str(cfg.video_length),
            })
        rc = _launch(cfg, tag, extra, journal)
        if rc != 0 or not os.path.exists(f"{stem}.trace.npz"):
            raise RuntimeError(f"cell {tag} failed (rc={rc}); see {stem}.log")

    journal.write("crossplay_end", cells=len(cells))
    with open(os.path.join(cfg.out_dir, "checkpoints.json"), "w") as fh:
        json.dump({f"{k[0]}_s{k[1]}": v for k, v in ckpts.items()}, fh, indent=1)
    return artifact.report(cfg.out_dir, echo=echo)


def _sweep_shim(cfg: CrossplayCfg):
    """Minimal object exposing what ``runner.stack_fingerprint`` reads."""

    class _S:
        isaaclab_dir = cfg.isaaclab_dir

    return _S()


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------


def wall_estimate(n_iterations: int, seeds: int, s_per_iter_fixed: float = 8.067, cell_s: float = 120.0,
                  physics: int = 2, policies: int = 2) -> dict[str, float]:
    """Cost of the matched-N protocol, as a function of N.

    MEASURED inputs (pass 31, n=3, 1024 envs, exclusive device): fixed 8.067
    s/iter. The evaluation cell cost is PROJECTED from pass 31's 88 s/cell for
    the numeric-only cross-eval, inflated for the per-step trace and the video.
    The adaptive arm's training is NOT priced here: its policy at iteration N is
    taken from the run that is already in flight.
    """
    train_h = seeds * n_iterations * s_per_iter_fixed / 3600.0
    eval_h = (seeds * policies * physics + physics) * cell_s / 3600.0
    return {
        "fixed_training_h": train_h,
        "crossplay_h": eval_h,
        "total_h": train_h + eval_h,
        "iterations": n_iterations,
        "seeds": seeds,
    }
