"""Sweep configuration: one file describes one experiment, the runner executes it.

The config is data, not code: every knob a run needs (arm, task, env count, seed,
horizon, solver env vars, video cadence, packing policy) is declared here so that
the same experiment can be re-launched, resumed, or handed to another machine
without reconstructing a command line by hand.

Two conversions are done HERE rather than by the operator, because getting them
wrong has cost this project GPU days:

  * VIDEO CADENCE. ``--video_interval`` counts ENV STEPS, not learning
    iterations. The config expresses the cadence in ITERATIONS and
    :meth:`VideoCfg.interval_env_steps` multiplies by ``num_steps_per_env``.
  * CELL ORDER. Arms are the innermost loop, so an interrupted sweep always
    ends on a complete matched pair rather than on N runs of one arm.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# A run whose wall time will be quoted as a measurement must have had the GPU to
# itself. Packing is therefore refused unless the config says, in as many words,
# that no timing from this sweep is load-bearing.
TIMING_SENSITIVE_DEFAULT = True


class ConfigError(ValueError):
    """The config file is unusable. Raised before any GPU work is started."""


def load_mapping(path: str) -> dict[str, Any]:
    """Read a .yaml/.yml/.json config into a plain dict."""
    with open(path) as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


@dataclass(frozen=True)
class Arm:
    """One physics configuration under comparison.

    ``solver`` is the ``--solver`` latch and is OPTIONAL, because it only exists
    for the Newton MuJoCo-Warp backend: the training entry point raises if it is
    passed against a physics cfg with no ``solver_cfg``, which is exactly the
    PhysX case. A PhysX arm therefore declares ``solver: null`` and selects its
    engine through ``extra_args: ["physics=physx"]`` alone.
    """

    name: str
    solver: str | None = None
    extra_args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    family: str = "newton"

    def cli_args(self) -> list[str]:
        if self.solver is None:
            return list(self.extra_args)
        return ["--solver", self.solver, *self.extra_args]


@dataclass(frozen=True)
class VideoCfg:
    enabled: bool = True
    every_iterations: int = 10
    length: int = 200

    def interval_env_steps(self, num_steps_per_env: int) -> int:
        """Convert an ITERATION cadence to the env-step count the CLI wants.

        rsl_rl collects ``num_steps_per_env`` env steps per learning iteration and
        the recorder's trigger counts env steps, so a cadence of N iterations is
        N * num_steps_per_env steps. Passing the iteration number directly makes
        the recorder fire every few iterations at 24x the intended rate.
        """
        if self.every_iterations < 1:
            raise ConfigError("video.every_iterations must be >= 1 (it is a cadence in iterations)")
        if num_steps_per_env < 1:
            raise ConfigError("num_steps_per_env must be >= 1")
        return self.every_iterations * num_steps_per_env


@dataclass(frozen=True)
class PackingCfg:
    """Opt-in concurrency for OUTCOME-ONLY sweeps.

    Concurrent runs contend for SMs and memory bandwidth, so every wall time they
    produce is contaminated. The runner refuses to enable this unless the sweep
    has declared itself not timing-sensitive, and tags every run with whether it
    ran packed so a packed wall can never be quoted as a measurement later.
    """

    enabled: bool = False
    per_run_gb: float = 12.0
    headroom_gb: float = 4.0
    max_concurrent: int = 2

    def slots(self, total_gb: float, used_by_others_gb: float) -> int:
        if not self.enabled:
            return 1
        free = total_gb - used_by_others_gb - self.headroom_gb
        if self.per_run_gb <= 0:
            raise ConfigError("packing.per_run_gb must be > 0")
        return max(1, min(self.max_concurrent, int(free // self.per_run_gb)))


@dataclass(frozen=True)
class PreflightCfg:
    """Parity dump taken before the sweep spends GPU hours.

    ``expected_differences`` lists the keys the arms are ALLOWED to differ in
    (for a fixed-vs-adaptive comparison, the solver class and the substep count
    are the intended difference). Any other difference aborts the sweep.

    ``by_family`` switches the check for a MULTI-ENGINE sweep, where the strict
    all-arms diff is the wrong question: PhysX and MuJoCo and SAP are meant to
    differ in every solver field, so demanding equality would either abort
    always or, once silenced, check nothing. Instead:

      * arms sharing a ``family`` are diffed strictly, as today -- that is where
        a real confound can hide, because those arms DO share a contact law;
      * across families only ``contract_keys`` must match. Those are the axes
        the comparison equalizes (the MDP, the control contract, the batch),
        and they are the only axes its claims are allowed to live on.

    A cross-family sweep with no ``contract_keys`` is refused: it would be a
    comparison with nothing held fixed.
    """

    enabled: bool = True
    num_envs: int = 8
    expected_differences: tuple[str, ...] = ()
    abort_on_diff: bool = True
    timeout_s: int = 1800
    by_family: bool = False
    contract_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class Cell:
    """One run: an arm at an env count, seed and horizon."""

    arm: Arm
    task: str
    num_envs: int
    seed: int
    iterations: int | None
    prefix: str

    @property
    def run_name(self) -> str:
        horizon = "stock" if self.iterations is None else str(self.iterations)
        return f"{self.prefix}-{self.arm.name}-{self.num_envs}x{horizon}-s{self.seed}"

    @property
    def pair_key(self) -> tuple[Any, ...]:
        """Cells sharing this key are the matched set an interrupted sweep must complete."""
        return (self.task, self.num_envs, self.iterations, self.seed)


@dataclass
class SweepCfg:
    name: str
    task: str
    num_steps_per_env: int
    decimation: int
    out_dir: str
    arms: tuple[Arm, ...]
    seeds: tuple[int, ...]
    env_counts: tuple[int, ...]
    horizons: tuple[int | None, ...]
    project: str = "rubato-trossen"
    isaaclab_dir: str = os.path.expanduser("~/Documents/code/IsaacLab")
    venv_dir: str = os.path.expanduser("~/Documents/code/IsaacLabRubato/.venv")
    rl_library: str = "rsl_rl"
    viz: str | None = "newton"
    timeout_s: int = 21600
    startup_abort_s: int = 600
    timing_sensitive: bool = TIMING_SENSITIVE_DEFAULT
    video: VideoCfg = field(default_factory=VideoCfg)
    packing: PackingCfg = field(default_factory=PackingCfg)
    preflight: PreflightCfg = field(default_factory=PreflightCfg)
    wandb_ledger: bool = False
    extra_args: tuple[str, ...] = ()
    source: str = "<memory>"

    # -- construction ------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> SweepCfg:
        return cls.from_mapping(load_mapping(path), source=os.path.abspath(path))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], source: str = "<memory>") -> SweepCfg:
        def req(key: str) -> Any:
            if key not in data:
                raise ConfigError(f"{source}: missing required key '{key}'")
            return data[key]

        arms_raw = req("arms")
        if not isinstance(arms_raw, Mapping) or not arms_raw:
            raise ConfigError(f"{source}: 'arms' must be a non-empty mapping of name -> arm spec")
        arms = tuple(
            Arm(
                name=name,
                solver=None if spec.get("solver") is None else str(spec["solver"]),
                extra_args=tuple(str(a) for a in spec.get("extra_args", ())),
                env={str(k): str(v) for k, v in dict(spec.get("env", {})).items()},
                family=str(spec.get("family", "newton")),
            )
            for name, spec in arms_raw.items()
        )

        axes = data.get("axes", {})
        horizons_raw = axes.get("iterations", data.get("iterations", [None]))
        if not isinstance(horizons_raw, Sequence) or isinstance(horizons_raw, str):
            horizons_raw = [horizons_raw]

        cfg = cls(
            name=str(req("name")),
            task=str(req("task")),
            num_steps_per_env=int(req("num_steps_per_env")),
            decimation=int(req("decimation")),
            out_dir=os.path.expanduser(str(req("out_dir"))),
            arms=arms,
            seeds=tuple(int(s) for s in axes.get("seeds", [42])),
            env_counts=tuple(int(n) for n in axes.get("num_envs", [1024])),
            horizons=tuple(None if h is None else int(h) for h in horizons_raw),
            project=str(data.get("project", "rubato-trossen")),
            isaaclab_dir=os.path.expanduser(str(data.get("isaaclab_dir", cls.isaaclab_dir))),
            venv_dir=os.path.expanduser(str(data.get("venv_dir", cls.venv_dir))),
            rl_library=str(data.get("rl_library", "rsl_rl")),
            viz=data.get("viz", "newton"),
            timeout_s=int(data.get("timeout_s", 21600)),
            startup_abort_s=int(data.get("startup_abort_s", 600)),
            timing_sensitive=bool(data.get("timing_sensitive", TIMING_SENSITIVE_DEFAULT)),
            video=VideoCfg(**dict(data.get("video", {}))),
            packing=PackingCfg(**dict(data.get("packing", {}))),
            preflight=PreflightCfg(
                **{
                    k: (tuple(v) if k in ("expected_differences", "contract_keys") else v)
                    for k, v in dict(data.get("preflight", {})).items()
                }
            ),
            wandb_ledger=bool(data.get("wandb_ledger", False)),
            extra_args=tuple(str(a) for a in data.get("extra_args", ())),
            source=source,
        )
        cfg.validate()
        return cfg

    # -- checks ------------------------------------------------------------

    def validate(self) -> None:
        if self.decimation < 1:
            raise ConfigError("decimation must be >= 1 (env steps are decimation solver boundaries)")
        if self.num_steps_per_env < 1:
            raise ConfigError("num_steps_per_env must be >= 1")
        if len(self.arms) != len({a.name for a in self.arms}):
            raise ConfigError("arm names must be unique")
        if self.packing.enabled and self.timing_sensitive:
            raise ConfigError(
                "packing.enabled requires timing_sensitive: false -- concurrent runs contaminate "
                "every wall time they produce, so a packed sweep may only be used for outcome-only "
                "cells. Set timing_sensitive: false explicitly if that is what this sweep is."
            )
        if self.packing.enabled and self.packing.max_concurrent < 2:
            raise ConfigError("packing.max_concurrent must be >= 2 when packing is enabled")
        if not self.seeds or not self.env_counts or not self.horizons:
            raise ConfigError("axes.seeds, axes.num_envs and axes.iterations must each be non-empty")
        for arm in self.arms:
            if arm.solver is None and not arm.extra_args:
                raise ConfigError(
                    f"arm '{arm.name}' selects no physics at all: with 'solver: null' it must name "
                    "its engine in extra_args (e.g. physics=physx). Otherwise it silently runs the "
                    "task's default preset under another arm's name, and the sweep compares one "
                    "engine with itself."
                )
        if self.preflight.enabled and len({a.family for a in self.arms}) > 1:
            if not self.preflight.by_family:
                raise ConfigError(
                    "arms span more than one physics family, so the strict all-arms preflight diff "
                    "would abort on differences that ARE the experiment. Set preflight.by_family: "
                    "true and list the axes this comparison equalizes in preflight.contract_keys."
                )
            if not self.preflight.contract_keys:
                raise ConfigError(
                    "preflight.by_family with no contract_keys checks nothing across engines. List "
                    "the axes this comparison equalizes -- the MDP terms, the action and observation "
                    "contract, the control rate, the batch -- because those are the only axes its "
                    "claims may live on."
                )

    # -- expansion ---------------------------------------------------------

    def cells(self) -> list[Cell]:
        """Expand the axes into the execution order.

        Arms innermost: a kill between cells leaves a complete matched pair
        (pass 30 lost an arm to a mid-run kill and had to restart it; pass 31
        interleaved for exactly this reason).
        """
        out: list[Cell] = []
        for n in self.env_counts:
            for h in self.horizons:
                for s in self.seeds:
                    for arm in self.arms:
                        out.append(
                            Cell(arm=arm, task=self.task, num_envs=n, seed=s, iterations=h, prefix=self.name)
                        )
        return out

    def matched_groups(self) -> Iterator[list[Cell]]:
        """Yield the cells of each matched set, in execution order."""
        group: list[Cell] = []
        key: tuple[Any, ...] | None = None
        for cell in self.cells():
            if key is not None and cell.pair_key != key:
                yield group
                group = []
            key = cell.pair_key
            group.append(cell)
        if group:
            yield group

    # -- derived -----------------------------------------------------------

    def video_interval_env_steps(self) -> int:
        return self.video.interval_env_steps(self.num_steps_per_env)

    def boundaries_per_iteration(self) -> int:
        """Solver boundaries per learning iteration, per world.

        One learning iteration collects ``num_steps_per_env`` env steps and each
        env step is ``decimation`` solver boundaries. Demand-normalized metrics
        must divide by this and never by ``num_steps_per_env`` alone -- mixing
        the two is a factor of ``decimation`` (pass 29 published that error).
        """
        return self.num_steps_per_env * self.decimation

    def log_path(self, cell: Cell) -> str:
        return os.path.join(self.out_dir, f"{cell.run_name}.log")

    def json_path(self, cell: Cell) -> str:
        return os.path.join(self.out_dir, f"{cell.run_name}.json")

    def journal_path(self) -> str:
        return os.path.join(self.out_dir, "journal.jsonl")

    def lock_path(self) -> str:
        return os.path.join(self.out_dir, ".lock")
