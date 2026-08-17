"""Turn a training log (and its solver telemetry) into one structured record per run.

Everything a later comparison needs is extracted here and written to JSON, so no
analysis ever has to re-grep a 300 MB log or trust a number a human typed into a
notes file.

DEMAND NORMALIZATION, stated once and enforced in code. The adaptive arm's cost
is only comparable to the fixed arm's after dividing by the integration work
actually done, and there are two different denominators one iteration can be
divided by:

    boundaries_per_iteration = num_steps_per_env * decimation
    env_steps_per_iteration  = num_steps_per_env

A solver boundary is one call into the solver; an env step is ``decimation``
boundaries. Quoting "substeps per env step" against another run's "substeps per
boundary" is a silent factor of ``decimation``. :func:`normalize_demand`
therefore returns BOTH, named, and asserts their ratio.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ITER = re.compile(r"Learning iteration (\d+)/(\d+)")
KV = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_/ ]*?):\s+(-?[\d.]+(?:e-?\d+)?)\s*$")
ITER_TIME = re.compile(r"Iteration time:\s+([\d.]+)s")
TRAINING_TIME = re.compile(r"^Training time: ([\d.]+) seconds")
OVERFLOW = re.compile(r"Triangle pair buffer overflowed (\d+) > (\d+)")
OOM = re.compile(r"out of memory|cudaErrorMemoryAllocation|Failed to allocate \d+ bytes", re.I)
LOGDIR = re.compile(r"Logging experiment in directory: (\S+)")
EXPNAME = re.compile(r"Exact experiment name requested from command line: (\S+)")
SOLVER_LATCH = re.compile(r"--solver=(\S+) -> (\{.*\})")

# NEWTON_ADAPTIVE_LOG lines, e.g.
#   frame=120 inner_dt[min=... max=...] cumulative_substeps=812 cumulative_accepted=406
TELEM_KV = re.compile(r"(\w+)=([-\d.eE+]+)")


@dataclass
class RunRecord:
    """One run's structured result. Written to <run_name>.json."""

    run_name: str = ""
    arm: str = ""
    task: str = ""
    num_envs: int = 0
    seed: int = 0
    iterations_requested: int | None = None
    iterations_completed: int = 0
    complete: bool = False
    exclusive: bool = True
    concurrent_peers: int = 0
    exit_code: int | None = None
    wall_s: float | None = None
    training_time_s: float | None = None
    startup_s: float | None = None
    gpu_peak_mib: int | None = None
    log_path: str = ""
    log_dir: str = ""
    num_steps_per_env: int = 0
    decimation: int = 0
    solver_latch: str = ""
    stack: dict[str, str] = field(default_factory=dict)
    env_overrides: dict[str, str] = field(default_factory=dict)
    iters: list[dict[str, Any]] = field(default_factory=list)
    overflow_events: list[dict[str, int]] = field(default_factory=list)
    first_overflow_iter: int | None = None
    oom: bool = False
    telemetry: dict[str, Any] = field(default_factory=dict)
    demand: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=1)

    @staticmethod
    def from_json(path: str) -> RunRecord:
        with open(path) as fh:
            data = json.load(fh)
        rec = RunRecord()
        for k, v in data.items():
            if hasattr(rec, k):
                setattr(rec, k, v)
        return rec

    # -- convenience for the aggregator ------------------------------------

    def series(self, key: str) -> list[tuple[int, float]]:
        return [(it["iter"], it[key]) for it in self.iters if key in it]

    def window_mean(self, key: str, lo: int, hi: int) -> float | None:
        vals = [it[key] for it in self.iters if key in it and lo <= it["iter"] <= hi]
        return sum(vals) / len(vals) if vals else None

    def last_n_mean(self, key: str, n: int) -> float | None:
        vals = [it[key] for it in self.iters if key in it]
        if not vals:
            return None
        vals = vals[-n:]
        return sum(vals) / len(vals)

    @property
    def clean_iters(self) -> int:
        """Iterations before the contact buffer started truncating silently.

        Past this point the physics is degraded, so any physics comparison must
        stop here even though the learning curve continues.
        """
        return self.first_overflow_iter if self.first_overflow_iter is not None else self.iterations_completed


def parse_training_log(path: str) -> RunRecord:
    """Parse an rsl_rl training log into a RunRecord (no telemetry, no demand)."""
    rec = RunRecord(log_path=os.path.abspath(path))
    cur: dict[str, Any] | None = None
    for raw in open(path, errors="replace"):
        line = ANSI.sub("", raw.rstrip("\n"))
        m = ITER.search(line)
        if m:
            if cur is not None:
                rec.iters.append(cur)
            cur = {"iter": int(m.group(1))}
            rec.iterations_requested = int(m.group(2))
            continue
        mo = OVERFLOW.search(line)
        if mo:
            at = cur["iter"] if cur else -1
            rec.overflow_events.append({"iter": at, "demanded": int(mo.group(1)), "capacity": int(mo.group(2))})
            if rec.first_overflow_iter is None:
                rec.first_overflow_iter = at
            continue
        mt = TRAINING_TIME.match(line)
        if mt:
            rec.training_time_s = float(mt.group(1))
            continue
        if not rec.log_dir:
            md = LOGDIR.search(line)
            if md:
                rec.log_dir = md.group(1)
        if not rec.solver_latch:
            ms = SOLVER_LATCH.search(line)
            if ms:
                rec.solver_latch = f"{ms.group(1)} -> {ms.group(2)}"
        if not rec.oom and OOM.search(line):
            rec.oom = True
        if cur is None:
            continue
        mi = ITER_TIME.search(line)
        if mi:
            cur["iter_time_s"] = float(mi.group(1))
            continue
        mk = KV.match(line)
        if mk:
            try:
                cur[mk.group(1).strip()] = float(mk.group(2))
            except ValueError:
                pass
    if cur is not None:
        rec.iters.append(cur)
    rec.iterations_completed = len(rec.iters)
    # rsl_rl numbers iterations 0..N-1, so a finished N-iteration run ends on
    # N-1 and then prints its total. Ctrl-C exits 0 with neither, which is why
    # completion is not inferred from the exit code.
    rec.complete = rec.training_time_s is not None or (
        rec.iterations_requested is not None and rec.iterations_completed >= rec.iterations_requested
    )
    return rec


def is_complete(log_path: str, iterations: int | None) -> bool:
    """Cheap skip-if-complete test: does this log show a finished run?

    Used before launching, so it must not parse a whole 300 MB log. Reads the
    tail for the completion print and, when the horizon is known, the
    last-iteration marker.
    """
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return False
    with open(log_path, "rb") as fh:
        fh.seek(max(0, os.path.getsize(log_path) - 262144))
        tail = ANSI.sub("", fh.read().decode("utf-8", "replace"))
    if TRAINING_TIME.search(tail):
        return True
    if iterations is not None and f"Learning iteration {iterations - 1}/{iterations}" in tail:
        return True
    return False


def parse_telemetry(path: str) -> dict[str, Any]:
    """Parse a NEWTON_ADAPTIVE_LOG telemetry file.

    Lines carry cumulative counters (``cumulative_substeps``,
    ``cumulative_accepted``) sampled every NEWTON_ADAPTIVE_LOG_EVERY frames, so
    the useful quantity is the delta over the run, not the last value alone.
    """
    if not os.path.exists(path):
        return {}
    samples: list[dict[str, float]] = []
    for raw in open(path, errors="replace"):
        kv = {k: float(v) for k, v in TELEM_KV.findall(raw)}
        if "cumulative_accepted" in kv or "cumulative_substeps" in kv:
            samples.append(kv)
    if not samples:
        return {}
    first, last = samples[0], samples[-1]
    out: dict[str, Any] = {"samples": len(samples), "path": os.path.abspath(path)}
    for key in ("cumulative_accepted", "cumulative_substeps", "frame", "ra_fires", "ra_cross"):
        if key in last:
            out[key + "_last"] = last[key]
            out[key + "_delta"] = last[key] - first.get(key, 0.0)
    if "cumulative_accepted_delta" in out and "cumulative_substeps_delta" in out and out["cumulative_substeps_delta"]:
        out["accept_ratio"] = out["cumulative_accepted_delta"] / out["cumulative_substeps_delta"]
    return out


def normalize_demand(
    accepted_substeps: float,
    wall_s: float,
    iterations: int,
    num_steps_per_env: int,
    decimation: int,
) -> dict[str, float]:
    """Demand-normalized price, with both denominators named and cross-checked.

    ``accepted_substeps`` is the BATCH-WIDE count of accepted substeps (marches)
    executed over ``iterations`` learning iterations -- the delta of the solver's
    cumulative accepted-step counter, not a per-world figure.

    Raw wall ratios between arms are trajectory-confounded: an arm that steers
    the policy into a more violent regime pays more wall for more integration
    work, which is not a solver-efficiency difference. ms per accepted substep
    is the price; accepted substeps per boundary is the demand.
    """
    if iterations <= 0 or accepted_substeps <= 0 or wall_s <= 0:
        return {}
    boundaries = iterations * num_steps_per_env * decimation
    env_steps = iterations * num_steps_per_env
    per_boundary = accepted_substeps / boundaries
    per_env_step = accepted_substeps / env_steps
    # The two denominators differ by exactly `decimation`. If this ever fails,
    # a caller has mixed the units the campaign has already mixed once.
    assert abs(per_env_step - per_boundary * decimation) < 1e-9 * max(1.0, per_env_step), (
        "demand normalization inconsistent: per_env_step must be decimation x per_boundary"
    )
    return {
        "accepted_substeps": accepted_substeps,
        "boundaries": boundaries,
        "env_steps": env_steps,
        "accepted_substeps_per_boundary": per_boundary,
        "accepted_substeps_per_env_step": per_env_step,
        "ms_per_accepted_substep": 1000.0 * wall_s / accepted_substeps,
        "decimation": decimation,
        "num_steps_per_env": num_steps_per_env,
    }
