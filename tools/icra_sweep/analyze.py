"""Replicate-aware aggregation: a between-arm difference is only reported
against a measured within-arm spread.

WHY THIS REFUSES SINGLE-SEED CLAIMS. Pass 30 ran one seed per arm and reported a
difference. Its own accidental control -- a killed adaptive run restarted from
scratch on the SAME seed -- then diverged 2.4x in mean reward by iteration 9.
The within-arm spread was larger than every between-arm difference reported. So
a cell with one replicate cannot produce a difference here: it produces the
verdict UNRESOLVED, with the reason attached.

Two comparisons are offered, and they answer different questions:
  * RAW WALL is what the experiment cost. It is trajectory-confounded: an arm
    that drives the policy somewhere more violent pays more wall for more
    integration work, which is not a solver-efficiency difference.
  * MS PER ACCEPTED SUBSTEP is the price of a unit of integration work. It is
    the number to quote for solver efficiency, and it needs the telemetry
    counter, so it is absent (not zero) when a run had none.
"""

from __future__ import annotations

import glob
import json
import os
import statistics
from dataclasses import dataclass, field
from typing import Any

from .parse import RunRecord

MIN_REPLICATES = 2


@dataclass
class Group:
    """All replicates of one (arm, num_envs, iterations) cell."""

    arm: str
    num_envs: int
    iterations: int | None
    runs: list[RunRecord] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, int, int | None]:
        return (self.arm, self.num_envs, self.iterations)

    @property
    def n(self) -> int:
        return len(self.runs)

    def values(self, metric: str, window: int = 50) -> list[float]:
        out = []
        for r in self.runs:
            v = _metric(r, metric, window)
            if v is not None:
                out.append(v)
        return out


def _metric(rec: RunRecord, metric: str, window: int) -> float | None:
    if metric == "ms_per_accepted_substep":
        return rec.demand.get("ms_per_accepted_substep")
    if metric == "accepted_substeps_per_boundary":
        return rec.demand.get("accepted_substeps_per_boundary")
    if metric == "samples_per_s":
        v = rec.last_n_mean("iter_time_s", window)
        if not v:
            return None
        return rec.num_envs * rec.num_steps_per_env / v
    if metric == "wall_s":
        return rec.wall_s
    return rec.last_n_mean(metric, window)


def load(out_dir: str) -> list[RunRecord]:
    recs = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        base = os.path.basename(path)
        if base.startswith("preflight") or base in ("summary.json",):
            continue
        try:
            rec = RunRecord.from_json(path)
        except Exception:
            continue
        if rec.run_name:
            recs.append(rec)
    return recs


def group(recs: list[RunRecord]) -> dict[tuple[str, int, int | None], Group]:
    groups: dict[tuple[str, int, int | None], Group] = {}
    for r in recs:
        key = (r.arm, r.num_envs, r.iterations_requested)
        groups.setdefault(key, Group(r.arm, r.num_envs, r.iterations_requested)).runs.append(r)
    return groups


def spread(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    out = {
        "n": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }
    out["range_frac"] = out["range"] / abs(out["mean"]) if out["mean"] else float("nan")
    if len(values) > 1:
        out["stdev"] = statistics.stdev(values)
    return out


def compare_arms(
    groups: dict[tuple[str, int, int | None], Group],
    metric: str,
    num_envs: int,
    iterations: int | None,
    window: int = 50,
) -> dict[str, Any]:
    """Compare every arm at one scale point, against the within-arm spread."""
    cells = {g.arm: g for g in groups.values() if g.num_envs == num_envs and g.iterations == iterations}
    per_arm = {arm: spread(g.values(metric, window)) for arm, g in cells.items()}
    result: dict[str, Any] = {
        "metric": metric,
        "num_envs": num_envs,
        "iterations": iterations,
        "window_iters": window,
        "arms": per_arm,
    }
    usable = {a: s for a, s in per_arm.items() if s and s["n"] >= MIN_REPLICATES}
    if len(usable) < 2:
        thin = {a: s.get("n", 0) for a, s in per_arm.items()}
        result["verdict"] = "UNRESOLVED"
        result["reason"] = (
            f"need >= {MIN_REPLICATES} replicates in at least two arms to separate a between-arm "
            f"difference from within-arm spread; have {thin}. "
            "A single-seed difference on this stack is not evidence (pass 30's same-seed restart "
            "diverged 2.4x in reward by iteration 9)."
        )
        return result

    arms = sorted(usable)
    a, b = arms[0], arms[-1]
    delta = usable[b]["mean"] - usable[a]["mean"]
    widest = max(usable[a]["range"], usable[b]["range"])
    result["delta"] = delta
    result["delta_frac"] = delta / usable[a]["mean"] if usable[a]["mean"] else float("nan")
    result["widest_within_arm_range"] = widest
    result["separated"] = abs(delta) > widest
    result["verdict"] = "SEPARATED" if result["separated"] else "NOT SEPARATED"
    result["reason"] = (
        f"|{b} - {a}| = {abs(delta):.4g} vs widest within-arm range {widest:.4g} "
        f"(n={usable[a]['n']}/{usable[b]['n']})"
    )
    return result


def throughput_table(recs: list[RunRecord], window: int = 50) -> list[dict[str, Any]]:
    """Per-run cost table: s/iter, samples/s, price, and the caveats attached."""
    rows = []
    for r in sorted(recs, key=lambda x: (x.num_envs, x.arm, x.seed)):
        s_iter = r.last_n_mean("iter_time_s", window)
        rows.append(
            {
                "run": r.run_name,
                "arm": r.arm,
                "num_envs": r.num_envs,
                "seed": r.seed,
                "iters": r.iterations_completed,
                "s_per_iter": s_iter,
                "samples_per_s": (r.num_envs * r.num_steps_per_env / s_iter) if s_iter else None,
                "ms_per_accepted_substep": r.demand.get("ms_per_accepted_substep"),
                "accepted_substeps_per_boundary": r.demand.get("accepted_substeps_per_boundary"),
                "gpu_peak_mib": r.gpu_peak_mib,
                "exclusive": r.exclusive,
                "timing_usable": r.exclusive,
                "clean_iters": r.clean_iters,
                "complete": r.complete,
            }
        )
    return rows


def report(out_dir: str, metrics: tuple[str, ...] = (), window: int = 50, echo=print) -> dict[str, Any]:
    metrics = metrics or (
        "Mean reward",
        "iter_time_s",
        "samples_per_s",
        "ms_per_accepted_substep",
        "accepted_substeps_per_boundary",
    )
    recs = load(out_dir)
    groups = group(recs)
    scales = sorted({(g.num_envs, g.iterations) for g in groups.values()}, key=lambda x: (x[0], x[1] or 0))
    out: dict[str, Any] = {"out_dir": os.path.abspath(out_dir), "runs": len(recs), "comparisons": []}

    echo(f"\n{len(recs)} run(s) in {out_dir}")
    echo("\n--- per-run cost (packed runs are marked; their wall is NOT a measurement)")
    header = f"{'run':52} {'s/iter':>8} {'samples/s':>10} {'ms/substep':>11} {'peak MiB':>9} {'excl':>5}"
    echo(header)
    for row in throughput_table(recs, window):
        echo(
            f"{row['run'][:52]:52} "
            f"{_fmt(row['s_per_iter']):>8} {_fmt(row['samples_per_s'], 0):>10} "
            f"{_fmt(row['ms_per_accepted_substep']):>11} {_fmt(row['gpu_peak_mib'], 0):>9} "
            f"{'yes' if row['exclusive'] else 'NO':>5}"
        )
    out["per_run"] = throughput_table(recs, window)

    for num_envs, iterations in scales:
        for metric in metrics:
            cmp = compare_arms(groups, metric, num_envs, iterations, window)
            out["comparisons"].append(cmp)
            echo(f"\n--- {metric} @ {num_envs} envs x {iterations} iters: {cmp['verdict']}")
            for arm, s in cmp["arms"].items():
                if s:
                    echo(
                        f"    {arm:10} n={s['n']} mean={s['mean']:.4g} "
                        f"range=[{s['min']:.4g}, {s['max']:.4g}] ({100 * s['range_frac']:.1f}% of mean)"
                    )
            echo(f"    {cmp['reason']}")

    path = os.path.join(out_dir, "summary.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    echo(f"\nwrote {path}")
    return out


def _fmt(v, digits: int = 2) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"
