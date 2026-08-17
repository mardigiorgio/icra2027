"""Preflight parity: prove the arms differ only where they are meant to.

A fixed-vs-adaptive comparison is worth nothing if the arms also differ in
tolerance, contact capacity, contact law, determinism or containment. Those
values are rewritten on the way to the solver by presets, env vars and
manager-side rules, so they are read back from the constructed objects (see
preflight_probe.py) and compared here BEFORE the sweep spends GPU hours.

Any difference outside ``preflight.expected_differences`` aborts the sweep.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from .config import Cell, SweepCfg

PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preflight_probe.py")

# Fields that are legitimately per-arm in a fixed-vs-adaptive comparison. Any
# other difference is a confound until someone says otherwise in the config.
INHERENT_DIFFERENCES = (
    "arm",
    "solver_class",
    "cfg_num_substeps",
    "resolved_num_substeps",
    "solver_substep_dt",
    "arm_identity.adaptive_tol",
    "env_overrides",
)


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = obj
    return out


def run_probe(cfg: SweepCfg, arm_name: str, out_json: str, echo=print) -> dict[str, Any]:
    arm = next(a for a in cfg.arms if a.name == arm_name)
    env = dict(os.environ)
    env["VIRTUAL_ENV"] = cfg.venv_dir
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env["ARM"] = arm_name
    env["RS_TASK"] = cfg.task
    env["RS_NENV"] = str(cfg.preflight.num_envs)
    env["CHECK_OUT"] = out_json
    env.update(arm.env)
    cmd = [os.path.join(cfg.isaaclab_dir, "isaaclab.sh"), "-p", PROBE, "--headless"]
    echo(f"preflight: {arm_name} -> {out_json}")
    log_path = out_json.replace(".json", ".log")
    with open(log_path, "w") as log:
        subprocess.run(
            cmd, cwd=cfg.isaaclab_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
            timeout=cfg.preflight.timeout_s,
        )
    if not os.path.exists(out_json):
        raise RuntimeError(f"preflight probe produced no JSON for arm '{arm_name}'; see {log_path}")
    with open(out_json) as fh:
        return json.load(fh)


def compare(dumps: dict[str, dict[str, Any]], expected: tuple[str, ...]) -> dict[str, Any]:
    """Diff the arms' resolved identity; classify each difference."""
    allowed = set(INHERENT_DIFFERENCES) | set(expected)
    flat = {arm: _flatten(d) for arm, d in dumps.items()}
    keys: set[str] = set()
    for f in flat.values():
        keys |= set(f)
    unexpected: dict[str, dict[str, Any]] = {}
    intended: dict[str, dict[str, Any]] = {}
    for key in sorted(keys):
        values = {arm: f.get(key) for arm, f in flat.items()}
        if len({json.dumps(v, sort_keys=True, default=str) for v in values.values()}) == 1:
            continue
        if any(key == a or key.startswith(a + ".") for a in allowed):
            intended[key] = values
        else:
            unexpected[key] = values
    return {"intended": intended, "unexpected": unexpected, "ok": not unexpected}


def preflight(cfg: SweepCfg, cells: list[Cell] | None = None, echo=print) -> dict[str, Any]:
    """Dump every arm and abort unless the differences are the intended ones."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    dumps: dict[str, dict[str, Any]] = {}
    for arm in cfg.arms:
        out = os.path.join(cfg.out_dir, f"preflight_{arm.name}.json")
        dumps[arm.name] = run_probe(cfg, arm.name, out, echo=echo)
        if not dumps[arm.name].get("ok"):
            raise RuntimeError(f"preflight probe failed for arm '{arm.name}': {dumps[arm.name].get('error')}")
    report = compare(dumps, cfg.preflight.expected_differences)
    report["dumps"] = dumps
    path = os.path.join(cfg.out_dir, "preflight_report.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1)
    echo(f"preflight report -> {path}")
    for key, values in report["intended"].items():
        echo(f"  intended difference  {key}: {values}")
    for key, values in report["unexpected"].items():
        echo(f"  UNEXPECTED difference {key}: {values}")
    if report["unexpected"] and cfg.preflight.abort_on_diff:
        raise RuntimeError(
            f"{len(report['unexpected'])} unintended difference(s) between arms -- "
            "the comparison would not be defensible. Fix them, or list them in "
            "preflight.expected_differences with a reason."
        )
    return report
