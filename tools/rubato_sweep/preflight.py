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

_HERE = os.path.dirname(os.path.abspath(__file__))
#: Single-family (SAP fixed-vs-adaptive) probe: reaches into the SAP solver and
#: dumps its tolerances, capacities and contact law.
PROBE = os.path.join(_HERE, "preflight_probe.py")
#: Multi-engine probe: backend-agnostic. Dumps the MDP and control contract that
#: the engines must share, and only best-effort solver detail, because no single
#: object graph spans PhysX and Newton.
CONTRACT_PROBE = os.path.join(_HERE, "contract_probe.py")

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
    probe = CONTRACT_PROBE if cfg.preflight.by_family else PROBE
    cmd = [os.path.join(cfg.isaaclab_dir, "isaaclab.sh"), "-p", probe, "--headless"]
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


def _matches(key: str, patterns: set[str]) -> bool:
    return any(key == p or key.startswith(p + ".") for p in patterns)


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
        if _matches(key, allowed):
            intended[key] = values
        else:
            unexpected[key] = values
    return {"intended": intended, "unexpected": unexpected, "ok": not unexpected}


def compare_by_family(
    dumps: dict[str, dict[str, Any]],
    families: dict[str, list[str]],
    expected: tuple[str, ...],
    contract_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Multi-engine preflight: strict within a family, contract-only across them.

    Returns the same ``intended``/``unexpected``/``ok`` shape as :func:`compare`,
    plus a per-family breakdown and the resolved contract, so the report says
    which check each finding came from.

    A key that is MISSING on one arm and present on another counts as a contract
    violation, not as agreement: an engine that cannot report the control rate
    has not demonstrated it matches.
    """
    per_family: dict[str, dict[str, Any]] = {}
    unexpected: dict[str, dict[str, Any]] = {}
    intended: dict[str, dict[str, Any]] = {}

    for fam, arms in sorted(families.items()):
        if len(arms) < 2:
            per_family[fam] = {"arms": arms, "intended": {}, "unexpected": {}, "ok": True,
                               "note": "single arm in this family: nothing to diff"}
            continue
        rep = compare({a: dumps[a] for a in arms}, expected)
        per_family[fam] = {"arms": arms, **{k: rep[k] for k in ("intended", "unexpected", "ok")}}
        for key, values in rep["intended"].items():
            intended[f"[{fam}] {key}"] = values
        for key, values in rep["unexpected"].items():
            unexpected[f"[{fam}] {key}"] = values

    flat = {arm: _flatten(d) for arm, d in dumps.items()}
    contract: dict[str, dict[str, Any]] = {}
    missing = set(contract_keys)
    for key in sorted({k for f in flat.values() for k in f}):
        if not _matches(key, set(contract_keys)):
            continue
        missing.discard(next((p for p in contract_keys if key == p or key.startswith(p + ".")), key))
        values = {arm: f.get(key, "<absent>") for arm, f in flat.items()}
        contract[key] = values
        if len({json.dumps(v, sort_keys=True, default=str) for v in values.values()}) != 1:
            unexpected[f"[contract] {key}"] = values

    for key in sorted(missing):
        unexpected[f"[contract] {key}"] = {arm: "<absent>" for arm in dumps}

    return {
        "mode": "by_family",
        "intended": intended,
        "unexpected": unexpected,
        "ok": not unexpected,
        "per_family": per_family,
        "contract": contract,
        "contract_keys_absent": sorted(missing),
    }


def preflight(cfg: SweepCfg, cells: list[Cell] | None = None, echo=print) -> dict[str, Any]:
    """Dump every arm and abort unless the differences are the intended ones."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    dumps: dict[str, dict[str, Any]] = {}
    for arm in cfg.arms:
        out = os.path.join(cfg.out_dir, f"preflight_{arm.name}.json")
        dumps[arm.name] = run_probe(cfg, arm.name, out, echo=echo)
        if not dumps[arm.name].get("ok"):
            raise RuntimeError(f"preflight probe failed for arm '{arm.name}': {dumps[arm.name].get('error')}")
    if cfg.preflight.by_family:
        fams: dict[str, list[str]] = {}
        for arm in cfg.arms:
            fams.setdefault(arm.family, []).append(arm.name)
        report = compare_by_family(
            dumps, fams, cfg.preflight.expected_differences, cfg.preflight.contract_keys
        )
    else:
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
