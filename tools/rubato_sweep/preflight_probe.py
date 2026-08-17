#!/usr/bin/env python
"""Dump one arm's RESOLVED solver identity, read off the constructed objects.

Run under Isaac Lab, once per arm, before a sweep spends GPU hours:

    cd ~/Documents/code/IsaacLab
    env ARM=fixed  RS_TASK=<task> RS_NENV=8 CHECK_OUT=/path/fixed.json \\
        ./isaaclab.sh -p <this file> --headless

The point is that a comparison is only defensible if the two arms differ in the
one thing under study. Config files do not settle that: presets, env vars and
manager-side rules all rewrite these values on the way to the solver, so every
field below is read from the CONSTRUCTED solver, jacobian and contact-solve
objects rather than from any cfg.

Env: ARM=fixed|adaptive, RS_TASK, RS_NENV, RS_SEED, RS_SUBSTEPS, CHECK_OUT.

STATUS: written pass 33, NOT YET EXECUTED (that pass was forbidden from using
the GPU). Its access paths are copied from the campaign's proven parity probe
(scratchpad p30_regime_probe.py, which produced p30_parity_*.json /
p31_parity_*.json); treat the first run as a shakedown.
"""

from __future__ import annotations

import json
import os
import traceback

from isaaclab.app import AppLauncher

import argparse  # noqa: E402  (AppLauncher must import first)

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

ARM = os.environ.get("ARM", "fixed")
TASK = os.environ.get("RS_TASK", "IsaacContrib-Lift-Spatula-Trossen-v0")
NENV = int(os.environ.get("RS_NENV", "8"))
SEED = int(os.environ.get("RS_SEED", "42"))
SUBSTEPS = int(os.environ.get("RS_SUBSTEPS", "2"))
OUT = os.environ.get("CHECK_OUT", "preflight.json")

res: dict = {
    "arm": ARM,
    "task": TASK,
    "num_envs": NENV,
    "seed": SEED,
    "env_overrides": {k: v for k, v in os.environ.items() if k.startswith(("NEWTON_", "SAP_", "ARM"))},
}


def _g(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


try:
    env_cfg = parse_env_cfg(TASK, num_envs=NENV)
    env_cfg.seed = SEED
    if ARM == "adaptive":
        env_cfg.sim.physics.num_substeps = 1
        env_cfg.sim.physics.solver_cfg.adaptive = True
    else:
        env_cfg.sim.physics.num_substeps = SUBSTEPS
    res["cfg_num_substeps"] = int(env_cfg.sim.physics.num_substeps)
    res["cfg_decimation"] = int(_g(env_cfg, "decimation") or -1)

    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped

    from isaaclab_newton.physics.mjwarp_manager import NewtonManager, NewtonMJWarpManager  # noqa: E402

    solver = NewtonManager._solver
    sap = getattr(solver, "_sap", None) or solver
    jac = sap.contact_jacobian
    cs = sap.contact_solve

    res["solver_class"] = type(solver).__name__
    res["solver_substep_dt"] = float(NewtonManager._solver_dt)
    res["physics_dt"] = float(NewtonManager._solver_dt * NewtonManager._num_substeps)
    res["resolved_num_substeps"] = int(NewtonManager._num_substeps)

    # Tolerance triple, capacities, contact law, determinism, containment.
    res["arm_identity"] = {
        "optimality_rel_tol": float(sap.optimality_rel_tol),
        "optimality_abs_tol": float(_g(sap, "optimality_abs_tol") or float("nan")),
        "cost_abs_tol": float(_g(sap, "cost_abs_tol") or 0.0),
        "cost_rel_tol": float(_g(sap, "cost_rel_tol") or 0.0),
        "max_iterations": int(_g(sap, "max_iterations", "_max_iterations") or -1),
        "line_search_max_iterations": int(_g(sap, "line_search_max_iterations") or -1),
        "line_search_variant": str(_g(sap, "line_search_variant")),
        "contact_preset_variant": str(_g(sap, "contact_preset_variant")),
        "contact_solve_precision": str(_g(cs, "solve_precision")),
        "contact_beta": float(cs.contact_beta),
        "contact_sigma": float(cs.contact_sigma),
        "max_rigid_contact_per_world": int(jac.max_rigid_contact),
        "jac_deterministic": bool(_g(jac, "_deterministic")),
        "containment_strict": bool(getattr(NewtonMJWarpManager, "_sap_strict", False)),
        "attempt_consistent_r": bool(getattr(solver, "_attempt_consistent_r", False)),
        "adaptive_tol": float(_g(solver, "_tol") or float("nan")) if ARM == "adaptive" else None,
    }

    res["contact_law"] = {
        "shape_count": int(jac.contact_shape_ke.shape[0]),
        "authored_shape_ke": sorted({round(float(x), 6) for x in jac.contact_shape_ke.numpy()})[:8],
        "authored_shape_tau": sorted({round(float(x), 8) for x in jac.contact_shape_tau.numpy()})[:8],
        "fallback_stiffness": float(jac.fallback_stiffness),
        "fallback_tau_d": float(jac.fallback_tau_d),
    }

    res["capacity"] = {
        "max_triangle_pairs": int(_g(jac, "max_triangle_pairs", "_max_triangle_pairs") or -1),
        "tri_pairs_per_world": None,
        "rigid_contact_max": int(_g(jac, "max_rigid_contact") or -1) * NENV,
    }
    tri = res["capacity"]["max_triangle_pairs"]
    if tri > 0:
        res["capacity"]["tri_pairs_per_world"] = tri // max(1, NENV)

    # Run-ahead's window must equal the env's decimation or it advances across a
    # boundary the policy was supposed to see.
    runahead = os.environ.get("NEWTON_SAP_RUNAHEAD", "0") == "1"
    res["runahead"] = {
        "enabled": runahead,
        "window": int(os.environ.get("NEWTON_SAP_RUNAHEAD_WINDOW", "4")),
        "decimation": res["cfg_decimation"],
        "window_matches_decimation": int(os.environ.get("NEWTON_SAP_RUNAHEAD_WINDOW", "4"))
        == res["cfg_decimation"],
    }

    env.close()
    res["ok"] = True
except Exception as exc:  # a probe that dies must say why, in the JSON
    res["ok"] = False
    res["error"] = f"{type(exc).__name__}: {exc}"
    res["traceback"] = traceback.format_exc()

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "traceback"}, indent=1))
app.close()
