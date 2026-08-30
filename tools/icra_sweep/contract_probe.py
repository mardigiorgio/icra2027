#!/usr/bin/env python
"""Dump the parts of a task that a MULTI-ENGINE comparison must hold equal.

    cd ~/Documents/code/IsaacLab
    env ARM=sap_fixed RS_TASK=<task> RS_NENV=8 CHECK_OUT=/path/sap_fixed.json \\
        ./isaaclab.sh -p <this file> --headless

WHY A SECOND PROBE. ``preflight_probe.py`` answers "do these two SAP arms solve
the same problem?" by reading the SAP solver's tolerances and contact law. That
question is well posed only inside one engine family. Across PhysX, MuJoCo and
SAP there is no shared object graph to read, and every solver field is SUPPOSED
to differ -- the contact law IS the difference. Diffing them would either abort
the sweep or, once silenced, check nothing at all.

So this probe dumps the other thing: THE CONTRACT. The axes a four-engine
comparison equalizes, and therefore the only axes its claims may live on:

  * the MDP -- the names and weights of every observation, reward, termination,
    event, command, curriculum and action term, and the resolved action and
    observation dimensions;
  * the control contract -- ``sim.dt``, ``decimation``, the derived control
    rate, ``episode_length_s`` and the resulting episode length in steps;
  * the batch -- ``num_envs``, and the PPO hyperparameters that decide what an
    "iteration" is;
  * the scene inventory -- which assets exist, their masses, and the joints and
    bodies the policy acts on.

If any of those differ, the four arms are not running the same experiment, and
no amount of care downstream repairs it.

WHAT IT DOES NOT CLAIM. Everything under ``engine`` is recorded for the archive,
NOT compared: it is read best-effort and each backend's block is allowed to be
absent. A missing block is reported as ``null`` with the reason, never as a
default, because a probe that invents a value for an engine it could not read is
worse than one that admits it.

Env: ARM (a key of icra_sweep.physics_arm.ARMS), RS_TASK, RS_NENV, RS_SEED,
CHECK_OUT.

STATUS: written pass 35 under a no-GPU rail and NOT EXECUTED. Its Newton access
paths are lifted from ``preflight_probe.py``; its PhysX block is written from
source and has never run. Treat the first launch as a shakedown, and expect the
PhysX arm to fail before this probe reports -- the task's contact sensors are
pinned to the Newton backend (see the pass-35 ledger entry).
"""

from __future__ import annotations

import json
import os
import sys
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.icra_sweep import physics_arm  # noqa: E402

ARM = os.environ.get("ARM", "sap_fixed")
TASK = os.environ.get("RS_TASK", "IsaacContrib-Lift-Spatula-Trossen-v0")
NENV = int(os.environ.get("RS_NENV", "8"))
SEED = int(os.environ.get("RS_SEED", "42"))
OUT = os.environ.get("CHECK_OUT", "contract.json")

res: dict = {
    "arm": ARM,
    "task": TASK,
    "num_envs": NENV,
    "seed": SEED,
    "env_overrides": {k: v for k, v in os.environ.items() if k.startswith(("NEWTON_", "SAP_"))},
}


def _term_names(manager) -> dict:
    """Term names, and weights and functions where the manager has them.

    Names alone would pass a comparison in which one arm's reward weights were
    all zero, or in which a term name was bound to a different function, so both
    are recorded. ``active_terms`` is a list on most managers and a group->list
    mapping on the observation manager; both shapes are handled.
    """
    out: dict = {}
    names = getattr(manager, "active_terms", None)
    if names is not None:
        out["terms"] = (
            {k: list(v) for k, v in names.items()} if isinstance(names, dict) else list(names)
        )
    cfgs = getattr(manager, "_term_cfgs", None)
    if cfgs is not None and isinstance(out.get("terms"), list) and len(cfgs) == len(out["terms"]):
        detail: dict = {}
        for name, c in zip(out["terms"], cfgs):
            fn = getattr(c, "func", None)
            entry: dict = {"func": getattr(fn, "__name__", None) or str(fn)}
            if hasattr(c, "weight"):
                entry["weight"] = float(c.weight)
            if hasattr(c, "mode"):
                entry["mode"] = str(c.mode)
            detail[name] = entry
        out["detail"] = detail
    return out


try:
    env_cfg = parse_env_cfg(TASK, num_envs=NENV)
    env_cfg, applied = physics_arm.apply_to(env_cfg, TASK, ARM)
    env_cfg.seed = SEED
    res["arm_applied"] = {
        "preset": applied.preset,
        "solver": applied.solver,
        "physics_cls": applied.physics_cls,
        "resolved_substeps": applied.resolved_substeps,
        "steps": applied.steps,
    }

    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped

    dt = float(u.cfg.sim.dt)
    dec = int(u.cfg.decimation)
    ep_s = float(u.cfg.episode_length_s)

    # ---- THE CONTRACT: every key below must be identical across all arms -----
    res["contract"] = {
        "control": {
            "sim_dt": dt,
            "decimation": dec,
            "control_dt": dt * dec,
            "control_hz": (1.0 / (dt * dec)) if dt * dec > 0 else None,
            "episode_length_s": ep_s,
            "episode_length_steps": int(round(ep_s / (dt * dec))) if dt * dec > 0 else None,
        },
        "spaces": {
            "num_actions": int(u.action_space.shape[-1]),
            "num_observations": {
                k: [int(x) for x in v.shape] for k, v in dict(u.observation_space.spaces).items()
            }
            if hasattr(u.observation_space, "spaces")
            else [int(x) for x in u.observation_space.shape],
            "num_envs": int(u.num_envs),
        },
        "mdp": {},
        "scene": {},
    }
    for name in ("observation_manager", "reward_manager", "termination_manager",
                 "action_manager", "event_manager", "command_manager", "curriculum_manager"):
        mgr = getattr(u, name, None)
        res["contract"]["mdp"][name] = _term_names(mgr) if mgr is not None else None

    scene: dict = {"assets": sorted(u.scene.keys()) if hasattr(u.scene, "keys") else None}
    try:
        robot = u.scene["robot"]
        scene["robot_joint_names"] = list(robot.data.joint_names)
        scene["robot_body_names"] = list(robot.data.body_names)
        scene["robot_default_joint_pos"] = [round(float(x), 6) for x in robot.data.default_joint_pos[0]]
    except Exception as exc:  # noqa: BLE001
        scene["robot_error"] = f"{type(exc).__name__}: {exc}"
    try:
        obj = u.scene["object"]
        scene["object_mass_kg"] = round(float(obj.root_physx_view.get_masses()[0].sum()), 8) if hasattr(
            obj, "root_physx_view"
        ) else round(float(obj.data.default_mass[0].sum()), 8)
    except Exception as exc:  # noqa: BLE001
        scene["object_mass_error"] = f"{type(exc).__name__}: {exc}"
    res["contract"]["scene"] = scene

    # ---- THE ENGINE: recorded, never compared ------------------------------
    engine: dict = {"family": physics_arm.get(ARM).family, "physics_cls": applied.physics_cls}
    try:
        from isaaclab_newton.physics.mjwarp_manager import NewtonManager  # noqa: PLC0415

        solver = NewtonManager._solver
        engine["newton"] = {
            "solver_class": type(solver).__name__,
            "solver_substep_dt": float(NewtonManager._solver_dt),
            "resolved_num_substeps": int(NewtonManager._num_substeps),
            "advance_per_boundary": float(NewtonManager._solver_dt * NewtonManager._num_substeps),
        }
        sap = getattr(solver, "_sap", None)
        if sap is not None or hasattr(solver, "contact_jacobian"):
            sap = sap or solver
            jac, cs = sap.contact_jacobian, sap.contact_solve
            engine["sap"] = {
                "optimality_rel_tol": float(sap.optimality_rel_tol),
                "max_rigid_contact_per_world": int(jac.max_rigid_contact),
                "contact_beta": float(cs.contact_beta),
                "contact_sigma": float(cs.contact_sigma),
                "authored_shape_ke": sorted({round(float(x), 6) for x in jac.contact_shape_ke.numpy()})[:8],
                "authored_shape_tau": sorted({round(float(x), 8) for x in jac.contact_shape_tau.numpy()})[:8],
                "attempt_consistent_r": bool(getattr(solver, "_attempt_consistent_r", False)),
            }
    except Exception as exc:  # noqa: BLE001
        engine["newton"] = None
        engine["newton_reason"] = f"{type(exc).__name__}: {exc}"
    try:
        phys = u.cfg.sim.physics
        engine["physx"] = {
            k: getattr(phys, k)
            for k in (
                "bounce_threshold_velocity",
                "friction_correlation_distance",
                "gpu_max_rigid_patch_count",
                "gpu_total_aggregate_pairs_capacity",
                "gpu_found_lost_aggregate_pairs_capacity",
            )
            if hasattr(phys, k)
        } or None
    except Exception as exc:  # noqa: BLE001
        engine["physx"] = None
        engine["physx_reason"] = f"{type(exc).__name__}: {exc}"
    res["engine"] = engine

    env.close()
    res["ok"] = True
except Exception as exc:  # a probe that dies must say why, in the JSON
    res["ok"] = False
    res["error"] = f"{type(exc).__name__}: {exc}"
    res["traceback"] = traceback.format_exc()

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1, sort_keys=True)
print(json.dumps({k: v for k, v in res.items() if k != "traceback"}, indent=1, sort_keys=True))
app.close()
