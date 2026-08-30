"""GPU-side rollout instrument for the artifact signatures. Runs under isaaclab.sh -p.

TWO MODES, and the second one is what makes the first defensible.

  RA_MODE=baseline   No policy. Measures, in ONE physics arm, the references the
                     signatures are judged against: the overlap the object shows
                     resting on the table under its own weight, the free-fall
                     acceleration and energy residuals this integrator produces
                     with no contacts at all, and the noise floor of the
                     object-vs-TCP speed comparison at rest. Writes
                     ``baseline_<arm>.json``.

  RA_MODE=rollout    Replays a checkpoint under a chosen physics arm and writes a
                     per-step, per-env trace (``<stem>.trace.npz``) plus a JSON
                     summary. Optionally records the rollout to video.

WHAT IS READ, AND FROM WHERE. Penetration comes from the SAP contact jacobian's
own ``contact_env_phi0_wp`` -- the per-(env, slot) signed gap at the solve
anchor, negative for overlap, in metres -- the same array pass 32's differentially
verified reduction reads. This probe adds a PER-ENV reduction (pass 32's is
batch-global) because a signature has to be correlated with the reward that
specific env earned, and it adds the body-identity mask
(``contact_env_body0/1_wp``) so gripper-vs-object overlap can be separated from
the object merely resting on the table.

WHY THE SAMPLE IS TAKEN AFTER ``env.step``. ``phi0`` is what the last committed
solve left behind, which is exactly the configuration the reward was computed
on. Sampling once per env step (not per substep) keeps the trace aligned with
the reward tensor index-for-index.

NO CONTACT FORCES ARE READ, deliberately. Pass 27 measured both SAP arms'
``update_contacts`` as no-ops that leave the frontend force array identically
zero, so every force-based test would silently read 0.0 on both arms. Every
signature here is therefore built from geometry and from the object's own
reported motion, both of which are live.

Env: RA_MODE, RA_PHYSICS (fixed|adaptive), RA_POLICY, RA_CKPT, RA_TASK, RA_NENV,
RA_STEPS, RA_SEED, RA_OUT (path stem), RA_VIDEO, RA_VIDEO_DIR, RA_VIDEO_LEN,
RA_STOCHASTIC, RA_DROP_Z.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback

from isaaclab.app import AppLauncher

_VIDEO = os.environ.get("RA_VIDEO", "0") == "1"

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
if _VIDEO:
    # This build has no --enable_cameras flag; the RL entrypoints set the same
    # attribute on the namespace (entrypoints/common.py:enable_cameras_for_video).
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

MODE = os.environ.get("RA_MODE", "rollout")
PHYSICS = os.environ.get("RA_PHYSICS", "fixed")
POLICY = os.environ.get("RA_POLICY", "")
CKPT = os.environ.get("RA_CKPT", "")
TASK = os.environ.get("RA_TASK", "IsaacContrib-Lift-Spatula-Trossen-v0")
NENV = int(os.environ.get("RA_NENV", "256"))
STEPS = int(os.environ.get("RA_STEPS", "600"))
SEED = int(os.environ.get("RA_SEED", "1234"))
OUT = os.environ.get("RA_OUT", "ra_out")
STOCHASTIC = os.environ.get("RA_STOCHASTIC", "0") == "1"
DROP_Z = float(os.environ.get("RA_DROP_Z", "0.6"))
G = 9.80665

res: dict = {
    "mode": MODE,
    "physics_arm": PHYSICS,
    "policy_arm": POLICY,
    "checkpoint": CKPT,
    "task": TASK,
    "num_envs": NENV,
    "steps": STEPS,
    "seed": SEED,
    "notes": [],
}


def note(msg: str) -> None:
    res["notes"].append(msg)
    print(f"[artifact_probe] {msg}", flush=True)


# --------------------------------------------------------------------------
# contact-set reduction, per env
# --------------------------------------------------------------------------


class ContactReader:
    """Per-env reductions over the SAP contact set.

    Three numbers per env per step:
      ``pen_obj``  deepest overlap on ANY contact involving the object [m]
      ``pen_grip`` deepest overlap on gripper<->object contacts only [m]
      ``gap_obj``  smallest signed gap over the object's contacts [m]; +inf when
                   the object has no contact row at all, which is the collision
                   pipeline's own statement that nothing is within reach of it

    The gap is the jacobian's ``phi0``, which is the witness-point distance MINUS
    both shapes' collision margins. The margins are read back per slot rather than
    assumed zero, so overlap is a true geometric overlap even if a margin is ever
    authored on this scene.
    """

    def __init__(self, jac, obj_body_of_env: torch.Tensor, is_grip_body: torch.Tensor):
        self.jac = jac
        self.obj = obj_body_of_env.view(-1, 1)  # (E, 1) global body index
        self.is_grip = is_grip_body  # (B,) bool
        self.E, self.C = jac.contact_env_phi0_wp.shape
        self.slot = None
        self.margin_ok = True

    def _margins(self, shape: tuple[int, int], device) -> torch.Tensor:
        m0 = getattr(self.jac, "_set_margin0", None)
        m1 = getattr(self.jac, "_set_margin1", None)
        if m0 is None or m1 is None:
            self.margin_ok = False
            return torch.zeros(shape, device=device, dtype=torch.float64)
        t0, t1 = wp.to_torch(m0), wp.to_torch(m1)
        if tuple(t0.shape) != shape or tuple(t1.shape) != shape:
            self.margin_ok = False
            return torch.zeros(shape, device=device, dtype=torch.float64)
        return (t0 + t1).to(torch.float64)

    def read(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        jac = self.jac
        phi = wp.to_torch(jac.contact_env_phi0_wp)  # (E, C) float64, margins already subtracted
        cnt = wp.to_torch(jac.contact_env_count_wp).long()  # (E,)
        b0 = wp.to_torch(jac.contact_env_body0_wp).long()  # (E, C)
        b1 = wp.to_torch(jac.contact_env_body1_wp).long()
        dev = phi.device
        if self.slot is None or self.slot.device != dev:
            self.slot = torch.arange(phi.shape[1], device=dev).view(1, -1)
        live = (self.slot < cnt.view(-1, 1)) & ((b0 >= 0) | (b1 >= 0))

        obj = self.obj.to(dev)
        touches_obj = live & ((b0 == obj) | (b1 == obj))
        grip = self.is_grip.to(dev)
        g0 = torch.where(b0 >= 0, grip[b0.clamp(min=0)], torch.zeros_like(live))
        g1 = torch.where(b1 >= 0, grip[b1.clamp(min=0)], torch.zeros_like(live))
        grip_obj = touches_obj & (g0 | g1)

        # phi0 = witness distance - margin0 - margin1; add the margins back so
        # "overlap" means shared volume rather than "inside the margin band".
        true_gap = phi + self._margins(tuple(phi.shape), dev)
        overlap = (-true_gap).clamp(min=0.0)

        zero = torch.zeros((), dtype=overlap.dtype, device=dev)
        inf = torch.full((), float("inf"), dtype=true_gap.dtype, device=dev)
        pen_obj = torch.where(touches_obj, overlap, zero).amax(dim=1)
        pen_grip = torch.where(grip_obj, overlap, zero).amax(dim=1)
        gap_obj = torch.where(touches_obj, true_gap, inf).amin(dim=1)
        n_obj = touches_obj.sum(dim=1)
        return pen_obj, pen_grip, gap_obj, n_obj


def resolve_bodies(model, num_envs: int, device) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Map each env's object body, and the set of gripper bodies, to global indices.

    Two independent routes are computed and cross-checked, because a silent
    off-by-one here would relabel every contact in the experiment:
      LABELS  match the per-env body label text;
      STRIDE  assume worlds are laid out contiguously with a constant body count.
    Disagreement is recorded rather than resolved silently.
    """
    labels = [str(x) for x in getattr(model, "body_label", [])]
    info: dict = {"body_count": len(labels), "route": None}
    if not labels:
        raise RuntimeError("model.body_label is empty: cannot identify the object body")

    obj_idx = [i for i, s in enumerate(labels) if s.rstrip("/").endswith("Object") or "/Object" in s]
    grip_idx = [i for i, s in enumerate(labels) if "carriage" in s.lower()]
    info["object_bodies_found"] = len(obj_idx)
    info["gripper_bodies_found"] = len(grip_idx)
    info["gripper_labels"] = [labels[i] for i in grip_idx[:4]]

    if len(obj_idx) == num_envs:
        by_label = sorted(obj_idx, key=lambda i: _env_of(labels[i]))
        info["route"] = "labels"
    elif len(obj_idx) >= 1 and len(labels) % num_envs == 0:
        stride = len(labels) // num_envs
        by_label = [obj_idx[0] + e * stride for e in range(num_envs)]
        info["route"] = "stride"
        info["stride"] = stride
        note(f"object body resolved by STRIDE (found {len(obj_idx)} labelled object bodies, stride {stride})")
    else:
        raise RuntimeError(
            f"cannot map object bodies: {len(obj_idx)} matches for {num_envs} envs over {len(labels)} bodies"
        )

    if len(labels) % num_envs == 0:
        stride = len(labels) // num_envs
        cross = [by_label[0] + e * stride for e in range(num_envs)]
        info["stride_agrees"] = bool(cross == list(by_label))
        if not info["stride_agrees"]:
            note("WARNING: label-derived and stride-derived object body indices DISAGREE")

    is_grip = torch.zeros(len(labels), dtype=torch.bool, device=device)
    if grip_idx:
        is_grip[torch.tensor(grip_idx, device=device)] = True
    else:
        note("WARNING: no gripper ('carriage') bodies found; gripper<->object overlap will read 0")
    return torch.tensor(by_label, dtype=torch.long, device=device), is_grip, info


def _env_of(label: str) -> int:
    for part in label.split("/"):
        if part.startswith("env_"):
            try:
                return int(part[4:])
            except ValueError:
                return 0
    return 0


def wall_thickness(model, obj_shape_ids: list[int]) -> float | None:
    """Thinnest collision-geometry extent of the object [m].

    A convex collision piece of a mug wall is a slab: its smallest bounding-box
    extent IS the wall thickness. Penetration past it means the finger is through
    the part, which no amount of contact compliance explains. Returns None when
    the geometry cannot be read, so the signature reports UNCALIBRATED rather
    than inventing a number.
    """
    try:
        sources = getattr(model, "shape_source", None)
        scale = getattr(model, "shape_scale", None)
        if sources is None:
            return None
        scale_np = scale.numpy() if hasattr(scale, "numpy") else None
        best = None
        for sid in obj_shape_ids:
            src = sources[sid] if sid < len(sources) else None
            verts = getattr(src, "vertices", None)
            if verts is None:
                continue
            v = np.asarray(verts, dtype=np.float64)
            if v.ndim != 2 or v.shape[0] < 4:
                continue
            ext = v.max(axis=0) - v.min(axis=0)
            if scale_np is not None and sid < len(scale_np):
                ext = ext * np.abs(np.asarray(scale_np[sid], dtype=np.float64))
            t = float(ext.min())
            if t > 0 and (best is None or t < best):
                best = t
        return best
    except Exception as exc:  # noqa: BLE001
        note(f"wall_thickness unavailable: {type(exc).__name__}: {exc}")
        return None


# --------------------------------------------------------------------------
# environment construction
# --------------------------------------------------------------------------


def build_env():
    env_cfg = parse_env_cfg(TASK, num_envs=NENV)
    env_cfg.seed = SEED
    # Same latches p31's cross-evaluation used, so the two instruments describe
    # the same two arms: backend sap, adaptivity from the arm, and the substep
    # count each arm's preset carries.
    env_cfg.sim.physics.solver_cfg.backend = "sap"
    env_cfg.sim.physics.solver_cfg.adaptive = False
    env_cfg.sim.physics.solver_cfg.sap_adaptive = PHYSICS == "adaptive"
    env_cfg.sim.physics.num_substeps = 1 if PHYSICS == "adaptive" else 2
    if _VIDEO:
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: PLC0415

        vdir = os.environ.get("RA_VIDEO_DIR") or os.path.dirname(os.path.abspath(OUT)) or "."
        os.makedirs(vdir, exist_ok=True)
        # An EXPLICIT output_dir and prefix per cell. `play` writes into the
        # checkpoint's own run directory, which made pass 30's cross cells
        # overwrite the same-arm ones; naming the destination here makes that
        # collision impossible.
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:newton",
                output_dir=vdir,
                output_filename_prefix=os.path.basename(OUT),
                video_length=int(os.environ.get("RA_VIDEO_LEN", "300")),
                video_interval=0,
                step_offset=0,
            )
        ]
    env = gym.make(TASK, cfg=env_cfg)
    return env, env_cfg


def solver_handles():
    from isaaclab_newton.physics.mjwarp_manager import NewtonManager  # noqa: PLC0415

    solver = NewtonManager._solver
    sap = getattr(solver, "_sap", None) or solver
    jac = getattr(sap, "contact_jacobian", None)
    res["solver_class"] = type(solver).__name__
    res["resolved_num_substeps"] = int(getattr(NewtonManager, "_num_substeps", 0))
    res["solver_substep_dt"] = float(getattr(NewtonManager, "_solver_dt", 0.0))
    return NewtonManager, solver, jac


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

env = None
try:
    env, env_cfg = build_env()
    u = env.unwrapped
    dev = u.device
    NewtonManager, solver, jac = solver_handles()
    if jac is None:
        raise RuntimeError(
            f"no SAP contact jacobian on {type(solver).__name__}: this probe measures SAP contact "
            "geometry and has nothing to read under another backend"
        )

    model = getattr(NewtonManager, "_model", None) or getattr(solver, "model", None)
    if model is None:
        raise RuntimeError("cannot reach the Newton model to identify bodies")
    obj_body, is_grip, body_info = resolve_bodies(model, u.num_envs, dev)
    res["body_resolution"] = body_info
    reader = ContactReader(jac, obj_body, is_grip)
    res["contact_capacity_per_world"] = int(reader.C)

    obj = u.scene["object"]
    robot = u.scene["robot"]
    mass = None
    for attr in ("default_mass", "masses"):
        m = getattr(obj.data, attr, None)
        if m is not None:
            t = m.torch if hasattr(m, "torch") else m
            mass = float(t.reshape(t.shape[0], -1)[0].sum().item())
            break
    res["object_mass_kg"] = mass
    if mass is None:
        note("object mass unavailable: the energy signature will be UNCALIBRATED")

    shape_labels = [str(x) for x in getattr(model, "shape_label", [])]
    obj_shape_ids = [i for i, s in enumerate(shape_labels) if "/Object/" in s or s.endswith("Object")]
    # One world's worth is enough for a geometric property.
    if obj_shape_ids and u.num_envs > 1:
        per_env = max(1, len(obj_shape_ids) // u.num_envs)
        obj_shape_ids = obj_shape_ids[:per_env]
    res["object_shape_ids_sampled"] = obj_shape_ids[:16]
    res["wall_thickness_m"] = wall_thickness(model, obj_shape_ids)

    # TCP offset along the EE link's local x, from the task's own constant.
    ee_name = "follower_left_link_6"
    ee_idx = robot.body_names.index(ee_name) if ee_name in robot.body_names else None
    tcp_off = torch.tensor([0.087, 0.0, 0.0], device=dev)
    if ee_idx is None:
        note(f"EE link '{ee_name}' not in body_names; TCP speed will read 0 and the ejection test dies")

    from isaaclab.utils.math import quat_apply  # noqa: E402

    def tcp_speed() -> torch.Tensor:
        if ee_idx is None:
            return torch.zeros(u.num_envs, device=dev)
        q = robot.data.body_quat_w.torch[:, ee_idx]
        v = robot.data.body_lin_vel_w.torch[:, ee_idx]
        w = robot.data.body_ang_vel_w.torch[:, ee_idx]
        r = quat_apply(q, tcp_off.expand(u.num_envs, 3))
        return torch.linalg.vector_norm(v + torch.cross(w, r, dim=-1), dim=-1)

    def obj_state():
        z = obj.data.root_pos_w.torch[:, 2]
        v = obj.data.root_lin_vel_w.torch
        w = obj.data.root_ang_vel_w.torch
        return z, v, w

    step_dt = float(getattr(u, "step_dt", u.cfg.sim.dt * u.cfg.decimation))
    res["step_dt"] = step_dt

    # ---------------------------------------------------------------- baseline
    if MODE == "baseline":
        act = torch.zeros((u.num_envs, u.action_space.shape[-1]), device=dev)
        env.reset()
        # (1) REST. Let the scene settle with a held action; the object carries
        #     exactly its own weight, so its overlap is the depth this compliant
        #     law needs for one body weight -- the unit the penetration signatures
        #     are quoted in.
        rest_pen, rest_gap, rest_vres = [], [], []
        for i in range(60):
            env.step(act)
            pen_obj, pen_grip, gap_obj, n_obj = reader.read()
            if i >= 30:
                rest_pen.append(pen_obj.detach().clone())
                rest_gap.append(gap_obj.detach().clone())
                z, v, w = obj_state()
                # Object and TCP are both at rest here, so what is left in this
                # comparison is the noise floor of the comparison itself.
                rest_vres.append((torch.linalg.vector_norm(v, dim=-1) - tcp_speed()).abs())
        rest_pen_t = torch.stack(rest_pen)
        res["rest_penetration_m"] = float(rest_pen_t.median().item())
        res["rest_penetration_p95_m"] = float(torch.quantile(rest_pen_t.flatten().float(), 0.95).item())
        res["rest_contact_rows_median"] = float(n_obj.float().median().item())
        res["velocity_noise_m_s"] = float(torch.stack(rest_vres).abs().max().item())
        if res["rest_penetration_m"] <= 0:
            note(
                "resting overlap measured as ZERO: the penetration ratio has no unit and will report "
                "UNCALIBRATED. Use the wall-thickness signature instead."
            )

        # (2) FREE FALL. Teleport the object above the scene with zero velocity
        #     and let it fall untouched. With no contacts the integrator has no
        #     excuse: a_z must be -g and 0.5 m|v|^2 + m g z must be constant.
        #     Whatever it is NOT is this arm's own error, and that is the
        #     tolerance every dynamics signature is judged against.
        root = obj.data.root_state_w.torch.clone()
        root[:, 2] = u.scene.env_origins[:, 2] + DROP_Z
        root[:, 7:13] = 0.0
        try:
            obj.write_root_state_to_sim(root)
        except Exception:  # noqa: BLE001
            obj.write_root_pose_to_sim(root[:, :7])
            obj.write_root_velocity_to_sim(root[:, 7:13])
        a_res, e_drift = [], []
        prev_vz = None
        prev_e = None
        for _ in range(14):
            env.step(act)
            pen_obj, pen_grip, gap_obj, n_obj = reader.read()
            z, v, w = obj_state()
            free = torch.isinf(gap_obj) | (gap_obj > 0)
            vz = v[:, 2]
            e = 0.5 * mass * (v * v).sum(dim=-1) + mass * G * z if mass else None
            if prev_vz is not None:
                az = (vz - prev_vz) / step_dt
                sel = free & torch.isfinite(az)
                if sel.any():
                    a_res.append((az[sel] + G).abs())
                    if e is not None and prev_e is not None:
                        e_drift.append((e - prev_e)[sel].abs())
            prev_vz, prev_e = vz.clone(), (e.clone() if e is not None else None)
        res["freefall_samples"] = int(sum(int(x.numel()) for x in a_res))
        res["freefall_accel_residual"] = float(torch.cat(a_res).median().item()) if a_res else None
        res["freefall_accel_residual_p95"] = (
            float(torch.quantile(torch.cat(a_res).float(), 0.95).item()) if a_res else None
        )
        res["freefall_energy_drift_j"] = float(torch.cat(e_drift).median().item()) if e_drift else None
        if not a_res:
            note("no contact-free samples during the drop: the levitation and energy tests stay UNCALIBRATED")

        res["minimal_height_m"] = 0.08
        res["table_top_z_m"] = 0.02
        res["contact_margin_read"] = bool(reader.margin_ok)
        res["ok"] = True

    # ---------------------------------------------------------------- rollout
    else:
        from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, filter_unsupported_rsl_rl_kwargs  # noqa: PLC0415

        agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
        runner = OnPolicyRunner(
            wrapped, filter_unsupported_rsl_rl_kwargs(agent_cfg.to_dict()), log_dir=None, device=str(dev)
        )
        runner.load(CKPT)
        policy = runner.get_inference_policy(device=dev)

        T, E = STEPS, u.num_envs
        buf = {
            k: torch.zeros((T, E), device=dev, dtype=torch.float32)
            for k in (
                "obj_z",
                "obj_speed",
                "obj_vz",
                "obj_energy_j",
                "tcp_speed",
                "obj_ang_speed",
                "pen_obj_max",
                "pen_grip_max",
                "gap_obj_min",
                "reward",
                "done",
            )
        }
        obs = wrapped.get_observations()
        with torch.inference_mode():
            for t in range(T):
                act = policy(obs, stochastic_output=True) if STOCHASTIC else policy(obs)
                obs, rew, dones, _ = wrapped.step(act)
                pen_obj, pen_grip, gap_obj, _n = reader.read()
                z, v, w = obj_state()
                sp = torch.linalg.vector_norm(v, dim=-1)
                buf["obj_z"][t] = z
                buf["obj_speed"][t] = sp
                buf["obj_vz"][t] = v[:, 2]
                buf["obj_ang_speed"][t] = torch.linalg.vector_norm(w, dim=-1)
                buf["obj_energy_j"][t] = (0.5 * mass * sp * sp + mass * G * z) if mass else 0.0
                buf["tcp_speed"][t] = tcp_speed()
                buf["pen_obj_max"][t] = pen_obj
                buf["pen_grip_max"][t] = pen_grip
                buf["gap_obj_min"][t] = gap_obj
                buf["reward"][t] = rew
                buf["done"][t] = dones.float()
                try:
                    policy.reset(dones)
                except Exception:  # noqa: BLE001
                    pass
        torch.cuda.synchronize()

        host = {k: v.detach().cpu().numpy() for k, v in buf.items()}
        meta = {
            "policy": POLICY,
            "physics": PHYSICS,
            "checkpoint": CKPT,
            "seed": SEED,
            "dt": step_dt,
            "contact_overflow": bool(res.get("contact_overflow", False)),
            "pen_channel_live": True,
            "notes": res["notes"],
        }
        np.savez_compressed(f"{OUT}.trace.npz", meta=json.dumps(meta), **host)
        res["trace"] = f"{OUT}.trace.npz"

        # A summary that does not need the analysis module, so a run can be
        # sanity-checked from the log alone.
        d = host["done"].astype(bool)
        r = host["reward"]
        closed = d.sum()
        res["completed_episodes"] = int(closed)
        res["mean_step_reward"] = float(r.mean())
        res["lift_step_fraction"] = float((host["obj_z"] > 0.08).mean())
        res["pen_grip_max_overall_m"] = float(host["pen_grip_max"].max())
        res["state_finite"] = bool(np.isfinite(host["obj_z"]).all())
        res["ok"] = True

except Exception as exc:  # noqa: BLE001
    res["ok"] = False
    res["error"] = f"{type(exc).__name__}: {exc}"
    res["traceback"] = traceback.format_exc()

out_json = f"{OUT}.json" if MODE == "rollout" else os.environ.get("RA_BASELINE_OUT", f"{OUT}.json")
os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
with open(out_json, "w") as fh:
    json.dump(res, fh, indent=2, sort_keys=True, default=str)
print(json.dumps({k: v for k, v in res.items() if k != "traceback"}, indent=2, sort_keys=True, default=str))
if not res.get("ok"):
    print(res.get("traceback", ""))
try:
    env.close()  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass
app.close()
