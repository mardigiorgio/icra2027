"""Author grasp_map.pt for the G1 dish-rack reset map (M1).

Two modes:

* ``search`` (default): batch-FK search — 64 envs, each env's right arm
  teleported (write_joint_state_to_sim, no dynamics) to a sampled config;
  palms read back in one shot; prints the best configs by distance to the
  palm target. Iterate ranges if needed.
* ``--save <7 joint values>``: verifies the given right-arm config
  dynamically (ramped joint targets, plate parked far away so the swing
  can't hit it), then snapshots the full joint state as stage 1
  ("just_about_to_grasp") with the plate on its proven drop-in spawn, and
  writes grasp_map.pt next to the task package.

The stored plate pose is ALWAYS the normal drop-in spawn: free-body root
teleports near colliders explode the solver, so the reset map
only repositions the ROBOT (articulation joint teleports are proven safe —
the G1's toes reset 2 cm from the table every episode).

Run from the IsaacLab root with
``PYTHONPATH=/home/mdigiorgio/Documents/code/newton-adaptive ./isaaclab.sh -p <this file> [args]``.
"""

import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.g1_dish_rack import g1_dish_rack_env_cfg as mod

ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Palm goal: beside the plate's top rim (rest center (0.25, -0.120, 0.864),
# rim top z ~0.962), robot-side. Deliberately >=0.10 m from the plate SPAWN
# point (0.25, -0.129, 1.052): the plate root-teleports there at reset, and a
# staged hand inside the ~0.1 m contact margin of that teleport would trigger
# the reset-explosion pathology.
PALM_TARGET = (0.25, -0.17, 0.94)

SEARCH_ENVS = 64
# sample ranges [rad] around zero pose per joint (min, max)
RANGES = {
    "right_shoulder_pitch_joint": (-1.6, 0.4),
    "right_shoulder_roll_joint": (-1.4, 0.1),
    "right_shoulder_yaw_joint": (-0.8, 0.8),
    "right_elbow_joint": (0.0, 1.4),
    "right_wrist_roll_joint": (-0.5, 0.5),
    "right_wrist_pitch_joint": (-0.8, 0.8),
    "right_wrist_yaw_joint": (-0.8, 0.8),
}


def build_env(num_envs):
    cfg = mod.G1DishRackEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.physics = mod.PhysicsCfg().newton_mjwarp
    env = gym.make("IsaacContrib-Dish-Rack-G1-v0", cfg=cfg).unwrapped
    env.reset()
    return env


def search():
    env = build_env(SEARCH_ENVS)
    robot = env.scene["robot"]
    all_names = robot.joint_names
    arm_ids = [all_names.index(n) for n in ARM_JOINTS]
    palm_id = robot.find_bodies(["right_hand_palm_link"])[0][0]

    torch.manual_seed(42)
    joint_pos = robot.data.default_joint_pos.torch.clone()
    lo = torch.tensor([RANGES[n][0] for n in ARM_JOINTS], device=env.device)
    hi = torch.tensor([RANGES[n][1] for n in ARM_JOINTS], device=env.device)
    samples = lo + (hi - lo) * torch.rand(SEARCH_ENVS, len(ARM_JOINTS), device=env.device)
    joint_pos[:, arm_ids] = samples
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))

    palms = robot.data.body_pos_w.torch[:, palm_id, :] - env.scene.env_origins
    target = torch.tensor(PALM_TARGET, device=env.device)
    dist = torch.linalg.vector_norm(palms - target, dim=1)

    # HARD CONSTRAINT: no right hand/wrist link may intrude into the plate's
    # fall corridor (expanded edgewise slab through the slot) -- a link inside
    # it tips the plate flat on landing.
    # Corridor: x in [0.13,0.37], y > -0.17, z in [0.75,1.10].
    hand_ids = [i for i, n in enumerate(robot.body_names) if "right" in n and ("hand" in n or "wrist" in n)]
    hp = robot.data.body_pos_w.torch[:, hand_ids, :] - env.scene.env_origins[:, None, :]
    in_corr = (
        (hp[..., 0] > 0.13) & (hp[..., 0] < 0.37) & (hp[..., 1] > -0.17) & (hp[..., 2] > 0.75) & (hp[..., 2] < 1.10)
    ).any(dim=1)
    dist = torch.where(in_corr, torch.full_like(dist, float("inf")), dist)
    print(f"[search] {int(in_corr.sum())}/{len(in_corr)} configs rejected for corridor intrusion")
    order = torch.argsort(dist)
    print("[search] top corridor-clean configs (dist | palm xyz | 7 joint values):")
    for k in order[:5].tolist():
        p = palms[k]
        vals = " ".join(f"{v:+.3f}" for v in samples[k].tolist())
        print(f"  {dist[k]:.3f} | ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) | {vals}")
    env.close()


def save(vals):
    env = build_env(1)
    robot = env.scene["robot"]
    plate = env.scene["plate"]
    # park the plate far away so the arm swing cannot touch it during the ramp
    park = torch.tensor([[-2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]], device=env.device)
    plate.write_root_pose_to_sim_index(root_pose=park + torch.cat(
        [env.scene.env_origins[:1], torch.zeros(1, 4, device=env.device)], dim=1))
    plate.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device))

    act_joint_names = env.action_manager.get_term("upper_body")._joint_names
    all_names = robot.joint_names
    default_pos = robot.data.default_joint_pos.torch[0]
    target_offsets = torch.zeros(len(act_joint_names), device=env.device)
    for name, tgt in zip(ARM_JOINTS, vals):
        i = act_joint_names.index(name)
        target_offsets[i] = (tgt - default_pos[all_names.index(name)]) / 0.5

    # teleport straight to the FK config (joint teleports are safe), then hold
    # it under the PD for 50 steps so the snapshot is a holdable, settled state
    all_ids = [all_names.index(n) for n in ARM_JOINTS]
    joint_pos = robot.data.default_joint_pos.torch.clone()
    joint_pos[0, all_ids] = torch.tensor(vals, device=env.device)
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    action = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    action[0, :] = target_offsets
    for _ in range(50):
        env.step(action)

    palm_id = robot.find_bodies(["right_hand_palm_link"])[0][0]
    palm = robot.data.body_pos_w.torch[0, palm_id] - env.scene.env_origins[0]
    err = torch.tensor(PALM_TARGET, device=env.device) - palm
    print(f"[save] dynamic palm = ({palm[0]:+.3f},{palm[1]:+.3f},{palm[2]:+.3f}) |err|={float(torch.linalg.norm(err)):.3f}")

    stage = {
        "name": "just_about_to_grasp",
        "joint_pos": robot.data.joint_pos.torch[0].detach().cpu().clone(),
        "joint_vel": torch.zeros(robot.data.joint_pos.torch.shape[1]),
        "object_pos": torch.tensor(mod.PLATE_SPAWN_POS),
        "object_quat": torch.tensor(mod.PLATE_SPAWN_QUAT),  # xyzw
    }
    torch.save({"stages": [stage]}, mod.GRASP_MAP_PATH)
    print(f"[save] wrote stage 'just_about_to_grasp' -> {mod.GRASP_MAP_PATH}")
    env.close()


if __name__ == "__main__":
    if "--save" in sys.argv:
        i = sys.argv.index("--save")
        save([float(v) for v in sys.argv[i + 1 : i + 8]])
    else:
        search()
