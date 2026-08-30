"""M0 settle test: zero actions, 500 steps — the plate must rest in the rack slot.

Run from the IsaacLab root:

    PYTHONPATH=/home/mdigiorgio/Documents/code/newton-adaptive ./isaaclab.sh -p \
        /home/mdigiorgio/Documents/code/IsaacLabRubato/experiments/g1_dish_rack/settle_test.py

Pass criteria (printed as PASS/FAIL at the end):
  - zero terminations across all envs for the whole horizon;
  - plate position drift from the post-drop settled pose < 2 cm;
  - plate orientation drift < 10 deg (stays on edge, no roll-away);
  - final-step plate<->rack contact force ~ resting weight (report only).
"""

import math

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  (registers the task)
from isaaclab_tasks.contrib.g1_dish_rack import g1_dish_rack_env_cfg as mod

NUM_ENVS = 8
STEPS = 500
SETTLE_STEPS = 50  # let the 4 mm spawn drop settle before baselining drift


def main():
    cfg = mod.G1DishRackEnvCfg()
    cfg.scene.num_envs = NUM_ENVS
    cfg.sim.physics = mod.PhysicsCfg().newton_mjwarp
    env = gym.make("IsaacContrib-Dish-Rack-G1-v0", cfg=cfg).unwrapped
    env.reset()

    plate = env.scene["plate"]
    sensor = env.scene.sensors["plate_rack_contact"]
    action = torch.zeros(NUM_ENVS, env.action_manager.total_action_dim, device=env.device)

    peak_force = torch.zeros(NUM_ENVS, device=env.device)
    final_force = torch.zeros(NUM_ENVS, device=env.device)
    dones_total = 0
    p_ref = q_ref = None

    for i in range(STEPS):
        _, _, terminated, truncated, _ = env.step(action)
        # timeouts (truncated) are expected past episode_length_s; only real
        # terminations indicate a physics problem
        dones_total += int(terminated.sum().item())
        forces = sensor.data.force_matrix_w
        if forces is not None:
            mag = torch.linalg.vector_norm(forces.torch, dim=-1).reshape(NUM_ENVS, -1).max(dim=1).values
            if i >= SETTLE_STEPS:
                peak_force = torch.maximum(peak_force, mag)
            final_force = mag
        if i == SETTLE_STEPS:
            p_ref = plate.data.root_pos_w.torch.clone()
            q_ref = plate.data.root_quat_w.torch.clone()

    pos_drift = torch.linalg.vector_norm(plate.data.root_pos_w.torch - p_ref, dim=1)
    dot = (plate.data.root_quat_w.torch * q_ref).sum(dim=1).abs().clamp(max=1.0)
    ang_drift_deg = torch.rad2deg(2.0 * torch.acos(dot))

    print(f"[settle] terminations: {dones_total}")
    print(f"[settle] pos drift [m]   max={pos_drift.max():.4f} mean={pos_drift.mean():.4f}")
    print(f"[settle] ang drift [deg] max={ang_drift_deg.max():.2f} mean={ang_drift_deg.mean():.2f}")
    print(f"[settle] rack force [N]  peak(post-settle)={peak_force.max():.2f} final mean={final_force.mean():.2f}")
    print(f"[settle] resting weight reference: m*g = {0.375 * 9.81:.2f} N")

    ok = dones_total == 0 and pos_drift.max() < 0.02 and ang_drift_deg.max() < 10.0
    print(f"[settle] {'PASS' if ok else 'FAIL'}")
    env.close()


if __name__ == "__main__":
    main()
