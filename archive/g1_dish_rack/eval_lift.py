"""M1 gate metric: % of episodes that lift the plate clear of the rack.

Loads an rsl_rl checkpoint, runs deterministic inference over N complete
episodes (reset map live, as in training), and classifies each episode:

* lifted   — after the plate settled into the slot (env-frame z < 0.90 at
  least once, which excludes the initial 19 cm drop-in), its z exceeded 0.95
  for >=10 consecutive steps (a held ~9 cm lift off the rest pose, rim clear
  of the slot slats);
* dropped  — episode ended by the plate_dropped termination;
* other    — timed out without a held lift.

Usage (from the IsaacLab root):

    PYTHONPATH=/home/mdigiorgio/Documents/code/newton-adaptive ./isaaclab.sh -p \
        /home/mdigiorgio/Documents/code/IsaacLabRubato/experiments/g1_dish_rack/eval_lift.py \
        <checkpoint.pt> [num_episodes]
"""

import sys

import gymnasium as gym
import torch

from importlib import metadata

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.contrib.g1_dish_rack import g1_dish_rack_env_cfg as mod
from isaaclab_tasks.contrib.g1_dish_rack.agents.rsl_rl_ppo_cfg import G1DishRackPPORunnerCfg

from rsl_rl.runners import OnPolicyRunner

NUM_ENVS = 64
SETTLE_BELOW = 0.90  # env-frame z [m]: plate has entered the slot region
LIFT_ABOVE = 0.95  # env-frame z [m]: held lift clear of the slot
HOLD_STEPS = 10


def main():
    ckpt = sys.argv[1]
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 192

    cfg = mod.G1DishRackEnvCfg()
    cfg.scene.num_envs = NUM_ENVS
    cfg.sim.physics = mod.PhysicsCfg().newton_mjwarp
    env = gym.make("IsaacContrib-Dish-Rack-G1-v0", cfg=cfg)
    agent_cfg = handle_deprecated_rsl_rl_cfg(G1DishRackPPORunnerCfg(), metadata.version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=str(env.unwrapped.device))
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    plate = env.unwrapped.scene["plate"]
    origins = env.unwrapped.scene.env_origins
    term_mgr = env.unwrapped.termination_manager

    settled = torch.zeros(NUM_ENVS, dtype=torch.bool, device=env.unwrapped.device)
    lifted = torch.zeros(NUM_ENVS, dtype=torch.bool, device=env.unwrapped.device)
    hold = torch.zeros(NUM_ENVS, dtype=torch.long, device=env.unwrapped.device)
    n_done = n_lift = n_drop = 0

    obs = wrapped.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    with torch.no_grad():
        while n_done < num_episodes:
            actions = policy(obs)
            obs, _, dones, _ = wrapped.step(actions)
            z = plate.data.root_pos_w.torch[:, 2] - origins[:, 2]
            settled |= z < SETTLE_BELOW
            hold = torch.where(settled & (z > LIFT_ABOVE), hold + 1, torch.zeros_like(hold))
            lifted |= hold >= HOLD_STEPS
            done_mask = dones.to(torch.bool).squeeze(-1)
            if done_mask.any():
                dropped = term_mgr.get_term("plate_dropped").to(torch.bool)
                n = int(done_mask.sum())
                n_done += n
                n_lift += int((lifted & done_mask).sum())
                n_drop += int((dropped & done_mask).sum())
                settled[done_mask] = False
                lifted[done_mask] = False
                hold[done_mask] = 0

    print(f"[eval] checkpoint: {ckpt}")
    print(f"[eval] episodes={n_done}  lifted={n_lift} ({100.0 * n_lift / n_done:.1f}%)  "
          f"dropped={n_drop} ({100.0 * n_drop / n_done:.1f}%)  "
          f"other={n_done - n_lift - n_drop}")
    env.close()


if __name__ == "__main__":
    main()
