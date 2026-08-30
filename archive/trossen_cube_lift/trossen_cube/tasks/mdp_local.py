"""Vendored lift MDP terms + generic mdp re-export (drift armor).

The experiment was written against the June-era ``isaaclab_tasks.core.lift`` API;
that package has since been rewritten (different rewards/observations/events) and
will keep moving. Everything here depends only on the STABLE generic layer
(``isaaclab.envs.mdp``, asset/sensor data buffers, the command manager), so the
Trossen tasks no longer break when the in-tree lift task churns.

The four functions below are the June-era ``core.lift.mdp`` implementations,
verbatim in behavior. Data buffers on current develop are warp-backed ProxyArrays;
the ``.torch`` property is the sanctioned torch view (see isaaclab.envs.mdp).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import *  # noqa: F401,F403  (generic terms: actions, resets, penalties, commands)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """1.0 when the object root is above ``minimal_height`` [m] (world z)."""
    obj = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w.torch[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reach shaping: ``1 - tanh(|object - ee| / std)`` using the ee_frame sensor's first target."""
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    distance = torch.norm(obj.data.root_pos_w.torch - ee_pos_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Goal-tracking shaping, gated on the object being lifted."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, des_pos_b)
    distance = torch.norm(des_pos_w - obj.data.root_pos_w.torch, dim=1)
    return (obj.data.root_pos_w.torch[:, 2] > minimal_height) * (1.0 - torch.tanh(distance / std))


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object root position expressed in the robot root frame (privileged teacher obs)."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, obj.data.root_pos_w.torch
    )
    return object_pos_b
