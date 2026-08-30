"""Vendored June-era lift base (drift armor).

``StationaryAiCubeLiftEnvCfg`` was written against the June ``LiftEnvCfg``; the
in-tree class has since been rewritten (different reward/event/observation terms) and
subclassing it now fails on every name the Trossen ``__post_init__`` touches. This
base reproduces the June API SURFACE the experiment depends on -- same manager term
names, same params -- on top of the stable generic layer only, so the debugged cube
config keeps working verbatim and future in-tree churn cannot reach it.

Term names deliberately preserved: ``rewards.reaching_object / lifting_object /
object_goal_tracking / object_goal_tracking_fine_grained / action_rate / joint_vel``,
``events.reset_all / reset_object_position``, ``commands.object_pose``,
``terminations.time_out / object_dropping``.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils.configclass import configclass

from . import mdp_local as mdp


@configclass
class TrossenLiftSceneCfg(InteractiveSceneCfg):
    """Slots the task config fills in ``__post_init__`` (robot, object, ee_frame)."""

    robot: ArticulationCfg | None = None
    object: RigidObjectCfg | None = None
    ee_frame: FrameTransformerCfg | None = None
    # The Stationary AI rig carries its own tabletop; the June base's foreign table is
    # represented as an always-None slot so ``self.scene.table = None`` stays valid.
    table: AssetBaseCfg | None = None
    plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="",  # set by the task config (EE_LINK)
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.1, 0.1),
            pos_y=(-0.1, 0.1),
            pos_z=(0.1, 0.3),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Filled by the task config; ``None`` terms are skipped by the action manager."""

    arm_action: mdp.JointPositionActionCfg | None = None
    gripper_action: mdp.BinaryJointPositionActionCfg | None = None


@configclass
class DefaultObservationsCfg:
    """Minimal default; the Trossen tasks replace this wholesale with their own groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """The June 4-term lift reward + rate/velocity penalties, names preserved."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class TrossenLiftEnvCfg(ManagerBasedRLEnvCfg):
    scene: TrossenLiftSceneCfg = TrossenLiftSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: DefaultObservationsCfg = DefaultObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        # June-era timing: 100 Hz physics, decimation 2 -> 50 Hz control, 5 s episodes.
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
