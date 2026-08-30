"""One definition of "a physics arm", shared by training, preflight and evaluation.

WHY THIS EXISTS. A four-engine comparison is only worth running if the engine a
policy TRAINED under is the same engine it is later EVALUATED under, resolved by
the same code path. Until now those two paths disagreed: training selects an
engine with ``--solver`` plus a hydra ``physics=`` preset, while the campaign's
evaluation probes reached past the preset and wrote
``env_cfg.sim.physics.solver_cfg.*`` by hand. Those happen to agree for the two
SAP arms and cannot agree for a PhysX arm, whose physics cfg has no
``solver_cfg`` at all.

So the arm table below is the single source of truth, and both sides read it.

TWO SELECTORS, NOT ONE. Isaac Lab splits engine selection in two:

  * ``physics=NAME`` picks a ``PresetCfg`` alternative on ``sim.physics``. This
    is the only selector that can reach PhysX, and it also carries
    ``num_substeps`` -- which is why the adaptive arm needs its own preset and
    not just a solver flag.
  * ``--solver NAME`` latches ``backend``/``adaptive``/``sap_adaptive`` onto an
    already-resolved Newton cfg. It REQUIRES a Newton preset and raises
    otherwise (``train_rsl_rl.py`` ``_apply_solver_choice``), so the PhysX arm
    must not pass it.

ORDERING IS LOAD-BEARING off the hydra path. ``apply_physics_preset`` reloads the
raw registry config to recover the preset alternatives, so anything written onto
``env_cfg`` before it is discarded. Call it first, then the solver choice, then
task-side overrides. :func:`apply_to` enforces that order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Task-config preset names. These are the alternatives declared on the Trossen
#: task's ``TrossenSpatulaLiftPhysicsCfg``; a task with different preset names
#: needs its own table rather than a rename here.
PRESET_NEWTON = "newton_mjwarp"
PRESET_NEWTON_ADAPTIVE = "newton_mjwarp_adaptive"
PRESET_PHYSX = "physx"


class UnknownArm(KeyError):
    """The arm name is not in :data:`ARMS`."""


@dataclass(frozen=True)
class PhysicsArm:
    """How one engine is selected, on every entry point that can select it.

    Attributes:
        name: The arm's identity. It appears in run names, trace files and the
            cross-play matrix, so it must not change once results exist.
        preset: The ``physics=`` preset alternative this arm resolves to.
        solver: The ``--solver`` latch, or ``None`` for engines that have no
            Newton solver cfg to latch onto (PhysX).
        family: The CONTACT-LAW family. Arms in the same family share a
            constitutive law and may be diffed field by field; arms in
            different families may not, because the law IS the difference.
            MuJoCo-Warp and SAP are both reached through the Newton manager but
            are separate families: one solves against ``solref``/``solimp``, the
            other against SAP's regularization R.
        expected_substeps: The ``num_substeps`` the preset is expected to carry,
            recorded so preflight can catch a preset that silently changed.
        substeps_override: Written onto the resolved cfg AFTER the preset, for
            reference arms only. A production arm leaves this ``None``.
        trainable: False for reference arms, which exist to be evaluated in and
            never to be trained in. The refinement-collapse comparison needs a
            finer version of an engine, not a fifth policy.
        notes: Anything a reader of a result table needs to know about this arm.
    """

    name: str
    preset: str
    solver: str | None
    family: str
    expected_substeps: int | None = None
    substeps_override: int | None = None
    trainable: bool = True
    notes: tuple[str, ...] = ()

    def train_args(self) -> list[str]:
        """CLI fragment for the training entry point."""
        args = [f"physics={self.preset}"]
        if self.solver is not None:
            args += ["--solver", self.solver]
        return args


ARMS: dict[str, PhysicsArm] = {
    "physx": PhysicsArm(
        name="physx",
        preset=PRESET_PHYSX,
        solver=None,
        family="physx",
        expected_substeps=None,
        notes=(
            "Boots Kit: a PhysxCfg makes has_kit_physics true, so this arm runs Isaac Sim where "
            "the Newton arms run kitless.",
            "The task's two contact sensors are pinned to the Newton implementation and the "
            "Newton visualizer is selected in __post_init__; neither survives this arm without a "
            "task-side change.",
        ),
    ),
    "mujoco_fixed": PhysicsArm(
        name="mujoco_fixed",
        preset=PRESET_NEWTON,
        solver="mujoco",
        family="mujoco",
        expected_substeps=2,
    ),
    "mujoco_adaptive": PhysicsArm(
        name="mujoco_adaptive",
        preset=PRESET_NEWTON_ADAPTIVE,
        solver="mujoco-adaptive",
        family="mujoco",
        expected_substeps=1,
    ),
    "sap_fixed": PhysicsArm(
        name="sap_fixed",
        preset=PRESET_NEWTON,
        solver="sap",
        family="sap",
        expected_substeps=2,
    ),
    "sap_adaptive": PhysicsArm(
        name="sap_adaptive",
        preset=PRESET_NEWTON_ADAPTIVE,
        solver="sap-adaptive",
        family="sap",
        expected_substeps=1,
    ),
}

ARMS["sap_fixed_ref"] = PhysicsArm(
    name="sap_fixed_ref",
    preset=PRESET_NEWTON,
    solver="sap",
    family="sap",
    expected_substeps=2,
    substeps_override=8,
    trainable=False,
    notes=(
        "Reference for the SAP family: the same contact law at a 4x finer fixed step.",
        "The adaptive arm's reference is this one, NOT a tightened estimator tolerance: tol is a "
        "campaign rail and it is the physics being demonstrated, so it is not an evaluation knob.",
    ),
)
ARMS["mujoco_fixed_ref"] = PhysicsArm(
    name="mujoco_fixed_ref",
    preset=PRESET_NEWTON,
    solver="mujoco",
    family="mujoco",
    expected_substeps=2,
    substeps_override=8,
    trainable=False,
    notes=("Reference for the MuJoCo family: same solref/solimp law at a 4x finer fixed step.",),
)

#: The four arms of the engine comparison, in the order results are tabulated.
FOURWAY: tuple[str, ...] = ("physx", "mujoco_fixed", "sap_fixed", "sap_adaptive")

#: Per family, the finer-step arm a policy is replayed under to ask whether its
#: score survives refinement of the SAME engine. A score that collapses here was
#: bought from that engine's discretization error, and this comparison needs no
#: cross-engine ground truth to say so.
#:
#: PhysX has no entry: its refinement knob is a solver-iteration and substep
#: schedule that nobody on this machine has ever run, so inventing one would be
#: a guess wearing a measurement's clothes. Add it once the PhysX arm is real.
REFERENCE_FOR: dict[str, str] = {
    "sap": "sap_fixed_ref",
    "mujoco": "mujoco_fixed_ref",
}


def get(name: str) -> PhysicsArm:
    if name not in ARMS:
        raise UnknownArm(f"unknown physics arm '{name}'; known: {sorted(ARMS)}")
    return ARMS[name]


def families(names: list[str] | tuple[str, ...]) -> dict[str, list[str]]:
    """Group arm names by engine family.

    Preflight compares strictly WITHIN a family and only contract-checks across
    families, because two engines are meant to differ in every solver field.
    """
    out: dict[str, list[str]] = {}
    for n in names:
        out.setdefault(get(n).family, []).append(n)
    return out


@dataclass
class Applied:
    """What :func:`apply_to` actually did, for the record a probe writes out."""

    arm: str
    preset: str
    solver: str | None
    physics_cls: str = ""
    resolved_substeps: int | None = None
    steps: list[str] = field(default_factory=list)


def apply_to(env_cfg: Any, task: str, arm_name: str) -> tuple[Any, Applied]:
    """Resolve ``arm_name`` onto a config produced by ``parse_env_cfg``.

    This is the evaluation-side twin of :meth:`PhysicsArm.train_args`. It exists
    so an evaluation cannot drift from the training it is evaluating: both go
    through the same preset, and neither writes solver fields by hand.

    ``apply_physics_preset`` returns a NEW config object rather than mutating in
    place, and discards everything written onto the old one, so the config is
    returned alongside the record: rebinding it is then impossible to forget.

    Args:
        env_cfg: The result of ``parse_env_cfg(task, ...)``, with no other
            mutations applied yet -- see the module docstring.
        task: Registered gym id, needed to reload the raw config carrying the
            live ``PresetCfg`` nodes.
        arm_name: A key of :data:`ARMS`.
    """
    arm = get(arm_name)
    rec = Applied(arm=arm.name, preset=arm.preset, solver=arm.solver)

    from isaaclab_tasks.utils.physics_presets import apply_physics_preset, apply_solver_choice

    env_cfg = apply_physics_preset(env_cfg, task, arm.preset)
    rec.steps.append(f"apply_physics_preset(preset={arm.preset})")
    if arm.solver is not None:
        apply_solver_choice(env_cfg, arm.solver)
        rec.steps.append(f"apply_solver_choice(solver={arm.solver})")

    physics = getattr(getattr(env_cfg, "sim", None), "physics", None)
    rec.physics_cls = type(physics).__name__ if physics is not None else ""
    preset_sub = getattr(physics, "num_substeps", None)
    # Checked against what the PRESET resolved to, before any reference-arm
    # override, so the guard still catches a preset that changed underneath.
    if arm.expected_substeps is not None and preset_sub != arm.expected_substeps:
        raise ValueError(
            f"arm '{arm.name}': preset '{arm.preset}' resolved num_substeps={preset_sub}, "
            f"expected {arm.expected_substeps}. The preset changed under the arm table; a comparison "
            "run on it would not be the comparison this table describes."
        )

    if arm.substeps_override is not None:
        env_cfg.sim.physics.num_substeps = arm.substeps_override
        rec.steps.append(f"num_substeps={arm.substeps_override} (reference arm)")
    sub = getattr(env_cfg.sim.physics, "num_substeps", None) if physics is not None else None
    rec.resolved_substeps = int(sub) if sub is not None else None
    return env_cfg, rec
