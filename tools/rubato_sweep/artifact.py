"""Artifact-exploitation signatures: is a policy's score bought with physics
that is not real?

THE QUESTION THIS ANSWERS. A policy trained under a coarse timestepper can find
states the integrator gets wrong and farm them. "It looks like it is cheating"
is an opinion; this module turns it into a number by asking, at every step that
PAYS reward, whether the physical configuration that earned the payment is
valid.

WHAT THE TASK PAYS FOR, and why that fixes the target. Of the Trossen mug-lift's
37 units of reward weight, 36 are gated on one predicate --
``object root world z > minimal_height`` (``lifting_object`` w=15 directly, and
``object_goal_tracking`` w=16 / fine-grained w=5 multiply by the same gate).
So "exploit" has an unambiguous meaning here: get the object's root above the
gate by means the solver permits and reality does not.

THE FOUR MECHANISMS, and the invariant each one violates:

  A1 INTERPENETRATION LIFT. The gripper is inside the object while carrying it.
     Invariant: two rigid bodies do not share volume. A compliant contact law
     legitimately overlaps a little -- exactly as much as the load requires --
     so the test is not "penetration > 0" but penetration measured against the
     depth that same law needs to carry a KNOWN load (the object at rest on the
     table, one body weight), plus an absolute geometric backstop: penetration
     deeper than the collision wall it is gripping means the finger is through
     the part, which no compliance explains.

  A2 LEVITATION. The object is held up by nothing. Invariant: an object with no
     active contact is in free fall. Measured as vertical acceleration that is
     not -g while the contact set is empty. The reference for "is not -g" is
     the free-fall residual measured in the SAME physics, so integration error
     is subtracted rather than assumed away.

  A3 ENERGY FROM THE INTEGRATOR. Invariant: during contact-free flight the
     object's mechanical energy cannot increase. Reference: the free-fall energy
     drift measured in the same physics.

  A4 EJECTION. Invariant: an object carried by a gripper cannot outrun it. A
     grasped body's speed is bounded by the TCP speed plus the object's own spin
     about the grasp; exceeding that means momentum arrived from somewhere else.

  A5 REWARD-WHILE-INVALID is the headline: the share of reward-bearing steps,
     and the share of RETURN, that any of A1-A4 flags. That is the fraction of
     the score bought with physics that is not real.

EVERY THRESHOLD IS A MEASURED BASELINE, NOT A CONSTANT. :class:`Baseline`
carries references produced by re-runnable probes in the SAME physics arm as the
trace being judged (``artifact_probe.py --baseline``). A signature with no
baseline reports ``UNCALIBRATED`` and is excluded from the verdict rather than
falling back to a number someone once wrote down.

THIS MODULE TOUCHES NO GPU. It consumes a :class:`Trace` of per-step arrays (an
``.npz`` written by the probe) so the analysis is re-runnable, testable on CPU
with synthetic traces, and auditable long after the run.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .analyze import MIN_REPLICATES, spread

# Standard gravity. Used only to form the free-fall reference acceleration; the
# tolerance around it is always a MEASURED baseline, never this constant.
G = 9.80665

# Verdict labels.
FLAGGED = "FLAGGED"
CLEAN = "CLEAN"
UNCALIBRATED = "UNCALIBRATED"
VOID = "VOID"


@dataclass
class Baseline:
    """References measured in the same physics arm, by ``artifact_probe.py --baseline``.

    Every field is a measurement with a stated provenance, and every one of them
    is allowed to be ``None`` -- a signature whose baseline is missing reports
    ``UNCALIBRATED`` instead of silently using a default.
    """

    physics_arm: str = ""

    # A1: penetration depth of the object resting on the table under its own
    # weight. This is what the authored compliant law needs to carry 1 x m g, so
    # it is the natural unit for "how much overlap is legitimate here".
    rest_penetration_m: float | None = None
    # A1 backstop: thinnest collision wall of the object, from its collision
    # geometry. Penetration past this is through-the-part, not compliance.
    wall_thickness_m: float | None = None
    # A2: |a_z + g| for the object in contact-free flight [m/s^2]. The
    # integrator's own error in reproducing free fall.
    freefall_accel_residual: float | None = None
    # A3: |dE| per control step in contact-free flight [J].
    freefall_energy_drift_j: float | None = None
    # A4: noise floor of the finite-differenced speed comparison [m/s].
    velocity_noise_m_s: float | None = None

    # Scene/task constants read off the constructed env, not hard-coded here.
    object_mass_kg: float | None = None
    minimal_height_m: float | None = None
    table_top_z_m: float | None = None

    # Multipliers applied to the baselines. They are declared here, in one place,
    # so a reader can see every judgement call at once. Each is a safety factor
    # over a measured floor, not a physical claim.
    pen_ratio_flag: float = 3.0
    accel_sigma: float = 10.0
    energy_sigma: float = 10.0
    velocity_sigma: float = 10.0

    @classmethod
    def from_json(cls, path: str) -> Baseline:
        with open(path) as fh:
            data = json.load(fh)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=1, sort_keys=True)


@dataclass
class Trace:
    """Per-step rollout record, shape (T, E) unless stated.

    Written by ``artifact_probe.py`` as an ``.npz``; consumed here. Storing the
    trace rather than the verdict is deliberate: the thresholds are arguable and
    the trace is not, so a later pass can re-judge without re-running the GPU.
    """

    # -- identity
    policy: str = ""  # which arm TRAINED the policy
    physics: str = ""  # which arm's physics it is being REPLAYED under
    checkpoint: str = ""
    seed: int = 0

    # -- kinematics
    obj_z: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    obj_speed: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    obj_vz: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    obj_energy_j: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    tcp_speed: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    obj_ang_speed: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    # -- contact geometry, from the solver's own contact set
    # deepest overlap on any contact involving the object [m], >= 0
    pen_obj_max: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    # deepest overlap on gripper<->object contacts specifically [m], >= 0
    pen_grip_max: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    # smallest signed gap over object contacts [m]; +inf when the object has no
    # contact candidate at all (i.e. nothing within the collision margin)
    gap_obj_min: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    # -- reward bookkeeping
    reward: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    done: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    # -- integrity
    dt: float = 0.0
    contact_overflow: bool = False
    pen_channel_live: bool = True
    notes: list[str] = field(default_factory=list)

    ARRAYS = (
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

    def save(self, path: str) -> None:
        meta = {
            "policy": self.policy,
            "physics": self.physics,
            "checkpoint": self.checkpoint,
            "seed": self.seed,
            "dt": self.dt,
            "contact_overflow": self.contact_overflow,
            "pen_channel_live": self.pen_channel_live,
            "notes": self.notes,
        }
        np.savez_compressed(path, meta=json.dumps(meta), **{k: getattr(self, k) for k in self.ARRAYS})

    @classmethod
    def load(cls, path: str) -> Trace:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            arrays = {k: np.asarray(z[k]) for k in cls.ARRAYS if k in z}
        return cls(**meta, **arrays)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.obj_z.shape)  # type: ignore[return-value]


@dataclass
class Signature:
    """One measured signature, its reference, and what it decided."""

    name: str
    value: float | None
    reference: float | None
    reference_basis: str
    verdict: str
    detail: str = ""
    residual_risk: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# episode segmentation
# --------------------------------------------------------------------------


def episode_ids(done: np.ndarray) -> np.ndarray:
    """Label each (t, e) sample with the index of the episode it belongs to.

    ``done[t, e]`` marks the LAST step of an episode, so the label increments on
    the step AFTER a done. Episodes are numbered per env from 0.
    """
    d = np.asarray(done, dtype=bool)
    if d.size == 0:
        return np.zeros_like(d, dtype=np.int64)
    started = np.cumsum(d, axis=0) - d.astype(np.int64)
    return started.astype(np.int64)


def episode_returns(reward: np.ndarray, done: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (returns, mask) per (episode_index, env), completed episodes only.

    An episode still running when the window ends is DISCARDED, not truncated, so
    returns stay comparable across cells with different termination rates.
    """
    r = np.asarray(reward, dtype=np.float64)
    d = np.asarray(done, dtype=bool)
    if r.size == 0:
        return np.zeros((0, 0)), np.zeros((0, 0), dtype=bool)
    ids = episode_ids(d)
    n_ep = int(ids.max()) + 1
    n_env = r.shape[1]
    out = np.zeros((n_ep, n_env))
    complete = np.zeros((n_ep, n_env), dtype=bool)
    for e in range(n_env):
        np.add.at(out[:, e], ids[:, e], r[:, e])
        closed = np.unique(ids[d[:, e], e])
        complete[closed, e] = True
    return out, complete


def high_return_mask(reward: np.ndarray, done: np.ndarray, quantile: float = 0.9) -> np.ndarray:
    """(T, E) mask selecting the steps of the top-``quantile`` completed episodes.

    Marco's question is about penetration DURING HIGH-REWARD EPISODES: a policy
    that occasionally clips through the mug while failing is not exploiting
    anything, and pooling all episodes hides the difference.
    """
    rets, complete = episode_returns(reward, done)
    if not complete.any():
        return np.zeros_like(np.asarray(reward), dtype=bool)
    cut = float(np.quantile(rets[complete], quantile))
    ids = episode_ids(done)
    sel = np.zeros_like(np.asarray(reward), dtype=bool)
    for e in range(sel.shape[1]):
        good = np.where(complete[:, e] & (rets[:, e] >= cut))[0]
        if good.size:
            sel[:, e] = np.isin(ids[:, e], good)
    return sel


# --------------------------------------------------------------------------
# the invalid-configuration masks (A1 - A4)
# --------------------------------------------------------------------------


def valid_sample(done: np.ndarray) -> np.ndarray:
    """Samples whose kinematics, contacts and reward all describe one state.

    The env resets a finished world INSIDE ``step``, so on a ``done`` step the
    object pose read afterwards is already the NEW episode's, while the contact
    set still holds the old episode's solve and the reward belongs to the old
    episode too. The step after a reset is equally unusable for anything built
    on a time difference, because the difference crosses a teleport. Both are
    dropped. On this task that is 2 of every ~150 steps, and dropping them can
    only REMOVE flags, never add any.
    """
    d = np.asarray(done, dtype=bool)
    prev = np.zeros_like(d)
    if d.shape[0] > 1:
        prev[1:] = d[:-1]
    return ~d & ~prev


def lift_mask(tr: Trace, base: Baseline) -> np.ndarray:
    """Steps where the task's reward gate is open (object root above the gate).

    Restricted to samples that describe a single consistent state (see
    :func:`valid_sample`), because a step whose pose and contacts come from
    different episodes cannot be judged either way.
    """
    if base.minimal_height_m is None:
        raise ValueError("Baseline.minimal_height_m is required: it defines what 'earning reward' means")
    return (np.asarray(tr.obj_z) > base.minimal_height_m) & valid_sample(tr.done)


def mask_interpenetration(tr: Trace, base: Baseline) -> np.ndarray | None:
    """A1: gripper overlapping the object by more than the object's own wall.

    The wall thickness is the backstop that no compliance argument can absorb:
    past it the finger is not squeezing the part, it is inside it.
    """
    if not tr.pen_channel_live or base.wall_thickness_m is None or tr.pen_grip_max.size == 0:
        return None
    return (np.asarray(tr.pen_grip_max) > base.wall_thickness_m) & valid_sample(tr.done)


def mask_levitation(tr: Trace, base: Baseline) -> np.ndarray | None:
    """A2: object accelerating unlike free fall while its contact set is empty.

    ``gap_obj_min`` is +inf when the collision pipeline produced no candidate
    involving the object, which is the pipeline's own statement that nothing is
    within reach of touching it.
    """
    if not tr.pen_channel_live or base.freefall_accel_residual is None or tr.gap_obj_min.size == 0:
        return None
    if tr.dt <= 0:
        return None
    vz = np.asarray(tr.obj_vz, dtype=np.float64)
    az = np.zeros_like(vz)
    az[1:] = (vz[1:] - vz[:-1]) / tr.dt
    free = np.isinf(np.asarray(tr.gap_obj_min)) | (np.asarray(tr.gap_obj_min) > 0.0)
    tol = base.accel_sigma * base.freefall_accel_residual
    out = free & (np.abs(az + G) > tol) & valid_sample(tr.done)
    out[0] = False  # no acceleration is defined on the first sample
    return out


def mask_energy_gain(tr: Trace, base: Baseline) -> np.ndarray | None:
    """A3: mechanical energy rising during contact-free flight."""
    if not tr.pen_channel_live or base.freefall_energy_drift_j is None or tr.obj_energy_j.size == 0:
        return None
    e = np.asarray(tr.obj_energy_j, dtype=np.float64)
    de = np.zeros_like(e)
    de[1:] = e[1:] - e[:-1]
    free = np.isinf(np.asarray(tr.gap_obj_min)) | (np.asarray(tr.gap_obj_min) > 0.0)
    tol = base.energy_sigma * base.freefall_energy_drift_j
    out = free & (de > tol) & valid_sample(tr.done)
    out[0] = False
    return out


def mask_ejection(tr: Trace, base: Baseline) -> np.ndarray | None:
    """A4: object outrunning the gripper that is supposedly carrying it.

    Bound: a body held at the TCP moves at the TCP speed plus its own spin about
    the grasp point, ``|v_obj| <= |v_tcp| + |w_obj| * r``. ``r`` is taken as the
    object's own half-extent proxy -- the wall-thickness baseline is not that, so
    the radius comes from the object's measured bounding radius when present and
    the term is dropped (conservatively, making the test HARDER to trip) when it
    is not.
    """
    if base.velocity_noise_m_s is None or tr.tcp_speed.size == 0:
        return None
    v = np.asarray(tr.obj_speed, dtype=np.float64)
    vt = np.asarray(tr.tcp_speed, dtype=np.float64)
    spin = np.asarray(tr.obj_ang_speed, dtype=np.float64) if tr.obj_ang_speed.size else np.zeros_like(v)
    radius = base.wall_thickness_m if base.wall_thickness_m else 0.0
    tol = base.velocity_sigma * base.velocity_noise_m_s
    return (v > (vt + spin * radius + tol)) & valid_sample(tr.done)


def invalid_mask(tr: Trace, base: Baseline) -> tuple[np.ndarray, dict[str, np.ndarray | None]]:
    """Union of the live invalidity masks, plus each one individually.

    A mask that is ``None`` (uncalibrated or channel dead) contributes nothing
    and is reported as such; it is never treated as "clean".
    """
    parts = {
        "interpenetration": mask_interpenetration(tr, base),
        "levitation": mask_levitation(tr, base),
        "energy_gain": mask_energy_gain(tr, base),
        "ejection": mask_ejection(tr, base),
    }
    live = [m for m in parts.values() if m is not None]
    if not live:
        return np.zeros(tr.shape, dtype=bool), parts
    union = np.zeros_like(live[0], dtype=bool)
    for m in live:
        union |= m
    return union, parts


# --------------------------------------------------------------------------
# the signatures
# --------------------------------------------------------------------------


def _pct(x: np.ndarray, q: float) -> float | None:
    return float(np.percentile(x, q)) if x.size else None


def signatures(tr: Trace, base: Baseline, high_quantile: float = 0.9) -> dict[str, Any]:
    """Every signature for one (policy, physics) cell, with its reference."""
    lift = lift_mask(tr, base)
    high = high_return_mask(tr.reward, tr.done, high_quantile)
    lift_high = lift & high
    union, parts = invalid_mask(tr, base)
    sigs: list[Signature] = []

    void_reason = ""
    if tr.contact_overflow:
        void_reason = (
            "the contact/triangle-pair buffer overflowed during this rollout: the contact set is "
            "TRUNCATED, so an empty contact set is not evidence of no contact and a shallow "
            "penetration is not evidence of no penetration"
        )

    def add(
        name: str,
        value: float | None,
        reference: float | None,
        basis: str,
        flagged: bool | None,
        detail: str = "",
        risk: str = "",
    ) -> None:
        if void_reason:
            verdict = VOID
        elif value is None or reference is None or flagged is None:
            verdict = UNCALIBRATED
        else:
            verdict = FLAGGED if flagged else CLEAN
        sigs.append(
            Signature(
                name=name,
                value=value,
                reference=reference,
                reference_basis=basis,
                verdict=verdict,
                detail=detail or (void_reason if void_reason else ""),
                residual_risk=risk,
            )
        )

    # ---- A1 penetration, conditioned on reward being earned -----------------
    pen = np.asarray(tr.pen_grip_max, dtype=np.float64) if tr.pen_grip_max.size else np.zeros((0,))
    pen_lift = pen[lift] if pen.size else np.zeros((0,))
    pen_high = pen[lift_high] if pen.size else np.zeros((0,))
    p95 = _pct(pen_lift, 95)
    ratio = (
        p95 / base.rest_penetration_m
        if (p95 is not None and base.rest_penetration_m not in (None, 0.0))
        else None
    )
    add(
        "pen_at_reward_p95_over_rest",
        ratio,
        base.pen_ratio_flag,
        "P95 gripper<->object overlap over reward-bearing steps, divided by the SAME physics' "
        "measured resting overlap of the object on the table (the depth this compliant law needs "
        "to carry exactly one body weight)",
        None if ratio is None else ratio > base.pen_ratio_flag,
        detail=f"P95 = {p95} m over {int(lift.sum())} reward-bearing steps"
        if p95 is not None
        else "no reward-bearing steps",
        risk="the resting reference carries 1x weight; a legitimate grasp also carries the pads' "
        "squeeze preload, which this reference does not bound. A ratio above the flag is "
        "therefore suggestive on its own and only becomes evidence next to the matched cell.",
    )
    add(
        "pen_at_reward_max_over_wall",
        (float(pen_lift.max()) / base.wall_thickness_m)
        if (pen_lift.size and base.wall_thickness_m)
        else None,
        1.0,
        "deepest gripper<->object overlap while earning reward, divided by the object's thinnest "
        "collision wall; >= 1 means the finger is through the part, which no compliance explains",
        (bool(pen_lift.max() > base.wall_thickness_m) if (pen_lift.size and base.wall_thickness_m) else None),
        detail=f"max = {float(pen_lift.max()) if pen_lift.size else None} m",
        risk="wall thickness is read from the collision mesh, which is a convex decomposition of "
        "the visual mesh; a thin decomposed piece can under-report the true wall.",
    )
    add(
        "pen_at_acquisition_p95",
        _pct(_acquisition_values(pen, lift, tr.done), 95) if pen.size else None,
        base.rest_penetration_m,
        "P95 overlap at the FIRST step of each episode's lift, against the resting overlap. The "
        "acquisition step is where a lift is bought; sinking into the object to buy it is the "
        "canonical exploit and it is invisible in a whole-episode average.",
        None
        if (pen.size == 0 or base.rest_penetration_m in (None, 0.0))
        else bool(
            (_pct(_acquisition_values(pen, lift, tr.done), 95) or 0.0)
            > base.pen_ratio_flag * base.rest_penetration_m
        ),
        risk="an episode whose lift predicate is already true at the window's first step has no "
        "observable acquisition and is skipped.",
    )
    add(
        "pen_high_return_p95",
        _pct(pen_high, 95) if pen.size else None,
        base.rest_penetration_m,
        f"P95 overlap over reward-bearing steps of the top-{100 * (1 - high_quantile):.0f}% "
        "episodes only, against the resting overlap",
        None
        if (pen_high.size == 0 or base.rest_penetration_m in (None, 0.0))
        else bool((_pct(pen_high, 95) or 0.0) > base.pen_ratio_flag * base.rest_penetration_m),
        risk="conditioning on outcome selects on the same variable the claim is about; this "
        "number is descriptive and belongs next to the unconditioned one, never alone.",
    )

    # ---- A2/A3/A4 rates -----------------------------------------------------
    for key, human, risk in (
        (
            "levitation",
            "share of reward-bearing steps with an EMPTY contact set and vertical acceleration "
            "further from -g than the same physics' measured free-fall residual allows",
            "an empty contact set is the collision pipeline's claim, not ground truth; it is "
            "void under buffer overflow, which is checked separately.",
        ),
        (
            "energy_gain",
            "share of reward-bearing steps where mechanical energy ROSE during contact-free "
            "flight, beyond the same physics' measured free-fall energy drift",
            "energy is computed from reported velocity, so a velocity the solver reports but "
            "does not integrate would be missed.",
        ),
        (
            "ejection",
            "share of reward-bearing steps where the object outruns the TCP by more than the "
            "measured finite-difference noise",
            "the bound assumes the object is carried by the gripper; an object legitimately "
            "thrown and re-caught would trip it, which video review must exclude.",
        ),
    ):
        m = parts[key]
        rate = float(m[lift].mean()) if (m is not None and lift.any()) else None
        add(
            f"{key}_rate_at_reward",
            rate,
            0.0,
            human + " -- reference 0: a valid rollout has none of these at all",
            None if rate is None else rate > 0.0,
            detail=f"{int(m[lift].sum()) if m is not None else 0} of {int(lift.sum())} steps",
            risk=risk,
        )

    # ---- A5 the headline ----------------------------------------------------
    n_lift = int(lift.sum())
    exploit_fraction = float(union[lift].mean()) if n_lift else None
    rew = np.asarray(tr.reward, dtype=np.float64)
    pos = np.clip(rew, 0.0, None)
    total_pos = float(pos.sum())
    reward_from_invalid = float(pos[union].sum() / total_pos) if total_pos > 0 else None
    add(
        "exploit_fraction",
        exploit_fraction,
        0.0,
        "share of reward-bearing steps flagged invalid by ANY live signature; reference 0",
        None if exploit_fraction is None else exploit_fraction > 0.0,
        detail=f"{int(union[lift].sum())} of {n_lift} reward-bearing steps",
        risk="the union is only as complete as the live signatures; an uncalibrated signature "
        "cannot flag, so this is a LOWER bound on invalidity, never an upper one.",
    )
    add(
        "reward_from_invalid",
        reward_from_invalid,
        0.0,
        "share of all POSITIVE reward collected at flagged steps -- how much of the score is "
        "bought with physics that is not real; reference 0",
        None if reward_from_invalid is None else reward_from_invalid > 0.0,
        risk="positive reward only; the penalty terms are excluded so the denominator is income, "
        "not net return.",
    )

    rets, complete = episode_returns(tr.reward, tr.done)
    live_names = [k for k, m in parts.items() if m is not None]
    return {
        "policy": tr.policy,
        "physics": tr.physics,
        "checkpoint": tr.checkpoint,
        "seed": tr.seed,
        "steps": int(tr.shape[0]) if tr.obj_z.size else 0,
        "envs": int(tr.shape[1]) if tr.obj_z.size else 0,
        "completed_episodes": int(complete.sum()),
        "mean_return": float(rets[complete].mean()) if complete.any() else None,
        "lift_step_fraction": float(lift.mean()) if lift.size else None,
        "contact_overflow": bool(tr.contact_overflow),
        "pen_channel_live": bool(tr.pen_channel_live),
        "live_signatures": live_names,
        "dead_signatures": [k for k, m in parts.items() if m is None],
        "signatures": [s.as_dict() for s in sigs],
    }


def _acquisition_values(values: np.ndarray, lift: np.ndarray, done: np.ndarray) -> np.ndarray:
    """``values`` sampled at each episode's first lift step.

    The acquisition step is the False->True edge of the lift predicate within an
    episode. An episode that starts already lifted has no observable edge and is
    skipped rather than credited to step 0.
    """
    if values.size == 0 or lift.size == 0:
        return np.zeros((0,))
    ids = episode_ids(done)
    out: list[float] = []
    for e in range(lift.shape[1]):
        col = lift[:, e]
        edge = np.zeros_like(col)
        edge[1:] = col[1:] & ~col[:-1]
        seen: set[int] = set()
        for t in np.where(edge)[0]:
            ep = int(ids[t, e])
            if ep in seen:
                continue
            seen.add(ep)
            out.append(float(values[t, e]))
    return np.asarray(out)


# --------------------------------------------------------------------------
# the 2x2 and its verdict
# --------------------------------------------------------------------------


def retention(matrix: dict[tuple[str, str], list[float]], policy: str, own: str, other: str) -> list[float]:
    """Per-replicate ratio of a policy's return under the OTHER physics to its own.

    Paired per replicate so seed-to-seed level differences cancel; an unpaired
    ratio of means would hide exactly the variance the campaign keeps being
    burned by.
    """
    a = matrix.get((policy, other), [])
    b = matrix.get((policy, own), [])
    n = min(len(a), len(b))
    return [a[i] / b[i] for i in range(n) if b[i] != 0]


# Effect-size floors, in the units of the quantity being compared. They exist
# because the replicate-spread rule alone divides by zero: two replicates that
# happen to agree closely give a within-cell range near zero, and then ANY
# between-arm difference "separates", including one far too small to be the
# phenomenon. These are declared judgements, not measurements:
#   RETENTION_FLOOR  0.05 -- "collapses when replayed under the other physics"
#     means losing a material share of the score. Five points of retained return
#     is the smallest difference this campaign has ever resolved on any metric.
#   EXPLOIT_FLOOR    0.01 -- one paying step in a hundred at an invalid
#     configuration is the smallest rate that could plausibly explain a
#     behavioural difference.
RETENTION_FLOOR = 0.05
EXPLOIT_FLOOR = 0.01


def crossplay_verdict(
    returns: dict[tuple[str, str], list[float]],
    exploit: dict[tuple[str, str], list[float]],
    fixed: str = "fixed",
    adaptive: str = "adaptive",
    retention_floor: float = RETENTION_FLOOR,
    exploit_floor: float = EXPLOIT_FLOOR,
) -> dict[str, Any]:
    """Decide what the 2x2 supports, and refuse to decide on one replicate.

    THREE CONDITIONS, and the verdict names which ones held:

      (i)  ASYMMETRY. The fixed-trained policy retains less of its score when
           replayed under adaptive physics than the adaptive-trained policy does
           in the mirror direction.
      (ii) MECHANISM. In FIXED physics, the fixed-trained policy earns a larger
           share of its reward at invalid configurations than the adaptive-
           trained policy does under the same physics. Comparing both policies
           under ONE physics is what separates "this policy exploits" from
           "this physics penetrates".
      (iii) LOCALITY. The fixed policy's exploit fraction collapses when the
           physics changes -- the states it was farming stop existing.

    (i) alone is brittleness, not exploitation: a policy can transfer badly for
    ordinary distribution-shift reasons. (ii) alone is a physics difference the
    policy never monetized. Only the conjunction is the claim.
    """
    out: dict[str, Any] = {"conditions": {}, "detail": {}}

    def sep(name: str, a: list[float], b: list[float], direction: str, floor: float) -> tuple[bool | None, str]:
        """Is a-vs-b separated, given the within-group spreads? None if too thin.

        A difference must clear BOTH the measured within-cell spread and the
        declared effect-size floor. The floor is what stops two agreeable
        replicates (range ~ 0) from certifying an arbitrarily small difference.
        """
        sa, sb = spread(a), spread(b)
        out["detail"][name] = {"a": sa, "b": sb, "floor": floor}
        if sa.get("n", 0) < MIN_REPLICATES or sb.get("n", 0) < MIN_REPLICATES:
            return None, (
                f"need >= {MIN_REPLICATES} replicates per cell; have "
                f"{sa.get('n', 0)} and {sb.get('n', 0)}. A single-seed difference on this stack is "
                "not evidence (pass 30's same-seed restart diverged 2.4x in reward by iteration 9)."
            )
        delta = sb["mean"] - sa["mean"]
        widest = max(sa["range"], sb["range"])
        bar = max(widest, floor)
        if abs(delta) <= bar:
            return False, (
                f"|delta| {abs(delta):.4g} <= {bar:.4g} "
                f"(widest within-cell range {widest:.4g}, effect floor {floor:.4g})"
            )
        held = (delta > 0) if direction == "b_greater" else (delta < 0)
        return bool(held), (
            f"delta {delta:+.4g} vs widest within-cell range {widest:.4g} and effect floor {floor:.4g}"
        )

    ret_fixed = retention(returns, fixed, fixed, adaptive)
    ret_adapt = retention(returns, adaptive, adaptive, fixed)
    out["retention"] = {fixed: ret_fixed, adaptive: ret_adapt}
    c1, r1 = sep("asymmetry", ret_fixed, ret_adapt, "b_greater", retention_floor)
    out["conditions"]["asymmetry"] = {"held": c1, "reason": r1}

    c2, r2 = sep(
        "mechanism",
        exploit.get((adaptive, fixed), []),
        exploit.get((fixed, fixed), []),
        "b_greater",
        exploit_floor,
    )
    out["conditions"]["mechanism"] = {"held": c2, "reason": r2}

    c3, r3 = sep(
        "locality",
        exploit.get((fixed, adaptive), []),
        exploit.get((fixed, fixed), []),
        "b_greater",
        exploit_floor,
    )
    out["conditions"]["locality"] = {"held": c3, "reason": r3}

    held = [k for k, v in out["conditions"].items() if v["held"] is True]
    unres = [k for k, v in out["conditions"].items() if v["held"] is None]
    if unres:
        out["verdict"] = "UNRESOLVED"
        out["reason"] = f"insufficient replicates for: {', '.join(unres)}"
    elif len(held) == 3:
        out["verdict"] = "ARTIFACT EXPLOITATION SUPPORTED"
        out["reason"] = "asymmetry, mechanism and locality all separated"
    elif "asymmetry" in held:
        out["verdict"] = "ASYMMETRIC TRANSFER WITHOUT AN IDENTIFIED ARTIFACT"
        out["reason"] = (
            "the fixed-trained policy transfers worse, but the invalid-configuration signatures do "
            "not separate. That is brittleness or distribution shift, and it is a real finding: it "
            "says the two timesteppers produce different dynamics, not that the policy learned to "
            "cheat."
        )
    elif held:
        out["verdict"] = "NO TRANSFER ASYMMETRY"
        out["reason"] = (
            f"conditions held: {', '.join(held)}, but the transfer asymmetry did not separate. "
            "The fixed-trained policy survives adaptive physics: whatever the signatures show, it "
            "is not being paid for it."
        )
    else:
        out["verdict"] = "NULL"
        out["reason"] = (
            "no condition separated. Both policies transfer, and the invalid-configuration rates "
            "do not distinguish them. At this horizon and this contact law, fixed-step integration "
            "produced no exploitable artifact that the policy found."
        )
    return out


# --------------------------------------------------------------------------
# video review
# --------------------------------------------------------------------------

VIDEO_GUIDE = """
WATCH THE VIDEOS BEFORE ACCEPTING ANY VERDICT ABOVE. One clip per cell in
videos/, named <policy>_s<seed>_on_<physics>_0000.mp4, all of env_0.

  fixed policy on FIXED physics    -- the cell the exploit is supposed to live
      in. Look for: a finger visibly INSIDE the mug rather than pinching it; the
      mug rising with no visible squeeze; the mug jittering or buzzing while
      "held"; a lift that starts the instant the gripper reaches the mug rather
      than after the jaws close. If exploit_fraction is high and the clip shows
      an ordinary clean grasp, the signatures are measuring something else --
      say so rather than quoting them.

  fixed policy on ADAPTIVE physics -- the collapse cell. Look for: the same
      approach and closure, then the mug staying on the table or squirting out
      sideways. That is the exploit failing to exist. If instead the arm behaves
      completely differently from the start, the transfer failure is upstream of
      contact and the artifact reading is not supported.

  adaptive policy on ADAPTIVE physics -- the control. Should look like whatever
      the adaptive arm actually learned; this is the reference for "what a
      policy at this horizon looks like".

  adaptive policy on FIXED physics -- the mirror. If this one ALSO collapses,
      the asymmetry is not asymmetric and the 2x2 is telling you the two
      timesteppers simply produce different dynamics.

TWO SPECIFIC CONFUSIONS TO RULE OUT BY EYE, because no signature here can:
  * a legitimate toss-and-catch trips the ejection test;
  * a mug scooped between a finger and the table, then carried, is a real grasp
      even though it looks wrong.
""".strip()


# --------------------------------------------------------------------------
# directory-level report
# --------------------------------------------------------------------------


def report(out_dir: str, echo=print) -> dict[str, Any]:
    """Aggregate every ``*.trace.npz`` in ``out_dir`` into the 2x2 and its verdict."""
    traces = sorted(glob.glob(os.path.join(out_dir, "*.trace.npz")))
    cells: list[dict[str, Any]] = []
    returns: dict[tuple[str, str], list[float]] = {}
    exploit: dict[tuple[str, str], list[float]] = {}
    for path in traces:
        tr = Trace.load(path)
        base_path = os.path.join(out_dir, f"baseline_{tr.physics}.json")
        if not os.path.exists(base_path):
            echo(f"  SKIP {os.path.basename(path)}: no baseline for physics '{tr.physics}' at {base_path}")
            continue
        base = Baseline.from_json(base_path)
        s = signatures(tr, base)
        s["trace"] = os.path.basename(path)
        cells.append(s)
        key = (tr.policy, tr.physics)
        if s["mean_return"] is not None:
            returns.setdefault(key, []).append(s["mean_return"])
        ef = next((x["value"] for x in s["signatures"] if x["name"] == "exploit_fraction"), None)
        if ef is not None:
            exploit.setdefault(key, []).append(float(ef))

    verdict = crossplay_verdict(returns, exploit)
    out = {
        "out_dir": os.path.abspath(out_dir),
        "cells": cells,
        "returns": {f"{k[0]}_on_{k[1]}": v for k, v in returns.items()},
        "exploit_fraction": {f"{k[0]}_on_{k[1]}": v for k, v in exploit.items()},
        "crossplay": verdict,
    }

    echo(f"\n{len(cells)} cell trace(s) in {out_dir}")
    echo("\n--- 2x2 mean return (rows = policy trained under, cols = physics replayed under)")
    arms = sorted({k[0] for k in returns} | {k[1] for k in returns})
    echo(f"{'policy':>12} " + "".join(f"{a:>18}" for a in arms))
    for p in arms:
        row = f"{p:>12} "
        for ph in arms:
            v = returns.get((p, ph), [])
            row += f"{(sum(v) / len(v) if v else float('nan')):>13.2f} (n={len(v)})"
        echo(row)
    echo("\n--- exploit fraction (share of reward-bearing steps at an invalid configuration)")
    for k in sorted(exploit):
        v = exploit[k]
        echo(f"  {k[0]:>9} policy on {k[1]:>9} physics : {sum(v) / len(v):.4f}  (n={len(v)})")
    echo(f"\n--- VERDICT: {verdict['verdict']}")
    echo(f"    {verdict['reason']}")
    for name, c in verdict["conditions"].items():
        echo(f"    {name:14} held={c['held']}  {c['reason']}")

    echo("\n--- per-signature detail (value / reference / verdict)")
    for c in cells:
        echo(f"  {c['policy']}_s{c['seed']}_on_{c['physics']}"
             f"   dead: {', '.join(c['dead_signatures']) or 'none'}")
        for s in c["signatures"]:
            echo(f"      {s['name']:32} {str(s['value'])[:12]:>12} vs {str(s['reference'])[:12]:>12}"
                 f"  {s['verdict']}")

    echo("\n" + VIDEO_GUIDE)
    path = os.path.join(out_dir, "artifact_summary.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    echo(f"\nwrote {path}")
    return out
