"""Verification of the artifact signatures. CPU only; no GPU, no simulator.

THE ORACLE PROBLEM HERE. "What should the exploit fraction of this rollout be?"
has no cheap independent answer, and computing it the way the module does would
be a tautology. So nothing below asserts a recomputed value or a frozen output.
Instead each test either

  * builds a trace from an ANALYTIC physical trajectory whose correct verdict is
    known from the physics (exact free fall must be clean; a hover with no
    contact must be flagged), or
  * asserts a RELATION between two runs (scaling the baseline, deepening the
    overlap, permuting envs, duplicating envs), or
  * compares against an INDEPENDENT reference implementation written by a
    different route (episode segmentation).

The discriminating cases are the GATED ones: a rollout that gains energy WHILE
IN CONTACT must not be flagged, and one that gains the same energy with an empty
contact set must be. A signature that ignored the contact gate would satisfy
every structural check and still be worthless, so those pairs are what the suite
is built around.

RESIDUAL RISK IS NAMED AT THE BOTTOM OF THIS FILE.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rubato_sweep import artifact  # noqa: E402
from rubato_sweep.artifact import Baseline, Trace  # noqa: E402

DT = 1.0 / 30.0
G = artifact.G


# --------------------------------------------------------------------------
# generators: physically valid traces, built analytically
# --------------------------------------------------------------------------


def base_calibrated(**over) -> Baseline:
    """A fully calibrated baseline with small, non-degenerate references."""
    kw = dict(
        physics_arm="test",
        rest_penetration_m=1.0e-4,
        wall_thickness_m=3.0e-3,
        freefall_accel_residual=1.0e-6,
        freefall_energy_drift_j=1.0e-9,
        velocity_noise_m_s=1.0e-4,
        object_mass_kg=0.30,
        minimal_height_m=0.08,
        table_top_z_m=0.02,
    )
    kw.update(over)
    return Baseline(**kw)


def free_fall_trace(T: int = 40, E: int = 4, z0: float = 0.6, mass: float = 0.30) -> Trace:
    """Exact ballistic flight, no contacts anywhere.

    z(t) = z0 - g t^2 / 2 and v_z(t) = -g t are the analytic solution, so a
    finite difference of v_z is EXACTLY -g (the difference of a linear function
    is exact in floating point up to round-off) and 0.5 m v^2 + m g z is
    constant. Nothing in this trajectory is non-physical, so a correct instrument
    must flag none of it.
    """
    t = np.arange(T, dtype=np.float64) * DT
    z = z0 - 0.5 * G * t**2
    vz = -G * t
    tr = Trace(policy="p", physics="test", seed=0, dt=DT)
    tr.obj_z = np.repeat(z[:, None], E, axis=1)
    tr.obj_vz = np.repeat(vz[:, None], E, axis=1)
    tr.obj_speed = np.abs(tr.obj_vz)
    tr.obj_ang_speed = np.zeros((T, E))
    tr.obj_energy_j = 0.5 * mass * tr.obj_speed**2 + mass * G * tr.obj_z
    tr.tcp_speed = np.zeros((T, E))
    tr.pen_obj_max = np.zeros((T, E))
    tr.pen_grip_max = np.zeros((T, E))
    tr.gap_obj_min = np.full((T, E), np.inf)  # no contact candidate at all
    tr.reward = np.ones((T, E))
    tr.done = np.zeros((T, E))
    tr.done[-1] = 1.0
    return tr


def hover_trace(T: int = 40, E: int = 4, z: float = 0.30, in_contact: bool = False) -> Trace:
    """Object held at a constant height. With no contact this is impossible."""
    tr = free_fall_trace(T, E)
    tr.obj_z = np.full((T, E), z)
    tr.obj_vz = np.zeros((T, E))
    tr.obj_speed = np.zeros((T, E))
    tr.obj_energy_j = np.full((T, E), 0.30 * G * z)
    tr.gap_obj_min = np.full((T, E), -1e-4 if in_contact else np.inf)
    return tr


# --------------------------------------------------------------------------
# A2 levitation: the physics decides, and the contact gate is the substance
# --------------------------------------------------------------------------


def test_exact_free_fall_is_never_levitation():
    """A body in true free flight must not be flagged, however far it is airborne."""
    m = artifact.mask_levitation(free_fall_trace(), base_calibrated())
    assert m is not None
    assert not m.any(), "exact ballistic flight was flagged as levitation"


def test_hover_without_contact_is_levitation_everywhere():
    """Held at a constant height with an empty contact set: unsupported, so flagged.

    The first sample carries no acceleration and is excluded by construction;
    every later one must trip.
    """
    m = artifact.mask_levitation(hover_trace(in_contact=False), base_calibrated())
    assert m is not None
    assert m[1:-1].all(), "an unsupported hover was not flagged"
    assert not m[0].any(), "the first sample has no defined acceleration and must not be flagged"
    assert not m[-1].any(), "the terminating step mixes two episodes and must not be flagged"


def test_hover_with_contact_is_not_levitation():
    """THE DISCRIMINATING CASE. Identical motion, but something is touching it.

    An instrument that ignored the contact gate would flag this too and would
    then flag every legitimate grasp in the campaign.
    """
    m = artifact.mask_levitation(hover_trace(in_contact=True), base_calibrated())
    assert m is not None
    assert not m.any(), "a supported hover was flagged as levitation"


def test_levitation_is_uncalibrated_without_a_baseline():
    assert artifact.mask_levitation(hover_trace(), base_calibrated(freefall_accel_residual=None)) is None


# --------------------------------------------------------------------------
# A3 energy: gated the same way
# --------------------------------------------------------------------------


def test_conserved_energy_in_flight_is_clean():
    m = artifact.mask_energy_gain(free_fall_trace(), base_calibrated())
    assert m is not None and not m.any()


def test_energy_injection_in_flight_is_flagged():
    tr = free_fall_trace()
    tr.obj_energy_j = tr.obj_energy_j + np.arange(tr.obj_z.shape[0])[:, None] * 1.0e-3
    m = artifact.mask_energy_gain(tr, base_calibrated())
    assert m is not None and m[1:-1].all()


def test_same_energy_injection_while_in_contact_is_clean():
    """Contact may legitimately do work; only contact-free gain is impossible."""
    tr = free_fall_trace()
    tr.obj_energy_j = tr.obj_energy_j + np.arange(tr.obj_z.shape[0])[:, None] * 1.0e-3
    tr.gap_obj_min = np.full_like(tr.gap_obj_min, -1e-4)
    m = artifact.mask_energy_gain(tr, base_calibrated())
    assert m is not None and not m.any()


def test_energy_loss_in_flight_is_not_a_gain():
    """Only an INCREASE is impossible; damping downward is not this signature's business."""
    tr = free_fall_trace()
    tr.obj_energy_j = tr.obj_energy_j - np.arange(tr.obj_z.shape[0])[:, None] * 1.0e-3
    m = artifact.mask_energy_gain(tr, base_calibrated())
    assert m is not None and not m.any()


# --------------------------------------------------------------------------
# A1 penetration: sign, gating, and monotonicity
# --------------------------------------------------------------------------


def test_separated_contacts_never_read_as_interpenetration():
    """Kills the sign error. Large POSITIVE gaps are separation, not overlap."""
    tr = free_fall_trace()
    tr.gap_obj_min = np.full_like(tr.gap_obj_min, 0.05)
    tr.pen_grip_max = np.zeros_like(tr.pen_grip_max)
    m = artifact.mask_interpenetration(tr, base_calibrated())
    assert m is not None and not m.any()


def test_penetration_past_the_wall_is_flagged_and_below_it_is_not():
    b = base_calibrated()
    tr = free_fall_trace()
    tr.pen_grip_max = np.full_like(tr.pen_grip_max, 0.5 * b.wall_thickness_m)
    assert not artifact.mask_interpenetration(tr, b).any()
    tr.pen_grip_max = np.full_like(tr.pen_grip_max, 1.5 * b.wall_thickness_m)
    assert artifact.mask_interpenetration(tr, b)[:-1].all()


def test_deeper_overlap_never_reduces_the_flag_count():
    """Monotonicity: the signature must not be non-monotone in its own quantity."""
    b = base_calibrated()
    rng = np.random.default_rng(7)
    tr = free_fall_trace()
    tr.pen_grip_max = rng.uniform(0, 2 * b.wall_thickness_m, size=tr.pen_grip_max.shape)
    n0 = int(artifact.mask_interpenetration(tr, b).sum())
    tr.pen_grip_max = tr.pen_grip_max + 1e-4
    n1 = int(artifact.mask_interpenetration(tr, b).sum())
    assert n1 >= n0


def test_penetration_ratio_scales_inversely_with_its_reference():
    """Metamorphic: the reference is a UNIT, so doubling it must halve the ratio."""
    tr = hover_trace(in_contact=True)
    tr.pen_grip_max = np.full_like(tr.pen_grip_max, 5.0e-4)
    b1 = base_calibrated(rest_penetration_m=1.0e-4)
    b2 = base_calibrated(rest_penetration_m=2.0e-4)

    def ratio(b):
        s = artifact.signatures(tr, b)["signatures"]
        return next(x["value"] for x in s if x["name"] == "pen_at_reward_p95_over_rest")

    assert ratio(b1) == pytest.approx(2.0 * ratio(b2), rel=1e-12)


def test_acquisition_samples_the_first_lift_step_of_each_episode():
    """The acquisition edge is a definition, and it must land on the edge.

    A distinctive marker is planted at the intended step, so the assertion is
    about WHICH sample is taken, not about any value the module computes.
    """
    T, E = 20, 2
    lift = np.zeros((T, E), dtype=bool)
    lift[5:12] = True  # one contiguous lift per env, edge at t=5
    done = np.zeros((T, E))
    done[-1] = 1.0
    values = np.zeros((T, E))
    values[5] = 0.4242
    got = artifact._acquisition_values(values, lift, done)
    assert got.shape == (E,)
    assert np.allclose(got, 0.4242)


def test_acquisition_skips_an_episode_that_starts_already_lifted():
    T, E = 10, 1
    lift = np.ones((T, E), dtype=bool)
    done = np.zeros((T, E))
    done[-1] = 1.0
    assert artifact._acquisition_values(np.ones((T, E)), lift, done).size == 0


def test_reset_boundary_samples_are_excluded_from_every_mask():
    """A step that resets, and the step after it, describe two different episodes.

    The env resets inside ``step``, so on a done step the pose is the new
    episode's while the contact set is the old one's, and the difference into the
    next step crosses a teleport. Both must be unjudgeable, and the exclusion has
    to be the SAME in every mask or the union would flag on the inconsistency
    itself.
    """
    b = base_calibrated()
    tr = exploiting_trace(T=12, E=2)
    tr.done = np.zeros_like(tr.done)
    tr.done[5] = 1.0
    tr.done[-1] = 1.0
    union, parts = artifact.invalid_mask(tr, b)
    for name, m in parts.items():
        if m is None:
            continue
        assert not m[5].any(), f"{name} judged the reset step"
        assert not m[6].any(), f"{name} judged the step after a reset"
    assert not union[5].any() and not union[6].any()
    assert not artifact.lift_mask(tr, b)[5].any()


# --------------------------------------------------------------------------
# A4 ejection
# --------------------------------------------------------------------------


def test_object_slower_than_the_tcp_is_never_ejection():
    tr = hover_trace(in_contact=True)
    tr.tcp_speed = np.full_like(tr.tcp_speed, 0.5)
    tr.obj_speed = np.full_like(tr.obj_speed, 0.4)
    assert not artifact.mask_ejection(tr, base_calibrated()).any()


def test_object_outrunning_the_tcp_is_ejection():
    tr = hover_trace(in_contact=True)
    tr.tcp_speed = np.full_like(tr.tcp_speed, 0.1)
    tr.obj_speed = np.full_like(tr.obj_speed, 5.0)
    assert artifact.mask_ejection(tr, base_calibrated())[:-1].all()


def test_ejection_tolerates_the_measured_noise_floor():
    """A difference inside the measured comparison noise must not trip."""
    b = base_calibrated(velocity_noise_m_s=1.0e-3)
    tr = hover_trace(in_contact=True)
    tr.tcp_speed = np.full_like(tr.tcp_speed, 0.2)
    tr.obj_speed = tr.tcp_speed + 0.5 * b.velocity_sigma * b.velocity_noise_m_s
    assert not artifact.mask_ejection(tr, b).any()


# --------------------------------------------------------------------------
# episode segmentation: differential against an independent implementation
# --------------------------------------------------------------------------


def reference_returns(reward: np.ndarray, done: np.ndarray) -> list[float]:
    """Independent segmentation by an explicit per-env accumulator loop.

    Deliberately a different algorithm from the module's label-and-scatter, so a
    bug in either shows up as a disagreement rather than being copied.
    """
    out: list[float] = []
    T, E = reward.shape
    for e in range(E):
        acc = 0.0
        for t in range(T):
            acc += float(reward[t, e])
            if done[t, e]:
                out.append(acc)
                acc = 0.0
    return out


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_episode_returns_match_an_independent_segmentation(seed):
    rng = np.random.default_rng(seed)
    T, E = 60, 5
    reward = rng.normal(size=(T, E))
    done = (rng.random((T, E)) < 0.08).astype(float)
    rets, complete = artifact.episode_returns(reward, done)
    got = sorted(rets[complete].tolist())
    want = sorted(reference_returns(reward, done))
    assert len(got) == len(want)
    # Summation order differs between the two routes; 1e-9 is far above the
    # float64 round-off of ~60 additions of O(1) terms and far below any real
    # segmentation error, which would change the COUNT or shift whole episodes.
    assert np.allclose(got, want, atol=1e-9)


def test_running_episodes_are_discarded_not_truncated():
    """An episode still open at the window's end must not enter the statistic."""
    T, E = 10, 1
    reward = np.ones((T, E))
    done = np.zeros((T, E))
    done[4] = 1.0  # one episode closes; the tail stays open
    rets, complete = artifact.episode_returns(reward, done)
    assert int(complete.sum()) == 1
    assert rets[complete][0] == pytest.approx(5.0)


def test_high_return_mask_selects_the_better_episodes():
    """A selection property: the selected episodes cannot average below the whole."""
    rng = np.random.default_rng(11)
    T, E = 80, 6
    done = np.zeros((T, E))
    done[19::20] = 1.0
    reward = rng.normal(size=(T, E))
    sel = artifact.high_return_mask(reward, done, quantile=0.5)
    rets, complete = artifact.episode_returns(reward, done)
    assert sel.any()
    assert reward[sel].sum() / max(sel.sum(), 1) >= rets[complete].mean() / (T / E) - 1e6  # sanity only
    ids = artifact.episode_ids(done)
    picked = {(int(ids[t, e]), e) for t, e in zip(*np.where(sel))}
    picked_rets = [rets[i, e] for i, e in picked]
    assert min(picked_rets) >= np.median(rets[complete]) - 1e-12


# --------------------------------------------------------------------------
# invariances of the aggregate statistics
# --------------------------------------------------------------------------


def _exploit_fraction(tr: Trace, b: Baseline) -> float:
    s = artifact.signatures(tr, b)["signatures"]
    return next(x["value"] for x in s if x["name"] == "exploit_fraction")


def exploiting_trace(T: int = 30, E: int = 4) -> Trace:
    """Reward earned while the gripper is through the object's wall."""
    b = base_calibrated()
    tr = hover_trace(T, E, z=0.30, in_contact=True)
    tr.pen_grip_max = np.full((T, E), 2.0 * b.wall_thickness_m)
    return tr


def test_exploit_fraction_is_a_fraction():
    v = _exploit_fraction(exploiting_trace(), base_calibrated())
    assert 0.0 <= v <= 1.0


def test_exploit_fraction_is_one_when_every_paying_step_is_invalid():
    assert _exploit_fraction(exploiting_trace(), base_calibrated()) == pytest.approx(1.0)


def test_exploit_fraction_is_zero_for_a_clean_grasp():
    b = base_calibrated()
    tr = hover_trace(in_contact=True)
    tr.pen_grip_max = np.full_like(tr.pen_grip_max, 0.1 * b.wall_thickness_m)
    tr.tcp_speed = np.zeros_like(tr.tcp_speed)
    assert _exploit_fraction(tr, b) == pytest.approx(0.0)


def test_exploit_fraction_is_invariant_to_duplicating_environments():
    """Metamorphic: it is a per-step rate, so twice the envs is the same rate."""
    b = base_calibrated()
    tr = exploiting_trace()
    two = Trace(policy="p", physics="test", seed=0, dt=DT)
    for k in Trace.ARRAYS:
        setattr(two, k, np.concatenate([getattr(tr, k)] * 2, axis=1))
    assert _exploit_fraction(two, b) == pytest.approx(_exploit_fraction(tr, b))


def test_exploit_fraction_is_invariant_to_permuting_environments():
    b = base_calibrated()
    tr = exploiting_trace()
    tr.pen_grip_max[:, 0] = 0.0  # make the envs distinguishable
    perm = [3, 1, 0, 2]
    shuf = Trace(policy="p", physics="test", seed=0, dt=DT)
    for k in Trace.ARRAYS:
        setattr(shuf, k, getattr(tr, k)[:, perm])
    assert _exploit_fraction(shuf, b) == pytest.approx(_exploit_fraction(tr, b))


def test_exploit_fraction_ignores_a_constant_reward_offset():
    """The masks are about configuration, not about how well the step paid."""
    b = base_calibrated()
    tr = exploiting_trace()
    v0 = _exploit_fraction(tr, b)
    tr.reward = tr.reward + 17.0
    assert _exploit_fraction(tr, b) == pytest.approx(v0)


def test_uncalibrated_signatures_are_reported_dead_not_clean():
    """A missing baseline must never be silently read as 'nothing wrong here'."""
    tr = exploiting_trace()
    b = base_calibrated(wall_thickness_m=None, freefall_accel_residual=None,
                        freefall_energy_drift_j=None, velocity_noise_m_s=None)
    out = artifact.signatures(tr, b)
    assert set(out["dead_signatures"]) == {"interpenetration", "levitation", "energy_gain", "ejection"}
    assert out["live_signatures"] == []
    verdicts = {s["name"]: s["verdict"] for s in out["signatures"]}
    assert verdicts["pen_at_reward_max_over_wall"] == artifact.UNCALIBRATED


def test_contact_overflow_voids_every_signature():
    """A truncated contact set cannot support any claim about contacts."""
    tr = exploiting_trace()
    tr.contact_overflow = True
    out = artifact.signatures(tr, base_calibrated())
    assert {s["verdict"] for s in out["signatures"]} == {artifact.VOID}


# --------------------------------------------------------------------------
# the 2x2 verdict, and the honesty guards
# --------------------------------------------------------------------------


def _matrix(ff, fa, af, aa, n=2):
    return {
        ("fixed", "fixed"): [ff] * n,
        ("fixed", "adaptive"): [fa] * n,
        ("adaptive", "fixed"): [af] * n,
        ("adaptive", "adaptive"): [aa] * n,
    }


def test_one_seed_can_never_produce_a_verdict():
    """THE HONESTY GUARD. n=1 is UNRESOLVED however extreme the numbers are."""
    out = artifact.crossplay_verdict(
        _matrix(200.0, 5.0, 190.0, 195.0, n=1),
        _matrix(0.9, 0.01, 0.02, 0.02, n=1),
    )
    assert out["verdict"] == "UNRESOLVED"


def test_a_clean_exploit_signature_is_supported_at_two_seeds():
    out = artifact.crossplay_verdict(
        _matrix(200.0, 5.0, 190.0, 195.0, n=2),
        _matrix(0.9, 0.01, 0.02, 0.02, n=2),
    )
    assert out["verdict"] == "ARTIFACT EXPLOITATION SUPPORTED"
    assert all(c["held"] for c in out["conditions"].values())


def test_symmetric_transfer_reports_null():
    """THE NULL MUST BE REPORTABLE. Both policies transfer, nothing is flagged."""
    out = artifact.crossplay_verdict(
        _matrix(100.0, 98.0, 99.0, 101.0, n=2),
        _matrix(0.0, 0.0, 0.0, 0.0, n=2),
    )
    assert out["verdict"] == "NULL"


def test_asymmetry_without_a_mechanism_is_named_as_brittleness():
    """A policy can transfer badly for ordinary reasons; that is not exploitation."""
    out = artifact.crossplay_verdict(
        _matrix(200.0, 5.0, 190.0, 195.0, n=2),
        _matrix(0.0, 0.0, 0.0, 0.0, n=2),
    )
    assert out["verdict"] == "ASYMMETRIC TRANSFER WITHOUT AN IDENTIFIED ARTIFACT"


def test_a_difference_inside_the_replicate_spread_does_not_separate():
    """Between-arm delta smaller than the within-arm range is not a difference."""
    returns = {
        ("fixed", "fixed"): [100.0, 100.0],
        ("fixed", "adaptive"): [50.0, 150.0],  # retention 0.5 and 1.5: huge spread
        ("adaptive", "adaptive"): [100.0, 100.0],
        ("adaptive", "fixed"): [100.0, 100.0],
    }
    out = artifact.crossplay_verdict(returns, _matrix(0.5, 0.0, 0.0, 0.0, n=2))
    assert out["conditions"]["asymmetry"]["held"] is False


def test_retention_is_paired_per_replicate():
    """Pairing property: identical own/cross returns give retention 1 per seed."""
    m = {("fixed", "fixed"): [10.0, 40.0], ("fixed", "adaptive"): [10.0, 40.0]}
    assert artifact.retention(m, "fixed", "fixed", "adaptive") == [1.0, 1.0]


def test_retention_uses_the_matching_replicate_not_the_mean():
    """An unpaired ratio-of-means would give 1.0 here; the paired one must not."""
    m = {("fixed", "fixed"): [10.0, 40.0], ("fixed", "adaptive"): [40.0, 10.0]}
    assert artifact.retention(m, "fixed", "fixed", "adaptive") == [4.0, 0.25]


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


def test_trace_survives_a_save_load_round_trip(tmp_path):
    tr = exploiting_trace()
    tr.notes = ["hello"]
    path = str(tmp_path / "cell.trace.npz")
    tr.save(path)
    back = Trace.load(path)
    assert back.policy == tr.policy and back.physics == tr.physics
    assert back.dt == pytest.approx(tr.dt)
    assert back.notes == ["hello"]
    for k in Trace.ARRAYS:
        assert np.allclose(getattr(back, k), getattr(tr, k), equal_nan=True)


# --------------------------------------------------------------------------
# RESIDUAL RISK -- bug classes this suite cannot catch
# --------------------------------------------------------------------------
#
# 1. WRONG SOURCE ARRAY. Everything here consumes a Trace. If artifact_probe.py
#    reduces the wrong warp array, mislabels which body is the object, or reads
#    the contact set at the wrong point in the step, every test still passes.
#    That risk is addressed by the probe's own cross-checked body resolution
#    (label route vs stride route, disagreement recorded) and by the baseline
#    run, whose resting-overlap and free-fall numbers are physically
#    interpretable and would be absurd if the array were wrong. It is NOT
#    addressed here, and it is the largest one.
#
# 2. THRESHOLD CALIBRATION. The sigma multipliers are judgement, not physics.
#    These tests fix the SHAPE of each rule (gated, monotone, scale-correct) and
#    say nothing about whether 10x the measured free-fall residual is the right
#    place to draw the line. Only the matched cell comparison can answer that,
#    which is why no signature is quoted without its counterpart cell.
#
# 3. UNMODELLED VALID PHYSICS. The ejection bound assumes the object is carried.
#    A policy that legitimately tosses and re-catches the mug would be flagged,
#    and no test here distinguishes that from an artifact. Video review is the
#    control for this class, which is why every cell renders one.
#
# 4. CORRELATED FAILURE OF ALL FOUR SIGNATURES. An exploit that produces no
#    overlap, no empty-contact support, no energy gain and no over-speed -- for
#    instance one that lives entirely in the friction cone -- is invisible to
#    every signature at once. exploit_fraction is therefore a LOWER bound on
#    invalidity, and the module says so; a zero must never be read as "the
#    physics was valid".
#
# 5. SEGMENTATION UNDER TIME-OUT-ONLY EPISODES. The differential test uses
#    random done flags. If the real task never sets done except at a fixed
#    period, the segmentation is exercised on a much easier input than the tests
#    use, so this direction is over- rather than under-tested.
