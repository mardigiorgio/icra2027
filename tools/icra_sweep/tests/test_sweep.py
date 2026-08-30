"""CPU-only tests for the sweep harness. No GPU, no Isaac Lab import.

    ~/Documents/code/icra2027/.venv/bin/python -m pytest tools/icra_sweep/tests -q

Each test constrains a property the harness must have, chosen because violating
it is a failure this project has already paid for. Where a real training log is
available on disk it is used as the fixture, so the parser is tested against the
format that actually ships rather than a hand-written imitation.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from icra_sweep import analyze, gpu, parse  # noqa: E402
from icra_sweep.config import ConfigError, PackingCfg, SweepCfg, VideoCfg  # noqa: E402

BASE = {
    "name": "t",
    "task": "SomeTask-v0",
    "num_steps_per_env": 24,
    "decimation": 4,
    "out_dir": "/tmp/icra_sweep_test",
    "arms": {"fixed": {"solver": "sap"}, "adaptive": {"solver": "sap-adaptive"}},
    "axes": {"seeds": [42, 7, 13], "num_envs": [1024], "iterations": [150]},
}


def cfg(**over):
    data = dict(BASE)
    data.update(over)
    return SweepCfg.from_mapping(data)


# -- (d) the video cadence trap ---------------------------------------------


def test_video_cadence_is_converted_to_env_steps_not_iterations():
    # The CLI counts env steps. Passing the iteration cadence straight through
    # fires the recorder num_steps_per_env times too often.
    v = VideoCfg(every_iterations=10)
    assert v.interval_env_steps(24) == 240
    assert v.interval_env_steps(24) != v.every_iterations
    # The conversion must track num_steps_per_env, not be a fixed constant.
    assert v.interval_env_steps(16) == 160


def test_video_cadence_rejects_zero():
    with pytest.raises(ConfigError):
        VideoCfg(every_iterations=0).interval_env_steps(24)


# -- (b) kill-safety: an interrupted sweep ends on complete matched sets -----


def test_arms_are_the_innermost_axis_so_prefixes_stay_balanced():
    c = cfg()
    cells = c.cells()
    n_arms = len(c.arms)
    assert len(cells) == n_arms * 3
    # Truncating after any whole number of arm-sets leaves the arms balanced --
    # this is what makes a kill land on a matched pair.
    for k in range(1, len(cells) // n_arms + 1):
        prefix = cells[: k * n_arms]
        counts = {a.name: sum(1 for x in prefix if x.arm.name == a.name) for a in c.arms}
        assert len(set(counts.values())) == 1, f"unbalanced after {k} sets: {counts}"


def test_matched_groups_share_everything_but_the_arm():
    for grp in cfg().matched_groups():
        assert len({g.pair_key for g in grp}) == 1
        assert len({g.arm.name for g in grp}) == len(grp)


def test_run_names_are_unique_and_encode_the_cell():
    cells = cfg(axes={"seeds": [1, 2], "num_envs": [1024, 2048], "iterations": [150]}).cells()
    assert len({c.run_name for c in cells}) == len(cells)
    for c in cells:
        assert str(c.num_envs) in c.run_name and f"s{c.seed}" in c.run_name


# -- (c)/(packing) one GPU process unless explicitly told otherwise ----------


def test_packing_is_refused_while_the_sweep_claims_measurement_grade_timing():
    with pytest.raises(ConfigError):
        cfg(packing={"enabled": True, "max_concurrent": 2})  # timing_sensitive defaults to True


def test_packing_allowed_only_with_an_explicit_outcome_only_declaration():
    c = cfg(timing_sensitive=False, packing={"enabled": True, "max_concurrent": 2, "per_run_gb": 12.0})
    assert c.packing.enabled


def test_slot_arithmetic_respects_headroom_and_the_cap():
    p = PackingCfg(enabled=True, per_run_gb=12.0, headroom_gb=4.0, max_concurrent=4)
    # 32.6 GB card, empty: (32.6 - 4) / 12 = 2.38 -> 2 runs.
    assert p.slots(32.6, 0.0) == 2
    # A card already holding one run has no room for two more.
    assert p.slots(32.6, 12.0) == 1
    # Never below one, never above the cap.
    assert p.slots(8.0, 0.0) == 1
    assert PackingCfg(enabled=True, per_run_gb=1.0, headroom_gb=0.0, max_concurrent=2).slots(64.0, 0.0) == 2
    # Disabled means exclusive regardless of how much memory is free.
    assert PackingCfg(enabled=False, per_run_gb=1.0).slots(64.0, 0.0) == 1


# -- (c) the census: an empty device and an unreadable one are not the same --


def _fake_smi(compute_apps: str, mem: str = "32607, 438"):
    def smi(args):
        return compute_apps if any("compute-apps" in a for a in args) else mem

    return smi


def test_empty_device_is_zero_apps_not_a_parse_artifact(monkeypatch):
    # The shell idiom `nvidia-smi ... | grep -c . || echo 0` yields the string
    # "0\n0" on an empty device, which then fails `[ "$n" -eq 0 ]` and never
    # breaks its wait loop. A real training chain sat idle on a free GPU for
    # half an hour on exactly this. Parsing is structural here.
    monkeypatch.setattr(gpu, "_smi", _fake_smi(""))
    c = gpu.census()
    assert c.ok and len(c.apps) == 0 and not c.apps


def test_census_counts_and_attributes_running_apps(monkeypatch):
    monkeypatch.setattr(gpu, "_smi", _fake_smi("111, /usr/bin/python, 11129\n222, /usr/bin/python, 9000\n"))
    c = gpu.census()
    assert [a.pid for a in c.apps] == [111, 222]
    assert c.used_gb_by(c.others({111})) == pytest.approx(9000 / 1024)


def test_an_unreadable_census_is_not_an_empty_device(monkeypatch):
    def boom(args):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu, "_smi", boom)
    c = gpu.census()
    assert not c.ok
    with pytest.raises(RuntimeError):
        gpu.wait_for_capacity(1, set(), 12.0, 4.0, timeout_s=0.1, poll_s=0.01, log=lambda *_: None)


# -- (g) demand normalization: the factor-of-decimation trap -----------------


def test_price_is_invariant_to_the_boundary_convention_but_demand_is_not():
    # ms per accepted substep is a price: it must not depend on whether work is
    # counted per env step or per solver boundary. Demand per boundary must.
    a = parse.normalize_demand(4000.0, 20.0, 10, 24, 4)
    b = parse.normalize_demand(4000.0, 20.0, 10, 24, 1)
    assert a["ms_per_accepted_substep"] == pytest.approx(b["ms_per_accepted_substep"])
    assert a["accepted_substeps_per_boundary"] == pytest.approx(
        b["accepted_substeps_per_boundary"] / 4
    )


def test_both_denominators_are_reported_and_differ_by_exactly_decimation():
    d = parse.normalize_demand(9600.0, 30.0, 10, 24, 4)
    assert d["accepted_substeps_per_env_step"] == pytest.approx(4 * d["accepted_substeps_per_boundary"])
    # 10 iters x 24 steps x 4 = 960 boundaries; 9600 substeps = 10 per boundary.
    assert d["accepted_substeps_per_boundary"] == pytest.approx(10.0)
    assert d["ms_per_accepted_substep"] == pytest.approx(1000.0 * 30.0 / 9600.0)


def test_degenerate_inputs_produce_no_metric_rather_than_a_wrong_one():
    assert parse.normalize_demand(0.0, 20.0, 10, 24, 4) == {}
    assert parse.normalize_demand(100.0, 0.0, 10, 24, 4) == {}


# -- (f) the aggregator refuses to call a single seed a difference ----------


def _rec(arm, seed, reward, n_iters=50):
    r = parse.RunRecord(
        run_name=f"{arm}-s{seed}", arm=arm, num_envs=1024, seed=seed, iterations_requested=150,
        num_steps_per_env=24, decimation=4,
    )
    r.iters = [{"iter": i, "Mean reward": reward, "iter_time_s": 8.0} for i in range(n_iters)]
    return r


def test_single_replicate_per_arm_is_unresolved_not_a_difference():
    recs = [_rec("fixed", 42, 100.0), _rec("adaptive", 42, 250.0)]
    out = analyze.compare_arms(analyze.group(recs), "Mean reward", 1024, 150)
    assert out["verdict"] == "UNRESOLVED"
    assert "delta" not in out


def test_a_difference_smaller_than_the_within_arm_spread_is_not_separated():
    recs = [
        _rec("fixed", 42, 100.0), _rec("fixed", 7, 200.0),      # spread 100
        _rec("adaptive", 42, 160.0), _rec("adaptive", 7, 200.0),  # mean 180 vs 150
    ]
    out = analyze.compare_arms(analyze.group(recs), "Mean reward", 1024, 150)
    assert out["verdict"] == "NOT SEPARATED"
    assert abs(out["delta"]) < out["widest_within_arm_range"]


def test_a_difference_larger_than_the_spread_is_separated():
    recs = [
        _rec("fixed", 42, 100.0), _rec("fixed", 7, 110.0),
        _rec("adaptive", 42, 500.0), _rec("adaptive", 7, 510.0),
    ]
    out = analyze.compare_arms(analyze.group(recs), "Mean reward", 1024, 150)
    assert out["verdict"] == "SEPARATED"


def test_throughput_uses_env_count_so_wider_runs_are_not_penalised():
    # s/iter alone makes a 4x wider run look 3x worse; samples/s is the metric
    # that survives a change of scale.
    narrow, wide = _rec("fixed", 42, 0.0), _rec("fixed", 43, 0.0)
    wide.num_envs = 4096
    for it in wide.iters:
        it["iter_time_s"] = 24.0  # 3x the narrow run's 8.0 s for 4x the samples
    rows = {r["run"]: r for r in analyze.throughput_table([narrow, wide])}
    assert rows["fixed-s43"]["samples_per_s"] > rows["fixed-s42"]["samples_per_s"]


# -- parser, against a real log if one is present ---------------------------

REAL_LOGS = [
    p
    for p in [
        "/tmp/claude-1002/-home-mdigiorgio-Documents-code/"
        "fe8a844e-d1b0-4d64-833c-48934ee6d700/scratchpad/p31_train_fixed_s42.log"
    ]
    if os.path.exists(p)
]


@pytest.mark.skipif(not REAL_LOGS, reason="no campaign training log on this machine")
def test_parses_a_real_training_log_consistently():
    rec = parse.parse_training_log(REAL_LOGS[0])
    assert rec.complete
    assert rec.iterations_completed == rec.iterations_requested
    # Two independently parsed quantities must agree: the printed total wall is
    # at least the sum of the per-iteration times it contains, and not wildly
    # more (the remainder is startup and the final save).
    per_iter_sum = sum(it["iter_time_s"] for it in rec.iters if "iter_time_s" in it)
    assert per_iter_sum <= rec.training_time_s + 1.0
    assert per_iter_sum >= 0.85 * rec.training_time_s
    # Every iteration block carries the fields the analysis depends on.
    assert all("Mean reward" in it for it in rec.iters)
    assert all("iter_time_s" in it for it in rec.iters)


@pytest.mark.skipif(not REAL_LOGS, reason="no campaign training log on this machine")
def test_completion_check_is_not_fooled_by_a_truncated_log(tmp_path):
    assert parse.is_complete(REAL_LOGS[0], 150)
    head = tmp_path / "partial.log"
    with open(REAL_LOGS[0], errors="replace") as fh:
        head.write_text("".join([next(fh) for _ in range(2000)]))
    assert not parse.is_complete(str(head), 150)
    assert not parse.is_complete(str(tmp_path / "missing.log"), 150)
