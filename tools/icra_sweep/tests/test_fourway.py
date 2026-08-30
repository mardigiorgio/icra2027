"""CPU-only tests for the multi-engine (four-way) additions. No GPU, no Isaac import.

    ~/Documents/code/icra2027/.venv/bin/python -m pytest tools/icra_sweep/tests -q

Each test constrains a property the four-engine comparison needs in order to be
worth running at all. The failures they encode are specific:

  * a PhysX arm silently running Newton, because ``--solver`` was emitted anyway
    or because the arm named no engine at all;
  * a cross-engine sweep whose preflight either aborts on the difference that IS
    the experiment, or is silenced and then checks nothing;
  * a reference (refined) arm being mistaken for a fifth training arm;
  * an arm table whose preset names drift away from the task config.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from icra_sweep import physics_arm, preflight  # noqa: E402
from icra_sweep.config import ConfigError, SweepCfg  # noqa: E402

FOUR = {
    "name": "t4",
    "task": "SomeTask-v0",
    "num_steps_per_env": 24,
    "decimation": 4,
    "out_dir": "/tmp/icra_sweep_test4",
    "arms": {
        "physx": {"solver": None, "extra_args": ["physics=physx"], "family": "physx"},
        "mujoco_fixed": {"solver": "mujoco", "extra_args": ["physics=newton_mjwarp"], "family": "mujoco"},
        "sap_fixed": {"solver": "sap", "extra_args": ["physics=newton_mjwarp"], "family": "sap"},
        "sap_adaptive": {
            "solver": "sap-adaptive",
            "extra_args": ["physics=newton_mjwarp_adaptive"],
            "family": "sap",
        },
    },
    "axes": {"seeds": [42, 7], "num_envs": [1024], "iterations": [200]},
    "preflight": {
        "by_family": True,
        "contract_keys": ["contract.control", "contract.spaces", "contract.mdp"],
    },
}


def cfg(**over):
    data = {k: (dict(v) if isinstance(v, dict) else v) for k, v in FOUR.items()}
    data.update(over)
    return SweepCfg.from_mapping(data)


# -- the PhysX arm must not carry a Newton-only flag -------------------------


def test_a_solverless_arm_emits_no_solver_flag():
    # --solver latches onto a Newton solver_cfg and raises when there is none,
    # which is exactly the PhysX case. Emitting it anyway kills the arm at
    # startup, and the startup-abort guard would then kill the whole sweep.
    c = cfg()
    physx = next(a for a in c.arms if a.name == "physx")
    assert "--solver" not in physx.cli_args()
    assert physx.cli_args() == ["physics=physx"]


def test_a_newton_arm_still_emits_its_solver_flag():
    c = cfg()
    sap = next(a for a in c.arms if a.name == "sap_adaptive")
    assert sap.cli_args()[:2] == ["--solver", "sap-adaptive"]


def test_an_arm_that_names_no_engine_is_refused():
    # solver: null with no extra_args selects nothing, so the run would silently
    # use the task's DEFAULT preset while being recorded under another name --
    # the sweep would then compare one engine with itself.
    arms = dict(FOUR["arms"])
    arms["ghost"] = {"solver": None, "family": "physx"}
    with pytest.raises(ConfigError, match="selects no physics"):
        cfg(arms=arms)


# -- preflight across engines ------------------------------------------------


def test_multi_family_arms_require_by_family_preflight():
    with pytest.raises(ConfigError, match="more than one physics family"):
        cfg(preflight={"by_family": False, "contract_keys": ["contract.control"]})


def test_by_family_preflight_without_a_contract_is_refused():
    # Silencing the cross-engine diff without naming what must still match is a
    # comparison with nothing held fixed.
    with pytest.raises(ConfigError, match="checks nothing across engines"):
        cfg(preflight={"by_family": True, "contract_keys": []})


def test_within_family_differences_still_abort_but_cross_family_ones_do_not():
    # The two SAP arms differing in a tolerance is a confound and must surface.
    # The SAP and MuJoCo arms differing in their contact law is the experiment.
    dumps = {
        "sap_fixed": {"engine": {"sap": {"optimality_rel_tol": 1e-6}},
                      "contract": {"control": {"control_hz": 30.0}}},
        "sap_adaptive": {"engine": {"sap": {"optimality_rel_tol": 1e-8}},
                         "contract": {"control": {"control_hz": 30.0}}},
        "mujoco_fixed": {"engine": {"mujoco": {"solref": [0.02, 1.0]}},
                         "contract": {"control": {"control_hz": 30.0}}},
    }
    fams = {"sap": ["sap_fixed", "sap_adaptive"], "mujoco": ["mujoco_fixed"]}
    rep = preflight.compare_by_family(dumps, fams, expected=(), contract_keys=("contract.control",))
    assert not rep["ok"]
    assert any("optimality_rel_tol" in k for k in rep["unexpected"])
    # The MuJoCo-only solref key must NOT be reported: it never enters a
    # within-family diff and it is not part of the contract.
    assert not any("solref" in k for k in rep["unexpected"])


def test_a_contract_key_that_differs_across_engines_is_fatal():
    dumps = {
        "sap_fixed": {"contract": {"control": {"control_hz": 30.0}}},
        "mujoco_fixed": {"contract": {"control": {"control_hz": 60.0}}},
    }
    fams = {"sap": ["sap_fixed"], "mujoco": ["mujoco_fixed"]}
    rep = preflight.compare_by_family(dumps, fams, expected=(), contract_keys=("contract.control",))
    assert not rep["ok"]
    assert "[contract] contract.control.control_hz" in rep["unexpected"]


def test_a_contract_key_missing_on_one_engine_is_not_agreement():
    # An engine that cannot report the control rate has not demonstrated it
    # matches; treating absent-vs-absent as equal is how a check evaporates.
    dumps = {
        "sap_fixed": {"contract": {"control": {"control_hz": 30.0}}},
        "physx": {"contract": {"control": {}}},
    }
    fams = {"sap": ["sap_fixed"], "physx": ["physx"]}
    rep = preflight.compare_by_family(dumps, fams, expected=(), contract_keys=("contract.control",))
    assert not rep["ok"]
    assert rep["unexpected"]["[contract] contract.control.control_hz"]["physx"] == "<absent>"


def test_a_contract_key_absent_everywhere_is_reported_not_passed():
    dumps = {"a": {"contract": {"control": {"control_hz": 30.0}}},
             "b": {"contract": {"control": {"control_hz": 30.0}}}}
    fams = {"x": ["a"], "y": ["b"]}
    rep = preflight.compare_by_family(dumps, fams, expected=(), contract_keys=("contract.mdp",))
    assert not rep["ok"]
    assert rep["contract_keys_absent"] == ["contract.mdp"]


# -- the arm table -----------------------------------------------------------


def test_every_fourway_arm_is_in_the_table_and_trainable():
    for name in physics_arm.FOURWAY:
        arm = physics_arm.get(name)
        assert arm.trainable, f"{name} is a reference arm and must not be trained in"


def test_only_the_physx_arm_omits_the_solver_latch():
    solverless = {n for n, a in physics_arm.ARMS.items() if a.solver is None}
    assert solverless == {"physx"}


def test_reference_arms_are_not_trainable_and_refine_their_own_family():
    for family, ref_name in physics_arm.REFERENCE_FOR.items():
        ref = physics_arm.get(ref_name)
        assert not ref.trainable
        assert ref.family == family, "a reference must refine the engine it is a reference for"
        assert ref.substeps_override is not None
        assert ref.substeps_override > (ref.expected_substeps or 0), "a reference must be FINER"


def test_the_two_sap_arms_are_one_family_and_mujoco_is_another():
    # Both are reached through the Newton manager, but they solve against
    # different constitutive laws, so they may not be diffed field for field.
    fams = physics_arm.families(physics_arm.FOURWAY)
    assert fams["sap"] == ["sap_fixed", "sap_adaptive"]
    assert fams["mujoco"] == ["mujoco_fixed"]
    assert fams["physx"] == ["physx"]


def test_an_unknown_arm_name_is_an_error_not_a_default():
    with pytest.raises(physics_arm.UnknownArm):
        physics_arm.get("sap_adptive")


def test_train_args_lead_with_the_preset_every_time():
    # --solver is applied AFTER preset resolution and raises if the resolved
    # physics is not Newton, so the preset must always be named.
    for arm in physics_arm.ARMS.values():
        assert arm.train_args()[0].startswith("physics=")
