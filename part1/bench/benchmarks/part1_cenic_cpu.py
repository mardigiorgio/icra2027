# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experiment 4 CPU arm: the CENIC reference implementation (Drake's
``integration_scheme="cenic"``, shipped since Drake 1.56) on the hard
clutter scene: throughput vs number of concurrent worlds.

Scene parity: constants copied from part1/scenes/cenic_scenes.py and the
initial conditions come from the SAME shared generator both stacks import
(part1/scenes/clutter_lattice.py), world i seeded BASE_SEED + i, so world
sets match the GPU benches exactly. Protocol matches part1_scaling.py:
0.2 s warm-up, 2 s timed, accuracy 1e-3, max step 0.1 s.

Timing (the honest form): every worker builds and warms up, reports
ready, and all workers start the timed window on one shared GO signal;
the parent measures ONE makespan from GO to the last worker's finish.
Per-worker self-timing under staggered starts overstated contended
throughput. Two workloads per N:
  free      -- each world advances its 2 s independently (Monte Carlo)
  lockstep  -- all worlds barrier at every 0.1 s boundary, the way an RL
               batch must advance
Run on an idle host; workers are not niced.

RUN (the drake venv, NOT the icra venv; single line)
  ~/Documents/code/drake-cenic/.venv/bin/python part1/bench/benchmarks/part1_cenic_cpu.py
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)

from part1.scenes.clutter_lattice import BASE_SEED, axis_angle_quat, clutter_lattice  # noqa: E402

# part1/scenes/cenic_scenes.py constants (hard clutter)
K = 1.0e5
HC_DISSIPATION = 1.0
MU = 0.5
STICTION = 1e-4
SPHERE_R = 0.025
CUBE_HALF = 0.025
BIN_HALF = 0.15
BIN_WALL_T = 0.02
BIN_WALL_H = 0.30
DENSITY = 1000.0
ACCURACY = 1e-3
MAX_STEP = 0.1
WARMUP_S = 0.2
TIMED_S = 2.0
BOUNDARY_S = 0.1
NS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
MODES = ("free", "lockstep")


def _worker(seed: int, rendezvous: str, mode: str) -> None:
    import numpy as np
    from pydrake.common.eigen_geometry import AngleAxis
    from pydrake.geometry import AddContactMaterial, Box, HalfSpace, ProximityProperties, Sphere
    from pydrake.math import RigidTransform, RotationMatrix
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, ContactModel, CoulombFriction
    from pydrake.multibody.tree import SpatialInertia
    from pydrake.systems.analysis import ApplySimulatorConfig, Simulator, SimulatorConfig
    from pydrake.systems.framework import DiagramBuilder

    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    plant.set_contact_model(ContactModel.kPoint)

    def props():
        p = ProximityProperties()
        AddContactMaterial(
            dissipation=HC_DISSIPATION, point_stiffness=K, friction=CoulombFriction(MU, MU), properties=p
        )
        return p

    plant.RegisterCollisionGeometry(plant.world_body(), HalfSpace.MakePose(np.array([0, 0, 1.0]), np.zeros(3)),
                                    HalfSpace(), "ground", props())
    hw, t, h = BIN_HALF, BIN_WALL_T, BIN_WALL_H
    for j, (px, py, hx, hy) in enumerate([
        (-(hw + t), 0.0, t, hw + t),
        (hw + t, 0.0, t, hw + t),
        (0.0, -(hw + t), hw + t, t),
        (0.0, hw + t, hw + t, t),
    ]):
        plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform([px, py, 0.0]),
                                        Box(2 * hx, 2 * hy, 2 * h), f"wall{j}", props())

    lattice = clutter_lattice(seed, hard=True)
    bodies = []
    for i, (is_cube, pos, axis, angle) in enumerate(lattice):
        if is_cube:
            inertia = SpatialInertia.SolidBoxWithDensity(DENSITY, 2 * CUBE_HALF, 2 * CUBE_HALF, 2 * CUBE_HALF)
            shape = Box(2 * CUBE_HALF, 2 * CUBE_HALF, 2 * CUBE_HALF)
        else:
            inertia = SpatialInertia.SolidSphereWithDensity(DENSITY, SPHERE_R)
            shape = Sphere(SPHERE_R)
        body = plant.AddRigidBody(f"body{i}", inertia)
        plant.RegisterCollisionGeometry(body, RigidTransform(), shape, f"geom{i}", props())
        bodies.append((body, pos, axis, angle))

    plant.set_stiction_tolerance(STICTION)
    plant.Finalize()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    pc = plant.GetMyMutableContextFromRoot(context)
    for body, pos, axis, angle in bodies:
        rot = RotationMatrix()
        if axis is not None:
            rot = RotationMatrix(AngleAxis(angle, np.array(axis)))
        plant.SetFreeBodyPose(pc, body, RigidTransform(rot, np.array(pos)))

    sim = Simulator(diagram, context)
    cfg = SimulatorConfig(integration_scheme="cenic", max_step_size=MAX_STEP, accuracy=ACCURACY,
                          use_error_control=True)
    ApplySimulatorConfig(cfg, sim)
    sim.Initialize()
    sim.AdvanceTo(WARMUP_S)

    me = os.getpid()
    open(os.path.join(rendezvous, f"ready_{me}"), "w").close()
    go = os.path.join(rendezvous, "go_0")
    while not os.path.exists(go):
        time.sleep(0.02)
    t0 = time.perf_counter()
    if mode == "free":
        sim.AdvanceTo(WARMUP_S + TIMED_S)
    else:  # lockstep: barrier at every control boundary, like an RL batch
        n_bounds = int(round(TIMED_S / BOUNDARY_S))
        for k in range(1, n_bounds + 1):
            sim.AdvanceTo(WARMUP_S + k * BOUNDARY_S)
            open(os.path.join(rendezvous, f"b{k}_{me}"), "w").close()
            gok = os.path.join(rendezvous, f"go_{k}")
            if k < n_bounds:
                while not os.path.exists(gok):
                    time.sleep(0.005)
    wall = time.perf_counter() - t0
    open(os.path.join(rendezvous, f"done_{me}"), "w").close()
    print("ROW " + json.dumps({"wall_s": wall}), flush=True)


def _count(rendezvous: str, prefix: str) -> int:
    return sum(1 for f in os.listdir(rendezvous) if f.startswith(prefix))


def _run_batch(n: int, mode: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="cenic_cpu_") as rv:
        procs = [
            subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--worker", str(BASE_SEED + i), rv, mode],
                stdout=subprocess.PIPE, text=True,
            )
            for i in range(n)
        ]
        while _count(rv, "ready_") < n:
            time.sleep(0.05)
        t0 = time.perf_counter()
        open(os.path.join(rv, "go_0"), "w").close()
        if mode == "lockstep":
            n_bounds = int(round(TIMED_S / BOUNDARY_S))
            for k in range(1, n_bounds):
                while _count(rv, f"b{k}_") < n:
                    time.sleep(0.005)
                open(os.path.join(rv, f"go_{k}"), "w").close()
        while _count(rv, "done_") < n:
            time.sleep(0.05)
        makespan = time.perf_counter() - t0
        walls = []
        for p in procs:
            stdout, _ = p.communicate(timeout=600)
            for line in stdout.splitlines():
                if line.startswith("ROW "):
                    walls.append(json.loads(line[4:])["wall_s"])
        assert len(walls) == n, f"{len(walls)}/{n} workers reported"
        return {
            "scheme": "drake-cenic-cpu",
            "mode": mode,
            "accuracy": ACCURACY,
            "n_worlds": n,
            "makespan_s": makespan,
            "throughput_simss_per_wall_s": n * TIMED_S / makespan,
            "worker_wall_median_s": statistics.median(walls),
            "worker_wall_max_s": max(walls),
        }


def main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        _worker(int(sys.argv[2]), sys.argv[3], sys.argv[4])
        return 0
    out = os.path.join(os.path.dirname(__file__), "..", "results", "part1_cenic_cpu.csv")
    rows = []
    for n in NS:
        for mode in MODES:
            row = _run_batch(n, mode)
            rows.append(row)
            print(row, flush=True)
    with open(os.path.abspath(out), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.abspath(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
