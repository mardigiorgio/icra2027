# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experiment 4 CPU arm: the CENIC reference implementation (Drake's
``integration_scheme="cenic"``, shipped since Drake 1.56) on the hard
clutter scene: WALL TIME PER 100 ms CONTROL BOUNDARY vs number of worlds
(the Experiment 3 metric, so the CPU and GPU arms read identically).

Scene parity: constants copied from part1/scenes/cenic_scenes.py; initial
conditions come from the SAME shared generator both stacks import
(part1/scenes/clutter_lattice.py), world i seeded BASE_SEED + i.

Execution model (Marco, 2026-09-02): worlds beyond the core count run
SEQUENTIALLY on the cores available, like any CPU batch. W = min(N, 96)
worker processes each build ONE diagram and host their share of worlds as
per-world contexts (~1 MB each; the 123 MB framework is paid once per
worker), so there is no artificial per-world residency cost and no memory
wall. Lockstep batch semantics: each 0.1 s boundary, every worker
advances each of its worlds one boundary in sequence, then all workers
barrier -- exactly how an RL batch consumes the simulator. All workers
start on one shared GO after building and warming up; the parent times
one makespan.

RUN (the drake venv, NOT the icra venv; single line)
  ~/Documents/code/drake-cenic/.venv/bin/python part1/bench/benchmarks/part1_cenic_cpu.py
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)

from part1.scenes.clutter_lattice import BASE_SEED, clutter_lattice  # noqa: E402

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
MAX_WORKERS = int(os.environ.get("CENIC_CPU_MAX_WORKERS", "96"))
NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
if os.environ.get("CENIC_CPU_NS"):
    NS = json.loads(os.environ["CENIC_CPU_NS"])
APPEND = os.environ.get("CENIC_CPU_APPEND") == "1"
# Lattice layers (2 = the paper's 8-body hard clutter since 2026-09-03; 5 =
# the 20-body original); the workers read it too, and the Newton scene of the
# same name draws the same bodies from the same stream.
LAYERS = int(os.environ.get("CENIC_CPU_LAYERS", "2"))


def _worker(world_seeds: list[int], rendezvous: str) -> None:
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

    bodies = []
    for i, (is_cube, _pos, _axis, _angle) in enumerate(clutter_lattice(BASE_SEED, hard=True, layers=LAYERS)):
        if is_cube:
            inertia = SpatialInertia.SolidBoxWithDensity(DENSITY, 2 * CUBE_HALF, 2 * CUBE_HALF, 2 * CUBE_HALF)
            shape = Box(2 * CUBE_HALF, 2 * CUBE_HALF, 2 * CUBE_HALF)
        else:
            inertia = SpatialInertia.SolidSphereWithDensity(DENSITY, SPHERE_R)
            shape = Sphere(SPHERE_R)
        body = plant.AddRigidBody(f"body{i}", inertia)
        plant.RegisterCollisionGeometry(body, RigidTransform(), shape, f"geom{i}", props())
        bodies.append(body)

    plant.set_stiction_tolerance(STICTION)
    plant.Finalize()
    diagram = builder.Build()

    sims = []
    for seed in world_seeds:
        ctx = diagram.CreateDefaultContext()
        pc = plant.GetMyMutableContextFromRoot(ctx)
        for body, (_is_cube, pos, axis, angle) in zip(bodies, clutter_lattice(seed, hard=True, layers=LAYERS)):
            rot = RotationMatrix(AngleAxis(angle, np.array(axis))) if axis is not None else RotationMatrix()
            plant.SetFreeBodyPose(pc, body, RigidTransform(rot, np.array(pos)))
        sim = Simulator(diagram, ctx)
        ApplySimulatorConfig(SimulatorConfig(integration_scheme="cenic", max_step_size=MAX_STEP,
                                             accuracy=ACCURACY, use_error_control=True), sim)
        sim.Initialize()
        sim.AdvanceTo(WARMUP_S)
        sims.append(sim)

    me = os.getpid()
    open(os.path.join(rendezvous, f"ready_{me}"), "w").close()
    while not os.path.exists(os.path.join(rendezvous, "go_0")):
        time.sleep(0.0002)
    n_bounds = int(round(TIMED_S / BOUNDARY_S))
    compute_s = 0.0  # integration time alone, without the barrier waits
    for k in range(1, n_bounds + 1):
        target = WARMUP_S + k * BOUNDARY_S
        t_adv = time.perf_counter()
        for sim in sims:
            sim.AdvanceTo(target)
        compute_s += time.perf_counter() - t_adv
        open(os.path.join(rendezvous, f"b{k}_{me}"), "w").close()
        if k < n_bounds:
            while not os.path.exists(os.path.join(rendezvous, f"go_{k}")):
                time.sleep(0.0002)
    open(os.path.join(rendezvous, f"done_{me}"), "w").close()
    print("ROW " + json.dumps({"worlds": len(sims), "compute_s": compute_s}), flush=True)


def _count(rendezvous: str, prefix: str) -> int:
    return sum(1 for f in os.listdir(rendezvous) if f.startswith(prefix))


def _run_batch(n: int) -> dict:
    workers = min(n, MAX_WORKERS)
    shares = [[BASE_SEED + i for i in range(n) if i % workers == w] for w in range(workers)]
    with tempfile.TemporaryDirectory(prefix="cenic_cpu_") as rv:
        procs = [
            subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--worker", json.dumps(share), rv],
                stdout=subprocess.PIPE, text=True,
            )
            for share in shares
        ]
        while _count(rv, "ready_") < workers:
            time.sleep(0.05)
        t0 = time.perf_counter()
        open(os.path.join(rv, "go_0"), "w").close()
        n_bounds = int(round(TIMED_S / BOUNDARY_S))
        for k in range(1, n_bounds):
            while _count(rv, f"b{k}_") < workers:
                time.sleep(0.0002)
            open(os.path.join(rv, f"go_{k}"), "w").close()
        while _count(rv, "done_") < workers:
            time.sleep(0.05)
        makespan = time.perf_counter() - t0
        compute = []
        for p in procs:
            out, _ = p.communicate(timeout=600)
            for line in out.splitlines():
                if line.startswith("ROW "):
                    compute.append(float(json.loads(line[4:]).get("compute_s", 0.0)))
        barrier_makespan = makespan
        if workers == n and compute:
            # One world per worker: every world runs alone on its core and the
            # lockstep barrier only synchronizes their clocks, so the batch's
            # time is the slowest world's integration time. The file barrier's
            # polling (measured ~55 ms per run at N = 1) is an artifact of the
            # harness, not of the reference implementation, and is left out.
            makespan = max(compute)
        return {
            "scheme": "drake-cenic-cpu",
            "mode": "lockstep",
            "accuracy": ACCURACY,
            "n_worlds": n,
            "workers": workers,
            "makespan_s": makespan,
            "wall_ms_per_boundary": makespan / n_bounds * 1000.0,
            "barrier_makespan_s": barrier_makespan,
        }


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _worker(json.loads(sys.argv[2]), sys.argv[3])
        return 0
    out = os.environ.get("CENIC_CPU_OUT") or os.path.join(os.path.dirname(__file__), "..", "results", "part1_cenic_cpu.csv")
    rows = []
    for n in NS:
        row = _run_batch(n)
        rows.append(row)
        print(row, flush=True)
    with open(os.path.abspath(out), "a" if APPEND else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not APPEND:
            w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.abspath(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
