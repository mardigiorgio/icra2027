# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experiment 4 CPU arm: the CENIC reference implementation (Drake's
``integration_scheme="cenic"``, shipped since Drake 1.56) on the hard
clutter scene, cost per world vs number of concurrent worlds.

The scene is a 1:1 port of part1/scenes/cenic_scenes.py hard clutter
(constants copied, lattice RNG replayed call-for-call with the same seed,
so both stacks integrate the same initial condition): 10 spheres
(r = 2.5 cm, density 1000) + 10 cubes (2.5 cm half), k = 1e5 N/m,
Hunt-Crossley dissipation 1.0 s/m, mu = 0.5, stiction tolerance 1e-4,
30 cm bin. Protocol matches part1_scaling.py: 0.2 s warm-up, 2 s timed,
accuracy 1e-3, max step 0.1 s.

CPU worlds are independent processes (one world each, nice +10 so a
concurrent GPU training keeps its host threads); per-world cost is each
process's own wall over the timed window.

RUN (the drake venv, NOT the icra venv; single line)
  ~/Documents/code/drake-cenic/.venv/bin/python part1/bench/benchmarks/part1_cenic_cpu.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time

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
LATTICE_SEED = 7
DENSITY = 1000.0
ACCURACY = 1e-3
MAX_STEP = 0.1
WARMUP_S = 0.2
TIMED_S = 2.0
NS = [1, 2, 4, 8, 16, 32, 64, 96]


def _lattice():
    """Replays cenic_scenes._clutter_template's RNG call-for-call."""
    rng = random.Random(LATTICE_SEED)
    bodies = []
    i = 0
    for layer in range(5):
        shift = 0.03 if layer % 2 else 0.0
        for cx, cy in ((-0.06, -0.06), (0.06, -0.06), (-0.06, 0.06), (0.06, 0.06)):
            x = cx + shift + rng.uniform(-0.015, 0.015)
            y = cy + shift + rng.uniform(-0.015, 0.015)
            z = 0.12 + 0.07 * layer + rng.uniform(-0.005, 0.005)
            axis, angle = None, 0.0
            if i % 2 == 1:  # hard clutter: odd indices are cubes, tilted
                ax = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
                n = math.sqrt(sum(a * a for a in ax)) or 1.0
                axis = tuple(a / n for a in ax)
                angle = rng.uniform(0.0, math.pi)
            bodies.append((i % 2 == 1, (x, y, z), axis, angle))
            i += 1
    return bodies


def _run_world() -> dict:
    os.nice(10)
    import numpy as np
    from pydrake.geometry import AddContactMaterial, Box, HalfSpace, ProximityProperties, Sphere
    from pydrake.math import RigidTransform, RotationMatrix
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, ContactModel, CoulombFriction
    from pydrake.multibody.tree import SpatialInertia
    from pydrake.systems.analysis import ApplySimulatorConfig, Simulator, SimulatorConfig
    from pydrake.systems.framework import DiagramBuilder

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
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
    for i, (is_cube, pos, axis, angle) in enumerate(_lattice()):
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
            from pydrake.common.eigen_geometry import AngleAxis

            rot = RotationMatrix(AngleAxis(angle, np.array(axis)))
        plant.SetFreeBodyPose(pc, body, RigidTransform(rot, np.array(pos)))

    sim = Simulator(diagram, context)
    cfg = SimulatorConfig(integration_scheme="cenic", max_step_size=MAX_STEP, accuracy=ACCURACY,
                          use_error_control=True)
    ApplySimulatorConfig(cfg, sim)
    sim.Initialize()
    sim.AdvanceTo(WARMUP_S)
    t0 = time.perf_counter()
    sim.AdvanceTo(WARMUP_S + TIMED_S)
    wall = time.perf_counter() - t0
    return {"wall_s": wall, "wall_s_per_sim_s": wall / TIMED_S}


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--single":
        print("ROW " + json.dumps(_run_world()), flush=True)
        return 0
    out = os.path.join(os.path.dirname(__file__), "..", "results", "part1_cenic_cpu.csv")
    rows = []
    for n in NS:
        procs = [
            subprocess.Popen([sys.executable, os.path.abspath(__file__), "--single"],
                             stdout=subprocess.PIPE, text=True)
            for _ in range(n)
        ]
        walls = []
        for p in procs:
            stdout, _ = p.communicate(timeout=3600)
            for line in stdout.splitlines():
                if line.startswith("ROW "):
                    walls.append(json.loads(line[4:])["wall_s_per_sim_s"])
        if len(walls) != n:
            print(f"FAIL n={n}: {len(walls)}/{n} worlds reported", flush=True)
            continue
        row = {
            "scheme": "drake-cenic-cpu",
            "accuracy": ACCURACY,
            "n_worlds": n,
            "wall_s_per_sim_s_per_world_median": statistics.median(walls),
            "wall_s_per_sim_s_per_world_min": min(walls),
            "wall_s_per_sim_s_per_world_max": max(walls),
        }
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
