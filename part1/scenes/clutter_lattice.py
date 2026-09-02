# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-world clutter initial conditions, shared verbatim by the Newton
scenes and the Drake CENIC harness so world i is bit-identical in both
stacks. Pure stdlib on purpose: the Drake venv imports this file too.

No-overlap guarantee (Marco, 2026-09-01): each body redraws its jitter
until it clears every placed body by the sum of circumradii plus a 5 mm
margin (a tilted cube counts as its 4.33 cm circumradius). Same-type
bodies stack in columns, so the layer spacing is 0.085 m: at the old
0.07 m an adversely jittered cube-above-cube pair could not clear two
cube circumradii under any redraw of the new body alone. If a body
exhausts its redraw budget the whole lattice redraws from a shifted
seed; every draw comes from one seeded stream, so the acceptance
history is part of the sequence and both stacks replay it identically.
"""

from __future__ import annotations

import math
import random

SPHERE_R = 0.025
CUBE_HALF = 0.025
CUBE_CIRCUM = CUBE_HALF * math.sqrt(3.0)
MARGIN = 0.005
Z_BASE = 0.12
Z_STEP = 0.085
BASE_SEED = 7  # world i draws from BASE_SEED + i
_REDRAWS = 1000


def _radius(is_cube: bool) -> float:
    return CUBE_CIRCUM if is_cube else SPHERE_R


def clutter_lattice(seed: int, hard: bool) -> list[tuple[bool, tuple[float, float, float], tuple | None, float]]:
    """20 bodies above the bin: 4 columns x 5 layers, alternate layers
    staggered, every body jittered; hard clutter alternates spheres and
    tilted cubes. Returns [(is_cube, (x, y, z), axis_or_None, angle)]."""
    attempt_seed = seed
    while True:
        rng = random.Random(attempt_seed)
        placed: list[tuple[tuple[float, float, float], float]] = []
        out = []
        i = 0
        feasible = True
        for layer in range(5):
            shift = 0.03 if layer % 2 else 0.0
            for cx, cy in ((-0.06, -0.06), (0.06, -0.06), (-0.06, 0.06), (0.06, 0.06)):
                is_cube = hard and i % 2 == 1
                r = _radius(is_cube)
                pos = None
                for _ in range(_REDRAWS):
                    x = cx + shift + rng.uniform(-0.015, 0.015)
                    y = cy + shift + rng.uniform(-0.015, 0.015)
                    z = Z_BASE + Z_STEP * layer + rng.uniform(-0.005, 0.005)
                    if all(
                        (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2 >= (r + pr + MARGIN) ** 2
                        for (px, py, pz), pr in placed
                    ):
                        pos = (x, y, z)
                        break
                if pos is None:
                    feasible = False
                    break
                placed.append((pos, r))
                axis, angle = None, 0.0
                if is_cube:
                    ax = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
                    n = math.sqrt(sum(a * a for a in ax)) or 1.0
                    axis = tuple(a / n for a in ax)
                    angle = rng.uniform(0.0, math.pi)
                out.append((is_cube, pos, axis, angle))
                i += 1
            if not feasible:
                break
        if feasible:
            return out
        attempt_seed += 100003  # deterministic relattice, disjoint stream


def axis_angle_quat(axis: tuple, angle: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) quaternion for the cube tilts, shared by both stacks."""
    s = math.sin(angle / 2.0)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2.0))
