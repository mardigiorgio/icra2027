# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Resting penetration vs step size and vs tolerance at ONE calibrated
stiffness: one 65 g sphere resting on the plane, k = 1e5 N/m (the hard
anchor; k = 1e3, the soft anchor, runs as a companion). No fitted tau(k)
curve anywhere: each system uses only its directly calibrated solref.

Axes (the PI's): penetration/(m*g/k) vs dt for the fixed arms, and vs the
requested tolerance for the error-controlled arms. MuJoCo appears in BOTH
its code forms (mujoco_warp constraint.py L130/L132): the reference solref
(timeconst, 1) that practitioners run, whose timeconst is clamped to
2*dt (L120), and the direct solref (-k, -b), a literal stiffness the clamp
cannot touch but which is only stable while omega*dt stays small. The
direct rows self-verify the calibration: at fine dt they must read ratio 1.

    uv run python -m part1.bench.benchmarks.part1_stiffness_sweep
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys

import numpy as np
import warp as wp

import newton

from part1.bench.four_arms import _make_icf, _make_icf_adaptive, _make_mujoco, _make_mujoco_adaptive
from part1.scenes.cenic_scenes import MUJOCO_TAU_K1E3, MUJOCO_TAU_K1E5

R = 0.025
MASS = 1000.0 * 4.0 / 3.0 * math.pi * R**3
BOUNDARY_S = 0.01
DTS = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4]
TOLS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
# Calibrated per system, no interpolation between them.
REF_SOLREF = {1e5: (MUJOCO_TAU_K1E5, 1.0), 1e3: (MUJOCO_TAU_K1E3, 1.0)}
# Direct format for the 65 g sphere at k = 1e5, from the committed
# calibration (tables/mujoco_stiffness_probe.md). Self-verifying: the
# fine-dt rows of this bench must read ratio 1, else the constant is wrong.
DIRECT_SOLREF = {1e5: (-1.703e5, -632.0)}


def _model(k):
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoAdaptive.register_custom_attributes(t)
    cfg = newton.ModelBuilder.ShapeConfig(ke=k, kd=0.02 * k, mu=0.5, margin=0.0, density=1000.0)
    b = t.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, R + 1e-4), q=wp.quat_identity()))
    t.add_shape_sphere(b, radius=R, cfg=cfg)
    bb = newton.ModelBuilder(); bb.replicate(t, 2); bb.add_ground_plane()
    return bb.finalize()


def _run(arm, knob, k):
    m = _model(k)
    icf = {"contact_stiffness": k, "contact_stiction_tolerance": 1e-4}
    if arm == "icf":
        a = _make_icf(m, int(round(BOUNDARY_S / knob)), BOUNDARY_S, icf)
    elif arm == "icf-adaptive":
        a = _make_icf_adaptive(m, knob, BOUNDARY_S, icf, 4096)
    elif arm == "mujoco":
        a = _make_mujoco(m, int(round(BOUNDARY_S / knob)), BOUNDARY_S, REF_SOLREF[k])
    elif arm == "mujoco-direct":
        a = _make_mujoco(m, int(round(BOUNDARY_S / knob)), BOUNDARY_S, DIRECT_SOLREF[k])
    elif arm == "mujoco-adaptive":
        a = _make_mujoco_adaptive(m, knob, BOUNDARY_S, 4096, REF_SOLREF[k])
    else:
        raise SystemExit(f"unknown arm {arm}")
    s0, s1, c = m.state(), m.state(), m.control()
    for _ in range(int(round(3.0 / BOUNDARY_S))):
        s0, s1 = a.boundary(s0, s1, c)
    z = s0.body_q.numpy().reshape(-1, 7)[:, 2]
    pen = float(R - z.mean())
    static = MASS * 9.81 / k
    # unstable = launched or non-finite: the resting readout is meaningless
    unstable = (not bool(np.isfinite(z).all())) or bool(z.max() > R + 0.05) or pen < -1e-4
    return {"pen_m": pen, "static_m": static, "ratio": pen / static, "unstable": unstable}


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--single":
        arm, (knob, k) = sys.argv[2], (float(sys.argv[3].split(",")[0]), float(sys.argv[3].split(",")[1]))
        print("ROW " + json.dumps(_run(arm, knob, k)), flush=True)
        return 0
    cells = []
    for k in (1e5, 1e3):
        for dt in DTS:
            cells.append(("icf", dt, k))
            cells.append(("mujoco", dt, k))
            if k in DIRECT_SOLREF:
                cells.append(("mujoco-direct", dt, k))
    for tol in TOLS:
        cells.append(("icf-adaptive", tol, 1e5))
        cells.append(("mujoco-adaptive", tol, 1e5))
    out = "part1/bench/results/part1_stiffness_sweep.csv"
    rows = []
    for arm, knob, k in cells:
        r = subprocess.run([sys.executable, "-m", "part1.bench.benchmarks.part1_stiffness_sweep", "--single", arm, f"{knob},{k}"],
                           capture_output=True, text=True, timeout=1800)
        got = None
        for line in r.stdout.splitlines():
            if line.startswith("ROW "):
                got = json.loads(line[4:])
        if got is None:
            print(f"FAIL {arm} {knob} k={k:g}: {r.stderr[-300:]}", flush=True); continue
        fixed = "adaptive" not in arm
        row = {"arm": arm, "dt_s": knob if fixed else "", "accuracy": "" if fixed else knob, "k_N_per_m": k, **got}
        rows.append(row); print(row, flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
