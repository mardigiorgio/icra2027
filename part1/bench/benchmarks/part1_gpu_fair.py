# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experiment 4 GPU arm on the CPU bench's exact protocol.

ICF-adaptive (eps = 1e-3) on hard clutter: warm to t = 0.2 s untimed
(part1_cenic_cpu.py WARMUP_S), then ONE makespan over the 20 timed control
boundaries 0.2 -> 2.2 s (TIMED_S), captured-graph replay from the pristine
start, one process per N. Appends rows to part1_gpu_fair_ladder.csv with
the solver commit, so every row is re-runnable at its own code state.

    ICF_MARCH_COMPACT=1 VIRTUAL_ENV=~/Documents/code/icra2027/.venv PYTHONPATH=$PWD \\
      .venv/bin/python -m part1.bench.benchmarks.part1_gpu_fair --ns 1024 2048 4096 8192 16384
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(_REPO, "part1", "bench", "results", "part1_gpu_fair_ladder.csv")
WARMUP_BOUNDARIES = 2
TIMED_BOUNDARIES = 20
BOUNDARY_S = 0.1
TOL = 1e-3


def _solver_commit() -> str:
    d = os.path.expanduser("~/Documents/code/icf_warp_adaptive")
    try:
        return subprocess.check_output(["git", "-C", d, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _one(n: int, scene: str, tol: float) -> dict:
    import warp as wp

    from part1.bench.four_arms import _icf, _icf_params, K_INIT, NEWTON_KAPPA, NEWTON_TOL_FLOOR, DT_INNER_MIN, build_model
    from part1.scenes.cenic_scenes import SCENES
    import newton

    model = build_model(n, scene=scene)
    icf = _icf()
    solver = icf.SolverICFAdaptive(
        model,
        params=_icf_params({**SCENES[scene].icf, "newton_tolerance": max(NEWTON_KAPPA * tol, NEWTON_TOL_FLOOR)}),
        adaptive=icf.IcfAdaptiveParams(tol=tol, dt_inner_init=K_INIT * BOUNDARY_S, dt_inner_min=DT_INNER_MIN,
                                       dt_inner_max=BOUNDARY_S, max_substeps=4096),
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    solver.attach_collision_pipeline(pipeline)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    pipeline.collide(s0, contacts)
    solver.step(s0, s1, ctrl, contacts, 0.01)  # module warm-up, eager

    def cap(a, b):
        with wp.ScopedCapture() as c:
            pipeline.collide(a, contacts)
            solver.step(a, b, ctrl, contacts, BOUNDARY_S)
        return c.graph

    g = [cap(s0, s1), cap(s1, s0)]
    for k in range(WARMUP_BOUNDARIES):
        wp.capture_launch(g[k % 2])
    wp.synchronize()
    t0 = time.perf_counter()
    for k in range(TIMED_BOUNDARIES):
        wp.capture_launch(g[k % 2])
    wp.synchronize()
    wall = time.perf_counter() - t0
    return {
        "scheme": "icf-adaptive-gpu",
        "protocol": "warm0.2s_time20x0.1s_makespan",
        "accuracy": tol,
        "n_worlds": n,
        "makespan_s": round(wall, 1),
        "ms_per_world": round(1000.0 * wall / n, 2),
        "march_compact": int(getattr(solver, "_march_compact", False)),
        "compact_width": int(getattr(solver, "_mc_cap", 0)),
        "solver_commit": _solver_commit(),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ns", nargs="*", type=int, default=[1024, 2048, 4096, 8192, 16384])
    p.add_argument("--single", type=int, default=None)
    p.add_argument("--scene", default="hard-clutter", help="a clutter scene from part1.scenes.cenic_scenes.SCENES")
    p.add_argument("--tol", type=float, default=TOL, help="the GPU controller's absolute position tolerance eps_acc [m]")
    args = p.parse_args()
    if args.single is not None:
        import json

        print("ROW " + json.dumps(_one(args.single, args.scene, args.tol)), flush=True)
        return 0
    rows = []
    out_path = OUT if args.scene == "hard-clutter" else OUT.replace("part1_gpu_fair_ladder.csv", f"part1_gpu_fair_ladder_{args.scene}.csv")
    for n in args.ns:
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--single", str(n), "--scene", args.scene, "--tol", repr(args.tol)],
                             capture_output=True, text=True)
        line = [ln for ln in out.stdout.splitlines() if ln.startswith("ROW ")]
        if not line:
            print(out.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"N={n} produced no row")
        import json

        rows.append(json.loads(line[-1][4:]))
        print(rows[-1], flush=True)
    new_file = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new_file:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
