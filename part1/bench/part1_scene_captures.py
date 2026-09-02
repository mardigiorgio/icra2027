# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene images straight from Newton's own viewer (ViewerGL headless),
replacing the retired matplotlib reconstructions: what the doc shows is
what the simulator draws. World 0 of each scene (the seed-7 randomized
no-overlap lattice), captured at t = 0 and after settling under ICF
error control at eps = 1e-3, plus the stiffness sphere. Side-by-side
pairs are joined by raw pixel paste (no plotting library draws a scene).

    VIRTUAL_ENV=~/Documents/code/icra2027/.venv PYTHONPATH=$PWD \
      .venv/bin/python part1/bench/part1_scene_captures.py
"""

from __future__ import annotations

import os

import numpy as np
import warp as wp

import newton
import newton.viewer

from part1.bench.four_arms import build_model, make_arm
from part1.scenes.cenic_scenes import DT_OUTER

FIG = os.path.join(os.path.dirname(__file__), "results", "figures")
W, H = 900, 700


def _capture(viewer, model, state, sim_time: float) -> np.ndarray:
    viewer.set_model(model)
    viewer.begin_frame(sim_time)
    viewer.log_state(state)
    viewer.end_frame()
    frame = viewer.get_frame().numpy().reshape(H, W, 3)
    return frame[::-1]  # GL rows are bottom-up


def _scene_states(scene: str, t_end: float):
    model = build_model(1, scene=scene)
    arm = make_arm(model, "icf-adaptive", scene=scene, tol=1e-3, max_substeps=4096)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    start = model.state()
    wp.copy(start.body_q, s0.body_q)
    for _ in range(int(round(t_end / DT_OUTER))):
        s0, s1 = arm.boundary(s0, s1, ctrl)
    return model, start, s0


def _save(img: np.ndarray, name: str) -> None:
    import matplotlib.image as mpimg  # PNG writer only; nothing is drawn

    mpimg.imsave(os.path.join(FIG, f"{name}.png"), img)
    print(f"wrote figures/{name}.png")


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    viewer = newton.viewer.ViewerGL(width=W, height=H, headless=True)
    gutter = np.full((H, 24, 3), 255, dtype=np.uint8)

    # clutter scenes: t = 0 beside settled
    for scene, out in (("hard-clutter", "capture_hard_clutter"), ("soft-clutter", "capture_soft_clutter")):
        model, start, settled = _scene_states(scene, 1.5)
        viewer.set_camera(wp.vec3(0.85, -0.85, 0.75), pitch=-24.5, yaw=135.0)
        a = _capture(viewer, model, start, 0.0)
        b = _capture(viewer, model, settled, 1.5)
        _save(np.concatenate([a, gutter, b], axis=1), out)

    # stiffness sphere at rest: single world (the sweep replicates two,
    # which the builder offsets apart -- one world frames cleanly)
    import part1.bench.benchmarks.part1_stiffness_sweep as sw

    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoAdaptive.register_custom_attributes(t)
    cfg = newton.ModelBuilder.ShapeConfig(ke=1e5, kd=0.02 * 1e5, mu=0.5, margin=0.0, density=1000.0)
    b = t.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, sw.R + 1e-4), q=wp.quat_identity()))
    t.add_shape_sphere(b, radius=sw.R, cfg=cfg)
    bb = newton.ModelBuilder()
    bb.replicate(t, 1)
    bb.add_ground_plane()
    model = bb.finalize()
    arm = make_arm(model, "icf-adaptive", tol=1e-3, max_substeps=4096)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    for _ in range(int(round(1.0 / arm.dt_outer))):
        s0, s1 = arm.boundary(s0, s1, ctrl)
    viewer.set_camera(wp.vec3(0.24, -0.24, 0.12), pitch=-15.0, yaw=135.0)
    _save(_capture(viewer, model, s0, 1.0), "capture_stiffness")

    viewer.close()


if __name__ == "__main__":
    main()
