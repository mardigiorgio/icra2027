# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the Part-1 experiment scenes.

The clutter scenes are drawn from the simulation state (initial pose and
after settling under ICF error control, eps 1e-3); the stiffness-sweep and
actuated-push scenes are schematic poses built from the scene constants.
A render shows ONLY what the figure it accompanies plots.

    uv run python part1/bench/part1_scenes_figure.py
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.colors
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from part1.bench.benchmarks.part1_stiffness_sweep import R as SWEEP_R  # noqa: E402
from part1.bench.four_arms import build_model, make_arm  # noqa: E402
from part1.scenes.actuated_press import BOX_HALF, TIP_R, X0, Z_CLEAR  # noqa: E402
from part1.scenes.cenic_scenes import BIN_HALF, BIN_WALL_H, DT_OUTER  # noqa: E402

FIG = os.path.join(os.path.dirname(__file__), "results", "figures")

# One fixed light for every render: per-face shading is what keeps the
# bodies of a settled pile from merging into a single flat silhouette.
_LIGHT = np.array([0.4, -0.6, 0.7])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)


def _quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class _Scene:
    """Every face of every object goes into ONE depth-sorted collection:
    matplotlib's 3D painter draws separate collections in call order, which
    is what made bodies show through each other and through the walls."""

    def __init__(self):
        self.polys, self.colors, self.edges, self.lws = [], [], [], []

    def box(self, center, half, rot, color, alpha=1.0, tiles=1, shade=False, tint=1.0, edge=0.15, lw=0.15):
        # each face tiled so large translucent walls sort against small bodies
        rgb = np.array(matplotlib.colors.to_rgb(color)) * tint
        for axis in range(3):
            for sign in (-1, 1):
                u, v = [i for i in range(3) if i != axis]
                n = np.zeros(3)
                n[axis] = sign
                lam = 0.60 + 0.40 * max(0.0, float((rot @ n) @ _LIGHT)) if shade else 1.0
                for i in range(tiles):
                    for j in range(tiles):
                        corners = []
                        for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                            c = np.zeros(3)
                            c[axis] = sign * half[axis]
                            c[u] = -half[u] + 2 * half[u] * (i + du) / tiles
                            c[v] = -half[v] + 2 * half[v] * (j + dv) / tiles
                            corners.append(c)
                        self.polys.append((rot @ np.array(corners).T).T + center)
                        self.colors.append((*(rgb * lam), alpha))
                        self.edges.append((0, 0, 0, edge))
                        self.lws.append(lw)

    def sphere(self, center, r, color, nu=16, nv=10, tint=1.0, edge=0.25, lw=0.25):
        u = np.linspace(0, 2 * np.pi, nu + 1)
        v = np.linspace(0, np.pi, nv + 1)
        rgb0 = np.array(matplotlib.colors.to_rgb(color)) * tint
        for i in range(nu):
            for j in range(nv):
                quad = []
                for uu, vv in ((u[i], v[j]), (u[i + 1], v[j]), (u[i + 1], v[j + 1]), (u[i], v[j + 1])):
                    quad.append(center + r * np.array([np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)]))
                n = (quad[0] + quad[2]) / 2 - center
                shade = 0.55 + 0.45 * max(0.0, float(n @ _LIGHT) / max(np.linalg.norm(n), 1e-9))
                self.polys.append(np.array(quad))
                self.colors.append((*(rgb0 * shade), 1.0))
                self.edges.append((0, 0, 0, edge))
                self.lws.append(lw)

    def draw(self, ax):
        coll = Poly3DCollection(
            self.polys, facecolors=self.colors, edgecolors=self.edges, linewidths=self.lws, zsort="average"
        )
        ax.add_collection3d(coll)


def _draw_state(ax, model, state, scene):
    from newton._src.geometry.types import GeoType

    bq = state.body_q.numpy().reshape(-1, 7)
    st, sb, sc = model.shape_type.numpy(), model.shape_body.numpy(), model.shape_scale.numpy()
    S = _Scene()
    if scene != "ball":
        hw, h = BIN_HALF, BIN_WALL_H
        S.box(np.array([0, 0, -0.004]), np.array([hw + 0.02, hw + 0.02, 0.004]), np.eye(3), "#c6dbef", 1.0, tiles=4)
        # Cutaway: only the two walls the camera looks INTO (at azim -58 the
        # camera sits on the +x/-y side) are drawn; near walls in front of
        # the bodies washed the whole pile out.
        for cx, cy, hx, hy in ((-hw - 0.005, 0, 0.005, hw + 0.01), (0, hw + 0.005, hw + 0.01, 0.005)):
            S.box(np.array([cx, cy, h / 2]), np.array([hx, hy, h / 2]), np.eye(3), "#9ecae1", 0.28, tiles=4)
    else:
        S.box(np.array([0, 0, -0.004]), np.array([0.3, 0.3, 0.004]), np.eye(3), "#c6dbef", 1.0, tiles=4)
    body_n = 0
    for i, b in enumerate(sb):
        if b < 0:
            continue
        # slight per-body brightness variation: settled neighbours share face
        # orientations, so identical colors would still merge under shading
        tint = 0.75 + 0.25 * ((body_n * 3) % 7) / 6.0
        body_n += 1
        p, q = bq[b, :3], bq[b, 3:]
        if GeoType(int(st[i])) == GeoType.SPHERE:
            S.sphere(p, sc[i][0], "#e6550d" if scene != "ball" else "#31a354", tint=tint)
        else:
            S.box(p, sc[i], _quat_to_mat(q), "#3182bd", shade=True, tint=tint, edge=0.55, lw=0.5)
    S.draw(ax)
    if scene == "ball":
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(0, 1.2)
        ax.set_box_aspect((1, 1, 2))
    else:
        L = BIN_HALF + 0.05
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_zlim(0, 0.5)
        ax.set_box_aspect((1, 1, 1.25))
    ax.set_axis_off()
    ax.view_init(elev=24, azim=-58)


def _save(fig, out):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{out}.{ext}"), dpi=200, bbox_inches="tight")
    print(f"wrote figures/{out}.png/.pdf")


ALL_SPECS = [("soft-clutter", "(a) Soft clutter"), ("hard-clutter", "(b) Hard clutter"), ("ball", "(c) Bouncing ball")]


def render(specs, out) -> None:
    """A render shows ONLY the scenes the figure it accompanies plots."""
    ncol = len(specs)
    fig = plt.figure(figsize=(3.2 * ncol, 5.6), constrained_layout=True)
    for col, (scene, title) in enumerate(specs):
        model = build_model(1, scene=scene)
        arm = make_arm(model, "icf-adaptive", scene=scene, tol=1e-3, max_substeps=4096)
        s0, s1, ctrl = model.state(), model.state(), model.control()
        ax = fig.add_subplot(2, ncol, col + 1, projection="3d")
        _draw_state(ax, model, s0, scene)
        ax.set_title(title + "\nt = 0", fontsize=7.5)
        t_end = 0.45 if scene == "ball" else 1.5
        for _ in range(int(round(t_end / DT_OUTER))):
            s0, s1 = arm.boundary(s0, s1, ctrl)
        ax = fig.add_subplot(2, ncol, col + 1 + ncol, projection="3d")
        _draw_state(ax, model, s0, scene)
        ax.set_title(f"t = {t_end:g} s (ICF error control, ε = 10⁻³)", fontsize=7.5)
    # No baked-in caption: the LaTeX caption in PART1.md carries the description.
    _save(fig, out)


def render_stiffness() -> None:
    """Schematic for the stiffness sweep: the 65 g sphere resting on the
    plane, gridded and shaded so it reads as a ball and not a flat disc."""
    fig = plt.figure(figsize=(3.4, 3.0), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    S = _Scene()
    S.box(np.array([0, 0, -0.002]), np.array([0.10, 0.10, 0.002]), np.eye(3), "#eeeeee", 1.0, tiles=4)
    S.sphere(np.array([0, 0, SWEEP_R + 0.002]), SWEEP_R, "#31a354", nu=24, nv=14, edge=0.35, lw=0.35)
    S.draw(ax)
    ax.set_xlim(-0.10, 0.10)
    ax.set_ylim(-0.10, 0.10)
    ax.set_zlim(0, 0.10)
    ax.set_box_aspect((2, 2, 1))
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-58)
    _save(fig, "scene_stiffness")


def render_actuated() -> None:
    """Schematic for the actuated push, frozen mid-push, showing HOW the box
    is moved: a virtual spring (the PD's P term, stiffness K_p) stretched
    between the fingertip and a commanded point that first descends and then
    travels sideways; the dragged fingertip presses the box and slides it."""
    fig = plt.figure(figsize=(4.8, 3.4), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    S = _Scene()
    S.box(np.array([0.02, 0, -0.002]), np.array([0.17, 0.12, 0.002]), np.eye(3), "#eeeeee", 1.0, tiles=4)
    S.box(np.array([0, 0, BOX_HALF]), np.array([BOX_HALF] * 3), np.eye(3), "#3182bd", shade=True, edge=0.55, lw=0.6)
    tip = np.array([-(BOX_HALF + TIP_R), 0.0, BOX_HALF])
    S.sphere(tip, TIP_R, "#e6550d", edge=0.3, lw=0.3)
    S.draw(ax)
    # bodies keep their internal depth sort; every line/label is an
    # annotation and always paints on top
    ax.computed_zorder = False
    grey = "#555555"
    start = np.array([X0, 0.0, BOX_HALF + Z_CLEAR])
    target = np.array([-0.015, 0.0, BOX_HALF])
    # the commanded point's fixed path: descend beside the box, then sideways
    ax.plot([start[0], start[0]], [0, 0], [start[2], BOX_HALF], "--", color=grey, lw=1.1, zorder=10)
    ax.plot([start[0], target[0]], [0, 0], [BOX_HALF, BOX_HALF], "--", color=grey, lw=1.1, zorder=10)
    ax.plot([target[0]], [0], [BOX_HALF], marker="o", mfc="none", mec=grey, mew=1.4, ms=8, zorder=11)
    # leader from the label to the commanded point (overdrawn on the box face)
    ax.plot([0.055, target[0] + 0.004], [-0.055, -0.004], [0.135, BOX_HALF + 0.006], ":", color=grey, lw=0.8, zorder=11)
    # the virtual spring, stretched tip -> commanded point
    xs = np.linspace(tip[0], target[0], 13)
    zig = np.array([0.0 if i in (0, 12) else (0.007 if i % 2 else -0.007) for i in range(13)])
    ax.plot(xs, np.zeros(13), BOX_HALF + zig, color="#d62728", lw=1.4, zorder=12)
    # the box's response
    ax.quiver(0.01, 0.0, 2 * BOX_HALF + 0.015, 0.055, 0, 0, color="#3182bd", arrow_length_ratio=0.35, lw=1.8, zorder=12)
    fs = 7.5
    ax.text(start[0] - 0.016, -0.03, 0.126, "commanded\npoint's path", color=grey, fontsize=fs, ha="right", zorder=15)
    ax.text(0.058, -0.058, 0.138, "commanded point", color=grey, fontsize=fs, ha="left", zorder=15)
    ax.text(0.01, -0.115, -0.004, "virtual spring, stiffness $K_p$", color="#d62728", fontsize=fs, ha="center", zorder=15)
    ax.text(-0.12, -0.05, 0.012, "fingertip", color="#e6550d", fontsize=fs, ha="center", zorder=15)
    ax.text(0.055, 0.01, 0.128, "box slides", color="#3182bd", fontsize=fs, ha="left", zorder=15)
    ax.set_xlim(-0.16, 0.16)
    ax.set_ylim(-0.12, 0.12)
    ax.set_zlim(0, 0.16)
    ax.set_box_aspect((2.0, 1.5, 1.0))
    ax.set_axis_off()
    # camera on the -x/-y side: the fingertip presses the -x face, so this is
    # the side the mechanism is visible from
    ax.view_init(elev=22, azim=-125)
    _save(fig, "scene_actuated")


def main() -> None:
    render(ALL_SPECS, "scenes")
    render([s for s in ALL_SPECS if "clutter" in s[0]], "scenes_clutter")
    render([s for s in ALL_SPECS if s[0] == "hard-clutter"], "scene_hard_clutter")
    render_stiffness()
    render_actuated()


if __name__ == "__main__":
    main()
