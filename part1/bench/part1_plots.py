# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Part-1 figures from the committed CSVs, with self-contained figure text:
every point states its accuracy eps_acc (adaptive) or time step dt (fixed).
CPU only; re-run after any sweep.

    uv run python part1/bench/part1_plots.py
"""

from __future__ import annotations

import csv
import math as _math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

STYLE = {
    "mujoco": dict(color="#c0392b", marker="s", ls="--", label="MuJoCo"),
    "mujoco-adaptive": dict(color="#e67e22", marker="^", ls="-", label="MuJoCo EC"),
    "icf": dict(color="#2980b9", marker="o", ls="--", label="ICF"),
    "icf-adaptive": dict(color="#27ae60", marker="D", ls="-", label="ICF EC (CENIC)"),
}
SCENE_NOTE = {
    "hard-clutter": "hard clutter: 10 spheres + 10 cubes in a bin, k = 10⁵ N/m, v_s = 0.1 mm/s",
    "soft-clutter": "soft clutter: 20 spheres in a bin, k = 10³ N/m, v_s = 1 cm/s",
    "ball": "0.1 kg ball, k = 10³ N/m, zero dissipation, 1 m drop, 10 s",
}


def _rows(name: str) -> list[dict]:
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = v
            out.append(row)
    return out


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote figures/{name}.png/.pdf")


def _dt_label(dt: float) -> str:
    return f"δt = {dt * 1e3:g} ms" if dt >= 1e-3 else f"δt = {dt * 1e6:g} µs"


SCENE_TITLE = {"soft-clutter": "Soft Clutter", "hard-clutter": "Hard Clutter"}
SCENE_ORDER = ("soft-clutter", "hard-clutter")  # least to most complex


def _wp_rows(scene: str, n: int) -> list[dict]:
    return _rows(f"part1_workprecision_{scene}_n{n}.csv")


def workprecision() -> None:
    """One column per scene, one row per world count
    (N=1 is the single-scene setting, N=1024 the GPU regime).
    x = requested accuracy, y = wall time per simulated second. A run that
    timed out (>100 s per simulated second) or exhausted its march budget
    is a cross at the top edge."""
    ns = [n for n in (1, 1024) if any(_wp_rows(sc, n) for sc in SCENE_ORDER)]
    scenes = [sc for sc in SCENE_ORDER if any(_wp_rows(sc, n) for n in ns)]
    if not ns or not scenes:
        return
    fig, axes = plt.subplots(
        len(ns),
        len(scenes),
        figsize=(3.6 * len(scenes), 2.9 * len(ns)),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
    )
    for i, n in enumerate(ns):
        for j, scene in enumerate(scenes):
            ax = axes[i][j]
            rows = _wp_rows(scene, n)
            ok = [r for r in rows if r["status"] == "ok"]
            bad = [r for r in rows if r["status"] != "ok" and r["accuracy"] != ""]
            for arm in ("icf-adaptive", "mujoco-adaptive"):
                pts = sorted(
                    (r["accuracy"], r["wall_s_per_sim_s"]) for r in ok if r["arm"] == arm and r["accuracy"] != ""
                )
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], ms=4, lw=1.2, **STYLE[arm])
            for arm in ("icf", "mujoco"):
                for r in sorted((r for r in ok if r["arm"] == arm and r["dt_s"] != ""), key=lambda r: -r["dt_s"]):
                    if r["dt_s"] not in (1e-2, 1e-3):
                        continue  # reference steps: 10 ms and 1 ms
                    ax.axhline(r["wall_s_per_sim_s"], color=STYLE[arm]["color"], ls=":", lw=0.9, alpha=0.9)
                    ax.text(
                        0.995,
                        r["wall_s_per_sim_s"],
                        f"{STYLE[arm]['label'].split(',')[0]} fixed {_dt_label(r['dt_s'])}",
                        fontsize=5.5,
                        color=STYLE[arm]["color"],
                        va="bottom",
                        ha="right",
                        transform=ax.get_yaxis_transform(),
                    )
            ax.set_xscale("log")
            ax.set_yscale("log")
            if not ax.xaxis_inverted():  # shared x: invert exactly once (accuracy tightens to the right)
                ax.invert_xaxis()
            thresh = 100.0 * n  # timeout: 100 s per simulated second of one scene
            top = max([r["wall_s_per_sim_s"] for r in ok if r["wall_s_per_sim_s"] != ""] + [1.0]) * 4.0
            if any(r["status"] == "timeout" for r in bad):
                top = max(top, thresh * 2.5)
            ax.set_ylim(top=top)
            if thresh < top:
                ax.axhline(thresh, color="gray", lw=0.8, ls="-.")
                ax.text(
                    0.005,
                    thresh,
                    f"timeout: 100 s per scene-second (×{n})" if n > 1 else "timeout: 100 s per simulated second",
                    fontsize=5.5,
                    color="gray",
                    va="bottom",
                    ha="left",
                    transform=ax.get_yaxis_transform(),
                )
            for r in bad:
                if r["status"] == "timeout":
                    ax.plot(r["accuracy"], thresh, marker="x", ms=6, mew=1.6, ls="none", color=STYLE[r["arm"]]["color"])
                else:  # 'budget' (practical wall cap) or 'budget-exhausted' / 'fail'
                    ax.plot(
                        r["accuracy"], top / 1.6, marker="+", ms=7, mew=1.6, ls="none", color=STYLE[r["arm"]]["color"]
                    )
                    ax.annotate(
                        r["status"],
                        (r["accuracy"], top / 1.6),
                        textcoords="offset points",
                        xytext=(0, -9),
                        ha="center",
                        fontsize=5,
                        color=STYLE[r["arm"]]["color"],
                    )
            ax.axhline(1.0, color="k", lw=0.6, alpha=0.35)
            ax.text(
                0.005,
                1.0,
                "real time",
                fontsize=5.5,
                color="k",
                alpha=0.6,
                va="bottom",
                ha="left",
                transform=ax.get_yaxis_transform(),
            )
            if i == 0:
                ax.set_title(SCENE_TITLE[scene], fontsize=9)
            if i == len(ns) - 1:
                ax.set_xlabel("Accuracy")
            if j == 0:
                ax.set_ylabel(f"Wall Time (s)\nN = {n}")
            ax.grid(True, which="both", alpha=0.3)
            ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=6.5, loc="upper left")
    _save(fig, "workprecision")




def _knob_label(r: dict) -> str:
    return f"ε={r['accuracy']:g}" if r.get("accuracy", "") != "" else _dt_label(r["dt_s"]).replace("δt = ", "")



def scaling() -> None:
    for scene in ("hard-clutter", "soft-clutter"):
        rows = _rows(f"part1_scaling_{scene}.csv")
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(5.4, 3.9), constrained_layout=True)
        for arm in STYLE:
            has_trials = rows and rows[0].get("wall_ms_trial_min", "") != ""
            lo_key, hi_key = (
                ("wall_ms_trial_min", "wall_ms_trial_max") if has_trials else ("wall_ms_median", "wall_ms_p90")
            )
            pts = sorted(
                (r["n_worlds"], r["wall_ms_median"], r[lo_key], r[hi_key], _knob_label(r))
                for r in rows
                if r["arm"] == arm
            )
            if not pts:
                continue
            xs = [p[0] for p in pts]
            st = dict(STYLE[arm])
            st["label"] = f"{st['label']}, {pts[0][4]}"
            ax.plot(xs, [p[1] for p in pts], **st)
            ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts], color=STYLE[arm]["color"], alpha=0.15, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Parallel worlds")
        trials = int(rows[0].get("trials", 1) or 1)
        band = f"band: spread of {trials} independent runs" if trials > 1 else "band: median → p90"
        ax.set_ylabel(f"Wall Time per {rows[0].get('dt_outer_s', 0.01) * 1e3:g} ms step (ms)")
        ax.set_title(SCENE_TITLE[scene], fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        _save(fig, f"scaling_{scene}")



_OBJECT_MASS = 1000.0 * 4.0 / 3.0 * 3.141592653589793 * 0.025**3  # 65 g clutter sphere
_SCENE_K = {"soft-clutter": 1e3, "hard-clutter": 1e5}


def _static_pen(scene: str) -> float:
    """Single-object static penetration m*g/k [m] -- the compliance the contact model prescribes."""
    return _OBJECT_MASS * 9.81 / _SCENE_K[scene]


_DROP_HEIGHT = 0.40  # top drop layer [m] -> impact speed sqrt(2 g h)


def _impact_pen(scene: str) -> float:
    """Deepest penetration the contact model itself produces for the drop's
    impact speed: v * sqrt(m/k) (Hertz-free linear spring, single body)."""
    v = _math.sqrt(2.0 * 9.81 * _DROP_HEIGHT)
    return v * _math.sqrt(_OBJECT_MASS / _SCENE_K[scene])





def stiffness_sweep() -> None:
    """Realized contact stiffness: resting penetration of one sphere over the
    model's m g / k as the requested k rises (ICF Fig. 18's axis), per arm at
    fixed dt and under error control."""
    rows = [r for r in _rows("part1_stiffness_sweep.csv") if r.get("finite") in (True, "True")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.9), constrained_layout=True)
    variants = {}
    for r in rows:
        variants.setdefault((r["arm"], _knob_label(r)), []).append((r["k_N_per_m"], r["ratio"]))
    # ICF's four configurations coincide to the pixel (that IS the finding),
    # so drawn separately three of them exist only in the legend. Collapse
    # them into ONE labeled line; MuJoCo's four genuinely separate curves
    # keep their own entries.
    icf_variants = {k: v for k, v in variants.items() if k[0] in ("icf", "icf-adaptive")}
    other_variants = {k: v for k, v in variants.items() if k[0] not in ("icf", "icf-adaptive")}
    if icf_variants:
        pts = sorted(next(iter(sorted(icf_variants.items())))[1])
        st = dict(STYLE["icf-adaptive"])
        st["label"] = "ICF & ICF EC (CENIC) — all steps and tolerances coincide"
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ms=5, lw=2.2, **st)
    for (arm, lab), pts in sorted(other_variants.items()):
        pts.sort()
        st = dict(STYLE[arm])
        st["label"] = f"{STYLE[arm]['label']}, {lab}"
        if "10 ms" in lab or "0.001" in lab:  # the coarser setting of each pair is hollow
            st["mfc"] = "none"
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ms=5, **st)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Requested contact stiffness k (N/m)")
    ax.set_ylabel("Resting penetration / (m g / k)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6)
    _save(fig, "stiffness_sweep")



def actuated() -> None:
    """Actuated push: box lift, box pitch rate, max tip penetration and
    tip-box relative velocity vs the controller gain K_p, per arm at one
    fixed dt and one accuracy; unstable cells are drawn as crosses on the
    top edge."""
    rows = _rows("part1_actuated.csv")
    if not rows or "box_lift_max_m" not in rows[0]:
        return
    picks = {"icf": ("dt_s", 1e-3), "mujoco": ("dt_s", 1e-3), "icf-adaptive": ("accuracy", 1e-3), "mujoco-adaptive": ("accuracy", 1e-3)}
    metrics = [("box_lift_max_m", 1e3, "Box lift (mm)"), ("box_pitch_rate_rms", 1.0, "Box pitch rate (rad/s)"),
               ("pen_tip_max_m", 1e3, "Tip penetration (mm)"), ("rel_vx_rms_m_s", 1.0, "Tip–box relative velocity (m/s)")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
    for ax, (key, scale, ylab) in zip(axes, metrics):
        floor = 1e-3 if scale == 1e3 else 1e-4
        for arm, (col, val) in picks.items():
            sel = [r for r in rows if r["arm"] == arm and r.get(col, "") != "" and abs(r[col] - val) < 1e-12]
            good = sorted((r["kp"], max(r[key] * scale, floor)) for r in sel if r.get("unstable") in (False, "False") and r.get(key, "") != "")
            bad = sorted(r["kp"] for r in sel if r.get("unstable") in (True, "True"))
            st = dict(STYLE[arm]); st["label"] = f"{STYLE[arm]['label']}, {_dt_label(val).replace('δt = ', '') if col == 'dt_s' else f'ε={val:g}'}"
            if good:
                ax.plot([p[0] for p in good], [p[1] for p in good], ms=5, **st)
            if bad:
                ax.plot(bad, [1.0] * len(bad), ls="none", marker="x", ms=8, mew=2, color=STYLE[arm]["color"], transform=ax.get_xaxis_transform(), clip_on=False)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("K_p (N/m)"); ax.set_ylabel(ylab); ax.grid(True, which="both", alpha=0.3)
    # Both ICF arms clip to the plot floor in the lift panel (lift <= 0):
    # one visible line, two legend entries — say so IN the panel so the
    # legend never claims a curve the eye cannot find.
    axes[0].annotate("ICF & CENIC: no lift\n(both at plot floor)", xy=(0.04, 0.08),
                     xycoords="axes fraction", fontsize=6, color=STYLE["icf-adaptive"]["color"])
    axes[0].legend(fontsize=6)
    k = rows[0]["k"]; v = rows[0]["slide_speed"]
    _save(fig, "actuated")




def actuated_scaling() -> None:
    """Throughput in the actuated regime: wall per world per boundary and
    inner attempts per world per boundary vs number of heterogeneous worlds."""
    rows = [r for r in _rows("part1_actuated_scaling.csv") if r.get("unstable") in (False, "False")]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for arm in STYLE:
        pts = sorted((r["n_worlds"], r["wall_ms_per_boundary"] / r["n_worlds"] * 1e3, r["steps_per_boundary"]) for r in rows if r["arm"] == arm)
        if not pts:
            continue
        axes[0].plot([p[0] for p in pts], [p[1] for p in pts], ms=5, **STYLE[arm])
        axes[1].plot([p[0] for p in pts], [p[2] for p in pts], ms=5, **STYLE[arm])
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("Worlds"); ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Wall Time per world (µs)"); axes[1].set_ylabel("Inner steps per boundary")
    axes[0].set_title("(a)", loc="left", fontsize=10); axes[1].set_title("(b)", loc="left", fontsize=10); axes[0].legend(fontsize=7)
    _save(fig, "actuated_scaling")


if __name__ == "__main__":
    workprecision()
    speed_bars()
    ball_energy()
    penetration()
    artifacts()
    scaling()
    scaling_per_world()
    ball_workprecision()
    realtime_trace()
    stiffness_sweep()
    consistency()
    actuated()
    actuated_chatter()
    actuated_scaling()
