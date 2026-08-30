#!/usr/bin/env python
"""Verify the icra2027 platform is wired correctly.

Asserts, in order:
  1. `import newton` resolves to the custom fork checkout (newton-adaptive),
     NOT Isaac Lab's stock git-pinned upstream Newton.
  2. The adaptive solver SolverMuJoCoAdaptive is importable from newton.solvers.
  3. Isaac Lab + its Newton backend extension import.
  4. The IsaacLab clone carries the newton_adaptive_ui Kit extension and the
     --solver CLI flag (both live in the fork's develop branch, not here).

Warns (does not fail) when the SAP solver variants are unavailable, since a
MuJoCo-only install without the sap_warp clone is still valid.

Exit 0 on success, 1 on the first failed assertion (with a diagnostic).
Run with the platform venv active, e.g.  `uv run python install/verify.py`
(install/install.sh runs it as its final step).
"""
from __future__ import annotations

import pathlib
import sys
import typing


def fail(msg: str) -> typing.NoReturn:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def warn(msg: str) -> None:
    print(f"WARN: {msg}", file=sys.stderr)


# 1. The Newton fork is the active import path (overrides Isaac Lab's pin).
try:
    import newton
except Exception as e:  # pragma: no cover
    fail(f"could not import newton: {e!r}")

newton_path = getattr(newton, "__file__", "") or ""
if "newton-adaptive" not in newton_path:
    fail(
        "the custom Newton fork is NOT active.\n"
        f"  newton resolves to: {newton_path}\n"
        "  expected a path under .../newton-adaptive/ (the adaptive fork).\n"
        "  Isaac Lab's stock pinned Newton is shadowing the fork -- re-run\n"
        "  `uv sync --inexact --locked` so the editable fork is installed last and wins."
    )

# 2. The adaptive solver is present in the fork.
try:
    import newton.solvers as solvers
except Exception as e:
    fail(f"could not import newton.solvers: {e!r}")

if not hasattr(solvers, "SolverMuJoCoAdaptive"):
    fail(
        "SolverMuJoCoAdaptive is missing from newton.solvers -- this Newton\n"
        f"  build ({newton_path}) is not the adaptive fork, or it is out of date."
    )

# 2b. SAP variants (warn-only: valid installs may omit the sap_warp clone).
sap_ok = getattr(solvers, "SolverSAP", None) is not None and getattr(solvers, "SolverSAPAdaptive", None) is not None
if not sap_ok:
    warn(
        "SolverSAP / SolverSAPAdaptive unavailable -- the sap_warp clone is\n"
        "  missing or broken (the fork degrades them to None). Clone\n"
        "  https://github.com/mardigiorgio/sap_warp.git next to this repo, or\n"
        "  export SAP_WARP_PATH, to enable the sap / sap-adaptive variants."
    )

# 3. Isaac Lab + its Newton backend extension import.
try:
    import isaaclab  # noqa: F401
    import isaaclab_newton  # noqa: F401
except Exception as e:
    fail(
        f"Isaac Lab import failed: {e!r}\n"
        "  Run `<IsaacLab>/isaaclab.sh -i` against this venv\n"
        "  (install/install.sh does this as its Isaac Lab step)."
    )

# 4. The IsaacLab clone carries the fork-side deliverables.
# isaaclab.__file__ = <IsaacLab>/source/isaaclab/isaaclab/__init__.py -> repo root is parents[3].
isaaclab_root = pathlib.Path(isaaclab.__file__).resolve().parents[3]
ui_ext = isaaclab_root / "source" / "newton_adaptive_ui"
if not ui_ext.is_dir():
    fail(
        f"newton_adaptive_ui extension not found at {ui_ext}.\n"
        "  The IsaacLab clone is not the fork's develop branch (the adaptive\n"
        "  GUI toggle ships there): `git -C <IsaacLab> checkout develop`."
    )
train_backend = isaaclab_root / "source" / "isaaclab_rl" / "isaaclab_rl" / "entrypoints" / "backends" / "train_rsl_rl.py"
if not train_backend.is_file() or "--solver" not in train_backend.read_text(encoding="utf-8", errors="ignore"):
    fail(
        f"--solver CLI wiring not found at {train_backend}.\n"
        "  The IsaacLab clone predates (or lost) the solver flag --\n"
        "  update the fork's develop branch."
    )

print(f"newton   : {newton.__version__}  (fork active, SolverMuJoCoAdaptive present)")
print(f"           {newton_path}")
print(f"sap      : {'ok  (SolverSAP + SolverSAPAdaptive)' if sap_ok else 'DISABLED (see warning above)'}")
print("isaaclab : ok  (isaaclab + isaaclab_newton import, newton_adaptive_ui + --solver present)")
print("OK: icra2027 platform verified.")
