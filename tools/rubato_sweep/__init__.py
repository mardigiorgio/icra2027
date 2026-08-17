"""rubato_sweep -- config-driven, restartable RL training sweeps on one GPU.

    python tools/sweep.py plan     configs/<experiment>.yaml   # no GPU, shows the plan
    python tools/sweep.py run      configs/<experiment>.yaml   # preflight + execute
    python tools/sweep.py analyze  configs/<experiment>.yaml   # replicate-aware report

See README.md for the worked example and the rules each guard exists to enforce.
"""

from .config import Arm, Cell, ConfigError, SweepCfg  # noqa: F401

__all__ = ["Arm", "Cell", "ConfigError", "SweepCfg"]
