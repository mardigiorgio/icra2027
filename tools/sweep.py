#!/usr/bin/env python
"""Entry point for the rubato_sweep training harness.

    ~/Documents/code/IsaacLabRubato/.venv/bin/python tools/sweep.py plan <config.yaml>
    ~/Documents/code/IsaacLabRubato/.venv/bin/python tools/sweep.py run  <config.yaml>

Kept as a flat script beside the other tools/ helpers so it needs no packaging;
the library itself lives in tools/rubato_sweep/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rubato_sweep.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
