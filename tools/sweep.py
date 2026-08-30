#!/usr/bin/env python
"""Entry point for the icra_sweep training harness.

    ~/Documents/code/icra2027/.venv/bin/python tools/sweep.py plan <config.yaml>
    ~/Documents/code/icra2027/.venv/bin/python tools/sweep.py run  <config.yaml>

Kept as a flat script beside the other tools/ helpers so it needs no packaging;
the library itself lives in tools/icra_sweep/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from icra_sweep.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
