# icra2027

Everything behind the ICRA 2027 CENIC-on-GPU submission in one repo: the
platform venv both code forks run in, the Part 1 pure-solver experiments and
their results, the Part 2 training campaign, and the paper package.

## Layout

| path | what |
|---|---|
| `install/`, `pyproject.toml`, `uv.lock` | the platform: Isaac Sim 6.0 wheels, cu128 torch, the `newton-adaptive` fork installed editable; `install/install.sh` builds `.venv` |
| `part1/` | pure-solver experiments (four configurations: MuJoCo / MuJoCo EC / ICF / ICF EC): scenes, benches, results, figures, tables, `results/PART1.md` |
| `part2/` | the training campaign: `trossen_campaign.sh`, probes, calibration results, run ledger |
| `paper/` | IEEE template (`conference_101719.tex` verbatim), `main.tex` outline, `figures/`, `references.bib`, the working notebook and its generator (`tools/make_notebook.py`) |

Code that must live inside the forks stays there: the task packages
(`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/trossen_*`) and the
solvers (`newton-adaptive/newton/_src/solvers/{mujoco,icf}`).

## Running

Python for everything: `icra2027/.venv/bin/python`. Isaac Lab resolves
`$VIRTUAL_ENV` first, so launch it with `VIRTUAL_ENV=$HOME/Documents/code/icra2027/.venv` exported.

* Part 1: `cd icra2027 && .venv/bin/python -m part1.bench.benchmarks.part1_penetration --scene hard-clutter`
* Part 2: `part2/trossen_campaign.sh` (read its preflight list first)
* Paper: `.venv/bin/python paper/tools/make_notebook.py`
