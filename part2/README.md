# Part 2 — the training campaign

The tasks (slide, lift, plate-from-rack, flip, mug-on-tree) are Isaac Lab task
packages in the IsaacLab fork, branch `develop`:
`source/isaaclab_tasks/isaaclab_tasks/contrib/trossen_{mug_slide,mug_lift,plate_rack,mug_flip}`.
The solver is selected per run through the Newton backend (`NewtonMJWarpManager`).

| here | what |
|---|---|
| `trossen_campaign.sh` | the campaign, per task: K1, K2, K3, adaptive, K3wall (fixed rerun at the adaptive run's wall clock); preflight list in the header |
| `probes/` | scene/reward/contact probes (`probe_*.py`), run from the IsaacLab root: `./isaaclab.sh -p ../icra2027/part2/probes/<probe>.py` |
| `probes/results/` | contact-compliance calibrations (`cal_*.json`) |

Run records: `IsaacLab/logs/rsl_rl/<task>/<timestamp>_<name>/` (checkpoints,
params, videos) and the W&B project `rubato-trossen` (the project name is a
live identifier of finished runs and is not renamed).
