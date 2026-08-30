# Part 2 — the training campaign

Part 2 lives in the IsaacLab fork (branch `develop`) — Marco develops it
across machines there, so nothing is mirrored here (copies rotted within
two days of being made; 2026-08-30).

| what | where |
|---|---|
| task packages (lift, slide, plate_rack, flip, mug_rack, mug_tree) | `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/trossen_*` |
| campaign ladder (K1/K2/K3/adaptive/K3wall per task; flip adaptive override) | `IsaacLab/scripts/experiments/trossen_campaign.sh` |
| probes (scene, reward, banks, flip FSM, calibrations) | `IsaacLab/scripts/probes/` |
| run records | `IsaacLab/logs/rsl_rl/<task>/`, W&B project `rubato-trossen` |

Results and figures distilled for the paper land in this repo once runs
complete.
