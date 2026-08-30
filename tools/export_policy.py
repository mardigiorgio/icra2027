"""Export a rubato-ppo-sweep rsl_rl actor to TorchScript.

Loads a model_*.pt checkpoint (rsl_rl 5.x layout: actor_state_dict with
mlp.N.* keys and a Gaussian distribution head), rebuilds the deterministic
actor MLP in plain torch, and saves a TorchScript module plus a sidecar
JSON with provenance. No rsl_rl import needed.

Usage:
    python export_policy.py --run velocity-flat-g1-mujoco-s46 [--checkpoint 1499]
    python export_policy.py --list
"""

import argparse
import glob
import json
import os
import re

import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP_LOGS = os.environ.get(
    "SWEEP_LOGS", os.path.join(_REPO_ROOT, "archive", "rubato-ppo-sweep", "logs", "rsl_rl", "g1_flat")
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exported")


def find_runs():
    return sorted(glob.glob(os.path.join(SWEEP_LOGS, "*")))


def build_actor(sd):
    """Rebuild the actor MLP (ELU activations) from mlp.N.weight shapes."""
    layer_ids = sorted(
        int(m.group(1)) for k in sd if (m := re.match(r"mlp\.(\d+)\.weight", k))
    )
    layers = []
    for i, lid in enumerate(layer_ids):
        w = sd[f"mlp.{lid}.weight"]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if i < len(layer_ids) - 1:
            layers.append(nn.ELU())
    net = nn.Sequential(*layers)
    # remap mlp.N.* onto sequential indices
    remap = {}
    seq_idx = [i for i, m in enumerate(net) if isinstance(m, nn.Linear)]
    for lid, si in zip(layer_ids, seq_idx):
        remap[f"{si}.weight"] = sd[f"mlp.{lid}.weight"]
        remap[f"{si}.bias"] = sd[f"mlp.{lid}.bias"]
    net.load_state_dict(remap)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="velocity-flat-g1-mujoco-s46",
                    help="substring matching one run dir under the sweep logs")
    ap.add_argument("--checkpoint", default=None,
                    help="iteration number (default: highest available)")
    ap.add_argument("--list", action="store_true", help="list runs and exit")
    args = ap.parse_args()

    runs = find_runs()
    if args.list:
        for r in runs:
            print(os.path.basename(r))
        return

    matches = [r for r in runs if args.run in os.path.basename(r)]
    if len(matches) != 1:
        raise SystemExit(
            f"--run '{args.run}' matched {len(matches)} runs: "
            f"{[os.path.basename(m) for m in matches]}"
        )
    run_dir = matches[0]

    if args.checkpoint is None:
        ckpts = glob.glob(os.path.join(run_dir, "model_*.pt"))
        it = max(int(re.search(r"model_(\d+)\.pt", c).group(1)) for c in ckpts)
    else:
        it = int(args.checkpoint)
    ckpt_path = os.path.join(run_dir, f"model_{it}.pt")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actor = build_actor(ckpt["actor_state_dict"]).eval()

    in_dim = actor[0].in_features
    out_dim = actor[-1].out_features
    example = torch.zeros(1, in_dim)
    scripted = torch.jit.trace(actor, example)

    os.makedirs(OUT_DIR, exist_ok=True)
    run_name = os.path.basename(run_dir)
    out_path = os.path.join(OUT_DIR, f"{run_name}_model_{it}.pt")
    scripted.save(out_path)

    meta = {
        "source_checkpoint": ckpt_path,
        "run": run_name,
        "iteration": it,
        "obs_dim": in_dim,
        "act_dim": out_dim,
        "obs_layout": [
            "base_lin_vel(3)", "base_ang_vel(3)", "projected_gravity(3)",
            "velocity_commands(3)", "joint_pos_rel(37)", "joint_vel(37)",
            "last_action(37)",
        ],
        "action_semantics": "joint position target = default_joint_pos + 0.5 * action",
        "control_hz": 50,
    }
    with open(out_path.replace(".pt", ".json"), "w") as f:
        json.dump(meta, f, indent=2)

    with torch.no_grad():
        y = actor(example)
    print(f"exported {out_path}")
    print(f"  obs {in_dim} -> act {out_dim}, zero-obs action mean {y.mean():+.4f}")


if __name__ == "__main__":
    main()
