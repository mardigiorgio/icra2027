"""Command line: one config file in, one experiment out."""

from __future__ import annotations

import argparse
import os
import sys

from . import analyze, artifact, crossplay as crossplay_mod, gpu, parse, preflight as preflight_mod, runner
from .config import SweepCfg


def _cfg(path: str) -> SweepCfg:
    cfg = SweepCfg.from_file(path)
    os.makedirs(cfg.out_dir, exist_ok=True)
    return cfg


def cmd_plan(args) -> int:
    """CPU-only: print the execution order, the resolved command and the cost."""
    cfg = _cfg(args.config)
    cells = cfg.cells()
    print(f"sweep      : {cfg.name}   ({cfg.source})")
    print(f"task       : {cfg.task}")
    print(f"arms       : {', '.join(a.name for a in cfg.arms)}")
    print(f"axes       : envs={list(cfg.env_counts)} iters={list(cfg.horizons)} seeds={list(cfg.seeds)}")
    print(f"cells      : {len(cells)}")
    print(f"video      : every {cfg.video.every_iterations} iterations "
          f"= --video_interval {cfg.video_interval_env_steps()} env steps "
          f"(num_steps_per_env={cfg.num_steps_per_env})")
    print(f"boundaries : {cfg.boundaries_per_iteration()} solver boundaries per iteration "
          f"(num_steps_per_env x decimation={cfg.decimation})")
    print(f"timing     : {'MEASUREMENT-GRADE (exclusive)' if cfg.timing_sensitive else 'outcome-only'}"
          f"{'  PACKED x' + str(cfg.packing.max_concurrent) if cfg.packing.enabled else ''}")
    print(f"out_dir    : {cfg.out_dir}")
    if args.cost_per_iter:
        print("\nprojected cost (s/iter supplied on the command line -- PROJECTION, not a measurement):")
        total = 0.0
        for c in cells:
            per = args.cost_per_iter.get(c.arm.name)
            if per is None or c.iterations is None:
                continue
            total += per * c.iterations
        print(f"  total {total / 3600:.1f} GPU-hours over {len(cells)} cells")
    print("\nexecution order (arms innermost: a kill lands on a complete matched set):")
    for i, group in enumerate(cfg.matched_groups()):
        for c in group:
            done = parse.is_complete(cfg.log_path(c), c.iterations)
            print(f"  set {i:>3} [{'COMPLETE' if done else '  todo  '}] {c.run_name}")
    if args.verbose:
        print("\nresolved commands:")
        import shlex

        for c in cells:
            print("  " + shlex.join(runner.build_command(cfg, c)))
    return 0


def cmd_run(args) -> int:
    cfg = _cfg(args.config)
    if args.no_preflight:
        cfg.preflight = type(cfg.preflight)(**{**cfg.preflight.__dict__, "enabled": False})
    with gpu.SweepLock(cfg.lock_path()):
        if cfg.preflight.enabled:
            preflight_mod.preflight(cfg)
        else:
            print("preflight DISABLED -- any between-arm difference this sweep reports is unverified")
        runner.run_sweep(cfg, dry_run=False)
    return 0


def cmd_preflight(args) -> int:
    cfg = _cfg(args.config)
    with gpu.SweepLock(cfg.lock_path()):
        preflight_mod.preflight(cfg)
    return 0


def cmd_analyze(args) -> int:
    cfg = _cfg(args.config) if args.config.endswith((".yaml", ".yml", ".json")) else None
    out_dir = cfg.out_dir if cfg else args.config
    analyze.report(out_dir, window=args.window)
    return 0


def cmd_crossplay(args) -> int:
    """The 2x2 cross-play evaluation and its artifact analysis."""
    cfg = crossplay_mod.CrossplayCfg.from_file(args.config)
    os.makedirs(cfg.out_dir, exist_ok=True)
    if args.plan:
        crossplay_mod.run(cfg, dry_run=True)
        return 0
    with gpu.SweepLock(os.path.join(cfg.out_dir, ".lock")):
        crossplay_mod.run(cfg, dry_run=False)
    return 0


def cmd_artifact(args) -> int:
    """CPU-only: re-judge existing traces. Touches no GPU."""
    out_dir = args.target
    if out_dir.endswith((".yaml", ".yml", ".json")):
        out_dir = crossplay_mod.CrossplayCfg.from_file(out_dir).out_dir
    artifact.report(out_dir)
    return 0


def cmd_cost(args) -> int:
    """CPU-only: what the matched-N protocol costs, as a function of N."""
    est = crossplay_mod.wall_estimate(args.iterations, args.seeds, s_per_iter_fixed=args.s_per_iter)
    print(f"matched-N protocol at N={args.iterations}, {args.seeds} seed(s) per arm:")
    print(f"  fixed-arm training  {est['fixed_training_h']:6.2f} h   "
          f"({args.seeds} x {args.iterations} x {args.s_per_iter:.3f} s/iter, MEASURED s/iter)")
    print(f"  cross-play + video  {est['crossplay_h']:6.2f} h   (PROJECTED cell cost)")
    print(f"  TOTAL               {est['total_h']:6.2f} h")
    print("  the adaptive arm is NOT priced: its policy at iteration N comes from the run already in flight")
    return 0


def cmd_census(args) -> int:
    print(gpu.census().line())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sweep", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("plan", help="print the plan; touches no GPU")
    pl.add_argument("config")
    pl.add_argument("-v", "--verbose", action="store_true")
    pl.add_argument("--cost-per-iter", type=_kv_floats, default={},
                    help="arm=seconds,... to print a projected cost, e.g. fixed=8.1,adaptive=18.2")
    pl.set_defaults(func=cmd_plan)

    r = sub.add_parser("run", help="preflight, then execute every incomplete cell")
    r.add_argument("config")
    r.add_argument("--no-preflight", action="store_true", help="skip the parity check (not recommended)")
    r.set_defaults(func=cmd_run)

    pf = sub.add_parser("preflight", help="dump and diff the arms' resolved identity")
    pf.add_argument("config")
    pf.set_defaults(func=cmd_preflight)

    an = sub.add_parser("analyze", help="replicate-aware report over a sweep's JSON records")
    an.add_argument("config", help="config file or an output directory")
    an.add_argument("--window", type=int, default=50, help="iterations averaged at the tail of each run")
    an.set_defaults(func=cmd_analyze)

    xp = sub.add_parser("crossplay", help="2x2 cross-play evaluation + artifact analysis")
    xp.add_argument("config")
    xp.add_argument("--plan", action="store_true", help="print the cells and resolved checkpoints; no GPU")
    xp.set_defaults(func=cmd_crossplay)

    ar = sub.add_parser("artifact", help="re-judge existing cross-play traces (CPU only)")
    ar.add_argument("target", help="cross-play config file or an output directory")
    ar.set_defaults(func=cmd_artifact)

    co = sub.add_parser("cost", help="price the matched-N protocol (CPU only)")
    co.add_argument("iterations", type=int, help="N, the matched horizon")
    co.add_argument("--seeds", type=int, default=2, help="training seeds per arm (default 2, the minimum)")
    co.add_argument("--s-per-iter", type=float, default=8.067,
                    help="fixed-arm seconds per iteration (default: pass-31 measured 8.067 at 1024 envs)")
    co.set_defaults(func=cmd_cost)

    cs = sub.add_parser("census", help="print the current GPU compute-app census")
    cs.set_defaults(func=cmd_census)

    args = p.parse_args(argv)
    return args.func(args)


def _kv_floats(text: str) -> dict[str, float]:
    out = {}
    for part in text.split(","):
        if not part.strip():
            continue
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


if __name__ == "__main__":
    sys.exit(main())
