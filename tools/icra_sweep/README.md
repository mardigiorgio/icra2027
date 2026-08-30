# icra_sweep — config-driven RL training sweeps on one GPU

One config file describes an experiment; the runner executes it, survives being
killed, and produces per-run JSON that a replicate-aware aggregator can read.

Every guard in here exists because the SAP campaign lost real GPU time to its
absence. The rules are listed at the bottom; read them before adding a cell.

## Layout

```
tools/sweep.py                     entry point
tools/icra_sweep/
    config.py       the schema, axis expansion, the two unit conversions
    gpu.py          census, admission control, the single-instance flock
    runner.py       launch, journal, skip-if-complete, abort conditions
    parse.py        rsl_rl log + solver telemetry -> one JSON per run
    analyze.py      replicate-aware aggregation, refuses n=1 differences
    preflight.py    dump both arms' resolved identity and diff them
    preflight_probe.py  the GPU-side dumper (runs under isaaclab.sh -p)
    crossplay.py    the 2x2 cross-play evaluation: policy x physics
    artifact.py     artifact-exploitation signatures and the 2x2 verdict (CPU)
    artifact_probe.py   the GPU-side rollout instrument and baseline measurer
    configs/        experiment configs, including the queued protocols
    tests/          CPU-only tests: python -m pytest tools/icra_sweep/tests -q
```

Reusable Python lives in `tools/` beside `wandb_done.py` and `dump_env_spec.py`;
a campaign's config, driver and (gitignored) run artifacts live in
`experiments/<campaign>/`, matching `archive/rubato-ppo-sweep/`.

## Worked example

`configs/trossen_sap_d7.yaml` is the sweep pass 31 ran by hand: fixed-step SAP
vs adaptive SAP, 3 seeds, 150 iterations, 1024 envs, arms interleaved.

Look before you leap — `plan` touches no GPU:

```
cd ~/Documents/code/icra2027
.venv/bin/python tools/sweep.py plan tools/icra_sweep/configs/trossen_sap_d7.yaml
```

It prints the cell order, which cells are already complete, the resolved
`--video_interval` in env steps, and the boundaries-per-iteration figure the
demand normalization uses. Then run the whole experiment in one command:

```
.venv/bin/python tools/sweep.py run tools/icra_sweep/configs/trossen_sap_d7.yaml
```

which (1) takes the sweep lock, (2) dumps both arms' resolved solver identity
and **aborts if they differ in anything not declared intended**, (3) runs each
incomplete cell with the device to itself, journalling a GPU census at every
boundary, and (4) writes `<run_name>.json` per run.

Re-running the same command after a kill resumes: complete runs are skipped, and
because arms are the innermost axis the series is always left on a complete
matched set.

Then:

```
.venv/bin/python tools/sweep.py analyze tools/icra_sweep/configs/trossen_sap_d7.yaml
```

which prints per-run cost (s/iter, samples/s, ms per accepted substep, GPU peak,
and whether the run had the device to itself) and, per metric, the between-arm
difference **against the within-arm replicate spread**.

## Cross-play: does a policy's score survive a different timestepper?

A training curve cannot tell you whether a policy learned the task or learned an
artifact of the integrator. Replaying it under the OTHER timestepper can:

```
.venv/bin/python tools/sweep.py crossplay <crossplay.yaml> --plan   # no GPU
.venv/bin/python tools/sweep.py crossplay <crossplay.yaml>          # 2x2 + video
.venv/bin/python tools/sweep.py artifact  <out_dir>                 # re-judge, no GPU
.venv/bin/python tools/sweep.py cost <N> --seeds 2                  # price it first
```

`crossplay` measures a per-physics BASELINE first (resting contact overlap, the
free-fall acceleration and energy residuals, the speed-comparison noise floor),
then runs every policy under every physics, writing per-step traces, a mean
episodic return and one video per cell. `artifact` judges the traces: four
invalid-configuration signatures (interpenetration past the object's own wall,
support with an empty contact set, energy gained in contact-free flight, the
object outrunning the gripper), the share of reward-bearing steps and of income
they account for, and a three-condition verdict over the 2x2.

Its rules, on top of the ones below:

10. **A threshold is a measured baseline or it does not exist.** Every signature
    is judged against a reference measured in the SAME physics arm by the same
    probe. A signature with no baseline reports `UNCALIBRATED` and is excluded
    from the verdict; it never falls back to a constant.

11. **`exploit_fraction` is a LOWER bound.** It is the union of the signatures
    that are live. An exploit none of them models is invisible, so a zero means
    "none of these four", never "the physics was valid".

12. **A truncated contact set voids every contact claim.** If the triangle-pair
    buffer overflowed, an empty contact set is not evidence of no contact and a
    shallow overlap is not evidence of no overlap. Every signature reports
    `VOID`.

13. **Transfer asymmetry alone is not exploitation.** The verdict requires
    asymmetry AND a mechanism (the fixed-trained policy's invalid-configuration
    rate exceeds the adaptive-trained policy's *under the same physics*) AND
    locality (that rate collapses when the physics changes). Asymmetry without a
    mechanism is reported as brittleness, which is a real and different finding.

14. **Watch the videos.** Every cell renders one to its own directory with its
    own filename prefix — `play` writes into the checkpoint's run directory,
    which made pass 30's cross cells overwrite the same-arm ones. `artifact`
    prints what to look for in each cell.

## The rules, and the failure each one encodes

1. **`--video_interval` counts ENV STEPS, not iterations.** This task runs
   `num_steps_per_env = 24`, so "every 10 iterations" is 240. Configs express
   the cadence in iterations; `VideoCfg.interval_env_steps` converts. Passing an
   iteration count straight through records 24x too often.

2. **Never quote a wall time from a run that shared the device.** Exclusivity is
   enforced by the runner (census + poll + flock), not by operator discipline.
   Packing is available for outcome-only sweeps but requires
   `timing_sensitive: false` in as many words, and every packed run is tagged
   `exclusive: false` so the analyzer can refuse to treat its wall as a timing.

3. **A startup abort is fatal to the sweep.** A run that exits non-zero before
   reaching iteration 0 means the stack is broken. Continuing once chained a
   second process onto a half-dead device. The runner raises instead.

4. **Arms are the innermost axis.** A sweep killed at any point has run complete
   matched sets. Pass 30 lost an arm to a mid-run kill and had to restart it.

5. **Two replicates minimum, or it is UNRESOLVED.** Pass 30's headline was
   refuted by its own accidental control: two same-seed adaptive runs differed
   2.4x in mean reward by iteration 9. `analyze.compare_arms` will not emit a
   difference from a single seed per arm; it emits UNRESOLVED and says why.

6. **Demand-normalize, and name the denominator.** `ms per accepted substep` is
   the price of integration work; raw wall is trajectory-confounded, because an
   arm that steers the policy somewhere more violent pays more wall for more
   work. There are two denominators — `num_steps_per_env` (env steps) and
   `num_steps_per_env * decimation` (solver boundaries) — and mixing them is a
   silent factor of `decimation`. `normalize_demand` returns both, named, and
   asserts their ratio.

7. **Preflight before you spend hours.** Presets, env vars and manager-side
   rules rewrite tolerances, capacities, contact law, determinism and
   containment on the way to the solver. The probe reads them off the
   constructed objects and the sweep aborts on any unintended difference.

8. **Record the stack, all of it.** `sap_warp` is joined by `sys.path`, so the
   physics can change with no commit in `newton-adaptive` or `IsaacLab`. Every
   run's JSON carries the short HEAD (and dirty flag) of all four repos.

9. **Overflow is silent; the parser makes it loud.** `Triangle pair buffer
   overflowed` warnings are captured per iteration and the first one is recorded
   as `first_overflow_iter`. Physics past that point is a truncated contact set;
   pass 30 spent two thirds of a comparison inside one without noticing.

## Status

The CPU paths (config, expansion, parsing, demand normalization, aggregation,
and every artifact signature) are covered by `tests/` and pass. The whole
artifact read/judge/report path has also been exercised end-to-end on synthetic
traces, including the null case.

`preflight_probe.py`, `artifact_probe.py` and the GPU launch paths were written
under a no-GPU rail and have **not been executed**; their solver access paths
are lifted from the campaign's proven probes (`p30_regime_probe.py`,
`p31_eval_probe.py`, `p32_pen_core.py`). Treat the first run as a shakedown, and
read the first baseline JSON before trusting anything downstream of it: its
resting overlap and free-fall residual are physically interpretable numbers, and
an absurd one means the probe is reading the wrong array.
