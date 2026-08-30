# Part 2 — every hyperparameter of the training campaign

Sources: the synced IsaacLab `develop` tree (43a781e4bf, 2026-08-30): per-task
`trossen_*/trossen_*_env_cfg.py` and `agents/rsl_rl_ppo_cfg.py`,
`scripts/experiments/trossen_campaign.sh`, `flip_fsm_core.py`,
`hang_fsm_core.py`, `trossen_mug_lift/mdp.py`. **A** = assumed/inherited
default, flagged.

| # | task (gym id) | iters | rungs, in order |
|---|---|---|---|
| 1–7 | slide (`IsaacContrib-Slide-Mug-Trossen-v0`) | 700 | K1 → K2 → K3 → K4 → K5 → adaptive → K5wall |
| 8–14 | lift (`IsaacContrib-Lift-Mug-Trossen-v0`) | 1000 | same |
| 15–21 | plate (`IsaacContrib-PlatePick-Trossen-v0`) | 1000 | same |
| 22–28 | flip (`IsaacContrib-Flip-Mug-Trossen-v0`) | 1000 | same |
| 29–35 | tree (`IsaacContrib-MugHang-Trossen-v0`) | 1000 | same |

*Table 1. The run list: 35 runs — the identical seven rungs on every task
(Marco, 2026-08-30) — each 2048 envs, seed 42, solver ICF (`--solver icf` /
`icf-adaptive`), `physics=newton`, W&B project `rubato-trossen`, video 200
frames every 300 iterations. The story: K1–K5 are the time steps a
practitioner would plausibly guess; the claim is that no guessed rung trains
the hard tasks, and the adaptive arm does, at the same control boundary,
untuned. K5wall = fixed K5 rerun with the iteration budget matching the
adaptive run's measured wall clock (startup excluded, from the campaign's
own logs); skipped automatically if a donor rung is missing.
`trossen_mug_rack` exists on disk but is out of scope (ruling 2026-08-30).
MuJoCo arms are a separate pass, still owed to the paper.*

| rung | sim.dt | decimation | physics per 1/30 s boundary |
|---|---|---|---|
| K1 | 1/30 | 1 | 1 step |
| K2 | 1/60 | 2 | 2 steps |
| K3 | 1/90 | 3 | 3 steps |
| K4 | 1/120 | 4 | 4 steps |
| K5 | 1/150 | 5 | 5 steps |
| adaptive | 1/90 boundary seed | 3 | error-controlled march |
| K5wall | 1/150 | 5 | budget from adaptive wall |

*Table 2. The stepping ladder — identical on every task, 30 Hz control
everywhere; every rung passes explicit step overrides. Flip and tree are
AUTHORED at 1/450×15 (the step their design phase needed — measured
2026-08-29, the flip reaches 100 % deterministic true-home success by
iteration 500 there), but that step never runs in the campaign; it stands
as design-phase evidence only. Their adaptive rungs pin the boundary to
1/90×3 like every other task's.*

| PPO (shared base, lift's cfg) | value |
|---|---|
| steps per env per update | 24 |
| minibatches / epochs | 4 / 5 |
| learning rate | 1e-4, adaptive schedule, desired KL 0.01 |
| γ / λ | 0.98 / 0.95 |
| clip | 0.2 |
| entropy coef | 0.006 |
| actor = critic nets | MLP [256, 128, 64], ELU, no obs normalization |
| grad-norm cap | 1.0 |
| value loss | coef 1.0, clipped |
| NaN check | off (a fixed step driving state non-finite is the object of study) |

*Table 3. The shared PPO recipe; per-task deviations in Table 4.*

| task | init_std (log) | std range | entropy | γ | obs norm | other |
|---|---|---|---|---|---|---|
| slide | 1.5 | (0.05, 3.0) | 0.006 | 0.98 | off | — |
| lift | 0.5 | (0.05, 1.5) | 0.006 | 0.98 | off | — |
| plate | 1.5 | (0.05, 3.0) | 0.006 | 0.98 | off | — |
| flip | per-dim [0.12, 0.12, 0.12, 0.40, 0.05, 0.50, 0.30, 0.30], scalar 0.05 | (0.05, 2.5) | 0.0 | 0.99 | on | zero-init policy head (banked start holds) |
| tree | 0.5 **A** (inherited from lift; a cfg comment says the entropy measurement used 1.5 — confirm) | (0.05, 3.0) | 0.0 | 0.99 | on | — |

*Table 4. Per-task exploration — "the recipe is the law" (d23f9b8cdc), flip
and tree updated 2026-08-28/29.*

| task | episode | arm action scale | gripper scale | clip |
|---|---|---|---|---|
| slide | 6 s | j0–2: 0.5, j3: 1.0, j4–5: 1.5 | 0.15 | ±6 |
| lift | 5 s | 0.1 (all joints) | 0.05 | ±6 |
| plate | 5 s **A** (inherits lift) | as lift **A** | as lift **A** | ±6 |
| flip | 8 s | j0–2: 0.5, j3: 1.0, j4–5: 1.5 | 0.15 | ±6 |
| tree | 8 s | as flip family | 0.15 **A** | ±6 |

*Table 5. Episodes and action spaces (joint-position offsets on the
default pose).*

| gate | value |
|---|---|
| success position | within 5 cm of goal |
| success tilt | ≤ acos(0.87) ≈ 29.5° |
| success hold | 30 consecutive steps (slide: tracked ≥ 95 % of steps in the moving-goal band) |
| early termination penalty | −50 (crush-the-mug divergence exploit, bc8e951e89) |
| robot_abnormal | joint velocity > 25 rad/s |
| other terminations | physics_diverged, time_out, object_off_table (slide/flip) |

*Table 6. The shared success/termination economy.*

| task | reward economy |
|---|---|
| slide | reaching 0.5 (std 0.2) · goal tracking 5.0 (gaussian std 0.04 on the 0.20 m/s moving goal, FINAL_GOAL (0.285, 0)) · table scrape −2 · action rate −3e-3 · jerk −1e-3 · joint vel −5e-4 · early term −50 |
| lift | fingers_to_object 3.0 · position_tracking 5.0 · success 10 · good_finger_contact 0.75 · contact_count 0.1 · action_l2 −0.001 · early term −50 |
| plate | lift's economy with plate-specific overrides (weight 3.0 term re-pointed; see cfg) |
| flip | 7-stage FSM economy (`flip_fsm_core`): stage buckets (0,1,2,3,3,4,4), SUCCESS_BONUS 90, grasp latch past horizontal (cos 0), upright cos 0.87, strict PBRS at γ 0.99; rotation-path bank + tosspath banks; far-field action penalties |
| tree | staged hang FSM (`hang_fsm_core`): STAGE_C 4.0 potential per rung, SUCCESS_BONUS 200, persistence-gated advancement with regression, strict PBRS at γ 0.99 |

*Table 7. Reward economies — the staged tasks are potential-based
(milestones one-shot, Φ = STAGE_C·stage + φ_stage), so the table names the
economy's constants rather than flattening the FSM into fake term weights.*

| solver / contact | value |
|---|---|
| ICF contact stiffness | icf_warp default 1e5 N/m **A** — the campaign exports no ICF_CONTACT_STIFFNESS (open question, notebook B21) |
| ICF HC dissipation | icf_warp default 10 s/m **A** |
| ICF stiction tolerance / σ | 1e-4 m/s / 1e-3 **A** |
| ICF contact cap | plate 8192, all others 1024 (`ICF_MAX_RIGID_CONTACT`, exported per task by the campaign) |
| adaptive tolerance | solver default ε = 1e-3 **A** (no NEWTON_ADAPTIVE_TOL export); seed δt = the task's sim.dt |
| MuJoCo-terms authoring (scenes) | condim 6, solref (0.02, 1.0) = MuJoCo default τ 20 ms; friction mug 0.2, plate 0.3, pads 1.0, table 0.6 |
| collision representation | per-piece convex hulls everywhere; only the dishrack a raw tri mesh (ruling 2026-08-25); rack task adds hull/bay/cage variants via RACK_COLLISION |

*Table 8. Solver-side settings. The **A** rows are defaults reaching the
runs implicitly — decide whether to export them explicitly before the
campaign so the paper can state them as chosen.*

## Campaign infrastructure
Launcher: `IsaacLab/scripts/experiments/trossen_campaign.sh` — waits for GPU
drain (up to 5 min), 5 launch attempts per run with stall detection (no
"Learning iteration" + CPU < 5 % for 5 checks), logs under
`$CAMPAIGN_LOG_DIR` (default /tmp/trossen_campaign), TROSSEN_RAILS=1.
Preflight, per its header: coefficient probe exits 0, mug convexified,
flip settle+reward smokes passed, GOAL_SPEED confirmed (0.20 committed),
GPU free.
