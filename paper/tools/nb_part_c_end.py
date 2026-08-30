"""Parts C (paper blocks), D (fact bank, decisions, repo map, plan), E (glossary), appendices F, G, H."""


def section(nb, title, questions, evidence, notes=5):
    nb.h(title, 2)
    nb.p("Purpose (one sentence, yours): ", bold=True); nb.lines(1)
    nb.p("Questions this section has to answer:", bold=True); nb.bl(questions)
    nb.p("Evidence on hand (pointers):", bold=True); nb.bl(evidence)
    nb.p("Your notes / draft:", bold=True); nb.lines(notes)
    nb.p("Status:  ☐ outlined   ☐ drafted   ☐ figures placed   ☐ read by Vince   ☐ done", italic=True)


def partC(nb):
    nb.pagebreak()
    nb.title("Part C — The paper, section by section")
    nb.p("Write the purpose sentence first; answer the questions in your words; pull numbers only from Part D; tick the boxes. Nothing here is text for the paper.", italic=True)
    nb.h("I. Introduction / Motivation", 1)
    section(nb, "Introduction", ["What does a fixed step do to stiff contact that a learner then exploits? (A1, B12, B20)", "Why is CENIC the right fix and what was missing to use it for robot learning? (A5–A6)", "What exactly is new vs CENIC? (A7: GPU, per-world, boundaries, speed)", "What is the one experiment the reader should remember? (B1, B20)"], ["CENIC Sec. I for cadence", "PART1_LITERATURE.md Theme D (artifact exploits: Cheney 2013, Factory 2022)", "CLAUDE.md Goal"])
    nb.h("II. Background", 1)
    section(nb, "Compliant contact and convex solvers", ["MuJoCo's solref and the clamp (A3, Appendix F)", "ICF and why convexity matters (A4)"], ["tables/mujoco_stiffness_probe.md", "CENIC Sec. II-E, IV", "references todorov2014, sap, icf"])
    section(nb, "Error-controlled integration for contact (CENIC)", ["Step doubling, the norm, the constants (A5)", "Reporting standard (Theme C)"], ["CENIC Sec. II-C, V, VII, Table III"])
    section(nb, "GPU simulation for robot learning", ["What RL imposes; why per-world adaptivity is not free (A6)"], ["Theme D references"])
    nb.h("III. CENIC on the GPU — implementation and speed", 1)
    section(nb, "Inner steps between fixed boundaries", ["Per world / per batch / on the host (A7)", "Landing on the boundary; δt_min/init/max", "Exhausted or diverged worlds"], ["solver docstrings (A7)", "four_arms.py", "a schematic figure — NOT MADE"])
    section(nb, "Error estimate and step selection on the device", ["Three solves, two queries (A5, A7)", "Newton tolerance rule (B14)", "Contact budgets (B16)"], ["tables/newton_tolerance_probe.md", "PART1.md 'Contact budgets'"])
    section(nb, "Speed optimizations (contribution 3)", ["Each mechanism in force in the ICF and MuJoCo arms, with a before/after number (Appendix G)", "The one probe still missing"], ["ledger copy in template_extras/", "tables/march_cost.md"])
    nb.h("IV. Results I — pure solver", 1)
    for t, q, e in [("Setup", ["Paper's vs assumed parameters (B2)", "MuJoCo calibration and the soft-clutter offset (B13, B3)"], ["cenic_scenes.py", "figures/scenes.pdf"]),
                    ("Realized stiffness", ["What each solver realizes at 10 ms / 1 ms / EC (B3)"], ["figures/stiffness_sweep.pdf"]),
                    ("Energy convergence", ["Orders; the honest ordering; the MuJoCo-EC gain (B4)"], ["figures/ball_energy.pdf, ball_workprecision.pdf"]),
                    ("Work-precision", ["Cost vs requested accuracy; per-attempt cost (B5, B15)"], ["figures/workprecision.pdf, speed_bars.pdf; tables/part1_table1.md"]),
                    ("Contact artifacts vs cost", ["Cheapest artifact-free per arm; who ejects (B6)"], ["figures/artifacts.pdf"]),
                    ("Error control pays when something happens", ["Where the saving is (B7)"], ["figures/realtime_trace_n64.pdf"]),
                    ("Measured error vs cost", ["The floor; where EC beats fixed at matched measured error (B9, B10)"], ["figures/consistency.pdf; tables/determinism_probe.md"]),
                    ("Wall vs worlds", ["Per-world cost at 2^13; the ICF/MuJoCo ratio (B8)"], ["figures/scaling_per_world_hard-clutter.pdf"]),
                    ("Actuated stiff contact", ["Stability map; the pushed box; what 10 ms hides (B12)"], ["figures/actuated.pdf, actuated_chatter.pdf"])]:
        section(nb, t, q, e)
    nb.h("V. Results II — application", 1)
    section(nb, "Protocol", ["Ladder K1/K2/K3, K3wall, adaptive; held-fixed set; success gates; representation (B17, B25)"], ["trossen_campaign.sh; probe_campaign_coefficients.py"])
    for task, ref in [("Slide mug across table", "B20"), ("Pick up mug (lift)", "B21"), ("Pick up dish from dish rack (plate)", "B22"), ("Mug on tree", "B24"), ("Flip mug by handle", "B23")]:
        section(nb, task, [f"Which arm trains a working policy; which fails or exploits what — with the metric and the video frame ({ref})", "Wall per iteration at matched accuracy; per-world accepted-substep counts"], ["W&B: ____", "video: ____", "run dir: ____"], notes=4)
    nb.h("VI. Real-robot experiments (Trossen WXAI)", 1)
    for task in ["Flip mug by handle", "Slide mug across table"]:
        section(nb, task, ["Which policy, how many trials, success rate; deployment (goal_time, tolerances, gains); what the fixed-step policy did (B28)"], ["hardware notes: ____", "video: ____"], notes=4)
    nb.h("VII. Limitations and conclusion", 1)
    section(nb, "Limitations / conclusion", ["Open items (Part D.2); which solver the speed claims are about; the near-rigid-regime question (D12)", "One sentence closing on the title claim"], ["tables/determinism_probe.md", "PART1.md open items", "ledger DECISION SUMMARY"])


def partD(nb):
    nb.pagebreak()
    nb.title("Part D — Fact bank, decisions, repo map, plan")
    nb.h("D1. Fact bank — measured numbers with their source (Part 1 re-runnable; Part 2 / campaign as logged)", 1)
    nb.tbl([["Fact", "Value", "Source"],
            ["Cheapest artifact-free, hard clutter, 64 scenes (wall per simulated s)", "MuJoCo fixed 2 ms 0.57; MuJoCo EC ε=1e-3 1.9; ICF EC ε=1e-2 2.3; ICF fixed 1 ms 4.1", "figures/artifacts.pdf; tables/results_tables.md"],
            ["Fixed ICF at 10 ms, hard clutter", "no ejections on the corrected scene; max pen 4.1 mm = 1.8x the impact depth", "part1_penetration_hard-clutter.csv"],
            ["Mean penetration hard clutter 10/5/2/1 ms (µm)", "ICF 226/34.6/8.4/6.4; MuJoCo 336/100/17/6.6; model 6.4", "tables/results_tables.md"],
            ["Work-precision hard, N=1, ε=1e-1..1e-6 (s per sim-s)", "ICF EC 0.38/0.60/1.35/2.89/7.91/18.2; MuJoCo EC 0.092/0.30/1.04/2.92/5.63/13.8", "tables/results_tables.md"],
            ["Work-precision hard, N=1024", "ICF EC 15/23.8/49.3/115/343/856; MuJoCo EC 0.47/1.54/5.02/15.8/37.8/88.4", "tables/results_tables.md"],
            ["Fixed levels hard N=1 / N=1024 (10,5,2,1 ms)", "ICF 0.20/0.29/0.59/1.13 and 7.5/10.9/20.9/38.2; MuJoCo 0.064/0.125/0.34/0.66 and 0.18/0.39/1.04/2.11", "tables/results_tables.md"],
            ["March cost hard ε=1e-3", "ICF 736 iters/sim-s at 3.1 ms; MuJoCo 699 at 2.1 ms", "tables/march_cost.md"],
            ["MuJoCo realized-stiffness cap", "~1e3 N/m at 10 ms; 1e5 at 1 ms; EC ε=1e-3 no recovery (17× at 1e5); ε=1e-5 recovers to the 1 ms level", "part1_stiffness_sweep.csv"],
            ["ICF realized stiffness", "k to 1e7 at 10 ms/1 ms/ε=1e-3/1e-5 (0.99); 0.87 at 1e8", "part1_stiffness_sweep.csv"],
            ["Ball, ICF fixed", "first order from 0.1 ms: −30.0/−16.3/−6.25/−3.17 % at 100/50/20/10 µs; 11 rebounds ≤ 0.1 ms; rest by 10 s at ≥ 1 ms", "part1_ball_energy.csv (last apex)"],
            ["Ball, MuJoCo fixed (direct undamped)", "−7.1 % 10 ms; −7.3 % 5 ms; +0.78 % 2 ms; −0.02 % 1 ms; ≤ 0.004 % below; 10 rebounds", "part1_ball_energy.csv"],
            ["Ball, MuJoCo error control", "+0.8 % ε ≥ 1e-2; +57/+22/+13/+4.0 % at 1e-3..1e-6 — OPEN DEFECT", "part1_ball_energy.csv; probe_ball_ec_energy.log"],
            ["Ball, ICF error control", "~−100 % ε ≥ 1e-4; −51 % at 1e-5 (budget-exhausted, 11 rebounds); exhausted at 1e-6", "part1_ball_energy.csv"],
            ["Wall per world per 100 ms boundary, 2^13 hard", "MuJoCo fixed 11 µs; MuJoCo EC 270 µs; ICF fixed 620 µs; ICF EC 1.5 ms (per boundary: 90.7 / 2190 / 5060 / 12,700 ms)", "tables/results_tables.md"],
            ["ICF/MuJoCo per-world cost at 2^13", "~55× fixed; ~6× EC; Newton tolerance 1e-5..1e-8 changes wall < 5 %", "tables/newton_tolerance_probe.md"],
            ["Real-time trace, 5 s drop, 64 scenes", "ICF EC ε=1e-2 ~10 % RT in impacts, > 100 % settled; ~half the cost of fixed ICF 1 ms", "figures/realtime_trace_n64.pdf"],
            ["Consistency floor (reference vs itself)", "soft 11 µm ICF / 0.2 µm MuJoCo; hard 0.62 / 0.28 mm", "part1_consistency_*.csv"],
            ["Consistency soft", "first order both; ICF EC ~2× less deviation than fixed ICF at equal cost; MuJoCo EC on its fixed line", "part1_consistency_soft-clutter.csv"],
            ["Consistency hard", "~O(δt^0.7); tight end within 2–4× of the floor; ICF EC ε=1e-3 measures 5.6 mm", "part1_consistency_hard-clutter.csv"],
            ["Determinism", "ball bit-exact; clutter not reproducible: ICF mm within 0.3 s (soft), both cm within 0.5 s (hard); MuJoCo µm on soft", "tables/determinism_probe.md"],
            ["Momentum, zero-g head-on", "all arms ≤ 1e-5 (ICF ≤ 1e-7; EC exact)", "tables/momentum_probe.md"],
            ["Actuated: stability", "ICF 40/40; MuJoCo unstable in 6 (K_p=1e5 δt ≥ 5 ms; 1e6 δt ≥ 2 ms & ε=0.1)", "part1_actuated.csv"],
            ["Actuated: box", "ICF lift ≤ 0, pitch ≤ 0.02 rad/s, 0.280 m (0.235 at K_p=1e2). MuJoCo lift 2–21 mm (≤1e4), 4–9 (1e5), 10–58 (1e6); pitch 0.7–3.4 rad/s; tip over box at 1e2; EC ε=1e-4 still 1.7–10 mm", "part1_actuated.csv; tables/actuated_trace.md"],
            ["Actuated: chatter at K_p=1e5 (tip–box rel. vel. RMS)", "ICF 0/0.083/0.20/0.27 m/s at 10/5/2/1 ms; MuJoCo 0.36 at 1 ms; ICF EC 0.08 (ε ≤ 1e-3), 0.26 (1e-4)", "part1_actuated.csv"],
            ["MuJoCo direct solref", "exact compliance at k=1e5 for δt ≤ 2 ms; launches at ≥ 5 ms (ω·δt ≲ 2)", "tables/mujoco_stiffness_probe.md"],
            ["Soft-clutter MuJoCo calibration", "τ=31.8 ms measured anchor; ratio 1.00 at k=10³ on every line", "part1_stiffness_sweep.csv"],
            ["Contact demand", "hard ~520/world ICF, ~380 MuJoCo; soft ~280/290", "verify_contact_budgets.py"],
            ["Plate representation", "tri–tri 3624 contacts/env, 96 s/iter at 2048 envs; hulls 4–5 s/iter", "IsaacLab 0c535d1668"],
            ["Slide, latest pairs (as logged)", "current rule (tracked ≥ 95 %): K3 0 (tracked 0.61, 1999 it), adaptive 0 (0.77 at 501/1000); previous held-delivery gate: K3 0.842 / adaptive 0.988", "logs/rsl_rl/trossen_mug_lift/smoke7-*, smoke-*"],
            ["Lift / plate / flip smokes (as logged)", "success_rate 0 in every campaign-shape smoke (lift Episode_Reward/success ≈ 6.7)", "logs/rsl_rl"],
            ["ICF adaptive march cost split", "narrow-phase 84.5 %, Cholesky 0.3 % of GPU time", "icf_warp_adaptive DEVIATIONS.md / IsaacLab 639ab761e5"],
            ["WXAI hardware", "goal_time 0.1 s at 30 Hz (Trossen LeRobot path); effort 27/7 N·m, gripper 100 N (sim 400); position cascade K/D = position kp", "WXAI brief"]])
    nb.h("D2. Open decisions (yours / Vince's) — merged from CLAUDE.md, PART1.md, the ledger's DECISION SUMMARY, the record", 1)
    nb.bl(["☐ Title; ☐ authors and affiliations; ☐ Overleaf project shared with vkurtz1@depaul.edu",
           "☐ Contribution 3: a before/after table on a paper scene for the ICF/MuJoCo arms does not exist — commission the probe or scope the claim",
           "☐ Assumed clutter parameters — confirm with Vince or state as assumed (B2)",
           "☐ Soft-clutter MuJoCo τ offset: state or rerun (B3)",
           "☐ MuJoCo-EC energy gain: fix SolverMuJoCoAdaptive or report as open (B4)",
           "☐ δt_max = 0.1 s on clutter (paper) vs 10 ms boundary in RL — how to present",
           "☐ Which Part-1 figures fit the page budget (14 exist)",
           "☐ Speed section: the missing before/after probe on a paper scene (Appendix G)",
           "☐ Per task: which run pair is the result, under which success definition (B20–B23); K3wall runs; MuJoCo arms; ICF stiffness pinned in the campaign",
           "☐ Mug-on-tree: where it lives (B24)",
           "☐ Does the paper make a near-rigid-regime claim for Part 2 at all (the training scenes use MuJoCo's default soft solref and a global ICF stiffness — Appendix F.4, B25)?",
           "☐ Determinism: how many seeds per comparison (single-seed differences are inside the run-to-run noise)",
           "☐ IRL: which policies go on the arm; action parameterization (delta vs absolute) and smoothing (B28)",
           "☐ GitHub push (auth broken, D14) — nothing this week is pushed"])
    nb.h("D3. Repo map and how to re-run", 1)
    nb.tbl([["What", "Where", "How"],
            ["Scenes", "newton-adaptive/scripts/scenes/cenic_scenes.py, actuated_press.py", "imported by the benches"],
            ["Four-arm harness", "scripts/bench/four_arms.py", "make_arm(model, name, ...) — captured boundaries, trackers, budgets, solref, K_INIT, eq. 34 rule"],
            ["Part-1 benches", "scripts/bench/benchmarks/part1_{workprecision, penetration, scaling, ball_energy, realtime_trace, consistency, stiffness_sweep, actuated}.py", "cd newton-adaptive; ~/Documents/code/IsaacLabRubato/.venv/bin/python -m scripts.bench.benchmarks.part1_<name> [--scene ...]; timing benches alone on the GPU; one subprocess per config"],
            ["Probes / certificates", "scripts/bench/probe_momentum.py, probe_determinism.py, probe_actuated_trace.py, probe_march_cost.py, verify_contact_budgets.py, verify_part1_penetration.py; part1_consistency.py --self-check", "same python; read the docstring first"],
            ["Figures / tables", "scripts/bench/part1_plots.py; part1_tables.py; part1_results_md.py; part1_scenes_figure.py", "CPU only; regenerate after any sweep"],
            ["Results write-up", "scripts/bench/results/PART1.md; tables/*.md; figures/*.pdf; PART1_LITERATURE.md", "the paper's Results I and Background are read from here"],
            ["Solvers", "newton/_src/solvers/mujoco/solver_mujoco_adaptive.py; adaptive_boundary.py; icf_warp_adaptive/icf_warp/{solver.py, solver_adaptive.py, contact_law.py, kernels_dof.py, DEVIATIONS.md}", "flags in A7 / Appendix G"],
            ["Training (Part 2)", "IsaacLab develop: contrib/trossen_{mug_lift, mug_slide, plate_rack, mug_flip}; scripts/experiments/trossen_campaign.sh; scripts/probes/*", "VIRTUAL_ENV=~/Documents/code/IsaacLabRubato/.venv ./isaaclab.sh -p ... --solver icf|icf-adaptive|mujoco|mujoco-adaptive; ICF_MAX_RIGID_CONTACT=1024/8192; W&B + video; one knob at a time; kill doomed runs"],
            ["Runs", "IsaacLab logs/rsl_rl/{trossen_mug_lift, trossen_mug_flip, trossen_spatula_lift}; wandb/", "run dirs map to W&B by timestamp; killed runs have no wandb-summary.json — grep output.log"],
            ["Paper package", "~/Documents/code/cenic-paper (conference_101719.tex verbatim, main.tex outline, figures/, references.bib, this notebook, tools/make_notebook.py)", "upload cenic-paper.zip to Overleaf; regenerate the notebook with PYTHONPATH=<pylib> python tools/make_notebook.py"]])
    nb.h("D4. Deadline plan (fill in)", 1)
    nb.tbl([["Day", "Sections to write", "Runs to launch / watch", "Done?"], ["Sat 08-30", "", "", "☐"], ["Sun 08-31", "", "", "☐"], ["Mon 09-01 (deadline)", "", "", "☐"]])


def partE(nb):
    nb.pagebreak()
    nb.title("Part E — Glossary")
    nb.tbl([["Term", "Meaning here"],
            ["World / environment (env)", "one independent copy of the scene simulated in parallel on the GPU"],
            ["Control boundary / outer step (dt_outer)", "the fixed interval at which the policy acts and the state is handed back; 1/30 s in training, 10 ms in the benches (0.1 s on the clutter benches = CENIC's δt_max)"],
            ["Inner step (δt)", "the solver's own time step inside a boundary; per world under error control"],
            ["Substep / decimation (K1/K2/K3)", "fixed substeps per control step in IsaacLab: K1 = 1 × 1/30 s, K2 = 2 × 1/60 s, K3 = 3 × 1/90 s"],
            ["ε_acc (accuracy, tol)", "the per-step position error tolerance the controller enforces (metres); 1e-3 in training"],
            ["Step doubling", "one full step vs two half steps; their difference is the local error estimate; the half-step result is kept"],
            ["Accept / reject", "commit and grow δt, or restore the snapshot and shrink δt"],
            ["k_Init, δt_max, δt_min", "first attempt = k_Init·δt_max (0.1 in benches; in training the seed is the fixed arm's step); ceiling and floor of the inner step"],
            ["March budget / exhaustion", "max inner attempts per boundary (256 default, 4096 in benches); a world that hits it is exhausted (latched diverged on the parity branch)"],
            ["Divergence / physics_diverged", "non-finite state or exhaustion in a world; per-world flag → episode termination with a −50 reward"],
            ["Contact budget", "max contacts stored (NCONMAX/NJMAX 1024, ICF_MAX_RIGID_CONTACT 2048 in benches; 1024/8192 in training); too small = silently dropped contacts"],
            ["Penetration / resting depth / impact depth", "overlap of bodies; m g / k; v √(m/k)"],
            ["Artifact", "a behaviour the step made, not the model: penetration above the impact depth, ejection, instability, hop, hidden chatter"],
            ["Passthrough / ejection", "a body moving more than its radius per step enters contact past its centre and is launched"],
            ["Chatter", "rapid make/break contact oscillation; here tip–box relative velocity in the cruise"],
            ["k, d, μ, v_s", "contact stiffness [N/m]; Hunt–Crossley dissipation [s/m]; friction coefficient; stiction tolerance velocity"],
            ["solref / solimp / refsafe", "MuJoCo's contact parameters (τ, ζ) or (−k, −b); impedance curve; the τ ≥ 2δt clamp"],
            ["Convex solver / Newton iteration / line search", "the per-step velocity problem is a convex minimization solved by Newton's method (exact line search in ICF)"],
            ["ICF / CENIC", "irrotational contact fields, the convex compliant-contact formulation (Castro et al.); CENIC = ICF + error-controlled step doubling (Kurtz & Castro); in our code ICF = icf_warp_adaptive"],
            ["Near-rigid regime", "the stiff-contact regime where ICF's regularization scales as 1/δt² and CENIC's mechanism bites; whether the training scenes are in it is an open decision"],
            ["Fixed arm / error-controlled (EC, adaptive) arm", "same solver with a fixed δt or with the per-world controller; four arms = 2 solvers × 2"],
            ["Wall-matched arm (K3wall)", "a fixed-step arm run for the wall-clock budget of the EC arm"],
            ["Matched accuracy", "wall-time comparisons only between artifact-free settings"],
            ["Work-precision", "wall time per simulated second vs requested accuracy (or vs measured error, B9)"],
            ["Self-consistency / floor row", "deviation from the solver's own fine-step reference after restarted windows; the reference vs itself = the noise floor"],
            ["Determinism", "same inputs → bit-identical outputs; false on clutter for both solvers"],
            ["CUDA graph / captured boundary / conditional tier", "a recorded launch sequence replayed without the CPU; each boundary of every arm is one; the conditional tier records the whole march as a while-node"],
            ["Host sync", "the GPU waits while the CPU reads a value; the 4-byte 'unfinished?' flag per iteration"],
            ["Tail / march compaction; narrowing", "restricting launches to still-active worlds (the campaign's scheduling mechanisms)"],
            ["Newton / Warp / MuJoCo Warp / IsaacLab / rsl_rl", "NVIDIA's engine; its GPU Python framework; DeepMind's GPU MuJoCo; the RL environment framework; the PPO library"],
            ["PPO knobs", "init_std / std_range (exploration noise), entropy_coef 0.006, lr 1e-4 adaptive by KL 0.01"],
            ["Hulls vs mesh", "convex-hull collision pieces (cheap) vs raw triangle mesh (only the dish rack)"],
            ["Bank / bedrock", "teleop- or IK-authored start poses a fraction of episodes reset into (grasp bank); the shared start-state recipe"],
            ["Trossen WXAI; goal_time", "the arm; the host driver's target-smoothing horizon (0.1 s at 30 Hz)"]])


def appendixF(nb):
    nb.pagebreak()
    nb.title("Appendix F — MuJoCo solref / solimp, the whole thing")
    nb.p("Read from the code that runs in our arms: mujoco_warp/_src/constraint.py (the kbi block, installed 3.11.0) and MuJoCo's defaults.", italic=True)
    nb.h("F1. What a MuJoCo contact actually is", 1)
    nb.p("Not a spring on the body. Each contact is a row of a constraint system with a REFERENCE ACCELERATION for its coordinate r (penetration, r < 0 when overlapping, measured from the margin) and an IMPEDANCE imp ∈ (0, 1). The convex solver finds constraint forces so the constraint acceleration blends the reference with the unconstrained one:")
    nb.p("    a_ref = −k · imp(r) · r − b · v          D = 1 / (invweight · (1 − imp) / imp)", italic=True)
    nb.p("k and b are in 1/s² and 1/s — an acceleration per metre of violation. The force scales with the pair's effective mass (invweight): the same solref makes a heavy body stiffer in N/m (the 1 kg box on four corners sat ~20× stiffer than the 65 g sphere at δt = 1 ms with the same τ).")
    nb.h("F2. The two formats, exactly", 1)
    nb.p("solref = (solref[0], solref[1]); solimp = (dmin, dmax, width, mid, power). Defaults: solref (0.02, 1.0) — τ = 20 ms, ζ = 1 — which is what the TRAINING scenes use; solimp (0.9, 0.95, 0.001, 0.5, 2).", bold=True)
    nb.bl(["Reference format (both positive): τ = solref[0], ζ = solref[1]; k = 1/(dmax²·τ²·ζ²), b = 2/(dmax·τ). Halving τ quadruples k.",
           "refsafe clamp (on by default): τ ← max(τ, 2·timestep). At timestep 10 ms any τ < 20 ms becomes 20 ms — the whole 'soft at coarse δt' effect. At 1/90 s the floor is 22 ms: the training scenes' τ = 20 ms is clamped to 22 ms at every substep — i.e. in training MuJoCo's contact is always at its default softness, k = 1/(0.95²·0.022²) ≈ 2290 s⁻² — for a 65 g body ~150 N/m in spring terms, for the 0.4 kg mug ~900 N/m.",
           "Direct format (entries ≤ 0): k = −solref[0]/dmax², b = −solref[1]/dmax; no clamp; b = 0 allowed; behaves like an explicit spring (stable while ω·δt ≲ 2).",
           "Impedance imp(r): x = |r|/width; smooth step with midpoint mid and exponent power from dmin (x = 0) to dmax (x ≥ 1): 90 % enforcement at zero violation, 95 % beyond 1 mm. Why penetration vs stiffness is sublinear and why ζ = 0 in the reference format still dissipates (diverged in our probe).",
           "Margin: pos_out = r + margin — activation at the margin distance; benches use 0; training uses gap 3 mm on the Newton pipeline side."])
    nb.h("F3. What Newton's automatic conversion did", 1)
    nb.p("SolverMuJoCo's convert_solref(ke, kd, 1, 1): for kd = 0.02·k it produced (τ = 1 ms, ζ = 3.16), for kd = 0 (ball) (20 ms, 1.0). The clamp then turned 1 ms into 2δt and ζ = 3.16 cut stiffness 10× more: MuJoCo rested ~100× deeper than m g/k and the ball was critically damped. Not MuJoCo's physics — our configuration. Rule since 2026-08-28: every scene sets geom_solref explicitly (four_arms._apply_solref).")
    nb.h("F4. What each arm uses", 1)
    nb.tbl([["Scene", "solref", "Why", "Calibration"],
            ["Hard clutter (k = 10⁵)", "(2.4 ms, 1)", "reference = MuJoCo's default and safety rule; τ so a 65 g sphere sinks m g/k", "6.3 µm vs 6.4 at δt = 1 ms"],
            ["Soft clutter (k = 10³)", "(31.8 ms, 1)", "measured anchor", "ratio 1.00 at k = 10³"],
            ["Ball (k = 10³, zero dissipation)", "(−2.24·10³, 0) direct", "reference cannot take ζ = 0", "rest 1.01 mm vs 0.98; energy within 0.03 % at δt ≤ 1 ms"],
            ["Actuated push (k = 10⁵)", "(2.4 ms, 1)", "same as hard clutter", "box sits 0.04–0.05× m g/(4k) at 1 ms — mass scaling"],
            ["Training scenes (rig, mug, plate)", "(20 ms, 1) MuJoCo default, condim 6", "as authored in IsaacLab assets.py/env cfgs", "clamped to 22 ms at 1/90 s; ~150 N/m per 65 g — NEVER stiff; state this in the paper's Part-2 setup"]])
    nb.h("F5. Say / don't say", 1)
    nb.bl(["Say: MuJoCo's stiffness is a time constant per effective mass, clamped to the step; at the learner's step it cannot represent k ≳ 10³ N/m per 65 g body; position-only error control does not see the softening.",
           "Say: the direct format removes the clamp but is only stable for ω δt ≲ 2 — 'turn refsafe off' is not a fix.",
           "Don't say: 'MuJoCo dissipates every impact' (retracted) or 'MuJoCo cannot be stiff' (it can at δt ≤ τ/2).",
           "State the effective-mass scaling and the training scenes' default solref, one sentence each (the soft-clutter offset is gone: measured anchor)."])


def appendixG(nb):
    nb.pagebreak()
    nb.title("Appendix G — Speed mechanisms in force in the arms the paper uses")
    nb.h("Mechanisms in the MuJoCo and ICF error-controlled arms", 1)
    nb.tbl([["Mechanism", "Where", "What it does", "Evidence on file"],
            ["One captured CUDA graph per boundary (all four arms)", "four_arms._CapturedBoundary; NEWTON_MJ_ADAPTIVE_GRAPH=1", "the whole boundary's launch stream recorded once and replayed; one 4-byte host flag per iteration", "fairness verified (captured vs eager physics agree to 1e-8); no before/after wall number"],
            ["Conditional whole-march capture", "NEWTON_MJ_ADAPTIVE_CONDITIONAL=1 (opt-in); ICF: wp.capture_while when an outer capture is active", "zero host syncs per boundary", "ICF: default path; no wall number on file for either"],
            ["Tail compaction", "NEWTON_ADAPTIVE_TAIL_COMPACT=1 (MuJoCo); ICF masks finished worlds out of mid-march queries", "launch only over worlds still marching", "no before/after number on a paper scene"],
            ["Shared forward prefix", "NEWTON_MJ_ADAPTIVE_SHARED_FWD (on)", "the half steps reuse the full step's forward pass; both estimates judge the same contact set", "no number on file"],
            ["Two-query march", "both arms (parity 3d672820; ICF 99aeea0)", "one geometry query at x^n serves the full step and the first half; one at x^{n+1/2}", "CENIC's economy; the ICF cost split says narrow-phase is 84.5 % of the march"],
            ["Newton tolerance rule eq. 34", "four_arms (NEWTON_KAPPA 1e-3, floor 1e-8)", "solve tolerance tied to ε", "march step count < 10 %, wall < 5 % (B14)"],
            ["ICF solver performance pass", "icf_warp_adaptive (fp32 throughout; corrected cost-based convergence; exact line search under graph conditionals)", "—", "1.24× recorded in commit messages; no table"]])
    nb.p("What contribution 3 lacks today: a committed before/after measurement on a scene the paper shows. The single probe that closes it: hard clutter at 1024 worlds, each mechanism default vs OFF, wall per simulated second and ms per accepted substep, on an idle GPU.", bold=True)


def appendixH(nb):
    nb.pagebreak()
    nb.title("Appendix H — The literature review, theme by theme (PART1_LITERATURE.md)")
    nb.tbl([["Theme", "Question", "Key references", "Takeaways / what we adopted"],
            ["A — Benchmarking physics engines on contact", "What does the field measure and on what axes?", "Erez, Tassa & Todorov ICRA 2015; Kang & Hwangbo SimBenchmark 2018; Acosta, Yang & Posa RA-L 2022; Choi et al. PNAS 2021; Castro et al. SAP T-RO 2022; Le Lidec et al. T-RO 2024; Howell et al. Dojo; ComFree-Sim; GAUGE; Isaac Lab", "Erez's consistency violation (fine reference, short re-initialised pieces); speed–accuracy curves; Acosta: 'decreasing the timestep further did not improve prediction'; SAP's closed-form resting penetration. Adopted: self-consistency on clutter (B9), zero-g momentum (B11), contact counts, signed ΔE with gains marked unphysical (B4), matched-accuracy speed star (B6)."],
            ["B — Compliant/convex contact formulations", "What each model is and how its authors validated it; how to configure MuJoCo fairly", "Todorov 2014 + MuJoCo docs; SAP T-RO 2023; Castro, Han & Masterjohn ICF T-RO 2025 (2312.03908); Kurtz & Castro CENIC; Masterjohn hydroelastic RA-L 2022; Dojo; Anitescu 2006; Hunt & Crossley 1975", "MuJoCo stiffness ∝ 1/(τ²ζ²) per effective mass, refsafe, slip-by-design; ICF: the gliding offset μ(δt+τ_d)‖v_t‖ does not vanish. Adopted: calibrated per-scene solref stated with equations (B13); undamped direct format measured; stiffness sweep (B3)."],
            ["C — Error-controlled stepping and work-precision", "The reporting standard", "Hairer, Nørsett & Wanner; Gustafsson/Söderlind PI controllers; Bari IVP test set; Drake IntegratorBase; CENIC Fig. 10; Studer; Acary; Potra; Zapolsky & Drumwright; Riley 2025; DiffMJX", "Work-precision = measured error vs cost against a reference; requested-tolerance vs cost acceptable only among error-controlled methods; chaotic systems need short re-anchored windows. Adopted: consistency bench with per-window timing (B9); Drake's constants cited; per-world error control on GPU during RL stated as unprecedented."],
            ["D — GPU simulators, RL throughput, artifact exploitation", "How throughput is reported; evidence that policies exploit integrator artifacts; is per-world adaptive stepping novel", "Isaac Gym 2021; Rudin CoRL 2021; Brax; MJX/Playground/MJWarp; Isaac Lab; ManiSkill3; Genesis critique; Lehman 2018 + Cheney 2013; Krakovna 2018; Baker 2019; Factory RSS 2022; DeXtreme; Hwangbo 2019/Lee 2020; Embedded IPC; Tallec 2019", "Throughput vs N with the saturation knee, settings stated; Cheney 2013 creatures gamed an adaptive-δt heuristic; Factory: RL 'exploit[s] any inaccuracies'; 'No batched robot simulator steps worlds adaptively … No controlled experiment varies only the stepping scheme.' Adopted: wall-matched and accuracy-matched fixed arms in Part 2 (B17); 'the controller cannot be gamed' (hard floor)."],
            ["E — Stiff, actuated contact", "Test cases and metrics for PD-driven stiff contact", "CENIC Franka-with-box + dishrack; TAMSI; SAP; ICF grasp; Drake #14694; ManipulationStation; Acosta 2022; MuJoCo docs; Isaac Gym FrankaCabinet; Factory; Beltran-Hernandez 2020; NIST ATB; Yu 2016 / Bauza & Rodriguez 2017 pushing datasets", "'No source reports penetration, chatter, energy and tracking together across δt and ε for an actuated contact.' Adopted: the PD gantry push (B12). Caveat: MuJoCo's implicitfast makes PD damping implicit; ICF couples PD implicitly."]])
    nb.p("Not adopted, with reason: hydroelastic/pressure-field contact (no backend); the CENIC gripper/peg wedge (the 0.01 mm gap is dropped under point contact without a margin); a Franka articulation (the gantry carries the same ingredients without model-import variance).", italic=True)


def appendixI(nb):
    """Part-1 numbers generated from the CSVs at build time (newton-adaptive/scripts/bench/part1_keynumbers.py)."""
    import subprocess, os
    nb.pagebreak()
    nb.title("Appendix I — Part-1 numbers, generated from the CSVs at build time")
    nb.p("Source: newton-adaptive/scripts/bench/part1_keynumbers.py run when this notebook was built; these override any number quoted by hand in Part B or Part D if they disagree.", italic=True)
    repo = os.path.expanduser("~/Documents/code/newton-adaptive")
    py = os.path.expanduser("~/Documents/code/IsaacLabRubato/.venv/bin/python")
    try:
        out = subprocess.run([py, "scripts/bench/part1_keynumbers.py"], cwd=repo, capture_output=True, text=True, timeout=300).stdout
    except Exception as e:  # noqa: BLE001
        nb.p(f"(key numbers unavailable: {e})"); return
    table = []
    for line in out.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-"):
                continue
            table.append(cells); continue
        if table:
            nb.tbl(table); table = []
        if line.startswith("### "):
            nb.h(line[4:], 2)
        elif line.startswith("- "):
            nb.bl([line[2:]])
        elif line.strip():
            nb.p(line)
    if table:
        nb.tbl(table)
