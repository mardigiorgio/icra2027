#!/bin/bash
# Resumable Part-1 rerun on the 8-body scenes: each step is skipped when its
# output already exists (the 20-body files were moved to *_20 beforehand).
S=/tmp/claude-1002/-home-mdigiorgio-Documents-code/fef98df8-95da-4aa4-a47c-133d6ad86ec5/scratchpad
L=$HOME/Documents/code/icra2027/runlogs/campaign_2026-09-03
cd ~/Documents/code/icra2027 || exit 1
export VIRTUAL_ENV=$PWD/.venv PYTHONPATH=$PWD ICF_MARCH_COMPACT=1
R=part1/bench/results
step() { echo ">>> $1 $(date +%H:%M)"; }
if [ ! -f $R/.captures_8body ]; then step captures; timeout 900 .venv/bin/python part1/bench/part1_scene_captures.py 2>&1 | grep -aE "wrote|saved|Error|Traceback" | tail -3 && touch $R/.captures_8body; fi
for sc in hard-clutter soft-clutter; do [ -f $R/part1_workprecision_${sc}_n1.csv ] || { step "wp $sc n=1"; timeout 3000 .venv/bin/python -m part1.bench.benchmarks.part1_workprecision --scene $sc --n 1 --trials 3 2>&1 | grep -aE "wrote|budget|Traceback|Error" | tail -2; }; done
for sc in hard-clutter soft-clutter; do [ -f $R/part1_scaling_${sc}.csv ] || { step "scaling $sc"; timeout 3000 .venv/bin/python -m part1.bench.benchmarks.part1_scaling --scene $sc 2>&1 | grep -aE "wrote|Traceback|Error" | tail -2; }; done
grep -q ",3e-05," $R/part1_gpu_fair_ladder.csv || { step "GPU ladder 3e-5"; timeout 3000 .venv/bin/python -m part1.bench.benchmarks.part1_gpu_fair --scene hard-clutter --tol 3e-5 --ns 16384 8192 4096 2048 1024 512 256 128 64 32 16 8 4 2 1 2>&1 | grep -aoE "appended.*|Traceback|Error.*"; }
[ -f $R/part1_stiffness_sweep.csv ] || { step "stiffness sweep"; timeout 3000 .venv/bin/python -m part1.bench.benchmarks.part1_stiffness_sweep 2>&1 | grep -aE "wrote|Error|Traceback" | tail -2; }
for sc in hard-clutter soft-clutter; do [ -f $R/part1_workprecision_${sc}_n1024.csv ] || { step "wp $sc n=1024"; timeout 5400 .venv/bin/python -m part1.bench.benchmarks.part1_workprecision --scene $sc --n 1024 --trials 3 2>&1 | grep -aE "wrote|budget|Traceback|Error" | tail -2; }; done
step regenerate; .venv/bin/python -m part1.bench.part1_plots 2>&1 | grep -E "wrote|Error|Traceback"; .venv/bin/python part1/report/build_p1_design_doc.py 2>&1 | tail -1; bash part1/report/deliver_p1_doc.sh 2>&1 | tail -1
step "P1 8-BODY DONE"
cd ~/Documents/code/IsaacLab && CAMPAIGN_LOG_DIR=$L CAMPAIGN_WAVE=hull16-v1 setsid nohup bash scripts/experiments/trossen_campaign.sh >> $L/launcher.log 2>&1 < /dev/null & disown
sleep 30; echo ">>> campaign relaunched $(date +%H:%M): $(tail -1 $L/campaign.log)"
