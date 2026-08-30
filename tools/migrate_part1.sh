#!/usr/bin/env bash
# Move the Part 1 pure-solver experiments (scenes, benches, results, figures,
# tables) from newton-adaptive/scripts/{bench,scenes} into this repo's part1/
# package, rewriting the module paths scripts.bench -> part1.bench and
# scripts.scenes -> part1.scenes. dish_rack.py and scripts/assets stay in
# newton-adaptive (Part 2 asset scene, not a Part 1 dependency).
#
# Run ONLY when no bench is running (the chain's subprocesses import
# scripts.bench from newton-adaptive). Idempotent on the copy; the removal
# from newton-adaptive is a separate git commit on its current branch.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NEWTON="${NEWTON_DIR:-$(dirname "$ROOT")/newton-adaptive}"
[ -d "$NEWTON/scripts/bench" ] || { echo "nothing to migrate: $NEWTON/scripts/bench missing" >&2; exit 1; }
pgrep -f "scripts.bench" >/dev/null && { echo "a bench is running; refusing" >&2; exit 1; }
mkdir -p "$ROOT/part1"
rsync -a --exclude __pycache__ "$NEWTON/scripts/bench/" "$ROOT/part1/bench/"
rsync -a --exclude __pycache__ --exclude dish_rack.py "$NEWTON/scripts/scenes/" "$ROOT/part1/scenes/"
touch "$ROOT/part1/__init__.py"
grep -rIl "scripts\.bench\|scripts\.scenes\|scripts/bench\|scripts/scenes" "$ROOT/part1" | xargs -r sed -i 's/scripts\.bench/part1.bench/g; s/scripts\.scenes/part1.scenes/g; s#scripts/bench#part1/bench#g; s#scripts/scenes#part1/scenes#g'
mkdir -p "$ROOT/paper/figures" && cp "$ROOT"/part1/bench/results/figures/*.pdf "$ROOT/paper/figures/" 2>/dev/null || true
cd "$ROOT" && .venv/bin/python -c "import part1.bench.four_arms, part1.bench.part1_plots, part1.bench.part1_story, part1.scenes.cenic_scenes; print('part1 imports ok')"
git -C "$ROOT" add -A part1 paper/figures
echo "copied; now remove the originals: git -C $NEWTON rm -r -q scripts/bench scripts/scenes/actuated_press.py scripts/scenes/cenic_scenes.py scripts/scenes/contact_objects.py && commit both repos"
