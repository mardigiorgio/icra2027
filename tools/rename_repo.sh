#!/usr/bin/env bash
# Rename this checkout IsaacLabRubato -> icra2027 in place: the directory, the
# venv's embedded absolute paths (console-script shebangs, activate scripts,
# sitecustomize), the git remote, the GitHub repo, and every path reference in
# the sibling repos and the CLAUDE.md/memory files. --dry-run prints the plan.
#
# Run ONLY when nothing is executing from IsaacLabRubato/.venv (bench chains
# launch subprocesses by the absolute interpreter path).
set -euo pipefail
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
CODE="$HOME/Documents/code"; OLD="$CODE/IsaacLabRubato"; NEW="$CODE/icra2027"
run() { if [ $DRY = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
[ -d "$OLD" ] || { echo "$OLD missing (already renamed?)" >&2; exit 1; }
[ -e "$NEW" ] && { echo "$NEW exists" >&2; exit 1; }
if pgrep -f "IsaacLabRubato/.venv" >/dev/null; then echo "processes still run from the old venv path:" >&2; pgrep -af "IsaacLabRubato/.venv" | cut -c1-120 >&2; [ $DRY = 1 ] || exit 1; fi
run mv "$OLD" "$NEW"
# venv: only text files under bin/, the activate scripts and sitecustomize embed the path
run "grep -Il 'IsaacLabRubato' $NEW/.venv/bin/* $NEW/.venv/lib/python3.12/site-packages/sitecustomize.py 2>/dev/null | xargs -r sed -i 's#/IsaacLabRubato/#/icra2027/#g'"
run git -C "$NEW" remote set-url origin https://github.com/mardigiorgio/icra2027.git
run gh repo rename icra2027 -R mardigiorgio/IsaacLabRubato --yes
# sibling repos and the session files
FILES=$(grep -rIl "IsaacLabRubato" "$CODE/CLAUDE.md" "$HOME/.claude/projects/-home-mdigiorgio-Documents-code/memory" "$CODE/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib" "$CODE/IsaacLab/scripts" "$CODE/newton-adaptive/scripts" "$NEW/paper/tools" "$NEW/part2" 2>/dev/null | grep -v "/wandb/\|/logs/\|\.diff$" || true)
for f in $FILES; do run sed -i "'s#IsaacLabRubato#icra2027#g'" "$f"; done
run "[ -d $CODE/cenic-paper ] && mv $CODE/cenic-paper $CODE/cenic-paper-archive/cenic-paper-pre-icra2027 && ln -s icra2027/paper $CODE/cenic-paper"
run "$NEW/.venv/bin/python -c 'import newton, isaaclab; print(\"venv ok\")' && $NEW/.venv/bin/pre-commit --version"
echo "done; remaining references:"; grep -rIl "IsaacLabRubato" "$CODE/CLAUDE.md" "$CODE/IsaacLab/scripts" "$CODE/newton-adaptive/scripts" "$NEW" --exclude-dir=.venv --exclude-dir=archive --exclude-dir=.git 2>/dev/null | head || true
