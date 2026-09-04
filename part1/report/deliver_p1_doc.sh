#!/bin/bash
# Deliver the rebuilt Part 1 design doc to Drive/ICRA 2027/Part 1 (overwrite
# in place, same name) and verify the checksum on both sides.
set -e
DOC=$HOME/Documents/code/icra2027/part1/bench/results/P1_core_experiments.docx
PART1=1MFiv-q9i06wNxuaAkuAtXvGkXeQBc1my
RC=$HOME/.local/bin/rclone
$RC copyto "$DOC" gdrive:P1_core_experiments.docx --drive-root-folder-id $PART1
LOCAL=$(md5sum "$DOC" | cut -d' ' -f1)
REMOTE=$($RC md5sum gdrive:P1_core_experiments.docx --drive-root-folder-id $PART1 | cut -d' ' -f1)
echo "local  $LOCAL"; echo "remote $REMOTE"
[ "$LOCAL" = "$REMOTE" ] && echo "DELIVERED (md5 match)" || { echo "MISMATCH"; exit 1; }
