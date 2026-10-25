#!/bin/bash
# Plays every pair of a set of checkpoints head to head and prints the
# transitivity analysis (transitivity.py).
# Usage: round_robin.sh <out dir> <name=checkpoint.bin | checkpoint.bin>... [--episodes N] [--threads T] [--src DIR]
#   A checkpoint without a name is named by its run and step count, e.g.
#   accelerator-rework@2.0B. Sizes come from each run's ini as in run.sh.
#   Pairs already in the out dir are kept, so an interrupted round robin
#   resumes, and adding checkpoints only plays the new pairs.
#   --src builds the harness from another checkout (default this repo).
set -e
A="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$A/../../.." && pwd)"
if [ $# -lt 3 ]; then
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
OUT="$1"
shift
NAMES=()
CKPTS=()
EPISODES=1400
THREADS="$(nproc)"
SRC="$REPO"
while [ $# -gt 0 ]; do
    case "$1" in
    --episodes) EPISODES="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --src) SRC="$2"; shift 2 ;;
    *=*) NAMES+=("${1%%=*}"); CKPTS+=("${1#*=}"); shift ;;
    *)
        run="$(basename "$(dirname "$1")")"
        NAMES+=("$run@$(basename "$1" .bin | awk '{printf "%.2gB", $0 / 1e9}')")
        CKPTS+=("$1")
        shift ;;
    esac
done
for name in "${NAMES[@]}"; do
    case "$name" in *_vs_*|*/*) echo "checkpoint names can't contain _vs_ or /: $name" >&2; exit 1 ;; esac
done
for ckpt in "${CKPTS[@]}"; do
    [ -f "$ckpt" ] || { echo "no checkpoint $ckpt" >&2; exit 1; }
done
mkdir -p "$OUT"
export IW_ANALYZE="$OUT/iw_analyze"
[ -x "$IW_ANALYZE" ] || "$A/build.sh" "$SRC" "$IW_ANALYZE" >&2

n=${#CKPTS[@]}
for ((i = 0; i < n; i++)); do
    for ((j = i + 1; j < n; j++)); do
        a="${NAMES[$i]}"
        b="${NAMES[$j]}"
        f="$OUT/${a}_vs_${b}.jsonl"
        [ -s "$f" ] || [ -s "$OUT/${b}_vs_${a}.jsonl" ] && continue
        t=$(date +%s)
        "$A/run.sh" "${CKPTS[$i]}" "$f.tmp" --opponent "${CKPTS[$j]}" --opponent-name "$b" \
            --episodes "$EPISODES" --threads "$THREADS" 2>"$OUT/last.log" || { echo "failed, see $OUT/last.log" >&2; exit 1; }
        mv "$f.tmp" "$f"
        echo "$a vs $b: $(( $(date +%s) - t ))s" >&2
    done
done
python3 "$A/transitivity.py" "$OUT" | tee "$OUT/summary.txt"
