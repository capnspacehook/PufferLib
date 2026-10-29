#!/bin/bash
# Checks GPU inference against the CPU on one checkpoint, to choose between
# fp32 and bf16: builds the CPU, GPU fp32 and GPU bf16 harnesses, plays the
# bot ladder with each, and prints their times, then their scores and
# behavior side by side (compare.py, CPU first). GPU games can't match CPU
# games exactly, so fp32 and bf16 should agree with the CPU within the
# confidence intervals. Each GPU build also replays some games with other
# --threads, which must give identical games.
# Usage: gpu_check.sh <run name | checkpoint.bin> [options]
#   --episodes N   per bot, default 1400
#   --threads T    default every core
#   --batch B      games per thread on the GPU, default iw_analyze's (64)
#   --out DIR      default build/iw_gpu_check
#   Policy sizes come from the run's ini as with run.sh; pass --hidden H
#   --layers L when there's none.
set -e
A="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$A/../../.." && pwd)"
if [ $# -lt 1 ] || [ "${1#-}" != "$1" ]; then
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
TARGET="$1"
shift
EPISODES=1400
THREADS="$(nproc)"
BATCH=()
SIZE=()
OUT="$REPO/build/iw_gpu_check"
while [ $# -gt 0 ]; do
    case "$1" in
    --episodes) EPISODES="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --batch) BATCH=(--batch "$2"); shift 2 ;;
    --hidden) SIZE+=(--hidden "$2"); shift 2 ;;
    --layers) SIZE+=(--layers "$2"); shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
    esac
done
mkdir -p "$OUT"

"$A/build.sh" "$REPO" "$OUT/iw_analyze_cpu" >&2
"$A/build.sh" "$REPO" "$OUT/iw_analyze_fp32" --gpu --fp32 >&2
"$A/build.sh" "$REPO" "$OUT/iw_analyze_bf16" --gpu >&2

# plays the bot ladder: name, binary, iw_analyze args
play() {
    local name="$1" bin="$2"
    shift 2
    IW_ANALYZE="$bin" "$A/run.sh" "$TARGET" "$OUT/$name.jsonl" "${SIZE[@]}" "$@" 2>"$OUT/$name.log" ||
        { echo "$name failed, see $OUT/$name.log" >&2; exit 1; }
    printf '%-12s %s\n' "$name" "$(grep -o '[0-9]* games in [0-9.]*s' "$OUT/$name.log" | tail -1)"
}

echo "time, $THREADS threads:"
play cpu "$OUT/iw_analyze_cpu" --episodes "$EPISODES" --threads "$THREADS"
for p in fp32 bf16; do
    play "$p" "$OUT/iw_analyze_$p" --episodes "$EPISODES" --threads "$THREADS" --gpu "${BATCH[@]}"
done
REPLAY_THREADS=$(( THREADS > 1 ? THREADS / 2 : 2 ))
REPLAY_EPISODES=$(( EPISODES < 140 ? EPISODES : 140 ))
for p in fp32 bf16; do
    play "${p}_replay" "$OUT/iw_analyze_$p" --episodes "$REPLAY_EPISODES" --threads "$REPLAY_THREADS" --gpu "${BATCH[@]}"
done

echo
echo "replays with $REPLAY_THREADS threads:"
for p in fp32 bf16; do
    # threatsNew is left out: the harness tracks threats by projectile
    # address, which allocation timing changes even on the CPU
    python3 - "$OUT/$p.jsonl" "$OUT/${p}_replay.jsonl" "$p" <<'EOF'
import json, sys

def load(path):
    games = {}
    for line in open(path):
        g = json.loads(line)
        for side in "pb":
            g[side].pop("threatsNew", None)
        games[(g["opp"], g["map"], g["seed"], g["seat"])] = g
    return games

full, replay = load(sys.argv[1]), load(sys.argv[2])
differ = [k for k, g in replay.items() if full.get(k) != g]
print(f"  {sys.argv[3]}: {len(replay) - len(differ)} / {len(replay)} games identical")
EOF
done

echo
C="python3 $A/compare.py"
$C cpu="$OUT/cpu.jsonl" fp32="$OUT/fp32.jsonl" bf16="$OUT/bf16.jsonl" --sections opp,pace,weapons,deaths
