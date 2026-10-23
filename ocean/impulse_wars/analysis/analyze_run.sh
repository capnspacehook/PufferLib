#!/bin/bash
# Runs the standard analysis of a training run and prints a summary.
# Usage: analyze_run.sh <run name> [options]
#   --ckpt PATH         checkpoint to analyze, default the run's latest
#   --baseline X        a run name or checkpoint to compare against: plays it
#                       head to head and gets its self-play and bot results on
#                       this build (out of distribution if the env changed)
#   --baseline-size H L policy size of a baseline with no ini
#   --src DIR           checkout to build the env from, default this repo; use a
#                       worktree at the run's commit if the tree has moved on
#   --out DIR           results directory, default build/iw_analysis/<run>
#   --traj N            checkpoints to play in self-play for trajectory.py,
#                       default 10, 0 to skip
#   --no-bots, --no-combos   skip the bot ladder or the crate tournament
#   --threads T         default every core
#   Results already in the directory are kept, so an interrupted analysis
#   resumes and a finished one only prints the summary again. Delete a file
#   to redo it. The summary is also written to summary.txt.
set -e
A="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$A/../../.." && pwd)"
if [ $# -lt 1 ] || [ "${1#-}" != "$1" ]; then
    sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
RUN="$1"
shift
CKPT=""
BASE=""
BASE_SIZE=()
SRC="$REPO"
OUT="$REPO/build/iw_analysis/$RUN"
TRAJ=10
BOTS=1
COMBOS=1
THREADS="$(nproc)"
while [ $# -gt 0 ]; do
    case "$1" in
    --ckpt) CKPT="$2"; shift 2 ;;
    --baseline) BASE="$2"; shift 2 ;;
    --baseline-size) BASE_SIZE=(--opponent-hidden "$2" --opponent-layers "$3"); shift 3 ;;
    --src) SRC="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --traj) TRAJ="$2"; shift 2 ;;
    --no-bots) BOTS=0; shift ;;
    --no-combos) COMBOS=0; shift ;;
    --threads) THREADS="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
    esac
done

CKPT_DIR="$REPO/checkpoints/impulse_wars/$RUN"
if [ -z "$CKPT" ]; then
    CKPT="$(ls "$CKPT_DIR"/*.bin 2>/dev/null | sort | tail -1)"
    [ -n "$CKPT" ] || { echo "no checkpoints in $CKPT_DIR" >&2; exit 1; }
fi
mkdir -p "$OUT/traj"

# the harness is built per run so analyses of different env versions don't
# overwrite each other's binary
export IW_ANALYZE="$OUT/iw_analyze"
"$A/build.sh" "$SRC" "$IW_ANALYZE" >&2

# runs one analysis unless its results already exist
step() {
    local out="$1"
    shift
    if [ -s "$out" ]; then
        echo "have $(basename "$out")" >&2
        return
    fi
    echo "running $(basename "$out")" >&2
    local t
    t=$(date +%s)
    "$A/run.sh" "$@" --threads "$THREADS" 2>"$out.log" || { echo "failed, see $out.log" >&2; rm -f "$out"; exit 1; }
    echo "  done in $(( $(date +%s) - t ))s" >&2
}

# checkpoint step count from its file name
steps() { basename "$1" .bin | sed 's/^0*//; s/^$/0/'; }

# the checkpoint of the run closest to a step count
closest() {
    ls "$CKPT_DIR"/*.bin | awk -v t="$1" '{n = $0; sub(/.*\//, "", n); sub(/\.bin$/, "", n); d = n - t; if (d < 0) d = -d; print d, $0}' |
        sort -n | head -1 | cut -d' ' -f2
}

FINAL=$(steps "$CKPT")
step "$OUT/self.jsonl" "$CKPT" "$OUT/self.jsonl" --opponent "$CKPT" --opponent-name self --weights-out "$OUT/weights.tsv"
[ $BOTS = 1 ] && step "$OUT/bots.jsonl" "$CKPT" "$OUT/bots.jsonl"

# head to head against the run's own checkpoints at a third and two thirds
# of the way, which shows whether it was still improving
H2H=()
for frac in 3 1.5; do
    other=$(closest "$(awk -v f="$FINAL" -v d="$frac" 'BEGIN {printf "%d", f / d}')")
    [ "$other" = "$CKPT" ] && continue
    name="$(awk -v s="$(steps "$other")" 'BEGIN {printf "%.2gB", s / 1e9}')"
    step "$OUT/h2h_$name.jsonl" "$CKPT" "$OUT/h2h_$name.jsonl" --opponent "$other" --opponent-name "$name"
    H2H+=("$OUT/h2h_$name.jsonl")
done

if [ -n "$BASE" ]; then
    step "$OUT/h2h_baseline.jsonl" "$CKPT" "$OUT/h2h_baseline.jsonl" --opponent "$BASE" --opponent-name baseline "${BASE_SIZE[@]}"
    H2H+=("$OUT/h2h_baseline.jsonl")
    # the baseline as the policy needs its size as --hidden/--layers
    SIZE=()
    if [ ${#BASE_SIZE[@]} -gt 0 ]; then
        SIZE=(--hidden "${BASE_SIZE[1]}" --layers "${BASE_SIZE[3]}" "${BASE_SIZE[@]}")
    fi
    step "$OUT/baseline_self.jsonl" "$BASE" "$OUT/baseline_self.jsonl" --opponent "$BASE" --opponent-name self "${SIZE[@]}"
    if [ $BOTS = 1 ]; then
        BSIZE=()
        [ ${#BASE_SIZE[@]} -gt 0 ] && BSIZE=(--hidden "${BASE_SIZE[1]}" --layers "${BASE_SIZE[3]}")
        step "$OUT/baseline_bots.jsonl" "$BASE" "$OUT/baseline_bots.jsonl" "${BSIZE[@]}"
    fi
fi

[ $COMBOS = 1 ] && step "$OUT/combos.jsonl" "$CKPT" "$OUT/combos.jsonl" --opponent "$CKPT" --opponent-name self --combos --crates --episodes 16

# self-play over checkpoints spaced evenly on a log scale from 1/64 of the
# run to the end, as behavior changes fastest early on
if [ "$TRAJ" -gt 0 ]; then
    ln -sf ../self.jsonl "$OUT/traj/$FINAL.jsonl"
    for t in $(awk -v f="$FINAL" -v n="$TRAJ" 'BEGIN {for (i = 0; i < n - 1; i++) printf "%d\n", f / 64 * exp(log(64) * i / (n - 1))}'); do
        c=$(closest "$t")
        s=$(steps "$c")
        [ "$s" = "$FINAL" ] && continue
        step "$OUT/traj/$s.jsonl" "$c" "$OUT/traj/$s.jsonl" --opponent "$c" --opponent-name self
    done
fi

# summary
{
    echo "run $RUN, checkpoint $CKPT"
    echo "results in $OUT"
    C="python3 $A/compare.py"
    SELF=(run="$OUT/self.jsonl")
    [ -n "$BASE" ] && SELF=(baseline="$OUT/baseline_self.jsonl" run="$OUT/self.jsonl")
    for h in "${H2H[@]}"; do
        echo "head to head, $(basename "$h" .jsonl | sed 's/^h2h_//'): $($C "$h" --sections opp | grep -m1 '^  all' | awk '{print $2}')"
    done
    if [ $BOTS = 1 ]; then
        BOTS_RUNS=(run="$OUT/bots.jsonl")
        [ -n "$BASE" ] && BOTS_RUNS=(baseline="$OUT/baseline_bots.jsonl" run="$OUT/bots.jsonl")
        $C "${BOTS_RUNS[@]}" --sections opp
    fi
    $C "${SELF[@]}" --sections pace,weapons,slots,deaths,behavior,standoff,mapweapons
    [ $COMBOS = 1 ] && $C "$OUT/combos.jsonl" --sections combos,pace
    [ "$TRAJ" -gt 0 ] && python3 "$A/trajectory.py" "$OUT/traj"
    python3 "$A/weights.py" "$OUT/weights.tsv" | head -40
} 2>&1 | tee "$OUT/summary.txt"
