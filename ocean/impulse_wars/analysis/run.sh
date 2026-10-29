#!/bin/bash
# Plays a policy against the scripted bots or another policy and writes one
# JSON line per game.
# Usage: run.sh <run name | checkpoint.bin> <out.jsonl> [--opponent <run name | checkpoint.bin>] [iw_analyze args...]
#   A run name uses the run's latest checkpoint in checkpoints/impulse_wars.
#   Policy sizes are read from the run's ini in logs/impulse_wars, found by
#   the run name or the checkpoint's directory, either as the file name or
#   its run_id; pass --hidden/--layers (--opponent-hidden/--opponent-layers)
#   to override or when there's no ini.
#   Defaults: every bot personality, 1400 games each (100 per map). With
#   --opponent, 1400 games against that policy, each seed played from both
#   drones; pass the same run twice for self-play.
#   Set IW_ANALYZE to use a binary other than build/iw_analyze (or
#   build/iw_analyze_gpu with --gpu), e.g. one built from a balance change's
#   worktree.
set -e
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
DEFAULT_BIN="$REPO/build/iw_analyze"
[[ " $* " == *" --gpu "* ]] && DEFAULT_BIN="$REPO/build/iw_analyze_gpu"
BIN="${IW_ANALYZE:-$DEFAULT_BIN}"
if [ $# -lt 2 ]; then
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

# sets CKPT and RUN from a run name or checkpoint path
resolve() {
    if [ -f "$1" ]; then
        CKPT="$1"
        RUN="$(basename "$(dirname "$CKPT")")"
    else
        RUN="$1"
        CKPT="$(ls "$REPO/checkpoints/impulse_wars/$RUN"/*.bin 2>/dev/null | sort | tail -1)"
        if [ -z "$CKPT" ]; then
            echo "no checkpoints for run $RUN" >&2
            exit 1
        fi
    fi
}

# prints "hidden layers" of a run from its ini
arch() {
    local ini="$REPO/logs/impulse_wars/$1.ini"
    if [ ! -f "$ini" ]; then
        ini="$(grep -lx "run_id = $1" "$REPO"/logs/impulse_wars/*.ini 2>/dev/null | head -1)"
    fi
    if [ ! -f "$ini" ]; then
        echo "no ini for run $1 to read the policy size from, pass its size explicitly" >&2
        exit 1
    fi
    awk -F' *= *' '$0 == "[policy]" {in_sec = 1; next} /^\[/ {in_sec = 0}
        in_sec && $1 == "hidden_size" {h = $2} in_sec && $1 == "num_layers" {l = $2}
        END {print h, l}' "$ini"
}

resolve "$1"
OUT="$2"
shift 2
POLICY_CKPT="$CKPT"
POLICY_RUN="$RUN"

# pull out --opponent so it can be resolved like the policy
ARGS=()
OPP_TARGET=""
while [ $# -gt 0 ]; do
    if [ "$1" = "--opponent" ] && [ $# -gt 1 ]; then
        OPP_TARGET="$2"
        shift 2
    else
        ARGS+=("$1")
        shift
    fi
done

EXTRA=()
if [[ " ${ARGS[*]} " != *" --hidden "* ]]; then
    read -r H L <<< "$(arch "$POLICY_RUN")"
    [ -n "$H" ] || exit 1
    EXTRA+=(--hidden "$H" --layers "$L")
fi
echo "policy: $POLICY_CKPT" >&2
if [ -n "$OPP_TARGET" ]; then
    resolve "$OPP_TARGET"
    EXTRA+=(--opponent "$CKPT")
    if [[ " ${ARGS[*]} " != *" --opponent-hidden "* ]]; then
        read -r H L <<< "$(arch "$RUN")"
        [ -n "$H" ] || exit 1
        EXTRA+=(--opponent-hidden "$H" --opponent-layers "$L")
    fi
    if [[ " ${ARGS[*]} " != *" --opponent-name "* ]]; then
        EXTRA+=(--opponent-name "$RUN")
    fi
    echo "opponent: $CKPT" >&2
fi

"$BIN" --policy "$POLICY_CKPT" "${EXTRA[@]}" --out "$OUT" "${ARGS[@]}"
