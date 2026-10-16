#!/bin/bash
# Builds the policy analysis harness.
# Usage: build.sh [src_root] [out]
#   src_root: checkout to take the env (ocean/impulse_wars) and src/puffercpu.c
#             from, e.g. a git worktree of a balance change; defaults to this
#             repo. The harness itself, vendored code and the box2d and raylib
#             libraries always come from this repo, as they're gitignored and
#             missing from worktrees.
#   out: binary path, defaults to build/iw_analyze in this repo
set -e
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
SRC="$(cd "${1:-$REPO}" && pwd)"
OUT="${2:-$REPO/build/iw_analyze}"
mkdir -p "$(dirname "$OUT")"
${CC:-clang} -O3 -march=native -fno-math-errno -fno-trapping-math \
    -fassociative-math -fno-signed-zeros -freciprocal-math -DPUF_CPU_BLOCKED_KERNELS \
    -I"$SRC/ocean/impulse_wars" -I"$SRC/src" -I"$REPO/vendor" -I"$REPO/vendor/collections-c" \
    -I"$REPO/raylib-5.5_linux_amd64/include" -I"$REPO/box2d-linux-amd64/include" -I"$REPO/box2d-linux-amd64/src" \
    "$REPO/ocean/impulse_wars/analysis/iw_analyze.c" -o "$OUT" \
    "$REPO/box2d-linux-amd64/libbox2d.a" "$REPO/raylib-5.5_linux_amd64/lib/libraylib.a" \
    -lm -lpthread -ldl -DPLATFORM_DESKTOP -DPUFFER_IMPULSE_WARS \
    -Wno-gnu-folding-constant -Wno-incompatible-pointer-types-discards-qualifiers
echo "built $OUT from $SRC"
