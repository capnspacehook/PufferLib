#!/bin/bash
# Builds the policy analysis harness.
# Usage: build.sh [src_root] [out] [--gpu] [--fp32]
#   src_root: checkout to take the env (ocean/impulse_wars) and src/puffercpu.c
#             from, e.g. a git worktree of a balance change; defaults to this
#             repo. The harness itself, vendored code and the box2d and raylib
#             libraries always come from this repo, as they're gitignored and
#             missing from worktrees.
#   out: binary path, defaults to build/iw_analyze in this repo, or
#        build/iw_analyze_gpu (build/iw_analyze_gpu_fp32) with --gpu
#   --gpu: also build GPU inference (iw_gpu.cu, the trainer's CUDA net, from
#          src_root too), used when iw_analyze gets --gpu; needs nvcc. The
#          forward is bf16 like training, or fp32 with --fp32; bf16 was 1.5x
#          faster with bot scores within noise of fp32 and the CPU. Set
#          NVCC_ARCH to build for another GPU than this machine's.
set -e
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
POS=()
GPU=0
FP32=0
for arg in "$@"; do
    case "$arg" in
    --gpu) GPU=1 ;;
    --fp32) FP32=1 ;;
    -*) echo "unknown option $arg" >&2; exit 1 ;;
    *) POS+=("$arg") ;;
    esac
done
if [ $FP32 = 1 ] && [ $GPU = 0 ]; then
    echo "--fp32 needs --gpu" >&2
    exit 1
fi
SRC="$(cd "${POS[0]:-$REPO}" && pwd)"
DEFAULT_OUT="$REPO/build/iw_analyze"
if [ $GPU = 1 ]; then
    DEFAULT_OUT="$REPO/build/iw_analyze_gpu"
    [ $FP32 = 1 ] && DEFAULT_OUT="${DEFAULT_OUT}_fp32"
fi
OUT="${POS[1]:-$DEFAULT_OUT}"
mkdir -p "$(dirname "$OUT")"
A="$REPO/ocean/impulse_wars/analysis"
INCLUDES=(
    -I"$SRC/ocean/impulse_wars" -I"$SRC/src" -I"$REPO/vendor" -I"$REPO/vendor/collections-c"
    -I"$REPO/raylib-5.5_linux_amd64/include" -I"$REPO/box2d-linux-amd64/include" -I"$REPO/box2d-linux-amd64/src"
)
LIBS=("$REPO/box2d-linux-amd64/libbox2d.a" "$REPO/raylib-5.5_linux_amd64/lib/libraylib.a" -lm -lpthread -ldl)
CFLAGS=(
    -O3 -march=native -fno-math-errno -fno-trapping-math
    -fassociative-math -fno-signed-zeros -freciprocal-math
    "${INCLUDES[@]}" -DPLATFORM_DESKTOP -DPUFFER_IMPULSE_WARS
    -Wno-gnu-folding-constant -Wno-incompatible-pointer-types-discards-qualifiers
)

if [ $GPU = 0 ]; then
    ${CC:-clang} "${CFLAGS[@]}" "$A/iw_analyze.c" -o "$OUT" "${LIBS[@]}"
    echo "built $OUT from $SRC"
    exit 0
fi

CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-$(dirname "$(dirname "$(which nvcc 2>/dev/null || echo .)")")}}
NVCC="$CUDA_HOME/bin/nvcc"
if [ ! -x "$NVCC" ]; then
    echo "--gpu needs nvcc: put it on PATH or set CUDA_HOME" >&2
    exit 1
fi
PRECISION=()
[ $FP32 = 1 ] && PRECISION=(-DPRECISION_FLOAT)
OBJ="$(mktemp -d)"
trap 'rm -rf "$OBJ"' EXIT
# the trainer's tensor substrate and nets: src/pufferl.cu up to its
# #include "algo.cu", without the multi-GPU and OpenMP headers
PUFFERL="$SRC/src/pufferl.cu"
grep -q '^#include "algo.cu"' "$PUFFERL" || { echo "no #include \"algo.cu\" in $PUFFERL" >&2; exit 1; }
sed -n '1,/^#include "algo.cu"/p' "$PUFFERL" | grep -v '^#include <\(nccl\|nvml\|omp\)\.h>' > "$OBJ/pufferl_algo.cu"
# as the trainer build (build.sh): narrowing brace init and unreachable code
# in the env headers
"$NVCC" -O3 --threads 0 -arch="${NVCC_ARCH:-native}" -std=c++17 "${PRECISION[@]}" \
    -I"$OBJ" "${INCLUDES[@]}" -I"$A" -I"$CUDA_HOME/include" \
    -DPUFFER_IMPULSE_WARS -DENV_HEADER='"impulse_wars.h"' -DPUFFER_ENV_NAME='"impulse_wars"' \
    -Xcompiler=-DPLATFORM_DESKTOP -Xcompiler=-Wno-narrowing -Xcompiler=-Wno-unreachable-code \
    --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128 \
    -c "$A/iw_gpu.cu" -o "$OBJ/iw_gpu.o"
${CC:-clang} "${CFLAGS[@]}" -DIW_GPU -I"$A" -c "$A/iw_analyze.c" -o "$OBJ/iw_analyze.o"
"$NVCC" -arch="${NVCC_ARCH:-native}" "$OBJ/iw_analyze.o" "$OBJ/iw_gpu.o" -o "$OUT" \
    "${LIBS[@]}" -L"$CUDA_HOME/lib64" -lcublas -lcurand -Xlinker -rpath -Xlinker "$CUDA_HOME/lib64"
echo "built $OUT from $SRC, GPU inference $([ $FP32 = 1 ] && echo fp32 || echo bf16)"
