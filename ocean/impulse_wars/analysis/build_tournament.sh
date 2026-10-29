#!/bin/bash
# builds the scripted bot tournament -> ./iw_tournament
set -e
cd "$(dirname "$0")/../.."
${CC:-clang} -O3 -mavx2 -mfma -flto -fno-math-errno -fno-trapping-math \
    -I. -Isrc -Ivendor -I./raylib-5.5_linux_amd64/include \
    -I./box2d-linux-amd64/include -I./box2d-linux-amd64/src -I./ocean/impulse_wars -I./vendor/collections-c \
    ocean/impulse_wars/bot_tournament.c -o "${1:-iw_tournament}" \
    ./box2d-linux-amd64/libbox2d.a ./raylib-5.5_linux_amd64/lib/libraylib.a \
    -lm -lpthread -ldl -DPLATFORM_DESKTOP -DPUFFER_IMPULSE_WARS -Wno-gnu-folding-constant -Wno-incompatible-pointer-types-discards-qualifiers "${@:2}"
