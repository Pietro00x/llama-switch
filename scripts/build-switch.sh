#!/usr/bin/env bash
#
# Build the Nintendo Switch homebrew app (llama-switch.nro) with devkitPro (devkitA64 + libnx + switch-tools).
#
# usage: scripts/build-switch.sh [extra cmake -D options...]
#
# env:
#   DEVKITPRO  devkitPro install prefix (default: /opt/devkitpro)
#   BUILD_DIR  build directory          (default: <repo>/build-switch)
#   JOBS       parallel jobs            (default: nproc)
#
set -euo pipefail

export DEVKITPRO="${DEVKITPRO:-/opt/devkitpro}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-switch}"
JOBS="${JOBS:-$(nproc)}"

SWITCH_CMAKE="$DEVKITPRO/portlibs/switch/bin/aarch64-none-elf-cmake"
if [ ! -x "$SWITCH_CMAKE" ]; then
    echo "error: $SWITCH_CMAKE not found - install devkitPro with the switch-dev group" >&2
    exit 1
fi

# the generator can only be chosen on the first configure of a build dir
GEN_ARGS=()
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    if ninja --version > /dev/null 2>&1; then
        GEN_ARGS=(-G Ninja)
    else
        GEN_ARGS=(-G "Unix Makefiles")
    fi
fi

"$SWITCH_CMAKE" -S "$ROOT" -B "$BUILD_DIR" "${GEN_ARGS[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_BACKEND_DL=OFF \
    -DGGML_CCACHE=OFF \
    -DLLAMA_BUILD_COMMON=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_OPENSSL=OFF \
    "$@"

cmake --build "$BUILD_DIR" --target llama-switch_nro -j "$JOBS"

NRO="$(find "$BUILD_DIR" -name 'llama-switch.nro' -print -quit)"
if [ -z "$NRO" ]; then
    echo "error: llama-switch.nro was not produced" >&2
    exit 1
fi
echo "NRO: $NRO"
