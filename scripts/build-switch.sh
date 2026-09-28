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

# CPU code generation for the Tegra X1 (Cortex-A57, ARMv8.0-A):
#   - GGML_CPU_ARM_ARCH pins ggml-cpu to ARMv8.0 + NEON (+crc/crypto, same as the devkitA64 defaults).
#     The A57 has no dotprod, i8mm, fp16 vector arithmetic, SVE or LSE atomics - code using any of them
#     dies with an illegal instruction, so never raise this to armv8.2-a or later.
#   - devkitPro's Release flags are "-g -O2"; use -O3 like a regular upstream build (the -g only lives
#     in the .elf, the .nro is not affected). Costs ~350 KB of .nro size.
#   - GGML_LTO=ON also builds and links with devkitA64 (slightly smaller .nro) but is left off: the hot
#     ggml-cpu kernels are reached through function pointer tables, so there is little to inline, and the
#     speed-up is unverified on hardware. Try it with: scripts/build-switch.sh -DGGML_LTO=ON
# Options passed on the command line come last and override these.
"$SWITCH_CMAKE" -S "$ROOT" -B "$BUILD_DIR" "${GEN_ARGS[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS_RELEASE="-g -O3 -DNDEBUG" \
    -DCMAKE_CXX_FLAGS_RELEASE="-g -O3 -DNDEBUG" \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ARM_ARCH=armv8-a+crc+crypto \
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
