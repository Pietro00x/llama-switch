#pragma once

// Multi-row GEMV kernels for single-token decode on AArch64 NEON cores without dotprod
// (Cortex-A57 / Nintendo Switch). See gemv-a57.c.

#include <stddef.h>

#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(__ARM_FEATURE_DOTPROD)
#define GGML_CPU_GEMV_A57 1
#endif

#ifdef GGML_CPU_GEMV_A57

#ifdef __cplusplus
extern "C" {
#endif

// s[r] = dot(row r of vx, vy) for r in [0, nr)
//   n : number of elements per row (multiple of 32)
//   vx: nr rows of block_q4_0 (or block_q8_0), row r starts at (const char *) vx + r*bx
//   vy: one row of n/32 block_q8_0 (the quantized src1 column)
void ggml_gemv_a57_q4_0_q8_0(int n, float * s, const void * vx, size_t bx, const void * vy, int nr);
// q8_0 variant: the activations in vy must be in [-127, 127] (as produced by quantize_row_q8_0)
void ggml_gemv_a57_q8_0_q8_0(int n, float * s, const void * vx, size_t bx, const void * vy, int nr);

#ifdef __cplusplus
}
#endif

#endif // GGML_CPU_GEMV_A57
