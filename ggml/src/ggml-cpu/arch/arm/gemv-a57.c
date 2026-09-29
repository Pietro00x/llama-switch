// Multi-row GEMV kernels for single-token decode (src1 has one column) on AArch64 NEON cores
// without dotprod, e.g. the Cortex-A57 of the Nintendo Switch.
//
// On these cores neither the repacked kernels (need dotprod/i8mm) nor llamafile sgemm (needs n >= 2)
// apply, so every output row of a mul_mat costs one vec_dot call, whose int8 dot product is emulated
// with vmull_s8 + vpaddlq_s16 per 8 products. Here 4 weight rows are processed together:
//   - the q8_0 activation block and its scale are loaded once for 4 rows,
//   - products are accumulated in int16 lanes and the 4 rows are reduced together with pairwise adds,
//     so the int16 -> int32 widening is done once per 4 rows instead of 4 times per row,
//   - the 4 row scales are converted and applied as one vector.
// For Q4_0 the "-8" offset of the nibbles is not subtracted per row: dot(q - 8, y) = dot(q, y) - 8*sum(y),
// with -8*sum(y) precomputed once per activation block for all rows of the call.

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-impl.h"
#include "simd-mappings.h"

#include "gemv-a57.h"

#ifdef GGML_CPU_GEMV_A57

#include <arm_neon.h>
#include <assert.h>
#include <stdbool.h>

// number of activation blocks whose (d, -8*sum(qs)) is precomputed at a time (2 KiB of stack);
// rows longer than GEMV_A57_TILE*32 elements are processed in several passes
#define GEMV_A57_TILE 256

// software prefetch distance in bytes for the 4 weight rows streamed at once (0 = off). The 4 interleaved
// row streams are not a single fixed-stride miss pattern, which the hardware prefetcher may not follow.
// Can be set with -DGGML_CPU_GEMV_A57_PREFETCH=<bytes> at configure time.
#ifndef GEMV_A57_PREFETCH
#define GEMV_A57_PREFETCH 256
#endif

// 16 unsigned nibble pairs (0..15) of a q4_0 block times the 32 int8 of a q8_0 block, 4 products per
// int16 lane: |sum| <= 4*15*128 = 7680
static inline int16x8_t gemv_a57_q4_dot16(const uint8_t * GGML_RESTRICT qs, const uint8x16_t m4b, const int8x16_t y0, const int8x16_t y1) {
    const uint8x16_t v  = vld1q_u8(qs);
    const int8x16_t  lo = vreinterpretq_s8_u8(vandq_u8(v, m4b));
    const int8x16_t  hi = vreinterpretq_s8_u8(vshrq_n_u8(v, 4));

    int16x8_t a = vmull_s8(vget_low_s8(lo), vget_low_s8(y0));
    a = vmlal_high_s8(a, lo, y0);
    a = vmlal_s8     (a, vget_low_s8(hi), vget_low_s8(y1));
    a = vmlal_high_s8(a, hi, y1);
    return a;
}

void ggml_gemv_a57_q4_0_q8_0(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, int nr) {
    assert(n % QK8_0 == 0);
    static_assert(QK4_0 == QK8_0, "QK4_0 != QK8_0");

    const int nb = n / QK8_0;

    const block_q8_0 * GGML_RESTRICT y = (const block_q8_0 *) vy;
    const uint8x16_t  m4b  = vdupq_n_u8(0x0F);
    const float32x4_t ones = vdupq_n_f32(1.0f);

    // per activation block: { d_y, -8*sum(y.qs) }
    float tab[2*GEMV_A57_TILE];

    for (int ib0 = 0; ib0 < nb; ib0 += GEMV_A57_TILE) {
        const int  nbt   = MIN(GEMV_A57_TILE, nb - ib0);
        const bool first = ib0 == 0;

        const block_q8_0 * GGML_RESTRICT yt = y + ib0;

        for (int ib = 0; ib < nbt; ++ib) {
            const int8x16_t y0 = vld1q_s8(yt[ib].qs);
            const int8x16_t y1 = vld1q_s8(yt[ib].qs + 16);
            const int32_t   sy = vaddlvq_s16(vpadalq_s8(vpaddlq_s8(y0), y1));
            const float     d  = GGML_CPU_FP16_TO_FP32(yt[ib].d);
            tab[2*ib + 0] = d;
            tab[2*ib + 1] = -8.0f*(float) sy;
        }

        int ir = 0;

        for (; ir + 4 <= nr; ir += 4) {
            const block_q4_0 * GGML_RESTRICT x0 = (const block_q4_0 *) ((const char *) vx + (size_t) (ir + 0)*bx) + ib0;
            const block_q4_0 * GGML_RESTRICT x1 = (const block_q4_0 *) ((const char *) vx + (size_t) (ir + 1)*bx) + ib0;
            const block_q4_0 * GGML_RESTRICT x2 = (const block_q4_0 *) ((const char *) vx + (size_t) (ir + 2)*bx) + ib0;
            const block_q4_0 * GGML_RESTRICT x3 = (const block_q4_0 *) ((const char *) vx + (size_t) (ir + 3)*bx) + ib0;

            float32x4_t sum = vdupq_n_f32(0.0f);

            for (int ib = 0; ib < nbt; ++ib) {
#if GEMV_A57_PREFETCH > 0
                __builtin_prefetch((const char *) (x0 + ib) + GEMV_A57_PREFETCH);
                __builtin_prefetch((const char *) (x1 + ib) + GEMV_A57_PREFETCH);
                __builtin_prefetch((const char *) (x2 + ib) + GEMV_A57_PREFETCH);
                __builtin_prefetch((const char *) (x3 + ib) + GEMV_A57_PREFETCH);
#endif
                const int8x16_t   y0 = vld1q_s8(yt[ib].qs);
                const int8x16_t   y1 = vld1q_s8(yt[ib].qs + 16);
                const float       dy = tab[2*ib + 0];
                const float       sc = tab[2*ib + 1];

                uint16x4_t dh = vld1_dup_u16((const uint16_t *) &x0[ib].d);
                dh = vld1_lane_u16((const uint16_t *) &x1[ib].d, dh, 1);
                dh = vld1_lane_u16((const uint16_t *) &x2[ib].d, dh, 2);
                dh = vld1_lane_u16((const uint16_t *) &x3[ib].d, dh, 3);
                const float32x4_t dx = vcvt_f32_f16(vreinterpret_f16_u16(dh));

                const int16x8_t a0 = gemv_a57_q4_dot16(x0[ib].qs, m4b, y0, y1);
                const int16x8_t a1 = gemv_a57_q4_dot16(x1[ib].qs, m4b, y0, y1);
                const int16x8_t a2 = gemv_a57_q4_dot16(x2[ib].qs, m4b, y0, y1);
                const int16x8_t a3 = gemv_a57_q4_dot16(x3[ib].qs, m4b, y0, y1);

                // lane i of p: 16 products of row i/2, |p| <= 30720 still fits int16
                const int16x8_t p = vpaddq_s16(vpaddq_s16(a0, a1), vpaddq_s16(a2, a3));
                const int32x4_t q = vpaddlq_s16(p); // dot(q, y) of rows 0..3

                // dot(q - 8, y) = dot(q, y) - 8*sum(y), exact in fp32 (integers below 2^17)
                const float32x4_t dot = vfmaq_n_f32(vcvtq_f32_s32(q), ones, sc);

                sum = vfmaq_f32(sum, dot, vmulq_n_f32(dx, dy));
            }

            if (!first) {
                sum = vaddq_f32(sum, vld1q_f32(s + ir));
            }
            vst1q_f32(s + ir, sum);
        }

        // remaining rows, one at a time
        for (; ir < nr; ++ir) {
            const block_q4_0 * GGML_RESTRICT x0 = (const block_q4_0 *) ((const char *) vx + (size_t) ir*bx) + ib0;

            float sum = 0.0f;

            for (int ib = 0; ib < nbt; ++ib) {
                const int8x16_t y0 = vld1q_s8(yt[ib].qs);
                const int8x16_t y1 = vld1q_s8(yt[ib].qs + 16);

                const int32_t q = vaddlvq_s16(gemv_a57_q4_dot16(x0[ib].qs, m4b, y0, y1));

                sum += ((float) q + tab[2*ib + 1])*(GGML_CPU_FP16_TO_FP32(x0[ib].d)*tab[2*ib + 0]);
            }

            s[ir] = first ? sum : s[ir] + sum;
        }
    }
}

// 32 int8 of a q8_0 weight block times the 32 int8 of a q8_0 activation block, 8 products per int32 lane.
// Two products are summed in int16 before widening: |sum| <= 2*128*127 = 32512 as long as the activations
// are in [-127, 127] (quantize_row_q8_0 output; the caller guarantees it), whatever the weights are.
static inline int32x4_t gemv_a57_q8_dot32(const int8_t * GGML_RESTRICT qs, const int8x16_t y0, const int8x16_t y1) {
    const int8x16_t x0 = vld1q_s8(qs);
    const int8x16_t x1 = vld1q_s8(qs + 16);

    const int16x8_t a = vmlal_high_s8(vmull_s8(vget_low_s8(x0), vget_low_s8(y0)), x0, y0);
    const int16x8_t b = vmlal_high_s8(vmull_s8(vget_low_s8(x1), vget_low_s8(y1)), x1, y1);

    return vpadalq_s16(vpaddlq_s16(a), b);
}

void ggml_gemv_a57_q8_0_q8_0(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, int nr) {
    assert(n % QK8_0 == 0);

    const int nb = n / QK8_0;

    const block_q8_0 * GGML_RESTRICT y = (const block_q8_0 *) vy;

    int ir = 0;

    for (; ir + 4 <= nr; ir += 4) {
        const block_q8_0 * GGML_RESTRICT x0 = (const block_q8_0 *) ((const char *) vx + (size_t) (ir + 0)*bx);
        const block_q8_0 * GGML_RESTRICT x1 = (const block_q8_0 *) ((const char *) vx + (size_t) (ir + 1)*bx);
        const block_q8_0 * GGML_RESTRICT x2 = (const block_q8_0 *) ((const char *) vx + (size_t) (ir + 2)*bx);
        const block_q8_0 * GGML_RESTRICT x3 = (const block_q8_0 *) ((const char *) vx + (size_t) (ir + 3)*bx);

        float32x4_t sum = vdupq_n_f32(0.0f);

        for (int ib = 0; ib < nb; ++ib) {
#if GEMV_A57_PREFETCH > 0
            __builtin_prefetch((const char *) (x0 + ib) + GEMV_A57_PREFETCH);
            __builtin_prefetch((const char *) (x1 + ib) + GEMV_A57_PREFETCH);
            __builtin_prefetch((const char *) (x2 + ib) + GEMV_A57_PREFETCH);
            __builtin_prefetch((const char *) (x3 + ib) + GEMV_A57_PREFETCH);
#endif
            const int8x16_t y0 = vld1q_s8(y[ib].qs);
            const int8x16_t y1 = vld1q_s8(y[ib].qs + 16);
            const float     dy = GGML_CPU_FP16_TO_FP32(y[ib].d);

            uint16x4_t dh = vld1_dup_u16((const uint16_t *) &x0[ib].d);
            dh = vld1_lane_u16((const uint16_t *) &x1[ib].d, dh, 1);
            dh = vld1_lane_u16((const uint16_t *) &x2[ib].d, dh, 2);
            dh = vld1_lane_u16((const uint16_t *) &x3[ib].d, dh, 3);
            const float32x4_t dx = vcvt_f32_f16(vreinterpret_f16_u16(dh));

            const int32x4_t a0 = gemv_a57_q8_dot32(x0[ib].qs, y0, y1);
            const int32x4_t a1 = gemv_a57_q8_dot32(x1[ib].qs, y0, y1);
            const int32x4_t a2 = gemv_a57_q8_dot32(x2[ib].qs, y0, y1);
            const int32x4_t a3 = gemv_a57_q8_dot32(x3[ib].qs, y0, y1);

            const int32x4_t q = vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3)); // rows 0..3

            sum = vfmaq_f32(sum, vcvtq_f32_s32(q), vmulq_n_f32(dx, dy));
        }

        vst1q_f32(s + ir, sum);
    }

    // remaining rows, one at a time
    for (; ir < nr; ++ir) {
        const block_q8_0 * GGML_RESTRICT x0 = (const block_q8_0 *) ((const char *) vx + (size_t) ir*bx);

        float sum = 0.0f;

        for (int ib = 0; ib < nb; ++ib) {
            const int8x16_t y0 = vld1q_s8(y[ib].qs);
            const int8x16_t y1 = vld1q_s8(y[ib].qs + 16);

            const int32_t q = vaddvq_s32(gemv_a57_q8_dot32(x0[ib].qs, y0, y1));

            sum += GGML_CPU_FP16_TO_FP32(x0[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d)*(float) q;
        }

        s[ir] = sum;
    }
}

#endif // GGML_CPU_GEMV_A57
