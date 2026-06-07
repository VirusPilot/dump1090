#ifndef __ARM_NEON
#  error Neon support not available
#endif

#include <arm_neon.h>

void STARCH_IMPL(count_above_u16, neon) (const uint16_t * restrict in, unsigned len, uint16_t threshold, unsigned *out_count)
{
    const uint16x8_t threshold_x8 = vdupq_n_u16(threshold);

    int32x4_t accumulator0 = vdupq_n_s32(0);
    int32x4_t accumulator1 = vdupq_n_s32(0);

    unsigned len8 = len >> 3;
    while (len8--) {
        uint16x8_t mag = vld1q_u16(in);
        int16x8_t compare = vreinterpretq_s16_u16(vcgeq_u16(mag, threshold_x8));
        accumulator0 = vsubw_s16(accumulator0, vget_low_s16(compare));
        accumulator1 = vsubw_s16(accumulator1, vget_high_s16(compare));

        in += 8;
    }

    // sum accumulators across all lanes
    int32x4_t sum2 = vaddq_s32(accumulator0, accumulator1);
    int32x2_t sum4 = vadd_s32(vget_low_s32(sum2), vget_high_s32(sum2));
    int32x2_t sum8 = vpadd_s32(sum4, sum4);
    int32x4_t sum8_x2 = vcombine_s32(sum8, sum8);

    unsigned len1 = len & 7;
    while (len1--) {
        uint16x4_t mag = vld1_dup_u16(in);
        int16x4_t compare = vreinterpret_s16_u16(vcge_u16(mag, vget_low_u16(threshold_x8)));
        sum8_x2 = vsubw_s16(sum8_x2, compare);

        in += 1;
    }

    *out_count = vgetq_lane_s32(sum8_x2, 0);
}
