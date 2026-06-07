/*
 * Given a buffer of uint16_t Q16 magnitude values
 * return the mean magnitude and mean squared magnitude
 * (normalized to 0..1)
 */

#ifndef __ARM_NEON
#  error Neon support not available
#endif

#include <arm_neon.h>

void STARCH_IMPL(mean_power_u16, neon_float) (const uint16_t * restrict in, unsigned len, double *out_mean_mag, double *out_mean_magsq)
{
    float32x4_t mag_sum_0 = vdupq_n_f32(0);
    float32x4_t magsq_sum_0 = vdupq_n_f32(0);
    float32x4_t mag_sum_1 = vdupq_n_f32(0);
    float32x4_t magsq_sum_1 = vdupq_n_f32(0);

    unsigned len8 = len >> 3;
    while (len8--) {
        uint16x8_t mag_u16 = vld1q_u16(in);
        uint16x4_t mag_u16_0 = vget_low_u16(mag_u16);
        uint16x4_t mag_u16_1 = vget_high_u16(mag_u16);

        float32x4_t mag_float32_0 = vcvtq_n_f32_u32(vmovl_u16(mag_u16_0), 16);
        float32x4_t mag_float32_1 = vcvtq_n_f32_u32(vmovl_u16(mag_u16_1), 16);

        mag_sum_0 = vaddq_f32(mag_sum_0, mag_float32_0);
        mag_sum_1 = vaddq_f32(mag_sum_1, mag_float32_1);

        magsq_sum_0 = vfmaq_f32(magsq_sum_0, mag_float32_0, mag_float32_0);
        magsq_sum_1 = vfmaq_f32(magsq_sum_1, mag_float32_1, mag_float32_1);

        in += 8;
    }

    // reduce sums to lane 0
    float32x4_t mag_sum_q = vaddq_f32(mag_sum_0, mag_sum_1);
    float32x2_t mag_sum = vadd_f32(vget_low_f32(mag_sum_q), vget_high_f32(mag_sum_q));
    mag_sum = vpadd_f32(mag_sum, mag_sum);

    float32x4_t magsq_sum_q = vaddq_f32(magsq_sum_0, magsq_sum_1);
    float32x2_t magsq_sum = vadd_f32(vget_low_f32(magsq_sum_q), vget_high_f32(magsq_sum_q));
    magsq_sum = vpadd_f32(magsq_sum, magsq_sum);

    unsigned len1 = len & 7;
    while (len1--) {
        uint16x4_t mag_u16 = vld1_dup_u16(in);
        // we process both lanes here, but lane 1's sums are ignored
        float32x2_t mag_float32 = vcvt_n_f32_u32(vget_low_u32(vmovl_u16(mag_u16)), 16);
        mag_sum = vadd_f32(mag_sum, mag_float32);
        magsq_sum = vfma_f32(magsq_sum, mag_float32, mag_float32);
        in += 1;
    }

    *out_mean_mag = vget_lane_f32(mag_sum, 0) / len;
    *out_mean_magsq = vget_lane_f32(magsq_sum, 0) / len;
}
