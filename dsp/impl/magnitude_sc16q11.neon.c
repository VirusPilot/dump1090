#include <math.h>

#include "compat/compat.h"

#include "dsp/helpers/tables.h"

/* Convert (little-endian) SC16 values with a range of -2048..+2047 to unsigned 16-bit magnitudes */

#ifndef __ARM_NEON
#  error Neon support not available
#endif

#include <arm_neon.h>

void STARCH_IMPL(magnitude_sc16q11, neon_vrsqrte) (const sc16_t * restrict in, uint16_t * restrict out, unsigned len)
{
    /* This uses NEON's floating-point reciprocal square root estimate instruction (vrsqrte).
     * The estimate is accurate to about 9 bits of mantissa, which is good enough for our purposes.
     */

    const int16_t * restrict in_i16 = (const int16_t *) in;
    
    unsigned len4 = len >> 2;
    while (len4--) {
        int16x4x2_t iq = vld2_s16(in_i16);
        int16x4_t i16 = iq.val[0]; /* Q11 */
        int16x4_t q16 = iq.val[1]; /* Q11 */

        uint32x4_t isq = vreinterpretq_u32_s32(vmull_s16(i16, i16)); /* Q22, unsigned */
        uint32x4_t qsq = vreinterpretq_u32_s32(vmull_s16(q16, q16)); /* Q22, unsigned */
        uint32x4_t magsq = vqaddq_u32(isq, qsq);                     /* Q22, unsigned */

        float32x4_t magsq_f32 = vcvtq_n_f32_u32(magsq, 22);
        float32x4_t mag_f32 = vmulq_f32(magsq_f32, vrsqrteq_f32(magsq_f32));  /* sqrt(x) = x * (1/sqrt(x)) */
        uint16x4_t mag_u16 = vqmovn_u32(vcvtq_n_u32_f32(mag_f32, 16));

        vst1_u16(out, mag_u16);

        in_i16 += 8;
        out += 4;
    }

    unsigned len1 = len & 3;
    while (len1--) {
        int16x4x2_t iq = vld2_dup_s16(in_i16);
        int16x4_t i16 = iq.val[0];
        int16x4_t q16 = iq.val[1];

        uint32x4_t isq = vreinterpretq_u32_s32(vmull_s16(i16, i16));
        uint32x4_t qsq = vreinterpretq_u32_s32(vmull_s16(q16, q16));
        uint32x4_t magsq = vqaddq_u32(isq, qsq);

        float32x4_t magsq_f32 = vcvtq_n_f32_u32(magsq, 22);
        float32x4_t mag_f32 = vmulq_f32(magsq_f32, vrsqrteq_f32(magsq_f32));
        uint16x4_t mag_u16 = vqmovn_u32(vcvtq_n_u32_f32(mag_f32, 16));

        vst1_lane_u16(out, mag_u16, 0);

        in_i16 += 2;
        out += 1;
    }
}
