#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdalign.h>
#include <inttypes.h>

#include "compat/compat.h"

#include "dsp/helpers/tables.h"

/* Convert UC8 values to unsigned 16-bit magnitudes */

void STARCH_IMPL(magnitude_uc8, lookup) (const uc8_t * restrict in, uint16_t * restrict out, unsigned len)
{
    const uint16_t * const mag_table = get_uc8_mag_table();
    const uc8_u16_t * restrict in_u = (const uc8_u16_t *) in;

    unsigned len1 = len;
    while (len1--) {
        uint16_t mag = mag_table[in_u[0].u16];
        out[0] = mag;

        out += 1;
        in_u += 1;
    }
}

void STARCH_IMPL(magnitude_uc8, lookup_unroll_4) (const uc8_t * restrict in, uint16_t * restrict out, unsigned len)
{
    const uint16_t * const mag_table = get_uc8_mag_table();
    const uc8_u16_t * restrict in_u = (const uc8_u16_t *) in;

    unsigned len4 = len >> 2;
    unsigned len1 = len & 3;

    while (len4--) {
        uint16_t mag0 = mag_table[in_u[0].u16];
        uint16_t mag1 = mag_table[in_u[1].u16];
        uint16_t mag2 = mag_table[in_u[2].u16];
        uint16_t mag3 = mag_table[in_u[3].u16];

        out[0] = mag0;
        out[1] = mag1;
        out[2] = mag2;
        out[3] = mag3;

        out += 4;
        in_u += 4;
    }

    while (len1--) {
        uint16_t mag = mag_table[in_u[0].u16];

        out[0] = mag;

        out += 1;
        in_u += 1;
    }
}

void STARCH_IMPL(magnitude_uc8, exact) (const uc8_t * restrict in, uint16_t * restrict out, unsigned len)
{
    unsigned len1 = len;

    while (len1--) {
        float I = (in[0].I - 127.4);
        float Q = (in[0].Q - 127.4);

        float magsq = I * I + Q * Q;
        float mag = sqrtf(magsq) * 65536.0 / 128.0;
        if (mag > 65535.0)
            mag = 65535.0;

        out[0] = (uint16_t)mag;

        in += 1;
        out += 1;
    }
}
