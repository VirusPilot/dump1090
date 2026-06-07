/*
 * Given a buffer of uint16_t Q16 magnitude values
 * return the mean magnitude and mean squared magnitude
 * (normalized to 0..1)
 */

void STARCH_IMPL(mean_power_u16, float) (const uint16_t * restrict in, unsigned len, double *out_mean_mag, double *out_mean_magsq)
{
    float sum = 0, sumsq = 0;
    unsigned n = len;
    while (n--) {
        uint16_t mag = in[0];
        sum += mag;
        sumsq += (uint32_t)mag * mag;
        in += 1;
    }

    *out_mean_mag = sum / len / 65536.0;
    *out_mean_magsq = sumsq / len / 65536.0 / 65536.0;
}

void STARCH_IMPL(mean_power_u16, u32) (const uint16_t * restrict in, unsigned len, double *out_mean_mag, double *out_mean_magsq)
{
    double sum = 0, sumsq = 0;

    unsigned remaining = len;
    while (remaining > 0) {
        uint32_t sum32 = 0, sumsq32 = 0;
        unsigned blocklen = (remaining > 65536 ? 65536 : remaining);
        remaining -= blocklen;

        while (blocklen--) {
            uint16_t mag = in[0];
            sum32 += mag;
            sumsq32 += ((uint32_t)mag * mag) >> 16;
            in += 1;
        }

        sum += sum32;
        sumsq += sumsq32;
    }

    *out_mean_mag = (double)sum / len / 65536.0;
    *out_mean_magsq = (double)sumsq / len / 65536.0;
}

void STARCH_IMPL(mean_power_u16, u64) (const uint16_t * restrict in, unsigned len, double *out_mean_mag, double *out_mean_magsq)
{
    uint64_t sum = 0, sumsq = 0;
    unsigned n = len;
    while (n--) {
        uint16_t mag = in[0];
        sum += mag;
        sumsq += (uint32_t)mag * mag;
        in += 1;
    }

    *out_mean_mag = (double)sum / len / 65536.0;
    *out_mean_magsq = (double)sumsq / len / 65536.0 / 65536.0;
}
