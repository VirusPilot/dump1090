/*
 * Count the number of samples in a uint16_t buffer that are >= a threshold.
 */
void STARCH_IMPL(count_above_u16, generic) (const uint16_t * restrict in, unsigned len, uint16_t threshold, unsigned *out_count)
{
    unsigned count = 0;
    while (len--) {
        if (in[0] >= threshold)
            ++count;
        ++in;
    }

    *out_count = count;
}
