/*
 *  cpu.c - flavor/CPU detection support for starch
 *
 *  Copyright (c) 2026 FlightAware All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are
 *  met:
 *
 *  1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *  notice, this list of conditions and the following disclaimer in the
 *  documentation and/or other materials provided with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "cpu.h"

#if defined(__GNUC__) && defined(__x86_64__)
#  include <cpuid.h>
#  define HAVE_X86_CPUID
#endif

#if defined(__linux__) && defined(__arm__)
#  include <asm/hwcap.h>
#  include <sys/auxv.h>
#  define HAVE_ARM32_AUXV
#endif

#if defined(__linux__) && defined(__aarch64__)
#  include <asm/hwcap.h>
#  include <sys/auxv.h>
#  define HAVE_ARM64_AUXV
#endif

int cpu_supports_armv7_neon_vfpv4(void)
{
#ifdef HAVE_ARM32_AUXV
    long hwcaps = getauxval(AT_HWCAP);
    return
        (hwcaps & HWCAP_ARM_NEON) &&
        (hwcaps & HWCAP_ARM_VFPv4);
#else
    return 0;
#endif
}

int cpu_supports_armv8_simd(void)
{
#ifdef HAVE_ARM64_AUXV
    long hwcaps = getauxval(AT_HWCAP);
    return (hwcaps & HWCAP_ASIMD);
#else
    return 0;
#endif
}

int cpu_supports_avx(void)
{
#ifdef HAVE_X86_CPUID
    unsigned int maxlevel = __get_cpuid_max (0, 0);
    if (maxlevel < 1)
        return 0;

    unsigned eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    if (!(ecx & bit_AVX))
        return 0;

    return 1;
#else
    return 0;
#endif
}

int cpu_supports_avx2(void)
{
#ifdef HAVE_X86_CPUID
    unsigned int maxlevel = __get_cpuid_max (0, 0);
    if (maxlevel < 7)
        return 0;

    unsigned eax, ebx, ecx, edx;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    if (!(ebx & bit_AVX2))
        return 0;

    return 1;
#else
    return 0;
#endif
}
