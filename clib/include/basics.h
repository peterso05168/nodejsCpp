#pragma once

#include <stdint.h>

#ifdef __CUDA_ARCH__
#define _HD_ __attribute__((always_inline)) __host__ __device__ inline
#else
#define _HD_ __attribute__((always_inline)) inline
#endif

/**
 * A variant of Melissa O'Neill's permuted congruential generator.
 * This produces high-quality random integers with a period of
 * 2**128 - 2**64.
 */
struct PCG {

    uint64_t state;
    uint64_t inc;

    _HD_ uint32_t generate() {

        uint64_t oldstate = state;

        // update LCG component:
        state = oldstate * 6364136223846793005ull + 3511ull;

        // variant of Melissa O'Neill's PCG:
        uint64_t xorshifted = ((oldstate >> ((oldstate >> 59u) + 5u)) ^ oldstate);
        uint32_t output = (uint32_t) ((xorshifted * (inc | 65537)) >> 32);

        // update LFSR component:
        bool galois = inc & 1;
        inc >>= 1;
        if (galois) { inc ^= 0x800000000000000DULL; }

        return output;
    }

    _HD_ void initialise(int xa, int xb, int xc, int xd) {

        // multiplication by prime constants
        uint32_t a = xa * 1414213573u;
        uint32_t b = xb * 1732050821u;
        uint32_t c = xc * 2236067989u;
        uint32_t d = xd * 2645751323u;

        // chacha20 quarter-round mixing function
        a += b; d ^= a; d = (d << 16) | (d >> 16);
        c += d; b ^= c; b = (b << 12) | (b >> 20);
        a += b; d ^= a; d = (d <<  8) | (d >>  8);
        c += d; b ^= c; b = (b <<  7) | (b >>  7);

        // combine into two odd 64-bit integers
        state = (((uint64_t) a) << 32) | b | 1;
        inc   = (((uint64_t) c) << 32) | d | 1;
    }

};
