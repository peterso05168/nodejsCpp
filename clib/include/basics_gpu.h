#pragma once

#include "basics.h"

#define _DI_ __attribute__((always_inline)) __device__ inline

/**
 * Thread-local shared memory allows efficient computed offsets
 * without spilling registers to local memory.
 */
template<typename T>
struct TLSM {

    T *ptr;

    _DI_ T& operator[](int idx) {
        return ptr[idx * blockDim.x + threadIdx.x];
    }

    _DI_ T operator[](int idx) const {
        return ptr[idx * blockDim.x + threadIdx.x];
    }

    _DI_ TLSM(T* ptr) : ptr(ptr) { }

};
