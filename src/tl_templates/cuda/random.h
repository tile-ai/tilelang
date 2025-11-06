#pragma once

#include "common.h"

namespace tl {

TL_DEVICE uint32_t umulhi_uint32(uint32_t a, uint32_t b) {
    uint32_t result;
    asm("mul.hi.u32 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
    return result;
}

TL_DEVICE void philox_impl_device(
    uint32_t* c0, uint32_t* c1, uint32_t* c2, uint32_t* c3,
    uint32_t k0, uint32_t k1, int n_rounds
) {
    const uint32_t PHILOX_KEY_A = 0x9E3779B9U;
    const uint32_t PHILOX_KEY_B = 0xBB67AE85U;
    const uint32_t PHILOX_ROUND_A = 0xD2511F53U;
    const uint32_t PHILOX_ROUND_B = 0xCD9E8D57U;
    uint32_t c0_val = *c0;
    uint32_t c1_val = *c1;
    uint32_t c2_val = *c2;
    uint32_t c3_val = *c3;
    uint32_t k0_val = k0;
    uint32_t k1_val = k1;
    for (int round = 0; round < n_rounds; round++) {
        uint32_t _c0 = c0_val;
        uint32_t _c2 = c2_val;
        uint32_t A = PHILOX_ROUND_A;
        uint32_t B = PHILOX_ROUND_B;
        c0_val = umulhi_uint32(B, _c2) ^ c1_val ^ k0_val;
        c2_val = umulhi_uint32(A, _c0) ^ c3_val ^ k1_val;
        c1_val = (uint32_t)((uint64_t)B * (uint64_t)_c2);
        c3_val = (uint32_t)((uint64_t)A * (uint64_t)_c0);
        k0_val = (uint32_t)((uint64_t)k0_val + PHILOX_KEY_A);
        k1_val = (uint32_t)((uint64_t)k1_val + PHILOX_KEY_B);
    }
    *c0 = c0_val;
    *c1 = c1_val;
    *c2 = c2_val;
    *c3 = c3_val;
}


TL_DEVICE float uint32_to_uniform_float_device(uint32_t x) {
    const float scale = 4.6566127342e-10f;
    int32_t x_int32;
    memcpy(&x_int32, &x, sizeof(uint32_t));
    int32_t x_abs = (x_int32 < 0) ? (-x_int32 - 1) : x_int32;
    return (float)x_abs * scale;
}


TL_DEVICE void philox_rand(
    float* output,
    int total_elems,
    uint64_t seed,
    int n_rounds
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;

    uint32_t seed_lo = (uint32_t)(seed & 0xFFFFFFFFULL);
    uint32_t seed_hi = (uint32_t)((seed >> 32) & 0xFFFFFFFFULL);
    uint32_t offset_lo = (uint32_t)idx;
    uint32_t offset_hi = 0U;
    uint32_t c0 = offset_lo;
    uint32_t c1 = offset_hi;
    uint32_t c2 = 0U;
    uint32_t c3 = 0U;
    philox_impl_device(&c0, &c1, &c2, &c3, seed_lo, seed_hi, n_rounds);
    output[idx] = uint32_to_uniform_float_device(c0);
}

} // namespace tl
