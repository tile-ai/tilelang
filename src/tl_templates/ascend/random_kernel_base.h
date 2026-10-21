#pragma once

// SIMT scalar primitives of the AscendC Philox-4x32 RNG.
//
// Vendored from the AscendC SDK's SIMT random base (namespace
// RandomKernelBase):
//   /usr/local/Ascend/cann/opp/built-in/op_impl/ai_core/tbe/impl/ops_math/
//     ascendc/random_common/arch35/random_kernel_base.h
//   (CANN 9.1.T560, __NPU_ARCH__ == 3510)
//
// Only the self-contained scalar `__simt_callee__` functions are vendored here,
// verbatim. The SDK header cannot be #include'd directly from a TileLang
// kernel: its file-scope SIMD code (MicroAPI::CastTrait /
// Ops::Base::GetVRegSize / the
// __VEC_SCOPE__ helpers and RandomKernelBaseOp) requires CANN's op-build
// internal environment (the reg_compute impl + pkg_inc/op_common), which is not
// on the Bisheng include path used by tilelang. Those SIMD/tiling pieces are
// unused by the SIMT path, so they are omitted here.
//
// The __simt_callee__ / __aicore__ attributes come from simt_api/asc_simt.h,
// pulled in by tl_templates/ascend/common.h.

#include <stdint.h>

namespace RandomKernelBase {

static constexpr uint16_t ALG_KEY_SIZE = 2;
static constexpr uint16_t ALG_COUNTER_SIZE = 4;
static constexpr uint32_t RIGHT_SHIFT = 32;
static constexpr int IDX_2 = 2;
static constexpr int IDX_3 = 3;
static constexpr uint32_t PHILOX_W32_A = 0x9E3779B9;
static constexpr uint32_t PHILOX_W32_B = 0xBB67AE85;
static constexpr uint32_t PHILOX_M4X32_A = 0xD2511F53;
static constexpr uint32_t PHILOX_M4X32_B = 0xCD9E8D57;
static constexpr float RAND_2POW32_INV = 2.3283064e-10f;
static constexpr float RAND_2POW32_INV_HALF = RAND_2POW32_INV / 2.0f;
static constexpr float DOUBLE_MULTIPLE = 2.0f;
static constexpr float PI = 3.14159265358979323846f;

constexpr uint64_t VEC_4 = 4;

template <uint16_t COPY_SIZE>
__simt_callee__ __aicore__ inline void CopyArray(uint32_t *dst,
                                                 const uint32_t *src) {
#pragma unroll
  for (uint16_t i = 0; i < COPY_SIZE; i++) {
    dst[i] = src[i];
  }
}

__simt_callee__ __aicore__ inline void SkipOne(uint32_t *counter) {
  if (++counter[0])
    return;
  if (++counter[1])
    return;
  if (++counter[IDX_2])
    return;
  ++counter[IDX_3];
}

__simt_callee__ __aicore__ inline void SkipLo(uint32_t *counter, uint64_t n) {
  const uint32_t nlo = static_cast<uint32_t>(n);
  uint32_t nhi = static_cast<uint32_t>(n >> RIGHT_SHIFT);

  counter[0] += nlo;
  if (counter[0] < nlo) {
    nhi++;
  }
  counter[1] += nhi;
  if (nhi <= counter[1])
    return;
  if (++counter[IDX_2])
    return;
  ++counter[IDX_3];
}

__simt_callee__ __aicore__ inline void SkipHi(uint32_t *counter, uint64_t n) {
  const uint32_t countLo = static_cast<uint32_t>(n);
  uint32_t countHi = static_cast<uint32_t>(n >> RIGHT_SHIFT);

  counter[IDX_2] += countLo;
  if (counter[IDX_2] < countLo) {
    countHi++;
  }
  counter[IDX_3] += countHi;
}

__simt_callee__ __aicore__ inline void
FlashCounter(uint64_t globalThreadIdx, uint64_t offset, uint32_t *counter) {
  SkipHi(counter, globalThreadIdx);
  SkipLo(counter, offset);
}

__simt_callee__ __aicore__ inline void PhiloxAlgParsInit(uint32_t *key,
                                                         uint32_t *counter,
                                                         int64_t seed,
                                                         int64_t offset) {
  key[0] = static_cast<uint32_t>(seed);
  key[1] = static_cast<uint32_t>(seed >> RIGHT_SHIFT);

  offset = (offset + VEC_4 - 1) / VEC_4;
  SkipLo(counter, offset);
}

__simt_callee__ __aicore__ inline void MultiplyHighLow(uint32_t a, uint32_t b,
                                                       uint32_t *resultLow,
                                                       uint32_t *resultHigh) {
  const uint64_t product = static_cast<uint64_t>(a) * b;
  *resultLow = static_cast<uint32_t>(product);
  *resultHigh = static_cast<uint32_t>(product >> RIGHT_SHIFT);
}

__simt_callee__ __aicore__ inline void Philox4x32Round(uint32_t *counter,
                                                       const uint32_t *key) {
  uint32_t lo0;
  uint32_t hi0;
  MultiplyHighLow(PHILOX_M4X32_A, counter[0], &lo0, &hi0);

  uint32_t lo1;
  uint32_t hi1;
  MultiplyHighLow(PHILOX_M4X32_B, counter[IDX_2], &lo1, &hi1);

  uint32_t result[ALG_COUNTER_SIZE];
  result[0] = hi1 ^ counter[1] ^ key[0];
  result[1] = lo1;
  result[IDX_2] = hi0 ^ counter[IDX_3] ^ key[1];
  result[IDX_3] = lo0;

  CopyArray<ALG_COUNTER_SIZE>(counter, result);
}

__simt_callee__ __aicore__ inline void KeyInc(uint32_t *key) {
  key[0] += PHILOX_W32_A;
  key[1] += PHILOX_W32_B;
}

// The algorithm uses temporaries; the passed key/counter are not modified.
__simt_callee__ __aicore__ inline void PhiloxRandomSimt(const uint32_t *key,
                                                        const uint32_t *counter,
                                                        uint32_t *results) {
  uint32_t keyTmp[ALG_KEY_SIZE];
  uint32_t counterTmp[ALG_COUNTER_SIZE];
  CopyArray<ALG_KEY_SIZE>(keyTmp, key);
  CopyArray<ALG_COUNTER_SIZE>(counterTmp, counter);

  Philox4x32Round(counterTmp, keyTmp); // 1
  KeyInc(keyTmp);
  Philox4x32Round(counterTmp, keyTmp); // 2
  KeyInc(keyTmp);
  Philox4x32Round(counterTmp, keyTmp); // 3
  KeyInc(keyTmp);
  Philox4x32Round(counterTmp, keyTmp); // 4
  KeyInc(keyTmp);
  Philox4x32Round(counterTmp, keyTmp); // 5
  KeyInc(keyTmp);
  Philox4x32Round(counterTmp, keyTmp); // 6
  KeyInc(keyTmp);
  Philox4x32Round(counterTmp, keyTmp); // 7
  KeyInc(keyTmp);
  Philox4x32Round(counterTmp, keyTmp); // 8
  KeyInc(keyTmp);
  Philox4x32Round(counterTmp, keyTmp); // 9
  KeyInc(keyTmp);
  Philox4x32Round(counterTmp, keyTmp); // 10
  CopyArray<ALG_COUNTER_SIZE>(results, counterTmp);
}

// float overload: Philox bits -> uniform [0, 1).
__simt_callee__ __aicore__ inline void
PhiloxRandomSimt(const uint32_t *key, const uint32_t *counter, float *results) {
  uint32_t resultU32[ALG_COUNTER_SIZE];
  PhiloxRandomSimt(key, counter, resultU32);
#pragma unroll
  for (uint16_t i = 0; i < ALG_COUNTER_SIZE; i++) {
    results[i] = resultU32[i] * RAND_2POW32_INV + RAND_2POW32_INV_HALF;
  }
}

// Box-Muller transform: uniforms -> standard normal.
//   X = sqrt(-2 * ln(U1)) * cos(2 * PI * U2)
//   Y = sqrt(-2 * ln(U1)) * sin(2 * PI * U2)
__simt_callee__ __aicore__ inline void BoxMullerFloat(float u1, const float u2,
                                                      float *z0, float *z1) {
  const float eps = 1.0e-7f;
  if (u1 < eps) {
    u1 = eps;
  }
  float v = static_cast<float>(DOUBLE_MULTIPLE * PI * u2);
  float r = sqrtf(-DOUBLE_MULTIPLE * logf(u1));
  sincosf(v, z0, z1);
  *z0 *= r;
  *z1 *= r;
}

} // namespace RandomKernelBase
