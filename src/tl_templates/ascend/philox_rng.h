#pragma once

// TileLang stateful Philox RNG wrapper for the Ascend SIMT path.
//
// Thin wrapper over the vendored AscendC SIMT Philox primitives in
// tl_templates/ascend/random_kernel_base.h (namespace RandomKernelBase). The
// codegen (src/ascend/codegen/codegen_ascend.cc) emits calls to the tl::
// functions below for tl.rng_init / tl.rng_rand / tl.rng_rand_float.

#include "tl_templates/ascend/random_kernel_base.h"

namespace tl {

// One state per SIMT thread: rng_init seeds it, and each rng_rand() draw pulls
// one 32-bit word, regenerating a fresh 4-word Philox block (and advancing the
// counter) whenever the block is exhausted.
struct AscendPhiloxState {
  uint32_t key[RandomKernelBase::ALG_KEY_SIZE];
  uint32_t counter[RandomKernelBase::ALG_COUNTER_SIZE];
  uint32_t buf[RandomKernelBase::ALG_COUNTER_SIZE];
  uint32_t idx;       // 0..4; 4 == exhausted, regenerate on next draw
  float normal_cache; // second value produced by Box-Muller
  bool has_normal;
};

__simt_callee__ __aicore__ inline void
philox_init(AscendPhiloxState *st, int64_t seed, int64_t seq, int64_t off) {
  st->counter[0] = 0;
  st->counter[1] = 0;
  st->counter[RandomKernelBase::IDX_2] = 0;
  st->counter[RandomKernelBase::IDX_3] = 0;
  // key = seed; advance the low 64 bits of the counter by the 4-aligned offset.
  RandomKernelBase::PhiloxAlgParsInit(st->key, st->counter, seed, off);
  // Place the per-thread subsequence (seq) in the high 64 bits, exactly as the
  // SDK's FlashCounter folds a global thread index into the counter.
  RandomKernelBase::SkipHi(st->counter, static_cast<uint64_t>(seq));
  st->idx =
      RandomKernelBase::ALG_COUNTER_SIZE; // force generation on first draw
  st->normal_cache = 0.0f;
  st->has_normal = false;
}

__simt_callee__ __aicore__ inline uint32_t philox_rand(AscendPhiloxState *st) {
  if (st->idx >= RandomKernelBase::ALG_COUNTER_SIZE) {
    RandomKernelBase::PhiloxRandomSimt(st->key, st->counter, st->buf);
    RandomKernelBase::SkipOne(st->counter);
    st->idx = 0;
  }
  return st->buf[st->idx++];
}

__simt_callee__ __aicore__ inline float
philox_rand_uniform(AscendPhiloxState *st) {
  return philox_rand(st) * RandomKernelBase::RAND_2POW32_INV +
         RandomKernelBase::RAND_2POW32_INV_HALF;
}

// Keep two consecutive uniform draws in one target-runtime call when the loop
// vectorizer broadcasts a stateful scalar RNG call to a float2.
__simt_callee__ __aicore__ inline float2
philox_rand_uniform2(AscendPhiloxState *st) {
  const float uniform0 = philox_rand_uniform(st);
  const float uniform1 = philox_rand_uniform(st);
  return make_float2(uniform0, uniform1);
}

__simt_callee__ __aicore__ inline float4
philox_rand_uniform4(AscendPhiloxState *st) {
  const float2 uniform01 = philox_rand_uniform2(st);
  const float2 uniform23 = philox_rand_uniform2(st);
  return make_float4(uniform01.x, uniform01.y, uniform23.x, uniform23.y);
}

__simt_callee__ __aicore__ inline float2
philox_rand_normal2(AscendPhiloxState *st) {
  const float uniform0 = philox_rand_uniform(st);
  const float uniform1 = philox_rand_uniform(st);
  float normal0;
  float normal1;
  RandomKernelBase::BoxMullerFloat(uniform0, uniform1, &normal0, &normal1);
  return make_float2(normal0, normal1);
}

__simt_callee__ __aicore__ inline float4
philox_rand_normal4(AscendPhiloxState *st) {
  const float2 normal01 = philox_rand_normal2(st);
  const float2 normal23 = philox_rand_normal2(st);
  return make_float4(normal01.x, normal01.y, normal23.x, normal23.y);
}

__simt_callee__ __aicore__ inline float
philox_rand_normal(AscendPhiloxState *st) {
  if (st->has_normal) {
    st->has_normal = false;
    return st->normal_cache;
  }
  float u1 = philox_rand_uniform(st);
  float u2 = philox_rand_uniform(st);
  float z0;
  float z1;
  RandomKernelBase::BoxMullerFloat(u1, u2, &z0, &z1);
  st->normal_cache = z1;
  st->has_normal = true;
  return z0;
}

} // namespace tl
