/*!
 * \file layout/gemm_layouts.cc
 * \brief Define Layout used in MMA and other operations.
 *
 */

#include <tvm/tir/stmt_functor.h>

#include <cmath>

#include "layout.h"

namespace tvm {
namespace tl {

IterVar make_itervar(std::string name, PrimExpr dom) {
  Var var = Var(name, dom->dtype);
  return IterVar(Range(0, dom), var, IterVarType::kDataPar);
}

Fragment makeGemmFragment8x4() {
  IterVar i = make_itervar("i", 8);
  IterVar j = make_itervar("j", 4);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = FloorDiv(j->var, 1) + 4 * i;
  PrimExpr index = FloorMod(j->var, 1);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

Fragment makeGemmFragment8x8() {
  IterVar i = make_itervar("i", 8);
  IterVar j = make_itervar("j", 8);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = FloorDiv(j->var, 2) + 4 * i;
  PrimExpr index = FloorMod(j->var, 2);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

Fragment makeGemmFragment8x16() {
  IterVar i = make_itervar("i", 8);
  IterVar j = make_itervar("j", 16);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = FloorDiv(j->var, 4) + 4 * i;
  PrimExpr index = FloorMod(j->var, 4);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

Fragment makeGemmFragment8x8Transposed() {
  IterVar i = make_itervar("i", 8);
  IterVar j = make_itervar("j", 8);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = FloorDiv(i->var, 2) + 4 * j;
  PrimExpr index = FloorMod(i->var, 2);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

/*
From https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator
./matrix_calculator.py --architecture cdna1 --instruction v_mfma_f32_16x16x16f16
--detail-instruction
*/
Fragment makeGemmFragmentAB16x16CDNA(const int k_pack) {
  IterVar i = make_itervar("i", 16);
  IterVar j = make_itervar("j", 16 * k_pack);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = 16 * FloorDiv(j->var, 4 * k_pack) + i;
  PrimExpr index = FloorMod(j->var, 4 * k_pack);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

Fragment makeGemmFragmentAB16x16CDNATransposed(const int k_pack) {
  IterVar i = make_itervar("i", 16 * k_pack);
  IterVar j = make_itervar("j", 16);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = 16 * FloorDiv(i->var, 4 * k_pack) + j;
  PrimExpr index = FloorMod(i->var, 4 * k_pack);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

Fragment makeGemmFragmentAB16x32CDNA(const int k_pack) {
  IterVar i = make_itervar("i", 16);
  IterVar j = make_itervar("j", 32 * k_pack);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = 16 * FloorDiv(j->var, 8 * k_pack) + i;
  PrimExpr index = FloorMod(j->var, 8 * k_pack);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

Fragment makeGemmFragmentAB16x32CDNATransposed(const int k_pack) {
  IterVar i = make_itervar("i", 32 * k_pack);
  IterVar j = make_itervar("j", 16);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = 16 * FloorDiv(i->var, 8 * k_pack) + j;
  PrimExpr index = FloorMod(i->var, 8 * k_pack);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

Fragment makeGemmFragmentC16x16CDNA() {
  IterVar i = make_itervar("i", 16);
  IterVar j = make_itervar("j", 16);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = 16 * FloorDiv(j->var, 4) + i;
  PrimExpr index = FloorMod(j->var, 4);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

Fragment makeGemmFragmentC_F64(const int block_m, const int block_n,
                               const int warp_m, const int warp_n) {
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0);
  ICHECK(warp_n % 8 == 0);
  auto base_layout = makeGemmFragment8x8();
  auto warp_layout =
      base_layout->Repeat({block_m / warp_m, block_n / warp_n}, true, false);
  auto block_layout =
      warp_layout->Repeat({warp_m / 8, warp_n / 8}, false, false);
  return block_layout;
}

Fragment makeGemmFragmentC(const int block_m, const int block_n,
                           const int warp_m, const int warp_n,
                           const int element_size) {
  if (element_size == 64)
    return makeGemmFragmentC_F64(block_m, block_n, warp_m, warp_n);
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0) << "warp_m=" << warp_m;
  ICHECK(warp_n % 8 == 0) << "warp_n=" << warp_n;
  auto base_layout = makeGemmFragment8x8()->Repeat({2, 1}, false);
  auto warp_layout =
      base_layout->Repeat({block_m / warp_m, block_n / warp_n}, true, false);
  auto block_layout =
      warp_layout->Repeat({warp_m / 16, warp_n / 8}, false, false);
  return block_layout;
}

Fragment makeGemmSparseFragmentC(const int block_m, const int block_n,
                                 const int warp_m, const int warp_n,
                                 const int element_size) {
  if (element_size == 64) {
    ICHECK(false) << "Not supported";
  }
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0) << "warp_m=" << warp_m;
  ICHECK(warp_n % 8 == 0) << "warp_n=" << warp_n;
  auto base_layout = makeGemmFragment8x8()->Repeat({2, 1}, false);
  // NOTE: This func wasn't implemented by following the CUTLASS 2 iterator
  // but by inspecting the output, it appears that we first need to
  // repeat the warp layout while avoiding duplicate thread mappings.
  auto warp_layout =
      base_layout->Repeat({warp_m / 16, warp_n / 8}, false, false);
  auto block_layout =
      warp_layout->Repeat({block_m / warp_m, block_n / warp_n}, true, false);
  return block_layout;
}

Fragment makeGemmFragmentCCDNA(const int block_m, const int block_n,
                               const int warp_m, const int warp_n,
                               const int element_size) {
  if (element_size == 64)
    LOG(FATAL) << "Not supported";
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0) << "warp_m=" << warp_m;
  ICHECK(warp_n % 16 == 0) << "warp_n=" << warp_n;
  auto base_layout = makeGemmFragmentC16x16CDNA()->Repeat({1, 1}, false);
  auto warp_layout =
      base_layout->Repeat({warp_m / 16, warp_n / 16}, false, true);
  auto block_layout =
      warp_layout->Repeat({block_m / warp_m, block_n / warp_n}, true, false);
  return block_layout;
}

Fragment makeGemmFragmentCHopper(const int block_m, const int block_n,
                                 const int warp_m, const int warp_n,
                                 const int element_size) {
  ICHECK(block_m % warp_m == 0);
  ICHECK(warp_m % 16 == 0) << "warp_m=" << warp_m;

  auto warp_layout = makeGemmFragment8x8()->Repeat({2, warp_n / 8}, false,
                                                   false); // 16 x N (1 warp)
  auto block_layout = warp_layout->Repeat({block_m / warp_m, block_n / warp_n},
                                          true, false); // 16*Y x N (Y warp)
  return block_layout->Repeat({warp_m / 16, 1}, false, false);
}

Fragment makeGemmFragmentA(const int block_m, const int block_n,
                           const int block_k, const int warp_m,
                           const int warp_n, const int element_size,
                           bool transposed) {
  // assume not transposed
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0);
  ICHECK(block_k % 16 == 0);
  // Only support 8-bit and 16-bit
  ICHECK(element_size == 8 || element_size == 16 || element_size == 32)
      << "unsupported element bitwidth=" << element_size;

  if (transposed) {
    auto base_layout =
        makeGemmFragment8x8Transposed()->Repeat({2, 2}, false, true);
    auto warp_layout = base_layout->Repeat({1, block_m / warp_m}, true, false)
                           ->Replicate(block_n / warp_n);
    auto block_layout =
        warp_layout->Repeat({block_k / 16, warp_m / 16}, false, true);
    return block_layout;
  } else {
    if (element_size == 8) {
      auto base_layout = makeGemmFragment8x16()->Repeat({2, 2}, false, false);
      auto warp_layout = base_layout->Repeat({block_m / warp_m, 1}, true)
                             ->Replicate(block_n / warp_n);
      auto block_layout =
          warp_layout->Repeat({warp_m / 16, block_k / 32}, false, false);
      return block_layout;
    } else if (element_size == 16) {
      auto base_layout = makeGemmFragment8x8()->Repeat({2, 2}, false, false);
      auto warp_layout = base_layout->Repeat({block_m / warp_m, 1}, true)
                             ->Replicate(block_n / warp_n);
      auto block_layout =
          warp_layout->Repeat({warp_m / 16, block_k / 16}, false, false);
      return block_layout;
    } else if (element_size == 32) {
      auto base_layout = makeGemmFragment8x4()->Repeat({2, 2}, false, false);
      auto warp_layout = base_layout->Repeat({block_m / warp_m, 1}, true)
                             ->Replicate(block_n / warp_n);
      auto block_layout =
          warp_layout->Repeat({warp_m / 16, block_k / 8}, false, false);
      return block_layout;
    } else {
      ICHECK(0);
      return Fragment();
    }
  }
}

Fragment makeGemmFragmentB(const int block_m, const int block_n,
                           const int block_k, const int warp_m,
                           const int warp_n, bool transposed) {
  // transposed
  ICHECK(warp_n % 8 == 0);
  ICHECK(block_k % 16 == 0);
  if (transposed) {
    auto base_layout = makeGemmFragment8x8()->Repeat({1, 2}, false, false);
    auto warp_layout = base_layout->Replicate(block_m / warp_m)
                           ->Repeat({block_n / warp_n, 1}, true, false);
    auto block_layout =
        warp_layout->Repeat({warp_n / 8, block_k / 16}, false, false);
    return block_layout;
  } else {
    auto base_layout =
        makeGemmFragment8x8Transposed()->Repeat({2, 1}, false, false);
    auto warp_layout = base_layout->Replicate(block_m / warp_m)
                           ->Repeat({1, block_n / warp_n}, true);
    auto block_layout =
        warp_layout->Repeat({block_k / 16, warp_n / 8}, false, true);
    return block_layout;
  }
}

Fragment makeGemmFragmentACDNA(const int block_m, const int block_n,
                               const int block_k, const int warp_m,
                               const int warp_n, const int element_size,
                               const int k_pack, bool transposed) {
  // assume not transposed
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0);
  const int mfma_k = k_pack * (element_size == 16 ? 16 : 32);
  ICHECK(block_k % mfma_k == 0);
  ICHECK(element_size == 8 || element_size == 16)
      << "element bitwidth=" << element_size;
  if (transposed) {
    auto base_layout =
        element_size == 16
            ? makeGemmFragmentAB16x16CDNATransposed(k_pack)->Repeat(
                  {1, 1}, false, false)
            : makeGemmFragmentAB16x32CDNATransposed(k_pack)->Repeat(
                  {1, 1}, false, false);
    auto warp_layout =
        base_layout->Repeat({block_k / mfma_k, warp_m / 16}, false, true);
    auto block_layout = warp_layout->Repeat({1, block_m / warp_m}, true, true)
                            ->Replicate(block_n / warp_n);
    return block_layout;
  } else {
    auto base_layout =
        element_size == 16
            ? makeGemmFragmentAB16x16CDNA(k_pack)->Repeat({1, 1}, false, false)
            : makeGemmFragmentAB16x32CDNA(k_pack)->Repeat({1, 1}, false, false);
    auto warp_layout =
        base_layout->Repeat({warp_m / 16, block_k / mfma_k}, false, false);
    auto block_layout = warp_layout->Repeat({block_m / warp_m, 1}, true, true)
                            ->Replicate(block_n / warp_n);
    return block_layout;
  }
}

Fragment makeGemmFragment32x32(int element_size) {
  IterVar i = make_itervar("i", 32);
  IterVar j = make_itervar("j", 32);
  IterVar rep = make_itervar("rep", 1);
  ICHECK(element_size == 16 || element_size == 32);
  if (element_size == 16) {
    PrimExpr thd = FloorMod(i, 4) + FloorDiv(FloorMod(i, 16), 8) * 4 +
                   FloorDiv(FloorMod(j, 16), 8) * 8 + FloorDiv(i, 16) * 16;
    PrimExpr idx = FloorMod(j, 4) + FloorDiv(j, 16) * 4 +
                   FloorDiv(FloorMod(i, 8), 4) * 8 +
                   FloorDiv(FloorMod(j, 8), 4) * 16;
    return Fragment({i, j}, {idx}, thd, rep);
  } else {
    PrimExpr thd = FloorMod(i, 2) + 2 * FloorDiv(FloorMod(j, 4), 2) +
                   FloorDiv(FloorMod(i, 16), 8) * 4 +
                   FloorDiv(FloorMod(j, 16), 8) * 8 + FloorDiv(i, 16) * 16;
    PrimExpr idx = FloorMod(j, 2) + 2 * FloorDiv(FloorMod(i, 4), 2) +
                   FloorDiv(j, 16) * 4 + FloorDiv(FloorMod(i, 8), 4) * 8 +
                   FloorDiv(FloorMod(j, 8), 4) * 16;
    return Fragment({i, j}, {idx}, thd, rep);
  }
}

Fragment makeGemmVoltaFragmentC(const int block_m, const int block_n,
                                const int warp_m, const int warp_n,
                                int element_size) {
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 32 == 0);
  ICHECK(warp_n % 32 == 0);
  auto base_layout = makeGemmFragment32x32(element_size);
  auto warp_layout =
      base_layout->Repeat({warp_m / 32, warp_n / 32}, false, false);
  auto block_layout =
      warp_layout->Repeat({block_m / warp_m, block_n / warp_n}, true);
  return block_layout;
}

Fragment makeGemmVoltaFragmentA(const int block_m, const int block_n,
                                const int block_k, const int warp_m,
                                const int warp_n) {
  // assume not transposed
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 32 == 0);
  ICHECK(block_k % 4 == 0);
  // this is a special case
  IterVar i = make_itervar("i", 32);
  IterVar j = make_itervar("j", 4);
  IterVar rep = make_itervar("rep", 2);
  PrimExpr thd = FloorDiv(FloorMod(i, 16), 8) * 4 + 16 * FloorDiv(i, 16) +
                 FloorMod(i, 4) + 8 * rep;
  PrimExpr idx = j + FloorDiv(FloorMod(i, 8), 4) * 4;
  Fragment base_layout = Fragment({i, j}, {idx}, thd, rep);
  auto warp_layout =
      base_layout->Repeat({warp_m / 32, block_k / 4}, false, false);
  auto block_layout = warp_layout->Replicate(block_n / warp_n)
                          ->Repeat({block_m / warp_m, 1}, true);
  return block_layout;
}

PrimExpr xor2x2(const PrimExpr &i, const PrimExpr &j) {
  return FloorMod(i + j, 2);
}

PrimExpr xor4x4(const PrimExpr &i, const PrimExpr &j) {
  PrimExpr i0 = FloorMod(i, 2);
  PrimExpr j0 = FloorMod(j, 2);
  PrimExpr i1 = FloorDiv(i, 2);
  PrimExpr j1 = FloorDiv(j, 2);
  return 2 * xor2x2(i1, j1) + xor2x2(i0, j0);
}

PrimExpr xor8x8(const PrimExpr &i, const PrimExpr j) {
  PrimExpr i0 = FloorMod(i, 2);
  PrimExpr j0 = FloorMod(j, 2);
  PrimExpr i1 = FloorDiv(i, 2);
  PrimExpr j1 = FloorDiv(j, 2);
  return 2 * xor4x4(i1, j1) + xor2x2(i0, j0);
}

// Layout swizzling for 32 bytes
Layout makeQuarterBankSwizzleLayout(int stride, int continuous,
                                    int element_size) {
  // Swizzle 1 bit
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  int vector_size = 128 / element_size;
  ICHECK(stride % 8 == 0) << "stride=" << stride;
  ICHECK(continuous % (vector_size * 2) == 0)
      << "continuous=" << continuous << ", vector_size=" << vector_size;
  PrimExpr ts = FloorDiv(i, 8);
  PrimExpr s = FloorMod(i, 8);
  PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 2);
  PrimExpr c = FloorMod(FloorDiv(j, vector_size), 2);
  PrimExpr vec = FloorMod(j, vector_size);
  PrimExpr c_swizzle = xor2x2(c, FloorDiv(s, 4));
  PrimExpr index = vec + (c_swizzle + s * 2) * vector_size;
  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}

// Layout swizzling for 64 bytes
Layout makeHalfBankSwizzleLayout(int stride, int continuous, int element_size) {
  // Swizzle 2 bit
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  int vector_size = 128 / element_size;
  ICHECK(stride % 8 == 0) << "stride=" << stride;
  ICHECK(continuous % (vector_size * 4) == 0)
      << "continuous=" << continuous << ", vector_size=" << vector_size;
  PrimExpr ts = FloorDiv(i, 8);
  PrimExpr s = FloorMod(i, 8);
  PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 4);
  PrimExpr c = FloorMod(FloorDiv(j, vector_size), 4);
  PrimExpr vec = FloorMod(j, vector_size);
  PrimExpr c_swizzle = xor4x4(c, FloorDiv(s, 2));
  PrimExpr index = vec + (c_swizzle + s * 4) * vector_size;
  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}

// Layout swizzling for 128 bytes
Layout makeFullBankSwizzleLayout(int stride, int continuous, int element_size) {
  // Swizzle 3 bit
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  int vector_size = 128 / element_size;
  ICHECK(stride % 8 == 0) << "stride=" << stride;
  ICHECK(continuous % (vector_size * 8) == 0)
      << "continuous=" << continuous << ", vector_size=" << vector_size;
  PrimExpr ts = FloorDiv(i, 8);
  PrimExpr s = FloorMod(i, 8);
  PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 8);
  PrimExpr c = FloorMod(FloorDiv(j, vector_size), 8);
  PrimExpr vec = FloorMod(j, vector_size);
  PrimExpr c_swizzle = xor8x8(c, s);
  PrimExpr index = vec + (c_swizzle + s * 8) * vector_size;
  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}

// Detail implementation please ref to
// bitblas::tl::mfma_layout::make_mfma_swizzle_layout
Layout makeMatrixCoreSwizzleLayout(int stride, int continuous, int element_size,
                                   int kPack = 1) {
  const int numBanks = 32;
  const int bankBitWidth = 32;
  const int SIMDWidth = 16;
  const int vecSize = (64 / element_size) * kPack;
  const int innerDimLength = continuous;
  const int typeWidthInBit = element_size;

  const int elemsPerOneBanksRow = (numBanks * bankBitWidth) / typeWidthInBit;
  const int perPhase = std::max(1, elemsPerOneBanksRow / innerDimLength);
  const int maxPhase = std::min(SIMDWidth / perPhase, innerDimLength / vecSize);

  IterVar row = make_itervar("row", stride);
  IterVar col = make_itervar("col", continuous);
  PrimExpr phase = FloorMod(row / perPhase, maxPhase);
  PrimExpr colOffSwizzled = ((col / vecSize) ^ phase) * vecSize;
  PrimExpr colOffOrdered = FloorMod(col, vecSize);
  PrimExpr colOff = colOffSwizzled + colOffOrdered;

  return Layout(Array{row, col}, {row, colOff});
}

Layout makeGemmABLayoutF64_Kinner(int stride, int continuous) {
  // Swizzle<2, 0, 4>
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  PrimExpr tc = FloorDiv(j, 16);
  PrimExpr ts = FloorDiv(i, 4);
  PrimExpr c = FloorMod(j, 16);
  PrimExpr s = FloorMod(i, 4);
  PrimExpr swizzled_c = FloorDiv(c, 4) * 4 + xor4x4(FloorMod(c, 4), s);
  PrimExpr index = swizzled_c + s * 16;
  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}

Layout makeGemmABLayoutF64_Kouter(int stride, int continuous) {
  // Swizzle<2, 2, 2>
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  PrimExpr tc = FloorDiv(j, 16);
  PrimExpr ts = FloorDiv(i, 4);
  PrimExpr c = FloorMod(j, 16);
  PrimExpr s = FloorMod(i, 4);
  PrimExpr swizzled_c = FloorMod(c, 4) + xor4x4(FloorDiv(c, 4), s) * 4;
  PrimExpr index = swizzled_c + s * 16;
  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}

// The Default Layout for Tensor Access (row-major linear layout)
Layout makeLinearLayout(Array<PrimExpr> shape) {
  int ndim = static_cast<int>(shape.size());
  Array<IterVar> iter_vars;
  for (int i = 0; i < ndim; i++) {
    iter_vars.push_back(make_itervar(std::string{char('i' + i)}, shape[i]));
  }
  // Row-major: index = i0 * (d1 * d2 * ...) + i1 * (d2 * ...) + ... + i_{n-1}
  PrimExpr linear_index = 0;
  for (int i = 0; i < ndim; i++) {
    PrimExpr stride = 1;
    for (int j = i + 1; j < ndim; j++) {
      stride = stride * shape[j];
    }
    linear_index = linear_index + iter_vars[i]->var * stride;
  }
  return Layout(iter_vars, {linear_index});
}

Layout makeGemmABLayoutPadded(int stride, int continuous, int element_size) {
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  int padded = continuous;
  // Add 128 bits padding when the last dim is a multiple of 256 bits
  if ((element_size * continuous) % 256 == 0)
    padded += 128 / element_size;
  return Layout(Array{i, j}, {i * padded + j});
}

Layout MakeGemmVoltaABLayoutCrosswise(int stride, int continuous) {
  ICHECK(stride % 32 == 0 && continuous % 32 == 0);
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  PrimExpr vec_contiguous_idx = FloorDiv(j, 4);
  PrimExpr vec_strided_within_tile = FloorMod(vec_contiguous_idx, 8);

  PrimExpr bit2 =
      FloorMod(FloorDiv(FloorMod(i, 32), 16) + FloorDiv(FloorMod(i, 16), 8) +
                   FloorDiv(vec_strided_within_tile, 4),
               2);
  PrimExpr bit1 = xor2x2(FloorDiv(FloorMod(i, 8), 4),
                         FloorDiv(FloorMod(vec_strided_within_tile, 4), 2));
  PrimExpr permuted_vec_contiguous =
      FloorDiv(i, 16) * 16 + FloorMod(i, 4) * 4 + bit2 * 2 + bit1;

  PrimExpr offset = FloorMod(j, 4) + permuted_vec_contiguous * 4 +
                    vec_contiguous_idx * stride * 4;
  return Layout(Array{i, j}, {offset});
}

Layout MakeGemmVoltaALayoutCongruous(int stride, int continuous) {
  ICHECK(stride % 4 == 0 && continuous % 64 == 0);
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  PrimExpr vec_contiguous_idx = FloorDiv(j, 8);
  PrimExpr vec_strided_idx = i;
  PrimExpr tile_contiguous_idx = FloorDiv(vec_contiguous_idx, 8);
  PrimExpr tile_strided_idx = FloorDiv(vec_strided_idx, 4);
  PrimExpr tile_contiguous_residual = FloorMod(vec_contiguous_idx, 8);
  PrimExpr tile_strided_residual = FloorMod(vec_strided_idx, 4);

  PrimExpr permuted_strided_within_tile = FloorDiv(tile_contiguous_residual, 2);
  PrimExpr permuted_contiguous_within_tile =
      FloorMod(tile_contiguous_residual, 2) * 4 +
      xor4x4(tile_strided_residual, permuted_strided_within_tile);

  PrimExpr element_strided =
      permuted_strided_within_tile + tile_strided_idx * 4;
  PrimExpr element_contiguous =
      FloorMod(j, 8) +
      (permuted_contiguous_within_tile + tile_contiguous_idx * 8) * 8;
  PrimExpr offset = element_strided * continuous + element_contiguous;
  return Layout(Array{i, j}, {offset});
}

Layout MakeGemmVoltaBLayoutCongruous(int stride, int continuous) {
  ICHECK(stride % 4 == 0 && continuous % 64 == 0);
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  PrimExpr vec_contiguous_idx = FloorDiv(j, 8);
  PrimExpr vec_strided_idx = i;
  PrimExpr tile_contiguous_idx = FloorDiv(vec_contiguous_idx, 8);
  PrimExpr tile_strided_idx = FloorDiv(vec_strided_idx, 4);
  PrimExpr tile_contiguous_residual = FloorMod(vec_contiguous_idx, 8);
  PrimExpr tile_strided_residual = FloorMod(vec_strided_idx, 4);

  PrimExpr permuted_strided_within_tile = FloorMod(tile_contiguous_residual, 4);
  PrimExpr permuted_contiguous_within_tile =
      FloorDiv(tile_contiguous_residual, 4) * 4 +
      xor4x4(tile_strided_residual, permuted_strided_within_tile);

  PrimExpr element_strided =
      permuted_strided_within_tile + tile_strided_idx * 4;
  PrimExpr element_contiguous =
      FloorMod(j, 8) +
      (permuted_contiguous_within_tile + tile_contiguous_idx * 8) * 8;
  PrimExpr offset = element_strided * continuous + element_contiguous;
  return Layout(Array{i, j}, {offset});
}

Layout makeGemmVoltaABLayout(int stride, int continuous, bool is_a,
                             bool k_inner) {
  if (k_inner && continuous % 32 == 0 && stride % 32 == 0)
    return MakeGemmVoltaABLayoutCrosswise(stride, continuous);
  if (is_a && continuous % 64 == 0 && stride % 4 == 0)
    return MakeGemmVoltaALayoutCongruous(stride, continuous);
  if (!is_a && continuous % 64 == 0 && stride % 4 == 0)
    return MakeGemmVoltaBLayoutCongruous(stride, continuous);
  return makeGemmABLayoutPadded(stride, continuous, 16);
}

// ref:
// https://github.com/nvidia/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/include/cutlass/layout/tensor_op_multiplicand_sm75.h#L54
// Although the four settings (T or NT) used distinct layouts in CUTLASS, they
// appeared to result in the same mem layout
Layout makeTensorOpMultiplicand(int mat_stride, int mat_continuous,
                                int elementsize, int crosswise) {
  /// This layout is optimized for 128b accesses
  static int const kAccessSize = 128;
  int kCrosswise = crosswise;

  int kElementSize = elementsize;
  int kElementsPerAccess = kAccessSize / kElementSize;

  /// Contiguous dimension of the tile shape matches one shared memory cache
  /// line - 128B.  For 128bit access size, it equals to 8 accesses.
  int kTileShapeContiguous = 128 / (kAccessSize / 8);

  int kFactor = kTileShapeContiguous * kElementsPerAccess / kCrosswise;

  ICHECK(kFactor > 0)
      << "kCrosswise should be no large than one shared memory cache line.";

  /// The strided dimension needs to be at least (WarpSize(32) /
  /// kTileShapeContiguous) for a warp to access.  To ensure conflict free
  /// access, it also needs to be at least (kTileShapeContiguous / kFactor).
  /// See comments below
  /// Fundamental tile shape in units of vectors to guarantee bank conflict free
  /// shared memory load/store.
  /// For kFactor = 1, TileShape = <8, 8>
  /// For kFactor > 1, TileShape = <8, 4>
  int kTileShapeStride =
      ((kTileShapeContiguous / kFactor) > (32 / kTileShapeContiguous))
          ? (kTileShapeContiguous / kFactor)
          : (32 / kTileShapeContiguous);

  const int kPartitionShapeContiguous = 4;
  const int kPartitionShapeStride = 4;

  // NOTE: it's always row major for tl
  IterVar i = make_itervar("i", mat_stride);
  IterVar j = make_itervar("j", mat_continuous);

  PrimExpr vec_contiguous_idx = FloorDiv(j, kElementsPerAccess);
  PrimExpr vec_strided_idx = FloorDiv(i, kFactor);

  // Compute the fundamental tile being accessed
  PrimExpr tile_contiguous_idx =
      FloorDiv(vec_contiguous_idx, FloorDiv(kTileShapeContiguous, kFactor));

  PrimExpr tile_contiguous_residual =
      FloorMod(vec_contiguous_idx, FloorDiv(kTileShapeContiguous, kFactor)) +
      (FloorMod(i, kFactor) * FloorDiv(kTileShapeContiguous, kFactor));
  PrimExpr tile_strided_residual = FloorMod(vec_strided_idx, kTileShapeStride);

  // Compute the 'partition' within the fundamental tile
  PrimExpr partition_contiguous_idx =
      FloorDiv(tile_contiguous_residual, kPartitionShapeContiguous);
  PrimExpr partition_strided_idx =
      FloorDiv(tile_strided_residual, kPartitionShapeStride);

  PrimExpr partition_contiguous_residual =
      FloorMod(tile_contiguous_residual, kPartitionShapeContiguous);
  PrimExpr partition_strided_residual =
      FloorMod(tile_strided_residual, kPartitionShapeStride);

  //
  // Then swizzle
  //

  PrimExpr permuted_vec_contiguous_within_partition = xor4x4(
      partition_contiguous_residual, FloorMod(partition_strided_residual, 4));

  PrimExpr permuted_partition_contiguous_within_tile =
      xor2x2(partition_contiguous_idx, FloorMod(partition_strided_idx, 2));

  //
  // Compute final element location
  //

  PrimExpr element_contiguous =
      (tile_contiguous_idx * kTileShapeContiguous +
       permuted_partition_contiguous_within_tile * kPartitionShapeContiguous +
       permuted_vec_contiguous_within_partition) *
          kElementsPerAccess +
      FloorMod(j, kElementsPerAccess);

  const PrimExpr &element_strided = vec_strided_idx;

  const int stride = mat_continuous;

  return Layout(Array{i, j},
                {element_contiguous + element_strided * stride * kFactor});
}

Layout makeGemmSparseAmpereABLayout(int mat_stride, int mat_continuous,
                                    int elementsize) {
  int kCrosswise = std::min(mat_continuous, (1024 / elementsize));
  return makeTensorOpMultiplicand(mat_stride, mat_continuous, elementsize,
                                  kCrosswise);
}

/*!
 * \brief Creates a memory layout for GEMM's A or B matrices.
 *
 * This function selects an appropriate memory layout based on the matrix
 * dimensions, element size, continuity, and a k-factor. It aims to optimize
 * memory access patterns, potentially using swizzling techniques or specialized
 * layouts for different data types and hardware characteristics.
 *
 * \param mat_stride The leading dimension of the matrix (e.g., K for a
 * row-major M x K matrix). This is the number of elements to skip to get to the
 * same column in the next row (row-major) or to the same row in the next column
 * (column-major). \param mat_continuous The length of the dimension stored
 * contiguously in memory (e.g., K for a row-major M x K matrix, or M for a
 * column-major M x K matrix). \param continuity The size of the dimension that
 * is continuous from the perspective of memory bank access. This is used to
 * select specific swizzling strategies. It might be the same as mat_continuous
 *                   or different based on tiling or hardware details.
 * \param element_size The size of each element in the matrix, in bits (e.g., 8,
 * 16, 32, 64). \param k_inner Whether the K dimension is in the inner loop.
 * selection, particularly for fp64 and int8 types. It often relates to how the
 * K dimension of the GEMM (M x K * K x N) is handled or tiled.
 *                - For fp64 (element_size == 64):
 *                  - k_inner == false often implies K is in the "outer" loop
 * (e.g., KxN matrix).
 *                  - k_inner == true often implies K is in the "inner" loop
 * (e.g., NxK matrix).
 *                - For int8 (element_size == 8):
 *                  - k_inner == false uses a padded layout.
 * \return A Layout object representing the chosen memory layout.
 */
Layout makeGemmABLayout(int mat_stride, int mat_continuous, int continuity,
                        int element_size, bool k_inner) {
  if (element_size == 64) {
    if (!k_inner && continuity % 16 == 0) // float64 KxN
      return makeGemmABLayoutF64_Kouter(mat_stride, mat_continuous);
    if (k_inner && continuity % 16 == 0) // float64 NxK
      return makeGemmABLayoutF64_Kinner(mat_stride, mat_continuous);
    return makeGemmABLayoutPadded(mat_stride, mat_continuous, element_size);
  }
  int vector_size = 128 / element_size;
  if (!k_inner && element_size == 8) // int8 KxN
    return makeGemmABLayoutPadded(mat_stride, mat_continuous, element_size);
  else if (mat_continuous % (vector_size * 8) == 0)
    return makeFullBankSwizzleLayout(mat_stride, mat_continuous, element_size);
  else if (mat_continuous % (vector_size * 4) == 0)
    return makeHalfBankSwizzleLayout(mat_stride, mat_continuous, element_size);
  else {
    return makeGemmABLayoutPadded(mat_stride, mat_continuous, element_size);
  }
}

Layout makeGemmABLayoutHopper(int mat_stride, int mat_continuous,
                              int continuity, int element_size, bool k_inner) {
  if (element_size == 64) {
    if (!k_inner && continuity % 16 == 0) // float64 KxN
      return makeGemmABLayoutF64_Kouter(mat_stride, mat_continuous);
    if (k_inner && continuity % 16 == 0) // float64 NxK
      return makeGemmABLayoutF64_Kinner(mat_stride, mat_continuous);
    // fallback for float64 when stride % 8 != 0
    if (mat_stride % 8 != 0)
      return makeLinearLayout(
          Array<PrimExpr>{Integer(mat_stride), Integer(mat_continuous)});
    return makeQuarterBankSwizzleLayout(mat_stride, mat_continuous,
                                        element_size);
  }
  int vector_size = 128 / element_size;

  if (mat_stride % 8 == 0) {
    if (mat_continuous % (vector_size * 8) == 0)
      return makeFullBankSwizzleLayout(mat_stride, mat_continuous,
                                       element_size);
    else if (mat_continuous % (vector_size * 4) == 0)
      return makeHalfBankSwizzleLayout(mat_stride, mat_continuous,
                                       element_size);
    else if (mat_continuous % (vector_size * 2) == 0)
      return makeQuarterBankSwizzleLayout(mat_stride, mat_continuous,
                                          element_size);
  }

  if (mat_continuous % vector_size == 0)
    return makeLinearLayout(
        Array<PrimExpr>{Integer(mat_stride), Integer(mat_continuous)});
  else
    ICHECK(0) << "Unsupported layout for Hopper with stride=" << mat_stride
              << ", continuous=" << mat_continuous
              << ", element_size=" << element_size << ", k_inner=" << k_inner;
}

Layout makeGemmABLayoutSm100(int mat_stride, int mat_continuous, int continuity,
                             int element_size, bool k_inner) {
  if (element_size == 64) {
    ICHECK(0) << "float64 on sm100 is not supported now";
  }
  int vector_size = 128 / element_size;

  if (mat_stride % 8 == 0) {
    if (mat_continuous % (vector_size * 8) == 0)
      return makeFullBankSwizzleLayout(mat_stride, mat_continuous,
                                       element_size);
    else if (mat_continuous % (vector_size * 4) == 0)
      return makeHalfBankSwizzleLayout(mat_stride, mat_continuous,
                                       element_size);
    else if (mat_continuous % (vector_size * 2) == 0)
      return makeQuarterBankSwizzleLayout(mat_stride, mat_continuous,
                                          element_size);
  }

  if (mat_continuous % vector_size == 0)
    return makeLinearLayout(
        Array<PrimExpr>{Integer(mat_stride), Integer(mat_continuous)});
  else
    ICHECK(0) << "Unsupported layout for sm100 with stride=" << mat_stride
              << ", continuous=" << mat_continuous
              << ", element_size=" << element_size << ", k_inner=" << k_inner;
  __builtin_unreachable(); // to prevent compiler warning
}

Layout makeGemmABLayoutCDNA(int stride, int continuous, int element_size,
                            int kPack) {
  return makeMatrixCoreSwizzleLayout(stride, continuous, element_size, kPack);
}
} // namespace tl
} // namespace tvm
