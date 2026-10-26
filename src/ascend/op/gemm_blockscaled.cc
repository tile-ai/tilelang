#include "compat/analyzer.h"
/*!
 * \file tl/ascend/op/gemm_blockscaled.cc
 * \brief Ascend instruction selection for block-scaled GEMM.
 */

#include "op/gemm_blockscaled.h"

#include "ascend/op/utils.h"
#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

using namespace ffi;
using namespace tirx;

void ValidateL0SFGemm(const GemmBlockScaled &gemm, const L0SFBindings &bindings,
                      sym::Analyzer analyzer) {
  auto validate = [&](const TensorRegion &data, const TensorRegion &sf,
                      bool transpose, bool is_a) {
    if (IsL1Buffer(data->source.as_or_throw<TensorVar>())) {
      ICHECK(IsL1Buffer(sf->source.as_or_throw<TensorVar>()))
          << "L1 gemm_blockscaled requires L1 scale operands";
      return;
    }
    ICHECK(is_a ? IsL0ABuffer(data->source.as_or_throw<TensorVar>())
                : IsL0BBuffer(data->source.as_or_throw<TensorVar>()))
        << "gemm_blockscaled requires L0A/L0B data operands";
    ICHECK(is_a ? IsL0ASFBuffer(sf->source.as_or_throw<TensorVar>())
                : IsL0BSFBuffer(sf->source.as_or_throw<TensorVar>()))
        << "gemm_blockscaled requires the matching L0A/L0B SF scope for "
        << data->source.as_or_throw<TensorVar>()->name;
    auto bound = bindings.Get(sf->source.as_or_throw<TensorVar>());
    ICHECK(bound.has_value())
        << "L0 SF " << sf->source.as_or_throw<TensorVar>()->name
        << " has no allocation binding; use alloc_l0a_sf/alloc_l0b_sf";
    ICHECK(bound.value().same_as(data->source.as_or_throw<TensorVar>()))
        << "L0 SF " << sf->source.as_or_throw<TensorVar>()->name
        << " is bound to " << bound.value()->name
        << ", but gemm_blockscaled consumes "
        << data->source.as_or_throw<TensorVar>()->name;
    size_t rank = data->region.size();
    ICHECK_GE(rank, 2U);
    ICHECK_EQ(sf->region.size(), rank)
        << "L0 SF and data must have matching leading dimensions";
    for (size_t i = 0; i + 2 < rank; ++i) {
      ICHECK(tl::arith::CanProveEqual(analyzer, data->region[i]->begin,
                                      sf->region[i]->begin) &&
             tl::arith::CanProveEqual(analyzer, data->region[i]->extent, 1) &&
             tl::arith::CanProveEqual(analyzer, sf->region[i]->extent, 1))
          << "L0 SF " << sf->source.as_or_throw<TensorVar>()->name
          << " and data must select the same leading index on axis " << i;
    }
    for (size_t i = rank - 2; i < rank; ++i) {
      ICHECK(tl::arith::CanProveEqual(analyzer, sf->region[i]->begin, 0))
          << "Compact L0 SF regions require zero-origin trailing dimensions";
    }
    int bits = sf->source.as_or_throw<TensorVar>().dtype().bits();
    ICHECK((bits == 8 || bits == 16) &&
           sf->source.as_or_throw<TensorVar>().dtype().lanes() == 1)
        << "L0 SF storage must contain 8-bit scales or 16-bit scale pairs";
    PrimExpr mn = data->region[rank - (transpose ? 1 : 2)]->extent;
    PrimExpr k = data->region[rank - (transpose ? 2 : 1)]->extent;
    ICHECK(tl::arith::CanProve(analyzer, sf->region[rank - 2]->extent >= mn))
        << "L0 SF region must cover the data operand's MN extent";
    ICHECK(tl::arith::CanProveEqual(
        analyzer, sf->region[rank - 1]->extent * (4 * bits), k))
        << "L0 SF region must match the compact data K extent (one scale "
           "per 32 K elements); slice both operands consistently";
  };
  validate(gemm->aRegion_, gemm->sfaRegion_, gemm->transA_, true);
  validate(gemm->bRegion_, gemm->sfbRegion_, !gemm->transB_, false);
}

namespace ascend {
namespace {

// Instruction key resolved by the Python registry
// (tilelang/ascend/op/gemm/__init__.py) to GemmMADBlockScaled.
constexpr const char *kAscendMADBlockScaled = "ascend.mad.blockscaled";

String SelectBlockScaledGemmInst(const GemmBlockScaled &op, int block_size,
                                 const Target &target) {
  (void)op;
  (void)block_size;
  (void)target;
  return kAscendMADBlockScaled;
}

} // namespace
} // namespace ascend

namespace {

bool MatchAscendGemmBlockScaledTarget(Target target) {
  return TargetIsAscend(target);
}

bool RegisterAscendGemmBlockScaled() {
  RegisterGemmBlockScaledImpl(GemmBlockScaledImpl{
      "ascend.GemmBlockScaled",
      MatchAscendGemmBlockScaledTarget,
      ascend::SelectBlockScaledGemmInst,
  });
  return true;
}

const bool ascend_gemm_blockscaled_registered = RegisterAscendGemmBlockScaled();

} // namespace

} // namespace tl
} // namespace tvm
