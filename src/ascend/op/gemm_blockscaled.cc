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
                      arith::Analyzer *analyzer) {
  auto validate = [&](const BufferRegion &data, const BufferRegion &sf,
                      bool transpose, bool is_a) {
    if (IsL1Buffer(data->buffer)) {
      ICHECK(IsL1Buffer(sf->buffer))
          << "L1 gemm_blockscaled requires L1 scale operands";
      return;
    }
    ICHECK(is_a ? IsL0ABuffer(data->buffer) : IsL0BBuffer(data->buffer))
        << "gemm_blockscaled requires L0A/L0B data operands";
    ICHECK(is_a ? IsL0ASFBuffer(sf->buffer) : IsL0BSFBuffer(sf->buffer))
        << "gemm_blockscaled requires the matching L0A/L0B SF scope for "
        << data->buffer->name;
    auto bound = bindings.Get(sf->buffer->data);
    ICHECK(bound.has_value())
        << "L0 SF " << sf->buffer->name
        << " has no allocation binding; use alloc_l0a_sf/alloc_l0b_sf";
    ICHECK(bound.value().same_as(data->buffer->data))
        << "L0 SF " << sf->buffer->name << " is bound to "
        << bound.value()->name_hint << ", but gemm_blockscaled consumes "
        << data->buffer->name;
    size_t rank = data->region.size();
    ICHECK_GE(rank, 2U);
    ICHECK_EQ(sf->region.size(), rank)
        << "L0 SF and data must have matching leading dimensions";
    for (size_t i = 0; i + 2 < rank; ++i) {
      ICHECK(
          analyzer->CanProveEqual(data->region[i]->min, sf->region[i]->min) &&
          analyzer->CanProveEqual(data->region[i]->extent, 1) &&
          analyzer->CanProveEqual(sf->region[i]->extent, 1))
          << "L0 SF " << sf->buffer->name
          << " and data must select the same leading index on axis " << i;
    }
    for (size_t i = rank - 2; i < rank; ++i) {
      ICHECK(analyzer->CanProveEqual(sf->region[i]->min, 0))
          << "Compact L0 SF regions require zero-origin trailing dimensions";
    }
    int bits = sf->buffer->dtype.bits();
    ICHECK((bits == 8 || bits == 16) && sf->buffer->dtype.lanes() == 1)
        << "L0 SF storage must contain 8-bit scales or 16-bit scale pairs";
    PrimExpr mn = data->region[rank - (transpose ? 1 : 2)]->extent;
    PrimExpr k = data->region[rank - (transpose ? 2 : 1)]->extent;
    ICHECK(analyzer->CanProve(sf->region[rank - 2]->extent >= mn))
        << "L0 SF region must cover the data operand's MN extent";
    ICHECK(
        analyzer->CanProveEqual(sf->region[rank - 1]->extent * (4 * bits), k))
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
