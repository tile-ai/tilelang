/*!
 * \file tl/cuda/op/fill.cc
 * \brief CUDA implementation for tl.fill lowering.
 */

#include "backend/common/op/fill.h"

#include "backend/common/target_utils.h"
#include "op/builtin.h"
#include "op/utils.h"
#include <tvm/arith/analyzer.h>
#include <tvm/tirx/op.h>

#include <optional>

namespace tvm {
namespace tl {

namespace {

using namespace tirx;

bool CanUseStBulkShared(Target target) {
  return TargetIsCuda(target) && !TargetIsCuTeDSL(target) &&
         TargetHasSMVersionGE(target, 100);
}

bool MatchCudaFillTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool IsZeroFillValue(const PrimExpr &value, arith::Analyzer *analyzer) {
  if (const auto *broadcast = value.as<BroadcastNode>()) {
    return IsZeroFillValue(broadcast->value, analyzer);
  }
  return analyzer->CanProveEqual(value, make_zero(value.dtype()));
}

std::optional<PrimExpr> FullRegionElements(const Buffer &buffer,
                                           const Array<Range> &region,
                                           arith::Analyzer *analyzer) {
  ICHECK_EQ(buffer->shape.size(), region.size());
  PrimExpr elements = make_const(DataType::Int(64), 1);
  for (size_t i = 0; i < region.size(); ++i) {
    if (!analyzer->CanProveEqual(region[i]->min, 0) ||
        !analyzer->CanProveEqual(region[i]->extent, buffer->shape[i])) {
      return std::nullopt;
    }
    elements = elements * cast(DataType::Int(64), region[i]->extent);
  }
  return elements;
}

std::optional<Stmt> TryLowerSharedBulkZeroFill(const FillNode &op,
                                               const LowerArgs &lower_args,
                                               arith::Analyzer *analyzer) {
  if (!CanUseStBulkShared(lower_args.target)) {
    return std::nullopt;
  }
  if (!IsSharedBuffer(op.dst) || !IsZeroFillValue(op.value, analyzer)) {
    return std::nullopt;
  }
  if (!op.dst->strides.empty()) {
    return std::nullopt;
  }

  auto elements = FullRegionElements(op.dst, op.region, analyzer);
  if (!elements.has_value()) {
    return std::nullopt;
  }

  int bits = op.dst->dtype.bits() * op.dst->dtype.lanes();
  if (bits % 8 != 0) {
    return std::nullopt;
  }
  PrimExpr bytes = analyzer->Simplify(elements.value() *
                                      IntImm(DataType::Int(64), bits / 8));
  auto bytes_imm = bytes.as<IntImmNode>();
  if (bytes_imm == nullptr || bytes_imm->value % 8 != 0) {
    return std::nullopt;
  }

  PrimExpr bulk_store =
      Call(DataType::Handle(), ptx_st_bulk_shared(),
           {op.dst->data, bytes, make_const(DataType::UInt(64), 0)});
  Stmt body = Evaluate(bulk_store);
  if (lower_args.thread_index.defined() && lower_args.thread_bounds.defined()) {
    body = IfThenElse(
        EQ(lower_args.thread_index, lower_args.thread_bounds->min), body);
  }
  return body;
}

Stmt LowerCudaFill(const FillNode &op, const LowerArgs &lower_args,
                   arith::Analyzer *analyzer) {
  if (auto bulk_fill = TryLowerSharedBulkZeroFill(op, lower_args, analyzer)) {
    return bulk_fill.value();
  }
  return backend::Fill::Lower(op, lower_args, analyzer);
}

bool RegisterCudaFill() {
  RegisterFillImpl(FillImpl{
      "cuda.Fill",
      MatchCudaFillTarget,
      LowerCudaFill,
  });
  return true;
}

const bool cuda_fill_registered = RegisterCudaFill();

} // namespace

} // namespace tl
} // namespace tvm
