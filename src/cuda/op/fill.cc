/*!
 * \file tl/cuda/op/fill.cc
 * \brief CUDA implementation for tl.fill lowering.
 */

#include "backend/common/op/fill.h"

#include "backend/common/target_utils.h"
#include "cuda/op/builtin.h"
#include "op/utils.h"
#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>

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

bool IsZeroFillValue(const PrimExpr &value, sym::Analyzer analyzer) {
  if (const auto *broadcast = value.as<BroadcastNode>()) {
    return IsZeroFillValue(broadcast->value, analyzer);
  }
  return IsZeroBitPattern(analyzer->Simplify(value));
}

std::optional<PrimExpr> FullRegionElements(const TensorVar &buffer,
                                           const Array<Range> &region,
                                           sym::Analyzer analyzer) {
  ICHECK_EQ(buffer.shape().size(), region.size());
  PrimExpr elements = MakeConst(PrimType::Int(64), 1);
  for (size_t i = 0; i < region.size(); ++i) {
    if (!analyzer->CanProveEqual(region[i]->begin, 0) ||
        !analyzer->CanProveEqual(region[i]->extent, buffer.shape()[i])) {
      return std::nullopt;
    }
    elements = elements * Cast(PrimType::Int(64), region[i]->extent);
  }
  return elements;
}

std::optional<Stmt> TryLowerSharedBulkZeroFill(const FillNode &op,
                                               const LowerArgs &lower_args,
                                               sym::Analyzer analyzer) {
  if (!CanUseStBulkShared(lower_args.target)) {
    return std::nullopt;
  }
  if (!IsSharedBuffer(op.dst) || !IsZeroFillValue(op.value, analyzer)) {
    return std::nullopt;
  }
  if (!op.dst.ty()->HasDefaultLayout(analyzer) ||
      op.dst.ty()->data_alignment < 8) {
    return std::nullopt;
  }

  auto elements = FullRegionElements(op.dst, op.region, analyzer);
  if (!elements.has_value()) {
    return std::nullopt;
  }

  int bits = op.dst.dtype().bits() * op.dst.dtype().lanes();
  if (bits % 8 != 0) {
    return std::nullopt;
  }
  PrimExpr bytes = analyzer->Simplify(elements.value() *
                                      IntImm(PrimType::Int(64), bits / 8));
  auto bytes_imm = bytes.as<IntImmNode>();
  if (bytes_imm == nullptr || bytes_imm->value % 8 != 0) {
    return std::nullopt;
  }

  // Emit the destination as a write access_ptr so ThreadSync records the
  // st.bulk as a shared write and inserts the required barrier.
  Expr dst_access_ptr =
      MakeAccessPtrFromRegion(NormalizeToBufferRegion(op.dst), kAccessWrite);
  Expr bulk_store =
      Call(TupleType(ffi::Array<Type>{}), ptx_st_bulk_shared(),
           {dst_access_ptr, bytes, MakeConst(PrimType::UInt(64), 0)});
  Stmt body = Evaluate(bulk_store);
  body = If(EQ(lower_args.thread_index, lower_args.thread_bounds->begin),
            SeqStmt(body));
  return body;
}

Stmt LowerCudaFill(const FillNode &op, const LowerArgs &lower_args,
                   sym::Analyzer analyzer) {
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
