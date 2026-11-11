/*!
 * \file tl/metal/op/fill.cc
 * \brief Metal implementation for tl.fill lowering.
 */

#include "op/fill.h"
#include <cstdint>
#include <tvm/runtime/logging.h>

#include "backend/common/target_utils.h"
#include "metal/op/builtin.h"
#include "metal/op/utils.h"
#include "metal/target_utils.h"
#include "op/utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

#include <tvm/tirx/builtin.h>

namespace tvm {
namespace tl {

using namespace tirx;

namespace metal {

namespace {

constexpr int64_t kSIMDGroupTileSize = 8;

Stmt LowerSIMDGroupPartialFill(const FillNode &op, const LowerArgs &lower_args,
                               arith::Analyzer *analyzer) {
  const SIMDGroupRegion tiles = GetSIMDGroupPartialRegion(
      op.dst, op.region, lower_args, analyzer, "Metal partial fill");

  Array<Stmt> stmts;
  for (int64_t i = 0; i < tiles.extent[0]; ++i) {
    for (int64_t j = 0; j < tiles.extent[1]; ++j) {
      // Row stride comes from the full buffer, not the selected region.
      const int64_t index = tiles.Index(i, j);
      stmts.push_back(Evaluate(
          Call(DataType::Handle(), builtin::make_filled_simdgroup_matrix(),
               {op.dst->data, IntImm(DataType::Int(32), index), op.value,
                IntImm(DataType::Int(32), kSIMDGroupTileSize),
                IntImm(DataType::Int(32), kSIMDGroupTileSize)})));
    }
  }
  if (stmts.size() == 1) {
    return stmts[0];
  }
  return SeqStmt(stmts);
}

// The existing cooperative-tensor lowering can fill tile zero directly.
bool IsCooperativeTensorPrefixTile(const FillNode &op,
                                   arith::Analyzer *analyzer) {
  if (!IsCooperativeTensorBuffer(op.dst) || !op.dst->strides.empty() ||
      !analyzer->CanProveEqual(op.dst->elem_offset, 0)) {
    return false;
  }
  PrimExpr elements = IntImm(DataType::Int(64), 1);
  bool full_suffix = true;
  for (size_t d = op.region.size(); d-- > 0;) {
    const Range &range = op.region[d];
    if (!analyzer->CanProveEqual(range->min, 0) ||
        !analyzer->CanProve(range->extent > 0) ||
        !analyzer->CanProve(range->extent <= op.dst->shape[d]) ||
        (!full_suffix && !analyzer->CanProveEqual(range->extent, 1))) {
      return false;
    }
    full_suffix &= analyzer->CanProveEqual(range->extent, op.dst->shape[d]);
    elements = elements * cast(DataType::Int(64), range->extent);
  }
  return analyzer->CanProveEqual(elements, 512);
}

} // namespace

struct Fill {
  static Stmt Lower(const FillNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    if ((IsSIMDGroupBuffer(op.dst) || IsCooperativeTensorBuffer(op.dst)) &&
        op.dst->shape.size() >= 3) {
      CHECK(IsWholeBufferRegion(op.dst, op.region, analyzer) ||
                IsCooperativeTensorPrefixTile(op, analyzer),
            ValueError)
          << "Metal partial fill: buffers with 3 or more dimensions are not "
             "supported for "
          << op.dst.scope() << " buffer " << op.dst->name
          << "; fill the whole buffer instead";
    }
    if (IsCooperativeTensorBuffer(op.dst)) {
      CHECK(IsWholeBufferRegion(op.dst, op.region, analyzer) ||
                IsCooperativeTensorPrefixTile(op, analyzer),
            ValueError)
          << "Metal partial fill: metal.cooperative_tensor supports only "
             "whole-buffer fills or a contiguous zero-origin prefix of exactly "
             "one 512-element tile; fill the whole buffer instead";
      int region_elements = 1;
      for (auto r : op.region) {
        auto imm = r->extent.as<IntImmNode>();
        TVM_FFI_ICHECK(imm)
            << "cooperative_tensor fill region must have constant extents";
        region_elements *= imm->value;
      }
      constexpr int kTileM = 16;
      constexpr int kTileN = 32;
      constexpr int kTileElems = kTileM * kTileN;
      TVM_FFI_ICHECK(region_elements % kTileElems == 0)
          << "cooperative_tensor buffer size must be multiple of " << kTileElems
          << ", got " << region_elements;
      int num_tiles = region_elements / kTileElems;
      PrimExpr fill_value = Cast(op.dst->dtype, op.value);
      Array<Stmt> stmts;
      for (int i = 0; i < num_tiles; i++) {
        stmts.push_back(
            Evaluate(Call(DataType::Handle(), cooperative_tensor_fill(),
                          {op.dst->data, IntImm(DataType::Int(32), i),
                           fill_value, IntImm(DataType::Int(32), kTileM),
                           IntImm(DataType::Int(32), kTileN)})));
      }
      if (stmts.size() == 1) {
        return stmts[0];
      }
      return SeqStmt(stmts);
    }

    if (IsSIMDGroupBuffer(op.dst)) {
      if ((op.dst->shape.size() == 1 || op.dst->shape.size() == 2) &&
          !IsWholeBufferRegion(op.dst, op.region, analyzer)) {
        return LowerSIMDGroupPartialFill(op, lower_args, analyzer);
      }
      int region_elements = 1;
      for (auto r : op.region) {
        auto imm = r->extent.as<IntImmNode>();
        TVM_FFI_ICHECK(imm)
            << "simdgroup fill region must have constant extents";
        region_elements *= imm->value;
      }
      TVM_FFI_ICHECK(region_elements % 64 == 0)
          << "simdgroup buffer size must be multiple of 64 (8x8), got "
          << region_elements;

      int num_matrices = region_elements / 64;
      PrimExpr fill_value = Cast(op.dst->dtype, op.value);
      Array<Stmt> stmts;
      for (int i = 0; i < num_matrices; i++) {
        stmts.push_back(Evaluate(Call(
            DataType::Handle(), builtin::make_filled_simdgroup_matrix(),
            {op.dst->data, IntImm(DataType::Int(32), i), fill_value,
             IntImm(DataType::Int(32), 8), IntImm(DataType::Int(32), 8)})));
      }
      if (stmts.size() == 1) {
        return stmts[0];
      }
      return SeqStmt(stmts);
    }

    if (IsFragmentBuffer(op.dst)) {
      auto par_op = ParallelOp(op.MakeSIMTLoop(analyzer));
      par_op->InferLayout({lower_args.target,
                           lower_args.thread_bounds,
                           lower_args.layout_map,
                           analyzer,
                           lower_args.buffer_remap,
                           {}},
                          InferLevel::kFree);
      auto thread_loop =
          PartitionLoop(par_op->GetRoot(), lower_args.thread_index, analyzer,
                        par_op->GetLoopLayout());
      auto vectorized_loop =
          VectorizeLoop(thread_loop, analyzer, lower_args.layout_map);
      auto unrolled_loop = PragmaUnrollLoop(vectorized_loop);

      if (par_op->GetPredicate(lower_args.thread_index).defined()) {
        return IfThenElse(par_op->GetPredicate(lower_args.thread_index).value(),
                          unrolled_loop);
      }
      return unrolled_loop;
    }

    if (IsLocalBuffer(op.dst) || IsLocalVarBuffer(op.dst)) {
      auto init_loop = op.MakeSIMTLoop(analyzer);
      auto vectorized_loop =
          VectorizeLoop(init_loop, analyzer, lower_args.layout_map);
      return PragmaUnrollLoop(vectorized_loop);
    }

    if (IsSharedBuffer(op.dst) || IsGlobalBuffer(op.dst)) {
      auto par_op = ParallelOp(op.MakeSIMTLoop(analyzer));
      par_op->InferLayout({lower_args.target,
                           lower_args.thread_bounds,
                           lower_args.layout_map,
                           analyzer,
                           lower_args.buffer_remap,
                           {}},
                          InferLevel::kFree);
      auto thread_loop =
          PartitionLoop(par_op->GetRoot(), lower_args.thread_index, analyzer,
                        par_op->GetLoopLayout());
      auto vectorized_loop =
          VectorizeLoop(thread_loop, analyzer, lower_args.layout_map);
      auto unrolled_loop = PragmaUnrollLoop(vectorized_loop);
      if (par_op->GetPredicate(lower_args.thread_index).defined()) {
        return IfThenElse(par_op->GetPredicate(lower_args.thread_index).value(),
                          unrolled_loop);
      }
      return unrolled_loop;
    }

    LOG(FATAL) << "Unsupported scope " << op.dst.scope();
    return Stmt();
  }
};

} // namespace metal

namespace {

bool MatchMetalFillTarget(Target target) { return TargetIsMetal(target); }

bool RegisterMetalFill() {
  RegisterFillImpl(FillImpl{
      "metal.Fill",
      MatchMetalFillTarget,
      metal::Fill::Lower,
  });
  return true;
}

const bool metal_fill_registered = RegisterMetalFill();

} // namespace

} // namespace tl
} // namespace tvm
