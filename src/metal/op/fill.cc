/*!
 * \file tl/metal/op/fill.cc
 * \brief Metal implementation for tl.fill lowering.
 */

#include "op/fill.h"
#include <cstdint>
#include <limits>
#include <tvm/runtime/logging.h>

#include "backend/common/target_utils.h"
#include "metal/op/builtin.h"
#include "metal/op/utils.h"
#include "metal/target_utils.h"
#include "op/utils.h"
#include "support/check.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

#include <tvm/tirx/builtin.h>

namespace tvm {
namespace tl {

using namespace tirx;

namespace metal {

namespace {

constexpr int64_t kSIMDGroupTileSize = 8;

bool IsWholeFill(const Buffer &buffer, const Array<Range> &region,
                 arith::Analyzer *analyzer) {
  for (size_t i = 0; i < region.size(); ++i) {
    if (!analyzer->CanProveEqual(region[i]->min, 0) ||
        !analyzer->CanProveEqual(region[i]->extent, buffer->shape[i])) {
      return false;
    }
  }
  return true;
}

Stmt LowerSIMDGroupPartialFill(const FillNode &op, const LowerArgs &lower_args,
                               arith::Analyzer *analyzer) {
  CHECK(op.dst->strides.empty(), ValueError)
      << "Metal partial fill: non-default stride is not supported for "
      << op.dst->name << ":" << op.dst->strides;
  CHECK(analyzer->CanProveEqual(op.dst->elem_offset, 0), ValueError)
      << "Metal partial fill: elem_offset must be provably zero for "
      << op.dst->name << ":" << op.dst->elem_offset;

  const int warp_size = TargetMetalGetWarpSize(lower_args.target);
  CHECK(analyzer->CanProveEqual(lower_args.block_num_threads, warp_size),
        ValueError)
      << "Metal partial fill: requires exactly one simdgroup (" << warp_size
      << " threads) in the kernel launch, got " << lower_args.block_num_threads
      << " for buffer " << op.dst->name;
  CHECK(
      analyzer->CanProveEqual(lower_args.thread_bounds->min, 0) &&
          analyzer->CanProveEqual(lower_args.thread_bounds->extent, warp_size),
      ValueError)
      << "Metal partial fill: requires a complete simdgroup execution range "
         "[0, "
      << warp_size << "), got " << lower_args.thread_bounds << " for buffer "
      << op.dst->name;

  int64_t tile_min[2];
  int64_t tile_extent[2];

  for (size_t d = 0; d < 2; ++d) {
    const auto *shape_imm = op.dst->shape[d].as<IntImmNode>();
    const auto *min_imm = op.region[d]->min.as<IntImmNode>();
    const auto *extent_imm = op.region[d]->extent.as<IntImmNode>();

    CHECK(shape_imm && min_imm && extent_imm, ValueError)
        << "Metal partial fill: requires constant shape, min and extent at "
        << "dimension " << d << " of " << op.dst->name
        << "; shape=" << op.dst->shape << ", region=" << op.region;

    const int64_t shape = shape_imm->value;
    const int64_t min = min_imm->value;
    const int64_t extent = extent_imm->value;
    CHECK(extent > 0, ValueError)
        << "Metal partial fill: requires positive extent at dimension " << d
        << " of " << op.dst->name << ", got " << extent;
    // Fill construction has checked constant region minima and bounds.
    CHECK(shape % kSIMDGroupTileSize == 0 && min % kSIMDGroupTileSize == 0 &&
              extent % kSIMDGroupTileSize == 0,
          ValueError)
        << "Metal partial fill: requires 8-aligned shape, min and extent at "
        << "dimension " << d << " of " << op.dst->name
        << "; shape=" << op.dst->shape << ", region=" << op.region;

    tile_min[d] = min / kSIMDGroupTileSize;
    tile_extent[d] = extent / kSIMDGroupTileSize;
  }
  const int64_t col_tiles =
      op.dst->shape[1].as<IntImmNode>()->value / kSIMDGroupTileSize;

  const int64_t last_row = tile_min[0] + tile_extent[0] - 1;
  const int64_t last_col = tile_min[1] + tile_extent[1] - 1;
  constexpr int64_t kMaxMatrixIndex = std::numeric_limits<int32_t>::max();
  CHECK(last_col <= kMaxMatrixIndex &&
            last_row <= (kMaxMatrixIndex - last_col) / col_tiles,
        ValueError)
      << "Metal partial fill: tile index exceeds the int32 intrinsic range for "
      << op.dst->name << "; shape=" << op.dst->shape
      << ", region=" << op.region;

  Array<Stmt> stmts;
  for (int64_t i = 0; i < tile_extent[0]; ++i) {
    for (int64_t j = 0; j < tile_extent[1]; ++j) {
      // Row stride comes from the full buffer, not the selected region.
      const int64_t index = (tile_min[0] + i) * col_tiles + tile_min[1] + j;
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

} // namespace

struct Fill {
  static Stmt Lower(const FillNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    if (IsCooperativeTensorBuffer(op.dst)) {
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
      if (op.dst->shape.size() == 2 &&
          !IsWholeFill(op.dst, op.region, analyzer)) {
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
