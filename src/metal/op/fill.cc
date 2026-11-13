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

// Destination tensors use the same 16x32 microtiles and lane-local element
// order as cooperative_tensor_store. A flat buffer is a sequence of these
// tiles; a 2D buffer uses its full column count, never the region width.
Stmt LowerCooperativeTensorPartialFill(const FillNode &op,
                                       const LowerArgs &args,
                                       arith::Analyzer *analyzer) {
  const size_t ndim = op.dst->shape.size();
  CHECK(op.dst->strides.empty() &&
            analyzer->CanProveEqual(op.dst->elem_offset, 0),
        ValueError)
      << "Metal partial fill: requires a compact cooperative_tensor buffer "
         "with zero elem_offset";
  CheckSIMDGroupExecution(args, analyzer, "Metal partial fill");
  int64_t shape[2] = {1, 1}, begin[2] = {0, 0}, end[2] = {1, 1};
  for (size_t d = 0; d < ndim; ++d) {
    const auto *size = op.dst->shape[d].as<IntImmNode>();
    const auto *min = op.region[d]->min.as<IntImmNode>();
    const auto *extent = op.region[d]->extent.as<IntImmNode>();
    CHECK(size && min && extent, ValueError)
        << "Metal partial fill: cooperative_tensor requires constant shape, "
           "min and extent";
    CHECK(size->value > 0 && min->value >= 0 && extent->value > 0 &&
              extent->value <= size->value &&
              min->value <= size->value - extent->value,
          ValueError)
        << "Metal partial fill: cooperative_tensor region exceeds buffer "
           "bounds";
    shape[d] = size->value;
    begin[d] = min->value;
    end[d] = min->value + extent->value;
  }
  CHECK((ndim == 1 && shape[0] % 512 == 0) ||
            (ndim == 2 && shape[0] % 16 == 0 && shape[1] % 32 == 0),
        ValueError)
      << "Metal partial fill: cooperative_tensor requires a buffer size "
         "divisible by 512 in 1D or a 16x32-aligned buffer shape in 2D";
  const int64_t rows = ndim == 1 ? shape[0] / 32 : shape[0];
  const int64_t cols = ndim == 1 ? 32 : shape[1];
  CHECK(rows / 16 <= std::numeric_limits<int32_t>::max() / (cols / 32),
        ValueError)
      << "Metal partial fill: cooperative_tensor tile count "
         "exceeds the int32 intrinsic range";
  CHECK(rows <= std::numeric_limits<int32_t>::max() / cols, ValueError)
      << "Metal partial fill: cooperative_tensor logical indices exceed int32";
  PrimExpr lane = args.thread_index;
  PrimExpr base_row = FloorDiv(lane, 16) * 4 + FloorMod(FloorDiv(lane, 2), 4);
  PrimExpr base_col =
      (FloorMod(FloorDiv(lane, 8), 2) * 2 + FloorMod(lane, 2)) * 4;
  PrimExpr value = Cast(op.dst->dtype, op.value);
  auto constant = [](int64_t value) {
    return IntImm(DataType::Int(32), value);
  };
  Array<Stmt> stmts;
  for (int64_t row = 0; row < rows; row += 16) {
    for (int64_t col = 0; col < cols; col += 32) {
      int64_t first = row * cols + col;
      int64_t last = (row + 15) * cols + col + 31;
      bool outside = ndim == 1 ? last < begin[0] || first >= end[0]
                               : row + 16 <= begin[0] || row >= end[0] ||
                                     col + 32 <= begin[1] || col >= end[1];
      if (outside)
        continue;
      int64_t tile = (row / 16) * (cols / 32) + col / 32;
      bool whole_tile = ndim == 1 ? begin[0] <= first && last < end[0]
                                  : begin[0] <= row && row + 16 <= end[0] &&
                                        begin[1] <= col && col + 32 <= end[1];
      if (whole_tile) {
        stmts.push_back(
            Evaluate(Call(DataType::Handle(), cooperative_tensor_fill(),
                          {op.dst->data, constant(tile), value, constant(16),
                           constant(32)})));
        continue;
      }
      for (int64_t element = 0; element < 16; ++element) {
        PrimExpr r = constant(row) + base_row + constant((element % 8) / 4 * 8);
        PrimExpr c =
            constant(col) + base_col + constant(element / 8 * 16 + element % 4);
        PrimExpr index = r * constant(cols) + c;
        PrimExpr predicate =
            ndim == 1 ? index >= constant(begin[0]) && index < constant(end[0])
                      : r >= constant(begin[0]) && r < constant(end[0]) &&
                            c >= constant(begin[1]) && c < constant(end[1]);
        stmts.push_back(IfThenElse(
            analyzer->Simplify(predicate),
            Evaluate(Call(
                DataType::Handle(), cooperative_tensor_set_element(),
                {op.dst->data, constant(tile), constant(element), value}))));
      }
    }
  }
  return stmts.size() == 1 ? stmts[0] : SeqStmt(stmts);
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
      if (op.dst->shape.size() <= 2 &&
          !IsWholeBufferRegion(op.dst, op.region, analyzer)) {
        bool legacy_prefix =
            op.dst->shape.size() == 2 &&
            IsCooperativeTensorPrefixTile(op, analyzer) &&
            !(analyzer->CanProveEqual(FloorMod(op.dst->shape[0], 16), 0) &&
              analyzer->CanProveEqual(FloorMod(op.dst->shape[1], 32), 0));
        if (!legacy_prefix) {
          return LowerCooperativeTensorPartialFill(op, lower_args, analyzer);
        }
        // Preserve flattened tile-zero prefixes for legacy non-matrix shapes.
        // Canonical matrix shapes must always use their actual coordinates.
        CheckSIMDGroupExecution(lower_args, analyzer, "Metal partial fill");
      }
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
