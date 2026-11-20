/*!
 * \file tl/cuda/op/transpose.cc
 * \brief CUDA implementation for tl.transpose lowering.
 */

#include "backend/common/op/transpose.h"

#include "backend/common/target_utils.h"
#include "cuda/op/builtin.h"
#include "cuda/target_utils.h"
#include "op/builtin.h"
#include "transform/loop_partition.h"

#include <tvm/tirx/builtin.h>

namespace tvm {
namespace tl {

namespace {

bool MatchCudaTransposeTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

// Element offset of a logical coordinate from the start of the shared
// allocation, including the buffer's own element offset.
PrimExpr PhysicalElementOffset(const Buffer &buffer,
                               const Array<PrimExpr> &indices,
                               const LayoutMap &layout_map) {
  Array<PrimExpr> coords = indices;
  Array<PrimExpr> shape = buffer->shape;
  if (layout_map.count(buffer)) {
    Layout layout = ExpandLayoutToMatchBuffer(layout_map[buffer], buffer);
    coords = layout->Forward(indices);
    shape = layout->OutputShape();
  }
  PrimExpr offset = cast(coords[0].dtype(), buffer->elem_offset);
  PrimExpr stride = make_const(coords[0].dtype(), 1);
  for (int i = static_cast<int>(coords.size()) - 1; i >= 0; --i) {
    offset = offset + coords[i] * stride;
    stride = stride * shape[i];
  }
  return offset;
}

Array<PrimExpr> RegionIndices(const Array<Range> &range, const PrimExpr &row,
                              const PrimExpr &col) {
  Array<PrimExpr> indices;
  size_t rank = range.size();
  for (size_t i = 0; i + 2 < rank; ++i)
    indices.push_back(range[i]->min);
  indices.push_back(range[rank - 2]->min + row);
  indices.push_back(range[rank - 1]->min + col);
  return indices;
}

// Every `width`-element row segment of the region must be one 16-byte-aligned
// run in the physical allocation, which is what ldmatrix and stmatrix address.
bool HasAlignedRowSegments(const Buffer &buffer, const Array<Range> &range,
                           int64_t rows, int64_t cols, int width,
                           const LayoutMap &layout_map,
                           arith::Analyzer *analyzer) {
  if (!buffer->strides.empty())
    return false;
  Var row("transpose_row"), segment("transpose_segment"),
      elem("transpose_elem");
  analyzer->Bind(row, Range(0, IntImm(row.dtype(), rows)));
  analyzer->Bind(segment, Range(0, IntImm(segment.dtype(), cols / width)));
  PrimExpr offset = PhysicalElementOffset(
      buffer, RegionIndices(range, row, segment * width + elem), layout_map);
  return IndicesCanVectorize(offset, elem, IntImm(elem.dtype(), width), width,
                             analyzer, /*allow_broadcast=*/false);
}

PrimExpr SharedAccessPtr(const Buffer &buffer, const Array<PrimExpr> &indices,
                         int extent, int rw_mask) {
  return Call(DataType::Handle(), tl::access_ptr(),
              {BufferLoad(buffer, indices), IntImm(DataType::Int(32), extent),
               IntImm(DataType::Int(32), rw_mask)});
}

// Transposes a shared-memory tile of 8- or 16-bit elements with warp-wide
// ldmatrix/stmatrix. Each lane supplies one 16-byte row address per matrix,
// so the matrices of one instruction may come from arbitrary tiles.
//
// 16-bit: an 8x8 block is loaded as one matrix and stored with
// stmatrix.trans to the mirrored block.
//
// 8-bit: a 16x16 byte block is loaded as two matrices with ldmatrix.trans,
// whose rows are source rows {0,1,4,5,...,13} and {2,3,6,7,...,15}. Each lane
// then holds a 2x2 byte block from both matrices, and byte permutes 0x6420 and
// 0x7531 assemble four consecutive source rows of one source column, which
// stmatrix writes as one destination row.
//
// Returns an undefined Stmt when the region does not satisfy these
// requirements; the caller then uses the generic SIMT lowering.
Stmt LowerWarpMatrixTranspose(const TransposeNode &op,
                              const LowerArgs &lower_args,
                              arith::Analyzer *analyzer) {
  const Buffer &src = op.src;
  const Buffer &dst = op.dst;
  if (!TargetHasStmatrix(lower_args.target) ||
      TargetIsCuTeDSL(lower_args.target))
    return Stmt();
  // A view made from an address can start partway into an allocation, even
  // when its own element offset is zero.
  if (!lower_args.allocated_shared_buffers.count(src->data) ||
      !lower_args.allocated_shared_buffers.count(dst->data) ||
      src->data.same_as(dst->data))
    return Stmt();
  if (src->dtype != dst->dtype || src->dtype.lanes() != 1)
    return Stmt();
  const int bits = src->dtype.bits();
  if (bits != 8 && bits != 16)
    return Stmt();

  // Both regions must be a single matrix: every axis before the final two
  // has extent 1 (a pipeline stage axis, for example).
  for (const Array<Range> *range : {&op.src_range, &op.dst_range}) {
    if (range->size() < 2)
      return Stmt();
    for (size_t i = 0; i + 2 < range->size(); ++i) {
      if (!is_one((*range)[i]->extent))
        return Stmt();
    }
  }
  const size_t src_rank = op.src_range.size();
  const size_t dst_rank = op.dst_range.size();
  const auto *rows_imm = as_const_int(op.src_range[src_rank - 2]->extent);
  const auto *cols_imm = as_const_int(op.src_range[src_rank - 1]->extent);
  if (rows_imm == nullptr || cols_imm == nullptr)
    return Stmt();
  const int64_t rows = *rows_imm;
  const int64_t cols = *cols_imm;
  const auto *dst_rows = as_const_int(op.dst_range[dst_rank - 2]->extent);
  const auto *dst_cols = as_const_int(op.dst_range[dst_rank - 1]->extent);
  if (dst_rows == nullptr || dst_cols == nullptr || *dst_rows != cols ||
      *dst_cols != rows)
    return Stmt();

  // ldmatrix/stmatrix rows past a buffer edge would reach the neighbouring row
  // or allocation; the SIMT loop predicates those elements instead.
  for (const auto &[buffer, range] :
       {std::pair{&src, &op.src_range}, std::pair{&dst, &op.dst_range}}) {
    for (size_t i = 0; i < range->size(); ++i) {
      const Range &axis = (*range)[i];
      if (!analyzer->CanProve(axis->min >= 0,
                              arith::ProofStrength::kSymbolicBound) ||
          !analyzer->CanProve(axis->min + axis->extent <= (*buffer)->shape[i],
                              arith::ProofStrength::kSymbolicBound))
        return Stmt();
    }
  }

  // A 16-byte row segment holds `tile` elements; a tile is tile x tile.
  const int tile = bits == 8 ? 16 : 8;
  if (rows % tile != 0 || cols % tile != 0)
    return Stmt();

  if (!lower_args.thread_bounds.defined())
    return Stmt();
  const auto *thread_min = as_const_int(lower_args.thread_bounds->min);
  const auto *thread_extent = as_const_int(lower_args.thread_bounds->extent);
  if (thread_min == nullptr || thread_extent == nullptr ||
      *thread_min % 32 != 0 || *thread_extent % 32 != 0 || *thread_extent == 0)
    return Stmt();

  if (!HasAlignedRowSegments(src, op.src_range, rows, cols, tile,
                             lower_args.layout_map, analyzer) ||
      !HasAlignedRowSegments(dst, op.dst_range, cols, rows, tile,
                             lower_args.layout_map, analyzer))
    return Stmt();

  const int64_t tile_cols = cols / tile;
  const int matrices_per_tile = bits == 8 ? 2 : 1;
  const int64_t matrices = rows / tile * tile_cols * matrices_per_tile;
  const int num = matrices % 4 == 0 ? 4 : (matrices % 2 == 0 ? 2 : 1);
  const int64_t groups = matrices / num;
  const int64_t num_warps = *thread_extent / 32;
  const int64_t iterations = (groups + num_warps - 1) / num_warps;

  DataType index_type = lower_args.thread_index.dtype();
  auto idx = [&](int64_t value) { return IntImm(index_type, value); };
  PrimExpr thread = lower_args.thread_index - lower_args.thread_bounds->min;
  PrimExpr lane = FloorMod(thread, idx(32));
  Var iteration("transpose_iter", index_type);
  PrimExpr group = iteration * idx(num_warps) + FloorDiv(thread, idx(32));
  PrimExpr matrix =
      group * idx(num) + FloorMod(FloorDiv(lane, idx(8)), idx(num));
  PrimExpr row = FloorMod(lane, idx(8));
  PrimExpr tile_index = FloorDiv(matrix, idx(matrices_per_tile));
  PrimExpr matrix_half = FloorMod(matrix, idx(matrices_per_tile));
  PrimExpr tile_row = FloorDiv(tile_index, idx(tile_cols)) * idx(tile);
  PrimExpr tile_col = FloorMod(tile_index, idx(tile_cols)) * idx(tile);

  PrimExpr src_row, dst_row;
  if (bits == 16) {
    src_row = tile_row + row;
    dst_row = tile_col + row;
  } else {
    src_row = tile_row + FloorDiv(row, idx(2)) * idx(4) +
              FloorMod(row, idx(2)) + matrix_half * idx(2);
    dst_row = tile_col + row * idx(2) + matrix_half;
  }
  Array<PrimExpr> src_indices =
      RegionIndices(op.src_range, analyzer->Simplify(src_row), tile_col);
  Array<PrimExpr> dst_indices =
      RegionIndices(op.dst_range, analyzer->Simplify(dst_row), tile_row);

  Buffer fragment = decl_buffer({IntImm(DataType::Int(32), num)},
                                DataType::Int(32), "transpose_frag", "local");
  PrimExpr load = Call(
      DataType::Handle(), tl::ptx_ldmatrix(),
      {IntImm(DataType::Int(32), bits == 8 ? 1 : 0),
       IntImm(DataType::Int(32), num),
       SharedAccessPtr(src, src_indices, tile, /*rw_mask=*/1),
       Call(DataType::Handle(), tl::access_ptr(),
            {BufferLoad(fragment, {IntImm(DataType::Int(32), 0)}),
             IntImm(DataType::Int(32), num), IntImm(DataType::Int(32), 2)})});

  Array<PrimExpr> store_args = {IntImm(DataType::Int(32), bits == 16 ? 1 : 0),
                                IntImm(DataType::Int(32), num),
                                SharedAccessPtr(dst, dst_indices, tile,
                                                /*rw_mask=*/2)};
  for (int i = 0; i < num; ++i) {
    if (bits == 16) {
      store_args.push_back(
          BufferLoad(fragment, {IntImm(DataType::Int(32), i)}));
      continue;
    }
    int pair = i / 2 * 2;
    store_args.push_back(
        Call(DataType::Int(32), builtin::call_pure_extern(),
             {StringImm("__byte_perm"),
              BufferLoad(fragment, {IntImm(DataType::Int(32), pair)}),
              BufferLoad(fragment, {IntImm(DataType::Int(32), pair + 1)}),
              IntImm(DataType::Int(32), i % 2 == 0 ? 0x6420 : 0x7531)}));
  }
  store_args.push_back(StringImm("m8n8"));
  PrimExpr store = Call(DataType::Handle(), tl::ptx_stmatrix(), store_args);

  Stmt body = SeqStmt({Evaluate(load), Evaluate(store)});
  // The guard depends only on the warp, so every lane of an issuing warp
  // executes the aligned matrix instructions.
  if (groups % num_warps != 0)
    body = IfThenElse(group < idx(groups), body);
  For loop = PragmaUnrollLoop(
      For(iteration, idx(0), idx(iterations), ForKind::kSerial, body));
  return SeqStmt({AllocBuffer(fragment), loop});
}

Stmt LowerCudaTranspose(const TransposeNode &op, const LowerArgs &lower_args,
                        arith::Analyzer *analyzer) {
  if (Stmt matrix = LowerWarpMatrixTranspose(op, lower_args, analyzer);
      matrix.defined())
    return matrix;
  return backend::Transpose::Lower(op, lower_args, analyzer);
}

bool RegisterCudaTranspose() {
  RegisterTransposeImpl(TransposeImpl{
      "cuda.Transpose",
      MatchCudaTransposeTarget,
      LowerCudaTranspose,
  });
  return true;
}

const bool cuda_transpose_registered = RegisterCudaTranspose();

} // namespace

} // namespace tl
} // namespace tvm
