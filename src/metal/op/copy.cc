/*!
 * \file tl/metal/op/copy.cc
 * \brief Metal implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "metal/op/builtin.h"
#include "metal/op/utils.h"
#include "metal/target_utils.h"
#include "op/utils.h"
#include "support/check.h"

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace tl {

using namespace tirx;

namespace metal {

namespace {

bool CheckSIMDGroupCopy(const CopyNode &op) {
  return IsSIMDGroupBuffer(op.src) &&
         (IsSharedBuffer(op.dst) || IsGlobalBuffer(op.dst));
}

bool CheckCooperativeTensorCopy(const CopyNode &op) {
  return IsCooperativeTensorBuffer(op.src) &&
         (IsSharedBuffer(op.dst) || IsGlobalBuffer(op.dst));
}

// Runtime dispatch encloses shared-memory barriers, so all threads must take
// the same branch. Follow Bind aliases and loop domains; reject memory reads
// and calls rather than assuming their results are uniform.
bool IsThreadgroupUniform(const PrimExpr &expr, const LowerArgs &lower_args) {
  bool uniform = true;
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (node.as<BufferLoadNode>() || node.as<CallNode>()) {
      uniform = false;
    } else if (const auto *var_node = node.as<VarNode>()) {
      Var var = GetRef<Var>(var_node);
      if (UsesVar(lower_args.thread_index, [&](const VarNode *thread_var) {
            return var.same_as(GetRef<Var>(thread_var));
          })) {
        uniform = false;
      } else if (lower_args.bind_var_to_expr.count(var)) {
        uniform &= IsThreadgroupUniform(lower_args.bind_var_to_expr.at(var),
                                        lower_args);
      } else if (lower_args.enclosing_loops.count(var)) {
        For loop = lower_args.enclosing_loops.at(var);
        uniform &= (loop->kind == ForKind::kSerial ||
                    loop->kind == ForKind::kUnrolled) &&
                   IsThreadgroupUniform(loop->min, lower_args) &&
                   IsThreadgroupUniform(loop->extent, lower_args) &&
                   (!loop->step.defined() ||
                    IsThreadgroupUniform(loop->step.value(), lower_args));
      }
    }
  });
  return uniform;
}

Stmt LowerSIMDGroupCopy(const CopyNode &op, const LowerArgs &lower_args,
                        arith::Analyzer *analyzer) {
  TVM_FFI_ICHECK(IsSIMDGroupBuffer(op.src));

  int total_elements = 1;
  for (auto s : op.src->shape) {
    auto imm = s.as<IntImmNode>();
    TVM_FFI_ICHECK(imm) << "simdgroup buffer must have constant shape";
    total_elements *= imm->value;
  }
  TVM_FFI_ICHECK(total_elements % 64 == 0)
      << "simdgroup buffer size must be multiple of 64 (8x8), got "
      << total_elements;

  TVM_FFI_ICHECK(op.src_range.size() == 2)
      << "Expected 2D source for simdgroup store";
  TVM_FFI_ICHECK(op.dst_range.size() == 2)
      << "Expected 2D destination for simdgroup store";
  PrimExpr dst_row_base = op.dst_range[0]->min;
  PrimExpr dst_col_base = op.dst_range[1]->min;
  PrimExpr dst_stride = op.dst->shape[op.dst->shape.size() - 1];

  bool row_major_destination = true;
  if (lower_args.buffer_remap.count(op.dst)) {
    // LowerTileOp remaps address_of after this lowering, but the intrinsic
    // keeps dst_stride. Direct stores require the same row-major address
    // mapping; proving the logical bounds alone does not bound a remapped
    // store.
    Layout layout = lower_args.layout_map.at(op.dst);
    Array<PrimExpr> vars = layout->GetForwardVars();
    Buffer physical = lower_args.buffer_remap.at(op.dst);
    PrimExpr capacity = IntImm(DataType::Int(64), 1);
    for (const PrimExpr &extent : physical->shape) {
      capacity = capacity * cast(DataType::Int(64), extent);
    }
    row_major_destination =
        vars.size() == 2 && layout->OutputDim() == physical->shape.size() &&
        analyzer->CanProveEqual(layout->GetLinearizedForwardIndex(),
                                vars[0] * dst_stride + vars[1]) &&
        analyzer->CanProve(cast(DataType::Int(64), op.dst->shape[0]) *
                               cast(DataType::Int(64), dst_stride) <=
                           capacity);
  }

  int warp_size = TargetMetalGetWarpSize(lower_args.target);
  const auto *block_size_imm =
      lower_args.thread_bounds->extent.as<IntImmNode>();
  TVM_FFI_ICHECK(block_size_imm)
      << "simdgroup copy requires constant thread bounds";
  int block_size = block_size_imm->value;
  int num_warps = block_size / warp_size;
  PrimExpr warp_id = FloorDiv(lower_args.thread_index, warp_size);

  const auto *m_imm = op.src_range[0]->extent.as<IntImmNode>();
  const auto *n_imm = op.src_range[1]->extent.as<IntImmNode>();
  TVM_FFI_ICHECK(m_imm && n_imm) << "simdgroup copy requires constant extents";
  int M = m_imm->value;
  int N = n_imm->value;

  constexpr int kMPerWarp = 8;
  constexpr int kNPerWarp = 8;
  auto [m_warp, n_warp] =
      ComputeSquareWarpPartition(num_warps, M, N, kMPerWarp, kNPerWarp);

  TVM_FFI_ICHECK(M >= m_warp * kMPerWarp && N >= n_warp * kNPerWarp)
      << "Cannot partition " << M << "x" << N << " matrix across " << m_warp
      << "x" << n_warp << " warps with 8x8 simdgroup tiles";
  int warp_row_tiles = M / m_warp / kMPerWarp;
  int warp_col_tiles = N / n_warp / kNPerWarp;
  TVM_FFI_ICHECK(warp_row_tiles > 0 && warp_col_tiles > 0);
  TVM_FFI_ICHECK(warp_row_tiles * warp_col_tiles * 64 <= total_elements)
      << "Warp partition produces more tiles than buffer capacity";

  PrimExpr warp_m = FloorMod(warp_id, m_warp);
  PrimExpr warp_n = FloorDiv(warp_id, m_warp);

  // Test the entire logical tile, not each SIMD group's footprint: a group
  // taking the staging path may execute threadgroup barriers inserted later.
  PrimExpr full_tile = const_true();
  for (size_t d = 0; d < 2; ++d) {
    PrimExpr extent = op.src_range[d]->extent;
    PrimExpr base = op.dst_range[d]->min;
    full_tile = full_tile && base >= 0 && base <= op.dst->shape[d] - extent &&
                op.dst_range[d]->extent >= extent;
  }
  full_tile = analyzer->Simplify(full_tile);
  bool runtime_dispatch =
      row_major_destination && !lower_args.has_multidimensional_threads &&
      analyzer->CanProveEqual(lower_args.thread_bounds->min, 0) &&
      block_size % warp_size == 0 && !analyzer->CanProve(full_tile) &&
      !analyzer->CanProve(!full_tile) &&
      IsThreadgroupUniform(full_tile, lower_args);

  Buffer scratch;
  Array<Stmt> stmts;
  Array<Stmt> direct_stmts;
  for (int i = 0; i < warp_row_tiles; i++) {
    for (int j = 0; j < warp_col_tiles; j++) {
      int tile_idx = i * warp_col_tiles + j;
      PrimExpr row =
          dst_row_base + warp_m * (warp_row_tiles * kMPerWarp) + i * kMPerWarp;
      PrimExpr col =
          dst_col_base + warp_n * (warp_col_tiles * kNPerWarp) + j * kNPerWarp;
      // simdgroup_store always writes a complete 8x8 matrix. A valid
      // starting address alone does not make a destination tail safe.
      Array<PrimExpr> start{row, col};
      bool direct_store = row_major_destination;
      for (size_t d = 0; d < start.size(); ++d) {
        PrimExpr relative = start[d] - op.dst_range[d]->min;
        direct_store &=
            analyzer->CanProve(start[d] >= 0) &&
            analyzer->CanProve(start[d] <= op.dst->shape[d] - 8) &&
            analyzer->CanProve(relative >= 0) &&
            analyzer->CanProve(relative <= op.dst_range[d]->extent - 8);
      }
      if (runtime_dispatch) {
        PrimExpr direct_ptr = Call(DataType::Handle(), builtin::address_of(),
                                   {BufferLoad(op.dst, {row, col})});
        direct_stmts.push_back(Evaluate(
            Call(DataType::Handle(), builtin::simdgroup_store(),
                 {op.src->data, IntImm(DataType::Int(32), tile_idx), direct_ptr,
                  dst_stride, IntImm(DataType::Int(32), kMPerWarp),
                  IntImm(DataType::Int(32), kNPerWarp),
                  Cast(DataType::Bool(), IntImm(DataType::Int(32), 0))})));
      }
      Buffer destination = op.dst;
      Array<PrimExpr> indices{row, col};
      PrimExpr stride = dst_stride;
      if (!direct_store) {
        if (!scratch.defined()) {
          // Each SIMD group owns one complete tile. Reuse it only after all
          // lanes have finished the previous masked scatter.
          PrimExpr workspace =
              lower_args.add_workspace(num_warps * 64, op.dst->dtype);
          const auto *access = workspace.as<CallNode>();
          TVM_FFI_ICHECK(access &&
                         access->op.same_as(builtin::tvm_access_ptr()));
          scratch = Buffer(Downcast<Var>(access->args[1]), op.dst->dtype,
                           {PrimExpr(num_warps * 64)}, {}, PrimExpr(0),
                           "simdgroup_staging", 0, 0, kDefault);
        }
        destination = scratch;
        indices = {warp_id * 64};
        stride = PrimExpr(8);
      }
      PrimExpr ptr = Call(DataType::Handle(), builtin::address_of(),
                          {BufferLoad(destination, indices)});
      stmts.push_back(Evaluate(
          Call(DataType::Handle(), builtin::simdgroup_store(),
               {op.src->data, IntImm(DataType::Int(32), tile_idx), ptr, stride,
                IntImm(DataType::Int(32), kMPerWarp),
                IntImm(DataType::Int(32), kNPerWarp),
                Cast(DataType::Bool(), IntImm(DataType::Int(32), 0))})));
      if (!direct_store) {
        Stmt sync =
            Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                          {StringImm("warp")}));
        stmts.push_back(sync);
        PrimExpr lane = FloorMod(lower_args.thread_index, warp_size);
        for (int element = 0; element < 64; element += warp_size) {
          PrimExpr index = lane + element;
          PrimExpr dst_row = row + FloorDiv(index, 8);
          PrimExpr dst_col = col + FloorMod(index, 8);
          PrimExpr in_bounds =
              dst_row >= 0 && dst_row < op.dst->shape[0] && dst_col >= 0 &&
              dst_col < op.dst->shape[1] && dst_row >= op.dst_range[0]->min &&
              dst_row < op.dst_range[0]->min + op.dst_range[0]->extent &&
              dst_col >= op.dst_range[1]->min &&
              dst_col < op.dst_range[1]->min + op.dst_range[1]->extent;
          stmts.push_back(IfThenElse(
              in_bounds,
              BufferStore(op.dst, BufferLoad(scratch, {warp_id * 64 + index}),
                          {dst_row, dst_col})));
        }
        stmts.push_back(sync);
      }
    }
  }
  Stmt body = stmts.size() == 1 ? stmts[0] : SeqStmt(stmts);
  if (runtime_dispatch && scratch.defined()) {
    Stmt direct =
        direct_stmts.size() == 1 ? direct_stmts[0] : SeqStmt(direct_stmts);
    return IfThenElse(full_tile, direct, body);
  }
  return body;
}

Stmt LowerCooperativeTensorCopy(const CopyNode &op, const LowerArgs &lower_args,
                                arith::Analyzer *analyzer) {
  (void)analyzer;
  TVM_FFI_ICHECK(IsCooperativeTensorBuffer(op.src));
  int total_elements = 1;
  for (auto s : op.src->shape) {
    auto imm = s.as<IntImmNode>();
    TVM_FFI_ICHECK(imm) << "cooperative_tensor buffer must have constant shape";
    total_elements *= imm->value;
  }

  constexpr int kTileSize = 16;
  constexpr int kTileElems = kTileSize * kTileSize;
  TVM_FFI_ICHECK(total_elements % kTileElems == 0)
      << "cooperative_tensor buffer size must be multiple of " << kTileElems
      << ", got " << total_elements;

  TVM_FFI_ICHECK(op.dst_range.size() == 2)
      << "Expected 2D destination for cooperative_tensor store";
  PrimExpr dst_row_base = op.dst_range[0]->min;
  PrimExpr dst_col_base = op.dst_range[1]->min;
  PrimExpr dst_stride = op.dst->shape[op.dst->shape.size() - 1];

  int warp_size = TargetMetalGetWarpSize(lower_args.target);
  const auto *block_size_imm =
      lower_args.thread_bounds->extent.as<IntImmNode>();
  TVM_FFI_ICHECK(block_size_imm)
      << "cooperative_tensor copy requires constant thread bounds";
  int block_size = block_size_imm->value;
  int num_warps = block_size / warp_size;
  PrimExpr warp_id = FloorDiv(lower_args.thread_index, warp_size);

  const auto *m_imm = op.src_range[0]->extent.as<IntImmNode>();
  const auto *n_imm = op.src_range[1]->extent.as<IntImmNode>();
  TVM_FFI_ICHECK(m_imm && n_imm)
      << "cooperative_tensor copy requires constant extents";
  int M = m_imm->value;
  int N = n_imm->value;

  int kMPerWarp = kTileSize;
  int kNPerWarp = kTileSize * 2;
  int m_warp = 1, n_warp = num_warps;
  int max_m = M / kMPerWarp;
  int max_n = N / kNPerWarp;

  float ideal = N > 0 ? static_cast<float>(M) / N : 1.f;
  float best_score = std::numeric_limits<float>::max();
  for (int m = 1; m <= std::min(num_warps, max_m); ++m) {
    if (num_warps % m != 0) {
      continue;
    }
    int n = num_warps / m;
    if (n > max_n) {
      continue;
    }
    float m_per = static_cast<float>(M) / (m * kMPerWarp);
    float n_per = static_cast<float>(N) / (n * kNPerWarp);
    float score = std::abs(m_per / n_per - ideal);
    if (score < best_score) {
      best_score = score;
      m_warp = m;
      n_warp = n;
    }
  }

  int elems_per_thread = total_elements / (num_warps * warp_size);
  int warp_M = M / m_warp;
  int warp_N = N / n_warp;
  int warp_tiles = elems_per_thread / (kTileSize * kTileSize / warp_size);

  int kTileN = warp_N;
  int kTileM = kTileSize;
  if (warp_tiles > 0 && warp_M > kTileSize) {
    kTileN = warp_N;
    kTileM = kTileSize;
  }
  if (kTileN > warp_N) {
    kTileN = warp_N;
  }

  int warp_row_tiles = warp_M / kTileM;
  int warp_col_tiles = warp_N / kTileN;

  TVM_FFI_ICHECK(warp_row_tiles > 0 && warp_col_tiles > 0)
      << "Cannot partition " << M << "x" << N << " matrix across " << m_warp
      << "x" << n_warp << " warps";

  int tile_elems_per_thread = kTileM * kTileN / warp_size;
  TVM_FFI_ICHECK(warp_row_tiles * warp_col_tiles * tile_elems_per_thread ==
                 elems_per_thread)
      << "Tile partition inconsistent with buffer size: " << warp_row_tiles
      << "x" << warp_col_tiles << " tiles of " << kTileM << "x" << kTileN
      << " = " << warp_row_tiles * warp_col_tiles * tile_elems_per_thread
      << " elems/thread, expected " << elems_per_thread;

  PrimExpr warp_m = FloorMod(warp_id, m_warp);
  PrimExpr warp_n = FloorDiv(warp_id, m_warp);

  Array<Stmt> stmts;
  for (int i = 0; i < warp_row_tiles; i++) {
    for (int j = 0; j < warp_col_tiles; j++) {
      int tile_idx = i * warp_col_tiles + j;
      PrimExpr row = dst_row_base + warp_m * warp_M + i * kTileM;
      PrimExpr col = dst_col_base + warp_n * warp_N + j * kTileN;
      PrimExpr ptr = Call(DataType::Handle(), builtin::address_of(),
                          {BufferLoad(op.dst, {row, col})});
      int kMMAK = kTileSize;
      stmts.push_back(Evaluate(Call(
          DataType::Handle(), cooperative_tensor_store(),
          {op.src->data, IntImm(DataType::Int(32), tile_idx), ptr, dst_stride,
           IntImm(DataType::Int(32), kTileM), IntImm(DataType::Int(32), kTileN),
           Cast(DataType::Bool(), IntImm(DataType::Int(32), 0)),
           IntImm(DataType::Int(32), kTileM), IntImm(DataType::Int(32), kTileN),
           IntImm(DataType::Int(32), kMMAK), IntImm(DataType::Int(32), 2)})));
    }
  }
  if (stmts.size() == 1) {
    return stmts[0];
  }
  return SeqStmt(stmts);
}

} // namespace

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    return op.InferSIMTLayout(layout_args, level);
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    if (CheckSIMDGroupCopy(op)) {
      return LowerSIMDGroupCopy(op, lower_args, analyzer);
    }
    if (CheckCooperativeTensorCopy(op)) {
      return LowerCooperativeTensorCopy(op, lower_args, analyzer);
    }
    return LowerNormalCopy(op, lower_args, analyzer);
  }
};

} // namespace metal

namespace {

bool MatchMetalCopyTarget(Target target) { return TargetIsMetal(target); }

bool RegisterMetalCopy() {
  RegisterCopyImpl(CopyImpl{
      "metal.Copy",
      MatchMetalCopyTarget,
      100,
      metal::Copy::InferLayout,
      metal::Copy::Lower,
  });
  return true;
}

const bool metal_copy_registered = RegisterMetalCopy();

} // namespace

} // namespace tl
} // namespace tvm
