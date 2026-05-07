/*!
 * \file tl/backend/metal/op/copy.cc
 * \brief Metal implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "op/utils.h"
#include "target/utils.h"

#include <tvm/tir/builtin.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace tvm {
namespace tl {

using namespace tir;

namespace metal {

namespace {

bool CheckSIMDGroupCopy(const CopyNode &op) {
  return IsSIMDGroupBuffer(op.src) &&
         (IsSharedBuffer(op.dst) || IsGlobalBuffer(op.dst));
}

Stmt LowerSIMDGroupCopy(const CopyNode &op, const LowerArgs &T,
                        arith::Analyzer *analyzer) {
  (void)analyzer;
  ICHECK(IsSIMDGroupBuffer(op.src));

  int total_elements = 1;
  for (auto s : op.src->shape) {
    auto imm = s.as<IntImmNode>();
    ICHECK(imm) << "simdgroup buffer must have constant shape";
    total_elements *= imm->value;
  }
  ICHECK(total_elements % 64 == 0)
      << "simdgroup buffer size must be multiple of 64 (8x8), got "
      << total_elements;

  ICHECK(op.src_range.size() == 2) << "Expected 2D source for simdgroup store";
  ICHECK(op.dst_range.size() == 2)
      << "Expected 2D destination for simdgroup store";
  PrimExpr dst_row_base = op.dst_range[0]->min;
  PrimExpr dst_col_base = op.dst_range[1]->min;
  PrimExpr dst_stride = op.dst->shape[op.dst->shape.size() - 1];

  int warp_size = TargetGetWarpSize(T.target);
  const auto *block_size_imm = T.thread_bounds->extent.as<IntImmNode>();
  ICHECK(block_size_imm)
      << "simdgroup copy requires constant thread bounds";
  int block_size = block_size_imm->value;
  int num_warps = block_size / warp_size;
  PrimExpr warp_id = FloorDiv(T.thread_var, warp_size);

  const auto *m_imm = op.src_range[0]->extent.as<IntImmNode>();
  const auto *n_imm = op.src_range[1]->extent.as<IntImmNode>();
  ICHECK(m_imm && n_imm) << "simdgroup copy requires constant extents";
  int M = m_imm->value;
  int N = n_imm->value;

  constexpr int kMPerWarp = 8;
  constexpr int kNPerWarp = 8;
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

  ICHECK(M >= m_warp * kMPerWarp && N >= n_warp * kNPerWarp)
      << "Cannot partition " << M << "x" << N << " matrix across " << m_warp
      << "x" << n_warp << " warps with 8x8 simdgroup tiles";
  int warp_row_tiles = M / m_warp / kMPerWarp;
  int warp_col_tiles = N / n_warp / kNPerWarp;
  ICHECK(warp_row_tiles > 0 && warp_col_tiles > 0);
  ICHECK(warp_row_tiles * warp_col_tiles * 64 <= total_elements)
      << "Warp partition produces more tiles than buffer capacity";

  PrimExpr warp_m = FloorMod(warp_id, m_warp);
  PrimExpr warp_n = FloorDiv(warp_id, m_warp);

  Array<Stmt> stmts;
  for (int i = 0; i < warp_row_tiles; i++) {
    for (int j = 0; j < warp_col_tiles; j++) {
      int tile_idx = i * warp_col_tiles + j;
      PrimExpr row =
          dst_row_base + warp_m * (warp_row_tiles * kMPerWarp) + i * kMPerWarp;
      PrimExpr col =
          dst_col_base + warp_n * (warp_col_tiles * kNPerWarp) + j * kNPerWarp;
      PrimExpr ptr = Call(DataType::Handle(), builtin::address_of(),
                          {BufferLoad(op.dst, {row, col})});
      stmts.push_back(Evaluate(Call(
          DataType::Handle(), builtin::simdgroup_store(),
          {op.src->data, IntImm(DataType::Int(32), tile_idx), ptr, dst_stride,
           IntImm(DataType::Int(32), kMPerWarp),
           IntImm(DataType::Int(32), kNPerWarp),
           Cast(DataType::Bool(), IntImm(DataType::Int(32), 0))})));
    }
  }
  if (stmts.size() == 1) {
    return stmts[0];
  }
  return SeqStmt(stmts);
}

} // namespace

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                               InferLevel level) {
    return op.InferSIMTLayout(T, level);
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    if (CheckSIMDGroupCopy(op)) {
      return LowerSIMDGroupCopy(op, T, analyzer);
    }
    return LowerNormalCopy(op, T, analyzer);
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
