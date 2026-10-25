/*!
 * \file tl/cpu/op/gemm.cc
 * \brief CPU implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"
#include "support/check.h"

#include "backend/common/target_utils.h"
#include "op/utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace cpu {

namespace {

constexpr const char *kCPUScalar = "cpu.scalar";

} // namespace

struct Gemm {
  static Array<PrimExpr> Indices(const TensorRegion &region, PrimExpr row,
                                 PrimExpr column) {
    const auto &ranges = region->region;
    TVM_FFI_CHECK_GE(ranges.size(), 2, ValueError)
        << "CPU GEMM requires matrix operands";
    Array<PrimExpr> indices;
    for (size_t axis = 0; axis + 2 < ranges.size(); ++axis) {
      TVM_FFI_CHECK(IsOne(ranges[axis]->extent), ValueError)
          << "CPU GEMM requires singleton leading dimensions";
      indices.push_back(ranges[axis]->begin);
    }
    indices.push_back(ranges[ranges.size() - 2]->begin + row);
    indices.push_back(ranges[ranges.size() - 1]->begin + column);
    return indices;
  }

  static Stmt Lower(const GemmNode &op, const LowerArgs &,
                    sym::Analyzer analyzer) {
    for (int extent : {op.m_, op.n_, op.k_}) {
      TVM_FFI_CHECK_GE(extent, 0, ValueError)
          << "CPU GEMM dimensions must be nonnegative";
    }
    auto check_shape = [&](const TensorRegion &region, int rows, int columns) {
      const auto &ranges = region->region;
      TVM_FFI_CHECK_GE(ranges.size(), 2, ValueError)
          << "CPU GEMM requires matrix operands";
      TVM_FFI_CHECK(
          analyzer->CanProve(ranges[ranges.size() - 2]->extent >= rows) &&
              analyzer->CanProve(ranges[ranges.size() - 1]->extent >= columns),
          ValueError)
          << "CPU GEMM region is smaller than its logical matrix";
    };
    check_shape(op.aRegion_, op.transA_ ? op.k_ : op.m_,
                op.transA_ ? op.m_ : op.k_);
    check_shape(op.bRegion_, op.transB_ ? op.n_ : op.k_,
                op.transB_ ? op.k_ : op.n_);
    check_shape(op.cRegion_, op.m_, op.n_);
    PrimVar row("gemm_row"), column("gemm_column"), reduction("gemm_reduction");
    Array<PrimExpr> output = Indices(op.cRegion_, row, column);
    PrimExpr left =
        TensorLoad(op.a_, Indices(op.aRegion_, op.transA_ ? reduction : row,
                                  op.transA_ ? row : reduction));
    PrimExpr right =
        TensorLoad(op.b_, Indices(op.bRegion_, op.transB_ ? column : reduction,
                                  op.transB_ ? reduction : column));
    PrimType accum_type = op.c_.dtype();
    PrimExpr value = TensorLoad(op.c_, output) +
                     Cast(accum_type, left) * Cast(accum_type, right);
    Stmt update = TensorStore(op.c_, output, value);
    Stmt inner =
        For(reduction, 0, op.k_, 1, ForKind::kDefault, SeqStmt(update));
    Stmt initialize =
        If(op.clearAccum_,
           SeqStmt(TensorStore(op.c_, output, MakeConst(accum_type, 0))),
           std::nullopt);
    Stmt columns = For(column, 0, op.n_, 1, ForKind::kDefault,
                       SeqStmt({initialize, inner}));
    return For(row, 0, op.m_, 1, ForKind::kDefault, SeqStmt(columns));
  }

  static LayoutMap InferLayout(const GemmNode &, const LayoutInferArgs &) {
    return {};
  }

  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    (void)op;
    (void)block_size;
    (void)target;
    return kCPUScalar;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    (void)M;
    (void)N;
    (void)block_size;
    (void)target;
    (void)gemm_inst;
    policy.m_warp = 1;
    policy.n_warp = 1;
    return {1, 1};
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    (void)gemm_inst;
    return false;
  }
};

} // namespace cpu

namespace {

bool MatchCPUGemmTarget(Target target) { return TargetIsCPU(target); }

bool RegisterCPUGemm() {
  RegisterGemmImpl(GemmImpl{
      "cpu.Gemm",
      MatchCPUGemmTarget,
      cpu::Gemm::SelectInst,
      cpu::Gemm::ComputeWarpPartition,
      cpu::Gemm::ReuseExistingSharedLayout,
      cpu::Gemm::Lower,
      cpu::Gemm::InferLayout,
  });
  return true;
}

const bool cpu_gemm_registered = RegisterCPUGemm();

} // namespace

} // namespace tl
} // namespace tvm
