/*!
 * \file tl/cpu/op/reduce.cc
 * \brief Serial CPU reduction of native local tensor regions.
 */

#include "op/reduce.h"
#include "backend/common/op/reduction_semantics.h"
#include "backend/common/target_utils.h"
#include "op/utils.h"
#include "transform/loop_partition.h"

namespace tvm::tl {
using namespace tirx;
using namespace prim;
using namespace ffi;

namespace cpu {
struct Reduce {
  static Stmt Lower(const ReduceOpNode &op, const LowerArgs &args,
                    sym::AnalyzerObj *analyzer) {
    if (op.nan_propagate &&
        (op.dst.dtype().MatchesElementType(kDLFloat, 16) ||
         op.dst.dtype().MatchesElementType(kDLBfloat, 16))) {
      TVM_FFI_THROW(ValueError) << "CPU reduce does not support "
                                   "nan_propagate=True for float16/bfloat16";
    }
    ICHECK(IsLocalBuffer(op.src) && IsLocalBuffer(op.dst, true))
        << "CPU reduce requires local source and local/local.var destination";
    ICHECK_EQ(op.batch, 1)
        << "CPU reduce has no GPU AllReduce barrier batching";
    // These loops use logical coordinates. LowerTileOp transforms their
    // loads/stores and tensor identities together after this callback returns.
    // Replacing only the tensor here would bypass a non-identity layout.
    const TensorVar src = op.src, dst = op.dst;
    ICHECK(src.dtype() == dst.dtype())
        << "CPU reduction input/output dtype mismatch";
    const Array<Range> &sr = op.srcRegion_->region;
    const Array<Range> &dr = op.dstRegion_->region;
    const size_t rank = sr.size(), out_rank = dr.size();
    ICHECK_GE(op.dim, 0);
    ICHECK_LT(static_cast<size_t>(op.dim), rank);
    ICHECK(out_rank + 1 == rank || out_rank == rank);
    const bool keepdim = out_rank == rank;
    auto local = analyzer ? analyzer->Clone() : sym::Analyzer();
    for (size_t i = 0, j = 0; i < rank; ++i) {
      if (i == static_cast<size_t>(op.dim)) {
        if (keepdim) {
          ICHECK(local->CanProveEqual(dr[j]->extent, 1))
              << "A kept reduction axis must have unit extent";
          ++j;
        }
      } else {
        ICHECK(local->CanProveEqual(sr[i]->extent, dr[j]->extent))
            << "CPU reduction destination region shape mismatch";
        ++j;
      }
    }
    Array<PrimVar> vars;
    Array<PrimExpr> dst_indices;
    for (size_t i = 0; i < out_rank; ++i) {
      PrimVar var("reduce_o" + std::to_string(i), dr[i]->extent.ty());
      vars.push_back(var);
      dst_indices.push_back(dr[i]->begin + var);
    }
    PrimVar rv("reduce_k", sr[op.dim]->extent.ty());
    Array<PrimExpr> src_indices;
    for (size_t i = 0; i < rank; ++i) {
      PrimExpr relative =
          i == static_cast<size_t>(op.dim)
              ? PrimExpr(rv)
              : PrimExpr(vars[keepdim                           ? i
                              : i < static_cast<size_t>(op.dim) ? i
                                                                : i - 1]);
      src_indices.push_back(sr[i]->begin + relative);
    }
    Array<Stmt> stmts;
    if (op.clear) {
      stmts.push_back(
          TensorStore(dst, dst_indices, backend::reduce::MakeInitValue(op)));
    }
    Stmt update = TensorStore(
        dst, dst_indices,
        backend::reduce::MakeReduce(op, 1, TensorLoad(dst, dst_indices),
                                    TensorLoad(src, src_indices)));
    stmts.push_back(For(rv, IntImm(rv.ty(), 0), sr[op.dim]->extent,
                        IntImm(rv.ty(), 1), ForKind::kDefault,
                        SeqStmt(update)));
    Stmt body = SeqStmt(stmts);
    for (size_t i = out_rank; i > 0; --i) {
      const auto &var = vars[i - 1];
      body = For(var, IntImm(var.ty(), 0), dr[i - 1]->extent,
                 IntImm(var.ty(), 1), ForKind::kDefault, SeqStmt(body));
    }
    // Preserve the previous intent marker on the outer loop only. The actual
    // accumulation loop stays serial, avoiding code-size growth and reordering.
    if (out_rank)
      body = PragmaUnrollLoop(body.as_or_throw<For>());
    return body;
  }
};
} // namespace cpu

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterReduceImpl(ReduceImpl{"cpu.Reduce", TargetIsCPU, cpu::Reduce::Lower});
}

} // namespace tvm::tl
