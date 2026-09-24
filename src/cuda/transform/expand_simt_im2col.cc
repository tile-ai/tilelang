/*!
 * \file expand_simt_im2col.cc
 * \brief Early exposure of CUDA SIMT im2col loads before scheduling.
 */

#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "arith/ir_mutator_with_analyzer.h"
#include "cuda/target_utils.h"
#include "op/copy.h"
#include "op/operator.h"
#include "op/utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

TVM_REGISTER_PASS_CONFIG_OPTION("tl.enable_early_simt_im2col", Bool);

namespace {

// Keep specialized implementations (such as Hopper TMA) and explicit operation
// annotations intact. Only expand the selected generic SIMT implementation.
class SIMTIm2ColExpander : public arith::IRMutatorWithAnalyzer {
public:
  explicit SIMTIm2ColExpander(arith::Analyzer *analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer) {}

private:
  Stmt VisitStmt_(const SBlockNode *node) final {
    // Explicit WS schedules refer to the original operations, not new loops.
    if (node->annotations.count("tl.ws_schedule")) {
      return GetRef<SBlock>(node);
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(node);
  }

  Stmt VisitStmt_(const EvaluateNode *node) final {
    Stmt stmt = arith::IRMutatorWithAnalyzer::VisitStmt_(node);
    const auto *call = stmt.as<EvaluateNode>()->value.as<CallNode>();
    if (!call || !call->op.same_as(Im2ColOp::Get()) ||
        !call->annotations.empty()) {
      return stmt;
    }
    Im2ColOp op = Downcast<Im2ColOp>(ParseOperator(GetRef<Call>(call)));
    const Buffer &src = op->src_;
    const Buffer &dst = op->dst_;
    const auto &region = op->dstRegion_->region;
    if (!IsGlobalBuffer(src) || !IsSharedBuffer(dst) ||
        src->shape.size() != 4 || region.size() != 2 ||
        src->dtype != dst->dtype || op->kernel_ <= 0 || op->stride_ <= 0 ||
        op->dilation_ <= 0) {
      return stmt;
    }
    // Do not reinterpret cropped source regions: the fallback currently uses
    // whole-buffer NHWC coordinates. Leave unsupported metadata untouched.
    for (size_t dim = 0; dim < 4; ++dim) {
      const Range &range = op->srcRegion_->region[dim];
      if (!analyzer_->CanProveEqual(range->min, 0) ||
          !analyzer_->CanProveEqual(range->extent, src->shape[dim])) {
        return stmt;
      }
    }
    PrimExpr h = src->shape[1], w = src->shape[2], c = src->shape[3];
    PrimExpr oh =
        FloorDiv(h + 2 * op->padding_ - (op->kernel_ - 1) * op->dilation_ - 1,
                 op->stride_) +
        1;
    PrimExpr ow =
        FloorDiv(w + 2 * op->padding_ - (op->kernel_ - 1) * op->dilation_ - 1,
                 op->stride_) +
        1;
    if (!analyzer_->CanProve(oh > 0) || !analyzer_->CanProve(ow > 0) ||
        !analyzer_->CanProve(c > 0)) {
      return stmt;
    }

    PrimExpr block_m = region[0]->extent, block_k = region[1]->extent;
    Var i("im2col_m", block_m.dtype()), j("im2col_k", block_k.dtype());
    PrimExpr m = op->nhw_step_ * block_m + i;
    PrimExpr k = op->c_step_ * block_k + j;
    PrimExpr batch = FloorDiv(m, oh * ow);
    PrimExpr ih = FloorDiv(FloorMod(m, oh * ow), ow) * op->stride_ +
                  FloorDiv(k, op->kernel_ * c) * op->dilation_ - op->padding_;
    PrimExpr iw = FloorMod(m, ow) * op->stride_ +
                  FloorMod(FloorDiv(k, c), op->kernel_) * op->dilation_ -
                  op->padding_;
    PrimExpr channel = FloorMod(k, c);
    // Preserve zero fill for spatial/batch tails and incomplete reduction
    // tiles. No divisibility assumption on channels or block_K is needed.
    // These predicates only inspect coordinates. Eager boolean conjunction
    // preserves the explicit T.Parallel form without short-circuit branches;
    // the actual memory access remains guarded by if_then_else below.
    PrimExpr valid = const_true();
    for (const PrimExpr &condition :
         Array<PrimExpr>{m >= 0, batch < src->shape[0], k >= 0,
                         k < op->kernel_ * op->kernel_ * c, ih >= 0, ih < h,
                         iw >= 0, iw < w}) {
      valid = bitwise_and(valid, condition);
    }
    PrimExpr value =
        if_then_else(valid, BufferLoad(src, {batch, ih, iw, channel}),
                     make_zero(src->dtype));
    Stmt body =
        BufferStore(dst, value, {region[0]->min + i, region[1]->min + j});
    body = For(j, make_zero(j.dtype()), block_k, ForKind::kParallel, body);
    return For(i, make_zero(i.dtype()), block_m, ForKind::kParallel, body,
               std::nullopt, {}, std::nullopt, node->span);
  }
};

} // namespace

tvm::transform::Pass ExpandSIMTIm2Col() {
  auto pass_func = [](PrimFunc f, const IRModule &,
                      tvm::transform::PassContext) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsCuda(target.value()) ||
        !Im2ColUsesSIMT(target.value())) {
      return f;
    }
    arith::Analyzer analyzer;
    SIMTIm2ColExpander expander(&analyzer);
    f.CopyOnWrite()->body = expander(f->body);
    return f;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.ExpandSIMTIm2Col", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def(
      "tl.cuda.transform.UsesSIMTIm2Col", [](Target target) {
        return TargetIsCuda(target) && Im2ColUsesSIMT(target);
      });
  reflection::GlobalDef().def("tl.cuda.transform.ExpandSIMTIm2Col",
                              ExpandSIMTIm2Col);
}

} // namespace tl
} // namespace tvm
