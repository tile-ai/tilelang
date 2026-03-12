/*!
 * \file lower_blackwell_2sm.cc
 * \brief Lower 2SM TCGEN5MMA and related on Blackwell target
 *
 * This pass runs before LowerTileOp. At that point the IR still has T.gemm
 * (tl_gemm / tl.tileop.gemm_py Call), not the lowered tl::tcgen5mma_gemm_ss/ts.
 * We detect Gemm ops that will be lowered to TCGEN5MMA with use_2cta and set
 * block attr.
 *
 * Tilelang gemm defaults to v2 (GemmPyNode); we only support v2, not v1 (Gemm).
 */

// todo: consider mixture of 1cta/2cta tcgen5mma in the same kernel

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "../op/gemm_py.h"
#include "../op/operator.h"
#include "../op/tcgen5_meta.h"
#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace attr {
constexpr const char *kUse2Cta = "use_2cta";
} // namespace attr

/**
 * \brief Detect 2SM TCGEN5MMA in the kernel (before LowerTileOp).
 * Looks for T.gemm (tl_gemm() Call); if it will be lowered to TCGEN5MMA with
 * use_2cta, sets the flag for the mutator to add block attr.
 * Only supports v2 (GemmPy); v1 (Gemm) is ignored.
 */
class Tcgen5_2SmLower : public StmtExprMutator {
public:
  explicit Tcgen5_2SmLower(Target target) : target_(std::move(target)) {}
  bool has_2sm_tcgen5mma() const { return has_2sm_tcgen5mma_; }

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      // Tilelang gemm defaults to v2 (GemmPy); we only support v2, not v1
      // (Gemm).
      if (call->op.same_as(GemmPy::Get())) {
        TileOperator tile_op = ParseOperator(ffi::GetRef<Stmt>(op));
        if (tile_op.defined()) {
          if (Optional<GemmPy> opt_gemm_py = tile_op.as<GemmPy>()) {
            const GemmPyNode *node = opt_gemm_py.value().get();
            if (node->allowTcgen5Mma(target_)) {
              auto [ok, meta] =
                  GetTCGEN5MMAMeta(node->m_, node->n_, node->k_,
                                   node->a_->dtype, node->c_->dtype);
              if (ok && meta.enable_2cta) {
                // LOG(INFO) << "Found 2SM TCGEN5MMA!";
                has_2sm_tcgen5mma_ = true;
                // NOTE(wt): Currently this only act as a detector of tcgen05
                // 2sm, while we may add the lower logic here in the future.
              }
            }
          }
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Target target_;
  bool has_2sm_tcgen5mma_ = false;
};

class Tcgen5_2SmAnnotator : public StmtExprMutator {
public:
  explicit Tcgen5_2SmAnnotator() {}

private:
  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    Stmt new_realize = StmtExprMutator::VisitStmt_(op);
    if (root_block_annotated_)
      return new_realize;
    const auto *realize = new_realize.as<BlockRealizeNode>();
    ICHECK(realize);
    Block block = realize->block;
    BlockNode *n = block.CopyOnWrite();
    // Set block attr: {use_2cta: 1}
    // lower_shared_tmem.cc will depend on this to allocate/deallocate tmem with
    // 2cta.
    n->annotations.Set(attr::kUse2Cta, IntImm(DataType::Int(32), 1));
    root_block_annotated_ = true;
    return BlockRealize(realize->iter_values, realize->predicate, block);
  }

  bool root_block_annotated_ = false;
};

using namespace tir::transform;

tvm::transform::Pass LowerBlackwell2SM() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    Optional<Target> opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!opt_target.defined() || !TargetIsSm100(opt_target.value())) {
      return f;
    }
    if (ctx->GetConfig(kDisable2CTATcgen5MMA, Optional<Bool>())
            .value_or(false)) {
      LOG(INFO) << "2CTA TCGEN5MMA is disabled by pass config";
      return f;
    }
    Stmt body = f->body;
    Tcgen5_2SmLower lower(opt_target.value());
    body = lower(std::move(body));
    if (lower.has_2sm_tcgen5mma()) {
      // Annotate block attr for using 2cta tcgen5
      Tcgen5_2SmAnnotator annotator;
      body = annotator(std::move(body));
    }
    return PrimFunc(f->params, body, f->ret_type, f->buffer_map, f->attrs);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerBlackwell2SM", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerBlackwell2SM", LowerBlackwell2SM);
}

} // namespace tl
} // namespace tvm
