/*!
 * \file lower_blackwell_2sm.cc
 * \brief Lower 2SM TCGEN5MMA and related on Blackwell target
 *
 * This pass runs before LowerTileOp. At that point the IR still has T.gemm
 * (tl.tileop.gemm Call), not the lowered tl::tcgen5mma_gemm_ss/ts. We detect
 * Gemm ops that will be lowered to TCGEN5MMA with use_2cta and set block attr.
 */

// todo: consider mixture of 1cta/2cta tcgen5mma in the same kernel

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/gemm.h"
#include "../op/operator.h"
#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace attr {
constexpr const char *kUse2Cta = "use_2cta";
} // namespace attr

/**
 * \brief Check if any block in the body has cluster_dims (2,1,1) or (1,2,1).
 * enable_2cta is only allowed when cluster_dims matches one of these.
 */
static bool HasValidClusterDimsFor2Cta(const Stmt &body) {
  bool found = false;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    if (found)
      return;
    if (const auto *block = node.as<BlockNode>()) {
      if (block->annotations.count("cluster_dims")) {
        if (auto arr = block->annotations.Get("cluster_dims")
                           ->try_cast<Array<Integer>>()) {
          if (arr.value().size() >= 3) {
            int64_t x = arr.value()[0]->value;
            int64_t y = arr.value()[1]->value;
            int64_t z = arr.value()[2]->value;
            found =
                (x == 2 && y == 1 && z == 1) || (x == 1 && y == 2 && z == 1);
          }
        }
      }
    }
  });
  return found;
}

/**
 * \brief Detect 2SM TCGEN5MMA in the kernel (before LowerTileOp).
 * Looks for T.gemm (tl.tileop.gemm Call); if it will be lowered to TCGEN5MMA
 * with use_2cta, sets the flag for the mutator to add block attr.
 */
class Tcgen5_2SmLower : public StmtExprMutator {
public:
  Tcgen5_2SmLower(bool cluster_dims_valid)
      : cluster_dims_valid_(cluster_dims_valid) {}
  bool has_2sm_tcgen5mma() const { return has_2sm_tcgen5mma_; }

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      TileOperator tile_op = ParseOperator(ffi::GetRef<Stmt>(op));
      if (tile_op.defined() && tile_op.as<Gemm>()) {
        // Check if the user explicitly requested 2CTA via the use_2cta
        // annotation on the Call node (set by T.tcgen05_gemm(use_2cta=True)).
        if (call->annotations.count(attr::kUse2Cta)) {
          auto val = call->annotations.Get(attr::kUse2Cta).value();
          if (const auto *imm = val.as<IntImmNode>()) {
            if (imm->value) {
              if (!cluster_dims_valid_) {
                LOG(WARNING) << "Invalid cluster_dims disables 2CTA "
                                "TCGEN5MMA, use 1CTA variant instead.";
                return StmtExprMutator::VisitStmt_(op);
              }
              has_2sm_tcgen5mma_ = true;
            }
          }
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  bool cluster_dims_valid_;
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
    Stmt body = f->body;
    bool cluster_dims_valid = HasValidClusterDimsFor2Cta(body);
    Tcgen5_2SmLower lower(cluster_dims_valid);
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
