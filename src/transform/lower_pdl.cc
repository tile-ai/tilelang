/*!
 * \file lower_pdl.cc
 * \brief Mark Device PrimFunc with attributes if CUDA PDL functions are called
 */

#include "../op/builtin.h"
#include "../target/utils.h"
#include "common/attr.h"
#include "support/check.h"
#include "tvm/ir/type.h"
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;

// NVCC has issues with __ldg when using PDL (Programmatic Dependent Launch)
// synchronization. Suppress the annotation when kHasGridSync is set.
class CheckLDGCalls : public StmtExprVisitor {
public:
  void VisitExpr_(const tirx::CallNode *op) final {
    if (op->op.same_as(tl::__ldg())) {
      LOG(FATAL) << "Cannot invoke __ldg function with pdl_sync";
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

class MarkCudaSyncCalls : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f, bool support_pdl) {
    MarkCudaSyncCalls mutator;
    PrimFunc new_f = f;
    new_f.CopyOnWrite()->body = mutator.VisitStmt(f->body);

    if (!support_pdl) {
      ICHECK(!mutator.has_trigger_launch_ && !mutator.has_grid_sync_)
          << "PDL is not supported";
    }

    if (mutator.has_trigger_launch_) {
      new_f = WithAttr(std::move(new_f), attr::kHasTriggerLaunch, 1);
    }
    if (mutator.has_grid_sync_) {
      new_f = WithAttr(std::move(new_f), attr::kHasGridSync, 1);
      CheckLDGCalls analyzer;
      analyzer(f->body);
    }
    return new_f;
  }

  PrimExpr VisitExpr_(const tirx::CallNode *op) final {
    if (op->op.same_as(tl::pdl_trigger())) {
      has_trigger_launch_ = true;
    } else if (op->op.same_as(tl::pdl_sync())) {
      has_grid_sync_ = true;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

private:
  bool has_trigger_launch_ = false;
  bool has_grid_sync_ = false;

  MarkCudaSyncCalls() = default;
};

using namespace tirx::transform;

tvm::transform::Pass MarkCudaSyncCallsPass(bool support_pdl) {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return MarkCudaSyncCalls::Substitute(f, support_pdl);
  };

  return CreatePrimFuncPass(pass_func, 0, "tl.MarkCudaSyncCalls", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MarkCudaSyncCalls",
                        MarkCudaSyncCallsPass);
}

} // namespace tl
} // namespace tvm
