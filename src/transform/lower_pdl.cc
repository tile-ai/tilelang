/*!
 * \file lower_pdl.cc
 * \brief Mark Device PrimFunc with attributes if CUDA PDL functions are called
 */

#include "../op/builtin.h"
#include "../target/utils.h"
#include "common/attr.h"
#include "tvm/ir/type.h"
#include "tvm/tir/builtin.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

class MarkCudaSyncCalls : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    MarkCudaSyncCalls mutator;
    PrimFunc new_f = f;
    new_f.CopyOnWrite()->body = mutator.VisitStmt(f->body);

    if (mutator.has_trigger_launch_) {
      new_f = WithAttr(std::move(new_f), attr::kHasTriggerLaunch, 1);
    }
    if (mutator.has_grid_sync_) {
      new_f = WithAttr(std::move(new_f), attr::kHasGridSync, 1);
    }
    return new_f;
  }

  PrimExpr VisitExpr_(const tir::CallNode *op) final {
    CheckCall(op);
    return StmtExprMutator::VisitExpr_(op);
  }

private:
  void CheckCall(const tir::CallNode *call) {
    if (!call)
      return;
    if (call->op.same_as(builtin::call_extern())) {
      if (!call->args.empty()) {
        if (const auto *str_node =
                call->args[0].as<tvm::tir::StringImmNode>()) {
          std::string func_name = str_node->value;
          if (func_name == "cudaTriggerProgrammaticLaunchCompletion") {
            has_trigger_launch_ = true;
          } else if (func_name == "cudaGridDependencySynchronize") {
            has_grid_sync_ = true;
          }
        }
      }
    }
  }

private:
  bool has_trigger_launch_ = false;
  bool has_grid_sync_ = false;

  MarkCudaSyncCalls() = default;
};

class EliminateCudaSyncCalls : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    EliminateCudaSyncCalls mutator;
    PrimFunc new_f = f;
    new_f.CopyOnWrite()->body = mutator.VisitStmt(f->body);

    return new_f;
  }

  PrimExpr VisitExpr_(const tir::CallNode *op) final {
    if (CheckCall(op)) {
      return make_zero(op->dtype);
    }

    return StmtExprMutator::VisitExpr_(op);
  }

private:
  bool CheckCall(const tir::CallNode *call) {
    if (!call)
      return false;

    if (call->op.same_as(builtin::call_extern())) {
      if (!call->args.empty()) {
        if (const auto *str_node =
                call->args[0].as<tvm::tir::StringImmNode>()) {
          std::string func_name = str_node->value;
          if (func_name == "cudaTriggerProgrammaticLaunchCompletion") {
            return true;
          } else if (func_name == "cudaGridDependencySynchronize") {
            return true;
          }
        }
      }
    }

    return false;
  }

private:
  EliminateCudaSyncCalls() = default;
};

using namespace tir::transform;

tvm::transform::Pass MarkCudaSyncCallsPass(bool have_pdl) {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return have_pdl ? MarkCudaSyncCalls::Substitute(f)
                    : EliminateCudaSyncCalls::Substitute(f);
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
