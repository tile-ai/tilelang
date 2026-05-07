/*!
 * \file tl/backend/cuda/transform/persist_threadblock.cc
 * \brief Persist thread blocks with cooperative groups.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "backend/cuda/runtime.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {

namespace attr {
// BlockAttr, Containing the layout for all the buffers in the block
constexpr const char *kUseCooperativeGroups = "use_cooperative_groups";
} // namespace attr

using namespace tir;

class PersistThreadblock : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    PrimFuncNode *fptr = f.CopyOnWrite();
    PersistThreadblock substituter;
    // Trace the buffer map for tvm_access_ptr
    fptr->body = substituter.VisitStmt(f->body);
    if (substituter.has_sync_grid_) {
      f = WithAttr(std::move(f), attr::kUseCooperativeGroups,
                   IntImm(DataType::Int(32), 1));
    }
    return f;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(sync_grid())) {
        has_sync_grid_ = true;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

private:
  PersistThreadblock() = default;
  bool has_sync_grid_ = false;
};

using namespace tir::transform;

tvm::transform::Pass PersistThreadblock() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return PersistThreadblock::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PersistThreadblock", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.PersistThreadblock", PersistThreadblock);
}

} // namespace tl
} // namespace tvm
