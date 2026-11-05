/*!
 * \file plan_update_buffer_allocation_location.cc
 * \brief Place native allocation definitions before their first scoped use.
 */
#include "../op/native_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

#include <vector>

namespace tvm::tl {
namespace {

// Native allocations are executable SSA definitions, rather than entries in
// an SBlock's alloc_buffers. Their enclosing region already gives the legal
// lifetime. Sinking a definition through a loop would turn loop-carried
// storage into fresh storage on each iteration. Only shorten a constant
// allocation's prefix inside its existing lexical sequence here. Physical
// hoisting/packing is subsequently handled by StorageRewrite.
class AllocationLocator : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    SeqStmt sequence = StmtExprMutator::Mutate_(op, mode)
                           .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                           .as_or_throw<SeqStmt>();
    const auto &statements = sequence->seq;
    std::vector<ffi::Array<Stmt>> insertions(statements.size());
    std::vector<bool> moved(statements.size(), false);
    bool changed = false;
    for (size_t i = 0; i < statements.size(); ++i) {
      const auto *binding = statements[i].as<BindObj>();
      if (!CanDelay(binding))
        continue;
      size_t first_use = statements.size();
      for (size_t j = i + 1; j < statements.size(); ++j) {
        // Types are visited at definitions. A tensor used by a loop header,
        // view, pointer alias, shape or layout must precede that entire
        // statement, never just its body.
        for (const Var &var : UndefinedVars(statements[j]))
          if (var.same_as(binding->var))
            first_use = j;
        if (first_use != statements.size() ||
            HasControlBoundary(statements[j])) {
          first_use = j;
          break;
        }
      }
      // Preserve unused allocations and their annotations for their owners.
      if (first_use == statements.size() || first_use <= i + 1)
        continue;
      insertions[first_use].push_back(statements[i]);
      moved[i] = changed = true;
    }
    if (!changed)
      return sequence;
    ffi::Array<Stmt> result;
    for (size_t i = 0; i < statements.size(); ++i) {
      for (const Stmt &binding : insertions[i])
        result.push_back(binding);
      if (!moved[i])
        result.push_back(statements[i]);
    }
    return SeqStmt(result, sequence->loc);
  }

private:
  static bool CanDelay(const BindObj *binding) {
    if (!binding)
      return false;
    const auto *call = binding->value.as<CallObj>();
    if (!call || !call->op.same_as(tirx::alloc_tensor_op()) ||
        !call->args.empty())
      return false;
    auto type = binding->var->ty.as<tirx::TensorType>();
    if (!type || !UndefinedVars(*type).empty())
      return false;
    const auto scope = type.value()->scope.name();
    // Barrier initialization, TMEM, Metal cooperative tensors and backend
    // special storage are coupled to their original owning region.
    if (scope != "local" && scope != "local.var" && scope != "shared" &&
        scope != "shared.dyn")
      return false;
    if (type.value()->layout || type.value()->dtype.IsScalableVector())
      return false;
    for (const PrimExpr &extent : type.value()->shape) {
      const auto *value = extent.as<IntImmObj>();
      if (!value || value->value <= 0)
        return false;
    }
    if (call->attrs) {
      auto attrs = call->attrs.value().as<DictAttrs>();
      if (!attrs || !attrs.value()->dict.empty())
        return false;
    }
    return true;
  }

  static bool HasControlBoundary(const Stmt &statement) {
    bool found = false;
    VisitNativePostOrder(statement, [&](const ffi::ObjectRef &node) {
      if (node.as<ReturnObj>() || node.as<BreakObj>() ||
          node.as<ContinueObj>() || node.as<AssertStmtObj>() ||
          node.as<WhileObj>() || node.as<RegionStmtObj>())
        found = true;
    });
    return found;
  }
};
} // namespace

namespace transform {
tvm::transform::Pass PlanAndUpdateBufferAllocationLocation() {
  auto rewrite = [](tirx::Function function, IRModule,
                    tvm::transform::PassContext) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto locator = ffi::make_object<AllocationLocator>();
    SeqStmt body = function->body;
    function.CopyOnWrite()->body = locator->Mutate(body).ValueOrUnchanged(body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(
      rewrite, 0, "tl.PlanAndUpdateBufferAllocationLocation");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.transform.PlanAndUpdateBufferAllocationLocation",
      PlanAndUpdateBufferAllocationLocation);
}
} // namespace transform
} // namespace tvm::tl
