/*!
 * \file annotate_read_only_params.cc
 * \brief Preserve const-parameter contracts over native pointer and tensor
 * views.
 */
#include "../ir/alias_analysis.h"
#include "../op/native_intrinsic.h"
#include "../op/native_utils.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_set>

namespace tvm::tl {
namespace {

tirx::Function MarkReadOnly(tirx::Function function) {
  if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return function;
  PhysicalRootAnalysis aliases(function);
  std::unordered_set<const VarObj *> parameters, written, non_restrict;
  for (const Var &parameter : function->params) {
    if (parameter->ty.as<PtrTypeObj>() ||
        parameter->ty.as<tirx::TensorTypeObj>())
      parameters.insert(parameter.get());
  }
  if (auto exceptions =
          function->GetAttr<ffi::Array<Var>>("tl.non_restrict_params")) {
    for (const Var &value : exceptions.value()) {
      if (auto root = aliases.Root(value))
        non_restrict.insert(root.value().get());
    }
  }
  auto mark = [&](const Expr &expression) {
    auto root = aliases.Root(expression);
    if (!root)
      written.insert(parameters.begin(), parameters.end());
    else if (parameters.count(root.value().get()))
      written.insert(root.value().get());
  };
  VisitNativePostOrder(function->body, [&](const ffi::ObjectRef &object) {
    if (const auto *store = object.as<TensorStoreObj>())
      mark(store->dest);
    if (const auto *call = object.as<CallObj>()) {
      // Creating local storage or an explicit alias does not modify inputs.
      if (call->op.same_as(tirx::alloc_tensor_op()) ||
          call->op.same_as(tirx::decl_tensor_op()))
        return;
      if (tvm::SideEffect(ffi::GetRef<Expr>(call)) <=
          CallEffectKind::kReadState)
        return;
      if (const auto *op = call->op.as<OpObj>()) {
        // These cooperative operations have explicit write operands. A load
        // updates its fragment, a store updates its memory pointer, and MMA
        // updates only its first fragment. Their other operands stay
        // readable through const input pointers in Metal source. Nested
        // operand calls are visited separately, so this cannot hide an
        // opaque producer.
        int destination = -1;
        if (op->name == "tl.cooperative_tensor_load" ||
            op->name == "tl.cooperative_tensor_fill" ||
            op->name == "tl.cooperative_tensor_multiply_accumulate")
          destination = 0;
        else if (op->name == "tl.cooperative_tensor_store")
          destination = 2;
        if (destination >= 0) {
          const auto operands = NativeIntrinsicOperands(call);
          TVM_FFI_ICHECK_LT(static_cast<size_t>(destination), operands.size());
          mark(operands[destination]);
          return;
        }
      }
      bool pointer_operand = false;
      for (const Expr &operand : call->args) {
        if (operand->ty.as<PtrTypeObj>() ||
            operand->ty.as<tirx::TensorTypeObj>()) {
          pointer_operand = true;
          mark(operand);
        }
      }
      // An opaque call can write through a previously escaped pointer even
      // when its current signature has no direct pointer operand.
      if (!pointer_operand)
        written.insert(parameters.begin(), parameters.end());
    }
  });
  const bool noalias = function->GetAttr(tirx::attr::kNoAlias).value_or(false);
  if (!noalias && !written.empty())
    written.insert(parameters.begin(), parameters.end());
  if (noalias) {
    bool aliased_write = false;
    for (const auto *parameter : non_restrict)
      aliased_write |= written.count(parameter) != 0;
    if (aliased_write)
      written.insert(non_restrict.begin(), non_restrict.end());
  }
  ffi::Array<int64_t> readonly;
  for (size_t i = 0; i < function->params.size(); ++i) {
    const auto *variable = function->params[i].get();
    if (parameters.count(variable) && !written.count(variable))
      readonly.push_back(i);
  }
  function = WithoutAttr(function, "tl.readonly_param_indices");
  if (!readonly.empty())
    function = WithAttr(function, "tl.readonly_param_indices", readonly);
  return function;
}
} // namespace

namespace transform {
tvm::transform::Pass AnnotateReadOnlyParams() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    return MarkReadOnly(function);
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.AnnotateReadOnlyParams");
}
} // namespace transform
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.AnnotateReadOnlyParams",
                                   transform::AnnotateReadOnlyParams);
}
} // namespace tvm::tl
