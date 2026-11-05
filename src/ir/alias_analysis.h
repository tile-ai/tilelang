/*!
 * \file tl/ir/alias_analysis.h
 * \brief Resolve native tensor and pointer definitions to physical identities.
 */
#ifndef TILELANG_IR_ALIAS_ANALYSIS_H_
#define TILELANG_IR_ALIAS_ANALYSIS_H_

#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/memory.h>
#include <unordered_set>

namespace tvm::tl {

// This only identifies storage, not byte ranges or non-overlap. Unknown calls
// never manufacture independent roots; clients must conservatively handle a
// missing answer as possibly aliasing every externally supplied pointer.
class PhysicalRootAnalysis {
public:
  explicit PhysicalRootAnalysis(const tirx::Function &function) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return;
    ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
        function->body,
        [&](const BindObj *bind) -> ffi::Expected<ffi::WalkResult> {
          if (bind->var->ty.as<PtrTypeObj>() ||
              bind->var->ty.as<tirx::TensorTypeObj>())
            definitions_.Set(bind->var, bind->value);
          return ffi::WalkResult::Advance();
        });
  }

  ffi::Optional<Var> Root(const Expr &expression) const {
    std::unordered_set<const VarObj *> active;
    return Root(expression, &active);
  }

private:
  ffi::Optional<Var> Root(const Expr &expression,
                          std::unordered_set<const VarObj *> *active) const {
    if (auto var = expression.as<Var>()) {
      if (!var.value()->ty.as<PtrTypeObj>() &&
          !var.value()->ty.as<tirx::TensorTypeObj>())
        return std::nullopt;
      auto definition = definitions_.Get(var.value());
      if (!definition)
        return var;
      if (!active->insert(var.value().get()).second)
        return std::nullopt;
      const auto *call = definition.value().as<CallObj>();
      ffi::Optional<Var> root;
      if (call && call->op.same_as(tirx::alloc_tensor_op()))
        root = var;
      else
        root = Root(definition.value(), active);
      active->erase(var.value().get());
      return root;
    }
    if (const auto *load = expression.as<TensorLoadObj>())
      return Root(load->source, active);
    if (const auto *call = expression.as<CallObj>()) {
      if (call->op.same_as(tirx::decl_tensor_op()) ||
          call->op.same_as(tirx::tensor_data_ptr_op()) ||
          call->op.same_as(tirx::address_of_op()) ||
          call->op.same_as(tirx::ptr_byte_offset_op()) ||
          call->op.same_as(tirx::reinterpret_op())) {
        if (!call->args.empty())
          return Root(call->args[0], active);
      }
      const auto *op = call->op.as<OpObj>();
      if (op && (op->name == "tl.access_ptr" || op->name == "tl.ptr_access") &&
          !call->args.empty())
        return Root(call->args[0], active);
    }
    return std::nullopt;
  }
  ffi::Map<Var, Expr> definitions_;
};

// Move alias exceptions into explicit operands of device regions before
// native splitting creates fresh pointer parameters.
tirx::Function StageNonRestrictForSplit(tirx::Function function);

} // namespace tvm::tl
#endif // TILELANG_IR_ALIAS_ANALYSIS_H_
