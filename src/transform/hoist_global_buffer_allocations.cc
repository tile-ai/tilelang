/*!
 * \file hoist_global_buffer_allocations.cc
 * \brief Stage native global workspaces in the host scope of each launch.
 */

#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>

namespace tvm::tl {
namespace {
using VarSet = std::unordered_set<const VarObj *>;

class RemoveHoistedDefinitions : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  explicit RemoveHoistedDefinitions(VarSet variables)
      : variables_(std::move(variables)) {}
  ffi::UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    if (variables_.count(op->var.get()))
      return SeqStmt({}, op->loc);
    return StmtExprMutator::Mutate_(op, mode);
  }

private:
  VarSet variables_;
};

class GlobalAllocationHoister : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  explicit GlobalAllocationHoister(const tirx::Function &function) {
    for (const Var &parameter : function->params) {
      host_variables_.insert(parameter.get());
      // A parameter reference is a use-site leaf; inspect its type explicitly
      // to include shape/stride symbols supplied by the packed ABI.
      for (const Var &symbol : UndefinedVars(parameter->ty))
        host_variables_.insert(symbol.get());
    }
  }

  ffi::UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    if (!prefix_) {
      auto result = StmtExprMutator::Mutate_(op, mode);
      host_variables_.insert(op->var.get());
      return result;
    }
    definitions_.insert_or_assign(op->var.get(), ffi::GetRef<Bind>(op));
    const auto *call = op->value.as<CallObj>();
    auto tensor = op->var.as<tirx::TensorVar>();
    if (!tensor || tensor->scope() != runtime::MemoryScope::Global() || !call ||
        !call->op.same_as(tirx::alloc_tensor_op()))
      return StmtExprMutator::Mutate_(op, mode);
    TVM_FFI_CHECK(call->args.empty(), ValueError)
        << "A global workspace with explicit placement cannot be hoisted";
    EnsureHost(tensor->ty(), op->var);
    if (call->attrs)
      EnsureHost(*call->attrs, op->var);
    prefix_->push_back(ffi::GetRef<Bind>(op));
    hoisted_.insert(op->var.get());
    host_variables_.insert(op->var.get());
    return SeqStmt({}, op->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    const bool launch = op->op.same_as(tirx::launch_thread_op()) ||
                        op->op->name == "tl.launch_kernel" ||
                        op->op->name == "tirx.device_scope" ||
                        op->op->name == "tirx.device_entry";
    if (!prefix_ && launch) {
      ffi::Array<Stmt> prefix;
      prefix_ = &prefix;
      definitions_.clear();
      hoisted_.clear();
      auto region = StmtExprMutator::Mutate_(op, mode).ValueOrUnchanged(
          ffi::GetRef<Stmt>(op));
      prefix_ = nullptr;
      if (prefix.empty())
        return region;
      // Dependencies can precede their workspace in the device body. Remove
      // those original definitions after discovering the entire launch, so
      // each SSA identity still has exactly one definition in the result.
      auto remover = ffi::make_object<RemoveHoistedDefinitions>(hoisted_);
      region = remover->Mutate(region).ValueOrUnchanged(region);
      prefix.push_back(region);
      return SeqStmt(prefix, op->loc);
    }
    auto saved = host_variables_;
    if (!prefix_)
      for (const Var &parameter : op->body_params)
        host_variables_.insert(parameter.get());
    auto result = StmtExprMutator::Mutate_(op, mode);
    if (!prefix_) {
      host_variables_ = std::move(saved);
      for (const Var &variable : op->result_vars)
        host_variables_.insert(variable.get());
    }
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    auto saved = host_variables_;
    if (!prefix_)
      host_variables_.insert(op->var.get());
    auto result = StmtExprMutator::Mutate_(op, mode);
    if (!prefix_)
      host_variables_ = std::move(saved);
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const IfObj *op, InplaceMode mode) final {
    auto saved = host_variables_;
    auto condition =
        Mutate(op->condition, mode).ValueOrUnchanged(op->condition);
    auto then_case =
        Mutate(op->then_case, mode).ValueOrUnchanged(op->then_case);
    if (!prefix_)
      host_variables_ = saved;
    ffi::Optional<SeqStmt> else_case;
    if (op->else_case)
      else_case = Mutate(*op->else_case, mode).ValueOrUnchanged(*op->else_case);
    if (!prefix_)
      host_variables_ = std::move(saved);
    return If(condition, then_case, else_case, op->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const WhileObj *op, InplaceMode mode) final {
    auto saved = host_variables_;
    auto result = StmtExprMutator::Mutate_(op, mode);
    if (!prefix_)
      host_variables_ = std::move(saved);
    return result;
  }

private:
  static bool SafeToStage(ffi::AnyView object) {
    bool safe = true;
    auto divisor = [&safe](const PrimExpr &value) {
      const auto *constant = value.as<IntImmObj>();
      // Exclude division by zero and signed min/-1 overflow. Moving a
      // potentially trapping expression out of a device predicate is unsafe.
      if (!constant || constant->value == 0 || constant->value == -1)
        safe = false;
    };
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        object,
        [&](const ffi::ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
          if (node.as<TensorLoadObj>())
            safe = false;
          // Purity alone does not prove that an opaque call terminates or
          // cannot trap. Host shape algebra is represented by native arithmetic
          // nodes.
          if (node.as<CallObj>())
            safe = false;
          if (const auto *value = node.as<prim::DivObj>())
            divisor(value->b);
          if (const auto *value = node.as<prim::ModObj>())
            divisor(value->b);
          if (const auto *value = node.as<prim::FloorDivObj>())
            divisor(value->b);
          if (const auto *value = node.as<prim::FloorModObj>())
            divisor(value->b);
          return ffi::WalkResult::Advance();
        });
    return safe;
  }

  void EnsureHost(ffi::AnyView object, const Var &workspace) {
    TVM_FFI_CHECK(SafeToStage(object), ValueError)
        << "Global workspace " << workspace->name
        << " has a shape/layout requiring device memory or unsafe evaluation";
    for (const Var &variable : UndefinedVars(object)) {
      if (host_variables_.count(variable.get()))
        continue;
      auto found = definitions_.find(variable.get());
      TVM_FFI_CHECK(found != definitions_.end() &&
                        variable->ty.as<PrimTypeObj>() &&
                        visiting_.insert(variable.get()).second,
                    ValueError)
          << "Global workspace " << workspace->name
          << " depends on device-defined variable " << variable->name;
      Bind binding = found->second;
      EnsureHost(binding->value, workspace);
      EnsureHost(variable->ty, workspace);
      visiting_.erase(variable.get());
      if (hoisted_.insert(variable.get()).second)
        prefix_->push_back(binding);
      host_variables_.insert(variable.get());
    }
  }

  VarSet host_variables_, hoisted_, visiting_;
  std::unordered_map<const VarObj *, Bind> definitions_;
  ffi::Array<Stmt> *prefix_{nullptr};
};
} // namespace

namespace transform {
tvm::transform::Pass HoistGlobalBufferAllocations() {
  auto rewrite = [](tirx::Function function, IRModule,
                    tvm::transform::PassContext) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto hoister = ffi::make_object<GlobalAllocationHoister>(function);
    SeqStmt body = function->body;
    function.CopyOnWrite()->body = hoister->Mutate(body).ValueOrUnchanged(body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(rewrite, 0,
                                             "tl.HoistGlobalBufferAllocations");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.HoistGlobalBufferAllocations",
                                   HoistGlobalBufferAllocations);
}
} // namespace transform
} // namespace tvm::tl
