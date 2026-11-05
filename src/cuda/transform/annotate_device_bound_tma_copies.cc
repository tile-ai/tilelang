/*!
 * \file annotate_device_bound_tma_copies.cc
 * \brief Preserve native copy base-pointer provenance before let inlining.
 */

#include "cuda/op/builtin.h"
#include "ir/annotations.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

bool IsDeviceRegion(const RegionStmtObj *region) {
  return region->op.same_as(launch_thread_op()) ||
         region->op->name == "tl.launch_kernel" ||
         region->op->name == "tirx.device_scope" ||
         region->op->name == "tirx.device_entry";
}

class BodyValues : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;

  explicit BodyValues(const tirx::Function &function) {
    for (const Var &parameter : function->params) {
      host_values.insert(parameter.get());
      for (const Var &symbol : UndefinedVars(parameter->ty))
        host_values.insert(symbol.get());
    }
  }

  Optional<VisitInterrupt> Visit_(const RegionStmtObj *region) final {
    bool previous = device_;
    device_ = device_ || IsDeviceRegion(region);
    auto result = StmtExprVisitor::Visit_(region);
    device_ = previous;
    return result;
  }

  Optional<VisitInterrupt> Visit_(const BindObj *binding) final {
    bindings.Set(binding->var, binding->value);
    return StmtExprVisitor::Visit_(binding);
  }

  Optional<VisitInterrupt> Visit_(const VarObj *variable) final {
    if (!device_ && def_region_kind() != kTVMFFIDefRegionKindNone)
      host_values.insert(variable);
    return StmtExprVisitor::Visit_(variable);
  }

  Map<Var, Expr> bindings;
  std::unordered_set<const VarObj *> host_values;

private:
  bool device_{false};
};

// Taking an address does not read the tensor lvalue. Other memory accesses
// and stateful calls make the base genuinely device-bound.
class BaseExpressionPurity : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;

  Optional<VisitInterrupt> Visit_(const TensorLoadObj *) final {
    pure = false;
    return VisitInterrupt();
  }

  Optional<VisitInterrupt> Visit_(const CallObj *call) final {
    static auto effects = Op::GetAttrMap(tvm::op_attr::kCallEffectKind);
    auto op = call->op.as<Op>();
    if (!op ||
        effects.get(*op, CallEffectKind::kOpaque) > CallEffectKind::kPure) {
      pure = false;
      return VisitInterrupt();
    }
    bool address = call->op.same_as(address_of_op()) ||
                   op.value()->name == "tl.access_ptr";
    for (size_t i = 0; i < call->args.size(); ++i) {
      if (address && i == 0) {
        if (const auto *load = call->args[i].as<TensorLoadObj>()) {
          Visit(load->source);
          Visit(load->indices);
          if (!pure)
            return VisitInterrupt();
          continue;
        }
      }
      Visit(call->args[i]);
      if (!pure)
        return VisitInterrupt();
    }
    return std::nullopt;
  }

  bool pure{true};
};

class HostAvailability {
public:
  explicit HostAvailability(const BodyValues &values) : values_(values) {}

  bool Available(const Var &variable) {
    if (values_.host_values.count(variable.get()))
      return true;
    auto known = available_.find(variable.get());
    if (known != available_.end())
      return known->second;
    // Protect the traversal from malformed cyclic definitions.
    available_[variable.get()] = false;
    auto binding = values_.bindings.Get(variable);
    if (!binding)
      return false;
    const auto *call = binding.value().as<CallObj>();
    bool result = false;
    if (call && call->op.same_as(alloc_tensor_op())) {
      auto tensor = variable.as<TensorVar>();
      // Global workspaces are staged by their owning pass. A TensorVar Bind
      // is not an old-style device-computed handle; type dependencies still
      // have to be expressible using host values.
      result = tensor && tensor->scope() == runtime::MemoryScope::Global() &&
               call->args.empty() && VariablesAvailable(variable->ty);
    } else {
      result = ExpressionAvailable(binding.value()) &&
               VariablesAvailable(variable->ty);
    }
    available_[variable.get()] = result;
    return result;
  }

private:
  bool VariablesAvailable(AnyView object) {
    for (const Var &variable : UndefinedVars(object))
      if (!Available(variable))
        return false;
    return true;
  }

  bool ExpressionAvailable(const Expr &expression) {
    auto checker = make_object<BaseExpressionPurity>();
    checker->Visit(expression);
    return checker->pure && VariablesAvailable(expression);
  }

  const BodyValues &values_;
  std::unordered_map<const VarObj *, bool> available_;
};

TensorVar CopyTensor(const Expr &operand) {
  if (auto tensor = operand.as<TensorVar>())
    return tensor.value();
  if (const auto *region = operand.as<TensorRegionObj>())
    return region->source.as_or_throw<TensorVar>();
  if (const auto *load = operand.as<TensorLoadObj>())
    return load->source.as_or_throw<TensorVar>();
  TVM_FFI_THROW(TypeError) << "A native copy requires TensorVar, TensorRegion "
                              "or TensorLoad operands";
}

class DeviceBoundCopyAnnotator : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  explicit DeviceBoundCopyAnnotator(const BodyValues &values)
      : availability_(values) {}

  UnchangedOr<Expr> Mutate_(const CallObj *call, InplaceMode mode) final {
    Call rewritten = StmtExprMutator::Mutate_(call, mode)
                         .ValueOrUnchanged(GetRef<Expr>(call))
                         .as_or_throw<Call>();
    auto op = rewritten->op.as<Op>();
    if (!op)
      return rewritten;
    const String name = op.value()->name;
    if (name != "tl.tileop.copy" && name != "tl.tileop.async_copy" &&
        name != "tl.tileop.tma_copy")
      return rewritten;
    auto arguments = DecodeTileOpArguments(rewritten.get());
    TVM_FFI_CHECK_EQ(arguments.operands.size(), 2U, ValueError);
    if (arguments.annotations.count(attr::kTmaDescriptorBaseIsDeviceBound))
      return rewritten;
    bool device_bound = false;
    for (const Expr &operand : arguments.operands) {
      TensorVar tensor = CopyTensor(operand);
      if (tensor.scope() == runtime::MemoryScope::Global() &&
          !availability_.Available(tensor))
        device_bound = true;
    }
    if (!device_bound)
      return rewritten;
    arguments.annotations.Set(attr::kTmaDescriptorBaseIsDeviceBound,
                              IntImm(PrimType::Int(32), 1));
    return MakeAnnotatedCall(rewritten->ty, op.value(), arguments.operands,
                             arguments.annotations, rewritten->loc);
  }

private:
  HostAvailability availability_;
};
} // namespace

tvm::transform::Pass AnnotateDeviceBoundTmaCopies() {
  auto rewrite = [](tirx::Function function, IRModule,
                    tvm::transform::PassContext) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto values = ffi::make_object<BodyValues>(function);
    values->Visit(function->body);
    auto annotator = ffi::make_object<DeviceBoundCopyAnnotator>(*values);
    SeqStmt body = function->body;
    function.CopyOnWrite()->body =
        annotator->Mutate(body).ValueOrUnchanged(body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(rewrite, 0,
                                             "tl.AnnotateDeviceBoundTmaCopies");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.cuda.transform.AnnotateDeviceBoundTmaCopies",
      AnnotateDeviceBoundTmaCopies);
}
} // namespace tvm::tl
