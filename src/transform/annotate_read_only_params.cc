#include <tvm/ir/op.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/type.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "common/read_only_params.h"
#include "op/builtin.h"
#include "op/operator.h"
#include "op/parallel.h"

namespace tvm::tl {
namespace {

using ParameterSet = std::unordered_set<const VarNode *>;
using AliasMap = std::unordered_map<const VarNode *, ParameterSet>;

bool IsAddress(const Expr &expression) {
  return expression->ty.as<PtrTypeNode>() ||
         expression->ty.as<tirx::TensorTypeNode>() ||
         expression.as<TensorRegionNode>();
}

class ParameterOrigins : public tirx::StmtExprVisitor {
public:
  explicit ParameterOrigins(const AliasMap &aliases) : aliases_(aliases) {}

  ParameterSet Collect(const Expr &expression) {
    Visit(expression);
    return std::move(origins_);
  }

private:
  ffi::Optional<ffi::VisitInterrupt> Visit_(const VarNode *variable) final {
    if (auto found = aliases_.find(variable); found != aliases_.end()) {
      origins_.insert(found->second.begin(), found->second.end());
    }
    return std::nullopt;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const TensorLoadNode *) final {
    return std::nullopt;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const CallNode *call) final {
    auto operation = call->op.as<Op>();
    if (operation &&
        (operation.value()->name == "tirx.address_of" ||
         operation.value()->name == "tl.access_ptr") &&
        !call->args.empty()) {
      if (const auto *load = call->args[0].as<TensorLoadNode>()) {
        return Visit(load->source);
      }
    }
    return StmtExprVisitor::Visit_(call);
  }

  const AliasMap &aliases_;
  ParameterSet origins_;
};

class ReadWriteMarker : public tirx::StmtExprVisitor {
public:
  explicit ReadWriteMarker(const ffi::Array<Var> &parameters) {
    for (const Var &parameter : parameters) {
      if (IsAddress(parameter)) {
        aliases_.emplace(parameter.get(), ParameterSet{parameter.get()});
      }
    }
  }

  const ParameterSet &written() const { return written_; }

private:
  ParameterSet Origins(const Expr &expression) const {
    return ffi::make_object<ParameterOrigins>(aliases_)->Collect(expression);
  }

  void MarkWritten(const Expr &expression) {
    ParameterSet origins = Origins(expression);
    written_.insert(origins.begin(), origins.end());
  }

  bool HasReadOnlyAccessContract(const Expr &expression) const {
    if (const auto *variable = expression.as<VarNode>())
      return read_only_pointers_.count(variable) != 0;
    auto call = expression.as<CallNode>();
    if (!call || !expression->ty.as<PtrTypeNode>())
      return false;
    ffi::Optional<PrimExpr> mask;
    if (call->op.same_as(access_ptr()) && call->args.size() == 3) {
      mask = call->args[2].as<PrimExpr>();
    } else if (call->op.same_as(tirx::address_of_op()) && call->attrs) {
      auto attrs = call->attrs.value().as<DictAttrs>();
      if (attrs)
        mask = attrs.value().GetAttr<PrimExpr>(attr::kAccessMask);
    }
    if (!mask)
      return false;
    auto value = GetConstInt(mask.value());
    return value && value.value() >= 0 && value.value() <= kAccessReadWrite &&
           (value.value() & kAccessWrite) == 0;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const BindNode *binding) final {
    auto interrupted = StmtExprVisitor::Visit_(binding);
    if (!interrupted.has_value() && IsAddress(binding->var)) {
      aliases_[binding->var.get()] = Origins(binding->value);
      if (HasReadOnlyAccessContract(binding->value))
        read_only_pointers_.insert(binding->var.get());
    }
    return interrupted;
  }

  ffi::Optional<ffi::VisitInterrupt>
  Visit_(const TensorStoreNode *statement) final {
    MarkWritten(statement->dest);
    return StmtExprVisitor::Visit_(statement);
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const CallNode *call) final {
    if (auto operation = ParseOperator(ffi::GetRef<Call>(call));
        operation.defined()) {
      if (const auto *parallel = operation.as<ParallelOpNode>())
        Visit(parallel->GetRoot());
      else
        for (const TensorRegion &region : operation->GetAccessRegions().writes)
          MarkWritten(region->source);
      return StmtExprVisitor::Visit_(call);
    }
    static auto effects = Op::GetAttrMap(tvm::op_attr::kCallEffectKind);
    if (effects.get(call->op, CallEffectKind::kOpaque) >=
        CallEffectKind::kUpdateState) {
      for (const Expr &argument : call->args) {
        if (!HasReadOnlyAccessContract(argument))
          MarkWritten(argument);
      }
    }
    return StmtExprVisitor::Visit_(call);
  }

  AliasMap aliases_;
  std::unordered_set<const VarNode *> read_only_pointers_;
  ParameterSet written_;
};

} // namespace

ffi::Array<IntImm> ReadOnlyParameterIndices(const tirx::Function &function) {
  if (!function->body.has_value()) {
    return {};
  }
  auto marker = ffi::make_object<ReadWriteMarker>(function->params);
  marker->Visit(function->body.value());
  ffi::Array<IntImm> indices;
  for (size_t index = 0; index < function->params.size(); ++index) {
    const Var &parameter = function->params[index];
    if (IsAddress(parameter) && !marker->written().count(parameter.get())) {
      indices.push_back(IntImm::Int32(static_cast<int32_t>(index)));
    }
  }
  return indices;
}

namespace transform {

tvm::transform::Pass AnnotateReadOnlyParams() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    if (!function->body)
      return function;
    auto indices = ReadOnlyParameterIndices(function);
    return WithAttr(std::move(function), "tl.readonly_param_indices", indices);
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.AnnotateReadOnlyParams");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.AnnotateReadOnlyParams",
                                   AnnotateReadOnlyParams);
}

} // namespace transform
} // namespace tvm::tl
