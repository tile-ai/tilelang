#include "common/reducer.h"
#include "compat/metadata.h"

#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <vector>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

enum class EpochState { kAllocated, kActive, kFinalized };

struct EpochContext {
  const StmtNode *statement;
  int branch;
  bool conditional;
  bool operator==(const EpochContext &) const = default;
};

struct Epoch {
  TensorVar tensor;
  EpochState state = EpochState::kAllocated;
  std::vector<EpochContext> context;
};

class ReducerEpochVerifier : public tvm::tl::StmtExprVisitor {
public:
  void Run(const tirx::Function &function) {
    for (const Var &parameter : function->params) {
      if (auto tensor = parameter.as<TensorVar>()) {
        TVM_FFI_CHECK(!IsReducerV2Buffer(*tensor), ValueError)
            << "Reducer handles must be local allocations, not function "
               "parameters";
      }
    }
    if (function->body)
      Visit(*function->body);
    for (const auto &[tensor, epoch] : epochs_) {
      TVM_FFI_CHECK(epoch.state != EpochState::kAllocated, ValueError)
          << "reducer `" << tensor->name
          << "` is allocated but never initialized";
      TVM_FFI_CHECK(epoch.state == EpochState::kFinalized, ValueError)
          << "reducer `" << tensor->name
          << "` is initialized but never finalized";
    }
  }

private:
  Epoch &Require(const TensorVar &tensor) {
    auto found = epochs_.find(tensor.get());
    TVM_FFI_CHECK(found != epochs_.end(), ValueError)
        << "Reducer operation requires a local allocation with a tl.reducer_op "
           "annotation";
    return found->second;
  }

  Optional<VisitInterrupt> Visit_(const BindNode *node) final {
    auto tensor = node->var.as<TensorVar>();
    const auto *allocation = node->value.as<CallNode>();
    if (tensor && IsReducerV2Buffer(*tensor) && allocation &&
        allocation->op.same_as(alloc_tensor_op())) {
      auto operation = ReducerOperation(allocation);
      TVM_FFI_CHECK(operation, ValueError)
          << "local.reducer allocation needs tl.reducer_op; use "
             "T.alloc_reducer";
      ReducerV2Identity(ParseReducerV2OpType(*operation), tensor->dtype());
      TVM_FFI_CHECK(epochs_.emplace(tensor->get(), Epoch{*tensor}).second,
                    ValueError)
          << "Reducer allocation appears more than once";
      return Visit(node->value);
    }
    return tvm::tl::StmtExprVisitor::Visit_(node);
  }

  Optional<VisitInterrupt> Visit_(const ForNode *node) final {
    Visit(node->begin);
    Visit(node->end);
    Visit(node->step);
    bool parallel = node->kind == ForKind::kParallel;
    parallel_depth_ += parallel;
    context_.push_back({node, 0, false});
    Visit(node->body);
    context_.pop_back();
    parallel_depth_ -= parallel;
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const RegionStmtNode *node) final {
    for (const Expr &argument : node->args)
      Visit(argument);
    return Visit(node->body);
  }

  Optional<VisitInterrupt> Visit_(const IfNode *node) final {
    Visit(node->condition);
    context_.push_back({node, 0, true});
    Visit(node->then_case);
    context_.pop_back();
    if (node->else_case) {
      context_.push_back({node, 1, true});
      Visit(*node->else_case);
      context_.pop_back();
    }
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const WhileNode *node) final {
    Visit(node->condition);
    context_.push_back({node, 0, false});
    Visit(node->body);
    context_.pop_back();
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const VarNode *node) final {
    if (auto tensor = GetRef<Var>(node).as<TensorVar>()) {
      TVM_FFI_CHECK(!IsReducerV2Buffer(*tensor), ValueError)
          << "Ordinary access or aliasing of reducer `" << node->name
          << "` is forbidden; use reducer_init, reducer_update and "
             "finalize_reducer";
    }
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const CallNode *node) final {
    if (node->op.same_as(Op::Get("tl.layout_hint")))
      return std::nullopt;
    if (node->op.same_as(ReducerInitOp::Get())) {
      ReducerInitOp operation(
          node->args,
          node->attrs.value_or(DictAttrs()).as_or_throw<DictAttrs>()->dict);
      Epoch &epoch = Require(operation->reducer);
      TVM_FFI_CHECK(parallel_depth_ == 0, ValueError)
          << "reducer_init may not appear inside a T.Parallel loop";
      TVM_FFI_CHECK(epoch.state == EpochState::kAllocated, ValueError)
          << "double reducer_init; each allocation supports exactly one epoch";
      epoch.state = EpochState::kActive;
      epoch.context = context_;
      if (operation->seed)
        Visit(*operation->seed);
      return std::nullopt;
    }
    if (node->op.same_as(reducer_update())) {
      auto update = ParseReducerUpdate(node);
      Epoch &epoch = Require(update.reducer);
      TVM_FFI_CHECK(epoch.state == EpochState::kActive, ValueError)
          << "reducer_update must appear after reducer_init and before "
             "finalize_reducer";
      TVM_FFI_CHECK(parallel_depth_ > 0, ValueError)
          << "reducer_update outside any T.Parallel loop is ambiguous";
      for (const PrimExpr &index : update.indices)
        Visit(index);
      return Visit(update.value);
    }
    if (node->op.same_as(FinalizeReducerV2Op::Get())) {
      FinalizeReducerV2Op operation(
          node->args,
          node->attrs.value_or(DictAttrs()).as_or_throw<DictAttrs>()->dict);
      Epoch &epoch = Require(operation->reducer);
      TVM_FFI_CHECK(parallel_depth_ == 0, ValueError)
          << "finalize_reducer may not appear inside a T.Parallel loop";
      TVM_FFI_CHECK(epoch.state == EpochState::kActive, ValueError)
          << "finalize_reducer requires one preceding init and may appear only "
             "once";
      TVM_FFI_CHECK(context_.size() >= epoch.context.size(), ValueError)
          << "finalize_reducer must remain in the init's control-flow scope";
      for (size_t index = 0; index < context_.size(); ++index) {
        TVM_FFI_CHECK(index < epoch.context.size()
                          ? context_[index] == epoch.context[index]
                          : context_[index].conditional,
                      ValueError)
            << "finalize_reducer must be in the same iteration and branch as "
               "reducer_init";
      }
      epoch.state = EpochState::kFinalized;
      return Visit(operation->dst);
    }
    return tvm::tl::StmtExprVisitor::Visit_(node);
  }

  std::unordered_map<const VarNode *, Epoch> epochs_;
  std::vector<EpochContext> context_;
  int parallel_depth_ = 0;
};
} // namespace

tvm::transform::Pass VerifyReducerEpoch() {
  return tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext) -> Optional<tirx::Function> {
        ffi::make_object<ReducerEpochVerifier>()->Run(function);
        return function;
      },
      0, "tl.VerifyReducerEpoch");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.VerifyReducerEpoch",
                                   VerifyReducerEpoch);
}
} // namespace tvm::tl
