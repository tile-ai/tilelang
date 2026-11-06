/*!
 * \file verify_reducer_epoch.cc
 * \brief Verify native reducer epochs, control-flow scopes and opaque handles.
 *
 * Each local.reducer allocation owns one static init/finalize pair. Updates
 * occur between them inside parallel loops. Finalize must be in the init's
 * control-flow scope or a conditional refinement, never a repeated loop. The
 * target TensorLoad of reducer_update is a descriptor; every other reducer
 * load, store, pointer conversion, alias, or unsupported use is rejected.
 */
#include "ir/annotations.h"
#include "reducer_metadata.h"

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <vector>

namespace tvm::tl {
namespace {
using namespace tirx;
using namespace prim;
using namespace ffi;
using reducer::Metadata;

enum class EpochState { kAllocated, kActive, kFinalized };

class ReducerEpochVerifier : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;

  void Run(const tirx::Function &function) {
    metadata_ = reducer::MetadataIndex::Collect(function);
    for (const auto &[tensor, allocation] : metadata_.allocations) {
      if (!IsReducerV2Buffer(tensor))
        continue;
      auto info = metadata_.v2.Get(tensor);
      TVM_FFI_CHECK(info.has_value(), ValueError)
          << "Reducer " << tensor
          << " has no reducer_info_v2 annotation; use T.alloc_reducer";
      auto op = info->Get("op");
      TVM_FFI_CHECK(op.has_value(), ValueError)
          << "reducer_info_v2 is missing the combine op";
      ParseReducerV2OpType(op->as_or_throw<String>());
      states_.emplace(tensor.get(), EpochState::kAllocated);
    }
    for (const auto &[tensor, info] : metadata_.v2)
      TVM_FFI_CHECK(states_.count(tensor.get()), ValueError)
          << "reducer_info_v2 must identify an allocated local.reducer tensor: "
          << tensor;
    if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      Visit(function->body);
    for (const auto &[tensor, state] : states_) {
      TVM_FFI_CHECK(state != EpochState::kAllocated, ValueError)
          << "reducer " << GetRef<Var>(tensor)
          << " is allocated but never initialized; every T.alloc_reducer needs "
             "exactly one T.reducer_init";
      TVM_FFI_CHECK(state != EpochState::kActive, ValueError)
          << "reducer " << GetRef<Var>(tensor)
          << " is initialized but never finalized; call T.finalize_reducer "
             "exactly once";
    }
  }

  Optional<VisitInterrupt> Visit_(const BindObj *op) final {
    if (states_.count(op->var.get())) {
      const auto *value = op->value.as<CallObj>();
      TVM_FFI_CHECK(value && value->op.same_as(alloc_tensor_op()), ValueError)
          << "A reducer handle cannot be aliased; allocate it with "
             "T.alloc_reducer";
      // Do not visit the handle definition as a runtime use.
      for (const Expr &arg : value->args)
        Visit(arg);
      return std::nullopt;
    }
    return StmtExprVisitor::Visit_(op);
  }

  Optional<VisitInterrupt> Visit_(const RegionStmtObj *op) final {
    if (op->op->name == "tl.attr") {
      auto fields = ReadAnnotations(op);
      auto key = fields.at("key").as_or_throw<String>();
      if (key == "tl.kernel")
        VisitMetadata(reducer::StringMap(fields.at("value")));
      else if (!reducer::IsReducerMetadataKey(key))
        Visit(fields.at("value"));
      Visit(op->body);
      return std::nullopt;
    }
    return StmtExprVisitor::Visit_(op);
  }

  Optional<VisitInterrupt> Visit_(const ForObj *op) final {
    Visit(op->begin);
    Visit(op->end);
    Visit(op->step);
    const bool parallel = op->kind == ForKind::kParallel;
    parallel_depth_ += parallel;
    context_.push_back({op, 0, false});
    Visit(op->body);
    context_.pop_back();
    parallel_depth_ -= parallel;
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const IfObj *op) final {
    Visit(op->condition);
    context_.push_back({op, 0, true});
    Visit(op->then_case);
    context_.pop_back();
    if (op->else_case) {
      context_.push_back({op, 1, true});
      Visit(op->else_case.value());
      context_.pop_back();
    }
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const WhileObj *op) final {
    Visit(op->condition);
    context_.push_back({op, 0, false});
    Visit(op->body);
    context_.pop_back();
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    if (reducer::IsMetadataCall(op)) {
      VisitMetadata(ReadAnnotations(op));
      return std::nullopt;
    }
    if (op->op.same_as(ReducerInitOp::Get())) {
      const auto args = DecodeTileOpArguments(op).operands;
      TVM_FFI_CHECK(args.size() == 1 || args.size() == 2, ValueError);
      auto tensor = tl::RegionTensor(NormalizeToTensorRegion(args[0]));
      auto &state = RequireReducer(tensor, "reducer_init");
      TVM_FFI_CHECK(parallel_depth_ == 0, ValueError)
          << "T.reducer_init may not appear inside a T.Parallel loop";
      TVM_FFI_CHECK(state == EpochState::kAllocated, ValueError)
          << "double T.reducer_init; each allocation supports exactly one "
             "epoch";
      if (args.size() == 2) {
        TVM_FFI_CHECK(args[1].as_or_throw<PrimExpr>().ty() == tensor.dtype(),
                      TypeError)
            << "T.reducer_init seed dtype does not match reducer";
        Visit(args[1]);
      }
      epoch_context_[tensor.get()] = context_;
      state = EpochState::kActive;
      return std::nullopt;
    }
    if (op->op.same_as(reducer_update())) {
      auto update = ParseReducerUpdate(op);
      auto &state = RequireReducer(update.reducer, "reducer_update");
      TVM_FFI_CHECK(state != EpochState::kAllocated, ValueError)
          << "T.reducer_update before T.reducer_init";
      TVM_FFI_CHECK(state != EpochState::kFinalized, ValueError)
          << "T.reducer_update after T.finalize_reducer";
      TVM_FFI_CHECK(parallel_depth_ > 0, ValueError)
          << "T.reducer_update outside any T.Parallel loop has ambiguous "
             "contribution multiplicity";
      for (const auto &index : update.indices)
        Visit(index);
      Visit(update.value);
      return std::nullopt;
    }
    if (op->op.same_as(FinalizeReducerV2Op::Get())) {
      const auto args = DecodeTileOpArguments(op).operands;
      TVM_FFI_CHECK_EQ(args.size(), 2U, ValueError);
      auto tensor = tl::RegionTensor(NormalizeToTensorRegion(args[0]));
      auto destination = tl::RegionTensor(NormalizeToTensorRegion(args[1]));
      auto &state = RequireReducer(tensor, "finalize_reducer");
      TVM_FFI_CHECK(parallel_depth_ == 0, ValueError)
          << "T.finalize_reducer may not appear inside a T.Parallel loop";
      TVM_FFI_CHECK(!IsReducerV2Buffer(destination), ValueError)
          << "T.finalize_reducer destination must be an ordinary tensor, not a "
             "reducer";
      TVM_FFI_CHECK(state != EpochState::kAllocated, ValueError)
          << "T.finalize_reducer before T.reducer_init";
      TVM_FFI_CHECK(state != EpochState::kFinalized, ValueError)
          << "double T.finalize_reducer";
      RequireConditionalRefinement(epoch_context_.at(tensor.get()));
      state = EpochState::kFinalized;
      return std::nullopt;
    }
    return StmtExprVisitor::Visit_(op);
  }

  Optional<VisitInterrupt> Visit_(const TensorLoadObj *op) final {
    auto tensor = op->source.as_or_throw<TensorVar>();
    TVM_FFI_CHECK(!states_.count(tensor.get()), ValueError)
        << "illegal read of reducer " << tensor
        << "; read the finalize destination instead";
    return StmtExprVisitor::Visit_(op);
  }

  Optional<VisitInterrupt> Visit_(const TensorStoreObj *op) final {
    auto tensor = op->dest.as_or_throw<TensorVar>();
    TVM_FFI_CHECK(!states_.count(tensor.get()), ValueError)
        << "illegal store to reducer " << tensor << "; use T.reducer_update";
    return StmtExprVisitor::Visit_(op);
  }

  Optional<VisitInterrupt> Visit_(const VarObj *op) final {
    TVM_FFI_CHECK(!states_.count(op), ValueError)
        << "illegal use of reducer handle " << op->name
        << " (address-of / aliasing / unsupported op)";
    return StmtExprVisitor::Visit_(op);
  }

private:
  struct Frame {
    const Object *node;
    int branch;
    bool conditional;
    bool operator==(const Frame &) const = default;
  };

  void VisitMetadata(const Metadata &metadata) {
    for (const auto &[key, value] : metadata)
      if (!reducer::IsReducerMetadataKey(key))
        Visit(value);
  }

  EpochState &RequireReducer(const TensorVar &tensor, const char *operation) {
    auto found = states_.find(tensor.get());
    TVM_FFI_CHECK(found != states_.end(), ValueError)
        << operation
        << ": tensor is not an allocated reducer; use T.alloc_reducer";
    return found->second;
  }

  void RequireConditionalRefinement(const std::vector<Frame> &initial) const {
    bool valid = initial.size() <= context_.size();
    for (size_t i = 0; valid && i < context_.size(); ++i)
      valid = i < initial.size() ? initial[i] == context_[i]
                                 : context_[i].conditional;
    TVM_FFI_CHECK(valid, ValueError)
        << "T.finalize_reducer is not in the scope of its T.reducer_init; "
           "close in the same loop iteration/branch or a nested conditional";
  }

  reducer::MetadataIndex metadata_;
  std::unordered_map<const VarObj *, EpochState> states_;
  std::unordered_map<const VarObj *, std::vector<Frame>> epoch_context_;
  std::vector<Frame> context_;
  int parallel_depth_ = 0;
};
} // namespace

tvm::transform::Pass VerifyReducerEpoch() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false)) {
      auto verifier = ffi::make_object<ReducerEpochVerifier>();
      verifier->Run(function);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.VerifyReducerEpoch");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.VerifyReducerEpoch",
                                   VerifyReducerEpoch);
}
} // namespace tvm::tl
