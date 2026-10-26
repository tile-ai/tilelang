#include "common/reducer.h"
#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "op/prim.h"

#include "compat/visit.h"
#include "op/fill.h"

#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

enum class LegacyPhase { kAllocated, kActive, kFinalized };

struct LegacyReducer {
  LegacyReducer(TensorVar tensor, ReducerV2OpType operation)
      : original(tensor),
        accumulator(ReducerTensor(tensor, tensor->name, "local.reducer")),
        destination(
            ReducerTensor(tensor, tensor->name + "_result", "local.fragment")),
        operation(operation) {}
  TensorVar original;
  TensorVar accumulator;
  TensorVar destination;
  ReducerV2OpType operation;
  LegacyPhase phase = LegacyPhase::kAllocated;
};

class LegacyReducerCanonicalizer : public tvm::tl::StmtExprMutator {
public:
  tirx::Function Run(tirx::Function function) {
    if (!function->body)
      return function;
    PostOrderVisit(*function->body, [&](const ObjectRef &object) {
      const auto *call = object.as<CallNode>();
      if (call && (call->op.same_as(ReducerInitOp::Get()) ||
                   call->op.same_as(reducer_update()) ||
                   call->op.same_as(FinalizeReducerV2Op::Get()))) {
        first_class_.insert(ReducerRegionTensor(call->args[0]).get());
      }
    });
    SeqStmt body = *function->body;
    function.CopyOnWrite()->body = Mutate(body).ValueOrUnchanged(body);
    for (const auto &[tensor, reducer] : reducers_) {
      TVM_FFI_CHECK(reducer.phase == LegacyPhase::kFinalized, ValueError)
          << "legacy reducer `" << tensor->name << "` has no finalize_reducer";
    }
    return function;
  }

private:
  UnchangedOr<Stmt> Mutate_(const BindNode *node, InplaceMode mode) final {
    auto tensor = node->var.as<TensorVar>();
    const auto *allocation = node->value.as<CallNode>();
    auto operation = ReducerOperation(allocation);
    if (!tensor || !allocation || !allocation->op.same_as(alloc_tensor_op()) ||
        !operation || first_class_.count(tensor->get())) {
      return tvm::tl::StmtExprMutator::Mutate_(node, mode);
    }
    auto [position, inserted] = reducers_.emplace(
        tensor->get(),
        LegacyReducer(*tensor, ParseReducerV2OpType(*operation)));
    TVM_FFI_CHECK(inserted, ValueError)
        << "Legacy reducer allocation appears more than once";
    const LegacyReducer &reducer = position->second;
    auto attributes = allocation->attrs.value().as_or_throw<DictAttrs>();
    attributes.CopyOnWrite()->dict.erase(kLegacyReducer);
    return SeqStmt({AllocateReducerTensor(reducer.accumulator, attributes),
                    AllocateReducerTensor(reducer.destination)});
  }

  LegacyReducer *Find(const Expr &expression) {
    TensorVar tensor = ReducerRegionTensor(expression);
    auto found = reducers_.find(tensor.get());
    return found == reducers_.end() ? nullptr : &found->second;
  }

  bool SameIndices(const Array<PrimExpr> &first,
                   const Array<PrimExpr> &second) {
    if (first.size() != second.size())
      return false;
    for (size_t index = 0; index < first.size(); ++index) {
      if (!tl::arith::CanProveEqual(analyzer_, first[index], second[index]))
        return false;
    }
    return true;
  }

  Optional<PrimExpr> Contribution(const LegacyReducer &reducer,
                                  const TensorStoreNode *store) {
    auto self = [&](const PrimExpr &expression) {
      const auto *load = expression.as<TensorLoadNode>();
      return load && load->source.same_as(reducer.original) &&
             SameIndices(load->indices, store->indices);
    };
    auto match = [&](const PrimExpr &first,
                     const PrimExpr &second) -> Optional<PrimExpr> {
      if (self(first))
        return second;
      if (self(second))
        return first;
      return std::nullopt;
    };
    if (reducer.operation == ReducerV2OpType::kSum) {
      if (const auto *add = store->value.as<prim::AddNode>())
        return match(add->a, add->b);
    } else if (reducer.operation == ReducerV2OpType::kMax) {
      if (const auto *maximum = store->value.as<prim::MaxNode>())
        return match(maximum->a, maximum->b);
    } else if (reducer.operation == ReducerV2OpType::kMin) {
      if (const auto *minimum = store->value.as<prim::MinNode>())
        return match(minimum->a, minimum->b);
    } else if (const auto *call = store->value.as<CallNode>();
               call && call->args.size() == 2) {
      PrimVar left("left", reducer.original.dtype());
      PrimVar right("right", reducer.original.dtype());
      auto expected =
          ReducerV2Combine(reducer.operation, left, right).as<Call>();
      if (expected && expected.value()->op.same_as(call->op)) {
        return match(call->args[0].as_or_throw<PrimExpr>(),
                     call->args[1].as_or_throw<PrimExpr>());
      }
    }
    return std::nullopt;
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateNode *node, InplaceMode mode) final {
    const auto *call = node->value.as<CallNode>();
    if (!call || call->args.empty())
      return tvm::tl::StmtExprMutator::Mutate_(node, mode);
    if (call->op.same_as(Op::Get("tl.layout_hint"))) {
      auto found = reducers_.find(call->args[0].as_or_throw<TensorVar>().get());
      if (found != reducers_.end()) {
        auto arguments = call->args;
        arguments.Set(0, found->second.destination);
        return Evaluate(Call(call->ty, call->op, arguments, call->attrs,
                             call->ty_args, call->loc),
                        node->loc);
      }
    }
    if (call->op.same_as(Fill::Get())) {
      if (LegacyReducer *reducer = Find(call->args[0])) {
        TVM_FFI_CHECK(reducer->phase == LegacyPhase::kAllocated, ValueError)
            << "second clear/fill on a legacy reducer; each allocation "
               "supports one epoch";
        PrimExpr value = call->args[1].as_or_throw<PrimExpr>();
        PrimExpr identity =
            ReducerV2Identity(reducer->operation, reducer->original.dtype());
        bool is_identity = tl::arith::CanProveEqual(analyzer_, value, identity);
        if (const auto *first = value.as<FloatImmNode>()) {
          if (const auto *second = identity.as<FloatImmNode>())
            is_identity = first->value == second->value;
        }
        Array<Expr> arguments{NormalizeToBufferRegion(reducer->accumulator)};
        if (!is_identity) {
          TVM_FFI_CHECK(reducer->operation != ReducerV2OpType::kSum, ValueError)
              << "legacy sum reducer is filled with a non-zero value; use "
                 "reducer_init for a logical seed";
          arguments.push_back(
              tl::CastValue(Mutate(value, mode).ValueOrUnchanged(value),
                            reducer->original.dtype()));
        }
        reducer->phase = LegacyPhase::kActive;
        return Evaluate(Call(PrimType::Void(), ReducerInitOp::Get(), arguments),
                        node->loc);
      }
    }
    if (call->op.same_as(FinalizeReducerOp::Get()) && call->args.size() == 1) {
      if (LegacyReducer *reducer = Find(call->args[0])) {
        TVM_FFI_CHECK(reducer->phase == LegacyPhase::kActive, ValueError)
            << "legacy finalize_reducer requires a preceding clear/fill";
        reducer->phase = LegacyPhase::kFinalized;
        return Evaluate(Call(PrimType::Void(), FinalizeReducerV2Op::Get(),
                             {NormalizeToBufferRegion(reducer->accumulator),
                              NormalizeToBufferRegion(reducer->destination)},
                             call->attrs),
                        node->loc);
      }
    }
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node,
                            InplaceMode mode) final {
    auto tensor = node->dest.as<TensorVar>();
    auto found = tensor ? reducers_.find(tensor->get()) : reducers_.end();
    if (found == reducers_.end())
      return tvm::tl::StmtExprMutator::Mutate_(node, mode);
    const LegacyReducer &reducer = found->second;
    TVM_FFI_CHECK(reducer.phase == LegacyPhase::kActive, ValueError)
        << "legacy reducer store must be between initialization and "
           "finalization";
    auto contribution = Contribution(reducer, node);
    TVM_FFI_CHECK(contribution, ValueError)
        << "legacy reducer store must combine its current element with one "
           "contribution";
    PrimExpr value =
        Mutate(*contribution, mode).ValueOrUnchanged(*contribution);
    return Evaluate(
        Call(PrimType::Void(), reducer_update(),
             {TensorLoad(reducer.accumulator, node->indices), value}),
        node->loc);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *node,
                                InplaceMode mode) final {
    auto tensor = node->source.as<TensorVar>();
    auto found = tensor ? reducers_.find(tensor->get()) : reducers_.end();
    if (found == reducers_.end())
      return tvm::tl::StmtExprMutator::Mutate_(node, mode);
    TVM_FFI_CHECK(found->second.phase == LegacyPhase::kFinalized, ValueError)
        << "legacy reducer may be read only after finalize_reducer";
    Array<PrimExpr> indices;
    for (const PrimExpr &index : node->indices)
      indices.push_back(Mutate(index, mode).ValueOrUnchanged(index));
    return TensorLoad(found->second.destination, indices, node->loc);
  }

  UnchangedOr<Expr> Mutate_(const VarNode *node, InplaceMode mode) final {
    auto found = reducers_.find(node);
    if (found == reducers_.end())
      return tvm::tl::StmtExprMutator::Mutate_(node, mode);
    TVM_FFI_CHECK(found->second.phase == LegacyPhase::kFinalized, ValueError)
        << "legacy reducer pointer access is forbidden before finalization";
    return found->second.destination;
  }

  sym::Analyzer analyzer_;
  std::unordered_map<const VarNode *, LegacyReducer> reducers_;
  std::unordered_set<const VarNode *> first_class_;
};
} // namespace

tvm::transform::Pass CanonicalizeLegacyReducer() {
  return tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext) -> Optional<tirx::Function> {
        return ffi::make_object<LegacyReducerCanonicalizer>()->Run(
            std::move(function));
      },
      0, "tl.CanonicalizeLegacyReducer");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.CanonicalizeLegacyReducer",
                                   CanonicalizeLegacyReducer);
}
} // namespace tvm::tl
