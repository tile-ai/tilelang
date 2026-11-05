/*!
 * \file lower_thread_allreduce.cc
 * \brief Native allreduce with TileLang-owned scratch and synchronization
 * policy.
 */
#include "../op/builtin.h"
#include "upstream_pass_bridge.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl::transform {
namespace {
using namespace tvm::prim;
using namespace tvm::tirx;

bool IsStorageSync(const CallObj *call, const char *scope) {
  if (!call || !call->op.same_as(tirx::gpu_storage_sync_op()) ||
      call->args.size() != 1)
    return false;
  const auto *value = call->args[0].as<StringImmObj>();
  return value && value->value == scope;
}

// The pinned native lowering leaves unrelated allocations and constant-operand
// user barriers unchanged. Keep their object identities alive while adapting
// only definitions and barriers introduced by that invocation.
class InputCollector final : public StmtExprVisitor {
public:
  explicit InputCollector(tirx::Function source) : source_(std::move(source)) {}
  tirx::Function source_;
  int dynamic_shared = 0;
  std::unordered_set<const VarObj *> allocations;
  std::unordered_set<const CallObj *> barriers;
  bool has_allreduce = false;

  ffi::Optional<VisitInterrupt> Visit_(const BindObj *node) final {
    const auto *call = node->value.as<CallObj>();
    if (call && call->op.same_as(tirx::alloc_tensor_op())) {
      allocations.insert(node->var.get());
      if (const auto *tensor = node->var->ty.as<TensorTypeObj>();
          tensor && tensor->scope.name() == "shared.dyn")
        ++dynamic_shared;
    }
    return StmtExprVisitor::Visit_(node);
  }

  ffi::Optional<VisitInterrupt> Visit_(const CallObj *node) final {
    if (node->op.same_as(tirx::gpu_storage_sync_op()))
      barriers.insert(node);
    if (node->op.same_as(tirx::gpu_thread_allreduce_op()))
      has_allreduce = true;
    return StmtExprVisitor::Visit_(node);
  }
};

// alloc_tensor accepts omitted annotations, while the pinned native allreduce
// remapper reads a DictAttrs value unconditionally. Supply its empty container
// at this boundary; existing annotations and allocation identities are
// retained.
class AllocationAttrsNormalizer final : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  UnchangedOr<Expr> Mutate_(const CallObj *node, InplaceMode mode) final {
    if (node->op.same_as(tirx::alloc_tensor_op()) && !node->attrs)
      return Call(node->ty, node->op, node->args, DictAttrs(), node->ty_args,
                  node->loc);
    return StmtExprMutator::Mutate_(node, mode);
  }
};

class ValueRewriter final : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  ffi::Map<Var, Expr> values;

  UnchangedOr<Expr> Mutate_(const VarObj *node, InplaceMode) final {
    if (auto replacement = values.Get(ffi::GetRef<Var>(node)))
      return *replacement;
    return ffi::Unchanged();
  }
};

class ScratchRewriter final : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit ScratchRewriter(const InputCollector &input) : input_(input) {}

  UnchangedOr<Stmt> Mutate_(const BindObj *node, InplaceMode mode) final {
    const auto *call = node->value.as<CallObj>();
    if (call && call->op.same_as(tirx::alloc_tensor_op()) &&
        !input_.allocations.count(node->var.get())) {
      auto type = node->var->ty.as_or_throw<TensorType>();
      if (type->scope.name() == "shared") {
        TensorType replacement_type(
            type->shape, type->dtype, runtime::MemoryScope::Get("shared.dyn"),
            type->data_alignment, type->layout, type->loc);
        TensorVar replacement(node->var->name, replacement_type);
        values_.Set(node->var, replacement);
        return Bind(replacement,
                    Call(replacement_type, call->op, call->args, call->attrs,
                         {replacement_type}, call->loc),
                    node->loc);
      }
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Expr> Mutate_(const VarObj *node, InplaceMode) final {
    if (auto replacement = values_.Get(ffi::GetRef<Var>(node)))
      return *replacement;
    return ffi::Unchanged();
  }

  UnchangedOr<Expr> Mutate_(const CallObj *node, InplaceMode mode) final {
    auto updated = StmtExprMutator::Mutate_(node, mode);
    if (!updated.UnchangedOrSameAs(ffi::GetRef<Expr>(node)) &&
        node->op.same_as(tirx::tensor_data_ptr_op())) {
      auto call = std::move(updated).ValueUnchecked().as_or_throw<Call>();
      return Call(std::nullopt, call->op, call->args, call->attrs,
                  call->ty_args, call->loc);
    }
    return updated;
  }

private:
  const InputCollector &input_;
  ffi::Map<Var, Expr> values_;
};

// Flatten only sequence containers: crossing an If/For/Region would lose its
// participant domain. The native memory stage is a sequence of load bindings,
// warp barrier, stores, warp barrier, repeated for each reduction distance.
void FlattenSequence(const Stmt &stmt, std::vector<Stmt> *items) {
  if (const auto *sequence = stmt.as<SeqStmtObj>()) {
    for (const Stmt &child : sequence->seq)
      FlattenSequence(child, items);
  } else {
    items->push_back(stmt);
  }
}

class GeneratedSyncRewriter final : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  GeneratedSyncRewriter(const InputCollector &input, ffi::String shared_scope,
                        ffi::Optional<PrimExpr> mask = std::nullopt)
      : input_(input), shared_scope_(std::move(shared_scope)),
        mask_(std::move(mask)) {}

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *node, InplaceMode mode) final {
    const auto *call = node->value.as<CallObj>();
    if (call && !input_.barriers.count(call)) {
      if (IsStorageSync(call, "warp") && mask_)
        return Evaluate(Call(PrimType::Void(), tl::sync_warp(), {*mask_},
                             std::nullopt, {}, call->loc),
                        node->loc);
      if (IsStorageSync(call, "shared") && shared_scope_ != "shared")
        return Evaluate(Call(PrimType::Void(), tirx::gpu_storage_sync_op(),
                             {StringImm(shared_scope_)}, call->attrs, {},
                             call->loc),
                        node->loc);
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

private:
  const InputCollector &input_;
  ffi::String shared_scope_;
  ffi::Optional<PrimExpr> mask_;
};

class MemoryStageRewriter final : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  MemoryStageRewriter(const InputCollector &input, ffi::String shared_scope,
                      bool masked_warp, bool cuda_mask)
      : input_(input), shared_scope_(std::move(shared_scope)),
        masked_warp_(masked_warp), cuda_mask_(cuda_mask) {}

  UnchangedOr<Stmt> Mutate_(const IfObj *node, InplaceMode mode) final {
    std::vector<Stmt> items;
    FlattenSequence(node->then_case, &items);
    bool generated_warp_stage = false;
    for (const auto &stmt : items)
      generated_warp_stage |= IsGeneratedWarpBarrier(stmt);
    if (!generated_warp_stage)
      return StmtExprMutator::Mutate_(node, mode);

    TVM_FFI_CHECK(!node->else_case.has_value(), ValueError)
        << "Unexpected native allreduce warp stage with an else branch";
    if (masked_warp_)
      return WithWarpMask(node->condition, node->then_case, node->loc);
    return WithUniformBlockBarriers(node->condition, items, node->loc);
  }

private:
  bool IsGeneratedWarpBarrier(const Stmt &stmt) const {
    const auto *evaluate = stmt.as<EvaluateObj>();
    const auto *call = evaluate ? evaluate->value.as<CallObj>() : nullptr;
    return IsStorageSync(call, "warp") && !input_.barriers.count(call);
  }

  Stmt WithWarpMask(PrimExpr predicate, SeqStmt body, Location loc) {
    const PrimType type = PrimType::UInt(cuda_mask_ ? 32 : 64);
    Var active("allreduce_active_mask", type);
    Var participant("allreduce_participant_mask", type);
    auto active_value =
        Call(PrimType::UInt(64), tl::activemask(), {}).as_or_throw<PrimExpr>();
    auto participant_value =
        Call(PrimType::UInt(64), tl::ballot_sync(), {active, predicate})
            .as_or_throw<PrimExpr>();
    auto rewriter = ffi::make_object<GeneratedSyncRewriter>(
        input_, shared_scope_, participant.as_or_throw<PrimExpr>());
    body = rewriter->Mutate(body).ValueOrUnchanged(body);
    return SeqStmt({Bind(active, Cast(type, active_value), loc),
                    Bind(participant, Cast(type, participant_value), loc),
                    If(predicate, body, std::nullopt, loc)},
                   loc);
  }

  Stmt WithUniformBlockBarriers(PrimExpr predicate,
                                const std::vector<Stmt> &items, Location loc) {
    ffi::Array<Stmt> result;
    size_t cursor = 0;
    while (cursor < items.size()) {
      ffi::Array<Stmt> guarded_loads, stores;
      auto values = ffi::make_object<ValueRewriter>();
      // Keep loads under the original predicate. Per-thread temporary tensors
      // carry their values across the uniform barrier without leaking a Bind
      // definition from one conditional scope into another.
      while (cursor < items.size() && !IsGeneratedWarpBarrier(items[cursor])) {
        const auto *binding = items[cursor++].as<BindObj>();
        TVM_FFI_CHECK(binding && binding->var->ty.as<PrimTypeObj>(), ValueError)
            << "Unexpected native allreduce memory-stage load sequence";
        TensorVar temporary("allreduce_warp_value",
                            TensorType({IntImm::Int32(1)},
                                       binding->var->ty.as_or_throw<PrimType>(),
                                       runtime::MemoryScope::Get("local")));
        result.push_back(Bind(temporary,
                              Call(temporary.ty(), tirx::alloc_tensor_op(), {},
                                   DictAttrs(), {temporary.ty()}, loc),
                              loc));
        guarded_loads.push_back(
            TensorStore(temporary, {IntImm::Int32(0)},
                        binding->value.as_or_throw<PrimExpr>(), binding->loc));
        values->values.Set(binding->var,
                           TensorLoad(temporary, {IntImm::Int32(0)}));
      }
      TVM_FFI_CHECK(!guarded_loads.empty() && cursor < items.size(), ValueError)
          << "Native allreduce memory stage lacks its load barrier";
      ++cursor;
      while (cursor < items.size() && !IsGeneratedWarpBarrier(items[cursor])) {
        const Stmt &store = items[cursor++];
        TVM_FFI_CHECK(store.as<TensorStoreObj>(), ValueError)
            << "Unexpected native allreduce memory-stage store sequence";
        stores.push_back(values->Mutate(store).ValueOrUnchanged(store));
      }
      TVM_FFI_CHECK(stores.size() == guarded_loads.size() &&
                        cursor < items.size(),
                    ValueError)
          << "Native allreduce memory stage lacks stores or its store barrier";
      ++cursor;
      auto barrier = [&]() {
        return Evaluate(Call(PrimType::Void(), tirx::gpu_storage_sync_op(),
                             {StringImm(shared_scope_)}, std::nullopt, {}, loc),
                        loc);
      };
      result.push_back(
          If(predicate, SeqStmt(guarded_loads), std::nullopt, loc));
      result.push_back(barrier());
      result.push_back(If(predicate, SeqStmt(stores), std::nullopt, loc));
      result.push_back(barrier());
    }
    return SeqStmt(result, loc);
  }

  const InputCollector &input_;
  ffi::String shared_scope_;
  bool masked_warp_;
  bool cuda_mask_;
};

class FinalBarrierVerifier final : public StmtExprVisitor {
public:
  explicit FinalBarrierVerifier(const InputCollector &input) : input_(input) {}
  ffi::Optional<VisitInterrupt> Visit_(const CallObj *node) final {
    TVM_FFI_CHECK(!IsStorageSync(node, "warp") || input_.barriers.count(node),
                  ValueError)
        << "Unrecognized native allreduce warp barrier; review the pinned "
           "memory-stage contract before rebasing";
    return StmtExprVisitor::Visit_(node);
  }

private:
  const InputCollector &input_;
};
} // namespace

tvm::transform::Pass LowerThreadAllreduce() {
  auto rewrite = [](tirx::Function function, IRModule,
                    tvm::transform::PassContext) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto input = ffi::make_object<InputCollector>(function);
    input->Visit(function->body);
    if (!input->has_allreduce)
      return function;
    auto target = function->GetAttr(tvm::attr::kTarget);
    TVM_FFI_CHECK(target.has_value(), ValueError)
        << "LowerThreadAllreduce requires a function target";
    ffi::String scope = input->dynamic_shared > 1 ? "shared.dyn" : "shared";
    const auto &kind = target.value()->kind->name;
    auto allocations = ffi::make_object<AllocationAttrsNormalizer>();
    auto original = function->body;
    auto normalized = allocations->Mutate(original).ValueOrUnchanged(original);
    if (!normalized.same_as(original))
      function.CopyOnWrite()->body = normalized;
    function = upstream_pass_bridge::LowerThreadAllreduce(std::move(function));
    auto body = function->body;
    if (scope == "shared.dyn") {
      auto scratch = ffi::make_object<ScratchRewriter>(*input);
      body = scratch->Mutate(body).ValueOrUnchanged(body);
    }
    auto stages = ffi::make_object<MemoryStageRewriter>(
        *input, scope, kind == "cuda" || kind == "rocm" || kind == "ascend",
        kind == "cuda");
    body = stages->Mutate(body).ValueOrUnchanged(body);
    auto syncs = ffi::make_object<GeneratedSyncRewriter>(*input, scope);
    body = syncs->Mutate(body).ValueOrUnchanged(body);
    ffi::make_object<FinalBarrierVerifier>(*input)->Visit(body);
    function.CopyOnWrite()->body = body;
    return function;
  };
  return tirx::transform::CreateFunctionPass(rewrite, 0,
                                             "tl.LowerThreadAllreduce");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerThreadAllreduce",
                                   LowerThreadAllreduce);
}
} // namespace tvm::tl::transform
