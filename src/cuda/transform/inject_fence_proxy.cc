/*!
 * \file inject_fence_proxy.cc
 * \brief Insert generic-to-async proxy fences using native control flow.
 */
#include "cuda/op/builtin.h"
#include "cuda/target_utils.h"
#include "op/native_intrinsic.h"
#include "transform/common/native_layout_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <cstdint>
#include <vector>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

// At a join, retain every possible most recent proxy. A loop header is the
// union of the entry state and the back-edge state, not just the first trip.
class ProxyStateSet {
public:
  static ProxyStateSet Unreachable() { return ProxyStateSet(0); }
  static ProxyStateSet None() { return ProxyStateSet(1); }
  static ProxyStateSet Generic() { return ProxyStateSet(2); }
  static ProxyStateSet Async() { return ProxyStateSet(4); }
  bool MayBeGeneric() const { return bits_ & 2; }
  bool IsReachable() const { return bits_ != 0; }
  ProxyStateSet Union(ProxyStateSet other) const {
    return ProxyStateSet(bits_ | other.bits_);
  }
  bool operator==(ProxyStateSet other) const { return bits_ == other.bits_; }

private:
  explicit ProxyStateSet(uint8_t bits) : bits_(bits) {}
  uint8_t bits_;
};

enum class ProxyEvent { kNone, kGeneric, kAsync, kNeutral };

bool IsShared(const Type &type) {
  runtime::MemoryScope scope;
  if (const auto *pointer = type.as<PtrTypeObj>())
    scope = pointer->scope;
  else if (const auto *tensor = type.as<TensorTypeObj>())
    scope = tensor->scope;
  else
    return false;
  const auto &name = scope.name();
  return name == "shared" || name.find("shared.") == 0;
}

bool ContainsShared(const Expr &expression) {
  bool found = IsShared(expression->ty);
  VisitNativePostOrder(expression, [&](const ObjectRef &object) {
    if (const auto *value = object.as<ExprObj>())
      found |= IsShared(value->ty);
  });
  return found;
}

// Respect explicit read-only pointers even when nested in a byte-offset or
// reinterpret operation. A raw typed shared pointer has no read-only promise.
bool MayWriteThrough(const Expr &expression) {
  // A load passes its scalar value, not the address of its source. Barrier
  // intrinsics also encode the barrier lvalue as a TensorLoad; they do not
  // constitute a generic data-pointer write before the next async issue.
  if (expression.as<TensorLoadObj>())
    return false;
  if (const auto *call = expression.as<CallObj>()) {
    if (call->op.same_as(Op::Get("tl.access_ptr")) ||
        call->op.same_as(Op::Get("tl.ptr_access"))) {
      size_t mask_index = call->op.same_as(Op::Get("tl.access_ptr")) ? 2 : 3;
      if (call->args.size() <= mask_index)
        return ContainsShared(expression);
      if (!ContainsShared(call->args[0]))
        return false;
      const auto *mask = call->args[mask_index].as<IntImmObj>();
      return !mask || (mask->value & 2) != 0;
    }
    if (call->op.same_as(reinterpret_op()) ||
        call->op.same_as(ptr_byte_offset_op()))
      return MayWriteThrough(call->args[0]);
  }
  if (const auto *tuple = expression.as<TupleObj>()) {
    for (const Expr &field : tuple->fields)
      if (MayWriteThrough(field))
        return true;
    return false;
  }
  // Native pointers/tensors carry their storage scope in the type. Do not
  // recursively interpret every loaded scalar/index as an escaping pointer.
  return IsShared(expression->ty);
}

bool IsAsyncIntrinsic(const CallObj *call) {
  return call &&
         (call->op.same_as(tma_load()) || call->op.same_as(tma_load_im2col()) ||
          call->op.same_as(tma_load_multicast()) ||
          call->op.same_as(tma_store()) ||
          call->op.same_as(tma_load_gather4()) ||
          call->op.same_as(tma_store_scatter4()) ||
          call->op.same_as(ptx_wgmma_ss()) ||
          call->op.same_as(ptx_wgmma_rs()) ||
          call->op.same_as(ptx_wgmma_sp_ss()) ||
          call->op.same_as(ptx_wgmma_sp_rs()) ||
          call->op.same_as(ptx_tcgen05_mma_ss()) ||
          call->op.same_as(ptx_tcgen05_mma_ts()) ||
          call->op.same_as(ptx_tcgen05_mma_blockscaled_ss()) ||
          call->op.same_as(ptx_tcgen05_cp_warpx4()) ||
          call->op.same_as(ptx_cp_async_bulk()));
}

ProxyEvent ClassifyCall(const CallObj *call) {
  if (!call)
    return ProxyEvent::kNone;
  if (call->op.same_as(fence_proxy_async()))
    return ProxyEvent::kNeutral;
  if (IsAsyncIntrinsic(call))
    return ProxyEvent::kAsync;
  if (SideEffect(GetRef<Expr>(call)) < CallEffectKind::kUpdateState)
    return ProxyEvent::kNone;
  // Runtime arguments participate in effects; static annotations/type arguments
  // do not execute. Decode intrinsic tails before querying pointer contracts.
  Array<Expr> arguments = call->args;
  if (const auto *op = call->op.as<OpObj>();
      op && op->name.find("tl.") == 0 && op->name != "tl.annotation")
    arguments = NativeIntrinsicOperands(call);
  for (const Expr &argument : arguments)
    if (MayWriteThrough(argument))
      return ProxyEvent::kGeneric;
  return ProxyEvent::kNone;
}

struct EventSummary {
  bool generic{false};
  bool async{false};
};

class EventSummarizer : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;
  Optional<VisitInterrupt> Visit_(const TensorStoreObj *store) final {
    summary.generic |= IsShared(store->dest->ty);
    return StmtExprVisitor::Visit_(store);
  }
  Optional<VisitInterrupt> Visit_(const EvaluateObj *evaluate) final {
    Record(evaluate->value.as<CallObj>());
    return std::nullopt;
  }
  Optional<VisitInterrupt> Visit_(const BindObj *binding) final {
    Record(binding->value.as<CallObj>());
    return std::nullopt;
  }
  EventSummary summary;

private:
  void Record(const CallObj *call) {
    ProxyEvent event = ClassifyCall(call);
    summary.generic |= event == ProxyEvent::kGeneric;
    summary.async |= event == ProxyEvent::kAsync;
  }
};

EventSummary Summarize(const Stmt &statement) {
  auto visitor = make_object<EventSummarizer>();
  visitor->Visit(statement);
  return visitor->summary;
}

Stmt Fence(Location location) {
  return Evaluate(
      Call(PrimType::Void(), fence_proxy_async(), {}, {}, {}, location),
      location);
}

class ProxyFenceRewriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *sequence,
                            InplaceMode mode) final {
    Array<Stmt> statements;
    for (const Stmt &statement : sequence->seq)
      statements.push_back(Mutate(statement, mode).ValueOrUnchanged(statement));
    return SeqStmt(statements, sequence->loc);
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *evaluate,
                            InplaceMode mode) final {
    Stmt statement = StmtExprMutator::Mutate_(evaluate, mode)
                         .ValueOrUnchanged(GetRef<Stmt>(evaluate));
    return ApplyEvent(
        statement,
        ClassifyCall(statement.as_or_throw<Evaluate>()->value.as<CallObj>()));
  }

  UnchangedOr<Stmt> Mutate_(const BindObj *binding, InplaceMode mode) final {
    Stmt statement = StmtExprMutator::Mutate_(binding, mode)
                         .ValueOrUnchanged(GetRef<Stmt>(binding));
    return ApplyEvent(
        statement,
        ClassifyCall(statement.as_or_throw<Bind>()->value.as<CallObj>()));
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj *store,
                            InplaceMode mode) final {
    Stmt statement = StmtExprMutator::Mutate_(store, mode)
                         .ValueOrUnchanged(GetRef<Stmt>(store));
    if (current_.IsReachable() && IsShared(store->dest->ty))
      current_ = ProxyStateSet::Generic();
    return statement;
  }

  UnchangedOr<Stmt> Mutate_(const BreakObj *statement, InplaceMode mode) final {
    TVM_FFI_CHECK(!loop_exits_.empty(), ValueError) << "Break outside a loop";
    auto &exits = loop_exits_.back();
    exits.breaks = exits.breaks.Union(current_);
    current_ = ProxyStateSet::Unreachable();
    return GetRef<Stmt>(statement);
  }

  UnchangedOr<Stmt> Mutate_(const ContinueObj *statement,
                            InplaceMode mode) final {
    TVM_FFI_CHECK(!loop_exits_.empty(), ValueError)
        << "Continue outside a loop";
    auto &exits = loop_exits_.back();
    exits.continues = exits.continues.Union(current_);
    current_ = ProxyStateSet::Unreachable();
    return GetRef<Stmt>(statement);
  }

  UnchangedOr<Stmt> Mutate_(const ReturnObj *statement,
                            InplaceMode mode) final {
    auto result = StmtExprMutator::Mutate_(statement, mode);
    current_ = ProxyStateSet::Unreachable();
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const IfObj *branch, InplaceMode mode) final {
    ProxyStateSet entry = current_;
    bool hoist = false;
    if (entry.MayBeGeneric() && branch->else_case) {
      auto lhs = Summarize(branch->then_case);
      auto rhs = Summarize(branch->else_case.value());
      hoist = lhs.async && !lhs.generic && rhs.async && !rhs.generic;
    }
    ProxyStateSet branch_entry = hoist ? ProxyStateSet::None() : entry;
    auto then_result = Rewrite(branch->then_case, branch_entry, mode);
    Optional<SeqStmt> otherwise;
    ProxyStateSet else_state = branch_entry;
    if (branch->else_case) {
      auto result = Rewrite(branch->else_case.value(), branch_entry, mode);
      otherwise = result.statement;
      else_state = result.state;
    }
    current_ = then_result.state.Union(else_state);
    Stmt result =
        If(branch->condition, then_result.statement, otherwise, branch->loc);
    return hoist ? Stmt(SeqStmt({Fence(branch->loc), result}, branch->loc))
                 : result;
  }

  UnchangedOr<Stmt> Mutate_(const ForObj *loop, InplaceMode mode) final {
    ProxyStateSet entry = current_;
    auto summary = Summarize(loop->body);
    bool hoist = entry.MayBeGeneric() && summary.async && !summary.generic;
    ProxyStateSet header = hoist ? ProxyStateSet::None() : entry;
    auto body = Fixpoint(loop->body, header, mode);
    bool may_be_zero = true;
    auto begin = loop->begin.as<IntImmObj>();
    auto end = loop->end.as<IntImmObj>();
    auto step = loop->step.as<IntImmObj>();
    if (begin && end && step) {
      if (step->value > 0)
        may_be_zero = begin->value >= end->value;
      else if (step->value < 0)
        may_be_zero = begin->value <= end->value;
    }
    current_ = may_be_zero ? header.Union(body.state) : body.state;
    Stmt result = For(loop->var, loop->begin, loop->end, loop->step, loop->kind,
                      body.statement, loop->annotations, loop->loc);
    return hoist ? Stmt(SeqStmt({Fence(loop->loc), result}, loop->loc))
                 : result;
  }

  UnchangedOr<Stmt> Mutate_(const WhileObj *loop, InplaceMode mode) final {
    ProxyStateSet entry = current_;
    auto summary = Summarize(loop->body);
    bool hoist = entry.MayBeGeneric() && summary.async && !summary.generic;
    ProxyStateSet header = hoist ? ProxyStateSet::None() : entry;
    auto body = Fixpoint(loop->body, header, mode);
    current_ = header.Union(body.state);
    Stmt result = While(loop->condition, body.statement, loop->loc);
    return hoist ? Stmt(SeqStmt({Fence(loop->loc), result}, loop->loc))
                 : result;
  }

private:
  struct Result {
    SeqStmt statement;
    ProxyStateSet state;
  };
  struct LoopExits {
    ProxyStateSet breaks{ProxyStateSet::Unreachable()};
    ProxyStateSet continues{ProxyStateSet::Unreachable()};
  };

  Result Rewrite(const SeqStmt &statement, ProxyStateSet entry,
                 InplaceMode mode) {
    auto saved = current_;
    current_ = entry;
    auto result = Mutate(statement, mode).ValueOrUnchanged(statement);
    auto state = current_;
    current_ = saved;
    return {result, state};
  }

  Result Fixpoint(const SeqStmt &body, ProxyStateSet entry, InplaceMode mode) {
    ProxyStateSet header = entry;
    Result result{body, entry};
    for (int iteration = 0; iteration < 8; ++iteration) {
      // Every iteration analyzes the original body; earlier speculative
      // insertions must not hide a generic back-edge in later iterations.
      loop_exits_.push_back({});
      result = Rewrite(body, header, InplaceMode::kDisallow);
      auto exits = loop_exits_.back();
      loop_exits_.pop_back();
      auto backedge = result.state.Union(exits.continues);
      auto next = entry.Union(backedge);
      if (next == header) {
        // Continue reaches the next header (including the final condition),
        // whereas Break reaches the successor without executing the suffix.
        result.state = backedge.Union(exits.breaks);
        return result;
      }
      header = next;
    }
    TVM_FFI_THROW(InternalError) << "Proxy fence state did not converge";
  }

  Stmt ApplyEvent(const Stmt &statement, ProxyEvent event) {
    if (!current_.IsReachable())
      return statement;
    switch (event) {
    case ProxyEvent::kNone:
      return statement;
    case ProxyEvent::kNeutral:
      current_ = ProxyStateSet::None();
      return statement;
    case ProxyEvent::kGeneric:
      current_ = ProxyStateSet::Generic();
      return statement;
    case ProxyEvent::kAsync: {
      bool needs_fence = current_.MayBeGeneric();
      current_ = ProxyStateSet::Async();
      return needs_fence ? Stmt(SeqStmt({Fence(statement->loc), statement},
                                        statement->loc))
                         : statement;
    }
    }
    return statement;
  }
  ProxyStateSet current_{ProxyStateSet::None()};
  std::vector<LoopExits> loop_exits_;
};
} // namespace

tvm::transform::Pass InjectFenceProxy() {
  auto rewrite = [](tirx::Function function, IRModule,
                    tvm::transform::PassContext) {
    auto target = function->GetAttr<Target>(tvm::attr::kTarget);
    if (!target || !TargetHasBulkCopy(*target) ||
        function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto rewriter = make_object<ProxyFenceRewriter>();
    auto body = function->body;
    function.CopyOnWrite()->body =
        rewriter->Mutate(body).ValueOrUnchanged(body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(rewrite, 0, "tl.InjectFenceProxy");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.InjectFenceProxy",
                                   InjectFenceProxy);
}
} // namespace tvm::tl
