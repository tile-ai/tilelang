/*!
 * \file tl/transform/reducer/verify_reducer_epochs.cc
 * \brief Semantic verification for first-class deferred reducer epochs.
 */

#include "reducer.h"
#include "reducer_metadata.h"

#include "op/region.h"
#include "op/utils.h"
#include "span_utils.h"
#include "support/check.h"

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <functional>
#include <optional>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

class ReducerMetadataCollector : public StmtExprVisitor {
public:
  static ReducerMetadata Collect(const PrimFunc &func) {
    ReducerMetadataCollector collector;
    collector(func->body);
    for (const auto &[var, buffer] : collector.metadata_.buffers) {
      ICHECK(collector.metadata_.info.count(var))
          << "Reducer buffer `" << buffer->name
          << "` is missing reducer metadata" << SpanHintSuffix(buffer->span);
    }
    for (const auto &[var, info] : collector.metadata_.info) {
      ICHECK(collector.metadata_.buffers.count(var))
          << "Reducer metadata does not refer to a local.reducer allocation: "
          << var;
      ICHECK(IsBuiltinCommutativeReduceType(info->combine_type));
    }
    return std::move(collector.metadata_);
  }

private:
  void VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      if (buffer.scope() == "local.reducer") {
        ICHECK(!metadata_.buffers.count(buffer->data))
            << "Reducer storage Var is allocated more than once: "
            << buffer->data;
        metadata_.buffers.emplace(buffer->data, buffer);
      }
    }

    if (Optional<Any> annotation = op->annotations.Get(attr::kReducerInfo)) {
      Map<Var, Map<String, Any>> definitions =
          annotation.value().cast<Map<Var, Map<String, Any>>>();
      for (const auto &[var, fields] : definitions) {
        ICHECK(!fields.count("rep"))
            << "Reducer v2 does not support the legacy `replication=` policy";
        Optional<Any> op_field = fields.Get("op");
        ICHECK(op_field.has_value()) << "Reducer metadata is missing `op`";
        std::optional<String> op_name = op_field.value().try_cast<String>();
        ICHECK(op_name.has_value()) << "Reducer metadata `op` must be a string";
        Optional<PrimExpr> seed;
        if (Optional<Any> seed_field = fields.Get("seed")) {
          seed = seed_field.value().cast<PrimExpr>();
        }
        ICHECK(!metadata_.info.count(var))
            << "Reducer metadata is defined more than once for " << var;
        metadata_.info.emplace(var, ReducerInfo(op_name.value(), seed));
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  ReducerMetadata metadata_;
};

enum class ReducerState { kAllocated, kActive, kFinalized };

class ReducerEpochVerifier : public StmtExprVisitor {
public:
  explicit ReducerEpochVerifier(const ReducerMetadata &metadata)
      : metadata_(metadata) {
    for (const auto &[var, _] : metadata.info) {
      states_.emplace(var, ReducerState::kAllocated);
    }
  }

  void Verify(const PrimFunc &func) {
    (*this)(func->body);
    for (const auto &[var, state] : states_) {
      const Buffer &buffer = metadata_.buffers.at(var);
      ICHECK(state == ReducerState::kFinalized)
          << "Reducer `" << buffer->name
          << "` must have exactly one explicit T.reducer_init and one "
             "T.finalize_reducer"
          << SpanHintSuffix(buffer->span);
    }
  }

private:
  void VisitStmt_(const ForNode *op) final {
    analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    ++nested_control_depth_;
    StmtExprVisitor::VisitStmt_(op);
    --nested_control_depth_;
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    VisitExpr(op->condition);
    ++nested_control_depth_;
    std::function<void()> exit_then = analyzer_.EnterConstraint(op->condition);
    VisitStmt(op->then_case);
    exit_then();
    if (op->else_case.defined()) {
      std::function<void()> exit_else =
          analyzer_.EnterConstraint(Not(op->condition));
      VisitStmt(op->else_case.value());
      exit_else();
    }
    --nested_control_depth_;
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar thread = Downcast<IterVar>(op->node);
      analyzer_.Bind(
          thread->var,
          Range::FromMinExtent(make_zero(thread->var.dtype()), op->value));
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    CheckOrdinaryAccess(op->buffer, "BufferStore", op->span);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    CheckOrdinaryAccess(op->buffer, "BufferLoad", op->span);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(ReducerInitOp::Get())) {
      ICHECK_EQ(op->args.size(), 1U);
      Var var = ReducerVarFromRegion(op->args[0]);
      RequireTopLevelLifecycleOp("T.reducer_init", op->span);
      ICHECK(states_.at(var) == ReducerState::kAllocated)
          << "T.reducer_init must occur exactly once per reducer allocation"
          << SpanHintSuffix(op->span);
      states_[var] = ReducerState::kActive;
      return;
    }
    if (op->op.same_as(ReducerUpdateOp::Get())) {
      ICHECK_EQ(op->args.size(), 2U);
      AccessRegion access =
          NormalizeToAccessRegion(op->args[0], kAccessReadWrite);
      Var var = RequireReducer(access.region->buffer);
      ICHECK(states_.at(var) == ReducerState::kActive)
          << "T.reducer_update must be dominated by T.reducer_init and precede "
             "T.finalize_reducer"
          << SpanHintSuffix(op->span);
      ICHECK_EQ(access.region->region.size(),
                access.region->buffer->shape.size());
      for (size_t i = 0; i < access.region->region.size(); ++i) {
        const Range &range = access.region->region[i];
        ICHECK(is_one(range->extent))
            << "T.reducer_update must name exactly one logical output element"
            << SpanHintSuffix(op->span);
        PrimExpr index = analyzer_.Simplify(range->min);
        ICHECK(analyzer_.CanProve(index >= make_zero(index.dtype())) &&
               analyzer_.CanProve(index < access.region->buffer->shape[i]))
            << "T.reducer_update index " << index
            << " is not provably within [0, " << access.region->buffer->shape[i]
            << ") for reducer dimension " << i << SpanHintSuffix(op->span);
        VisitExpr(range->min);
      }
      ICHECK_LE(SideEffect(op->args[1]), CallEffectKind::kReadState)
          << "T.reducer_update contribution may read state but must not write "
             "state or invoke an opaque effect"
          << SpanHintSuffix(op->span);
      VisitExpr(op->args[1]);
      return;
    }
    if (op->op.same_as(FinalizeReducerOp::Get())) {
      ICHECK_EQ(op->args.size(), 2U);
      Var var = ReducerVarFromRegion(op->args[0]);
      RequireTopLevelLifecycleOp("T.finalize_reducer", op->span);
      ICHECK(states_.at(var) == ReducerState::kActive)
          << "T.finalize_reducer must occur exactly once after initialization"
          << SpanHintSuffix(op->span);
      AccessRegion destination =
          NormalizeToAccessRegion(op->args[1], kAccessWrite);
      ICHECK(!metadata_.info.count(destination.region->buffer->data))
          << "T.finalize_reducer destination must not alias reducer storage"
          << SpanHintSuffix(op->span);
      states_[var] = ReducerState::kFinalized;
      return;
    }

    if (op->op.same_as(builtin::tvm_access_ptr()) && op->args.size() > 1) {
      if (Optional<Var> var = op->args[1].as<Var>()) {
        CheckOrdinaryVarAccess(var.value(), "access_ptr", op->span);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  Var ReducerVarFromRegion(const PrimExpr &arg) const {
    AccessRegion access = NormalizeToAccessRegion(arg, kAccessReadWrite);
    return RequireReducer(access.region->buffer);
  }

  Var RequireReducer(const Buffer &buffer) const {
    ICHECK(metadata_.info.count(buffer->data))
        << "First-class reducer op expects a local.reducer handle, got buffer `"
        << buffer->name << "` in scope `" << buffer.scope() << "`"
        << SpanHintSuffix(buffer->span);
    return buffer->data;
  }

  void RequireTopLevelLifecycleOp(const char *name, const Span &span) const {
    ICHECK_EQ(nested_control_depth_, 0)
        << name
        << " must be in participant-uniform top-level control flow in reducer "
           "v2"
        << SpanHintSuffix(span);
  }

  void CheckOrdinaryAccess(const Buffer &buffer, const char *kind,
                           const Span &span) const {
    CheckOrdinaryVarAccess(buffer->data, kind, span);
  }

  void CheckOrdinaryVarAccess(const Var &var, const char *kind,
                              const Span &span) const {
    ICHECK(!metadata_.info.count(var))
        << "Reducer v2 forbids ordinary " << kind
        << " access; use T.reducer_init, T.reducer_update, and "
           "T.finalize_reducer"
        << SpanHintSuffix(span);
  }

  const ReducerMetadata &metadata_;
  std::unordered_map<Var, ReducerState, ObjectPtrHash, ObjectPtrEqual> states_;
  arith::Analyzer analyzer_;
  int nested_control_depth_{0};
};

} // namespace

ReducerMetadata CollectReducerMetadata(const PrimFunc &func) {
  return ReducerMetadataCollector::Collect(func);
}

void VerifyReducerEpochSemantics(const PrimFunc &func,
                                 const ReducerMetadata &metadata) {
  ReducerEpochVerifier(metadata).Verify(func);
}

tvm::transform::Pass VerifyReducerEpochs() {
  auto pass_func = [](PrimFunc func, IRModule, tvm::transform::PassContext) {
    ReducerMetadata metadata = CollectReducerMetadata(func);
    VerifyReducerEpochSemantics(func, metadata);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.VerifyReducerEpochs", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.VerifyReducerEpochs",
                        VerifyReducerEpochs);
}

} // namespace tl
} // namespace tvm
