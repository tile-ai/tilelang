/*!
 * \file tl/transform/reducer.cc
 * \brief Verification and materialization passes for deferred reducers.
 */

#include "reducer.h"

#include "../op/deferred_reducer.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "span_utils.h"
#include "support/check.h"

#include <tvm/ir/cast.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

using ReducerInfoMap =
    std::unordered_map<Var, ReducerInfo, ObjectPtrHash, ObjectPtrEqual>;
using ReducerBufferMap =
    std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual>;

struct ReducerMetadata {
  ReducerInfoMap info;
  ReducerBufferMap buffers;
};

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

class ReducerMaterializer : public arith::IRMutatorWithAnalyzer {
public:
  ReducerMaterializer(const ReducerMetadata &metadata,
                      arith::Analyzer *analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer), metadata_(metadata) {
    for (const auto &[var, buffer] : metadata_.buffers) {
      const auto *pointer_type =
          buffer->data->type_annotation.as<PointerTypeNode>();
      ICHECK(pointer_type != nullptr)
          << "Reducer buffer data Var must have a PointerType annotation";
      Type local_type = PointerType(pointer_type->element_type, "local");
      Var local_var(buffer->data->name_hint, local_type, buffer->data->span);
      Buffer local_buffer = buffer;
      local_buffer.CopyOnWrite()->data = local_var;
      buffer_remap_.emplace(buffer, local_buffer);
    }
  }

  static PrimFunc Rewrite(PrimFunc func, const ReducerMetadata &metadata) {
    arith::Analyzer analyzer;
    ReducerMaterializer materializer(metadata, &analyzer);
    PrimFuncNode *func_ptr = func.CopyOnWrite();
    func_ptr->body = materializer.VisitStmt(func->body);
    return func;
  }

private:
  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock block =
        Downcast<SBlock>(arith::IRMutatorWithAnalyzer::VisitStmt_(op));
    SBlockNode *block_ptr = block.CopyOnWrite();
    for (size_t i = 0; i < block_ptr->alloc_buffers.size(); ++i) {
      auto it = buffer_remap_.find(block_ptr->alloc_buffers[i]);
      if (it != buffer_remap_.end()) {
        block_ptr->alloc_buffers.Set(i, it->second);
      }
    }
    block_ptr->annotations.erase(attr::kReducerInfo);
    return block;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    bool is_parallel = op->kind == ForKind::kParallel;
    bool starts_replicated_parallel = false;
    if (is_parallel && parallel_depth_ == 0) {
      Optional<Any> layout_ref = op->annotations.Get(attr::kParallelLoopLayout);
      ICHECK(layout_ref.has_value())
          << "Reducer update is inside a T.Parallel loop without an inferred "
             "loop layout"
          << SpanHintSuffix(op->span);
      Fragment layout = layout_ref.value().cast<Fragment>();
      starts_replicated_parallel =
          !analyzer_->CanProveEqual(layout->ReplicateExtent(), Integer(1));
    }
    bool variant_bounds =
        !is_parallel &&
        (IsReplicaVariantValue(op->min) || IsReplicaVariantValue(op->extent) ||
         (op->step.defined() && IsReplicaVariantValue(op->step.value())));
    parallel_depth_ += is_parallel ? 1 : 0;
    replicated_parallel_depth_ += starts_replicated_parallel ? 1 : 0;
    replica_variant_control_depth_ += variant_bounds ? 1 : 0;
    Stmt result = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    replica_variant_control_depth_ -= variant_bounds ? 1 : 0;
    replicated_parallel_depth_ -= starts_replicated_parallel ? 1 : 0;
    parallel_depth_ -= is_parallel ? 1 : 0;
    return result;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    bool variant_condition = IsReplicaVariantValue(op->condition);
    replica_variant_control_depth_ += variant_condition ? 1 : 0;
    Stmt result = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    replica_variant_control_depth_ -= variant_condition ? 1 : 0;
    return result;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    bool is_physical_thread_scope = false;
    Var physical_thread;
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar thread = Downcast<IterVar>(op->node);
      std::string thread_tag = thread->thread_tag;
      const int64_t *extent = as_const_int(op->value);
      if (thread_tag.rfind("threadIdx.", 0) == 0 &&
          (extent == nullptr || *extent > 1)) {
        physical_thread = thread->var;
        physical_thread_vars_.insert(physical_thread);
        is_physical_thread_scope = true;
      }
    }
    Stmt result = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    if (is_physical_thread_scope) {
      physical_thread_vars_.erase(physical_thread);
    }
    return result;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load =
        Downcast<BufferLoad>(arith::IRMutatorWithAnalyzer::VisitExpr_(op));
    auto it = buffer_remap_.find(op->buffer);
    if (it != buffer_remap_.end()) {
      load.CopyOnWrite()->buffer = it->second;
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store =
        Downcast<BufferStore>(arith::IRMutatorWithAnalyzer::VisitStmt_(op));
    auto it = buffer_remap_.find(op->buffer);
    if (it != buffer_remap_.end()) {
      store.CopyOnWrite()->buffer = it->second;
    }
    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    bool is_init = op->op.same_as(ReducerInitOp::Get());
    bool is_update = op->op.same_as(ReducerUpdateOp::Get());
    bool is_finalize = op->op.same_as(FinalizeReducerOp::Get());
    Optional<ReducerInfo> info;
    if (is_init || is_update || is_finalize) {
      AccessRegion access =
          NormalizeToAccessRegion(op->args[0], kAccessReadWrite);
      auto it = metadata_.info.find(access.region->buffer->data);
      ICHECK(it != metadata_.info.end());
      info = it->second;
    }
    if (is_update && replicated_parallel_depth_ > 0) {
      AccessRegion update_access =
          NormalizeToAccessRegion(op->args[0], kAccessReadWrite);
      bool variant_index = false;
      for (const Range &range : update_access.region->region) {
        variant_index |= IsReplicaVariantValue(range->min);
      }
      ICHECK_EQ(replica_variant_control_depth_, 0)
          << "T.reducer_update is controlled by a replica-dependent condition "
             "or loop inside a replicated T.Parallel layout"
          << SpanHintSuffix(op->span);
      ICHECK(!variant_index && !IsReplicaVariantValue(op->args[1]))
          << "T.reducer_update indices and contribution must be invariant "
             "across physical replicas; use logical T.Parallel indices and "
             "fragment/global/shared values instead of threadIdx or ordinary "
             "thread-private local values"
          << SpanHintSuffix(op->span);
    }

    Call call = Downcast<Call>(arith::IRMutatorWithAnalyzer::VisitExpr_(op));
    if (!info.defined()) {
      return call;
    }
    CallNode *call_ptr = call.CopyOnWrite();
    call_ptr->annotations.Set(
        attr::kReducerType,
        StringImm(std::string(ReduceTypeName(info.value()->combine_type))));
    if (is_finalize && info.value()->seed.defined()) {
      call_ptr->annotations.Set(attr::kReducerSeed, info.value()->seed.value());
    }
    if (is_update) {
      call_ptr->annotations.Set(attr::kReducerParallelOnce,
                                Bool(parallel_depth_ > 0));
    }
    return call;
  }

  const ReducerMetadata &metadata_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffer_remap_;
  bool UsesPhysicalThreadVar(const PrimExpr &expr) const {
    return UsesVar(expr, [&](const VarNode *node) {
      return physical_thread_vars_.count(GetRef<Var>(node)) != 0;
    });
  }

  bool IsReplicaVariantValue(const PrimExpr &expr) const {
    if (UsesPhysicalThreadVar(expr)) {
      return true;
    }
    bool has_thread_private_load = false;
    PostOrderVisit(expr, [&](const ObjectRef &object) {
      if (const auto *load = object.as<BufferLoadNode>()) {
        has_thread_private_load |= IsLocalBuffer(load->buffer, true);
      }
    });
    return has_thread_private_load;
  }

  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> physical_thread_vars_;
  int parallel_depth_{0};
  int replicated_parallel_depth_{0};
  int replica_variant_control_depth_{0};
};

class ReducerLoweredVerifier : public StmtExprVisitor {
private:
  void VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      ICHECK_NE(buffer.scope(), "local.reducer")
          << "local.reducer allocation reached backend lowering: " << buffer;
    }
    ICHECK(!op->annotations.count(attr::kReducerInfo))
        << "Reducer metadata reached backend lowering";
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    ICHECK_NE(op->attr_key, attr::kParallelMultiplicity)
        << "Unconsumed reducer parallel multiplicity marker reached backend "
           "lowering";
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    ICHECK(!op->op.same_as(ReducerInitOp::Get()) &&
           !op->op.same_as(ReducerUpdateOp::Get()) &&
           !op->op.same_as(FinalizeReducerOp::Get()))
        << "First-class reducer op reached backend lowering: "
        << GetRef<Call>(op);
    StmtExprVisitor::VisitExpr_(op);
  }
};

} // namespace

tvm::transform::Pass VerifyReducerEpochs() {
  auto pass_func = [](PrimFunc func, IRModule, tvm::transform::PassContext) {
    ReducerMetadata metadata = ReducerMetadataCollector::Collect(func);
    ReducerEpochVerifier(metadata).Verify(func);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.VerifyReducerEpochs", {});
}

tvm::transform::Pass PlanAndMaterializeReducers() {
  auto pass_func = [](PrimFunc func, IRModule, tvm::transform::PassContext) {
    ReducerMetadata metadata = ReducerMetadataCollector::Collect(func);
    if (metadata.info.empty()) {
      return func;
    }
    // Planning is the final semantic boundary before physical storage is
    // introduced. Re-verify here because earlier pipeline transforms (for
    // example warp specialization or software pipelining) may have changed the
    // execution scope since the frontend verification pass.
    ReducerEpochVerifier(metadata).Verify(func);
    return ReducerMaterializer::Rewrite(std::move(func), metadata);
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.PlanAndMaterializeReducers", {});
}

tvm::transform::Pass VerifyReducerLowered() {
  auto pass_func = [](PrimFunc func, IRModule, tvm::transform::PassContext) {
    ReducerLoweredVerifier()(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.VerifyReducerLowered", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef()
      .def("tl.transform.VerifyReducerEpochs", VerifyReducerEpochs)
      .def("tl.transform.PlanAndMaterializeReducers",
           PlanAndMaterializeReducers)
      .def("tl.transform.VerifyReducerLowered", VerifyReducerLowered);
}

} // namespace tl
} // namespace tvm
