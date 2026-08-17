/*!
 * \file verify_reducer_epoch.cc
 * \brief Verify the lifecycle and access rules of reducer v2 epochs.
 *
 * Enforced rules (first version):
 *   - every `local.reducer` allocation has exactly one reducer_init,
 *     zero or more reducer_update, and exactly one finalize_reducer
 *     (statically: one epoch site per allocation);
 *   - the finalize sits in the init's control-flow scope (the same stack of
 *     enclosing serial loops and conditional branches) or in a conditional
 *     refinement of it, so every dynamic execution of the init is closed by
 *     at most one finalize in the same iteration (a finalize skipped by a
 *     branch leaves the partials unread — harmless). Neither may appear
 *     inside a `T.Parallel` loop. An epoch nested in a serial loop reopens
 *     once per iteration; the enclosing control flow must be thread-uniform,
 *     as with any other collective tile op;
 *   - updates appear only between init and finalize, and only inside a
 *     `T.Parallel` loop;
 *   - the reducer is opaque outside the three first-class ops: ordinary
 *     loads/stores, fill/clear/copy, access_ptr and aliasing are rejected;
 *   - the finalize destination is an ordinary buffer, not a reducer;
 *   - a declared seed has the reducer's dtype.
 *
 * The pass runs early (before warp specialization and pipelining) so
 * diagnostics point at code the user wrote.
 */

#include "support/check.h"
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../op/builtin.h"
#include "../op/reducer.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

enum class EpochState : int { kAllocated = 0, kActive = 1, kFinalized = 2 };

class ReducerEpochVerifier : public StmtExprVisitor {
public:
  void Run(const PrimFunc &f) {
    VisitStmt(f->body);
    for (const auto &[var, state] : state_) {
      if (state == EpochState::kAllocated) {
        LOG(FATAL) << "reducer `" << var
                   << "` is allocated but never initialized; every "
                      "T.alloc_reducer needs exactly one T.reducer_init and "
                      "one T.finalize_reducer.";
      }
      if (state == EpochState::kActive) {
        LOG(FATAL) << "reducer `" << var
                   << "` is initialized but never finalized; call "
                      "T.finalize_reducer(acc, dst) exactly once.";
      }
    }
  }

private:
  // ---- allocation / annotation collection -------------------------------

  void VisitStmt_(const SBlockNode *op) final {
    if (auto anno = op->annotations.Get(attr::kReducerInfoV2)) {
      auto map = anno.value().as<Map<Var, Map<String, Any>>>();
      ICHECK(map) << "malformed reducer_info_v2 annotation";
      for (const auto &[var, info] : map.value()) {
        auto op_str = info.Get("op");
        ICHECK(op_str) << "reducer_info_v2 for `" << var
                       << "` is missing the combine op";
        // Validates the op string (fatal on unknown ops).
        ParseReducerV2OpType(op_str.value().cast<String>());
        info_.emplace(var.get(), info);
        state_.emplace(var, EpochState::kAllocated);
      }
    }
    for (const auto &buffer : op->alloc_buffers) {
      if (IsReducerV2Buffer(buffer)) {
        ICHECK(info_.count(buffer->data.get()))
            << "buffer `" << buffer
            << "` is allocated in scope local.reducer but has no "
               "reducer_info_v2 annotation; allocate reducers with "
               "T.alloc_reducer.";
        var_to_buffer_.emplace(buffer->data.get(), buffer);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  // ---- control-flow tracking ---------------------------------------------

  // One frame per enclosing loop or conditional branch. `branch`
  // distinguishes the then (0) and else (1) arms of a conditional: an epoch
  // opened in one arm may not close in the other.
  struct ContextFrame {
    const Object *node;
    int branch;
    bool operator==(const ContextFrame &other) const {
      return node == other.node && branch == other.branch;
    }
  };

  void VisitStmt_(const ForNode *op) final {
    bool is_parallel = op->kind == ForKind::kParallel;
    parallel_depth_ += is_parallel;
    VisitExpr(op->min);
    VisitExpr(op->extent);
    ctx_stack_.push_back({op, 0});
    VisitStmt(op->body);
    ctx_stack_.pop_back();
    parallel_depth_ -= is_parallel;
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    VisitExpr(op->condition);
    ctx_stack_.push_back({op, 0});
    VisitStmt(op->then_case);
    ctx_stack_.pop_back();
    if (op->else_case) {
      ctx_stack_.push_back({op, 1});
      VisitStmt(op->else_case.value());
      ctx_stack_.pop_back();
    }
  }

  void VisitStmt_(const WhileNode *op) final {
    VisitExpr(op->condition);
    ctx_stack_.push_back({op, 0});
    VisitStmt(op->body);
    ctx_stack_.pop_back();
  }

  // ---- reducer op events / opaque-access enforcement ---------------------

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(ReducerInitOp::Get())) {
      Var var = RegionArgBufferVar(op->args[0], "reducer_init");
      RequireReducer(var, "reducer_init");
      if (parallel_depth_ > 0) {
        LOG(FATAL) << "T.reducer_init on reducer `" << var
                   << "` may not appear inside a T.Parallel loop.";
      }
      epoch_ctx_[var.get()] = ctx_stack_;
      if (op->args.size() >= 2) {
        const Buffer &buffer = var_to_buffer_.at(var.get());
        ICHECK(op->args[1].dtype() == buffer->dtype)
            << "T.reducer_init init value dtype " << op->args[1].dtype()
            << " does not match reducer `" << var << "` dtype "
            << buffer->dtype;
      }
      auto &state = state_.at(var);
      if (state != EpochState::kAllocated) {
        LOG(FATAL) << "double T.reducer_init on reducer `" << var
                   << "`; each allocation supports exactly one epoch.";
      }
      state = EpochState::kActive;
      // The optional init value is an ordinary read expression.
      for (size_t i = 1; i < op->args.size(); ++i) {
        VisitExpr(op->args[i]);
      }
      return;
    }
    if (op->op.same_as(reducer_update())) {
      const auto *load = op->args[0].as<BufferLoadNode>();
      ICHECK(load) << "reducer_update target must be written as "
                      "`acc[indices]` in the first argument position, got "
                   << op->args[0];
      Var var = load->buffer->data;
      RequireReducer(var, "reducer_update");
      auto &state = state_.at(var);
      if (state == EpochState::kAllocated) {
        LOG(FATAL) << "T.reducer_update on reducer `" << var
                   << "` before T.reducer_init.";
      }
      if (state == EpochState::kFinalized) {
        LOG(FATAL) << "T.reducer_update on reducer `" << var
                   << "` after T.finalize_reducer.";
      }
      if (parallel_depth_ == 0) {
        LOG(FATAL)
            << "T.reducer_update on reducer `" << var
            << "` outside any T.Parallel loop is not supported in the "
               "first version: the contribution multiplicity of "
               "thread-uniform execution is ambiguous. Wrap the update in a "
               "T.Parallel loop.";
      }
      // Only the target's indices and the contribution expression are real
      // reads; the target BufferLoad itself is an update descriptor, not a
      // load of the reducer (visiting it would trip the illegal-read check).
      for (const auto &index : load->indices) {
        VisitExpr(index);
      }
      VisitExpr(op->args[1]);
      return;
    }
    if (op->op.same_as(FinalizeReducerV2Op::Get())) {
      Var var = RegionArgBufferVar(op->args[0], "finalize_reducer");
      Var dst_var = RegionArgBufferVar(op->args[1], "finalize_reducer");
      RequireReducer(var, "finalize_reducer");
      auto ctx_it = epoch_ctx_.find(var.get());
      if (ctx_it != epoch_ctx_.end()) {
        RequireConditionalRefinement(ctx_it->second, var);
      }
      if (info_.count(dst_var.get())) {
        LOG(FATAL) << "T.finalize_reducer destination `" << dst_var
                   << "` must be an ordinary fragment, not a reducer.";
      }
      auto &state = state_.at(var);
      if (state == EpochState::kAllocated) {
        LOG(FATAL) << "T.finalize_reducer on reducer `" << var
                   << "` before T.reducer_init.";
      }
      if (state == EpochState::kFinalized) {
        LOG(FATAL) << "double T.finalize_reducer on reducer `" << var << "`.";
      }
      state = EpochState::kFinalized;
      return; // both region args consumed
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (info_.count(op->buffer->data.get())) {
      LOG(FATAL) << "illegal read of reducer `" << op->buffer
                 << "`: a reducer has no readable value before "
                    "T.finalize_reducer; read the finalize destination "
                    "instead.";
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (info_.count(op->buffer->data.get())) {
      LOG(FATAL) << "illegal store to reducer `" << op->buffer
                 << "`: use T.reducer_update(acc[indices], value) instead of "
                    "ordinary writes (including `acc[i] += v`, T.clear and "
                    "T.fill).";
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const VarNode *op) final {
    if (info_.count(op)) {
      LOG(FATAL) << "illegal use of reducer handle `" << op->name_hint
                 << "` (address-of / aliasing / unsupported op): a reducer "
                    "may only appear in T.reducer_init, T.reducer_update and "
                    "T.finalize_reducer.";
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  // ---- helpers ------------------------------------------------------------

  /*! \brief T.finalize_reducer must sit in the init's scope or a conditional
   *  refinement of it: the init context must be a prefix of the current
   *  context, and every extra frame must be a conditional branch. A finalize
   *  under extra conditionals runs 0 or 1 times per init (safe: a skipped
   *  finalize leaves partials unread). Extra LOOP frames would run the
   *  collective repeatedly on already-combined partials and are rejected, as
   *  is a finalize outside the init's scope (it could run without any init).
   */
  void RequireConditionalRefinement(const std::vector<ContextFrame> &init_ctx,
                                    const Var &var) const {
    bool ok = init_ctx.size() <= ctx_stack_.size();
    for (size_t i = 0; ok && i < ctx_stack_.size(); ++i) {
      if (i < init_ctx.size()) {
        ok = init_ctx[i] == ctx_stack_[i];
      } else {
        ok = ctx_stack_[i].node->IsInstance<IfThenElseNode>();
      }
    }
    if (!ok) {
      LOG(FATAL)
          << "T.finalize_reducer on reducer `" << var
          << "` is not in the scope of its T.reducer_init: the epoch must "
             "close in the same loop iteration and branch it opened in, or "
             "in a conditional branch nested inside that scope.";
    }
  }

  Var RegionArgBufferVar(const PrimExpr &arg, const char *who) const {
    if (auto call = arg.as<CallNode>()) {
      if (call->op.same_as(region())) {
        if (auto load = call->args[0].as<BufferLoadNode>()) {
          return load->buffer->data;
        }
      }
    }
    LOG(FATAL) << who
               << ": expected a tl.region argument wrapping a buffer, got "
               << arg;
    return Var(); // unreachable
  }

  void RequireReducer(const Var &var, const char *who) const {
    if (!info_.count(var.get())) {
      LOG(FATAL) << who << ": `" << var
                 << "` is not a reducer; allocate it with T.alloc_reducer.";
    }
  }

  std::unordered_map<const VarNode *, Map<String, Any>> info_;
  std::unordered_map<const VarNode *, Buffer> var_to_buffer_;
  std::unordered_map<Var, EpochState, ObjectPtrHash, ObjectPtrEqual> state_;
  std::unordered_map<const VarNode *, std::vector<ContextFrame>> epoch_ctx_;
  std::vector<ContextFrame> ctx_stack_;
  int parallel_depth_ = 0;
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass VerifyReducerEpoch() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    ReducerEpochVerifier verifier;
    verifier.Run(f);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.VerifyReducerEpoch", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.VerifyReducerEpoch", VerifyReducerEpoch);
}

} // namespace tl
} // namespace tvm
