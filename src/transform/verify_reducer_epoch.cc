/*!
 * \file verify_reducer_epoch.cc
 * \brief Verify the lifecycle and access rules of reducer v2 epochs.
 *
 * Enforced rules (first version):
 *   - every `local.reducer` allocation has exactly one reducer_init,
 *     zero or more reducer_update, and exactly one finalize_reducer;
 *   - init and finalize execute exactly once: they may not appear inside
 *     any loop or conditional;
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

#include "../op/reducer.h"
#include "../op/region.h"

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

  void VisitStmt_(const ForNode *op) final {
    bool is_parallel = op->kind == ForKind::kParallel;
    parallel_depth_ += is_parallel;
    loop_depth_ += 1;
    StmtExprVisitor::VisitStmt_(op);
    loop_depth_ -= 1;
    parallel_depth_ -= is_parallel;
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    if_depth_ += 1;
    StmtExprVisitor::VisitStmt_(op);
    if_depth_ -= 1;
  }

  void VisitStmt_(const WhileNode *op) final {
    loop_depth_ += 1;
    StmtExprVisitor::VisitStmt_(op);
    loop_depth_ -= 1;
  }

  // ---- reducer op events / opaque-access enforcement ---------------------

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(ReducerInitOp::Get())) {
      Var var = RegionArgBufferVar(op->args[0], "reducer_init");
      RequireReducer(var, "reducer_init");
      RequireStraightLine(var, "T.reducer_init");
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
    if (op->op.same_as(ReducerUpdateOp::Get())) {
      Var var = RegionArgBufferVar(op->args[0], "reducer_update");
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
      // Only the contribution expression is a real read; the target region
      // is an update descriptor, not a load of the reducer.
      VisitExpr(op->args[1]);
      return;
    }
    if (op->op.same_as(FinalizeReducerV2Op::Get())) {
      Var var = RegionArgBufferVar(op->args[0], "finalize_reducer");
      Var dst_var = RegionArgBufferVar(op->args[1], "finalize_reducer");
      RequireReducer(var, "finalize_reducer");
      RequireStraightLine(var, "T.finalize_reducer");
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

  Var RegionArgBufferVar(const PrimExpr &arg, const char *who) const {
    if (auto call = arg.as<CallNode>()) {
      if (call->op.same_as(RegionOp::Get())) {
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

  void RequireStraightLine(const Var &var, const char *who) const {
    if (loop_depth_ > 0 || if_depth_ > 0) {
      LOG(FATAL) << who << " on reducer `" << var
                 << "` must execute exactly once and may not appear inside "
                    "a loop or conditional.";
    }
  }

  std::unordered_map<const VarNode *, Map<String, Any>> info_;
  std::unordered_map<const VarNode *, Buffer> var_to_buffer_;
  std::unordered_map<Var, EpochState, ObjectPtrHash, ObjectPtrEqual> state_;
  int parallel_depth_ = 0;
  int loop_depth_ = 0;
  int if_depth_ = 0;
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
