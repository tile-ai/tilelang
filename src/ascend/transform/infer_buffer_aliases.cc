/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file infer_buffer_aliases.cc
 * \brief Infer pairwise buffer-reuse compatibility for manual schedules.
 */
#include "buffer_alias.h"
#include "merge_ub_common.h"
#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ascend/op/builtin.h"
#include "ascend_pipe.h"
#include "core_mask.h"
#include "op/builtin.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

using namespace merge_ub;

// Sync-Aware Linearizer

enum class EventKind : uint8_t {
  kBufferAccess,  // BufferLoad or BufferStore
  kSetFlag,       // ascend_set_flag
  kWaitFlag,      // ascend_wait_flag
  kPipeBarrier,   // ascend_pipe_barrier
  kCrossCoreSet,  // ascend_cross_core_set_flag
  kCrossCoreWait, // ascend_cross_core_wait_flag
  kForBegin,      // entry of a For loop body
  kForEnd,        // exit of a For loop body
  kIfBegin,       // entry of IfThenElse then-branch
  kIfEnd,         // exit of IfThenElse
  kElseBegin,     // entry of IfThenElse else-branch
};

// The hardware pipe an event executes on. Textual adjacency only implies a
// genuine hardware happens-before within one pipe: different pipes run
// concurrently and are ordered solely by explicit flag handshakes
// (AddSyncEdges). Program-order wiring (rules 1/2/3 in
// HappensBeforeGraph::AddProgramOrderEdges) must therefore be scoped per
// pipe; wiring across pipes fabricates ordering the hardware never
// provides (e.g. a Fixpipe wait followed by an MTE1 store — the wait does
// not block MTE1 at all).
struct LinearEvent {
  EventKind kind;
  int scope_level; // nesting depth

  // Buffer access
  std::vector<const VarNode *> touched;
  std::vector<bool> is_write; // parallel to touched

  // Sync primitives
  std::string hard_event;
  PrimExpr event_id_expr; // raw event_id expression (may contain loop vars)
  std::string pipe_name;  // for pipe_barrier
  int mode_id = 0;        // for cross_core
  CoreMask core_mask = kCoreUnassigned;
  ResourcePipe exec_pipe = ResourcePipe::kUnknown;

  // Scope markers
  const ForNode *for_node = nullptr;
  const IfThenElseNode *if_node = nullptr;

  // Alloc info
  struct AllocInfo {
    const AllocBufferNode *alloc = nullptr;
    size_t scope_level = 0;
  };
  bool IsBufferAccess() const { return kind == EventKind::kBufferAccess; }
  bool IsSync() const {
    return kind == EventKind::kSetFlag || kind == EventKind::kWaitFlag ||
           kind == EventKind::kPipeBarrier ||
           kind == EventKind::kCrossCoreSet ||
           kind == EventKind::kCrossCoreWait;
  }
};

// Visitor that checks whether a PrimExpr references a Var with "sid" in its
// name_hint — used to detect cross-core sid-dependent event_id expressions.
class SidDetector : public ExprVisitor {
public:
  bool found = false;
  void VisitExpr_(const VarNode *op) final {
    if (std::string(op->name_hint).find("sid") != std::string::npos)
      found = true;
    ExprVisitor::VisitExpr_(op);
  }
};

static bool ContainsSid(const PrimExpr &e) {
  SidDetector d;
  d(e);
  return d.found;
}

class SyncAwareLinearizer : public StmtExprVisitor {
public:
  explicit SyncAwareLinearizer(const std::string &target_scope)
      : target_scope_(target_scope) {}

  std::vector<LinearEvent> linear_events;
  std::unordered_map<const VarNode *, LinearEvent::AllocInfo> alloc_info;
  // Pipe-relevant op names that the shared classifier could not map.
  std::unordered_set<std::string> unknown_instr_ops;

private:
  bool IsTargetMemory(const Var &var) {
    if (!target_scope_.empty())
      return IsScopeMemory(var, target_scope_);
    return IsDynamicSharedMemory(var);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    size_t level = scope_stack_.size();
    const VarNode *buf = op->buffer->data.get();
    if (IsTargetMemory(op->buffer->data)) {
      LinearEvent::AllocInfo info;
      info.alloc = op;
      info.scope_level = level;
      alloc_info[buf] = info;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    LinearEvent begin;
    begin.kind = EventKind::kForBegin;
    begin.scope_level = static_cast<int>(scope_stack_.size());
    begin.for_node = op;
    linear_events.push_back(begin);

    loop_stack_.push_back(op);
    StmtExprVisitor::VisitStmt_(op);
    loop_stack_.pop_back();

    LinearEvent end;
    end.kind = EventKind::kForEnd;
    end.scope_level = static_cast<int>(scope_stack_.size());
    end.for_node = op;
    linear_events.push_back(end);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    LinearEvent if_begin;
    if_begin.kind = EventKind::kIfBegin;
    if_begin.scope_level = static_cast<int>(scope_stack_.size());
    if_begin.if_node = op;
    linear_events.push_back(if_begin);

    StmtExprVisitor::VisitStmt(op->then_case);

    if (op->else_case.defined()) {
      LinearEvent else_begin;
      else_begin.kind = EventKind::kElseBegin;
      else_begin.scope_level = static_cast<int>(scope_stack_.size());
      else_begin.if_node = op;
      linear_events.push_back(else_begin);

      StmtExprVisitor::VisitStmt(op->else_case.value());
    }

    LinearEvent if_end;
    if_end.kind = EventKind::kIfEnd;
    if_end.scope_level = static_cast<int>(scope_stack_.size());
    if_end.if_node = op;
    linear_events.push_back(if_end);
  }

  void VisitStmt_(const SBlockNode *op) final {
    CoreMask saved = current_core_mask_;
    if (op->name_hint == "VECTOR")
      current_core_mask_ = kCoreVector;
    else if (op->name_hint == "CUBE")
      current_core_mask_ = kCoreCube;
    // Scope blocks provide the pipe for bare BufferStore/BufferLoad accesses
    // that are not nested in an instruction call.
    ResourcePipe block_pipe = GetAscendBlockPipe(op->name_hint);
    bool has_block_pipe = block_pipe != ResourcePipe::kUnknown;
    if (has_block_pipe)
      pipe_stack_.push_back(block_pipe);
    StmtExprVisitor::VisitStmt_(op);
    if (has_block_pipe)
      pipe_stack_.pop_back();
    current_core_mask_ = saved;
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent && !in_thread_env_) {
      in_thread_env_ = true;
      scope_stack_.push_back(op);
      StmtExprVisitor::VisitStmt_(op);
      scope_stack_.pop_back();
      in_thread_env_ = false;
    } else if (op->attr_key == tirx::attr::extern_scope) {
      scope_stack_.push_back(op);
      StmtExprVisitor::VisitStmt_(op);
      scope_stack_.pop_back();
    } else if (op->attr_key == s_tir::attr::virtual_thread) {
      scope_stack_.push_back(op);
      StmtExprVisitor::VisitStmt_(op);
      scope_stack_.pop_back();
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    StmtExprVisitor::VisitStmt_(op);
    const VarNode *buf = op->buffer->data.get();
    if (alloc_info.count(buf)) {
      LinearEvent e;
      e.kind = EventKind::kBufferAccess;
      e.scope_level = static_cast<int>(scope_stack_.size());
      e.touched.push_back(buf);
      e.is_write.push_back(true);
      e.core_mask = current_core_mask_;
      e.exec_pipe = CurrentExecPipe();
      linear_events.push_back(e);
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    const VarNode *buf = op->buffer->data.get();
    if (alloc_info.count(buf)) {
      LinearEvent e;
      e.kind = EventKind::kBufferAccess;
      e.scope_level = static_cast<int>(scope_stack_.size());
      e.touched.push_back(buf);
      e.is_write.push_back(false);
      e.core_mask = current_core_mask_;
      e.exec_pipe = CurrentExecPipe();
      linear_events.push_back(e);
    }
  }

  void VisitExpr_(const VarNode *op) final {
    // Direct variable references count as reads for liveness.  Buffer-var
    // arguments of tvm_access_ptr are handled by the CallNode override below
    // (which knows the read/write mask); suppress the generic read here so we
    // do not mislabel every access_ptr operand as a read.
    if (suppress_var_read_.count(op))
      return;
    if (alloc_info.count(op)) {
      LinearEvent e;
      e.kind = EventKind::kBufferAccess;
      e.scope_level = static_cast<int>(scope_stack_.size());
      e.touched.push_back(op);
      e.is_write.push_back(false);
      e.core_mask = current_core_mask_;
      e.exec_pipe = CurrentExecPipe();
      linear_events.push_back(e);
    }
  }

  void VisitStmt_(const EvaluateNode *op) final {
    // Check if this Evaluate wraps a sync Call.
    if (auto *call = op->value.as<CallNode>()) {
      if (TryRecordSync(call))
        return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (TryRecordSync(op))
      return;
    Call call = GetRef<Call>(op);
    uint16_t pipe_mask = GetAscendCallPipeMask(call);
    // Composite instructions expose a pipe for each operand. In particular,
    // gemm_l1 reads L1 inputs on MTE1 and writes its L0C result on Cube.
    if (pipe_mask != kAllResourcePipes && pipe_mask != 0 &&
        !IsSingleResourcePipe(pipe_mask)) {
      for (size_t arg_idx = 0; arg_idx < op->args.size(); ++arg_idx) {
        ResourcePipe operand_pipe = GetAscendCallArgumentPipe(call, arg_idx);
        pipe_stack_.push_back(operand_pipe);
        VisitExpr(op->args[arg_idx]);
        pipe_stack_.pop_back();
      }
      return;
    }

    ResourcePipe inst_pipe = GetAscendCallArgumentPipe(call, 0);
    bool pushed_pipe = false;
    if (inst_pipe != ResourcePipe::kUnknown) {
      pipe_stack_.push_back(inst_pipe);
      pushed_pipe = true;
    } else if (!op->op.same_as(builtin::tvm_access_ptr()) &&
               IsAscendPipeRelevantCall(call)) {
      unknown_instr_ops.insert(call->op.as<Op>().value()->name);
    }
    // tvm_access_ptr(type, buffer_var, offset, extent, rw_mask): record the
    // access with the correct read/write flag from rw_mask (1=read, 2=write,
    // 3=read+write) instead of letting the bare buffer_var count as a read.
    if (op->op.same_as(builtin::tvm_access_ptr()) && op->args.size() >= 5) {
      if (const auto *buf = op->args[1].as<VarNode>()) {
        if (alloc_info.count(buf)) {
          int64_t rw = 0;
          if (const auto *imm = op->args[4].as<IntImmNode>())
            rw = imm->value;
          bool is_read = (rw & 1) != 0;
          bool is_write = (rw & 2) != 0;
          // A pure access_ptr with an unknown/zero mask is treated as a read
          // (conservative: keeps the buffer live) rather than being dropped.
          if (!is_read && !is_write)
            is_read = true;
          LinearEvent e;
          e.kind = EventKind::kBufferAccess;
          e.scope_level = static_cast<int>(scope_stack_.size());
          e.touched.push_back(buf);
          e.is_write.push_back(is_write);
          e.core_mask = current_core_mask_;
          e.exec_pipe = CurrentExecPipe();
          linear_events.push_back(e);
          // Suppress the generic VarNode read for this buffer var while we
          // visit the rest of the access_ptr sub-expression.
          suppress_var_read_.insert(buf);
          for (const PrimExpr &arg : op->args)
            VisitExpr(arg);
          suppress_var_read_.erase(buf);
          return;
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
    if (pushed_pipe) {
      pipe_stack_.pop_back();
    }
  }

  bool TryRecordSync(const CallNode *call) {
    LinearEvent e;
    e.scope_level = static_cast<int>(scope_stack_.size());

    if (call->op.same_as(ascend_set_flag())) {
      e.kind = EventKind::kSetFlag;
      ICHECK_GE(call->args.size(), 2U);
      if (auto *str_imm = call->args[0].as<StringImmNode>()) {
        e.hard_event = str_imm->value;
        e.exec_pipe = GetFlagEndpointPipe(e.hard_event, FlagEndpoint::kSet);
      }
      e.event_id_expr = call->args[1];
      e.core_mask = current_core_mask_;
      linear_events.push_back(e);
      return true;
    }
    if (call->op.same_as(ascend_wait_flag())) {
      e.kind = EventKind::kWaitFlag;
      ICHECK_GE(call->args.size(), 2U);
      if (auto *str_imm = call->args[0].as<StringImmNode>()) {
        e.hard_event = str_imm->value;
        e.exec_pipe = GetFlagEndpointPipe(e.hard_event, FlagEndpoint::kWait);
      }
      e.event_id_expr = call->args[1];
      e.core_mask = current_core_mask_;
      linear_events.push_back(e);
      return true;
    }
    if (call->op.same_as(ascend_pipe_barrier())) {
      e.kind = EventKind::kPipeBarrier;
      ICHECK_GE(call->args.size(), 1U);
      if (auto *str_imm = call->args[0].as<StringImmNode>()) {
        e.hard_event = str_imm->value; // store pipe name in hard_event
        e.exec_pipe = ParseAscendPipe(e.hard_event);
      }
      e.core_mask = current_core_mask_;
      linear_events.push_back(e);
      return true;
    }
    if (call->op.same_as(ascend_cross_core_set_flag())) {
      e.kind = EventKind::kCrossCoreSet;
      ICHECK_GE(call->args.size(), 3U);
      if (auto *int_imm = call->args[0].as<IntImmNode>())
        e.mode_id = static_cast<int>(int_imm->value);
      if (auto *str_imm = call->args[1].as<StringImmNode>()) {
        e.hard_event = str_imm->value; // store pipe name
        e.exec_pipe = ParseAscendPipe(e.hard_event);
      }
      e.event_id_expr = call->args[2];
      e.core_mask = current_core_mask_;
      linear_events.push_back(e);
      return true;
    }
    if (call->op.same_as(ascend_cross_core_wait_flag())) {
      e.kind = EventKind::kCrossCoreWait;
      ICHECK_GE(call->args.size(), 3U);
      if (auto *int_imm = call->args[0].as<IntImmNode>())
        e.mode_id = static_cast<int>(int_imm->value);
      if (auto *str_imm = call->args[1].as<StringImmNode>()) {
        e.hard_event = str_imm->value;
        e.exec_pipe = ParseAscendPipe(e.hard_event);
      }
      e.event_id_expr = call->args[2];
      e.core_mask = current_core_mask_;
      linear_events.push_back(e);
      return true;
    }
    return false;
  }

  std::string target_scope_;
  bool in_thread_env_ = false;
  CoreMask current_core_mask_ = kCoreUnassigned;

  std::vector<ResourcePipe> pipe_stack_;
  ResourcePipe CurrentExecPipe() const {
    return pipe_stack_.empty() ? ResourcePipe::kUnknown : pipe_stack_.back();
  }
  // Buffer vars whose bare VarNode read should be suppressed because the
  // enclosing tvm_access_ptr already recorded the access with its rw_mask.
  std::unordered_set<const VarNode *> suppress_var_read_;
  std::vector<const Object *> scope_stack_; // generic scope tracking
  std::vector<const ForNode *> loop_stack_; // active For loops
};

// Coalesce runs of consecutive buffer-access events into fewer events to keep
// the happens-before instance count (and the unroll budget) small.
static std::vector<LinearEvent>
CoalesceBufferAccessRuns(const std::vector<LinearEvent> &events) {
  std::vector<LinearEvent> out;
  out.reserve(events.size());
  // Within the current run: buffer var → index in `out` of its access node.
  std::unordered_map<const VarNode *, int> run_nodes;
  std::optional<CoreMask> run_core_mask;
  std::optional<ResourcePipe> run_exec_pipe;
  auto reset_run = [&]() {
    run_nodes.clear();
    run_core_mask.reset();
    run_exec_pipe.reset();
  };
  for (const LinearEvent &ev : events) {
    if (ev.kind != EventKind::kBufferAccess) {
      // Any non-access event (sync / barrier / scope marker) ends the run.
      reset_run();
      out.push_back(ev);
      continue;
    }
    if (!run_core_mask.has_value() || ev.core_mask != run_core_mask.value() ||
        !run_exec_pipe.has_value() || ev.exec_pipe != run_exec_pipe.value())
      reset_run();
    run_core_mask = ev.core_mask;
    run_exec_pipe = ev.exec_pipe;
    // Each access event touches exactly one buffer (see SyncAwareLinearizer).
    ICHECK_EQ(ev.touched.size(), 1U);
    const VarNode *buf = ev.touched[0];
    bool w = !ev.is_write.empty() && ev.is_write[0];
    auto it = run_nodes.find(buf);
    if (it == run_nodes.end()) {
      run_nodes[buf] = static_cast<int>(out.size());
      out.push_back(ev);
    } else if (w) {
      LinearEvent &dst = out[it->second];
      if (!dst.is_write.empty())
        dst.is_write[0] = true;
    }
  }
  return out;
}

// Pipeline Unroller

struct UnrollPlan {
  // The For loop this plan covers (nullptr for non-loop / toplevel).
  const ForNode *for_node = nullptr;
  int phys_count = 1;
  // Force accesses in this loop to be treated as co-resident for reuse. This
  // is used by conservative fallbacks that cannot precisely materialize every
  // pipelined iteration but still need whole-loop liveness protection.
  bool force_loop_live = false;
  // phys_for_unrolled[u] = physical iteration index for unrolled iteration u.
  std::vector<int> phys_for_unrolled;
  // For each linear event index, the list of unrolled iterations where it
  // executes (empty if not inside this loop).
  std::unordered_map<int, std::vector<int>> event_unrolled_iters;
  // The loop variable value at each unrolled iteration:
  //   loop_var_val[u] = loop_start + phys_for_unrolled[u] * loop_step
  std::vector<PrimExpr> loop_var_vals;

  // For the outermost (non-loop) scope: the single unrolled iteration.
  static UnrollPlan TopLevel() {
    UnrollPlan p;
    p.phys_for_unrolled.push_back(0);
    return p;
  }
};

// Describes the active physical-iteration range for a block of events.
struct ActiveRange {
  int first_phys = 0; // inclusive
  int last_phys = 0;  // inclusive
  bool valid = false;
};

// Parse `loop_var >= loop_start + K * step` and return K. Zero also
// represents either a zero offset or an unrecognized condition; callers only
// need positive prologue offsets.
static int ParseLowerGuardOffset(const PrimExpr &cond,
                                 const ForNode *for_node) {
  const auto *ge = cond.as<GENode>();
  if (ge == nullptr || ge->a.as<VarNode>() != for_node->loop_var.get())
    return 0;
  const auto *add = ge->b.as<AddNode>();
  if (add == nullptr)
    return 0;
  const auto *offset = add->b.as<IntImmNode>();
  if (offset == nullptr)
    return 0;

  int64_t step = 1;
  if (for_node->step.defined()) {
    if (const auto *step_imm = for_node->step.value().as<IntImmNode>())
      step = step_imm->value;
  }
  if (step != 0)
    return static_cast<int>(offset->value / step);
  return 0;
}

// Evaluate an if-guard condition at a concrete loop-variable value.
// Substitutes the loop var (any Var in the condition that is not "sid") with
// `loop_val`, simplifies, and returns:
//   +1  -> provably true   (event executes at this iteration)
//    0  -> provably false  (event does NOT execute)
//   -1  -> unknown / could not decide
// Used by the full-unroll path so that guarded software-pipeline events
// (e.g. `if 3 <= k`, `if k < 3 and 1 <= k`) are placed in exactly the
// physical iterations where they actually run — which is what makes their
// loop-var-dependent flag ids (k-1, k%2+2, ...) evaluate correctly.
static int EvalGuardAtLoopVal(const PrimExpr &cond, const ForNode *for_node,
                              int64_t loop_val) {
  Map<Var, PrimExpr> subs;
  subs.Set(GetRef<Var>(for_node->loop_var.get()),
           IntImm(for_node->loop_var->dtype, loop_val));
  PrimExpr e = Substitute(cond, subs);
  arith::Analyzer analyzer;
  e = analyzer.Simplify(e);
  if (const auto *imm = e.as<IntImmNode>())
    return imm->value != 0 ? 1 : 0;
  return -1;
}

static UnrollPlan MakeConservativeLoopPlan(const ForNode *for_node,
                                           int phys_count,
                                           int64_t loop_start_val,
                                           int begin_idx, int end_idx) {
  UnrollPlan fallback;
  fallback.for_node = for_node;
  fallback.phys_count = phys_count;
  fallback.force_loop_live = phys_count > 1;
  fallback.phys_for_unrolled.push_back(0);
  fallback.loop_var_vals.push_back(
      IntImm(for_node->loop_var->dtype, loop_start_val));
  for (int i = begin_idx; i <= end_idx; ++i)
    fallback.event_unrolled_iters[i].push_back(0);
  return fallback;
}

// Collect the software-pipeline stage structure implied by how the loop
// variable is used in guards and flag ids. We distinguish:
//   - lower bounds  (C <= k, k >= C, k == C): the prologue depth — a guarded
//     body turns ON at iteration C, so head iterations [loop_start, C] must be
//     materialized individually.
//   - upper bounds  (k < C, k <= C, k == C): the epilogue depth — a guarded
//     body turns OFF near the end, so tail iterations [C-1 .. last] matter;
//     the caller converts this to (phys_count - (C - loop_start)).
//   - additive offsets (k - D, k + D): a flag id at iteration k references
//     stage k∓D, so ±D iterations of context are needed at head and tail.
//   - modulus (k % M): the steady-state id pattern repeats with period M.
// Crucially the natural trip-count bound (k < N) contributes tail depth
// phys_count - N ≈ 0, NOT N, so large loops don't force a large unroll.
class LoopVarConstCollector : public ExprVisitor {
public:
  explicit LoopVarConstCollector(const VarNode *loop_var, int64_t loop_start)
      : loop_var_(loop_var), loop_start_(loop_start) {}

  int max_lower = 0;  // largest (C - loop_start) from C <= k
  int max_upper = -1; // largest C from k < C (in absolute loop-var units)
  int max_offset = 0; // largest |D| from k ± D
  int max_mod = 1;    // largest modulus M from k % M
  int mod_period = 1; // lcm of non-trivial moduli from k % M

  static bool RefersLoopVar(const PrimExpr &e, const VarNode *lv) {
    bool found = false;
    PostOrderVisit(e, [&](const ObjectRef &n) {
      if (n.get() == lv)
        found = true;
    });
    return found;
  }

  void RecordMod(const PrimExpr &a, const PrimExpr &b) {
    if (RefersLoopVar(a, loop_var_)) {
      if (const auto *c = b.as<IntImmNode>()) {
        int mod = static_cast<int>(c->value);
        max_mod = std::max(max_mod, mod);
        if (mod > 1) {
          int gcd = std::gcd(mod_period, mod);
          mod_period = (mod_period / gcd) * mod;
        }
      }
    }
  }
  void VisitExpr_(const ModNode *op) final {
    RecordMod(op->a, op->b);
    ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const FloorModNode *op) final {
    RecordMod(op->a, op->b);
    ExprVisitor::VisitExpr_(op);
  }

  // lower bound: loop_var side >= const
  void RecordLower(int64_t c) {
    max_lower = std::max<int>(max_lower, static_cast<int>(c - loop_start_));
  }
  // upper bound: loop_var side < const  (k <= C is normalized to < C+1)
  void RecordUpper(int64_t c) {
    max_upper = std::max<int>(max_upper, static_cast<int>(c));
  }

  //   a < b
  void VisitExpr_(const LTNode *op) final {
    if (RefersLoopVar(op->a, loop_var_)) {
      if (const auto *c = op->b.as<IntImmNode>())
        RecordUpper(c->value); // k < C
    } else if (RefersLoopVar(op->b, loop_var_)) {
      if (const auto *c = op->a.as<IntImmNode>())
        RecordLower(c->value + 1); // C < k  → k >= C+1
    }
    ExprVisitor::VisitExpr_(op);
  }
  //   a <= b
  void VisitExpr_(const LENode *op) final {
    if (RefersLoopVar(op->a, loop_var_)) {
      if (const auto *c = op->b.as<IntImmNode>())
        RecordUpper(c->value + 1); // k <= C → k < C+1
    } else if (RefersLoopVar(op->b, loop_var_)) {
      if (const auto *c = op->a.as<IntImmNode>())
        RecordLower(c->value); // C <= k
    }
    ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const GTNode *op) final {
    if (RefersLoopVar(op->a, loop_var_)) {
      if (const auto *c = op->b.as<IntImmNode>())
        RecordLower(c->value + 1); // k > C → k >= C+1
    } else if (RefersLoopVar(op->b, loop_var_)) {
      if (const auto *c = op->a.as<IntImmNode>())
        RecordUpper(c->value); // C > k → k < C
    }
    ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const GENode *op) final {
    if (RefersLoopVar(op->a, loop_var_)) {
      if (const auto *c = op->b.as<IntImmNode>())
        RecordLower(c->value); // k >= C
    } else if (RefersLoopVar(op->b, loop_var_)) {
      if (const auto *c = op->a.as<IntImmNode>())
        RecordUpper(c->value + 1); // C >= k → k <= C → k < C+1
    }
    ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const EQNode *op) final {
    // k == C constrains both ends.
    const IntImmNode *c = nullptr;
    if (RefersLoopVar(op->a, loop_var_))
      c = op->b.as<IntImmNode>();
    else if (RefersLoopVar(op->b, loop_var_))
      c = op->a.as<IntImmNode>();
    if (c) {
      RecordLower(c->value);
      RecordUpper(c->value + 1);
    }
    ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const SubNode *op) final {
    if (RefersLoopVar(op->a, loop_var_))
      if (const auto *c = op->b.as<IntImmNode>())
        max_offset = std::max<int>(max_offset, static_cast<int>(c->value));
    ExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const AddNode *op) final {
    if (RefersLoopVar(op->a, loop_var_))
      if (const auto *c = op->b.as<IntImmNode>())
        max_offset = std::max<int>(max_offset, static_cast<int>(c->value));
    if (RefersLoopVar(op->b, loop_var_))
      if (const auto *c = op->a.as<IntImmNode>())
        max_offset = std::max<int>(max_offset, static_cast<int>(c->value));
    ExprVisitor::VisitExpr_(op);
  }

private:
  const VarNode *loop_var_;
  int64_t loop_start_;
};

// Build an UnrollPlan for a pipelined loop.
//
// We walk the linear events between kForBegin / kForEnd and examine
// kIfBegin conditions to determine the stage active ranges.
static UnrollPlan BuildLoopUnrollPlan(const ForNode *for_node,
                                      const std::vector<LinearEvent> &events,
                                      int begin_idx, int end_idx) {
  UnrollPlan plan;
  plan.for_node = for_node;

  int64_t phys_extent_val = 1;
  if (auto *ext_imm = for_node->extent.as<IntImmNode>())
    phys_extent_val = ext_imm->value;
  else
    phys_extent_val = 256; // conservative fallback
  plan.phys_count = static_cast<int>(phys_extent_val);

  int64_t step_val = 1;
  if (for_node->step.defined())
    if (auto *s = for_node->step.value().as<IntImmNode>())
      step_val = s->value;

  int64_t loop_start_val = 0;
  if (auto *s = for_node->min.as<IntImmNode>())
    loop_start_val = s->value;
  DataType loop_var_dtype = for_node->loop_var->dtype;

  // Bounded unroll path for software-pipelined loops.
  //
  // A loop annotated `enable_offset` was software-pipelined by the scheduler:
  // its set/wait flag ids are loop-var-dependent (k, k-1, k%2+2, ...) and its
  // body statements are wrapped in stage guards (`if 3 <= k`,
  // `if k < 3 and 1 <= k`, ...). Correct happens-before recovery needs each
  // materialized iteration to carry its true k value so every flag id
  // evaluates to the concrete number the hardware uses and set/wait pair up.
  // The old prologue/steady/epilogue approximation produced only 2 unrolled
  // iterations (k∈{0,1}) with mis-parsed bounds, so ids like `k-1` evaluated
  // to -1 and distinct iterations aliased onto the same id.
  //
  // We do NOT fully unroll (the extent can be large, e.g. 32 KV blocks →
  // O(N^3) reachability blow-up). Instead we unroll only as many head and
  // tail iterations as the loop-var-related constants require: the largest
  // comparison bound / modulus / additive offset found in the guards and flag
  // ids determines the stage depth. Head iterations [0, B) cover the prologue
  // where guards switch on; tail iterations [phys-B, phys) cover the epilogue;
  // between them we materialize `period` aligned steady iterations that repeat
  // the id pattern. This keeps every flag id exact where it matters while
  // bounding the number of unrolled iterations.
  bool enable_offset = false;
  if (auto ann = for_node->annotations.Get("enable_offset"))
    enable_offset = ann.value().cast<bool>();
  if (enable_offset && for_node->extent.as<IntImmNode>() &&
      plan.phys_count >= 1) {
    // 1) Scan guards + flag-id expressions for loop-var-related constants.
    LoopVarConstCollector coll(for_node->loop_var.get(), loop_start_val);
    for (int i = begin_idx; i <= end_idx; ++i) {
      const LinearEvent &ev = events[i];
      if (ev.kind == EventKind::kIfBegin && ev.if_node) {
        coll(ev.if_node->condition);
      }
      if ((ev.kind == EventKind::kSetFlag || ev.kind == EventKind::kWaitFlag ||
           ev.kind == EventKind::kCrossCoreSet ||
           ev.kind == EventKind::kCrossCoreWait) &&
          ev.event_id_expr.defined()) {
        coll(ev.event_id_expr);
      }
    }
    // Head depth: prologue guards (C <= k) plus additive-offset context.
    // Tail depth: how many iterations before the end an upper-bound guard
    // (k < C) switches off, i.e. phys_count - (C - loop_start); plus offset.
    int head_depth = coll.max_lower + coll.max_offset + coll.max_mod;
    int tail_depth = coll.max_offset + coll.max_mod;
    if (coll.max_upper >= 0) {
      int upper_from_end =
          plan.phys_count - (coll.max_upper - static_cast<int>(loop_start_val));
      tail_depth =
          std::max(tail_depth, upper_from_end + coll.max_offset + coll.max_mod);
    }
    int period = std::max(coll.mod_period, 1);
    head_depth = std::max(head_depth, period);
    tail_depth = std::max(tail_depth, period);

    // 2) Choose the set of physical iterations to materialize:
    //    head [0, head_depth) + `period` aligned steady iters + tail.
    std::vector<int> phys_list;
    auto add_phys = [&](int p) {
      if (p >= 0 && p < plan.phys_count)
        phys_list.push_back(p);
    };
    for (int p = 0; p < head_depth; ++p)
      add_phys(p);
    // Steady sample: `period` iterations right after the head, so every
    // modulus residue appears at least once with an exact k value.
    for (int s = 0; s < period; ++s)
      add_phys(head_depth + s);
    for (int p = plan.phys_count - tail_depth; p < plan.phys_count; ++p)
      add_phys(p);
    std::sort(phys_list.begin(), phys_list.end());
    phys_list.erase(std::unique(phys_list.begin(), phys_list.end()),
                    phys_list.end());

    // Hard cap on total analysis work. The happens-before closure is
    // superlinear in the instance count (≈ events_in_loop × materialized
    // iterations). Loops that exceed the budget fall back to a conservative
    // single-instance plan with whole-loop liveness protection.
    constexpr int kMaxInstances = 8000;
    int events_in_loop = end_idx - begin_idx + 1;
    if (static_cast<int64_t>(events_in_loop) *
            static_cast<int64_t>(phys_list.size()) >
        kMaxInstances) {
      LOG(WARNING) << "InferBufferAliases precise unroll budget exceeded for "
                   << "loop " << for_node->loop_var->name_hint << ": "
                   << events_in_loop << " events * " << phys_list.size()
                   << " iterations > " << kMaxInstances
                   << "; falling back to conservative whole-loop liveness";
      return MakeConservativeLoopPlan(for_node, plan.phys_count, loop_start_val,
                                      begin_idx, end_idx);
    }

    // 3) Build the unrolled-iteration table.
    for (int p : phys_list) {
      plan.phys_for_unrolled.push_back(p);
      plan.loop_var_vals.push_back(
          IntImm(loop_var_dtype, loop_start_val + p * step_val));
    }

    // 4) Track enclosing if-conditions per event, then include each event only
    //    in the materialized iterations where all its guards hold.
    std::vector<std::vector<PrimExpr>> cond_stack_by_event(events.size());
    std::vector<PrimExpr> cond_stack;
    for (int i = begin_idx; i <= end_idx; ++i) {
      const LinearEvent &ev = events[i];
      if (ev.kind == EventKind::kIfBegin) {
        cond_stack_by_event[i] = cond_stack;
        if (ev.if_node)
          cond_stack.push_back(ev.if_node->condition);
      } else if (ev.kind == EventKind::kIfEnd) {
        if (!cond_stack.empty())
          cond_stack.pop_back();
        cond_stack_by_event[i] = cond_stack;
      } else {
        cond_stack_by_event[i] = cond_stack;
      }
    }

    int total = static_cast<int>(plan.phys_for_unrolled.size());
    for (int i = begin_idx; i <= end_idx; ++i) {
      for (int u = 0; u < total; ++u) {
        int64_t kval = loop_start_val + plan.phys_for_unrolled[u] * step_val;
        bool active = true;
        for (const PrimExpr &c : cond_stack_by_event[i]) {
          int guard_state = EvalGuardAtLoopVal(c, for_node, kval);
          if (guard_state < 0) {
            return MakeConservativeLoopPlan(for_node, plan.phys_count,
                                            loop_start_val, begin_idx, end_idx);
          }
          if (guard_state == 0) {
            active = false;
            break;
          }
        }
        if (active)
          plan.event_unrolled_iters[i].push_back(u);
      }
    }
    return plan;
  }

  // Collect per-event active ranges from If-guards.
  // For events not inside any If-guard, the range is [0, phys_count-1].
  // We track the "current active range" that applies to events inside an
  // If-block.
  std::vector<ActiveRange> event_ranges(events.size());
  std::vector<ActiveRange> range_stack; // stack of active ranges

  for (int i = begin_idx; i <= end_idx; ++i) {
    const LinearEvent &ev = events[i];
    if (ev.kind == EventKind::kIfBegin) {
      ActiveRange r;
      r.first_phys = 0;
      r.last_phys = plan.phys_count - 1;
      // Try to parse stage guard from the if-condition.
      if (ev.if_node) {
        const PrimExpr &cond = ev.if_node->condition;
        // Check for loop_var >= loop_start + K
        int lb_off = ParseLowerGuardOffset(cond, for_node);
        // Check for conjunction
        if (auto *and_node = cond.as<AndNode>()) {
          lb_off = ParseLowerGuardOffset(and_node->a, for_node);
          if (lb_off == 0)
            lb_off = ParseLowerGuardOffset(and_node->b, for_node);
        }
        if (lb_off > 0)
          r.first_phys = std::max(r.first_phys, lb_off);
      }
      r.valid = true;
      range_stack.push_back(r);
      event_ranges[i] = r;
    } else if (ev.kind == EventKind::kIfEnd) {
      if (!range_stack.empty())
        range_stack.pop_back();
      event_ranges[i] = range_stack.empty()
                            ? ActiveRange{0, plan.phys_count - 1, true}
                            : range_stack.back();
    } else if (ev.kind == EventKind::kElseBegin) {
      // For else branch, use the same range as the if-begin (conservative).
      event_ranges[i] = range_stack.empty()
                            ? ActiveRange{0, plan.phys_count - 1, true}
                            : range_stack.back();
    } else {
      event_ranges[i] = range_stack.empty()
                            ? ActiveRange{0, plan.phys_count - 1, true}
                            : range_stack.back();
    }
  }

  // Determine if this is a pipelined loop: check for sync events inside.
  bool has_set = false, has_wait = false;
  for (int i = begin_idx; i <= end_idx; ++i) {
    if (events[i].kind == EventKind::kSetFlag ||
        events[i].kind == EventKind::kCrossCoreSet)
      has_set = true;
    if (events[i].kind == EventKind::kWaitFlag ||
        events[i].kind == EventKind::kCrossCoreWait)
      has_wait = true;
  }

  if (!has_set || !has_wait) {
    // Non-pipelined loop: single unrolled iteration.
    plan.phys_for_unrolled.push_back(0);
    int64_t start_val = 0;
    if (auto *s = for_node->min.as<IntImmNode>())
      start_val = s->value;
    plan.loop_var_vals.push_back(IntImm(loop_var_dtype, start_val));
    for (int i = begin_idx; i <= end_idx; ++i)
      plan.event_unrolled_iters[i].push_back(0);
    return plan;
  }

  // Pipelined: determine prologue / steady / epilogue boundaries.
  int max_first = 0;
  int min_last = plan.phys_count - 1;
  for (int i = begin_idx; i <= end_idx; ++i) {
    if (event_ranges[i].valid) {
      max_first = std::max(max_first, event_ranges[i].first_phys);
      min_last = std::min(min_last, event_ranges[i].last_phys);
    }
  }

  int prologue_iters = max_first; // physical iters [0, max_first-1] are partial
  int epilogue_iters = plan.phys_count - 1 - min_last;
  constexpr int kSteadyIters = 2;

  const bool unroll_all =
      prologue_iters + kSteadyIters + epilogue_iters > plan.phys_count;

  // Build phys_for_unrolled.
  if (unroll_all) {
    // Unroll all.
    for (int p = 0; p < plan.phys_count; ++p) {
      plan.phys_for_unrolled.push_back(p);
      plan.loop_var_vals.push_back(
          IntImm(loop_var_dtype, loop_start_val + p * step_val));
    }
  } else {
    // Prologue.
    for (int p = 0; p < prologue_iters; ++p) {
      plan.phys_for_unrolled.push_back(p);
      plan.loop_var_vals.push_back(
          IntImm(loop_var_dtype, loop_start_val + p * step_val));
    }
    // Steady state (2 iterations).
    int steady_start = max_first;
    for (int s = 0; s < kSteadyIters; ++s) {
      int p = steady_start + s;
      plan.phys_for_unrolled.push_back(p);
      plan.loop_var_vals.push_back(
          IntImm(loop_var_dtype, loop_start_val + p * step_val));
    }
    // Epilogue.
    int epi_start = min_last - epilogue_iters + 1;
    if (epi_start < steady_start + kSteadyIters)
      epi_start = steady_start + kSteadyIters;
    for (int e = 0; e < epilogue_iters; ++e) {
      int p = epi_start + e;
      if (p >= plan.phys_count)
        break;
      plan.phys_for_unrolled.push_back(p);
      plan.loop_var_vals.push_back(
          IntImm(loop_var_dtype, loop_start_val + p * step_val));
    }
  }

  int total_unrolled = static_cast<int>(plan.phys_for_unrolled.size());

  // Map each event to its unrolled iterations.
  for (int i = begin_idx; i <= end_idx; ++i) {
    const ActiveRange &r = event_ranges[i];
    if (!r.valid)
      continue;
    for (int u = 0; u < total_unrolled; ++u) {
      int phys = plan.phys_for_unrolled[u];
      if (phys >= r.first_phys && phys <= r.last_phys)
        plan.event_unrolled_iters[i].push_back(u);
    }
  }

  return plan;
}

// Happens-Before Graph

struct EventInstance {
  int node_id;
  int linear_idx;    // index into LinearEvent array
  int unrolled_iter; // which unrolled iteration
  int physical_iter; // physical loop iteration
  PrimExpr loop_val; // loop variable value at this instance
  EventKind kind;
  std::vector<const VarNode *> touched;
  std::vector<bool> is_write;
  std::string hard_event;
  PrimExpr event_id_expr;
  int mode_id = 0;
  CoreMask core_mask = kCoreUnassigned;
  ResourcePipe exec_pipe = ResourcePipe::kUnknown;
  // Ids of *all* enclosing pipelined (multi-iteration) loops this instance
  // lives in, innermost last. Empty if the instance is at top level / only in
  // single-iteration loops. Two buffers that are both accessed inside the
  // *same* pipelined loop are rolling / double-buffered and continuously live
  // for that loop, so they must never reuse each other's storage regardless of
  // the stage-offset separation the software pipeliner introduces between their
  // accesses. The full ancestor chain (not just the innermost id) is recorded
  // so that a buffer touched only inside a nested inner loop still shares the
  // enclosing pipelined loop id with sibling buffers touched in other inner
  // loops of the same outer pipeline (e.g. C1's K-tile L0 and C2's V-tile L0,
  // which are concurrent stages of one outer software pipeline).
  std::vector<int> loop_ids;
};

// Recursively build unrolled event instances and the per-loop unroll plan.
//
// Scans the linear events top-down.  For each kForBegin/kForEnd pair,
// builds an UnrollPlan and creates instances for events inside the loop.
// Events at the top level (outside any loop) get a single instance.
static void BuildEventInstancesRecursive(
    const std::vector<LinearEvent> &events, int scope_begin, int scope_end,
    const std::vector<int> &outer_unrolled_iters,
    const std::vector<PrimExpr> &outer_loop_vals,
    std::vector<EventInstance> &instances, const std::vector<int> &cur_loop_ids,
    int *next_loop_id,
    const std::unordered_map<int, std::vector<int>> *event_active_iters =
        nullptr) {

  // Collect all event indices inside this scope that are NOT inside a nested
  // For loop.  Nested For loops are handled recursively.
  std::vector<int> local_event_indices;
  int i = scope_begin;
  while (i <= scope_end) {
    const LinearEvent &ev = events[i];
    if (ev.kind == EventKind::kForBegin) {
      // Find matching kForEnd.
      int depth = 1;
      int j = i + 1;
      for (; j <= scope_end && depth > 0; ++j) {
        if (events[j].kind == EventKind::kForBegin)
          ++depth;
        else if (events[j].kind == EventKind::kForEnd)
          --depth;
      }
      int for_end = j - 1; // points to kForEnd

      UnrollPlan plan = BuildLoopUnrollPlan(ev.for_node, events, i, for_end);

      // Build the set of unrolled iterations for this loop.
      std::vector<int> local_unrolled;
      std::vector<PrimExpr> local_loop_vals;
      for (int u = 0; u < static_cast<int>(plan.phys_for_unrolled.size());
           ++u) {
        local_unrolled.push_back(u);
        local_loop_vals.push_back(plan.loop_var_vals[u]);
      }

      // Recurse into the loop body (between kForBegin+1 and kForEnd).
      // A loop that materializes more than one iteration, or a conservative
      // fallback for a multi-iteration pipelined loop, gets a fresh loop id so
      // buffers accessed inside it are recognized as continuously live. The id
      // is *appended* to the enclosing chain (rather than replacing it) so that
      // accesses nested inside this loop still carry every ancestor pipelined
      // loop id — a buffer touched only in an inner loop must still share the
      // outer pipeline's id with sibling buffers in other inner loops.
      std::vector<int> body_loop_ids = cur_loop_ids;
      if (plan.phys_for_unrolled.size() > 1 || plan.force_loop_live)
        body_loop_ids.push_back((*next_loop_id)++);
      BuildEventInstancesRecursive(events, i + 1, for_end - 1, local_unrolled,
                                   local_loop_vals, instances, body_loop_ids,
                                   next_loop_id, &plan.event_unrolled_iters);
      i = for_end + 1;
    } else {
      local_event_indices.push_back(i);
      ++i;
    }
  }

  // Create instances for local events (not inside any nested For loop).
  if (outer_unrolled_iters.empty()) {
    // Top-level: single instance.
    for (int idx : local_event_indices) {
      const LinearEvent &ev = events[idx];
      EventInstance inst;
      inst.node_id = static_cast<int>(instances.size());
      inst.linear_idx = idx;
      inst.unrolled_iter = 0;
      inst.physical_iter = 0;
      inst.loop_val = IntImm(DataType::Int(32), 0);
      inst.kind = ev.kind;
      inst.touched = ev.touched;
      inst.is_write = ev.is_write;
      inst.hard_event = ev.hard_event;
      inst.event_id_expr = ev.event_id_expr;
      inst.mode_id = ev.mode_id;
      inst.core_mask = ev.core_mask;
      inst.exec_pipe = ev.exec_pipe;
      inst.loop_ids = cur_loop_ids;
      instances.push_back(inst);
    }
  } else {
    for (int u = 0; u < static_cast<int>(outer_unrolled_iters.size()); ++u) {
      for (int idx : local_event_indices) {
        const LinearEvent &ev = events[idx];
        // If the enclosing loop's UnrollPlan restricts this event to specific
        // unrolled iterations (software-pipeline stage guards under full
        // unroll), honor it: only instantiate the event in iterations where
        // its guards hold. Without a plan (or if the event is unlisted), fall
        // back to instantiating in every iteration.
        if (event_active_iters != nullptr) {
          auto it = event_active_iters->find(idx);
          if (it != event_active_iters->end()) {
            const std::vector<int> &active = it->second;
            if (std::find(active.begin(), active.end(), u) == active.end())
              continue; // event does not execute in this iteration
          }
        }
        EventInstance inst;
        inst.node_id = static_cast<int>(instances.size());
        inst.linear_idx = idx;
        inst.unrolled_iter = outer_unrolled_iters[u];
        inst.physical_iter = outer_unrolled_iters[u];
        inst.loop_val = outer_loop_vals[u];
        inst.kind = ev.kind;
        inst.touched = ev.touched;
        inst.is_write = ev.is_write;
        inst.hard_event = ev.hard_event;
        inst.event_id_expr = ev.event_id_expr;
        inst.mode_id = ev.mode_id;
        inst.core_mask = ev.core_mask;
        inst.exec_pipe = ev.exec_pipe;
        inst.loop_ids = cur_loop_ids;
        instances.push_back(inst);
      }
    }
  }
}

// Build the happens-before reachability matrix.
//
// Steps:
// 1. Build event instances from LinearEvent + UnrollPlan.
// 2. Add program-order edges.
// 3. Add sync edges (set_flag → wait_flag with matching event_id).
// 4. Add pipe_barrier edges.
// 5. Add loop-carried edges between consecutive physical iterations.
// 6. Floyd-Warshall for full reachability.
struct HappensBeforeGraph {
  std::vector<EventInstance> instances;
  std::vector<std::vector<bool>> reachable; // [i][j]
  int N = 0;

  void Build(const std::vector<LinearEvent> &events) {
    instances.clear();
    int next_loop_id = 0;
    BuildEventInstancesRecursive(
        events, 0, static_cast<int>(events.size()) - 1, {}, {}, instances,
        /*cur_loop_ids=*/std::vector<int>{}, &next_loop_id);
    N = static_cast<int>(instances.size());
    if (N == 0)
      return;

    // Initialize reachability matrix.
    reachable.assign(N, std::vector<bool>(N, false));
    for (int i = 0; i < N; ++i)
      reachable[i][i] = true;

    AddProgramOrderEdges(events);
    AddSyncEdges();
    AddPipeBarrierEdges(events);
    AddLoopCarriedEdges();
    ComputeTransitiveClosure();
  }

  // Collect *every* node id that touches `var` — no per-iteration first/last
  // collapsing. Used by buffer reuse analysis for the strict "all access
  // pairs" ordering rule.
  void GetAllBufferAccessNodes(const VarNode *var,
                               std::vector<int> &ids) const {
    for (int i = 0; i < N; ++i) {
      const EventInstance &inst = instances[i];
      if (inst.kind != EventKind::kBufferAccess)
        continue;
      for (const VarNode *v : inst.touched) {
        if (v == var) {
          ids.push_back(i);
          break;
        }
      }
    }
  }

  // For each buffer, collect the node ids of its first and last accesses.
  void GetBufferAccessNodes(const VarNode *var, std::vector<int> &first_ids,
                            std::vector<int> &last_ids) const {
    // Track the "earliest" and "latest" access per unrolled iteration.
    // An access is a "first access" if it has no earlier access in the same
    // iteration.  Since we only have happens-before edges (not linear time),
    // we use the program order: among accesses to var in the same unrolled
    // iteration, the one with smallest linear_idx is the first, largest is
    // the last.
    std::unordered_map<int, int> first_per_iter; // unrolled_iter → node_id
    std::unordered_map<int, int> last_per_iter;  // unrolled_iter → node_id

    for (int i = 0; i < N; ++i) {
      const EventInstance &inst = instances[i];
      if (inst.kind != EventKind::kBufferAccess)
        continue;
      bool touches = false;
      for (const VarNode *v : inst.touched) {
        if (v == var) {
          touches = true;
          break;
        }
      }
      if (!touches)
        continue;

      int u = inst.unrolled_iter;
      if (!first_per_iter.count(u) ||
          instances[first_per_iter[u]].linear_idx > inst.linear_idx)
        first_per_iter[u] = i;
      if (!last_per_iter.count(u) ||
          instances[last_per_iter[u]].linear_idx < inst.linear_idx)
        last_per_iter[u] = i;
    }

    for (const auto &kv : first_per_iter)
      first_ids.push_back(kv.second);
    for (const auto &kv : last_per_iter)
      last_ids.push_back(kv.second);

    if (first_ids.empty()) {
      // No access found: buffer is unused.
      first_ids.clear();
      last_ids.clear();
    }
  }

private:
  void AddEdge(int from, int to) {
    if (from >= 0 && from < N && to >= 0 && to < N)
      reachable[from][to] = true;
  }

  void AddProgramOrderEdges(const std::vector<LinearEvent> &events) {
    // Within each unrolled iteration, model program order as release-acquire:
    //   - Sync events (set/wait/pipe_barrier/cross_core_*) form a linear
    //     chain in lin order.
    //   - Each buffer_access is wired to the *nearest* preceding sync (sync
    //     → access) and the *nearest* following sync (access → sync).
    //   - buffer_access ↔ buffer_access edges are deliberately omitted.
    //
    // Why no access↔access edges: two memory accesses with no synchronization
    // between them have NO happens-before relationship — they are racing
    // unless an explicit sync event sits between them. If we drew a chain
    // "access1 → access2 → access3 → ..." in pure lin order, the transitive
    // closure would falsely report that access1 happens-before access3 even
    // when nothing prevents reordering. That false closure is what made the
    // buffer analysis add ordering edges (e.g. temp1 → temp3) in kernels that
    // use mutex-style buffer-pool primitives like ascend_get_buf/rls_buf, which
    // we do not model — leading to incorrect lifetime analysis and overlapping
    // packing.
    //
    // With this model, two accesses can only reach each other through sync
    // events between them. Code with no inter-access sync (e.g. the mutex
    // example) yields fully unordered access pairs → the all-pair rule never
    // fires → no buffer pair is marked reusable, which is the safe answer when
    // synchronization is invisible to us.
    //
    // Release-acquire wiring is done per concrete core mask. In a Mix kernel
    // the Cube and Vector instruction streams are interleaved in the
    // single linear event list but execute concurrently, so an AIV buffer
    // access must NOT be wired to the nearest AIC sync (or vice-versa) merely
    // because they are adjacent in linear order — that would fabricate an
    // intra-core happens-before across the core boundary and, after transitive
    // closure, make almost every AIC node precede every AIV node (observed to
    // overlap two simultaneously-live L1 buffers). Genuine cross-core ordering
    // is established separately in AddSyncEdges by pairing CrossCore set→wait.
    // Nodes with kCoreUnassigned (single-core / non-Mix kernels, where the core
    // is undetermined) all share one group, reproducing the previous behavior.
    auto is_sync = [&](int idx) {
      EventKind k = instances[idx].kind;
      return k == EventKind::kSetFlag || k == EventKind::kWaitFlag ||
             k == EventKind::kPipeBarrier || k == EventKind::kCrossCoreSet ||
             k == EventKind::kCrossCoreWait;
    };
    auto is_release = [&](int idx) {
      EventKind k = instances[idx].kind;
      return k == EventKind::kSetFlag || k == EventKind::kCrossCoreSet ||
             k == EventKind::kPipeBarrier;
    };
    auto is_acquire = [&](int idx) {
      EventKind k = instances[idx].kind;
      return k == EventKind::kWaitFlag || k == EventKind::kCrossCoreWait ||
             k == EventKind::kPipeBarrier;
    };

    // Group by (unrolled_iter, core_mask, exec_pipe) so wiring never crosses
    // the AIC/AIV boundary and never crosses a pipe boundary. Textual adjacency
    // only implies a hardware happens-before within one pipe: different pipes
    // execute concurrently and are ordered solely by flag handshakes
    // (AddSyncEdges). Events whose pipe is unknown are excluded from
    // program-order wiring entirely — conservative: fewer edges can only reject
    // reuse, never fabricate it.
    std::map<std::tuple<int, CoreMask, ResourcePipe>, std::vector<int>>
        by_iter_core_pipe;
    for (int i = 0; i < N; ++i) {
      if (instances[i].exec_pipe == ResourcePipe::kUnknown)
        continue;
      by_iter_core_pipe[{instances[i].unrolled_iter, instances[i].core_mask,
                         instances[i].exec_pipe}]
          .push_back(i);
    }

    for (auto &kv : by_iter_core_pipe) {
      auto &nodes = kv.second;
      std::sort(nodes.begin(), nodes.end(), [this](int a, int b) {
        return instances[a].linear_idx < instances[b].linear_idx;
      });

      // 1) Sync chain (within this core and pipe).
      int prev_sync = -1;
      for (int idx : nodes) {
        if (!is_sync(idx))
          continue;
        if (prev_sync != -1)
          AddEdge(prev_sync, idx);
        prev_sync = idx;
      }

      // 2) Each access ← nearest preceding *acquire* sync (within this core).
      // A wait/barrier acquires: accesses after it happen-after the paired set.
      prev_sync = -1;
      for (int idx : nodes) {
        if (is_acquire(idx)) {
          prev_sync = idx;
        } else if (!is_sync(idx) && prev_sync != -1) {
          AddEdge(prev_sync, idx);
        }
      }

      // 3) Each access → nearest following *release* sync (within this core).
      // A set/barrier releases: accesses before it happen-before the paired
      // wait. A wait_flag does NOT release, so an access does not become
      // ordered merely because a wait follows it.
      int next_sync = -1;
      for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        int idx = *it;
        if (is_release(idx)) {
          next_sync = idx;
        } else if (!is_sync(idx) && next_sync != -1) {
          AddEdge(idx, next_sync);
        }
      }
    }
  }

  void AddSyncEdges() {
    // Pair each set_flag with the *next* wait_flag in program order that
    // shares the same (hard_event, mode_id, is_cross_core, evaluated_event_id).
    //
    // Background: kernels often reuse the same event_id sequentially —
    // e.g.   set V_MTE3,0 ; wait V_MTE3,0 ; ... ; set V_MTE3,0 ; wait V_MTE3,0
    // A naive Cartesian-product match would draw an edge from the SECOND set
    // to the FIRST wait, which combined with program order creates a HB cycle
    // and makes everything reachable from everything, so overlapping
    // lifetimes become invisible to pairwise reuse analysis.
    //
    // The hardware semantics is "next wait consumes the most recent matching
    // set"; we model it the same way. For instances inside a pipelined loop,
    // EvaluateEventId substitutes the (constant) loop_val so different
    // unrolled iterations produce different numeric ids and naturally fall
    // into distinct groups.
    struct SyncKey {
      std::string hard_event;
      int mode_id;
      bool is_cross_core;
      int64_t event_id_val;
      bool operator==(const SyncKey &o) const {
        return hard_event == o.hard_event && mode_id == o.mode_id &&
               is_cross_core == o.is_cross_core &&
               event_id_val == o.event_id_val;
      }
    };
    struct SyncKeyHash {
      size_t operator()(const SyncKey &k) const {
        size_t h = std::hash<std::string>()(k.hard_event);
        h ^= std::hash<int>()(k.mode_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>()(k.is_cross_core) + 0x9e3779b9 + (h << 6) +
             (h >> 2);
        h ^= std::hash<int64_t>()(k.event_id_val) + 0x9e3779b9 + (h << 6) +
             (h >> 2);
        return h;
      }
    };

    // Bucket every set/wait instance by its fully-evaluated key. If sid is
    // present we expand it (sid ∈ {0,1}) and put the instance into both
    // buckets — sid is double-buffer state, so an event tagged "sid=*" can
    // legitimately pair with either parity.
    struct Endpoint {
      int idx;
      bool is_set;
    };
    std::unordered_map<SyncKey, std::vector<Endpoint>, SyncKeyHash> buckets;

    auto add_endpoint = [&](int idx, bool is_set, bool is_cross,
                            const std::string &hard_event, int mode_id) {
      EventInstance &inst = instances[idx];
      bool has_sid = ContainsSid(inst.event_id_expr);
      int sid_lo = 0, sid_hi = has_sid ? 2 : 1;
      for (int sid_val = sid_lo; sid_val < sid_hi; ++sid_val) {
        int64_t val = 0;
        if (!EvaluateEventId(inst.event_id_expr, inst.loop_val, sid_val,
                             has_sid, &val))
          continue;
        SyncKey key{hard_event, mode_id, is_cross, val};
        buckets[key].push_back(Endpoint{idx, is_set});
      }
    };

    for (int i = 0; i < N; ++i) {
      const EventInstance &inst = instances[i];
      switch (inst.kind) {
      case EventKind::kSetFlag:
        add_endpoint(i, true, false, inst.hard_event, 0);
        break;
      case EventKind::kWaitFlag:
        add_endpoint(i, false, false, inst.hard_event, 0);
        break;
      case EventKind::kCrossCoreSet:
        // For cross-core flags the `pipe` arg of set/wait is LOCAL to each
        // side (e.g. AIC sets through PIPE_FIX, AIV waits through PIPE_MTE3),
        // so pipe name is intentionally excluded from the bucket key —
        // pairing is by (mode_id, flag_id) only.
        add_endpoint(i, true, true, "", inst.mode_id);
        break;
      case EventKind::kCrossCoreWait:
        add_endpoint(i, false, true, "", inst.mode_id);
        break;
      default:
        break;
      }
    }

    // Within each bucket, pair endpoints by happens-before semantics. For
    // intra-core flags the rule is purely textual ("next wait consumes the
    // most recent set" within the single core). For cross-core flags, AIC
    // and AIV are CONCURRENT tracks: a wait on one side may textually
    // precede the set on the other side that releases it. We pair per-
    // (setter_core, waiter_core) by position — the i-th set on the setter side
    // pairs with the i-th wait on the waiter side, regardless of global
    // textual order.
    for (auto &kv : buckets) {
      const SyncKey &key = kv.first;
      auto &eps = kv.second;
      // Order endpoints by (unrolled_iter, linear_idx, node_id). unrolled_iter
      // MUST be the primary key: positional pairing below wires the k-th set on
      // a track to the k-th wait, so both tracks have to enumerate iterations
      // in lockstep. linear_idx alone is NOT enough — it is the source
      // LinearEvent index and is identical across every unrolled iteration of
      // the same event, so a loop-invariant (constant) cross-core flag id
      // issued inside a software-pipelined loop drops every iteration's
      // set/wait into this one bucket with equal linear_idx. std::sort is not
      // stable, so without the unrolled_iter key different iterations'
      // instances interleave nondeterministically and pair_across can wire
      // iteration 2's set to iteration 0's wait, fabricating a cross-core
      // happens-before the hardware never guarantees. node_id is the final
      // tiebreak for full determinism.
      std::sort(eps.begin(), eps.end(),
                [this](const Endpoint &a, const Endpoint &b) {
                  const EventInstance &ia = instances[a.idx];
                  const EventInstance &ib = instances[b.idx];
                  if (ia.unrolled_iter != ib.unrolled_iter)
                    return ia.unrolled_iter < ib.unrolled_iter;
                  if (ia.linear_idx != ib.linear_idx)
                    return ia.linear_idx < ib.linear_idx;
                  return ia.node_id < ib.node_id;
                });

      if (key.is_cross_core) {
        auto collect = [&](CoreMask core, bool is_set) {
          std::vector<int> out;
          for (const Endpoint &e : eps) {
            if (e.is_set == is_set && instances[e.idx].core_mask == core)
              out.push_back(e.idx);
          }
          return out; // already in per-track linear order
        };
        auto pair_across = [&](CoreMask setter_core, CoreMask waiter_core) {
          std::vector<int> sets = collect(setter_core, true);
          std::vector<int> waits = collect(waiter_core, false);
          size_t n = std::min(sets.size(), waits.size());
          for (size_t k = 0; k < n; ++k)
            AddEdge(sets[k], waits[k]);
        };
        pair_across(kCoreCube, kCoreVector); // AIC set → AIV wait
        pair_across(kCoreVector, kCoreCube); // AIV set → AIC wait

        // Defensive fallback for cross-core endpoints outside any
        // CUBE/VECTOR block (unassigned): pair them among themselves by
        // textual order, so we don't double-pair with the edges above.
        std::vector<int> pending;
        for (const Endpoint &ep : eps) {
          if (instances[ep.idx].core_mask != kCoreUnassigned)
            continue;
          if (ep.is_set) {
            pending.push_back(ep.idx);
          } else if (!pending.empty()) {
            AddEdge(pending.front(), ep.idx);
            pending.erase(pending.begin());
          }
        }
      } else {
        // Intra-core: each wait pairs with the most recent unmatched set
        // on the same core.
        std::vector<int> pending_sets;
        for (const Endpoint &ep : eps) {
          if (ep.is_set) {
            pending_sets.push_back(ep.idx);
          } else if (!pending_sets.empty()) {
            int s_idx = pending_sets.front();
            pending_sets.erase(pending_sets.begin());
            AddEdge(s_idx, ep.idx);
          }
          // Dangling wait: leave unmatched.
        }
      }
    }
  }

  void AddPipeBarrierEdges(const std::vector<LinearEvent> &events) {
    // Collect pipe_barrier instances, grouped by unrolled iteration.
    std::unordered_map<int, std::vector<int>> barriers_by_iter;
    for (int i = 0; i < N; ++i) {
      if (instances[i].kind == EventKind::kPipeBarrier)
        barriers_by_iter[instances[i].unrolled_iter].push_back(i);
    }

    // A pipe_barrier is a PER-CORE pipe fence: it orders only the instruction
    // stream of the core that issued it. On a Mix kernel the Cube and Vector
    // streams execute concurrently, so a barrier on one core must NOT be
    // wired to the other core's nodes (that would fabricate an intra-core
    // happens-before across the AIC/AIV boundary). We therefore only connect a
    // barrier to instances sharing its core mask. An unassigned barrier
    // (single-core / non-Mix kernel, where the core is undetermined) connects
    // all unassigned nodes, reproducing the previous behavior.
    for (auto &kv : barriers_by_iter) {
      int u = kv.first;
      for (int b_node : kv.second) {
        int b_linear = instances[b_node].linear_idx;
        CoreMask barrier_core = instances[b_node].core_mask;

        // Nodes on the barrier's own pipe issued before it in this iteration
        // -> barrier. All barrier wiring — incoming AND outgoing — is gated on
        // the barrier's pipe being known: for a kUnknown barrier we cannot be
        // sure it is a valid pipe_barrier at all, and fabricating outgoing
        // edges for it would violate the conservative kUnknown policy used
        // everywhere else in this pass (fewer edges only reject reuse).
        if (instances[b_node].exec_pipe != ResourcePipe::kUnknown) {
          // A kAll barrier (PipeBarrier<PIPE_ALL>) fences every pipe; a
          // per-pipe barrier fences only its own pipe.
          auto pipe_matches = [&](int i) {
            return instances[b_node].exec_pipe == ResourcePipe::kAll ||
                   instances[i].exec_pipe == instances[b_node].exec_pipe;
          };

          // Nodes ordered into the barrier: issued before it on this core,
          // on a pipe the barrier fences.
          for (int i = 0; i < N; ++i) {
            if (instances[i].core_mask == barrier_core &&
                instances[i].unrolled_iter == u && pipe_matches(i) &&
                instances[i].linear_idx < b_linear) {
              AddEdge(i, b_node);
            }
          }

          // Nodes ordered after the barrier: issued after it on this core
          // (this iteration or the next), on a pipe the barrier fences.
          for (int i = 0; i < N; ++i) {
            if (instances[i].core_mask == barrier_core && pipe_matches(i) &&
                ((instances[i].unrolled_iter == u &&
                  instances[i].linear_idx > b_linear) ||
                 instances[i].unrolled_iter == u + 1)) {
              AddEdge(b_node, i);
            }
          }
        }
      }
    }
  }

  void AddLoopCarriedEdges() {
    // Between materialized iterations, all events in the earlier iteration
    // happen-before all events in the later one. We materialize a *sampled*
    // set of physical iterations (head + steady + tail) for software-pipelined
    // loops, so consecutive unrolled iterations u, u+1 may map to
    // non-adjacent physical iterations (e.g. phys 2 then phys 33). That is
    // fine for a sequential loop: iteration 2 still fully precedes iteration
    // 33, so the happens-before edge is valid as long as the physical index
    // strictly increases. Requiring phys_next == phys_u + 1 (the old rule)
    // would leave a gap between the head and tail samples, making a buffer
    // used only in head iterations look disjoint from one used only in tail
    // iterations — and thus falsely reusable — even though both are live
    // across the whole loop (e.g. double-buffered K/V in flash-attention).
    //
    // Loop-carried ordering is per core. On a Mix kernel the Cube and Vector
    // loops are independent parallel sequences that each number their
    // iterations 0,1,2,… in the same unrolled_iter space. Wiring "iter u → iter
    // u+1" across all core masks would fabricate a cross-core happens-before
    // (AIC's
    // iteration k appearing to precede AIV's iteration k+1) that the hardware
    // never guarantees — the only legitimate AIC↔AIV ordering is a CrossCore
    // set→wait pair (added in AddSyncEdges). So we group by (core_mask,
    // unrolled_iter) and add iteration edges only within a single core.
    // kCoreUnassigned (single-core / non-Mix) keeps its own group.
    std::unordered_map<CoreMask, std::map<int, std::vector<int>>>
        nodes_by_core_iter;
    std::unordered_map<CoreMask, std::map<int, int>> phys_by_core_iter;
    for (int i = 0; i < N; ++i) {
      CoreMask core = instances[i].core_mask;
      int u = instances[i].unrolled_iter;
      nodes_by_core_iter[core][u].push_back(i);
      phys_by_core_iter[core][u] = instances[i].physical_iter;
    }

    for (auto &core_kv : nodes_by_core_iter) {
      CoreMask core = core_kv.first;
      auto &nodes_by_iter = core_kv.second;
      auto &phys_for_unrolled = phys_by_core_iter[core];
      for (const auto &it_kv : nodes_by_iter) {
        int u = it_kv.first;
        auto next = nodes_by_iter.find(u + 1);
        if (next == nodes_by_iter.end())
          continue;
        int phys_u = phys_for_unrolled[u];
        int phys_next = phys_for_unrolled[u + 1];
        // Add the edge whenever the later unrolled iteration is a later
        // physical iteration (monotonic progression through the loop).
        if (phys_next <= phys_u)
          continue;
        for (int a : it_kv.second)
          for (int b : next->second)
            AddEdge(a, b);
      }
    }
  }

  void ComputeTransitiveClosure() {
    // Floyd-Warshall with word-level OR. reachable[] is vector<vector<bool>>
    // which cannot be OR'd a word at a time, so mirror each row into a packed
    // uint64 bitset, run the closure there (O(N^3/64)), then write back. This
    // keeps large software-pipelined kernels (N in the thousands) fast; a
    // plain element-wise O(N^3) closure is the compile-time bottleneck there.
    const int W = (N + 63) / 64;
    std::vector<uint64_t> bits(static_cast<size_t>(N) * W, 0);
    auto row = [&](int i) { return bits.data() + static_cast<size_t>(i) * W; };
    for (int i = 0; i < N; ++i) {
      uint64_t *ri = row(i);
      const std::vector<bool> &src = reachable[i];
      for (int j = 0; j < N; ++j)
        if (src[j])
          ri[j >> 6] |= (uint64_t(1) << (j & 63));
    }
    for (int k = 0; k < N; ++k) {
      const uint64_t *rk = row(k);
      const int kw = k >> 6;
      const uint64_t kbit = uint64_t(1) << (k & 63);
      for (int i = 0; i < N; ++i) {
        uint64_t *ri = row(i);
        if (ri[kw] & kbit)
          for (int w = 0; w < W; ++w)
            ri[w] |= rk[w];
      }
    }
    for (int i = 0; i < N; ++i) {
      const uint64_t *ri = row(i);
      std::vector<bool> &dst = reachable[i];
      for (int j = 0; j < N; ++j)
        dst[j] = (ri[j >> 6] >> (j & 63)) & 1ULL;
    }
  }

  // Evaluate an event_id expression by substituting the loop variable value
  // and (optionally) sid.  Returns true and writes the integer value if the
  // expression simplified to a constant.
  static bool EvaluateEventId(const PrimExpr &expr, const PrimExpr &loop_val,
                              int sid_val, bool has_sid, int64_t *out) {
    PrimExpr e = expr;
    // Collect all Vars in the expression that look like loop variables
    // (they appear in the expression as the For loop variable).
    // Substitute them with loop_val.
    class LoopVarFinder : public ExprVisitor {
    public:
      std::vector<Var> loop_vars;
      void VisitExpr_(const VarNode *op) final {
        if (std::string(op->name_hint).find("sid") == std::string::npos)
          loop_vars.push_back(tvm::ffi::GetRef<Var>(op));
        ExprVisitor::VisitExpr_(op);
      }
    };

    LoopVarFinder finder;
    finder(e);

    Map<Var, PrimExpr> subs;
    for (const Var &v : finder.loop_vars)
      subs.Set(v, loop_val);

    if (has_sid) {
      // Find the sid Var and substitute it.
      class SidVarFinder : public ExprVisitor {
      public:
        Var sid_var;
        void VisitExpr_(const VarNode *op) final {
          if (std::string(op->name_hint).find("sid") != std::string::npos)
            sid_var = tvm::ffi::GetRef<Var>(op);
          ExprVisitor::VisitExpr_(op);
        }
      };
      SidVarFinder svf;
      svf(e);
      if (svf.sid_var.defined())
        subs.Set(svf.sid_var, IntImm(DataType::Int(32), sid_val));
    }

    e = Substitute(e, subs);

    arith::Analyzer analyzer;
    e = analyzer.Simplify(e);

    if (auto *imm = e.as<IntImmNode>()) {
      *out = imm->value;
      return true;
    }
    return false;
  }
};

// Pairwise buffer lifetime ordering.

class BufferReuseAnalysis {
public:
  int num_buffers = 0;
  std::vector<const VarNode *> buffer_vars; // index → VarNode*
  // ordered[A][B] means every access to A happens before every access to B.
  std::vector<std::vector<bool>> ordered;

  bool CanReuse(int lhs, int rhs) const {
    // A cycle is not a lifetime separation: the old SCC-based allocator kept
    // both buffers in one non-reusable component in this case.
    return ordered[lhs][rhs] != ordered[rhs][lhs];
  }

  void Build(const HappensBeforeGraph &hb,
             const std::unordered_map<const VarNode *, const AllocBufferNode *>
                 &shmem_allocs,
             const std::vector<const VarNode *> &alloc_order) {
    for (const VarNode *v : alloc_order) {
      if (shmem_allocs.count(v))
        buffer_vars.push_back(v);
    }
    if (buffer_vars.size() < shmem_allocs.size()) {
      std::unordered_set<const VarNode *> seen(buffer_vars.begin(),
                                               buffer_vars.end());
      std::vector<const VarNode *> tail;
      for (const auto &kv : shmem_allocs)
        if (!seen.count(kv.first))
          tail.push_back(kv.first);
      std::sort(tail.begin(), tail.end(),
                [](const VarNode *a, const VarNode *b) {
                  if (a->name_hint != b->name_hint)
                    return a->name_hint < b->name_hint;
                  return a < b;
                });
      buffer_vars.insert(buffer_vars.end(), tail.begin(), tail.end());
    }
    num_buffers = static_cast<int>(buffer_vars.size());
    // Collect ALL access nodes per buffer (not first/last per iter): edge
    // formation must consider every access point, otherwise per-iter
    // aggregation can drop accesses that prove lifetime overlap.
    std::vector<std::vector<int>> all_nodes(num_buffers);
    for (int i = 0; i < num_buffers; ++i)
      hb.GetAllBufferAccessNodes(buffer_vars[i], all_nodes[i]);

    // Build directional whole-lifetime ordering.
    //
    // Edge A → B exists iff EVERY access pair (a ∈ accesses(A),
    // b ∈ accesses(B)) is synchronized in the happens-before graph
    // (i.e. a reaches b).  If any pair is unordered, A and B have
    // overlapping lifetimes and cannot reuse one allocation.
    //
    // Cross-core ordering is trustworthy here: all *structural* edges
    // (program order, pipe_barrier, loop-carried) are built strictly per-core
    // (see AddProgramOrderEdges / AddPipeBarrierEdges / AddLoopCarriedEdges),
    // so the only way reachability crosses the AIC/AIV boundary is through a
    // genuine CrossCore set→wait pairing (AddSyncEdges). A cross-core-witnessed
    // separation therefore reflects a real hardware handshake — e.g. the
    // serialized dual_copy staging buffers in a Cube+Vector kernel, where AIV
    // signals "buf_a consumed" before AIC writes buf_b — and is honored.
    ordered.assign(num_buffers, std::vector<bool>(num_buffers, false));

    // For each buffer, collect the set of pipelined (multi-iteration) loop ids
    // it is accessed inside. Two buffers that both appear inside the same
    // pipelined loop are rolling / double-buffered: the software pipeliner
    // staggers their accesses across stage offsets, but they are continuously
    // live for the whole loop and must never reuse each other's storage. The
    // per-access reachability check below can be fooled into seeing a one-way
    // ordering (all of A's sampled accesses land in head iterations, all of B's
    // in tail iterations) and wrongly grant a reuse edge; this co-residence
    // test overrides that.
    std::vector<std::unordered_set<int>> buf_loops(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
      for (int na : all_nodes[i]) {
        if (na < hb.N)
          for (int lid : hb.instances[na].loop_ids)
            buf_loops[i].insert(lid);
      }
    }
    auto share_pipelined_loop = [&](int a, int b) {
      for (int la : buf_loops[a])
        if (buf_loops[b].count(la))
          return true;
      return false;
    };

    for (int a = 0; a < num_buffers; ++a) {
      if (all_nodes[a].empty())
        continue;
      for (int b = 0; b < num_buffers; ++b) {
        if (a == b)
          continue;
        if (all_nodes[b].empty())
          continue;
        // Both accessed inside the same pipelined loop → continuously live,
        // never reusable.
        if (share_pipelined_loop(a, b))
          continue;
        bool edge = true;
        for (int na : all_nodes[a]) {
          for (int nb : all_nodes[b]) {
            if (na >= hb.N || nb >= hb.N || !hb.reachable[na][nb]) {
              edge = false;
              break;
            }
          }
          if (!edge)
            break;
        }
        if (edge)
          ordered[a][b] = true;
      }
    }
  }
};

/*! \brief Infer pairwise reuse compatibility for one allocation scope. */
BufferAliasMap
InferBufferAliasesForScope(const Stmt &stmt, const AllocationMap &allocations,
                           const AllocationOrder &allocation_order,
                           const std::string &target_scope, bool verbose,
                           bool disable_reuse) {
  if (disable_reuse || allocations.size() < 2) {
    BufferAliasMap aliases;
    for (const auto &[var, _] : allocations)
      AddBufferAliasStorage(&aliases, GetRef<Var>(var));
    return aliases;
  }
  SyncAwareLinearizer linearizer(target_scope);
  linearizer(stmt);
  if (verbose && !linearizer.unknown_instr_ops.empty()) {
    std::vector<std::string> ops(linearizer.unknown_instr_ops.begin(),
                                 linearizer.unknown_instr_ops.end());
    std::sort(ops.begin(), ops.end());

    LOG(WARNING)
        << "InferBufferAliases: instructions with unknown executing pipe for "
           "scope \""
        << target_scope
        << "\" — their accesses are conservatively excluded from "
           "program-order wiring (some reuse may be missed). Add them to "
           "GetAscendCallPipeMask in ascend_pipe.h:";
    for (const std::string &name : ops)
      LOG(WARNING) << " - " << name;
  }
  std::vector<LinearEvent> events =
      CoalesceBufferAccessRuns(linearizer.linear_events);
  HappensBeforeGraph happens_before;
  happens_before.Build(events);

  BufferReuseAnalysis reuse_analysis;
  reuse_analysis.Build(happens_before, allocations, allocation_order);

  BufferAliasMap aliases;
  for (const VarNode *var : reuse_analysis.buffer_vars)
    AddBufferAliasStorage(&aliases, GetRef<Var>(var));
  for (int lhs = 0; lhs < reuse_analysis.num_buffers; ++lhs) {
    for (int rhs = lhs + 1; rhs < reuse_analysis.num_buffers; ++rhs) {
      if (reuse_analysis.CanReuse(lhs, rhs)) {
        AddBufferAlias(&aliases, GetRef<Var>(reuse_analysis.buffer_vars[lhs]),
                       GetRef<Var>(reuse_analysis.buffer_vars[rhs]));
      }
    }
  }
  ValidateBufferAliasMap(aliases);

  if (verbose) {
    LOG(DEBUG) << "Manual-schedule buffer-alias contract for scope \""
               << target_scope << "\":";
    for (int a = 0; a < reuse_analysis.num_buffers; ++a) {
      std::stringstream stream;
      stream << "    " << reuse_analysis.buffer_vars[a]->name_hint << " -> {";
      for (int b = 0; b < reuse_analysis.num_buffers; ++b) {
        if (a != b && reuse_analysis.CanReuse(a, b)) {
          stream << reuse_analysis.buffer_vars[b]->name_hint << " ";
        }
      }
      stream << "}";
      LOG(DEBUG) << stream.str();
    }
  }
  return aliases;
}

void MergeBufferAliasMaps(BufferAliasMap *destination,
                          const BufferAliasMap &source) {
  for (const auto &[storage, _] : source) {
    ICHECK(!destination->count(storage))
        << "Buffer storage appears in more than one allocation scope: "
        << storage;
    AddBufferAliasStorage(destination, storage);
  }
  for (const auto &[storage, compatible] : source) {
    for (const Var &other : compatible) {
      if (!HasBufferAlias(*destination, storage, other))
        AddBufferAlias(destination, storage, other);
    }
  }
}

class BufferAliasContractStripper : public StmtMutator {
private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == kBufferAliasMap)
      return VisitStmt(op->body);
    return StmtMutator::VisitStmt_(op);
  }
};

Stmt InferBufferAliases(Stmt stmt, bool verbose, bool disable_reuse) {
  stmt = BufferAliasContractStripper()(std::move(stmt));
  AllocateCollector collector;
  collector(stmt);
  BufferAliasMap aliases;
  auto analyze_scope = [&](const AllocationMap &allocations,
                           const AllocationOrder &allocation_order,
                           const std::string &target_scope) {
    if (allocations.empty())
      return;
    BufferAliasMap local =
        InferBufferAliasesForScope(stmt, allocations, allocation_order,
                                   target_scope, verbose, disable_reuse);
    MergeBufferAliasMaps(&aliases, local);
  };

  analyze_scope(collector.dyn_shmem_allocs, collector.dyn_shmem_order, "");
  std::vector<std::string> scopes;
  scopes.reserve(collector.ascend_scope_allocs.size());
  for (const auto &[scope, _] : collector.ascend_scope_allocs)
    scopes.push_back(scope);
  std::sort(scopes.begin(), scopes.end());
  for (const std::string &scope : scopes) {
    analyze_scope(collector.ascend_scope_allocs.at(scope),
                  collector.ascend_scope_order.at(scope), scope);
  }
  ValidateBufferAliasMap(aliases);
  return AttrStmt(std::move(aliases), kBufferAliasMap, Integer(1),
                  std::move(stmt));
}

} // namespace

using namespace tirx::transform;

namespace transform {

Pass InferBufferAliases() {
  auto pass_func = [](PrimFunc func, const IRModule &, PassContext context) {
    bool verbose =
        context
            ->GetConfig<Bool>(kDebugMergeSharedMemoryAllocations, Bool(false))
            .value();
    bool disable_reuse =
        context->GetConfig<Bool>(kDisableSharedMemoryReuse, Bool(false))
            .value();
    PrimFuncNode *writer = func.CopyOnWrite();
    writer->body =
        tl::InferBufferAliases(std::move(writer->body), verbose, disable_reuse);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InferBufferAliases", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.InferBufferAliases", InferBufferAliases);
}

} // namespace transform
} // namespace tl
} // namespace tvm
