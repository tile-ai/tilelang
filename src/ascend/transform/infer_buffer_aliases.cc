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
 * \brief Infer native Ascend reuse from per-core/per-pipe completion edges.
 *
 * Fixed native loops are expanded with exact Var substitutions under a bounded
 * analysis budget. Unknown control or wrapping induction never invents a loop
 * count, a branch outcome, or an event-id value: such a scope retains distinct
 * allocations. No memory reuse decision depends on a variable's printed name.
 */
#include "ascend/ir/core_mask.h"
#include "ascend/ir/task.h"
#include "ascend/transform/source_scopes.h"
#include "ascend_pipe.h"
#include "auto_schedule/multi_buffer.h"
#include "buffer_alias.h"
#include "ir/annotations.h"
#include "ir/loop_math.h"
#include "layout/expr_utils.h"
#include "merge_ub_common.h"
#include "op/builtin.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <sstream>
#include <tuple>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {
namespace {
using namespace tirx;
using namespace ffi;
using namespace prim;
using namespace ascend;
using namespace merge_ub;

enum class EventKind : uint8_t {
  kBufferAccess,
  kSetFlag,
  kWaitFlag,
  kPipeBarrier,
  kCrossCoreSet,
  kCrossCoreWait
};
struct LinearEvent {
  EventKind kind{EventKind::kBufferAccess};
  std::vector<const VarObj *> touched;
  std::vector<bool> is_write;
  std::string hard_event;
  PrimExpr event_id_expr{IntImm::Int32(0)};
  int mode_id{0};
  CoreMask core_mask{kCoreUnassigned};
  ResourcePipe exec_pipe{ResourcePipe::kUnknown};
  std::vector<int> loop_ids;
};
struct EventInstance : LinearEvent {
  int node_id{0}, linear_idx{0}, unrolled_iter{0}, physical_iter{0};
};

class SyncAwareLinearizer : public StmtExprVisitor {
public:
  using Result = Optional<VisitInterrupt>;
  using StmtExprVisitor::Visit_;
  SyncAwareLinearizer(const Stmt &stmt, AllocationMap allocations)
      : allocations_(std::move(allocations)) {
    storage_.Collect(stmt);
  }
  std::vector<LinearEvent> linear_events;
  std::unordered_set<std::string> unknown_instr_ops;
  bool uncertain_control{false};

private:
  static constexpr size_t kMaxEvents = 8000;
  AllocationMap allocations_;
  StorageProvenance storage_ = MakeAscendStorageProvenance();
  sym::Analyzer analyzer_;
  Map<Var, PrimExpr> values_;
  CoreMask core_{kCoreUnassigned};
  ResourcePipe pipe_{ResourcePipe::kUnknown};
  std::vector<int> loops_;
  int next_loop_{0};

  PrimExpr Evaluate(const PrimExpr &expr) {
    return analyzer_->Simplify(tirx::SubstituteWithDataTypeLegalization(
        expr, [&](const Var &var) -> ffi::Optional<PrimExpr> {
          return values_.Get(var);
        }));
  }
  void Emit(LinearEvent event) {
    if (linear_events.size() == kMaxEvents) {
      uncertain_control = true;
      return;
    }
    event.core_mask = core_;
    event.loop_ids = loops_;
    if (event.exec_pipe == ResourcePipe::kUnknown)
      event.exec_pipe = pipe_;
    linear_events.push_back(std::move(event));
  }
  void Touch(const Expr &value, bool write) {
    auto address = storage_.Address(value);
    if (address.exact) {
      auto variable = address.root.as<Var>();
      if (variable && allocations_.count(variable->get())) {
        LinearEvent event;
        event.touched = {variable->get()};
        event.is_write = {write};
        Emit(event);
      }
      return;
    }
    // An opaque pointer can still have an allocation operand. Do not erase
    // that lifetime merely because address arithmetic is not representable.
    std::unordered_set<const VarObj *> seen;
    std::function<void(const Expr &)> scan = [&](const Expr &expression) {
      for (const auto &var : UndefinedVars(expression)) {
        if (!seen.insert(var.get()).second)
          continue;
        if (allocations_.count(var.get())) {
          LinearEvent event;
          event.touched = {var.get()};
          event.is_write = {write};
          event.exec_pipe = ResourcePipe::kUnknown;
          Emit(event);
          uncertain_control = true;
        } else if (auto binding = storage_.bindings.Get(var))
          scan(*binding);
      }
    };
    scan(value);
  }
  Result Visit_(const SeqStmtObj *op) final {
    for (const auto &stmt : op->seq) {
      if (linear_events.size() == kMaxEvents) {
        uncertain_control = true;
        break;
      }
      Visit(stmt);
    }
    return std::nullopt;
  }
  Result Visit_(const BindObj *op) final {
    Expr value = AllocationValue(op);
    const auto *call = value.as<CallObj>();
    if (call && (call->op.same_as(alloc_tensor_op()) ||
                 call->op.same_as(decl_tensor_op()))) {
      Visit(op->var->ty);
      if (call->op.same_as(decl_tensor_op()))
        AddressOperands(call->args[0]);
      return std::nullopt;
    }
    Visit(value);
    if (auto scalar = value.as<PrimExpr>();
        scalar && op->var->ty.as<PrimTypeObj>())
      values_.Set(op->var, Evaluate(*scalar));
    return std::nullopt;
  }
  Result Visit_(const VarObj *op) final {
    if (op->ty.as<TensorTypeObj>() || op->ty.as<PtrTypeObj>())
      Touch(GetRef<Var>(op), false);
    return std::nullopt;
  }
  Result Visit_(const TensorLoadObj *op) final {
    for (const auto &index : op->indices)
      Visit(index);
    Visit(op->source->ty);
    Touch(op->source, false);
    return std::nullopt;
  }
  Result Visit_(const TensorStoreObj *op) final {
    Visit(op->value);
    for (const auto &index : op->indices)
      Visit(index);
    Visit(op->dest->ty);
    Touch(op->dest, true);
    return std::nullopt;
  }
  Result Visit_(const ForObj *op) final {
    Visit(op->begin);
    Visit(op->end);
    Visit(op->step);
    auto before = values_;
    For loop = GetRef<For>(op);
    auto *writer = loop.CopyOnWrite();
    writer->begin = Evaluate(op->begin);
    writer->end = Evaluate(op->end);
    writer->step = Evaluate(op->step);
    int count = ConstantUnrollTripCount(loop.get(), analyzer_);
    if (count < 0 || count > static_cast<int>(kMaxEvents)) {
      uncertain_control = true;
      Visit(op->body);
      values_ = before;
      return std::nullopt;
    }
    if (count > 1)
      loops_.push_back(next_loop_++);
    for (int iteration = 0; iteration < count; ++iteration) {
      if (linear_events.size() == kMaxEvents) {
        uncertain_control = true;
        break;
      }
      values_ = before;
      values_.Set(op->var, UnrolledLoopValue(loop.get(), iteration, analyzer_));
      Visit(op->body);
    }
    if (count > 1)
      loops_.pop_back();
    values_ = before;
    return std::nullopt;
  }
  Result Visit_(const IfObj *op) final {
    Visit(op->condition);
    auto condition = TryConstInt(Evaluate(op->condition));
    auto before = values_;
    if (!condition || *condition)
      Visit(op->then_case);
    values_ = before;
    if ((!condition || !*condition) && op->else_case)
      Visit(*op->else_case);
    values_ = before;
    if (!condition)
      uncertain_control = true;
    return std::nullopt;
  }
  Result Visit_(const WhileObj *op) final {
    uncertain_control = true;
    Visit(op->condition);
    Visit(op->body);
    return std::nullopt;
  }
  Result Visit_(const BreakObj *) final {
    uncertain_control = true;
    return std::nullopt;
  }
  Result Visit_(const ContinueObj *) final {
    uncertain_control = true;
    return std::nullopt;
  }
  Result Visit_(const ReturnObj *op) final {
    uncertain_control = true;
    return StmtExprVisitor::Visit_(op);
  }
  Result Visit_(const RegionStmtObj *op) final {
    auto old_core = core_;
    auto old_pipe = pipe_;
    auto old_values = values_;
    if (IsAscendBlock(op)) {
      auto name = op->attrs.GetAttr<ffi::String>("name");
      if (*name == "VECTOR")
        core_ = kCoreVector;
      if (*name == "CUBE")
        core_ = kCoreCube;
      auto block = GetAscendBlockPipe(*name);
      if (block != ResourcePipe::kUnknown)
        pipe_ = block;
    }
    if (IsAscendBlock(op) || IsSemanticTask(op) || IsScheduleUnit(op) ||
        IsStageScope(op)) {
      // Annotation operands describe compiler metadata, not executed accesses.
      Visit(op->body);
    } else if (auto attribute = ReadSourceAttribute(op)) {
      if (CanRewriteMultiBufferAttrNode(attribute->key, attribute->node))
        Visit(attribute->node.cast<Expr>());
      Visit(op->body);
    } else {
      for (const auto &argument : op->args)
        Visit(argument);
      Visit(op->body);
    }
    core_ = old_core;
    pipe_ = old_pipe;
    values_ = old_values;
    return std::nullopt;
  }
  void AddressOperands(const Expr &value) {
    if (value.as<VarObj>())
      return;
    if (const auto *call = value.as<CallObj>()) {
      if (call->op.same_as(tensor_data_ptr_op())) {
        Visit(call->args[0]->ty);
        return;
      }
      if (call->op.same_as(reinterpret_op())) {
        AddressOperands(call->args[0]);
        return;
      }
      if (call->op.same_as(ptr_byte_offset_op())) {
        AddressOperands(call->args[0]);
        Visit(call->args[1]);
        return;
      }
    }
    Visit(value);
  }
  bool RecordSync(const CallObj *call) {
    LinearEvent event;
    auto args = DecodeCallOperands(call);
    if (call->op.same_as(ascend_set_flag()) ||
        call->op.same_as(ascend_wait_flag())) {
      TVM_FFI_CHECK_GE(args.size(), 2U, ValueError);
      event.kind = call->op.same_as(ascend_set_flag()) ? EventKind::kSetFlag
                                                       : EventKind::kWaitFlag;
      event.hard_event = args[0].as_or_throw<StringImm>()->value;
      event.exec_pipe = GetFlagEndpointPipe(event.hard_event,
                                            event.kind == EventKind::kSetFlag
                                                ? FlagEndpoint::kSet
                                                : FlagEndpoint::kWait);
      event.event_id_expr = Evaluate(args[1].as_or_throw<PrimExpr>());
    } else if (call->op.same_as(ascend_pipe_barrier())) {
      TVM_FFI_CHECK_GE(args.size(), 1U, ValueError);
      event.kind = EventKind::kPipeBarrier;
      event.hard_event = args[0].as_or_throw<StringImm>()->value;
      event.exec_pipe = ParseAscendPipe(event.hard_event);
    } else if (call->op.same_as(ascend_cross_core_set_flag()) ||
               call->op.same_as(ascend_cross_core_wait_flag())) {
      TVM_FFI_CHECK_GE(args.size(), 3U, ValueError);
      event.kind = call->op.same_as(ascend_cross_core_set_flag())
                       ? EventKind::kCrossCoreSet
                       : EventKind::kCrossCoreWait;
      auto mode = TryConstInt(Evaluate(args[0].as_or_throw<PrimExpr>()));
      TVM_FFI_CHECK(mode && *mode >= 0 &&
                        *mode <= std::numeric_limits<int>::max(),
                    ValueError);
      event.mode_id = static_cast<int>(*mode);
      event.hard_event = args[1].as_or_throw<StringImm>()->value;
      event.exec_pipe = ParseAscendPipe(event.hard_event);
      event.event_id_expr = Evaluate(args[2].as_or_throw<PrimExpr>());
    } else
      return false;
    Emit(event);
    return true;
  }
  Result Visit_(const CallObj *op) final {
    if (IsAnnotationCall(op))
      return std::nullopt;
    if (IsScheduledBind(op))
      return Visit(ReadScheduledBinding(op).value);
    if (RecordSync(op))
      return std::nullopt;
    if (op->op.same_as(tl::access_ptr())) {
      TVM_FFI_CHECK_EQ(op->args.size(), 3U, ValueError);
      auto load = op->args[0].as_or_throw<TensorLoad>();
      for (const auto &index : load->indices)
        Visit(index);
      Visit(op->args[1]);
      Visit(op->args[2]);
      int mask = GetConservativeAccessMask(
          Evaluate(op->args[2].as_or_throw<PrimExpr>()));
      Touch(load->source, (mask & kAccessWrite) != 0);
      return std::nullopt;
    }
    if (op->op.same_as(tl::ptr_access())) {
      TVM_FFI_CHECK_EQ(op->args.size(), 4U, ValueError);
      AddressOperands(op->args[0]);
      for (size_t i = 1; i < op->args.size(); ++i)
        Visit(op->args[i]);
      int mask = GetConservativeAccessMask(
          Evaluate(op->args[3].as_or_throw<PrimExpr>()));
      Touch(op->args[0], (mask & kAccessWrite) != 0);
      return std::nullopt;
    }
    Call call = GetRef<Call>(op);
    auto old = pipe_;
    auto arguments = DecodeCallOperands(op);
    for (size_t index = 0; index < arguments.size(); ++index) {
      auto pipe = GetAscendCallArgumentPipe(call, index);
      if (pipe != ResourcePipe::kUnknown)
        pipe_ = pipe;
      else if (IsAscendPipeRelevantCall(call)) {
        pipe_ = ResourcePipe::kUnknown;
        unknown_instr_ops.insert(call->op.as_or_throw<Op>()->name);
      }
      Visit(arguments[index]);
      pipe_ = old;
    }
    return std::nullopt;
  }
};

struct HappensBeforeGraph {
  std::vector<EventInstance> instances;
  std::vector<std::vector<bool>> reachable; // [i][j]
  int N = 0;

  void Build(const std::vector<LinearEvent> &events) {
    instances.clear();
    for (const auto &event : events) {
      EventInstance instance;
      instance.node_id = static_cast<int>(instances.size());
      instance.linear_idx = instance.node_id;
      instance.unrolled_iter = 0;
      instance.physical_iter = 0;
      instance.kind = event.kind;
      instance.touched = event.touched;
      instance.is_write = event.is_write;
      instance.hard_event = event.hard_event;
      instance.event_id_expr = event.event_id_expr;
      instance.mode_id = event.mode_id;
      instance.core_mask = event.core_mask;
      instance.exec_pipe = event.exec_pipe;
      instance.loop_ids = event.loop_ids;
      instances.push_back(std::move(instance));
    }
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
    ComputeTransitiveClosure();
  }

  // Collect *every* node id that touches `var` — no per-iteration first/last
  // collapsing. Used by buffer reuse analysis for the strict "all access
  // pairs" ordering rule.
  void GetAllBufferAccessNodes(const VarObj *var, std::vector<int> &ids) const {
    for (int i = 0; i < N; ++i) {
      const EventInstance &inst = instances[i];
      if (inst.kind != EventKind::kBufferAccess)
        continue;
      for (const VarObj *v : inst.touched) {
        if (v == var) {
          ids.push_back(i);
          break;
        }
      }
    }
  }

  // For each buffer, collect the node ids of its first and last accesses.
  void GetBufferAccessNodes(const VarObj *var, std::vector<int> &first_ids,
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
      for (const VarObj *v : inst.touched) {
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
      CoreMask core;
      bool operator==(const SyncKey &o) const {
        return hard_event == o.hard_event && mode_id == o.mode_id &&
               is_cross_core == o.is_cross_core &&
               event_id_val == o.event_id_val && core == o.core;
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
        h ^= std::hash<int>()(k.core) + 0x9e3779b9 + (h << 6) + (h >> 2);
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
      sym::Analyzer analyzer;
      auto value = TryConstInt(analyzer->Simplify(inst.event_id_expr));
      if (!value)
        return;
      SyncKey key{hard_event, mode_id, is_cross, *value,
                  is_cross ? kCoreUnassigned : inst.core_mask};
      buckets[key].push_back(Endpoint{idx, is_set});
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
};

// Pairwise buffer lifetime ordering.

class BufferReuseAnalysis {
public:
  int num_buffers = 0;
  std::vector<const VarObj *> buffer_vars; // index → VarObj*
  // ordered[A][B] means every access to A happens before every access to B.
  std::vector<std::vector<bool>> ordered;

  bool CanReuse(int lhs, int rhs) const {
    // A cycle is not a lifetime separation: the old SCC-based allocator kept
    // both buffers in one non-reusable component in this case.
    return ordered[lhs][rhs] != ordered[rhs][lhs];
  }

  void
  Build(const HappensBeforeGraph &hb,
        const std::unordered_map<const VarObj *, const BindObj *> &shmem_allocs,
        const std::vector<const VarObj *> &alloc_order) {
    for (const VarObj *v : alloc_order) {
      if (shmem_allocs.count(v))
        buffer_vars.push_back(v);
    }
    if (buffer_vars.size() < shmem_allocs.size()) {
      std::unordered_set<const VarObj *> seen(buffer_vars.begin(),
                                              buffer_vars.end());
      std::vector<const VarObj *> tail;
      for (const auto &kv : shmem_allocs)
        if (!seen.count(kv.first))
          tail.push_back(kv.first);
      std::sort(tail.begin(), tail.end(), [](const VarObj *a, const VarObj *b) {
        if (a->name != b->name)
          return a->name < b->name;
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
  auto linearizer = make_object<SyncAwareLinearizer>(stmt, allocations);
  linearizer->Visit(stmt);
  if (linearizer->uncertain_control) {
    BufferAliasMap aliases;
    for (const auto &[var, unused] : allocations)
      AddBufferAliasStorage(&aliases, GetRef<Var>(var));
    return aliases;
  }
  if (verbose && !linearizer->unknown_instr_ops.empty()) {
    std::vector<std::string> ops(linearizer->unknown_instr_ops.begin(),
                                 linearizer->unknown_instr_ops.end());
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
  std::vector<LinearEvent> events = linearizer->linear_events;
  HappensBeforeGraph happens_before;
  happens_before.Build(events);

  BufferReuseAnalysis reuse_analysis;
  reuse_analysis.Build(happens_before, allocations, allocation_order);

  BufferAliasMap aliases;
  for (const VarObj *var : reuse_analysis.buffer_vars)
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
      stream << "    " << reuse_analysis.buffer_vars[a]->name << " -> {";
      for (int b = 0; b < reuse_analysis.num_buffers; ++b) {
        if (a != b && reuse_analysis.CanReuse(a, b)) {
          stream << reuse_analysis.buffer_vars[b]->name << " ";
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

class BufferAliasContractStripper : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    SeqStmt sequence = StmtExprMutator::Mutate_(op, mode)
                           .ValueOrUnchanged(GetRef<SeqStmt>(op))
                           .as_or_throw<SeqStmt>();
    Array<Stmt> statements;
    for (const Stmt &statement : sequence->seq) {
      const auto *empty = statement.as<SeqStmtObj>();
      if (!empty || !empty->seq.empty())
        statements.push_back(statement);
    }
    return statements.size() == sequence->seq.size()
               ? Stmt(sequence)
               : Stmt(SeqStmt(statements, sequence->loc));
  }
  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    const auto *call = op->value.as<CallObj>();
    if (IsAnnotationCall(call)) {
      auto fields = ReadAnnotations(call);
      if (fields.count(kBufferAliasMap)) {
        fields.erase(kBufferAliasMap);
        return fields.empty() ? Stmt(SeqStmt({}))
                              : MakeAnnotation(fields, op->loc);
      }
    }
    return StmtExprMutator::Mutate_(op, mode);
  }
  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    if (auto attr = ReadSourceAttribute(op)) {
      if (attr->key == kBufferAliasMap)
        return Mutate(op->body, mode).ValueOrUnchanged(op->body);
      if (attr->key == "tl.loop.annotations") {
        auto fields = attr->value.cast<Map<String, Any>>();
        auto updated = fields;
        updated.erase(kBufferAliasMap);
        if (!StructuralEqual()(fields, updated)) {
          Stmt next =
              MakeAttribute(attr->node, attr->key, updated, op->body, op->loc);
          return Mutate(next, mode).ValueOrUnchanged(next);
        }
      }
    }
    return StmtExprMutator::Mutate_(op, mode);
  }
};

// A metadata map cannot mention a definition outside its native lexical scope.
// Emit disjoint per-sequence maps after their local allocations. Cross-scope
// pairs keep distinct storage; this is a missing optimization, not permission
// to hoist dynamic operands into an enclosing RegionStmt header.
class BufferAliasContractEmitter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  explicit BufferAliasContractEmitter(BufferAliasMap aliases)
      : aliases_(std::move(aliases)) {}
  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    SeqStmt sequence = StmtExprMutator::Mutate_(op, mode)
                           .ValueOrUnchanged(GetRef<SeqStmt>(op))
                           .as_or_throw<SeqStmt>();
    std::vector<Var> local;
    for (const auto &stmt : sequence->seq) {
      const auto *bind = stmt.as<BindObj>();
      if (bind && aliases_.count(bind->var))
        local.push_back(bind->var);
    }
    if (local.empty())
      return sequence;
    BufferAliasMap contract;
    for (const Var &var : local)
      AddBufferAliasStorage(&contract, var);
    for (size_t a = 0; a < local.size(); ++a)
      for (size_t b = a + 1; b < local.size(); ++b)
        if (HasBufferAlias(aliases_, local[a], local[b]))
          AddBufferAlias(&contract, local[a], local[b]);
    auto body = sequence->seq;
    body.push_back(
        MakeAnnotation({{kBufferAliasMap, contract}}, sequence->loc));
    return SeqStmt(body, sequence->loc);
  }

private:
  BufferAliasMap aliases_;
};

Stmt InferBufferAliases(Stmt stmt, bool verbose, bool disable_reuse) {
  auto stripper = make_object<BufferAliasContractStripper>();
  stmt = stripper->Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(stmt);
  AllocateCollector collector;
  collector(stmt);
  BufferAliasMap aliases;
  auto analyze = [&](const AllocationMap &allocations,
                     const AllocationOrder &order, const std::string &scope) {
    if (!allocations.empty())
      MergeBufferAliasMaps(
          &aliases, InferBufferAliasesForScope(stmt, allocations, order, scope,
                                               verbose, disable_reuse));
  };
  analyze(collector.dyn_shmem_allocs, collector.dyn_shmem_order, "");
  std::vector<std::string> scopes;
  for (const auto &[scope, unused] : collector.ascend_scope_allocs)
    scopes.push_back(scope);
  std::sort(scopes.begin(), scopes.end());
  for (const auto &scope : scopes)
    analyze(collector.ascend_scope_allocs.at(scope),
            collector.ascend_scope_order.at(scope), scope);
  ValidateBufferAliasMap(aliases);
  auto emitter = make_object<BufferAliasContractEmitter>(aliases);
  return emitter->Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(stmt);
}
} // namespace
namespace transform {
tvm::transform::Pass InferBufferAliases() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext context) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    bool verbose = context->GetConfig<bool>(kDebugMergeSharedMemoryAllocations)
                       .value_or(false);
    bool disabled =
        context->GetConfig<bool>(kDisableSharedMemoryReuse).value_or(false);
    auto body = function->body;
    function.CopyOnWrite()->body =
        SeqStmt(tl::InferBufferAliases(body, verbose, disabled));
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.InferBufferAliases");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.InferBufferAliases",
                                   InferBufferAliases);
}
} // namespace transform
} // namespace tvm::tl
