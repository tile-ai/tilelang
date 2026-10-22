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

// Shared storage-granular access-order heuristic. This is eligibility analysis,
// not the must-write proof used by allocation reuse.
#pragma once

#include "./memory_detector.h"
#include "./multi_buffer.h"
#include "./scheduled_tir.h"

namespace tvm::tl::ascend {

// ---------------------------------------------------------------------------
// WriteFirstClassifier: program-order classification of a storage's access
// pattern inside a Stmt. All Buffer aliases sharing buffer->data participate.
//
// Four-state lattice:
//   kUntouched         — subtree provably never accesses the buffer
//   kMaybeWriteFirst   — either kUntouched or kWriteFirst (write not certain
//                        but no read-before-write risk)
//   kWriteFirst        — first access is a guaranteed write
//   kReadFirst         — a read may happen before any guaranteed write
//                        (absorbing/unsafe state)
//
// Both kUntouched and kMaybeWriteFirst and kWriteFirst are "safe" for
// multi-buffering (prior content is irrelevant). kReadFirst is not. This is
// deliberately a storage-granular heuristic: a write to any region is treated
// as making the storage write-first; it does not prove that every later-read
// region was overwritten. Kernels carrying untouched regions across iterations
// must opt out via T.annotate_buffer_versions({buf: 1}) or explicitly choose an
// owner whose epoch overwrites the complete read footprint.
// ---------------------------------------------------------------------------
enum class AccessOrder {
  kUntouched,
  kMaybeWriteFirst,
  kWriteFirst,
  kReadFirst,
};

// Terminal states determine the answer; further sub-events cannot change it.
inline bool IsTerminal(AccessOrder order) {
  return order == AccessOrder::kReadFirst || order == AccessOrder::kWriteFirst;
}

// Sequential composition of two sub-events in program order. Terminal states
// absorb the suffix; a conditional write followed by a read remains read-first
// because the write may have been skipped.
inline AccessOrder SeqCompose(AccessOrder first, AccessOrder second) {
  if (IsTerminal(first))
    return first;
  if (first == AccessOrder::kUntouched)
    return second;
  if (second == AccessOrder::kUntouched)
    return AccessOrder::kMaybeWriteFirst;
  return second;
}

// Merge mutually exclusive then/else branch classifications.
inline AccessOrder MergeIfThenElse(AccessOrder then_order,
                                   AccessOrder else_order) {
  if (then_order == AccessOrder::kReadFirst ||
      else_order == AccessOrder::kReadFirst) {
    return AccessOrder::kReadFirst;
  }
  if (then_order == AccessOrder::kUntouched &&
      else_order == AccessOrder::kUntouched) {
    return AccessOrder::kUntouched;
  }
  if (then_order == AccessOrder::kWriteFirst &&
      else_order == AccessOrder::kWriteFirst) {
    return AccessOrder::kWriteFirst;
  }
  return AccessOrder::kMaybeWriteFirst;
}

// Merge a single then branch with an implicit untouched else branch.
inline AccessOrder MergeIfThenOnly(AccessOrder then_order) {
  if (then_order == AccessOrder::kReadFirst)
    return AccessOrder::kReadFirst;
  if (then_order == AccessOrder::kUntouched)
    return AccessOrder::kUntouched;
  return AccessOrder::kMaybeWriteFirst;
}

// A possibly skipped scope cannot provide a guaranteed first write.
inline AccessOrder DemoteIfNotMustExecute(AccessOrder order,
                                          bool must_execute) {
  if (!must_execute && order == AccessOrder::kWriteFirst)
    return AccessOrder::kMaybeWriteFirst;
  return order;
}

inline AccessOrder AssumeForExecutes(AccessOrder body) {
  // NOTE: intentionally aggressive. Treat every For as executing at least
  // once, and assume a maybe-write-first body takes a writing path in some
  // iteration. A zero-trip or fully skipped loop may therefore make a later
  // read appear to have a preceding write.
  return body == AccessOrder::kMaybeWriteFirst ? AccessOrder::kWriteFirst
                                               : body;
}

class WriteFirstClassifier : public StmtExprVisitor {
public:
  explicit WriteFirstClassifier(bool physical_reads = false)
      : physical_reads_(physical_reads) {}

  AccessOrder Classify(const Stmt &s, const Var &storage) {
    target_storage_ = storage;
    result_ = AccessOrder::kUntouched;
    StmtExprVisitor::VisitStmt(s);
    return result_;
  }

  AccessOrder Classify(const PrimExpr &e, const Var &storage) {
    target_storage_ = storage;
    result_ = AccessOrder::kUntouched;
    StmtExprVisitor::VisitExpr(e);
    return result_;
  }

private:
  bool physical_reads_;
  Var target_storage_;
  AccessOrder result_ = AccessOrder::kUntouched;

  // Dispatch short-circuit: skip further work once a terminal is reached.
  void VisitStmt(const Stmt &s) final {
    if (IsTerminal(result_))
      return;
    StmtExprVisitor::VisitStmt(s);
  }
  void VisitExpr(const PrimExpr &e) final {
    if (IsTerminal(result_))
      return;
    StmtExprVisitor::VisitExpr(e);
  }

  template <typename F> AccessOrder ClassifyLocal(F f) {
    AccessOrder saved = result_;
    result_ = AccessOrder::kUntouched;
    f();
    AccessOrder local = result_;
    result_ = saved;
    return local;
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    for (const auto &idx : op->indices) {
      VisitExpr(idx);
      if (IsTerminal(result_))
        return;
    }
    if (target_storage_.same_as(op->buffer->data))
      result_ = AccessOrder::kReadFirst;
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    // value first, then indices, then the store itself.
    VisitExpr(op->value);
    if (IsTerminal(result_))
      return;
    for (const auto &idx : op->indices) {
      VisitExpr(idx);
      if (IsTerminal(result_))
        return;
    }
    if (target_storage_.same_as(op->buffer->data))
      result_ = AccessOrder::kWriteFirst;
  }

  void VisitStmt_(const SeqStmtNode *op) final {
    for (const auto &s : op->seq) {
      VisitStmt(s);
      if (IsTerminal(result_))
        return;
    }
  }

  void VisitStmt_(const EvaluateNode *op) final { VisitExpr(op->value); }

  void VisitStmt_(const ForNode *op) final {
    VisitExpr(op->min);
    if (IsTerminal(result_))
      return;
    VisitExpr(op->extent);
    if (IsTerminal(result_))
      return;
    if (op->step.has_value()) {
      VisitExpr(op->step.value());
      if (IsTerminal(result_))
        return;
    }
    auto body = ClassifyLocal([&] { VisitStmt(op->body); });
    result_ = SeqCompose(result_, AssumeForExecutes(body));
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    VisitExpr(op->condition);
    if (IsTerminal(result_))
      return;
    auto t = ClassifyLocal([&] { VisitStmt(op->then_case); });
    AccessOrder branch;
    if (op->else_case) {
      auto e = ClassifyLocal([&] { VisitStmt(op->else_case.value()); });
      branch = MergeIfThenElse(t, e);
    } else {
      branch = MergeIfThenOnly(t);
    }
    result_ = SeqCompose(result_, branch);
  }

  void VisitStmt_(const BindNode *op) final { VisitExpr(op->value); }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (CanRewriteMultiBufferAttrNode(op->attr_key, op->node)) {
      VisitExpr(Downcast<PrimExpr>(op->node));
      if (IsTerminal(result_))
        return;
    }
    VisitExpr(op->value);
    if (IsTerminal(result_))
      return;
    VisitStmt(op->body);
  }

  void VisitStmt_(const WhileNode *op) final {
    VisitExpr(op->condition);
    if (IsTerminal(result_))
      return;
    auto body = ClassifyLocal([&] { VisitStmt(op->body); });
    result_ = SeqCompose(result_,
                         DemoteIfNotMustExecute(body, /*must_execute=*/false));
  }

  void VisitStmt_(const SBlockNode *op) final {
    // Reduce-init runs only on the first iteration of the surrounding loop;
    // its writes are not guaranteed across iterations -> kMaybeWriteFirst.
    if (op->init.defined()) {
      auto init = ClassifyLocal([&] { VisitStmt(op->init.value()); });
      AccessOrder init_eff = init;
      if (init == AccessOrder::kWriteFirst)
        init_eff = AccessOrder::kMaybeWriteFirst;
      result_ = SeqCompose(result_, init_eff);
      if (IsTerminal(result_))
        return;
    }
    VisitStmt(op->body);
  }

  void VisitStmt_(const SBlockRealizeNode *op) final {
    VisitExpr(op->predicate);
    if (IsTerminal(result_))
      return;
    bool must_execute = false;
    if (const auto *imm = op->predicate.as<IntImmNode>())
      must_execute = (imm->value != 0);
    auto body = ClassifyLocal([&] { VisitStmt(op->block); });
    result_ = SeqCompose(result_, DemoteIfNotMustExecute(body, must_execute));
  }

  void VisitExpr_(const CallNode *op) final {
    if (physical_reads_) {
      // Some TileOps encode a destination as `w` even when they accumulate
      // onto it (for example reduce(clear=false)). Reuse physical access
      // accounting for the input side; this does not require dense writes.
      MemoryAccessDetector accesses(
          MemoryAccessDetector::AccessMode::kPhysical);
      accesses.Analyze(Evaluate(ffi::GetRef<Call>(op)));
      for (const BufferRegion &read : accesses.GetReadRegions()) {
        if (read->buffer->data.same_as(target_storage_)) {
          result_ = AccessOrder::kReadFirst;
          return;
        }
      }
    }
    static const Op &region_op = region();
    static const auto access_ptr_op = Op::Get("tl.access_ptr");

    if (op->op.same_as(region_op)) {
      HandleRegionCall(op);
      return;
    }
    if (op->op.same_as(access_ptr_op)) {
      HandleAccessPtrCall(op);
      return;
    }
    for (const auto &arg : op->args) {
      VisitExpr(arg);
      if (IsTerminal(result_))
        return;
    }
  }

  // tl.region(BufferLoad(buf, indices...), access_type, extents...)
  //   access_type: bit 0 = read, bit 1 = write.
  void HandleRegionCall(const CallNode *op) {
    if (op->args.size() < 2) {
      for (const auto &arg : op->args) {
        VisitExpr(arg);
        if (IsTerminal(result_))
          return;
      }
      return;
    }
    const auto *bl = op->args[0].as<BufferLoadNode>();
    const auto *mode = op->args[1].as<IntImmNode>();
    if (!bl || !mode) {
      for (const auto &arg : op->args) {
        VisitExpr(arg);
        if (IsTerminal(result_))
          return;
      }
      return;
    }
    for (const auto &idx : bl->indices) {
      VisitExpr(idx);
      if (IsTerminal(result_))
        return;
    }
    for (size_t i = 2; i < op->args.size(); ++i) {
      VisitExpr(op->args[i]);
      if (IsTerminal(result_))
        return;
    }
    if (target_storage_.same_as(bl->buffer->data)) {
      int m = mode->value;
      if (m & 1)
        result_ = AccessOrder::kReadFirst;
      else if (m & 2)
        result_ = AccessOrder::kWriteFirst;
    }
  }

  // tl.access_ptr(BufferLoad(buf, indices...), extent, rw_mask)
  //   rw_mask: bit 0 = read, bit 1 = write.
  void HandleAccessPtrCall(const CallNode *op) {
    if (op->args.size() < 3) {
      for (const auto &arg : op->args) {
        VisitExpr(arg);
        if (IsTerminal(result_))
          return;
      }
      return;
    }
    const auto *bl = op->args[0].as<BufferLoadNode>();
    const auto *mask = op->args[2].as<IntImmNode>();
    if (!bl || !mask) {
      for (const auto &arg : op->args) {
        VisitExpr(arg);
        if (IsTerminal(result_))
          return;
      }
      return;
    }
    for (const auto &idx : bl->indices) {
      VisitExpr(idx);
      if (IsTerminal(result_))
        return;
    }
    VisitExpr(op->args[1]);
    if (IsTerminal(result_))
      return;
    if (target_storage_.same_as(bl->buffer->data)) {
      int rw = mask->value;
      if (rw & 1)
        result_ = AccessOrder::kReadFirst;
      else if (rw & 2)
        result_ = AccessOrder::kWriteFirst;
    }
  }
};

inline int RequestedStage(const IRStructure *node) {
  int stage = node->GetStage();
  return stage == kUnscheduledStage ? 0 : stage;
}

inline std::vector<const IRStructure *>
GetWriteFirstOrder(const std::vector<std::shared_ptr<IRStructure>> &nodes) {
  std::vector<const IRStructure *> result;
  result.reserve(nodes.size());
  bool has_requested_stage = false;
  for (const auto &node : nodes) {
    result.push_back(node.get());
    has_requested_stage |= node->GetStage() != kUnscheduledStage;
  }
  if (has_requested_stage) {
    std::stable_sort(result.begin(), result.end(),
                     [](const IRStructure *lhs, const IRStructure *rhs) {
                       return RequestedStage(lhs) < RequestedStage(rhs);
                     });
  }
  return result;
}

// MaterializeScheduleUnits turns structured branches into guarded siblings.
// Track the condition under which an earlier guaranteed write has occurred,
// and prove that every later read executes only inside that condition. This
// recovers both equal guards and exhaustive opposite guards without rebuilding
// an IfThenElse tree.
class IRWriteFirstClassifier {
public:
  static AccessOrder
  Classify(const std::vector<std::shared_ptr<IRStructure>> &nodes,
           const Var &storage, const ConstrSet &outer_ctx,
           bool writes_only = false) {
    State state;
    for (const IRStructure *node : GetWriteFirstOrder(nodes)) {
      if (!writes_only || node->WritesStorage(storage))
        ProcessNode(node, storage, outer_ctx, &state, writes_only);
      if (state.unsafe)
        return AccessOrder::kReadFirst;
    }
    if (!state.touched)
      return AccessOrder::kUntouched;
    return IsCovered(Bool(true), state.written_guard, outer_ctx)
               ? AccessOrder::kWriteFirst
               : AccessOrder::kMaybeWriteFirst;
  }

private:
  struct State {
    bool touched{false};
    bool unsafe{false};
    PrimExpr written_guard{Bool(false)};
  };

  static bool IsCovered(const PrimExpr &access_guard,
                        const PrimExpr &written_guard,
                        const ConstrSet &outer_ctx) {
    return GuardsEquivalent(access_guard, written_guard, outer_ctx) ||
           GuardImplies(access_guard, written_guard, outer_ctx);
  }

  static PrimExpr MergeWriteGuard(const PrimExpr &known, const PrimExpr &added,
                                  const ConstrSet &outer_ctx) {
    if (is_zero(known))
      return added;
    if (is_zero(added) || GuardsEquivalent(known, added, outer_ctx) ||
        GuardImplies(added, known, outer_ctx)) {
      return known;
    }
    if (GuardImplies(known, added, outer_ctx))
      return added;
    return known || added;
  }

  static void ProcessAccess(AccessOrder order, const PrimExpr &active_guard,
                            const ConstrSet &outer_ctx, State *state) {
    ICHECK(state != nullptr);
    if (order == AccessOrder::kUntouched || state->unsafe)
      return;
    state->touched = true;
    if (order == AccessOrder::kReadFirst) {
      state->unsafe = !IsCovered(active_guard, state->written_guard, outer_ctx);
      return;
    }
    if (order == AccessOrder::kWriteFirst) {
      state->written_guard =
          MergeWriteGuard(state->written_guard, active_guard, outer_ctx);
    }
  }

  static AccessOrder ClassifyPayload(const IRStructure *node,
                                     const Var &storage, bool writes_only) {
    WriteFirstClassifier classifier(/*physical_reads=*/writes_only);
    if (node->IsTask()) {
      return classifier.Classify(static_cast<const TaskNode *>(node)->stmt,
                                 storage);
    }

    const auto *control = static_cast<const ControlNode *>(node);
    AccessOrder header = classifier.Classify(control->task->stmt, storage);
    if (header == AccessOrder::kReadFirst ||
        header == AccessOrder::kWriteFirst) {
      return header;
    }

    AccessOrder body = Classify(control->children, storage,
                                control->GetLoopBodyContext(), writes_only);
    return SeqCompose(header, AssumeForExecutes(body));
  }

  static void ProcessNode(const IRStructure *node, const Var &storage,
                          const ConstrSet &outer_ctx, State *state,
                          bool writes_only) {
    PrimExpr active_guard = Bool(true);
    WriteFirstClassifier classifier;
    for (const auto &guard : node->GetGuards()) {
      if (guard->IsCondition()) {
        const PrimExpr &condition =
            static_cast<const ConditionGuard *>(guard.get())->condition;
        ProcessAccess(classifier.Classify(condition, storage), active_guard,
                      outer_ctx, state);
        active_guard =
            is_one(active_guard) ? condition : active_guard && condition;
        continue;
      }

      const auto *attribute = static_cast<const AttributeGuard *>(guard.get());
      if (auto node_expr = attribute->node.try_cast<PrimExpr>()) {
        ProcessAccess(classifier.Classify(node_expr.value(), storage),
                      active_guard, outer_ctx, state);
      }
    }
    ProcessAccess(ClassifyPayload(node, storage, writes_only), active_guard,
                  outer_ctx, state);
  }
};

} // namespace tvm::tl::ascend
