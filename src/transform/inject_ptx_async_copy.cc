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
 * \brief Rewrite eligible global->shared copies into PTX cp.async
 * \file inject_ptx_async_copy.cc
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <vector>

#include "../op/builtin.h"
#include "../target/utils.h"
#include "tir/ir/buffer_common.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace tl {

using namespace tir;

class PTXAsyncCopyInjector : public StmtMutator {
public:
  explicit PTXAsyncCopyInjector(bool enable_auto_async_copy)
      : enable_auto_async_copy_(enable_auto_async_copy) {}

  Stmt Finalize(Stmt body) {
    if (pending_sync_copies_ <= 0) {
      return body;
    }
    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(body);
    if (uncommitted_sync_copies_ > 0) {
      seq.push_back(MakeCommitGroupStmt());
    }
    seq.push_back(MakeWaitGroupStmt(0));
    pending_sync_copies_ = 0;
    uncommitted_sync_copies_ = 0;
    return SeqStmt(seq);
  }

  Stmt VisitStmt_(const AttrStmtNode *attr) final {
    if (attr->attr_key == tir::attr::async_scope) {
      // async_scope is treated as an explicit request for async lowering; we
      // keep behavior compatible with historical usage by not auto-inserting
      // synchronous waits for copies injected under this scope.
      ICHECK(!in_async_scope_) << "Nested async scopes not supported";
      in_async_scope_ = true;
      Stmt body = this->VisitStmt(attr->body);
      in_async_scope_ = false;
      // Drop the marker after lowering.
      return body;
    }
    return StmtMutator::VisitStmt_(attr);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Rewrite a vectorized copy loop into a single cp.async.
    //
    // This pass runs before PipelinePlanning (and before the later global
    // VectorizeLoop), so user-written SIMT copies may still appear as scalar
    // float16 BufferStore statements inside a `ForKind::kVectorized` loop:
    //   for vec in T.vectorized(8):
    //     S[base + vec] = A[base + vec]
    //
    // We collapse such a loop into one `tir.ptx_cp_async` of 16 bytes so that
    // downstream pipeline passes can observe and schedule cp.async groups.
    if (op->kind == ForKind::kVectorized) {
      Optional<Stmt> injected = TryInjectVectorizedCopyLoop(op);
      if (injected.defined()) {
        return injected.value();
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  Optional<Stmt> TryInjectPTX(const BufferLoadNode *load,
                              const BufferStoreNode *store,
                              bool predicated = false,
                              const PrimExpr &predicate_value = PrimExpr()) {
    if (!IsGlobalLikeScope(load->buffer.scope())) {
      return Optional<Stmt>();
    }

    Optional<PrimExpr> src_index_opt =
        FlattenToLinearOffset(load->buffer, load->indices);
    Optional<PrimExpr> dst_index_opt =
        FlattenToLinearOffset(store->buffer, store->indices);
    if (!src_index_opt.defined() || !dst_index_opt.defined()) {
      return Optional<Stmt>();
    }
    PrimExpr src_index = src_index_opt.value();
    PrimExpr dst_index = dst_index_opt.value();

    if (src_index->dtype.lanes() != dst_index->dtype.lanes()) {
      // Not a straightforward vectorized copy; skip.
      return Optional<Stmt>();
    }

    const int index_lanes = src_index->dtype.lanes();
    const int value_lanes = load->dtype.lanes();
    if (value_lanes > 1 && index_lanes > 1 && value_lanes != index_lanes) {
      // Mismatched vector lane representations; be conservative.
      return Optional<Stmt>();
    }
    const int lanes = std::max(value_lanes, index_lanes);
    const int bytes = lanes * load->dtype.bytes();
    if (bytes != 4 && bytes != 8 && bytes != 16) {
      return Optional<Stmt>();
    }

    auto dst_elem_type = GetPointerType(store->buffer->data->type_annotation);
    auto src_elem_type = GetPointerType(load->buffer->data->type_annotation);
    if (!dst_elem_type.has_value() || !src_elem_type.has_value()) {
      // Be conservative: if pointer metadata is missing, skip injection.
      return Optional<Stmt>();
    }

    if (index_lanes == 1) {
      PrimExpr src_offset = src_index;
      PrimExpr dst_offset = dst_index;

      // Calculate the number of elements based on bytes and dtype
      int dst_elem_count = bytes / dst_elem_type->bytes();
      int src_elem_count = bytes / src_elem_type->bytes();

      // Create access_ptr for destination (shared memory, write access)
      PrimExpr dst_access_ptr = store->buffer.access_ptr(
          2, DataType::Handle(), 1, dst_offset, PrimExpr(dst_elem_count));

      // Create access_ptr for source (global memory, read access)
      PrimExpr src_access_ptr = load->buffer.access_ptr(
          1, DataType::Handle(), 1, src_offset, PrimExpr(src_elem_count));

      ffi::Array<PrimExpr> cp_async_args;
      if (predicated) {
        // Predicated cp.async with 4 arguments
        cp_async_args = {dst_access_ptr, src_access_ptr, PrimExpr(bytes),
                         predicate_value};
      } else {
        // Non-predicated cp.async with 3 arguments
        cp_async_args = {dst_access_ptr, src_access_ptr, PrimExpr(bytes)};
      }
      return Evaluate(Call(store->buffer->dtype,
                           tvm::tir::builtin::ptx_cp_async(), cp_async_args));
    }

    auto vector_base = [](const PrimExpr &e) -> PrimExpr {
      if (const auto *r = e.as<RampNode>()) {
        return r->base;
      }
      if (const auto *add = e.as<AddNode>()) {
        // Common pattern after flattening a vectorized N-D buffer access:
        //   (broadcast(base_offset) + ramp(vec_base, 1, lanes))
        // or its commuted form:
        //   (ramp(vec_base, 1, lanes) + broadcast(base_offset))
        const PrimExpr &a = add->a;
        const PrimExpr &b = add->b;
        if (const auto *ra = a.as<RampNode>()) {
          if (const auto *bb = b.as<BroadcastNode>()) {
            return tir::Add(ra->base, bb->value);
          }
        }
        if (const auto *rb = b.as<RampNode>()) {
          if (const auto *ba = a.as<BroadcastNode>()) {
            return tir::Add(rb->base, ba->value);
          }
        }
      }
      return PrimExpr();
    };

    if (lanes != index_lanes) {
      // Vector indices must cover the full transfer width.
      return Optional<Stmt>();
    }

    PrimExpr src_offset = vector_base(src_index);
    PrimExpr dst_offset = vector_base(dst_index);

    if (!src_offset.defined() || !dst_offset.defined()) {
      // If we can't extract offsets from vectorized indices, fall back.
      if (predicated) {
        LOG(WARNING)
            << "Cannot extract offsets from vectorized indices for predicated "
               "cp.async; falling back to regular buffer store/load";
      }
      return Optional<Stmt>();
    }

    // Calculate the number of elements based on bytes and dtype
    int dst_elem_count = bytes / dst_elem_type->bytes();
    int src_elem_count = bytes / src_elem_type->bytes();

    // Create access_ptr for destination (shared memory, write access)
    PrimExpr dst_access_ptr = store->buffer.access_ptr(
        2, DataType::Handle(), 1, dst_offset, PrimExpr(dst_elem_count));

    // Create access_ptr for source (global memory, read access)
    PrimExpr src_access_ptr = load->buffer.access_ptr(
        1, DataType::Handle(), 1, src_offset, PrimExpr(src_elem_count));

    ffi::Array<PrimExpr> cp_async_args;
    if (predicated) {
      cp_async_args = {dst_access_ptr, src_access_ptr, PrimExpr(bytes),
                       predicate_value};
    } else {
      cp_async_args = {dst_access_ptr, src_access_ptr, PrimExpr(bytes)};
    }
    return Evaluate(Call(store->buffer->dtype,
                         tvm::tir::builtin::ptx_cp_async(), cp_async_args));
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    // Insert commit+wait at statement boundaries to preserve synchronous
    // semantics for normal global->shared BufferStore copies.
    //
    // Important: avoid flushing inside inner loop bodies just because there
    // are trailing no-op statements (e.g., Evaluate(0)) after the injected
    // cp.async. Instead, treat "pure copy region" statements as part of the
    // copy run and only flush right before the next non-copy statement.
    Array<Stmt> out;
    out.reserve(op->seq.size() + 2);

    bool open_copy_region = pending_sync_copies_ > 0;
    bool uncommitted = uncommitted_sync_copies_ > 0;
    pending_sync_copies_ = 0;
    uncommitted_sync_copies_ = 0;

    for (const Stmt &stmt : op->seq) {
      pending_sync_copies_ = 0;
      uncommitted_sync_copies_ = 0;
      Stmt visited = this->VisitStmt(stmt);
      const AsyncIntrinSummary async_summary =
          SummarizeAsyncIntrinsics(visited);
      bool stmt_has_pending = pending_sync_copies_ > 0;
      bool stmt_has_uncommitted = uncommitted_sync_copies_ > 0;
      bool stmt_is_pure_copy_region = IsPureCopyRegion(visited);

      // Before we execute a non-copy statement, we must preserve synchronous
      // semantics for injected cp.async stores by making the data visible.
      if (open_copy_region && !stmt_is_pure_copy_region) {
        if (uncommitted) {
          out.push_back(MakeCommitGroupStmt());
        }
        out.push_back(MakeWaitGroupStmt(0));
        open_copy_region = false;
        uncommitted = false;
      }

      // If we are carrying uncommitted injected cp.async into an explicit wait,
      // ensure they are committed so the wait actually covers them.
      if (open_copy_region && uncommitted && async_summary.wait > 0) {
        out.push_back(MakeCommitGroupStmt());
        uncommitted = false;
      }

      out.push_back(visited);

      if (stmt_has_pending) {
        open_copy_region = true;
        uncommitted = uncommitted || stmt_has_uncommitted;
      }

      if (async_summary.commit > 0) {
        // A commit closes the currently open group, so there are no longer any
        // uncommitted injected cp.async transfers.
        uncommitted = false;
      }

      if (async_summary.wait > 0) {
        // Any explicit wait serves as a synchronization boundary for injected
        // synchronous copies.
        open_copy_region = false;
        uncommitted = false;
      }
    }

    pending_sync_copies_ = open_copy_region ? 1 : 0;
    uncommitted_sync_copies_ = uncommitted ? 1 : 0;

    if (out.empty()) {
      return Evaluate(0);
    }
    if (out.size() == 1) {
      return out[0];
    }
    return SeqStmt(out);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    // Treat branches as separate control flow paths. We propagate pending
    // synchronous copies into both branches (they occur before the branch),
    // but do not let mutations in one branch affect the other.
    int pending_before = pending_sync_copies_;
    int uncommitted_before = uncommitted_sync_copies_;

    pending_sync_copies_ = pending_before;
    uncommitted_sync_copies_ = uncommitted_before;
    Stmt then_case = this->VisitStmt(op->then_case);
    int pending_then = pending_sync_copies_;
    int uncommitted_then = uncommitted_sync_copies_;

    int pending_else = pending_before;
    int uncommitted_else = uncommitted_before;
    Optional<Stmt> else_case;
    if (op->else_case.defined()) {
      pending_sync_copies_ = pending_before;
      uncommitted_sync_copies_ = uncommitted_before;
      else_case = this->VisitStmt(op->else_case.value());
      pending_else = pending_sync_copies_;
      uncommitted_else = uncommitted_sync_copies_;
    }

    pending_sync_copies_ = std::max(pending_then, pending_else);
    uncommitted_sync_copies_ = std::max(uncommitted_then, uncommitted_else);

    if (then_case.same_as(op->then_case) &&
        (!else_case.defined() || else_case.same_as(op->else_case))) {
      return tvm::ffi::GetRef<Stmt>(op);
    }
    return IfThenElse(op->condition, then_case, else_case);
  }

  Stmt VisitStmt_(const BufferStoreNode *store) final {
    bool is_shared = (store->buffer.scope() == "shared" ||
                      store->buffer.scope() == "shared.dyn");
    if (!is_shared) {
      return StmtMutator::VisitStmt_(store);
    }

    // Only lower copies that are either explicitly marked async_scope, or are
    // in the automatic lowering mode controlled by pass config.
    if (!in_async_scope_ && !enable_auto_async_copy_) {
      return StmtMutator::VisitStmt_(store);
    }

    if (auto *load = store->value.as<BufferLoadNode>()) {
      Optional<Stmt> injected = TryInjectPTX(load, store);
      if (injected.defined()) {
        if (!in_async_scope_) {
          ++pending_sync_copies_;
          ++uncommitted_sync_copies_;
        }
        return injected.value();
      }
      return StmtMutator::VisitStmt_(store);
    }

    if (auto *call = store->value.as<CallNode>()) {
      // tir.if_then_else is a call to tir::builtin::if_then_else()
      if (call->op.same_as(builtin::if_then_else()) && call->args.size() == 3) {
        if (auto *load = call->args[1].as<BufferLoadNode>()) {
          // Only default value of 0 is supported since 0 is the default value
          // used by cp.async ptx.
          bool else_value_is_zero = IsZeroValue(call->args[2]);
          if (else_value_is_zero) {
            Optional<Stmt> injected =
                TryInjectPTX(load, store, /*predicated=*/true,
                             /*predicate_value=*/call->args[0]);
            if (injected.defined()) {
              if (!in_async_scope_) {
                ++pending_sync_copies_;
                ++uncommitted_sync_copies_;
              }
              return injected.value();
            }
          }
        }
      }
    }

    return StmtMutator::VisitStmt_(store);
  }

private:
  static bool IsGlobalLikeScope(const String &scope) {
    return scope == "global" || scope.empty();
  }

  static Optional<PrimExpr>
  FlattenToLinearOffset(const Buffer &buf,
                        const ffi::Array<PrimExpr> &indices) {
    // Convert N-D indices (potentially with axis_separators) into a single
    // row-major linear element offset.
    ffi::Array<PrimExpr> physical = buf.OffsetOf(indices);
    Buffer flattened_buf = buf.GetFlattenedBuffer();
    if (physical.size() != flattened_buf->shape.size() || physical.empty()) {
      return Optional<PrimExpr>();
    }

    PrimExpr linear = physical[0];
    for (size_t i = 1; i < physical.size(); ++i) {
      linear = linear * flattened_buf->shape[i] + physical[i];
    }
    return linear;
  }

  static Optional<PrimExpr> StripUnitStrideVar(const PrimExpr &expr,
                                               const Var &var) {
    // Return `base` such that expr == base + var (unit-stride contiguous).
    // Handles nested adds like (base0 + var) + base1.
    if (!UsesVar(expr, [&](const VarNode *n) { return n == var.get(); })) {
      return Optional<PrimExpr>();
    }
    if (expr.same_as(var)) {
      return make_zero(var.dtype());
    }
    if (const auto *add = expr.as<AddNode>()) {
      // Direct match: var + base or base + var.
      if (add->a.same_as(var) &&
          !UsesVar(add->b, [&](const VarNode *n) { return n == var.get(); })) {
        return add->b;
      }
      if (add->b.same_as(var) &&
          !UsesVar(add->a, [&](const VarNode *n) { return n == var.get(); })) {
        return add->a;
      }

      // Nested: (subexpr_with_var) + other_const
      if (!UsesVar(add->b, [&](const VarNode *n) { return n == var.get(); })) {
        Optional<PrimExpr> base = StripUnitStrideVar(add->a, var);
        if (base.defined()) {
          return base.value() + add->b;
        }
      }
      if (!UsesVar(add->a, [&](const VarNode *n) { return n == var.get(); })) {
        Optional<PrimExpr> base = StripUnitStrideVar(add->b, var);
        if (base.defined()) {
          return add->a + base.value();
        }
      }
    }
    return Optional<PrimExpr>();
  }

  Optional<Stmt> TryInjectVectorizedCopyLoop(const ForNode *loop) {
    // Only lower copies that are either explicitly marked async_scope, or are
    // in the automatic lowering mode controlled by pass config.
    if (!in_async_scope_ && !enable_auto_async_copy_) {
      return Optional<Stmt>();
    }

    const auto *extent_imm = loop->extent.as<IntImmNode>();
    if (!extent_imm) {
      return Optional<Stmt>();
    }
    if (!is_zero(loop->min)) {
      // Only handle canonical 0-based vectorized loops.
      return Optional<Stmt>();
    }
    int lanes = static_cast<int>(extent_imm->value);
    if (lanes <= 0) {
      return Optional<Stmt>();
    }

    // Only handle the simplest form:
    //   for vec in vectorized(L):
    //     S[...] = A[...]
    // or predicated zero-fill:
    //     S[...] = if_then_else(pred, A[...], 0)
    const auto *store = loop->body.as<BufferStoreNode>();
    if (!store) {
      return Optional<Stmt>();
    }

    bool is_shared = (store->buffer.scope() == "shared" ||
                      store->buffer.scope() == "shared.dyn");
    if (!is_shared) {
      return Optional<Stmt>();
    }

    const BufferLoadNode *load = store->value.as<BufferLoadNode>();
    bool predicated = false;
    PrimExpr predicate_value;
    if (!load) {
      const auto *call = store->value.as<CallNode>();
      if (call && call->op.same_as(builtin::if_then_else()) &&
          call->args.size() == 3) {
        load = call->args[1].as<BufferLoadNode>();
        if (load && IsZeroValue(call->args[2])) {
          predicated = true;
          predicate_value = call->args[0];
        }
      }
    }
    if (!load) {
      return Optional<Stmt>();
    }
    if (!IsGlobalLikeScope(load->buffer.scope())) {
      return Optional<Stmt>();
    }

    if (predicated && UsesVar(predicate_value, [&](const VarNode *n) {
          return n == loop->loop_var.get();
        })) {
      // cp.async predicate applies to the whole transaction; skip if predicate
      // varies per-lane.
      return Optional<Stmt>();
    }

    const int bytes = lanes * load->dtype.bytes();
    if (bytes != 4 && bytes != 8 && bytes != 16) {
      return Optional<Stmt>();
    }

    // Extract base element offsets for src/dst such that linear == base + vec.
    Optional<PrimExpr> dst_linear_opt =
        FlattenToLinearOffset(store->buffer, store->indices);
    Optional<PrimExpr> src_linear_opt =
        FlattenToLinearOffset(load->buffer, load->indices);
    if (!dst_linear_opt.defined() || !src_linear_opt.defined()) {
      return Optional<Stmt>();
    }

    Optional<PrimExpr> dst_base_opt =
        StripUnitStrideVar(dst_linear_opt.value(), loop->loop_var);
    Optional<PrimExpr> src_base_opt =
        StripUnitStrideVar(src_linear_opt.value(), loop->loop_var);
    if (!dst_base_opt.defined() || !src_base_opt.defined()) {
      return Optional<Stmt>();
    }
    PrimExpr dst_offset = dst_base_opt.value();
    PrimExpr src_offset = src_base_opt.value();

    auto dst_elem_type = GetPointerType(store->buffer->data->type_annotation);
    auto src_elem_type = GetPointerType(load->buffer->data->type_annotation);
    if (!dst_elem_type.has_value() || !src_elem_type.has_value()) {
      return Optional<Stmt>();
    }

    int dst_elem_count = bytes / dst_elem_type->bytes();
    int src_elem_count = bytes / src_elem_type->bytes();
    if (dst_elem_count <= 0 || src_elem_count <= 0) {
      return Optional<Stmt>();
    }

    PrimExpr dst_access_ptr = store->buffer.access_ptr(
        2, DataType::Handle(), 1, dst_offset, PrimExpr(dst_elem_count));
    PrimExpr src_access_ptr = load->buffer.access_ptr(
        1, DataType::Handle(), 1, src_offset, PrimExpr(src_elem_count));

    ffi::Array<PrimExpr> cp_async_args;
    if (predicated) {
      cp_async_args = {dst_access_ptr, src_access_ptr, PrimExpr(bytes),
                       predicate_value};
    } else {
      cp_async_args = {dst_access_ptr, src_access_ptr, PrimExpr(bytes)};
    }

    if (!in_async_scope_) {
      ++pending_sync_copies_;
      ++uncommitted_sync_copies_;
    }
    return Evaluate(
        Call(DataType::Handle(), builtin::ptx_cp_async(), cp_async_args));
  }

  struct AsyncIntrinSummary {
    int cp_async = 0;
    int commit = 0;
    int wait = 0;
  };

  static AsyncIntrinSummary SummarizeAsyncIntrinsics(const Stmt &stmt) {
    AsyncIntrinSummary summary;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      const auto *call = node.as<CallNode>();
      if (!call) {
        return;
      }
      if (call->op.same_as(builtin::ptx_cp_async()) ||
          call->op.same_as(tl::ptx_cp_async())) {
        ++summary.cp_async;
      } else if (call->op.same_as(builtin::ptx_commit_group())) {
        ++summary.commit;
      } else if (call->op.same_as(builtin::ptx_wait_group())) {
        ++summary.wait;
      }
    });
    return summary;
  }

  static bool IsZeroValue(const PrimExpr &e) {
    if (auto *b = e.as<BroadcastNode>()) {
      return IsZeroValue(b->value);
    }
    if (auto *f = e.as<FloatImmNode>()) {
      return f->value == 0.0f;
    }
    if (auto *i = e.as<IntImmNode>()) {
      return i->value == 0;
    }
    return false;
  }

  static Stmt MakeCommitGroupStmt() {
    return Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
  }

  static Stmt MakeWaitGroupStmt(int n) {
    return Evaluate(Call(DataType::Handle(), builtin::ptx_wait_group(),
                         {IntImm(DataType::Int(32), n)}));
  }

  bool IsPureCopyRegion(const Stmt &stmt) const {
    if (!stmt.defined()) {
      return true;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        if (!IsPureCopyRegion(s)) {
          return false;
        }
      }
      return true;
    }
    if (const auto *ite = stmt.as<IfThenElseNode>()) {
      if (!IsPureCopyRegion(ite->then_case)) {
        return false;
      }
      if (ite->else_case.defined() &&
          !IsPureCopyRegion(ite->else_case.value())) {
        return false;
      }
      return true;
    }
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (is_const_int(eval->value)) {
        return true;
      }
      const auto *call = eval->value.as<CallNode>();
      if (!call) {
        return false;
      }
      return call->op.same_as(builtin::ptx_cp_async()) ||
             call->op.same_as(tl::ptx_cp_async()) ||
             call->op.same_as(builtin::ptx_commit_group()) ||
             call->op.same_as(builtin::ptx_wait_group());
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      return IsPureCopyRegion(let->body);
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return IsPureCopyRegion(attr->body);
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      return IsPureCopyRegion(loop->body);
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      if (block->init.defined() && !IsPureCopyRegion(block->init.value())) {
        return false;
      }
      return IsPureCopyRegion(block->body);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      // Treat the predicate as pure control flow (no side effects). We only
      // care whether the realized body is a pure copy region so we can hoist
      // the final commit+wait out of sequential loop nests.
      const BlockNode *block = realize->block.get();
      if (block->init.defined() && !IsPureCopyRegion(block->init.value())) {
        return false;
      }
      return IsPureCopyRegion(block->body);
    }
    return false;
  }

  bool enable_auto_async_copy_{true};
  bool in_async_scope_{false};
  int pending_sync_copies_{0};
  int uncommitted_sync_copies_{0};
};

using namespace tir::transform;

tvm::transform::Pass InjectPTXAsyncCopy() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto target_opt = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target_opt.defined()) {
      return f;
    }
    Target target = target_opt.value();
    if (target->kind->name != "cuda") {
      return f;
    }
    if (tl::TargetIsCuTeDSL(target)) {
      return f;
    }

    bool enable_auto_async_copy =
        ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(true)).value();
    if (!TargetHasAsyncCopy(target)) {
      // Graceful fallback on older architectures.
      return f;
    }

    auto *n = f.CopyOnWrite();
    PTXAsyncCopyInjector injector(enable_auto_async_copy);
    n->body = injector.Finalize(injector(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectPTXAsyncCopy", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectPTXAsyncCopy", InjectPTXAsyncCopy);
}

} // namespace tl
} // namespace tvm
