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
 * \file storage_access.cc
 */
#include "storage_access.h"

#include <tvm/arith/analyzer.h>
#include <tvm/target/target_info.h>
#include <tvm/tir/op.h>

#include <string>
#include <utility>

#include "../op/builtin.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;

void TileLangStorageAccessVisitor::VisitExpr_(const BufferLoadNode *op) {
  Var buf = op->buffer->data;
  buffer_data_to_buffer_.Set(tvm::ffi::GetRef<Var>(buf.get()), op->buffer);
  StorageScope scope = GetScope(buf);
  if (Enabled(buf.get(), scope)) {
    ICHECK(allow_append_) << tvm::ffi::GetRef<BufferLoad>(op) << " "
                          << scope.to_string();
    AccessEntry e;
    e.threads = env_threads();
    e.thread_range = this->ComputeThreadRange(e.threads);
    e.buffer = buf;
    e.buffer_indices = op->indices;
    e.dtype = op->dtype.element_of();
    for (const auto &index : op->indices) {
      e.touched.push_back(arith::IntSet::Vector(index));
    }
    e.type = kRead;
    e.scope = scope;
    curr_stmt_.access.emplace_back(std::move(e));
  }
  // traverse child
  IRVisitorWithAnalyzer::VisitExpr_(op);
}

void TileLangStorageAccessVisitor::VisitStmt_(const BufferStoreNode *op) {
  allow_append_ = true;
  ICHECK_EQ(curr_stmt_.access.size(), 0U);
  curr_stmt_.stmt = op;

  Var buf = op->buffer->data;
  buffer_data_to_buffer_.Set(tvm::ffi::GetRef<Var>(buf.get()), op->buffer);
  StorageScope scope = GetScope(buf);
  if (Enabled(buf.get(), scope)) {
    AccessEntry e;
    e.threads = env_threads();
    e.thread_range = this->ComputeThreadRange(e.threads);
    e.buffer = buf;
    e.buffer_indices = op->indices;
    e.dtype = op->value.dtype().element_of();
    for (const auto &index : op->indices) {
      e.touched.push_back(arith::IntSet::Vector(index));
    }
    e.type = kWrite;
    e.scope = scope;
    curr_stmt_.access.emplace_back(std::move(e));
  }
  // traverse child
  IRVisitorWithAnalyzer::VisitStmt_(op);
  // push to the scope
  scope_.back().push_back(curr_stmt_);
  // clear access entry.
  curr_stmt_.access.clear();
  allow_append_ = false;
}

void TileLangStorageAccessVisitor::VisitStmt_(const EvaluateNode *op) {
  allow_append_ = true;
  ICHECK_EQ(curr_stmt_.access.size(), 0U);
  curr_stmt_.stmt = op;
  IRVisitorWithAnalyzer::VisitStmt_(op);
  // push to the scope
  if (!curr_stmt_.access.empty()) {
    scope_.back().push_back(curr_stmt_);
    curr_stmt_.access.clear();
  }
  allow_append_ = false;
}

void TileLangStorageAccessVisitor::VisitStmt_(const LetStmtNode *op) {
  allow_append_ = true;
  ICHECK_EQ(curr_stmt_.access.size(), 0U);
  curr_stmt_.stmt = op;
  this->VisitExpr(op->value);
  // push to the scope
  scope_.back().push_back(curr_stmt_);
  // clear access entry.
  curr_stmt_.access.clear();
  allow_append_ = false;
  // traverse body block
  this->VisitStmt(op->body);
}

void TileLangStorageAccessVisitor::VisitStmt_(const BlockNode *op) {
  auto block = Downcast<Block>(op);
  for (const auto &buffer : block->alloc_buffers) {
    ICHECK(buffer->IsInstance<BufferNode>());
    buffer_data_to_buffer_.Set(buffer->data, buffer);
  }
  IRVisitorWithAnalyzer::VisitStmt_(op);
}

void TileLangStorageAccessVisitor::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == tvm::tir::attr::double_buffer_write) {
    ICHECK(double_buffer_write_ == nullptr);
    double_buffer_write_ = op->node.as<VarNode>();
    scope_.push_back(std::vector<StmtEntry>());
    IRVisitorWithAnalyzer::VisitStmt_(op);
    StmtEntry s;
    s.stmt = op;
    s.access = Summarize(std::move(scope_.back()), nullptr);
    scope_.pop_back();
    if (!s.access.empty()) {
      for (AccessEntry &e : s.access) {
        if (e.type == kWrite && e.buffer.get() == double_buffer_write_) {
          e.double_buffer_write = true;
        }
      }
      scope_.back().emplace_back(std::move(s));
    }
    double_buffer_write_ = nullptr;
  } else if (op->attr_key == tvm::tir::attr::coproc_scope) {
    IterVar iv = Downcast<IterVar>(op->node);
    env_threads_.push_back(iv);
    IRVisitorWithAnalyzer::VisitStmt_(op);
    env_threads_.pop_back();
  } else if (op->attr_key == tvm::tir::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    env_threads_.push_back(iv);
    ICHECK_NE(iv->thread_tag.length(), 0U);
    analyzer_.Bind(
        iv->var, Range::FromMinExtent(IntImm(op->value->dtype, 0), op->value));

    if (!in_device_env_) {
      in_device_env_ = true;
      scope_.push_back(std::vector<StmtEntry>());
      IRVisitorWithAnalyzer::VisitStmt_(op);
      // no need to take the result as the thread barrier automatically syncs.
      Summarize(std::move(scope_.back()), nullptr);
      in_device_env_ = false;
      scope_.pop_back();
    } else {
      IRVisitorWithAnalyzer::VisitStmt_(op);
    }
    env_threads_.pop_back();
  } else if (op->attr_key == tvm::tir::attr::hand_threaded) {
    // skip this pass on blocks that were hand_threaded
    // this avoids control flow and read/write conflicts
    // between hand-threaded kernels and automatic threading
  } else {
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }
}

void TileLangStorageAccessVisitor::VisitStmt_(const ForNode *op) {
  scope_.push_back(std::vector<StmtEntry>());
  IRVisitorWithAnalyzer::VisitStmt_(op);
  StmtEntry s;
  s.stmt = op;
  s.access = Summarize(std::move(scope_.back()), op);
  scope_.pop_back();
  if (!s.access.empty()) {
    // relax the touched set to contain all ranges in the loop.
    std::unordered_map<const VarNode *, arith::IntSet> relax_map;
    relax_map[op->loop_var.get()] =
        arith::IntSet::FromRange(Range::FromMinExtent(op->min, op->extent));
    for (AccessEntry &e : s.access) {
      if (e.buffer.defined()) {
        ICHECK(!e.touched.empty());
        Array<arith::IntSet> new_touched;
        for (const auto &touched : e.touched) {
          new_touched.push_back(arith::EvalSet(touched, relax_map));
        }
        e.touched = std::move(new_touched);
      }
    }
  }
  if (!s.access.empty()) {
    scope_.back().emplace_back(std::move(s));
  }
}

bool IsThreadInvariant(const PrimExpr &cond) {
  if (auto call = cond.as<CallNode>()) {
    if (auto opt_call_op = call->op.as<Op>()) {
      const auto &call_op = opt_call_op.value();
      if (call_op.same_as(builtin::tvm_thread_invariant())) {
        return true;
      }
    }
  }
  return false;
}

/**
 * @brief Visit an IfThenElse statement and collect storage access summaries for
 * its branches.
 *
 * Visits the if-then-else node's condition and both branches to summarize
 * buffer reads, writes, and synchronization events under the condition's
 * constraints. If the condition is not thread-invariant, increments an internal
 * condition counter for the duration of processing.
 *
 * Behavior and side effects:
 * - Evaluates the condition expression (using ExtractRealCondition) and applies
 * it as a constraint while summarizing the then-branch.
 * - For the else-branch (when present), applies the negated,
 * analyzer-simplified condition
 *   (analyzer_.rewrite_simplify(Not(real_condition))) as the constraint.
 * - Accumulates summarized StmtEntry access information for the then/else
 * branches and appends a combined StmtEntry for the IfThenElseNode into the
 * current scope.
 * - Temporarily toggles allow_append_ and clears curr_stmt_.access during
 * condition evaluation and branch summarization.
 * - Modifies internal state: scope_ (push/pop of temporary branch scopes),
 * curr_stmt_.access, and condition_counter_ (incremented/decremented when the
 * condition is not thread-invariant).
 */
void TileLangStorageAccessVisitor::VisitStmt_(const IfThenElseNode *op) {
  bool is_thread_invariant = IsThreadInvariant(op->condition);
  if (!is_thread_invariant) {
    ++condition_counter_;
  }

  allow_append_ = true;
  this->VisitExpr(op->condition);
  PrimExpr real_condition = ExtractRealCondition(op->condition);

  // Preserve accesses collected from the condition expression so they
  // participate in dependency analysis. Otherwise, a write to shared memory
  // immediately followed by an if-condition reading that memory would not
  // trigger a sync before the if-statement.
  std::vector<AccessEntry> cond_access = std::move(curr_stmt_.access);
  allow_append_ = false;

  scope_.push_back(std::vector<StmtEntry>());
  {
    With<arith::ConstraintContext> constraint(&analyzer_, real_condition);
    this->VisitStmt(op->then_case);
  }

  StmtEntry s;
  s.stmt = op;
  s.access = Summarize(std::move(scope_.back()), nullptr);
  scope_.pop_back();
  // Merge the condition's access summary into the if-statement's access list
  // so the planner can insert a sync before the if when necessary.
  if (!cond_access.empty()) {
    s.access.insert(s.access.begin(), cond_access.begin(), cond_access.end());
  }
  if (op->else_case) {
    scope_.push_back(std::vector<StmtEntry>());
    {
      With<arith::ConstraintContext> constraint(
          &analyzer_, analyzer_.rewrite_simplify(Not(real_condition)));
      this->VisitStmt(op->else_case.value());
    }
    auto v = Summarize(std::move(scope_.back()), nullptr);
    scope_.pop_back();
    s.access.insert(s.access.end(), v.begin(), v.end());
  }
  scope_.back().emplace_back(std::move(s));
  if (!is_thread_invariant) {
    --condition_counter_;
  }
}

void TileLangStorageAccessVisitor::VisitStmt_(const WhileNode *op) {
  bool is_thread_invariant = IsThreadInvariant(op->condition);
  if (!is_thread_invariant) {
    ++condition_counter_;
  }
  this->VisitExpr(op->condition);
  scope_.push_back(std::vector<StmtEntry>());
  this->VisitStmt(op->body);
  StmtEntry s;
  s.stmt = op;
  s.access = Summarize(std::move(scope_.back()), nullptr);
  scope_.pop_back();
  scope_.back().emplace_back(std::move(s));
  if (!is_thread_invariant) {
    --condition_counter_;
  }
}

void TileLangStorageAccessVisitor::VisitExpr_(const CallNode *op) {
  // Mark async TMA load context so that tvm_access_ptr within the call
  // can be tagged accordingly.
  auto is_tma_load = [&]() {
    if (auto opt = op->op.as<Op>()) {
      const Op &call_op = opt.value();
      return call_op.same_as(tl::tma_load()) ||
             call_op.same_as(tl::tma_load_im2col());
    }
    return false;
  }();
  if (is_tma_load) {
    tma_depth_++;
    for (const auto &a : op->args) {
      this->VisitExpr(a);
    }
    tma_depth_--;
    return;
  }
  if (op->op.same_as(builtin::address_of())) {
    ICHECK_EQ(op->args.size(), 1U);
    if (auto load = op->args[0].as<BufferLoadNode>()) {
      Buffer buffer = load->buffer;
      DataType dtype = buffer->dtype;
      const VarNode *buffer_var = buffer->data.as<VarNode>();
      buffer_data_to_buffer_.Set(tvm::ffi::GetRef<Var>(buffer_var), buffer);
      StorageScope scope = GetScope(tvm::ffi::GetRef<Var>(buffer_var));
      Array<Range> buffer_ranges;
      // from indices to buffer indices
      ICHECK(buffer->shape.size() == load->indices.size());
      // Use buffer shape and indices to compute the buffer_ranges for each
      // dimension.
      for (size_t i = 0; i < buffer->shape.size(); ++i) {
        PrimExpr min = load->indices[i];
        PrimExpr extent = make_const(buffer->shape[i].dtype(), 1);
        buffer_ranges.push_back(Range::FromMinExtent(min, extent));
      }
      if (Enabled(buffer_var, scope)) {
        ICHECK(allow_append_);
        AccessEntry e;
        e.threads = env_threads();
        e.thread_range = this->ComputeThreadRange(e.threads);
        e.dtype = dtype;
        e.buffer = Downcast<Var>(buffer->data);
        e.buffer_ranges = buffer_ranges;
        for (const auto &index : load->indices) {
          e.touched.push_back(arith::IntSet::Vector(index));
        }
        e.is_pointer_access = true;
        e.type = kRead;
        e.scope = scope;
        curr_stmt_.access.emplace_back(e);
      }
      IRVisitorWithAnalyzer::VisitExpr_(load);
    } else {
      IRVisitorWithAnalyzer::VisitExpr_(op);
    }
  } else if (op->op.same_as(builtin::tvm_access_ptr())) {
    ICHECK_EQ(op->args.size(), 5U);
    DataType dtype = op->args[0].dtype();
    const VarNode *buffer_var = op->args[1].as<VarNode>();
    PrimExpr offset = op->args[2];
    PrimExpr extent = op->args[3];
    const IntImmNode *flag = op->args[4].as<IntImmNode>();
    StorageScope scope = GetScope(tvm::ffi::GetRef<Var>(buffer_var));
    // The buffer scope.
    if (Enabled(buffer_var, scope)) {
      ICHECK(allow_append_);
      Array<Range> buffer_ranges;
      if (buffer_data_to_buffer_.find(tvm::ffi::GetRef<Var>(buffer_var)) ==
          buffer_data_to_buffer_.end()) {
        // cannot find buffer map, use the default buffer
        buffer_ranges = {Range::FromMinExtent(offset, extent)};
      } else {
        Buffer buffer =
            buffer_data_to_buffer_.at(tvm::ffi::GetRef<Var>(buffer_var));
        auto buffer_shape = buffer->shape;
        // convert 1d offset to multi-dimensional index
        auto linear_to_indices = [this](PrimExpr offset,
                                        const Array<PrimExpr> &shape) {
          Array<PrimExpr> indices;
          PrimExpr remaining = std::move(offset);
          for (size_t i = 0; i < shape.size(); ++i) {
            PrimExpr stride = make_const(DataType::Int(32), 1);
            for (size_t j = i + 1; j < shape.size(); ++j) {
              stride = stride * shape[j];
            }
            PrimExpr idx = FloorDiv(remaining, stride);
            remaining = FloorMod(remaining, stride);
            indices.push_back(analyzer_.Simplify(idx));
          }
          return indices;
        };
        Array<PrimExpr> start_indices = linear_to_indices(offset, buffer_shape);
        Array<PrimExpr> end_indices =
            linear_to_indices(offset + extent, buffer_shape);
        for (size_t i = 0; i < buffer_shape.size(); ++i) {
          buffer_ranges.push_back(Range::FromMinExtent(
              start_indices[i],
              analyzer_.Simplify(end_indices[i] - start_indices[i])));
        }
      }
      AccessEntry e;
      e.threads = env_threads();
      e.thread_range = this->ComputeThreadRange(e.threads);
      e.dtype = dtype;
      e.buffer = tvm::ffi::GetRef<Var>(buffer_var);
      e.buffer_ranges = buffer_ranges;
      e.is_pointer_access = true;
      e.touched = {
          arith::IntSet::FromRange(Range::FromMinExtent(offset, extent))};
      e.scope = scope;
      if (flag->value & 1) {
        e.type = kRead;
        e.is_async_copy = (tma_depth_ > 0);
        curr_stmt_.access.emplace_back(e);
      }
      if (flag->value & 2) {
        e.type = kWrite;
        e.is_async_copy = (tma_depth_ > 0);
        curr_stmt_.access.emplace_back(e);
      }
    }
    IRVisitorWithAnalyzer::VisitExpr_(op);
  } else if (op->op.same_as(builtin::tvm_storage_sync())) {
    ICHECK(allow_append_);
    const std::string &s = op->args[0].as<StringImmNode>()->value;
    if (s != "warp") {
      StorageScope scope = StorageScope::Create(s);
      AccessEntry e;
      e.threads = env_threads();
      e.thread_range = this->ComputeThreadRange(e.threads);
      e.type = kSync;
      e.scope = StorageScope::Create(s);
      curr_stmt_.access.emplace_back(std::move(e));
    }
  } else {
    IRVisitorWithAnalyzer::VisitExpr_(op);
  }
}

Map<Var, Range> TileLangStorageAccessVisitor::ComputeThreadRange(
    const Array<IterVar> &threads) {
  Map<Var, Range> thread_range;
  for (const auto &th : threads) {
    auto thread_tag = th->thread_tag;
    if (thread_tag == "threadIdx.x" || thread_tag == "threadIdx.y" ||
        thread_tag == "threadIdx.z") {
      auto const_int_bound = analyzer_.const_int_bound(th->var);
      auto min_value = const_int_bound->min_value;
      auto max_value = const_int_bound->max_value;
      auto extent = max_value - min_value + 1;
      auto dtype = th->var.dtype();
      thread_range.Set(th->var, Range::FromMinExtent(IntImm(dtype, min_value),
                                                     IntImm(dtype, extent)));
    }
  }
  return thread_range;
}

StorageScope
TileLangStorageAccessVisitor::GetScope(const Var &buffer_var) const {
  if (buffer_var->type_annotation.as<PointerTypeNode>()) {
    return StorageScope::Create(GetPtrStorageScope(buffer_var));
  }
  return StorageScope(); // global by default
}

} // namespace tl
} // namespace tvm
