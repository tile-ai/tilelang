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
 * \file merge_shared_memory_allocations.cc
 * \brief Each GPU kernel is allowed to have only one dynamic or static shared
 * memory allocation. This pass merges multiple TIR-level dynamic or static
 * shared memory allocations into one allocation.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <queue>
#include <sstream>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../layout/layout.h"
#include "../op/builtin.h"
#include "../target/utils.h"
#include "./common/epoch_graph.h"
#include "./common/shared_access_analysis.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"
#include "tvm/tir/function.h"

namespace tvm {
namespace tl {

using namespace tir;

using runtime::StorageRank;
using runtime::StorageScope;

static bool IsDynamicSharedMemory(Var buffer_var) {
  StorageScope storage_scope =
      runtime::StorageScope::Create(GetPtrStorageScope(std::move(buffer_var)));
  return storage_scope.rank == runtime::StorageRank::kShared &&
         storage_scope.tag == ".dyn";
}

static bool IsStaticSharedMemory(Var buffer_var) {
  StorageScope storage_scope =
      runtime::StorageScope::Create(GetPtrStorageScope(std::move(buffer_var)));
  return storage_scope.rank == runtime::StorageRank::kShared &&
         storage_scope.tag.empty();
}

/*!
 * \brief collect the mapping from the buffer var to its allocate
 */
class AllocateCollector : public StmtExprVisitor {
public:
  void VisitStmt_(const AllocateNode *op) final {
    if (IsDynamicSharedMemory(op->buffer_var)) {
      dyn_shmem_allocs_[op->buffer_var.get()] = op;
    } else if (IsStaticSharedMemory(op->buffer_var)) {
      static_shmem_allocs_[op->buffer_var.get()] = op;
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  // The dynamic mapping from the original buffer var to its allocate
  std::unordered_map<const VarNode *, const AllocateNode *> dyn_shmem_allocs_;
  // The static mapping from the original buffer var to its allocate
  std::unordered_map<const VarNode *, const AllocateNode *>
      static_shmem_allocs_;
};

// Find a linear pattern of storage access
// Used for liveness analysis.
// "linear" means fitting a complex access pattern into an array of StmtEntry
//
// Define "scope" as the body of For/thread_launch/IfThenElse
// Composite scopes(loop/thread_launch/IfThen) is represented by three
// StmtEntry: before_scope -> scope_body -> after_scope
//
// This pass tries to detect last point that we need to keep memory
// alive under the same scope as Allocate.
// The storage need to be kept alive between Allocate and last access.
// The free point is only inserted at the same scope of Allocate.
//
class SharedMemLinearAccessPatternFinder final : public ConstrVisitor {
public:
  explicit SharedMemLinearAccessPatternFinder(
      bool is_dynamic = true, bool enable_aggressive_merge = false,
      bool verbose = false)
      : is_dynamic_(is_dynamic),
        enable_aggressive_merge_(enable_aggressive_merge), verbose_(verbose) {}
  /*! \brief record the touch list of statement. */
  struct StmtEntry {
    // The statement
    const Object *stmt{};
    // The index in the linear_seq_ to point to end of the nested scope.
    // This is only set to non-zero if stmt is a nested scope.
    // if offset > 0, means this is the begin, the end entry is current_index +
    // offset if offset < 0, means this is the end, the begin entry is
    // current_index + offset
    int64_t scope_pair_offset{0};
    // The buffer variables this statement touched.
    std::vector<const VarNode *> touched;
    // Read accesses attributed to this statement.
    std::vector<const VarNode *> read_touched;
    // Write accesses attributed to this statement.
    std::vector<const VarNode *> write_touched;
    // Precise access records for shared_access_analysis.
    std::vector<shared_access_analysis::AccessEntry> access;
    // Source object for each precise access entry.
    std::vector<const Object *> access_nodes;
    // Whether this statement is an explicit shared-memory sync.
    bool is_sync{false};
  };
  // The scope of each allocation
  struct AllocEntry {
    // the level in the scope stack
    size_t level{0};
    // allocation stmt
    const AllocateNode *alloc{nullptr};
  };

  struct StmtAttr {
    // the level in the scope stack
    size_t level{0};
    // The nearest enclosing if/else scope begin index in linear_seq_, if any.
    int enclosing_if_begin{-1};
    // The full enclosing if/else ancestry from outermost to innermost.
    std::vector<int> enclosing_if_path;
  };

  void UpdateStmtAttr(const Object *stmt, size_t level) {
    if (stmt_attrs_.find(stmt) == stmt_attrs_.end()) {
      stmt_attrs_[stmt] =
          StmtAttr{level, CurrentIfScopeBegin(), if_scope_begin_stack_};
    } else {
      stmt_attrs_[stmt].level = level;
      stmt_attrs_[stmt].enclosing_if_begin = CurrentIfScopeBegin();
      stmt_attrs_[stmt].enclosing_if_path = if_scope_begin_stack_;
    }
  }

  int CurrentIfScopeBegin() const {
    if (if_scope_begin_stack_.empty()) {
      return -1;
    }
    return if_scope_begin_stack_.back();
  }

  void VisitStmt_(const AllocateNode *op) final {
    size_t level = scope_.size();
    const VarNode *buf = op->buffer_var.get();
    // Record the allocation site and depth so liveness can reason about the
    // original scope.
    alloc_info_[buf].alloc = op;
    alloc_info_[buf].level = level;
    StmtExprVisitor::VisitStmt_(op);
  }

  static void PushUnique(std::vector<const VarNode *> *buffers,
                         const VarNode *buf) {
    if (std::find(buffers->begin(), buffers->end(), buf) == buffers->end()) {
      buffers->push_back(buf);
    }
  }

  void RecordAccess(size_t scope_index, const VarNode *buf, bool is_write,
                    const Object *access_stmt = nullptr) {
    if (access_stmt != nullptr) {
      last_access_stmt_[buf] = access_stmt;
    }
    PushUnique(&scope_[scope_index].touched, buf);
    if (is_write) {
      PushUnique(&scope_[scope_index].write_touched, buf);
    } else {
      PushUnique(&scope_[scope_index].read_touched, buf);
    }
  }

  runtime::StorageScope GetSummaryScope(const VarNode *buf) const {
    return runtime::StorageScope::Create(
        GetPtrStorageScope(tvm::ffi::GetRef<Var>(buf)));
  }

  void RecordPreciseAccess(size_t scope_index,
                           shared_access_analysis::AccessEntry access,
                           const Object *access_node) {
    if (scope_index < scope_.size()) {
      scope_[scope_index].access.push_back(std::move(access));
      scope_[scope_index].access_nodes.push_back(access_node);
    }
  }

  Array<Range> MakeRangesFromIndices(const Buffer &buffer,
                                     const Array<PrimExpr> &indices) const {
    Array<Range> ranges;
    if (!buffer.defined() || buffer->shape.size() != indices.size()) {
      return ranges;
    }
    for (const PrimExpr &index : indices) {
      if (const auto *ramp = index.as<RampNode>()) {
        ranges.push_back(Range::FromMinExtent(ramp->base, ramp->lanes));
      } else {
        ranges.push_back(Range::FromMinExtent(index, 1));
      }
    }
    return ranges;
  }

  Array<Range> MakeRangesFromLinearAccess(const Buffer &buffer, PrimExpr offset,
                                          PrimExpr extent) const {
    Array<Range> buffer_ranges;
    if (!buffer.defined() || buffer->shape.empty()) {
      buffer_ranges.push_back(Range::FromMinExtent(offset, extent));
      return buffer_ranges;
    }

    auto linear_to_indices = [](PrimExpr linear_offset,
                                const Array<PrimExpr> &shape) {
      Array<PrimExpr> indices;
      DataType index_dtype = linear_offset.dtype();
      ICHECK(index_dtype.is_int() || index_dtype.is_uint())
          << "Expected integer offset dtype in tvm_access_ptr, but got "
          << index_dtype;
      PrimExpr remaining = std::move(linear_offset);
      for (size_t i = 0; i < shape.size(); ++i) {
        PrimExpr stride = make_const(index_dtype, 1);
        for (size_t j = i + 1; j < shape.size(); ++j) {
          PrimExpr dim = shape[j];
          if (dim.dtype() != index_dtype) {
            dim = tir::Cast(index_dtype, dim);
          }
          stride = stride * dim;
        }
        PrimExpr idx = FloorDiv(remaining, stride);
        remaining = FloorMod(remaining, stride);
        indices.push_back(idx);
      }
      return indices;
    };

    Array<PrimExpr> start_indices = linear_to_indices(offset, buffer->shape);
    Array<PrimExpr> end_indices =
        linear_to_indices(offset + extent, buffer->shape);
    for (size_t i = 0; i < buffer->shape.size(); ++i) {
      buffer_ranges.push_back(Range::FromMinExtent(
          start_indices[i], end_indices[i] - start_indices[i]));
    }
    return buffer_ranges;
  }

  shared_access_analysis::AccessEntry
  MakePreciseAccess(const Buffer &buffer, const Array<PrimExpr> &indices,
                    DataType dtype, shared_access_analysis::AccessType type,
                    bool is_pointer_access = false) const {
    shared_access_analysis::AccessEntry access;
    access.cset = GetConstrSet();
    access.threads = env_threads_;
    access.buffer = buffer->data;
    access.buffer_name = buffer;
    access.buffer_indices = indices;
    access.buffer_ranges = MakeRangesFromIndices(buffer, indices);
    access.dtype = dtype;
    for (const PrimExpr &index : indices) {
      access.touched.push_back(arith::IntSet::Vector(index));
    }
    access.type = type;
    access.scope = GetSummaryScope(buffer->data.get());
    access.is_pointer_access = is_pointer_access;
    return access;
  }

  shared_access_analysis::AccessEntry
  MakePointerAccess(const VarNode *buf,
                    shared_access_analysis::AccessType type) const {
    shared_access_analysis::AccessEntry access;
    access.cset = GetConstrSet();
    access.threads = env_threads_;
    access.buffer = tvm::ffi::GetRef<Var>(buf);
    access.dtype = DataType::UInt(8);
    access.type = type;
    access.scope = GetSummaryScope(buf);
    access.is_pointer_access = true;
    return access;
  }

  shared_access_analysis::AccessEntry
  MakeLinearPointerAccess(const VarNode *buf, const Buffer &buffer,
                          PrimExpr offset, PrimExpr extent, DataType dtype,
                          shared_access_analysis::AccessType type) const {
    shared_access_analysis::AccessEntry access;
    access.cset = GetConstrSet();
    access.threads = env_threads_;
    access.buffer = tvm::ffi::GetRef<Var>(buf);
    access.buffer_name = buffer;
    access.buffer_ranges = MakeRangesFromLinearAccess(buffer, offset, extent);
    access.dtype = dtype;
    access.touched = {
        arith::IntSet::FromRange(Range::FromMinExtent(offset, extent))};
    access.type = type;
    access.scope = GetSummaryScope(buf);
    access.is_pointer_access = true;
    return access;
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    scope_.push_back(StmtEntry());
    // visit subexpr
    ConstrVisitor::VisitStmt_(op);
    // Add write access.
    buffer_data_to_buffer_.Set(op->buffer->data, op->buffer);
    const VarNode *buf = op->buffer->data.get();
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.alloc) {
      ICHECK_LT(it->second.level, scope_.size());
      if (IsAppropriateSharedMemory(tvm::ffi::GetRef<Var>(buf))) {
        // When accesses happen under an if/else tree, attributing them to the
        // innermost statement gives a much tighter approximation than pushing
        // everything to the allocation frame.  This allows branch-exclusive
        // shared buffers to reuse the same arena slot even without enabling
        // the more global aggressive mode.
        bool use_innermost_scope =
            enable_aggressive_merge_ || if_scope_depth_ > 0;
        size_t access_level =
            use_innermost_scope ? scope_.size() - 1 : it->second.level;
        if (use_innermost_scope) {
          RecordAccess(scope_.size() - 1, buf, true, op);
        } else {
          RecordAccess(it->second.level, buf, true, op);
        }
        RecordPreciseAccess(access_level,
                            MakePreciseAccess(op->buffer, op->indices,
                                              op->value.dtype().element_of(),
                                              shared_access_analysis::kWrite),
                            op);
      }
    }

    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (!e.touched.empty() || !e.access.empty()) {
      e.stmt = op;
      UpdateStmtAttr(op, scope_level_);
      linear_seq_.push_back(e);
    }
  }

  void VisitStmt_(const EvaluateNode *op) final {
    bool is_shared_sync = false;
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_storage_sync()) &&
          !call->args.empty()) {
        if (const auto *scope_imm = call->args[0].as<StringImmNode>()) {
          is_shared_sync = IsRelevantSharedSync(scope_imm->value);
        }
      }
    }
    scope_.push_back(StmtEntry());
    // visit subexpr
    ConstrVisitor::VisitStmt_(op);
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (!e.touched.empty() || !e.access.empty() || is_shared_sync) {
      e.stmt = op;
      e.is_sync = is_shared_sync;
      UpdateStmtAttr(op, scope_level_);
      linear_seq_.push_back(e);
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    // Add write access.
    ConstrVisitor::VisitExpr_(op);
    buffer_data_to_buffer_.Set(op->buffer->data, op->buffer);
    const VarNode *buf = op->buffer->data.get();
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.alloc) {
      // Earlier we required `alloc_level < scope_.size()`, assuming every load
      // would occur strictly inside a nested scope.  In practice the lowering
      // pipeline may materialise reads in the very same frame that owns the
      // allocation (e.g. when the buffer value is passed directly to a call),
      // which used to trigger the CHECK.  Treat same-level accesses as valid so
      // the merged allocator can reason about their lifetime correctly.
      ICHECK_LE(it->second.level, scope_.size())
          << "Load memory in places other than store.";
      if (IsAppropriateSharedMemory(tvm::ffi::GetRef<Var>(buf))) {
        bool use_innermost_scope =
            enable_aggressive_merge_ || if_scope_depth_ > 0;
        size_t access_level = 0;
        if (use_innermost_scope) {
          access_level = scope_.size() - 1;
          RecordAccess(scope_.size() - 1, buf, false, op);
        } else {
          // When the access happens in the same scope frame as the allocation
          // we attribute it to that frame instead of the outer parent.  This
          // keeps the liveness window tight while still accounting for nested
          // scopes that legitimately touch the buffer deeper in the tree.
          access_level = std::min(it->second.level, scope_.size() - 1);
          RecordAccess(access_level, buf, false, op);
        }
        RecordPreciseAccess(access_level,
                            MakePreciseAccess(op->buffer, op->indices,
                                              op->dtype.element_of(),
                                              shared_access_analysis::kRead),
                            op);
      }
    }
  }

  void VisitExpr_(const VarNode *buf) final {
    // Directly reference to the variable count as a read.
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.alloc) {
      // Same rationale as the BufferLoad path above: direct references can be
      // emitted at the allocation level after flattening, so accept them and
      // record the touch for liveness planning.
      ICHECK_LE(it->second.level, scope_.size());
      if (IsAppropriateSharedMemory(tvm::ffi::GetRef<Var>(buf))) {
        bool use_innermost_scope =
            enable_aggressive_merge_ || if_scope_depth_ > 0;
        if (use_innermost_scope) {
          RecordAccess(scope_.size() - 1, buf, false, buf);
          RecordPreciseAccess(
              scope_.size() - 1,
              MakePointerAccess(buf, shared_access_analysis::kRead), buf);
        } else {
          // Attribute same-level uses to the allocation frame, mirroring the
          // BufferLoad handling to keep reuse decisions consistent.
          size_t access_level = std::min(it->second.level, scope_.size() - 1);
          RecordAccess(access_level, buf, false, buf);
          RecordPreciseAccess(
              access_level,
              MakePointerAccess(buf, shared_access_analysis::kRead), buf);
        }
      }
    }
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::access_ptr())) {
      ICHECK_EQ(op->args.size(), 3U);
      const auto *base_load = op->args[0].as<BufferLoadNode>();
      const IntImmNode *flag = op->args[2].as<IntImmNode>();
      if (base_load != nullptr && flag != nullptr) {
        buffer_data_to_buffer_.Set(base_load->buffer->data, base_load->buffer);
        const VarNode *buf = base_load->buffer->data.get();
        auto it = alloc_info_.find(buf);
        if (it != alloc_info_.end() && it->second.alloc &&
            IsAppropriateSharedMemory(tvm::ffi::GetRef<Var>(buf))) {
          size_t access_level =
              enable_aggressive_merge_ || if_scope_depth_ > 0
                  ? scope_.size() - 1
                  : std::min(it->second.level, scope_.size() - 1);
          if (flag->value & 1) {
            RecordAccess(access_level, buf, false, op);
            RecordPreciseAccess(access_level,
                                MakePreciseAccess(base_load->buffer,
                                                  base_load->indices,
                                                  base_load->dtype.element_of(),
                                                  shared_access_analysis::kRead,
                                                  /*is_pointer_access=*/true),
                                op);
          }
          if (flag->value & 2) {
            RecordAccess(access_level, buf, true, op);
            RecordPreciseAccess(
                access_level,
                MakePreciseAccess(base_load->buffer, base_load->indices,
                                  base_load->dtype.element_of(),
                                  shared_access_analysis::kWrite,
                                  /*is_pointer_access=*/true),
                op);
          }
        }
      }
    } else if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5U);
      const VarNode *buffer_var = op->args[1].as<VarNode>();
      const IntImmNode *flag = op->args[4].as<IntImmNode>();
      if (buffer_var != nullptr && flag != nullptr) {
        auto it = alloc_info_.find(buffer_var);
        if (it != alloc_info_.end() && it->second.alloc &&
            IsAppropriateSharedMemory(tvm::ffi::GetRef<Var>(buffer_var))) {
          Buffer buffer;
          if (auto buffer_opt = buffer_data_to_buffer_.Get(
                  tvm::ffi::GetRef<Var>(buffer_var))) {
            buffer = buffer_opt.value();
          }
          size_t access_level =
              enable_aggressive_merge_ || if_scope_depth_ > 0
                  ? scope_.size() - 1
                  : std::min(it->second.level, scope_.size() - 1);
          if (flag->value & 1) {
            RecordAccess(access_level, buffer_var, false, op);
            RecordPreciseAccess(
                access_level,
                MakeLinearPointerAccess(buffer_var, buffer, op->args[2],
                                        op->args[3], op->args[0].dtype(),
                                        shared_access_analysis::kRead),
                op);
          }
          if (flag->value & 2) {
            RecordAccess(access_level, buffer_var, true, op);
            RecordPreciseAccess(
                access_level,
                MakeLinearPointerAccess(buffer_var, buffer, op->args[2],
                                        op->args[3], op->args[0].dtype(),
                                        shared_access_analysis::kWrite),
                op);
          }
        }
      }
    }
    ConstrVisitor::VisitExpr_(op);
  }

  template <typename T> void VisitNewScope(const T *op) {
    scope_.push_back(StmtEntry());
    StmtEntry e;
    e.stmt = op;
    UpdateStmtAttr(op, scope_level_);
    int64_t begin_index = static_cast<int64_t>(linear_seq_.size());
    // before scope.
    linear_seq_.push_back(e);
    ConstrVisitor::VisitStmt_(op);
    // after scope.
    e.touched = std::move(scope_.back().touched);
    e.read_touched = std::move(scope_.back().read_touched);
    e.write_touched = std::move(scope_.back().write_touched);
    e.access = std::move(scope_.back().access);
    e.access_nodes = std::move(scope_.back().access_nodes);
    scope_.pop_back();
    int64_t end_index = static_cast<int64_t>(linear_seq_.size());
    ICHECK_GT(end_index, begin_index);
    // The paired entries serve as scope sentinels once we flatten the
    // control-flow tree.
    e.scope_pair_offset = begin_index - end_index;
    linear_seq_.push_back(e);
    // record the pointer to end index.
    ICHECK_NE(end_index, 0U);
    linear_seq_[begin_index].scope_pair_offset = end_index - begin_index;
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    // Only record the outer most thread extent.
    if (op->attr_key == tir::attr::thread_extent && !in_thread_env_) {
      IterVar iv = Downcast<IterVar>(op->node);
      env_threads_.push_back(iv);
      in_thread_env_ = true;
      VisitNewScope(op);
      in_thread_env_ = false;
      env_threads_.pop_back();
    } else if (op->attr_key == tir::attr::extern_scope) {
      VisitNewScope(op);
    } else if (op->attr_key == tir::attr::virtual_thread) {
      VisitNewScope(op);
    } else if (op->attr_key == "kWarpSpecializationScope") {
      VisitWarpSpecializationBody(op->body);
    } else if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      env_threads_.push_back(iv);
      ConstrVisitor::VisitStmt_(op);
      env_threads_.pop_back();
    } else {
      ConstrVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    if_scope_depth_++;
    if_scope_begin_stack_.push_back(static_cast<int>(linear_seq_.size()));
    VisitNewScope(op);
    if_scope_begin_stack_.pop_back();
    if_scope_depth_--;
  }

  bool ContainsSeqStmt(const Stmt &stmt) {
    if (stmt->IsInstance<SeqStmtNode>()) {
      return true;
    }
    if (const auto *if_node = stmt.as<IfThenElseNode>()) {
      return ContainsSeqStmt(if_node->then_case) ||
             (if_node->else_case.defined() &&
              ContainsSeqStmt(if_node->else_case.value()));
    }
    return false;
  }

  void VisitStmt_(const ForNode *op) final {
    if (ContainsSeqStmt(op->body)) {
      scope_level_++;
      VisitNewScope(op);
      scope_level_--;
    } else {
      VisitNewScope(op);
    }
  }

  void VisitStmt_(const WhileNode *op) final { VisitNewScope(op); }

  void VisitStmt_(const AssertStmtNode *op) final { VisitNewScope(op); }

  // linearized access sequence.
  std::vector<StmtEntry> linear_seq_;
  // The storage scope of each buffer
  std::unordered_map<const VarNode *, AllocEntry> alloc_info_;
  // The attribute of each statement
  std::unordered_map<const Object *, StmtAttr> stmt_attrs_;
  // The most recent concrete statement/expression that accessed a buffer.
  std::unordered_map<const VarNode *, const Object *> last_access_stmt_;

private:
  void VisitWarpSpecializationBody(const Stmt &stmt) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &sub_stmt : seq->seq) {
        VisitWarpSpecializationBody(sub_stmt);
      }
      return;
    }
    if (const auto *if_node = stmt.as<IfThenElseNode>()) {
      this->VisitStmt(if_node->then_case);
      if (if_node->else_case.defined()) {
        this->VisitStmt(if_node->else_case.value());
      }
      return;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      VisitWarpSpecializationBody(attr->body);
      return;
    }
    if (const auto *let_node = stmt.as<LetStmtNode>()) {
      this->VisitExpr(let_node->value);
      VisitWarpSpecializationBody(let_node->body);
      return;
    }
    ConstrVisitor::VisitStmt(stmt);
  }

  // Wrapper function to determine if the shared memory allocation for a
  // variable is appropriate.
  bool IsAppropriateSharedMemory(const Var &var) {
    return is_dynamic_ ? IsDynamicSharedMemory(var) : IsStaticSharedMemory(var);
  }

  bool IsRelevantSharedSync(const std::string &sync_scope) const {
    return sync_scope == "shared" || sync_scope == "shared.dyn";
  }

  // Whether do dynamic analysis.
  bool is_dynamic_{true};
  // Whether do aggressive merge.
  bool enable_aggressive_merge_{false};
  // Whether do verbose logging.
  bool verbose_{false};
  // Whether already in thread env.
  bool in_thread_env_{false};
  // Current thread environment for precise access entries.
  Array<IterVar> env_threads_;
  // Optional mapping from buffer data var to full Buffer object.
  ffi::Map<Var, Buffer> buffer_data_to_buffer_;
  // The scope stack.
  std::vector<StmtEntry> scope_;
  // The size of the scope.
  size_t scope_level_{0};
  // Whether we are currently traversing an if/else region.
  size_t if_scope_depth_{0};
  // The begin sentinel indices for nested if/else scopes in linear_seq_.
  std::vector<int> if_scope_begin_stack_;
};

class SharedMemoryAlignmentPlanner : public StmtExprVisitor {

public:
  static std::unordered_map<const VarNode *, int> Plan(const Stmt &stmt) {
    SharedMemoryAlignmentPlanner planner;
    planner(stmt);
    return planner.shmem_alignment_map_;
  }

private:
  // Helper to record alignment for a shared/shared.dyn Var under alignment
  // scope
  void MarkSharedVarIfNeeded(const VarNode *op) {
    if (!op || !under_alignment_scope_)
      return;
    auto ptr_type = op->type_annotation.as<PointerTypeNode>();
    if (!ptr_type)
      return;
    auto scope = GetPtrStorageScope(tvm::ffi::GetRef<Var>(op));
    if (scope == "shared" || scope == "shared.dyn") {
      auto target = Target::Current();
      ICHECK(target.defined()) << "Target is not defined";
      // TMA bulk-copy operands have stricter shared-memory alignment than
      // ordinary scalar/shared accesses. Hopper keeps the existing 1024-byte
      // alignment used by the WGMMA/TMA path, while Blackwell/SM120 also needs
      // TMA sources at least 128-byte aligned to avoid misaligned-address
      // launch failures.
      int alignment = 16;
      if (TargetIsHopper(target)) {
        alignment = 1024;
      } else if (TargetHasBulkCopy(target)) {
        alignment = 128;
      }
      shmem_alignment_map_[op] = alignment;
    }
  }

  void VisitExpr_(const CallNode *op) {
    if (op->op.same_as(tl::tl_gemm()) || op->op.same_as(tl::tl_gemm_sp()) ||
        op->op.same_as(tl::tma_load()) || op->op.same_as(tl::tma_store()) ||
        op->op.same_as(tl::initialize_wgmma_descriptor()) ||
        op->op.same_as(tl::initialize_tcgen05_descriptor())) {
      // These intrinsics introduce stricter SMEM alignment requirements; mark
      // the subtree.
      under_alignment_scope_ = true;
      StmtExprVisitor::VisitExpr_(op);
      under_alignment_scope_ = false;
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

  void VisitExpr_(const VarNode *op) {
    MarkSharedVarIfNeeded(op);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) {
    // If we encounter address_of(BufferLoad(...)) or any direct BufferLoad
    // within an alignment scope, make sure we mark the underlying shared var.
    if (op && under_alignment_scope_) {
      const VarNode *data_var = op->buffer->data.get();
      MarkSharedVarIfNeeded(data_var);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  bool under_alignment_scope_{false};

  std::unordered_map<const VarNode *, int> shmem_alignment_map_;
};

/*!
 * \brief Collect per-buffer layout signatures from `Block` annotations.
 *
 * `T.annotate_layout(...)` writes the user-supplied physical-to-logical mapping
 * into `block.annotations[attr::kLayoutMap]` as a `Map<Var, Layout>`. When two
 * shared-memory buffers carry distinct annotated layouts, downstream passes
 * (layout_inference, lower_tile_op, swizzle lowering) may emit access patterns
 * that are only sound under the *original* layout of each buffer. Aliasing the
 * underlying storage in that case can silently corrupt indexing arithmetic
 * even when the per-epoch dataflow proves the two buffers are never alive
 * simultaneously.
 *
 * This collector harvests the (buffer Var, Layout) pairs so that
 * `SharedMemoryRewriter` can require structural Layout equality before
 * permitting the per-epoch relaxer to drop a conflict between two annotated
 * buffers.  Buffers without an entry are treated as layout-agnostic and remain
 * eligible for aliasing.
 */
class LayoutSignatureCollector final : public StmtVisitor {
public:
  std::unordered_map<const VarNode *, Layout> layouts;

  void VisitStmt_(const BlockNode *op) final {
    auto it = op->annotations.find(attr::kLayoutMap);
    if (it != op->annotations.end()) {
      if (auto layout_map_opt = (*it).second.as<Map<Var, Layout>>()) {
        for (const auto &kv : layout_map_opt.value()) {
          // Keep the first non-trivial entry; conflicting annotations on the
          // same Var across nested blocks are already caught by upstream
          // layout-inference passes.
          layouts.emplace(kv.first.get(), kv.second);
        }
      }
    }
    StmtVisitor::VisitStmt_(op);
  }
};

/*!
 * \brief merge the buffers whose live range has no intersection and rewrite the
 * body
 */
class SharedMemoryRewriter : public StmtExprMutator {
public:
  explicit SharedMemoryRewriter(
      const std::unordered_map<const VarNode *, const AllocateNode *>
          &shmem_allocs,
      bool is_dynamic = true, bool verbose = false, int align_bytes = 0)
      : is_dynamic_{is_dynamic}, shmem_allocs_{shmem_allocs}, verbose_{verbose},
        align_bytes_{align_bytes} {
    if (!is_dynamic) {
      merged_buf_var_ =
          Var("buf_shmem", PointerType(PrimType(DataType::UInt(8)), "shared"));
    }
  }

  /*!
   * \brief plan the memory reuse for all the buffer allocated in the statement
   * \param stmt the statement
   */
  void PlanReuse(const Stmt &stmt, bool is_dynamic = true,
                 bool enable_aggressive_merge = false, bool verbose = false) {
    SharedMemLinearAccessPatternFinder finder(is_dynamic,
                                              enable_aggressive_merge, verbose);
    finder(stmt);
    shmem_alignment_map_ = SharedMemoryAlignmentPlanner::Plan(stmt);
    last_access_stmt_ = finder.last_access_stmt_;
    // Collect per-buffer annotated layouts before liveness analysis so the
    // B2 alias gate (`LayoutAliasIncompatible`) can short-circuit the
    // per-epoch relaxer for buffers with structurally distinct layouts.
    layout_sigs_.clear();
    {
      LayoutSignatureCollector layout_collector;
      layout_collector(stmt);
      layout_sigs_ = std::move(layout_collector.layouts);
    }
    // First compute liveness over the flattened schedule, then feed it into the
    // arena packer.
    this->LivenessAnalysis(finder.linear_seq_, finder.stmt_attrs_);

    // ----- Per-epoch liveness (always-on, drives alias decisions) -----
    //
    // Build the EpochGraph and the (buffer, epoch) access map keyed by
    // VarNode*, then solve the forward+backward liveness fixed point. The
    // resulting `live_epochs` set per buffer is consumed by `PlanMemory`
    // via `liveness_epochs_by_var_` to relax conflicts that the per-epoch
    // dataflow proves safe.
    liveness_epochs_by_var_.clear();
    {
      epoch_graph::EpochGraphBuilder eg_builder;
      epoch_graph::EpochGraph eg = eg_builder.Build(stmt);
      struct EpochAccessAgg {
        bool has_read = false;
        bool has_write = false;
      };
      std::map<int, std::map<const VarNode *, EpochAccessAgg>> agg;
      for (size_t i = 0; i < finder.linear_seq_.size(); ++i) {
        const auto &e = finder.linear_seq_[i];
        int epoch_id = e.stmt ? eg.EpochOf(e.stmt) : -1;
        if (epoch_id < 0)
          continue;
        for (const VarNode *v : e.read_touched)
          agg[epoch_id][v].has_read = true;
        for (const VarNode *v : e.write_touched)
          agg[epoch_id][v].has_write = true;
      }
      std::unordered_map<const VarNode *,
                         std::unordered_map<int, epoch_graph::EpochAccess>>
          live_input;
      for (const auto &epoch_kv : agg) {
        for (const auto &buf_kv : epoch_kv.second) {
          auto &cell = live_input[buf_kv.first][epoch_kv.first];
          cell.def = cell.def || buf_kv.second.has_write;
          cell.use = cell.use || buf_kv.second.has_read;
        }
      }
      auto live_out =
          epoch_graph::ComputePerEpochLiveness<const VarNode *>(eg, live_input);
      for (const auto &kv : live_out) {
        std::set<int> &dst = liveness_epochs_by_var_[kv.first];
        for (const auto &p : kv.second) {
          if (p.second.Live())
            dst.insert(p.first);
        }
      }

      if (verbose_) {
        this->LogBoundarySummaryDeltas(finder.linear_seq_, finder.stmt_attrs_);
        // Per-stmt witness dump: print every linear_seq_ index, scope offset,
        // and the buffers it reads/writes/touches.  Used to map planner indices
        // back to actual TIR statements (and hence CUDA lines) for the lowbit
        // kernel investigation.
        for (size_t i = 0; i < finder.linear_seq_.size(); ++i) {
          const auto &e = finder.linear_seq_[i];
          std::stringstream rs, ws, ts;
          for (const VarNode *v : e.read_touched)
            rs << v->name_hint << " ";
          for (const VarNode *v : e.write_touched)
            ws << v->name_hint << " ";
          for (const VarNode *v : e.touched)
            ts << v->name_hint << " ";
          std::string tk = e.stmt ? std::string(e.stmt->GetTypeKey())
                                  : std::string("<null>");
          int epoch_id = e.stmt ? eg.EpochOf(e.stmt) : -1;
          std::cerr << "[MSMA-SEQ] i=" << i << " kind=" << tk
                    << " scope_off=" << e.scope_pair_offset
                    << " sync=" << (e.is_sync ? 1 : 0) << " epoch=" << epoch_id
                    << " R=[" << rs.str() << "]"
                    << " W=[" << ws.str() << "]"
                    << " T=[" << ts.str() << "]\n";
        }
        eg.Dump(std::cerr);
        for (const auto &epoch_kv : agg) {
          for (const auto &buf_kv : epoch_kv.second) {
            std::cerr << "[MSMA-EPOCH-ACCESS] epoch=" << epoch_kv.first
                      << " buf=" << buf_kv.first->name_hint
                      << " r=" << (buf_kv.second.has_read ? 1 : 0)
                      << " w=" << (buf_kv.second.has_write ? 1 : 0) << "\n";
          }
        }
        // Re-render liveness with name_hint for log readability.
        std::map<int, std::vector<
                          std::pair<std::string, epoch_graph::EpochLiveness>>>
            by_epoch;
        for (const auto &kv : live_out) {
          for (const auto &p : kv.second) {
            if (p.second.Live()) {
              by_epoch[p.first].push_back({kv.first->name_hint, p.second});
            }
          }
        }
        for (const auto &kv : by_epoch) {
          for (const auto &p : kv.second) {
            std::cerr << "[MSMA-EPOCH-LIVE] epoch=" << kv.first
                      << " buf=" << p.first
                      << " in=" << (p.second.live_in ? 1 : 0)
                      << " out=" << (p.second.live_out ? 1 : 0) << "\n";
          }
        }
      }
    }

    this->PlanMemory(finder.linear_seq_, finder.stmt_attrs_);
  }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent && !allocated_) {
      // Allocate one dynamic shared memory allocation at the beginning of
      // thread scope

      if (verbose_) {

        LOG(DEBUG) << "Memory Allocation Plan for "
                   << (is_dynamic_ ? "Dynamic" : "Static") << " Shared Memory:";
        LOG(DEBUG) << "  Merged Buffer Name: " << merged_buf_var_->name_hint;
        LOG(DEBUG) << "  Total Merged Size: " << merged_alloc_size_ << " bytes";
        LOG(DEBUG) << "  Individual Buffer Allocations:";
        for (const auto &pair : buffer_byte_offsets_) {
          const VarNode *buffer_var_node = pair.first;
          PrimExpr byte_offset = pair.second;
          auto alloc_it = shmem_allocs_.find(buffer_var_node);
          if (alloc_it != shmem_allocs_.end()) {
            const AllocateNode *alloc = alloc_it->second;
            PrimExpr buffer_size_bytes =
                alloc->extents[0] * alloc->dtype.bytes() * alloc->dtype.lanes();
            LOG(DEBUG) << "    Buffer: " << buffer_var_node->name_hint
                       << " (Type: " << alloc->dtype << ")"
                       << ", Start Offset: " << byte_offset
                       << ", Size: " << buffer_size_bytes << " bytes"
                       << ", End Offset: "
                       << (byte_offset + buffer_size_bytes - 1);
          } else {
            LOG(DEBUG) << "    Buffer: " << buffer_var_node->name_hint
                       << ", Start Offset: " << byte_offset
                       << " (Original allocation info not found)";
          }
        }
        LOG(DEBUG) << "End of Memory Allocation Plan.";
      }

      allocated_ = true;
      Allocate new_body(merged_buf_var_, DataType::UInt(8),
                        {merged_alloc_size_}, const_true(),
                        StmtExprMutator::VisitStmt(op->body));
      return AttrStmt(op->node, op->attr_key, op->value, new_body, op->span);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocateNode *op) final {
    if (IsAppropriateSharedMemory(op->buffer_var)) {
      return StmtExprMutator::VisitStmt(op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const DeclBufferNode *op) final {
    stmt_visit_index_++;
    auto node = Downcast<DeclBuffer>(StmtExprMutator::VisitStmt_(op));
    auto new_buf = GetUpdatedBuffer(node->buffer);
    if (!new_buf.same_as(node->buffer)) {
      node.CopyOnWrite()->buffer = new_buf;
    }
    return std::move(node);
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    stmt_visit_index_++;
    auto node = Downcast<Block>(StmtExprMutator::VisitStmt_(op));

    auto rewrite_region = [this, op](const BufferRegion &region) {
      if (!region.defined() ||
          !IsAppropriateSharedMemory(region->buffer->data)) {
        return region;
      }
      Array<Range> new_region;
      new_region.reserve(region->region.size());
      PrimExpr elem_offset =
          GetBufferOffset(region->buffer->data, region->buffer->dtype, op);
      if (verbose_) {
        auto binding_it =
            buffer_segment_bindings_.find(region->buffer->data.get());
        if (binding_it != buffer_segment_bindings_.end() &&
            binding_it->second.size() > 1) {
          std::ostringstream os;
          os << "Block BufferRegion rewrite: buffer="
             << region->buffer->data->name_hint
             << ", access_node=" << op->GetTypeKey()
             << ", elem_offset=" << elem_offset;
          if (!region->region.empty()) {
            os << ", first_min=" << region->region[0]->min
               << ", first_extent=" << region->region[0]->extent;
          }
          LOG(DEBUG) << os.str();
        }
      }
      for (size_t i = 0; i < region->region.size(); ++i) {
        Range range = region->region[i];
        if (i == 0) {
          new_region.push_back(
              Range::FromMinExtent(range->min + elem_offset, range->extent));
        } else {
          new_region.push_back(range);
        }
      }

      Buffer new_buffer = GetUpdatedBuffer(region->buffer);
      if (new_buffer.same_as(region->buffer) &&
          new_region.same_as(region->region)) {
        return region;
      }
      return BufferRegion(new_buffer, new_region);
    };

    auto writer = node.CopyOnWrite();
    writer->reads = node->reads.Map(rewrite_region);
    writer->writes = node->writes.Map(rewrite_region);
    writer->alloc_buffers = node->alloc_buffers.Map(
        [this](const Buffer &buffer) { return GetUpdatedBuffer(buffer); });
    writer->match_buffers = node->match_buffers.Map(
        [&rewrite_region, this](const MatchBufferRegion &match_buffer) {
          Buffer new_buffer = GetUpdatedBuffer(match_buffer->buffer);
          BufferRegion new_source = rewrite_region(match_buffer->source);
          if (new_buffer.same_as(match_buffer->buffer) &&
              new_source.same_as(match_buffer->source)) {
            return match_buffer;
          }
          return MatchBufferRegion(new_buffer, new_source);
        });
    return std::move(node);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    stmt_visit_index_++;
    auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    return VisitBufferAccess(std::move(node), op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    stmt_visit_index_++;
    auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    return VisitBufferAccess(std::move(node), op);
  }

  template <typename Node>
  Node VisitBufferAccess(Node node, const Object *access_node) {
    if (IsAppropriateSharedMemory(node->buffer->data)) {
      ICHECK_EQ(node->indices.size(), 1)
          << "MergeSharedMemoryAllocations expects flat memory buffers, "
          << "and is to be run after "
          << "StorageFlatten (TE schedules) or FlattenBuffer (TIR schedules)";
      Array<PrimExpr> indices = {node->indices[0] +
                                 this->GetBufferOffset(node->buffer->data,
                                                       node->buffer->dtype,
                                                       access_node)};

      auto writer = node.CopyOnWrite();
      writer->buffer = GetUpdatedBuffer(node->buffer);
      writer->indices = indices;
    }

    return node;
  }

  Buffer GetUpdatedBuffer(Buffer buffer) {
    auto key = buffer.get();
    auto it = buffer_remap_.find(key);
    if (it != buffer_remap_.end()) {
      return it->second;
    }

    if (IsAppropriateSharedMemory(buffer->data)) {
      ICHECK_EQ(buffer->shape.size(), 1)
          << "Buffer " << buffer << " has shape " << buffer->shape << ".  "
          << "MergeSharedMemoryAllocations expects flat memory buffers, "
          << "and is to be run after "
          << "StorageFlatten (TE schedules) or FlattenBuffer (TIR schedules)";
      auto writer = buffer.CopyOnWrite();
      writer->data = merged_buf_var_;
    }

    buffer_remap_[key] = buffer;
    return buffer;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    stmt_visit_index_++;
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5U);
      DataType dtype = op->args[0].dtype();
      Var buffer = Downcast<Var>(op->args[1]);
      if (!IsAppropriateSharedMemory(buffer)) {
        return StmtExprMutator::VisitExpr_(op);
      }
      PrimExpr extra_offset = GetBufferOffset(buffer, dtype, op);

      PrimExpr offset = this->VisitExpr(op->args[2]);
      PrimExpr extent = this->VisitExpr(op->args[3]);
      return Call(op->dtype, op->op,
                  {op->args[0], merged_buf_var_, extra_offset + offset, extent,
                   op->args[4]});
    } else if (op->op.same_as(builtin::ptx_cp_async()) ||
               op->op.same_as(tl::ptx_cp_async())) {
      ICHECK(op->args.size() == 3U || op->args.size() == 4U)
          << "ptx_cp_async expects 3 or 4 arguments (dst_access_ptr, "
             "src_access_ptr, count[, predicate])";

      // Extract dst_access_ptr. Only the legacy ``tvm_access_ptr`` form
      // exposes the buffer var directly; the newer ``tl::access_ptr``
      // form wraps a BufferLoad whose buffer var will be rewritten by the
      // BufferLoad mutator on the default visit path, so just fall
      // through there.
      Call dst_access_ptr = Downcast<Call>(op->args[0]);
      if (!dst_access_ptr->op.same_as(builtin::tvm_access_ptr())) {
        return StmtExprMutator::VisitExpr_(op);
      }

      // tvm_access_ptr(ptype, data, offset, extent, rw_mask)
      Var buffer = Downcast<Var>(dst_access_ptr->args[1]);
      if (!IsAppropriateSharedMemory(buffer)) {
        return StmtExprMutator::VisitExpr_(op);
      }

      DataType dtype = op->dtype;
      DataType ptr_dtype = dst_access_ptr->args[0].dtype();
      PrimExpr extra_offset =
          GetBufferOffset(buffer, ptr_dtype, dst_access_ptr.get());
      if (verbose_) {
        auto binding_it = buffer_segment_bindings_.find(buffer.get());
        if (binding_it != buffer_segment_bindings_.end() &&
            binding_it->second.size() > 1) {
          LOG(DEBUG) << "ptx_cp_async rewrite: buffer=" << buffer->name_hint
                     << ", access_node=" << dst_access_ptr->GetTypeKey()
                     << ", extra_offset=" << extra_offset
                     << ", raw_offset=" << dst_access_ptr->args[2];
        }
      }
      PrimExpr offset = this->VisitExpr(dst_access_ptr->args[2]);

      // Create new dst_access_ptr with merged buffer and adjusted offset
      auto new_dst_access_ptr =
          Call(DataType::Handle(), builtin::tvm_access_ptr(),
               {
                   dst_access_ptr->args[0], // ptype
                   merged_buf_var_,         // merged buffer
                   extra_offset + offset,   // adjusted offset
                   dst_access_ptr->args[3], // extent
                   dst_access_ptr->args[4]  // rw_mask
               });

      Array<PrimExpr> cp_async_args = {new_dst_access_ptr, op->args[1],
                                       op->args[2]};
      if (op->args.size() == 4U) {
        cp_async_args.push_back(op->args[3]);
      }
      return Call(dtype, op->op, cp_async_args);
    } else {
      if (verbose_) {
        const OpNode *call_op = op->op.as<OpNode>();
        std::string call_name = call_op != nullptr
                                    ? std::string(call_op->name.c_str())
                                    : std::string(op->op->GetTypeKey());
        for (const PrimExpr &arg : op->args) {
          if (const auto *region = arg.as<BufferRegionNode>()) {
            const VarNode *buffer_var = region->buffer->data.get();
            auto binding_it = buffer_segment_bindings_.find(buffer_var);
            if (IsAppropriateSharedMemory(region->buffer->data) &&
                binding_it != buffer_segment_bindings_.end() &&
                binding_it->second.size() > 1) {
              LOG(DEBUG)
                  << "Opaque call with multi-segment BufferRegion arg: op="
                  << call_name << ", buffer=" << buffer_var->name_hint
                  << ", segments=" << binding_it->second.size();
            }
          }
        }
      }
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  PrimExpr GetBufferOffset(const Var &buffer_var, DataType dtype,
                           const Object *access_node = nullptr) {
    auto binding_it = buffer_segment_bindings_.find(buffer_var.get());
    if (access_node != nullptr) {
      if (auto segment_id = ResolveSegmentId(buffer_var.get(), access_node)) {
        if (binding_it != buffer_segment_bindings_.end() &&
            binding_it->second.size() > static_cast<size_t>(*segment_id)) {
          size_t offset =
              binding_it->second[static_cast<size_t>(*segment_id)].offset;
          if (verbose_ && binding_it->second.size() > 1) {
            LOG(DEBUG) << "Resolved segment-aware offset for buffer "
                       << buffer_var->name_hint << " on access node "
                       << access_node->GetTypeKey() << ": seg=" << *segment_id
                       << ", elem_offset="
                       << (offset / (dtype.bytes() * dtype.lanes()))
                       << ", byte_offset=" << offset;
          }
          return indexdiv(
              make_const(DataType::Int(32), static_cast<int64_t>(offset)),
              dtype.bytes() * dtype.lanes());
        }
      }
      if (verbose_) {
        if (binding_it != buffer_segment_bindings_.end() &&
            binding_it->second.size() > 1) {
          LOG(WARNING) << "Missing segment-specific offset for buffer "
                       << buffer_var->name_hint << " on access node "
                       << access_node->GetTypeKey()
                       << "; falling back to base offset.";
        }
      }
    }
    auto it = buffer_byte_offsets_.find(buffer_var.get());
    ICHECK(it != buffer_byte_offsets_.end())
        << "buffer_var = " << buffer_var->name_hint << ", dtype = " << dtype;
    return indexdiv(it->second, dtype.bytes() * dtype.lanes());
  }

  std::optional<int> LookupSegmentId(const VarNode *buffer_var,
                                     const Object *access_node) const {
    if (access_node == nullptr) {
      return std::nullopt;
    }
    auto access_it = access_segment_ids_.find(access_node);
    if (access_it == access_segment_ids_.end()) {
      return std::nullopt;
    }
    auto seg_it = access_it->second.find(buffer_var);
    if (seg_it == access_it->second.end()) {
      return std::nullopt;
    }
    return seg_it->second;
  }

  std::optional<int> ResolveSegmentId(const VarNode *buffer_var,
                                      const Object *access_node) const {
    if (auto direct = LookupSegmentId(buffer_var, access_node)) {
      return direct;
    }

    class NestedAccessProbe final : public StmtExprVisitor {
    public:
      NestedAccessProbe(const SharedMemoryRewriter *rewriter,
                        const VarNode *buffer_var)
          : rewriter_(rewriter), buffer_var_(buffer_var) {}

      std::optional<int> Result() const { return result_; }

      void VisitExpr_(const BufferLoadNode *op) final {
        Probe(op);
        if (!result_) {
          StmtExprVisitor::VisitExpr_(op);
        }
      }

      void VisitStmt_(const BufferStoreNode *op) final {
        Probe(op);
        if (!result_) {
          StmtExprVisitor::VisitStmt_(op);
        }
      }

      void VisitExpr_(const CallNode *op) final {
        Probe(op);
        if (!result_) {
          StmtExprVisitor::VisitExpr_(op);
        }
      }

      void VisitExpr_(const VarNode *op) final {
        Probe(op);
        if (!result_) {
          StmtExprVisitor::VisitExpr_(op);
        }
      }

    private:
      void Probe(const Object *node) {
        if (!result_) {
          result_ = rewriter_->LookupSegmentId(buffer_var_, node);
        }
      }

      const SharedMemoryRewriter *rewriter_;
      const VarNode *buffer_var_;
      std::optional<int> result_;
    };

    NestedAccessProbe probe(this, buffer_var);
    if (access_node->IsInstance<StmtNode>()) {
      probe(tvm::ffi::GetRef<Stmt>(static_cast<const StmtNode *>(access_node)));
    } else if (access_node->IsInstance<PrimExprNode>()) {
      probe(tvm::ffi::GetRef<PrimExpr>(
          static_cast<const PrimExprNode *>(access_node)));
    }
    return probe.Result();
  }

  // Wrapper function to determine if the shared memory allocation for a
  // variable is appropriate.
  bool IsAppropriateSharedMemory(const Var &var) {
    return is_dynamic_ ? IsDynamicSharedMemory(var) : IsStaticSharedMemory(var);
  }

  /*!
   * \brief B2 layout-signature alias gate.
   *
   * Returns true iff both `a` and `b` carry an annotated `Layout` (collected
   * from `Block` annotations under `attr::kLayoutMap`) and the two layouts are
   * not structurally equal.  The caller treats a true return as "alias is
   * unsafe regardless of liveness", short-circuiting the per-epoch relaxer.
   *
   * Buffers without an entry in `layout_sigs_` are layout-agnostic and never
   * trigger the gate, preserving the existing arena-packing behaviour for the
   * common (un-annotated) case.
   */
  bool LayoutAliasIncompatible(const VarNode *a, const VarNode *b) const {
    auto ia = layout_sigs_.find(a);
    if (ia == layout_sigs_.end())
      return false;
    auto ib = layout_sigs_.find(b);
    if (ib == layout_sigs_.end())
      return false;
    return !StructuralEqual()(ia->second, ib->second);
  }

  using StmtEntry = SharedMemLinearAccessPatternFinder::StmtEntry;
  using StmtAttr = SharedMemLinearAccessPatternFinder::StmtAttr;
  using SummaryStmtEntry = shared_access_analysis::StmtEntry;
  using SummaryAccessEntry = shared_access_analysis::AccessEntry;
  using SummaryResult = shared_access_analysis::SequenceSummaryResult;

  // Metadata about a single shared-memory allocation prior to merging.  This
  // is used to build lifetimes, alignment requirements, and final offsets.
  struct BufInfo {
    const VarNode *var{nullptr};
    std::string name;
    PrimExpr size_expr;
    std::optional<int64_t> const_size_bytes; // in bytes if compile-time known.
    int alignment{0};                        // required byte alignment.
    int start{0}; // first statement index touching the buf.
    int end{0};   // one-past-last statement index.
    std::vector<std::pair<int, int>>
        segments; // disjoint [start, end) intervals.
    DataType size_dtype{DataType::Int(32)};
  };

  // Interval describing the liveness window of a (constant-sized) allocation.
  struct Interval {
    int start{0};
    int end{0};
    size_t size_bytes{0};
    int alignment{0};
    const VarNode *var{nullptr};
    int segment_id{0};
    // Optional pointer into `liveness_epochs_by_var_`. If non-null and the
    // pointed-to set is non-empty, two intervals A and B are treated as
    // non-conflicting when their live epoch sets are disjoint, even if their
    // legacy `[start, end)` linear-seq intervals overlap. This is a strict
    // refinement: the legacy decision is preserved unless the per-epoch
    // dataflow can *prove* the buffers are never simultaneously alive.
    const std::set<int> *live_epochs{nullptr};
  };

  // Result of a linear-scan arena packing.  Offsets contain the byte offset for
  // each constant-sized buffer, arena_size is the total constant footprint.
  struct ArenaPlan {
    size_t arena_size{0};
    std::unordered_map<const VarNode *, size_t> offsets;
    std::unordered_map<const VarNode *, std::vector<size_t>> segment_offsets;
  };

  struct SegmentBinding {
    int start{0};
    int end{0};
    size_t offset{0};
  };

  struct BranchScopeInfo {
    int begin{-1};
    int end{-1};
  };

  static size_t AlignUpSize(size_t value, size_t alignment) {
    if (alignment == 0) {
      return value;
    }
    size_t remainder = value % alignment;
    if (remainder == 0) {
      return value;
    }
    return value + (alignment - remainder);
  }

  ArenaPlan LinearScanPack(std::vector<Interval> intervals) {
    // Process intervals in program order so lifetimes correspond to the
    // linearised CFG.
    std::sort(intervals.begin(), intervals.end(),
              [](const Interval &lhs, const Interval &rhs) {
                if (lhs.start != rhs.start) {
                  return lhs.start < rhs.start;
                }
                if (lhs.size_bytes != rhs.size_bytes) {
                  return lhs.size_bytes > rhs.size_bytes;
                }
                return lhs.var->name_hint < rhs.var->name_hint;
              });

    size_t arena_top = 0;
    std::unordered_map<const VarNode *, size_t> offsets;
    std::unordered_map<const VarNode *, std::vector<size_t>> segment_offsets;

    for (const Interval &interval : intervals) {
      size_t offset = 0;
      bool found_slot = false;
      for (;;) {
        offset = AlignUpSize(offset, interval.alignment);
        bool overlaps = false;
        for (const Interval &other : intervals) {
          if (&other == &interval) {
            continue;
          }
          if (other.var == interval.var &&
              other.segment_id == interval.segment_id) {
            continue;
          }
          bool live_overlap =
              !(interval.end <= other.start || other.end <= interval.start);
          if (!live_overlap) {
            continue;
          }
          // Strict relaxer: if both intervals have a non-empty live-epoch
          // set and the sets are disjoint, the legacy 1-D linear-seq overlap
          // is spurious (two buffers happen to span the same syntactic range
          // but are never simultaneously *alive* under per-epoch dataflow).
          //
          // B2 gate: even when liveness proves disjointness, two buffers with
          // structurally distinct annotated layouts must NOT alias, since
          // downstream layout-driven access lowering can emit indexing
          // arithmetic that is only sound under each buffer's original
          // layout.  Skip the relaxer in that case to preserve the legacy
          // conflict.
          if (interval.live_epochs && other.live_epochs &&
              !interval.live_epochs->empty() && !other.live_epochs->empty() &&
              !LayoutAliasIncompatible(interval.var, other.var)) {
            const auto &la = *interval.live_epochs;
            const auto &lb = *other.live_epochs;
            auto ia = la.begin(), ib = lb.begin();
            bool intersects = false;
            while (ia != la.end() && ib != lb.end()) {
              if (*ia < *ib)
                ++ia;
              else if (*ib < *ia)
                ++ib;
              else {
                intersects = true;
                break;
              }
            }
            if (!intersects) {
              continue;
            }
          }
          auto other_it = segment_offsets.find(other.var);
          if (other_it == segment_offsets.end() ||
              other_it->second.size() <=
                  static_cast<size_t>(other.segment_id)) {
            continue;
          }
          size_t other_offset =
              other_it->second[static_cast<size_t>(other.segment_id)];
          bool mem_overlap = !(offset + interval.size_bytes <= other_offset ||
                               other_offset + other.size_bytes <= offset);
          if (mem_overlap) {
            overlaps = true;
            offset = other_offset + other.size_bytes;
            break;
          }
        }
        if (!overlaps) {
          found_slot = true;
          break;
        }
      }
      if (!found_slot) {
        offset = AlignUpSize(arena_top, interval.alignment);
      }
      arena_top = std::max(arena_top, offset + interval.size_bytes);
      if (interval.segment_id == 0) {
        offsets[interval.var] = offset;
      }
      auto &per_var_offsets = segment_offsets[interval.var];
      if (per_var_offsets.size() <= static_cast<size_t>(interval.segment_id)) {
        per_var_offsets.resize(static_cast<size_t>(interval.segment_id) + 1, 0);
      }
      per_var_offsets[static_cast<size_t>(interval.segment_id)] = offset;
    }

    return ArenaPlan{arena_top, std::move(offsets), std::move(segment_offsets)};
  }

  PrimExpr AlignPrimExpr(const PrimExpr &value, int alignment) const {
    if (alignment <= 1) {
      return value;
    }
    DataType dtype = value.dtype();
    ICHECK(dtype.is_int() || dtype.is_uint())
        << "Expected integer dtype for alignment, but got " << dtype;
    PrimExpr align_expr = make_const(dtype, alignment);
    PrimExpr adjust = make_const(dtype, alignment - 1);
    return indexdiv(value + adjust, align_expr) * align_expr;
  }

  runtime::StorageScope CurrentSummaryScope() const {
    return runtime::StorageScope::Create(is_dynamic_ ? "shared.dyn" : "shared");
  }

  SummaryAccessEntry
  MakeSyntheticAccess(const VarNode *var,
                      shared_access_analysis::AccessType type) const {
    SummaryAccessEntry access;
    access.buffer = tvm::ffi::GetRef<Var>(var);
    access.dtype = DataType::UInt(8);
    access.type = type;
    access.scope =
        runtime::StorageScope::Create(GetPtrStorageScope(access.buffer));
    access.is_pointer_access = true;
    return access;
  }

  std::vector<SummaryStmtEntry>
  ConvertToSummarySequence(const std::vector<StmtEntry> &seq) const {
    std::vector<SummaryStmtEntry> summary_seq;
    summary_seq.reserve(seq.size());
    for (const StmtEntry &entry : seq) {
      SummaryStmtEntry summary_entry;
      summary_entry.stmt = entry.stmt;
      if (entry.is_sync) {
        SummaryAccessEntry sync_access;
        sync_access.type = shared_access_analysis::kSync;
        sync_access.scope = CurrentSummaryScope();
        summary_entry.access.push_back(sync_access);
      }
      if (!entry.access.empty()) {
        summary_entry.access.insert(summary_entry.access.end(),
                                    entry.access.begin(), entry.access.end());
      } else {
        for (const VarNode *var : entry.read_touched) {
          summary_entry.access.push_back(
              MakeSyntheticAccess(var, shared_access_analysis::kRead));
        }
        for (const VarNode *var : entry.write_touched) {
          summary_entry.access.push_back(
              MakeSyntheticAccess(var, shared_access_analysis::kWrite));
        }
      }
      summary_seq.push_back(std::move(summary_entry));
    }
    return summary_seq;
  }

  std::vector<SummaryStmtEntry>
  ConvertToSummarySequenceRange(const std::vector<StmtEntry> &seq, size_t begin,
                                size_t end) const {
    if (begin >= end || begin >= seq.size()) {
      return {};
    }
    end = std::min(end, seq.size());
    std::vector<StmtEntry> slice;
    slice.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
      slice.push_back(seq[i]);
    }
    return ConvertToSummarySequence(slice);
  }

  std::string SummaryAccessToString(const SummaryAccessEntry &access) const {
    std::ostringstream os;
    switch (access.type) {
    case shared_access_analysis::kRead:
      os << "R:";
      break;
    case shared_access_analysis::kWrite:
      os << "W:";
      break;
    case shared_access_analysis::kSync:
      os << "SYNC:" << access.scope.to_string();
      return os.str();
    case shared_access_analysis::kAlloc:
      os << "ALLOC:";
      break;
    case shared_access_analysis::kReadAcquire:
      os << "RACQ:";
      break;
    }
    if (access.buffer.defined()) {
      os << access.buffer->name_hint;
    } else {
      os << "<anon>";
    }
    return os.str();
  }

  std::vector<std::string> AccessSetToSortedStrings(
      const std::vector<SummaryAccessEntry> &accesses) const {
    std::vector<std::string> result;
    result.reserve(accesses.size());
    for (const SummaryAccessEntry &access : accesses) {
      result.push_back(SummaryAccessToString(access));
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
  }

  std::vector<std::string> BufferSetToSortedStrings(
      const std::vector<SummaryAccessEntry> &accesses) const {
    std::vector<std::string> result;
    result.reserve(accesses.size());
    for (const SummaryAccessEntry &access : accesses) {
      if (access.type == shared_access_analysis::kSync ||
          !access.buffer.defined()) {
        continue;
      }
      result.push_back(access.buffer->name_hint);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
  }

  std::vector<std::string>
  VectorDiff(const std::vector<std::string> &lhs,
             const std::vector<std::string> &rhs) const {
    std::vector<std::string> result;
    std::set_difference(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                        std::back_inserter(result));
    return result;
  }

  std::string JoinStrings(const std::vector<std::string> &items) const {
    if (items.empty()) {
      return "<empty>";
    }
    std::ostringstream os;
    for (size_t i = 0; i < items.size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << items[i];
    }
    return os.str();
  }

  std::string VarVecToString(const std::vector<const VarNode *> &vars) const {
    std::vector<std::string> names;
    names.reserve(vars.size());
    for (const VarNode *var : vars) {
      names.push_back(var->name_hint);
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    return JoinStrings(names);
  }

  bool ContainsBuffer(const std::vector<const VarNode *> &vars,
                      const std::string &buffer) const {
    return std::any_of(vars.begin(), vars.end(), [&](const VarNode *var) {
      return var->name_hint == buffer;
    });
  }

  struct BoundaryDeltaRecord {
    size_t boundary{0};
    std::vector<std::string> post_buffers;
    std::vector<std::string> closed_buffers;
    std::vector<std::string> persistent_buffers;
    std::vector<std::string> new_buffers;
    const Object *left_stmt{nullptr};
    const Object *right_stmt{nullptr};
  };

  enum class BufferPhaseClass {
    kAnchor,
    kReadMostlyAnchor,
    kDerivedScratch,
    kPhaseLocalScratch,
    kPhaseScopedState,
    kLocalWorkingSet,
  };

  enum class BufferSplitPolicy {
    kNeverSplitAcrossLoop,
    kSplitAtStableClosure,
    kSplitWithLoopAwareSummary,
  };

  struct BufferLifetimePlanSeed {
    std::string buffer;
    std::vector<std::string> stable_closures;
    std::vector<std::string> unstable_closures;
    std::vector<std::string> carry_over_boundaries;
    std::vector<std::string> loop_local_boundaries;
    std::vector<std::string> loop_carried_boundaries;
    std::vector<std::string> loop_middle_boundaries;
    int read_stmt_count{0};
    int write_stmt_count{0};
    int gen_count{0};
    int kill_count{0};
    BufferPhaseClass inferred_phase_class{BufferPhaseClass::kLocalWorkingSet};
    BufferSplitPolicy suggested_split_policy{
        BufferSplitPolicy::kSplitAtStableClosure};
    bool reclaim_root{false};
    bool requires_loop_aware_summary{false};
  };

  struct BufferAccessStats {
    std::string buffer;
    int read_stmt_count{0};
    int write_stmt_count{0};
    int gen_count{0};
    int kill_count{0};
  };

  struct LoopBoundarySummary {
    std::vector<std::string> loop_local_buffers;
    std::vector<std::string> loop_carried_buffers;
  };

  struct LoopLifetimeSummary {
    std::vector<std::string> head_buffers;
    std::vector<std::string> tail_buffers;
    std::vector<std::string> loop_local_buffers;
    std::vector<std::string> loop_carried_buffers;
    std::vector<std::string> middle_buffers;
  };

  struct BufferLocalSegment {
    std::string buffer;
    int start_stmt{-1};
    int end_stmt{-1};
    std::string start_kind;
    std::string end_kind;
  };

  struct BufferCarryEdge {
    std::string buffer;
    int src_stmt{-1};
    int dst_stmt{-1};
    int min_iteration_distance{1};
    std::string kind;
  };

  struct BufferSegmentRelation {
    std::string buffer;
    int from_segment{-1};
    int to_segment{-1};
    int gap_start{-1};
    int gap_end{-1};
    std::string relation;
    bool overwrite_safe{false};
    bool value_live_across_gap{false};
    std::string gap_value_flow;
    std::string gap_semantics;
    bool subregion_live_across_gap{false};
    int last_lhs_access_stmt{-1};
    std::string last_lhs_access_kind;
    std::string last_lhs_access_expr;
    std::string last_lhs_access_footprint;
    int first_gap_access_stmt{-1};
    std::string first_gap_access_kind;
    int first_rhs_access_stmt{-1};
    std::string first_rhs_access_kind;
    int first_post_gap_read_stmt{-1};
    int first_post_gap_write_stmt{-1};
    int first_rhs_read_stmt{-1};
    int first_rhs_write_stmt{-1};
    std::string first_rhs_read_expr;
    std::string first_rhs_write_expr;
    std::string first_rhs_read_footprint;
    std::string first_rhs_write_footprint;
    std::string gap_region_signal;
  };

  struct BufferAccessObservation {
    int stmt{-1};
    int access_index{-1};
    bool has_read{false};
    bool has_write{false};
  };

  int AccessWitnessScore(
      const shared_access_analysis::AccessEntry &access) const {
    if (!access.buffer_indices.empty()) {
      return 3;
    }
    if (!access.buffer_ranges.empty()) {
      return 2;
    }
    if (!access.touched.empty()) {
      return 1;
    }
    return 0;
  }

  std::optional<int> SelectPreferredAccessIndex(
      const StmtEntry &entry, const std::string &buffer,
      std::optional<shared_access_analysis::AccessType> type = std::nullopt,
      bool prefer_last = false) const {
    std::optional<int> best_index;
    int best_score = -1;
    for (size_t i = 0; i < entry.access.size(); ++i) {
      const auto &access = entry.access[i];
      if (!AccessMatchesBuffer(access, buffer)) {
        continue;
      }
      if (type.has_value() && access.type != type.value()) {
        continue;
      }
      int score = AccessWitnessScore(access);
      if (!best_index.has_value() || score > best_score ||
          (score == best_score &&
           ((prefer_last && static_cast<int>(i) > best_index.value()) ||
            (!prefer_last && static_cast<int>(i) < best_index.value())))) {
        best_index = static_cast<int>(i);
        best_score = score;
      }
    }
    return best_index;
  }

  struct BufferSemanticSegment {
    std::string buffer;
    int semantic_segment{-1};
    int start_stmt{-1};
    int end_stmt{-1};
    int from_structural_segment{-1};
    int to_structural_segment{-1};
  };

  struct BufferSemanticGap {
    std::string buffer;
    int from_semantic_segment{-1};
    int to_semantic_segment{-1};
    int gap_start{-1};
    int gap_end{-1};
    std::string structural_relation;
    std::string gap_semantics;
    bool overwrite_safe{false};
  };

  struct BufferPlannerSegment {
    std::string buffer;
    int planner_segment{-1};
    int start_stmt{-1};
    int end_stmt{-1};
    int from_structural_segment{-1};
    int to_structural_segment{-1};
  };

  enum class GapValueFlow {
    kNoReaccess,
    kReadBeforeWrite,
    kWriteBeforeRead,
    kReadWriteSameStmt,
  };

  std::string GapValueFlowToString(GapValueFlow flow) const {
    switch (flow) {
    case GapValueFlow::kNoReaccess:
      return "no_reaccess";
    case GapValueFlow::kReadBeforeWrite:
      return "read_before_write";
    case GapValueFlow::kWriteBeforeRead:
      return "write_before_read";
    case GapValueFlow::kReadWriteSameStmt:
      return "read_write_same_stmt";
    }
    return "unknown";
  }

  std::string GapSemanticsFromValueFlow(GapValueFlow flow) const {
    switch (flow) {
    case GapValueFlow::kNoReaccess:
      return "no_reaccess_after_gap";
    case GapValueFlow::kReadBeforeWrite:
      return "value_live_across_gap";
    case GapValueFlow::kWriteBeforeRead:
      return "overwrite_before_read";
    case GapValueFlow::kReadWriteSameStmt:
      return "mixed_same_stmt";
    }
    return "unknown";
  }

  bool IsSubregionFootprint(const std::string &footprint) const {
    return footprint.rfind("scalar(", 0) == 0 ||
           footprint.rfind("ramp(", 0) == 0 ||
           footprint.rfind("multi_dim(", 0) == 0 ||
           footprint.rfind("multi_dim_range(", 0) == 0;
  }

  std::string AccessObservationKind(bool has_read, bool has_write) const {
    if (has_read && has_write) {
      return "read_write";
    }
    if (has_read) {
      return "read";
    }
    if (has_write) {
      return "write";
    }
    return "none";
  }

  std::string PrintPrimExpr(const PrimExpr &expr) const {
    if (!expr.defined()) {
      return "none";
    }
    std::ostringstream os;
    os << expr;
    return os.str();
  }

  std::string DescribeBufferAccessExpr(const Object *stmt,
                                       const std::string &buffer,
                                       bool want_read) const {
    if (stmt == nullptr) {
      return "none";
    }

    class BufferAccessExprExtractor final : public StmtExprVisitor {
    public:
      BufferAccessExprExtractor(const std::string &buffer, bool want_read)
          : buffer_(buffer), want_read_(want_read) {}

      std::optional<std::string> Result() const { return result_; }

      void VisitExpr_(const BufferLoadNode *op) final {
        if (!result_.has_value() && want_read_ &&
            op->buffer->data->name_hint == buffer_) {
          std::ostringstream os;
          os << "load(" << op->buffer->name << ", [";
          for (size_t i = 0; i < op->indices.size(); ++i) {
            if (i != 0) {
              os << ", ";
            }
            os << op->indices[i];
          }
          os << "])";
          result_ = os.str();
        }
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitStmt_(const BufferStoreNode *op) final {
        if (!result_.has_value() && !want_read_ &&
            op->buffer->data->name_hint == buffer_) {
          std::ostringstream os;
          os << "store(" << op->buffer->name << ", [";
          for (size_t i = 0; i < op->indices.size(); ++i) {
            if (i != 0) {
              os << ", ";
            }
            os << op->indices[i];
          }
          os << "])";
          result_ = os.str();
        }
        StmtExprVisitor::VisitStmt_(op);
      }

      void VisitExpr_(const CallNode *op) final {
        if (!result_.has_value() && op->op.same_as(builtin::tvm_access_ptr()) &&
            op->args.size() == 5) {
          const VarNode *buffer_var = op->args[1].as<VarNode>();
          const IntImmNode *flag = op->args[4].as<IntImmNode>();
          bool has_read = flag != nullptr && (flag->value & 1);
          bool has_write = flag != nullptr && (flag->value & 2);
          if (buffer_var != nullptr && buffer_var->name_hint == buffer_ &&
              ((want_read_ && has_read) || (!want_read_ && has_write))) {
            std::ostringstream os;
            os << "access_ptr(" << buffer_var->name_hint
               << ", offset=" << op->args[2] << ", extent=" << op->args[3]
               << ", rw=" << (want_read_ ? "read" : "write") << ")";
            result_ = os.str();
          }
        }
        StmtExprVisitor::VisitExpr_(op);
      }

    private:
      std::string buffer_;
      bool want_read_{false};
      std::optional<std::string> result_;
    };

    BufferAccessExprExtractor extractor(buffer, want_read);
    if (stmt->IsInstance<StmtNode>()) {
      extractor(tvm::ffi::GetRef<Stmt>(static_cast<const StmtNode *>(stmt)));
    } else if (stmt->IsInstance<PrimExprNode>()) {
      extractor(
          tvm::ffi::GetRef<PrimExpr>(static_cast<const PrimExprNode *>(stmt)));
    }
    return extractor.Result().value_or("none");
  }

  std::string DescribeIndexFootprint(const Array<PrimExpr> &indices) const {
    if (indices.empty()) {
      return "none";
    }
    if (indices.size() == 1) {
      const PrimExpr &index = indices[0];
      if (const auto *ramp = index.as<RampNode>()) {
        std::ostringstream os;
        os << "ramp(base=" << ramp->base << ", stride=" << ramp->stride
           << ", lanes=" << ramp->lanes << ")";
        return os.str();
      }
      std::ostringstream os;
      os << "scalar(" << index << ")";
      return os.str();
    }
    std::ostringstream os;
    os << "multi_dim(";
    for (size_t i = 0; i < indices.size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      if (const auto *ramp = indices[i].as<RampNode>()) {
        os << "ramp(base=" << ramp->base << ", stride=" << ramp->stride
           << ", lanes=" << ramp->lanes << ")";
      } else {
        os << indices[i];
      }
    }
    os << ")";
    return os.str();
  }

  std::string DescribeRangeFootprint(const Array<Range> &ranges) const {
    if (ranges.empty()) {
      return "none";
    }
    if (ranges.size() == 1) {
      return DescribeLinearFootprint(ranges[0]->min, ranges[0]->extent);
    }
    std::ostringstream os;
    os << "multi_dim_range(";
    for (size_t i = 0; i < ranges.size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << "min=" << ranges[i]->min << ", extent=" << ranges[i]->extent;
    }
    os << ")";
    return os.str();
  }

  std::string AccessBufferName(const SummaryAccessEntry &access) const {
    if (access.buffer_name.defined()) {
      return access.buffer_name->data->name_hint;
    }
    if (access.buffer.defined()) {
      return access.buffer->name_hint;
    }
    return "<anon>";
  }

  bool AccessMatchesBuffer(const SummaryAccessEntry &access,
                           const std::string &buffer) const {
    if (access.buffer_name.defined()) {
      return access.buffer_name->data->name_hint == buffer;
    }
    return access.buffer.defined() && access.buffer->name_hint == buffer;
  }

  std::string DescribeAccessExpr(const SummaryAccessEntry &access) const {
    std::ostringstream os;
    std::string buffer = AccessBufferName(access);
    if (!access.buffer_indices.empty()) {
      os << (access.type == shared_access_analysis::kWrite ? "store(" : "load(")
         << buffer << ", [";
      for (size_t i = 0; i < access.buffer_indices.size(); ++i) {
        if (i != 0) {
          os << ", ";
        }
        os << access.buffer_indices[i];
      }
      os << "])";
      return os.str();
    }
    if (!access.buffer_ranges.empty()) {
      os << "access_ptr(" << buffer << ", "
         << DescribeRangeFootprint(access.buffer_ranges) << ", rw="
         << (access.type == shared_access_analysis::kWrite ? "write" : "read")
         << ")";
      return os.str();
    }
    if (!access.touched.empty()) {
      os << "access_ptr(" << buffer << ", touched=" << access.touched[0]
         << ", rw="
         << (access.type == shared_access_analysis::kWrite ? "write" : "read")
         << ")";
      return os.str();
    }
    return "none";
  }

  std::string DescribeAccessFootprint(const SummaryAccessEntry &access) const {
    if (!access.buffer_indices.empty()) {
      return DescribeIndexFootprint(access.buffer_indices);
    }
    if (!access.buffer_ranges.empty()) {
      return DescribeRangeFootprint(access.buffer_ranges);
    }
    if (!access.touched.empty()) {
      std::ostringstream os;
      os << access.touched[0];
      return os.str();
    }
    return "none";
  }

  std::string DescribeLinearFootprint(const PrimExpr &offset,
                                      const PrimExpr &extent) const {
    std::ostringstream os;
    os << "range(offset=" << offset << ", extent=" << extent << ")";
    return os.str();
  }

  std::string DescribeBufferAccessFootprint(const Object *stmt,
                                            const std::string &buffer,
                                            bool want_read) const {
    if (stmt == nullptr) {
      return "none";
    }

    class BufferAccessFootprintExtractor final : public StmtExprVisitor {
    public:
      BufferAccessFootprintExtractor(const SharedMemoryRewriter *planner,
                                     const std::string &buffer, bool want_read)
          : planner_(planner), buffer_(buffer), want_read_(want_read) {}

      std::optional<std::string> Result() const { return result_; }

      void VisitExpr_(const BufferLoadNode *op) final {
        if (!result_.has_value() && want_read_ &&
            op->buffer->data->name_hint == buffer_) {
          result_ = planner_->DescribeIndexFootprint(op->indices);
        }
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitStmt_(const BufferStoreNode *op) final {
        if (!result_.has_value() && !want_read_ &&
            op->buffer->data->name_hint == buffer_) {
          result_ = planner_->DescribeIndexFootprint(op->indices);
        }
        StmtExprVisitor::VisitStmt_(op);
      }

      void VisitExpr_(const CallNode *op) final {
        if (!result_.has_value() && op->op.same_as(builtin::tvm_access_ptr()) &&
            op->args.size() == 5) {
          const VarNode *buffer_var = op->args[1].as<VarNode>();
          const IntImmNode *flag = op->args[4].as<IntImmNode>();
          bool has_read = flag != nullptr && (flag->value & 1);
          bool has_write = flag != nullptr && (flag->value & 2);
          if (buffer_var != nullptr && buffer_var->name_hint == buffer_ &&
              ((want_read_ && has_read) || (!want_read_ && has_write))) {
            result_ =
                planner_->DescribeLinearFootprint(op->args[2], op->args[3]);
          }
        }
        StmtExprVisitor::VisitExpr_(op);
      }

    private:
      const SharedMemoryRewriter *planner_;
      std::string buffer_;
      bool want_read_{false};
      std::optional<std::string> result_;
    };

    BufferAccessFootprintExtractor extractor(this, buffer, want_read);
    if (stmt->IsInstance<StmtNode>()) {
      extractor(tvm::ffi::GetRef<Stmt>(static_cast<const StmtNode *>(stmt)));
    } else if (stmt->IsInstance<PrimExprNode>()) {
      extractor(
          tvm::ffi::GetRef<PrimExpr>(static_cast<const PrimExprNode *>(stmt)));
    }
    return extractor.Result().value_or("none");
  }

  struct CutpointNode {
    int id{-1};
    std::string stmt_type;
    int scope_level{-1};
  };

  struct ExecutionEdge {
    int src_cutpoint{-1};
    int dst_cutpoint{-1};
    std::string kind;
    std::vector<std::string> reads;
    std::vector<std::string> writes;
  };

  struct CompressedCutpoint {
    int id{-1};
    int start_stmt{-1};
    int end_stmt{-1};
    std::string reason;
    std::vector<std::string> changed_buffers;
    std::vector<std::string> born_buffers;
    std::vector<std::string> closed_buffers;
    std::vector<std::string> persistent_buffers;
  };

  std::string BufferPhaseClassToString(BufferPhaseClass phase_class) const {
    switch (phase_class) {
    case BufferPhaseClass::kAnchor:
      return "anchor";
    case BufferPhaseClass::kReadMostlyAnchor:
      return "read_mostly_anchor";
    case BufferPhaseClass::kDerivedScratch:
      return "derived_scratch";
    case BufferPhaseClass::kPhaseLocalScratch:
      return "phase_local_scratch";
    case BufferPhaseClass::kPhaseScopedState:
      return "phase_scoped_state";
    case BufferPhaseClass::kLocalWorkingSet:
      return "local_working_set";
    }
    return "unknown";
  }

  std::string BufferSplitPolicyToString(BufferSplitPolicy split_policy) const {
    switch (split_policy) {
    case BufferSplitPolicy::kNeverSplitAcrossLoop:
      return "never_split_across_loop";
    case BufferSplitPolicy::kSplitAtStableClosure:
      return "split_at_stable_closure";
    case BufferSplitPolicy::kSplitWithLoopAwareSummary:
      return "split_with_loop_aware_summary";
    }
    return "unknown";
  }

  BufferPhaseClass InferBufferPhaseClass(const BufferLifetimePlanSeed &seed,
                                         size_t total_boundaries) const {
    size_t carry_count = seed.carry_over_boundaries.size();
    size_t stable_count = seed.stable_closures.size();
    size_t unstable_count = seed.unstable_closures.size();

    size_t anchor_threshold = std::max<size_t>(4, total_boundaries / 2);
    bool single_birth = seed.gen_count <= 1;
    bool read_mostly = seed.read_stmt_count > 0 && seed.write_stmt_count <= 1;
    bool repeated_working_writes =
        seed.write_stmt_count > 1 || seed.gen_count > 1;
    bool tiny_carry = carry_count <= 1;
    bool derived_single_write =
        seed.write_stmt_count == 1 && seed.gen_count == 1;

    if (carry_count >= anchor_threshold && unstable_count == 0) {
      return BufferPhaseClass::kAnchor;
    }
    if (derived_single_write && tiny_carry && stable_count > 0) {
      return BufferPhaseClass::kDerivedScratch;
    }
    if (single_birth && read_mostly &&
        (carry_count >= 2 || unstable_count > 0 || stable_count > 0)) {
      return BufferPhaseClass::kReadMostlyAnchor;
    }
    if (carry_count <= 1 && unstable_count == 0 && stable_count > 0) {
      return BufferPhaseClass::kPhaseLocalScratch;
    }
    if (repeated_working_writes &&
        (carry_count > 0 || unstable_count > 0 || stable_count > 0)) {
      return BufferPhaseClass::kPhaseScopedState;
    }
    return BufferPhaseClass::kLocalWorkingSet;
  }

  BufferSplitPolicy
  SuggestSplitPolicy(const BufferLifetimePlanSeed &seed) const {
    switch (seed.inferred_phase_class) {
    case BufferPhaseClass::kAnchor:
    case BufferPhaseClass::kReadMostlyAnchor:
      return BufferSplitPolicy::kNeverSplitAcrossLoop;
    case BufferPhaseClass::kDerivedScratch:
    case BufferPhaseClass::kPhaseLocalScratch:
      return BufferSplitPolicy::kSplitAtStableClosure;
    case BufferPhaseClass::kPhaseScopedState:
      return BufferSplitPolicy::kSplitWithLoopAwareSummary;
    case BufferPhaseClass::kLocalWorkingSet:
      return BufferSplitPolicy::kSplitAtStableClosure;
    }
    return BufferSplitPolicy::kSplitAtStableClosure;
  }

  std::vector<BufferLifetimePlanSeed> BuildLifetimePlanSeeds(
      const std::vector<BoundaryDeltaRecord> &records,
      const std::unordered_map<std::string, BufferAccessStats>
          &access_stats_by_buffer,
      const std::unordered_map<size_t, LoopBoundarySummary> &loop_summaries)
      const {
    std::vector<BufferLifetimePlanSeed> seeds;
    if (records.empty()) {
      return seeds;
    }

    std::unordered_set<std::string> buffer_names;
    for (const BoundaryDeltaRecord &record : records) {
      buffer_names.insert(record.post_buffers.begin(),
                          record.post_buffers.end());
      buffer_names.insert(record.closed_buffers.begin(),
                          record.closed_buffers.end());
      buffer_names.insert(record.persistent_buffers.begin(),
                          record.persistent_buffers.end());
      buffer_names.insert(record.new_buffers.begin(), record.new_buffers.end());
    }

    std::vector<std::string> sorted_buffers(buffer_names.begin(),
                                            buffer_names.end());
    std::sort(sorted_buffers.begin(), sorted_buffers.end());

    auto contains = [](const std::vector<std::string> &items,
                       const std::string &value) {
      return std::binary_search(items.begin(), items.end(), value);
    };

    for (const std::string &buffer : sorted_buffers) {
      BufferLifetimePlanSeed seed;
      seed.buffer = buffer;
      if (auto stats_it = access_stats_by_buffer.find(buffer);
          stats_it != access_stats_by_buffer.end()) {
        seed.read_stmt_count = stats_it->second.read_stmt_count;
        seed.write_stmt_count = stats_it->second.write_stmt_count;
        seed.gen_count = stats_it->second.gen_count;
        seed.kill_count = stats_it->second.kill_count;
      }

      for (size_t i = 0; i < records.size(); ++i) {
        const BoundaryDeltaRecord &record = records[i];
        if (contains(record.persistent_buffers, buffer)) {
          seed.carry_over_boundaries.push_back(std::to_string(record.boundary));
        }
        if (auto loop_it = loop_summaries.find(record.boundary);
            loop_it != loop_summaries.end()) {
          if (contains(loop_it->second.loop_local_buffers, buffer)) {
            seed.loop_local_boundaries.push_back(
                std::to_string(record.boundary));
          }
          if (contains(loop_it->second.loop_carried_buffers, buffer)) {
            seed.loop_carried_boundaries.push_back(
                std::to_string(record.boundary));
          }
          std::vector<std::string> middle_only =
              VectorDiff(loop_it->second.loop_carried_buffers,
                         loop_it->second.loop_local_buffers);
          if (contains(middle_only, buffer)) {
            seed.loop_middle_boundaries.push_back(
                std::to_string(record.boundary));
          }
        }
        if (!contains(record.closed_buffers, buffer)) {
          continue;
        }

        std::optional<size_t> reappears_at;
        for (size_t j = i + 1; j < records.size(); ++j) {
          if (contains(records[j].new_buffers, buffer) ||
              contains(records[j].persistent_buffers, buffer) ||
              contains(records[j].post_buffers, buffer)) {
            reappears_at = records[j].boundary;
            break;
          }
        }

        if (reappears_at.has_value()) {
          seed.unstable_closures.push_back(
              std::to_string(record.boundary) + "->" +
              std::to_string(reappears_at.value()));
        } else {
          seed.stable_closures.push_back(std::to_string(record.boundary));
        }
      }

      seed.inferred_phase_class = InferBufferPhaseClass(seed, records.size());
      seed.suggested_split_policy = SuggestSplitPolicy(seed);
      seed.reclaim_root =
          seed.inferred_phase_class == BufferPhaseClass::kPhaseLocalScratch ||
          seed.inferred_phase_class == BufferPhaseClass::kDerivedScratch;
      seed.requires_loop_aware_summary =
          seed.suggested_split_policy ==
              BufferSplitPolicy::kSplitWithLoopAwareSummary ||
          !seed.loop_carried_boundaries.empty();
      seeds.push_back(std::move(seed));
    }

    return seeds;
  }

  std::unordered_map<std::string, BufferAccessStats>
  BuildBufferAccessStats(const std::vector<StmtEntry> &seq) const {
    std::unordered_map<std::string, BufferAccessStats> stats_by_buffer;

    auto touch_stats = [&](const std::vector<const VarNode *> &vars,
                           bool is_write) {
      for (const VarNode *var : vars) {
        BufferAccessStats &stats = stats_by_buffer[var->name_hint];
        stats.buffer = var->name_hint;
        if (is_write) {
          ++stats.write_stmt_count;
        } else {
          ++stats.read_stmt_count;
        }
      }
    };

    for (const StmtEntry &entry : seq) {
      touch_stats(entry.read_touched, false);
      touch_stats(entry.write_touched, true);

      auto event_it = event_map_.find(entry.stmt);
      if (event_it == event_map_.end()) {
        continue;
      }
      for (const VarNode *var : event_it->second.gen) {
        BufferAccessStats &stats = stats_by_buffer[var->name_hint];
        stats.buffer = var->name_hint;
        ++stats.gen_count;
      }
      for (const VarNode *var : event_it->second.kill) {
        BufferAccessStats &stats = stats_by_buffer[var->name_hint];
        stats.buffer = var->name_hint;
        ++stats.kill_count;
      }
    }

    return stats_by_buffer;
  }

  LoopLifetimeSummary AnalyzeLoopLifetime(const std::vector<StmtEntry> &seq,
                                          size_t loop_begin,
                                          size_t loop_end) const {
    LoopLifetimeSummary result;
    if (loop_end <= loop_begin + 1 || loop_end > seq.size()) {
      return result;
    }

    const auto *loop = static_cast<const ForNode *>(seq[loop_begin].stmt);
    std::vector<SummaryStmtEntry> loop_body =
        ConvertToSummarySequenceRange(seq, loop_begin + 1, loop_end);
    if (loop_body.empty()) {
      return result;
    }

    size_t middle_index = loop_body.size() / 2;
    std::vector<SummaryStmtEntry> head_seq{loop_body.front()};
    std::vector<SummaryStmtEntry> middle_seq;
    middle_seq.reserve(middle_index + 1);
    for (size_t i = 0; i <= middle_index && i < loop_body.size(); ++i) {
      middle_seq.push_back(loop_body[i]);
    }
    auto head_summary = shared_access_analysis::SummarizeAccessSequence(
        head_seq, nullptr, CurrentSummaryScope(), ffi::Array<tir::IterVar>(),
        ConstrSet{}, {}, false);
    auto middle_summary = shared_access_analysis::SummarizeAccessSequence(
        middle_seq, nullptr, CurrentSummaryScope(), ffi::Array<tir::IterVar>(),
        ConstrSet{}, {}, false);
    auto tail_summary = shared_access_analysis::SummarizeAccessSequence(
        loop_body, loop, CurrentSummaryScope(), ffi::Array<tir::IterVar>(),
        ConstrSet{}, {}, false);

    result.head_buffers =
        BufferSetToSortedStrings(head_summary.exposed_accesses);
    result.middle_buffers =
        BufferSetToSortedStrings(middle_summary.exposed_accesses);
    result.tail_buffers =
        BufferSetToSortedStrings(tail_summary.exposed_accesses);
    result.loop_carried_buffers = result.tail_buffers;
    result.loop_local_buffers =
        VectorDiff(result.middle_buffers, result.tail_buffers);
    return result;
  }

  std::unordered_map<size_t, LoopBoundarySummary>
  BuildLoopBoundarySummaries(const std::vector<StmtEntry> &seq) const {
    std::unordered_map<size_t, LoopBoundarySummary> loop_summaries;
    for (size_t i = 0; i < seq.size(); ++i) {
      int64_t offset = seq[i].scope_pair_offset;
      if (offset <= 0) {
        continue;
      }

      const Object *stmt = seq[i].stmt;
      if (stmt == nullptr || !stmt->IsInstance<ForNode>()) {
        continue;
      }

      size_t end_index = static_cast<size_t>(static_cast<int64_t>(i) + offset);
      if (end_index <= i + 1 || end_index > seq.size()) {
        continue;
      }

      LoopLifetimeSummary loop_summary = AnalyzeLoopLifetime(seq, i, end_index);
      if (loop_summary.head_buffers.empty() &&
          loop_summary.tail_buffers.empty()) {
        continue;
      }
      loop_summaries.emplace(
          i, LoopBoundarySummary{loop_summary.loop_local_buffers,
                                 loop_summary.loop_carried_buffers});
    }
    return loop_summaries;
  }

  std::unordered_map<const Object *, int>
  BuildStmtTimeline(const std::vector<StmtEntry> &seq) const {
    std::unordered_map<const Object *, int> timeline;
    for (size_t i = 0; i < seq.size(); ++i) {
      timeline[seq[i].stmt] = static_cast<int>(i);
    }
    return timeline;
  }

  std::vector<CutpointNode> BuildCutpointNodes(
      const std::vector<StmtEntry> &seq,
      const std::unordered_map<const Object *, StmtAttr> &stmt_attrs) const {
    std::vector<CutpointNode> nodes;
    nodes.reserve(seq.size());
    for (size_t i = 0; i < seq.size(); ++i) {
      int scope_level = -1;
      if (auto attr_it = stmt_attrs.find(seq[i].stmt);
          attr_it != stmt_attrs.end()) {
        scope_level = static_cast<int>(attr_it->second.level);
      }
      nodes.push_back(CutpointNode{static_cast<int>(i),
                                   seq[i].stmt->GetTypeKey(), scope_level});
    }
    return nodes;
  }

  std::string InferEdgeKind(const StmtEntry &src, const StmtEntry &dst) const {
    if (src.is_sync || dst.is_sync) {
      return "sync_edge";
    }
    if (src.stmt->IsInstance<ForNode>() || dst.stmt->IsInstance<ForNode>()) {
      return "loop_edge";
    }
    if (src.stmt->IsInstance<IfThenElseNode>() ||
        dst.stmt->IsInstance<IfThenElseNode>()) {
      return "branch_edge";
    }
    return "seq_edge";
  }

  std::vector<ExecutionEdge>
  BuildExecutionEdges(const std::vector<StmtEntry> &seq) const {
    std::vector<ExecutionEdge> edges;
    if (seq.size() < 2) {
      return edges;
    }
    edges.reserve(seq.size() - 1);
    for (size_t i = 0; i + 1 < seq.size(); ++i) {
      std::vector<std::string> reads;
      std::vector<std::string> writes;
      for (const VarNode *var : seq[i + 1].read_touched) {
        reads.push_back(var->name_hint);
      }
      for (const VarNode *var : seq[i + 1].write_touched) {
        writes.push_back(var->name_hint);
      }
      std::sort(reads.begin(), reads.end());
      reads.erase(std::unique(reads.begin(), reads.end()), reads.end());
      std::sort(writes.begin(), writes.end());
      writes.erase(std::unique(writes.begin(), writes.end()), writes.end());
      edges.push_back(
          ExecutionEdge{static_cast<int>(i), static_cast<int>(i + 1),
                        InferEdgeKind(seq[i], seq[i + 1]), reads, writes});
    }
    return edges;
  }

  void LogCutpointGraph(
      const std::vector<StmtEntry> &seq,
      const std::unordered_map<const Object *, StmtAttr> &stmt_attrs) const {
    std::vector<CutpointNode> nodes = BuildCutpointNodes(seq, stmt_attrs);
    std::vector<ExecutionEdge> edges = BuildExecutionEdges(seq);

    for (const CutpointNode &node : nodes) {
      std::ostringstream os;
      os << "[MergeSharedCutpoint] id=" << node.id << "\n";
      os << "  stmt_type: " << node.stmt_type << "\n";
      os << "  scope_level: " << node.scope_level;
      LOG(INFO) << os.str();
    }

    for (const ExecutionEdge &edge : edges) {
      std::ostringstream os;
      os << "[MergeSharedExecEdge] src=" << edge.src_cutpoint << "\n";
      os << "  dst: " << edge.dst_cutpoint << "\n";
      os << "  kind: " << edge.kind << "\n";
      os << "  reads: " << JoinStrings(edge.reads) << "\n";
      os << "  writes: " << JoinStrings(edge.writes);
      LOG(INFO) << os.str();
    }
  }

  std::vector<CompressedCutpoint> BuildCompressedCutpoints(
      const std::vector<BoundaryDeltaRecord> &records) const {
    std::vector<CompressedCutpoint> cutpoints;
    for (const BoundaryDeltaRecord &record : records) {
      std::vector<std::string> changed_buffers = record.closed_buffers;
      changed_buffers.insert(changed_buffers.end(), record.new_buffers.begin(),
                             record.new_buffers.end());
      changed_buffers.insert(changed_buffers.end(),
                             record.persistent_buffers.begin(),
                             record.persistent_buffers.end());
      std::sort(changed_buffers.begin(), changed_buffers.end());
      changed_buffers.erase(
          std::unique(changed_buffers.begin(), changed_buffers.end()),
          changed_buffers.end());
      if (changed_buffers.empty()) {
        continue;
      }

      std::string reason = "state_change";
      if (!record.closed_buffers.empty() && record.new_buffers.empty()) {
        reason = "closure";
      } else if (record.closed_buffers.empty() && !record.new_buffers.empty()) {
        reason = "birth";
      } else if (!record.closed_buffers.empty() &&
                 !record.new_buffers.empty()) {
        reason = "transition";
      }

      cutpoints.push_back(CompressedCutpoint{
          static_cast<int>(cutpoints.size()), static_cast<int>(record.boundary),
          static_cast<int>(record.boundary + 1), reason, changed_buffers,
          record.new_buffers, record.closed_buffers,
          record.persistent_buffers});
    }
    return cutpoints;
  }

  std::string DescribeStmtCutpointContext(
      int stmt, const std::vector<CompressedCutpoint> &cutpoints) const {
    if (stmt < 0) {
      return "none";
    }

    const CompressedCutpoint *prev = nullptr;
    const CompressedCutpoint *next = nullptr;
    for (const CompressedCutpoint &cutpoint : cutpoints) {
      if (cutpoint.end_stmt <= stmt) {
        prev = &cutpoint;
      }
      if (next == nullptr && cutpoint.start_stmt >= stmt) {
        next = &cutpoint;
      }
    }

    auto describe_cutpoint = [&](const CompressedCutpoint *cutpoint) {
      if (cutpoint == nullptr) {
        return std::string("none");
      }
      std::ostringstream os;
      os << "cp" << cutpoint->id << ":" << cutpoint->reason << "["
         << cutpoint->start_stmt << "," << cutpoint->end_stmt << "]";
      return os.str();
    };

    std::ostringstream os;
    os << "prev=" << describe_cutpoint(prev)
       << ", next=" << describe_cutpoint(next);
    return os.str();
  }

  void LogCompressedCutpoints(
      const std::vector<BoundaryDeltaRecord> &records) const {
    std::vector<CompressedCutpoint> cutpoints =
        BuildCompressedCutpoints(records);
    for (const CompressedCutpoint &cutpoint : cutpoints) {
      std::ostringstream os;
      os << "[MergeSharedCompressedCutpoint] id=" << cutpoint.id << "\n";
      os << "  start_stmt: " << cutpoint.start_stmt << "\n";
      os << "  end_stmt: " << cutpoint.end_stmt << "\n";
      os << "  reason: " << cutpoint.reason << "\n";
      os << "  changed_buffers: " << JoinStrings(cutpoint.changed_buffers)
         << "\n";
      os << "  born_buffers: " << JoinStrings(cutpoint.born_buffers) << "\n";
      os << "  closed_buffers: " << JoinStrings(cutpoint.closed_buffers)
         << "\n";
      os << "  persistent_buffers: "
         << JoinStrings(cutpoint.persistent_buffers);
      LOG(INFO) << os.str();
    }
  }

  std::vector<BufferLocalSegment> BuildLocalSegmentsFromCompressedCutpoints(
      const std::vector<CompressedCutpoint> &cutpoints) const {
    std::vector<BufferLocalSegment> segments;
    std::unordered_map<std::string, BufferLocalSegment> open_segments;

    auto ensure_open = [&](const std::string &buffer, int start_stmt,
                           const std::string &kind) {
      if (open_segments.count(buffer)) {
        return;
      }
      open_segments.emplace(
          buffer, BufferLocalSegment{buffer, start_stmt, -1, kind, ""});
    };

    auto close_open = [&](const std::string &buffer, int end_stmt,
                          const std::string &kind) {
      auto it = open_segments.find(buffer);
      if (it == open_segments.end()) {
        return;
      }
      it->second.end_stmt = end_stmt;
      it->second.end_kind = kind;
      segments.push_back(it->second);
      open_segments.erase(it);
    };

    for (const CompressedCutpoint &cutpoint : cutpoints) {
      for (const std::string &buffer : cutpoint.born_buffers) {
        ensure_open(buffer, cutpoint.start_stmt, "born");
      }
      for (const std::string &buffer : cutpoint.closed_buffers) {
        close_open(buffer, cutpoint.end_stmt, "closed");
      }
      for (const std::string &buffer : cutpoint.persistent_buffers) {
        ensure_open(buffer, cutpoint.start_stmt, "persistent");
      }
    }

    int tail_stmt = cutpoints.empty() ? 0 : cutpoints.back().end_stmt;
    for (auto &kv : open_segments) {
      kv.second.end_stmt = tail_stmt;
      kv.second.end_kind = "tail";
      segments.push_back(kv.second);
    }

    std::sort(segments.begin(), segments.end(),
              [](const BufferLocalSegment &lhs, const BufferLocalSegment &rhs) {
                if (lhs.buffer != rhs.buffer) {
                  return lhs.buffer < rhs.buffer;
                }
                if (lhs.start_stmt != rhs.start_stmt) {
                  return lhs.start_stmt < rhs.start_stmt;
                }
                return lhs.end_stmt < rhs.end_stmt;
              });
    return segments;
  }

  std::vector<BufferLocalSegment>
  BuildLocalSegments(const std::vector<BoundaryDeltaRecord> &records) const {
    return BuildLocalSegmentsFromCompressedCutpoints(
        BuildCompressedCutpoints(records));
  }

  bool StmtTouchesBuffer(const StmtEntry &entry,
                         const std::string &buffer) const {
    for (const shared_access_analysis::AccessEntry &access : entry.access) {
      if (AccessMatchesBuffer(access, buffer)) {
        return true;
      }
    }
    return ContainsBuffer(entry.touched, buffer) ||
           ContainsBuffer(entry.read_touched, buffer) ||
           ContainsBuffer(entry.write_touched, buffer);
  }

  int FindFirstAccessNodeIndex(const StmtEntry &entry,
                               const std::string &buffer) const {
    for (size_t i = 0; i < entry.access.size(); ++i) {
      if (AccessMatchesBuffer(entry.access[i], buffer)) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  int FindLastAccessNodeIndex(const StmtEntry &entry,
                              const std::string &buffer) const {
    for (int i = static_cast<int>(entry.access.size()) - 1; i >= 0; --i) {
      if (AccessMatchesBuffer(entry.access[static_cast<size_t>(i)], buffer)) {
        return i;
      }
    }
    return -1;
  }

  std::vector<BufferLocalSegment> RefineLocalSegmentsToConcreteAccesses(
      const std::vector<BufferLocalSegment> &segments,
      const std::vector<StmtEntry> &seq) const {
    std::vector<BufferLocalSegment> refined;
    refined.reserve(segments.size());
    const int seq_len = static_cast<int>(seq.size());
    for (const BufferLocalSegment &segment : segments) {
      int begin = std::max(0, segment.start_stmt);
      int end = std::min(segment.end_stmt, seq_len);
      int first_touch = -1;
      int last_touch = -1;
      for (int i = begin; i < end; ++i) {
        if (!StmtTouchesBuffer(seq[static_cast<size_t>(i)], segment.buffer)) {
          continue;
        }
        if (first_touch < 0) {
          first_touch = i;
        }
        last_touch = i + 1;
      }
      if (first_touch < 0 || last_touch <= first_touch) {
        continue;
      }
      BufferLocalSegment concrete = segment;
      concrete.start_stmt = first_touch;
      concrete.end_stmt = last_touch;
      refined.push_back(std::move(concrete));
    }
    std::sort(refined.begin(), refined.end(),
              [](const BufferLocalSegment &lhs, const BufferLocalSegment &rhs) {
                if (lhs.buffer != rhs.buffer) {
                  return lhs.buffer < rhs.buffer;
                }
                if (lhs.start_stmt != rhs.start_stmt) {
                  return lhs.start_stmt < rhs.start_stmt;
                }
                return lhs.end_stmt < rhs.end_stmt;
              });
    return refined;
  }

  void BuildAccessSegmentBindings(
      const std::vector<BufferPlannerSegment> &planner_segments,
      const std::unordered_map<std::string, const VarNode *> &name_to_var,
      const std::vector<StmtEntry> &seq) {
    const int seq_len = static_cast<int>(seq.size());
    for (const BufferPlannerSegment &segment : planner_segments) {
      auto var_it = name_to_var.find(segment.buffer);
      if (var_it == name_to_var.end()) {
        continue;
      }
      const VarNode *var = var_it->second;
      for (int stmt_id = std::max(0, segment.start_stmt);
           stmt_id < std::min(segment.end_stmt, seq_len); ++stmt_id) {
        const StmtEntry &entry = seq[static_cast<size_t>(stmt_id)];
        int first_access_index =
            FindFirstAccessNodeIndex(entry, segment.buffer);
        int last_access_index = FindLastAccessNodeIndex(entry, segment.buffer);
        if (first_access_index < 0 || last_access_index < 0) {
          continue;
        }
        for (int access_index = first_access_index;
             access_index <= last_access_index; ++access_index) {
          const shared_access_analysis::AccessEntry &access =
              entry.access[static_cast<size_t>(access_index)];
          if (!AccessMatchesBuffer(access, segment.buffer)) {
            continue;
          }
          if (entry.access_nodes.size() <= static_cast<size_t>(access_index)) {
            continue;
          }
          const Object *access_node =
              entry.access_nodes[static_cast<size_t>(access_index)];
          if (access_node == nullptr) {
            continue;
          }
          access_segment_ids_[access_node][var] = segment.planner_segment;
        }
      }
    }
  }

  std::vector<BufferCarryEdge>
  BuildCarryEdges(const std::vector<BufferLocalSegment> &segments) const {
    std::vector<BufferCarryEdge> carry_edges;
    std::unordered_map<std::string, std::vector<BufferLocalSegment>> by_buffer;
    for (const BufferLocalSegment &segment : segments) {
      by_buffer[segment.buffer].push_back(segment);
    }

    for (auto &kv : by_buffer) {
      auto &buffer_segments = kv.second;
      std::sort(
          buffer_segments.begin(), buffer_segments.end(),
          [](const BufferLocalSegment &lhs, const BufferLocalSegment &rhs) {
            return lhs.start_stmt < rhs.start_stmt;
          });
      for (size_t i = 0; i + 1 < buffer_segments.size(); ++i) {
        carry_edges.push_back(
            BufferCarryEdge{kv.first, buffer_segments[i].end_stmt,
                            buffer_segments[i + 1].start_stmt, 1, "candidate"});
      }
    }
    return carry_edges;
  }

  std::vector<BufferSegmentRelation> BuildSegmentRelations(
      const std::vector<BufferLocalSegment> &segments,
      const std::vector<StmtEntry> &seq,
      const std::unordered_map<const Object *, StmtAttr> &stmt_attrs) const {
    std::vector<BufferSegmentRelation> relations;
    std::unordered_map<std::string, std::vector<BufferLocalSegment>> by_buffer;
    for (const BufferLocalSegment &segment : segments) {
      by_buffer[segment.buffer].push_back(segment);
    }

    auto analyze_gap_value_flow = [&](const std::string &buffer, int gap_start,
                                      int rhs_end_stmt) -> GapValueFlow {
      if (seq.empty()) {
        return GapValueFlow::kNoReaccess;
      }
      int begin = std::max(0, gap_start + 1);
      int end = std::min(rhs_end_stmt, static_cast<int>(seq.size()) - 1);
      if (begin > end) {
        return GapValueFlow::kNoReaccess;
      }
      for (int i = begin; i <= end; ++i) {
        const StmtEntry &entry = seq[static_cast<size_t>(i)];
        bool matched_precise = false;
        bool has_read = false;
        bool has_write = false;
        for (const shared_access_analysis::AccessEntry &access : entry.access) {
          if (!AccessMatchesBuffer(access, buffer)) {
            continue;
          }
          matched_precise = true;
          has_read = has_read || access.type == shared_access_analysis::kRead;
          has_write =
              has_write || access.type == shared_access_analysis::kWrite;
        }
        if (!matched_precise) {
          has_read = ContainsBuffer(entry.read_touched, buffer);
          has_write = ContainsBuffer(entry.write_touched, buffer);
        }
        if (!has_read && !has_write) {
          continue;
        }
        if (has_read && has_write) {
          return GapValueFlow::kReadWriteSameStmt;
        }
        if (has_read) {
          return GapValueFlow::kReadBeforeWrite;
        }
        return GapValueFlow::kWriteBeforeRead;
      }
      return GapValueFlow::kNoReaccess;
    };

    auto observe_last_access_in_range =
        [&](const std::string &buffer, int begin,
            int end) -> BufferAccessObservation {
      BufferAccessObservation observation;
      if (seq.empty()) {
        return observation;
      }
      begin = std::max(0, begin);
      end = std::min(end, static_cast<int>(seq.size()) - 1);
      if (begin > end) {
        return observation;
      }
      for (int i = end; i >= begin; --i) {
        const StmtEntry &entry = seq[static_cast<size_t>(i)];
        if (auto access_index =
                SelectPreferredAccessIndex(entry, buffer, std::nullopt, true)) {
          const shared_access_analysis::AccessEntry &access =
              entry.access[static_cast<size_t>(access_index.value())];
          observation.stmt = i;
          observation.access_index = access_index.value();
          observation.has_read = access.type == shared_access_analysis::kRead;
          observation.has_write = access.type == shared_access_analysis::kWrite;
          return observation;
        }
        bool has_read = ContainsBuffer(entry.read_touched, buffer);
        bool has_write = ContainsBuffer(entry.write_touched, buffer);
        if (!has_read && !has_write) {
          continue;
        }
        observation.stmt = i;
        observation.has_read = has_read;
        observation.has_write = has_write;
        return observation;
      }
      return observation;
    };

    auto observe_first_access_in_range =
        [&](const std::string &buffer, int begin,
            int end) -> BufferAccessObservation {
      BufferAccessObservation observation;
      if (seq.empty()) {
        return observation;
      }
      begin = std::max(0, begin);
      end = std::min(end, static_cast<int>(seq.size()) - 1);
      if (begin > end) {
        return observation;
      }
      for (int i = begin; i <= end; ++i) {
        const StmtEntry &entry = seq[static_cast<size_t>(i)];
        if (auto access_index = SelectPreferredAccessIndex(
                entry, buffer, std::nullopt, false)) {
          const shared_access_analysis::AccessEntry &access =
              entry.access[static_cast<size_t>(access_index.value())];
          observation.stmt = i;
          observation.access_index = access_index.value();
          observation.has_read = access.type == shared_access_analysis::kRead;
          observation.has_write = access.type == shared_access_analysis::kWrite;
          return observation;
        }
        bool has_read = ContainsBuffer(entry.read_touched, buffer);
        bool has_write = ContainsBuffer(entry.write_touched, buffer);
        if (!has_read && !has_write) {
          continue;
        }
        observation.stmt = i;
        observation.has_read = has_read;
        observation.has_write = has_write;
        return observation;
      }
      return observation;
    };

    auto observe_first_read_in_range = [&](const std::string &buffer, int begin,
                                           int end) -> int {
      if (seq.empty()) {
        return -1;
      }
      begin = std::max(0, begin);
      end = std::min(end, static_cast<int>(seq.size()) - 1);
      if (begin > end) {
        return -1;
      }
      for (int i = begin; i <= end; ++i) {
        const StmtEntry &entry = seq[static_cast<size_t>(i)];
        if (SelectPreferredAccessIndex(entry, buffer,
                                       shared_access_analysis::kRead, false)) {
          return i;
        }
        bool matched_precise =
            std::any_of(entry.access.begin(), entry.access.end(),
                        [&](const shared_access_analysis::AccessEntry &access) {
                          return AccessMatchesBuffer(access, buffer);
                        });
        if (!matched_precise && ContainsBuffer(entry.read_touched, buffer)) {
          return i;
        }
      }
      return -1;
    };

    auto observe_first_write_in_range = [&](const std::string &buffer,
                                            int begin, int end) -> int {
      if (seq.empty()) {
        return -1;
      }
      begin = std::max(0, begin);
      end = std::min(end, static_cast<int>(seq.size()) - 1);
      if (begin > end) {
        return -1;
      }
      for (int i = begin; i <= end; ++i) {
        const StmtEntry &entry = seq[static_cast<size_t>(i)];
        if (SelectPreferredAccessIndex(entry, buffer,
                                       shared_access_analysis::kWrite, false)) {
          return i;
        }
        bool matched_precise =
            std::any_of(entry.access.begin(), entry.access.end(),
                        [&](const shared_access_analysis::AccessEntry &access) {
                          return AccessMatchesBuffer(access, buffer);
                        });
        if (!matched_precise && ContainsBuffer(entry.write_touched, buffer)) {
          return i;
        }
      }
      return -1;
    };

    auto same_enclosing_if_scope = [&](int lhs_stmt, int rhs_stmt) -> bool {
      if (lhs_stmt < 0 || rhs_stmt < 0 ||
          lhs_stmt >= static_cast<int>(seq.size()) ||
          rhs_stmt >= static_cast<int>(seq.size())) {
        return false;
      }
      auto lhs_it = stmt_attrs.find(seq[static_cast<size_t>(lhs_stmt)].stmt);
      auto rhs_it = stmt_attrs.find(seq[static_cast<size_t>(rhs_stmt)].stmt);
      if (lhs_it == stmt_attrs.end() || rhs_it == stmt_attrs.end()) {
        return false;
      }
      const std::vector<int> &lhs_path = lhs_it->second.enclosing_if_path;
      const std::vector<int> &rhs_path = rhs_it->second.enclosing_if_path;
      if (lhs_path.empty() || rhs_path.empty()) {
        return false;
      }
      size_t common = 0;
      while (common < lhs_path.size() && common < rhs_path.size() &&
             lhs_path[common] == rhs_path[common]) {
        ++common;
      }
      return common > 0;
    };

    auto gap_is_single_branch_alternative = [&](int lhs_end,
                                                int rhs_start) -> bool {
      if (lhs_end < 0 || rhs_start < 0 ||
          lhs_end >= static_cast<int>(seq.size()) ||
          rhs_start >= static_cast<int>(seq.size()) || lhs_end > rhs_start) {
        return false;
      }
      if (!same_enclosing_if_scope(lhs_end, rhs_start)) {
        return false;
      }
      auto lhs_it = stmt_attrs.find(seq[static_cast<size_t>(lhs_end)].stmt);
      auto rhs_it = stmt_attrs.find(seq[static_cast<size_t>(rhs_start)].stmt);
      if (lhs_it == stmt_attrs.end() || rhs_it == stmt_attrs.end()) {
        return false;
      }
      const std::vector<int> &lhs_path = lhs_it->second.enclosing_if_path;
      const std::vector<int> &rhs_path = rhs_it->second.enclosing_if_path;
      size_t common = 0;
      while (common < lhs_path.size() && common < rhs_path.size() &&
             lhs_path[common] == rhs_path[common]) {
        ++common;
      }
      if (common == 0) {
        return false;
      }
      int if_begin = lhs_path[common - 1];
      if (if_begin < 0 || if_begin >= static_cast<int>(seq.size())) {
        return false;
      }
      int64_t offset = seq[static_cast<size_t>(if_begin)].scope_pair_offset;
      if (offset <= 0) {
        return false;
      }
      int if_end = if_begin + static_cast<int>(offset);
      return lhs_end < if_end && rhs_start < if_end;
    };

    for (auto &kv : by_buffer) {
      auto &buffer_segments = kv.second;
      std::sort(
          buffer_segments.begin(), buffer_segments.end(),
          [](const BufferLocalSegment &lhs, const BufferLocalSegment &rhs) {
            return lhs.start_stmt < rhs.start_stmt;
          });
      for (size_t i = 0; i + 1 < buffer_segments.size(); ++i) {
        const BufferLocalSegment &lhs = buffer_segments[i];
        const BufferLocalSegment &rhs = buffer_segments[i + 1];
        std::string relation = "continuation";
        GapValueFlow gap_value_flow =
            analyze_gap_value_flow(kv.first, lhs.end_stmt, rhs.end_stmt);
        BufferAccessObservation last_lhs_access = observe_last_access_in_range(
            kv.first, lhs.start_stmt, lhs.end_stmt);
        const SummaryAccessEntry *last_lhs_precise_access =
            last_lhs_access.stmt >= 0 && last_lhs_access.access_index >= 0
                ? &seq[static_cast<size_t>(last_lhs_access.stmt)]
                       .access[static_cast<size_t>(
                           last_lhs_access.access_index)]
                : nullptr;
        std::string last_lhs_access_expr =
            last_lhs_precise_access != nullptr
                ? DescribeAccessExpr(*last_lhs_precise_access)
            : last_lhs_access.stmt >= 0
                ? DescribeBufferAccessExpr(
                      seq[static_cast<size_t>(last_lhs_access.stmt)].stmt,
                      kv.first, last_lhs_access.has_read)
                : "none";
        std::string last_lhs_access_footprint =
            last_lhs_precise_access != nullptr
                ? DescribeAccessFootprint(*last_lhs_precise_access)
            : last_lhs_access.stmt >= 0
                ? DescribeBufferAccessFootprint(
                      seq[static_cast<size_t>(last_lhs_access.stmt)].stmt,
                      kv.first, last_lhs_access.has_read)
                : "none";
        BufferAccessObservation first_gap_access =
            observe_first_access_in_range(kv.first, lhs.end_stmt + 1,
                                          rhs.start_stmt - 1);
        BufferAccessObservation first_rhs_access =
            observe_first_access_in_range(kv.first, rhs.start_stmt,
                                          rhs.end_stmt);
        int first_post_gap_read_stmt = observe_first_read_in_range(
            kv.first, lhs.end_stmt + 1, rhs.end_stmt);
        int first_post_gap_write_stmt = observe_first_write_in_range(
            kv.first, lhs.end_stmt + 1, rhs.end_stmt);
        int first_rhs_read_stmt =
            observe_first_read_in_range(kv.first, rhs.start_stmt, rhs.end_stmt);
        int first_rhs_write_stmt = observe_first_write_in_range(
            kv.first, rhs.start_stmt, rhs.end_stmt);
        BufferAccessObservation first_rhs_read_access =
            observe_first_access_in_range(kv.first, first_rhs_read_stmt,
                                          first_rhs_read_stmt);
        BufferAccessObservation first_rhs_write_access =
            observe_first_access_in_range(kv.first, first_rhs_write_stmt,
                                          first_rhs_write_stmt);
        const SummaryAccessEntry *first_rhs_read_precise_access =
            first_rhs_read_access.stmt >= 0 &&
                    first_rhs_read_access.access_index >= 0
                ? &seq[static_cast<size_t>(first_rhs_read_access.stmt)]
                       .access[static_cast<size_t>(
                           first_rhs_read_access.access_index)]
                : nullptr;
        const SummaryAccessEntry *first_rhs_write_precise_access =
            first_rhs_write_access.stmt >= 0 &&
                    first_rhs_write_access.access_index >= 0
                ? &seq[static_cast<size_t>(first_rhs_write_access.stmt)]
                       .access[static_cast<size_t>(
                           first_rhs_write_access.access_index)]
                : nullptr;
        std::string first_rhs_read_expr =
            first_rhs_read_precise_access != nullptr
                ? DescribeAccessExpr(*first_rhs_read_precise_access)
            : first_rhs_read_stmt >= 0
                ? DescribeBufferAccessExpr(
                      seq[static_cast<size_t>(first_rhs_read_stmt)].stmt,
                      kv.first, true)
                : "none";
        std::string first_rhs_write_expr =
            first_rhs_write_precise_access != nullptr
                ? DescribeAccessExpr(*first_rhs_write_precise_access)
            : first_rhs_write_stmt >= 0
                ? DescribeBufferAccessExpr(
                      seq[static_cast<size_t>(first_rhs_write_stmt)].stmt,
                      kv.first, false)
                : "none";
        std::string first_rhs_read_footprint =
            first_rhs_read_precise_access != nullptr
                ? DescribeAccessFootprint(*first_rhs_read_precise_access)
            : first_rhs_read_stmt >= 0
                ? DescribeBufferAccessFootprint(
                      seq[static_cast<size_t>(first_rhs_read_stmt)].stmt,
                      kv.first, true)
                : "none";
        std::string first_rhs_write_footprint =
            first_rhs_write_precise_access != nullptr
                ? DescribeAccessFootprint(*first_rhs_write_precise_access)
            : first_rhs_write_stmt >= 0
                ? DescribeBufferAccessFootprint(
                      seq[static_cast<size_t>(first_rhs_write_stmt)].stmt,
                      kv.first, false)
                : "none";
        auto footprint_is_subregion = [&](const std::string &footprint) {
          return IsSubregionFootprint(footprint);
        };
        std::string gap_region_signal = "unknown";
        if (footprint_is_subregion(first_rhs_read_footprint) ||
            footprint_is_subregion(first_rhs_write_footprint) ||
            footprint_is_subregion(last_lhs_access_footprint)) {
          gap_region_signal = "subregion_access_pattern";
        }
        bool subregion_live_across_gap =
            gap_value_flow == GapValueFlow::kReadBeforeWrite &&
            footprint_is_subregion(first_rhs_read_footprint) &&
            (footprint_is_subregion(last_lhs_access_footprint) ||
             footprint_is_subregion(first_rhs_write_footprint));
        bool value_live_across_gap =
            gap_value_flow == GapValueFlow::kReadBeforeWrite ||
            gap_value_flow == GapValueFlow::kReadWriteSameStmt;
        if (gap_value_flow == GapValueFlow::kWriteBeforeRead &&
            gap_is_single_branch_alternative(lhs.end_stmt, rhs.start_stmt)) {
          value_live_across_gap = true;
        }
        bool partial_overwrite_only =
            gap_value_flow == GapValueFlow::kWriteBeforeRead &&
            footprint_is_subregion(first_rhs_write_footprint);
        if (partial_overwrite_only && value_live_across_gap) {
          partial_overwrite_only = false;
        }
        if (subregion_live_across_gap) {
          value_live_across_gap = false;
        }
        bool overwrite_safe = false;
        if (lhs.end_kind == "closed" && rhs.start_kind == "born") {
          relation = "rebirth";
        } else if (lhs.end_kind == "closed" && rhs.start_kind == "persistent") {
          relation = "reopen_after_gap";
        }
        if (!value_live_across_gap &&
            gap_value_flow == GapValueFlow::kWriteBeforeRead) {
          overwrite_safe = true;
        }
        if (partial_overwrite_only) {
          overwrite_safe = false;
        }
        std::string gap_semantics = GapSemanticsFromValueFlow(gap_value_flow);
        if (subregion_live_across_gap) {
          gap_semantics = "subregion_live_across_gap";
        } else if (partial_overwrite_only) {
          gap_semantics = "partial_overwrite_only";
        }
        relations.push_back(BufferSegmentRelation{
            kv.first,
            static_cast<int>(i),
            static_cast<int>(i + 1),
            lhs.end_stmt,
            rhs.start_stmt,
            relation,
            overwrite_safe,
            value_live_across_gap,
            GapValueFlowToString(gap_value_flow),
            gap_semantics,
            subregion_live_across_gap,
            last_lhs_access.stmt,
            AccessObservationKind(last_lhs_access.has_read,
                                  last_lhs_access.has_write),
            last_lhs_access_expr,
            last_lhs_access_footprint,
            first_gap_access.stmt,
            AccessObservationKind(first_gap_access.has_read,
                                  first_gap_access.has_write),
            first_rhs_access.stmt,
            AccessObservationKind(first_rhs_access.has_read,
                                  first_rhs_access.has_write),
            first_post_gap_read_stmt,
            first_post_gap_write_stmt,
            first_rhs_read_stmt,
            first_rhs_write_stmt,
            first_rhs_read_expr,
            first_rhs_write_expr,
            first_rhs_read_footprint,
            first_rhs_write_footprint,
            gap_region_signal});
      }
    }
    return relations;
  }

  std::vector<BufferSemanticSegment> BuildSemanticSegments(
      const std::vector<BufferLocalSegment> &segments,
      const std::vector<BufferSegmentRelation> &relations) const {
    std::vector<BufferSemanticSegment> semantic_segments;
    std::unordered_map<std::string, std::vector<BufferLocalSegment>>
        by_buffer_segments;
    std::unordered_map<std::string, std::vector<BufferSegmentRelation>>
        by_buffer_relations;
    for (const BufferLocalSegment &segment : segments) {
      by_buffer_segments[segment.buffer].push_back(segment);
    }
    for (const BufferSegmentRelation &relation : relations) {
      by_buffer_relations[relation.buffer].push_back(relation);
    }

    for (auto &kv : by_buffer_segments) {
      auto &buffer_segments = kv.second;
      std::sort(
          buffer_segments.begin(), buffer_segments.end(),
          [](const BufferLocalSegment &lhs, const BufferLocalSegment &rhs) {
            return lhs.start_stmt < rhs.start_stmt;
          });
      auto relation_it = by_buffer_relations.find(kv.first);
      const std::vector<BufferSegmentRelation> *buffer_relations =
          relation_it == by_buffer_relations.end() ? nullptr
                                                   : &relation_it->second;

      if (buffer_segments.empty()) {
        continue;
      }

      int semantic_id = 0;
      int semantic_start = buffer_segments.front().start_stmt;
      int semantic_end = buffer_segments.front().end_stmt;
      int semantic_from_structural = 0;

      auto flush_semantic = [&](int to_structural_segment) {
        semantic_segments.push_back(BufferSemanticSegment{
            kv.first, semantic_id, semantic_start, semantic_end,
            semantic_from_structural, to_structural_segment});
      };

      for (size_t i = 0; i + 1 < buffer_segments.size(); ++i) {
        const BufferSegmentRelation *relation = nullptr;
        if (buffer_relations != nullptr && i < buffer_relations->size()) {
          relation = &(*buffer_relations)[i];
        }

        bool carry_value_across_gap =
            relation != nullptr && relation->value_live_across_gap;
        if (carry_value_across_gap) {
          semantic_end =
              std::max(semantic_end, buffer_segments[i + 1].end_stmt);
          continue;
        }

        flush_semantic(static_cast<int>(i));
        ++semantic_id;
        semantic_start = buffer_segments[i + 1].start_stmt;
        semantic_end = buffer_segments[i + 1].end_stmt;
        semantic_from_structural = static_cast<int>(i + 1);
      }

      flush_semantic(static_cast<int>(buffer_segments.size() - 1));
    }

    std::sort(
        semantic_segments.begin(), semantic_segments.end(),
        [](const BufferSemanticSegment &lhs, const BufferSemanticSegment &rhs) {
          if (lhs.buffer != rhs.buffer) {
            return lhs.buffer < rhs.buffer;
          }
          return lhs.semantic_segment < rhs.semantic_segment;
        });
    return semantic_segments;
  }

  std::vector<BufferSemanticGap>
  BuildSemanticGaps(const std::vector<BufferSegmentRelation> &relations) const {
    std::vector<BufferSemanticGap> semantic_gaps;
    std::unordered_map<std::string, int> semantic_cursor;
    for (const BufferSegmentRelation &relation : relations) {
      int current_semantic = semantic_cursor[relation.buffer];
      bool split_after = !relation.value_live_across_gap;
      if (split_after) {
        semantic_gaps.push_back(BufferSemanticGap{
            relation.buffer, current_semantic, current_semantic + 1,
            relation.gap_start, relation.gap_end, relation.relation,
            relation.gap_semantics, relation.overwrite_safe});
        semantic_cursor[relation.buffer] = current_semantic + 1;
      }
    }
    return semantic_gaps;
  }

  std::vector<BufferPlannerSegment> BuildPlannerSegments(
      const std::vector<BufferLocalSegment> &segments,
      const std::vector<BufferSegmentRelation> &relations) const {
    std::vector<BufferPlannerSegment> planner_segments;
    std::unordered_map<std::string, std::vector<BufferLocalSegment>>
        by_buffer_segments;
    std::unordered_map<std::string, std::vector<BufferSegmentRelation>>
        by_buffer_relations;
    for (const BufferLocalSegment &segment : segments) {
      by_buffer_segments[segment.buffer].push_back(segment);
    }
    for (const BufferSegmentRelation &relation : relations) {
      by_buffer_relations[relation.buffer].push_back(relation);
    }

    for (auto &kv : by_buffer_segments) {
      auto &buffer_segments = kv.second;
      std::sort(
          buffer_segments.begin(), buffer_segments.end(),
          [](const BufferLocalSegment &lhs, const BufferLocalSegment &rhs) {
            return lhs.start_stmt < rhs.start_stmt;
          });
      auto relation_it = by_buffer_relations.find(kv.first);
      const std::vector<BufferSegmentRelation> *buffer_relations =
          relation_it == by_buffer_relations.end() ? nullptr
                                                   : &relation_it->second;

      if (buffer_segments.empty()) {
        continue;
      }

      int planner_id = 0;
      int planner_start = buffer_segments.front().start_stmt;
      int planner_end = buffer_segments.front().end_stmt;
      int planner_from_structural = 0;

      auto flush_planner = [&](int to_structural_segment) {
        planner_segments.push_back(BufferPlannerSegment{
            kv.first, planner_id, planner_start, planner_end,
            planner_from_structural, to_structural_segment});
      };

      for (size_t i = 0; i + 1 < buffer_segments.size(); ++i) {
        const BufferSegmentRelation *relation = nullptr;
        if (buffer_relations != nullptr && i < buffer_relations->size()) {
          relation = &(*buffer_relations)[i];
        }

        bool split_storage_epoch =
            relation != nullptr &&
            relation->gap_semantics == "partial_overwrite_only";
        if (!split_storage_epoch) {
          planner_end = std::max(planner_end, buffer_segments[i + 1].end_stmt);
          continue;
        }

        flush_planner(static_cast<int>(i));
        ++planner_id;
        planner_start = buffer_segments[i + 1].start_stmt;
        planner_end = buffer_segments[i + 1].end_stmt;
        planner_from_structural = static_cast<int>(i + 1);
      }

      flush_planner(static_cast<int>(buffer_segments.size() - 1));
    }

    std::sort(
        planner_segments.begin(), planner_segments.end(),
        [](const BufferPlannerSegment &lhs, const BufferPlannerSegment &rhs) {
          if (lhs.buffer != rhs.buffer) {
            return lhs.buffer < rhs.buffer;
          }
          return lhs.planner_segment < rhs.planner_segment;
        });
    return planner_segments;
  }

  std::vector<BufferPlannerSegment> MergeLoopCarriedPlannerSegments(
      const std::vector<BufferPlannerSegment> &planner_segments,
      const std::vector<StmtEntry> &seq) const {
    std::unordered_map<size_t, LoopBoundarySummary> loop_summaries =
        BuildLoopBoundarySummaries(seq);
    if (loop_summaries.empty()) {
      return planner_segments;
    }

    struct LoopInterval {
      int begin{-1};
      int end{-1};
      std::vector<std::string> loop_carried_buffers;
    };

    std::vector<LoopInterval> loop_intervals;
    loop_intervals.reserve(loop_summaries.size());
    for (const auto &kv : loop_summaries) {
      size_t loop_begin = kv.first;
      if (loop_begin >= seq.size()) {
        continue;
      }
      int64_t offset = seq[loop_begin].scope_pair_offset;
      if (offset <= 0) {
        continue;
      }
      int begin = static_cast<int>(loop_begin) + 1;
      int end = static_cast<int>(loop_begin + static_cast<size_t>(offset));
      if (begin >= end) {
        continue;
      }
      loop_intervals.push_back(
          LoopInterval{begin, end, kv.second.loop_carried_buffers});
    }
    if (loop_intervals.empty()) {
      return planner_segments;
    }

    auto contains = [](const std::vector<std::string> &items,
                       const std::string &value) {
      return std::find(items.begin(), items.end(), value) != items.end();
    };

    std::unordered_map<std::string, std::vector<BufferPlannerSegment>>
        by_buffer;
    for (const BufferPlannerSegment &segment : planner_segments) {
      by_buffer[segment.buffer].push_back(segment);
    }

    std::vector<BufferPlannerSegment> merged_segments;
    merged_segments.reserve(planner_segments.size());

    for (auto &kv : by_buffer) {
      auto &segments = kv.second;
      std::sort(
          segments.begin(), segments.end(),
          [](const BufferPlannerSegment &lhs, const BufferPlannerSegment &rhs) {
            if (lhs.start_stmt != rhs.start_stmt) {
              return lhs.start_stmt < rhs.start_stmt;
            }
            return lhs.end_stmt < rhs.end_stmt;
          });

      if (segments.size() <= 1) {
        merged_segments.insert(merged_segments.end(), segments.begin(),
                               segments.end());
        continue;
      }

      std::vector<int> parent(segments.size());
      for (size_t i = 0; i < parent.size(); ++i) {
        parent[i] = static_cast<int>(i);
      }

      std::function<int(int)> find_root = [&](int x) {
        if (parent[static_cast<size_t>(x)] != x) {
          parent[static_cast<size_t>(x)] =
              find_root(parent[static_cast<size_t>(x)]);
        }
        return parent[static_cast<size_t>(x)];
      };

      auto unify = [&](int lhs, int rhs) {
        int lhs_root = find_root(lhs);
        int rhs_root = find_root(rhs);
        if (lhs_root != rhs_root) {
          parent[static_cast<size_t>(rhs_root)] = lhs_root;
        }
      };

      for (const LoopInterval &loop : loop_intervals) {
        if (!contains(loop.loop_carried_buffers, kv.first)) {
          continue;
        }

        std::vector<int> overlapping;
        for (size_t i = 0; i < segments.size(); ++i) {
          const BufferPlannerSegment &segment = segments[i];
          bool overlaps_loop = !(segment.end_stmt <= loop.begin ||
                                 loop.end <= segment.start_stmt);
          if (overlaps_loop) {
            overlapping.push_back(static_cast<int>(i));
          }
        }
        for (size_t i = 1; i < overlapping.size(); ++i) {
          unify(overlapping[0], overlapping[i]);
        }
      }

      std::unordered_map<int, BufferPlannerSegment> grouped;
      for (size_t i = 0; i < segments.size(); ++i) {
        int root = find_root(static_cast<int>(i));
        const BufferPlannerSegment &segment = segments[i];
        auto it = grouped.find(root);
        if (it == grouped.end()) {
          grouped.emplace(root, segment);
          continue;
        }
        BufferPlannerSegment &merged = it->second;
        merged.start_stmt = std::min(merged.start_stmt, segment.start_stmt);
        merged.end_stmt = std::max(merged.end_stmt, segment.end_stmt);
        merged.from_structural_segment = std::min(
            merged.from_structural_segment, segment.from_structural_segment);
        merged.to_structural_segment = std::max(merged.to_structural_segment,
                                                segment.to_structural_segment);
      }

      std::vector<BufferPlannerSegment> compacted;
      compacted.reserve(grouped.size());
      for (auto &group_kv : grouped) {
        compacted.push_back(std::move(group_kv.second));
      }
      std::sort(
          compacted.begin(), compacted.end(),
          [](const BufferPlannerSegment &lhs, const BufferPlannerSegment &rhs) {
            if (lhs.start_stmt != rhs.start_stmt) {
              return lhs.start_stmt < rhs.start_stmt;
            }
            return lhs.end_stmt < rhs.end_stmt;
          });
      for (size_t i = 0; i < compacted.size(); ++i) {
        compacted[i].planner_segment = static_cast<int>(i);
      }
      merged_segments.insert(merged_segments.end(), compacted.begin(),
                             compacted.end());
    }

    std::sort(
        merged_segments.begin(), merged_segments.end(),
        [](const BufferPlannerSegment &lhs, const BufferPlannerSegment &rhs) {
          if (lhs.buffer != rhs.buffer) {
            return lhs.buffer < rhs.buffer;
          }
          return lhs.planner_segment < rhs.planner_segment;
        });
    return merged_segments;
  }

  std::vector<BufferPlannerSegment>
  CoalescePlannerSegmentsWithAmbiguousAccessNodes(
      const std::vector<BufferPlannerSegment> &planner_segments,
      const std::vector<StmtEntry> &seq) const {
    std::unordered_map<std::string, std::vector<BufferPlannerSegment>>
        by_buffer;
    for (const BufferPlannerSegment &segment : planner_segments) {
      by_buffer[segment.buffer].push_back(segment);
    }

    std::vector<BufferPlannerSegment> coalesced_segments;
    coalesced_segments.reserve(planner_segments.size());

    for (auto &kv : by_buffer) {
      auto &segments = kv.second;
      std::sort(
          segments.begin(), segments.end(),
          [](const BufferPlannerSegment &lhs, const BufferPlannerSegment &rhs) {
            if (lhs.start_stmt != rhs.start_stmt) {
              return lhs.start_stmt < rhs.start_stmt;
            }
            return lhs.end_stmt < rhs.end_stmt;
          });

      if (segments.size() <= 1) {
        coalesced_segments.insert(coalesced_segments.end(), segments.begin(),
                                  segments.end());
        continue;
      }

      std::vector<int> parent(segments.size());
      for (size_t i = 0; i < parent.size(); ++i) {
        parent[i] = static_cast<int>(i);
      }

      std::function<int(int)> find_root = [&](int x) {
        if (parent[static_cast<size_t>(x)] != x) {
          parent[static_cast<size_t>(x)] =
              find_root(parent[static_cast<size_t>(x)]);
        }
        return parent[static_cast<size_t>(x)];
      };

      auto unify = [&](int lhs, int rhs) {
        int lhs_root = find_root(lhs);
        int rhs_root = find_root(rhs);
        if (lhs_root != rhs_root) {
          parent[static_cast<size_t>(rhs_root)] = lhs_root;
        }
      };

      std::unordered_map<const Object *, int> access_owner;
      const int seq_len = static_cast<int>(seq.size());
      for (size_t segment_index = 0; segment_index < segments.size();
           ++segment_index) {
        const BufferPlannerSegment &segment = segments[segment_index];
        for (int stmt_id = std::max(0, segment.start_stmt);
             stmt_id < std::min(segment.end_stmt, seq_len); ++stmt_id) {
          const StmtEntry &entry = seq[static_cast<size_t>(stmt_id)];
          for (size_t access_index = 0; access_index < entry.access.size();
               ++access_index) {
            const shared_access_analysis::AccessEntry &access =
                entry.access[access_index];
            if (!AccessMatchesBuffer(access, kv.first)) {
              continue;
            }
            if (entry.access_nodes.size() <= access_index) {
              continue;
            }
            const Object *access_node = entry.access_nodes[access_index];
            if (access_node == nullptr) {
              continue;
            }
            auto it = access_owner.find(access_node);
            if (it == access_owner.end()) {
              access_owner.emplace(access_node,
                                   static_cast<int>(segment_index));
            } else {
              unify(it->second, static_cast<int>(segment_index));
            }
          }
        }
      }

      std::unordered_map<int, BufferPlannerSegment> grouped;
      for (size_t i = 0; i < segments.size(); ++i) {
        int root = find_root(static_cast<int>(i));
        const BufferPlannerSegment &segment = segments[i];
        auto it = grouped.find(root);
        if (it == grouped.end()) {
          grouped.emplace(root, segment);
          continue;
        }
        BufferPlannerSegment &merged = it->second;
        merged.start_stmt = std::min(merged.start_stmt, segment.start_stmt);
        merged.end_stmt = std::max(merged.end_stmt, segment.end_stmt);
        merged.from_structural_segment = std::min(
            merged.from_structural_segment, segment.from_structural_segment);
        merged.to_structural_segment = std::max(merged.to_structural_segment,
                                                segment.to_structural_segment);
      }

      std::vector<BufferPlannerSegment> compacted;
      compacted.reserve(grouped.size());
      for (auto &group_kv : grouped) {
        compacted.push_back(std::move(group_kv.second));
      }
      std::sort(
          compacted.begin(), compacted.end(),
          [](const BufferPlannerSegment &lhs, const BufferPlannerSegment &rhs) {
            if (lhs.start_stmt != rhs.start_stmt) {
              return lhs.start_stmt < rhs.start_stmt;
            }
            return lhs.end_stmt < rhs.end_stmt;
          });
      for (size_t i = 0; i < compacted.size(); ++i) {
        compacted[i].planner_segment = static_cast<int>(i);
      }
      coalesced_segments.insert(coalesced_segments.end(), compacted.begin(),
                                compacted.end());
    }

    std::sort(
        coalesced_segments.begin(), coalesced_segments.end(),
        [](const BufferPlannerSegment &lhs, const BufferPlannerSegment &rhs) {
          if (lhs.buffer != rhs.buffer) {
            return lhs.buffer < rhs.buffer;
          }
          return lhs.planner_segment < rhs.planner_segment;
        });
    return coalesced_segments;
  }

  std::vector<BufferPlannerSegment> MergePlannerSegmentsAcrossLoopTailWrites(
      const std::vector<BufferPlannerSegment> &planner_segments,
      const std::vector<StmtEntry> &seq) const {
    struct LoopInterval {
      int begin{-1};
      int end{-1};
    };

    std::vector<LoopInterval> loop_intervals;
    for (size_t i = 0; i < seq.size(); ++i) {
      int64_t offset = seq[i].scope_pair_offset;
      if (offset <= 0) {
        continue;
      }
      if (seq[i].stmt == nullptr || !seq[i].stmt->IsInstance<ForNode>()) {
        continue;
      }
      int begin = static_cast<int>(i) + 1;
      int end = static_cast<int>(i + static_cast<size_t>(offset));
      if (begin < end) {
        loop_intervals.push_back(LoopInterval{begin, end});
      }
    }
    if (loop_intervals.empty()) {
      return planner_segments;
    }

    struct BufferLoopUse {
      int last_read{-1};
      int last_write{-1};
    };

    std::unordered_map<std::string, std::vector<BufferPlannerSegment>>
        by_buffer;
    for (const BufferPlannerSegment &segment : planner_segments) {
      by_buffer[segment.buffer].push_back(segment);
    }

    std::vector<BufferPlannerSegment> merged_segments;
    merged_segments.reserve(planner_segments.size());

    for (auto &kv : by_buffer) {
      auto &segments = kv.second;
      std::sort(
          segments.begin(), segments.end(),
          [](const BufferPlannerSegment &lhs, const BufferPlannerSegment &rhs) {
            if (lhs.start_stmt != rhs.start_stmt) {
              return lhs.start_stmt < rhs.start_stmt;
            }
            return lhs.end_stmt < rhs.end_stmt;
          });

      if (segments.size() <= 1) {
        merged_segments.insert(merged_segments.end(), segments.begin(),
                               segments.end());
        continue;
      }

      std::vector<int> parent(segments.size());
      for (size_t i = 0; i < parent.size(); ++i) {
        parent[i] = static_cast<int>(i);
      }

      std::function<int(int)> find_root = [&](int x) {
        if (parent[static_cast<size_t>(x)] != x) {
          parent[static_cast<size_t>(x)] =
              find_root(parent[static_cast<size_t>(x)]);
        }
        return parent[static_cast<size_t>(x)];
      };

      auto unify = [&](int lhs, int rhs) {
        int lhs_root = find_root(lhs);
        int rhs_root = find_root(rhs);
        if (lhs_root != rhs_root) {
          parent[static_cast<size_t>(rhs_root)] = lhs_root;
        }
      };

      for (const LoopInterval &loop : loop_intervals) {
        BufferLoopUse use;
        for (int stmt_id = std::max(0, loop.begin);
             stmt_id < std::min(loop.end, static_cast<int>(seq.size()));
             ++stmt_id) {
          const StmtEntry &entry = seq[static_cast<size_t>(stmt_id)];
          for (const shared_access_analysis::AccessEntry &access :
               entry.access) {
            if (!AccessMatchesBuffer(access, kv.first)) {
              continue;
            }
            if (access.type == shared_access_analysis::kRead) {
              use.last_read = stmt_id;
            } else if (access.type == shared_access_analysis::kWrite) {
              use.last_write = stmt_id;
            }
          }
        }

        if (!(use.last_read >= 0 && use.last_write > use.last_read)) {
          continue;
        }

        std::vector<int> overlapping_segments;
        for (size_t i = 0; i < segments.size(); ++i) {
          const BufferPlannerSegment &segment = segments[i];
          bool overlaps_tail_transition =
              !(segment.end_stmt <= use.last_read ||
                use.last_write < segment.start_stmt);
          if (overlaps_tail_transition) {
            overlapping_segments.push_back(static_cast<int>(i));
          }
        }
        for (size_t i = 1; i < overlapping_segments.size(); ++i) {
          unify(overlapping_segments[0], overlapping_segments[i]);
        }
      }

      std::unordered_map<int, BufferPlannerSegment> grouped;
      for (size_t i = 0; i < segments.size(); ++i) {
        int root = find_root(static_cast<int>(i));
        const BufferPlannerSegment &segment = segments[i];
        auto it = grouped.find(root);
        if (it == grouped.end()) {
          grouped.emplace(root, segment);
          continue;
        }
        BufferPlannerSegment &merged = it->second;
        merged.start_stmt = std::min(merged.start_stmt, segment.start_stmt);
        merged.end_stmt = std::max(merged.end_stmt, segment.end_stmt);
        merged.from_structural_segment = std::min(
            merged.from_structural_segment, segment.from_structural_segment);
        merged.to_structural_segment = std::max(merged.to_structural_segment,
                                                segment.to_structural_segment);
      }

      std::vector<BufferPlannerSegment> compacted;
      compacted.reserve(grouped.size());
      for (auto &group_kv : grouped) {
        compacted.push_back(std::move(group_kv.second));
      }
      std::sort(
          compacted.begin(), compacted.end(),
          [](const BufferPlannerSegment &lhs, const BufferPlannerSegment &rhs) {
            if (lhs.start_stmt != rhs.start_stmt) {
              return lhs.start_stmt < rhs.start_stmt;
            }
            return lhs.end_stmt < rhs.end_stmt;
          });
      for (size_t i = 0; i < compacted.size(); ++i) {
        compacted[i].planner_segment = static_cast<int>(i);
      }
      merged_segments.insert(merged_segments.end(), compacted.begin(),
                             compacted.end());
    }

    std::sort(
        merged_segments.begin(), merged_segments.end(),
        [](const BufferPlannerSegment &lhs, const BufferPlannerSegment &rhs) {
          if (lhs.buffer != rhs.buffer) {
            return lhs.buffer < rhs.buffer;
          }
          return lhs.planner_segment < rhs.planner_segment;
        });
    return merged_segments;
  }

  void LogTimelineSegmentsAndCarryEdges(
      const std::vector<BoundaryDeltaRecord> &records,
      const std::vector<StmtEntry> &seq,
      const std::unordered_map<const Object *, StmtAttr> &stmt_attrs) const {
    std::vector<BufferLocalSegment> segments = BuildLocalSegments(records);
    std::vector<BufferCarryEdge> carry_edges = BuildCarryEdges(segments);
    std::vector<BufferSegmentRelation> relations =
        BuildSegmentRelations(segments, seq, stmt_attrs);
    std::vector<CompressedCutpoint> cutpoints =
        BuildCompressedCutpoints(records);
    std::vector<BufferSemanticSegment> semantic_segments =
        BuildSemanticSegments(segments, relations);
    std::vector<BufferSemanticGap> semantic_gaps = BuildSemanticGaps(relations);
    std::vector<BufferPlannerSegment> planner_segments =
        BuildPlannerSegments(segments, relations);

    for (const BufferLocalSegment &segment : segments) {
      std::ostringstream os;
      os << "[MergeSharedTimelineSegment] buffer=" << segment.buffer << "\n";
      os << "  start_stmt: " << segment.start_stmt << "\n";
      os << "  end_stmt: " << segment.end_stmt << "\n";
      os << "  start_kind: " << segment.start_kind << "\n";
      os << "  end_kind: " << segment.end_kind;
      LOG(INFO) << os.str();
    }

    for (const BufferCarryEdge &edge : carry_edges) {
      std::ostringstream os;
      os << "[MergeSharedCarryEdge] buffer=" << edge.buffer << "\n";
      os << "  src_stmt: " << edge.src_stmt << "\n";
      os << "  dst_stmt: " << edge.dst_stmt << "\n";
      os << "  min_iteration_distance: " << edge.min_iteration_distance << "\n";
      os << "  kind: " << edge.kind;
      LOG(INFO) << os.str();
    }

    for (const BufferSegmentRelation &relation : relations) {
      std::ostringstream os;
      os << "[MergeSharedSegmentRelation] buffer=" << relation.buffer << "\n";
      os << "  from_segment: " << relation.from_segment << "\n";
      os << "  to_segment: " << relation.to_segment << "\n";
      os << "  gap_start: " << relation.gap_start << "\n";
      os << "  gap_end: " << relation.gap_end << "\n";
      os << "  relation: " << relation.relation << "\n";
      os << "  gap_value_flow: " << relation.gap_value_flow << "\n";
      os << "  gap_semantics: " << relation.gap_semantics << "\n";
      os << "  subregion_live_across_gap: "
         << (relation.subregion_live_across_gap ? "yes" : "no") << "\n";
      os << "  last_lhs_access_stmt: " << relation.last_lhs_access_stmt << "\n";
      os << "  last_lhs_access_kind: " << relation.last_lhs_access_kind << "\n";
      os << "  last_lhs_access_expr: " << relation.last_lhs_access_expr << "\n";
      os << "  last_lhs_access_footprint: "
         << relation.last_lhs_access_footprint << "\n";
      os << "  first_gap_access_stmt: " << relation.first_gap_access_stmt
         << "\n";
      os << "  first_gap_access_kind: " << relation.first_gap_access_kind
         << "\n";
      os << "  first_rhs_access_stmt: " << relation.first_rhs_access_stmt
         << "\n";
      os << "  first_rhs_access_kind: " << relation.first_rhs_access_kind
         << "\n";
      os << "  first_post_gap_read_stmt: " << relation.first_post_gap_read_stmt
         << "\n";
      os << "  first_post_gap_read_ctx: "
         << DescribeStmtCutpointContext(relation.first_post_gap_read_stmt,
                                        cutpoints)
         << "\n";
      os << "  first_post_gap_write_stmt: "
         << relation.first_post_gap_write_stmt << "\n";
      os << "  first_post_gap_write_ctx: "
         << DescribeStmtCutpointContext(relation.first_post_gap_write_stmt,
                                        cutpoints)
         << "\n";
      os << "  first_rhs_read_stmt: " << relation.first_rhs_read_stmt << "\n";
      os << "  first_rhs_read_ctx: "
         << DescribeStmtCutpointContext(relation.first_rhs_read_stmt, cutpoints)
         << "\n";
      os << "  first_rhs_read_expr: " << relation.first_rhs_read_expr << "\n";
      os << "  first_rhs_read_footprint: " << relation.first_rhs_read_footprint
         << "\n";
      os << "  first_rhs_write_stmt: " << relation.first_rhs_write_stmt << "\n";
      os << "  first_rhs_write_ctx: "
         << DescribeStmtCutpointContext(relation.first_rhs_write_stmt,
                                        cutpoints)
         << "\n";
      os << "  first_rhs_write_expr: " << relation.first_rhs_write_expr << "\n";
      os << "  first_rhs_write_footprint: "
         << relation.first_rhs_write_footprint << "\n";
      os << "  gap_region_signal: " << relation.gap_region_signal << "\n";
      os << "  value_live_across_gap: "
         << (relation.value_live_across_gap ? "yes" : "no") << "\n";
      os << "  overwrite_safe: " << (relation.overwrite_safe ? "yes" : "no");
      LOG(INFO) << os.str();
    }

    for (const BufferSemanticSegment &segment : semantic_segments) {
      std::ostringstream os;
      os << "[MergeSharedSemanticSegment] buffer=" << segment.buffer << "\n";
      os << "  semantic_segment: " << segment.semantic_segment << "\n";
      os << "  start_stmt: " << segment.start_stmt << "\n";
      os << "  end_stmt: " << segment.end_stmt << "\n";
      os << "  from_structural_segment: " << segment.from_structural_segment
         << "\n";
      os << "  to_structural_segment: " << segment.to_structural_segment;
      LOG(INFO) << os.str();
    }

    for (const BufferSemanticGap &gap : semantic_gaps) {
      std::ostringstream os;
      os << "[MergeSharedSemanticGap] buffer=" << gap.buffer << "\n";
      os << "  from_semantic_segment: " << gap.from_semantic_segment << "\n";
      os << "  to_semantic_segment: " << gap.to_semantic_segment << "\n";
      os << "  gap_start: " << gap.gap_start << "\n";
      os << "  gap_end: " << gap.gap_end << "\n";
      os << "  structural_relation: " << gap.structural_relation << "\n";
      os << "  gap_semantics: " << gap.gap_semantics << "\n";
      os << "  overwrite_safe: " << (gap.overwrite_safe ? "yes" : "no");
      LOG(INFO) << os.str();
    }

    for (const BufferPlannerSegment &segment : planner_segments) {
      std::ostringstream os;
      os << "[MergeSharedPlannerSegment] buffer=" << segment.buffer << "\n";
      os << "  planner_segment: " << segment.planner_segment << "\n";
      os << "  start_stmt: " << segment.start_stmt << "\n";
      os << "  end_stmt: " << segment.end_stmt << "\n";
      os << "  from_structural_segment: " << segment.from_structural_segment
         << "\n";
      os << "  to_structural_segment: " << segment.to_structural_segment;
      LOG(INFO) << os.str();
    }
  }

  void
  LogLifetimePlanSeeds(const std::vector<BufferLifetimePlanSeed> &seeds) const {
    for (const BufferLifetimePlanSeed &seed : seeds) {
      if (seed.stable_closures.empty() && seed.unstable_closures.empty() &&
          seed.carry_over_boundaries.empty()) {
        continue;
      }

      std::ostringstream os;
      os << "[MergeSharedLifetimeIR] buffer=" << seed.buffer << "\n";
      os << "  inferred_phase_class: "
         << BufferPhaseClassToString(seed.inferred_phase_class) << "\n";
      os << "  suggested_split_policy: "
         << BufferSplitPolicyToString(seed.suggested_split_policy) << "\n";
      os << "  reclaim_root: " << (seed.reclaim_root ? "yes" : "no") << "\n";
      os << "  requires_loop_aware_summary: "
         << (seed.requires_loop_aware_summary ? "yes" : "no") << "\n";
      os << "  read_stmt_count: " << seed.read_stmt_count << "\n";
      os << "  write_stmt_count: " << seed.write_stmt_count << "\n";
      os << "  gen_count: " << seed.gen_count << "\n";
      os << "  kill_count: " << seed.kill_count << "\n";
      os << "  stable_closures: " << JoinStrings(seed.stable_closures) << "\n";
      os << "  unstable_closures: " << JoinStrings(seed.unstable_closures)
         << "\n";
      os << "  carry_over_boundaries: "
         << JoinStrings(seed.carry_over_boundaries) << "\n";
      os << "  loop_local_boundaries: "
         << JoinStrings(seed.loop_local_boundaries) << "\n";
      os << "  loop_middle_boundaries: "
         << JoinStrings(seed.loop_middle_boundaries) << "\n";
      os << "  loop_carried_boundaries: "
         << JoinStrings(seed.loop_carried_boundaries);
      LOG(INFO) << os.str();
    }
  }

  void
  LogCandidateStability(const std::vector<BoundaryDeltaRecord> &records,
                        const std::unordered_map<std::string, BufferAccessStats>
                            &access_stats_by_buffer,
                        const std::unordered_map<size_t, LoopBoundarySummary>
                            &loop_summaries) const {
    if (records.empty()) {
      return;
    }

    std::unordered_set<std::string> buffer_names;
    for (const BoundaryDeltaRecord &record : records) {
      buffer_names.insert(record.post_buffers.begin(),
                          record.post_buffers.end());
      buffer_names.insert(record.closed_buffers.begin(),
                          record.closed_buffers.end());
      buffer_names.insert(record.persistent_buffers.begin(),
                          record.persistent_buffers.end());
      buffer_names.insert(record.new_buffers.begin(), record.new_buffers.end());
    }

    std::vector<std::string> sorted_buffers(buffer_names.begin(),
                                            buffer_names.end());
    std::sort(sorted_buffers.begin(), sorted_buffers.end());

    auto contains = [](const std::vector<std::string> &items,
                       const std::string &value) {
      return std::binary_search(items.begin(), items.end(), value);
    };

    for (const std::string &buffer : sorted_buffers) {
      std::vector<std::string> stable_closures;
      std::vector<std::string> unstable_closures;
      std::vector<std::string> carry_over_boundaries;

      for (size_t i = 0; i < records.size(); ++i) {
        const BoundaryDeltaRecord &record = records[i];
        if (contains(record.persistent_buffers, buffer)) {
          carry_over_boundaries.push_back(std::to_string(record.boundary));
        }
        if (!contains(record.closed_buffers, buffer)) {
          continue;
        }

        std::optional<size_t> reappears_at;
        for (size_t j = i + 1; j < records.size(); ++j) {
          if (contains(records[j].new_buffers, buffer) ||
              contains(records[j].persistent_buffers, buffer) ||
              contains(records[j].post_buffers, buffer)) {
            reappears_at = records[j].boundary;
            break;
          }
        }

        if (reappears_at.has_value()) {
          unstable_closures.push_back(std::to_string(record.boundary) + "->" +
                                      std::to_string(reappears_at.value()));
        } else {
          stable_closures.push_back(std::to_string(record.boundary));
        }
      }

      if (stable_closures.empty() && unstable_closures.empty() &&
          carry_over_boundaries.empty()) {
        continue;
      }

      std::ostringstream os;
      os << "[MergeSharedCandidateStability] buffer=" << buffer << "\n";
      os << "  stable_closures: " << JoinStrings(stable_closures) << "\n";
      os << "  unstable_closures: " << JoinStrings(unstable_closures) << "\n";
      os << "  carry_over_boundaries: " << JoinStrings(carry_over_boundaries);
      LOG(INFO) << os.str();
    }

    LogLifetimePlanSeeds(BuildLifetimePlanSeeds(records, access_stats_by_buffer,
                                                loop_summaries));
  }

  std::vector<BoundaryDeltaRecord>
  CollectBoundaryDeltaRecords(const std::vector<StmtEntry> &seq,
                              bool emit_logs) const {
    std::vector<BoundaryDeltaRecord> emitted_records;
    if (seq.size() < 2) {
      return emitted_records;
    }

    std::vector<SummaryStmtEntry> summary_seq = ConvertToSummarySequence(seq);
    ffi::Array<tir::IterVar> env_threads;
    ConstrSet current_cset;
    runtime::StorageScope sync_scope = CurrentSummaryScope();
    bool has_previous_emitted = false;
    std::vector<std::string> previous_post_accesses;
    std::vector<std::string> previous_closed_buffers;
    std::vector<std::string> previous_persistent_buffers;
    std::vector<std::string> previous_new_buffers;
    for (size_t boundary = 0; boundary + 1 < summary_seq.size(); ++boundary) {
      std::vector<SummaryStmtEntry> pre_seq(summary_seq.begin(),
                                            summary_seq.begin() + boundary + 1);
      std::vector<SummaryStmtEntry> post_seq(summary_seq.begin() + boundary + 1,
                                             summary_seq.end());
      SummaryResult pre_summary =
          shared_access_analysis::SummarizeAccessSequence(
              std::move(pre_seq), nullptr, sync_scope, env_threads,
              current_cset, {}, false);
      SummaryResult post_summary =
          shared_access_analysis::SummarizeAccessSequence(
              std::move(post_seq), nullptr, sync_scope, env_threads,
              current_cset, {}, false);

      std::vector<std::string> pre_accesses =
          AccessSetToSortedStrings(pre_summary.exposed_accesses);
      std::vector<std::string> post_accesses =
          AccessSetToSortedStrings(post_summary.exposed_accesses);
      std::vector<std::string> pre_buffers =
          BufferSetToSortedStrings(pre_summary.exposed_accesses);
      std::vector<std::string> post_buffers =
          BufferSetToSortedStrings(post_summary.exposed_accesses);

      if (pre_accesses == post_accesses && pre_buffers == post_buffers) {
        continue;
      }

      std::vector<std::string> closed_accesses =
          VectorDiff(pre_accesses, post_accesses);
      std::vector<std::string> new_accesses =
          VectorDiff(post_accesses, pre_accesses);
      std::vector<std::string> persistent_accesses =
          VectorDiff(pre_accesses, closed_accesses);
      std::vector<std::string> closed_buffers =
          VectorDiff(pre_buffers, post_buffers);
      std::vector<std::string> new_buffers =
          VectorDiff(post_buffers, pre_buffers);
      std::vector<std::string> persistent_buffers =
          VectorDiff(pre_buffers, closed_buffers);

      if (has_previous_emitted && previous_post_accesses == post_accesses &&
          previous_closed_buffers == closed_buffers &&
          previous_persistent_buffers == persistent_buffers &&
          previous_new_buffers == new_buffers) {
        continue;
      }

      if (emit_logs) {
        std::ostringstream os;
        os << "[MergeSharedSummaryDelta] scope=" << sync_scope.to_string()
           << " boundary=" << boundary
           << " left=" << seq[boundary].stmt->GetTypeKey()
           << " right=" << seq[boundary + 1].stmt->GetTypeKey() << "\n";
        os << "  left_reads: " << VarVecToString(seq[boundary].read_touched)
           << "\n";
        os << "  left_writes: " << VarVecToString(seq[boundary].write_touched)
           << "\n";
        os << "  right_reads: "
           << VarVecToString(seq[boundary + 1].read_touched) << "\n";
        os << "  right_writes: "
           << VarVecToString(seq[boundary + 1].write_touched) << "\n";
        if (seq[boundary].is_sync || seq[boundary + 1].is_sync) {
          os << "  boundary_sync: left="
             << (seq[boundary].is_sync ? "yes" : "no")
             << ", right=" << (seq[boundary + 1].is_sync ? "yes" : "no")
             << "\n";
        }
        auto left_event_it = event_map_.find(seq[boundary].stmt);
        if (left_event_it != event_map_.end() &&
            (!left_event_it->second.gen.empty() ||
             !left_event_it->second.kill.empty())) {
          os << "  left_event_gen: "
             << VarVecToString(left_event_it->second.gen) << "\n";
          os << "  left_event_kill: "
             << VarVecToString(left_event_it->second.kill) << "\n";
        }
        auto right_event_it = event_map_.find(seq[boundary + 1].stmt);
        if (right_event_it != event_map_.end() &&
            (!right_event_it->second.gen.empty() ||
             !right_event_it->second.kill.empty())) {
          os << "  right_event_gen: "
             << VarVecToString(right_event_it->second.gen) << "\n";
          os << "  right_event_kill: "
             << VarVecToString(right_event_it->second.kill) << "\n";
        }
        os << "  pre_exposed: " << JoinStrings(pre_accesses) << "\n";
        os << "  post_exposed: " << JoinStrings(post_accesses) << "\n";
        os << "  closed_accesses: " << JoinStrings(closed_accesses) << "\n";
        os << "  persistent_accesses: " << JoinStrings(persistent_accesses)
           << "\n";
        os << "  new_accesses: " << JoinStrings(new_accesses) << "\n";
        os << "  closed_buffers: " << JoinStrings(closed_buffers) << "\n";
        os << "  persistent_buffers: " << JoinStrings(persistent_buffers)
           << "\n";
        os << "  new_buffers: " << JoinStrings(new_buffers) << "\n";
        os << "  closure_candidates: " << JoinStrings(closed_buffers) << "\n";
        os << "  carry_over_buffers: " << JoinStrings(persistent_buffers);
        LOG(INFO) << os.str();
      }

      previous_post_accesses = post_accesses;
      previous_closed_buffers = closed_buffers;
      previous_persistent_buffers = persistent_buffers;
      previous_new_buffers = new_buffers;
      has_previous_emitted = true;
      emitted_records.push_back(BoundaryDeltaRecord{
          boundary, post_buffers, closed_buffers, persistent_buffers,
          new_buffers, seq[boundary].stmt, seq[boundary + 1].stmt});
    }

    return emitted_records;
  }

  void LogBoundarySummaryDeltas(
      const std::vector<StmtEntry> &seq,
      const std::unordered_map<const Object *, StmtAttr> &stmt_attrs) const {
    std::vector<BoundaryDeltaRecord> emitted_records =
        CollectBoundaryDeltaRecords(seq, true);
    if (emitted_records.empty()) {
      return;
    }
    LogCandidateStability(emitted_records, BuildBufferAccessStats(seq),
                          BuildLoopBoundarySummaries(seq));
    LogCutpointGraph(seq, stmt_attrs);
    LogCompressedCutpoints(emitted_records);
    LogTimelineSegmentsAndCarryEdges(emitted_records, seq, stmt_attrs);
  }

  // Event entry in liveness analysis
  struct EventEntry {
    // variables we generate
    std::vector<const VarNode *> gen;
    // variables we kill
    std::vector<const VarNode *> kill;
  };

  void PlanAlignment(const Stmt &stmt) {
    DLOG(INFO) << "PlanAlignment";
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (const auto *call = node.as<CallNode>()) {
        if (call->op.same_as(tl::tl_gemm()) ||
            call->op.same_as(tl::tl_gemm_sp())) {
          DLOG(INFO) << "PostOrderVisit CallNode tl_gemm and tl_gemm_sp: "
                     << call->op;
        }
      }
    });
  }
  /*!
   * \brief Liveness analysis to find gen and kill point of each variable.
   * \param seq the linear pattern of storage access
   */
  void LivenessAnalysis(
      const std::vector<StmtEntry> &seq,
      const std::unordered_map<const Object *, StmtAttr> &stmt_attrs) {
    // find kill point, do a reverse linear scan.
    std::unordered_set<const VarNode *> touched;
    for (size_t i = seq.size(); i != 0; --i) {
      const StmtEntry &s = seq[i - 1];
      for (const VarNode *buffer : s.touched) {
        if (!touched.count(buffer)) {
          touched.insert(buffer);
          event_map_[s.stmt].kill.push_back(buffer);
        }
      }
    }
    // find gen point, do forward scan
    touched.clear();
    for (size_t i = 0; i < seq.size(); ++i) {
      int64_t offset = seq[i].scope_pair_offset;
      if (offset < 0)
        continue;
      const StmtEntry &s = seq[i + offset];
      for (const VarNode *buffer : s.touched) {
        if (!touched.count(buffer)) {
          touched.insert(buffer);
          event_map_[s.stmt].gen.push_back(buffer);
        }
      }
    }

    if (verbose_) {
      std::vector<const Object *> stmt_keys;
      for (const auto &stmt_entry : seq) {
        auto stmt = stmt_entry.stmt;
        if (std::find(stmt_keys.begin(), stmt_keys.end(), stmt) ==
            stmt_keys.end()) {
          stmt_keys.push_back(stmt);
        }
      }
      LOG(DEBUG) << "Before reorder kill points, Liveness Analysis Results for "
                 << (is_dynamic_ ? "Dynamic" : "Static") << " Shared Memory:";
      for (const auto &stmt_key : stmt_keys) {
        auto it = event_map_.find(stmt_key);
        if (it == event_map_.end())
          continue;

        const EventEntry &entry = it->second;
        if (entry.gen.empty() && entry.kill.empty())
          continue;
        ICHECK(stmt_attrs.count(stmt_key))
            << "stmt_key = " << stmt_key->GetTypeKey();
        auto level = stmt_attrs.at(stmt_key).level;
        LOG(DEBUG) << "  Statement: " << stmt_key->GetTypeKey()
                   << " (scope_level: " << level << ")";

        std::stringstream gen_vars_ss;
        bool x_generated = false;
        for (const VarNode *var : entry.gen) {
          gen_vars_ss << var->name_hint << " ";
          if (var->name_hint == "x") {
            x_generated = true;
          }
        }
        if (!entry.gen.empty()) {
          std::string gen_log_msg = "    GEN: " + gen_vars_ss.str();
          if (x_generated) {
            gen_log_msg += " <-- Buffer 'x' generated";
          }
          LOG(DEBUG) << gen_log_msg;
        }

        std::stringstream kill_vars_ss;
        bool x_killed = false;
        for (const VarNode *var : entry.kill) {
          kill_vars_ss << var->name_hint << " ";
          if (var->name_hint == "x") {
            x_killed = true;
          }
        }
        if (!entry.kill.empty()) {
          std::string kill_log_msg = "    KILL: " + kill_vars_ss.str();
          if (x_killed) {
            kill_log_msg += " <-- Buffer 'x' killed";
          }
          LOG(DEBUG) << kill_log_msg;
        }
      }
      LOG(DEBUG) << "End of Liveness Analysis Results.";
    }

    // Reorder kill points:
    // For each buffer, if its kill statement is at a deeper scope level than
    // its gen statement, we need to move the kill point to the end of the gen
    // statement's scope level. This ensures proper memory deallocation at the
    // right scope boundary.
    std::vector<StmtEntry> gen_kill_seq;
    for (const auto &stmt_entry : seq) {
      // if has gen and kill, add to gen_kill_seq
      if (!event_map_[stmt_entry.stmt].gen.empty() ||
          !event_map_[stmt_entry.stmt].kill.empty()) {
        gen_kill_seq.push_back(stmt_entry);
      }
    }

    // Pending kill insertions are deferred until after the per-event loop
    // below: ``event_map_[last_stmt_at_level].kill.push_back(...)`` may
    // target the very same vector we are currently iterating, in which case
    // the push_back can reallocate the underlying storage and invalidate
    // ``it``. MSVC's debug iterator verification catches this on the next
    // ``it != event.kill.end()`` check ("vector iterators incompatible");
    // release builds silently dereference freed memory. Buffering the
    // push-backs and applying them later removes the aliasing entirely.
    std::vector<std::pair<const Object *, const VarNode *>>
        pending_kill_inserts;

    for (auto &event_pair : event_map_) {
      const Object *stmt = event_pair.first;
      EventEntry &event = event_pair.second;

      // Skip if no kill points to process
      if (event.kill.empty())
        continue;

      // Get scope level of current statement
      ICHECK(stmt_attrs.count(stmt));
      int kill_level = stmt_attrs.at(stmt).level;

      std::unordered_set<const VarNode *> visited_buffers;

      // For each killed buffer, find its gen statement and check scope levels
      for (auto it = event.kill.begin(); it != event.kill.end();) {
        const VarNode *buffer = *it;
        bool found_gen = false;
        int gen_level = 0;

        // Find the gen statement for this buffer
        for (const auto &gen_pair : event_map_) {
          const auto &gen_event = gen_pair.second;
          if (std::find(gen_event.gen.begin(), gen_event.gen.end(), buffer) !=
              gen_event.gen.end()) {
            found_gen = true;
            gen_level = stmt_attrs.at(gen_pair.first).level;
            break;
          }
        }

        if (found_gen && kill_level > gen_level) {
          if (visited_buffers.count(buffer)) {
            ++it;
            continue;
          }
          // Need to move kill point - remove from current event
          it = event.kill.erase(it);

          // Find the last statement at gen_level and add kill point there
          // Find the last statement at gen_level in the sequence
          const Object *last_stmt_at_level = nullptr;
          auto stmt_it = gen_kill_seq.begin();
          for (; stmt_it != gen_kill_seq.end(); ++stmt_it) {
            if (stmt_it->stmt == stmt) {
              break;
            }
          }
          // start from current statement and find the last statement at
          // gen_level
          //
          // Additionally, stop if the next statement generates (births) a
          // different shared-memory buffer.  Without this check the
          // reordered kill can land *past* another buffer's gen, creating
          // a false liveness overlap that blocks memory reuse even when the
          // two buffers' true lifetimes are disjoint (e.g., Q_shared and
          // O_shared in Flash Attention can share the same shared memory
          // region).
          //
          // This is safe because shared-memory allocations (T.alloc_shared)
          // are always placed *outside* pipelined loop bodies — no new
          // shared buffer is born inside the deep scope where kills are
          // being reordered from.

          for (; stmt_it != gen_kill_seq.end(); ++stmt_it) {
            auto next_it = stmt_it + 1;
            if (next_it == gen_kill_seq.end() ||
                stmt_attrs.at(next_it->stmt).level == gen_level) {
              last_stmt_at_level = stmt_it->stmt;
              break;
            }
            // Stop if the next statement births a different shared buffer.
            auto next_event_it = event_map_.find(next_it->stmt);
            if (next_event_it != event_map_.end() &&
                !next_event_it->second.gen.empty()) {
              bool has_other_gen = false;
              for (const VarNode *gen_buf : next_event_it->second.gen) {
                if (gen_buf != buffer) {
                  has_other_gen = true;
                  break;
                }
              }
              if (has_other_gen) {
                last_stmt_at_level = stmt_it->stmt;
                break;
              }
            }
          }
          if (last_stmt_at_level) {
            // Defer: pushing into event.kill (the vector ``it`` iterates) or
            // into any other event.kill while the outer ``event_map_`` range
            // loop is live can invalidate iterators / dangling references.
            pending_kill_inserts.emplace_back(last_stmt_at_level, buffer);
            visited_buffers.insert(buffer);
          }
        } else {
          ++it;
        }
      }
    }

    // Apply deferred kill insertions now that no iterator into ``event_map_``
    // / ``event.kill`` is live.
    for (const auto &insert : pending_kill_inserts) {
      event_map_[insert.first].kill.push_back(insert.second);
    }

    std::vector<const Object *> stmt_keys;
    for (const auto &stmt_entry : seq) {
      auto stmt = stmt_entry.stmt;
      if (std::find(stmt_keys.begin(), stmt_keys.end(), stmt) ==
          stmt_keys.end()) {
        stmt_keys.push_back(stmt);
      }
    }

    if (verbose_) {
      LOG(DEBUG) << "Liveness Analysis Results for "
                 << (is_dynamic_ ? "Dynamic" : "Static") << " Shared Memory:";
      for (const auto &stmt_key : stmt_keys) {
        auto it = event_map_.find(stmt_key);
        if (it == event_map_.end())
          continue;

        const EventEntry &entry = it->second;
        if (entry.gen.empty() && entry.kill.empty())
          continue;
        ICHECK(stmt_attrs.count(stmt_key))
            << "stmt_key = " << stmt_key->GetTypeKey();
        auto level = stmt_attrs.at(stmt_key).level;
        LOG(DEBUG) << "  Statement: " << stmt_key->GetTypeKey()
                   << " (scope_level: " << level << ")";

        std::stringstream gen_vars_ss;
        bool x_generated = false;
        for (const VarNode *var : entry.gen) {
          gen_vars_ss << var->name_hint << " ";
          if (var->name_hint == "x") {
            x_generated = true;
          }
        }
        if (!entry.gen.empty()) {
          std::string gen_log_msg = "    GEN: " + gen_vars_ss.str();
          if (x_generated) {
            gen_log_msg += " <-- Buffer 'x' generated";
          }
          LOG(DEBUG) << gen_log_msg;
        }

        std::stringstream kill_vars_ss;
        bool x_killed = false;
        for (const VarNode *var : entry.kill) {
          kill_vars_ss << var->name_hint << " ";
          if (var->name_hint == "x") {
            x_killed = true;
          }
        }
        if (!entry.kill.empty()) {
          std::string kill_log_msg = "    KILL: " + kill_vars_ss.str();
          if (x_killed) {
            kill_log_msg += " <-- Buffer 'x' killed";
          }
          LOG(DEBUG) << kill_log_msg;
        }
      }
      LOG(DEBUG) << "End of Liveness Analysis Results.";
    }
  }

  /*!
   * \brief Memory plan algorithm
   * \param seq the linear pattern of storage access
   * \param alloc_info
   */
  void
  PlanMemory(const std::vector<StmtEntry> &seq,
             const std::unordered_map<const Object *, StmtAttr> &stmt_attrs) {
    buffer_byte_offsets_.clear();
    buffer_segment_bindings_.clear();
    access_segment_ids_.clear();
    (void)stmt_attrs;

    if (shmem_allocs_.empty()) {
      merged_alloc_size_ = make_const(DataType::Int(64), 0);
      return;
    }

    const int seq_len = static_cast<int>(seq.size());
    std::unordered_map<const VarNode *, std::vector<std::pair<int, int>>>
        structural_segments_by_var;
    std::unordered_map<const VarNode *, std::optional<int>> open_segment_start;
    for (size_t i = 0; i < seq.size(); ++i) {
      auto event_it = event_map_.find(seq[i].stmt);
      if (event_it != event_map_.end()) {
        for (const VarNode *var : event_it->second.gen) {
          auto &open = open_segment_start[var];
          if (!open.has_value()) {
            open = static_cast<int>(i);
          }
        }
        for (const VarNode *var : event_it->second.kill) {
          auto &open = open_segment_start[var];
          if (open.has_value()) {
            structural_segments_by_var[var].push_back(
                {open.value(), static_cast<int>(i) + 1});
            open.reset();
          }
        }
      }
    }
    for (auto &kv : open_segment_start) {
      if (kv.second.has_value()) {
        auto kill_it = event_map_.find(last_access_stmt_[kv.first]);
        if (kill_it != event_map_.end()) {
          kill_it->second.kill.push_back(kv.first);
        }
        structural_segments_by_var[kv.first].push_back(
            {kv.second.value(), seq_len});
        kv.second.reset();
      }
    }

    std::vector<BoundaryDeltaRecord> boundary_records =
        CollectBoundaryDeltaRecords(seq, false);
    std::vector<BufferLocalSegment> structural_segments =
        RefineLocalSegmentsToConcreteAccesses(
            BuildLocalSegments(boundary_records), seq);
    if (verbose_) {
      for (const auto &s : structural_segments) {
        std::cerr << "[MSMA-EPOCHSEG] buffer=" << s.buffer
                  << " start=" << s.start_stmt << " end=" << s.end_stmt
                  << " start_kind=" << s.start_kind
                  << " end_kind=" << s.end_kind << "\n";
      }
    }

    std::vector<BufferSegmentRelation> segment_relations =
        BuildSegmentRelations(structural_segments, seq, stmt_attrs);
    auto stage0 = BuildPlannerSegments(structural_segments, segment_relations);
    auto stage1 = MergeLoopCarriedPlannerSegments(stage0, seq);
    auto stage2 = CoalescePlannerSegmentsWithAmbiguousAccessNodes(stage1, seq);
    auto stage3 = MergePlannerSegmentsAcrossLoopTailWrites(stage2, seq);
    std::vector<BufferPlannerSegment> planner_segments = stage3;
    if (verbose_) {
      auto dump_stage = [](const char *tag,
                           const std::vector<BufferPlannerSegment> &segs) {
        for (const auto &p : segs) {
          std::cerr << tag << " buffer=" << p.buffer
                    << " id=" << p.planner_segment << " start=" << p.start_stmt
                    << " end=" << p.end_stmt << "\n";
        }
      };
      dump_stage("[MSMA-STAGE0-RAW]", stage0);
      dump_stage("[MSMA-STAGE1-AFTER-LOOPCARRY]", stage1);
      dump_stage("[MSMA-STAGE2-AFTER-COALESCE]", stage2);
      dump_stage("[MSMA-STAGE3-AFTER-TAILWRITE]", stage3);
      for (const auto &r : segment_relations) {
        std::cerr << "[MSMA-RELATION] buffer=" << r.buffer
                  << " from=" << r.from_segment << " to=" << r.to_segment
                  << " gap=[" << r.gap_start << "," << r.gap_end << "]"
                  << " value_flow=" << r.gap_value_flow
                  << " gap_semantics=" << r.gap_semantics
                  << " overwrite_safe=" << (r.overwrite_safe ? 1 : 0)
                  << " value_live=" << (r.value_live_across_gap ? 1 : 0)
                  << "\n";
      }
    }

    std::unordered_map<std::string, const VarNode *> name_to_var;
    for (const auto &kv : shmem_allocs_) {
      name_to_var[kv.first->name_hint] = kv.first;
    }

    std::unordered_map<const VarNode *, std::vector<std::pair<int, int>>>
        segments_by_var;
    for (const BufferPlannerSegment &segment : planner_segments) {
      auto var_it = name_to_var.find(segment.buffer);
      if (var_it == name_to_var.end()) {
        continue;
      }
      segments_by_var[var_it->second].push_back(
          {segment.start_stmt, segment.end_stmt});
    }

    for (auto &kv : segments_by_var) {
      std::sort(kv.second.begin(), kv.second.end());
    }

    // Create a sorted vector of keys from shmem_allocs_ for deterministic
    // iteration
    std::vector<const VarNode *> sorted_vars;
    sorted_vars.reserve(shmem_allocs_.size());
    for (const auto &kv : shmem_allocs_) {
      sorted_vars.push_back(kv.first);
    }
    std::sort(sorted_vars.begin(), sorted_vars.end(),
              [](const VarNode *a, const VarNode *b) {
                return a->name_hint < b->name_hint;
              });

    std::vector<BufInfo> buf_infos;
    buf_infos.reserve(shmem_allocs_.size());
    // Build a BufInfo for all allocations that participate in liveness.
    for (const VarNode *var : sorted_vars) {
      auto seg_it = segments_by_var.find(var);
      if (seg_it == segments_by_var.end() || seg_it->second.empty()) {
        continue;
      }

      BufInfo info;
      info.var = var;
      info.name = var->name_hint;
      info.segments = seg_it->second;
      info.start = info.segments.front().first;
      info.end = std::max(info.segments.back().second, info.start + 1);
      info.alignment = align_bytes_;
      auto align_it = shmem_alignment_map_.find(var);
      if (align_it != shmem_alignment_map_.end()) {
        info.alignment = std::max(info.alignment, align_it->second);
      }

      const AllocateNode *alloc = shmem_allocs_.at(var);
      int64_t bytes_per_elem =
          static_cast<int64_t>(alloc->dtype.bytes() * alloc->dtype.lanes());
      DataType size_dtype = DataType::Int(32);
      if (!alloc->extents.empty()) {
        size_dtype = alloc->extents[0].dtype();
      }
      if (!size_dtype.is_int() && !size_dtype.is_uint()) {
        size_dtype = DataType::Int(32);
      }

      PrimExpr size_expr = make_const(size_dtype, bytes_per_elem);
      for (const PrimExpr &extent : alloc->extents) {
        PrimExpr e = extent;
        if (e.dtype() != size_dtype) {
          e = cast(size_dtype, e);
        }
        size_expr = size_expr * e;
      }
      info.size_dtype = size_dtype;
      info.size_expr = size_expr;

      int64_t const_extent = alloc->ConstantAllocationSize();
      if (const_extent >= 0) {
        info.const_size_bytes = const_extent * bytes_per_elem;
      }

      buf_infos.push_back(std::move(info));
    }

    // Stable order so the later passes have deterministic behaviour.
    std::sort(buf_infos.begin(), buf_infos.end(),
              [](const BufInfo &a, const BufInfo &b) {
                if (a.start != b.start)
                  return a.start < b.start;
                if (a.end != b.end)
                  return a.end < b.end;
                return a.name < b.name;
              });

    std::vector<Interval> intervals;
    intervals.reserve(buf_infos.size());
    for (const BufInfo &info : buf_infos) {
      if (!info.const_size_bytes.has_value())
        continue;
      // Only constant-sized buffers participate in the arena packing because
      // dynamic sizes must be placed sequentially later.
      for (size_t segment_id = 0; segment_id < info.segments.size();
           ++segment_id) {
        Interval interval;
        interval.start = info.segments[segment_id].first;
        interval.end = info.segments[segment_id].second;
        interval.size_bytes = static_cast<size_t>(
            std::max<int64_t>(0, info.const_size_bytes.value()));
        interval.alignment = info.alignment;
        interval.var = info.var;
        interval.segment_id = static_cast<int>(segment_id);
        auto live_it = liveness_epochs_by_var_.find(info.var);
        if (live_it != liveness_epochs_by_var_.end() &&
            !live_it->second.empty()) {
          interval.live_epochs = &live_it->second;
        }
        intervals.push_back(interval);
      }
    }

    std::vector<Interval> packed_intervals = intervals;
    ArenaPlan plan = LinearScanPack(std::move(intervals));
    size_t arena_size_const = plan.arena_size;

    if (verbose_) {
      LOG(DEBUG) << "ArenaPlan (constant buffers): arena_size="
                 << arena_size_const;
      for (const auto &kv : plan.offsets) {
        const VarNode *var = kv.first;
        LOG(DEBUG) << "  " << var->name_hint << " -> offset=" << kv.second;
      }
      for (const auto &kv : plan.segment_offsets) {
        const VarNode *var = kv.first;
        for (size_t segment_id = 0; segment_id < kv.second.size();
             ++segment_id) {
          LOG(DEBUG) << "  " << var->name_hint << "[seg" << segment_id
                     << "] -> offset=" << kv.second[segment_id];
        }
      }
    }

    // Cursor tracks the running byte offset within the merged arena.
    DataType offset_dtype =
        buf_infos.empty() ? DataType::Int(32) : buf_infos.front().size_dtype;
    PrimExpr total_size =
        make_const(offset_dtype, static_cast<int64_t>(arena_size_const));
    PrimExpr cursor = AlignPrimExpr(
        make_const(offset_dtype, static_cast<int64_t>(arena_size_const)),
        align_bytes_);

    auto CastToOffset = [&](PrimExpr expr) -> PrimExpr {
      if (expr.dtype() == offset_dtype) {
        return expr;
      }
      return cast(offset_dtype, expr);
    };

    for (const BufInfo &info : buf_infos) {
      PrimExpr offset_expr;
      auto it = plan.offsets.find(info.var);
      if (it != plan.offsets.end()) {
        offset_expr =
            make_const(offset_dtype, static_cast<int64_t>(it->second));
      } else {
        // Dynamic-sized buffers are appended after the constant arena.
        cursor = AlignPrimExpr(cursor, info.alignment);
        PrimExpr size_expr = CastToOffset(info.size_expr);
        offset_expr = cursor;
        cursor = offset_expr + size_expr;
      }

      buffer_byte_offsets_[info.var] = offset_expr;
      auto segment_it = plan.segment_offsets.find(info.var);
      if (segment_it != plan.segment_offsets.end()) {
        std::vector<SegmentBinding> bindings;
        bindings.reserve(segment_it->second.size());
        for (size_t segment_id = 0; segment_id < segment_it->second.size();
             ++segment_id) {
          bindings.push_back(SegmentBinding{info.segments[segment_id].first,
                                            info.segments[segment_id].second,
                                            segment_it->second[segment_id]});
        }
        if (!bindings.empty()) {
          buffer_segment_bindings_[info.var] = bindings;
          buffer_byte_offsets_[info.var] = make_const(
              offset_dtype, static_cast<int64_t>(bindings.front().offset));
          if (verbose_) {
            for (size_t binding_id = 0; binding_id < bindings.size();
                 ++binding_id) {
              LOG(DEBUG) << "  binding " << info.name << "[seg" << binding_id
                         << "] start=" << bindings[binding_id].start
                         << " end=" << bindings[binding_id].end
                         << " offset=" << bindings[binding_id].offset;
            }
          }
        }
      }
      PrimExpr buf_end = offset_expr + CastToOffset(info.size_expr);
      total_size = max(total_size, buf_end);
    }

    BuildAccessSegmentBindings(planner_segments, name_to_var, seq);

    merged_alloc_size_ = buf_infos.empty()
                             ? make_const(offset_dtype, 0)
                             : AlignPrimExpr(total_size, align_bytes_);

    bool overlap_detected = false;

    if (verbose_) {
      LOG(DEBUG) << "Memory Allocation Plan for "
                 << (is_dynamic_ ? "Dynamic" : "Static") << " Shared Memory:";
      LOG(DEBUG) << "  Total Merged Size (aligned): " << merged_alloc_size_;
      for (const BufInfo &info : buf_infos) {
        const PrimExpr &offset = buffer_byte_offsets_.at(info.var);
        LOG(DEBUG) << "    Buffer: " << info.name << " start=" << info.start
                   << " end=" << info.end << " alignment=" << info.alignment
                   << " offset=" << offset << " size=" << info.size_expr;
      }
      // Sanity check for overlapping constant buffer segments.
      for (size_t i = 0; i < packed_intervals.size(); ++i) {
        const Interval &a = packed_intervals[i];
        auto a_it = plan.segment_offsets.find(a.var);
        if (a_it == plan.segment_offsets.end() ||
            a_it->second.size() <= static_cast<size_t>(a.segment_id)) {
          continue;
        }
        int64_t a_off = static_cast<int64_t>(
            a_it->second[static_cast<size_t>(a.segment_id)]);
        int64_t a_end = a_off + static_cast<int64_t>(a.size_bytes);
        for (size_t j = i + 1; j < packed_intervals.size(); ++j) {
          const Interval &b = packed_intervals[j];
          if (a.var == b.var && a.segment_id == b.segment_id) {
            continue;
          }
          auto b_it = plan.segment_offsets.find(b.var);
          if (b_it == plan.segment_offsets.end() ||
              b_it->second.size() <= static_cast<size_t>(b.segment_id)) {
            continue;
          }
          bool live_overlap = !(a.end <= b.start || b.end <= a.start);
          if (!live_overlap) {
            continue;
          }
          // Same relaxer as in `LinearScanPack`: skip the overlap check if
          // per-epoch dataflow proves the buffers are never simultaneously
          // alive.  The B2 layout-signature gate keeps the legacy conflict
          // for buffers whose annotated layouts differ structurally.
          if (a.live_epochs && b.live_epochs && !a.live_epochs->empty() &&
              !b.live_epochs->empty() &&
              !LayoutAliasIncompatible(a.var, b.var)) {
            const auto &la = *a.live_epochs;
            const auto &lb = *b.live_epochs;
            auto ia = la.begin(), ib = lb.begin();
            bool intersects = false;
            while (ia != la.end() && ib != lb.end()) {
              if (*ia < *ib)
                ++ia;
              else if (*ib < *ia)
                ++ib;
              else {
                intersects = true;
                break;
              }
            }
            if (!intersects)
              continue;
          }
          int64_t b_off = static_cast<int64_t>(
              b_it->second[static_cast<size_t>(b.segment_id)]);
          int64_t b_end = b_off + static_cast<int64_t>(b.size_bytes);
          bool mem_overlap = !(a_end <= b_off || b_end <= a_off);
          if (mem_overlap) {
            overlap_detected = true;
            LOG(WARNING) << "Buffer overlap detected between "
                         << a.var->name_hint << "[seg" << a.segment_id
                         << "] and " << b.var->name_hint << "[seg"
                         << b.segment_id
                         << "] (lifetime overlap with offset ranges [" << a_off
                         << ", " << a_end << ") and [" << b_off << ", " << b_end
                         << ")).";
          }
        }
      }
    }

    if (overlap_detected) {
      LOG(WARNING) << "Detected overlapping constant buffers; falling back to "
                   << "sequential allocation without reuse.";
      buffer_byte_offsets_.clear();
      buffer_segment_bindings_.clear();
      access_segment_ids_.clear();
      // In the fallback path we simply lay buffers out sequentially.
      PrimExpr new_cursor = make_const(offset_dtype, 0);
      PrimExpr new_total = make_const(offset_dtype, 0);
      for (const BufInfo &info : buf_infos) {
        new_cursor = AlignPrimExpr(new_cursor, info.alignment);
        PrimExpr size_expr = CastToOffset(info.size_expr);
        buffer_byte_offsets_[info.var] = new_cursor;
        PrimExpr buf_end = new_cursor + size_expr;
        new_total = max(new_total, buf_end);
        new_cursor = buf_end;
      }
      merged_alloc_size_ = buf_infos.empty()
                               ? make_const(offset_dtype, 0)
                               : AlignPrimExpr(new_total, align_bytes_);
    }
  }

  // Whether enable dynamic analysis.
  bool is_dynamic_{true};

  // Whether enable verbose logging.
  bool verbose_{false};
  // The alignment bytes for the merged buffer
  int align_bytes_{16};
  // The var for the merged buffer
  Var merged_buf_var_{"buf_dyn_shmem",
                      PointerType(PrimType(DataType::UInt(8)), "shared.dyn")};
  // The mapping from the original buffer var to its allocate
  std::unordered_map<const VarNode *, const AllocateNode *> shmem_allocs_;
  // The size of the merged buffer
  PrimExpr merged_alloc_size_{0};
  // The mapping from the original buffer var to its offset in the merged buffer
  std::unordered_map<const VarNode *, PrimExpr> buffer_byte_offsets_;
  // Optional segment-aware bindings for buffers with repeated phased reuse.
  std::unordered_map<const VarNode *, std::vector<SegmentBinding>>
      buffer_segment_bindings_;
  // Mapping from linearized access node to chosen segment id per buffer.
  std::unordered_map<const Object *, std::unordered_map<const VarNode *, int>>
      access_segment_ids_;
  // The mapping from the original buffer objects to their location in the
  // merged buffer.
  std::unordered_map<const BufferNode *, Buffer> buffer_remap_;
  // The flag indicating whether the merged buffer has been allocated
  bool allocated_{false};
  // Locations of free ops.
  std::unordered_map<const Object *, EventEntry> event_map_;
  // The mapping of buffer bytes alignment
  std::unordered_map<const VarNode *, int> shmem_alignment_map_;
  // Last concrete access statement recorded during linearization.
  std::unordered_map<const VarNode *, const Object *> last_access_stmt_;
  // Per-buffer set of live epoch ids, computed from the EpochGraph + per-epoch
  // access map by `ComputePerEpochLiveness`. Empty if liveness was not
  // computed (e.g. EpochGraph build skipped). Used by `LinearScanPack` as a
  // strict relaxer on the legacy 1-D interval conflict test.
  std::unordered_map<const VarNode *, std::set<int>> liveness_epochs_by_var_;
  // Per-buffer annotated `Layout` collected from `Block` annotations under
  // `attr::kLayoutMap`. Populated by `PlanReuse` via `LayoutSignatureCollector`
  // and consulted by `LayoutAliasIncompatible` to gate the per-epoch relaxer:
  // two buffers with non-trivial *and* structurally distinct layouts must not
  // alias, even when their live-epoch sets are disjoint.
  std::unordered_map<const VarNode *, Layout> layout_sigs_;
  // Approximate statement position during rewrite.
  int stmt_visit_index_{0};
};

Stmt MergeSharedMemoryAllocations(Stmt stmt, bool merge_static_smem,
                                  bool enable_aggressive_merge,
                                  int align_bytes = 16, bool verbose = false) {
  AllocateCollector collector;
  collector(stmt);
  if (collector.dyn_shmem_allocs_.size() > 1) {
    SharedMemoryRewriter rewriter(collector.dyn_shmem_allocs_, true, verbose,
                                  align_bytes);
    rewriter.PlanReuse(stmt, true, enable_aggressive_merge);
    stmt = rewriter(std::move(stmt));
  }
  if (merge_static_smem && collector.static_shmem_allocs_.size() > 1) {
    SharedMemoryRewriter rewriter(collector.static_shmem_allocs_, false,
                                  verbose, align_bytes);
    rewriter.PlanReuse(stmt, false, enable_aggressive_merge);
    stmt = rewriter(std::move(stmt));
  }
  return stmt;
}

using namespace tir::transform;

namespace transform {

Pass MergeSharedMemoryAllocations(bool enable_aggressive_merge = false,
                                  int align_bytes = 16) {
  auto pass_func = [enable_aggressive_merge, align_bytes](
                       PrimFunc f, const IRModule &m, PassContext ctx) {
    bool merge_static_smem =
        ctx->GetConfig<Bool>("tir.merge_static_smem", Bool(false)).value();
    bool debug_merge_shared_memory_allocations =
        ctx->GetConfig<Bool>(kDebugMergeSharedMemoryAllocations, Bool(false))
            .value();
    auto *n = f.CopyOnWrite();
    n->body = tl::MergeSharedMemoryAllocations(
        std::move(n->body), merge_static_smem, enable_aggressive_merge,
        align_bytes, debug_merge_shared_memory_allocations);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MergeSharedMemoryAllocations",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MergeSharedMemoryAllocations",
                        MergeSharedMemoryAllocations);
}

} // namespace transform
} // namespace tl
} // namespace tvm
