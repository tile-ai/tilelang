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
#include <limits>
#include <optional>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../op/builtin.h"
#include "../target/utils.h"
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
class SharedMemLinearAccessPatternFinder final : public StmtExprVisitor {
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
  };

  void UpdateStmtAttr(const Object *stmt, size_t level) {
    if (stmt_attrs_.find(stmt) == stmt_attrs_.end()) {
      stmt_attrs_[stmt] = StmtAttr{level};
    } else {
      stmt_attrs_[stmt].level = level;
    }
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

  void VisitStmt_(const BufferStoreNode *op) final {
    scope_.push_back(StmtEntry());
    // visit subexpr
    StmtExprVisitor::VisitStmt_(op);
    // Add write access.
    const VarNode *buf = op->buffer->data.get();
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.alloc) {
      ICHECK_LT(it->second.level, scope_.size());
      if (IsAppropriateSharedMemory(tvm::ffi::GetRef<Var>(buf))) {
        // set into scope_.size() - 1 for aggressive memory reuse
        auto enable_aggressive_merge = enable_aggressive_merge_;
        if (enable_aggressive_merge) {
          scope_[scope_.size() - 1].touched.push_back(buf);
        } else {
          scope_[it->second.level].touched.push_back(buf);
        }
      }
    }

    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (!e.touched.empty()) {
      e.stmt = op;
      UpdateStmtAttr(op, scope_level_);
      linear_seq_.push_back(e);
    }
  }

  void VisitStmt_(const EvaluateNode *op) final {
    scope_.push_back(StmtEntry());
    // visit subexpr
    StmtExprVisitor::VisitStmt_(op);
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (!e.touched.empty()) {
      e.stmt = op;
      UpdateStmtAttr(op, scope_level_);
      linear_seq_.push_back(e);
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    // Add write access.
    StmtExprVisitor::VisitExpr_(op);
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
        auto enable_aggressive_merge = enable_aggressive_merge_;
        if (enable_aggressive_merge) {
          scope_[scope_.size() - 1].touched.push_back(buf);
        } else {
          // When the access happens in the same scope frame as the allocation
          // we attribute it to that frame instead of the outer parent.  This
          // keeps the liveness window tight while still accounting for nested
          // scopes that legitimately touch the buffer deeper in the tree.
          size_t access_level = std::min(it->second.level, scope_.size() - 1);
          scope_[access_level].touched.push_back(buf);
        }
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
        auto enable_aggressive_merge = enable_aggressive_merge_;
        if (enable_aggressive_merge) {
          scope_[scope_.size() - 1].touched.push_back(buf);
        } else {
          // Attribute same-level uses to the allocation frame, mirroring the
          // BufferLoad handling to keep reuse decisions consistent.
          size_t access_level = std::min(it->second.level, scope_.size() - 1);
          scope_[access_level].touched.push_back(buf);
        }
      }
    }
  }

  template <typename T> void VisitNewScope(const T *op) {
    scope_.push_back(StmtEntry());
    StmtEntry e;
    e.stmt = op;
    UpdateStmtAttr(op, scope_level_);
    int64_t begin_index = static_cast<int64_t>(linear_seq_.size());
    // before scope.
    linear_seq_.push_back(e);
    StmtExprVisitor::VisitStmt_(op);
    // after scope.
    e.touched = std::move(scope_.back().touched);
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
      in_thread_env_ = true;
      VisitNewScope(op);
      in_thread_env_ = false;
    } else if (op->attr_key == tir::attr::extern_scope) {
      VisitNewScope(op);
    } else if (op->attr_key == tir::attr::virtual_thread) {
      VisitNewScope(op);
    } else if (op->attr_key == "kWarpSpecializationScope") {
      IfThenElse body = Downcast<IfThenElse>(op->body);
      this->VisitStmt(body->then_case);
      this->VisitStmt(body->else_case.value());
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const IfThenElseNode *op) final { VisitNewScope(op); }

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

private:
  // Wrapper function to determine if the shared memory allocation for a
  // variable is appropriate.
  bool IsAppropriateSharedMemory(const Var &var) {
    return is_dynamic_ ? IsDynamicSharedMemory(var) : IsStaticSharedMemory(var);
  }
  // Whether do dynamic analysis.
  bool is_dynamic_{true};
  // Whether do aggressive merge.
  bool enable_aggressive_merge_{false};
  // Whether do verbose logging.
  bool verbose_{false};
  // Whether already in thread env.
  bool in_thread_env_{false};
  // The scope stack.
  std::vector<StmtEntry> scope_;
  // The size of the scope.
  size_t scope_level_{0};
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
      const int alignment = TargetIsHopper(target) ? 1024 : 16;
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
    // First compute liveness over the flattened schedule, then feed it into the
    // arena packer.
    this->LivenessAnalysis(finder.linear_seq_, finder.stmt_attrs_);
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
    auto node = Downcast<DeclBuffer>(StmtExprMutator::VisitStmt_(op));
    auto new_buf = GetUpdatedBuffer(node->buffer);
    if (!new_buf.same_as(node->buffer)) {
      node.CopyOnWrite()->buffer = new_buf;
    }
    return std::move(node);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    return VisitBufferAccess(std::move(node));
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    return VisitBufferAccess(std::move(node));
  }

  template <typename Node> Node VisitBufferAccess(Node node) {
    if (IsAppropriateSharedMemory(node->buffer->data)) {
      ICHECK_EQ(node->indices.size(), 1)
          << "MergeSharedMemoryAllocations expects flat memory buffers, "
          << "and is to be run after "
          << "StorageFlatten (TE schedules) or FlattenBuffer (TIR schedules)";
      Array<PrimExpr> indices = {
          node->indices[0] +
          this->GetBufferOffset(node->buffer->data, node->buffer->dtype)};

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
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5U);
      DataType dtype = op->args[0].dtype();
      Var buffer = Downcast<Var>(op->args[1]);
      if (!IsAppropriateSharedMemory(buffer)) {
        return StmtExprMutator::VisitExpr_(op);
      }
      PrimExpr extra_offset = GetBufferOffset(buffer, dtype);

      PrimExpr offset = this->VisitExpr(op->args[2]);
      PrimExpr extent = this->VisitExpr(op->args[3]);
      return Call(op->dtype, op->op,
                  {op->args[0], merged_buf_var_, extra_offset + offset, extent,
                   op->args[4]});
    } else if (op->op.same_as(builtin::ptx_cp_async())) {
      ICHECK((op->args.size() == 5U) || (op->args.size() == 6U));
      DataType dtype = op->dtype;
      Var buffer = Downcast<Var>(op->args[0]);
      if (!IsAppropriateSharedMemory(buffer)) {
        return StmtExprMutator::VisitExpr_(op);
      }
      PrimExpr extra_offset = GetBufferOffset(buffer, dtype);
      PrimExpr offset = this->VisitExpr(op->args[1]);
      // the dst shared memory is a byte buffer generated by merging shared
      // memory. we need to multiply the offset index by the byte size of the
      // original value dtype, to get the correct offset of merged shared
      // buffer.
      int index_factor = dtype.bytes();
      if (op->args.size() == 5)
        return Call(dtype, op->op,
                    {merged_buf_var_,
                     mul(extra_offset + offset, PrimExpr(index_factor)),
                     op->args[2], op->args[3], op->args[4]});
      else
        return Call(dtype, op->op,
                    {merged_buf_var_,
                     mul(extra_offset + offset, PrimExpr(index_factor)),
                     op->args[2], op->args[3], op->args[4], op->args[5]});
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  PrimExpr GetBufferOffset(const Var &buffer_var, DataType dtype) {
    auto it = buffer_byte_offsets_.find(buffer_var.get());
    ICHECK(it != buffer_byte_offsets_.end())
        << "buffer_var = " << buffer_var->name_hint << ", dtype = " << dtype;
    return indexdiv(it->second, dtype.bytes() * dtype.lanes());
  }

  // Wrapper function to determine if the shared memory allocation for a
  // variable is appropriate.
  bool IsAppropriateSharedMemory(const Var &var) {
    return is_dynamic_ ? IsDynamicSharedMemory(var) : IsStaticSharedMemory(var);
  }

  using StmtEntry = SharedMemLinearAccessPatternFinder::StmtEntry;
  using StmtAttr = SharedMemLinearAccessPatternFinder::StmtAttr;

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
    DataType size_dtype{DataType::Int(32)};
  };

  // Interval describing the liveness window of a (constant-sized) allocation.
  struct Interval {
    int start{0};
    int end{0};
    size_t size_bytes{0};
    int alignment{0};
    const VarNode *var{nullptr};
  };

  // Result of a linear-scan arena packing.  Offsets contain the byte offset for
  // each constant-sized buffer, arena_size is the total constant footprint.
  struct ArenaPlan {
    size_t arena_size{0};
    std::unordered_map<const VarNode *, size_t> offsets;
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

  struct FreeBlock {
    size_t offset{0};
    size_t size{0};
  };

  class FreeList {
  public:
    std::optional<size_t> Allocate(size_t need, size_t alignment) {
      // Best-fit search: pick the slot that wastes the least space after
      // alignment.
      int best = -1;
      size_t best_waste = std::numeric_limits<size_t>::max();
      for (int i = 0, n = static_cast<int>(blocks_.size()); i < n; ++i) {
        size_t aligned = AlignUpSize(blocks_[i].offset, alignment);
        size_t head = aligned - blocks_[i].offset;
        if (head <= blocks_[i].size && (blocks_[i].size - head) >= need) {
          size_t waste = blocks_[i].size - head - need;
          if (waste < best_waste) {
            best_waste = waste;
            best = i;
          }
        }
      }
      if (best < 0) {
        return std::nullopt;
      }
      FreeBlock blk = blocks_[best];
      size_t aligned = AlignUpSize(blk.offset, alignment);
      size_t head = aligned - blk.offset;
      size_t tail = blk.size - head - need;
      blocks_.erase(blocks_.begin() + best);
      if (head) {
        blocks_.push_back({blk.offset, head});
      }
      if (tail) {
        blocks_.push_back({aligned + need, tail});
      }
      Normalize();
      return aligned;
    }

    void Free(size_t offset, size_t size) {
      if (size == 0)
        return;
      blocks_.push_back({offset, size});
      Normalize();
    }

  private:
    void Normalize() {
      if (blocks_.empty())
        return;
      std::sort(blocks_.begin(), blocks_.end(),
                [](const FreeBlock &a, const FreeBlock &b) {
                  return a.offset < b.offset;
                });
      std::vector<FreeBlock> merged;
      merged.reserve(blocks_.size());
      for (const FreeBlock &blk : blocks_) {
        if (merged.empty()) {
          merged.push_back(blk);
          continue;
        }
        FreeBlock &last = merged.back();
        size_t last_end = last.offset + last.size;
        if (blk.offset <= last_end) {
          size_t blk_end = blk.offset + blk.size;
          if (blk_end > last_end) {
            last.size = blk_end - last.offset;
          }
        } else {
          merged.push_back(blk);
        }
      }
      blocks_ = std::move(merged);
    }

    std::vector<FreeBlock> blocks_;
  };

  struct ActiveInterval {
    int end{0};
    size_t offset{0};
    size_t size{0};
    const VarNode *var{nullptr};
    bool operator>(const ActiveInterval &other) const {
      return end > other.end;
    }
  };

  static ArenaPlan LinearScanPack(std::vector<Interval> intervals) {
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
                return lhs.var < rhs.var;
              });

    std::priority_queue<ActiveInterval, std::vector<ActiveInterval>,
                        std::greater<ActiveInterval>>
        active;
    FreeList freelist;
    size_t arena_top = 0;
    std::unordered_map<const VarNode *, size_t> offsets;

    // Expire intervals that end before or at program counter `pc`.
    auto retire = [&](int pc) {
      while (!active.empty() && active.top().end <= pc) {
        const ActiveInterval top = active.top();
        active.pop();
        freelist.Free(top.offset, top.size);
      }
    };

    for (const Interval &interval : intervals) {
      retire(interval.start);
      size_t offset = 0;
      // Try to recycle previously freed memory first; fall back to bumping the
      // arena.
      if (auto slot =
              freelist.Allocate(interval.size_bytes, interval.alignment)) {
        offset = slot.value();
      } else {
        offset = AlignUpSize(arena_top, interval.alignment);
        arena_top = offset + interval.size_bytes;
      }
      active.push(ActiveInterval{interval.end, offset, interval.size_bytes,
                                 interval.var});
      offsets[interval.var] = offset;
    }

    return ArenaPlan{arena_top, std::move(offsets)};
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

          for (; stmt_it != gen_kill_seq.end(); ++stmt_it) {
            // Check if next statement has different level
            auto next_it = stmt_it + 1;
            if (next_it == gen_kill_seq.end() ||
                stmt_attrs.at(next_it->stmt).level == gen_level) {
              last_stmt_at_level = stmt_it->stmt;
              break;
            }
          }
          if (last_stmt_at_level) {
            event_map_[last_stmt_at_level].kill.push_back(buffer);
            visited_buffers.insert(buffer);
          }
        } else {
          ++it;
        }
      }
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
    (void)stmt_attrs;

    if (shmem_allocs_.empty()) {
      merged_alloc_size_ = make_const(DataType::Int(64), 0);
      return;
    }

    // Discover the first and last touch for every allocation.
    std::unordered_map<const VarNode *, int> start_index;
    std::unordered_map<const VarNode *, int> end_index;

    for (size_t i = 0; i < seq.size(); ++i) {
      auto it = event_map_.find(seq[i].stmt);
      if (it == event_map_.end())
        continue;
      for (const VarNode *var : it->second.gen) {
        start_index.emplace(var, static_cast<int>(i));
      }
      for (const VarNode *var : it->second.kill) {
        end_index[var] = std::max(end_index[var], static_cast<int>(i) + 1);
      }
    }

    const int seq_len = static_cast<int>(seq.size());
    for (const auto &kv : start_index) {
      if (!end_index.count(kv.first)) {
        end_index[kv.first] = seq_len;
      }
    }

    std::vector<BufInfo> buf_infos;
    buf_infos.reserve(shmem_allocs_.size());
    // Build a BufInfo for all allocations that participate in liveness.
    for (const auto &kv : shmem_allocs_) {
      const VarNode *var = kv.first;
      auto start_it = start_index.find(var);
      if (start_it == start_index.end()) {
        continue;
      }

      BufInfo info;
      info.var = var;
      info.name = var->name_hint;
      info.start = start_it->second;
      info.end = std::max(end_index[var], info.start + 1);
      info.alignment = align_bytes_;
      auto align_it = shmem_alignment_map_.find(var);
      if (align_it != shmem_alignment_map_.end()) {
        info.alignment = std::max(info.alignment, align_it->second);
      }

      const AllocateNode *alloc = kv.second;
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
      Interval interval;
      interval.start = info.start;
      interval.end = info.end;
      interval.size_bytes = static_cast<size_t>(
          std::max<int64_t>(0, info.const_size_bytes.value()));
      interval.alignment = info.alignment;
      interval.var = info.var;
      intervals.push_back(interval);
    }

    ArenaPlan plan = LinearScanPack(std::move(intervals));
    size_t arena_size_const = plan.arena_size;

    if (verbose_) {
      LOG(DEBUG) << "ArenaPlan (constant buffers): arena_size="
                 << arena_size_const;
      for (const auto &kv : plan.offsets) {
        const VarNode *var = kv.first;
        LOG(DEBUG) << "  " << var->name_hint << " -> offset=" << kv.second;
      }
    }

    // Cursor tracks the running byte offset within the merged arena.
    DataType offset_dtype =
        buf_infos.empty() ? DataType::Int(32) : buf_infos.front().size_dtype;
    PrimExpr total_size = make_const(offset_dtype, 0);
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
      PrimExpr buf_end = offset_expr + CastToOffset(info.size_expr);
      total_size = max(total_size, buf_end);
    }

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
      // Sanity check for overlapping constant buffers.
      for (size_t i = 0; i < buf_infos.size(); ++i) {
        const BufInfo &a = buf_infos[i];
        auto a_off_imm = buffer_byte_offsets_.at(a.var).as<IntImmNode>();
        if (!a.const_size_bytes.has_value() || a_off_imm == nullptr)
          continue;
        int64_t a_off = a_off_imm->value;
        int64_t a_end = a_off + a.const_size_bytes.value();
        for (size_t j = i + 1; j < buf_infos.size(); ++j) {
          const BufInfo &b = buf_infos[j];
          auto b_off_imm = buffer_byte_offsets_.at(b.var).as<IntImmNode>();
          if (!b.const_size_bytes.has_value() || b_off_imm == nullptr)
            continue;
          bool live_overlap = !(a.end <= b.start || b.end <= a.start);
          if (!live_overlap)
            continue;
          int64_t b_off = b_off_imm->value;
          int64_t b_end = b_off + b.const_size_bytes.value();
          bool mem_overlap = !(a_end <= b_off || b_end <= a_off);
          if (mem_overlap) {
            overlap_detected = true;
            LOG(WARNING) << "Buffer overlap detected between " << a.name
                         << " and " << b.name << " (lifetime overlap with "
                         << "offset ranges [" << a_off << ", " << a_end
                         << ") and [" << b_off << ", " << b_end << ")).";
          }
        }
      }
    }

    if (overlap_detected) {
      LOG(WARNING) << "Detected overlapping constant buffers; falling back to "
                   << "sequential allocation without reuse.";
      buffer_byte_offsets_.clear();
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
  // The mapping from the original buffer objects to their location in the
  // merged buffer.
  std::unordered_map<const BufferNode *, Buffer> buffer_remap_;
  // The flag indicating whether the merged buffer has been allocated
  bool allocated_{false};
  // Locations of free ops.
  std::unordered_map<const Object *, EventEntry> event_map_;
  // The mapping of buffer bytes alignment
  std::unordered_map<const VarNode *, int> shmem_alignment_map_;
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
