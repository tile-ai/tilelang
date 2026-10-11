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
 * \file merge_ub_common.h
 * \brief Allocation planning and rewriting shared by Ascend MergeUB passes.
 */

#pragma once

#include <tvm/ffi/container/array.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "op/builtin.h"
#include "runtime/thread_storage_scope.h"
#include "support/check.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {
namespace merge_ub {

using AllocationMap =
    std::unordered_map<const tirx::VarNode *, const tirx::AllocBufferNode *>;
using AllocationOrder = std::vector<const tirx::VarNode *>;
using BufferByteOffsetMap = std::unordered_map<const tirx::VarNode *, PrimExpr>;

struct MemoryPlan {
  BufferByteOffsetMap byte_offsets;
  PrimExpr total_size;
};

inline bool IsDynamicSharedMemory(tirx::Var buffer_var) {
  if (!buffer_var->type_annotation.as<PointerTypeNode>())
    return false;
  runtime::StorageScope storage_scope =
      runtime::StorageScope::Create(GetPtrStorageScope(std::move(buffer_var)));
  return storage_scope.rank == runtime::StorageRank::kShared &&
         storage_scope.tag == ".dyn";
}

inline bool IsScopeMemory(tirx::Var buffer_var,
                          const std::string &target_scope) {
  if (!buffer_var->type_annotation.as<PointerTypeNode>())
    return false;
  return GetPtrStorageScope(std::move(buffer_var)) == target_scope;
}

inline int AscendScopeAlignBytes(const std::string &scope) {
  if (scope == "shared.l0c" || scope == "shared.l0c.dyn")
    return 1024;
  if (scope == "shared.l1" || scope == "shared.l1.dyn" ||
      scope == "shared.l0a" || scope == "shared.l0a.dyn" ||
      scope == "shared.l0b" || scope == "shared.l0b.dyn")
    return 512;
  return 0;
}

inline int AscendScopeAlignBytes(const tirx::AllocBufferNode *alloc) {
  if (alloc == nullptr)
    return 0;
  return AscendScopeAlignBytes(GetPtrStorageScope(alloc->buffer->data));
}

inline int64_t PackedBytesFromElems(int64_t elements, int elem_bits) {
  return (elem_bits % 8 == 0) ? elements * (elem_bits / 8)
                              : (elements * elem_bits + 7) / 8;
}

inline PrimExpr PackedBytesFromElems(PrimExpr elements, int elem_bits) {
  if (elem_bits % 8 == 0)
    return elements * tirx::make_const(elements.dtype(), elem_bits / 8);
  return tirx::FloorDiv(elements *
                                tirx::make_const(elements.dtype(), elem_bits) +
                            tirx::make_const(elements.dtype(), 7),
                        tirx::make_const(elements.dtype(), 8));
}

inline PrimExpr PackedElemsFromBytes(PrimExpr bytes, int elem_bits) {
  return tirx::FloorDiv(bytes * tirx::make_const(bytes.dtype(), 8),
                        tirx::make_const(bytes.dtype(), elem_bits));
}

inline PrimExpr AlignPrimExpr(const PrimExpr &value, int alignment) {
  if (alignment <= 1)
    return value;
  DataType dtype = value.dtype();
  ICHECK(dtype.is_int() || dtype.is_uint())
      << "Expected integer dtype for alignment, but got " << dtype;
  PrimExpr align_expr = tirx::make_const(dtype, alignment);
  PrimExpr adjust = tirx::make_const(dtype, alignment - 1);
  return indexdiv(value + adjust, align_expr) * align_expr;
}

inline PrimExpr GetBufferSizeBytes(const tirx::AllocBufferNode *alloc,
                                   DataType result_dtype) {
  DataType size_dtype = DataType::Int(32);
  if (!alloc->buffer->shape.empty())
    size_dtype = alloc->buffer->shape[0].dtype();
  if (!size_dtype.is_int() && !size_dtype.is_uint())
    size_dtype = DataType::Int(32);

  PrimExpr total_elems = tirx::make_const(size_dtype, 1);
  for (const PrimExpr &extent : alloc->buffer->shape) {
    PrimExpr normalized_extent = extent;
    if (normalized_extent.dtype() != size_dtype)
      normalized_extent = cast(size_dtype, normalized_extent);
    total_elems = total_elems * normalized_extent;
  }
  int elem_bits = alloc->buffer->dtype.bits() * alloc->buffer->dtype.lanes();
  PrimExpr size_bytes = PackedBytesFromElems(total_elems, elem_bits);
  if (size_bytes.dtype() != result_dtype)
    size_bytes = cast(result_dtype, size_bytes);
  return size_bytes;
}

class AllocateCollector : public tirx::StmtExprVisitor {
public:
  void VisitStmt_(const tirx::AllocBufferNode *op) final {
    if (IsDynamicSharedMemory(op->buffer->data)) {
      const tirx::VarNode *key = op->buffer->data.get();
      if (dyn_shmem_allocs.emplace(key, op).second)
        dyn_shmem_order.push_back(key);
    } else {
      std::string scope = GetPtrStorageScope(op->buffer->data);
      for (const std::string &candidate :
           {"shared.l0a", "shared.l0b", "shared.l0c", "shared.l1"}) {
        if (scope != candidate)
          continue;
        const tirx::VarNode *key = op->buffer->data.get();
        if (ascend_scope_allocs[candidate].emplace(key, op).second)
          ascend_scope_order[candidate].push_back(key);
        break;
      }
    }
    tirx::StmtExprVisitor::VisitStmt_(op);
  }

  AllocationMap dyn_shmem_allocs;
  std::unordered_map<std::string, AllocationMap> ascend_scope_allocs;
  AllocationOrder dyn_shmem_order;
  std::unordered_map<std::string, AllocationOrder> ascend_scope_order;
};

inline MemoryPlan BuildSequentialPlan(const AllocationMap &allocations,
                                      int align_bytes) {
  MemoryPlan plan;
  if (allocations.empty()) {
    plan.total_size = tirx::make_const(DataType::Int(64), 0);
    return plan;
  }

  std::vector<const tirx::VarNode *> sorted_vars;
  sorted_vars.reserve(allocations.size());
  for (const auto &[var, _] : allocations)
    sorted_vars.push_back(var);
  std::sort(sorted_vars.begin(), sorted_vars.end(),
            [](const tirx::VarNode *lhs, const tirx::VarNode *rhs) {
              return lhs->name_hint < rhs->name_hint;
            });

  DataType offset_dtype = DataType::Int(32);
  PrimExpr cursor = tirx::make_const(offset_dtype, 0);
  PrimExpr total_size = tirx::make_const(offset_dtype, 0);
  for (const tirx::VarNode *var : sorted_vars) {
    const tirx::AllocBufferNode *alloc = allocations.at(var);
    int alignment = std::max(align_bytes, AscendScopeAlignBytes(alloc));
    cursor = AlignPrimExpr(cursor, alignment);
    plan.byte_offsets[var] = cursor;
    PrimExpr buffer_end = cursor + GetBufferSizeBytes(alloc, offset_dtype);
    total_size = max(total_size, buffer_end);
    cursor = buffer_end;
  }
  plan.total_size = AlignPrimExpr(total_size, align_bytes);
  return plan;
}

class SharedMemoryRewriter : public tirx::StmtExprMutator {
public:
  SharedMemoryRewriter(const AllocationMap &allocations, bool verbose = false,
                       int align_bytes = 0, const std::string &merged_name = "",
                       const std::string &merged_scope = "",
                       const std::string &target_scope = "")
      : allocations_(allocations), verbose_(verbose), align_bytes_(align_bytes),
        target_scope_(target_scope) {
    if (!merged_name.empty() && !merged_scope.empty()) {
      merged_buf_var_ = tirx::Var(
          merged_name, PointerType(PrimType(DataType::UInt(8)), merged_scope));
    }
  }

  void SetPlan(MemoryPlan plan) {
    buffer_byte_offsets_ = std::move(plan.byte_offsets);
    merged_alloc_size_ = std::move(plan.total_size);
  }

  void PlanNoReuse() {
    SetPlan(BuildSequentialPlan(allocations_, align_bytes_));
    if (!verbose_)
      return;
    LOG(DEBUG) << "No-Reuse Memory Allocation Plan for Ascend UB:";
    LOG(DEBUG) << "  Total Merged Size: " << merged_alloc_size_ << " bytes";
    for (const auto &[var, offset] : buffer_byte_offsets_)
      LOG(DEBUG) << "    Buffer " << var->name_hint << " offset=" << offset;
  }

private:
  tirx::Stmt VisitStmt_(const tirx::AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent && !allocated_) {
      if (verbose_) {
        LOG(DEBUG) << "Memory Allocation Plan for Ascend UB:";
        LOG(DEBUG) << "  Merged Buffer Name: " << merged_buf_var_->name_hint;
        LOG(DEBUG) << "  Total Merged Size: " << merged_alloc_size_ << " bytes";
        for (const auto &[var, byte_offset] : buffer_byte_offsets_) {
          auto allocation = allocations_.find(var);
          if (allocation == allocations_.end())
            continue;
          const tirx::AllocBufferNode *alloc = allocation->second;
          int elem_bits =
              alloc->buffer->dtype.bits() * alloc->buffer->dtype.lanes();
          PrimExpr buffer_size_bytes =
              PackedBytesFromElems(alloc->buffer->shape[0], elem_bits);
          LOG(DEBUG) << "    Buffer: " << var->name_hint
                     << " (Type: " << alloc->buffer->dtype << ")"
                     << ", Start Offset: " << byte_offset
                     << ", Size: " << buffer_size_bytes << " bytes";
        }
        LOG(DEBUG) << "End of Memory Allocation Plan.";
      }

      allocated_ = true;
      tirx::Buffer merged_buf(merged_buf_var_, DataType::UInt(8),
                              {merged_alloc_size_}, {}, PrimExpr(),
                              merged_buf_var_->name_hint, 0, 0, tirx::kDefault);
      ffi::Array<tirx::Stmt> sequence;
      sequence.push_back(tirx::AllocBuffer(merged_buf));
      sequence.push_back(tirx::StmtExprMutator::VisitStmt(op->body));
      return tirx::AttrStmt(op->node, op->attr_key, op->value,
                            tirx::SeqStmt(std::move(sequence)), op->span);
    }
    return tirx::StmtMutator::VisitStmt_(op);
  }

  tirx::Stmt VisitStmt_(const tirx::SeqStmtNode *op) final {
    ffi::Array<tirx::Stmt> sequence;
    for (const tirx::Stmt &stmt : op->seq) {
      tirx::Stmt rewritten = VisitStmt(stmt);
      if (!rewritten.defined())
        continue;
      if (const auto *evaluate = rewritten.as<tirx::EvaluateNode>()) {
        if (const auto *integer = evaluate->value.as<tirx::IntImmNode>()) {
          if (integer->value == 0)
            continue;
        }
      }
      if (const auto *nested = rewritten.as<tirx::SeqStmtNode>()) {
        sequence.insert(sequence.end(), nested->seq.begin(), nested->seq.end());
      } else {
        sequence.push_back(std::move(rewritten));
      }
    }
    if (sequence.empty())
      return tirx::Evaluate(0);
    return tirx::SeqStmt(std::move(sequence));
  }

  tirx::Stmt VisitStmt_(const tirx::AllocBufferNode *op) final {
    if (IsAppropriateSharedMemory(op->buffer->data))
      return tirx::Evaluate(0);
    return tirx::StmtExprMutator::VisitStmt_(op);
  }

  tirx::Stmt VisitStmt_(const tirx::DeclBufferNode *op) final {
    tirx::DeclBuffer node =
        Downcast<tirx::DeclBuffer>(tirx::StmtExprMutator::VisitStmt_(op));
    tirx::Buffer new_buffer = GetUpdatedBuffer(node->buffer);
    if (!new_buffer.same_as(node->buffer))
      node.CopyOnWrite()->buffer = std::move(new_buffer);
    return node;
  }

  PrimExpr VisitExpr_(const tirx::BufferLoadNode *op) final {
    tirx::BufferLoad node =
        Downcast<tirx::BufferLoad>(tirx::StmtExprMutator::VisitExpr_(op));
    return VisitBufferAccess(std::move(node));
  }

  tirx::Stmt VisitStmt_(const tirx::BufferStoreNode *op) final {
    tirx::BufferStore node =
        Downcast<tirx::BufferStore>(tirx::StmtExprMutator::VisitStmt_(op));
    return VisitBufferAccess(std::move(node));
  }

  template <typename Node> Node VisitBufferAccess(Node node) {
    if (!IsAppropriateSharedMemory(node->buffer->data))
      return node;
    ICHECK_EQ(node->indices.size(), 1)
        << "MergeUBAllocations expects flat memory buffers, and is to be run "
           "after StorageFlatten (TE schedules) or FlattenBuffer (TIR "
           "schedules)";
    ffi::Array<PrimExpr> indices = {
        node->indices[0] +
        GetBufferOffset(node->buffer->data, node->buffer->dtype)};
    auto *writer = node.CopyOnWrite();
    writer->buffer = GetUpdatedBuffer(node->buffer);
    writer->indices = std::move(indices);
    return node;
  }

  tirx::Buffer GetUpdatedBuffer(tirx::Buffer buffer) {
    auto cached = buffer_remap_.find(buffer);
    if (cached != buffer_remap_.end())
      return cached->second;
    tirx::Buffer original = buffer;
    if (IsAppropriateSharedMemory(buffer->data)) {
      ICHECK_EQ(buffer->shape.size(), 1)
          << "Buffer " << buffer << " has shape " << buffer->shape
          << ". MergeUBAllocations expects flat memory buffers, and is to be "
             "run after StorageFlatten (TE schedules) or FlattenBuffer (TIR "
             "schedules)";
      buffer.CopyOnWrite()->data = merged_buf_var_;
    }
    buffer_remap_.emplace(std::move(original), buffer);
    return buffer;
  }

  PrimExpr VisitExpr_(const tirx::CallNode *op) final {
    if (!op->op.same_as(tirx::builtin::tvm_access_ptr()))
      return tirx::StmtExprMutator::VisitExpr_(op);
    ICHECK_EQ(op->args.size(), 5U);
    DataType dtype = op->args[0].dtype();
    tirx::Var buffer = Downcast<tirx::Var>(op->args[1]);
    if (!IsAppropriateSharedMemory(buffer) || !IsManagedAllocation(buffer))
      return tirx::StmtExprMutator::VisitExpr_(op);
    ICHECK(HasBufferOffset(buffer))
        << "Ascend buffer allocation " << buffer->name_hint
        << " was collected for merging but was not assigned an offset.";
    PrimExpr offset = VisitExpr(op->args[2]);
    PrimExpr extent = VisitExpr(op->args[3]);
    return tirx::Call(op->dtype, op->op,
                      {op->args[0], merged_buf_var_,
                       GetBufferOffset(buffer, dtype) + offset, extent,
                       op->args[4]});
  }

  PrimExpr GetBufferOffset(const tirx::Var &buffer_var, DataType dtype) const {
    auto offset = buffer_byte_offsets_.find(buffer_var.get());
    ICHECK(offset != buffer_byte_offsets_.end())
        << "buffer_var = " << buffer_var->name_hint << ", dtype = " << dtype;
    return PackedElemsFromBytes(offset->second, dtype.bits() * dtype.lanes());
  }

  bool HasBufferOffset(const tirx::Var &buffer_var) const {
    return buffer_byte_offsets_.count(buffer_var.get());
  }

  bool IsManagedAllocation(const tirx::Var &buffer_var) const {
    return allocations_.count(buffer_var.get());
  }

  bool IsAppropriateSharedMemory(const tirx::Var &var) const {
    if (!target_scope_.empty())
      return IsScopeMemory(var, target_scope_);
    return IsDynamicSharedMemory(var);
  }

  AllocationMap allocations_;
  bool verbose_{false};
  int align_bytes_{16};
  tirx::Var merged_buf_var_{
      "buf_dyn_shmem", PointerType(PrimType(DataType::UInt(8)), "shared.dyn")};
  std::string target_scope_;
  PrimExpr merged_alloc_size_{0};
  BufferByteOffsetMap buffer_byte_offsets_;
  std::unordered_map<tirx::Buffer, tirx::Buffer, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      buffer_remap_;
  bool allocated_{false};
};

template <typename Planner>
tirx::Stmt MergeAllocations(tirx::Stmt stmt, int align_bytes, bool verbose,
                            bool disable_reuse, const Planner &planner) {
  AllocateCollector collector;
  collector(stmt);
  auto run = [&](SharedMemoryRewriter &rewriter,
                 const AllocationMap &allocations,
                 const AllocationOrder &allocation_order,
                 const std::string &target_scope) {
    if (disable_reuse) {
      rewriter.PlanNoReuse();
    } else {
      rewriter.SetPlan(planner(stmt, allocations, allocation_order,
                               target_scope, align_bytes, verbose));
    }
    stmt = rewriter(std::move(stmt));
  };

  if (collector.dyn_shmem_allocs.size() > 1) {
    SharedMemoryRewriter rewriter(collector.dyn_shmem_allocs, verbose,
                                  align_bytes);
    run(rewriter, collector.dyn_shmem_allocs, collector.dyn_shmem_order, "");
  }

  std::vector<std::string> scopes;
  scopes.reserve(collector.ascend_scope_allocs.size());
  for (const auto &[scope, _] : collector.ascend_scope_allocs)
    scopes.push_back(scope);
  std::sort(scopes.begin(), scopes.end());
  for (const std::string &scope : scopes) {
    const AllocationMap &allocations = collector.ascend_scope_allocs.at(scope);
    if (allocations.size() <= 1)
      continue;
    auto order = collector.ascend_scope_order.find(scope);
    const AllocationOrder empty_order;
    const AllocationOrder &allocation_order =
        order == collector.ascend_scope_order.end() ? empty_order
                                                    : order->second;
    std::string suffix = scope.substr(7);
    SharedMemoryRewriter rewriter(allocations, verbose, align_bytes,
                                  "buf_dyn_" + suffix, scope + ".dyn", scope);
    run(rewriter, allocations, allocation_order, scope);
  }
  return stmt;
}

} // namespace merge_ub
} // namespace tl
} // namespace tvm
