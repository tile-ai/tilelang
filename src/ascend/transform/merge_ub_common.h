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

#include <tvm/ir/analysis.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/region.h>

#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ascend/transform/native_storage.h"
#include "support/check.h"
#include "transform/common/storage_size.h"

namespace tvm::tl::merge_ub {

using AllocationMap = std::unordered_map<const VarNode *, const BindNode *>;
using AllocationOrder = std::vector<const VarNode *>;
using BufferByteOffsetMap = std::unordered_map<const VarNode *, PrimExpr>;

struct MemoryPlan {
  BufferByteOffsetMap byte_offsets;
  PrimExpr total_size{IntImm::Int64(0)};
};

inline std::string StorageScope(const Var &variable) {
  if (const auto *tensor = variable->ty.as<tirx::TensorTypeNode>())
    return tensor->scope.name();
  if (const auto *pointer = variable->ty.as<PtrTypeNode>())
    return pointer->scope.name();
  return "";
}

inline bool IsDynamicSharedMemory(const Var &variable) {
  return StorageScope(variable) == "shared.dyn";
}

inline bool IsScopeMemory(const Var &variable, const std::string &scope) {
  return StorageScope(variable) == scope;
}

inline int AscendScopeAlignBytes(const std::string &scope) {
  if (scope == "shared.ss" || scope == "shared.ss.dyn")
    return 32;
  if (scope == "shared.l0c" || scope == "shared.l0c.dyn")
    return 1024;
  if (scope == "shared.l1" || scope == "shared.l1.dyn" ||
      scope == "shared.l0a" || scope == "shared.l0a.dyn" ||
      scope == "shared.l0b" || scope == "shared.l0b.dyn")
    return 512;
  return 0;
}

inline int AscendScopeAlignBytes(const BindNode *allocation) {
  auto tensor = allocation->var.as_or_throw<tirx::TensorVar>();
  return std::max(AscendScopeAlignBytes(StorageScope(tensor)),
                  tensor.ty()->data_alignment);
}

inline PrimExpr AlignPrimExpr(const PrimExpr &value, int alignment) {
  PrimExpr normalized = ascend::StorageIndex(value);
  return floordiv(normalized + IntImm::Int64(alignment - 1),
                  IntImm::Int64(alignment)) *
         IntImm::Int64(alignment);
}

inline PrimExpr GetBufferSizeBytes(const BindNode *allocation,
                                   PrimType result_type) {
  return prim::cast(
      GetBufferStorageSizeBytes(allocation->var.as_or_throw<tirx::TensorVar>()),
      result_type);
}

struct AllocationPosition {
  const SeqStmtNode *sequence;
  size_t index;
};

class AllocateCollector : public ascend::AnalysisVisitor {
public:
  AllocationMap dyn_shmem_allocs;
  AllocationOrder dyn_shmem_order;
  std::unordered_map<std::string, AllocationMap> ascend_scope_allocs;
  std::unordered_map<std::string, AllocationOrder> ascend_scope_order;
  std::unordered_map<const VarNode *, std::vector<AllocationPosition>> paths;
  std::unordered_map<const SeqStmtNode *, ffi::Array<Var>> parameters;
  std::unordered_map<const VarNode *, const RegionStmtNode *> boundaries;

private:
  void VisitStmt_(const SeqStmtNode *node) final {
    path_.push_back({node, 0});
    for (size_t index = 0; index < node->seq.size(); ++index) {
      path_.back().index = index;
      Visit(node->seq[index]);
    }
    path_.pop_back();
  }

  void VisitStmt_(const ForNode *node) final {
    parameters[node->body.get()] = {node->var};
    Visit(node->body);
  }

  void VisitStmt_(const RegionStmtNode *node) final {
    parameters[node->body.get()] = node->body_params;
    const RegionStmtNode *previous = boundary_;
    if (node->op.same_as(tirx::device_entry_op()) ||
        node->op.same_as(tirx::device_context_op()) ||
        node->op.same_as(tirx::compute_scope_op()))
      boundary_ = node;
    Visit(node->body);
    boundary_ = previous;
  }

  void VisitStmt_(const BindNode *node) final {
    const auto *call = node->value.as<CallNode>();
    if (!call || !call->op.same_as(tirx::alloc_tensor_op()))
      return;
    auto tensor = node->var.as_or_throw<tirx::TensorVar>();
    const VarNode *key = tensor.get();
    std::string scope = StorageScope(tensor);
    if (IsDynamicSharedMemory(tensor)) {
      if (dyn_shmem_allocs.emplace(key, node).second)
        dyn_shmem_order.push_back(key);
    } else if (scope == "shared.l0a" || scope == "shared.l0b" ||
               scope == "shared.l0c" || scope == "shared.l1" ||
               scope == "shared.ss") {
      if (ascend_scope_allocs[scope].emplace(key, node).second)
        ascend_scope_order[scope].push_back(key);
    } else {
      return;
    }
    paths.emplace(key, path_);
    boundaries.emplace(key, boundary_);
  }

  std::vector<AllocationPosition> path_;
  const RegionStmtNode *boundary_{nullptr};
};

inline MemoryPlan BuildSequentialPlan(const AllocationMap &allocations,
                                      int align_bytes,
                                      const AllocationOrder &order) {
  MemoryPlan plan;
  PrimExpr cursor = IntImm::Int64(0);
  for (const VarNode *variable : order) {
    const BindNode *allocation = allocations.at(variable);
    int alignment = std::max(align_bytes, AscendScopeAlignBytes(allocation));
    cursor = AlignPrimExpr(cursor, alignment);
    plan.byte_offsets.insert_or_assign(variable, cursor);
    cursor = cursor + GetBufferSizeBytes(allocation, PrimType::Int(64));
  }
  plan.total_size = AlignPrimExpr(cursor, align_bytes);
  return plan;
}

class SharedMemoryRewriter : public ascend::NativeMutator {
public:
  std::unordered_map<const BindNode *, Stmt> replacements;
  std::unordered_map<const SeqStmtNode *, std::map<size_t, ffi::Array<Stmt>>>
      insertions;

private:
  Stmt VisitStmt_(const BindNode *node) final {
    auto replacement = replacements.find(node);
    return replacement == replacements.end() ? ffi::GetRef<Stmt>(node)
                                             : replacement->second;
  }

  Stmt VisitStmt_(const SeqStmtNode *node) final {
    auto insertions_at = insertions.find(node);
    if (insertions_at == insertions.end())
      return ascend::NativeMutator::VisitStmt_(node);
    ffi::Array<Stmt> statements;
    for (size_t index = 0; index < node->seq.size(); ++index) {
      auto before = insertions_at->second.find(index);
      if (before != insertions_at->second.end())
        for (const Stmt &statement : before->second)
          statements.push_back(statement);
      statements.push_back(VisitStmt(node->seq[index]));
    }
    return SeqStmt(statements, node->loc);
  }
};

template <typename Planner>
tirx::Function MergeAllocations(tirx::Function function, int align_bytes,
                                bool verbose, const Planner &planner,
                                bool ss_only = false) {
  auto collector = ffi::make_object<AllocateCollector>();
  collector->Visit(function->body);
  auto rewriter = ffi::make_object<SharedMemoryRewriter>();
  auto merge = [&](const AllocationMap &all_allocations,
                   const AllocationOrder &all_order, const std::string &scope) {
    if (ss_only && scope != "shared.ss")
      return;
    std::vector<const RegionStmtNode *> boundaries;
    for (const VarNode *variable : all_order) {
      const RegionStmtNode *boundary = collector->boundaries.at(variable);
      if (std::find(boundaries.begin(), boundaries.end(), boundary) ==
          boundaries.end())
        boundaries.push_back(boundary);
    }
    for (const RegionStmtNode *boundary : boundaries) {
      AllocationMap allocations;
      AllocationOrder order;
      for (const VarNode *variable : all_order) {
        if (collector->boundaries.at(variable) == boundary) {
          allocations.emplace(variable, all_allocations.at(variable));
          order.push_back(variable);
        }
      }
      if (allocations.size() <= 1)
        continue;
      std::vector<AllocationPosition> common = collector->paths.at(order[0]);
      for (const VarNode *variable : order) {
        const auto &path = collector->paths.at(variable);
        size_t shared = 0;
        while (shared < common.size() && shared < path.size() &&
               common[shared].sequence == path[shared].sequence) {
          common[shared].index =
              std::min(common[shared].index, path[shared].index);
          ++shared;
        }
        common.resize(shared);
      }
      TVM_FFI_CHECK(!common.empty(), ValueError)
          << "Ascend allocations need a common lexical scope";
      std::unordered_set<const VarNode *> available;
      for (const Var &parameter : function->params)
        available.insert(parameter.get());
      for (const AllocationPosition &position : common) {
        if (auto parameters = collector->parameters.find(position.sequence);
            parameters != collector->parameters.end())
          for (const Var &parameter : parameters->second)
            available.insert(parameter.get());
        for (size_t index = 0; index < position.index; ++index)
          if (const auto *binding =
                  position.sequence->seq[index].as<BindNode>())
            available.insert(binding->var.get());
      }
      int alignment = std::max(align_bytes, AscendScopeAlignBytes(scope));
      int arena_alignment = alignment;
      for (const VarNode *variable : order) {
        const BindNode *allocation = allocations.at(variable);
        const auto *call = allocation->value.as<CallNode>();
        TVM_FFI_CHECK(call->args.empty(), ValueError)
            << "Placed Ascend allocations must be lowered before arena merging";
        TVM_FFI_CHECK(
            !call->attrs ||
                call->attrs.value().as_or_throw<DictAttrs>()->dict.empty(),
            ValueError)
            << "Ascend allocation attributes must be lowered before arena "
               "merging";
        int requested = AscendScopeAlignBytes(allocation);
        TVM_FFI_CHECK(requested <= 0 || (requested <= (1 << 27) &&
                                         (requested & (requested - 1)) == 0),
                      ValueError)
            << "Ascend allocation alignment must be a power of two";
        arena_alignment = std::max(arena_alignment, requested);
      }
      MemoryPlan plan = scope == "shared.ss"
                            ? BuildSequentialPlan(allocations, alignment, order)
                            : planner(function->body.value(), allocations,
                                      order, scope, alignment, verbose);
      sym::Analyzer analyzer;
      PrimExpr bytes = analyzer->Simplify(plan.total_size);
      TVM_FFI_CHECK(SideEffect(bytes) == CallEffectKind::kPure, ValueError)
          << "Ascend arena sizes must be side-effect-free";
      for (const Var &dependency : tvm::UndefinedVars(bytes))
        TVM_FFI_CHECK(available.count(dependency.get()), ValueError)
            << "Cannot hoist Ascend arena size across the definition of "
            << dependency->name;
      std::string arena_scope = scope.empty() ? "shared.dyn" : scope + ".dyn";
      tirx::TensorType arena_type({bytes}, PrimType::UInt(8),
                                  runtime::MemoryScope::Get(arena_scope),
                                  arena_alignment);
      tirx::TensorVar arena("merged_" + arena_scope, arena_type);
      const AllocationPosition &anchor = common.back();
      rewriter->insertions[anchor.sequence][anchor.index].push_back(
          Bind(arena, Call(arena_type, tirx::alloc_tensor_op(), {}, DictAttrs(),
                           {arena_type})));
      for (const VarNode *variable : order) {
        const BindNode *allocation = allocations.at(variable);
        auto tensor = allocation->var.as_or_throw<tirx::TensorVar>();
        PrimExpr offset = analyzer->Simplify(plan.byte_offsets.at(variable));
        Expr pointer = Call(tensor.ty()->DataPtrType(),
                            tirx::ptr_byte_offset_op(), {arena.data(), offset});
        rewriter->replacements.emplace(
            allocation, Bind(tensor,
                             Call(tensor.ty(), tirx::decl_tensor_op(),
                                  {pointer}, std::nullopt, {tensor.ty()}),
                             allocation->loc));
      }
    }
  };
  merge(collector->dyn_shmem_allocs, collector->dyn_shmem_order, "");
  std::vector<std::string> scopes;
  for (const auto &[scope, allocations] : collector->ascend_scope_allocs)
    scopes.push_back(scope);
  std::sort(scopes.begin(), scopes.end());
  for (const std::string &scope : scopes)
    merge(collector->ascend_scope_allocs.at(scope),
          collector->ascend_scope_order.at(scope), scope);
  function.CopyOnWrite()->body =
      rewriter->VisitStmt(function->body.value()).as_or_throw<SeqStmt>();
  return function;
}

} // namespace tvm::tl::merge_ub
