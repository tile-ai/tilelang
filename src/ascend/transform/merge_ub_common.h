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
 * \brief Native allocation identities, sizing and typed Ascend arena views.
 */
#pragma once

#include "ascend/analysis/storage.h"
#include "ascend/ir/task.h"
#include "ir/analyzer.h"
#include "layout/expr_utils.h"
#include "transform/common/storage_size.h"
#include <algorithm>
#include <limits>
#include <string>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl::merge_ub {
using namespace tirx;
using namespace prim;
using namespace ffi;
using AllocationMap = std::unordered_map<const VarObj *, const BindObj *>;
using AllocationOrder = std::vector<const VarObj *>;
using BufferByteOffsetMap = std::unordered_map<const VarObj *, PrimExpr>;

struct MemoryPlan {
  BufferByteOffsetMap byte_offsets;
  PrimExpr total_size{IntImm::Int64(0)};
};

inline TensorVar AllocationTensor(const BindObj *binding) {
  return binding->var.as_or_throw<TensorVar>();
}
inline Expr AllocationValue(const BindObj *binding) {
  const auto *call = binding->value.as<CallObj>();
  return ascend::IsScheduledBind(call)
             ? ascend::ReadScheduledBinding(call).value
             : binding->value;
}
inline String StorageScope(const Var &var) {
  if (auto tensor = var.as<TensorVar>())
    return tensor->scope().name();
  if (auto pointer = var->ty.as<PtrType>())
    return pointer.value()->scope.name();
  return "";
}
inline bool IsDynamicSharedMemory(const Var &var) {
  return StorageScope(var) == "shared.dyn";
}
inline bool IsScopeMemory(const Var &var, const std::string &scope) {
  return StorageScope(var) == scope;
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
inline int AscendScopeAlignBytes(const BindObj *binding) {
  return binding ? AscendScopeAlignBytes(
                       std::string(AllocationTensor(binding).scope().name()))
                 : 0;
}
inline PrimExpr AlignPrimExpr(PrimExpr value, int alignment) {
  TVM_FFI_CHECK_GT(alignment, 0, ValueError);
  value = cast(value, PrimType::Int(64));
  if (auto constant = TryConstInt(value)) {
    TVM_FFI_CHECK(*constant >= 0 &&
                      *constant <=
                          std::numeric_limits<int64_t>::max() - alignment + 1,
                  ValueError)
        << "Ascend arena alignment overflows int64";
    return IntImm::Int64((*constant + alignment - 1) / alignment * alignment);
  }
  return FloorDiv(value + IntImm::Int64(alignment - 1),
                  IntImm::Int64(alignment)) *
         IntImm::Int64(alignment);
}
inline PrimExpr GetBufferSizeBytes(const BindObj *binding,
                                   PrimType result_type = PrimType::Int(64)) {
  return cast(GetBufferStorageSizeBytes(AllocationTensor(binding)),
              result_type);
}

class AllocateCollector {
public:
  void operator()(const Stmt &stmt) {
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        stmt, [&](const BindObj *binding) -> ffi::Expected<ffi::WalkResult> {
          auto tensor = binding->var.as<TensorVar>();
          auto value = AllocationValue(binding);
          const auto *call = value.as<CallObj>();
          if (!tensor || !call || !call->op.same_as(alloc_tensor_op()))
            return ffi::WalkResult::Advance();
          const VarObj *key = tensor->get();
          std::string scope = tensor->scope().name();
          if (scope == "shared.dyn") {
            if (dyn_shmem_allocs.emplace(key, binding).second)
              dyn_shmem_order.push_back(key);
          } else if (scope == "shared.l0a" || scope == "shared.l0b" ||
                     scope == "shared.l0c" || scope == "shared.l1" ||
                     scope == "shared.ss") {
            if (ascend_scope_allocs[scope].emplace(key, binding).second)
              ascend_scope_order[scope].push_back(key);
          }
          return ffi::WalkResult::Advance();
        });
  }
  AllocationMap dyn_shmem_allocs;
  std::unordered_map<std::string, AllocationMap> ascend_scope_allocs;
  AllocationOrder dyn_shmem_order;
  std::unordered_map<std::string, AllocationOrder> ascend_scope_order;
};

inline MemoryPlan
BuildSequentialPlan(const AllocationMap &allocations, int align_bytes,
                    const AllocationOrder &allocation_order = {}) {
  MemoryPlan plan;
  std::vector<const VarObj *> order;
  std::unordered_set<const VarObj *> seen;
  for (const auto *var : allocation_order)
    if (allocations.count(var) && seen.insert(var).second)
      order.push_back(var);
  for (const auto &[var, unused] : allocations)
    if (seen.insert(var).second)
      order.push_back(var);
  std::stable_sort(
      order.begin(), order.end(),
      [](const VarObj *a, const VarObj *b) { return a->name < b->name; });
  PrimExpr cursor = IntImm::Int64(0);
  auto analyzer = CreateTileLangAnalyzer();
  for (const VarObj *var : order) {
    auto binding = allocations.at(var);
    auto tensor = AllocationTensor(binding);
    int alignment = std::max({align_bytes, AscendScopeAlignBytes(binding),
                              tensor.ty()->data_alignment});
    cursor = AlignPrimExpr(cursor, alignment);
    plan.byte_offsets.emplace(var, cursor);
    PrimExpr size = GetBufferSizeBytes(binding);
    auto offset_constant = TryConstInt(cursor),
         size_constant = TryConstInt(size);
    TVM_FFI_CHECK(!offset_constant || !size_constant ||
                      *offset_constant <=
                          std::numeric_limits<int64_t>::max() - *size_constant,
                  ValueError)
        << "Ascend arena size overflows int64";
    cursor = analyzer->Simplify(cursor + size);
  }
  plan.total_size = AlignPrimExpr(cursor, align_bytes);
  return plan;
}

// Allocation shapes can use preceding scalar lets. Inline only immutable,
// non-reading, non-trapping definitions when sizing the kernel-local arena.
// Never hoist a loop variable, a load, or a mutable scalar across its domain.
class ArenaSizeExpander : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  explicit ArenaSizeExpander(const Stmt &stmt) {
    for (const Var &var : UndefinedVars(stmt))
      available_.insert(var.get());
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        stmt, [&](const BindObj *bind) -> ffi::Expected<ffi::WalkResult> {
          if (definitions_.count(bind->var))
            repeated_.insert(bind->var.get());
          definitions_.Set(bind->var, AllocationValue(bind));

          return ffi::WalkResult::Advance();
        });
  }
  UnchangedOr<Expr> Mutate_(const VarObj *op, InplaceMode mode) final {
    if (available_.count(op))
      return Unchanged();
    auto value = definitions_.Get(GetRef<Var>(op));
    TVM_FFI_CHECK(value && op->ty.as<PrimTypeObj>() && !repeated_.count(op) &&
                      active_.insert(op).second,
                  ValueError)
        << "Ascend arena size depends on non-hoistable local variable "
        << op->name;
    bool safe = true;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        *value, [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
          if (node.as<TensorLoadObj>() || node.as<CallObj>())
            safe = false;
          auto divisor = [&](const PrimExpr &d) {
            auto c = TryConstInt(d);
            safe &= c && *c != 0 && *c != -1;
          };
          if (const auto *n = node.as<DivObj>())
            divisor(n->b);
          if (const auto *n = node.as<ModObj>())
            divisor(n->b);
          if (const auto *n = node.as<FloorDivObj>())
            divisor(n->b);
          if (const auto *n = node.as<FloorModObj>())
            divisor(n->b);
          return ffi::WalkResult::Advance();
        });
    TVM_FFI_CHECK(safe, ValueError) << "Ascend arena size cannot hoist a "
                                       "memory read or trapping expression";
    Expr result = Mutate(*value, mode).ValueOrUnchanged(*value);
    active_.erase(op);
    return result;
  }

private:
  Map<Var, Expr> definitions_;
  std::unordered_set<const VarObj *> available_, active_, repeated_;
};

class SharedMemoryRewriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  Map<Var, Expr> replacements;
  UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    if (auto value = replacements.Get(op->var)) {
      Bind result(op->var, *value, op->loc);
      if (ascend::IsScheduledBind(op)) {
        auto marker = ascend::ReadScheduledBinding(op->value.as<CallObj>());
        return ascend::MakeScheduledBinding(result, marker.stage, marker.task);
      }
      return result;
    }
    return StmtExprMutator::Mutate_(op, mode);
  }
};

template <typename Planner>
Stmt MergeAllocations(Stmt stmt, int align_bytes, bool verbose,
                      bool disable_reuse, const Planner &planner) {
  TVM_FFI_CHECK(align_bytes > 0 && !(align_bytes & (align_bytes - 1)) &&
                    align_bytes <= (1 << 27),
                ValueError)
      << "Ascend arena alignment must be a power of two in [1, 2^27]";
  AllocateCollector collector;
  collector(stmt);
  auto rewriter = make_object<SharedMemoryRewriter>();
  auto expander = make_object<ArenaSizeExpander>(stmt);
  auto analyzer = CreateTileLangAnalyzer();
  Array<Stmt> prefix;
  auto run = [&](const AllocationMap &allocations, const AllocationOrder &order,
                 const std::string &target_scope) {
    if (allocations.size() <= 1)
      return;
    MemoryPlan plan = disable_reuse || target_scope == "shared.ss"
                          ? BuildSequentialPlan(allocations, align_bytes, order)
                          : planner(stmt, allocations, order, target_scope,
                                    align_bytes, verbose);
    const std::string scope =
        target_scope.empty() ? "shared.dyn" : target_scope + ".dyn";
    PrimExpr size = expander->Mutate(plan.total_size, InplaceMode::kDisallow)
                        .ValueOrUnchanged(plan.total_size)
                        .template as_or_throw<PrimExpr>();
    size = analyzer->Simplify(max(size, IntImm::Int64(1)));
    int alignment = align_bytes;
    Map<String, Any> arena_attrs;
    for (const auto &[var, binding] : allocations) {
      auto tensor = AllocationTensor(binding);
      alignment = std::max({alignment, AscendScopeAlignBytes(binding),
                            tensor.ty()->data_alignment});
      const Call call = AllocationValue(binding).template as_or_throw<Call>();
      TVM_FFI_CHECK(
          call->args.empty() ||
              (call->args.size() == 1 && call->args[0]
                                             .template as_or_throw<tvm::Tuple>()
                                             ->fields.empty()),
          ValueError)
          << "Ascend arena merge requires placement coordinates to be lowered "
             "first";
      if (call->attrs) {
        for (const auto &[key, value] :
             call->attrs.value().template as_or_throw<DictAttrs>()->dict) {
          if (key == "tl.source_block_alloc") {
            const auto origin = value.template as<IntImm>();
            TVM_FFI_CHECK(origin && origin.value()->value == 1, ValueError)
                << "tl.source_block_alloc must be the source allocation marker";
            // CPU launch preparation uses this source-origin hint to place
            // per-iteration allocations. Ascend has already materialized its
            // lexical allocations and scheduled their uses before arena
            // planning. The selected memory plan owns their lifetimes; the
            // replacement arena must not inherit the old source hint.
            continue;
          }
          TVM_FFI_CHECK(key == tirx::attr::kVolatile.name, ValueError)
              << "Ascend arena merge must consume allocation annotation " << key
              << " first";
          if (value.template cast<bool>())
            arena_attrs.Set(key, true);
        }
      }
    }
    TensorType type({size}, PrimType::UInt(8), runtime::MemoryScope::Get(scope),
                    alignment);
    TensorVar arena(target_scope.empty() ? "buf_dyn_shmem"
                                         : "buf_dyn_" + target_scope.substr(7),
                    type);
    Optional<Attrs> attrs = arena_attrs.empty()
                                ? Optional<Attrs>()
                                : Optional<Attrs>(DictAttrs(arena_attrs));
    prefix.push_back(
        Bind(arena, Call(type, alloc_tensor_op(), {}, attrs, {type})));
    for (const auto &[var, binding] : allocations) {
      auto offset = plan.byte_offsets.find(var);
      TVM_FFI_CHECK(offset != plan.byte_offsets.end(), ValueError)
          << "Collected Ascend allocation was not assigned a byte offset: "
          << var->name;
      auto tensor = AllocationTensor(binding);
      PrimExpr bytes = expander->Mutate(offset->second, InplaceMode::kDisallow)
                           .ValueOrUnchanged(offset->second)
                           .template as_or_throw<PrimExpr>();
      Expr pointer = Call(arena.ty()->DataPtrType(), ptr_byte_offset_op(),
                          {arena.data(), bytes});
      // Views keep their original type, including dtype, shape, scope and
      // layout. Byte offsets belong to the backing pointer exactly once.
      pointer =
          tirx::reinterpret(tensor.ty()->DataPtrType(), pointer, binding->loc);
      rewriter->replacements.Set(tensor,
                                 Call(tensor.ty(), decl_tensor_op(), {pointer},
                                      {}, {tensor.ty()}, binding->loc));
      if (verbose)
        LOG(INFO) << "Ascend arena " << scope << " " << var->name
                  << " offset=" << bytes;
    }
  };
  run(collector.dyn_shmem_allocs, collector.dyn_shmem_order, "");
  std::vector<std::string> scopes;
  for (const auto &[scope, unused] : collector.ascend_scope_allocs)
    scopes.push_back(scope);
  std::sort(scopes.begin(), scopes.end());
  for (const auto &scope : scopes)
    run(collector.ascend_scope_allocs.at(scope),
        collector.ascend_scope_order.at(scope), scope);
  if (prefix.empty())
    return stmt;
  Stmt result =
      rewriter->Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(stmt);
  for (const Stmt &child : SeqStmt(result)->seq)
    prefix.push_back(child);
  return SeqStmt(prefix, stmt->loc);
}
} // namespace tvm::tl::merge_ub
