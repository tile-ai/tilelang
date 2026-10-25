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
 * \file merge_ub_allocations.cc
 * \brief Merge Ascend on-chip allocations from a buffer-alias contract.
 */

#include "buffer_alias.h"
#include "merge_ub_common.h"
#include "transform/common/attr.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

using namespace merge_ub;

size_t AlignUp(size_t value, int alignment) {
  if (alignment <= 1)
    return value;
  size_t align = static_cast<size_t>(alignment);
  size_t remainder = value % align;
  return remainder == 0 ? value : value + align - remainder;
}

struct CliqueVertex {
  Var storage;
  size_t size;
  int alignment;
  size_t stable_index;
  int conflict_degree{0};
};

struct AliasClique {
  std::vector<size_t> members;
  size_t size{0};
  int alignment{1};
  size_t stable_index{std::numeric_limits<size_t>::max()};
};

struct CliqueLayout {
  std::vector<size_t> byte_offsets;
  size_t total_size{0};
};

CliqueLayout LayoutCliques(const std::vector<AliasClique> &cliques,
                           int final_alignment) {
  std::vector<size_t> order(cliques.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
    const AliasClique &a = cliques[lhs];
    const AliasClique &b = cliques[rhs];
    if (a.alignment != b.alignment)
      return a.alignment > b.alignment;
    if (a.size != b.size)
      return a.size > b.size;
    if (a.stable_index != b.stable_index)
      return a.stable_index < b.stable_index;
    return lhs < rhs;
  });

  CliqueLayout result;
  result.byte_offsets.resize(cliques.size());
  size_t cursor = 0;
  for (size_t clique_index : order) {
    const AliasClique &clique = cliques[clique_index];
    cursor = AlignUp(cursor, clique.alignment);
    result.byte_offsets[clique_index] = cursor;
    cursor += clique.size;
  }
  result.total_size = AlignUp(cursor, final_alignment);
  return result;
}

struct CliquePartition {
  std::vector<int> assignment;
  std::vector<AliasClique> cliques;
  CliqueLayout layout;
};

class GreedyCliquePartitioner {
public:
  GreedyCliquePartitioner(std::vector<CliqueVertex> vertices,
                          std::vector<std::vector<uint8_t>> conflicts,
                          int final_alignment)
      : vertices_(std::move(vertices)), conflicts_(std::move(conflicts)),
        final_alignment_(final_alignment) {}

  CliquePartition Solve() const { return BuildGreedyPartition(); }

private:
  bool CanJoin(size_t vertex, const AliasClique &clique) const {
    for (size_t member : clique.members) {
      if (conflicts_[vertex][member])
        return false;
    }
    return true;
  }

  static void AddVertex(size_t vertex, const CliqueVertex &item,
                        AliasClique *clique) {
    clique->members.push_back(vertex);
    clique->size = std::max(clique->size, item.size);
    clique->alignment = std::max(clique->alignment, item.alignment);
    clique->stable_index = std::min(clique->stable_index, item.stable_index);
  }

  size_t SelectVertex(const std::vector<int> &assignment,
                      size_t num_cliques) const {
    size_t selected = vertices_.size();
    int best_saturation = -1;
    int best_degree = -1;
    size_t best_size = 0;
    int best_alignment = 0;
    for (size_t vertex = 0; vertex < vertices_.size(); ++vertex) {
      if (assignment[vertex] >= 0)
        continue;
      std::vector<bool> adjacent_colors(num_cliques, false);
      int saturation = 0;
      for (size_t other = 0; other < vertices_.size(); ++other) {
        int color = assignment[other];
        if (color < 0 || !conflicts_[vertex][other] || adjacent_colors[color]) {
          continue;
        }
        adjacent_colors[color] = true;
        ++saturation;
      }
      const CliqueVertex &item = vertices_[vertex];
      bool better = saturation > best_saturation;
      better |=
          saturation == best_saturation && item.conflict_degree > best_degree;
      better |= saturation == best_saturation &&
                item.conflict_degree == best_degree && item.size > best_size;
      better |= saturation == best_saturation &&
                item.conflict_degree == best_degree && item.size == best_size &&
                item.alignment > best_alignment;
      better |= saturation == best_saturation &&
                item.conflict_degree == best_degree && item.size == best_size &&
                item.alignment == best_alignment &&
                (selected == vertices_.size() ||
                 item.stable_index < vertices_[selected].stable_index);
      if (!better)
        continue;
      selected = vertex;
      best_saturation = saturation;
      best_degree = item.conflict_degree;
      best_size = item.size;
      best_alignment = item.alignment;
    }
    ICHECK_LT(selected, vertices_.size());
    return selected;
  }

  CliquePartition BuildGreedyPartition() const {
    CliquePartition result;
    result.assignment.assign(vertices_.size(), -1);
    size_t assigned = 0;
    while (assigned < vertices_.size()) {
      size_t vertex = SelectVertex(result.assignment, result.cliques.size());
      struct Candidate {
        size_t clique;
        bool is_new;
        size_t total_size;
      };
      std::vector<Candidate> candidates;
      for (size_t index = 0; index < result.cliques.size(); ++index) {
        if (!CanJoin(vertex, result.cliques[index]))
          continue;
        std::vector<AliasClique> trial = result.cliques;
        AddVertex(vertex, vertices_[vertex], &trial[index]);
        candidates.push_back(
            {index, false, LayoutCliques(trial, final_alignment_).total_size});
      }
      std::vector<AliasClique> trial = result.cliques;
      AliasClique new_clique;
      AddVertex(vertex, vertices_[vertex], &new_clique);
      trial.push_back(std::move(new_clique));
      candidates.push_back({result.cliques.size(), true,
                            LayoutCliques(trial, final_alignment_).total_size});
      std::stable_sort(candidates.begin(), candidates.end(),
                       [](const Candidate &lhs, const Candidate &rhs) {
                         if (lhs.total_size != rhs.total_size)
                           return lhs.total_size < rhs.total_size;
                         if (lhs.is_new != rhs.is_new)
                           return !lhs.is_new;
                         return lhs.clique < rhs.clique;
                       });
      const Candidate &chosen = candidates.front();
      if (chosen.is_new) {
        AliasClique clique;
        AddVertex(vertex, vertices_[vertex], &clique);
        result.cliques.push_back(std::move(clique));
      } else {
        AddVertex(vertex, vertices_[vertex], &result.cliques[chosen.clique]);
      }
      result.assignment[vertex] = static_cast<int>(chosen.clique);
      ++assigned;
    }
    result.layout = LayoutCliques(result.cliques, final_alignment_);
    return result;
  }

  std::vector<CliqueVertex> vertices_;
  std::vector<std::vector<uint8_t>> conflicts_;
  int final_alignment_;
};

CliquePartition BuildAliasCliquePartition(std::vector<CliqueVertex> vertices,
                                          int final_alignment,
                                          const BufferAliasMap &aliases) {
  ValidateBufferAliasMap(aliases);
  std::vector<std::vector<uint8_t>> conflicts(
      vertices.size(), std::vector<uint8_t>(vertices.size(), 0));
  for (CliqueVertex &vertex : vertices)
    vertex.conflict_degree = 0;
  for (size_t lhs = 0; lhs < vertices.size(); ++lhs) {
    for (size_t rhs = lhs + 1; rhs < vertices.size(); ++rhs) {
      bool conflict = !HasBufferAlias(aliases, vertices[lhs].storage,
                                      vertices[rhs].storage);
      conflicts[lhs][rhs] = conflicts[rhs][lhs] = conflict;
      vertices[lhs].conflict_degree += conflict;
      vertices[rhs].conflict_degree += conflict;
    }
  }
  return GreedyCliquePartitioner(std::move(vertices), std::move(conflicts),
                                 final_alignment)
      .Solve();
}

MemoryPlan BuildAliasMemoryPlan(const AllocationMap &allocations,
                                const AllocationOrder &allocation_order,
                                int align_bytes, bool verbose,
                                const BufferAliasMap &aliases) {
  ValidateBufferAliasMap(aliases);

  std::vector<Var> ordered_vars;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> seen;
  for (const VarNode *var_node : allocation_order) {
    Var var = GetRef<Var>(var_node);
    if (allocations.count(var_node) && seen.insert(var).second)
      ordered_vars.push_back(std::move(var));
  }
  std::vector<Var> tail;
  for (const auto &[var_node, _] : allocations) {
    Var var = GetRef<Var>(var_node);
    if (seen.insert(var).second)
      tail.push_back(std::move(var));
  }
  std::sort(tail.begin(), tail.end(), [](const Var &lhs, const Var &rhs) {
    return lhs->name < rhs->name;
  });
  ordered_vars.insert(ordered_vars.end(), tail.begin(), tail.end());

  auto can_alias = [&](const Var &lhs, const Var &rhs) {
    return HasBufferAlias(aliases, lhs, rhs);
  };
  std::vector<CliqueVertex> constant;
  std::vector<Var> dynamic;
  for (size_t index = 0; index < ordered_vars.size(); ++index) {
    const Var &var = ordered_vars[index];
    const BindNode *alloc = allocations.at(var.get());
    sym::Analyzer analyzer;
    PrimExpr bytes =
        analyzer->Simplify(GetBufferSizeBytes(alloc, PrimType::Int(64)));
    const auto *constant_bytes = bytes.as<IntImmNode>();
    if (!constant_bytes) {
      dynamic.push_back(var);
      continue;
    }
    TVM_FFI_CHECK(constant_bytes->value >= 0, ValueError)
        << "Ascend allocation size cannot be negative";
    auto byte_count = constant_bytes->value.as<int64_t>();
    TVM_FFI_CHECK(byte_count, ValueError)
        << "Ascend allocation size overflows int64";
    size_t size = byte_count.value();
    constant.push_back({var, size,
                        std::max(align_bytes, AscendScopeAlignBytes(alloc)),
                        index});
  }

  MemoryPlan plan;
  size_t arena_top = 0;
  CliquePartition partition;
  if (!constant.empty()) {
    partition = BuildAliasCliquePartition(constant, align_bytes, aliases);
    ICHECK_EQ(partition.assignment.size(), constant.size());
    for (size_t clique_index = 0; clique_index < partition.cliques.size();
         ++clique_index) {
      const AliasClique &clique = partition.cliques[clique_index];
      for (size_t i = 0; i < clique.members.size(); ++i) {
        for (size_t j = i + 1; j < clique.members.size(); ++j) {
          ICHECK(can_alias(constant[clique.members[i]].storage,
                           constant[clique.members[j]].storage))
              << "Buffer-reuse clique contains incompatible storages "
              << constant[clique.members[i]].storage->name << " and "
              << constant[clique.members[j]].storage->name;
        }
      }
    }
    for (size_t vertex = 0; vertex < constant.size(); ++vertex) {
      int clique = partition.assignment[vertex];
      ICHECK_GE(clique, 0);
      ICHECK_LT(static_cast<size_t>(clique),
                partition.layout.byte_offsets.size());
      size_t offset = partition.layout.byte_offsets[clique];
      plan.byte_offsets.insert_or_assign(constant[vertex].storage.get(),
                                         IntImm::Int64(offset));
    }
    arena_top = partition.layout.total_size;
  }

  PrimType offset_dtype = PrimType::Int(64);
  PrimExpr cursor = IntImm(offset_dtype, static_cast<int64_t>(arena_top));
  for (const Var &var : dynamic) {
    const BindNode *alloc = allocations.at(var.get());
    int alignment = std::max(align_bytes, AscendScopeAlignBytes(alloc));
    cursor = AlignPrimExpr(cursor, alignment);
    plan.byte_offsets.insert_or_assign(var.get(), cursor);
    cursor = cursor + GetBufferSizeBytes(alloc, offset_dtype);
  }
  plan.total_size = AlignPrimExpr(cursor, align_bytes);

  if (verbose) {
    LOG(DEBUG) << "Buffer-alias-contract allocation plan:";
    LOG(DEBUG) << "  Total merged size: " << plan.total_size << " bytes";
    LOG(DEBUG) << "  Constant clique partition: " << partition.cliques.size()
               << " slot(s), deterministic DSATUR greedy";
    for (size_t index = 0; index < partition.cliques.size(); ++index) {
      const AliasClique &clique = partition.cliques[index];
      LOG(DEBUG) << "    slot " << index
                 << " offset=" << partition.layout.byte_offsets[index]
                 << " size=" << clique.size
                 << " alignment=" << clique.alignment;
      for (size_t member : clique.members)
        LOG(DEBUG) << "      " << constant[member].storage->name;
    }
    for (const auto &[var, offset] : plan.byte_offsets)
      LOG(DEBUG) << "    " << var->name << " offset=" << offset;
  }
  return plan;
}

class BufferAliasContractExtractor : public ascend::NativeMutator {
public:
  bool FoundContract() const { return found_contract_; }
  const BufferAliasMap &Aliases() const { return aliases_; }
  const AivStorageMap &AivStorages() const { return aiv_storages_; }

  DictAttrs Extract(const DictAttrs &annotations) {
    auto attrs = annotations->dict;
    if (auto annotation = attrs.Get(kAivStorageMap)) {
      for (const auto &[storage, value] : annotation->cast<Map<Var, Any>>()) {
        auto sid = GetIntAnnotation(value);
        TVM_FFI_CHECK(sid && (*sid == 0 || *sid == 1), ValueError)
            << "AIV storage ID must be zero or one";
        auto existing = aiv_storages_.Get(storage);
        TVM_FFI_CHECK(!existing || existing.value() == *sid, ValueError)
            << "Conflicting AIV storage IDs";
        aiv_storages_.Set(storage, static_cast<int>(*sid));
      }
      attrs.erase(kAivStorageMap);
    }
    if (auto annotation = attrs.Get(kBufferAliasMap)) {
      BufferAliasMap local = annotation->cast<BufferAliasMap>();
      ValidateBufferAliasMap(local);
      for (const auto &[storage, compatible] : local) {
        TVM_FFI_CHECK(!aliases_.count(storage), ValueError)
            << "A function contains multiple buffer-alias contracts for "
               "storage "
            << storage;
        AddBufferAliasStorage(&aliases_, storage);
      }
      for (const auto &[storage, compatible] : local)
        for (const Var &other : compatible)
          if (!HasBufferAlias(aliases_, storage, other))
            AddBufferAlias(&aliases_, storage, other);
      found_contract_ = true;
      attrs.erase(kBufferAliasMap);
    }
    return DictAttrs(attrs);
  }

private:
  Stmt VisitStmt_(const ForNode *node) final {
    DictAttrs attrs = Extract(DictAttrs(node->annotations));
    auto loop = ascend::NativeMutator::VisitStmt_(node).as_or_throw<For>();
    loop.CopyOnWrite()->annotations = attrs->dict;
    return loop;
  }

  Stmt VisitStmt_(const RegionStmtNode *node) final {
    DictAttrs attrs = Extract(node->attrs);
    auto region =
        ascend::NativeMutator::VisitStmt_(node).as_or_throw<RegionStmt>();
    region.CopyOnWrite()->attrs = std::move(attrs);
    return region;
  }

  BufferAliasMap aliases_;
  AivStorageMap aiv_storages_;
  bool found_contract_{false};
};

tirx::Function MergeUBAllocations(tirx::Function function, int align_bytes,
                                  bool verbose, bool disable_reuse) {
  auto extractor = make_object<BufferAliasContractExtractor>();
  DictAttrs attrs = extractor->Extract(function->attrs);
  SeqStmt body =
      extractor->VisitStmt(function->body.value()).as_or_throw<SeqStmt>();
  bool has_contract = extractor->FoundContract();
  if (has_contract) {
    function.CopyOnWrite()->attrs = std::move(attrs);
    function.CopyOnWrite()->body = std::move(body);
  }
  BufferAliasMap aliases = extractor->Aliases();
  ValidateBufferAliasMap(aliases);

  AivStorageMap aiv_storages = extractor->AivStorages();
  auto planner = [aliases = std::move(aliases), aiv_storages, disable_reuse](
                     const Stmt &, const AllocationMap &allocations,
                     const AllocationOrder &allocation_order,
                     const std::string &scope, int alignment, bool debug) {
    auto plan = [&](const AllocationMap &group, const AllocationOrder &order) {
      return disable_reuse ? BuildSequentialPlan(group, alignment, order)
                           : BuildAliasMemoryPlan(group, order, alignment,
                                                  debug, aliases);
    };
    if (!scope.empty() || aiv_storages.empty())
      return plan(allocations, allocation_order);
    // Common allocations reserve a prefix in both UBs. Private allocations
    // independently start after that prefix; equal numeric offsets on V0/V1
    // are different physical memory, including when lifetime reuse is disabled.
    std::array<AllocationMap, 3> groups;
    std::array<AllocationOrder, 3> orders;
    for (const auto &[storage, allocation] : allocations) {
      auto sid = aiv_storages.Get(GetRef<Var>(storage));
      int group = sid.has_value() ? sid.value() : 2;
      groups[group].emplace(storage, allocation);
    }
    for (const auto *storage : allocation_order) {
      for (int group = 0; group < 3; ++group)
        if (groups[group].count(storage))
          orders[group].push_back(storage);
    }
    MemoryPlan result = plan(groups[2], orders[2]);
    PrimExpr prefix = AlignPrimExpr(result.total_size, alignment);
    PrimExpr private_size = IntImm(prefix.ty(), 0);
    for (int sid = 0; sid < 2; ++sid) {
      MemoryPlan local = plan(groups[sid], orders[sid]);
      for (const auto &[storage, offset] : local.byte_offsets)
        result.byte_offsets.insert_or_assign(
            storage, prefix + prim::cast(offset, prefix.ty()));
      private_size =
          tvm::max(private_size, prim::cast(local.total_size, prefix.ty()));
    }
    result.total_size = prefix + private_size;
    return result;
  };
  return merge_ub::MergeAllocations(std::move(function), align_bytes, verbose,
                                    planner, !has_contract);
}

} // namespace

using namespace tirx::transform;

namespace transform {

tvm::transform::Pass MergeUBAllocations(int align_bytes = 16,
                                        bool disable_reuse = false) {
  TVM_FFI_CHECK(align_bytes > 0 && align_bytes <= (1 << 27) &&
                    (align_bytes & (align_bytes - 1)) == 0,
                ValueError)
      << "Ascend arena alignment must be a positive power of two";
  auto pass_func = [align_bytes,
                    disable_reuse](tirx::Function function, const IRModule &,
                                   const tvm::transform::PassContext &context) {
    if (!function->body)
      return function;
    bool verbose =
        context->GetConfig<bool>("tl.debug_merge_shared_memory_allocations")
            .value_or(false);
    return tl::MergeUBAllocations(std::move(function), align_bytes, verbose,
                                  disable_reuse);
  };
  return CreateFunctionPass(pass_func, 0, "tl.MergeUBAllocations");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.MergeUBAllocations", MergeUBAllocations);
}

} // namespace transform
} // namespace tl
} // namespace tvm
