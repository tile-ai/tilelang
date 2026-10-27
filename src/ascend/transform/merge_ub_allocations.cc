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

#include "ascend/target_utils.h"
#include "buffer_addresses.h"
#include "buffer_alias.h"
#include "merge_ub_common.h"

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
    return lhs->name_hint < rhs->name_hint;
  });
  ordered_vars.insert(ordered_vars.end(), tail.begin(), tail.end());

  auto can_alias = [&](const Var &lhs, const Var &rhs) {
    return HasBufferAlias(aliases, lhs, rhs);
  };
  std::vector<CliqueVertex> constant;
  std::vector<Var> dynamic;
  for (size_t index = 0; index < ordered_vars.size(); ++index) {
    const Var &var = ordered_vars[index];
    const AllocBufferNode *alloc = allocations.at(var.get());
    std::optional<int64_t> elements =
        GetRef<AllocBuffer>(alloc).ConstantAllocationSize();
    if (!elements.has_value()) {
      dynamic.push_back(var);
      continue;
    }
    int elem_bits = alloc->buffer->dtype.bits() * alloc->buffer->dtype.lanes();
    size_t size =
        static_cast<size_t>(PackedBytesFromElems(elements.value(), elem_bits));
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
              << constant[clique.members[i]].storage->name_hint << " and "
              << constant[clique.members[j]].storage->name_hint;
        }
      }
    }
    for (size_t vertex = 0; vertex < constant.size(); ++vertex) {
      int clique = partition.assignment[vertex];
      ICHECK_GE(clique, 0);
      ICHECK_LT(static_cast<size_t>(clique),
                partition.layout.byte_offsets.size());
      size_t offset = partition.layout.byte_offsets[clique];
      plan.byte_offsets[constant[vertex].storage.get()] =
          make_const(DataType::Int(32), static_cast<int64_t>(offset));
    }
    arena_top = partition.layout.total_size;
  }

  DataType offset_dtype = DataType::Int(64);
  PrimExpr cursor = make_const(offset_dtype, static_cast<int64_t>(arena_top));
  for (const Var &var : dynamic) {
    const AllocBufferNode *alloc = allocations.at(var.get());
    int alignment = std::max(align_bytes, AscendScopeAlignBytes(alloc));
    cursor = AlignPrimExpr(cursor, alignment);
    plan.byte_offsets[var.get()] = cursor;
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
        LOG(DEBUG) << "      " << constant[member].storage->name_hint;
    }
    for (const auto &[var, offset] : plan.byte_offsets)
      LOG(DEBUG) << "    " << var->name_hint << " offset=" << offset;
  }
  return plan;
}

class BufferPlacementValidator : public StmtVisitor {
public:
  explicit BufferPlacementValidator(std::unordered_set<std::string> scopes)
      : scopes_(std::move(scopes)) {}

  bool HasSimtVF() const { return has_simt_vf_; }

  void VisitStmt_(const SBlockNode *op) final {
    has_simt_vf_ |= op->name_hint == "SIMT_VF";
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    std::string scope = GetPtrStorageScope(op->buffer->data);
    if (scope.size() > 4 && scope.substr(scope.size() - 4) == ".dyn" &&
        scope != "shared.dyn") {
      ICHECK(!scopes_.count(scope.substr(0, scope.size() - 4)))
          << kBufferAddresses << " cannot coexist with an existing " << scope
          << " arena: " << op->buffer;
    }
    if (scopes_.count(scope)) {
      ICHECK(storages_.insert(op->buffer->data).second)
          << kBufferAddresses
          << " requires one allocation per storage: " << op->buffer;
    }
    StmtVisitor::VisitStmt_(op);
  }

private:
  std::unordered_set<std::string> scopes_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> storages_;
  bool has_simt_vf_{false};
};

// Allocations are flattened and include every physical version and padding
// element. Reserve exact intervals first, then first-fit the remaining buffers.
// Placement never establishes permission to reuse storage, even when lifetimes
// are disjoint. Private V0/V1 UB intervals occupy different physical memories.
MemoryPlan BuildPlacedMemoryPlan(const AllocationMap &allocations,
                                 const AllocationOrder &allocation_order,
                                 const BufferAddresses &addresses,
                                 const std::string &scope, int align_bytes,
                                 bool has_simt_vf,
                                 const AivStorageMap &aiv_storages) {
  auto info = GetBufferPlacementInfo(scope);
  int alignment = std::max(align_bytes, info.alignment);
  int64_t capacity = info.capacity;
  if (scope == "shared.dyn" && has_simt_vf)
    capacity -= kAscendSimtContextBytes;

  struct Interval {
    Var storage;
    int64_t begin;
    int64_t end;
    int core_mask;
  };
  std::vector<Interval> intervals;
  MemoryPlan plan;
  int64_t peak = 0;
  auto core_mask = [&](const Var &storage) {
    if (scope == "shared.dyn") {
      if (auto sid = aiv_storages.Get(storage))
        return 1 << sid.value()->value;
    }
    return 3;
  };
  auto size_bytes = [&](const Var &storage) {
    auto allocation = allocations.find(storage.get());
    ICHECK(allocation != allocations.end())
        << kBufferAddresses << " references unknown allocation " << storage;
    const Buffer &buffer = allocation->second->buffer;
    ICHECK(buffer->strides.empty() && is_zero(buffer->elem_offset))
        << kBufferAddresses
        << " requires flattened dense allocations: " << buffer;
    int64_t elem_bits =
        static_cast<int64_t>(buffer->dtype.bits()) * buffer->dtype.lanes();
    ICHECK_GT(elem_bits, 0);
    int64_t max_elements = capacity * 8 / elem_bits;
    int64_t elements = 1;
    for (const PrimExpr &extent : buffer->shape) {
      const auto *constant = extent.as<IntImmNode>();
      ICHECK(constant && constant->value > 0)
          << kBufferAddresses << " requires positive constant allocation "
          << "sizes; storage " << storage << " has extent " << extent;
      ICHECK_LE(constant->value, max_elements / elements)
          << "Allocation " << storage << " exceeds " << scope << " capacity of "
          << capacity << " bytes";
      elements *= constant->value;
    }
    return (elements * elem_bits + 7) / 8;
  };
  auto reserve = [&](const Var &storage, int64_t begin, int64_t bytes) {
    ICHECK_EQ(begin % alignment, 0)
        << "Buffer placement for " << storage << " requires " << alignment
        << "-byte alignment";
    ICHECK(begin >= 0 && begin <= capacity && bytes <= capacity - begin)
        << "Buffer placement for " << storage << " at byte " << begin
        << " with size " << bytes << " exceeds " << scope << " capacity of "
        << capacity << " bytes";
    int64_t end = begin + bytes;
    int mask = core_mask(storage);
    for (const Interval &other : intervals) {
      ICHECK(!(mask & other.core_mask) || end <= other.begin ||
             other.end <= begin)
          << "Buffer placements overlap in " << scope << ": " << other.storage
          << " [" << other.begin << ", " << other.end << ") and " << storage
          << " [" << begin << ", " << end << ")";
    }
    intervals.push_back({storage, begin, end, mask});
    plan.byte_offsets[storage.get()] = IntImm(DataType::Int(32), begin);
    peak = std::max(peak, end);
  };
  for (const auto &[storage, address] : addresses)
    reserve(storage, address->value, size_bytes(storage));

  for (const VarNode *node : allocation_order) {
    Var storage = GetRef<Var>(node);
    if (addresses.count(storage))
      continue;
    int64_t bytes = size_bytes(storage);
    int64_t begin = 0;
    int mask = core_mask(storage);
    std::sort(
        intervals.begin(), intervals.end(),
        [](const Interval &a, const Interval &b) { return a.begin < b.begin; });
    for (const Interval &other : intervals) {
      if (!(mask & other.core_mask) || other.end <= begin)
        continue;
      if (bytes <= other.begin - begin)
        break;
      begin = AlignUp(other.end, alignment);
    }
    reserve(storage, begin, bytes);
  }
  ICHECK_EQ(plan.byte_offsets.size(), allocations.size());
  int64_t total = AlignUp(peak, alignment);
  ICHECK_LE(total, capacity)
      << "Buffer placements exceed " << scope << " capacity of " << capacity
      << " bytes after alignment";
  plan.total_size = IntImm(DataType::Int(32), total);
  return plan;
}

class BufferAliasContractExtractor : public StmtMutator {
public:
  Stmt Extract(const Stmt &stmt) { return VisitStmt(stmt); }
  bool FoundContract() const { return found_contract_; }
  const BufferAliasMap &Aliases() const { return aliases_; }
  const AivStorageMap &AivStorages() const { return aiv_storages_; }
  const Optional<BufferAddresses> &Addresses() const { return addresses_; }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == kBufferAddresses) {
      ICHECK(!addresses_.has_value())
          << "A PrimFunc must contain one " << kBufferAddresses << " contract";
      addresses_ = ParseBufferAddresses(op->node);
      return VisitStmt(op->body);
    }
    if (op->attr_key == kAivStorageMap) {
      for (const auto &[storage, sid] : op->node.cast<AivStorageMap>()) {
        ICHECK(sid->value == 0 || sid->value == 1);
        aiv_storages_.Set(storage, sid);
      }
      return VisitStmt(op->body);
    }
    if (op->attr_key != kBufferAliasMap)
      return StmtMutator::VisitStmt_(op);
    BufferAliasMap local = op->node.cast<BufferAliasMap>();
    ValidateBufferAliasMap(local);
    for (const auto &[storage, _] : local) {
      ICHECK(!aliases_.count(storage))
          << "A PrimFunc contains multiple buffer-alias contracts for "
             "storage "
          << storage;
      AddBufferAliasStorage(&aliases_, storage);
    }
    for (const auto &[storage, compatible] : local) {
      for (const Var &other : compatible) {
        if (!HasBufferAlias(aliases_, storage, other))
          AddBufferAlias(&aliases_, storage, other);
      }
    }
    found_contract_ = true;
    return VisitStmt(op->body);
  }

  BufferAliasMap aliases_;
  AivStorageMap aiv_storages_;
  Optional<BufferAddresses> addresses_;
  bool found_contract_{false};
};

Stmt MergeUBAllocations(Stmt stmt, int align_bytes, bool verbose,
                        bool disable_reuse) {
  BufferAliasContractExtractor extractor;
  stmt = extractor.Extract(stmt);
  if (!extractor.FoundContract()) {
    ICHECK(!extractor.Addresses().has_value())
        << kBufferAddresses << " requires the Ascend scheduled allocation "
        << "pipeline and its buffer-alias contract";
    return stmt;
  }
  BufferAliasMap aliases = extractor.Aliases();
  ValidateBufferAliasMap(aliases);

  AivStorageMap aiv_storages = extractor.AivStorages();
  std::unordered_map<std::string, MemoryPlan> placed_plans;
  std::unordered_set<std::string> placed_scopes;
  if (extractor.Addresses().has_value()) {
    std::unordered_map<std::string, BufferAddresses> scope_addresses;
    for (const auto &[storage, address] : extractor.Addresses().value()) {
      std::string scope = GetPtrStorageScope(storage);
      placed_scopes.insert(scope);
      scope_addresses[scope].Set(storage, address);
    }
    BufferPlacementValidator validator(placed_scopes);
    validator(stmt);
    AllocateCollector collector;
    collector(stmt);
    for (const auto &[scope, addresses] : scope_addresses) {
      const AllocationMap &allocations =
          scope == "shared.dyn" ? collector.dyn_shmem_allocs
                                : collector.ascend_scope_allocs[scope];
      const AllocationOrder &order = scope == "shared.dyn"
                                         ? collector.dyn_shmem_order
                                         : collector.ascend_scope_order[scope];
      placed_plans.emplace(
          scope, BuildPlacedMemoryPlan(allocations, order, addresses, scope,
                                       align_bytes, validator.HasSimtVF(),
                                       aiv_storages));
    }
  }
  auto planner = [aliases = std::move(aliases), aiv_storages, disable_reuse,
                  placed_plans](const Stmt &, const AllocationMap &allocations,
                                const AllocationOrder &allocation_order,
                                const std::string &scope, int alignment,
                                bool debug) {
    auto placed = placed_plans.find(scope.empty() ? "shared.dyn" : scope);
    if (placed != placed_plans.end())
      return placed->second;
    auto plan = [&](const AllocationMap &group, const AllocationOrder &order) {
      return disable_reuse ? BuildSequentialPlan(group, alignment)
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
      int group = sid.has_value() ? static_cast<int>(sid.value()->value) : 2;
      groups[group].emplace(storage, allocation);
    }
    for (const auto *storage : allocation_order) {
      for (int group = 0; group < 3; ++group)
        if (groups[group].count(storage))
          orders[group].push_back(storage);
    }
    MemoryPlan result = plan(groups[2], orders[2]);
    PrimExpr prefix = AlignPrimExpr(result.total_size, alignment);
    PrimExpr private_size = make_zero(prefix.dtype());
    for (int sid = 0; sid < 2; ++sid) {
      MemoryPlan local = plan(groups[sid], orders[sid]);
      for (const auto &[storage, offset] : local.byte_offsets)
        result.byte_offsets[storage] = prefix + cast(prefix.dtype(), offset);
      private_size = max(private_size, cast(prefix.dtype(), local.total_size));
    }
    result.total_size = prefix + private_size;
    return result;
  };
  return merge_ub::MergeAllocations(std::move(stmt), align_bytes, verbose,
                                    false, planner, placed_scopes);
}

} // namespace

using namespace tirx::transform;

namespace transform {

Pass MergeUBAllocations(int align_bytes = 16, bool disable_reuse = false) {
  auto pass_func = [align_bytes, disable_reuse](PrimFunc func, const IRModule &,
                                                PassContext context) {
    bool verbose =
        context
            ->GetConfig<Bool>(kDebugMergeSharedMemoryAllocations, Bool(false))
            .value();
    PrimFuncNode *writer = func.CopyOnWrite();
    writer->body = tl::MergeUBAllocations(std::move(writer->body), align_bytes,
                                          verbose, disable_reuse);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MergeUBAllocations", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.MergeUBAllocations", MergeUBAllocations);
}

} // namespace transform
} // namespace tl
} // namespace tvm
