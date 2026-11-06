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

#include "ascend/transform/source_scopes.h"
#include "buffer_alias.h"
#include "ir/annotations.h"
#include "layout/expr_utils.h"
#include "merge_ub_common.h"
#include "op/builtin.h"

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
using namespace prim;
using namespace ascend;
using namespace ffi;

namespace {

using namespace merge_ub;

size_t AlignUp(size_t value, int alignment) {
  if (alignment <= 1)
    return value;
  size_t align = static_cast<size_t>(alignment);
  size_t remainder = value % align;
  TVM_FFI_CHECK(value <=
                    static_cast<size_t>(std::numeric_limits<int64_t>::max()) -
                        align + 1,
                ValueError)
      << "Ascend arena alignment overflows int64";
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
    TVM_FFI_CHECK(cursor <=
                      static_cast<size_t>(std::numeric_limits<int64_t>::max()) -
                          clique.size,
                  ValueError)
        << "Ascend arena clique allocation overflows int64";
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
  for (const VarObj *var_node : allocation_order) {
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
    const BindObj *alloc = allocations.at(var.get());
    auto bytes = TryConstInt(GetBufferSizeBytes(alloc));
    if (!bytes.has_value()) {
      dynamic.push_back(var);
      continue;
    }
    TVM_FFI_CHECK_GE(*bytes, 0, ValueError);
    size_t size = static_cast<size_t>(*bytes);
    constant.push_back(
        {var, size,
         std::max({align_bytes, AscendScopeAlignBytes(alloc),
                   AllocationTensor(alloc).ty()->data_alignment}),
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
      plan.byte_offsets.emplace(constant[vertex].storage.get(),
                                IntImm::Int64(static_cast<int64_t>(offset)));
    }
    arena_top = partition.layout.total_size;
  }

  PrimType offset_dtype = PrimType::Int(64);
  PrimExpr cursor = IntImm(offset_dtype, static_cast<int64_t>(arena_top));
  for (const Var &var : dynamic) {
    const BindObj *alloc = allocations.at(var.get());
    int alignment = std::max({align_bytes, AscendScopeAlignBytes(alloc),
                              AllocationTensor(alloc).ty()->data_alignment});
    cursor = AlignPrimExpr(cursor, alignment);
    plan.byte_offsets.emplace(var.get(), cursor);
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

class BufferAliasContractExtractor : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  bool found_contract{false};
  BufferAliasMap aliases;
  AivStorageMap aiv_storages;

  void AddAliases(const Any &value) {
    BufferAliasMap local = value.cast<BufferAliasMap>();
    ValidateBufferAliasMap(local);
    for (const auto &[storage, unused] : local) {
      TVM_FFI_CHECK(!aliases.count(storage), ValueError)
          << "A function contains multiple alias contracts for storage "
          << storage;
      AddBufferAliasStorage(&aliases, storage);
    }
    for (const auto &[storage, compatible] : local)
      for (const Var &other : compatible)
        if (!HasBufferAlias(aliases, storage, other))
          AddBufferAlias(&aliases, storage, other);
    found_contract = true;
  }
  void AddAiv(const Any &value) {
    for (const auto &[storage_any, sid_any] : value.cast<Map<Any, Any>>()) {
      auto sid = sid_any.try_cast<int64_t>();
      if (!sid)
        sid = TryConstInt(sid_any.cast<PrimExpr>());
      TVM_FFI_CHECK(sid && (*sid == 0 || *sid == 1), ValueError)
          << "AIV storage identity must be zero or one";
      Var storage = storage_any.cast<Var>();
      auto previous = aiv_storages.Get(storage);
      TVM_FFI_CHECK(!previous || *previous == *sid, ValueError)
          << "Conflicting AIV identities for storage " << storage;
      aiv_storages.Set(storage, static_cast<int>(*sid));
    }
  }
  Map<String, Any> ExtractFields(Map<String, Any> fields) {
    if (auto value = fields.Get(kBufferAliasMap)) {
      AddAliases(*value);
      fields.erase(kBufferAliasMap);
    }
    if (auto value = fields.Get(kAivStorageMap)) {
      AddAiv(*value);
      fields.erase(kAivStorageMap);
    }
    return fields;
  }
  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    const auto *call = op->value.as<CallObj>();
    if (!IsAnnotationCall(call))
      return StmtExprMutator::Mutate_(op, mode);
    auto fields = ReadAnnotations(call);
    auto result = ExtractFields(fields);
    if (StructuralEqual()(fields, result))
      return Unchanged();
    return result.empty() ? Stmt(SeqStmt({})) : MakeAnnotation(result, op->loc);
  }
  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    RegionStmt region = StmtExprMutator::Mutate_(op, mode)
                            .ValueOrUnchanged(GetRef<RegionStmt>(op))
                            .as_or_throw<RegionStmt>();
    if (IsAscendBlock(region.get())) {
      auto fields = AscendBlockAnnotations(region.get());
      auto result = ExtractFields(fields);
      if (!StructuralEqual()(fields, result))
        return MakeAscendBlock(AscendBlockName(region.get()), result,
                               region->body, region->loc);
      return region;
    }
    auto attr = ReadSourceAttribute(region.get());
    if (!attr)
      return region;
    if (attr->key == kBufferAliasMap || attr->key == kAivStorageMap) {
      if (attr->key == kBufferAliasMap)
        AddAliases(attr->node);
      else
        AddAiv(attr->node);
      return region->body;
    }
    if (attr->key == "tl.sblock" || attr->key == "tl.loop.annotations") {
      auto fields = attr->value.cast<Map<String, Any>>();
      auto result = ExtractFields(fields);
      if (attr->key == "tl.sblock")
        if (auto annotations = result.Get("annotations"))
          result.Set("annotations",
                     ExtractFields(annotations->cast<Map<String, Any>>()));
      if (!StructuralEqual()(fields, result))
        return MakeAttribute(attr->node, attr->key, result, region->body,
                             region->loc);
    }
    return region;
  }
};

Stmt MergeUBAllocations(Stmt stmt, int align_bytes, bool verbose,
                        bool disable_reuse) {
  auto extractor = make_object<BufferAliasContractExtractor>();
  Stmt cleaned =
      extractor->Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(stmt);
  // In particular, do not discard the independent AIV contract when no alias
  // analysis has run yet; another invocation can still consume it.
  if (!extractor->found_contract)
    return stmt;
  stmt = cleaned;
  BufferAliasMap aliases = extractor->aliases;
  ValidateBufferAliasMap(aliases);
  AivStorageMap aiv_storages = extractor->aiv_storages;
  auto planner = [aliases, aiv_storages, disable_reuse](
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
    std::array<AllocationMap, 3> groups;
    std::array<AllocationOrder, 3> orders;
    for (const auto &[storage, allocation] : allocations) {
      auto sid = aiv_storages.Get(GetRef<Var>(storage));
      groups[sid.value_or(2)].emplace(storage, allocation);
    }
    for (const auto *storage : allocation_order)
      for (int group = 0; group < 3; ++group)
        if (groups[group].count(storage))
          orders[group].push_back(storage);
    MemoryPlan result = plan(groups[2], orders[2]);
    PrimExpr prefix = AlignPrimExpr(result.total_size, alignment);
    PrimExpr private_size = IntImm::Int64(0);
    for (int sid = 0; sid < 2; ++sid) {
      MemoryPlan local = plan(groups[sid], orders[sid]);
      for (const auto &[storage, offset] : local.byte_offsets)
        result.byte_offsets.insert_or_assign(
            storage, prefix + cast(offset, PrimType::Int(64)));
      private_size =
          max(private_size, cast(local.total_size, PrimType::Int(64)));
    }
    result.total_size = prefix + private_size;
    return result;
  };
  return merge_ub::MergeAllocations(stmt, align_bytes, verbose, false, planner);
}
} // namespace

namespace transform {
tvm::transform::Pass MergeUBAllocations(int align_bytes = 16,
                                        bool disable_reuse = false) {
  auto pass = [=](tirx::Function function, IRModule,
                  tvm::transform::PassContext context) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    bool verbose = context->GetConfig<bool>(kDebugMergeSharedMemoryAllocations)
                       .value_or(false);
    const auto body = function->body;
    function.CopyOnWrite()->body = SeqStmt(
        tl::MergeUBAllocations(body, align_bytes, verbose, disable_reuse));
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.MergeUBAllocations");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.MergeUBAllocations",
                                   MergeUBAllocations);
}
} // namespace transform
} // namespace tl
} // namespace tvm
