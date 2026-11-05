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
 * \file dependency_analysis.h
 * \brief Shared schedule and synchronization dependency analysis.
 */

#pragma once

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../buffer_version.h"
#include "./ir_structure.h"

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;

using DependencyTaskPair = std::pair<TaskNode *, TaskNode *>;

enum class DependencyKind {
  // A directed producer-to-consumer data dependency.
  kData,
  // Two disjoint owners of one multi-buffer-eligible storage.
  // `prod_node` and `cons_node` only establish a stable pair orientation;
  // `task_pairs` stores matching accesses on those two sides. Scheduling must
  // choose either direction and keep both storage lifetimes disjoint.
  kOwnerExclusion,
};

struct DepInfo {
  IRStructure *prod_node;
  IRStructure *cons_node;
  // Proven storage identity shared by every tensor alias participating in this
  // dependency. Undefined for an explicit conflict between distinct storage
  // keys; those edges do not participate in buffer versioning.
  ffi::Optional<Var> storage;
  // Conflicting access pairs. For kData these are directed producer→consumer
  // pairs; for kOwnerExclusion they identify the two unordered owner sides.
  std::vector<DependencyTaskPair> task_pairs;
  // Cross-iteration distance of this dependency:
  //   0   : same-iteration dependency.
  //   -1  : unresolved cross-iteration dependency. A versioned storage may
  //         resolve it to its ring width; otherwise it means one iteration.
  //   >=1 : dependency on a manually multi-buffered buffer, at the physical
  //         iteration distance solved per access pair.
  // kOwnerExclusion leaves this as zero during shared analysis. InsertSync
  // resolves both directed distances from the final stage and physical order.
  int distance;
  DependencyKind kind{DependencyKind::kData};
};

// A lexical child list and its enclosing iteration clock identify one query.
// The then and else lists of a ScopeNode must never share a cached result.
struct DependencyCacheKey {
  ControlNode *loop;
  std::vector<IRStructure *> nodes;
  std::shared_ptr<const void> scoped_hint_identity;

  bool operator<(const DependencyCacheKey &other) const {
    if (scoped_hint_identity.owner_before(other.scoped_hint_identity))
      return true;
    if (other.scoped_hint_identity.owner_before(scoped_hint_identity))
      return false;
    if (loop != other.loop)
      return std::less<ControlNode *>()(loop, other.loop);
    return std::lexicographical_compare(nodes.begin(), nodes.end(),
                                        other.nodes.begin(), other.nodes.end(),
                                        std::less<IRStructure *>());
  }
};
using DependencyCache = std::map<DependencyCacheKey, std::vector<DepInfo>>;

// Conflict-hint entries. Each entry is
// [operand_a, operand_b, IntImm(cross_code), Bool(is_conflict)]. Kernel-root
// entries apply only while analyzing the outermost sequence.
using ConflictHintList = ffi::Array<ffi::Any>;

// Analysis-only index over one current kernel forest. The forest must outlive
// the index. Clone, reparenting, statement/context changes or core
// specialization require a new index; validation precedes dependency-cache
// lookups so an old index can never certify a changed tree.
class ScopedConflictHintIndex {
public:
  explicit ScopedConflictHintIndex(IRStructure::ChildList &tree,
                                   const ConstrSet &outer_ctx = {});
  ~ScopedConflictHintIndex();
  ScopedConflictHintIndex(const ScopedConflictHintIndex &) = delete;
  ScopedConflictHintIndex &operator=(const ScopedConflictHintIndex &) = delete;

  void Validate(const std::vector<IRStructure *> &nodes,
                const ControlNode *loop) const;
  const std::shared_ptr<const void> &CacheIdentity() const;
  std::optional<bool> Query(ControlNode *loop, int offset,
                            const TaskNode &a_task, const Var &a_storage,
                            const Optional<TensorRegion> &a_region,
                            const TaskNode &b_task, const Var &b_storage,
                            const Optional<TensorRegion> &b_region) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Analyze data dependencies among scheduled task, control and lexical-scope
// nodes. Each segment of a loop child list between lexical scopes is copied
// and stable-sorted by stage before dependency directions are derived; lexical
// headers and the kernel root list keep their original order. `loop` identifies
// the ControlNode whose ordered children are being analyzed; null means the
// kernel root list. Its parent chain supplies the complete enclosing serial
// loop nest for cross-iteration analysis.
// `manual_buffer_versions` maps user-declared manual multi-buffer data Vars to
// their version count; empty means no manual buffers.
// `multi_buffer_owners` maps each storage treated as automatic multi-buffered
// in the current phase to its owner loops. Dependencies between distinct
// owners are returned as kOwnerExclusion; physical ring size is handled by the
// owner-local protocol.
// `dependency_cache` is shared for one scheduling or synchronization phase and
// is keyed by the ordered child list and `loop`. An `i == j` composite-node
// self-dependency whose producer→consumer pairs are already established
// anywhere inside that node's own subtree is dropped.
std::vector<DepInfo>
AnalyzeDependencies(std::vector<IRStructure *> nodes,
                    ControlNode *loop = nullptr,
                    const BufferVersionMap &manual_buffer_versions = {},
                    const MultiBufferOwnerMap &multi_buffer_owners = {},
                    DependencyCache *dependency_cache = nullptr,
                    const ConflictHintList &root_conflicts = {},
                    const ScopedConflictHintIndex *scoped_hints = nullptr);

// Check whether two task regions may overlap under their respective symbolic
// contexts. Only a proof of disjointness removes a conflict.
// `offset == 0` compares accesses in the same iteration. A positive
// offset constrains the current consumer loop coordinate to equal the producer
// coordinate plus that offset times the native loop step. A negative offset
// denotes an automatic cross-iteration check. Normally it renames the loop
// suffix below the nearest ancestor with another access to the storage and
// requires at least one coordinate to differ. Only a multi-owner physical ring
// at one of its owner loops renames the complete enclosing loop nest and
// permits equal coordinates: an identically-shaped iteration in another owner
// can still be a distinct epoch. Shared by scheduling-time IRStructure analysis
// and the scheduled-TIR synchronization pass. Copy normalization makes
// instruction transfer regions explicit before scheduling. Compare those
// regions directly, including for conflict hints.
bool RegionsMayConflict(const ConstrSet &a_ctx, const TensorRegion &a_region,
                        const ConstrSet &b_ctx, const TensorRegion &b_region,
                        ControlNode *loop, int offset,
                        size_t num_storage_owners = 0,
                        const ConflictHintList &root_conflicts = {},
                        Optional<PrimExpr> a_pitch = std::nullopt,
                        Optional<PrimExpr> b_pitch = std::nullopt,
                        const NativeStorageContext &a_storage = {},
                        const NativeStorageContext &b_storage = {},
                        const ScopedConflictHintIndex *scoped_hints = nullptr,
                        const TaskNode *a_task = nullptr,
                        const TaskNode *b_task = nullptr);

} // namespace ascend
} // namespace tl
} // namespace tvm
