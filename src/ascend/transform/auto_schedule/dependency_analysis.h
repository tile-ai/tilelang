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

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../buffer_version.h"
#include "./ir_structure.h"

namespace tvm {
namespace tl {

using namespace tirx;

using DependencyTaskPair = std::pair<TaskNode *, TaskNode *>;
using StorageSet =
    std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

struct DepInfo {
  IRStructure *prod_node;
  IRStructure *cons_node;
  // Storage identity shared by every Buffer alias participating in this
  // dependency.
  Var storage;
  // Conflicting (producer access, consumer access) pairs. Each pair is a
  // concrete producer→consumer ordering that needs a sync.
  std::vector<DependencyTaskPair> task_pairs;
  // Cross-iteration distance of this dependency:
  //   0   : same-iteration dependency.
  //   -1  : unresolved cross-iteration dependency on an auto-versioned
  //         buffer. InsertSync resolves it to the full ring width at the owner
  //         loop, or to one local iteration in a descendant loop.
  //   >=1 : dependency on a manually multi-buffered buffer, at the physical
  //         iteration distance solved per access pair.
  int distance;
};

// Per-phase dependency-analysis cache. A ControlNode key represents the
// dependency result for its ordered child list at that phase.
using DependencyCache = std::map<ControlNode *, std::vector<DepInfo>>;

// Analyze data dependencies among scheduled TaskNode/ControlNode nodes. A loop
// child list is copied and stable-sorted by stage before dependency directions
// are derived; the kernel root list keeps its original order. `loop` identifies
// the ControlNode whose ordered children are being analyzed; null means the
// kernel root list. Its parent chain supplies the complete enclosing serial
// loop nest for cross-iteration analysis.
// `manual_buffer_versions` maps user-declared manual multi-buffer data Vars to
// their version count; empty means no manual buffers.
// `multi_buffer_owners` maps each storage treated as automatic multi-buffered
// in the current phase to its owner loops.
// AutoSchedule derives it from eligible owners; InsertSync derives it from the
// final physical plan, so an eligible-but-unselected storage is absent there.
// `dependency_cache` is shared for one scheduling or synchronization phase and
// is keyed by `loop`. An `i == j` control-node self-dependency whose
// producer→consumer pairs are already established anywhere inside that node's
// own subtree is dropped.
std::vector<DepInfo>
AnalyzeDependencies(std::vector<IRStructure *> nodes,
                    ControlNode *loop = nullptr,
                    const BufferVersionMap &manual_buffer_versions = {},
                    const MultiBufferOwnerMap &multi_buffer_owners = {},
                    DependencyCache *dependency_cache = nullptr);

// Check whether two task regions may overlap under their respective symbolic
// contexts. `offset == 0` compares accesses in the same iteration. A positive
// offset constrains the current consumer loop coordinate to equal the producer
// coordinate plus that offset. A negative offset denotes an automatic
// cross-iteration check. Normally it renames the loop suffix below the nearest
// ancestor with another access to the storage and requires at least one
// coordinate to differ. Only a multi-owner physical ring at one of its owner
// loops renames the complete enclosing loop nest and permits equal coordinates:
// an identically-shaped iteration in another owner can still be a distinct
// epoch. Shared by scheduling-time IRStructure analysis and the scheduled-TIR
// synchronization pass.
bool RegionsMayConflict(const ConstrSet &a_ctx, const BufferRegion &a_region,
                        const ConstrSet &b_ctx, const BufferRegion &b_region,
                        ControlNode *loop, int offset,
                        size_t num_storage_owners = 0);

} // namespace tl
} // namespace tvm
