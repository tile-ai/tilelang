/*!
 * \file auto_schedule.h
 * \brief AutoSchedule pass structures and declarations for TileLang
 */

#pragma once

#include "../target/utils.h"
#include "./auto_schedule/barrier.h"
#include "./auto_schedule/ir_structure.h"
#include "./auto_schedule/latency_estimator.h"
#include "./auto_schedule/memory_detector.h"
#include <tvm/runtime/logging.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

// Simple Union-Find (Disjoint Set Union) for task grouping
class TaskUnionFind {
public:
  TaskUnionFind(int n) : parent(n), rank(n, 0) {
    for (int i = 0; i < n; i++) {
      parent[i] = i;
    }
  }

  int find(int x) {
    if (parent[x] != x) {
      parent[x] = find(parent[x]); // path compression
    }
    return parent[x];
  }

  int find(int x) const {
    int root = x;
    while (parent[root] != root) {
      root = parent[root];
    }
    return root;
  }

  void unite(int x, int y) {
    int root_x = find(x);
    int root_y = find(y);
    if (root_x == root_y)
      return;

    // union by rank
    if (rank[root_x] < rank[root_y]) {
      parent[root_x] = root_y;
    } else if (rank[root_x] > rank[root_y]) {
      parent[root_y] = root_x;
    } else {
      parent[root_y] = root_x;
      rank[root_x]++;
    }
  }

private:
  std::vector<int> parent;
  std::vector<int> rank;
};

// Structure for component information used in warpgroup assignment
struct ComponentInfo {
  int root;
  int64_t weighted_latency; // total weighted latency in this component
  std::vector<int> task_indices;
  bool uses_tma_core_{false};
  bool uses_tensor_core_{false};
};

// Warp specialization architecture enum
enum class WarpSpecializeArch : uint8_t {
  kHopper = 0,
  kBlackwell = 1,
  kUnsupported = 2,
};

// Configuration for warp specialization
struct WarpSpecializeConfig {
  WarpSpecializeArch arch = WarpSpecializeArch::kUnsupported;
  int consumer_max_nreg = 0;
  int producer_max_nreg = 0;
  int producer_thread_count = 0;
  bool enable_set_max_nreg = false;
  bool enable_warpgroup_partition = false;
  bool enable_thread_extend = false;
  bool enable_warp_partition = false;
};

// Factory function to get warp specialization configuration for a target
inline WarpSpecializeConfig GetWarpSpecializeConfig(Target target) {
  if (TargetIsHopper(target)) {
    return {WarpSpecializeArch::kHopper, 240, 24, 128, true, true, true, false};
  } else if (TargetIsSm100(target)) {
    return {WarpSpecializeArch::kBlackwell, 0, 0, 32, false, true, false, true};
  } else {
    return {
        WarpSpecializeArch::kUnsupported, 0, 0, 0, false, false, false, false};
  }
}

// Global warpgroup id assignment - should be called from the top level
// Tasks that use the same register region must have the same warpgroup id
// Goal: balance weighted latency between two warpgroups (0 and 1)
// Weighted latency = latency * tripcount (tripcount = 100 for non-constant loop
// extent)
bool AssignWarpgroupIdsGlobal(IRStructure *root,
                              bool enable_warp_partition = false);

// Function to rewrite alloc_buffers for multi-version support
Stmt RewriteAllocBuffers(
    const Stmt &stmt, const std::vector<MultiVersionBufferInfo> &buffer_infos);

} // namespace tl
} // namespace tvm
