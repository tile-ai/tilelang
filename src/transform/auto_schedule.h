/*!
 * \file auto_schedule.h
 * \brief AutoSchedule pass structures and declarations for TileLang
 */

#pragma once

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

// Global warpgroup id assignment - should be called from the top level
// Tasks that use the same register region must have the same warpgroup id
// Goal: balance weighted latency between two warpgroups (0 and 1)
// Weighted latency = latency * tripcount (tripcount = 100 for non-constant loop
// extent)
bool AssignWarpgroupIdsGlobal(IRStructure *root);

} // namespace tl
} // namespace tvm
