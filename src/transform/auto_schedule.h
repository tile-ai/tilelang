/*!
 * \file auto_schedule.h
 * \brief AutoSchedule pass structures and declarations for TileLang
 */

#pragma once

#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/runtime/logging.h>

#include <utility>
#include <vector>
#include <memory>
#include <unordered_map>

namespace tvm {
namespace tl {

using namespace tir;

// Forward declarations
class IRStructure;
class TaskNode;
class ControlNode;
class SequenceNode;

// Memory type classification
enum class MemoryType {
  kGlobal,    // Global memory (DRAM)
  kShared,    // Shared memory (L1/shared)
  kRegister,  // Register/local memory
  kUnknown    // Unknown memory type
};

// Helper function to determine memory type from buffer scope
inline MemoryType GetMemoryTypeFromScope(const String& scope) {
  if (scope == "global") {
    return MemoryType::kGlobal;
  } else if (scope == "shared" || scope == "shared.dyn") {
    return MemoryType::kShared;
  } else if (scope == "local" || scope == "local.var" || scope == "local.fragment") {
    return MemoryType::kRegister;
  }
  return MemoryType::kUnknown;
}

// Helper function to compare if two regions are equal
bool RegionsEqual(const Region& a, const Region& b);

// Base class for all IR nodes in scheduling
class IRStructure {
public:
  enum class Kind {
    kTask,
    kControl,
    kSequence
  };

  virtual ~IRStructure() = default;
  virtual Kind GetKind() const = 0;
  virtual std::unique_ptr<IRStructure> Clone() const = 0;

  // Helper methods for safe casting
  bool IsTask() const { return GetKind() == Kind::kTask; }
  bool IsControl() const { return GetKind() == Kind::kControl; }
  bool IsSequence() const { return GetKind() == Kind::kSequence; }

  // Resource usage flags (accessible by all IR nodes)
  virtual bool UsesCUDACore() const = 0;
  virtual bool UsesTMACore() const = 0;
  virtual bool UsesTensorCore() const = 0;

  // Memory access regions (collected during analysis)
  virtual std::vector<BufferRegion> GetReadRegions() const = 0;
  virtual std::vector<BufferRegion> GetWriteRegions() const = 0;

  // Latency estimation
  virtual int64_t GetLatency() const = 0;      // Estimated latency in cycles
  virtual int64_t GetII() const = 0;           // Initiation interval in cycles

  // Setters (for analysis passes to update these values)
  virtual void SetUsesCUDACore(bool value) = 0;
  virtual void SetUsesTMACore(bool value) = 0;
  virtual void SetUsesTensorCore(bool value) = 0;
  virtual void SetReadRegions(const std::vector<BufferRegion>& regions) = 0;
  virtual void SetWriteRegions(const std::vector<BufferRegion>& regions) = 0;
  virtual void SetLatency(int64_t latency) = 0;
  virtual void SetII(int64_t ii) = 0;

  // Helper methods to add regions (for incremental analysis)
  virtual void AddReadRegion(const BufferRegion& region) = 0;
  virtual void AddWriteRegion(const BufferRegion& region) = 0;
};

// Task node: contains a vector of statements
class TaskNode : public IRStructure {
public:
  std::vector<Stmt> stmts;

  Kind GetKind() const override { return Kind::kTask; }

  // Resource usage flags
  bool UsesCUDACore() const override { return uses_cuda_core_; }
  bool UsesTMACore() const override { return uses_tma_core_; }
  bool UsesTensorCore() const override { return uses_tensor_core_; }

  // Memory access regions (collected during analysis)
  std::vector<BufferRegion> GetReadRegions() const override { return read_regions_; }
  std::vector<BufferRegion> GetWriteRegions() const override { return write_regions_; }

  // Latency estimation
  int64_t GetLatency() const override { return latency_; }
  int64_t GetII() const override { return ii_; }

  // Setters
  void SetUsesCUDACore(bool value) override { uses_cuda_core_ = value; }
  void SetUsesTMACore(bool value) override { uses_tma_core_ = value; }
  void SetUsesTensorCore(bool value) override { uses_tensor_core_ = value; }
  void SetReadRegions(const std::vector<BufferRegion>& regions) override { read_regions_ = regions; }
  void SetWriteRegions(const std::vector<BufferRegion>& regions) override { write_regions_ = regions; }
  void SetLatency(int64_t latency) override { latency_ = latency; }
  void SetII(int64_t ii) override { ii_ = ii; }

  // Start time for scheduling
  void SetStartTime(int64_t start_time) { start_time_ = start_time; }
  int64_t GetStartTime() const { return start_time_; }

  // Promote flag for software pipelining
  void SetPromote(bool promote) { promote_ = promote; }
  bool GetPromote() const { return promote_; }

  // Warpgroup id for warpgroup specialization
  void SetWarpgroupId(int warpgroup_id) { warpgroup_id_ = warpgroup_id; }
  int GetWarpgroupId() const { return warpgroup_id_; }

  // Clone method
  std::unique_ptr<IRStructure> Clone() const override;

  // Helper methods to add regions (for incremental analysis)
  void AddReadRegion(const BufferRegion& region) override {
    // Check for duplicate regions
    for (const auto& existing : read_regions_) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        return; // Region already exists
      }
    }
    read_regions_.push_back(region);
  }

  void AddWriteRegion(const BufferRegion& region) override {
    // Check for duplicate regions
    for (const auto& existing : write_regions_) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        return; // Region already exists
      }
    }
    write_regions_.push_back(region);
  }

private:
  // Resource usage flags
  bool uses_cuda_core_{false};
  bool uses_tma_core_{false};
  bool uses_tensor_core_{false};

  // Memory access regions (collected during analysis)
  std::vector<BufferRegion> read_regions_;
  std::vector<BufferRegion> write_regions_;

  // Latency estimation
  int64_t latency_{0};      // Estimated latency in cycles
  int64_t ii_{0};           // Initiation interval in cycles
  int64_t start_time_{0};   // Scheduled start time in cycles
  bool promote_{false};     // Promote flag for software pipelining
  int warpgroup_id_{-1};    // Warpgroup id for warpgroup specialization (-1 means unassigned)
};

// Control node: contains a For operation and a child IRStructure
class ControlNode : public IRStructure {
public:
  For control;  // The For operation
  std::unique_ptr<IRStructure> child;

  Kind GetKind() const override { return Kind::kControl; }

  // Resource usage flags (aggregate from child)
  bool UsesCUDACore() const override { return child ? child->UsesCUDACore() : false; }
  bool UsesTMACore() const override { return child ? child->UsesTMACore() : false; }
  bool UsesTensorCore() const override { return child ? child->UsesTensorCore() : false; }

  // Memory access regions (aggregate from child)
  std::vector<BufferRegion> GetReadRegions() const override {
    return child ? child->GetReadRegions() : std::vector<BufferRegion>{};
  }
  std::vector<BufferRegion> GetWriteRegions() const override {
    return child ? child->GetWriteRegions() : std::vector<BufferRegion>{};
  }

  // Latency estimation (aggregate from child)
  int64_t GetLatency() const override { return child ? child->GetLatency() : 0; }
  int64_t GetII() const override { return child ? child->GetII() : 0; }

  // Setters (delegate to child if exists)
  void SetUsesCUDACore(bool value) override { if (child) child->SetUsesCUDACore(value); }
  void SetUsesTMACore(bool value) override { if (child) child->SetUsesTMACore(value); }
  void SetUsesTensorCore(bool value) override { if (child) child->SetUsesTensorCore(value); }
  void SetReadRegions(const std::vector<BufferRegion>& regions) override { if (child) child->SetReadRegions(regions); }
  void SetWriteRegions(const std::vector<BufferRegion>& regions) override { if (child) child->SetWriteRegions(regions); }
  void SetLatency(int64_t latency) override { if (child) child->SetLatency(latency); }
  void SetII(int64_t ii) override { if (child) child->SetII(ii); }

  // Helper methods to add regions (delegate to child)
  void AddReadRegion(const BufferRegion& region) override { if (child) child->AddReadRegion(region); }
  void AddWriteRegion(const BufferRegion& region) override { if (child) child->AddWriteRegion(region); }

  // Clone method
  std::unique_ptr<IRStructure> Clone() const override;
};

// Sequence node: contains a vector of child IRStructures
class SequenceNode : public IRStructure {
public:
  std::vector<std::unique_ptr<IRStructure>> children;

  Kind GetKind() const override { return Kind::kSequence; }

  // Resource usage flags (aggregate from all children)
  bool UsesCUDACore() const override;
  bool UsesTMACore() const override;
  bool UsesTensorCore() const override;

  // Memory access regions (aggregate from all children)
  std::vector<BufferRegion> GetReadRegions() const override;
  std::vector<BufferRegion> GetWriteRegions() const override;

  // Latency estimation (aggregate from all children)
  int64_t GetLatency() const override;
  int64_t GetII() const override;

  // Setters (delegate to first child if exists)
  void SetUsesCUDACore(bool value) override;
  void SetUsesTMACore(bool value) override;
  void SetUsesTensorCore(bool value) override;
  void SetReadRegions(const std::vector<BufferRegion>& regions) override;
  void SetWriteRegions(const std::vector<BufferRegion>& regions) override;
  void SetLatency(int64_t latency) override;
  void SetII(int64_t ii) override;

  // Helper methods to add regions (delegate to first child if exists)
  void AddReadRegion(const BufferRegion& region) override;
  void AddWriteRegion(const BufferRegion& region) override;

  // Clone method
  std::unique_ptr<IRStructure> Clone() const override;
};

// ScheduleUnit: a group of consecutive TaskNodes that can be scheduled together
struct ScheduleUnit {
  std::vector<TaskNode*> tasks;  // consecutive TaskNodes
  SequenceNode* parent_seq{nullptr};  // Parent SequenceNode containing these tasks
  size_t start_idx{0};  // Starting index in parent_seq->children
  bool inside_control_node{false};  // Whether this unit is inside a ControlNode (For loop)
  ControlNode* control_node{nullptr};  // Pointer to containing ControlNode (if inside_control_node is true)
  // Additional metadata could be added here, e.g., resource usage summary
};

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
    if (root_x == root_y) return;

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

// Structure to store TaskNode with its loop context information
struct TaskNodeWithContext {
  TaskNode* task;
  ControlNode* control_node{nullptr};  // nullptr if not inside a loop
  int64_t tripcount{1};  // tripcount multiplier for weighted latency
};

// Structure for component information used in warpgroup assignment
struct ComponentInfo {
  int root;
  int64_t weighted_latency; // total weighted latency in this component
  std::vector<int> task_indices;
};

// Helper function to check if a buffer region is in register memory
bool IsRegisterRegion(const BufferRegion& region);

// Helper function to collect all register regions from a task
std::vector<BufferRegion> CollectRegisterRegions(const TaskNode* task);

// Helper function to check if two TaskNodes use the same register region
bool UseSameRegisterRegion(const TaskNode* a, const TaskNode* b);

// Helper function to count register regions in a task
int CountRegisterRegions(const TaskNode* task);

// Helper function to collect all TaskNodes with context information
void CollectAllTaskNodesWithContext(const IRStructure* node,
                                    std::vector<TaskNodeWithContext>& all_tasks,
                                    ControlNode* current_control_node = nullptr);

// Global warpgroup id assignment - should be called from the top level
// Tasks that use the same register region must have the same warpgroup id
// Goal: balance weighted latency between two warpgroups (0 and 1)
// Weighted latency = latency * tripcount (tripcount = 100 for non-constant loop extent)
void AssignWarpgroupIdsGlobal(IRStructure* root);


// Apply warpgroup partition to a ScheduleUnit
// Split tasks into two groups based on warpgroup id and insert conditional branching
// if tx < 128: execute warpgroup 0 tasks, else: execute warpgroup 1 tasks
// Note: cross-warpgroup dependencies are ignored for now (will be handled later with barriers)
void ApplyWarpgroupPartition(ScheduleUnit& unit);

} // namespace tl
} // namespace tvm