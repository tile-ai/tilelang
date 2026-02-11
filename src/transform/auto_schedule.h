/*!
 * \file auto_schedule.h
 * \brief AutoSchedule pass structures and declarations for TileLang
 */

#pragma once

#include <tvm/runtime/logging.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

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
  kGlobal,   // Global memory (DRAM)
  kShared,   // Shared memory (L1/shared)
  kRegister, // Register/local memory
  kUnknown   // Unknown memory type
};

// Helper function to determine memory type from buffer scope
inline MemoryType GetMemoryTypeFromScope(const String &scope) {
  if (scope == "global") {
    return MemoryType::kGlobal;
  } else if (scope == "shared" || scope == "shared.dyn") {
    return MemoryType::kShared;
  } else if (scope == "local" || scope == "local.var" ||
             scope == "local.fragment") {
    return MemoryType::kRegister;
  }
  return MemoryType::kUnknown;
}

// Helper function to compare if two regions are equal
inline bool RegionsEqual(const Region &a, const Region &b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!tir::is_one(a[i]->min - b[i]->min) ||
        !tir::is_one(a[i]->extent - b[i]->extent)) {
      return false;
    }
  }
  return true;
}

// Base class for all IR nodes in scheduling
class IRStructure {
public:
  enum class Kind { kTask, kControl, kSequence, kWrapper };

  virtual ~IRStructure() = default;
  virtual Kind GetKind() const = 0;
  virtual std::unique_ptr<IRStructure> Clone() const = 0;

  // Helper methods for safe casting
  bool IsTask() const { return GetKind() == Kind::kTask; }
  bool IsControl() const { return GetKind() == Kind::kControl; }
  bool IsSequence() const { return GetKind() == Kind::kSequence; }
  bool IsWrapper() const { return GetKind() == Kind::kWrapper; }

  // Resource usage flags (accessible by all IR nodes)
  virtual bool UsesCUDACore() const = 0;
  virtual bool UsesTMACore() const = 0;
  virtual bool UsesTensorCore() const = 0;

  // Memory access regions (collected during analysis)
  virtual std::vector<BufferRegion> GetReadRegions() const = 0;
  virtual std::vector<BufferRegion> GetWriteRegions() const = 0;

  // Latency estimation
  virtual int64_t GetLatency() const = 0; // Estimated latency in cycles
  virtual int64_t GetII() const = 0;      // Initiation interval in cycles

  // Setters (for analysis passes to update these values)
  virtual void SetUsesCUDACore(bool value) = 0;
  virtual void SetUsesTMACore(bool value) = 0;
  virtual void SetUsesTensorCore(bool value) = 0;
  virtual void SetReadRegions(const std::vector<BufferRegion> &regions) = 0;
  virtual void SetWriteRegions(const std::vector<BufferRegion> &regions) = 0;
  virtual void SetLatency(int64_t latency) = 0;
  virtual void SetII(int64_t ii) = 0;

  // Helper methods to add regions (for incremental analysis)
  virtual void AddReadRegion(const BufferRegion &region) = 0;
  virtual void AddWriteRegion(const BufferRegion &region) = 0;
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
  std::vector<BufferRegion> GetReadRegions() const override {
    return read_regions_;
  }
  std::vector<BufferRegion> GetWriteRegions() const override {
    return write_regions_;
  }
  std::vector<std::pair<BufferRegion, bool>> GetReadWriteRegions() const {
    std::vector<std::pair<BufferRegion, bool>> read_write_regions;
    std::set<Buffer> buffers;
    for (const auto &region : GetWriteRegions()) {
      if (buffers.find(region->buffer) == buffers.end()) {
        buffers.insert(region->buffer);
        read_write_regions.push_back({region, true});
      }
    }
    for (const auto &region : GetReadRegions()) {
      if (buffers.find(region->buffer) == buffers.end()) {
        buffers.insert(region->buffer);
        read_write_regions.push_back({region, false});
      }
    }
    return read_write_regions;
  }

  // Latency estimation
  int64_t GetLatency() const override { return latency_; }
  int64_t GetII() const override { return ii_; }

  // Setters
  void SetUsesCUDACore(bool value) override { uses_cuda_core_ = value; }
  void SetUsesTMACore(bool value) override { uses_tma_core_ = value; }
  void SetUsesTensorCore(bool value) override { uses_tensor_core_ = value; }
  void SetReadRegions(const std::vector<BufferRegion> &regions) override {
    read_regions_ = regions;
  }
  void SetWriteRegions(const std::vector<BufferRegion> &regions) override {
    write_regions_ = regions;
  }
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
  void AddReadRegion(const BufferRegion &region) override {
    // Check for duplicate regions
    for (const auto &existing : read_regions_) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        return; // Region already exists
      }
    }
    read_regions_.push_back(region);
  }

  void AddWriteRegion(const BufferRegion &region) override {
    // Check for duplicate regions
    for (const auto &existing : write_regions_) {
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
  int64_t latency_{0};    // Estimated latency in cycles
  int64_t ii_{0};         // Initiation interval in cycles
  int64_t start_time_{0}; // Scheduled start time in cycles
  bool promote_{false};   // Promote flag for software pipelining
  int warpgroup_id_{
      -1}; // Warpgroup id for warpgroup specialization (-1 means unassigned)
};

// Control node: contains a For operation and a child IRStructure
class ControlNode : public IRStructure {
public:
  For control; // The For operation
  std::unique_ptr<IRStructure> child;

  Kind GetKind() const override { return Kind::kControl; }

  // Resource usage flags (aggregate from child)
  bool UsesCUDACore() const override {
    return child ? child->UsesCUDACore() : false;
  }
  bool UsesTMACore() const override {
    return child ? child->UsesTMACore() : false;
  }
  bool UsesTensorCore() const override {
    return child ? child->UsesTensorCore() : false;
  }

  // Memory access regions (aggregate from child)
  std::vector<BufferRegion> GetReadRegions() const override {
    return child ? child->GetReadRegions() : std::vector<BufferRegion>{};
  }
  std::vector<BufferRegion> GetWriteRegions() const override {
    return child ? child->GetWriteRegions() : std::vector<BufferRegion>{};
  }

  // Latency estimation (aggregate from child)
  int64_t GetLatency() const override { return latency_; }
  int64_t GetII() const override { return ii_; }

  // Setters (delegate to child if exists)
  void SetUsesCUDACore(bool value) override {
    if (child)
      child->SetUsesCUDACore(value);
  }
  void SetUsesTMACore(bool value) override {
    if (child)
      child->SetUsesTMACore(value);
  }
  void SetUsesTensorCore(bool value) override {
    if (child)
      child->SetUsesTensorCore(value);
  }
  void SetReadRegions(const std::vector<BufferRegion> &regions) override {
    if (child)
      child->SetReadRegions(regions);
  }
  void SetWriteRegions(const std::vector<BufferRegion> &regions) override {
    if (child)
      child->SetWriteRegions(regions);
  }
  void SetLatency(int64_t latency) override { latency_ = latency; }
  void SetII(int64_t ii) override { ii_ = ii; }

  // Helper methods to add regions (delegate to child)
  void AddReadRegion(const BufferRegion &region) override {
    if (child)
      child->AddReadRegion(region);
  }
  void AddWriteRegion(const BufferRegion &region) override {
    if (child)
      child->AddWriteRegion(region);
  }

  // Clone method
  std::unique_ptr<IRStructure> Clone() const override;

private:
  // Latency estimation
  int64_t latency_{0}; // Estimated latency in cycles
  int64_t ii_{0};      // Initiation interval in cycles
};

// Wrapper node: contains a Wrapper statement with variable, value, and child
// IRStructure
class WrapperNode : public IRStructure {
public:
  Stmt wrapper;
  std::unique_ptr<IRStructure> child;

  Kind GetKind() const override { return Kind::kWrapper; }

  // Resource usage flags (aggregate from child)
  bool UsesCUDACore() const override {
    return child ? child->UsesCUDACore() : false;
  }
  bool UsesTMACore() const override {
    return child ? child->UsesTMACore() : false;
  }
  bool UsesTensorCore() const override {
    return child ? child->UsesTensorCore() : false;
  }

  // Memory access regions (aggregate from child)
  std::vector<BufferRegion> GetReadRegions() const override {
    return child ? child->GetReadRegions() : std::vector<BufferRegion>{};
  }
  std::vector<BufferRegion> GetWriteRegions() const override {
    return child ? child->GetWriteRegions() : std::vector<BufferRegion>{};
  }

  // Latency estimation (aggregate from child)
  int64_t GetLatency() const override { return latency_; }
  int64_t GetII() const override { return ii_; }

  // Setters (delegate to child if exists)
  void SetUsesCUDACore(bool value) override {
    if (child)
      child->SetUsesCUDACore(value);
  }
  void SetUsesTMACore(bool value) override {
    if (child)
      child->SetUsesTMACore(value);
  }
  void SetUsesTensorCore(bool value) override {
    if (child)
      child->SetUsesTensorCore(value);
  }
  void SetReadRegions(const std::vector<BufferRegion> &regions) override {
    if (child)
      child->SetReadRegions(regions);
  }
  void SetWriteRegions(const std::vector<BufferRegion> &regions) override {
    if (child)
      child->SetWriteRegions(regions);
  }
  void SetLatency(int64_t latency) override { latency_ = latency; }
  void SetII(int64_t ii) override { ii_ = ii; }

  // Helper methods to add regions (delegate to child)
  void AddReadRegion(const BufferRegion &region) override {
    if (child)
      child->AddReadRegion(region);
  }
  void AddWriteRegion(const BufferRegion &region) override {
    if (child)
      child->AddWriteRegion(region);
  }

  // Clone method
  std::unique_ptr<IRStructure> Clone() const override;

private:
  // Latency estimation
  int64_t latency_{0}; // Estimated latency in cycles
  int64_t ii_{0};      // Initiation interval in cycles
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
  void SetReadRegions(const std::vector<BufferRegion> &regions) override;
  void SetWriteRegions(const std::vector<BufferRegion> &regions) override;
  void SetLatency(int64_t latency) override;
  void SetII(int64_t ii) override;

  // Helper methods to add regions (delegate to first child if exists)
  void AddReadRegion(const BufferRegion &region) override;
  void AddWriteRegion(const BufferRegion &region) override;

  // Clone method
  std::unique_ptr<IRStructure> Clone() const override;

private:
  // Latency estimation
  int64_t latency_{0}; // Estimated latency in cycles
  int64_t ii_{0};      // Initiation interval in cycles
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

// Structure to store TaskNode with its loop context information
struct TaskNodeWithContext {
  TaskNode *task;
  ControlNode *control_node{nullptr}; // nullptr if not inside a loop
  int64_t tripcount{1}; // tripcount multiplier for weighted latency
};

// Structure for component information used in warpgroup assignment
struct ComponentInfo {
  int root;
  int64_t weighted_latency; // total weighted latency in this component
  std::vector<int> task_indices;
};

// Helper function to check if a buffer region is in register memory
inline bool IsRegisterRegion(const BufferRegion &region) {
  const Buffer &buffer = region->buffer;
  String scope = buffer.scope();
  MemoryType mem_type = GetMemoryTypeFromScope(scope);
  return mem_type == MemoryType::kRegister;
}

// Helper function to collect all register regions from an IRStructure node
inline std::vector<BufferRegion>
CollectRegisterRegions(const IRStructure *node) {
  std::vector<BufferRegion> reg_regions;
  // Check read regions
  for (const auto &region : node->GetReadRegions()) {
    if (IsRegisterRegion(region)) {
      reg_regions.push_back(region);
    }
  }
  // Check write regions
  for (const auto &region : node->GetWriteRegions()) {
    if (IsRegisterRegion(region)) {
      reg_regions.push_back(region);
    }
  }
  return reg_regions;
}

// Helper function to check if two IRStructure nodes use the same register
// region
inline bool UseSameRegisterRegion(const IRStructure *a, const IRStructure *b) {
  if (!a || !b)
    return false;

  auto reg_regions_a = CollectRegisterRegions(a);
  auto reg_regions_b = CollectRegisterRegions(b);

  // For each pair of register regions, check if they refer to the same buffer
  // If buffer is the same, consider it as the same region (simplified)
  for (const auto &region_a : reg_regions_a) {
    for (const auto &region_b : reg_regions_b) {
      // Check if same buffer
      if (region_a->buffer.same_as(region_b->buffer)) {
        // Buffer相同就认为是同一个region
        return true;
      }
    }
  }
  return false;
}

// Helper function to count register regions in an IRStructure node
inline int CountRegisterRegions(const IRStructure *node) {
  auto reg_regions = CollectRegisterRegions(node);
  return static_cast<int>(reg_regions.size());
}

// Helper function to collect all TaskNodes with context information
void CollectAllTaskNodesWithContext(
    const IRStructure *node, std::vector<TaskNodeWithContext> &all_tasks,
    ControlNode *current_control_node = nullptr);

// Global warpgroup id assignment - should be called from the top level
// Tasks that use the same register region must have the same warpgroup id
// Goal: balance weighted latency between two warpgroups (0 and 1)
// Weighted latency = latency * tripcount (tripcount = 100 for non-constant loop
// extent)
void AssignWarpgroupIdsGlobal(IRStructure *root);

// Helper function to print BufferRegion details
void PrintBufferRegion(const BufferRegion &region, const std::string &indent) {
  const Buffer &buffer = region->buffer;
  const Region &ranges = region->region;

  std::string buffer_name = buffer->name;
  if (buffer_name.empty())
    buffer_name = "unnamed_buffer";

  LOG(INFO) << indent << "Buffer: " << buffer_name;

  // Get scope information
  String scope_str = buffer.scope();
  LOG(INFO) << indent << "  Scope: " << scope_str;

  // Build shape string
  std::ostringstream shape_ss;
  shape_ss << "[";
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    if (i > 0)
      shape_ss << ", ";
    shape_ss << buffer->shape[i];
  }
  shape_ss << "]";
  LOG(INFO) << indent << "  Shape: " << shape_ss.str();

  LOG(INFO) << indent << "  Region:";
  for (size_t i = 0; i < ranges.size(); ++i) {
    const Range &range = ranges[i];
    std::ostringstream range_ss;
    range_ss << "[" << range->min << ", " << range->min + range->extent
             << ") (extent=" << range->extent << ")";
    LOG(INFO) << indent << "    dim " << i << ": " << range_ss.str();
  }
}

// Helper function to print ScheduleUnits

// Helper function to print all stmts in IRStructure
void PrintAllStmts(const IRStructure *node, int indent = 0) {
  if (!node)
    return;

  std::string indent_str(indent * 2, ' ');

  if (node->IsTask()) {
    const TaskNode *task = static_cast<const TaskNode *>(node);
    LOG(INFO) << indent_str << "TaskNode with " << task->stmts.size()
              << " statements:";
    for (size_t i = 0; i < task->stmts.size(); i++) {
      LOG(INFO) << indent_str << "  Statement " << i << ":";
      LOG(INFO) << indent_str + "    " << task->stmts[i];
    }
    LOG(INFO) << indent_str << "  Resource usage: CUDA=" << task->UsesCUDACore()
              << ", TMA=" << task->UsesTMACore()
              << ", Tensor=" << task->UsesTensorCore();
    LOG(INFO) << indent_str << "  Latency: " << task->GetLatency()
              << " cycles, II: " << task->GetII()
              << " cycles, warpgroup_id: " << task->GetWarpgroupId()
              << ", promote: " << task->GetPromote();
  } else if (node->IsControl()) {
    const ControlNode *control = static_cast<const ControlNode *>(node);
    LOG(INFO) << indent_str << "ControlNode (For loop):";
    // Print the For statement itself
    LOG(INFO) << indent_str << "  For statement:";
    LOG(INFO) << indent_str + "    " << control->control;

    // Recursively print child statements
    if (control->child) {
      LOG(INFO) << indent_str << "  Loop body:";
      PrintAllStmts(control->child.get(), indent + 2);
    }
  } else if (node->IsSequence()) {
    const SequenceNode *seq = static_cast<const SequenceNode *>(node);
    LOG(INFO) << indent_str << "SequenceNode with " << seq->children.size()
              << " children:";
    for (size_t i = 0; i < seq->children.size(); i++) {
      LOG(INFO) << indent_str << "  Child " << i << ":";
      PrintAllStmts(seq->children[i].get(), indent + 2);
    }
  } else if (node->GetKind() == IRStructure::Kind::kWrapper) {
    const WrapperNode *wrapper = static_cast<const WrapperNode *>(node);
    LOG(INFO) << indent_str << "WrapperNode:";
    LOG(INFO) << indent_str << "  Wrapper: " << wrapper->wrapper;
    // Recursively print child statements
    if (wrapper->child) {
      LOG(INFO) << indent_str << "  Wrapper body:";
      PrintAllStmts(wrapper->child.get(), indent + 2);
    }
  }
}

// Original helper function to print IRStructure (kept for backward
// compatibility)
void PrintIRStructure(const IRStructure *node, int indent = 0) {
  if (!node)
    return;

  std::string indent_str(indent * 2, ' ');

  if (node->IsTask()) {
    const TaskNode *task = static_cast<const TaskNode *>(node);
    LOG(INFO) << indent_str << "TaskNode:";
    LOG(INFO) << indent_str << "  stmts: " << task->stmts.size()
              << " statements";
    for (auto &stmt : task->stmts) {
      LOG(INFO) << indent_str << "  stmt: " << stmt;
    }
    for (auto &region : task->GetReadRegions()) {
      LOG(INFO) << indent_str << "  Read Region: " << region;
    }
    for (auto &region : task->GetWriteRegions()) {
      LOG(INFO) << indent_str << "  Write Region: " << region;
    }
    LOG(INFO) << indent_str << "  uses_cuda_core: " << task->UsesCUDACore();
    LOG(INFO) << indent_str << "  uses_tma_core: " << task->UsesTMACore();
    LOG(INFO) << indent_str << "  uses_tensor_core: " << task->UsesTensorCore();
    LOG(INFO) << indent_str << "  latency: " << task->GetLatency() << " cycles";
    LOG(INFO) << indent_str << "  II: " << task->GetII() << " cycles";
    LOG(INFO) << indent_str << "  warpgroup_id: " << task->GetWarpgroupId();
    LOG(INFO) << indent_str << "  promote: " << task->GetPromote();
  } else if (node->IsControl()) {
    const ControlNode *control = static_cast<const ControlNode *>(node);
    LOG(INFO) << indent_str << "ControlNode (For loop):";
    // Could print loop info if needed
    if (control->child) {
      LOG(INFO) << indent_str << "  Child:";
      PrintIRStructure(control->child.get(), indent + 2);
    }
  } else if (node->IsSequence()) {
    const SequenceNode *seq = static_cast<const SequenceNode *>(node);
    LOG(INFO) << indent_str << "SequenceNode: " << seq->children.size()
              << " children";
    for (size_t i = 0; i < seq->children.size(); i++) {
      LOG(INFO) << indent_str << "  Child " << i << ":";
      PrintIRStructure(seq->children[i].get(), indent + 2);
    }
  } else if (node->GetKind() == IRStructure::Kind::kWrapper) {
    const WrapperNode *wrapper = static_cast<const WrapperNode *>(node);
    LOG(INFO) << indent_str << "WrapperNode:";
    LOG(INFO) << indent_str << "  Wrapper: " << wrapper->wrapper;
    if (wrapper->child) {
      LOG(INFO) << indent_str << "  Child:";
      PrintIRStructure(wrapper->child.get(), indent + 2);
    }
  }
}

// Latency estimator for H100 GPU
class LatencyEstimator {
public:
  // H100 latency parameters (in cycles)
  struct H100Params {
    // Base latencies
    int64_t global_memory_read = 400; // Global memory read latency
    int64_t global_memory_write =
        200; // Global memory write latency (usually lower)
    int64_t shared_memory_read = 20;  // Shared memory read latency
    int64_t shared_memory_write = 20; // Shared memory write latency
    int64_t register_access = 1;      // Register access latency
    int64_t cuda_core_operation =
        4; // Basic CUDA core operation (add, mul, etc.)
    int64_t tensor_core_operation =
        64;                      // Tensor core operation (matrix multiply)
    int64_t tma_operation = 100; // TMA operation latency

    // Bandwidth parameters (bytes per cycle)
    // H100: ~2TB/s global memory, 1.8GHz clock → ~1111 bytes/cycle
    // H100: ~19TB/s shared memory → ~10556 bytes/cycle
    int64_t global_memory_bandwidth = 1111;  // bytes per cycle
    int64_t shared_memory_bandwidth = 10556; // bytes per cycle

    // Pipeline initiation capabilities
    int64_t max_memory_ops_per_cycle =
        1; // Max memory ops that can start per cycle
  };

  LatencyEstimator() = default;

  // Estimate latency for a TaskNode
  void Estimate(TaskNode *task) {
    int64_t total_latency = 0;
    int64_t memory_latency = 0;
    int64_t compute_latency = 0;

    // Count memory operations and track bytes by memory type
    int num_memory_ops = 0;
    int64_t global_memory_bytes = 0;
    int64_t shared_memory_bytes = 0;
    int64_t register_bytes = 0;

    // Estimate latency from memory accesses and track bandwidth usage
    for (const auto &region : task->GetReadRegions()) {
      int64_t region_latency =
          EstimateMemoryAccessLatency(region, true); // read
      memory_latency += region_latency;
      num_memory_ops++;

      // Track bandwidth usage by memory type
      const Buffer &buffer = region->buffer;
      String scope = buffer.scope();
      MemoryType mem_type = GetMemoryTypeFromScope(scope);
      int64_t access_bytes = CalculateAccessBytes(region);

      switch (mem_type) {
      case MemoryType::kGlobal:
        global_memory_bytes += access_bytes;
        break;
      case MemoryType::kShared:
        shared_memory_bytes += access_bytes;
        break;
      case MemoryType::kRegister:
        register_bytes += access_bytes;
        break;
      default:
        global_memory_bytes += access_bytes; // Conservative
        break;
      }
    }

    for (const auto &region : task->GetWriteRegions()) {
      int64_t region_latency =
          EstimateMemoryAccessLatency(region, false); // write
      memory_latency += region_latency;
      num_memory_ops++;

      // Track bandwidth usage by memory type
      const Buffer &buffer = region->buffer;
      String scope = buffer.scope();
      MemoryType mem_type = GetMemoryTypeFromScope(scope);
      int64_t access_bytes = CalculateAccessBytes(region);

      switch (mem_type) {
      case MemoryType::kGlobal:
        global_memory_bytes += access_bytes;
        break;
      case MemoryType::kShared:
        shared_memory_bytes += access_bytes;
        break;
      case MemoryType::kRegister:
        register_bytes += access_bytes;
        break;
      default:
        global_memory_bytes += access_bytes; // Conservative
        break;
      }
    }

    // Estimate compute latency based on resource usage
    if (task->UsesCUDACore()) {
      // Simple heuristic: assume some number of CUDA operations
      // For now, assume 1 operation per statement as a rough estimate
      compute_latency = params_.cuda_core_operation *
                        std::max(1, static_cast<int>(task->stmts.size()));
    }

    if (task->UsesTensorCore()) {
      compute_latency =
          std::max(compute_latency, params_.tensor_core_operation);
    }

    if (task->UsesTMACore()) {
      compute_latency = std::max(compute_latency, params_.tma_operation);
    }

    // Total latency is sum of memory and compute (assuming sequential for now)
    total_latency = memory_latency + compute_latency;

    // Calculate initiation interval (II)
    int64_t ii = 1; // Default minimum II

    if (task->UsesTMACore()) {
      // TMA operations (async memory copies): instruction latency can be hidden
      // II is determined by bandwidth constraints only
      if (global_memory_bytes > 0) {
        int64_t bandwidth_ii =
            (global_memory_bytes + params_.global_memory_bandwidth - 1) /
            params_.global_memory_bandwidth;
        ii = std::max(ii, bandwidth_ii);
      }
      if (shared_memory_bytes > 0) {
        int64_t bandwidth_ii =
            (shared_memory_bytes + params_.shared_memory_bandwidth - 1) /
            params_.shared_memory_bandwidth;
        ii = std::max(ii, bandwidth_ii);
      }
    } else {
      // Regular operations
      // According to requirements:
      // 1. If there's only one operation and it's a memory access, II = memory
      // latency
      // 2. Otherwise, II = total latency
      ii = total_latency;

      if (num_memory_ops == 1 && task->stmts.size() == 1) {
        // Single operation that is a memory access
        // Check if this is likely a memory operation (has read/write regions)
        if (!task->GetReadRegions().empty() ||
            !task->GetWriteRegions().empty()) {
          ii = memory_latency;
        }
      }

      // Additional II constraints from bandwidth limitations
      // II must be at least the time needed to transfer data based on bandwidth
      if (global_memory_bytes > 0) {
        int64_t bandwidth_ii =
            (global_memory_bytes + params_.global_memory_bandwidth - 1) /
            params_.global_memory_bandwidth;
        ii = std::max(ii, bandwidth_ii);
      }

      if (shared_memory_bytes > 0) {
        int64_t bandwidth_ii =
            (shared_memory_bytes + params_.shared_memory_bandwidth - 1) /
            params_.shared_memory_bandwidth;
        ii = std::max(ii, bandwidth_ii);
      }
    }

    // II must be at least 1 cycle
    if (ii < 1)
      ii = 1;

    // Store results in task node
    task->SetLatency(total_latency);
    task->SetII(ii);
  }

private:
  H100Params params_;

  // Helper function to calculate total bytes accessed in a region
  int64_t CalculateAccessBytes(const BufferRegion &region) {
    const Buffer &buffer = region->buffer;
    const Region &ranges = region->region;

    // Calculate total number of elements
    int64_t total_elements = 1;
    for (const auto &range : ranges) {
      // Try to get constant extent if possible
      if (const auto *extent_int = range->extent.as<IntImmNode>()) {
        total_elements *= extent_int->value;
      } else {
        // For non-constant extents, use a conservative estimate
        // Assume at least 1 element per dimension
        // TODO: Better estimation for symbolic extents
        total_elements *= 1;
      }
    }

    // Get data type size in bytes
    DataType dtype(buffer->dtype);
    int64_t element_size = dtype.bytes();

    return total_elements * element_size;
  }

  // Estimate latency for a single memory access
  int64_t EstimateMemoryAccessLatency(const BufferRegion &region,
                                      bool is_read) {
    const Buffer &buffer = region->buffer;
    String scope = buffer.scope();
    MemoryType mem_type = GetMemoryTypeFromScope(scope);

    int64_t access_bytes = CalculateAccessBytes(region);

    switch (mem_type) {
    case MemoryType::kGlobal:
      // Global memory latency depends on data size
      // Base latency + bandwidth-limited component
      // Latency = base_latency + bytes / bytes_per_cycle
      // Subtract cache line size (32 bytes) since first cache line has base
      // latency
      if (is_read) {
        // Base read latency + bandwidth component
        return params_.global_memory_read +
               std::max(0L,
                        (access_bytes - 32) / params_.global_memory_bandwidth);
      } else {
        // Write latency usually lower
        return params_.global_memory_write +
               std::max(0L,
                        (access_bytes - 32) / params_.global_memory_bandwidth);
      }
    case MemoryType::kShared:
      // Shared memory has high bandwidth, less sensitive to size
      // Subtract typical burst size (128 bytes) for base latency
      if (is_read) {
        return params_.shared_memory_read +
               std::max(0L,
                        (access_bytes - 128) / params_.shared_memory_bandwidth);
      } else {
        return params_.shared_memory_write +
               std::max(0L,
                        (access_bytes - 128) / params_.shared_memory_bandwidth);
      }
    case MemoryType::kRegister:
      // Register access latency is constant and very small
      return params_.register_access;
    default:
      // Unknown memory type, use global memory as conservative estimate
      if (is_read) {
        return params_.global_memory_read +
               std::max(0L,
                        (access_bytes - 32) / params_.global_memory_bandwidth);
      } else {
        return params_.global_memory_write +
               std::max(0L,
                        (access_bytes - 32) / params_.global_memory_bandwidth);
      }
    }
  }
};

// MemoryAccessDetector: detect read/write regions in statements
// Adapted from BlockReadWriteDetector in TVM
class MemoryAccessDetector : public StmtExprVisitor {
public:
  MemoryAccessDetector() = default;

  // Analyze a statement and collect read/write regions
  void Analyze(const Stmt &stmt) {
    read_buffers_.clear();
    write_buffers_.clear();
    read_regions_.clear();
    write_regions_.clear();
    dom_map_.clear();
    hint_map_.clear();
    pending_conditions_.clear();
    let_bindings_.clear();
    operator()(stmt);
  }

  // Return collected read regions
  std::vector<BufferRegion> GetReadRegions() const {
    return CollectRegions(read_buffers_, read_regions_);
  }

  // Return collected write regions
  std::vector<BufferRegion> GetWriteRegions() const {
    return CollectRegions(write_buffers_, write_regions_);
  }

private:
  /*! \brief Iteration range for loop_vars */
  std::unordered_map<const VarNode *, arith::IntSet> dom_map_;
  /*! \brief Extra iteration range hint for free vars */
  std::unordered_map<const VarNode *, arith::IntSet> hint_map_;
  /*! \brief Unresolved conditions within current scope. */
  std::vector<PrimExpr> pending_conditions_;
  /*! \brief The buffers that the current block reads */
  std::vector<Buffer> read_buffers_;
  /*! \brief The buffers that the current block writes */
  std::vector<Buffer> write_buffers_;
  /*! \brief The read regions of the current block */
  std::vector<std::vector<tvm::arith::IntSet>> read_regions_;
  /*! \brief The write regions of the current block */
  std::vector<std::vector<tvm::arith::IntSet>> write_regions_;
  /*!\ brief Internal analyzer. */
  arith::Analyzer ana_;
  /*! \brief let bindings inside the block */
  std::unordered_map<const VarNode *, PrimExpr> let_bindings_;

  /*!
   * \brief Update read/write buffers and regions with provided buffer and
   * region
   */
  void Update(std::vector<Buffer> *buffers,
              std::vector<std::vector<arith::IntSet>> *regions, Buffer buffer,
              std::vector<arith::IntSet> region) {
    // Check if buffer already exists
    for (size_t i = 0; i < buffers->size(); ++i) {
      if ((*buffers)[i].same_as(buffer)) {
        // Merge regions
        ICHECK_EQ((*regions)[i].size(), region.size());
        for (size_t j = 0; j < region.size(); ++j) {
          (*regions)[i][j] = arith::Union({(*regions)[i][j], region[j]});
        }
        return;
      }
    }
    // New buffer
    buffers->push_back(buffer);
    regions->push_back(region);
  }

  /*!
   * \brief Process a buffer region argument from reduce operation
   * \param arg The argument which could be BufferRegion, BufferLoad, or
   * tl.tileop.region call
   * \param is_read Whether this is a read (true) or write (false) access
   */
  void ProcessBufferRegion(const PrimExpr &arg, bool is_read) {
    // Check if it's a BufferRegion
    if (const auto *buffer_region = arg.as<BufferRegionNode>()) {
      Buffer buffer = buffer_region->buffer;
      const Region &region = buffer_region->region;
      std::vector<arith::IntSet> int_sets;
      int_sets.reserve(region.size());
      for (const auto &range : region) {
        // Create IntSet for range [min, min + extent)
        int_sets.push_back(arith::IntSet::FromRange(
            Range::FromMinExtent(range->min, range->extent)));
      }
      if (is_read) {
        Update(&read_buffers_, &read_regions_, buffer, int_sets);
      } else {
        Update(&write_buffers_, &write_regions_, buffer, int_sets);
      }
      return;
    }

    // Check if it's a BufferLoad
    if (const auto *buffer_load = arg.as<BufferLoadNode>()) {
      Buffer buffer = buffer_load->buffer;
      std::vector<arith::IntSet> int_sets;
      int_sets.reserve(buffer_load->indices.size());
      for (PrimExpr index : buffer_load->indices) {
        // Create IntSet for single point
        int_sets.push_back(RelaxAccessIndex(index));
      }
      if (is_read) {
        Update(&read_buffers_, &read_regions_, buffer, int_sets);
      } else {
        Update(&write_buffers_, &write_regions_, buffer, int_sets);
      }
      return;
    }

    // Check if it's a tl.tileop.region call (should already be handled by
    // VisitExpr_) but we can still process it recursively
    if (const auto *call = arg.as<CallNode>()) {
      static const auto region_op = Op::Get("tl.tileop.region");
      if (call->op.same_as(region_op)) {
        // Recursively visit this call to handle it
        VisitExpr_(call);
        return;
      }
    }

    // If we reach here, the argument type is not supported
    LOG(WARNING) << "Unsupported argument type in tl.tileop.reduce: "
                 << arg->GetTypeKey();
  }

  /*! \brief Helper function to collect access regions. */
  std::vector<BufferRegion> CollectRegions(
      const std::vector<Buffer> &buffers,
      const std::vector<std::vector<tvm::arith::IntSet>> &regions) const {
    std::vector<BufferRegion> result;
    result.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
      const Buffer &buffer = buffers[i];
      const std::vector<arith::IntSet> &int_sets = regions[i];
      Region region;
      size_t ndim = buffer->shape.size();
      size_t region_ndim = int_sets.size();

      // Assert that region dimension equals buffer dimension
      ICHECK_EQ(region_ndim, ndim) << "Region dimension " << region_ndim
                                   << " must equal buffer dimension " << ndim;

      region.reserve(ndim);
      for (size_t j = 0; j < ndim; ++j) {
        const tvm::arith::IntSet &int_set = int_sets[j];
        region.push_back(
            int_set.CoverRange(Range::FromMinExtent(0, buffer->shape[j])));
      }

      result.push_back(BufferRegion(buffer, region));
    }
    return result;
  }

  /*! \brief Helper function to relax the buffer indices */
  arith::IntSet RelaxAccessIndex(const PrimExpr &index) {
    PrimExpr current = index;
    PrimExpr remapped = Substitute(current, let_bindings_);
    while (!remapped.same_as(current)) {
      current = remapped;
      remapped = Substitute(current, let_bindings_);
    }
    return arith::EvalSet(arith::IntSet::Vector(current), dom_map_);
  }

  void operator()(const Stmt &stmt) { StmtExprVisitor::operator()(stmt); }

  void VisitStmt_(const ForNode *op) override {
    Range range = Range::FromMinExtent(op->min, op->extent);
    dom_map_[op->loop_var.get()] = arith::IntSet::FromRange(range);
    StmtVisitor::VisitStmt_(op);
    dom_map_.erase(op->loop_var.get());
  }

  void VisitStmt_(const IfThenElseNode *op) override {
    VisitExpr(op->condition);
    {
      // Visit then branch
      // Simplified: we don't handle conditional bounds for now
      StmtExprVisitor::VisitStmt(op->then_case);
    }
    if (op->else_case) {
      // Visit else branch
      StmtExprVisitor::VisitStmt(op->else_case.value());
    }
  }

  void VisitStmt_(const LetStmtNode *op) override {
    let_bindings_[op->var.get()] = op->value;
    StmtVisitor::VisitStmt_(op);
    let_bindings_.erase(op->var.get());
  }

  void VisitExpr_(const BufferLoadNode *op) override {
    std::vector<arith::IntSet> relaxed_region;
    size_t num_indices = op->indices.size();
    size_t buffer_ndim = op->buffer->shape.size();

    // Assert that indices count equals buffer dimension
    ICHECK_EQ(num_indices, buffer_ndim)
        << "BufferLoad indices count " << num_indices
        << " must equal buffer dimension " << buffer_ndim;

    for (PrimExpr index : op->indices) {
      relaxed_region.push_back(RelaxAccessIndex(index));
    }
    Update(&read_buffers_, &read_regions_, op->buffer, relaxed_region);
    ExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) override {
    std::vector<arith::IntSet> relaxed_region;
    size_t num_indices = op->indices.size();
    size_t buffer_ndim = op->buffer->shape.size();

    // Assert that indices count equals buffer dimension
    ICHECK_EQ(num_indices, buffer_ndim)
        << "BufferStore indices count " << num_indices
        << " must equal buffer dimension " << buffer_ndim;

    for (PrimExpr index : op->indices) {
      relaxed_region.push_back(RelaxAccessIndex(index));
    }
    Update(&write_buffers_, &write_regions_, op->buffer, relaxed_region);
    StmtVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) override {
    static const auto region_op = Op::Get("tl.tileop.region");
    static const auto reduce_op = Op::Get("tl.tileop.reduce");

    // Check for tl.tileop.region call
    if (op->op.same_as(region_op)) {
      // Handle tl.tileop.region call for memory access analysis
      // args[0] = buffer (BufferLoad), args[1] = access_type (1: read, 2:
      // write, 3: read/write) args[2..] = extents
      if (op->args.size() >= 2) {
        // Extract access type
        const auto *access_int = op->args[1].as<IntImmNode>();
        ICHECK(access_int);
        int access_type = access_int->value;

        // Extract buffer from BufferLoad
        if (const auto *buffer_load = op->args[0].as<BufferLoadNode>()) {
          Buffer buffer = buffer_load->buffer;
          std::vector<arith::IntSet> relaxed_region;

          // Assert that BufferLoad accesses a single element (no Ramp indices)
          for (size_t i = 0; i < buffer_load->indices.size(); ++i) {
            const PrimExpr &index = buffer_load->indices[i];
            // Check if index is a Ramp (vector access)
            if (index.as<RampNode>()) {
              LOG(FATAL) << "BufferLoad in tl.tileop.region should access a "
                            "single element, "
                         << "but found Ramp index at dimension " << i;
            }
          }

          // Use provided extents if available, otherwise use buffer load
          // indices
          size_t num_indices = buffer_load->indices.size();
          size_t buffer_ndim = buffer->shape.size();

          // Assert that indices count equals buffer dimension
          ICHECK_EQ(num_indices, buffer_ndim)
              << "BufferLoad indices count " << num_indices
              << " must equal buffer dimension " << buffer_ndim;

          if (op->args.size() > 2) {
            // args[2..] are extents for the region
            // Number of extents provided
            size_t num_extents = op->args.size() - 2;

            // Assert that extents count equals indices count
            ICHECK_EQ(num_extents, num_indices)
                << "Extents count " << num_extents
                << " must equal indices count " << num_indices;

            relaxed_region.reserve(num_indices);
            for (size_t i = 0; i < num_indices; ++i) {
              PrimExpr min = buffer_load->indices[i];
              PrimExpr extent = op->args[2 + i];

              // Create IntSet for range [min, min + extent)
              relaxed_region.push_back(
                  arith::IntSet::FromRange(Range::FromMinExtent(min, extent)));
            }
          } else {
            // No extents provided: each dimension is a single point at the
            // index
            for (PrimExpr index : buffer_load->indices) {
              // Create IntSet for single point
              relaxed_region.push_back(RelaxAccessIndex(index));
            }
          }

          // Add to appropriate list based on access type
          if (access_type == 1 || access_type == 3) { // read or read/write
            Update(&read_buffers_, &read_regions_, buffer, relaxed_region);
          }
          if (access_type == 2 || access_type == 3) { // write or read/write
            Update(&write_buffers_, &write_regions_, buffer, relaxed_region);
          }
        } else {
          LOG(FATAL)
              << "First argument of tl.tileop.region should be a BufferLoad";
        }
      }
      return;
    }

    // Check for tl.tileop.reduce call
    if (op->op.same_as(reduce_op)) {
      // Handle tl.tileop.reduce call for memory access analysis
      // args[0] = input buffer region (read)
      // args[1] = output buffer region (write)
      // args[2] = reduce_type (string)
      // args[3] = dim (int)
      // args[4] = clear (bool)
      if (op->args.size() >= 2) {
        // Process first argument as read region
        ProcessBufferRegion(op->args[0], true); // is_read = true
        // Process second argument as write region
        ProcessBufferRegion(op->args[1], false); // is_read = false
      }
      return;
    }

    // Handle other calls (e.g., builtin::tvm_access_ptr)
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      // Simplified: skip for now
      StmtExprVisitor::VisitExpr_(op);
      return;
    }

    StmtExprVisitor::VisitExpr_(op);
  }

  // Skip block-specific handling for now
  void VisitStmt_(const BlockRealizeNode *op) override {
    // Don't visit child blocks recursively
  }
};

} // namespace tl
} // namespace tvm
