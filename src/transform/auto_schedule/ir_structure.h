#pragma once

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

#include "../../op/gemm.h"

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
using ffi::String;

// Forward declarations
class IRStructure;
class TaskNode;
class ControlNode;
class SequenceNode;
class WrapperNode;

// Structure to store region access information with warpgroup id
struct RegionAccessInfo {
  BufferRegion region;
  bool is_write;    // true for write, false for read
  int warpgroup_id; // warpgroup id of the innermost TaskNode

  RegionAccessInfo(BufferRegion region, bool is_write, int warpgroup_id)
      : region(region), is_write(is_write), warpgroup_id(warpgroup_id) {}
};

// Helper function to compare if two regions are equal
inline bool RegionsEqual(const Region &a, const Region &b) {
  if (a.size() != b.size())
    return false;

  arith::Analyzer analyzer;
  for (size_t i = 0; i < a.size(); ++i) {
    // Check if min values are equal
    if (!analyzer.CanProveEqual(a[i]->min, b[i]->min)) {
      return false;
    }
    // Check if extent values are equal
    if (!analyzer.CanProveEqual(a[i]->extent, b[i]->extent)) {
      return false;
    }
  }
  return true;
}

// Base class for all IR nodes in scheduling
class IRStructure {
public:
  enum class Kind { kTask, kControl, kSequence, kWrapper, kSchedule };

  virtual ~IRStructure() = default;
  virtual Kind GetKind() const = 0;
  virtual std::shared_ptr<IRStructure> Clone() const = 0;

  // Helper methods for safe casting
  bool IsTask() const { return GetKind() == Kind::kTask; }
  bool IsControl() const { return GetKind() == Kind::kControl; }
  bool IsSequence() const { return GetKind() == Kind::kSequence; }
  bool IsWrapper() const { return GetKind() == Kind::kWrapper; }
  bool IsScheduleUnit() const { return GetKind() == Kind::kSchedule; }

  // Resource usage flags (accessible by all IR nodes)
  virtual bool UsesCUDACore() const = 0;
  virtual bool UsesTMACore() const = 0;
  virtual bool UsesTensorCore() const = 0;

  // Memory access regions (collected during analysis)
  virtual std::vector<BufferRegion> GetReadRegions() const = 0;
  virtual std::vector<BufferRegion> GetWriteRegions() const = 0;

  // Variable access (used for dependency analysis)
  virtual std::vector<Var> GetReadVars() const = 0;
  virtual std::vector<Var> GetWriteVars() const = 0;

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

  // Recursive region collection method
  virtual void CollectRegions(
      std::vector<RegionAccessInfo> &result,
      std::set<std::pair<Buffer, std::pair<int, int>>> &visited) const = 0;

  std::vector<RegionAccessInfo> GetReadWriteRegions() const {
    std::vector<RegionAccessInfo> result;
    std::set<std::pair<Buffer, std::pair<int, int>>> visited;
    CollectRegions(result, visited);
    return result;
  }

  // Substitute a variable throughout this IR node
  virtual void SubstituteVar(const Var &old_var, const Var &new_var) = 0;

  // Get warpgroup id for this node (-1 if not applicable)
  virtual int GetWarpgroupId() const { return -1; }

  virtual bool containWarpgroupId(int id) const = 0;

  // Start time for scheduling
  void SetStartTime(int64_t start_time) { start_time_ = start_time; }
  int64_t GetStartTime() const { return start_time_; }
  int64_t start_time_{0}; // Scheduled start time in cycles
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
  std::vector<Var> GetReadVars() const override { return read_vars_; }
  std::vector<Var> GetWriteVars() const override { return write_vars_; }
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
  void SetReadVars(const std::vector<Var> &vars) { read_vars_ = vars; }
  void SetWriteVars(const std::vector<Var> &vars) { write_vars_ = vars; }
  void SubstituteVar(const Var &old_var, const Var &new_var) override {
    for (auto &var : read_vars_) {
      if (var.same_as(old_var)) {
        var = new_var;
      }
    }
    for (auto &var : write_vars_) {
      if (var.same_as(old_var)) {
        var = new_var;
      }
    }
    for (size_t i = 0; i < stmts.size(); ++i) {
      stmts[i] = Substitute(stmts[i], {{old_var, new_var}});
    }
  }
  void SetLatency(int64_t latency) override { latency_ = latency; }
  void SetII(int64_t ii) override { ii_ = ii; }

  // Warpgroup id for warpgroup specialization
  void SetWarpgroupId(int warpgroup_id) { warpgroup_id_ = warpgroup_id; }
  int GetWarpgroupId() const override { return warpgroup_id_; }

  // TMA load flag
  void SetHasTMALoad(bool value) { has_tma_load_ = value; }
  bool HasTMALoad() const { return has_tma_load_; }

  // Tensor Core shape information structure
  struct TensorCoreShape {
    int64_t m;
    int64_t n;
    int64_t k;
    TensorCoreShape(int64_t m = 0, int64_t n = 0, int64_t k = 0)
        : m(m), n(n), k(k) {}
  };
  void AddTensorCoreShape(int64_t m, int64_t n, int64_t k) {
    tensor_core_shapes_.emplace_back(m, n, k);
  }

  // Get shape information
  size_t GetTensorCoreShapeCount() const { return tensor_core_shapes_.size(); }
  const TensorCoreShape &GetTensorCoreShape(size_t index) const {
    return tensor_core_shapes_[index];
  }

  // Helper methods for single shape (backward compatibility)
  int64_t GetTensorCoreM() const {
    return tensor_core_shapes_.empty() ? 0 : tensor_core_shapes_[0].m;
  }
  int64_t GetTensorCoreN() const {
    return tensor_core_shapes_.empty() ? 0 : tensor_core_shapes_[0].n;
  }
  int64_t GetTensorCoreK() const {
    return tensor_core_shapes_.empty() ? 0 : tensor_core_shapes_[0].k;
  }
  bool HasTensorCoreShape() const { return !tensor_core_shapes_.empty(); }

  // GemmInst: the final tensor core instruction for this task.
  // All gemm operations within a single task must use the same instruction.
  void SetGemmInst(GemmInst inst) {
    ICHECK(!has_gemm_inst_ || gemm_inst_ == inst)
        << "All gemm operations in a task must use the same GemmInst, "
        << "but got " << GemmInstToString(gemm_inst_) << " and "
        << GemmInstToString(inst);
    gemm_inst_ = inst;
    has_gemm_inst_ = true;
  }
  bool HasGemmInst() const { return has_gemm_inst_; }
  GemmInst GetGemmInst() const {
    ICHECK(has_gemm_inst_) << "GemmInst not set";
    return gemm_inst_;
  }

  // Check if this task uses WGMMA instruction
  bool is_WGMMA() const {
    return has_gemm_inst_ && gemm_inst_ == GemmInst::kWGMMA;
  }

  // Check if this task uses TCGEN5MMA instruction
  bool is_TCGEN05() const {
    return has_gemm_inst_ && gemm_inst_ == GemmInst::kTCGEN5MMA;
  }

  // Get aggregated shape information for II estimation
  int64_t GetTotalTensorCoreOps() const {
    int64_t total_ops = 0;
    for (const auto &shape : tensor_core_shapes_) {
      total_ops += shape.m * shape.n * shape.k * 2; // 2 ops per element
    }
    return total_ops;
  }

  int64_t GetTotalWGMMATiles() const {
    int64_t total_tiles = 0;
    int64_t tile_m = 16, tile_n = 16, tile_k = 16;
    for (const auto &shape : tensor_core_shapes_) {
      int64_t num_tiles_m = (shape.m + tile_m - 1) / tile_m;
      int64_t num_tiles_n = (shape.n + tile_n - 1) / tile_n;
      int64_t num_tiles_k = (shape.k + tile_k - 1) / tile_k;
      total_tiles += num_tiles_m * num_tiles_n * num_tiles_k;
    }
    return total_tiles;
  }

  // Clone method
  std::shared_ptr<IRStructure> Clone() const override;

  // Helper methods to add regions (for incremental analysis)
  void AddReadRegion(const BufferRegion &region) {
    // Check for duplicate regions
    for (const auto &existing : read_regions_) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        return; // Region already exists
      }
    }
    read_regions_.push_back(region);
  }

  void AddWriteRegion(const BufferRegion &region) {
    // Check for duplicate regions
    for (const auto &existing : write_regions_) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        return; // Region already exists
      }
    }
    write_regions_.push_back(region);
  }

  void AddReadVar(const Var &var) { read_vars_.push_back(var); }
  void AddWriteVar(const Var &var) { write_vars_.push_back(var); }

  void CollectRegions(
      std::vector<RegionAccessInfo> &result,
      std::set<std::pair<Buffer, std::pair<int, int>>> &visited) const override;

  bool containWarpgroupId(int id) const override {
    return ContainsLoopBreak() || warpgroup_id_ == id;
  }

  // Check if this task contains loop_break call
  bool ContainsLoopBreak() const;

private:
  // Resource usage flags
  bool uses_cuda_core_{false};
  bool uses_tma_core_{false};
  bool uses_tensor_core_{false};

  // Memory access regions (collected during analysis)
  std::vector<BufferRegion> read_regions_;
  std::vector<BufferRegion> write_regions_;

  std::vector<Var> read_vars_;
  std::vector<Var> write_vars_;

  // Latency estimation
  int64_t latency_{0}; // Estimated latency in cycles
  int64_t ii_{0};      // Initiation interval in cycles
  int warpgroup_id_{
      -1}; // Warpgroup id for warpgroup specialization (-1 means unassigned)

  // TMA information
  bool has_tma_load_{false};

  // Tensor Core shape information (M, N, K dimensions)
  // A task may contain multiple Tensor Core operations with different shapes
  std::vector<TensorCoreShape> tensor_core_shapes_;

  // GemmInst: the final tensor core instruction for this task.
  // All gemm operations within a single task must use the same instruction.
  GemmInst gemm_inst_{GemmInst::kMMA};
  bool has_gemm_inst_{false};

  // Cached flag for loop_break detection
  mutable std::optional<bool> contains_loop_break_cache_;
};

// Control node: contains a For operation and a child IRStructure
class ControlNode : public IRStructure {
public:
  For control; // The For operation
  std::shared_ptr<TaskNode> task;
  std::shared_ptr<IRStructure> child;

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

  // Memory access regions (aggregate from child & task)
  std::vector<BufferRegion> GetReadRegions() const override {
    std::vector<BufferRegion> regions =
        child ? child->GetReadRegions() : std::vector<BufferRegion>{};
    if (task) {
      auto task_regions = task->GetReadRegions();
      regions.insert(regions.end(), task_regions.begin(), task_regions.end());
    }
    return regions;
  }
  std::vector<BufferRegion> GetWriteRegions() const override {
    std::vector<BufferRegion> regions =
        child ? child->GetWriteRegions() : std::vector<BufferRegion>{};
    if (task) {
      auto task_regions = task->GetWriteRegions();
      regions.insert(regions.end(), task_regions.begin(), task_regions.end());
    }
    return regions;
  }

  // Variable access (aggregate from child & task)
  std::vector<Var> GetReadVars() const override {
    std::vector<Var> vars = child ? child->GetReadVars() : std::vector<Var>{};
    if (task) {
      auto task_vars = task->GetReadVars();
      vars.insert(vars.end(), task_vars.begin(), task_vars.end());
    }
    return vars;
  }
  std::vector<Var> GetWriteVars() const override {
    std::vector<Var> vars = child ? child->GetWriteVars() : std::vector<Var>{};
    if (task) {
      auto task_vars = task->GetWriteVars();
      vars.insert(vars.end(), task_vars.begin(), task_vars.end());
    }
    return vars;
  }
  void SubstituteVar(const Var &old_var, const Var &new_var) override {
    if (child) {
      child->SubstituteVar(old_var, new_var);
    }
    if (task)
      task->SubstituteVar(old_var, new_var);
    // Also substitute in the For statement's min/extent/step
    For new_for = control;
    new_for.CopyOnWrite()->min = Substitute(control->min, {{old_var, new_var}});
    new_for.CopyOnWrite()->extent =
        Substitute(control->extent, {{old_var, new_var}});
    if (control->step.has_value()) {
      new_for.CopyOnWrite()->step =
          Substitute(control->step.value(), {{old_var, new_var}});
    }
    control = new_for;
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

  void CollectRegions(
      std::vector<RegionAccessInfo> &result,
      std::set<std::pair<Buffer, std::pair<int, int>>> &visited) const override;

  bool hasPromote() const { return has_promote_; }

  void SetPromote(bool promote) { has_promote_ = promote; }

  void SetIIperIter(int64_t ii) { ii_per_iter_ = ii; }

  int64_t GetIIperIter() const { return ii_per_iter_; }

  // Clone method
  std::shared_ptr<IRStructure> Clone() const override;

  bool containWarpgroupId(int id) const override {
    return child->containWarpgroupId(id);
  }

private:
  // Latency estimation
  int64_t latency_{0}; // Estimated latency in cycles
  int64_t ii_{0};      // Initiation interval in cycles
  bool has_promote_{false};
  int64_t ii_per_iter_{0};
};

// Wrapper node: contains a Wrapper statement with variable, value, and child
// IRStructure
class WrapperNode : public IRStructure {
public:
  Stmt wrapper;
  std::shared_ptr<IRStructure> child;
  std::shared_ptr<TaskNode> task;

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

  // Variable access (aggregate from child)
  std::vector<Var> GetReadVars() const override {
    return child ? child->GetReadVars() : std::vector<Var>{};
  }
  std::vector<Var> GetWriteVars() const override {
    return child ? child->GetWriteVars() : std::vector<Var>{};
  }

  void SubstituteVar(const Var &old_var, const Var &new_var) override {
    if (wrapper.defined()) {
      wrapper = Substitute(wrapper, {{old_var, new_var}});
    }
    if (child) {
      child->SubstituteVar(old_var, new_var);
    }
    if (task) {
      task->SubstituteVar(old_var, new_var);
    }
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

  void CollectRegions(
      std::vector<RegionAccessInfo> &result,
      std::set<std::pair<Buffer, std::pair<int, int>>> &visited) const override;

  // Clone method
  std::shared_ptr<IRStructure> Clone() const override;

  bool containWarpgroupId(int id) const override {
    return child->containWarpgroupId(id);
  }

private:
  // Latency estimation
  int64_t latency_{0}; // Estimated latency in cycles
  int64_t ii_{0};      // Initiation interval in cycles
};

class ScheduleUnit : public IRStructure {
public:
  int stage;
  std::vector<std::vector<Stmt>> before, after;
  std::shared_ptr<IRStructure> child;

  ScheduleUnit() {
    for (unsigned idx = 0; idx != 2; ++idx) {
      before.emplace_back();
      after.emplace_back();
    }
  }

  Kind GetKind() const override { return Kind::kSchedule; }

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

  std::vector<Var> GetReadVars() const override {
    return child ? child->GetReadVars() : std::vector<Var>{};
  }
  std::vector<Var> GetWriteVars() const override {
    return child ? child->GetWriteVars() : std::vector<Var>{};
  }

  void SubstituteVar(const Var &old_var, const Var &new_var) override {
    if (child) {
      child->SubstituteVar(old_var, new_var);
    }
    for (auto &stmts : before) {
      for (auto &stmt : stmts) {
        stmt = Substitute(stmt, {{old_var, new_var}});
      }
    }
    for (auto &stmts : after) {
      for (auto &stmt : stmts) {
        stmt = Substitute(stmt, {{old_var, new_var}});
      }
    }
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

  void CollectRegions(
      std::vector<RegionAccessInfo> &result,
      std::set<std::pair<Buffer, std::pair<int, int>>> &visited) const override;

  int GetStage() const { return stage; }
  bool isInnerTask() const { return child->IsTask(); }
  int GetWarpgroupId() const override {
    ICHECK(isInnerTask());
    const TaskNode *task = static_cast<const TaskNode *>(child.get());
    return task->GetWarpgroupId();
  }

  // Clone method
  std::shared_ptr<IRStructure> Clone() const override;

  bool containWarpgroupId(int id) const override {
    return child->containWarpgroupId(id);
  }

private:
  // Latency estimation
  int64_t latency_{0}; // Estimated latency in cycles
  int64_t ii_{0};      // Initiation interval in cycles
};

// Sequence node: contains a vector of child IRStructures
class SequenceNode : public IRStructure {
public:
  std::vector<std::shared_ptr<IRStructure>> children;

  Kind GetKind() const override { return Kind::kSequence; }

  // Resource usage flags (aggregate from all children)
  bool UsesCUDACore() const override;
  bool UsesTMACore() const override;
  bool UsesTensorCore() const override;

  // Memory access regions (aggregate from all children)
  std::vector<BufferRegion> GetReadRegions() const override;
  std::vector<BufferRegion> GetWriteRegions() const override;

  std::vector<Var> GetReadVars() const override;
  std::vector<Var> GetWriteVars() const override;

  void SubstituteVar(const Var &old_var, const Var &new_var) override {
    for (auto &child : children) {
      child->SubstituteVar(old_var, new_var);
    }
  }

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

  void CollectRegions(
      std::vector<RegionAccessInfo> &result,
      std::set<std::pair<Buffer, std::pair<int, int>>> &visited) const override;

  // Clone method
  std::shared_ptr<IRStructure> Clone() const override;

  bool containWarpgroupId(int id) const override {
    for (auto &child : children) {
      if (child->containWarpgroupId(id)) {
        return true;
      }
    }
    return false;
  }

private:
  // Latency estimation
  int64_t latency_{0}; // Estimated latency in cycles
  int64_t ii_{0};      // Initiation interval in cycles
};

// Structure to store TaskNode with its loop context information
struct TaskNodeWithContext {
  TaskNode *task;
  ControlNode *control_node{nullptr}; // nullptr if not inside a loop
  int64_t tripcount{1}; // tripcount multiplier for weighted latency
};

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

  // Check if either task contains loop_break
  // Tasks with loop_break should not share register regions with other tasks
  if (a->IsTask() && b->IsTask()) {
    const TaskNode *task_a = static_cast<const TaskNode *>(a);
    const TaskNode *task_b = static_cast<const TaskNode *>(b);
    if (task_a->ContainsLoopBreak() || task_b->ContainsLoopBreak()) {
      return false;
    }
  }

  auto reg_regions_a = CollectRegisterRegions(a);
  auto reg_regions_b = CollectRegisterRegions(b);

  // For each pair of register regions, check if they refer to the same buffer
  // If buffer is the same, consider it as the same region (simplified)
  for (const auto &region_a : reg_regions_a) {
    for (const auto &region_b : reg_regions_b) {
      // Check if same buffer
      if (region_a->buffer.same_as(region_b->buffer)) {
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
    IRStructure *node, std::vector<TaskNodeWithContext> &all_tasks,
    ControlNode *current_control_node = nullptr);

// Helper function to print BufferRegion details
inline void PrintBufferRegion(const BufferRegion &region,
                              const std::string &indent) {
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

// Helper function to print all stmts in IRStructure
inline void PrintAllStmts(const IRStructure *node, int indent = 0) {
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
              << " cycles, warpgroup_id: " << task->GetWarpgroupId();
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
  } else if (node->IsWrapper()) {
    const WrapperNode *wrapper = static_cast<const WrapperNode *>(node);
    LOG(INFO) << indent_str << "WrapperNode:";
    LOG(INFO) << indent_str << "  Wrapper: " << wrapper->wrapper;
    // Recursively print child statements
    if (wrapper->child) {
      LOG(INFO) << indent_str << "  Wrapper body:";
      PrintAllStmts(wrapper->child.get(), indent + 2);
    }
  } else if (node->IsScheduleUnit()) {
    const ScheduleUnit *promote = static_cast<const ScheduleUnit *>(node);
    LOG(INFO) << indent_str << "ScheduleUnit:";
    LOG(INFO) << indent_str << "  Promote: " << promote->stage;
    for (unsigned idx = 0; idx != promote->before.size(); ++idx) {
      for (auto &stmt : promote->before[idx]) {
        LOG(INFO) << indent_str << "  Before " << idx << " : " << stmt;
      }
    }
    for (unsigned idx = 0; idx != promote->after.size(); ++idx) {
      for (auto &stmt : promote->after[idx]) {
        LOG(INFO) << indent_str << "  After " << idx << " : " << stmt;
      }
    }
    // Recursively print child statements
    if (promote->child) {
      LOG(INFO) << indent_str << "  Promote body:";
      PrintAllStmts(promote->child.get(), indent + 2);
    }
  }
}

inline Stmt GetLetDecl(const LetStmtNode *let_node) {
  if (let_node == nullptr) {
    return Stmt();
  }
  return LetStmt(let_node->var, let_node->value, Evaluate(0));
}

inline Stmt GetAttrDecl(const AttrStmtNode *attr_node) {
  if (attr_node == nullptr) {
    return Stmt();
  }
  return AttrStmt(attr_node->node, attr_node->attr_key, attr_node->value,
                  Evaluate(0));
}

// Original helper function to print IRStructure (kept for backward
// compatibility)
inline void PrintIRStructure(const IRStructure *node, int indent = 0) {
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
    for (auto &var : task->GetReadVars()) {
      LOG(INFO) << indent_str << "  Read Var: " << var;
    }
    for (auto &var : task->GetWriteVars()) {
      LOG(INFO) << indent_str << "  Write Var: " << var;
    }
    LOG(INFO) << indent_str << "  uses_cuda_core: " << task->UsesCUDACore();
    LOG(INFO) << indent_str << "  uses_tma_core: " << task->UsesTMACore();
    LOG(INFO) << indent_str << "  uses_tensor_core: " << task->UsesTensorCore();
    LOG(INFO) << indent_str << "  latency: " << task->GetLatency() << " cycles";
    LOG(INFO) << indent_str << "  II: " << task->GetII() << " cycles";
    LOG(INFO) << indent_str << "  warpgroup_id: " << task->GetWarpgroupId();
  } else if (node->IsControl()) {
    const ControlNode *control = static_cast<const ControlNode *>(node);
    LOG(INFO) << indent_str << "ControlNode (For loop):";
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
  } else if (node->IsWrapper()) {
    const WrapperNode *wrapper = static_cast<const WrapperNode *>(node);
    LOG(INFO) << indent_str << "WrapperNode:";
    LOG(INFO) << indent_str << "  Wrapper: " << wrapper->wrapper;
    LOG(INFO) << indent_str << "  Task:";
    PrintIRStructure(wrapper->task.get(), indent + 4);
    if (wrapper->child) {
      LOG(INFO) << indent_str << "  Child:";
      PrintIRStructure(wrapper->child.get(), indent + 2);
    }
  } else if (node->IsScheduleUnit()) {
    const ScheduleUnit *promote = static_cast<const ScheduleUnit *>(node);
    LOG(INFO) << indent_str << "ScheduleUnit:";
    LOG(INFO) << indent_str << "  Promote: " << promote->stage;
    for (unsigned idx = 0; idx != promote->before.size(); ++idx) {
      for (auto &stmt : promote->before[idx]) {
        LOG(INFO) << indent_str << "  Before " << idx << " : " << stmt;
      }
    }
    for (unsigned idx = 0; idx != promote->after.size(); ++idx) {
      for (auto &stmt : promote->after[idx]) {
        LOG(INFO) << indent_str << "  After " << idx << " : " << stmt;
      }
    }
    if (promote->child) {
      LOG(INFO) << indent_str << "  Promote body:";
      PrintAllStmts(promote->child.get(), indent + 2);
    }
  }
}

} // namespace tl
} // namespace tvm
