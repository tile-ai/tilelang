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
 * \file auto_schedule.cc
 * \brief AutoSchedule pass for TileLang
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/runtime/logging.h>

#include <utility>
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <queue>
#include <unordered_map>

#include "./common/attr.h"
#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;
using ffi::GetRef;

// Forward declaration
class IRStructure;

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
  } else if (scope == "local" || scope == "local.var") {
    return MemoryType::kRegister;
  }
  return MemoryType::kUnknown;
}

// Base class for IR structure nodes
class IRStructure {
public:
  enum class Kind {
    kTask,
    kControl,
    kSequence
  };

  virtual ~IRStructure() = default;
  virtual Kind GetKind() const = 0;

  // Helper methods for safe casting
  bool IsTask() const { return GetKind() == Kind::kTask; }
  bool IsControl() const { return GetKind() == Kind::kControl; }
  bool IsSequence() const { return GetKind() == Kind::kSequence; }
};


// Task node: contains a vector of statements
class TaskNode : public IRStructure {
public:
  std::vector<Stmt> stmts;

  // Resource usage flags
  bool uses_cuda_core{false};
  bool uses_tma_core{false};
  bool uses_tensor_core{false};

  // Memory access regions (collected during analysis)
  std::vector<BufferRegion> read_regions;
  std::vector<BufferRegion> write_regions;

  // Latency estimation
  int64_t latency{0};      // Estimated latency in cycles
  int64_t ii{0};           // Initiation interval in cycles

  Kind GetKind() const override { return Kind::kTask; }
};

// Control node: contains a For operation and a child IRStructure
class ControlNode : public IRStructure {
public:
  For control;  // The For operation
  std::unique_ptr<IRStructure> child;

  Kind GetKind() const override { return Kind::kControl; }
};

// Sequence node: contains a vector of child IRStructures
class SequenceNode : public IRStructure {
public:
  std::vector<std::unique_ptr<IRStructure>> children;

  Kind GetKind() const override { return Kind::kSequence; }
};

// ScheduleUnit: a group of consecutive TaskNodes that can be scheduled together
struct ScheduleUnit {
  std::vector<TaskNode*> tasks;  // consecutive TaskNodes
  SequenceNode* parent_seq{nullptr};  // Parent SequenceNode containing these tasks
  size_t start_idx{0};  // Starting index in parent_seq->children
  bool inside_control_node{false};  // Whether this unit is inside a ControlNode (For loop)
  // Additional metadata could be added here, e.g., resource usage summary
};

// MemoryAccessDetector: detect read/write regions in statements
// Adapted from BlockReadWriteDetector in TVM
class MemoryAccessDetector : public StmtExprVisitor {
public:
  MemoryAccessDetector() = default;

  // Analyze a statement and collect read/write regions
  void Analyze(const Stmt& stmt) {
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
  std::unordered_map<const VarNode*, arith::IntSet> dom_map_;
  /*! \brief Extra iteration range hint for free vars */
  std::unordered_map<const VarNode*, arith::IntSet> hint_map_;
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
  std::unordered_map<const VarNode*, PrimExpr> let_bindings_;

  /*!
   * \brief Update read/write buffers and regions with provided buffer and region
   */
  void Update(std::vector<Buffer>* buffers,
              std::vector<std::vector<arith::IntSet>>* regions,
              Buffer buffer,
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
   * \param arg The argument which could be BufferRegion, BufferLoad, or tl.tileop.region call
   * \param is_read Whether this is a read (true) or write (false) access
   */
  void ProcessBufferRegion(const PrimExpr& arg, bool is_read) {
    // Check if it's a BufferRegion
    if (const auto* buffer_region = arg.as<BufferRegionNode>()) {
      Buffer buffer = buffer_region->buffer;
      const Region& region = buffer_region->region;
      std::vector<arith::IntSet> int_sets;
      int_sets.reserve(region.size());
      for (const auto& range : region) {
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
    if (const auto* buffer_load = arg.as<BufferLoadNode>()) {
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

    // Check if it's a tl.tileop.region call (should already be handled by VisitExpr_)
    // but we can still process it recursively
    if (const auto* call = arg.as<CallNode>()) {
      static const auto region_op = Op::Get("tl.tileop.region");
      if (call->op.same_as(region_op)) {
        // Recursively visit this call to handle it
        VisitExpr_(call);
        return;
      }
    }

    // If we reach here, the argument type is not supported
    LOG(WARNING) << "Unsupported argument type in tl.tileop.reduce: " << arg->GetTypeKey();
  }

  /*! \brief Helper function to collect access regions. */
  std::vector<BufferRegion> CollectRegions(
      const std::vector<Buffer>& buffers,
      const std::vector<std::vector<tvm::arith::IntSet>>& regions) const {
    std::vector<BufferRegion> result;
    result.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
      const Buffer& buffer = buffers[i];
      const std::vector<arith::IntSet>& int_sets = regions[i];
      Region region;
      size_t ndim = buffer->shape.size();
      size_t region_ndim = int_sets.size();

      // Assert that region dimension equals buffer dimension
      ICHECK_EQ(region_ndim, ndim) << "Region dimension " << region_ndim
                                   << " must equal buffer dimension " << ndim;

      region.reserve(ndim);
      for (size_t j = 0; j < ndim; ++j) {
        const tvm::arith::IntSet& int_set = int_sets[j];
        region.push_back(int_set.CoverRange(Range::FromMinExtent(0, buffer->shape[j])));
      }

      result.push_back(BufferRegion(buffer, region));
    }
    return result;
  }

  /*! \brief Helper function to relax the buffer indices */
  arith::IntSet RelaxAccessIndex(const PrimExpr& index) {
    PrimExpr current = index;
    PrimExpr remapped = Substitute(current, let_bindings_);
    while (!remapped.same_as(current)) {
      current = remapped;
      remapped = Substitute(current, let_bindings_);
    }
    return arith::EvalSet(arith::IntSet::Vector(current), dom_map_);
  }

  void operator()(const Stmt& stmt) {
    StmtExprVisitor::operator()(stmt);
  }

  void VisitStmt_(const ForNode* op) override {
    Range range = Range::FromMinExtent(op->min, op->extent);
    dom_map_[op->loop_var.get()] = arith::IntSet::FromRange(range);
    StmtVisitor::VisitStmt_(op);
    dom_map_.erase(op->loop_var.get());
  }

  void VisitStmt_(const IfThenElseNode* op) override {
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

  void VisitStmt_(const LetStmtNode* op) override {
    let_bindings_[op->var.get()] = op->value;
    StmtVisitor::VisitStmt_(op);
    let_bindings_.erase(op->var.get());
  }

  void VisitExpr_(const BufferLoadNode* op) override {
    std::vector<arith::IntSet> relaxed_region;
    size_t num_indices = op->indices.size();
    size_t buffer_ndim = op->buffer->shape.size();

    // Assert that indices count equals buffer dimension
    ICHECK_EQ(num_indices, buffer_ndim) << "BufferLoad indices count " << num_indices
                                        << " must equal buffer dimension " << buffer_ndim;

    for (PrimExpr index : op->indices) {
      relaxed_region.push_back(RelaxAccessIndex(index));
    }
    Update(&read_buffers_, &read_regions_, op->buffer, relaxed_region);
    ExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) override {
    std::vector<arith::IntSet> relaxed_region;
    size_t num_indices = op->indices.size();
    size_t buffer_ndim = op->buffer->shape.size();

    // Assert that indices count equals buffer dimension
    ICHECK_EQ(num_indices, buffer_ndim) << "BufferStore indices count " << num_indices
                                        << " must equal buffer dimension " << buffer_ndim;

    for (PrimExpr index : op->indices) {
      relaxed_region.push_back(RelaxAccessIndex(index));
    }
    Update(&write_buffers_, &write_regions_, op->buffer, relaxed_region);
    StmtVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode* op) override {
    static const auto region_op = Op::Get("tl.tileop.region");
    static const auto reduce_op = Op::Get("tl.tileop.reduce");

    // Check for tl.tileop.region call
    if (op->op.same_as(region_op)) {
      // Handle tl.tileop.region call for memory access analysis
      // args[0] = buffer (BufferLoad), args[1] = access_type (1: read, 2: write, 3: read/write)
      // args[2..] = extents
      if (op->args.size() >= 2) {
        // Extract access type
        const auto* access_int = op->args[1].as<IntImmNode>();
        ICHECK(access_int);
        int access_type = access_int->value;

        // Extract buffer from BufferLoad
        if (const auto* buffer_load = op->args[0].as<BufferLoadNode>()) {
          Buffer buffer = buffer_load->buffer;
          std::vector<arith::IntSet> relaxed_region;

          // Assert that BufferLoad accesses a single element (no Ramp indices)
          for (size_t i = 0; i < buffer_load->indices.size(); ++i) {
            const PrimExpr& index = buffer_load->indices[i];
            // Check if index is a Ramp (vector access)
            if (index.as<RampNode>()) {
              LOG(FATAL) << "BufferLoad in tl.tileop.region should access a single element, "
                         << "but found Ramp index at dimension " << i;
            }
          }

          // Use provided extents if available, otherwise use buffer load indices
          size_t num_indices = buffer_load->indices.size();
          size_t buffer_ndim = buffer->shape.size();

          // Assert that indices count equals buffer dimension
          ICHECK_EQ(num_indices, buffer_ndim) << "BufferLoad indices count " << num_indices
                                              << " must equal buffer dimension " << buffer_ndim;

          if (op->args.size() > 2) {
            // args[2..] are extents for the region
            // Number of extents provided
            size_t num_extents = op->args.size() - 2;

            // Assert that extents count equals indices count
            ICHECK_EQ(num_extents, num_indices) << "Extents count " << num_extents
                                                << " must equal indices count " << num_indices;

            relaxed_region.reserve(num_indices);
            for (size_t i = 0; i < num_indices; ++i) {
              PrimExpr min = buffer_load->indices[i];
              PrimExpr extent = op->args[2 + i];

              // Create IntSet for range [min, min + extent)
              relaxed_region.push_back(arith::IntSet::FromRange(
                  Range::FromMinExtent(min, extent)));
            }
          } else {
            // No extents provided: each dimension is a single point at the index
            for (PrimExpr index : buffer_load->indices) {
              // Create IntSet for single point
              relaxed_region.push_back(RelaxAccessIndex(index));
            }
          }

          // Add to appropriate list based on access type
          if (access_type == 1 || access_type == 3) {  // read or read/write
            Update(&read_buffers_, &read_regions_, buffer, relaxed_region);
          }
          if (access_type == 2 || access_type == 3) {  // write or read/write
            Update(&write_buffers_, &write_regions_, buffer, relaxed_region);
          }
        } else {
          LOG(FATAL) << "First argument of tl.tileop.region should be a BufferLoad";
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
        ProcessBufferRegion(op->args[0], true);  // is_read = true
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
  void VisitStmt_(const BlockRealizeNode* op) override {
    // Don't visit child blocks recursively
  }
};

// Helper function to compare if two regions are equal
bool RegionsEqual(const Region& a, const Region& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!tir::is_one(a[i]->min - b[i]->min) || !tir::is_one(a[i]->extent - b[i]->extent)) {
      return false;
    }
  }
  return true;
}

// Latency estimator for H100 GPU
class LatencyEstimator {
public:
  // H100 latency parameters (in cycles)
  struct H100Params {
    // Base latencies
    int64_t global_memory_read = 400;   // Global memory read latency
    int64_t global_memory_write = 200;  // Global memory write latency (usually lower)
    int64_t shared_memory_read = 20;    // Shared memory read latency
    int64_t shared_memory_write = 20;   // Shared memory write latency
    int64_t register_access = 1;        // Register access latency
    int64_t cuda_core_operation = 4;    // Basic CUDA core operation (add, mul, etc.)
    int64_t tensor_core_operation = 64; // Tensor core operation (matrix multiply)
    int64_t tma_operation = 100;        // TMA operation latency

    // Bandwidth parameters (bytes per cycle)
    // H100: ~2TB/s global memory, 1.8GHz clock → ~1111 bytes/cycle
    // H100: ~19TB/s shared memory → ~10556 bytes/cycle
    int64_t global_memory_bandwidth = 1111;   // bytes per cycle
    int64_t shared_memory_bandwidth = 10556;  // bytes per cycle

    // Pipeline initiation capabilities
    int64_t max_memory_ops_per_cycle = 1;     // Max memory ops that can start per cycle
  };

  LatencyEstimator() = default;

  // Estimate latency for a TaskNode
  void Estimate(TaskNode* task) {
    int64_t total_latency = 0;
    int64_t memory_latency = 0;
    int64_t compute_latency = 0;

    // Count memory operations and track bytes by memory type
    int num_memory_ops = 0;
    int64_t global_memory_bytes = 0;
    int64_t shared_memory_bytes = 0;
    int64_t register_bytes = 0;

    // Estimate latency from memory accesses and track bandwidth usage
    for (const auto& region : task->read_regions) {
      int64_t region_latency = EstimateMemoryAccessLatency(region, true); // read
      memory_latency += region_latency;
      num_memory_ops++;

      // Track bandwidth usage by memory type
      const Buffer& buffer = region->buffer;
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

    for (const auto& region : task->write_regions) {
      int64_t region_latency = EstimateMemoryAccessLatency(region, false); // write
      memory_latency += region_latency;
      num_memory_ops++;

      // Track bandwidth usage by memory type
      const Buffer& buffer = region->buffer;
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
    if (task->uses_cuda_core) {
      // Simple heuristic: assume some number of CUDA operations
      // For now, assume 1 operation per statement as a rough estimate
      compute_latency = params_.cuda_core_operation * std::max(1, static_cast<int>(task->stmts.size()));
    }

    if (task->uses_tensor_core) {
      compute_latency = std::max(compute_latency, params_.tensor_core_operation);
    }

    if (task->uses_tma_core) {
      compute_latency = std::max(compute_latency, params_.tma_operation);
    }

    // Total latency is sum of memory and compute (assuming sequential for now)
    total_latency = memory_latency + compute_latency;

    // Calculate initiation interval (II)
    int64_t ii = 1;  // Default minimum II

    if (task->uses_tma_core) {
      // TMA operations (async memory copies): instruction latency can be hidden
      // II is determined by bandwidth constraints only
      if (global_memory_bytes > 0) {
        int64_t bandwidth_ii = (global_memory_bytes + params_.global_memory_bandwidth - 1) / params_.global_memory_bandwidth;
        ii = std::max(ii, bandwidth_ii);
      }
      if (shared_memory_bytes > 0) {
        int64_t bandwidth_ii = (shared_memory_bytes + params_.shared_memory_bandwidth - 1) / params_.shared_memory_bandwidth;
        ii = std::max(ii, bandwidth_ii);
      }
    } else {
      // Regular operations
      // According to requirements:
      // 1. If there's only one operation and it's a memory access, II = memory latency
      // 2. Otherwise, II = total latency
      ii = total_latency;

      if (num_memory_ops == 1 && task->stmts.size() == 1) {
        // Single operation that is a memory access
        // Check if this is likely a memory operation (has read/write regions)
        if (!task->read_regions.empty() || !task->write_regions.empty()) {
          ii = memory_latency;
        }
      }

      // Additional II constraints from bandwidth limitations
      // II must be at least the time needed to transfer data based on bandwidth
      if (global_memory_bytes > 0) {
        int64_t bandwidth_ii = (global_memory_bytes + params_.global_memory_bandwidth - 1) / params_.global_memory_bandwidth;
        ii = std::max(ii, bandwidth_ii);
      }

      if (shared_memory_bytes > 0) {
        int64_t bandwidth_ii = (shared_memory_bytes + params_.shared_memory_bandwidth - 1) / params_.shared_memory_bandwidth;
        ii = std::max(ii, bandwidth_ii);
      }
    }

    // II must be at least 1 cycle
    if (ii < 1) ii = 1;

    // Store results in task node
    task->latency = total_latency;
    task->ii = ii;
  }

private:
  H100Params params_;

  // Helper function to calculate total bytes accessed in a region
  int64_t CalculateAccessBytes(const BufferRegion& region) {
    const Buffer& buffer = region->buffer;
    const Region& ranges = region->region;

    // Calculate total number of elements
    int64_t total_elements = 1;
    for (const auto& range : ranges) {
      // Try to get constant extent if possible
      if (const auto* extent_int = range->extent.as<IntImmNode>()) {
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
  int64_t EstimateMemoryAccessLatency(const BufferRegion& region, bool is_read) {
    const Buffer& buffer = region->buffer;
    String scope = buffer.scope();
    MemoryType mem_type = GetMemoryTypeFromScope(scope);

    int64_t access_bytes = CalculateAccessBytes(region);

    switch (mem_type) {
      case MemoryType::kGlobal:
        // Global memory latency depends on data size
        // Base latency + bandwidth-limited component
        // Latency = base_latency + bytes / bytes_per_cycle
        // Subtract cache line size (32 bytes) since first cache line has base latency
        if (is_read) {
          // Base read latency + bandwidth component
          return params_.global_memory_read + std::max(0L, (access_bytes - 32) / params_.global_memory_bandwidth);
        } else {
          // Write latency usually lower
          return params_.global_memory_write + std::max(0L, (access_bytes - 32) / params_.global_memory_bandwidth);
        }
      case MemoryType::kShared:
        // Shared memory has high bandwidth, less sensitive to size
        // Subtract typical burst size (128 bytes) for base latency
        if (is_read) {
          return params_.shared_memory_read + std::max(0L, (access_bytes - 128) / params_.shared_memory_bandwidth);
        } else {
          return params_.shared_memory_write + std::max(0L, (access_bytes - 128) / params_.shared_memory_bandwidth);
        }
      case MemoryType::kRegister:
        // Register access latency is constant and very small
        return params_.register_access;
      default:
        // Unknown memory type, use global memory as conservative estimate
        if (is_read) {
          return params_.global_memory_read + std::max(0L, (access_bytes - 32) / params_.global_memory_bandwidth);
        } else {
          return params_.global_memory_write + std::max(0L, (access_bytes - 32) / params_.global_memory_bandwidth);
        }
    }
  }
};

// Builder that collects ScheduleUnits from IRStructure
class ScheduleUnitBuilder {
public:
  std::vector<ScheduleUnit> Build(IRStructure* root) {
    units_.clear();
    current_unit_ = nullptr;
    current_seq_ = nullptr;
    current_child_idx_ = 0;
    control_nesting_depth_ = 0;
    Collect(root);
    // Finalize the last unit if any
    FinalizeCurrentUnit();

    // Apply topological sort to each unit and update IRStructure
    for (auto& unit : units_) {
      ApplyTopologicalSort(unit);
    }

    return std::move(units_);
  }

private:
  std::vector<ScheduleUnit> units_;
  ScheduleUnit* current_unit_{nullptr};
  SequenceNode* current_seq_{nullptr};
  size_t current_child_idx_{0};
  int control_nesting_depth_{0};

  // Check if two regions refer to the same buffer
  bool SameBuffer(const BufferRegion& a, const BufferRegion& b) const {
    return a->buffer.same_as(b->buffer);
  }

  // Check if two TaskNodes have data dependency (excluding read-after-read)
  bool HasDependency(const TaskNode* a, const TaskNode* b) const {
    // Check all combinations of accesses
    // a writes, b reads (RAW)
    // a reads, b writes (WAR)
    // a writes, b writes (WAW)
    // a reads, b reads (RAR) - no dependency

    // For simplicity, we check if they access the same buffer
    // and at least one of them writes to that buffer
    for (const auto& write_region_a : a->write_regions) {
      for (const auto& read_region_b : b->read_regions) {
        if (SameBuffer(write_region_a, read_region_b)) return true;
      }
      for (const auto& write_region_b : b->write_regions) {
        if (SameBuffer(write_region_a, write_region_b)) return true;
      }
    }
    for (const auto& read_region_a : a->read_regions) {
      for (const auto& write_region_b : b->write_regions) {
        if (SameBuffer(read_region_a, write_region_b)) return true;
      }
    }
    return false;
  }

  // Perform topological sort on tasks based on data dependencies
  std::vector<TaskNode*> TopologicalSort(const std::vector<TaskNode*>& tasks) const {
    size_t n = tasks.size();
    if (n <= 1) return tasks;

    // Build adjacency list and indegree array
    std::vector<std::vector<size_t>> adj(n);
    std::vector<int> indegree(n, 0);

    // For each pair (i, j) where i < j, check dependency
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        bool dep_ij = HasDependency(tasks[i], tasks[j]);
        bool dep_ji = HasDependency(tasks[j], tasks[i]);

        if (dep_ij && !dep_ji) {
          // i -> j dependency (i must come before j)
          adj[i].push_back(j);
          indegree[j]++;
        } else if (!dep_ij && dep_ji) {
          // j -> i dependency (j must come before i)
          adj[j].push_back(i);
          indegree[i]++;
        } else if (dep_ij && dep_ji) {
          // Mutual dependency (e.g., WAW on same buffer)
          // Choose direction based on original order (i before j)
          adj[i].push_back(j);
          indegree[j]++;
          LOG(INFO) << "Mutual dependency between tasks " << i << " and " << j
                    << ", using original order (i before j)";
        }
        // else: no dependency
      }
    }

    // Kahn's algorithm for topological sort (modified: always pop largest index)
    std::vector<TaskNode*> result;
    result.reserve(n);
    std::vector<size_t> zero_indegree_nodes;

    // Add nodes with zero indegree
    for (size_t i = 0; i < n; ++i) {
      if (indegree[i] == 0) {
        zero_indegree_nodes.push_back(i);
      }
    }

    while (!zero_indegree_nodes.empty()) {
      // Find the node with largest index in zero_indegree_nodes
      auto max_it = std::max_element(zero_indegree_nodes.begin(), zero_indegree_nodes.end());
      size_t u = *max_it;
      // Remove it from zero_indegree_nodes (swap with last element and pop)
      *max_it = zero_indegree_nodes.back();
      zero_indegree_nodes.pop_back();

      result.push_back(tasks[u]);

      for (size_t v : adj[u]) {
        indegree[v]--;
        if (indegree[v] == 0) {
          zero_indegree_nodes.push_back(v);
        }
      }
    }

    // If not all nodes are included, there's a cycle
    // For now, return original order if cycle detected
    if (result.size() != n) {
      LOG(WARNING) << "Cycle detected in data dependencies, using original order";
      return tasks;
    }

    return result;
  }

  // Apply topological sort to a ScheduleUnit and update parent SequenceNode
  void ApplyTopologicalSort(ScheduleUnit& unit) {
    if (unit.tasks.size() <= 1) {
      // No reordering needed for single task
      return;
    }

    if (!unit.parent_seq) {
      // No parent sequence to update (should not happen for tasks from SequenceNode)
      LOG(WARNING) << "ScheduleUnit has no parent SequenceNode, skipping reorder";
      return;
    }

    // Perform topological sort
    std::vector<TaskNode*> sorted_tasks = TopologicalSort(unit.tasks);

    // Check if order actually changed
    bool order_changed = false;
    for (size_t i = 0; i < sorted_tasks.size(); ++i) {
      if (sorted_tasks[i] != unit.tasks[i]) {
        order_changed = true;
        break;
      }
    }

    if (!order_changed) {
      // Order unchanged, nothing to do
      return;
    }

    // Update unit.tasks to sorted order
    unit.tasks = std::move(sorted_tasks);

    // Reorder children in parent SequenceNode
    size_t start_idx = unit.start_idx;
    size_t num_tasks = unit.tasks.size();

    // Extract the relevant children from parent sequence
    std::vector<std::unique_ptr<IRStructure>> task_children;
    task_children.reserve(num_tasks);

    // First, move the task children out of parent sequence
    for (size_t i = 0; i < num_tasks; ++i) {
      task_children.push_back(std::move(unit.parent_seq->children[start_idx + i]));
    }

    // Create a mapping from TaskNode pointer to its index in task_children
    std::unordered_map<TaskNode*, size_t> task_to_old_index;
    for (size_t i = 0; i < num_tasks; ++i) {
      // Each child should be a TaskNode
      IRStructure* child_ptr = task_children[i].get();
      if (child_ptr->IsTask()) {
        TaskNode* task = static_cast<TaskNode*>(child_ptr);
        task_to_old_index[task] = i;
      } else {
        LOG(FATAL) << "Expected TaskNode child in ScheduleUnit";
      }
    }

    // Reorder task_children according to sorted_tasks
    std::vector<std::unique_ptr<IRStructure>> reordered_children;
    reordered_children.reserve(num_tasks);

    for (TaskNode* task : unit.tasks) {
      auto it = task_to_old_index.find(task);
      if (it == task_to_old_index.end()) {
        LOG(FATAL) << "TaskNode not found in extracted children";
      }
      size_t old_idx = it->second;
      reordered_children.push_back(std::move(task_children[old_idx]));
    }

    // Move reordered children back to parent sequence
    for (size_t i = 0; i < num_tasks; ++i) {
      unit.parent_seq->children[start_idx + i] = std::move(reordered_children[i]);
    }

    LOG(INFO) << "Reordered " << num_tasks << " TaskNodes in SequenceNode";
  }

  void StartNewUnit() {
    FinalizeCurrentUnit();
    units_.emplace_back();
    current_unit_ = &units_.back();
    current_unit_->inside_control_node = (control_nesting_depth_ > 0);
  }

  void FinalizeCurrentUnit() {
    if (current_unit_ && current_unit_->tasks.empty()) {
      // Remove empty unit
      units_.pop_back();
    }
    current_unit_ = nullptr;
  }

  void Collect(IRStructure* node) {
    if (!node) return;

    if (node->IsTask()) {
      TaskNode* task = static_cast<TaskNode*>(node);
      if (!current_unit_) {
        StartNewUnit();
        // Set parent sequence and start index for this unit
        current_unit_->parent_seq = current_seq_;
        current_unit_->start_idx = current_child_idx_;
      }
      current_unit_->tasks.push_back(task);
    } else if (node->IsSequence()) {
      SequenceNode* seq = static_cast<SequenceNode*>(node);
      // Save previous context
      SequenceNode* prev_seq = current_seq_;
      size_t prev_child_idx = current_child_idx_;

      // Set new context for this sequence
      current_seq_ = seq;
      current_child_idx_ = 0;

      for (auto& child : seq->children) {
        Collect(child.get());
        current_child_idx_++;
        // If child is a ControlNode, it will have finalized current_unit_
        // and set it to nullptr. So we need to ensure that after a ControlNode,
        // we don't continue adding to the previous unit.
        if (child->IsControl()) {
          // Already finalized by Collect
        }
      }

      // Restore previous context
      current_seq_ = prev_seq;
      current_child_idx_ = prev_child_idx;
    } else if (node->IsControl()) {
      ControlNode* ctrl = static_cast<ControlNode*>(node);
      // Finalize any current unit before entering control node
      FinalizeCurrentUnit();
      // Increase control nesting depth
      control_nesting_depth_++;
      // Process the body of the control node (creates new units internally)
      Collect(ctrl->child.get());
      // Decrease control nesting depth
      control_nesting_depth_--;
      // After control node, current_unit_ should remain nullptr
      FinalizeCurrentUnit();
    }
  }
};

// Helper function to print BufferRegion details
void PrintBufferRegion(const BufferRegion& region, const std::string& indent) {
  const Buffer& buffer = region->buffer;
  const Region& ranges = region->region;

  std::string buffer_name = buffer->name;
  if (buffer_name.empty()) buffer_name = "unnamed_buffer";

  LOG(INFO) << indent << "Buffer: " << buffer_name;

  // Build shape string
  std::ostringstream shape_ss;
  shape_ss << "[";
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    if (i > 0) shape_ss << ", ";
    shape_ss << buffer->shape[i];
  }
  shape_ss << "]";
  LOG(INFO) << indent << "  Shape: " << shape_ss.str();

  LOG(INFO) << indent << "  Region:";
  for (size_t i = 0; i < ranges.size(); ++i) {
    const Range& range = ranges[i];
    std::ostringstream range_ss;
    range_ss << "[" << range->min << ", " << range->min + range->extent
             << ") (extent=" << range->extent << ")";
    LOG(INFO) << indent << "    dim " << i << ": " << range_ss.str();
  }
}

// Helper function to print ScheduleUnits
void PrintScheduleUnits(const std::vector<ScheduleUnit>& units) {
  LOG(INFO) << "ScheduleUnits (" << units.size() << " units):";
  for (size_t i = 0; i < units.size(); i++) {
    const auto& unit = units[i];
    LOG(INFO) << "  Unit " << i << " contains " << unit.tasks.size() << " TaskNodes:";
    LOG(INFO) << "    Parent Sequence: " << (unit.parent_seq ? "present" : "none")
              << ", Start index: " << unit.start_idx
              << ", Inside ControlNode: " << (unit.inside_control_node ? "yes" : "no");
    for (size_t j = 0; j < unit.tasks.size(); j++) {
      const TaskNode* task = unit.tasks[j];
      LOG(INFO) << "    Task " << j << ": uses_cuda_core=" << task->uses_cuda_core
                << ", uses_tma_core=" << task->uses_tma_core
                << ", uses_tensor_core=" << task->uses_tensor_core
                << ", read_regions=" << task->read_regions.size()
                << ", write_regions=" << task->write_regions.size()
                << ", latency=" << task->latency << " cycles"
                << ", II=" << task->ii << " cycles";

      // Print statements in this task
      if (!task->stmts.empty()) {
        LOG(INFO) << "      Statements (" << task->stmts.size() << "):";
        for (size_t k = 0; k < task->stmts.size(); ++k) {
          LOG(INFO) << "        Statement " << k << ":";
          LOG(INFO) << "          " << task->stmts[k];
        }
      }

      // Print read regions if any
      if (!task->read_regions.empty()) {
        LOG(INFO) << "      Read regions:";
        for (size_t k = 0; k < task->read_regions.size(); ++k) {
          LOG(INFO) << "      Region " << k << ":";
          PrintBufferRegion(task->read_regions[k], "        ");
        }
      }

      // Print write regions if any
      if (!task->write_regions.empty()) {
        LOG(INFO) << "      Write regions:";
        for (size_t k = 0; k < task->write_regions.size(); ++k) {
          LOG(INFO) << "      Region " << k << ":";
          PrintBufferRegion(task->write_regions[k], "        ");
        }
      }
    }
  }
}

// Helper function to print all stmts in IRStructure
void PrintAllStmts(const IRStructure* node, int indent = 0) {
  if (!node) return;

  std::string indent_str(indent * 2, ' ');

  if (node->IsTask()) {
    const TaskNode* task = static_cast<const TaskNode*>(node);
    LOG(INFO) << indent_str << "TaskNode with " << task->stmts.size() << " statements:";
    for (size_t i = 0; i < task->stmts.size(); i++) {
      LOG(INFO) << indent_str << "  Statement " << i << ":";
      LOG(INFO) << indent_str + "    " << task->stmts[i];
    }
    LOG(INFO) << indent_str << "  Resource usage: CUDA=" << task->uses_cuda_core
              << ", TMA=" << task->uses_tma_core
              << ", Tensor=" << task->uses_tensor_core;
    LOG(INFO) << indent_str << "  Latency: " << task->latency << " cycles, II: " << task->ii << " cycles";
  } else if (node->IsControl()) {
    const ControlNode* control = static_cast<const ControlNode*>(node);
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
    const SequenceNode* seq = static_cast<const SequenceNode*>(node);
    LOG(INFO) << indent_str << "SequenceNode with " << seq->children.size() << " children:";
    for (size_t i = 0; i < seq->children.size(); i++) {
      LOG(INFO) << indent_str << "  Child " << i << ":";
      PrintAllStmts(seq->children[i].get(), indent + 2);
    }
  }
}

// Original helper function to print IRStructure (kept for backward compatibility)
void PrintIRStructure(const IRStructure* node, int indent = 0) {
  if (!node) return;

  std::string indent_str(indent * 2, ' ');

  if (node->IsTask()) {
    const TaskNode* task = static_cast<const TaskNode*>(node);
    LOG(INFO) << indent_str << "TaskNode:";
    LOG(INFO) << indent_str << "  stmts: " << task->stmts.size() << " statements";
    LOG(INFO) << indent_str << "  uses_cuda_core: " << task->uses_cuda_core;
    LOG(INFO) << indent_str << "  uses_tma_core: " << task->uses_tma_core;
    LOG(INFO) << indent_str << "  uses_tensor_core: " << task->uses_tensor_core;
    LOG(INFO) << indent_str << "  latency: " << task->latency << " cycles";
    LOG(INFO) << indent_str << "  II: " << task->ii << " cycles";
  } else if (node->IsControl()) {
    const ControlNode* control = static_cast<const ControlNode*>(node);
    LOG(INFO) << indent_str << "ControlNode (For loop):";
    // Could print loop info if needed
    if (control->child) {
      LOG(INFO) << indent_str << "  Child:";
      PrintIRStructure(control->child.get(), indent + 2);
    }
  } else if (node->IsSequence()) {
    const SequenceNode* seq = static_cast<const SequenceNode*>(node);
    LOG(INFO) << indent_str << "SequenceNode: " << seq->children.size() << " children";
    for (size_t i = 0; i < seq->children.size(); i++) {
      LOG(INFO) << indent_str << "  Child " << i << ":";
      PrintIRStructure(seq->children[i].get(), indent + 2);
    }
  }
}


// Rewriter to convert IRStructure back to TIR statements
class IRStructureRewriter {
public:
  Stmt Rewrite(const IRStructure* node) {
    if (!node) {
      return Stmt();
    }

    if (node->IsTask()) {
      const TaskNode* task = static_cast<const TaskNode*>(node);
      if (task->stmts.empty()) {
        return Stmt();
      }
      // If there's only one statement, return it directly
      if (task->stmts.size() == 1) {
        return task->stmts[0];
      }
      // Otherwise, create a SeqStmt
      return SeqStmt(task->stmts);

    } else if (node->IsControl()) {
      const ControlNode* control = static_cast<const ControlNode*>(node);
      // Rebuild the For loop with rewritten body
      Stmt body = Rewrite(control->child.get());
      if (!body.defined()) {
        LOG(WARNING) << "ControlNode body is undefined";
        return control->control;
      }
      // Create a new For loop with the same parameters but updated body
      return For(control->control->loop_var,
                 control->control->min,
                 control->control->extent,
                 control->control->kind,
                 body,
                 control->control->thread_binding,
                 control->control->annotations);

    } else if (node->IsSequence()) {
      const SequenceNode* seq = static_cast<const SequenceNode*>(node);
      std::vector<Stmt> stmts;
      stmts.reserve(seq->children.size());

      for (const auto& child : seq->children) {
        Stmt child_stmt = Rewrite(child.get());
        if (child_stmt.defined()) {
          stmts.push_back(child_stmt);
        }
      }

      if (stmts.empty()) {
        return Stmt();
      } else if (stmts.size() == 1) {
        return stmts[0];
      } else {
        return SeqStmt(stmts);
      }
    }

    LOG(FATAL) << "Unknown IRStructure kind";
    return Stmt();
  }
};

// Visitor to extract the body of tilelang_root block
class TilelangRootBodyExtractor : public StmtVisitor {
public:
  Stmt body;

  void VisitStmt_(const BlockNode* op) override {
    if (op->name_hint == "tilelang_root") {
      LOG(INFO) << "Found tilelang_root block, extracting its body";
      body = op->body;
      return; // Don't visit children
    }
    StmtVisitor::VisitStmt_(op);
  }
};

// Mutator to replace the body of tilelang_root block
class TilelangRootBodyReplacer : public StmtMutator {
public:
  explicit TilelangRootBodyReplacer(Stmt new_body) : new_body_(new_body) {}

  Stmt VisitStmt_(const BlockNode* op) override {
    auto block = GetRef<Block>(op);
    if (op->name_hint == "tilelang_root") {
      LOG(INFO) << "Replacing body of tilelang_root block";
      // Keep all block attributes but replace the body
      return Block(op->iter_vars, op->reads, op->writes, op->name_hint, new_body_,
                   op->init, op->alloc_buffers, op->match_buffers, op->annotations);
    }
    return StmtMutator::VisitStmt_(op);
  }

private:
  Stmt new_body_;
};

// Visitor to build IRStructure from TIR statements
class IRStructureBuilder : public StmtVisitor {
public:
  std::unique_ptr<IRStructure> Build(const Stmt& stmt) {
    VisitStmt(stmt);
    if (!root_) {
      LOG(WARNING) << "IRStructureBuilder: root_ is null after visiting statement. "
                   << "This may indicate an unhandled statement type.";
      // Return an empty TaskNode as fallback
      auto task_node = std::make_unique<TaskNode>();
      task_node->stmts.push_back(stmt);
      return task_node;
    }
    return std::move(root_);
  }

protected:

  void VisitStmt_(const SeqStmtNode* op) override {
    auto seq_node = std::make_unique<SequenceNode>();

    for (size_t i = 0; i < op->seq.size(); i++) {
      VisitStmt(op->seq[i]);
      if (root_) {
        seq_node->children.push_back(std::move(root_));
      }
    }
    root_ = std::move(seq_node);
  }

  void VisitStmt_(const ForNode* op) override {
    // Determine if this is a sequential or parallel for
    if (op->kind == ForKind::kSerial) {
      // Sequential For -> ControlNode
      auto control_node = std::make_unique<ControlNode>();
      control_node->control = GetRef<For>(op);

      // Process the loop body
      VisitStmt(op->body);
      if (root_) {
        control_node->child = std::move(root_);
      } else {
      }

      root_ = std::move(control_node);
    } else {
      // Parallel For -> TaskNode
      auto task_node = std::make_unique<TaskNode>();
      task_node->stmts.push_back(GetRef<Stmt>(op));

      // Analyze the loop body for resource usage
      AnalyzeResourceUsage(op->body, task_node.get());

      root_ = std::move(task_node);
    }
  }

  void VisitStmt_(const EvaluateNode* op) override {
    // Evaluate statement (usually a Call) -> TaskNode
    auto task_node = std::make_unique<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    // Analyze the expression for resource usage
    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());

    root_ = std::move(task_node);
  }

  void VisitStmt_(const IfThenElseNode* op) override {
    // If statement -> treat as TaskNode for now (could be refined later)
    auto task_node = std::make_unique<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    // Analyze both branches for resource usage
    AnalyzeResourceUsage(op->then_case, task_node.get());
    if (op->else_case) {
      AnalyzeResourceUsage(op->else_case.value(), task_node.get());
    }

    root_ = std::move(task_node);
  }

  void VisitStmt_(const LetStmtNode* op) override {
    // Let statement -> treat as TaskNode
    auto task_node = std::make_unique<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    AnalyzeResourceUsage(op->body, task_node.get());

    root_ = std::move(task_node);
  }

  void VisitStmt_(const WhileNode* op) override {
    LOG(INFO) << "VisitStmt_: WhileNode -> TaskNode";
    auto task_node = std::make_unique<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    // Analyze condition and body for resource usage
    AnalyzeResourceUsage(Evaluate(op->condition), task_node.get());
    AnalyzeResourceUsage(op->body, task_node.get());

    LOG(INFO) << "  TaskNode resource usage: CUDA=" << task_node->uses_cuda_core
              << ", TMA=" << task_node->uses_tma_core
              << ", Tensor=" << task_node->uses_tensor_core;

    root_ = std::move(task_node);
  }

  void VisitStmt_(const BlockNode* op) override {
    // All blocks are treated as TaskNode
    // Note: tilelang_root block should have been extracted by TilelangRootBodyExtractor
    // If we encounter it here, it means we're processing the entire function body
    // (not extracted), which should only happen when there's no tilelang_root block
    auto task_node = std::make_unique<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));
    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());
    root_ = std::move(task_node);
  }


private:
  std::unique_ptr<IRStructure> root_;

  void AnalyzeResourceUsage(const Stmt& stmt, TaskNode* task_node) {
    // Recursively analyze statements to determine resource usage
    struct ResourceAnalyzer : public StmtExprVisitor {
      TaskNode* task_node;
      bool found_tma{false};
      bool found_tensor{false};
      bool found_cuda{false};

      ResourceAnalyzer(TaskNode* node) : task_node(node) {}

      void VisitExpr_(const CallNode* op) override {
        // Check for specific TileLang operations
        static const auto copy_op = Op::Get("tl.tileop.copy");
        static const auto gemm_py_op = Op::Get("tl.tileop.gemm_py");
        static const auto gemm_op = Op::Get("tl.tileop.gemm");
        static const auto reduce_op = Op::Get("tl.tileop.reduce");
        static const auto fill_op = Op::Get("tl.tileop.fill");
        static const auto region_op = Op::Get("tl.tileop.region");

        // Try to get operation name for logging
        std::string op_name = "unknown";
        if (const auto* op_ptr = op->op.as<OpNode>()) {
          op_name = op_ptr->name;
        }

        // Check if this is a TMA copy operation
        if (op->op.same_as(copy_op)) {
          found_tma = true;
        } else if (op->op.same_as(gemm_py_op) || op->op.same_as(gemm_op)) {
          found_tensor = true;
        } else if (op->op.same_as(reduce_op) || op->op.same_as(fill_op)) {
          // Reduce and fill operations use CUDA core
          found_cuda = true;
        } else if (op->op.same_as(region_op)) {
          // Handle tl.tileop.region call for memory access analysis
          // args[0] = buffer (BufferLoad), args[1] = access_type (1: read, 2: write, 3: read/write)
          // args[2..] = extents
          if (op->args.size() >= 2) {
            // Extract access type
            if (const auto* access_int = op->args[1].as<IntImmNode>()) {
              int access_type = access_int->value;
              // For now, just mark as CUDA operation (memory access)
              found_cuda = true;
              // TODO: Extract buffer region and add to task_node->read_regions or write_regions
              // BufferLoad buffer_load = Downcast<BufferLoad>(op->args[0]);
              // Construct BufferRegion from buffer_load and extents
              // if (access_type == 1 || access_type == 3) task_node->read_regions.push_back(region);
              // if (access_type == 2 || access_type == 3) task_node->write_regions.push_back(region);
            }
          }
        } else {
          // Check for other known operations that use CUDA core
          // For now, assume any other call is a basic computation
          found_cuda = true;
        }

        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const AddNode* op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const SubNode* op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const MulNode* op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const DivNode* op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }

    };

    ResourceAnalyzer analyzer(task_node);
    analyzer(stmt);

    // Set task node flags based on what was found
    if (analyzer.found_tma) {
      task_node->uses_tma_core = true;
    }
    if (analyzer.found_tensor) {
      task_node->uses_tensor_core = true;
    }
    // If neither TMA nor Tensor core was used, and CUDA operations were found, set CUDA core flag
    if (!analyzer.found_tma && !analyzer.found_tensor && analyzer.found_cuda) {
      task_node->uses_cuda_core = true;
    }

    // Analyze memory access regions
    MemoryAccessDetector memory_detector;
    memory_detector.Analyze(stmt);
    std::vector<BufferRegion> read_regions = memory_detector.GetReadRegions();
    std::vector<BufferRegion> write_regions = memory_detector.GetWriteRegions();

    // Merge with existing regions (avoid duplicates)
    for (const auto& region : read_regions) {
      bool found = false;
      for (const auto& existing : task_node->read_regions) {
        if (existing->buffer.same_as(region->buffer) &&
            RegionsEqual(existing->region, region->region)) {
          found = true;
          break;
        }
      }
      if (!found) {
        task_node->read_regions.push_back(region);
      }
    }

    for (const auto& region : write_regions) {
      bool found = false;
      for (const auto& existing : task_node->write_regions) {
        if (existing->buffer.same_as(region->buffer) &&
            RegionsEqual(existing->region, region->region)) {
          found = true;
          break;
        }
      }
      if (!found) {
        task_node->write_regions.push_back(region);
      }
    }

    // Estimate latency and initiation interval for this task
    LatencyEstimator latency_estimator;
    latency_estimator.Estimate(task_node);
  }
};

// The main pass function
tvm::transform::Pass AutoSchedule() {
  using namespace tir::transform;
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    LOG(INFO) << "AutoSchedule pass started for function: " << func->GetAttr<String>(tvm::attr::kGlobalSymbol).value_or("unnamed");

    // Extract the body of tilelang_root block if it exists
    TilelangRootBodyExtractor extractor;
    extractor(func->body);
    Stmt body_to_schedule;
    bool has_tilelang_root = false;

    if (extractor.body.defined()) {
      LOG(INFO) << "Found tilelang_root block, scheduling its body only";
      body_to_schedule = extractor.body;
      has_tilelang_root = true;
    } else {
      LOG(INFO) << "No tilelang_root block found, scheduling entire function body";
      body_to_schedule = func->body;
    }

    // Build IRStructure from the body to schedule
    LOG(INFO) << "Building IRStructure from body...";
    IRStructureBuilder builder;
    auto ir_structure = builder.Build(body_to_schedule);

    // Print the built IRStructure with all statements
    if (ir_structure) {
      LOG(INFO) << "IRStructure built successfully:";

      // First print the summary view
      LOG(INFO) << "IRStructure summary:";
      PrintIRStructure(ir_structure.get());

      // Then print all statements
      LOG(INFO) << "=================== IRStructure Statements ===================";
      PrintAllStmts(ir_structure.get());
      LOG(INFO) << "=================== End IRStructure Statements ===================";

      // Build ScheduleUnits from IRStructure
      LOG(INFO) << "Building ScheduleUnits...";
      ScheduleUnitBuilder unit_builder;
      auto schedule_units = unit_builder.Build(ir_structure.get());
      PrintScheduleUnits(schedule_units);

      // Print the modified summary view
      LOG(INFO) << "IRStructure modified summary:";
      PrintIRStructure(ir_structure.get());

    } else {
      LOG(INFO) << "IRStructure is null (empty body?)";
    }

    // Rewrite body based on IRStructure
    Stmt new_body;
    if (ir_structure) {
      IRStructureRewriter rewriter;
      new_body = rewriter.Rewrite(ir_structure.get());
      LOG(INFO) << "Rewrote body based on IRStructure";
    } else {
      LOG(INFO) << "IRStructure is null, keeping original body";
      new_body = body_to_schedule;
    }

    // If we extracted from tilelang_root block, replace the body
    Stmt final_body;
    if (has_tilelang_root) {
      TilelangRootBodyReplacer replacer(new_body);
      final_body = replacer(func->body);
      LOG(INFO) << "Replaced tilelang_root block body";
    } else {
      final_body = new_body;
    }

    // Create a new PrimFunc with the updated body
    auto new_func = PrimFunc(func->params, final_body, func->ret_type, func->buffer_map, func->attrs);
    LOG(INFO) << "AutoSchedule pass completed";
    return new_func;
  };

  return CreatePrimFuncPass(pass_func, 0, "tl.AutoSchedule", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AutoSchedule", AutoSchedule);
}

} // namespace tl
} // namespace tvm