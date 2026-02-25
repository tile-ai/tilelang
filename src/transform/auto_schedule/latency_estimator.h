#pragma once

#include <tvm/runtime/logging.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

#include "./ir_structure.h"
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
        64; // Tensor core operation (matrix multiply) - base latency
    int64_t tma_operation = 100; // TMA operation latency

    // Tensor Core shape-aware parameters
    int64_t tensor_core_base_latency =
        32; // Base latency for tensor core operation
    int64_t tensor_core_per_element_latency =
        1; // Additional latency per matrix element
    int64_t tensor_core_throughput =
        4; // Number of tensor core operations per cycle (throughput)
    int64_t wgmma_base_latency = 40;    // Base latency for WGMMA operation
    int64_t wgmma_per_tile_latency = 2; // Additional latency per tile

    // Tensor Core II (Initiation Interval) parameters
    int64_t tensor_core_min_ii = 4; // Minimum II for tensor core operations
    int64_t tensor_core_ii_per_tile = 1;        // Additional II per WGMMA tile
    int64_t tensor_core_max_parallel_tiles = 8; // Max parallel tiles per SM

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
      // Shape-aware Tensor Core latency estimation
      int64_t tensor_core_latency =
          params_.tensor_core_operation; // Default fallback

      // Check if we have shape information
      if (task->HasTensorCoreShape()) {
        // Calculate total operations across all Tensor Core shapes
        int64_t total_ops = task->GetTotalTensorCoreOps();

        // Simple model: latency = base + ops / throughput
        // This is a simplified model that can be refined based on actual
        // hardware measurements
        tensor_core_latency = params_.tensor_core_base_latency +
                              (total_ops + params_.tensor_core_throughput - 1) /
                                  params_.tensor_core_throughput;

        // Clamp to reasonable values
        tensor_core_latency =
            std::max(tensor_core_latency, params_.tensor_core_base_latency);
        tensor_core_latency = std::min(
            tensor_core_latency, static_cast<int64_t>(1000)); // Max 1000 cycles

        // For WGMMA operations (common in TileLang), use a different model
        // WGMMA typically operates on tiles of fixed size (e.g., 16x16x16 for
        // fp16) Calculate total number of WGMMA tiles across all shapes
        int64_t total_wgmma_tiles = task->GetTotalWGMMATiles();
        int64_t wgmma_latency =
            params_.wgmma_base_latency +
            total_wgmma_tiles * params_.wgmma_per_tile_latency;

        // Use the minimum of the two models
        tensor_core_latency = std::min(tensor_core_latency, wgmma_latency);
      }

      compute_latency = std::max(compute_latency, tensor_core_latency);
    }

    if (task->UsesTMACore()) {
      compute_latency = std::max(compute_latency, params_.tma_operation);
    }

    // Total latency is sum of memory and compute (assuming sequential for now)
    total_latency = memory_latency + compute_latency;

    // Calculate initiation interval (II)
    int64_t ii = 1; // Default minimum II

    bool has_tma = task->UsesTMACore();
    bool has_tensor = task->UsesTensorCore();

    if (has_tma && !has_tensor) {
      // Case 1: Only TMA operations (no Tensor Core)
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
    } else if (!has_tma && has_tensor) {
      // Case 2: Only Tensor Core operations (no TMA)
      // Tensor Core operations are highly pipelined

      // Start with minimum II
      ii = params_.tensor_core_min_ii;

      // If we have shape information, calculate more accurate II
      if (task->HasTensorCoreShape()) {
        // Calculate total number of WGMMA tiles across all shapes
        int64_t total_wgmma_tiles = task->GetTotalWGMMATiles();

        // II based on tile count and parallelism
        // More tiles may require larger II due to resource constraints
        int64_t tile_based_ii =
            params_.tensor_core_min_ii +
            (total_wgmma_tiles + params_.tensor_core_max_parallel_tiles - 1) /
                params_.tensor_core_max_parallel_tiles;

        ii = std::max(ii, tile_based_ii);

        // For very small operations, II can be smaller
        if (total_wgmma_tiles <= 2) {
          ii = std::min(ii, static_cast<int64_t>(2));
        }
      }

      // II must be at least the bandwidth-limited II
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

      // For Tensor Core, II should be reasonable compared to latency
      // Typically II << latency for pipelined operations
      int64_t max_ii_ratio = 4; // II should not exceed latency/4
      int64_t max_reasonable_ii =
          (total_latency + max_ii_ratio - 1) / max_ii_ratio;
      ii =
          std::min(ii, std::max(max_reasonable_ii, params_.tensor_core_min_ii));

    } else {
      // Case 3: Other cases (both TMA and Tensor Core, or neither)
      // Force II = total_latency for conservative scheduling
      ii = total_latency;

      // Special case: single memory operation
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

} // namespace tl
} // namespace tvm
