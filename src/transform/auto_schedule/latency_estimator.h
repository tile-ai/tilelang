#pragma once

#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include "../../op/builtin.h"
#include "../../target/utils.h"
#include "./ir_structure.h"
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;
using ffi::String;

// Latency estimator for GPU with configurable hardware parameters
class LatencyEstimator {
public:
  // GPU latency parameters (H100 defaults, in cycles)
  struct GPUParams {
    // Base latencies
    int64_t global_memory_read = 1000; // Global memory read latency
    int64_t global_memory_write =
        800; // Global memory write latency (usually lower)
    int64_t shared_memory_read = 26;  // Shared memory read latency
    int64_t shared_memory_write = 26; // Shared memory write latency
    int64_t register_access = 1;      // Register access latency
    int64_t cuda_core_operation =
        4;                       // Basic CUDA core operation (add, mul, etc.)
    int64_t tma_operation = 100; // TMA operation latency
    int64_t tmem_read = 0;       // Tensor memory read latency (0 = not present)
    int64_t tmem_write = 0;     // Tensor memory write latency (0 = not present)
    int64_t tmem_bandwidth = 0; // Tensor memory bandwidth in bytes/cycle

    // Tensor Core shape-aware parameters
    int64_t tensor_core_base_latency =
        32; // Base latency for tensor core operation
    int64_t tensor_core_per_element_latency =
        1; // Additional latency per matrix element
    int64_t tensor_core_throughput =
        4096; // Number of tensor core operations per cycle (throughput)
    int64_t wgmma_base_latency = 40;    // Base latency for WGMMA operation
    int64_t wgmma_per_tile_latency = 2; // Additional latency per tile

    // Tensor Core II (Initiation Interval) parameters
    int64_t tensor_core_min_ii = 4; // Minimum II for tensor core operations
    int64_t tensor_core_ii_per_tile = 1;        // Additional II per WGMMA tile
    int64_t tensor_core_max_parallel_tiles = 8; // Max parallel tiles per SM

    // Bandwidth parameters (bytes per cycle)
    int64_t global_memory_bandwidth = 64;  // bytes per cycle
    int64_t shared_memory_bandwidth = 512; // bytes per cycle

    // Pipeline initiation capabilities
    int64_t max_memory_ops_per_cycle =
        1; // Max memory ops that can start per cycle

    // Different operation throughputs (operations per cycle)
    int64_t add_throughput = 4; // Addition throughput (ops/cycle)
    int64_t sub_throughput = 4; // Subtraction throughput (ops/cycle)
    int64_t mul_throughput = 4; // Multiplication throughput (ops/cycle)
    int64_t div_throughput =
        1; // Division throughput (ops/cycle, usually slower)
    int64_t mod_throughput = 1;     // Modulo throughput (ops/cycle)
    int64_t min_max_throughput = 4; // Min/Max throughput (ops/cycle)
    int64_t cmp_throughput = 4;     // Comparison throughput (ops/cycle)
    int64_t logic_throughput = 4;   // Logical operations throughput (ops/cycle)
    int64_t bitwise_throughput = 4; // Bitwise operations throughput (ops/cycle)
    int64_t shift_throughput = 4;   // Shift operations throughput (ops/cycle)
    int64_t special_func_throughput =
        1; // Special functions throughput (ops/cycle, exp2, log2, sin, cos,
           // etc.)
  };

  struct B200Params : GPUParams {
    B200Params() {
      global_memory_read = 714;
      global_memory_write = 571;
      shared_memory_read = 19;
      shared_memory_write = 19;
      cuda_core_operation = 3;
      tma_operation = 71;
      tensor_core_base_latency = 23;
      tensor_core_throughput = 9300;
      wgmma_base_latency = 29;
      wgmma_per_tile_latency = 1;
      global_memory_bandwidth = 109;
      add_throughput = 6;
      sub_throughput = 6;
      mul_throughput = 6;
      min_max_throughput = 6;
      cmp_throughput = 6;
      logic_throughput = 6;
      bitwise_throughput = 6;
      shift_throughput = 6;
      tmem_read = 1;
      tmem_write = 1;
      tmem_bandwidth = 0;
    }
  };

  explicit LatencyEstimator(Target target = Target()) {
    if (target.defined() && TargetIsSm100(target)) {
      params_ = B200Params();
    }
  }

  // Set thread count for parallel execution
  void SetThreadCount(int64_t thread_count) { thread_count_ = thread_count; }

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
    int64_t tmem_bytes = 0;

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
      case MemoryType::kTmem:
        tmem_bytes += access_bytes;
        break;
      default:
        global_memory_bytes += access_bytes;
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
      case MemoryType::kTmem:
        tmem_bytes += access_bytes;
        break;
      default:
        global_memory_bytes += access_bytes;
        break;
      }
    }

    // Estimate compute latency based on operation counting
    if (task->UsesCUDACore()) {
      // Use OperationCounter visitor to count operations and estimate latency
      OperationCounter counter(thread_count_, &params_);

      // Visit all statements in the task
      for (const auto &stmt : task->stmts) {
        counter(stmt);
      }

      // Get estimated latency
      compute_latency = counter.GetEstimatedLatency();
    }

    if (task->UsesTensorCore()) {
      // Shape-aware Tensor Core latency estimation
      int64_t tensor_core_latency = 32;
      ICHECK(task->HasTensorCoreShape());

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
    // if (has_tensor && !has_tma) LOG(FATAL) << memory_latency << " " <<
    // compute_latency;

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
      ii = total_latency;
      // Case 2: Only Tensor Core operations (no TMA)
      // Tensor Core operations are highly pipelined

      // Start with minimum II
      // ii = params_.tensor_core_min_ii;

      // If we have shape information, calculate more accurate II
      // if (task->HasTensorCoreShape()) {
      // Calculate total number of WGMMA tiles across all shapes
      // int64_t total_wgmma_tiles = task->GetTotalWGMMATiles();

      // II based on tile count and parallelism
      // More tiles may require larger II due to resource constraints
      // int64_t tile_based_ii =
      //     params_.tensor_core_min_ii +
      //     (total_wgmma_tiles + params_.tensor_core_max_parallel_tiles - 1) /
      //         params_.tensor_core_max_parallel_tiles;

      // ii = std::max(ii, tile_based_ii);

      // For very small operations, II can be smaller
      //   if (total_wgmma_tiles <= 2) {
      //     ii = std::min(ii, static_cast<int64_t>(2));
      //   }
      // }

      // II must be at least the bandwidth-limited II
      // if (global_memory_bytes > 0) {
      //   int64_t bandwidth_ii =
      //       (global_memory_bytes + params_.global_memory_bandwidth - 1) /
      //       params_.global_memory_bandwidth;
      //   ii = std::max(ii, bandwidth_ii);
      // }

      // if (shared_memory_bytes > 0) {
      //   int64_t bandwidth_ii =
      //       (shared_memory_bytes + params_.shared_memory_bandwidth - 1) /
      //       params_.shared_memory_bandwidth;
      //   ii = std::max(ii, bandwidth_ii);
      // }

      // For Tensor Core, II should be reasonable compared to latency
      // Typically II << latency for pipelined operations
      // int64_t max_ii_ratio = 4; // II should not exceed latency/4
      // int64_t max_reasonable_ii =
      //     (total_latency + max_ii_ratio - 1) / max_ii_ratio;
      // ii =
      //     std::min(ii, std::max(max_reasonable_ii,
      //     params_.tensor_core_min_ii));

    } else {
      // Case 3: Other cases (both TMA and Tensor Core, or neither)
      // Force II = total_latency for conservative scheduling
      ii = total_latency;

      // // Special case: single memory operation
      // if (num_memory_ops == 1 && task->stmts.size() == 1) {
      //   // Single operation that is a memory access
      //   // Check if this is likely a memory operation (has read/write
      //   regions) if (!task->GetReadRegions().empty() ||
      //       !task->GetWriteRegions().empty()) {
      //     ii = memory_latency;
      //   }
      // }

      // // Additional II constraints from bandwidth limitations
      // // II must be at least the time needed to transfer data based on
      // bandwidth if (global_memory_bytes > 0) {
      //   int64_t bandwidth_ii =
      //       (global_memory_bytes + params_.global_memory_bandwidth - 1) /
      //       params_.global_memory_bandwidth;
      //   ii = std::max(ii, bandwidth_ii);
      // }

      // if (shared_memory_bytes > 0) {
      //   int64_t bandwidth_ii =
      //       (shared_memory_bytes + params_.shared_memory_bandwidth - 1) /
      //       params_.shared_memory_bandwidth;
      //   ii = std::max(ii, bandwidth_ii);
      // }
    }

    // II must be at least 1 cycle
    if (ii < 1)
      ii = 1;

    // Store results in task node
    task->SetLatency(total_latency);
    task->SetII(ii);
  }

private:
  GPUParams params_;
  int64_t thread_count_ = 1; // Default to 1 (no parallelism)

  // Operation counter visitor with latency estimation
  class OperationCounter : public StmtExprVisitor {
  public:
    // Loop dimension information
    struct LoopDimension {
      const VarNode *var;
      int64_t trip_count;
      int depth;
    };

    int64_t total_latency = 0;
    int64_t thread_count = 1; // Thread count for parallel execution
    const LatencyEstimator::GPUParams *params = nullptr;

    // Track loop dimensions for loop-invariant detection
    std::vector<LoopDimension> loop_stack;
    std::unordered_map<const VarNode *, int> var_to_depth;

    OperationCounter(int64_t thread_count = 1,
                     const LatencyEstimator::GPUParams *params = nullptr)
        : thread_count(thread_count), params(params) {}

    // Operator to visit a statement
    void operator()(const Stmt &stmt) { VisitStmt(stmt); }

    // Get estimated latency
    int64_t GetEstimatedLatency() const { return total_latency; }

    // Helper function to update latency for an operation based on throughput
    void update_operation(
        int64_t throughput,
        const std::unordered_set<const VarNode *> &contained_vars) {
      int64_t effective_factor =
          CalculateEffectiveParallelFactor(contained_vars);

      // Throughput is operations per cycle
      // For 1 operation: cycles = 1 / throughput
      // For effective_factor operations: cycles = effective_factor / throughput
      // With thread-level parallelism: cycles = (effective_factor / throughput)
      // / thread_count

      if (throughput > 0) {
        // Calculate cycles needed for effective_factor operations with given
        // throughput
        int64_t cycles_for_operations =
            (effective_factor + throughput - 1) / throughput;

        if (cycles_for_operations >= thread_count) {
          total_latency +=
              (cycles_for_operations + thread_count - 1) / thread_count;
        } else {
          total_latency += 1; // At least 1 cycle
        }
      } else {
        // Zero or negative throughput, assume 1 cycle
        total_latency += 1;
      }
    }

    // Analyze which loop variables are contained in an expression
    std::unordered_set<const VarNode *>
    AnalyzeContainedLoopVars(const PrimExpr &expr) {
      class VarCollector : public ExprVisitor {
      public:
        const std::unordered_map<const VarNode *, int> &var_to_depth;
        std::unordered_set<const VarNode *> collected_vars;

        VarCollector(
            const std::unordered_map<const VarNode *, int> &var_to_depth)
            : var_to_depth(var_to_depth) {}

        void VisitExpr_(const VarNode *op) final {
          if (var_to_depth.count(op)) {
            collected_vars.insert(op);
          }
          ExprVisitor::VisitExpr_(op);
        }
      };

      VarCollector collector(var_to_depth);
      collector(expr);
      return collector.collected_vars;
    }

    // Calculate effective parallel factor based on contained loop variables
    int64_t CalculateEffectiveParallelFactor(
        const std::unordered_set<const VarNode *> &contained_vars) {
      if (contained_vars.empty()) {
        return 1; // Completely loop-invariant
      }

      // Find the maximum depth of contained loop variables
      int max_depth = -1;
      for (const VarNode *var : contained_vars) {
        auto it = var_to_depth.find(var);
        if (it != var_to_depth.end()) {
          max_depth = std::max(max_depth, it->second);
        }
      }

      // Calculate effective parallel factor: product of trip counts for loops
      // with depth <= max_depth
      int64_t effective_factor = 1;
      for (const auto &loop : loop_stack) {
        if (loop.depth <= max_depth) {
          effective_factor *= loop.trip_count;
        }
      }

      return effective_factor;
    }

    void VisitExpr_(const AddNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->add_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const SubNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->sub_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const MulNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->mul_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const DivNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->div_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const ModNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->mod_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const FloorDivNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->div_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const FloorModNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->mod_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const LTNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const LENode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const GTNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const GENode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const EQNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const NENode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const AndNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->logic_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const OrNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->logic_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const NotNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->logic_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const MinNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->min_max_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const MaxNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->min_max_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const CallNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));

      // Check for special math functions by name
      if (op->op.as<OpNode>()) {
        auto op_node = op->op.as<OpNode>();
        std::string op_name = op_node->name;

        // Check for special math functions
        if (op_name == "exp2" || op_name == "log2" || op_name == "exp" ||
            op_name == "log" || op_name == "sin" || op_name == "cos" ||
            op_name == "tan" || op_name == "asin" || op_name == "acos" ||
            op_name == "atan" || op_name == "sinh" || op_name == "cosh" ||
            op_name == "tanh" || op_name == "sqrt" || op_name == "rsqrt" ||
            op_name == "pow" || op_name == "erf" || op_name == "sigmoid") {
          // Special math functions have lower throughput
          update_operation(params->special_func_throughput, contained_vars);
        } else if (op_name.find("copy") != std::string::npos ||
                   op_name.find("gemm") != std::string::npos ||
                   op_name.find("tma_load") != std::string::npos ||
                   op_name.find("tma_store") != std::string::npos) {
          // Ignore copy, gemm, tma_load, tma_store operations
          // Just visit arguments but don't count the call itself
        } else {
          // Default: assume it's a regular function call with basic throughput
          update_operation(params->cuda_core_operation, contained_vars);
        }
      } else {
        // Not an OpNode, use default throughput
        update_operation(params->cuda_core_operation, contained_vars);
      }
      // Visit arguments
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const ForNode *op) final {
      // Calculate trip count for the loop
      int64_t trip_count = 1;
      PrimExpr loop_extent = op->extent;
      PrimExpr loop_step = op->step.has_value() ? op->step.value()
                                                : IntImm(DataType::Int(32), 1);

      // Try to get constant values
      if (const int64_t *extent_ptr = as_const_int(loop_extent)) {
        if (const int64_t *step_ptr = as_const_int(loop_step)) {
          int64_t extent = *extent_ptr;
          int64_t step = *step_ptr;
          if (step > 0) {
            // ceil(extent / step) = (extent + step - 1) / step
            trip_count = (extent + step - 1) / step;
          } else {
            trip_count = extent; // Invalid step, use extent
          }
        } else {
          trip_count = 100; // Non-constant step, use default
        }
      } else {
        trip_count = 100; // Non-constant extent, use default
      }

      // Create loop dimension information
      LoopDimension loop_dim{.var = op->loop_var.get(),
                             .trip_count = trip_count,
                             .depth = static_cast<int>(loop_stack.size())};

      // Push loop onto stack and update mapping
      loop_stack.push_back(loop_dim);
      var_to_depth[op->loop_var.get()] = loop_dim.depth;

      // Visit loop body
      StmtExprVisitor::VisitStmt_(op);

      // Pop loop from stack
      var_to_depth.erase(op->loop_var.get());
      loop_stack.pop_back();
    }

    void VisitStmt_(const EvaluateNode *op) final {
      if (op->value.defined()) {
        StmtExprVisitor::VisitExpr(op->value);
      }
    }

    void VisitStmt_(const BufferStoreNode *op) final {
      // Count operations in indices
      for (const auto &index : op->indices) {
        StmtExprVisitor::VisitExpr(index);
      }
      // Count operations in value
      StmtExprVisitor::VisitExpr(op->value);
    }

    void VisitStmt_(const SeqStmtNode *op) final {
      for (const auto &child : op->seq) {
        StmtExprVisitor::VisitStmt(child);
      }
    }

    void VisitStmt_(const AttrStmtNode *op) final {
      StmtExprVisitor::VisitStmt(op->body);
    }

    void VisitStmt_(const LetStmtNode *op) final {
      // Let binding: the value expression is evaluated once
      StmtExprVisitor::VisitExpr(op->value);
      StmtExprVisitor::VisitStmt(op->body);
    }

    void VisitStmt_(const IfThenElseNode *op) final {
      VisitExpr(op->condition);

      // For if-then-else branches, we need to take the maximum latency of both
      // paths Save current latency
      int64_t old_latency = total_latency;

      // Count latency in then branch
      StmtExprVisitor::VisitStmt(op->then_case);
      int64_t then_latency = total_latency;

      // Restore latency and count else branch
      total_latency = old_latency;
      if (op->else_case) {
        StmtExprVisitor::VisitStmt(op->else_case.value());
      }
      int64_t else_latency = total_latency;

      // Take the maximum of both branches
      total_latency = old_latency + std::max(then_latency - old_latency,
                                             else_latency - old_latency);
    }

    void VisitStmt_(const BlockNode *op) final {
      StmtExprVisitor::VisitStmt(op->body);
    }
  };

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
               (access_bytes - 32) / params_.global_memory_bandwidth;
      } else {
        // Write latency usually lower
        return params_.global_memory_write +
               (access_bytes - 32) / params_.global_memory_bandwidth;
      }
    case MemoryType::kShared:
      // Shared memory has high bandwidth, less sensitive to size
      // Subtract typical burst size (128 bytes) for base latency
      if (is_read) {
        return params_.shared_memory_read +
               (access_bytes - 128) / params_.shared_memory_bandwidth;
      } else {
        return params_.shared_memory_write +
               (access_bytes - 128) / params_.shared_memory_bandwidth;
      }
    case MemoryType::kRegister:
      return params_.register_access;
    case MemoryType::kTmem:
      if (params_.tmem_read == 0) {
        return params_.global_memory_read +
               (access_bytes - 32) / params_.global_memory_bandwidth;
      }
      return is_read ? params_.tmem_read : params_.tmem_write;
    default:
      // Unknown memory type, use global memory as conservative estimate
      if (is_read) {
        return params_.global_memory_read +
               (access_bytes - 32) / params_.global_memory_bandwidth;
      } else {
        return params_.global_memory_write +
               (access_bytes - 32) / params_.global_memory_bandwidth;
      }
    }
  }
};

} // namespace tl
} // namespace tvm
