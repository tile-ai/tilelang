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
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "../op/gemm_py.h"
#include "../target/utils.h"
#include "./common/attr.h"
#include "./common/collector.h"
#include "auto_schedule.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using ffi::GetRef;

// Mutator to update thread extent in AttrStmt nodes
// Used after warpgroup partition to double thread extent
class ThreadExtentUpdater : public StmtExprMutator {
public:
  explicit ThreadExtentUpdater(PrimExpr updated_extent)
      : updated_thread_extent_(updated_extent) {}

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      auto thread_iv_ = Downcast<IterVar>(op->node);
      if (thread_iv_->thread_tag == "threadIdx.x") {
        // Visit the body first (to update any references)
        AttrStmt attr_stmt =
            Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));

        // Update the thread extent

        // Create new IterVar with updated domain
        Range new_dom =
            Range::FromMinExtent(thread_iv_->dom->min, updated_thread_extent_);

        // Update the AttrStmt with new IterVar and value
        thread_iv_.CopyOnWrite()->dom = new_dom;
        attr_stmt.CopyOnWrite()->node = thread_iv_;
        attr_stmt.CopyOnWrite()->value = updated_thread_extent_;

        // Clear the saved reference
        thread_iv_ = {};

        return attr_stmt;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

private:
  PrimExpr updated_thread_extent_;
  IterVar thread_iv_;
};

// Visitor to extract the body of tilelang_root block
class TilelangRootBodyExtractor : public StmtVisitor {
public:
  Stmt body;

  void VisitStmt_(const BlockNode *op) override {
    if (op->name_hint == "tilelang_root") {
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

  Stmt VisitStmt_(const BlockNode *op) override {
    auto block = GetRef<Block>(op);
    if (op->name_hint == "tilelang_root") {
      // Keep all block attributes but replace the body
      return Block(op->iter_vars, op->reads, op->writes, op->name_hint,
                   new_body_, op->init, op->alloc_buffers, op->match_buffers,
                   op->annotations);
    }
    return StmtMutator::VisitStmt_(op);
  }

private:
  Stmt new_body_;
};

// Visitor to build IRStructure from TIR statements
class IRStructureBuilder : public StmtVisitor {
public:
  std::shared_ptr<IRStructure> Build(const Stmt &stmt, int64_t thread_count = 1,
                                     Target target = Target()) {
    thread_count_ = thread_count;
    target_ = target;
    VisitStmt(stmt);
    if (!root_) {
      LOG(WARNING)
          << "IRStructureBuilder: root_ is null after visiting statement. "
          << "This may indicate an unhandled statement type.";
      // Return an empty TaskNode as fallback
      auto task_node = std::make_shared<TaskNode>();
      task_node->stmts.push_back(stmt);
      return task_node;
    }
    return std::move(root_);
  }

protected:
  void VisitStmt_(const SeqStmtNode *op) override {
    auto seq_node = std::make_shared<SequenceNode>();

    for (size_t i = 0; i < op->seq.size(); i++) {
      VisitStmt(op->seq[i]);
      if (root_) {
        seq_node->children.push_back(std::move(root_));
      }
    }
    root_ = std::move(seq_node);
  }

  void VisitStmt_(const ForNode *op) override {
    // Determine if this is a sequential or parallel for
    if (op->kind == ForKind::kSerial) {
      // Sequential For -> ControlNode
      auto control_node = std::make_shared<ControlNode>();
      control_node->control = GetRef<For>(op);
      control_node->task = std::make_shared<TaskNode>();
      control_node->task->stmts.push_back(
          For(op->loop_var, op->min, op->extent, op->kind, Evaluate(0),
              op->thread_binding, op->annotations, op->step, op->span));
      AnalyzeResourceUsage(Evaluate(op->min), control_node->task.get(), true);
      AnalyzeResourceUsage(Evaluate(op->extent), control_node->task.get(),
                           true);
      if (op->step.defined())
        AnalyzeResourceUsage(Evaluate(GetRef<PrimExpr>(op->step.get())),
                             control_node->task.get(), true);

      // Process the loop body
      VisitStmt(op->body);
      if (root_) {
        control_node->child = std::move(root_);
      } else {
      }

      root_ = std::move(control_node);
    } else {
      // Parallel For -> TaskNode
      auto task_node = std::make_shared<TaskNode>();
      task_node->stmts.push_back(GetRef<Stmt>(op));

      // Analyze the loop body for resource usage)
      AnalyzeResourceUsage(op->body, task_node.get());
      AnalyzeResourceUsage(Evaluate(op->min), task_node.get(), true);
      AnalyzeResourceUsage(Evaluate(op->extent), task_node.get(), true);
      if (op->step.defined())
        AnalyzeResourceUsage(Evaluate(GetRef<PrimExpr>(op->step.get())),
                             task_node.get(), true);

      root_ = std::move(task_node);
    }
  }

  void VisitStmt_(const EvaluateNode *op) override {
    // Evaluate statement (usually a Call) -> TaskNode
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    // Analyze the expression for resource usage
    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());

    root_ = std::move(task_node);
  }

  void VisitStmt_(const IfThenElseNode *op) override {
    // If statement -> treat as TaskNode for now (could be refined later)
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    AnalyzeMemoryExpr(op->condition, task_node.get());
    AnalyzeResourceUsage(Evaluate(op->condition), task_node.get(), true);

    // Analyze both branches for resource usage
    AnalyzeResourceUsage(op->then_case, task_node.get());
    if (op->else_case) {
      AnalyzeResourceUsage(op->else_case.value(), task_node.get());
    }

    root_ = std::move(task_node);
  }

  void VisitStmt_(const LetStmtNode *op) override {
    // Wrapper statement -> WrapperNode
    auto wrapper_node = std::make_shared<WrapperNode>();
    wrapper_node->wrapper = GetRef<Stmt>(op);
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetLetDecl(op));
    AnalyzeResourceUsage(GetLetDecl(op), task_node.get());
    wrapper_node->task = std::move(task_node);

    // Process the wrapperbody
    VisitStmt(op->body);
    if (root_) {
      wrapper_node->child = std::move(root_);
    }

    root_ = std::move(wrapper_node);
  }

  void VisitStmt_(const AttrStmtNode *op) override {
    // Wrapper statement -> WrapperNode
    auto wrapper_node = std::make_shared<WrapperNode>();
    wrapper_node->wrapper = GetRef<Stmt>(op);
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetAttrDecl(op));
    AnalyzeResourceUsage(GetAttrDecl(op), task_node.get());
    wrapper_node->task = std::move(task_node);

    // Process the wrapperbody
    VisitStmt(op->body);
    if (root_) {
      wrapper_node->child = std::move(root_);
    }

    root_ = std::move(wrapper_node);
  }

  void VisitStmt_(const WhileNode *op) override {
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    // Analyze condition and body for resource usage
    AnalyzeResourceUsage(Evaluate(op->condition), task_node.get());
    AnalyzeResourceUsage(op->body, task_node.get());

    root_ = std::move(task_node);
  }

  void VisitStmt_(const BlockNode *op) override {
    // All blocks are treated as TaskNode
    // Note: tilelang_root block should have been extracted by
    // TilelangRootBodyExtractor If we encounter it here, it means we're
    // processing the entire function body (not extracted), which should only
    // happen when there's no tilelang_root block
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));
    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());
    root_ = std::move(task_node);
  }

private:
  std::shared_ptr<IRStructure> root_;
  int64_t thread_count_ = 1;
  Target target_;

  void AnalyzeResourceUsage(const Stmt &stmt, TaskNode *task_node,
                            bool only_variables = false) {
    // Recursively analyze statements to determine resource usage
    struct ResourceAnalyzer : public StmtExprVisitor {
      TaskNode *task_node;
      bool found_tma{false};
      bool found_tensor{false};
      bool found_cuda{false};

      bool found_tma_load{false};

      // Tensor Core shape information (multiple shapes possible)
      struct TensorCoreShape {
        int64_t m;
        int64_t n;
        int64_t k;
        TensorCoreShape(int64_t m = 0, int64_t n = 0, int64_t k = 0)
            : m(m), n(n), k(k) {}
      };
      std::vector<TensorCoreShape> tensor_core_shapes;

      // GemmInst: the resolved tensor core instruction (single, asserted
      // shared)
      GemmInst gemm_inst{GemmInst::kMMA};
      bool has_gemm_inst{false};

      // Target and block_size for GemmInst determination
      Target target;
      int64_t block_size;

      ResourceAnalyzer(TaskNode *node, Target target = Target(),
                       int64_t block_size = 128)
          : task_node(node), target(target), block_size(block_size) {}

      void VisitExpr_(const CallNode *op) override {
        // Check for specific TileLang operations
        static const auto copy_op = Op::Get("tl.tileop.copy");
        static const auto gemm_py_op = Op::Get("tl.tileop.gemm_py");
        static const auto gemm_op = Op::Get("tl.tileop.gemm");
        static const auto reduce_op = Op::Get("tl.tileop.reduce");
        static const auto fill_op = Op::Get("tl.tileop.fill");
        static const auto region_op = Op::Get("tl.tileop.region");

        // Try to get operation name for logging
        std::string op_name = "unknown";
        if (const auto *op_ptr = op->op.as<OpNode>()) {
          op_name = op_ptr->name;
        }

        // Check if this is a TMA copy operation
        if (op->op.same_as(copy_op)) {
          bool found_global = false, found_shared = false;
          int idx_global = -1, idx_shared = -1;
          for (unsigned idx = 0; idx != 2; ++idx) {
            auto region = Downcast<Call>(op->args[idx]);
            if (const auto *buffer_load =
                    region->args[0].as<BufferLoadNode>()) {
              Buffer buffer = buffer_load->buffer;
              String scope = buffer.scope();
              MemoryType mem_type = GetMemoryTypeFromScope(scope);
              if (mem_type == MemoryType::kGlobal) {
                found_global = true;
                idx_global = idx;
              }
              if (mem_type == MemoryType::kShared) {
                found_shared = true;
                idx_shared = idx;
              }
            }
          }
          found_tma = false;
          if (found_global && found_shared) {
            if (idx_global == 0 && idx_shared == 1) {
              found_tma = true;
              found_tma_load = true;
            }
            if (idx_global == 1 && idx_shared == 0)
              found_tma = true;
          }
        } else if (op->op.same_as(gemm_py_op) || op->op.same_as(gemm_op)) {
          found_tensor = true;

          int64_t m = op->args[5].as<IntImmNode>()->value;
          int64_t n = op->args[6].as<IntImmNode>()->value;
          int64_t k = op->args[7].as<IntImmNode>()->value;
          tensor_core_shapes.emplace_back(m, n, k);

          // Determine the final GemmInst using GemmPyNode::getGemmInst
          if (target.defined()) {
            GemmPy gemm_py(op->args);
            GemmInst inst =
                gemm_py->getGemmInst(static_cast<int>(block_size), target);
            ICHECK(!has_gemm_inst || gemm_inst == inst)
                << "All gemm operations in a task must use the same GemmInst, "
                << "but got " << GemmInstToString(gemm_inst) << " and "
                << GemmInstToString(inst);
            gemm_inst = inst;
            has_gemm_inst = true;
          }
        } else if (op->op.same_as(reduce_op) || op->op.same_as(fill_op)) {
          // Reduce and fill operations use CUDA core
          found_cuda = true;
        } else if (op->op.same_as(region_op)) {
          // Handle tl.tileop.region call for memory access analysis
          // args[0] = buffer (BufferLoad), args[1] = access_type (1: read, 2:
          // write, 3: read/write) args[2..] = extents
          if (op->args.size() >= 2) {
            // Extract access type
            if (const auto *access_int = op->args[1].as<IntImmNode>()) {
              int access_type = access_int->value;
              // For now, just mark as CUDA operation (memory access)
              found_cuda = true;
              // TODO: Extract buffer region and add to task_node->read_regions
              // or write_regions BufferLoad buffer_load =
              // Downcast<BufferLoad>(op->args[0]); Construct BufferRegion from
              // buffer_load and extents if (access_type == 1 || access_type ==
              // 3) task_node->read_regions.push_back(region); if (access_type
              // == 2 || access_type == 3)
              // task_node->write_regions.push_back(region);
            }
          }
        } else {
          // Check for other known operations that use CUDA core
          // For now, assume any other call is a basic computation
          found_cuda = true;
        }

        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const AddNode *op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const SubNode *op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const MulNode *op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const DivNode *op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }
    };

    ResourceAnalyzer analyzer(task_node, target_, thread_count_);
    analyzer(stmt);

    if (!only_variables) {
      // Set task node flags based on what was found
      if (analyzer.found_tma) {
        task_node->SetUsesTMACore(true);
        if (analyzer.found_tma_load) {
          task_node->SetHasTMALoad(true);
        }
      }
      if (analyzer.found_tensor) {
        task_node->SetUsesTensorCore(true);
        // Set Tensor Core shape information if available
        for (const auto &shape : analyzer.tensor_core_shapes) {
          if (shape.m > 0 && shape.n > 0 && shape.k > 0) {
            task_node->AddTensorCoreShape(shape.m, shape.n, shape.k);
          }
        }
        // Set GemmInst information
        if (analyzer.has_gemm_inst) {
          task_node->SetGemmInst(analyzer.gemm_inst);
        }
      }
      // If neither TMA nor Tensor core was used, and CUDA operations were
      // found, set CUDA core flag
      if (!analyzer.found_tma && !analyzer.found_tensor) {
        task_node->SetUsesCUDACore(true);
      }
    }

    // Analyze memory access regions
    MemoryAccessDetector memory_detector;
    memory_detector.Analyze(stmt);
    std::vector<BufferRegion> read_regions = memory_detector.GetReadRegions();
    std::vector<BufferRegion> write_regions = memory_detector.GetWriteRegions();
    std::vector<Var> read_vars = memory_detector.GetReadVars();
    std::vector<Var> write_vars = memory_detector.GetWriteVars();

    // Merge with existing regions (avoid duplicates)
    for (const auto &region : read_regions) {
      task_node->AddReadRegion(region);
    }

    for (const auto &region : write_regions) {
      task_node->AddWriteRegion(region);
    }

    for (const auto &var : read_vars) {
      task_node->AddReadVar(var);
    }

    for (const auto &var : write_vars) {
      task_node->AddWriteVar(var);
    }

    // Estimate latency and initiation interval for this task
    LatencyEstimator latency_estimator(target_);
    latency_estimator.SetThreadCount(thread_count_);
    latency_estimator.Estimate(task_node);
  }

  void AnalyzeMemoryExpr(const PrimExpr &expr, TaskNode *task_node) {
    // Analyze memory access regions
    MemoryAccessDetector memory_detector;
    memory_detector.Analyze(expr);
    std::vector<BufferRegion> read_regions = memory_detector.GetReadRegions();
    std::vector<BufferRegion> write_regions = memory_detector.GetWriteRegions();

    // Merge with existing regions (avoid duplicates)
    for (const auto &region : read_regions) {
      task_node->AddReadRegion(region);
    }

    for (const auto &region : write_regions) {
      task_node->AddWriteRegion(region);
    }

    // Estimate latency and initiation interval for this task
    LatencyEstimator latency_estimator(target_);
    latency_estimator.SetThreadCount(thread_count_);
    latency_estimator.Estimate(task_node);
  }
};

Stmt ReNestLetStmts(const Stmt &stmt);

// The main pass function
tvm::transform::Pass AutoSchedule(const bool enable_epi) {
  using namespace tir::transform;
  auto pass_func =
      [enable_epi](PrimFunc func, const IRModule &mod,
                   const tvm::transform::PassContext &ctx) -> PrimFunc {
    // Get target from PrimFunc attribute for GemmInst determination
    auto target_opt = func->GetAttr<Target>(tvm::attr::kTarget);
    Target target;
    if (target_opt.defined()) {
      target = target_opt.value();
    }
    auto config = GetWarpSpecializeConfig(target);

    // Extract the body of tilelang_root block if it exists
    TilelangRootBodyExtractor extractor;
    extractor(func->body);
    Stmt body_to_schedule;
    bool has_tilelang_root = false;
    PrimExpr updated_thread_extent; // Will be set if warpgroup partition
                                    // doubles thread extent
    IterVar thread_var; // Thread index variable for warpgroup partition

    if (extractor.body.defined()) {
      body_to_schedule = extractor.body;
      has_tilelang_root = true;
    } else {
      LOG(FATAL);
      body_to_schedule = func->body;
    }

    // Get thread index variable for warpgroup partition
    // First try to get from body_to_schedule, if not found, try from the entire
    // function body
    thread_var = ThreadTagChecker::GetThreadVar(body_to_schedule);
    if (!thread_var.defined()) {
      thread_var = ThreadTagChecker::GetThreadVar(func->body);
    }

    // Calculate thread count for latency estimation
    int64_t latency_thread_count = 1;
    if (thread_var.defined() && thread_var->dom.defined()) {
      PrimExpr thread_extent = thread_var->dom->extent;
      if (const int64_t *extent_ptr = as_const_int(thread_extent)) {
        latency_thread_count = *extent_ptr;
        if (latency_thread_count < 1)
          latency_thread_count = 1;
      }
    }

    // Build IRStructure from the body to schedule
    IRStructureBuilder builder;
    auto ir_structure =
        builder.Build(body_to_schedule, latency_thread_count, target);

    // Print the built IRStructure with all statements
    ICHECK(ir_structure) << "IRStructure is null (empty body?)";

    // First print the summary view
    // PrintIRStructure(ir_structure.get());

    // Then print all statements
    // PrintAllStmts(ir_structure.get());

    // Build ScheduleUnits from IRStructure
    ScheduleUnitBuilder unit_builder;
    // Get thread index variable for warpgroup partition
    // First try to get from body_to_schedule, if not found, try from the entire
    // function body
    thread_var = ThreadTagChecker::GetThreadVar(body_to_schedule);
    if (!thread_var.defined()) {
      thread_var = ThreadTagChecker::GetThreadVar(func->body);
    }
    if (thread_var.defined()) {
      unit_builder.SetThreadVar(thread_var);
    } else {
      LOG(FATAL) << "Could not find thread index variable, warpgroup "
                    "partition will use default";
    }
    unit_builder.SetEnableWarpPartition(config.enable_warp_partition);
    unit_builder.SetSharedMemoryLimit(config.shared_memory_limit);
    bool double_thread = unit_builder.Build(ir_structure);

    if (!config.enable_warpgroup_partition) {
      Stmt new_body = ConvertIRStructureToStmt(ir_structure.get(), enable_epi);

      // If we extracted from tilelang_root block, replace the body
      Stmt final_body;
      TilelangRootBodyReplacer replacer(new_body);
      final_body = replacer(func->body);

      final_body = ReNestLetStmts(final_body);

      // Create a new PrimFunc with the updated body
      auto new_func = PrimFunc(func->params, final_body, func->ret_type,
                               func->buffer_map, func->attrs);
      return new_func;
    }

    // Print the modified summary view
    // PrintIRStructure(ir_structure.get());

    // Analyze buffer dependencies and insert barriers before warpgroup
    // partition
    int next_barrier_id = 1;
    std::vector<Buffer> barrier_buffers;
    Map<ObjectRef, ObjectRef> barrier_map;
    // Determine thread count for barrier arrive_count calculations
    PrimExpr thread_count[2];
    if (!config.enable_thread_extend) {
      ICHECK(config.enable_warp_partition);
      // sm_100: use fixed warp size (32) for both partitions
      thread_count[0] = IntImm(DataType::Int(32), 32);
      thread_count[1] = IntImm(DataType::Int(32), 32);
    } else {
      // sm_90: original behavior
      thread_count[0] = thread_var->dom->extent;
      thread_count[1] = double_thread ? thread_var->dom->extent
                                      : IntImm(DataType::Int(32),
                                               config.producer_thread_count);
    }
    LoopNestingInfo loop_info;
    std::vector<MultiVersionBufferInfo> buffer_infos;
    PrimExpr barrier_count = config.enable_thread_extend
                                 ? thread_count[0] + thread_count[1]
                                 : thread_var->dom->extent;
    Buffer neutral_sync_shared_barrier =
        makeBarrierBuffer(barrier_count, "neutral_sync_shared_barrier", 1,
                          barrier_buffers, barrier_map);
    AnalyzeAndInsertBarriers(
        ir_structure.get(), next_barrier_id, barrier_buffers, barrier_map,
        thread_count, loop_info, buffer_infos, neutral_sync_shared_barrier);

    // Print the modified summary view
    // PrintIRStructure(ir_structure.get());

    // Apply warpgroup partition to entire IRStructure
    Stmt new_body = ApplyWarpgroupPartitionToIRStructure(
        ir_structure.get(), thread_var, barrier_buffers, barrier_map,
        enable_epi, thread_count, double_thread, config,
        neutral_sync_shared_barrier);

    if (config.enable_thread_extend) {
      // sm_90: may need to update thread extent
      if (double_thread) {
        updated_thread_extent = thread_var->dom->extent * 2;
      } else {
        updated_thread_extent =
            thread_var->dom->extent +
            IntImm(DataType::Int(32), config.producer_thread_count);
      }
    }

    // If we extracted from tilelang_root block, replace the body
    Stmt final_body;
    TilelangRootBodyReplacer replacer(new_body);
    final_body = replacer(func->body);
    // Apply thread extent update if warpgroup partition was applied (sm_90
    // only)
    if (config.enable_thread_extend) {
      ThreadExtentUpdater extent_updater(updated_thread_extent);
      final_body = extent_updater(final_body);
    }
    // Add barrier buffers to tilelang_root block's alloc_buffers
    if (!barrier_buffers.empty()) {
      class TilelangRootAllocBufferAdder : public StmtMutator {
      public:
        explicit TilelangRootAllocBufferAdder(
            const std::vector<Buffer> &buffers_to_add,
            Map<ObjectRef, ObjectRef> &barrier_map)
            : buffers_to_add_(buffers_to_add), barrier_map_(barrier_map) {}

        Stmt VisitStmt_(const BlockNode *op) override {
          auto block = GetRef<Block>(op);
          if (op->name_hint == "tilelang_root") {
            // Combine existing alloc_buffers with new buffers
            Array<Buffer> new_alloc_buffers = op->alloc_buffers;
            for (const auto &buffer : buffers_to_add_) {
              new_alloc_buffers.push_back(buffer);
            }
            auto new_annotations = op->annotations;
            new_annotations.Set("barrier_init", barrier_map_);
            // Create new block with updated alloc_buffers
            return Block(op->iter_vars, op->reads, op->writes, op->name_hint,
                         op->body, op->init, new_alloc_buffers,
                         op->match_buffers, new_annotations);
          }
          return StmtMutator::VisitStmt_(op);
        }

      private:
        std::vector<Buffer> buffers_to_add_;
        Map<ObjectRef, ObjectRef> &barrier_map_;
      };

      TilelangRootAllocBufferAdder adder(barrier_buffers, barrier_map);
      final_body = adder(final_body);
    }

    // Apply multi-version alloc_buffer rewrite if needed
    if (!buffer_infos.empty()) {
      final_body = RewriteAllocBuffers(final_body, buffer_infos);
    }

    final_body = ReNestLetStmts(final_body);

    // Create a new PrimFunc with the updated body
    auto new_func = PrimFunc(func->params, final_body, func->ret_type,
                             func->buffer_map, func->attrs);
    return new_func;
  };

  return CreatePrimFuncPass(pass_func, 0, "tl.AutoSchedule", {});
}

// Re-write LetStmt to nest them properly
// Example transformation:
//   SeqStmt {
//     let x = 42 { Evaluate(0) }     // standalone, empty body
//     let y = x+1 { Evaluate(0) }    // standalone, empty body
//     compute(x, y)                   // actual work
//     store(result)
//   }
// becomes:
//   let x = 42 {
//     let y = x+1 {
//       SeqStmt {
//         compute(x, y)
//         store(result)
//       }
//     }
//   }
class LetStmtNester : public StmtMutator {
public:
  Stmt VisitStmt_(const SeqStmtNode *op) override {
    Array<Stmt> stmts;
    for (const auto &stmt : op->seq) {
      stmts.push_back(this->VisitStmt(stmt));
    }

    Array<Stmt> flat_stmts;
    for (const auto &stmt : stmts) {
      if (const auto *inner_seq = stmt.as<SeqStmtNode>()) {
        for (const auto &inner_stmt : inner_seq->seq) {
          flat_stmts.push_back(inner_stmt);
        }
      } else {
        flat_stmts.push_back(stmt);
      }
    }
    stmts = flat_stmts;

    for (int i = static_cast<int>(stmts.size()) - 2; i >= 0; --i) {
      if (const auto *let = stmts[i].as<LetStmtNode>()) {
        if (IsEmptyBody(let->body)) {
          Stmt absorbed_body = CollectRemaining(stmts, i + 1);
          stmts = TruncateAndReplace(
              stmts, i, LetStmt(let->var, let->value, absorbed_body));
        }
      } else if (const auto *attr = stmts[i].as<AttrStmtNode>()) {
        if (IsEmptyBody(attr->body)) {
          Stmt absorbed_body = CollectRemaining(stmts, i + 1);
          stmts = TruncateAndReplace(
              stmts, i,
              AttrStmt(attr->node, attr->attr_key, attr->value, absorbed_body));
        }
      }
    }

    if (stmts.empty())
      return Evaluate(0);
    if (stmts.size() == 1)
      return stmts[0];

    return SeqStmt(stmts);
  }

private:
  // Check if a statement body is Evaluate(0) — the empty placeholder
  static bool IsEmptyBody(const Stmt &stmt) {
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (const auto *imm = eval->value.as<IntImmNode>()) {
        return imm->value == 0;
      }
    }
    return false;
  }

  // Collect stmts[start .. end) into a single Stmt
  static Stmt CollectRemaining(const Array<Stmt> &stmts, int start) {
    int n = static_cast<int>(stmts.size());
    if (start >= n) {
      return Evaluate(0);
    }
    if (start == n - 1) {
      return stmts[start];
    }
    Array<Stmt> remaining;
    for (int j = start; j < n; ++j) {
      remaining.push_back(stmts[j]);
    }
    return SeqStmt(remaining);
  }

  // Keep stmts[0..index), replace stmts[index] with new_stmt,
  // discard everything after (already absorbed into new_stmt body)
  static Array<Stmt> TruncateAndReplace(const Array<Stmt> &stmts, int index,
                                        Stmt new_stmt) {
    Array<Stmt> result;
    for (int j = 0; j < index; ++j) {
      result.push_back(stmts[j]);
    }
    result.push_back(new_stmt);
    return result;
  }
};

Stmt ReNestLetStmts(const Stmt &stmt) {
  LetStmtNester nester;
  return nester(stmt);
}

// StmtMutator to rewrite alloc_buffers in Block nodes
class AllocBufferRewriter : public StmtMutator {
public:
  AllocBufferRewriter(const std::vector<MultiVersionBufferInfo> &buffer_infos)
      : buffer_infos_(buffer_infos) {
    // Create mapping from original buffer to new buffer
    for (const auto &info : buffer_infos_) {
      buffer_remap_[info.buffer] = info.new_buffer;
    }
  }

private:
  Stmt VisitStmt_(const BlockNode *op) override {
    Stmt new_body = this->VisitStmt(op->body);

    // Check if we need to update alloc_buffers
    bool needs_update = false;
    Array<Buffer> new_alloc_buffers;

    for (auto buffer : op->alloc_buffers) {
      auto it = buffer_remap_.find(buffer);
      if (it != buffer_remap_.end()) {
        new_alloc_buffers.push_back(it->second);
        needs_update = true;
      } else {
        new_alloc_buffers.push_back(buffer);
      }
    }

    auto new_block = CopyOnWrite(op);
    new_block->body = new_body;
    if (needs_update) {
      new_block->alloc_buffers = new_alloc_buffers;
    }
    return Stmt(new_block);
  }

  const std::vector<MultiVersionBufferInfo> &buffer_infos_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffer_remap_;
};

// Main function to rewrite alloc_buffers
Stmt RewriteAllocBuffers(
    const Stmt &stmt, const std::vector<MultiVersionBufferInfo> &buffer_infos) {
  if (buffer_infos.empty()) {
    return stmt;
  }

  AllocBufferRewriter rewriter(buffer_infos);
  return rewriter(stmt);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AutoSchedule", AutoSchedule);
}

} // namespace tl
} // namespace tvm
