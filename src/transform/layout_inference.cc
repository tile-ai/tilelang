// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file layout_inference.cc
 * \brief infer the fragment/shared memory layout
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include <queue>

#include "../op/parallel.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "common/loop_fusion_utils.h"
#include "loop_partition.h"
#include "loop_vectorize.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief collect the mapping from the buffer var to its allocate
 */
class ThreadBindingCollector : public StmtExprVisitor {
public:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      thread_binding_[iv->var.get()] = iv;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  // The thread binding map
  std::unordered_map<const VarNode *, IterVar> thread_binding_;
};

using namespace tir;
using arith::IRMutatorWithAnalyzer;

struct LayoutInferenceResult {
  Map<Buffer, Layout> layout_map;
  Map<For, Fragment> for_map;
  Map<For, PrimExpr> predicate_map;
};

class BufferUseDefCollector : public StmtExprVisitor {
public:
  BufferUseDefCollector(bool skip_thread_partition)
      : skip_thread_partition_(skip_thread_partition) {}

  LayoutInferenceResult Run() {
    // Basic consistency check: infer_list_ and thread_var_vec_ should have the
    // same size
    ICHECK_EQ(infer_list_.size(), thread_var_vec_.size())
        << "Size mismatch: infer_list_ and thread_var_vec_ must match in "
           "length.";

    // If needed, you can also check that annotated_layout_map_ is not empty, or
    // anything else relevant to your setup.

    // Copy the annotated layout map to local variable
    Map<Buffer, Layout> layout_map = annotated_layout_map_;
    int num_infer = infer_list_.size();

    // Prepare BFS queue for iterative inference
    std::queue<int> q;
    std::vector<bool> in_queue(num_infer, true);
    for (int i = 0; i < num_infer; i++) {
      // Check that each infer_list_ entry is valid
      ICHECK(infer_list_[i] != nullptr)
          << "infer_list_[" << i
          << "] is null. The inference object is not allocated properly.";

      // Check that each thread_var_vec_ entry is defined
      if (!thread_var_vec_[i].defined() && skip_thread_partition_) {
        // TODO(lei): This is a hack for cpu backend
        if (!thread_var_.defined()) {
          // Fake thread var to inference predicate for the buffer
          thread_var_ = IterVar(Range::FromMinExtent(PrimExpr(0), PrimExpr(1)),
                                Var(""), IterVarType::kDataPar);
        }
        thread_var_vec_[i] = thread_var_;
      }
      q.push(i);
    }
    auto run_infer_step = [&](int cur_infer_id, InferLevel level,
                              bool update_queue) {
      // Range check for cur_infer_id
      ICHECK_GE(cur_infer_id, 0)
          << "cur_infer_id is negative, which is invalid.";
      ICHECK_LT(cur_infer_id, num_infer)
          << "cur_infer_id " << cur_infer_id << " is out of range, must be < "
          << num_infer << ".";

      // Make sure we can safely access infer_list_[cur_infer_id] and
      // thread_var_vec_[cur_infer_id]
      auto &next = infer_list_[cur_infer_id];
      auto iter_var = thread_var_vec_[cur_infer_id];

      // Double-check that 'next' is valid
      ICHECK(next != nullptr) << "infer_list_[" << cur_infer_id
                              << "] is null inside run_infer_step.";

      // Check iter_var->dom and dom->extent
      ICHECK(iter_var.defined())
          << "thread_var_vec_[" << cur_infer_id << "] is not defined.";
      ICHECK(iter_var->dom.defined())
          << "iter_var->dom is not defined for infer_list_[" << cur_infer_id
          << "].";
      ICHECK(iter_var->dom->extent.defined())
          << "iter_var->dom->extent is not defined for infer_list_["
          << cur_infer_id << "].";

      const int64_t *extent_ptr = as_const_int(iter_var->dom->extent);
      ICHECK(extent_ptr != nullptr)
          << "iter_var->dom->extent is not a constant integer, which is "
             "required for layout inference.";

      // Run InferLayout
      auto updates = next->InferLayout(
          LayoutInferArgs{target_, static_cast<size_t>(*extent_ptr),
                          layout_map},
          level);
      // Process the returned updates
      for (const auto &[buffer, layout] : updates) {
        // Basic validity checks
        ICHECK(buffer.defined()) << "InferLayout returned an undefined buffer.";
        ICHECK(layout.defined()) << "InferLayout returned an undefined layout.";

        if (layout_map.count(buffer)) {
          // If already in map, ensure they are structurally equal
          ICHECK(StructuralEqual()(layout, layout_map[buffer]))
              << "Get different layout for " << buffer
              << " current layout: " << layout->DebugOutput()
              << " previous layout: " << layout_map[buffer]->DebugOutput();
        } else {
          // Otherwise, update map
          layout_map.Set(buffer, layout);
          if (!update_queue)
            continue;

          // Check if buffer exists in use_list_
          if (!use_list_.count(buffer)) {
            LOG(WARNING) << "Buffer " << buffer << " not found in use_list_. "
                         << "Potential mismatch between inference updates and "
                         << "use_list_.";
            continue;
          }

          // Push back into BFS queue
          for (int idx : use_list_[buffer]) {
            ICHECK_GE(idx, 0) << "Index in use_list_ for buffer " << buffer
                              << " is negative.";
            ICHECK_LT(idx, num_infer)
                << "Index in use_list_ for buffer " << buffer
                << " out of range: " << idx << " >= " << num_infer << ".";

            if (!in_queue[idx] && idx != cur_infer_id) {
              in_queue[idx] = true;
              q.push(idx);
            }
          }
        }
      }
    };

    auto finish_infer_queue = [&]() {
      while (!q.empty()) {
        int cur_infer_id = q.front();
        q.pop();
        // Range check again, just to be safe
        ICHECK_GE(cur_infer_id, 0);
        ICHECK_LT(cur_infer_id, num_infer);

        in_queue[cur_infer_id] = false;
        run_infer_step(cur_infer_id, InferLevel::kCommon, true);
      }
    };

    // step 1: infer strict layout
    for (int i = 0; i < num_infer; i++) {
      run_infer_step(i, InferLevel::kStrict, false);
    }

    // step 2: infer common layout with BFS
    finish_infer_queue();

    // step 3: relax constraints to free and re-run
    for (int i = 0; i < num_infer; i++) {
      run_infer_step(i, InferLevel::kFree, true);
      finish_infer_queue();
    }

    // Check that all local.fragment buffers have inferred layouts
    for (const auto &[buffer, _] : use_list_) {
      if (buffer.scope() == "local.fragment") {
        ICHECK_NE(layout_map.count(buffer), 0)
            << "The layout for fragment " << buffer
            << " can not be inferred correctly.";
      }
    }

    // Collect layout info for For nodes
    Map<For, Fragment> for_map;
    Map<For, PrimExpr> predicate_map;
    ICHECK(infer_list_.size() == thread_var_vec_.size())
        << "infer_list_ and thread_var_vec_ size mismatch";
    for (int i = 0; i < infer_list_.size(); i++) {
      std::unique_ptr<Operator> base_infer = std::move(infer_list_[i]);
      auto thread_var = thread_var_vec_[i];

      // Check if base_infer is valid
      ICHECK(base_infer != nullptr) << "Null pointer encountered in "
                                       "infer_list_ while collecting for_map.";

      if (auto for_infer = dynamic_cast<ParallelOp *>(base_infer.get())) {
        // Check that the loop layout is defined
        ICHECK(for_infer->GetLoopLayout().defined())
            << "The Layout for Parallel for cannot be inferred correctly:\n"
            << for_infer->GetRoot();
        for_map.Set(for_infer->GetRoot(), for_infer->GetLoopLayout());

        // thread_var_ should be defined if we rely on it
        ICHECK(thread_var.defined())
            << "thread_var is not defined. Cannot retrieve predicate.";

        if (auto predicate = for_infer->GetPredicate(thread_var->var)) {
          predicate_map.Set(for_infer->GetRoot(), predicate.value());
        }
      }
    }

    return {layout_map, for_map, predicate_map};
  }

  void Collect(const PrimFunc &f) {
    for (const auto &[_, buffer] : f->buffer_map) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "Layout_Inference: Require the target attribute";
    target_ = target.value();
    this->operator()(f->body);
  }

private:
  void VisitExpr_(const CallNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    // Do not analysis the call node to the global function.
    if (op->op.as<GlobalVarNode>())
      return;

    auto p = ParseOperator(GetRef<Call>(op), buffer_data_to_buffer_);
    if (p != nullptr) {
      for (const auto &arg : op->args) {
        if (auto buffer = getBufferFromAccessPtr(arg)) {
          addToUseList(buffer.value());
        }
      }
      infer_list_.push_back(std::move(p));
      thread_var_vec_.push_back(thread_var_);
    }
  }

  Optional<Buffer> getBufferFromAccessPtr(const PrimExpr &expr) {
    auto call = expr.as<CallNode>();
    if (call && call->op.same_as(builtin::tvm_access_ptr())) {
      auto var = call->args[1].as<Var>().value();
      return buffer_data_to_buffer_[var];
    }
    return NullOpt;
  }

  void addToUseList(const Buffer &buffer) {
    int infer_idx = infer_list_.size();
    if (use_list_.find(buffer) == use_list_.end()) {
      use_list_[buffer] = {};
    }
    use_list_[buffer].push_back(infer_idx);
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel) {
      auto infer = std::make_unique<ParallelOp>(GetRef<For>(op));
      for (const auto &[buffer, _] : infer->GetIndiceMap()) {
        addToUseList(buffer);
      }
      infer_list_.push_back(std::move(infer));
      thread_var_vec_.push_back(thread_var_);
    } else {
      StmtExprVisitor::VisitStmt(op->body);
    }
  }

  void VisitStmt_(const BlockNode *op) final {
    for (auto buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    if (op->annotations.count(attr::kLayoutMap)) {
      auto map =
          op->annotations.Get(attr::kLayoutMap).as<Map<Var, Layout>>().value();
      for (const auto &[var, layout] : map) {
        auto buffer = buffer_data_to_buffer_[var];
        ICHECK(StructuralEqual()(layout->InputShape(), buffer->shape));
        annotated_layout_map_.Set(buffer, layout);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_var_ = iv;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  std::vector<std::unique_ptr<Operator>> infer_list_;
  std::unordered_map<Buffer, std::vector<int>, ObjectPtrHash, ObjectPtrEqual>
      use_list_;
  IterVar thread_var_;
  std::vector<IterVar> thread_var_vec_;
  Target target_;
  LayoutMap annotated_layout_map_;
  bool skip_thread_partition_{false};
};

class LayoutInferencer : public IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, bool skip_thread_partition = false) {
    arith::Analyzer analyzer;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = ParallelLoopFuser::Fuse(f->body);
    BufferUseDefCollector collector(skip_thread_partition);
    collector.Collect(f);
    auto result = collector.Run();
    LayoutInferencer substituter(result, skip_thread_partition, &analyzer);
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  LayoutInferencer(const LayoutInferenceResult result,
                   bool skip_thread_partition, arith::Analyzer *analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer), result_(result),
        skip_thread_partition_(skip_thread_partition){};

  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = Downcast<Block>(IRMutatorWithAnalyzer::VisitStmt_(op));

    for (auto buffer : block->alloc_buffers) {
      if (buffer.scope() == "local.framgent") {
        ICHECK(result_.layout_map.count(buffer))
            << "Cannot inference fragment layout for " << buffer;
      }
    }
    auto block_ptr = block.CopyOnWrite();
    block_ptr->annotations.Set(attr::kLayoutMap, result_.layout_map);
    return block;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    For for_node = Downcast<For>(IRMutatorWithAnalyzer::VisitStmt_(op));
    if (result_.for_map.count(GetRef<For>(op))) {
      auto loop_layout = result_.for_map[GetRef<For>(op)];

      if (!skip_thread_partition_) {
        // If none thread bindings are provided, partition the loop
        for_node =
            PartitionLoop(for_node, thread_var_->var, analyzer_, loop_layout);
      }
      for_node = VectorizeLoop(for_node);
      if (result_.predicate_map.count(GetRef<For>(op))) {
        return IfThenElse(result_.predicate_map[GetRef<For>(op)], for_node);
      } else {
        return for_node;
      }
    }
    return for_node;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      ICHECK_NE(iv->thread_tag.length(), 0U);
      if (iv->thread_tag == "threadIdx.x") {
        thread_var_ = iv;
      }
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

private:
  const LayoutInferenceResult result_;
  IterVar thread_var_;
  bool skip_thread_partition_{false};
};

tvm::transform::Pass LayoutInference() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    ThreadBindingCollector collector;
    collector(f->body);
    bool has_thread_binding = collector.thread_binding_.size() > 0;
    bool skip_thread_partition = !has_thread_binding;
    return LayoutInferencer::Substitute(std::move(f), skip_thread_partition);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LayoutInference", {});
}

TVM_REGISTER_GLOBAL("tl.transform.LayoutInference")
    .set_body_typed(LayoutInference);

} // namespace tl
} // namespace tvm
