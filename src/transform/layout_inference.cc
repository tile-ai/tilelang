/*!
 * \file layout_inference.cc
 * \brief infer the fragment/shared memory layout
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include <queue>

#include "../layout/utils.h"
#include "../op/copy.h"
#include "../op/parallel.h"
#include "../op/region.h"

#include "arith/ir_mutator_with_analyzer.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "common/loop_fusion_utils.h"
#include "common/loop_parallel_transform_utils.h"
#include "common/union_find.h"
#include "layout_reducer.h"
#include "loop_partition.h"
#include "loop_vectorize.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief collect the mapping from the buffer var to it allocated buffer
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
using arith::IRVisitorWithAnalyzer;

struct LayoutInferenceResult {
  Map<Buffer, Layout> layout_map;
  Map<For, Fragment> for_map;
  Map<For, PrimExpr> predicate_map;
};

class BufferUseDefCollector : public IRVisitorWithAnalyzer {
public:
  BufferUseDefCollector(bool skip_thread_partition)
      : skip_thread_partition_(skip_thread_partition) {}

  using arith::IRVisitorWithAnalyzer::IRVisitorWithAnalyzer;

  void RunInferStep(int cur_infer_id, InferLevel level, bool update_queue,
                    LayoutMap &layout_map, const LayoutMap &strict_layout_map,
                    std::queue<int> &q, std::vector<bool> &in_queue) {
    auto num_infer = infer_list_.size();

    // Range check for cur_infer_id
    ICHECK_GE(cur_infer_id, 0) << "cur_infer_id is negative, which is invalid.";
    ICHECK_LT(cur_infer_id, num_infer)
        << "cur_infer_id " << cur_infer_id << " is out of range, must be < "
        << num_infer << ".";

    // Make sure we can safely access infer_list_[cur_infer_id] and
    // thread_var_vec_[cur_infer_id]
    auto &next = infer_list_[cur_infer_id];
    auto iter_var = thread_var_vec_[cur_infer_id];
    auto thread_bounds = thread_bounds_vec_[cur_infer_id];
    auto buffer_oob = buffer_oob_vec_[cur_infer_id];
    // Double-check that 'next' is valid
    ICHECK(next.defined()) << "infer_list_[" << cur_infer_id
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
    DLOG(INFO) << "[RunInferStep] working on " << cur_infer_id << '\n';
    auto updates =
        next->InferLayout(LayoutInferArgs{target_, thread_bounds, layout_map,
                                          &analyzer_, buffer_oob},
                          level);
    // Process the returned updates
    for (const auto &[buffer, layout] : updates) {
      DLOG(INFO) << "    consider update " << buffer << " as "
                 << layout->DebugOutput() << '\n';

      // Basic validity checks
      ICHECK(buffer.defined()) << "InferLayout returned an undefined buffer.";
      ICHECK(layout.defined()) << "InferLayout returned an undefined layout.";

      if (layout_map.count(buffer)) {
        // If new layout contains the old one, update map
        if (buffer.scope() == "local.fragment" &&
            level != InferLevel::kStrict && !strict_layout_map.count(buffer)) {
          // Actually this test has been done in ParallelOp::InferLayout
          // already. Just do it again to avoid missing implementations in other
          // `TileOperator`s.

          auto dst_layout_opt = layout.as<Fragment>();
          ICHECK(dst_layout_opt.has_value())
              << "Failed to cast layout to Fragment for buffer " << buffer
              << ", layout type is " << layout->GetTypeKey();
          const auto &dst_layout = dst_layout_opt.value();
          auto src_layout_opt = layout_map[buffer].as<Fragment>();
          ICHECK(src_layout_opt.has_value())
              << "Failed to cast layout_map[buffer] to Fragment for buffer "
              << buffer << ", layout type is "
              << layout_map[buffer]->GetTypeKey();
          const auto &src_layout = src_layout_opt.value();
          ICHECK(dst_layout->InputDim() == src_layout->InputDim());
          Array<PrimExpr> indices;
          indices.reserve(dst_layout->InputDim());
          arith::Analyzer inner_analyzer;
          for (int i = 0; i < dst_layout->InputDim(); ++i) {
            auto x = InputPlaceholder(i);
            indices.push_back(x);
            // should be literal - literal = 0, any analyzer will work
            ICHECK(is_zero(inner_analyzer.Simplify(
                dst_layout->InputShape()[i] - src_layout->InputShape()[i])));
            inner_analyzer.Bind(x, Range(0, dst_layout->InputShape()[i]));
          }
          if (ProveFragmentContains(src_layout, dst_layout, indices, indices,
                                    inner_analyzer)) {
            layout_map.Set(buffer, layout);
            DLOG(INFO) << "    layout broadcast from "
                       << src_layout->DebugOutput() << ", accepted" << '\n';
            continue;
          }
        }
        // If already in map, ensure they are structurally equal
        ICHECK(layout->IsEqual(layout_map[buffer].get()))
            << "Get different layout for " << buffer
            << "\n current layout: " << layout->DebugOutput()
            << "\n previous layout: " << layout_map[buffer]->DebugOutput();
      } else {
        // Otherwise, update map
        layout_map.Set(buffer, layout);
        DLOG(INFO) << "    new layout accepted" << '\n';
        if (!update_queue)
          continue;

        // Check if buffer exists in use_list_
        if (!use_list_.count(buffer)) {
          LOG(WARNING) << "Layout inference failed for buffer " << buffer
                       << ". "
                       << "The buffer cannot be inferred with current layout "
                          "inference rules.";
          continue;
        }

        // Push back into BFS queue
        for (int idx : use_list_[buffer]) {
          ICHECK_GE(idx, 0)
              << "Index in use_list_ for buffer " << buffer << " is negative.";
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

  void FinishInferQueue(InferLevel level, LayoutMap &layout_map,
                        const LayoutMap &strict_layout_map, std::queue<int> &q,
                        std::vector<bool> &in_queue) {
    auto num_infer = infer_list_.size();
    while (!q.empty()) {
      int cur_infer_id = q.front();
      q.pop();
      // Range check again, just to be safe
      ICHECK_GE(cur_infer_id, 0);
      ICHECK_LT(cur_infer_id, num_infer);

      in_queue[cur_infer_id] = false;
      RunInferStep(cur_infer_id, level, true, layout_map, strict_layout_map, q,
                   in_queue);
    }
  };

  LayoutInferenceResult Run() {
    // Basic consistency check: infer_list_ and thread_var_vec_ should have the
    // same size
    ICHECK_EQ(infer_list_.size(), thread_var_vec_.size())
        << "Size mismatch: infer_list_ and thread_var_vec_ must match in "
           "length.";
    ICHECK_EQ(thread_bounds_vec_.size(), infer_list_.size())
        << "Size mismatch: thread_bounds_vec_ and infer_list_ must match in "
           "length.";
    ICHECK_EQ(buffer_oob_vec_.size(), infer_list_.size())
        << "Size mismatch: buffer_oob_vec_ and infer_list_ must match in "
           "length.";

    DLOG(INFO) << "[InferLayout] all participating operators:" << '\n';
    for (int i = 0; i < infer_list_stmt_.size(); ++i) {
      DLOG(INFO) << "    op " << i << ":" << infer_list_stmt_[i] << '\n';
    }

    // If needed, you can also check that annotated_layout_map_ is not empty, or
    // anything else relevant to your setup.

    // Copy the annotated layout map to local variable
    Map<Buffer, Layout> layout_map = annotated_layout_map_;
    Map<Buffer, Layout> strict_layout_map;
    int num_infer = infer_list_.size();

    // Prepare BFS queue for iterative inference
    std::queue<int> q;
    std::vector<bool> in_queue(num_infer, true);
    for (int i = 0; i < num_infer; i++) {
      // Check that each infer_list_ entry is valid
      ICHECK(infer_list_[i].defined())
          << "infer_list_[" << i
          << "] is null. The inference object is not allocated properly.";

      // Check that each thread_var_vec_ entry is defined
      if (!thread_var_vec_[i].defined() && skip_thread_partition_) {
        thread_var_vec_[i] = thread_var_;
      }
      q.push(i);
    }

    // step 1: infer strict layout
    for (int i = 0; i < num_infer; i++) {
      RunInferStep(i, InferLevel::kStrict, false, layout_map, strict_layout_map,
                   q, in_queue);
    }

    for (const auto &[buffer, layout] : layout_map) {
      strict_layout_map.Set(buffer, layout);
    }

    // step 2: infer common layout with BFS
    FinishInferQueue(InferLevel::kCommon, layout_map, strict_layout_map, q,
                     in_queue);

    // step 3: relax constraints to free and re-run
    InferInFreeMode(layout_map, strict_layout_map);

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
      TileOperator base_infer = std::move(infer_list_[i]);
      auto thread_var = thread_var_vec_[i];

      // Check if base_infer is valid
      ICHECK(base_infer.defined()) << "Null pointer encountered in "
                                      "infer_list_ while collecting for_map.";
      if (auto for_infer = base_infer.as<ParallelOpNode>()) {
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
    IRVisitorWithAnalyzer::VisitExpr_(op);
    // Do not analysis the call node to the global function.
    if (op->op.as<GlobalVarNode>())
      return;

    auto p = ParseOperator(tvm::ffi::GetRef<Call>(op), buffer_data_to_buffer_);
    if (p.defined()) {
      for (const auto &arg : op->args) {
        if (auto buffer = getBufferFromAccessPtr(arg)) {
          addToUseList(buffer.value());
        }
      }
      // Compute thread_var_ and thread_bounds_
      thread_var_vec_.push_back(thread_var_);
      if (analyzer_.const_int_bound.IsBound(thread_var_->var)) {
        auto const_int_bound = analyzer_.const_int_bound(thread_var_);
        auto min_value = const_int_bound->min_value;
        auto max_value = const_int_bound->max_value;
        auto extent = max_value - min_value + 1;
        auto dtype = thread_var_->var.dtype();
        thread_bounds_vec_.push_back(Range::FromMinExtent(
            IntImm(dtype, min_value), IntImm(dtype, extent)));
      } else {
        thread_bounds_vec_.push_back(Range::FromMinExtent(0, 1));
      }

      // Compute buffer oob for each buffer in the op
      if (const auto *copy = p.as<CopyNode>()) {
        auto src_tensor = copy->src;
        auto dst_tensor = copy->dst;
        auto src_range = copy->src_range;
        auto dst_range = copy->dst_range;
        bool src_oob = false;
        bool dst_oob = false;
        for (size_t i = 0; i < src_range.size(); i++) {
          if (!analyzer_.CanProve(src_range[i]->min + src_range[i]->extent <=
                                      src_tensor->shape[i],
                                  arith::ProofStrength::kSymbolicBound)) {
            src_oob = true;
            break;
          }
        }
        for (size_t i = 0; i < dst_range.size(); i++) {
          if (!analyzer_.CanProve(dst_range[i]->min + dst_range[i]->extent <=
                                      dst_tensor->shape[i],
                                  arith::ProofStrength::kSymbolicBound)) {
            dst_oob = true;
            break;
          }
        }
        buffer_oob_vec_.push_back(src_oob || dst_oob);
      } else {
        buffer_oob_vec_.push_back(false);
      }

      // Add the tile operator to infer_list_
      infer_list_stmt_.push_back(tvm::ffi::GetRef<ObjectRef>(op));
      infer_list_.push_back(std::move(p));
    }
  }

  Optional<Buffer> getBufferFromAccessPtr(const PrimExpr &expr) {
    auto call = expr.as<CallNode>();
    if (!call) {
      return std::nullopt;
    }
    if (call->op.same_as(builtin::tvm_access_ptr())) {
      auto var_opt = call->args[1].as<Var>();
      if (!var_opt.has_value()) {
        DLOG(WARNING) << "[getBufferFromAccessPtr] args[1] is not a Var, type: "
                      << call->args[1]->GetTypeKey();
        return std::nullopt;
      }
      const auto &var = var_opt.value();
      return buffer_data_to_buffer_[var];
    } else if (call->op.same_as(RegionOp::Get())) {
      return call->args[0].as<BufferLoadNode>()->buffer;
    }
    return std::nullopt;
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
      auto infer = ParallelOp(tvm::ffi::GetRef<For>(op));
      for (const auto &[buffer, _] : infer->GetIndiceMap()) {
        addToUseList(buffer);
      }
      infer_list_stmt_.push_back(tvm::ffi::GetRef<ObjectRef>(op));
      infer_list_.push_back(std::move(infer));
      thread_var_vec_.push_back(thread_var_);
      if (thread_var_.defined() &&
          analyzer_.const_int_bound.IsBound(thread_var_->var)) {
        auto const_int_bound = analyzer_.const_int_bound(thread_var_);
        auto dtype = thread_var_->var.dtype();
        auto extent =
            const_int_bound->max_value - const_int_bound->min_value + 1;
        thread_bounds_vec_.push_back(Range::FromMinExtent(
            IntImm(dtype, const_int_bound->min_value), IntImm(dtype, extent)));
      } else {
        thread_bounds_vec_.push_back(Range::FromMinExtent(0, 1));
      }
      buffer_oob_vec_.push_back(false);
    } else {
      IRVisitorWithAnalyzer::VisitStmt(op->body);
    }
  }

  void VisitStmt_(const BlockNode *op) final {
    for (auto buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    if (op->annotations.count(attr::kLayoutMap)) {
      // Check if the layout map is Map<Var, Layout>
      auto map =
          op->annotations.Get(attr::kLayoutMap)->as<Map<Var, Layout>>().value();
      for (const auto &[var, layout] : map) {
        ICHECK(buffer_data_to_buffer_.count(var))
            << "buffer " << var << " is not found in the block";
        auto buffer = buffer_data_to_buffer_[var];
        ICHECK(StructuralEqual()(layout->InputShape(), buffer->shape));
        annotated_layout_map_.Set(buffer, layout);
      }
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_var_ = iv;
      }
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  std::vector<ObjectRef> infer_list_stmt_;
  std::vector<TileOperator> infer_list_;
  std::unordered_map<Buffer, std::vector<int>, ObjectPtrHash, ObjectPtrEqual>
      use_list_;
  // This is a workaround for cpu backend,
  // we need to define a thread_var for the serial loop.
  IterVar thread_var_ = IterVar(Range::FromMinExtent(0, 1), Var("v_thread"),
                                IterVarType::kDataPar);
  std::vector<IterVar> thread_var_vec_;
  std::vector<Range> thread_bounds_vec_;
  std::vector<bool> buffer_oob_vec_;
  Target target_;
  LayoutMap annotated_layout_map_;
  bool skip_thread_partition_{false};

  std::vector<TileOperator> BackupInferList() {
    std::vector<TileOperator> back_infer_list;
    back_infer_list.reserve(infer_list_.size());
    for (auto &&p : infer_list_) {
      back_infer_list.push_back(p->Clone());
    }
    return back_infer_list;
  }

  void InferInFreeMode(LayoutMap &layout_map,
                       const LayoutMap &strict_layout_map) {

    DLOG(INFO) << "Enforced layout maps:" << '\n';
    for (auto &&[k, v] : layout_map) {
      DLOG(INFO) << "    " << k << ": " << v->DebugOutput() << '\n';
    }
    DLOG(INFO) << '\n';

    // Group operators into connected components
    UnionFind<int> uf;
    for (int i = 0; i < infer_list_.size(); i++) {
      uf.MakeSet(i);
    }
    for (const auto &[buffer, infer_indices] : use_list_) {
      if (infer_indices.empty())
        continue;

      // Union all infer_list_ indices that share the same buffer
      int first_idx = infer_indices[0];
      for (size_t i = 1; i < infer_indices.size(); i++) {
        uf.Union(first_idx, infer_indices[i]);
      }
    }
    std::unordered_map<int, std::vector<int>> components;
    for (int i = 0; i < infer_list_.size(); i++) {
      int root = uf.Find(i);
      components[root].push_back(i);
    }
    // Create a map from root to buffers
    std::unordered_map<int, std::vector<Buffer>> components_buffers;
    for (const auto &[buffer, infer_indices] : use_list_) {
      int root = uf.Find(infer_indices[0]);
      components_buffers[root].push_back(buffer);
    }
    // Keep components_buffers for debug purpose
    (void)components_buffers;

    // For each component, try each op as root, and determine the least
    // replicated one
    std::queue<int> q;
    std::vector<bool> in_queue(infer_list_.size(), false);

    for (auto &&[root, members] : components) {
      DLOG(INFO) << "======================= processing component " << root
                 << '\n';
      decltype(infer_list_) best_infer_list;
      LayoutMap best_layout_map;
      int64_t min_reg_num = INT64_MAX;
      int min_reg_num_infer_root = -1;

      // Try each member as the root of inference for this component
      for (int attempt_infer_root : members) {
        DLOG(INFO) << "----------------------- try root " << attempt_infer_root
                   << '\n';
        // Backup the current infer_list_ state
        auto back_infer_list = BackupInferList();
        // Copy the current layout_map for temporary use
        LayoutMap tmp_layout_map = layout_map;
        bool do_update = true;
        try {
          // Run inference starting from attempt_infer_root
          RunInferStep(attempt_infer_root, InferLevel::kFree, true,
                       tmp_layout_map, strict_layout_map, q, in_queue);
          FinishInferQueue(InferLevel::kFree, tmp_layout_map, strict_layout_map,
                           q, in_queue);

          // After the first search, run inference for all other members in
          // order
          for (int other_infer_root : members) {
            if (other_infer_root != attempt_infer_root) {
              RunInferStep(other_infer_root, InferLevel::kFree, true,
                           tmp_layout_map, strict_layout_map, q, in_queue);
              FinishInferQueue(InferLevel::kFree, tmp_layout_map,
                               strict_layout_map, q, in_queue);
            }
          }
        } catch (const LayoutConflictException &e) {
          do_update = false;
          DLOG(INFO) << "attempt failed due to LayoutConflictException "
                     << e.what() << '\n';
        } catch (const NormalizeIterException &e) {
          do_update = false;
          DLOG(INFO) << "attempt failed due to NormalizeIterException "
                     << e.what() << '\n';
        }

        if (do_update) {
          // Compute the total register number for this layout
          int64_t reg_num = 0;
          for (const auto &[buffer, layout] : tmp_layout_map) {
            if (auto frag = layout.as<Fragment>()) {
              int64_t frag_reg_num = 1;
              for (auto i : frag.value()->OutputShape()) {
                auto pci = as_const_int(i);
                ICHECK(pci != nullptr);
                frag_reg_num *= *pci;
              }
              reg_num += frag_reg_num;
            }
          }
          // Update the best plan if this one uses fewer registers
          if (reg_num < min_reg_num ||
              (reg_num == min_reg_num &&
               attempt_infer_root < min_reg_num_infer_root)) {
            best_infer_list =
                BackupInferList(); // Use backup to avoid moving out infer_list_
            best_layout_map = tmp_layout_map;
            min_reg_num = reg_num;
            min_reg_num_infer_root = attempt_infer_root;
          }
        }
        // Restore infer_list_ state for the next attempt
        infer_list_ = std::move(back_infer_list);
      }
      ICHECK(min_reg_num < INT64_MAX) << "no available layout found" << '\n';
      // Apply the best plan for this component
      infer_list_ = std::move(best_infer_list);
      layout_map = best_layout_map;
      DLOG(INFO) << "[InferInFreeMode] Final selection is attempt_infer_root = "
                 << min_reg_num_infer_root << '\n';
    }
  }
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
  LayoutInferencer(const LayoutInferenceResult &result,
                   bool skip_thread_partition, arith::Analyzer *analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer), result_(result),
        skip_thread_partition_(skip_thread_partition) {};

  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  /**
   * @brief Visit and mutate a Block node to attach inferred layout information.
   *
   * Converts the visited Block via the base visitor, asserts that every buffer
   * allocated with scope "local.framgent" has an inferred layout in
   * result_.layout_map, and attaches result_.layout_map to the Block's
   * annotations under attr::kLayoutMap.
   *
   * If any "local.framgent" buffer lacks an entry in result_.layout_map an
   * ICHECK will fail with the offending buffer printed.
   *
   * @return Stmt The (possibly modified) Block statement with the layout-map
   * annotation set.
   */
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

  /**
   * @brief Visit and transform For nodes according to inferred layout
   * information.
   *
   * If the For node is present in result_.for_map, this method applies
   * loop-level layout-driven transformations: it optionally partitions the loop
   * across the thread index, vectorizes the loop body, and wraps the loop with
   * a predicate if one was inferred for the loop root.
   *
   * Detailed behavior:
   * - Reads reducer information from the For node's attr::kReducerInfo
   * annotation (if present) to detect reduction targets.
   * - Detects register-local buffer stores (buffers with scope "local") in the
   *   original loop body; if only register-local stores are present the loop is
   *   treated as a register-local scenario and is not partitioned across
   * threads.
   * - Obtains the loop layout from result_.for_map[root] and, unless the loop
   * is register-local or skip_thread_partition_ is set, partitions the loop via
   *   PartitionLoop using thread_var_ and analyzer_.
   * - Scans the transformed loop body to determine whether it accesses any
   *   non-local buffers (scopes other than "local" or "local.fragment").
   * - Scans the transformed loop body to detect reducers (based on
   * reducer_info). If a reducer is present the loop is NOT vectorized
   * (reduction axes are excluded from vectorization as a conservative
   * workaround).
   * - If the loop has non-local accesses and no reducer, the loop is vectorized
   *   via VectorizeLoop.
   * - If a predicate exists in result_.predicate_map for the loop root and the
   *   loop was partitioned, the method returns an IfThenElse surrounding the
   *   (possibly partitioned/vectorized) loop with that predicate; otherwise it
   *   returns the transformed For.
   *
   * @return The possibly transformed For statement (or an IfThenElse wrapping
   * it)
   */
  Stmt VisitStmt_(const ForNode *op) final {
    Map<Var, ReducerInfo> reducer_info;
    if (op->annotations.count(attr::kReducerInfo))
      reducer_info = op->annotations.Get(attr::kReducerInfo)
                         ->as<Map<Var, ReducerInfo>>()
                         .value();

    For for_node = Downcast<For>(IRMutatorWithAnalyzer::VisitStmt_(op));
    if (result_.for_map.count(tvm::ffi::GetRef<For>(op))) {
      auto root = tvm::ffi::GetRef<For>(op);
      // This check is a workaround to support T.Parallel for local buffers.
      // For example:
      //   for i in T.Parallel(1024):
      //     A_local[i] = A_global[i]
      // Here, A_local is a register-local buffer held independently by each
      // thread, so explicit thread binding is not required.
      bool store_into_local = false;
      PostOrderVisit(root, [&](const ObjectRef &obj) {
        if (const auto *store = obj.as<BufferStoreNode>()) {
          if (store->buffer.scope() == "local") {
            store_into_local = true;
          }
          // if the case is like:
          // for i in T.Parallel(1024):
          //     A_local[i] = B_global[i]
          //     A_frag[i] = A_global[i]
          // exception will be raise in Parallel::LayoutInference
        }
      });
      // This check if for the loop that only manuplates "local" buffers,
      // for i in T.Parallel(1024):
      //     A_local[i] = B_local[i]
      // Though this might be illegal
      // We use PostOrderVisit to detect whether the loop only manuplates
      // "local" buffers, which indicates register usage and justifies skipping
      // thread binding.
      bool local_register_only = true;
      PostOrderVisit(root, [&](const ObjectRef &obj) {
        if (const auto *store = obj.as<BufferStoreNode>()) {
          if (store->buffer.scope() != "local") {
            local_register_only = false;
          }
        } else if (const auto *load = obj.as<BufferLoadNode>()) {
          if (load->buffer.scope() != "local") {
            local_register_only = false;
          }
        }
      });

      auto loop_layout = result_.for_map[root];
      // FIXME: tell in-Parallel and out-of-Parallel `local`s apart
      // NOTE(lei): a bit ugly, we should rethink about this part in future.
      bool parallel_loop =
          !skip_thread_partition_ && !local_register_only && !store_into_local;

      if (parallel_loop) {
        for_node =
            PartitionLoop(for_node, thread_var_->var, analyzer_, loop_layout);
      }
      // If none thread bindings are provided, partition the loop
      bool has_non_local = false;
      PostOrderVisit(for_node->body, [&](const ObjectRef &obj) {
        if (const auto *load = obj.as<BufferLoadNode>()) {
          String scope = load->buffer.scope();
          if (scope != "local" && scope != "local.fragment") {
            has_non_local = true;
          }
        } else if (const auto *store = obj.as<BufferStoreNode>()) {
          String scope = store->buffer.scope();
          if (scope != "local" && scope != "local.fragment") {
            has_non_local = true;
          }
        }
      });
      // Workaround: if reducer is presented, don't vectorize loop
      // Best solution should be isolate reduction axis out of vectorization
      bool has_reducer = false;
      PostOrderVisit(for_node->body, [&](const ObjectRef &obj) {
        if (!has_reducer)
          if (const auto *store = obj.as<BufferStoreNode>()) {
            has_reducer = reducer_info.count(store->buffer->data) != 0;
          }
      });

      // If a cast operation exists, vectorization may still be required
      bool has_cast_operations = false;
      PostOrderVisit(for_node->body, [&](const ObjectRef &obj) {
        if (const auto *store = obj.as<BufferStoreNode>()) {
          // Check if this is a non-reducer store with Cast operation
          if (store->value.as<CastNode>()) {
            has_cast_operations = true;
          }
        }
      });

      if ((has_non_local || has_cast_operations) && !has_reducer) {
        for_node = VectorizeLoop(for_node);
      }

      if (result_.predicate_map.count(root) && parallel_loop) {
        return IfThenElse(result_.predicate_map[root], for_node);
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
  IterVar thread_var_ = IterVar(Range::FromMinExtent(0, 1), Var("v_thread"),
                                IterVarType::kDataPar);
  bool skip_thread_partition_{false};
};

tvm::transform::Pass LayoutInference() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    f.CopyOnWrite()->body = ParallelLoopTransformer::Substitute(f->body);
    ThreadBindingCollector collector;
    collector(f->body);
    bool has_thread_binding = !collector.thread_binding_.empty();
    bool skip_thread_partition = !has_thread_binding;
    return LayoutInferencer::Substitute(std::move(f), skip_thread_partition);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LayoutInference", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LayoutInference", LayoutInference);
}

} // namespace tl
} // namespace tvm
