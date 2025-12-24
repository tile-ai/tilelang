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

#include <algorithm>
#include <deque>
#include <memory>
#include <queue>

#include "../layout/utils.h"
#include "../op/copy.h"
#include "../op/parallel.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "../target/utils.h"

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
                    std::deque<int> &q, std::vector<bool> &in_queue) {
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
    arith::Analyzer *cur_analyzer = analyzer_vec_[cur_infer_id].get();
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
    auto updates = next->InferLayout(LayoutInferArgs{target_,
                                                     thread_bounds,
                                                     layout_map,
                                                     cur_analyzer,
                                                     buffer_oob,
                                                     {},
                                                     let_var_to_expr_},
                                     level);

    // Process the returned updates
    for (const auto &[buffer, layout] : updates) {
      // Basic validity checks
      ICHECK(buffer.defined()) << "InferLayout returned an undefined buffer.";
      ICHECK(layout.defined()) << "InferLayout returned an undefined layout.";

      // Helper: propagate inferred layout to alias buffers (same data Var)
      auto propagate_alias = [&](const Buffer &src_buffer,
                                 const Layout &src_layout) {
        if (!buffer_data_to_buffers_.count(src_buffer->data))
          return;
        const auto &siblings = buffer_data_to_buffers_[src_buffer->data];
        for (const auto &sib : siblings) {
          if (sib.same_as(src_buffer))
            continue;
          bool shapes_equal =
              src_layout->InputShape().size() == sib->shape.size();
          if (shapes_equal) {
            for (size_t i = 0; i < src_layout->InputShape().size(); ++i) {
              if (!analyzer_.CanProveEqual(src_layout->InputShape()[i],
                                           sib->shape[i])) {
                shapes_equal = false;
                break;
              }
            }
          }
          Layout target_layout =
              shapes_equal
                  ? src_layout
                  : src_layout->Reshape(sib->shape, &analyzer_,
                                        Integer(src_buffer->dtype.bytes()),
                                        Integer(sib->dtype.bytes()));
          if (layout_map.count(sib)) {
            ICHECK(target_layout->IsEqual(layout_map[sib].get()))
                << "Get different layout for alias buffer " << sib
                << " (data-shared with " << src_buffer
                << ")\n current: " << target_layout->DebugOutput()
                << "\n previous: " << layout_map[sib]->DebugOutput();
          } else {
            layout_map.Set(sib, target_layout);
            if (update_queue && use_list_.count(sib)) {
              for (int idx : use_list_[sib]) {
                EnqueueWithPriority(idx, q, in_queue, cur_infer_id, layout_map);
              }
            }
          }
        }
      };

      if (layout_map.count(buffer)) {
        // If new layout contains the old one, update map
        if (IsFragmentBuffer(buffer) && level != InferLevel::kStrict &&
            !strict_layout_map.count(buffer)) {
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
            // Propagate to alias buffers as well
            propagate_alias(buffer, layout);
            continue;
          }
        }
        // If already in map, ensure they are structurally equal
        ICHECK(layout->IsEqual(layout_map[buffer].get()))
            << "Get different layout for " << buffer
            << "\n current layout: " << layout->DebugOutput()
            << "\n previous layout: " << layout_map[buffer]->DebugOutput();
        // Ensure aliases are consistent too
        propagate_alias(buffer, layout);
      } else {
        // Otherwise, update map
        layout_map.Set(buffer, layout);
        // Propagate to alias buffers (may enqueue their users)
        propagate_alias(buffer, layout);
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

          EnqueueWithPriority(idx, q, in_queue, cur_infer_id, layout_map);
        }
      }
    }
  };

  void FinishInferQueue(InferLevel level, LayoutMap &layout_map,
                        const LayoutMap &strict_layout_map, std::deque<int> &q,
                        std::vector<bool> &in_queue) {
    auto num_infer = infer_list_.size();

    while (!q.empty()) {
      int cur_infer_id = q.front();
      q.pop_front();
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
    ICHECK_EQ(analyzer_vec_.size(), infer_list_.size())
        << "Size mismatch: analyzer_vec_ and infer_list_ must match in "
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
    std::deque<int> q;
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
      q.push_back(i);
    }

    // step 0: set fully replicated layout for floating fragment buffers
    // Floating buffers are accessed outside TileOps (e.g., in if conditions),
    // so they must be replicated across all threads.
    for (const auto &[buffer, thread_bounds] : floating_fragment_buffers_) {
      if (layout_map.count(buffer))
        continue;
      auto frag =
          Fragment::FullyReplicated(buffer->shape, thread_bounds->extent);
      layout_map.Set(buffer, frag);
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

    // step 4: finalize alias layouts by Var
    // For each storage var, if any buffer in the group has a layout,
    // propagate (reshape if needed) to the rest to ensure completeness.
    for (const auto &[var, buffers] : buffer_data_to_buffers_) {
      // Find a representative with existing layout
      Optional<Buffer> rep;
      Optional<Layout> rep_layout;
      for (const auto &buf : buffers) {
        if (layout_map.count(buf)) {
          rep = buf;
          rep_layout = layout_map[buf];
          break;
        }
      }
      if (!rep_layout.defined())
        continue;
      for (const auto &buf : buffers) {
        if (!layout_map.count(buf)) {
          bool shapes_equal =
              rep_layout.value()->InputShape().size() == buf->shape.size();
          if (shapes_equal) {
            for (size_t i = 0; i < rep_layout.value()->InputShape().size();
                 ++i) {
              if (!analyzer_.CanProveEqual(rep_layout.value()->InputShape()[i],
                                           buf->shape[i])) {
                shapes_equal = false;
                break;
              }
            }
          }

          Layout reshaped = shapes_equal
                                ? rep_layout.value()
                                : rep_layout.value()->Reshape(
                                      buf->shape, &analyzer_,
                                      Integer(rep.value()->dtype.bytes()),
                                      Integer(buf->dtype.bytes()));
          layout_map.Set(buf, reshaped);
        }
      }
    }

    // Check that all local.fragment buffers have inferred layouts
    for (const auto &[buffer, _] : use_list_) {
      if (IsFragmentBuffer(buffer)) {
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
      if (buffer_data_to_buffers_.count(buffer->data)) {
        auto buffers = buffer_data_to_buffers_[buffer->data];
        buffers.push_back(buffer);
        buffer_data_to_buffers_.Set(buffer->data, buffers);
      } else {
        buffer_data_to_buffers_.Set(buffer->data, {buffer});
      }
    }
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "Layout_Inference: Require the target attribute";
    target_ = target.value();
    this->operator()(f->body);
    // Compute floating fragment buffers after collection
    ComputeFloatingFragmentBuffers(f->body);
  }

private:
  Map<Var, Buffer> GetBufferMap() const {
    Map<Var, Buffer> buffer_map;
    for (const auto &[var, buffers] : buffer_data_to_buffers_) {
      // Use the first buffer for each var
      // TODO(lei): phaseout buffer_map in future.
      if (!buffers.empty()) {
        buffer_map.Set(var, buffers[0]);
      }
    }
    return buffer_map;
  }

  // Return true if all buffers that this op (idx) touches already have
  // inferred layouts in layout_map. Used to prioritize enqueue order.
  bool ShouldPrioritize(int idx, const LayoutMap &layout_map) const {
    auto it = op_touched_buffers_.find(idx);
    if (it == op_touched_buffers_.end() || it->second.empty())
      return false;
    for (const auto &buf : it->second) {
      if (!layout_map.count(buf))
        return false;
    }
    return true;
  }

  // Enqueue idx to q with priority if all its buffers already
  // have layouts. Also guards against duplicates and self-enqueue.
  void EnqueueWithPriority(int idx, std::deque<int> &q,
                           std::vector<bool> &in_queue, int cur_infer_id,
                           const LayoutMap &layout_map) const {
    if (idx == cur_infer_id)
      return;
    if (idx < 0 || idx >= static_cast<int>(in_queue.size()))
      return;
    if (in_queue[idx])
      return;
    in_queue[idx] = true;
    if (ShouldPrioritize(idx, layout_map)) {
      q.push_front(idx);
    } else {
      q.push_back(idx);
    }
  }

  void VisitExpr_(const CallNode *op) final {
    IRVisitorWithAnalyzer::VisitExpr_(op);
    // Do not analysis the call node to the global function.
    if (op->op.as<GlobalVarNode>())
      return;

    auto p = ParseOperator(tvm::ffi::GetRef<Call>(op));
    if (p.defined()) {
      for (const auto &arg : op->args) {
        if (auto buffer = getBufferFromAccessPtr(arg)) {
          addToUseList(buffer.value());
        } else if (auto buffer = getBufferFromRegion(arg)) {
          addToUseList(buffer.value());
        }
        // Check if the argument uses any LetStmt variables that reference
        // fragment buffers. If so, add those buffers to the use list.
        // This handles cases like: a = block_mask_f[i]; T.copy(A[a, 0], ...)
        CollectFragmentBuffersFromExpr(arg);
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
      analyzer_vec_.push_back(analyzer_.Clone());

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
    if (auto bl = expr.as<BufferLoadNode>()) {
      return bl->buffer;
    }
    auto call = expr.as<CallNode>();
    if (!call) {
      return std::nullopt;
    }
    if (call->op.same_as(builtin::tvm_access_ptr())) {
      auto var_opt = call->args[1].as<Var>();
      if (!var_opt.has_value()) {
        LOG(WARNING) << "[getBufferFromAccessPtr] args[1] is not a Var, type: "
                     << call->args[1]->GetTypeKey();
        return std::nullopt;
      }
      const auto &var = var_opt.value();
      if (buffer_data_to_buffers_.count(var)) {
        const auto &buffers = buffer_data_to_buffers_[var];
        if (!buffers.empty()) {
          return buffers[0]; // Return the first buffer
        }
      }
      return std::nullopt;
    }
    return std::nullopt;
  }

  Optional<Buffer> getBufferFromRegion(const PrimExpr &expr) {
    if (auto call = expr.as<CallNode>()) {
      if (call->op.same_as(RegionOp::Get())) {
        if (auto bl = call->args[0].as<BufferLoadNode>()) {
          return bl->buffer;
        }
        return std::nullopt;
      }
    }
    return std::nullopt;
  }

  void addToUseList(const Buffer &buffer) {
    // buffer scope must be local.fragment
    if (!IsFragmentBuffer(buffer)) {
      return;
    }
    int infer_idx = infer_list_.size();
    if (use_list_.find(buffer) == use_list_.end()) {
      use_list_[buffer] = {};
    }
    use_list_[buffer].push_back(infer_idx);

    // Track which buffers this op (infer_idx) touches for prioritization.
    // Avoid duplicates.
    auto &vec = op_touched_buffers_[infer_idx];
    bool exists = false;
    for (const auto &b : vec) {
      if (b.same_as(buffer)) {
        exists = true;
        break;
      }
    }
    if (!exists)
      vec.push_back(buffer);
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel) {
      auto infer = ParallelOp(tvm::ffi::GetRef<For>(op));
      for (const auto &[buffer, _] : infer->GetIndiceMap()) {
        addToUseList(buffer);
      }

      PostOrderVisit(op->body, [this](const ObjectRef &node) {
        if (auto *buffer_load = node.as<BufferLoadNode>()) {
          if (buffer_load->buffer.defined() &&
              buffer_load->buffer->data.defined()) {
            if (buffer_data_to_buffers_.count(buffer_load->buffer->data)) {
              // Check if this buffer is already in the list
              auto buffers = buffer_data_to_buffers_[buffer_load->buffer->data];
              bool found = false;
              for (const auto &buf : buffers) {
                if (buf.same_as(buffer_load->buffer)) {
                  found = true;
                  break;
                }
              }
              if (!found) {
                buffers.push_back(buffer_load->buffer);
                buffer_data_to_buffers_.Set(buffer_load->buffer->data, buffers);
                DLOG(INFO) << "[LayoutInference] BufferStore: added buffer "
                           << buffer_load->buffer
                           << " buffer.get() = " << buffer_load->buffer.get()
                           << " data = " << buffer_load->buffer->data.get();
              }
            } else {
              buffer_data_to_buffers_.Set(buffer_load->buffer->data,
                                          {buffer_load->buffer});
              DLOG(INFO) << "[LayoutInference] BufferStore: new buffer "
                         << buffer_load->buffer
                         << " buffer.get() = " << buffer_load->buffer.get()
                         << " data = " << buffer_load->buffer->data.get();
            }
          }
        } else if (auto *buffer_store = node.as<BufferStoreNode>()) {
          if (buffer_store->buffer.defined() &&
              buffer_store->buffer->data.defined()) {
            if (buffer_data_to_buffers_.count(buffer_store->buffer->data)) {
              auto buffers =
                  buffer_data_to_buffers_[buffer_store->buffer->data];
              bool found = false;
              for (const auto &buf : buffers) {
                if (buf.same_as(buffer_store->buffer)) {
                  found = true;
                  break;
                }
              }
              if (!found) {
                buffers.push_back(buffer_store->buffer);
                buffer_data_to_buffers_.Set(buffer_store->buffer->data,
                                            buffers);
                DLOG(INFO) << "[LayoutInference] BufferStore: added buffer "
                           << buffer_store->buffer
                           << " buffer.get() = " << buffer_store->buffer.get()
                           << " data = " << buffer_store->buffer->data.get();
              }
            } else {
              buffer_data_to_buffers_.Set(buffer_store->buffer->data,
                                          {buffer_store->buffer});
              DLOG(INFO) << "[LayoutInference] BufferStore: new buffer "
                         << buffer_store->buffer
                         << " buffer.get() = " << buffer_store->buffer.get()
                         << " data = " << buffer_store->buffer->data.get();
            }
          }
        }
      });
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
      analyzer_vec_.push_back(analyzer_.Clone());
      buffer_oob_vec_.push_back(false);
    } else {
      IRVisitorWithAnalyzer::VisitStmt(op->body);
    }
  }

  void VisitStmt_(const BlockNode *op) final {
    for (auto buffer : op->alloc_buffers) {
      if (buffer_data_to_buffers_.count(buffer->data)) {
        auto buffers = buffer_data_to_buffers_[buffer->data];
        buffers.push_back(buffer);
        buffer_data_to_buffers_.Set(buffer->data, buffers);
      } else {
        buffer_data_to_buffers_.Set(buffer->data, {buffer});
      }
    }

    // First, visit the block body to collect all buffers from
    // BufferLoad/BufferStore
    IRVisitorWithAnalyzer::VisitStmt_(op);

    // After visiting, apply layouts to all collected buffers
    if (op->annotations.count(attr::kLayoutMap)) {
      // Check if the layout map is Map<Var, Layout>
      auto map =
          op->annotations.Get(attr::kLayoutMap)->as<Map<Var, Layout>>().value();
      for (const auto &[var, layout] : map) {
        ICHECK(buffer_data_to_buffers_.count(var))
            << "buffer " << var << " is not found in the block";
        const auto &buffers = buffer_data_to_buffers_[var];
        ICHECK(!buffers.empty()) << "buffer list for " << var << " is empty";
        // Apply layout to all buffers associated with this var
        for (const auto &buffer : buffers) {

          // Reshape the layout to match the buffer's shape
          // Check if shapes are structurally equal
          bool shapes_equal =
              layout->InputShape().size() == buffer->shape.size();
          if (shapes_equal) {
            for (size_t i = 0; i < layout->InputShape().size(); ++i) {
              if (!analyzer_.CanProveEqual(layout->InputShape()[i],
                                           buffer->shape[i])) {
                shapes_equal = false;
                break;
              }
            }
          }

          if (shapes_equal) {
            annotated_layout_map_.Set(buffer, layout);
          } else {
            // Use the first buffer sharing this var as the base for dtype ratio
            int base_bytes = buffers[0]->dtype.bytes();
            auto reshaped_layout =
                layout->Reshape(buffer->shape, &analyzer_, Integer(base_bytes),
                                Integer(buffer->dtype.bytes()));
            annotated_layout_map_.Set(buffer, reshaped_layout);
          }
        }
      }
    }
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

  void VisitStmt_(const LetStmtNode *op) final {
    // Record Let variable to its bound expression.
    // This enables tracking fragment buffer accesses through let bindings.
    let_var_to_expr_.Set(op->var, op->value);
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  // Helper: recursively collect fragment buffers from an expression,
  // following let bindings chain.
  void CollectFragmentBuffersFromExpr(const PrimExpr &expr) {
    PostOrderVisit(expr, [this](const ObjectRef &node) {
      if (auto bl = node.as<BufferLoadNode>()) {
        if (IsFragmentBuffer(bl->buffer)) {
          addToUseList(bl->buffer);
        }
      } else if (auto var_node = node.as<VarNode>()) {
        auto var = tvm::ffi::GetRef<Var>(var_node);
        if (let_var_to_expr_.count(var)) {
          CollectFragmentBuffersFromExpr(let_var_to_expr_[var]);
        }
      }
    });
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    // Collect buffer from BufferLoad
    if (op->buffer.defined() && op->buffer->data.defined()) {
      if (buffer_data_to_buffers_.count(op->buffer->data)) {
        // Check if this buffer is already in the list
        auto buffers = buffer_data_to_buffers_[op->buffer->data];
        bool found = false;
        for (const auto &buf : buffers) {
          if (buf.same_as(op->buffer)) {
            found = true;
            break;
          }
        }
        if (!found) {
          buffers.push_back(op->buffer);
          buffer_data_to_buffers_.Set(op->buffer->data, buffers);
          DLOG(INFO) << "[LayoutInference] BufferLoad: added buffer "
                     << op->buffer << " buffer.get() = " << op->buffer.get()
                     << " data = " << op->buffer->data.get();
        }
      } else {
        buffer_data_to_buffers_.Set(op->buffer->data, {op->buffer});
        DLOG(INFO) << "[LayoutInference] BufferLoad: new buffer " << op->buffer
                   << " buffer.get() = " << op->buffer.get()
                   << " data = " << op->buffer->data.get();
      }
    }
    IRVisitorWithAnalyzer::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    // Collect buffer from BufferStore
    if (op->buffer.defined() && op->buffer->data.defined()) {
      if (buffer_data_to_buffers_.count(op->buffer->data)) {
        // Check if this buffer is already in the list
        auto buffers = buffer_data_to_buffers_[op->buffer->data];
        bool found = false;
        for (const auto &buf : buffers) {
          if (buf.same_as(op->buffer)) {
            found = true;
            break;
          }
        }
        if (!found) {
          buffers.push_back(op->buffer);
          buffer_data_to_buffers_.Set(op->buffer->data, buffers);
          DLOG(INFO) << "[LayoutInference] BufferStore: added buffer "
                     << op->buffer << " buffer.get() = " << op->buffer.get()
                     << " data = " << op->buffer->data.get();
        }
      } else {
        buffer_data_to_buffers_.Set(op->buffer->data, {op->buffer});
        DLOG(INFO) << "[LayoutInference] BufferStore: new buffer " << op->buffer
                   << " buffer.get() = " << op->buffer.get()
                   << " data = " << op->buffer->data.get();
      }
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  // Compute floating fragment buffers after collection is done.
  //
  // A "floating" fragment buffer is one that has accesses outside of any
  // TileOp (Copy, Gemm, Reduce, Parallel, etc.). For example:
  //
  //   T.copy(BlockMask[by, :], block_mask_f)  // block_mask_f accessed IN
  //   TileOp for i in T.Pipelined(N_S):
  //       if block_mask_f[i] >= 0:           // block_mask_f accessed OUTSIDE
  //       TileOp (floating!)
  //           T.copy(A[...], A_shared)
  //
  // In this example, `block_mask_f[i]` in the if-condition is a "floating"
  // access because it's not inside any TileOp. Such buffers need special
  // handling: they must be fully replicated across all threads since the
  // access pattern cannot be inferred from TileOp semantics.
  //
  // This function identifies these buffers by:
  // 1. Collecting all IR nodes that are inside TileOps (from infer_list_stmt_)
  // 2. Scanning the entire function body for fragment buffer accesses
  // 3. Any access not inside a TileOp means the buffer is "floating"
  // 4. Recording the thread_bounds at the point of each floating access
  void ComputeFloatingFragmentBuffers(const Stmt &func_body) {
    // Step 1: Collect all nodes that are inside TileOps
    std::unordered_set<const Object *> nodes_in_tileops;
    for (const auto &stmt : infer_list_stmt_) {
      PostOrderVisit(stmt, [&](const ObjectRef &node) {
        nodes_in_tileops.insert(node.get());
      });
    }

    // Step 2: Use a visitor to scan for floating accesses while tracking thread
    // context
    class FloatingBufferCollector : public IRVisitorWithAnalyzer {
    public:
      FloatingBufferCollector(
          const std::unordered_set<const Object *> &nodes_in_tileops,
          std::unordered_map<Buffer, Range, ObjectPtrHash, ObjectPtrEqual>
              &floating_buffers)
          : nodes_in_tileops_(nodes_in_tileops),
            floating_buffers_(floating_buffers) {}

      void VisitStmt_(const AttrStmtNode *op) final {
        if (op->attr_key == tir::attr::thread_extent) {
          IterVar iv = Downcast<IterVar>(op->node);
          if (iv->thread_tag == "threadIdx.x") {
            thread_var_ = iv;
          }
        }
        IRVisitorWithAnalyzer::VisitStmt_(op);
      }

      void VisitExpr_(const BufferLoadNode *op) final {
        CheckFloatingAccess(op->buffer, op);
        IRVisitorWithAnalyzer::VisitExpr_(op);
      }

      void VisitStmt_(const BufferStoreNode *op) final {
        CheckFloatingAccess(op->buffer, op);
        IRVisitorWithAnalyzer::VisitStmt_(op);
      }

    private:
      void CheckFloatingAccess(const Buffer &buffer, const Object *node) {
        if (!IsFragmentBuffer(buffer))
          return;
        if (nodes_in_tileops_.find(node) != nodes_in_tileops_.end())
          return;
        // This is a floating access - record buffer with current thread_bounds
        if (floating_buffers_.find(buffer) != floating_buffers_.end())
          return; // Already recorded
        Range thread_bounds = Range::FromMinExtent(0, 1);
        if (thread_var_.defined() &&
            analyzer_.const_int_bound.IsBound(thread_var_->var)) {
          auto const_int_bound = analyzer_.const_int_bound(thread_var_);
          auto dtype = thread_var_->var.dtype();
          auto extent =
              const_int_bound->max_value - const_int_bound->min_value + 1;
          thread_bounds = Range::FromMinExtent(
              IntImm(dtype, const_int_bound->min_value), IntImm(dtype, extent));
        }
        floating_buffers_[buffer] = thread_bounds;
      }

      const std::unordered_set<const Object *> &nodes_in_tileops_;
      std::unordered_map<Buffer, Range, ObjectPtrHash, ObjectPtrEqual>
          &floating_buffers_;
      IterVar thread_var_;
    };

    FloatingBufferCollector collector(nodes_in_tileops,
                                      floating_fragment_buffers_);
    collector(func_body);

    // Debug log floating fragment buffers
    if (!floating_fragment_buffers_.empty()) {
      DLOG(INFO)
          << "Floating fragment buffers (have accesses outside TileOps):";
      for (const auto &[buffer, thread_bounds] : floating_fragment_buffers_) {
        DLOG(INFO) << "    " << buffer
                   << " with thread_bounds: " << thread_bounds;
      }
    }
  }

  Map<Var, Array<Buffer>> buffer_data_to_buffers_;
  // Map from LetStmt variable to its bound expression
  Map<Var, PrimExpr> let_var_to_expr_;
  std::vector<ObjectRef> infer_list_stmt_;
  std::vector<TileOperator> infer_list_;
  // Fragment buffers that have accesses outside of TileOps.
  // These "floating" buffers need fully replicated layouts since their
  // access patterns cannot be inferred from TileOp semantics.
  // Maps buffer -> thread_bounds at the point of floating access.
  // See ComputeFloatingFragmentBuffers() for detailed explanation.
  std::unordered_map<Buffer, Range, ObjectPtrHash, ObjectPtrEqual>
      floating_fragment_buffers_;
  std::unordered_map<Buffer, std::vector<int>, ObjectPtrHash, ObjectPtrEqual>
      use_list_;
  // Per-op list of buffers it touches (fragment scope), used for prioritization
  std::unordered_map<int, std::vector<Buffer>> op_touched_buffers_;
  // This is a workaround for cpu backend,
  // we need to define a thread_var for the serial loop.
  IterVar thread_var_ = IterVar(Range::FromMinExtent(0, 1), Var("v_thread"),
                                IterVarType::kDataPar);
  std::vector<IterVar> thread_var_vec_;
  std::vector<Range> thread_bounds_vec_;
  std::vector<std::unique_ptr<arith::Analyzer>> analyzer_vec_;
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

      // Union all infer_list_ indices that share the same Buffer object
      int first_idx = infer_indices[0];
      for (size_t i = 1; i < infer_indices.size(); i++) {
        uf.Union(first_idx, infer_indices[i]);
      }
    }
    // Additionally, union across buffers that share the same underlying
    // buffer->data (Var). This handles cases like reshape where multiple
    // Buffer objects alias the same storage.
    for (const auto &[var, buffers] : buffer_data_to_buffers_) {
      std::vector<int> merged;
      for (const auto &buf : buffers) {
        auto it = use_list_.find(buf);
        if (it != use_list_.end()) {
          const auto &vec = it->second;
          merged.insert(merged.end(), vec.begin(), vec.end());
        }
      }
      if (merged.size() > 1) {
        std::sort(merged.begin(), merged.end());
        merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
        int first = merged[0];
        for (size_t i = 1; i < merged.size(); ++i) {
          uf.Union(first, merged[i]);
        }
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
    std::deque<int> q;
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
                   << " members " << members.size() << '\n';
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
        } catch (const LoopLayoutInjectiveException &e) {
          do_update = false;
          DLOG(INFO) << "attempt failed due to LoopLayoutInjectiveException "
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
                ICHECK(pci != nullptr)
                    << "Can not use non-constant range to "
                       "iterate over a fragment/local "
                       "buffer. Non-constant shape expr is: "
                    << i
                    << ". This is possibly because you use symbolic shape when "
                       "accessing a fragment/local buffer.";
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
    if (!result_.for_map.count(tvm::ffi::GetRef<For>(op))) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    // the analyzer will be modified in PartitionLoop and VectorizeLoop
    // we need to save its state to prevent conflicted bindings
    auto saved_analyzer = analyzer_->Clone();
    For for_node = Downcast<For>(IRMutatorWithAnalyzer::VisitStmt_(op));
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
        if (IsLocalBuffer(store->buffer)) {
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
        if (!IsLocalBuffer(store->buffer)) {
          local_register_only = false;
        }
      } else if (const auto *load = obj.as<BufferLoadNode>()) {
        if (!IsLocalBuffer(load->buffer)) {
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
        if (!IsLocalBuffer(load->buffer) && !IsFragmentBuffer(load->buffer)) {
          has_non_local = true;
        }
      } else if (const auto *store = obj.as<BufferStoreNode>()) {
        String scope = store->buffer.scope();
        if (!IsLocalBuffer(store->buffer) && !IsFragmentBuffer(store->buffer)) {
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
      if (const auto *cast = obj.as<CastNode>()) {
        // Check if this is a non-reducer store with Cast operation
        DataType from_ty = cast->value.dtype();
        DataType target_ty = cast->dtype;
        if (IsCudaVectorizableCast(from_ty, target_ty) &&
            TargetIsCuda(Target::Current())) {
          has_cast_operations = true;
        }
      }
    });

    if ((has_non_local || has_cast_operations) && !has_reducer) {
      DLOG(INFO) << "Try to vectorize loop";
      for_node = VectorizeLoop(for_node, saved_analyzer.get());
    }

    if (result_.predicate_map.count(root) && parallel_loop) {
      return IfThenElse(result_.predicate_map[root], for_node);
    } else {
      return for_node;
    }
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
