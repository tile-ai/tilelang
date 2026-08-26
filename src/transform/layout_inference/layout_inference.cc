/*!
 * \file layout_inference.cc
 * \brief infer the fragment/shared memory layout
 */

#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/ir/repr.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/utils.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/index_map.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <unordered_set>

#include "../../config.h"
#include "../../layout/layout.h"
#include "../../layout/utils.h"
#include "../../op/builtin.h"
#include "../../op/copy.h"
#include "../../op/parallel.h"
#include "../../op/reducer.h"
#include "../../op/utils.h"
#include "../../span_utils.h"
#include "../common/loop_fusion_utils.h"
#include "../common/pipeline_utils.h"
#include "../common/union_find.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "backend/common/target_utils.h"
#include "layout_cost_model.h"
#include "parallel_loop_layout_validator.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

int64_t GetElementStorageBits(DataType dtype) {
  // Layout aliasing must be reasoned about in logical storage bits per element,
  // not in bytes.  For sub-byte dtypes such as fp4, `dtype.bytes()` rounds up
  // to 1 and loses the "two fp4 values share one byte" relationship that
  // reinterpreting views rely on.
  return static_cast<int64_t>(dtype.bits()) * dtype.lanes();
}

bool ShapesEqual(const Array<PrimExpr> &lhs, const Array<PrimExpr> &rhs,
                 arith::Analyzer *analyzer) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!analyzer->CanProveEqual(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

Optional<Buffer> FindLayoutAnchorBuffer(const Array<Buffer> &buffers,
                                        const Layout &layout,
                                        arith::Analyzer *analyzer) {
  for (const auto &buffer : buffers) {
    if (ShapesEqual(layout->InputShape(), buffer->shape, analyzer)) {
      return buffer;
    }
  }
  return Optional<Buffer>();
}

// A fragment layout published into the global layout map must be a closed
// function of the layout placeholders (input indices + replication). Loop
// completion can legitimately produce layouts referencing an enclosing serial
// loop var (CompleteBufferFragment on `src[i, k]` bakes in `i`); such a
// layout is meaningful only inside that loop's scope and would misdirect
// every other consumer of the buffer. The placeholders are memoized
// singletons, so pointer identity is the right membership test.
bool FragmentReferencesForeignVars(const Fragment &fragment) {
  std::unordered_set<const VarNode *> allowed;
  allowed.insert(ReplicationPlaceholder().get());
  for (int i = 0; i < static_cast<int>(fragment->InputDim()); ++i) {
    allowed.insert(InputPlaceholder(i).get());
  }
  bool foreign = false;
  auto scan = [&](const PrimExpr &expr) {
    PostOrderVisit(expr, [&](const ObjectRef &obj) {
      if (const auto *var = obj.as<VarNode>()) {
        if (!allowed.count(var)) {
          foreign = true;
        }
      }
    });
  };
  for (const auto &expr : fragment->GetForwardIndex()) {
    scan(expr);
  }
  scan(fragment->GetForwardThread());
  return foreign;
}

// Commit-point form of the closed-layout invariant: open completions are
// still valid inside their own op's scope (LowerTileOp's local re-inference
// serves genuinely loop-local buffers), so such an entry is dropped rather
// than rejected — a later proposer in program order supplies the closed form.
bool IsOpenFragmentLayout(const Buffer &buffer, const Layout &layout) {
  if (!IsFragmentBuffer(buffer)) {
    return false;
  }
  auto fragment = layout.as<Fragment>();
  return fragment && FragmentReferencesForeignVars(fragment.value());
}

// ---------------------------------------------------------------------------
// Free-mode attempt scoring (layout RFC, design B)
// ---------------------------------------------------------------------------

} // namespace

using namespace tirx;
using arith::IRMutatorWithAnalyzer;
using arith::IRVisitorWithAnalyzer;

struct LayoutInferenceResult {
  Map<Buffer, Layout> layout_map;
  Map<For, Fragment> for_map;
  Map<For, PrimExpr> predicate_map;
  Map<For, Bool> padding_guard_map;
};

/*! \brief Everything the inference engine knows about reducer dst-steering,
 *  behind one interface: which finalize op owns which unconstrained
 *  destination (reservation), whether a commit into the global layout map
 *  respects that ownership, which finalizes to wake when an update-site nest
 *  solves (the reducer edge), and the seed layouts for the last-resort wide
 *  fallback attempt. The generic engine calls these entry points and stays
 *  otherwise reducer-blind; the structural gates live in
 *  FinalizeReducerV2OpNode (CanSteerDst / FallbackDstLayout), so reservation
 *  and the proposal can never drift apart.
 *
 *  Rationale: an unowned finalize dst must take its first layout only from
 *  its finalize — the proposal is the planner verdict computed early, and a
 *  consumer completing the buffer first is exactly how that verdict used to
 *  get bypassed (then billed after inference as a thread-indexed publish
 *  copy). Buffers whose finalize fails the structural gates are never
 *  reserved and keep the legacy first-completer behavior; annotated
 *  destinations are seeded into the layout map before any level runs and
 *  never reach the ownership check. */
class ReducerDstSteering {
public:
  /*! \brief Register every finalize that is a capable proposer: recorded
   *  update sites plus FinalizeReducerV2OpNode::CanSteerDst. Also builds the
   *  wake table for the reducer edge (see FinalizesAwaitingSite). */
  void Reserve(
      const std::vector<TileOperator> &infer_list,
      const std::function<std::vector<int>(const VarNode *)> &update_site_ops,
      const std::vector<Range> &thread_bounds_vec,
      const std::vector<std::unique_ptr<arith::Analyzer>> &analyzer_vec,
      const Map<Buffer, Layout> &annotated_layouts) {
    for (int i = 0; i < static_cast<int>(infer_list.size()); ++i) {
      const auto *finalize = infer_list[i].as<FinalizeReducerV2OpNode>();
      if (finalize == nullptr) {
        continue;
      }
      const Buffer &dst = finalize->dst;
      std::vector<int> site_ops =
          update_site_ops(finalize->reducer->data.get());
      if (site_ops.empty() || annotated_layouts.count(dst) ||
          !FinalizeReducerV2OpNode::CanSteerDst(finalize->reducer, dst,
                                                thread_bounds_vec[i],
                                                analyzer_vec[i].get())) {
        continue;
      }
      owners_[dst].insert(i);
      finalize_ops_.insert(i);
      for (int site_op : site_ops) {
        auto &wake = site_wakes_[site_op];
        if (std::find(wake.begin(), wake.end(), i) == wake.end()) {
          wake.push_back(i);
        }
      }
      DLOG(INFO) << "[ReducerDstSteering] buffer " << dst
                 << " reserved for finalize op " << i;
    }
  }

  /*! \brief First-assignment ownership check for the commit point. Strict
   *  deductions stay authoritative (e.g. constant indexing inside a parallel
   *  loop genuinely forces replication); common/free proposals must come
   *  from the owning finalize. Consumers frozen before the owner's proposal
   *  lands re-validate when the engine re-enqueues them. */
  bool AllowsCommit(const Buffer &buffer, int proposer,
                    InferLevel level) const {
    if (level == InferLevel::kStrict) {
      return true;
    }
    auto it = owners_.find(buffer);
    return it == owners_.end() || it->second.count(proposer);
  }

  /*! \brief The reducer edge, made real in the worklist graph: finalizes to
   *  wake when the update-site op `op_idx` has a solved loop layout. The
   *  dependency "update nest solved -> finalize can compute its verdict"
   *  flows through no buffer (the reducer carries no layout and solving a
   *  nest may commit nothing), so use_list_ notifications cannot deliver
   *  it. Waking is idempotent: finalize is silent once dst is owned. A
   *  completed attempt must never leave a reserved dst unset — absent
   *  buffers cost nothing under any cost model, so "nobody decided" would
   *  outbid every honest attempt. */
  const std::vector<int> &FinalizesAwaitingSite(int op_idx) const {
    static const std::vector<int> kEmpty;
    auto it = site_wakes_.find(op_idx);
    return it == site_wakes_.end() ? kEmpty : it->second;
  }

  /*! \brief Seeds for the wide fallback attempt: every reserved dst of the
   *  component pinned to the universally readable replicated layout. Empty
   *  for components without reservations, which therefore fall through to
   *  the ordinary "no available layout" failure unchanged. */
  std::vector<std::pair<Buffer, Fragment>>
  FallbackSeeds(const std::vector<int> &members,
                const std::vector<TileOperator> &infer_list,
                const std::vector<Range> &thread_bounds_vec) const {
    std::vector<std::pair<Buffer, Fragment>> seeds;
    for (int member : members) {
      if (finalize_ops_.count(member) == 0) {
        continue;
      }
      const auto *finalize = infer_list[member].as<FinalizeReducerV2OpNode>();
      ICHECK(finalize);
      seeds.emplace_back(finalize->dst,
                         FinalizeReducerV2OpNode::FallbackDstLayout(
                             finalize->dst, thread_bounds_vec[member]));
    }
    return seeds;
  }

private:
  std::unordered_map<Buffer, std::unordered_set<int>, ObjectPtrHash,
                     ObjectPtrEqual>
      owners_;
  std::unordered_set<int> finalize_ops_;
  std::unordered_map<int, std::vector<int>> site_wakes_;
};

class BufferUseDefCollector : public IRVisitorWithAnalyzer {
public:
  BufferUseDefCollector() = default;

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
    // thread_index_vec_[cur_infer_id]
    auto &next = infer_list_[cur_infer_id];
    auto thread_index = thread_index_vec_[cur_infer_id];
    auto thread_bounds = thread_bounds_vec_[cur_infer_id];
    arith::Analyzer *cur_analyzer = analyzer_vec_[cur_infer_id].get();
    // Double-check that 'next' is valid
    ICHECK(next.defined()) << "infer_list_[" << cur_infer_id
                           << "] is null inside run_infer_step.";

    // Check the logical thread index and the thread bounds extent.
    ICHECK(thread_index.defined())
        << "thread_index_vec_[" << cur_infer_id << "] is not defined.";
    ICHECK(thread_bounds.defined())
        << "thread_bounds_vec_[" << cur_infer_id << "] is not defined.";

    const int64_t *extent_ptr = as_const_int(thread_bounds->extent);
    ICHECK(extent_ptr != nullptr)
        << "thread_bounds->extent is not a constant integer, which is "
           "required for layout inference.";

    // dst-steering: hand a FinalizeReducerV2Op the current state of its
    // reducer's update sites so it can propose a layout for an unconstrained
    // dst (see FinalizeReducerV2OpNode::InferLayout). Free mode swaps cloned
    // op objects in and out of infer_list_, so the sites resolve their
    // ParallelOps through infer_list_ at call time, never through cached
    // node pointers.
    std::vector<ReducerUpdateSiteHint> reducer_sites;
    const std::vector<ReducerUpdateSiteHint> *reducer_sites_ptr = nullptr;
    if (const auto *finalize = next.as<FinalizeReducerV2OpNode>()) {
      auto site_it = reducer_update_sites_.find(finalize->reducer->data.get());
      if (site_it != reducer_update_sites_.end()) {
        for (const auto &record : site_it->second) {
          const auto *parallel_op =
              infer_list_[record.infer_idx].as<ParallelOpNode>();
          ICHECK(parallel_op) << "reducer_update site infer_list_["
                              << record.infer_idx << "] is not a ParallelOp";
          ReducerUpdateSiteHint hint;
          hint.loop_layout = parallel_op->GetLoopLayout(); // may be unsolved
          for (const auto &iv : parallel_op->GetLoopVars()) {
            hint.loop_vars.push_back(iv->var);
          }
          hint.indices = record.indices;
          hint.value = record.value;
          reducer_sites.push_back(std::move(hint));
        }
        reducer_sites_ptr = &reducer_sites;
      }
    }

    // Run InferLayout
    LayoutMap updates;
    try {
      updates = next->InferLayout(LayoutInferArgs{target_,
                                                  thread_bounds,
                                                  layout_map,
                                                  cur_analyzer,
                                                  {},
                                                  bind_var_to_expr_,
                                                  false,
                                                  reducer_sites_ptr},
                                  level);
    } catch (const std::bad_optional_access &e) {
      LOG(FATAL) << "bad_optional_access while inferring layout for op "
                 << cur_infer_id << " (" << next->GetTypeKey() << ") at level "
                 << InferLevelToString(level)
                 << "\nthread_bounds=" << thread_bounds
                 << "\nstmt=" << infer_list_stmt_[cur_infer_id];
    }

    // Process the returned updates
    for (const auto &[buffer, layout] : updates) {
      // Basic validity checks
      ICHECK(buffer.defined()) << "InferLayout returned an undefined buffer.";
      ICHECK(layout.defined()) << "InferLayout returned an undefined layout.";

      // Gate 1 of the global map: only closed layouts may enter (see
      // IsOpenFragmentLayout).
      if (IsOpenFragmentLayout(buffer, layout)) {
        DLOG(INFO) << "[RunInferStep] dropping open layout for buffer "
                   << buffer << " from op " << cur_infer_id;
        continue;
      }

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
                  // Alias buffers may reinterpret the same storage with a
                  // different element width.  Reshape the inferred layout using
                  // the old/new storage bit ratio so that layout inference
                  // keeps the physical storage footprint unchanged while
                  // allowing the logical element count to change.
                  : src_layout->Reshape(
                        sib->shape, &analyzer_,
                        Integer(GetElementStorageBits(src_buffer->dtype)),
                        Integer(GetElementStorageBits(sib->dtype)));
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

        // If already in map, check if they are structurally equal
        if (!layout->IsEqual(layout_map[buffer].get())) {
          // Try to merge swizzle layouts if both are swizzle layouts
          const Layout &existing = layout_map[buffer];
          if (!layout.as<Fragment>() && !existing.as<Fragment>()) {
            if (auto merged = MergeSwizzleLayouts(existing, layout, buffer)) {
              DLOG(WARNING) << "Swizzle layout conflict for buffer " << buffer
                            << ", merging to smaller granularity";
              layout_map.Set(buffer, merged.value());
              propagate_alias(buffer, merged.value());
              continue;
            }
          }
          // If not swizzle layouts or merge failed, raise error
          LOG(FATAL) << "Get different layout for " << buffer
                     << "\n current layout: " << layout->DebugOutput()
                     << "\n previous layout: "
                     << layout_map[buffer]->DebugOutput()
                     << SpanHintSuffix(buffer->span);
        }
        // Ensure aliases are consistent too
        propagate_alias(buffer, layout);
      } else {
        // Gate 2 of the global map: a reserved finalize destination takes
        // its first layout only from its owning finalize (see
        // ReducerDstSteering::AllowsCommit).
        if (!steering_.AllowsCommit(buffer, cur_infer_id, level)) {
          DLOG(INFO) << "[RunInferStep] dropping layout for reserved "
                     << "finalize dst " << buffer << " from op "
                     << cur_infer_id;
          continue;
        }
        // Otherwise, update map
        layout_map.Set(buffer, layout);
        // Propagate to alias buffers (may enqueue their users)
        propagate_alias(buffer, layout);
        if (!update_queue)
          continue;

        // Check if buffer exists in use_list_
        if (!use_list_.count(buffer) && IsFragmentBuffer(buffer)) {
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

    // The reducer edge (see ReducerDstSteering::FinalizesAwaitingSite):
    // solving an update-site nest is the event its finalize waits for, but
    // it flows through no buffer, so the use_list_ notifications above
    // cannot deliver it. Proposals only happen at kFree, so only wake there.
    if (update_queue && level == InferLevel::kFree) {
      const std::vector<int> &wake =
          steering_.FinalizesAwaitingSite(cur_infer_id);
      if (!wake.empty()) {
        const auto *parallel_op = next.as<ParallelOpNode>();
        if (parallel_op && parallel_op->GetLoopLayout().defined()) {
          for (int finalize_idx : wake) {
            EnqueueWithPriority(finalize_idx, q, in_queue, cur_infer_id,
                                layout_map);
          }
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
    // Basic consistency check: infer_list_ and thread_index_vec_ should have
    // the same size
    ICHECK_EQ(infer_list_.size(), thread_index_vec_.size())
        << "Size mismatch: infer_list_ and thread_index_vec_ must match in "
           "length.";
    ICHECK_EQ(thread_bounds_vec_.size(), infer_list_.size())
        << "Size mismatch: thread_bounds_vec_ and infer_list_ must match in "
           "length.";
    ICHECK_EQ(analyzer_vec_.size(), infer_list_.size())
        << "Size mismatch: analyzer_vec_ and infer_list_ must match in "
           "length.";
    DLOG(INFO) << "[InferLayout] all participating operators:" << '\n';
    for (int i = 0; i < infer_list_stmt_.size(); ++i) {
      DLOG(INFO) << "    op " << i << ":" << infer_list_stmt_[i] << '\n';
    }

    steering_.Reserve(
        infer_list_,
        [this](const VarNode *reducer_var) {
          std::vector<int> site_ops;
          auto it = reducer_update_sites_.find(reducer_var);
          if (it != reducer_update_sites_.end()) {
            site_ops.reserve(it->second.size());
            for (const auto &record : it->second) {
              site_ops.push_back(record.infer_idx);
            }
          }
          return site_ops;
        },
        thread_bounds_vec_, analyzer_vec_, annotated_layout_map_);

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
      q.push_back(i);
    }

    // step 0: set fully replicated layout for floating fragment buffers
    // Floating buffers are accessed outside TileOps (e.g., in if conditions),
    // so they must be replicated across all threads.
    for (const auto &[buffer, thread_bounds] : floating_fragment_buffers_) {
      if (layout_map.count(buffer))
        continue;
      auto frag =
          Fragment::FullyReplicated(buffer->shape, thread_bounds->extent)
              ->BindThreadRange(thread_bounds);
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
          Layout reshaped =
              shapes_equal
                  ? rep_layout.value()
                  : rep_layout.value()->Reshape(
                        buf->shape, &analyzer_,
                        Integer(GetElementStorageBits(rep.value()->dtype)),
                        Integer(GetElementStorageBits(buf->dtype)));
          layout_map.Set(buf, reshaped);
        }
      }
    }

    // Check that all local.fragment buffers have inferred layouts
    for (const auto &[buffer, _] : use_list_) {
      if (IsFragmentBuffer(buffer)) {
        ICHECK_NE(layout_map.count(buffer), 0)
            << "The layout for fragment " << buffer
            << " can not be inferred correctly."
            << SpanHintSuffix(buffer->span);
      }
    }

    // Collect layout info for For nodes
    Map<For, Fragment> for_map;
    Map<For, PrimExpr> predicate_map;
    Map<For, Bool> padding_guard_map;
    ICHECK(infer_list_.size() == thread_index_vec_.size())
        << "infer_list_ and thread_index_vec_ size mismatch";
    for (int i = 0; i < infer_list_.size(); i++) {
      TileOperator base_infer = std::move(infer_list_[i]);
      auto thread_index = thread_index_vec_[i];

      // Check if base_infer is valid
      ICHECK(base_infer.defined()) << "Null pointer encountered in "
                                      "infer_list_ while collecting for_map.";
      if (auto for_infer = base_infer.as<ParallelOpNode>()) {
        // Check that the loop layout is defined
        ICHECK(for_infer->GetLoopLayout().defined())
            << "The Layout for Parallel for cannot be inferred correctly:\n"
            << for_infer->GetRoot();
        for_map.Set(for_infer->GetRoot(), for_infer->GetLoopLayout());
        if (for_infer->LoopLayoutRequiresPaddingGuard()) {
          padding_guard_map.Set(for_infer->GetRoot(), Bool(true));
        }
        // thread_index should be defined if we rely on it
        ICHECK(thread_index.defined())
            << "thread_index is not defined. Cannot retrieve predicate.";

        if (auto predicate = for_infer->GetPredicate(thread_index)) {
          predicate_map.Set(for_infer->GetRoot(), predicate.value());
        }
      }
    }

    return {layout_map, for_map, predicate_map, padding_guard_map};
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

  // Return true if any buffer that this op (idx) touches already has
  // an inferred layout in layout_map. Used to prioritize enqueue order.
  bool HasKnownLayoutAnchor(int idx, const LayoutMap &layout_map) const {
    auto it = op_touched_buffers_.find(idx);
    if (it == op_touched_buffers_.end() || it->second.empty())
      return false;
    for (const auto &buf : it->second) {
      if (layout_map.count(buf))
        return true;
    }
    return false;
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
    if (HasKnownLayoutAnchor(idx, layout_map)) {
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

    TileOperator p;
    try {
      p = ParseOperator(GetRef<Call>(op));
    } catch (const std::bad_optional_access &e) {
      LOG(FATAL) << "bad_optional_access while parsing tile op call: "
                 << GetRef<Call>(op);
    }
    if (p.defined()) {
      for (const auto &arg : op->args) {
        if (auto buffer = getBufferFromAccessPtr(arg)) {
          addToUseList(buffer.value());
        } else if (auto buffer = getBufferFromRegion(arg)) {
          addToUseList(buffer.value());
        }
        // Check if the argument uses any Bind variables that reference
        // fragment buffers. If so, add those buffers to the use list.
        // This handles cases like: a = block_mask_f[i]; T.copy(A[a, 0], ...)
        CollectFragmentBuffersFromExpr(arg);
      }
      // Compute thread_index and thread_bounds
      thread_index_vec_.push_back(CurrentThreadIndex());
      thread_bounds_vec_.push_back(CurrentThreadBounds());
      analyzer_vec_.push_back(analyzer_.Clone());

      // Add the tile operator to infer_list_
      infer_list_stmt_.push_back(GetRef<ObjectRef>(op));
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
      if (call->op.same_as(region())) {
        if (auto bl = call->args[0].as<BufferLoadNode>()) {
          return bl->buffer;
        }
        return std::nullopt;
      }
    }
    return std::nullopt;
  }

  void addToUseList(const Buffer &buffer) {
    if (IsReducerV2Buffer(buffer)) {
      // No layout of its own, but it links the epoch's ops (update nests,
      // init, finalize) into one free-mode component.
      reducer_use_list_[buffer->data.get()].push_back(infer_list_.size());
      return;
    }
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
    if (std::none_of(vec.begin(), vec.end(),
                     [&](const Buffer &b) { return b.same_as(buffer); })) {
      vec.push_back(buffer);
    }
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel) {
      auto infer = ParallelOp(GetRef<For>(op));
      for (const auto &buffer : infer->GetAccessOrder()) {
        addToUseList(buffer);
      }

      // This nest becomes infer_list_[op_idx] (pushed below).
      int op_idx = static_cast<int>(infer_list_.size());
      PostOrderVisit(op->body, [this, op_idx](const ObjectRef &node) {
        if (auto *call = node.as<CallNode>()) {
          if (call->op.same_as(reducer_update())) {
            // Record the update site for finalize's dst-steering hints.
            ReducerUpdateArgs update = ParseReducerUpdate(call);
            reducer_update_sites_[update.reducer->data.get()].push_back(
                ReducerUpdateSiteRecord{op_idx, update.indices, update.value});
            reducer_use_list_[update.reducer->data.get()].push_back(op_idx);
          }
        }
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
      infer_list_stmt_.push_back(GetRef<ObjectRef>(op));
      infer_list_.push_back(std::move(infer));
      thread_index_vec_.push_back(CurrentThreadIndex());
      thread_bounds_vec_.push_back(CurrentThreadBounds());
      analyzer_vec_.push_back(analyzer_.Clone());
    } else {
      IRVisitorWithAnalyzer::VisitStmt(op->body);
    }
  }

  void VisitStmt_(const SBlockNode *op) final {
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
        Optional<Buffer> anchor_buffer =
            FindLayoutAnchorBuffer(buffers, layout, &analyzer_);
        int64_t anchor_bits =
            anchor_buffer.defined()
                ? GetElementStorageBits(anchor_buffer.value()->dtype)
                : GetElementStorageBits(buffers[0]->dtype);
        // Apply layout to all buffers associated with this var
        for (const auto &buffer : buffers) {

          // Reshape the layout to match the buffer's shape
          // Check if shapes are structurally equal
          bool shapes_equal =
              ShapesEqual(layout->InputShape(), buffer->shape, &analyzer_);

          Layout resolved_layout =
              shapes_equal
                  ? layout
                  : layout->Reshape(
                        buffer->shape, &analyzer_, Integer(anchor_bits),
                        Integer(GetElementStorageBits(buffer->dtype)));
          if (IsSharedBuffer(buffer)) {
            arith::IterMapResult injectivity =
                resolved_layout->DetectInjective();
            if (!injectivity->errors.empty()) {
              TVM_FFI_THROW(ValueError)
                  << "Invalid layout for shared buffer `" << buffer->name
                  << "`: the forward map must be injective. Details: "
                  << injectivity->errors
                  << ". Layout: " << resolved_layout->DebugOutput()
                  << ". Check the layout passed to T.annotate_layout.";
            }
          }
          annotated_layout_map_.Set(buffer, resolved_layout);
        }
      }
    }
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_binding_ = iv;
      }
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  void VisitStmt_(const BindNode *op) final {
    // Record Bind variable to its bound expression.
    // This enables tracking fragment buffer accesses through Bind values.
    bind_var_to_expr_.Set(op->var, op->value);
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  // Helper: recursively collect fragment buffers from an expression,
  // following Bind value chains.
  void CollectFragmentBuffersFromExpr(const PrimExpr &expr) {
    PostOrderVisit(expr, [this](const ObjectRef &node) {
      if (auto bl = node.as<BufferLoadNode>()) {
        if (IsFragmentBuffer(bl->buffer)) {
          addToUseList(bl->buffer);
        }
      } else if (auto var_node = node.as<VarNode>()) {
        auto var = GetRef<Var>(var_node);
        if (bind_var_to_expr_.count(var)) {
          CollectFragmentBuffersFromExpr(bind_var_to_expr_[var]);
        }
      }
    });
  }

  Range CurrentThreadBounds() const {
    return ComputeThreadBounds(thread_binding_, analyzer_);
  }

  // Logical thread index for the current collection point: the real
  // threadIdx.x Var when a thread_extent binding exists, otherwise constant
  // 0 (e.g. CPU serial launch). Never an unbound synthetic Var.
  PrimExpr CurrentThreadIndex() const {
    if (thread_binding_.defined()) {
      return thread_binding_->var;
    }
    return IntImm(DataType::Int(32), 0);
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
        if (op->attr_key == tirx::attr::thread_extent) {
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
        floating_buffers_[buffer] = CurrentThreadBounds();
      }

      Range CurrentThreadBounds() const {
        return ComputeThreadBounds(thread_var_, analyzer_);
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
  // Map from Bind variable to its bound expression
  Map<Var, PrimExpr> bind_var_to_expr_;
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
  // One reducer_update site inside a parallel nest: the infer_list_ index of
  // the enclosing ParallelOp plus the update's logical indices and
  // contribution expression. Assembled into ReducerUpdateSiteHints — reading
  // the CURRENT op object's loop layout through infer_list_ — every time a
  // FinalizeReducerV2Op runs InferLayout, so finalize can steer an
  // unconstrained dst toward the reduction's natural placement.
  struct ReducerUpdateSiteRecord {
    int infer_idx;
    Array<PrimExpr> indices;
    PrimExpr value;
  };
  // reducer data Var -> its update sites, in program order.
  std::unordered_map<const VarNode *, std::vector<ReducerUpdateSiteRecord>>
      reducer_update_sites_;
  // reducer data Var -> infer_list_ indices of the ops touching the reducer
  // (update nests, init, finalize). Reducer buffers carry no layout and stay
  // out of use_list_, so free-mode component grouping must union through
  // them explicitly: finalize's dst proposal has to compete in the same
  // attempt search as the update loop's layout choice.
  std::unordered_map<const VarNode *, std::vector<int>> reducer_use_list_;
  // All reducer dst-steering knowledge, behind one interface (see the class
  // comment). Populated once in Run(), consulted at the commit point and by
  // the free-mode attempt runner.
  ReducerDstSteering steering_;
  // Per-op list of buffers it touches (fragment scope), used for prioritization
  std::unordered_map<int, std::vector<Buffer>> op_touched_buffers_;
  // Real threadIdx.x binding of the enclosing thread_extent scope, when one
  // exists. Stays undefined for targets without thread bindings (e.g. CPU),
  // where the logical thread index is the constant 0 and thread bounds are
  // [0, 1) — no synthetic fallback Var is ever created.
  IterVar thread_binding_;
  std::vector<PrimExpr> thread_index_vec_;
  std::vector<Range> thread_bounds_vec_;
  std::vector<std::unique_ptr<arith::Analyzer>> analyzer_vec_;
  Target target_;
  LayoutMap annotated_layout_map_;

  std::vector<TileOperator> BackupInferList() {
    std::vector<TileOperator> back_infer_list;
    back_infer_list.reserve(infer_list_.size());
    for (auto &&p : infer_list_) {
      back_infer_list.push_back(p->Clone());
    }
    return back_infer_list;
  }

  // One complete free-mode attempt over `members`: seed pre-owned layouts
  // (used by the wide fallback), solve `attempt_root` first, propagate
  // breadth-first, then run the remaining members in program order. A
  // finalize that runs before its update nests are solved stays silent; the
  // reducer-edge wake in RunInferStep re-enqueues it the moment a site nest
  // solves (see ReducerDstSteering::FinalizesAwaitingSite), so every
  // completed attempt carries the verdict layout on its reserved dsts.
  // Returns the scored snapshot, or nullopt when the attempt dies on a
  // layout conflict. infer_list_ is restored to its entry state either way.
  struct AttemptOutcome {
    std::vector<TileOperator> infer_list;
    LayoutMap layout_map;
    AttemptCost cost;
  };
  std::optional<AttemptOutcome>
  RunOneAttempt(int attempt_root, const std::vector<int> &members,
                const LayoutMap &base_layout_map,
                const LayoutMap &strict_layout_map,
                const std::vector<std::pair<Buffer, Fragment>> &seed_layouts,
                const LayoutCostModel &cost_model, std::deque<int> &q,
                std::vector<bool> &in_queue) {
    auto back_infer_list = BackupInferList();
    LayoutMap tmp_layout_map = base_layout_map;
    for (const auto &[buffer, fragment] : seed_layouts) {
      if (!tmp_layout_map.count(buffer)) {
        tmp_layout_map.Set(buffer, fragment);
      }
    }
    bool ok = true;
    std::string failure;
    try {
      RunInferStep(attempt_root, InferLevel::kFree, true, tmp_layout_map,
                   strict_layout_map, q, in_queue);
      FinishInferQueue(InferLevel::kFree, tmp_layout_map, strict_layout_map, q,
                       in_queue);
      for (int other : members) {
        if (other != attempt_root) {
          RunInferStep(other, InferLevel::kFree, true, tmp_layout_map,
                       strict_layout_map, q, in_queue);
          FinishInferQueue(InferLevel::kFree, tmp_layout_map, strict_layout_map,
                           q, in_queue);
        }
      }
    } catch (const LayoutConflictException &e) {
      ok = false;
      failure = e.what();
    } catch (const NormalizeIterException &e) {
      ok = false;
      failure = e.what();
    } catch (const LoopLayoutInjectiveException &e) {
      ok = false;
      failure = e.what();
    }
    std::optional<AttemptOutcome> outcome;
    if (ok) {
      AttemptCost cost = cost_model.Score(members, infer_list_, tmp_layout_map);
      outcome =
          AttemptOutcome{BackupInferList(), std::move(tmp_layout_map), cost};
    } else {
      DLOG(INFO) << "[InferInFreeMode] attempt root " << attempt_root
                 << " discarded: " << failure;
    }
    infer_list_ = std::move(back_infer_list);
    return outcome;
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
    // Union through reducer buffers: an epoch's update nests and its
    // init/finalize ops share no fragment buffer directly (the reducer is
    // not in use_list_), yet finalize's dst-steering proposal must compete
    // in the same attempt search as the update loop's layout choice.
    for (const auto &[var, infer_indices] : reducer_use_list_) {
      if (infer_indices.empty())
        continue;
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
    std::deque<int> q;
    std::vector<bool> in_queue(infer_list_.size(), false);

    std::unique_ptr<LayoutCostModel> cost_model =
        LayoutCostModel::Create(tl_config::LayoutCostModelName(), target_);
    DLOG(INFO) << "[InferInFreeMode] cost model: " << cost_model->Name();
    for (auto &&[root, members] : components) {
      DLOG(INFO) << "======================= processing component " << root
                 << '\n';
      std::vector<TileOperator> best_infer_list;
      LayoutMap best_layout_map;
      AttemptCost best_cost;
      bool has_best = false;
      int best_infer_root = -1;

      auto adopt = [&](AttemptOutcome &&outcome, int attempt_root) {
        best_infer_list = std::move(outcome.infer_list);
        best_layout_map = std::move(outcome.layout_map);
        best_cost = outcome.cost;
        has_best = true;
        best_infer_root = attempt_root;
      };

      // Try each member as the root of inference for this component.
      for (int attempt_infer_root : members) {
        DLOG(INFO) << "----------------------- try root " << attempt_infer_root
                   << " members " << members.size() << '\n';
        auto outcome = RunOneAttempt(attempt_infer_root, members, layout_map,
                                     strict_layout_map, /*seed_layouts=*/{},
                                     *cost_model, q, in_queue);
        if (!outcome) {
          continue;
        }
        DLOG(INFO) << "[InferInFreeMode] attempt root " << attempt_infer_root
                   << " cost model " << cost_model->Name()
                   << " output: mem=" << outcome->cost.mem
                   << " regs=" << outcome->cost.regs;
        // Keep the cheapest attempt; ties resolve to the earliest root so
        // the selection stays deterministic (and, with the cost model
        // disabled, byte-identical to the legacy register ordering).
        if (!has_best || outcome->cost.BetterThan(best_cost) ||
            (!best_cost.BetterThan(outcome->cost) &&
             attempt_infer_root < best_infer_root)) {
          adopt(std::move(*outcome), attempt_infer_root);
        }
      }
      if (!has_best) {
        // Reducer-only rescue (dst-steering): the verdict can be an induced
        // narrow layout no consumer ordering can live with (e.g. a consumer
        // also reads a strict-pinned conflicting source), killing every
        // attempt. Retry once with the reserved dsts pre-seeded to the
        // universally readable wide layout — finalize stays silent (dst
        // owned) and consumers adapt, trading the narrow plan for a
        // compiling wide one. Components without reservations get no seeds
        // and fall through unchanged; a component that fails even this is a
        // genuine inference failure.
        auto seeds =
            steering_.FallbackSeeds(members, infer_list_, thread_bounds_vec_);
        if (!seeds.empty()) {
          DLOG(INFO) << "[InferInFreeMode] all attempts failed; retrying with "
                     << "wide fallback dst layouts";
          auto outcome =
              RunOneAttempt(members.front(), members, layout_map,
                            strict_layout_map, seeds, *cost_model, q, in_queue);
          if (outcome) {
            adopt(std::move(*outcome), members.front());
          }
        }
      }
      ICHECK(has_best) << "no available layout found" << '\n';
      // Apply the best plan for this component
      infer_list_ = std::move(best_infer_list);
      layout_map = best_layout_map;
      DLOG(INFO) << "[InferInFreeMode] final selection: attempt root "
                 << best_infer_root << " cost model " << cost_model->Name()
                 << " output: mem=" << best_cost.mem
                 << " regs=" << best_cost.regs;
    }
  }
};

class LayoutInferencer : public IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = ParallelLoopFuser::Fuse(f->body);
    BufferUseDefCollector collector;
    collector.Collect(f);
    auto result = collector.Run();
    LayoutInferencer substituter(result, &analyzer);
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  LayoutInferencer(const LayoutInferenceResult &result,
                   arith::Analyzer *analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer), result_(result) {};

  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  /**
   * @brief Visit and mutate a Block node to attach inferred layout information.
   *
   * Converts the visited Block via the base visitor and attaches
   * result_.layout_map to the Block's annotations under attr::kLayoutMap.
   *
   * @return Stmt The (possibly modified) Block statement with the layout-map
   * annotation set.
   */
  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock block = Downcast<SBlock>(IRMutatorWithAnalyzer::VisitStmt_(op));

    auto block_ptr = block.CopyOnWrite();
    block_ptr->annotations.Set(attr::kLayoutMap, result_.layout_map);
    return block;
  }

  /**
   * @brief Visit and transform For nodes by storing inferred layout information
   *        as annotations instead of expanding the loop.
   *
   * If the For node is present in result_.for_map, this method stores the
   * inferred loop layout and predicate as annotations on the For node, rather
   * than performing loop partition and vectorization.
   *
   * The stored annotations are:
   * - attr::kParallelLoopLayout: The Fragment layout for the parallel loop
   * - attr::kParallelLoopPredicate: The predicate expression (if any)
   * - attr::kParallelLoopRequiresPaddingGuard: Whether inverse lowering must
   *   allow and guard padded points from a ragged SIMT partition
   *
   * @return The For statement with layout annotations attached
   */
  Stmt VisitStmt_(const ForNode *op) final {
    if (!result_.for_map.count(GetRef<For>(op))) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    For for_node = Downcast<For>(IRMutatorWithAnalyzer::VisitStmt_(op));
    auto root = GetRef<For>(op);

    auto loop_layout = result_.for_map[root];

    // Store the loop layout as an annotation on the For node (outermost)
    auto for_ptr = for_node.CopyOnWrite();
    for_ptr->annotations.Set(attr::kParallelLoopLayout, loop_layout);
    if (result_.padding_guard_map.count(root)) {
      for_ptr->annotations.Set(attr::kParallelLoopRequiresPaddingGuard,
                               result_.padding_guard_map[root]);
    }

    // Store the predicate as an annotation if it exists and is not trivially
    // true
    if (result_.predicate_map.count(root)) {
      PrimExpr predicate = analyzer_->Simplify(result_.predicate_map[root]);
      // Only store predicate if it's not trivially true
      if (!is_const_int(predicate, 1)) {
        for_ptr->annotations.Set(attr::kParallelLoopPredicate, predicate);
      }
    }

    return for_node;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

private:
  const LayoutInferenceResult result_;
};

tvm::transform::Pass LayoutInference() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    f = LayoutInferencer::Substitute(std::move(f));
    // Validate parallel loop layout annotations
    ParallelLoopLayoutValidator::Validate(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LayoutInference", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.LayoutInference", LayoutInference);
}

} // namespace tl
} // namespace tvm
