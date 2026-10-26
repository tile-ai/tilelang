#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "op/prim.h"
/*!
 * \file layout_inference.cc
 * \brief infer the fragment/shared memory layout
 */

#include "compat/region.h"
#include "support/check.h"
#include <tvm/runtime/logging.h>
#include <tvm/tirx/index_map.h>
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
#include "../../op/reduce.h"
#include "../../op/reducer.h"
#include "../../op/utils.h"
#include "../../span_utils.h"
#include "../common/loop_aware_visitor.h"
#include "../common/loop_fusion_utils.h"
#include "../common/union_find.h"
#include "backend/common/target_utils.h"
#include "compat/thread.h"
#include "compat/visit.h"
#include "layout_cost_model.h"
#include "layout_inference.h"
#include "parallel_loop_layout_validator.h"
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

int64_t GetElementStorageBits(PrimType dtype) {
  // Layout aliasing must be reasoned about in logical storage bits per element,
  // not in bytes.  For sub-byte dtypes such as fp4, `dtype.bytes()` rounds up
  // to 1 and loses the "two fp4 values share one byte" relationship that
  // reinterpreting views rely on.
  return static_cast<int64_t>(dtype.bits()) * dtype.lanes();
}

bool ShapesEqual(const Array<PrimExpr> &lhs, const Array<PrimExpr> &rhs,
                 sym::Analyzer analyzer) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!tl::arith::CanProveEqual(analyzer, lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

Optional<TensorVar> FindLayoutAnchorBuffer(const Array<TensorVar> &buffers,
                                           const Layout &layout,
                                           sym::Analyzer analyzer) {
  for (const auto &buffer : buffers) {
    if (ShapesEqual(layout->InputShape(), buffer.shape(), analyzer)) {
      return buffer;
    }
  }
  return std::nullopt;
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
bool IsOpenFragmentLayout(const TensorVar &buffer, const Layout &layout) {
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
using IRVisitorWithAnalyzer = LoopAwareVisitor;

struct LayoutInferenceResult {
  Map<TensorVar, Layout> layout_map;
  Map<For, Fragment> for_map;
  Map<For, PrimExpr> predicate_map;
  Map<For, bool> padding_guard_map;
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
  /*! \brief Register every finalize that is a capable proposer
   *  (FinalizeReducerV2OpNode::CanSteerDst). The reducer edge needs no wake
   *  table anymore: the reducer buffer carries a PartialFragment in
   *  use_list_, so an update nest's commit re-enqueues the finalize like any
   *  buffer edge, and zero-update reducers are pre-seeded wide by the
   *  engine — finalize always finds a solved partial to read. */
  void Reserve(const std::vector<TileOperator> &infer_list,
               const std::vector<Range> &thread_bounds_vec,
               const std::vector<sym::Analyzer> analyzer_vec,
               const Map<TensorVar, Layout> &annotated_layouts) {
    for (int i = 0; i < static_cast<int>(infer_list.size()); ++i) {
      const auto *finalize = infer_list[i].as<FinalizeReducerV2OpNode>();
      if (finalize == nullptr) {
        continue;
      }
      const TensorVar &dst = finalize->dst;
      if (annotated_layouts.count(dst) ||
          !FinalizeReducerV2OpNode::CanSteerDst(
              finalize->reducer, dst, thread_bounds_vec[i], analyzer_vec[i])) {
        continue;
      }
      owners_[dst].insert(i);
      finalize_ops_.insert(i);
      DLOG(INFO) << "[ReducerDstSteering] buffer " << dst
                 << " reserved for finalize op " << i;
    }
  }

  /*! \brief First-assignment ownership check for the commit point. Strict
   *  deductions stay authoritative (e.g. constant indexing inside a parallel
   *  loop genuinely forces replication); common/free proposals must come
   *  from the owning finalize. Consumers frozen before the owner's proposal
   *  lands re-validate when the engine re-enqueues them. */
  bool AllowsCommit(const TensorVar &buffer, int proposer,
                    InferLevel level) const {
    if (level == InferLevel::kStrict) {
      return true;
    }
    auto it = owners_.find(buffer);
    return it == owners_.end() || it->second.count(proposer);
  }

  /*! \brief Seeds for the wide fallback attempt: every reserved dst of the
   *  component pinned to the universally readable replicated layout. Empty
   *  for components without reservations, which therefore fall through to
   *  the ordinary "no available layout" failure unchanged. */
  std::vector<std::pair<TensorVar, Fragment>>
  FallbackSeeds(const std::vector<int> &members,
                const std::vector<TileOperator> &infer_list,
                const std::vector<Range> &thread_bounds_vec) const {
    std::vector<std::pair<TensorVar, Fragment>> seeds;
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
  std::unordered_map<TensorVar, std::unordered_set<int>, ObjectPtrHash,
                     ObjectPtrEqual>
      owners_;
  std::unordered_set<int> finalize_ops_;
};

class BufferUseDefCollector : public IRVisitorWithAnalyzer {
public:
  BufferUseDefCollector() = default;

  explicit BufferUseDefCollector(Target target,
                                 LayoutInferenceOptions options = {})
      : options_(std::move(options)), target_(std::move(target)) {}

  void RunInferStep(int cur_infer_id, InferLevel level, bool update_queue,
                    LayoutMap &layout_map, const LayoutMap &strict_layout_map,
                    std::deque<int> &q, std::vector<bool> &in_queue,
                    int candidate_vector_size_limit = 0) {
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
    sym::Analyzer cur_analyzer = analyzer_vec_[cur_infer_id];
    // Double-check that 'next' is valid
    ICHECK(next.defined()) << "infer_list_[" << cur_infer_id
                           << "] is null inside run_infer_step.";

    // Check the logical thread index and the thread bounds extent.
    ICHECK(thread_index.defined())
        << "thread_index_vec_[" << cur_infer_id << "] is not defined.";
    ICHECK(thread_bounds.defined())
        << "thread_bounds_vec_[" << cur_infer_id << "] is not defined.";

    auto extent_ptr = GetConstInt(thread_bounds->extent);
    ICHECK(extent_ptr.has_value())
        << "thread_bounds->extent is not a constant integer, which is "
           "required for layout inference.";

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
                                                  strict_layout_map,
                                                  candidate_vector_size_limit},
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

      if (IsReducerV2Buffer(buffer) && tl_config::ReducerPlanVerboseEnabled()) {
        LOG(INFO) << "[ReducerLayout] op " << cur_infer_id << " at "
                  << InferLevelToString(level) << " proposes "
                  << layout->DebugOutput();
      }

      // Helper: propagate inferred layout to alias buffers (same data Var)
      auto propagate_alias = [&](const TensorVar &src_buffer,
                                 const Layout &src_layout) {
        if (!buffer_data_to_buffers_.count(StorageRoot(src_buffer)))
          return;
        const auto &siblings = buffer_data_to_buffers_[StorageRoot(src_buffer)];
        for (const auto &sib : siblings) {
          if (sib.same_as(src_buffer))
            continue;
          bool shapes_equal =
              src_layout->InputShape().size() == sib.shape().size();
          if (shapes_equal) {
            for (size_t i = 0; i < src_layout->InputShape().size(); ++i) {
              if (!tl::arith::CanProveEqual(
                      analyzer_, src_layout->InputShape()[i], sib.shape()[i])) {
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
                        sib.shape(), analyzer_,
                        tl::MakeConst(
                            PrimType::Int(32),
                            GetElementStorageBits(src_buffer.dtype())),
                        tl::MakeConst(PrimType::Int(32),
                                      GetElementStorageBits(sib.dtype())));
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
        // Reducer partials: monotone widen-on-conflict lattice
        // (unset -> narrow -> wide). Equal proposals are absorbed;
        // disagreeing update sites widen to the participant-wide plan,
        // which every already-committed loop layout is compatible with, so
        // late widening never rolls back a decision. Fatal conflicts and
        // ProveFragmentContains (whose fully-replicated shortcut assumes
        // equal copies, not addends) do not apply here.
        if (IsReducerV2Buffer(buffer)) {
          const auto *existing = layout_map[buffer].as<PartialFragmentNode>();
          const auto *incoming = layout.as<PartialFragmentNode>();
          ICHECK(existing != nullptr && incoming != nullptr)
              << "reducer " << buffer << " must carry a PartialFragment, got "
              << layout->GetTypeKey()
              << " (mapped: " << layout_map[buffer]->GetTypeKey() << ")";
          if (incoming->IsEqual(existing)) {
            continue;
          }
          // A strict partial is a user annotation (annotated layouts seed
          // the strict snapshot; nothing else pins a reducer at kStrict):
          // never widen the user's plan away — surface the conflict. Thrown,
          // not fatal: free-mode attempts catch it and may find an ordering
          // (update nest first, deriving its loop from the pinned partial)
          // that satisfies the annotation.
          if (strict_layout_map.count(buffer)) {
            std::ostringstream oss;
            oss << "Layout infer conflict for reducer `" << buffer->name
                << "`: an update site induces a partial layout different "
                << "from the annotated one.\n  annotated: "
                << existing->DebugOutput()
                << "\n  induced:   " << incoming->DebugOutput()
                << "\nAdjust the T.annotate_layout PartialFragment or the "
                << "update loop structure so they agree.";
            throw LayoutConflictException(oss.str());
          }
          Layout widened = PartialFragment::FullyReplicated(
              buffer.shape(), thread_bounds->extent, thread_bounds);
          if (widened->IsEqual(existing)) {
            continue; // already at the lattice top
          }
          DLOG(INFO) << "[RunInferStep] update sites disagree on reducer "
                     << buffer << "; widening to the participant-wide plan";
          layout_map.Set(buffer, widened);
          if (update_queue && use_list_.count(buffer)) {
            for (int idx : use_list_[buffer]) {
              EnqueueWithPriority(idx, q, in_queue, cur_infer_id, layout_map);
            }
          }
          continue;
        }
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
          sym::Analyzer inner_analyzer;
          for (int i = 0; i < dst_layout->InputDim(); ++i) {
            auto x = InputPlaceholder(i);
            indices.push_back(x);
            // should be literal - literal = 0, any analyzer will work
            ICHECK(tl::arith::CanProveEqual(inner_analyzer,
                                            dst_layout->InputShape()[i],
                                            src_layout->InputShape()[i]));
            tl::arith::Bind(inner_analyzer, x,
                            Range(0, dst_layout->InputShape()[i]));
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
                     << SpanHintSuffix(buffer->loc);
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
        // First commit for a reducer must already be the partial kind.
        if (IsReducerV2Buffer(buffer)) {
          ICHECK(layout.as<PartialFragmentNode>())
              << "reducer " << buffer << " must carry a PartialFragment, got "
              << layout->GetTypeKey();
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

    // (The former reducer-edge wake table is gone: an update nest's
    // PartialFragment commit re-enqueues the finalize through use_list_.)
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

    steering_.Reserve(infer_list_, thread_bounds_vec_, analyzer_vec_,
                      annotated_layout_map_);

    // If needed, you can also check that annotated_layout_map_ is not empty, or
    // anything else relevant to your setup.

    // Copy the annotated layout map to local variable
    Map<TensorVar, Layout> layout_map = annotated_layout_map_;
    Map<TensorVar, Layout> strict_layout_map;
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
          Fragment::FullyReplicated(buffer.shape(), thread_bounds->extent)
              ->BindThreadRange(thread_bounds);
      layout_map.Set(buffer, frag);
    }

    // step 0.5: seed the wide-plan floor for zero-update reducers. Reducers
    // with update sites get their partial layout proposed by the update
    // nests; an epoch with no update at all (seed/init-only) has no proposer,
    // so pin it to the participant-wide plan here — deterministically, before
    // any level runs, so finalize can always read a solved partial.
    for (const auto &[buffer, users] : use_list_) {
      if (!IsReducerV2Buffer(buffer) || layout_map.count(buffer) ||
          reducer_update_sites_.count(buffer.get())) {
        continue;
      }
      ICHECK(!users.empty());
      const Range &thread_bounds = thread_bounds_vec_[users.front()];
      if (tl_config::ReducerPlanVerboseEnabled()) {
        LOG(INFO) << "[ReducerLayout] `" << buffer->name
                  << "` has no update sites";
      }
      layout_map.Set(buffer,
                     PartialFragment::FullyReplicated(
                         buffer.shape(), thread_bounds->extent, thread_bounds));
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
      Optional<TensorVar> rep;
      Optional<Layout> rep_layout;
      for (const auto &buf : buffers) {
        if (layout_map.count(buf)) {
          rep = buf;
          rep_layout = layout_map[buf];
          break;
        }
      }
      if (!rep_layout.has_value())
        continue;
      for (const auto &buf : buffers) {
        if (!layout_map.count(buf)) {
          bool shapes_equal =
              rep_layout.value()->InputShape().size() == buf.shape().size();
          if (shapes_equal) {
            for (size_t i = 0; i < rep_layout.value()->InputShape().size();
                 ++i) {
              if (!tl::arith::CanProveEqual(analyzer_,
                                            rep_layout.value()->InputShape()[i],
                                            buf.shape()[i])) {
                shapes_equal = false;
                break;
              }
            }
          }
          Layout reshaped =
              shapes_equal
                  ? rep_layout.value()
                  : rep_layout.value()->Reshape(
                        buf.shape(), analyzer_,
                        tl::MakeConst(
                            PrimType::Int(32),
                            GetElementStorageBits(rep.value().dtype())),
                        tl::MakeConst(PrimType::Int(32),
                                      GetElementStorageBits(buf.dtype())));
          layout_map.Set(buf, reshaped);
        }
      }
    }

    // Check that all local.fragment buffers have inferred layouts
    for (const auto &[buffer, _] : use_list_) {
      if (IsFragmentBuffer(buffer)) {
        ICHECK_NE(layout_map.count(buffer), 0)
            << "The layout for fragment " << buffer
            << " can not be inferred correctly." << SpanHintSuffix(buffer->loc);
      } else if (IsReducerV2Buffer(buffer)) {
        // Every reducer must end up with a partial layout: update nests
        // propose it, and zero-update epochs were seeded wide in step 0.5.
        ICHECK_NE(layout_map.count(buffer), 0)
            << "The partial layout for reducer " << buffer
            << " can not be inferred correctly." << SpanHintSuffix(buffer->loc);
      }
    }

    // Collect layout info for For nodes
    Map<For, Fragment> for_map;
    Map<For, PrimExpr> predicate_map;
    Map<For, bool> padding_guard_map;
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
          padding_guard_map.Set(for_infer->GetRoot(), true);
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

  void Collect(const tirx::Function &function) {
    for (const Var &parameter : function->params) {
      if (auto tensor = parameter.as<TensorVar>())
        RegisterTensor(tensor.value());
    }
    if (!function->body.has_value())
      return;
    PostOrderVisit(function->body.value(), [this](const ObjectRef &object) {
      if (const auto *bind = object.as<BindNode>()) {
        if (auto tensor = bind->var.as<TensorVar>()) {
          Var storage = tensor.value();
          if (const auto *call = bind->value.as<CallNode>();
              call && call->op.same_as(tirx::decl_tensor_op()) &&
              !call->args.empty() && tensor.value().ty()->HasDefaultLayout()) {
            if (auto pointer = call->args[0].as<Var>())
              storage = pointer.value();
            if (const auto *data = call->args[0].as<CallNode>();
                data && data->op.same_as(tirx::tensor_data_ptr_op())) {
              storage = StorageRoot(data->args[0].as_or_throw<TensorVar>());
            }
          }
          RegisterTensor(tensor.value(), storage);
        }
      }
      if (auto tensor = object.as<TensorVar>())
        RegisterTensor(tensor.value());
    });
    if (auto layouts = function->GetAttr<LayoutMap>(attr::kLayoutMap)) {
      for (const auto &[tensor, layout] : layouts.value())
        RecordAnnotatedLayout(tensor, layout);
    }
    Visit(function->body.value());
    ComputeFloatingFragmentBuffers(function->body.value());
  }

private:
  Var StorageRoot(const TensorVar &tensor) const {
    return storage_roots_.Get(tensor).value_or(tensor);
  }

  void RegisterTensor(const TensorVar &tensor,
                      Optional<Var> storage = std::nullopt) {
    Var root = storage.value_or(StorageRoot(tensor));
    if (auto previous = storage_roots_.Get(tensor);
        previous && !previous.value().same_as(root)) {
      auto aliases = buffer_data_to_buffers_.at(previous.value());
      Array<TensorVar> retained;
      for (const TensorVar &alias : aliases) {
        if (!alias.same_as(tensor))
          retained.push_back(alias);
      }
      buffer_data_to_buffers_.Set(previous.value(), retained);
    }
    storage_roots_.Set(tensor, root);
    Array<TensorVar> aliases =
        buffer_data_to_buffers_.Get(root).value_or(Array<TensorVar>{});
    for (const TensorVar &alias : aliases) {
      if (alias.same_as(tensor))
        return;
    }
    aliases.push_back(tensor);
    buffer_data_to_buffers_.Set(root, aliases);
  }

  void RecordAnnotatedLayout(const TensorVar &tensor, const Layout &layout) {
    RegisterTensor(tensor);
    const auto &aliases = buffer_data_to_buffers_.at(StorageRoot(tensor));
    int64_t anchor_bits = GetElementStorageBits(tensor.dtype());
    for (const TensorVar &alias : aliases) {
      TVM_FFI_CHECK(alias.ty()->HasDefaultLayout(), ValueError)
          << "Tensor `" << alias->name
          << "` cannot own both native and TileLang layouts";
      bool same_shape =
          ShapesEqual(layout->InputShape(), alias.shape(), analyzer_);
      if (IsReducerV2Buffer(alias)) {
        TVM_FFI_CHECK(layout.as<PartialFragmentNode>() && same_shape,
                      ValueError)
            << "Reducer layout requires a PartialFragment with exactly the "
               "reducer shape";
      }
      Layout resolved =
          same_shape ? layout
                     : layout->Reshape(
                           alias.shape(), analyzer_,
                           tl::MakeConst(PrimType::Int(32), anchor_bits),
                           tl::MakeConst(PrimType::Int(32),
                                         GetElementStorageBits(alias.dtype())));
      if (IsSharedBuffer(alias) || IsReducerV2Buffer(alias)) {
        auto injectivity = resolved->DetectInjective();
        TVM_FFI_CHECK(injectivity->errors.empty(), ValueError)
            << "Invalid layout for tensor `" << alias->name
            << "`: " << injectivity->errors;
      }
      if (auto previous = annotated_layout_map_.Get(alias)) {
        TVM_FFI_CHECK(previous.value()->IsEqual(resolved.get()), ValueError)
            << "Conflicting layout annotations for tensor `" << alias->name
            << "`";
      }
      annotated_layout_map_.Set(alias, resolved);
    }
  }

  bool HasKnownLayoutAnchor(int index, const LayoutMap &layouts) const {
    auto touched = op_touched_buffers_.find(index);
    if (touched == op_touched_buffers_.end())
      return false;
    for (const TensorVar &tensor : touched->second) {
      if (layouts.count(tensor))
        return true;
    }
    return false;
  }

  void EnqueueWithPriority(int index, std::deque<int> &queue,
                           std::vector<bool> &pending, int current,
                           const LayoutMap &layouts) const {
    if (index == current || pending[index])
      return;
    pending[index] = true;
    if (HasKnownLayoutAnchor(index, layouts))
      queue.push_front(index);
    else
      queue.push_back(index);
  }

  void addToUseList(const TensorVar &tensor) {
    if (!IsFragmentBuffer(tensor) && !IsReducerV2Buffer(tensor))
      return;
    int index = static_cast<int>(infer_list_.size());
    auto &users = use_list_[tensor];
    if (std::find(users.begin(), users.end(), index) == users.end())
      users.push_back(index);
    auto &touched = op_touched_buffers_[index];
    if (std::none_of(
            touched.begin(), touched.end(),
            [&](const TensorVar &other) { return other.same_as(tensor); }))
      touched.push_back(tensor);
  }

  void RecordOperator(const TileOperator &operation,
                      const ObjectRef &statement) {
    if (options_.validate_operator)
      options_.validate_operator(operation, analyzer_);
    AccessRegions regions = operation->GetAccessRegions();
    for (const auto &accesses : {regions.reads, regions.writes}) {
      for (const TensorRegion &region : accesses)
        addToUseList(region->source.as_or_throw<TensorVar>());
    }
    infer_list_stmt_.push_back(statement);
    infer_list_.push_back(operation);
    thread_index_vec_.push_back(CurrentThreadIndex());
    thread_bounds_vec_.push_back(
        ComputeThreadBounds(thread_binding_, analyzer_));
    analyzer_vec_.push_back(analyzer_->Clone());
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const CallNode *node) final {
    if (node->op.as<OpNode>() &&
        node->op.as_or_throw<Op>()->name == "tl.layout_hint") {
      TVM_FFI_CHECK(node->args.size() == 1 && node->attrs.has_value(),
                    ValueError)
          << "Malformed layout hint";
      auto attributes = node->attrs.value().as_or_throw<DictAttrs>();
      RecordAnnotatedLayout(node->args[0].as_or_throw<TensorVar>(),
                            attributes.GetAttr<Layout>("layout").value());
      return std::nullopt;
    }
    IRVisitorWithAnalyzer::Visit_(node);
    TileOperator operation =
        ParseOperator(GetRef<Call>(node), block_annotations_);
    if (operation.defined()) {
      for (const Expr &argument : node->args)
        CollectFragmentBuffersFromExpr(argument);
      RecordOperator(operation, GetRef<ObjectRef>(node));
    }
    return std::nullopt;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const ForNode *node) final {
    if (node->kind != ForKind::kParallel)
      return IRVisitorWithAnalyzer::Visit_(node);
    ParallelOp operation(GetRef<For>(node));
    for (const TensorVar &tensor : operation->GetAccessOrder())
      addToUseList(tensor);
    int index = static_cast<int>(infer_list_.size());
    PostOrderVisit(node->body, [this, index](const ObjectRef &object) {
      if (const auto *call = object.as<CallNode>();
          call && call->op.same_as(reducer_update())) {
        ReducerUpdateArgs update = ParseReducerUpdate(call);
        reducer_update_sites_[update.reducer.get()].push_back(
            {index, update.indices, update.value});
        addToUseList(update.reducer);
      }
      if (const auto *bind = object.as<BindNode>())
        bind_var_to_expr_.Set(bind->var, bind->value);
    });
    RecordOperator(operation, GetRef<ObjectRef>(node));
    return std::nullopt;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const RegionStmtNode *node) final {
    if (options_.opaque_region && options_.opaque_region(node))
      return std::nullopt;
    auto previous_thread = thread_binding_;
    auto previous_annotations = block_annotations_;
    if (node->op.same_as(tirx::launch_thread_op()) &&
        node->args[0].as_or_throw<StringImm>()->value == "threadIdx.x") {
      thread_binding_ = node->body_params[0].as_or_throw<PrimVar>();
    }
    if (node->op->name == "tl.scope" ||
        node->op.same_as(::tvm::tl::scope_op())) {
      for (const auto &[key, value] : node->attrs->dict)
        block_annotations_.Set(key, value);
    }
    IRVisitorWithAnalyzer::Visit_(node);
    if (auto layouts = node->attrs.GetAttr<LayoutMap>(attr::kLayoutMap)) {
      for (const auto &[tensor, layout] : layouts.value())
        RecordAnnotatedLayout(tensor, layout);
    }
    block_annotations_ = previous_annotations;
    thread_binding_ = previous_thread;
    return std::nullopt;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const BindNode *node) final {
    bind_var_to_expr_.Set(node->var, node->value);
    return IRVisitorWithAnalyzer::Visit_(node);
  }

  void CollectFragmentBuffersFromExpr(const Expr &expression) {
    std::unordered_set<const VarNode *> visited;
    std::function<void(const Expr &)> collect = [&](const Expr &value) {
      PostOrderVisit(value, [&](const ObjectRef &object) {
        if (const auto *load = object.as<TensorLoadNode>()) {
          addToUseList(load->source.as_or_throw<TensorVar>());
        } else if (auto variable = object.as<Var>()) {
          if (visited.insert(variable.value().get()).second) {
            if (auto binding = bind_var_to_expr_.Get(variable.value()))
              collect(binding.value());
          }
        }
      });
    };
    collect(expression);
  }

  PrimExpr CurrentThreadIndex() const {
    return thread_binding_.has_value() ? PrimExpr(thread_binding_.value())
                                       : PrimExpr(0);
  }

  void ComputeFloatingFragmentBuffers(const Stmt &body) {
    std::unordered_set<const Object *> inside;
    for (const ObjectRef &statement : infer_list_stmt_) {
      PostOrderVisit(statement, [&](const ObjectRef &object) {
        inside.insert(object.get());
      });
    }
    class FloatingCollector : public IRVisitorWithAnalyzer {
    public:
      FloatingCollector(const std::unordered_set<const Object *> &inside,
                        std::unordered_map<TensorVar, Range, ObjectPtrHash,
                                           ObjectPtrEqual> &floating,
                        const LayoutInferenceOptions &options)
          : inside_(inside), floating_(floating), options_(options) {}

    private:
      ffi::Optional<ffi::VisitInterrupt>
      Visit_(const RegionStmtNode *node) final {
        if (options_.opaque_region && options_.opaque_region(node))
          return std::nullopt;
        auto previous = thread_;
        if (node->op.same_as(tirx::launch_thread_op()) &&
            node->args[0].as_or_throw<StringImm>()->value == "threadIdx.x") {
          thread_ = node->body_params[0].as_or_throw<PrimVar>();
        }
        IRVisitorWithAnalyzer::Visit_(node);
        thread_ = previous;
        return std::nullopt;
      }
      ffi::Optional<ffi::VisitInterrupt>
      Visit_(const TensorLoadNode *node) final {
        Record(node->source.as_or_throw<TensorVar>(), node);
        return IRVisitorWithAnalyzer::Visit_(node);
      }
      ffi::Optional<ffi::VisitInterrupt>
      Visit_(const TensorStoreNode *node) final {
        Record(node->dest.as_or_throw<TensorVar>(), node);
        return IRVisitorWithAnalyzer::Visit_(node);
      }
      void Record(const TensorVar &tensor, const Object *node) {
        if (IsFragmentBuffer(tensor) && !inside_.count(node) &&
            !floating_.count(tensor)) {
          floating_.emplace(tensor, ComputeThreadBounds(thread_, analyzer_));
        }
      }
      const std::unordered_set<const Object *> &inside_;
      std::unordered_map<TensorVar, Range, ObjectPtrHash, ObjectPtrEqual>
          &floating_;
      Optional<PrimVar> thread_;
      const LayoutInferenceOptions &options_;
    };
    make_object<FloatingCollector>(inside, floating_fragment_buffers_, options_)
        ->Visit(body);
  }

  LayoutInferenceOptions options_;
  Map<TensorVar, Var> storage_roots_;
  Map<Var, Array<TensorVar>> buffer_data_to_buffers_;
  Map<Var, Expr> bind_var_to_expr_;
  BlockAnnotations block_annotations_;
  std::vector<ObjectRef> infer_list_stmt_;
  std::vector<TileOperator> infer_list_;
  std::unordered_map<TensorVar, Range, ObjectPtrHash, ObjectPtrEqual>
      floating_fragment_buffers_;
  std::unordered_map<TensorVar, std::vector<int>, ObjectPtrHash, ObjectPtrEqual>
      use_list_;
  struct ReducerUpdateSiteRecord {
    int infer_idx;
    Array<PrimExpr> indices;
    PrimExpr value;
  };
  std::unordered_map<const VarNode *, std::vector<ReducerUpdateSiteRecord>>
      reducer_update_sites_;
  ReducerDstSteering steering_;
  std::unordered_map<int, std::vector<TensorVar>> op_touched_buffers_;
  Optional<PrimVar> thread_binding_;
  std::vector<PrimExpr> thread_index_vec_;
  std::vector<Range> thread_bounds_vec_;
  std::vector<sym::Analyzer> analyzer_vec_;
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
  std::optional<AttemptOutcome> RunOneAttempt(
      int attempt_root, const std::vector<int> &members,
      const LayoutMap &base_layout_map, const LayoutMap &strict_layout_map,
      const std::vector<std::pair<TensorVar, Fragment>> &seed_layouts,
      const LayoutCostModel &cost_model, int candidate_vector_size_limit = 0) {
    auto back_infer_list = BackupInferList();
    LayoutMap tmp_layout_map = base_layout_map;
    // A failed attempt can leave pending propagation work. Keep both the
    // queue and its membership flags local so no later attempt inherits it.
    std::deque<int> q;
    std::vector<bool> in_queue(infer_list_.size(), false);
    for (const auto &[buffer, fragment] : seed_layouts) {
      if (!tmp_layout_map.count(buffer)) {
        tmp_layout_map.Set(buffer, fragment);
      }
    }
    bool ok = true;
    std::string failure;
    try {
      // Only the root's first inference receives the candidate cap. Its
      // solved layout is frozen; propagation and later visits use normal args.
      RunInferStep(attempt_root, InferLevel::kFree, true, tmp_layout_map,
                   strict_layout_map, q, in_queue, candidate_vector_size_limit);
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
      if (tl_config::ReducerPlanVerboseEnabled())
        LOG(INFO) << "[ReducerLayout] attempt " << attempt_root
                  << " discarded: " << failure;
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

      // Union all infer_list_ indices that share the same TensorVar object
      int first_idx = infer_indices[0];
      for (size_t i = 1; i < infer_indices.size(); i++) {
        uf.Union(first_idx, infer_indices[i]);
      }
    }
    // Additionally, union across buffers that share the same underlying
    // StorageRoot(buffer) (Var). This handles cases like reshape where multiple
    // TensorVar objects alias the same storage.
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
    // (An epoch's update nests and its init/finalize ops now share the
    // reducer buffer in use_list_, so the union above already links them.)

    std::unordered_map<int, std::vector<int>> components;
    for (int i = 0; i < infer_list_.size(); i++) {
      int root = uf.Find(i);
      components[root].push_back(i);
    }
    // Create a map from root to buffers
    std::unordered_map<int, std::vector<TensorVar>> components_buffers;
    for (const auto &[buffer, infer_indices] : use_list_) {
      int root = uf.Find(infer_indices[0]);
      components_buffers[root].push_back(buffer);
    }
    // Keep components_buffers for debug purpose
    (void)components_buffers;

    // For each component, try each op as root, and determine the least
    // replicated one
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
      int batch_vector_limit = 0;
      for (int member : members) {
        if (const auto *reduction = infer_list_[member].as<ReduceOpNode>();
            reduction && reduction->batch > 1) {
          batch_vector_limit =
              std::max(batch_vector_limit,
                       128 / static_cast<int>(reduction->src.dtype().bits()));
        }
      }

      auto adopt = [&](AttemptOutcome &&outcome, int attempt_root) {
        best_infer_list = std::move(outcome.infer_list);
        best_layout_map = std::move(outcome.layout_map);
        best_cost = outcome.cost;
        has_best = true;
        best_infer_root = attempt_root;
      };

      // Try the native plan first, then a scalar alternative at eligible
      // reducer roots. Scalar ownership can avoid replicated accumulator
      // slots; the existing cost model decides whether that is worthwhile.
      for (int attempt_infer_root : members) {
        const auto *loop = infer_list_[attempt_infer_root].as<ParallelOpNode>();
        bool try_scalar =
            cost_model->ExploreReducerScalarLayouts() && loop &&
            loop->HasReducerUpdates() && !loop->GetLoopLayout().defined() &&
            !loop->annotated_layout_unbound_.has_value() &&
            !loop->GetRoot()->annotations.count(attr::kCoalescedWidth);
        std::vector<int> vector_limits{0};
        if (batch_vector_limit &&
            (loop || infer_list_[attempt_infer_root].as<CopyNode>())) {
          for (int limit = batch_vector_limit; limit > 0; limit /= 2)
            vector_limits.push_back(limit);
        } else if (try_scalar) {
          vector_limits.push_back(1);
        }
        for (int candidate_vector_size_limit : vector_limits) {
          DLOG(INFO) << "----------------------- try root "
                     << attempt_infer_root << " members " << members.size()
                     << " candidate_vector_size_limit="
                     << candidate_vector_size_limit << '\n';
          auto outcome = RunOneAttempt(
              attempt_infer_root, members, layout_map, strict_layout_map,
              /*seed_layouts=*/{}, *cost_model, candidate_vector_size_limit);
          if (!outcome) {
            continue;
          }
          DLOG(INFO) << "[InferInFreeMode] attempt root " << attempt_infer_root
                     << " cost model " << cost_model->Name()
                     << " output: mem=" << outcome->cost.mem
                     << " regs=" << outcome->cost.regs;
          // Ties keep the earliest root and its native plan. The scalar
          // alternative must improve the score to replace the same root.
          if (!has_best || outcome->cost.BetterThan(best_cost) ||
              (!best_cost.BetterThan(outcome->cost) &&
               attempt_infer_root < best_infer_root)) {
            adopt(std::move(*outcome), attempt_infer_root);
          }
          if (batch_vector_limit && !try_scalar)
            break;
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
          auto outcome = RunOneAttempt(members.front(), members, layout_map,
                                       strict_layout_map, seeds, *cost_model);
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

class LayoutInferencer : public tvm::tl::StmtExprMutator {
public:
  explicit LayoutInferencer(LayoutInferenceResult result,
                            LayoutInferenceOptions options = {})
      : result_(std::move(result)), options_(std::move(options)) {}

  static tirx::Function Run(tirx::Function function,
                            LayoutInferenceOptions options = {}) {
    if (!function->body.has_value())
      return function;
    auto target = function->GetAttr<Target>("target");
    TVM_FFI_CHECK(target.has_value(), ValueError)
        << "LayoutInference requires a target attribute";
    if (options.fuse_parallel_loops) {
      function.CopyOnWrite()->body =
          SeqStmt(ParallelLoopFuser::Fuse(function->body.value()));
    }
    auto collector =
        make_object<BufferUseDefCollector>(target.value(), options);
    collector->Collect(function);
    auto result = collector->Run();
    auto mutator = make_object<LayoutInferencer>(result, options);
    SeqStmt body = function->body.value();
    body = mutator->Mutate(body).ValueOrUnchanged(body).as_or_throw<SeqStmt>();
    ParallelLoopLayoutValidator::Validate(body, options.opaque_region);
    Array<Stmt> statements;
    for (const Var &parameter : function->params) {
      if (auto tensor = parameter.as<TensorVar>()) {
        if (auto layout = result.layout_map.Get(tensor.value())) {
          statements.push_back(MakeHint(tensor.value(), layout.value()));
        }
      }
    }
    for (const Stmt &statement : body->seq)
      statements.push_back(statement);
    auto attributes = function->attrs;
    attributes.CopyOnWrite()->dict.erase(attr::kLayoutMap);
    auto *node = function.CopyOnWrite();
    node->body = SeqStmt(statements);
    node->attrs = attributes;
    return function;
  }

private:
  static Stmt MakeHint(const TensorVar &tensor, const Layout &layout) {
    return Evaluate(Call(std::nullopt, Op::Get("tl.layout_hint"), {tensor},
                         DictAttrs({{"layout", layout}})));
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *node,
                            InplaceMode inplace_mode) final {
    Stmt statement = tvm::tl::StmtExprMutator::Mutate_(node, inplace_mode)
                         .ValueOrUnchanged(GetRef<Stmt>(node));
    if (auto tensor = node->var.as<TensorVar>()) {
      if (auto layout = result_.layout_map.Get(tensor.value())) {
        return SeqStmt({statement, MakeHint(tensor.value(), layout.value())});
      }
    }
    return statement;
  }

  UnchangedOr<Stmt> Mutate_(const ForNode *node,
                            InplaceMode inplace_mode) final {
    For loop = tvm::tl::StmtExprMutator::Mutate_(node, inplace_mode)
                   .ValueOrUnchanged(GetRef<Stmt>(node))
                   .as_or_throw<For>();
    For original = GetRef<For>(node);
    if (auto layout = result_.for_map.Get(original)) {
      auto *updated = loop.CopyOnWrite();
      updated->annotations.Set(attr::kParallelLoopLayout, layout.value());
      if (auto padding = result_.padding_guard_map.Get(original)) {
        updated->annotations.Set(attr::kParallelLoopRequiresPaddingGuard,
                                 padding.value());
      }
      if (auto predicate = result_.predicate_map.Get(original)) {
        PrimExpr simplified =
            tl::arith::Simplify(sym::Analyzer(), predicate.value());
        if (!IsOne(simplified))
          updated->annotations.Set(attr::kParallelLoopPredicate, simplified);
      }
    }
    return loop;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                            InplaceMode inplace_mode) final {
    if (options_.opaque_region && options_.opaque_region(node))
      return ffi::Unchanged{};
    RegionStmt region = tvm::tl::StmtExprMutator::Mutate_(node, inplace_mode)
                            .ValueOrUnchanged(GetRef<Stmt>(node))
                            .as_or_throw<RegionStmt>();
    if (region->attrs->dict.count(attr::kLayoutMap)) {
      DictAttrs attributes = region->attrs;
      attributes.CopyOnWrite()->dict.erase(attr::kLayoutMap);
      region.CopyOnWrite()->attrs = attributes;
    }
    return region;
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateNode *node,
                            InplaceMode inplace_mode) final {
    if (const auto *call = node->value.as<CallNode>();
        call && call->op.as<OpNode>() &&
        call->op.as_or_throw<Op>()->name == "tl.layout_hint") {
      return SeqStmt({});
    }
    return tvm::tl::StmtExprMutator::Mutate_(node, inplace_mode);
  }

  LayoutInferenceResult result_;
  LayoutInferenceOptions options_;
};

tirx::Function InferLayouts(tirx::Function function,
                            LayoutInferenceOptions options) {
  return LayoutInferencer::Run(std::move(function), std::move(options));
}

tvm::transform::Pass LayoutInference() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    return InferLayouts(std::move(function));
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LayoutInference");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.LayoutInference", LayoutInference);
}

} // namespace tl
} // namespace tvm
