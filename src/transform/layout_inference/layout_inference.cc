/*!
 * \file layout_inference.cc
 * \brief infer the fragment/shared memory layout
 */

#include "ascend/op/builtin.h"
#include "ir/analyzer.h"
#include "layout/expr_utils.h"
#include "support/check.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op/region.h>
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
#include "../loop_vectorize.h"
#include "backend/common/target_utils.h"
#include "ir/alias_analysis.h"
#include "ir/annotations.h"
#include "ir/lexical_scope.h"
#include "ir/loop_annotations.h"
#include "ir/stmt_functor.h"
#include "ir/tensor_utils.h"
#include "layout_cost_model.h"
#include "parallel_loop_layout_validator.h"
#include "policy.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;
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
                 sym::AnalyzerObj *analyzer) {
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

// Applying a physical layout to a shared allocation still requires a static
// tile shape. Plain symbolic shared allocations without a layout are valid.
// Diagnose at the map boundary, before remapping or querying layout capacity.
void ValidateSharedLayoutShape(const TensorVar &buffer) {
  if (!IsSharedBuffer(buffer))
    return;
  for (const PrimExpr &extent : buffer.shape()) {
    TVM_FFI_CHECK(TryConstInt(extent).has_value(), ValueError)
        << "Shared buffer `" << buffer->name << "` has symbolic extent "
        << extent << "; applying a layout requires constant extents"
        << SpanHintSuffix(buffer->loc);
  }
}

Optional<TensorVar> FindLayoutAnchorBuffer(const Array<TensorVar> &buffers,
                                           const Layout &layout,
                                           sym::AnalyzerObj *analyzer) {
  for (const auto &buffer : buffers) {
    if (ShapesEqual(layout->InputShape(), buffer.shape(), analyzer)) {
      return buffer;
    }
  }
  return Optional<TensorVar>();
}

// A fragment layout published into the global layout map must be a closed
// function of the layout placeholders (input indices + replication). Loop
// completion can legitimately produce layouts referencing an enclosing serial
// loop var (CompleteBufferFragment on `src[i, k]` bakes in `i`); such a
// layout is meaningful only inside that loop's scope and would misdirect
// every other consumer of the buffer. The placeholders are memoized
// singletons, so pointer identity is the right membership test.
bool FragmentReferencesForeignVars(const Fragment &fragment) {
  std::unordered_set<const VarObj *> allowed;
  allowed.insert(ReplicationPlaceholder().get());
  for (int i = 0; i < static_cast<int>(fragment->InputDim()); ++i) {
    allowed.insert(InputPlaceholder(i).get());
  }
  bool foreign = false;
  auto scan = [&](const PrimExpr &expr) {
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        expr, [&](const VarObj *var) -> ffi::Expected<ffi::WalkResult> {
          if (!allowed.count(var)) {
            foreign = true;
          }

          return ffi::WalkResult::Advance();
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
using namespace prim;

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
               const std::vector<sym::Analyzer> &analyzer_vec,
               const Map<TensorVar, Layout> &annotated_layouts) {
    for (int i = 0; i < static_cast<int>(infer_list.size()); ++i) {
      const auto *finalize = infer_list[i].as<FinalizeReducerV2OpNode>();
      if (finalize == nullptr) {
        continue;
      }
      const TensorVar &dst = finalize->dst;
      if (annotated_layouts.count(dst) ||
          !FinalizeReducerV2OpNode::CanSteerDst(finalize->reducer, dst,
                                                thread_bounds_vec[i],
                                                analyzer_vec[i].get())) {
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

class TensorUseDefCollector : public StmtExprVisitorWithAnalyzer {
public:
  explicit TensorUseDefCollector(LayoutInferenceOptions options = {})
      : options_(std::move(options)) {}

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
    sym::AnalyzerObj *cur_analyzer = analyzer_vec_[cur_infer_id].get();
    // Double-check that 'next' is valid
    ICHECK(next.defined()) << "infer_list_[" << cur_infer_id
                           << "] is null inside run_infer_step.";

    // Check the logical thread index and the thread bounds extent.
    ICHECK(thread_index.defined())
        << "thread_index_vec_[" << cur_infer_id << "] is not defined.";
    ICHECK(thread_bounds.defined())
        << "thread_bounds_vec_[" << cur_infer_id << "] is not defined.";

    const auto extent_ptr = TryConstInt(thread_bounds->extent);
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
      ValidateSharedLayoutShape(buffer);

      // Gate 1 of the global map: only closed layouts may enter (see
      // IsOpenFragmentLayout).
      if (IsOpenFragmentLayout(buffer, layout)) {
        DLOG(INFO) << "[RunInferStep] dropping open layout for buffer "
                   << buffer << " from op " << cur_infer_id;
        continue;
      }

      // Helper: propagate inferred layout to alias buffers (same data Var)
      auto propagate_alias = [&](const TensorVar &src_buffer,
                                 const Layout &src_layout) {
        if (!buffer_data_to_buffers_.count(aliases_.Identity(src_buffer)))
          return;
        const auto &siblings =
            buffer_data_to_buffers_[aliases_.Identity(src_buffer)];
        for (const auto &sib : siblings) {
          if (sib.same_as(src_buffer))
            continue;
          bool shapes_equal =
              src_layout->InputShape().size() == sib.shape().size();
          if (shapes_equal) {
            for (size_t i = 0; i < src_layout->InputShape().size(); ++i) {
              if (!GetSyncedAnalyzer()->CanProveEqual(
                      src_layout->InputShape()[i], sib.shape()[i])) {
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
                        sib.shape(), GetSyncedAnalyzer(),
                        IntImm::Int32(
                            GetElementStorageBits(src_buffer.dtype())),
                        IntImm::Int32(GetElementStorageBits(sib.dtype())));
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
            ICHECK(IsZero(inner_analyzer->Simplify(
                dst_layout->InputShape()[i] - src_layout->InputShape()[i])));
            BindAnalyzer(inner_analyzer, x,
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

    // A CPU grid task owns all of its fragment elements. CPU copy/fill do
    // not propose a GPU thread distribution, and finalize's steering rule
    // intentionally applies only to a multi-participant collective. Seed
    // the ordinary one-participant layout here so fragment-only copy chains
    // and legacy reducer destinations have a physical storage contract too.
    // An explicit user layout remains authoritative and is still validated.
    if (TargetIsCPU(target_)) {
      for (const auto &[buffer, users] : use_list_) {
        if (!IsFragmentBuffer(buffer) || layout_map.count(buffer))
          continue;
        const Range bounds = Range(IntImm::Int32(0), IntImm::Int32(1));
        layout_map.Set(
            buffer, Fragment::FullyReplicated(buffer.shape(), IntImm::Int32(1))
                        ->BindThreadRange(bounds));
      }
    }

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
              if (!GetSyncedAnalyzer()->CanProveEqual(
                      rep_layout.value()->InputShape()[i], buf.shape()[i])) {
                shapes_equal = false;
                break;
              }
            }
          }
          Layout reshaped =
              shapes_equal
                  ? rep_layout.value()
                  : rep_layout.value()->Reshape(
                        buf.shape(), GetSyncedAnalyzer(),
                        IntImm::Int32(
                            GetElementStorageBits(rep.value().dtype())),
                        IntImm::Int32(GetElementStorageBits(buf.dtype())));
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
          padding_guard_map.Set(for_infer->GetRoot(), bool(true));
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

  void Collect(const tirx::Function &f) {
    aliases_.Collect(f);
    buffer_data_to_buffers_ = aliases_.groups;
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    TVM_FFI_CHECK(target.has_value(), ValueError)
        << "LayoutInference requires a target";
    target_ = *target;
    if (!(!f->GetAttr(tvm::attr::kDeclaration).value_or(false)))
      return;
    Visit(f->body);
    ApplyAnnotatedLayouts();
    ComputeFloatingFragmentBuffers(f->body);
  }

private:
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

  ffi::Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    if (op->op.as<OpObj>() && IsAnnotationCall(op)) {
      auto metadata = ReadAnnotations(op);
      // The enclosing source owner already supplied the complete merged map,
      // including declarations later in this sequence. Do not replace it with
      // one earlier declaration's partial map while visiting its payload.
      if (metadata.count("tl.source_block_annotation"))
        return std::nullopt;
      for (const auto &[key, value] : metadata)
        block_annotations_.Set(key, value);
      if (auto value = metadata.Get(attr::kLayoutMap))
        RecordLayoutMap(*value);
      return std::nullopt;
    }
    TileOperator tile = ParseOperator(GetRef<Call>(op), block_annotations_);
    if (!tile.defined())
      return StmtExprVisitorWithAnalyzer::Visit_(op);
    auto accesses = tile->GetAccessRegions();
    for (const auto &region : accesses.reads)
      addToUseList(RegionTensor(region));
    for (const auto &region : accesses.writes)
      addToUseList(RegionTensor(region));
    auto operands = DecodeTileOpArguments(op).operands;
    for (const Expr &value : operands)
      CollectFragmentBuffersFromExpr(value);
    thread_index_vec_.push_back(CurrentThreadIndex());
    thread_bounds_vec_.push_back(CurrentThreadBounds());
    analyzer_vec_.push_back(GetSyncedAnalyzer()->Clone());
    infer_list_stmt_.push_back(GetRef<Call>(op));
    infer_list_.push_back(std::move(tile));
    return std::nullopt;
  }

  void addToUseList(const TensorVar &buffer) {
    // Fragment buffers and reducers (whose PartialFragment travels the same
    // use_list_ edges: commits notify users, shared buffers union free-mode
    // components) both register; anything else has no layout to solve.
    if (!IsFragmentBuffer(buffer) && !IsReducerV2Buffer(buffer)) {
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
                     [&](const TensorVar &b) { return b.same_as(buffer); })) {
      vec.push_back(buffer);
    }
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForObj *op) final {
    if (op->kind != ForKind::kParallel)
      return StmtExprVisitorWithAnalyzer::Visit_(op);
    auto hints = ReadLoopAnnotations(op);
    for (const auto &[key, value] : loop_annotations_)
      hints.Set(key, value);
    auto infer = ParallelOp(GetRef<For>(op), hints);
    for (const TensorVar &tensor : infer->GetAccessOrder())
      addToUseList(tensor);
    const int index = static_cast<int>(infer_list_.size());
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        op->body, [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
          if (call->op.same_as(reducer_update())) {
            auto update = ParseReducerUpdate(call);
            reducer_update_sites_[update.reducer.get()].push_back(
                ReducerUpdateSiteRecord{index, update.indices, update.value});
            addToUseList(update.reducer);
          }

          return ffi::WalkResult::Advance();
        });
    infer_list_stmt_.push_back(GetRef<For>(op));
    infer_list_.push_back(std::move(infer));
    thread_index_vec_.push_back(CurrentThreadIndex());
    thread_bounds_vec_.push_back(CurrentThreadBounds());
    analyzer_vec_.push_back(GetSyncedAnalyzer()->Clone());
    return std::nullopt;
  }

  void RecordLayoutMap(const Any &value) {
    for (const auto &[tensor, layout] : value.cast<Map<Expr, Layout>>()) {
      if (auto previous = source_layouts_.Get(tensor)) {
        TVM_FFI_CHECK((*previous)->IsEqual(layout.get()), ValueError)
            << "Conflicting annotated layouts for " << tensor;
      }
      source_layouts_.Set(tensor, layout);
    }
  }

  void ApplyAnnotatedLayouts() {
    for (const auto &[var, layout] : source_layouts_) {
      Expr identity = aliases_.Identity(var);
      auto group = buffer_data_to_buffers_.Get(identity);
      TVM_FFI_CHECK(group.has_value() && !group->empty(), ValueError)
          << "Annotated tensor is not defined in the function: " << var;
      const auto &buffers = *group;
      auto anchor =
          FindLayoutAnchorBuffer(buffers, layout, GetSyncedAnalyzer());
      int64_t anchor_bits =
          GetElementStorageBits(anchor.value_or(buffers[0]).dtype());
      for (const TensorVar &buffer : buffers) {
        if (IsReducerV2Buffer(buffer)) {
          const auto *partial = layout.as<PartialFragmentNode>();
          TVM_FFI_CHECK(partial, ValueError)
              << "T.annotate_layout on a reducer requires a PartialFragment";
          TVM_FFI_CHECK(ShapesEqual(layout->InputShape(), buffer.shape(),
                                    GetSyncedAnalyzer()),
                        ValueError)
              << "PartialFragment shape must equal the reducer shape exactly";
          TVM_FFI_CHECK(partial->DetectInjective()->errors.empty(), ValueError)
              << "Reducer partial map must be injective over (thread, index)";
        }
        Layout resolved =
            ShapesEqual(layout->InputShape(), buffer.shape(),
                        GetSyncedAnalyzer())
                ? layout
                : layout->Reshape(
                      buffer.shape(), GetSyncedAnalyzer(),
                      IntImm::Int32(anchor_bits),
                      IntImm::Int32(GetElementStorageBits(buffer.dtype())));
        if (IsSharedBuffer(buffer)) {
          auto injectivity = resolved->DetectInjective();
          TVM_FFI_CHECK(injectivity->errors.empty(), ValueError)
              << "T.annotate_layout on shared buffer `" << buffer->name
              << "` must be injective: " << injectivity->errors;
          ValidateSharedLayoutShape(buffer);
        }
        annotated_layout_map_.Set(buffer, resolved);
      }
    }
  }

  ffi::Optional<VisitInterrupt> Visit_(const SeqStmtObj *op) final {
    auto previous = block_annotations_;
    auto result = StmtExprVisitorWithAnalyzer::Visit_(op);
    block_annotations_ = std::move(previous);
    return result;
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj *op) final {
    if (options_.IsOpaque(op))
      return std::nullopt;
    auto previous_thread = thread_binding_;
    auto previous_loop = loop_annotations_;
    auto previous_block = block_annotations_;
    if (op->op.same_as(launch_thread_op()) &&
        op->args[0].as_or_throw<StringImm>()->value == "threadIdx.x")
      thread_binding_ = op->body_params[0].as_or_throw<PrimVar>();
    if (ascend::IsAscendBlock(op) || IsLexicalScope(op)) {
      auto metadata = CollectSourceBlockAnnotations(op->body);
      for (const auto &[name, value] : metadata)
        block_annotations_.Set(name, value);
      if (auto value = metadata.Get(attr::kLayoutMap))
        RecordLayoutMap(*value);
    }
    if (op->op->name == "tl.attr") {
      auto fields = ReadAnnotations(op);
      String key = fields.at("key").cast<String>();
      if (key == "tl.loop.annotations")
        loop_annotations_ = fields.at("value").cast<Map<String, Any>>();
      else if (key == attr::kLayoutMap)
        RecordLayoutMap(fields.at("value"));
      else if (key == "tl.kernel")
        for (const auto &[name, value] :
             fields.at("value").cast<Map<String, Any>>())
          block_annotations_.Set(name, value);
      else
        block_annotations_.Set(key, fields.at("value"));
      if (key == "tl.kernel") {
        const auto metadata = CollectSourceBlockAnnotations(op->body);
        for (const auto &[name, value] : metadata)
          block_annotations_.Set(name, value);
        if (auto value = metadata.Get(attr::kLayoutMap))
          RecordLayoutMap(*value);
      }
    }
    auto result = StmtExprVisitorWithAnalyzer::Visit_(op);
    thread_binding_ = previous_thread;
    loop_annotations_ = std::move(previous_loop);
    block_annotations_ = std::move(previous_block);
    return result;
  }

  ffi::Optional<VisitInterrupt> Visit_(const BindObj *op) final {
    bind_var_to_expr_.Set(op->var, op->value);
    return StmtExprVisitorWithAnalyzer::Visit_(op);
  }

  void CollectFragmentBuffersFromExpr(const Expr &expr) {
    std::unordered_set<const VarObj *> expanded;
    std::function<void(const Expr &)> collect = [&](const Expr &value) {
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
          value, [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
            if (const auto *load = node.as<TensorLoadObj>())
              addToUseList(load->source.as_or_throw<TensorVar>());
            else if (const auto *variable = node.as<VarObj>()) {
              if (!expanded.insert(variable).second)
                return ffi::WalkResult::Advance();
              if (auto bound = bind_var_to_expr_.Get(GetRef<Var>(variable)))
                collect(*bound);
            }
            return ffi::WalkResult::Advance();
          });
    };
    collect(expr);
  }

  Range CurrentThreadBounds() const {
    return ComputeThreadBounds(thread_binding_, GetAnalyzer());
  }
  PrimExpr CurrentThreadIndex() const {
    return thread_binding_.has_value() ? PrimExpr(*thread_binding_)
                                       : PrimExpr(0);
  }

  void ComputeFloatingFragmentBuffers(const Stmt &body) {
    std::unordered_set<const Object *> contained;
    for (const auto &op : infer_list_stmt_)
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
          op, [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
            contained.insert(node.get());
            return ffi::WalkResult::Advance();
          });
    class Collector : public StmtExprVisitorWithAnalyzer {
    public:
      Collector(const std::unordered_set<const Object *> &contained,
                std::unordered_map<TensorVar, Range, ObjectPtrHash,
                                   ObjectPtrEqual> &result,
                const LayoutInferenceOptions &options)
          : contained_(contained), result_(result), options_(options) {}
      ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj *op) final {
        if (options_.IsOpaque(op))
          return std::nullopt;
        auto previous = thread_;
        if (op->op.same_as(launch_thread_op()) &&
            op->args[0].as_or_throw<StringImm>()->value == "threadIdx.x")
          thread_ = op->body_params[0].as_or_throw<PrimVar>();
        auto result = StmtExprVisitorWithAnalyzer::Visit_(op);
        thread_ = previous;
        return result;
      }
      ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj *op) final {
        Record(op->source.as_or_throw<TensorVar>(), op);
        return StmtExprVisitorWithAnalyzer::Visit_(op);
      }
      ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj *op) final {
        Record(op->dest.as_or_throw<TensorVar>(), op);
        return StmtExprVisitorWithAnalyzer::Visit_(op);
      }
      ffi::Optional<VisitInterrupt> Visit_(const CallObj *op) final {
        if (op->op.as<OpObj>() && IsAnnotationCall(op))
          return std::nullopt;
        return StmtExprVisitorWithAnalyzer::Visit_(op);
      }

    private:
      void Record(const TensorVar &tensor, const Object *node) {
        if (!IsFragmentBuffer(tensor) || contained_.count(node))
          return;
        Range range = ComputeThreadBounds(thread_, GetAnalyzer());
        if (auto found = result_.find(tensor); found != result_.end()) {
          auto previous = found->second;
          PrimExpr begin =
              GetSyncedAnalyzer()->Simplify(min(previous->begin, range->begin));
          PrimExpr end = GetSyncedAnalyzer()->Simplify(
              max(previous->begin + previous->extent,
                  range->begin + range->extent));
          found->second = Range(begin, end);
        } else
          result_.emplace(tensor, range);
      }
      const std::unordered_set<const Object *> &contained_;
      std::unordered_map<TensorVar, Range, ObjectPtrHash, ObjectPtrEqual>
          &result_;
      ffi::Optional<PrimVar> thread_;
      const LayoutInferenceOptions &options_;
    };
    auto collector =
        make_object<Collector>(contained, floating_fragment_buffers_, options_);
    collector->Visit(body);
  }

  LayoutInferenceOptions options_;
  TensorAliasGroups aliases_;
  Map<Expr, Array<TensorVar>> buffer_data_to_buffers_;
  Map<Expr, Layout> source_layouts_;
  Map<Var, Expr> bind_var_to_expr_;
  Map<String, Any> block_annotations_;
  Map<String, Any> loop_annotations_;
  std::vector<ObjectRef> infer_list_stmt_;
  std::vector<TileOperator> infer_list_;
  // Fragment buffers that have accesses outside of TileOps.
  // These "floating" buffers need fully replicated layouts since their
  // access patterns cannot be inferred from TileOp semantics.
  // Maps buffer -> thread_bounds at the point of floating access.
  // See ComputeFloatingFragmentBuffers() for detailed explanation.
  std::unordered_map<TensorVar, Range, ObjectPtrHash, ObjectPtrEqual>
      floating_fragment_buffers_;
  std::unordered_map<TensorVar, std::vector<int>, ObjectPtrHash, ObjectPtrEqual>
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
  std::unordered_map<const VarObj *, std::vector<ReducerUpdateSiteRecord>>
      reducer_update_sites_;
  // All reducer dst-steering knowledge, behind one interface (see the class
  // comment). Populated once in Run(), consulted at the commit point and by
  // the free-mode attempt runner.
  ReducerDstSteering steering_;
  // Per-op list of buffers it touches (fragment scope), used for prioritization
  std::unordered_map<int, std::vector<TensorVar>> op_touched_buffers_;
  // Real threadIdx.x binding of the enclosing thread_extent scope, when one
  // exists. Stays undefined for targets without thread bindings (e.g. CPU),
  // where the logical thread index is the constant 0 and thread bounds are
  // [0, 1) — no synthetic fallback Var is ever created.
  ffi::Optional<PrimVar> thread_binding_;
  std::vector<PrimExpr> thread_index_vec_;
  std::vector<Range> thread_bounds_vec_;
  std::vector<sym::Analyzer> analyzer_vec_;
  Target target_{ffi::UnsafeInit{}};
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
      const LayoutCostModel &cost_model, int candidate_vector_size_limit = 0,
      std::string *failure_message = nullptr) {
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
      DLOG(INFO) << "[InferInFreeMode] attempt root " << attempt_root
                 << " discarded: " << failure;
      if (failure_message != nullptr) {
        *failure_message = failure;
      }
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
    // aliases_.Identity(buffer) (Var). This handles cases like reshape where
    // multiple TensorVar objects alias the same storage.
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
      // Keep the caught cause local to this component. Retrying a candidate
      // must not erase the original operator's useful diagnostic when every
      // candidate fails, or inherit an unrelated cause from an earlier group.
      std::string last_failure;

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
        bool try_scalar = cost_model->ExploreReducerScalarLayouts() && loop &&
                          loop->HasReducerUpdates() &&
                          !loop->GetLoopLayout().defined() &&
                          !loop->annotated_layout_unbound_.has_value() &&
                          !loop->GetHints().count(attr::kCoalescedWidth);
        for (int candidate_vector_size_limit : {0, 1}) {
          if (candidate_vector_size_limit != 0 && !try_scalar) {
            continue;
          }
          DLOG(INFO) << "----------------------- try root "
                     << attempt_infer_root << " members " << members.size()
                     << " candidate_vector_size_limit="
                     << candidate_vector_size_limit << '\n';
          auto outcome = RunOneAttempt(
              attempt_infer_root, members, layout_map, strict_layout_map,
              /*seed_layouts=*/{}, *cost_model, candidate_vector_size_limit,
              &last_failure);
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
        }
      }
      if (!has_best) {
        // A vectorized copy can induce a sparse reducer owner map whose
        // rectangular inverse is unavailable. Try scalar copy ownership only
        // after every normal plan failed; all injectivity, owner and lowering
        // checks still apply. Explicit layout/width hints stay authoritative.
        for (int attempt_root : members) {
          const auto *copy = infer_list_[attempt_root].as<CopyNode>();
          if (!copy || copy->annotations.count(attr::kParallelLoopLayout) ||
              copy->annotations.count(attr::kCoalescedWidth))
            continue;
          auto outcome = RunOneAttempt(
              attempt_root, members, layout_map, strict_layout_map,
              /*seed_layouts=*/{}, *cost_model,
              /*candidate_vector_size_limit=*/1, &last_failure);
          if (outcome && (!has_best || outcome->cost.BetterThan(best_cost)))
            adopt(std::move(*outcome), attempt_root);
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
                            strict_layout_map, seeds, *cost_model,
                            /*candidate_vector_size_limit=*/0, &last_failure);
          if (outcome) {
            adopt(std::move(*outcome), members.front());
          }
        }
      }
      ICHECK(has_best) << "no available layout found" << '\n' << last_failure;
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

class LayoutInferencer : public StmtExprMutatorWithAnalyzer {
public:
  LayoutInferencer(LayoutInferenceResult result, sym::AnalyzerObj *analyzer,
                   LayoutInferenceOptions options)
      : StmtExprMutatorWithAnalyzer(analyzer), result_(std::move(result)),
        options_(std::move(options)) {}
  using StmtExprMutatorWithAnalyzer::Mutate_;

  static tirx::Function Apply(tirx::Function function,
                              const LayoutInferenceOptions &options) {
    if (!(!function->GetAttr(tvm::attr::kDeclaration).value_or(false)))
      return function;
    function.CopyOnWrite()->body =
        SeqStmt(ParallelLoopFuser::Fuse(function->body, options));
    auto collector = make_object<TensorUseDefCollector>(options);
    collector->Collect(function);
    auto result = collector->Run();
    sym::Analyzer analyzer;
    auto rewriter =
        make_object<LayoutInferencer>(result, analyzer.get(), options);
    SeqStmt body =
        rewriter->Mutate(function->body).ValueOrUnchanged(function->body);
    LayoutMap param_layouts;
    for (const Var &param : function->params)
      if (auto tensor = param.as<TensorVar>())
        if (auto layout = result.layout_map.Get(*tensor))
          param_layouts.Set(*tensor, *layout);
    if (!param_layouts.empty())
      body = SeqStmt(
          {MakeAnnotation({{attr::kLayoutMap, param_layouts}}, function->loc),
           body});
    function.CopyOnWrite()->body = body;
    ParallelLoopLayoutValidator::Validate(body, options);
    return function;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    if (options_.IsOpaque(op))
      return ffi::Unchanged();
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }

  UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    Stmt result =
        StmtExprMutatorWithAnalyzer::Mutate_(op, mode).ValueOrUnchanged(
            GetRef<Stmt>(op));
    if (auto tensor = op->var.as<TensorVar>()) {
      if (auto layout = result_.layout_map.Get(*tensor))
        return SeqStmt(
            {result,
             MakeAnnotation({{attr::kLayoutMap, LayoutMap{{*tensor, *layout}}}},
                            op->loc)});
    }
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    if (const auto *call = op->value.as<CallObj>()) {
      if (call->op.as<OpObj>() && IsAnnotationCall(call)) {
        auto metadata = ReadAnnotations(call);
        metadata.erase(attr::kLayoutMap);
        return metadata.empty() ? Stmt(SeqStmt(ffi::Array<Stmt>{}))
                                : MakeAnnotation(metadata, op->loc);
      }
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }

  UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    auto result =
        StmtExprMutatorWithAnalyzer::Mutate_(op, mode).ValueOrUnchanged(
            GetRef<Stmt>(op));
    For original = GetRef<For>(op);
    auto layout = result_.for_map.Get(original);
    if (!layout.has_value())
      return result;
    Map<String, Any> metadata{{attr::kParallelLoopLayout, *layout}};
    if (auto padding = result_.padding_guard_map.Get(original))
      metadata.Set(attr::kParallelLoopRequiresPaddingGuard, *padding);
    if (auto predicate = result_.predicate_map.Get(original)) {
      PrimExpr value = GetSyncedAnalyzer()->Simplify(*predicate);
      if (!IsConstInt(value, 1))
        metadata.Set(attr::kParallelLoopPredicate, value);
    }
    return MakeAttribute(nullptr, "tl.loop.annotations", metadata,
                         SeqStmt(result), op->loc);
  }

private:
  LayoutInferenceResult result_;
  LayoutInferenceOptions options_;
};

tvm::transform::Pass LayoutInferenceWithPolicy(LayoutInferenceOptions options) {
  auto pass = [options = std::move(options)](tirx::Function function, IRModule,
                                             tvm::transform::PassContext) {
    VectorizeMemoryScope memory_scope(function);
    return LayoutInferencer::Apply(std::move(function), options);
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LayoutInference");
}

tvm::transform::Pass LayoutInference() { return LayoutInferenceWithPolicy({}); }

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LayoutInference",
                                   LayoutInference);
}

} // namespace tl
} // namespace tvm
