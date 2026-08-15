/*!
 * \file reducer_plan_materialize.cc
 * \brief Plan physical storage/communication for reducer v2 epochs and
 *        materialize the first-class reducer ops into ordinary IR.
 *
 * Runs after LayoutInference and before LowerTileOp. Loop layouts are frozen
 * at this point: this pass reads them (and the thread bounds) but never
 * writes loop layouts, predicates or other buffers' layouts. It decides
 * three things only: where partials live (storage), how participants
 * communicate (finalize plan), and how many times each update executes
 * (multiplicity markers).
 *
 * Two physical plans exist, chosen per epoch:
 *
 * Narrow plan (optimization; every proof failure falls back to wide):
 *   The reduction axes of an update site are the parallel loop vars that do
 *   not appear in the update indices, plus the loop's replication. Projecting
 *   them out of the site's loop layout yields the induced partial layout:
 *   compact per-thread storage where every replica executes its update
 *   unguarded, so each partial accumulates exactly the contributions of the
 *   iterations mapped to its thread. The finalize collective combines only
 *   the thread-expression splits sourced from reduction axes (extracted with
 *   the same machinery T.reduce uses); splits from loop replication become
 *   value replication and are not reduced. Zero splits means zero
 *   collectives (the LocalComplete case). Proof obligations: direct
 *   var-to-dim update indices, replica-safe contribution values, full-block
 *   thread coverage, a single power-of-two reduce step, structurally equal
 *   plans across all update sites, and destination-layout containment.
 *
 * Wide plan (canonical FullParticipant baseline; always available):
 *   One full logical-shape partial per participant (FullyReplicated
 *   storage), updates guarded to the canonical replica via the generic
 *   `tl.parallel_multiplicity` marker, participant-wide AllReduce per
 *   logical output, then a copy into the destination fragment.
 *
 * Common rules: partials initialize to the combine identity; an optional
 * seed is combined exactly once per logical output after the collective.
 * The pass config `tl.reducer_force_baseline` disables narrow plans for
 * differential testing.
 */

#include "support/check.h"
#include <tvm/arith/iter_affine_map.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../config.h"
#include "../layout/layout.h"
#include "../layout/utils.h"
#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/fill.h"
#include "../op/reducer.h"
#include "../op/utils.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "backend/common/op/reduce.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
using arith::IRMutatorWithAnalyzer;
using arith::IRVisitorWithAnalyzer;

namespace {

/*! \brief Build a tl.region(...) call over the full extent of `buffer`. */
PrimExpr MakeFullRegion(const Buffer &buffer, int access_mask) {
  Array<PrimExpr> zeros;
  zeros.reserve(buffer->shape.size());
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    zeros.push_back(make_zero(DataType::Int(32)));
  }
  Array<PrimExpr> args;
  args.push_back(BufferLoad(buffer, zeros));
  args.push_back(IntImm(DataType::Int(32), access_mask));
  for (const auto &extent : buffer->shape) {
    args.push_back(extent);
  }
  return Call(DataType::Handle(), region(), args);
}

// ---------------------------------------------------------------------------
// Phase A: epoch collection
// ---------------------------------------------------------------------------

struct UpdateSite {
  Fragment loop_layout;    // solved layout of the enclosing parallel nest
  Array<Var> loop_vars;    // nest loop vars in order
  Array<PrimExpr> indices; // logical output indices of the update target
  PrimExpr value;          // contribution expression
  // Serial loops between the parallel nest and the update (outermost
  // first). They accumulate on one thread; the packed-accumulation
  // optimization uses the innermost one as its lane source.
  Array<Var> serial_vars;
  Array<PrimExpr> serial_extents;
};

struct EpochInfo {
  Buffer buffer;
  ReducerV2OpType op{ReducerV2OpType::kSum};
  Optional<PrimExpr> seed;
  int64_t thread_extent{-1}; // participant extent at the init site
  int64_t thread_min{0};
  std::vector<UpdateSite> updates;
  Buffer dst;
  int64_t batch{1};
  // False when some structural prerequisite for plan analysis is missing
  // (e.g. an update site without a solved loop layout).
  bool analyzable{true};
};

class ReducerEpochCollector : public IRVisitorWithAnalyzer {
public:
  std::unordered_map<const VarNode *, EpochInfo> epochs_;

  void Collect(const PrimFunc &f) { VisitStmt(f->body); }

private:
  void VisitStmt_(const SBlockNode *op) final {
    if (auto anno = op->annotations.Get(attr::kReducerInfoV2)) {
      auto map = anno.value().as<Map<Var, Map<String, Any>>>();
      ICHECK(map) << "malformed reducer_info_v2 annotation";
      for (const auto &[var, info] : map.value()) {
        EpochInfo &epoch = epochs_[var.get()];
        epoch.op = ParseReducerV2OpType(info.Get("op").value().cast<String>());
      }
    }
    for (const auto &buffer : op->alloc_buffers) {
      if (IsReducerV2Buffer(buffer)) {
        epochs_[buffer->data.get()].buffer = buffer;
      }
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    auto prev_thread_var = thread_var_;
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        thread_var_ = iv;
      }
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
    thread_var_ = prev_thread_var;
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel &&
        op->annotations.count(tl::attr::kParallelLoopLayout)) {
      auto prev_layout = cur_loop_layout_;
      auto prev_vars = cur_loop_vars_;
      cur_loop_layout_ = Downcast<Fragment>(
          op->annotations.Get(tl::attr::kParallelLoopLayout).value());
      // Gather the consecutive parallel nest the annotation covers.
      Array<Var> vars;
      const ForNode *cur = op;
      while (true) {
        vars.push_back(cur->loop_var);
        const auto *inner = cur->body.as<ForNode>();
        if (inner != nullptr && inner->kind == ForKind::kParallel) {
          cur = inner;
        } else {
          break;
        }
      }
      cur_loop_vars_ = vars;
      auto prev_serial_vars = cur_serial_vars_;
      auto prev_serial_extents = cur_serial_extents_;
      cur_serial_vars_.clear();
      cur_serial_extents_.clear();
      IRVisitorWithAnalyzer::VisitStmt_(op);
      cur_loop_layout_ = prev_layout;
      cur_loop_vars_ = prev_vars;
      cur_serial_vars_ = prev_serial_vars;
      cur_serial_extents_ = prev_serial_extents;
      return;
    }
    if (op->kind == ForKind::kSerial && cur_loop_layout_.defined()) {
      cur_serial_vars_.push_back(op->loop_var);
      cur_serial_extents_.push_back(op->extent);
      IRVisitorWithAnalyzer::VisitStmt_(op);
      cur_serial_vars_.pop_back();
      cur_serial_extents_.pop_back();
      return;
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(ReducerInitOp::Get())) {
      if (EpochInfo *epoch = FindEpoch(op->args[0])) {
        if (op->args.size() >= 2) {
          epoch->seed = op->args[1];
        }
        if (thread_var_.defined() &&
            analyzer_.const_int_bound.IsBound(thread_var_->var)) {
          auto bound = analyzer_.const_int_bound(thread_var_->var);
          epoch->thread_min = bound->min_value;
          epoch->thread_extent = bound->max_value - bound->min_value + 1;
        } else {
          epoch->analyzable = false;
        }
      }
      return;
    }
    if (op->op.same_as(reducer_update())) {
      const auto *load = op->args[0].as<BufferLoadNode>();
      auto it = load != nullptr ? epochs_.find(load->buffer->data.get())
                                : epochs_.end();
      if (it != epochs_.end()) {
        EpochInfo *epoch = &it->second;
        if (cur_loop_layout_.defined()) {
          ReducerUpdateArgs update = ParseReducerUpdate(op);
          epoch->updates.push_back(UpdateSite{
              cur_loop_layout_.value(), cur_loop_vars_, update.indices,
              update.value, cur_serial_vars_, cur_serial_extents_});
        } else {
          epoch->analyzable = false;
        }
      }
      return;
    }
    if (op->op.same_as(FinalizeReducerV2Op::Get())) {
      if (EpochInfo *epoch = FindEpoch(op->args[0])) {
        if (auto call2 = op->args[1].as<CallNode>()) {
          if (call2->op.same_as(region())) {
            if (auto load = call2->args[0].as<BufferLoadNode>()) {
              epoch->dst = load->buffer;
            }
          }
        }
        if (auto batch = op->annotations.Get("batch")) {
          if (const auto *imm = batch.value().as<IntImmNode>()) {
            epoch->batch = imm->value;
          }
        }
      }
      return;
    }
    IRVisitorWithAnalyzer::VisitExpr_(op);
  }

  EpochInfo *FindEpoch(const PrimExpr &region_arg) {
    if (auto call = region_arg.as<CallNode>()) {
      if (call->op.same_as(region())) {
        if (auto load = call->args[0].as<BufferLoadNode>()) {
          auto it = epochs_.find(load->buffer->data.get());
          if (it != epochs_.end()) {
            return &it->second;
          }
        }
      }
    }
    return nullptr;
  }

  IterVar thread_var_;
  Optional<Fragment> cur_loop_layout_;
  Array<Var> cur_loop_vars_;
  Array<Var> cur_serial_vars_;
  Array<PrimExpr> cur_serial_extents_;
};

// ---------------------------------------------------------------------------
// Phase B: narrow-plan analysis
// ---------------------------------------------------------------------------

struct NarrowDecision {
  Fragment storage_layout; // induced partial layout (also post-collective)
  // Collective steps: (reducing_threads, scale) per thread-expression split
  // sourced from a reduction axis. Empty = no communication (LocalComplete).
  std::vector<std::pair<int, int>> steps;
  // True when the destination's free-level inferred layout is replaced by
  // the induced layout (legal only for unconstrained destinations); the
  // chain lists further downstream fragments (fp32->fp16 staging hops etc.)
  // that must be overridden together so the connecting copies stay
  // slot-compatible.
  bool override_dst_layout{false};
  std::vector<Buffer> override_chain;
  // Packed partial accumulation (16-bit floats): updates write two
  // interleaved lanes per logical slot selected by the parity of
  // `pack_lane_var` (the innermost on-thread reduction loop), which breaks
  // the serial combine dependence chain and lets the vectorizer emit
  // paired 16-bit (half2-style) operations. A fold loop combines the lanes
  // into the plain induced storage before the collective, so the plan's
  // communication and destination proofs are untouched.
  bool packed{false};
  Var pack_lane_var;
  Fragment packed_layout; // storage_layout with an extra innermost lane dim
};

/*! \brief Per-buffer access census plus the copy graph, used to decide
 *  whether a finalize destination's inferred layout may be replaced. A
 *  destination is unconstrained when it is written only by the finalize and
 *  every read is the source of a copy into global memory or into another
 *  fragment that is itself unconstrained (written only by that copy): such
 *  uses lower against whatever layout the buffers have, and the free-level
 *  layouts LayoutInference picked for them never constrained anything
 *  else. */
struct BufferUseCensus {
  int64_t loads{0};        // every BufferLoad occurrence (incl. regions)
  int64_t stores{0};       // ordinary stores (must be zero)
  int64_t finalize_dst{0}; // finalize_reducer destination regions
  int64_t copy_src{0};     // tl.copy source regions
  int64_t copy_dst{0};     // tl.copy destination regions
};

struct CopyEdge {
  const VarNode *src{nullptr};
  const VarNode *dst{nullptr};
  bool dst_global{false};
  bool dst_fragment{false};
  Buffer dst_buffer;
};

struct UseGraph {
  std::unordered_map<const VarNode *, BufferUseCensus> census;
  std::vector<CopyEdge> copies;
};

/*! \brief Check the destination override chain rooted at `root` and collect
 *  the downstream fragments (excluding the root) that must be overridden
 *  with it. Returns false when any buffer on the chain has uses beyond
 *  "written once by its producer, read only by copies to global or further
 *  chain members". */
bool CollectOverrideChain(const VarNode *root, const UseGraph &graph,
                          std::vector<Buffer> *chain) {
  std::unordered_set<const VarNode *> visited;
  std::function<bool(const VarNode *)> visit = [&](const VarNode *var) -> bool {
    if (!visited.insert(var).second) {
      return false; // cycle
    }
    auto it = graph.census.find(var);
    BufferUseCensus c =
        it == graph.census.end() ? BufferUseCensus{} : it->second;
    bool is_root = (var == root);
    if (c.stores != 0) {
      return false;
    }
    if (is_root ? (c.finalize_dst != 1 || c.copy_dst != 0)
                : (c.finalize_dst != 0 || c.copy_dst != 1)) {
      return false;
    }
    // Every load occurrence must be accounted for by the uses above.
    if (c.loads != c.copy_src + c.copy_dst + (is_root ? c.finalize_dst : 0)) {
      return false;
    }
    for (const CopyEdge &edge : graph.copies) {
      if (edge.src != var) {
        continue;
      }
      if (edge.dst_global) {
        continue;
      }
      if (!edge.dst_fragment || !edge.dst_buffer.defined()) {
        return false;
      }
      if (!visit(edge.dst)) {
        return false;
      }
      chain->push_back(edge.dst_buffer);
    }
    return true;
  };
  return visit(root);
}

bool IsPowerOfTwo(int64_t x) { return x > 0 && (x & (x - 1)) == 0; }

/*! \brief A contribution is replica-safe when every physical execution of
 *  the same logical iteration computes the same value: pure expressions of
 *  loop vars plus loads from buffers whose replicas are value-equal by
 *  contract (fragments, validated against the loop layout by inference) or
 *  uniform by address (shared/global). Plain local reads and side effects
 *  disqualify the site. */
bool ValueIsReplicaSafe(const PrimExpr &value) {
  if (SideEffect(value) > CallEffectKind::kReadState) {
    return false;
  }
  bool safe = true;
  PostOrderVisit(value, [&](const ObjectRef &obj) {
    if (const auto *load = obj.as<BufferLoadNode>()) {
      const Buffer &buffer = load->buffer;
      if (!IsFragmentBuffer(buffer) && !IsSharedBuffer(buffer) &&
          !IsGlobalBuffer(buffer)) {
        safe = false;
      }
    }
  });
  return safe;
}

std::optional<NarrowDecision>
TryNarrowPlan(const EpochInfo &epoch, const Map<Buffer, Layout> &known_layouts,
              const UseGraph &use_graph, arith::Analyzer *analyzer,
              std::string *reason) {
  auto fail = [&](const std::string &why) -> std::optional<NarrowDecision> {
    *reason = why;
    return std::nullopt;
  };
  if (!epoch.analyzable) {
    return fail("incomplete epoch structure");
  }
  if (epoch.updates.empty()) {
    return fail("no update sites");
  }
  if (epoch.thread_extent <= 0) {
    return fail("unknown participant extent");
  }
  const Buffer &buffer = epoch.buffer;

  NarrowDecision decision;
  bool first_site = true;
  for (const UpdateSite &site : epoch.updates) {
    size_t ndim = site.loop_vars.size();
    if (site.loop_layout->InputDim() != ndim) {
      return fail("loop layout rank does not match the parallel nest");
    }
    if (site.indices.size() != buffer->shape.size()) {
      return fail("update index rank does not match the reducer shape");
    }

    // Map update indices to loop dims: each index must be a distinct nest
    // var (in any order — direct identity ownership up to permutation), or
    // a constant zero on a unit reducer dim.
    std::vector<bool> is_output_dim(ndim, false);
    // acc dim -> loop dim it is driven by, or -1 for a constant unit dim.
    std::vector<int> acc_dim_to_loop_dim(site.indices.size(), -1);
    for (size_t d = 0; d < site.indices.size(); ++d) {
      const PrimExpr &index = site.indices[d];
      if (const auto *var = index.as<VarNode>()) {
        int pos = -1;
        for (size_t i = 0; i < ndim; ++i) {
          if (site.loop_vars[i].get() == var) {
            pos = static_cast<int>(i);
            break;
          }
        }
        if (pos < 0 || is_output_dim[pos]) {
          return fail("update index is not a distinct parallel loop var");
        }
        const int64_t *loop_extent =
            as_const_int(site.loop_layout->InputShape()[pos]);
        const int64_t *dim_extent = as_const_int(buffer->shape[d]);
        if (!loop_extent || !dim_extent || *loop_extent != *dim_extent) {
          return fail("loop extent does not match the reducer dim extent");
        }
        is_output_dim[pos] = true;
        acc_dim_to_loop_dim[d] = pos;
      } else if (is_zero(index)) {
        const int64_t *dim_extent = as_const_int(buffer->shape[d]);
        if (!dim_extent || *dim_extent != 1) {
          return fail("constant update index on a non-unit reducer dim");
        }
      } else {
        return fail("unsupported update index expression");
      }
    }

    // Full-block coverage keeps the collective groups and any garbage
    // threads self-contained and the barrier uniform.
    const int64_t *layout_threads =
        as_const_int(site.loop_layout->ThreadExtent());
    if (!layout_threads || *layout_threads != epoch.thread_extent) {
      return fail("loop layout does not cover the full participant extent");
    }
    if (site.loop_layout->ThreadRange().defined()) {
      const int64_t *range_min =
          as_const_int(site.loop_layout->ThreadRange()->min);
      if (!range_min || *range_min != epoch.thread_min) {
        return fail("loop layout thread range mismatch");
      }
    } else if (epoch.thread_min != 0) {
      return fail("loop layout thread range mismatch");
    }

    if (!ValueIsReplicaSafe(site.value)) {
      return fail("contribution value is not replica-safe");
    }

    // Induced partial layout: project every reduction dim (descending, so
    // dim numbers stay stable while dims are removed). Its input dims are
    // the surviving loop dims in NEST order.
    Fragment induced = site.loop_layout;
    for (int dim = static_cast<int>(ndim) - 1; dim >= 0; --dim) {
      if (!is_output_dim[dim]) {
        induced = backend::reduce::ComputeReducerLayout(induced, dim);
      }
    }
    // Rebuild the fragment over the reducer's own dim order: permuted
    // indices reorder the inputs, and constant unit dims insert inputs the
    // forward expressions never reference. `nest_rank[p]` is the position
    // of loop dim p among the surviving dims (= its input slot in
    // `induced`); feed each such slot the placeholder of the acc dim it
    // drives.
    {
      std::vector<int> nest_rank(ndim, -1);
      int rank = 0;
      for (size_t p = 0; p < ndim; ++p) {
        if (is_output_dim[p]) {
          nest_rank[p] = rank++;
        }
      }
      // When every dim is projected (all-constant indices),
      // ComputeReducerLayout keeps one synthetic unit input.
      bool synthetic_unit = (rank == 0);
      size_t expected_rank = synthetic_unit ? 1 : static_cast<size_t>(rank);
      if (expected_rank != induced->InputShape().size()) {
        return fail("induced layout rank mismatch");
      }
      std::vector<PrimExpr> slot_placeholders(expected_rank, PrimExpr());
      bool identity = (expected_rank == buffer->shape.size());
      if (synthetic_unit) {
        // The synthetic slot is never referenced by the forward exprs; feed
        // it the first reducer-dim placeholder for the (rare) rebuild.
        slot_placeholders[0] = InputPlaceholder(0);
      }
      for (size_t d = 0; d < acc_dim_to_loop_dim.size(); ++d) {
        int p = acc_dim_to_loop_dim[d];
        if (p < 0) {
          continue; // constant unit dim: no slot to feed
        }
        slot_placeholders[nest_rank[p]] = InputPlaceholder(d);
        if (nest_rank[p] != static_cast<int>(d)) {
          identity = false;
        }
      }
      if (!identity) {
        Array<PrimExpr> slot_args(slot_placeholders.begin(),
                                  slot_placeholders.end());
        Array<PrimExpr> fwd_index = induced->Forward(slot_args);
        PrimExpr fwd_thread =
            induced->ForwardThread(slot_args, ReplicationPlaceholder());
        induced = Fragment(buffer->shape, fwd_index, fwd_thread,
                           induced->ReplicateExtent(), std::nullopt)
                      ->BindThreadRange(site.loop_layout->ThreadRange());
      }
    }
    if (induced->InputShape().size() != buffer->shape.size()) {
      return fail("induced layout rank mismatch");
    }
    for (size_t d = 0; d < buffer->shape.size(); ++d) {
      if (!analyzer->CanProveEqual(induced->InputShape()[d],
                                   buffer->shape[d])) {
        return fail("induced layout shape mismatch");
      }
    }

    // Collective steps: only thread-expression splits sourced from
    // reduction vars are reduced. Splits from loop replication become value
    // replication; reduction vars absent from the thread expression
    // accumulate serially on one thread and need no communication.
    Map<Var, Range> var_ranges;
    for (size_t i = 0; i < ndim; ++i) {
      var_ranges.Set(InputPlaceholder(i),
                     Range::FromMinExtent(make_zero(DataType::Int(32)),
                                          site.loop_layout->InputShape()[i]));
    }
    var_ranges.Set(ReplicationPlaceholder(),
                   Range::FromMinExtent(make_zero(DataType::Int(32)),
                                        site.loop_layout->ReplicateExtent()));
    auto iter_sum = arith::NormalizeToIterSum(
        site.loop_layout->GetForwardThread(), var_ranges, analyzer);
    std::vector<backend::reduce::ThreadReduceStep> steps;
    for (size_t i = 0; i < ndim; ++i) {
      if (is_output_dim[i]) {
        continue;
      }
      auto var_steps = backend::reduce::CollectThreadReduceSteps(
          iter_sum, Downcast<Var>(InputPlaceholder(i)));
      steps.insert(steps.end(), var_steps.begin(), var_steps.end());
    }
    std::vector<std::pair<int, int>> site_steps;
    for (const auto &step : steps) {
      if (!IsPowerOfTwo(step.extent)) {
        return fail("collective width is not a power of two");
      }
      int reducing_threads = step.ReducingThreads();
      if (reducing_threads > epoch.thread_extent) {
        return fail("collective width exceeds the participant extent");
      }
      site_steps.emplace_back(reducing_threads, step.scale);
    }

    if (first_site) {
      decision.storage_layout = induced;
      decision.steps = std::move(site_steps);
      first_site = false;

      // Packed partial accumulation (single-site 16-bit epochs only): pick
      // a lane source whose iterations accumulate serially on one thread.
      // Prefer an enclosing serial loop; otherwise the innermost parallel
      // reduction dim when the layout leaves it an even on-thread run.
      // Lane assignment never changes which thread a contribution lands
      // on, so this is profit-only: any layout keeps the plan correct.
      if (epoch.updates.size() == 1 &&
          (buffer->dtype.is_float16() || buffer->dtype.is_bfloat16()) &&
          induced->OutputDim() == 1) {
        // NOTE: `tirx::Var()` default-constructs a REAL variable named "v",
        // not a null ref; use Optional to represent "no lane source found".
        Optional<Var> lane_var;
        if (!site.serial_vars.empty()) {
          const int64_t *extent =
              as_const_int(site.serial_extents[site.serial_extents.size() - 1]);
          if (extent && *extent >= 2 && *extent % 2 == 0) {
            lane_var = site.serial_vars[site.serial_vars.size() - 1];
          }
        }
        if (!lane_var.has_value() && ndim > 0 && !is_output_dim[ndim - 1]) {
          const int64_t *extent =
              as_const_int(site.loop_layout->InputShape()[ndim - 1]);
          if (extent) {
            int64_t thread_multiplicity = 1;
            // Packing pays off only when the dim's LOW BIT stays on one
            // thread: a thread-expression split at an odd lower_factor
            // pins the lane parity per thread, leaving one lane idle.
            bool bit0_serial = true;
            auto inner_steps = backend::reduce::CollectThreadReduceSteps(
                iter_sum, Downcast<Var>(InputPlaceholder(ndim - 1)));
            for (const auto &step : inner_steps) {
              thread_multiplicity *= step.extent;
              if (step.lower_factor % 2 != 0) {
                bit0_serial = false;
              }
            }
            int64_t run = *extent / thread_multiplicity;
            if (bit0_serial && run >= 2 && run % 2 == 0) {
              lane_var = site.loop_vars[ndim - 1];
            }
          }
        }
        if (lane_var.has_value()) {
          Array<PrimExpr> slot_args;
          for (size_t d = 0; d < buffer->shape.size(); ++d) {
            slot_args.push_back(InputPlaceholder(d));
          }
          Array<PrimExpr> fwd_index = induced->Forward(slot_args);
          PrimExpr fwd_thread =
              induced->ForwardThread(slot_args, ReplicationPlaceholder());
          PrimExpr lane = InputPlaceholder(buffer->shape.size());
          Array<PrimExpr> packed_shape = buffer->shape;
          packed_shape.push_back(IntImm(DataType::Int(32), 2));
          decision.packed_layout =
              Fragment(packed_shape, {fwd_index[0] * 2 + lane}, fwd_thread,
                       induced->ReplicateExtent(), std::nullopt)
                  ->BindThreadRange(site.loop_layout->ThreadRange());
          decision.packed = true;
          decision.pack_lane_var = lane_var.value();
        }
      }
    } else {
      if (!StructuralEqual()(decision.storage_layout, induced) ||
          decision.steps != site_steps) {
        return fail("update sites induce incompatible plans");
      }
    }
  }

  // Destination containment: every thread that owns a dst slot for element
  // `e` must hold `e`'s final value after the collective — which is exactly
  // the induced layout's thread image (reduction lanes all receive the
  // AllReduce result; replication groups compute equal values).
  if (auto dst_layout = known_layouts.Get(epoch.dst)) {
    auto dst_frag = dst_layout.value().as<Fragment>();
    if (!dst_frag) {
      return fail("destination has a non-fragment layout");
    }
    Array<PrimExpr> element_indices;
    for (size_t d = 0; d < buffer->shape.size(); ++d) {
      Var placeholder = InputPlaceholder(d);
      analyzer->Bind(
          placeholder,
          Range::FromMinExtent(make_zero(DataType::Int(32)), buffer->shape[d]),
          /*allow_override=*/true);
      element_indices.push_back(placeholder);
    }
    if (!ProveFragmentContains(dst_frag.value(), decision.storage_layout,
                               element_indices, element_indices, *analyzer)) {
      // The inferred layout mismatches the reduction's natural placement.
      // When the whole destination chain is unconstrained (finalize-written,
      // read only by copies to global or by staging copies into further
      // unconstrained fragments), its free-level layouts were arbitrary
      // choices and can be replaced by the induced layout together.
      // Downstream copy lowering re-infers loop layouts against the
      // overrides; ParallelOp validates its candidates against the solved
      // fragment layouts and returns an operand-compatible one, so
      // multi-slot overrides are safe.
      std::vector<Buffer> chain;
      if (CollectOverrideChain(epoch.dst->data.get(), use_graph, &chain)) {
        decision.override_dst_layout = true;
        decision.override_chain = std::move(chain);
      } else {
        return fail("destination layout is not covered by the induced layout");
      }
    }
  }
  // (No inferred dst layout: the materializer assigns the induced layout.)

  return decision;
}

// ---------------------------------------------------------------------------
// Phase C: materialization
// ---------------------------------------------------------------------------

class ReducerPlanAndMaterializeRewriter : public IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, bool force_baseline) {
    // Phase A: collect epochs.
    ReducerEpochCollector collector;
    collector.Collect(f);
    if (collector.epochs_.empty()) {
      return f;
    }
    // The inferred layout map (identical on every block) feeds the
    // destination-containment proof; the per-buffer use census decides
    // whether an inferred destination layout may be replaced.
    Map<Buffer, Layout> known_layouts;
    UseGraph use_graph;
    PostOrderVisit(f->body, [&](const ObjectRef &obj) {
      if (const auto *block = obj.as<SBlockNode>()) {
        if (auto anno = block->annotations.Get(tl::attr::kLayoutMap)) {
          if (auto as_map = anno.value().as<Map<Buffer, Layout>>()) {
            for (const auto &[buffer, layout] : as_map.value()) {
              known_layouts.Set(buffer, layout);
            }
          }
        }
      } else if (const auto *load = obj.as<BufferLoadNode>()) {
        use_graph.census[load->buffer->data.get()].loads++;
      } else if (const auto *store = obj.as<BufferStoreNode>()) {
        use_graph.census[store->buffer->data.get()].stores++;
      } else if (const auto *call = obj.as<CallNode>()) {
        auto region_buffer = [](const PrimExpr &arg) -> Buffer {
          if (auto region_call = arg.as<CallNode>()) {
            if (region_call->op.same_as(region())) {
              if (auto ld = region_call->args[0].as<BufferLoadNode>()) {
                return ld->buffer;
              }
            }
          }
          return Buffer();
        };
        if (call->op.same_as(Copy::Get()) && call->args.size() >= 2) {
          Buffer src = region_buffer(call->args[0]);
          Buffer dst = region_buffer(call->args[1]);
          if (src.defined() && dst.defined()) {
            use_graph.census[src->data.get()].copy_src++;
            use_graph.census[dst->data.get()].copy_dst++;
            use_graph.copies.push_back(
                CopyEdge{src->data.get(), dst->data.get(), IsGlobalBuffer(dst),
                         IsFragmentBuffer(dst), dst});
          }
        } else if (call->op.same_as(FinalizeReducerV2Op::Get())) {
          Buffer dst = region_buffer(call->args[1]);
          if (dst.defined()) {
            use_graph.census[dst->data.get()].finalize_dst++;
          }
        }
      }
    });

    // Phase B: decide a physical plan per epoch.
    arith::Analyzer analyzer;
    ReducerPlanAndMaterializeRewriter rewriter(&analyzer);
    rewriter.known_layouts_ = known_layouts;
    for (const auto &[var, epoch] : collector.epochs_) {
      if (!epoch.buffer.defined()) {
        continue;
      }
      std::string reason = "forced baseline (tl.reducer_force_baseline)";
      std::optional<NarrowDecision> decision;
      if (!force_baseline) {
        decision =
            TryNarrowPlan(epoch, known_layouts, use_graph, &analyzer, &reason);
      }
      bool verbose = tl_config::ReducerPlanVerboseEnabled();
      if (decision.has_value()) {
        std::string msg = "[ReducerPlan] `" + std::string(epoch.buffer->name) +
                          "`: narrow plan, ";
        if (decision->steps.empty()) {
          msg += "no collective";
        } else {
          for (const auto &[rt, s] : decision->steps) {
            msg += "AllReduce<" + std::to_string(rt) + "," + std::to_string(s) +
                   "> ";
          }
        }
        if (decision->packed) {
          msg += ", packed lanes";
        }
        if (verbose) {
          LOG(INFO) << msg;
        } else {
          DLOG(INFO) << msg;
        }
        rewriter.narrow_decisions_.emplace(var, *decision);
        // Destination-layout overrides must be registered BEFORE traversal:
        // LayoutInference publishes the stale entry on every block, and the
        // materializer's block post-processing runs bottom-up — a sibling
        // block processed before the finalize statement would otherwise
        // keep the stale entry and shadow the override in LowerTileOp's
        // block-by-block annotation accumulation.
        if (decision->override_dst_layout) {
          rewriter.extra_layout_entries_.Set(epoch.dst,
                                             decision->storage_layout);
          for (const Buffer &staged : decision->override_chain) {
            rewriter.extra_layout_entries_.Set(staged,
                                               decision->storage_layout);
          }
        }
      } else {
        std::string msg =
            "[ReducerPlan] `" + std::string(epoch.buffer->name) +
            "`: wide plan (FullParticipant); narrow rejected: " + reason;
        if (verbose) {
          LOG(INFO) << msg;
        } else {
          DLOG(INFO) << msg;
        }
      }
    }

    // Phase C: materialize.
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = rewriter.VisitStmt(f->body);
    return f;
  }

private:
  explicit ReducerPlanAndMaterializeRewriter(arith::Analyzer *analyzer)
      : IRMutatorWithAnalyzer(analyzer) {}

  struct Plan {
    Buffer old_buffer;
    Buffer new_buffer;
    Fragment layout;
    ReducerV2OpType op;
    Optional<PrimExpr> seed;
    bool narrow{false};
    std::vector<std::pair<int, int>> steps;
    // Packed accumulation: updates RMW `packed_buffer[slot, lane_var % 2]`;
    // finalize folds the lanes into `new_buffer` before the collective.
    bool packed{false};
    Buffer packed_buffer;
    Fragment packed_layout;
    Var pack_lane_var;
  };

  // ---- context tracking ---------------------------------------------------

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    auto prev_thread_var = thread_var_;
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        thread_var_ = iv;
      }
    }
    auto result = IRMutatorWithAnalyzer::VisitStmt_(op);
    thread_var_ = prev_thread_var;
    return result;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    if (auto anno = op->annotations.Get(attr::kReducerInfoV2)) {
      auto map = anno.value().as<Map<Var, Map<String, Any>>>();
      ICHECK(map) << "malformed reducer_info_v2 annotation";
      for (const auto &[var, info] : map.value()) {
        reducer_info_.emplace(var.get(), info);
      }
    }

    auto result = IRMutatorWithAnalyzer::VisitStmt_(op).as<SBlock>().value();
    auto *p_result = result.CopyOnWrite();

    // Swap materialized storage into this block's allocations and publish
    // the storage layouts (plus any planner-assigned destination layouts)
    // for LowerTileOp's buffer remap.
    bool changed = false;
    Array<Buffer> new_allocs;
    Map<Buffer, Layout> layout_map;
    if (auto layout_anno = p_result->annotations.Get(tl::attr::kLayoutMap)) {
      auto as_map = layout_anno.value().as<Map<Buffer, Layout>>();
      ICHECK(as_map) << "kLayoutMap must be Buffer-keyed after LayoutInference";
      layout_map = as_map.value();
    }
    for (const auto &buffer : p_result->alloc_buffers) {
      auto it = plans_.find(buffer->data.get());
      if (it != plans_.end()) {
        new_allocs.push_back(it->second.new_buffer);
        layout_map.Set(it->second.new_buffer, it->second.layout);
        if (it->second.packed) {
          new_allocs.push_back(it->second.packed_buffer);
          layout_map.Set(it->second.packed_buffer, it->second.packed_layout);
        }
        changed = true;
      } else {
        new_allocs.push_back(buffer);
        if (auto extra = extra_layout_entries_.Get(buffer)) {
          layout_map.Set(buffer, extra.value());
          changed = true;
        }
      }
    }
    // Layout overrides must win on EVERY block: LayoutInference publishes
    // the full inferred map on each block, and LowerTileOp accumulates the
    // annotations block by block — a stale entry on an inner block would
    // silently shadow the override.
    for (const auto &[buffer, layout] : extra_layout_entries_) {
      if (layout_map.count(buffer) && !layout_map[buffer].same_as(layout)) {
        layout_map.Set(buffer, layout);
        changed = true;
      }
    }
    if (changed) {
      p_result->alloc_buffers = new_allocs;
      p_result->annotations.Set(tl::attr::kLayoutMap, layout_map);
    }
    if (p_result->annotations.count(attr::kReducerInfoV2)) {
      p_result->annotations.erase(attr::kReducerInfoV2);
    }
    return result;
  }

  // ---- epoch op materialization -------------------------------------------

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    if (call->op.same_as(ReducerInitOp::Get())) {
      return MaterializeInit(call);
    }
    if (call->op.same_as(reducer_update())) {
      return MaterializeUpdate(call);
    }
    if (call->op.same_as(FinalizeReducerV2Op::Get())) {
      return MaterializeFinalize(call);
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt MaterializeInit(const CallNode *call) {
    Buffer old_buffer = RegionArgBuffer(call->args[0]);
    const auto *var = old_buffer->data.get();
    ICHECK(reducer_info_.count(var))
        << "reducer_init on unknown reducer `" << old_buffer << "`";
    ICHECK(!plans_.count(var))
        << "double reducer_init on `" << old_buffer
        << "` (should have been rejected by VerifyReducerEpoch)";

    Plan plan;
    plan.old_buffer = old_buffer;
    const auto &info = reducer_info_.at(var);
    plan.op = ParseReducerV2OpType(info.Get("op").value().cast<String>());
    if (call->args.size() >= 2) {
      // Logical starting value from T.reducer_init(acc, init): combined
      // exactly once per logical output at finalize time.
      plan.seed = call->args[1];
    }

    auto narrow_it = narrow_decisions_.find(var);
    if (narrow_it != narrow_decisions_.end()) {
      const NarrowDecision &decision = narrow_it->second;
      plan.narrow = true;
      plan.layout = decision.storage_layout;
      plan.steps = decision.steps;
      plan.packed = decision.packed;
      plan.packed_layout = decision.packed_layout;
      plan.pack_lane_var = decision.pack_lane_var;
    } else {
      // Wide-plan storage: one full logical partial per participant. The
      // analyzer narrows threadIdx.x inside warp-specialized branches.
      ICHECK(thread_var_.defined())
          << "reducer_init must execute inside a kernel launch";
      ICHECK(analyzer_->const_int_bound.IsBound(thread_var_->var))
          << "reducer_init: threadIdx.x bounds are unknown";
      auto bound = analyzer_->const_int_bound(thread_var_->var);
      int64_t thread_extent = bound->max_value - bound->min_value + 1;
      ICHECK_GT(thread_extent, 0);

      static constexpr int64_t kMaxWidePlanSlots = 256;
      int64_t slots = 1;
      for (const auto &extent : old_buffer->shape) {
        const int64_t *p = as_const_int(extent);
        ICHECK(p) << "reducer shape must be compile-time constant, got "
                  << old_buffer->shape;
        slots *= *p;
      }
      ICHECK_LE(slots, kMaxWidePlanSlots)
          << "reducer `" << old_buffer << "` needs " << slots
          << " per-thread partial slots under the FullParticipant baseline, "
             "exceeding the current policy limit of "
          << kMaxWidePlanSlots
          << "; split the reduction into tiles or use T.reduce_* per tile.";
      plan.layout = Fragment::FullyReplicated(old_buffer->shape,
                                              static_cast<int>(thread_extent));
    }

    Var new_var(old_buffer->data->name_hint,
                PointerType(PrimType(old_buffer->dtype), "local.fragment"));
    plan.new_buffer = Buffer(
        new_var, old_buffer->dtype, old_buffer->shape, old_buffer->strides,
        old_buffer->elem_offset, old_buffer->name, old_buffer->data_alignment,
        old_buffer->offset_factor, old_buffer->buffer_type);
    if (plan.packed) {
      Array<PrimExpr> packed_shape = old_buffer->shape;
      packed_shape.push_back(IntImm(DataType::Int(32), 2));
      Var packed_var(
          old_buffer->data->name_hint + "_pk",
          PointerType(PrimType(old_buffer->dtype), "local.fragment"));
      plan.packed_buffer =
          Buffer(packed_var, old_buffer->dtype, packed_shape,
                 /*strides=*/{}, old_buffer->elem_offset,
                 old_buffer->name + "_pk", old_buffer->data_alignment,
                 old_buffer->offset_factor, old_buffer->buffer_type);
    }
    plans_.emplace(var, plan);
    const Plan &stored = plans_.at(var);

    // Every participant starts from the combine identity; the seed is
    // combined exactly once at finalize. Packed plans accumulate into the
    // lane buffer (`new_buffer` is fully written by the finalize fold).
    const Buffer &init_target =
        stored.packed ? stored.packed_buffer : stored.new_buffer;
    PrimExpr identity = ReducerV2Identity(stored.op, init_target->dtype);
    return Evaluate(
        Call(DataType::Handle(), Fill::Get(),
             {MakeFullRegion(init_target, kAccessWrite), identity}));
  }

  Stmt MaterializeUpdate(const CallNode *call) {
    ReducerUpdateArgs update = ParseReducerUpdate(call);
    auto it = plans_.find(update.reducer->data.get());
    ICHECK(it != plans_.end())
        << "reducer_update on `" << update.reducer
        << "` before reducer_init (should have been rejected by "
           "VerifyReducerEpoch)";
    const Plan &plan = it->second;

    PrimExpr value = VisitExpr(update.value);
    Array<PrimExpr> indices = update.indices;
    Buffer target = plan.new_buffer;
    if (plan.packed) {
      // Alternate between the two lanes of the executing thread's slot:
      // the lane parity splits the on-thread combine chain in half and
      // makes adjacent iterations touch adjacent physical elements.
      target = plan.packed_buffer;
      indices.push_back(
          FloorMod(plan.pack_lane_var, IntImm(DataType::Int(32), 2)));
    }
    PrimExpr current = BufferLoad(target, indices);
    Stmt store =
        BufferStore(target, ReducerV2Combine(plan.op, current, value), indices);
    if (plan.narrow) {
      // Narrow plan: every replica executes the update. Each thread's
      // partial accumulates exactly the contributions of the iterations
      // mapped to it, so no multiplicity guard is needed.
      return store;
    }
    // Wide plan: generic execution-multiplicity contract — one dynamic
    // logical iteration of the enclosing T.Parallel loop contributes exactly
    // once, no matter how the loop layout replicates iterations over threads.
    return AttrStmt(plan.new_buffer->data, attr::kParallelMultiplicity,
                    IntImm(DataType::Int(32), 1), store);
  }

  Stmt MaterializeFinalize(const CallNode *call) {
    Buffer old_buffer = RegionArgBuffer(call->args[0]);
    Buffer dst = RegionArgBuffer(call->args[1]);
    auto it = plans_.find(old_buffer->data.get());
    ICHECK(it != plans_.end())
        << "finalize_reducer on `" << old_buffer
        << "` before reducer_init (should have been rejected by "
           "VerifyReducerEpoch)";
    const Plan &plan = it->second;

    // A destination only written by finalize (and read by scalar or uniform
    // code) may have no inferred layout. The planner then assigns one that
    // is trivially covered: the induced layout itself (narrow) or a
    // FullyReplicated layout (wide). A layout inferred from downstream
    // consumers is respected (destination-layout OVERRIDES are registered in
    // Phase B, before traversal). This fresh assignment is order-safe here:
    // no block carries a stale entry for a buffer inference never layouted,
    // and the allocating block is an ancestor of this finalize statement.
    bool planner_assigned_dst_layout = false;
    if (IsFragmentBuffer(dst) && !known_layouts_.count(dst) &&
        !extra_layout_entries_.count(dst)) {
      Fragment dst_layout =
          plan.narrow ? plan.layout
                      : Fragment::FullyReplicated(
                            dst->shape, plan.layout->ReplicateExtent());
      extra_layout_entries_.Set(dst, dst_layout);
      planner_assigned_dst_layout = true;
    }

    Array<Stmt> seq;
    if (plan.narrow) {
      if (plan.packed) {
        // Fold the two accumulation lanes into the plain induced storage.
        // The fold loop is a frozen-layout parallel nest (the storage
        // fragment doubles as its loop layout): every thread combines the
        // lanes of exactly the slots it owns, replicas included, so the
        // collective and destination proofs below see the unpacked layout
        // they were made for.
        Array<Var> fold_vars;
        Array<PrimExpr> fold_indices;
        for (size_t d = 0; d < plan.new_buffer->shape.size(); ++d) {
          Var v("__red_fold_" + std::to_string(d), DataType::Int(32));
          fold_vars.push_back(v);
          fold_indices.push_back(v);
        }
        Array<PrimExpr> lane0 = fold_indices;
        lane0.push_back(IntImm(DataType::Int(32), 0));
        Array<PrimExpr> lane1 = fold_indices;
        lane1.push_back(IntImm(DataType::Int(32), 1));
        Stmt fold = BufferStore(
            plan.new_buffer,
            ReducerV2Combine(plan.op, BufferLoad(plan.packed_buffer, lane0),
                             BufferLoad(plan.packed_buffer, lane1)),
            fold_indices);
        for (int d = static_cast<int>(fold_vars.size()) - 1; d >= 0; --d) {
          Map<String, Any> annotations;
          if (d == 0) {
            annotations.Set(tl::attr::kParallelLoopLayout, plan.layout);
          }
          fold = For(fold_vars[d], make_zero(DataType::Int(32)),
                     plan.new_buffer->shape[d], ForKind::kParallel, fold,
                     std::nullopt, annotations);
        }
        seq.push_back(fold);
      }
      // Narrow plan: combine only the reduction-axis splits (when any);
      // replication groups already hold equal values.
      if (!plan.steps.empty() || plan.seed.defined()) {
        Array<PrimExpr> args = {
            MakeFullRegion(plan.new_buffer, kAccessReadWrite),
            IntImm(DataType::Int(32), static_cast<int>(plan.op))};
        for (const auto &[reducing_threads, scale] : plan.steps) {
          args.push_back(IntImm(DataType::Int(32), reducing_threads));
          args.push_back(IntImm(DataType::Int(32), scale));
        }
        Map<String, ObjectRef> annotations = call->annotations;
        annotations.Set("plan", Integer(1));
        if (plan.seed.defined()) {
          annotations.Set("seed", plan.seed.value());
        }
        seq.push_back(Evaluate(Call(
            DataType::Handle(), FinalizeReducerOp::Get(), args, annotations)));
      }
      // Publish via tl.copy: it partitions by the destination layout, and
      // destination containment was proven (or the destination layout IS
      // the induced layout).
      seq.push_back(Evaluate(Call(DataType::Handle(), Copy::Get(),
                                  {MakeFullRegion(plan.new_buffer, kAccessRead),
                                   MakeFullRegion(dst, kAccessWrite)})));
      return seq.size() == 1 ? seq[0] : SeqStmt(seq);
    }

    // Wide plan: participant-extent AllReduce per logical output, then the
    // optional seed combined once per slot inside the finalize lowering.
    // The finalize call's annotations (e.g. `batch`) are forwarded.
    {
      Map<String, ObjectRef> annotations = call->annotations;
      if (plan.seed.defined()) {
        annotations.Set("seed", plan.seed.value());
      }
      seq.push_back(
          Evaluate(Call(DataType::Handle(), FinalizeReducerOp::Get(),
                        {MakeFullRegion(plan.new_buffer, kAccessReadWrite),
                         IntImm(DataType::Int(32), static_cast<int>(plan.op))},
                        annotations)));
    }

    // Publish the logical result into the independent destination fragment.
    if (planner_assigned_dst_layout) {
      // The planner declared dst FullyReplicated, so EVERY replica must be
      // written. tl.copy would only produce the canonical replica; instead
      // emit a per-thread local copy — after the collective every
      // participant holds the complete result, and both layouts use
      // identity physical indexing.
      Array<PrimExpr> indices;
      Array<Var> loop_vars;
      for (size_t i = 0; i < dst->shape.size(); ++i) {
        Var v("__finred_cp_" + std::to_string(i), DataType::Int(32));
        loop_vars.push_back(v);
        indices.push_back(v);
      }
      Stmt body =
          BufferStore(dst, BufferLoad(plan.new_buffer, indices), indices);
      for (int i = static_cast<int>(loop_vars.size()) - 1; i >= 0; --i) {
        body = For(loop_vars[i], make_zero(DataType::Int(32)), dst->shape[i],
                   ForKind::kSerial, body);
      }
      seq.push_back(body);
    } else {
      // dst's layout came from LayoutInference (real consumers); tl.copy
      // writes each thread's owned slots, reading from the fully
      // replicated source.
      seq.push_back(Evaluate(Call(DataType::Handle(), Copy::Get(),
                                  {MakeFullRegion(plan.new_buffer, kAccessRead),
                                   MakeFullRegion(dst, kAccessWrite)})));
    }
    return seq.size() == 1 ? seq[0] : SeqStmt(seq);
  }

  // ---- helpers ------------------------------------------------------------

  static Buffer RegionArgBuffer(const PrimExpr &arg) {
    if (auto call = arg.as<CallNode>()) {
      if (call->op.same_as(region())) {
        if (auto load = call->args[0].as<BufferLoadNode>()) {
          return load->buffer;
        }
      }
    }
    LOG(FATAL) << "expected a tl.region argument wrapping a buffer, got "
               << arg;
    return Buffer(); // unreachable
  }

  IterVar thread_var_;
  std::unordered_map<const VarNode *, Map<String, Any>> reducer_info_;
  std::unordered_map<const VarNode *, Plan> plans_;
  std::unordered_map<const VarNode *, NarrowDecision> narrow_decisions_;
  Map<Buffer, Layout> known_layouts_;
  Map<Buffer, Layout> extra_layout_entries_;
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass ReducerPlanAndMaterialize() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    bool force_baseline =
        ctx->GetConfig<Bool>(kReducerForceBaseline, Bool(false)).value();
    return ReducerPlanAndMaterializeRewriter::Substitute(std::move(f),
                                                         force_baseline);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ReducerPlanAndMaterialize", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.ReducerPlanAndMaterialize",
                        ReducerPlanAndMaterialize);
}

} // namespace tl
} // namespace tvm
