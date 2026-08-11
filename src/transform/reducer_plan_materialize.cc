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

#include <optional>
#include <unordered_map>
#include <vector>

#include "../layout/layout.h"
#include "../layout/utils.h"
#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/fill.h"
#include "../op/reducer.h"
#include "../op/region.h"
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
  return Call(DataType::Handle(), RegionOp::Get(), args);
}

// ---------------------------------------------------------------------------
// Phase A: epoch collection
// ---------------------------------------------------------------------------

struct UpdateSite {
  Fragment loop_layout;    // solved layout of the enclosing parallel nest
  Array<Var> loop_vars;    // nest loop vars in order
  Array<PrimExpr> indices; // logical output indices of the update target
  PrimExpr value;          // contribution expression
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
        if (auto seed = info.Get("seed")) {
          epoch.seed = seed.value().cast<PrimExpr>();
        }
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
      IRVisitorWithAnalyzer::VisitStmt_(op);
      cur_loop_layout_ = prev_layout;
      cur_loop_vars_ = prev_vars;
      return;
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(ReducerInitOp::Get())) {
      if (EpochInfo *epoch = FindEpoch(op->args[0])) {
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
    if (op->op.same_as(ReducerUpdateOp::Get())) {
      if (EpochInfo *epoch = FindEpoch(op->args[0])) {
        if (cur_loop_layout_.defined()) {
          ReducerUpdateOp update(op->args, {});
          const auto *node = update.as<ReducerUpdateOpNode>();
          epoch->updates.push_back(UpdateSite{cur_loop_layout_.value(),
                                              cur_loop_vars_, node->indices,
                                              node->value});
        } else {
          epoch->analyzable = false;
        }
      }
      return;
    }
    if (op->op.same_as(FinalizeReducerV2Op::Get())) {
      if (EpochInfo *epoch = FindEpoch(op->args[0])) {
        if (auto call2 = op->args[1].as<CallNode>()) {
          if (call2->op.same_as(RegionOp::Get())) {
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
      if (call->op.same_as(RegionOp::Get())) {
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
};

// ---------------------------------------------------------------------------
// Phase B: narrow-plan analysis
// ---------------------------------------------------------------------------

struct NarrowDecision {
  Fragment storage_layout; // induced partial layout (also post-collective)
  bool has_step{false};
  int reducing_threads{0};
  int scale{1};
  // True when the destination's free-level inferred layout is replaced by
  // the induced layout (legal only for unconstrained destinations).
  bool override_dst_layout{false};
};

/*! \brief Access census of a finalize destination, used to decide whether
 *  its inferred layout may be replaced. A destination is unconstrained when
 *  it is written only by the finalize and read only as the source of copies
 *  into global memory: such uses lower against whatever layout the buffer
 *  has, and the free-level layout LayoutInference picked for it never
 *  constrained any other buffer. */
struct DstUseCensus {
  int64_t loads{0};         // every BufferLoad occurrence (incl. regions)
  int64_t stores{0};        // ordinary stores (must be zero)
  int64_t safe_copy_src{0}; // tl.copy(dst -> global) source regions
  int64_t finalize_dst{0};  // finalize_reducer destination regions

  bool Unconstrained() const {
    return stores == 0 && finalize_dst == 1 &&
           loads == safe_copy_src + finalize_dst;
  }
};

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
              const DstUseCensus &dst_census, arith::Analyzer *analyzer,
              std::string *reason) {
  auto fail = [&](const std::string &why) -> std::optional<NarrowDecision> {
    *reason = why;
    return std::nullopt;
  };
  if (!epoch.analyzable) {
    return fail("incomplete epoch structure");
  }
  if (epoch.seed.defined()) {
    return fail("seed not yet supported by the narrow plan");
  }
  if (epoch.batch > 1) {
    return fail("batched finalize not yet supported by the narrow plan");
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
    // var used in nest order (direct identity ownership), or a constant
    // zero on a unit reducer dim.
    std::vector<bool> is_output_dim(ndim, false);
    int last_pos = -1;
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
        if (pos <= last_pos) {
          return fail("update indices permute the parallel loop order");
        }
        const int64_t *loop_extent =
            as_const_int(site.loop_layout->InputShape()[pos]);
        const int64_t *dim_extent = as_const_int(buffer->shape[d]);
        if (!loop_extent || !dim_extent || *loop_extent != *dim_extent) {
          return fail("loop extent does not match the reducer dim extent");
        }
        is_output_dim[pos] = true;
        last_pos = pos;
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
    // dim numbers stay stable while dims are removed).
    Fragment induced = site.loop_layout;
    for (int dim = static_cast<int>(ndim) - 1; dim >= 0; --dim) {
      if (!is_output_dim[dim]) {
        induced = backend::reduce::ComputeReducerLayout(induced, dim);
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
    if (steps.size() > 1) {
      return fail("multi-step collectives not yet supported");
    }
    bool has_step = !steps.empty();
    int reducing_threads = 0;
    int scale = 1;
    if (has_step) {
      if (!IsPowerOfTwo(steps[0].extent)) {
        return fail("collective width is not a power of two");
      }
      reducing_threads = steps[0].ReducingThreads();
      scale = steps[0].scale;
      if (reducing_threads > epoch.thread_extent) {
        return fail("collective width exceeds the participant extent");
      }
    }

    if (first_site) {
      decision.storage_layout = induced;
      decision.has_step = has_step;
      decision.reducing_threads = reducing_threads;
      decision.scale = scale;
      first_site = false;
    } else {
      if (!StructuralEqual()(decision.storage_layout, induced) ||
          decision.has_step != has_step ||
          decision.reducing_threads != reducing_threads ||
          decision.scale != scale) {
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
      // When the destination is unconstrained (finalize-written, read only
      // by copies to global), its free-level layout was an arbitrary choice
      // and can be replaced by the induced layout — but only when the
      // induced layout keeps one slot per thread: downstream copy lowering
      // re-infers a loop layout and must be able to validate against the
      // override, which multi-slot induced layouts are not guaranteed to
      // survive (proof failure must never become a compile error).
      int64_t induced_slots = 1;
      for (const auto &extent : decision.storage_layout->OutputShape()) {
        const int64_t *p = as_const_int(extent);
        induced_slots = (p == nullptr) ? -1 : induced_slots * *p;
      }
      if (dst_census.Unconstrained() && induced_slots == 1) {
        decision.override_dst_layout = true;
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
    std::unordered_map<const VarNode *, DstUseCensus> census;
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
        census[load->buffer->data.get()].loads++;
      } else if (const auto *store = obj.as<BufferStoreNode>()) {
        census[store->buffer->data.get()].stores++;
      } else if (const auto *call = obj.as<CallNode>()) {
        auto region_buffer = [](const PrimExpr &arg) -> const VarNode * {
          if (auto region = arg.as<CallNode>()) {
            if (region->op.same_as(RegionOp::Get())) {
              if (auto ld = region->args[0].as<BufferLoadNode>()) {
                return ld->buffer->data.get();
              }
            }
          }
          return nullptr;
        };
        if (call->op.same_as(Copy::Get()) && call->args.size() >= 2) {
          const VarNode *src = region_buffer(call->args[0]);
          const VarNode *dst = region_buffer(call->args[1]);
          if (src != nullptr && dst != nullptr) {
            if (auto dst_region = call->args[1].as<CallNode>()) {
              if (auto ld = dst_region->args[0].as<BufferLoadNode>()) {
                if (IsGlobalBuffer(ld->buffer)) {
                  census[src].safe_copy_src++;
                }
              }
            }
          }
        } else if (call->op.same_as(FinalizeReducerV2Op::Get())) {
          if (const VarNode *dst = region_buffer(call->args[1])) {
            census[dst].finalize_dst++;
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
        DstUseCensus dst_census;
        if (epoch.dst.defined()) {
          auto census_it = census.find(epoch.dst->data.get());
          if (census_it != census.end()) {
            dst_census = census_it->second;
          }
        }
        decision =
            TryNarrowPlan(epoch, known_layouts, dst_census, &analyzer, &reason);
      }
      if (decision.has_value()) {
        DLOG(INFO) << "[ReducerPlan] `" << epoch.buffer->name
                   << "`: narrow plan, "
                   << (decision->has_step
                           ? "AllReduce<" +
                                 std::to_string(decision->reducing_threads) +
                                 "," + std::to_string(decision->scale) + ">"
                           : std::string("no collective"));
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
        }
      } else {
        DLOG(INFO) << "[ReducerPlan] `" << epoch.buffer->name
                   << "`: wide plan (FullParticipant); narrow rejected: "
                   << reason;
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
    bool has_step{false};
    int reducing_threads{0};
    int scale{1};
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
    if (call->op.same_as(ReducerUpdateOp::Get())) {
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
    if (auto seed = info.Get("seed")) {
      plan.seed = seed.value().cast<PrimExpr>();
    }

    auto narrow_it = narrow_decisions_.find(var);
    if (narrow_it != narrow_decisions_.end()) {
      const NarrowDecision &decision = narrow_it->second;
      plan.narrow = true;
      plan.layout = decision.storage_layout;
      plan.has_step = decision.has_step;
      plan.reducing_threads = decision.reducing_threads;
      plan.scale = decision.scale;
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
    plans_.emplace(var, plan);
    const Plan &stored = plans_.at(var);

    // Every participant starts from the combine identity; the seed is
    // combined exactly once at finalize.
    PrimExpr identity = ReducerV2Identity(stored.op, stored.new_buffer->dtype);
    return Evaluate(
        Call(DataType::Handle(), Fill::Get(),
             {MakeFullRegion(stored.new_buffer, kAccessWrite), identity}));
  }

  Stmt MaterializeUpdate(const CallNode *call) {
    // Reuse the op parser for (buffer, indices, value) extraction.
    ReducerUpdateOp update(call->args, {});
    const auto *node = update.as<ReducerUpdateOpNode>();
    auto it = plans_.find(node->reducer->data.get());
    ICHECK(it != plans_.end())
        << "reducer_update on `" << node->reducer
        << "` before reducer_init (should have been rejected by "
           "VerifyReducerEpoch)";
    const Plan &plan = it->second;

    PrimExpr value = VisitExpr(node->value);
    Array<PrimExpr> indices = node->indices;
    PrimExpr current = BufferLoad(plan.new_buffer, indices);
    Stmt store = BufferStore(
        plan.new_buffer, ReducerV2Combine(plan.op, current, value), indices);
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
      // Narrow plan: combine only the reduction-axis splits (when any);
      // replication groups already hold equal values.
      if (plan.has_step) {
        seq.push_back(
            Evaluate(Call(DataType::Handle(), FinalizeReducerOp::Get(),
                          {MakeFullRegion(plan.new_buffer, kAccessReadWrite),
                           IntImm(DataType::Int(32), static_cast<int>(plan.op)),
                           IntImm(DataType::Int(32), plan.reducing_threads),
                           IntImm(DataType::Int(32), plan.scale)},
                          call->annotations)));
      }
      ICHECK(!plan.seed.defined())
          << "narrow plan with seed should have been rejected in analysis";
      // Publish via tl.copy: it partitions by the destination layout, and
      // destination containment was proven (or the destination layout IS
      // the induced layout).
      seq.push_back(Evaluate(Call(DataType::Handle(), Copy::Get(),
                                  {MakeFullRegion(plan.new_buffer, kAccessRead),
                                   MakeFullRegion(dst, kAccessWrite)})));
      return seq.size() == 1 ? seq[0] : SeqStmt(seq);
    }

    // Wide plan: participant-extent AllReduce per logical output. The
    // finalize call's annotations (e.g. `batch`) are forwarded.
    seq.push_back(
        Evaluate(Call(DataType::Handle(), FinalizeReducerOp::Get(),
                      {MakeFullRegion(plan.new_buffer, kAccessReadWrite),
                       IntImm(DataType::Int(32), static_cast<int>(plan.op))},
                      call->annotations)));

    // Seed: after the collective every participant holds the combined value,
    // so a uniform per-thread combine applies the seed exactly once per
    // logical output while keeping all replicas equal.
    if (plan.seed.defined()) {
      Array<PrimExpr> indices;
      Array<Var> loop_vars;
      for (size_t i = 0; i < plan.new_buffer->shape.size(); ++i) {
        Var v("__seed_" + std::to_string(i), DataType::Int(32));
        loop_vars.push_back(v);
        indices.push_back(v);
      }
      PrimExpr current = BufferLoad(plan.new_buffer, indices);
      Stmt body = BufferStore(
          plan.new_buffer,
          ReducerV2Combine(plan.op, current, plan.seed.value()), indices);
      for (int i = static_cast<int>(loop_vars.size()) - 1; i >= 0; --i) {
        body = For(loop_vars[i], make_zero(DataType::Int(32)),
                   plan.new_buffer->shape[i], ForKind::kSerial, body);
      }
      seq.push_back(body);
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
      if (call->op.same_as(RegionOp::Get())) {
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
