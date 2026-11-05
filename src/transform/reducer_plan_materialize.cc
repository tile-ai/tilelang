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
 * Two physical plans exist. The reducer's solved PartialFragment IS the
 * plan (LayoutInference is the authority: update nests propose the induced
 * projection when every per-site proof passes, the engine widens on
 * multi-site disagreement, zero-update epochs are seeded wide,
 * tl.reducer_force_baseline pins the proposal): its algebra is the
 * storage, CombineSize/CombineSteps the communication, and the
 * FullParticipant object (combine == replicate == participants,
 * thread == _rep) marks the wide fallback. This pass only derives the
 * lowering details that live outside the layout — packed lanes from the
 * update site's serial structure and the destination-containment
 * lowerability gate (the one remaining materialize-time demotion).
 *
 * Narrow plan (per-site proofs live in the inference-side proposal):
 *   The reduction axes of an update site are the parallel loop vars that do
 *   not appear in the update indices, plus the loop's replication. Projecting
 *   them out of the site's loop layout yields the induced partial layout:
 *   compact per-thread storage where every replica executes its update
 *   unguarded, so each partial accumulates exactly the contributions of the
 *   iterations mapped to its thread. The finalize collective combines only
 *   the thread-expression splits sourced from reduction axes (extracted with
 *   the same machinery T.reduce uses); splits from loop replication become
 *   value replication and are not reduced. Zero splits means zero
 *   collectives (the LocalComplete case). The per-site proofs (direct
 *   var-to-dim update indices, replica-safe contribution values, full-block
 *   thread coverage, power-of-two reduce steps) and the multi-site
 *   agreement rule run at proposal/commit time in LayoutInference; only
 *   destination-layout containment is checked here.
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

#include "backend/common/op/reduce.h"
#include "backend/common/target_utils.h"
#include "common/native_reducer_utils.h"
#include "common/pipeline_utils.h"
#include "config.h"
#include "ir/analyzer.h"
#include "op/copy.h"
#include "op/fill.h"

#include "ir/stmt_functor.h"
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace tvm::tl {
namespace {
using namespace tirx;
using namespace prim;
using namespace ffi;
using namespace reducer_native;

struct UpdateSite {
  Fragment loop_layout;
  Array<PrimVar> loop_vars;
  Array<PrimExpr> indices;
  PrimExpr value{UnsafeInit{}};
  Array<PrimVar> serial_vars;
  Array<PrimExpr> serial_extents;
};

struct EpochInfo {
  TensorVar buffer{UnsafeInit{}};
  ReducerV2OpType op{ReducerV2OpType::kSum};
  Optional<PrimExpr> seed;
  int64_t thread_extent{-1};
  int64_t thread_min{0};
  std::vector<UpdateSite> updates;
  TensorVar dst{UnsafeInit{}};
  int64_t batch{1};
  bool analyzable{true};
};

class ReducerEpochCollector : public StmtExprVisitorWithAnalyzer {
public:
  ReducerEpochCollector() {}
  using StmtExprVisitorWithAnalyzer::Visit_;
  std::unordered_map<const VarObj *, EpochInfo> epochs;
  MetadataIndex metadata;
  bool cpu{false};

  void Collect(const tirx::Function &function) {
    metadata = MetadataIndex::Collect(function);
    cpu = TargetIsCPU(function->GetAttr<Target>(tvm::attr::kTarget).value());
    for (const auto &[tensor, info] : metadata.v2) {
      TVM_FFI_CHECK(metadata.allocations.count(tensor), ValueError)
          << "Reducer metadata must refer to an allocation";
      auto &epoch = epochs[tensor.get()];
      epoch.buffer = tensor;
      epoch.op = ParseReducerV2OpType(info.at("op").as_or_throw<String>());
    }
    if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      Visit(function->body);
  }

  Optional<VisitInterrupt> Visit_(const RegionStmtObj *op) final {
    auto previous_thread = thread_;
    auto previous_pending = pending_layout_;
    if (op->op.same_as(launch_thread_op()) &&
        op->args[0].as_or_throw<StringImm>()->value == "threadIdx.x")
      thread_ = op->body_params[0].as_or_throw<PrimVar>();
    if (op->op->name == "tl.attr") {
      auto fields = ReadNativeAnnotations(op);
      if (fields.at("key").as_or_throw<String>() == "tl.loop.annotations") {
        auto values = StringMap(fields.at("value"));
        if (auto layout = values.Get(attr::kParallelLoopLayout))
          pending_layout_ = layout->as_or_throw<Fragment>();
      }
      // The annotation tuple carries expressions and tensor references for
      // analysis. It is not an execution of those values.
      Visit(op->body);
    } else {
      StmtExprVisitorWithAnalyzer::Visit_(op);
    }
    thread_ = previous_thread;
    pending_layout_ = previous_pending;
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const ForObj *op) final {
    auto annotations = ReadLoopAnnotations(op);
    Optional<Fragment> layout = pending_layout_;
    if (auto value = annotations.Get(attr::kParallelLoopLayout))
      layout = value->as_or_throw<Fragment>();
    if (op->kind == ForKind::kParallel && layout) {
      auto previous_layout = loop_layout_;
      auto previous_vars = loop_vars_;
      auto previous_serial_vars = serial_vars_;
      auto previous_serial_extents = serial_extents_;
      auto previous_pending = pending_layout_;
      pending_layout_ = std::nullopt;
      loop_layout_ = layout;
      loop_vars_.clear();
      for (const ForObj *current = op; current != nullptr;) {
        loop_vars_.push_back(current->var);
        const auto *inner = SingleNestedFor(current->body);
        current = inner && inner->kind == ForKind::kParallel ? inner : nullptr;
      }
      serial_vars_.clear();
      serial_extents_.clear();
      StmtExprVisitorWithAnalyzer::Visit_(op);
      loop_layout_ = previous_layout;
      loop_vars_ = previous_vars;
      serial_vars_ = previous_serial_vars;
      serial_extents_ = previous_serial_extents;
      pending_layout_ = previous_pending;
      return std::nullopt;
    }
    const bool serial =
        op->kind == ForKind::kDefault && loop_layout_.has_value();
    if (serial) {
      serial_vars_.push_back(op->var);
      serial_extents_.push_back(LoopExtent(op));
    }
    StmtExprVisitorWithAnalyzer::Visit_(op);
    if (serial) {
      serial_vars_.pop_back();
      serial_extents_.pop_back();
    }
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    if (IsMetadataCall(op))
      return std::nullopt;
    if (op->op.same_as(ReducerInitOp::Get())) {
      auto args = DecodeTileOpArguments(op).operands;
      auto &epoch = Find(args[0]);
      if (args.size() == 2)
        epoch.seed = args[1].as_or_throw<PrimExpr>();
      if (cpu) {
        epoch.thread_extent = 1;
        epoch.thread_min = 0;
      } else {
        TVM_FFI_CHECK(
            thread_ && GetSyncedAnalyzer()->const_int_bound.IsBound(*thread_),
            ValueError)
            << "reducer_init must execute inside a kernel launch with known "
               "threadIdx.x bounds";
        auto bound = GetSyncedAnalyzer()->const_int_bound(*thread_);
        epoch.thread_min = bound->min_value;
        epoch.thread_extent = bound->max_value - bound->min_value + 1;
      }
      TVM_FFI_CHECK_GT(epoch.thread_extent, 0, ValueError);
      return std::nullopt;
    }
    if (op->op.same_as(reducer_update())) {
      auto update = ParseReducerUpdate(op);
      auto &epoch = epochs.at(update.reducer.get());
      if (loop_layout_)
        epoch.updates.push_back({*loop_layout_, loop_vars_, update.indices,
                                 update.value, serial_vars_, serial_extents_});
      else
        epoch.analyzable = false;
      return std::nullopt;
    }
    if (op->op.same_as(FinalizeReducerV2Op::Get())) {
      auto decoded = DecodeTileOpArguments(op);
      auto &epoch = Find(decoded.operands[0]);
      epoch.dst = reducer_native::RegionTensor(decoded.operands[1]);
      if (auto batch = decoded.annotations.Get("batch"))
        epoch.batch = GetIntegerAnnotation(*batch);
      return std::nullopt;
    }
    return StmtExprVisitorWithAnalyzer::Visit_(op);
  }

private:
  EpochInfo &Find(const Expr &region) {
    auto tensor = reducer_native::RegionTensor(region);
    auto found = epochs.find(tensor.get());
    TVM_FFI_CHECK(found != epochs.end(), ValueError)
        << "Unknown reducer " << tensor;
    return found->second;
  }
  Optional<PrimVar> thread_;
  Optional<Fragment> pending_layout_;
  Optional<Fragment> loop_layout_;
  Array<PrimVar> loop_vars_, serial_vars_;
  Array<PrimExpr> serial_extents_;
};

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
  std::vector<TensorVar> override_chain;
  // Packed partial accumulation (16-bit floats): updates write two
  // interleaved lanes per logical slot selected by the parity of
  // `pack_lane_var` (the innermost on-thread reduction loop), which breaks
  // the serial combine dependence chain and lets the vectorizer emit
  // paired 16-bit (half2-style) operations. A fold loop combines the lanes
  // into the plain induced storage before the collective, so the plan's
  // communication and destination proofs are untouched.
  bool packed{false};
  PrimVar pack_lane_var{UnsafeInit{}};
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
  int64_t loads{0}; // every BufferLoad occurrence (incl. regions)
  int64_t stores{0};
  int64_t other_uses{0};   // ordinary stores (must be zero)
  int64_t finalize_dst{0}; // finalize_reducer destination regions
  int64_t copy_src{0};     // tl.copy source regions
  int64_t copy_dst{0};     // tl.copy destination regions
};

struct CopyEdge {
  const VarObj *src{nullptr};
  const VarObj *dst{nullptr};
  bool dst_global{false};
  bool dst_fragment{false};
  TensorVar dst_buffer{UnsafeInit{}};
};

struct UseGraph {
  std::unordered_map<const VarObj *, BufferUseCensus> census;
  std::vector<CopyEdge> copies;
};

/*! \brief Check the destination override chain rooted at `root` and collect
 *  the downstream fragments (excluding the root) that must be overridden
 *  with it. Returns false when any buffer on the chain has uses beyond
 *  "written once by its producer, read only by copies to global or further
 *  chain members". */
bool CollectOverrideChain(const VarObj *root, const UseGraph &graph,
                          std::vector<TensorVar> *chain) {
  std::unordered_set<const VarObj *> visited;
  std::function<bool(const VarObj *)> visit = [&](const VarObj *var) -> bool {
    if (!visited.insert(var).second) {
      return false; // cycle
    }
    auto it = graph.census.find(var);
    BufferUseCensus c =
        it == graph.census.end() ? BufferUseCensus{} : it->second;
    bool is_root = (var == root);
    if (c.stores != 0 || c.other_uses != 0) {
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

/*! \brief Wide-plan destination override. The wide finalize publishes through
 *  an `accumulator -> dst` copy. When LayoutInference gave dst a distributed
 *  free-level layout (picked only to serve the later `dst -> global` copy),
 *  that publication copy partitions by dst ownership and reads the fully
 *  replicated accumulator registers at thread-dependent physical indices;
 *  ptxas lowers the whole array to local memory. When the destination chain
 *  is unconstrained (finalize-written, read only by copies to global or into
 *  further unconstrained fragments), those free-level layouts were arbitrary
 *  choices: replace them with participant-wide replication so every
 *  fragment-to-fragment copy on the chain lowers to per-thread identity
 *  slots and the final global publication takes the static-index
 *  replica-zero path. Returns the layout entries to publish, or nothing when
 *  the chain has other consumers (their inferred layouts then stand). */
std::optional<std::vector<std::pair<TensorVar, Fragment>>>
TryWideFinalizeDstOverride(const EpochInfo &epoch,
                           const Map<TensorVar, Layout> &known_layouts,
                           const UseGraph &use_graph) {
  if (!epoch.dst.defined() || !IsFragmentBuffer(epoch.dst) ||
      epoch.thread_extent <= 0) {
    return std::nullopt;
  }
  // Without an inferred layout MaterializeFinalize already assigns
  // participant-wide replication itself.
  if (!known_layouts.count(epoch.dst)) {
    return std::nullopt;
  }
  std::vector<TensorVar> chain;
  if (!CollectOverrideChain(epoch.dst.get(), use_graph, &chain)) {
    return std::nullopt;
  }
  std::vector<std::pair<TensorVar, Fragment>> entries;
  auto add_entry = [&](const TensorVar &buffer) -> bool {
    for (const auto &dim : buffer.shape()) {
      if (!TryConstInt(dim)) {
        return false;
      }
    }
    Fragment layout = Fragment::FullyReplicated(
        buffer.shape(), IntImm(PrimType::Int(32), epoch.thread_extent));
    if (epoch.thread_min != 0) {
      // Copy loop layouts derived from these fragments carry the thread
      // range; warp-specialized participants need the real one, not
      // FullyReplicated's default [0, extent).
      layout = layout->BindThreadRange(Range(
          IntImm(PrimType::Int(32), epoch.thread_min),
          IntImm(PrimType::Int(32), epoch.thread_min + epoch.thread_extent)));
    }
    entries.emplace_back(buffer, layout);
    return true;
  };
  if (!add_entry(epoch.dst)) {
    return std::nullopt;
  }
  for (const TensorVar &staged : chain) {
    if (!add_entry(staged)) {
      return std::nullopt;
    }
  }
  return entries;
}

std::optional<NarrowDecision>
DeriveNarrowLowering(const EpochInfo &epoch, const PartialFragment &partial,
                     const Map<TensorVar, Layout> &known_layouts,
                     const UseGraph &use_graph, sym::AnalyzerObj *analyzer,
                     std::string *reason) {
  auto fail = [&](const std::string &why) -> std::optional<NarrowDecision> {
    *reason = why;
    return std::nullopt;
  };
  const TensorVar &buffer = epoch.buffer;

  // The solved partial IS the plan: storage and collective steps are read
  // off the node (LayoutInference proved them at the update sites and
  // resolved every failure to the FullParticipant partial, so a narrow node
  // reaching this pass is authoritative). Only lowering details that live
  // outside the layout are derived here: the packed-lane heuristic (serial
  // loop structure) and the destination-containment lowerability gate.
  NarrowDecision decision;
  decision.storage_layout = partial.AsPostCollective();
  decision.steps = partial->CombineSteps();

  // Packed partial accumulation (single-site 16-bit epochs only): pick a
  // lane source whose iterations accumulate serially on one thread. Prefer
  // an enclosing serial loop; otherwise the innermost parallel reduction
  // dim when the layout leaves it an even on-thread run. Lane assignment
  // never changes which thread a contribution lands on, so this is
  // profit-only: any layout keeps the plan correct.
  const Fragment &storage = decision.storage_layout;
  if (epoch.updates.size() == 1 && epoch.analyzable &&
      (buffer.dtype().MatchesElementType(kDLFloat, 16) ||
       buffer.dtype().MatchesElementType(kDLBfloat, 16)) &&
      storage->OutputDim() == 1) {
    const UpdateSite &site = epoch.updates.front();
    size_t ndim = site.loop_vars.size();
    // NOTE: `tirx::Var()` default-constructs a REAL variable named "v",
    // not a null ref; use Optional to represent "no lane source found".
    Optional<PrimVar> lane_var;
    if (!site.serial_vars.empty()) {
      auto extent =
          TryConstInt(site.serial_extents[site.serial_extents.size() - 1]);
      if (extent && *extent >= 2 && *extent % 2 == 0) {
        lane_var = site.serial_vars[site.serial_vars.size() - 1];
      }
    }
    if (!lane_var.has_value() && ndim > 0) {
      // The site analysis provides the loop-level facts the lane heuristic
      // needs (thread splits of the innermost dim, dim survivorship); the
      // plan itself no longer depends on it.
      ReducerUpdateSiteHint hint{site.loop_layout, site.loop_vars, site.indices,
                                 site.value};
      ReducerSiteAnalysis analysis =
          AnalyzeReducerUpdateSite(hint, buffer.shape(), epoch.thread_extent,
                                   epoch.thread_min, analyzer);
      if (analysis.narrow_eligible && !analysis.is_output_dim[ndim - 1]) {
        auto extent = TryConstInt(site.loop_layout->InputShape()[ndim - 1]);
        if (extent) {
          int64_t thread_multiplicity = 1;
          // Packing pays off only when the dim's LOW BIT stays on one
          // thread: a thread-expression split at an odd lower_factor
          // pins the lane parity per thread, leaving one lane idle.
          bool bit0_serial = true;
          auto inner_steps = backend::reduce::CollectThreadReduceSteps(
              analysis.iter_sum.value(), InputPlaceholder(ndim - 1));
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
    }
    if (lane_var.has_value()) {
      Array<PrimExpr> slot_args;
      for (size_t d = 0; d < buffer.shape().size(); ++d) {
        slot_args.push_back(InputPlaceholder(d));
      }
      Array<PrimExpr> fwd_index = storage->Forward(slot_args);
      PrimExpr fwd_thread =
          storage->ForwardThread(slot_args, ReplicationPlaceholder());
      PrimExpr lane = InputPlaceholder(buffer.shape().size());
      Array<PrimExpr> packed_shape = buffer.shape();
      packed_shape.push_back(IntImm(PrimType::Int(32), 2));
      decision.packed_layout =
          Fragment(packed_shape, {fwd_index[0] * 2 + lane}, fwd_thread,
                   storage->ReplicateExtent(), std::nullopt)
              ->BindThreadRange(storage->ThreadRange());
      decision.packed = true;
      decision.pack_lane_var = lane_var.value();
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
    for (size_t d = 0; d < buffer.shape().size(); ++d) {
      PrimVar placeholder = InputPlaceholder(d);
      BindAnalyzer(analyzer, placeholder,
                   Range::FromMinExtent(IntImm::Int32(0), buffer.shape()[d]),
                   /*allow_override=*/true);
      element_indices.push_back(placeholder);
    }
    if (!ProveFragmentContains(dst_frag.value(), decision.storage_layout,
                               element_indices, element_indices,
                               GetRef<sym::Analyzer>(analyzer))) {
      // The inferred layout mismatches the reduction's natural placement.
      // When the whole destination chain is unconstrained (finalize-written,
      // read only by copies to global or by staging copies into further
      // unconstrained fragments), its free-level layouts were arbitrary
      // choices and can be replaced by the induced layout together.
      // Downstream copy lowering re-infers loop layouts against the
      // overrides; ParallelOp validates its candidates against the solved
      // fragment layouts and returns an operand-compatible one, so
      // multi-slot overrides are safe.
      std::vector<TensorVar> chain;
      if (CollectOverrideChain(epoch.dst.get(), use_graph, &chain)) {
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

// TensorRegion has no implicit TensorLoad child in native IR. Census its
// address descriptor once, and count arbitrary tensor/address uses separately.
// Otherwise an unsafe alias could be mistaken for a free finalize destination.
class UseGraphCollector : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;
  UseGraph graph;
  Optional<VisitInterrupt> Visit_(const BindObj *op) final {
    return Visit(op->value);
  }
  Optional<VisitInterrupt> Visit_(const VarObj *op) final {
    if (op->ty.as<TensorTypeObj>())
      graph.census[op].other_uses++;
    return std::nullopt;
  }
  Optional<VisitInterrupt> Visit_(const TensorLoadObj *op) final {
    graph.census[op->source.as_or_throw<TensorVar>().get()].loads++;
    for (const auto &index : op->indices)
      Visit(index);
    return std::nullopt;
  }
  Optional<VisitInterrupt> Visit_(const TensorRegionObj *op) final {
    graph.census[op->source.as_or_throw<TensorVar>().get()].loads++;
    for (const auto &range : op->region) {
      Visit(range->begin);
      Visit(range->extent);
    }
    return std::nullopt;
  }
  Optional<VisitInterrupt> Visit_(const TensorStoreObj *op) final {
    graph.census[op->dest.as_or_throw<TensorVar>().get()].stores++;
    Visit(op->value);
    for (const auto &index : op->indices)
      Visit(index);
    return std::nullopt;
  }
  Optional<VisitInterrupt> Visit_(const RegionStmtObj *op) final {
    if (op->op->name == "tl.attr")
      return Visit(op->body);
    return StmtExprVisitor::Visit_(op);
  }
  Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    if (IsMetadataCall(op))
      return std::nullopt;
    bool copy = op->op.same_as(Copy::Get());
    if (!copy && op->op.as<OpObj>()) {
      TileOperator tile = ParseOperator(GetRef<Call>(op));
      if (const auto *refinement = tile.as<CopyNode>())
        copy = refinement->type_index() != CopyNode::RuntimeTypeIndex();
    }
    if (copy || op->op.same_as(FinalizeReducerV2Op::Get())) {
      auto decoded = DecodeTileOpArguments(op);
      TVM_FFI_CHECK_GE(decoded.operands.size(), 2U, ValueError);
      auto source = NormalizeToTensorRegion(decoded.operands[0]);
      auto dest = NormalizeToTensorRegion(decoded.operands[1]);
      auto src_tensor = tl::RegionTensor(source),
           dst_tensor = tl::RegionTensor(dest);
      if (copy) {
        graph.census[src_tensor.get()].copy_src++;
        graph.census[dst_tensor.get()].copy_dst++;
        graph.copies.push_back({src_tensor.get(), dst_tensor.get(),
                                IsGlobalBuffer(dst_tensor),
                                IsFragmentBuffer(dst_tensor), dst_tensor});
      } else
        graph.census[dst_tensor.get()].finalize_dst++;
      Visit(source);
      Visit(dest);
      for (size_t i = 2; i < decoded.operands.size(); ++i)
        Visit(decoded.operands[i]);
      Visit(decoded.annotations);
      return std::nullopt;
    }
    return StmtExprVisitor::Visit_(op);
  }
};

struct Plan {
  TensorVar old_tensor{UnsafeInit{}};
  TensorVar storage{UnsafeInit{}};
  Fragment layout;
  ReducerV2OpType op;
  Optional<TensorVar> seed_storage;
  Optional<PrimExpr> seed;
  bool narrow{false};
  bool cpu{false};
  std::vector<std::pair<int, int>> steps;
  Optional<TensorVar> packed_storage;
  Fragment packed_layout;
  Optional<PrimVar> pack_lane_var;
};

class ReducerMaterializer : public StmtExprMutatorWithAnalyzer {
public:
  explicit ReducerMaterializer(const sym::Analyzer &analyzer)
      : StmtExprMutatorWithAnalyzer(analyzer) {}
  using StmtExprMutatorWithAnalyzer::Mutate;
  using StmtExprMutatorWithAnalyzer::Mutate_;

  tirx::Function Run(tirx::Function function) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto collector = make_object<ReducerEpochCollector>();
    collector->Collect(function);
    if (collector->epochs.empty())
      return function;
    known_layouts_ = collector->metadata.layouts;
    auto census = make_object<UseGraphCollector>();
    census->Visit(function->body);
    for (const auto &[var, epoch] : collector->epochs) {
      TVM_FFI_CHECK(epoch.dst.defined() && epoch.thread_extent > 0, ValueError)
          << "Reducer epoch is missing init/finalize; run VerifyReducerEpoch "
             "first";
      std::optional<NarrowDecision> decision;
      std::string reason;
      if (!collector->cpu) {
        auto entry = known_layouts_.Get(epoch.buffer);
        TVM_FFI_CHECK(entry.has_value(), ValueError)
            << "No solved partial layout for " << epoch.buffer
            << "; run LayoutInference first";
        auto partial = entry->as_or_throw<PartialFragment>();
        auto combine = TryConstInt(partial->CombineSize());
        auto replication = TryConstInt(partial->ReplicateExtent());
        TVM_FFI_CHECK(combine && replication, ValueError)
            << "Reducer partial decomposition must be compile-time constant";
        bool wide = *combine == *replication &&
                    *replication == epoch.thread_extent &&
                    partial->IsCompletedReplicated();
        if (!wide)
          decision =
              DeriveNarrowLowering(epoch, partial, known_layouts_,
                                   census->graph, GetSyncedAnalyzer(), &reason);
        else
          reason = "FullParticipant partial";
      }
      if (decision && decision->override_dst_layout) {
        extra_layouts_.Set(epoch.dst, decision->storage_layout);
        for (const auto &tensor : decision->override_chain)
          extra_layouts_.Set(tensor, decision->storage_layout);
      } else if (!collector->cpu && !decision) {
        if (auto overrides = TryWideFinalizeDstOverride(epoch, known_layouts_,
                                                        census->graph))
          for (const auto &[tensor, layout] : *overrides)
            extra_layouts_.Set(tensor, layout);
      }
      Plan plan;
      plan.old_tensor = epoch.buffer;
      plan.op = epoch.op;
      plan.cpu = collector->cpu;
      plan.narrow = decision.has_value();
      plan.storage =
          WithScope(epoch.buffer, plan.cpu ? "local" : "local.fragment");
      if (epoch.seed) {
        plan.seed_storage =
            TensorVar(epoch.buffer->name + "_seed",
                      TensorType({1}, epoch.buffer.dtype(),
                                 runtime::MemoryScope::Get("local")),
                      epoch.buffer->loc);
        plan.seed = TensorLoad(*plan.seed_storage, {0}, epoch.buffer->loc);
      }
      if (decision) {
        plan.layout = decision->storage_layout;
        plan.steps = decision->steps;
        if (decision->packed) {
          auto shape = epoch.buffer.shape();
          shape.push_back(IntImm::Int32(2));
          plan.packed_storage =
              TensorVar(epoch.buffer->name + "_pk",
                        TensorType(shape, epoch.buffer.dtype(),
                                   runtime::MemoryScope::Get("local.fragment"),
                                   epoch.buffer.ty()->data_alignment),
                        epoch.buffer->loc);
          plan.packed_layout = decision->packed_layout;
          plan.pack_lane_var = decision->pack_lane_var;
        }
      } else {
        if (!plan.cpu) {
          constexpr int64_t limit = 256;
          int64_t slots = 1;
          for (const auto &extent : epoch.buffer.shape()) {
            auto count = TryConstInt(extent);
            TVM_FFI_CHECK(count && *count > 0, ValueError)
                << "reducer shape must be compile-time positive";
            TVM_FFI_CHECK(*count <= limit && slots <= limit / *count,
                          ValueError)
                << "Reducer needs more than " << limit
                << " per-thread partial slots under the FullParticipant "
                   "baseline";
            slots *= *count;
          }
        }
        plan.layout = Fragment::FullyReplicated(
            epoch.buffer.shape(), IntImm::Int32(epoch.thread_extent));
        plan.layout = plan.layout->BindThreadRange(
            Range(IntImm::Int32(epoch.thread_min),
                  IntImm::Int32(epoch.thread_min + epoch.thread_extent)));
      }
      if (!plan.cpu && IsFragmentBuffer(epoch.dst) &&
          !known_layouts_.count(epoch.dst) &&
          !extra_layouts_.count(epoch.dst)) {
        extra_layouts_.Set(
            epoch.dst,
            plan.narrow ? plan.layout
                        : Fragment::FullyReplicated(
                              epoch.dst.shape(), plan.layout->ReplicateExtent())
                              ->BindThreadRange(plan.layout->ThreadRange()));
        assigned_destinations_.insert(epoch.dst.get());
      }
      if (tl_config::ReducerPlanVerboseEnabled())
        LOG(INFO) << "[ReducerPlan] " << epoch.buffer->name << ": "
                  << (plan.cpu      ? "CPU sequential partial"
                      : plan.narrow ? "narrow"
                                    : "wide FullParticipant")
                  << (plan.packed_storage ? ", packed lanes" : "") << " "
                  << reason;
      plans_.emplace(var, std::move(plan));
    }
    SeqStmt body = Mutate(function->body).ValueOrUnchanged(function->body);
    auto rewrite = make_object<MetadataRewriter>([this](Metadata metadata) {
      metadata.erase(attr::kReducerInfoV2);
      if (auto entry = metadata.Get(attr::kLayoutMap)) {
        Map<TensorVar, Layout> layouts;
        for (const auto &[tensor, layout] : ReadLayouts(*entry)) {
          if (IsReducerV2Buffer(tensor))
            continue;
          layouts.Set(tensor, extra_layouts_.Get(tensor).value_or(layout));
        }
        if (layouts.empty())
          metadata.erase(attr::kLayoutMap);
        else
          metadata.Set(attr::kLayoutMap, layouts);
      }
      return metadata;
    });
    body = rewrite->Mutate(body).ValueOrUnchanged(body);
    function.CopyOnWrite()->body = body;
    return function;
  }

  UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    if (auto found = plans_.find(op->var.get()); found != plans_.end()) {
      const auto *call = op->value.as<CallObj>();
      TVM_FFI_CHECK(call && call->op.same_as(alloc_tensor_op()), ValueError)
          << "A reducer must be an allocation, not an aliased tensor";
      const Plan &plan = found->second;
      Array<Stmt> statements{Allocate(plan.storage, op->loc, call->attrs)};
      Map<TensorVar, Layout> layouts;
      if (!plan.cpu)
        layouts.Set(plan.storage, plan.layout);
      if (plan.packed_storage) {
        statements.push_back(
            Allocate(*plan.packed_storage, op->loc, call->attrs));
        layouts.Set(*plan.packed_storage, plan.packed_layout);
      }
      if (plan.seed_storage)
        statements.push_back(Allocate(*plan.seed_storage, op->loc));
      if (!layouts.empty())
        statements.push_back(
            MakeNativeAnnotation({{attr::kLayoutMap, layouts}}, op->loc));
      return SeqStmt(statements, op->loc);
    }
    auto result =
        StmtExprMutatorWithAnalyzer::Mutate_(op, mode).ValueOrUnchanged(
            GetRef<Stmt>(op));
    if (auto tensor = op->var.as<TensorVar>())
      if (auto extra = extra_layouts_.Get(*tensor))
        return SeqStmt(
            {result,
             MakeNativeAnnotation({{attr::kLayoutMap,
                                    Map<TensorVar, Layout>{{*tensor, *extra}}}},
                                  op->loc)},
            op->loc);
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    if (op->op->name == "tl.attr") {
      auto copy = make_object<RegionStmtObj>(*op);
      copy->body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
      return Stmt(copy);
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    const auto *call = op->value.as<CallObj>();
    if (!call)
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    if (IsMetadataCall(call))
      return Unchanged();
    if (call->op.same_as(ReducerInitOp::Get()))
      return MaterializeInit(call, mode);
    if (call->op.same_as(reducer_update()))
      return MaterializeUpdate(call, mode);
    if (call->op.same_as(FinalizeReducerV2Op::Get()))
      return MaterializeFinalize(call, mode);
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }

private:
  const Plan &FindPlan(const Expr &region) const {
    auto tensor = reducer_native::RegionTensor(region);
    auto found = plans_.find(tensor.get());
    TVM_FFI_CHECK(found != plans_.end(), ValueError)
        << "Unknown reducer " << tensor;
    return found->second;
  }

  Stmt MaterializeInit(const CallObj *call, InplaceMode mode) {
    auto args = DecodeTileOpArguments(call).operands;
    const Plan &plan = FindPlan(args[0]);
    const TensorVar target = plan.packed_storage.value_or(plan.storage);
    Stmt fill = TileCall(
        Fill::Get(),
        {FullRegion(target), ReducerV2Identity(plan.op, target.dtype())}, {},
        call->loc);
    if (plan.seed_storage) {
      TVM_FFI_CHECK_EQ(args.size(), 2U, ValueError)
          << "Duplicated epoch sites disagree about the seed";
      auto value = args[1].as_or_throw<PrimExpr>();
      value = Mutate(value, mode).ValueOrUnchanged(value);
      return SeqStmt(
          {TensorStore(*plan.seed_storage, {0}, value, call->loc), fill},
          call->loc);
    }
    TVM_FFI_CHECK_EQ(args.size(), 1U, ValueError);
    return fill;
  }

  Stmt MaterializeUpdate(const CallObj *call, InplaceMode mode) {
    auto update = ParseReducerUpdate(call);
    const Plan &plan = plans_.at(update.reducer.get());
    auto value = Mutate(update.value, mode).ValueOrUnchanged(update.value);
    Array<PrimExpr> indices;
    for (const auto &index : update.indices)
      indices.push_back(Mutate(index, mode).ValueOrUnchanged(index));
    auto target = plan.packed_storage.value_or(plan.storage);
    if (plan.pack_lane_var)
      indices.push_back(FloorMod(plan.pack_lane_var.value(), IntImm::Int32(2)));
    Stmt store =
        TensorStore(target, indices,
                    ReducerV2Combine(
                        plan.op, TensorLoad(target, indices, call->loc), value),
                    call->loc);
    if (plan.narrow || plan.cpu)
      return store;
    return MakeNativeAttribute(plan.storage, attr::kParallelMultiplicity,
                               IntImm::Int32(1), SeqStmt(store), call->loc);
  }

  // All generated loops use the new begin/end/step contract and their dynamic
  // layout metadata lives in region operands, outside the static For attrs.
  Stmt ElementLoop(const TensorVar &tensor,
                   const std::function<Stmt(const Array<PrimExpr> &)> &element,
                   bool parallel, Location loc,
                   Optional<Fragment> layout = std::nullopt) {
    Array<PrimVar> variables;
    Array<PrimExpr> indices;
    for (size_t d = 0; d < tensor.shape().size(); ++d) {
      PrimVar var("__reducer_" + std::to_string(d), tensor.shape()[d].ty());
      variables.push_back(var);
      indices.push_back(var);
    }
    Stmt body = element(indices);
    for (int d = static_cast<int>(variables.size()) - 1; d >= 0; --d) {
      auto type = variables[d].ty();
      body =
          For(variables[d], IntImm(type, 0), tensor.shape()[d], IntImm(type, 1),
              parallel ? ForKind::kParallel : ForKind::kDefault, SeqStmt(body),
              {}, loc);
    }
    if (layout)
      body = MakeNativeAttribute(nullptr, "tl.loop.annotations",
                                 Metadata{{attr::kParallelLoopLayout, *layout}},
                                 SeqStmt(body), loc);
    return body;
  }

  Stmt MaterializeFinalize(const CallObj *call, InplaceMode mode) {
    auto decoded = DecodeTileOpArguments(call);
    const Plan &plan = FindPlan(decoded.operands[0]);
    auto destination = reducer_native::RegionTensor(decoded.operands[1]);
    Array<Stmt> statements;
    // Finalize consumes whole logical tensors, including when the operands
    // were written as regions. Layout steering is optional, so its shape
    // comparison cannot establish this contract for the generated copy.
    TVM_FFI_CHECK(plan.storage.dtype() == destination.dtype(), TypeError)
        << "finalize_reducer: reducer dtype " << plan.storage.dtype()
        << " does not match destination dtype " << destination.dtype();
    TVM_FFI_CHECK_EQ(plan.storage.shape().size(), destination.shape().size(),
                     ValueError)
        << "finalize_reducer: reducer and destination must have the same "
           "logical shape";
    for (size_t d = 0; d < plan.storage.shape().size(); ++d) {
      const auto &source_extent = plan.storage.shape()[d];
      const auto &destination_extent = destination.shape()[d];
      if (GetSyncedAnalyzer()->CanProveEqual(source_extent, destination_extent))
        continue;
      std::string message =
          "finalize_reducer: reducer and destination must have the same "
          "logical shape (axis " +
          std::to_string(d) + ")";
      TVM_FFI_CHECK(
          !GetSyncedAnalyzer()->CanProve(source_extent != destination_extent),
          ValueError)
          << message << ": " << source_extent << " vs " << destination_extent;
      // Unknown is not unequal. Keep dynamic shapes valid when they agree
      // at runtime, and check before any finalize storage access. This uses
      // the mutator's current branch/assert facts, not an unscoped analyzer.
      statements.push_back(AssertStmt(source_extent == destination_extent,
                                      StringImm("ValueError"),
                                      {StringImm(message)}, call->loc));
    }
    if (plan.cpu) {
      // The CPU kernel has one logical reducer participant per grid task.
      // Capturing the seed remains necessary even without a collective.
      statements.push_back(ElementLoop(
          destination,
          [&](const Array<PrimExpr> &indices) -> Stmt {
            PrimExpr result = TensorLoad(plan.storage, indices, call->loc);
            if (plan.seed)
              result = ReducerV2Combine(plan.op, result, *plan.seed);
            return TensorStore(destination, indices, result, call->loc);
          },
          false, call->loc));
      return SeqStmt(statements, call->loc);
    }
    if (plan.packed_storage) {
      statements.push_back(ElementLoop(
          plan.storage,
          [&](const Array<PrimExpr> &indices) -> Stmt {
            auto first = indices, second = indices;
            first.push_back(IntImm::Int32(0));
            second.push_back(IntImm::Int32(1));
            return TensorStore(
                plan.storage, indices,
                ReducerV2Combine(
                    plan.op, TensorLoad(*plan.packed_storage, first, call->loc),
                    TensorLoad(*plan.packed_storage, second, call->loc)),
                call->loc);
          },
          true, call->loc, plan.layout));
    }
    if (!plan.narrow || !plan.steps.empty() || plan.seed) {
      Array<Expr> args{FullRegion(plan.storage),
                       IntImm::Int32(static_cast<int>(plan.op))};
      Metadata metadata = decoded.annotations;
      if (plan.narrow) {
        metadata.Set("plan", 1);
        for (const auto &[threads, scale] : plan.steps) {
          args.push_back(IntImm::Int32(threads));
          args.push_back(IntImm::Int32(scale));
        }
      }
      if (plan.seed)
        metadata.Set("seed", *plan.seed);
      statements.push_back(
          TileCall(FinalizeReducerOp::Get(), args, metadata, call->loc));
    }
    if (!plan.narrow && assigned_destinations_.count(destination.get())) {
      statements.push_back(ElementLoop(
          destination,
          [&](const Array<PrimExpr> &indices) -> Stmt {
            return TensorStore(destination, indices,
                               TensorLoad(plan.storage, indices, call->loc),
                               call->loc);
          },
          false, call->loc));
    } else {
      statements.push_back(TileCall(
          Copy::Get(), {FullRegion(plan.storage), FullRegion(destination)}, {},
          call->loc));
    }
    return SeqStmt(statements, call->loc);
  }

  Map<TensorVar, Layout> known_layouts_, extra_layouts_;
  std::unordered_map<const VarObj *, Plan> plans_;
  std::unordered_set<const VarObj *> assigned_destinations_;
};
} // namespace

tvm::transform::Pass ReducerPlanAndMaterialize() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    auto analyzer = CreateTileLangAnalyzer();
    return make_object<ReducerMaterializer>(analyzer)->Run(std::move(function));
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.ReducerPlanAndMaterialize");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.ReducerPlanAndMaterialize",
                                   ReducerPlanAndMaterialize);
}
} // namespace tvm::tl
