/*!
 * \file tl/op/reducer.cc
 * \brief Reducer v2 first-class op definitions. These ops are pure IR
 *        carriers: ReducerPlanAndMaterialize consumes them after
 *        LayoutInference; Lower() must never be reached.
 */

#include "reducer.h"

#include <cmath>
#include <limits>
#include <vector>

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>

#include "backend/common/op/reduce.h"
#include "builtin_registry.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tirx;

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

ReducerSiteAnalysis
AnalyzeReducerUpdateSite(const ReducerUpdateSiteHint &site,
                         const ffi::Array<PrimExpr> &reducer_shape,
                         int64_t thread_extent, int64_t thread_min,
                         arith::Analyzer *analyzer) {
  ReducerSiteAnalysis analysis;
  auto reject = [&](const std::string &why) -> ReducerSiteAnalysis {
    analysis.reason = why;
    return analysis;
  };
  if (!site.loop_layout.defined()) {
    return reject("update loop layout unavailable");
  }
  size_t ndim = site.loop_vars.size();
  if (site.loop_layout->InputDim() != ndim) {
    return reject("loop layout rank does not match the parallel nest");
  }
  if (site.indices.size() != reducer_shape.size()) {
    return reject("update index rank does not match the reducer shape");
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
        return reject("update index is not a distinct parallel loop var");
      }
      const int64_t *loop_extent =
          as_const_int(site.loop_layout->InputShape()[pos]);
      const int64_t *dim_extent = as_const_int(reducer_shape[d]);
      if (!loop_extent || !dim_extent || *loop_extent != *dim_extent) {
        return reject("loop extent does not match the reducer dim extent");
      }
      is_output_dim[pos] = true;
      acc_dim_to_loop_dim[d] = pos;
    } else if (is_zero(index)) {
      const int64_t *dim_extent = as_const_int(reducer_shape[d]);
      if (!dim_extent || *dim_extent != 1) {
        return reject("constant update index on a non-unit reducer dim");
      }
    } else {
      return reject("unsupported update index expression");
    }
  }

  // Full-block coverage keeps the collective groups and any garbage
  // threads self-contained and the barrier uniform.
  const int64_t *layout_threads =
      as_const_int(site.loop_layout->ThreadExtent());
  if (!layout_threads || *layout_threads != thread_extent) {
    return reject("loop layout does not cover the full participant extent");
  }
  if (site.loop_layout->ThreadRange().defined()) {
    const int64_t *range_min =
        as_const_int(site.loop_layout->ThreadRange()->min);
    if (!range_min || *range_min != thread_min) {
      return reject("loop layout thread range mismatch");
    }
  } else if (thread_min != 0) {
    return reject("loop layout thread range mismatch");
  }

  if (!ValueIsReplicaSafe(site.value)) {
    return reject("contribution value is not replica-safe");
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
      return reject("induced layout rank mismatch");
    }
    std::vector<PrimExpr> slot_placeholders(expected_rank, PrimExpr());
    bool identity = (expected_rank == reducer_shape.size());
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
      induced = Fragment(reducer_shape, fwd_index, fwd_thread,
                         induced->ReplicateExtent(), std::nullopt)
                    ->BindThreadRange(site.loop_layout->ThreadRange());
    }
  }
  if (induced->InputShape().size() != reducer_shape.size()) {
    return reject("induced layout rank mismatch");
  }
  for (size_t d = 0; d < reducer_shape.size(); ++d) {
    if (!analyzer->CanProveEqual(induced->InputShape()[d], reducer_shape[d])) {
      return reject("induced layout shape mismatch");
    }
  }
  analysis.induced = induced;
  analysis.is_output_dim = is_output_dim;

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
  analysis.iter_sum = arith::NormalizeToIterSum(
      site.loop_layout->GetForwardThread(), var_ranges, analyzer);
  std::vector<backend::reduce::ThreadReduceStep> steps;
  for (size_t i = 0; i < ndim; ++i) {
    if (is_output_dim[i]) {
      continue;
    }
    auto var_steps = backend::reduce::CollectThreadReduceSteps(
        analysis.iter_sum, Downcast<Var>(InputPlaceholder(i)));
    steps.insert(steps.end(), var_steps.begin(), var_steps.end());
  }
  auto is_power_of_two = [](int64_t x) { return x > 0 && (x & (x - 1)) == 0; };
  for (const auto &step : steps) {
    if (!is_power_of_two(step.extent)) {
      return reject("collective width is not a power of two");
    }
    int reducing_threads = step.ReducingThreads();
    if (reducing_threads > thread_extent) {
      return reject("collective width exceeds the participant extent");
    }
    analysis.steps.emplace_back(reducing_threads, step.scale);
  }

  analysis.narrow_eligible = true;
  return analysis;
}

ReducerV2OpType ParseReducerV2OpType(const ffi::String &op_str) {
  if (op_str == "sum")
    return ReducerV2OpType::kSum;
  if (op_str == "max")
    return ReducerV2OpType::kMax;
  if (op_str == "min")
    return ReducerV2OpType::kMin;
  if (op_str == "bitand")
    return ReducerV2OpType::kBitAnd;
  if (op_str == "bitor")
    return ReducerV2OpType::kBitOr;
  if (op_str == "bitxor")
    return ReducerV2OpType::kBitXor;
  LOG(FATAL) << "reducer v2: unsupported combine op `" << op_str
             << "`; expected one of sum/max/min/bitand/bitor/bitxor";
  return ReducerV2OpType::kSum; // unreachable
}

PrimExpr ReducerV2Identity(ReducerV2OpType op, DataType dtype) {
  bool is_int = dtype.is_int();
  bool is_uint = dtype.is_uint();
  int bits = dtype.bits();
  auto signed_min = [&]() -> int64_t {
    return bits >= 64 ? std::numeric_limits<int64_t>::min()
                      : -(static_cast<int64_t>(1) << (bits - 1));
  };
  auto signed_max = [&]() -> int64_t {
    return bits >= 64 ? std::numeric_limits<int64_t>::max()
                      : (static_cast<int64_t>(1) << (bits - 1)) - 1;
  };
  auto unsigned_max = [&]() -> uint64_t {
    return bits >= 64 ? std::numeric_limits<uint64_t>::max()
                      : (static_cast<uint64_t>(1) << bits) - 1;
  };
  auto check_bitwise_dtype = [&]() {
    ICHECK(is_int || is_uint)
        << "bitwise reducer combine ops require an integer dtype, got "
        << dtype;
  };
  switch (op) {
  case ReducerV2OpType::kSum:
    return make_zero(dtype);
  case ReducerV2OpType::kMax:
    if (is_int)
      return make_const(dtype, signed_min());
    if (is_uint)
      return make_const(dtype, 0);
    return make_const(dtype, -INFINITY);
  case ReducerV2OpType::kMin:
    if (is_int)
      return make_const(dtype, signed_max());
    if (is_uint)
      return make_const(dtype, unsigned_max());
    return make_const(dtype, INFINITY);
  case ReducerV2OpType::kBitAnd:
    // All-ones: x & identity == x for every bit pattern.
    check_bitwise_dtype();
    if (is_uint)
      return make_const(dtype, unsigned_max());
    return make_const(dtype, -1);
  case ReducerV2OpType::kBitOr:
  case ReducerV2OpType::kBitXor:
    check_bitwise_dtype();
    return make_zero(dtype);
  }
  LOG(FATAL) << "unreachable";
  return PrimExpr();
}

PrimExpr ReducerV2Combine(ReducerV2OpType op, const PrimExpr &lhs,
                          const PrimExpr &rhs) {
  switch (op) {
  case ReducerV2OpType::kSum:
    return lhs + rhs;
  case ReducerV2OpType::kMax:
    return Max(lhs, rhs);
  case ReducerV2OpType::kMin:
    return Min(lhs, rhs);
  case ReducerV2OpType::kBitAnd:
    return lhs & rhs;
  case ReducerV2OpType::kBitOr:
    return lhs | rhs;
  case ReducerV2OpType::kBitXor:
    return lhs ^ rhs;
  }
  LOG(FATAL) << "unreachable";
  return PrimExpr();
}

// ---------------------------------------------------------------------------
// ReducerInitOp
// ---------------------------------------------------------------------------

ReducerInitOp::ReducerInitOp(ffi::Array<PrimExpr> args,
                             ffi::Map<ffi::String, ffi::ObjectRef>) {
  ICHECK(args.size() == 1 || args.size() == 2)
      << "reducer_init expects (region) or (region, init value)";
  auto node = tvm::ffi::make_object<ReducerInitOpNode>();
  auto access = NormalizeToAccessRegion(args[0], kAccessWrite);
  access.region = BufferRegion::FullRegion(access.region->buffer);
  access.access_mask = kAccessWrite;
  node->reducer = access.region->buffer;
  if (args.size() == 2) {
    ICHECK(args[1].dtype() == node->reducer->dtype)
        << "reducer_init init value dtype " << args[1].dtype()
        << " does not match reducer dtype " << node->reducer->dtype;
    node->seed = args[1];
  }
  node->SetAccessRegions({access});
  data_ = std::move(node);
}

Stmt ReducerInitOpNode::Lower(const LowerArgs &, arith::Analyzer *) const {
  LOG(FATAL) << "reducer_init on `" << reducer
             << "` reached LowerTileOp; it must be materialized by "
                "ReducerPlanAndMaterialize first.";
}

LayoutMap ReducerInitOpNode::InferLayout(const LayoutInferArgs &,
                                         InferLevel) const {
  // The reducer handle carries no fragment layout during inference; the
  // planner assigns physical storage afterwards.
  return {};
}

TileOperator ReducerInitOpNode::Clone() const {
  auto node = tvm::ffi::make_object<ReducerInitOpNode>(*this);
  return TileOperator(node);
}

TIR_REGISTER_TL_TILE_OP(ReducerInitOp, reducer_init)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// ---------------------------------------------------------------------------
// tl.reducer_update: per-iteration builtin intrinsic (see reducer.h)
// ---------------------------------------------------------------------------

ReducerUpdateArgs ParseReducerUpdate(const tirx::CallNode *call) {
  ICHECK(call->op.same_as(reducer_update()));
  ICHECK_EQ(call->args.size(), 2)
      << "reducer_update expects (acc[indices], contribution value)";
  const auto *load = call->args[0].as<BufferLoadNode>();
  ICHECK(load) << "reducer_update target must be written as `acc[indices]` "
                  "in the first argument position, got "
               << call->args[0];
  ReducerUpdateArgs result;
  result.reducer = load->buffer;
  result.indices = load->indices;
  result.value = call->args[1];
  ICHECK(result.value.dtype() == result.reducer->dtype)
      << "reducer_update contribution dtype " << result.value.dtype()
      << " does not match reducer dtype " << result.reducer->dtype;
  return result;
}

TIR_DEFINE_TL_BUILTIN(reducer_update)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// ---------------------------------------------------------------------------
// FinalizeReducerV2Op
// ---------------------------------------------------------------------------

FinalizeReducerV2Op::FinalizeReducerV2Op(
    ffi::Array<PrimExpr> args, ffi::Map<ffi::String, ffi::ObjectRef>) {
  ICHECK_EQ(args.size(), 2)
      << "finalize_reducer (v2) expects (reducer region, dst region)";
  auto node = tvm::ffi::make_object<FinalizeReducerV2OpNode>();
  auto reducer_access = NormalizeToAccessRegion(args[0], kAccessReadWrite);
  reducer_access.region =
      BufferRegion::FullRegion(reducer_access.region->buffer);
  reducer_access.access_mask = kAccessReadWrite;
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  dst_access.region = BufferRegion::FullRegion(dst_access.region->buffer);
  dst_access.access_mask = kAccessWrite;
  node->reducer = reducer_access.region->buffer;
  node->dst = dst_access.region->buffer;
  ICHECK(node->reducer->dtype == node->dst->dtype)
      << "finalize_reducer: reducer dtype " << node->reducer->dtype
      << " does not match destination dtype " << node->dst->dtype;
  ICHECK_EQ(node->reducer->shape.size(), node->dst->shape.size())
      << "finalize_reducer: reducer and destination must have the same "
         "logical shape";
  node->SetAccessRegions({reducer_access, dst_access});
  data_ = std::move(node);
}

Stmt FinalizeReducerV2OpNode::Lower(const LowerArgs &,
                                    arith::Analyzer *) const {
  LOG(FATAL) << "finalize_reducer (v2) on `" << reducer
             << "` reached LowerTileOp; it must be materialized by "
                "ReducerPlanAndMaterialize first.";
}

LayoutMap FinalizeReducerV2OpNode::InferLayout(const LayoutInferArgs &args,
                                               InferLevel level) const {
  // dst is an ordinary fragment: producers/consumers may constrain it at any
  // level and finalize never overrules them. At kFree, when nothing has
  // constrained it, propose the layout the reducer plan itself will pick
  // (dst-steering): the update sites' induced layout when every narrow-plan
  // site proof passes, participant-wide replication otherwise. The proofs
  // are shared with ReducerPlanAndMaterialize (AnalyzeReducerUpdateSite), so
  // the proposal is the planner's own verdict computed early — a steered dst
  // can never make the plan worse, it only removes the arbitrary free-mode
  // choice that used to break narrow containment or (wide) force a
  // thread-indexed publish copy.
  if (level != InferLevel::kFree) {
    return {};
  }
  if (args.layout_map.count(dst)) {
    return {};
  }
  if (args.reducer_update_sites == nullptr ||
      args.reducer_update_sites->empty()) {
    return {};
  }
  if (!CanSteerDst(reducer, dst, args.thread_bounds, args.analyzer)) {
    return {};
  }
  const int64_t *extent_ptr = as_const_int(args.thread_bounds->extent);
  const int64_t *min_ptr = as_const_int(args.thread_bounds->min);
  ICHECK(extent_ptr && min_ptr); // guaranteed by CanSteerDst
  // Stay silent while any update nest is unsolved: a later call in this
  // attempt (or another attempt ordering) sees the solved state.
  for (const ReducerUpdateSiteHint &site : *args.reducer_update_sites) {
    if (!site.loop_layout.defined()) {
      return {};
    }
  }
  bool steer_narrow = true;
  Fragment induced;
  std::vector<std::pair<int, int>> steps;
  bool first = true;
  for (const ReducerUpdateSiteHint &site : *args.reducer_update_sites) {
    ReducerSiteAnalysis analysis = AnalyzeReducerUpdateSite(
        site, reducer->shape, *extent_ptr, *min_ptr, args.analyzer);
    // Mirror the planner's multi-site agreement check as well: sites that
    // are individually eligible but induce different plans still fall back
    // to the wide baseline.
    if (!analysis.narrow_eligible) {
      steer_narrow = false;
      break;
    }
    if (first) {
      induced = analysis.induced;
      steps = std::move(analysis.steps);
      first = false;
    } else if (!StructuralEqual()(induced, analysis.induced) ||
               steps != analysis.steps) {
      steer_narrow = false;
      break;
    }
  }
  LayoutMap result;
  if (steer_narrow) {
    result.Set(dst, induced);
    return result;
  }
  // The epoch will take the wide plan: after its participant-wide AllReduce
  // every thread holds every logical output, and a replicated dst keeps the
  // publish copy (and any further fragment staging) a per-thread identity
  // move instead of a thread-indexed gather from replicated registers.
  result.Set(dst, FallbackDstLayout(dst, args.thread_bounds));
  return result;
}

Fragment
FinalizeReducerV2OpNode::FallbackDstLayout(const Buffer &dst,
                                           const Range &thread_bounds) {
  return Fragment::FullyReplicated(dst->shape, thread_bounds->extent)
      ->BindThreadRange(thread_bounds);
}

bool FinalizeReducerV2OpNode::CanSteerDst(const Buffer &reducer,
                                          const Buffer &dst,
                                          const Range &thread_bounds,
                                          arith::Analyzer *analyzer) {
  if (!IsFragmentBuffer(dst)) {
    return false;
  }
  // The induced layout is expressed over the reducer's logical shape; it can
  // be handed to dst verbatim only when the shapes match per dim (the ctor
  // checks dtype and rank, not extents).
  if (reducer->shape.size() != dst->shape.size()) {
    return false;
  }
  for (size_t d = 0; d < reducer->shape.size(); ++d) {
    if (!analyzer->CanProveEqual(reducer->shape[d], dst->shape[d])) {
      return false;
    }
  }
  const int64_t *extent_ptr = as_const_int(thread_bounds->extent);
  const int64_t *min_ptr = as_const_int(thread_bounds->min);
  return extent_ptr && min_ptr && *extent_ptr > 1;
}

TileOperator FinalizeReducerV2OpNode::Clone() const {
  auto node = tvm::ffi::make_object<FinalizeReducerV2OpNode>(*this);
  return TileOperator(node);
}

TIR_REGISTER_TL_TILE_OP(FinalizeReducerV2Op, finalize_reducer_v2)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// ---------------------------------------------------------------------------
// FinalizeReducerOp: the materialized collective
// ---------------------------------------------------------------------------

namespace {

std::vector<FinalizeReducerImpl> &FinalizeReducerImplRegistry() {
  static std::vector<FinalizeReducerImpl> registry;
  return registry;
}

const FinalizeReducerImpl &ResolveFinalizeReducerImpl(Target target) {
  const auto &registry = FinalizeReducerImplRegistry();
  const FinalizeReducerImpl *matched_impl = nullptr;
  for (const FinalizeReducerImpl &impl : registry) {
    if (impl.match_target(target)) {
      ICHECK(matched_impl == nullptr)
          << "tl.finalize_reducer found multiple target-specific "
             "implementations for "
          << target->str() << ": " << matched_impl->name << " and "
          << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.finalize_reducer requires a target-specific implementation, but "
         "no finalize_reducer implementation is registered for "
      << target->str();
  return *matched_impl;
}

} // namespace

void RegisterFinalizeReducerImpl(FinalizeReducerImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.lower != nullptr);
  FinalizeReducerImplRegistry().push_back(impl);
}

FinalizeReducerOp::FinalizeReducerOp(
    ffi::Array<PrimExpr> args,
    ffi::Map<ffi::String, ffi::ObjectRef> annotations) {
  auto node = tvm::ffi::make_object<FinalizeReducerOpNode>();
  auto reducer_access = NormalizeToAccessRegion(args[0], kAccessReadWrite);
  reducer_access.region =
      BufferRegion::FullRegion(reducer_access.region->buffer);
  reducer_access.access_mask = kAccessReadWrite;
  node->reducer = reducer_access.region->buffer;
  node->SetAccessRegions({reducer_access});
  node->op = (ReducerV2OpType)*as_const_int(args[1]);
  // Optional explicit collective plan (reducer v2 narrow plans): flattened
  // (reducing_threads, scale) pairs starting at args[2].
  ICHECK(args.size() % 2 == 0) << "finalize_reducer: plan steps must come in "
                                  "(reducing_threads, scale) pairs";
  for (size_t i = 2; i + 1 < args.size(); i += 2) {
    int reducing_threads = (int)*as_const_int(args[i]);
    int scale = (int)*as_const_int(args[i + 1]);
    ICHECK_GT(reducing_threads, 0)
        << "finalize_reducer: explicit reducing_threads must be positive";
    ICHECK_GT(scale, 0) << "finalize_reducer: explicit scale must be positive";
    node->plan_steps.push_back(Integer(reducing_threads));
    node->plan_steps.push_back(Integer(scale));
  }
  // Read explicit batch size from annotations (0 means auto-detect).
  if (annotations.count("batch")) {
    node->batch = (int)*as_const_int(Downcast<PrimExpr>(annotations["batch"]));
    ICHECK_GE(node->batch, 1)
        << "finalize_reducer: batch must be >= 1, got " << node->batch;
  }
  if (annotations.count("plan")) {
    node->explicit_plan = true;
  }
  if (annotations.count("seed")) {
    node->seed = Downcast<PrimExpr>(annotations["seed"]);
  }
  data_ = std::move(node);
}

Stmt FinalizeReducerOpNode::Lower(const LowerArgs &lower_args,
                                  arith::Analyzer *analyzer) const {
  return ResolveFinalizeReducerImpl(lower_args.target)
      .lower(*this, lower_args, analyzer);
}

LayoutMap FinalizeReducerOpNode::InferLayout(const LayoutInferArgs &layout_args,
                                             InferLevel level) const {
  // Materialized after LayoutInference; preserves the storage layout the
  // planner assigned.
  LayoutMap layout_map;
  layout_map.Set(reducer, layout_args.layout_map.Get(reducer).value());
  return layout_map;
}

TileOperator FinalizeReducerOpNode::Clone() const {
  auto node = tvm::ffi::make_object<FinalizeReducerOpNode>(*this);
  return TileOperator(node);
}

TIR_REGISTER_TL_TILE_OP(FinalizeReducerOp, finalize_reducer)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  ReducerInitOpNode::RegisterReflection();
  FinalizeReducerV2OpNode::RegisterReflection();
  FinalizeReducerOpNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
