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
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/stmt_functor.h>

#include "backend/common/op/reduce_layout.h"
#include "builtin_registry.h"
#include "compat/visit.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

bool ValueIsReplicaSafe(const PrimExpr &value) {
  if (SideEffect(value) > CallEffectKind::kReadState) {
    return false;
  }
  bool safe = true;
  PostOrderVisit(value, [&](const ObjectRef &obj) {
    if (const auto *load = obj.as<TensorLoadNode>()) {
      const TensorVar &buffer = load->source.as_or_throw<TensorVar>();
      if (!IsFragmentBuffer(buffer) && !IsSharedBuffer(buffer) &&
          !IsGlobalBuffer(buffer)) {
        safe = false;
      }
    }
  });
  return safe;
}

/*! \brief One projection step of the induced-layout construction: identical
 *  to ComputeReducerLayout minus the scale-ordered
 *  CondenseReplicateVar (which would destroy the low-bits combine/copy
 *  boundary) and the thread-range bind (done once after the final,
 *  boundary-preserving condensation). The projected dim's lanes take the
 *  LOW bits of the new replication coordinate, the pre-existing replication
 *  moves high, so successive projections keep every reduction-sourced lane
 *  contiguous at the bottom. */
static Fragment ProjectReducerDimNoCondense(const Fragment &src_layout,
                                            int dim) {
  PrimExpr src_rep_extent = src_layout->ReplicateExtent();
  PrimExpr indice_rep_extent = src_layout->InputShape()[dim];
  PrimExpr reducer_rep_extent = indice_rep_extent * src_rep_extent;
  auto fwd = backend::reduce::InputPlaceholders(src_layout->InputDim() - 1);
  fwd.insert(fwd.begin() + dim,
             FloorMod(ReplicationPlaceholder(), indice_rep_extent));
  auto thd = src_layout->ForwardThread(
      fwd, FloorDiv(ReplicationPlaceholder(), indice_rep_extent));
  auto reducer_shape = src_layout->InputShape();
  reducer_shape.erase(reducer_shape.begin() + dim);
  if (reducer_shape.empty()) {
    reducer_shape.push_back(1);
  }
  return Fragment(reducer_shape, {}, thd, reducer_rep_extent, std::nullopt);
}

ReducerSiteAnalysis
AnalyzeReducerUpdateSite(const ReducerUpdateSiteHint &site,
                         const ffi::Array<PrimExpr> &reducer_shape,
                         int64_t thread_extent, int64_t thread_min,
                         sym::Analyzer analyzer) {
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
      auto loop_extent = GetConstInt(site.loop_layout->InputShape()[pos]);
      auto dim_extent = GetConstInt(reducer_shape[d]);
      if (!loop_extent || !dim_extent || *loop_extent != *dim_extent) {
        return reject("loop extent does not match the reducer dim extent");
      }
      is_output_dim[pos] = true;
      acc_dim_to_loop_dim[d] = pos;
    } else if (IsZero(index)) {
      auto dim_extent = GetConstInt(reducer_shape[d]);
      if (!dim_extent || *dim_extent != 1) {
        return reject("constant update index on a non-unit reducer dim");
      }
    } else {
      return reject("unsupported update index expression");
    }
  }

  // Full-block coverage keeps the collective groups and any garbage
  // threads self-contained and the barrier uniform.
  auto layout_threads = GetConstInt(site.loop_layout->ThreadExtent());
  if (!layout_threads || *layout_threads != thread_extent) {
    return reject("loop layout does not cover the full participant extent");
  }
  if (site.loop_layout->ThreadRange().has_value()) {
    auto range_min =
        GetConstInt(site.loop_layout->ThreadRange().value()->begin);
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
  // the surviving loop dims in NEST order. Every projection stacks the
  // projected dim's lanes below the existing replication, so the combine
  // block (reduction-sourced lanes) stays contiguous in the low bits with
  // the loop's own replication (copy groups) above it; one final
  // boundary-preserving condensation compacts both halves without mixing
  // them, keeping `_rep % combine` = addend lanes true on the result.
  Fragment induced = site.loop_layout;
  PrimExpr combine_raw = MakeConst(PrimType::Int(32), 1);
  for (int dim = static_cast<int>(ndim) - 1; dim >= 0; --dim) {
    if (!is_output_dim[dim]) {
      combine_raw = combine_raw * induced->InputShape()[dim];
      induced = ProjectReducerDimNoCondense(induced, dim);
    }
  }
  {
    auto [condensed, combine_expr] =
        CondenseReplicateVarKeepingBoundary(induced, combine_raw);
    induced = condensed->BindThreadRange(site.loop_layout->ThreadRange());
    auto combine_ptr = GetConstInt(combine_expr);
    if (!combine_ptr.has_value()) {
      return reject("non-constant combine width");
    }
    analysis.combine_size = *combine_ptr;
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
    std::vector<PrimExpr> slot_placeholders(expected_rank,
                                            PrimExpr(ffi::UnsafeInit{}));
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
  Map<PrimVar, Range> var_ranges;
  for (size_t i = 0; i < ndim; ++i) {
    var_ranges.Set(InputPlaceholder(i),
                   Range::FromMinExtent(MakeConst(PrimType::Int(32), 0),
                                        site.loop_layout->InputShape()[i]));
  }
  var_ranges.Set(ReplicationPlaceholder(),
                 Range::FromMinExtent(MakeConst(PrimType::Int(32), 0),
                                      site.loop_layout->ReplicateExtent()));
  analysis.iter_sum = sym::NormalizeToIterSum(
      site.loop_layout->GetForwardThread(), var_ranges, analyzer);
  std::vector<backend::reduce::ThreadReduceStep> steps;
  for (size_t i = 0; i < ndim; ++i) {
    if (is_output_dim[i]) {
      continue;
    }
    auto var_steps = backend::reduce::CollectThreadReduceSteps(
        site.loop_layout->GetForwardThread(), var_ranges, InputPlaceholder(i),
        analyzer);
    steps.insert(steps.end(), var_steps.begin(), var_steps.end());
  }
  auto is_power_of_two = [](int64_t x) { return x > 0 && (x & (x - 1)) == 0; };
  for (const auto &step : steps) {
    if (!is_power_of_two(step.extent)) {
      return reject("collective width is not a power of two");
    }
    // The XOR butterfly can only address the reduce coordinate when the
    // thread stride between consecutive participants is a power of two.
    if (!is_power_of_two(step.scale)) {
      return reject("collective thread stride is not a power of two");
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

PrimExpr ReducerV2Identity(ReducerV2OpType op, PrimType dtype) {
  bool is_int = dtype.MatchesCode(kDLInt);
  bool is_uint = dtype.MatchesCode(kDLUInt);
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
    return MakeConst(dtype, 0);
  case ReducerV2OpType::kMax:
    if (is_int)
      return MakeConst(dtype, signed_min());
    if (is_uint)
      return MakeConst(dtype, 0);
    return MakeConst(dtype, -INFINITY);
  case ReducerV2OpType::kMin:
    if (is_int)
      return MakeConst(dtype, signed_max());
    if (is_uint)
      return MakeConst(dtype, unsigned_max());
    return MakeConst(dtype, INFINITY);
  case ReducerV2OpType::kBitAnd:
    // All-ones: x & identity == x for every bit pattern.
    check_bitwise_dtype();
    if (is_uint)
      return MakeConst(dtype, unsigned_max());
    return MakeConst(dtype, -1);
  case ReducerV2OpType::kBitOr:
  case ReducerV2OpType::kBitXor:
    check_bitwise_dtype();
    return MakeConst(dtype, 0);
  }
  LOG(FATAL) << "unreachable";
  return PrimExpr(ffi::UnsafeInit{});
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
  return PrimExpr(ffi::UnsafeInit{});
}

// ---------------------------------------------------------------------------
// ReducerInitOp
// ---------------------------------------------------------------------------

ReducerInitOp::ReducerInitOp(ffi::Array<Expr> args, BlockAnnotations) {
  ICHECK(args.size() == 1 || args.size() == 2)
      << "reducer_init expects (region) or (region, init value)";
  auto node = tvm::ffi::make_object<ReducerInitOpNode>(ffi::UnsafeInit{});
  auto access = NormalizeToAccessRegion(args[0], kAccessWrite);
  access.region =
      NormalizeToBufferRegion(access.region->source.as_or_throw<TensorVar>());
  access.access_mask = kAccessWrite;
  node->reducer = access.region->source.as_or_throw<TensorVar>();
  if (args.size() == 2) {
    ICHECK(args[1].as_or_throw<PrimExpr>().ty() == node->reducer.dtype())
        << "reducer_init init value dtype "
        << args[1].as_or_throw<PrimExpr>().ty()
        << " does not match reducer dtype " << node->reducer.dtype();
    node->seed = args[1].as_or_throw<PrimExpr>();
  }
  node->SetAccessRegions({access});
  data_ = std::move(node);
}

Stmt ReducerInitOpNode::Lower(const LowerArgs &, sym::Analyzer) const {
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

TIR_REGISTER_TL_TILE_OP(ReducerInitOp, reducer_init, -1);

// ---------------------------------------------------------------------------
// tl.reducer_update: per-iteration builtin intrinsic (see reducer.h)
// ---------------------------------------------------------------------------

ReducerUpdateArgs ParseReducerUpdate(const CallNode *call) {
  ICHECK(call->op.same_as(reducer_update()));
  ICHECK_EQ(call->args.size(), 2)
      << "reducer_update expects (acc[indices], contribution value)";
  const auto *load = call->args[0].as<TensorLoadNode>();
  ICHECK(load) << "reducer_update target must be written as `acc[indices]` "
                  "in the first argument position, got "
               << call->args[0];
  ReducerUpdateArgs result{load->source.as_or_throw<TensorVar>(), load->indices,
                           call->args[1].as_or_throw<PrimExpr>()};
  ICHECK(result.value.ty() == result.reducer.dtype())
      << "reducer_update contribution dtype " << result.value.ty()
      << " does not match reducer dtype " << result.reducer.dtype();
  return result;
}

TIR_DEFINE_TL_BUILTIN(reducer_update, 2, kOpaque, false);

// ---------------------------------------------------------------------------
// FinalizeReducerV2Op
// ---------------------------------------------------------------------------

FinalizeReducerV2Op::FinalizeReducerV2Op(ffi::Array<Expr> args,
                                         BlockAnnotations) {
  ICHECK_EQ(args.size(), 2)
      << "finalize_reducer (v2) expects (reducer region, dst region)";
  auto node = tvm::ffi::make_object<FinalizeReducerV2OpNode>(ffi::UnsafeInit{});
  auto reducer_access = NormalizeToAccessRegion(args[0], kAccessReadWrite);
  reducer_access.region = NormalizeToBufferRegion(
      reducer_access.region->source.as_or_throw<TensorVar>());
  reducer_access.access_mask = kAccessReadWrite;
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  dst_access.region = NormalizeToBufferRegion(
      dst_access.region->source.as_or_throw<TensorVar>());
  dst_access.access_mask = kAccessWrite;
  node->reducer = reducer_access.region->source.as_or_throw<TensorVar>();
  node->dst = dst_access.region->source.as_or_throw<TensorVar>();
  ICHECK(node->reducer.dtype() == node->dst.dtype())
      << "finalize_reducer: reducer dtype " << node->reducer.dtype()
      << " does not match destination dtype " << node->dst.dtype();
  ICHECK_EQ(node->reducer.shape().size(), node->dst.shape().size())
      << "finalize_reducer: reducer and destination must have the same "
         "logical shape";
  node->SetAccessRegions({reducer_access, dst_access});
  data_ = std::move(node);
}

Stmt FinalizeReducerV2OpNode::Lower(const LowerArgs &, sym::Analyzer) const {
  LOG(FATAL) << "finalize_reducer (v2) on `" << reducer
             << "` reached LowerTileOp; it must be materialized by "
                "ReducerPlanAndMaterialize first.";
}

LayoutMap FinalizeReducerV2OpNode::InferLayout(const LayoutInferArgs &args,
                                               InferLevel level) const {
  // dst is an ordinary fragment: producers/consumers may constrain it at any
  // level and finalize never overrules them. At kFree, when nothing has
  // constrained it, propose the post-collective reading of the reducer's
  // solved partial layout (dst-steering). The partial IS the plan verdict:
  // update nests propose the induced projection when every per-site narrow
  // proof passes (AnalyzeReducerUpdateSite, shared with the planner), the
  // engine widens on multi-site disagreement, and zero-update epochs are
  // pre-seeded wide — so a steered dst can never make the plan worse, it
  // only removes the arbitrary free-mode choice that used to break narrow
  // containment or (wide) force a thread-indexed publish copy.
  if (level != InferLevel::kFree) {
    return {};
  }
  if (args.layout_map.count(dst)) {
    return {};
  }
  if (!CanSteerDst(reducer, dst, args.thread_bounds, args.analyzer)) {
    return {};
  }
  // Stay silent while the reducer is unsolved: the nest's PartialFragment
  // commit re-enqueues this op through use_list_.
  auto partial_entry = args.layout_map.Get(reducer);
  if (!partial_entry.has_value()) {
    return {};
  }
  auto partial = partial_entry.value().as<PartialFragment>();
  ICHECK(partial.has_value())
      << "reducer " << reducer << " carries a non-partial layout: "
      << partial_entry.value()->DebugOutput();
  // After the finalize collective every replica of a partial holds the
  // combined value, so the same algebraic map read as a plain Fragment is
  // exactly the destination's natural placement. For the wide plan this is
  // the fully replicated dst that keeps the publish copy a per-thread
  // identity move instead of a thread-indexed gather.
  Fragment dst_layout = partial.value().AsPostCollective();
  if (!dst_layout->ThreadRange().has_value()) {
    // User-annotated partials carry no thread range; bind the epoch's
    // participant range so downstream copy predicates stay well-formed.
    dst_layout = dst_layout->BindThreadRange(args.thread_bounds);
  }
  LayoutMap result;
  result.Set(dst, dst_layout);
  return result;
}

Fragment
FinalizeReducerV2OpNode::FallbackDstLayout(const TensorVar &dst,
                                           const Range &thread_bounds) {
  return Fragment::FullyReplicated(dst.shape(), thread_bounds->extent)
      ->BindThreadRange(thread_bounds);
}

bool FinalizeReducerV2OpNode::CanSteerDst(const TensorVar &reducer,
                                          const TensorVar &dst,
                                          const Range &thread_bounds,
                                          sym::Analyzer analyzer) {
  if (!IsFragmentBuffer(dst)) {
    return false;
  }
  // The induced layout is expressed over the reducer's logical shape; it can
  // be handed to dst verbatim only when the shapes match per dim (the ctor
  // checks dtype and rank, not extents).
  if (reducer.shape().size() != dst.shape().size()) {
    return false;
  }
  for (size_t d = 0; d < reducer.shape().size(); ++d) {
    if (!analyzer->CanProveEqual(reducer.shape()[d], dst.shape()[d])) {
      return false;
    }
  }
  auto extent_ptr = GetConstInt(thread_bounds->extent);
  auto min_ptr = GetConstInt(thread_bounds->begin);
  return extent_ptr && min_ptr && *extent_ptr > 0;
}

TileOperator FinalizeReducerV2OpNode::Clone() const {
  auto node = tvm::ffi::make_object<FinalizeReducerV2OpNode>(*this);
  return TileOperator(node);
}

TIR_REGISTER_TL_TILE_OP(FinalizeReducerV2Op, finalize_reducer_v2, 2);

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
          << target << ": " << matched_impl->name << " and " << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.finalize_reducer requires a target-specific implementation, but "
         "no finalize_reducer implementation is registered for "
      << target;
  return *matched_impl;
}

} // namespace

void RegisterFinalizeReducerImpl(FinalizeReducerImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.lower != nullptr);
  FinalizeReducerImplRegistry().push_back(impl);
}

FinalizeReducerOp::FinalizeReducerOp(ffi::Array<Expr> args,
                                     BlockAnnotations annotations) {
  auto node = tvm::ffi::make_object<FinalizeReducerOpNode>(ffi::UnsafeInit{});
  auto reducer_access = NormalizeToAccessRegion(args[0], kAccessReadWrite);
  reducer_access.region = NormalizeToBufferRegion(
      reducer_access.region->source.as_or_throw<TensorVar>());
  reducer_access.access_mask = kAccessReadWrite;
  node->reducer = reducer_access.region->source.as_or_throw<TensorVar>();
  node->SetAccessRegions({reducer_access});
  node->op = (ReducerV2OpType)*GetConstInt(args[1]);
  // Optional explicit collective plan (reducer v2 narrow plans): flattened
  // (reducing_threads, scale) pairs starting at args[2].
  ICHECK(args.size() % 2 == 0) << "finalize_reducer: plan steps must come in "
                                  "(reducing_threads, scale) pairs";
  for (size_t i = 2; i + 1 < args.size(); i += 2) {
    int reducing_threads = (int)*GetConstInt(args[i]);
    int scale = (int)*GetConstInt(args[i + 1]);
    ICHECK_GT(reducing_threads, 0)
        << "finalize_reducer: explicit reducing_threads must be positive";
    ICHECK_GT(scale, 0) << "finalize_reducer: explicit scale must be positive";
    node->plan_steps.push_back(IntImm(PrimType::Int(32), reducing_threads));
    node->plan_steps.push_back(IntImm(PrimType::Int(32), scale));
  }
  // Read explicit batch size from annotations (0 means auto-detect).
  if (annotations.count("batch")) {
    auto batch = GetIntAnnotation(annotations.at("batch"));
    TVM_FFI_CHECK(batch && *batch >= 1 &&
                      *batch <= std::numeric_limits<int>::max(),
                  ValueError)
        << "finalize_reducer: batch must be a positive integer";
    node->batch = static_cast<int>(*batch);
  }
  if (annotations.count("plan")) {
    node->explicit_plan = true;
  }
  if (annotations.count("seed")) {
    node->seed = annotations.at("seed").as_or_throw<PrimExpr>();
  }
  data_ = std::move(node);
}

Stmt FinalizeReducerOpNode::Lower(const LowerArgs &lower_args,
                                  sym::Analyzer analyzer) const {
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

TIR_REGISTER_TL_TILE_OP(FinalizeReducerOp, finalize_reducer, -1);

TVM_FFI_STATIC_INIT_BLOCK() {
  ReducerInitOpNode::RegisterReflection();
  FinalizeReducerV2OpNode::RegisterReflection();
  FinalizeReducerOpNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
