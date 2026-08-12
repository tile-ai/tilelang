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

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include "utils.h"

namespace tvm {
namespace tl {

using namespace tirx;

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
  ICHECK_EQ(args.size(), 1)
      << "reducer_init expects exactly one region argument";
  auto node = tvm::ffi::make_object<ReducerInitOpNode>();
  auto access = NormalizeToAccessRegion(args[0], kAccessWrite);
  access.region = BufferRegion::FullRegion(access.region->buffer);
  access.access_mask = kAccessWrite;
  node->reducer = access.region->buffer;
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
// ReducerUpdateOp
// ---------------------------------------------------------------------------

ReducerUpdateOp::ReducerUpdateOp(ffi::Array<PrimExpr> args,
                                 ffi::Map<ffi::String, ffi::ObjectRef>) {
  ICHECK_EQ(args.size(), 2)
      << "reducer_update expects (target region, contribution value)";
  auto node = tvm::ffi::make_object<ReducerUpdateOpNode>();
  auto access = NormalizeToAccessRegion(args[0], kAccessReadWrite);
  access.access_mask = kAccessReadWrite;
  node->reducer = access.region->buffer;
  for (const auto &range : access.region->region) {
    ICHECK(is_one(range->extent))
        << "reducer_update target must be a point region (one logical "
           "output), got extent "
        << range->extent << " on `" << node->reducer << "`";
    node->indices.push_back(range->min);
  }
  node->value = args[1];
  ICHECK(node->value.dtype() == node->reducer->dtype)
      << "reducer_update contribution dtype " << node->value.dtype()
      << " does not match reducer dtype " << node->reducer->dtype;
  node->SetAccessRegions({access});
  data_ = std::move(node);
}

Stmt ReducerUpdateOpNode::Lower(const LowerArgs &, arith::Analyzer *) const {
  LOG(FATAL) << "reducer_update on `" << reducer
             << "` reached LowerTileOp; it must be materialized by "
                "ReducerPlanAndMaterialize first.";
}

LayoutMap ReducerUpdateOpNode::InferLayout(const LayoutInferArgs &,
                                           InferLevel) const {
  return {};
}

TileOperator ReducerUpdateOpNode::Clone() const {
  auto node = tvm::ffi::make_object<ReducerUpdateOpNode>(*this);
  return TileOperator(node);
}

TIR_REGISTER_TL_TILE_OP(ReducerUpdateOp, reducer_update)
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

LayoutMap FinalizeReducerV2OpNode::InferLayout(const LayoutInferArgs &,
                                               InferLevel) const {
  // dst is an ordinary fragment; its layout is inferred from consumers.
  return {};
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
  ReducerUpdateOpNode::RegisterReflection();
  FinalizeReducerV2OpNode::RegisterReflection();
  FinalizeReducerOpNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
