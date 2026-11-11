/*!
 * \file tl/op/arg_reduce.cc
 * \brief Contracts and layout inference for argmin/argmax.
 */
#include "arg_reduce.h"

#include <limits>
#include <numeric>
#include <vector>

#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include "layout/utils.h"
#include "op/utils.h"
#include "support/check.h"

namespace tvm {
namespace tl {
using namespace tirx;
using namespace ffi;

namespace {
std::vector<ArgReduceImpl> &Implementations() {
  static std::vector<ArgReduceImpl> impls;
  return impls;
}

Fragment DefaultLayout(const Buffer &buffer, const Range &threads) {
  PrimExpr flat = Integer(0);
  int64_t elements = 1;
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    flat = flat * buffer->shape[i] + InputPlaceholder(i);
    int64_t extent = *as_const_int(buffer->shape[i]);
    CHECK(elements <= std::numeric_limits<int>::max() / extent, ValueError)
        << "arg_reduce tile is too large for GPU indexing";
    elements *= extent;
  }
  int64_t count = *as_const_int(threads->extent);
  int64_t lanes = std::gcd(elements, count);
  // Ordinary fragment consumers require an exact ownership map. Use
  // replication for a ragged default instead of changing the logical shape
  // or introducing padded slots that those consumers cannot decode.
  return Fragment(buffer->shape, {FloorDiv(flat, Integer(lanes))},
                  FloorMod(flat, Integer(lanes)) +
                      ReplicationPlaceholder() * Integer(lanes),
                  Integer(count / lanes), std::nullopt)
      ->BindThreadRange(threads);
}
} // namespace

void RegisterArgReduceImpl(ArgReduceImpl impl) {
  Implementations().push_back(impl);
}

ArgReduceOp::ArgReduceOp(Array<PrimExpr> args, Map<String, ObjectRef>) {
  CHECK(args.size() == 4, ValueError)
      << "arg_reduce expects source, destination, kind, and axis";
  auto node = make_object<ArgReduceOpNode>();
  auto src = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst = NormalizeToAccessRegion(args[1], kAccessWrite);
  // Effects are fixed by this operator, not by the region transport masks.
  src.access_mask = kAccessRead;
  dst.access_mask = kAccessWrite;
  node->src_region = src.region;
  node->dst_region = dst.region;
  node->SetAccessRegions({src, dst});
  auto kind = args[2].as<StringImm>();
  CHECK(kind && (kind.value()->value == "max" || kind.value()->value == "min"),
        ValueError)
      << "arg_reduce kind must be max or min";
  node->is_max = kind.value()->value == "max";
  auto axis = args[3].as<IntImm>();
  int rank = src.region->region.size();
  CHECK(axis && axis.value()->dtype.is_int() && axis.value()->value >= -rank &&
            axis.value()->value < rank,
        ValueError)
      << "arg_reduce axis is out of range for input rank " << rank;
  node->dim = axis.value()->value;
  if (node->dim < 0)
    node->dim += rank;

  arith::Analyzer analyzer;
  for (const auto &region : {src.region, dst.region}) {
    Buffer buffer = region->buffer;
    CHECK(IsSharedBuffer(buffer) || IsFragmentBuffer(buffer), ValueError)
        << "arg_reduce requires shared or fragment buffers, got "
        << buffer.scope();
    for (size_t i = 0; i < region->region.size(); ++i) {
      const Range &range = region->region[i];
      auto extent = analyzer.Simplify(range->extent).as<IntImm>();
      CHECK(extent && extent.value()->value > 0, ValueError)
          << "arg_reduce requires positive static region extents";
      if (IsFragmentBuffer(buffer)) {
        CHECK(analyzer.CanProveEqual(range->min, 0) &&
                  analyzer.CanProveEqual(range->extent, buffer->shape[i]),
              ValueError)
            << "arg_reduce requires a full fragment; copy to shared memory to "
               "select a subregion";
      }
      CHECK(
          !analyzer.CanProve(range->min < 0) &&
              !analyzer.CanProve(range->min + range->extent > buffer->shape[i]),
          ValueError)
          << "arg_reduce region is outside buffer bounds";
    }
  }
  DataType value_type = src.region->buffer->dtype;
  bool integer = (value_type.is_int() || value_type.is_uint()) &&
                 (value_type.bits() == 8 || value_type.bits() == 16 ||
                  value_type.bits() == 32 || value_type.bits() == 64);
  CHECK(value_type.is_scalar() &&
            (integer || value_type.is_float16() || value_type.is_bfloat16() ||
             value_type == DataType::Float(32) ||
             value_type == DataType::Float(64)),
        TypeError)
      << "arg_reduce requires a scalar integer or floating-point input, got "
      << value_type;
  DataType index_type = dst.region->buffer->dtype;
  CHECK(index_type == DataType::Int(32) || index_type == DataType::Int(64),
        TypeError)
      << "arg_reduce requires an int32 or int64 output, got " << index_type;
  if (index_type == DataType::Int(32)) {
    CHECK(*as_const_int(
              analyzer.Simplify(src.region->region[node->dim]->extent)) <=
              std::numeric_limits<int32_t>::max(),
          ValueError)
        << "arg_reduce axis extent exceeds int32 index capacity";
  }
  Array<PrimExpr> expected;
  if (rank == 1) {
    expected = {Integer(1)};
  } else {
    for (int i = 0; i < rank; ++i) {
      if (i != node->dim)
        expected.push_back(src.region->region[i]->extent);
      else if (dst.region->region.size() == rank)
        expected.push_back(Integer(1));
    }
  }
  bool shape_matches = expected.size() == dst.region->region.size();
  for (size_t i = 0; shape_matches && i < expected.size(); ++i) {
    shape_matches =
        analyzer.CanProveEqual(expected[i], dst.region->region[i]->extent);
  }
  CHECK(shape_matches, ValueError)
      << "Invalid arg_reduce output shape: expected " << expected;
  if (src.region->buffer->data.same_as(dst.region->buffer->data)) {
    bool disjoint = false;
    if (src.region->buffer.same_as(dst.region->buffer) &&
        src.region->region.size() == dst.region->region.size()) {
      for (size_t i = 0; i < src.region->region.size(); ++i) {
        Range a = src.region->region[i], b = dst.region->region[i];
        disjoint |= analyzer.CanProve(a->min + a->extent <= b->min) ||
                    analyzer.CanProve(b->min + b->extent <= a->min);
      }
    }
    CHECK(disjoint, ValueError)
        << "arg_reduce input and output must not overlap";
  }
  data_ = std::move(node);
}

TileOperator ArgReduceOpNode::Clone() const {
  return ArgReduceOp(make_object<ArgReduceOpNode>(*this));
}

Stmt ArgReduceOpNode::Lower(const LowerArgs &args,
                            arith::Analyzer *analyzer) const {
  const ArgReduceImpl *matched = nullptr;
  for (const auto &impl : Implementations()) {
    if (impl.match_target(args.target)) {
      ICHECK(matched == nullptr)
          << "Multiple arg_reduce implementations for " << args.target;
      matched = &impl;
    }
  }
  CHECK(matched != nullptr, ValueError)
      << "arg_reduce has no implementation for target " << args.target;
  return matched->lower(*this, args, analyzer);
}

LayoutMap ArgReduceOpNode::InferLayout(const LayoutInferArgs &args,
                                       InferLevel level) const {
  if (level >= InferLevel::kStrict)
    return {};
  LayoutMap result;
  Buffer src = src_region->buffer, dst = dst_region->buffer;
  if (IsFragmentBuffer(src) && args.layout_map.count(src) &&
      !args.strict_layout_map.count(src)) {
    auto layout = Downcast<Fragment>(args.layout_map[src]);
    PrimExpr logical_size = layout->ReplicateExtent();
    PrimExpr physical_size = layout->ThreadExtent();
    for (const auto &extent : layout->InputShape())
      logical_size = logical_size * extent;
    for (const auto &extent : layout->OutputShape())
      physical_size = physical_size * extent;
    if (!args.analyzer->CanProveEqual(logical_size, physical_size)) {
      // Reject this free-inference candidate so the search can select the
      // exact default below. Do not pad or reshape an existing GEMM layout.
      throw LayoutConflictException(
          "arg_reduce requires an exact fragment ownership map; the inferred "
          "source layout has padded physical slots");
    }
  }
  if (IsFragmentBuffer(src) && !args.layout_map.count(src) &&
      level == InferLevel::kFree) {
    result.Set(src, DefaultLayout(src, args.thread_bounds));
  }
  if (IsFragmentBuffer(dst) && !args.layout_map.count(dst)) {
    if (IsFragmentBuffer(src) && args.layout_map.count(src)) {
      auto layout = Downcast<Fragment>(args.layout_map[src]);
      result.Set(dst, ComputeReducerLayout(layout, dim)
                          ->Reshape(dst->shape, args.analyzer));
    } else if (level == InferLevel::kFree) {
      result.Set(dst, DefaultLayout(dst, args.thread_bounds));
    }
  }
  // Existing destination constraints are legal: lowering can redistribute via
  // shared memory.
  return result;
}

TIR_REGISTER_TL_TILE_OP(ArgReduceOp, arg_reduce)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  ArgReduceOpNode::RegisterReflection();
  reflection::GlobalDef().def(
      "tl.ArgReduceOp",
      [](Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
        return ArgReduceOp(std::move(args), std::move(annotations));
      });
}
} // namespace tl
} // namespace tvm
