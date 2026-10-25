/*!
 * \file tl/op/reduce.cc
 * \brief Implementation of reduction operators
 */

#include "reduce.h"
#include "support/check.h"
#include <tvm/ffi/extra/structural_equal.h>

#include <tvm/tirx/stmt_functor.h>

#include "../layout/layout.h"
#include "../layout/utils.h"
#include "../op/parallel.h"
#include "../transform/loop_partition.h"
#include "builtin.h"
#include "tvm/ir/expr.h"
#include "utils.h"
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <limits>
#include <sstream>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

int64_t IntegerArgument(const Any &argument, const char *name) {
  if (auto value = argument.as<bool>()) {
    return value.value();
  }
  if (auto value = argument.as<int64_t>()) {
    return value.value();
  }
  if (auto expression = argument.as<PrimExpr>()) {
    if (auto value = GetConstInt(expression.value())) {
      return *value;
    }
  }
  TVM_FFI_THROW(ValueError)
      << "ReduceOp " << name << " must be a constant integer";
}

std::vector<ReduceImpl> &ReduceImplRegistry() {
  static std::vector<ReduceImpl> registry;
  return registry;
}

const ReduceImpl &ResolveReduceImpl(Target target) {
  const auto &registry = ReduceImplRegistry();
  const ReduceImpl *matched_impl = nullptr;
  for (const ReduceImpl &impl : registry) {
    if (impl.match_target(target)) {
      ICHECK(matched_impl == nullptr)
          << "tl.reduce found multiple target-specific implementations for "
          << target->str() << ": " << matched_impl->name << " and "
          << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.reduce requires a target-specific implementation, but no reduce "
         "implementation is registered for "
      << target->str();
  return *matched_impl;
}

} // namespace

void RegisterReduceImpl(ReduceImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.lower != nullptr);
  ReduceImplRegistry().push_back(impl);
}

ReduceOp::ReduceOp(Array<Expr> args, BlockAnnotations annotations) {
  TVM_FFI_CHECK_EQ(args.size(), 5, ValueError)
      << "ReduceOp expects five arguments";
  ObjectPtr<ReduceOpNode> node = make_object<ReduceOpNode>(ffi::UnsafeInit{});
  node->annotations = annotations;
  // Accept TensorRegion/TensorLoad for src/dst
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessReadWrite);
  node->srcRegion_ = src_access.region;
  node->dstRegion_ = dst_access.region;
  node->SetAccessRegions({src_access, dst_access});
  node->src = node->srcRegion_->source.as_or_throw<TensorVar>();
  node->dst = node->dstRegion_->source.as_or_throw<TensorVar>();
  std::string reduce_type = args[2].as_or_throw<StringImm>()->value;
  int64_t dimension = IntegerArgument(args[3], "dimension");
  TVM_FFI_CHECK(dimension >= 0 &&
                    dimension < static_cast<int64_t>(node->src.shape().size()),
                ValueError)
      << "ReduceOp dimension is out of range";
  node->dim = dimension;
  node->type = ReduceType(reduce_type);
  int64_t clear = IntegerArgument(args[4], "clear");
  TVM_FFI_CHECK(clear == 0 || clear == 1, ValueError)
      << "ReduceOp clear must be boolean";
  node->clear = clear != 0;
  // Optional "batch" annotation: number of output elements per batched
  // AllReduce call (default 1 = scalar).
  if (auto opt = annotations.Get("batch")) {
    int64_t batch = IntegerArgument(opt.value(), "batch");
    TVM_FFI_CHECK(batch >= 1 && batch <= std::numeric_limits<int>::max(),
                  ValueError)
        << "ReduceOp batch must be a positive integer";
    node->batch = batch;
  }
  // Optional annotation: "nan_propagate" — for fp16/bf16 max/min/absmax,
  // when true, lower to CUDA __hmax_nan/__hmin_nan so NaNs propagate.
  if (auto opt = annotations.Get("nan_propagate")) {
    int64_t propagate = IntegerArgument(opt.value(), "nan_propagate");
    TVM_FFI_CHECK(propagate == 0 || propagate == 1, ValueError)
        << "ReduceOp nan_propagate must be boolean";
    node->nan_propagate = propagate != 0;
  }
  const bool is_minmax =
      node->type->IsMax() || node->type->IsMin() || node->type->IsAbsMax();

  if (node->nan_propagate && is_minmax) {
    TVM_FFI_CHECK(node->dst.dtype().MatchesElementType(kDLFloat, 16) ||
                      node->dst.dtype().MatchesElementType(kDLBfloat, 16),
                  ValueError)
        << "reduce_" << reduce_type
        << " with nan_propagate=True requires float16 or bfloat16 output, got "
        << node->dst.dtype();
  }
  if (node->type->IsBitAnd() || node->type->IsBitOr() ||
      node->type->IsBitXor()) {
    TVM_FFI_CHECK(node->src.dtype().MatchesCode(kDLInt, kDLUInt) &&
                      node->dst.dtype().MatchesCode(kDLInt, kDLUInt),
                  ValueError)
        << "ReduceOp bitwise reductions require integer element types";
  }
  sym::Analyzer analyzer;
  const auto &source_ranges = node->srcRegion_->region;
  const auto &destination_ranges = node->dstRegion_->region;
  TVM_FFI_CHECK(destination_ranges.size() == source_ranges.size() ||
                    destination_ranges.size() + 1 == source_ranges.size(),
                ValueError)
      << "ReduceOp destination rank must remove or retain the reduced axis";
  for (const TensorRegion &region : {node->srcRegion_, node->dstRegion_}) {
    auto tensor = region->source.as_or_throw<TensorVar>();
    for (size_t axis = 0; axis < region->region.size(); ++axis) {
      const auto &domain = region->region[axis];
      TVM_FFI_CHECK(!analyzer->CanProve(domain->begin < 0) &&
                        !analyzer->CanProve(domain->extent < 0) &&
                        !analyzer->CanProve(domain->begin + domain->extent >
                                            tensor.shape()[axis]),
                    ValueError)
          << "ReduceOp region must lie within its tensor";
    }
  }
  bool keep_dimension = destination_ranges.size() == source_ranges.size();
  for (size_t axis = 0; axis < destination_ranges.size(); ++axis) {
    size_t source_axis = keep_dimension || axis < static_cast<size_t>(node->dim)
                             ? axis
                             : axis + 1;
    PrimExpr expected =
        keep_dimension && axis == static_cast<size_t>(node->dim)
            ? MakeConst(destination_ranges[axis]->extent.ty(), 1)
            : source_ranges[source_axis]->extent;
    TVM_FFI_CHECK(
        !analyzer->CanProve(destination_ranges[axis]->extent != expected),
        ValueError)
        << "ReduceOp destination region has an incompatible extent";
  }
  data_ = std::move(node);
}

AccessRegions ReduceOpNode::GetAccessRegions() const {
  AccessRegions result;
  result.reads.push_back(srcRegion_);
  if (!clear) {
    result.reads.push_back(dstRegion_);
  }
  result.writes.push_back(dstRegion_);
  return result;
}

TileOperator ReduceOpNode::Clone() const {
  auto op = make_object<ReduceOpNode>(*this);
  return ReduceOp(op);
}

static Array<PrimExpr> InputPlaceholders(size_t n) {
  Array<PrimExpr> result;
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    result.push_back(InputPlaceholder(i));
  }
  return result;
}

/**
 * @brief Lower the Reduce operator to a TIR statement.
 *
 * Lowers a ReduceOpNode operating on fragment-scoped buffers into a sequence of
 * TIR statements implementing: optional initialization, thread-local reduction
 * (unrolled inner loops), inter-thread reduction via a backend-provided
 * runtime AllReduce call, and an optional accumulation or copy back to the
 * destination buffer when a temporary clear buffer is used.
 *
 * Behavior notes:
 * - Only supports src and dst in "local.fragment" scope; otherwise it checks
 *   and aborts with "Reduce for shared memory not implemented.".
 * - Supports both 1D reductions (scalar output) and reductions along a single
 *   extra dimension; validates layout dimensionality consistency.
 * - If `clear` is set (or for sum/abssum reductions), an initial value is
 *   written to the clear buffer; for non-clearing sum/abssum a duplicate
 *   temporary buffer is allocated and accumulated back into dst after
 * reduction.
 * - Performs iterator compression for local reduction loops using `analyzer`.
 * - Detects parallel thread splitting from the normalized iterator sum and
 *   emits a call to a templated `tl::AllReduce<...>::run`
 *   via `builtin::call_extern`. For sufficiently large reducing thread counts
 *   (> 32) a workspace is allocated via T.add_workspace and passed to the
 *   AllReduce call.
 * - The final body is wrapped in parallel loops over the destination spatial
 *   dimensions and partitioned by the lowering thread variable. If a temporary
 *   clear buffer is used, it is allocated for the body.
 *
 * @param lower_args Lowering context providing buffer and layout maps, thread
 * bounds, target information, thread variable, and workspace allocation helper.
 * @param analyzer Analyzer used for iterator compression and arithmetic
 * normalization.
 * @return Stmt Lowered TIR statement implementing the reduction.
 */
Stmt ReduceOpNode::Lower(const LowerArgs &lower_args,
                         sym::Analyzer analyzer) const {
  return ResolveReduceImpl(lower_args.target)
      .lower(*this, lower_args, analyzer);
}

LayoutMap ReduceOpNode::InferLayout(const LayoutInferArgs &layout_args,
                                    InferLevel level) const {
  if (level >= InferLevel::kStrict)
    return {};

  if (IsFragmentBuffer(src) && IsFragmentBuffer(dst) &&
      layout_args.layout_map.count(src)) {
    auto src_layout = layout_args.layout_map[src].as<Fragment>().value();
    auto reducer_layout = ComputeReducerLayout(src_layout, this->dim);
    if (batch > 1) {
      PrimExpr elements = 1;
      for (const PrimExpr &extent : reducer_layout->OutputShape())
        elements = elements * extent;
      auto count = GetConstInt(layout_args.analyzer->Simplify(elements));
      if (!count || *count < batch || *count % batch != 0) {
        throw LayoutConflictException(
            "ReduceOp batch must evenly divide the per-thread output "
            "element count of the source layout");
      }
    }

    // Match the frontend's reduction shape contract before Reshape, which
    // only checks the total element count.
    Array<PrimExpr> expected_shape = reducer_layout->InputShape();
    if (src_layout->InputDim() == 1 && dst.shape().empty()) {
      expected_shape.clear();
    }
    if (dst.shape().size() == src_layout->InputShape().size()) {
      // Keeping the reduced axis requires its extent to be one.
      expected_shape = src_layout->InputShape();
      expected_shape.Set(dim, MakeConst(expected_shape[dim].ty(), 1));
    }
    bool valid_shape = dst.shape().size() == expected_shape.size();
    for (size_t i = 0; valid_shape && i < expected_shape.size(); ++i) {
      valid_shape = layout_args.analyzer->CanProveEqual(dst.shape()[i],
                                                        expected_shape[i]);
    }
    ICHECK(valid_shape) << "ReduceOp: invalid destination shape " << dst.shape()
                        << " for " << dst << ", expected " << expected_shape
                        << " when reducing axis " << dim << " of "
                        << src.shape();
    reducer_layout = reducer_layout->Reshape(dst.shape(), layout_args.analyzer)
                         .as_or_throw<Fragment>();

    if (!layout_args.layout_map.count(dst)) {
      return {{dst, reducer_layout}};
    }

    auto orig_dst_layout =
        layout_args.layout_map.Get(dst).value().as<Fragment>().value();
    ICHECK(reducer_layout->InputDim() == orig_dst_layout->InputDim());

    // Keep placeholder bindings local to this containment check.
    sym::Analyzer analyzer;
    auto indices = InputPlaceholders(reducer_layout->InputDim());
    for (size_t i = 0; i < indices.size(); i++) {
      analyzer->Bind(indices[i].as_or_throw<PrimVar>(),
                     Range(0, reducer_layout->InputShape()[i]));
    }
    if (!ProveFragmentContains(orig_dst_layout, reducer_layout, indices,
                               indices, analyzer)) {
      std::ostringstream oss;
      oss << "Layout may conflict with ReduceOp for buffer " << dst << " vs. "
          << src << "\n"
          << "src_layout = " << src_layout << "\n"
          << "reducer_layout = " << reducer_layout << "\n"
          << "orig_dst_layout = " << orig_dst_layout << "\n"
          << "You may need to use a shared memory to transform the "
             "layout";
      throw LayoutConflictException(oss.str());
    }
  }
  return {};
}

TIR_REGISTER_TL_TILE_OP(ReduceOp, reduce, 5)

TVM_FFI_STATIC_INIT_BLOCK() {
  ReduceOpNode::RegisterReflection();
  ReduceTypeNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
