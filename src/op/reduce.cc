/*!
 * \file tl/op/reduce.cc
 * \brief Implementation of reduction operators
 */

#include "reduce.h"
#include "support/check.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/cast.h>

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>

#include "../layout/layout.h"
#include "../layout/utils.h"
#include "../op/parallel.h"
#include "../transform/loop_partition.h"
#include "builtin.h"
#include "tir/transforms/ir_utils.h"
#include "tvm/ir/expr.h"
#include "utils.h"
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <sstream>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

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

void CheckReductionDType(DataType dtype, const char *op) {
  CHECK(dtype.bits() >= 8 || dtype.is_bool(), ValueError)
      << op << " does not support sub-byte accumulator dtype " << dtype
      << ". Use float16/float32 or an integer dtype of at least 8 bits for "
         "the output/reducer buffer.";
}

ReduceOp::ReduceOp(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<ReduceOpNode> node = make_object<ReduceOpNode>();
  node->annotations = annotations;
  // Accept BufferRegion/BufferLoad for src/dst
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessReadWrite);
  node->srcRegion_ = src_access.region;
  node->dstRegion_ = dst_access.region;
  node->SetAccessRegions({src_access, dst_access});
  node->src = node->srcRegion_->buffer;
  node->dst = node->dstRegion_->buffer;
  CheckReductionDType(node->dst->dtype, "reduce");
  std::string reduce_type = args[2].as<StringImm>().value()->value;
  node->dim = args[3].as<IntImm>().value()->value;
  node->type = ReduceType(reduce_type);
  node->clear = args[4].as<Bool>().value();
  if (node->type->IsLogical()) {
    for (const auto &[name, buffer] :
         {std::pair{"input", node->src}, {"output", node->dst}}) {
      CHECK(IsSharedBuffer(buffer) || IsFragmentBuffer(buffer), ValueError)
          << "reduce_" << reduce_type << " requires a shared or fragment "
          << name << " buffer, got " << buffer.scope();
    }
    DataType src_dtype = node->src->dtype;
    CHECK(src_dtype.is_scalar() &&
              (src_dtype.is_bool() || src_dtype.is_int() ||
               src_dtype.is_uint() || src_dtype.is_float16() ||
               src_dtype == DataType::Float(32) ||
               src_dtype == DataType::Float(64) || src_dtype.is_bfloat16()),
          TypeError)
        << "reduce_" << reduce_type
        << " requires a boolean, integer, or floating-point input, got "
        << src_dtype;
    CHECK(node->dst->dtype == DataType::Bool(), TypeError)
        << "reduce_" << reduce_type << " requires a bool output, got "
        << node->dst->dtype;
  } else if (node->type->IsBitAnd() || node->type->IsBitOr() ||
             node->type->IsBitXor()) {
    for (const auto &[name, buffer] :
         {std::pair{"input", node->src}, {"output", node->dst}}) {
      DataType dtype = buffer->dtype;
      CHECK(dtype.is_bool() || dtype.is_int() || dtype.is_uint(), ValueError)
          << "reduce_" << reduce_type << " requires an integer/bool " << name
          << " buffer, got dtype " << dtype;
    }
  }

  arith::Analyzer analyzer;
  const auto &src_region = node->srcRegion_->region;
  const auto &dst_region = node->dstRegion_->region;
  bool whole_region = node->type->IsLogical() && node->dim == -1;
  CHECK(whole_region ||
            (node->dim >= 0 && node->dim < static_cast<int>(src_region.size())),
        ValueError)
      << "reduce_" << reduce_type << " axis " << node->dim
      << " is out of range for input rank " << src_region.size();

  // Shared operands are staged through fragments by the language API. These
  // restrictions belong to that path, not to the reduction kind or local path.
  for (const auto &[name, region] :
       {std::pair{"input", node->srcRegion_}, {"output", node->dstRegion_}}) {
    if (!IsSharedBuffer(region->buffer) && !IsFragmentBuffer(region->buffer)) {
      continue;
    }
    for (size_t i = 0; i < region->region.size(); ++i) {
      const Range &range = region->region[i];
      if (IsFragmentBuffer(region->buffer)) {
        CHECK(
            analyzer.CanProveEqual(range->min, 0) &&
                analyzer.CanProveEqual(range->extent, region->buffer->shape[i]),
            ValueError)
            << "reduce_" << reduce_type
            << " currently requires a full fragment " << name
            << "; copy the full fragment to shared memory before "
               "selecting a subregion";
      }
      PrimExpr simplified_extent = analyzer.Simplify(range->extent);
      const auto *extent = simplified_extent.as<IntImmNode>();
      CHECK(extent && extent->value >= 0, ValueError)
          << "reduce_" << reduce_type
          << " requires non-negative static region extents for fragment "
             "lowering; use a fixed-size tile with neutral padding";
    }
  }

  Array<PrimExpr> expected_shape;
  if (whole_region || (src_region.size() == 1 && (IsFragmentBuffer(node->dst) ||
                                                  IsSharedBuffer(node->dst)))) {
    expected_shape = {Integer(1)};
  } else {
    for (size_t i = 0; i < src_region.size(); ++i) {
      PrimExpr extent = src_region[i]->extent;
      if (i != node->dim) {
        expected_shape.push_back(extent);
      } else if (src_region.size() == dst_region.size()) {
        expected_shape.push_back(make_const(extent.dtype(), 1));
      }
    }
  }
  bool valid_shape = expected_shape.size() == dst_region.size();
  for (size_t i = 0; valid_shape && i < expected_shape.size(); ++i) {
    valid_shape =
        analyzer.CanProveEqual(expected_shape[i], dst_region[i]->extent);
  }
  CHECK(valid_shape, ValueError)
      << "Invalid reduce output shape for reduce_" << reduce_type << ": "
      << node->dstRegion_ << ", expected " << expected_shape;
  // Optional "batch" annotation: number of output elements per batched
  // AllReduce call (default 1 = scalar).
  if (auto opt = annotations.Get("batch")) {
    if (auto i = opt.value().as<IntImm>()) {
      node->batch = static_cast<int>(i.value()->value);
      ICHECK_GE(node->batch, 1) << "ReduceOp: batch must be >= 1";
    }
  }
  // Optional annotation: "nan_propagate" — for fp16/bf16 max/min/absmax,
  // when true, lower to CUDA __hmax_nan/__hmin_nan so NaNs propagate.
  if (auto opt = annotations.Get("nan_propagate")) {
    if (auto b = opt.value().as<Bool>()) {
      node->nan_propagate = b.value();
    } else if (auto i = opt.value().as<IntImm>()) {
      node->nan_propagate = i.value()->value != 0;
    }
  }
  const bool is_minmax =
      node->type->IsMax() || node->type->IsMin() || node->type->IsAbsMax();

  if (node->nan_propagate && is_minmax) {
    CHECK(node->dst->dtype.is_float16() || node->dst->dtype.is_bfloat16(),
          ValueError)
        << "reduce_" << reduce_type
        << " with nan_propagate=True requires float16 or bfloat16 output, got "
        << node->dst->dtype;
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

Fragment ReduceOpNode::NormalizeSourceLayoutForReduction(
    const Fragment &layout, arith::Analyzer *analyzer) const {
  if (dim != -1) {
    return layout;
  }
  ICHECK(type->IsLogical());
  PrimExpr extent = Integer(1);
  for (const PrimExpr &size : layout->InputShape()) {
    extent = analyzer->Simplify(extent * size);
  }
  return Downcast<Fragment>(layout->Reshape({extent}, analyzer));
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
                         arith::Analyzer *analyzer) const {
  return ResolveReduceImpl(lower_args.target)
      .lower(*this, lower_args, analyzer);
}

LayoutMap ReduceOpNode::InferLayout(const LayoutInferArgs &layout_args,
                                    InferLevel level) const {
  if (level >= InferLevel::kStrict)
    return {};

  if (IsFragmentBuffer(src) && IsFragmentBuffer(dst) &&
      layout_args.layout_map.count(src)) {
    auto src_layout = NormalizeSourceLayoutForReduction(
        layout_args.layout_map[src].as<Fragment>().value(),
        layout_args.analyzer);
    int reduce_dim = ReductionDim();
    auto reducer_layout = ComputeReducerLayout(src_layout, reduce_dim);

    reducer_layout = Downcast<Fragment>(
        reducer_layout->Reshape(dst->shape, layout_args.analyzer));

    if (!layout_args.layout_map.count(dst)) {
      return {{dst, reducer_layout}};
    }

    auto orig_dst_layout =
        layout_args.layout_map.Get(dst).value().as<Fragment>().value();
    ICHECK(reducer_layout->InputDim() == orig_dst_layout->InputDim());

    // Keep placeholder bindings local to this containment check.
    arith::Analyzer analyzer;
    auto indices = InputPlaceholders(reducer_layout->InputDim());
    for (size_t i = 0; i < indices.size(); i++) {
      analyzer.Bind(Downcast<Var>(indices[i]),
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

TIR_REGISTER_TL_TILE_OP(ReduceOp, reduce)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  ReduceOpNode::RegisterReflection();
  ReduceTypeNode::RegisterReflection();
  reflection::GlobalDef().def(
      "tl.ReduceOp",
      [](Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
        return ReduceOp(std::move(args), std::move(annotations));
      });
}

} // namespace tl
} // namespace tvm
