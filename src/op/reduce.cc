/*!
 * \file tl/op/reduce.cc
 * \brief Implementation of reduction operators
 */

#include "reduce.h"
#include "ir/analyzer.h"
#include "layout/expr_utils.h"
#include "support/check.h"
#include <tvm/ffi/extra/structural_equal.h>

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/stmt_functor.h>

#include "../layout/layout.h"
#include "../layout/utils.h"
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
using namespace prim;
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

ReduceOp::ReduceOp(Array<Expr> args, Map<String, Any> annotations) {
  ICHECK_EQ(args.size(), 5U);
  ObjectPtr<ReduceOpNode> node = make_object<ReduceOpNode>();
  node->annotations = annotations;
  // Accept TensorRegion/TensorLoad for src/dst
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessReadWrite);
  node->srcRegion_ = src_access.region;
  node->dstRegion_ = dst_access.region;
  node->SetAccessRegions({src_access, dst_access});
  node->src = RegionTensor(node->srcRegion_);
  node->dst = RegionTensor(node->dstRegion_);
  std::string reduce_type = args[2].as<StringImm>().value()->value;
  node->dim = static_cast<int>(GetIntegerArgument(args[3]));
  node->type = ReduceType(reduce_type);
  node->clear = GetBoolAnnotation(args[4]);
  if (auto value = annotations.Get("batch")) {
    auto batch = GetIntegerAnnotation(value.value());
    ICHECK_GE(batch, 1);
    ICHECK_LE(batch, std::numeric_limits<int>::max());
    node->batch = static_cast<int>(batch);
  }
  if (auto value = annotations.Get("nan_propagate"))
    node->nan_propagate = GetBoolAnnotation(value.value());
  ICHECK_GE(node->dim, 0);
  ICHECK_LT(static_cast<size_t>(node->dim), node->src.shape().size());
  if (node->type->IsBitAnd() || node->type->IsBitOr() ||
      node->type->IsBitXor()) {
    ICHECK(node->dst.dtype().MatchesCode(kDLInt, kDLUInt, kDLBool))
        << "Bitwise reductions require an integer or bool tensor";
  }
  const bool is_minmax =
      node->type->IsMax() || node->type->IsMin() || node->type->IsAbsMax();

  if (node->nan_propagate && is_minmax) {
    CHECK(node->dst.dtype().MatchesElementType(kDLFloat, 16) ||
              node->dst.dtype().MatchesElementType(kDLBfloat, 16),
          ValueError)
        << "reduce_" << reduce_type
        << " with nan_propagate=True requires float16 or bfloat16 output, got "
        << node->dst.dtype();
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
                         sym::AnalyzerObj *analyzer) const {
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

    // Batch is a contract on the per-thread accumulator, so check it while
    // source layouts are still candidates. A wider memory vector can leave
    // fewer output slots per thread than the requested batch. Reject that
    // candidate and let normal layout inference try another owner mapping;
    // never silently change the requested batch or an explicit layout.
    if (batch > 1) {
      int64_t elements = 1;
      for (const auto &extent : reducer_layout->OutputShape()) {
        const auto count = TryConstInt(extent);
        if (!count || *count <= 0 ||
            elements > std::numeric_limits<int64_t>::max() / *count) {
          throw LayoutConflictException("ReduceOp: batch > 1 requires a "
                                        "constant per-thread output shape");
        }
        elements *= *count;
      }
      if (elements < batch || elements % batch != 0) {
        std::ostringstream message;
        message << "ReduceOp: batch=" << batch
                << " must evenly divide the per-thread output element count N="
                << elements << " of the candidate source layout";
        throw LayoutConflictException(message.str());
      }
    }

    // Match the frontend's reduction shape contract before Reshape, which
    // only checks the total element count.
    Array<PrimExpr> expected_shape = reducer_layout->InputShape();
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
    auto analyzer = CreateTileLangAnalyzer();
    auto indices = InputPlaceholders(reducer_layout->InputDim());
    for (size_t i = 0; i < indices.size(); i++) {
      BindAnalyzer(analyzer, indices[i].as_or_throw<PrimVar>(),
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

TVM_FFI_STATIC_INIT_BLOCK() {
  ReduceOpNode::RegisterReflection();
  ReduceTypeNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
