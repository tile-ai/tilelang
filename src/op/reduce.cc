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

ReduceOp::ReduceOp(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<ReduceOpNode> node = make_object<ReduceOpNode>();
  // Accept BufferRegion/BufferLoad for src/dst
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessReadWrite);
  node->srcRegion_ = src_access.region;
  node->dstRegion_ = dst_access.region;
  node->SetAccessRegions({src_access, dst_access});
  node->src = node->srcRegion_->buffer;
  node->dst = node->dstRegion_->buffer;
  std::string reduce_type = args[2].as<StringImm>().value()->value;
  node->dim = args[3].as<IntImm>().value()->value;
  node->type = ReduceType(reduce_type);
  node->clear = args[4].as<Bool>().value();
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

static Fragment ComputeReducerLayout(const Fragment &src_layout, int dim) {
  PrimExpr src_rep_extent = src_layout->ReplicateExtent();
  PrimExpr indice_rep_extent = src_layout->InputShape()[dim];
  PrimExpr reducer_rep_extent = indice_rep_extent * src_rep_extent;

  auto fwd = InputPlaceholders(src_layout->InputDim() - 1);
  fwd.insert(fwd.begin() + dim,
             FloorMod(ReplicationPlaceholder(), indice_rep_extent));

  auto thd = src_layout->ForwardThread(
      fwd, FloorDiv(ReplicationPlaceholder(), indice_rep_extent));

  auto reducer_shape = src_layout->InputShape();
  reducer_shape.erase(reducer_shape.begin() + dim);
  if (reducer_shape.empty()) {
    reducer_shape.push_back(1);
  }

  auto reducer_layout =
      Fragment(reducer_shape, {}, thd, reducer_rep_extent, std::nullopt)
          ->CondenseReplicateVar()
          ->BindThreadRange(src_layout->ThreadRange());
  return reducer_layout;
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
 *   (> 32) a workspace is allocated via T.AddWorkspace and passed to the
 *   AllReduce call.
 * - The final body is wrapped in parallel loops over the destination spatial
 *   dimensions and partitioned by the lowering thread variable. If a temporary
 *   clear buffer is used, it is allocated for the body.
 *
 * @param T Lowering context providing buffer and layout maps, thread bounds,
 *          target information, thread variable, and workspace allocation
 * helper.
 * @param analyzer Analyzer used for iterator compression and arithmetic
 * normalization.
 * @return Stmt Lowered TIR statement implementing the reduction.
 */
Stmt ReduceOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  return ResolveReduceImpl(T.target).lower(*this, T, analyzer);
}

LayoutMap ReduceOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  if (level >= InferLevel::kStrict)
    return {};

  if (IsFragmentBuffer(src) && IsFragmentBuffer(dst) &&
      T.layout_map.count(src)) {
    auto src_layout = T.layout_map[src].as<Fragment>().value();
    auto reducer_layout = ComputeReducerLayout(src_layout, this->dim);

    if (!T.layout_map.count(dst)) {
      return {{dst, reducer_layout}};
    }

    auto orig_dst_layout = T.layout_map.Get(dst).value().as<Fragment>().value();
    ICHECK(reducer_layout->InputDim() == orig_dst_layout->InputDim());

    auto indices = InputPlaceholders(reducer_layout->InputDim());
    arith::Analyzer analyzer;
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
}

} // namespace tl
} // namespace tvm
