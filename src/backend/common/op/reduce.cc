/*!
 * \file tl/backend/common/op/reduce.cc
 * \brief Shared non-template tl.reduce lowering utilities.
 */

#include "reduce.h"

#include <tvm/ir/with_context.h>

namespace tvm {
namespace tl {
namespace backend {
namespace reduce {

using namespace tirx;

Range ResolveAllReduceThreadRange(const Fragment &red_layout,
                                  const Range &thread_bounds,
                                  const Target &target) {
  const int64_t *block_min = as_const_int(thread_bounds->min);
  const int64_t *block_extent = as_const_int(thread_bounds->extent);
  const int64_t *replicate = as_const_int(red_layout->ReplicateExtent());
  if (block_min == nullptr || block_extent == nullptr || replicate == nullptr) {
    LOG(FATAL) << "tl.reduce: cannot resolve the scalar AllReduce barrier: "
                  "the CTA thread bounds or reduce layout replicate extent "
                  "are not compile-time constants.";
  }
  ICHECK_GT(*block_extent, 0)
      << "tl.reduce: CTA thread extent must be positive";
  ICHECK_GT(*replicate, 0)
      << "tl.reduce: reduce layout replicate extent must be positive";

  arith::Analyzer analyzer;
  for (size_t i = 0; i < red_layout->InputShape().size(); ++i) {
    Var placeholder = InputPlaceholder(i);
    analyzer.Bind(placeholder,
                  Range::FromMinExtent(make_zero(placeholder.dtype()),
                                       red_layout->InputShape()[i]));
  }
  Var replicate_var = ReplicationPlaceholder();
  analyzer.Bind(replicate_var,
                Range::FromMinExtent(make_zero(replicate_var.dtype()),
                                     red_layout->ReplicateExtent()));

  // PartitionLoop feeds (threadIdx.x - ThreadRange.min) into the inverse
  // layout. Convert the forward map back to the corresponding absolute CTA
  // thread ID before computing its image.
  PrimExpr thread_expr = red_layout->GetForwardThread();
  if (red_layout->ThreadRange().defined()) {
    thread_expr =
        analyzer.Simplify(thread_expr + red_layout->ThreadRange()->min);
  }
  auto bound = analyzer.const_int_bound(thread_expr);
  if (bound->min_value == arith::ConstIntBoundNode::kNegInf ||
      bound->max_value == arith::ConstIntBoundNode::kPosInf) {
    LOG(FATAL) << "tl.reduce: cannot determine the scalar AllReduce "
                  "participating thread range.";
  }

  Var thread_var("allreduce_thread", thread_expr.dtype());
  analyzer.Bind(thread_var, thread_bounds);
  int64_t count;
  {
    With<arith::ConstraintContext> image_constraint(&analyzer,
                                                    thread_var == thread_expr);
    count = analyzer.z3_prover.CountSatisfyingValues(thread_var, *block_extent);
  }
  ICHECK_GT(count, 0)
      << "tl.reduce: cannot determine the participating threads for scalar "
         "AllReduce";

  const int64_t base = bound->min_value;
  const int64_t end = bound->max_value;
  const int64_t span = end - base + 1;
  ICHECK_EQ(count, span)
      << "tl.reduce: partial scalar AllReduce requires one contiguous thread "
         "range, but got "
      << count << " distinct threads spanning [" << base << ", " << end << "]";
  ICHECK_GE(base, *block_min)
      << "tl.reduce: scalar AllReduce participating thread range starts "
         "before the CTA thread bounds";
  ICHECK_LT(end, *block_min + *block_extent)
      << "tl.reduce: scalar AllReduce participating thread range ends after "
         "the CTA thread bounds";

  int64_t warp_size = 32;
  if (auto warp_size_attr = target->GetAttr<Integer>("thread_warp_size")) {
    warp_size = warp_size_attr.value()->value;
  }
  ICHECK_EQ((base - *block_min) % warp_size, 0)
      << "tl.reduce: partial scalar AllReduce requires a warp-aligned "
         "participating thread range, got base "
      << base << " in a block starting at " << *block_min;
  ICHECK_EQ(count % warp_size, 0)
      << "tl.reduce: partial scalar AllReduce requires a warp-aligned thread "
         "range, got "
      << count << " threads";

  return Range::FromMinExtent(make_const(thread_expr.dtype(), base),
                              make_const(thread_bounds->extent.dtype(), count));
}

} // namespace reduce
} // namespace backend
} // namespace tl
} // namespace tvm
