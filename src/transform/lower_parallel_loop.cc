#include "loop_partition.h"
#include "loop_vectorize.h"

namespace tvm::tl {
using namespace ffi;

Stmt LowerParallelLoop(For loop, const Fragment &loop_layout,
                       PrimExpr thread_index, sym::Analyzer analyzer,
                       const LayoutMap &layout_map,
                       Optional<PrimExpr> predicate, bool parallel_loop,
                       bool require_padding_guard, bool enable_vectorize) {
  // Save analyzer state to prevent conflicted bindings during vectorization
  auto saved_analyzer = analyzer->Clone();

  For result_loop = loop;
  // Strip parallel-loop layout/predicate annotations on the original loop.
  // After partitioning/vectorization, keeping them can confuse later passes.
  // Also, annotations may contain complex expressions; mutators do not visit
  // inside annotation payloads, so explicit removal here prevents stale state
  // from leaking into subsequent transforms.
  // Note: Map::erase(key) is a no-op if key doesn't exist.
  result_loop.CopyOnWrite()->annotations.erase(attr::kParallelLoopLayout);
  result_loop.CopyOnWrite()->annotations.erase(attr::kParallelLoopPredicate);
  result_loop.CopyOnWrite()->annotations.erase(
      attr::kParallelLoopRequiresPaddingGuard);

  // Step 1: Partition the loop based on the layout (if this is a parallel loop)
  if (parallel_loop) {
    result_loop = PartitionLoop(result_loop, thread_index, analyzer,
                                loop_layout, require_padding_guard);
  }

  // Step 2: Vectorize the loop; the planner picks the size per loop
  // (1 = scalar) from its access analysis.
  if (enable_vectorize) {
    result_loop = VectorizeLoop(result_loop, saved_analyzer, layout_map);
  }

  result_loop = PragmaUnrollLoop(result_loop);

  // Step 3: Wrap with predicate if provided and this is a parallel loop
  if (predicate.has_value() && parallel_loop) {
    return If(predicate.value(), SeqStmt(result_loop), std::nullopt, loop->loc);
  }

  return result_loop;
}

} // namespace tvm::tl
