/*!
 * \file parallel_loop_layout_validator.h
 * \brief Validator for parallel loop layout annotations.
 */

#ifndef TVM_TL_TRANSFORM_PARALLEL_LOOP_LAYOUT_VALIDATOR_H_
#define TVM_TL_TRANSFORM_PARALLEL_LOOP_LAYOUT_VALIDATOR_H_

#include <tvm/tir/stmt_functor.h>

#include "../layout/layout.h"

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Count the number of consecutive nested parallel loops starting from
 *        the given For node.
 * \param op The outermost For node to start counting from.
 * \return The number of consecutive nested parallel loops.
 */
inline int CountNestedParallelLoops(const ForNode *op) {
  int count = 0;
  const ForNode *current = op;
  while (current != nullptr && current->kind == ForKind::kParallel) {
    count++;
    current = current->body.as<ForNode>();
  }
  return count;
}

/*!
 * \brief Validator that checks parallel loop layout annotations.
 *
 * Rationale: in TileLang's design, inner loops cannot control their outer
 * loops, while the outermost loop can manage its inner nested region. Hence
 * the layout annotation is placed on the outermost parallel loop so passes
 * can reason about and transform the whole nest from the outside.
 *
 * This validator checks:
 * 1. All parallel loops must have layout annotations (either directly or via
 *    an outer nested parallel loop).
 * 2. For nested parallel loops, only the outermost parallel loop should have
 *    the layout annotation.
 * 3. The layout's InputDim must equal the number of consecutive nested
 *    parallel loops.
 */
class ParallelLoopLayoutValidator : public StmtVisitor {
public:
  /*!
   * \brief Validate parallel loop layout annotations in the given statement.
   * \param stmt The statement to validate.
   */
  static void Validate(const Stmt &stmt) {
    ParallelLoopLayoutValidator validator;
    validator.VisitStmt(stmt);
  }

private:
  void VisitStmt_(const ForNode *op) final {
    // Only validate parallel loops
    if (op->kind != ForKind::kParallel) {
      StmtVisitor::VisitStmt_(op);
      return;
    }

    // Check if this parallel loop has a layout annotation
    bool has_layout = op->annotations.count(attr::kParallelLoopLayout) > 0;

    // Count the number of consecutive nested parallel loops
    int nested_count = CountNestedParallelLoops(op);

    if (has_layout) {
      // This is the outermost parallel loop with layout annotation
      auto loop_layout = Downcast<Fragment>(
          op->annotations.Get(attr::kParallelLoopLayout).value());

      // Validate that layout's InputDim matches the number of nested parallel
      // loops
      int layout_input_dim = static_cast<int>(loop_layout->InputDim());
      ICHECK(layout_input_dim == nested_count)
          << "Layout InputDim mismatch for parallel loop.\n"
          << "Expected: " << nested_count
          << " (number of consecutive nested parallel loops)\n"
          << "Got: " << layout_input_dim << " (layout InputDim)\n"
          << "Loop: " << tvm::ffi::GetRef<For>(op) << "\n"
          << "For nested parallel loops, the layout annotation should be on "
          << "the outermost loop, and its InputDim should equal the total "
          << "number of nested parallel loops.";

      // Validate that inner parallel loops do NOT have layout annotations
      ValidateInnerParallelLoopsNoLayout(op->body, nested_count - 1);

      // Skip visiting inner parallel loops as they are part of this nested
      // structure. Visit the body of the innermost parallel loop instead.
      const ForNode *innermost = op;
      for (int i = 1; i < nested_count; i++) {
        innermost = innermost->body.as<ForNode>();
      }
      StmtVisitor::VisitStmt(innermost->body);
    } else {
      // This parallel loop doesn't have a layout annotation
      // This is only valid if it's an inner loop of a nested parallel structure
      // But since we process from outermost to innermost, if we reach here
      // without a layout annotation, it's an error.
      LOG(FATAL)
          << "Parallel loop missing layout annotation.\n"
          << "Loop: " << tvm::ffi::GetRef<For>(op) << "\n"
          << "All parallel loops must have a layout annotation after "
          << "LayoutInference pass. For nested parallel loops, the annotation "
          << "should be on the outermost loop.";
    }
  }

  /*!
   * \brief Validate that inner parallel loops do not have layout annotations.
   * \param body The body to check (should be inner parallel loops).
   * \param remaining_count Number of remaining inner parallel loops to check.
   */
  void ValidateInnerParallelLoopsNoLayout(const Stmt &body,
                                          int remaining_count) {
    if (remaining_count <= 0) {
      return;
    }

    const ForNode *inner_for = body.as<ForNode>();
    ICHECK(inner_for != nullptr && inner_for->kind == ForKind::kParallel)
        << "Expected inner parallel loop but found: " << body;

    ICHECK(!inner_for->annotations.count(attr::kParallelLoopLayout))
        << "Inner parallel loop should NOT have layout annotation.\n"
        << "Loop: " << tvm::ffi::GetRef<For>(inner_for) << "\n"
        << "For nested parallel loops, only the outermost parallel loop "
        << "should have the layout annotation.";

    ValidateInnerParallelLoopsNoLayout(inner_for->body, remaining_count - 1);
  }
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_PARALLEL_LOOP_LAYOUT_VALIDATOR_H_
