/*!
 * \file tl/op/parallel.h
 * \brief Infer layout from ops and parallel for
 */

#ifndef TVM_TL_OP_PARALLEL_H_
#define TVM_TL_OP_PARALLEL_H_

#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>

#include "../layout/layout.h"
#include "operator.h"

/**
 * Exception representing a layout conflict detected during layout inference.
 *
 * Stores an explanatory message retrievable via what().
 */

/**
 * Determine whether `small_frag` is guaranteed to be contained within
 * `large_frag` under the given index mappings and using the provided arithmetic
 * analyzer.
 *
 * @param small_frag The smaller fragment to test for containment.
 * @param large_frag The larger fragment that may contain `small_frag`.
 * @param small_frag_indices Index expressions mapping the small fragment into
 * buffer space.
 * @param large_frag_indices Index expressions mapping the large fragment into
 * buffer space.
 * @param analyzer_ Arithmetic analyzer used to simplify and prove index
 * relations.
 * @return true if containment can be proven; false otherwise.
 */

/**
 * Visitor that traverses a parallel loop nest to collect buffer access and
 * loop-structure information for a ParallelOpNode.
 *
 * The visitor records loop variables, buffer read/write accesses, and builds
 * predicates as it encounters BufferLoad/BufferStore and For nodes.
 */

/**
 * Represents a parallel for-loop operator in TileLang.
 *
 * Holds the root For loop, collects and exposes loop layout and access-index
 * information, and provides layout inference and lowering to TIR.
 *
 * Public methods expose the inferred loop layout, root loop, buffer index
 * mappings, and any per-thread predicate; Lower and InferLayout perform the
 * operator's lowering and layout inference respectively.
 */

/**
 * Create a ParallelOpNode from a root For loop.
 *
 * @param root The root For node representing the parallel loop nest.
 */

/**
 * Lower this parallel operator into a TIR statement suitable for codegen.
 *
 * @param T Lowering arguments and context.
 * @param analyzer Arithmetic analyzer for expression simplification during
 * lowering.
 * @return A TIR statement representing the lowered parallel loop.
 */

/**
 * Infer the layout mapping for this parallel operator at the specified level.
 *
 * @param T Arguments and context for layout inference.
 * @param level Inference granularity level.
 * @return A LayoutMap describing inferred buffer/layout relationships for the
 * operator.
 */

/**
 * Copy-construct a ParallelOpNode, preserving inferred layout and predicate.
 */

/**
 * Get the inferred loop layout fragment.
 *
 * @return The Fragment representing the loop's inferred layout (may be lazily
 * computed).
 */

/**
 * Get the root For loop of this operator.
 *
 * @return The root For AST node.
 */

/**
 * Get the mapping from each buffer to the array of index expressions used to
 * access it within the loop nest.
 *
 * @return A Map from Buffer to Array<PrimExpr> of access indices.
 */

/**
 * Retrieve the predicate expression associated with a given thread variable, if
 * any.
 *
 * @param thread_var The thread variable whose predicate is requested.
 * @return An Optional<PrimExpr> containing the predicate when present.
 */

/**
 * Create a deep copy of this operator as a TileOperator handle.
 *
 * @return A TileOperator that references a copy of this node.
 */

/**
 * Visitor helper: complete the fragment layout for a buffer (internal).
 *
 * (Private helper — not part of the public API.)
 */

/**
 * Helper to check whether a buffer's access indices are the common loop indices
 * (internal).
 *
 * (Private helper — not part of the public API.)
 */

/**
 * Add `expr` to the current predicate by logical AND; sets predicate if none
 * exists.
 *
 * (Private helper — not part of the public API.)
 */

/**
 * Thin handle type exposing ParallelOpNode as a TileOperator.
 *
 * Construct from a root For loop to create and own a ParallelOpNode instance.
 */

/**
 * Construct a ParallelOp handle from a root For loop.
 *
 * @param root The root For node representing the parallel loop nest.
 */
namespace tvm {
namespace tl {

using namespace tir;

class LayoutConflictException : public std::exception {
public:
  const char *what() const noexcept override { return msg_.c_str(); }
  LayoutConflictException(const std::string &msg) : msg_(msg) {}

private:
  std::string msg_;
};

bool ProveFragmentContains(Fragment small_frag, Fragment large_frag,
                           Array<PrimExpr> small_frag_indices,
                           Array<PrimExpr> large_frag_indices,
                           arith::Analyzer &analyzer_);

class ParallelOpNode;

class ParallelLoopNestVisitor : public StmtExprVisitor {
private:
  ParallelLoopNestVisitor(ParallelOpNode *op) : p(op){};
  void VisitStmt_(const ForNode *op) override;
  void VisitStmt_(const BufferStoreNode *op) override;
  void VisitExpr_(const BufferLoadNode *op) override;

  ParallelOpNode *p;

  friend class ParallelOpNode;
};

// ParallelOpNode represents a parallel for loop operator in TileLang.
// It is responsible for inferring layouts, holding loop structure, and managing
// predicates.
class ParallelOpNode : public TileOperatorNode {
public:
  // The inferred layout for the loop, mutable to allow lazy inference.
  mutable Fragment loop_layout_;
  // The predicate expression for the loop, if any, mutable for lazy
  // construction.
  mutable Optional<PrimExpr> predicate_;

  // Type key for TVM object system.
  static constexpr const char *_type_key = "tl.ParallelOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(ParallelOpNode, TileOperatorNode);

  // Construct from a root For loop.
  ParallelOpNode(For root);

  // Lower the operator to a TIR statement.
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;

  // Infer the layout for this parallel operator.
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  // Copy constructor for ParallelOpNode.
  ParallelOpNode(const ParallelOpNode &other) : ParallelOpNode(other.root_) {
    loop_layout_ = other.loop_layout_;
    predicate_ = other.predicate_;
  }

  // Get the inferred loop layout.
  Fragment GetLoopLayout() const { return loop_layout_; }
  // Get the root For loop.
  For GetRoot() const { return root_; }
  // Get the mapping from buffer to access indices.
  Map<Buffer, Array<PrimExpr>> GetIndiceMap() const { return indice_map_; }
  // Get the predicate for a given thread variable.
  Optional<PrimExpr> GetPredicate(Var thread_var) const;

  // Clone this operator.
  TileOperator Clone() const;

private:
  // Complete the fragment layout for a given buffer.
  Fragment CompleteBufferFragment(const Buffer &buffer) const;
  // Check if the buffer is accessed with common indices (i.e., loop variables).
  bool IsCommonAccessIndice(const Buffer &buffer) const;
  // Add a predicate to the current predicate expression.
  void AddPredicate(PrimExpr expr) const {
    predicate_ = predicate_.defined() ? And(expr, predicate_.value()) : expr;
  }
  // Allow ParallelLoopNestVisitor to access private members.
  friend class ParallelLoopNestVisitor;

  // The root For loop node.
  For root_;
  // Visitor for collecting loop nest information.
  ParallelLoopNestVisitor V;
  // Mapping from buffer to their access indices in the loop.
  Map<Buffer, Array<PrimExpr>> indice_map_;
  // Set of buffers that are written to in the loop.
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_is_write_;
  // The loop variables for the parallel loop nest.
  Array<IterVar> loop_vars_;
  // Analyzer for simplifying and analyzing expressions, mutable for lazy use.
  mutable arith::Analyzer analyzer_;
};

class ParallelOp : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(ParallelOp, TileOperator, ParallelOpNode);

  ParallelOp(For root) {
    auto op = make_object<ParallelOpNode>(root);
    data_ = std::move(op);
  }
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_PARALLEL_H_
