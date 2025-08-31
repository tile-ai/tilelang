/*!
 * \file tl/op/parallel.h
 * \brief Infer layout from ops and parallel for
 */

#ifndef TVM_TL_OP_PARALLEL_H_
#define TVM_TL_OP_PARALLEL_H_

#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>

#include "../layout/layout.h"
#include "../transform/layout_reducer.h"
#include "./operator.h"

/**
 * Exception indicating a layout conflict during layout inference or validation.
 * The stored message is returned by what().
 */

/**
 * Verify that `small_frag` is contained within `large_frag` under the provided
 * index mappings and using symbolic reasoning via `analyzer_`.
 *
 * @param small_frag Fragment describing the smaller layout fragment.
 * @param large_frag Fragment describing the larger layout fragment.
 * @param small_frag_indices Index expressions that map accesses into
 * `small_frag`.
 * @param large_frag_indices Index expressions that map accesses into
 * `large_frag`.
 * @param analyzer_ Analyzer used for symbolic simplification and proving
 * relations.
 * @return true if `small_frag` can be proven to be contained in `large_frag`
 * given the index mappings and analyzer; false otherwise.
 */

/**
 * Visitor that traverses a parallel loop nest to collect loop structure,
 * buffer access patterns, and to populate the associated ParallelOpNode.
 */

/**
 * Construct a ParallelOpNode from a root For loop.
 *
 * @param root The TIR For node that is the root of the parallel loop nest.
 */

/**
 * Lower this ParallelOpNode to a TIR statement.
 *
 * Performs lowering of the operator (including any necessary predicates,
 * reductions, and loop transformations) to produce an equivalent tir::Stmt.
 *
 * @param T Lowering options and context.
 * @param analyzer Optional analyzer for symbolic simplification during
 * lowering.
 * @return A tir::Stmt representing the lowered operator.
 */

/**
 * Infer layouts for buffers used by this parallel operator.
 *
 * This performs layout inference at the requested level and returns a mapping
 * from buffers to their inferred layout fragments.
 *
 * @param T Layout inference arguments and context.
 * @param level Granularity level for inference.
 * @return LayoutMap mapping buffers to inferred fragments.
 */

/**
 * Return an optional predicate expression associated with the given thread
 * variable.
 *
 * If the loop nest imposes a condition on `thread_var` (e.g., bounds checks or
 * tiling edge predicates), this returns the combined predicate; otherwise
 * returns an empty Optional.
 *
 * @param thread_var The thread variable for which to retrieve the predicate.
 * @return Optional containing the predicate expression if present.
 */

/**
 * Create and return a clone of this operator as a TileOperator (deep copy of
 * operator state necessary for further transformations).
 *
 * @return A TileOperator referencing a cloned ParallelOpNode.
 */

/**
 * Complete the layout fragment for `buffer` by filling in any missing
 * dimension or stride information derived from access patterns in the loop
 * nest.
 *
 * @param buffer The buffer whose fragment should be completed.
 * @return A Fragment representing the completed layout for `buffer`.
 */

/**
 * Determine whether `buffer` is accessed using only the loop-common indices
 * (i.e., indices that correspond to the loop variables of this parallel nest).
 *
 * @param buffer The buffer to inspect.
 * @return true if accesses use common loop indices; false otherwise.
 */

/**
 * Conjoin `expr` into the operator's predicate (logical AND). If no predicate
 * exists yet, `expr` becomes the predicate.
 *
 * @param expr Predicate expression to add.
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
  // Mapping from buffer to reducer info.
  Map<Var, ReducerInfo> reducer_info_map_;
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
