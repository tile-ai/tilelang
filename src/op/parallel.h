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
 * Conjoin `expr` into the operator's predicate (logical AND). If no predicate
 * exists yet, `expr` becomes the predicate.
 *
 * @param expr Predicate expression to add.
 */
namespace tvm {
namespace tl {

using namespace tir;

bool ProveFragmentContains(Fragment small_frag, Fragment large_frag,
                           Array<PrimExpr> small_frag_indices,
                           Array<PrimExpr> large_frag_indices,
                           arith::Analyzer &analyzer_);

class ParallelOpNode;

class ParallelLoopNestVisitor : public StmtExprVisitor {
private:
  ParallelLoopNestVisitor(ParallelOpNode *op) : p(op) {};
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
  // The root For loop node.
  For root_;
  // The inferred layout for the loop, mutable to allow lazy inference.
  mutable Fragment loop_layout_;
  // The predicate expression for the loop, if any, mutable for lazy
  // construction.
  mutable Optional<PrimExpr> predicate_;

  // Type key for TVM object system.
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ParallelOp", ParallelOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ParallelOpNode>()
        .def_ro("root", &ParallelOpNode::root_)
        .def_ro("loop_layout", &ParallelOpNode::loop_layout_)
        .def_ro("predicate", &ParallelOpNode::predicate_);
  }

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
  TileOperator Clone() const override;

private:
  // Complete the fragment layout for a given buffer.
  Fragment CompleteBufferFragment(const Buffer &buffer) const;
  // Check if the buffer is accessed with common indices (i.e., loop variables).
  bool IsCommonAccessIndice(const Buffer &buffer) const;
  // Add a predicate to the current predicate expression.
  void AddPredicate(const PrimExpr &expr) const {
    predicate_ = predicate_.defined() ? And(expr, predicate_.value()) : expr;
  }
  // Expand let bindings to find fragment buffer accesses and add them to
  // indice_map_. This handles cases like: a = block_mask_f[i]; T.copy(A[a, 0],
  // ...)
  void ExpandLetBindings(const Map<Var, PrimExpr> &let_var_to_expr);

  // Allow ParallelLoopNestVisitor to access private members.
  friend class ParallelLoopNestVisitor;

  // Visitor for collecting loop nest information.
  ParallelLoopNestVisitor V;
  // Mapping from buffer to their access indices in the loop.
  Map<Buffer, Array<PrimExpr>> indice_map_;
  // Set of buffers that are written to in the loop.
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_is_write_;
  // The loop variables for the parallel loop nest.
  Array<IterVar> loop_vars_;
  // The inner_vars_
  Map<Var, IterVar> inner_vars_;
  // Analyzer for simplifying and analyzing expressions, mutable for lazy use.
  mutable arith::Analyzer analyzer_;
  // Mapping from buffer to reducer info.
  Map<Var, ReducerInfo> reducer_info_map_;
};

class ParallelOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ParallelOp, TileOperator,
                                             ParallelOpNode);

  ParallelOp(const For &root) {
    auto op = tvm::ffi::make_object<ParallelOpNode>(root);
    data_ = std::move(op);
  }
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_PARALLEL_H_
