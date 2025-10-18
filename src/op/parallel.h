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
  static constexpr const char *_type_key = "tl.ParallelOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(ParallelOpNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ParallelOpNode>()
        .def_ro("root", &ParallelOpNode::root_)
        .def_ro("loop_layout", &ParallelOpNode::loop_layout_)
        .def_ro("predicate", &ParallelOpNode::predicate_);
  }

  bool SEqualReduce(const ParallelOpNode *other, SEqualReducer equal) const {
    return equal(root_, other->root_) &&
           equal(loop_layout_, other->loop_layout_) &&
           equal(predicate_, other->predicate_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(root_);
    hash_reduce(loop_layout_);
    hash_reduce(predicate_);
  }
  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;

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
  TVM_DEFINE_OBJECT_REF_METHODS(ParallelOp, TileOperator, ParallelOpNode);

  ParallelOp(const For &root) {
    auto op = make_object<ParallelOpNode>(root);
    data_ = std::move(op);
  }
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_PARALLEL_H_
