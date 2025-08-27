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

class ParallelOpNode : public TileOperatorNode {
public:
  static constexpr const char *_type_key = "tl.ParallelOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(ParallelOpNode, TileOperatorNode);

  ParallelOpNode(For root);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  ParallelOpNode(const ParallelOpNode &other) : ParallelOpNode(other.root_) {
    loop_layout_ = other.loop_layout_;
    predicate_ = other.predicate_;
  }

  Fragment GetLoopLayout() const { return loop_layout_; }
  For GetRoot() const { return root_; }
  Map<Buffer, Array<PrimExpr>> GetIndiceMap() const { return indice_map_; }
  Optional<PrimExpr> GetPredicate(Var thread_var) const;

  TileOperator Clone() const;

private:
  Fragment CompleteBufferFragment(const Buffer &buffer) const;
  bool IsCommonAccessIndice(const Buffer &buffer) const;
  void AddPredicate(PrimExpr expr) const {
    predicate_ = predicate_.defined() ? And(expr, predicate_.value()) : expr;
  }

  For root_;

  ParallelLoopNestVisitor V;

  Map<Buffer, Array<PrimExpr>> indice_map_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_is_write_;
  Array<IterVar> loop_vars_;

  mutable Fragment loop_layout_;
  mutable arith::Analyzer analyzer_;
  mutable Optional<PrimExpr> predicate_;

  friend class ParallelLoopNestVisitor;
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
