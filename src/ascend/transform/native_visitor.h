#include "compat/metadata.h"
#ifndef TILELANG_ASCEND_TRANSFORM_NATIVE_VISITOR_H_
#define TILELANG_ASCEND_TRANSFORM_NATIVE_VISITOR_H_

#include "transform/common/loop_aware_mutator.h"

#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>

namespace tvm::tl::ascend {

using namespace prim;
using namespace tirx;

template <typename Base> class NativeMutatorBase : public Base {
public:
  using Base::Base;
  using Base::Mutate;
  Stmt VisitStmt(const Stmt &statement) {
    return this->Mutate(statement).ValueOrUnchanged(statement);
  }
  Expr VisitExpr(const Expr &expression) {
    return this->Mutate(expression).ValueOrUnchanged(expression);
  }
  PrimExpr VisitExpr(const PrimExpr &expression) {
    return this->Mutate(expression).ValueOrUnchanged(expression);
  }
  template <typename Value> Value operator()(const Value &value) {
    return this->Mutate(value).ValueOrUnchanged(value);
  }

protected:
#define TL_ASCEND_MUTATE_STMT(Node)                                            \
  virtual Stmt VisitStmt_(const Node *node) {                                  \
    return Base::Mutate_(node, ffi::InplaceMode::kDisallow)                    \
        .ValueOrUnchanged(ffi::GetRef<Stmt>(node));                            \
  }                                                                            \
  ffi::UnchangedOr<Stmt> Mutate_(const Node *node, ffi::InplaceMode)           \
      override {                                                               \
    return VisitStmt_(node);                                                   \
  }
  TL_ASCEND_MUTATE_STMT(BindNode)
  TL_ASCEND_MUTATE_STMT(RegionStmtNode)
  TL_ASCEND_MUTATE_STMT(ForNode)
  TL_ASCEND_MUTATE_STMT(WhileNode)
  TL_ASCEND_MUTATE_STMT(IfNode)
  TL_ASCEND_MUTATE_STMT(SeqStmtNode)
  TL_ASCEND_MUTATE_STMT(TensorStoreNode)
  TL_ASCEND_MUTATE_STMT(EvaluateNode)
  TL_ASCEND_MUTATE_STMT(AssertStmtNode)
  TL_ASCEND_MUTATE_STMT(BreakNode)
  TL_ASCEND_MUTATE_STMT(ContinueNode)
  TL_ASCEND_MUTATE_STMT(ReturnNode)
#undef TL_ASCEND_MUTATE_STMT

#define TL_ASCEND_MUTATE_EXPR(Node, Result)                                    \
  virtual Result VisitExpr_(const Node *node) {                                \
    return Base::Mutate_(node, ffi::InplaceMode::kDisallow)                    \
        .ValueOrUnchanged(ffi::GetRef<Result>(node));                          \
  }                                                                            \
  ffi::UnchangedOr<Result> Mutate_(const Node *node, ffi::InplaceMode)         \
      override {                                                               \
    return VisitExpr_(node);                                                   \
  }
  TL_ASCEND_MUTATE_EXPR(CallNode, Expr)
  TL_ASCEND_MUTATE_EXPR(VarNode, Expr)
  TL_ASCEND_MUTATE_EXPR(TensorLoadNode, PrimExpr)
#undef TL_ASCEND_MUTATE_EXPR
};

using NativeMutator = NativeMutatorBase<tvm::tl::StmtExprMutator>;
using AnalyzerMutator = NativeMutatorBase<::tvm::tl::LoopAwareMutator>;

class AnalysisVisitor : public tvm::tl::StmtExprVisitor {
public:
  void VisitStmt(const Stmt &statement) { Visit(statement); }
  void VisitExpr(const Expr &expression) { Visit(expression); }
  void operator()(const ffi::ObjectRef &value) { Visit(value); }

protected:
#define TL_ASCEND_VISIT_STMT(Node)                                             \
  virtual void VisitStmt_(const Node *node) {                                  \
    tvm::tl::StmtExprVisitor::Visit_(node);                                    \
  }                                                                            \
  ffi::Optional<ffi::VisitInterrupt> Visit_(const Node *node) override {       \
    VisitStmt_(node);                                                          \
    return std::nullopt;                                                       \
  }
  TL_ASCEND_VISIT_STMT(BindNode)
  TL_ASCEND_VISIT_STMT(RegionStmtNode)
  TL_ASCEND_VISIT_STMT(ForNode)
  TL_ASCEND_VISIT_STMT(WhileNode)
  TL_ASCEND_VISIT_STMT(IfNode)
  TL_ASCEND_VISIT_STMT(SeqStmtNode)
  TL_ASCEND_VISIT_STMT(TensorStoreNode)
  TL_ASCEND_VISIT_STMT(EvaluateNode)
  TL_ASCEND_VISIT_STMT(AssertStmtNode)
  TL_ASCEND_VISIT_STMT(BreakNode)
  TL_ASCEND_VISIT_STMT(ContinueNode)
  TL_ASCEND_VISIT_STMT(ReturnNode)
#undef TL_ASCEND_VISIT_STMT

#define TL_ASCEND_VISIT_EXPR(Node)                                             \
  virtual void VisitExpr_(const Node *node) {                                  \
    tvm::tl::StmtExprVisitor::Visit_(node);                                    \
  }                                                                            \
  ffi::Optional<ffi::VisitInterrupt> Visit_(const Node *node) override {       \
    VisitExpr_(node);                                                          \
    return std::nullopt;                                                       \
  }
  TL_ASCEND_VISIT_EXPR(CallNode)
  TL_ASCEND_VISIT_EXPR(TensorLoadNode)
  TL_ASCEND_VISIT_EXPR(VarNode)
#undef TL_ASCEND_VISIT_EXPR
};

template <typename Signature> class AnalysisFunctor;

template <typename Result, typename... Arguments>
class AnalysisFunctor<Result(const Stmt &, Arguments...)>
    : public tirx::StmtFunctor<Result(const Stmt &, Arguments...)> {
public:
  Result VisitStmt(const Stmt &statement, Arguments... arguments) {
    return this->Dispatch(statement, std::forward<Arguments>(arguments)...);
  }

protected:
#define TL_ASCEND_DISPATCH(Node)                                               \
  virtual Result VisitStmt_(const Node *node, Arguments... arguments) {        \
    return this->DispatchDefault_(node,                                        \
                                  std::forward<Arguments>(arguments)...);      \
  }                                                                            \
  Result Dispatch_(const Node *node, Arguments... arguments) override {        \
    return VisitStmt_(node, std::forward<Arguments>(arguments)...);            \
  }
  TL_ASCEND_DISPATCH(BindNode)
  TL_ASCEND_DISPATCH(RegionStmtNode)
  TL_ASCEND_DISPATCH(ForNode)
  TL_ASCEND_DISPATCH(WhileNode)
  TL_ASCEND_DISPATCH(IfNode)
  TL_ASCEND_DISPATCH(SeqStmtNode)
  TL_ASCEND_DISPATCH(TensorStoreNode)
  TL_ASCEND_DISPATCH(EvaluateNode)
  TL_ASCEND_DISPATCH(AssertStmtNode)
  TL_ASCEND_DISPATCH(BreakNode)
  TL_ASCEND_DISPATCH(ContinueNode)
  TL_ASCEND_DISPATCH(ReturnNode)
#undef TL_ASCEND_DISPATCH
};

} // namespace tvm::tl::ascend

#endif
