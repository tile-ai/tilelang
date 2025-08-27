/*!
 * \file tl/op/reduce.h
 * \brief Define reduce operator.
 *
 */

#ifndef TVM_TL_OP_REDUCE_H_
#define TVM_TL_OP_REDUCE_H_

#include "operator.h"

namespace tvm {
namespace tl {

using namespace tir;

enum class ReduceType {
  kSum,
  kAbsSum,
  kMax,
  kMin,
  kAbsMax,
};

class ReduceOpNode : public TileOperatorNode {
public:
  tir::Buffer src, dst;
  int dim;
  ReduceType type;
  bool clear;

  static constexpr const char *_type_key = "tl.ReduceOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReduceOpNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const;

private:
  PrimExpr MakeInitValue() const;
  PrimExpr MakeReduce(const PrimExpr &a, const PrimExpr &b) const;
  std::string MakeCodegenReducer() const;
};

class ReduceOp : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(ReduceOp, TileOperator, ReduceOpNode);
  TVM_DLL ReduceOp(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

class CumSumOpNode : public TileOperatorNode {
public:
  tir::Buffer src, dst;
  int dim;
  bool reverse;
  static constexpr const char *_type_key = "tl.CumSumOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(CumSumOpNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const;
};

class CumSumOp : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(CumSumOp, TileOperator, CumSumOpNode);
  TVM_DLL CumSumOp(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_REDUCE_H_