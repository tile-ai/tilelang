/*!
 * \file tl/op/elem.h
 * \brief Define elment-wise operators.
 *
 */

#ifndef TVM_TL_OP_ELEM_H_
#define TVM_TL_OP_ELEM_H_

#include "operator.h"
#include "parallel.h"

namespace tvm {
namespace tl {

using namespace tir;

class FillNode : public TileOperatorNode {
public:
  tir::Buffer dst;
  PrimExpr value;
  Array<Range> region;
  static constexpr const char *_type_key = "tl.Fill";
  TVM_DECLARE_FINAL_OBJECT_INFO(FillNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const;
  static const Op &Get();

  TileOperator Clone() const;

private:
  For MakeSIMTLoop(arith::Analyzer *analyzer) const;
};

class Fill : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(Fill, TileOperator, FillNode);
  TVM_DLL Fill(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_ELEM_H_