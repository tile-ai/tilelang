/*!
 * \file tl/op/elem.h
 * \brief Define elment-wise operators.
 *
 */

#ifndef TVM_TL_OP_ELEM_H_
#define TVM_TL_OP_ELEM_H_

#include "op.h"
#include "parallel.h"

namespace tvm {
namespace tl {

using namespace tir;

class Fill : public Operator {
public:
  Fill(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  static const Op &Get();

  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<Fill>(*this);
  }

private:
  For MakeSIMTLoop(arith::Analyzer *analyzer) const;
  tir::Buffer dst;
  PrimExpr value;
  Array<Range> region;
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_ELEM_H_