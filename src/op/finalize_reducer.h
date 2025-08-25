// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file src/op/finalize_reducer.h
 * \brief Define finalize_reducer operator.
 */

#ifndef TVM_TL_OP_FINALIZE_REDUCER_H_
#define TVM_TL_OP_FINALIZE_REDUCER_H_

#include "../transform/layout_reducer.h"
#include "op.h"

namespace tvm {
namespace tl {

using namespace tir;

class FinalizeReducer : public Operator {
public:
  FinalizeReducer(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  static const Op &Get();
  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<FinalizeReducer>(*this);
  }

  FinalizeReducer(const FinalizeReducer &) = default;

private:
  tir::Buffer reducer_;
  ReducerOpType op_;
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_REDUCE_H_