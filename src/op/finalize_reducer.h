// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file src/op/finalize_reducer.h
 * \brief Define finalize_reducer operator.
 */

#ifndef TVM_TL_OP_FINALIZE_REDUCER_H_
#define TVM_TL_OP_FINALIZE_REDUCER_H_

#include "../transform/layout_reducer.h"
#include "./operator.h"

/**
 * Get the Op singleton for the public FinalizeReducerOp handle.
 *
 * @return A reference to the Op describing FinalizeReducer.
 */
namespace tvm {
namespace tl {

using namespace tir;

class FinalizeReducerOpNode : public TileOperatorNode {
public:
  tir::Buffer reducer;
  ReducerOpType op;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.FinalizeReducerOp",
                                    FinalizeReducerOpNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FinalizeReducerOpNode>()
        .def_ro("reducer", &FinalizeReducerOpNode::reducer)
        .def_ro("op", &FinalizeReducerOpNode::op);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const;
};

class FinalizeReducerOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(FinalizeReducerOp, TileOperator,
                                             FinalizeReducerOpNode);
  TVM_DLL FinalizeReducerOp(Array<PrimExpr> args);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_FINALIZE_REDUCER_H_
