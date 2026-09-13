/*!
 * \file tl/ascend/op/simt_vf.h
 * \brief SimtVF control TileOperator.
 */

#ifndef TVM_TL_ASCEND_OP_SIMT_VF_H_
#define TVM_TL_ASCEND_OP_SIMT_VF_H_

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>

#include "op/operator.h"

namespace tvm {
namespace tl {

using namespace tirx;

class SimtVFOpNode : public TileOperatorNode {
public:
  SBlock root_;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SimtVFOp", SimtVFOpNode,
                                    TileOperatorNode);

  static void RegisterReflection();

  explicit SimtVFOpNode(SBlock root) : root_(std::move(root)) {}

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;

  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  TileOperator Clone() const override;
};

class SimtVFOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SimtVFOp, TileOperator,
                                             SimtVFOpNode);

  explicit SimtVFOp(const SBlock &root) {
    auto op = tvm::ffi::make_object<SimtVFOpNode>(root);
    data_ = std::move(op);
  }
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_OP_SIMT_VF_H_
