/*!
 * \file tl/op/simd_vf.h
 * \brief SimdVF control TileOperator.
 */

#ifndef TVM_TL_OP_SIMD_VF_H_
#define TVM_TL_OP_SIMD_VF_H_

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>

#include "./operator.h"

namespace tvm {
namespace tl {

using namespace tirx;

class SimdVFOpNode : public TileOperatorNode {
public:
  SBlock root_;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SimdVFOp", SimdVFOpNode,
                                    TileOperatorNode);

  static void RegisterReflection();

  explicit SimdVFOpNode(SBlock root) : root_(std::move(root)) {}

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;

  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  TileOperator Clone() const override;
};

class SimdVFOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SimdVFOp, TileOperator,
                                             SimdVFOpNode);

  explicit SimdVFOp(const SBlock &root) {
    auto op = tvm::ffi::make_object<SimdVFOpNode>(root);
    data_ = std::move(op);
  }
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_SIMD_VF_H_
