/*!
 * \file tl/op/transpose.h
 * \brief Transpose operation for 2D shared memory buffers.
 */

#ifndef TVM_TL_OP_TRANSPOSE_H_
#define TVM_TL_OP_TRANSPOSE_H_

#include "operator.h"

namespace tvm {
namespace tl {

using namespace tir;

/// Node class for transpose operations: dst[j, i] = src[i, j]
class TransposeNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_range, dst_range;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Transpose", TransposeNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TransposeNode>()
        .def_ro("src", &TransposeNode::src)
        .def_ro("dst", &TransposeNode::dst)
        .def_ro("src_range", &TransposeNode::src_range)
        .def_ro("dst_range", &TransposeNode::dst_range);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  TileOperator Clone() const override;

private:
  /// Create iterator variables for dimensions with extent > 1.
  Array<IterVar> MakeIterVars() const;

  /// Generate source (src_dst=0) or destination (src_dst=1) index expressions.
  /// For the destination side, non-trivial dimension indices are reversed to
  /// implement the transpose.
  Array<PrimExpr> MakeIndices(const Array<IterVar> &ivs, int src_dst) const;

  /// Build boundary predicate with transposed index mapping for dst.
  PrimExpr MakePredicate(arith::Analyzer *analyzer, const Array<IterVar> &ivs,
                         Array<PrimExpr> extents, int src_dst) const;

  /// Build a SIMT-style nested parallel loop implementing the transpose.
  For MakeSIMTLoop(arith::Analyzer *analyzer) const;
};

/// Wrapper class for transpose operations
class Transpose : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Transpose, TileOperator,
                                             TransposeNode);
  TVM_DLL
  Transpose(Array<PrimExpr> args,
            Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_TRANSPOSE_H_
