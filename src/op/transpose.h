/*!
 * \file tl/op/transpose.h
 * \brief Transpose operation that swaps the final two buffer axes.
 */

#ifndef TVM_TL_OP_TRANSPOSE_H_
#define TVM_TL_OP_TRANSPOSE_H_

#include "operator.h"
#include "parallel.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

/// Node class for transpose operations over the final two axes.
class TransposeNode : public TileOperatorNode {
public:
  explicit TransposeNode(ffi::UnsafeInit tag) : src(tag), dst(tag) {}

  TensorVar src, dst;
  Array<Range> src_range, dst_range;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Transpose", TransposeNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<TransposeNode>()
        .def_ro("src", &TransposeNode::src)
        .def_ro("dst", &TransposeNode::dst)
        .def_ro("src_range", &TransposeNode::src_range)
        .def_ro("dst_range", &TransposeNode::dst_range);
  }

  Stmt Lower(const LowerArgs &lower_args,
             sym::Analyzer analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &layout_args,
                        InferLevel level) const override;
  TileOperator Clone() const override;

  /// Build a SIMT-style nested parallel loop implementing the transpose.
  For MakeSIMTLoop(sym::Analyzer analyzer) const;

private:
  /// Create iterator variables for dimensions with extent > 1.
  Array<LayoutAxis> MakeIterVars() const;

  /// Generate source (src_dst=0) or destination (src_dst=1) index expressions.
  /// For the destination side, the final two source axes are exchanged.
  Array<PrimExpr> MakeIndices(const Array<LayoutAxis> &ivs, int src_dst) const;

  /// Build a boundary predicate for generated buffer indices.
  Optional<PrimExpr> MakePredicate(sym::Analyzer analyzer,
                                   const Array<PrimExpr> &indices,
                                   const Array<PrimExpr> &extents) const;
};

using TransposeTargetPredicate = bool (*)(Target target);

struct TransposeImpl {
  const char *name;
  TransposeTargetPredicate match_target;

  Stmt (*lower)(const TransposeNode &op, const LowerArgs &lower_args,
                sym::Analyzer analyzer);
};

void RegisterTransposeImpl(TransposeImpl impl);

/// Wrapper class for transpose operations
class Transpose : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Transpose, TileOperator,
                                             TransposeNode);
  TILELANG_DLL
  Transpose(Array<Expr> args,
            BlockAnnotations annotations = BlockAnnotations());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_TRANSPOSE_H_
