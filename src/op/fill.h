/*!
 * \file tl/op/fill.h
 * \brief Fill operations for tensor initialization
 */

#ifndef TVM_TL_OP_FILL_H_
#define TVM_TL_OP_FILL_H_

#include "operator.h"
#include "parallel.h"

namespace tvm {
namespace tl {

using namespace tir;

/// Node class for fill operations
class FillNode : public TileOperatorNode {
public:
  tir::Buffer dst;     ///< Destination buffer to fill
  PrimExpr value;      ///< Value to fill with
  Array<Range> region; ///< Region to fill within the buffer
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Fill", FillNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const;
  static const Op &Get();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FillNode>()
        .def_ro("dst", &FillNode::dst)
        .def_ro("value", &FillNode::value)
        .def_ro("region", &FillNode::region);
  }

  TileOperator Clone() const;

private:
  /// Create SIMT-style parallel loop for filling
  For MakeSIMTLoop(arith::Analyzer *analyzer) const;
};

/// Wrapper class for fill operations
class Fill : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Fill, TileOperator, FillNode);
  TVM_DLL Fill(Array<PrimExpr> args);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_FILL_H_
