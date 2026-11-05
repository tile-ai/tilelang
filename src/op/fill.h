/*!
 * \file tl/op/fill.h
 * \brief Fill operations for tensor initialization
 */

#ifndef TVM_TL_OP_FILL_H_
#define TVM_TL_OP_FILL_H_

#include "operator.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;

/// Node class for fill operations
class FillNode : public TileOperatorNode {
public:
  tirx::TensorVar dst;      ///< Destination buffer to fill
  PrimExpr value;           ///< Value to fill with
  ffi::Array<Range> region; ///< Region to fill within the buffer
  FillNode(tirx::TensorVar dst, PrimExpr value, ffi::Array<Range> region)
      : dst(std::move(dst)), value(std::move(value)),
        region(std::move(region)) {}

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Fill", FillNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &lower_args,
             sym::AnalyzerObj *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &layout_args,
                        InferLevel level) const override;
  static const Op &Get();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FillNode>()
        .def_ro("dst", &FillNode::dst)
        .def_ro("value", &FillNode::value)
        .def_ro("region", &FillNode::region);
  }

  TileOperator Clone() const;

  /// Create SIMT-style parallel loop for filling
  For MakeSIMTLoop(sym::AnalyzerObj *analyzer) const;
};

using FillTargetPredicate = bool (*)(Target target);

struct FillImpl {
  const char *name;
  FillTargetPredicate match_target;

  Stmt (*lower)(const FillNode &op, const LowerArgs &lower_args,
                sym::AnalyzerObj *analyzer);
};

void RegisterFillImpl(FillImpl impl);

/// Wrapper class for fill operations
class Fill : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Fill, TileOperator, FillNode);
  TVM_TIRX_DLL Fill(ffi::Array<Expr> args,
                    ffi::Map<ffi::String, ffi::Any> annotations =
                        ffi::Map<ffi::String, ffi::Any>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_FILL_H_
