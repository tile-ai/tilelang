/*!
 * \file tl/op/scan.h
 * \brief Inclusive scan operators for tensor computations.
 */

#ifndef TVM_TL_OP_SCAN_H_
#define TVM_TL_OP_SCAN_H_

#include "operator.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

/// Node class for cumulative sum operations
class CumSumOpNode : public TileOperatorNode {
public:
  tirx::Buffer src, dst; ///< Source and destination buffers
  BufferRegion srcRegion_, dstRegion_;
  int dim;      ///< Dimension along which to compute cumulative sum
  bool reverse; ///< Whether to compute in reverse order
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.CumSumOp", CumSumOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<CumSumOpNode>()
        .def_ro("src", &CumSumOpNode::src)
        .def_ro("dst", &CumSumOpNode::dst)
        .def_ro("srcRegion", &CumSumOpNode::srcRegion_)
        .def_ro("dstRegion", &CumSumOpNode::dstRegion_)
        .def_ro("dim", &CumSumOpNode::dim)
        .def_ro("reverse", &CumSumOpNode::reverse);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const;
};

using CumSumTargetPredicate = bool (*)(Target target);

struct CumSumImpl {
  const char *name;
  CumSumTargetPredicate match_target;

  Stmt (*lower)(const CumSumOpNode &op, const LowerArgs &T,
                arith::Analyzer *analyzer);
};

void RegisterCumSumImpl(CumSumImpl impl);

/// Wrapper class for cumulative sum operations
class CumSumOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(CumSumOp, TileOperator,
                                             CumSumOpNode);
  TVM_DLL
  CumSumOp(Array<PrimExpr> args,
           Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

/// Node class for cumulative maximum operations
class CumMaxOpNode : public TileOperatorNode {
public:
  tirx::Buffer src, dst; ///< Source and destination buffers
  BufferRegion srcRegion_, dstRegion_;
  int dim;      ///< Dimension along which to compute cumulative maximum
  bool reverse; ///< Whether to compute in reverse order
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.CumMaxOp", CumMaxOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<CumMaxOpNode>()
        .def_ro("src", &CumMaxOpNode::src)
        .def_ro("dst", &CumMaxOpNode::dst)
        .def_ro("srcRegion", &CumMaxOpNode::srcRegion_)
        .def_ro("dstRegion", &CumMaxOpNode::dstRegion_)
        .def_ro("dim", &CumMaxOpNode::dim)
        .def_ro("reverse", &CumMaxOpNode::reverse);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const;
};

using CumMaxTargetPredicate = bool (*)(Target target);

struct CumMaxImpl {
  const char *name;
  CumMaxTargetPredicate match_target;

  Stmt (*lower)(const CumMaxOpNode &op, const LowerArgs &T,
                arith::Analyzer *analyzer);
};

void RegisterCumMaxImpl(CumMaxImpl impl);

/// Wrapper class for cumulative maximum operations
class CumMaxOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(CumMaxOp, TileOperator,
                                             CumMaxOpNode);
  TVM_DLL
  CumMaxOp(Array<PrimExpr> args,
           Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_SCAN_H_
