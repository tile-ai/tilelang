/*!
 * \file tl/op/atomic_add.h
 * \brief Atomic addition operations for concurrent memory updates
 */

#ifndef TVM_TL_OP_ATOMIC_ADD_H_
#define TVM_TL_OP_ATOMIC_ADD_H_

#include "atomic_reduce.h"

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Node class for atomic addition operations.
 *
 * Inherits from AtomicOpBaseNode and adds TMA support and vectorization.
 */
class AtomicAddNode : public AtomicOpBaseNode {
public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.AtomicAdd", AtomicAddNode,
                                    TileOperatorNode);

  /// Override Lower to add TMA support
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const;

  /// Override InferLayout to add TMA layout inference
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const;

  static const Op &Get();
  const Op &GetElemOp() const override;
  TileOperator Clone() const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AtomicAddNode>()
        .def_ro("src", &AtomicAddNode::src)
        .def_ro("src_value", &AtomicAddNode::src_value)
        .def_ro("dst", &AtomicAddNode::dst)
        .def_ro("src_range", &AtomicAddNode::src_range)
        .def_ro("dst_range", &AtomicAddNode::dst_range)
        .def_ro("annotations", &AtomicAddNode::annotations);
  }
};

using AtomicAddTargetPredicate = bool (*)(Target target);

struct AtomicAddImpl {
  const char *name;
  AtomicAddTargetPredicate match_target;

  LayoutMap (*infer_layout)(const AtomicAddNode &op, const LayoutInferArgs &T,
                            InferLevel level);

  Stmt (*lower)(const AtomicAddNode &op, const LowerArgs &T,
                arith::Analyzer *analyzer);
};

void RegisterAtomicAddImpl(AtomicAddImpl impl);

/// Wrapper class for atomic addition operations
class AtomicAdd : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(AtomicAdd, TileOperator,
                                             AtomicAddNode);
  TVM_DLL
  AtomicAdd(Array<PrimExpr> args,
            Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_ATOMIC_ADD_H_
