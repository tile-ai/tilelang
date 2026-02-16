/*!
 * \file tl/op/atomic_reduce.h
 * \brief Atomic operations base class and reduction operations (max/min)
 */

#ifndef TVM_TL_OP_ATOMIC_REDUCE_H_
#define TVM_TL_OP_ATOMIC_REDUCE_H_

#include "operator.h"
#include "parallel.h"

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Base node class for atomic operations (add/max/min).
 *
 * This base class provides common functionality for all atomic
 * operations including buffer management, loop generation, and layout
 * inference.
 */
class AtomicOpBaseNode : public TileOperatorNode {
public:
  PrimExpr src_value; ///< Source values, for cases src is not a buffer
  Buffer src, dst;    ///< Source and destination buffers
  Array<Range> src_range,
      dst_range; ///< Access ranges for source and destination
  Map<String, ObjectRef> annotations; ///< Annotations for the atomic operation
  // Supported annotation keys:
  //   - "coalesced_width": IntImm, width for memory coalescing optimization
  //   - "memory_order": IntImm, memory order for atomic operations

  mutable ParallelOp par_op_; ///< Associated parallel operation

  /// Default Lower implementation for non-TMA atomic ops
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const;

  /// Default InferLayout implementation
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const;

  /// Get memory order from annotations (default: relaxed = 0)
  int GetMemoryOrder() const {
    if (auto val = annotations.Get("memory_order")) {
      if (auto int_val = val->as<IntImmNode>()) {
        return int_val->value;
      }
    }
    return 0;
  }

  /// Get the element-wise operation Op (pure virtual, implemented by derived)
  virtual const Op &GetElemOp() const = 0;

protected:
  /// Create SIMT-style parallel loop structure
  For MakeSIMTLoop(arith::Analyzer *analyzer) const;

  /// Generate iteration variables for loop nest
  Array<IterVar> MakeIterVars() const;

  /// Generate buffer indices from iteration variables
  Array<PrimExpr> MakeIndices(const Array<IterVar> &ivs, int src_dst) const;

  /// Create boundary predicate for memory safety
  PrimExpr MakePredicate(arith::Analyzer *analyzer, const Array<IterVar> &ivs,
                         Array<PrimExpr> extents, int src_dst) const;
};

/// Node class for atomic maximum operations
class AtomicMaxNode : public AtomicOpBaseNode {
public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.AtomicMax", AtomicMaxNode,
                                    TileOperatorNode);

  static const Op &Get();
  const Op &GetElemOp() const override;
  TileOperator Clone() const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AtomicMaxNode>()
        .def_ro("src", &AtomicMaxNode::src)
        .def_ro("src_value", &AtomicMaxNode::src_value)
        .def_ro("dst", &AtomicMaxNode::dst)
        .def_ro("src_range", &AtomicMaxNode::src_range)
        .def_ro("dst_range", &AtomicMaxNode::dst_range)
        .def_ro("annotations", &AtomicMaxNode::annotations);
  }
};

/// Wrapper class for atomic maximum operations
class AtomicMax : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(AtomicMax, TileOperator,
                                             AtomicMaxNode);
  TVM_DLL
  AtomicMax(Array<PrimExpr> args,
            Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

/// Node class for atomic minimum operations
class AtomicMinNode : public AtomicOpBaseNode {
public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.AtomicMin", AtomicMinNode,
                                    TileOperatorNode);

  static const Op &Get();
  const Op &GetElemOp() const override;
  TileOperator Clone() const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AtomicMinNode>()
        .def_ro("src", &AtomicMinNode::src)
        .def_ro("src_value", &AtomicMinNode::src_value)
        .def_ro("dst", &AtomicMinNode::dst)
        .def_ro("src_range", &AtomicMinNode::src_range)
        .def_ro("dst_range", &AtomicMinNode::dst_range)
        .def_ro("annotations", &AtomicMinNode::annotations);
  }
};

/// Wrapper class for atomic minimum operations
class AtomicMin : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(AtomicMin, TileOperator,
                                             AtomicMinNode);
  TVM_DLL
  AtomicMin(Array<PrimExpr> args,
            Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_ATOMIC_REDUCE_H_
