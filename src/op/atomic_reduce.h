/*!
 * \file tl/op/atomic_reduce.h
 * \brief Atomic operations base class and reduction operations (max/min)
 */

#ifndef TVM_TL_OP_ATOMIC_REDUCE_H_
#define TVM_TL_OP_ATOMIC_REDUCE_H_

#include "operator.h"
#include "parallel.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;
using namespace ffi;

/*!
 * \brief Base node class for atomic operations (add/max/min).
 *
 * This base class stores the shared atomic operation data and dispatches
 * lowering/layout inference to target-specific implementations.
 */
class AtomicOpBaseNode : public TileOperatorNode {
public:
  Optional<PrimExpr> src_value; ///< Scalar source, mutually exclusive with src.
  Optional<TensorVar> src;
  TensorVar dst{ffi::UnsafeInit{}}; ///< Required destination tensor.
  Array<Range> src_range,
      dst_range;                ///< Access ranges for source and destination
  Map<String, Any> annotations; ///< Annotations for the atomic operation
  // Supported annotation keys:
  //   - "coalesced_width": IntImm, width for memory coalescing optimization
  //   - "memory_order": IntImm, memory order for atomic operations

  mutable ParallelOp par_op_; ///< Associated parallel operation

  /// Lower through the registered target implementation.
  Stmt Lower(const LowerArgs &lower_args,
             sym::AnalyzerObj *analyzer) const override;

  /// Infer layout through the registered target implementation.
  LayoutMap InferLayout(const LayoutInferArgs &layout_args,
                        InferLevel level) const;

  /// Get memory order from annotations (default: relaxed = 0)
  int GetMemoryOrder() const {
    if (auto val = annotations.Get("memory_order")) {
      if (auto int_val = val->as<IntImmObj>()) {
        return static_cast<int>(int_val->value.as<int64_t>().value());
      }
      if (auto value = val->as<int64_t>())
        return static_cast<int>(*value);
    }
    return 0;
  }

  /// Get the element-wise operation Op (pure virtual, implemented by derived)
  virtual const Op &GetElemOp() const = 0;
};

using AtomicReduceTargetPredicate = bool (*)(Target target);

struct AtomicReduceImpl {
  const char *name;
  AtomicReduceTargetPredicate match_target;

  LayoutMap (*infer_layout)(const AtomicOpBaseNode &op,
                            const LayoutInferArgs &layout_args,
                            InferLevel level);

  Stmt (*lower)(const AtomicOpBaseNode &op, const LowerArgs &lower_args,
                sym::AnalyzerObj *analyzer);
};

void RegisterAtomicReduceImpl(AtomicReduceImpl impl);

/// Node class for atomic maximum operations
class AtomicMaxNode : public AtomicOpBaseNode {
public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.AtomicMax", AtomicMaxNode,
                                    TileOperatorNode);

  static const Op &Get();
  const Op &GetElemOp() const override;
  TileOperator Clone() const;

  static void RegisterReflection() {
    namespace refl = reflection;
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
  TVM_TIRX_DLL
  AtomicMax(Array<Expr> args,
            Map<String, Any> annotations = Map<String, Any>());
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
    namespace refl = reflection;
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
  TVM_TIRX_DLL
  AtomicMin(Array<Expr> args,
            Map<String, Any> annotations = Map<String, Any>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_ATOMIC_REDUCE_H_
