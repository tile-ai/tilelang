/*!
 * \file tl/op/comm.h
 * \brief Implementation of Inter-core Communication Operators
 */

#ifndef TVM_TL_OP_COMM_H_
#define TVM_TL_OP_COMM_H_

#include "../target/sunmmio_utils.h"
#include "operator.h"

namespace tvm {
namespace tl {

TVM_DLL const Op &broadcast_();

// Positional argument layout of the broadcast_() leaf intrinsic. Producers
// (comm.cc, via AppendBroadcastArgs) and consumers (the broadcast-barrier
// analysis in InjectSunmmioSync) must index it through these constants.
enum BroadcastArg : int {
  kBroadcastArgSrc = 0,           // source region
  kBroadcastArgDst = 1,           // destination region
  kBroadcastArgDirection = 2,     // 0 = horizontal/row, 1 = vertical/col
  kBroadcastArgMask = 3,          // i64 direction-local receiver bitmask
  kBroadcastArgSrcOffsetByte = 4, // source-pointer byte offset
  kBroadcastArgCount = 5,   // fixed args before optional src_core/metadata
  kBroadcastArgSrcCore = 5, // optional; before odma_unit
};

using namespace tir;

class BroadcastOpNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_range, dst_range;
  PrimExpr src_expr, dst_expr;
  IntImm size;
  PrimExpr src_core;
  int direction;
  // Byte offset added to the source pointer at codegen. Default 0. Set by
  // the Sunmmio bf16 GEMM legalization pass (via AllgatherOp) so the second
  // HLink pass re-stages south-bound data into the destination's north bank.
  int srcOffsetByte_ = 0;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.comm_broadcast", BroadcastOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BroadcastOpNode>()
        .def_ro("src", &BroadcastOpNode::src)
        .def_ro("dst", &BroadcastOpNode::dst)
        .def_ro("src_range", &BroadcastOpNode::src_range)
        .def_ro("dst_range", &BroadcastOpNode::dst_range)
        .def_ro("src_core", &BroadcastOpNode::src_core);
  }

  TileOperator Clone() const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
};

class BroadcastOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BroadcastOp, TileOperator,
                                             BroadcastOpNode);
  TVM_DLL BroadcastOp(Array<PrimExpr> args,
                      Map<String, ObjectRef> annotations = {});
  static const Op &Get();
};

class PutOpNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_range, dst_range;
  PrimExpr src_expr, dst_expr;
  PrimExpr src_core, dst_core;
  IntImm size;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.comm_put", PutOpNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PutOpNode>()
        .def_ro("src", &PutOpNode::src)
        .def_ro("dst", &PutOpNode::dst)
        .def_ro("src_range", &PutOpNode::src_range)
        .def_ro("dst_range", &PutOpNode::dst_range)
        .def_ro("src_core", &PutOpNode::src_core)
        .def_ro("dst_core", &PutOpNode::dst_core)
        .def_ro("size", &PutOpNode::size);
  }

  TileOperator Clone() const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
};

class PutOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PutOp, TileOperator, PutOpNode);
  TVM_DLL PutOp(Array<PrimExpr> args, Map<String, ObjectRef> annotations = {});
  static const Op &Get();
};

class AllgatherOpNode : public TileOperatorNode {
public:
  PrimExpr send, recv;
  int direction;
  IntImm size;
  // -1 sentinel = legacy mode (recv has an extra leading axis of length K).
  // Otherwise the (already-normalized non-negative) axis along which recv
  // concatenates the K per-core contributions.
  int axis;
  // Current core id, passed from the Python frontend as the blockIdx.x binding.
  // Optional during migration so older 5-argument call sites remain parseable.
  PrimExpr cid;
  // Supported annotation keys:
  //   - kAttrSrcOffsetByte ("src_offset_byte"): IntImm, byte offset added to
  //     the source pointer at codegen. Set by the Sunmmio bf16 GEMM
  //     legalization pass to re-stage south-bound A data into a destination
  //     buffer's north bank.
  Map<String, ObjectRef> annotations;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.comm_allgather", AllgatherOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AllgatherOpNode>()
        .def_ro("send", &AllgatherOpNode::send)
        .def_ro("recv", &AllgatherOpNode::recv)
        .def_ro("direction", &AllgatherOpNode::direction)
        .def_ro("size", &AllgatherOpNode::size)
        .def_ro("axis", &AllgatherOpNode::axis)
        .def_ro("cid", &AllgatherOpNode::cid)
        .def_ro("annotations", &AllgatherOpNode::annotations);
  }

  int GetSrcOffsetByte() const {
    if (auto val = annotations.Get(kAttrSrcOffsetByte)) {
      if (auto int_val = val->as<IntImmNode>()) {
        return int_val->value;
      }
    }
    return 0;
  }

  TileOperator Clone() const override;
  LayoutMap ComputeLayout(const LayoutInferArgs &T, InferLevel level,
                          Buffer src, Buffer dst) const;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
};

class AllgatherOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(AllgatherOp, TileOperator,
                                             AllgatherOpNode);
  TVM_DLL AllgatherOp(Array<PrimExpr> args,
                      Map<String, ObjectRef> annotations = {});
  static const Op &Get();
};

/** Mesh directions in which a communication call consumes its source. */
enum class CommunicationDirections {
  kNone,
  kHorizontal,
  kVertical,
  kHorizontalAndVertical,
};

/**
 * \brief Whether a TIR call is an inter-core communication tile op.
 */
TVM_DLL bool IsCommunicationOp(const Call &call);

/**
 * \brief Mesh directions in which a communication call consumes its original
 * source.
 *
 * Two-dimensional broadcast and allgather use different first-phase
 * directions. Fixed put routes are derived from their core coordinates;
 * dynamic put routes conservatively return both directions.
 */
TVM_DLL CommunicationDirections
GetCommunicationSourceDirections(const Call &call, Target target);

class AllreduceOpNode : public TileOperatorNode {
public:
  PrimExpr src, dst;
  PrimExpr row_allgather, col_allgather;
  PrimExpr dst_copy;
  StringImm type;
  int direction;
  IntImm dim;
  IntImm clear;
  PrimExpr cid;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.comm_allreduce", AllreduceOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AllreduceOpNode>()
        .def_ro("src", &AllreduceOpNode::src)
        .def_ro("dst", &AllreduceOpNode::dst)
        .def_ro("row_allgather", &AllreduceOpNode::row_allgather)
        .def_ro("col_allgather", &AllreduceOpNode::col_allgather)
        .def_ro("type", &AllreduceOpNode::type)
        .def_ro("dim", &AllreduceOpNode::dim)
        .def_ro("clear", &AllreduceOpNode::clear)
        .def_ro("direction", &AllreduceOpNode::direction)
        .def_ro("dst_copy", &AllreduceOpNode::dst_copy)
        .def_ro("cid", &AllreduceOpNode::cid);
  }

  TileOperator Clone() const override;
  LayoutMap ComputeLayout(const LayoutInferArgs &T, InferLevel level,
                          Buffer src, Buffer dst, int dim) const;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
};

class AllreduceOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(AllreduceOp, TileOperator,
                                             AllreduceOpNode);
  TVM_DLL AllreduceOp(Array<PrimExpr> args,
                      Map<String, ObjectRef> annotations = {});
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_COMM_H_
