/*!
 * \file tl/op/multimem.h
 * \brief Unified multimem operator that reuses T.copy's layout inference.
 *
 * Design: MakeSIMTLoop creates element-wise BufferLoad/BufferStore loop,
 * then ParallelOp + InferLayout + VectorizeLoop runs the standard pipeline.
 * Post-process replaces vectorized loads/stores on mcast buffers with
 * multimem call_extern instructions.
 */

#ifndef TVM_TL_OP_MULTIMEM_H_
#define TVM_TL_OP_MULTIMEM_H_

#include <tvm/target/target.h>
#include <tvm/tirx/stmt_functor.h>

#include "operator.h"
#include "parallel.h"

namespace tvm {
namespace tl {

using namespace tirx;

enum class MultimemMode : int {
  kLdReduce = 0,
  kSt = 1,
  kRed = 2,
  kTmaStore = 3, // multimem.cp.async.bulk: shared → mcast_global (plain store)
  kTmaRedStore =
      4, // multimem.cp.reduce.async.bulk: shared → mcast_global (reduce)
};

/*!
 * \brief Unified multimem operator for NVSwitch SHARP multicast operations.
 *
 * Supports three modes:
 *  - kLdReduce: load-reduce from multicast address into local buffer
 *  - kSt: store to multicast address (broadcast)
 *  - kRed: reduce into multicast address (no read-back)
 *
 * Lower flow:
 *  1. MakeSIMTLoop: creates element-wise parallel loop (BufferLoad ->
 * BufferStore)
 *  2. ParallelLoopFuser::Fuse + ParallelLoopTransformer::Substitute
 *  3. ParallelOp -> InferLayout at multiple levels
 *  4. LowerParallelLoop (PartitionLoop + VectorizeLoop)
 *  5. MultimemRewriter: post-process to replace mcast buffer accesses with
 * call_extern
 */
class MultimemOpNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_range, dst_range;
  MultimemMode mode;
  int reduce_op; // 0=ADD, 1=MIN, 2=MAX

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.MultimemOp", MultimemOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MultimemOpNode>()
        .def_ro("src", &MultimemOpNode::src)
        .def_ro("dst", &MultimemOpNode::dst)
        .def_ro("src_range", &MultimemOpNode::src_range)
        .def_ro("dst_range", &MultimemOpNode::dst_range);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  TileOperator Clone() const override;

private:
  For MakeSIMTLoop(arith::Analyzer *analyzer) const;
  For MakeTransformedSIMTLoop(arith::Analyzer *analyzer) const;
  Array<IterVar> MakeIterVars() const;
  Array<PrimExpr> MakeIndices(const Array<IterVar> &ivs, int src_dst) const;
  PrimExpr MakePredicate(arith::Analyzer *analyzer, const Array<IterVar> &ivs,
                         Array<PrimExpr> extents, int src_dst) const;
  bool IsPacked16BitMultimem() const;
  Stmt LowerPacked16Bit(const LowerArgs &T,
                        arith::Analyzer *analyzer) const;
  Stmt LowerBulkCopy(const LowerArgs &T, arith::Analyzer *analyzer) const;
};

class MultimemOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(MultimemOp, TileOperator,
                                             MultimemOpNode);
  TVM_DLL
  MultimemOp(Array<PrimExpr> args,
             Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_MULTIMEM_H_
