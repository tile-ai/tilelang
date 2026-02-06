/*!
 * \file tl/op/sync.h
 * \brief Synchronization intrinsics.
 *
 */

#ifndef TVM_TL_OP_SYNC_H_
#define TVM_TL_OP_SYNC_H_

#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>

#include "operator.h"

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief TileOperatorNode for wait operation.
 *
 * WaitOpNode represents a wait primitive,
 * which waits until a condition on a memory address is met.
 */
class WaitOpNode : public TileOperatorNode {
public:
  PrimExpr addr;     ///< The address to watch.
  PrimExpr expected; ///< The expected value to compare against.
  PrimExpr peer;     ///< The peer to compare against.
  int relation;      ///< The relation to compare against.

  bool is_distributed() const;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.WaitOp", WaitOpNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<WaitOpNode>()
        .def_ro("addr", &WaitOpNode::addr)
        .def_ro("expected", &WaitOpNode::expected)
        .def_ro("peer", &WaitOpNode::peer)
        .def_ro("relation", &WaitOpNode::relation);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const override;
};

/*!
 * \brief Wrapper for the WaitOp operator.
 */
class WaitOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(WaitOp, TileOperator, WaitOpNode);
  TVM_DLL WaitOp(Array<PrimExpr> args,
                 Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

/*!
 * \brief Synchronize all blocks at a system-level barrier
 *
 * void barrier_blocks(barrier, rank, num_ranks)
 *
 */
class BarrierBlocksOpNode : public TileOperatorNode {
public:
  PrimExpr local_bar_addr;       ///< Address expression for the local barrier
  PrimExpr offset;               ///< Byte offset within the barrier buffer
  Buffer local_bar;              ///< Local barrier buffer reference
  Array<PrimExpr> local_indices; ///< Indices used to access the barrier buffer
  bool need_fence;               ///< Whether need sys-level fence

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.BarrierBlocksOp", BarrierBlocksOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BarrierBlocksOpNode>()
        .def_ro("local_bar_addr", &BarrierBlocksOpNode::local_bar_addr)
        .def_ro("offset", &BarrierBlocksOpNode::offset)
        .def_ro("local_bar", &BarrierBlocksOpNode::local_bar)
        .def_ro("local_indices", &BarrierBlocksOpNode::local_indices)
        .def_ro("need_fence", &BarrierBlocksOpNode::need_fence);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const override;

  PrimExpr get_offset(const BufferLoadNode *load) const;

private:
  PrimExpr MakeLocalBarAddr(const LowerArgs &T) const;
};

/*!
 * \brief Wrapper for the BarrierBlocks operator
 */
class BarrierBlocksOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BarrierBlocksOp, TileOperator,
                                             BarrierBlocksOpNode);
  TVM_DLL BarrierBlocksOp(
      Array<PrimExpr> args,
      Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

/*!
 * \brief Initialize a barrier for GPU-level synchronization
 *
 * void init_barrier_gpu(barrier, expected)
 */
TVM_DLL const Op &init_barrier_gpu();

/*!
 * \brief Arrive at a barrier for GPU-level synchronization
 *
 * void arrive_barrier_gpu(barrier)
 */
TVM_DLL const Op &arrive_barrier_gpu();

/*!
 * \brief Wait at a barrier for GPU-level synchronization
 *
 * void wait_barrier_gpu(barrier)
 */
TVM_DLL const Op &wait_barrier_gpu();

/*!
 * \brief Synchronize at a barrier for GPU-level synchronization
 *
 * void sync_barrier_gpu(barrier)
 */
TVM_DLL const Op &sync_barrier_gpu();

/*!
 * \brief Create a memory fence at the block level (visible to all threads in
 * the current block)
 *
 * void fence_cta()
 */
TVM_DLL const Op &fence_cta();

/*!
 * \brief Synchronize all threads at the GPU level (visible to all blocks on the
 * current device)
 *
 * void fence_gpu()
 */
TVM_DLL const Op &fence_gpu();

/*!
 * \brief Synchronize all threads at the system level (visible in a node)
 *
 * void fence_sys()
 */
TVM_DLL const Op &fence_sys();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_SYNC_H_
