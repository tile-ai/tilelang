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
 * \brief Wait until *addr == expected* for GPU-level synchronization
 * void wait_eq(barrier, expected)
 */

TVM_DLL const Op &wait_eq();

/*!
 * \brief Synchronize at a barrier for GPU-level synchronization
 *
 * void sync_barrier_gpu(barrier)
 */
TVM_DLL const Op &sync_barrier_gpu();

/*!
 * \brief Synchronize at a barrier for GPU-level synchronization in cooperative
 * group style
 *
 * void sync_grid(barrier)
 */
TVM_DLL const Op &sync_grid();

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
  bool need_fence;              ///< Whether need sys-level fence

  static constexpr const char *_type_key = "tl.BarrierBlocksOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(BarrierBlocksOpNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BarrierBlocksOpNode>()
        .def_ro("local_bar_addr", &BarrierBlocksOpNode::local_bar_addr)
        .def_ro("offset", &BarrierBlocksOpNode::offset)
        .def_ro("local_bar", &BarrierBlocksOpNode::local_bar)
        .def_ro("local_indices", &BarrierBlocksOpNode::local_indices);
  }

  bool SEqualReduce(const BarrierBlocksOpNode *other,
                    SEqualReducer equal) const {
    return equal(local_bar_addr, other->local_bar_addr) &&
           equal(offset, other->offset) && equal(local_bar, other->local_bar) &&
           equal(local_indices, other->local_indices);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(local_bar_addr);
    hash_reduce(offset);
    hash_reduce(local_bar);
    hash_reduce(local_indices);
  }

  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;

  PrimExpr get_offset(const BufferLoadNode *load) const;

private:
  PrimExpr MakeLocalBarAddr(const LowerArgs &T) const;
};

/*!
 * \brief Wrapper for the BarrierBlocks operator
 */
class BarrierBlocksOp : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(BarrierBlocksOp, TileOperator,
                                BarrierBlocksOpNode);
  TVM_DLL BarrierBlocksOp(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

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
