/*!
 * \file tl/ascend/op/ascend_allreduce_policy.h
 * \brief Ascend answers to the two AllReduce questions the shared lowerers ask.
 *
 * The shared reduce lowerer does not branch on the target itself; it asks the
 * backend `Impl` whether the lowering needs a shared-memory workspace. That
 * answer depends on the all-reduce *algorithm* rather than on the IR, so it
 * lives here, next to the Ascend implementation it describes. It has to mirror
 * the constexpr dispatch in src/tl_templates/ascend/reduce.h, which is why it
 * is stated once here instead of being restated in shared code.
 *
 * The XOR-butterfly power-of-two rule lives in backend/common/op/reduce.h as
 * CheckXorButterflyWidth and is simply not called by Ascend: AscendAllReduce
 * falls back to a shared-memory tree reduction (ub_reduce) for arbitrary
 * widths, so there is nothing to enforce.
 */

#ifndef TVM_TL_ASCEND_OP_ASCEND_ALLREDUCE_POLICY_H_
#define TVM_TL_ASCEND_OP_ASCEND_ALLREDUCE_POLICY_H_

namespace tvm {
namespace tl {
namespace ascend {

// Mirror AscendAllReduce<>::run()'s constexpr dispatch: the warp and cross-warp
// XOR-butterfly paths run entirely in registers, while the generic ub_reduce
// path needs the shared-memory workspace.
inline bool AllReduceNeedsWorkspace(int reducing_threads, int scale) {
  const bool is_pow2 = (reducing_threads & (reducing_threads - 1)) == 0;
  const bool scale_is_pow2 = (scale & (scale - 1)) == 0;
  if (reducing_threads <= 32 && is_pow2 && scale_is_pow2) {
    return false;
  }
  if (reducing_threads > 32 && is_pow2 && scale <= 32 && scale_is_pow2) {
    return true;
  }
  return reducing_threads > 1;
}

} // namespace ascend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_OP_ASCEND_ALLREDUCE_POLICY_H_
