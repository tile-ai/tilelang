/*!
 * \file tl/ascend/op/ascend_allreduce_policy.h
 * \brief Ascend answers to the two AllReduce questions the shared lowerers ask.
 *
 * The shared reduce/finalize lowerers do not branch on the target themselves;
 * they ask the backend `Impl` for policy. Two of those answers depend on the
 * all-reduce *algorithm* rather than on the IR, so they live here, next to the
 * Ascend implementation they describe:
 *
 *   - whether the reduction width must be a power of two, and
 *   - whether the lowering needs a shared-memory workspace.
 *
 * Both are properties of `tl::AscendAllReduce`; the second has to mirror the
 * constexpr dispatch in src/tl_templates/ascend/reduce.h, which is why it is
 * stated once here instead of being restated in shared code.
 */

#ifndef TVM_TL_ASCEND_OP_ASCEND_ALLREDUCE_POLICY_H_
#define TVM_TL_ASCEND_OP_ASCEND_ALLREDUCE_POLICY_H_

namespace tvm {
namespace tl {
namespace ascend {

// AscendAllReduce falls back to a shared-memory tree reduction (ub_reduce) for
// widths that are not powers of two, so the XOR-butterfly power-of-two
// requirement does not apply.
inline constexpr bool kAllReduceWidthRequiresPowerOfTwo = false;

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
