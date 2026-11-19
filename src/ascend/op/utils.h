/*!
 * \file tl/ascend/op/utils.h
 * \brief Ascend on-chip scope predicates and DMA pointer helpers.
 *
 * The scopes named here (L1, L0A/L0B/L0C and their dynamic variants) exist only
 * on Ascend, so they stay out of the backend-neutral src/op/utils.h.
 */

#ifndef TVM_TL_ASCEND_OP_UTILS_H_
#define TVM_TL_ASCEND_OP_UTILS_H_

#include <tvm/tirx/op.h>

#include <algorithm>

#include "op/builtin.h"
#include "op/utils.h"

namespace tvm {
namespace tl {

constexpr const char *kL0SFBindings = "tl.l0_sf_bindings";
using L0SFBindings = ffi::Map<tirx::Var, tirx::Var>;

namespace ascend {

// `asc_quant_mode` (fixpipe ``QuantMode_t``) values for the ``quant_pre``
// argument of the L0C copy C APIs. The compiler header declares this enum per
// target, and the a2a3 costmodel uses a different numbering under the same
// names (DEQF16=5, QS322BF16_PRE=35, REQ8=3). These come from the AscendC enum
// instead — the same one the float32 rows in ``GetCCQuantPre`` were taken from,
// which agrees with this numbering on every name it uses.
constexpr int kQuantNoQuant = 0;
constexpr int kQuantDEQF16 = 11;       // int32 -> half,     scalar quant
constexpr int kQuantQS322BF16Pre = 36; // int32 -> bfloat16, scalar quant

/*!
 * \brief True for the int32-source dequantisation modes.
 *
 * Unlike the plain casts, these multiply the accumulator by the scalar in the
 * fixpipe QUANT_PRE register, so the producer has to write that register before
 * the copy for the result to be well defined.
 */
inline bool IsInt32L0CDequantMode(int quant_pre) {
  return quant_pre == kQuantDEQF16 || quant_pre == kQuantQS322BF16Pre;
}

} // namespace ascend

// The annotation is emitted by alloc_l0*_sf, never inferred from consumers.
L0SFBindings CollectL0SFBindings(const tirx::Stmt &body);
class GemmBlockScaled;
void ValidateL0SFGemm(const GemmBlockScaled &gemm, const L0SFBindings &bindings,
                      arith::Analyzer *analyzer);

inline bool IsSSBuffer(const Buffer &buffer) {
  return buffer.defined() &&
         (buffer.scope() == "shared.ss" || buffer.scope() == "shared.ss.dyn");
}

inline bool IsL1Buffer(const Buffer &buffer) {
  return buffer.defined() &&
         (buffer.scope() == "shared.l1" || buffer.scope() == "shared.l1.dyn");
}

inline bool IsL0ABuffer(const Buffer &buffer) {
  return buffer.defined() &&
         (buffer.scope() == "shared.l0a" || buffer.scope() == "shared.l0a.dyn");
}

inline bool IsL0BBuffer(const Buffer &buffer) {
  return buffer.defined() &&
         (buffer.scope() == "shared.l0b" || buffer.scope() == "shared.l0b.dyn");
}

// MX scale-factor handles (alloc_l0a_sf/alloc_l0b_sf): dedicated scopes for
// the slot shadow of an L0A/L0B data tile. They never materialize storage —
// the hardware keys the slots to the bound data tile's address — so every
// allocation accounting excludes them by scope. Versioning and address reuse
// group them with their data tile; logical access analysis keeps them separate.
inline bool IsL0ASFBuffer(const Buffer &buffer) {
  return buffer.defined() && buffer.scope() == "shared.l0a.sf";
}

inline bool IsL0BSFBuffer(const Buffer &buffer) {
  return buffer.defined() && buffer.scope() == "shared.l0b.sf";
}

inline bool IsL0SFBuffer(const Buffer &buffer) {
  return IsL0ASFBuffer(buffer) || IsL0BSFBuffer(buffer);
}

inline bool IsL0CBuffer(const Buffer &buffer) {
  return buffer.defined() &&
         (buffer.scope() == "shared.l0c" || buffer.scope() == "shared.l0c.dyn");
}

inline bool IsAscendOnChipBuffer(const Buffer &buffer) {
  return IsSharedBuffer(buffer) || IsL1Buffer(buffer) || IsL0ABuffer(buffer) ||
         IsL0BBuffer(buffer) || IsL0CBuffer(buffer) || IsSSBuffer(buffer);
}

/*!
 * \brief A view-preserving pointer whose innermost 2-D position travels as
 *        separate intrinsic arguments.
 *
 * The Ascend L1->L0 loads pass the M/K start positions as their own arguments
 * (``ascend_load_cbuf_to_ca(dst, src, mStartPosition, kStartPosition, ...)``
 * and the ``asc_copy_l12l0a_mx`` scale companion). The pointer is therefore
 * expected to carry only the *leading* dimensions -- the pipeline version, for
 * example -- and counting the last two dims into the offset as well would place
 * that position twice.
 *
 * Keep the Buffer identity until layout lowering: a data Var alone cannot
 * identify which canonical matrix view supplies the physical addresses.
 */
inline PrimExpr MakeAscendLeadingDimAccessPtr(const BufferRegion &region,
                                              int rw_mask) {
  // ffi::Array is copy-on-write, so Set() here does not disturb the caller.
  auto leading = region->region;
  int n = static_cast<int>(leading.size());
  // Only a region with at least two dims has an inner 2-D position to separate.
  // A 1-D region keeps the neutral behaviour, where the min *is* the offset.
  for (int i = std::max(0, n - 2); n >= 2 && i < n; ++i) {
    const auto &r = leading[i];
    leading.Set(i, Range::FromMinExtent(make_zero(r->min.dtype()), r->extent));
  }
  ffi::Array<PrimExpr> indices;
  for (const Range &range : leading)
    indices.push_back(range->min);
  PrimExpr extent = leading.back()->extent;
  if (n >= 2)
    extent = extent * leading[n - 2]->extent;
  return Call(DataType::Handle(), tl::access_ptr(),
              {BufferLoad(region->buffer, indices), extent,
               IntImm(DataType::Int(32), rw_mask)});
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_OP_UTILS_H_
