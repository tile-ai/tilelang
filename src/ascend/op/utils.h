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

#include "op/utils.h"

namespace tvm {
namespace tl {

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
// storage-planning pass excludes them by scope, and the scale-load lowering
// resolves their address through the tile they are bound to.
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
         IsL0BBuffer(buffer) || IsL0CBuffer(buffer);
}

/*!
 * \brief tvm_access_ptr for a region whose innermost 2-D position travels as
 *        separate intrinsic arguments.
 *
 * The Ascend L1->L0 loads pass the M/K start positions as their own arguments
 * (``ascend_load_cbuf_to_ca(dst, src, mStartPosition, kStartPosition, ...)``
 * and the ``asc_copy_l12l0a_mx`` scale companion). The pointer is therefore
 * expected to carry only the *leading* dimensions -- the pipeline version, for
 * example -- and counting the last two dims into the offset as well would place
 * that position twice.
 *
 * This builds exactly that pointer, on top of the neutral
 * MakeAccessPtrFromRegion: the innermost two dims are re-based to zero before
 * delegating, which leaves their extents (and hence the reported extent) alone
 * and drops them from the offset.
 *
 * The buffer keeps ``require_2d == false`` semantics, so 1-D regions still take
 * the neutral function's dedicated path and are unaffected.
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
  return MakeAccessPtrFromRegion(BufferRegion(region->buffer, leading),
                                 rw_mask);
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_OP_UTILS_H_
