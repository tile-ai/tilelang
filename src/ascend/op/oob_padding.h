/*!
 * \file tl/ascend/op/oob_padding.h
 * \brief Shared helpers for Ascend DMA copy out-of-bounds (OOB) clamping and
 *        L1 padding fills.
 *
 * These were originally private helpers inside ascend/op/copy.cc. They are
 * exposed here so a pre-AutoSchedule pass can perform the same OOB clamp and
 * emit the padding fills as standalone statements (visible to scheduling),
 * while copy.cc's lowering keeps consuming the already-clamped ranges.
 */

#ifndef TVM_TL_ASCEND_OP_OOB_PADDING_H_
#define TVM_TL_ASCEND_OP_OOB_PADDING_H_

#include "op/copy.h"

#include "ascend/layout/ascend_layouts.h"
#include "layout/layout.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;

/*! \brief Ascend DMA route selected from the source/destination scopes. */
enum class DMAPath {
  kNone,
  kGMToUB,
  kUBToGM,
  kGMToL1,
  kL1ToL0A,
  kL1ToL0B,
  kL0CToUB,
  kUBToL1,
  kL0CToGM,
};

/*!
 * \brief Classify a copy between the given buffer scopes.
 * \return The matching Ascend DMA route, or DMAPath::kNone when the copy is
 * not handled by an Ascend DMA instruction.
 */
DMAPath GetDMAPath(const Buffer &src, const Buffer &dst);

/*!
 * \brief Look up a buffer's inferred Layout in a layout map, matching either
 * the buffer object or its underlying data Var. Returns nullopt if absent.
 */
Optional<Layout> FindLayoutForBuffer(const LayoutMap &layout_map,
                                     const Buffer &buffer);

/*!
 * \brief Result of clamping a DMA copy's ranges to stay in bounds.
 *
 * `src` / `dst` are the (possibly shrunk) semantic copy ranges, whose extents
 * may be zero. A zero extent carries both an empty transfer and an invalid
 * fixed/singleton-axis index (for example, `src[group, ...]` with an out-of-
 * range `group`). `clamped` reports whether the helper rewrote those ranges.
 * Runtime guards are deliberately not part of this result; DMA lowering
 * derives them from the final ranges after scheduling.
 */
struct BoundedDMACopyRanges {
  Array<Range> src;
  Array<Range> dst;
  bool clamped;
};

/*!
 * \brief Clamp a copy's trailing tail to the valid in-bounds region.
 *
 * Pairs non-singleton axes between src and dst (swapping the trailing two for a
 * transposed copy) and shrinks any axis whose range exceeds the buffer shape.
 * Each paired non-singleton tile origin is required to start in bounds
 * (`0 <= min < shape`); only the upper end of the tile may exceed its shape.
 * Negative or fully out-of-bounds non-singleton origins are outside this
 * helper's contract. The clamped ranges preserve possibly-zero semantic
 * extents for empty dynamic extents and fixed/singleton-axis validity. Returns
 * the original ranges with `clamped == false` when no clamp is needed or the
 * shape structure is unsupported.
 */
BoundedDMACopyRanges ClampDMACopyTail(const CopyNode &op,
                                      arith::Analyzer *analyzer);

/*!
 * \brief Build a target-neutral tl.fill for an L1 buffer region.
 *
 * Keeping the semantic region on tl.fill lets AutoSchedule reason about the
 * exact write footprint. Ascend fill lowering converts it to
 * ascend_fill_l1 after scheduling, which codegen emits as asc_fill_l1.
 */
Stmt MakeL1Fill(const Buffer &dst, const Array<Range> &region,
                const PrimExpr &value);

/*!
 * \brief Emit a tl.fill for the column (C0-block) tail of a padded GM->L1
 *        destination, or nullopt when there is no tail.
 *
 * \param valid_cols In-bounds column extent actually filled by the copy.
 */
Optional<Stmt> MakeL1ColPadding(const CopyNode &op,
                                const AscendFractalLayoutInfo &layout,
                                arith::Analyzer *analyzer,
                                const PrimExpr &valid_cols);

/*!
 * \brief Emit a tl.fill for the row tail of a padded GM->L1 destination, or
 *        nullopt when there is no tail.
 *
 * A full-row tail denotes a fully out-of-bounds non-singleton tile origin and
 * is omitted according to ClampDMACopyTail's origin contract.
 */
Optional<Stmt> MakeL1RowPadding(const CopyNode &op,
                                const AscendFractalLayoutInfo &layout,
                                arith::Analyzer *analyzer,
                                const PrimExpr &valid_rows);

} // namespace ascend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_OP_OOB_PADDING_H_
