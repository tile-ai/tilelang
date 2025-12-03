/*!
 * \file tl/op/utils.h
 * \brief Common utilities for TL ops.
 */

#ifndef TVM_TL_OP_UTILS_H_
#define TVM_TL_OP_UTILS_H_

#include "./operator.h"
#include "region.h"
#include <tvm/tir/buffer.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace tl {

using namespace tir;

// Normalize an argument (BufferRegion/BufferLoad/tl.region)
// to BufferRegion so ops can uniformly consume regions.
// Note: tvm_access_ptr is no longer supported here.
TVM_DLL BufferRegion NormalizeToBufferRegion(const PrimExpr &arg);

// Build a tvm_access_ptr(handle) from a BufferRegion.
// - If `require_2d` is true, checks buffer ndim >= 2.
// - For 1D regions (when allowed), offset=min, extent=extent.
// - For ndim >= 2, offset sums all but last two dims using row-major strides,
//   extent is product of the last two extents.
TVM_DLL PrimExpr MakeAccessPtrFromRegion(const BufferRegion &region,
                                         int rw_mask, bool require_2d = false);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_UTILS_H_
