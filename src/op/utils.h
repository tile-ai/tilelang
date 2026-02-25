/*!
 * \file tl/op/utils.h
 * \brief Common utilities for TL ops.
 */

#ifndef TVM_TL_OP_UTILS_H_
#define TVM_TL_OP_UTILS_H_

#include "../target/stubs/cuda.h"
#include "./operator.h"
#include "region.h"
#include "tvm/runtime/base.h"
#include <tvm/tir/buffer.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace tl {

using namespace tir;

// Maps TVM DataType to CUDA's CUtensorMapDataType enum value.
TVM_DLL int to_CUtensorMapDataType(DataType dtype);

// Reverses an array (used for row-major/column-major layout conversion).
template <typename T> Array<T> ReverseArray(Array<T> array) {
  return Array<T>{array.rbegin(), array.rend()};
}

// Check if an PrimExpr is a buffer-like (BufferRegion/BufferLoad/tl.region)
// expression.
TVM_DLL bool IsBufferLikeExpr(const PrimExpr &expr);

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

// Build a tvm_access_ptr(handle) from a BufferLoad.
TVM_DLL PrimExpr MakeAccessPtrFromBufferLoad(const BufferLoad &load,
                                             int rw_mask);

// Check if a buffer is a fragment buffer (scope == "local.fragment")
inline bool IsFragmentBuffer(const Buffer &buffer) {
  return buffer.defined() && buffer.scope() == "local.fragment";
}

// Expand a lower-rank layout by prepending the leading dimensions of `buffer`
// so that the resulting layout input shape matches `buffer->shape`.
//
// This is useful when we infer a 2D swizzle layout from the trailing matrix
// dimensions of a higher-rank buffer (e.g. batched GEMM shared-memory buffers).
inline Layout ExpandLayoutToMatchBuffer(const Layout &layout,
                                        const Buffer &buffer) {
  if (!layout.defined() || !buffer.defined()) {
    return layout;
  }
  const size_t buffer_ndim = buffer->shape.size();
  const size_t layout_ndim = layout->InputDim();
  if (buffer_ndim <= layout_ndim) {
    return layout;
  }

  Array<PrimExpr> leading_shape;
  leading_shape.reserve(buffer_ndim - layout_ndim);
  for (size_t i = 0; i < buffer_ndim - layout_ndim; ++i) {
    leading_shape.push_back(buffer->shape[i]);
  }
  return layout->Expand(leading_shape);
}

inline bool IsSharedBuffer(const Buffer &buffer, bool allow_dynamic = true) {
  if (allow_dynamic) {
    return buffer.defined() &&
           (buffer.scope() == "shared" || buffer.scope() == "shared.dyn");
  } else {
    return buffer.defined() && buffer.scope() == "shared";
  }
}

inline bool IsGlobalBuffer(const Buffer &buffer) {
  return buffer.defined() && buffer.scope() == "global";
}

inline bool IsLocalBuffer(const Buffer &buffer, bool allow_var = false) {
  if (allow_var) {
    return buffer.defined() &&
           (buffer.scope() == "local" || buffer.scope() == "local.var");
  } else {
    return buffer.defined() && buffer.scope() == "local";
  }
}

inline bool IsLocalVarBuffer(const Buffer &buffer) {
  return buffer.defined() && buffer.scope() == "local.var";
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_UTILS_H_
