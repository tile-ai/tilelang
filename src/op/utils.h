/*!
 * \file tl/op/utils.h
 * \brief Common utilities for TL ops.
 */

#ifndef TVM_TL_OP_UTILS_H_
#define TVM_TL_OP_UTILS_H_

#include "../ir/dtype.h"
#include "../ir/tensor_utils.h"
#include "../layout/expr_utils.h"
#include "./operator.h"
#include "cuda/stubs/cuda.h"
#include "support/check.h"
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/expr.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;

inline int64_t GetIntegerArgument(const Expr &value) {
  auto literal = value.as_or_throw<IntImm>()->value.as<int64_t>();
  ICHECK(literal.has_value()) << "Integer control argument is outside int64";
  return literal.value();
}

inline int64_t GetIntegerAnnotation(const ffi::Any &value) {
  if (auto integer = value.as<int64_t>())
    return integer.value();
  return GetIntegerArgument(value.as_or_throw<Expr>());
}

inline bool GetBoolAnnotation(const ffi::Any &value) {
  if (auto boolean = value.as<bool>())
    return boolean.value();
  return GetIntegerAnnotation(value) != 0;
}

// Padding belongs to the source tensor's logical element type. TF32 is a
// TileLang storage type accepted by native FloatImm, but it is outside
// MakeConst's built-in dtype dispatch. Native cast also retains the
// scalar/vector broadcast contract without changing valid loaded values.
template <typename ValueType>
inline PrimExpr MakeSafeValueConstant(PrimType dtype, ValueType value) {
  if (IsTensorFloat32(dtype))
    return cast(FloatImm(dtype.WithLanes(1), static_cast<double>(value)),
                dtype);
  return MakeConst(dtype, value);
}

// Annotation payloads intentionally preserve Python literals as Any values.
// A typed Map view does not convert its values. Construct numeric literals
// with the source element type before they reach analysis or annotation COW.
inline ffi::Map<Var, PrimExpr>
NormalizeSafeValueMap(const ffi::Any &annotation) {
  ffi::Map<Var, PrimExpr> result;
  for (const auto &[key, raw] :
       annotation.as_or_throw<ffi::Map<ffi::Any, ffi::Any>>()) {
    Expr expression = key.as_or_throw<Expr>();
    if (auto call = expression.as<Call>();
        call && call.value()->op.same_as(tirx::tensor_data_ptr_op()))
      expression = call.value()->args[0];
    auto tensor = expression.as_or_throw<TensorVar>();
    PrimExpr value{ffi::UnsafeInit{}};
    if (auto expr = raw.as<PrimExpr>())
      value = expr.value();
    else if (auto boolean = raw.as<bool>())
      value = MakeSafeValueConstant(tensor.dtype(), boolean.value());
    else if (auto integer = raw.as<int64_t>())
      value = MakeSafeValueConstant(tensor.dtype(), integer.value());
    else if (auto real = raw.as<double>())
      value = MakeSafeValueConstant(tensor.dtype(), real.value());
    else
      TVM_FFI_THROW(TypeError)
          << "safe_value_map values must be primitive expressions or numbers";
    result.Set(tensor, value);
  }
  return result;
}

/*! \brief Whether a constant has an all-zero bit pattern, excluding -0.0. */
TVM_TIRX_DLL bool IsZeroBitPattern(const PrimExpr &expr);

// Maps TVM PrimType to CUDA's CUtensorMapDataType enum value.
TVM_TIRX_DLL int to_CUtensorMapDataType(PrimType dtype);

// Reverses an array (used for row-major/column-major layout conversion).
template <typename T> ffi::Array<T> ReverseArray(ffi::Array<T> array) {
  return ffi::Array<T>{array.rbegin(), array.rend()};
}

// Whether an explicit operand names a native tensor or a logical tensor region.
TVM_TIRX_DLL bool IsTensorLikeExpr(const Expr &expr);

// Normalize TensorVar/TensorRegion/TensorLoad to a native TensorRegion.
// Pointer expressions deliberately cannot recover a tensor identity.
TVM_TIRX_DLL TensorRegion NormalizeToTensorRegion(const Expr &arg);

// The operation supplies the effect of a native region operand. Dynamic ranges
// stay in the TensorRegion expression, not in an attribute dictionary.
TVM_TIRX_DLL AccessRegion NormalizeToAccessRegion(
    const Expr &arg, int default_access_mask = kAccessReadWrite);

// Build a typed tl.access_ptr from a native TensorRegion.
// - If `require_2d` is true, checks tensor ndim >= 2.
// - The TensorLoad operand keeps all logical minima until layout lowering;
//   this helper does not flatten them or compute a row-major byte offset.
// - The access extent counts elements: one for rank 0, the region extent for
//   rank 1, or the product of the last two region extents for higher ranks.
TVM_TIRX_DLL Expr MakeAccessPtrFromRegion(const TensorRegion &region,
                                          int rw_mask, bool require_2d = false);

// Build a typed tl.access_ptr from a native TensorLoad.
TVM_TIRX_DLL Expr MakeAccessPtrFromTensorLoad(const TensorLoad &load,
                                              int rw_mask);

inline bool IsFragmentBuffer(const TensorVar &buffer) {
  return buffer.defined() && buffer.scope().name() == "local.fragment";
}

// Expand a lower-rank layout by prepending the leading dimensions of `buffer`
// so that the resulting layout input shape matches `buffer.shape()`.
//
// This is useful when we infer a 2D swizzle layout from the trailing matrix
// dimensions of a higher-rank buffer (e.g. batched GEMM shared-memory buffers).
inline Layout ExpandLayoutToMatchBuffer(const Layout &layout,
                                        const TensorVar &buffer) {
  if (!layout.defined() || !buffer.defined()) {
    return layout;
  }
  const size_t buffer_ndim = buffer.shape().size();
  const size_t layout_ndim = layout->InputDim();
  if (buffer_ndim <= layout_ndim) {
    return layout;
  }

  ffi::Array<PrimExpr> leading_shape;
  leading_shape.reserve(buffer_ndim - layout_ndim);
  for (size_t i = 0; i < buffer_ndim - layout_ndim; ++i) {
    leading_shape.push_back(buffer.shape()[i]);
  }
  return layout->Expand(leading_shape);
}

inline bool IsSharedBuffer(const TensorVar &buffer, bool allow_dynamic = true) {
  if (!buffer.defined()) {
    return false;
  }
  if (allow_dynamic) {
    return buffer.scope().name() == "shared" ||
           buffer.scope().name() == "shared.dyn";
  }
  return buffer.scope().name() == "shared";
}

inline bool IsTmemBuffer(const TensorVar &buffer) {
  if (!buffer.defined()) {
    return false;
  }
  return buffer.scope().name() == "shared.tmem";
}

inline bool IsGlobalBuffer(const TensorVar &buffer) {
  return buffer.defined() && buffer.scope().name() == "global";
}

inline bool IsValidCPAsyncTransferBytes(int bytes) {
  return bytes == 4 || bytes == 8 || bytes == 16;
}

inline bool IsLocalBuffer(const TensorVar &buffer, bool allow_var = false) {
  if (!buffer.defined()) {
    return false;
  }
  if (allow_var) {
    return buffer.scope().name() == "local" ||
           buffer.scope().name() == "local.var";
  }
  return buffer.scope().name() == "local";
}

inline bool IsLocalVarBuffer(const TensorVar &buffer) {
  return buffer.defined() && buffer.scope().name() == "local.var";
}

// True when global packed FP4 is copied into f8f6f4/mxf8f6f4 unpacked FP4 SMEM.
inline bool IsFP4PackedToUnpackedStorageCopy(PrimType global_dtype,
                                             PrimType shared_dtype) {
  return global_dtype.MatchesCode(kDLFloat4_e2m1fn) &&
         shared_dtype.MatchesCode(
             static_cast<DLDataTypeCode>(kFloat4E2M1Unpacked));
}

inline bool IsValidTMALoadDtypePair(PrimType global_dtype,
                                    PrimType shared_dtype) {
  if (global_dtype.MatchesCode(
          static_cast<DLDataTypeCode>(kFloat4E2M1Unpacked)) ||
      shared_dtype.MatchesCode(
          static_cast<DLDataTypeCode>(kFloat4E2M1Unpacked))) {
    return IsFP4PackedToUnpackedStorageCopy(global_dtype, shared_dtype);
  }
  if (global_dtype == shared_dtype) {
    return true;
  }
  return false;
}

inline bool IsValidTMAStoreDtypePair(PrimType global_dtype,
                                     PrimType shared_dtype) {
  return global_dtype == shared_dtype &&
         !shared_dtype.MatchesCode(
             static_cast<DLDataTypeCode>(kFloat4E2M1Unpacked));
}

inline bool IsValidTMADtypePair(bool is_load, PrimType global_dtype,
                                PrimType shared_dtype) {
  if (is_load) {
    return IsValidTMALoadDtypePair(global_dtype, shared_dtype);
  }
  return IsValidTMAStoreDtypePair(global_dtype, shared_dtype);
}

// Valid dtype pairs for TMA global->shared copies.
inline bool IsValidTMACopyDtypePair(PrimType global_dtype,
                                    PrimType shared_dtype) {
  return IsValidTMALoadDtypePair(global_dtype, shared_dtype);
}

// True only for the supported TMA load transition from packed global FP4 to
// unpacked shared FP4. The reverse store direction is not a valid TMA path.
inline bool IsFP4UnpackLoad(const TensorVar &src, const TensorVar &dst) {
  return IsGlobalBuffer(src) && IsSharedBuffer(dst) &&
         IsFP4PackedToUnpackedStorageCopy(src.dtype(), dst.dtype());
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_UTILS_H_
