#ifndef TVM_TL_TRANSFORM_COMMON_STORAGE_SIZE_H_
#define TVM_TL_TRANSFORM_COMMON_STORAGE_SIZE_H_

#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>

#include <cstdint>

namespace tvm {
namespace tl {

/*! \brief Return whether a dtype uses packed four-bit buffer storage. */
inline bool IsPacked4BitStorage(DataType dtype) {
  return dtype.is_float4_e2m1fn() ||
         (dtype.bits() == 4 && (dtype.is_int() || dtype.is_uint()));
}

/*!
 * \brief Return the physical storage bits of one element of \p dtype.
 *
 * One element is a single scalar for scalar dtypes and a whole vector for
 * vector dtypes, e.g. 32 for `int32`, 4 for `float4_e2m1fn`, 128 for
 * `float16x8`.
 */
inline int64_t GetElemStorageBits(DataType dtype) {
  return static_cast<int64_t>(dtype.bits()) * dtype.lanes();
}

/*!
 * \brief Convert a byte offset into the merged shared-memory arena to the
 *        matching element offset of \p dtype.
 *
 * Packed four-bit dtypes are stored two logical elements per byte, so their
 * element offsets are not a whole number of bytes: `float4_e2m1fn` and `int4`
 * hold `8 / GetElemStorageBits(dtype)` logical elements per byte.
 * `DataType::bytes() * lanes()` rounds such an element up to a byte and would
 * halve the offset.
 *
 * Every other dtype -- including sub-byte ones that are not packed, such as
 * `float6_e3m2fnx4` -- is sized as `dtype.bytes() * lanes()` bytes per element
 * by `GetBufferStorageSizeBytes`, and is converted the same way so that sizing
 * and offsetting agree.
 */
inline PrimExpr ByteOffsetToElemOffset(const PrimExpr &byte_offset,
                                       DataType dtype) {
  DataType index_dtype = byte_offset.dtype();
  if (IsPacked4BitStorage(dtype)) {
    int64_t elem_bits = GetElemStorageBits(dtype);
    if (elem_bits < 8) {
      // float4_e2m1fn / int4: two logical elements per byte.
      return byte_offset * tirx::make_const(index_dtype, 8 / elem_bits);
    }
    if (elem_bits % 8 == 0) {
      return indexdiv(byte_offset,
                      tirx::make_const(index_dtype, elem_bits / 8));
    }
    // An odd lane count is not a whole number of elements per byte but still
    // packs: a logical element spans lanes() / 2 bytes.
    return indexdiv(byte_offset * tirx::make_const(index_dtype, 2),
                    tirx::make_const(index_dtype, dtype.lanes()));
  }
  return indexdiv(byte_offset,
                  tirx::make_const(index_dtype, dtype.bytes() * dtype.lanes()));
}

/*! \brief Return the physical storage bytes for a constant element count. */
inline int64_t GetBufferStorageSizeBytes(int64_t num_elements, DataType dtype) {
  if (IsPacked4BitStorage(dtype)) {
    int64_t total_bits = num_elements * dtype.bits() * dtype.lanes();
    return (total_bits + 7) / 8;
  }
  return num_elements * dtype.bytes() * dtype.lanes();
}

/*! \brief Return the physical storage bytes for a symbolic element count. */
inline PrimExpr GetBufferStorageSizeBytes(const PrimExpr &num_elements,
                                          DataType dtype) {
  DataType index_dtype = num_elements.dtype();
  if (IsPacked4BitStorage(dtype)) {
    int64_t storage_bits = static_cast<int64_t>(dtype.bits()) * dtype.lanes();
    PrimExpr total_bits =
        num_elements * tirx::make_const(index_dtype, storage_bits);
    return ceildiv(total_bits, tirx::make_const(index_dtype, 8));
  }
  int64_t storage_bytes = static_cast<int64_t>(dtype.bytes()) * dtype.lanes();
  return num_elements * tirx::make_const(index_dtype, storage_bytes);
}

/*!
 * \brief Return the physical storage bytes required by a compact buffer.
 *
 * DataType::bytes() rounds each scalar lane up to one byte.  Packed FP4 and
 * four-bit integer buffers are exceptions because their buffer ABI stores two
 * logical scalar values per byte.
 */
inline PrimExpr GetBufferStorageSizeBytes(const tirx::Buffer &buffer) {
  DataType index_dtype = DataType::Int(32);
  if (!buffer->shape.empty()) {
    index_dtype = buffer->shape[0].dtype();
  }
  if (!index_dtype.is_int() && !index_dtype.is_uint()) {
    index_dtype = DataType::Int(32);
  }

  PrimExpr num_elements = tirx::make_const(index_dtype, 1);
  for (const PrimExpr &extent : buffer->shape) {
    PrimExpr typed_extent = extent;
    if (typed_extent.dtype() != index_dtype) {
      typed_extent = tirx::Cast(index_dtype, typed_extent);
    }
    num_elements = num_elements * typed_extent;
  }
  return GetBufferStorageSizeBytes(num_elements, buffer->dtype);
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_STORAGE_SIZE_H_
