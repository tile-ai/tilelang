#ifndef TVM_TL_TRANSFORM_COMMON_STORAGE_SIZE_H_
#define TVM_TL_TRANSFORM_COMMON_STORAGE_SIZE_H_

#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>

#include <cstdint>
#include <numeric>

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
 * This is the width the merged shared-memory arena reserves per element: 32
 * for `int32`, 4 for `float4_e2m1fn`, 128 for `float16x8`.  Only four-bit
 * dtypes pack two values into a byte; every other dtype -- including sub-byte
 * vectors such as `float6_e3m2fnx4`, which is four bytes wide rather than its
 * 24 declared bits -- is stored as `DataType::bytes() * lanes()` whole bytes,
 * which is what `GetBufferStorageSizeBytes` reserves.
 */
inline int64_t GetElemStorageBits(DataType dtype) {
  if (IsPacked4BitStorage(dtype)) {
    return static_cast<int64_t>(dtype.bits()) * dtype.lanes();
  }
  return static_cast<int64_t>(dtype.bytes()) * dtype.lanes() * 8;
}

/*!
 * \brief Return the byte granularity at which an element of \p dtype starts.
 *
 * Elements can only start on byte offsets that are multiples of
 * `GetElemStorageBits(dtype) / gcd(GetElemStorageBits(dtype), 8)`: two-per-byte
 * elements start anywhere (1), `float16x8` on a multiple of 16, and elements
 * spanning an odd number of bytes -- `int8x3`, or the odd-lane packed
 * `int4x3` -- only on a multiple of 3.
 */
inline int64_t GetElemStorageAlignmentBytes(DataType dtype) {
  int64_t elem_bits = GetElemStorageBits(dtype);
  ICHECK_GT(elem_bits, 0) << "dtype " << dtype << " has no storage footprint";
  return elem_bits / std::gcd<int64_t>(elem_bits, 8);
}

/*!
 * \brief Convert a byte offset into the merged shared-memory arena to the
 *        matching element offset of \p dtype.
 *
 * An element starts every `GetElemStorageBits(dtype)` bits whether or not it
 * packs, so the offset is simply `byte_offset * 8 / GetElemStorageBits(dtype)`.
 * Dividing by `dtype.bytes() * dtype.lanes()` instead rounds a four-bit element
 * up to a whole byte and halves the offset, and would also ignore that a tight
 * sub-byte vector such as `float6_e3m2fnx4` occupies four bytes.
 *
 * Callers must place the buffer on a `GetElemStorageAlignmentBytes(dtype)`
 * boundary; a constant \p byte_offset is then exactly representable and the
 * division has no remainder.
 */
inline PrimExpr ByteOffsetToElemOffset(const PrimExpr &byte_offset,
                                       DataType dtype) {
  DataType index_dtype = byte_offset.dtype();
  int64_t elem_bits = GetElemStorageBits(dtype);
  ICHECK_GT(elem_bits, 0) << "dtype " << dtype << " has no storage footprint";
  if (const auto *imm = byte_offset.as<IntImmNode>()) {
    ICHECK_EQ(imm->value * 8 % elem_bits, 0)
        << "Shared-memory arena byte offset " << imm->value
        << " is not representable as an element index of " << dtype;
  }
  return indexdiv(byte_offset * tirx::make_const(index_dtype, 8),
                  tirx::make_const(index_dtype, elem_bits));
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
