/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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
