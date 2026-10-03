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

/*!
 * \file backend/common/bf16_numeric_op.h
 * \brief Shared classification of bfloat16 numeric operations.
 *
 * Metal represents bf16x4/x8 vectors as packed uintN carriers that are only
 * valid for pure memory copies.  The vectorizer and the Metal codegen both
 * need the same notion of "bf16 numeric operation" (arithmetic, comparison,
 * min/max, non-identity cast, numeric call) so the packed carriers are never
 * fed into integer-typed MSL operations.
 */
#ifndef TVM_TL_BACKEND_COMMON_BF16_NUMERIC_OP_H_
#define TVM_TL_BACKEND_COMMON_BF16_NUMERIC_OP_H_

#include <tvm/runtime/data_type.h>

namespace tvm {
namespace tl {

/*!
 * \brief Whether a binary arithmetic/comparison/min/max node is a bf16
 * numeric operation.
 *
 * An operation is classified as numeric when bfloat16 appears on either side
 * or as the result dtype.
 */
inline bool IsBF16NumericBinaryOp(DataType lhs, DataType rhs, DataType result) {
  return lhs.is_bfloat16() || rhs.is_bfloat16() || result.is_bfloat16();
}

/*!
 * \brief Whether a cast is a bf16 numeric conversion.
 *
 * Identity casts are bit-level pass-throughs (pure copies) and are excluded.
 */
inline bool IsBF16NumericCastOp(DataType from, DataType to) {
  return (from.is_bfloat16() || to.is_bfloat16()) && from != to;
}

/*!
 * \brief Whether a call is a bf16 numeric operation based on its result dtype.
 *
 * if_then_else is a bit-level pick used by predicated pure copies and is not
 * a numeric operation even when its result is bfloat16.
 */
inline bool IsBF16NumericCallOp(DataType result_dtype,
                                bool is_bit_level_pure_copy) {
  return result_dtype.is_bfloat16() && !is_bit_level_pure_copy;
}

/*!
 * \brief Whether a call operand is a bf16 numeric operand.
 *
 * The codegen rejects packed bf16 carriers in call operands unless the call
 * is a bit-level pure-copy call (if_then_else / reinterpret).
 */
inline bool IsBF16NumericCallArg(DataType arg_dtype,
                                 bool is_bit_level_pure_copy) {
  return arg_dtype.is_bfloat16() && !is_bit_level_pure_copy;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_BF16_NUMERIC_OP_H_