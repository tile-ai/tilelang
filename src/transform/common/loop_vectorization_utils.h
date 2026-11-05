/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file tl/transform/common/loop_vectorization_utils.h
 * \brief Reuse native dtype legalization for vector address expressions.
 */
#ifndef TILELANG_LOOP_VECTORIZATION_UTILS_H_
#define TILELANG_LOOP_VECTORIZATION_UTILS_H_
#include "../../../3rdparty/tvm-tirx/src/tirx/ir/data_type_rewriter.h"
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl {

namespace detail {
// General native substitution preserves an explicit Cast's destination type.
// Replacing a scalar lane with a Ramp additionally changes that type's lanes.
class VectorExpressionLegalizer : public tirx::DataTypeLegalizer {
public:
  using DataTypeLegalizer::Mutate;
  using DataTypeLegalizer::Mutate_;

  UnchangedOr<PrimExpr> Mutate_(const prim::CastObj *op,
                                InplaceMode mode) final {
    PrimExpr value = Mutate(op->value, mode).ValueOrUnchanged(op->value);
    if (value.same_as(op->value))
      return ffi::Unchanged();
    return prim::Cast(
        op->ty.as_or_throw<PrimType>().WithLanes(value.ty().lanes()), value,
        op->loc);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *op,
                                InplaceMode mode) final {
    auto result = StmtExprMutator::Mutate_(op, InplaceMode::kDisallow);
    if (result.UnchangedOrSameAs(ffi::GetRef<PrimExpr>(op)))
      return ffi::Unchanged();
    auto load = std::move(result).ValueUnchecked().as_or_throw<TensorLoad>();
    return TensorLoad(load->source, load->indices, load->loc);
  }
};
} // namespace detail

// The expression query is used for scalar integer address algebra. Native
// dtype legalization broadcasts changed operands; TileLang also repairs the
// lanes stored by casts and tensor loads after introducing the lane Ramp.
// It is deliberately separate from statement execution/scalarization.
inline PrimExpr VectorizeNativeExpr(const PrimExpr &expr, const PrimVar &var,
                                    const PrimExpr &lanes) {
  PrimExpr ramp = prim::Ramp(IntImm(var.ty(), 0), IntImm(var.ty(), 1), lanes);
  auto rewrite = ffi::make_object<detail::VectorExpressionLegalizer>();
  rewrite->VarRemapSet(var, ramp);
  return rewrite->Mutate(expr, InplaceMode::kDisallow).ValueOrUnchanged(expr);
}

} // namespace tvm::tl
#endif // TILELANG_LOOP_VECTORIZATION_UTILS_H_
