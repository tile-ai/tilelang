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

#ifndef TVM_TL_TRANSFORM_TENSOR_DESCRIPTOR_H_
#define TVM_TL_TRANSFORM_TENSOR_DESCRIPTOR_H_

#include <tvm/ir/op.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

namespace tvm::tl {

// Frontend compatibility operation. Its operand is an original tensor
// parameter, so definition renewal and specialization use native Var identity.
const Op &tensor_descriptor_ptr_op();

// After frontend specialization, the original packed argument slot is stable
// while generic tensor passes rewrite local storage views. Only TileLang's
// packed binder consumes this operation. It borrows the descriptor, not data.
const Op &input_tensor_descriptor_op();

int InputTensorDescriptorIndex(const CallObj *call,
                               const ffi::Array<Var> &params);

tirx::Function LowerTensorDescriptorParameters(tirx::Function function);
tvm::transform::Pass LowerTensorDescriptors();

// Parameter pruning and nullable-input analysis must account for descriptor
// uses even after the TensorVar operand has become a packed argument index.
void CollectInputTensorDescriptorUses(const SeqStmt &body,
                                      const ffi::Array<Var> &params,
                                      std::unordered_set<const VarObj *> *uses);

// Call this before replacing a lowered function's parameter array. The caller
// owns the new signature; every descriptor-bearing parameter must be retained.
SeqStmt RemapInputTensorDescriptors(const tirx::Function &function,
                                    const ffi::Array<Var> &new_params);

} // namespace tvm::tl

#endif // TVM_TL_TRANSFORM_TENSOR_DESCRIPTOR_H_
