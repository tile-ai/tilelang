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
 * \file flatten_buffer.cc
 */

#include "transform/tirx_passes.h"

#include <tvm/tirx/transform.h>

#include "common/index_legalizer.h"

namespace tvm::tl {

tvm::transform::Pass FlattenBuffer() {
  auto legalize = tirx::transform::CreateFunctionPass(
      [](tirx::Function function, const IRModule &,
         const tvm::transform::PassContext &) {
        return LegalizeIndexArithmetic(std::move(function));
      },
      0, "tl.LegalizeIndexArithmetic");
  return tvm::transform::Sequential(
      {legalize, tirx_passes::LowerTensorLayouts(), legalize,
       tirx_passes::FlattenTensor(), legalize},
      "tl.FlattenBuffer");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.FlattenBuffer", FlattenBuffer);
}

} // namespace tvm::tl
