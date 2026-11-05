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
 * \brief TileLang entry point for native physical tensor flattening.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/transform.h>

#include "config_index_bitwidth.h"
#include "tensor_descriptor.h"

namespace tvm::tl {

tvm::transform::Pass FlattenBuffer() {
  // LowerTensorLayouts materializes logical views before this pass. Native
  // flattening preserves the public TensorType signature, rebuilds physical
  // tensor definitions and folds each view origin into its index exactly once.
  // Local variable initializers are explicit stores from the language facade.
  return tvm::transform::Sequential({LowerTensorDescriptors(),
                                     tirx::transform::FlattenTensor(),
                                     transform::LegalizeIndexArithmetic(false)},
                                    "tl.FlattenBuffer");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.FlattenBuffer", FlattenBuffer);
}

} // namespace tvm::tl
