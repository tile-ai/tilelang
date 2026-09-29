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
 * \file tl/ascend/target.cc
 * \brief Ascend target kind registration.
 */

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

#include "dlpack/dlpack.h"

namespace tvm {

namespace refl = tvm::ffi::reflection;

TVM_REGISTER_TARGET_KIND("ascend", kDLExtDev)
    .add_attr_option<ffi::String>("mcpu")
    .add_attr_option<ffi::String>("arch")
    // Kernel entries return void, so the host cannot receive an int32
    // status code from the device the way CPU-codegen targets do.
    .add_attr_option<bool>("supports_kernel_status_return",
                           refl::DefaultValue(false))
    // AscendC emits lane-wise vector predicates, so vectorized Select does
    // not require a uniform condition.
    .add_attr_option<bool>("supports_vector_predicate",
                           refl::DefaultValue(true))
    // Widest vector load/store issued by the vectorizer, in bits.
    .add_attr_option<int64_t>("max_vector_bits", refl::DefaultValue(64))
    // Kernels are launched as a grid of AI cores only; threadIdx domains
    // are region-local (SimtVF) and are not runtime launch dimensions.
    .add_attr_option<bool>("launch_grid_only", refl::DefaultValue(true))
    .set_default_keys({"ascend"});

} // namespace tvm
