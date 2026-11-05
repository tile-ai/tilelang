/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef TVM_TL_TRANSFORM_SIZE_VAR_DEFAULTS_H_
#define TVM_TL_TRANSFORM_SIZE_VAR_DEFAULTS_H_

#include <tvm/tirx/function.h>

namespace tvm::tl {

// A source size variable is an ordinary native Var. This construction-time
// registry tracks native lifetime, including temporary Python wrappers. It
// adds no facts to a native Analyzer and is never the serialized source of
// truth.
TVM_TIRX_DLL Var MarkSizeVarDefault(Var variable);
TVM_TIRX_DLL bool HasSizeVarDefault(const Var &variable);

// Read the current Function's explicit operand carrier. Its entries are
// optional defaults for exact ABI decoding, not permanent nonnegative facts.
TVM_TIRX_DLL ffi::Array<Var>
CollectSizeVarDefaults(const tirx::Function &function);

// Finish a frontend Function by attaching registered parameter/type sources.
// Existing serialized carriers are retained even without the registry.
TVM_TIRX_DLL tirx::Function AttachSizeVarDefaults(tirx::Function function);

// Transfer defaults to a derived typed signature (for example an output
// allocator). Filter by current signature identity, never by printed names.
TVM_TIRX_DLL tirx::Function
WithSizeVarDefaults(tirx::Function function, const ffi::Array<Var> &defaults);

// Erase the compile-time carrier after the ABI consumer has read it. This
// wrapper introduces no definitions or allocation scope of its own.
TVM_TIRX_DLL tirx::Function StripSizeVarDefaults(tirx::Function function);

} // namespace tvm::tl

#endif // TVM_TL_TRANSFORM_SIZE_VAR_DEFAULTS_H_
