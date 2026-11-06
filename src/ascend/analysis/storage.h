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

#ifndef TILELANG_ASCEND_ANALYSIS_STORAGE_H_
#define TILELANG_ASCEND_ANALYSIS_STORAGE_H_

#include "ir/storage_provenance.h"

namespace tvm::tl::ascend {

// Allocation policy and register spelling do not create new hardware spaces.
enum class StorageSpace {
  kUnknown,
  kGlobal,
  kLocal,
  kUB,
  kL1,
  kL0A,
  kL0B,
  kL0ASF,
  kL0BSF,
  kL0C,
  kSS,
};
TVM_TIRX_DLL StorageSpace StorageSpaceOf(const runtime::MemoryScope &scope);
TVM_TIRX_DLL bool CompatibleStorageScopes(const Type &a, const Type &b);
TVM_TIRX_DLL bool ProvenDisjointAscendStorage(const StorageAddress &a,
                                              const StorageAddress &b);
TVM_TIRX_DLL StorageProvenance MakeAscendStorageProvenance();
TVM_TIRX_DLL ParameterAliasContract CollectAscendParameterAliasContract(
    const tirx::Function &function,
    const ffi::Map<ffi::String, ffi::Any> &kernel_annotations = {});

} // namespace tvm::tl::ascend
#endif // TILELANG_ASCEND_ANALYSIS_STORAGE_H_
