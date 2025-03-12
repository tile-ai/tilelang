// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/runtime/runtime.h
 * \brief Runtime functions.
 *
 */

#ifndef TVM_TL_RUNTIME_RUNTIME_H_
#define TVM_TL_RUNTIME_RUNTIME_H_

namespace tvm {
namespace tl {

#if (__CUDACC_VER_MAJOR__ >= 12)
constexpr const char *tvm_tensormap_create_tiled =
    "__tvm_tensormap_create_tiled";
constexpr const char *tvm_tensormap_create_im2col =
    "__tvm_tensormap_create_im2col";
#endif // (__CUDACC_VER_MAJOR__ >= 12)
} // namespace tl
} // namespace tvm

#endif //  TVM_TL_RUNTIME_RUNTIME_H_