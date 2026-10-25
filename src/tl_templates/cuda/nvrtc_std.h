/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifdef __CUDACC_RTC__

// Disable problematic CUDA standard library headers in NVRTC environment.
// Vector types (float4, uchar, etc.) are built-in to NVRTC and don't need
// these headers. CCCL renamed the include guard between releases:
//   * libcudacxx (CUDA <=12): _LIBCUDACXX___TUPLE_VECTOR_TYPES_H
//   * CCCL (CUDA 13+):        _CUDA_STD___TUPLE_VECTOR_TYPES_H  (header now
//                             lives at
//                             cccl/cuda/std/__tuple_dir/vector_types.h)
// Define both so the inclusion is suppressed regardless of toolkit version.
#define _LIBCUDACXX___TUPLE_VECTOR_TYPES_H
#define _CUDA_STD___TUPLE_VECTOR_TYPES_H

// CUDA 13+ CCCL ships ``cuda/std/__tuple_dir/structured_bindings.h`` which
// forward-declares ``std::tuple_size<class _Tp>`` / ``std::tuple_element<...>``
// using a single-type parameter. Cutlass's ``cute/container/tuple.hpp``
// independently forward-declares the same names with variadic packs under
// ``__CUDACC_RTC__``, producing an ODR template-parameter mismatch. By
// predefining CCCL's include guard before any cute header is parsed, CCCL's
// header becomes a no-op and cute's variadic declarations win — cute is what
// the kernel templates rely on under NVRTC.
#define _CUDA_STD___TUPLE_STRUCTURED_BINDINGS_H

using int8_t = signed char;
using uint8_t = unsigned char;
using int16_t = signed short;
using uint16_t = unsigned short;
using int32_t = signed int;
using uint32_t = unsigned int;
using int64_t = signed long long;
using uint64_t = unsigned long long;
using cuuint64_t = unsigned long long;

#ifndef CU_TENSOR_MAP_NUM_QWORDS
#define CU_TENSOR_MAP_NUM_QWORDS 16

struct CUtensorMap_st {
#if defined(__cplusplus) && (__cplusplus >= 201103L)
  alignas(64)
#elif __STDC_VERSION__ >= 201112L
  _Alignas(64)
#endif
      cuuint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS];
};

using CUtensorMap = CUtensorMap_st;
#endif

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

namespace std {
using cuda::std::conditional;
using cuda::std::conditional_t;
using cuda::std::enable_if;
using cuda::std::enable_if_t;
using cuda::std::extent;
using cuda::std::extent_v;
using cuda::std::false_type;
using cuda::std::index_sequence;
using cuda::std::integral_constant;
using cuda::std::is_integral;
using cuda::std::is_integral_v;
using cuda::std::is_pointer;
using cuda::std::is_pointer_v;
using cuda::std::is_same;
using cuda::std::is_same_v;
using cuda::std::is_void;
using cuda::std::is_void_v;
using cuda::std::make_index_sequence;
using cuda::std::max;
using cuda::std::min;
using cuda::std::remove_extent;
using cuda::std::remove_extent_t;
using cuda::std::true_type;
} // namespace std

#endif // __CUDACC_RTC__
