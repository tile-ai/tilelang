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

namespace std {

template <class T, T v> struct integral_constant {
  static constexpr T value = v;

  using value_type = T;
  using type = integral_constant;

  __device__ constexpr operator value_type() const noexcept { return value; }

  __device__ constexpr value_type operator()() const noexcept { return value; }
};

using false_type = integral_constant<bool, false>;
using true_type = integral_constant<bool, true>;

template <class T, class U> struct is_same : false_type {};

template <class T> struct is_same<T, T> : true_type {};

template <class T, class U>
inline constexpr bool is_same_v = is_same<T, U>::value;

namespace index_sequence_impl {

// Based on https://stackoverflow.com/a/32223343/11717224
template <size_t... Ints> struct index_sequence {
  using type = index_sequence;
  using value_type = size_t;
  static constexpr size_t size() noexcept { return sizeof...(Ints); }
};

template <class Sequence1, class Sequence2> struct _merge_and_renumber;

template <size_t... I1, size_t... I2>
struct _merge_and_renumber<index_sequence<I1...>, index_sequence<I2...>>
    : index_sequence<I1..., (sizeof...(I1) + I2)...> {};

template <size_t N>
struct make_index_sequence
    : _merge_and_renumber<typename make_index_sequence<N / 2>::type,
                          typename make_index_sequence<N - N / 2>::type> {};

template <> struct make_index_sequence<0> : index_sequence<> {};
template <> struct make_index_sequence<1> : index_sequence<0> {};

} // namespace index_sequence_impl

template <size_t... Ns>
using index_sequence = index_sequence_impl::index_sequence<Ns...>;

template <size_t N>
using make_index_sequence = index_sequence_impl::make_index_sequence<N>;

template <typename T> constexpr T min(T a, T b) { return a < b ? a : b; }

template <typename T> constexpr T max(T a, T b) { return a > b ? a : b; }

template <bool B, class T, class F> struct conditional {
  using type = T;
};

template <class T, class F> struct conditional<false, T, F> {
  using type = F;
};

template <bool B, class T, class F>
using conditional_t = typename conditional<B, T, F>::type;

template <bool B, class T = void> struct enable_if {};

template <class T> struct enable_if<true, T> {
  using type = T;
};
} // namespace std

#endif