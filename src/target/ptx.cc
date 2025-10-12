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
 * \file ptx.cc
 */

#include "ptx.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tvm::tl {
namespace codegen {

// PTX related data structures and functions.
namespace ptx {

static const char *enum_to_str[] = {
    "kInt4",        "kUInt4",         "kInt8",    "kUInt8",    "kInt16",
    "kUInt16",      "kInt32",         "kUInt32",  "kInt64",    "kUInt64",
    "kFloat8_e4m3", "kFloat8_e5m2",   "kFloat16", "kBFloat16", "kFloat16x2",
    "kFloat32",     "kTensorFloat32", "kFloat64", "kBit1",     "kBit8",
    "kBit16",       "kBit32",         "kBit64"};

static const char *dtype_str[] = {
    ".s4",   ".u4",  ".s8",   ".u8",   ".s16", ".u16",  ".s32",   ".u32",
    ".s64",  ".u64", ".e4m3", ".e5m2", ".f16", ".bf16", ".f16x2", ".f32",
    ".tf32", ".f64", ".b1",   ".b8",   ".b16", ".b32",  ".b64"};
static const uint32_t num_bits[] = {4,  4,  8, 8, 16, 16, 32, 32,
                                    64, 64, 8, 8, 16, 16, 32, 32,
                                    32, 64, 1, 8, 16, 32, 64};

/*!
 * \brief Create PTX data type from string.
 */
DataType DTypeFromString(const std::string str) {
  if (str == "int4" || str == ".s4") {
    return DataType::kInt4;
  } else if (str == "uint4" || str == ".u4") {
    return DataType::kUInt4;
  } else if (str == "int8" || str == ".s8") {
    return DataType::kInt8;
  } else if (str == "uint8" || str == ".u8") {
    return DataType::kUInt8;
  } else if (str == "int16" || str == ".s16") {
    return DataType::kInt16;
  } else if (str == "uint16" || str == ".u16") {
    return DataType::kUInt16;
  } else if (str == "int32" || str == ".s32") {
    return DataType::kInt32;
  } else if (str == "uint32" || str == ".u32") {
    return DataType::kUInt32;
  } else if (str == "int64" || str == ".s64") {
    return DataType::kInt64;
  } else if (str == "uint64" || str == ".u64") {
    return DataType::kUInt64;
  } else if (str == "float8_e4m3" || str == "e4m3" || str == ".e4m3") {
    return DataType::kFloat8_e4m3;
  } else if (str == "float8_e5m2" || str == "e5m2" || str == ".e5m2") {
    return DataType::kFloat8_e5m2;
  } else if (str == "float16" || str == "fp16" || str == ".f16") {
    return DataType::kFloat16;
  } else if (str == "bfloat16" || str == "bf16") {
    return DataType::kBFloat16;
  } else if (str == ".f16x2") {
    return DataType::kFloat16x2;
  } else if (str == "float32" || str == "fp32" || str == ".f32") {
    return DataType::kFloat32;
  } else if (str == "tf32") {
    return DataType::kTensorFloat32;
  } else if (str == "float64" || str == "fp64" || str == ".f64") {
    return DataType::kFloat64;
  } else if (str == "int1" || str == ".b1") {
    return DataType::kBit1;
  } else if (str == ".b8") {
    return DataType::kBit8;
  } else if (str == ".b16") {
    return DataType::kBit16;
  } else if (str == ".b32") {
    return DataType::kBit32;
  } else if (str == ".b64") {
    return DataType::kBit64;
  } else {
    LOG(FATAL) << "Unrecognized PTX data type " << str;
  }
}

std::string DTypeEnumToString(const ptx::DataType &dtype) {
  return "tl::DataType::" + std::string(enum_to_str[static_cast<int>(dtype)]);
}

std::string DTypeEnumToString(const std::string &dtype) {
  return "tl::DataType::" +
         std::string(enum_to_str[static_cast<int>(DTypeFromString(dtype))]);
}

/*!
 * \brief Get the string representation of given PTX data type.
 */
inline std::string DTypeToString(DataType dtype) {
  return dtype_str[static_cast<int>(dtype)];
}

/*!
 * \brief Get the number of bits of given PTX data type.
 */
inline uint32_t DTypeBits(DataType dtype) {
  return num_bits[static_cast<int>(dtype)];
}

inline bool DTypeIsInteger(DataType dtype) {
  return dtype == DataType::kInt4 || dtype == DataType::kInt8 ||
         dtype == DataType::kInt16 || dtype == DataType::kInt32 ||
         dtype == DataType::kInt64 || dtype == DataType::kUInt4 ||
         dtype == DataType::kUInt8 || dtype == DataType::kUInt16 ||
         dtype == DataType::kUInt32 || dtype == DataType::kUInt64;
}

/*!
 * \brief Extract the value m, n, k from string m*n*k*
 */
std::tuple<int, int, int> ParseMMAShape(const std::string &str) {
  size_t pos_m = str.find('m'), pos_n = str.find('n'), pos_k = str.find('k');
  CHECK(pos_m != str.npos && pos_n != str.npos && pos_k != str.npos)
      << "Cannot parse MMA shape " << str;
  int m = std::stoi(str.substr(pos_m + 1, pos_n - pos_m - 1)),
      n = std::stoi(str.substr(pos_n + 1, pos_k - pos_n - 1)),
      k = std::stoi(str.substr(pos_k + 1));
  return std::make_tuple(m, n, k);
}

/*!
 * \brief Layout Type
 */
enum class LayoutType : int { kRowMajor = 0, kColumnMajor = 1 };

/*!
 * \brief Parse layout type
 */
LayoutType LayoutTypeFromString(const std::string &str) {
  if (str == "row") {
    return LayoutType::kRowMajor;
  } else if (str == "col") {
    return LayoutType::kColumnMajor;
  } else {
    LOG(FATAL) << "Unrecognized layout type " << str;
  }
}

/*!
 * \brief Parse layout type from bool.
 */
LayoutType LayoutTypeFromBool(const bool &layout) {
  if (layout) {
    return LayoutType::kRowMajor;
  } else {
    return LayoutType::kColumnMajor;
  }
}

static const char *layout_type_str[] = {"row", "col"};

/*!
 * \brief Convert layout type to string.
 */
inline std::string LayoutTypeToString(LayoutType layout) {
  return layout_type_str[static_cast<int>(layout)];
}

/*!
 * \brief MMA Configurations, used to determine validity.
 */
struct MMAConfig {
  explicit MMAConfig(int m, int n, int k, DataType dtype_mul, bool use_bit_op,
                     bool sparse)
      : m(m), n(n), k(k), dtype_mul(dtype_mul), use_bit_op(use_bit_op),
        sparse(sparse) {}
  int m, n, k;
  DataType dtype_mul;
  bool use_bit_op;
  bool sparse;
  inline bool operator==(const MMAConfig &other) {
    return m == other.m && n == other.n && k == other.k &&
           dtype_mul == other.dtype_mul && use_bit_op == other.use_bit_op &&
           sparse == other.sparse;
  }
};

/*!
 * \brief Valid MMA configurations
 * \note Reference:
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-shape
 */
const MMAConfig valid_mma_configs[] = {
    MMAConfig(8, 8, 4, DataType::kFloat64, false, false),
    MMAConfig(8, 8, 4, DataType::kFloat16, false, false),
    MMAConfig(16, 8, 8, DataType::kFloat16, false, false),
    MMAConfig(16, 8, 16, DataType::kFloat16, false, false),
    MMAConfig(16, 8, 8, DataType::kBFloat16, false, false),
    MMAConfig(16, 8, 16, DataType::kBFloat16, false, false),
    MMAConfig(16, 8, 4, DataType::kFloat32, false, false),
    MMAConfig(16, 8, 8, DataType::kFloat32, false, false),
    MMAConfig(16, 8, 4, DataType::kTensorFloat32, false, false),
    MMAConfig(16, 8, 8, DataType::kTensorFloat32, false, false),
    MMAConfig(8, 8, 16, DataType::kInt8, false, false),
    MMAConfig(16, 8, 16, DataType::kInt8, false, false),
    MMAConfig(16, 8, 32, DataType::kInt8, false, false),
    MMAConfig(8, 8, 16, DataType::kUInt8, false, false),
    MMAConfig(16, 8, 16, DataType::kUInt8, false, false),
    MMAConfig(16, 8, 32, DataType::kUInt8, false, false),
    MMAConfig(8, 8, 32, DataType::kInt4, false, false),
    MMAConfig(16, 8, 32, DataType::kInt4, false, false),
    MMAConfig(16, 8, 64, DataType::kInt4, false, false),
    MMAConfig(8, 8, 32, DataType::kUInt4, false, false),
    MMAConfig(16, 8, 32, DataType::kUInt4, false, false),
    MMAConfig(16, 8, 64, DataType::kUInt4, false, false),
    MMAConfig(8, 8, 128, DataType::kBit1, true, false),
    MMAConfig(16, 8, 128, DataType::kBit1, true, false),
    MMAConfig(16, 8, 256, DataType::kBit1, true, false),
    MMAConfig(16, 8, 16, DataType::kFloat16, false, true),
    MMAConfig(16, 8, 32, DataType::kFloat16, false, true),
    MMAConfig(16, 8, 16, DataType::kBFloat16, false, true),
    MMAConfig(16, 8, 32, DataType::kBFloat16, false, true),
    MMAConfig(16, 8, 8, DataType::kTensorFloat32, false, true),
    MMAConfig(16, 8, 16, DataType::kTensorFloat32, false, true),
    MMAConfig(16, 8, 32, DataType::kInt8, false, true),
    MMAConfig(16, 8, 64, DataType::kInt8, false, true),
    MMAConfig(16, 8, 32, DataType::kUInt8, false, true),
    MMAConfig(16, 8, 64, DataType::kUInt8, false, true),
    MMAConfig(16, 8, 64, DataType::kInt4, false, true),
    MMAConfig(16, 8, 128, DataType::kInt4, false, true),
    MMAConfig(16, 8, 64, DataType::kUInt4, false, true),
    MMAConfig(16, 8, 128, DataType::kUInt4, false, true),
    MMAConfig(16, 8, 32, DataType::kFloat8_e4m3, false, false),
    MMAConfig(16, 8, 64, DataType::kFloat8_e4m3, false, true),
    MMAConfig(16, 8, 32, DataType::kFloat8_e5m2, false, false),
    MMAConfig(16, 8, 64, DataType::kFloat8_e5m2, false, true),
};

struct WGMMAConfig {
  explicit WGMMAConfig(int m, int n, int k, DataType dtype_a, DataType dtype_b,
                       DataType dtype_c, bool sparse)
      : m(m), n(n), k(k), dtype_a(dtype_a), dtype_b(dtype_b), dtype_c(dtype_c),
        sparse(sparse) {}
  int m, n, k;
  DataType dtype_a, dtype_b, dtype_c;
  bool sparse;
  inline bool operator==(const WGMMAConfig &other) {
    return m == other.m && n == other.n && k == other.k &&
           dtype_a == other.dtype_a && dtype_b == other.dtype_b &&
           dtype_c == other.dtype_c && sparse == other.sparse;
  }
};

const WGMMAConfig valid_wgmma_configs[] = {
    // Dense FP16 configurations
    WGMMAConfig(64, 8, 16, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat16, false),
    WGMMAConfig(64, 16, 16, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat16, false),
    WGMMAConfig(64, 32, 16, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat16, false),
    WGMMAConfig(64, 64, 16, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat16, false),
    WGMMAConfig(64, 96, 16, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat16, false),
    WGMMAConfig(64, 128, 16, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat16, false),
    WGMMAConfig(64, 192, 16, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat16, false),
    WGMMAConfig(64, 256, 16, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat16, false),

    // Dense FP16 to FP32 accumulation
    WGMMAConfig(64, 8, 16, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat32, false),
    WGMMAConfig(64, 16, 16, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat32, false),
    WGMMAConfig(64, 32, 16, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat32, false),
    WGMMAConfig(64, 64, 16, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat32, false),
    WGMMAConfig(64, 96, 16, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat32, false),
    WGMMAConfig(64, 128, 16, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat32, false),
    WGMMAConfig(64, 192, 16, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat32, false),
    WGMMAConfig(64, 256, 16, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat32, false),

    // Dense BFloat16 configurations
    WGMMAConfig(64, 8, 16, DataType::kBFloat16, DataType::kBFloat16,
                DataType::kFloat32, false),
    WGMMAConfig(64, 16, 16, DataType::kBFloat16, DataType::kBFloat16,
                DataType::kFloat32, false),
    WGMMAConfig(64, 32, 16, DataType::kBFloat16, DataType::kBFloat16,
                DataType::kFloat32, false),
    WGMMAConfig(64, 64, 16, DataType::kBFloat16, DataType::kBFloat16,
                DataType::kFloat32, false),
    WGMMAConfig(64, 96, 16, DataType::kBFloat16, DataType::kBFloat16,
                DataType::kFloat32, false),
    WGMMAConfig(64, 128, 16, DataType::kBFloat16, DataType::kBFloat16,
                DataType::kFloat32, false),
    WGMMAConfig(64, 192, 16, DataType::kBFloat16, DataType::kBFloat16,
                DataType::kFloat32, false),
    WGMMAConfig(64, 256, 16, DataType::kBFloat16, DataType::kBFloat16,
                DataType::kFloat32, false),

    // Dense TF32 configurations
    WGMMAConfig(64, 8, 8, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, false),
    WGMMAConfig(64, 16, 8, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, false),
    WGMMAConfig(64, 32, 8, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, false),
    WGMMAConfig(64, 64, 8, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, false),
    WGMMAConfig(64, 96, 8, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, false),
    WGMMAConfig(64, 128, 8, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, false),
    WGMMAConfig(64, 192, 8, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, false),
    WGMMAConfig(64, 256, 8, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, false),
    WGMMAConfig(64, 24, 8, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, false),
    WGMMAConfig(64, 40, 8, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, false),

    // Dense INT8 configurations
    WGMMAConfig(64, 8, 32, DataType::kInt8, DataType::kInt8, DataType::kInt32,
                false),
    WGMMAConfig(64, 16, 32, DataType::kInt8, DataType::kInt8, DataType::kInt32,
                false),
    WGMMAConfig(64, 32, 32, DataType::kInt8, DataType::kInt8, DataType::kInt32,
                false),
    WGMMAConfig(64, 64, 32, DataType::kInt8, DataType::kInt8, DataType::kInt32,
                false),
    WGMMAConfig(64, 96, 32, DataType::kInt8, DataType::kInt8, DataType::kInt32,
                false),
    WGMMAConfig(64, 128, 32, DataType::kInt8, DataType::kInt8, DataType::kInt32,
                false),
    WGMMAConfig(64, 192, 32, DataType::kInt8, DataType::kInt8, DataType::kInt32,
                false),
    WGMMAConfig(64, 256, 32, DataType::kInt8, DataType::kInt8, DataType::kInt32,
                false),

    // Dense UINT8 configurations
    WGMMAConfig(64, 8, 32, DataType::kUInt8, DataType::kUInt8, DataType::kInt32,
                false),
    WGMMAConfig(64, 16, 32, DataType::kUInt8, DataType::kUInt8,
                DataType::kInt32, false),
    WGMMAConfig(64, 32, 32, DataType::kUInt8, DataType::kUInt8,
                DataType::kInt32, false),
    WGMMAConfig(64, 64, 32, DataType::kUInt8, DataType::kUInt8,
                DataType::kInt32, false),
    WGMMAConfig(64, 96, 32, DataType::kUInt8, DataType::kUInt8,
                DataType::kInt32, false),
    WGMMAConfig(64, 128, 32, DataType::kUInt8, DataType::kUInt8,
                DataType::kInt32, false),
    WGMMAConfig(64, 192, 32, DataType::kUInt8, DataType::kUInt8,
                DataType::kInt32, false),
    WGMMAConfig(64, 256, 32, DataType::kUInt8, DataType::kUInt8,
                DataType::kInt32, false),

    // Dense INT4 configurations
    WGMMAConfig(64, 8, 64, DataType::kInt4, DataType::kInt4, DataType::kInt32,
                false),
    WGMMAConfig(64, 16, 64, DataType::kInt4, DataType::kInt4, DataType::kInt32,
                false),
    WGMMAConfig(64, 32, 64, DataType::kInt4, DataType::kInt4, DataType::kInt32,
                false),
    WGMMAConfig(64, 64, 64, DataType::kInt4, DataType::kInt4, DataType::kInt32,
                false),
    WGMMAConfig(64, 96, 64, DataType::kInt4, DataType::kInt4, DataType::kInt32,
                false),
    WGMMAConfig(64, 128, 64, DataType::kInt4, DataType::kInt4, DataType::kInt32,
                false),
    WGMMAConfig(64, 192, 64, DataType::kInt4, DataType::kInt4, DataType::kInt32,
                false),
    WGMMAConfig(64, 256, 64, DataType::kInt4, DataType::kInt4, DataType::kInt32,
                false),

    // Dense UINT4 configurations
    WGMMAConfig(64, 8, 64, DataType::kUInt4, DataType::kUInt4, DataType::kInt32,
                false),
    WGMMAConfig(64, 16, 64, DataType::kUInt4, DataType::kUInt4,
                DataType::kInt32, false),
    WGMMAConfig(64, 32, 64, DataType::kUInt4, DataType::kUInt4,
                DataType::kInt32, false),
    WGMMAConfig(64, 64, 64, DataType::kUInt4, DataType::kUInt4,
                DataType::kInt32, false),
    WGMMAConfig(64, 96, 64, DataType::kUInt4, DataType::kUInt4,
                DataType::kInt32, false),
    WGMMAConfig(64, 128, 64, DataType::kUInt4, DataType::kUInt4,
                DataType::kInt32, false),
    WGMMAConfig(64, 192, 64, DataType::kUInt4, DataType::kUInt4,
                DataType::kInt32, false),
    WGMMAConfig(64, 256, 64, DataType::kUInt4, DataType::kUInt4,
                DataType::kInt32, false),

    // Dense FP8 E4M3 configurations
    WGMMAConfig(64, 8, 32, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat16, false),
    WGMMAConfig(64, 16, 32, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat16, false),
    WGMMAConfig(64, 32, 32, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat16, false),
    WGMMAConfig(64, 64, 32, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat16, false),
    WGMMAConfig(64, 96, 32, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat16, false),
    WGMMAConfig(64, 128, 32, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat16, false),
    WGMMAConfig(64, 192, 32, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat16, false),
    WGMMAConfig(64, 256, 32, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat16, false),
    WGMMAConfig(64, 8, 32, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat32, false),
    WGMMAConfig(64, 16, 32, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat32, false),
    WGMMAConfig(64, 32, 32, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat32, false),
    WGMMAConfig(64, 64, 32, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat32, false),
    WGMMAConfig(64, 96, 32, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat32, false),
    WGMMAConfig(64, 128, 32, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat32, false),
    WGMMAConfig(64, 192, 32, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat32, false),
    WGMMAConfig(64, 256, 32, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat32, false),

    // Dense FP8 E5M2 configurations
    WGMMAConfig(64, 8, 32, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat16, false),
    WGMMAConfig(64, 16, 32, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat16, false),
    WGMMAConfig(64, 32, 32, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat16, false),
    WGMMAConfig(64, 64, 32, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat16, false),
    WGMMAConfig(64, 96, 32, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat16, false),
    WGMMAConfig(64, 128, 32, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat16, false),
    WGMMAConfig(64, 192, 32, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat16, false),
    WGMMAConfig(64, 256, 32, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat16, false),
    WGMMAConfig(64, 8, 32, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat32, false),
    WGMMAConfig(64, 16, 32, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat32, false),
    WGMMAConfig(64, 32, 32, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat32, false),
    WGMMAConfig(64, 64, 32, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat32, false),
    WGMMAConfig(64, 96, 32, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat32, false),
    WGMMAConfig(64, 128, 32, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat32, false),
    WGMMAConfig(64, 192, 32, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat32, false),
    WGMMAConfig(64, 256, 32, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat32, false),

    // Sparse FP16 configurations (k doubled for sparsity)
    WGMMAConfig(64, 8, 32, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat16, true),
    WGMMAConfig(64, 16, 32, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat16, true),
    WGMMAConfig(64, 32, 32, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat16, true),
    WGMMAConfig(64, 64, 32, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat16, true),
    WGMMAConfig(64, 96, 32, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat16, true),
    WGMMAConfig(64, 128, 32, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat16, true),
    WGMMAConfig(64, 192, 32, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat16, true),
    WGMMAConfig(64, 256, 32, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat16, true),

    // Sparse FP16 to FP32 accumulation
    WGMMAConfig(64, 8, 32, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat32, true),
    WGMMAConfig(64, 16, 32, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat32, true),
    WGMMAConfig(64, 32, 32, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat32, true),
    WGMMAConfig(64, 64, 32, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat32, true),
    WGMMAConfig(64, 96, 32, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat32, true),
    WGMMAConfig(64, 128, 32, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat32, true),
    WGMMAConfig(64, 192, 32, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat32, true),
    WGMMAConfig(64, 256, 32, DataType::kFloat16, DataType::kFloat16,
                DataType::kFloat32, true),

    // Sparse BFloat16 configurations
    WGMMAConfig(64, 8, 32, DataType::kBFloat16, DataType::kBFloat16,
                DataType::kFloat32, true),
    WGMMAConfig(64, 16, 32, DataType::kBFloat16, DataType::kBFloat16,
                DataType::kFloat32, true),
    WGMMAConfig(64, 32, 32, DataType::kBFloat16, DataType::kBFloat16,
                DataType::kFloat32, true),
    WGMMAConfig(64, 64, 32, DataType::kBFloat16, DataType::kBFloat16,
                DataType::kFloat32, true),
    WGMMAConfig(64, 96, 32, DataType::kBFloat16, DataType::kBFloat16,
                DataType::kFloat32, true),
    WGMMAConfig(64, 128, 32, DataType::kBFloat16, DataType::kBFloat16,
                DataType::kFloat32, true),
    WGMMAConfig(64, 192, 32, DataType::kBFloat16, DataType::kBFloat16,
                DataType::kFloat32, true),
    WGMMAConfig(64, 256, 32, DataType::kBFloat16, DataType::kBFloat16,
                DataType::kFloat32, true),

    // Sparse TF32 configurations
    WGMMAConfig(64, 8, 16, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, true),
    WGMMAConfig(64, 16, 16, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, true),
    WGMMAConfig(64, 32, 16, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, true),
    WGMMAConfig(64, 64, 16, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, true),
    WGMMAConfig(64, 96, 16, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, true),
    WGMMAConfig(64, 128, 16, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, true),
    WGMMAConfig(64, 192, 16, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, true),
    WGMMAConfig(64, 256, 16, DataType::kTensorFloat32, DataType::kTensorFloat32,
                DataType::kFloat32, true),

    // Sparse INT8 configurations
    WGMMAConfig(64, 8, 64, DataType::kInt8, DataType::kInt8, DataType::kInt32,
                true),
    WGMMAConfig(64, 16, 64, DataType::kInt8, DataType::kInt8, DataType::kInt32,
                true),
    WGMMAConfig(64, 32, 64, DataType::kInt8, DataType::kInt8, DataType::kInt32,
                true),
    WGMMAConfig(64, 64, 64, DataType::kInt8, DataType::kInt8, DataType::kInt32,
                true),
    WGMMAConfig(64, 96, 64, DataType::kInt8, DataType::kInt8, DataType::kInt32,
                true),
    WGMMAConfig(64, 128, 64, DataType::kInt8, DataType::kInt8, DataType::kInt32,
                true),
    WGMMAConfig(64, 192, 64, DataType::kInt8, DataType::kInt8, DataType::kInt32,
                true),
    WGMMAConfig(64, 256, 64, DataType::kInt8, DataType::kInt8, DataType::kInt32,
                true),

    // Sparse UINT8 configurations
    WGMMAConfig(64, 8, 64, DataType::kUInt8, DataType::kUInt8, DataType::kInt32,
                true),
    WGMMAConfig(64, 16, 64, DataType::kUInt8, DataType::kUInt8,
                DataType::kInt32, true),
    WGMMAConfig(64, 32, 64, DataType::kUInt8, DataType::kUInt8,
                DataType::kInt32, true),
    WGMMAConfig(64, 64, 64, DataType::kUInt8, DataType::kUInt8,
                DataType::kInt32, true),
    WGMMAConfig(64, 96, 64, DataType::kUInt8, DataType::kUInt8,
                DataType::kInt32, true),
    WGMMAConfig(64, 128, 64, DataType::kUInt8, DataType::kUInt8,
                DataType::kInt32, true),
    WGMMAConfig(64, 192, 64, DataType::kUInt8, DataType::kUInt8,
                DataType::kInt32, true),
    WGMMAConfig(64, 256, 64, DataType::kUInt8, DataType::kUInt8,
                DataType::kInt32, true),

    // Sparse INT4 configurations
    WGMMAConfig(64, 8, 128, DataType::kInt4, DataType::kInt4, DataType::kInt32,
                true),
    WGMMAConfig(64, 16, 128, DataType::kInt4, DataType::kInt4, DataType::kInt32,
                true),
    WGMMAConfig(64, 32, 128, DataType::kInt4, DataType::kInt4, DataType::kInt32,
                true),
    WGMMAConfig(64, 64, 128, DataType::kInt4, DataType::kInt4, DataType::kInt32,
                true),
    WGMMAConfig(64, 96, 128, DataType::kInt4, DataType::kInt4, DataType::kInt32,
                true),
    WGMMAConfig(64, 128, 128, DataType::kInt4, DataType::kInt4,
                DataType::kInt32, true),
    WGMMAConfig(64, 192, 128, DataType::kInt4, DataType::kInt4,
                DataType::kInt32, true),
    WGMMAConfig(64, 256, 128, DataType::kInt4, DataType::kInt4,
                DataType::kInt32, true),

    // Sparse UINT4 configurations
    WGMMAConfig(64, 8, 128, DataType::kUInt4, DataType::kUInt4,
                DataType::kInt32, true),
    WGMMAConfig(64, 16, 128, DataType::kUInt4, DataType::kUInt4,
                DataType::kInt32, true),
    WGMMAConfig(64, 32, 128, DataType::kUInt4, DataType::kUInt4,
                DataType::kInt32, true),
    WGMMAConfig(64, 64, 128, DataType::kUInt4, DataType::kUInt4,
                DataType::kInt32, true),
    WGMMAConfig(64, 96, 128, DataType::kUInt4, DataType::kUInt4,
                DataType::kInt32, true),
    WGMMAConfig(64, 128, 128, DataType::kUInt4, DataType::kUInt4,
                DataType::kInt32, true),
    WGMMAConfig(64, 192, 128, DataType::kUInt4, DataType::kUInt4,
                DataType::kInt32, true),
    WGMMAConfig(64, 256, 128, DataType::kUInt4, DataType::kUInt4,
                DataType::kInt32, true),

    // Sparse FP8 E4M3 configurations
    WGMMAConfig(64, 8, 64, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat16, true),
    WGMMAConfig(64, 16, 64, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat16, true),
    WGMMAConfig(64, 32, 64, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat16, true),
    WGMMAConfig(64, 64, 64, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat16, true),
    WGMMAConfig(64, 96, 64, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat16, true),
    WGMMAConfig(64, 128, 64, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat16, true),
    WGMMAConfig(64, 192, 64, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat16, true),
    WGMMAConfig(64, 256, 64, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat16, true),
    WGMMAConfig(64, 8, 64, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat32, true),
    WGMMAConfig(64, 16, 64, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat32, true),
    WGMMAConfig(64, 32, 64, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat32, true),
    WGMMAConfig(64, 64, 64, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat32, true),
    WGMMAConfig(64, 96, 64, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat32, true),
    WGMMAConfig(64, 128, 64, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat32, true),
    WGMMAConfig(64, 192, 64, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat32, true),
    WGMMAConfig(64, 256, 64, DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                DataType::kFloat32, true),

    // Sparse FP8 E5M2 configurations
    WGMMAConfig(64, 8, 64, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat16, true),
    WGMMAConfig(64, 16, 64, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat16, true),
    WGMMAConfig(64, 32, 64, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat16, true),
    WGMMAConfig(64, 64, 64, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat16, true),
    WGMMAConfig(64, 96, 64, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat16, true),
    WGMMAConfig(64, 128, 64, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat16, true),
    WGMMAConfig(64, 192, 64, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat16, true),
    WGMMAConfig(64, 256, 64, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat16, true),
    WGMMAConfig(64, 8, 64, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat32, true),
    WGMMAConfig(64, 16, 64, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat32, true),
    WGMMAConfig(64, 32, 64, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat32, true),
    WGMMAConfig(64, 64, 64, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat32, true),
    WGMMAConfig(64, 96, 64, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat32, true),
    WGMMAConfig(64, 128, 64, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat32, true),
    WGMMAConfig(64, 192, 64, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat32, true),
    WGMMAConfig(64, 256, 64, DataType::kFloat8_e5m2, DataType::kFloat8_e5m2,
                DataType::kFloat32, true)};

/*!
 * \brief Check whether the multiplicand data type and accumulator data type is
 * valid for MMA computation. \param dtype_a The data type of multiplicand a.
 * \param dtype_b The data type of multiplicand b.
 * \param dtype_c The data type of accumulator c.
 * \note Reference:
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-data-types
 */
void CheckMMADTypeCompatible(DataType dtype_a, DataType dtype_b,
                             DataType dtype_c) {
  std::string ab_not_match_err_str = "The multiplicands' data type " +
                                     DTypeToString(dtype_a) +
                                     DTypeToString(dtype_b) + " do not match.";
  // check a and b
  switch (dtype_a) {
  case DataType::kBit1:
  case DataType::kFloat16:
  case DataType::kBFloat16:
  case DataType::kFloat32:
  case DataType::kTensorFloat32:
  case DataType::kFloat64:
    CHECK(dtype_a == dtype_b) << ab_not_match_err_str;
    break;
  case DataType::kInt4:
  case DataType::kUInt4:
    CHECK(dtype_b == DataType::kInt4 || dtype_b == DataType::kUInt4)
        << ab_not_match_err_str;
    break;
  case DataType::kInt8:
  case DataType::kUInt8:
    CHECK(dtype_b == DataType::kInt8 || dtype_b == DataType::kUInt8)
        << ab_not_match_err_str;
    break;
  case DataType::kFloat8_e4m3:
  case DataType::kFloat8_e5m2:
    CHECK(dtype_b == DataType::kFloat8_e4m3 ||
          dtype_b == DataType::kFloat8_e5m2)
        << ab_not_match_err_str;
    break;
  default:
    CHECK(false) << "Invalid multiplicand data types: "
                 << DTypeToString(dtype_a) << DTypeToString(dtype_b);
  }
  // check a,b and c
  switch (dtype_a) {
  case DataType::kBit1:
  case DataType::kInt4:
  case DataType::kUInt4:
  case DataType::kInt8:
  case DataType::kUInt8:
    CHECK(dtype_c == DataType::kInt32)
        << "For multiplicand data type " << DTypeToString(dtype_a)
        << DTypeToString(dtype_b) << ", accumulator data type should be s32.";
    break;
  case DataType::kFloat16:
    CHECK(dtype_c == DataType::kFloat16 || dtype_c == DataType::kFloat32)
        << "For multiplicand data type f16, accumulator data type should be "
           "f16/f32.";
    break;
  case DataType::kBFloat16:
  case DataType::kFloat32:
  case DataType::kTensorFloat32:
    CHECK(dtype_c == DataType::kFloat32)
        << "For multiplicand data type bf16/tf32, accumulator data type can "
           "only be f32.";
    break;
  case DataType::kFloat64:
    CHECK(dtype_c == DataType::kFloat64)
        << "For multiplicand data type f64, accumulator data type can only be "
           "f64.";
    break;
  case DataType::kFloat8_e4m3:
  case DataType::kFloat8_e5m2:
    CHECK(dtype_c == DataType::kFloat32)
        << "For multiplicand data type e4m3/e5m2, accumulator data type can "
           "only be f32.";
    break;
  default:
    CHECK(false) << "Invalid multiplicand/accumulator data types: "
                 << DTypeToString(dtype_a) << DTypeToString(dtype_b)
                 << DTypeToString(dtype_c) << ".";
  }
}

/*!
 * \brief Check whether the given configuration is valid for MMA computation.
 * \param m The M in mMnNkK of MMA instructions.
 * \param n The N in mMnNkK of MMA instructions.
 * \param k The K in mMnNkK of MMA instructions.
 * \param layout_a The layout of multiplicand A (row/col).
 * \param layout_b The layout of multiplicand B (row/col).
 * \param dtype_a The data type of multiplicand A.
 * \param dtype_b The data type of multiplicand B.
 * \param dtype_c The data type of accumulator C.
 * \param bit_op The bit operator for 1-bit MMA computation, can be "xor"/"and"
 * or ""(if it's not 1-bit MMA). \param sparse Whether it's Sparse MMA or not.
 * \param saturate Whether saturate output or not.
 */
void CheckMMAConfigValidity(int m, int n, int k, LayoutType layout_a,
                            LayoutType layout_b, DataType dtype_a,
                            DataType dtype_b, DataType dtype_c,
                            const std::string &bit_op, bool sparse,
                            bool saturate) {
  CHECK(bit_op == "xor" || bit_op == "and" || bit_op.empty())
      << "Unrecognized 1-bit operation " << bit_op << " , can only be xor/and.";
  bool use_bit_op = !bit_op.empty();
  if (use_bit_op) {
    CHECK(dtype_a == DataType::kBit1)
        << "Bit operator is only compatible with 1-bit multiplicand.";
  }
  CheckMMADTypeCompatible(dtype_a, dtype_b, dtype_c);
  if (saturate) {
    CHECK(dtype_a == DataType::kInt4 || dtype_a == DataType::kUInt4 ||
          dtype_a == DataType::kInt8 || dtype_a == DataType::kUInt8)
        << "Output saturation only applicable to multiplicand type "
           "s4/u4/s8/u8.";
  }

  if (!(m == 8 && n == 8 && k == 4 && dtype_a == ptx::DataType::kFloat16)) {
    // Only MMA on m8n8k4 for fp16 supports customized layouts.
    CHECK(layout_a == LayoutType::kRowMajor &&
          layout_b == LayoutType::kColumnMajor)
        << "Invalid layout combination " << LayoutTypeToString(layout_a) << ","
        << LayoutTypeToString(layout_b) << ".";
  }

  MMAConfig config(m, n, k, dtype_a, use_bit_op, sparse);
  bool match = false;
  for (const MMAConfig &valid_config : valid_mma_configs) {
    if (config == valid_config) {
      match = true;
      break;
    }
  }
  CHECK(match) << "Cannot find matched MMA configurations.";
}

void CheckWGMMAConfigValidity(int m, int n, int k, LayoutType layout_a,
                              LayoutType layout_b, DataType dtype_a,
                              DataType dtype_b, DataType dtype_c, bool sparse) {
  // Same DataType Compatibility as MMA
  CheckMMADTypeCompatible(dtype_a, dtype_b, dtype_c);

  // Check if configuration exists in valid_wgmma_configs
  WGMMAConfig config(m, n, k, dtype_a, dtype_b, dtype_c, sparse);
  bool match = false;
  for (const WGMMAConfig &valid_config : valid_wgmma_configs) {
    if (config == valid_config) {
      match = true;
      break;
    }
  }
  CHECK(match) << "Cannot find matched WGMMA configurations for m " << m
               << " n " << n << " k " << k << " dtype_a "
               << DTypeToString(dtype_a) << " dtype_b "
               << DTypeToString(dtype_b) << " dtype_c "
               << DTypeToString(dtype_c) << " sparse " << sparse;
}
/*!
 * \brief Fragment attributes
 */
class FragAttrs {
public:
  explicit FragAttrs(char reg_type, uint32_t size, std::string ptr_type)
      : reg_type(reg_type), size(size), ptr_type(ptr_type) {}
  /*! \brief PTX register type */
  char reg_type;
  /*! \brief Fragment size */
  uint32_t size;
  /*! \brief Fragment pointer type */
  std::string ptr_type;
};

/*!
 * \brief Fragment attributes of given data type.
 */
inline FragAttrs GetFragAttrs(DataType dtype) {
  switch (dtype) {
  case DataType::kBit1:
  case DataType::kInt4:
  case DataType::kUInt4:
  case DataType::kInt8:
  case DataType::kUInt8:
  case DataType::kFloat8_e4m3:
  case DataType::kFloat8_e5m2:
  case DataType::kBit16:
  case DataType::kFloat16: // .f16x2 register
  case DataType::kBFloat16:
  case DataType::kTensorFloat32:
    return FragAttrs('r', 32, "(unsigned *)");
  case DataType::kInt32:
    return FragAttrs('r', 32, "(int *)");
  case DataType::kFloat32:
    return FragAttrs('f', 32, "(float *)");
  case DataType::kFloat64:
    return FragAttrs('d', 64, "(double *)");
  default:
    ICHECK(false) << DTypeToString(dtype) << " is not matrix data type in MMA.";
    return FragAttrs('\0', 0, "");
  }
}

}; // namespace ptx

/*!
 * \brief Get the number of MMA computations for given shape and datatype.
 */
inline uint32_t GetNumMMAComputations(int m, int n, int k,
                                      ptx::DataType dtype) {
  if (m == 8 && n == 8 && k == 4 && dtype == ptx::DataType::kFloat16) {
    // MMA for m8n8k4 on fp16 would launch 4 MMA computations instead of one.
    return 4;
  } else {
    return 1;
  }
}

/*!
 * \brief Return template string, input operands string and output operands
 * string. \param m The M in mMnNkK of MMA instructions. \param n The N in
 * mMnNkK of MMA instructions. \param k The K in mMnNkK of MMA instructions.
 * \param dtype_a The data type of multiplicand a.
 * \param dtype_b The data type of multiplicand b.
 * \param dtype_c The data type of accumulator c.
 * \param sparse Whether it's Sparse MMA or not.
 */
inline std::tuple<std::string, std::string, std::string>
GetMMAOperands(int m, int n, int k, ptx::DataType dtype_a,
               ptx::DataType dtype_b, ptx::DataType dtype_c, bool sparse) {
  std::stringstream templates, inputs, outputs;
  const ptx::FragAttrs frag_attr_a = ptx::GetFragAttrs(dtype_a),
                       frag_attr_b = ptx::GetFragAttrs(dtype_b),
                       frag_attr_c = ptx::GetFragAttrs(dtype_c);
  constexpr uint32_t warp_size = 32;
  const uint32_t threads = warp_size / GetNumMMAComputations(m, n, k, dtype_a);
  const int num_operands_a = (m * k) * ptx::DTypeBits(dtype_a) /
                             frag_attr_a.size / threads / (sparse ? 2 : 1),
            num_operands_b =
                (k * n) * ptx::DTypeBits(dtype_b) / frag_attr_b.size / threads,
            num_operands_c =
                (m * n) * ptx::DTypeBits(dtype_c) / frag_attr_c.size / threads;

  // generate templates;
  int arg_counter = 0;
  templates << "{"
            << "%" << arg_counter++;
  for (int i = 1; i < num_operands_c; ++i) {
    templates << ", %" << arg_counter++;
  }
  templates << "}, {"
            << "%" << arg_counter++;
  for (int i = 1; i < num_operands_a; ++i) {
    templates << ", %" << arg_counter++;
  }
  templates << "}, {"
            << "%" << arg_counter++;
  for (int i = 1; i < num_operands_b; ++i) {
    templates << ", %" << arg_counter++;
  }
  templates << "}, {"
            << "%" << arg_counter++;
  for (int i = 1; i < num_operands_c; ++i) {
    templates << ", %" << arg_counter++;
  }
  templates << "}";
  // templates of metadata and sparse selector for sparse mma.
  if (sparse) {
    templates << ", %" << (arg_counter++) << ", F";
  }

  // generate inputs
  for (int i = 0; i < num_operands_a; ++i) {
    if (i != 0) {
      inputs << ", ";
    }
    inputs << "\"" << frag_attr_a.reg_type << "\"((" << frag_attr_a.ptr_type
           << "(A))[" << i << "])";
  }
  for (int i = 0; i < num_operands_b; ++i) {
    inputs << ", \"" << frag_attr_b.reg_type << "\"((" << frag_attr_b.ptr_type
           << "(B))[" << i << "])";
  }
  for (int i = 0; i < num_operands_c; ++i) {
    inputs << ", \"" << frag_attr_c.reg_type << "\"((" << frag_attr_c.ptr_type
           << "(C))[" << i << "])";
  }
  // input of metadata for sparse mma.
  if (sparse) {
    inputs << ", \"r\"(((unsigned *)(E))[0])";
  }

  // generate outputs
  for (int i = 0; i < num_operands_c; ++i) {
    if (i != 0) {
      outputs << ",";
    }
    outputs << " \"=" << frag_attr_c.reg_type << "\"((" << frag_attr_c.ptr_type
            << "(D))[" << i << "])";
  }
  return std::make_tuple(templates.str(), inputs.str(), outputs.str());
}

inline std::tuple<std::string, std::string, std::string, std::string>
GetWGMMAOperands(int m, int n, int k, ptx::DataType dtype_a,
                 ptx::DataType dtype_b, ptx::DataType dtype_c, bool sparse,
                 bool a_is_shared) {
  std::stringstream templates, inputs, outputs, predicate;
  const ptx::FragAttrs frag_attr_a = ptx::GetFragAttrs(dtype_a),
                       frag_attr_b = ptx::GetFragAttrs(dtype_b),
                       frag_attr_c = ptx::GetFragAttrs(dtype_c);
  constexpr uint32_t warp_size = 32;
  const uint32_t threads =
      4 * warp_size / GetNumMMAComputations(m, n, k, dtype_a);
  const int num_operands_a = (m * k) * ptx::DTypeBits(dtype_a) /
                             frag_attr_a.size / threads / (sparse ? 2 : 1),
            num_operands_c =
                (m * n) * ptx::DTypeBits(dtype_c) / frag_attr_c.size / threads;
  const bool support_ldmatrix_transposed =
      ptx::DTypeBits(dtype_a) == 16 && ptx::DTypeBits(dtype_b) == 16;
  const bool support_scale_input =
      !ptx::DTypeIsInteger(dtype_a) || !ptx::DTypeIsInteger(dtype_b);

  // generate templates;
  int arg_counter = 0;
  templates << "{"
            << "%" << arg_counter++;
  for (int i = 1; i < num_operands_c; ++i) {
    templates << ", %" << arg_counter++;
  }
  if (!a_is_shared) {
    templates << "}, {"
              << "%" << arg_counter++;
    for (int i = 1; i < num_operands_a; ++i) {
      templates << ", %" << arg_counter++;
    }
    templates << "}";
  } else {
    templates << "}, %" << arg_counter++;
  }

  // desc_b
  templates << ", "
            << "%" << arg_counter++;

  // scale_out
  predicate << "%" << arg_counter++;
  templates << ", "
            << "p";

  // scale_in_a
  if (support_scale_input) {
    templates << ", "
              << "%" << arg_counter++;
    // scale_in_b
    templates << ", "
              << "%" << arg_counter++;
  }
  if (support_ldmatrix_transposed) {
    if (a_is_shared) {
      // trans_a
      templates << ", "
                << "%" << arg_counter++;
    }
    // trans_b
    templates << ", "
              << "%" << arg_counter++;
  }
  // templates of metadata and sparse selector for sparse mma.
  if (sparse) {
    LOG(FATAL) << "Sparse WGMMA is not supported yet.";
  }

  // generate inputs
  if (a_is_shared) {
    inputs << "\"l\"(uint64_t((desc_a) + (A_offset)))";
  } else {
    for (int i = 0; i < num_operands_a; ++i) {
      if (i != 0) {
        inputs << ", ";
      }
      inputs << "\"" << frag_attr_a.reg_type << "\"((" << frag_attr_a.ptr_type
             << "((A)))[" << i << "])";
    }
  }
  inputs << ", \"l\"(uint64_t((desc_b) + (B_offset)))";

  // input of metadata for sparse mma.
  if (sparse) {
    inputs << ", \"r\"(((unsigned *)((E)))[0])";
  }

  inputs << ", \"r\"(int32_t((scale_out)))";
  // scale_in_a
  if (support_scale_input) {
    inputs << ", \"n\"(int32_t((scale_in_a)))";
    // scale_in_b
    inputs << ", \"n\"(int32_t((scale_in_b)))";
  }
  if (support_ldmatrix_transposed) {
    if (a_is_shared) {
      // trans_a
      inputs << ", \"n\"(int32_t((trans_a)))";
    }
    // trans_b
    inputs << ", \"n\"(int32_t((trans_b)))";
  }
  // generate outputs
  for (int i = 0; i < num_operands_c; ++i) {
    if (i != 0) {
      outputs << ",";
    }
    outputs << "\"+" << frag_attr_c.reg_type << "\"((" << frag_attr_c.ptr_type
            << "((D)))[" << i << "])";
  }

  return std::make_tuple(templates.str(), inputs.str(), outputs.str(),
                         predicate.str());
}

std::string
PrintMMAAssembly(const std::string &shape, const std::string &A_layout,
                 const std::string &B_layout, const std::string &A_dtype,
                 const std::string &B_dtype, const std::string &C_dtype,
                 const std::string &a_ptr, const std::string &a_elem_offset,
                 const std::string &b_ptr, const std::string &b_elem_offset,
                 const std::string &c_ptr, const std::string &c_elem_offset,
                 const std::string &metadata,
                 const std::string &metadata_offset,
                 const std::string &sparsity_selector,
                 const std::string &bit_op, bool sparse, bool saturate) {
  ptx::DataType dtype_a = ptx::DTypeFromString(A_dtype),
                dtype_b = ptx::DTypeFromString(B_dtype),
                dtype_c = ptx::DTypeFromString(C_dtype);
  if (dtype_a == ptx::DataType::kFloat32) {
    dtype_a = ptx::DataType::kTensorFloat32;
  }
  if (dtype_b == ptx::DataType::kFloat32) {
    dtype_b = ptx::DataType::kTensorFloat32;
  }
  ptx::LayoutType layout_a = ptx::LayoutTypeFromString(A_layout),
                  layout_b = ptx::LayoutTypeFromString(B_layout);
  auto [m, n, k] = ptx::ParseMMAShape(shape);
  CheckMMAConfigValidity(m, n, k, layout_a, layout_b, dtype_a, dtype_b, dtype_c,
                         bit_op, sparse, saturate);
  std::string asm_code = R"(
  {
    __asm__ __volatile__(
      "mma{.sparse}.sync.aligned{.shape}{.alayout}{.blayout}{.saturate}{.dtype}{.atype}{.btype}{.ctype}{.bitop}"
      "{templates};\n"
      : {outputs}
      : {inputs});
  }
)";
  auto [templates_str, inputs_str, outputs_str] =
      GetMMAOperands(m, n, k, dtype_a, dtype_b, dtype_c, sparse);

  // replace patterns
  Replacer replacer;
  replacer.register_rule("{.sparse}", sparse ? ".sp" : "");
  replacer.register_rule("{.shape}", "." + shape);
  replacer.register_rule("{.saturate}", saturate ? ".satfinite" : "");
  replacer.register_rule("{.alayout}", "." + A_layout);
  replacer.register_rule("{.blayout}", "." + B_layout);
  replacer.register_rule("{.atype}", ptx::DTypeToString(dtype_a));
  replacer.register_rule("{.btype}", ptx::DTypeToString(dtype_b));
  replacer.register_rule("{.ctype}", ptx::DTypeToString(dtype_c));
  replacer.register_rule("{.dtype}", ptx::DTypeToString(dtype_c));
  replacer.register_rule("{.bitop}",
                         bit_op.empty() ? "" : "." + bit_op + ".popc");
  replacer.register_rule("{templates}", templates_str);
  replacer.register_rule("{outputs}", outputs_str);
  replacer.register_rule("{inputs}", inputs_str);
  asm_code = replacer.rewrite(asm_code);
  replacer.empty_rules();
  replacer.register_rule("A", a_ptr + " + " + a_elem_offset);
  replacer.register_rule("B", b_ptr + " + " + b_elem_offset);
  replacer.register_rule("C", c_ptr + " + " + c_elem_offset);
  replacer.register_rule("D", c_ptr + " + " + c_elem_offset);
  replacer.register_rule("E", metadata + " + " + metadata_offset);
  replacer.register_rule("F", sparsity_selector);
  asm_code = replacer.rewrite(asm_code);
  return asm_code;
}

std::string
PrintWGMMAAssembly(const std::string &shape, const bool &a_is_k_major,
                   const bool &b_is_k_major, const std::string &A_dtype,
                   const std::string &B_dtype, const std::string &C_dtype,
                   const std::string &a_desc, const std::string &A_offset,
                   const std::string &b_desc, const std::string &B_offset,
                   const std::string &c_ptr, const std::string &c_offset,
                   const bool &scale_out, const bool &scale_in_a,
                   const bool &scale_in_b, const bool &a_is_shared,
                   const std::string &metadata,
                   const std::string &metadata_offset,
                   const std::string &sparsity_selector, bool sparse) {
  ptx::DataType dtype_a = ptx::DTypeFromString(A_dtype),
                dtype_b = ptx::DTypeFromString(B_dtype),
                dtype_c = ptx::DTypeFromString(C_dtype);
  if (dtype_a == ptx::DataType::kFloat32) {
    dtype_a = ptx::DataType::kTensorFloat32;
  }
  if (dtype_b == ptx::DataType::kFloat32) {
    dtype_b = ptx::DataType::kTensorFloat32;
  }

  ptx::LayoutType layout_a = ptx::LayoutTypeFromBool(!a_is_k_major),
                  layout_b = ptx::LayoutTypeFromBool(b_is_k_major);
  auto [m, n, k] = ptx::ParseMMAShape(shape);
  CheckWGMMAConfigValidity(m, n, k, layout_a, layout_b, dtype_a, dtype_b,
                           dtype_c, sparse);
  std::string asm_code = R"(
  {
    __asm__ __volatile__(
      "{.reg .pred p;\n"
      "setp.ne.b32 p, {predicate}, 0;\n"
      "wgmma.mma_async{.sparse}.sync.aligned{.shape}{.dtype}{.atype}{.btype}"
      "{templates};\n}"
      : {outputs}
      : {inputs});
  }
)";
  auto [templates_str, inputs_str, outputs_str, predicate_str] =
      GetWGMMAOperands(m, n, k, dtype_a, dtype_b, dtype_c, sparse, a_is_shared);

  // replace patterns
  Replacer replacer;
  replacer.register_rule("{.sparse}", sparse ? ".sp" : "");
  replacer.register_rule("{.shape}", "." + shape);
  replacer.register_rule("{.atype}", ptx::DTypeToString(dtype_a));
  replacer.register_rule("{.btype}", ptx::DTypeToString(dtype_b));
  replacer.register_rule("{.dtype}", ptx::DTypeToString(dtype_c));
  replacer.register_rule("{templates}", templates_str);
  replacer.register_rule("{outputs}", outputs_str);
  replacer.register_rule("{inputs}", inputs_str);
  replacer.register_rule("{predicate}", predicate_str);
  asm_code = replacer.rewrite(asm_code);
  replacer.empty_rules();
  if (a_is_shared) {
    replacer.register_rule("(desc_a)", a_desc);
    replacer.register_rule("(A_offset)", A_offset);
  } else {
    replacer.register_rule("(A)", a_desc + " + " + A_offset);
  }
  replacer.register_rule("(desc_b)", b_desc);
  replacer.register_rule("(B_offset)", B_offset);
  replacer.register_rule("(C)", c_ptr + " + " + c_offset);
  replacer.register_rule("(D)", c_ptr + " + " + c_offset);
  replacer.register_rule("(E)", metadata + " + " + metadata_offset);
  replacer.register_rule("(F)", sparsity_selector);
  replacer.register_rule("(scale_out)", scale_out ? "1" : "0");
  replacer.register_rule("(scale_in_a)", scale_in_a ? "1" : "-1");
  replacer.register_rule("(scale_in_b)", scale_in_b ? "1" : "-1");
  replacer.register_rule("(trans_a)", a_is_k_major ? "0" : "1");
  replacer.register_rule("(trans_b)", b_is_k_major ? "0" : "1");
  asm_code = replacer.rewrite(asm_code);
  return asm_code;
}

inline std::tuple<std::string, std::string>
GetLoadMatrixOperands(int num, const std::string &local_ptr,
                      const std::string &local_elem_offset) {
  std::stringstream templates, outputs;
  int arg_counter = 0;
  // generate templates
  templates << "{%" << arg_counter++;
  for (int i = 1; i < num; ++i) {
    templates << ", %" << arg_counter++;
  }
  templates << "}, [%" << arg_counter++ << "]";
  // generate outputs
  std::string ptr_type = "(unsigned *)";
  for (int i = 0; i < num; ++i) {
    if (i != 0) {
      outputs << ", ";
    }
    outputs << "\"=r\"((" << ptr_type << "(" << local_ptr << " + "
            << local_elem_offset << "))[" << i << "])";
  }
  return std::make_tuple(templates.str(), outputs.str());
}

std::string PrintLoadMatrixAssembly(bool trans, int num,
                                    const std::string &type,
                                    const std::string &local_ptr,
                                    const std::string &local_elem_offset,
                                    const std::string &smem_ptr,
                                    const std::string &smem_elem_offset) {
  CHECK(num == 1 || num == 2 || num == 4)
      << "ldmatrix only accept loading 1/2/4 matrices.";
  ptx::DataType data_type = ptx::DTypeFromString(type);
  CHECK(data_type == ptx::DataType::kBit16)
      << "ldmatrix only accept matrix with type .b16.";
  std::string asm_code = R"(
  {
    unsigned int addr = cast_smem_ptr_to_int({smem_addr});
    __asm__ __volatile__(
      "ldmatrix.sync.aligned{.shape}{.num}{.trans}{.ss}{.type}"
      "{templates};\n"
      : {outputs}
      : "r"(addr)
    );
  }
)";
  auto [templates_str, outputs_str] =
      GetLoadMatrixOperands(num, local_ptr, local_elem_offset);

  Replacer replacer;
  replacer.register_rule("{.shape}", ".m8n8");
  replacer.register_rule("{.num}", ".x" + std::to_string(num));
  replacer.register_rule("{.trans}", trans ? ".trans" : "");
  replacer.register_rule("{.ss}", ".shared");
  replacer.register_rule("{.type}", ptx::DTypeToString(data_type));
  replacer.register_rule("{smem_addr}", smem_ptr + " + " + smem_elem_offset);
  replacer.register_rule("{templates}", templates_str);
  replacer.register_rule("{outputs}", outputs_str);
  asm_code = replacer.rewrite(asm_code);
  return asm_code;
}

std::string PrintCpAsyncAssembly(const std::string &shared_ptr,
                                 const std::string &shared_elem_offset,
                                 const std::string &global_ptr,
                                 const std::string &global_elem_offset,
                                 const std::string &bytes) {
  std::string asm_code = R"(
  {
    unsigned int addr = cast_smem_ptr_to_int({smem_addr});
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.{cg_or_ca}.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.{cg_or_ca}.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)({global_ptr})), "n"({bytes})
    );
  }
)";
  Replacer replacer;
  replacer.register_rule("{smem_addr}",
                         shared_ptr + " + " + shared_elem_offset);
  replacer.register_rule("{global_ptr}",
                         global_ptr + " + " + global_elem_offset);
  replacer.register_rule("{bytes}", bytes);
  replacer.register_rule("{cg_or_ca}", bytes == "16" ? "cg" : "ca");
  asm_code = replacer.rewrite(asm_code);
  return asm_code;
}

std::string PrintPredicatedCpAsyncAssembly(
    const std::string &shared_ptr, const std::string &shared_elem_offset,
    const std::string &global_ptr, const std::string &global_elem_offset,
    const std::string &bytes, const std::string &predicate_value) {
  CHECK(bytes == "16" || bytes == "12" || bytes == "8" || bytes == "4" ||
        bytes == "2" || bytes == "1")
      << "Only support 16, 12, 8, 4, 2, 1 bytes for predicated cp.async";
  std::string predicated_asm_code = R"(
  {
    unsigned int addr = cast_smem_ptr_to_int({smem_addr});
    int pred_guard = (int){pred_guard};
    __asm__ __volatile__(
        "{  .reg .pred p;"
        "  setp.ne.b32 p, %0, 0;"
      #if TVM_ENABLE_L2_PREFETCH
        " @p cp.async.{cg_or_ca}.shared.global.L2::128B [%1], [%2], %3;"
      #else
        " @p cp.async.{cg_or_ca}.shared.global [%1], [%2], %3;"
      #endif
      "  @!p {store_shared};}"
        :: "r"(pred_guard), "r"(addr), "l"((void*)({global_ptr})), "n"({bytes}), {nopreg}
    );
  }
)";
  auto [store_shared, nopreg] = [](const std::string &bytes) {
    if (bytes == "16")
      return std::make_tuple("st.shared.v4.u32 [%1], {%4, %5, %6, %7}",
                             "\"r\"(0), \"r\"(0), \"r\"(0),\"r\"(0)");
    else if (bytes == "12")
      return std::make_tuple("st.shared.v3.u32 [%1], {%4, %5, %6}",
                             "\"r\"(0), \"r\"(0), \"r\"(0)");
    else if (bytes == "8")
      return std::make_tuple("st.shared.v2.u32 [%1], {%4, %5}",
                             "\"r\"(0), \"r\"(0)");
    else if (bytes == "4")
      return std::make_tuple("st.shared.u32 [%1], {%4}", "\"r\"(0)");
    else if (bytes == "2")
      return std::make_tuple("st.shared.u16 [%1], {%4}", "\"r\"(0)");
    else if (bytes == "1")
      return std::make_tuple("st.shared.u8 [%1], {%4}", "\"r\"(0)");
    else
      return std::make_tuple("", "");
  }(bytes);

  Replacer replacer;
  replacer.register_rule("{smem_addr}",
                         shared_ptr + " + " + shared_elem_offset);
  replacer.register_rule("{global_ptr}",
                         global_ptr + " + " + global_elem_offset);
  replacer.register_rule("{bytes}", bytes);
  replacer.register_rule("{cg_or_ca}", bytes == "16" ? "cg" : "ca");
  replacer.register_rule("{store_shared}", store_shared);
  replacer.register_rule("{nopreg}", nopreg);
  replacer.register_rule("{pred_guard}", predicate_value);
  predicated_asm_code = replacer.rewrite(predicated_asm_code);
  return predicated_asm_code;
}

std::string PrintCpAsyncBulkAsm(const std::string &shared_ptr,
                                const std::string &shared_elem_offset,
                                const std::string &global_ptr,
                                const std::string &global_elem_offset,
                                const std::string &bytes,
                                const std::string &barrier) {
  std::string asm_code = R"(
  {
    unsigned int smem_addr_int = cast_smem_ptr_to_int({smem_addr});
    unsigned int barrier_addr_int = cast_smem_ptr_to_int({barrier});
    __asm__ __volatile__(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      :: "r"(smem_addr_int), "l"({global_ptr}), "r"({bytes}), "r"(barrier_addr_int)
      : "memory"
    );
  }
)";

  Replacer replacer;
  replacer.register_rule("{smem_addr}",
                         shared_ptr + " + " + shared_elem_offset);
  replacer.register_rule("{global_ptr}",
                         global_ptr + " + " + global_elem_offset);
  replacer.register_rule("{bytes}", bytes);
  replacer.register_rule("{barrier}", "&" + barrier);
  asm_code = replacer.rewrite(asm_code);
  return asm_code;
}

std::string PrintCpAsyncBarrierAsm(const std::string &barrier) {
  std::string predicated_asm_code = R"(
  {
    unsigned int barrier_addr_int = cast_smem_ptr_to_int({barrier});
    __asm__ __volatile__(
      "cp.async.mbarrier.arrive.shared.b64 [%0];"
      :: "r" (barrier_addr_int)
    );
  }
)";

  Replacer replacer;
  replacer.register_rule("{barrier}", "&" + barrier);
  predicated_asm_code = replacer.rewrite(predicated_asm_code);
  return predicated_asm_code;
}

std::string PrintInitBarrierThreadCountAsm(const std::string &barrier,
                                           const std::string &thread_count) {
  std::string predicated_asm_code = R"(
  {
    unsigned int barrier_addr_int = cast_smem_ptr_to_int({barrier});
    int thread_count = {thread_count};
    __asm__ __volatile__(
      "mbarrier.init.shared.b64 [%0], %1;"
      :: "r"(barrier_addr_int), "r"(thread_count)
    );
  }
)";

  Replacer replacer;
  replacer.register_rule("{barrier}", "&" + barrier);
  replacer.register_rule("{thread_count}", thread_count);
  predicated_asm_code = replacer.rewrite(predicated_asm_code);
  return predicated_asm_code;
}

std::string PrintArriveBarrierAsm(const std::string &barrier) {
  std::string predicated_asm_code = R"(
  {
    unsigned int barrier_addr_int = cast_smem_ptr_to_int({barrier});
    __asm__ __volatile__(
      "{ .reg .b64 state; mbarrier.arrive.shared.b64 state, [%0]; }"
      :: "r"(barrier_addr_int)
    );
  }
)";

  Replacer replacer;
  replacer.register_rule("{barrier}", "&" + barrier);
  predicated_asm_code = replacer.rewrite(predicated_asm_code);
  return predicated_asm_code;
}

std::string PrintArriveBarrierExpectTxAsm(const std::string &barrier,
                                          const std::string &byte_count) {
  std::string predicated_asm_code = R"(
  {
    unsigned int barrier_addr_int = cast_smem_ptr_to_int({barrier});
    int byte_count = {byte_count};
    __asm__ __volatile__(
      "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
      :: "r"(barrier_addr_int), "r"(byte_count)
    );
  }
)";

  Replacer replacer;
  replacer.register_rule("{barrier}", "&" + barrier);
  replacer.register_rule("{byte_count}", byte_count);
  predicated_asm_code = replacer.rewrite(predicated_asm_code);
  return predicated_asm_code;
}

std::string PrintWaitBarrierAsm(const std::string &barrier) {
  std::string predicated_asm_code = R"(
  {
    unsigned int barrier_addr_int = cast_smem_ptr_to_int({barrier});
    constexpr int phase_bit = 0;
    __asm__ __volatile__(
      "{ .reg .pred P; WAIT: mbarrier.try_wait.parity.shared.b64 P, [%0], %1; @P bra.uni DONE; bra.uni WAIT; DONE: }"
      :: "r"(barrier_addr_int), "r"(phase_bit)
    );
  }
)";

  Replacer replacer;
  replacer.register_rule("{barrier}", "&" + barrier);
  predicated_asm_code = replacer.rewrite(predicated_asm_code);
  return predicated_asm_code;
}

std::string GetMMARegisterType(const ptx::DataType &dtype) {
  switch (dtype) {
  case ptx::DataType::kInt32:
    return "unsigned";
  case ptx::DataType::kUInt32:
    return "unsigned";
  case ptx::DataType::kFloat32:
    return "float";
  case ptx::DataType::kFloat64:
    return "double";
  default:
    return "unsigned";
  }
}

} // namespace codegen
} // namespace tvm::tl
