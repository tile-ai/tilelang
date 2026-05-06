/*!
 * \file tl/runtime/runtime.h
 * \brief Runtime functions.
 *
 */

#include "runtime.h"

#include "backend/cuda/codegen/stubs/cuda.h"
#include <cstdint>
#include <sstream>
#include <tvm/ffi/function.h>
#include <tvm/node/node.h>
#include <vector>

namespace tvm {
namespace tl {

#if 1
// Thread-local storage for restoring the L2 persisting cache limit
static thread_local size_t __tl_prev_persisting_l2_cache_size = 0;
static thread_local bool __tl_prev_persisting_l2_cache_saved = false;
#endif

#if (CUDA_MAJOR_VERSION >= 12)
template <typename T> static std::string ArrayToStr(const T *ptr, size_t n) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < n; i++) {
    if (i > 0)
      ss << ", ";
    ss << ptr[i]; // NOLINT(clang-analyzer-security.ArrayBound)
  }
  ss << "]";
  return ss.str();
}

static uint64_t PtrModulo(const void *ptr, uint64_t align) {
  return reinterpret_cast<uintptr_t>(ptr) % align;
}

// The vendored CUDA stub may lag the installed toolkit. Keep the numeric
// values from the CUDA Driver API enum order so runtime validation can accept
// descriptors produced for newer drivers without requiring newer headers.
static constexpr int kTensorMapDataType16U4Align8B = 13;
static constexpr int kTensorMapDataType16U4Align16B = 14;
static constexpr int kTensorMapDataType16U6Align16B = 15;

static constexpr int kTensorMapSwizzle128BAtom32B = 4;
static constexpr int kTensorMapSwizzle128BAtom32BFlip8B = 5;
static constexpr int kTensorMapSwizzle128BAtom64B = 6;

static const char *TensorMapDataTypeToString(CUtensorMapDataType type) {
  switch (static_cast<int>(type)) {
  case CU_TENSOR_MAP_DATA_TYPE_UINT8:
    return "CU_TENSOR_MAP_DATA_TYPE_UINT8";
  case CU_TENSOR_MAP_DATA_TYPE_UINT16:
    return "CU_TENSOR_MAP_DATA_TYPE_UINT16";
  case CU_TENSOR_MAP_DATA_TYPE_UINT32:
    return "CU_TENSOR_MAP_DATA_TYPE_UINT32";
  case CU_TENSOR_MAP_DATA_TYPE_INT32:
    return "CU_TENSOR_MAP_DATA_TYPE_INT32";
  case CU_TENSOR_MAP_DATA_TYPE_UINT64:
    return "CU_TENSOR_MAP_DATA_TYPE_UINT64";
  case CU_TENSOR_MAP_DATA_TYPE_INT64:
    return "CU_TENSOR_MAP_DATA_TYPE_INT64";
  case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
    return "CU_TENSOR_MAP_DATA_TYPE_FLOAT16";
  case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
    return "CU_TENSOR_MAP_DATA_TYPE_FLOAT32";
  case CU_TENSOR_MAP_DATA_TYPE_FLOAT64:
    return "CU_TENSOR_MAP_DATA_TYPE_FLOAT64";
  case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
    return "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16";
  case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
    return "CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ";
  case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
    return "CU_TENSOR_MAP_DATA_TYPE_TFLOAT32";
  case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ:
    return "CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ";
  case kTensorMapDataType16U4Align8B:
    return "CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B";
  case kTensorMapDataType16U4Align16B:
    return "CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B";
  case kTensorMapDataType16U6Align16B:
    return "CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B";
  default:
    return "<unknown CUtensorMapDataType>";
  }
}

static const char *
TensorMapInterleaveToString(CUtensorMapInterleave interleave) {
  switch (interleave) {
  case CU_TENSOR_MAP_INTERLEAVE_NONE:
    return "CU_TENSOR_MAP_INTERLEAVE_NONE";
  case CU_TENSOR_MAP_INTERLEAVE_16B:
    return "CU_TENSOR_MAP_INTERLEAVE_16B";
  case CU_TENSOR_MAP_INTERLEAVE_32B:
    return "CU_TENSOR_MAP_INTERLEAVE_32B";
  default:
    return "<unknown CUtensorMapInterleave>";
  }
}

static const char *TensorMapSwizzleToString(CUtensorMapSwizzle swizzle) {
  switch (static_cast<int>(swizzle)) {
  case CU_TENSOR_MAP_SWIZZLE_NONE:
    return "CU_TENSOR_MAP_SWIZZLE_NONE";
  case CU_TENSOR_MAP_SWIZZLE_32B:
    return "CU_TENSOR_MAP_SWIZZLE_32B";
  case CU_TENSOR_MAP_SWIZZLE_64B:
    return "CU_TENSOR_MAP_SWIZZLE_64B";
  case CU_TENSOR_MAP_SWIZZLE_128B:
    return "CU_TENSOR_MAP_SWIZZLE_128B";
  case kTensorMapSwizzle128BAtom32B:
    return "CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B";
  case kTensorMapSwizzle128BAtom32BFlip8B:
    return "CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B";
  case kTensorMapSwizzle128BAtom64B:
    return "CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B";
  default:
    return "<unknown CUtensorMapSwizzle>";
  }
}

static const char *TensorMapL2PromotionToString(CUtensorMapL2promotion l2) {
  switch (l2) {
  case CU_TENSOR_MAP_L2_PROMOTION_NONE:
    return "CU_TENSOR_MAP_L2_PROMOTION_NONE";
  case CU_TENSOR_MAP_L2_PROMOTION_L2_64B:
    return "CU_TENSOR_MAP_L2_PROMOTION_L2_64B";
  case CU_TENSOR_MAP_L2_PROMOTION_L2_128B:
    return "CU_TENSOR_MAP_L2_PROMOTION_L2_128B";
  case CU_TENSOR_MAP_L2_PROMOTION_L2_256B:
    return "CU_TENSOR_MAP_L2_PROMOTION_L2_256B";
  default:
    return "<unknown CUtensorMapL2promotion>";
  }
}

static const char *TensorMapOOBFillToString(CUtensorMapFloatOOBfill oob_fill) {
  switch (oob_fill) {
  case CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE:
    return "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE";
  case CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA:
    return "CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA";
  default:
    return "<unknown CUtensorMapFloatOOBfill>";
  }
}

static uint64_t TensorMapDataTypeBits(CUtensorMapDataType type) {
  switch (static_cast<int>(type)) {
  case CU_TENSOR_MAP_DATA_TYPE_UINT8:
    return 8;
  case CU_TENSOR_MAP_DATA_TYPE_UINT16:
  case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
  case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
    return 16;
  case CU_TENSOR_MAP_DATA_TYPE_UINT32:
  case CU_TENSOR_MAP_DATA_TYPE_INT32:
  case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
  case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
  case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
  case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ:
    return 32;
  case CU_TENSOR_MAP_DATA_TYPE_UINT64:
  case CU_TENSOR_MAP_DATA_TYPE_INT64:
  case CU_TENSOR_MAP_DATA_TYPE_FLOAT64:
    return 64;
  case kTensorMapDataType16U4Align8B:
  case kTensorMapDataType16U4Align16B:
    return 4;
  case kTensorMapDataType16U6Align16B:
    return 6;
  default:
    return 0;
  }
}

static bool IsFloatTensorMapType(CUtensorMapDataType type) {
  switch (type) {
  case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
  case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
  case CU_TENSOR_MAP_DATA_TYPE_FLOAT64:
  case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
  case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
  case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
  case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ:
    return true;
  default:
    return false;
  }
}

static bool IsTensorMapDataType16U4Align8B(CUtensorMapDataType type) {
  return static_cast<int>(type) == kTensorMapDataType16U4Align8B;
}

static bool IsTensorMapDataType16U4Align16B(CUtensorMapDataType type) {
  return static_cast<int>(type) == kTensorMapDataType16U4Align16B;
}

static bool IsTensorMapDataType16U6Align16B(CUtensorMapDataType type) {
  return static_cast<int>(type) == kTensorMapDataType16U6Align16B;
}

static bool IsPackedAlign16TensorMapType(CUtensorMapDataType type) {
  return IsTensorMapDataType16U4Align16B(type) ||
         IsTensorMapDataType16U6Align16B(type);
}

static bool IsTensorMapSwizzle128BFamily(CUtensorMapSwizzle swizzle) {
  switch (static_cast<int>(swizzle)) {
  case CU_TENSOR_MAP_SWIZZLE_128B:
  case kTensorMapSwizzle128BAtom32B:
  case kTensorMapSwizzle128BAtom32BFlip8B:
  case kTensorMapSwizzle128BAtom64B:
    return true;
  default:
    return false;
  }
}

static bool IsSupportedPackedTensorMapSwizzle(CUtensorMapDataType type,
                                              CUtensorMapSwizzle swizzle) {
  int swizzle_value = static_cast<int>(swizzle);
  if (IsTensorMapDataType16U6Align16B(type)) {
    return swizzle_value == CU_TENSOR_MAP_SWIZZLE_NONE ||
           swizzle_value == CU_TENSOR_MAP_SWIZZLE_128B ||
           swizzle_value == kTensorMapSwizzle128BAtom32B ||
           swizzle_value == kTensorMapSwizzle128BAtom64B;
  }
  if (IsTensorMapDataType16U4Align16B(type)) {
    return swizzle_value == CU_TENSOR_MAP_SWIZZLE_NONE ||
           swizzle_value == CU_TENSOR_MAP_SWIZZLE_128B ||
           swizzle_value == kTensorMapSwizzle128BAtom32B;
  }
  return true;
}

static uint64_t
RequiredGlobalAddressAlignment(CUtensorMapDataType type,
                               CUtensorMapInterleave interleave) {
  if (interleave == CU_TENSOR_MAP_INTERLEAVE_32B ||
      IsPackedAlign16TensorMapType(type)) {
    return 32;
  }
  return 16;
}

static uint64_t
RequiredGlobalStrideAlignment(CUtensorMapDataType type,
                              CUtensorMapInterleave interleave) {
  if (interleave == CU_TENSOR_MAP_INTERLEAVE_32B ||
      IsPackedAlign16TensorMapType(type)) {
    return 32;
  }
  return 16;
}

static uint64_t SwizzleSpanBytes(CUtensorMapSwizzle swizzle) {
  if (IsTensorMapSwizzle128BFamily(swizzle)) {
    return 128;
  }
  switch (static_cast<int>(swizzle)) {
  case CU_TENSOR_MAP_SWIZZLE_NONE:
    return 0;
  case CU_TENSOR_MAP_SWIZZLE_32B:
    return 32;
  case CU_TENSOR_MAP_SWIZZLE_64B:
    return 64;
  default:
    return 0;
  }
}

static std::string CudaResultToString(CUresult result) {
  const char *error_name = nullptr;
  const char *error_string = nullptr;
  (void)cuGetErrorName(result, &error_name);
  (void)cuGetErrorString(result, &error_string);

  std::stringstream ss;
  ss << result;
  if (error_name != nullptr) {
    ss << " (" << error_name;
    if (error_string != nullptr) {
      ss << ": " << error_string;
    }
    ss << ")";
  } else if (error_string != nullptr) {
    ss << " (" << error_string << ")";
  }
  return ss.str();
}

static std::string
FormatValidationIssues(const std::vector<std::string> &issues) {
  std::stringstream ss;
  for (size_t i = 0; i < issues.size(); ++i) {
    ss << "  [" << (i + 1) << "] " << issues[i] << '\n';
  }
  return ss.str();
}

struct TensorMapArgs {
  CUtensorMap *map;
  CUtensorMapDataType type;
  cuuint32_t tensorRank;
  void *globalAddress;
  cuuint64_t globalDim[5], globalStride[5];
  cuuint32_t boxDim[5], elementStrides[5];
  CUtensorMapInterleave interleave;
  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2Promotion;
  CUtensorMapFloatOOBfill oobFill;

  static TensorMapArgs Extract(PackedArgs args) {
    TensorMapArgs T;
    int idx = 0;
    ICHECK(args.size() >= 8);
    T.map = reinterpret_cast<CUtensorMap *>(args[idx++].cast<void *>());
    T.type = static_cast<CUtensorMapDataType>(args[idx++].cast<int64_t>());
    T.tensorRank = static_cast<cuuint32_t>(args[idx++].cast<int64_t>());
    T.globalAddress = args[idx++].cast<void *>();
    ICHECK(T.tensorRank >= 1 && T.tensorRank <= 5);
    ICHECK(args.size() == static_cast<int>(8 + T.tensorRank * 4));
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.globalDim[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.globalStride[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.boxDim[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.elementStrides[i] = args[idx++].cast<cuuint64_t>();
    }
    T.interleave =
        static_cast<CUtensorMapInterleave>(args[idx++].cast<int64_t>());
    T.swizzle = static_cast<CUtensorMapSwizzle>(args[idx++].cast<int64_t>());
    T.l2Promotion =
        static_cast<CUtensorMapL2promotion>(args[idx++].cast<int64_t>());
    T.oobFill =
        static_cast<CUtensorMapFloatOOBfill>(args[idx++].cast<int64_t>());
    return T;
  }

  std::string ToDebugString() {
    std::stringstream ss;
    ss << "TMA Desc Addr:   " << map << " (mod64=" << PtrModulo(map, 64)
       << ")\n"
       << "format         " << type << " (" << TensorMapDataTypeToString(type)
       << ")\n"
       << "dim            " << tensorRank << '\n'
       << "gmem_address   " << globalAddress
       << " (mod16=" << PtrModulo(globalAddress, 16)
       << ", mod32=" << PtrModulo(globalAddress, 32) << ")\n"
       << "globalDim      " << ArrayToStr(globalDim, tensorRank) << '\n'
       << "globalStridesRaw " << ArrayToStr(globalStride, tensorRank) << '\n'
       << "cudaGlobalStrides "
       << ArrayToStr(globalStride + 1, tensorRank == 0 ? 0 : tensorRank - 1)
       << '\n'
       << "boxDim         " << ArrayToStr(boxDim, tensorRank) << '\n'
       << "elementStrides " << ArrayToStr(elementStrides, tensorRank) << '\n'
       << "interleave     " << interleave << " ("
       << TensorMapInterleaveToString(interleave) << ")\n"
       << "swizzle        " << swizzle << " ("
       << TensorMapSwizzleToString(swizzle) << ")\n"
       << "l2Promotion    " << l2Promotion << " ("
       << TensorMapL2PromotionToString(l2Promotion) << ")\n"
       << "oobFill        " << oobFill << " ("
       << TensorMapOOBFillToString(oobFill) << ")\n";
    return ss.str();
  }
};

static std::vector<std::string> ValidateTensorMapArgs(const TensorMapArgs &T) {
  std::vector<std::string> issues;
  uint64_t type_bits = TensorMapDataTypeBits(T.type);
  uint64_t addr_align = RequiredGlobalAddressAlignment(T.type, T.interleave);
  uint64_t stride_align = RequiredGlobalStrideAlignment(T.type, T.interleave);

  if (T.map == nullptr) {
    issues.push_back("tensorMap must be non-null");
  } else if (PtrModulo(T.map, 64) != 0) {
    issues.push_back("tensorMap address must be 64-byte aligned, but got " +
                     std::to_string(reinterpret_cast<uintptr_t>(T.map)) +
                     " with mod64=" + std::to_string(PtrModulo(T.map, 64)));
  }

  if (type_bits == 0) {
    issues.push_back(
        "tensorDataType is not a supported CUtensorMapDataType enum: " +
        std::to_string(static_cast<int>(T.type)));
  }

  if (T.tensorRank == 0 || T.tensorRank > 5) {
    issues.push_back("tensorRank must be in [1, 5], but got " +
                     std::to_string(T.tensorRank));
  }
  if (T.interleave != CU_TENSOR_MAP_INTERLEAVE_NONE && T.tensorRank < 3) {
    issues.push_back("tensorRank must be >= 3 when interleave is not NONE");
  }

  if (T.globalAddress == nullptr) {
    issues.push_back("globalAddress must be non-null");
  } else if (PtrModulo(T.globalAddress, addr_align) != 0) {
    issues.push_back("globalAddress must be " + std::to_string(addr_align) +
                     "-byte aligned, but mod" + std::to_string(addr_align) +
                     "=" +
                     std::to_string(PtrModulo(T.globalAddress, addr_align)));
  }

  for (size_t i = 0; i < T.tensorRank; ++i) {
    if (T.globalDim[i] == 0) {
      issues.push_back("globalDim[" + std::to_string(i) + "] must be non-zero");
    }
    if (T.globalDim[i] > (uint64_t{1} << 32)) {
      issues.push_back("globalDim[" + std::to_string(i) +
                       "] must be <= 2^32, but got " +
                       std::to_string(T.globalDim[i]));
    }
  }
  if (T.tensorRank > 0) {
    if (IsPackedAlign16TensorMapType(T.type) && T.globalDim[0] % 128 != 0) {
      issues.push_back("globalDim[0] must be a multiple of 128 for " +
                       std::string(TensorMapDataTypeToString(T.type)) +
                       ", but got " + std::to_string(T.globalDim[0]));
    }
    if (IsTensorMapDataType16U4Align8B(T.type) && T.globalDim[0] % 2 != 0) {
      issues.push_back("globalDim[0] must be a multiple of 2 for " +
                       std::string(TensorMapDataTypeToString(T.type)) +
                       ", but got " + std::to_string(T.globalDim[0]));
    }
  }

  for (size_t raw_i = 1; raw_i < T.tensorRank; ++raw_i) {
    cuuint64_t stride = T.globalStride[raw_i];
    size_t cuda_i = raw_i - 1;
    if (stride == 0) {
      issues.push_back("effective cuda globalStrides[" +
                       std::to_string(cuda_i) + "] (raw globalStride[" +
                       std::to_string(raw_i) + "]) must be non-zero");
    }
    if (stride % stride_align != 0) {
      issues.push_back("effective cuda globalStrides[" +
                       std::to_string(cuda_i) + "] (raw globalStride[" +
                       std::to_string(raw_i) + "] = " + std::to_string(stride) +
                       ") must be a multiple of " +
                       std::to_string(stride_align) + " bytes");
    }
    if (stride >= (uint64_t{1} << 40)) {
      issues.push_back("effective cuda globalStrides[" +
                       std::to_string(cuda_i) + "] (raw globalStride[" +
                       std::to_string(raw_i) + "] = " + std::to_string(stride) +
                       ") must be < 2^40");
    }
  }

  for (size_t i = 0; i < T.tensorRank; ++i) {
    if (T.boxDim[i] == 0) {
      issues.push_back("boxDim[" + std::to_string(i) + "] must be non-zero");
    }
    if (T.boxDim[i] > 256) {
      issues.push_back("boxDim[" + std::to_string(i) +
                       "] must be <= 256, but got " +
                       std::to_string(T.boxDim[i]));
    }
  }
  if (T.tensorRank > 0 && IsPackedAlign16TensorMapType(T.type) &&
      T.boxDim[0] != 128) {
    issues.push_back("boxDim[0] must be 128 for " +
                     std::string(TensorMapDataTypeToString(T.type)) +
                     ", but got " + std::to_string(T.boxDim[0]));
  }

  if (T.tensorRank > 0 && T.interleave == CU_TENSOR_MAP_INTERLEAVE_NONE &&
      type_bits != 0 && ((uint64_t{T.boxDim[0]} * type_bits) % 128 != 0)) {
    issues.push_back("boxDim[0] * elementSize must be a multiple of 16 bytes "
                     "when interleave is NONE, but got " +
                     std::to_string(T.boxDim[0]) + " * " +
                     std::to_string(type_bits) + " bits");
  }

  for (size_t i = 0; i < T.tensorRank; ++i) {
    if (T.elementStrides[i] == 0 || T.elementStrides[i] > 8) {
      issues.push_back("elementStrides[" + std::to_string(i) +
                       "] must be in [1, 8], but got " +
                       std::to_string(T.elementStrides[i]));
    }
  }

  if (T.interleave == CU_TENSOR_MAP_INTERLEAVE_32B &&
      T.swizzle != CU_TENSOR_MAP_SWIZZLE_32B) {
    issues.push_back("swizzle must be CU_TENSOR_MAP_SWIZZLE_32B when "
                     "interleave is CU_TENSOR_MAP_INTERLEAVE_32B");
  }
  if (IsTensorMapDataType16U6Align16B(T.type) &&
      T.interleave != CU_TENSOR_MAP_INTERLEAVE_NONE) {
    issues.push_back("interleave must be CU_TENSOR_MAP_INTERLEAVE_NONE for " +
                     std::string(TensorMapDataTypeToString(T.type)));
  }
  if (!IsSupportedPackedTensorMapSwizzle(T.type, T.swizzle)) {
    issues.push_back(std::string(TensorMapSwizzleToString(T.swizzle)) +
                     " is not supported for " +
                     TensorMapDataTypeToString(T.type));
  }

  uint64_t swizzle_bytes = SwizzleSpanBytes(T.swizzle);
  if (T.tensorRank > 0 && T.interleave == CU_TENSOR_MAP_INTERLEAVE_NONE &&
      swizzle_bytes != 0 && type_bits != 0 &&
      (uint64_t{T.boxDim[0]} * type_bits > swizzle_bytes * 8)) {
    issues.push_back("boxDim[0] * elementSize must be <= swizzle span (" +
                     std::to_string(swizzle_bytes) +
                     " bytes) when swizzle is enabled, but got " +
                     std::to_string(T.boxDim[0]) + " * " +
                     std::to_string(type_bits) + " bits");
  }

  if (T.oobFill == CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA &&
      !IsFloatTensorMapType(T.type)) {
    issues.push_back("CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA can "
                     "only be used with floating-point tensorDataType");
  }

  return issues;
}

// set device api
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Register using the canonical names defined in runtime.h
  refl::GlobalDef().def_packed(
      tl::tvm_tensormap_create_tiled, [](PackedArgs args, Any *ret) {
        TensorMapArgs T = TensorMapArgs::Extract(args);
        std::vector<std::string> issues = ValidateTensorMapArgs(T);
        if (!issues.empty()) {
          LOG_FATAL << "Invalid TMA descriptor arguments for "
                    << tl::tvm_tensormap_create_tiled << ":\n"
                    << FormatValidationIssues(issues) << T.ToDebugString();
        }
        CUresult result = cuTensorMapEncodeTiled(
            T.map, T.type, T.tensorRank, T.globalAddress, T.globalDim,
            T.globalStride + 1, T.boxDim, T.elementStrides, T.interleave,
            T.swizzle, T.l2Promotion, T.oobFill);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to initialize the TMA descriptor "
                    << CudaResultToString(result) << '\n'
                    << "No local tiled-TMA constraint violation was detected "
                       "before calling cuTensorMapEncodeTiled.\n"
                    << T.ToDebugString();
        }
        *ret = static_cast<int>(result);
      });
}

struct TensorMapIm2ColArgs {
  CUtensorMap *map;
  CUtensorMapDataType type;
  cuuint32_t tensorRank;
  void *globalAddress;
  cuuint64_t globalDim[5], globalStride[5];
  cuuint32_t elementStrides[5];
  int pixelBoxLowerCorner[3], pixelBoxUpperCorner[3];
  cuuint32_t smem_box_channel, smem_box_pixel;
  CUtensorMapInterleave interleave;
  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2Promotion;
  CUtensorMapFloatOOBfill oobFill;

  static TensorMapIm2ColArgs Extract(PackedArgs args) {
    TensorMapIm2ColArgs T;
    int idx = 0;
    ICHECK(args.size() >= 8);
    T.map = reinterpret_cast<CUtensorMap *>(args[idx++].cast<void *>());
    T.type = static_cast<CUtensorMapDataType>(args[idx++].cast<int64_t>());
    T.tensorRank = static_cast<cuuint32_t>(args[idx++].cast<int64_t>());
    T.globalAddress = args[idx++].cast<void *>();
    ICHECK(T.tensorRank >= 3 && T.tensorRank <= 5);
    ICHECK(args.size() == static_cast<int>(6 + T.tensorRank * 5));
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.globalDim[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.globalStride[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.elementStrides[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank - 2; i++) {
      T.pixelBoxLowerCorner[i] = args[idx++].cast<int>();
    }
    for (size_t i = 0; i < T.tensorRank - 2; i++) {
      T.pixelBoxUpperCorner[i] = args[idx++].cast<int>();
    }
    T.smem_box_pixel = args[idx++].cast<cuuint64_t>();
    T.smem_box_channel = args[idx++].cast<cuuint64_t>();
    T.interleave =
        static_cast<CUtensorMapInterleave>(args[idx++].cast<int64_t>());
    T.swizzle = static_cast<CUtensorMapSwizzle>(args[idx++].cast<int64_t>());
    T.l2Promotion =
        static_cast<CUtensorMapL2promotion>(args[idx++].cast<int64_t>());
    T.oobFill =
        static_cast<CUtensorMapFloatOOBfill>(args[idx++].cast<int64_t>());
    return T;
  }

  std::string ToDebugString() {
    std::stringstream ss;
    ss << "TMA Desc Addr:   " << map << " (mod64=" << PtrModulo(map, 64)
       << ")\n"
       << "format         " << type << " (" << TensorMapDataTypeToString(type)
       << ")\n"
       << "dim            " << tensorRank << '\n'
       << "gmem_address   " << globalAddress
       << " (mod16=" << PtrModulo(globalAddress, 16)
       << ", mod32=" << PtrModulo(globalAddress, 32) << ")\n"
       << "globalDim      " << ArrayToStr(globalDim, tensorRank) << '\n'
       << "globalStridesRaw " << ArrayToStr(globalStride, tensorRank) << '\n'
       << "cudaGlobalStrides "
       << ArrayToStr(globalStride + 1, tensorRank == 0 ? 0 : tensorRank - 1)
       << '\n'
       << "smem_box_pixel " << smem_box_pixel << '\n'
       << "smem_box_channel " << smem_box_channel << '\n'
       << "pixelBoxLowerCorner  "
       << ArrayToStr(pixelBoxLowerCorner, tensorRank - 2) << '\n'
       << "pixelBoxUpperCorner  "
       << ArrayToStr(pixelBoxUpperCorner, tensorRank - 2) << '\n'
       << "elementStrides " << ArrayToStr(elementStrides, tensorRank) << '\n'
       << "interleave     " << interleave << " ("
       << TensorMapInterleaveToString(interleave) << ")\n"
       << "swizzle        " << swizzle << " ("
       << TensorMapSwizzleToString(swizzle) << ")\n"
       << "l2Promotion    " << l2Promotion << " ("
       << TensorMapL2PromotionToString(l2Promotion) << ")\n"
       << "oobFill        " << oobFill << " ("
       << TensorMapOOBFillToString(oobFill) << ")\n";
    return ss.str();
  }
};

static std::vector<std::string>
ValidateTensorMapIm2ColArgs(const TensorMapIm2ColArgs &T) {
  std::vector<std::string> issues;
  uint64_t type_bits = TensorMapDataTypeBits(T.type);
  uint64_t addr_align = RequiredGlobalAddressAlignment(T.type, T.interleave);
  uint64_t stride_align = RequiredGlobalStrideAlignment(T.type, T.interleave);

  if (T.map == nullptr) {
    issues.push_back("tensorMap must be non-null");
  } else if (PtrModulo(T.map, 64) != 0) {
    issues.push_back("tensorMap address must be 64-byte aligned, but mod64=" +
                     std::to_string(PtrModulo(T.map, 64)));
  }

  if (type_bits == 0) {
    issues.push_back(
        "tensorDataType is not a supported CUtensorMapDataType enum: " +
        std::to_string(static_cast<int>(T.type)));
  }

  if (T.tensorRank < 3 || T.tensorRank > 5) {
    issues.push_back("tensorRank must be in [3, 5] for im2col, but got " +
                     std::to_string(T.tensorRank));
  }

  if (T.globalAddress == nullptr) {
    issues.push_back("globalAddress must be non-null");
  } else if (PtrModulo(T.globalAddress, addr_align) != 0) {
    issues.push_back("globalAddress must be " + std::to_string(addr_align) +
                     "-byte aligned, but mod" + std::to_string(addr_align) +
                     "=" +
                     std::to_string(PtrModulo(T.globalAddress, addr_align)));
  }

  for (size_t i = 0; i < T.tensorRank; ++i) {
    if (T.globalDim[i] == 0) {
      issues.push_back("globalDim[" + std::to_string(i) + "] must be non-zero");
    }
    if (T.globalDim[i] > (uint64_t{1} << 32)) {
      issues.push_back("globalDim[" + std::to_string(i) +
                       "] must be <= 2^32, but got " +
                       std::to_string(T.globalDim[i]));
    }
    if (T.elementStrides[i] == 0 || T.elementStrides[i] > 8) {
      issues.push_back("elementStrides[" + std::to_string(i) +
                       "] must be in [1, 8], but got " +
                       std::to_string(T.elementStrides[i]));
    }
  }
  if (T.tensorRank > 0) {
    if (IsPackedAlign16TensorMapType(T.type) && T.globalDim[0] % 128 != 0) {
      issues.push_back("globalDim[0] must be a multiple of 128 for " +
                       std::string(TensorMapDataTypeToString(T.type)) +
                       ", but got " + std::to_string(T.globalDim[0]));
    }
    if (IsTensorMapDataType16U4Align8B(T.type) && T.globalDim[0] % 2 != 0) {
      issues.push_back("globalDim[0] must be a multiple of 2 for " +
                       std::string(TensorMapDataTypeToString(T.type)) +
                       ", but got " + std::to_string(T.globalDim[0]));
    }
  }

  for (size_t raw_i = 1; raw_i < T.tensorRank; ++raw_i) {
    cuuint64_t stride = T.globalStride[raw_i];
    size_t cuda_i = raw_i - 1;
    if (stride == 0) {
      issues.push_back("effective cuda globalStrides[" +
                       std::to_string(cuda_i) + "] (raw globalStride[" +
                       std::to_string(raw_i) + "]) must be non-zero");
    }
    if (stride % stride_align != 0) {
      issues.push_back("effective cuda globalStrides[" +
                       std::to_string(cuda_i) + "] (raw globalStride[" +
                       std::to_string(raw_i) + "] = " + std::to_string(stride) +
                       ") must be a multiple of " +
                       std::to_string(stride_align) + " bytes");
    }
    if (stride >= (uint64_t{1} << 40)) {
      issues.push_back("effective cuda globalStrides[" +
                       std::to_string(cuda_i) + "] (raw globalStride[" +
                       std::to_string(raw_i) + "] = " + std::to_string(stride) +
                       ") must be < 2^40");
    }
  }

  if (T.smem_box_channel == 0 || T.smem_box_channel > 256) {
    issues.push_back("channelsPerPixel must be in [1, 256], but got " +
                     std::to_string(T.smem_box_channel));
  }
  if (IsPackedAlign16TensorMapType(T.type) && T.smem_box_channel != 128) {
    issues.push_back("channelsPerPixel must be 128 for " +
                     std::string(TensorMapDataTypeToString(T.type)) +
                     ", but got " + std::to_string(T.smem_box_channel));
  }
  if (T.smem_box_pixel == 0 || T.smem_box_pixel > 1024) {
    issues.push_back("pixelsPerColumn must be in [1, 1024], but got " +
                     std::to_string(T.smem_box_pixel));
  }

  if (T.interleave == CU_TENSOR_MAP_INTERLEAVE_32B &&
      T.swizzle != CU_TENSOR_MAP_SWIZZLE_32B) {
    issues.push_back("swizzle must be CU_TENSOR_MAP_SWIZZLE_32B when "
                     "interleave is CU_TENSOR_MAP_INTERLEAVE_32B");
  }
  if (IsTensorMapDataType16U6Align16B(T.type) &&
      T.interleave != CU_TENSOR_MAP_INTERLEAVE_NONE) {
    issues.push_back("interleave must be CU_TENSOR_MAP_INTERLEAVE_NONE for " +
                     std::string(TensorMapDataTypeToString(T.type)));
  }
  if (!IsSupportedPackedTensorMapSwizzle(T.type, T.swizzle)) {
    issues.push_back(std::string(TensorMapSwizzleToString(T.swizzle)) +
                     " is not supported for " +
                     TensorMapDataTypeToString(T.type));
  }

  if (T.oobFill == CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA &&
      !IsFloatTensorMapType(T.type)) {
    issues.push_back("CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA can "
                     "only be used with floating-point tensorDataType");
  }

  return issues;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      tl::tvm_tensormap_create_im2col, [](PackedArgs args, Any *ret) {
        TensorMapIm2ColArgs T = TensorMapIm2ColArgs::Extract(args);
        std::vector<std::string> issues = ValidateTensorMapIm2ColArgs(T);
        if (!issues.empty()) {
          LOG_FATAL << "Invalid TMA im2col descriptor arguments for "
                    << tl::tvm_tensormap_create_im2col << ":\n"
                    << FormatValidationIssues(issues) << T.ToDebugString();
        }
        CUresult result = cuTensorMapEncodeIm2col(
            T.map, T.type, T.tensorRank, T.globalAddress, T.globalDim,
            T.globalStride + 1, T.pixelBoxLowerCorner, T.pixelBoxUpperCorner,
            T.smem_box_channel, T.smem_box_pixel, T.elementStrides,
            T.interleave, T.swizzle, T.l2Promotion, T.oobFill);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to initialize the TMA descriptor "
                    << CudaResultToString(result) << '\n'
                    << "No local im2col-TMA constraint violation was detected "
                       "before calling cuTensorMapEncodeIm2col.\n"
                    << T.ToDebugString();
        }
        *ret = static_cast<int>(result);
      });
}

#endif // (CUDA_MAJOR_VERSION >= 12)

//
// CUDA L2 Persisting Cache Access Policy Window helpers.
// Exposed as TVM FFI packed functions similar to TMA initialization.
//
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Set stream access policy window and adjust persisting L2 cache size
  // Args:
  //  [0]: void* base_ptr (required)
  //  [1]: int64 num_bytes (required)
  //  [2]: float hit_ratio (optional, default 0.8)
  //  [3]: void* stream (optional, default 0 => default stream)
  //  [4]: int64 l2_limit_bytes (optional, default = num_bytes)
  refl::GlobalDef().def_packed(
      tl::tvm_cuda_stream_set_access_policy_window,
      [](PackedArgs args, Any *ret) {
        ICHECK(args.size() >= 2) << "Expected at least base_ptr and num_bytes";

        void *base_ptr = args[0].cast<void *>();
        size_t num_bytes = static_cast<size_t>(args[1].cast<int64_t>());
        float hit_ratio = 0.8f;
        if (args.size() >= 3) {
          // Accept double/float
          hit_ratio = static_cast<float>(args[2].cast<double>());
        }
        CUstream stream = nullptr;
        if (args.size() >= 4) {
          stream = reinterpret_cast<CUstream>(args[3].cast<void *>());
        }
        size_t l2_limit_bytes = num_bytes;
        if (args.size() >= 5) {
          l2_limit_bytes = static_cast<size_t>(args[4].cast<int64_t>());
        }

        // Clamp requested limit to device capability
        CUdevice device;
        CUresult result = cuCtxGetDevice(&device);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to get current CUDA device: " << result;
        }
        int max_persisting = 0;
        result = cuDeviceGetAttribute(
            &max_persisting, CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
            device);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to query MAX_PERSISTING_L2_CACHE_SIZE: "
                    << result;
        }
        if (max_persisting > 0 &&
            l2_limit_bytes > static_cast<size_t>(max_persisting)) {
          l2_limit_bytes = static_cast<size_t>(max_persisting);
        }

        // Save current limit to restore later
        size_t init_persisting_l2_cache_size = 0;
        result = cuCtxGetLimit(&init_persisting_l2_cache_size,
                               CU_LIMIT_PERSISTING_L2_CACHE_SIZE);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to get current persisting L2 cache size limit: "
                    << result;
        }
        __tl_prev_persisting_l2_cache_size = init_persisting_l2_cache_size;
        __tl_prev_persisting_l2_cache_saved = true;

        // Set new limit
        result =
            cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, l2_limit_bytes);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to set persisting L2 cache size limit: "
                    << result;
        }

        // Apply access policy window to stream
        CUstreamAttrValue stream_attribute;
        memset(&stream_attribute, 0, sizeof(stream_attribute));
        stream_attribute.accessPolicyWindow.base_ptr = base_ptr;
        stream_attribute.accessPolicyWindow.num_bytes = l2_limit_bytes;
        stream_attribute.accessPolicyWindow.hitRatio = hit_ratio;
        stream_attribute.accessPolicyWindow.hitProp =
            CU_ACCESS_PROPERTY_PERSISTING;
        stream_attribute.accessPolicyWindow.missProp =
            CU_ACCESS_PROPERTY_STREAMING;

        result = cuStreamSetAttribute(stream,
                                      CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                                      &stream_attribute);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to set stream access policy window: " << result;
        }

        *ret = static_cast<int>(result);
      });

  // Reset stream access policy window and restore the previous L2 cache size
  // Args:
  //  [0]: void* stream (optional, default 0)
  refl::GlobalDef().def_packed(
      tl::tvm_cuda_stream_reset_access_policy_window,
      [](PackedArgs args, Any *ret) {
        CUstream stream = nullptr;
        if (args.size() >= 1) {
          stream = reinterpret_cast<CUstream>(args[0].cast<void *>());
        }

        CUstreamAttrValue stream_attribute;
        memset(&stream_attribute, 0, sizeof(stream_attribute));
        // num_bytes = 0 disables the access policy window on the stream
        stream_attribute.accessPolicyWindow.num_bytes = 0;

        CUresult result = cuStreamSetAttribute(
            stream, CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
            &stream_attribute);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to reset stream access policy window: "
                    << result;
        }

        result = cuCtxResetPersistingL2Cache();
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to reset persisting L2 cache lines: " << result;
        }

        if (__tl_prev_persisting_l2_cache_saved) {
          result = cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE,
                                 __tl_prev_persisting_l2_cache_size);
          if (result != CUDA_SUCCESS) {
            LOG_FATAL << "Failed to restore persisting L2 cache size limit: "
                      << result;
          }
          __tl_prev_persisting_l2_cache_saved = false;
        }

        *ret = static_cast<int>(result);
      });
}

} // namespace tl
} // namespace tvm
