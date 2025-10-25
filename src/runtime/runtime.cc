/*!
 * \file tl/runtime/runtime.h
 * \brief Runtime functions.
 *
 */

#include "runtime.h"

#include "../target/cuda.h"
#include <tvm/ffi/function.h>
#include <tvm/node/node.h>

namespace tvm {
namespace tl {

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
    ss << "TMA Desc Addr:   " << map << '\n'
       << "format         " << type << '\n'
       << "dim            " << tensorRank << '\n'
       << "gmem_address   " << globalAddress << '\n'
       << "globalDim      " << ArrayToStr(globalDim, tensorRank) << '\n'
       << "globalStrides  " << ArrayToStr(globalStride, tensorRank) << '\n'
       << "boxDim         " << ArrayToStr(boxDim, tensorRank) << '\n'
       << "elementStrides " << ArrayToStr(elementStrides, tensorRank) << '\n'
       << "interleave     " << interleave << '\n'
       << "swizzle        " << swizzle << '\n'
       << "l2Promotion    " << l2Promotion << '\n'
       << "oobFill        " << oobFill << '\n';
    return ss.str();
  }
};

// set device api
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("tvm_tensormap_create_tiled", [](PackedArgs args,
                                                                Any *ret) {
    TensorMapArgs T = TensorMapArgs::Extract(args);
    CUresult result = cuTensorMapEncodeTiled(
        T.map, T.type, T.tensorRank, T.globalAddress, T.globalDim,
        T.globalStride + 1, T.boxDim, T.elementStrides, T.interleave, T.swizzle,
        T.l2Promotion, T.oobFill);
    if (result != CUDA_SUCCESS) {
      LOG_FATAL << "Failed to initialize the TMA descriptor " << result << '\n'
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
    ss << "TMA Desc Addr:   " << map << '\n'
       << "format         " << type << '\n'
       << "dim            " << tensorRank << '\n'
       << "gmem_address   " << globalAddress << '\n'
       << "globalDim      " << ArrayToStr(globalDim, tensorRank) << '\n'
       << "globalStrides  " << ArrayToStr(globalStride, tensorRank) << '\n'
       << "smem_box_pixel " << smem_box_pixel << '\n'
       << "smem_box_channel " << smem_box_channel << '\n'
       << "pixelBoxLowerCorner  "
       << ArrayToStr(pixelBoxLowerCorner, tensorRank - 2) << '\n'
       << "pixelBoxUpperCorner  "
       << ArrayToStr(pixelBoxUpperCorner, tensorRank - 2) << '\n'
       << "elementStrides " << ArrayToStr(elementStrides, tensorRank) << '\n'
       << "interleave     " << interleave << '\n'
       << "swizzle        " << swizzle << '\n'
       << "l2Promotion    " << l2Promotion << '\n'
       << "oobFill        " << oobFill << '\n';
    return ss.str();
  }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      "tvm_tensormap_create_im2col", [](PackedArgs args, Any *ret) {
        TensorMapIm2ColArgs T = TensorMapIm2ColArgs::Extract(args);
        CUresult result = cuTensorMapEncodeIm2col(
            T.map, T.type, T.tensorRank, T.globalAddress, T.globalDim,
            T.globalStride + 1, T.pixelBoxLowerCorner, T.pixelBoxUpperCorner,
            T.smem_box_channel, T.smem_box_pixel, T.elementStrides,
            T.interleave, T.swizzle, T.l2Promotion, T.oobFill);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to initialize the TMA descriptor " << result
                    << '\n'
                    << T.ToDebugString();
        }
        *ret = static_cast<int>(result);
      });
}

#endif // (CUDA_MAJOR_VERSION >= 12)

} // namespace tl
} // namespace tvm
