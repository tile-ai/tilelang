/**
 * \file cudart.cc
 * \brief CUDA Runtime API stub library for lazy loading libcudart.so at
 * runtime.
 *
 * Motivation
 * ----------
 * The primary purpose is to resolve SONAME mismatches (e.g., libcudart.so.11.0
 * vs libcudart.so.12), allowing a single build to work across different CUDA
 * versions. This is achieved by reusing the CUDA runtime already loaded by
 * frameworks like PyTorch.
 *
 * This stub exports the subset of CUDA Runtime API entrypoints used by TVM in
 * this repository. The real libcudart is loaded lazily via dlopen() on first
 * API call, and symbols are resolved via dlsym().
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda_runtime_api.h>

#if defined(_WIN32) && !defined(__CYGWIN__)
#error "cudart_stub is currently POSIX-only (requires <dlfcn.h> / dlopen). "       \
    "On Windows, build TileLang from source with -DTILELANG_USE_CUDA_STUBS=OFF " \
    "to link against the real CUDA libraries."
#endif

#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// This stub supports CUDA 11+.
//
// Note: CUDA 12 changed the `cudaGraphInstantiate` entrypoint signature in
// `cuda_runtime_api.h` from a legacy 5-parameter form to a 3-parameter form
// (with flags). To keep wheels usable across CUDA majors, this stub resolves
// and dispatches to `cudaGraphInstantiateWithFlags` when available
// (CUDA 11.4+), otherwise falls back to the legacy `cudaGraphInstantiate`
// symbol.
#ifndef CUDART_VERSION
#error                                                                         \
    "CUDART_VERSION is not defined. Ensure CUDA runtime headers are available."
#endif
static_assert(CUDART_VERSION >= 11000,
              "cudart_stub requires CUDA Toolkit headers >= 11.0 "
              "(CUDART_VERSION >= 11000).");

// Export symbols with default visibility for the shared stub library.
#define TILELANG_CUDART_STUB_API __attribute__((visibility("default")))

namespace {

using CudaGraphInstantiateLegacy = cudaError_t (*)(cudaGraphExec_t *pGraphExec,
                                                   cudaGraph_t graph,
                                                   cudaGraphNode_t *pErrorNode,
                                                   char *pLogBuffer,
                                                   size_t bufferSize);
using CudaGraphInstantiateWithFlags = cudaError_t (*)(
    cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags);

void *TryLoadLibCudart() {
  // First, check if the symbols are already available globally.
  // This handles cases where PyTorch or another library has already loaded
  // libcudart, making its symbols available in the global namespace.
  // We use a representative symbol like cudaGetErrorString.
  // dlsym with RTLD_DEFAULT searches the global scope.
  void *sym = dlsym(RTLD_DEFAULT, "cudaGetErrorString");
  if (sym != nullptr && sym != reinterpret_cast<void *>(&cudaGetErrorString)) {
    return RTLD_DEFAULT;
  }
  sym = dlsym(RTLD_NEXT, "cudaGetErrorString");
  if (sym != nullptr) {
    return RTLD_NEXT;
  }

  fprintf(stderr,
          "TileLang Error: libcudart symbols not found globally. "
          "Make sure PyTorch with CUDA is installed before using TileLang.\n");
  abort();
}

template <typename T> T GetSymbol(void *handle, const char *name) {
  (void)dlerror();
  void *sym = dlsym(handle, name);
  const char *error = dlerror();
  if (error != nullptr) {
    return nullptr;
  }
  return reinterpret_cast<T>(sym);
}

struct CUDARuntimeAPI {
  decltype(&::cudaGetErrorString) cudaGetErrorString_{nullptr};
  decltype(&::cudaGetLastError) cudaGetLastError_{nullptr};
  decltype(&::cudaPeekAtLastError) cudaPeekAtLastError_{nullptr};

  decltype(&::cudaSetDevice) cudaSetDevice_{nullptr};
  decltype(&::cudaGetDevice) cudaGetDevice_{nullptr};
  decltype(&::cudaGetDeviceCount) cudaGetDeviceCount_{nullptr};
  decltype(&::cudaDeviceGetAttribute) cudaDeviceGetAttribute_{nullptr};
  decltype(&::cudaGetDeviceProperties) cudaGetDeviceProperties_{nullptr};

  decltype(&::cudaMemGetInfo) cudaMemGetInfo_{nullptr};
  decltype(&::cudaMalloc) cudaMalloc_{nullptr};
  decltype(&::cudaFree) cudaFree_{nullptr};
  decltype(&::cudaMallocHost) cudaMallocHost_{nullptr};
  decltype(&::cudaFreeHost) cudaFreeHost_{nullptr};
  decltype(&::cudaMemset) cudaMemset_{nullptr};
  decltype(&::cudaMemsetAsync) cudaMemsetAsync_{nullptr};

  decltype(&::cudaMemcpy) cudaMemcpy_{nullptr};
  decltype(&::cudaMemcpyAsync) cudaMemcpyAsync_{nullptr};
  decltype(&::cudaMemcpyPeerAsync) cudaMemcpyPeerAsync_{nullptr};

  decltype(&::cudaStreamCreate) cudaStreamCreate_{nullptr};
  decltype(&::cudaStreamCreateWithFlags) cudaStreamCreateWithFlags_{nullptr};
  decltype(&::cudaStreamDestroy) cudaStreamDestroy_{nullptr};
  decltype(&::cudaStreamSynchronize) cudaStreamSynchronize_{nullptr};
  decltype(&::cudaStreamWaitEvent) cudaStreamWaitEvent_{nullptr};

  decltype(&::cudaEventCreate) cudaEventCreate_{nullptr};
  decltype(&::cudaEventDestroy) cudaEventDestroy_{nullptr};
  decltype(&::cudaEventRecord) cudaEventRecord_{nullptr};
  decltype(&::cudaEventSynchronize) cudaEventSynchronize_{nullptr};
  decltype(&::cudaEventElapsedTime) cudaEventElapsedTime_{nullptr};

  decltype(&::cudaDeviceSynchronize) cudaDeviceSynchronize_{nullptr};

  decltype(&::cudaStreamBeginCapture) cudaStreamBeginCapture_{nullptr};
  decltype(&::cudaStreamEndCapture) cudaStreamEndCapture_{nullptr};
  // `cudaGraphInstantiate` changed signature in CUDA 12. Use explicit function
  // pointer typedefs and dispatch based on available symbols.
  CudaGraphInstantiateLegacy cudaGraphInstantiate_{nullptr};
  CudaGraphInstantiateWithFlags cudaGraphInstantiateWithFlags_{nullptr};
  decltype(&::cudaGraphLaunch) cudaGraphLaunch_{nullptr};
  decltype(&::cudaGraphDestroy) cudaGraphDestroy_{nullptr};
  decltype(&::cudaGraphExecDestroy) cudaGraphExecDestroy_{nullptr};

  decltype(&::cudaIpcGetMemHandle) cudaIpcGetMemHandle_{nullptr};
  decltype(&::cudaIpcOpenMemHandle) cudaIpcOpenMemHandle_{nullptr};
  decltype(&::cudaIpcCloseMemHandle) cudaIpcCloseMemHandle_{nullptr};

  // Not currently required by default build, but cheap to include for optional
  // contribs (e.g. vllm kernels).
  decltype(&::cudaFuncSetAttribute) cudaFuncSetAttribute_{nullptr};
};

void *GetLibCudartHandle() {
  static void *handle = TryLoadLibCudart();
  return handle;
}

cudaError_t MissingLibraryError() { return cudaErrorUnknown; }

const char *FallbackCudaErrorString(cudaError_t error) {
  if (error == cudaSuccess) {
    return "cudaSuccess";
  }
  if (error == cudaErrorUnknown) {
    return "cudaErrorUnknown (CUDA runtime stub: libcudart not found)";
  }
  return "cudaError (CUDA runtime stub: libcudart not found)";
}

CUDARuntimeAPI CreateCUDARuntimeAPI() {
  CUDARuntimeAPI api{};
  void *handle = GetLibCudartHandle();
#define LOOKUP_REQUIRED(name)                                                  \
  api.name##_ = GetSymbol<decltype(api.name##_)>(handle, #name);               \
  if (api.name##_ == nullptr) {                                                \
    return CUDARuntimeAPI{};                                                   \
  }

  // NOTE: cudaGetErrorString is optional in the sense that we can provide a
  // fallback string, but when libcudart is present it should always exist.
  api.cudaGetErrorString_ = GetSymbol<decltype(api.cudaGetErrorString_)>(
      handle, "cudaGetErrorString");

  LOOKUP_REQUIRED(cudaGetLastError)
  LOOKUP_REQUIRED(cudaPeekAtLastError)
  LOOKUP_REQUIRED(cudaSetDevice)
  LOOKUP_REQUIRED(cudaGetDevice)
  LOOKUP_REQUIRED(cudaGetDeviceCount)
  LOOKUP_REQUIRED(cudaDeviceGetAttribute)
  LOOKUP_REQUIRED(cudaGetDeviceProperties)
  LOOKUP_REQUIRED(cudaMemGetInfo)
  LOOKUP_REQUIRED(cudaMalloc)
  LOOKUP_REQUIRED(cudaFree)
  LOOKUP_REQUIRED(cudaMallocHost)
  LOOKUP_REQUIRED(cudaFreeHost)
  LOOKUP_REQUIRED(cudaMemset)
  LOOKUP_REQUIRED(cudaMemsetAsync)
  LOOKUP_REQUIRED(cudaMemcpy)
  LOOKUP_REQUIRED(cudaMemcpyAsync)
  LOOKUP_REQUIRED(cudaMemcpyPeerAsync)
  LOOKUP_REQUIRED(cudaStreamCreate)
  LOOKUP_REQUIRED(cudaStreamCreateWithFlags)
  LOOKUP_REQUIRED(cudaStreamDestroy)
  LOOKUP_REQUIRED(cudaStreamSynchronize)
  LOOKUP_REQUIRED(cudaStreamWaitEvent)
  LOOKUP_REQUIRED(cudaEventCreate)
  LOOKUP_REQUIRED(cudaEventDestroy)
  LOOKUP_REQUIRED(cudaEventRecord)
  LOOKUP_REQUIRED(cudaEventSynchronize)
  LOOKUP_REQUIRED(cudaEventElapsedTime)
  LOOKUP_REQUIRED(cudaDeviceSynchronize)
  LOOKUP_REQUIRED(cudaStreamBeginCapture)
  LOOKUP_REQUIRED(cudaStreamEndCapture)
  LOOKUP_REQUIRED(cudaGraphInstantiate)
  api.cudaGraphInstantiateWithFlags_ = GetSymbol<CudaGraphInstantiateWithFlags>(
      handle, "cudaGraphInstantiateWithFlags");
  LOOKUP_REQUIRED(cudaGraphLaunch)
  LOOKUP_REQUIRED(cudaGraphDestroy)
  LOOKUP_REQUIRED(cudaGraphExecDestroy)
  LOOKUP_REQUIRED(cudaIpcGetMemHandle)
  LOOKUP_REQUIRED(cudaIpcOpenMemHandle)
  LOOKUP_REQUIRED(cudaIpcCloseMemHandle)

  // Optional
  api.cudaFuncSetAttribute_ = GetSymbol<decltype(api.cudaFuncSetAttribute_)>(
      handle, "cudaFuncSetAttribute");

#undef LOOKUP_REQUIRED

  return api;
}

CUDARuntimeAPI *GetCUDARuntimeAPI() {
  static CUDARuntimeAPI singleton = CreateCUDARuntimeAPI();
  return &singleton;
}

cudaError_t GraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                             unsigned long long flags,
                             cudaGraphNode_t *pErrorNode, char *pLogBuffer,
                             size_t bufferSize) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaGraphInstantiateWithFlags_ != nullptr) {
    return api->cudaGraphInstantiateWithFlags_(pGraphExec, graph, flags);
  }
  if (api->cudaGraphInstantiate_ == nullptr) {
    if (pGraphExec != nullptr) {
      *pGraphExec = nullptr;
    }
    return MissingLibraryError();
  }
  // Legacy API (CUDA 11.0-11.3): `cudaGraphInstantiate` has no flags parameter.
  // The caller (TVM) uses flags=0 and passes NULL diagnostics.
  (void)flags;
  return api->cudaGraphInstantiate_(pGraphExec, graph, pErrorNode, pLogBuffer,
                                    bufferSize);
}

} // namespace

extern "C" {

TILELANG_CUDART_STUB_API const char *cudaGetErrorString(cudaError_t error) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaGetErrorString_ != nullptr) {
    return api->cudaGetErrorString_(error);
  }
  return FallbackCudaErrorString(error);
}

TILELANG_CUDART_STUB_API cudaError_t cudaGetLastError(void) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaGetLastError_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaGetLastError_();
}

TILELANG_CUDART_STUB_API cudaError_t cudaPeekAtLastError(void) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaPeekAtLastError_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaPeekAtLastError_();
}

TILELANG_CUDART_STUB_API cudaError_t cudaSetDevice(int device) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaSetDevice_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaSetDevice_(device);
}

TILELANG_CUDART_STUB_API cudaError_t cudaGetDevice(int *device) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaGetDevice_ == nullptr) {
    if (device != nullptr) {
      *device = 0;
    }
    return MissingLibraryError();
  }
  return api->cudaGetDevice_(device);
}

TILELANG_CUDART_STUB_API cudaError_t cudaGetDeviceCount(int *count) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaGetDeviceCount_ == nullptr) {
    if (count != nullptr) {
      *count = 0;
    }
    return MissingLibraryError();
  }
  return api->cudaGetDeviceCount_(count);
}

TILELANG_CUDART_STUB_API cudaError_t cudaDeviceGetAttribute(int *value,
                                                            cudaDeviceAttr attr,
                                                            int device) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaDeviceGetAttribute_ == nullptr) {
    if (value != nullptr) {
      *value = 0;
    }
    return MissingLibraryError();
  }
  return api->cudaDeviceGetAttribute_(value, attr, device);
}

TILELANG_CUDART_STUB_API cudaError_t
cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaGetDeviceProperties_ == nullptr) {
    if (prop != nullptr) {
      memset(prop, 0, sizeof(*prop));
    }
    return MissingLibraryError();
  }
  return api->cudaGetDeviceProperties_(prop, device);
}

TILELANG_CUDART_STUB_API cudaError_t cudaMemGetInfo(size_t *free,
                                                    size_t *total) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaMemGetInfo_ == nullptr) {
    if (free != nullptr) {
      *free = 0;
    }
    if (total != nullptr) {
      *total = 0;
    }
    return MissingLibraryError();
  }
  return api->cudaMemGetInfo_(free, total);
}

TILELANG_CUDART_STUB_API cudaError_t cudaMalloc(void **devPtr, size_t size) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaMalloc_ == nullptr) {
    if (devPtr != nullptr) {
      *devPtr = nullptr;
    }
    return MissingLibraryError();
  }
  return api->cudaMalloc_(devPtr, size);
}

TILELANG_CUDART_STUB_API cudaError_t cudaFree(void *devPtr) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaFree_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaFree_(devPtr);
}

TILELANG_CUDART_STUB_API cudaError_t cudaMallocHost(void **ptr, size_t size) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaMallocHost_ == nullptr) {
    if (ptr != nullptr) {
      *ptr = nullptr;
    }
    return MissingLibraryError();
  }
  return api->cudaMallocHost_(ptr, size);
}

TILELANG_CUDART_STUB_API cudaError_t cudaFreeHost(void *ptr) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaFreeHost_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaFreeHost_(ptr);
}

TILELANG_CUDART_STUB_API cudaError_t cudaMemset(void *devPtr, int value,
                                                size_t count) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaMemset_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaMemset_(devPtr, value, count);
}

TILELANG_CUDART_STUB_API cudaError_t cudaMemsetAsync(void *devPtr, int value,
                                                     size_t count,
                                                     cudaStream_t stream) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaMemsetAsync_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaMemsetAsync_(devPtr, value, count, stream);
}

TILELANG_CUDART_STUB_API cudaError_t cudaMemcpy(void *dst, const void *src,
                                                size_t count,
                                                cudaMemcpyKind kind) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaMemcpy_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaMemcpy_(dst, src, count, kind);
}

TILELANG_CUDART_STUB_API cudaError_t cudaMemcpyAsync(void *dst, const void *src,
                                                     size_t count,
                                                     cudaMemcpyKind kind,
                                                     cudaStream_t stream) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaMemcpyAsync_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaMemcpyAsync_(dst, src, count, kind, stream);
}

TILELANG_CUDART_STUB_API cudaError_t
cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice,
                    size_t count, cudaStream_t stream) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaMemcpyPeerAsync_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaMemcpyPeerAsync_(dst, dstDevice, src, srcDevice, count,
                                   stream);
}

TILELANG_CUDART_STUB_API cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaStreamCreate_ == nullptr) {
    if (pStream != nullptr) {
      *pStream = nullptr;
    }
    return MissingLibraryError();
  }
  return api->cudaStreamCreate_(pStream);
}

TILELANG_CUDART_STUB_API cudaError_t
cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaStreamCreateWithFlags_ == nullptr) {
    if (pStream != nullptr) {
      *pStream = nullptr;
    }
    return MissingLibraryError();
  }
  return api->cudaStreamCreateWithFlags_(pStream, flags);
}

TILELANG_CUDART_STUB_API cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaStreamDestroy_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaStreamDestroy_(stream);
}

TILELANG_CUDART_STUB_API cudaError_t
cudaStreamSynchronize(cudaStream_t stream) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaStreamSynchronize_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaStreamSynchronize_(stream);
}

TILELANG_CUDART_STUB_API cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
                                                         cudaEvent_t event,
                                                         unsigned int flags) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaStreamWaitEvent_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaStreamWaitEvent_(stream, event, flags);
}

TILELANG_CUDART_STUB_API cudaError_t cudaEventCreate(cudaEvent_t *event) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaEventCreate_ == nullptr) {
    if (event != nullptr) {
      *event = nullptr;
    }
    return MissingLibraryError();
  }
  return api->cudaEventCreate_(event);
}

TILELANG_CUDART_STUB_API cudaError_t cudaEventDestroy(cudaEvent_t event) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaEventDestroy_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaEventDestroy_(event);
}

TILELANG_CUDART_STUB_API cudaError_t cudaEventRecord(cudaEvent_t event,
                                                     cudaStream_t stream) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaEventRecord_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaEventRecord_(event, stream);
}

TILELANG_CUDART_STUB_API cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaEventSynchronize_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaEventSynchronize_(event);
}

TILELANG_CUDART_STUB_API cudaError_t cudaEventElapsedTime(float *ms,
                                                          cudaEvent_t start,
                                                          cudaEvent_t end) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaEventElapsedTime_ == nullptr) {
    if (ms != nullptr) {
      *ms = 0.0f;
    }
    return MissingLibraryError();
  }
  return api->cudaEventElapsedTime_(ms, start, end);
}

TILELANG_CUDART_STUB_API cudaError_t cudaDeviceSynchronize(void) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaDeviceSynchronize_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaDeviceSynchronize_();
}

TILELANG_CUDART_STUB_API cudaError_t
cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaStreamBeginCapture_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaStreamBeginCapture_(stream, mode);
}

TILELANG_CUDART_STUB_API cudaError_t cudaStreamEndCapture(cudaStream_t stream,
                                                          cudaGraph_t *graph) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaStreamEndCapture_ == nullptr) {
    if (graph != nullptr) {
      *graph = nullptr;
    }
    return MissingLibraryError();
  }
  return api->cudaStreamEndCapture_(stream, graph);
}

#if CUDART_VERSION >= 12000
TILELANG_CUDART_STUB_API cudaError_t cudaGraphInstantiate(
    cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags) {
  return GraphInstantiate(pGraphExec, graph, flags, nullptr, nullptr, 0);
}
#else
TILELANG_CUDART_STUB_API cudaError_t cudaGraphInstantiate(
    cudaGraphExec_t *pGraphExec, cudaGraph_t graph, cudaGraphNode_t *pErrorNode,
    char *pLogBuffer, size_t bufferSize) {
  return GraphInstantiate(pGraphExec, graph, 0, pErrorNode, pLogBuffer,
                          bufferSize);
}
#endif

TILELANG_CUDART_STUB_API cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec,
                                                     cudaStream_t stream) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaGraphLaunch_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaGraphLaunch_(graphExec, stream);
}

TILELANG_CUDART_STUB_API cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaGraphDestroy_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaGraphDestroy_(graph);
}

TILELANG_CUDART_STUB_API cudaError_t
cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaGraphExecDestroy_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaGraphExecDestroy_(graphExec);
}

TILELANG_CUDART_STUB_API cudaError_t
cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaIpcGetMemHandle_ == nullptr) {
    if (handle != nullptr) {
      memset(handle, 0, sizeof(*handle));
    }
    return MissingLibraryError();
  }
  return api->cudaIpcGetMemHandle_(handle, devPtr);
}

TILELANG_CUDART_STUB_API cudaError_t cudaIpcOpenMemHandle(
    void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaIpcOpenMemHandle_ == nullptr) {
    if (devPtr != nullptr) {
      *devPtr = nullptr;
    }
    return MissingLibraryError();
  }
  return api->cudaIpcOpenMemHandle_(devPtr, handle, flags);
}

TILELANG_CUDART_STUB_API cudaError_t cudaIpcCloseMemHandle(void *devPtr) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaIpcCloseMemHandle_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaIpcCloseMemHandle_(devPtr);
}

TILELANG_CUDART_STUB_API cudaError_t
cudaFuncSetAttribute(const void *func, cudaFuncAttribute attr, int value) {
  auto *api = GetCUDARuntimeAPI();
  if (api->cudaFuncSetAttribute_ == nullptr) {
    return MissingLibraryError();
  }
  return api->cudaFuncSetAttribute_(func, attr, value);
}

} // extern "C"
