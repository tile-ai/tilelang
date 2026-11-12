/**
 * \file ascendcl.h
 * \brief Stub library header for lazy loading the CANN runtime (libascendcl).
 *
 * This mirrors the CUDA/ROCm stubs in src/cuda/stubs/ and src/rocm/stubs/:
 * - Instead of linking against libascendcl.so at build time, TileLang links
 *   against a small stub library (libstub_ascendcl.so) that resolves the ACL
 *   symbols via dlopen()/dlsym() on first use.
 *
 * This enables:
 * 1. Importing TileLang on machines without CANN installed.
 * 2. Reusing the libascendcl copy already loaded by torch_npu.
 * 3. Building a single wheel that can run across environments.
 *
 * Unlike the CUDA/ROCm stubs, no CANN headers are required (or vendored): the
 * TileLang runtime only uses a handful of entrypoints whose ABI is expressed
 * here with opaque pointer and fixed-width integer types.
 *
 * Usage:
 *   #include "ascend/stubs/ascendcl.h"
 *   int32_t error = aclrtGetDevice(&device_id);
 */

#pragma once

#include <cstddef>
#include <cstdint>

// Symbol visibility macros for shared library export.
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef TILELANG_ASCENDCL_STUB_EXPORTS
#define TILELANG_ASCENDCL_STUB_API __declspec(dllexport)
#else
#define TILELANG_ASCENDCL_STUB_API __declspec(dllimport)
#endif
#else
#define TILELANG_ASCENDCL_STUB_API __attribute__((visibility("default")))
#endif

namespace tvm::tl::ascendcl {

// ABI of acldumpTensorInfo from CANN 9.2+ acl/acl_dump.h. Keep the enums as
// int32_t and the fixed-size shape arrays intact; no SDK headers are needed.
struct DumpTensorInfo {
  int32_t type;
  size_t tensor_size;
  int32_t format;
  int32_t data_type;
  int64_t *tensor_addr;
  int32_t addr_type;
  int32_t placement;
  uint32_t args_offset;
  uint32_t shape_num;
  uint32_t origin_shape_num;
  uint64_t shape[25];
  uint64_t origin_shape[25];
};
static_assert(sizeof(DumpTensorInfo) == 456, "CANN dump tensor ABI");
static_assert(offsetof(DumpTensorInfo, shape) == 56, "CANN dump shape ABI");
using ExceptionInfoCallback = void (*)(void *);

/**
 * \brief ACL API accessor struct with lazy loading support.
 *
 * Similar to tvm::tl::hip::HIPDriverAPI, this struct resolves libascendcl.so
 * symbols lazily on first use.
 *
 * Function pointer members have a trailing underscore to avoid collisions with
 * the exported global wrapper functions.
 */
struct TILELANG_ASCENDCL_STUB_API AscendCLAPI {
  int32_t (*aclrtBinaryLoadFromData_)(const void *, size_t, const void *,
                                      void **);
  int32_t (*aclrtBinaryGetFunction_)(void *, const char *, void **);
  int32_t (*aclrtBinaryUnLoad_)(void *);
  int32_t (*aclrtGetDevice_)(int32_t *);
  int32_t (*aclrtLaunchKernelWithHostArgs_)(void *, uint32_t, void *, void *,
                                            void *, size_t, void *, size_t);
  // Optional: absent on CANN versions that predate the API.
  const char *(*aclGetRecentErrMsg_)(void);
  // Optional: absent on CANN versions that predate the API. Used only by
  // Ascend native profiling (TILELANG_ASCEND_PROFILER).
  int32_t (*aclrtGetFunctionAttribute_)(void *, int32_t, int64_t *);
  int32_t (*aclrtGetLogicDevIdByUserDevId_)(int32_t, int32_t *);

  // Optional: used directly by the native exception-dump runtime through this
  // table. No exported ACL forwarding functions are needed for these entries.
  int32_t (*aclrtExceptionInfoCallbackRegister_)(ExceptionInfoCallback);
  int32_t (*aclrtGetArgsFromExceptionInfo_)(const void *, void **, uint32_t *);
  int32_t (*aclrtMemcpy_)(void *, size_t, const void *, size_t, int32_t);
  int32_t (*acldumpSaveExceptionInfo_)(const char *, const char *,
                                       const DumpTensorInfo *, size_t);

  /// Throws std::runtime_error when libascendcl.so (or a required symbol
  /// within it) is unavailable.
  static AscendCLAPI *get();
  /// Return an API table with null function pointers if loading/resolution
  /// fails, without the required-symbol errors reported by get().
  static AscendCLAPI *get_or_null();
  static bool is_available();
  static void *get_handle();
};

} // namespace tvm::tl::ascendcl

// ============================================================================
// Global wrapper functions for the lazy-loaded ACL API
// ============================================================================
// These functions provide drop-in replacements for the CANN runtime
// entrypoints used by TileLang. The implementations are in ascendcl.cc.

extern "C" {

TILELANG_ASCENDCL_STUB_API int32_t aclrtBinaryLoadFromData(const void *data,
                                                           size_t size,
                                                           const void *options,
                                                           void **binHandle);
TILELANG_ASCENDCL_STUB_API int32_t aclrtBinaryGetFunction(void *binHandle,
                                                          const char *name,
                                                          void **funcHandle);
TILELANG_ASCENDCL_STUB_API int32_t aclrtBinaryUnLoad(void *binHandle);
TILELANG_ASCENDCL_STUB_API int32_t aclrtGetDevice(int32_t *deviceId);
TILELANG_ASCENDCL_STUB_API int32_t aclrtLaunchKernelWithHostArgs(
    void *func, uint32_t numBlocks, void *stream, void *config, void *args,
    size_t argsSize, void *reserved, size_t reservedSize);
/// Returns nullptr (instead of throwing) when libascendcl.so or the symbol is
/// unavailable, so error-reporting paths stay usable.
TILELANG_ASCENDCL_STUB_API const char *aclGetRecentErrMsg(void);

/// Returns a nonzero ACL error (instead of throwing) when libascendcl.so or
/// the symbol is unavailable, so best-effort profiling paths can fall back.
TILELANG_ASCENDCL_STUB_API int32_t
aclrtGetFunctionAttribute(void *func, int32_t attrType, int64_t *attrValue);
TILELANG_ASCENDCL_STUB_API int32_t
aclrtGetLogicDevIdByUserDevId(int32_t userDeviceId, int32_t *logicDeviceId);

} // extern "C"
