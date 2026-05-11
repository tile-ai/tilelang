/*!
 * \file tl/cuda/runtime.h
 * \brief Runtime functions.
 *
 */

#ifndef TVM_TL_BACKEND_CUDA_RUNTIME_H_
#define TVM_TL_BACKEND_CUDA_RUNTIME_H_

#include <cstddef>
#include <cstdint>

namespace tvm {
namespace tl {

#if (CUDA_MAJOR_VERSION >= 12)
void SetRemoteTensorMapMetaData(const uint64_t *table, size_t table_size);

constexpr const char *tvm_tensormap_create_tiled =
    "__tvm_tensormap_create_tiled";
// Remote descriptor initialization is currently consumed by the TVM FFI host
// path. The Cython/NVRTC source wrappers intentionally reject this helper until
// they learn to keep a host-side distributed base table as well.
constexpr const char *tvm_tensormap_create_remote_tiled =
    "__tvm_tensormap_create_remote_tiled";
constexpr const char *tvm_tensormap_create_im2col =
    "__tvm_tensormap_create_im2col";
#endif // (CUDA_MAJOR_VERSION >= 12)

// CUDA stream access policy window helpers
constexpr const char *tvm_cuda_stream_set_access_policy_window =
    "__tvm_cuda_stream_set_access_policy_window";
constexpr const char *tvm_cuda_stream_reset_access_policy_window =
    "__tvm_cuda_stream_reset_access_policy_window";
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_CUDA_RUNTIME_H_
