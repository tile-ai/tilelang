#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <torch/python.h>
#include <torch/types.h>

#include <cstdio>
#include <cstring>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#include "ts_ext_ops.h"
#include "exception.h"

namespace py = pybind11;

static size_t numel_of(const std::vector<int64_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t{1},
                         [](size_t a, int64_t b) {
                           if (b < 0)
                             throw std::runtime_error("Negative dim");
                           return a * (size_t)b;
                         });
}

static size_t dtype_nbytes(c10::ScalarType dtype) {
  return (size_t)at::elementSize(dtype);
}

torch::Tensor create_tensor(const std::vector<int64_t> &shape,
                            c10::ScalarType dtype) {
  auto current_device = c10::cuda::current_device();
  auto options =
      at::TensorOptions(at::kCUDA).dtype(dtype).device_index(current_device);

  const size_t bytes = dtype_nbytes(dtype) * numel_of(shape);

  CUDA_CHECK(cudaDeviceSynchronize());
  void *ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, bytes));

  return at::from_blob(
      ptr, shape,
      [](void *p) {
        cudaError_t cerr = cudaFree(p);
        if (cerr != cudaSuccess) {
          std::fprintf(stderr, "cudaFree failed in deleter: %s\n",
                       cudaGetErrorString(cerr));
        }
      },
      options);
}

py::bytearray create_ipc_handle(void *ptr) {
  cudaIpcMemHandle_t handle{};
  CUDA_CHECK(cudaIpcGetMemHandle(&handle, ptr));
  return py::bytearray(reinterpret_cast<const char *>(handle.reserved),
                       CUDA_IPC_HANDLE_SIZE);
}

void sync_ipc_handles(
    int rank, const std::vector<int> &device_ids, void **buffer_ptrs_gpu,
    const std::vector<std::optional<py::bytearray>> &all_gathered_handles,
    const std::optional<py::bytearray> & /*root_unique_id_opt*/) {

  const int num = (int)device_ids.size();
  const int rdma_rank = 0;

  TS_HOST_ASSERT((size_t)num == all_gathered_handles.size());

  std::vector<cudaIpcMemHandle_t> ipc_handles(num);
  std::vector<void *> buffer_ptrs(num, nullptr);

  for (int i = 0, offset = rdma_rank * num; i < num; ++i) {
    TS_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
    std::string s = (std::string)all_gathered_handles[offset + i].value();
    TS_HOST_ASSERT(s.size() == CUDA_IPC_HANDLE_SIZE);
    if (offset + i != rank) {
      std::memcpy(ipc_handles[i].reserved, s.data(), CUDA_IPC_HANDLE_SIZE);
      CUDA_CHECK(cudaIpcOpenMemHandle(&buffer_ptrs[i], ipc_handles[i],
                                      cudaIpcMemLazyEnablePeerAccess));
    }
  }

  CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs.data(),
                        sizeof(void *) * buffer_ptrs.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
}
