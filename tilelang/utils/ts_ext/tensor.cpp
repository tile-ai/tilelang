#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <torch/extension.h>
#include <vector>

#include "ts_ext_ops.h"
#include "exception.h"

static int64_t safe_mul_int64(int64_t a, int64_t b) {
  if (a == 0 || b == 0)
    return 0;
  int64_t maxv = std::numeric_limits<int64_t>::max();
  if (a > maxv / b)
    throw std::overflow_error("integer overflow in multiplication");
  return a * b;
}

static at::ScalarType dtype_from_string(const std::string &s) {
  if (s == "float32" || s == "float")
    return at::kFloat;
  if (s == "float16" || s == "half")
    return at::kHalf;
  if (s == "bfloat16" || s == "bfloat")
    return at::kBFloat16;
  if (s == "float64" || s == "double")
    return at::kDouble;
  if (s == "uint32")
    return at::kUInt32;
  if (s == "uint64")
    return at::kUInt64;
  if (s == "int32" || s == "int")
    return at::kInt;
  if (s == "int64" || s == "long")
    return at::kLong;
  if (s == "uint8" || s == "byte")
    return at::kByte;
  if (s == "int8")
    return at::kChar;
  if (s == "bool")
    return at::kBool;
  throw std::runtime_error("Unsupported dtype string: " + s);
}

torch::Tensor tensor_from_ptr(uint64_t ptr_val, std::vector<int64_t> shape,
                              const std::string &dtype, int64_t device,
                              bool take_ownership) {
  if (ptr_val == 0)
    throw std::runtime_error("Received null pointer (0).");
  void *data_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(ptr_val));

  at::ScalarType st = dtype_from_string(dtype);
  auto options = torch::TensorOptions().dtype(st).device(
      torch::kCUDA, static_cast<int>(device));

  int64_t nelems = 1;
  for (auto d : shape) {
    if (d < 0)
      throw std::runtime_error("Negative dimension in shape");
    nelems = safe_mul_int64(nelems, d);
  }

  std::function<void(void *)> deleter;
  if (take_ownership) {
    uint64_t saved_ptr = ptr_val;
    deleter = [saved_ptr](void *) {
      void *p = reinterpret_cast<void *>(static_cast<uintptr_t>(saved_ptr));
      cudaError_t cerr = cudaFree(p);
      if (cerr != cudaSuccess) {
        std::fprintf(stderr, "tensor_from_ptr deleter cudaFree failed: %s\n",
                     cudaGetErrorString(cerr));
      }
    };
  } else {
    deleter = [](void *) {};
  }

  if (nelems == 0) {
    return torch::empty(shape, options);
  } else {
    return at::from_blob(data_ptr, shape, deleter, options);
  }
}

torch::Tensor get_device_tensor(torch::Tensor tensor) {
  void* device_ptr = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&device_ptr, tensor.data_ptr(), 0));
  std::vector<int64_t> shape(tensor.sizes().begin(), tensor.sizes().end());
  std::string dtype_name(tensor.dtype().name());
  return tensor_from_ptr(reinterpret_cast<uint64_t>(device_ptr), shape, dtype_name, tensor.device().index(), false);
}