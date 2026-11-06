// Test-only inspection of native packed-call tensor descriptors.
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace ffi = tvm::ffi;
namespace {
// Copy fields and values while the packed-call stack owns the descriptor.
// Numeric addresses are retained only for identity checks, never dereferenced
// after the callback returns.
std::vector<ffi::Array<ffi::Any>> descriptor_observations;

ffi::Array<ffi::Any> ReadDescriptor(DLTensor *tensor, int64_t tag) {
  TVM_FFI_CHECK(tensor && tensor->data, ValueError);
  TVM_FFI_CHECK(tensor->ndim >= 0 && tensor->ndim <= 4, ValueError);
  TVM_FFI_CHECK(tensor->dtype.code == kDLFloat && tensor->dtype.bits == 32 &&
                    tensor->dtype.lanes == 1,
                ValueError);
  ffi::Array<int64_t> shape;
  ffi::Array<int64_t> raw_strides;
  std::vector<int64_t> strides(tensor->ndim);
  int64_t count = 1;
  for (int i = 0; i < tensor->ndim; ++i) {
    TVM_FFI_CHECK(tensor->shape && tensor->shape[i] > 0, ValueError);
    shape.push_back(tensor->shape[i]);
    count *= tensor->shape[i];
    if (tensor->strides) {
      raw_strides.push_back(tensor->strides[i]);
    }
  }
  TVM_FFI_CHECK(count <= 32, ValueError);
  int64_t compact_stride = 1;
  for (int i = tensor->ndim - 1; i >= 0; --i) {
    strides[i] = tensor->strides ? tensor->strides[i] : compact_stride;
    compact_stride *= tensor->shape[i];
  }
  ffi::Array<double> values;
  for (int64_t ordinal = 0; ordinal < count; ++ordinal) {
    int64_t remaining = ordinal;
    int64_t element = 0;
    for (int i = tensor->ndim - 1; i >= 0; --i) {
      element += (remaining % tensor->shape[i]) * strides[i];
      remaining /= tensor->shape[i];
    }
    float value;
    const char *address = static_cast<const char *>(tensor->data) +
                          tensor->byte_offset + element * sizeof(float);
    std::memcpy(&value, address, sizeof(value));
    values.push_back(static_cast<double>(value));
  }
  return {tag,
          static_cast<int64_t>(reinterpret_cast<uintptr_t>(tensor)),
          static_cast<int64_t>(reinterpret_cast<uintptr_t>(tensor->data)),
          static_cast<int64_t>(reinterpret_cast<uintptr_t>(tensor->shape)),
          static_cast<int64_t>(reinterpret_cast<uintptr_t>(tensor->strides)),
          int64_t(tensor->ndim),
          int64_t(tensor->device.device_type),
          int64_t(tensor->device.device_id),
          int64_t(tensor->dtype.code),
          int64_t(tensor->dtype.bits),
          int64_t(tensor->dtype.lanes),
          static_cast<int64_t>(tensor->byte_offset),
          shape,
          tensor->strides ? ffi::Any(raw_strides) : ffi::Any(nullptr),
          values};
}
} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(abi154_observe, [](DLTensor *tensor,
                                                 int64_t tag) {
  descriptor_observations.push_back(ReadDescriptor(tensor, tag));
  return tag + 1000;
});
TVM_FFI_DLL_EXPORT_TYPED_FUNC(abi154_records, []() {
  return ffi::Array<ffi::Array<ffi::Any>>(descriptor_observations.begin(),
                                          descriptor_observations.end());
});
TVM_FFI_DLL_EXPORT_TYPED_FUNC(abi154_clear,
                              []() { descriptor_observations.clear(); });
