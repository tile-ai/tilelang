/*!
 * \file output_frame.cc
 * \brief Allocate through the active framework, retaining outputs with RAII.
 */
#include "output_frame.h"

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstdint>
#include <limits>

namespace tvm::tl {

void OutputFrame::Allocate(int slot, ffi::ShapeView shape) {
  TVM_FFI_CHECK(slot >= 0 && static_cast<size_t>(slot) < outputs_.size(),
                ValueError)
      << "Invalid output slot " << slot;
  TVM_FFI_CHECK(!outputs_[slot].defined(), ValueError)
      << "Output slot allocated twice";
  const auto &spec = specs_[slot];
  TVM_FFI_CHECK_EQ(shape.size(), spec.rank, ValueError)
      << "Output rank mismatch";
  uint64_t elements = 1;
  bool empty = false;
  for (int64_t extent : shape) {
    TVM_FFI_CHECK_GE(extent, 0, ValueError) << "Negative output dimension";
    empty |= extent == 0;
  }
  if (!empty) {
    for (int64_t extent : shape) {
      TVM_FFI_CHECK_LE(static_cast<uint64_t>(extent),
                       std::numeric_limits<uint64_t>::max() / elements,
                       ValueError)
          << "Output shape product overflows";
      elements *= static_cast<uint64_t>(extent);
    }
    uint64_t bits = static_cast<uint64_t>(spec.storage_dtype.bits) *
                    spec.storage_dtype.lanes;
    TVM_FFI_CHECK_GT(bits, 0U, ValueError) << "Invalid output storage dtype";
    TVM_FFI_CHECK_LE(
        elements, (std::numeric_limits<uint64_t>::max() - 7) / bits, ValueError)
        << "Output byte size overflows";
    TVM_FFI_CHECK_LE((elements * bits + 7) / 8,
                     static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                     ValueError)
        << "Output storage is too large";
  }
  ffi::Tensor result = ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, shape,
                                                 spec.storage_dtype, device_);
  // Validate metadata structure before the typed compute binder reads it.
  // That binder owns the declared logical dtype/shape/stride/offset contract,
  // including compatible storage dtypes and singleton/empty stride rules.
  const DLTensor *descriptor = result.GetDLTensorPtr();
  TVM_FFI_CHECK_EQ(descriptor->ndim, spec.rank, ValueError)
      << "Output allocator returned an unexpected rank";
  TVM_FFI_CHECK(spec.rank == 0 || descriptor->shape != nullptr, ValueError)
      << "Output allocator returned a NULL shape";
  for (int i = 0; i < spec.rank; ++i) {
    TVM_FFI_CHECK_GE(descriptor->shape[i], 0, ValueError)
        << "Output allocator returned a negative dimension";
  }
  outputs_[slot] = std::move(result);
}

const ffi::Tensor &OutputFrame::Output(int slot) const {
  TVM_FFI_CHECK(slot >= 0 && static_cast<size_t>(slot) < outputs_.size(),
                ValueError)
      << "Invalid output slot";
  TVM_FFI_CHECK(outputs_[slot].defined(), RuntimeError)
      << "Allocation worker returned without output slot " << slot;
  return outputs_[slot];
}

ffi::Any OutputFrame::Complete() const {
  if (outputs_.size() == 1)
    return Output(0);
  ffi::Array<ffi::Tensor> ordered;
  for (size_t i = 0; i < outputs_.size(); ++i)
    ordered.push_back(Output(static_cast<int>(i)));
  return ordered;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def_packed(
      "tl.runtime.AllocateOutput", [](ffi::PackedArgs args, ffi::Any *result) {
        TVM_FFI_CHECK_GE(args.size(), 2, TypeError)
            << "Expected frame, slot, and dimensions";
        auto *frame = static_cast<OutputFrame *>(args[0].cast<void *>());
        TVM_FFI_CHECK(frame != nullptr, ValueError) << "Output frame is NULL";
        std::vector<int64_t> dimensions;
        for (int i = 2; i < args.size(); ++i)
          dimensions.push_back(args[i].cast<int64_t>());
        frame->Allocate(args[1].cast<int>(),
                        ffi::ShapeView(dimensions.data(), dimensions.size()));
        *result = 0;
      });
}
} // namespace tvm::tl
