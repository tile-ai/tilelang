/*!
 * \file output_frame.h
 * \brief Per-invocation ownership for callee-allocated outputs.
 */
#ifndef TVM_TL_RUNTIME_OUTPUT_FRAME_H_
#define TVM_TL_RUNTIME_OUTPUT_FRAME_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <vector>

namespace tvm::tl {

struct OutputSlotSpec {
  int parameter_index;
  DLDataType storage_dtype;
  int rank;
};

/*! \brief Own all outputs until validation and compute complete successfully.
 *
 * The allocation worker only borrows this frame during a synchronous packed
 * call. An exception in allocation, argument validation, compute, or return
 * construction releases every populated slot through Tensor's normal deleter.
 * Device lifetime and stream ordering remain owned by the framework allocator.
 */
class OutputFrame {
public:
  OutputFrame(const std::vector<OutputSlotSpec> &specs, DLDevice device)
      : specs_(specs), outputs_(specs.size()), device_(device) {}
  OutputFrame(const OutputFrame &) = delete;
  OutputFrame &operator=(const OutputFrame &) = delete;

  void Allocate(int slot, ffi::ShapeView shape);
  const ffi::Tensor &Output(int slot) const;
  ffi::Any Complete() const;

private:
  const std::vector<OutputSlotSpec> &specs_;
  std::vector<ffi::Tensor> outputs_;
  DLDevice device_;
};

} // namespace tvm::tl
#endif // TVM_TL_RUNTIME_OUTPUT_FRAME_H_
