/*!
 * \file rt_mod_metal.cc
 * \brief Native MPS submission for TVM-FFI Host IR.
 */
#include <torch/mps.h>
#include <tvm/ffi/function.h>

namespace {
int CallOnMPS(TVMFFIObjectHandle func, TVMFFIAny *args, int32_t count,
              TVMFFIAny *result) {
  TVM_FFI_SAFE_CALL_BEGIN();
  static const auto set_stream =
      tvm::ffi::Function::GetGlobalRequired("metal.SetStream");
  set_stream(static_cast<void *>(torch::mps::get_command_buffer()));
  return TVMFFIFunctionCall(func, args, count, result);
  TVM_FFI_SAFE_CALL_END();
}
} // namespace

extern "C" TVM_FFI_DLL int TileLangMetalFunctionCall(TVMFFIObjectHandle func,
                                                     TVMFFIAny *args,
                                                     int32_t count,
                                                     TVMFFIAny *result) {
  __block int status = 0;
  __block TVMFFIObjectHandle error = nullptr;
  dispatch_sync(torch::mps::get_dispatch_queue(), ^{
    status = CallOnMPS(func, args, count, result);
    if (status != 0)
      TVMFFIErrorMoveFromRaised(&error);
  });
  if (error != nullptr) {
    TVMFFIErrorSetRaised(error);
    TVMFFIObjectDecRef(error);
  }
  return status;
}
