#include <tvm/ffi/c_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl {

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.runtime.CallGlobalWithResult",
      [](ffi::String name, void *arguments, int32_t count, void *result) {
        auto function = ffi::Function::GetGlobal(name);
        if (!function) {
          std::string message =
              "Unknown global packed function: " + std::string(name);
          TVMFFIErrorSetRaisedFromCStr("ValueError", message.c_str());
          return -1;
        }
        return TVMFFIFunctionCall(
            const_cast<ffi::FunctionObj *>(function->get()),
            static_cast<TVMFFIAny *>(arguments), count,
            static_cast<TVMFFIAny *>(result));
      });
}

} // namespace tvm::tl
