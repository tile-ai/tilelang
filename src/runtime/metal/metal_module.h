#ifndef TVM_RUNTIME_METAL_METAL_MODULE_H_
#define TVM_RUNTIME_METAL_METAL_MODULE_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/codegen.h>

#include <string>
#include <utility>

namespace tvm {
namespace codegen {

inline ffi::Module MetalModuleCreate(ffi::Map<ffi::String, ffi::Bytes> smap,
                                      ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                      ffi::String fmt, ffi::String source) {
  auto fcreate = ffi::Function::GetGlobal("ffi.Module.create.metal");
  if (fcreate.has_value()) {
    return (*fcreate)(smap, fmt, fmap,
                       ffi::Map<ffi::String, ffi::String>{{"metal", source}})
        .cast<ffi::Module>();
  }
  auto fallback = ffi::Function::GetGlobal("ffi.Module.create.metal_fallback");
  if (fallback.has_value()) {
    return (*fallback)(smap, fmt, fmap,
                        ffi::Map<ffi::String, ffi::String>{{"metal", source}})
        .cast<ffi::Module>();
  }
  LOG(FATAL) << "Metal module factory not available.";
}

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_RUNTIME_METAL_METAL_MODULE_H_
