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

#include "target/metal/metal_fallback_module.h"

namespace tvm {
namespace codegen {

inline ffi::Module
MetalModuleCreate(ffi::Map<ffi::String, ffi::Bytes> smap,
                  ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                  ffi::String fmt, ffi::String source) {
  return target::MetalModuleCreateWithFallback(
      std::move(smap), std::move(fmt), std::move(fmap),
      ffi::Map<ffi::String, ffi::String>{{"metal", std::move(source)}});
}

} // namespace codegen
} // namespace tvm

#endif // TVM_RUNTIME_METAL_METAL_MODULE_H_
