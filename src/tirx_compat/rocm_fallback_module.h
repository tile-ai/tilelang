#ifndef TILELANG_TIRX_COMPAT_ROCM_FALLBACK_MODULE_H_
#define TILELANG_TIRX_COMPAT_ROCM_FALLBACK_MODULE_H_

#include <tvm/ffi/extra/module.h>

#include "tirx_compat/metadata.h"

namespace tvm::target {

ffi::Module
ROCmFallbackModuleCreate(ffi::Bytes code, ffi::String format,
                         ffi::Map<ffi::String, runtime::FunctionInfo> functions,
                         ffi::Map<ffi::String, ffi::String> source);

}

#endif
