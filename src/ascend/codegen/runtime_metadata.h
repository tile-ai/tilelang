#ifndef TVM_TL_ASCEND_CODEGEN_RUNTIME_METADATA_H_
#define TVM_TL_ASCEND_CODEGEN_RUNTIME_METADATA_H_

#include <tvm/ffi/extra/module.h>
#include <tvm/ir/module.h>
#include <tvm/tirx/function.h>

#include "runtime/metadata.h"

namespace tvm {
namespace ascend {
ffi::Module
AscendModuleCreate(ffi::Bytes code, ffi::String fmt,
                   ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                   ffi::Map<ffi::String, ffi::String> source);
} // namespace ascend
namespace codegen {
ffi::Map<ffi::String, runtime::FunctionInfo>
ExtractAscendFuncInfo(const IRModule &mod);
} // namespace codegen
} // namespace tvm

#endif // TVM_TL_ASCEND_CODEGEN_RUNTIME_METADATA_H_
