#ifndef TILELANG_TRANSFORM_COMMON_READ_ONLY_PARAMS_H_
#define TILELANG_TRANSFORM_COMMON_READ_ONLY_PARAMS_H_

#include <tvm/tirx/function.h>

namespace tvm::tl {

ffi::Array<IntImm> ReadOnlyParameterIndices(const tirx::Function &function);

}

#endif
