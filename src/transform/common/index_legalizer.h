#ifndef TVM_TL_TRANSFORM_COMMON_INDEX_LEGALIZER_H_
#define TVM_TL_TRANSFORM_COMMON_INDEX_LEGALIZER_H_

#include <tvm/tirx/function.h>

namespace tvm::tl {

tirx::Function LegalizeIndexArithmetic(tirx::Function function);

} // namespace tvm::tl

#endif
