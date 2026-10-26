#ifndef TILELANG_TRANSFORM_COMMON_POINTER_VALUE_TYPE_H_
#define TILELANG_TRANSFORM_COMMON_POINTER_VALUE_TYPE_H_

#include <tvm/ir/transform.h>
#include <tvm/tirx/function.h>

namespace tvm::tl {

tirx::Function RewritePointerValueTypes(tirx::Function function,
                                        bool storage_only);
tvm::transform::Pass RewritePointerValueTypes();

} // namespace tvm::tl
#endif
