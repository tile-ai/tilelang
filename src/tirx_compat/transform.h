#ifndef TL_TIRX_COMPAT_TRANSFORM_H_
#define TL_TIRX_COMPAT_TRANSFORM_H_

#include <tvm/tirx/function.h>

namespace tvm::tirx {

Stmt ConvertSSA(Stmt stmt);
Function PointerValueTypeRewrite(Function function, bool allow_untyped_pointers,
                                 bool rewrite_tensor_params,
                                 bool rewrite_pointer_params,
                                 bool rewrite_alloc_tensor_node,
                                 bool rewrite_indices, bool rewrite_let_node,
                                 bool rewrite_scalar_read_to_vector_shuffle);

} // namespace tvm::tirx
#endif
