#ifndef TILELANG_IR_LOOP_ANNOTATIONS_H_
#define TILELANG_IR_LOOP_ANNOTATIONS_H_

#include <functional>
#include <tvm/ir/stmt.h>

namespace tvm::tl {
// Analysis only: body-local operands do not become legal at the For definition.
TVM_TIRX_DLL ffi::Map<ffi::String, ffi::Any>
ReadLoopAnnotations(const ForObj *loop);
using LoopAnnotationRewrite = std::function<ffi::Map<ffi::String, ffi::Any>(
    const ffi::Map<ffi::String, ffi::Any> &)>;
// Rewrite at each original definition site; reject escaping local operands.
TVM_TIRX_DLL For RewriteLoopAnnotationCarriers(
    const For &loop, const LoopAnnotationRewrite &rewrite);
TVM_TIRX_DLL For AppendLoopAnnotation(
    const For &loop, const ffi::Map<ffi::String, ffi::Any> &fields);
} // namespace tvm::tl
#endif // TILELANG_IR_LOOP_ANNOTATIONS_H_
