/*!
 * \brief Internal helper for pipeline buffer multi-versioning.
 * \file multi_version_buffer_rewriter.h
 */

#ifndef TVM_TL_TRANSFORM_MULTI_VERSION_BUFFER_REWRITER_H_
#define TVM_TL_TRANSFORM_MULTI_VERSION_BUFFER_REWRITER_H_

#include <tvm/tirx/function.h>

namespace tvm {
namespace tl {

tirx::PrimFunc ApplyMultiVersionBufferRewriter(tirx::PrimFunc f);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_MULTI_VERSION_BUFFER_REWRITER_H_
