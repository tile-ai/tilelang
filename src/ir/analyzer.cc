/*!
 * \file tl/ir/analyzer.cc
 * \brief Native Analyzer construction with TileLang's proof policy.
 */

#include "analyzer.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl {

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.CreateAnalyzer", CreateTileLangAnalyzer);
}

} // namespace tvm::tl
