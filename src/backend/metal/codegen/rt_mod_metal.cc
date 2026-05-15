/*!
 * \file rt_mod_metal.cc
 * \brief Metal codegen entry point.
 *
 * Metal codegen is handled by CodeGenCHost (target/codegen_c_host.cc), which
 * has built-in Metal context support via the is_in_metal_context flag.
 * When IR contains AttrStmt with attr_key == "metal_context", the host
 * codegen emits Metal-specific dispatch_sync / MTLCommandBuffer code.
 */
#include "target/codegen_c_host.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace codegen {

ffi::Module BuildTileLangMetal(IRModule mod, Target target) {
  return tl::BuildTileLangCHost(mod, target);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.tilelang_metal", BuildTileLangMetal);
}

} // namespace codegen
} // namespace tvm
