/*!
 * \file rt_mod_metal.cc
 * \brief Metal codegen entry point.
 *
 * Metal codegen is implemented in target/codegen_metal.cc, which handles
 * simdgroup types, intrinsics, and MSL emission.
 * This file exists to satisfy the backend/metal/CMakeLists.txt dependency
 * but delegates to the main implementation.
 */
#include "target/codegen_c_host.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace codegen {

// Metal codegen entry point is in target/codegen_metal.cc.
// This backend path is kept for future migration.

} // namespace codegen
} // namespace tvm
