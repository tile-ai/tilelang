/*!
 * \file rt_mod_metal.cc
 * \brief Metal codegen entry point.
 *
 * Metal codegen is implemented in backend/common/codegen/codegen_metal.cc,
 * which handles simdgroup types, intrinsics, and MSL emission. This file exists
 * to satisfy the metal/CMakeLists.txt dependency but delegates to the main
 * implementation.
 */
#include "support/check.h"

namespace tvm {
namespace codegen {

// Metal codegen entry point is in backend/common/codegen/codegen_metal.cc.

} // namespace codegen
} // namespace tvm
