/*!
 * \file tl/cpu/op/builtin.cc
 * \brief Registration of CPU-specific TileLang pass configurations.
 */

#include "builtin.h"

#include <tvm/ir/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;

TVM_FFI_STATIC_INIT_BLOCK() {
  transform::PassContext::RegisterConfigOption<bool>(kCPUParallel);
  transform::PassContext::RegisterConfigOption<int64_t>(kCPUParallelMinTrip);
}

} // namespace tl
} // namespace tvm
