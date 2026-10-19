/*!
 * \file tl/cpu/op/builtin.cc
 * \brief Registration of CPU-specific TileLang pass configurations.
 */

#include "builtin.h"

#include <tvm/ir/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;

TVM_REGISTER_PASS_CONFIG_OPTION(kCPUParallel, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kCPUParallelMinTrip, Integer);

} // namespace tl
} // namespace tvm
