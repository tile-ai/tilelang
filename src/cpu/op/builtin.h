/*!
 * \file tl/cpu/op/builtin.h
 * \brief CPU-specific launch attributes and pass configuration keys.
 */

#ifndef TVM_TL_CPU_OP_BUILTIN_H_
#define TVM_TL_CPU_OP_BUILTIN_H_

#include "op/builtin.h"

namespace tvm {
namespace tl {

namespace attr {

// Grid axis in outer-to-inner launch order.
constexpr const char *kCPUGridDim = "tl.cpu_grid_dim";

// Atomic kernels require a serial grid because CPU atomics lower to plain RMW.
constexpr const char *kCPUHadAtomics = "tl.cpu_had_atomics";

// Per-launch OpenMP thread count, lowered onto the outer grid loop.
constexpr const char *kCPUNumThreads = "tl.cpu_num_threads";

} // namespace attr

static constexpr const char *kCPUParallel = "tl.cpu_parallel";
static constexpr const char *kCPUParallelMinTrip = "tl.cpu_parallel_min_trip";

} // namespace tl
} // namespace tvm

#endif // TVM_TL_CPU_OP_BUILTIN_H_
