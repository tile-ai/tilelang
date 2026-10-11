/*!
 * \file thread_sync_types.h
 */
#ifndef TVM_TL_THREAD_BOUND_KEY_H_
#define TVM_TL_THREAD_BOUND_KEY_H_

#include <cstdint>
#include <functional>

namespace tvm {
namespace tl {

// There are 16 Named Barriers provided by Hardware starting in Hopper
// Their IDs are in the range 0-15
// Number of threads syncing using the barrier must be a multiple of warp-size
// ID 0 should not be used for safety, as other driver APIs (i.e. __syncthreads)
// may use it and conflict with other uses.
enum class ReservedNamedBarriers : uint8_t {
  kSyncThreads = 0,
  kReduce_0 = 1,
  kReduce_1 = 2,
  kFirstUsedBarrier = kReduce_1 + 1
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_THREAD_BOUND_KEY_H_
