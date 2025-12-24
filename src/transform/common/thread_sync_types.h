/*!
 * \file thread_sync_types.h
 */
#ifndef TVM_TL_THREAD_BOUND_KEY_H_
#define TVM_TL_THREAD_BOUND_KEY_H_

#include <cstdint>
#include <functional>

namespace tvm {
namespace tl {

struct ThreadBoundKey {
  int64_t tx_min, tx_max, ty_min, ty_max, tz_min, tz_max;
  bool operator==(const ThreadBoundKey &other) const {
    return tx_min == other.tx_min && tx_max == other.tx_max &&
           ty_min == other.ty_min && ty_max == other.ty_max &&
           tz_min == other.tz_min && tz_max == other.tz_max;
  }
};

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

namespace std {
template <> struct hash<tvm::tl::ThreadBoundKey> {
  size_t operator()(const tvm::tl::ThreadBoundKey &k) const {
    size_t h = std::hash<int64_t>()(k.tx_min);
    h = h * 31 + std::hash<int64_t>()(k.tx_max);
    h = h * 31 + std::hash<int64_t>()(k.ty_min);
    h = h * 31 + std::hash<int64_t>()(k.ty_max);
    h = h * 31 + std::hash<int64_t>()(k.tz_min);
    h = h * 31 + std::hash<int64_t>()(k.tz_max);
    return h;
  }
};
} // namespace std

#endif // TVM_TL_THREAD_BOUND_KEY_H_
