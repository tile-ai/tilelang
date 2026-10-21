/*!
 * \file tl/ascend/target_utils.h
 * \brief Ascend target attribute helpers.
 */

#ifndef TVM_TL_ASCEND_TARGET_UTILS_H_
#define TVM_TL_ASCEND_TARGET_UTILS_H_

#include <cstdint>

#include <tvm/runtime/data_type.h>
#include <tvm/target/target.h>

namespace tvm {
namespace tl {

constexpr int64_t kAscendUBMemoryBytes = 248 * 1024;
constexpr int64_t kAscendSimtContextBytes = 32 * 1024;
constexpr int64_t kAscendL1MemoryBytes = 512 * 1024;
constexpr int64_t kAscendL0ABMemoryBytes = 64 * 1024;
constexpr int64_t kAscendL0CMemoryBytes = 256 * 1024;
constexpr int64_t kAscendSSMemoryBytes = 3 * 1024;

bool TargetIsAscend(Target target);
bool IsAscendVectorizableFP8(DataType dtype);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TARGET_UTILS_H_
