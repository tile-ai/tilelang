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

// dav-3510 default usable Unified Buffer. Physical capacity is 256 KiB;
// the reserves below stay reserved unless a compile flag releases them.
constexpr int64_t kAscendUBMemoryBytes = 248 * 1024;
// Released by --cce-disable-vf-stack-reserved-ubuf (SIMD VF spill area).
constexpr int64_t kAscendVFStackReserveBytes = 6 * 1024;
// Released by --cce-disable-asc-reserved-ubuf (Ascend C API area).
constexpr int64_t kAscendApiReserveBytes = 2 * 1024;
constexpr int64_t kAscendSimtContextBytes = 32 * 1024;
constexpr int64_t kAscendL1MemoryBytes = 512 * 1024;
constexpr int64_t kAscendL0ABMemoryBytes = 64 * 1024;
constexpr int64_t kAscendL0CMemoryBytes = 256 * 1024;
constexpr int64_t kAscendSSMemoryBytes = 3 * 1024;

bool TargetIsAscend(Target target);
// Whether the target selects PTO codegen within the Ascend architecture
// ("ascend" kind + "pto" in target keys). Mirrors Python target_is_pto.
bool TargetIsPTO(Target target);
bool IsAscendVectorizableFP8(DataType dtype);

/*!
 * \brief Bytes added to the default UB budget by the current compile flags.
 *
 * `--cce-disable-vf-stack-reserved-ubuf` releases 6 KiB and
 * `--cce-disable-asc-reserved-ubuf` releases 2 KiB. Each flag matches a whole
 * whitespace-separated token in `tl.device_compile_flags`. An absent config
 * releases nothing.
 */
int64_t AscendReleasedUBReserveBytes();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TARGET_UTILS_H_
