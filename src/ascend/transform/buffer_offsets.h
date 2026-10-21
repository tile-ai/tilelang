/*!
 * \file buffer_offsets.h
 * \brief Exact byte placement of Ascend on-chip storages.
 */

#ifndef TVM_TL_ASCEND_TRANSFORM_BUFFER_OFFSETS_H_
#define TVM_TL_ASCEND_TRANSFORM_BUFFER_OFFSETS_H_

#include <string>

#include <tvm/ffi/container/map.h>
#include <tvm/tirx/expr.h>

#include "ascend/target_utils.h"
#include "support/check.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

constexpr const char *kBufferOffsets = "tl.ascend_buffer_offsets";
using BufferOffsets = ffi::Map<tirx::Var, Integer>;

struct BufferPlacementInfo {
  int alignment;
  int64_t capacity;
};

inline BufferPlacementInfo GetBufferPlacementInfo(const std::string &scope) {
  if (scope == "shared.dyn")
    return {32, kAscendUBMemoryBytes};
  if (scope == "shared.l1")
    return {512, kAscendL1MemoryBytes};
  if (scope == "shared.l0a" || scope == "shared.l0b")
    return {512, kAscendL0ABMemoryBytes};
  if (scope == "shared.l0c")
    return {1024, kAscendL0CMemoryBytes};
  if (scope == "shared.ss")
    return {32, kAscendSSMemoryBytes};
  LOG(FATAL)
      << kBufferOffsets << " does not support storage scope " << scope
      << "; expected UB, L1, L0A, L0B, L0C or SS allocations. "
      << "Place bound L0 scale-factor handles through their data buffer.";
  return {};
}

inline BufferOffsets ParseBufferOffsets(const ffi::Any &value) {
  auto offsets = value.try_cast<BufferOffsets>();
  ICHECK(offsets.has_value())
      << kBufferOffsets << " must map storage Vars to constant byte offsets";
  ICHECK(!offsets.value().empty()) << kBufferOffsets << " must not be empty";
  for (const auto &[storage, offset] : offsets.value()) {
    std::string scope = GetPtrStorageScope(storage);
    auto info = GetBufferPlacementInfo(scope);
    ICHECK(offset->dtype.is_int() && offset->value >= 0 &&
           offset->value % info.alignment == 0)
        << kBufferOffsets << " requires non-negative, " << info.alignment
        << "-byte aligned integer offsets for " << scope << "; storage "
        << storage << " has " << offset;
    ICHECK_LT(offset->value, info.capacity)
        << "Buffer placement for " << storage << " exceeds " << scope
        << " capacity of " << info.capacity << " bytes";
  }
  return offsets.value();
}

inline BufferOffsets
RemapBufferOffsets(const BufferOffsets &offsets,
                   const ffi::Map<tirx::Var, tirx::Var> &storage_remap) {
  BufferOffsets result;
  for (const auto &[storage, offset] : offsets) {
    tirx::Var remapped = storage_remap.Get(storage).value_or(storage);
    auto existing = result.Get(remapped);
    ICHECK(!existing.has_value() || existing.value()->value == offset->value)
        << "Conflicting " << kBufferOffsets << " after remapping storage "
        << remapped;
    result.Set(remapped, offset);
  }
  return result;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_BUFFER_OFFSETS_H_
