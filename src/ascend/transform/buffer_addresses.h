/*!
 * \file buffer_addresses.h
 * \brief Exact byte placement of Ascend on-chip storages.
 */

#ifndef TVM_TL_ASCEND_TRANSFORM_BUFFER_ADDRESSES_H_
#define TVM_TL_ASCEND_TRANSFORM_BUFFER_ADDRESSES_H_

#include <string>

#include <tvm/ffi/container/map.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>

#include "ascend/target_utils.h"
#include "support/check.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

constexpr const char *kBufferAddresses = "tl.ascend_buffer_addresses";
using BufferAddresses = ffi::Map<tirx::Var, Integer>;

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
      << kBufferAddresses << " does not support storage scope " << scope
      << "; expected UB, L1, L0A, L0B, L0C or SS allocations. "
      << "Place bound L0 scale-factor handles through their data buffer.";
  return {};
}

inline BufferAddresses ParseBufferAddresses(const ffi::Any &value) {
  auto addresses = value.try_cast<BufferAddresses>();
  ICHECK(addresses.has_value())
      << kBufferAddresses
      << " must map storage Vars to constant byte addresses";
  ICHECK(!addresses.value().empty())
      << kBufferAddresses << " must not be empty";
  for (const auto &[storage, address] : addresses.value()) {
    std::string scope = GetPtrStorageScope(storage);
    auto info = GetBufferPlacementInfo(scope);
    ICHECK(address->dtype.is_int() && address->value >= 0 &&
           address->value % info.alignment == 0)
        << kBufferAddresses << " requires non-negative, " << info.alignment
        << "-byte aligned integer addresses for " << scope << "; storage "
        << storage << " has " << address;
    ICHECK_LT(address->value, info.capacity)
        << "Buffer placement for " << storage << " exceeds " << scope
        << " capacity of " << info.capacity << " bytes";
  }
  return addresses.value();
}

inline BufferAddresses
RemapBufferAddresses(const BufferAddresses &addresses,
                     const ffi::Map<tirx::Var, tirx::Var> &storage_remap) {
  BufferAddresses result;
  for (const auto &[storage, address] : addresses) {
    tirx::Var remapped = storage_remap.Get(storage).value_or(storage);
    auto existing = result.Get(remapped);
    ICHECK(!existing.has_value() || existing.value()->value == address->value)
        << "Conflicting " << kBufferAddresses << " after remapping storage "
        << remapped;
    result.Set(remapped, address);
  }
  return result;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_BUFFER_ADDRESSES_H_
