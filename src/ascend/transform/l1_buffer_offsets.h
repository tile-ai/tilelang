/*!
 * \file l1_buffer_offsets.h
 * \brief Exact byte placement of complete Ascend L1 storages.
 */

#ifndef TVM_TL_ASCEND_TRANSFORM_L1_BUFFER_OFFSETS_H_
#define TVM_TL_ASCEND_TRANSFORM_L1_BUFFER_OFFSETS_H_

#include <tvm/ffi/container/map.h>
#include <tvm/tirx/expr.h>

#include "support/check.h"

namespace tvm {
namespace tl {

constexpr const char *kL1BufferOffsets = "tl.ascend_l1_offsets";
using L1BufferOffsets = ffi::Map<tirx::Var, Integer>;

inline L1BufferOffsets ParseL1BufferOffsets(const ffi::Any &value) {
  auto offsets = value.try_cast<L1BufferOffsets>();
  ICHECK(offsets.has_value())
      << kL1BufferOffsets << " must map storage Vars to constant byte offsets";
  ICHECK(!offsets.value().empty()) << kL1BufferOffsets << " must not be empty";
  for (const auto &[storage, offset] : offsets.value()) {
    ICHECK(offset->dtype.is_int() && offset->value >= 0 &&
           offset->value % 512 == 0)
        << kL1BufferOffsets << " requires non-negative, 512-byte aligned "
        << "integer offsets; storage " << storage << " has " << offset;
  }
  return offsets.value();
}

inline L1BufferOffsets
RemapL1BufferOffsets(const L1BufferOffsets &offsets,
                     const ffi::Map<tirx::Var, tirx::Var> &storage_remap) {
  L1BufferOffsets result;
  for (const auto &[storage, offset] : offsets) {
    tirx::Var remapped = storage_remap.Get(storage).value_or(storage);
    auto existing = result.Get(remapped);
    ICHECK(!existing.has_value() || existing.value()->value == offset->value)
        << "Conflicting " << kL1BufferOffsets << " after remapping storage "
        << remapped;
    result.Set(remapped, offset);
  }
  return result;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_L1_BUFFER_OFFSETS_H_
