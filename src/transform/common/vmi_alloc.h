/*!
 * \file vmi_alloc.h
 * \brief Shared decoding and validation of VMI allocation markers.
 *
 * The marker contract is defined in attr.h.  This header owns the shared
 * parser/contract checks used by early unroll, LowerOpaqueBlock,
 * NormalizePTOAllocMarkers and Ascend PTO code generation, plus the
 * whole-function statement collector consumed only by NormalizePTOAllocMarkers
 * and Ascend PTO code generation.  It must not depend on Ascend code
 * generation.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_VMI_ALLOC_H_
#define TVM_TL_TRANSFORM_COMMON_VMI_ALLOC_H_

#include "support/check.h"
#include "transform/common/attr.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/map.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace tl {
namespace vmi {

// Owning maps: cloned/intermediate Vars stay alive for the whole pass even
// after the IR node that first produced them is replaced.
using AllocKindMap =
    std::unordered_map<tirx::Var, ffi::String, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>;
using AllocationList =
    std::unordered_map<tirx::Var, std::vector<tirx::Buffer>, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>;

/*!
 * \brief Marked allocations grouped by their data Var.
 *
 * `kinds` contains only data Vars with an active allocation.  `allocations`
 * contains every actual allocation/view that shares a marked data Var.
 */
struct AllocMarkers {
  AllocKindMap kinds;
  AllocationList allocations;
};

/*!
 * \brief Validate one allocation kind value.
 * \param kind The encoded kind.
 * \param context Caller description used in diagnostics.
 * \return The validated kind.
 */
inline ffi::String CheckAllocKind(const ffi::String &kind,
                                  const char *context) {
  ICHECK(kind == tl::attr::kVmiAllocKindData ||
         kind == tl::attr::kVmiAllocKindDataArray)
      << context << " supports only `" << tl::attr::kVmiAllocKindData
      << "` and `" << tl::attr::kVmiAllocKindDataArray << "` for `"
      << tl::attr::kVmiAllocKind << "`, got `" << kind << "`";
  return kind;
}

/*!
 * \brief Check whether a dtype can back a mutable VMI data slot.
 * \param dtype The allocation (possibly vector) dtype.
 * \return true for numeric, non-bool element types supported by local.var.
 */
inline bool IsDataElementType(DataType dtype) {
  DataType element = dtype.is_vector() ? dtype.element_of() : dtype;
  if (element.is_bool() || element.is_handle()) {
    return false;
  }
  if (element.is_float4_e2m1fn() || element.is_float8_e4m3() ||
      element.is_float8_e4m3fn() || element.is_float8_e5m2() ||
      element.is_bfloat16()) {
    return true;
  }
  if (element.is_float()) {
    return element.bits() == 16 || element.bits() == 32;
  }
  if (element.is_int() || element.is_uint()) {
    return element.bits() == 8 || element.bits() == 16 ||
           element.bits() == 32 || element.bits() == 64;
  }
  return false;
}

/*!
 * \brief Validate the storage contract of one marked allocation/view.
 * \param buffer The actual allocation or view.
 * \param context Caller description used in diagnostics.
 */
inline void ValidateDataSlot(const tirx::BufferNode *buffer,
                             const char *context) {
  ICHECK(buffer != nullptr) << context << " received an undefined allocation";
  ffi::String scope = ffi::GetRef<tirx::Buffer>(buffer).scope();
  ICHECK(scope == "local.var")
      << context << " `" << tl::attr::kVmiAllocKind
      << "` may only mark a local.var allocation, got scope `" << scope
      << "` for buffer `" << buffer->name << "`";
  ICHECK_EQ(buffer->shape.size(), 1U)
      << context << " `" << tl::attr::kVmiAllocKind
      << "` requires a single-element slot, got rank " << buffer->shape.size()
      << " for buffer `" << buffer->name << "`";
  const auto *extent = buffer->shape[0].as<tirx::IntImmNode>();
  ICHECK(extent != nullptr && extent->value == 1)
      << context << " `" << tl::attr::kVmiAllocKind
      << "` requires shape (1,), got " << buffer->shape[0] << " for buffer `"
      << buffer->name << "`";
  ICHECK(IsDataElementType(buffer->dtype))
      << context << " `" << tl::attr::kVmiAllocKind
      << "` requires a numeric non-bool element type, got " << buffer->dtype
      << " for buffer `" << buffer->name << "`";
}

/*!
 * \brief Check whether a dtype can back a VMI register array.
 * \param dtype The allocation (vector) dtype.
 * \return true for numeric, non-bool, non-handle element types.
 *
 * Unlike IsDataElementType this keeps every numeric width and lane count the
 * frontend can construct; instruction-level limits stay with each operation.
 */
inline bool IsDataArrayElementType(DataType dtype) {
  DataType element = dtype.is_vector() ? dtype.element_of() : dtype;
  if (element.is_bool() || element.is_handle() || element.is_void()) {
    return false;
  }
  return element.is_int() || element.is_uint() || element.is_float() ||
         element.is_bfloat() || element.is_float8() || element.is_float6() ||
         element.is_float4() || element.is_tfloat();
}

/*!
 * \brief Validate the storage contract of one marked register array allocation.
 * \param buffer The actual allocation.
 * \param context Caller description used in diagnostics.
 *
 * Access views are checked separately by the consumers: flattening may change
 * their rank and shape, so only the actual allocation must keep a positive,
 * static, multidimensional shape.
 */
inline void ValidateDataArray(const tirx::BufferNode *buffer,
                              const char *context) {
  ICHECK(buffer != nullptr) << context << " received an undefined allocation";
  ffi::String scope = ffi::GetRef<tirx::Buffer>(buffer).scope();
  ICHECK(scope == "local") << context << " `" << tl::attr::kVmiAllocKind
                           << "` `" << tl::attr::kVmiAllocKindDataArray
                           << "` may only mark a local allocation, got scope `"
                           << scope << "` for buffer `" << buffer->name << "`";
  ICHECK(buffer->dtype.lanes() > 1)
      << context << " `" << tl::attr::kVmiAllocKind << "` `"
      << tl::attr::kVmiAllocKindDataArray
      << "` requires a multi-lane vector element dtype, got " << buffer->dtype
      << " for buffer `" << buffer->name << "`";
  ICHECK(!buffer->shape.empty())
      << context << " `" << tl::attr::kVmiAllocKind << "` `"
      << tl::attr::kVmiAllocKindDataArray
      << "` requires at least one array dimension, got a rank-0 allocation "
         "for buffer `"
      << buffer->name << "`";
  for (const PrimExpr &dim : buffer->shape) {
    const auto *extent = dim.as<tirx::IntImmNode>();
    ICHECK(extent != nullptr && extent->value > 0)
        << context << " `" << tl::attr::kVmiAllocKind << "` `"
        << tl::attr::kVmiAllocKindDataArray
        << "` requires every array dimension to be a positive compile-time "
           "integer, got "
        << dim << " for buffer `" << buffer->name << "`";
  }
  ICHECK(IsDataArrayElementType(buffer->dtype))
      << context << " `" << tl::attr::kVmiAllocKind << "` `"
      << tl::attr::kVmiAllocKindDataArray
      << "` requires a numeric non-bool element type, got " << buffer->dtype
      << " for buffer `" << buffer->name << "`";
}

/*!
 * \brief Validate one marked allocation against the contract of its kind.
 * \param buffer The actual allocation.
 * \param kind The allocation kind.
 * \param context Caller description used in diagnostics.
 */
inline void ValidateAllocation(const tirx::BufferNode *buffer,
                               const ffi::String &kind, const char *context) {
  ffi::String checked = CheckAllocKind(kind, context);
  if (checked == tl::attr::kVmiAllocKindDataArray) {
    ValidateDataArray(buffer, context);
    return;
  }
  ValidateDataSlot(buffer, context);
}

/*!
 * \brief Merge one allocation kind into a data Var -> kind table.
 * \param kinds The accumulated owning table.
 * \param data The allocation data Var (kept alive by the table).
 * \param kind The validated kind.
 * \param context Caller description used in diagnostics.
 */
inline void MergeAllocKind(AllocKindMap *kinds, const tirx::Var &data,
                           ffi::String kind, const char *context) {
  ICHECK(kinds != nullptr && data.defined())
      << context << " received an invalid `" << tl::attr::kVmiAllocKind
      << "` entry";
  auto it = kinds->find(data);
  if (it == kinds->end()) {
    kinds->emplace(data, std::move(kind));
    return;
  }
  ICHECK(it->second == kind)
      << context << " has conflicting `" << tl::attr::kVmiAllocKind
      << "` entries for one allocation: `" << it->second << "` vs `" << kind
      << "`";
}

/*!
 * \brief Decode a block-level `Var -> String` allocation kind map.
 * \param value The annotation value.
 * \param kinds The accumulated owning table.
 * \param context Caller description used in diagnostics.
 */
inline void CollectAllocKindMap(const ffi::Any &value, AllocKindMap *kinds,
                                const char *context) {
  auto map = value.try_cast<ffi::Map<ffi::Any, ffi::Any>>();
  ICHECK(map.has_value())
      << context << " expects `" << tl::attr::kVmiAllocKind
      << "` to be a map from allocation data Vars to strings, got "
      << value.GetTypeKey();
  for (const auto &kv : map.value()) {
    const auto *data = kv.first.as<tirx::VarNode>();
    ICHECK(data != nullptr) << context << " `" << tl::attr::kVmiAllocKind
                            << "` map keys must be allocation data Vars, got "
                            << kv.first.GetTypeKey();
    auto kind = kv.second.try_cast<ffi::String>();
    ICHECK(kind.has_value())
        << context << " `" << tl::attr::kVmiAllocKind
        << "` map values must be strings, got " << kv.second.GetTypeKey();
    CheckAllocKind(kind.value(), context);
    MergeAllocKind(kinds, ffi::GetRef<tirx::Var>(data), kind.value(), context);
  }
}

/*!
 * \brief Read and validate an allocation-level kind annotation.
 * \param annotations The allocation annotations.
 * \param context Caller description used in diagnostics.
 * \return The kind when present.
 */
inline std::optional<ffi::String>
FindAllocKind(const ffi::Map<ffi::String, ffi::Any> &annotations,
              const char *context) {
  auto it = annotations.find(tl::attr::kVmiAllocKind);
  if (it == annotations.end()) {
    return std::nullopt;
  }
  auto kind = (*it).second.try_cast<ffi::String>();
  ICHECK(kind.has_value()) << context << " allocation `"
                           << tl::attr::kVmiAllocKind
                           << "` must be a string, got "
                           << (*it).second.GetTypeKey();
  return CheckAllocKind(kind.value(), context);
}

/*!
 * \brief Collect marked allocations from a statement.
 *
 * Both block-map and allocation-annotation forms are accepted.  A map entry
 * without an active allocation is a residual entry and is ignored.  Only data
 * Vars that are actually marked get contract checks; unmarked allocations
 * keep their previous, unconstrained behavior.  Every allocation/view that
 * shares a marked data Var is validated, and differing dtypes/lanes are
 * diagnosed.
 */
inline void CollectAllocMarkers(const tirx::Stmt &stmt, AllocMarkers *markers,
                                const char *context) {
  ICHECK(markers != nullptr) << context << " received no marker storage";
  markers->kinds.clear();
  markers->allocations.clear();

  AllocationList all_allocations;
  tirx::PostOrderVisit(stmt, [&](const ffi::ObjectRef &node) {
    if (const auto *alloc = node.as<tirx::AllocBufferNode>()) {
      all_allocations[alloc->buffer->data].push_back(alloc->buffer);
      return;
    }
    const auto *block = node.as<tirx::SBlockNode>();
    if (block == nullptr) {
      return;
    }
    for (const tirx::Buffer &buffer : block->alloc_buffers) {
      all_allocations[buffer->data].push_back(buffer);
    }
  });

  tirx::PostOrderVisit(stmt, [&](const ffi::ObjectRef &node) {
    if (const auto *alloc = node.as<tirx::AllocBufferNode>()) {
      auto kind = FindAllocKind(alloc->annotations, context);
      if (kind.has_value()) {
        MergeAllocKind(&markers->kinds, alloc->buffer->data, kind.value(),
                       context);
      }
      return;
    }
    const auto *block = node.as<tirx::SBlockNode>();
    if (block == nullptr) {
      return;
    }
    auto annotation = block->annotations.find(tl::attr::kVmiAllocKind);
    if (annotation == block->annotations.end()) {
      return;
    }
    AllocKindMap parsed;
    CollectAllocKindMap((*annotation).second, &parsed, context);
    for (const auto &[data, kind] : parsed) {
      if (all_allocations.find(data) == all_allocations.end()) {
        // Residual entry whose allocation no longer exists.  It is never
        // matched by name and is removed with the consumed block map.
        continue;
      }
      MergeAllocKind(&markers->kinds, data, kind, context);
    }
  });

  for (const auto &[data, kind] : markers->kinds) {
    auto allocation = all_allocations.find(data);
    ICHECK(allocation != all_allocations.end())
        << context << " `" << tl::attr::kVmiAllocKind
        << "` marks a data Var without an active allocation";
    const std::vector<tirx::Buffer> &buffers = allocation->second;
    ICHECK(!buffers.empty()) << context << " `" << tl::attr::kVmiAllocKind
                             << "` marks a data Var without an allocation view";
    for (const tirx::Buffer &buffer : buffers) {
      ValidateAllocation(buffer.get(), kind, context);
    }
    for (size_t i = 1; i < buffers.size(); ++i) {
      ICHECK(buffers[i]->dtype == buffers.front()->dtype)
          << context << " marked VMI allocation has conflicting dtype views: "
          << buffers.front()->dtype << " vs " << buffers[i]->dtype;
    }
    markers->allocations[data] = buffers;
  }
}

} // namespace vmi
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_VMI_ALLOC_H_
