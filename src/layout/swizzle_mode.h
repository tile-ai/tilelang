/*!
 * \file swizzle_mode.h
 * \brief Canonical shared-memory swizzle mode, as a cross-language FFI enum.
 *
 * One enum, used by both C++ and Python, with no raw-int encodings at the FFI
 * boundary. The dense ordinal encoding matches the CUDA driver's
 * ``CU_TENSOR_MAP_SWIZZLE_*`` values:
 *
 *   NONE = 0, SWIZZLE_32B = 1, SWIZZLE_64B = 2, SWIZZLE_128B = 3
 *
 * The hardware *descriptor field* encodings (WGMMA ``layout_type_`` and
 * TCGEN05) use a different, reversed numbering; those are exposed as derived
 * projections (WgmmaLayoutType / Tcgen05LayoutType), NOT as the stored ordinal.
 */

#ifndef TVM_TL_LAYOUT_SWIZZLE_MODE_H_
#define TVM_TL_LAYOUT_SWIZZLE_MODE_H_

#include <tvm/ffi/cast.h>
#include <tvm/ffi/enum.h>
#include <tvm/ffi/object.h>

#include "support/check.h"

namespace tvm {
namespace tl {

// A canonical swizzle granularity, registered as an FFI enum (see
// swizzle_mode.cc). Variants are process-wide unique singletons, so ``==`` /
// ``same_as`` is identity comparison and ``->_value`` is the dense ordinal.
class SwizzleModeObj : public ffi::EnumObj {
public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SwizzleMode", SwizzleModeObj,
                                    ffi::EnumObj);
};

class SwizzleMode : public ffi::Enum {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SwizzleMode, ffi::Enum,
                                             SwizzleModeObj);

  // Singleton accessors. Ordinals are fixed by registration order in
  // swizzle_mode.cc and equal the CU_TENSOR_MAP_SWIZZLE_* values.
  static SwizzleMode None() { return Get("NONE"); }
  static SwizzleMode Swizzle32B() { return Get("SWIZZLE_32B"); }
  static SwizzleMode Swizzle64B() { return Get("SWIZZLE_64B"); }
  static SwizzleMode Swizzle128B() { return Get("SWIZZLE_128B"); }

  // Recover the singleton from its canonical ordinal (0..3). This is exactly
  // the CU_TENSOR_MAP_SWIZZLE_* value.
  static SwizzleMode FromOrdinal(int64_t ordinal) {
    switch (ordinal) {
    case 0:
      return None();
    case 1:
      return Swizzle32B();
    case 2:
      return Swizzle64B();
    case 3:
      return Swizzle128B();
    default:
      ICHECK(false) << "Invalid SwizzleMode ordinal: " << ordinal;
      return None();
    }
  }

  // The dense ordinal == CU_TENSOR_MAP_SWIZZLE_* value (0/1/2/3).
  int CanonicalOrdinal() const {
    return static_cast<int>(operator->()->_value);
  }

  bool IsNone() const { return CanonicalOrdinal() == 0; }

  // WGMMA descriptor ``layout_type_`` field: none->0, 32B->3, 64B->2, 128B->1.
  int WgmmaLayoutType() const {
    int o = CanonicalOrdinal();
    return o == 0 ? 0 : 4 - o;
  }

  // TCGEN05 descriptor swizzle field: none->0, 32B->6, 64B->4, 128B->2.
  int Tcgen05LayoutType() const {
    int o = CanonicalOrdinal();
    return o == 0 ? 0 : 2 * (4 - o);
  }

  // Swizzle period in bytes: none->1, 32B->32, 64B->64, 128B->128.
  int ByteWidth() const {
    int o = CanonicalOrdinal();
    return o == 0 ? 1 : (16 << o);
  }

  // Required shared-memory base alignment (bytes): the base must lie on a
  // swizzle-pattern repeat boundary (swizzle byte width x 8 rows) or the
  // hardware applies the swizzle with a phase shift -> silently wrong data.
  // none->128 (bulk-copy base requirement), 32B->256, 64B->512, 128B->1024.
  int SmemAlignment() const { return 128 << CanonicalOrdinal(); }

private:
  static SwizzleMode Get(const ffi::String &name) {
    ffi::Enum e = ffi::EnumObj::Get<SwizzleModeObj>(name);
    const auto *node = e.as<SwizzleModeObj>();
    ICHECK(node != nullptr)
        << "SwizzleMode entry `" << name << "` is not a SwizzleModeObj";
    return ffi::GetRef<SwizzleMode>(node);
  }
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_LAYOUT_SWIZZLE_MODE_H_
