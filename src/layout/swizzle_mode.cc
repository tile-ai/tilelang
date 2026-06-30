/*!
 * \file swizzle_mode.cc
 * \brief Registration and FFI surface for tl::SwizzleMode.
 */

#include "swizzle_mode.h"

#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/creator.h>
#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tl {

// Register the enum type and its variants. Declaration order fixes the dense
// ordinals (0..3), which equal the CU_TENSOR_MAP_SWIZZLE_* values.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // EnumObj subclasses have no __ffi_init__; allocate via init(false).
  refl::ObjectDef<SwizzleModeObj>(
      refl::init(false)); // NOLINT(bugprone-unused-raii)
  refl::TypeAttrDef<SwizzleModeObj>().def(
      refl::type_attr::kConvert,
      &refl::details::FFIConvertFromAnyViewToObjectRef<SwizzleMode>);
  refl::EnumDef<SwizzleModeObj>("NONE");         // ordinal 0
  refl::EnumDef<SwizzleModeObj>("SWIZZLE_32B");  // ordinal 1
  refl::EnumDef<SwizzleModeObj>("SWIZZLE_64B");  // ordinal 2
  refl::EnumDef<SwizzleModeObj>("SWIZZLE_128B"); // ordinal 3
}

// Projection helpers exposed to Python so the descriptor-field encodings live
// in exactly one place (here).
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.swizzle_mode_wgmma_layout_type",
           [](const SwizzleMode &m) { return m.WgmmaLayoutType(); })
      .def("tl.swizzle_mode_tcgen05_layout_type",
           [](const SwizzleMode &m) { return m.Tcgen05LayoutType(); })
      .def("tl.swizzle_mode_byte_width",
           [](const SwizzleMode &m) { return m.ByteWidth(); })
      .def("tl.swizzle_mode_smem_alignment",
           [](const SwizzleMode &m) { return m.SmemAlignment(); })
      .def("tl.swizzle_mode_from_ordinal",
           [](int64_t ordinal) { return SwizzleMode::FromOrdinal(ordinal); });
}

} // namespace tl
} // namespace tvm
