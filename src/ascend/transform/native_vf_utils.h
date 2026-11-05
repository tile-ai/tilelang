/*!
 * \file ascend/transform/native_vf_utils.h
 * \brief Recognize Ascend VF scopes in native TileLang region metadata.
 */
#ifndef TILELANG_ASCEND_TRANSFORM_NATIVE_VF_UTILS_H_
#define TILELANG_ASCEND_TRANSFORM_NATIVE_VF_UTILS_H_

#include "ir/annotations.h"
#include <tvm/ir/stmt.h>

namespace tvm::tl {

enum class AscendVFKind { kNone, kSimd, kSimt };

inline ffi::String AscendVFString(ffi::AnyView value) {
  if (const auto *literal = value.as<StringImmObj>())
    return literal->value;
  return value.cast<ffi::String>();
}

inline bool AscendVFBoolean(ffi::AnyView value) {
  if (auto boolean = value.as<bool>())
    return *boolean;
  if (const auto *literal = value.as<IntImmObj>()) {
    TVM_FFI_CHECK(literal->value == 0 || literal->value == 1, ValueError)
        << "Ascend VF scope annotation must be a boolean";
    return literal->value == 1;
  }
  auto integer = value.cast<int64_t>();
  TVM_FFI_CHECK(integer == 0 || integer == 1, ValueError)
      << "Ascend VF scope annotation must be a boolean";
  return integer != 0;
}

// The frontend's named sblock and the normalized lexical marker denote the
// same scope. A coincidentally named variable or ordinary attribute does not.
inline AscendVFKind GetAscendVFKind(const RegionStmtObj *region) {
  if (!region || region->op->name != "tl.attr")
    return AscendVFKind::kNone;
  TVM_FFI_CHECK_EQ(region->args.size(), 1U, ValueError);
  TVM_FFI_CHECK_EQ(region->attrs.GetAttr<int64_t>("schema_version").value_or(0),
                   1, ValueError);
  auto fields =
      DecodeAnnotations(
          region->args[0].as_or_throw<tvm::Tuple>(),
          region->attrs.GetAttr<ffi::Array<ffi::Any>>("annotation_schema")
              .value())
          .cast<ffi::Map<ffi::String, ffi::Any>>();
  const auto key = AscendVFString(fields.at("key"));
  bool simd = key == "tl.simdvf_scope";
  bool simt = key == "tl.simtvf_scope";
  if (key == "tl.sblock") {
    auto block = fields.at("value").cast<ffi::Map<ffi::String, ffi::Any>>();
    const auto name = AscendVFString(block.at("name"));
    simd = name == "SIMD_VF";
    simt = name == "SIMT_VF";
    if (auto metadata = block.Get("annotations")) {
      auto annotations = metadata->cast<ffi::Map<ffi::String, ffi::Any>>();
      if (auto value = annotations.Get("tl.simdvf_scope"))
        simd |= AscendVFBoolean(*value);
      if (auto value = annotations.Get("tl.simtvf_scope"))
        simt |= AscendVFBoolean(*value);
    }
  }
  TVM_FFI_CHECK(!(simd && simt), ValueError)
      << "An Ascend scope cannot be both SIMD_VF and SIMT_VF";
  if (simd || simt) {
    TVM_FFI_CHECK(region->body_params.empty() && region->result_vars.empty(),
                  ValueError)
        << "Ascend VF metadata cannot introduce hidden region bindings";
  }
  return simd   ? AscendVFKind::kSimd
         : simt ? AscendVFKind::kSimt
                : AscendVFKind::kNone;
}

} // namespace tvm::tl
#endif // TILELANG_ASCEND_TRANSFORM_NATIVE_VF_UTILS_H_
