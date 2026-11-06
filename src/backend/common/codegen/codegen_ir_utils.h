/*!
 * \file backend/common/codegen/codegen_ir_utils.h
 * \brief Typed helpers for the native TileLang source emitters.
 */
#ifndef TILELANG_CODEGEN_IR_UTILS_H_
#define TILELANG_CODEGEN_IR_UTILS_H_

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op/memory.h>

#include <vector>

#include "ir/annotations.h"
#include "ir/dtype.h"

namespace tvm::codegen {

/*! \brief The explicit metadata operands of a TileLang attribute region. */
struct TileLangAttribute {
  ffi::Any node;
  ffi::String key;
  ffi::Any value;
};

inline TileLangAttribute DecodeTileLangAttribute(const RegionStmtObj *region) {
  TVM_FFI_ICHECK_EQ(region->op->name, "tl.attr");
  TVM_FFI_ICHECK_EQ(region->args.size(), 1);
  auto schema =
      region->attrs.GetAttr<ffi::Array<ffi::Any>>("annotation_schema").value();
  auto values =
      tl::DecodeAnnotations(region->args[0].as_or_throw<tvm::Tuple>(), schema)
          .cast<ffi::Map<ffi::Any, ffi::Any>>();
  return {values.at(ffi::String("node")),
          values.at(ffi::String("key")).cast<ffi::String>(),
          values.at(ffi::String("value"))};
}

inline ffi::String CodegenAnnotationString(const ffi::Any &value) {
  if (const auto *literal = value.as<StringImmObj>())
    return literal->value;
  return value.cast<ffi::String>();
}

inline bool CodegenBool(const ffi::Any &value) {
  if (const auto *literal = value.as<IntImmObj>())
    return literal->value != 0;
  return value.cast<bool>();
}
inline tirx::TensorVar CodegenTensor(const VarObj *tensor) {
  return ffi::GetRef<Var>(tensor).as_or_throw<tirx::TensorVar>();
}
inline PrimType CodegenTensorDType(const VarObj *tensor) {
  return CodegenTensor(tensor).dtype();
}
inline std::string CodegenStorageScope(const Var &var) {
  if (const auto *tensor = var->ty.as<tirx::TensorTypeObj>())
    return tensor->scope.name();
  return var->ty.as_or_throw<PtrType>()->scope.name();
}
inline int64_t CodegenInt(const IntImmObj *value) {
  TVM_FFI_CHECK(value, TypeError)
      << "Code generation requires an integer literal";
  return value->value.as<int64_t>().value();
}
inline int64_t CodegenInt(const Expr &expr) {
  return CodegenInt(expr.as<IntImmObj>());
}
inline bool CodegenIsFloat8(const PrimType &type) {
  return type.bits() == 8 &&
         type.MatchesCode(kDLFloat8_e3m4, kDLFloat8_e4m3, kDLFloat8_e4m3b11fnuz,
                          kDLFloat8_e4m3fn, kDLFloat8_e4m3fnuz, kDLFloat8_e5m2,
                          kDLFloat8_e5m2fnuz, kDLFloat8_e8m0fnu);
}
inline bool CodegenIsFloat6(const PrimType &type) {
  return type.bits() == 6 &&
         type.MatchesCode(kDLFloat6_e2m3fn, kDLFloat6_e3m2fn);
}
inline bool CodegenIsFloat4(const PrimType &type) {
  return type.MatchesElementType(kDLFloat4_e2m1fn, 4) ||
         tl::IsFloat4E2M1Unpacked(type);
}
inline bool CodegenIsVectorBool(const PrimType &type) {
  return !type.IsScalar() && type.MatchesCode(kDLBool);
}
inline size_t CodegenAllocationElements(const tirx::TensorVar &tensor) {
  ffi::BigInt total = 1;
  for (const PrimExpr &dim : tensor.shape()) {
    const auto *value = dim.as<IntImmObj>();
    TVM_FFI_CHECK(value, ValueError)
        << "Static allocation required for " << tensor;
    TVM_FFI_CHECK_GT(value->value, 0, ValueError)
        << "Allocation extent must be positive";
    total = total * value->value;
  }
  auto size = total.as<size_t>();
  TVM_FFI_CHECK(size.has_value(), ValueError)
      << "Allocation size exceeds size_t";
  return size.value();
}

} // namespace tvm::codegen
#endif // TILELANG_CODEGEN_IR_UTILS_H_
