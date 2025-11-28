/*!
 * \file tl/runtime/error_helpers.h
 * \brief Error helper FFI names for TileLang runtime.
 */

#ifndef TVM_TL_RUNTIME_ERROR_HELPERS_H_
#define TVM_TL_RUNTIME_ERROR_HELPERS_H_

namespace tvm {
namespace tl {

// Error helper packed functions
constexpr const char *tvm_error_dtype_mismatch = "__tvm_error_dtype_mismatch";
constexpr const char *tvm_error_ndim_mismatch = "__tvm_error_ndim_mismatch";
constexpr const char *tvm_error_byte_offset_mismatch =
    "__tvm_error_byte_offset_mismatch";
constexpr const char *tvm_error_device_type_mismatch =
    "__tvm_error_device_type_mismatch";
constexpr const char *tvm_error_null_ptr = "__tvm_error_null_ptr";
constexpr const char *tvm_error_expect_eq = "__tvm_error_expect_eq";
constexpr const char *tvm_error_constraint_violation =
    "__tvm_error_constraint_violation";

} // namespace tl
} // namespace tvm

#endif // TVM_TL_RUNTIME_ERROR_HELPERS_H_
