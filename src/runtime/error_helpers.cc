/*
 * Helper functions for nicer runtime error messages.
 */
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/data_type.h>

#include <sstream>
#include <string>

namespace tvm {
namespace tl {

// Return non-zero so that tvm_call_packed sites treat it as failure and return
// -1.
static int DTypeMismatch(const tvm::ffi::String &kernel_name,
                         const tvm::ffi::String &buffer_name,
                         int64_t actual_code, int64_t actual_bits,
                         int64_t actual_lanes, int64_t expect_code,
                         int64_t expect_bits, int64_t expect_lanes) {
  tvm::runtime::DataType actual(static_cast<int>(actual_code),
                                static_cast<int>(actual_bits),
                                static_cast<int>(actual_lanes));
  tvm::runtime::DataType expect(static_cast<int>(expect_code),
                                static_cast<int>(expect_bits),
                                static_cast<int>(expect_lanes));
  std::ostringstream os;
  os << std::string(kernel_name) << ": dtype of " << std::string(buffer_name)
     << " is expected to be " << expect << ", but got " << actual;
  TVMFFIErrorSetRaisedFromCStr("RuntimeError", os.str().c_str());
  return -1;
}

// Variant without names, to avoid passing extra raw strings through packed
// args.
static int DTypeMismatchNoNames(int64_t actual_code, int64_t actual_bits,
                                int64_t actual_lanes, int64_t expect_code,
                                int64_t expect_bits, int64_t expect_lanes) {
  tvm::runtime::DataType actual(static_cast<int>(actual_code),
                                static_cast<int>(actual_bits),
                                static_cast<int>(actual_lanes));
  tvm::runtime::DataType expect(static_cast<int>(expect_code),
                                static_cast<int>(expect_bits),
                                static_cast<int>(expect_lanes));
  std::ostringstream os;
  os << "dtype mismatch: expected " << expect << ", but got " << actual;
  TVMFFIErrorSetRaisedFromCStr("RuntimeError", os.str().c_str());
  return -1;
}

} // namespace tl
} // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tilelang_error_dtype_mismatch",
                        &tvm::tl::DTypeMismatch);
  refl::GlobalDef().def("tilelang_error_dtype_mismatch2",
                        &tvm::tl::DTypeMismatchNoNames);
}
