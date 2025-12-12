/*
 * Helper functions for nicer runtime error messages.
 */
#include "error_helpers.h"

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>

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
  os << "kernel " << std::string(kernel_name) << " input "
     << std::string(buffer_name) << " dtype expected " << expect << ", but got "
     << actual;
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

// Register packed versions, following the design in runtime.cc
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  // Packed: __tvm_error_dtype_mismatch(kernel_name, buffer_name,
  //                                    actual_code, actual_bits, actual_lanes,
  //                                    expect_code, expect_bits, expect_lanes)
  refl::GlobalDef().def_packed(
      tl::tvm_error_dtype_mismatch,
      [](tvm::ffi::PackedArgs args, tvm::ffi::Any *ret) {
        ICHECK(args.size() == 8) << "Expected 8 args: kernel, buffer, "
                                    "actual_code, actual_bits, actual_lanes, "
                                 << "expect_code, expect_bits, expect_lanes";

        auto kernel_name = args[0].cast<tvm::ffi::String>();
        auto buffer_name = args[1].cast<tvm::ffi::String>();
        int64_t actual_code = args[2].cast<int64_t>();
        int64_t actual_bits = args[3].cast<int64_t>();
        int64_t actual_lanes = args[4].cast<int64_t>();
        int64_t expect_code = args[5].cast<int64_t>();
        int64_t expect_bits = args[6].cast<int64_t>();
        int64_t expect_lanes = args[7].cast<int64_t>();

        // Reuse the helper to format the message
        (void)DTypeMismatch(kernel_name, buffer_name, actual_code, actual_bits,
                            actual_lanes, expect_code, expect_bits,
                            expect_lanes);
        // Provide a return value for completeness, then signal the error
        *ret = -1;
        throw ::tvm::ffi::EnvErrorAlreadySet();
      });

  // kernel, buffer, expect:int64, got:int64
  refl::GlobalDef().def_packed(
      tl::tvm_error_ndim_mismatch,
      [](tvm::ffi::PackedArgs args, tvm::ffi::Any *ret) {
        ICHECK(args.size() == 4)
            << "__tvm_error_ndim_mismatch(kernel, buffer, expect, got)";
        auto kernel = args[0].cast<tvm::ffi::String>();
        auto buffer = args[1].cast<tvm::ffi::String>();
        int64_t expect = args[2].cast<int64_t>();
        int64_t got = args[3].cast<int64_t>();
        std::ostringstream os;
        os << "kernel " << std::string(kernel) << " input "
           << std::string(buffer) << " ndim expected " << expect << ", but got "
           << got;
        TVMFFIErrorSetRaisedFromCStr("RuntimeError", os.str().c_str());
        *ret = -1;
        throw ::tvm::ffi::EnvErrorAlreadySet();
      });

  // kernel, buffer, expect:int64, got:int64
  refl::GlobalDef().def_packed(
      tl::tvm_error_byte_offset_mismatch,
      [](tvm::ffi::PackedArgs args, tvm::ffi::Any *ret) {
        ICHECK(args.size() == 4)
            << "__tvm_error_byte_offset_mismatch(kernel, buffer, expect, got)";
        auto kernel = args[0].cast<tvm::ffi::String>();
        auto buffer = args[1].cast<tvm::ffi::String>();
        int64_t expect = args[2].cast<int64_t>();
        int64_t got = args[3].cast<int64_t>();
        std::ostringstream os;
        os << "kernel " << std::string(kernel) << " input "
           << std::string(buffer) << " byte_offset expected " << expect
           << ", but got " << got;
        TVMFFIErrorSetRaisedFromCStr("RuntimeError", os.str().c_str());
        *ret = -1;
        throw ::tvm::ffi::EnvErrorAlreadySet();
      });

  // kernel, buffer, expect:int64, got:int64
  refl::GlobalDef().def_packed(
      tl::tvm_error_device_type_mismatch,
      [](tvm::ffi::PackedArgs args, tvm::ffi::Any *ret) {
        ICHECK(args.size() == 4)
            << "__tvm_error_device_type_mismatch(kernel, buffer, expect, got)";
        auto kernel = args[0].cast<tvm::ffi::String>();
        auto buffer = args[1].cast<tvm::ffi::String>();
        int64_t expect = args[2].cast<int64_t>();
        int64_t got = args[3].cast<int64_t>();
        const char *expect_str =
            tvm::runtime::DLDeviceType2Str(static_cast<int>(expect));
        const char *got_str =
            tvm::runtime::DLDeviceType2Str(static_cast<int>(got));
        std::ostringstream os;
        os << "kernel " << std::string(kernel) << " input "
           << std::string(buffer) << " device_type expected " << expect_str
           << ", but got " << got_str;
        TVMFFIErrorSetRaisedFromCStr("RuntimeError", os.str().c_str());
        *ret = -1;
        throw ::tvm::ffi::EnvErrorAlreadySet();
      });

  // kernel, buffer, field:String
  refl::GlobalDef().def_packed(
      tl::tvm_error_null_ptr,
      [](tvm::ffi::PackedArgs args, tvm::ffi::Any *ret) {
        ICHECK(args.size() == 3)
            << "__tvm_error_null_ptr(kernel, buffer, field)";
        auto kernel = args[0].cast<tvm::ffi::String>();
        auto buffer = args[1].cast<tvm::ffi::String>();
        auto field = args[2].cast<tvm::ffi::String>();
        std::ostringstream os;
        os << "kernel " << std::string(kernel) << " input "
           << std::string(buffer) << ' ' << std::string(field)
           << " expected non-NULL, but got NULL";
        TVMFFIErrorSetRaisedFromCStr("RuntimeError", os.str().c_str());
        *ret = -1;
        throw ::tvm::ffi::EnvErrorAlreadySet();
      });

  // kernel, buffer, field:String, expect:int64, got:int64
  refl::GlobalDef().def_packed(
      tl::tvm_error_expect_eq,
      [](tvm::ffi::PackedArgs args, tvm::ffi::Any *ret) {
        ICHECK(args.size() == 5)
            << "__tvm_error_expect_eq(kernel, buffer, field, expect, got)";
        auto kernel = args[0].cast<tvm::ffi::String>();
        auto buffer = args[1].cast<tvm::ffi::String>();
        auto field = args[2].cast<tvm::ffi::String>();
        int64_t expect = args[3].cast<int64_t>();
        int64_t got = args[4].cast<int64_t>();
        std::ostringstream os;
        os << "kernel " << std::string(kernel) << " input "
           << std::string(buffer) << ' ' << std::string(field) << " expected "
           << expect << ", but got " << got;
        TVMFFIErrorSetRaisedFromCStr("RuntimeError", os.str().c_str());
        *ret = -1;
        throw ::tvm::ffi::EnvErrorAlreadySet();
      });

  // kernel, buffer, field:String [, reason:String]
  refl::GlobalDef().def_packed(
      tl::tvm_error_constraint_violation,
      [](tvm::ffi::PackedArgs args, tvm::ffi::Any *ret) {
        ICHECK(args.size() == 3 || args.size() == 4)
            << "__tvm_error_constraint_violation(kernel, buffer, field[, "
               "reason])";
        auto kernel = args[0].cast<tvm::ffi::String>();
        auto buffer = args[1].cast<tvm::ffi::String>();
        auto field = args[2].cast<tvm::ffi::String>();
        std::string reason;
        if (args.size() == 4) {
          reason = args[3].cast<tvm::ffi::String>();
        }
        std::ostringstream os;
        os << "kernel " << std::string(kernel) << " input "
           << std::string(buffer) << ' ' << std::string(field)
           << " constraint not satisfied";
        if (!reason.empty()) {
          os << ": " << reason;
        }
        TVMFFIErrorSetRaisedFromCStr("RuntimeError", os.str().c_str());
        *ret = -1;
        throw ::tvm::ffi::EnvErrorAlreadySet();
      });

  // Legacy typed registrations for backward compatibility
  refl::GlobalDef().def("tilelang_error_dtype_mismatch",
                        &tvm::tl::DTypeMismatch);
  refl::GlobalDef().def("tilelang_error_dtype_mismatch2",
                        &tvm::tl::DTypeMismatchNoNames);
}

} // namespace tl
} // namespace tvm
