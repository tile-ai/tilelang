/*!
 * \file tl/ir/dtype.cc
 * \brief Custom dtype names consumed by the fixed TVM FFI parser and printer.
 */

#include "dtype.h"

#include <tvm/ffi/reflection/registry.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace tvm::tl {
namespace {

class CustomTypes {
public:
  static CustomTypes &Global() {
    static CustomTypes instance;
    return instance;
  }

  void Register(const std::string &name, int code) {
    // The frozen TVM registry converts the packed int to uint8 before checking
    // its kCustomBegin (129). Its two maps permit assignment, including
    // aliases; this callback is not the dtype string grammar or a uniqueness
    // validator.
    const uint8_t type_code = static_cast<uint8_t>(code);
    TVM_FFI_ICHECK(type_code >= 129)
        << "Please choose a type code >= DataType::kCustomBegin for custom "
           "types";
    std::lock_guard<std::mutex> lock(mutex_);
    names_[name] = type_code;
    codes_[type_code] = name;
  }

  int Code(const std::string &name) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = names_.find(name);
    TVM_FFI_ICHECK(found != names_.end())
        << "Type name " << name << " not registered";
    return found->second;
  }

  std::string Name(int code) {
    std::lock_guard<std::mutex> lock(mutex_);
    const uint8_t type_code = static_cast<uint8_t>(code);
    const auto found = codes_.find(type_code);
    TVM_FFI_ICHECK(found != codes_.end())
        << "Type code " << static_cast<unsigned>(type_code)
        << " not registered";
    return found->second;
  }

  bool Contains(int code) {
    std::lock_guard<std::mutex> lock(mutex_);
    return codes_.count(static_cast<uint8_t>(code)) != 0;
  }

private:
  CustomTypes()
      : names_{{"tfloat32", kTensorFloat32},
               {"float4_e2m1_unpacked", kFloat4E2M1Unpacked}},
        codes_{{kTensorFloat32, "tfloat32"},
               {kFloat4E2M1Unpacked, "float4_e2m1_unpacked"}} {}

  std::mutex mutex_;
  std::unordered_map<std::string, int> names_;
  std::unordered_map<int, std::string> codes_;
};

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  // These names are the extension hooks used by TVM FFI 8fd93039.  Their
  // implementation belongs to the compiler provider, not a second FFI library.
  ffi::reflection::GlobalDef()
      .def("dtype.get_custom_type_code",
           [](const std::string &name) {
             return CustomTypes::Global().Code(name);
           })
      .def("dtype.get_custom_type_name",
           [](int code) { return CustomTypes::Global().Name(code); })
      .def("dtype.register_custom_type",
           [](const std::string &name, int code) {
             CustomTypes::Global().Register(name, code);
           })
      .def("runtime._datatype_get_type_registered",
           [](int code) { return CustomTypes::Global().Contains(code); });
}

} // namespace tvm::tl
