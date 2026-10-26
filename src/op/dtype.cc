#include "op/dtype.h"

#include <tvm/ffi/reflection/registry.h>

#include <mutex>
#include <string>
#include <unordered_map>

namespace tvm::tl {
namespace {
struct CustomDTypeRegistry {
  std::mutex mutex;
  std::unordered_map<std::string, int> codes;
  std::unordered_map<int, std::pair<std::string, PrimType>> types;

  static CustomDTypeRegistry &Global() {
    static CustomDTypeRegistry registry;
    return registry;
  }
};
} // namespace

void RegisterCustomDType(ffi::String name, int code, PrimType storage_type) {
  TVM_FFI_CHECK(!name.empty() &&
                    std::string(name).find_first_of("[]") == std::string::npos,
                ValueError)
      << "Custom dtype names must be nonempty and cannot contain brackets";
  TVM_FFI_CHECK(
      code >= static_cast<int>(ffi::DLExtDataTypeCode::kDLExtCustomBegin) &&
          code <= 255,
      ValueError)
      << "Custom dtype codes must be in [129, 255]";
  TVM_FFI_CHECK(storage_type.IsScalar() && !tl::IsCustomDType(storage_type) &&
                    storage_type.bits() > 0 && storage_type.bits() % 8 == 0 &&
                    storage_type.MatchesCode(kDLFloat, kDLInt, kDLUInt),
                ValueError)
      << "Custom dtype storage must be a byte-addressable native scalar";
  auto &registry = CustomDTypeRegistry::Global();
  std::lock_guard<std::mutex> lock(registry.mutex);
  if (auto entry = registry.codes.find(name); entry != registry.codes.end()) {
    TVM_FFI_CHECK(entry->second == code &&
                      registry.types.at(code).second == storage_type,
                  ValueError)
        << "Custom dtype name is already registered: " << name;
    return;
  }
  TVM_FFI_CHECK(!registry.types.count(code), ValueError)
      << "Custom dtype code is already registered: " << code;
  registry.codes.emplace(name, code);
  registry.types.emplace(code, std::make_pair(std::string(name), storage_type));
}

PrimType CustomDTypeStorage(PrimType dtype) {
  if (!tl::IsCustomDType(dtype))
    return dtype;
  auto &registry = CustomDTypeRegistry::Global();
  std::lock_guard<std::mutex> lock(registry.mutex);
  auto entry = registry.types.find(static_cast<int>(dtype.code()));
  TVM_FFI_CHECK(entry != registry.types.end(), ValueError)
      << "Unregistered custom dtype code: " << static_cast<int>(dtype.code());
  TVM_FFI_CHECK(dtype.bits() == entry->second.second.bits(), ValueError)
      << "Custom dtype " << entry->second.first << " requires "
      << entry->second.second.bits() << " storage bits, got " << dtype.bits();
  DLDataType storage = entry->second.second->dtype;
  storage.lanes = dtype->dtype.lanes;
  return PrimType(storage);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.ir.RegisterCustomDType", RegisterCustomDType)
      .def("tl.ir.CustomDTypeStorage", CustomDTypeStorage)
      .def("dtype.get_custom_type_code",
           [](ffi::String name) {
             auto &registry = CustomDTypeRegistry::Global();
             std::lock_guard<std::mutex> lock(registry.mutex);
             auto entry = registry.codes.find(name);
             TVM_FFI_CHECK(entry != registry.codes.end(), ValueError)
                 << "Unregistered custom dtype name: " << name;
             return entry->second;
           })
      .def("dtype.get_custom_type_name", [](int code) {
        auto &registry = CustomDTypeRegistry::Global();
        std::lock_guard<std::mutex> lock(registry.mutex);
        auto entry = registry.types.find(code);
        TVM_FFI_CHECK(entry != registry.types.end(), ValueError)
            << "Unregistered custom dtype code: " << code;
        return ffi::String(entry->second.first);
      });
}

} // namespace tvm::tl
