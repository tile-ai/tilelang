// Test-only owned storage and objects for the public Any-list ABI.
#include <tl_templates/cpp/anylist.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace ffi = tvm::ffi;
namespace {
std::map<int64_t, int64_t> destructions;

class Tracked final : public ffi::Object {
public:
  int64_t id{0};
  ~Tracked() { ++destructions[id]; }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.testing.ABI106Tracked", Tracked,
                                    ffi::Object);
};

class Arena final : public ffi::Object {
public:
  std::vector<ffi::Any> slots;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.testing.ABI106Arena", Arena,
                                    ffi::Object);
};

ffi::Any Make(int64_t kind, int64_t id) {
  switch (kind) {
  case 0:
    return nullptr;
  case 1:
    return int64_t(1000 + id);
  case 2:
    return double(3.25 + id);
  case 3:
    return true;
  case 4:
    return ffi::String("short");
  case 5:
    return ffi::String(std::string(80, 'x') + std::to_string(id));
  case 6: {
    auto obj = ffi::make_object<Tracked>();
    obj->id = id;
    return ffi::ObjectRef(obj);
  }
  default:
    TVM_FFI_THROW(ValueError) << "ABI106 requested callback failure";
  }
}

ffi::ObjectRef NewArena(int64_t size) {
  TVM_FFI_CHECK(size >= 0 && size < 1000, ValueError);
  auto obj = ffi::make_object<Arena>();
  obj->slots.resize(size);
  return ffi::ObjectRef(obj);
}

Arena *AsArena(const ffi::ObjectRef &arena) {
  return const_cast<Arena *>(arena.as<Arena>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::ObjectDef<Tracked>();
  ffi::reflection::ObjectDef<Arena>();
}
} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(new_arena, NewArena);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(abi106_make, Make);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(abi106_echo,
                              [](ffi::Any value) { return value; });
TVM_FFI_DLL_EXPORT_TYPED_FUNC(abi106_mix, [](int64_t a, double b, bool c,
                                             ffi::Any value) {
  TVM_FFI_CHECK(a == 11 && b == 2.5 && c, ValueError)
      << "Mixed argument contract lost";
  return value;
});
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pointer, [](ffi::ObjectRef arena) -> void * {
  return AsArena(arena)->slots.data();
});
TVM_FFI_DLL_EXPORT_TYPED_FUNC(put, [](ffi::ObjectRef arena, int64_t index,
                                      ffi::Any value) {
  AsArena(arena)->slots.at(index) = std::move(value);
});
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get,
                              [](ffi::ObjectRef arena,
                                 int64_t index) -> ffi::Any {
                                return AsArena(arena)->slots.at(index);
                              });
TVM_FFI_DLL_EXPORT_TYPED_FUNC(type_index, [](ffi::ObjectRef arena,
                                             int64_t index) {
  return AsArena(arena)->slots.at(index).type_index();
});
TVM_FFI_DLL_EXPORT_TYPED_FUNC(tracked_id,
                              [](ffi::ObjectRef arena, int64_t index) {
                                return AsArena(arena)
                                    ->slots.at(index)
                                    .cast<ffi::ObjectRef>()
                                    .as<Tracked>()
                                    ->id;
                              });
TVM_FFI_DLL_EXPORT_TYPED_FUNC(destroyed,
                              [](int64_t id) { return destructions[id]; });
TVM_FFI_DLL_EXPORT_TYPED_FUNC(reset_raw, [](void *arena, int64_t index) {
  TVM_FFI_CHECK_SAFE_CALL(TLAnyListResetItem(arena, index));
});
TVM_FFI_DLL_EXPORT_TYPED_FUNC(move_roundtrip, [](ffi::ObjectRef arena,
                                                 int64_t id) {
  TVMFFIAny result{};
  {
    ffi::Any object = Make(6, id);
    TVMFFIAny borrowed = ffi::AnyView(object).CopyToTVMFFIAny();
    TVM_FFI_CHECK_SAFE_CALL(TVMFFIAnyViewToOwnedAny(&borrowed, &result));
  }
  TVM_FFI_CHECK_SAFE_CALL(
      TLAnyListMoveResult(AsArena(arena)->slots.data(), 0, &result));
  TVM_FFI_CHECK(result.type_index == 0 && result.zero_padding == 0 &&
                    result.v_int64 == 0,
                RuntimeError)
      << "Moved return slot was not cleared";
  TVMFFIAny borrowed{};
  TVM_FFI_CHECK_SAFE_CALL(
      TLAnyListBorrow(AsArena(arena)->slots.data(), 0, &borrowed));
  TVM_FFI_CHECK(borrowed.type_index == AsArena(arena)->slots[0].type_index(),
                RuntimeError);
  TVM_FFI_CHECK_SAFE_CALL(TLAnyListResetItem(AsArena(arena)->slots.data(), 0));
  TVM_FFI_CHECK(destructions[id] == 1, RuntimeError)
      << "Borrow unexpectedly retained the object";
  // The borrowed argument is not owning. Clearing it must not destroy again.
  borrowed = TVMFFIAny{};
  TVM_FFI_CHECK_SAFE_CALL(TLAnyListResetItem(AsArena(arena)->slots.data(), 0));
  TVM_FFI_CHECK(destructions[id] == 1, RuntimeError);
});

namespace {
// Copy fields and values while the packed-call stack owns the descriptor.
// Numeric addresses are retained only for identity checks, never dereferenced
// after the callback returns.
std::vector<ffi::Array<ffi::Any>> descriptor_observations;

ffi::Array<ffi::Any> ReadDescriptor(DLTensor *tensor, int64_t tag) {
  TVM_FFI_CHECK(tensor && tensor->data, ValueError);
  TVM_FFI_CHECK(tensor->ndim >= 0 && tensor->ndim <= 4, ValueError);
  TVM_FFI_CHECK(tensor->dtype.code == kDLFloat && tensor->dtype.bits == 32 &&
                    tensor->dtype.lanes == 1,
                ValueError);
  ffi::Array<int64_t> shape;
  ffi::Array<int64_t> raw_strides;
  std::vector<int64_t> strides(tensor->ndim);
  int64_t count = 1;
  for (int i = 0; i < tensor->ndim; ++i) {
    TVM_FFI_CHECK(tensor->shape && tensor->shape[i] > 0, ValueError);
    shape.push_back(tensor->shape[i]);
    count *= tensor->shape[i];
    if (tensor->strides) {
      raw_strides.push_back(tensor->strides[i]);
    }
  }
  TVM_FFI_CHECK(count <= 32, ValueError);
  int64_t compact_stride = 1;
  for (int i = tensor->ndim - 1; i >= 0; --i) {
    strides[i] = tensor->strides ? tensor->strides[i] : compact_stride;
    compact_stride *= tensor->shape[i];
  }
  ffi::Array<double> values;
  for (int64_t ordinal = 0; ordinal < count; ++ordinal) {
    int64_t remaining = ordinal;
    int64_t element = 0;
    for (int i = tensor->ndim - 1; i >= 0; --i) {
      element += (remaining % tensor->shape[i]) * strides[i];
      remaining /= tensor->shape[i];
    }
    float value;
    const char *address = static_cast<const char *>(tensor->data) +
                          tensor->byte_offset + element * sizeof(float);
    std::memcpy(&value, address, sizeof(value));
    values.push_back(static_cast<double>(value));
  }
  return {tag,
          static_cast<int64_t>(reinterpret_cast<uintptr_t>(tensor)),
          static_cast<int64_t>(reinterpret_cast<uintptr_t>(tensor->data)),
          static_cast<int64_t>(reinterpret_cast<uintptr_t>(tensor->shape)),
          static_cast<int64_t>(reinterpret_cast<uintptr_t>(tensor->strides)),
          int64_t(tensor->ndim),
          int64_t(tensor->device.device_type),
          int64_t(tensor->device.device_id),
          int64_t(tensor->dtype.code),
          int64_t(tensor->dtype.bits),
          int64_t(tensor->dtype.lanes),
          static_cast<int64_t>(tensor->byte_offset),
          shape,
          tensor->strides ? ffi::Any(raw_strides) : ffi::Any(nullptr),
          values};
}
} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(abi154_observe, [](DLTensor *tensor,
                                                 int64_t tag) {
  descriptor_observations.push_back(ReadDescriptor(tensor, tag));
  return tag + 1000;
});
TVM_FFI_DLL_EXPORT_TYPED_FUNC(abi154_records, []() {
  return ffi::Array<ffi::Array<ffi::Any>>(descriptor_observations.begin(),
                                          descriptor_observations.end());
});
TVM_FFI_DLL_EXPORT_TYPED_FUNC(abi154_clear,
                              []() { descriptor_observations.clear(); });
