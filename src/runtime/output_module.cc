/*!
 * \file output_module.cc
 * \brief Packed wrappers with per-call output ownership and an exportable ABI.
 */
#include "output_module.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstdint>
#include <mutex>
#include <string_view>
#include <unordered_set>

namespace tvm::tl {
namespace {
constexpr uint32_t kSchemaVersion = 1;
constexpr size_t kMaximumSchemaBytes = 1024 * 1024;

void WriteU32(std::string *bytes, uint32_t value) {
  for (int i = 0; i < 4; ++i)
    bytes->push_back(static_cast<char>((value >> (i * 8)) & 255));
}
void WriteString(std::string *bytes, const std::string &value) {
  TVM_FFI_CHECK_LE(value.size(), kMaximumSchemaBytes, ValueError)
      << "Output ABI name too long";
  WriteU32(bytes, static_cast<uint32_t>(value.size()));
  bytes->append(value);
}
class SchemaReader {
public:
  explicit SchemaReader(const ffi::Bytes &bytes)
      : bytes_(bytes.data(), bytes.size()) {
    TVM_FFI_CHECK_LE(bytes.size(), kMaximumSchemaBytes, ValueError)
        << "Output ABI schema too large";
  }
  uint32_t U32() {
    TVM_FFI_CHECK_GE(bytes_.size(), 4U, ValueError)
        << "Truncated output ABI schema";
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i)
      result |= static_cast<uint32_t>(static_cast<unsigned char>(bytes_[i]))
                << (i * 8);
    bytes_.remove_prefix(4);
    return result;
  }
  std::string String() {
    uint32_t size = U32();
    TVM_FFI_CHECK_LE(size, bytes_.size(), ValueError)
        << "Truncated output ABI string";
    std::string result(bytes_.substr(0, size));
    bytes_.remove_prefix(size);
    return result;
  }
  bool empty() const { return bytes_.empty(); }

private:
  std::string_view bytes_;
};

std::vector<OutputEntrySpec> Decode(const ffi::Bytes &bytes) {
  SchemaReader reader(bytes);
  TVM_FFI_CHECK_EQ(reader.String(), "tilelang.output", ValueError)
      << "Invalid output ABI tag";
  TVM_FFI_CHECK_EQ(reader.U32(), kSchemaVersion, ValueError)
      << "Unknown output ABI version";
  TVM_FFI_CHECK_EQ(reader.String(), TILELANG_FFI_TARGET_REF, ValueError)
      << "Output ABI requires a different FFI provider";
  uint32_t count = reader.U32();
  TVM_FFI_CHECK(count > 0 && count <= 4096, ValueError)
      << "Invalid output entry count";
  std::vector<OutputEntrySpec> entries;
  std::unordered_set<std::string> public_names, worker_names;
  bool has_main = false;
  for (uint32_t i = 0; i < count; ++i) {
    OutputEntrySpec entry;
    entry.public_name = reader.String();
    entry.allocate_name = reader.String();
    entry.compute_name = reader.String();
    TVM_FFI_CHECK(!entry.public_name.empty() &&
                      public_names.insert(entry.public_name).second,
                  ValueError)
        << "Duplicate or empty output entry";
    TVM_FFI_CHECK(!entry.allocate_name.empty() && !entry.compute_name.empty() &&
                      worker_names.insert(entry.allocate_name).second &&
                      worker_names.insert(entry.compute_name).second,
                  ValueError)
        << "Duplicate or empty output worker";
    entry.parameter_count = static_cast<int>(reader.U32());
    entry.device_type = static_cast<int>(reader.U32());
    uint32_t main = reader.U32();
    TVM_FFI_CHECK_LE(main, 1U, ValueError) << "Invalid output entry flag";
    entry.is_main = main != 0;
    TVM_FFI_CHECK(!entry.is_main || !has_main, ValueError)
        << "Multiple main output entries";
    has_main |= entry.is_main;
    TVM_FFI_CHECK(entry.parameter_count > 0 && entry.parameter_count <= 65536 &&
                      entry.device_type > 0,
                  ValueError)
        << "Invalid output parameter contract";
    uint32_t nout = reader.U32();
    TVM_FFI_CHECK(nout > 0 &&
                      nout <= static_cast<uint32_t>(entry.parameter_count),
                  ValueError)
        << "Invalid output count";
    std::unordered_set<int> indices;
    for (uint32_t j = 0; j < nout; ++j) {
      int index = static_cast<int>(reader.U32());
      uint32_t code = reader.U32(), bits = reader.U32(), lanes = reader.U32();
      int rank = static_cast<int>(reader.U32());
      TVM_FFI_CHECK(index >= 0 && index < entry.parameter_count &&
                        indices.insert(index).second,
                    ValueError)
          << "Invalid or duplicate output index";
      TVM_FFI_CHECK(code <= 255 && bits > 0 && bits <= 255 && lanes > 0 &&
                        lanes <= 65535 && rank >= 0 && rank <= 1024,
                    ValueError)
          << "Invalid output tensor contract";
      entry.outputs.push_back(
          {index,
           DLDataType{static_cast<uint8_t>(code), static_cast<uint8_t>(bits),
                      static_cast<uint16_t>(lanes)},
           rank});
    }
    entries.push_back(std::move(entry));
  }
  TVM_FFI_CHECK(reader.empty(), ValueError)
      << "Trailing output ABI schema bytes";
  return entries;
}

class OutputModuleObj final : public ffi::ModuleObj {
public:
  explicit OutputModuleObj(ffi::Bytes schema)
      : schema_(std::move(schema)), entries_(Decode(schema_)) {}
  const char *kind() const final { return "tilelang.output"; }
  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }
  ffi::Bytes SaveToBytes() const final { return schema_; }
  ffi::String InspectSource(const ffi::String &format) const final {
    return imports().empty()
               ? ffi::String()
               : imports()[0].cast<ffi::Module>()->InspectSource(format);
  }
  void ImportModule(const ffi::Module &module) final {
    std::lock_guard<std::mutex> lock(mutex_);
    TVM_FFI_CHECK(!frozen_ && imports().empty(), ValueError)
        << "An output module has one immutable worker import";
    ffi::ModuleObj::ImportModule(module);
  }
  ffi::Optional<ffi::Function> GetFunction(const ffi::String &name) final {
    const OutputEntrySpec *selected = nullptr;
    for (const auto &entry : entries_) {
      if (entry.public_name == name || (entry.is_main && name == "main")) {
        selected = &entry;
        break;
      }
    }
    if (!selected)
      return std::nullopt;
    std::lock_guard<std::mutex> lock(mutex_);
    TVM_FFI_CHECK_EQ(imports().size(), 1U, RuntimeError)
        << "Missing output worker module";
    frozen_ = true;
    ffi::Module worker = imports()[0].cast<ffi::Module>();
    auto allocate = worker->GetFunction(selected->allocate_name, true);
    auto compute = worker->GetFunction(selected->compute_name, true);
    TVM_FFI_CHECK(allocate.has_value() && compute.has_value(), RuntimeError)
        << "Missing callable output allocation or compute worker";
    auto owner = ffi::GetObjectPtr<ffi::Object>(this);
    return ffi::Function::FromPacked([owner, selected, allocate = *allocate,
                                      compute = *compute](ffi::PackedArgs args,
                                                          ffi::Any *result) {
      int num_inputs = selected->parameter_count -
                       static_cast<int>(selected->outputs.size());
      TVM_FFI_CHECK_EQ(args.size(), num_inputs + 1, TypeError)
          << selected->public_name << " expects " << num_inputs
          << " inputs and an allocator anchor";
      ffi::TensorView anchor = args[num_inputs].cast<ffi::TensorView>();
      TVM_FFI_CHECK_EQ(anchor.device().device_type, selected->device_type,
                       RuntimeError)
          << "Allocator anchor device type does not match the kernel";
      OutputFrame frame(selected->outputs, anchor.device());
      std::vector<ffi::AnyView> prepare_args;
      for (int i = 0; i < num_inputs; ++i)
        prepare_args.push_back(args[i]);
      prepare_args.push_back(static_cast<void *>(&frame));
      ffi::Any ignored;
      allocate.CallPacked(prepare_args.data(), prepare_args.size(), &ignored);
      std::vector<ffi::AnyView> compute_args(selected->parameter_count);
      std::unordered_set<int> output_indices;
      for (size_t slot = 0; slot < selected->outputs.size(); ++slot) {
        int index = selected->outputs[slot].parameter_index;
        output_indices.insert(index);
        compute_args[index] = frame.Output(static_cast<int>(slot));
      }
      int input = 0;
      for (int index = 0; index < selected->parameter_count; ++index) {
        if (!output_indices.count(index))
          compute_args[index] = args[input++];
      }
      compute.CallPacked(compute_args.data(), compute_args.size(), &ignored);
      *result = frame.Complete();
    });
  }

private:
  ffi::Bytes schema_;
  std::vector<OutputEntrySpec> entries_;
  std::mutex mutex_;
  bool frozen_{false};
};
} // namespace

ffi::Bytes EncodeOutputSchema(const std::vector<OutputEntrySpec> &entries) {
  std::string bytes;
  WriteString(&bytes, "tilelang.output");
  WriteU32(&bytes, kSchemaVersion);
  WriteString(&bytes, TILELANG_FFI_TARGET_REF);
  WriteU32(&bytes, entries.size());
  for (const auto &entry : entries) {
    WriteString(&bytes, entry.public_name);
    WriteString(&bytes, entry.allocate_name);
    WriteString(&bytes, entry.compute_name);
    WriteU32(&bytes, entry.parameter_count);
    WriteU32(&bytes, entry.device_type);
    WriteU32(&bytes, entry.is_main);
    WriteU32(&bytes, entry.outputs.size());
    for (const auto &slot : entry.outputs) {
      WriteU32(&bytes, slot.parameter_index);
      WriteU32(&bytes, slot.storage_dtype.code);
      WriteU32(&bytes, slot.storage_dtype.bits);
      WriteU32(&bytes, slot.storage_dtype.lanes);
      WriteU32(&bytes, slot.rank);
    }
  }
  ffi::Bytes result(bytes);
  Decode(result);
  return result;
}

ffi::Module MakeOutputModule(ffi::Module worker, ffi::Bytes schema) {
  auto node = ffi::make_object<OutputModuleObj>(std::move(schema));
  node->ImportModule(worker);
  return ffi::Module(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.runtime.MakeOutputModule", MakeOutputModule)
      .def("ffi.Module.load_from_bytes.tilelang.output", [](ffi::Bytes bytes) {
        return ffi::Module(ffi::make_object<OutputModuleObj>(std::move(bytes)));
      });
}
} // namespace tvm::tl
