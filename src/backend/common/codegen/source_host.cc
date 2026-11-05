/*!
 * \file source_host.cc
 * \brief Emit native host control flow for source-based runtime adapters.
 */
#include "codegen_c_host.h"

#include <algorithm>
#include <array>
#include <unordered_map>
#include <unordered_set>

#include <tvm/ffi/dtype.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/abi.h>

#include "backend/cuda/launch/launch_plan.h"
#include "backend/cuda/launch/launch_source.h"

namespace tvm::tl {
namespace {

// PTO's separately compiled fat object uses the CCE global-address-space
// calling convention. Keep that spelling distinct from the Ascend C emitter's
// internal fp8 aliases; both prototypes and call-site casts use this contract.
std::string PTOArgumentType(const Type &type) {
  Type element = type;
  bool is_pointer = false;
  if (const auto *pointer = type.as<PtrTypeObj>()) {
    TVM_FFI_CHECK(pointer->scope == runtime::MemoryScope::Global(), TypeError)
        << "PTO kernel parameters require global pointer storage scope";
    element = pointer->element_type;
    is_pointer = true;
  }
  auto primitive = element.as<PrimType>();
  TVM_FFI_CHECK(primitive && primitive->IsScalar(), TypeError)
      << "PTO kernel parameters require a primitive scalar or global pointer";
  static const std::unordered_map<std::string, std::string> types{
      {"float32", "float"},
      {"float16", "half"},
      {"bfloat16", "bfloat16_t"},
      {"float64", "double"},
      {"float8_e4m3", "float8_e4m3_t"},
      {"float8_e4m3fn", "float8_e4m3_t"},
      {"float8_e5m2", "float8_e5m2_t"},
      {"float8_e8m0fnu", "float8_e8m0_t"},
      {"float4_e2m1fn", "float4_e2m1x2_t"},
      {"float4_e2m1fnx2", "float4_e2m1x2_t"},
      {"int64", "int64_t"},
      {"uint64", "uint64_t"},
      {"int32", "int"},
      {"uint32", "unsigned int"},
      {"bool", "int8_t"},
      {"int8", "int8_t"},
      {"uint8", "uint8_t"},
      {"int16", "int16_t"},
      {"uint16", "uint16_t"}};
  std::string name = ffi::DLDataTypeToString(primitive->get()->dtype);
  auto found = types.find(name);
  TVM_FFI_CHECK(found != types.end(), TypeError)
      << "Unsupported PTO argument dtype: " << name;
  return is_pointer ? "__gm__ " + found->second + " *" : found->second;
}

class SourceHostCodegen final : public CodeGenCHost {
public:
  using CodeGenCHost::Dispatch_;
  using CodeGenCHost::PrintType;

  void PrintType(const Type &type, std::ostream &os) final {
    if (const auto *pointer = type.as<PtrTypeObj>()) {
      auto element = pointer->element_type.as<PrimType>();
      if (element &&
          (element->code() == kDLFloat4_e2m1fn ||
           (element->bits() < 8 && !element->MatchesCode(kDLBool)))) {
        // Raw source adapters pass storage addresses, never a C scalar for a
        // nibble. Retain the IR's element units for address lowering and cast
        // to the device emitter's declared pointer type at the launch site.
        os << "void*";
        return;
      }
    }
    CodeGenCHost::PrintType(type, os);
  }

  void InitSource(bool include_preamble) {
    Init(false, true, true, "c", {});
    if (!include_preamble) {
      // A CPU device translation unit already owns the runtime declarations.
      // Clear only Init's fixed prefix, before any function, constant, or
      // external-call declarations have been emitted.
      preamble_stream_.str("");
      decl_stream.str("");
    }
    // Source libraries are loaded through ctypes, without a TVM Module
    // import context. Resolve callbacks through the one process-wide FFI
    // registry and retain the result until it has been consumed or discarded.
    // Operand evaluation is sequenced by PrintPackedGlobalCall, including
    // expressions whose code generation requires statement preludes.
    decl_stream
        << "#include <tvm/ffi/function.h>\n#include <initializer_list>\n"
        << R"PACKED(
static tvm::ffi::Any tl_source_call_packed(
    const char* name, std::initializer_list<tvm::ffi::AnyView> arguments) {
  tvm::ffi::Any result;
  tvm::ffi::Function::GetGlobalRequired(name).CallPacked(
      arguments.begin(), static_cast<int32_t>(arguments.size()), &result);
  return result;
}
)PACKED";
  }

  void InitDeviceSource(const IRModule &device, const Target &target) {
    InitSource(true);
    device_kind_ = target->kind->name;
    is_pto_ = device_kind_ == "ascend" &&
              std::find(target->keys.begin(), target->keys.end(), "pto") !=
                  target->keys.end();
    for (const auto &[global, base] : device->functions) {
      auto function = base.as_or_throw<tirx::Function>();
      kernels_.emplace(function->GetAttr(tvm::attr::kGlobalSymbol)
                           .value_or(global->name_hint),
                       function);
    }
    decl_stream << "#include <tvm/ffi/function.h>\n";
    if (device_kind_ == "cuda") {
      decl_stream << "#include <cuda.h>\n#include <cuda_runtime.h>\n"
                  << "#include <map>\n#include <memory>\n"
                  << tvm_cuda_launch::kSupportSource << R"CUDA(
static void tl_source_cuda_launch(const void* kernel,
    tvm_cuda_launch::LaunchValues values,
    const tvm_cuda_launch::KernelRequirements& requirements,
    void* stream, void** parameters) {
  using namespace tvm_cuda_launch;
  int device;
  Check(cudaGetDevice(&device));
  CUcontext context;
  unsigned long long context_id;
  Check(cuCtxGetCurrent(&context));
  Check(cuCtxGetId(context, &context_id));
  static std::mutex mutex;
  static std::map<std::pair<const void*, unsigned long long>,
                  std::unique_ptr<ResourceState>> states;
  ResourceState* state;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto& slot = states[{kernel, context_id}];
    if (!slot) slot = std::make_unique<ResourceState>();
    state = slot.get();
  }
  values = Normalize(values, requirements);
  PrepareResources<RuntimeAPI>(kernel, device, &values, state);
  cudaLaunchConfig_t config{};
  std::array<cudaLaunchAttribute, kMaxAttributes> attributes{};
  EncodeRuntime(values, requirements, static_cast<cudaStream_t>(stream),
                &config, &attributes);
  Check(cudaLaunchKernelExC(&config, kernel, parameters));
}
)CUDA";
    } else if (device_kind_ == "rocm" || device_kind_ == "hip") {
      decl_stream << "#include <hip/hip_runtime.h>\n#include <array>\n"
                  << R"HIP(
static void tl_source_hip_check(hipError_t status) {
  TVM_FFI_CHECK(status == hipSuccess, RuntimeError) << hipGetErrorString(status);
}
static void tl_source_hip_launch(const void* kernel, dim3 grid, dim3 block,
    size_t shared_bytes, void* stream, void** parameters) {
  if (shared_bytes != 0) {
    int device, maximum;
    tl_source_hip_check(hipGetDevice(&device));
    tl_source_hip_check(hipDeviceGetAttribute(
        &maximum, hipDeviceAttributeMaxSharedMemoryPerBlock, device));
    TVM_FFI_CHECK(maximum > 0 && shared_bytes <= static_cast<size_t>(maximum), ValueError)
        << "Requested dynamic shared memory exceeds the HIP device limit";
  }
  tl_source_hip_check(hipLaunchKernel(kernel, grid, block, parameters,
                                    shared_bytes, static_cast<hipStream_t>(stream)));
  tl_source_hip_check(hipGetLastError());
}
)HIP";
    } else if (device_kind_ == "ascend") {
      decl_stream << "#include <acl/acl.h>\n";
      if (is_pto_) {
        decl_stream << "#ifndef AICORE\n#define AICORE [aicore]\n#endif\n";
        std::vector<std::string> symbols;
        for (const auto &[symbol, function] : kernels_)
          symbols.push_back(symbol);
        std::sort(symbols.begin(), symbols.end());
        for (const auto &symbol : symbols) {
          const auto &function = kernels_.at(symbol);
          decl_stream << "extern \"C\" __global__ AICORE void " << symbol
                      << "(";
          for (size_t i = 0; i < function->params.size(); ++i) {
            if (i)
              decl_stream << ", ";
            decl_stream << PTOArgumentType(function->params[i]->ty);
          }
          decl_stream << ");\n";
        }
      }
    } else {
      TVM_FFI_THROW(ValueError)
          << "Native source host is not enabled for " << device_kind_;
    }
    if (device_kind_ != "cuda") {
      decl_stream << "#include <limits>\n#include <type_traits>\n"
                  << R"LAUNCH(
template <typename T> static uint32_t tl_source_launch_dimension(T value) {
  static_assert(std::is_integral_v<T>);
  if constexpr (std::is_signed_v<T>) {
    TVM_FFI_CHECK(value >= 0, ValueError) << "Negative kernel launch dimension";
  }
  TVM_FFI_CHECK(static_cast<uint64_t>(value) <= std::numeric_limits<uint32_t>::max(),
                ValueError) << "Kernel launch dimension exceeds uint32";
  // Match the provider runtime's LaunchParamConfig for empty tensor grids.
  return value == 0 ? 1 : static_cast<uint32_t>(value);
}
template <typename T> static size_t tl_source_shared_bytes(T value) {
  static_assert(std::is_integral_v<T>);
  if constexpr (std::is_signed_v<T>) {
    TVM_FFI_CHECK(value >= 0, ValueError) << "Negative dynamic shared-memory size";
  }
  TVM_FFI_CHECK(static_cast<uint64_t>(value) <= std::numeric_limits<size_t>::max(),
                ValueError) << "Dynamic shared-memory size exceeds size_t";
  return static_cast<size_t>(value);
}
)LAUNCH";
    }
  }

  void PrintFuncPrefix(std::ostream &os) final {
    CodeGenCHost::PrintFuncPrefix(os);
    os << "TVM_FFI_DLL_EXPORT ";
  }

  void AddSourceFunction(const GlobalVar &global,
                         const tirx::Function &function) {
    function_allocations_.str("");
    allocation_position_ = std::streampos(-1);
    auto stream_index =
        function->GetAttr<int64_t>("tl.source_stream_param_index");
    if (stream_index) {
      TVM_FFI_CHECK(*stream_index >= 0 && static_cast<size_t>(*stream_index) <
                                              function->params.size(),
                    ValueError)
          << "Invalid source host stream parameter index";
      current_stream_ = function->params[*stream_index];
    } else {
      current_stream_ = std::nullopt;
    }
    if (function->HasNonzeroAttr("tl.source_entry")) {
      const auto result_type = function->ret_type.as<PrimType>();
      TVM_FFI_CHECK(result_type && result_type.value() == PrimType::Int(32),
                    TypeError)
          << "A source adapter entry must return an int32 error code";
      InitFuncState(function);
      PrintFunctionSignature(GetFunctionName(global), function, stream);
      stream << " {\n  TVM_FFI_SAFE_CALL_BEGIN();\n  try {\n";
      allocation_position_ = stream.tellp();
      int scope = BeginScope();
      PrintStmt(function->body);
      EndScope(scope);
      stream << "  } catch (...) {\n";
      if (device_kind_ == "cuda")
        stream << "    cudaGetLastError();\n";
      if (device_kind_ == "rocm" || device_kind_ == "hip")
        stream << "    hipGetLastError();\n";
      stream << "    throw;\n  }\n  TVM_FFI_SAFE_CALL_END();\n}\n\n";
    } else {
      CodeGenCHost::AddFunction(global, function, true);
    }
    if (!function_allocations_.str().empty()) {
      TVM_FFI_ICHECK(allocation_position_ != std::streampos(-1));
      auto text = stream.str();
      text.insert(static_cast<size_t>(allocation_position_),
                  function_allocations_.str());
      stream.str(text);
      stream.seekp(0, std::ios::end);
    }
    current_stream_ = std::nullopt;
  }

  void PreFunctionBody(const tirx::Function &function) final {
    CodeGenCHost::PreFunctionBody(function);
    allocation_position_ = stream.tellp();
  }

  void Dispatch_(const CallObj *call, std::ostream &os) final {
    if (packed_call_depth_ && call->op.same_as(tirx::stack_alloca_op())) {
      // A returned pointer can alias an inline stack argument. Keep its
      // storage in the enclosing host function, as the LLVM emitter does,
      // rather than ending its lifetime with a packed-call lambda.
      std::ostringstream surrounding;
      stream.swap(surrounding);
      try {
        CodeGenCHost::Dispatch_(call, os);
        function_allocations_ << stream.str();
      } catch (...) {
        stream.swap(surrounding);
        throw;
      }
      stream.swap(surrounding);
      return;
    }
    if (call->op.same_as(tirx::call_packed_op())) {
      PrintPackedGlobalCall(call, os, false);
      return;
    }
    if (!call->op.same_as(tirx::call_ffi_kernel_op()) || device_kind_.empty()) {
      CodeGenCHost::Dispatch_(call, os);
      return;
    }
    TVM_FFI_CHECK(current_stream_, ValueError)
        << "A source host kernel launch requires an explicit stream parameter";
    auto symbol = call->args[0].as_or_throw<StringImm>()->value;
    auto found = kernels_.find(symbol);
    TVM_FFI_CHECK(found != kernels_.end(), ValueError)
        << "Source host refers to missing device kernel " << symbol;
    const auto *attrs = call->attrs
                            ? call->attrs.value().as<tirx::CallFFIKernelAttr>()
                            : nullptr;
    TVM_FFI_CHECK(attrs, ValueError)
        << "Kernel launch requires native CallFFIKernelAttr";
    if (device_kind_ != "cuda") {
      PrintGenericLaunch(call, *attrs, found->second, symbol);
      os << "0";
      return;
    }
    tvm_cuda_launch::PackedLaunchPlan plan(
        attrs->launch_fields, attrs->launch_params, attrs->kernel_attrs);
    size_t num_args = found->second->params.size();
    TVM_FFI_CHECK(call->args.size() == 1 + num_args + plan.num_values,
                  ValueError)
        << "Source host launch arity disagrees with the device Function";
    TVM_FFI_CHECK(attrs->num_kernel_args < 0 ||
                      static_cast<size_t>(attrs->num_kernel_args) == num_args,
                  ValueError)
        << "Source host launch metadata disagrees with the device Function";
    // Evaluate all actual call operands once, in order, at this call site.
    // Dynamic bounds, local variables and repeated calls never come from the
    // device signature or a search in its printed representation.
    std::vector<std::string> arguments;
    for (size_t i = 1; i < call->args.size(); ++i) {
      std::string value = PrintExpr(call->args[i]);
      std::string name = name_supply_->FreshName("launch_arg");
      PrintIndent();
      stream << "auto " << name << " = " << value << ";\n";
      arguments.push_back(name);
    }
    auto values = name_supply_->FreshName("launch_values");
    PrintIndent();
    stream << "tvm_cuda_launch::LaunchValues " << values << ";\n";
    for (const auto &operand : plan.operands) {
      const auto &field =
          tvm_cuda_launch::kFields[static_cast<size_t>(operand.field)];
      std::string setter = "SetInt";
      std::string value = std::to_string(operand.constant);
      if (operand.index >= 0) {
        size_t index = num_args + operand.index;
        const Type &type = call->args[index + 1]->ty;
        auto primitive = type.as<PrimType>();
        value = arguments[index];
        switch (field.kind) {
        case tvm_cuda_launch::ValueKind::kInteger:
          TVM_FFI_CHECK(primitive && primitive->IsScalar() &&
                            primitive->MatchesCode(kDLInt, kDLUInt),
                        TypeError)
              << "CUDA launch field " << field.name
              << " requires a scalar integer";
          break;
        case tvm_cuda_launch::ValueKind::kFloat:
          TVM_FFI_CHECK(primitive && primitive->IsScalar() &&
                            primitive->MatchesCode(kDLFloat),
                        TypeError)
              << "CUDA launch field " << field.name
              << " requires a scalar float";
          setter = "SetFloat";
          break;
        case tvm_cuda_launch::ValueKind::kHandle:
          TVM_FFI_CHECK(type.as<PtrTypeObj>(), TypeError)
              << "CUDA launch field " << field.name << " requires a pointer";
          setter = "SetHandle";
          value = "const_cast<void*>(static_cast<const void*>(" + value + "))";
          break;
        }
      }
      PrintIndent();
      stream << values << "." << setter
             << "(static_cast<tvm_cuda_launch::Field>("
             << static_cast<int>(operand.field) << "), " << value << ");\n";
    }
    if (plan.legacy) {
      PrintIndent();
      stream << "tvm_cuda_launch::NormalizeLegacyPresence(&" << values
             << ");\n";
    }
    auto requirements = name_supply_->FreshName("kernel_requirements");
    PrintIndent();
    stream << "tvm_cuda_launch::KernelRequirements " << requirements << "{"
           << (plan.requirements.required_block_size ? "true" : "false")
           << ", {";
    for (int axis = 0; axis != 3; ++axis) {
      if (axis)
        stream << ", ";
      stream << plan.requirements.block[axis];
    }
    stream << "}, {";
    for (int axis = 0; axis != 3; ++axis) {
      if (axis)
        stream << ", ";
      stream << plan.requirements.cluster[axis];
    }
    stream << "}};\n";
    auto parameter_array = name_supply_->FreshName("kernel_arguments");
    PrintIndent();
    stream << "std::array<void*, " << num_args << "> " << parameter_array
           << "{{";
    for (size_t i = 0; i < num_args; ++i) {
      if (i)
        stream << ", ";
      const auto *pointer = found->second->params[i]->ty.as<PtrTypeObj>();
      bool by_value_map =
          pointer && pointer->element_type.as<tirx::TensorMapTypeObj>();
      stream << "(void*)" << (by_value_map ? "" : "&") << arguments[i];
    }
    stream << "}};\n";
    auto source_stream = PrintExpr(current_stream_.value());
    PrintIndent();
    stream << "tl_source_cuda_launch((const void*)" << symbol << ", " << values
           << ", " << requirements << ", " << source_stream << ", "
           << parameter_array << ".data());\n";
    os << "0";
  }

  void Dispatch_(const EvaluateObj *statement) final {
    if (const auto *call = statement->value.as<CallObj>();
        call && call->op.same_as(tirx::call_packed_op())) {
      PrintIndent();
      stream << "(void)";
      PrintPackedGlobalCall(call, stream, true);
      stream << ";\n";
      return;
    }
    CodeGenCHost::Dispatch_(statement);
  }

private:
  std::ostringstream function_allocations_;
  std::streampos allocation_position_{-1};
  size_t packed_call_depth_{0};

  void PrintPackedGlobalCall(const CallObj *call, std::ostream &os,
                             bool discard_result) {
    TVM_FFI_CHECK(!call->args.empty() && call->args[0].as<StringImmObj>(),
                  ValueError)
        << "A source host packed call requires a static global function name";
    std::ostringstream expression;
    std::ostringstream surrounding;
    stream.swap(surrounding);
    int scope = BeginScope();
    ++packed_call_depth_;
    try {
      std::vector<std::string> arguments;
      for (size_t i = 1; i < call->args.size(); ++i) {
        const auto &argument = call->args[i];
        const auto primitive = argument->ty.as<PrimType>();
        const bool pointer = argument->ty.as<PtrTypeObj>() != nullptr;
        TVM_FFI_CHECK(argument.as<StringImmObj>() || pointer ||
                          (primitive && primitive->IsScalar() &&
                           primitive.value() != PrimType::Void()),
                      TypeError)
            << "Source host packed arguments require scalar, pointer or string "
               "types";
        auto value = PrintExpr(argument);
        if (pointer)
          value = "static_cast<void*>(" + value + ")";
        auto name = name_supply_->FreshName("packed_arg");
        PrintIndent();
        stream << "auto " << name << " = " << value << ";\n";
        arguments.push_back(std::move(name));
      }
      PrintIndent();
      stream << "return tl_source_call_packed(" << PrintExpr(call->args[0])
             << ", {";
      for (size_t i = 0; i < arguments.size(); ++i) {
        if (i)
          stream << ", ";
        stream << arguments[i];
      }
      stream << "});\n";
      expression << "([&]() {\n" << stream.str() << "}())";
    } catch (...) {
      --packed_call_depth_;
      EndScope(scope);
      stream.swap(surrounding);
      throw;
    }
    --packed_call_depth_;
    EndScope(scope);
    stream.swap(surrounding);
    const auto empty_tuple = call->ty.as<TupleTypeObj>();
    const auto primitive_result = call->ty.as<PrimType>();
    if (discard_result || (primitive_result && primitive_result->IsVoid()) ||
        (empty_tuple && empty_tuple->fields.empty())) {
      os << expression.str();
      return;
    }
    std::ostringstream type_name;
    PrintType(call->ty, type_name);
    if (call->ty.as<PtrTypeObj>()) {
      os << "static_cast<" << type_name.str() << ">(" << expression.str()
         << ".cast<void*>())";
    } else {
      const auto primitive = call->ty.as<PrimType>();
      TVM_FFI_CHECK(primitive && primitive->IsScalar(), TypeError)
          << "Source host packed results require scalar, pointer or void types";
      os << expression.str() << ".cast<" << type_name.str() << ">()";
    }
  }

  void PrintGenericLaunch(const CallObj *call,
                          const tirx::CallFFIKernelAttr &attrs,
                          const tirx::Function &kernel,
                          const std::string &symbol) {
    TVM_FFI_CHECK(attrs.launch_fields.empty() && attrs.kernel_attrs.empty(),
                  ValueError)
        << "CUDA launch configuration cannot be used by " << device_kind_;
    size_t num_args = kernel->params.size();
    TVM_FFI_CHECK(attrs.num_kernel_args < 0 ||
                      static_cast<size_t>(attrs.num_kernel_args) == num_args,
                  ValueError)
        << "Source host launch metadata disagrees with the device Function";
    TVM_FFI_CHECK(call->args.size() ==
                      1 + num_args + attrs.launch_params.size(),
                  ValueError)
        << "Source host launch arity disagrees with the device Function";
    std::vector<std::string> arguments;
    for (size_t i = 1; i < call->args.size(); ++i) {
      std::string expression = PrintExpr(call->args[i]);
      std::string temporary = name_supply_->FreshName("launch_arg");
      PrintIndent();
      stream << "auto " << temporary << " = " << expression << ";\n";
      arguments.push_back(temporary);
    }
    static const std::array<std::string, 6> tags{"blockIdx.x",  "blockIdx.y",
                                                 "blockIdx.z",  "threadIdx.x",
                                                 "threadIdx.y", "threadIdx.z"};
    std::array<std::string, 6> dimensions{"1", "1", "1", "1", "1", "1"};
    std::unordered_set<std::string> seen;
    std::string shared = "0";
    for (size_t i = 0; i < attrs.launch_params.size(); ++i) {
      std::string tag = attrs.launch_params[i];
      TVM_FFI_CHECK(seen.insert(tag).second, ValueError)
          << "Duplicate kernel launch tag " << tag;
      const Type &type = call->args[1 + num_args + i]->ty;
      auto primitive = type.as<PrimType>();
      TVM_FFI_CHECK(primitive && primitive->IsScalar() &&
                        primitive->MatchesCode(kDLInt, kDLUInt),
                    TypeError)
          << "Kernel launch field " << tag << " requires a scalar integer";
      auto value = arguments[num_args + i];
      auto position = std::find(tags.begin(), tags.end(), tag);
      if (tag == "tirx.use_dyn_shared_memory") {
        TVM_FFI_CHECK_EQ(i + 1, attrs.launch_params.size(), ValueError)
            << "Dynamic shared-memory bytes must be the last launch operand";
        shared = name_supply_->FreshName("shared_bytes");
        PrintIndent();
        stream << "size_t " << shared << " = tl_source_shared_bytes(" << value
               << ");\n";
      } else {
        TVM_FFI_CHECK(position != tags.end(), ValueError)
            << "Unsupported " << device_kind_ << " launch tag " << tag;
        size_t index = position - tags.begin();
        TVM_FFI_CHECK(device_kind_ != "ascend" || index < 3, ValueError)
            << "An Ascend kernel launch has a core grid and no thread block";
        dimensions[index] = name_supply_->FreshName("launch_dim");
        PrintIndent();
        stream << "uint32_t " << dimensions[index]
               << " = tl_source_launch_dimension(" << value << ");\n";
      }
    }
    auto source_stream = PrintExpr(current_stream_.value());
    if (device_kind_ == "ascend") {
      auto grid = name_supply_->FreshName("core_grid");
      PrintIndent();
      stream << "uint64_t " << grid << " = static_cast<uint64_t>("
             << dimensions[0] << ") * " << dimensions[1] << ";\n";
      // Check before multiplying the third axis, avoiding uint64 overflow.
      PrintIndent();
      stream << "TVM_FFI_CHECK(" << grid << " <= UINT32_MAX / " << dimensions[2]
             << ", ValueError) << \"Ascend core grid exceeds uint32\";\n";
      PrintIndent();
      stream << grid << " *= " << dimensions[2] << ";\n";
      auto print_type =
          ffi::Function::GetGlobalRequired("tl.codegen.AscendArgumentType");
      PrintIndent();
      stream << symbol << "<<<static_cast<uint32_t>(" << grid << "), " << shared
             << ", " << source_stream << ">>>(";
      for (size_t i = 0; i < num_args; ++i) {
        if (i)
          stream << ", ";
        // The device emitter owns FP8/FP4 and address-space spellings. The
        // host legalizer may use byte storage while the device takes that
        // exact typed pointer; no signature is recovered from printed code.
        auto type = is_pto_
                        ? ffi::String(PTOArgumentType(kernel->params[i]->ty))
                        : print_type(kernel->params[i]->ty).cast<ffi::String>();
        stream << "(" << type << ")(" << arguments[i] << ")";
      }
      stream << ");\n";
      auto status = name_supply_->FreshName("launch_status");
      PrintIndent();
      stream << "aclError " << status
             << " = aclrtGetLastError(ACL_RT_THREAD_LEVEL);\n";
      PrintIndent();
      stream << "TVM_FFI_CHECK(" << status << " == ACL_SUCCESS, RuntimeError)"
             << " << \"Ascend kernel launch failed with aclError \" << "
             << status << ";\n";
    } else {
      auto parameters = name_supply_->FreshName("kernel_arguments");
      PrintIndent();
      stream << "std::array<void*, " << num_args << "> " << parameters << "{{";
      for (size_t i = 0; i < num_args; ++i) {
        if (i)
          stream << ", ";
        stream << "(void*)&" << arguments[i];
      }
      stream << "}};\n";
      PrintIndent();
      stream << "tl_source_hip_launch((const void*)" << symbol << ", dim3("
             << dimensions[0] << ", " << dimensions[1] << ", " << dimensions[2]
             << "), dim3(" << dimensions[3] << ", " << dimensions[4] << ", "
             << dimensions[5] << "), " << shared << ", " << source_stream
             << ", " << parameters << ".data());\n";
    }
  }

  std::string device_kind_;
  bool is_pto_{false};
  std::unordered_map<std::string, tirx::Function> kernels_;
  ffi::Optional<Var> current_stream_;
};

ffi::String GenerateSourceHost(IRModule module, bool include_preamble) {
  SourceHostCodegen codegen;
  codegen.InitSource(include_preamble);
  std::vector<std::pair<GlobalVar, tirx::Function>> functions;
  for (const auto &[global, base] : module->functions) {
    auto function = base.as_or_throw<tirx::Function>();
    TVM_FFI_CHECK(!function->HasNonzeroAttr(tirx::attr::kIsEntryFunc.name),
                  ValueError)
        << "A raw source entry must not request a packed main stub";
    functions.emplace_back(global, function);
  }
  std::sort(functions.begin(), functions.end(),
            [](const auto &left, const auto &right) {
              return left.first->name_hint < right.first->name_hint;
            });
  for (const auto &[global, function] : functions)
    codegen.DeclareFunction(global, function);
  for (const auto &[global, function] : functions)
    codegen.AddSourceFunction(global, function);
  return codegen.Finish();
}

ffi::String GenerateDeviceSourceHost(IRModule module, IRModule device,
                                     Target target) {
  SourceHostCodegen codegen;
  codegen.InitDeviceSource(device, target);
  std::vector<std::pair<GlobalVar, tirx::Function>> functions;
  for (const auto &[global, base] : module->functions)
    functions.emplace_back(global, base.as_or_throw<tirx::Function>());
  std::sort(functions.begin(), functions.end(),
            [](const auto &left, const auto &right) {
              return left.first->name_hint < right.first->name_hint;
            });
  for (const auto &[global, function] : functions)
    codegen.DeclareFunction(global, function);
  for (const auto &[global, function] : functions)
    codegen.AddSourceFunction(global, function);
  return codegen.Finish();
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.codegen.GenerateSourceHost",
                                   GenerateSourceHost);
  ffi::reflection::GlobalDef().def("tl.codegen.GenerateDeviceSourceHost",
                                   GenerateDeviceSourceHost);
}

} // namespace tvm::tl
