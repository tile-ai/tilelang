/*!
 * \file codegen_c_host.cc
 */
#include "codegen_c_host.h"
#include "op/builtin.h"
#include "support/check.h"

#include <tvm/codegen/codegen.h>
#include <tvm/ffi/extra/module.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// For escaping strings embedded into generated C sources
#include "codegen_utils.h"

namespace tvm {
namespace tl {

using namespace ffi;

namespace {
bool IsOpaqueHostPointer(const Type &type) {
  const auto *pointer = type.as<PtrTypeObj>();
  if (!pointer)
    return false;
  const auto *element = pointer->element_type.as<PrimTypeObj>();
  if (!element)
    return false;
  PrimType dtype = ffi::GetRef<PrimType>(element);
  if (dtype.IsVoid() || dtype == PrimType::Bool())
    return false;
  // Device-only element types need an address on the host, not a host scalar
  // implementation. Actual loads/stores still require storage legalization.
  return !dtype.IsScalar() || dtype.bits() < 8 ||
         !dtype.MatchesCode(kDLInt, kDLUInt, kDLFloat, kDLOpaqueHandle);
}
} // namespace

CodeGenCHost::CodeGenCHost() {
  module_name_ = name_supply_->FreshName(symbol::tvm_ffi_library_ctx);
}

void CodeGenCHost::Init(bool output_ssa, bool emit_asserts,
                        bool emit_fwd_func_decl, std::string target_str,
                        const std::unordered_set<std::string> &devices) {
  emit_asserts_ = emit_asserts;
  emit_fwd_func_decl_ = emit_fwd_func_decl;
  declared_globals_.clear();
  preamble_stream_ << "// tilelang target: " << target_str << "\n";
  preamble_stream_ << "#define TVM_EXPORTS\n";
  preamble_stream_ << "#include \"tvm/runtime/base.h\"\n";
  preamble_stream_ << "#include \"tvm/runtime/c_backend_api.h\"\n";
  preamble_stream_ << "#include \"tvm/ffi/c_api.h\"\n";
  preamble_stream_ << "#include <tvm/ffi/extra/c_env_api.h>\n";
  preamble_stream_ << "#include <math.h>\n";
  // snprintf for richer assert messages with actual values
  preamble_stream_ << "#include <stdio.h>\n";
  preamble_stream_ << "#include <stdbool.h>\n";
  preamble_stream_ << "#include <string.h>\n";
  // CPU compute code includes half.hpp. Host declarations only need to name
  // its pointer type, including when the host unit is compiled as plain C.
  preamble_stream_ << "#ifdef __cplusplus\n"
                      "namespace half_float { class half; }\n"
                      "using half_float::half;\n"
                      "#else\n"
                      "typedef struct TLFloat16 half;\n"
                      "#endif\n";
  // Portable alignment specifier: MSVC uses __declspec(align(N)), every other
  // compiler nvcc invokes (gcc/clang) accepts __attribute__((aligned(N))).
  preamble_stream_ << "#if defined(_MSC_VER)\n";
  preamble_stream_ << "#define TL_ALIGN(N) __declspec(align(N))\n";
  preamble_stream_ << "#else\n";
  preamble_stream_ << "#define TL_ALIGN(N) __attribute__((aligned(N)))\n";
  preamble_stream_ << "#endif\n";
  // Match TensorMapType's 128-byte payload and 64-byte host alignment. Keeping
  // an opaque host record avoids requiring CUDA SDK headers in C host builds.
  preamble_stream_ << "typedef struct TL_ALIGN(64) TLTensorMap {\n"
                   << "  unsigned char data[128];\n"
                   << "} TLTensorMap;\n";

  preamble_stream_ << "#ifdef __OBJC__\n";
  preamble_stream_ << "#include \"tvm/runtime/device_api.h\"\n";
  preamble_stream_ << "#include \"tvm/ffi/function.h\"\n";

  preamble_stream_ << "#include <Metal/Metal.h>\n";
  preamble_stream_ << "#include <Foundation/Foundation.h>\n";

  preamble_stream_ << "#include <torch/mps.h>\n";
  preamble_stream_ << "#endif\n";

  CodeGenCHost::InitGlobalContext();
  tvm::codegen::CodeGenC::Init(output_ssa);
}

void CodeGenCHost::InitGlobalContext() {
  // Host and CPU compute modules can be emitted as separate C translation
  // units and linked into one library. Match LLVM's link-once context slot:
  // all units use the single exported address initialized by the FFI loader.
  // On ELF the slot must also bind locally: RTLD_GLOBAL must not redirect a
  // second library's packed imports through the first library's context.
  decl_stream << "#ifdef __cplusplus\nextern \"C\" {\n#endif\n"
              << "TVM_FFI_WEAK\n"
              << "#if defined(__ELF__)\n"
              << "__attribute__((visibility(\"protected\")))\n"
              << "#else\nTVM_FFI_DLL_EXPORT\n#endif\n"
              << "void* " << symbol::tvm_ffi_library_ctx << " = NULL;\n"
              << "#ifdef __cplusplus\n}\n#endif\n";
}

void CodeGenCHost::DefineModuleName() {
  decl_stream << "void* " << module_name_ << " = NULL;\n";
}

void CodeGenCHost::AddFunction(const tvm::GlobalVar &gvar,
                               const tvm::tirx::Function &func) {
  return AddFunction(gvar, func, /*emit_fwd_func_decl=*/false);
}

void CodeGenCHost::AddFunction(const tvm::GlobalVar &gvar,
                               const tvm::tirx::Function &func,
                               bool emit_fwd_func_decl) {
  auto global_symbol = func->GetAttr(tvm::attr::kGlobalSymbol);
  if (global_symbol) {
    function_names_.push_back(global_symbol.value());
  }

  emit_fwd_func_decl_ = emit_fwd_func_decl;
  tvm::codegen::CodeGenC::AddFunction(gvar, func);
  if (func->HasNonzeroAttr(tvm::tirx::attr::kIsEntryFunc.name) &&
      !has_main_func_) {
    ICHECK(global_symbol.has_value())
        << "CodeGenCHost: The entry func must have the global_symbol "
           "attribute, "
        << "but function " << gvar << " only has attributes " << func->attrs;
    if (global_symbol.value() != symbol::tvm_ffi_main) {
      function_names_.push_back(symbol::tvm_ffi_main);
      stream << "// CodegenC: NOTE: Auto-generated entry function\n";
      PrintFuncPrefix(stream);
      PrintType(func->ret_type, stream);
      stream << " " << symbol::tvm_ffi_main
             << "(void* self, void* args,int num_args, void* result) {\n";
      stream << "  return " << static_cast<std::string>(global_symbol.value())
             << "(self, args, num_args, result);\n";
      stream << "}\n";
    }
    has_main_func_ = true;
  }
}

void CodeGenCHost::GenerateForwardFunctionDeclarations(
    String global_symbol, const Array<tvm::Type> &arg_types,
    const tvm::Type &ret_type) {
  if (!emit_fwd_func_decl_) {
    return;
  }
  // string.h already declares these byte-oriented functions. Tensor pointer
  // arguments convert at the call site; using their pointee types to invent
  // a new prototype conflicts with the C library declaration.
  static const std::unordered_set<std::string> string_functions = {
      "memcpy", "memmove", "memset", "memcmp", "strlen"};
  if (string_functions.count(std::string(global_symbol))) {
    return;
  }
  for (auto &func_already_defined : GetFunctionNames()) {
    if (global_symbol == func_already_defined) {
      return;
    }
  }
  this->PrintFuncPrefix(fwd_decl_stream);
  this->PrintType(ret_type, fwd_decl_stream);
  fwd_decl_stream << " " << global_symbol << "(";
  for (size_t i = 0; i < arg_types.size(); ++i) {
    if (i > 0) {
      fwd_decl_stream << ", ";
    }
    PrintType(arg_types[i], fwd_decl_stream);
  }
  fwd_decl_stream << ");\n";
}

void CodeGenCHost::PrintFuncPrefix(std::ostream &os) { // NOLINT(*)
  os << "#ifdef __cplusplus\n"
     << "extern \"C\"\n"
     << "#endif\n";
}

void CodeGenCHost::PrintFunctionSignature(const ffi::String &name,
                                          const tirx::Function &function,
                                          std::ostream &os) {
  bool has_tensor_map = false;
  for (const Var &parameter : function->params)
    if (const auto *pointer = parameter->ty.as<PtrTypeObj>())
      has_tensor_map |=
          pointer->element_type.as<tirx::TensorMapTypeObj>() != nullptr;
  if (!has_tensor_map) {
    tvm::codegen::CodeGenC::PrintFunctionSignature(name, function, os);
    return;
  }
  // The generic emitter treats TensorMap pointers as CUDA grid-constant
  // values. Host helpers instead receive pointers to the 128-byte host
  // storage object. Keep normal naming, alias and element-type registration
  // while consuming this host-only ABI here.
  std::unordered_set<size_t> may_alias;
  if (auto indices = function->GetAttr<ffi::Array<int64_t>>(
          "tl.may_alias_param_indices")) {
    for (int64_t index : indices.value()) {
      TVM_FFI_CHECK(index >= 0 &&
                        static_cast<size_t>(index) < function->params.size(),
                    ValueError)
          << "may_alias_param_indices contains an invalid parameter index";
      TVM_FFI_CHECK(function->params[index]->ty.as<PtrTypeObj>(), TypeError)
          << "may_alias_param_indices must name pointer parameters";
      may_alias.insert(index);
    }
  }
  PrintFuncPrefix(os);
  PrintType(function->ret_type, os);
  PrintExtraAttrs(function, os);
  os << " " << name << "(";
  for (size_t i = 0; i < function->params.size(); ++i) {
    const Var &parameter = function->params[i];
    if (i != 0)
      os << ", ";
    PrintStorageQualifier(parameter.get(), os);
    PrintType(parameter->ty, os);
    if (parameter->ty.as<PtrTypeObj>() &&
        function->HasNonzeroAttr(tirx::attr::kNoAlias.name) &&
        !may_alias.count(i))
      PrintRestrict(parameter, os);
    os << " " << AllocVarID(parameter.get());
  }
  os << ")";
  for (const Var &parameter : function->params)
    if (const auto *pointer = parameter->ty.as<PtrTypeObj>())
      if (const auto *element = pointer->element_type.as<PrimTypeObj>())
        RegisterHandleType(parameter.get(), ffi::GetRef<PrimType>(element));
}

void CodeGenCHost::PrintType(const tvm::Type &type, std::ostream &os) {
  if (IsOpaqueHostPointer(type)) {
    os << "void*";
    return;
  }
  if (type.as<tirx::TensorMapTypeObj>()) {
    os << "TLTensorMap";
    return;
  }
  tvm::codegen::CodeGenC::PrintType(type, os);
}

void CodeGenCHost::PrintType(const tvm::PrimType &t,
                             std::ostream &os) { // NOLINT(*)
  if (t.IsVoid()) {
    os << "void";
    return;
  }
  int lanes = t.lanes();
  if (t.MatchesCode(kDLOpaqueHandle)) {
    ICHECK_EQ(lanes, 1) << "does not support vector types";
    os << "void*";
    return;
  }
  if (t == tvm::PrimType::Bool()) {
    os << "bool";
    return;
  }
  bool fail = false;
  if (t.MatchesCode(kDLFloat)) {
    switch (t.bits()) {
    case 16:
      os << "half";
      break;
    case 32:
      os << "float";
      break;
    case 64:
      os << "double";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1)
      return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  }
  if (t.MatchesElementType(kDLBfloat, 16)) {
    os << "__bf16";
    return;
  }
  if (t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt)) {
    if (t.MatchesCode(kDLUInt)) {
      os << 'u';
    }
    switch (t.bits()) {
    case 8:
      os << "int8_t";
      break;
    case 16:
      os << "int16_t";
      break;
    case 32:
      os << "int32_t";
      break;
    case 64:
      os << "int64_t";
      break;
    case 1:
      os << "int32_t";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1)
      return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  }
  TVM_FFI_THROW(InternalError) << "Cannot convert type " << t << " to C type";
}

void CodeGenCHost::Dispatch_(const tvm::FloatImmObj *op, std::ostream &os) {
  PrimType type = op->ty.as_or_throw<PrimType>();
  if (type.MatchesCode(kDLFloat) && (type.bits() == 32 || type.bits() == 64) &&
      !std::isfinite(op->value)) {
    // Reducer identities include infinities. iostream spellings such as
    // "inff" are not C literals; math.h provides portable typed expressions.
    std::ostringstream literal;
    literal << "((";
    PrintType(type, literal);
    literal << ")";
    if (std::signbit(op->value))
      literal << '-';
    literal << (std::isnan(op->value) ? "NAN" : "INFINITY") << ')';
    MarkConst(literal.str());
    os << literal.str();
    return;
  }
  tvm::codegen::CodeGenC::Dispatch_(op, os);
}

void CodeGenCHost::Dispatch_(const tvm::prim::BroadcastObj *op,
                             std::ostream &os) { // NOLINT(*)
  std::string v = PrintExpr(op->value);
  int lanes = op->ty.as_or_throw<PrimType>().lanes();
  os << "((";
  PrintType(op->ty.as_or_throw<PrimType>(), os);
  os << ")(";
  for (int i = 0; i < lanes; ++i) {
    if (i != 0)
      os << ", ";
    os << v;
  }
  os << "))";
}

void CodeGenCHost::PrintGetFuncFromBackend(
    const std::string &func_name, const std::string &packed_func_name) {
  this->PrintIndent();
  this->stream << "if (" << packed_func_name << " == NULL) {\n";
  int packed_func_if_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "if (TVMFFIEnvModLookupFromImports(" << module_name_ << ", \""
               << func_name << "\""
               << ", &" << packed_func_name << ") != 0) {\n";
  int get_func_env_scope = this->BeginScope();
  PrintLine("return -1;");
  this->EndScope(get_func_env_scope);
  this->PrintIndent();
  this->stream << "}\n";
  this->EndScope(packed_func_if_scope);
  this->PrintIndent();
  this->stream << "}\n";
}

void CodeGenCHost::PrintCallPacked(const tvm::CallObj *op, std::ostream &os) {
  using namespace tvm::tirx;
  const StringImmObj *func_name = op->args[0].as<StringImmObj>();
  ICHECK(func_name != nullptr)
      << "tvm_call_[c]packed_lowered expects first argument as function name";
  int64_t begin = op->args[2].as<IntImmObj>()->value.as<int64_t>().value();
  int64_t end = op->args[3].as<IntImmObj>()->value.as<int64_t>().value();
  int64_t num_args = end - begin;
  ICHECK_GE(num_args, 0);

  std::string packed_func_name;
  if (op->op.same_as(tirx::call_packed_lowered_op())) {
    packed_func_name = GetPackedName(op);
    this->PrintGetFuncFromBackend(func_name->value, packed_func_name);
  } else {
    // directly use the original symbol
    ICHECK(op->op.same_as(tirx::call_cpacked_lowered_op()));
    packed_func_name = symbol::tvm_ffi_symbol_prefix + func_name->value;
  }

  std::string args_stack = PrintExpr(op->args[1]);
  this->PrintIndent();
  std::string result = name_supply_->FreshName("result");
  if (is_in_metal_context) {
    this->stream << "__block ";
  }
  this->stream << "TVMFFIAny* " << result << " = ((TVMFFIAny*) " << args_stack
               << ") + " << end << ";\n";
  this->PrintIndent();
  // must make sure type_index is set to none
  this->stream << result << "->type_index = kTVMFFINone;\n";
  this->PrintIndent();
  this->stream << result << "->zero_padding = 0;\n";
  this->PrintIndent();
  this->stream << result << "->v_int64 = 0;\n";

  int metal_scope = 0;
  std::string metal_result;
  if (is_in_metal_context) {
    metal_result = name_supply_->FreshName("metal_ret");
    this->PrintLine("__block int ", metal_result, " = 0;");
    this->PrintLine("auto serialQueue = torch::mps::get_dispatch_queue();");
    this->PrintLine("dispatch_sync(serialQueue, ^() {");
    metal_scope = this->BeginScope();

    this->PrintLine("const id<MTLCommandBuffer> commandBuffer = "
                    "torch::mps::get_command_buffer();");
    this->PrintLine(
        "const auto f = tvm::ffi::Function::GetGlobal(\"metal.SetStream\");");
    this->PrintLine("(*f)(static_cast<TVMStreamHandle>(commandBuffer));");
  }

  this->PrintIndent();
  if (op->op.same_as(tirx::call_packed_lowered_op())) {
    this->stream << "if (TVMFFIFunctionCall(" << packed_func_name << ", ";
  } else {
    this->stream << "if (" << packed_func_name << "(NULL, ";
  }
  this->stream << "((TVMFFIAny*) " << args_stack << ") + " << begin << ", "
               << num_args << ", " << result << ") != 0) {\n";
  int func_call_scope = this->BeginScope();
  if (is_in_metal_context) {
    this->PrintLine(metal_result, " = -1;");
  } else {
    PrintLine("return -1;");
  }
  this->EndScope(func_call_scope);

  this->PrintLine("}");

  if (is_in_metal_context) {
    this->EndScope(metal_scope);
    this->PrintLine("});");
    this->PrintLine("if (", metal_result, " != 0) {");
    int error_scope = this->BeginScope();
    PrintLine("return -1;");
    this->EndScope(error_scope);
    this->PrintLine("}");
  }

  const auto primitive_result = op->ty.as<PrimType>();
  if (IsVoidType(op->ty) || (primitive_result && primitive_result->IsVoid()))
    return;

  const char *field = "v_ptr";
  if (!op->ty.as<PtrTypeObj>()) {
    const auto type = op->ty.as_or_throw<PrimType>();
    TVM_FFI_CHECK(type.IsScalar(), TypeError)
        << "Packed calls require a scalar or pointer result";
    if (type.MatchesCode(kDLFloat)) {
      field = "v_float64";
    } else {
      TVM_FFI_CHECK(type.MatchesCode(kDLInt, kDLUInt, kDLBool), TypeError)
          << "Unsupported packed call result type " << type;
      field = "v_int64";
    }
  }
  // A following packed call may reuse this result slot. Capture the value now
  // so nested expressions keep each call's result and execute it only once.
  std::ostringstream result_type;
  PrintType(op->ty, result_type);
  std::string value = name_supply_->FreshName("packed_result");
  PrintLine(result_type.str(), " ", value, " = ((", result_type.str(), ")",
            result, "->", field, ");");
  os << value;
}

std::string CodeGenCHost::GetPackedName(const tvm::CallObj *op) {
  using namespace tvm::tirx;
  const StringImmObj *s = op->args[0].as<StringImmObj>();
  ICHECK(s != nullptr)
      << "tvm_call_packed_lowered expects first argument as function name";
  std::string func_name = s->value;
  std::string packed_func_name = func_name + "_packed";
  std::string unique_name;
  auto it = declared_globals_.find(packed_func_name);
  if (it != declared_globals_.end()) {
    unique_name = it->second;
  } else {
    unique_name = name_supply_->FreshName(packed_func_name);
    declared_globals_[packed_func_name] = unique_name;
    decl_stream << "static void* " << unique_name << " = NULL;\n";
  }
  return unique_name;
}

void CodeGenCHost::Dispatch_(const tvm::CallObj *op,
                             std::ostream &os) { // NOLINT(*)
  using namespace tvm::tirx;
  if (op->op.same_as(tl::lookup_param())) {
    ffi::GetRef<Call>(op).Validate();
    os << "__tvm_param__" << op->args[0].as_or_throw<StringImm>()->value;
  } else if (op->op.same_as(tirx::reinterpret_op()) &&
             op->ty.as<PtrTypeObj>()) {
    os << "((";
    PrintType(op->ty, os);
    os << ")" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tirx::reinterpret_op()) &&
             op->ty.as<PrimTypeObj>() && op->args[0]->ty.as<PrimTypeObj>() &&
             op->ty.as_or_throw<PrimType>().IsScalar() &&
             op->args[0]->ty.as_or_throw<PrimType>().IsScalar()) {
    const auto from = op->args[0]->ty.as_or_throw<PrimType>();
    const auto to = op->ty.as_or_throw<PrimType>();
    TVM_FFI_CHECK_EQ(from.bits(), to.bits(), TypeError)
        << "reinterpret expects source and target to have the same number of "
           "bits";
    TVM_FFI_CHECK(from.bits() >= 8 && from.bits() % 8 == 0, TypeError)
        << "C host scalar reinterpret requires byte-sized primitive storage; "
           "lower sub-byte values before code generation";
    // A literal or arithmetic result has no address. A small inline helper
    // copies the bits without type-punning and keeps argument evaluation at
    // the call site, including inside a conditional branch.
    const std::string helper = AllocHelperFunctionName("__tl_bitcast");
    decl_stream << "static inline ";
    PrintType(to, decl_stream);
    decl_stream << ' ' << helper << '(';
    PrintType(from, decl_stream);
    decl_stream << " value) {\n  ";
    PrintType(to, decl_stream);
    decl_stream << " result;\n  memcpy(&result, &value, sizeof(result));\n"
                   "  return result;\n}\n";
    os << helper << '(' << PrintExpr(op->args[0]) << ')';
  } else if (op->op.same_as(tirx::ptr_byte_offset_op()) &&
             IsOpaqueHostPointer(op->ty)) {
    os << "((void*)((char*)" << PrintExpr(op->args[0]) << " + "
       << PrintExpr(op->args[1]) << "))";
  } else if (op->op.same_as(tl::tvm_ffi_call_with_result())) {
    ICHECK_EQ(op->args.size(), 4U);
    const auto *func_name = op->args[0].as<StringImmObj>();
    const auto *num_args = op->args[2].as<IntImmObj>();
    ICHECK(func_name != nullptr);
    ICHECK(num_args != nullptr);

    std::string packed_func_name = GetPackedName(op);
    PrintGetFuncFromBackend(func_name->value, packed_func_name);
    os << "TVMFFIFunctionCall(" << packed_func_name << ", (TVMFFIAny*) "
       << PrintExpr(op->args[1]) << ", " << num_args->value << ", (TVMFFIAny*) "
       << PrintExpr(op->args[3]) << ")";
  } else if (op->op.same_as(tirx::stack_alloca_op())) {
    std::string stack_name = name_supply_->FreshName("stack");
    const std::string &type = op->args[0].as<StringImmObj>()->value;
    const IntImmObj *num = op->args[1].as<IntImmObj>();
    ICHECK(num != nullptr);
    if (type == "tensormap") {
      const auto count = num->value.as<size_t>();
      TVM_FFI_CHECK(count.has_value() && *count > 0 &&
                        *count <= std::numeric_limits<size_t>::max() / 128,
                    ValueError)
          << "tensormap stack allocation requires a positive constant count";
      this->PrintIndent();
      this->stream << "TLTensorMap " << stack_name << "[" << *count << "];\n";
      os << stack_name;
      return;
    }
    static_assert(alignof(TVMFFIAny) % alignof(DLTensor) == 0, "invariant");
    size_t unit = sizeof(TVMFFIAny);
    size_t size = 0;
    if (type == "shape") {
      size =
          (num->value.as<size_t>().value() * sizeof(int64_t) + unit - 1) / unit;
    } else if (type == "tvm_ffi_any") {
      size = (num->value.as<size_t>().value() * sizeof(TVMFFIAny) + unit - 1) /
             unit;
    } else if (type == "array") {
      size = (num->value.as<size_t>().value() * sizeof(DLTensor) + unit - 1) /
             unit;
    } else {
      TVM_FFI_THROW(InternalError) << "Unknown stack alloca type " << type;
    }
    this->PrintIndent();
    if (type == "tvm_ffi_any") {
      // TMA descriptors are materialized through tvm_ffi_any stack allocas and
      // are passed by value to the kernel as ``__grid_constant__ const
      // CUtensorMap``.
      this->stream << "TL_ALIGN(128) TVMFFIAny " << stack_name << "[" << size
                   << "];\n";
    } else {
      this->stream << "TVMFFIAny " << stack_name << "[" << size << "];\n";
    }
    os << stack_name;
  } else if (op->op.same_as(tirx::call_packed_lowered_op())) {
    this->PrintCallPacked(op, os);
  } else if (op->op.same_as(tirx::call_cpacked_lowered_op())) {
    this->PrintCallPacked(op, os);
  } else if (op->op.same_as(tirx::throw_last_error_op())) {
    this->PrintIndent();
    this->stream << "return -1;\n";
  } else {
    tvm::codegen::CodeGenC::Dispatch_(op, os);
  }
}

void CodeGenCHost::Dispatch_(const tvm::EvaluateObj *op) { // NOLINT(*)
  using namespace tvm::tirx;
  if (const auto *call = op->value.as<CallObj>()) {
    if (call->op.same_as(tirx::abi_field_set_op())) {
      ICHECK_EQ(call->args.size(), 4U);
      const auto *field = call->args[2].as<IntImmObj>();
      ICHECK(field != nullptr);
      if (field->value == tirx::kDLTensorShape) {
        std::string ref = GetStructRef(call->args[3]->ty, call->args[0],
                                       call->args[1].as_or_throw<PrimExpr>(),
                                       tirx::kDLTensorShape);
        PrintIndent();
        stream << ref << " = (int64_t*)" << PrintExpr(call->args[3]) << ";\n";
        return;
      }
    }
  }
  tvm::codegen::CodeGenC::Dispatch_(op);
}

void CodeGenCHost::Dispatch_(const tvm::BindObj *op) {
  if (!IsOpaqueHostPointer(op->var->ty)) {
    if (const auto *call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::alloc_tensor_op()) && !call->attrs) {
      // Native alloc_tensor permits absent annotations, while the C emitter
      // expects a dictionary. Normalize only this emission-local call; keep
      // the allocation's Var identity, type, operands and source IR intact.
      Bind normalized(op->var,
                      Call(call->ty, call->op, call->args, DictAttrs(),
                           call->ty_args, call->loc),
                      op->loc);
      tvm::codegen::CodeGenC::Dispatch_(normalized.get());
      return;
    }
    tvm::codegen::CodeGenC::Dispatch_(op);
    return;
  }
  RegisterHandleTypeFromPointer(op->var, &op->value);
  std::string value = PrintExpr(op->value);
  if (print_ssa_form_) {
    TVM_FFI_ICHECK(!var_idmap_.count(op->var.get()));
    var_idmap_[op->var.get()] = value;
  } else {
    PrintIndent();
    // Preserve the pointer contract instead of printing an unsupported host
    // scalar through the generic emitter's handle_data_type_ shortcut.
    PrintType(op->var->ty, stream);
    stream << ' ' << AllocVarID(op->var.get()) << " = " << value << ";\n";
  }
}

void CodeGenCHost::Dispatch_(const tvm::AssertStmtObj *op) { // NOLINT(*)
  if (emit_asserts_) {
    std::string cond = PrintExpr(op->condition);
    std::string error_kind = tvm::codegen::EscapeCString(op->error_kind->value);
    PrintIndent();
    stream << "if (!(" << cond << ")) {\n";
    int assert_if_scope = this->BeginScope();
    {
      // Join message_parts into a single string
      std::string joined_msg;
      for (const auto &part : op->message_parts) {
        joined_msg += part->value;
      }
      if (!joined_msg.empty()) {
        std::string esc_msg = tvm::codegen::EscapeCString(joined_msg);

        // AssertStmt owns its diagnostic text. Preserve it exactly, and do
        // not re-evaluate either side of its condition while reporting errors.
        PrintIndent();
        stream << "TVMFFIErrorSetRaisedFromCStr(\"" << error_kind << "\", \""
               << esc_msg << "\");\n";
      } else {
        PrintIndent();
        stream << "TVMFFIErrorSetRaisedFromCStr(\"" << error_kind
               << "\", "
                  "\"assertion failed\");\n";
      }
    }
    PrintIndent();
    stream << "return -1;\n";
    this->EndScope(assert_if_scope);
    PrintIndent();
    stream << "}\n";
  }
}

void CodeGenCHost::Dispatch_(const tvm::RegionStmtObj *op) {
  if (op->op->name != "tl.attr")
    return tvm::codegen::CodeGenC::Dispatch_(op);
  auto attr = tvm::codegen::DecodeTileLangAttribute(op);
  TVM_FFI_CHECK_EQ(attr.key, "metal_context", ValueError)
      << "Unlowered TileLang host attribute " << attr.key;
  ICHECK(!is_in_metal_context) << "Nested metal context";
  is_in_metal_context = true;
  PrintStmt(op->body);
  is_in_metal_context = false;
}

void CodeGenCHost::Dispatch_(const tvm::prim::MinObj *op,
                             std::ostream &os) { // NOLINT(*)
  PrintTernaryCondExpr(op, "<", os);
}

void CodeGenCHost::Dispatch_(const tvm::prim::MaxObj *op,
                             std::ostream &os) { // NOLINT(*)
  PrintTernaryCondExpr(op, ">", os);
}

template <typename T>
inline void CodeGenCHost::PrintTernaryCondExpr(const T *op, const char *compare,
                                               std::ostream &os) { // NOLINT(*)
  std::ostringstream temp_a;
  Dispatch(op->a, temp_a);
  std::string a_id = SSAGetID(temp_a.str(), op->a.ty(),
                              SideEffect(op->a) == CallEffectKind::kPure);
  std::ostringstream temp_b;
  Dispatch(op->b, temp_b);
  std::string b_id = SSAGetID(temp_b.str(), op->b.ty(),
                              SideEffect(op->b) == CallEffectKind::kPure);

  os << "((" << a_id << ") " << compare << " (" << b_id << ") "
     << "? (" << a_id << ") : (" << b_id << "))";
}

} // namespace tl
} // namespace tvm

namespace tvm {
namespace tl {

using tvm::codegen::CodeGenSourceBase;
using tvm::codegen::CSourceModuleCreate;

// Build function that mirrors TVM's host C codegen, registered under a
// TileLang-specific name.
// NOTE(chaofan): Different from codegen_c / BuildTileLangC, this CodeGen class
// is only used to generate C host code for TileLang when the host codegen is
// enabled. If you use TileLang to generate CPU code (in this case, C is the
// device code) , it will be generated by BuildTileLangC.
Module BuildTileLangCHost(IRModule mod, Target target) {
  bool output_ssa = false;
  bool emit_asserts = true;
  bool emit_fwd_func_decl = true;

  std::unordered_set<std::string> devices;
  if (mod->GetAttr<Map<GlobalVar, String>>("device_contexts").has_value()) {
    Map<GlobalVar, String> device_contexts =
        mod->GetAttr<Map<GlobalVar, String>>("device_contexts").value();
    for (auto const &context : device_contexts) {
      devices.insert(context.second.data());
    }
  }

  CodeGenCHost cg;
  cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), devices);
  cg.SetEmitLineDirectives(tvm::codegen::TileLangEmitLineDirectivesEnabled());
  cg.SetConstantsByteAlignment(
      target->GetAttr<int64_t>("constants-byte-alignment").value_or(16));

  auto is_aot_executor_fn = [](::tvm::tirx::Function const &func) -> bool {
    return func->GetAttr<bool>("runner_function", false).value();
  };

  std::vector<std::pair<::tvm::GlobalVar, ::tvm::tirx::Function>> funcs;
  for (auto [gvar, base_func] : mod->functions) {
    ICHECK(base_func->IsInstance<::tvm::tirx::FunctionObj>())
        << "TileLangCodegenCHost: Can only take Function";
    auto prim_func = base_func.as_or_throw<::tvm::tirx::Function>();
    funcs.push_back({gvar, prim_func});
  }

  auto sort_key = [&is_aot_executor_fn](const auto &kv) {
    return std::tuple{is_aot_executor_fn(kv.second), kv.first->name_hint};
  };
  std::sort(funcs.begin(), funcs.end(),
            [&sort_key](const auto &kv_a, const auto &kv_b) {
              return sort_key(kv_a) < sort_key(kv_b);
            });

  for (const auto &[gvar, prim_func] : funcs) {
    cg.DeclareFunction(gvar, prim_func);
  }

  for (const auto &[gvar, prim_func] : funcs) {
    cg.AddFunction(gvar, prim_func, emit_fwd_func_decl);
  }

  std::string code = cg.Finish();
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_c_host_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  return ::tvm::codegen::CSourceModuleCreate(code, "c", cg.GetFunctionNames());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("target.build.tilelang_c_host", BuildTileLangCHost);
}

} // namespace tl
} // namespace tvm
