#include "op/dtype.h"
#include "tirx_compat/source_module.h"
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file codegen_c_host.cc
 */
#include "codegen_c_host.h"

#include "backend/common/codegen/attr.h"
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/abi.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace tvm::tl::codegen_impl {
namespace attr = tvm::tl::codegen_attr;

CodeGenCHost::CodeGenCHost() {
  module_name_ = name_supply_->FreshName(ffi::symbol::tvm_ffi_library_ctx);
}

void CodeGenCHost::Init(bool output_ssa, bool emit_asserts,
                        bool emit_fwd_func_decl, std::string target_str) {
  emit_asserts_ = emit_asserts;
  emit_fwd_func_decl_ = emit_fwd_func_decl;
  declared_globals_.clear();
  vector_types_.clear();
  decl_stream << "// tvm target: " << target_str << "\n";
  decl_stream << "#include \"tvm/ffi/extra/c_env_api.h\"\n";
  decl_stream << "#include \"tvm/runtime/base.h\"\n";
  decl_stream << "#include \"tvm/runtime/c_backend_api.h\"\n";
  decl_stream << "#include \"tvm/ffi/c_api.h\"\n";
  decl_stream << "#define TVM_DLL TVM_FFI_DLL_EXPORT\n";
  decl_stream << "#include <math.h>\n";
  decl_stream << "#include <stdbool.h>\n";
  decl_stream << "#if defined(_MSC_VER)\n"
              << "typedef __declspec(align(64)) struct { uint64_t data[16]; } "
                 "TVMTensorMap;\n"
              << "#else\n"
              << "typedef struct { uint64_t data[16]; } "
                 "__attribute__((aligned(64))) TVMTensorMap;\n"
              << "#endif\n";
  CodeGenCHost::InitGlobalContext();
  CodeGenC::Init(output_ssa);
  header_stream_ << decl_stream.str();
  decl_stream.str("");
}

void CodeGenCHost::InitGlobalContext() {
  decl_stream << "void* " << ffi::symbol::tvm_ffi_library_ctx << " = NULL;\n";
}

void CodeGenCHost::Dispatch_(const ForNode *op) {
  auto previous = parallel_chain_;
  if (op->kind == ForKind::kParallel && !parallel_chain_.count(op)) {
    int depth = 1;
    const ForNode *current = op;
    while (current->body->seq.size() == 1) {
      auto child = current->body->seq[0].as<ForNode>();
      if (!child || child->kind != ForKind::kParallel)
        break;
      parallel_chain_.insert(child);
      current = child;
      ++depth;
    }
    PrintIndent();
    stream << "#pragma omp parallel for";
    if (depth > 1)
      stream << " collapse(" << depth << ")";
    if (auto threads = op->annotations.Get("codegen.num_threads")) {
      auto count = threads->cast<IntImm>();
      TVM_FFI_CHECK(count->value > 0, ValueError)
          << "OpenMP thread count must be positive";
      stream << " num_threads(" << count->value << ")";
    }
    stream << "\n";
  }
  CodeGenC::Dispatch_(op);
  parallel_chain_ = std::move(previous);
}

void CodeGenCHost::DefineModuleName() {
  decl_stream << "void* " << module_name_ << " = NULL;\n";
}

void CodeGenCHost::AddFunction(const GlobalVar &gvar, const Function &func) {
  return AddFunction(gvar, func, /*emit_fwd_func_decl=*/false);
}

void CodeGenCHost::AddFunction(const GlobalVar &gvar, const Function &func,
                               bool emit_fwd_func_decl) {
  auto global_symbol = func->GetAttr(tvm::attr::kGlobalSymbol);
  if (global_symbol) {
    function_names_.push_back(global_symbol.value());
  }

  emit_fwd_func_decl_ = emit_fwd_func_decl;
  CodeGenC::AddFunction(gvar, func);
  if (func->HasNonzeroAttr(tvm::tirx::attr::kIsEntryFunc.name) &&
      !has_tvm_ffi_main_func_) {
    TVM_FFI_ICHECK(global_symbol.has_value())
        << "CodeGenCHost: The entry func must have the global_symbol "
           "attribute, "
        << "but function " << gvar << " only has attributes " << func->attrs;

    function_names_.push_back(ffi::symbol::tvm_ffi_main);
    stream << "// CodegenC: NOTE: Auto-generated entry function\n";
    PrintFuncPrefix(stream);
    PrintType(func->ret_type, stream);
    stream << " " << ffi::symbol::tvm_ffi_main
           << "(void* self, void* args,int num_args, void* result) {\n";
    stream << "  return " << global_symbol.value()
           << "(self, args, num_args, result);\n";
    stream << "}\n";
  }
}

void CodeGenCHost::GenerateForwardFunctionDeclarations(
    ffi::String global_symbol, const ffi::Array<Type> &arg_types,
    const Type &ret_type) {
  if (!emit_fwd_func_decl_ || global_symbol == "TVMFFIEnvTensorAlloc") {
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
    CodeGenSourceBase::PrintType(arg_types[i], fwd_decl_stream);
  }
  fwd_decl_stream << ");\n";
}

void CodeGenCHost::PrintCallExtern(Type ret_type, ffi::String global_symbol,
                                   const ffi::Array<Expr> &args,
                                   bool skip_first_arg, std::ostream &os) {
  if (global_symbol == "TVMFFIEnvTensorAlloc") {
    size_t begin = static_cast<size_t>(skip_first_arg);
    TVM_FFI_ICHECK_EQ(args.size(), begin + 2);
    os << "TVMFFIEnvTensorAlloc((DLTensor*) " << PrintExpr(args[begin])
       << ", (TVMFFIObjectHandle*) " << PrintExpr(args[begin + 1]) << ")";
  } else {
    CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg,
                              os);
  }
}

void CodeGenCHost::PrintFuncPrefix(std::ostream &os) { // NOLINT(*)
  os << "#ifdef __cplusplus\n"
     << "extern \"C\"\n"
     << "#endif\n"
     << "TVM_DLL ";
}

void CodeGenCHost::PrintStorageType(const Type &type, std::ostream &os) {
  if (auto pointer = type.as<PtrType>()) {
    PrintStorageType(pointer.value()->element_type, os);
    os << "*";
    return;
  }
  if (type.as<tirx::TensorMapTypeNode>()) {
    os << "TVMTensorMap";
    return;
  }
  if (auto primitive = type.as<PrimType>();
      primitive && primitive->IsScalar() &&
      primitive->MatchesElementType(kDLBfloat, 16)) {
    os << "uint16_t";
  } else if (primitive && primitive->IsScalar() && primitive->bits() > 0 &&
             primitive->bits() < 8 &&
             primitive->MatchesCode(kDLInt, kDLUInt, kDLFloat4_e2m1fn)) {
    os << "uint8_t";
  } else {
    PrintType(type, os);
  }
}

void CodeGenCHost::PrintType(const PrimType &type,
                             std::ostream &os) { // NOLINT(*)
  if (::tvm::tl::IsCustomDType(type)) {
    PrintType(::tvm::tl::CustomDTypeStorage(type), os);
    return;
  }
  int lanes = type.lanes();
  if (lanes > 1) {
    TVM_FFI_CHECK(lanes <= 16 && (lanes & (lanes - 1)) == 0, ValueError)
        << "C vectors require 2, 4, 8, or 16 lanes";
    PrimType element = type.WithLanes(1);
    if (element.MatchesCode(kDLBool))
      element = PrimType::UInt(8);
    std::ostringstream scalar;
    PrintType(element, scalar);
    std::string name = scalar.str() + std::to_string(lanes);
    if (vector_types_.insert(name).second) {
      header_stream_ << "typedef " << scalar.str() << " " << name
                     << " __attribute__((vector_size("
                     << element.StorageBytes() * lanes << "), aligned("
                     << element.StorageBytes() << "), may_alias));\n";
      header_stream_ << "static inline " << name << " tvm_make_" << name << "(";
      for (int index = 0; index < lanes; ++index) {
        if (index)
          header_stream_ << ", ";
        header_stream_ << scalar.str() << " value_" << index;
      }
      header_stream_ << ") { return (" << name << "){";
      for (int index = 0; index < lanes; ++index) {
        if (index)
          header_stream_ << ", ";
        header_stream_ << "value_" << index;
      }
      header_stream_ << "}; }\n";
    }
    os << name;
    return;
  }
  if (type.MatchesCode(DLDataTypeCode::kDLOpaqueHandle) && !type.IsVoid()) {
    TVM_FFI_ICHECK_EQ(lanes, 1) << "does not support vector types";
    os << "void*";
    return;
  }
  if (type.IsVoid()) {
    os << "void";
    return;
  }
  if (type.MatchesCode(DLDataTypeCode::kDLBool) && lanes == 1) {
    os << "bool";
    return;
  }
  bool fail = false;
  if (type.MatchesCode(DLDataTypeCode::kDLFloat)) {
    switch (type.bits()) {
    case 16:
      os << "_Float16";
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
  } else if (type.MatchesCode(DLDataTypeCode::kDLUInt,
                              DLDataTypeCode::kDLInt)) {
    if (type.MatchesCode(DLDataTypeCode::kDLUInt)) {
      os << 'u';
    }
    switch (type.bits()) {
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
  TVM_FFI_THROW(InternalError)
      << "Cannot convert type " << type->dtype << " to C type";
}

std::string CodeGenCHost::CastFromTo(std::string value, const PrimType &from,
                                     const PrimType &target) {
  if (from == target || (from.IsScalar() && target.IsScalar())) {
    return CodeGenC::CastFromTo(std::move(value), from, target);
  }
  TVM_FFI_CHECK_EQ(from.lanes(), target.lanes(), ValueError)
      << "C vector casts must preserve the number of lanes";
  std::string source = SSAGetID(std::move(value), from);
  std::ostringstream result;
  PrintVecConstructor(target, result);
  result << "(";
  for (int index = 0; index < target.lanes(); ++index) {
    if (index)
      result << ", ";
    std::ostringstream element;
    PrintVecElemLoad(source, from, index, element);
    result << CodeGenC::CastFromTo(element.str(), from.WithLanes(1),
                                   target.WithLanes(1));
  }
  result << ")";
  return result.str();
}

void CodeGenCHost::Dispatch_(const prim::BroadcastNode *op,
                             std::ostream &os) { // NOLINT(*)
  std::string value = SSAGetID(PrintExpr(op->value), op->value.ty());
  int lanes = op->ty.as_or_throw<PrimType>().lanes();
  PrintVecConstructor(op->ty.as_or_throw<PrimType>(), os);
  os << "(";
  for (int index = 0; index < lanes; ++index) {
    if (index != 0)
      os << ", ";
    os << value;
  }
  os << ")";
}

void CodeGenCHost::PrintVecConstructor(const PrimType &type,
                                       std::ostream &output) {
  std::ostringstream name;
  PrintType(type, name);
  output << "tvm_make_" << name.str();
}

void CodeGenCHost::PrintVecBinaryOp(const std::string &operation,
                                    const PrimType &type, PrimExpr lhs,
                                    PrimExpr rhs, std::ostream &output) {
  if (!type.MatchesCode(kDLBool)) {
    return CodeGenC::PrintVecBinaryOp(operation, type, lhs, rhs, output);
  }
  std::string left = SSAGetID(PrintExpr(lhs), lhs.ty());
  std::string right = SSAGetID(PrintExpr(rhs), rhs.ty());
  PrintVecConstructor(type, output);
  output << "(";
  for (int index = 0; index < type.lanes(); ++index) {
    if (index)
      output << ", ";
    output << "(" << left << "[" << index << "] " << operation << " " << right
           << "[" << index << "])";
  }
  output << ")";
}

void CodeGenCHost::Dispatch_(const prim::NotNode *node, std::ostream &output) {
  PrimType type = node->ty.as_or_throw<PrimType>();
  if (type.IsScalar())
    return CodeGenC::Dispatch_(node, output);
  std::string value = SSAGetID(PrintExpr(node->a), node->a.ty());
  PrintVecConstructor(type, output);
  output << "(";
  for (int index = 0; index < type.lanes(); ++index) {
    if (index)
      output << ", ";
    output << "(!" << value << "[" << index << "])";
  }
  output << ")";
}

void CodeGenCHost::Dispatch_(const prim::SelectNode *node,
                             std::ostream &output) {
  PrimType type = node->ty.as_or_throw<PrimType>();
  if (type.IsScalar())
    return CodeGenC::Dispatch_(node, output);
  std::string condition =
      SSAGetID(PrintExpr(node->condition), node->condition.ty());
  std::string true_value =
      SSAGetID(PrintExpr(node->true_value), node->true_value.ty());
  std::string false_value =
      SSAGetID(PrintExpr(node->false_value), node->false_value.ty());
  PrintVecConstructor(type, output);
  output << "(";
  for (int index = 0; index < type.lanes(); ++index) {
    if (index)
      output << ", ";
    output << "(" << condition;
    if (!node->condition.ty().IsScalar())
      output << "[" << index << "]";
    output << " ? " << true_value << "[" << index << "] : " << false_value
           << "[" << index << "])";
  }
  output << ")";
}

void CodeGenCHost::PrintVecElemLoadExpr(const PrimType &type, int index,
                                        const std::string &value,
                                        std::ostream &output) {
  if (index == 0) {
    PrintVecConstructor(type, output);
    output << "(";
  }
  output << value;
  output << (index + 1 == type.lanes() ? ")" : ", ");
}

void CodeGenCHost::PrintVecElemLoad(const std::string &value,
                                    const PrimType &type, int index,
                                    std::ostream &output) {
  output << "(" << value << ")[" << index << "]";
}

void CodeGenCHost::PrintVecElemStore(const std::string &value,
                                     const PrimType &type, int index,
                                     const std::string &element) {
  PrintIndent();
  stream << "(" << value << ")[" << index << "] = " << element << ";\n";
}

void CodeGenCHost::Dispatch_(const prim::RampNode *node, std::ostream &output) {
  std::string base = SSAGetID(PrintExpr(node->base), node->base.ty());
  std::string stride = SSAGetID(PrintExpr(node->stride), node->stride.ty());
  PrimType type = node->ty.as_or_throw<PrimType>();
  PrintVecConstructor(type, output);
  output << "(";
  for (int index = 0; index < type.lanes(); ++index) {
    if (index)
      output << ", ";
    output << "(" << base << ") + (" << stride << ") * " << index;
  }
  output << ")";
}

void CodeGenCHost::PrintGetFuncFromBackend(
    const std::string &func_name, const std::string &packed_func_name) {
  this->PrintIndent();
  this->stream << "if (" << packed_func_name << " == NULL) {\n";
  int packed_func_if_scope = this->BeginScope();
  std::string global_name = name_supply_->FreshName("packed_global_name");
  this->PrintIndent();
  this->stream << "TVMFFIByteArray " << global_name << " = {\"" << func_name
               << "\", " << func_name.size() << "};\n";
  this->PrintIndent();
  this->stream << "if ((" << module_name_
               << " != NULL ? TVMFFIEnvModLookupFromImports(" << module_name_
               << ", \"" << func_name << "\", &" << packed_func_name
               << ") : TVMFFIFunctionGetGlobal(&" << global_name << ", &"
               << packed_func_name << ")) != 0) {\n";
  int get_func_env_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "return -1;\n";
  this->EndScope(get_func_env_scope);
  this->PrintIndent();
  this->stream << "}\n";
  this->PrintIndent();
  this->stream << "if (" << packed_func_name << " == NULL) {\n";
  int missing_func_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "TVMFFIErrorSetRaisedFromCStr(\"RuntimeError\", \"Global "
                  "function not found: "
               << func_name << "\");\n";
  this->PrintIndent();
  this->stream << "return -1;\n";
  this->EndScope(missing_func_scope);
  this->PrintIndent();
  this->stream << "}\n";
  this->EndScope(packed_func_if_scope);
  this->PrintIndent();
  this->stream << "}\n";
}

std::string CodeGenCHost::PrintCallPacked(const CallNode *op) {
  const StringImmNode *func_name = op->args[0].as<StringImmNode>();
  TVM_FFI_ICHECK(func_name != nullptr)
      << "call_[c]packed_lowered expects first argument as function name";
  int64_t begin = static_cast<int64_t>(op->args[2].as<IntImmNode>()->value);
  int64_t end = static_cast<int64_t>(op->args[3].as<IntImmNode>()->value);
  int64_t num_args = end - begin;
  TVM_FFI_ICHECK_GE(num_args, 0);

  std::string packed_func_name;
  if (op->op.same_as(tirx::call_packed_lowered_op())) {
    packed_func_name = GetPackedName(op);
    this->PrintGetFuncFromBackend(func_name->value, packed_func_name);
  } else {
    // directly use the original symbol
    TVM_FFI_ICHECK(op->op.same_as(tirx::call_cpacked_lowered_op()));
    packed_func_name = ffi::symbol::tvm_ffi_symbol_prefix + func_name->value;
  }

  std::string args_stack = PrintExpr(op->args[1]);
  this->PrintIndent();
  std::string result = name_supply_->FreshName("result");
  this->stream << "TVMFFIAny " << result << ";\n";
  this->PrintIndent();
  // must make sure type_index is set to none
  this->stream << result << ".type_index = kTVMFFINone;\n";
  this->PrintIndent();
  this->stream << result << ".zero_padding = 0;\n";
  this->PrintIndent();
  this->stream << result << ".v_int64 = 0;\n";
  this->PrintIndent();
  if (op->op.same_as(tirx::call_packed_lowered_op())) {
    this->stream << "if (TVMFFIFunctionCall(" << packed_func_name << ", ";
  } else {
    this->stream << "if (" << packed_func_name << "(NULL, ";
  }
  this->stream << "((TVMFFIAny*) " << args_stack << ") + " << begin << ", "
               << num_args << ", "
               << "&" << result << ") != 0) {\n";
  int func_call_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "return -1;\n";
  this->EndScope(func_call_scope);
  this->PrintIndent();
  this->stream << "}\n";
  return result;
}

std::string CodeGenCHost::GetPackedName(const CallNode *op) {
  const StringImmNode *s = op->args[0].as<StringImmNode>();
  TVM_FFI_ICHECK(s != nullptr)
      << "call_packed_lowered expects first argument as function name";
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

void CodeGenCHost::Dispatch_(const CallNode *op,
                             std::ostream &os) { // NOLINT(*)
  if (op->op.same_as(prim::isnan_op()) &&
      !op->ty.as_or_throw<PrimType>().IsScalar()) {
    PrimExpr value = op->args[0].as_or_throw<PrimExpr>();
    return PrintVecBinaryOp("!=", op->ty.as_or_throw<PrimType>(), value, value,
                            os);
  }
  if (op->op.same_as(tirx::stack_alloca_op())) {
    std::string stack_name = name_supply_->FreshName("stack");
    const std::string &type = op->args[0].as<StringImmNode>()->value;
    const IntImmNode *num = op->args[1].as<IntImmNode>();
    TVM_FFI_ICHECK(num != nullptr);
    static_assert(alignof(TVMFFIAny) % alignof(DLTensor) == 0, "invariant");
    size_t count = num->value.as<size_t>().value();
    if (type == "tensormap") {
      TVM_FFI_CHECK_GT(count, 0U, ValueError)
          << "Tensor-map allocation count must be positive";
      PrintIndent();
      stream << "TVMTensorMap " << stack_name << "[" << count << "];\n";
      os << stack_name;
      return;
    }
    size_t unit = sizeof(TVMFFIAny);
    size_t size = 0;
    if (type == "shape") {
      size = (count * sizeof(ffi::Shape::index_type) + unit - 1) / unit;
    } else if (type == "tvm_ffi_any") {
      size = (count * sizeof(TVMFFIAny) + unit - 1) / unit;
    } else if (type == "array") {
      size = (count * sizeof(DLTensor) + unit - 1) / unit;
    } else {
      TVM_FFI_THROW(InternalError) << "Unknown stack alloca type " << type;
    }
    this->PrintIndent();
    this->stream << "TVMFFIAny " << stack_name << "[" << size << "];\n";
    os << "((";
    PrintType(op->ty, os);
    os << ")" << stack_name << ")";
  } else if (op->op.same_as(tirx::call_packed_lowered_op()) ||
             op->op.same_as(tirx::call_cpacked_lowered_op())) {
    std::string result = PrintCallPacked(op);
    if (op->ty.as<PtrTypeNode>()) {
      os << "((";
      PrintType(op->ty, os);
      os << ")" << result << ".v_ptr)";
    } else {
      PrimType type = op->ty.as_or_throw<PrimType>();
      if (type.IsVoid()) {
        os << "0";
      } else {
        TVM_FFI_CHECK(type.IsScalar(), TypeError)
            << "Packed calls return scalar values";
        os << "((";
        PrintType(type, os);
        os << ")" << result
           << (type.MatchesCode(DLDataTypeCode::kDLFloat) ? ".v_float64)"
                                                          : ".v_int64)");
      }
    }
  } else if (op->op.same_as(tirx::throw_last_error_op())) {
    this->PrintIndent();
    this->stream << "return -1;\n";
  } else {
    CodeGenC::Dispatch_(op, os);
  }
}

void CodeGenCHost::Dispatch_(const AssertStmtNode *op) { // NOLINT(*)
  if (emit_asserts_) {
    std::string cond = PrintExpr(op->condition);
    PrintIndent();
    stream << "if (!(" << cond << ")) {\n";
    int assert_if_scope = this->BeginScope();
    int num_parts = static_cast<int>(op->message_parts.size());
    PrintIndent();
    stream << "const char* __tvm_assert_parts[" << num_parts << "] = {";
    for (int i = 0; i < num_parts; ++i) {
      if (i > 0)
        stream << ", ";
      PrintEscapedCString(op->message_parts[i]->value, stream);
    }
    stream << "};\n";
    PrintIndent();
    stream << "TVMFFIErrorSetRaisedFromCStrParts(";
    PrintEscapedCString(op->error_kind->value, stream);
    stream << ", __tvm_assert_parts, " << num_parts << ");\n";
    PrintIndent();
    stream << "return -1;\n";
    this->EndScope(assert_if_scope);
    PrintIndent();
    stream << "}\n";
  }
}

void CodeGenCHost::Dispatch_(const prim::MinNode *op,
                             std::ostream &os) { // NOLINT(*)
  PrintTernaryCondExpr(op, "<", os);
}

void CodeGenCHost::Dispatch_(const prim::MaxNode *op,
                             std::ostream &os) { // NOLINT(*)
  PrintTernaryCondExpr(op, ">", os);
}

template <typename T>
inline void CodeGenCHost::PrintTernaryCondExpr(const T *op, const char *compare,
                                               std::ostream &os) { // NOLINT(*)
  std::ostringstream temp_a;
  Dispatch(op->a, temp_a);
  std::string a_id = SSAGetID(temp_a.str(), op->a.ty());
  std::ostringstream temp_b;
  Dispatch(op->b, temp_b);
  std::string b_id = SSAGetID(temp_b.str(), op->b.ty());

  PrimType type = op->ty.template as_or_throw<PrimType>();
  if (!type.IsScalar()) {
    PrintVecConstructor(type, os);
    os << "(";
    for (int index = 0; index < type.lanes(); ++index) {
      if (index)
        os << ", ";
      os << "(" << a_id << "[" << index << "] " << compare << " " << b_id << "["
         << index << "] ? " << a_id << "[" << index << "] : " << b_id << "["
         << index << "])";
    }
    os << ")";
    return;
  }

  os << "((" << a_id << ") " << compare << " (" << b_id << ") "
     << "? (" << a_id << ") : (" << b_id << "))";
}

ffi::Module BuildCHost(IRModule mod, Target target) {
  bool output_ssa = false;
  bool emit_asserts = true;
  bool emit_fwd_func_decl = true;

  CodeGenCHost cg;
  cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str());
  cg.ConfigureModule(mod);
  cg.SetConstantsByteAlignment(
      target->GetAttr<int64_t>("constants-byte-alignment").value_or(16));

  std::vector<std::pair<GlobalVar, Function>> funcs;
  for (auto [gvar, base_func] : mod->functions) {
    TVM_FFI_ICHECK(base_func->IsInstance<FunctionNode>())
        << "CodegenCHost: Can only take Function";
    auto function = base_func.as_or_throw<Function>();
    funcs.push_back({gvar, function});
  }

  // Sort functions by name for deterministic output.
  std::sort(funcs.begin(), funcs.end(), [](const auto &kv_a, const auto &kv_b) {
    return kv_a.first->name_hint < kv_b.first->name_hint;
  });

  for (const auto &[gvar, function] : funcs) {
    cg.DeclareFunction(gvar, function);
  }

  // Codegen all functions.  Passing emit_fwd_func_decl=true adds a
  // forward declaration for any `tirx::call_extern_op`, based on the
  // arguments provided to it.
  for (const auto &[gvar, function] : funcs) {
    cg.AddFunction(gvar, function, emit_fwd_func_decl);
  }

  std::string code = cg.Finish();
  if (auto postproc = mod->GetAttr(attr::kPostproc)) {
    code = ffi::Function::GetGlobalRequired(*postproc)(code, target)
               .cast<std::string>();
  }
  return ::tvm::codegen::CSourceModuleCreate(code, "c", cg.GetFunctionNames());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.codegen.build_c", BuildCHost);
}
} // namespace tvm::tl::codegen_impl
