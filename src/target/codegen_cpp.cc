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
#include "codegen_cpp.h"

#include <tvm/runtime/module.h>
#include <tvm/target/codegen.h>

#include <string>
#include <unordered_set>
#include <utility>

#include "../op/builtin.h"
#include "../support/ffi_aliases.h"
#include "support/str_escape.h"
#include "target/build_common.h"
#include "target/source/codegen_params.h"

namespace tvm {
namespace codegen {

CodeGenTileLangCPP::CodeGenTileLangCPP() {
  module_name_ = name_supply_->FreshName("__tvm_ffi_library_ctx");
}

void CodeGenTileLangCPP::Init(bool output_ssa, bool emit_asserts,
                              bool emit_fwd_func_decl, std::string target_str,
                              const std::unordered_set<std::string> &devices) {
  emit_asserts_ = emit_asserts;
  emit_fwd_func_decl_ = emit_fwd_func_decl;
  declared_globals_.clear();
  decl_stream << "// tilelang target: " << target_str << "\n";
  decl_stream << "#include <tl_templates/cpp/common.h>\n";
  decl_stream << "#include <tl_templates/cpp/gemm.h>\n";
  decl_stream << "\n";
  CodeGenC::Init(output_ssa);
}

void CodeGenTileLangCPP::InitGlobalContext() {
  decl_stream << "void* " << ffi::symbol::tvm_ffi_library_ctx << " = NULL;\n";
}

void CodeGenTileLangCPP::DefineModuleName() {
  decl_stream << "void* " << module_name_ << " = NULL;\n";
}

void CodeGenTileLangCPP::GenerateForwardFunctionDeclarations(
    String global_symbol,

    const Array<Type> &arg_types, const Type &ret_type) {
  if (!emit_fwd_func_decl_) {
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

void CodeGenTileLangCPP::PrintFuncPrefix(std::ostream &os) { // NOLINT(*)
  os << "#ifdef __cplusplus\n"
     << "extern \"C\"\n"
     << "#endif\n";
}

void CodeGenTileLangCPP::PrintType(DataType t, std::ostream &os) { // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "does not support vector types";
    os << "void*";
    return;
  }
  if (t.is_void()) {
    os << "void";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }
  bool fail = false;
  if (t.is_float()) {
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
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
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
  LOG(FATAL) << "Cannot convert type " << t << " to C type";
}

void CodeGenTileLangCPP::VisitExpr_(const BroadcastNode *op,
                                    std::ostream &os) { // NOLINT(*)
  std::string v = PrintExpr(op->value);
  int lanes = op->dtype.lanes();
  os << "((";
  PrintType(op->dtype, os);
  os << ")(";
  for (int i = 0; i < lanes; ++i) {
    if (i != 0)
      os << ", ";
    os << v;
  }
  os << "))";
}

void CodeGenTileLangCPP::PrintGetFuncFromBackend(
    const std::string &func_name, const std::string &packed_func_name) {
  this->PrintIndent();
  this->stream << "if (" << packed_func_name << " == NULL) {\n";
  int packed_func_if_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "if (TVMBackendGetFuncFromEnv(" << module_name_ << ", \""
               << func_name << "\""
               << ", &" << packed_func_name << ") != 0) {\n";
  int get_func_env_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "return -1;\n";
  this->EndScope(get_func_env_scope);
  this->PrintIndent();
  this->stream << "}\n";
  this->EndScope(packed_func_if_scope);
  this->PrintIndent();
  this->stream << "}\n";
}

void CodeGenTileLangCPP::PrintFuncCall(const std::string &packed_func_name,
                                       int num_args) {
  this->PrintIndent();
  std::string ret_val = name_supply_->FreshName("ret_val");
  std::string ret_type_code = name_supply_->FreshName("ret_type_code");
  this->stream << "TVMFFIAny " << ret_val << ";\n";
  this->PrintIndent();
  this->stream << "int " << ret_type_code << ";\n";
  this->PrintIndent();
  this->stream << "if (TVMFuncCall(" << packed_func_name << ", "
               << "(TVMFFIAny*) stack_value"
               << ", "
               << "(int*) stack_tcode"
               << ", " << num_args << ", "
               << "&" << ret_val << ", "
               << "&" << ret_type_code << ") != 0) {\n";
  int func_call_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "return -1;\n";
  this->EndScope(func_call_scope);
  this->PrintIndent();
  this->stream << "}\n";
}

void CodeGenTileLangCPP::PrintFuncCallC(
    const std::string &packed_func_name, int num_args,
    const std::string &resource_handle_name) {
  this->PrintIndent();
  std::string ret_val = name_supply_->FreshName("ret_val");
  std::string ret_type_code = name_supply_->FreshName("ret_type_code");
  this->stream << "TVMFFIAny " << ret_val << ";\n";
  this->PrintIndent();
  this->stream << "int " << ret_type_code << ";\n";
  this->PrintIndent();

  this->stream << "if (" << packed_func_name << "( "
               << "(TVMFFIAny*) stack_value "
               << ", "
               << "(int*) stack_tcode"
               << ", " << num_args << ", "
               << "&" << ret_val << ", "
               << "&" << ret_type_code << ", " << resource_handle_name
               << ") != 0){\n";

  int func_call_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "return -1;\n";
  this->EndScope(func_call_scope);
  this->PrintIndent();
  this->stream << "}\n";
}

void CodeGenTileLangCPP::AddFunction(const PrimFunc &f) {
  // clear previous generated state.
  this->InitFuncState(f);
  // reserve keywords
  ReserveKeywordsAsUnique();

  auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol)
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);
  std::unordered_set<const VarNode *> non_restrict;
  if (auto opt =
          f->GetAttr<ffi::Array<tir::Var>>(tl::attr::kNonRestrictParams)) {
    for (const tir::Var &v : opt.value())
      non_restrict.insert(v.get());
  }

  this->PrintFuncPrefix(stream);
  CodeGenC::PrintType(f->ret_type, stream);
  this->PrintExtraAttrs(f, stream);
  this->stream << " " << static_cast<std::string>(global_symbol.value()) << "(";

  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0)
      stream << ", ";
    if (v.dtype().is_handle()) {
      // work around for grid constant parameters.
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (ptr->storage_scope == "grid_constant") {
          stream << "__grid_constant__ const ";
          CodeGenC::PrintType(ptr->element_type, stream);
          stream << ' ' << vid;
          continue;
        }
      }

      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, stream);
      }

      CodeGenC::PrintType(GetType(v), stream);
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }

      if (no_alias && !non_restrict.count(v.get())) {
        PrintRestrict(v, stream);
      }
    } else {
      CodeGenC::PrintType(GetType(v), stream);
    }
    stream << ' ' << vid;
  }
  stream << ") {\n";
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

std::string CodeGenTileLangCPP::GetPackedName(const CallNode *op) {
  const StringImmNode *s = op->args[0].as<StringImmNode>();
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

CodeGenTileLangCPP::FunctionInfo
CodeGenTileLangCPP::GetFunctionInfo(const CallNode *op,
                                    bool has_resource_handle) {
  const StringImmNode *s = op->args[0].as<StringImmNode>();
  ICHECK(s != nullptr)
      << "tvm_call_[c]packed_lowered expects first argument as function name";
  int64_t begin = op->args[3].as<IntImmNode>()->value;
  int64_t end = op->args[4].as<IntImmNode>()->value;
  int64_t num_args = end - begin;
  ICHECK_GE(num_args, 0);
  std::string func_name = s->value;

  if (has_resource_handle) {
    const StringImmNode *resource_handle_var = op->args[5].as<StringImmNode>();
    if (resource_handle_var != nullptr) {
      std::string resource_handle_name = resource_handle_var->value;
      return {func_name, num_args - 1, resource_handle_name};
    } else {
      // The final arg should be "(void*) NULL" to indicate the empty
      // resource_handle.
      num_args--;

      const CallNode *reinterpret_call = op->args[5].as<CallNode>();
      ICHECK_NE(reinterpret_call, (void *)nullptr)
          << "At CallNode to " << s
          << "arg 5: Expect either StringImm naming the resource_handle var "
             "from interface API or "
          << "reinterpret(0); got: " << op->args[5];
      ICHECK_EQ(reinterpret_call->op, builtin::reinterpret())
          << "At CallNode to " << s
          << "arg 5: Expect either StringImm naming the resource_handle var "
             "from interface API or "
          << "reinterpret(0); got: " << op->args[5];
      ICHECK(is_zero(reinterpret_call->args[0]))
          << "At CallNode to " << s
          << " arg 5: Expect either StringImm naming the "
             "resource_handle var from interface API, or "
          << "zero; got " << op->args[5];
    }
  }
  return {func_name, num_args, "NULL"};
}

void CodeGenTileLangCPP::VisitExpr_(const CallNode *op,
                                    std::ostream &os) { // NOLINT(*)
  if (op->op.same_as(builtin::tvm_stack_alloca())) {
    std::string stack_name = name_supply_->FreshName("stack");
    const std::string &type = op->args[0].as<StringImmNode>()->value;
    const IntImmNode *num = op->args[1].as<IntImmNode>();
    ICHECK(num != nullptr);
    static_assert(alignof(TVMFFIAny) % alignof(DLTensor) == 0, "invariant");
    size_t unit = sizeof(TVMFFIAny);
    size_t size = 0;
    if (type == "shape") {
      size = (num->value * sizeof(runtime::tvm_index_t) + unit - 1) / unit;
    } else if (type == "arg_value") {
      size = (num->value * sizeof(TVMFFIAny) + unit - 1) / unit;
    } else if (type == "arg_tcode") {
      size = (num->value * sizeof(int) + unit - 1) / unit;
    } else if (type == "array") {
      size = (num->value * sizeof(DLTensor) + unit - 1) / unit;
    } else {
      LOG(FATAL) << "Unknown stack alloca type " << type;
    }
    this->PrintIndent();
    this->stream << "TVMFFIAny " << stack_name << "[" << size << "];\n";
    os << stack_name;
  } else if (op->op.same_as(builtin::tvm_call_packed_lowered())) {
    auto function_info = GetFunctionInfo(op, false /* has_resource_handle */);
    std::string func_name_packed = GetPackedName(op);
    this->PrintGetFuncFromBackend(function_info.func_name, func_name_packed);
    this->PrintFuncCall(func_name_packed, function_info.num_args);
  } else if (op->op.same_as(builtin::tvm_call_cpacked_lowered())) {
    auto function_info = GetFunctionInfo(op, true /* has_resource_handle */);
    this->PrintFuncCallC(function_info.func_name, function_info.num_args,
                         function_info.resource_handle_name);
  } else if (op->op.same_as(builtin::tvm_throw_last_error())) {
    this->PrintIndent();
    this->stream << "return -1;\n";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenTileLangCPP::VisitStmt_(const AssertStmtNode *op) { // NOLINT(*)
  if (emit_asserts_) {
    std::string cond = PrintExpr(op->condition);
    PrintIndent();
    stream << "if (!(" << cond << ")) {\n";
    int assert_if_scope = this->BeginScope();
    PrintIndent();
    stream << "TVMAPISetLastError(\"" << op->message.as<StringImmNode>()->value
           << "\");\n";
    PrintIndent();
    stream << "return -1;\n";
    this->EndScope(assert_if_scope);
    PrintIndent();
    stream << "}\n";
  }
  this->PrintStmt(op->body);
}

void CodeGenTileLangCPP::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  this->PrintIndent();
  std::string scope = GetPtrStorageScope(op->buffer_var);

  PrintType(op->dtype, stream);
  size_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0)
      << "Can only handle constant size stack allocation for now";

  stream << ' ' << vid << '[' << constant_size << "];\n";

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

void CodeGenTileLangCPP::VisitExpr_(const MinNode *op,
                                    std::ostream &os) { // NOLINT(*)
  PrintTernaryCondExpr(op, "<", os);
}

void CodeGenTileLangCPP::VisitExpr_(const MaxNode *op,
                                    std::ostream &os) { // NOLINT(*)
  PrintTernaryCondExpr(op, ">", os);
}

template <typename T>
inline void
CodeGenTileLangCPP::PrintTernaryCondExpr(const T *op, const char *compare,
                                         std::ostream &os) { // NOLINT(*)
  std::ostringstream temp_a;
  VisitExpr(op->a, temp_a);
  std::string a_id = SSAGetID(temp_a.str(), op->a.dtype());
  std::ostringstream temp_b;
  VisitExpr(op->b, temp_b);
  std::string b_id = SSAGetID(temp_b.str(), op->b.dtype());

  os << "((" << a_id << ") " << compare << " (" << b_id << ") "
     << "? (" << a_id << ") : (" << b_id << "))";
}

} // namespace codegen
} // namespace tvm
