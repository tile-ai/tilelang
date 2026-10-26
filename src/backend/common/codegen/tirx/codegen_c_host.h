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
 * \file codegen_c_host.h
 * \brief Generate C host code.
 */
#ifndef TL_CODEGEN_IMPL_C_CODEGEN_C_HOST_H_
#define TL_CODEGEN_IMPL_C_CODEGEN_C_HOST_H_

#include <tvm/ir/prim/expr.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backend/common/codegen/attr.h"
#include "codegen_c.h"

namespace tvm::tl::codegen_impl {
namespace attr = tvm::tl::codegen_attr;

class CodeGenCHost : public CodeGenC {
public:
  CodeGenCHost();
  void Init(bool output_ssa, bool emit_asserts, bool emit_fwd_func_decl,
            std::string target_str);

  void InitGlobalContext();

  void AddFunction(const GlobalVar &gvar, const Function &f) override;
  void AddFunction(const GlobalVar &gvar, const Function &f,
                   bool emit_fwd_func_decl);
  /*!
   * \brief Add functions from the (unordered) range to the current module in a
   * deterministic order. This helps with debugging.
   *
   * \param functions A vector of unordered range of current module.
   */
  void AddFunctionsOrdered(
      std::vector<std::pair<tvm::GlobalVar, tvm::BaseFunc>> functions);
  void DefineModuleName();

  using CodeGenC::PrintType;
  void PrintType(const PrimType &t, std::ostream &os) override; // NOLINT(*)
  void PrintStorageType(const Type &type, std::ostream &os) override;
  std::string CastFromTo(std::string value, const PrimType &from,
                         const PrimType &target) override;
  void PrintVecConstructor(const PrimType &type, std::ostream &output) override;
  void PrintVecBinaryOp(const std::string &operation, const PrimType &type,
                        PrimExpr lhs, PrimExpr rhs,
                        std::ostream &output) override;
  void PrintVecElemLoad(const std::string &value, const PrimType &type,
                        int index, std::ostream &output) override;
  void PrintVecElemStore(const std::string &value, const PrimType &type,
                         int index, const std::string &element) override;
  void PrintVecElemLoadExpr(const PrimType &type, int index,
                            const std::string &value,
                            std::ostream &output) override;
  void Dispatch_(const prim::RampNode *node, std::ostream &output) override;
  void Dispatch_(const prim::NotNode *node, std::ostream &output) override;
  void Dispatch_(const prim::SelectNode *node, std::ostream &output) override;
  void PrintFuncPrefix(std::ostream &os) final; // NOLINT(*)

  // overload visitor functions
  void Dispatch_(const prim::BroadcastNode *op,
                 std::ostream &os) final;                        // NOLINT(*)
  void Dispatch_(const CallNode *op, std::ostream &os) override; // NOLINT(*)
  // overload min and max to use the ternary operator, so we don't rely on the
  // standard library implementations
  void Dispatch_(const prim::MinNode *op, std::ostream &os) final; // NOLINT(*)
  void Dispatch_(const prim::MaxNode *op, std::ostream &os) final; // NOLINT(*)

  void Dispatch_(const AssertStmtNode *op) final; // NOLINT(*)
  void Dispatch_(const ForNode *op) final;

  void GenerateForwardFunctionDeclarations(ffi::String global_symbol,
                                           const ffi::Array<Type> &arg_types,
                                           const Type &ret_type) override;
  void PrintCallExtern(Type ret_type, ffi::String global_symbol,
                       const ffi::Array<Expr> &args, bool skip_first_arg,
                       std::ostream &os) override;
  ffi::Array<ffi::String> GetFunctionNames() { return function_names_; }

protected:
  // Plain C has no address-space qualifiers.  Buffer storage scopes still
  // control allocation lowering, but are not part of emitted pointer types.
  bool IsScopePartOfType() const final { return false; }

private:
  std::string module_name_;
  /* \brief mapping global packed func to the unique name */
  std::unordered_map<std::string, std::string> declared_globals_;
  /* \brief names of the functions declared in this module */
  ffi::Array<ffi::String> function_names_;
  /*! \brief whether to emit asserts in the resulting C code */
  bool emit_asserts_;
  /*! \brief whether to emit forwared function declarations in the resulting C
   * code */
  bool emit_fwd_func_decl_;
  /*! \brief whether to generate the entry function if encountered */
  bool has_main_func_ = false;
  std::unordered_set<const ForNode *> parallel_chain_;
  std::unordered_set<std::string> vector_types_;

  std::string GetPackedName(const CallNode *op);
  void PrintGetFuncFromBackend(const std::string &func_name,
                               const std::string &packed_func_name);
  std::string PrintCallPacked(const CallNode *op);
  /*!
   * \brief Print ternary conditional operator implementing binary `op`
   * Forces the operands to be in SSA form.
   * \param op binary operator being expressed
   * \param compare string representation of comparison operator
   * \param os stream reference to print into
   */
  template <typename T>
  inline void PrintTernaryCondExpr(const T *op, const char *compare,
                                   std::ostream &os); // NOLINT(*)
};

} // namespace tvm::tl::codegen_impl

#endif // TL_CODEGEN_IMPL_C_CODEGEN_C_HOST_H_
