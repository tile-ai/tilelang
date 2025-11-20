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
 * \brief Generate C host code (TileLang copy).
 */
#ifndef TL_TARGET_SOURCE_CODEGEN_C_HOST_H_
#define TL_TARGET_SOURCE_CODEGEN_C_HOST_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "target/source/codegen_c.h"
#include "tvm/target/codegen.h"
#include "tvm/tir/expr.h"

namespace tvm {
namespace tl {

// TileLang copy of TVM's CodeGenCHost, under the tl namespace.
// Inherits from tvm::codegen::CodeGenC.
class CodeGenCHost : public tvm::codegen::CodeGenC {
public:
  CodeGenCHost();
  void Init(bool output_ssa, bool emit_asserts, bool emit_fwd_func_decl,
            std::string target_str,
            const std::unordered_set<std::string> &devices);

  void InitGlobalContext();

  void AddFunction(const tvm::GlobalVar &gvar,
                   const tvm::tir::PrimFunc &f) override;
  void AddFunction(const tvm::GlobalVar &gvar, const tvm::tir::PrimFunc &f,
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

  using tvm::codegen::CodeGenC::PrintType;
  void PrintType(tvm::DataType t, std::ostream &os) final; // NOLINT(*)
  void PrintFuncPrefix(std::ostream &os) final;            // NOLINT(*)

  // overload visitor functions
  void VisitExpr_(const tvm::tir::BroadcastNode *op,
                  std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const tvm::tir::CallNode *op,
                  std::ostream &os) override; // NOLINT(*)
  // overload min and max to use the ternary operator, so we don't rely on the
  // standard library implementations
  void VisitExpr_(const tvm::tir::MinNode *op,
                  std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const tvm::tir::MaxNode *op,
                  std::ostream &os) final; // NOLINT(*)

  void VisitStmt_(const tvm::tir::AssertStmtNode *op) final; // NOLINT(*)

  void GenerateForwardFunctionDeclarations(
      tvm::ffi::String global_symbol,
      const tvm::ffi::Array<tvm::Type> &arg_types,
      const tvm::Type &ret_type) override;
  tvm::ffi::Array<tvm::ffi::String> GetFunctionNames() {
    return function_names_;
  }

private:
  std::string module_name_;
  /* \brief mapping global packed func to the unique name */
  std::unordered_map<std::string, std::string> declared_globals_;
  /* \brief names of the functions declared in this module */
  tvm::ffi::Array<tvm::ffi::String> function_names_;
  /*! \brief whether to emit asserts in the resulting C code */
  bool emit_asserts_;
  /*! \brief whether to emit forwared function declarations in the resulting C
   * code */
  bool emit_fwd_func_decl_;
  /*! \brief whether to generate the entry function if encountered */
  bool has_main_func_ = false;

  std::string GetPackedName(const tvm::tir::CallNode *op);
  void PrintGetFuncFromBackend(const std::string &func_name,
                               const std::string &packed_func_name);
  void PrintCallPacked(const tvm::tir::CallNode *op);
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

} // namespace tl
} // namespace tvm

#endif // TL_TARGET_SOURCE_CODEGEN_C_HOST_H_
