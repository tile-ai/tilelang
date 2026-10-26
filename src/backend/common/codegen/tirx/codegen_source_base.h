
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
 * \file codegen_source_base.h
 * \brief Common utilities to source code in text form.
 */
#ifndef TL_CODEGEN_IMPL_C_CODEGEN_SOURCE_BASE_H_
#define TL_CODEGEN_IMPL_C_CODEGEN_SOURCE_BASE_H_

#include "backend/common/codegen/attr.h"
#include <tvm/ir/expr.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/unique_name_supply.h>

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "tirx_compat/metadata.h"

namespace tvm::tl::codegen_impl {
namespace attr = tvm::tl::codegen_attr;

/*!
 * \brief A base class to generate source code.
 * Contains helper utilities to generate nest and ssa form.
 */
class CodeGenSourceBase {
public:
  virtual ~CodeGenSourceBase() = default;
  /*!
   * \brief Register constant value appeared in expresion tree
   *  This avoid generated a ssa id for each appearance of the value
   * \param value The constant value.
   */
  void MarkConst(std::string value);
  /*!
   * Print Type representation of type type.
   * \param t The type representation.
   * \param os The stream to print the ctype into
   */
  virtual void PrintType(const PrimType &type, std::ostream &os); // NOLINT(*)
  /*!
   * Print Type representation of type type.
   * \param type The type representation.
   * \param os The stream to print the ctype into
   */
  virtual void PrintType(const Type &type, std::ostream &os); // NOLINT(*)

  virtual void PrintStorageType(const Type &type, std::ostream &os) {
    PrintType(type, os);
  }

protected:
  /*! \brief entry in ssa assign map */
  struct SSAEntry {
    /*! \brief The value id */
    std::string vid;
    /*! \brief The scope id, used to check if this entry is invalid. */
    int scope_id;
  };
  /*! \brief Clear the states that might relates to function generation */
  void ClearFuncState();
  /*! \brief print the current indented value */
  void PrintIndent();
  /*!
   * \brief Allocate a variable name for a newly defined var.
   * \param v The variable.
   * \return the variable name.
   */
  std::string AllocVarID(const tvm::VarNode *v);
  /*!
   * \brief Allocate a variable name using an explicit diagnostic hint.
   * \param v The variable.
   * \param name_hint The name hint.
   * \return the variable name.
   */
  std::string AllocVarID(const tvm::VarNode *v, std::string name_hint);
  /*!
   * \brief Get a variable name.
   * \param v The variable.
   * \return the variable name.
   */
  std::string GetVarID(const tvm::VarNode *v) const;
  /*!
   * \brief Get the SSA ID corresponds to src
   *  If necessary, generate new assignment
   * \param src The source expression
   * \param t The type of the expression.
   */
  std::string SSAGetID(std::string src, const Type &t);
  /*!
   * \brief mark the beginning of a new scope
   * \return The scope id.
   */
  int BeginScope();
  /*!
   * \brief mark the end of an old scope.
   * \param scope_id The scope id to be ended.
   */
  void EndScope(int scope_id);
  /*!
   * \brief Print assignment of src to the id in ssa entry.
   * \param target id of target variable.
   * \param src The source expression.
   * \param t The type of target.
   */
  virtual void PrintSSAAssign(const std::string &target, const std::string &src,
                              const Type &t) = 0;

  /*! \brief the declaration stream */
  std::ostringstream decl_stream;
  /*! \brief the stream to be printed */
  std::ostringstream stream;
  /*! \brief the forward declaration stream */
  std::ostringstream fwd_decl_stream;
  /*! \brief name of each variable */
  std::unordered_map<const tvm::VarNode *, std::string> var_idmap_;
  /*! \brief Unique name supply for allocation */
  UniqueNameSupply name_supply_;
  /*! \brief The current indentation value */
  int indent_{0};

private:
  /*! \brief assignment map of ssa */
  std::unordered_map<std::string, SSAEntry> ssa_assign_map_;
  /*! \brief array to check whether we are inside certain scope */
  std::vector<bool> scope_mark_;
};

} // namespace tvm::tl::codegen_impl
#endif
