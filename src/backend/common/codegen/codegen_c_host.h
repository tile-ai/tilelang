/*!
 * \file codegen_c_host.h
 * \brief Generate C host code with TVM FFI when Host CodeGen is enabled.
 */
#ifndef TVM_TL_CODEGEN_C_HOST_H_
#define TVM_TL_CODEGEN_C_HOST_H_

#include "support/check.h"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backend/common/codegen/codegen_c_base.h"
#include <tvm/codegen/codegen.h>
#include <tvm/tirx/expr.h>

namespace tvm {
namespace tl {

using namespace ffi;

// TileLang copy of TVM's CodeGenCHost, under the tl namespace.
class CodeGenCHost : public tvm::codegen::CodeGenTileLangCBase {
public:
  using tvm::codegen::CodeGenTileLangCBase::Dispatch;
  using tvm::codegen::CodeGenTileLangCBase::Dispatch_;

  CodeGenCHost();
  void Init(bool output_ssa, bool emit_asserts, bool emit_fwd_func_decl,
            std::string target_str,
            const std::unordered_set<std::string> &devices);

  void InitGlobalContext();

  void AddFunction(const tvm::GlobalVar &gvar,
                   const tvm::tirx::Function &f) override;
  void AddFunction(const tvm::GlobalVar &gvar, const tvm::tirx::Function &f,
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
  void PrintType(const tvm::Type &t, std::ostream &os) override; // NOLINT(*)
  void PrintType(const tvm::PrimType &t,
                 std::ostream &os) override;       // NOLINT(*)
  void PrintFuncPrefix(std::ostream &os) override; // NOLINT(*)
  void PrintFunctionSignature(const ffi::String &name,
                              const tvm::tirx::Function &function,
                              std::ostream &os) override;

  // overload visitor functions
  void Dispatch_(const tvm::FloatImmObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const tvm::prim::BroadcastObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const tvm::CallObj *op,
                 std::ostream &os) override; // NOLINT(*)
  // overload min and max to use the ternary operator, so we don't rely on the
  // standard library implementations
  void Dispatch_(const tvm::prim::MinObj *op,
                 std::ostream &os) final; // NOLINT(*)
  void Dispatch_(const tvm::prim::MaxObj *op,
                 std::ostream &os) final; // NOLINT(*)

  void Dispatch_(const tvm::EvaluateObj *op) override; // NOLINT(*)
  void Dispatch_(const tvm::BindObj *op) override;     // NOLINT(*)

  void Dispatch_(const tvm::AssertStmtObj *op) final; // NOLINT(*)

  void Dispatch_(const tvm::RegionStmtObj *op) final; // NOLINT(*)

  void GenerateForwardFunctionDeclarations(String global_symbol,
                                           const Array<tvm::Type> &arg_types,
                                           const tvm::Type &ret_type) override;
  Array<String> GetFunctionNames() { return function_names_; }

protected:
  bool IsScopePartOfType() const override { return false; }

private:
  std::string module_name_;
  /* \brief mapping global packed func to the unique name */
  std::unordered_map<std::string, std::string> declared_globals_;
  /* \brief names of the functions declared in this module */
  Array<String> function_names_;
  /*! \brief whether to emit asserts in the resulting C code */
  bool emit_asserts_;
  /*! \brief whether to emit forwarded function declarations in the resulting C
   * code */
  bool emit_fwd_func_decl_;
  /*! \brief whether to generate the entry function if encountered */
  bool has_main_func_ = false;

  bool is_in_metal_context = false;

  std::string GetPackedName(const tvm::CallObj *op);
  void PrintGetFuncFromBackend(const std::string &func_name,
                               const std::string &packed_func_name);
  void PrintCallPacked(const tvm::CallObj *op, std::ostream &os,
                       bool discard_result = false);
  void PrintAnyListInvoke(const tvm::CallObj *op, std::ostream &os);
  void PrintAnyListShape(const tvm::CallObj *op, std::ostream &os);
  void PrintAnyListDescriptor(const tvm::CallObj *op, std::ostream &os);
  void PrintPackedFailureReturn();
  bool anylist_helpers_emitted_ = false;
  std::vector<std::string> anylist_error_cleanups_;
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

  template <typename... Args> void PrintLine(Args &&...args) {
    this->PrintIndent();
    (this->stream << ... << args) << '\n';
  }
};

/*!
 * \brief Build a TileLang C host module for the given IRModule and target.
 * Also handles Metal target through is_in_metal_context flag in IR.
 */
Module BuildTileLangCHost(::tvm::IRModule mod, ::tvm::Target target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_CODEGEN_C_HOST_H_
