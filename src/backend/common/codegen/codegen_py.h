#ifndef TVM_TL_TARGET_CODEGEN_PY_H_
#define TVM_TL_TARGET_CODEGEN_PY_H_

#include "compat/expr.h"
#include "support/check.h"
#include <tvm/codegen/codegen.h>
#include <tvm/codegen/source_base.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/ir/op.h>
#include <tvm/ir/unique_name_supply.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <unordered_map>

namespace tvm::codegen {

using namespace prim;
using ffi::Array;
using ffi::String;
using tirx::TensorVar;

PrimType PythonDType(const Expr &expression);
bool PythonIsPointer(const Expr &expression);
std::string PythonStorageScope(const Var &variable);
bool PythonIsOp(const Expr &expression, const std::string &name);
PrimExpr PythonTensorOffset(const TensorVar &tensor,
                            const Array<PrimExpr> &indices);
PrimExpr PythonTensorSize(const TensorVar &tensor);

class CodeGenTileLangPY
    : public tvm::ExprFunctor<void(const Expr &, std::ostream &)>,
      public tirx::StmtFunctor<void(const Stmt &)>,
      public CodeGenSourceBase {
public:
  virtual void AddFunction(const GlobalVar &global,
                           const tirx::Function &function);
  virtual std::string Finish();

protected:
  String GetFunctionName_(const GlobalVar &global);
  virtual void RegisterFunction_(const GlobalVar &global,
                                 const tirx::Function &function);
  virtual void InitFuncState_(const tirx::Function &function);
  virtual void PrintFunctionSignature_(const String &name,
                                       const tirx::Function &function,
                                       std::ostream &output);
  virtual void PrintFuncDecorator_(std::ostream &output) {}
  virtual void PreFunctionBody_(const tirx::Function &function) {}
  void ReserveKeywordsAsUnique_();
  void PrintSSAAssign(const std::string &target, const std::string &source,
                      const Type &type) override;
  virtual void PrintType(PrimType type, std::ostream &output);
  void PrintType(const Type &type, std::ostream &output) override;
  void PrintStmt_(const Stmt &statement) {
    auto beginning = stream.tellp();
    tirx::StmtFunctor<void(const Stmt &)>::Dispatch(statement);
    if (stream.tellp() == beginning) {
      PrintIndent();
      stream << "pass\n";
    }
  }
  void PrintExpr_(const Expr &expression, std::ostream &output) {
    tvm::ExprFunctor<void(const Expr &, std::ostream &)>::Dispatch(expression,
                                                                   output);
  }
  std::string PrintExpr_(const Expr &expression) {
    std::ostringstream output;
    PrintExpr_(expression, output);
    return output.str();
  }

  void Dispatch_(const VarNode *node, std::ostream &output) override;
  void Dispatch_(const IntImmNode *node, std::ostream &output) override;
  void Dispatch_(const FloatImmNode *node, std::ostream &output) override;
  void Dispatch_(const StringImmNode *node, std::ostream &output) override;
  void Dispatch_(const CastNode *node, std::ostream &output) override;
  void Dispatch_(const AddNode *node, std::ostream &output) override;
  void Dispatch_(const SubNode *node, std::ostream &output) override;
  void Dispatch_(const MulNode *node, std::ostream &output) override;
  void Dispatch_(const DivNode *node, std::ostream &output) override;
  void Dispatch_(const ModNode *node, std::ostream &output) override;
  void Dispatch_(const FloorDivNode *node, std::ostream &output) override;
  void Dispatch_(const FloorModNode *node, std::ostream &output) override;
  void Dispatch_(const MinNode *node, std::ostream &output) override;
  void Dispatch_(const MaxNode *node, std::ostream &output) override;
  void Dispatch_(const EQNode *node, std::ostream &output) override;
  void Dispatch_(const NENode *node, std::ostream &output) override;
  void Dispatch_(const LTNode *node, std::ostream &output) override;
  void Dispatch_(const LENode *node, std::ostream &output) override;
  void Dispatch_(const GTNode *node, std::ostream &output) override;
  void Dispatch_(const GENode *node, std::ostream &output) override;
  void Dispatch_(const AndNode *node, std::ostream &output) override;
  void Dispatch_(const OrNode *node, std::ostream &output) override;
  void Dispatch_(const NotNode *node, std::ostream &output) override;
  void Dispatch_(const LShiftNode *node, std::ostream &output) override;
  void Dispatch_(const RShiftNode *node, std::ostream &output) override;
  void Dispatch_(const BitwiseAndNode *node, std::ostream &output) override;
  void Dispatch_(const BitwiseOrNode *node, std::ostream &output) override;
  void Dispatch_(const BitwiseXorNode *node, std::ostream &output) override;
  void Dispatch_(const BitwiseNotNode *node, std::ostream &output) override;
  void Dispatch_(const SelectNode *node, std::ostream &output) override;
  void Dispatch_(const RampNode *node, std::ostream &output) override;
  void Dispatch_(const LetNode *node, std::ostream &output) override;
  void Dispatch_(const CallNode *node, std::ostream &output) override;
  void Dispatch_(const TensorLoadNode *node, std::ostream &output) override;
  void Dispatch_(const TupleNode *node, std::ostream &output) override;
  void Dispatch_(const TupleGetItemNode *node, std::ostream &output) override;
  void Dispatch_(const TensorStoreNode *node) override;
  void Dispatch_(const BindNode *node) override;
  void Dispatch_(const AssertStmtNode *node) override;
  virtual void PrintAllocTensor_(const BindNode *node,
                                 const CallNode *allocation);
  void Dispatch_(const RegionStmtNode *node) override;
  void Dispatch_(const ForNode *node) override;
  void Dispatch_(const WhileNode *node) override;
  void Dispatch_(const IfNode *node) override;
  void Dispatch_(const SeqStmtNode *node) override;
  void Dispatch_(const EvaluateNode *node) override;
  void Dispatch_(const ReturnNode *node) override;
  void Dispatch_(const BreakNode *node) override;
  void Dispatch_(const ContinueNode *node) override;

  virtual std::string CastFromTo_(const std::string &value, PrimType source,
                                  PrimType target);
  virtual void PrintBinaryExpr_(const std::string &symbol, PrimType type,
                                Expr lhs, Expr rhs, std::ostream &output);
  virtual void PrintBinaryIntrinsic_(const CallNode *node, const char *symbol,
                                     std::ostream &output);
  virtual void PrintCallExtern_(Type type, String symbol,
                                const Array<Expr> &arguments, bool skip_first,
                                std::ostream &output);
  virtual std::string GetBufferRef_(PrimType type, const TensorVar &tensor,
                                    PrimExpr index);
  void RegisterHandleType_(const VarNode *variable, PrimType type);
  bool HandleTypeMatch_(const VarNode *variable, PrimType type) const;
  std::unordered_map<const VarNode *, std::string> alloc_storage_scope_;
  OpAttrMap<String> op_attr_global_symbol_ =
      Op::GetAttrMap<String>("TGlobalSymbol");

private:
  std::unordered_map<const VarNode *, PrimType> handle_data_type_;
  std::unordered_map<GlobalVar, String> internal_functions_;
  UniqueNameSupply func_name_supply_;
  void EscapeStringLiteral_(const std::string &value, std::ostream &output);
};

} // namespace tvm::codegen
#endif
