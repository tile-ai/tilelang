/*!
 * \file codegen_py.h
 * \brief Common Python source emission over native shared IR.
 */
#ifndef TVM_TL_TARGET_CODEGEN_PY_H_
#define TVM_TL_TARGET_CODEGEN_PY_H_

#include <tvm/ir/expr_functor.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/unique_name_supply.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <unordered_map>

#include "backend/common/codegen/codegen_ir_utils.h"
#include "codegen/c/codegen_source_base.h"

namespace tvm::codegen {
using namespace tirx;
using namespace ffi;
using namespace tvm::prim;

/*! \brief Shared emitter for Python-language backends, with no legacy IR
 * objects. */
class CodeGenTileLangPY
    : public tvm::ExprFunctor<void(const Expr &, std::ostream &)>,
      public tirx::StmtFunctor<void(const Stmt &)>,
      public CodeGenSourceBase {
public:
  virtual void AddFunction(const GlobalVar &gvar, const tirx::Function &func);
  virtual std::string Finish();

protected:
  String GetFunctionName_(const GlobalVar &gvar);
  virtual void RegisterFunction_(const GlobalVar &gvar,
                                 const tirx::Function &func);
  virtual void InitFuncState_(const tirx::Function &func);
  virtual void PrintFunctionSignature_(const String &name,
                                       const tirx::Function &func,
                                       std::ostream &os);
  virtual void PrintFuncDecorator_(std::ostream &os) {}
  virtual void PreFunctionBody_(const tirx::Function &func) {}
  void ReserveKeywordsAsUnique_();
  void PrintSSAAssign(const std::string &target, const std::string &src,
                      const Type &type) override;
  using CodeGenSourceBase::PrintType;
  void PrintType(const PrimType &type, std::ostream &os) override;

  void PrintStmt_(const Stmt &stmt) {
    tirx::StmtFunctor<void(const Stmt &)>::Dispatch(stmt);
  }
  void PrintSuite_(const Stmt &stmt);
  void PrintExpr_(const Expr &expr, std::ostream &os) {
    tvm::ExprFunctor<void(const Expr &, std::ostream &)>::Dispatch(expr, os);
  }
  std::string PrintExpr_(const Expr &expr) {
    std::ostringstream os;
    PrintExpr_(expr, os);
    return os.str();
  }

#define TILELANG_PY_EXPR_HOOK(Node)                                            \
  void Dispatch_(const Node *op, std::ostream &os) override
  TILELANG_PY_EXPR_HOOK(VarObj);
  TILELANG_PY_EXPR_HOOK(IntImmObj);
  TILELANG_PY_EXPR_HOOK(FloatImmObj);
  TILELANG_PY_EXPR_HOOK(StringImmObj);
  TILELANG_PY_EXPR_HOOK(DataTypeImmObj);
  TILELANG_PY_EXPR_HOOK(GenericConstObj);
  TILELANG_PY_EXPR_HOOK(TupleObj);
  TILELANG_PY_EXPR_HOOK(TupleGetItemObj);
  TILELANG_PY_EXPR_HOOK(CastObj);
  TILELANG_PY_EXPR_HOOK(AddObj);
  TILELANG_PY_EXPR_HOOK(SubObj);
  TILELANG_PY_EXPR_HOOK(MulObj);
  TILELANG_PY_EXPR_HOOK(DivObj);
  TILELANG_PY_EXPR_HOOK(ModObj);
  TILELANG_PY_EXPR_HOOK(FloorDivObj);
  TILELANG_PY_EXPR_HOOK(FloorModObj);
  TILELANG_PY_EXPR_HOOK(MinObj);
  TILELANG_PY_EXPR_HOOK(MaxObj);
  TILELANG_PY_EXPR_HOOK(EQObj);
  TILELANG_PY_EXPR_HOOK(NEObj);
  TILELANG_PY_EXPR_HOOK(LTObj);
  TILELANG_PY_EXPR_HOOK(LEObj);
  TILELANG_PY_EXPR_HOOK(GTObj);
  TILELANG_PY_EXPR_HOOK(GEObj);
  TILELANG_PY_EXPR_HOOK(AndObj);
  TILELANG_PY_EXPR_HOOK(OrObj);
  TILELANG_PY_EXPR_HOOK(NotObj);
  TILELANG_PY_EXPR_HOOK(LShiftObj);
  TILELANG_PY_EXPR_HOOK(RShiftObj);
  TILELANG_PY_EXPR_HOOK(BitwiseAndObj);
  TILELANG_PY_EXPR_HOOK(BitwiseOrObj);
  TILELANG_PY_EXPR_HOOK(BitwiseXorObj);
  TILELANG_PY_EXPR_HOOK(BitwiseNotObj);
  TILELANG_PY_EXPR_HOOK(SelectObj);
  TILELANG_PY_EXPR_HOOK(RampObj);
  TILELANG_PY_EXPR_HOOK(LetObj);
  TILELANG_PY_EXPR_HOOK(CallObj);
  TILELANG_PY_EXPR_HOOK(TensorLoadObj);
#undef TILELANG_PY_EXPR_HOOK

#define TILELANG_PY_STMT_HOOK(Node) void Dispatch_(const Node *op) override
  TILELANG_PY_STMT_HOOK(TensorStoreObj);
  TILELANG_PY_STMT_HOOK(BindObj);
  TILELANG_PY_STMT_HOOK(RegionStmtObj);
  TILELANG_PY_STMT_HOOK(ForObj);
  TILELANG_PY_STMT_HOOK(WhileObj);
  TILELANG_PY_STMT_HOOK(IfObj);
  TILELANG_PY_STMT_HOOK(SeqStmtObj);
  TILELANG_PY_STMT_HOOK(EvaluateObj);
  TILELANG_PY_STMT_HOOK(AssertStmtObj);
  TILELANG_PY_STMT_HOOK(ReturnObj);
  TILELANG_PY_STMT_HOOK(BreakObj);
  TILELANG_PY_STMT_HOOK(ContinueObj);
#undef TILELANG_PY_STMT_HOOK

  virtual void EmitTensorAllocation_(const BindObj *bind, const CallObj *call);
  virtual void EmitTensorDeclaration_(const BindObj *bind, const CallObj *call);
  virtual std::string CastFromTo_(const std::string &value, PrimType from,
                                  PrimType target);
  virtual void PrintBinaryExpr_(const std::string &op, PrimType type,
                                PrimExpr lhs, PrimExpr rhs, std::ostream &os);
  virtual void PrintBinaryIntrinsic_(const CallObj *op, const char *symbol,
                                     std::ostream &os);
  virtual void PrintCallExtern_(Type type, String symbol,
                                const Array<Expr> &args, bool skip_first_arg,
                                std::ostream &os);
  virtual std::string GetBufferRef_(PrimType type, const VarObj *tensor,
                                    PrimExpr index);
  void RegisterHandleType_(const VarObj *var, PrimType type);
  bool HandleTypeMatch_(const VarObj *var, PrimType type) const;
  void EscapeStringLiteral_(const std::string &value, std::ostream &os);
  std::unordered_map<const VarObj *, std::string> alloc_storage_scope_;
  OpAttrMap<ffi::String> op_attr_global_symbol_ =
      Op::GetAttrMap(tirx::op_attr::kGlobalSymbol);
  const Op &builtin_call_extern_ = tirx::call_extern_op();
  const Op &builtin_call_pure_extern_ = tirx::call_pure_extern_op();

private:
  std::unordered_map<const VarObj *, PrimType> handle_data_type_;
  std::unordered_map<GlobalVar, String> internal_functions_;
  UniqueNameSupply func_name_supply_;
};
} // namespace tvm::codegen
#endif // TVM_TL_TARGET_CODEGEN_PY_H_
