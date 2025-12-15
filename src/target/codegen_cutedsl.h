/*!
 * \file target/codegen_cutedsl.h
 * \brief Utility to generate CuTeDSL code
 */
#ifndef TVM_TL_TARGET_CODEGEN_CUTEDSL_H_
#define TVM_TL_TARGET_CODEGEN_CUTEDSL_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "codegen_py.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangCuTeDSL final : public CodeGenTileLangPY {
public:
  CodeGenTileLangCuTeDSL();

protected:
  void PrintFuncDecorator_(std::ostream &os) override; // NOLINT(*)
  void PreFunctionBody_(const PrimFunc &f) override;

protected:
  void PrintType(DataType t, std::ostream &os) override; // NOLINT(*)

  void VisitExpr_(const BroadcastNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op,
                  std::ostream &os) override;                     // NOLINT(*)
  void VisitExpr_(const CastNode *op, std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const DivNode *op, std::ostream &os) override;  // NOLINT(*)
  void VisitExpr_(const MinNode *op, std::ostream &os) override;  // NOLINT(*)
  void VisitExpr_(const MaxNode *op, std::ostream &os) override;  // NOLINT(*)
  void VisitExpr_(const CallNode *op, std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const BufferLoadNode *op,
                  std::ostream &os) override; // NOLINT(*)

  void VisitStmt_(const BufferStoreNode *op) override;
  void VisitStmt_(const AllocateNode *op) override;
  void VisitStmt_(const AttrStmtNode *op) override;
  void VisitStmt_(const ForNode *op) override;
  void VisitStmt_(const IfThenElseNode *op) override;
  void VisitStmt_(const EvaluateNode *op) override;

protected:
  virtual void PrintVecElemLoad_(const std::string &vec, DataType t, int i,
                                 std::ostream &os); // NOLINT(*)
  virtual void PrintVecElemStore_(const std::string &vec, DataType t, int i,
                                  const std::string &value);
  virtual void PrintVecStore_(const BufferNode *buffer, DataType t,
                              PrimExpr base, const std::string &value);
  void PrintVecBinaryOp_(const std::string &opstr, DataType dtype, PrimExpr lhs,
                         PrimExpr rhs,
                         std::ostream &os); // NOLINT(*)
  void PrintBinaryExpr_(const std::string &opstr, DataType dtype, PrimExpr lhs,
                        PrimExpr rhs,
                        std::ostream &os) override; // NOLINT(*)
  void PrintBinaryIntrinsic_(const CallNode *op, const char *opstr,
                             std::ostream &os) override; // NOLINT(*)

  void PrintCallExtern_(Type ret_type, ffi::String global_symbol,
                        const ffi::Array<PrimExpr> &args, bool skip_first_arg,
                        std::ostream &os) override; // NOLINT(*)

  std::string GetBufferPtr_(const BufferNode *buffer, PrimExpr index);
  std::string GetBufferRef_(DataType t, const BufferNode *buffer,
                            PrimExpr index) override;

  /*!
   * \brief Print expr representing the thread tag
   * \param IterVar iv The thread index to be binded;
   */
  virtual void BindThreadIndex_(const IterVar &iv); // NOLINT(*)

  virtual void PrintStorageSync_(const CallNode *op);

  std::string
  CanonicalizeFastmathFunctionName_(const std::string &func_name) const;

private:
  // The name of the mbarrier array in shared memory
  const std::string mbarrier_name_ = "mbarrier";

  std::unordered_map<const VarNode *, IntImm> unroll_factor_;

  std::vector<std::string> eviction_policy_names_ = {
      "EVICT_NORMAL", "EVICT_FIRST", "EVICT_LAST"};

  // Fastmath configuration (read from PassContext)
  bool enable_fastmath_ = false;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_CUTEDSL_H_
