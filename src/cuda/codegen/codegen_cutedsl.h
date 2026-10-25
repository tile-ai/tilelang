/*!
 * \file target/codegen_cutedsl.h
 * \brief Utility to generate CuTeDSL code
 */
#ifndef TVM_TL_TARGET_CODEGEN_CUTEDSL_H_
#define TVM_TL_TARGET_CODEGEN_CUTEDSL_H_

#include "support/check.h"
#include <tvm/codegen/codegen.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op/memory.h>

#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "backend/common/codegen/codegen_py.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangCuTeDSL final : public CodeGenTileLangPY {
public:
  CodeGenTileLangCuTeDSL();

protected:
  void InitFuncState_(const tirx::Function &f) override;

  void PrintFuncDecorator_(std::ostream &os) override; // NOLINT(*)
  void PreFunctionBody_(const tirx::Function &f) override;

protected:
  void PrintType(PrimType t, std::ostream &os) override; // NOLINT(*)

  void Dispatch_(const BroadcastNode *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const FloatImmNode *op,
                 std::ostream &os) override;                       // NOLINT(*)
  void Dispatch_(const IntImmNode *op, std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const CastNode *op, std::ostream &os) override;   // NOLINT(*)
  void Dispatch_(const DivNode *op, std::ostream &os) override;    // NOLINT(*)
  void Dispatch_(const MinNode *op, std::ostream &os) override;    // NOLINT(*)
  void Dispatch_(const MaxNode *op, std::ostream &os) override;    // NOLINT(*)
  void Dispatch_(const CallNode *op, std::ostream &os) override;   // NOLINT(*)
  void Dispatch_(const SelectNode *op, std::ostream &os) override; // NOLINT(*)
                                                                   // NOLINT(*)
  void Dispatch_(const ShuffleNode *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const TensorLoadNode *op,
                 std::ostream &os) override; // NOLINT(*)

  void Dispatch_(const TensorStoreNode *op) override;
  void Dispatch_(const BindNode *op) override;
  void PrintAllocTensor_(const BindNode *op,
                         const CallNode *allocation) override;
  void Dispatch_(const RegionStmtNode *op) override;
  void Dispatch_(const ForNode *op) override;
  void Dispatch_(const BreakNode *op) override;
  void Dispatch_(const IfNode *op) override;
  void Dispatch_(const EvaluateNode *op) override;
  void Dispatch_(const SeqStmtNode *op) override;

protected:
  virtual void PrintVecElemLoad_(const std::string &vec, PrimType t, int i,
                                 std::ostream &os); // NOLINT(*)
  virtual void PrintVecElemStore_(const std::string &vec, PrimType t, int i,
                                  const std::string &value);
  virtual void PrintVecStore_(const TensorVar &buffer, PrimType t,
                              PrimExpr base, const std::string &value);
  void PrintVecBinaryOp_(const std::string &opstr, PrimType dtype, Expr lhs,
                         Expr rhs,
                         std::ostream &os); // NOLINT(*)
  void PrintBinaryExpr_(const std::string &opstr, PrimType dtype, Expr lhs,
                        Expr rhs,
                        std::ostream &os) override; // NOLINT(*)
  void PrintBinaryIntrinsic_(const CallNode *op, const char *opstr,
                             std::ostream &os) override; // NOLINT(*)

  void PrintCallExtern_(Type ret_type, ffi::String global_symbol,
                        const ffi::Array<Expr> &args, bool skip_first_arg,
                        std::ostream &os) override; // NOLINT(*)

  std::string GetBufferPtr_(const TensorVar &buffer, PrimExpr index);
  std::string GetSubByteBufferPtrAsByte_(const TensorVar &buffer,
                                         PrimExpr index);
  std::string PrintBytePointerExpr_(const Expr &expr);
  std::string GetBufferRef_(PrimType t, const TensorVar &buffer,
                            PrimExpr index) override;
  PrimExpr LinearizeBufferIndices_(const TensorVar &buffer,
                                   const ffi::Array<PrimExpr> &indices);

  // Get pointer string from a Var expression (local buffer -> vid.iterator)
  std::string GetVarPtr_(const Expr &expr);

  /*!
   * \brief Print expr representing the thread tag
   * \param IterVar iv The thread index to be binded;
   */
  virtual void BindThreadIndex_(const PrimVar &variable,
                                const std::string &tag); // NOLINT(*)

  virtual void PrintStorageSync_(const CallNode *op);

  std::string
  CanonicalizeFastmathFunctionName_(const std::string &func_name) const;

private:
  // The name of the mbarrier array in shared memory
  const std::string mbarrier_name_ = "mbarrier";

  std::unordered_map<const VarNode *, IntImm> unroll_factor_;

  std::unordered_set<const VarNode *> raw_pointer_vars_;
  std::unordered_set<const VarNode *> zero_like_vars_;

  std::vector<std::string> eviction_policy_names_ = {
      "EVICT_NORMAL", "EVICT_FIRST", "EVICT_LAST"};

  std::optional<std::tuple<int64_t, int64_t, int64_t>> cluster_dims_;

  // Fastmath configuration (read from PassContext)
  bool enable_fastmath_ = false;

  // Loop-break guard transformation state
  // When a for-loop contains loop_break(), we replace `break` with a guard
  // variable pattern since CuTeDSL doesn't support early exit (break).
  bool in_break_loop_ = false;
  int loop_break_counter_ = 0;
  int current_break_id_ = -1;
  // Set to true when loop_break() replacement is emitted within current SeqStmt
  bool break_emitted_in_seq_ = false;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_CUTEDSL_H_
