/*!
 * \file target/codegen_cutedsl.h
 * \brief Utility to generate CuTeDSL code
 */
#ifndef TVM_TL_TARGET_CODEGEN_CUTEDSL_H_
#define TVM_TL_TARGET_CODEGEN_CUTEDSL_H_

#include "support/check.h"
#include <tvm/codegen/codegen.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/region.h>

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
  using CodeGenTileLangPY::Dispatch_;
  using CodeGenTileLangPY::PrintType;
  void InitFuncState_(const tirx::Function &f) override;

  void PrintFuncDecorator_(std::ostream &os) override; // NOLINT(*)
  void PreFunctionBody_(const tirx::Function &f) override;

protected:
  void PrintType(const PrimType &t, std::ostream &os) override; // NOLINT(*)

  void Dispatch_(const prim::BroadcastObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const FloatImmObj *op,
                 std::ostream &os) override;                      // NOLINT(*)
  void Dispatch_(const IntImmObj *op, std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const prim::CastObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const prim::DivObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const prim::ModObj *op, std::ostream &os) override;
  void Dispatch_(const prim::FloorDivObj *op, std::ostream &os) override;
  void Dispatch_(const prim::FloorModObj *op, std::ostream &os) override;
  void Dispatch_(const prim::LShiftObj *op, std::ostream &os) override;
  void Dispatch_(const prim::RShiftObj *op, std::ostream &os) override;
  void PrintIntegerDivMod_(const char *mode, PrimType dtype, PrimExpr lhs,
                           PrimExpr rhs, std::ostream &os);
  void Dispatch_(const prim::MinObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const prim::MaxObj *op,
                 std::ostream &os) override;                    // NOLINT(*)
  void Dispatch_(const CallObj *op, std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const prim::SelectObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const prim::LetObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const prim::ShuffleObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const TensorLoadObj *op,
                 std::ostream &os) override; // NOLINT(*)

  void Dispatch_(const TensorStoreObj *op) override;
  void Dispatch_(const BindObj *op) override;
  void EmitTensorAllocation_(const BindObj *op,
                             const CallObj *allocation) override;
  void EmitTensorDeclaration_(const BindObj *op, const CallObj *call) override;
  void Dispatch_(const RegionStmtObj *op) override;
  void Dispatch_(const ForObj *op) override;
  void Dispatch_(const BreakObj *op) override;
  void Dispatch_(const IfObj *op) override;
  void Dispatch_(const EvaluateObj *op) override;
  void Dispatch_(const SeqStmtObj *op) override;

protected:
  virtual void PrintVecElemLoad_(const std::string &vec, PrimType t, int i,
                                 std::ostream &os); // NOLINT(*)
  virtual void PrintVecElemStore_(const std::string &vec, PrimType t, int i,
                                  const std::string &value);
  virtual void PrintVecStore_(const VarObj *buffer, PrimType t, PrimExpr base,
                              const std::string &value);
  void PrintVecBinaryOp_(const std::string &opstr, PrimType dtype, PrimExpr lhs,
                         PrimExpr rhs,
                         std::ostream &os); // NOLINT(*)
  void PrintBinaryExpr_(const std::string &opstr, PrimType dtype, PrimExpr lhs,
                        PrimExpr rhs,
                        std::ostream &os) override; // NOLINT(*)
  void PrintBinaryIntrinsic_(const CallObj *op, const char *opstr,
                             std::ostream &os) override; // NOLINT(*)

  void PrintCallExtern_(Type ret_type, ffi::String global_symbol,
                        const ffi::Array<Expr> &args, bool skip_first_arg,
                        std::ostream &os) override; // NOLINT(*)

  std::string GetBufferPtr_(const VarObj *buffer, PrimExpr index);
  std::string GetSubByteBufferPtrAsByte_(const VarObj *buffer, PrimExpr index);
  std::string PrintBytePointerExpr_(const Expr &expr);
  std::string GetBufferRef_(PrimType t, const VarObj *buffer,
                            PrimExpr index) override;
  PrimExpr LinearizeBufferIndices_(const VarObj *buffer,
                                   const ffi::Array<PrimExpr> &indices);

  // Get pointer string from a Var expression (local buffer -> vid.iterator)
  std::string GetVarPtr_(const Expr &expr);

  /*!
   * \brief Print expr representing the thread tag
   * \param IterVar iv The thread index to be binded;
   */
  virtual void BindThreadIndex_(const PrimVar &var,
                                const String &thread_tag); // NOLINT(*)

  virtual void PrintStorageSync_(const CallObj *op);

  std::string
  CanonicalizeFastmathFunctionName_(const std::string &func_name) const;

private:
  // Both the public native Break and retained intrinsic use the same guard.
  void EmitLoopBreak_();

  // The name of the mbarrier array in shared memory
  const std::string mbarrier_name_ = "mbarrier";

  std::unordered_map<const VarObj *, IntImm> unroll_factor_;

  std::unordered_set<const VarObj *> raw_pointer_vars_;
  std::unordered_set<const VarObj *> zero_like_vars_;

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
