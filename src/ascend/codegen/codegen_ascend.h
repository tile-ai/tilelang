#ifndef TVM_TL_ASCEND_CODEGEN_CODEGEN_ASCEND_H_
#define TVM_TL_ASCEND_CODEGEN_CODEGEN_ASCEND_H_

#include <cstddef>
#include <initializer_list>
#include <string>

#include "backend/common/codegen/codegen_c_line_directives.h"
#include "target/source/codegen_c.h"
#include <tvm/arith/analyzer.h>

namespace tvm {
namespace codegen {

class CodeGenTileLangAscend final : public CodeGenCWithLineDirectives {
public:
  using CodeGenC::PrintType;

  void Init(bool output_ssa);
  void AddFunction(const PrimFunc &f);
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void PrintCCEVectorSuffix(DataType t, std::ostream &os);
  void VisitStmt_(const SBlockNode *op) final;
  void VisitStmt_(const ForNode *op) final;
  void VisitStmt_(const WhileNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitStmt_(const AllocBufferNode *op) final;
  void VisitStmt_(const BindNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void PrintStorageScope(const std::string &scope, std::ostream &os) final;
  bool IsScopePartOfType() const final;

  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const RampNode *op, std::ostream &os) final;
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;
  void VisitExpr_(const CastNode *op, std::ostream &os) final;
  void VisitExpr_(const VarNode *op, std::ostream &os) final;
  void VisitExpr_(const NotNode *op, std::ostream &os) final;
  void VisitExpr_(const BufferLoadNode *op, std::ostream &os) final;
  void VisitExpr_(const SelectNode *op, std::ostream &os) final;
  void VisitExpr_(const ShuffleNode *op, std::ostream &os) final;
  void VisitStmt_(const BufferStoreNode *op) final;

  std::string GetBufferRef(DataType t, const BufferNode *buffer,
                           PrimExpr index) final;

  // Vector element access (use .x/.y/.z/.w like CUDA, not .s0/.s1 from base)
  void PrintVecElemLoad(const std::string &vec, DataType t, int i,
                        std::ostream &os) final;
  void PrintVecElemStore(const std::string &vec, DataType t, int i,
                         const std::string &value) final;
  void PreFunctionBody(const PrimFunc &f) final;

protected:
  // Scalarize vector math calls (like CUDA)
  void PrintCallExtern(Type ret_type, ffi::String global_symbol,
                       const ffi::Array<PrimExpr> &args, bool skip_first_arg,
                       std::ostream &os) final;

  // Scalarize vector operations without Ascend SIMT vector support.
  void PrintVecBinaryOp(const std::string &op, DataType t, PrimExpr lhs,
                        PrimExpr rhs, std::ostream &os) final;

private:
  struct CApiArgument {
    PrimExpr value;
    std::string prefix{};
    std::string suffix{};
  };

  enum class VFMode { kNone, kSimt, kSimd };

  class VFModeScope {
  public:
    VFModeScope(CodeGenTileLangAscend *codegen, VFMode mode);
    ~VFModeScope();

    VFModeScope(const VFModeScope &) = delete;
    VFModeScope &operator=(const VFModeScope &) = delete;

  private:
    CodeGenTileLangAscend *codegen_;
    VFMode previous_mode_;
  };

  // Keep textual SSA reuse local to one emitted operation. This preserves
  // single-evaluation semantics without carrying memory-dependent values
  // across later stores.
  class SSAOperationScope {
  public:
    explicit SSAOperationScope(CodeGenTileLangAscend *codegen)
        : codegen_(codegen), scope_id_(codegen->BeginScope()) {}
    ~SSAOperationScope() { codegen_->EndScope(scope_id_); }

    SSAOperationScope(const SSAOperationScope &) = delete;
    SSAOperationScope &operator=(const SSAOperationScope &) = delete;

  private:
    CodeGenTileLangAscend *codegen_;
    int scope_id_;
  };

  std::string current_function_name_;
  int simtvf_helper_counter_{0};
  int simdvf_helper_counter_{0};
  VFMode vf_mode_{VFMode::kNone};

  bool has_gemm_l0_{false};
  bool has_gemm_l1_{false};
  bool has_gemm_l1_included_{false};
  bool has_nd2nz_copy_{false};
  bool has_nd2nz_copy_included_{false};
  bool has_simd_ops_{false};
  bool has_simd_inst_included_{false};
  bool has_gm_bypass_dcache_{false};
  bool has_gm_bypass_dcache_included_{false};
  bool has_ascend_std_math_included_{false};
  bool has_philox_rng_{false};
  bool has_philox_rng_included_{false};
  bool has_cooperative_groups_included_{false};
  std::string ascend_rng_state_var_;

  std::unordered_map<Var, IntImm, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      unroll_factor_;

  ffi::Array<Var> CollectVFCaptures(const SBlockNode *op) const;
  void EmitVFFunction(const SBlockNode *op, const ffi::Array<Var> &captures,
                      const std::string &helper_name,
                      const std::string &func_attrs, VFMode vf_mode);
  void EmitPhiloxVectorFloat(std::ostream &os, const std::string &distribution,
                             int lanes);
  bool EmitSimdMergingCall(const CallNode *op, std::ostream &os);
  std::string ResolveSimdIntrinsicName_(const Call &call,
                                        DataType result_dtype) const;
  void PrintSimdCall_(const Call &call, size_t num_expr_args, std::ostream &os);
  void EmitCApiCall_(const std::string &function_name,
                     std::initializer_list<CApiArgument> arguments);
  bool EmitAscendMemoryCall_(const CallNode *op);
  void EmitPipeBarrier_(const CallNode *op);
  void EmitHardEventSync_(const CallNode *op, bool notify);
  void EmitCrossCoreSync_(const CallNode *op, bool arrive);
  void EmitSetCopyPadValue_(const CallNode *op);
  void EmitGmToUbufCopy_(const CallNode *op);
  void EmitUbufToGmCopy_(const CallNode *op);
  void EmitGmToL1Copy_(const CallNode *op);
  void EmitFillL1_(const CallNode *op);
  void EmitL1ToL0Copy_(const CallNode *op, bool is_l0a);
  void EmitMxSfLoad_(const CallNode *op, bool is_l0a);
  void EmitL0cToUbufCopy_(const CallNode *op);
  void EmitL0cToGmCopy_(const CallNode *op);
  void EmitUbufToL1Copy_(const CallNode *op);
  void EmitNd2NzScatter_(const CallNode *op);
  void EmitNd2NzPostCopy_(const CallNode *op);
  void EmitMad_(const CallNode *op, bool is_mx);
  void EmitGemmL1_(const CallNode *op);
  void EmitBlockscaledGemmL1_(const CallNode *op);
  bool IsGlobalGmBuffer(const BufferNode *buffer) const;
  std::string GetGmBypassPtr(DataType dtype, const BufferNode *buffer,
                             PrimExpr index);

  enum class AscendKernelMode { kVector, kCube, kMix };
  AscendKernelMode kernel_mode_{AscendKernelMode::kVector};
  int mix_aiv_count_{2};
  bool enable_fast_math_{false};

  // Accumulates constraints along the traversal
  arith::Analyzer analyzer_;

  const char *VFModeName(VFMode mode) const;
  bool IsInsideSimtVF() const { return vf_mode_ == VFMode::kSimt; }
  bool IsInsideSimdVF() const { return vf_mode_ == VFMode::kSimd; }
  bool IsOutsideVF() const { return vf_mode_ == VFMode::kNone; }
  void ValidateSimdVectorType(DataType dtype, const std::string &context) const;
  void RejectGenericVectorInSimd(const char *op_name, DataType dtype) const;
  void ValidateGenericVectorType(const char *op_name, DataType dtype);

  bool VectorAccessIsAligned(const PrimExpr &index, int lanes);

  void EmitSimtVectorPredicateNot(const PrimExpr &value, DataType dtype,
                                  std::ostream &os);
  void EmitScalarizedLoad(const BufferLoadNode *op, std::ostream &os);
  bool EmitScalarizedStore(const BufferStoreNode *op);
  bool EmitOutOfVFBroadcastStore(const BufferStoreNode *op);
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_ASCEND_CODEGEN_CODEGEN_ASCEND_H_
