#ifndef TVM_TL_ASCEND_CODEGEN_CODEGEN_ASCEND_H_
#define TVM_TL_ASCEND_CODEGEN_CODEGEN_ASCEND_H_

#include <cstddef>
#include <initializer_list>
#include <string>

#include "backend/common/codegen/codegen_c_base.h"
#include "backend/common/codegen/codegen_py.h"
#include <tvm/sym/analyzer.h>

namespace tvm {
namespace codegen {

class CodeGenTileLangAscend final : public CodeGenTileLangCBase {
public:
  using CodeGenC::PrintType;

  void Init(bool output_ssa);
  void AddFunction(const GlobalVar &global, const tirx::Function &f);
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintType(const PrimType &t, std::ostream &os) final; // NOLINT(*)
  void PrintCCEVectorSuffix(PrimType t, std::ostream &os);
  void Dispatch_(const RegionStmtNode *op) final;
  void Dispatch_(const ForNode *op) final;
  void Dispatch_(const WhileNode *op) final;
  void PrintAllocTensor(const BindNode *op, const CallNode *allocation);
  void Dispatch_(const BindNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void Dispatch_(const CallNode *op, std::ostream &os) final;
  void PrintStorageScope(const runtime::MemoryScope &memory_scope,
                         std::ostream &os) final;
  bool IsScopePartOfType() const final;

  void Dispatch_(const BroadcastNode *op, std::ostream &os) final; // NOLINT(*)
  void Dispatch_(const RampNode *op, std::ostream &os) final;
  void Dispatch_(const FloatImmNode *op, std::ostream &os) final;
  void Dispatch_(const CastNode *op, std::ostream &os) final;
  void Dispatch_(const VarNode *op, std::ostream &os) final;
  void Dispatch_(const NotNode *op, std::ostream &os) final;
  void Dispatch_(const TensorLoadNode *op, std::ostream &os) final;
  void Dispatch_(const SelectNode *op, std::ostream &os) final;
  void Dispatch_(const ShuffleNode *op, std::ostream &os) final;
  void Dispatch_(const TensorStoreNode *op) final;

  std::string GetBufferRef(const PrimType &t, const VarNode *buffer,
                           PrimExpr index) final;

  // Vector element access (use .x/.y/.z/.w like CUDA, not .s0/.s1 from base)
  void PrintVecElemLoad(const std::string &vec, const PrimType &t, int i,
                        std::ostream &os) final;
  void PrintVecElemStore(const std::string &vec, const PrimType &t, int i,
                         const std::string &value) final;
  void PreFunctionBody(const tirx::Function &f) final;

protected:
  // Scalarize vector math calls (like CUDA)
  void PrintCallExtern(Type ret_type, ffi::String global_symbol,
                       const ffi::Array<Expr> &args, bool skip_first_arg,
                       std::ostream &os) final;

  // Scalarize vector operations without Ascend SIMT vector support.
  void PrintVecBinaryOp(const std::string &op, const PrimType &t, PrimExpr lhs,
                        PrimExpr rhs, std::ostream &os) final;

private:
  struct CApiArgument {
    Expr value;
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
  std::string sub_block_id_var_;

  ffi::Array<Var> CollectVFCaptures(const RegionStmtNode *op) const;
  void EmitVFFunction(const RegionStmtNode *op, const ffi::Array<Var> &captures,
                      const std::string &helper_name,
                      const std::string &func_attrs, VFMode vf_mode);
  void EmitPhiloxVectorFloat(std::ostream &os, const std::string &distribution,
                             int lanes);
  bool EmitSimdMergingCall(const CallNode *op, std::ostream &os);
  std::string ResolveSimdIntrinsicName_(const Call &call,
                                        PrimType result_dtype) const;
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
  bool IsGlobalGmBuffer(const VarNode *buffer) const;
  std::string GetGmBypassPtr(PrimType dtype, const VarNode *buffer,
                             PrimExpr index);

  enum class AscendKernelMode { kVector, kCube, kMix };
  AscendKernelMode kernel_mode_{AscendKernelMode::kVector};
  int mix_aiv_count_{2};
  bool enable_fast_math_{false};

  // Accumulates constraints along the traversal
  sym::Analyzer analyzer_;

  const char *VFModeName(VFMode mode) const;
  bool IsInsideSimtVF() const { return vf_mode_ == VFMode::kSimt; }
  bool IsInsideSimdVF() const { return vf_mode_ == VFMode::kSimd; }
  bool IsOutsideVF() const { return vf_mode_ == VFMode::kNone; }
  void ValidateSimdVectorType(PrimType dtype, const std::string &context) const;
  void RejectGenericVectorInSimd(const char *op_name, PrimType dtype) const;
  void ValidateGenericVectorType(const char *op_name, PrimType dtype);

  bool VectorAccessIsAligned(const PrimExpr &index, int lanes);

  void EmitSimtVectorPredicateNot(const PrimExpr &value, PrimType dtype,
                                  std::ostream &os);
  void EmitScalarizedLoad(const TensorLoadNode *op, std::ostream &os);
  bool EmitScalarizedStore(const TensorStoreNode *op);
  bool EmitOutOfVFBroadcastStore(const TensorStoreNode *op);
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_ASCEND_CODEGEN_CODEGEN_ASCEND_H_
