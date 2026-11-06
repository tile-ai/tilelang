#ifndef TVM_TL_ASCEND_CODEGEN_CODEGEN_ASCEND_H_
#define TVM_TL_ASCEND_CODEGEN_CODEGEN_ASCEND_H_

#include <cstddef>
#include <initializer_list>
#include <string>

#include "backend/common/codegen/codegen_c_base.h"
#include "codegen/c/codegen_c.h"
#include "ir/analyzer.h"
#include <tvm/sym/analyzer.h>

namespace tvm {
namespace codegen {

class CodeGenTileLangAscend final : public CodeGenTileLangCBase {
public:
  using CodeGenTileLangCBase::Dispatch_;
  using CodeGenTileLangCBase::PrintType;
  void InitFuncState(const tirx::Function &f) override {
    CodeGenTileLangCBase::InitFuncState(f);
    analyzer_ = tl::CreateTileLangAnalyzer();
    name_supply_->ReserveName("block_idx");
    for (const auto &param : f->params) {
      if (const auto *pointer = param->ty.as<PtrTypeObj>())
        alloc_storage_scope_[param.get()] = pointer->scope;
    }
  }

  void Init(bool output_ssa);
  void AddFunction(const GlobalVar &gvar, const tirx::Function &f);
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintType(const PrimType &t, std::ostream &os) final; // NOLINT(*)
  void PrintCCEVectorSuffix(const PrimType &t, std::ostream &os);
  void EmitBlockRegion_(const RegionStmtObj *op);
  void Dispatch_(const ForObj *op) final;
  void Dispatch_(const WhileObj *op) final;
  void Dispatch_(const RegionStmtObj *op) final;
  void EmitTensorAllocation_(const BindObj *op);
  void Dispatch_(const BindObj *op) final;
  void PrintStorageSync(const CallObj *op) final;
  void Dispatch_(const CallObj *op, std::ostream &os) final;
  void PrintStorageScope(const runtime::MemoryScope &memory_scope,
                         std::ostream &os) final;
  bool IsScopePartOfType() const final;

  void Dispatch_(const prim::BroadcastObj *op,
                 std::ostream &os) final; // NOLINT(*)
  void Dispatch_(const prim::RampObj *op, std::ostream &os) final;
  void Dispatch_(const FloatImmObj *op, std::ostream &os) final;
  void Dispatch_(const prim::CastObj *op, std::ostream &os) final;
  void Dispatch_(const VarObj *op, std::ostream &os) final;
  void Dispatch_(const prim::NotObj *op, std::ostream &os) final;
  void Dispatch_(const TensorLoadObj *op, std::ostream &os) final;
  void Dispatch_(const prim::SelectObj *op, std::ostream &os) final;
  void Dispatch_(const prim::ShuffleObj *op, std::ostream &os) final;
  void Dispatch_(const TensorStoreObj *op) final;

  std::string GetBufferRef(const PrimType &t, const VarObj *buffer,
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

  std::unordered_map<Var, IntImm, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      unroll_factor_;

  ffi::Array<Var> CollectVFCaptures(const RegionStmtObj *op) const;
  void EmitVFFunction(const RegionStmtObj *op, const ffi::Array<Var> &captures,
                      const std::string &helper_name,
                      const std::string &func_attrs, VFMode vf_mode);
  void EmitPhiloxVectorFloat(std::ostream &os, const std::string &distribution,
                             int lanes);
  bool EmitSimdMergingCall(const CallObj *op, std::ostream &os);
  std::string ResolveSimdIntrinsicName_(const Call &call,
                                        PrimType result_dtype) const;
  void PrintSimdCall_(const Call &call, size_t num_expr_args, std::ostream &os);
  void EmitCApiCall_(const std::string &function_name,
                     std::initializer_list<CApiArgument> arguments);
  bool EmitAscendMemoryCall_(const CallObj *op);
  void EmitPipeBarrier_(const CallObj *op);
  void EmitHardEventSync_(const CallObj *op, bool notify);
  void EmitCrossCoreSync_(const CallObj *op, bool arrive);
  void EmitSetCopyPadValue_(const CallObj *op);
  void EmitGmToUbufCopy_(const CallObj *op);
  void EmitUbufToGmCopy_(const CallObj *op);
  void EmitGmToL1Copy_(const CallObj *op);
  void EmitFillL1_(const CallObj *op);
  void EmitL1ToL0Copy_(const CallObj *op, bool is_l0a);
  void EmitMxSfLoad_(const CallObj *op, bool is_l0a);
  void EmitL0cToUbufCopy_(const CallObj *op);
  void EmitL0cToGmCopy_(const CallObj *op);
  void EmitUbufToL1Copy_(const CallObj *op);
  void EmitNd2NzScatter_(const CallObj *op);
  void EmitNd2NzPostCopy_(const CallObj *op);
  void EmitMad_(const CallObj *op, bool is_mx);
  void EmitGemmL1_(const CallObj *op);
  void EmitBlockscaledGemmL1_(const CallObj *op);
  bool IsGlobalGmBuffer(const VarObj *buffer) const;
  std::string GetGmBypassPtr(PrimType dtype, const VarObj *buffer,
                             PrimExpr index);

  enum class AscendKernelMode { kVector, kCube, kMix };
  AscendKernelMode kernel_mode_{AscendKernelMode::kVector};
  int mix_aiv_count_{2};
  bool enable_fast_math_{false};

  // Accumulates constraints along the traversal
  sym::Analyzer analyzer_{tl::CreateTileLangAnalyzer()};

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
  void EmitScalarizedLoad(const TensorLoadObj *op, std::ostream &os);
  bool EmitScalarizedStore(const TensorStoreObj *op);
  bool EmitOutOfVFBroadcastStore(const TensorStoreObj *op);
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_ASCEND_CODEGEN_CODEGEN_ASCEND_H_
