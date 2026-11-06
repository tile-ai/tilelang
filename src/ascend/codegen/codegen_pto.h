/*!
 * \file ascend/codegen/codegen_pto.h
 * \brief Utility to generate PTO Python source.
 */
#ifndef TVM_TL_PTO_CODEGEN_CODEGEN_PTO_H_
#define TVM_TL_PTO_CODEGEN_CODEGEN_PTO_H_

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backend/common/codegen/codegen_py.h"
#include "ir/analyzer.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangPTO final : public CodeGenTileLangPY {
public:
  CodeGenTileLangPTO();
  void AddFunction(const GlobalVar &gvar, const tirx::Function &func) override;
  std::string Finish() override;

protected:
  void PrintFuncDecorator_(std::ostream &os) override; // NOLINT(*)
  void PrintFunctionSignature_(const ffi::String &function_name,
                               const tirx::Function &func,
                               std::ostream &os) override; // NOLINT(*)

  void Dispatch_(const BindObj *op) override;
  void Dispatch_(const TensorStoreObj *op) override;
  void Dispatch_(const RegionStmtObj *op) override;
  void Dispatch_(const SeqStmtObj *op) override;
  void Dispatch_(const ForObj *op) override;
  void Dispatch_(const WhileObj *op) override;
  void Dispatch_(const IfObj *op) override;
  void Dispatch_(const EvaluateObj *op) override;

  void Dispatch_(const prim::BroadcastObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const prim::RampObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const TensorLoadObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const prim::LetObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const prim::CastObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const FloatImmObj *op,
                 std::ostream &os) override;                    // NOLINT(*)
  void Dispatch_(const CallObj *op, std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const prim::NotObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const prim::SelectObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const prim::MinObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const prim::MaxObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const prim::ModObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const prim::DivObj *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const prim::LShiftObj *op, std::ostream &os) override;
  void Dispatch_(const prim::RShiftObj *op, std::ostream &os) override;
  void Dispatch_(const prim::ShuffleObj *op,
                 std::ostream &os) override; // NOLINT(*)

private:
  struct PTOGemmEmitContext {
    bool blockscaled{false};
    int64_t tile_m{0};
    int64_t tile_n{0};
    int64_t tile_k{0};
    int64_t base_k{0};
    int64_t sf_nz_stride{0};
    PrimType input_dtype = PrimType::Void();
    PrimType accum_dtype = PrimType::Void();
    std::string helper_name;
    std::string a_l0_name;
    std::string b_l0_name;
  };

  std::string ScalarType(PrimType t) const;
  std::string PointerTypeName(PrimType t, const std::string &space) const;
  /*! \brief Returns true if a pto.castptr is needed to access buffer_var
   *         as elem_dtype.  Handles PTO type-name equivalence (e.g.
   *         UInt(8) and Int(8) both map to pto.i8). */
  bool NeedsCastptr_(const VarObj *buffer_var, PrimType elem_dtype) const;
  std::string ScalarPointerBase_(const VarObj *buffer_var, PrimType elem_dtype,
                                 const std::string &scope);
  std::string GetPointerExpr(const VarObj *buffer_var, PrimType elem_dtype,
                             const PrimExpr &index);
  std::string GetPointerExpr(const VarObj *buffer, const PrimExpr &index);
  std::string GetVectorLocalRef(const VarObj *buffer_var, const PrimExpr &index,
                                const std::string &context);
  std::string GetMutableVectorRef(const Expr &address,
                                  const std::string &op_name);
  std::string PrintCondition(const Expr &condition);
  std::string GetAscendCopyGmUbExpr_(const CallObj *op);
  std::string GetAscendCopyUbGmExpr_(const CallObj *op);
  void RecordAscendCopyPadValue_(const CallObj *op);
  void EmitAscendSetAtomic(const CallObj *op);
  void EmitCopyPadMerge_(int merged_value_id, PrimType expected_dtype);
  std::string GetCopyPadValueExpr_(const Expr &value);
  void GetCopyEndpoint_(const Expr &expr, const char *context,
                        const VarObj **buffer_var, PrimExpr *index,
                        PrimType *dtype, std::string *scope) const;
  void ValidateUBCopyLayout_(const PrimExpr &index, PrimType dtype,
                             const PrimExpr &burst_num,
                             const PrimExpr &burst_len,
                             const PrimExpr &ub_stride,
                             const PrimExpr &left_padding,
                             const PrimExpr &right_padding, bool uses_padding,
                             const char *context) const;
  std::string EmitAllReduceExpr_(const std::string &func_name,
                                 const CallObj *op);
  void EmitInlineSimtVF(const RegionStmtObj *op, int64_t thread_x,
                        int64_t thread_y, int64_t thread_z);
  void ExtractSimtThreadExtents(const RegionStmtObj *op, int64_t *thread_x,
                                int64_t *thread_y, int64_t *thread_z) const;
  std::string ScalarLoad(const VarObj *buffer, const PrimExpr &index);
  std::string LocalVarStoreValue(const PrimExpr &value, PrimType dtype);
  void EmitScalarStore(const VarObj *buffer, const std::string &value,
                       const PrimExpr &index);
  void EmitBlockRegion_(const RegionStmtObj *op);
  void EmitTensorAllocation_(const BindObj *op, const CallObj *call) override;
  void EmitTensorDeclaration_(const BindObj *op, const CallObj *call) override;
  void EmitBufferAllocation(const VarObj *buffer);
  void EmitLocalVarInitialization_(const VarObj *buffer, std::string vid);
  void EmitScalarizedLoad(const TensorLoadObj *op, std::ostream &os);
  void EmitScalarizedStore(const TensorStoreObj *op);
  bool TryEmitRngBroadcastStore(const TensorStoreObj *op,
                                const PrimExpr &index);
  std::string ScopeOfBuffer(const VarObj *buffer) const;
  void PrintBinaryExpr_(const std::string &opstr, PrimType dtype, PrimExpr lhs,
                        PrimExpr rhs,
                        std::ostream &os) override; // NOLINT(*)
  void ValidateVecMode_(const CallObj *op, size_t mode_idx,
                        const char *expected = "MODE_ZEROING");
  bool EmitSimdMergingCall_(const CallObj *op,
                            std::ostream &os); // NOLINT(*)
  void PrintFloatMinMax_(const char *op_name, PrimType dtype, PrimExpr lhs,
                         PrimExpr rhs, std::ostream &os); // NOLINT(*)
  // Table-driven unary math mapping (sqrt/rsqrt/exp/log, extern C names and
  // tirx intrinsic names). Returns false when `name` is not covered.
  bool TryEmitUnaryMath_(const std::string &name, const Expr &arg,
                         std::ostream &os); // NOLINT(*)

  std::string current_function_name_;
  bool inside_simtvf_body_{false};
  bool enable_fast_math_{false};
  sym::Analyzer index_analyzer_{tl::CreateTileLangAnalyzer()};
  // Outer local buffers referenced by a SIMT section. Var identity is the
  // ownership key because lowering can leave duplicate name hints behind.
  std::unordered_set<const VarObj *> persistent_buffer_vars_;
  std::string GetLocalPtrExpr(const Expr &expr, const std::string &space,
                              PrimType fallback_dtype);
  std::string GetE8M0ScalePtrExpr(const Expr &expr);
  std::string GetLocalByteAddrExpr(const PrimExpr &index, PrimType elem_dtype,
                                   const std::string &context);
  std::string GetAccPtrExpr(const Expr &expr, PrimType dtype);
  const PTOGemmEmitContext &EnsureGemmHelper(const CallObj *op);
  const PTOGemmEmitContext &EnsureBlockscaledGemmHelper(const CallObj *op);
  void ValidateFractalAddressAlignment_(const PrimExpr &index, PrimType dtype,
                                        const char *context) const;
  void EmitAscendCopyGmToCbuf(const CallObj *op);
  void EmitAscendFillL1(const CallObj *op);
  void EmitAscendLoadCbufToL0(const CallObj *op, bool is_ca);
  void EmitAscendLoadMxSf(const CallObj *op, bool is_ca);
  void EmitAscendCrossCoreFlag(const CallObj *op, bool is_set);
  void EmitAscendMad(const CallObj *op);
  void EmitAscendGemmL1(const CallObj *op);
  void EmitAscendBlockscaledGemmL1(const CallObj *op);
  void EmitAscendCopyMatrixCcToUb(const CallObj *op);
  void EmitAscendCopyMatrixCcToGm(const CallObj *op);
  void EmitAscendCopyUbufToCbuf(const CallObj *op);
  void EmitAscendNd2NzPostCopy(const CallObj *op);
  void EmitAscendNd2NzScatter(const CallObj *op);
  void EmitGemmRun(const PTOGemmEmitContext &ctx, const std::string &a_mat,
                   const std::string &b_mat, const std::string &acc,
                   const std::string &clear_accum,
                   const std::string &unit_flag_ctrl, int64_t hf32_mode);
  void
  EmitBlockscaledGemmRun(const PTOGemmEmitContext &ctx,
                         const std::string &a_mat, const std::string &b_mat,
                         const std::string &sfa_mat, const std::string &sfb_mat,
                         const std::string &acc, const std::string &clear_accum,
                         const std::string &sf_k_offset,
                         const std::string &unit_flag_ctrl);
  std::string LocalVarID(const VarObj *var);
  bool IsLocalVarBuffer(const VarObj *var) const;
  void EmitMixedEntrySnapshot(const VarObj *var);
  void RestoreMixedSectionVariables(const RegionStmtObj *section);
  bool HasAscendGemmL1(const tirx::Function &func) const;
  bool HasAscendBlockscaledGemmL1(const tirx::Function &func) const;
  bool HasAscendMad(const tirx::Function &func) const;
  bool IsAscendCubeKernel(const tirx::Function &func) const;
  bool IsAscendMixedKernel(const tirx::Function &func) const;

  bool current_function_has_gemm_{false};
  bool current_function_is_cube_{false};
  bool current_function_is_mixed_{false};
  // Mixed entry functions are Cube-capable overall, but loops authored in a
  // Vector section must use Python range so PTODSL can infer loop-carried SSA.
  bool in_mixed_vector_section_{false};
  bool inside_mixed_section_{false};
  // AscendC exposes padding through a stateful register write followed by a
  // dataSelect GM->UB copy. PTODSL models the same operation as an inline
  // pad=(value, left, right) argument. Capture each setter into a unique SSA
  // alias so later scalar mutation cannot change the reaching hardware value.
  int copy_pad_value_counter_{0};
  int current_copy_pad_value_id_{-1};
  PrimType current_copy_pad_value_dtype_ = PrimType::Void();
  // A single function-wide constant pad value is safe to materialize directly
  // at a guarded copy when scheduling split its setter into an earlier guard.
  ffi::Optional<PrimExpr> uniform_const_copy_pad_value_;
  // Cube-side allocations hold integer addresses despite their TIR pointer
  // annotations. Vector-side allocations and make_tensor binds hold pointers.
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> integer_address_vars_;
  std::unordered_set<const VarObj *> local_var_buffers_;
  std::unordered_map<Call, int64_t, ObjectPtrHash, ObjectPtrEqual>
      hf32_mode_by_gemm_;
  std::unordered_map<Call, bool, ObjectPtrHash, ObjectPtrEqual>
      n_direction_by_mad_;
  std::unordered_map<Var, std::string, ObjectPtrHash, ObjectPtrEqual>
      loop_condition_snapshot_ids_;
  // pragma_unroll_factor is lowered to an AttrStmt around its loop. Retain the
  // annotated variable so nested unrolled loops do not inherit the factor.
  Optional<Var> current_unroll_factor_loop_var_;
  int64_t current_unroll_factor_{0};
  // A physical Cube/Vector section owns its SSA definitions.  Keep a shadow
  // of each outer local.var captured by any section and restore it at every
  // section boundary so definitions cannot leak between sibling regions.
  std::unordered_set<const VarObj *> mixed_captured_local_vars_;
  std::unordered_map<const RegionStmtObj *, std::unordered_set<const VarObj *>>
      mixed_external_vars_by_section_;
  std::unordered_map<const VarObj *, std::string> mixed_entry_snapshot_ids_;
  // Pair-producing simd ops (vintlv/vdintlv/vld2) bind a tuple value; each
  // unpack via pair_get picks element 0 (low) or 1 (high). Maps the bound
  // VarObj to its (low, high) SSA name pair.
  std::unordered_map<const VarObj *, std::pair<std::string, std::string>>
      simd_pair_vars_;
  std::vector<PTOGemmEmitContext> gemm_emit_contexts_;
  std::unordered_map<Call, size_t, ObjectPtrHash, ObjectPtrEqual>
      gemm_emit_context_by_call_;
  bool gemm_zero_addr_emitted_{false};
  // Native Python break is valid only while emitting a Python for/while body.
  int native_loop_depth_{0};
  // Nesting depth of tl.simdvf_scope; only the outermost emits pto.vecscope.
  int simdvf_nesting_depth_{0};
  // True when a Bind of simd_pset / rematerializable simd_vdup should be
  // inlined at each use (avoids vreg/mask SSA crossing scf.for / range).
  bool IsInlineableInvariantSimdBind(const PrimExpr &value) const;
  std::pair<std::string, std::string>
  ParseHardEventPair(const std::string &hard_event) const;

  // RNG state: the Philox key is immutable, while the draw counter and cached
  // Box-Muller value are Python SSA names carried through runtime control flow.
  std::string pto_rng_state_var_;
  std::string pto_rng_counter_var_;
  std::string pto_rng_normal_cache_var_;
  std::string pto_rng_has_normal_var_;
  bool pto_rng_initialized_{false};
  int pto_rng_result_counter_{0};
  int pto_if_result_counter_{0};
  void ResetRngState_();
  // Clears all function-scoped emission state; see AddFunction.
  void ResetFunctionState_();
  void EmitRngInit(const CallObj *op);
  std::string EmitRngRand(const CallObj *op);
  std::string EmitRngRandFloat(const CallObj *op);
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_PTO_CODEGEN_CODEGEN_PTO_H_
