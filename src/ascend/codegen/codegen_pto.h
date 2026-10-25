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

namespace tvm {
namespace codegen {

using ffi::ObjectPtrEqual;
using ffi::ObjectPtrHash;
using ffi::Optional;

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

  void Dispatch_(const TensorStoreNode *op) override;
  void Dispatch_(const BindNode *op) override;
  void EmitAllocation(const TensorVar &tensor);
  void Dispatch_(const SeqStmtNode *op) override;
  void Dispatch_(const ForNode *op) override;
  void Dispatch_(const WhileNode *op) override;
  void Dispatch_(const RegionStmtNode *op) override;
  void Dispatch_(const IfNode *op) override;
  void Dispatch_(const EvaluateNode *op) override;

  void Dispatch_(const BroadcastNode *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const RampNode *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const TensorLoadNode *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const LetNode *op,
                 std::ostream &os) override;                     // NOLINT(*)
  void Dispatch_(const CastNode *op, std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const FloatImmNode *op,
                 std::ostream &os) override;                     // NOLINT(*)
  void Dispatch_(const CallNode *op, std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const NotNode *op, std::ostream &os) override;  // NOLINT(*)
  void Dispatch_(const SelectNode *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const MinNode *op,
                 std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const MaxNode *op,
                 std::ostream &os) override;                    // NOLINT(*)
  void Dispatch_(const ModNode *op, std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const DivNode *op, std::ostream &os) override; // NOLINT(*)
  void Dispatch_(const ShuffleNode *op,
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
  bool NeedsCastptr_(const VarNode *buffer_var, PrimType elem_dtype) const;
  std::string ScalarPointerBase_(const VarNode *buffer_var, PrimType elem_dtype,
                                 const std::string &scope);
  std::string GetAccessPtrExpr_(const CallNode *op);
  std::string GetPointerExpr(const VarNode *buffer_var, PrimType elem_dtype,
                             const Expr &index);
  std::string GetPointerExpr(const VarNode *buffer, const Expr &index);
  std::string GetVectorLocalRef(const VarNode *buffer_var, const Expr &index,
                                const std::string &context);
  std::string GetMutableVectorRef(const Expr &address,
                                  const std::string &op_name);
  std::string PrintCondition(const Expr &condition);
  std::string GetAddressOfExpr_(const CallNode *op);
  std::string GetAscendCopyGmUbExpr_(const CallNode *op);
  std::string GetAscendCopyUbGmExpr_(const CallNode *op);
  void RecordAscendCopyPadValue_(const CallNode *op);
  void EmitAscendSetAtomic(const CallNode *op);
  void EmitCopyPadMerge_(int merged_value_id, PrimType expected_dtype);
  std::string GetCopyPadValueExpr_(const Expr &value);
  void GetCopyEndpoint_(const Expr &expr, const char *context,
                        const VarNode **buffer_var, PrimExpr *index,
                        PrimType *dtype, std::string *scope) const;
  void ValidateUBCopyLayout_(const Expr &index, PrimType dtype,
                             const Expr &burst_num, const Expr &burst_len,
                             const Expr &ub_stride, const Expr &left_padding,
                             const Expr &right_padding, bool uses_padding,
                             const char *context) const;
  std::string EmitAllReduceExpr_(const std::string &func_name,
                                 const CallNode *op);
  void EmitInlineSimtVF(const RegionStmtNode *op, int64_t thread_x,
                        int64_t thread_y, int64_t thread_z);
  void ExtractSimtThreadExtents(const RegionStmtNode *op, int64_t *thread_x,
                                int64_t *thread_y, int64_t *thread_z) const;
  std::string ScalarLoad(const VarNode *buffer, const Expr &index);
  std::string LocalVarStoreValue(const Expr &value, PrimType dtype);
  void EmitScalarStore(const VarNode *buffer, const std::string &value,
                       const Expr &index);
  void EmitBufferAllocation(const TensorVar &buffer);
  void EmitLocalVarInitialization_(const TensorVar &buffer, std::string vid);
  void EmitScalarizedLoad(const TensorLoadNode *op, std::ostream &os);
  void EmitScalarizedStore(const TensorStoreNode *op);
  bool TryEmitRngBroadcastStore(const TensorStoreNode *op);
  std::string ScopeOfBuffer(const VarNode *buffer) const;
  void PrintBinaryExpr_(const std::string &opstr, PrimType dtype, Expr lhs,
                        Expr rhs,
                        std::ostream &os) override; // NOLINT(*)
  void ValidateVecMode_(const CallNode *op, size_t mode_idx,
                        const char *expected = "MODE_ZEROING");
  bool EmitSimdMergingCall_(const CallNode *op,
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
  // Outer local buffers referenced by a SIMT section. Var identity is the
  // ownership key because lowering can leave duplicate name hints behind.
  std::unordered_set<const VarNode *> persistent_buffer_vars_;
  std::string GetLocalPtrExpr(const Expr &expr, const std::string &space,
                              PrimType fallback_dtype);
  std::string GetE8M0ScalePtrExpr(const Expr &expr);
  std::string GetLocalByteAddrExpr(const Expr &index, PrimType elem_dtype,
                                   const std::string &context);
  std::string GetAccPtrExpr(const Expr &expr, PrimType dtype);
  const PTOGemmEmitContext &EnsureGemmHelper(const CallNode *op);
  const PTOGemmEmitContext &EnsureBlockscaledGemmHelper(const CallNode *op);
  void ValidateFractalAddressAlignment_(const Expr &index, PrimType dtype,
                                        const char *context) const;
  void EmitAscendCopyGmToCbuf(const CallNode *op);
  void EmitAscendFillL1(const CallNode *op);
  void EmitAscendLoadCbufToL0(const CallNode *op, bool is_ca);
  void EmitAscendLoadMxSf(const CallNode *op, bool is_ca);
  void EmitAscendCrossCoreFlag(const CallNode *op, bool is_set);
  void EmitAscendMad(const CallNode *op);
  void EmitAscendGemmL1(const CallNode *op);
  void EmitAscendBlockscaledGemmL1(const CallNode *op);
  void EmitAscendCopyMatrixCcToUb(const CallNode *op);
  void EmitAscendCopyMatrixCcToGm(const CallNode *op);
  void EmitAscendCopyUbufToCbuf(const CallNode *op);
  void EmitAscendNd2NzPostCopy(const CallNode *op);
  void EmitAscendNd2NzScatter(const CallNode *op);
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
  std::string LocalVarID(const VarNode *var);
  bool IsLocalVarBuffer(const VarNode *var) const;
  void EmitMixedEntrySnapshot(const VarNode *var);
  void RestoreMixedSectionVariables(const RegionStmtNode *section);
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
  Optional<PrimExpr> uniform_const_copy_pad_value_;
  // Cube-side allocations hold integer addresses despite their TIR pointer
  // annotations. Vector-side allocations and make_tensor binds hold pointers.
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> integer_address_vars_;
  std::unordered_set<const VarNode *> local_var_buffers_;
  std::unordered_map<Call, int64_t, ObjectPtrHash, ObjectPtrEqual>
      hf32_mode_by_gemm_;
  std::unordered_map<Call, bool, ObjectPtrHash, ObjectPtrEqual>
      n_direction_by_mad_;
  std::unordered_map<Var, std::string, ObjectPtrHash, ObjectPtrEqual>
      loop_condition_snapshot_ids_;
  // A physical Cube/Vector section owns its SSA definitions.  Keep a shadow
  // of each outer local.var captured by any section and restore it at every
  // section boundary so definitions cannot leak between sibling regions.
  std::unordered_set<const VarNode *> mixed_captured_local_vars_;
  std::unordered_map<const RegionStmtNode *,
                     std::unordered_set<const VarNode *>>
      mixed_external_vars_by_section_;
  std::unordered_map<const VarNode *, std::string> mixed_entry_snapshot_ids_;
  // Pair-producing simd ops (vintlv/vdintlv/vld2) bind a tuple value; each
  // unpack via pair_get picks element 0 (low) or 1 (high). Maps the bound
  // VarNode to its (low, high) SSA name pair.
  std::unordered_map<const VarNode *, std::pair<std::string, std::string>>
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
  bool IsInlineableInvariantSimdBind(const Expr &value) const;
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
  void EmitRngInit(const CallNode *op);
  std::string EmitRngRand(const CallNode *op);
  std::string EmitRngRandFloat(const CallNode *op);
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_PTO_CODEGEN_CODEGEN_PTO_H_
