/*!
 * \file ascend/codegen/codegen_pto.h
 * \brief Utility to generate PTO Python source.
 */
#ifndef TVM_TL_PTO_CODEGEN_CODEGEN_PTO_H_
#define TVM_TL_PTO_CODEGEN_CODEGEN_PTO_H_

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cuda/codegen/codegen_py.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangPTO final : public CodeGenTileLangPY {
public:
  CodeGenTileLangPTO();

  void AddFunction(const GlobalVar &gvar, const PrimFunc &func) override;
  std::string Finish() override;

protected:
  void PrintFuncDecorator_(std::ostream &os) override; // NOLINT(*)
  void PrintFunctionSignature_(const ffi::String &function_name,
                               const PrimFunc &func,
                               std::ostream &os) override; // NOLINT(*)

  void VisitStmt_(const DeclBufferNode *op) override;
  void VisitStmt_(const BufferStoreNode *op) override;
  void VisitStmt_(const BindNode *op) override;
  void VisitStmt_(const AllocBufferNode *op) override;
  void VisitStmt_(const AttrStmtNode *op) override;
  void VisitStmt_(const ForNode *op) override;
  void VisitStmt_(const WhileNode *op) override;
  void VisitStmt_(const SBlockNode *op) override;
  void VisitStmt_(const IfThenElseNode *op) override;
  void VisitStmt_(const EvaluateNode *op) override;

  void VisitExpr_(const BroadcastNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const BufferLoadNode *op,
                  std::ostream &os) override;                     // NOLINT(*)
  void VisitExpr_(const CastNode *op, std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const CallNode *op, std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op,
                  std::ostream &os) override;                    // NOLINT(*)
  void VisitExpr_(const MinNode *op, std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const MaxNode *op, std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const AndNode *op, std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const OrNode *op, std::ostream &os) override;  // NOLINT(*)
  void VisitExpr_(const NotNode *op, std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const SelectNode *op,
                  std::ostream &os) override;                    // NOLINT(*)
  void VisitExpr_(const LetNode *op, std::ostream &os) override; // NOLINT(*)

private:
  struct FragmentInfo {
    int lanes{0};
    DataType dtype;
  };

  struct PTOGemmEmitContext {
    bool initialized{false};
    int64_t tile_m{0};
    int64_t tile_n{0};
    int64_t tile_k{0};
    int64_t base_k{0};
    DataType input_dtype;
    DataType accum_dtype;
    std::string helper_name;
    std::string a_l0_name;
    std::string b_l0_name;
  };

  struct PTOBlockscaledGemmEmitContext {
    bool initialized{false};
    int64_t tile_m{0};
    int64_t tile_n{0};
    int64_t tile_k{0};
    int64_t base_k{0};
    int64_t sf_nz_stride{0};
    DataType input_dtype;
    DataType accum_dtype;
    std::string helper_name;
    std::string a_l0_name;
    std::string b_l0_name;
  };

  std::string PtoScalarType(DataType t) const;
  std::string PtoPtrType(DataType t, const std::string &space) const;
  bool NeedsPtoCastptr_(const VarNode *buffer_var, DataType elem_dtype) const;
  std::string PtoScalarPointerBase_(const VarNode *buffer_var,
                                    DataType elem_dtype,
                                    const std::string &scope);
  std::string GetAccessPtrExpr_(const CallNode *op);
  std::string GetPtoPointerExpr(const VarNode *buffer_var, DataType elem_dtype,
                                const PrimExpr &index);
  std::string GetPtoPointerExpr(const BufferNode *buffer,
                                const PrimExpr &index);
  std::string GetAddressOfExpr_(const CallNode *op);
  std::string GetPTOCopyPadValueExpr_(const PrimExpr &value);
  void GetPTOCopyEndpoint_(const PrimExpr &expr, const char *context,
                           const VarNode **buffer_var, PrimExpr *index,
                           DataType *dtype, std::string *scope) const;
  void ValidatePTOUBCopyLayout_(const PrimExpr &index, DataType dtype,
                                const PrimExpr &burst_num,
                                const PrimExpr &burst_len,
                                const PrimExpr &ub_stride,
                                const PrimExpr &left_padding,
                                const PrimExpr &right_padding,
                                bool uses_padding, const char *context) const;
  std::string GetAscendCopyGmUbExpr_(const CallNode *op);
  std::string GetAscendCopyUbGmExpr_(const CallNode *op);
  std::string EmitPTOAllReduceExpr_(const std::string &func_name,
                                    const CallNode *op);
  void EmitInlineSimtVF(const SBlockNode *op, int64_t thread_x,
                        int64_t thread_y, int64_t thread_z);
  void ExtractSimtThreadExtents(const SBlockNode *op, int64_t *thread_x,
                                int64_t *thread_y, int64_t *thread_z) const;
  std::string PtoScalarLoad(const BufferNode *buffer, const PrimExpr &index);
  void EmitPtoScalarStore(const BufferNode *buffer, const std::string &value,
                          const PrimExpr &index);
  void EmitPtoBufferAllocation(const Buffer &buffer);
  void EmitScalarizedLoad(const BufferLoadNode *op, std::ostream &os);
  void EmitScalarizedStore(const BufferStoreNode *op);
  std::string ScopeOfBuffer(const BufferNode *buffer) const;
  std::optional<std::string>
  PtoSpaceForStorageScope(const std::string &scope) const;
  void PrintPtoSelectValue_(const PrimExpr &value, DataType dtype,
                            std::ostream &os);
  void PrintPtoSelect_(const PrimExpr &condition, const PrimExpr &true_value,
                       const PrimExpr &false_value, DataType dtype,
                       std::ostream &os);
  void PrintPtoIfThenElse_(const CallNode *op, std::ostream &os);
  void PrintPtoLogicalNot_(const PrimExpr &value, std::ostream &os);
  void PrintBinaryExpr_(const std::string &opstr, DataType dtype, PrimExpr lhs,
                        PrimExpr rhs,
                        std::ostream &os) override; // NOLINT(*)
  void PrintPtoFloatMinMax_(const char *op_name, DataType dtype, PrimExpr lhs,
                            PrimExpr rhs, std::ostream &os); // NOLINT(*)
  // Table-driven unary math mapping (sqrt/rsqrt/exp/log, extern C names and
  // tirx intrinsic names). Returns false when `name` is not covered.
  bool TryEmitPtoUnaryMath_(const std::string &name, const PrimExpr &arg,
                            std::ostream &os); // NOLINT(*)
  // tl.rng_init: bind a fresh PhiloxRNG helper in the SIMT body.
  void EmitRngInit(const CallNode *op);
  // tl.rng_rand / tl.rng_rand_float: return the inline draw expression.
  std::string EmitRngDrawExpr(const CallNode *op);
  // Broadcast(rng_expression, 2) store: evaluate the complete scalar
  // expression twice and emit two ordered scalar stores. Returns true when
  // `op` was handled.
  bool TryEmitRngBroadcastStore(const BufferStoreNode *op);

  std::string current_function_name_;
  // `tl.simd.vdiv` defaults to precise fp32 division when fast math is off.
  // PTOAS exposes that precision only for tile-level `tdiv`, not for the
  // vector-level `pto.vdiv` used by this backend.
  bool enable_fast_math_{false};
  bool inside_simtvf_body_{false};
  // Nonzero while emitting a runtime loop or a dynamic branch. PhiloxRNG owns
  // trace-time SSA state, so initializing or drawing in device-side control
  // flow would silently produce incorrect runtime state transitions.
  int inside_dynamic_control_flow_{0};
  // Active Philox helper variable emitted by tl.rng_init. It is reset at each
  // PrimFunc and SIMT VF boundary because the helper owns section-local SSA
  // state and cannot be shared by sibling sections.
  std::string rng_state_var_;
  bool uses_rng_{false};
  // Outer local buffers referenced by a SIMT section. Var identity is the
  // ownership key because lowering can leave duplicate name hints behind.
  std::unordered_set<const VarNode *> persistent_buffer_vars_;
  std::string GetPtoLocalPtrExpr(const PrimExpr &expr, const std::string &space,
                                 DataType fallback_dtype);
  std::string GetPtoLocalByteAddrExpr(const PrimExpr &index,
                                      DataType elem_dtype,
                                      const std::string &context);
  std::string GetPtoE8M0ScalePtrExpr(const PrimExpr &expr);
  std::string GetPtoAccPtrExpr(const PrimExpr &expr, DataType dtype);
  std::string GetPtoL0APtrExpr(const PrimExpr &expr, DataType dtype);
  std::string GetPtoL0BPtrExpr(const PrimExpr &expr, DataType dtype);
  std::string GetPtoMatPtrExpr(const PrimExpr &expr, DataType dtype);
  std::string GetPtoUbPtrExpr(const PrimExpr &expr, DataType dtype);
  void EnsurePTOGemmHelper(const CallNode *op);
  void EnsurePTOBlockscaledGemmHelper(const CallNode *op);
  void EmitUnitFlagDispatch(
      const PrimExpr &unit_flag_expr,
      const std::function<std::string(int64_t)> &map_unit_flag,
      const std::function<void(const std::string &)> &emit_operation);
  void EmitAscendCopyGmToCbuf(const CallNode *op);
  void EmitAscendFillL1(const CallNode *op);
  void EmitAscendLoadCbufToL0(const CallNode *op, bool is_l0a);
  void EmitAscendGemmL1(const CallNode *op);
  void EmitAscendMad(const CallNode *op);
  void EmitAscendBlockscaledGemmL1(const CallNode *op);
  void EmitAscendMadMx(const CallNode *op);
  void EmitAscendCopyMatrixCcToUb(const CallNode *op);
  void EmitAscendCopyMatrixCcToGm(const CallNode *op);
  void EmitPTOGemmRun(const std::string &a_mat, const std::string &b_mat,
                      const std::string &acc, const std::string &clear_accum,
                      const std::string &unit_flag_ctrl, int64_t hf32_mode);
  void EmitPTOBlockscaledGemmRun(
      const std::string &a_mat, const std::string &b_mat,
      const std::string &sfa_e8m0_mat, const std::string &sfb_e8m0_mat,
      const std::string &acc, const std::string &sf_k_offset,
      const std::string &clear_accum, const std::string &unit_flag_ctrl);
  std::string PrintVmiAnnotationValue(const std::string &key,
                                      const ObjectRef &value);
  void PrintPtoVmiCall_(const CallNode *op, std::ostream &os);
  std::string LocalVarID(const VarNode *var);
  bool IsLocalVarBuffer(const VarNode *var) const;
  bool IsVmiLocalRegisterBuffer(const BufferNode *buffer) const;
  // Vector local.var buffers allocated before ``for_stmt`` and stored in its
  // body. These must be emitted as PTODSL loop-carried state for ``T.serial``.
  std::vector<const VarNode *>
  CollectLoopCarriedLocalVars(const Stmt &body) const;
  void CheckVmiLocalRegisterIndex(const BufferNode *buffer,
                                  const PrimExpr &index) const;
  bool HasAscendGemm(const PrimFunc &func) const;
  bool HasAscendGemmL1(const PrimFunc &func) const;

  bool current_function_has_gemm_{false};
  bool current_function_has_mixed_sections_{false};
  bool has_gemm_l1_{false};
  std::unordered_map<const VarNode *, FragmentInfo> fragment_info_;
  std::unordered_set<const VarNode *> local_var_buffers_;
  std::unordered_map<Call, int64_t, ObjectPtrHash, ObjectPtrEqual>
      hf32_mode_by_cube_call_;
  // Pad bindings are resolved once before printing. The maps are keyed by the
  // exact TIR Call nodes so codegen does not infer state from print order.
  std::unordered_map<Call, int64_t, ObjectPtrHash, ObjectPtrEqual>
      pad_binding_by_copy_;
  std::unordered_map<Call, int64_t, ObjectPtrHash, ObjectPtrEqual>
      pad_binding_by_setter_;
  std::unordered_map<int64_t, DataType> pad_binding_dtype_by_id_;
  PTOGemmEmitContext gemm_emit_ctx_;
  PTOBlockscaledGemmEmitContext blockscaled_gemm_emit_ctx_;
  std::pair<std::string, std::string>
  ParseHardEventPair(const std::string &hard_event) const;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_PTO_CODEGEN_CODEGEN_PTO_H_
