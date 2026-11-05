/*!
 * \file metal/codegen/codegen_metal.h
 * \brief Generate Metal device code.
 */
#ifndef TILELANG_METAL_CODEGEN_CODEGEN_METAL_H_
#define TILELANG_METAL_CODEGEN_CODEGEN_METAL_H_

#include <tvm/codegen/codegen.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "backend/common/codegen/codegen_c_base.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangMetal final : public CodeGenTileLangCBase {
public:
  using CodeGenTileLangCBase::Dispatch_;
  using CodeGenTileLangCBase::PrintType;
  explicit CodeGenTileLangMetal(Target target);
  // override print thread tag.
  void PrintArgUnionDecl();
  void AddFunction(const GlobalVar &gvar, const tirx::Function &func) final;
  void InitFuncState(const tirx::Function &f) final;
  void PrintStorageScope(const runtime::MemoryScope &scope,
                         std::ostream &os) final;            // NOLINT(*)
  void PrintStorageSync(const CallObj *op) final;            // NOLINT(*)
  void PrintType(const PrimType &t, std::ostream &os) final; // NOLINT(*)
  void BindThreadIndex(const PrimVar &var,
                       const ffi::String &thread_tag) final; // NOLINT(*)
  void Dispatch_(const TensorLoadObj *op,
                 std::ostream &os) final;         // NOLINT(*)
  void Dispatch_(const TensorStoreObj *op) final; // NOLINT(*)
  // print load of single element
  void PrintVecElemLoad(const std::string &vec, const PrimType &t, int i,
                        std::ostream &os) final; // NOLINT(*)
  // print store of single element.
  void PrintVecElemStore(const std::string &vec, const PrimType &t, int i,
                         const std::string &value) final;
  // overload visitor
  void Dispatch_(const BindObj *op) final;       // NOLINT(*)
  void PrintTensorAllocation(const BindObj *op); // NOLINT(*)
  void Dispatch_(const RegionStmtObj *op) final; // NOLINT(*)
  void Dispatch_(const ForObj *op) final;        // NOLINT(*)
  void Dispatch_(const prim::SelectObj *op,
                 std::ostream &os) final; // NOLINT(*)
  void Dispatch_(const prim::BroadcastObj *op,
                 std::ostream &os) final;                         // NOLINT(*)
  void Dispatch_(const prim::AddObj *op, std::ostream &os) final; // NOLINT(*)
  void Dispatch_(const prim::RShiftObj *op,
                 std::ostream &os) final; // NOLINT(*)
  void Dispatch_(const prim::BitwiseAndObj *op,
                 std::ostream &os) final;                          // NOLINT(*)
  void Dispatch_(const prim::CastObj *op, std::ostream &os) final; // NOLINT(*)
  void Dispatch_(const CallObj *op, std::ostream &os) final;       // NOLINT(*)
  void Dispatch_(const FloatImmObj *op, std::ostream &os) final;   // NOLINT(*)

protected:
  void EmitPreamble_(std::ostream &os) const final;

private:
  std::string GetStorageScopeOf(const Expr &ptr_expr) const;
  std::string GetAddrSpaceOf(const Expr &ptr_expr) const;
  std::string GetPointeeTypeOf(const Expr &ptr_expr,
                               const std::string &fallback);
  bool IsThreadIdxXExpr(const Expr &expr) const;
  bool IsBlockIdxExpr(const Expr &expr, int dim) const;
  bool IsConstIntExpr(const Expr &expr, int64_t value) const;
  bool IsMlxPanelRemainderExpr(const Expr &expr) const;
  bool IsMlxPanelRowExpr(const Expr &expr) const;
  bool IsMlxLogicalBlockXExpr(const Expr &expr) const;
  bool IsMlxLogicalBlockYExpr(const Expr &expr) const;
  bool TryPrintMlxLogicalYAffineExpr(const Expr &expr, std::ostream &os);
  bool TryPrintMlxSwizzleExpr(const Expr &expr, std::ostream &os);
  bool TryPrintSimdgroupIndexExpr(const CallObj *op, std::ostream &os);
  void PrintSimdgroupIndexExpr(int64_t group_mask, int64_t group_shift,
                               std::ostream &os) const;
  void EnsureFragmentLaneVars();
  void EnsureCooperativeTensorBuffer(const Var &var);

  std::unordered_map<Var, std::string, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      simdgroup_dtype_;
  std::unordered_map<Var, std::string, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      cooperative_tensor_dtype_;
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      ct_c_inlined_;
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      ct_c_storage_elided_;
  ffi::Optional<Var> thread_idx_x_var_;
  ffi::Optional<Var> block_idx_x_var_;
  ffi::Optional<Var> block_idx_y_var_;
  int active_mlx_swizzle_panel_{0};
  int active_mlx_swizzle_log_{0};
  bool emitted_metal_simdgroup_id_{false};
  bool emitted_frag_lane_vars_{false};
  bool uses_cooperative_tensor_{false};
  bool needs_fragment_lane_vars_{false};
  int thread_index_bits_{32};
  int thread_work_dim_{0};
  Target target_;
};
} // namespace codegen
} // namespace tvm

#endif // TILELANG_METAL_CODEGEN_CODEGEN_METAL_H_
