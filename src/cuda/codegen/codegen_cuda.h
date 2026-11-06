/*!
 * \file target/codegen.h
 * \brief Utility to generate code
 */
#ifndef TVM_TL_TARGET_CODEGEN_CUDA_H_
#define TVM_TL_TARGET_CODEGEN_CUDA_H_

#include "backend/common/codegen/codegen_ir_utils.h"
#include "support/check.h"
#include <optional>
#include <tvm/codegen/codegen.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "backend/common/codegen/codegen_c_base.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangCUDA final : public CodeGenTileLangCBase {
public:
  using CodeGenTileLangCBase::Dispatch_;
  using CodeGenTileLangCBase::PrintType;
  CodeGenTileLangCUDA();
  // override behavior
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const tirx::Function &f, std::ostream &os) override;
  void Dispatch_(const ForObj *op) final;
  void PrintStorageSync(const CallObj *op) final;
  void PrintStorageScope(const runtime::MemoryScope &scope,
                         std::ostream &os) final; // NOLINT(*)
  void PrintVecBinaryOp(const std::string &op, const PrimType &t, PrimExpr lhs,
                        PrimExpr rhs,
                        std::ostream &os) final;             // NOLINT(*)
  void PrintType(const PrimType &t, std::ostream &os) final; // NOLINT(*)
  void PrintType(const Type &t, std::ostream &os) final;     // NOLINT(*)
  void PrintVecConstructor(const PrimType &t,
                           std::ostream &os) final; // NOLINT(*)
  void PrintVecElemLoad(const std::string &vec, const PrimType &t, int i,
                        std::ostream &os) final; // NOLINT(*)
  void PrintVecElemStore(const std::string &vec, const PrimType &t, int i,
                         const std::string &value) final;
  std::string GetVecLoad(const PrimType &t, const VarObj *buffer,
                         PrimExpr base) final;
  void PrintVecStore(const VarObj *buffer, const PrimType &t, PrimExpr base,
                     const std::string &value) final;
  void BindThreadIndex(const PrimVar &var,
                       const ffi::String &thread_tag) final; // NOLINT(*)
  void PrintVecElemLoadExpr(const PrimType &t, int i, const std::string &value,
                            std::ostream &os) final;
  std::string CastFromTo(std::string value, const PrimType &from,
                         const PrimType &target) final;
  // overload visitor
  void Dispatch_(const prim::RampObj *op, std::ostream &os) final; // NOLINT(*)
  void Dispatch_(const prim::BroadcastObj *op,
                 std::ostream &os) final; // NOLINT(*)
  void Dispatch_(const FloatImmObj *op, std::ostream &os) final;
  void Dispatch_(const CallObj *op, std::ostream &os) final;
  void Dispatch_(const prim::CastObj *op, std::ostream &os) final;
  void Dispatch_(const prim::ShuffleObj *op, std::ostream &os) final;
  void Dispatch_(const prim::MinObj *op, std::ostream &os) final;
  void Dispatch_(const prim::MaxObj *op, std::ostream &os) final;
  void Dispatch_(const prim::NotObj *op, std::ostream &os) final;
  void Dispatch_(const prim::BitwiseNotObj *op, std::ostream &os) final;
  void Dispatch_(const AssertStmtObj *op) final;
  void Dispatch_(const EvaluateObj *op) final;
  void Dispatch_(const ReturnObj *op) final;
  void Dispatch_(const BindObj *op) final;
  void Dispatch_(const RegionStmtObj *op) final;
  void Dispatch_(const TensorLoadObj *op, std::ostream &os) final;
  void Dispatch_(const TensorStoreObj *op) final;
  void Dispatch_(const prim::SelectObj *op, std::ostream &os) final;

  // Override this as a work around for __grid_constant__ parameter
  void AddFunction(const GlobalVar &gvar, const tirx::Function &f);
  void PrintFunctionSignature(const ffi::String &function_name,
                              const tirx::Function &func, std::ostream &os);

protected:
  void EmitPreamble_(std::ostream &os) const final;
  void ReserveKeywordsAsUnique_();
  virtual std::string GetBufferRef(const PrimType &t, const VarObj *buffer,
                                   PrimExpr index) final;
  void PrintCallExtern(Type ret_type, ffi::String global_symbol,
                       const ffi::Array<Expr> &args, bool skip_first_arg,
                       std::ostream &os) final; // NOLINT(*)

private:
  void PrintCast(const PrimExpr &value, const PrimType &target_type,
                 const std::string &rounding, bool saturate,
                 ffi::Optional<PrimExpr> random_bits, std::ostream &os);
  // Handle volatile loads
  void HandleVolatileLoads(const std::string &value, const TensorLoadObj *op,
                           std::ostream &os) final;
  bool HandleLateIntrinsicCall(const CallObj *op, std::ostream &os);
  bool TryEmitPackedBinaryOp_(const std::string &op, PrimType result_type,
                              const PrimExpr &lhs, const PrimExpr &rhs,
                              std::ostream &os);
  bool TryEmitPackedComparison_(const std::string &op, PrimType result_type,
                                PrimType lhs_type, PrimType rhs_type,
                                const std::string &lhs, const std::string &rhs,
                                const std::string &result);
  void EmitScalarizedBinaryOp_(const std::string &op, PrimType result_type,
                               PrimType lhs_type, PrimType rhs_type,
                               const std::string &lhs, const std::string &rhs,
                               const std::string &result);
  bool TryEmitFastMathCall_(const Call &call, std::ostream &os);
  bool TryEmitIEEEMathCall_(const Call &call, std::ostream &os);
  bool TryEmitPackedMathCall_(const Call &call, std::ostream &os);
  // Emit a vector op of dtype t as lanes/2 packed x2 calls (tl::mul2,
  // tl::fma2, ...). All args must already have dtype t; requires
  // CanEmitPackedX2Math(t).
  void EmitPackedX2Call(const std::string &tl_func, const PrimType &t,
                        const std::vector<PrimExpr> &args, std::ostream &os);
  // Emit a vector op of dtype t as one scalar call per lane.
  void EmitPerLaneScalarCall(const std::string &func_name, const PrimType &t,
                             const std::vector<PrimExpr> &args,
                             std::ostream &os);

  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final { return false; }

  friend void PrintConst(const FloatImmObj *op, std::ostream &os,
                         CodeGenTileLangCUDA *p);

  // Global curand state
  std::string curand_random_generator_state;
  std::string curand_random_generator_state_type;
  // Function-scope curand state declarations: tl.rng_init call node -> var
  // name. States are declared at kernel top (see AddFunction) because
  // sync-insertion passes may split the block containing rng_init across
  // __syncthreads(), which would put a call-site declaration out of scope
  // for later rng_rand / rng_rand_float uses.
  std::unordered_map<const CallObj *, std::string> rng_state_name_map_;

  // whether enable fp16
  bool enable_fp16_{false};
  // whether enable bf16
  bool enable_bf16_{false};
  // whether enable fp8
  bool enable_fp8_{false};
  // whether enable fp6
  bool enable_fp6_{false};
  // whether enable fp4
  bool enable_fp4_{false};
  // whether enable int8
  bool enable_int8_{false};
  // whether enable sparse gemm
  bool enable_sparse_gemm_{false};
  // whether enable warp shuffle intrinsics
  bool enable_warp_shuffle_{false};
  // whether need math_constants.h
  bool need_math_constants_h_{false};
  // whether need tl CUDA math helpers
  bool need_math_h_{false};
  // whether need tl copy helpers
  bool need_copy_h_{false};
  // whether need tl SM90 TMA copy helpers
  bool need_copy_sm90_h_{false};
  // whether need tl SM100 TMA/tensor-memory copy helpers
  bool need_copy_sm100_h_{false};
  // whether need guarded ordinary vector memory helpers
  // whether need tl mbarrier helpers
  bool need_barrier_h_{false};
  // whether need mma.h
  bool need_mma_h_{false};
  // whether need tl mma instruction header
  bool need_mma_instruction_h_{false};
  // whether need tl block-scaled MMA instruction header
  bool need_mma_block_scale_instruction_h_{false};
  // whether need tl wgmma instruction header
  bool need_wgmma_instruction_h_{false};
  // whether need tl tcgen05mma instruction header
  bool need_tcgen05mma_instruction_h_{false};
  // whether need tl mma_sm70 instruction header
  bool need_mma_sm70_instruction_h_{false};
  // whether need tl mma_sp instruction header
  bool need_mma_sp_instruction_h_{false};
  // whether need tl wgmma_sp instruction header
  bool need_wgmma_sp_instruction_h_{false};
  // whether need tcgen_05 common header
  bool need_tcgen05_common_h_{false};
  // whether need tl runtime intrinsic helpers
  bool need_intrin_h_{false};
  // whether need tl atomic helpers
  bool need_atomic_h_{false};
  // whether need cast_smem_ptr_to_int helper function
  bool need_cast_smem_ptr_to_int_{false};
  // whether need cooperative_groups.h
  bool need_cooperative_groups_{false};
  // whether need curand_kernel.h
  bool need_curand_kernel_h_{false};
  // whether need cluster.h
  bool need_cluster_h_{false};
  // Op attribute map
  OpAttrMap<bool> op_need_warp_shuffle_ =
      Op::GetAttrMap<bool>("cuda.need_warp_shuffle");

  // The name of the barrier array in shared memory
  std::string legacy_barrier_name_;
  std::string BarrierReference(const Expr &barrier);
  // The size of the barrier array in shared memory
  int barrier_count_ = -1;
  // The name of the mbarrier array in shared memory
  // The same as injected_mbarrier_name_ in transform/common/mbarrier.h
  // The type name of the mbarrier array
  const std::string mbarrier_dtype_ = "Barrier";
  // The alignment of the barrier array in shared memory
  // Set to 16 to maintain minimum alignment requirements for async bulk copy
  const int barrier_alignment_bytes_ = 16;

  std::unordered_map<const VarObj *, std::string> fragment_shapes;
  std::unordered_map<const VarObj *, std::string> fragment_layouts;
  std::unordered_map<const VarObj *, IntImm> unroll_factor;
  std::optional<std::tuple<int64_t, int64_t, int64_t>> cluster_dims;
  // Physical backing variable name for each packed local FP4 buffer.
  std::unordered_map<Var, std::string, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      fp4_packed_buffers_;
  friend void PrintConst(const FloatImmObj *op, std::ostream &os,
                         CodeGenTileLangCUDA *p);
  void PrintWmmaScope(const std::string &scope, const PrimType &t,
                      const VarObj *variable, std::ostream &os);
  int32_t GetWmmaFragmentSize(const std::string &scope, const VarObj *variable,
                              int32_t size);

  std::vector<std::string> eviction_policy_names_ = {
      "EVICT_NORMAL", "EVICT_FIRST", "EVICT_LAST"};
  std::unordered_set<std::string> bf16_supported_ops_ = {
      "bf1622float2", "bf1622int16", "float22bf162", "bf162bf162"};
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_CUDA_H_
