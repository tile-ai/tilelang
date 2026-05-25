/*!
 * \file target/codegen.h
 * \brief Utility to generate code
 */
#ifndef TVM_TL_TARGET_CODEGEN_CUDA_H_
#define TVM_TL_TARGET_CODEGEN_CUDA_H_

#include <optional>
#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "target/source/codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangCUDA final : public CodeGenC {
public:
  CodeGenTileLangCUDA();
  std::string Finish();
  // override behavior
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const PrimFunc &f);
  void VisitStmt_(const ForNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void PrintStorageScope(const std::string &scope,
                         std::ostream &os) final; // NOLINT(*)
  void PrintVecBinaryOp(const std::string &op, DataType t, PrimExpr lhs,
                        PrimExpr rhs,
                        std::ostream &os) final;      // NOLINT(*)
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void PrintVecElemLoad(const std::string &vec, DataType t, int i,
                        std::ostream &os) final; // NOLINT(*)
  void PrintVecElemStore(const std::string &vec, DataType t, int i,
                         const std::string &value) final;
  std::string GetVecLoad(DataType t, const BufferNode *buffer,
                         PrimExpr base) final;
  void PrintVecStore(const BufferNode *buffer, DataType t, PrimExpr base,
                     const std::string &value) final;
  void BindThreadIndex(const IterVar &iv) final; // NOLINT(*)
  void PrintVecElemLoadExpr(DataType t, int i, const std::string &value,
                            std::ostream &os) final;
  std::string CastFromTo(std::string value, DataType from,
                         DataType target) final;
  // overload visitor
  void VisitExpr_(const RampNode *op, std::ostream &os) final;      // NOLINT(*)
  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void VisitExpr_(const CastNode *op, std::ostream &os) final;
  void VisitExpr_(const ShuffleNode *op, std::ostream &os) final;
  void VisitExpr_(const MinNode *op, std::ostream &os) final;
  void VisitExpr_(const MaxNode *op, std::ostream &os) final;
  void VisitStmt_(const EvaluateNode *op) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitStmt_(const IfThenElseNode *op) final;
  void VisitExpr_(const BufferLoadNode *op, std::ostream &os) final;
  void VisitStmt_(const BufferStoreNode *op) final;

  // Override this as a work around for __grid_constant__ parameter
  void AddFunction(const GlobalVar &gvar, const PrimFunc &f);
  void PrintFunctionSignature(const ffi::String &function_name,
                              const PrimFunc &func, std::ostream &os);

protected:
  void ReserveKeywordsAsUnique_();
  virtual std::string GetBufferRef(DataType t, const BufferNode *buffer,
                                   PrimExpr index) final;
  void PrintCallExtern(Type ret_type, ffi::String global_symbol,
                       const ffi::Array<PrimExpr> &args, bool skip_first_arg,
                       std::ostream &os) final; // NOLINT(*)

private:
  // Handle volatile loads
  void HandleVolatileLoads(const std::string &value, const BufferLoadNode *op,
                           std::ostream &os) final;
  bool HandleLateIntrinsicCall(const CallNode *op, std::ostream &os);
  void FlushPendingTmemAllocs();

  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final { return false; }

  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangCUDA *p);

  // Whether global barrier is needed.
  bool need_global_barrier_{false};
  // Global barrier state
  std::string vid_global_barrier_state_;
  // Global barrier expected node.
  std::string vid_global_barrier_expect_;
  // Global curand state
  std::string curand_random_generator_state;
  std::string curand_random_generator_state_type;

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
  // whether need mma.h
  bool need_mma_h_{false};
  // whether need tl mma instruction header
  bool need_mma_instruction_h_{false};
  // whether need tl wgmma instruction header
  bool need_wgmma_instruction_h_{false};
  // whether need tl tcgen05mma instruction header
  bool need_tcgen05mma_instruction_h_{false};
  // whether need tl mma_sm70 instruction header
  bool need_mma_sm70_instruction_h_{false};
  // whether need tcgen_05 common header
  bool need_tcgen05_common_h_{false};
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
  const std::string barrier_name_ = "barrier";
  // The size of the barrier array in shared memory
  int barrier_count_ = -1;
  // The name of the mbarrier array in shared memory
  // The same as injected_mbarrier_name_ in transform/common/mbarrier.h
  const std::string mbarrier_name_ = "mbarrier";
  // The type name of the mbarrier array
  const std::string mbarrier_dtype_ = "Barrier";
  // Buffered TMEM allocations for sorted emission (avo layout parity).
  // Each entry: {sort_key (buffer arg text), full call string}.
  std::vector<std::pair<std::string, std::string>> pending_tmem_allocs_;
  // The alignment of the barrier array in shared memory
  // Set to 16 to maintain minimum alignment requirements for async bulk copy
  const int barrier_alignment_bytes_ = 16;

  std::unordered_map<const VarNode *, std::string> fragment_shapes;
  std::unordered_map<const VarNode *, std::string> fragment_layouts;
  std::unordered_map<const VarNode *, IntImm> unroll_factor;
  std::optional<std::tuple<int64_t, int64_t, int64_t>> cluster_dims;
  // Map from VarNode to packed buffer variable name for fp4 packed storage
  std::unordered_map<const VarNode *, std::string> fp4_packed_buffers_;
  // Stream for outlined __noinline__ device functions (emitted before kernel)
  std::ostringstream outlined_fns_stream_;
  // Track shared-memory barrier/TMEM declarations for device function params
  // Each entry: (var_name, type_string) e.g. ("mbar_s0", "Barrier*")
  std::vector<std::pair<std::string, std::string>> shared_state_vars_;
  // Whether we're currently inside an outlined device function emission
  bool in_outlined_fn_{false};
  // Counter for unique device function names
  int outlined_fn_count_{0};

  // --- Device function outlining state ---
  // The threadIdx.x Var node (set in BindThreadIndex)
  const VarNode *thread_x_var_{nullptr};
  // Total dynamic shared memory size (set when visiting shared.dyn AllocateNode)
  int64_t dyn_shmem_size_{0};
  // Whether outlining is enabled (read from pass config)
  bool outline_warp_spec_enabled_{false};
  // Innermost for-loop variable name (for passing to device functions)
  std::string current_loop_var_name_;

  struct LocalAllocInfo {
    std::string var_name;
    DataType dtype;
    int64_t size;
    bool outline_persistent{false};
    // Buffer variable handle so we can check which branches touch this alloc
    // during outlining. Without this we conservatively re-declare every kernel-
    // scope local in every outlined fn, forcing ptxas to keep dead `= {}`
    // initializers alive and inflating register pressure (e.g. the math0
    // outlined fn would declare S1_reg[128] + P1_cast[128] + O*_reg[128]
    // even though it never touches them, costing >300 unused regs/thread).
    const VarNode *buffer_var{nullptr};
  };
  std::vector<LocalAllocInfo> local_allocs_;
  // Barrier variable names declared as __shared__ (e.g., "mbar_s0")
  std::vector<std::string> barrier_var_names_;
  // TMEM handle variable names declared as __shared__ (e.g., "S0_tmem")
  std::vector<std::string> tmem_var_names_;
  // Kernel function parameters (name, type_string) for device fn forwarding
  struct KernelParamInfo {
    std::string var_name;
    std::string type_str;  // e.g. "const CUtensorMap"
    bool is_grid_constant{false};
  };
  std::vector<KernelParamInfo> kernel_param_infos_;
  std::unordered_set<const VarNode *> outline_persistent_vars_;

  // Helper: check if condition involves threadIdx.x comparison
  bool IsThreadXComparison(const PrimExpr &cond) const;
  // Helper: emit one outlined device function, returns the function name
  std::string EmitOutlinedDeviceFunction(const Stmt &body);
  // Helper: flatten nested if-else on threadIdx.x into branches
  struct WarpBranch {
    PrimExpr condition;
    Stmt body;
  };
  void FlattenWarpBranches(const IfThenElseNode *node,
                           std::vector<WarpBranch> *branches);
  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangCUDA *p);
  void PrintWmmaScope(const std::string &scope, DataType t,
                      const VarNode *variable, std::ostream &os);
  int32_t GetWmmaFragmentSize(const std::string &scope, const VarNode *variable,
                              int32_t size);

  std::vector<std::string> eviction_policy_names_ = {
      "EVICT_NORMAL", "EVICT_FIRST", "EVICT_LAST"};
  std::unordered_set<std::string> bf16_supported_ops_ = {
      "bf1622float2", "bf1622int16", "float22bf162", "bf162bf162"};
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_CUDA_H_
