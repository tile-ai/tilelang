/*!
 * \file target/codegen.h
 * \brief Utility to generate code
 */
#ifndef TVM_TL_TARGET_CODEGEN_CUDA_H_
#define TVM_TL_TARGET_CODEGEN_CUDA_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>
#include <unordered_map>

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
  void VisitStmt_(const EvaluateNode *op) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitExpr_(const BufferLoadNode *op, std::ostream &os) final;

  // Override this as a work around for __grid_constant__ parameter
  void AddFunction(const GlobalVar &gvar, const PrimFunc &f);
  void PrintFunctionSignature(const String &function_name, const PrimFunc &func,
                              std::ostream &os);

protected:
  virtual std::string GetBufferRef(DataType t, const BufferNode *buffer,
                                   PrimExpr index) final;
  void PrintCallExtern(Type ret_type, String global_symbol,
                       const Array<PrimExpr> &args, bool skip_first_arg,
                       std::ostream &os) final; // NOLINT(*)

private:
  // Handle volatile loads
  void HandleVolatileLoads(const std::string &value, const BufferLoadNode *op,
                           std::ostream &os) final;

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
  /**
 * CUDA codegen helper fields and utilities used internally by CodeGenTileLangCUDA.
 *
 * Holds flags that indicate required CUDA headers or helper functions (e.g., MMA,
 * cooperative groups, smem pointer casting), an operator attribute map for
 * warp-shuffle requirements, identifiers and alignment for shared-memory barrier
 * arrays, and metadata for WMMA fragments and eviction/bfloat16 support.
 *
 * These members are internal implementation details used when emitting CUDA
 * code: `barrier_name_`, `barrier_count_`, `mbarrier_name_`, `mbarrier_dtype_`,
 * and `barrier_alignment_bytes_` control naming, sizing, typing, and alignment
 * of per-kernel shared-memory barrier storage; `fragment_shapes` and
 * `fragment_layouts` record WMMA fragment shape/layout strings keyed by
 * fragment variables; `eviction_policy_names_` and `bf16_supported_ops_` are
 * lookup tables used during code emission. The boolean flags (e.g.
 * `need_mma_h_`, `need_cooperative_groups_`) indicate required auxiliary
 * headers or helpers and are consulted when finalizing emitted code.
 *
 * Related helper methods:
 * - PrintWmmaScope(...) : emit WMMA fragment scope declaration for a variable.
 * - GetWmmaFragmentSize(...) : compute the element count for a WMMA fragment.
 */
  bool need_mma_h_{false};
  /** The name of the secondary barrier array placed in shared memory. */
 
/** The element type name used for the secondary barrier array (e.g., a POD
 * struct or typedef exposed to emitted CUDA code). */

/**
 * Emit the WMMA fragment scope qualifier for a variable.
 *
 * @param scope The WMMA scope string (e.g., "matrix_a", "matrix_b", "accumulator").
 * @param t The element data type of the fragment.
 * @param variable The TIR variable node that represents the fragment.
 * @param os Output stream to which the scope/qualifier should be printed.
 */

/**
 * Compute the number of scalar elements for a WMMA fragment.
 *
 * Returns the total element count for the fragment described by `scope` and
 * `variable`, constrained by the provided `size` when applicable. Used to
 * allocate or index WMMA fragment storage during code emission.
 *
 * @param scope The WMMA scope string (e.g., "matrix_a", "matrix_b", "accumulator").
 * @param variable The TIR variable node that represents the fragment.
 * @param size A hint or requested number of elements; interpretation may vary
 *             by scope and fragment layout.
 * @return The computed fragment element count (int32_t).
 */
  bool need_cast_smem_ptr_to_int_{false};
  /**
 * Print WMMA fragment scope qualifiers for a variable.
 *
 * Emits the appropriate WMMA memory scope and layout qualifiers for `variable`
 * based on `scope` and element `t`, writing the result to `os`.
 *
 * @param scope WMMA scope name (e.g., "wmma.matrix_a", "wmma.matrix_b", "wmma.accumulator").
 * @param t element data type of the fragment.
 * @param variable pointer to the VarNode representing the fragment variable.
 * @param os output stream to which the scope qualifiers are printed.
 */
 
/**
 * Compute the number of scalar elements in a WMMA fragment.
 *
 * Returns the fragment size (number of elements) for `variable` under the
 * given WMMA `scope`. `size` is a fallback or context-dependent value (e.g.,
 * requested number of elements or vector length) used when the fragment shape
 * cannot be derived directly from metadata.
 *
 * @param scope WMMA scope name.
 * @param variable pointer to the VarNode representing the fragment variable.
 * @param size contextual size/value used as a fallback when fragment shape is unknown.
 * @return the fragment size in number of scalar elements.
 */
  bool need_cooperative_groups_{false};
  // Op attribute map
  OpAttrMap<bool> op_need_warp_shuffle_ =
      Op::GetAttrMap<bool>("cuda.need_warp_shuffle");

  // The name of the barrier array in shared memory
  const std::string barrier_name_ = "barrier";
  // The size of the barrier array in shared memory
  int barrier_count_ = -1;
  // The name of the mbarrier array in shared memory
  const std::string mbarrier_name_ = "mbarrier";
  // The type name of the mbarrier array
  const std::string mbarrier_dtype_ = "Barrier";
  // The alignment of the barrier array in shared memory
  // Set to 16 to maintain minimum alignment requirements for async bulk copy
  const int barrier_alignment_bytes_ = 16;

  std::unordered_map<const VarNode *, std::string> fragment_shapes;
  std::unordered_map<const VarNode *, std::string> fragment_layouts;
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
