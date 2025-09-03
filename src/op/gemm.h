/*!
 * \file tl/op/gemm.h
 * \brief Define gemm operator.
 *
 */

#ifndef TVM_TL_OP_GEMM_H_
#define TVM_TL_OP_GEMM_H_

#include "operator.h"

namespace tvm {
/**
 * Check whether the target and configuration allow using WGMMA (wavefront-group
 * MMA) for this GEMM.
 *
 * @returns true if WGMMA can be used for the current node configuration and
 * target; false otherwise.
 */
/**
 * Lower this GEMM operator to a TVM Stmt for the given lowering context.
 *
 * @param T Lowering arguments and context (tile mappings, target, etc.).
 * @param analyzer Arithmetic analyzer used for symbolic simplification and
 * bounds reasoning.
 * @returns A lowered Stmt implementing the GEMM.
 */
/**
 * Infer memory/layout mapping for GEMM inputs/outputs at the given inference
 * level.
 *
 * @param T Layout inference inputs (buffers, shapes, constraints).
 * @param level Inference level that controls how aggressive/specific the
 * inferred layouts should be.
 * @returns A LayoutMap describing how logical tensor axes map to storage/layout
 * axes.
 */
/**
 * Create a deep/shallow copy of this TileOperator node as a TileOperator
 * reference.
 *
 * @returns A TileOperator reference that represents a clone of this GemmNode.
 */
/**
 * Determine the specific GEMM instruction variant to use for the given block
 * size and target.
 *
 * @param block_size The tile/block size (in elements or threads) used to select
 * instruction variant.
 * @param target The compilation target describing architecture and instruction
 * set.
 * @returns The GemmInst enum value representing the chosen GEMM instruction
 * family.
 */
/**
 * Compute how to partition work across warps for the given number of warps and
 * GEMM instruction.
 *
 * The returned pair is (warp_rows, warp_cols), describing the per-warp tiling
 * in row and column dimensions respectively.
 *
 * @param num_warps Total number of warps available for the block.
 * @param gemm_inst The GEMM instruction variant selected for the target.
 * @param target The compilation target which may constrain or influence
 * partitioning.
 * @returns A pair<int,int> = (warp_rows, warp_cols) describing the warp
 * partition.
 */
/**
 * Construct a Gemm operator handle from call arguments and a buffer mapping.
 *
 * @param args Array of call-time PrimExpr arguments passed to the operator.
 * @param vmap Mapping from buffer names/indices to tir::Buffer objects used by
 * this GEMM.
 */
/**
 * Obtain the registered Op descriptor for the GEMM operator.
 *
 * @returns A const reference to the Op representing "tl.Gemm".
 */
namespace tl {

using namespace tir;

enum class GemmWarpPolicy : uint8_t {
  kSquare = 0,
  kFullRow = 1,
  kFullCol = 2,
};

class GemmNode : public TileOperatorNode {
public:
  bool CheckWGMMA() const;
  Array<PrimExpr> call_args;
  tir::Buffer A, B, C;
  // pointer to the A, B, C
  PrimExpr Aptr, Bptr, Cptr;
  bool trans_A, trans_B;
  int M, N, K;
  int stride_A, stride_B;
  int offset_A, offset_B;
  bool clear_accum = false;
  // k_pack please ref to bitblas/tl/mfma_macro_generator.py::k_pack
  // only will be enabled under cdna mfma instructions
  int kPack = 1;
  int wg_wait = 0;
  GemmWarpPolicy policy;

  static constexpr const char *_type_key = "tl.Gemm";
  TVM_DECLARE_FINAL_OBJECT_INFO(GemmNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  TileOperator Clone() const;

private:
  // Target GEMM instruction
  enum class GemmInst : uint8_t { kMMA, kWGMMA, kUTCMMA, kMFMA };
  GemmInst GetGemmInst(int block_size, Target target) const;

  std::pair<int, int> ComputeWarpPartition(int num_warps, GemmInst gemm_inst,
                                           Target target) const;

  mutable bool completed_ = false;
};

class Gemm : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(Gemm, TileOperator, GemmNode);
  TVM_DLL Gemm(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_GEMM_H_