/*!
 * \file tl/op/gemm_sp.h
 * \brief Define gemm_sp operator.
 *
 */

#ifndef TVM_TL_OP_GEMM_SP_H_
#define TVM_TL_OP_GEMM_SP_H_

#include "operator.h"

namespace tvm {
namespace tl {

using namespace tir;

class GemmSPNode : public TileOperatorNode {
public:
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const;
  enum class GemmWarpPolicy {
    kSquare = 0,
    kFullRow = 1,
    kFullCol = 2,
  } policy;

  std::pair<int, int>
  ComputeWarpPartition(int num_warps, Target target,
                       bool maybe_hopper_wgmma = true) const;

  Array<PrimExpr> call_args;
  tir::Buffer A, B, C, E;
  bool trans_A, trans_B;
  int M, N, K;
  bool clear_accum = false;
  // k_pack please ref to bitblas/tl/mfma_macro_generator.py::k_pack
  // only will be enabled under cdna mfma instructions
  int kPack = 1;
  int wg_wait = 0;

  TileOperator Clone() const;

private:
  mutable bool completed_ = false;
};

class GemmSP : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(GemmSP, TileOperator, GemmSPNode);
  TVM_DLL GemmSP(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_GEMM_SP_H_
