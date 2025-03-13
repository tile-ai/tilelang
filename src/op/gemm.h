// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/gemm.h
 * \brief Define gemm operator.
 *
 */

#ifndef TVM_TL_OP_GEMM_H_
#define TVM_TL_OP_GEMM_H_

#include "op.h"

namespace tvm {
namespace tl {

using namespace tir;

class Gemm : public Operator {
public:
  Gemm(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) final;
  static const Op &Get();
  enum class GemmWarpPolicy {
    kSquare = 0,
    kFullRow = 1,
    kFullCol = 2,
  } policy;

private:
  std::pair<int, int>
  ComputeWarpPartition(int num_warps, Target target,
                       bool maybe_hopper_wgmma = true) const;

  Array<PrimExpr> call_args;
  tir::Buffer A, B, C;
  bool trans_A, trans_B;
  int M, N, K;
  // k_pack please ref to bitblas/tl/mfma_macro_generator.py::k_pack
  // only will be enabled under cdna mfma instructions
  int kPack = 1;
  bool completed_ = false;
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_GEMM_H_