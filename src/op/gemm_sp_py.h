/*!
 * \file tl/op/gemm_sp_py.h
 * \brief Define gemm_sp_py operator.
 *
 */

// TODO: @botbw: remove redundant code with gemm_py.h

#ifndef TVM_TL_OP_GEMM_SP_PY_H_
#define TVM_TL_OP_GEMM_SP_PY_H_

#include "gemm_sp.h"
#include "operator.h"

namespace tvm {

namespace tl {

using namespace tir;

class GemmSPPyNode : public TileOperatorNode {
public:
  bool CheckWGMMA() const;
  tir::Buffer A, E, B, C;
  // pointer to the A, E, B, C
  BufferRegion aRegion_, eRegion_, bRegion_, cRegion_;
  bool trans_A, trans_B, trans_E;
  int M, N, K;
  int stride_A, stride_B;
  int offset_A, offset_B;
  PrimExpr clear_accum = const_false();
  // k_pack please ref to bitblas/tl/mfma_macro_generator.py::k_pack
  // only will be enabled under cdna mfma instructions
  int kPack = 1;
  int wg_wait = 0;

  // use GemmWarp Policy here as the atom size are flexible in v2
  mutable GemmWarpPolicy policy;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.GemmSPPy", GemmSPPyNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmSPPyNode>()
        .def_ro("A", &GemmSPPyNode::A)
        .def_ro("E", &GemmSPPyNode::E)
        .def_ro("B", &GemmSPPyNode::B)
        .def_ro("C", &GemmSPPyNode::C)
        .def_ro("aRegion", &GemmSPPyNode::aRegion_)
        .def_ro("eRegion", &GemmSPPyNode::eRegion_)
        .def_ro("bRegion", &GemmSPPyNode::bRegion_)
        .def_ro("cRegion", &GemmSPPyNode::cRegion_)
        .def_ro("trans_A", &GemmSPPyNode::trans_A)
        .def_ro("trans_B", &GemmSPPyNode::trans_B)
        .def_ro("trans_E", &GemmSPPyNode::trans_E)
        .def_ro("M", &GemmSPPyNode::M)
        .def_ro("N", &GemmSPPyNode::N)
        .def_ro("K", &GemmSPPyNode::K)
        .def_ro("stride_A", &GemmSPPyNode::stride_A)
        .def_ro("stride_B", &GemmSPPyNode::stride_B)
        .def_ro("offset_A", &GemmSPPyNode::offset_A)
        .def_ro("offset_B", &GemmSPPyNode::offset_B)
        .def_ro("clear_accum", &GemmSPPyNode::clear_accum)
        .def_ro("kPack", &GemmSPPyNode::kPack)
        .def_ro("wg_wait", &GemmSPPyNode::wg_wait)
        .def_ro("policy", &GemmSPPyNode::policy);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  TileOperator Clone() const;

private:
  // Target GEMM instruction
  GemmInst GetGemmInst(int block_size, Target target) const;

  mutable bool completed_ = false;
};

class GemmSPPy : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GemmSPPy, TileOperator,
                                             GemmSPPyNode);
  TVM_DLL GemmSPPy(Array<PrimExpr> args);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_GEMM_SP_PY_H_
