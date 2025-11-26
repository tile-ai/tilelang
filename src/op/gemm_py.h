/*!
 * \file tl/op/gemm_py.h
 * \brief Define gemm operator.
 *
 */

#ifndef TVM_TL_OP_GEMM_PY_H_
#define TVM_TL_OP_GEMM_PY_H_

#include "gemm.h"
#include "operator.h"

namespace tvm {

namespace tl {

using namespace tir;

class GemmPyNode : public TileOperatorNode {
public:
  bool checkWgmma() const;
  bool allowTcgen5Mma(Target target) const;
  bool allowWgmma(int block_size, Target target) const;
  tir::Buffer a_, b_, c_;
  // BufferRegion for A, B and C
  BufferRegion aRegion_, bRegion_, cRegion_;
  bool transA_, transB_;
  int m_, n_, k_;
  int strideA_, strideB_;
  int offsetA_, offsetB_;
  PrimExpr clearAccum_ = const_false();
  BufferRegion mbarRegion_;
  tir::Buffer mbar_; // mbar is optional, only used for TCGEN5MMA
  Array<PrimExpr> cCoords_;
  // k_pack please ref to bitblas/tl/mfma_macro_generator.py::k_pack
  // only will be enabled under cdna mfma instructions
  int kPack_ = 1;
  int wgWait_ = 0;
  mutable GemmWarpPolicy policy_;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.GemmPy", GemmPyNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmPyNode>()
        .def_ro("a", &GemmPyNode::a_)
        .def_ro("b", &GemmPyNode::b_)
        .def_ro("c", &GemmPyNode::c_)
        .def_ro("aRegion", &GemmPyNode::aRegion_)
        .def_ro("bRegion", &GemmPyNode::bRegion_)
        .def_ro("cRegion", &GemmPyNode::cRegion_)
        .def_ro("transA", &GemmPyNode::transA_)
        .def_ro("transB", &GemmPyNode::transB_)
        .def_ro("m", &GemmPyNode::m_)
        .def_ro("n", &GemmPyNode::n_)
        .def_ro("k", &GemmPyNode::k_)
        .def_ro("strideA", &GemmPyNode::strideA_)
        .def_ro("strideB", &GemmPyNode::strideB_)
        .def_ro("offsetA", &GemmPyNode::offsetA_)
        .def_ro("offsetB", &GemmPyNode::offsetB_)
        .def_ro("clearAccum", &GemmPyNode::clearAccum_)
        .def_ro("mbarRegion", &GemmPyNode::mbarRegion_)
        .def_ro("mbar", &GemmPyNode::mbar_)
        .def_ro("cCoords", &GemmPyNode::cCoords_)
        .def_ro("kPack", &GemmPyNode::kPack_)
        .def_ro("wgWait", &GemmPyNode::wgWait_)
        .def_ro("policy", &GemmPyNode::policy_);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  TileOperator Clone() const;

  // Target GEMM instruction
  GemmInst getGemmInst(int block_size, Target target) const;

private:
  mutable bool completed_ = false;
};

class GemmPy : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GemmPy, TileOperator, GemmPyNode);
  TVM_DLL GemmPy(Array<PrimExpr> args);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_GEMM_PY_H_
