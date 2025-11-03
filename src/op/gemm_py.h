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
  bool CheckWGMMA() const;
  bool AllowTCGEN5MMA(Target target) const;
  bool AllowWGMMA(int block_size, Target target) const;
  tir::Buffer A, B, C;
  // pointer to the A, B, C
  PrimExpr Aptr, Bptr, Cptr;
  bool trans_A, trans_B;
  int M, N, K;
  int stride_A, stride_B;
  int offset_A, offset_B;
  PrimExpr clear_accum = const_false();
  PrimExpr mbarptr;
  Array<PrimExpr> C_coords;
  // k_pack please ref to bitblas/tl/mfma_macro_generator.py::k_pack
  // only will be enabled under cdna mfma instructions
  int kPack = 1;
  int wg_wait = 0;
  mutable GemmWarpPolicy policy;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.GemmPy", GemmPyNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmPyNode>()
        .def_ro("A", &GemmPyNode::A)
        .def_ro("B", &GemmPyNode::B)
        .def_ro("C", &GemmPyNode::C)
        .def_ro("Aptr", &GemmPyNode::Aptr)
        .def_ro("Bptr", &GemmPyNode::Bptr)
        .def_ro("Cptr", &GemmPyNode::Cptr)
        .def_ro("trans_A", &GemmPyNode::trans_A)
        .def_ro("trans_B", &GemmPyNode::trans_B)
        .def_ro("M", &GemmPyNode::M)
        .def_ro("N", &GemmPyNode::N)
        .def_ro("K", &GemmPyNode::K)
        .def_ro("stride_A", &GemmPyNode::stride_A)
        .def_ro("stride_B", &GemmPyNode::stride_B)
        .def_ro("offset_A", &GemmPyNode::offset_A)
        .def_ro("offset_B", &GemmPyNode::offset_B)
        .def_ro("clear_accum", &GemmPyNode::clear_accum)
        .def_ro("mbarptr", &GemmPyNode::mbarptr)
        .def_ro("C_coords", &GemmPyNode::C_coords)
        .def_ro("kPack", &GemmPyNode::kPack)
        .def_ro("wg_wait", &GemmPyNode::wg_wait)
        .def_ro("policy", &GemmPyNode::policy);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  TileOperator Clone() const;

  // Target GEMM instruction
  GemmInst GetGemmInst(int block_size, Target target) const;

private:
  mutable bool completed_ = false;
};

class GemmPy : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GemmPy, TileOperator, GemmPyNode);
  TVM_DLL GemmPy(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_GEMM_PY_H_
