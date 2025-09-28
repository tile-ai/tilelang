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
  mutable GemmWarpPolicy policy;

  static constexpr const char *_type_key = "tl.GemmPy";
  TVM_DECLARE_FINAL_OBJECT_INFO(GemmPyNode, TileOperatorNode);

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
        .def_ro("kPack", &GemmPyNode::kPack)
        .def_ro("wg_wait", &GemmPyNode::wg_wait)
        .def_ro("policy", &GemmPyNode::policy);
  }

  bool SEqualReduce(const GemmPyNode *other, SEqualReducer equal) const {
    return equal(A, other->A) && equal(B, other->B) && equal(C, other->C) &&
           equal(Aptr, other->Aptr) && equal(Bptr, other->Bptr) &&
           equal(Cptr, other->Cptr) && equal(trans_A, other->trans_A) &&
           equal(trans_B, other->trans_B) && equal(M, other->M) &&
           equal(N, other->N) && equal(K, other->K) &&
           equal(stride_A, other->stride_A) &&
           equal(stride_B, other->stride_B) &&
           equal(offset_A, other->offset_B) &&
           equal(offset_B, other->offset_B) &&
           equal(clear_accum, other->clear_accum) &&
           equal(kPack, other->kPack) && equal(wg_wait, other->wg_wait) &&
           equal(policy, other->policy);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(A);
    hash_reduce(B);
    hash_reduce(C);
    hash_reduce(Aptr);
    hash_reduce(Bptr);
    hash_reduce(Cptr);
    hash_reduce(trans_A);
    hash_reduce(trans_B);
    hash_reduce(M);
    hash_reduce(N);
    hash_reduce(K);
    hash_reduce(stride_A);
    hash_reduce(stride_B);
    hash_reduce(offset_A);
    hash_reduce(offset_B);
    hash_reduce(clear_accum);
    hash_reduce(kPack);
    hash_reduce(wg_wait);
    hash_reduce(policy);
  }
  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  TileOperator Clone() const;

private:
  // Target GEMM instruction
  GemmInst GetGemmInst(int block_size, Target target) const;

  mutable bool completed_ = false;
};

class GemmPy : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(GemmPy, TileOperator, GemmPyNode);
  TVM_DLL GemmPy(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_GEMM_PY_H_