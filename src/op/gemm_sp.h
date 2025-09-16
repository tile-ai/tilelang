/*!
 * \file tl/op/gemm_sp.h
 * \brief Define gemm_sp operator.
 *
 */

#ifndef TVM_TL_OP_GEMM_SP_H_
#define TVM_TL_OP_GEMM_SP_H_

#include "gemm.h"
#include "operator.h"

namespace tvm {

namespace tl {

using namespace tir;

class GemmSPWarpPolicyNode : public GemmWarpPolicyNode {
public:
  std::pair<int, int> ComputeWarpPartition(int M, int N, int block_size,
                                           Target target, bool use_wgmma,
                                           int bits) const;
};

class GemmSPWarpPolicy : public ObjectRef {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(GemmSPWarpPolicy, ObjectRef,
                                GemmSPWarpPolicyNode);

  explicit GemmSPWarpPolicy(GemmWarpPolicyType policy_type) {
    auto node = make_object<GemmSPWarpPolicyNode>();
    node->policy_type = (int)policy_type;
    data_ = std::move(node);
  }

  explicit GemmSPWarpPolicy(int policy_type) {
    auto node = make_object<GemmSPWarpPolicyNode>();
    node->policy_type = policy_type;
    data_ = std::move(node);
  }

  explicit GemmSPWarpPolicy(int m_warp, int n_warp) {
    auto node = make_object<GemmSPWarpPolicyNode>();
    node->m_warp = m_warp;
    node->n_warp = n_warp;
    node->policy_type = (int)GemmWarpPolicyType::kFree;
    data_ = std::move(node);
  }
};

class GemmSPNode : public TileOperatorNode {
public:
  tir::Buffer A, B, C, E;
  bool trans_A, trans_B;
  int M, N, K;
  bool clear_accum = false;
  // k_pack please ref to bitblas/tl/mfma_macro_generator.py::k_pack
  // only will be enabled under cdna mfma instructions
  int kPack = 1;
  int wg_wait = 0;

  mutable GemmSPWarpPolicy policy;

  static constexpr const char *_type_key = "tl.GemmSP";
  TVM_DECLARE_FINAL_OBJECT_INFO(GemmSPNode, TileOperatorNode);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  TileOperator Clone() const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmSPNode>()
        .def_ro("policy", &GemmSPNode::policy)
        .def_ro("A", &GemmSPNode::A)
        .def_ro("B", &GemmSPNode::B)
        .def_ro("C", &GemmSPNode::C)
        .def_ro("E", &GemmSPNode::E)
        .def_ro("trans_A", &GemmSPNode::trans_A)
        .def_ro("trans_B", &GemmSPNode::trans_B)
        .def_ro("M", &GemmSPNode::M)
        .def_ro("N", &GemmSPNode::N)
        .def_ro("K", &GemmSPNode::K)
        .def_ro("clear_accum", &GemmSPNode::clear_accum)
        .def_ro("kPack", &GemmSPNode::kPack)
        .def_ro("wg_wait", &GemmSPNode::wg_wait);
  }

  bool SEqualReduce(const GemmSPNode *other, SEqualReducer equal) const {
    return equal(A, other->A) && equal(B, other->B) && equal(C, other->C) &&
           equal(E, other->E) && equal(trans_A, other->trans_A) &&
           equal(trans_B, other->trans_B) && equal(M, other->M) &&
           equal(N, other->N) && equal(K, other->K) &&
           equal(clear_accum, other->clear_accum) &&
           equal(kPack, other->kPack) && equal(wg_wait, other->wg_wait);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(policy);
    hash_reduce(A);
    hash_reduce(B);
    hash_reduce(C);
    hash_reduce(E);
    hash_reduce(trans_A);
    hash_reduce(trans_B);
    hash_reduce(M);
    hash_reduce(N);
    hash_reduce(K);
    hash_reduce(clear_accum);
    hash_reduce(kPack);
    hash_reduce(wg_wait);
  }

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
