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
  std::pair<int, int> computeWarpPartition(int M, int N, int block_size,
                                           Target target, bool use_wgmma,
                                           int bits) const;
  TVM_FFI_DECLARE_OBJECT_INFO("tl.GemmSPWarpPolicy", GemmSPWarpPolicyNode,
                              GemmWarpPolicyNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmSPWarpPolicyNode>()
        .def_ro("policy_type", &GemmSPWarpPolicyNode::policy_type)
        .def_ro("m_warp", &GemmSPWarpPolicyNode::m_warp)
        .def_ro("n_warp", &GemmSPWarpPolicyNode::n_warp);
  }
};

class GemmSPWarpPolicy : public ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GemmSPWarpPolicy, ObjectRef,
                                             GemmSPWarpPolicyNode);

  explicit GemmSPWarpPolicy(GemmWarpPolicyType policy_type) {
    auto node = tvm::ffi::make_object<GemmSPWarpPolicyNode>();
    node->policy_type = (int)policy_type;
    data_ = std::move(node);
  }

  explicit GemmSPWarpPolicy(int policy_type) {
    auto node = tvm::ffi::make_object<GemmSPWarpPolicyNode>();
    node->policy_type = policy_type;
    data_ = std::move(node);
  }

  explicit GemmSPWarpPolicy(int m_warp, int n_warp) {
    auto node = tvm::ffi::make_object<GemmSPWarpPolicyNode>();
    node->m_warp = m_warp;
    node->n_warp = n_warp;
    node->policy_type = (int)GemmWarpPolicyType::kFree;
    data_ = std::move(node);
  }
};

class GemmSPNode : public TileOperatorNode {
public:
  BufferRegion aRegion_, bRegion_, cRegion_, eRegion_;
  tir::Buffer a_, b_, c_, e_;
  bool transA_, transB_;
  int m_, n_, k_;
  bool clearAccum_ = false;
  // k_pack please ref to bitblas/tl/mfma_macro_generator.py::k_pack
  // only will be enabled under cdna mfma instructions
  int kPack_ = 1;
  int wgWait_ = 0;

  mutable GemmSPWarpPolicy policy_;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.GemmSP", GemmSPNode, TileOperatorNode);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  TileOperator Clone() const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmSPNode>()
        .def_ro("policy", &GemmSPNode::policy_)
        .def_ro("aRegion", &GemmSPNode::aRegion_)
        .def_ro("bRegion", &GemmSPNode::bRegion_)
        .def_ro("cRegion", &GemmSPNode::cRegion_)
        .def_ro("eRegion", &GemmSPNode::eRegion_)
        .def_ro("a", &GemmSPNode::a_)
        .def_ro("b", &GemmSPNode::b_)
        .def_ro("c", &GemmSPNode::c_)
        .def_ro("e", &GemmSPNode::e_)
        .def_ro("transA", &GemmSPNode::transA_)
        .def_ro("transB", &GemmSPNode::transB_)
        .def_ro("m", &GemmSPNode::m_)
        .def_ro("n", &GemmSPNode::n_)
        .def_ro("k", &GemmSPNode::k_)
        .def_ro("clearAccum", &GemmSPNode::clearAccum_)
        .def_ro("kPack", &GemmSPNode::kPack_)
        .def_ro("wgWait", &GemmSPNode::wgWait_);
  }

private:
  mutable bool completed_ = false;
};

class GemmSP : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GemmSP, TileOperator, GemmSPNode);
  TVM_DLL GemmSP(Array<PrimExpr> args);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_GEMM_SP_H_
