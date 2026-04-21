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
                                           Target target, GemmInst gemm_inst,
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

  // use GemmWarp Policy here as the atom size are flexible
  mutable GemmWarpPolicy policy;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.GemmSP", GemmSPNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmSPNode>()
        .def_ro("A", &GemmSPNode::A)
        .def_ro("E", &GemmSPNode::E)
        .def_ro("B", &GemmSPNode::B)
        .def_ro("C", &GemmSPNode::C)
        .def_ro("aRegion", &GemmSPNode::aRegion_)
        .def_ro("eRegion", &GemmSPNode::eRegion_)
        .def_ro("bRegion", &GemmSPNode::bRegion_)
        .def_ro("cRegion", &GemmSPNode::cRegion_)
        .def_ro("trans_A", &GemmSPNode::trans_A)
        .def_ro("trans_B", &GemmSPNode::trans_B)
        .def_ro("trans_E", &GemmSPNode::trans_E)
        .def_ro("M", &GemmSPNode::M)
        .def_ro("N", &GemmSPNode::N)
        .def_ro("K", &GemmSPNode::K)
        .def_ro("stride_A", &GemmSPNode::stride_A)
        .def_ro("stride_B", &GemmSPNode::stride_B)
        .def_ro("offset_A", &GemmSPNode::offset_A)
        .def_ro("offset_B", &GemmSPNode::offset_B)
        .def_ro("clear_accum", &GemmSPNode::clear_accum)
        .def_ro("kPack", &GemmSPNode::kPack)
        .def_ro("wg_wait", &GemmSPNode::wg_wait)
        .def_ro("policy", &GemmSPNode::policy);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  AccessRegions GetAccessRegions() const override;

  TileOperator Clone() const;

private:
  // Target GEMM instruction
  GemmInst GetGemmInst(int block_size, Target target) const;

  mutable bool completed_ = false;
};

class GemmSP : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GemmSP, TileOperator, GemmSPNode);
  TVM_DLL GemmSP(Array<PrimExpr> args,
                 Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_GEMM_SP_H_
