/*!
 * \file tl/op/gemm.h
 * \brief Define gemm operator.
 *
 */

#ifndef TVM_TL_OP_GEMM_H_
#define TVM_TL_OP_GEMM_H_

#include "operator.h"

namespace tvm {

namespace tl {

using namespace tir;

enum class GemmWarpPolicyType : uint8_t {
  kSquare = 0,
  kFullRow = 1,
  kFullCol = 2,
  kFree = 3,
};

/// Convert GemmWarpPolicyType enum to string for debugging
inline const char *GemmWarpPolicyTypeToString(GemmWarpPolicyType type) {
  switch (type) {
  case GemmWarpPolicyType::kSquare:
    return "Square";
  case GemmWarpPolicyType::kFullRow:
    return "FullRow";
  case GemmWarpPolicyType::kFullCol:
    return "FullCol";
  case GemmWarpPolicyType::kFree:
    return "Free";
  default:
    return "Unknown";
  }
}

// Target GEMM instruction
enum class GemmInst : uint8_t { kMMA, kWGMMA, kTCGEN5MMA, kMFMA };

/// Convert GemmInst enum to string for debugging
inline const char *GemmInstToString(GemmInst inst) {
  switch (inst) {
  case GemmInst::kMMA:
    return "MMA";
  case GemmInst::kWGMMA:
    return "WGMMA";
  case GemmInst::kTCGEN5MMA:
    return "TCGEN5MMA";
  case GemmInst::kMFMA:
    return "MFMA";
  default:
    return "Unknown";
  }
}

class GemmWarpPolicyNode : public Object {
public:
  mutable int m_warp{0};
  mutable int n_warp{0};
  int policy_type;

  TVM_FFI_DECLARE_OBJECT_INFO("tl.GemmWarpPolicy", GemmWarpPolicyNode, Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmWarpPolicyNode>()
        .def_ro("policy_type", &GemmWarpPolicyNode::policy_type)
        .def_ro("m_warp", &GemmWarpPolicyNode::m_warp)
        .def_ro("n_warp", &GemmWarpPolicyNode::n_warp);
  }

  std::pair<int, int> computeWarpPartition(int M, int N, int block_size,
                                           Target target,
                                           GemmInst gemm_inst) const;

  bool isSquare() const {
    return policy_type == int(GemmWarpPolicyType::kSquare);
  }
  bool isFullRow() const {
    return policy_type == int(GemmWarpPolicyType::kFullRow);
  }
  bool isFullCol() const {
    return policy_type == int(GemmWarpPolicyType::kFullCol);
  }
  bool isFree() const { return policy_type == int(GemmWarpPolicyType::kFree); }
};

class GemmWarpPolicy : public ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GemmWarpPolicy, ObjectRef,
                                             GemmWarpPolicyNode);

  explicit GemmWarpPolicy(GemmWarpPolicyType policy_type) {
    auto node = tvm::ffi::make_object<GemmWarpPolicyNode>();
    node->policy_type = (int)policy_type;
    data_ = std::move(node);
  }

  explicit GemmWarpPolicy(int policy_type) {
    auto node = tvm::ffi::make_object<GemmWarpPolicyNode>();
    node->policy_type = policy_type;
    data_ = std::move(node);
  }

  explicit GemmWarpPolicy(int m_warp, int n_warp) {
    auto node = tvm::ffi::make_object<GemmWarpPolicyNode>();
    node->m_warp = m_warp;
    node->n_warp = n_warp;
    node->policy_type = (int)GemmWarpPolicyType::kFree;
    data_ = std::move(node);
  }
};

class GemmNode : public TileOperatorNode {
public:
  bool checkWgmma() const;
  tir::Buffer a_, b_, c_;
  // BufferRegion for A, B and C
  BufferRegion aRegion_, bRegion_, cRegion_;
  bool transA_, transB_;
  int m_, n_, k_;
  int strideA_, strideB_;
  int offsetA_, offsetB_;
  PrimExpr clearAccum_ = const_false();
  // k_pack please ref to bitblas/tl/mfma_macro_generator.py::k_pack
  // only will be enabled under cdna mfma instructions
  int kPack_ = 1;
  int wgWait_ = 0;
  BufferRegion mbarRegion_;
  std::optional<tir::Buffer> mbar_; // mbar is optional, only used for TCGEN5MMA
  Array<PrimExpr> cCoords_;
  mutable GemmWarpPolicy policy_;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Gemm", GemmNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmNode>()
        .def_ro("a", &GemmNode::a_)
        .def_ro("b", &GemmNode::b_)
        .def_ro("c", &GemmNode::c_)
        .def_ro("aRegion", &GemmNode::aRegion_)
        .def_ro("bRegion", &GemmNode::bRegion_)
        .def_ro("cRegion", &GemmNode::cRegion_)
        .def_ro("transA", &GemmNode::transA_)
        .def_ro("transB", &GemmNode::transB_)
        .def_ro("m", &GemmNode::m_)
        .def_ro("n", &GemmNode::n_)
        .def_ro("k", &GemmNode::k_)
        .def_ro("strideA", &GemmNode::strideA_)
        .def_ro("strideB", &GemmNode::strideB_)
        .def_ro("offsetA", &GemmNode::offsetA_)
        .def_ro("offsetB", &GemmNode::offsetB_)
        .def_ro("clearAccum", &GemmNode::clearAccum_)
        .def_ro("kPack", &GemmNode::kPack_)
        .def_ro("wgWait", &GemmNode::wgWait_)
        .def_ro("policy", &GemmNode::policy_);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  TileOperator Clone() const;

private:
  GemmInst getGemmInst(int block_size, Target target) const;
  bool allowTcgen5Mma(Target target) const;
  bool allowWgmma(int block_size, Target target) const;

  mutable bool completed_ = false;
};

class Gemm : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Gemm, TileOperator, GemmNode);
  TVM_DLL Gemm(Array<PrimExpr> args);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_GEMM_H_
