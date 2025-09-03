/*!
 * \file tl/op/gemm.h
 * \brief Define gemm operator.
 *
 */

#ifndef TVM_TL_OP_GEMM_H_
#define TVM_TL_OP_GEMM_H_

#include "operator.h"

namespace tvm {

/**
 * Construct a Gemm operator handle from call arguments and a buffer mapping.
 *
 * @param args Array of call-time PrimExpr arguments passed to the operator.
 * @param vmap Mapping from buffer names/indices to tir::Buffer objects used by
 * this GEMM.
 */
/**
 * Obtain the registered Op descriptor for the GEMM operator.
 *
 * @returns A const reference to the Op representing "tl.Gemm".
 */
namespace tl {

using namespace tir;

enum class GemmWarpPolicy : uint8_t {
  kSquare = 0,
  kFullRow = 1,
  kFullCol = 2,
};

class GemmNode : public TileOperatorNode {
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
  GemmWarpPolicy policy;

  static constexpr const char *_type_key = "tl.Gemm";
  TVM_DECLARE_FINAL_OBJECT_INFO(GemmNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmNode>()
    // TODO(lei): legalize policy into a object node
      .def_ro("A", &GemmNode::A)
      .def_ro("B", &GemmNode::B)
      .def_ro("C", &GemmNode::C)
      .def_ro("Aptr", &GemmNode::Aptr)
      .def_ro("Bptr", &GemmNode::Bptr)
      .def_ro("Cptr", &GemmNode::Cptr)
      .def_ro("trans_A", &GemmNode::trans_A)
      .def_ro("trans_B", &GemmNode::trans_B)
      .def_ro("M", &GemmNode::M)
      .def_ro("N", &GemmNode::N)
      .def_ro("K", &GemmNode::K)
      .def_ro("stride_A", &GemmNode::stride_A)
      .def_ro("stride_B", &GemmNode::stride_B)
      .def_ro("offset_A", &GemmNode::offset_A)
      .def_ro("offset_B", &GemmNode::offset_B)
      .def_ro("clear_accum", &GemmNode::clear_accum)
      .def_ro("kPack", &GemmNode::kPack)
      .def_ro("wg_wait", &GemmNode::wg_wait);
  }

  bool SEqualReduce(const GemmNode *other, SEqualReducer equal) const {
    return equal(A, other->A) && equal(B, other->B) && equal(C, other->C) && equal(Aptr, other->Aptr) && equal(Bptr, other->Bptr) && equal(Cptr, other->Cptr) && equal(trans_A, other->trans_A) && equal(trans_B, other->trans_B) && equal(M, other->M) && equal(N, other->N) && equal(K, other->K) && equal(stride_A, other->stride_A) && equal(stride_B, other->stride_B) && equal(offset_A, other->offset_B) && equal(offset_B, other->offset_B) && equal(clear_accum, other->clear_accum) && equal(kPack, other->kPack) && equal(wg_wait, other->wg_wait);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    // TODO(lei): legalize policy into a object node
    // hash_reduce(policy);`
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
  }
  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  TileOperator Clone() const;

private:
  // Target GEMM instruction
  enum class GemmInst : uint8_t { kMMA, kWGMMA, kUTCMMA, kMFMA };
  GemmInst GetGemmInst(int block_size, Target target) const;

  std::pair<int, int> ComputeWarpPartition(int num_warps, GemmInst gemm_inst,
                                           Target target) const;

  mutable bool completed_ = false;
};

class Gemm : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(Gemm, TileOperator, GemmNode);
  TVM_DLL Gemm(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_GEMM_H_