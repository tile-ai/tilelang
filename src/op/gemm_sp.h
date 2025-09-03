/*!
 * \file tl/op/gemm_sp.h
 * \brief Define gemm_sp operator.
 *
 */

#ifndef TVM_TL_OP_GEMM_SP_H_
#define TVM_TL_OP_GEMM_SP_H_

#include "operator.h"

namespace tvm {
/**
 * Lower the GemmSP operator into a TIR statement for the given lowering
 * context.
 *
 * Produces the TIR Stmt that implements this operator using the provided
 * lowering arguments. The `analyzer` is used for arithmetic simplifications and
 * may be null.
 *
 * @param T Lowering context and arguments.
 * @returns A TIR `Stmt` implementing the lowered operator.
 */
/**
 * Infer memory/layout mapping for operands and outputs of this operator.
 *
 * Computes a LayoutMap describing how logical tensor layouts map to physical
 * buffer layouts for the given inference `level`.
 *
 * @param T Layout inference inputs (shapes, buffer info, etc.).
 * @param level Inference granularity/level.
 * @returns A LayoutMap describing inferred layouts.
 */
/**
 * Compute a warp-level partitioning (rows, cols) for the given number of warps.
 *
 * Returns a pair (warps_per_row, warps_per_col) describing how to tile the GEMM
 * across warps for the specified `target`. The optional `maybe_hopper_wgmma`
 * enables target-specific adjustments (e.g., CDNA WG/MMA variants) when set.
 *
 * @param num_warps Total number of warps available for the tile.
 * @param target Target device/architecture used to guide partitioning choices.
 * @param maybe_hopper_wgmma Enable target-specific WG/MMA adjustments when
 * true.
 * @returns Pair<int,int> of (warps_per_row, warps_per_col).
 */
/**
 * Create a copy of this TileOperator node as a TileOperator reference.
 *
 * The returned TileOperator refers to a new node that is a copy of this node.
 *
 * @returns A TileOperator that is a clone of this node.
 */
/**
 * Construct a GemmSP TileOperator from call arguments and a buffer map.
 *
 * @param args Array of PrimExpr specifying call-site arguments for the
 * operator.
 * @param vmap Mapping from buffer names to tir::Buffer objects for
 * operands/outputs.
 */
/**
 * Return the singleton Op descriptor for the GemmSP operator.
 *
 * @returns Reference to the operator's Op registration object.
 */
namespace tl {

using namespace tir;

class GemmSPNode : public TileOperatorNode {
public:
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const;
  enum class GemmWarpPolicy : uint8_t {
    kSquare = 0,
    kFullRow = 1,
    kFullCol = 2,
  } policy;

  std::pair<int, int>
  ComputeWarpPartition(int num_warps, Target target,
                       bool maybe_hopper_wgmma = true) const;

  tir::Buffer A, B, C, E;
  bool trans_A, trans_B;
  int M, N, K;
  bool clear_accum = false;
  // k_pack please ref to bitblas/tl/mfma_macro_generator.py::k_pack
  // only will be enabled under cdna mfma instructions
  int kPack = 1;
  int wg_wait = 0;

  TileOperator Clone() const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmSPNode>()
        // TODO(lei): legalize policy into a object node
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
    // TODO(lei): legalize policy into a object node
    // hash_reduce(policy);
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
