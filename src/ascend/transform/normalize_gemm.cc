/*!
 * \file normalize_gemm.cc
 * \brief Downgrade L1-input GEMMs to L0A/L0B GEMMs.
 */

#include <optional>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cast.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <string>
#include <vector>

#include "ascend/op/utils.h"
#include "op/builtin.h"
#include "op/gemm.h"
#include "op/operator.h"
#include "op/utils.h"
#include "tir/transforms/ir_utils.h"
#include "tvm/ffi/cast.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

namespace {

constexpr int kPhysicalL0ABSize = 64 * 1024;
constexpr int kL0Stages = 2;
constexpr int kFracN = 16;

PrimExpr MakeRegion(const Buffer &buffer, const Array<Range> &ranges,
                    AccessMask access_mask) {
  Array<PrimExpr> mins;
  Array<PrimExpr> region_args;
  for (const Range &range : ranges)
    mins.push_back(range->min);
  region_args.push_back(BufferLoad(buffer, mins));
  region_args.push_back(IntImm(DataType::Int(32), access_mask));
  for (const Range &range : ranges)
    region_args.push_back(range->extent);
  return Call(DataType::Handle(), ::tvm::tl::region(), std::move(region_args));
}

struct L1TileGeometry {
  int tile_k_sub;
  int sub_k;
};

L1TileGeometry L1TilePlan(int m, int n, int k, int elem_bits) {
  int c0 = 256 / elem_bits;
  int l0_stage_bytes = kPhysicalL0ABSize / kL0Stages;
  int max_k = std::min({l0_stage_bytes * 8 / (m * elem_bits),
                        l0_stage_bytes * 8 / (n * elem_bits), k});
  int tile_k_sub = max_k / c0 * c0;
  TVM_FFI_ICHECK(tile_k_sub > 0) << "Cannot fit any sub-K tile in L0: M=" << m
                                 << ", N=" << n << ", elem_bits=" << elem_bits;
  while (k % tile_k_sub != 0 && tile_k_sub > c0)
    tile_k_sub -= c0;
  TVM_FFI_ICHECK(k % tile_k_sub == 0)
      << "K (" << k
      << ") is not divisible by any valid TILE_K_SUB <= " << max_k;
  TVM_FFI_ICHECK(m % kFracN == 0 && n % kFracN == 0)
      << "M (" << m << ") and N (" << n << ") must be multiples of " << kFracN;
  return {tile_k_sub, k / tile_k_sub};
}

PrimExpr AsBool(const PrimExpr &value) {
  if (value.dtype().is_bool())
    return value;
  return NE(value, make_const(value.dtype(), 0));
}

class GemmNormalizer : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt stmt) { return GemmNormalizer()(std::move(stmt)); }

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr || !call->op.same_as(Op::Get("tl.tileop.gemm")))
      return StmtExprMutator::VisitStmt_(op);
    // Only process GEMMs whose A/B inputs are in L1.
    TileOperator tile_op;
    try {
      tile_op = ParseOperator(GetRef<Call>(call));
    } catch (const std::bad_optional_access &e) {
      LOG(FATAL) << "bad_optional_access while parsing tile op call"
                 << GetRef<Call>(call);
    }
    const auto *gemm = tile_op.as<GemmNode>();
    if (gemm == nullptr || !IsL1Buffer(gemm->aRegion_->buffer) ||
        !IsL1Buffer(gemm->bRegion_->buffer)) {
      return StmtExprMutator::VisitStmt_(op);
    }
    return Splice(GetRef<Call>(call), gemm);
  }

  Stmt Splice(const Call &call, const GemmNode *gemm) {
    const BufferRegion &a = gemm->aRegion_;
    const BufferRegion &b = gemm->bRegion_;
    const BufferRegion &c = gemm->cRegion_;
    DataType dtype = a->buffer->dtype;
    L1TileGeometry plan =
        L1TilePlan(gemm->m_, gemm->n_, gemm->k_, dtype.bits() * dtype.lanes());
    bool multi_step = plan.sub_k > 1;
    // An NN GEMM keeps B as (K, N); an NT GEMM keeps it as (N, K).
    bool b_transposed = gemm->transB_;
    TVM_FFI_ICHECK(!gemm->transA_)
        << "NormalizeGemm only supports trans_A=False (Ascend L1 GEMM is NT or "
           "NN)";

    Array<PrimExpr> tile_k{IntImm(DataType::Int(32), plan.tile_k_sub)};
    Array<PrimExpr> m_extent{IntImm(DataType::Int(32), gemm->m_)};
    Array<PrimExpr> n_extent{IntImm(DataType::Int(32), gemm->n_)};
    Buffer a_tile =
        decl_buffer({m_extent[0], tile_k[0]}, dtype, "l0a", "shared.l0a");
    Buffer b_tile =
        decl_buffer({n_extent[0], tile_k[0]}, dtype, "l0b", "shared.l0b");
    TVM_FFI_ICHECK(!pending_allocs_.empty());
    pending_allocs_.back().push_back(a_tile);
    pending_allocs_.back().push_back(b_tile);

    Var sk("sk", DataType::Int(32));
    PrimExpr k_offset = sk * tile_k[0];
    Array<Range> a_l1_ranges{
        Range::FromMinExtent(a->region[0]->min, m_extent[0]),
        Range::FromMinExtent(a->region[1]->min + k_offset, tile_k[0])};
    Array<Range> b_l1_ranges =
        b_transposed
            ? Array<Range>{Range::FromMinExtent(b->region[0]->min, n_extent[0]),
                           Range::FromMinExtent(b->region[1]->min + k_offset,
                                                tile_k[0])}
            : Array<Range>{
                  Range::FromMinExtent(b->region[0]->min + k_offset, tile_k[0]),
                  Range::FromMinExtent(b->region[1]->min, n_extent[0])};
    Array<Range> a_l0_ranges{
        Range::FromMinExtent(IntImm(DataType::Int(32), 0), m_extent[0]),
        Range::FromMinExtent(IntImm(DataType::Int(32), 0), tile_k[0])};
    Array<Range> b_l0_ranges{
        Range::FromMinExtent(IntImm(DataType::Int(32), 0), n_extent[0]),
        Range::FromMinExtent(IntImm(DataType::Int(32), 0), tile_k[0])};

    static const Op &copy_op = Op::Get("tl.tileop.ascend_copy");
    Stmt copy_a = Evaluate(
        Call(DataType::Handle(), copy_op,
             {MakeRegion(a->buffer, a_l1_ranges, AccessMask::kAccessRead),
              MakeRegion(a_tile, a_l0_ranges, AccessMask::kAccessWrite)}));
    // Ascend applies B's transpose during the L1→L0B copy.
    Map<String, ObjectRef> copy_b_annotations;
    if (!b_transposed) {
      copy_b_annotations.Set("transpose", IntImm(DataType::Int(32), 1));
    }
    Stmt copy_b = Evaluate(
        Call(DataType::Handle(), copy_op,
             {MakeRegion(b->buffer, b_l1_ranges, AccessMask::kAccessRead),
              MakeRegion(b_tile, b_l0_ranges, AccessMask::kAccessWrite)},
             copy_b_annotations));

    PrimExpr clear_accum = gemm->clearAccum_;
    bool clears_constant = false;
    if (const auto *imm = clear_accum.as<IntImmNode>())
      clears_constant = imm->value != 0;
    PrimExpr step_clear = multi_step ? And(AsBool(clear_accum),
                                           EQ(sk, IntImm(DataType::Int(32), 0)))
                                     : clear_accum;

    // For multi-step GEMM, set unit_flag_ctrl to 3 on the last iteration to
    // enable fixpipe output, and to 2 on non-last iterations to prevent fixpipe
    // from writing out before computation completes.
    Map<String, ObjectRef> annotations = call->annotations;
    if (Optional<ObjectRef> unit_flag =
            call->annotations.Get("unit_flag_ctrl")) {
      PrimExpr base_uf = Downcast<PrimExpr>(unit_flag.value());
      annotations.Set(
          "unit_flag_ctrl",
          multi_step ? Select(EQ(sk, IntImm(DataType::Int(32), plan.sub_k - 1)),
                              base_uf,
                              Select(EQ(base_uf, IntImm(DataType::Int(32), 3)),
                                     IntImm(DataType::Int(32), 2), base_uf))
                     : base_uf);
    }

    static const Op &gemm_op = Op::Get("tl.tileop.gemm");
    AccessMask c_mask = (!multi_step && clears_constant)
                            ? AccessMask::kAccessWrite
                            : AccessMask::kAccessReadWrite;
    // B transpose is applied by the L1→L0B copy, so the MAD never transposes
    // and is always built as NT.
    Call mad(DataType::Handle(), gemm_op,
             {MakeRegion(a_tile, a_l0_ranges, AccessMask::kAccessRead),
              MakeRegion(b_tile, b_l0_ranges, AccessMask::kAccessRead),
              MakeRegion(c->buffer, c->region, c_mask), call->args[3],
              Bool(true), call->args[5], call->args[6], tile_k[0],
              call->args[8], step_clear, call->args[10], call->args[11],
              call->args[12]},
             annotations);

    For loop(sk, IntImm(DataType::Int(32), 0),
             IntImm(DataType::Int(32), plan.sub_k), ForKind::kSerial,
             SeqStmt({copy_a, copy_b, Evaluate(mad)}));
    return loop;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    pending_allocs_.emplace_back();
    Stmt rewritten = StmtExprMutator::VisitStmt_(op);
    std::vector<Buffer> pending = std::move(pending_allocs_.back());
    pending_allocs_.pop_back();
    if (pending.empty())
      return rewritten;
    SBlock block = Downcast<SBlock>(rewritten);
    SBlockNode *writer = block.CopyOnWrite();
    for (Buffer &buffer : pending)
      writer->alloc_buffers.push_back(std::move(buffer));
    return block;
  }

  std::vector<std::vector<Buffer>> pending_allocs_;
};

} // namespace

namespace transform {

Pass NormalizeGemm() {
  auto pass_func = [](PrimFunc func, const IRModule &, PassContext) {
    PrimFuncNode *writer = func.CopyOnWrite();
    writer->body = GemmNormalizer::Rewrite(std::move(writer->body));
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.NormalizeGemm", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.NormalizeGemm", NormalizeGemm);
}

} // namespace transform
} // namespace tl
} // namespace tvm
