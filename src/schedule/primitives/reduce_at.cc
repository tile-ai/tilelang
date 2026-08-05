/*!
 * \file reduce_at.cc
 * \brief Implements the ReduceAt schedule primitive for tilelang.
 *
 * Given a source block (whose write buffer is the reduction source),
 * a destination block (or the same block), a loop level, reduction type,
 * and dimension, this primitive inserts a `tl.tileop.reduce` statement.
 *
 * The reduce operation reads from the source buffer's region and writes
 * to the destination buffer's region, performing the specified reduction
 * (sum, max, min, abssum, absmax) along the given dimension.
 *
 * This is essential for:
 * - General reduction templates (softmax, layernorm, etc.)
 * - Cross-thread reductions within a tile
 * - Multi-stage reduction pipelines
 */

#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include "s_tir/schedule/transform.h"
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/utils.h>

#include "runtime/thread_storage_scope.h"
#include "s_tir/schedule/utils.h"
#include "s_tir/support/nd_int_set.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tvm::s_tir;
using support::NDIntSet;

// ---------------------------------------------------------------------------
// BlockRealizeReplacer: replace the first SBlockRealize that wraps `target_`
// with `replacement_`, preserving other statements in the loop body.
// ---------------------------------------------------------------------------
class BlockRealizeReplacer : public StmtMutator {
public:
  BlockRealizeReplacer(const SBlockNode *target, Stmt replacement)
      : target_(target), replacement_(std::move(replacement)),
        replaced_(false) {}

  Stmt VisitStmt(const Stmt &stmt) final {
    return replaced_ ? stmt : StmtMutator::VisitStmt(stmt);
  }

  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    if (op->block.get() == target_) {
      replaced_ = true;
      return replacement_;
    }
    return StmtMutator::VisitStmt_(op);
  }

  bool replaced() const { return replaced_; }

private:
  const SBlockNode *target_;
  Stmt replacement_;
  bool replaced_;
};

// ---------------------------------------------------------------------------
// DirectBlockLoopReplacer: replace the first For whose body directly contains
// the target SBlockRealize with `replacement_`.
// This is used to collapse an inner serial reduction loop while preserving
// sibling statements in the outer loop body (e.g. cached T.copy).
// ---------------------------------------------------------------------------
class DirectBlockLoopReplacer : public StmtMutator {
public:
  DirectBlockLoopReplacer(const SBlockNode *target, Stmt replacement)
      : target_(target), replacement_(std::move(replacement)),
        replaced_(false) {}

  Stmt VisitStmt(const Stmt &stmt) final {
    return replaced_ ? stmt : StmtMutator::VisitStmt(stmt);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (BodyDirectlyContainsTarget(op->body, target_)) {
      replaced_ = true;
      return replacement_;
    }
    return StmtMutator::VisitStmt_(op);
  }

  bool replaced() const { return replaced_; }

private:
  static bool BodyDirectlyContainsTarget(const Stmt &body,
                                         const SBlockNode *target) {
    if (const auto *realize = body.as<SBlockRealizeNode>()) {
      return realize->block.get() == target;
    }
    if (const auto *seq = body.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        if (const auto *realize = s.as<SBlockRealizeNode>()) {
          if (realize->block.get() == target)
            return true;
        }
      }
    }
    return false;
  }

  const SBlockNode *target_;
  Stmt replacement_;
  bool replaced_;
};

// ---------------------------------------------------------------------------
// Helper: Compute the relaxed access region of a buffer within a loop.
// ---------------------------------------------------------------------------
static ffi::Array<Range> ComputeRelaxedRegion(s_tir::ScheduleState self,
                                              const StmtSRef &loop_sref,
                                              const StmtSRef &block_sref,
                                              const Buffer &buf,
                                              BufferIndexType buffer_type) {
  const SBlockNode *block = TVM_SREF_TO_SBLOCK(block_sref);

  SBlockRealize realize = GetSBlockRealize(self, block_sref);
  ffi::Map<Var, PrimExpr> bindings = s_tir::GetBindings(realize);

  runtime::StorageScope scope = runtime::StorageScope::Create("local");
  ffi::Map<Var, arith::IntSet> var_dom =
      arith::AsIntSet(LoopDomainOfSRefTreePathSkipBlocks(
          ffi::GetRef<StmtSRef>(self->stmt2ref.at(block)->parent), loop_sref,
          scope));

  const auto &regions = (buffer_type == s_tir::BufferIndexType::kRead)
                            ? block->reads
                            : block->writes;

  std::vector<NDIntSet> relaxed_regions;
  for (const BufferRegion &buffer_region : regions) {
    if (buffer_region->buffer.same_as(buf)) {
      ffi::Array<arith::IntSet> relaxed =
          arith::EvalSet(Substitute(buffer_region->region, bindings), var_dom);
      relaxed_regions.push_back({relaxed.begin(), relaxed.end()});
    }
  }
  ICHECK(!relaxed_regions.empty()) << "ValueError: buffer " << buf->name
                                   << " is not accessed in the specified block";

  NDIntSet unified = support::NDIntSetUnion(relaxed_regions);
  int ndim = static_cast<int>(unified.size());

  arith::Analyzer analyzer;
  ffi::Array<Range> result;
  result.reserve(ndim);
  for (int d = 0; d < ndim; ++d) {
    PrimExpr mn = analyzer.Simplify(unified[d].min());
    PrimExpr mx = analyzer.Simplify(unified[d].max());
    PrimExpr extent = analyzer.Simplify(mx - mn + 1);
    result.push_back(Range::FromMinExtent(mn, extent));
  }
  return result;
}

// ---------------------------------------------------------------------------
// ReduceAt: main entry point
//
// Replaces the specified reduction block or its direct inner loop with a
// tl.tileop.reduce statement.  The source is a block's read buffer and the
// destination is the block's write buffer.
// ---------------------------------------------------------------------------
static void ReduceAt(s_tir::ScheduleState self, const StmtSRef &loop_sref,
                     const StmtSRef &block_sref, int read_buffer_index,
                     int write_buffer_index, const ffi::String &reduce_type,
                     int dim, bool clear) {
  CheckLoopIsAncestorOfBlock(self, loop_sref, block_sref, "reduce_at");
  // ---- Step 1: Obtain source and destination buffers -----------------------
  const SBlockNode *block = TVM_SREF_TO_SBLOCK(block_sref);
  SBlock block_ref = ffi::GetRef<SBlock>(block);

  Buffer src = GetNthAccessBuffer(self, block_ref, read_buffer_index,
                                  s_tir::BufferIndexType::kRead);
  Buffer dst = GetNthAccessBuffer(self, block_ref, write_buffer_index,
                                  s_tir::BufferIndexType::kWrite);

  const ForNode *loop = TVM_SREF_TO_FOR(loop_sref);

  // ---- Step 2: Compute the relaxed regions --------------------------------
  ffi::Array<Range> src_region = ComputeRelaxedRegion(
      self, loop_sref, block_sref, src, s_tir::BufferIndexType::kRead);
  ffi::Array<Range> dst_region = ComputeRelaxedRegion(
      self, loop_sref, block_sref, dst, s_tir::BufferIndexType::kWrite);

  // ---- Step 3: Build the T.reduce call ------------------------------------
  PrimExpr src_region_arg = MakeRegionCall(src, src_region, /*access_mask=*/1);
  PrimExpr dst_region_arg = MakeRegionCall(dst, dst_region, /*access_mask=*/2);

  Stmt reduce_stmt =
      Evaluate(Call(DataType::Handle(), Op::Get("tl.tileop.reduce"),
                    {src_region_arg, dst_region_arg, StringImm(reduce_type),
                     IntImm(DataType::Int(32), dim), Bool(clear)}));

  // ---- Step 4: Update loop body -------------------------------------------
  ObjectPtr<ForNode> new_loop_node = ffi::make_object<ForNode>(*loop);
  // Try to collapse the innermost loop that directly wraps the block.
  DirectBlockLoopReplacer loop_replacer(block, reduce_stmt);
  Stmt replaced_body = loop_replacer(loop->body);
  if (loop_replacer.replaced()) {
    new_loop_node->body = replaced_body;
  } else {
    // Fallback: replace only the block realize.
    BlockRealizeReplacer block_replacer(block, reduce_stmt);
    replaced_body = block_replacer(loop->body);
    // Final fallback: replace the whole loop body.
    new_loop_node->body =
        block_replacer.replaced() ? replaced_body : reduce_stmt;
  }
  For new_loop(new_loop_node);

  // ---- Step 5: Replace in the scope root block ----------------------------
  StmtSRef scope_root_sref =
      GetScopeRoot(self, loop_sref, /*require_stage_pipeline=*/false);
  const SBlockNode *scope_block = TVM_SREF_TO_SBLOCK(scope_root_sref);

  ffi::Map<SBlock, SBlock> block_sref_reuse;
  SBlock new_scope_block = Downcast<SBlock>(
      LoopReplacer(loop, new_loop)(ffi::GetRef<SBlock>(scope_block)));

  block_sref_reuse.Set(ffi::GetRef<SBlock>(scope_block), new_scope_block);
  self->Replace(scope_root_sref, new_scope_block, block_sref_reuse);
}

// ---------------------------------------------------------------------------
// FFI Registration
// ---------------------------------------------------------------------------
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.schedule.ScheduleReduceAt",
                        [](s_tir::Schedule self, const s_tir::LoopRV &loop_rv,
                           const SBlockRV &block_rv, int read_buffer_index,
                           int write_buffer_index,
                           const ffi::String &reduce_type, int dim,
                           bool clear) {
                          ReduceAt(self->state(), self->GetSRef(loop_rv),
                                   self->GetSRef(block_rv), read_buffer_index,
                                   write_buffer_index, reduce_type, dim, clear);
                        });
}

} // namespace tl
} // namespace tvm
