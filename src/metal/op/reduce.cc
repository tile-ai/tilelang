/*!
 * \file tl/metal/op/reduce.cc
 * \brief Metal implementation for tl.reduce.
 *
 * Strategy (correctness-first, v1):
 *   Phase 1: per-thread local partials into the (possibly duplicated)
 *            accumulator buffer. Identical semantics to the generic GPU
 *            lowering.
 *   Phase 2: lockstep single-simdgroup XOR butterfly through a threadgroup
 *            scratch buffer. All threads execute the same barrier sequence
 *            (MSL has no CUDA-style named barriers), so non-participating
 *            threads contribute the reduce identity element instead of
 *            skipping the barrier. After log2(32) exchange levels every
 *            participating scratch slot holds the total; participants store
 *            it back to their accumulator slots under the red-layout
 *            participation predicate (the same partition guard the generic
 *            lowering uses), and only the layout owners later write dst.
 *            Phase 2 is omitted entirely when the reduce plan has no thread
 *            step (e.g. raw extent=1): Phase-1 partials are then already
 *            final.
 *   Phase 3: duplicate-buffer update back into dst (guarded by fragment
 *            ownership, same as the generic lowering).
 *
 * Hard domain (enforced at lowering entry): the
 * XOR-butterfly AllReduce is only correct when every value exchange stays
 * inside a single simdgroup. Within one butterfly level each thread reads
 * its partner's old value while writing its own; that ordering is
 * guaranteed by SIMD lockstep inside a simdgroup, but NOT across
 * simdgroups, and MSL has no CUDA-style named barriers to restrict the
 * barrier to the participating range. The threadgroup must be [0, N) with
 * a compile-time-constant N (raw thread index addresses the scratch), and
 * every reduce-plan thread step must satisfy nt = extent*scale a power of
 * two with nt <= 32: the participating range [0, nt) is closed
 * under the butterfly masks (nt/2 halving down to scale) iff nt is a
 * power of two — each mask then flips a single lane bit below log2(nt),
 * so no XOR partner escapes the range (a non-power-of-two nt like
 * (8,3)=24 lets tid^12 = 28 escape even though every offset is < 32).
 * nt <= 32 keeps the closure inside one 32-lane block (no XOR partner
 * across a simdgroup boundary). The raw butterfly executes on ALL N
 * threads (no tid < nt guard), so the actual execution prefix [0, N)
 * must itself be a union of complete nt-blocks: N % nt == 0. The largest
 * mask nt/2 partitions
 * lanes into aligned nt-blocks, and an incomplete tail block reads
 * partners >= N (OOB: scratch sized N) or never-written padding slots
 * (e.g. extent=8, scale=2, nt=16, N=24: tid 16..23 ^ 8 = 24..31).
 * Multi-simdgroup threadgroups whose steps stay inside 32-lane closures
 * with N % nt == 0 (for example, a DeepSeek V4 Flash-style grouped
 * reduction with threads=128, extent=32, scale=1, and offsets 16..1) are
 * therefore allowed; cross-simdgroup exchange, non-
 * power-of-two participating widths, or misaligned threadgroup extents
 * are rejected with an unsupported diagnostic.
 *
 * bf16 destinations/accumulators accumulate in fp32 (upstream AccType
 * design): the accumulator and the threadgroup scratch are fp32, bf16 is
 * materialized only at the final ownership write-back. With multiple
 * thread steps, every intermediate step round-trips through the fp32
 * accumulator (the next step reloads the updated group totals) and only
 * the last step casts fp32 -> bf16 into dst. Casting after every step would
 * discard intermediate fp32 group totals. bf16 bitwise reduce is rejected
 * because fp32 routing would change bit semantics.
 *
 * This avoids both call_extern (unsupported by the Metal codegen) and
 * global-workspace plumbing (not present in the Metal runtime).
 *
 * v1 documented limitations (follow-ups):
 *   - op.batch > 1 is not supported yet (LOG(FATAL)).
 *   - vectorized (packed) local reduction is disabled (vsize = 1).
 *   - fp16/bf16 nan_propagate reducers are not supported (no MSL __hmax_nan).
 *   - Reducer v2 (tl.finalize_reducer / FinalizeReducerOp) is NOT
 *     implemented on Metal: upstream registers finalize_reducer only for
 *     CUDA and ROCm, so a v2 epoch fails loudly at lowering with
 *     "no finalize_reducer implementation is registered for metal".
 *     This file covers only the legacy T.reduce path; the v2 boundary is
 *     exercised by test_finalize_reducer_v2_rejected_on_metal.
 */

#include "backend/common/op/reduce.h"
#include "backend/common/target_utils.h"
#include "op/utils.h"
#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>

#include "op/parallel.h"
#include "tir/transforms/ir_utils.h"
#include "transform/loop_partition.h"

#include <sstream>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace metal {

struct MetalReduce : backend::ReduceLowerer<MetalReduce> {
  static bool SupportsFp16Bf16NanReduce(Target target) { return false; }

  static int GetPreferredVectorizedSize(DataType dt, Target target) {
    return 1;
  }

  static Stmt Lower(const ReduceOpNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    if (op.nan_propagate &&
        (op.dst->dtype.is_float16() || op.dst->dtype.is_bfloat16())) {
      LOG(FATAL) << "ReduceOp: nan_propagate=True for fp16/bf16 "
                    "max/min/absmax is not supported on Metal targets "
                    "(no __hmax_nan equivalent in MSL). Target was: "
                 << lower_args.target->str();
    }
    auto get_buffer = [&](const Buffer &buffer) {
      auto it = lower_args.buffer_remap.find(buffer);
      return it == lower_args.buffer_remap.end() ? buffer : (*it).second;
    };

    if (IsLocalBuffer(op.src) && IsLocalBuffer(op.dst, /*allow_var*/ true)) {
      return LowerLocal(op, get_buffer(op.src), get_buffer(op.dst), lower_args);
    }

    if (IsFragmentBuffer(op.src) && IsFragmentBuffer(op.dst)) {
      // The XOR-butterfly AllReduce below is only correct when every
      // value exchange stays inside a single simdgroup. Within one level
      // each thread reads its partner's old value while writing its own;
      // that ordering is guaranteed by SIMD lockstep inside a simdgroup, but
      // NOT across simdgroups, and MSL has no CUDA-style named barriers to
      // restrict the barrier to the participating range.
      //
      // The threadgroup must be [0, N) with compile-time-constant N (the raw
      // thread index addresses the scratch buffer and PartitionLoop guards);
      // the actual single-simdgroup discipline is enforced on the reduce
      // plan after MakeReduceOwnershipPlan below. Per thread step,
      // nt = extent*scale must be a power of two and <= 32 — the XOR-
      // closed, single-simdgroup closure; the threadgroup extent N
      // must additionally satisfy N % nt == 0 so the ALL-N-thread execution
      // prefix is a union of complete nt-blocks).
      const int64_t *tb_min = as_const_int(lower_args.thread_bounds->min);
      const int64_t *tb_extent = as_const_int(lower_args.thread_bounds->extent);
      if (tb_min == nullptr || tb_extent == nullptr || *tb_min != 0) {
        LOG(FATAL) << "Metal reduce: fragment reduce requires a "
                      "compile-time-constant threadgroup [0, N) (got "
                      "thread_bounds = "
                   << lower_args.thread_bounds
                   << "); the raw thread index addresses the scratch "
                      "buffer and the partition guards.";
      }
      auto src_buffer = get_buffer(op.src);
      auto dst_buffer = get_buffer(op.dst);
      auto src_layout = lower_args.layout_map[op.src].as<Fragment>().value();
      auto dst_layout = lower_args.layout_map[op.dst].as<Fragment>().value();
      auto ctx = backend::reduce::MakeFragmentReduceContext(
          op, src_buffer, dst_buffer, src_layout, dst_layout, analyzer);
      auto red_layout = ctx.red_layout;
      auto dst_dim = ctx.dst_dim;
      auto &dst_vars = ctx.dst_vars;
      auto &dst_indices = ctx.dst_indices;
      auto &red_indices = ctx.red_indices;
      auto &reduce_plan = ctx.reduce_plan;

      ICHECK_EQ(op.batch, 1)
          << "Metal reduce: batch > 1 is not supported yet (op.batch="
          << op.batch << ")";

      // Plan-level enforcement is per thread_step:
      // thread_step.extent is the PER-STEP participating thread range,
      // NOT the threadgroup extent. The raw XOR butterfly must be closed
      // on the participating range [0, nt): within one butterfly level
      // each thread reads its partner's old value while writing its own;
      // that ordering is only guaranteed by SIMD lockstep inside a
      // simdgroup, and every partner must be a real, written lane.
      //
      // Explicit alignment contract (participating range vs nt blocks):
      // the participating range [0, nt) of every thread step must sit on
      // complete nt-blocks of the [0, N) execution prefix, enforced below
      // by N % nt == 0. This is a hard correctness invariant, not an
      // optimization: the raw butterfly runs on ALL N threads without a
      // tid < nt guard, so an incomplete tail block would read partners
      // >= N (outside the threadgroup; the scratch is sized N) or
      // never-written codegen-padding slots.
      //
      // Reject iff nt = extent*scale is not a power of two OR nt > 32:
      //  - XOR closure: [0, nt) is closed under every butterfly
      //    mask (nt/2 halving down to scale) iff nt is a power of two —
      //    each mask then flips a single lane bit below log2(nt), so
      //    tid^mask stays in [0, nt). For a non-power-of-two nt the
      //    partner of a participating lane escapes the range even when
      //    every mask is < 32: e.g. extent=8, scale=3 (nt=24, masks
      //    12/6/3) gives tid=16 ^ 12 = 28 >= 24 — a scratch read of a
      //    never-written slot / OOB. CheckAllReduceWidth
      //    (backend/common/op/reduce.h) only requires logical_width ==
      //    extent to be a positive power of two and scale > 0; scale
      //    need NOT be a power of two, so both the wider-than-simdgroup case
      //    (32 < nt < 64, e.g. (16,3)=48 -> partner 32^24 = 56) and the
      //    and the non-power-of-two case (nt <= 32, e.g. (8,3)=24 -> partner
      //    16^12 = 28) were reachable through the shared lowering.
      //  - single-simdgroup closure: nt > 32 means the range
      //    exceeds one 32-lane simdgroup and the largest mask nt/2
      //    pairs threads across a simdgroup boundary.
      //  - threadgroup alignment:
      //    the raw butterfly executes on ALL N threads without a
      //    tid < nt guard, so the ACTUAL closure domain is the whole
      //    execution prefix [0, N), not just the first block [0, nt).
      //    The largest mask nt/2 partitions lanes into aligned blocks
      //    of size nt; [0, N) is closed under it (and hence under every
      //    smaller mask 2^i | nt) iff N % nt == 0.  If N is not an
      //    integer multiple of nt, the last incomplete block reads
      //    partners >= N (outside the threadgroup; the scratch is sized
      //    N) or never-written codegen-padding slots (e.g. extent=8,
      //    scale=2, nt=16, N=24: tid 16..23 ^ 8 = 24..31 escapes the
      //    24-slot scratch).  N is the compile-time threadgroup extent
      //    checked at entry above.
      //
      // Group-local closures in multi-simdgroup threadgroups (every step
      // nt a power of two <= 32 with N % nt == 0, e.g.
      // a DeepSeek V4 Flash-style grouped reduction with threads=128,
      // extent=32, scale=1, offsets 16..1, or power-of-two replication
      // extent=16, scale=2, nt=32, or N=2*nt complete two-block closures)
      // perform zero cross-simdgroup value exchange and are allowed even
      // though the threadgroup extent is 128. An empty thread_steps plan
      // has no butterfly at all and is always safe.
      for (const auto &thread_step : reduce_plan.thread_steps) {
        int64_t nt =
            static_cast<int64_t>(thread_step.extent) * thread_step.scale;
        const int64_t N = *tb_extent;
        ICHECK_GT(nt, 0) << "Metal reduce: reduce plan thread step must have a "
                            "positive extent*scale (got extent="
                         << thread_step.extent
                         << ", scale=" << thread_step.scale << ")";
        int shift = 0;
        const bool nt_is_pow2 =
            tirx::is_const_power_of_two_integer(Integer(nt), &shift);
        if (!nt_is_pow2 || nt > 32 || N % nt != 0) {
          std::ostringstream msg;
          msg << "Metal reduce: fragment reduce butterfly must stay inside "
                 "a single-simdgroup, XOR-closed participating range: "
                 "thread_step extent="
              << thread_step.extent << " scale=" << thread_step.scale
              << " (butterfly offsets " << nt / 2 << " .. " << thread_step.scale
              << ", nt = extent*scale = " << nt
              << ", threadgroup extent N = " << N << ")";
          if (!nt_is_pow2) {
            msg << "; nt is not a power of two, so the participating "
                   "range [0, "
                << nt
                << ") is not closed under the XOR masks: a partner "
                   "can escape the range even though every offset "
                   "is < 32 (e.g. tid^"
                << nt / 2 << " may be >= " << nt
                << "), reading a scratch slot that no thread ever wrote "
                   "(undefined) or lying outside the threadgroup (OOB)";
          }
          if (N % nt != 0) {
            msg << "; N = " << N << " is not an integer multiple of nt = " << nt
                << ", so the last incomplete nt-block is not aligned: the "
                   "raw butterfly runs on all "
                << N
                << " threads without a tid < nt guard, and the tail block "
                   "reads partners >= "
                << N
                << " (outside the threadgroup; the scratch is sized N) or "
                   "never-written codegen-padding slots — complete "
                   "nt-blocks (32-lane aligned when nt = 32) are required "
                   "so every partner is a participating, written lane";
          }
          if (nt_is_pow2 && nt > 32) {
            msg << "; nt > 32 exceeds one 32-lane simdgroup, and "
                   "cross-simdgroup in-place combine is unsupported on "
                   "Metal (no named barriers; the partner-read/self-write "
                   "ordering is only guaranteed by SIMD lockstep inside "
                   "a simdgroup)";
          }
          msg << ". Use 32-thread threadgroups or per-simdgroup "
                 "group-local closures with power-of-two extent*scale "
                 "<= 32 and threadgroup extent a multiple of it.";
          LOG(FATAL) << msg.str();
        }
      }

      Array<Stmt> stmts;

      auto plan = backend::reduce::MakeReduceBufferPlan(
          op, dst_buffer, dst_layout, red_layout, analyzer);
      auto require_init = plan.require_init;
      auto clear_buffer = plan.clear_buffer;
      auto need_duplicate = plan.need_duplicate;
      auto need_update = plan.need_update;

      // A bf16 accumulator/destination reduce
      // runs entirely in fp32. MSL has no bf16 min/max overloads and the
      // codegen cannot print a bf16 infinity literal, so a bf16
      // accumulator/scratch cannot compile. Mirror the upstream CUDA
      // AccType design (fp16/bf16 -> fp32 accumulation): all phase-1
      // partials and the threadgroup scratch live in fp32; bf16 is only
      // materialized at the final ownership write-back (cast).
      const bool bf16_accum = clear_buffer->dtype.is_bfloat16();
      if (bf16_accum &&
          (op.type->IsBitAnd() || op.type->IsBitOr() || op.type->IsBitXor())) {
        LOG(FATAL) << "Metal reduce: bitwise reduce over a bfloat16 "
                      "accumulator/destination is not supported (an fp32 "
                      "accumulator would change the bit semantics of the "
                      "bfloat16 operands).";
      }
      // fp32 accumulator used by phases 1/2 (== clear_buffer unless a
      // private fp32 scratch buffer is required).
      Buffer accum_buffer = clear_buffer;
      bool alloc_accum_buffer = false;
      if (bf16_accum) {
        if (need_duplicate) {
          // The duplicated clear buffer is a private temp owned by this
          // lowering: redeclare it in fp32 and accumulate there.
          clear_buffer =
              decl_buffer(red_layout->OutputShape(), DataType::Float(32),
                          dst_buffer->name + "_clear",
                          GetPtrStorageScope(dst_buffer->data));
          accum_buffer = clear_buffer;
        } else {
          accum_buffer =
              decl_buffer(red_layout->OutputShape(), DataType::Float(32),
                          dst_buffer->name + "_accum",
                          GetPtrStorageScope(dst_buffer->data));
          alloc_accum_buffer = true;
        }
      }
      // Reduce identity in the accumulator dtype (fp32 for bf16 dst).
      PrimExpr init_value =
          bf16_accum ? PrimExpr(Cast(DataType::Float(32),
                                     backend::reduce::MakeInitValue(op)))
                     : backend::reduce::MakeInitValue(op);

      Array<PrimExpr> src_indice_compressed = reduce_plan.local_src_indices;
      Array<IterVar> src_var_compressed = reduce_plan.local_reduce_vars;

      // ---- Phase 1: per-thread local partials (vsize = 1). ----
      stmts.push_back(backend::reduce::MakeUnvectorizedLocalReduce(
          op, accum_buffer, src_buffer, src_indice_compressed,
          src_var_compressed, red_indices, init_value, require_init,
          need_duplicate, src_layout->OutputDim()));

      auto phase1 = stmts.size() > 1 ? SeqStmt(stmts) : stmts[0];
      phase1 = backend::reduce::MakeParallelPartitionLoop(
          phase1, dst_vars, lower_args.thread_index, lower_args.thread_bounds,
          analyzer, red_layout);

      // ---- Phase 2: lockstep threadgroup butterfly. ----
      // Faithful to the CUDA AllReduce template semantics: one step per
      // reduce-plan thread step; each step runs a XOR-butterfly over the
      // participating thread range [0, extent*scale) with offsets halving
      // down to `scale`, so replicated fragment holders land in distinct
      // groups and are never double-counted. All threads execute the same
      // barrier sequence (MSL has no CUDA-style named barriers);
      // non-participants contribute the reduce identity element, and their
      // scratch slots are never read (XOR partners stay inside the
      // participating range). The plan-level checks guarantee per
      // step nt = extent*scale is a power of two and <= 32 with the
      // threadgroup extent N % nt == 0, so [0, nt) is closed under every
      // mask (each mask flips a single lane bit below log2(nt)) and the
      // closure never leaves the thread's own simdgroup (group-local
      // closures); N % nt == 0 additionally makes every nt-block complete
      // (the raw butterfly runs on all N threads without a tid < nt
      // guard, so a partial tail block would read partners >= N or
      // never-written padding slots). Results round-trip through the
      // accumulator buffer between steps, mirroring the per-step extern
      // calls on CUDA.
      //
      // A raw extent=1 reduce has an empty
      // thread_steps plan (no thread-owned reduce factor), so the Phase-1
      // partials are already final and Phase 2 (butterfly + scratch +
      // barriers) is omitted entirely. No dummy loop / dummy barrier is
      // constructed; the accumulator buffer is only allocated when a
      // butterfly actually runs.
      Stmt butterfly_body;
      Buffer scratch;
      if (!reduce_plan.thread_steps.empty()) {
        // Scratch is sized by the full threadgroup extent (all threads
        // execute the barrier sequence and write their slot; group-local
        // closures in multi-simdgroup threadgroups are legal). The extent is
        // compile-time-constant by the entry check above. Requiring
        // N % nt == 0 per step ensures every read
        // partner of the all-thread butterfly stays within these N slots.
        const int64_t *thread_extent =
            as_const_int(lower_args.thread_bounds->extent);
        ICHECK(thread_extent != nullptr)
            << "Metal reduce: threadgroup extent must be a compile-time "
               "constant";
        int64_t total_elements = 1;
        for (const auto &s : dst_layout->InputShape()) {
          const int64_t *p = as_const_int(s);
          ICHECK(p != nullptr)
              << "Metal reduce: output shape must be compile-time constant";
          total_elements *= *p;
        }

        std::vector<int64_t> dst_strides(dst_dim, 1);
        for (int d = static_cast<int>(dst_dim) - 2; d >= 0; --d) {
          const int64_t *p = as_const_int(dst_layout->InputShape()[d + 1]);
          dst_strides[d] = dst_strides[d + 1] * *p;
        }

        // The threadgroup scratch uses the fp32 accumulator dtype for
        // bf16 destinations (bf16 scratch cannot compile: no MSL max/min
        // overloads, no bf16 infinity literal).
        scratch =
            decl_buffer({Integer(*thread_extent)}, accum_buffer->dtype,
                        accum_buffer->name + "_metal_reduce_scratch", "shared");

        // Build inside-out: single element iteration body.
        {
          Var e_var("e");
          Array<PrimExpr> elem_dst_indices;
          for (size_t d = 0; d < dst_dim; ++d) {
            elem_dst_indices.push_back(
                FloorMod(FloorDiv(e_var, Integer(dst_strides[d])),
                         dst_layout->InputShape()[d]));
          }
          auto elem_red_indices = red_layout->Forward(elem_dst_indices);

          // The scratch-write partial
          // and the step_store must be gated by "thread t holds a real
          // partial for element e", i.e. the red_layout partition guard —
          // the same predicate PartitionLoop generates for the generic
          // lowering. The dst-layout ownership predicate is NOT equivalent:
          // for layouts where every thread holds elements of every row
          // (e.g. [4,128] x 32 threads: 4 columns/thread/row), ownership
          // matches only a subset of the participating threads and the
          // butterfly silently drops the other partials (wrong totals).
          // Ownership still gates the final dst write-back (Phase 3 / copy
          // out).
          PrimExpr predicate = Bool(true);
          {
            PrimExpr local_thread_index = lower_args.thread_index;
            if (red_layout->ThreadRange().defined()) {
              local_thread_index =
                  local_thread_index - red_layout->ThreadRange()->min;
            }
            auto red_th = red_layout->Forward(elem_dst_indices);
            red_th.push_back(local_thread_index);
            auto inv = red_layout->Inverse()->Forward(red_th);
            inv.pop_back();
            for (size_t i = 0; i < static_cast<size_t>(dst_dim); ++i) {
              predicate = predicate && (inv[i] == elem_dst_indices[i]);
            }
            predicate = analyzer->Simplify(predicate);
          }

          auto sync_shared =
              Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                            {StringImm("shared")}));

          // Guarded scratch write: owner -> local partial, else identity.
          PrimExpr partial =
              analyzer->CanProve(predicate)
                  ? PrimExpr(BufferLoad(accum_buffer, elem_red_indices))
                  : PrimExpr(Select(predicate,
                                    BufferLoad(accum_buffer, elem_red_indices),
                                    init_value));

          Stmt elem_body;
          for (const auto &thread_step : reduce_plan.thread_steps) {
            int64_t nt =
                static_cast<int64_t>(thread_step.extent) * thread_step.scale;
            backend::reduce::CheckAllReduceWidth(
                static_cast<int>(nt), thread_step.scale, "tl.reduce (metal)");
            Stmt step_body = SeqStmt(
                {BufferStore(scratch, partial, {lower_args.thread_index}),
                 sync_shared});
            for (int64_t offset = nt / 2; offset >= thread_step.scale;
                 offset /= 2) {
              PrimExpr partner = lower_args.thread_index ^ Integer(offset);
              // No tid<nt guard: XOR partners self-organize inside the
              // participating block, and neutral values from non-
              // participants never leak into block slots. The plan checks
              // guarantee [0, nt) is closed under every mask (nt a power
              // of two <= 32) and that the threadgroup extent N is an
              // integer multiple of nt (N % nt == 0), so every
              // nt-block of the ALL-N-thread execution prefix is complete
              // and the partner of every participating lane is a
              // participating, written lane.
              Stmt combine = BufferStore(
                  scratch,
                  backend::reduce::MakeReduce(
                      op, 1, BufferLoad(scratch, {lower_args.thread_index}),
                      BufferLoad(scratch, {partner})),
                  {lower_args.thread_index});
              step_body = SeqStmt({step_body, combine, sync_shared});
            }
            // Every participant holds its group total; owners store it
            // back. With multiple
            // thread steps, ALL non-final steps round-trip through the fp32
            // accumulator so the next step's `partial` reloads the updated
            // group totals (per-step bf16 casts to dst silently dropped
            // every intermediate result: the next step re-read the stale
            // Phase-1 partial -> deterministic wrong totals). bf16 is
            // materialized only at the LAST step's ownership write-back
            // (cast fp32 -> bf16); a single-step plan is unchanged (its
            // only step is final).
            const bool is_final_step =
                (&thread_step == &reduce_plan.thread_steps.back());
            Stmt step_store;
            if (bf16_accum && !need_duplicate && is_final_step) {
              step_store = IfThenElse(
                  predicate,
                  BufferStore(
                      dst_buffer,
                      Cast(dst_buffer->dtype,
                           BufferLoad(scratch, {lower_args.thread_index})),
                      elem_red_indices));
            } else {
              step_store = IfThenElse(
                  predicate,
                  BufferStore(accum_buffer,
                              BufferLoad(scratch, {lower_args.thread_index}),
                              elem_red_indices));
            }
            step_body = SeqStmt({step_body, step_store});
            elem_body = elem_body.defined() ? SeqStmt({elem_body, step_body})
                                            : step_body;
          }

          butterfly_body = For(e_var, 0, Integer(total_elements),
                               ForKind::kSerial, elem_body, std::nullopt);
        }
      }

      // ---- Phase 3: duplicate-buffer update (guarded). ----
      Stmt phase3;
      if (need_duplicate) {
        // The clear=False update for a bf16 destination is computed in fp32
        // (accumulator dtype) and only the final store casts back to bf16.
        phase3 = backend::reduce::MakeDuplicateUpdatePhase(
            op, dst_buffer, accum_buffer, dst_indices, red_indices, dst_vars,
            dst_layout, need_update, /*cast_to_dst=*/bf16_accum,
            /*cast_dst_load_to_accum=*/bf16_accum, lower_args.thread_index,
            lower_args.thread_bounds, analyzer, red_layout);
      }

      // ---- Phase 2.5: bf16 empty-plan write-back. ----
      // With an empty thread_steps plan (raw extent=1) the butterfly is
      // skipped, so no step_store exists to materialize bf16. Phase 1 then
      // left the fp32 partials in the private accum buffer; write them back
      // to dst under the same element-ownership predicate as Phase 1.
      Stmt bf16_writeback;
      if (bf16_accum && !need_duplicate && reduce_plan.thread_steps.empty()) {
        Stmt wb = BufferStore(
            dst_buffer,
            Cast(dst_buffer->dtype, BufferLoad(accum_buffer, red_indices)),
            red_indices);
        bf16_writeback = backend::reduce::MakeParallelPartitionLoop(
            wb, dst_vars, lower_args.thread_index, lower_args.thread_bounds,
            analyzer, red_layout);
      }

      Array<Stmt> parts;
      parts.push_back(phase1);
      if (bf16_writeback.defined()) {
        parts.push_back(bf16_writeback);
      }
      if (butterfly_body.defined()) {
        parts.push_back(butterfly_body);
      }
      if (phase3.defined()) {
        parts.push_back(phase3);
      }
      Stmt body = parts.size() > 1 ? SeqStmt(parts) : parts[0];
      if (need_duplicate) {
        body = SeqStmt({AllocBuffer(clear_buffer), body});
      }
      if (alloc_accum_buffer) {
        body = SeqStmt({AllocBuffer(accum_buffer), body});
      }
      if (scratch.defined()) {
        body = SeqStmt({AllocBuffer(scratch), body});
      }
      return body;
    }

    LOG(FATAL) << "Metal reduce for buffers in scope (" << op.src.scope()
               << ", " << op.dst.scope() << ") is not implemented.";
    return Stmt();
  }
};

} // namespace metal

namespace {

bool MatchMetalReduceTarget(Target target) { return TargetIsMetal(target); }

bool RegisterMetalReduce() {
  RegisterReduceImpl(ReduceImpl{
      "metal.Reduce",
      MatchMetalReduceTarget,
      metal::MetalReduce::Lower,
  });
  return true;
}

const bool metal_reduce_registered = RegisterMetalReduce();

} // namespace

} // namespace tl
} // namespace tvm
