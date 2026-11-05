/*!
 * \file tl/backend/common/op/finalize_reducer.h
 * \brief Shared tl.finalize_reducer lowering for GPU backends.
 */

#ifndef TVM_TL_BACKEND_COMMON_OP_FINALIZE_REDUCER_H_
#define TVM_TL_BACKEND_COMMON_OP_FINALIZE_REDUCER_H_

#include "backend/common/op/reduce.h"
#include "op/reducer.h"
#include "support/check.h"

#include <tvm/tirx/op/math.h>
#include <tvm/tirx/op/memory.h>

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace backend {

using namespace tirx;
using namespace prim;
using namespace ffi;

template <typename Impl> struct FinalizeReducerLowerer {
  static Stmt Lower(const FinalizeReducerOpNode &op,
                    const LowerArgs &lower_args, sym::AnalyzerObj *) {
    auto buffer = lower_args.buffer_remap[op.reducer];
    auto opt_layout = lower_args.layout_map.Get(op.reducer);
    ICHECK(opt_layout);
    ICHECK(opt_layout->as<Fragment>());
    auto layout = opt_layout->as<Fragment>().value();
    auto make_indices = [&]() -> Array<PrimExpr> {
      Array<PrimExpr> indices;
      indices.reserve(layout->OutputDim());
      for (int i = 0; i < layout->OutputDim(); ++i) {
        indices.push_back(PrimVar("__finred_" + std::to_string(i),
                                  layout->OutputShape()[i].ty()));
      }
      return indices;
    };

    // Collective steps: explicit narrow plan when present, otherwise the
    // legacy wide plan (one participant-extent step derived from the
    // storage layout's replicate extent).
    std::vector<std::pair<int, int>> steps;
    if (op.explicit_plan) {
      for (size_t i = 0; i + 1 < op.plan_steps.size(); i += 2) {
        steps.emplace_back(static_cast<int>(op.plan_steps[i]),
                           static_cast<int>(op.plan_steps[i + 1]));
      }
    } else {
      const auto p_extent = TryConstInt(layout->ReplicateExtent());
      ICHECK(p_extent);
      int extent = *p_extent;
      ICHECK(extent == 1 ||
             extent == *TryConstInt(lower_args.thread_bounds->extent))
          << "Illegal finalize_reducer: extent=" << extent
          << "; T.thread_bounds=" << lower_args.thread_bounds;

      if (extent > 1) {
        steps.emplace_back(extent, 1);
      }
    }

    std::array op_names{"tl::SumOp",    "tl::MaxOp",   "tl::MinOp",
                        "tl::BitAndOp", "tl::BitOrOp", "tl::BitXorOp"};
    auto op_str = op_names[static_cast<int>(op.op)];
    auto thread_offset = lower_args.thread_bounds->begin;

    int64_t layout_batch_size = 1;
    for (int i = 0; i < layout->OutputDim(); ++i) {
      const auto p = TryConstInt(layout->OutputShape()[i]);
      if (p == std::nullopt) {
        layout_batch_size = -1;
        break;
      }
      layout_batch_size *= *p;
    }

    int64_t effective_batch = static_cast<int64_t>(op.batch);

    if (effective_batch > 1 && layout_batch_size > 0) {
      ICHECK_LE(effective_batch, layout_batch_size)
          << "finalize_reducer: batch (" << effective_batch
          << ") exceeds total output elements (" << layout_batch_size << ")";
      ICHECK_EQ(layout_batch_size % effective_batch, 0)
          << "finalize_reducer: batch (" << effective_batch
          << ") must evenly divide total output elements (" << layout_batch_size
          << ")";
    }

    Array<Stmt> step_stmts;
    for (const auto &[reducing_threads, scale] : steps) {
      reduce::CheckAllReduceWidth(reducing_threads, scale,
                                  "tl.finalize_reducer");

      bool use_batch = effective_batch > 1 && layout_batch_size > 0 &&
                       reducing_threads > Impl::WarpSize(lower_args.target);

      if (use_batch) {
        int workspace_stride =
            static_cast<int>(*TryConstInt(lower_args.thread_bounds->extent));
        std::string allreduce = Impl::MakeBatchAllReduce(
            op_str, reducing_threads, scale, thread_offset,
            lower_args.thread_bounds->extent, static_cast<int>(effective_batch),
            workspace_stride, lower_args.target);
        int ws_size = workspace_stride * static_cast<int>(effective_batch);
        Expr workspace = lower_args.add_workspace(ws_size, buffer.dtype());

        // The remapped tensor has the committed per-thread physical shape.
        // Keep its identity in the address operand instead of constructing
        // an unbound flattened view with an independent type.
        for (int64_t offset = 0; offset < layout_batch_size;
             offset += effective_batch) {
          Expr ptr = MakeTensorAccessPtr(
              TensorLoad(buffer,
                         ReductionPhysicalIndices(
                             IntImm(PrimType::Int(32), offset), buffer)),
              IntImm(PrimType::Int(32), effective_batch), 3);
          Array<Expr> args = {StringImm(allreduce), ptr, workspace};
          step_stmts.push_back(
              Evaluate(Call(PrimType::Void(), call_extern_op(), args)));
        }
        continue;
      }

      std::string allreduce = Impl::MakeScalarAllReduce(
          op_str, reducing_threads, scale, thread_offset,
          lower_args.thread_bounds->extent, lower_args.target);
      // Native definition identities are unique across the whole Function,
      // including sibling collective phases and the trailing seed loop.
      const Array<PrimExpr> indices_0 = make_indices();
      Array<Expr> thread_reduce_args = {StringImm(allreduce),
                                        TensorLoad(buffer, indices_0)};
      if (reducing_threads > Impl::WarpSize(lower_args.target)) {
        Expr workspace = lower_args.add_workspace(
            *TryConstInt(lower_args.thread_bounds->extent), buffer.dtype());
        thread_reduce_args.push_back(workspace);
      }
      auto call = Call(buffer.dtype(), call_extern_op(), thread_reduce_args);
      Stmt body = TensorStore(buffer, indices_0, call.as_or_throw<PrimExpr>());

      for (int i = layout->OutputDim() - 1; i >= 0; i--) {
        body = For(indices_0[i].as<PrimVar>().value(),
                   IntImm(indices_0[i].as<PrimVar>().value().ty(), 0),
                   layout->OutputShape()[i],
                   IntImm(indices_0[i].as<PrimVar>().value().ty(), 1),
                   ForKind::kParallel, SeqStmt(body));
      }
      step_stmts.push_back(body);
    }

    // Optional logical seed: after the collective every physical slot holds
    // the final combined value (replicas are equal), so combining the seed
    // uniformly into each slot applies it exactly once per logical output.
    if (op.seed.has_value()) {
      const Array<PrimExpr> indices_0 = make_indices();
      Stmt body =
          TensorStore(buffer, indices_0,
                      ReducerV2Combine(op.op, TensorLoad(buffer, indices_0),
                                       op.seed.value()));
      for (int i = layout->OutputDim() - 1; i >= 0; i--) {
        body = For(indices_0[i].as<PrimVar>().value(),
                   IntImm(indices_0[i].as<PrimVar>().value().ty(), 0),
                   layout->OutputShape()[i],
                   IntImm(indices_0[i].as<PrimVar>().value().ty(), 1),
                   ForKind::kParallel, SeqStmt(body));
      }
      step_stmts.push_back(body);
    }

    if (step_stmts.empty()) {
      return Evaluate(0);
    }
    return step_stmts.size() == 1 ? step_stmts[0] : SeqStmt(step_stmts);
  }
};

} // namespace backend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_OP_FINALIZE_REDUCER_H_
