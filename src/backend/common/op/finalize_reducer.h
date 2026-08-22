/*!
 * \file tl/backend/common/op/finalize_reducer.h
 * \brief Shared tl.finalize_reducer lowering for GPU backends.
 */

#ifndef TVM_TL_BACKEND_COMMON_OP_FINALIZE_REDUCER_H_
#define TVM_TL_BACKEND_COMMON_OP_FINALIZE_REDUCER_H_

#include "backend/common/op/reduce.h"
#include "op/reducer.h"
#include "support/check.h"

#include <tvm/tirx/builtin.h>

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace backend {

using namespace tirx;
using namespace ffi;

template <typename Impl> struct FinalizeReducerLowerer {
  static Stmt Lower(const FinalizeReducerOpNode &op,
                    const LowerArgs &lower_args, arith::Analyzer *) {
    auto buffer = lower_args.buffer_remap[op.reducer];
    auto opt_layout = lower_args.layout_map.Get(op.reducer);
    ICHECK(opt_layout);
    ICHECK(opt_layout->as<Fragment>());
    auto layout = opt_layout->as<Fragment>().value();
    Array<PrimExpr> indices_0;
    indices_0.reserve(layout->OutputDim());
    for (int i = 0; i < layout->OutputDim(); ++i) {
      indices_0.push_back(Var("__finred_" + std::to_string(i)));
    }

    // Collective steps: explicit narrow plan when present, otherwise the
    // legacy wide plan (one participant-extent step derived from the
    // storage layout's replicate extent).
    std::vector<std::pair<int, int>> steps;
    if (op.explicit_plan) {
      for (size_t i = 0; i + 1 < op.plan_steps.size(); i += 2) {
        steps.emplace_back(static_cast<int>(op.plan_steps[i]->value),
                           static_cast<int>(op.plan_steps[i + 1]->value));
      }
    } else {
      const int64_t *p_extent = as_const_int(layout->ReplicateExtent());
      ICHECK(p_extent);
      int extent = *p_extent;
      ICHECK(extent == 1 ||
             extent == *as_const_int(lower_args.thread_bounds->extent))
          << "Illegal finalize_reducer: extent=" << extent
          << "; T.thread_bounds=" << lower_args.thread_bounds;

      if (extent > 1) {
        steps.emplace_back(extent, 1);
      }
    }

    std::array op_names{"tl::SumOp",    "tl::MaxOp",   "tl::MinOp",
                        "tl::BitAndOp", "tl::BitOrOp", "tl::BitXorOp"};
    auto op_str = op_names[static_cast<int>(op.op)];
    auto thread_offset = lower_args.thread_bounds->min;

    int64_t layout_batch_size = 1;
    for (int i = 0; i < layout->OutputDim(); ++i) {
      const int64_t *p = as_const_int(layout->OutputShape()[i]);
      if (p == nullptr) {
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
            static_cast<int>(*as_const_int(lower_args.thread_bounds->extent));
        std::string allreduce = Impl::MakeBatchAllReduce(
            op_str, reducing_threads, scale, thread_offset,
            lower_args.thread_bounds->extent, static_cast<int>(effective_batch),
            workspace_stride, lower_args.target);
        int ws_size = workspace_stride * static_cast<int>(effective_batch);
        PrimExpr workspace = lower_args.add_workspace(ws_size, buffer->dtype);

        std::vector<int64_t> shape_values;
        shape_values.reserve(layout->OutputDim());
        for (int i = 0; i < layout->OutputDim(); ++i) {
          shape_values.push_back(*as_const_int(layout->OutputShape()[i]));
        }
        std::vector<int64_t> strides(layout->OutputDim(), 1);
        for (int i = layout->OutputDim() - 2; i >= 0; --i) {
          strides[i] = strides[i + 1] * shape_values[i + 1];
        }

        int num_chunks = static_cast<int>(layout_batch_size / effective_batch);
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
          int64_t flat_offset = static_cast<int64_t>(chunk) * effective_batch;
          Array<PrimExpr> chunk_indices;
          for (int i = 0; i < layout->OutputDim(); ++i) {
            int64_t index = (flat_offset / strides[i]) % shape_values[i];
            chunk_indices.push_back(Integer(index));
          }
          PrimExpr ptr = Call(DataType::Handle(), builtin::address_of(),
                              {BufferLoad(buffer, chunk_indices)});
          Array<PrimExpr> args = {StringImm(allreduce), ptr, workspace};
          step_stmts.push_back(
              Evaluate(Call(DataType::Handle(), builtin::call_extern(), args)));
        }
        continue;
      }

      std::string allreduce = Impl::MakeScalarAllReduce(
          op_str, reducing_threads, scale, thread_offset,
          lower_args.thread_bounds->extent, lower_args.target);
      Array<PrimExpr> thread_reduce_args = {StringImm(allreduce),
                                            BufferLoad(buffer, indices_0)};
      if (reducing_threads > Impl::WarpSize(lower_args.target)) {
        PrimExpr workspace = lower_args.add_workspace(
            *as_const_int(lower_args.thread_bounds->extent), buffer->dtype);
        thread_reduce_args.push_back(workspace);
      }
      auto call =
          Call(buffer->dtype, builtin::call_extern(), thread_reduce_args);
      Stmt body = BufferStore(buffer, call, indices_0);

      for (int i = layout->OutputDim() - 1; i >= 0; i--) {
        body = For(indices_0[i].as<Var>().value(), 0, layout->OutputShape()[i],
                   ForKind::kParallel, body);
      }
      step_stmts.push_back(body);
    }

    // Optional logical seed: after the collective every physical slot holds
    // the final combined value (replicas are equal), so combining the seed
    // uniformly into each slot applies it exactly once per logical output.
    if (op.seed.defined()) {
      Stmt body =
          BufferStore(buffer,
                      ReducerV2Combine(op.op, BufferLoad(buffer, indices_0),
                                       op.seed.value()),
                      indices_0);
      for (int i = layout->OutputDim() - 1; i >= 0; i--) {
        body = For(indices_0[i].as<Var>().value(), 0, layout->OutputShape()[i],
                   ForKind::kParallel, body);
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
