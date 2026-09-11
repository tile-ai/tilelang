/*!
 * \file tl/ascend/op/finalize_reducer.cc
 * \brief Ascend implementation for tl.finalize_reducer AllReduce lowering.
 */

#include "backend/common/op/reduce.h"
#include "backend/common/target_utils.h"
#include "op/reducer.h"
#include "support/check.h"

#include <tvm/tirx/builtin.h>

#include <array>
#include <cstdint>
#include <sstream>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace ascend {

Stmt LowerFinalizeReducer(const FinalizeReducerOpNode &op,
                          const LowerArgs &lower_args, arith::Analyzer *) {
  ICHECK(op.op == ReducerV2OpType::kSum || op.op == ReducerV2OpType::kMax ||
         op.op == ReducerV2OpType::kMin)
      << "Ascend Reducer v2 currently supports sum, max, and min only; "
         "bitand, bitor, and bitxor are not supported by the Ascend "
         "collective backend.";

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

  std::array op_names{"tl::SumOp", "tl::MaxOp", "tl::MinOp"};
  auto op_str = op_names[static_cast<int>(op.op)];

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

  auto thread_offset = lower_args.thread_bounds->min;
  Array<Stmt> step_stmts;
  for (const auto &[reducing_threads, scale] : steps) {
    backend::reduce::CheckAllReduceWidth(reducing_threads, scale,
                                         "tl.finalize_reducer");

    std::stringstream ss;
    ss << "tl::AscendAllReduce<" << op_str << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset << ">::run";

    Array<PrimExpr> thread_reduce_args = {StringImm(ss.str()),
                                          BufferLoad(buffer, indices_0)};
    // With a validated scale, only power-of-two widths <= 32 need no workspace.
    if (reducing_threads > 32 ||
        (reducing_threads & (reducing_threads - 1)) != 0) {
      PrimExpr workspace = lower_args.add_workspace(
          *as_const_int(lower_args.thread_bounds->extent), buffer->dtype);
      thread_reduce_args.push_back(workspace);
    }
    auto call = Call(buffer->dtype, builtin::call_extern(), thread_reduce_args);
    Stmt body = BufferStore(buffer, call, indices_0);
    for (int i = layout->OutputDim() - 1; i >= 0; --i) {
      body = For(indices_0[i].as<Var>().value(), 0, layout->OutputShape()[i],
                 ForKind::kParallel, body);
    }
    step_stmts.push_back(body);
  }

  if (op.seed.defined()) {
    Stmt body = BufferStore(
        buffer,
        ReducerV2Combine(op.op, BufferLoad(buffer, indices_0), op.seed.value()),
        indices_0);
    for (int i = layout->OutputDim() - 1; i >= 0; --i) {
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

} // namespace ascend

namespace {

bool MatchAscendFinalizeReducerTarget(Target target) {
  return TargetIsAscend(target);
}

bool RegisterAscendFinalizeReducer() {
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{
      "ascend.FinalizeReducer",
      MatchAscendFinalizeReducerTarget,
      ascend::LowerFinalizeReducer,
  });
  return true;
}

const bool ascend_finalize_reducer_registered = RegisterAscendFinalizeReducer();

} // namespace

} // namespace tl
} // namespace tvm
