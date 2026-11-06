/*!
 * \file tl/ascend/op/finalize_reducer.cc
 * \brief Ascend implementation for tl.finalize_reducer AllReduce lowering.
 */

#include "ascend/op/reduce.h"
#include "backend/common/op/reduce.h"
#include "backend/common/target_utils.h"
#include "layout/expr_utils.h"
#include "op/reducer.h"
#include "support/check.h"

#include <tvm/tirx/op/math.h>
#include <tvm/tirx/op/memory.h>

#include <array>
#include <cstdint>
#include <sstream>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;

namespace ascend {

Stmt LowerFinalizeReducer(const FinalizeReducerOpNode &op,
                          const LowerArgs &lower_args, sym::AnalyzerObj *) {
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
  auto make_indices = [&]() -> Array<PrimExpr> {
    Array<PrimExpr> indices;
    for (size_t i = 0; i < layout->OutputDim(); ++i) {
      indices.push_back(PrimVar("__finred_" + std::to_string(i),
                                layout->OutputShape()[i].ty()));
    }
    return indices;
  };

  std::vector<std::pair<int, int>> steps;
  if (op.explicit_plan) {
    for (size_t i = 0; i + 1 < op.plan_steps.size(); i += 2) {
      steps.emplace_back(static_cast<int>(op.plan_steps[i]),
                         static_cast<int>(op.plan_steps[i + 1]));
    }
  } else {
    auto p_extent = TryConstInt(layout->ReplicateExtent());
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

  std::array op_names{"tl::SumOp", "tl::MaxOp", "tl::MinOp"};
  auto op_str = op_names[static_cast<int>(op.op)];

  int64_t layout_batch_size = 1;
  for (size_t i = 0; i < layout->OutputDim(); ++i) {
    auto p = TryConstInt(layout->OutputShape()[i]);
    if (!p.has_value()) {
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

  auto thread_offset = lower_args.thread_bounds->begin;
  Array<Stmt> step_stmts;
  for (const auto &[reducing_threads, scale] : steps) {
    // Same policy as ascend::Reduce::CheckAllReduceWidth: AscendAllReduce gates
    // its butterfly and hardware-reduce paths on a power-of-two thread count,
    // so the XOR-butterfly rule is vacuous and only the universal checks apply.
    ascend::CheckAllReduceWidth(reducing_threads, scale, "tl.finalize_reducer");

    std::stringstream ss;
    ss << "tl::AscendAllReduce<" << op_str << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset << ">::run";

    const Array<PrimExpr> indices_0 = make_indices();
    Array<Expr> thread_reduce_args = {StringImm(ss.str()),
                                      TensorLoad(buffer, indices_0)};
    // With a validated scale, only power-of-two widths <= 32 need no workspace.
    if (reducing_threads > 32 ||
        (reducing_threads & (reducing_threads - 1)) != 0) {
      Expr workspace = lower_args.add_workspace(
          *TryConstInt(lower_args.thread_bounds->extent), buffer.dtype());
      thread_reduce_args.push_back(workspace);
    }
    auto call = Call(buffer.dtype(), call_extern_op(), thread_reduce_args);
    Stmt body = TensorStore(buffer, indices_0, call.as_or_throw<PrimExpr>());
    for (int i = layout->OutputDim() - 1; i >= 0; --i) {
      body =
          For(indices_0[i].as_or_throw<PrimVar>(), IntImm(indices_0[i].ty(), 0),
              layout->OutputShape()[i], IntImm(indices_0[i].ty(), 1),
              ForKind::kParallel, SeqStmt(body));
    }
    step_stmts.push_back(body);
  }

  if (op.seed.has_value()) {
    const Array<PrimExpr> indices_0 = make_indices();
    Stmt body =
        TensorStore(buffer, indices_0,
                    ReducerV2Combine(op.op, TensorLoad(buffer, indices_0),
                                     op.seed.value()));
    for (int i = layout->OutputDim() - 1; i >= 0; --i) {
      body =
          For(indices_0[i].as_or_throw<PrimVar>(), IntImm(indices_0[i].ty(), 0),
              layout->OutputShape()[i], IntImm(indices_0[i].ty(), 1),
              ForKind::kParallel, SeqStmt(body));
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
