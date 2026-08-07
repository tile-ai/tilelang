/*!
 * \file tl/backend/common/op/finalize_reducer.h
 * \brief Shared out-of-place deferred reducer lowering for GPU backends.
 */

#ifndef TVM_TL_BACKEND_COMMON_OP_FINALIZE_REDUCER_H_
#define TVM_TL_BACKEND_COMMON_OP_FINALIZE_REDUCER_H_

#include "backend/common/op/reduce.h"
#include "op/deferred_reducer.h"
#include "op/utils.h"
#include "support/check.h"

#include <tvm/tirx/builtin.h>

#include <string>

namespace tvm {
namespace tl {
namespace backend {

using namespace tirx;

template <typename Impl> struct FinalizeReducerLowerer {
  static Stmt Lower(const FinalizeReducerOpNode &op,
                    const LowerArgs &lower_args, arith::Analyzer *analyzer) {
    auto get_buffer = [&](const Buffer &buffer) {
      auto it = lower_args.buffer_remap.find(buffer);
      return it == lower_args.buffer_remap.end() ? buffer : (*it).second;
    };

    Buffer partial = get_buffer(op.reducer);
    Buffer destination = get_buffer(op.destination);
    ICHECK(IsLocalBuffer(partial))
        << "T.finalize_reducer expects materialized local partial storage, got "
        << partial.scope();
    ICHECK(IsFragmentBuffer(op.destination))
        << "T.finalize_reducer currently requires a local.fragment "
           "destination, "
           "got "
        << op.destination.scope();

    Optional<Layout> destination_layout_ref =
        lower_args.layout_map.Get(op.destination);
    ICHECK(destination_layout_ref.defined())
        << "T.finalize_reducer destination layout was not inferred";
    Fragment destination_layout =
        Downcast<Fragment>(destination_layout_ref.value());

    const int64_t *participant_min =
        as_const_int(lower_args.thread_bounds->min);
    const int64_t *participant_extent =
        as_const_int(lower_args.thread_bounds->extent);
    ICHECK(participant_min != nullptr && participant_extent != nullptr)
        << "T.finalize_reducer requires a compiler-known contiguous "
           "participant "
           "Range, got "
        << lower_args.thread_bounds;
    ICHECK_GT(*participant_extent, 0);

    if (op.local_complete_layout.defined()) {
      Fragment planned_layout = op.local_complete_layout.value();
      ICHECK(planned_layout->IsEqual(destination_layout.get()))
          << "LocalComplete reducer plan no longer matches the destination "
             "layout";
      ICHECK_EQ(partial->shape.size(), planned_layout->OutputShape().size());
      ICHECK_EQ(destination->shape.size(),
                planned_layout->OutputShape().size());
      for (size_t i = 0; i < partial->shape.size(); ++i) {
        ICHECK(analyzer->CanProveEqual(partial->shape[i],
                                       planned_layout->OutputShape()[i]) &&
               analyzer->CanProveEqual(destination->shape[i],
                                       planned_layout->OutputShape()[i]))
            << "LocalComplete reducer and destination physical shapes must "
               "match the planned layout: "
            << partial->shape << " vs. " << destination->shape << " vs. "
            << planned_layout->OutputShape();
      }

      Array<Var> physical_vars;
      Array<PrimExpr> physical_indices;
      physical_vars.reserve(partial->shape.size());
      physical_indices.reserve(partial->shape.size());
      for (size_t i = 0; i < partial->shape.size(); ++i) {
        Var var("reducer_finalize_local_" + std::to_string(i));
        physical_vars.push_back(var);
        physical_indices.push_back(var);
      }

      PrimExpr result = BufferLoad(partial, physical_indices);
      if (op.seed.defined()) {
        result = MakeReduceCombine(op.combine_type, result, op.seed.value());
      }
      Stmt body = BufferStore(destination, result, physical_indices);
      for (int i = static_cast<int>(physical_vars.size()) - 1; i >= 0; --i) {
        body =
            For(physical_vars[i], 0, partial->shape[i], ForKind::kSerial, body);
      }
      return body;
    }

    ICHECK_EQ(partial->shape.size(), op.destination->shape.size())
        << "Reducer and destination ranks must match";
    for (size_t i = 0; i < partial->shape.size(); ++i) {
      ICHECK(
          analyzer->CanProveEqual(partial->shape[i], op.destination->shape[i]))
          << "Reducer and destination shapes must match: " << partial->shape
          << " vs. " << op.destination->shape;
    }

    Array<Var> logical_vars;
    Array<PrimExpr> logical_indices;
    logical_vars.reserve(partial->shape.size());
    logical_indices.reserve(partial->shape.size());
    for (size_t i = 0; i < partial->shape.size(); ++i) {
      Var var("reducer_finalize_" + std::to_string(i));
      logical_vars.push_back(var);
      logical_indices.push_back(var);
    }

    PrimExpr reduced = BufferLoad(partial, logical_indices);
    if (*participant_extent > 1) {
      reduce::CheckAllReduceWidth(static_cast<int>(*participant_extent), 1,
                                  "tl.finalize_reducer");
      std::string allreduce = Impl::MakeScalarAllReduce(
          ReduceCodegenName(op.combine_type),
          static_cast<int>(*participant_extent), 1,
          lower_args.thread_bounds->min, lower_args.thread_bounds->extent,
          lower_args.target);
      Array<PrimExpr> args = {StringImm(allreduce), reduced};
      if (*participant_extent > Impl::WarpSize(lower_args.target)) {
        PrimExpr workspace = lower_args.add_workspace(
            static_cast<int>(*participant_extent), partial->dtype);
        args.push_back(workspace);
      }
      reduced = Call(partial->dtype, builtin::call_extern(), args);
    }
    if (op.seed.defined()) {
      reduced = MakeReduceCombine(op.combine_type, reduced, op.seed.value());
    }

    Array<PrimExpr> destination_indices =
        destination_layout->Forward(logical_indices);
    PrimExpr local_thread = lower_args.thread_index;
    if (destination_layout->ThreadRange().defined()) {
      local_thread = local_thread - destination_layout->ThreadRange()->min;
    }
    Array<PrimExpr> inverse_args = destination_indices;
    inverse_args.push_back(local_thread);
    Array<PrimExpr> inverse =
        destination_layout->Inverse()->Forward(inverse_args);
    ICHECK_GE(inverse.size(), logical_indices.size());
    PrimExpr owns_result = Bool(true);
    for (size_t i = 0; i < logical_indices.size(); ++i) {
      owns_result = And(owns_result, inverse[i] == logical_indices[i]);
    }
    owns_result = analyzer->Simplify(owns_result);

    Var result("reducer_result", partial->dtype);
    Stmt store = BufferStore(destination, result, destination_indices);
    if (!analyzer->CanProve(owns_result)) {
      store = IfThenElse(owns_result, store);
    }
    Stmt body = SeqStmt({Bind(result, reduced), store});
    for (int i = static_cast<int>(logical_vars.size()) - 1; i >= 0; --i) {
      body = For(logical_vars[i], 0, partial->shape[i], ForKind::kSerial, body);
    }

    // `batch` is a non-semantic hint. The v2 correctness baseline deliberately
    // falls back to scalar collectives until batched full-array lowering is
    // proven equivalent.
    return body;
  }
};

} // namespace backend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_OP_FINALIZE_REDUCER_H_
