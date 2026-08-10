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

    Buffer destination = get_buffer(op.destination);
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
    ICHECK_EQ(op.partials.size(), op.partial_plans.size());

    auto make_logical_vars = [](const Array<PrimExpr> &shape,
                                const std::string &prefix) {
      Array<IterVar> variables;
      variables.reserve(shape.size());
      for (size_t i = 0; i < shape.size(); ++i) {
        Var var(prefix + std::to_string(i));
        variables.push_back(
            IterVar(Range(0, shape[i]), var, IterVarType::kDataPar));
      }
      return variables;
    };

    auto variable_exprs = [](const Array<IterVar> &variables) {
      return variables.Map(
          [](const IterVar &variable) { return PrimExpr(variable->var); });
    };

    auto partition_loop = [&](Stmt body, const Array<IterVar> &variables,
                              const Fragment &layout) {
      ICHECK(!variables.empty());
      for (int i = static_cast<int>(variables.size()) - 1; i >= 0; --i) {
        body = For(variables[i]->var, 0, variables[i]->dom->extent,
                   ForKind::kParallel, body);
      }
      body = PartitionLoop(Downcast<For>(body), lower_args.thread_index,
                           analyzer, layout);
      return PragmaUnrollLoop(Downcast<For>(body));
    };

    auto destination_owner = [&](const Array<PrimExpr> &logical_indices,
                                 const Array<PrimExpr> &physical_indices) {
      PrimExpr local_thread = lower_args.thread_index;
      if (destination_layout->ThreadRange().defined()) {
        local_thread = local_thread - destination_layout->ThreadRange()->min;
      }
      Array<PrimExpr> inverse_args = physical_indices;
      inverse_args.push_back(local_thread);
      Array<PrimExpr> inverse =
          destination_layout->Inverse()->Forward(inverse_args);
      ICHECK_GE(inverse.size(), logical_indices.size());
      PrimExpr owns_result = Bool(true);
      for (size_t i = 0; i < logical_indices.size(); ++i) {
        owns_result = And(owns_result, inverse[i] == logical_indices[i]);
      }
      return analyzer->Simplify(owns_result);
    };

    auto combine_destination = [&](const Array<PrimExpr> &logical_indices,
                                   const PrimExpr &value,
                                   const PrimExpr &predicate,
                                   const std::string &name) {
      Array<PrimExpr> destination_indices =
          destination_layout->Forward(logical_indices);
      PrimExpr owns_result = And(
          predicate, destination_owner(logical_indices, destination_indices));
      Var result(name, destination->dtype);
      PrimExpr combined = MakeReduceCombine(
          op.combine_type, BufferLoad(destination, destination_indices),
          result);
      Stmt store = BufferStore(destination, combined, destination_indices);
      if (!analyzer->CanProve(owns_result)) {
        store = IfThenElse(owns_result, store);
      }
      return SeqStmt({Bind(result, value), store});
    };

    Array<Stmt> statements;
    Array<IterVar> destination_vars = make_logical_vars(
        destination_layout->InputShape(), "reducer_destination_init_");
    Array<PrimExpr> destination_logical = variable_exprs(destination_vars);
    Array<PrimExpr> destination_physical =
        destination_layout->Forward(destination_logical);
    Stmt initialize_destination = BufferStore(
        destination, MakeReduceIdentity(op.combine_type, destination->dtype),
        destination_physical);
    statements.push_back(partition_loop(initialize_destination,
                                        destination_vars, destination_layout));

    for (size_t group = 0; group < op.partials.size(); ++group) {
      Buffer partial = get_buffer(op.partials[group]);
      const ReducerPartialPlan &plan = op.partial_plans[group];
      ICHECK(IsLocalBuffer(partial))
          << "T.finalize_reducer expects local partial storage, got "
          << partial.scope();

      if (plan->canonical) {
        ICHECK(!plan->partial_layout.defined());
        ICHECK_EQ(partial->shape.size(), op.destination->shape.size());
        Array<IterVar> logical_vars = make_logical_vars(
            op.destination->shape,
            "reducer_canonical_" + std::to_string(group) + "_");
        Array<PrimExpr> logical_indices = variable_exprs(logical_vars);
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
            args.push_back(lower_args.add_workspace(
                static_cast<int>(*participant_extent), partial->dtype));
          }
          reduced = Call(partial->dtype, builtin::call_extern(), args);
        }
        Stmt body = combine_destination(logical_indices, reduced, Bool(true),
                                        "reducer_canonical_result_" +
                                            std::to_string(group));
        for (int i = static_cast<int>(logical_vars.size()) - 1; i >= 0; --i) {
          body = For(logical_vars[i]->var, 0, logical_vars[i]->dom->extent,
                     ForKind::kSerial, body);
        }
        statements.push_back(std::move(body));
        continue;
      }

      ICHECK(plan->partial_layout.defined());
      Fragment partial_layout = plan->partial_layout.value();
      ICHECK_EQ(plan->step_extents.size(), plan->step_scales.size());
      ICHECK_EQ(partial->shape.size(), partial_layout->OutputShape().size());
      // Execute every collective uniformly across the participant scope. A
      // projected layout may cover fewer than a warp of physical threads; a
      // PartitionLoop guard around the call would then make CUDA warp shuffles
      // observe an incomplete active mask. Iterate physical local slots
      // instead, recover the logical output through the inverse layout, and
      // feed identity from threads outside the projected image.
      Array<IterVar> physical_vars = make_logical_vars(
          partial->shape, "reducer_projected_" + std::to_string(group) + "_");
      Array<PrimExpr> partial_indices = variable_exprs(physical_vars);
      PrimExpr local_thread = lower_args.thread_index;
      PrimExpr valid = Bool(true);
      if (partial_layout->ThreadRange().defined()) {
        Range thread_range = partial_layout->ThreadRange();
        valid = And(valid, local_thread >= thread_range->min);
        valid =
            And(valid, local_thread < thread_range->min + thread_range->extent);
        local_thread = local_thread - thread_range->min;
      }
      Array<PrimExpr> inverse_args = partial_indices;
      inverse_args.push_back(local_thread);
      Array<PrimExpr> inverse =
          partial_layout->Inverse()->Forward(inverse_args);
      ICHECK_GE(inverse.size(), partial_layout->InputDim());
      Array<PrimExpr> logical_indices;
      logical_indices.reserve(partial_layout->InputDim());
      for (size_t i = 0; i < partial_layout->InputDim(); ++i) {
        PrimExpr index = inverse[i];
        logical_indices.push_back(index);
        valid = And(valid, index >= make_zero(index.dtype()));
        valid = And(valid, index < partial_layout->InputShape()[i]);
      }
      if (inverse.size() > partial_layout->InputDim()) {
        PrimExpr replica = inverse[partial_layout->InputDim()];
        valid = And(valid, replica >= make_zero(replica.dtype()));
        valid = And(valid, replica < partial_layout->ReplicateExtent());
      }
      valid = analyzer->Simplify(valid);
      PrimExpr identity = MakeReduceIdentity(op.combine_type, partial->dtype);
      PrimExpr reduced = BufferLoad(partial, partial_indices);
      if (!analyzer->CanProve(valid)) {
        reduced = Select(valid, reduced, identity);
      }
      for (size_t step_index = 0; step_index < plan->step_extents.size();
           ++step_index) {
        int extent = static_cast<int>(plan->step_extents[step_index]->value);
        int scale = static_cast<int>(plan->step_scales[step_index]->value);
        int reducing_threads = extent * scale;
        reduce::CheckAllReduceWidth(reducing_threads, scale,
                                    "tl.finalize_reducer");
        std::string allreduce = Impl::MakeScalarAllReduce(
            ReduceCodegenName(op.combine_type), reducing_threads, scale,
            lower_args.thread_bounds->min, lower_args.thread_bounds->extent,
            lower_args.target);
        Array<PrimExpr> args = {StringImm(allreduce), reduced};
        if (reducing_threads > Impl::WarpSize(lower_args.target)) {
          args.push_back(lower_args.add_workspace(
              static_cast<int>(*participant_extent), partial->dtype));
        }
        reduced = Call(partial->dtype, builtin::call_extern(), args);
      }
      Stmt body = combine_destination(logical_indices, reduced, valid,
                                      "reducer_projected_result_" +
                                          std::to_string(group));
      for (int i = static_cast<int>(physical_vars.size()) - 1; i >= 0; --i) {
        body = For(physical_vars[i]->var, 0, physical_vars[i]->dom->extent,
                   ForKind::kSerial, body);
      }
      statements.push_back(PragmaUnrollLoop(Downcast<For>(body)));
    }

    if (op.seed.defined()) {
      Array<IterVar> seed_vars = make_logical_vars(
          destination_layout->InputShape(), "reducer_destination_seed_");
      Array<PrimExpr> seed_logical = variable_exprs(seed_vars);
      Array<PrimExpr> seed_physical = destination_layout->Forward(seed_logical);
      Stmt apply_seed =
          BufferStore(destination,
                      MakeReduceCombine(op.combine_type,
                                        BufferLoad(destination, seed_physical),
                                        op.seed.value()),
                      seed_physical);
      statements.push_back(
          partition_loop(apply_seed, seed_vars, destination_layout));
    }

    // `batch` remains a non-semantic hint. Each physical partial group uses
    // scalar collectives until a group-aware batched lowering is proven.
    return SeqStmt(statements);
  }
};

} // namespace backend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_OP_FINALIZE_REDUCER_H_
