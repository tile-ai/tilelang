#ifndef TVM_TL_BACKEND_COMMON_OP_FINALIZE_REDUCER_H_
#define TVM_TL_BACKEND_COMMON_OP_FINALIZE_REDUCER_H_
#include "op/prim.h"

#include "backend/common/op/reduce.h"
#include "compat/expr.h"
#include "op/reducer.h"

#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>

#include <array>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace tvm::tl::backend {
using namespace tirx;
using namespace ffi;

template <typename Impl> struct FinalizeReducerLowerer {
  static Stmt Lower(const FinalizeReducerOpNode &op, const LowerArgs &arguments,
                    sym::Analyzer analyzer) {
    TensorVar storage = arguments.buffer_remap.at(op.reducer);
    Fragment layout =
        arguments.layout_map.at(op.reducer).template as_or_throw<Fragment>();
    Array<PrimExpr> indices;
    for (int axis = 0; axis < layout->OutputDim(); ++axis) {
      indices.push_back(PrimVar("finalize_index_" + std::to_string(axis),
                                layout->OutputShape()[axis].ty()));
    }
    std::vector<std::pair<int, int>> steps;
    if (op.explicit_plan) {
      TVM_FFI_CHECK_EQ(op.plan_steps.size() % 2, 0, ValueError);
      for (size_t index = 0; index < op.plan_steps.size(); index += 2) {
        steps.emplace_back(
            static_cast<int64_t>(op.plan_steps[index]->value),
            static_cast<int64_t>(op.plan_steps[index + 1]->value));
      }
    } else {
      auto extent = GetConstInt(layout->ReplicateExtent());
      auto participants = GetConstInt(arguments.thread_bounds->extent);
      TVM_FFI_CHECK(extent && participants &&
                        (*extent == 1 || *extent == *participants),
                    ValueError)
          << "finalize_reducer requires either one replica or a replica per "
             "participant";
      if (*extent > 1)
        steps.emplace_back(*extent, 1);
    }
    constexpr std::array names{"tl::SumOp",    "tl::MaxOp",   "tl::MinOp",
                               "tl::BitAndOp", "tl::BitOrOp", "tl::BitXorOp"};
    const char *operation = names.at(static_cast<int>(op.op));
    PrimExpr offset = arguments.thread_bounds->begin;
    PrimExpr participants = arguments.thread_bounds->extent;
    int64_t elements = 1;
    for (const PrimExpr &dimension : layout->OutputShape()) {
      auto extent = GetConstInt(dimension);
      if (!extent) {
        elements = -1;
        break;
      }
      TVM_FFI_CHECK(*extent > 0 &&
                        elements <=
                            std::numeric_limits<int64_t>::max() / *extent,
                    ValueError)
          << "Invalid finalize_reducer physical shape";
      elements *= *extent;
    }
    int64_t batch = op.batch;
    TVM_FFI_CHECK(batch >= 1, ValueError)
        << "finalize_reducer batch must be positive";
    if (batch > 1 && elements > 0) {
      TVM_FFI_CHECK(batch <= elements, ValueError)
          << "finalize_reducer batch exceeds total output elements";
      TVM_FFI_CHECK(elements % batch == 0, ValueError)
          << "finalize_reducer batch must evenly divide total output elements";
    }
    auto loop_nest = [&](Stmt body) {
      Array<PrimVar> variables;
      Map<PrimVar, PrimExpr> substitutions;
      for (const PrimExpr &index : indices) {
        PrimVar original = index.as_or_throw<PrimVar>();
        PrimVar variable(original->name, original.ty());
        variables.push_back(variable);
        substitutions.Set(original, variable);
      }
      body = tl::Substitute(body, substitutions);
      for (int axis = layout->OutputDim() - 1; axis >= 0; --axis) {
        PrimType index_type = indices[axis].ty();
        body = For(variables[axis], tl::MakeConst(index_type, 0),
                   layout->OutputShape()[axis], tl::MakeConst(index_type, 1),
                   ForKind::kParallel, SeqStmt(body));
      }
      return body;
    };
    Array<Stmt> result;
    for (const auto &[width, scale] : steps) {
      reduce::CheckAllReduceWidth(width, scale, "tl.finalize_reducer");
      if (batch > 1 && elements > 0 &&
          width > Impl::WarpSize(arguments.target)) {
        auto stride = GetConstInt(participants);
        TVM_FFI_CHECK(stride && *stride > 0 &&
                          batch <= std::numeric_limits<int>::max() / *stride,
                      ValueError)
            << "finalize_reducer workspace requires static participant bounds";
        std::string helper = Impl::MakeBatchAllReduce(
            operation, width, scale, offset, participants, batch, *stride,
            arguments.target);
        Expr workspace =
            arguments.add_workspace(*stride * batch, storage.dtype());
        for (int64_t start = 0; start < elements; start += batch) {
          Array<PrimExpr> coordinates;
          int64_t remaining = start;
          for (int axis = layout->OutputDim() - 1; axis >= 0; --axis) {
            int64_t extent = *GetConstInt(layout->OutputShape()[axis]);
            coordinates.insert(coordinates.begin(),
                               IntImm::Int64(remaining % extent));
            remaining /= extent;
          }
          Expr pointer = Call(std::nullopt, address_of_op(),
                              {TensorLoad(storage, coordinates)});
          result.push_back(
              Evaluate(Call(PrimType::Void(), call_extern_op(),
                            {StringImm(helper), pointer, workspace})));
        }
      } else {
        std::string helper = Impl::MakeScalarAllReduce(
            operation, width, scale, offset, participants, arguments.target);
        Array<Expr> operands{StringImm(helper), TensorLoad(storage, indices)};
        if (width > Impl::WarpSize(arguments.target)) {
          auto count = GetConstInt(participants);
          TVM_FFI_CHECK(count && *count > 0, ValueError)
              << "finalize_reducer workspace requires static participant "
                 "bounds";
          operands.push_back(arguments.add_workspace(*count, storage.dtype()));
        }
        PrimExpr value = Call(storage.dtype(), call_extern_op(), operands)
                             .template as_or_throw<PrimExpr>();
        result.push_back(loop_nest(TensorStore(storage, indices, value)));
      }
    }
    if (op.seed) {
      result.push_back(loop_nest(TensorStore(
          storage, indices,
          ReducerV2Combine(op.op, TensorLoad(storage, indices), *op.seed))));
    }
    return SeqStmt(result);
  }
};
} // namespace tvm::tl::backend

#endif
