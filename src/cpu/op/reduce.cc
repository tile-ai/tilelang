/*!
 * \file tl/cpu/op/reduce.cc
 * \brief CPU implementation for tl.reduce lowering (serial, local buffers).
 *
 * First version is purely serial: the reduce dimension is accumulated in a
 * kSerial loop with no SIMD horizontal reduction and no thread-level
 * parallelism. Correctness and test coverage first; performance work
 * (SIMD / multi-thread reduce) is tracked in cpu_ops.md enhancement track.
 *
 * Scope is limited to local -> local/local.var buffers, which is the only
 * path the frontend emits on CPU (reduce_op.py:69-81 emits ReduceOp directly
 * only for local src with local/local.var dst; shared/fragment paths go
 * through alloc_fragment, which CPU does not have).
 */

#include "op/reduce.h"
#include "op/reducer.h"

#include <tvm/runtime/logging.h>

#include "backend/common/op/reduce_value.h"
#include "backend/common/target_utils.h"
#include "op/utils.h"
#include "support/check.h"
#include "transform/loop_partition.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace cpu {

struct Reduce {
  static Stmt Lower(const ReduceOpNode &op, const LowerArgs &lower_args,
                    sym::Analyzer analyzer) {
    TVM_FFI_CHECK(!backend::reduce::UseNanOp(op, op.dst.dtype()), ValueError)
        << "CPU reduce does not support nan_propagate for float16/bfloat16";
    TVM_FFI_CHECK(IsLocalBuffer(op.src) && IsLocalBuffer(op.dst, true),
                  ValueError)
        << "CPU reduce requires local source and local/local.var destination "
           "tensors";
    TVM_FFI_CHECK_EQ(op.batch, 1, ValueError)
        << "CPU reduce does not support GPU AllReduce batching";
    TVM_FFI_CHECK(op.src.dtype() == op.dst.dtype(), ValueError)
        << "CPU reduce source and destination element types must match";
    auto get_tensor = [&](const TensorVar &tensor) {
      return lower_args.buffer_remap.Get(tensor).value_or(tensor);
    };
    auto get_indices = [&](const TensorVar &tensor,
                           const Array<PrimExpr> &indices) {
      if (lower_args.buffer_remap.count(tensor) &&
          lower_args.layout_map.count(tensor)) {
        return lower_args.layout_map[tensor]->Forward(indices);
      }
      return indices;
    };
    TensorVar source = get_tensor(op.src);
    TensorVar destination = get_tensor(op.dst);
    const auto &source_ranges = op.srcRegion_->region;
    const auto &destination_ranges = op.dstRegion_->region;
    Array<PrimVar> destination_vars;
    Array<PrimExpr> destination_indices;
    for (size_t axis = 0; axis < destination_ranges.size(); ++axis) {
      PrimVar variable("reduce_output_" + std::to_string(axis),
                       destination_ranges[axis]->extent.ty());
      destination_vars.push_back(variable);
      destination_indices.push_back(destination_ranges[axis]->begin + variable);
    }
    destination_indices = get_indices(op.dst, destination_indices);
    PrimVar reduce_variable("reduce_axis", source_ranges[op.dim]->extent.ty());
    Array<PrimExpr> source_indices;
    bool keep_dimension = destination_ranges.size() == source_ranges.size();
    for (size_t axis = 0; axis < source_ranges.size(); ++axis) {
      PrimExpr coordinate = reduce_variable;
      if (axis != static_cast<size_t>(op.dim)) {
        coordinate = destination_vars[keep_dimension ||
                                              axis < static_cast<size_t>(op.dim)
                                          ? axis
                                          : axis - 1];
      }
      source_indices.push_back(source_ranges[axis]->begin + coordinate);
    }
    source_indices = get_indices(op.src, source_indices);
    Array<Stmt> statements;
    if (op.clear) {
      statements.push_back(TensorStore(destination, destination_indices,
                                       backend::reduce::MakeInitValue(op)));
    }
    PrimExpr value = backend::reduce::MakeReduce(
        op, 1, TensorLoad(destination, destination_indices),
        TensorLoad(source, source_indices));
    Stmt reduction = TensorStore(destination, destination_indices, value);
    PrimType reduce_type = reduce_variable.ty();
    statements.push_back(For(reduce_variable, MakeConst(reduce_type, 0),
                             source_ranges[op.dim]->extent,
                             MakeConst(reduce_type, 1), ForKind::kDefault,
                             SeqStmt(reduction)));
    Stmt body = SeqStmt(statements);
    for (size_t axis = destination_vars.size(); axis-- > 0;) {
      PrimType type = destination_vars[axis].ty();
      body = For(destination_vars[axis], MakeConst(type, 0),
                 destination_ranges[axis]->extent, MakeConst(type, 1),
                 ForKind::kDefault, SeqStmt(body));
    }
    if (!destination_vars.empty()) {
      body = PragmaUnrollLoop(body.as_or_throw<For>());
    }
    return body;
  }
};

struct FinalizeReducer {
  static Stmt Lower(const FinalizeReducerOpNode &operation,
                    const LowerArgs &arguments, sym::Analyzer analyzer) {
    TVM_FFI_CHECK(analyzer->CanProveEqual(arguments.thread_bounds->extent, 1),
                  ValueError)
        << "CPU reducers require one logical participant per grid iteration";
    for (size_t index = 0; index < operation.plan_steps.size(); index += 2) {
      TVM_FFI_CHECK(analyzer->CanProveEqual(operation.plan_steps[index], 1),
                    ValueError)
          << "A GPU collective plan cannot be lowered as a CPU reducer";
    }
    if (!operation.seed)
      return SeqStmt({});
    TensorVar storage = arguments.buffer_remap.Get(operation.reducer)
                            .value_or(operation.reducer);
    Array<PrimExpr> indices;
    for (size_t axis = 0; axis < storage.shape().size(); ++axis) {
      indices.push_back(PrimVar("finalize_index_" + std::to_string(axis),
                                storage.shape()[axis].ty()));
    }
    Stmt result =
        TensorStore(storage, indices,
                    ReducerV2Combine(operation.op, TensorLoad(storage, indices),
                                     *operation.seed));
    for (size_t axis = indices.size(); axis-- > 0;) {
      PrimVar index = indices[axis].as_or_throw<PrimVar>();
      result =
          For(index, MakeConst(index.ty(), 0), storage.shape()[axis],
              MakeConst(index.ty(), 1), ForKind::kDefault, SeqStmt(result));
    }
    return result;
  }
};

} // namespace cpu

namespace {

bool MatchCPUReduceTarget(Target target) { return TargetIsCPU(target); }

bool RegisterCPUReduce() {
  RegisterReduceImpl(ReduceImpl{
      "cpu.Reduce",
      MatchCPUReduceTarget,
      cpu::Reduce::Lower,
  });
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{"cpu.FinalizeReducer",
                                                  MatchCPUReduceTarget,
                                                  cpu::FinalizeReducer::Lower});
  return true;
}

const bool cpu_reduce_registered = RegisterCPUReduce();

} // namespace

} // namespace tl
} // namespace tvm
