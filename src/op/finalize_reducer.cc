/*!
 * \file src/op/finalize_reducer.cc
 *
 * Define finalize_reducer operator.
 */

#include "finalize_reducer.h"

#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../target/utils.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

/**
 * @brief Construct a FinalizeReducerOp from TL operator arguments and a buffer
 * map.
 *
 * Extracts the reducer Buffer from `vmap` using the variable referenced by
 * `args[0]` and sets the reduction operation type from the integer code in
 * `args[1]`.
 *
 * @param args TL operator arguments: expects at least two elements where
 *             `args[0]` is an access pointer identifying the reducer variable
 * and `args[1]` is an integer encoding a `ReducerOpType` (e.g., Sum/Max/Min).
 */
FinalizeReducerOp::FinalizeReducerOp(Array<PrimExpr> args,
                                     Map<String, ObjectRef> annotations) {
  auto node = tvm::ffi::make_object<FinalizeReducerOpNode>();
  auto reducer_access = NormalizeToAccessRegion(args[0], kAccessReadWrite);
  reducer_access.region =
      BufferRegion::FullRegion(reducer_access.region->buffer);
  reducer_access.access_mask = kAccessReadWrite;
  node->reducer = reducer_access.region->buffer;
  node->SetAccessRegions({reducer_access});
  node->op = (ReducerOpType)*as_const_int(args[1]);
  // Read explicit batch size from annotations (0 means auto-detect).
  if (annotations.count("batch")) {
    node->batch = (int)*as_const_int(Downcast<PrimExpr>(annotations["batch"]));
    CHECK_GE(node->batch, 1)
        << "finalize_reducer: batch must be >= 1, got " << node->batch;
  }
  data_ = std::move(node);
}

/**
 * @brief Lower the finalize_reducer TL operator to a TIR statement.
 *
 * Lowers the operator that finalizes a reducer by performing a thread-wide
 * AllReduce across the reducer's output elements and writing the reduced value
 * back into the reducer buffer. The function:
 * - Fetches the reducer buffer and expects its layout to be a Fragment.
 * - Builds index Vars for each output dimension.
 * - Reads the layout's ReplicateExtent and:
 *   - if extent == 1, emits a no-op Evaluate(0);
 *   - otherwise constructs an AllReduce extern call (uses `NamedBarrier` when
 *     the compilation target is Hopper) with an optional workspace (allocated
 * via T.AddWorkspace when reducing_threads >= 32) and stores the result via
 *     BufferStore.
 * - Wraps the store in parallel outer For loops over each output dimension.
 *
 * @param T Lowering context containing buffer remapping, layout map, thread
 * bounds, target, and helper methods (e.g., AddWorkspace).
 * @param analyzer Arithmetic analyzer (unused by this implementation but
 * provided for consistency with lowering API).
 * @return Stmt The lowered TIR statement representing the AllReduce and
 * surrounding loops.
 *
 * @note The function ICHECKs that the reducer layout is present and a Fragment,
 *       and that ReplicateExtent is either 1 or equal to the thread block
 * extent; violations cause a fatal check failure.
 */
Stmt FinalizeReducerOpNode::Lower(const LowerArgs &T,
                                  arith::Analyzer *analyzer) const {
  auto buffer = T.buffer_remap[reducer];
  auto opt_layout = T.layout_map.Get(reducer);
  ICHECK(opt_layout);
  ICHECK(opt_layout->as<Fragment>());
  auto layout = opt_layout->as<Fragment>().value();
  Array<PrimExpr> indices_0;
  indices_0.reserve(layout->OutputDim());
  for (int i = 0; i < layout->OutputDim(); ++i)
    indices_0.push_back(Var("__finred_" + std::to_string(i)));

  const int64_t *p_extent = as_const_int(layout->ReplicateExtent());
  ICHECK(p_extent);
  int extent = *p_extent, scale = 1;
  ICHECK(extent == 1 || extent == *as_const_int(T.thread_bounds->extent))
      << "Illegal finalize_reducer: extent=" << extent
      << "; T.thread_bounds=" << T.thread_bounds;

  if (extent == 1)
    return Evaluate(0);

  std::array op_names{"tl::SumOp", "tl::MaxOp", "tl::MinOp"};
  auto op_str = op_names[(int)op];

  // adopted from ReduceOp
  int reducing_threads = extent;
  auto thread_offset = T.thread_bounds->min;

  // Validate batch against the layout's total output element count.
  int64_t layout_batch_size = 1;
  for (int i = 0; i < layout->OutputDim(); ++i) {
    const int64_t *p = as_const_int(layout->OutputShape()[i]);
    if (p == nullptr) {
      layout_batch_size = -1;
      break;
    }
    layout_batch_size *= *p;
  }

  int64_t effective_batch = static_cast<int64_t>(this->batch);

  if (effective_batch > 1 && layout_batch_size > 0) {
    CHECK_LE(effective_batch, layout_batch_size)
        << "finalize_reducer: batch (" << effective_batch
        << ") exceeds total output elements (" << layout_batch_size << ")";
    CHECK_EQ(layout_batch_size % effective_batch, 0)
        << "finalize_reducer: batch (" << effective_batch
        << ") must evenly divide total output elements (" << layout_batch_size
        << ")";
  }

  // ROCm wavefronts are 64-wide; only batch when reducing across warps.
  const int warp_size = TargetIsRocm(T.target) ? 64 : 32;
  bool use_batch = effective_batch > 1 && reducing_threads > warp_size;

  if (use_batch) {
    // Batched AllReduce: single butterfly pass for all output elements.
    int workspace_stride =
        static_cast<int>(*as_const_int(T.thread_bounds->extent));
    std::stringstream ss;
    if (TargetHasSMVersionGE(T.target, 90)) {
      auto all_threads = T.thread_bounds->extent;
      ss << "tl::AllReduce<" << op_str << ", " << reducing_threads << ", " << 1
         << ", " << thread_offset << ", tl::NamedBarrier<" << all_threads
         << ">, " << effective_batch << ", " << workspace_stride
         << ">::run_batch";
    } else if (TargetIsRocm(T.target)) {
      // HIP AllReduce has no Barrier type parameter.
      ss << "tl::AllReduce<" << op_str << ", " << reducing_threads << ", " << 1
         << ", " << thread_offset << ", " << effective_batch << ", "
         << workspace_stride << ">::run_batch";
    } else {
      ss << "tl::AllReduce<" << op_str << ", " << reducing_threads << ", " << 1
         << ", " << thread_offset << ", tl::SyncThreadsBarrier, "
         << effective_batch << ", " << workspace_stride << ">::run_batch";
    }
    int ws_size = workspace_stride * static_cast<int>(effective_batch);
    PrimExpr workspace = T.AddWorkspace(ws_size, buffer->dtype);
    Array<PrimExpr> args = {StringImm(ss.str()), buffer->data, workspace};
    return Evaluate(Call(DataType::Handle(), builtin::call_extern(), args));
  }

  // Scalar AllReduce path (original).
  std::stringstream ss;
  if (TargetHasSMVersionGE(T.target, 90)) {
    auto all_threads = T.thread_bounds->extent;
    ss << "tl::AllReduce<" << op_str << ", " << reducing_threads << ", " << 1
       << ", " << thread_offset << ", tl::NamedBarrier<" << all_threads
       << ">>::run";
  } else {
    ss << "tl::AllReduce<" << op_str << ", " << reducing_threads << ", " << 1
       << ", " << thread_offset << ">::run";
  }
  Array<PrimExpr> thread_reduce_args = {StringImm(ss.str()),
                                        BufferLoad(buffer, indices_0)};
  if (reducing_threads >= 32) {
    PrimExpr workspace =
        T.AddWorkspace(*as_const_int(T.thread_bounds->extent), buffer->dtype);
    thread_reduce_args.push_back(workspace);
  }
  auto call = Call(buffer->dtype, builtin::call_extern(), thread_reduce_args);
  Stmt body = BufferStore(buffer, call, indices_0);

  // make the outer spatial loop
  for (int i = layout->OutputDim() - 1; i >= 0; i--) {
    body = For(indices_0[i].as<Var>().value(), 0, layout->OutputShape()[i],
               ForKind::kParallel, body);
  }

  return body;
}

/**
 * @brief Infer and return the layout mapping for the reducer buffer.
 *
 * Copies the existing layout for the reducer from the provided LayoutInferArgs
 * into a new LayoutMap and returns it. The inference does not modify the
 * layout; it preserves the reducer's current layout.
 *
 * @param T Provides the input layout map from which the reducer's layout is
 * copied.
 * @param level Unused by this operator; present for API compatibility.
 * @return LayoutMap A map that contains the reducer buffer mapped to its
 * original layout.
 */
LayoutMap FinalizeReducerOpNode::InferLayout(const LayoutInferArgs &T,
                                             InferLevel level) const {
  LayoutMap layout_map;
  layout_map.Set(reducer, T.layout_map.Get(reducer).value());
  return layout_map;
}

/**
 * @brief Create a deep copy of this FinalizeReducerOpNode and wrap it as a
 * TileOperator.
 *
 * Constructs a new FinalizeReducerOpNode by copying the current node state and
 * returns a TileOperator that owns the copied node.
 *
 * @return TileOperator A TileOperator that contains a deep copy of this node.
 */
TileOperator FinalizeReducerOpNode::Clone() const {
  auto node = tvm::ffi::make_object<FinalizeReducerOpNode>(*this);
  return TileOperator(node);
}

TIR_REGISTER_TL_TILE_OP(FinalizeReducerOp, finalize_reducer)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { FinalizeReducerOpNode::RegisterReflection(); }
} // namespace tl
} // namespace tvm
