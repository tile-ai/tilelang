/*!
 * \file tl/op/fill.cc
 *
 * Define elment-wise operators.
 */

#include "fill.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../layout/tcgen05_layout.h"
#include "../target/utils.h"
#include "../transform/common/loop_fusion_utils.h"
#include "../transform/common/loop_parallel_transform_utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"
#include "builtin.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

/**
 * @brief Construct a Fill operator node from call arguments and a buffer map.
 *
 * This constructor builds a FillNode describing an element-wise fill of a
 * destination buffer region with a scalar/vector value and stores it in
 * `data_`.
 *
 * Detailed behavior:
 * - If `args[0]` is a `BufferLoad`, the loaded buffer becomes the destination
 * and the load indices are converted to per-dimension ranges:
 *   - `Ramp(base, lanes, stride)` is converted to `Range(base, lanes)`. Only
 * stride == 1 and constant `lanes` are supported.
 *   - Non-ramp indices become `Range(index, 1)`.
 * - Otherwise `args[0]` is treated as an access pointer; the destination buffer
 * is resolved via `vmap[GetVarFromAccessPtr(args[0])]` and the region is the
 * full buffer shape for each dimension.
 * - `args[1]` is used as the fill value; it is cast to the destination buffer's
 * dtype if necessary.
 * - Performs validation:
 *   - Region dimensionality must match destination rank.
 *   - For statically-known region mins and extents, checks that mins >= 0 and
 * extents do not exceed the corresponding destination shape extents.
 *
 * Parameters:
 * @param args Call arguments: expected layout is [dst_access_or_bufferload,
 * value].
 *             - args[0]: destination access (BufferLoad or pointer expression).
 *             - args[1]: value to fill (scalar or vector).
 *
 * Notes:
 * - The constructor enforces constraints (e.g., stride == 1 ramps, constant
 * lanes) and will terminate (via CHECK/ICHECK) if inputs are unsupported or out
 * of bounds.
 */
Fill::Fill(Array<PrimExpr> args) {
  ObjectPtr<FillNode> node = tvm::ffi::make_object<FillNode>();

  BufferRegion region = NormalizeToBufferRegion(args[0]);
  node->dst = region->buffer;
  node->region = region->region;

  if (args[1]->dtype != node->dst->dtype) {
    node->value = Cast(node->dst->dtype, args[1]);
  } else {
    node->value = args[1];
  }

  ICHECK(node->region.size() == node->dst->shape.size())
      << "region size = " << node->region.size()
      << " != " << node->dst->shape.size();
  for (int i = 0; i < node->region.size(); i++) {
    // bound check if region is static
    if (const auto *min_imm = node->region[i]->min.as<IntImmNode>()) {
      int64_t min = min_imm->value;
      ICHECK_GE(min, 0) << "region[" << i << "] = " << min << " < 0";
    }
    if (const auto *extent_imm = node->region[i]->extent.as<IntImmNode>()) {
      // Only perform the upper-bound check when the destination shape
      // extent is also statically known. If the shape is symbolic (e.g., Var),
      // skip this static check to avoid invalid downcasts.
      if (const auto *shape_imm = node->dst->shape[i].as<IntImmNode>()) {
        ICHECK_LE(extent_imm->value, shape_imm->value)
            << "region[" << i << "] = " << extent_imm->value << " > "
            << node->dst->shape[i];
      }
    }
  }
  data_ = std::move(node);
}

/**
 * @brief Create a copy of this FillNode and return it as a TileOperator.
 *
 * Constructs a new FillNode by copying the current node and wraps the copy in a
 * Fill TileOperator.
 *
 * @return TileOperator A TileOperator that owns the copied FillNode.
 */
TileOperator FillNode::Clone() const {
  auto op = tvm::ffi::make_object<FillNode>(*this);
  return Fill(op);
}

/**
 * @brief Build a SIMT-style nested parallel loop that fills the destination
 * buffer.
 *
 * Constructs per-dimension data-parallel loop iterators matching this node's
 * region extents, emits a BufferStore that writes the node's `value` into `dst`
 * at the loop indices, and nests the loops (innermost to outermost) as parallel
 * `For` nodes. Returns the outermost `For` loop representing the complete
 * multi-dimensional fill kernel.
 *
 * @return For Outermost parallel `For` loop of the generated nested SIMT loop.
 */
For FillNode::MakeSIMTLoop(arith::Analyzer *analyzer) const {
  int ndim = dst->shape.size();
  Array<IterVar> loop_vars;
  Array<PrimExpr> dst_indices;
  for (int i = 0; i < ndim; i++) {
    Var var = Var(std::string{char('i' + i)}, region[i]->extent->dtype);
    loop_vars.push_back({region[i], var, IterVarType::kDataPar});
    // Offset the loop induction variable by region min to honor sliced regions
    dst_indices.push_back(region[i]->min + var);
  }
  Stmt body = BufferStore(dst, value, dst_indices);
  for (int i = ndim - 1; i >= 0; i--) {
    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body);
  }
  return Downcast<For>(body);
}

/**
 * @brief Lower this Fill operator to a TIR statement for the target.
 *
 * Lowers the FillNode into a Stmt according to the destination buffer scope:
 * - "local.fragment" and shared ("shared", "shared.dyn"): create a parallel
 *   operation from a SIMT loop, infer its layout, partition the root loop by
 *   the thread variable, vectorize the resulting thread loop, and, if a
 *   per-thread predicate exists, guard the vectorized loop with that
 *   predicate.
 * - "local": build a SIMT loop and return its vectorized form.
 * - other scopes: fatal error.
 *
 * The lowering may query layout and thread information from @p T and uses the
 * provided analyzer for any required arithmetic/layout analysis.
 *
 * @param T Lowering arguments (target, thread bounds, thread var, layout map).
 * @return Stmt The lowered TIR statement implementing the fill.
 */
Stmt FillNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (IsFragmentBuffer(dst)) {
    auto par_op = ParallelOp(MakeSIMTLoop(analyzer));
    par_op->InferLayout({T.target,
                         T.thread_bounds,
                         T.layout_map,
                         analyzer,
                         false,
                         T.buffer_remap,
                         {}},
                        InferLevel::kFree);
    auto thread_loop = PartitionLoop(par_op->GetRoot(), T.thread_var, analyzer,
                                     par_op->GetLoopLayout());
    auto vectorized_thread_loop = VectorizeLoop(thread_loop, analyzer);
    if (par_op->GetPredicate(T.thread_var).defined()) {
      return IfThenElse(par_op->GetPredicate(T.thread_var).value(),
                        vectorized_thread_loop);
    }
    return vectorized_thread_loop;
  } else if (IsLocalBuffer(dst)) {
    auto init_loop = MakeSIMTLoop(analyzer);
    auto vectorized_thread_loop = VectorizeLoop(init_loop, analyzer);
    return vectorized_thread_loop;
  } else if (IsSharedBuffer(dst) || IsGlobalBuffer(dst)) {
    auto par_op = ParallelOp(MakeSIMTLoop(analyzer));
    par_op->InferLayout({T.target,
                         T.thread_bounds,
                         T.layout_map,
                         analyzer,
                         false,
                         T.buffer_remap,
                         {}},
                        InferLevel::kFree);
    auto thread_loop = PartitionLoop(par_op->GetRoot(), T.thread_var, analyzer,
                                     par_op->GetLoopLayout());
    auto vectorized_thread_loop = VectorizeLoop(thread_loop, analyzer);
    if (par_op->GetPredicate(T.thread_var).defined()) {
      return IfThenElse(par_op->GetPredicate(T.thread_var).value(),
                        vectorized_thread_loop);
    }
    return vectorized_thread_loop;
  } else {
    LOG(FATAL) << "Unsupported scope " << dst.scope();
    return Stmt();
  }
}

/**
 * @brief Infer memory/layout mapping for the Fill operator.
 *
 * Returns the layout mapping produced by layout inference for this FillNode.
 * Currently no layout inference is performed for Fill and the function returns
 * an empty LayoutMap.
 *
 * @param T Context required for layout inference (unused).
 * @param level The inference level requested (unused).
 * @return LayoutMap Empty map indicating no inferred layouts for this operator.
 */
LayoutMap FillNode::InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const {
  return {};
}

TIR_REGISTER_TL_TILE_OP(Fill, fill)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { FillNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
