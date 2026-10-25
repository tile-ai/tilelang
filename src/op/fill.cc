/*!
 * \file tl/op/fill.cc
 *
 * Define elment-wise operators.
 */

#include "fill.h"
#include "compat/visit.h"
#include "support/check.h"

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/attr.h>

#include "utils.h"

#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

std::vector<FillImpl> &FillImplRegistry() {
  static std::vector<FillImpl> registry;
  return registry;
}

const FillImpl &ResolveFillImpl(Target target) {
  const auto &registry = FillImplRegistry();
  const FillImpl *matched_impl = nullptr;
  for (const FillImpl &impl : registry) {
    if (impl.match_target(target)) {
      ICHECK(matched_impl == nullptr)
          << "tl.fill found multiple target-specific implementations for "
          << target << ": " << matched_impl->name << " and " << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.fill requires a target-specific implementation, but no fill "
         "implementation is registered for "
      << target;
  return *matched_impl;
}

} // namespace

void RegisterFillImpl(FillImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.lower != nullptr);
  FillImplRegistry().push_back(impl);
}

/**
 * @brief Construct a Fill operator node from call arguments and a buffer map.
 *
 * This constructor builds a FillNode describing an element-wise fill of a
 * destination buffer region with a scalar/vector value and stores it in
 * `data_`.
 *
 * Detailed behavior:
 * - If `args[0]` is a `TensorLoad`, the loaded buffer becomes the destination
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
 *             - args[0]: destination access (TensorLoad or pointer expression).
 *             - args[1]: value to fill (scalar or vector).
 *
 * Notes:
 * - The constructor enforces constraints (e.g., stride == 1 ramps, constant
 * lanes) and will terminate (via ICHECK) if inputs are unsupported or out
 * of bounds.
 */
Fill::Fill(Array<Expr> args, BlockAnnotations annotations) {
  TVM_FFI_CHECK_EQ(args.size(), 2U, ValueError);
  ObjectPtr<FillNode> node = make_object<FillNode>(ffi::UnsafeInit{});
  AccessRegion dst_access = NormalizeToAccessRegion(args[0], kAccessWrite);
  node->dst = dst_access.region->source.as_or_throw<TensorVar>();
  node->region = dst_access.region->region;
  node->SetAccessRegions({dst_access});
  node->value = Cast(node->dst.dtype(), args[1].as_or_throw<PrimExpr>());
  ICHECK_EQ(node->region.size(), node->dst.shape().size());
  sym::Analyzer analyzer;
  for (size_t axis = 0; axis < node->region.size(); ++axis) {
    const Range &domain = node->region[axis];
    TVM_FFI_CHECK(!analyzer->CanProve(domain->begin < 0), ValueError)
        << "Fill region must start within the destination tensor";
    TVM_FFI_CHECK(!analyzer->CanProve(domain->extent < 0), ValueError)
        << "Fill region must have nonnegative extents";
    TVM_FFI_CHECK(!analyzer->CanProve(domain->begin + domain->extent >
                                      node->dst.shape()[axis]),
                  ValueError)
        << "Fill region exceeds the destination tensor";
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
  auto op = make_object<FillNode>(*this);
  return Fill(op);
}

/**
 * @brief Build a SIMT-style nested parallel loop that fills the destination
 * buffer.
 *
 * Constructs per-dimension data-parallel loop iterators matching this node's
 * region extents, emits a TensorStore that writes the node's `value` into `dst`
 * at the loop indices, and nests the loops (innermost to outermost) as parallel
 * `For` nodes. Returns the outermost `For` loop representing the complete
 * multi-dimensional fill kernel.
 *
 * @return For Outermost parallel `For` loop of the generated nested SIMT loop.
 */
For FillNode::MakeSIMTLoop(sym::Analyzer analyzer) const {
  Array<PrimVar> loop_vars;
  Array<PrimExpr> indices;
  for (size_t axis = 0; axis < region.size(); ++axis) {
    PrimVar variable("fill_axis_" + std::to_string(axis),
                     region[axis]->extent.ty());
    loop_vars.push_back(variable);
    indices.push_back(analyzer->Simplify(region[axis]->begin + variable));
  }
  Stmt body = TensorStore(dst, indices, value, dst->loc);
  if (region.empty()) {
    return For(PrimVar("fill_scalar"), 0, 1, 1, ForKind::kParallel,
               SeqStmt(body));
  }
  for (size_t axis = region.size(); axis-- > 0;) {
    PrimType type = loop_vars[axis].ty();
    body = For(loop_vars[axis], MakeConst(type, 0), region[axis]->extent,
               MakeConst(type, 1), ForKind::kParallel, SeqStmt(body));
  }
  return body.as_or_throw<For>();
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
 * @param lower_args Lowering arguments (target, thread bounds, thread var,
 * layout map).
 * @return Stmt The lowered TIR statement implementing the fill.
 */
Stmt FillNode::Lower(const LowerArgs &lower_args,
                     sym::Analyzer analyzer) const {
  return ResolveFillImpl(lower_args.target).lower(*this, lower_args, analyzer);
}

/**
 * @brief Infer memory/layout mapping for the Fill operator.
 *
 * Returns the layout mapping produced by layout inference for this FillNode.
 * Currently no layout inference is performed for Fill and the function returns
 * an empty LayoutMap.
 *
 * @param layout_args Context required for layout inference (unused).
 * @param level The inference level requested (unused).
 * @return LayoutMap Empty map indicating no inferred layouts for this operator.
 */
LayoutMap FillNode::InferLayout(const LayoutInferArgs &layout_args,
                                InferLevel level) const {
  return {};
}

TIR_REGISTER_TL_TILE_OP(Fill, fill, 2);

TVM_FFI_STATIC_INIT_BLOCK() { FillNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
