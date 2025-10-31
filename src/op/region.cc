/*!
 * \file tl/op/region.cc
 * \brief Define region operator.
 *
 */

#include "region.h"
#include <tvm/tir/op.h>

namespace tvm {
namespace tl {
using namespace tir;

/**
 * @brief Construct a RegionOp from TL operator arguments.
 *
 * Parses the TL `region` operator call arguments to populate the RegionOpNode:
 * - Expects args[0] to be a `BufferLoad` whose `indices` are the per-dimension
 * minima.
 * - args[1] must be a constant integer used as the access mask.
 * - args[2 + i] provides the extent for dimension `i`.
 *
 * The constructor validates that the number of load indices equals `args.size()
 * - 2` and will abort via ICHECK on mismatch or if args[0] is not a
 * `BufferLoad`.
 *
 * Parameters:
 * - args: TL operator call arguments in the form
 *     [BufferLoad(min_i...), access_mask, extent_0, extent_1, ...,
 * extent_{n-1}] where n = number of dimensions.
 * - vmap: BufferMap passed through by the caller (not documented here as a
 * generic utility).
 */
RegionOp::RegionOp(Array<PrimExpr> args, BufferMap vmap) {
  size_t n = args.size();
  size_t ndim = n - 2;
  auto load = args[0].as<BufferLoadNode>();
  ICHECK(load);
  ICHECK(load->indices.size() == ndim)
      << "load->indices.size() = " << load->indices << " ndim = " << ndim;
  Array<Range> ranges;
  for (size_t i = 0; i < ndim; i++) {
    PrimExpr min = load->indices[i];
    PrimExpr extent = args[2 + i];
    ranges.push_back(Range::FromMinExtent(min, extent));
  }
  ObjectPtr<RegionOpNode> node = tvm::ffi::make_object<RegionOpNode>();
  node->buffer_ = load->buffer;
  node->access_mask_ = static_cast<int>(*as_const_int(args[1]));
  node->ranges_ = ranges;
  data_ = std::move(node);
}

/**
 * @brief Create a copy of this RegionOpNode and return it as a TileOperator.
 *
 * @return TileOperator A new TileOperator that owns a copied RegionOpNode.
 */
TileOperator RegionOpNode::Clone() const {
  auto op = tvm::ffi::make_object<RegionOpNode>(*this);
  return RegionOp(op);
}

/**
 * @brief Check whether the region spans the entire underlying buffer.
 *
 * Returns true if for every dimension the range minimum is zero and the
 * range extent is structurally equal to the corresponding buffer shape
 * dimension. Otherwise returns false.
 *
 * @return true if the region covers the full buffer in all dimensions; false
 * otherwise.
 */
bool RegionOpNode::IsFullRegion() const {
  for (size_t i = 0; i < ranges_.size(); i++) {
    if (!is_zero(ranges_[i]->min))
      return false;
    if (!StructuralEqual()(ranges_[i]->extent, buffer_->shape[i]))
      return false;
  }
  return true;
}

/**
 * @brief Lower the region operator to a TIR statement.
 *
 * Lowers this RegionOpNode into a TIR Stmt by delegating to the operator's
 * evaluation path (currently `Evaluate(0)`).
 *
 * @param T Lowering context (provides buffers, producers/consumers and other
 *          environment required for lowering).
 * @param analyzer Optional arithmetic analyzer used for simplification during
 *                 lowering.
 * @return Stmt The lowered TIR statement representing this region operation.
 */
Stmt RegionOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  return Evaluate(0);
}

/**
 * @brief Infers data layout for the region operator.
 *
 * This operator does not provide any layout inference; the function always
 * returns an empty LayoutMap regardless of the provided arguments or inference
 * level.
 *
 * @param T Layout inference arguments (ignored).
 * @param level Inference granularity level (ignored).
 * @return LayoutMap Empty map indicating no inferred layouts.
 */
LayoutMap RegionOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  return {};
}

TIR_REGISTER_TL_OP(RegionOp, region)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TVM_FFI_STATIC_INIT_BLOCK() { RegionOpNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
