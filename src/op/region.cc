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
  ObjectPtr<RegionOpNode> node = make_object<RegionOpNode>();
  node->buffer_ = load->buffer;
  node->access_mask_ = static_cast<int>(*as_const_int(args[1]));
  node->ranges_ = ranges;
  data_ = std::move(node);
}

TileOperator RegionOpNode::Clone() const {
  auto op = make_object<RegionOpNode>(*this);
  return RegionOp(op);
}

bool RegionOpNode::IsFullRegion() const {
  for (size_t i = 0; i < ranges_.size(); i++) {
    if (!is_zero(ranges_[i]->min))
      return false;
    if (!StructuralEqual()(ranges_[i]->extent, buffer_->shape[i]))
      return false;
  }
  return true;
}

Stmt RegionOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  return Evaluate(0);
}

LayoutMap RegionOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  return {};
}

TIR_REGISTER_TL_OP(RegionOp, region)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

} // namespace tl
} // namespace tvm
