/*!
 * \file tl/op/region.cc
 * \brief Define region operator (bridge to carry BufferRegion via Call args).
 *
 * Notes:
 * - BufferLoad/Ramp cannot represent a general PrimExpr as a vector lane
 *   count. Dynamic extents like (H1 - H0) cannot be encoded as
 *   Ramp(lanes = H1 - H0), and lowering BufferRegion to BufferLoad loses the
 *   explicit extent information.
 * - tl.region carries both mins and extents in Call args and lets the backend
 *   reconstruct a BufferRegion faithfully.
 */

#include "region.h"
#include <tvm/tir/op.h>

namespace tvm {
namespace tl {
using namespace tir;

RegionOp::RegionOp(Array<PrimExpr> args) {
  size_t n = args.size();
  size_t ndim = n - 2;
  auto load = args[0].as<BufferLoadNode>();
  ICHECK(load);
  ICHECK(load->indices.size() == ndim)
      << "load->indices.size() = " << load->indices << " ndim = " << ndim;
  Array<Range> ranges;
  // Rebuild per-axis ranges from mins (BufferLoad indices) and provided extents
  for (size_t i = 0; i < ndim; i++) {
    PrimExpr index = load->indices[i];
    PrimExpr extent = args[2 + i];
    if (const auto *ramp = index.as<RampNode>()) {
      const auto *stride_imm = ramp->stride.as<IntImmNode>();
      ICHECK(stride_imm && stride_imm->value == 1)
          << "RegionOp expects stride-1 Ramp for index";
      if (const auto *lanes_imm = ramp->lanes.as<IntImmNode>()) {
        if (const auto *ext_imm = extent.as<IntImmNode>()) {
          ICHECK_EQ(lanes_imm->value, ext_imm->value)
              << "Ramp lanes and provided extent must match";
        }
      }
      ranges.push_back(Range::FromMinExtent(ramp->base, ramp->lanes));
    } else {
      ranges.push_back(Range::FromMinExtent(index, extent));
    }
  }
  ObjectPtr<RegionOpNode> node = tvm::ffi::make_object<RegionOpNode>();
  node->buffer_ = load->buffer;
  node->access_mask_ = static_cast<int>(*as_const_int(args[1]));
  node->ranges_ = ranges;
  data_ = std::move(node);
}

TileOperator RegionOpNode::Clone() const {
  auto op = tvm::ffi::make_object<RegionOpNode>(*this);
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

TIR_REGISTER_TL_TILE_OP(RegionOp, region)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TVM_FFI_STATIC_INIT_BLOCK() { RegionOpNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
