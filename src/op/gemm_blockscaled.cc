/*!
 * \file tl/op/gemm_blockscaled.cc
 * \brief Common semantics and registration for block-scaled GEMM.
 */

#include "gemm_blockscaled.h"

#include <utility>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op_attr_types.h>

#include "span_utils.h"
#include "utils.h"

namespace tvm {
namespace tl {

GemmBlockScaled::GemmBlockScaled(
    ffi::Array<PrimExpr> args,
    ffi::Map<ffi::String, ffi::ObjectRef> annotations) {
  ICHECK_EQ(args.size(), 16)
      << "tl.tileop.gemm_blockscaled expects the 13 dense GEMM slots followed "
         "by SFA, SFB and k_start, but got "
      << args.size() << " arguments.";
  auto node = ffi::make_object<GemmBlockScaledNode>();
  GemmNode::InitFromDenseArgs(node.get(), args, annotations);
  node->sfaRegion_ = NormalizeToBufferRegion(args[13]);
  node->sfbRegion_ = NormalizeToBufferRegion(args[14]);
  node->sfKStart_ = args[15].as<PrimExpr>().value();
  ICHECK(node->sfaRegion_.defined() && node->sfbRegion_.defined())
      << "Block-scaled GEMM requires both SFA and SFB scale-factor regions.";
  data_ = std::move(node);
}

AccessRegions GemmBlockScaledNode::GetAccessRegions() const {
  AccessRegions result = GemmNode::GetAccessRegions();
  result.reads.push_back(sfaRegion_);
  result.reads.push_back(sfbRegion_);
  return result;
}

ffi::Array<tirx::BufferRegion>
GemmBlockScaledNode::GetReadBeforeWriteRegions() const {
  ffi::Array<tirx::BufferRegion> result = GemmNode::GetReadBeforeWriteRegions();
  result.push_back(sfaRegion_);
  result.push_back(sfbRegion_);
  return result;
}

TileOperator GemmBlockScaledNode::Clone() const {
  auto op = ffi::make_object<GemmBlockScaledNode>(*this);
  return GemmBlockScaled(op);
}

ffi::String GemmBlockScaledNode::GetGemmInstructionKey(int block_size,
                                                       Target target) const {
  const GemmImpl &impl = ResolveGemmImpl(target);
  if (impl.select_blockscaled_inst == nullptr) {
    LOG(FATAL) << "Block-scaled GEMM is not supported by the " << impl.name
               << " backend (target=" << target->str()
               << "); a dense GEMM lowering would silently drop the SFA/SFB "
                  "scale factors."
               << SpanHintSuffix({a_->span, b_->span, c_->span});
  }
  return impl.select_blockscaled_inst(ffi::GetRef<GemmBlockScaled>(this),
                                      block_size, target);
}

TIR_REGISTER_TL_TILE_OP(GemmBlockScaled, gemm_blockscaled)
    .set_num_inputs(-1)
    .set_attr<tirx::TCallEffectKind>("TCallEffectKind",
                                     Integer(tirx::CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { GemmBlockScaledNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
