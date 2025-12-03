/*!
 * \file tl/op/utils.cc
 * \brief Common utilities implementation for TL ops.
 */

#include "utils.h"

#include <tvm/tir/builtin.h>

namespace tvm {
namespace tl {

using namespace tir;

BufferRegion NormalizeToBufferRegion(const PrimExpr &arg) {
  // Case 1: Already a BufferRegion
  if (arg->IsInstance<BufferRegionNode>()) {
    return Downcast<BufferRegion>(arg);
  }

  // Case 2: BufferLoad — convert indices to ranges (Ramp -> lanes, else
  // extent=1)
  if (const auto *load = arg.as<BufferLoadNode>()) {
    Array<Range> ranges;
    for (const PrimExpr &index : load->indices) {
      if (const auto *ramp = index.as<RampNode>()) {
        ICHECK(ramp->stride.as<IntImmNode>()) << "Ramp stride must be IntImm";
        ICHECK_EQ(ramp->stride.as<IntImmNode>()->value, 1)
            << "Only stride-1 Ramp is supported in region conversion";
        ICHECK(ramp->lanes.as<IntImmNode>())
            << "Scalable vector lanes not supported in region conversion";
        ranges.push_back(Range::FromMinExtent(ramp->base, ramp->lanes));
      } else {
        ranges.push_back(Range::FromMinExtent(index, 1));
      }
    }
    return BufferRegion(load->buffer, ranges);
  }

  // Case 3: tl.region(...) — reconstruct via RegionOp (bridge)
  if (const auto *call = arg.as<CallNode>()) {
    if (call->op.same_as(RegionOp::Get())) {
      RegionOp region(call->args);
      return BufferRegion(region->GetBuffer(), region->GetRanges());
    }
    LOG(FATAL) << "Unsupported argument for BufferRegion (expect "
                  "BufferLoad/BufferRegion/tl.region): "
               << arg;
  }

  LOG(FATAL) << "Unsupported argument for BufferRegion: " << arg;
  throw; // Unreachable
}

PrimExpr MakeAccessPtrFromRegion(const BufferRegion &region, int rw_mask,
                                 bool require_2d) {
  Buffer buf = region->buffer;
  int ndim = static_cast<int>(buf->shape.size());
  if (require_2d) {
    ICHECK(ndim >= 2) << "Expect buffers with at least 2 dims";
  }

  PrimExpr offset, extent;
  if (ndim == 1) {
    // 1D: straightforward
    auto axis = region->region[0];
    offset = axis->min;
    extent = axis->extent;
  } else {
    // Compute row-major strides
    std::vector<PrimExpr> strides(ndim);
    PrimExpr one = make_const(buf->shape[0].dtype(), 1);
    PrimExpr cur = one;
    for (int i = ndim - 1; i >= 0; --i) {
      strides[i] = cur;
      cur = cur * buf->shape[i];
    }
    // Offset: sum_{i in [0..ndim-3]} min_i * stride_i
    offset = make_const(buf->shape[0].dtype(), 0);
    for (int i = 0; i < ndim - 2; ++i) {
      offset = offset + region->region[i]->min * strides[i];
    }
    // Extent: last two extents product (elements)
    extent =
        region->region[ndim - 2]->extent * region->region[ndim - 1]->extent;
  }

  // ptype and return handle
  PrimExpr ptype = tir::TypeAnnotation(buf->dtype);
  Array<PrimExpr> acc_args{ptype, buf->data, offset, extent,
                           IntImm(DataType::Int(32), rw_mask)};
  return Call(DataType::Handle(), builtin::tvm_access_ptr(), acc_args);
}

} // namespace tl
} // namespace tvm
