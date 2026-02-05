/*!
 * \file tl/op/utils.cc
 * \brief Common utilities implementation for TL ops.
 */

#include "utils.h"
#include "tvm/tir/expr.h"

#include <tvm/tir/builtin.h>

namespace tvm {
namespace tl {

using namespace tir;

bool IsBufferLikeExpr(const PrimExpr &expr) {
  if (expr.as<BufferLoadNode>() || expr.as<BufferRegionNode>()) {
    return true;
  }
  if (const auto *call = expr.as<CallNode>()) {
    return (call->op.same_as(RegionOp::Get()));
  }
  return false;
}

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

// Maps TVM DataType to CUDA's CUtensorMapDataType enum value.
int to_CUtensorMapDataType(DataType dtype) {
  CUtensorMapDataType tp;
  if (dtype.is_float()) {
    switch (dtype.bits()) {
    case 64:
      tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
      break;
    case 32:
      tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
      break;
    case 16:
      tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
      break;
    case 8:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else if (dtype.is_bfloat16()) {
    tp = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if (dtype.is_float8()) {
    tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if (dtype.is_int()) {
    switch (dtype.bits()) {
    case 64:
      tp = CU_TENSOR_MAP_DATA_TYPE_INT64;
      break;
    case 32:
      tp = CU_TENSOR_MAP_DATA_TYPE_INT32;
      break;
    case 16:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT16;
      break;
    case 8:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else if (dtype.is_uint()) {
    switch (dtype.bits()) {
    case 64:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT64;
      break;
    case 32:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT32;
      break;
    case 16:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT16;
      break;
    case 8:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else {
    ICHECK(0) << dtype;
  }
  return static_cast<int>(tp);
}

} // namespace tl
} // namespace tvm
