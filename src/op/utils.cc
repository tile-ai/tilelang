/*!
 * \file tl/op/utils.cc
 * \brief Common utilities implementation for TL ops.
 */

#include "utils.h"
#include "ir/tensor_utils.h"

#include <cmath>

#include "builtin.h"
#include "support/check.h"
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;

bool IsZeroBitPattern(const PrimExpr &expr) {
  if (const auto *broadcast = expr.as<BroadcastObj>())
    return IsZeroBitPattern(broadcast->value);
  if (const auto *imm = expr.as<IntImmObj>())
    return imm->value == 0;
  if (const auto *imm = expr.as<FloatImmObj>())
    return imm->value == 0.0 && !std::signbit(imm->value);
  return false;
}

bool IsTensorLikeExpr(const Expr &expr) {
  return expr.as<TensorLoadObj>() || expr.as<TensorRegionObj>() ||
         expr.as<TensorVar>().has_value();
}

TensorRegion NormalizeToTensorRegion(const Expr &arg) {
  if (auto region = arg.as<TensorRegion>()) {
    TensorVar tensor = RegionTensor(region.value());
    ICHECK_EQ(tensor.shape().size(), region.value()->region.size());
    return region.value();
  }
  if (auto tensor = arg.as<TensorVar>())
    return FullTensorRegion(tensor.value());
  if (const auto *load = arg.as<TensorLoadObj>()) {
    TensorVar tensor = load->source.as_or_throw<TensorVar>();
    ffi::Array<Range> ranges;
    for (const PrimExpr &index : load->indices) {
      if (const auto *ramp = index.as<RampObj>()) {
        ICHECK(IsOne(ramp->stride))
            << "Only unit-stride vector regions are supported";
        ICHECK(IsConstInt(ramp->lanes))
            << "Scalable vector lanes are not a finite region";
        ranges.push_back(Range::FromMinExtent(ramp->base, ramp->lanes));
      } else {
        ranges.push_back(Range::FromMinExtent(index, IntImm(index.ty(), 1)));
      }
    }
    return TensorRegion(tensor, ranges, TensorRegionType(), arg->loc);
  }
  TVM_FFI_THROW(TypeError)
      << "Expected a native TensorVar, TensorRegion or TensorLoad; got " << arg;
}

AccessRegion NormalizeToAccessRegion(const Expr &arg, int default_access_mask) {
  ICHECK_GE(default_access_mask, 0);
  ICHECK_LE(default_access_mask, 3);
  return {NormalizeToTensorRegion(arg), default_access_mask};
}

Expr MakeAccessPtrFromRegion(const TensorRegion &region, int rw_mask,
                             bool require_2d) {
  TensorVar tensor = RegionTensor(region);
  const size_t rank = tensor.shape().size();
  ICHECK_EQ(rank, region->region.size());
  if (require_2d)
    ICHECK_GE(rank, 2U) << "Expect a tensor with at least two axes";
  PrimExpr extent = IntImm(tensor.ty()->DefaultIndexType(), 1);
  if (rank == 1) {
    extent = region->region[0]->extent;
  } else if (rank >= 2) {
    extent =
        region->region[rank - 2]->extent * region->region[rank - 1]->extent;
  }
  // Keep the logical base until layout has been committed. Recomputing a
  // row-major offset here would lose strides, offsets, swizzles and aliasing.
  return MakeTensorAccessPtr(region, extent, rw_mask);
}

Expr MakeAccessPtrFromTensorLoad(const TensorLoad &load, int rw_mask) {
  TensorVar tensor = load->source.as_or_throw<TensorVar>();
  return MakeTensorAccessPtr(load, IntImm(tensor.ty()->DefaultIndexType(), 1),
                             rw_mask);
}

int to_CUtensorMapDataType(PrimType dtype) {
  // CUDA 13 adds packed U4 TensorMap formats. The vendored CUDA stub may lag
  // the installed toolkit, so keep the enum value by CUDA's documented order.
  constexpr int kTensorMapDataType16U4Align8B = 13;
  constexpr int kTensorMapDataType16U4Align16B = 14;
  if (dtype.MatchesCode(static_cast<DLDataTypeCode>(kFloat4E2M1Unpacked))) {
    return kTensorMapDataType16U4Align16B;
  }
  if (dtype.MatchesCode(kDLFloat4_e2m1fn)) {
    return kTensorMapDataType16U4Align8B;
  }

  CUtensorMapDataType tp;
  if (dtype.MatchesCode(kDLFloat)) {
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
  } else if (dtype.MatchesElementType(kDLBfloat, 16)) {
    tp = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if (dtype.MatchesCode(static_cast<DLDataTypeCode>(kTensorFloat32))) {
    // tfloat32 uses same memory layout as float32
    tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else if (dtype.MatchesCode(kDLFloat8_e3m4, kDLFloat8_e4m3,
                               kDLFloat8_e4m3b11fnuz, kDLFloat8_e4m3fn,
                               kDLFloat8_e4m3fnuz, kDLFloat8_e5m2,
                               kDLFloat8_e5m2fnuz, kDLFloat8_e8m0fnu)) {
    tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if (dtype.MatchesCode(kDLInt)) {
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
  } else if (dtype.MatchesCode(kDLUInt)) {
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
