/*!
 * \file tl/op/utils.cc
 * \brief Common utilities implementation for TL ops.
 */

#include "utils.h"

#include "builtin.h"
#include "cuda/stubs/cuda.h"
#include "support/check.h"
#include <tvm/tirx/expr.h>

#include <cmath>

namespace tvm::tl {
namespace {

TensorRegion CheckTensorRegion(const TensorRegion &region) {
  tirx::TensorVar tensor = region->source.as_or_throw<tirx::TensorVar>();
  TVM_FFI_CHECK_EQ(tensor.shape().size(), region->region.size(), ValueError)
      << "Tensor region rank does not match its source";
  for (const Range &range : region->region) {
    if (const auto *extent = range->extent.as<IntImmNode>()) {
      TVM_FFI_CHECK(extent->value >= 0, ValueError)
          << "Tensor region extent must be nonnegative";
    }
  }
  return region;
}

TensorRegion RegionFromLoad(const TensorLoad &load) {
  ffi::Array<Range> ranges;
  for (const PrimExpr &index : load->indices) {
    if (const auto *ramp = index.as<RampNode>()) {
      TVM_FFI_CHECK(GetConstInt(ramp->stride) == 1, ValueError)
          << "Only stride-1 Ramp is supported in region conversion";
      TVM_FFI_CHECK(GetConstInt(ramp->lanes).has_value(), ValueError)
          << "Scalable vector lanes are not supported in region conversion";
      ranges.push_back(
          Range::FromMinExtent(ramp->base, ramp->lanes, index->loc));
    } else {
      PrimExpr base = index;
      if (const auto *broadcast = index.as<BroadcastNode>()) {
        base = broadcast->value;
      }
      ranges.push_back(
          Range::FromMinExtent(base, MakeConst(base.ty(), 1), index->loc));
    }
  }
  return CheckTensorRegion(
      TensorRegion(load->source, ranges, TensorRegionType(), load->loc));
}

AccessRegion DecodeRegionCall(const Call &call) {
  TVM_FFI_CHECK_GE(call->args.size(), 2U, ValueError)
      << "tl.region requires a tensor region and an access mask";
  auto mask = GetConstInt(call->args[1].as_or_throw<PrimExpr>());
  TVM_FFI_CHECK(mask.has_value() && *mask >= 0 && *mask <= kAccessReadWrite,
                ValueError)
      << "tl.region access mask must be a constant in [0, 3]";
  if (auto region = call->args[0].as<TensorRegion>()) {
    TVM_FFI_CHECK_EQ(call->args.size(), 2U, ValueError);
    return {CheckTensorRegion(region.value()), static_cast<int>(*mask)};
  }
  TensorLoad load = call->args[0].as_or_throw<TensorLoad>();
  TVM_FFI_CHECK_EQ(call->args.size(), load->indices.size() + 2, ValueError)
      << "tl.region requires one extent per tensor dimension";
  ffi::Array<Range> ranges;
  for (size_t axis = 0; axis < load->indices.size(); ++axis) {
    PrimExpr index = load->indices[axis];
    PrimExpr extent = call->args[axis + 2].as_or_throw<PrimExpr>();
    if (const auto *ramp = index.as<RampNode>()) {
      TVM_FFI_CHECK(GetConstInt(ramp->stride) == 1, ValueError)
          << "tl.region expects a stride-1 Ramp";
      sym::Analyzer analyzer;
      TVM_FFI_CHECK(analyzer->CanProveEqual(ramp->lanes, extent), ValueError)
          << "Ramp lanes and provided extent must match";
      ranges.push_back(Range::FromMinExtent(ramp->base, extent, index->loc));
    } else {
      ranges.push_back(Range::FromMinExtent(index, extent, index->loc));
    }
  }
  return {CheckTensorRegion(TensorRegion(load->source, ranges,
                                         TensorRegionType(), call->loc)),
          static_cast<int>(*mask)};
}

Type InferAccessPtrType(const CallNode *call) {
  TVM_FFI_CHECK_EQ(call->args.size(), 3U, ValueError)
      << "tl.access_ptr requires a tensor access, extent and mask";
  if (const auto *load = call->args[0].as<TensorLoadNode>()) {
    return load->source.as_or_throw<tirx::TensorVar>().ty()->DataPtrType();
  }
  return NormalizeToBufferRegion(call->args[0])
      ->source.as_or_throw<tirx::TensorVar>()
      .ty()
      ->DataPtrType();
}

void ValidateAccessPtr(const CallNode *call) {
  Type expected = InferAccessPtrType(call);
  TVM_FFI_CHECK(ffi::StructuralEqual()(call->ty, expected), TypeError)
      << "tl.access_ptr result must preserve the tensor pointer type";
  TVM_FFI_CHECK(call->ty_args.empty(), TypeError)
      << "tl.access_ptr does not accept type arguments";
  PrimExpr extent = call->args[1].as_or_throw<IntExpr>();
  if (const auto *constant = extent.as<IntImmNode>()) {
    TVM_FFI_CHECK(constant->value >= 0, ValueError)
        << "tl.access_ptr extent must be nonnegative";
  }
  PrimExpr mask = call->args[2].as_or_throw<IntExpr>();
  if (const auto *constant = mask.as<IntImmNode>()) {
    TVM_FFI_CHECK(constant->value >= 0 && constant->value <= 3, ValueError)
        << "tl.access_ptr mask must be in [0, 3]";
  }
}

} // namespace

bool IsZeroBitPattern(const PrimExpr &expression) {
  if (const auto *broadcast = expression.as<BroadcastNode>()) {
    return IsZeroBitPattern(broadcast->value);
  }
  if (const auto *integer = expression.as<IntImmNode>()) {
    return integer->value == 0;
  }
  if (const auto *floating = expression.as<FloatImmNode>()) {
    return floating->value == 0.0 && !std::signbit(floating->value);
  }
  return false;
}

bool IsBufferLikeExpr(const Expr &expression) {
  if (expression.as<tirx::TensorVar>() || expression.as<TensorLoadNode>() ||
      expression.as<TensorRegionNode>()) {
    return true;
  }
  if (const auto *call = expression.as<CallNode>()) {
    return call->op.same_as(region());
  }
  return false;
}

TensorRegion NormalizeToBufferRegion(const Expr &expression) {
  if (auto region = expression.as<TensorRegion>()) {
    return CheckTensorRegion(region.value());
  }
  if (auto tensor = expression.as<tirx::TensorVar>()) {
    ffi::Array<Range> ranges;
    for (const PrimExpr &extent : tensor.value().shape()) {
      ranges.push_back(Range::FromMinExtent(MakeConst(extent.ty(), 0), extent));
    }
    return TensorRegion(tensor.value(), ranges, TensorRegionType(),
                        expression->loc);
  }
  if (auto load = expression.as<TensorLoad>()) {
    return RegionFromLoad(load.value());
  }
  if (auto call = expression.as<Call>();
      call && call.value()->op.same_as(region())) {
    return DecodeRegionCall(call.value()).region;
  }
  TVM_FFI_THROW(TypeError)
      << "Expected TensorVar, TensorLoad, TensorRegion or tl.region, got "
      << expression;
}

AccessRegion NormalizeToAccessRegion(const Expr &expression,
                                     int default_access_mask) {
  if (auto call = expression.as<Call>();
      call && call.value()->op.same_as(region())) {
    return DecodeRegionCall(call.value());
  }
  return {NormalizeToBufferRegion(expression), default_access_mask};
}

Expr MakeAccessPtrFromRegion(const TensorRegion &region, int rw_mask,
                             bool require_2d) {
  CheckTensorRegion(region);
  size_t rank = region->region.size();
  TVM_FFI_CHECK(!require_2d || rank >= 2, ValueError)
      << "Expected a tensor region with at least two dimensions";
  PrimExpr extent = 1;
  size_t first_axis = rank > 2 ? rank - 2 : 0;
  for (size_t axis = first_axis; axis < rank; ++axis) {
    extent = extent * region->region[axis]->extent;
  }
  Call pointer(std::nullopt, access_ptr(), {region, extent, PrimExpr(rw_mask)},
               std::nullopt, {}, region->loc);
  pointer.Validate();
  return pointer;
}

Expr MakeAccessPtrFromBufferLoad(const TensorLoad &load, int rw_mask) {
  Call pointer(std::nullopt, access_ptr(),
               {load, PrimExpr(1), PrimExpr(rw_mask)}, std::nullopt, {},
               load->loc);
  pointer.Validate();
  return pointer;
}

Expr LowerAccessPtrCall(const Call &call) {
  TVM_FFI_CHECK(call->op.same_as(access_ptr()), TypeError)
      << "Expected tl.access_ptr";
  call.Validate();
  Expr base = call->args[0];
  if (const auto *load = base.as<TensorLoadNode>()) {
    ffi::Array<PrimExpr> indices =
        load->indices.Map([](const PrimExpr &index) -> PrimExpr {
          if (const auto *ramp = index.as<RampNode>())
            return ramp->base;
          if (const auto *broadcast = index.as<BroadcastNode>())
            return broadcast->value;
          return index;
        });
    base = TensorLoad(load->source, indices, base->loc);
  }
  TensorRegion region = NormalizeToBufferRegion(base);
  ffi::Array<PrimExpr> indices =
      region->region.Map([](const Range &range) { return range->begin; });
  ffi::Map<ffi::String, ffi::Any> annotations;
  if (call->attrs.has_value()) {
    annotations = call->attrs.value().as_or_throw<DictAttrs>()->dict;
  }
  annotations.Set(attr::kAccessRegion, region);
  annotations.Set(attr::kAccessExtent, call->args[1]);
  annotations.Set(attr::kAccessMask, call->args[2]);
  return Call(call->ty, tirx::address_of_op(),
              {TensorLoad(region->source, indices, call->loc)},
              DictAttrs(annotations), {}, call->loc);
}

int to_CUtensorMapDataType(PrimType type) {
  TVM_FFI_CHECK(type.IsScalar(), TypeError)
      << "TensorMap elements must have scalar types";
  if (IsFP4UnpackedType(type))
    return 14;
  if (IsTensorFloat32(type))
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  if (type.MatchesElementType(kDLFloat4_e2m1fn, 4))
    return 13;
  if (type.MatchesCode(kDLBfloat) && type.bits() == 16)
    return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  if (type.MatchesCode(kDLFloat8_e3m4, kDLFloat8_e4m3, kDLFloat8_e4m3b11fnuz,
                       kDLFloat8_e4m3fn, kDLFloat8_e4m3fnuz, kDLFloat8_e5m2,
                       kDLFloat8_e5m2fnuz, kDLFloat8_e8m0fnu)) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  }
  if (type.MatchesCode(kDLFloat)) {
    if (type.bits() == 16)
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    if (type.bits() == 32)
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    if (type.bits() == 64)
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
  }
  if (type.MatchesCode(kDLInt, kDLUInt)) {
    if (type.bits() == 8)
      return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    if (type.bits() == 16)
      return CU_TENSOR_MAP_DATA_TYPE_UINT16;
    if (type.bits() == 32) {
      return type.MatchesCode(kDLInt) ? CU_TENSOR_MAP_DATA_TYPE_INT32
                                      : CU_TENSOR_MAP_DATA_TYPE_UINT32;
    }
    if (type.bits() == 64) {
      return type.MatchesCode(kDLInt) ? CU_TENSOR_MAP_DATA_TYPE_INT64
                                      : CU_TENSOR_MAP_DATA_TYPE_UINT64;
    }
  }
  TVM_FFI_THROW(TypeError) << "Unsupported CUDA TensorMap element type: "
                           << type;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.access_ptr")
      .set_native_fn_attr<&InferAccessPtrType>(tvm::op_attr::kInferType)
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallNode *)>::FromNative<&ValidateAccessPtr>(),
                     true);
  OpDef("tl.region")
      .set_attr(tvm::op_attr::kFixedReturnType, TensorRegionType());
  ffi::reflection::GlobalDef()
      .def("tl.NormalizeToBufferRegion", NormalizeToBufferRegion)
      .def("tl.NormalizeToAccessRegion",
           [](Expr expression, int access_mask) {
             AccessRegion access =
                 NormalizeToAccessRegion(expression, access_mask);
             return ffi::Array<ffi::Any>{access.region, access.access_mask};
           })
      .def("tl.MakeAccessPtrFromRegion", MakeAccessPtrFromRegion)
      .def("tl.MakeAccessPtrFromBufferLoad", MakeAccessPtrFromBufferLoad)
      .def("tl.LowerAccessPtrCall", LowerAccessPtrCall)
      .def("tl.IsZeroBitPattern", IsZeroBitPattern)
      .def("tl.ToCUtensorMapDataType", to_CUtensorMapDataType);
}

} // namespace tvm::tl
