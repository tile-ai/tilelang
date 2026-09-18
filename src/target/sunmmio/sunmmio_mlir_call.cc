#include "sunmmio_mlir_call.h"

#include "../sunmmio_utils.h"
#include "sunmmio_mlir_type.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Operation.h"
#include "npuir/Dialect/SUVM/IR/Attributes.h"
#include "npuir/Dialect/SUVM/IR/Ops.h"
#include "npuir/Dialect/SUVM/IR/Types.h"

#include <tvm/runtime/logging.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace tvm {
namespace codegen {

namespace {

std::string MapMemoryScopeName(mlir::suvm::MemorySpace space) {
  switch (space) {
  case mlir::suvm::MemorySpace::global:
    return "global";
  case mlir::suvm::MemorySpace::asram:
    return "shared.asram";
  case mlir::suvm::MemorySpace::wsram:
    return "shared.wsram";
  case mlir::suvm::MemorySpace::rsram:
    return "shared.rsram";
  }
  LOG(FATAL) << "Unsupported SUVM memory space";
  TVM_FFI_UNREACHABLE();
}

template <typename OpType> void VerifyAsyncOp(OpType op, const char *callee) {
  ICHECK(llvm::succeeded(op.verify()))
      << callee << " generated an invalid SUVM operation";
}

template <typename OpType>
void VerifyA4EMulticastOp(OpType op, const char *callee) {
  VerifyAsyncOp(op, callee);
  ICHECK(llvm::succeeded(op.verifyWithDeviceArch(mlir::suvm::DeviceArch::a4e)))
      << callee << " violates A4E multicast data-path constraints";
}

SunmmioMlirContext::SyncUnitMask MapSyncUnits(int64_t mask) {
  constexpr tl::SunmmioSyncUnits kKnownUnits =
      tl::kSunmmioSyncOdma0 | tl::kSunmmioSyncOdma1 | tl::kSunmmioSyncTc |
      tl::kSunmmioSyncHlink | tl::kSunmmioSyncVlink | tl::kSunmmioSyncVector |
      tl::kSunmmioSyncRsram;
  ICHECK_GE(mask, 0) << "Sunmmio sync unit mask must be non-negative";
  ICHECK_EQ(static_cast<tl::SunmmioSyncUnits>(mask) & ~kKnownUnits, 0U)
      << "Sunmmio sync unit mask contains unsupported bits: " << mask;

  SunmmioMlirContext::SyncUnitMask units = SunmmioMlirContext::kNoSyncUnits;
  auto add = [&](tl::SunmmioSyncUnits bit, mlir::suvm::SyncUnits unit) {
    if ((static_cast<tl::SunmmioSyncUnits>(mask) & bit) != 0) {
      units |= SunmmioMlirContext::ToSyncUnitMask(unit);
    }
  };
  add(tl::kSunmmioSyncOdma0, mlir::suvm::SyncUnits::odma0);
  add(tl::kSunmmioSyncOdma1, mlir::suvm::SyncUnits::odma1);
  add(tl::kSunmmioSyncTc, mlir::suvm::SyncUnits::tc);
  add(tl::kSunmmioSyncHlink, mlir::suvm::SyncUnits::hlink);
  add(tl::kSunmmioSyncVlink, mlir::suvm::SyncUnits::vlink);
  add(tl::kSunmmioSyncVector, mlir::suvm::SyncUnits::vector);
  add(tl::kSunmmioSyncRsram, mlir::suvm::SyncUnits::rsram);
  return units;
}

SunMMIOType ConvertMemTensorType(mlir::suvm::MemTensorType memtensor_type,
                                 DataType dtype) {
  SunMMIOType type;
  type.kind = SunMMIOType::Kind::kMemTensor;
  type.dtype = dtype.with_lanes(1);
  type.lanes = 1;
  for (int64_t dim : memtensor_type.getShape()) {
    ICHECK(!mlir::ShapedType::isDynamic(dim))
        << "MX unpack alias memtensor must have static shape";
    type.shape.push_back(IntImm(DataType::Int(32), dim));
  }

  mlir::suvm::LayoutAttr layout = memtensor_type.getLayout();
  for (int64_t dim : layout.getFlattenedHShape()) {
    ICHECK(!mlir::ShapedType::isDynamic(dim))
        << "MX unpack alias layout shape must be static";
    type.layout_hshape.push_back(IntImm(DataType::Int(32), dim));
  }
  for (int64_t stride : layout.getFlattenedHStride()) {
    ICHECK(!mlir::ShapedType::isDynamic(stride))
        << "MX unpack alias layout stride must be static";
    type.layout_hstride.push_back(IntImm(DataType::Int(32), stride));
  }
  for (uint8_t level : layout.getDimLevels()) {
    type.layout_dim_levels.push_back(level);
  }
  type.memory_scope =
      MapMemoryScopeName(memtensor_type.getMemorySpace().getValue());
  type.byte_offset = memtensor_type.getByteOffset();
  return type;
}

} // namespace

SunmmioMlirCall::SunmmioMlirCall(SunmmioMlirContext &ctx) : ctx_(ctx) {}

SunMMIOValue SunmmioMlirCall::RegionCall(
    const std::string &result_name, const std::string &buffer_handle,
    const std::vector<SunMMIOValue> &mins, const std::vector<int64_t> &extents,
    DataType ret_dtype, const SunMMIOType &ret_type, int64_t byte_offset) {
  SunmmioMlirType type(ctx_);

  mlir::Value source = ctx_.LookupMLIRValue(buffer_handle);
  ICHECK(source) << "Missing MLIR source buffer for tl.tileop.region `"
                 << buffer_handle << "`";

  auto memtensor_ty =
      mlir::dyn_cast<mlir::suvm::MemTensorType>(source.getType());
  ICHECK(memtensor_ty)
      << "tl.tileop.region expects source buffer to be a suvm.memtensor";

  mlir::Value source_for_view = source;
  if (byte_offset != 0) {
    ICHECK_GE(byte_offset, 0)
        << "tl.tileop.region byte_offset must be non-negative";
    int64_t shifted_offset = memtensor_ty.getByteOffset() + byte_offset;
    auto shifted_ty = mlir::suvm::MemTensorType::get(
        memtensor_ty.getShape(), memtensor_ty.getElementType(),
        memtensor_ty.getLayout(), memtensor_ty.getMemorySpace(),
        shifted_offset);
    auto shift_op = mlir::suvm::ShiftMemTensorOp::create(
        ctx_.builder, type.MakeDebugLoc("shift_memtensor"), shifted_ty, source,
        static_cast<uint64_t>(byte_offset));
    source_for_view = shift_op.getResult();
  }

  mlir::SmallVector<mlir::Value, 4> indices;
  indices.reserve(mins.size());
  for (int64_t i = 0; i < static_cast<int64_t>(mins.size()); ++i) {
    const auto &min = mins[i];
    mlir::Value index = ctx_.LookupMLIRValue(min.value);
    ICHECK(index) << "Missing MLIR min value in tl.tileop.region for `"
                  << min.value << "`";
    index = type.EnsureIndex(index);
    indices.push_back(index);
  }

  mlir::SmallVector<int64_t, 4> shape;
  mlir::SmallVector<int64_t, 4> tiled_dims;

  shape.reserve(extents.size());
  for (int64_t i = 0; i < static_cast<int64_t>(extents.size()); ++i) {
    if (extents[i] != 1) {
      shape.push_back(extents[i]);
      tiled_dims.push_back(i);
    }
  }
  if (shape.empty()) {
    ICHECK(!extents.empty())
        << "tl.tileop.region expects at least one region dimension";
    shape.push_back(1);
    tiled_dims.push_back(0);
  }
  ICHECK(shape.size() == 1 || shape.size() == 2)
      << "tl.tileop.region expects one or two tiled dims with extent != 1, "
         "but got "
      << shape.size();

  mlir::Type elem_ty = memtensor_ty.getElementType();
  mlir::Type tile_view_ty =
      mlir::suvm::TileViewType::get(&ctx_.mlir_ctx, shape, elem_ty);
  auto tiled_dims_attr = ctx_.builder.getDenseI64ArrayAttr(tiled_dims);

  auto view_op = mlir::suvm::GetPartitionedTileViewOp::create(
      ctx_.builder, type.MakeDebugLoc("region"), tile_view_ty, source_for_view,
      indices, tiled_dims_attr);
  ctx_.BindMLIRValue(result_name, view_op->getResult(0));
  return SunMMIOValue{ret_dtype, result_name, ret_type};
}

std::pair<SunMMIOValue, SunMMIOValue>
SunmmioMlirCall::MXUnpack(const std::string &scale_name,
                          const std::string &data_name, const SunMMIOValue &mx,
                          DataType scale_dtype, DataType data_dtype) {
  ICHECK(!scale_name.empty() && !data_name.empty())
      << "MXUnpack expects named scale/data alias results";
  SunmmioMlirType type(ctx_);

  mlir::Value mx_value = ctx_.LookupMLIRValue(mx.value);
  ICHECK(mx_value) << "Missing MLIR MX memtensor for `" << mx.value << "`";
  auto mx_type = mlir::dyn_cast<mlir::suvm::MemTensorType>(mx_value.getType());
  ICHECK(mx_type) << "MXUnpack expects source to be a suvm.memtensor";

  mlir::Location loc = type.MakeDebugLoc("mx_unpack");
  mlir::SmallVector<mlir::Type, 2> result_types;
  mlir::LogicalResult inferred = mlir::suvm::UnpackOp::inferReturnTypes(
      &ctx_.mlir_ctx, loc, mlir::ValueRange{mx_value}, mlir::DictionaryAttr(),
      nullptr, mlir::RegionRange(), result_types);
  ICHECK(mlir::succeeded(inferred))
      << "Failed to infer suvm.unpack result types from MX memtensor";
  ICHECK_EQ(result_types.size(), 2U)
      << "suvm.unpack must infer scale and data result types";

  mlir::OperationState st(loc, "suvm.unpack");
  st.addOperands(mx_value);
  st.addTypes(result_types);
  mlir::Operation *unpack_op = ctx_.builder.create(st);
  ICHECK(unpack_op && unpack_op->getNumResults() == 2)
      << "suvm.unpack lowering expected two results";

  mlir::Value scale_value = unpack_op->getResult(0);
  mlir::Value data_value = unpack_op->getResult(1);
  auto scale_type =
      mlir::dyn_cast<mlir::suvm::MemTensorType>(scale_value.getType());
  auto data_type =
      mlir::dyn_cast<mlir::suvm::MemTensorType>(data_value.getType());
  ICHECK(scale_type && data_type) << "suvm.unpack results must be memtensors";
  ICHECK(scale_type.getElementType() == type.MapElementType(scale_dtype))
      << "suvm.unpack scale dtype does not match expected TileLang dtype";
  ICHECK(data_type.getElementType() == type.MapElementType(data_dtype))
      << "suvm.unpack data dtype does not match expected TileLang dtype";

  ctx_.BindMLIRValue(scale_name, scale_value);
  ctx_.BindMLIRValue(data_name, data_value);
  return {SunMMIOValue{scale_dtype, scale_name,
                       ConvertMemTensorType(scale_type, scale_dtype)},
          SunMMIOValue{data_dtype, data_name,
                       ConvertMemTensorType(data_type, data_dtype)}};
}

SunMMIOValue SunmmioMlirCall::Call(const std::string &result_name,
                                   const std::string &callee,
                                   const std::vector<SunMMIOValue> &operands,
                                   const SunMMIOCallAttrs &attrs,
                                   const std::string &category,
                                   DataType ret_dtype,
                                   const SunMMIOType &ret_type) {
  SunmmioMlirType type(ctx_);
  auto get_int_attr = [&](const char *key) -> std::optional<int64_t> {
    auto it = attrs.find(key);
    if (it == attrs.end()) {
      return std::nullopt;
    }
    const int64_t *value = std::get_if<int64_t>(&it->second);
    ICHECK(value) << "SunMMIO call attr `" << key << "` must be int64_t";
    return *value;
  };
  auto get_bool_attr = [&](const char *key) -> std::optional<bool> {
    auto it = attrs.find(key);
    if (it == attrs.end()) {
      return std::nullopt;
    }
    const bool *value = std::get_if<bool>(&it->second);
    ICHECK(value) << "SunMMIO call attr `" << key << "` must be bool";
    return *value;
  };
  auto get_string_attr = [&](const char *key) -> std::optional<std::string> {
    auto it = attrs.find(key);
    if (it == attrs.end()) {
      return std::nullopt;
    }
    const std::string *value = std::get_if<std::string>(&it->second);
    ICHECK(value) << "SunMMIO call attr `" << key << "` must be string";
    return *value;
  };
  auto get_int_vector_attr = [&](const char *key) -> std::vector<int64_t> {
    auto it = attrs.find(key);
    if (it == attrs.end()) {
      return {};
    }
    const std::vector<int64_t> *value =
        std::get_if<std::vector<int64_t>>(&it->second);
    ICHECK(value) << "SunMMIO call attr `" << key
                  << "` must be vector<int64_t>";
    std::vector<int64_t> masks;
    for (int64_t mask : *value) {
      if (std::find(masks.begin(), masks.end(), mask) == masks.end()) {
        masks.push_back(mask);
      }
    }
    return masks;
  };
  auto parse_participant_mask = [&]() -> int64_t {
    return get_int_attr(SunMMIOCallAttrKey::kParticipantMask).value_or(-1);
  };
  auto parse_candidate_masks = [&]() -> std::vector<int64_t> {
    return get_int_vector_attr(SunMMIOCallAttrKey::kCandidateMasks);
  };
  auto parse_barrier_mask_key = [&]() -> std::string {
    std::optional<std::string> key =
        get_string_attr(SunMMIOCallAttrKey::kBarrierMaskKey);
    ICHECK(key.has_value()) << "barrier call requires barrier_mask_key attr";
    return *key;
  };
  auto ensure_i64 = [&](mlir::Value value,
                        const char *arg_name) -> mlir::Value {
    ICHECK(value) << arg_name << " is missing";
    mlir::Type ty = value.getType();
    mlir::Type i64_ty = ctx_.builder.getI64Type();
    if (ty == i64_ty) {
      return value;
    }
    if (ty.isIndex()) {
      return mlir::arith::IndexCastOp::create(ctx_.builder, type.Loc(), i64_ty,
                                              value)
          .getResult();
    }
    auto int_ty = mlir::dyn_cast<mlir::IntegerType>(ty);
    ICHECK(int_ty) << arg_name << " must be an integer or index value";
    unsigned width = int_ty.getWidth();
    ICHECK_NE(width, 64U) << arg_name << " has unsupported integer type";
    if (width < 64) {
      return mlir::arith::ExtUIOp::create(ctx_.builder, type.Loc(), i64_ty,
                                          value)
          .getResult();
    }
    return mlir::arith::TruncIOp::create(ctx_.builder, type.Loc(), i64_ty,
                                         value)
        .getResult();
  };
  auto require_bool_attr = [&](const char *key, const char *arg_name) -> bool {
    std::optional<bool> value = get_bool_attr(key);
    if (value.has_value()) {
      return *value;
    }
    LOG(FATAL) << arg_name << " is missing from attrs";
    TVM_FFI_UNREACHABLE();
  };
  auto require_odma_unit = [&]() -> mlir::suvm::Unit {
    std::optional<std::string> unit =
        get_string_attr(SunMMIOCallAttrKey::kUnit);
    ICHECK(unit.has_value()) << callee << " requires a resolved ODMA unit";
    if (*unit == "odma0") {
      return mlir::suvm::Unit::Odma0;
    }
    if (*unit == "odma1") {
      return mlir::suvm::Unit::Odma1;
    }
    LOG(FATAL) << callee << " has unsupported ODMA unit " << *unit;
    TVM_FFI_UNREACHABLE();
  };
  auto ensure_static_barrier_for_mask = [&](int64_t mask) -> mlir::Value {
    ICHECK_GE(mask, 0) << "barrier participant_mask must be non-negative";
    auto it = ctx_.static_barrier_by_mask.find(mask);
    if (it != ctx_.static_barrier_by_mask.end() && it->second) {
      return it->second;
    }
    mlir::IntegerAttr mask_attr = ctx_.builder.getI64IntegerAttr(mask);
    auto barrier_op = mlir::suvm::BarrierInitOp::create(
        ctx_.builder, type.MakeDebugLoc("barrier_init"), mlir::Value{},
        mask_attr, mlir::IntegerAttr{});
    ctx_.static_barrier_by_mask[mask] = barrier_op.getBarrier();
    return barrier_op.getBarrier();
  };
  auto lookup_static_barrier_for_mask = [&](int64_t mask) -> mlir::Value {
    auto it = ctx_.static_barrier_by_mask.find(mask);
    ICHECK(it != ctx_.static_barrier_by_mask.end() && it->second)
        << "tl.barrier_arrive_and_wait candidate_mask=" << mask
        << " has no corresponding tl.barrier_init.";
    return it->second;
  };
  auto ensure_dynamic_barrier_for_mask =
      [&](mlir::Value mask, const std::string &key) -> mlir::Value {
    mask = ensure_i64(mask, "tl.barrier_init mask");
    auto it = ctx_.barrier_by_mask.find(key);
    if (it != ctx_.barrier_by_mask.end() && it->second) {
      return it->second;
    }
    auto barrier_op = mlir::suvm::BarrierInitOp::create(
        ctx_.builder, type.MakeDebugLoc("barrier_init"), mask,
        mlir::IntegerAttr{}, mlir::IntegerAttr{});
    ctx_.barrier_by_mask[key] = barrier_op.getBarrier();
    return barrier_op.getBarrier();
  };
  auto emit_barrier_arrive_and_wait = [&](mlir::Value barrier) {
    (void)mlir::suvm::BarrierArriveAndWaitOp::create(
        ctx_.builder, type.MakeDebugLoc("barrier_arrive_and_wait"), barrier);
  };
  auto emit_candidate_barrier_wait =
      [&](mlir::Value dynamic_mask, const std::vector<int64_t> &candidates) {
        ICHECK(dynamic_mask) << "dynamic barrier mask is missing";
        ICHECK(!candidates.empty())
            << "dynamic barrier wait requires static candidate masks";
        dynamic_mask =
            ensure_i64(dynamic_mask, "tl.barrier_arrive_and_wait mask");

        std::function<void(size_t)> emit_case = [&](size_t index) {
          ICHECK_LT(index, candidates.size());
          int64_t candidate = candidates[index];
          mlir::Value cst = mlir::arith::ConstantIntOp::create(
                                ctx_.builder, type.Loc(), candidate, 64)
                                .getResult();
          mlir::Value is_match = mlir::arith::CmpIOp::create(
              ctx_.builder, type.Loc(), mlir::arith::CmpIPredicate::eq,
              dynamic_mask, cst);
          auto if_op =
              mlir::scf::IfOp::create(ctx_.builder, type.Loc(), is_match,
                                      /*withElseRegion=*/true);

          mlir::Block &then_block = if_op.getThenRegion().front();
          ctx_.builder.setInsertionPointToStart(&then_block);
          emit_barrier_arrive_and_wait(
              lookup_static_barrier_for_mask(candidate));

          mlir::Block &else_block = if_op.getElseRegion().front();
          ctx_.builder.setInsertionPointToStart(&else_block);
          if (index + 1 < candidates.size()) {
            emit_case(index + 1);
          }

          ctx_.builder.setInsertionPointAfter(if_op);
        };

        emit_case(0);
      };

  if (callee == "tl.barrier_init") {
    std::string barrier_key = parse_barrier_mask_key();
    int64_t participant_mask = parse_participant_mask();
    std::vector<int64_t> candidates = parse_candidate_masks();
    mlir::Value result_barrier;
    if (participant_mask >= 0) {
      result_barrier = ensure_static_barrier_for_mask(participant_mask);
      ctx_.barrier_by_mask[barrier_key] = result_barrier;
    } else if (!candidates.empty()) {
      for (int64_t mask : candidates) {
        (void)ensure_static_barrier_for_mask(mask);
      }
    } else {
      ICHECK_EQ(operands.size(), 1U)
          << "tl.barrier_init dynamic mask expects one mask operand";
      mlir::Value dynamic_mask = ctx_.LookupMLIRValue(operands[0].value);
      if (!dynamic_mask) {
        dynamic_mask =
            type.ResolveValue(operands[0], ctx_.builder.getI64Type());
      }
      result_barrier =
          ensure_dynamic_barrier_for_mask(dynamic_mask, barrier_key);
    }
    if (!result_name.empty() && result_barrier) {
      ctx_.BindMLIRValue(result_name, result_barrier);
    }
    return SunMMIOValue{ret_dtype, result_name, ret_type};
  } else if (callee == "tl.barrier_arrive_and_wait") {
    std::string barrier_key = parse_barrier_mask_key();
    std::optional<int64_t> participant_mask =
        get_int_attr(SunMMIOCallAttrKey::kParticipantMask);
    std::vector<int64_t> candidates = parse_candidate_masks();
    if (!candidates.empty()) {
      ICHECK_EQ(operands.size(), 1U)
          << "tl.barrier_arrive_and_wait candidate fallback expects one mask "
             "operand";
      mlir::Value dynamic_mask = ctx_.LookupMLIRValue(operands[0].value);
      if (!dynamic_mask) {
        dynamic_mask =
            type.ResolveValue(operands[0], ctx_.builder.getI64Type());
      }
      emit_candidate_barrier_wait(dynamic_mask, candidates);
    } else {
      auto barrier_it = ctx_.barrier_by_mask.find(barrier_key);
      ICHECK(barrier_it != ctx_.barrier_by_mask.end() && barrier_it->second)
          << "tl.barrier_arrive_and_wait participant_mask="
          << (participant_mask ? std::to_string(*participant_mask)
                               : barrier_key)
          << " has no corresponding tl.barrier_init.";
      emit_barrier_arrive_and_wait(barrier_it->second);
    }
    return SunMMIOValue{ret_dtype, result_name, ret_type};
  } else if (callee == "tl.sunmmio_sync") {
    int64_t mask = get_int_attr(SunMMIOCallAttrKey::kSyncUnits).value_or(0);
    SunmmioMlirContext::SyncUnitMask units_mask = MapSyncUnits(mask);
    ICHECK_NE(units_mask, SunmmioMlirContext::kNoSyncUnits)
        << "tl.sunmmio_sync requires at least one hardware unit";
    mlir::suvm::SyncUnits units = SunmmioMlirContext::ToSyncUnits(units_mask);
    mlir::suvm::SyncOp::create(
        ctx_.builder, type.MakeDebugLoc("sunmmio_sync"),
        mlir::suvm::SyncUnitsAttr::get(&ctx_.mlir_ctx, units));
    ctx_.CompletePendingSyncUnits(units_mask);
    return SunMMIOValue{ret_dtype, result_name, ret_type};
  } else if (callee == "tl.dma_copy") {
    ICHECK_GE(operands.size(), 2)
        << "tl.dma_copy expects src and dst tile views";

    mlir::Value src = ctx_.LookupMLIRValue(operands[0].value);
    ICHECK(src) << "Missing MLIR source tile view for tl.dma_copy `"
                << operands[0].value << "`";
    auto src_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(src.getType());
    ICHECK(src_ty) << "tl.dma_copy expects source to be a suvm.tile_view";

    mlir::Value dst = ctx_.LookupMLIRValue(operands[1].value);
    ICHECK(dst) << "Missing MLIR destination tile view for tl.dma_copy `"
                << operands[1].value << "`";
    auto dst_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(dst.getType());
    ICHECK(dst_ty) << "tl.dma_copy expects destination to be a suvm.tile_view";

    auto copy_op = mlir::suvm::CopyAsyncOp::create(
        ctx_.builder, type.MakeDebugLoc("dma_copy"), mlir::Type(), src, dst,
        require_odma_unit());
    VerifyAsyncOp(copy_op, "tl.dma_copy");
    ctx_.AddPendingSyncUnits(copy_op.getUnit() == mlir::suvm::Unit::Odma0
                                 ? mlir::suvm::SyncUnits::odma0
                                 : mlir::suvm::SyncUnits::odma1);

    return SunMMIOValue{ret_dtype, result_name, ret_type};
  } else if (callee == "tl.sunmmio_layout_transform") {
    ICHECK_GE(operands.size(), 2)
        << "tl.sunmmio_layout_transform expects src and dst tile views";

    mlir::Value src = ctx_.LookupMLIRValue(operands[0].value);
    ICHECK(src)
        << "Missing MLIR source tile view for tl.sunmmio_layout_transform `"
        << operands[0].value << "`";
    auto src_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(src.getType());
    ICHECK(src_ty)
        << "tl.sunmmio_layout_transform expects source to be a suvm.tile_view";

    mlir::Value dst = ctx_.LookupMLIRValue(operands[1].value);
    ICHECK(dst) << "Missing MLIR destination tile view for "
                   "tl.sunmmio_layout_transform `"
                << operands[1].value << "`";
    auto dst_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(dst.getType());
    ICHECK(dst_ty) << "tl.sunmmio_layout_transform expects destination to be a "
                      "suvm.tile_view";

    auto transform_op = mlir::suvm::TransformAsyncOp::create(
        ctx_.builder, type.MakeDebugLoc("sunmmio_layout_transform"),
        mlir::Type(), src, dst, mlir::suvm::PadModeAttr{}, require_odma_unit());
    VerifyAsyncOp(transform_op, "tl.sunmmio_layout_transform");
    ctx_.AddPendingSyncUnits(mlir::suvm::SyncUnits::odma1);

    return SunMMIOValue{ret_dtype, result_name, ret_type};
  } else if (callee == "tl.sunmmio_transpose") {
    ICHECK_GE(operands.size(), 2)
        << "tl.sunmmio_transpose expects src and dst tile views";

    mlir::Value src = ctx_.LookupMLIRValue(operands[0].value);
    ICHECK(src) << "Missing MLIR source tile view for tl.sunmmio_transpose `"
                << operands[0].value << "`";
    auto src_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(src.getType());
    ICHECK(src_ty)
        << "tl.sunmmio_transpose expects source to be a suvm.tile_view";

    mlir::Value dst = ctx_.LookupMLIRValue(operands[1].value);
    ICHECK(dst)
        << "Missing MLIR destination tile view for tl.sunmmio_transpose `"
        << operands[1].value << "`";
    auto dst_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(dst.getType());
    ICHECK(dst_ty)
        << "tl.sunmmio_transpose expects destination to be a suvm.tile_view";

    auto transpose_op = mlir::suvm::TransposeAsyncOp::create(
        ctx_.builder, type.MakeDebugLoc("sunmmio_transpose"), mlir::Type(), src,
        dst, require_odma_unit());
    VerifyAsyncOp(transpose_op, "tl.sunmmio_transpose");
    ctx_.AddPendingSyncUnits(mlir::suvm::SyncUnits::odma1);

    return SunMMIOValue{ret_dtype, result_name, ret_type};
  } else if (callee == "tl.broadcast_") {
    ICHECK(operands.size() == 3 || operands.size() == 4)
        << "tl.broadcast_ expects src, dst, mask, and optional src_core "
           "operands";

    mlir::Value src = ctx_.LookupMLIRValue(operands[0].value);
    ICHECK(src) << "Missing MLIR source tile view for tl.broadcast_ `"
                << operands[0].value << "`";
    auto src_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(src.getType());
    ICHECK(src_ty) << "tl.broadcast_ expects source to be a suvm.tile_view";

    mlir::Value dst = ctx_.LookupMLIRValue(operands[1].value);
    ICHECK(dst) << "Missing MLIR destination tile view for tl.broadcast_ `"
                << operands[1].value << "`";
    auto dst_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(dst.getType());
    ICHECK(dst_ty)
        << "tl.broadcast_ expects destination to be a suvm.tile_view";

    mlir::Value mask = ctx_.LookupMLIRValue(operands[2].value);
    if (!mask) {
      mask = type.ResolveValue(operands[2], ctx_.builder.getI64Type());
    }
    mask = ensure_i64(mask, "tl.broadcast_ mask");

    std::string direction_name =
        get_string_attr(SunMMIOCallAttrKey::kDirection).value_or("");
    ICHECK(direction_name == "row" || direction_name == "col")
        << "tl.broadcast_ direction must be encoded as row or col";
    auto direction = direction_name == "row" ? mlir::suvm::McastDirection::row
                                             : mlir::suvm::McastDirection::col;

    auto create_mcast = [&]() {
      auto mcast_op = mlir::suvm::MulticastTokOp::create(
          ctx_.builder, type.MakeDebugLoc("broadcast"), mlir::Type(), src, dst,
          mask, direction, require_odma_unit());
      VerifyA4EMulticastOp(mcast_op, "tl.broadcast_");
    };
    auto emit_link_sync = [&]() {
      mlir::suvm::SyncUnits link = direction == mlir::suvm::McastDirection::row
                                       ? mlir::suvm::SyncUnits::hlink
                                       : mlir::suvm::SyncUnits::vlink;
      mlir::suvm::SyncOp::create(
          ctx_.builder, type.MakeDebugLoc("broadcast_sync"),
          mlir::suvm::SyncUnitsAttr::get(&ctx_.mlir_ctx, link));
    };

    if (operands.size() == 4) {
      mlir::Value src_core = ctx_.LookupMLIRValue(operands[3].value);
      if (!src_core) {
        src_core = type.ResolveValue(operands[3], ctx_.builder.getI64Type());
      }
      src_core = ensure_i64(src_core, "tl.broadcast_ src_core");

      mlir::Value core_id = mlir::suvm::GetCoreIdOp::create(
                                ctx_.builder, type.MakeDebugLoc("get_core_id"))
                                .getResult();
      core_id = ensure_i64(core_id, "suvm.get_core_id result");
      mlir::Value is_src = mlir::arith::CmpIOp::create(
          ctx_.builder, type.Loc(), mlir::arith::CmpIPredicate::eq, core_id,
          src_core);

      auto if_op = mlir::scf::IfOp::create(ctx_.builder, type.Loc(),
                                           mlir::TypeRange{}, is_src,
                                           /*withElseRegion=*/false);

      mlir::Block &then_block = if_op.getThenRegion().front();
      ctx_.builder.setInsertionPointToStart(&then_block);
      create_mcast();
      emit_link_sync();

      ctx_.builder.setInsertionPointAfter(if_op);
    } else {
      create_mcast();
      ctx_.AddPendingSyncUnits(direction == mlir::suvm::McastDirection::row
                                   ? mlir::suvm::SyncUnits::hlink
                                   : mlir::suvm::SyncUnits::vlink);
    }

    return SunMMIOValue{ret_dtype, result_name, ret_type};
  } else if (callee == "tl.mma_sunmmio") {
    ICHECK_EQ(operands.size(), 4)
        << "tl.mma_sunmmio expects A/B/C tile views and an accumulate "
           "condition as operands";

    mlir::Value a = ctx_.LookupMLIRValue(operands[0].value);
    ICHECK(a) << "Missing MLIR activation tile view for tl.mma_sunmmio `"
              << operands[0].value << "`";
    auto a_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(a.getType());
    ICHECK(a_ty) << "tl.mma_sunmmio expects activation to be a suvm.tile_view";

    mlir::Value w = ctx_.LookupMLIRValue(operands[1].value);
    ICHECK(w) << "Missing MLIR weight tile view for tl.mma_sunmmio `"
              << operands[1].value << "`";
    auto w_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(w.getType());
    ICHECK(w_ty) << "tl.mma_sunmmio expects weight to be a suvm.tile_view";

    mlir::Value c = ctx_.LookupMLIRValue(operands[2].value);
    ICHECK(c) << "Missing MLIR accumulator tile view for tl.mma_sunmmio `"
              << operands[2].value << "`";
    auto c_ty = mlir::dyn_cast<mlir::suvm::TileViewType>(c.getType());
    ICHECK(c_ty) << "tl.mma_sunmmio expects accumulator to be a suvm.tile_view";

    mlir::Value accumulate = ctx_.LookupMLIRValue(operands[3].value);
    ICHECK(accumulate)
        << "Missing MLIR accumulate condition for tl.mma_sunmmio `"
        << operands[3].value << "`";
    accumulate = type.EnsureI1(accumulate);

    bool trans_a =
        require_bool_attr(SunMMIOCallAttrKey::kTransA, "tl.mma_sunmmio transA");
    ICHECK(!trans_a)
        << "tl.mma_sunmmio lowering to suvm.tc.mma does not support transA";
    bool trans_b =
        require_bool_attr(SunMMIOCallAttrKey::kTransB, "tl.mma_sunmmio transB");
    mlir::UnitAttr trans_attr =
        trans_b ? ctx_.builder.getUnitAttr() : mlir::UnitAttr();

    auto mma_op = mlir::suvm::TcMmaOp::create(
        ctx_.builder, type.MakeDebugLoc("mma_sunmmio"), mlir::Type(), c, a, w,
        c, accumulate, trans_attr);
    VerifyAsyncOp(mma_op, "tl.mma_sunmmio");
    ctx_.AddPendingSyncUnits(mlir::suvm::SyncUnits::tc);

    return SunMMIOValue{ret_dtype, result_name, ret_type};
  } else if (callee == "tir.ret") {
    ICHECK_EQ(operands.size(), 1) << "tir.ret expects one operand";

    const SunMMIOValue &ret = operands[0];
    bool is_zero = ret.value == "0";
    if (!is_zero) {
      mlir::Value ret_value = ctx_.LookupMLIRValue(ret.value);
      if (ret_value) {
        if (auto const_op =
                ret_value.getDefiningOp<mlir::arith::ConstantOp>()) {
          if (auto int_attr =
                  mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
            is_zero = int_attr.getInt() == 0;
          }
        }
      }
    }

    ICHECK(is_zero) << "SunMMIO device kernel only supports T.ret(0); got "
                    << ret.value;
    return SunMMIOValue{
        DataType::Void(), "",
        SunMMIOType{SunMMIOType::Kind::kUnknown, DataType::Void(), 1, {}}};
  } else {
    LOG(FATAL) << "Unsupported SunMMIO call lowering for `" << callee
               << "` (category=" << category << ", operands=" << operands.size()
               << ")";
    TVM_FFI_UNREACHABLE();
  }
}

} // namespace codegen
} // namespace tvm
