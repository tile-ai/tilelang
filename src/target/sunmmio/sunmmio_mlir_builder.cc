#include "sunmmio_mlir_builder.h"

#include "sunmmio_mlir_call.h"
#include "sunmmio_mlir_expr.h"
#include "sunmmio_mlir_function.h"
#include "sunmmio_mlir_memory.h"
#include "sunmmio_mlir_tile_op.h"

#include "npuir/Dialect/SUVM/IR/Ops.h"
#include "llvm/Support/raw_ostream.h"

namespace tvm {
namespace codegen {

SuvmSunmmioBuilder::SuvmSunmmioBuilder()
    : function_(std::make_unique<SunmmioMlirFunction>(ctx_)),
      expr_(std::make_unique<SunmmioMlirExpr>(ctx_)),
      memory_(std::make_unique<SunmmioMlirMemory>(ctx_)),
      tile_(std::make_unique<SunmmioMlirTileOp>(ctx_)),
      call_(std::make_unique<SunmmioMlirCall>(ctx_)) {}

SuvmSunmmioBuilder::~SuvmSunmmioBuilder() = default;

void SuvmSunmmioBuilder::Init() { Clear(); }

void SuvmSunmmioBuilder::Clear() { ctx_.Clear(); }

std::string SuvmSunmmioBuilder::Finish() {
  ICHECK(ctx_.module)
      << "SuvmSunmmioBuilder::Finish called without an initialized MLIR module";
  std::string out;
  llvm::raw_string_ostream os(out);
  ctx_.module->print(os);
  return os.str();
}

void SuvmSunmmioBuilder::BeginModule() { function_->BeginModule(); }
void SuvmSunmmioBuilder::EndModule() { function_->EndModule(); }

void SuvmSunmmioBuilder::BeginFunction(const std::string &name,
                                       const std::vector<BuilderArg> &args) {
  function_->BeginFunction(name, args);
}

void SuvmSunmmioBuilder::EndFunction() { function_->EndFunction(); }
void SuvmSunmmioBuilder::EmitReturn() { function_->EmitReturn(); }

SunMMIOValue SuvmSunmmioBuilder::ConstantInt(const std::string &result_name,
                                             int64_t v, const SunMMIOType &type,
                                             DataType dtype) {
  return expr_->ConstantInt(result_name, v, type, dtype);
}

SunMMIOValue SuvmSunmmioBuilder::ConstantFloat(const std::string &result_name,
                                               const std::string &literal,
                                               const SunMMIOType &type,
                                               DataType dtype) {
  return expr_->ConstantFloat(result_name, literal, type, dtype);
}

SunMMIOValue SuvmSunmmioBuilder::Cast(const std::string &result_name,
                                      const SunMMIOValue &v,
                                      const SunMMIOType &dst_type,
                                      DataType dst_dtype) {
  if (v.type.kind == SunMMIOType::Kind::kTile &&
      dst_type.kind == SunMMIOType::Kind::kTile) {
    return tile_->Cast(result_name, v, dst_type, dst_dtype);
  }
  return expr_->Cast(result_name, v, dst_type, dst_dtype);
}

SunMMIOValue SuvmSunmmioBuilder::Binary(const std::string &result_name,
                                        BinaryOp op, ArithmeticFlavor flavor,
                                        const SunMMIOValue &a,
                                        const SunMMIOValue &b,
                                        const SunMMIOType &result_type,
                                        DataType dtype) {
  if (result_type.kind == SunMMIOType::Kind::kTile) {
    return tile_->Binary(result_name, op, flavor, a, b, result_type, dtype);
  }
  return expr_->Binary(result_name, op, flavor, a, b, result_type, dtype);
}

SunMMIOValue SuvmSunmmioBuilder::Unary(const std::string &result_name,
                                       TileUnaryOp op, const SunMMIOValue &data,
                                       const SunMMIOType &result_type,
                                       DataType dtype) {
  return tile_->Unary(result_name, op, data, result_type, dtype);
}

SunMMIOValue SuvmSunmmioBuilder::Compare(const std::string &result_name,
                                         CompareOp op, CompareDomain domain,
                                         const SunMMIOValue &a,
                                         const SunMMIOValue &b,
                                         const SunMMIOType &operand_type) {
  if (operand_type.kind == SunMMIOType::Kind::kTile) {
    return tile_->Compare(result_name, op, domain, a, b, operand_type);
  }
  return expr_->Compare(result_name, op, domain, a, b, operand_type);
}

SunMMIOValue SuvmSunmmioBuilder::Select(const std::string &result_name,
                                        const SunMMIOValue &cond,
                                        const SunMMIOValue &tv,
                                        const SunMMIOValue &fv,
                                        const SunMMIOType &result_type,
                                        DataType dtype) {
  return expr_->Select(result_name, cond, tv, fv, result_type, dtype);
}

SunMMIOValue SuvmSunmmioBuilder::BindValueAlias(const std::string &result_name,
                                                const SunMMIOValue &value) {
  mlir::Value mlir_value = ctx_.LookupMLIRValue(value.value);
  ICHECK(mlir_value) << "Cannot alias missing MLIR value " << value.value;
  ctx_.BindMLIRValue(result_name, mlir_value);
  return SunMMIOValue{value.dtype, result_name, value.type};
}

SunMMIOValue SuvmSunmmioBuilder::BindLayout(
    const std::string &result_name, const SunMMIOValue &source,
    const std::vector<SunMMIOValue> &dynamic_shapes,
    const std::vector<SunMMIOValue> &dynamic_strides) {
  mlir::Value source_value = ctx_.LookupMLIRValue(source.value);
  ICHECK(source_value) << "Cannot bind layout for missing MLIR value "
                       << source.value;

  SunmmioMlirType type(ctx_);
  mlir::SmallVector<mlir::Value, 4> shape_values;
  mlir::SmallVector<mlir::Value, 4> stride_values;
  shape_values.reserve(dynamic_shapes.size());
  stride_values.reserve(dynamic_strides.size());
  for (const SunMMIOValue &value : dynamic_shapes) {
    mlir::Value mlir_value = ctx_.LookupMLIRValue(value.value);
    ICHECK(mlir_value) << "Missing dynamic layout shape value " << value.value;
    shape_values.push_back(type.EnsureIndex(mlir_value));
  }
  for (const SunMMIOValue &value : dynamic_strides) {
    mlir::Value mlir_value = ctx_.LookupMLIRValue(value.value);
    ICHECK(mlir_value) << "Missing dynamic layout stride value " << value.value;
    stride_values.push_back(type.EnsureIndex(mlir_value));
  }

  auto bind_op = mlir::suvm::BindLayoutOp::create(
      ctx_.builder, type.MakeDebugLoc("bind_layout"), source_value.getType(),
      source_value, shape_values, stride_values);
  ctx_.BindMLIRValue(result_name, bind_op.getResult());
  return SunMMIOValue{source.dtype, result_name, source.type};
}

SunMMIOValue SuvmSunmmioBuilder::Alloc(
    const std::string &result_name, const SunMMIOType &memref_type,
    const std::vector<SunMMIOValue> &dyn_extents, const std::string &scope_name,
    DataType dtype, std::optional<std::string> ping_pong) {
  return memory_->Alloc(result_name, memref_type, dyn_extents, scope_name,
                        dtype, std::move(ping_pong));
}

SunMMIOValue SuvmSunmmioBuilder::Load(const std::string &result_name,
                                      const std::string &buffer_handle,
                                      const std::vector<SunMMIOValue> &indices,
                                      const SunMMIOType &memref_type,
                                      DataType dtype,
                                      const SunMMIOType &result_type) {
  return memory_->Load(result_name, buffer_handle, indices, memref_type, dtype,
                       result_type);
}

SunMMIOValue SuvmSunmmioBuilder::GetPartitionedTileView(
    const std::string &result_name, const SunMMIOValue &memtensor,
    const std::vector<SunMMIOValue> &indices,
    const std::vector<int64_t> &tiled_dims, const SunMMIOType &view_type,
    DataType dtype) {
  return tile_->GetPartitionedTileView(result_name, memtensor, indices,
                                       tiled_dims, view_type, dtype);
}

SunMMIOValue SuvmSunmmioBuilder::TileLoad(
    const std::string &result_name, const SunMMIOValue &tile_view,
    const SunMMIOType &tile_type, const std::optional<SunMMIOValue> &mask,
    const std::optional<SunMMIOValue> &maskedoff, DataType dtype) {
  return tile_->TileLoad(result_name, tile_view, tile_type, mask, maskedoff,
                         dtype);
}

SunMMIOValue SuvmSunmmioBuilder::TileFill(const std::string &result_name,
                                          const SunMMIOValue &scalar,
                                          const SunMMIOType &tile_type,
                                          DataType dtype) {
  return tile_->TileFill(result_name, scalar, tile_type, dtype);
}

SunMMIOValue SuvmSunmmioBuilder::TileRange(const std::string &result_name,
                                           const SunMMIOType &tile_type,
                                           DataType dtype) {
  return tile_->TileRange(result_name, tile_type, dtype);
}

SunMMIOValue SuvmSunmmioBuilder::TileUnsqueeze(const std::string &result_name,
                                               const SunMMIOValue &tile,
                                               const SunMMIOType &tile_type,
                                               int64_t axis, DataType dtype) {
  return tile_->TileUnsqueeze(result_name, tile, tile_type, axis, dtype);
}

SunMMIOValue SuvmSunmmioBuilder::TileBroadcast(const std::string &result_name,
                                               const SunMMIOValue &tile,
                                               const SunMMIOType &tile_type,
                                               DataType dtype) {
  return tile_->TileBroadcast(result_name, tile, tile_type, dtype);
}

SunMMIOValue
SuvmSunmmioBuilder::TileSlice(const std::string &result_name,
                              const SunMMIOValue &tile,
                              const std::vector<SunMMIOValue> &offsets,
                              const SunMMIOType &tile_type, DataType dtype) {
  return tile_->TileSlice(result_name, tile, offsets, tile_type, dtype);
}

SunMMIOValue SuvmSunmmioBuilder::TileInsertSlice(
    const std::string &result_name, const SunMMIOValue &base,
    const SunMMIOValue &slice, const std::vector<SunMMIOValue> &offsets,
    const SunMMIOType &result_type, DataType dtype) {
  return tile_->TileInsertSlice(result_name, base, slice, offsets, result_type,
                                dtype);
}

SunMMIOValue SuvmSunmmioBuilder::TileRectMask(const std::string &result_name,
                                              const SunMMIOValue &valid_rows,
                                              const SunMMIOValue &valid_cols,
                                              const SunMMIOType &tile_type,
                                              DataType index_dtype) {
  return tile_->TileRectMask(result_name, valid_rows, valid_cols, tile_type,
                             index_dtype);
}

SunMMIOValue SuvmSunmmioBuilder::TileAxisMask(const std::string &result_name,
                                              int64_t axis,
                                              const SunMMIOValue &valid_extent,
                                              const SunMMIOType &tile_type,
                                              DataType index_dtype) {
  return tile_->TileAxisMask(result_name, axis, valid_extent, tile_type,
                             index_dtype);
}

SunMMIOValue SuvmSunmmioBuilder::TileMaskAnd(const std::string &result_name,
                                             const SunMMIOValue &lhs,
                                             const SunMMIOValue &rhs,
                                             const SunMMIOType &tile_type) {
  return tile_->TileMaskAnd(result_name, lhs, rhs, tile_type);
}

SunMMIOValue SuvmSunmmioBuilder::TileSelect(const std::string &result_name,
                                            const SunMMIOValue &mask,
                                            const SunMMIOValue &true_value,
                                            const SunMMIOValue &false_value,
                                            const SunMMIOType &result_type,
                                            DataType dtype) {
  return tile_->TileSelect(result_name, mask, true_value, false_value,
                           result_type, dtype);
}

SunMMIOValue SuvmSunmmioBuilder::TileReduce(const std::string &result_name,
                                            const std::string &predicate,
                                            const SunMMIOValue &data,
                                            const SunMMIOType &result_type,
                                            int64_t axis, DataType dtype) {
  return tile_->TileReduce(result_name, predicate, data, result_type, axis,
                           dtype);
}

SunMMIOValue SuvmSunmmioBuilder::TileSqueeze(const std::string &result_name,
                                             const SunMMIOValue &tile,
                                             const SunMMIOType &tile_type,
                                             int64_t axis, DataType dtype) {
  return tile_->TileSqueeze(result_name, tile, tile_type, axis, dtype);
}

SunMMIOValue
SuvmSunmmioBuilder::TilePick(const std::string &result_name,
                             const SunMMIOValue &tile,
                             const std::vector<SunMMIOValue> &indices,
                             const SunMMIOType &result_type, DataType dtype) {
  return tile_->TilePick(result_name, tile, indices, result_type, dtype);
}

SunMMIOValue
SuvmSunmmioBuilder::TileSet(const std::string &result_name,
                            const SunMMIOValue &value, const SunMMIOValue &tile,
                            const std::vector<SunMMIOValue> &indices,
                            const SunMMIOType &result_type, DataType dtype) {
  return tile_->TileSet(result_name, value, tile, indices, result_type, dtype);
}

void SuvmSunmmioBuilder::Store(const SunMMIOValue &value,
                               const std::string &buffer_handle,
                               const std::vector<SunMMIOValue> &indices,
                               const SunMMIOType &memref_type) {
  memory_->Store(value, buffer_handle, indices, memref_type);
}

void SuvmSunmmioBuilder::TileStore(const SunMMIOValue &value,
                                   const SunMMIOValue &tile_view,
                                   const std::optional<SunMMIOValue> &mask) {
  tile_->TileStore(value, tile_view, mask);
}

SunMMIOValue SuvmSunmmioBuilder::Call(const std::string &result_name,
                                      const std::string &callee,
                                      const std::vector<SunMMIOValue> &operands,
                                      const SunMMIOCallAttrs &attrs,
                                      const std::string &category,
                                      DataType ret_dtype,
                                      const SunMMIOType &ret_type) {
  return call_->Call(result_name, callee, operands, attrs, category, ret_dtype,
                     ret_type);
}

SunMMIOValue SuvmSunmmioBuilder::RegionCall(
    const std::string &result_name, const std::string &buffer_handle,
    const std::vector<SunMMIOValue> &mins, const std::vector<int64_t> &extents,
    DataType ret_dtype, const SunMMIOType &ret_type, int64_t byte_offset) {
  return call_->RegionCall(result_name, buffer_handle, mins, extents, ret_dtype,
                           ret_type, byte_offset);
}

std::pair<SunMMIOValue, SunMMIOValue> SuvmSunmmioBuilder::MXUnpack(
    const std::string &scale_name, const std::string &data_name,
    const SunMMIOValue &mx, DataType scale_dtype, DataType data_dtype) {
  return call_->MXUnpack(scale_name, data_name, mx, scale_dtype, data_dtype);
}

SunMMIOValue SuvmSunmmioBuilder::Ramp(const std::string &result_name,
                                      const SunMMIOValue &base,
                                      const SunMMIOValue &stride, int lanes,
                                      const SunMMIOType &elem_type,
                                      const SunMMIOType &vec_type,
                                      DataType dtype) {
  return expr_->Ramp(result_name, base, stride, lanes, elem_type, vec_type,
                     dtype);
}

SunMMIOValue SuvmSunmmioBuilder::Broadcast(const std::string &result_name,
                                           const SunMMIOValue &scalar,
                                           int lanes,
                                           const SunMMIOType &scalar_type,
                                           const SunMMIOType &vec_type,
                                           DataType dtype) {
  return expr_->Broadcast(result_name, scalar, lanes, scalar_type, vec_type,
                          dtype);
}

void SuvmSunmmioBuilder::BeginFor(
    const std::string &iv, const SunMMIOValue &lb, const SunMMIOValue &ub,
    const SunMMIOValue &step,
    const ffi::Map<ffi::String, ffi::Any> &annotations,
    const std::vector<SunMMIOValue> &live_out_values) {
  function_->BeginFor(iv, lb, ub, step, annotations, live_out_values);
}

void SuvmSunmmioBuilder::EndFor() { function_->EndFor(); }

void SuvmSunmmioBuilder::BeginIf(
    const SunMMIOValue &cond,
    const std::vector<SunMMIOValue> &live_out_values) {
  function_->BeginIf(cond, live_out_values);
}

void SuvmSunmmioBuilder::BeginElse() { function_->BeginElse(); }

void SuvmSunmmioBuilder::EndIf() { function_->EndIf(); }

void SuvmSunmmioBuilder::BeginWhile(
    const std::vector<SunMMIOValue> &live_out_values) {
  function_->BeginWhile(live_out_values);
}

void SuvmSunmmioBuilder::BeginWhileBody(const SunMMIOValue &cond) {
  function_->BeginWhileBody(cond);
}

void SuvmSunmmioBuilder::EndWhile() { function_->EndWhile(); }

void SuvmSunmmioBuilder::EmitAssert(const SunMMIOValue &cond,
                                    const std::string &msg_text) {
  function_->EmitAssert(cond, msg_text);
}

SunMMIOValue SuvmSunmmioBuilder::GetCoreId(const std::string &result_name,
                                           DataType dtype) {
  SunMMIOType i64_type{SunMMIOType::Kind::kScalar, DataType::Int(64), 1, {}};
  auto core_id = mlir::suvm::GetCoreIdOp::create(
      ctx_.builder, SunmmioMlirType(ctx_).MakeDebugLoc("get_core_id"),
      ctx_.builder.getI64Type());
  ctx_.BindMLIRValue(result_name, core_id.getResult());
  SunMMIOValue value{DataType::Int(64), result_name, i64_type};
  SunMMIOType dst_type{SunMMIOType::Kind::kScalar, dtype, 1, {}};
  if (dtype == DataType::Int(64)) {
    return value;
  }
  return expr_->Cast(result_name, value, dst_type, dtype);
}

void SuvmSunmmioBuilder::PushLayoutScope(
    const TirLayoutMap &layout_map, const TirLayoutMap &global_layout_map) {
  ctx_.PushLayoutScope(layout_map, global_layout_map);
}

void SuvmSunmmioBuilder::PopLayoutScope() { ctx_.PopLayoutScope(); }

ffi::Optional<tl::Layout>
SuvmSunmmioBuilder::LookupLayout(const tir::Buffer &buffer) const {
  return ctx_.LookupLayout(buffer);
}

void SuvmSunmmioBuilder::ApplyLayoutToType(const tir::Buffer &buffer,
                                           SunMMIOType *type) const {
  ctx_.ApplyLayoutToType(buffer, type);
}

} // namespace codegen
} // namespace tvm
