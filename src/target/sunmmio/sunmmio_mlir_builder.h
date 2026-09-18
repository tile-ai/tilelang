#ifndef TVM_TL_TARGET_SUNMMIO_MLIR_BUILDER_H_
#define TVM_TL_TARGET_SUNMMIO_MLIR_BUILDER_H_

#include "codegen_sunmmio.h"
#include "sunmmio_mlir_context.h"

#include <memory>

namespace tvm {
namespace codegen {

class SunmmioMlirFunction;
class SunmmioMlirExpr;
class SunmmioMlirMemory;
class SunmmioMlirTileOp;
class SunmmioMlirCall;

class SuvmSunmmioBuilder final : public SunMMIOBuilder {
public:
  SuvmSunmmioBuilder();
  ~SuvmSunmmioBuilder() final;

  void Init() final;
  void Clear() final;
  std::string Finish() final;

  void BeginModule() final;
  void EndModule() final;

  void BeginFunction(const std::string &name,
                     const std::vector<BuilderArg> &args) final;
  void EndFunction() final;
  void EmitReturn() final;

  SunMMIOValue ConstantInt(const std::string &result_name, int64_t v,
                           const SunMMIOType &type, DataType dtype) final;
  SunMMIOValue ConstantFloat(const std::string &result_name,
                             const std::string &literal,
                             const SunMMIOType &type, DataType dtype) final;

  SunMMIOValue Cast(const std::string &result_name, const SunMMIOValue &v,
                    const SunMMIOType &dst_type, DataType dst_dtype) final;

  SunMMIOValue Binary(const std::string &result_name, BinaryOp op,
                      ArithmeticFlavor flavor, const SunMMIOValue &a,
                      const SunMMIOValue &b, const SunMMIOType &result_type,
                      DataType dtype) final;

  SunMMIOValue Unary(const std::string &result_name, TileUnaryOp op,
                     const SunMMIOValue &data, const SunMMIOType &result_type,
                     DataType dtype) final;

  SunMMIOValue Compare(const std::string &result_name, CompareOp op,
                       CompareDomain domain, const SunMMIOValue &a,
                       const SunMMIOValue &b,
                       const SunMMIOType &operand_type) final;

  SunMMIOValue Select(const std::string &result_name, const SunMMIOValue &cond,
                      const SunMMIOValue &tv, const SunMMIOValue &fv,
                      const SunMMIOType &result_type, DataType dtype) final;

  SunMMIOValue BindValueAlias(const std::string &result_name,
                              const SunMMIOValue &value) final;

  SunMMIOValue
  BindLayout(const std::string &result_name, const SunMMIOValue &source,
             const std::vector<SunMMIOValue> &dynamic_shapes,
             const std::vector<SunMMIOValue> &dynamic_strides) final;

  SunMMIOValue Alloc(const std::string &result_name,
                     const SunMMIOType &memref_type,
                     const std::vector<SunMMIOValue> &dyn_extents,
                     const std::string &scope_name, DataType dtype,
                     std::optional<std::string> ping_pong) final;

  SunMMIOValue Load(const std::string &result_name,
                    const std::string &buffer_handle,
                    const std::vector<SunMMIOValue> &indices,
                    const SunMMIOType &memref_type, DataType dtype,
                    const SunMMIOType &result_type) final;

  SunMMIOValue GetPartitionedTileView(const std::string &result_name,
                                      const SunMMIOValue &memtensor,
                                      const std::vector<SunMMIOValue> &indices,
                                      const std::vector<int64_t> &tiled_dims,
                                      const SunMMIOType &view_type,
                                      DataType dtype) final;

  SunMMIOValue TileLoad(const std::string &result_name,
                        const SunMMIOValue &tile_view,
                        const SunMMIOType &tile_type,
                        const std::optional<SunMMIOValue> &mask,
                        const std::optional<SunMMIOValue> &maskedoff,
                        DataType dtype) final;

  SunMMIOValue TileFill(const std::string &result_name,
                        const SunMMIOValue &scalar,
                        const SunMMIOType &tile_type, DataType dtype) final;

  SunMMIOValue TileRange(const std::string &result_name,
                         const SunMMIOType &tile_type, DataType dtype) final;

  SunMMIOValue TileUnsqueeze(const std::string &result_name,
                             const SunMMIOValue &tile,
                             const SunMMIOType &tile_type, int64_t axis,
                             DataType dtype) final;

  SunMMIOValue TileBroadcast(const std::string &result_name,
                             const SunMMIOValue &tile,
                             const SunMMIOType &tile_type,
                             DataType dtype) final;

  SunMMIOValue TileSlice(const std::string &result_name,
                         const SunMMIOValue &tile,
                         const std::vector<SunMMIOValue> &offsets,
                         const SunMMIOType &tile_type, DataType dtype) final;

  SunMMIOValue TileInsertSlice(const std::string &result_name,
                               const SunMMIOValue &base,
                               const SunMMIOValue &slice,
                               const std::vector<SunMMIOValue> &offsets,
                               const SunMMIOType &result_type,
                               DataType dtype) final;

  SunMMIOValue TileRectMask(const std::string &result_name,
                            const SunMMIOValue &valid_rows,
                            const SunMMIOValue &valid_cols,
                            const SunMMIOType &tile_type,
                            DataType index_dtype) final;

  SunMMIOValue TileAxisMask(const std::string &result_name, int64_t axis,
                            const SunMMIOValue &valid_extent,
                            const SunMMIOType &tile_type,
                            DataType index_dtype) final;

  SunMMIOValue TileMaskAnd(const std::string &result_name,
                           const SunMMIOValue &lhs, const SunMMIOValue &rhs,
                           const SunMMIOType &tile_type) final;

  SunMMIOValue TileSelect(const std::string &result_name,
                          const SunMMIOValue &mask,
                          const SunMMIOValue &true_value,
                          const SunMMIOValue &false_value,
                          const SunMMIOType &result_type, DataType dtype) final;

  SunMMIOValue TileReduce(const std::string &result_name,
                          const std::string &predicate,
                          const SunMMIOValue &data,
                          const SunMMIOType &result_type, int64_t axis,
                          DataType dtype) final;

  SunMMIOValue TileSqueeze(const std::string &result_name,
                           const SunMMIOValue &tile,
                           const SunMMIOType &tile_type, int64_t axis,
                           DataType dtype) final;

  SunMMIOValue TilePick(const std::string &result_name,
                        const SunMMIOValue &tile,
                        const std::vector<SunMMIOValue> &indices,
                        const SunMMIOType &result_type, DataType dtype) final;

  SunMMIOValue TileSet(const std::string &result_name,
                       const SunMMIOValue &value, const SunMMIOValue &tile,
                       const std::vector<SunMMIOValue> &indices,
                       const SunMMIOType &result_type, DataType dtype) final;

  void Store(const SunMMIOValue &value, const std::string &buffer_handle,
             const std::vector<SunMMIOValue> &indices,
             const SunMMIOType &memref_type) final;

  void TileStore(const SunMMIOValue &value, const SunMMIOValue &tile_view,
                 const std::optional<SunMMIOValue> &mask) final;

  SunMMIOValue Call(const std::string &result_name, const std::string &callee,
                    const std::vector<SunMMIOValue> &operands,
                    const SunMMIOCallAttrs &attrs, const std::string &category,
                    DataType ret_dtype, const SunMMIOType &ret_type) final;

  SunMMIOValue RegionCall(const std::string &result_name,
                          const std::string &buffer_handle,
                          const std::vector<SunMMIOValue> &mins,
                          const std::vector<int64_t> &extents,
                          DataType ret_dtype, const SunMMIOType &ret_type,
                          int64_t byte_offset = 0) final;

  std::pair<SunMMIOValue, SunMMIOValue> MXUnpack(const std::string &scale_name,
                                                 const std::string &data_name,
                                                 const SunMMIOValue &mx,
                                                 DataType scale_dtype,
                                                 DataType data_dtype) final;

  SunMMIOValue Ramp(const std::string &result_name, const SunMMIOValue &base,
                    const SunMMIOValue &stride, int lanes,
                    const SunMMIOType &elem_type, const SunMMIOType &vec_type,
                    DataType dtype) final;

  SunMMIOValue Broadcast(const std::string &result_name,
                         const SunMMIOValue &scalar, int lanes,
                         const SunMMIOType &scalar_type,
                         const SunMMIOType &vec_type, DataType dtype) final;

  void BeginFor(const std::string &iv, const SunMMIOValue &lb,
                const SunMMIOValue &ub, const SunMMIOValue &step,
                const ffi::Map<ffi::String, ffi::Any> &annotations,
                const std::vector<SunMMIOValue> &live_out_values) final;
  void EndFor() final;

  void BeginIf(const SunMMIOValue &cond,
               const std::vector<SunMMIOValue> &live_out_values) final;
  void BeginElse() final;
  void EndIf() final;

  void BeginWhile(const std::vector<SunMMIOValue> &live_out_values) final;
  void BeginWhileBody(const SunMMIOValue &cond) final;
  void EndWhile() final;

  void EmitAssert(const SunMMIOValue &cond, const std::string &msg_text) final;
  SunMMIOValue GetCoreId(const std::string &result_name, DataType dtype) final;

  void PushLayoutScope(const TirLayoutMap &layout_map,
                       const TirLayoutMap &global_layout_map) final;
  void PopLayoutScope() final;
  ffi::Optional<tl::Layout> LookupLayout(const tir::Buffer &buffer) const final;
  void ApplyLayoutToType(const tir::Buffer &buffer,
                         SunMMIOType *type) const final;

  SunmmioMlirContext &Context() { return ctx_; }
  const SunmmioMlirContext &Context() const { return ctx_; }

private:
  SunmmioMlirContext ctx_;
  std::unique_ptr<SunmmioMlirFunction> function_;
  std::unique_ptr<SunmmioMlirExpr> expr_;
  std::unique_ptr<SunmmioMlirMemory> memory_;
  std::unique_ptr<SunmmioMlirTileOp> tile_;
  std::unique_ptr<SunmmioMlirCall> call_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_MLIR_BUILDER_H_
