#ifndef TVM_TL_TARGET_CODEGEN_SUNMMIO_H_
#define TVM_TL_TARGET_CODEGEN_SUNMMIO_H_

#include <tvm/ir/module.h>
#include <tvm/ir/type.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include "../../layout/layout.h"
#include "sunmmio_mlir_type.h"

#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tvm {
namespace codegen {

using SunMMIOCallAttrValue =
    std::variant<int64_t, bool, std::string, std::vector<int64_t>>;

using SunMMIOCallAttrs = std::unordered_map<std::string, SunMMIOCallAttrValue>;

namespace SunMMIOCallAttrKey {
constexpr const char *kUnit = "unit";
constexpr const char *kDirection = "direction";
constexpr const char *kTransA = "trans_a";
constexpr const char *kTransB = "trans_b";
constexpr const char *kParticipantMask = "participant_mask";
constexpr const char *kCandidateMasks = "candidate_masks";
constexpr const char *kBarrierMaskKey = "barrier_mask_key";
constexpr const char *kSyncUnits = "sync_units";
} // namespace SunMMIOCallAttrKey

enum class TileUnaryOp {
  kAbs,
  kCeil,
  kExp,
  kFloor,
  kLn,
  kNeg,
  kRecip,
  kRound,
  kRsqrt,
  kTrunc,
};

class SunMMIOBuilder {
public:
  virtual ~SunMMIOBuilder() = default;

  using TirLayoutMap = ffi::Map<tir::Buffer, tl::Layout>;

  virtual void Init() = 0;
  virtual void Clear() = 0;
  virtual std::string Finish() = 0;

  virtual void BeginModule() = 0;
  virtual void EndModule() = 0;

  virtual void BeginFunction(const std::string &name,
                             const std::vector<BuilderArg> &args) = 0;
  virtual void EndFunction() = 0;
  virtual void EmitReturn() = 0;

  virtual SunMMIOValue ConstantInt(const std::string &result_name, int64_t v,
                                   const SunMMIOType &type, DataType dtype) = 0;
  virtual SunMMIOValue ConstantFloat(const std::string &result_name,
                                     const std::string &literal,
                                     const SunMMIOType &type,
                                     DataType dtype) = 0;

  virtual SunMMIOValue Cast(const std::string &result_name,
                            const SunMMIOValue &v, const SunMMIOType &dst_type,
                            DataType dst_dtype) = 0;

  virtual SunMMIOValue Binary(const std::string &result_name, BinaryOp op,
                              ArithmeticFlavor flavor, const SunMMIOValue &a,
                              const SunMMIOValue &b,
                              const SunMMIOType &result_type,
                              DataType dtype) = 0;

  virtual SunMMIOValue Unary(const std::string &result_name, TileUnaryOp op,
                             const SunMMIOValue &data,
                             const SunMMIOType &result_type,
                             DataType dtype) = 0;

  virtual SunMMIOValue Compare(const std::string &result_name, CompareOp op,
                               CompareDomain domain, const SunMMIOValue &a,
                               const SunMMIOValue &b,
                               const SunMMIOType &operand_type) = 0;

  virtual SunMMIOValue Select(const std::string &result_name,
                              const SunMMIOValue &cond, const SunMMIOValue &tv,
                              const SunMMIOValue &fv,
                              const SunMMIOType &result_type,
                              DataType dtype) = 0;

  virtual SunMMIOValue BindValueAlias(const std::string &result_name,
                                      const SunMMIOValue &value) = 0;

  virtual SunMMIOValue
  BindLayout(const std::string &result_name, const SunMMIOValue &source,
             const std::vector<SunMMIOValue> &dynamic_shapes,
             const std::vector<SunMMIOValue> &dynamic_strides) = 0;

  virtual SunMMIOValue Alloc(const std::string &result_name,
                             const SunMMIOType &memref_type,
                             const std::vector<SunMMIOValue> &dyn_extents,
                             const std::string &scope_name, DataType dtype,
                             std::optional<std::string> ping_pong) = 0;

  virtual SunMMIOValue Load(const std::string &result_name,
                            const std::string &buffer_handle,
                            const std::vector<SunMMIOValue> &indices,
                            const SunMMIOType &memref_type, DataType dtype,
                            const SunMMIOType &result_type) = 0;

  virtual SunMMIOValue
  GetPartitionedTileView(const std::string &result_name,
                         const SunMMIOValue &memtensor,
                         const std::vector<SunMMIOValue> &indices,
                         const std::vector<int64_t> &tiled_dims,
                         const SunMMIOType &view_type, DataType dtype) = 0;

  virtual SunMMIOValue TileLoad(const std::string &result_name,
                                const SunMMIOValue &tile_view,
                                const SunMMIOType &tile_type,
                                const std::optional<SunMMIOValue> &mask,
                                const std::optional<SunMMIOValue> &maskedoff,
                                DataType dtype) = 0;

  virtual SunMMIOValue TileFill(const std::string &result_name,
                                const SunMMIOValue &scalar,
                                const SunMMIOType &tile_type,
                                DataType dtype) = 0;

  virtual SunMMIOValue TileRange(const std::string &result_name,
                                 const SunMMIOType &tile_type,
                                 DataType dtype) = 0;

  virtual SunMMIOValue TileUnsqueeze(const std::string &result_name,
                                     const SunMMIOValue &tile,
                                     const SunMMIOType &tile_type, int64_t axis,
                                     DataType dtype) = 0;

  virtual SunMMIOValue TileBroadcast(const std::string &result_name,
                                     const SunMMIOValue &tile,
                                     const SunMMIOType &tile_type,
                                     DataType dtype) = 0;

  virtual SunMMIOValue TileSlice(const std::string &result_name,
                                 const SunMMIOValue &tile,
                                 const std::vector<SunMMIOValue> &offsets,
                                 const SunMMIOType &tile_type,
                                 DataType dtype) = 0;

  virtual SunMMIOValue TileInsertSlice(const std::string &result_name,
                                       const SunMMIOValue &base,
                                       const SunMMIOValue &slice,
                                       const std::vector<SunMMIOValue> &offsets,
                                       const SunMMIOType &result_type,
                                       DataType dtype) = 0;

  virtual SunMMIOValue TileRectMask(const std::string &result_name,
                                    const SunMMIOValue &valid_rows,
                                    const SunMMIOValue &valid_cols,
                                    const SunMMIOType &tile_type,
                                    DataType index_dtype) = 0;

  virtual SunMMIOValue TileAxisMask(const std::string &result_name,
                                    int64_t axis,
                                    const SunMMIOValue &valid_extent,
                                    const SunMMIOType &tile_type,
                                    DataType index_dtype) = 0;

  virtual SunMMIOValue TileMaskAnd(const std::string &result_name,
                                   const SunMMIOValue &lhs,
                                   const SunMMIOValue &rhs,
                                   const SunMMIOType &tile_type) = 0;

  virtual SunMMIOValue
  TileSelect(const std::string &result_name, const SunMMIOValue &mask,
             const SunMMIOValue &true_value, const SunMMIOValue &false_value,
             const SunMMIOType &result_type, DataType dtype) = 0;

  virtual SunMMIOValue TileReduce(const std::string &result_name,
                                  const std::string &predicate,
                                  const SunMMIOValue &data,
                                  const SunMMIOType &result_type, int64_t axis,
                                  DataType dtype) = 0;

  virtual SunMMIOValue TileSqueeze(const std::string &result_name,
                                   const SunMMIOValue &tile,
                                   const SunMMIOType &tile_type, int64_t axis,
                                   DataType dtype) = 0;

  virtual SunMMIOValue TilePick(const std::string &result_name,
                                const SunMMIOValue &tile,
                                const std::vector<SunMMIOValue> &indices,
                                const SunMMIOType &result_type,
                                DataType dtype) = 0;

  virtual SunMMIOValue
  TileSet(const std::string &result_name, const SunMMIOValue &value,
          const SunMMIOValue &tile, const std::vector<SunMMIOValue> &indices,
          const SunMMIOType &result_type, DataType dtype) = 0;

  virtual void Store(const SunMMIOValue &value,
                     const std::string &buffer_handle,
                     const std::vector<SunMMIOValue> &indices,
                     const SunMMIOType &memref_type) = 0;

  virtual void TileStore(const SunMMIOValue &value,
                         const SunMMIOValue &tile_view,
                         const std::optional<SunMMIOValue> &mask) = 0;

  virtual SunMMIOValue Call(const std::string &result_name,
                            const std::string &callee,
                            const std::vector<SunMMIOValue> &operands,
                            const SunMMIOCallAttrs &attrs,
                            const std::string &category, DataType ret_dtype,
                            const SunMMIOType &ret_type) = 0;

  virtual SunMMIOValue
  RegionCall(const std::string &result_name, const std::string &buffer_handle,
             const std::vector<SunMMIOValue> &mins,
             const std::vector<int64_t> &extents, DataType ret_dtype,
             const SunMMIOType &ret_type, int64_t byte_offset = 0) = 0;

  virtual std::pair<SunMMIOValue, SunMMIOValue>
  MXUnpack(const std::string &scale_name, const std::string &data_name,
           const SunMMIOValue &mx, DataType scale_dtype,
           DataType data_dtype) = 0;

  virtual SunMMIOValue Ramp(const std::string &result_name,
                            const SunMMIOValue &base,
                            const SunMMIOValue &stride, int lanes,
                            const SunMMIOType &elem_type,
                            const SunMMIOType &vec_type, DataType dtype) = 0;

  virtual SunMMIOValue Broadcast(const std::string &result_name,
                                 const SunMMIOValue &scalar, int lanes,
                                 const SunMMIOType &scalar_type,
                                 const SunMMIOType &vec_type,
                                 DataType dtype) = 0;

  virtual void BeginFor(const std::string &iv, const SunMMIOValue &lb,
                        const SunMMIOValue &ub, const SunMMIOValue &step,
                        const ffi::Map<ffi::String, ffi::Any> &annotations,
                        const std::vector<SunMMIOValue> &live_out_values) = 0;
  virtual void EndFor() = 0;

  virtual void BeginIf(const SunMMIOValue &cond,
                       const std::vector<SunMMIOValue> &live_out_values) = 0;
  virtual void BeginElse() = 0;
  virtual void EndIf() = 0;

  virtual void BeginWhile(const std::vector<SunMMIOValue> &live_out_values) = 0;
  virtual void BeginWhileBody(const SunMMIOValue &cond) = 0;
  virtual void EndWhile() = 0;

  virtual void EmitAssert(const SunMMIOValue &cond,
                          const std::string &msg_text) = 0;
  virtual SunMMIOValue GetCoreId(const std::string &result_name,
                                 DataType dtype) = 0;

  virtual void PushLayoutScope(const TirLayoutMap &layout_map,
                               const TirLayoutMap &global_layout_map) {
    (void)layout_map;
    (void)global_layout_map;
  }
  virtual void PopLayoutScope() {}
  virtual ffi::Optional<tl::Layout>
  LookupLayout(const tir::Buffer &buffer) const {
    (void)buffer;
    return ffi::Optional<tl::Layout>();
  }
  virtual void ApplyLayoutToType(const tir::Buffer &buffer,
                                 SunMMIOType *type) const {
    (void)buffer;
    (void)type;
  }
};

class CodeGenTileLangSunMMIO final
    : public tir::StmtVisitor,
      public tir::ExprFunctor<SunMMIOValue(const tvm::PrimExpr &)> {
public:
  CodeGenTileLangSunMMIO();
  ~CodeGenTileLangSunMMIO() noexcept override = default;

  void SetTarget(tvm::Target target);
  void Init();
  void Clear();
  void AddFunction(const GlobalVar &gvar, const tir::PrimFunc &f);
  std::string Finish();

protected:
  void VisitStmt_(const tir::SeqStmtNode *op) override;
  void VisitStmt_(const tir::ForNode *op) override;
  void VisitStmt_(const tir::LetStmtNode *op) override;
  void VisitStmt_(const tir::AttrStmtNode *op) override;
  void VisitStmt_(const tir::IfThenElseNode *op) override;
  void VisitStmt_(const tir::WhileNode *op) override;
  void VisitStmt_(const tir::AllocateNode *op) override;
  void VisitStmt_(const tir::AllocateConstNode *op) override;
  void VisitStmt_(const tir::DeclBufferNode *op) override;
  void VisitStmt_(const tir::BufferStoreNode *op) override;
  void VisitStmt_(const tir::BufferRealizeNode *op) override;
  void VisitStmt_(const tir::AssertStmtNode *op) override;
  void VisitStmt_(const tir::EvaluateNode *op) override;
  void VisitStmt_(const tir::BlockNode *op) override;
  void VisitStmt_(const tir::BlockRealizeNode *op) override;
  void VisitStmtDefault_(const Object *op) override;

  SunMMIOValue VisitExpr_(const tir::VarNode *op) override;
  SunMMIOValue VisitExpr_(const tir::SizeVarNode *op) override;
  SunMMIOValue VisitExpr_(const tir::IntImmNode *op) override;
  SunMMIOValue VisitExpr_(const tir::FloatImmNode *op) override;
  SunMMIOValue VisitExpr_(const tir::StringImmNode *op) override;
  SunMMIOValue VisitExpr_(const tir::CastNode *op) override;
  SunMMIOValue VisitExpr_(const tir::CallNode *op) override;
  SunMMIOValue VisitExpr_(const tir::AddNode *op) override;
  SunMMIOValue VisitExpr_(const tir::SubNode *op) override;
  SunMMIOValue VisitExpr_(const tir::MulNode *op) override;
  SunMMIOValue VisitExpr_(const tir::DivNode *op) override;
  SunMMIOValue VisitExpr_(const tir::ModNode *op) override;
  SunMMIOValue VisitExpr_(const tir::FloorDivNode *op) override;
  SunMMIOValue VisitExpr_(const tir::FloorModNode *op) override;
  SunMMIOValue VisitExpr_(const tir::MinNode *op) override;
  SunMMIOValue VisitExpr_(const tir::MaxNode *op) override;
  SunMMIOValue VisitExpr_(const tir::EQNode *op) override;
  SunMMIOValue VisitExpr_(const tir::NENode *op) override;
  SunMMIOValue VisitExpr_(const tir::LTNode *op) override;
  SunMMIOValue VisitExpr_(const tir::LENode *op) override;
  SunMMIOValue VisitExpr_(const tir::GTNode *op) override;
  SunMMIOValue VisitExpr_(const tir::GENode *op) override;
  SunMMIOValue VisitExpr_(const tir::AndNode *op) override;
  SunMMIOValue VisitExpr_(const tir::OrNode *op) override;
  SunMMIOValue VisitExpr_(const tir::NotNode *op) override;
  SunMMIOValue VisitExpr_(const tir::SelectNode *op) override;
  SunMMIOValue VisitExpr_(const tir::BufferLoadNode *op) override;
  SunMMIOValue VisitExpr_(const tir::ProducerLoadNode *op) override;
  SunMMIOValue VisitExpr_(const tir::RampNode *op) override;
  SunMMIOValue VisitExpr_(const tir::BroadcastNode *op) override;
  SunMMIOValue VisitExpr_(const tir::ShuffleNode *op) override;
  SunMMIOValue VisitExpr_(const tir::LetNode *op) override;
  SunMMIOValue VisitExprDefault_(const Object *op) override;

private:
  enum class CoverageDomain { kMain, kTiles };

  struct CoverageData {
    std::set<std::string> expected_node_types;
    std::set<std::string> visited_node_types;
    std::set<std::string> expected_call_ops;
    std::set<std::string> visited_call_ops;
  };

  enum class CallBucket {
    kBuiltin,
    kExternPure,
    kExternSideEffect,
    kMath,
    kMemory,
    kSync,
    kVector,
    kTileLangIntrinsic,
    kSunMMIOIntrinsic,
    kUnsupported
  };

  struct ScopedAttr {
    ffi::Any node;
    ffi::String key;
    SunMMIOValue value;
  };

  struct PendingExternalBuffer {
    tir::Buffer buffer;
    std::string handle;
    SunMMIOType type;
  };

  SunMMIOValue EvalExpr(const tvm::PrimExpr &expr);
  void VisitStmtTracked(const tir::Stmt &stmt);
  void CollectExpectedCoverage(const tir::PrimFunc &f);
  void CollectDeclBuffers(const tir::Stmt &stmt);
  bool IsSunmmioReduceRegisterTempBuffer(const tir::Buffer &buffer) const;
  bool IsSunmmioReduceLoopCarriedTempBuffer(const tir::Buffer &buffer) const;
  bool IsSunmmioReduceLocalTempBuffer(const tir::Buffer &buffer) const;
  void MarkVisitedNodeType(const std::string &type_key);
  void MarkVisitedCallOpFromExpr(const tvm::PrimExpr &expr);
  void MarkVisitedExprRoot(const tvm::PrimExpr &expr);
  void MarkVisitedExprTree(const tvm::PrimExpr &expr);
  tir::BufferRegion NormalizeRegionTracked(const tvm::PrimExpr &expr);
  bool TryConsumeSunmmioOdmaUnit(const tvm::PrimExpr &expr,
                                 SunMMIOCallAttrs *attrs);
  void WriteCoverageReport() const;
  void CheckCoverageOrFail() const;
  SunMMIOValue EmitBinary(const char *op_name, const tvm::PrimExpr &lhs,
                          const tvm::PrimExpr &rhs, tvm::DataType dtype);
  SunMMIOValue EmitCmp(const char *pred, const tvm::PrimExpr &lhs,
                       const tvm::PrimExpr &rhs);
  SunMMIOValue EmitCast(const SunMMIOValue &v, tvm::DataType target_dtype);
  SunMMIOValue EmitCall(const tir::CallNode *op);
  SunMMIOValue EmitMXPackOrUnpack(const tir::CallNode *op, bool is_pack);
  SunMMIOValue EmitRegionCall(const tvm::PrimExpr &region_expr,
                              int64_t byte_offset = 0);
  SunMMIOValue EmitLoad(const tir::Buffer &buffer,
                        const ffi::Array<PrimExpr> &indices);
  void EmitStore(const tir::Buffer &buffer, const ffi::Array<PrimExpr> &indices,
                 const SunMMIOValue &value);
  void EmitAlloc(const tir::Buffer &buffer, const std::string &scope_hint,
                 const ffi::Map<ffi::String, ffi::Any> &annotations);
  void EmitLocalVarAlloc(const tir::AllocateNode *op,
                         const tir::Buffer &buffer);
  SunMMIOValue EmitLocalVarLoad(const tir::Buffer &buffer,
                                const ffi::Array<PrimExpr> &indices);
  void EmitLocalVarStore(const tir::Buffer &buffer,
                         const ffi::Array<PrimExpr> &indices,
                         const SunMMIOValue &value);
  SunMMIOValue EmitScalarTilePick(const tir::BufferLoadNode *op);
  void EmitScalarTileSet(const tir::BufferStoreNode *op);
  std::vector<SunMMIOValue>
  CollectLocalVarLiveOutValues(const tir::Stmt &stmt) const;
  bool TryLowerTilesScope(const tir::ForNode *op);
  void EmitFor(const tir::ForNode *op);
  void EmitIf(const tir::IfThenElseNode *op);
  void EmitWhile(const tir::WhileNode *op);

  SunMMIOType MapType(tvm::DataType dtype) const;
  SunMMIOType MapBufferType(const tir::Buffer &buffer) const;
  std::string MapStorageScope(const std::string &scope) const;
  std::string NewValueName();
  SunMMIOValue EmitConstIndex(int64_t v);
  SunMMIOValue EnsureIndex(const SunMMIOValue &v);
  SunMMIOValue EnsureType(const SunMMIOValue &v, const SunMMIOType &target_type,
                          DataType dtype);
  ArithmeticFlavor GetArithmeticFlavor(DataType dtype) const;
  CompareDomain GetCompareDomain(DataType dtype) const;
  SunMMIOValue BindVar(const tir::Var &var, const SunMMIOValue &value);
  const SunMMIOValue &LookupVar(const tir::VarNode *var) const;
  const SunMMIOValue &LookupLocalVar(const tir::VarNode *var) const;
  SunMMIOValue MaterializeDynamicLayoutExpr(const tvm::PrimExpr &expr);
  std::vector<SunMMIOValue>
  CollectDynamicLayoutValues(const std::vector<PrimExpr> &exprs);
  void BindExternalBufferLayout(const PendingExternalBuffer &pending);
  void RegisterBuffer(const tir::Buffer &buffer, bool is_external,
                      const std::string &handle_hint = "");
  const BufferBinding &LookupBuffer(const tir::Buffer &buffer) const;
  void EnterScope();
  void ExitScope();

  CallBucket ClassifyCall(const tir::CallNode *op) const;
  const char *CallBucketName(CallBucket bucket) const;
  [[noreturn]] void UnsupportedStmt(const Object *op,
                                    const std::string &detail = "") const;
  [[noreturn]] void UnsupportedExpr(const Object *op,
                                    const std::string &detail = "") const;

  std::unique_ptr<SunMMIOBuilder> builder_;
  tvm::Target target_;
  bool initialized_{false};
  int ssa_counter_{0};

  std::unordered_map<const tir::VarNode *, SunMMIOValue> var_table_;
  std::unordered_map<const tir::VarNode *, SunMMIOValue> local_var_table_;
  std::unordered_map<const tir::BufferNode *, BufferBinding> buffer_registry_;
  std::unordered_map<const tir::VarNode *, tir::Buffer> buffer_data_to_buffer_;
  std::unordered_map<const tir::VarNode *, int> reduce_register_temp_roles_;
  std::vector<ScopedAttr> attr_stack_;

  std::vector<const tir::VarNode *> scoped_vars_;
  std::vector<const tir::VarNode *> scoped_local_vars_;
  std::vector<const tir::BufferNode *> scoped_buffers_;
  std::vector<size_t> var_scope_markers_;
  std::vector<size_t> local_var_scope_markers_;
  std::vector<size_t> buffer_scope_markers_;

  // Main and Tiles lowering are checked independently so the main path cannot
  // hide a missing type/op mark in custom Tiles lowering.
  CoverageDomain coverage_domain_{CoverageDomain::kMain};
  CoverageData main_coverage_;
  CoverageData tiles_coverage_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_SUNMMIO_H_
