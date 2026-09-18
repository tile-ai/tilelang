#include "codegen_sunmmio.h"
#include "sunmmio_mlir_builder.h"

#include "../../layout/cute_layout.h"
#include "../../layout/layout.h"
#include "../../op/builtin.h"
#include "../../op/comm.h"
#include "../../op/region.h"
#include "../../op/utils.h"
#include "../../tileview/tileview_planner_common.h"
#include "../../transform/common/attr.h"
#include "../sunmmio_utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ir/type.h>
#include <tvm/node/script_printer.h>
#include <tvm/node/structural_equal.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <tvm/runtime/logging.h>

namespace tvm {
namespace codegen {
using namespace tir;

namespace {

class DeclBufferCollector final : public tir::StmtVisitor {
public:
  std::unordered_map<const tir::VarNode *, tir::Buffer> buffer_data_to_buffer;
  std::unordered_map<const tir::VarNode *, int> reduce_register_temp_roles;

private:
  void VisitStmt_(const tir::DeclBufferNode *op) final {
    Record(op->buffer);
    tir::StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tir::AllocateNode *op) final {
    auto role_attr = op->annotations.Get(tl::attr::kSunmmioReduceRegisterTemp);
    if (role_attr) {
      const auto *role = role_attr.value().as<IntImmNode>();
      ICHECK(role) << tl::attr::kSunmmioReduceRegisterTemp
                   << " Allocate annotation expects an integer role";
      ICHECK(role->value ==
                 static_cast<int>(tl::ReduceRegisterTempRole::kAccumulator) ||
             role->value ==
                 static_cast<int>(tl::ReduceRegisterTempRole::kResult))
          << tl::attr::kSunmmioReduceRegisterTemp << " has unknown role value "
          << role->value;
      reduce_register_temp_roles[op->buffer_var.get()] =
          static_cast<int>(role->value);
    }
    tir::StmtVisitor::VisitStmt_(op);
  }

  void Record(const tir::Buffer &buffer) {
    if (!buffer.defined() || !buffer->data.defined()) {
      return;
    }
    const tir::VarNode *data = buffer->data.get();
    auto it = buffer_data_to_buffer.find(data);
    if (it == buffer_data_to_buffer.end()) {
      buffer_data_to_buffer.emplace(data, buffer);
      // return;
    } else {
      LOG(WARNING) << "Found duplicate DeclBuffer for data var " << data;
    }
  }
};

std::string GetAllocateStorageScope(const tir::Var &buffer_var) {
  if (const auto *ptr = buffer_var->type_annotation.as<PointerTypeNode>()) {
    const std::string &scope = ptr->storage_scope;
    if (scope == "shared.asram" || scope == "shared.wsram" ||
        scope == "shared.rsram") {
      return scope;
    } else {
      LOG(FATAL) << "get Allocate StorageScope error:  " << scope;
    }
  }
  LOG(FATAL) << "SunMMIO SUVM allocate expects PointerType buffer_var";
  TVM_FFI_UNREACHABLE();
}

bool IsSunmmioLocalVarBuffer(const tir::Buffer &buffer) {
  if (!buffer.defined()) {
    return false;
  }
  const std::string scope = buffer.scope();
  return scope == "local.var";
}

bool IsSunmmioRsramScope(const std::string &scope) {
  return scope == tl::kSunmmioScopeRSRAM || scope == "rsram";
}

bool IsSunmmioAsramScope(const std::string &scope) {
  return scope == tl::kSunmmioScopeASRAM || scope == "asram";
}

bool IsSunmmioWsramScope(const std::string &scope) {
  return scope == tl::kSunmmioScopeWSRAM || scope == "wsram";
}

SunMMIOType MakeScalarType(DataType dtype) {
  dtype = CanonicalizeSuvmDType(dtype).with_lanes(1);
  return SunMMIOType{SunMMIOType::Kind::kScalar, dtype, 1, {}};
}

SunMMIOType MakeTileTypeForShape(DataType dtype,
                                 const std::vector<int64_t> &shape,
                                 SunMMIOType::Kind kind) {
  dtype = CanonicalizeSuvmDType(dtype).with_lanes(1);
  SunMMIOType type;
  type.kind = kind;
  type.dtype = dtype;
  type.lanes = 1;
  for (int64_t extent : shape) {
    type.shape.push_back(IntImm(DataType::Int(32), extent));
  }
  return type;
}

std::string PrimExprArrayToString(const ffi::Array<PrimExpr> &exprs) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < exprs.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << exprs[i];
  }
  os << "]";
  return os.str();
}

std::string IntVectorToString(const std::vector<int64_t> &values) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << values[i];
  }
  os << "]";
  return os.str();
}

int64_t StaticIntOrNegative(const PrimExpr &expr, arith::Analyzer *analyzer) {
  PrimExpr simplified = analyzer->Simplify(expr);
  const auto *imm = simplified.as<IntImmNode>();
  if (!imm || imm->value <= 0) {
    return -1;
  }
  return imm->value;
}

struct ScalarTileAccessPlan {
  std::vector<int64_t> tile_shape;
  std::vector<int64_t> tiled_dims;
  std::vector<PrimExpr> partition_indices;
  std::vector<PrimExpr> local_indices;
};

std::string DescribeScalarTileAccess(const tir::Buffer &buffer,
                                     const ffi::Array<PrimExpr> &indices,
                                     DataType dtype,
                                     const ffi::Optional<tl::Layout> &layout) {
  std::ostringstream os;
  os << "buffer=" << buffer->name << ", scope=" << buffer.scope()
     << ", dtype=" << dtype << ", buffer_rank=" << buffer->shape.size()
     << ", indices=" << PrimExprArrayToString(indices)
     << ", logical_shape=" << PrimExprArrayToString(buffer->shape);
  if (layout.defined()) {
    os << ", layout=" << layout.value()->DebugOutput();
  } else {
    os << ", layout=<plain row-major fallback>";
  }
  return os.str();
}

std::vector<int64_t> GetScalarAccessLayoutExtents(
    const tir::Buffer &buffer, const ffi::Optional<tl::Layout> &layout,
    arith::Analyzer *analyzer, std::string *failure_reason) {
  ffi::Array<PrimExpr> extents;
  if (layout.defined()) {
    const auto *cute = layout.value().as<tl::CuteLayoutNode>();
    if (!cute) {
      if (failure_reason != nullptr) {
        *failure_reason = "Sunmmio scalar tile access planning expects a "
                          "CuteLayout when a layout annotation is present.";
      }
      return {};
    }
    extents = cute->GetCoveredShape();
  } else {
    extents = buffer->shape;
  }

  std::vector<int64_t> static_extents;
  static_extents.reserve(extents.size());
  for (const PrimExpr &extent : extents) {
    static_extents.push_back(StaticIntOrNegative(extent, analyzer));
  }
  return static_extents;
}

int64_t TileElementCount(const std::vector<int64_t> &shape) {
  int64_t elems = 1;
  for (int64_t extent : shape) {
    elems *= extent;
  }
  return elems;
}

bool TileTransferSizeIsLegal(const tir::Buffer &buffer, DataType dtype,
                             const std::vector<int64_t> &tile_shape,
                             const tl::SunmmioTileProcessorConfig &config) {
  if (!IsSunmmioRsramScope(buffer.scope())) {
    return true;
  }
  if (config.rsram_align_bytes <= 0) {
    return true;
  }
  int64_t tile_bits =
      TileElementCount(tile_shape) * static_cast<int64_t>(dtype.bits());
  if (tile_bits % 8 != 0) {
    return false;
  }
  int64_t tile_bytes = tile_bits / 8;
  return tile_bytes >= config.rsram_align_bytes &&
         tile_bytes % config.rsram_align_bytes == 0;
}

bool CandidateExtentsAreLegal(const tl::TrailingTilePattern &pattern,
                              const std::vector<int64_t> &layout_extents,
                              std::string *failure_reason) {
  for (size_t i = 0; i < pattern.mapped_dims.size(); ++i) {
    int dim = pattern.mapped_dims[i];
    int64_t extent = layout_extents[dim];
    int64_t tile_extent = pattern.tile_shape[i];
    if (extent < 0) {
      continue;
    }
    if (tile_extent > extent) {
      if (failure_reason != nullptr) {
        std::ostringstream os;
        os << "tile extent " << tile_extent << " exceeds layout extent "
           << extent << " at buffer dim " << dim << ".";
        *failure_reason = os.str();
      }
      return false;
    }
    if (extent % tile_extent != 0) {
      if (failure_reason != nullptr) {
        std::ostringstream os;
        os << "layout extent " << extent << " is not divisible by tile extent "
           << tile_extent << " at buffer dim " << dim << ".";
        *failure_reason = os.str();
      }
      return false;
    }
  }
  return true;
}

bool HasTrailingRowMajorContiguousOrder(
    const tir::Buffer &buffer,
    const ffi::Map<tir::Buffer, tl::Layout> &layout_map,
    std::string *failure_reason) {
  std::vector<tl::ContiguousStep> steps =
      tl::GetBufferContiguousSteps(buffer, layout_map);
  if (steps.empty()) {
    if (failure_reason != nullptr) {
      *failure_reason = "cannot recover a contiguous trailing layout step.";
    }
    return false;
  }
  int width_dim = static_cast<int>(buffer->shape.size()) - 1;
  if (steps.front().dim != width_dim) {
    if (failure_reason != nullptr) {
      std::ostringstream os;
      os << "the innermost contiguous layout step is dim " << steps.front().dim
         << ", not the trailing width dim " << width_dim
         << "; suvm.tile.load/store require row-major tile views.";
      *failure_reason = os.str();
    }
    return false;
  }
  return true;
}

std::optional<tl::TrailingTilePattern>
SelectScalarTilePattern(const tir::Buffer &buffer, int tile_rank,
                        DataType dtype,
                        const ffi::Map<tir::Buffer, tl::Layout> &layout_map,
                        const std::vector<int64_t> &layout_extents,
                        const tl::SunmmioTileProcessorConfig &config,
                        arith::Analyzer *analyzer, std::string *last_reason) {
  std::vector<tl::TrailingTilePattern> legal_patterns;
  for (const tl::TrailingTilePattern &pattern :
       tl::EnumerateInferredTrailingTilePatterns(buffer, tile_rank, layout_map,
                                                 config, analyzer,
                                                 tl::AlignmentMode::kStrict)) {
    if (static_cast<int>(pattern.tile_shape.size()) != tile_rank) {
      continue;
    }
    std::vector<int64_t> tile_shape(pattern.tile_shape.begin(),
                                    pattern.tile_shape.end());
    if (!TileTransferSizeIsLegal(buffer, dtype, tile_shape, config)) {
      if (last_reason != nullptr) {
        std::ostringstream os;
        os << "tile shape " << IntVectorToString(tile_shape)
           << " does not satisfy the " << config.rsram_align_bytes
           << "-byte RSRAM vector-transfer size/alignment constraint.";
        *last_reason = os.str();
      }
      continue;
    }
    if (!CandidateExtentsAreLegal(pattern, layout_extents, last_reason)) {
      continue;
    }
    legal_patterns.push_back(pattern);
  }

  if (legal_patterns.empty()) {
    return std::nullopt;
  }

  std::sort(legal_patterns.begin(), legal_patterns.end(),
            [](const tl::TrailingTilePattern &lhs,
               const tl::TrailingTilePattern &rhs) {
              int lhs_elems = tl::TileElements(lhs.tile_shape);
              int rhs_elems = tl::TileElements(rhs.tile_shape);
              if (lhs_elems != rhs_elems) {
                return lhs_elems > rhs_elems;
              }
              return lhs.tile_shape > rhs.tile_shape;
            });
  return legal_patterns.front();
}

std::optional<ScalarTileAccessPlan> PlanScalarTileAccess(
    const tir::Buffer &buffer, const ffi::Array<PrimExpr> &indices,
    DataType dtype, const ffi::Optional<tl::Layout> &layout,
    const tl::SunmmioTileProcessorConfig &config, std::string *failure_reason) {
  int buffer_rank = static_cast<int>(buffer->shape.size());
  if (buffer_rank < 1) {
    if (failure_reason != nullptr) {
      *failure_reason = "scalar tile access requires a rank >= 1 buffer.";
    }
    return std::nullopt;
  }
  if (static_cast<int>(indices.size()) != buffer_rank) {
    if (failure_reason != nullptr) {
      std::ostringstream os;
      os << "indices rank " << indices.size()
         << " does not match source buffer rank " << buffer_rank << ".";
      *failure_reason = os.str();
    }
    return std::nullopt;
  }

  arith::Analyzer analyzer;
  ffi::Map<tir::Buffer, tl::Layout> layout_map;
  if (layout.defined()) {
    layout_map.Set(buffer, layout.value());
  }

  std::string local_reason;
  std::vector<int64_t> layout_extents =
      GetScalarAccessLayoutExtents(buffer, layout, &analyzer, &local_reason);
  if (layout_extents.empty()) {
    if (failure_reason != nullptr) {
      *failure_reason = local_reason;
    }
    return std::nullopt;
  }
  if (static_cast<int>(layout_extents.size()) != buffer_rank) {
    if (failure_reason != nullptr) {
      std::ostringstream os;
      os << "layout covered rank " << layout_extents.size()
         << " does not match buffer rank " << buffer_rank << ".";
      *failure_reason = os.str();
    }
    return std::nullopt;
  }

  if (!HasTrailingRowMajorContiguousOrder(buffer, layout_map, &local_reason)) {
    if (failure_reason != nullptr) {
      *failure_reason = local_reason;
    }
    return std::nullopt;
  }

  std::vector<int> ranks_to_try;
  if (buffer_rank >= 2) {
    ranks_to_try.push_back(2);
  }
  ranks_to_try.push_back(1);

  std::optional<tl::TrailingTilePattern> selected;
  for (int tile_rank : ranks_to_try) {
    selected = SelectScalarTilePattern(buffer, tile_rank, dtype, layout_map,
                                       layout_extents, config, &analyzer,
                                       &local_reason);
    if (selected.has_value()) {
      break;
    }
  }

  if (!selected.has_value()) {
    if (failure_reason != nullptr) {
      std::ostringstream os;
      os << "cannot infer a legal trailing rank-1/rank-2 tile_view";
      if (!local_reason.empty()) {
        os << ": " << local_reason;
      }
      os << ". Use an aligned-row-major staging layout when logical extents "
            "are "
            "not compatible with 64B RSRAM vector transfers.";
      *failure_reason = os.str();
    }
    return std::nullopt;
  }

  ScalarTileAccessPlan plan;
  plan.tile_shape.assign(selected->tile_shape.begin(),
                         selected->tile_shape.end());
  plan.tiled_dims.assign(selected->mapped_dims.begin(),
                         selected->mapped_dims.end());
  plan.partition_indices.reserve(buffer_rank);
  std::unordered_map<int, int64_t> tiled_dim_to_extent;
  for (size_t i = 0; i < selected->mapped_dims.size(); ++i) {
    tiled_dim_to_extent[selected->mapped_dims[i]] = selected->tile_shape[i];
  }

  for (int dim = 0; dim < buffer_rank; ++dim) {
    auto it = tiled_dim_to_extent.find(dim);
    if (it == tiled_dim_to_extent.end()) {
      plan.partition_indices.push_back(indices[dim]);
      continue;
    }
    PrimExpr tile_extent = IntImm(indices[dim].dtype(), it->second);
    plan.partition_indices.push_back(
        analyzer.Simplify(floordiv(indices[dim], tile_extent)));
    plan.local_indices.push_back(
        analyzer.Simplify(floormod(indices[dim], tile_extent)));
  }

  return plan;
}

bool SameTypeShape(const SunMMIOType &lhs, const SunMMIOType &rhs) {
  if (lhs.shape.size() != rhs.shape.size()) {
    return false;
  }
  StructuralEqual equal;
  for (size_t i = 0; i < lhs.shape.size(); ++i) {
    if (!equal(lhs.shape[i], rhs.shape[i])) {
      return false;
    }
  }
  return true;
}

std::string LocalVarValueName(const tir::VarNode *var) {
  ICHECK(var != nullptr);
  return "%local_var_" + var->name_hint;
}

DataType ExpectedMXDataDType(DataType mx_dtype) {
  ICHECK(tl::sunmmio::IsMXDType(mx_dtype))
      << "MX pack/unpack expects mxfp8 or mxfp4, got " << mx_dtype;
  if (mx_dtype.bits() == 8) {
    return DataType::Float8E4M3FN();
  }
  if (mx_dtype.bits() == 4) {
    return DataType::Float4E2M1FN();
  }
  LOG(FATAL) << "Unsupported MX dtype " << mx_dtype;
  TVM_FFI_UNREACHABLE();
}

DataType ExpectedMXScaleDType() { return DataType::Float8E8M0FNU(); }

SunMMIOType MakeTileType(DataType dtype, const std::vector<int64_t> &shape) {
  SunMMIOType type;
  type.kind = SunMMIOType::Kind::kTile;
  type.dtype = CanonicalizeSuvmDType(dtype).with_lanes(1);
  type.lanes = 1;
  for (int64_t dim : shape) {
    type.shape.push_back(IntImm(DataType::Int(32), dim));
  }
  return type;
}

SunMMIOType MakeTileViewType(DataType dtype,
                             const std::vector<int64_t> &shape) {
  SunMMIOType type = MakeTileType(dtype, shape);
  type.kind = SunMMIOType::Kind::kTileView;
  return type;
}

std::vector<int64_t> ExtractStaticPrimExprs(const std::vector<PrimExpr> &exprs,
                                            const char *what) {
  std::vector<int64_t> values;
  values.reserve(exprs.size());
  for (const PrimExpr &expr : exprs) {
    const auto *imm = expr.as<IntImmNode>();
    ICHECK(imm) << what << " must be static, got " << expr;
    values.push_back(static_cast<int64_t>(imm->value));
  }
  return values;
}

std::vector<int64_t> ExtractStaticShape(const SunMMIOType &type) {
  return ExtractStaticPrimExprs(type.shape, "SunMMIO type shape");
}

std::vector<int64_t> ExtractPhysicalExtents(const SunMMIOType &type) {
  std::vector<int64_t> shape = ExtractStaticShape(type);
  if (type.layout_hshape.empty()) {
    return shape;
  }
  std::vector<int64_t> hshape =
      ExtractStaticPrimExprs(type.layout_hshape, "SunMMIO layout shape");
  if (type.layout_dim_levels.empty()) {
    return hshape;
  }
  ICHECK_EQ(type.layout_dim_levels.size(), shape.size())
      << "SunMMIO layout dim levels rank mismatch";
  std::vector<int64_t> physical;
  physical.reserve(type.layout_dim_levels.size());
  size_t offset = 0;
  for (uint8_t levels : type.layout_dim_levels) {
    ICHECK_GT(levels, 0);
    ICHECK_LE(offset + levels, hshape.size());
    int64_t prod = 1;
    for (uint8_t i = 0; i < levels; ++i) {
      prod *= hshape[offset + i];
    }
    physical.push_back(prod);
    offset += levels;
  }
  ICHECK_EQ(offset, hshape.size());
  return physical;
}

} // namespace

CodeGenTileLangSunMMIO::CodeGenTileLangSunMMIO() = default;

void CodeGenTileLangSunMMIO::SetTarget(tvm::Target target) { target_ = target; }

void CodeGenTileLangSunMMIO::Init() {
  Clear();
  builder_ = std::make_unique<SuvmSunmmioBuilder>();
  builder_->Init();
  builder_->BeginModule();
  initialized_ = true;
}

void CodeGenTileLangSunMMIO::Clear() {
  if (builder_) {
    builder_->Clear();
  }
  builder_.reset();
  ssa_counter_ = 0;
  var_table_.clear();
  local_var_table_.clear();
  buffer_registry_.clear();
  buffer_data_to_buffer_.clear();
  reduce_register_temp_roles_.clear();
  attr_stack_.clear();
  scoped_vars_.clear();
  scoped_local_vars_.clear();
  scoped_buffers_.clear();
  var_scope_markers_.clear();
  local_var_scope_markers_.clear();
  buffer_scope_markers_.clear();
  main_coverage_ = CoverageData{};
  tiles_coverage_ = CoverageData{};
  coverage_domain_ = CoverageDomain::kMain;
  initialized_ = false;
}

SunMMIOValue CodeGenTileLangSunMMIO::EvalExpr(const tvm::PrimExpr &expr) {
  MarkVisitedExprRoot(expr);
  return tir::ExprFunctor<SunMMIOValue(const tvm::PrimExpr &)>::VisitExpr(expr);
}

void CodeGenTileLangSunMMIO::VisitStmtTracked(const tir::Stmt &stmt) {
  // ForNode selects its coverage domain in VisitStmt_(ForNode*) before it is
  // marked.  All other statements inherit the domain of their lowering path.
  if (stmt.defined() && !stmt.as<tir::ForNode>()) {
    MarkVisitedNodeType(stmt->GetTypeKey());
  }
  tir::StmtVisitor::VisitStmt(stmt);
}

void CodeGenTileLangSunMMIO::MarkVisitedNodeType(const std::string &type_key) {
  CoverageData &coverage = coverage_domain_ == CoverageDomain::kTiles
                               ? tiles_coverage_
                               : main_coverage_;
  coverage.visited_node_types.insert(type_key);
}

void CodeGenTileLangSunMMIO::MarkVisitedCallOpFromExpr(
    const tvm::PrimExpr &expr) {
  const auto *call = expr.as<tir::CallNode>();
  if (!call) {
    return;
  }
  CoverageData &coverage = coverage_domain_ == CoverageDomain::kTiles
                               ? tiles_coverage_
                               : main_coverage_;
  if (const auto *op_node = call->op.as<OpNode>()) {
    coverage.visited_call_ops.insert(op_node->name);
  } else if (const auto *gv = call->op.as<GlobalVarNode>()) {
    coverage.visited_call_ops.insert(std::string("global::") + gv->name_hint);
  } else {
    coverage.visited_call_ops.insert("unknown_call_target");
  }
}

void CodeGenTileLangSunMMIO::MarkVisitedExprRoot(const tvm::PrimExpr &expr) {
  if (!expr.defined()) {
    return;
  }
  MarkVisitedNodeType(expr->GetTypeKey());
  MarkVisitedCallOpFromExpr(expr);
}

void CodeGenTileLangSunMMIO::MarkVisitedExprTree(const tvm::PrimExpr &expr) {
  if (!expr.defined()) {
    return;
  }
  tir::PreOrderVisit(expr, [&](const ObjectRef &obj) {
    if (obj.as<PrimExprNode>()) {
      MarkVisitedExprRoot(Downcast<PrimExpr>(obj));
    }
    return true;
  });
}

tir::BufferRegion
CodeGenTileLangSunMMIO::NormalizeRegionTracked(const tvm::PrimExpr &expr) {
  return tl::NormalizeToBufferRegion(expr,
                                     [this](const PrimExpr &consumed_root) {
                                       MarkVisitedExprRoot(consumed_root);
                                     });
}

bool CodeGenTileLangSunMMIO::TryConsumeSunmmioOdmaUnit(
    const tvm::PrimExpr &expr, SunMMIOCallAttrs *attrs) {
  std::optional<tl::SunmmioOdmaUnit> unit = tl::ParseSunmmioOdmaUnitExpr(expr);
  if (!unit) {
    return false;
  }
  MarkVisitedExprRoot(expr);
  const auto *call = expr.as<tir::CallNode>();
  ICHECK(call);
  MarkVisitedExprRoot(call->args[0]);
  (*attrs)[SunMMIOCallAttrKey::kUnit] =
      std::string(tl::StringifySunmmioOdmaUnit(*unit));
  return true;
}

void CodeGenTileLangSunMMIO::CollectExpectedCoverage(const tir::PrimFunc &f) {
  auto record = [](const ObjectRef &obj, CoverageData *coverage) {
    if (!obj.defined()) {
      return;
    }
    coverage->expected_node_types.insert(obj->GetTypeKey());
    if (const auto *call = obj.as<tir::CallNode>()) {
      if (const auto *op_node = call->op.as<OpNode>()) {
        coverage->expected_call_ops.insert(op_node->name);
      } else if (const auto *gv = call->op.as<GlobalVarNode>()) {
        coverage->expected_call_ops.insert(std::string("global::") +
                                           gv->name_hint);
      } else {
        coverage->expected_call_ops.insert("unknown_call_target");
      }
    }
  };

  std::vector<tir::For> tiles_roots;
  tir::PreOrderVisit(f->body, [&](const ObjectRef &obj) {
    if (const auto *loop = obj.as<tir::ForNode>();
        loop && loop->annotations.count(tl::attr::kTileDomain)) {
      tiles_roots.push_back(ffi::GetRef<tir::For>(loop));
      return false;
    }
    record(obj, &main_coverage_);
    return true;
  });

  for (const tir::For &root : tiles_roots) {
    tir::PreOrderVisit(root, [&](const ObjectRef &obj) {
      record(obj, &tiles_coverage_);
      return true;
    });
  }
}

void CodeGenTileLangSunMMIO::CollectDeclBuffers(const tir::Stmt &stmt) {
  DeclBufferCollector collector;
  collector(stmt);
  buffer_data_to_buffer_ = std::move(collector.buffer_data_to_buffer);
  reduce_register_temp_roles_ = std::move(collector.reduce_register_temp_roles);
}

bool CodeGenTileLangSunMMIO::IsSunmmioReduceRegisterTempBuffer(
    const tir::Buffer &buffer) const {
  if (!buffer.defined() || !IsSunmmioRsramScope(buffer.scope())) {
    return false;
  }
  return reduce_register_temp_roles_.count(buffer->data.get()) != 0;
}

bool CodeGenTileLangSunMMIO::IsSunmmioReduceLoopCarriedTempBuffer(
    const tir::Buffer &buffer) const {
  if (!IsSunmmioReduceRegisterTempBuffer(buffer)) {
    return false;
  }
  auto it = buffer.defined()
                ? reduce_register_temp_roles_.find(buffer->data.get())
                : reduce_register_temp_roles_.end();
  return it != reduce_register_temp_roles_.end() &&
         it->second ==
             static_cast<int>(tl::ReduceRegisterTempRole::kAccumulator);
}

bool CodeGenTileLangSunMMIO::IsSunmmioReduceLocalTempBuffer(
    const tir::Buffer &buffer) const {
  if (!IsSunmmioReduceRegisterTempBuffer(buffer)) {
    return false;
  }
  auto it = buffer.defined()
                ? reduce_register_temp_roles_.find(buffer->data.get())
                : reduce_register_temp_roles_.end();
  return it != reduce_register_temp_roles_.end() &&
         it->second == static_cast<int>(tl::ReduceRegisterTempRole::kResult);
}

void CodeGenTileLangSunMMIO::WriteCoverageReport() const {
  const char *path = std::getenv("TL_SUNMMIO_CODEGEN_COVERAGE_PATH");
  if (path == nullptr || std::string(path).empty()) {
    return;
  }
  std::ofstream os(path, std::ios::out | std::ios::trunc);
  if (!os.is_open()) {
    LOG(WARNING) << "CodeGenTileLangSunMMIO: failed to open coverage path: "
                 << path;
    return;
  }
  auto write_list = [&os](int indent, const char *key,
                          const std::set<std::string> &values) {
    os << std::string(indent, ' ') << "\"" << key << "\": [";
    bool first = true;
    for (const auto &item : values) {
      if (!first) {
        os << ", ";
      }
      first = false;
      os << "\"" << item << "\"";
    }
    os << "]";
  };
  auto diff = [](const std::set<std::string> &a,
                 const std::set<std::string> &b) {
    std::set<std::string> out;
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                        std::inserter(out, out.begin()));
    return out;
  };
  auto write_domain = [&](const char *name, const CoverageData &coverage) {
    std::set<std::string> missing_nodes =
        diff(coverage.expected_node_types, coverage.visited_node_types);
    std::set<std::string> missing_calls =
        diff(coverage.expected_call_ops, coverage.visited_call_ops);

    os << "  \"" << name << "\": {\n";
    write_list(4, "expected_node_types", coverage.expected_node_types);
    os << ",\n";
    write_list(4, "visited_node_types", coverage.visited_node_types);
    os << ",\n";
    write_list(4, "missing_node_types", missing_nodes);
    os << ",\n";
    write_list(4, "expected_call_ops", coverage.expected_call_ops);
    os << ",\n";
    write_list(4, "visited_call_ops", coverage.visited_call_ops);
    os << ",\n";
    write_list(4, "missing_call_ops", missing_calls);
    os << "\n  }";
  };

  os << "{\n";
  write_domain("main", main_coverage_);
  os << ",\n";
  write_domain("tiles", tiles_coverage_);
  os << "\n}\n";
}

void CodeGenTileLangSunMMIO::CheckCoverageOrFail() const {
  auto diff = [](const std::set<std::string> &a,
                 const std::set<std::string> &b) {
    std::set<std::string> out;
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                        std::inserter(out, out.begin()));
    return out;
  };
  struct MissingCoverage {
    std::set<std::string> node_types;
    std::set<std::string> call_ops;
  };
  auto get_missing = [&](const CoverageData &coverage) {
    return MissingCoverage{
        diff(coverage.expected_node_types, coverage.visited_node_types),
        diff(coverage.expected_call_ops, coverage.visited_call_ops)};
  };
  MissingCoverage main_missing = get_missing(main_coverage_);
  MissingCoverage tiles_missing = get_missing(tiles_coverage_);

  const char *strict_env = std::getenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT");
  bool strict = strict_env != nullptr && std::string(strict_env) == "1";

  auto has_missing = [](const MissingCoverage &missing) {
    return !missing.node_types.empty() || !missing.call_ops.empty();
  };
  auto warn_missing = [&](const char *domain, const MissingCoverage &missing) {
    if (has_missing(missing)) {
      LOG(WARNING) << "CodeGenTileLangSunMMIO coverage gaps: domain=" << domain
                   << ", missing_nodes=" << missing.node_types.size()
                   << ", missing_call_ops=" << missing.call_ops.size();
    }
  };
  warn_missing("main", main_missing);
  warn_missing("tiles", tiles_missing);

  if (strict && (has_missing(main_missing) || has_missing(tiles_missing))) {
    std::ostringstream err;
    err << "SunMMIO codegen traversal incomplete.";
    auto append_missing = [&](const char *domain,
                              const MissingCoverage &missing) {
      if (!has_missing(missing)) {
        return;
      }
      err << " Domain " << domain << " missing node types: ";
      for (const auto &s : missing.node_types) {
        err << s << "; ";
      }
      err << "missing call ops: ";
      for (const auto &s : missing.call_ops) {
        err << s << "; ";
      }
    };
    append_missing("main", main_missing);
    append_missing("tiles", tiles_missing);
    LOG(FATAL) << err.str();
  }
}

void CodeGenTileLangSunMMIO::AddFunction(const GlobalVar &gvar,
                                         const tir::PrimFunc &f) {
  if (!initialized_) {
    Init();
  }
  ICHECK(builder_) << "CodeGenTileLangSunMMIO builder is not initialized";
  CollectExpectedCoverage(f);
  CollectDeclBuffers(f->body);

  SunMMIOBuilder::TirLayoutMap layout_map;
  SunMMIOBuilder::TirLayoutMap global_layout_map;
  if (auto opt =
          f->GetAttr<SunMMIOBuilder::TirLayoutMap>(tl::attr::kLayoutMap)) {
    layout_map = opt.value();
  }
  if (auto opt = f->GetAttr<SunMMIOBuilder::TirLayoutMap>(
          tl::attr::kGlobalLayoutMap)) {
    global_layout_map = opt.value();
  }
  builder_->PushLayoutScope(layout_map, global_layout_map);

  EnterScope();
  std::vector<BuilderArg> args;
  std::vector<PendingExternalBuffer> pending_external_buffers;
  int arg_index = 0;
  for (const tir::Var &p : f->params) {
    std::string arg_name = "%arg" + std::to_string(arg_index++);
    auto buffer_it = buffer_data_to_buffer_.find(p.get());
    if (buffer_it != buffer_data_to_buffer_.end()) {
      const tir::Buffer &buffer = buffer_it->second;
      SunMMIOType buf_ty = MapBufferType(buffer);
      args.push_back({arg_name, buf_ty});
      BindVar(p, SunMMIOValue{p.dtype(), arg_name, buf_ty});
      RegisterBuffer(buffer, true, arg_name);
      pending_external_buffers.push_back({buffer, arg_name, buf_ty});
    } else {
      SunMMIOType arg_ty = MapType(p.dtype());
      args.push_back({arg_name, arg_ty});
      BindVar(p, SunMMIOValue{p.dtype(), arg_name, arg_ty});
    }
  }

  builder_->BeginFunction(gvar->name_hint, args);
  for (const PendingExternalBuffer &pending : pending_external_buffers) {
    BindExternalBufferLayout(pending);
  }
  VisitStmtTracked(f->body);
  builder_->EmitReturn();
  builder_->EndFunction();
  ExitScope();
  builder_->PopLayoutScope();
}

std::string CodeGenTileLangSunMMIO::Finish() {
  if (!initialized_) {
    Init();
  }
  ICHECK(builder_) << "CodeGenTileLangSunMMIO builder is not initialized";
  WriteCoverageReport();
  CheckCoverageOrFail();
  builder_->EndModule();
  std::string out = builder_->Finish();
  initialized_ = false;
  return out;
}

std::string CodeGenTileLangSunMMIO::NewValueName() {
  return "%v" + std::to_string(ssa_counter_++);
}

SunMMIOType CodeGenTileLangSunMMIO::MapType(tvm::DataType dtype) const {
  dtype = CanonicalizeSuvmDType(dtype);
  if (dtype.lanes() > 1) {
    return SunMMIOType{
        SunMMIOType::Kind::kVector, dtype.with_lanes(1), dtype.lanes(), {}};
  }
  if (dtype.is_handle()) {
    return SunMMIOType{SunMMIOType::Kind::kHandle, dtype, 1, {}};
  }
  if (dtype.is_void()) {
    return SunMMIOType{SunMMIOType::Kind::kUnknown, dtype, 1, {}};
  }
  return SunMMIOType{SunMMIOType::Kind::kScalar, dtype, 1, {}};
}

std::string
CodeGenTileLangSunMMIO::MapStorageScope(const std::string &scope) const {
  if (scope.empty()) {
    return "global";
  }
  std::string out = scope;
  std::replace(out.begin(), out.end(), '.', '_');
  return out;
}

SunMMIOType
CodeGenTileLangSunMMIO::MapBufferType(const tir::Buffer &buffer) const {
  std::vector<PrimExpr> shape;
  shape.reserve(buffer->shape.size());
  for (const PrimExpr &dim : buffer->shape) {
    shape.push_back(dim);
  }
  SunMMIOType type;
  type.kind = SunMMIOType::Kind::kMemTensor;
  type.dtype = buffer->dtype.with_lanes(1);
  type.shape = std::move(shape);
  type.memory_scope = buffer.scope();
  type.byte_offset = 0;
  if (builder_) {
    builder_->ApplyLayoutToType(buffer, &type);
  }
  return type;
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::SeqStmtNode *op) {
  for (const Stmt &stmt : op->seq) {
    VisitStmtTracked(stmt);
  }
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitConstIndex(int64_t v) {
  return builder_->ConstantInt(
      NewValueName(), v,
      SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
      DataType::Int(32));
}

SunMMIOValue CodeGenTileLangSunMMIO::EnsureIndex(const SunMMIOValue &v) {
  if (v.type.kind == SunMMIOType::Kind::kIndex) {
    return v;
  }
  return builder_->Cast(
      NewValueName(), v,
      SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
      DataType::Int(32));
}

SunMMIOValue CodeGenTileLangSunMMIO::EnsureType(const SunMMIOValue &v,
                                                const SunMMIOType &target_type,
                                                DataType dtype) {
  if (v.type.kind == target_type.kind && v.type.dtype == target_type.dtype &&
      v.type.lanes == target_type.lanes && SameTypeShape(v.type, target_type)) {
    return v;
  }
  return builder_->Cast(NewValueName(), v, target_type, dtype);
}

ArithmeticFlavor
CodeGenTileLangSunMMIO::GetArithmeticFlavor(DataType dtype) const {
  if (dtype.is_float() || dtype.is_bfloat16()) {
    return ArithmeticFlavor::kFloat;
  }
  if (dtype.is_uint()) {
    return ArithmeticFlavor::kUnsignedInt;
  }
  if (dtype.is_bool()) {
    return ArithmeticFlavor::kBool;
  }
  return ArithmeticFlavor::kSignedInt;
}

CompareDomain CodeGenTileLangSunMMIO::GetCompareDomain(DataType dtype) const {
  if (dtype.is_float() || dtype.is_bfloat16()) {
    return CompareDomain::kFloat;
  }
  if (dtype.is_uint()) {
    return CompareDomain::kUnsignedInt;
  }
  if (dtype.is_bool()) {
    return CompareDomain::kBool;
  }
  return CompareDomain::kSignedInt;
}

SunMMIOValue CodeGenTileLangSunMMIO::BindVar(const tir::Var &var,
                                             const SunMMIOValue &value) {
  var_table_[var.get()] = value;
  scoped_vars_.push_back(var.get());
  return value;
}

const SunMMIOValue &
CodeGenTileLangSunMMIO::LookupVar(const tir::VarNode *var) const {
  ICHECK(var != nullptr);
  auto it = var_table_.find(var);
  if (it != var_table_.end()) {
    return it->second;
  }
  LOG(FATAL) << "CodeGenTileLangSunMMIO: unbound TIR var `" << var->name_hint
             << "` reached SunMMIO codegen without a parameter, loop, let, "
                "allocation, or thread binding";
  TVM_FFI_UNREACHABLE();
}

const SunMMIOValue &
CodeGenTileLangSunMMIO::LookupLocalVar(const tir::VarNode *var) const {
  ICHECK(var != nullptr);
  auto it = local_var_table_.find(var);
  ICHECK(it != local_var_table_.end())
      << "CodeGenTileLangSunMMIO: unknown local.var buffer data "
      << var->name_hint;
  return it->second;
}

SunMMIOValue CodeGenTileLangSunMMIO::MaterializeDynamicLayoutExpr(
    const tvm::PrimExpr &expr) {
  arith::Analyzer analyzer;
  PrimExpr simplified = analyzer.Simplify(expr);
  tir::PostOrderVisit(simplified, [&](const ObjectRef &node) {
    if (!node.defined()) {
      return;
    }
    if (const auto *var = node.as<VarNode>()) {
      ICHECK(var_table_.count(var))
          << "SunMMIO dynamic layout expression depends on unbound runtime "
             "variable "
          << var->name_hint << " in " << simplified;
      return;
    }
    ICHECK(!node.as<BufferLoadNode>() && !node.as<ProducerLoadNode>() &&
           !node.as<CallNode>() && !node.as<RampNode>() &&
           !node.as<BroadcastNode>() && !node.as<ShuffleNode>() &&
           !node.as<LetNode>())
        << "SunMMIO dynamic layout expression must be scalar arithmetic over "
           "runtime parameters, got unsupported node "
        << node->GetTypeKey() << " in " << simplified;
  });
  return EnsureIndex(EvalExpr(simplified));
}

std::vector<SunMMIOValue> CodeGenTileLangSunMMIO::CollectDynamicLayoutValues(
    const std::vector<PrimExpr> &exprs) {
  std::vector<SunMMIOValue> values;
  values.reserve(exprs.size());
  for (const PrimExpr &expr : exprs) {
    arith::Analyzer analyzer;
    PrimExpr simplified = analyzer.Simplify(expr);
    if (simplified.as<IntImmNode>()) {
      continue;
    }
    values.push_back(MaterializeDynamicLayoutExpr(simplified));
  }
  return values;
}

void CodeGenTileLangSunMMIO::BindExternalBufferLayout(
    const PendingExternalBuffer &pending) {
  std::vector<SunMMIOValue> dynamic_shapes =
      CollectDynamicLayoutValues(pending.type.layout_hshape);
  std::vector<SunMMIOValue> dynamic_strides =
      CollectDynamicLayoutValues(pending.type.layout_hstride);
  if (dynamic_shapes.empty() && dynamic_strides.empty()) {
    return;
  }

  SunMMIOValue source{pending.buffer->dtype, pending.handle, pending.type};
  std::string bound_handle = pending.handle + "_layout";
  SunMMIOValue bound = builder_->BindLayout(bound_handle, source,
                                            dynamic_shapes, dynamic_strides);

  var_table_[pending.buffer->data.get()] = bound;
  auto it = buffer_registry_.find(pending.buffer.get());
  ICHECK(it != buffer_registry_.end())
      << "Missing registered external buffer for dynamic layout binding: "
      << pending.buffer->name;
  it->second.handle = bound.value;
  it->second.buffer_type = bound.type;
}

void CodeGenTileLangSunMMIO::RegisterBuffer(const tir::Buffer &buffer,
                                            bool is_external,
                                            const std::string &handle_hint) {
  if (!buffer.defined()) {
    return;
  }
  if (IsSunmmioLocalVarBuffer(buffer)) {
    return;
  }
  if (buffer_registry_.count(buffer.get())) {
    return;
  }
  BufferBinding binding;
  binding.buffer = buffer;
  binding.scope = buffer.scope();
  binding.buffer_type = MapBufferType(buffer);
  binding.is_external = is_external;
  if (IsSunmmioReduceRegisterTempBuffer(buffer)) {
    buffer_registry_[buffer.get()] = std::move(binding);
    scoped_buffers_.push_back(buffer.get());
    return;
  }
  if (!handle_hint.empty()) {
    binding.handle = handle_hint;
    auto storage_it = var_table_.find(buffer->data.get());
    if (storage_it != var_table_.end() &&
        storage_it->second.type.kind == SunMMIOType::Kind::kMemTensor) {
      binding.buffer_type = storage_it->second.type;
    }
  } else {
    const SunMMIOValue &storage = LookupVar(buffer->data.get());
    binding.handle = storage.value;
    if (storage.type.kind == SunMMIOType::Kind::kMemTensor) {
      binding.buffer_type = storage.type;
    }
  }
  buffer_registry_[buffer.get()] = std::move(binding);
  scoped_buffers_.push_back(buffer.get());
}

const BufferBinding &
CodeGenTileLangSunMMIO::LookupBuffer(const tir::Buffer &buffer) const {
  auto it = buffer_registry_.find(buffer.get());
  if (it == buffer_registry_.end()) {
    auto data_it = buffer_data_to_buffer_.find(buffer->data.get());
    ICHECK(data_it != buffer_data_to_buffer_.end())
        << "SunMMIO LookupBuffer: missing buffer registration for name="
        << buffer->name << ", buffer_ptr=" << buffer.get()
        << ", data_name=" << buffer->data->name_hint
        << ", data_ptr=" << buffer->data.get();
    auto fallback_it = buffer_registry_.find(data_it->second.get());
    ICHECK(fallback_it != buffer_registry_.end())
        << "SunMMIO LookupBuffer: fallback buffer for name=" << buffer->name
        << " is not registered; fallback_name=" << data_it->second->name
        << ", fallback_ptr=" << data_it->second.get();
    return fallback_it->second;
  }
  return it->second;
}

void CodeGenTileLangSunMMIO::EmitAlloc(
    const tir::Buffer &buffer, const std::string &scope_hint,
    const ffi::Map<ffi::String, ffi::Any> &annotations) {
  std::vector<SunMMIOValue> dyn_extents;
  for (const PrimExpr &dim : buffer->shape) {
    MarkVisitedExprRoot(dim);
    if (!dim.as<IntImmNode>()) {
      dyn_extents.push_back(EnsureIndex(EvalExpr(dim)));
    }
  }

  SunMMIOType memtensor_type = MapBufferType(buffer);
  if (memtensor_type.memory_scope.empty()) {
    memtensor_type.memory_scope = scope_hint;
  }

  std::optional<std::string> ping_pong;
  auto ping_pong_it = annotations.find(tl::attr::kSunmmioAllocPingPong);
  if (ping_pong_it != annotations.end()) {
    ping_pong =
        static_cast<std::string>(Downcast<ffi::String>((*ping_pong_it).second));
  }

  SunMMIOValue alloc =
      builder_->Alloc(NewValueName(), memtensor_type, dyn_extents, scope_hint,
                      buffer->dtype, std::move(ping_pong));
  BindVar(buffer->data, alloc);

  auto it = buffer_registry_.find(buffer.get());
  if (it != buffer_registry_.end()) {
    it->second.handle = alloc.value;
    it->second.buffer_type = alloc.type;
  }
}

void CodeGenTileLangSunMMIO::EmitLocalVarAlloc(const tir::AllocateNode *op,
                                               const tir::Buffer &buffer) {
  ICHECK(IsSunmmioLocalVarBuffer(buffer));
  ICHECK_EQ(op->ConstantAllocationSize(), 1)
      << "local.var allocation must be scalar-sized";

  PrimExpr init = tir::make_const(op->dtype, 0);
  auto init_it = op->annotations.find(tl::attr::kLocalVarInit);
  if (init_it != op->annotations.end()) {
    PrimExpr user_init = Downcast<PrimExpr>((*init_it).second);
    if (!user_init.dtype().is_void() && user_init.dtype() != op->dtype) {
      user_init = tir::Cast(op->dtype, user_init);
    }
    init = user_init;
  }

  DataType dtype = CanonicalizeSuvmDType(buffer->dtype);
  SunMMIOValue init_value = EnsureType(EvalExpr(init), MapType(dtype), dtype);
  SunMMIOValue state = builder_->BindValueAlias(
      LocalVarValueName(buffer->data.get()), init_value);
  local_var_table_[buffer->data.get()] = state;
  scoped_local_vars_.push_back(buffer->data.get());
}

SunMMIOValue
CodeGenTileLangSunMMIO::EmitLocalVarLoad(const tir::Buffer &buffer,
                                         const ffi::Array<PrimExpr> &indices) {
  ICHECK(IsSunmmioLocalVarBuffer(buffer));
  ICHECK_EQ(indices.size(), 1)
      << "local.var buffer loads must use exactly one scalar index";
  arith::Analyzer analyzer;
  PrimExpr index = analyzer.Simplify(indices[0]);
  const auto *index_imm = index.as<IntImmNode>();
  ICHECK(index_imm && index_imm->value == 0)
      << "local.var buffer loads only support index 0";
  MarkVisitedExprTree(indices[0]);
  return LookupLocalVar(buffer->data.get());
}

void CodeGenTileLangSunMMIO::EmitLocalVarStore(
    const tir::Buffer &buffer, const ffi::Array<PrimExpr> &indices,
    const SunMMIOValue &value) {
  ICHECK(IsSunmmioLocalVarBuffer(buffer));
  ICHECK_EQ(indices.size(), 1)
      << "local.var buffer stores must use exactly one scalar index";
  arith::Analyzer analyzer;
  PrimExpr index = analyzer.Simplify(indices[0]);
  const auto *index_imm = index.as<IntImmNode>();
  ICHECK(index_imm && index_imm->value == 0)
      << "local.var buffer stores only support index 0";
  MarkVisitedExprTree(indices[0]);

  const SunMMIOValue &current = LookupLocalVar(buffer->data.get());
  DataType dtype = CanonicalizeSuvmDType(buffer->dtype);
  SunMMIOValue casted = EnsureType(value, MapType(dtype), dtype);
  SunMMIOValue state = builder_->BindValueAlias(current.value, casted);
  local_var_table_[buffer->data.get()] = state;
}

std::vector<SunMMIOValue> CodeGenTileLangSunMMIO::CollectLocalVarLiveOutValues(
    const tir::Stmt &stmt) const {
  class Collector final : public tir::StmtVisitor {
  public:
    explicit Collector(const std::unordered_map<const tir::VarNode *,
                                                SunMMIOValue> &local_var_table)
        : local_var_table_(local_var_table) {}

    std::vector<SunMMIOValue> values;

  private:
    void VisitStmt_(const tir::BufferStoreNode *op) final {
      if (IsSunmmioLocalVarBuffer(op->buffer)) {
        const tir::VarNode *data = op->buffer->data.get();
        auto it = local_var_table_.find(data);
        if (it != local_var_table_.end() && seen_.insert(data).second) {
          values.push_back(it->second);
        }
      }
      tir::StmtVisitor::VisitStmt_(op);
    }

    const std::unordered_map<const tir::VarNode *, SunMMIOValue>
        &local_var_table_;
    std::unordered_set<const tir::VarNode *> seen_;
  };

  Collector collector(local_var_table_);
  collector(stmt);
  return std::move(collector.values);
}

static void
AppendUniqueLocalVarLiveOutValues(std::vector<SunMMIOValue> *dst,
                                  const std::vector<SunMMIOValue> &src) {
  std::unordered_set<std::string> seen;
  for (const SunMMIOValue &value : *dst) {
    seen.insert(value.value);
  }
  for (const SunMMIOValue &value : src) {
    if (seen.insert(value.value).second) {
      dst->push_back(value);
    }
  }
}

void CodeGenTileLangSunMMIO::EmitFor(const tir::ForNode *op) {
  std::vector<SunMMIOValue> local_live_out_values =
      CollectLocalVarLiveOutValues(op->body);

  SunMMIOValue min = EnsureIndex(EvalExpr(op->min));
  SunMMIOValue extent = EnsureIndex(EvalExpr(op->extent));
  SunMMIOValue step = op->step.has_value()
                          ? EnsureIndex(EvalExpr(op->step.value()))
                          : EmitConstIndex(1);
  SunMMIOValue upper = builder_->Binary(
      NewValueName(), BinaryOp::kAdd, ArithmeticFlavor::kIndex, min, extent,
      SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}},
      DataType::Int(32));
  std::string iv = "%" + op->loop_var->name_hint;
  builder_->BeginFor(iv, min, upper, step, op->annotations,
                     local_live_out_values);
  EnterScope();
  BindVar(
      op->loop_var,
      SunMMIOValue{
          op->loop_var.dtype(), iv,
          SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}}});
  VisitStmtTracked(op->body);
  ExitScope();
  builder_->EndFor();
}

void CodeGenTileLangSunMMIO::EmitIf(const tir::IfThenElseNode *op) {
  SunMMIOValue cond = EnsureType(
      EvalExpr(op->condition),
      SunMMIOType{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}},
      DataType::Bool());
  std::vector<SunMMIOValue> local_live_out_values =
      CollectLocalVarLiveOutValues(op->then_case);
  if (op->else_case.defined()) {
    std::vector<SunMMIOValue> else_live_out_values =
        CollectLocalVarLiveOutValues(op->else_case.value());
    AppendUniqueLocalVarLiveOutValues(&local_live_out_values,
                                      else_live_out_values);
  }
  builder_->BeginIf(cond, local_live_out_values);
  VisitStmtTracked(op->then_case);
  if (op->else_case.defined()) {
    builder_->BeginElse();
    VisitStmtTracked(op->else_case.value());
  }
  builder_->EndIf();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::ForNode *op) {
  CoverageDomain saved_domain = coverage_domain_;
  if (op->annotations.count(tl::attr::kTileDomain)) {
    coverage_domain_ = CoverageDomain::kTiles;
  }
  MarkVisitedNodeType(op->GetTypeKey());
  if (TryLowerTilesScope(op)) {
    coverage_domain_ = saved_domain;
    return;
  }
  EmitFor(op);
  coverage_domain_ = saved_domain;
}

void CodeGenTileLangSunMMIO::EmitWhile(const tir::WhileNode *op) {
  std::vector<SunMMIOValue> local_live_out_values =
      CollectLocalVarLiveOutValues(op->body);
  builder_->BeginWhile(local_live_out_values);
  SunMMIOValue cond = EnsureType(
      EvalExpr(op->condition),
      SunMMIOType{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}},
      DataType::Bool());
  builder_->BeginWhileBody(cond);
  EnterScope();
  VisitStmtTracked(op->body);
  ExitScope();
  builder_->EndWhile();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::LetStmtNode *op) {
  SunMMIOValue value = EvalExpr(op->value);
  EnterScope();
  BindVar(op->var, value);
  VisitStmtTracked(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AttrStmtNode *op) {
  if (op->attr_key == tir::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag == "blockIdx.x") {
      EnterScope();
      BindVar(iv->var, builder_->GetCoreId(NewValueName(), iv->var.dtype()));
      VisitStmtTracked(op->body);
      ExitScope();
    } else {
      VisitStmtTracked(op->body);
    }
    return;
  }
  ScopedAttr attr{op->node, op->attr_key, EvalExpr(op->value)};
  attr_stack_.push_back(attr);
  VisitStmtTracked(op->body);
  attr_stack_.pop_back();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::IfThenElseNode *op) {
  EmitIf(op);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::WhileNode *op) {
  EmitWhile(op);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AllocateNode *op) {
  EnterScope();
  auto buffer_it = buffer_data_to_buffer_.find(op->buffer_var.get());
  if (buffer_it != buffer_data_to_buffer_.end()) {
    const tir::Buffer &buffer = buffer_it->second;
    if (IsSunmmioLocalVarBuffer(buffer)) {
      EmitLocalVarAlloc(op, buffer);
    } else if (IsSunmmioReduceRegisterTempBuffer(buffer)) {
      // TIR materializes reduce intermediates as alloc_buffer so the algorithm
      // can be expressed with BufferLoad/Store.  On SunMMIO these values live
      // in vector-core tile registers and are lowered inside the Tiles scope as
      // SSA tiles, not as rsram memtensors.
    } else {
      std::string scope = GetAllocateStorageScope(op->buffer_var);
      EmitAlloc(buffer_it->second, scope, op->annotations);
    }
  } else {
    LOG(FATAL) << "SunMMIO SUVM allocate cannot find buffer for variable "
               << op->buffer_var->name_hint;
    TVM_FFI_UNREACHABLE();
  }
  VisitStmtTracked(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AllocateConstNode *op) {
  UnsupportedStmt(
      op, "AllocateConstNode should be lowered/eliminated before SunMMIO "
          "codegen");
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::DeclBufferNode *op) {
  EnterScope();
  if (!IsSunmmioLocalVarBuffer(op->buffer) &&
      !IsSunmmioReduceRegisterTempBuffer(op->buffer)) {
    auto data_it = var_table_.find(op->buffer->data.get());
    RegisterBuffer(op->buffer, false,
                   data_it != var_table_.end() ? data_it->second.value
                                               : NewValueName());
  }
  VisitStmtTracked(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BufferStoreNode *op) {
  if (IsSunmmioLocalVarBuffer(op->buffer)) {
    EmitLocalVarStore(op->buffer, op->indices, EvalExpr(op->value));
    return;
  }
  if (op->predicate.defined()) {
    SunMMIOType bool_ty{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
    SunMMIOValue cond =
        EnsureType(EvalExpr(op->predicate.value()), bool_ty, DataType::Bool());
    builder_->BeginIf(cond, std::vector<SunMMIOValue>{});
    EmitScalarTileSet(op);
    builder_->EndIf();
    return;
  }
  EmitScalarTileSet(op);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BufferRealizeNode *op) {
  UnsupportedStmt(
      op, "BufferRealizeNode should be lowered into a concrete view/alias "
          "representation before SunMMIO codegen");
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::AssertStmtNode *op) {
  SunMMIOValue cond = EnsureType(
      EvalExpr(op->condition),
      SunMMIOType{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}},
      DataType::Bool());
  SunMMIOValue msg = EvalExpr(op->message);
  std::string text = msg.value.empty() ? "\"assertion failed\"" : msg.value;
  builder_->EmitAssert(cond, text);
  VisitStmtTracked(op->body);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::EvaluateNode *op) {
  if (const auto *call = op->value.as<tir::CallNode>()) {
    if (call->op.same_as(tir::builtin::ret())) {
      MarkVisitedExprRoot(op->value);
      ICHECK_EQ(call->args.size(), 1) << "tir.ret expects one argument";
      const auto *imm = call->args[0].as<tir::IntImmNode>();
      ICHECK(imm && imm->value == 0)
          << "SunMMIO device kernel only supports T.ret(0)";
      MarkVisitedNodeType(imm->GetTypeKey());
      return;
    }
  }
  (void)EvalExpr(op->value);
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BlockNode *op) {
  auto traverse_range = [this](const Range &range) {
    (void)EvalExpr(range->min);
    (void)EvalExpr(range->extent);
  };
  auto traverse_buffer_region = [&traverse_range](const BufferRegion &region) {
    for (const Range &r : region->region) {
      traverse_range(r);
    }
  };
  auto traverse_annotation_obj = [this](const ffi::Any &value,
                                        const auto &self_ref) -> void {
    if (auto maybe_expr = value.as<PrimExpr>()) {
      (void)EvalExpr(maybe_expr.value());
      return;
    }
    if (auto maybe_arr_expr = value.as<ffi::Array<PrimExpr>>()) {
      for (const PrimExpr &item : maybe_arr_expr.value()) {
        (void)EvalExpr(item);
      }
      return;
    }
    if (auto maybe_arr_any = value.as<ffi::Array<ffi::Any>>()) {
      for (const ffi::Any &item : maybe_arr_any.value()) {
        self_ref(item, self_ref);
      }
      return;
    }
    if (auto maybe_map_expr = value.as<ffi::Map<ffi::String, PrimExpr>>()) {
      for (const auto &kv : maybe_map_expr.value()) {
        (void)EvalExpr(kv.second);
      }
      return;
    }
    if (auto maybe_map_any = value.as<ffi::Map<ffi::String, ffi::Any>>()) {
      for (const auto &kv : maybe_map_any.value()) {
        self_ref(kv.second, self_ref);
      }
      return;
    }
    if (auto maybe_map_any_any = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
      for (const auto &kv : maybe_map_any_any.value()) {
        self_ref(kv.first, self_ref);
        self_ref(kv.second, self_ref);
      }
      return;
    }
  };

  EnterScope();
  for (const IterVar &iv : op->iter_vars) {
    if (!var_table_.count(iv->var.get())) {
      LOG(FATAL) << "CodeGenTileLangSunMMIO: unbound block iter var `"
                 << iv->var->name_hint
                 << "` reached SunMMIO codegen without a BlockRealize binding";
      TVM_FFI_UNREACHABLE();
    }
  }
  for (const Buffer &alloc : op->alloc_buffers) {
    RegisterBuffer(alloc, false);
  }
  for (const MatchBufferRegion &match : op->match_buffers) {
    if (match->source.defined()) {
      RegisterBuffer(match->source->buffer, false);
      traverse_buffer_region(match->source);
    }
    if (!buffer_registry_.count(match->buffer.get())) {
      if (match->source.defined() &&
          buffer_registry_.count(match->source->buffer.get())) {
        const BufferBinding &src = LookupBuffer(match->source->buffer);
        RegisterBuffer(match->buffer, false, src.handle);
      } else {
        RegisterBuffer(match->buffer, false, NewValueName());
      }
    }
  }
  for (const BufferRegion &r : op->reads) {
    RegisterBuffer(r->buffer, false);
    traverse_buffer_region(r);
  }
  for (const BufferRegion &r : op->writes) {
    RegisterBuffer(r->buffer, false);
    traverse_buffer_region(r);
  }
  for (const auto &kv : op->annotations) {
    traverse_annotation_obj(kv.second, traverse_annotation_obj);
  }
  if (op->init.defined()) {
    VisitStmtTracked(op->init.value());
  }
  VisitStmtTracked(op->body);
  ExitScope();
}

void CodeGenTileLangSunMMIO::VisitStmt_(const tir::BlockRealizeNode *op) {
  UnsupportedStmt(
      op, "BlockRealizeNode should be eliminated by LowerOpaqueBlock before "
          "SunMMIO codegen");
}

void CodeGenTileLangSunMMIO::VisitStmtDefault_(const Object *op) {
  UnsupportedStmt(op, "No direct MLIR lowering handler implemented.");
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::VarNode *op) {
  return LookupVar(op);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::SizeVarNode *op) {
  return LookupVar(op);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::IntImmNode *op) {
  DataType dtype = CanonicalizeSuvmDType(op->dtype);
  SunMMIOType ty = MapType(dtype);
  return builder_->ConstantInt(NewValueName(), op->value, ty, dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::FloatImmNode *op) {
  std::ostringstream os;
  os << op->value;
  DataType dtype = CanonicalizeSuvmDType(op->dtype);
  SunMMIOType ty = MapType(dtype);
  return builder_->ConstantFloat(NewValueName(), os.str(), ty, dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::StringImmNode *op) {
  DataType dtype = CanonicalizeSuvmDType(op->dtype);
  return SunMMIOValue{dtype, "\"" + static_cast<std::string>(op->value) + "\"",
                      SunMMIOType{SunMMIOType::Kind::kUnknown, dtype, 1, {}}};
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::CastNode *op) {
  return EmitCast(EvalExpr(op->value), op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::CallNode *op) {
  return EmitCall(op);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::AddNode *op) {
  return EmitBinary("add", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::SubNode *op) {
  return EmitBinary("sub", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::MulNode *op) {
  return EmitBinary("mul", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::DivNode *op) {
  return EmitBinary("div", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::ModNode *op) {
  return EmitBinary("mod", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::FloorDivNode *op) {
  return EmitBinary("floordiv", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::FloorModNode *op) {
  return EmitBinary("floormod", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::MinNode *op) {
  return EmitBinary("min", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::MaxNode *op) {
  return EmitBinary("max", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::EQNode *op) {
  return EmitCmp("eq", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::NENode *op) {
  return EmitCmp("ne", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::LTNode *op) {
  return EmitCmp("lt", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::LENode *op) {
  return EmitCmp("le", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::GTNode *op) {
  return EmitCmp("gt", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::GENode *op) {
  return EmitCmp("ge", op->a, op->b);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::AndNode *op) {
  return EmitBinary("and", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::OrNode *op) {
  return EmitBinary("or", op->a, op->b, op->dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::NotNode *op) {
  SunMMIOType bool_ty{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
  SunMMIOValue v = EnsureType(EvalExpr(op->a), bool_ty, DataType::Bool());
  SunMMIOValue one =
      builder_->ConstantInt(NewValueName(), 1, bool_ty, DataType::Bool());
  return builder_->Binary(NewValueName(), BinaryOp::kXor,
                          ArithmeticFlavor::kBool, v, one, bool_ty,
                          DataType::Bool());
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::SelectNode *op) {
  SunMMIOType bool_ty{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
  SunMMIOValue cond =
      EnsureType(EvalExpr(op->condition), bool_ty, DataType::Bool());
  SunMMIOValue tv = EvalExpr(op->true_value);
  SunMMIOValue fv = EvalExpr(op->false_value);
  fv = EnsureType(fv, tv.type, tv.dtype);
  DataType dtype = CanonicalizeSuvmDType(op->dtype);
  return builder_->Select(NewValueName(), cond, tv, fv, tv.type, dtype);
}

SunMMIOValue
CodeGenTileLangSunMMIO::EmitLoad(const tir::Buffer &buffer,
                                 const ffi::Array<PrimExpr> &indices) {
  if (IsSunmmioLocalVarBuffer(buffer)) {
    return EmitLocalVarLoad(buffer, indices);
  }
  const BufferBinding &binding = LookupBuffer(buffer);
  std::vector<SunMMIOValue> idx_vals;
  for (const PrimExpr &idx : indices) {
    idx_vals.push_back(EnsureIndex(EvalExpr(idx)));
  }
  DataType dtype = CanonicalizeSuvmDType(buffer->dtype);
  return builder_->Load(NewValueName(), binding.handle, idx_vals,
                        binding.buffer_type, dtype, MapType(dtype));
}

SunMMIOValue
CodeGenTileLangSunMMIO::EmitScalarTilePick(const tir::BufferLoadNode *op) {
  const tir::Buffer &buffer = op->buffer;
  const ffi::Array<PrimExpr> &indices = op->indices;
  const std::string scope = buffer.scope();
  DataType dtype = CanonicalizeSuvmDType(buffer->dtype).with_lanes(1);
  auto layout = builder_->LookupLayout(buffer);
  auto describe = [&]() {
    return DescribeScalarTileAccess(buffer, indices, dtype, layout);
  };

  if (op->predicate.defined()) {
    UnsupportedExpr(op,
                    "Sunmmio scalar BufferLoad with predicate is not supported "
                    "yet; use an explicit if_then_else or guard the scalar "
                    "access in control flow before codegen. " +
                        describe());
  }
  if (IsSunmmioAsramScope(scope) || IsSunmmioWsramScope(scope)) {
    UnsupportedExpr(
        op, "Sunmmio scalar BufferLoad cannot read from ASRAM/WSRAM; stage "
            "readable side data through RSRAM first. " +
                describe());
  }
  if (!IsSunmmioRsramScope(scope)) {
    UnsupportedExpr(
        op, "Sunmmio scalar BufferLoad from DRAM/global must be legalized by "
            "staging through RSRAM before codegen. " +
                describe());
  }

  const BufferBinding &binding = LookupBuffer(buffer);
  if (!SupportsSuvmTilePickDType(dtype)) {
    UnsupportedExpr(op, "Sunmmio scalar BufferLoad tile.pick supports only "
                        "i16/ui16/i32/ui32/bf16/f32. " +
                            describe());
  }
  if (indices.empty()) {
    UnsupportedExpr(op, "Sunmmio RSRAM scalar BufferLoad tile.pick currently "
                        "requires a rank >= 1 RSRAM buffer. " +
                            describe());
  }

  const tl::SunmmioTileProcessorConfig tile_processor_config =
      tl::GetSunmmioTileProcessorConfig(target_);
  std::string plan_failure;
  std::optional<ScalarTileAccessPlan> plan = PlanScalarTileAccess(
      buffer, indices, dtype, layout, tile_processor_config, &plan_failure);
  if (!plan.has_value()) {
    UnsupportedExpr(op, "Sunmmio scalar BufferLoad cannot infer a legal "
                        "tile.pick view: " +
                            plan_failure + ". " + describe());
  }
  for (int dim : plan->tiled_dims) {
    MarkVisitedExprTree(indices[dim]);
  }

  std::vector<SunMMIOValue> partition_indices;
  partition_indices.reserve(plan->partition_indices.size());
  for (const PrimExpr &idx : plan->partition_indices) {
    partition_indices.push_back(EnsureIndex(EvalExpr(idx)));
  }
  std::vector<SunMMIOValue> local_indices;
  local_indices.reserve(plan->local_indices.size());
  for (const PrimExpr &idx : plan->local_indices) {
    local_indices.push_back(EnsureIndex(EvalExpr(idx)));
  }

  SunMMIOValue memtensor{dtype, binding.handle, binding.buffer_type};
  SunMMIOType tile_view_type = MakeTileTypeForShape(
      dtype, plan->tile_shape, SunMMIOType::Kind::kTileView);
  SunMMIOValue tile_view = builder_->GetPartitionedTileView(
      NewValueName(), memtensor, partition_indices, plan->tiled_dims,
      tile_view_type, dtype);
  SunMMIOType tile_type =
      MakeTileTypeForShape(dtype, plan->tile_shape, SunMMIOType::Kind::kTile);
  SunMMIOValue tile = builder_->TileLoad(NewValueName(), tile_view, tile_type,
                                         std::nullopt, std::nullopt, dtype);
  return builder_->TilePick(NewValueName(), tile, local_indices,
                            MakeScalarType(dtype), dtype);
}

void CodeGenTileLangSunMMIO::EmitScalarTileSet(const tir::BufferStoreNode *op) {
  const tir::Buffer &buffer = op->buffer;
  const ffi::Array<PrimExpr> &indices = op->indices;
  const std::string scope = buffer.scope();
  DataType dtype = CanonicalizeSuvmDType(buffer->dtype).with_lanes(1);
  auto layout = builder_->LookupLayout(buffer);
  auto describe = [&]() {
    return DescribeScalarTileAccess(buffer, indices, dtype, layout);
  };

  if (IsSunmmioAsramScope(scope) || IsSunmmioWsramScope(scope)) {
    UnsupportedStmt(op,
                    "Sunmmio scalar BufferStore cannot update ASRAM/WSRAM via "
                    "tile.set because the old tile value must be read first; "
                    "stage mutable scalar side data through RSRAM. " +
                        describe());
  }
  if (!IsSunmmioRsramScope(scope)) {
    UnsupportedStmt(op, "Sunmmio scalar BufferStore to DRAM/global must be "
                        "legalized by staging through RSRAM before codegen. " +
                            describe());
  }

  const BufferBinding &binding = LookupBuffer(buffer);
  if (!SupportsSuvmTilePickDType(dtype)) {
    UnsupportedStmt(op, "Sunmmio scalar BufferStore tile.set supports only "
                        "i16/ui16/i32/ui32/bf16/f32. " +
                            describe());
  }
  if (indices.empty()) {
    UnsupportedStmt(op, "Sunmmio RSRAM scalar BufferStore tile.set requires a "
                        "rank >= 1 RSRAM buffer. " +
                            describe());
  }

  const tl::SunmmioTileProcessorConfig tile_processor_config =
      tl::GetSunmmioTileProcessorConfig(target_);
  std::string plan_failure;
  std::optional<ScalarTileAccessPlan> plan = PlanScalarTileAccess(
      buffer, indices, dtype, layout, tile_processor_config, &plan_failure);
  if (!plan.has_value()) {
    UnsupportedStmt(op, "Sunmmio scalar BufferStore cannot infer a legal "
                        "tile.set view: " +
                            plan_failure + ". " + describe());
  }
  for (int dim : plan->tiled_dims) {
    MarkVisitedExprTree(indices[dim]);
  }

  std::vector<SunMMIOValue> partition_indices;
  partition_indices.reserve(plan->partition_indices.size());
  for (const PrimExpr &idx : plan->partition_indices) {
    partition_indices.push_back(EnsureIndex(EvalExpr(idx)));
  }
  std::vector<SunMMIOValue> local_indices;
  local_indices.reserve(plan->local_indices.size());
  for (const PrimExpr &idx : plan->local_indices) {
    local_indices.push_back(EnsureIndex(EvalExpr(idx)));
  }

  SunMMIOValue memtensor{dtype, binding.handle, binding.buffer_type};
  SunMMIOType tile_view_type = MakeTileTypeForShape(
      dtype, plan->tile_shape, SunMMIOType::Kind::kTileView);
  SunMMIOValue tile_view = builder_->GetPartitionedTileView(
      NewValueName(), memtensor, partition_indices, plan->tiled_dims,
      tile_view_type, dtype);
  SunMMIOType tile_type =
      MakeTileTypeForShape(dtype, plan->tile_shape, SunMMIOType::Kind::kTile);
  SunMMIOValue old_tile = builder_->TileLoad(
      NewValueName(), tile_view, tile_type, std::nullopt, std::nullopt, dtype);
  SunMMIOValue scalar_value =
      EnsureType(EvalExpr(op->value), MakeScalarType(dtype), dtype);
  SunMMIOValue new_tile = builder_->TileSet(
      NewValueName(), scalar_value, old_tile, local_indices, tile_type, dtype);
  builder_->TileStore(new_tile, tile_view, std::nullopt);
}

void CodeGenTileLangSunMMIO::EmitStore(const tir::Buffer &buffer,
                                       const ffi::Array<PrimExpr> &indices,
                                       const SunMMIOValue &value) {
  if (IsSunmmioLocalVarBuffer(buffer)) {
    EmitLocalVarStore(buffer, indices, value);
    return;
  }
  const BufferBinding &binding = LookupBuffer(buffer);
  std::vector<SunMMIOValue> idx_vals;
  for (const PrimExpr &idx : indices) {
    idx_vals.push_back(EnsureIndex(EvalExpr(idx)));
  }
  DataType dtype = CanonicalizeSuvmDType(buffer->dtype);
  SunMMIOValue casted = EnsureType(value, MapType(dtype), dtype);
  builder_->Store(casted, binding.handle, idx_vals, binding.buffer_type);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::BufferLoadNode *op) {
  if (IsSunmmioLocalVarBuffer(op->buffer)) {
    return EmitLocalVarLoad(op->buffer, op->indices);
  }
  return EmitScalarTilePick(op);
}

SunMMIOValue
CodeGenTileLangSunMMIO::VisitExpr_(const tir::ProducerLoadNode *op) {
  UnsupportedExpr(op, "ProducerLoadNode is not supported.");
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::RampNode *op) {
  DataType vec_dtype = CanonicalizeSuvmDType(op->dtype);
  DataType elem_dtype = vec_dtype.with_lanes(1);
  SunMMIOType elem_ty = MapType(elem_dtype);
  SunMMIOType vec_ty = MapType(vec_dtype);

  SunMMIOValue base = EvalExpr(op->base);
  SunMMIOValue stride = EvalExpr(op->stride);
  base = EnsureType(base, elem_ty, elem_dtype);
  stride = EnsureType(stride, elem_ty, elem_dtype);

  return builder_->Ramp(NewValueName(), base, stride, vec_dtype.lanes(),
                        elem_ty, vec_ty, vec_dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::BroadcastNode *op) {
  SunMMIOValue scalar = EvalExpr(op->value);
  DataType vec_dtype = CanonicalizeSuvmDType(op->dtype);
  DataType scalar_dtype = vec_dtype.with_lanes(1);
  SunMMIOType scalar_ty = MapType(scalar_dtype);
  SunMMIOType vec_ty = MapType(vec_dtype);
  scalar = EnsureType(scalar, scalar_ty, scalar_dtype);

  return builder_->Broadcast(NewValueName(), scalar, vec_dtype.lanes(),
                             scalar_ty, vec_ty, vec_dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::ShuffleNode *op) {
  UnsupportedExpr(op, "ShuffleNode lowering is not implemented yet.");
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExpr_(const tir::LetNode *op) {
  SunMMIOValue value = EvalExpr(op->value);
  EnterScope();
  BindVar(op->var, value);
  SunMMIOValue body = EvalExpr(op->body);
  ExitScope();
  return body;
}

SunMMIOValue CodeGenTileLangSunMMIO::VisitExprDefault_(const Object *op) {
  UnsupportedExpr(op, "Expr node is not supported in SunMMIO direct lowering.");
}

void CodeGenTileLangSunMMIO::EnterScope() {
  var_scope_markers_.push_back(scoped_vars_.size());
  local_var_scope_markers_.push_back(scoped_local_vars_.size());
  buffer_scope_markers_.push_back(scoped_buffers_.size());
}

void CodeGenTileLangSunMMIO::ExitScope() {
  ICHECK(!var_scope_markers_.empty());
  ICHECK(!local_var_scope_markers_.empty());
  ICHECK(!buffer_scope_markers_.empty());

  size_t var_marker = var_scope_markers_.back();
  var_scope_markers_.pop_back();
  while (scoped_vars_.size() > var_marker) {
    const tir::VarNode *var = scoped_vars_.back();
    scoped_vars_.pop_back();
    var_table_.erase(var);
  }

  size_t local_var_marker = local_var_scope_markers_.back();
  local_var_scope_markers_.pop_back();
  while (scoped_local_vars_.size() > local_var_marker) {
    const tir::VarNode *var = scoped_local_vars_.back();
    scoped_local_vars_.pop_back();
    local_var_table_.erase(var);
  }

  size_t buffer_marker = buffer_scope_markers_.back();
  buffer_scope_markers_.pop_back();
  while (scoped_buffers_.size() > buffer_marker) {
    const tir::BufferNode *buffer = scoped_buffers_.back();
    scoped_buffers_.pop_back();
    buffer_registry_.erase(buffer);
  }
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitBinary(const char *op_name,
                                                const tvm::PrimExpr &lhs,
                                                const tvm::PrimExpr &rhs,
                                                tvm::DataType dtype) {
  dtype = CanonicalizeSuvmDType(dtype);
  SunMMIOValue a = EvalExpr(lhs);
  SunMMIOValue b = EvalExpr(rhs);
  SunMMIOType result_type = MapType(dtype);
  a = EnsureType(a, result_type, dtype);
  b = EnsureType(b, result_type, dtype);
  std::string out = NewValueName();
  const std::string op(op_name);
  BinaryOp bin_op;
  if (op == "add")
    bin_op = BinaryOp::kAdd;
  else if (op == "sub")
    bin_op = BinaryOp::kSub;
  else if (op == "mul")
    bin_op = BinaryOp::kMul;
  else if (op == "div" || op == "floordiv")
    bin_op = BinaryOp::kDiv;
  else if (op == "mod" || op == "floormod")
    bin_op = BinaryOp::kMod;
  else if (op == "min")
    bin_op = BinaryOp::kMin;
  else if (op == "max")
    bin_op = BinaryOp::kMax;
  else if (op == "and")
    bin_op = BinaryOp::kAnd;
  else if (op == "or")
    bin_op = BinaryOp::kOr;
  else if (op == "xor")
    bin_op = BinaryOp::kXor;
  else if (op == "shl")
    bin_op = BinaryOp::kShl;
  else if (op == "shr")
    bin_op = BinaryOp::kShr;
  else
    UnsupportedExpr(lhs.get(), "Unsupported binary op in EmitBinary: " + op);

  return builder_->Binary(out, bin_op, GetArithmeticFlavor(dtype), a, b,
                          result_type, dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitCmp(const char *pred,
                                             const tvm::PrimExpr &lhs,
                                             const tvm::PrimExpr &rhs) {
  SunMMIOValue a = EvalExpr(lhs);
  SunMMIOValue b = EvalExpr(rhs);
  SunMMIOType ty = a.type;
  b = EnsureType(b, ty, a.dtype);
  std::string out = NewValueName();
  CompareOp cmp_op;
  std::string p(pred);
  if (p == "eq")
    cmp_op = CompareOp::kEQ;
  else if (p == "ne")
    cmp_op = CompareOp::kNE;
  else if (p == "lt")
    cmp_op = CompareOp::kLT;
  else if (p == "le")
    cmp_op = CompareOp::kLE;
  else if (p == "gt")
    cmp_op = CompareOp::kGT;
  else if (p == "ge")
    cmp_op = CompareOp::kGE;
  else
    UnsupportedExpr(lhs.get(), "Unsupported compare op in EmitCmp: " + p);
  return builder_->Compare(out, cmp_op, GetCompareDomain(a.dtype), a, b, ty);
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitCast(const SunMMIOValue &v,
                                              tvm::DataType target_dtype) {
  target_dtype = CanonicalizeSuvmDType(target_dtype);
  SunMMIOType dst = MapType(target_dtype);
  if (v.type.kind == dst.kind && v.type.dtype == dst.dtype &&
      v.type.lanes == dst.lanes && SameTypeShape(v.type, dst)) {
    return v;
  }
  return builder_->Cast(NewValueName(), v, dst, target_dtype);
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitMXPackOrUnpack(const tir::CallNode *op,
                                                        bool is_pack) {
  auto normalize_region = [&](const PrimExpr &expr) {
    return NormalizeRegionTracked(expr);
  };
  auto check_full_region = [&](const BufferRegion &region, const char *name) {
    ICHECK_EQ(region->region.size(), region->buffer->shape.size())
        << "tl.mx_pack/unpack " << name
        << " region rank must match buffer rank";
    arith::Analyzer analyzer;
    for (size_t i = 0; i < region->region.size(); ++i) {
      const Range &range = region->region[i];
      bool min_is_zero =
          analyzer.CanProveEqual(range->min, make_zero(range->min.dtype()));
      ICHECK(min_is_zero) << "tl.mx_pack/unpack only supports full regions; "
                          << name << " dim " << i << " has min " << range->min;
      MarkVisitedExprTree(range->min);
      bool extent_matches =
          analyzer.CanProveEqual(range->extent, region->buffer->shape[i]);
      ICHECK(extent_matches)
          << "tl.mx_pack/unpack only supports full regions; " << name << " dim "
          << i << " has extent " << range->extent << " but buffer shape is "
          << region->buffer->shape[i];
      MarkVisitedExprTree(range->extent);
      MarkVisitedExprTree(region->buffer->shape[i]);
    }
  };
  auto check_rank2_static = [&](const Buffer &buffer, const char *name) {
    ICHECK_EQ(buffer->shape.size(), 2U)
        << "tl.mx_pack/unpack expects rank-2 " << name << " buffer";
    for (const PrimExpr &dim : buffer->shape) {
      ICHECK(dim.as<IntImmNode>()) << "tl.mx_pack/unpack expects static "
                                   << name << " shape, got " << dim;
      MarkVisitedExprRoot(dim);
    }
  };
  auto make_index_type = []() {
    return SunMMIOType{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}};
  };
  auto make_loop_index = [&](const std::string &name) {
    return SunMMIOValue{DataType::Int(32), name, make_index_type()};
  };
  auto begin_for = [&](const std::string &iv_name, int64_t upper) {
    builder_->BeginFor(iv_name, EmitConstIndex(0), EmitConstIndex(upper),
                       EmitConstIndex(1), ffi::Map<ffi::String, ffi::Any>(),
                       std::vector<SunMMIOValue>{});
  };
  auto make_memtensor_value = [](const BufferBinding &binding, DataType dtype) {
    return SunMMIOValue{CanonicalizeSuvmDType(dtype).with_lanes(1),
                        binding.handle, binding.buffer_type};
  };
  auto get_tile_view = [&](const SunMMIOValue &memtensor, DataType dtype,
                           const std::vector<SunMMIOValue> &indices,
                           const std::vector<int64_t> &tiled_dims,
                           const std::vector<int64_t> &shape) {
    return builder_->GetPartitionedTileView(
        NewValueName(), memtensor, indices, tiled_dims,
        MakeTileViewType(dtype, shape),
        CanonicalizeSuvmDType(dtype).with_lanes(1));
  };
  auto load_tile = [&](const SunMMIOValue &view, DataType dtype,
                       const std::vector<int64_t> &shape) {
    return builder_->TileLoad(NewValueName(), view, MakeTileType(dtype, shape),
                              std::nullopt, std::nullopt,
                              CanonicalizeSuvmDType(dtype).with_lanes(1));
  };
  auto copy_tile = [&](const SunMMIOValue &src, const SunMMIOValue &dst,
                       DataType dtype, const std::vector<SunMMIOValue> &indices,
                       const std::vector<int64_t> &tiled_dims,
                       const std::vector<int64_t> &shape) {
    SunMMIOValue src_view =
        get_tile_view(src, dtype, indices, tiled_dims, shape);
    SunMMIOValue dst_view =
        get_tile_view(dst, dtype, indices, tiled_dims, shape);
    SunMMIOValue tile = load_tile(src_view, dtype, shape);
    builder_->TileStore(tile, dst_view, std::nullopt);
  };
  auto emit_scale_copy = [&](const SunMMIOValue &src, const SunMMIOValue &dst,
                             tl::sunmmio::MXLayoutKind layout_kind) {
    constexpr int64_t kScaleValidElems = 32;
    constexpr int64_t kScaleAccessElems = 64;
    ICHECK(layout_kind != tl::sunmmio::MXLayoutKind::kRowMajor)
        << "tl.mx_pack/unpack row-major scale copy is not implemented yet; "
           "waiting for stable suvm.unpack scale alias layout";
    std::vector<int64_t> scale_shape = ExtractStaticShape(src.type);
    std::vector<int64_t> dst_scale_shape = ExtractStaticShape(dst.type);
    std::vector<int64_t> src_physical = ExtractPhysicalExtents(src.type);
    std::vector<int64_t> dst_physical = ExtractPhysicalExtents(dst.type);
    ICHECK(scale_shape == dst_scale_shape)
        << "MX scale source/destination logical shapes must match";
    ICHECK_EQ(scale_shape.size(), 2U)
        << "MX scale alias/user buffer must be rank-2";
    ICHECK_EQ(scale_shape[1], kScaleValidElems)
        << "MX scale logical width must be 32, got " << scale_shape[1];
    ICHECK_EQ(src_physical.size(), 2U)
        << "MX scale source physical extent must be rank-2";
    ICHECK_EQ(dst_physical.size(), 2U)
        << "MX scale destination physical extent must be rank-2";
    ICHECK_GE(src_physical[1], kScaleValidElems)
        << "MX scale source physical width must cover 32 logical elements";
    ICHECK_GE(dst_physical[1], kScaleValidElems)
        << "MX scale destination physical width must cover 32 logical elements";

    ICHECK(src_physical[1] >= kScaleAccessElems &&
           dst_physical[1] >= kScaleAccessElems)
        << "MX scale copy requires source and destination physical width to "
           "cover a 64B fp8 tile access; got source width "
        << src_physical[1] << " and destination width " << dst_physical[1];

    std::string row_name = NewValueName();
    begin_for(row_name, scale_shape[0]);
    SunMMIOValue row = make_loop_index(row_name);
    std::vector<SunMMIOValue> indices{row, EmitConstIndex(0)};
    std::vector<int64_t> tiled_dims{0, 1};
    std::vector<int64_t> valid_shape{1, kScaleValidElems};

    std::vector<int64_t> access_shape{1, kScaleAccessElems};
    std::vector<SunMMIOValue> offsets{EmitConstIndex(0), EmitConstIndex(0)};

    SunMMIOValue src_view = get_tile_view(src, ExpectedMXScaleDType(), indices,
                                          tiled_dims, access_shape);
    SunMMIOValue src64 =
        load_tile(src_view, ExpectedMXScaleDType(), access_shape);
    SunMMIOValue src32 =
        builder_->TileSlice(NewValueName(), src64, offsets,
                            MakeTileType(ExpectedMXScaleDType(), valid_shape),
                            ExpectedMXScaleDType());

    SunMMIOValue dst_view = get_tile_view(dst, ExpectedMXScaleDType(), indices,
                                          tiled_dims, access_shape);
    SunMMIOValue dst64 =
        load_tile(dst_view, ExpectedMXScaleDType(), access_shape);
    SunMMIOValue merged = builder_->TileInsertSlice(
        NewValueName(), dst64, src32, offsets,
        MakeTileType(ExpectedMXScaleDType(), access_shape),
        ExpectedMXScaleDType());
    builder_->TileStore(merged, dst_view, std::nullopt);
    builder_->EndFor();
  };
  auto emit_row_major_data_copy = [&](const SunMMIOValue &src,
                                      const SunMMIOValue &dst, DataType dtype) {
    std::vector<int64_t> shape = ExtractStaticShape(src.type);
    std::vector<int64_t> physical = ExtractPhysicalExtents(src.type);
    ICHECK_EQ(shape.size(), 2U);
    ICHECK_EQ(physical.size(), 2U);
    int64_t access_elems = 64 * 8 / dtype.bits();
    ICHECK_GT(access_elems, 0);
    ICHECK_EQ(physical[1] % access_elems, 0)
        << "MX row-major data physical width must be aligned to 64B, got "
        << physical[1] << " elements for dtype " << dtype;
    int64_t col_tiles = physical[1] / access_elems;

    std::string row_name = NewValueName();
    begin_for(row_name, shape[0]);
    SunMMIOValue row = make_loop_index(row_name);
    std::string col_name = NewValueName();
    begin_for(col_name, col_tiles);
    SunMMIOValue col = make_loop_index(col_name);
    copy_tile(src, dst, dtype, {row, col}, {0, 1}, {1, access_elems});
    builder_->EndFor();
    builder_->EndFor();
  };
  auto emit_blockwise_data_copy = [&](const SunMMIOValue &src,
                                      const SunMMIOValue &dst, DataType dtype,
                                      bool zn_order) {
    std::vector<int64_t> physical = ExtractPhysicalExtents(src.type);
    ICHECK_EQ(physical.size(), 2U);
    ICHECK_EQ(physical[0] % 32, 0);
    ICHECK_EQ(physical[1] % 32, 0);
    int64_t block_m = physical[0] / 32;
    int64_t block_n = physical[1] / 32;
    std::vector<int64_t> tiled_dims =
        zn_order ? std::vector<int64_t>{1, 0} : std::vector<int64_t>{0, 1};

    std::string bm_name = NewValueName();
    begin_for(bm_name, block_m);
    SunMMIOValue bm = make_loop_index(bm_name);
    std::string bn_name = NewValueName();
    begin_for(bn_name, block_n);
    SunMMIOValue bn = make_loop_index(bn_name);
    copy_tile(src, dst, dtype, {bm, bn}, tiled_dims, {32, 32});
    builder_->EndFor();
    builder_->EndFor();
  };

  ICHECK_EQ(op->args.size(), 3U)
      << (is_pack ? "tl.mx_pack" : "tl.mx_unpack") << " expects 3 args";
  BufferRegion data_region;
  BufferRegion scale_region;
  BufferRegion mx_region;
  if (is_pack) {
    data_region = normalize_region(op->args[0]);
    scale_region = normalize_region(op->args[1]);
    mx_region = normalize_region(op->args[2]);
  } else {
    mx_region = normalize_region(op->args[0]);
    data_region = normalize_region(op->args[1]);
    scale_region = normalize_region(op->args[2]);
  }
  check_full_region(data_region, "data");
  check_full_region(scale_region, "scale");
  check_full_region(mx_region, "mx");
  check_rank2_static(data_region->buffer, "data");
  check_rank2_static(scale_region->buffer, "scale");
  check_rank2_static(mx_region->buffer, "mx");

  const Buffer &data_buffer = data_region->buffer;
  const Buffer &scale_buffer = scale_region->buffer;
  const Buffer &mx_buffer = mx_region->buffer;
  ICHECK(data_buffer.scope() == tl::kSunmmioScopeRSRAM &&
         scale_buffer.scope() == tl::kSunmmioScopeRSRAM &&
         mx_buffer.scope() == tl::kSunmmioScopeRSRAM)
      << "tl.mx_pack/unpack expects data, scale, and mx in shared.rsram";
  ICHECK(data_buffer->dtype == ExpectedMXDataDType(mx_buffer->dtype))
      << "data dtype does not match mx dtype";
  ICHECK(scale_buffer->dtype == ExpectedMXScaleDType())
      << "scale dtype must be float8_e8m0fnu";

  ffi::Optional<tl::Layout> mx_layout_opt = builder_->LookupLayout(mx_buffer);
  ICHECK(mx_layout_opt.defined())
      << "tl.mx_pack/unpack requires a concrete MX layout for mx buffer";
  arith::Analyzer analyzer;
  auto analysis = tl::sunmmio::AnalyzeMXLayout(mx_layout_opt.value(),
                                               mx_buffer->dtype, &analyzer);
  ICHECK(analysis.has_value())
      << "tl.mx_pack/unpack supports only MX row-major, MXZZ, and MXZNZ";
  ICHECK(analysis->kind != tl::sunmmio::MXLayoutKind::kMXZNN)
      << "tl.mx_pack/unpack does not accept MXZNN as a user mx buffer layout; "
         "use MXZNZ for RSRAM data staged before WSRAM MXZNN";

  const BufferBinding &data_binding = LookupBuffer(data_buffer);
  const BufferBinding &scale_binding = LookupBuffer(scale_buffer);
  const BufferBinding &mx_binding = LookupBuffer(mx_buffer);
  SunMMIOValue mx_value = make_memtensor_value(mx_binding, mx_buffer->dtype);
  auto unpacked = builder_->MXUnpack(NewValueName(), NewValueName(), mx_value,
                                     ExpectedMXScaleDType(),
                                     ExpectedMXDataDType(mx_buffer->dtype));
  SunMMIOValue scale_alias = unpacked.first;
  SunMMIOValue data_alias = unpacked.second;
  SunMMIOValue data_value =
      make_memtensor_value(data_binding, data_buffer->dtype);
  SunMMIOValue scale_value =
      make_memtensor_value(scale_binding, scale_buffer->dtype);

  SunMMIOValue data_src = is_pack ? data_value : data_alias;
  SunMMIOValue data_dst = is_pack ? data_alias : data_value;
  SunMMIOValue scale_src = is_pack ? scale_value : scale_alias;
  SunMMIOValue scale_dst = is_pack ? scale_alias : scale_value;

  switch (analysis->kind) {
  case tl::sunmmio::MXLayoutKind::kRowMajor:
    emit_row_major_data_copy(data_src, data_dst, data_buffer->dtype);
    break;
  case tl::sunmmio::MXLayoutKind::kMXZZ:
  case tl::sunmmio::MXLayoutKind::kMXZNZ:
    emit_blockwise_data_copy(data_src, data_dst, data_buffer->dtype,
                             /*zn_order=*/false);
    break;
  case tl::sunmmio::MXLayoutKind::kMXZNN:
    LOG(FATAL) << "MXZNN is an internal WSRAM layout and is not accepted by "
                  "tl.mx_pack/unpack";
    break;
  }
  emit_scale_copy(scale_src, scale_dst, analysis->kind);

  return SunMMIOValue{op->dtype, "", MapType(op->dtype)};
}

CodeGenTileLangSunMMIO::CallBucket
CodeGenTileLangSunMMIO::ClassifyCall(const tir::CallNode *op) const {
  if (op->op.as<GlobalVarNode>()) {
    PrimExpr expr = tvm::ffi::GetRef<PrimExpr>(op);
    tir::CallEffectKind effect = SideEffect(expr);
    return effect <= tir::CallEffectKind::kPure ? CallBucket::kExternPure
                                                : CallBucket::kExternSideEffect;
  }
  const auto *op_node = op->op.as<OpNode>();
  if (!op_node) {
    return CallBucket::kUnsupported;
  }
  std::string name = op_node->name;
  if (name == "tl.mma_sunmmio" || name == "tl.dma_copy" ||
      name == "tl.mx_pack" || name == "tl.mx_unpack" ||
      name == "tl.broadcast_" || name.find("sunmmio") != std::string::npos) {
    return CallBucket::kSunMMIOIntrinsic;
  }
  if (name.rfind("tl.", 0) == 0) {
    return CallBucket::kTileLangIntrinsic;
  }
  if (name == "tir.tvm_access_ptr" || name == "tir.address_of" ||
      name.find("alloc") != std::string::npos ||
      name.find("reinterpret") != std::string::npos) {
    return CallBucket::kMemory;
  }
  if (name.find("sync") != std::string::npos ||
      name.find("barrier") != std::string::npos) {
    return CallBucket::kSync;
  }
  if (name.find("shuffle") != std::string::npos ||
      name.find("vector") != std::string::npos) {
    return CallBucket::kVector;
  }
  if (name.find("exp") != std::string::npos ||
      name.find("log") != std::string::npos ||
      name.find("sin") != std::string::npos ||
      name.find("cos") != std::string::npos ||
      name.find("sqrt") != std::string::npos ||
      name.find("pow") != std::string::npos) {
    return CallBucket::kMath;
  }
  if (name.rfind("tir.", 0) == 0) {
    return CallBucket::kBuiltin;
  }
  PrimExpr expr = tvm::ffi::GetRef<PrimExpr>(op);
  tir::CallEffectKind effect = SideEffect(expr);
  return effect <= tir::CallEffectKind::kPure ? CallBucket::kExternPure
                                              : CallBucket::kExternSideEffect;
}

const char *CodeGenTileLangSunMMIO::CallBucketName(CallBucket bucket) const {
  switch (bucket) {
  case CallBucket::kBuiltin:
    return "builtin";
  case CallBucket::kExternPure:
    return "extern_pure";
  case CallBucket::kExternSideEffect:
    return "extern_side_effect";
  case CallBucket::kMath:
    return "math";
  case CallBucket::kMemory:
    return "memory";
  case CallBucket::kSync:
    return "sync";
  case CallBucket::kVector:
    return "vector";
  case CallBucket::kTileLangIntrinsic:
    return "tilelang_intrinsic";
  case CallBucket::kSunMMIOIntrinsic:
    return "sunmmio_intrinsic";
  case CallBucket::kUnsupported:
    return "unsupported";
  }
  return "unsupported";
}

SunMMIOValue
CodeGenTileLangSunMMIO::EmitRegionCall(const tvm::PrimExpr &region_expr,
                                       int64_t byte_offset) {
  BufferRegion region = NormalizeRegionTracked(region_expr);
  const BufferBinding &binding = LookupBuffer(region->buffer);
  std::vector<SunMMIOValue> mins;
  std::vector<int64_t> extents;
  mins.reserve(region->region.size());
  extents.reserve(region->region.size());
  arith::Analyzer analyzer;
  for (const Range &range : region->region) {
    const auto *extent_imm = range->extent.as<IntImmNode>();
    ICHECK(extent_imm) << "tl.tileop.region extent must be IntImm";
    MarkVisitedExprRoot(range->extent);
    extents.push_back(static_cast<int64_t>(extent_imm->value));
    PrimExpr min = floordiv(range->min, range->extent);
    min = analyzer.Simplify(min);
    MarkVisitedExprTree(range->min);
    mins.push_back(EvalExpr(min));
  }
  SunMMIOType ret_ty = MapType(region_expr.dtype());
  std::string result_name = region_expr.dtype().is_void() ? "" : NewValueName();
  return builder_->RegionCall(result_name, binding.handle, mins, extents,
                              region_expr.dtype(), ret_ty, byte_offset);
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitCall(const tir::CallNode *op) {
  CallBucket bucket = ClassifyCall(op);
  if (bucket == CallBucket::kUnsupported) {
    UnsupportedExpr(op, "Unsupported call target.");
  }
  std::string callee = "unknown";
  if (const auto *op_node = op->op.as<OpNode>()) {
    callee = op_node->name;
  } else if (const auto *gv = op->op.as<GlobalVarNode>()) {
    callee = gv->name_hint;
  }
  std::vector<SunMMIOValue> operands;
  SunMMIOCallAttrs attrs;
  if (callee == "tir.if_then_else") {
    ICHECK_EQ(op->args.size(), 3U) << "tir.if_then_else expects 3 arguments";
    SunMMIOType bool_ty{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
    SunMMIOValue cond =
        EnsureType(EvalExpr(op->args[0]), bool_ty, DataType::Bool());
    SunMMIOValue tv = EvalExpr(op->args[1]);
    SunMMIOValue fv = EvalExpr(op->args[2]);
    fv = EnsureType(fv, tv.type, tv.dtype);
    DataType dtype = CanonicalizeSuvmDType(op->dtype);
    return builder_->Select(NewValueName(), cond, tv, fv, tv.type, dtype);
  } else if (callee == "tl.tileop.region") {
    return EmitRegionCall(tvm::ffi::GetRef<PrimExpr>(op));
  } else if (callee == "tl.mx_pack") {
    return EmitMXPackOrUnpack(op, /*is_pack=*/true);
  } else if (callee == "tl.mx_unpack") {
    return EmitMXPackOrUnpack(op, /*is_pack=*/false);
  } else if (callee == "tl.barrier_init" ||
             callee == "tl.barrier_arrive_and_wait") {
    ICHECK_GE(op->args.size(), 1U)
        << callee << " expects participant_mask argument";
    const PrimExpr &mask = op->args[0];
    attrs[SunMMIOCallAttrKey::kBarrierMaskKey] =
        TVMScriptPrinter::Script(mask, std::nullopt);
    if (const auto *imm = mask.as<IntImmNode>()) {
      MarkVisitedNodeType(imm->GetTypeKey());
      int64_t mask_value = static_cast<int64_t>(imm->value);
      if (mask_value >= 0) {
        attrs[SunMMIOCallAttrKey::kParticipantMask] = mask_value;
      }
    } else {
      operands.push_back(EvalExpr(mask));
    }
    std::vector<int64_t> candidate_masks;
    for (size_t i = 1; i < op->args.size(); ++i) {
      const auto *imm = op->args[i].as<IntImmNode>();
      ICHECK(imm) << callee << " candidate masks must be IntImm";
      MarkVisitedNodeType(imm->GetTypeKey());
      int64_t candidate_mask = static_cast<int64_t>(imm->value);
      if (std::find(candidate_masks.begin(), candidate_masks.end(),
                    candidate_mask) == candidate_masks.end()) {
        candidate_masks.push_back(candidate_mask);
      }
    }
    if (!candidate_masks.empty()) {
      attrs[SunMMIOCallAttrKey::kCandidateMasks] = std::move(candidate_masks);
    }
  } else if (callee == "tl.sunmmio_sync") {
    ICHECK_EQ(op->args.size(), 1U)
        << "tl.sunmmio_sync expects one constant unit mask";
    const auto *units = op->args[0].as<IntImmNode>();
    ICHECK(units) << "tl.sunmmio_sync unit mask must be an IntImm";
    ICHECK_GE(units->value, 0)
        << "tl.sunmmio_sync unit mask must be non-negative";
    MarkVisitedNodeType(units->GetTypeKey());
    attrs[SunMMIOCallAttrKey::kSyncUnits] = static_cast<int64_t>(units->value);
  } else if (callee == "tl.dma_copy") {
    ICHECK_EQ(op->args.size(), 4)
        << "tl.dma_copy expects src region, dst region, src_offset_byte, "
           "and odma_unit";
    auto count_tiled_dims = [](const PrimExpr &region_expr) -> int {
      BufferRegion region = tl::NormalizeToBufferRegion(region_expr);
      int count = 0;
      for (const Range &range : region->region) {
        const auto *extent_imm = range->extent.as<IntImmNode>();
        ICHECK(extent_imm) << "tl.dma_copy region extent must be IntImm";
        if (extent_imm->value != 1) {
          ++count;
        }
      }
      return count;
    };

    int src_tiled_dims = count_tiled_dims(op->args[0]);
    int dst_tiled_dims = count_tiled_dims(op->args[1]);

    const auto *src_offset_imm = op->args[2].as<IntImmNode>();
    ICHECK(src_offset_imm)
        << "tl.dma_copy src_offset_byte must be a constant IntImm";
    int64_t src_offset_byte = static_cast<int64_t>(src_offset_imm->value);
    ICHECK_GE(src_offset_byte, 0)
        << "tl.dma_copy src_offset_byte must be non-negative";
    MarkVisitedNodeType(src_offset_imm->GetTypeKey());

    operands.reserve(2);

    ICHECK(TryConsumeSunmmioOdmaUnit(op->args[3], &attrs))
        << "tl.dma_copy expects fourth argument to be tl.odma_unit";

    operands.push_back(EmitRegionCall(op->args[0], src_offset_byte));
    operands.push_back(EmitRegionCall(op->args[1]));
  } else if (callee == "tl.sunmmio_layout_transform") {
    ICHECK_EQ(op->args.size(), 3)
        << "tl.sunmmio_layout_transform expects src region, dst region, and "
           "odma_unit";
    struct LayoutTransformRegionInfo {
      int rank{0};
      int tiled_dims{0};
    };
    auto get_region_info =
        [](const PrimExpr &region_expr) {
          BufferRegion region = tl::NormalizeToBufferRegion(region_expr);
          LayoutTransformRegionInfo info;
          info.rank = static_cast<int>(region->region.size());
          for (const Range &range : region->region) {
            const auto *extent_imm = range->extent.as<IntImmNode>();
            ICHECK(extent_imm)
                << "tl.sunmmio_layout_transform region extent must be IntImm";
            if (extent_imm->value != 1) {
              ++info.tiled_dims;
            }
          }
          return info;
        };

    LayoutTransformRegionInfo src_info = get_region_info(op->args[0]);
    LayoutTransformRegionInfo dst_info = get_region_info(op->args[1]);
    auto is_singleton_1d_region = [](const LayoutTransformRegionInfo &info) {
      return info.rank == 1 && info.tiled_dims == 0;
    };
    bool is_2d_transform = src_info.tiled_dims == 2 && dst_info.tiled_dims == 2;
    bool is_singleton_1d_transform =
        is_singleton_1d_region(src_info) && is_singleton_1d_region(dst_info);
    ICHECK(is_2d_transform || is_singleton_1d_transform)
        << "tl.sunmmio_layout_transform expects source and destination "
           "regions to both have exactly 2 tiled dims, or to both be rank-1 "
           "singleton regions; got source rank="
        << src_info.rank << ", tiled dims=" << src_info.tiled_dims
        << ", destination rank=" << dst_info.rank
        << ", tiled dims=" << dst_info.tiled_dims;

    operands.reserve(2);
    operands.push_back(EmitRegionCall(op->args[0]));
    operands.push_back(EmitRegionCall(op->args[1]));

    ICHECK(TryConsumeSunmmioOdmaUnit(op->args[2], &attrs))
        << "tl.sunmmio_layout_transform expects third argument to be "
           "tl.odma_unit";
  } else if (callee == "tl.sunmmio_transpose") {
    ICHECK_EQ(op->args.size(), 3)
        << "tl.sunmmio_transpose expects src region, dst region, and "
           "odma_unit";
    auto validate_region = [](const PrimExpr &region_expr,
                              const char *operand) {
      BufferRegion region = tl::NormalizeToBufferRegion(region_expr);
      ICHECK_EQ(region->region.size(), 2U)
          << "tl.sunmmio_transpose " << operand << " region must be rank-2";
      for (const Range &range : region->region) {
        const auto *extent = range->extent.as<IntImmNode>();
        ICHECK(extent) << "tl.sunmmio_transpose region extents must be IntImm";
        ICHECK_GT(extent->value, 1)
            << "tl.sunmmio_transpose requires two tiled dimensions";
      }
    };
    validate_region(op->args[0], "source");
    validate_region(op->args[1], "destination");

    operands.reserve(2);
    operands.push_back(EmitRegionCall(op->args[0]));
    operands.push_back(EmitRegionCall(op->args[1]));
    ICHECK(TryConsumeSunmmioOdmaUnit(op->args[2], &attrs))
        << "tl.sunmmio_transpose expects third argument to be tl.odma_unit";
  } else if (callee == "tir.bitwise_and" || callee == "tir.bitwise_or" ||
             callee == "tir.bitwise_xor" || callee == "tir.shift_left" ||
             callee == "tir.shift_right") {
    ICHECK_EQ(op->args.size(), 2) << callee << " expects exactly two arguments";
    if (callee == "tir.bitwise_and") {
      return EmitBinary("and", op->args[0], op->args[1], op->dtype);
    }
    if (callee == "tir.bitwise_or") {
      return EmitBinary("or", op->args[0], op->args[1], op->dtype);
    }
    if (callee == "tir.bitwise_xor") {
      return EmitBinary("xor", op->args[0], op->args[1], op->dtype);
    }
    if (callee == "tir.shift_left") {
      return EmitBinary("shl", op->args[0], op->args[1], op->dtype);
    }
    return EmitBinary("shr", op->args[0], op->args[1], op->dtype);
  } else if (callee == "tl.broadcast_") {
    size_t fixed_and_source_args = op->args.size();
    ICHECK(fixed_and_source_args > 0 &&
           TryConsumeSunmmioOdmaUnit(op->args.back(), &attrs))
        << "tl.broadcast_ expects tl.odma_unit as its final argument";
    --fixed_and_source_args;
    ICHECK(fixed_and_source_args ==
               static_cast<size_t>(tl::kBroadcastArgCount) ||
           fixed_and_source_args ==
               static_cast<size_t>(tl::kBroadcastArgCount + 1))
        << "tl.broadcast_ expects src region, dst region, direction, mask, "
           "src_offset_byte, optional src_core, and odma_unit";

    const auto *direction_imm =
        op->args[tl::kBroadcastArgDirection].as<IntImmNode>();
    ICHECK(direction_imm)
        << "tl.broadcast_ direction must be a constant IntImm";
    int64_t direction = static_cast<int64_t>(direction_imm->value);
    ICHECK(direction == 0 || direction == 1)
        << "tl.broadcast_ MLIR lowering only supports direction 0 or 1, got "
        << direction;
    MarkVisitedNodeType(direction_imm->GetTypeKey());

    int64_t src_offset_byte = 0;
    const auto *src_offset_imm =
        op->args[tl::kBroadcastArgSrcOffsetByte].as<IntImmNode>();
    ICHECK(src_offset_imm)
        << "tl.broadcast_ src_offset_byte must be a constant IntImm";
    src_offset_byte = static_cast<int64_t>(src_offset_imm->value);
    ICHECK_GE(src_offset_byte, 0)
        << "tl.broadcast_ src_offset_byte must be non-negative";
    MarkVisitedNodeType(src_offset_imm->GetTypeKey());

    operands.reserve(4);
    operands.push_back(
        EmitRegionCall(op->args[tl::kBroadcastArgSrc], src_offset_byte));
    operands.push_back(EmitRegionCall(op->args[tl::kBroadcastArgDst]));
    operands.push_back(EvalExpr(op->args[tl::kBroadcastArgMask]));
    if (fixed_and_source_args ==
        static_cast<size_t>(tl::kBroadcastArgCount + 1)) {
      operands.push_back(EvalExpr(op->args[tl::kBroadcastArgSrcCore]));
    }

    attrs[SunMMIOCallAttrKey::kDirection] =
        std::string(direction == 0 ? "row" : "col");
  } else if (callee == "tl.mma_sunmmio") {
    ICHECK_EQ(op->args.size(), 7) << "tl.mma_sunmmio expects A/B/C regions, "
                                     "three flag operands, and "
                                     "acc_offset_byte";
    auto parse_bool_arg = [&](const PrimExpr &arg,
                              const char *arg_name) -> bool {
      const auto *imm = arg.as<IntImmNode>();
      ICHECK(imm) << arg_name << " must be a constant bool";
      ICHECK(imm->dtype.is_bool()) << arg_name << " must have bool dtype";
      return imm->value != 0;
    };

    const auto *acc_offset_imm = op->args[6].as<IntImmNode>();
    ICHECK(acc_offset_imm)
        << "tl.mma_sunmmio acc_offset_byte must be a constant IntImm";
    int64_t acc_offset_byte = static_cast<int64_t>(acc_offset_imm->value);
    ICHECK_GE(acc_offset_byte, 0)
        << "tl.mma_sunmmio acc_offset_byte must be non-negative";
    MarkVisitedNodeType(acc_offset_imm->GetTypeKey());

    operands.reserve(4);
    operands.push_back(EmitRegionCall(op->args[0]));
    operands.push_back(EmitRegionCall(op->args[1]));
    operands.push_back(EmitRegionCall(op->args[2], acc_offset_byte));

    MarkVisitedNodeType(op->args[5]->GetTypeKey());
    arith::Analyzer analyzer;
    PrimExpr accumulate = analyzer.Simplify(tir::Not(op->args[5]));
    SunMMIOType bool_ty{SunMMIOType::Kind::kScalar, DataType::Bool(), 1, {}};
    operands.push_back(
        EnsureType(EvalExpr(accumulate), bool_ty, DataType::Bool()));

    attrs[SunMMIOCallAttrKey::kTransA] =
        parse_bool_arg(op->args[3], "tl.mma_sunmmio transA");
    attrs[SunMMIOCallAttrKey::kTransB] =
        parse_bool_arg(op->args[4], "tl.mma_sunmmio transB");

  } else {
    for (int i = 0, e = static_cast<int>(op->args.size()); i < e; ++i) {
      const PrimExpr &arg = op->args[i];
      if (TryConsumeSunmmioOdmaUnit(arg, &attrs)) {
        continue;
      }
      if (const auto *s = arg.as<StringImmNode>()) {
        MarkVisitedNodeType(s->GetTypeKey());
        continue;
      }
      operands.push_back(EvalExpr(arg));
    }
  }
  DataType ret_dtype = CanonicalizeSuvmDType(op->dtype);
  SunMMIOType ret_ty = MapType(ret_dtype);
  std::string result_name = op->dtype.is_void() ? "" : NewValueName();
  return builder_->Call(result_name, callee, operands, attrs,
                        CallBucketName(bucket), ret_dtype, ret_ty);
}

/*!
 * \brief Backend input invariants for generic SunMMIO codegen.
 *
 * This backend is not a generic TIR code generator. It expects the input to
 * have already been lowered into a SunMMIO-oriented form where tiled buffers
 * are accessed through tile-aware paths rather than generic element-wise
 * BufferLoad/BufferStore. Nodes such as BlockRealize, BufferRealize, and
 * DeclBuffer should normally have been eliminated or lowered before reaching
 * this layer. Likewise, generic BufferLoad/BufferStore on tiled buffers are
 * treated as pipeline violations unless intercepted by a dedicated tile-based
 * lowering path earlier in the pipeline.
 *
 * The generic SunMMIO codegen path is expected to handle control flow, scalar
 * and index expressions, loop structure, target intrinsics, and tile-aware
 * operations that have already been normalized into backend-expected forms.
 * Reaching unsupported structural nodes here should fail loudly so pipeline
 * regressions are caught early.
 */

[[noreturn]] void
CodeGenTileLangSunMMIO::UnsupportedStmt(const Object *op,
                                        const std::string &detail) const {
  // Generic SunMMIO codegen intentionally rejects pre-lowered structural forms.
  // Reaching unsupported nodes here indicates a pipeline invariant violation.
  std::ostringstream os;
  os << "CodeGenTileLangSunMMIO unsupported stmt: " << op->GetTypeKey();
  if (!detail.empty()) {
    os << " (" << detail << ")";
  }
  LOG(FATAL) << os.str();
  TVM_FFI_UNREACHABLE();
}

[[noreturn]] void
CodeGenTileLangSunMMIO::UnsupportedExpr(const Object *op,
                                        const std::string &detail) const {
  // Generic SunMMIO codegen intentionally rejects pre-lowered structural forms.
  // Reaching unsupported nodes here indicates a pipeline invariant violation.
  std::ostringstream os;
  os << "CodeGenTileLangSunMMIO unsupported expr: " << op->GetTypeKey();
  if (!detail.empty()) {
    os << " (" << detail << ")";
  }
  LOG(FATAL) << os.str();
  TVM_FFI_UNREACHABLE();
}
} // namespace codegen
} // namespace tvm
