/*!
 * \file tileview/tileview_planner.cc
 * \brief TileView planning helpers for T.Tiles scopes.
 */

#include "tileview_planner.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>

#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../transform/common/attr.h"
#include "tileview_planner_common.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

struct IndexBinding {
  bool uses_loop_var{false};
  int domain_axis{-1};
  PrimExpr offset{Integer(0)};
};

struct AccessTileCandidate {
  TileView tileview;
  std::vector<int> tiled_domain_axes;
  std::unordered_set<int> non_tiled_domain_axes;
  std::vector<int> tile_shape;
};

struct AccessInfo {
  Buffer buffer;
  Array<PrimExpr> indices;
  std::unordered_set<int> used_domain_axes;
  std::vector<AccessTileCandidate> strict_candidates;
  std::vector<AccessTileCandidate> relaxed_candidates;
};

struct ExecutionPlanCandidate {
  std::vector<int> execution_domain_axes;
  std::vector<int> tile_shape;
};

class BufferAccessCollector : public StmtExprVisitor {
public:
  static std::vector<BufferAccessRecord> Collect(const Stmt &stmt) {
    BufferAccessCollector collector;
    collector(stmt);
    return std::move(collector.accesses_);
  }

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    if (op->buffer.scope() == "local.var") {
      StmtExprVisitor::VisitExpr_(op);
      return;
    }
    accesses_.push_back({op->buffer, op->indices, /*is_store=*/false});
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (op->buffer.scope() == "local.var") {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    accesses_.push_back({op->buffer, op->indices, /*is_store=*/true});
    StmtExprVisitor::VisitStmt_(op);
  }

  std::vector<BufferAccessRecord> accesses_;
};

std::optional<IndexBinding>
AnalyzeIndexBinding(const PrimExpr &index, const Array<Var> &loop_vars,
                    const std::vector<int> &loop_var_domain_axes,
                    const std::unordered_set<const VarNode *> &loop_var_nodes,
                    arith::Analyzer *analyzer) {
  ICHECK_EQ(loop_vars.size(), loop_var_domain_axes.size());
  if (!UsesVar(index, [&loop_var_nodes](const VarNode *node) {
        return loop_var_nodes.count(node) != 0;
      })) {
    return IndexBinding{false, -1, analyzer->Simplify(index)};
  }

  Array<PrimExpr> coeffs = arith::DetectLinearEquation(index, loop_vars);
  if (coeffs.empty()) {
    return std::nullopt;
  }

  int matched_axis = -1;
  PrimExpr base = analyzer->Simplify(coeffs[coeffs.size() - 1]);
  for (int i = 0; i < static_cast<int>(loop_vars.size()); ++i) {
    PrimExpr coeff = analyzer->Simplify(coeffs[i]);
    PrimExpr zero = make_zero(coeff.dtype());
    PrimExpr one = make_const(coeff.dtype(), 1);
    if (analyzer->CanProve(coeff == zero)) {
      continue;
    }
    if (!analyzer->CanProve(coeff == one) || matched_axis != -1) {
      return std::nullopt;
    }
    matched_axis = loop_var_domain_axes[i];
  }

  if (matched_axis < 0) {
    return std::nullopt;
  }
  return IndexBinding{true, matched_axis, base};
}

std::unordered_set<int>
CollectNonTiledDomainAxes(const std::vector<IndexBinding> &bindings,
                          const std::vector<int> &tiled_dims) {
  std::unordered_set<int> tiled_dim_set(tiled_dims.begin(), tiled_dims.end());
  std::unordered_set<int> non_tiled_axes;
  for (int dim = 0; dim < static_cast<int>(bindings.size()); ++dim) {
    if (tiled_dim_set.count(dim) != 0) {
      continue;
    }
    const IndexBinding &binding = bindings[dim];
    if (binding.uses_loop_var) {
      non_tiled_axes.insert(binding.domain_axis);
    }
  }
  return non_tiled_axes;
}

bool SameIntVector(const std::vector<int> &lhs, const std::vector<int> &rhs) {
  return lhs == rhs;
}

void AddRank1Candidate(std::vector<AccessTileCandidate> *candidates,
                       const Buffer &buffer, const Array<PrimExpr> &indices,
                       const std::vector<IndexBinding> &bindings,
                       int mapped_dim, int tile_width,
                       arith::Analyzer *analyzer, bool strict_checks,
                       TileView tv = TileView()) {
  const IndexBinding &binding = bindings[mapped_dim];
  if (!binding.uses_loop_var) {
    ICHECK(!strict_checks)
        << "1D TileView inside T.Tiles must bind to a tile loop var for buffer "
        << buffer->name << ".";
    return;
  }

  if (strict_checks) {
    RequireDivisible(analyzer, binding.offset, tile_width, indices[mapped_dim],
                     buffer);
  } else if (!CanProveDivisible(analyzer, binding.offset, tile_width)) {
    return;
  }

  if (!tv.defined()) {
    tv = makeTileView(
        buffer->shape, {Integer(tile_width)},
        {Integer(mapped_dim - static_cast<int>(buffer->shape.size()))});
  }

  candidates->push_back({tv,
                         {binding.domain_axis},
                         CollectNonTiledDomainAxes(bindings, {mapped_dim}),
                         {tile_width}});
}

void AddRank2Candidate(std::vector<AccessTileCandidate> *candidates,
                       const Buffer &buffer, const Array<PrimExpr> &indices,
                       const std::vector<IndexBinding> &bindings,
                       int mapped_height_dim, int mapped_width_dim,
                       int tile_height, int tile_width,
                       arith::Analyzer *analyzer, bool strict_checks,
                       TileView tv = TileView()) {
  const IndexBinding &height_binding = bindings[mapped_height_dim];
  const IndexBinding &width_binding = bindings[mapped_width_dim];

  if (!height_binding.uses_loop_var || !width_binding.uses_loop_var) {
    ICHECK(!strict_checks) << "Tiled buffer dimensions of buffer "
                           << buffer->name
                           << " must bind to tile loop vars inside T.Tiles.";
    return;
  }
  if (height_binding.domain_axis == width_binding.domain_axis) {
    ICHECK(!strict_checks) << "The tiled dimensions of buffer " << buffer->name
                           << " cannot bind to the same tile loop var.";
    return;
  }

  if (strict_checks) {
    RequireDivisible(analyzer, width_binding.offset, tile_width,
                     indices[mapped_width_dim], buffer);
  } else if (!CanProveDivisible(analyzer, width_binding.offset, tile_width)) {
    return;
  }

  if (strict_checks) {
    RequireDivisible(analyzer, height_binding.offset, tile_height,
                     indices[mapped_height_dim], buffer);
  } else if (!CanProveDivisible(analyzer, height_binding.offset, tile_height)) {
    return;
  }

  if (!tv.defined()) {
    int ndim = static_cast<int>(buffer->shape.size());
    tv = makeTileView(
        buffer->shape, {Integer(tile_height), Integer(tile_width)},
        {Integer(mapped_height_dim - ndim), Integer(mapped_width_dim - ndim)});
  }

  candidates->push_back(
      {tv,
       {height_binding.domain_axis, width_binding.domain_axis},
       CollectNonTiledDomainAxes(bindings,
                                 {mapped_height_dim, mapped_width_dim}),
       {tile_height, tile_width}});
}

void AddCandidateFromPattern(std::vector<AccessTileCandidate> *candidates,
                             const Buffer &buffer,
                             const Array<PrimExpr> &indices,
                             const std::vector<IndexBinding> &bindings,
                             const TrailingTilePattern &pattern,
                             arith::Analyzer *analyzer, bool strict_checks) {
  TileView tv = MakeTrailingTileView(buffer->shape, pattern);
  if (pattern.tile_shape.size() == 1) {
    AddRank1Candidate(candidates, buffer, indices, bindings,
                      pattern.mapped_dims[0], pattern.tile_shape[0], analyzer,
                      strict_checks, tv);
    return;
  }

  ICHECK_EQ(pattern.tile_shape.size(), 2)
      << "T.Tiles currently supports only 1D or 2D trailing TileView patterns.";
  AddRank2Candidate(candidates, buffer, indices, bindings,
                    pattern.mapped_dims[0], pattern.mapped_dims[1],
                    pattern.tile_shape[0], pattern.tile_shape[1], analyzer,
                    strict_checks, tv);
}

std::vector<AccessTileCandidate> EnumerateManualCandidates(
    const Buffer &buffer, const Array<PrimExpr> &indices,
    const std::vector<IndexBinding> &bindings, const TileView &manual_tv,
    int exec_rank, const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer) {
  std::vector<AccessTileCandidate> candidates;
  TrailingTilePattern pattern = ValidateManualTrailingTileView(
      buffer, manual_tv, exec_rank, layout_map, config, analyzer,
      {TileExtentPolicy::kTilesExecutionLayoutBounded, AlignmentMode::kStrict},
      "TileView inside T.Tiles");
  AddCandidateFromPattern(&candidates, buffer, indices, bindings, pattern,
                          analyzer, /*strict_checks=*/true);
  return candidates;
}

std::vector<AccessTileCandidate> EnumerateInferredCandidates(
    const Buffer &buffer, const Array<PrimExpr> &indices,
    const std::vector<IndexBinding> &bindings, int exec_rank,
    const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer,
    AlignmentMode alignment_mode = AlignmentMode::kStrict) {
  std::vector<AccessTileCandidate> candidates;
  for (const TrailingTilePattern &pattern :
       EnumerateInferredTrailingTilePatterns(
           buffer, exec_rank, layout_map, config, analyzer,
           {TileExtentPolicy::kTilesExecutionLayoutBounded, alignment_mode})) {
    AddCandidateFromPattern(&candidates, buffer, indices, bindings, pattern,
                            analyzer, /*strict_checks=*/false);
  }
  return candidates;
}

std::vector<AccessTileCandidate> EnumerateAccessTileCandidates(
    const BufferAccessRecord &access, const std::vector<IndexBinding> &bindings,
    int exec_rank, const TileViewMap &manual_tileviews,
    const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer,
    AlignmentMode alignment_mode = AlignmentMode::kStrict) {
  auto manual_it = manual_tileviews.find(access.buffer->data);
  if (manual_it != manual_tileviews.end()) {
    return EnumerateManualCandidates(access.buffer, access.indices, bindings,
                                     manual_it->second, exec_rank, layout_map,
                                     config, analyzer);
  }

  return EnumerateInferredCandidates(access.buffer, access.indices, bindings,
                                     exec_rank, layout_map, config, analyzer,
                                     alignment_mode);
}

bool IsEligibleForAligned1DBridgeSearch(
    const BufferAccessRecord &access, const std::vector<IndexBinding> &bindings,
    int exec_rank) {
  if (access.indices.size() != bindings.size()) {
    return false;
  }

  int active_dims = 0;
  for (const IndexBinding &binding : bindings) {
    if (binding.uses_loop_var) {
      ++active_dims;
    }
  }

  // A rank-1 execution store with one unit-stride binding is injective in its
  // only active tile axis.  Stores in a higher-rank execution need additional
  // dependence analysis before carrier read-modify-write can be made safe.
  if (access.is_store && exec_rank != 1) {
    return false;
  }
  return active_dims == 1;
}

bool RequiresAligned1DBridgeCandidate(
    const AccessTileCandidate &candidate, const Buffer &buffer,
    const SunmmioTileProcessorConfig &config) {
  if (candidate.tile_shape.size() != 1 ||
      buffer.scope() != kSunmmioScopeRSRAM) {
    return false;
  }

  int tile_extent = candidate.tile_shape[0];
  int align_elems =
      GetSunmmioRsramAlignmentElems(config.rsram_align_bytes, buffer->dtype);
  return tile_extent > 0 && tile_extent < align_elems &&
         align_elems % tile_extent == 0;
}

bool SupportsAligned1DBridgeDType(DataType dtype) {
  return dtype.bits() >= 8 && !sunmmio::IsMXDType(dtype);
}

bool SupportsCrossCarrierAligned1DBridgeDType(DataType dtype) {
  return dtype.is_bfloat16() || (dtype.is_float() && dtype.bits() == 32);
}

bool SupportsCrossCarrierAligned1DBridgeCandidate(
    const AccessTileCandidate &candidate, const Buffer &buffer,
    const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config) {
  if (candidate.tile_shape.size() != 1 ||
      buffer.scope() != kSunmmioScopeRSRAM ||
      !SupportsCrossCarrierAligned1DBridgeDType(buffer->dtype)) {
    return false;
  }

  int tile_extent = candidate.tile_shape[0];
  int align_elems =
      GetSunmmioRsramAlignmentElems(config.rsram_align_bytes, buffer->dtype);
  if (tile_extent <= 0 || tile_extent >= align_elems ||
      align_elems % tile_extent == 0) {
    return false;
  }

  auto layout_it = layout_map.find(buffer);
  if (layout_it == layout_map.end()) {
    return false;
  }
  const auto *cute = (*layout_it).second.as<CuteLayoutNode>();
  if (cute == nullptr) {
    return false;
  }

  int buffer_rank = static_cast<int>(buffer->shape.size());
  if (buffer_rank <= 0) {
    return false;
  }
  Array<Integer> dim_levels = cute->GetDimLevels();
  if (static_cast<int>(dim_levels.size()) != buffer_rank ||
      !std::all_of(
          dim_levels.begin(), dim_levels.end(),
          [](const Integer &levels) { return levels.IntValue() == 1; })) {
    return false;
  }

  Array<PrimExpr> index_map = candidate.tileview->IndexMap();
  if (index_map.size() != 1 ||
      NormalizeMappedDim(index_map[0], buffer_rank) != buffer_rank - 1) {
    return false;
  }

  Array<PrimExpr> covered_shape = cute->GetCoveredShape();
  if (static_cast<int>(covered_shape.size()) != buffer_rank) {
    return false;
  }
  int64_t covered_width = GetStaticIntValue(covered_shape[buffer_rank - 1]);
  if (covered_width <= 0 || covered_width % align_elems != 0) {
    return false;
  }

  for (int dim = 0; dim < buffer_rank; ++dim) {
    Array<PrimExpr> strides = cute->GetModeStrideOfDim(dim);
    if (strides.size() != 1) {
      return false;
    }
    int64_t stride = GetStaticIntValue(strides[0]);
    if (stride <= 0 ||
        (dim == buffer_rank - 1 ? stride != 1 : stride % align_elems != 0)) {
      return false;
    }
  }
  return true;
}

bool SupportsAligned1DBridgeCandidate(
    const AccessTileCandidate &candidate, const Buffer &buffer,
    const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config) {
  if (candidate.tile_shape.size() != 1 ||
      buffer.scope() != kSunmmioScopeRSRAM) {
    return true;
  }

  int tile_extent = candidate.tile_shape[0];
  int align_elems =
      GetSunmmioRsramAlignmentElems(config.rsram_align_bytes, buffer->dtype);
  if (tile_extent > 0 && tile_extent % align_elems == 0) {
    return true;
  }
  if (tile_extent <= 0 || tile_extent >= align_elems ||
      align_elems % tile_extent != 0) {
    return false;
  }
  if (!SupportsAligned1DBridgeDType(buffer->dtype)) {
    return false;
  }

  auto layout_it = layout_map.find(buffer);
  if (layout_it == layout_map.end()) {
    return true;
  }
  const auto *cute = (*layout_it).second.as<CuteLayoutNode>();
  if (cute == nullptr) {
    return false;
  }

  Array<Integer> dim_levels = cute->GetDimLevels();
  bool is_flat_layout =
      std::all_of(dim_levels.begin(), dim_levels.end(),
                  [](const Integer &levels) { return levels.IntValue() == 1; });
  if (is_flat_layout) {
    return true;
  }

  int buffer_rank = static_cast<int>(buffer->shape.size());
  Array<PrimExpr> index_map = candidate.tileview->IndexMap();
  if (index_map.size() != 1) {
    return false;
  }
  int tiled_dim = NormalizeMappedDim(index_map[0], buffer_rank);

  for (int dim = 0; dim < buffer_rank; ++dim) {
    Array<PrimExpr> mode_shapes = cute->GetModeShapeOfDim(dim);
    Array<PrimExpr> mode_strides = cute->GetModeStrideOfDim(dim);
    if (mode_shapes.empty() || mode_shapes.size() != mode_strides.size()) {
      return false;
    }
    for (int mode = 0; mode < static_cast<int>(mode_shapes.size()); ++mode) {
      int64_t stride = GetStaticIntValue(mode_strides[mode]);
      if (stride <= 0) {
        return false;
      }
      if (dim == tiled_dim && mode == 0) {
        int64_t inner_extent = GetStaticIntValue(mode_shapes[mode]);
        if (stride != 1 || inner_extent <= 0 ||
            inner_extent % align_elems != 0) {
          return false;
        }
      } else if (stride % align_elems != 0) {
        return false;
      }
    }
  }
  return true;
}

std::optional<std::vector<int>>
GetSmall2DCarrierShape(const Buffer &buffer,
                       const std::vector<int> &logical_shape,
                       const Map<Buffer, Layout> &layout_map,
                       const SunmmioTileProcessorConfig &config) {
  if (logical_shape.size() != 2 || logical_shape[0] <= 1 ||
      logical_shape[1] <= 1 || buffer.scope() != kSunmmioScopeRSRAM ||
      !layout_map.count(buffer)) {
    return std::nullopt;
  }
  DataType dtype = buffer->dtype;
  if (!dtype.is_bfloat16() && !(dtype.is_float() && dtype.bits() == 32)) {
    return std::nullopt;
  }
  const auto *cute = layout_map[buffer].as<CuteLayoutNode>();
  if (!cute || cute->GetDimLevels().size() != 2) {
    return std::nullopt;
  }

  int capacity_elems = GetCapacityElems(buffer, config);
  int carrier_height = 0;
  int carrier_width = 0;
  Array<Integer> dim_levels = cute->GetDimLevels();
  if (dim_levels[0].IntValue() == 1 && dim_levels[1].IntValue() == 1) {
    Array<PrimExpr> covered_shape = cute->GetCoveredShape();
    Array<PrimExpr> height_strides = cute->GetModeStrideOfDim(0);
    Array<PrimExpr> width_strides = cute->GetModeStrideOfDim(1);
    if (covered_shape.size() != 2 || height_strides.size() != 1 ||
        width_strides.size() != 1) {
      return std::nullopt;
    }
    carrier_width = GetStaticIntValue(covered_shape[1]);
    int covered_height = GetStaticIntValue(covered_shape[0]);
    if (carrier_width <= 0 || covered_height <= 0 ||
        GetStaticIntValue(width_strides[0]) != 1 ||
        GetStaticIntValue(height_strides[0]) != carrier_width ||
        capacity_elems % carrier_width != 0) {
      return std::nullopt;
    }
    carrier_height = capacity_elems / carrier_width;
    if (carrier_height > covered_height) {
      return std::nullopt;
    }
  } else if (auto block = sunmmio::GetZZBlockShape(layout_map[buffer])) {
    carrier_width = block->width;
    if (carrier_width <= 0 || capacity_elems < carrier_width) {
      return std::nullopt;
    }
    carrier_height = std::min(block->height, capacity_elems / carrier_width);
  } else {
    return std::nullopt;
  }

  if (logical_shape[0] > carrier_height || logical_shape[1] > carrier_width ||
      carrier_height % logical_shape[0] != 0 ||
      carrier_width % logical_shape[1] != 0) {
    return std::nullopt;
  }
  return std::vector<int>{carrier_height, carrier_width};
}

std::optional<TileViewPlan> TryPlanDomainSizedSmall2D(
    const Array<PrimExpr> &domain, const Array<Var> &loop_vars,
    const std::vector<BufferAccessRecord> &accesses,
    const TileViewMap &manual_tileviews, const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer) {
  if (domain.size() != 2 || loop_vars.size() != 2) {
    return std::nullopt;
  }
  int tile_height = GetStaticIntValue(domain[0]);
  int tile_width = GetStaticIntValue(domain[1]);
  if (tile_height <= 1 || tile_width <= 1) {
    return std::nullopt;
  }

  std::unordered_set<const VarNode *> loop_var_nodes{loop_vars[0].get(),
                                                     loop_vars[1].get()};
  bool saw_rank2_carrier = false;
  for (const BufferAccessRecord &access : accesses) {
    if (manual_tileviews.count(access.buffer->data)) {
      return std::nullopt;
    }
    std::vector<IndexBinding> bindings;
    bindings.reserve(access.indices.size());
    int active_dims = 0;
    for (const PrimExpr &index : access.indices) {
      auto binding = AnalyzeIndexBinding(index, loop_vars, {0, 1},
                                         loop_var_nodes, analyzer);
      if (!binding.has_value()) {
        return std::nullopt;
      }
      active_dims += binding->uses_loop_var ? 1 : 0;
      bindings.push_back(binding.value());
    }
    if (active_dims == 0) {
      continue;
    }
    if (active_dims == 1) {
      if (access.is_store) {
        return std::nullopt;
      }
      int mapped_dim = -1;
      for (int dim = 0; dim < static_cast<int>(bindings.size()); ++dim) {
        if (bindings[dim].uses_loop_var) {
          mapped_dim = dim;
          break;
        }
      }
      int domain_axis = bindings[mapped_dim].domain_axis;
      int tile_extent = domain_axis == 0 ? tile_height : tile_width;
      std::vector<AccessTileCandidate> candidates;
      AddRank1Candidate(&candidates, access.buffer, access.indices, bindings,
                        mapped_dim, tile_extent, analyzer,
                        /*strict_checks=*/false);
      if (candidates.empty() ||
          !SupportsAligned1DBridgeCandidate(candidates.front(), access.buffer,
                                            layout_map, config)) {
        return std::nullopt;
      }
      continue;
    }
    int rank = static_cast<int>(bindings.size());
    if (active_dims != 2 || rank < 2 || !bindings[rank - 2].uses_loop_var ||
        !bindings[rank - 1].uses_loop_var ||
        bindings[rank - 2].domain_axis != 0 ||
        bindings[rank - 1].domain_axis != 1 ||
        !CanProveDivisible(analyzer, bindings[rank - 2].offset, tile_height) ||
        !CanProveDivisible(analyzer, bindings[rank - 1].offset, tile_width) ||
        !GetSmall2DCarrierShape(access.buffer, {tile_height, tile_width},
                                layout_map, config)
             .has_value()) {
      return std::nullopt;
    }
    saw_rank2_carrier = true;
  }
  if (!saw_rank2_carrier) {
    return std::nullopt;
  }

  return TileViewPlan{makeTileView(domain,
                                   {Integer(tile_height), Integer(tile_width)},
                                   {Integer(-2), Integer(-1)}),
                      {0, 1},
                      false,
                      true};
}

std::optional<std::vector<AccessInfo>> AnalyzeAccessesForExecutionAxes(
    const std::vector<BufferAccessRecord> &accesses,
    const Array<Var> &execution_loop_vars,
    const std::vector<int> &execution_domain_axes, int exec_rank,
    const TileViewMap &manual_tileviews, const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer,
    bool rank_reduction_search, std::optional<int> cross_carrier_domain_extent,
    std::unordered_set<const BufferNode *> *warned_unsupported_bridge_buffers) {
  ICHECK_EQ(execution_loop_vars.size(), execution_domain_axes.size());
  ICHECK(warned_unsupported_bridge_buffers != nullptr);

  std::unordered_set<const VarNode *> execution_loop_var_nodes;
  for (const Var &loop_var : execution_loop_vars) {
    execution_loop_var_nodes.insert(loop_var.get());
  }

  std::vector<AccessInfo> analyzed_accesses;
  analyzed_accesses.reserve(accesses.size());
  for (const auto &access : accesses) {
    auto manual_it = manual_tileviews.find(access.buffer->data);
    // An explicit higher-rank TileView must not be silently replaced by a
    // lower-rank inferred plan.
    if (rank_reduction_search && manual_it != manual_tileviews.end() &&
        manual_it->second->TileDim() != 1) {
      return std::nullopt;
    }

    std::vector<IndexBinding> bindings;
    std::unordered_set<int> used_domain_axes;
    bindings.reserve(access.indices.size());
    for (const PrimExpr &index : access.indices) {
      auto binding =
          AnalyzeIndexBinding(index, execution_loop_vars, execution_domain_axes,
                              execution_loop_var_nodes, analyzer);
      if (!binding.has_value()) {
        return std::nullopt;
      }
      bindings.push_back(binding.value());
      if (binding->uses_loop_var) {
        used_domain_axes.insert(binding->domain_axis);
      }
    }

    std::vector<AccessTileCandidate> candidates = EnumerateAccessTileCandidates(
        access, bindings, exec_rank, manual_tileviews, layout_map, config,
        analyzer);
    std::vector<AccessTileCandidate> relaxed_candidates = candidates;
    if (IsEligibleForAligned1DBridgeSearch(access, bindings, exec_rank)) {
      relaxed_candidates = EnumerateAccessTileCandidates(
          access, bindings, exec_rank, manual_tileviews, layout_map, config,
          analyzer, AlignmentMode::kRelaxed);
      if (cross_carrier_domain_extent.has_value() &&
          manual_it == manual_tileviews.end()) {
        int mapped_dim = -1;
        for (int dim = 0; dim < static_cast<int>(bindings.size()); ++dim) {
          if (bindings[dim].uses_loop_var) {
            mapped_dim = dim;
            break;
          }
        }
        ICHECK_GE(mapped_dim, 0);
        std::vector<AccessTileCandidate> complete_domain_candidates;
        AddRank1Candidate(&complete_domain_candidates, access.buffer,
                          access.indices, bindings, mapped_dim,
                          *cross_carrier_domain_extent, analyzer,
                          /*strict_checks=*/false);
        for (AccessTileCandidate &candidate : complete_domain_candidates) {
          if (!SupportsCrossCarrierAligned1DBridgeCandidate(
                  candidate, access.buffer, layout_map, config)) {
            continue;
          }
          bool duplicate = std::any_of(
              relaxed_candidates.begin(), relaxed_candidates.end(),
              [&](const AccessTileCandidate &existing) {
                return existing.tile_shape == candidate.tile_shape &&
                       existing.tiled_domain_axes ==
                           candidate.tiled_domain_axes;
              });
          if (!duplicate) {
            relaxed_candidates.push_back(std::move(candidate));
          }
        }
      }
      bool has_unsupported_bridge =
          !SupportsAligned1DBridgeDType(access.buffer->dtype) &&
          std::any_of(relaxed_candidates.begin(), relaxed_candidates.end(),
                      [&](const AccessTileCandidate &candidate) {
                        return RequiresAligned1DBridgeCandidate(
                            candidate, access.buffer, config);
                      });
      if (has_unsupported_bridge &&
          warned_unsupported_bridge_buffers->insert(access.buffer.get())
              .second) {
        LOG(WARNING)
            << "Skipping aligned 1D bridge candidates for buffer "
            << access.buffer->name << " with dtype " << access.buffer->dtype
            << ". Sub-byte and MX dtypes are currently supported only by "
               "their blockwise paths; T.Tiles will try a strict plan or "
               "scalar fallback.";
      }
      relaxed_candidates.erase(
          std::remove_if(
              relaxed_candidates.begin(), relaxed_candidates.end(),
              [&](const AccessTileCandidate &candidate) {
                return !SupportsAligned1DBridgeCandidate(
                           candidate, access.buffer, layout_map, config) &&
                       !(cross_carrier_domain_extent.has_value() &&
                         SupportsCrossCarrierAligned1DBridgeCandidate(
                             candidate, access.buffer, layout_map, config));
              }),
          relaxed_candidates.end());
    }

    analyzed_accesses.push_back(
        {access.buffer, access.indices, std::move(used_domain_axes),
         std::move(candidates), std::move(relaxed_candidates)});
  }
  return analyzed_accesses;
}

const std::vector<AccessTileCandidate> &
CandidatesForSearch(const AccessInfo &access, AlignmentMode mode) {
  return mode == AlignmentMode::kRelaxed ? access.relaxed_candidates
                                         : access.strict_candidates;
}

bool UsesExecutionAxisInNonTiledDims(const AccessTileCandidate &candidate,
                                     const std::vector<int> &exec_domain_axes) {
  for (int exec_domain_axis : exec_domain_axes) {
    if (candidate.non_tiled_domain_axes.count(exec_domain_axis) != 0) {
      return true;
    }
  }
  return false;
}

bool Supports1DPlan(const AccessTileCandidate &candidate, int domain_axis) {
  return candidate.tiled_domain_axes.size() == 1 &&
         candidate.tile_shape.size() == 1 &&
         candidate.tiled_domain_axes[0] == domain_axis &&
         !UsesExecutionAxisInNonTiledDims(candidate, {domain_axis});
}

bool Supports1DPlan(const AccessTileCandidate &candidate, int domain_axis,
                    int tile_extent) {
  return Supports1DPlan(candidate, domain_axis) &&
         candidate.tile_shape[0] == tile_extent;
}

bool Supports2DPlan(const AccessTileCandidate &candidate,
                    const ExecutionPlanCandidate &plan) {
  if (UsesExecutionAxisInNonTiledDims(candidate, plan.execution_domain_axes)) {
    return false;
  }

  if (candidate.tile_shape.size() == 2) {
    return candidate.tiled_domain_axes.size() == 2 &&
           candidate.tiled_domain_axes[0] == plan.execution_domain_axes[0] &&
           candidate.tiled_domain_axes[1] == plan.execution_domain_axes[1] &&
           SameIntVector(candidate.tile_shape, plan.tile_shape);
  }

  if (candidate.tile_shape.size() != 1 ||
      candidate.tiled_domain_axes.size() != 1) {
    return false;
  }

  int bound_axis = candidate.tiled_domain_axes[0];
  if (bound_axis == plan.execution_domain_axes[0]) {
    return candidate.tile_shape[0] == plan.tile_shape[0];
  }
  if (bound_axis == plan.execution_domain_axes[1]) {
    return candidate.tile_shape[0] == plan.tile_shape[1];
  }
  return false;
}

std::vector<ExecutionPlanCandidate>
CollectRank2PlanCandidates(const std::vector<AccessTileCandidate> &candidates) {
  std::vector<ExecutionPlanCandidate> plans;
  for (const auto &candidate : candidates) {
    if (candidate.tile_shape.size() != 2) {
      continue;
    }
    bool exists = std::any_of(
        plans.begin(), plans.end(), [&](const ExecutionPlanCandidate &plan) {
          return SameIntVector(plan.execution_domain_axes,
                               candidate.tiled_domain_axes) &&
                 SameIntVector(plan.tile_shape, candidate.tile_shape);
        });
    if (!exists) {
      plans.push_back({candidate.tiled_domain_axes, candidate.tile_shape});
    }
  }
  return plans;
}

std::optional<TileViewPlan>
TrySelectPlan(const Array<PrimExpr> &domain,
              const std::vector<AccessInfo> &accesses, int domain_rank,
              int exec_rank, arith::Analyzer *analyzer, AlignmentMode mode,
              bool fail_on_error, int rank1_domain_axis = 0) {
  for (const auto &access : accesses) {
    if (exec_rank == 1 &&
        access.used_domain_axes.count(rank1_domain_axis) == 0) {
      continue;
    }
    if (CandidatesForSearch(access, mode).empty()) {
      if (fail_on_error) {
        LOG(FATAL)
            << "Cannot infer any feasible TileView candidate for access to "
               "buffer "
            << access.buffer->name << " with indices " << access.indices << ".";
      }
      return std::nullopt;
    }
  }

  if (exec_rank == 1) {
    std::vector<int> plan_extents;
    for (const auto &access : accesses) {
      if (access.used_domain_axes.count(rank1_domain_axis) == 0) {
        continue;
      }
      for (const auto &candidate : CandidatesForSearch(access, mode)) {
        if (Supports1DPlan(candidate, rank1_domain_axis)) {
          plan_extents.push_back(candidate.tile_shape[0]);
        }
      }
    }

    if (plan_extents.empty()) {
      if (fail_on_error) {
        LOG(FATAL) << "Cannot infer any feasible 1D execution tileview for "
                      "T.Tiles domain "
                   << domain << ".";
      }
      return std::nullopt;
    }

    std::sort(plan_extents.begin(), plan_extents.end());
    plan_extents.erase(std::unique(plan_extents.begin(), plan_extents.end()),
                       plan_extents.end());
    std::sort(plan_extents.begin(), plan_extents.end(), std::greater<int>());

    for (int tile_extent : plan_extents) {
      bool all_supported = true;
      for (const auto &access : accesses) {
        if (access.used_domain_axes.count(rank1_domain_axis) == 0) {
          continue;
        }
        const auto &candidates = CandidatesForSearch(access, mode);
        bool access_supported = std::any_of(
            candidates.begin(), candidates.end(),
            [&](const AccessTileCandidate &candidate) {
              return Supports1DPlan(candidate, rank1_domain_axis, tile_extent);
            });
        if (!access_supported) {
          all_supported = false;
          break;
        }
      }

      if (all_supported) {
        return TileViewPlan{
            makeTileView(domain, {Integer(tile_extent)},
                         {Integer(rank1_domain_axis - domain_rank)}),
            {rank1_domain_axis},
            mode == AlignmentMode::kRelaxed};
      }
    }

    if (fail_on_error) {
      LOG(FATAL) << "Cannot infer a common 1D execution tileview for T.Tiles "
                 << "domain " << domain
                 << ". The per-access feasible tileview sets do not intersect.";
    }
    return std::nullopt;
  }

  std::vector<ExecutionPlanCandidate> plan_candidates;
  bool saw_rank2_candidates = false;
  for (const auto &access : accesses) {
    std::vector<ExecutionPlanCandidate> access_plan_candidates =
        CollectRank2PlanCandidates(CandidatesForSearch(access, mode));
    if (access_plan_candidates.empty()) {
      continue;
    }

    if (!saw_rank2_candidates) {
      plan_candidates = std::move(access_plan_candidates);
      saw_rank2_candidates = true;
      continue;
    }

    std::vector<ExecutionPlanCandidate> common_plan_candidates;
    for (const auto &plan_candidate : plan_candidates) {
      bool supported = std::any_of(
          access_plan_candidates.begin(), access_plan_candidates.end(),
          [&](const ExecutionPlanCandidate &access_plan_candidate) {
            return SameIntVector(plan_candidate.execution_domain_axes,
                                 access_plan_candidate.execution_domain_axes) &&
                   SameIntVector(plan_candidate.tile_shape,
                                 access_plan_candidate.tile_shape);
          });
      if (supported) {
        common_plan_candidates.push_back(plan_candidate);
      }
    }
    plan_candidates = std::move(common_plan_candidates);
  }

  if (!saw_rank2_candidates) {
    if (fail_on_error) {
      LOG(FATAL)
          << "2D T.Tiles requires at least one access with a feasible rank-2 "
             "TileView candidate.";
    }
    return std::nullopt;
  }

  if (plan_candidates.empty()) {
    if (fail_on_error) {
      LOG(FATAL)
          << "Cannot infer a common execution tileview for T.Tiles domain "
          << domain
          << ". The rank-2 access candidates do not share a common axis "
             "binding and tile shape.";
    }
    return std::nullopt;
  }

  std::sort(
      plan_candidates.begin(), plan_candidates.end(),
      [](const ExecutionPlanCandidate &lhs, const ExecutionPlanCandidate &rhs) {
        int64_t lhs_elems = TileElements(lhs.tile_shape);
        int64_t rhs_elems = TileElements(rhs.tile_shape);
        if (lhs_elems != rhs_elems) {
          return lhs_elems > rhs_elems;
        }
        if (lhs.tile_shape[0] != rhs.tile_shape[0]) {
          return lhs.tile_shape[0] > rhs.tile_shape[0];
        }
        if (lhs.tile_shape[1] != rhs.tile_shape[1]) {
          return lhs.tile_shape[1] > rhs.tile_shape[1];
        }
        return lhs.execution_domain_axes < rhs.execution_domain_axes;
      });

  for (const auto &plan_candidate : plan_candidates) {
    const auto &exec_domain_axes = plan_candidate.execution_domain_axes;
    int tile_height = plan_candidate.tile_shape[0];
    int tile_width = plan_candidate.tile_shape[1];

    bool all_supported = true;
    for (const auto &access : accesses) {
      const auto &candidates = CandidatesForSearch(access, mode);
      bool access_supported =
          std::any_of(candidates.begin(), candidates.end(),
                      [&](const AccessTileCandidate &candidate) {
                        return Supports2DPlan(candidate, plan_candidate);
                      });
      if (!access_supported) {
        all_supported = false;
        break;
      }
    }

    if (all_supported) {
      return TileViewPlan{
          makeTileView(domain, {Integer(tile_height), Integer(tile_width)},
                       {Integer(exec_domain_axes[0] - domain_rank),
                        Integer(exec_domain_axes[1] - domain_rank)}),
          exec_domain_axes, mode == AlignmentMode::kRelaxed};
    }
  }

  if (fail_on_error) {
    LOG(FATAL) << "Cannot infer a common execution tileview for T.Tiles domain "
               << domain
               << ". The per-access feasible tileview sets do not intersect.";
  }
  return std::nullopt;
}

} // namespace

std::vector<BufferAccessRecord> CollectBufferAccesses(const Stmt &stmt) {
  return BufferAccessCollector::Collect(stmt);
}

std::optional<TileViewPlan> TryPlanTileViewsForTilesScope(
    const Array<PrimExpr> &domain,
    const std::vector<const ForNode *> &scope_loops,
    const std::vector<BufferAccessRecord> &accesses,
    const TileViewMap &manual_tileviews, const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &tile_processor_config) {
  ICHECK(!domain.empty()) << "T.Tiles domain must be non-empty.";
  ICHECK_EQ(domain.size(), scope_loops.size())
      << "T.Tiles scope loop rank does not match the declared domain rank.";
  ICHECK(!accesses.empty()) << "T.Tiles scope must access at least one buffer.";

  int domain_rank = static_cast<int>(domain.size());
  int exec_rank = domain_rank == 1 ? 1 : 2;

  arith::Analyzer analyzer;
  std::unordered_set<const BufferNode *> warned_unsupported_bridge_buffers;

  Array<Var> loop_vars;
  std::vector<int> all_domain_axes;
  for (const ForNode *loop : scope_loops) {
    loop_vars.push_back(loop->loop_var);
    all_domain_axes.push_back(static_cast<int>(all_domain_axes.size()));
  }

  std::optional<int> cross_carrier_domain_extent;
  int64_t static_domain_extent = GetStaticIntValue(domain[0]);
  if (domain_rank == 1 && static_domain_extent > 0 &&
      static_domain_extent <= std::numeric_limits<int>::max()) {
    cross_carrier_domain_extent = static_cast<int>(static_domain_extent);
  }

  std::optional<TileViewPlan> small_2d_plan =
      TryPlanDomainSizedSmall2D(domain, loop_vars, accesses, manual_tileviews,
                                layout_map, tile_processor_config, &analyzer);
  auto choose_normal_or_small = [&](const TileViewPlan &normal_plan) {
    if (small_2d_plan.has_value() &&
        normal_plan.execution_tileview->TileDim() == 2 &&
        normal_plan.execution_domain_axes == std::vector<int>{0, 1} &&
        (GetStaticIntValue(normal_plan.execution_tileview->TileShape()[0]) >
             GetStaticIntValue(domain[0]) ||
         GetStaticIntValue(normal_plan.execution_tileview->TileShape()[1]) >
             GetStaticIntValue(domain[1]))) {
      return small_2d_plan.value();
    }
    return normal_plan;
  };

  auto full_rank_accesses = AnalyzeAccessesForExecutionAxes(
      accesses, loop_vars, all_domain_axes, exec_rank, manual_tileviews,
      layout_map, tile_processor_config, &analyzer,
      /*rank_reduction_search=*/false, cross_carrier_domain_extent,
      &warned_unsupported_bridge_buffers);
  if (full_rank_accesses.has_value()) {
    if (auto plan = TrySelectPlan(
            domain, full_rank_accesses.value(), domain_rank, exec_rank,
            &analyzer, AlignmentMode::kStrict, /*fail_on_error=*/false)) {
      return choose_normal_or_small(plan.value());
    }

    if (auto plan = TrySelectPlan(
            domain, full_rank_accesses.value(), domain_rank, exec_rank,
            &analyzer, AlignmentMode::kRelaxed, /*fail_on_error=*/false)) {
      return choose_normal_or_small(plan.value());
    }
  }

  if (small_2d_plan.has_value()) {
    return small_2d_plan;
  }

  if (domain_rank > 1) {
    for (int domain_axis = domain_rank - 1; domain_axis >= 0; --domain_axis) {
      Array<Var> rank1_loop_vars{loop_vars[domain_axis]};
      std::vector<int> rank1_domain_axes{domain_axis};
      auto rank1_accesses = AnalyzeAccessesForExecutionAxes(
          accesses, rank1_loop_vars, rank1_domain_axes, /*exec_rank=*/1,
          manual_tileviews, layout_map, tile_processor_config, &analyzer,
          /*rank_reduction_search=*/true,
          /*cross_carrier_domain_extent=*/std::nullopt,
          &warned_unsupported_bridge_buffers);
      if (!rank1_accesses.has_value()) {
        continue;
      }
      if (auto plan =
              TrySelectPlan(domain, rank1_accesses.value(), domain_rank,
                            /*exec_rank=*/1, &analyzer, AlignmentMode::kStrict,
                            /*fail_on_error=*/false, domain_axis)) {
        return plan;
      }
      if (auto plan =
              TrySelectPlan(domain, rank1_accesses.value(), domain_rank,
                            /*exec_rank=*/1, &analyzer, AlignmentMode::kRelaxed,
                            /*fail_on_error=*/false, domain_axis)) {
        return plan;
      }
    }
  }
  return std::nullopt;
}

} // namespace tl
} // namespace tvm
