/*!
 * \file tile_scheduler.h
 * \brief Internal tile-scheduler spec and binding utilities.
 */

#pragma once

#include <optional>

#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include "./attr.h"

namespace tvm {
namespace tl {
namespace tile_scheduler {

using namespace tir;

constexpr const char *kPersistentKernel = "tilelang.is_persistent_kernel";
constexpr const char *kTileScheduleKind = "tilelang.tile_schedule_kind";
constexpr const char *kTilePermutationKind = "tilelang.tile_permutation_kind";
constexpr const char *kTilePermutationPanelSize =
    "tilelang.tile_permutation_panel_size";
constexpr const char *kLegacyPersistentSwizzleOrder =
    "tilelang.persistent_swizzle_order";
constexpr const char *kLegacyPersistentSwizzlePanelSize =
    "tilelang.persistent_swizzle_panel_size";
constexpr const char *kLegacyThreadblockSwizzlePattern =
    "threadblock_swizzle_pattern";

enum class DistributionKind : uint8_t {
  kStatic,
  kPersistentStatic,
  kDynamicPersistent,
  kClusterLaunchControl,
};

enum class PermutationKind : uint8_t {
  kIdentity,
  kSwizzleRow,
  kSwizzleColumn,
};

struct TilePermutationSpec {
  PermutationKind kind{PermutationKind::kIdentity};
  int panel_size{0};

  bool IsIdentity() const { return kind == PermutationKind::kIdentity; }
};

struct TileScheduleSpec {
  DistributionKind distribution_kind{DistributionKind::kStatic};
  TilePermutationSpec permutation{};
  std::optional<ffi::Array<Integer>> cluster_dims{std::nullopt};
  int num_persistent_workers{0};

  bool RequiresScheduling() const {
    return distribution_kind != DistributionKind::kStatic;
  }
};

struct TileWorkBinding {
  PrimExpr worker_id;
  PrimExpr iteration_id;
  PrimExpr logical_tile_id;
  PrimExpr tile_x;
  PrimExpr tile_y;
  PrimExpr is_valid;
};

inline bool BlockRequestsTileSchedule(const Block &blk) {
  if (!blk->annotations.defined()) {
    return false;
  }
  if (auto kind = blk->annotations.Get(kTileScheduleKind)) {
    if (auto kind_str = kind->try_cast<ffi::String>()) {
      return kind_str.value() != "static";
    }
  }
  if (!blk->annotations.count(kPersistentKernel)) {
    return false;
  }
  auto ann = blk->annotations.Get(kPersistentKernel);
  if (!ann) {
    return false;
  }
  if (const auto *imm = ann.value().as<IntImmNode>()) {
    return imm->value != 0;
  }
  return true;
}

inline std::optional<TilePermutationSpec>
ParseLegacyThreadblockSwizzle(const PrimExpr &value) {
  const auto *call = value.as<CallNode>();
  if (!call || !call->op.same_as(tir::builtin::tvm_tuple()) ||
      call->args.size() < 2) {
    return std::nullopt;
  }

  const auto *name_node = call->args[0].as<StringImmNode>();
  const auto *size_node = call->args[1].as<IntImmNode>();
  if (!name_node || !size_node || size_node->value <= 0) {
    return std::nullopt;
  }

  TilePermutationSpec spec;
  if (name_node->value == "rasterization2DRow") {
    spec.kind = PermutationKind::kSwizzleRow;
  } else if (name_node->value == "rasterization2DColumn") {
    spec.kind = PermutationKind::kSwizzleColumn;
  } else {
    return std::nullopt;
  }
  spec.panel_size = static_cast<int>(size_node->value);
  return spec;
}

class StripLegacyThreadblockSwizzleAttr : public StmtExprMutator {
public:
  static std::pair<Stmt, std::optional<TilePermutationSpec>> Run(Stmt stmt) {
    StripLegacyThreadblockSwizzleAttr mutator;
    return {mutator(std::move(stmt)), mutator.permutation_};
  }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != kLegacyThreadblockSwizzlePattern) {
      return StmtExprMutator::VisitStmt_(op);
    }

    auto parsed = ParseLegacyThreadblockSwizzle(op->value);
    if (parsed.has_value()) {
      permutation_ = parsed;
    } else {
      LOG(WARNING) << "TileSchedule: ignoring unsupported "
                   << kLegacyThreadblockSwizzlePattern
                   << " on scheduled kernel; only row/column swizzles are "
                      "supported.";
    }
    return this->VisitStmt(op->body);
  }

  std::optional<TilePermutationSpec> permutation_;
};

inline std::optional<TileScheduleSpec>
ParseTileScheduleSpec(const Block &blk, int num_persistent_workers) {
  if (!blk->annotations.defined()) {
    return std::nullopt;
  }

  TileScheduleSpec spec;
  spec.num_persistent_workers = num_persistent_workers;

  if (auto kind = blk->annotations.Get(kTileScheduleKind)) {
    auto kind_str = kind->try_cast<ffi::String>();
    ICHECK(kind_str) << "tile schedule kind must be a string";
    if (kind_str.value() == "persistent_static") {
      spec.distribution_kind = DistributionKind::kPersistentStatic;
    } else if (kind_str.value() == "static") {
      spec.distribution_kind = DistributionKind::kStatic;
    } else if (kind_str.value() == "dynamic_persistent") {
      spec.distribution_kind = DistributionKind::kDynamicPersistent;
    } else if (kind_str.value() == "cluster_launch_control") {
      spec.distribution_kind = DistributionKind::kClusterLaunchControl;
    } else {
      LOG(FATAL) << "Unsupported tile schedule kind: " << kind_str.value();
    }
  } else if (BlockRequestsTileSchedule(blk)) {
    spec.distribution_kind = DistributionKind::kPersistentStatic;
  } else {
    return std::nullopt;
  }

  if (auto permutation = blk->annotations.Get(kTilePermutationKind)) {
    auto kind_str = permutation->try_cast<ffi::String>();
    ICHECK(kind_str) << "tile permutation kind must be a string";
    if (kind_str.value() == "identity") {
      spec.permutation.kind = PermutationKind::kIdentity;
    } else if (kind_str.value() == "swizzle_row") {
      spec.permutation.kind = PermutationKind::kSwizzleRow;
    } else if (kind_str.value() == "swizzle_column") {
      spec.permutation.kind = PermutationKind::kSwizzleColumn;
    } else {
      LOG(FATAL) << "Unsupported tile permutation kind: " << kind_str.value();
    }
  } else {
    auto order_it = blk->annotations.Get(kLegacyPersistentSwizzleOrder);
    auto panel_it = blk->annotations.Get(kLegacyPersistentSwizzlePanelSize);
    if (order_it || panel_it) {
      ICHECK(order_it && panel_it)
          << "PersistentKernel swizzle requires both order and panel_size";
      auto order_str = order_it->try_cast<ffi::String>();
      auto panel_int = panel_it->try_cast<Integer>();
      ICHECK(order_str && panel_int && panel_int.value()->value > 0)
          << "PersistentKernel swizzle expects (order, positive panel_size)";
      if (order_str.value() == "row") {
        spec.permutation.kind = PermutationKind::kSwizzleRow;
      } else if (order_str.value() == "column") {
        spec.permutation.kind = PermutationKind::kSwizzleColumn;
      } else {
        LOG(FATAL) << "Unsupported PersistentKernel swizzle order: "
                   << order_str.value();
      }
      spec.permutation.panel_size =
          static_cast<int>(panel_int.value()->value);
    }
  }

  if (!spec.permutation.IsIdentity()) {
    auto panel_it = blk->annotations.Get(kTilePermutationPanelSize);
    if (panel_it) {
      auto panel_int = panel_it->try_cast<Integer>();
      ICHECK(panel_int && panel_int.value()->value > 0)
          << "tile permutation panel size must be positive";
      spec.permutation.panel_size =
          static_cast<int>(panel_int.value()->value);
    }
    ICHECK_GT(spec.permutation.panel_size, 0)
        << "non-identity tile permutation requires positive panel_size";
  }

  if (auto cluster_dims = blk->annotations.Get("cluster_dims")) {
    if (auto arr = cluster_dims->try_cast<ffi::Array<Integer>>()) {
      spec.cluster_dims = arr.value();
    }
  }

  return spec;
}

inline TileScheduleSpec ResolveLegacyPermutation(
    TileScheduleSpec spec, const std::optional<TilePermutationSpec> &legacy,
    bool *emit_override_warning) {
  if (!legacy.has_value()) {
    return spec;
  }
  if (!spec.permutation.IsIdentity()) {
    if (emit_override_warning) {
      *emit_override_warning = true;
    }
    return spec;
  }
  spec.permutation = legacy.value();
  return spec;
}

inline std::pair<PrimExpr, PrimExpr>
ApplyTilePermutation(const TilePermutationSpec &spec,
                     const PrimExpr &logical_tile_id, const PrimExpr &grid_x,
                     const PrimExpr &grid_y) {
  if (spec.kind == PermutationKind::kIdentity) {
    return {FloorMod(logical_tile_id, grid_x), FloorDiv(logical_tile_id, grid_x)};
  }

  PrimExpr total_tiles = grid_x * grid_y;
  PrimExpr panel_imm = IntImm(logical_tile_id.dtype(), spec.panel_size);
  PrimExpr one = IntImm(logical_tile_id.dtype(), 1);
  PrimExpr two = IntImm(logical_tile_id.dtype(), 2);

  if (spec.kind == PermutationKind::kSwizzleRow) {
    PrimExpr panel_span = panel_imm * grid_x;
    PrimExpr panel_offset = FloorMod(logical_tile_id, panel_span);
    PrimExpr panel_idx = FloorDiv(logical_tile_id, panel_span);
    PrimExpr total_panels = ceildiv(total_tiles, panel_span);
    PrimExpr stride = Select(
        LT(panel_idx + one, total_panels), panel_imm,
        FloorDiv(total_tiles - panel_idx * panel_span, grid_x));
    PrimExpr tile_x = Select(EQ(FloorMod(panel_idx, two), one),
                             grid_x - one - FloorDiv(panel_offset, stride),
                             FloorDiv(panel_offset, stride));
    PrimExpr tile_y =
        FloorMod(panel_offset, stride) + panel_idx * panel_imm;
    return {tile_x, tile_y};
  }

  ICHECK(spec.kind == PermutationKind::kSwizzleColumn);
  PrimExpr panel_span = panel_imm * grid_y;
  PrimExpr panel_offset = FloorMod(logical_tile_id, panel_span);
  PrimExpr panel_idx = FloorDiv(logical_tile_id, panel_span);
  PrimExpr total_panels = ceildiv(total_tiles, panel_span);
  PrimExpr stride = Select(
      LT(panel_idx + one, total_panels), panel_imm,
      FloorDiv(total_tiles - panel_idx * panel_span, grid_y));
  PrimExpr tile_y = Select(EQ(FloorMod(panel_idx, two), one),
                           grid_y - one - FloorDiv(panel_offset, stride),
                           FloorDiv(panel_offset, stride));
  PrimExpr tile_x = FloorMod(panel_offset, stride) + panel_idx * panel_imm;
  return {tile_x, tile_y};
}

inline TileWorkBinding MakePersistentStaticTileWorkBinding(
    const TileScheduleSpec &spec, const PrimExpr &worker_id,
    const Var &iteration_id, const PrimExpr &grid_x, const PrimExpr &grid_y) {
  ICHECK(spec.distribution_kind == DistributionKind::kPersistentStatic)
      << "Only persistent_static distribution is implemented";
  PrimExpr total_tiles = grid_x * grid_y;
  PrimExpr worker_count = IntImm(worker_id.dtype(), spec.num_persistent_workers);
  PrimExpr logical_tile_id = worker_count * iteration_id + worker_id;
  auto coords =
      ApplyTilePermutation(spec.permutation, logical_tile_id, grid_x, grid_y);
  return {
      worker_id,
      iteration_id,
      logical_tile_id,
      coords.first,
      coords.second,
      LT(logical_tile_id, total_tiles),
  };
}

} // namespace tile_scheduler
} // namespace tl
} // namespace tvm
