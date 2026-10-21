/*!
 * \file tileview/reduce_tileview_planner.h
 * \brief TileView planning helpers specialized for Sunmmio reductions.
 */

#ifndef TVM_TL_TILEVIEW_REDUCE_TILEVIEW_PLANNER_H_
#define TVM_TL_TILEVIEW_REDUCE_TILEVIEW_PLANNER_H_

#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/stmt.h>

#include "../layout/layout.h"
#include "../target/sunmmio_utils.h"
#include "tileview.h"

namespace tvm {
namespace tl {

using namespace tir;

struct ReduceTileViewHints {
  Optional<TileView> src_tileview;
  Optional<TileView> dst_tileview;
  bool allow_tiled_reduction{true};
};

struct ReduceTileViewPlan {
  Array<PrimExpr> source_domain;
  TileView src_tileview;
  TileView dst_tileview;
  std::vector<int> execution_domain_axes;
  std::vector<int> src_dim_to_dst_dim;
  int reduce_tile_axis{-1};
};

/*!
 * \brief Plan the execution TileView for a Sunmmio reduction.
 *
 * Reduction planning differs from generic T.Tiles planning:
 * - src drives the execution space,
 * - dst is derived by projecting away the reduced source dimension,
 * - acc/dst_res are local temporaries derived later by lowering.
 */
ReduceTileViewPlan
PlanReduceTileViews(const BufferRegion &src_region,
                    const BufferRegion &dst_region, int reduce_dim,
                    const ReduceTileViewHints &hints,
                    const Map<Buffer, Layout> &layout_map,
                    const SunmmioTileProcessorConfig &tile_processor_config,
                    arith::Analyzer *analyzer);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TILEVIEW_REDUCE_TILEVIEW_PLANNER_H_
