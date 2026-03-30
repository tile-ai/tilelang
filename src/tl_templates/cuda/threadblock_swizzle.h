#pragma once

#include "common.h"

namespace tl {

template <int panel_width> TL_DEVICE dim3 rasterization2DRow() {
  const unsigned int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
  const unsigned int grid_size = gridDim.x * gridDim.y;
  const unsigned int panel_size = panel_width * gridDim.x;
  const unsigned int panel_offset = block_idx % panel_size;
  const unsigned int panel_idx = block_idx / panel_size;
  const unsigned int total_panel = cutlass::ceil_div(grid_size, panel_size);
  const unsigned int stride =
      panel_idx + 1 < total_panel
          ? panel_width
          : (grid_size - panel_idx * panel_size) / gridDim.x;
  const unsigned int col_idx = (panel_idx & 1)
                                   ? gridDim.x - 1 - panel_offset / stride
                                   : panel_offset / stride;
  const unsigned int row_idx = panel_offset % stride + panel_idx * panel_width;
  return {col_idx, row_idx, blockIdx.z};
}

template <int panel_width> TL_DEVICE dim3 rasterization2DColumn() {
  const unsigned int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
  const unsigned int grid_size = gridDim.x * gridDim.y;
  const unsigned int panel_size = panel_width * gridDim.y;
  const unsigned int panel_offset = block_idx % panel_size;
  const unsigned int panel_idx = block_idx / panel_size;
  const unsigned int total_panel = cutlass::ceil_div(grid_size, panel_size);
  const unsigned int stride =
      panel_idx + 1 < total_panel
          ? panel_width
          : (grid_size - panel_idx * panel_size) / gridDim.y;
  const unsigned int row_idx = (panel_idx & 1)
                                   ? gridDim.y - 1 - panel_offset / stride
                                   : panel_offset / stride;
  const unsigned int col_idx = panel_offset % stride + panel_idx * panel_width;
  return {col_idx, row_idx, blockIdx.z};
}

// Cluster-aware row-major swizzle: cluster_dim_x CTAs are grouped together
// in the x-direction as one cluster unit. The swizzle operates at cluster
// granularity; each CTA within the cluster retains its intra-cluster x offset.
//
// Example: cluster_dim_x=2, gridDim.x=8 → 4 clusters in x.
//   The 4 clusters are swizzled as if gridDim.x were 4, then the final
//   col_idx = swizzled_cluster_x * 2 + (blockIdx.x % 2).
template <int panel_width, int cluster_dim_x>
TL_DEVICE dim3 rasterization2DRowWithCluster() {
  auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
  const unsigned int num_cluster_x = gridDim.x / cluster_dim_x;
  const unsigned int intra_cluster_x = blockIdx.x % cluster_dim_x;
  const unsigned int cluster_x = blockIdx.x / cluster_dim_x;
  const unsigned int cluster_idx = cluster_x + blockIdx.y * num_cluster_x;
  const unsigned int cluster_grid_size = num_cluster_x * gridDim.y;
  const unsigned int panel_size = panel_width * num_cluster_x;
  const unsigned int panel_offset = cluster_idx % panel_size;
  const unsigned int panel_idx = cluster_idx / panel_size;
  const unsigned int total_panel = ceil_div(cluster_grid_size, panel_size);
  const unsigned int stride =
      panel_idx + 1 < total_panel
          ? panel_width
          : (cluster_grid_size - panel_idx * panel_size) / num_cluster_x;
  const unsigned int swizzled_cluster_x =
      (panel_idx & 1) ? num_cluster_x - 1 - panel_offset / stride
                      : panel_offset / stride;
  const unsigned int swizzled_cluster_y =
      panel_offset % stride + panel_idx * panel_width;
  const unsigned int col_idx =
      swizzled_cluster_x * cluster_dim_x + intra_cluster_x;
  const unsigned int row_idx = swizzled_cluster_y;
  return {col_idx, row_idx, blockIdx.z};
}

// Cluster-aware column-major swizzle: cluster_dim_x CTAs are grouped together
// in the x-direction. The swizzle operates at cluster granularity in a
// column-major fashion.
template <int panel_width, int cluster_dim_x>
TL_DEVICE dim3 rasterization2DColumnWithCluster() {
  auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
  const unsigned int num_cluster_x = gridDim.x / cluster_dim_x;
  const unsigned int intra_cluster_x = blockIdx.x % cluster_dim_x;
  const unsigned int cluster_x = blockIdx.x / cluster_dim_x;
  const unsigned int cluster_idx = cluster_x + blockIdx.y * num_cluster_x;
  const unsigned int cluster_grid_size = num_cluster_x * gridDim.y;
  const unsigned int panel_size = panel_width * gridDim.y;
  const unsigned int panel_offset = cluster_idx % panel_size;
  const unsigned int panel_idx = cluster_idx / panel_size;
  const unsigned int total_panel = ceil_div(cluster_grid_size, panel_size);
  const unsigned int stride =
      panel_idx + 1 < total_panel
          ? panel_width
          : (cluster_grid_size - panel_idx * panel_size) / gridDim.y;
  const unsigned int swizzled_cluster_y =
      (panel_idx & 1) ? gridDim.y - 1 - panel_offset / stride
                      : panel_offset / stride;
  const unsigned int swizzled_cluster_x =
      panel_offset % stride + panel_idx * panel_width;
  const unsigned int col_idx =
      swizzled_cluster_x * cluster_dim_x + intra_cluster_x;
  const unsigned int row_idx = swizzled_cluster_y;
  return {col_idx, row_idx, blockIdx.z};
}

} // namespace tl
