#pragma once

#include "common.h"

// Config
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) &&                       \
     ((__CUDACC_VER_MAJOR__ >= 12) ||                                          \
      ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8))))
#define TILELANG_CLUSTER_ENABLED
#endif

namespace tl {

TL_DEVICE void cluster_arrive_relaxed() {
#if defined(TILELANG_CLUSTER_ENABLED)
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : :);
#else
  TILELANG_UNREACHABLE("TILELANG_CLUSTER_ENABLED is not defined");
#endif
}

TL_DEVICE void cluster_arrive() {
#if defined(TILELANG_CLUSTER_ENABLED)
  asm volatile("barrier.cluster.arrive.aligned;\n" : :);
#else
  TILELANG_UNREACHABLE("TILELANG_CLUSTER_ENABLED is not defined");
#endif
}

TL_DEVICE void cluster_wait() {
#if defined(TILELANG_CLUSTER_ENABLED)
  asm volatile("barrier.cluster.wait.aligned;\n" : :);
#else
  TILELANG_UNREACHABLE("TILELANG_CLUSTER_ENABLED is not defined");
#endif
}

TL_DEVICE void cluster_sync() {
  cluster_arrive();
  cluster_wait();
}

// Returns the dim3 grid size in terms of number of clusters.
TL_DEVICE dim3 cluster_grid_dims() {
#if defined(TILELANG_CLUSTER_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%nclusterid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%nclusterid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%nclusterid.z;\n" : "=r"(z) :);
  return {x, y, z};
#else
  TILELANG_UNREACHABLE("TILELANG_CLUSTER_ENABLED is not defined");
#endif
}

// Returns the dim3 cluster rank in the grid.
TL_DEVICE dim3 cluster_id_in_grid() {
#if defined(TILELANG_CLUSTER_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%clusterid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%clusterid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%clusterid.z;\n" : "=r"(z) :);
  return {x, y, z};
#else
  TILELANG_UNREACHABLE("TILELANG_CLUSTER_ENABLED is not defined");
#endif
}

// Returns the dim3 cluster shape.
TL_DEVICE dim3 cluster_shape() {
#if defined(TILELANG_CLUSTER_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%cluster_nctaid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%cluster_nctaid.z;\n" : "=r"(z) :);
  return {x, y, z};
#else
  TILELANG_UNREACHABLE("TILELANG_CLUSTER_ENABLED is not defined");
#endif
}

// Returns the relative dim3 block rank local to the cluster.
TL_DEVICE dim3 block_id_in_cluster() {
#if defined(TILELANG_CLUSTER_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%cluster_ctaid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%cluster_ctaid.z;\n" : "=r"(z) :);
  return {x, y, z};
#else
  TILELANG_UNREACHABLE("TILELANG_CLUSTER_ENABLED is not defined");
#endif
}

// Get 1D ctaid in a cluster.
TL_DEVICE int block_rank_in_cluster() {
#if defined(TILELANG_CLUSTER_ENABLED)
  // NOTE(wt): cluster_ctarank is a uint32_t inherently,
  // we return as int32 for TL analysis convenience.
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank) :);
  return static_cast<int>(rank);
#else
  TILELANG_UNREACHABLE("TILELANG_CLUSTER_ENABLED is not defined");
#endif
}

// Set the destination block-ID in cluster for a given SMEM Address
TL_DEVICE uint32_t set_block_rank(uint32_t smemAddr, uint32_t rank) {
#if defined(TILELANG_CLUSTER_ENABLED)
  uint32_t result;
  asm volatile("mapa.shared::cluster.u32  %0, %1, %2;\n"
               : "=r"(result)
               : "r"(smemAddr), "r"(rank));
  return result;
#else
  TILELANG_UNREACHABLE("TILELANG_CLUSTER_ENABLED is not defined");
#endif
}

} // namespace tl
