#pragma once

#include "common.h"

// Config
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && \
  ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8))))
#  define CLUSTER_ENABLED
#endif

namespace tl {

TL_DEVICE void cluster_arrive_relaxed() {
#if defined(CLUSTER_ENABLED)
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : : );
#else
  TILELANG_CHECK(false, "CLUSTER_ENABLED is not defined");
#endif
}

TL_DEVICE void cluster_arrive() {
#if defined(CLUSTER_ENABLED)
  asm volatile("barrier.cluster.arrive.aligned;\n" : : );
#else
  TILELANG_CHECK(false, "CLUSTER_ENABLED is not defined");
#endif
}

TL_DEVICE void cluster_wait() {
#if defined(CLUSTER_ENABLED)
  asm volatile("barrier.cluster.wait.aligned;\n" : : );
#else
  TILELANG_CHECK(false, "CLUSTER_ENABLED is not defined");
#endif
}

TL_DEVICE void cluster_sync() {
#if defined(CLUSTER_ENABLED)
  cluster_arrive();
  cluster_wait();
#else
  TILELANG_CHECK(false, "CLUSTER_ENABLED is not defined");
#endif
}

// Returns the dim3 grid size in terms of number of clusters.
TL_DEVICE dim3 cluster_grid_dims() {
#if defined(CLUSTER_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%nclusterid.x;\n" : "=r"(x) : );
  asm volatile("mov.u32 %0, %%nclusterid.y;\n" : "=r"(y) : );
  asm volatile("mov.u32 %0, %%nclusterid.z;\n" : "=r"(z) : );
  return {x, y, z};
#else
  TILELANG_CHECK(false, "CLUSTER_ENABLED is not defined");
#endif
}

// Returns the dim3 cluster rank in the grid.
TL_DEVICE dim3 cluster_id_in_grid() {
#if defined(CLUSTER_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%clusterid.x;\n" : "=r"(x) : );
  asm volatile("mov.u32 %0, %%clusterid.y;\n" : "=r"(y) : );
  asm volatile("mov.u32 %0, %%clusterid.z;\n" : "=r"(z) : );
  return {x, y, z};
#else
  TILELANG_CHECK(false, "CLUSTER_ENABLED is not defined");
#endif
}

// Returns the dim3 cluster shape.
TL_DEVICE dim3 cluster_shape() {
#if defined(CLUSTER_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;\n" : "=r"(x) : );
  asm volatile("mov.u32 %0, %%cluster_nctaid.y;\n" : "=r"(y) : );
  asm volatile("mov.u32 %0, %%cluster_nctaid.z;\n" : "=r"(z) : );
  return {x, y, z};
#else
  TILELANG_CHECK(false, "CLUSTER_ENABLED is not defined");
#endif
}

// Returns the relative dim3 block rank local to the cluster.
TL_DEVICE dim3 block_id_in_cluster() {
#if defined(CLUSTER_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;\n" : "=r"(x) : );
  asm volatile("mov.u32 %0, %%cluster_ctaid.y;\n" : "=r"(y) : );
  asm volatile("mov.u32 %0, %%cluster_ctaid.z;\n" : "=r"(z) : );
  return {x, y, z};
#else
  TILELANG_CHECK(false, "CLUSTER_ENABLED is not defined");
#endif
}

// Get 1D ctaid in a cluster.
TL_DEVICE uint32_t block_rank_in_cluster() {
#if defined(CLUSTER_ENABLED)
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank) :);
  return rank;
#else
  TILELANG_CHECK(false, "CLUSTER_ENABLED is not defined");
#endif
}

// Set the destination block-ID in cluster for a given SMEM Address
TL_DEVICE uint32_t set_block_rank(uint32_t smemAddr, uint32_t rank) {
#if defined(CLUSTER_ENABLED)
  uint32_t result;
  asm volatile("mapa.shared::cluster.u32  %0, %1, %2;\n"
                : "=r"(result)
                : "r"(smemAddr), "r"(rank));
  return result;
#else
  TILELANG_CHECK(false, "CLUSTER_ENABLED is not defined");
#endif
}

}