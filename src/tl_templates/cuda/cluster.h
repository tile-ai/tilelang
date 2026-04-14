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

/* Cluster launch control for tile schedule (Available on sm100) */

TL_DEVICE void clc_try_cancel(void *result_ptr, void *mbar_ptr) {
#if defined(CUTLASS_ARCH_CLC_ENABLED)
  uint32_t result_addr = smem_ptr_to_uint(result_ptr);
  uint32_t mbar_addr = smem_ptr_to_uint(mbar_ptr);
  asm volatile("{\n\t"
               "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::"
               "complete_tx::bytes.b128 [%0], [%1];\n\t"
               "}\n"
               :
               : "r"(result_addr), "r"(mbar_addr));
#else
  TILELANG_UNREACHABLE("CUTLASS_ARCH_CLC_ENABLED is not defined");
#endif
}

TL_DEVICE void clc_try_cancel_multicast(void *result_ptr, void *mbar_ptr) {
#if defined(CUTLASS_ARCH_CLC_ENABLED)
  uint32_t result_addr = smem_ptr_to_uint(result_ptr);
  uint32_t mbar_addr = smem_ptr_to_uint(mbar_ptr);
  asm volatile("{\n\t"
               "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::"
               "complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n\t"
               "}\n"
               :
               : "r"(result_addr), "r"(mbar_addr));
#else
  TILELANG_UNREACHABLE("CUTLASS_ARCH_CLC_ENABLED is not defined");
#endif
}

// CLC query responses are produced through the async shared-memory proxy and
// must be fenced before normal shared-memory loads decode the 16-byte result.
TL_DEVICE void clc_fence_proxy_async_shared_cta() {
#if defined(CUTLASS_ARCH_CLC_ENABLED)
  asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
#else
  TILELANG_UNREACHABLE("CUTLASS_ARCH_CLC_ENABLED is not defined");
#endif
}

TL_DEVICE int clc_is_canceled(void const *result_ptr) {
#if defined(CUTLASS_ARCH_CLC_ENABLED)
  uint32_t result_addr = smem_ptr_to_uint(result_ptr);
  uint32_t is_canceled = 0;
  clc_fence_proxy_async_shared_cta();
  asm volatile("{\n\t"
               ".reg .pred p1;\n\t"
               ".reg .b128 clc_result;\n\t"
               "ld.shared.b128 clc_result, [%1];\n\t"
               "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
               "clc_result;\n\t"
               "selp.u32 %0, 1, 0, p1;\n\t"
               "}\n"
               : "=r"(is_canceled)
               : "r"(result_addr)
               : "memory");
  return static_cast<int>(is_canceled);
#else
  TILELANG_UNREACHABLE("CUTLASS_ARCH_CLC_ENABLED is not defined");
#endif
}

TL_DEVICE uint32_t clc_get_first_ctaid_x(void const *result_ptr) {
#if defined(CUTLASS_ARCH_CLC_ENABLED)
  uint32_t result_addr = smem_ptr_to_uint(result_ptr);
  uint32_t x = 0, y = 0, z = 0;
  clc_fence_proxy_async_shared_cta();
  asm volatile(
      "{\n\t"
      ".reg .pred p1;\n\t"
      ".reg .b128 clc_result;\n\t"
      "ld.shared.b128 clc_result, [%3];\n\t"
      "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
      "clc_result;\n\t"
      "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, "
      "%1, %2, _}, clc_result;\n\t"
      "}\n"
      : "=r"(x), "=r"(y), "=r"(z)
      : "r"(result_addr)
      : "memory");
  return x;
#else
  TILELANG_UNREACHABLE("CUTLASS_ARCH_CLC_ENABLED is not defined");
#endif
}

TL_DEVICE uint32_t clc_get_first_ctaid_y(void const *result_ptr) {
#if defined(CUTLASS_ARCH_CLC_ENABLED)
  uint32_t result_addr = smem_ptr_to_uint(result_ptr);
  uint32_t x = 0, y = 0, z = 0;
  clc_fence_proxy_async_shared_cta();
  asm volatile(
      "{\n\t"
      ".reg .pred p1;\n\t"
      ".reg .b128 clc_result;\n\t"
      "ld.shared.b128 clc_result, [%3];\n\t"
      "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
      "clc_result;\n\t"
      "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, "
      "%1, %2, _}, clc_result;\n\t"
      "}\n"
      : "=r"(x), "=r"(y), "=r"(z)
      : "r"(result_addr)
      : "memory");
  return y;
#else
  TILELANG_UNREACHABLE("CUTLASS_ARCH_CLC_ENABLED is not defined");
#endif
}

TL_DEVICE uint32_t clc_get_first_ctaid_z(void const *result_ptr) {
#if defined(CUTLASS_ARCH_CLC_ENABLED)
  uint32_t result_addr = smem_ptr_to_uint(result_ptr);
  uint32_t x = 0, y = 0, z = 0;
  clc_fence_proxy_async_shared_cta();
  asm volatile(
      "{\n\t"
      ".reg .pred p1;\n\t"
      ".reg .b128 clc_result;\n\t"
      "ld.shared.b128 clc_result, [%3];\n\t"
      "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
      "clc_result;\n\t"
      "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, "
      "%1, %2, _}, clc_result;\n\t"
      "}\n"
      : "=r"(x), "=r"(y), "=r"(z)
      : "r"(result_addr)
      : "memory");
  return z;
#else
  TILELANG_UNREACHABLE("CUTLASS_ARCH_CLC_ENABLED is not defined");
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
