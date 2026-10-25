/*!
 * \file tl/cuda/op/builtin.cc
 * \brief Registration of CUDA-specific TileLang intrinsic Ops.
 */

#include "builtin.h"

#include <tvm/ir/transform.h>

#include "op/builtin_registry.h"

namespace tvm {
namespace tl {

using namespace tirx;

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kDisableWarpSpecialized);
}
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::transform::PassContext::RegisterConfigOption<ffi::String>(
      kEnableAutoWarpSpecialization);
}
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::transform::PassContext::RegisterConfigOption<bool>(kDisableTMALower);
}
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::transform::PassContext::RegisterConfigOption<int64_t>(
      kPtxasRegisterUsageLevel);
}
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::transform::PassContext::RegisterConfigOption<bool>(kDisableVectorize256);
}
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kEnableFP32x2Reduction);
}
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::transform::PassContext::RegisterConfigOption<bool>(kDisableWGMMA);
}
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::transform::PassContext::RegisterConfigOption<bool>(kDisableShuffleElect);
}
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::transform::PassContext::RegisterConfigOption<bool>(kEnableLowerLDGSTG);
}
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kEnableLowerLDGSTGPredicated);
}

PrimType CuTensorMapType() { return PrimType::UInt(8, 128); }

TIR_DEFINE_TL_BUILTIN(__exp, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(__exp10, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(__log, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(__log2, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(__log10, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(__tan, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(__cos, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(__sin, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(fast_rcp, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(max_nan, 2, kPure, false);

TIR_DEFINE_TL_BUILTIN(min_nan, 2, kPure, false);

TIR_DEFINE_TL_BUILTIN(ieee_add, 3, kPure, false);

TIR_DEFINE_TL_BUILTIN(ieee_sub, 3, kPure, false);

TIR_DEFINE_TL_BUILTIN(ieee_mul, 3, kPure, false);

TIR_DEFINE_TL_BUILTIN(ieee_fmaf, 4, kPure, false);

TIR_DEFINE_TL_BUILTIN(ieee_frcp, 2, kPure, false);

TIR_DEFINE_TL_BUILTIN(ieee_fsqrt, 2, kPure, false);

TIR_DEFINE_TL_BUILTIN(ieee_frsqrt, 1, kPure, false);

TIR_DEFINE_TL_BUILTIN(ieee_fdiv, 3, kPure, false);

TIR_DEFINE_TL_BUILTIN(fma, 3, kPure, true);

TIR_DEFINE_TL_BUILTIN(fmul, 2, kPure, true);

TIR_DEFINE_TL_BUILTIN(max2_nan, 2, kPure, false);

TIR_DEFINE_TL_BUILTIN(min2_nan, 2, kPure, false);

TIR_DEFINE_TL_BUILTIN(create_tma_descriptor, -1, kPure, false);

TIR_DEFINE_TL_BUILTIN(create_tma_im2col_descriptor, -1, kPure, false);

TIR_DEFINE_TL_BUILTIN(prefetch_tma_descriptor, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tma_load, -1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tma_load_im2col, -1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tma_load_multicast, -1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tma_store, -1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tma_load_gather4, -1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tma_store_scatter4, -1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_fence_barrier_init, -1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_arrive_cluster_barrier, 2, kOpaque, false);
TIR_DEFINE_TL_BUILTIN(mbarrier_arrive, 1, kOpaque, false);
TIR_DEFINE_TL_BUILTIN(mbarrier_arrive_expect_tx, 2, kOpaque, false);
TIR_DEFINE_TL_BUILTIN(mbarrier_init, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_wgmma_ss, 15, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_wgmma_rs, 15, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_wgmma_sp_ss, 18, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_wgmma_sp_rs, 17, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_mma_block_scale, 21, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_tcgen05_mma_ss, 14, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_tcgen05_mma_ts, 13, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_tcgen05_mma_blockscaled_ss, 16, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_tcgen05_cp_warpx4, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_tcgen05_sf_warp_transpose, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(deallocate_tmem, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_init_tensor_memory, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_deallocate_tensor_memory, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_mma_sm70, 13, kOpaque, false);
TIR_DEFINE_TL_BUILTIN(ptx_mma_sp, 16, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_ldmatrix, 4, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_cp_async_barrier_noinc, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_st_bulk_shared, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(fence_proxy_async, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tcgen05_before_thread_sync, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tcgen05_after_thread_sync, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tma_store_arrive, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tma_store_wait, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(set_max_nreg, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(warpgroup_arrive, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(warpgroup_commit_batch, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(warpgroup_wait, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(warpgroup_fence_operand, 4, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(get_lane_idx, -1, kPure, false);

TIR_DEFINE_TL_BUILTIN(get_warp_idx_sync, -1, kPure, false);

TIR_DEFINE_TL_BUILTIN(get_warp_idx, -1, kPure, false);

TIR_DEFINE_TL_BUILTIN(get_warp_group_idx, -1, kPure, false);

TIR_DEFINE_TL_BUILTIN(pack_b8x4, 4, kPure, false);

TIR_DEFINE_TL_BUILTIN(cluster_arrive_relaxed, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(cluster_arrive, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(cluster_wait, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(cluster_sync, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(block_rank_in_cluster, 0, kPure, false);

TIR_DEFINE_TL_BUILTIN(clc_try_cancel, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(clc_try_cancel_multicast, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(clc_is_canceled, 1, kPure, false);

TIR_DEFINE_TL_BUILTIN(clc_get_first_ctaid_x, 1, kPure, false);

TIR_DEFINE_TL_BUILTIN(clc_get_first_ctaid_y, 1, kPure, false);

TIR_DEFINE_TL_BUILTIN(clc_get_first_ctaid_z, 1, kPure, false);

TIR_DEFINE_TL_BUILTIN(named_barrier_arrive, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(pdl_trigger, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(pdl_sync, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tl_shuffle_elect, 1, kPure, false);

TIR_DEFINE_TL_BUILTIN(initialize_wgmma_descriptor, 5, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(initialize_tcgen05_descriptor, 7, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(increase_descriptor_offset, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tcgen05_mma_arrive, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tcgen05_ld, -1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tcgen05_st, -1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(__ffs, 1, kPure, false);

TIR_DEFINE_TL_BUILTIN(__fns, 3, kPure, false);

TIR_DEFINE_TL_BUILTIN(ldg32, -1, kReadState, false);

TIR_DEFINE_TL_BUILTIN(ldg64, -1, kReadState, false);

TIR_DEFINE_TL_BUILTIN(ldg128, -1, kReadState, false);

TIR_DEFINE_TL_BUILTIN(lds32, 1, kPure, false);

TIR_DEFINE_TL_BUILTIN(lds64, 1, kPure, false);

TIR_DEFINE_TL_BUILTIN(lds128, 1, kPure, false);

TIR_DEFINE_TL_BUILTIN(ldg256, -1, kReadState, false);

TIR_DEFINE_TL_BUILTIN(stg32, -1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(stg64, -1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(stg128, -1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(sts32, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(sts64, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(sts128, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(stg256, -1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_cluster_store, 4, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tma_store_cluster, 5, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(mvb_stage_index, 1, kPure, false);

} // namespace tl
} // namespace tvm

#undef TIR_DEFINE_TL_BUILTIN
