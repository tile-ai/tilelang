/*!
 * \file tl/cuda/op/builtin.cc
 * \brief Registration of CUDA-specific TileLang intrinsic Ops.
 */

#include "builtin.h"

#include <tvm/ir/transform.h>

#include "op/native_intrinsic.h"

namespace tvm {
namespace tl {

using namespace tirx;

PtrType CuTensorMapType() { return PtrType(TensorMapType()); }

const Op &__exp() {
  static const Op op = Op::Get("tl.__exp");
  return op;
}

const Op &__exp10() {
  static const Op op = Op::Get("tl.__exp10");
  return op;
}

const Op &__log() {
  static const Op op = Op::Get("tl.__log");
  return op;
}

const Op &__log2() {
  static const Op op = Op::Get("tl.__log2");
  return op;
}

const Op &__log10() {
  static const Op op = Op::Get("tl.__log10");
  return op;
}

const Op &__tan() {
  static const Op op = Op::Get("tl.__tan");
  return op;
}

const Op &__cos() {
  static const Op op = Op::Get("tl.__cos");
  return op;
}

const Op &__sin() {
  static const Op op = Op::Get("tl.__sin");
  return op;
}

const Op &fast_rcp() {
  static const Op op = Op::Get("tl.fast_rcp");
  return op;
}

const Op &max_nan() {
  static const Op op = Op::Get("tl.max_nan");
  return op;
}

const Op &min_nan() {
  static const Op op = Op::Get("tl.min_nan");
  return op;
}

const Op &ieee_add() {
  static const Op op = Op::Get("tl.ieee_add");
  return op;
}

const Op &ieee_sub() {
  static const Op op = Op::Get("tl.ieee_sub");
  return op;
}

const Op &ieee_mul() {
  static const Op op = Op::Get("tl.ieee_mul");
  return op;
}

const Op &ieee_fmaf() {
  static const Op op = Op::Get("tl.ieee_fmaf");
  return op;
}

const Op &ieee_frcp() {
  static const Op op = Op::Get("tl.ieee_frcp");
  return op;
}

const Op &ieee_fsqrt() {
  static const Op op = Op::Get("tl.ieee_fsqrt");
  return op;
}

const Op &ieee_frsqrt() {
  static const Op op = Op::Get("tl.ieee_frsqrt");
  return op;
}

const Op &ieee_fdiv() {
  static const Op op = Op::Get("tl.ieee_fdiv");
  return op;
}

const Op &fma() {
  static const Op op = Op::Get("tl.fma");
  return op;
}

const Op &fmul() {
  static const Op op = Op::Get("tl.fmul");
  return op;
}

const Op &max2_nan() {
  static const Op op = Op::Get("tl.max2_nan");
  return op;
}

const Op &min2_nan() {
  static const Op op = Op::Get("tl.min2_nan");
  return op;
}

const Op &create_tma_descriptor() {
  static const Op op = Op::Get("tl.create_tma_descriptor");
  return op;
}

const Op &create_tma_im2col_descriptor() {
  static const Op op = Op::Get("tl.create_tma_im2col_descriptor");
  return op;
}

const Op &prefetch_tma_descriptor() {
  static const Op op = Op::Get("tl.prefetch_tma_descriptor");
  return op;
}

const Op &tma_load() {
  static const Op op = Op::Get("tl.tma_load");
  return op;
}

const Op &tma_load_im2col() {
  static const Op op = Op::Get("tl.tma_load_im2col");
  return op;
}

const Op &tma_load_multicast() {
  static const Op op = Op::Get("tl.tma_load_multicast");
  return op;
}

const Op &tma_store() {
  static const Op op = Op::Get("tl.tma_store");
  return op;
}

const Op &tma_load_gather4() {
  static const Op op = Op::Get("tl.tma_load_gather4");
  return op;
}

const Op &tma_store_scatter4() {
  static const Op op = Op::Get("tl.tma_store_scatter4");
  return op;
}

const Op &ptx_fence_barrier_init() {
  static const Op op = Op::Get("tl.ptx_fence_barrier_init");
  return op;
}

const Op &ptx_arrive_cluster_barrier() {
  static const Op op = Op::Get("tl.ptx_arrive_cluster_barrier");
  return op;
}

const Op &ptx_wgmma_ss() {
  static const Op op = Op::Get("tl.ptx_wgmma_ss");
  return op;
}

const Op &ptx_wgmma_rs() {
  static const Op op = Op::Get("tl.ptx_wgmma_rs");
  return op;
}

const Op &ptx_wgmma_sp_ss() {
  static const Op op = Op::Get("tl.ptx_wgmma_sp_ss");
  return op;
}

const Op &ptx_wgmma_sp_rs() {
  static const Op op = Op::Get("tl.ptx_wgmma_sp_rs");
  return op;
}

const Op &ptx_mma_block_scale() {
  static const Op op = Op::Get("tl.ptx_mma_block_scale");
  return op;
}

const Op &ptx_tcgen05_mma_ss() {
  static const Op op = Op::Get("tl.ptx_tcgen05_mma_ss");
  return op;
}

const Op &ptx_tcgen05_mma_ts() {
  static const Op op = Op::Get("tl.ptx_tcgen05_mma_ts");
  return op;
}

const Op &ptx_tcgen05_mma_blockscaled_ss() {
  static const Op op = Op::Get("tl.ptx_tcgen05_mma_blockscaled_ss");
  return op;
}

const Op &ptx_tcgen05_cp_warpx4() {
  static const Op op = Op::Get("tl.ptx_tcgen05_cp_warpx4");
  return op;
}

const Op &ptx_tcgen05_sf_warp_transpose() {
  static const Op op = Op::Get("tl.ptx_tcgen05_sf_warp_transpose");
  return op;
}

const Op &deallocate_tmem() {
  static const Op op = Op::Get("tl.deallocate_tmem");
  return op;
}

const Op &ptx_init_tensor_memory() {
  static const Op op = Op::Get("tl.ptx_init_tensor_memory");
  return op;
}

const Op &ptx_deallocate_tensor_memory() {
  static const Op op = Op::Get("tl.ptx_deallocate_tensor_memory");
  return op;
}

const Op &ptx_mma_sm70() {
  static const Op op = Op::Get("tl.ptx_mma_sm70");
  return op;
}

const Op &ptx_ldmatrix() {
  static const Op op = Op::Get("tl.ptx_ldmatrix");
  return op;
}

const Op &ptx_cp_async_barrier_noinc() {
  static const Op op = Op::Get("tl.ptx_cp_async_barrier_noinc");
  return op;
}

const Op &ptx_st_bulk_shared() {
  static const Op op = Op::Get("tl.ptx_st_bulk_shared");
  return op;
}

const Op &fence_proxy_async() {
  static const Op op = Op::Get("tl.fence_proxy_async");
  return op;
}

const Op &tcgen05_before_thread_sync() {
  static const Op op = Op::Get("tl.tcgen05_before_thread_sync");
  return op;
}

const Op &tcgen05_after_thread_sync() {
  static const Op op = Op::Get("tl.tcgen05_after_thread_sync");
  return op;
}

const Op &tma_store_arrive() {
  static const Op op = Op::Get("tl.tma_store_arrive");
  return op;
}

const Op &tma_store_wait() {
  static const Op op = Op::Get("tl.tma_store_wait");
  return op;
}

const Op &set_max_nreg() {
  static const Op op = Op::Get("tl.set_max_nreg");
  return op;
}

const Op &warpgroup_arrive() {
  static const Op op = Op::Get("tl.warpgroup_arrive");
  return op;
}

const Op &warpgroup_commit_batch() {
  static const Op op = Op::Get("tl.warpgroup_commit_batch");
  return op;
}

const Op &warpgroup_wait() {
  static const Op op = Op::Get("tl.warpgroup_wait");
  return op;
}

const Op &warpgroup_fence_operand() {
  static const Op op = Op::Get("tl.warpgroup_fence_operand");
  return op;
}

const Op &get_lane_idx() {
  static const Op op = Op::Get("tl.get_lane_idx");
  return op;
}

const Op &get_warp_idx_sync() {
  static const Op op = Op::Get("tl.get_warp_idx_sync");
  return op;
}

const Op &get_warp_idx() {
  static const Op op = Op::Get("tl.get_warp_idx");
  return op;
}

const Op &get_warp_group_idx() {
  static const Op op = Op::Get("tl.get_warp_group_idx");
  return op;
}

const Op &pack_b8x4() {
  static const Op op = Op::Get("tl.pack_b8x4");
  return op;
}

const Op &cluster_arrive_relaxed() {
  static const Op op = Op::Get("tl.cluster_arrive_relaxed");
  return op;
}

const Op &cluster_arrive() {
  static const Op op = Op::Get("tl.cluster_arrive");
  return op;
}

const Op &cluster_wait() {
  static const Op op = Op::Get("tl.cluster_wait");
  return op;
}

const Op &cluster_sync() {
  static const Op op = Op::Get("tl.cluster_sync");
  return op;
}

const Op &block_rank_in_cluster() {
  static const Op op = Op::Get("tl.block_rank_in_cluster");
  return op;
}

const Op &clc_try_cancel() {
  static const Op op = Op::Get("tl.clc_try_cancel");
  return op;
}

const Op &clc_try_cancel_multicast() {
  static const Op op = Op::Get("tl.clc_try_cancel_multicast");
  return op;
}

const Op &clc_is_canceled() {
  static const Op op = Op::Get("tl.clc_is_canceled");
  return op;
}

const Op &clc_get_first_ctaid_x() {
  static const Op op = Op::Get("tl.clc_get_first_ctaid_x");
  return op;
}

const Op &clc_get_first_ctaid_y() {
  static const Op op = Op::Get("tl.clc_get_first_ctaid_y");
  return op;
}

const Op &clc_get_first_ctaid_z() {
  static const Op op = Op::Get("tl.clc_get_first_ctaid_z");
  return op;
}

const Op &named_barrier_arrive() {
  static const Op op = Op::Get("tl.named_barrier_arrive");
  return op;
}

const Op &pdl_trigger() {
  static const Op op = Op::Get("tl.pdl_trigger");
  return op;
}

const Op &pdl_sync() {
  static const Op op = Op::Get("tl.pdl_sync");
  return op;
}

const Op &tl_shuffle_elect() {
  static const Op op = Op::Get("tl.tl_shuffle_elect");
  return op;
}

const Op &initialize_wgmma_descriptor() {
  static const Op op = Op::Get("tl.initialize_wgmma_descriptor");
  return op;
}

const Op &initialize_tcgen05_descriptor() {
  static const Op op = Op::Get("tl.initialize_tcgen05_descriptor");
  return op;
}

const Op &increase_descriptor_offset() {
  static const Op op = Op::Get("tl.increase_descriptor_offset");
  return op;
}

const Op &tcgen05_mma_arrive() {
  static const Op op = Op::Get("tl.tcgen05_mma_arrive");
  return op;
}

const Op &tcgen05_ld() {
  static const Op op = Op::Get("tl.tcgen05_ld");
  return op;
}

const Op &tcgen05_st() {
  static const Op op = Op::Get("tl.tcgen05_st");
  return op;
}

const Op &__ffs() {
  static const Op op = Op::Get("tl.__ffs");
  return op;
}

const Op &__fns() {
  static const Op op = Op::Get("tl.__fns");
  return op;
}

const Op &ldg32() {
  static const Op op = Op::Get("tl.ldg32");
  return op;
}

const Op &ldg64() {
  static const Op op = Op::Get("tl.ldg64");
  return op;
}

const Op &ldg128() {
  static const Op op = Op::Get("tl.ldg128");
  return op;
}

const Op &lds32() {
  static const Op op = Op::Get("tl.lds32");
  return op;
}

const Op &lds64() {
  static const Op op = Op::Get("tl.lds64");
  return op;
}

const Op &lds128() {
  static const Op op = Op::Get("tl.lds128");
  return op;
}

const Op &ldg256() {
  static const Op op = Op::Get("tl.ldg256");
  return op;
}

const Op &stg32() {
  static const Op op = Op::Get("tl.stg32");
  return op;
}

const Op &stg64() {
  static const Op op = Op::Get("tl.stg64");
  return op;
}

const Op &stg128() {
  static const Op op = Op::Get("tl.stg128");
  return op;
}

const Op &sts32() {
  static const Op op = Op::Get("tl.sts32");
  return op;
}

const Op &sts64() {
  static const Op op = Op::Get("tl.sts64");
  return op;
}

const Op &sts128() {
  static const Op op = Op::Get("tl.sts128");
  return op;
}

const Op &stg256() {
  static const Op op = Op::Get("tl.stg256");
  return op;
}

const Op &ptx_cluster_store() {
  static const Op op = Op::Get("tl.ptx_cluster_store");
  return op;
}

const Op &tma_store_cluster() {
  static const Op op = Op::Get("tl.tma_store_cluster");
  return op;
}

const Op &mvb_stage_index() {
  static const Op op = Op::Get("tl.mvb_stage_index");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kDisableWarpSpecialized);
  tvm::transform::PassContext::RegisterConfigOption<ffi::String>(
      kEnableAutoWarpSpecialization);
  tvm::transform::PassContext::RegisterConfigOption<bool>(kDisableTMALower);
  tvm::transform::PassContext::RegisterConfigOption<int64_t>(
      kPtxasRegisterUsageLevel);
  tvm::transform::PassContext::RegisterConfigOption<bool>(kDisableVectorize256);
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kEnableFP32x2Reduction);
  tvm::transform::PassContext::RegisterConfigOption<bool>(kDisableWGMMA);
  tvm::transform::PassContext::RegisterConfigOption<bool>(kDisableShuffleElect);
  tvm::transform::PassContext::RegisterConfigOption<bool>(kEnableLowerLDGSTG);
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kEnableLowerLDGSTGPredicated);
  DefineNativeIntrinsic(OpDef("tl.__exp"), 1, 1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.__exp10"), 1, 1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.__log"), 1, 1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.__log2"), 1, 1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.__log10"), 1, 1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.__tan"), 1, 1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.__cos"), 1, 1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.__sin"), 1, 1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.fast_rcp"), 1, 1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.max_nan"), 2, 2, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.min_nan"), 2, 2, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ieee_add"), 3, 3, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ieee_sub"), 3, 3, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ieee_mul"), 3, 3, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ieee_fmaf"), 4, 4, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ieee_frcp"), 2, 2, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ieee_fsqrt"), 2, 2, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ieee_frsqrt"), 1, 1, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ieee_fdiv"), 3, 3, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.fma"), 3, 3, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, true);
  DefineNativeIntrinsic(OpDef("tl.fmul"), 2, 2, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, true);
  DefineNativeIntrinsic(OpDef("tl.max2_nan"), 2, 2, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.min2_nan"), 2, 2, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.create_tma_descriptor"), 0, -1,
                        CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.create_tma_im2col_descriptor"), 0, -1,
                        CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.prefetch_tma_descriptor"), 1, 1,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tma_load"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tma_load_im2col"), 0, -1,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tma_load_multicast"), 0, -1,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tma_store"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tma_load_gather4"), 9, 9,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tma_store_scatter4"), 8, 8,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_fence_barrier_init"), 0, 0,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_arrive_cluster_barrier"), 2, 2,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_wgmma_ss"), 15, 15,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_wgmma_rs"), 15, 15,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_wgmma_sp_ss"), 18, 18,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_wgmma_sp_rs"), 17, 17,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_mma_block_scale"), 21, 21,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_tcgen05_mma_ss"), 14, 14,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_tcgen05_mma_ts"), 13, 13,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_tcgen05_mma_blockscaled_ss"), 16, 16,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_tcgen05_cp_warpx4"), 3, 3,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_tcgen05_sf_warp_transpose"), 1, 1,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.deallocate_tmem"), 1, 1,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_init_tensor_memory"), 2, 2,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_deallocate_tensor_memory"), 2, 2,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_mma_sm70"), 13, 13,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_ldmatrix"), 4, 4, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_cp_async_barrier_noinc"), 1, 1,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_st_bulk_shared"), 3, 3,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.fence_proxy_async"), 0, 0,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tcgen05_before_thread_sync"), 0, 0,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tcgen05_after_thread_sync"), 0, 0,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tma_store_arrive"), 0, 0,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tma_store_wait"), 2, 2,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.set_max_nreg"), 2, 2, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.warpgroup_arrive"), 0, 0,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.warpgroup_commit_batch"), 0, 0,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.warpgroup_wait"), 1, 1,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.warpgroup_fence_operand"), 4, 4,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.get_lane_idx"), 0, 1, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.get_warp_idx_sync"), 0, 1,
                        CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.get_warp_idx"), 0, 1, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.get_warp_group_idx"), 0, 2,
                        CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.pack_b8x4"), 4, 4, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.cluster_arrive_relaxed"), 0, 0,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.cluster_arrive"), 0, 0,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.cluster_wait"), 0, 0, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.cluster_sync"), 0, 0, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.block_rank_in_cluster"), 0, 0,
                        CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.clc_try_cancel"), 2, 2,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.clc_try_cancel_multicast"), 2, 2,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.clc_is_canceled"), 1, 1,
                        CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.clc_get_first_ctaid_x"), 1, 1,
                        CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.clc_get_first_ctaid_y"), 1, 1,
                        CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.clc_get_first_ctaid_z"), 1, 1,
                        CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.named_barrier_arrive"), 2, 2,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.pdl_trigger"), 0, 0, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.pdl_sync"), 0, 0, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tl_shuffle_elect"), 1, 1,
                        CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.initialize_wgmma_descriptor"), 5, 5,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.initialize_tcgen05_descriptor"), 7, 7,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.increase_descriptor_offset"), 2, 2,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tcgen05_mma_arrive"), 1, 1,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tcgen05_ld"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tcgen05_st"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.__ffs"), 1, 1, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.__fns"), 3, 3, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ldg32"), 0, -1, CallEffectKind::kReadState,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ldg64"), 0, -1, CallEffectKind::kReadState,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ldg128"), 0, -1, CallEffectKind::kReadState,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.lds32"), 1, 1, CallEffectKind::kReadState,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.lds64"), 1, 1, CallEffectKind::kReadState,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.lds128"), 1, 1, CallEffectKind::kReadState,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ldg256"), 0, -1, CallEffectKind::kReadState,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.stg32"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.stg64"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.stg128"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.sts32"), 2, 2, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.sts64"), 2, 2, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.sts128"), 2, 2, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.stg256"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ptx_cluster_store"), 4, 4,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.tma_store_cluster"), 5, 5,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.mvb_stage_index"), 1, 1,
                        CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
}

namespace {
void ValidateWmmaLoadMatrixSync(const CallObj *call) {
  TVM_FFI_CHECK_EQ(call->args.size(), 8U, ValueError)
      << "WMMA load requires fragment, m, n, k, index, pointer, stride and "
         "layout";
  const auto result_type = call->ty.as<PrimType>();
  TVM_FFI_CHECK(result_type && result_type->IsVoid() && call->ty_args.empty(),
                TypeError)
      << "WMMA load has a fixed void result";
  const auto &fragment_type = call->args[0]->ty;
  TVM_FFI_CHECK(fragment_type.as<tirx::TensorTypeObj>() ||
                    fragment_type.as<PtrTypeObj>(),
                TypeError)
      << "WMMA load requires a tensor or pointer fragment";
  TVM_FFI_CHECK(call->args[5]->ty.as<PtrTypeObj>(), TypeError)
      << "WMMA load requires a typed source pointer";
  for (size_t index : {1U, 2U, 3U, 4U, 6U}) {
    auto value = call->args[index].as<PrimExpr>();
    TVM_FFI_CHECK(value && value->ty().IsScalar() &&
                      value->ty().MatchesCode(kDLInt, kDLUInt),
                  TypeError)
        << "WMMA load dimensions, index and stride must be scalar integers";
  }
  for (size_t index : {1U, 2U, 3U}) {
    const auto *dimension = call->args[index].as<IntImmObj>();
    TVM_FFI_CHECK(dimension && dimension->value > 0, ValueError)
        << "WMMA load dimensions must be positive integer constants";
  }
  const auto *layout = call->args[7].as<StringImmObj>();
  TVM_FFI_CHECK(layout && (layout->value == "row_major" ||
                           layout->value == "column_major"),
                ValueError)
      << "WMMA load layout must be row_major or column_major";
}
} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  // This operation reads the source pointer and writes the fragment operand.
  // Keep TileLang's contract local: the native gpu_load_matrix_sync operator
  // at the supported upstream baseline is only marked as reading state.
  OpDef("tl.wmma_load_matrix_sync")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateWmmaLoadMatrixSync>())
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .signature(sig::arg("fragment", "The matrix fragment."),
                 sig::arg<IntExpr>("m", "The M dimension."),
                 sig::arg<IntExpr>("n", "The N dimension."),
                 sig::arg<IntExpr>("k", "The K dimension."),
                 sig::arg<IntExpr>("index", "The fragment index."),
                 sig::arg("buffer_ptr", "The source buffer pointer."),
                 sig::arg<IntExpr>("stride", "The source stride."),
                 sig::arg("layout", "The fragment layout."))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .expose();
}

} // namespace tl
} // namespace tvm
