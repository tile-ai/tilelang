/*!
 * \file tl/op/builtin.cc
 * \brief Builtin intrinsics.
 *
 */

#include "builtin.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../target/stubs/cuda.h"
#include "../target/utils.h"

namespace tvm {
namespace tl {

TVM_REGISTER_PASS_CONFIG_OPTION(kDebugMergeSharedMemoryAllocations, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableTMALower, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableSafeMemoryLegalize, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableWarpSpecialized, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableThreadStorageSync, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kConfigIndexBitwidth, Integer);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableAggressiveSharedMemoryMerge, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kForceLetInline, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableFastMath, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableFastMath, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kPtxasRegisterUsageLevel, Integer);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnablePTXASVerboseOutput, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableVectorize256, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableVectorizePlannerVerbose, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableWGMMA, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableShuffleElect, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kStorageRewriteDetectInplace, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kASTPrintEnable, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kLayoutVisualizationEnable, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kLayoutVisualizationFormats, String);
TVM_REGISTER_PASS_CONFIG_OPTION(kDeviceCompileFlags, ffi::Array<ffi::String>);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableDataRaceCheck, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableLowerLDGSTG, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableLowerLDGSTGPredicated, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableLoopUnswitching, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableOutOfBoundWarning, Bool);

DataType cuTensorMapType() { return DataType::UInt(8, 128); }

#define TIR_DEFINE_TL_BUILTIN(OpName)                                          \
  const Op &OpName() {                                                         \
    static const Op &op = Op::Get("tl." #OpName);                              \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl." #OpName)                                               \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)

// fast math related op
TIR_DEFINE_TL_BUILTIN(__exp).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(__exp10).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(__log).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(__log2).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(__log10).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(__tan).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(__cos).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(__sin).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// high precision with IEEE-compliant
TIR_DEFINE_TL_BUILTIN(ieee_add).set_num_inputs(3).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(ieee_sub).set_num_inputs(3).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(ieee_mul).set_num_inputs(3).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(ieee_fmaf).set_num_inputs(4).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(ieee_frcp).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(ieee_fsqrt)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(ieee_frsqrt)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(ieee_fdiv).set_num_inputs(3).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(rng_init).set_num_inputs(4).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(rng_rand).set_num_inputs(0).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(rng_rand_float)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(create_list_of_mbarrier)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(create_tma_descriptor)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(create_tma_im2col_descriptor)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(get_mbarrier)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(tma_load).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(tma_load_im2col)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(tma_store).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_fence_barrier_init)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(mbarrier_wait_parity)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(mbarrier_expect_tx)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_wgmma_ss)
    .set_num_inputs(15)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_wgmma_rs)
    .set_num_inputs(15)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_tcgen05_mma_ss)
    .set_num_inputs(14)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_tcgen05_mma_ts)
    .set_num_inputs(13)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_init_tensor_memory)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_deallocate_tensor_memory)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_mma_sm70)
    .set_num_inputs(13)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_ldmatrix)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_stmatrix)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_cp_async_barrier_noinc)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_cp_async)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(fence_proxy_async)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(tma_store_arrive)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(tma_store_wait)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(set_max_nreg)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(no_set_max_nreg)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(warpgroup_arrive)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(warpgroup_commit_batch)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(warpgroup_wait)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(warpgroup_fence_operand)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(get_lane_idx)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(get_warp_idx_sync)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(get_warp_idx)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(get_warp_group_idx)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(wait_wgmma)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(pack_b16).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(sync_grid).set_num_inputs(0).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(sync_warp).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(pdl_trigger)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(pdl_sync).set_num_inputs(0).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(loop_break)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(tl_gemm).set_num_inputs(4).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(tl_gemm_sp)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(tvm_mfma).set_num_inputs(12).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(tvm_mfma_store)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(tvm_rdna_wmma)
    .set_num_inputs(12)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(tvm_rdna_wmma_store)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(tl_shuffle_elect)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(initialize_wgmma_descriptor)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(initialize_tcgen05_descriptor)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(increase_descriptor_offset)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_add_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_add_ret_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_addx2_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_addx4_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_load_elem_op)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_store_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_max_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_max_ret_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_min_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_min_ret_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(device_assert)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(device_assert_with_msg)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(tcgen05_mma_arrive)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(warp_reduce_sum)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(warp_reduce_max)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(warp_reduce_min)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(warp_reduce_bitand)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(warp_reduce_bitor)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// __ldg(BufferLoad | Buffer, idx?) -> value
// Treat as a pure call that returns the loaded value.
TIR_DEFINE_TL_BUILTIN(__ldg).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

// ldg32(address, predicate(optional)) -> 32-bit value
// Global memory load with 32-bit vector width
TIR_DEFINE_TL_BUILTIN(ldg32).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

// ldg64(address, predicate(optional)) -> 64-bit value
// Global memory load with 64-bit vector width
TIR_DEFINE_TL_BUILTIN(ldg64).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

// ldg128(address, predicate(optional)) -> 128-bit value
// Global memory load with 128-bit vector width
TIR_DEFINE_TL_BUILTIN(ldg128).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

// ldg256(address, predicate(optional)) -> 256-bit value
// Global memory load with 256-bit vector width
TIR_DEFINE_TL_BUILTIN(ldg256).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

// stg32(Buffer, idx, value) -> void
// Global memory store with 32-bit vector width
TIR_DEFINE_TL_BUILTIN(stg32).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// stg64(Buffer, idx, value) -> void
// Global memory store with 64-bit vector width
TIR_DEFINE_TL_BUILTIN(stg64).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// stg128(Buffer, idx, value) -> void
// Global memory store with 128-bit vector width
TIR_DEFINE_TL_BUILTIN(stg128).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

// stg256(Buffer, idx, value) -> void
// Global memory store with 256-bit vector width
TIR_DEFINE_TL_BUILTIN(stg256).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm
