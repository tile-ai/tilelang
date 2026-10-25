/*!
 * \file tl/op/builtin.cc
 * \brief Registration of backend-neutral TileLang intrinsic Ops.
 */

#include "builtin.h"

#include "builtin_registry.h"
#include "dtype.h"

#include <tvm/runtime/memory_scope.h>

namespace tvm {
namespace tl {

using namespace tirx;

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterCustomDType("tfloat32", kTensorFloat32, PrimType::Float(32));
  RegisterCustomDType("float4_e2m1_unpacked", kFloat4E2M1Unpacked,
                      PrimType::UInt(8));
  for (const char *scope :
       {"local.fragment", "local.reducer", "local.descriptor.wgmma",
        "local.descriptor.tcgen05_smem", "local.descriptor.tcgen05_instr",
        "shared.barrier", "shared.cluster_barrier", "shared.tmem",
        "metal.cooperative_tensor"}) {
    if (!runtime::MemoryScope::IsDefined(scope)) {
      runtime::MemoryScope::Register(scope);
    }
  }
}

TIR_DEFINE_TL_BUILTIN(tvm_ffi_call_with_result, 4, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(access_ptr, 3, kPure, false);

TIR_DEFINE_TL_BUILTIN(clamp, 3, kPure, true);

TIR_DEFINE_TL_BUILTIN(cast_with_rounding, -1, kPure, false);

TIR_DEFINE_TL_BUILTIN(region, -1, kPure, false);

TIR_DEFINE_TL_BUILTIN(launch_thread_idx, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(add2, 2, kPure, false);

TIR_DEFINE_TL_BUILTIN(sub2, 2, kPure, false);

TIR_DEFINE_TL_BUILTIN(mul2, 2, kPure, false);

TIR_DEFINE_TL_BUILTIN(fma2, 3, kPure, false);

TIR_DEFINE_TL_BUILTIN(max2, 2, kPure, false);

TIR_DEFINE_TL_BUILTIN(min2, 2, kPure, false);

TIR_DEFINE_TL_BUILTIN(abs2, 1, kPure, false);

TIR_DEFINE_TL_BUILTIN(mbarrier_wait_parity, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(mbarrier_expect_tx, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_stmatrix, -1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ptx_cp_async, -1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(annotate_producer_reg_dealloc, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(annotate_consumer_reg_alloc, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(no_set_max_nreg, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(wait_wgmma, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(pack_b16, 2, kPure, false);

TIR_DEFINE_TL_BUILTIN(sync_grid, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(sync_warp, -1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(any_sync, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(all_sync, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ballot_sync, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ballot, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(activemask, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(syncthreads_count, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(syncthreads_and, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(syncthreads_or, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(shfl_sync, 4, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(shfl_xor_sync, 4, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(shfl_down_sync, 4, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(shfl_up_sync, 4, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(match_any_sync, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(match_all_sync, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(loop_break, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(atomic_add_elem_op, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(atomic_add_ret_elem_op, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(atomic_addx2_elem_op, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(atomic_addx2_ret_elem_op, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(atomic_addx4_elem_op, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(atomic_addx4_ret_elem_op, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(atomic_load_elem_op, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(atomic_store_elem_op, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(atomic_or_elem_op, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(atomic_max_elem_op, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(atomic_max_ret_elem_op, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(atomic_min_elem_op, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(atomic_min_ret_elem_op, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(warp_reduce_sum, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(warp_reduce_max, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(warp_reduce_min, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(warp_reduce_bitand, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(warp_reduce_bitor, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(__ldg, -1, kPure, false);

TIR_DEFINE_TL_BUILTIN(rng_init, 4, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(rng_rand, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(rng_rand_float, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(device_assert, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(device_assert_with_msg, 2, kOpaque, false);

} // namespace tl
} // namespace tvm

#undef TIR_DEFINE_TL_BUILTIN
