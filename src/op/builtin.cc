/*!
 * \file tl/op/builtin.cc
 * \brief Registration of backend-neutral TileLang intrinsic Ops.
 */

#include "builtin.h"

#include <tvm/ir/transform.h>

#include "builtin_registry.h"

namespace tvm {
namespace tl {

using namespace tirx;

TVM_REGISTER_PASS_CONFIG_OPTION(kDebugMergeSharedMemoryAllocations, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableSafeMemoryLegalize, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableThreadStorageSync, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kConfigIndexBitwidth, Integer);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableAggressiveSharedMemoryMerge, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableSharedMemoryReuse, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kForceLetInline, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableFastMath, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableAsyncCopy, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kReducerForceBaseline, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableReducerPlanVerbose, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kLayoutCostModel, ffi::String);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableVectorizePlannerVerbose, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kStorageRewriteDetectInplace, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kASTPrintEnable, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kLayoutVisualizationEnable, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kLayoutVisualizationFormats, ffi::String);
TVM_REGISTER_PASS_CONFIG_OPTION(kDeviceCompileFlags, ffi::Array<ffi::String>);
TVM_REGISTER_PASS_CONFIG_OPTION(kEmitLineDirectives, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableDataRaceCheck, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableBufferInitCheck, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableLoopUnswitching, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kLoopUnswitchingAllowNonTrivialElse, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kIfStmtBindingInlineReplayableBinds, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableOutOfBoundWarning, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableDumpIR, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDumpIRDir, ffi::String);
TVM_REGISTER_PASS_CONFIG_OPTION(kPassProfile, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kPassProfileThresholdMs, FloatImm);

TIR_DEFINE_TL_BUILTIN(tvm_ffi_call_with_result)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// VMI builtin registration macros (mirrors TIR_DEFINE_TL_BUILTIN from
// builtin_registry.h, but under the tl.vmi.* op prefix).
#define TIR_DEFINE_TL_VMI_BUILTIN(Name)                                        \
  const Op &vmi_##Name() {                                                     \
    static const Op &op = Op::Get("tl.vmi." #Name);                            \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl.vmi." #Name)                                             \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", "tl.vmi." #Name)

#define TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(Name)                                 \
  TIR_DEFINE_TL_VMI_BUILTIN(Name).set_attr<TCallEffectKind>(                   \
      "TCallEffectKind", Integer(CallEffectKind::kOpaque))

#define TIR_DEFINE_TL_VMI_PURE_BUILTIN(Name)                                   \
  TIR_DEFINE_TL_VMI_BUILTIN(Name).set_attr<TCallEffectKind>(                   \
      "TCallEffectKind", Integer(CallEffectKind::kPure))

TIR_DEFINE_TL_BUILTIN(access_ptr)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(clamp)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_TL_BUILTIN(region).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(launch_thread_idx)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(add2).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(sub2).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(mul2).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(fma2).set_num_inputs(3).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(max2).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(min2).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(abs2).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(mbarrier_wait_parity)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(mbarrier_expect_tx)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_stmatrix)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_cp_async)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(annotate_producer_reg_dealloc)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(annotate_consumer_reg_alloc)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(no_set_max_nreg)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

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

// VMI builtins are registered separately from the Python wrappers so later
// passes/codegen can match on stable tl.vmi.* op identities. The positional
// operand ABI plus annotation lowering rules are documented next to the
// declarations in builtin.h and correspond to tilelang/ascend/language/vmi.py.
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vload).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vstore).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(create_mask).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vci).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vbrc).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vintlv).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vdintlv).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vunzip).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vzip).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_PURE_BUILTIN(pair_get).set_num_inputs(2);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vadd).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vsub).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vmul).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vdiv).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vmax).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vmin).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vand).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vor).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vxor).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vshl).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vshr).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vabs).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vneg).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vrelu).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vexp).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vln).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vsqrt).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vnot).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vadds).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vmuls).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vmaxs).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vmins).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vshls).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vshrs).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vcmp).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vcmps).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vsel).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vselr).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vcadd).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vcmax).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vcmin).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vcvt).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vinterpret_cast).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vexpdif).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vaxpy).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vlrelu).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vprelu).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vmull).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vmula).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vdhist).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vchist).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vgather).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vgatherb).set_num_inputs(-1);
TIR_DEFINE_TL_VMI_OPAQUE_BUILTIN(vscatter).set_num_inputs(-1);

TIR_DEFINE_TL_BUILTIN(any_sync).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(all_sync).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ballot_sync)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ballot).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(activemask)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(syncthreads_count)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(syncthreads_and)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(syncthreads_or)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(shfl_sync).set_num_inputs(4).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(shfl_xor_sync)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(shfl_down_sync)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(shfl_up_sync)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(match_any_sync)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(match_all_sync)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(loop_break)
    .set_num_inputs(0)
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

TIR_DEFINE_TL_BUILTIN(atomic_addx2_ret_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_addx4_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_addx4_ret_elem_op)
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

TIR_DEFINE_TL_BUILTIN(atomic_or_elem_op)
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

TIR_DEFINE_TL_BUILTIN(__ldg).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(rng_init).set_num_inputs(4).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(rng_rand).set_num_inputs(0).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(rng_rand_float)
    .set_num_inputs(1)
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

} // namespace tl
} // namespace tvm

#undef TIR_DEFINE_TL_BUILTIN
