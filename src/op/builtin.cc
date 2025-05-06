// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/builtin.cc
 * \brief Builtin intrinsics.
 *
 */

#include "builtin.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../target/cuda.h"
#include "../target/utils.h"

namespace tvm {
namespace tl {

TVM_REGISTER_PASS_CONFIG_OPTION(kDisableTMALower, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableSafeMemoryLegalize, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableWarpSpecialized, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kConfigIndexBitwidth, Integer);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableDynamicTailSplit, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDynamicAlignment, Integer);

#define TIR_DEFINE_TL_BUILTIN(OpName)                                          \
  const Op &OpName() {                                                         \
    static const Op &op = Op::Get("tl." #OpName);                              \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl." #OpName)                                               \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)

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

TIR_DEFINE_TL_BUILTIN(mbarrier_wait_parity)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(mbarrier_expect_tx)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_ldmatirx)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_stmatirx)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(sync_thread_partial)
    .set_num_inputs(1)
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

TIR_DEFINE_TL_BUILTIN(wait_wgmma)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(pack_b16).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));
} // namespace tl
} // namespace tvm