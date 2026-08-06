/*!
 * \file tl/rocm/op/builtin.cc
 * \brief Registration of ROCm-specific TileLang intrinsic Ops.
 */

#include "builtin.h"

#include "op/builtin_registry.h"

namespace tvm {
namespace tl {

using namespace tirx;

TIR_DEFINE_TL_BUILTIN(ds_read_tr16_b64)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(ds_read_tr8_b64)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

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

} // namespace tl
} // namespace tvm

#undef TIR_DEFINE_TL_BUILTIN
