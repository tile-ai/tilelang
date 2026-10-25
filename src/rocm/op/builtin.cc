/*!
 * \file tl/rocm/op/builtin.cc
 * \brief Registration of ROCm-specific TileLang intrinsic Ops.
 */

#include "builtin.h"

#include "op/builtin_registry.h"

namespace tvm {
namespace tl {

using namespace tirx;

TIR_DEFINE_TL_BUILTIN(ds_read_tr16_b64, 1, kPure, false);

TIR_DEFINE_TL_BUILTIN(ds_read_tr8_b64, 1, kPure, false);

TIR_DEFINE_TL_BUILTIN(tvm_mfma, 12, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tvm_mfma_store, 6, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tvm_rdna_wmma, 12, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(tvm_rdna_wmma_store, 6, kOpaque, false);

} // namespace tl
} // namespace tvm

#undef TIR_DEFINE_TL_BUILTIN
