/*!
 * \file tl/ascend/op/builtin.cc
 * \brief Registration of Ascend-specific TileLang intrinsic Ops.
 */

#include "builtin.h"

#include <tvm/ir/transform.h>
#include <tvm/runtime/memory_scope.h>

#include "op/builtin_registry.h"

namespace tvm {
namespace tl {

using namespace tirx;

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::transform::PassContext::RegisterConfigOption<bool>(kEnableAutoSchedule);
  for (const char *scope :
       {"local.var", "shared.aiv0", "shared.aiv1", "shared.bt", "shared.fp",
        "shared.l0a", "shared.l0a.dyn", "shared.l0a.sf", "shared.l0b",
        "shared.l0b.dyn", "shared.l0b.sf", "shared.l0c", "shared.l0c.dyn",
        "shared.l1", "shared.l1.dyn", "shared.ss", "shared.ss.dyn"}) {
    if (!runtime::MemoryScope::IsDefined(scope)) {
      runtime::MemoryScope::Register(scope);
    }
  }
}

// Emitted by tilelang.ascend.language.assume_conflict/assume_no_conflict and
// consumed by src/ascend/transform/normalize_conflict_hints.cc, so it lives
// with the rest of the Ascend op registrations.
TIR_DEFINE_TL_BUILTIN(conflict_hint, -1, kPure, false);

// The T.simd.* ops are registered under a "tl.simd.<name>" prefix while their
// accessors are simd_<name>, so they need a macro of their own; the neutral
// TIR_DEFINE_TL_BUILTIN (op/builtin_registry.h) maps both names to the same
// identifier.
#define TIR_DEFINE_TL_SIMD_BUILTIN(Name, NumInputs, Effect, Vectorizable)      \
  TL_DEFINE_NAMED_BUILTIN(simd_##Name, "tl.simd." #Name, NumInputs, Effect,    \
                          Vectorizable)

TIR_DEFINE_TL_BUILTIN(ascend_pipe_barrier, 1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(pset, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(pge, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(pand, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(por, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(pxor, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(pnot, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(psel, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(pld, 2, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(pst, 3, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vld, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vld2, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vsts, 4, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vst2, 3, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vadd, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vaddc, 3, kOpaque, false);
TIR_DEFINE_TL_SIMD_BUILTIN(vsubc, 3, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vaddcs, 4, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vsubcs, 4, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vmull, 3, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vlrelu, 3, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vprelu, 3, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(update_mask, 2, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(ppack, 2, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(punpack, 2, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(pintlv, 3, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(pdintlv, 3, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vunpack, 2, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vusqz, 1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vsub, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vmul, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vmula, 5, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vmadd, 5, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vaxpy, 5, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vdiv, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vmax, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vmin, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vand, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vor, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vxor, 4, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vshl, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vshr, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vexp, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vln, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vsqrt, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vabs, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vneg, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vrelu, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vnot, 3, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vdup, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vdupv, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vcpadd, 3, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vcadd, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vcmax, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vcmin, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vci, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vcmp, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vcmps, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vcgadd, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vcgmax, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vcgmin, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vsqz, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(dhistv2, 4, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(chistv2, 4, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vsstb, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vgatherb, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vgather2, 3, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vscatter, 4, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vintlv, 2, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vdintlv, 2, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(pair_get, 2, kPure, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vpack, 2, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vcvt, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vexpdif, 3, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vabsdif, 4, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vsel, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vselr, 2, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vmaxs, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vmins, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vmuls, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vadds, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vshls, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(vshrs, -1, kOpaque, false);

TIR_DEFINE_TL_SIMD_BUILTIN(mem_bar, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_set_flag, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_wait_flag, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_threadfence, 0, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_cross_core_set_flag, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_cross_core_wait_flag, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_copy_gm_to_ubuf, 11, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_set_copy_pad_value, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_copy_ubuf_to_gm, 8, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_copy_gm_to_cbuf, 11, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_fill_l1, 7, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_load_cbuf_to_ca, 9, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_load_cbuf_to_cb, 9, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_load_ca_sf, 8, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_load_cb_sf, 8, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_copy_matrix_cc_to_ub, 26, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_copy_ubuf_to_cbuf, 7, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_nd2nz_scatter, 7, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_nd2nz_post_copy, 6, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_copy_matrix_cc_to_gm, 25, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_mad, 10, kOpaque, false);

// L0 block-scaled MAD (MXFP8). Same args as ascend_mad; reads the per-block
// scale factors from the L0A/L0B MX registers loaded beforehand.
TIR_DEFINE_TL_BUILTIN(ascend_mad_mx, 10, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_gemm_l1, 11, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_blockscaled_gemm_l1, 17, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_read_gm_bypass_dcache, 1, kPure, false);

TIR_DEFINE_TL_BUILTIN(ascend_write_gm_bypass_dcache, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_get_buf, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_rls_buf, 3, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_set_hf32_mode, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_set_mmad_direction, 1, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_set_atomic, 2, kOpaque, false);

TIR_DEFINE_TL_BUILTIN(ascend_set_atomic_none, 0, kOpaque, false);

} // namespace tl
} // namespace tvm
