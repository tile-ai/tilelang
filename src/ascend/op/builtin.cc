/*!
 * \file ascend/op/builtin.cc
 * \brief Native Ascend intrinsic registrations and operand metadata.
 */
#include "builtin.h"
#include "op/native_intrinsic.h"
#include <tvm/ir/transform.h>

namespace tvm::tl {
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::transform::PassContext::RegisterConfigOption<bool>(kEnableAutoSchedule);
}

const Op &conflict_hint() {
  static const Op op = Op::Get("tl.conflict_hint");
  return op;
}
const Op &ascend_pipe_barrier() {
  static const Op op = Op::Get("tl.ascend_pipe_barrier");
  return op;
}
const Op &simd_pset() {
  static const Op op = Op::Get("tl.simd.pset");
  return op;
}
const Op &simd_pge() {
  static const Op op = Op::Get("tl.simd.pge");
  return op;
}
const Op &simd_pand() {
  static const Op op = Op::Get("tl.simd.pand");
  return op;
}
const Op &simd_por() {
  static const Op op = Op::Get("tl.simd.por");
  return op;
}
const Op &simd_pxor() {
  static const Op op = Op::Get("tl.simd.pxor");
  return op;
}
const Op &simd_pnot() {
  static const Op op = Op::Get("tl.simd.pnot");
  return op;
}
const Op &simd_psel() {
  static const Op op = Op::Get("tl.simd.psel");
  return op;
}
const Op &simd_pld() {
  static const Op op = Op::Get("tl.simd.pld");
  return op;
}
const Op &simd_pst() {
  static const Op op = Op::Get("tl.simd.pst");
  return op;
}
const Op &simd_vld() {
  static const Op op = Op::Get("tl.simd.vld");
  return op;
}
const Op &simd_vld2() {
  static const Op op = Op::Get("tl.simd.vld2");
  return op;
}
const Op &simd_vsts() {
  static const Op op = Op::Get("tl.simd.vsts");
  return op;
}
const Op &simd_vst2() {
  static const Op op = Op::Get("tl.simd.vst2");
  return op;
}
const Op &simd_vadd() {
  static const Op op = Op::Get("tl.simd.vadd");
  return op;
}
const Op &simd_vaddc() {
  static const Op op = Op::Get("tl.simd.vaddc");
  return op;
}
const Op &simd_vsubc() {
  static const Op op = Op::Get("tl.simd.vsubc");
  return op;
}
const Op &simd_vaddcs() {
  static const Op op = Op::Get("tl.simd.vaddcs");
  return op;
}
const Op &simd_vsubcs() {
  static const Op op = Op::Get("tl.simd.vsubcs");
  return op;
}
const Op &simd_vmull() {
  static const Op op = Op::Get("tl.simd.vmull");
  return op;
}
const Op &simd_vlrelu() {
  static const Op op = Op::Get("tl.simd.vlrelu");
  return op;
}
const Op &simd_vprelu() {
  static const Op op = Op::Get("tl.simd.vprelu");
  return op;
}
const Op &simd_update_mask() {
  static const Op op = Op::Get("tl.simd.update_mask");
  return op;
}
const Op &simd_ppack() {
  static const Op op = Op::Get("tl.simd.ppack");
  return op;
}
const Op &simd_punpack() {
  static const Op op = Op::Get("tl.simd.punpack");
  return op;
}
const Op &simd_pintlv() {
  static const Op op = Op::Get("tl.simd.pintlv");
  return op;
}
const Op &simd_pdintlv() {
  static const Op op = Op::Get("tl.simd.pdintlv");
  return op;
}
const Op &simd_vunpack() {
  static const Op op = Op::Get("tl.simd.vunpack");
  return op;
}
const Op &simd_vusqz() {
  static const Op op = Op::Get("tl.simd.vusqz");
  return op;
}
const Op &simd_vsub() {
  static const Op op = Op::Get("tl.simd.vsub");
  return op;
}
const Op &simd_vmul() {
  static const Op op = Op::Get("tl.simd.vmul");
  return op;
}
const Op &simd_vmula() {
  static const Op op = Op::Get("tl.simd.vmula");
  return op;
}
const Op &simd_vmadd() {
  static const Op op = Op::Get("tl.simd.vmadd");
  return op;
}
const Op &simd_vaxpy() {
  static const Op op = Op::Get("tl.simd.vaxpy");
  return op;
}
const Op &simd_vdiv() {
  static const Op op = Op::Get("tl.simd.vdiv");
  return op;
}
const Op &simd_vmax() {
  static const Op op = Op::Get("tl.simd.vmax");
  return op;
}
const Op &simd_vmin() {
  static const Op op = Op::Get("tl.simd.vmin");
  return op;
}
const Op &simd_vand() {
  static const Op op = Op::Get("tl.simd.vand");
  return op;
}
const Op &simd_vor() {
  static const Op op = Op::Get("tl.simd.vor");
  return op;
}
const Op &simd_vxor() {
  static const Op op = Op::Get("tl.simd.vxor");
  return op;
}
const Op &simd_vshl() {
  static const Op op = Op::Get("tl.simd.vshl");
  return op;
}
const Op &simd_vshr() {
  static const Op op = Op::Get("tl.simd.vshr");
  return op;
}
const Op &simd_vexp() {
  static const Op op = Op::Get("tl.simd.vexp");
  return op;
}
const Op &simd_vln() {
  static const Op op = Op::Get("tl.simd.vln");
  return op;
}
const Op &simd_vsqrt() {
  static const Op op = Op::Get("tl.simd.vsqrt");
  return op;
}
const Op &simd_vabs() {
  static const Op op = Op::Get("tl.simd.vabs");
  return op;
}
const Op &simd_vneg() {
  static const Op op = Op::Get("tl.simd.vneg");
  return op;
}
const Op &simd_vrelu() {
  static const Op op = Op::Get("tl.simd.vrelu");
  return op;
}
const Op &simd_vnot() {
  static const Op op = Op::Get("tl.simd.vnot");
  return op;
}
const Op &simd_vdup() {
  static const Op op = Op::Get("tl.simd.vdup");
  return op;
}
const Op &simd_vdupv() {
  static const Op op = Op::Get("tl.simd.vdupv");
  return op;
}
const Op &simd_vcpadd() {
  static const Op op = Op::Get("tl.simd.vcpadd");
  return op;
}
const Op &simd_vcadd() {
  static const Op op = Op::Get("tl.simd.vcadd");
  return op;
}
const Op &simd_vcmax() {
  static const Op op = Op::Get("tl.simd.vcmax");
  return op;
}
const Op &simd_vcmin() {
  static const Op op = Op::Get("tl.simd.vcmin");
  return op;
}
const Op &simd_vci() {
  static const Op op = Op::Get("tl.simd.vci");
  return op;
}
const Op &simd_vcmp() {
  static const Op op = Op::Get("tl.simd.vcmp");
  return op;
}
const Op &simd_vcmps() {
  static const Op op = Op::Get("tl.simd.vcmps");
  return op;
}
const Op &simd_vcgadd() {
  static const Op op = Op::Get("tl.simd.vcgadd");
  return op;
}
const Op &simd_vcgmax() {
  static const Op op = Op::Get("tl.simd.vcgmax");
  return op;
}
const Op &simd_vcgmin() {
  static const Op op = Op::Get("tl.simd.vcgmin");
  return op;
}
const Op &simd_vsqz() {
  static const Op op = Op::Get("tl.simd.vsqz");
  return op;
}
const Op &simd_dhistv2() {
  static const Op op = Op::Get("tl.simd.dhistv2");
  return op;
}
const Op &simd_chistv2() {
  static const Op op = Op::Get("tl.simd.chistv2");
  return op;
}
const Op &simd_vsstb() {
  static const Op op = Op::Get("tl.simd.vsstb");
  return op;
}
const Op &simd_vgatherb() {
  static const Op op = Op::Get("tl.simd.vgatherb");
  return op;
}
const Op &simd_vgather2() {
  static const Op op = Op::Get("tl.simd.vgather2");
  return op;
}
const Op &simd_vscatter() {
  static const Op op = Op::Get("tl.simd.vscatter");
  return op;
}
const Op &simd_vintlv() {
  static const Op op = Op::Get("tl.simd.vintlv");
  return op;
}
const Op &simd_vdintlv() {
  static const Op op = Op::Get("tl.simd.vdintlv");
  return op;
}
const Op &simd_pair_get() {
  static const Op op = Op::Get("tl.simd.pair_get");
  return op;
}
const Op &simd_vpack() {
  static const Op op = Op::Get("tl.simd.vpack");
  return op;
}
const Op &simd_vcvt() {
  static const Op op = Op::Get("tl.simd.vcvt");
  return op;
}
const Op &simd_vexpdif() {
  static const Op op = Op::Get("tl.simd.vexpdif");
  return op;
}
const Op &simd_vabsdif() {
  static const Op op = Op::Get("tl.simd.vabsdif");
  return op;
}
const Op &simd_vsel() {
  static const Op op = Op::Get("tl.simd.vsel");
  return op;
}
const Op &simd_vselr() {
  static const Op op = Op::Get("tl.simd.vselr");
  return op;
}
const Op &simd_vmaxs() {
  static const Op op = Op::Get("tl.simd.vmaxs");
  return op;
}
const Op &simd_vmins() {
  static const Op op = Op::Get("tl.simd.vmins");
  return op;
}
const Op &simd_vmuls() {
  static const Op op = Op::Get("tl.simd.vmuls");
  return op;
}
const Op &simd_vadds() {
  static const Op op = Op::Get("tl.simd.vadds");
  return op;
}
const Op &simd_vshls() {
  static const Op op = Op::Get("tl.simd.vshls");
  return op;
}
const Op &simd_vshrs() {
  static const Op op = Op::Get("tl.simd.vshrs");
  return op;
}
const Op &simd_mem_bar() {
  static const Op op = Op::Get("tl.simd.mem_bar");
  return op;
}
const Op &ascend_set_flag() {
  static const Op op = Op::Get("tl.ascend_set_flag");
  return op;
}
const Op &ascend_wait_flag() {
  static const Op op = Op::Get("tl.ascend_wait_flag");
  return op;
}
const Op &ascend_threadfence() {
  static const Op op = Op::Get("tl.ascend_threadfence");
  return op;
}
const Op &ascend_cross_core_set_flag() {
  static const Op op = Op::Get("tl.ascend_cross_core_set_flag");
  return op;
}
const Op &ascend_cross_core_wait_flag() {
  static const Op op = Op::Get("tl.ascend_cross_core_wait_flag");
  return op;
}
const Op &ascend_copy_gm_to_ubuf() {
  static const Op op = Op::Get("tl.ascend_copy_gm_to_ubuf");
  return op;
}
const Op &ascend_set_copy_pad_value() {
  static const Op op = Op::Get("tl.ascend_set_copy_pad_value");
  return op;
}
const Op &ascend_copy_ubuf_to_gm() {
  static const Op op = Op::Get("tl.ascend_copy_ubuf_to_gm");
  return op;
}
const Op &ascend_copy_gm_to_cbuf() {
  static const Op op = Op::Get("tl.ascend_copy_gm_to_cbuf");
  return op;
}
const Op &ascend_fill_l1() {
  static const Op op = Op::Get("tl.ascend_fill_l1");
  return op;
}
const Op &ascend_load_cbuf_to_ca() {
  static const Op op = Op::Get("tl.ascend_load_cbuf_to_ca");
  return op;
}
const Op &ascend_load_cbuf_to_cb() {
  static const Op op = Op::Get("tl.ascend_load_cbuf_to_cb");
  return op;
}
const Op &ascend_load_ca_sf() {
  static const Op op = Op::Get("tl.ascend_load_ca_sf");
  return op;
}
const Op &ascend_load_cb_sf() {
  static const Op op = Op::Get("tl.ascend_load_cb_sf");
  return op;
}
const Op &ascend_copy_matrix_cc_to_ub() {
  static const Op op = Op::Get("tl.ascend_copy_matrix_cc_to_ub");
  return op;
}
const Op &ascend_copy_ubuf_to_cbuf() {
  static const Op op = Op::Get("tl.ascend_copy_ubuf_to_cbuf");
  return op;
}
const Op &ascend_nd2nz_scatter() {
  static const Op op = Op::Get("tl.ascend_nd2nz_scatter");
  return op;
}
const Op &ascend_nd2nz_post_copy() {
  static const Op op = Op::Get("tl.ascend_nd2nz_post_copy");
  return op;
}
const Op &ascend_copy_matrix_cc_to_gm() {
  static const Op op = Op::Get("tl.ascend_copy_matrix_cc_to_gm");
  return op;
}
const Op &ascend_mad() {
  static const Op op = Op::Get("tl.ascend_mad");
  return op;
}
const Op &ascend_mad_mx() {
  static const Op op = Op::Get("tl.ascend_mad_mx");
  return op;
}
const Op &ascend_gemm_l1() {
  static const Op op = Op::Get("tl.ascend_gemm_l1");
  return op;
}
const Op &ascend_blockscaled_gemm_l1() {
  static const Op op = Op::Get("tl.ascend_blockscaled_gemm_l1");
  return op;
}
const Op &ascend_read_gm_bypass_dcache() {
  static const Op op = Op::Get("tl.ascend_read_gm_bypass_dcache");
  return op;
}
const Op &ascend_write_gm_bypass_dcache() {
  static const Op op = Op::Get("tl.ascend_write_gm_bypass_dcache");
  return op;
}
const Op &ascend_get_buf() {
  static const Op op = Op::Get("tl.ascend_get_buf");
  return op;
}
const Op &ascend_rls_buf() {
  static const Op op = Op::Get("tl.ascend_rls_buf");
  return op;
}
const Op &ascend_set_hf32_mode() {
  static const Op op = Op::Get("tl.ascend_set_hf32_mode");
  return op;
}
const Op &ascend_set_mmad_direction() {
  static const Op op = Op::Get("tl.ascend_set_mmad_direction");
  return op;
}
const Op &ascend_set_atomic() {
  static const Op op = Op::Get("tl.ascend_set_atomic");
  return op;
}
const Op &ascend_set_atomic_none() {
  static const Op op = Op::Get("tl.ascend_set_atomic_none");
  return op;
}

namespace {
// These return-value instructions acquire one explicit destination when
// LegalizeSimdMerging turns an assignment into a void read-modify-write call.
// Keep their arity checked on both sides of that lowering boundary.
void DefineReturnValueSimdIntrinsic(const char *name, int operand_count) {
  OpDef def(name);
  DefineNativeIntrinsic(def, operand_count, operand_count + 1,
                        CallEffectKind::kOpaque);
  def.set_attr(
      kNativeIntrinsicExtraValidator,
      NativeIntrinsicExtraValidator([operand_count](Call call) {
        auto operands = NativeIntrinsicOperands(call.get());
        auto result = call->ty.as<PrimType>();
        bool is_void = result && result->IsVoid();
        if (static_cast<int>(operands.size()) == operand_count) {
          TVM_FFI_CHECK(result && !is_void, TypeError)
              << "Return-value SIMD form requires a primitive result";
          return;
        }
        const auto *mode = operands.back().as<StringImmObj>();
        TVM_FFI_CHECK(is_void && operands[0]->ty.as<PtrTypeObj>() && mode &&
                          mode->value == "MODE_MERGING",
                      ValueError)
            << "SIMD form with an explicit destination requires a void "
               "MODE_MERGING call and a typed destination pointer";
      }));
}
} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineNativeIntrinsic(OpDef("tl.conflict_hint"), 0, -1, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_pipe_barrier"), 1, 1,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.pset"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.pge"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.pand"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.por"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.pxor"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.pnot"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.psel"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.pld"), 2, 2, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.pst"), 3, 3, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vld"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vld2"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vsts"), 4, 4, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vst2"), 3, 3, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vadd"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vaddc"), 3, 3, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vsubc"), 3, 3, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vaddcs"), 4, 4, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vsubcs"), 4, 4, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vmull"), 3, 3, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vlrelu"), 3, 3, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vprelu"), 3, 3, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.update_mask"), 2, 2,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.ppack"), 2, 2, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.punpack"), 2, 2, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.pintlv"), 3, 3, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.pdintlv"), 3, 3, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vunpack"), 2, 2, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vusqz"), 1, 1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vsub"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vmul"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vmula"), 5, 5, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vmadd"), 5, 5, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vaxpy"), 5, 5, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vdiv"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vmax"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vmin"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vand"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vor"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineReturnValueSimdIntrinsic("tl.simd.vxor", 4);
  DefineNativeIntrinsic(OpDef("tl.simd.vshl"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vshr"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vexp"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vln"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vsqrt"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vabs"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vneg"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vrelu"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineReturnValueSimdIntrinsic("tl.simd.vnot", 3);
  DefineNativeIntrinsic(OpDef("tl.simd.vdup"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vdupv"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineReturnValueSimdIntrinsic("tl.simd.vcpadd", 3);
  DefineNativeIntrinsic(OpDef("tl.simd.vcadd"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vcmax"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vcmin"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vci"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vcmp"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vcmps"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vcgadd"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vcgmax"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vcgmin"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vsqz"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.dhistv2"), 4, 4, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.chistv2"), 4, 4, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vsstb"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vgatherb"), 0, -1,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vgather2"), 3, 3,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vscatter"), 4, 4,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vintlv"), 2, 2, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vdintlv"), 2, 2, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.pair_get"), 2, 2, CallEffectKind::kPure,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vpack"), 2, 2, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vcvt"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vexpdif"), 3, 3, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vabsdif"), 4, 4, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vsel"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vselr"), 2, 2, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vmaxs"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vmins"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vmuls"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vadds"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vshls"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.vshrs"), 0, -1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.simd.mem_bar"), 1, 1, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_set_flag"), 2, 2,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_wait_flag"), 2, 2,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_threadfence"), 0, 0,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_cross_core_set_flag"), 3, 3,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_cross_core_wait_flag"), 3, 3,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_copy_gm_to_ubuf"), 11, 11,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_set_copy_pad_value"), 1, 1,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_copy_ubuf_to_gm"), 8, 8,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_copy_gm_to_cbuf"), 11, 11,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_fill_l1"), 7, 7,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_load_cbuf_to_ca"), 9, 9,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_load_cbuf_to_cb"), 9, 9,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_load_ca_sf"), 8, 8,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_load_cb_sf"), 8, 8,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_copy_matrix_cc_to_ub"), 26, 26,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_copy_ubuf_to_cbuf"), 7, 7,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_nd2nz_scatter"), 7, 7,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_nd2nz_post_copy"), 7, 7,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_copy_matrix_cc_to_gm"), 25, 25,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_mad"), 10, 10, CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_mad_mx"), 10, 10,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_gemm_l1"), 11, 11,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_blockscaled_gemm_l1"), 17, 17,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_read_gm_bypass_dcache"), 1, 1,
                        CallEffectKind::kReadState,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_write_gm_bypass_dcache"), 2, 2,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_get_buf"), 3, 3,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_rls_buf"), 3, 3,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_set_hf32_mode"), 1, 1,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_set_mmad_direction"), 1, 1,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_set_atomic"), 2, 2,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
  DefineNativeIntrinsic(OpDef("tl.ascend_set_atomic_none"), 0, 0,
                        CallEffectKind::kOpaque,
                        NativeIntrinsicResult::kAnyScalar, false);
}
} // namespace tvm::tl
