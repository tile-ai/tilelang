/*!
 * \file ascend/op/builtin.cc
 * \brief Native Ascend intrinsic registrations and operand metadata.
 */
#include "builtin.h"
#include "ir/annotations.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/op/attr.h>

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
using PointerExpr = TypedExpr<PtrType>;

void CheckAscendResult(const CallObj *call, const Type &result) {
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  if (!call->ty_args.empty())
    TVM_FFI_CHECK(ffi::StructuralEqual()(call->ty_args[0], result), TypeError)
        << "Ascend result type disagrees with its type argument";
  TVM_FFI_CHECK(ffi::StructuralEqual()(call->ty, result), TypeError)
      << "Ascend result type disagrees with its operands";
}

template <typename T> void CheckAscendOperand(const Expr &expr) {
  expr.as_or_throw<T>();
  if constexpr (std::is_same_v<T, IntImm>) {
    const auto dtype = expr.as_or_throw<PrimExpr>().ty();
    TVM_FFI_CHECK(dtype.IsScalar() &&
                      dtype.MatchesCode(kDLInt, kDLUInt, kDLBool),
                  TypeError);
  } else if constexpr (std::is_same_v<T, IntExpr>)
    TVM_FFI_CHECK(expr.as_or_throw<IntExpr>().ty().IsScalar(), TypeError)
        << "Ascend index and control operands must be scalar integers";
  else if constexpr (std::is_same_v<T, PrimExpr>)
    TVM_FFI_CHECK(!expr.as_or_throw<PrimExpr>().ty().IsVoid(), TypeError);
}

PrimType CheckSimdVector(const Expr &expr) {
  const auto dtype = expr.as_or_throw<PrimExpr>().ty();
  TVM_FFI_CHECK(!dtype.IsVoid() && !dtype.IsScalar() &&
                    dtype.bits() * dtype.lanes() == 2048,
                TypeError)
      << "SIMD value must be a full 2048-bit vector register";
  return dtype;
}

void CheckSimdMask(const Expr &expr) {
  TVM_FFI_CHECK_EQ(expr.as_or_throw<PrimExpr>().ty(), PrimType::Bool(256),
                   TypeError)
      << "SIMD mask must have type boolx256";
}

void CheckSimdMode(const Expr &expr) {
  const auto mode = expr.as_or_throw<StringImm>()->value;
  TVM_FFI_CHECK(mode == "MODE_ZEROING" || mode == "MODE_MERGING" ||
                    mode == "MODE_STORED",
                ValueError)
      << "Invalid SIMD inactive-lane mode";
}

PrimType ExplicitSimdVector(const CallObj *call) {
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  const Type result = call->ty_args.empty() ? call->ty : call->ty_args[0];
  TVM_FFI_CHECK(!result.as<MissingTypeObj>(), TypeError)
      << "SIMD operation requires an explicit result type";
  const auto dtype = result.as_or_throw<PrimType>();
  TVM_FFI_CHECK(!dtype.IsVoid() && !dtype.IsScalar() &&
                    dtype.bits() * dtype.lanes() == 2048,
                TypeError)
      << "SIMD result must be a full vector register";
  return dtype;
}

template <typename... Operands>
void ValidateAscendStatement(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), sizeof...(Operands), ValueError)
      << "Ascend instruction has the wrong number of runtime operands";
  size_t index = 0;
  (CheckAscendOperand<Operands>(args[index++]), ...);
  CheckAscendResult(call, PrimType::Void());
}

// Sync/configuration, MTE DMA, and Cube instructions have real void results.
// Each registration below declares its own typed fields; no argc attributes or
// untyped result-category registry is needed.
template <typename... Operands>
void DefineAscendStatement(const char *name, sig::arg<Operands>... operands) {
  OpDef(name)
      .set_validator(
          ffi::reflection::NativeFunctionView<void(const CallObj *)>::
              FromNative<&ValidateAscendStatement<Operands...>>())
      .signature(operands..., sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

template <typename... Operands>
void ValidateSimdPredicate(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), sizeof...(Operands), ValueError)
      << "SIMD predicate instruction has the wrong number of runtime operands";
  size_t index = 0;
  auto check = [&]<typename T>() {
    const auto &expr = args[index++];
    CheckAscendOperand<T>(expr);
    if constexpr (std::is_same_v<T, PrimExpr>)
      CheckSimdMask(expr);
  };
  (check.template operator()<Operands>(), ...);
  CheckAscendResult(call, PrimType::Bool(256));
}

template <typename... Operands>
void DefineSimdPredicate(const char *name, sig::arg<Operands>... operands) {
  OpDef(name)
      .set_validator(
          ffi::reflection::NativeFunctionView<void(const CallObj *)>::
              FromNative<&ValidateSimdPredicate<Operands...>>())
      .signature(operands..., sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Bool(256))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

// Arithmetic/cross-lane operators either produce a vector or update an
// explicit destination inserted by LegalizeSimdMerging. The second form adds
// exactly one pointer and has a void result; annotations remain outside both.
template <int N, int MaskIndex, bool ScalarRhs, bool WidenSum>
Type InferSimdArithmetic(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK(args.size() == N || args.size() == N + 1, ValueError)
      << "SIMD arithmetic has the wrong number of runtime operands";
  const bool merging = args.size() == N + 1;
  const auto source = CheckSimdVector(args[merging ? 1 : 0]);
  const auto value_type =
      WidenSum && source.bits() == 16 && source.MatchesCode(kDLInt, kDLUInt)
          ? PrimType(source.code(), 32, 64)
          : source;
  if (merging) {
    const auto destination = args[0].as_or_throw<PointerExpr>().ty();
    TVM_FFI_CHECK_EQ(destination->element_type.as_or_throw<PrimType>(),
                     value_type, TypeError)
        << "SIMD merging destination type must match the arithmetic value";
    TVM_FFI_CHECK_EQ(args.back().as_or_throw<StringImm>()->value,
                     "MODE_MERGING", ValueError)
        << "SIMD destination form requires MODE_MERGING";
    return PrimType::Void();
  }
  return value_type;
}

template <int N, int MaskIndex, bool ScalarRhs, bool WidenSum>
void ValidateSimdArithmetic(const CallObj *call) {
  const Type result =
      InferSimdArithmetic<N, MaskIndex, ScalarRhs, WidenSum>(call);
  const auto args = DecodeCallOperands(call);
  const size_t offset = args.size() == N + 1 ? 1 : 0;
  const auto source = CheckSimdVector(args[offset]);
  if constexpr (N == 4 && MaskIndex == 2) {
    if constexpr (ScalarRhs) {
      const auto scalar = args[offset + 1].as_or_throw<PrimExpr>().ty();
      TVM_FFI_CHECK(scalar.IsScalar() && !scalar.IsVoid(), TypeError);
    } else {
      TVM_FFI_CHECK_EQ(CheckSimdVector(args[offset + 1]), source, TypeError)
          << "SIMD arithmetic source vectors must have matching types";
    }
  }
  if constexpr (N == 4 && MaskIndex == 1)
    args[offset + 2].as_or_throw<StringImm>(); // vdupv lane position
  CheckSimdMask(args[offset + MaskIndex]);
  CheckSimdMode(args.back());
  CheckAscendResult(call, result);
}

template <int N, int MaskIndex, bool ScalarRhs = false, bool WidenSum = false>
void DefineSimdArithmetic(const char *name) {
  OpDef def(name);
  def.set_validator(
      ffi::reflection::NativeFunctionView<void(const CallObj *)>::FromNative<
          &ValidateSimdArithmetic<N, MaskIndex, ScalarRhs, WidenSum>>());
  // In the lowered form this first operand is a pointer. The source vector,
  // optional second source/position, mask, and mode are checked at their actual
  // offset by the owning validator rather than mistaking a mask for metadata.
  if constexpr (N == 3)
    def.signature(sig::arg<Expr>("source_or_merging_destination"),
                  sig::arg<PrimExpr>("source_or_mask"),
                  sig::arg<Expr>("mask_or_mode"),
                  sig::var_args<Expr>("mode_and_annotations"),
                  sig::var_ty_args<PrimType>("result_type"));
  else
    def.signature(sig::arg<Expr>("source_or_merging_destination"),
                  sig::arg<PrimExpr>("source_or_rhs_or_mask"),
                  sig::arg<Expr>("rhs_or_mask_or_position"),
                  sig::arg<Expr>("mask_or_position_or_mode"),
                  sig::var_args<Expr>("mode_and_annotations"),
                  sig::var_ty_args<PrimType>("result_type"));
  def.set_native_fn_attr<
         &InferSimdArithmetic<N, MaskIndex, ScalarRhs, WidenSum>>(
         tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

template <int MaskIndex, typename... Operands>
Type InferSimdRegisterTransform(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), sizeof...(Operands), ValueError)
      << "SIMD register transform has the wrong number of runtime operands";
  return CheckSimdVector(args[0]);
}

template <int MaskIndex, typename... Operands>
void ValidateSimdRegisterTransform(const CallObj *call) {
  const auto dtype = InferSimdRegisterTransform<MaskIndex, Operands...>(call)
                         .template as_or_throw<PrimType>();
  const auto args = DecodeCallOperands(call);
  size_t index = 0;
  (CheckAscendOperand<Operands>(args[index++]), ...);
  if constexpr (MaskIndex >= 0)
    CheckSimdMask(args[MaskIndex]);
  if (call->op.same_as(simd_vlrelu())) {
    TVM_FFI_CHECK(args[1].as_or_throw<PrimExpr>().ty().IsScalar(), TypeError);
  } else if (call->op.same_as(simd_vselr())) {
    const auto indices = CheckSimdVector(args[1]);
    TVM_FFI_CHECK(indices.MatchesCode(kDLUInt), TypeError)
        << "vselr requires unsigned integer vector indices";
  } else {
    TVM_FFI_CHECK_EQ(CheckSimdVector(args[1]), dtype, TypeError)
        << "SIMD register inputs must have matching types";
  }
  if (call->op.same_as(simd_vaddc()) || call->op.same_as(simd_vsubc()) ||
      call->op.same_as(simd_vaddcs()) || call->op.same_as(simd_vsubcs()) ||
      call->op.same_as(simd_vmull()))
    TVM_FFI_CHECK(dtype.MatchesCode(kDLInt, kDLUInt) && dtype.bits() == 32,
                  TypeError)
        << "SIMD carry/multiply pair requires int32 or uint32 vectors";
  if (call->op.same_as(simd_vaddcs()) || call->op.same_as(simd_vsubcs()))
    CheckSimdMask(args[2]);
  CheckAscendResult(call, dtype);
}

template <int MaskIndex, typename... Operands>
void DefineSimdRegisterTransform(const char *name,
                                 sig::arg<Operands>... operands) {
  OpDef(name)
      .set_validator(
          ffi::reflection::NativeFunctionView<void(const CallObj *)>::
              FromNative<
                  &ValidateSimdRegisterTransform<MaskIndex, Operands...>>())
      .signature(operands..., sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .template set_native_fn_attr<
          &InferSimdRegisterTransform<MaskIndex, Operands...>>(
          tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

template <bool Scalar> void ValidateSimdCompare(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 4U, ValueError)
      << "SIMD compare expects four runtime operands";
  const auto source = CheckSimdVector(args[0]);
  if constexpr (Scalar)
    TVM_FFI_CHECK(args[1].as_or_throw<PrimExpr>().ty().IsScalar(), TypeError);
  else
    TVM_FFI_CHECK_EQ(CheckSimdVector(args[1]), source, TypeError);
  CheckSimdMask(args[2]);
  args[3].as_or_throw<StringImm>();
  CheckAscendResult(call, PrimType::Bool(256));
}

template <bool Scalar> void DefineSimdCompare(const char *name) {
  OpDef(name)
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateSimdCompare<Scalar>>())
      .signature(sig::arg<PrimExpr>("source"),
                 sig::arg<PrimExpr>(Scalar ? "scalar" : "other"),
                 sig::arg<PrimExpr>("mask"), sig::arg<StringImm>("comparison"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Bool(256))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

Type InferSimdBroadcast(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK(args.size() == 3 || args.size() == 4, ValueError)
      << "SIMD scalar broadcast expects three or four runtime operands";
  return args.size() == 4 ? Type(PrimType::Void())
                          : Type(ExplicitSimdVector(call));
}

void ValidateSimdBroadcast(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  const Type result = InferSimdBroadcast(call);
  const size_t offset = args.size() == 4 ? 1 : 0;
  if (offset) {
    const auto destination = args[0].as_or_throw<PointerExpr>().ty();
    const auto dtype = destination->element_type.as_or_throw<PrimType>();
    TVM_FFI_CHECK(!dtype.IsVoid() && dtype.bits() * dtype.lanes() == 2048,
                  TypeError);
    TVM_FFI_CHECK_EQ(args.back().as_or_throw<StringImm>()->value,
                     "MODE_MERGING", ValueError);
  }
  TVM_FFI_CHECK(args[offset].as_or_throw<PrimExpr>().ty().IsScalar(),
                TypeError);
  CheckSimdMask(args[offset + 1]);
  CheckSimdMode(args.back());
  CheckAscendResult(call, result);
}

Type InferSimdExplicitVector(const CallObj *call) {
  return ExplicitSimdVector(call);
}

void ValidateSimdRamp(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 2U, ValueError)
      << "SIMD index ramp expects two runtime operands";
  TVM_FFI_CHECK(args[0].as_or_throw<PrimExpr>().ty().IsScalar(), TypeError);
  args[1].as_or_throw<StringImm>();
  CheckAscendResult(call, ExplicitSimdVector(call));
}

void ValidateSimdUnsqueeze(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 1U, ValueError)
      << "SIMD unsqueeze expects one runtime operand";
  CheckSimdMask(args[0]);
  const auto dtype = ExplicitSimdVector(call);
  TVM_FFI_CHECK(dtype.MatchesCode(kDLInt, kDLUInt), TypeError);
  CheckAscendResult(call, dtype);
}

template <bool Widen> Type InferSimdPack(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 2U, ValueError)
      << "SIMD pack/unpack expects two runtime operands";
  const auto source = CheckSimdVector(args[0]);
  TVM_FFI_CHECK(source.MatchesCode(kDLInt, kDLUInt, kDLFloat), TypeError);
  TVM_FFI_CHECK(Widen ? (source.bits() == 8 || source.bits() == 16)
                      : (source.bits() == 16 || source.bits() == 32),
                TypeError);
  return PrimType(source.code(), Widen ? source.bits() * 2 : source.bits() / 2,
                  Widen ? source.lanes() / 2 : source.lanes() * 2);
}

template <bool Widen> void ValidateSimdPack(const CallObj *call) {
  const Type result = InferSimdPack<Widen>(call);
  DecodeCallOperands(call)[1].as_or_throw<StringImm>();
  CheckAscendResult(call, result);
}

template <bool Widen> void DefineSimdPack(const char *name) {
  OpDef(name)
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateSimdPack<Widen>>())
      .signature(sig::arg<PrimExpr>("source"), sig::arg<StringImm>("part"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .template set_native_fn_attr<&InferSimdPack<Widen>>(
          tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

Type InferSimdConversion(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK(!args.empty(), ValueError)
      << "SIMD conversion requires runtime operands";
  const bool merging = args[0]->ty.as<PtrTypeObj>();
  return merging ? Type(PrimType::Void()) : Type(ExplicitSimdVector(call));
}

void ValidateSimdConversion(const CallObj *call) {
  const Type result = InferSimdConversion(call);
  const auto args = DecodeCallOperands(call);
  const size_t offset = args[0]->ty.as<PtrTypeObj>() ? 1 : 0;
  TVM_FFI_CHECK(args.size() >= offset + 4 && args.size() <= offset + 6,
                ValueError)
      << "SIMD conversion expects source, mask, and two to four option runtime "
         "operands";
  CheckSimdVector(args[offset]);
  CheckSimdMask(args[offset + 1]);
  for (size_t i = offset + 2; i < args.size(); ++i)
    args[i].as_or_throw<StringImm>();
  CheckSimdMode(args.back());
  if (offset) {
    const auto destination = args[0].as_or_throw<PointerExpr>().ty();
    const auto dtype = destination->element_type.as_or_throw<PrimType>();
    TVM_FFI_CHECK(!dtype.IsVoid() && dtype.bits() * dtype.lanes() == 2048,
                  TypeError);
    TVM_FFI_CHECK_EQ(args.back().as_or_throw<StringImm>()->value,
                     "MODE_MERGING", ValueError);
  }
  CheckAscendResult(call, result);
}

template <bool Paired> Type InferSimdLoad(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK(args.size() == 2 || args.size() == 3, ValueError)
      << "SIMD vector load expects two or three runtime operands";
  const auto pointer = args[0].as_or_throw<PointerExpr>().ty();
  auto element = pointer->element_type.as_or_throw<PrimType>();
  TVM_FFI_CHECK(element.IsScalar() && !element.IsVoid(), TypeError)
      << "SIMD vector load requires a typed scalar element pointer";
  const auto dist = args[1].as_or_throw<StringImm>()->value;
  if constexpr (!Paired) {
    int broadcast_bits = dist == "BRC_B8"    ? 8
                         : dist == "BRC_B16" ? 16
                         : dist == "BRC_B32" ? 32
                                             : 0;
    if (broadcast_bits && broadcast_bits != element.bits())
      element = PrimType::UInt(broadcast_bits);
  }
  TVM_FFI_CHECK(element.bits() > 0 && 2048 % element.bits() == 0, TypeError);
  return element.WithLanes(2048 / element.bits());
}

template <bool Paired> void ValidateSimdLoad(const CallObj *call) {
  const Type result = InferSimdLoad<Paired>(call);
  const auto args = DecodeCallOperands(call);
  if (args.size() == 3) {
    const auto increment = args[2].as_or_throw<IntExpr>().ty();
    TVM_FFI_CHECK(increment.IsScalar(), TypeError);
    if constexpr (!Paired)
      TVM_FFI_CHECK_EQ(increment, PrimType::Int(32), TypeError)
          << "vld post-update increment must be int32";
  }
  CheckAscendResult(call, result);
}

template <bool Paired> void DefineSimdLoad(const char *name) {
  OpDef(name)
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateSimdLoad<Paired>>())
      .signature(
          sig::arg<PointerExpr>("pointer"), sig::arg<StringImm>("distribution"),
          sig::var_args<Expr>("offset_or_post_increment_and_annotations"),
          sig::var_ty_args<PrimType>("result_type"))
      .template set_native_fn_attr<&InferSimdLoad<Paired>>(
          tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

template <bool Elements> Type InferSimdGather(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK(args.size() == 3 || (!Elements && args.size() == 2), ValueError)
      << "SIMD gather has the wrong number of runtime operands";
  auto element = args[0]
                     .as_or_throw<PointerExpr>()
                     .ty()
                     ->element_type.as_or_throw<PrimType>();
  TVM_FFI_CHECK(element.IsScalar() && !element.IsVoid(), TypeError);
  if constexpr (Elements)
    if (element.bits() == 8 && element.MatchesCode(kDLInt, kDLUInt))
      element = PrimType(element.code(), 16);
  return element.WithLanes(2048 / element.bits());
}

template <bool Elements> void ValidateSimdGather(const CallObj *call) {
  const Type result = InferSimdGather<Elements>(call);
  const auto args = DecodeCallOperands(call);
  const auto indices = CheckSimdVector(args[1]);
  TVM_FFI_CHECK(indices.MatchesCode(kDLInt, kDLUInt), TypeError);
  if (args.size() == 3)
    CheckSimdMask(args[2]);
  CheckAscendResult(call, result);
}

template <bool Elements> void DefineSimdGather(const char *name) {
  OpDef(name)
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateSimdGather<Elements>>())
      .signature(sig::arg<PointerExpr>("base"), sig::arg<PrimExpr>("indices"),
                 sig::var_args<Expr>("mask_and_annotations"),
                 sig::var_ty_args<PrimType>("result_type"))
      .template set_native_fn_attr<&InferSimdGather<Elements>>(
          tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

Type InferSimdScatterBlocks(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK(args.size() == 4 || args.size() == 5, ValueError)
      << "vsstb expects four or five runtime operands";
  return args.size() == 5 ? Type(args[1].as_or_throw<PointerExpr>().ty())
                          : Type(PrimType::Void());
}

void ValidateSimdScatterBlocks(const CallObj *call) {
  const Type result = InferSimdScatterBlocks(call);
  const auto args = DecodeCallOperands(call);
  const auto source = CheckSimdVector(args[0]);
  const auto pointer = args[1].as_or_throw<PointerExpr>().ty();
  TVM_FFI_CHECK_EQ(pointer->element_type.as_or_throw<PrimType>(),
                   source.WithLanes(1), TypeError);
  CheckAscendOperand<IntExpr>(args[2]);
  CheckSimdMask(args[3]);
  if (args.size() == 5)
    TVM_FFI_CHECK_EQ(args[4].as_or_throw<StringImm>()->value, "POST_UPDATE",
                     ValueError);
  CheckAscendResult(call, result);
}

Type InferSimdPairComponent(const CallObj *call) {
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  const Type result = call->ty_args.empty() ? call->ty : call->ty_args[0];
  const auto primitive = result.as<PrimType>();
  TVM_FFI_CHECK(result.as<PtrType>() || (primitive && !primitive->IsVoid()),
                TypeError)
      << "SIMD pair extraction requires an explicit result type for its "
         "component";
  return result;
}

void ValidateSimdPairComponent(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 2U, ValueError)
      << "SIMD pair extraction expects two runtime operands";
  const auto pair = args[0].as_or_throw<PrimExpr>().ty();
  TVM_FFI_CHECK(!pair.IsVoid(), TypeError);
  const auto index = args[1].as_or_throw<IntImm>()->value.as<int64_t>();
  TVM_FFI_CHECK(index && (*index == 0 || *index == 1), ValueError)
      << "SIMD pair index must be 0 or 1";
  CheckAscendResult(call, InferSimdPairComponent(call));
}

enum class SimdStore {
  Predicate,
  Vector,
  Interleave,
  Scatter,
  Accumulate,
  Histogram
};

template <SimdStore Kind, typename... Operands>
void ValidateSimdStore(const CallObj *call) {
  ValidateAscendStatement<Operands...>(call);
  const auto args = DecodeCallOperands(call);
  if constexpr (Kind == SimdStore::Predicate) {
    CheckSimdMask(args[1]);
  } else if constexpr (Kind == SimdStore::Vector) {
    CheckSimdVector(args[1]);
    CheckSimdMask(args[2]);
  } else if constexpr (Kind == SimdStore::Interleave) {
    const auto dtype = CheckSimdVector(args[1]);
    TVM_FFI_CHECK_EQ(CheckSimdVector(args[2]), dtype, TypeError);
    TVM_FFI_CHECK_EQ(args[0]
                         .as_or_throw<PointerExpr>()
                         .ty()
                         ->element_type.as_or_throw<PrimType>(),
                     dtype.WithLanes(1), TypeError);
  } else if constexpr (Kind == SimdStore::Scatter) {
    CheckSimdVector(args[0]);
    TVM_FFI_CHECK(CheckSimdVector(args[2]).MatchesCode(kDLInt, kDLUInt),
                  TypeError);
    CheckSimdMask(args[3]);
  } else if constexpr (Kind == SimdStore::Accumulate) {
    const auto dtype = CheckSimdVector(args[1]);
    TVM_FFI_CHECK_EQ(args[0]
                         .as_or_throw<PointerExpr>()
                         .ty()
                         ->element_type.as_or_throw<PrimType>(),
                     dtype, TypeError);
    if (call->op.same_as(simd_vaxpy()))
      TVM_FFI_CHECK(args[2].as_or_throw<PrimExpr>().ty().IsScalar(), TypeError);
    else
      TVM_FFI_CHECK_EQ(CheckSimdVector(args[2]), dtype, TypeError);
    CheckSimdMask(args[3]);
    CheckSimdMode(args[4]);
  } else {
    TVM_FFI_CHECK_EQ(args[0]
                         .as_or_throw<PointerExpr>()
                         .ty()
                         ->element_type.as_or_throw<PrimType>(),
                     PrimType::UInt(16, 128), TypeError);
    TVM_FFI_CHECK_EQ(CheckSimdVector(args[1]), PrimType::UInt(8, 256),
                     TypeError);
    CheckSimdMask(args[2]);
    const auto bin = args[3].as_or_throw<IntImm>()->value.as<int64_t>();
    TVM_FFI_CHECK(bin && (*bin == 0 || *bin == 1), ValueError);
  }
}

template <SimdStore Kind, typename... Operands>
void DefineSimdStore(const char *name, sig::arg<Operands>... operands) {
  OpDef(name)
      .set_validator(
          ffi::reflection::NativeFunctionView<void(const CallObj *)>::
              FromNative<&ValidateSimdStore<Kind, Operands...>>())
      .signature(operands..., sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

Type InferBypassRead(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 1U, ValueError)
      << "Ascend bypass read expects one runtime operand";
  const auto result = args[0]
                          .as_or_throw<PointerExpr>()
                          .ty()
                          ->element_type.as_or_throw<PrimType>();
  TVM_FFI_CHECK(result.IsScalar() && !result.IsVoid(), TypeError);
  return result;
}
void ValidateBypassRead(const CallObj *call) {
  CheckAscendResult(call, InferBypassRead(call));
}

Type InferConflictHint(const CallObj *call) {
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  const Type result = !call->ty_args.empty()           ? call->ty_args[0]
                      : !call->ty.as<MissingTypeObj>() ? call->ty
                                                       : Type(PrimType::Void());
  TVM_FFI_CHECK(result.as<PrimType>() || result.as<PtrType>(), TypeError);
  return result;
}
void ValidateConflictHint(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 6U, ValueError)
      << "Conflict hint expects six runtime operands";
  // Scheduling owns the two storage/region reference contracts and their
  // scope-dependent meaning. The marker owns its scalar control fields.
  args[2].as_or_throw<IntImm>();
  args[3].as_or_throw<IntImm>();
  args[4].as_or_throw<StringImm>();
  args[5].as_or_throw<IntImm>();
  CheckAscendResult(call, InferConflictHint(call));
}
} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.conflict_hint")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateConflictHint>())
      .signature(sig::arg<Expr>("left_region"), sig::arg<Expr>("right_region"),
                 sig::arg<IntImm>("level"), sig::arg<IntImm>("cross"),
                 sig::arg<StringImm>("group"), sig::arg<IntImm>("is_conflict"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<Type>("result_type"))
      .set_native_fn_attr<&InferConflictHint>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_attr(tirx::op_attr::kVectorizable, false);
  DefineAscendStatement("tl.ascend_pipe_barrier", sig::arg<StringImm>("pipe"));
  DefineSimdPredicate("tl.simd.pset", sig::arg<IntImm>("element_bits"),
                      sig::arg<StringImm>("distribution"));
  DefineSimdPredicate("tl.simd.pge", sig::arg<IntImm>("element_bits"),
                      sig::arg<StringImm>("distribution"));
  DefineSimdPredicate("tl.simd.pand", sig::arg<PrimExpr>("left"),
                      sig::arg<PrimExpr>("right"), sig::arg<PrimExpr>("mask"));
  DefineSimdPredicate("tl.simd.por", sig::arg<PrimExpr>("left"),
                      sig::arg<PrimExpr>("right"), sig::arg<PrimExpr>("mask"));
  DefineSimdPredicate("tl.simd.pxor", sig::arg<PrimExpr>("left"),
                      sig::arg<PrimExpr>("right"), sig::arg<PrimExpr>("mask"));
  DefineSimdPredicate("tl.simd.pnot", sig::arg<PrimExpr>("source"),
                      sig::arg<PrimExpr>("mask"));
  DefineSimdPredicate("tl.simd.psel", sig::arg<PrimExpr>("left"),
                      sig::arg<PrimExpr>("right"), sig::arg<PrimExpr>("mask"));
  DefineSimdPredicate("tl.simd.pld", sig::arg<PointerExpr>("pointer"),
                      sig::arg<StringImm>("distribution"));
  DefineSimdStore<SimdStore::Predicate>(
      "tl.simd.pst", sig::arg<PointerExpr>("pointer"),
      sig::arg<PrimExpr>("source"), sig::arg<StringImm>("distribution"));
  DefineSimdLoad<false>("tl.simd.vld");
  DefineSimdLoad<true>("tl.simd.vld2");
  DefineSimdStore<SimdStore::Vector>(
      "tl.simd.vsts", sig::arg<PointerExpr>("pointer"),
      sig::arg<PrimExpr>("source"), sig::arg<PrimExpr>("mask"),
      sig::arg<StringImm>("distribution"));
  DefineSimdStore<SimdStore::Interleave>(
      "tl.simd.vst2", sig::arg<PointerExpr>("pointer"),
      sig::arg<PrimExpr>("left"), sig::arg<PrimExpr>("right"));
  DefineSimdArithmetic<4, 2>("tl.simd.vadd");
  DefineSimdRegisterTransform<2>("tl.simd.vaddc", sig::arg<PrimExpr>("left"),
                                 sig::arg<PrimExpr>("right"),
                                 sig::arg<PrimExpr>("mask"));
  DefineSimdRegisterTransform<2>("tl.simd.vsubc", sig::arg<PrimExpr>("left"),
                                 sig::arg<PrimExpr>("right"),
                                 sig::arg<PrimExpr>("mask"));
  DefineSimdRegisterTransform<3>(
      "tl.simd.vaddcs", sig::arg<PrimExpr>("left"), sig::arg<PrimExpr>("right"),
      sig::arg<PrimExpr>("carry"), sig::arg<PrimExpr>("mask"));
  DefineSimdRegisterTransform<3>(
      "tl.simd.vsubcs", sig::arg<PrimExpr>("left"), sig::arg<PrimExpr>("right"),
      sig::arg<PrimExpr>("carry"), sig::arg<PrimExpr>("mask"));
  DefineSimdRegisterTransform<2>("tl.simd.vmull", sig::arg<PrimExpr>("left"),
                                 sig::arg<PrimExpr>("right"),
                                 sig::arg<PrimExpr>("mask"));
  DefineSimdRegisterTransform<2>("tl.simd.vlrelu", sig::arg<PrimExpr>("source"),
                                 sig::arg<PrimExpr>("alpha"),
                                 sig::arg<PrimExpr>("mask"));
  DefineSimdRegisterTransform<2>("tl.simd.vprelu", sig::arg<PrimExpr>("left"),
                                 sig::arg<PrimExpr>("right"),
                                 sig::arg<PrimExpr>("mask"));
  DefineSimdPredicate("tl.simd.update_mask", sig::arg<IntExpr>("active_lanes"),
                      sig::arg<IntImm>("element_bits"));
  DefineSimdPredicate("tl.simd.ppack", sig::arg<PrimExpr>("source"),
                      sig::arg<StringImm>("part"));
  DefineSimdPredicate("tl.simd.punpack", sig::arg<PrimExpr>("source"),
                      sig::arg<StringImm>("part"));
  DefineSimdPredicate("tl.simd.pintlv", sig::arg<PrimExpr>("left"),
                      sig::arg<PrimExpr>("right"),
                      sig::arg<IntImm>("element_bits"));
  DefineSimdPredicate("tl.simd.pdintlv", sig::arg<PrimExpr>("left"),
                      sig::arg<PrimExpr>("right"),
                      sig::arg<IntImm>("element_bits"));
  DefineSimdPack<true>("tl.simd.vunpack");
  OpDef("tl.simd.vusqz")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateSimdUnsqueeze>())
      .signature(sig::arg<PrimExpr>("mask"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_native_fn_attr<&InferSimdExplicitVector>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  DefineSimdArithmetic<4, 2>("tl.simd.vsub");
  DefineSimdArithmetic<4, 2>("tl.simd.vmul");
  DefineSimdStore<SimdStore::Accumulate>(
      "tl.simd.vmula", sig::arg<PointerExpr>("destination"),
      sig::arg<PrimExpr>("left"), sig::arg<PrimExpr>("right"),
      sig::arg<PrimExpr>("mask"), sig::arg<StringImm>("mode"));
  DefineSimdStore<SimdStore::Accumulate>(
      "tl.simd.vmadd", sig::arg<PointerExpr>("destination"),
      sig::arg<PrimExpr>("left"), sig::arg<PrimExpr>("right"),
      sig::arg<PrimExpr>("mask"), sig::arg<StringImm>("mode"));
  DefineSimdStore<SimdStore::Accumulate>(
      "tl.simd.vaxpy", sig::arg<PointerExpr>("destination"),
      sig::arg<PrimExpr>("source"), sig::arg<PrimExpr>("scalar"),
      sig::arg<PrimExpr>("mask"), sig::arg<StringImm>("mode"));
  DefineSimdArithmetic<4, 2>("tl.simd.vdiv");
  DefineSimdArithmetic<4, 2>("tl.simd.vmax");
  DefineSimdArithmetic<4, 2>("tl.simd.vmin");
  DefineSimdArithmetic<4, 2>("tl.simd.vand");
  DefineSimdArithmetic<4, 2>("tl.simd.vor");
  DefineSimdArithmetic<4, 2>("tl.simd.vxor");
  DefineSimdArithmetic<4, 2>("tl.simd.vshl");
  DefineSimdArithmetic<4, 2>("tl.simd.vshr");
  DefineSimdArithmetic<3, 1>("tl.simd.vexp");
  DefineSimdArithmetic<3, 1>("tl.simd.vln");
  DefineSimdArithmetic<3, 1>("tl.simd.vsqrt");
  DefineSimdArithmetic<3, 1>("tl.simd.vabs");
  DefineSimdArithmetic<3, 1>("tl.simd.vneg");
  DefineSimdArithmetic<3, 1>("tl.simd.vrelu");
  DefineSimdArithmetic<3, 1>("tl.simd.vnot");
  OpDef("tl.simd.vdup")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateSimdBroadcast>())
      .signature(sig::arg<Expr>("scalar_or_merging_destination"),
                 sig::arg<PrimExpr>("scalar_or_mask"),
                 sig::arg<Expr>("mask_or_mode"),
                 sig::var_args<Expr>("mode_and_annotations"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_native_fn_attr<&InferSimdBroadcast>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  DefineSimdArithmetic<4, 1>("tl.simd.vdupv");
  DefineSimdArithmetic<3, 1>("tl.simd.vcpadd");
  DefineSimdArithmetic<3, 1, false, true>("tl.simd.vcadd");
  DefineSimdArithmetic<3, 1>("tl.simd.vcmax");
  DefineSimdArithmetic<3, 1>("tl.simd.vcmin");
  OpDef("tl.simd.vci")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateSimdRamp>())
      .signature(sig::arg<PrimExpr>("index"), sig::arg<StringImm>("order"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_native_fn_attr<&InferSimdExplicitVector>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  DefineSimdCompare<false>("tl.simd.vcmp");
  DefineSimdCompare<true>("tl.simd.vcmps");
  DefineSimdArithmetic<3, 1>("tl.simd.vcgadd");
  DefineSimdArithmetic<3, 1>("tl.simd.vcgmax");
  DefineSimdArithmetic<3, 1>("tl.simd.vcgmin");
  DefineSimdArithmetic<3, 1>("tl.simd.vsqz");
  DefineSimdStore<SimdStore::Histogram>(
      "tl.simd.dhistv2", sig::arg<PointerExpr>("destination"),
      sig::arg<PrimExpr>("source"), sig::arg<PrimExpr>("mask"),
      sig::arg<IntImm>("bin"));
  DefineSimdStore<SimdStore::Histogram>(
      "tl.simd.chistv2", sig::arg<PointerExpr>("destination"),
      sig::arg<PrimExpr>("source"), sig::arg<PrimExpr>("mask"),
      sig::arg<IntImm>("bin"));
  OpDef("tl.simd.vsstb")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateSimdScatterBlocks>())
      .signature(sig::arg<PrimExpr>("source"), sig::arg<PointerExpr>("base"),
                 sig::arg<IntExpr>("stride"), sig::arg<PrimExpr>("mask"),
                 sig::var_args<Expr>("optional_operands_and_annotations"),
                 sig::var_ty_args<Type>("result_type"))
      .set_native_fn_attr<&InferSimdScatterBlocks>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  DefineSimdGather<false>("tl.simd.vgatherb");
  DefineSimdGather<true>("tl.simd.vgather2");
  DefineSimdStore<SimdStore::Scatter>(
      "tl.simd.vscatter", sig::arg<PrimExpr>("source"),
      sig::arg<PointerExpr>("base"), sig::arg<PrimExpr>("indices"),
      sig::arg<PrimExpr>("mask"));
  DefineSimdRegisterTransform<-1>("tl.simd.vintlv", sig::arg<PrimExpr>("left"),
                                  sig::arg<PrimExpr>("right"));
  DefineSimdRegisterTransform<-1>("tl.simd.vdintlv", sig::arg<PrimExpr>("left"),
                                  sig::arg<PrimExpr>("right"));
  OpDef("tl.simd.pair_get")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateSimdPairComponent>())
      .signature(sig::arg<PrimExpr>("pair"), sig::arg<IntImm>("index"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<Type>("result_type"))
      .set_native_fn_attr<&InferSimdPairComponent>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_attr(tirx::op_attr::kVectorizable, false);
  DefineSimdPack<false>("tl.simd.vpack");
  OpDef("tl.simd.vcvt")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateSimdConversion>())
      .signature(sig::arg<Expr>("source_or_merging_destination"),
                 sig::arg<PrimExpr>("source_or_mask"),
                 sig::arg<Expr>("mask_or_first_option"),
                 sig::arg<StringImm>("conversion_option"),
                 sig::var_args<Expr>("conversion_options_and_annotations"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_native_fn_attr<&InferSimdConversion>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  DefineSimdRegisterTransform<2>("tl.simd.vexpdif", sig::arg<PrimExpr>("left"),
                                 sig::arg<PrimExpr>("right"),
                                 sig::arg<PrimExpr>("mask"));
  DefineSimdArithmetic<4, 2>("tl.simd.vabsdif");
  DefineSimdRegisterTransform<2>("tl.simd.vsel", sig::arg<PrimExpr>("left"),
                                 sig::arg<PrimExpr>("right"),
                                 sig::arg<PrimExpr>("mask"));
  DefineSimdRegisterTransform<-1>("tl.simd.vselr", sig::arg<PrimExpr>("source"),
                                  sig::arg<PrimExpr>("indices"));
  DefineSimdArithmetic<4, 2, true>("tl.simd.vmaxs");
  DefineSimdArithmetic<4, 2, true>("tl.simd.vmins");
  DefineSimdArithmetic<4, 2, true>("tl.simd.vmuls");
  DefineSimdArithmetic<4, 2, true>("tl.simd.vadds");
  DefineSimdArithmetic<4, 2, true>("tl.simd.vshls");
  DefineSimdArithmetic<4, 2, true>("tl.simd.vshrs");
  DefineAscendStatement("tl.simd.mem_bar", sig::arg<StringImm>("memory_type"));
  DefineAscendStatement("tl.ascend_set_flag", sig::arg<StringImm>("hard_event"),
                        sig::arg<IntExpr>("event_id"));
  DefineAscendStatement("tl.ascend_wait_flag",
                        sig::arg<StringImm>("hard_event"),
                        sig::arg<IntExpr>("event_id"));
  DefineAscendStatement("tl.ascend_threadfence");
  DefineAscendStatement(
      "tl.ascend_cross_core_set_flag", sig::arg<IntImm>("mode_id"),
      sig::arg<StringImm>("pipe"), sig::arg<IntExpr>("flag_id"));
  DefineAscendStatement(
      "tl.ascend_cross_core_wait_flag", sig::arg<IntImm>("mode_id"),
      sig::arg<StringImm>("pipe"), sig::arg<IntExpr>("flag_id"));
  DefineAscendStatement(
      "tl.ascend_copy_gm_to_ubuf", sig::arg<PointerExpr>("destination"),
      sig::arg<PointerExpr>("source"), sig::arg<IntExpr>("sid"),
      sig::arg<IntExpr>("burst_count"), sig::arg<IntExpr>("burst_bytes"),
      sig::arg<IntExpr>("left_padding"), sig::arg<IntExpr>("right_padding"),
      sig::arg<IntExpr>("pad_mode"), sig::arg<IntExpr>("l2_cache"),
      sig::arg<IntExpr>("source_stride"),
      sig::arg<IntExpr>("destination_stride"));
  DefineAscendStatement("tl.ascend_set_copy_pad_value",
                        sig::arg<PrimExpr>("value"));
  DefineAscendStatement(
      "tl.ascend_copy_ubuf_to_gm", sig::arg<PointerExpr>("destination"),
      sig::arg<PointerExpr>("source"), sig::arg<IntExpr>("sid"),
      sig::arg<IntExpr>("burst_count"), sig::arg<IntExpr>("burst_bytes"),
      sig::arg<IntExpr>("l2_cache"), sig::arg<IntExpr>("destination_stride"),
      sig::arg<IntExpr>("source_stride"));
  DefineAscendStatement(
      "tl.ascend_copy_gm_to_cbuf", sig::arg<PointerExpr>("destination"),
      sig::arg<PointerExpr>("source"), sig::arg<IntExpr>("sid"),
      sig::arg<IntExpr>("source_row_bytes"), sig::arg<IntExpr>("l2_cache"),
      sig::arg<IntExpr>("n_value"), sig::arg<IntExpr>("d_value"),
      sig::arg<IntExpr>("reserved0"), sig::arg<IntExpr>("reserved1"),
      sig::arg<IntImm>("transpose"), sig::arg<IntExpr>("nz_c0_stride"),
      sig::arg<StringImm>("physical_dtype"));
  DefineAscendStatement(
      "tl.ascend_fill_l1", sig::arg<PointerExpr>("destination"),
      sig::arg<IntExpr>("byte_offset"), sig::arg<PrimExpr>("raw_value"),
      sig::arg<IntExpr>("repeat_count"), sig::arg<IntExpr>("block_count"),
      sig::arg<IntExpr>("destination_gap"), sig::arg<IntImm>("fill_word_bits"));
  DefineAscendStatement(
      "tl.ascend_load_cbuf_to_ca", sig::arg<PointerExpr>("destination"),
      sig::arg<PointerExpr>("source"), sig::arg<IntExpr>("m_start"),
      sig::arg<IntExpr>("k_start"), sig::arg<IntExpr>("m_step"),
      sig::arg<IntExpr>("k_step"), sig::arg<IntExpr>("source_stride"),
      sig::arg<IntExpr>("destination_stride"), sig::arg<IntImm>("transpose"));
  DefineAscendStatement(
      "tl.ascend_load_cbuf_to_cb", sig::arg<PointerExpr>("destination"),
      sig::arg<PointerExpr>("source"), sig::arg<IntExpr>("m_start"),
      sig::arg<IntExpr>("k_start"), sig::arg<IntExpr>("m_step"),
      sig::arg<IntExpr>("k_step"), sig::arg<IntExpr>("source_stride"),
      sig::arg<IntExpr>("destination_stride"), sig::arg<IntImm>("transpose"));
  DefineAscendStatement(
      "tl.ascend_load_ca_sf", sig::arg<PointerExpr>("destination"),
      sig::arg<PointerExpr>("scales"), sig::arg<IntExpr>("x_start"),
      sig::arg<IntExpr>("y_start"), sig::arg<IntExpr>("x_step"),
      sig::arg<IntExpr>("y_step"), sig::arg<IntExpr>("source_stride"),
      sig::arg<IntExpr>("destination_stride"));
  DefineAscendStatement(
      "tl.ascend_load_cb_sf", sig::arg<PointerExpr>("destination"),
      sig::arg<PointerExpr>("scales"), sig::arg<IntExpr>("x_start"),
      sig::arg<IntExpr>("y_start"), sig::arg<IntExpr>("x_step"),
      sig::arg<IntExpr>("y_step"), sig::arg<IntExpr>("source_stride"),
      sig::arg<IntExpr>("destination_stride"));
  DefineAscendStatement(
      "tl.ascend_copy_matrix_cc_to_ub", sig::arg<PointerExpr>("destination"),
      sig::arg<PointerExpr>("source"), sig::arg<IntExpr>("sid"),
      sig::arg<IntExpr>("n_size"), sig::arg<IntExpr>("m_size"),
      sig::arg<IntExpr>("destination_stride"),
      sig::arg<IntExpr>("source_stride"), sig::arg<IntExpr>("dual_destination"),
      sig::arg<IntExpr>("sub_block_id"), sig::arg<IntExpr>("clip_relu_pre"),
      sig::arg<IntExpr>("unit_flag"), sig::arg<IntExpr>("quant_pre"),
      sig::arg<IntExpr>("relu_pre"), sig::arg<IntExpr>("split"),
      sig::arg<IntExpr>("nz_to_nd"), sig::arg<IntExpr>("quant_post"),
      sig::arg<IntExpr>("relu_post"), sig::arg<IntExpr>("clip_relu_post"),
      sig::arg<IntExpr>("loop_enhance"), sig::arg<IntExpr>("eltwise"),
      sig::arg<IntExpr>("eltwise_antiquant"),
      sig::arg<IntExpr>("loop_enhance_merge"), sig::arg<IntExpr>("c0_pad"),
      sig::arg<IntExpr>("wino_post"), sig::arg<IntExpr>("broadcast"),
      sig::arg<IntExpr>("nz_to_dn"));
  DefineAscendStatement(
      "tl.ascend_copy_ubuf_to_cbuf", sig::arg<PointerExpr>("destination"),
      sig::arg<PointerExpr>("source"), sig::arg<IntExpr>("sub_block_id"),
      sig::arg<IntExpr>("burst_count"), sig::arg<IntExpr>("burst_bytes"),
      sig::arg<IntExpr>("source_gap"), sig::arg<IntExpr>("destination_gap"));
  DefineAscendStatement(
      "tl.ascend_nd2nz_scatter", sig::arg<PointerExpr>("source"),
      sig::arg<PointerExpr>("temporary"), sig::arg<IntExpr>("rows"),
      sig::arg<IntExpr>("columns"), sig::arg<IntExpr>("destination_pitch"),
      sig::arg<StringImm>("source_dtype"),
      sig::arg<StringImm>("destination_dtype"));
  DefineAscendStatement(
      "tl.ascend_nd2nz_post_copy", sig::arg<PointerExpr>("destination"),
      sig::arg<PointerExpr>("source"), sig::arg<IntExpr>("rows"),
      sig::arg<IntExpr>("columns"), sig::arg<IntExpr>("source_pitch"),
      sig::arg<IntExpr>("destination_pitch"), sig::arg<StringImm>("dtype"));
  DefineAscendStatement(
      "tl.ascend_copy_matrix_cc_to_gm", sig::arg<PointerExpr>("destination"),
      sig::arg<PointerExpr>("source"), sig::arg<IntExpr>("sid"),
      sig::arg<IntExpr>("n_size"), sig::arg<IntExpr>("m_size"),
      sig::arg<IntExpr>("destination_stride"),
      sig::arg<IntExpr>("source_stride"), sig::arg<IntExpr>("l2_cache"),
      sig::arg<IntExpr>("clip_relu_pre"), sig::arg<IntExpr>("unit_flag"),
      sig::arg<IntExpr>("quant_pre"), sig::arg<IntExpr>("relu_pre"),
      sig::arg<IntExpr>("split"), sig::arg<IntExpr>("nz_to_nd"),
      sig::arg<IntExpr>("quant_post"), sig::arg<IntExpr>("relu_post"),
      sig::arg<IntExpr>("clip_relu_post"), sig::arg<IntExpr>("loop_enhance"),
      sig::arg<IntExpr>("eltwise"), sig::arg<IntExpr>("eltwise_antiquant"),
      sig::arg<IntExpr>("loop_enhance_merge"), sig::arg<IntExpr>("c0_pad"),
      sig::arg<IntExpr>("wino_post"), sig::arg<IntExpr>("broadcast"),
      sig::arg<IntExpr>("nz_to_dn"));
  DefineAscendStatement(
      "tl.ascend_mad", sig::arg<PointerExpr>("destination"),
      sig::arg<PointerExpr>("a"), sig::arg<PointerExpr>("b"),
      sig::arg<IntExpr>("m"), sig::arg<IntExpr>("k"), sig::arg<IntExpr>("n"),
      sig::arg<IntExpr>("unit_flag"), sig::arg<IntExpr>("gemv"),
      sig::arg<IntExpr>("bt_buffer"), sig::arg<PrimExpr>("clear_accumulator"));
  DefineAscendStatement(
      "tl.ascend_mad_mx", sig::arg<PointerExpr>("destination"),
      sig::arg<PointerExpr>("a"), sig::arg<PointerExpr>("b"),
      sig::arg<IntExpr>("m"), sig::arg<IntExpr>("k"), sig::arg<IntExpr>("n"),
      sig::arg<IntExpr>("unit_flag"), sig::arg<IntExpr>("gemv"),
      sig::arg<IntExpr>("bt_buffer"), sig::arg<PrimExpr>("clear_accumulator"));
  DefineAscendStatement(
      "tl.ascend_gemm_l1", sig::arg<PointerExpr>("destination"),
      sig::arg<PointerExpr>("a"), sig::arg<PointerExpr>("b"),
      sig::arg<IntImm>("m"), sig::arg<IntImm>("k"), sig::arg<IntImm>("n"),
      sig::arg<IntImm>("tile_k_sub"), sig::arg<IntImm>("transpose_b"),
      sig::arg<PrimExpr>("clear_accumulator"), sig::arg<StringImm>("dtype"),
      sig::arg<IntExpr>("reserved"), sig::arg<IntExpr>("unit_flag"));
  DefineAscendStatement(
      "tl.ascend_blockscaled_gemm_l1", sig::arg<PointerExpr>("destination"),
      sig::arg<PointerExpr>("a"), sig::arg<PointerExpr>("b"),
      sig::arg<PointerExpr>("scale_a"), sig::arg<PointerExpr>("scale_b"),
      sig::arg<IntImm>("m"), sig::arg<IntImm>("k"), sig::arg<IntImm>("n"),
      sig::arg<IntImm>("tile_k_sub"), sig::arg<IntImm>("transpose_b"),
      sig::arg<PrimExpr>("clear_accumulator"),
      sig::arg<StringImm>("input_dtype"), sig::arg<StringImm>("scale_dtype"),
      sig::arg<StringImm>("accumulator_dtype"), sig::arg<IntExpr>("reserved"),
      sig::arg<IntExpr>("scale_k_offset"), sig::arg<IntImm>("scale_nz_stride"),
      sig::arg<IntExpr>("unit_flag"));
  OpDef("tl.ascend_read_gm_bypass_dcache")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateBypassRead>())
      .signature(sig::arg<PointerExpr>("pointer"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_native_fn_attr<&InferBypassRead>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kReadState)
      .set_attr(tirx::op_attr::kVectorizable, false);
  DefineAscendStatement("tl.ascend_write_gm_bypass_dcache",
                        sig::arg<PointerExpr>("pointer"),
                        sig::arg<PrimExpr>("value"));
  DefineAscendStatement("tl.ascend_get_buf", sig::arg<StringImm>("pipe"),
                        sig::arg<IntExpr>("buffer_id"),
                        sig::arg<IntImm>("mode"));
  DefineAscendStatement("tl.ascend_rls_buf", sig::arg<StringImm>("pipe"),
                        sig::arg<IntExpr>("buffer_id"),
                        sig::arg<IntImm>("mode"));
  DefineAscendStatement("tl.ascend_set_hf32_mode", sig::arg<IntImm>("mode"));
  DefineAscendStatement("tl.ascend_set_mmad_direction",
                        sig::arg<StringImm>("direction"));
  DefineAscendStatement("tl.ascend_set_atomic",
                        sig::arg<StringImm>("operation"),
                        sig::arg<PrimExpr>("typed_zero"));
  DefineAscendStatement("tl.ascend_set_atomic_none");
}
} // namespace tvm::tl
