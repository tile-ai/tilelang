/*!
 * \file tl/cuda/op/builtin.cc
 * \brief Registration of CUDA-specific TileLang intrinsic Ops.
 */

#include "builtin.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/op/attr.h>

#include "ir/annotations.h"

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

namespace {
using PointerExpr = TypedExpr<PtrType>;

void CheckCudaResult(const CallObj *call, const Type &result) {
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  if (!call->ty_args.empty())
    TVM_FFI_CHECK(ffi::StructuralEqual()(call->ty_args[0], result), TypeError)
        << "Intrinsic result type disagrees with its type argument";
  TVM_FFI_CHECK(ffi::StructuralEqual()(call->ty, result), TypeError)
      << "Intrinsic result type disagrees with its operands";
}

template <int N, bool Rounding> Type InferCudaArithmetic(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), N + (Rounding ? 1U : 0U), ValueError)
      << "CUDA arithmetic has the wrong number of runtime operands";
  const auto dtype = args[0].as_or_throw<PrimExpr>().ty();
  TVM_FFI_CHECK(!dtype.IsVoid(), TypeError);
  for (int i = 1; i < N; ++i)
    TVM_FFI_CHECK_EQ(args[i].as_or_throw<PrimExpr>().ty(), dtype, TypeError)
        << "CUDA arithmetic operands must have matching primitive types";
  return dtype;
}

template <int N, bool Rounding>
void ValidateCudaArithmetic(const CallObj *call) {
  CheckCudaResult(call, InferCudaArithmetic<N, Rounding>(call));
  if constexpr (Rounding) {
    const auto args = DecodeCallOperands(call);
    const auto mode = args[N].as_or_throw<StringImm>()->value;
    TVM_FFI_CHECK(mode == "rn" || mode == "rz" || mode == "rm" || mode == "rp",
                  ValueError)
        << "CUDA arithmetic has an invalid rounding mode";
  }
}

template <int N, bool Rounding = false>
void DefineCudaArithmetic(const char *name, CallEffectKind effect,
                          bool vectorizable = false) {
  OpDef def(name);
  def.set_validator(
         ffi::reflection::NativeFunctionView<void(const CallObj *)>::FromNative<
             &ValidateCudaArithmetic<N, Rounding>>())
      .template set_native_fn_attr<&InferCudaArithmetic<N, Rounding>>(
          tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, effect)
      .set_attr(tirx::op_attr::kVectorizable, vectorizable);
  if constexpr (N == 1 && !Rounding)
    def.signature(sig::arg<PrimExpr>("x"),
                  sig::var_args<tvm::Tuple>("annotation_payload"),
                  sig::var_ty_args<PrimType>("result_type"));
  else if constexpr (N == 1)
    def.signature(sig::arg<PrimExpr>("x"), sig::arg<StringImm>("rounding_mode"),
                  sig::var_args<tvm::Tuple>("annotation_payload"),
                  sig::var_ty_args<PrimType>("result_type"));
  else if constexpr (N == 2 && !Rounding)
    def.signature(sig::arg<PrimExpr>("x"), sig::arg<PrimExpr>("y"),
                  sig::var_args<tvm::Tuple>("annotation_payload"),
                  sig::var_ty_args<PrimType>("result_type"));
  else if constexpr (N == 2)
    def.signature(sig::arg<PrimExpr>("x"), sig::arg<PrimExpr>("y"),
                  sig::arg<StringImm>("rounding_mode"),
                  sig::var_args<tvm::Tuple>("annotation_payload"),
                  sig::var_ty_args<PrimType>("result_type"));
  else if constexpr (!Rounding)
    def.signature(sig::arg<PrimExpr>("x"), sig::arg<PrimExpr>("y"),
                  sig::arg<PrimExpr>("z"),
                  sig::var_args<tvm::Tuple>("annotation_payload"),
                  sig::var_ty_args<PrimType>("result_type"));
  else
    def.signature(sig::arg<PrimExpr>("x"), sig::arg<PrimExpr>("y"),
                  sig::arg<PrimExpr>("z"), sig::arg<StringImm>("rounding_mode"),
                  sig::var_args<tvm::Tuple>("annotation_payload"),
                  sig::var_ty_args<PrimType>("result_type"));
}

// PTX instructions used as statements mutate storage or device state. Their
// source wrappers historically retain handle or accumulator-dtype results;
// typed C++ lowerings use void. This compatibility is confined to that ABI.
Type InferPtxStatement(const CallObj *call) {
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  const Type result = !call->ty_args.empty()           ? call->ty_args[0]
                      : !call->ty.as<MissingTypeObj>() ? call->ty
                                                       : Type(PrimType::Void());
  TVM_FFI_CHECK(result.as<PrimType>() || result.as<PtrType>(), TypeError)
      << "PTX statement result must be primitive or pointer typed";
  return result;
}

// This checks the typed operand descriptors used by this backend's instruction
// and integer-query families. Runtime arity is checked after metadata decoding.
// PointerExpr is the native TypedExpr<PtrType> view, not an untyped handle.
template <typename T> void CheckPtxOperand(const Expr &expr) {
  expr.as_or_throw<T>();
  if constexpr (std::is_same_v<T, IntImm>) {
    const auto dtype = expr.as_or_throw<PrimExpr>().ty();
    TVM_FFI_CHECK(dtype.IsScalar() &&
                      dtype.MatchesCode(kDLInt, kDLUInt, kDLBool),
                  TypeError);
  } else if constexpr (std::is_same_v<T, IntExpr>)
    TVM_FFI_CHECK(expr.as_or_throw<IntExpr>().ty().IsScalar(), TypeError)
        << "PTX index and control operands must be scalar integers";
  else if constexpr (std::is_same_v<T, PrimExpr>)
    TVM_FFI_CHECK(!expr.as_or_throw<PrimExpr>().ty().IsVoid(), TypeError);
  else if constexpr (std::is_same_v<T, Expr>) {
    // Descriptors and barrier references can be typed pointers or integer
    // register values, depending on the lowering stage.
    const auto primitive = expr->ty.as<PrimType>();
    TVM_FFI_CHECK(expr->ty.as<PtrType>() ||
                      (primitive && primitive->IsScalar() &&
                       primitive->MatchesCode(kDLInt, kDLUInt, kDLBool)),
                  TypeError)
        << "PTX descriptor or barrier must be a pointer or scalar integer";
  }
}

template <typename... Operands> void ValidatePtxStatement(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), sizeof...(Operands), ValueError)
      << "PTX instruction has the wrong number of runtime operands";
  size_t index = 0;
  (CheckPtxOperand<Operands>(args[index++]), ...);
  CheckCudaResult(call, InferPtxStatement(call));
}

template <typename... Operands>
void DefinePtxStatement(const char *name, sig::arg<Operands>... operands) {
  OpDef(name)
      .set_validator(
          ffi::reflection::NativeFunctionView<void(const CallObj *)>::
              FromNative<&ValidatePtxStatement<Operands...>>())
      .signature(operands..., sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<Type>("statement_type"))
      .template set_native_fn_attr<&InferPtxStatement>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

template <typename... Operands>
void ValidateCudaIntegerQuery(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), sizeof...(Operands), ValueError)
      << "CUDA integer query has the wrong number of runtime operands";
  size_t index = 0;
  (CheckPtxOperand<Operands>(args[index++]), ...);
  CheckCudaResult(
      call, Op::GetAttrMap(
                tvm::op_attr::kFixedReturnType)[call->op.as_or_throw<Op>()]);
  if (call->op.same_as(pack_b8x4()))
    for (const Expr &arg : args) {
      const auto dtype = arg.as_or_throw<PrimExpr>().ty();
      TVM_FFI_CHECK(dtype.IsScalar() && dtype.bits() == 8, TypeError)
          << "pack_b8x4 requires four scalar 8-bit values";
    }
}

template <typename... Operands>
void DefineCudaIntegerQuery(const char *name, PrimType result,
                            sig::arg<Operands>... operands) {
  OpDef(name)
      .set_validator(
          ffi::reflection::NativeFunctionView<void(const CallObj *)>::
              FromNative<&ValidateCudaIntegerQuery<Operands...>>())
      .signature(operands..., sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_attr(tvm::op_attr::kFixedReturnType, result)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

Type InferThreadCoordinate(const CallObj *call) {
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  const Type result = call->ty_args.empty() ? call->ty : call->ty_args[0];
  TVM_FFI_CHECK(!result.as<MissingTypeObj>(), TypeError)
      << "Thread coordinate requires an explicit result type";
  const auto dtype = result.as_or_throw<PrimType>();
  TVM_FFI_CHECK(dtype.IsScalar() && dtype.MatchesCode(kDLInt, kDLUInt),
                TypeError)
      << "Thread coordinate requires a scalar integer result";
  return dtype;
}

template <int MaxParameters>
void ValidateThreadCoordinate(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_LE(args.size(), MaxParameters, ValueError)
      << "Thread coordinate has too many runtime operands";
  for (const Expr &arg : args)
    CheckPtxOperand<IntExpr>(arg);
  CheckCudaResult(call, InferThreadCoordinate(call));
}

template <int MaxParameters> void DefineThreadCoordinate(const char *name) {
  OpDef(name)
      .set_validator(
          ffi::reflection::NativeFunctionView<void(const CallObj *)>::
              FromNative<&ValidateThreadCoordinate<MaxParameters>>())
      .signature(
          sig::var_args<Expr>("optional_warp_dimensions_and_annotations"),
          sig::var_ty_args<PrimType>("index_type"))
      .template set_native_fn_attr<&InferThreadCoordinate>(
          tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

Type InferTensorMap(const CallObj *call) {
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  const Type result = !call->ty_args.empty() ? call->ty_args[0]
                      : !call->ty.as<MissingTypeObj>()
                          ? call->ty
                          : Type(CuTensorMapType());
  return result.as_or_throw<PtrType>();
}

template <bool Im2Col> void ValidateTensorMap(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_GE(args.size(), 3U, ValueError)
      << "TMA descriptor requires dtype, rank, and global pointer runtime "
         "operands";
  args[0].as_or_throw<IntImm>();
  const auto rank = args[1].as_or_throw<IntImm>()->value.as<int64_t>();
  TVM_FFI_CHECK(rank && *rank >= (Im2Col ? 3 : 1) && *rank <= 5, ValueError)
      << "TMA descriptor rank must be in the supported hardware range";
  // TMADesc carries four rank-sized vectors. Im2Col carries three rank-sized
  // vectors, two rank-2 corner vectors, and two box dimensions.
  TVM_FFI_CHECK_EQ(args.size(), Im2Col ? 5 * *rank + 5 : 4 * *rank + 7,
                   ValueError)
      << "TMA descriptor runtime operands disagree with its rank";
  args[2].as_or_throw<PointerExpr>();
  for (size_t i = 3; i < args.size(); ++i)
    CheckPtxOperand<IntExpr>(args[i]);
  CheckCudaResult(call, InferTensorMap(call));
}

template <bool Im2Col> void DefineTensorMap(const char *name) {
  OpDef(name)
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateTensorMap<Im2Col>>())
      .signature(
          sig::arg<IntImm>("data_type"), sig::arg<IntImm>("rank"),
          sig::arg<PointerExpr>("global_pointer"),
          sig::var_args<Expr>("ranked_shape_stride_options_and_annotations"),
          sig::var_ty_args<PtrType>("descriptor_type"))
      .template set_native_fn_attr<&InferTensorMap>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

enum class TmaTransfer { Load, Multicast, Im2Col, Store, Gather4, Scatter4 };

template <TmaTransfer Kind> void ValidateTmaTransfer(const CallObj *call) {
  constexpr bool load =
      Kind != TmaTransfer::Store && Kind != TmaTransfer::Scatter4;
  constexpr bool multicast = Kind == TmaTransfer::Multicast;
  const auto args = DecodeCallOperands(call);
  if constexpr (Kind == TmaTransfer::Gather4 || Kind == TmaTransfer::Scatter4)
    TVM_FFI_CHECK_EQ(args.size(), load ? 9U : 8U, ValueError)
        << "TMA gather/scatter has the wrong number of runtime operands";
  else if constexpr (Kind == TmaTransfer::Im2Col)
    TVM_FFI_CHECK(args.size() == 8 || args.size() == 10 || args.size() == 12,
                  ValueError)
        << "TMA im2col runtime operands require rank 3, 4, or 5";
  else
    TVM_FFI_CHECK(args.size() >= (multicast ? 6U : 5U) &&
                      args.size() <= (multicast ? 10U : 9U),
                  ValueError)
        << "TMA transfer runtime operands require one to five coordinates";
  args[0].as_or_throw<PointerExpr>();
  if constexpr (load)
    CheckPtxOperand<Expr>(args[1]);
  args[load ? 2 : 1].as_or_throw<PointerExpr>();
  for (size_t i = load ? 3 : 2; i < args.size(); ++i)
    CheckPtxOperand<IntExpr>(args[i]);
  args.back().as_or_throw<IntImm>();
  if constexpr (Kind == TmaTransfer::Store)
    args[args.size() - 2].as_or_throw<IntImm>();
  CheckCudaResult(call, InferPtxStatement(call));
}

template <TmaTransfer Kind> void DefineTmaTransfer(const char *name) {
  OpDef def(name);
  def.set_validator(
         ffi::reflection::NativeFunctionView<void(
             const CallObj *)>::FromNative<&ValidateTmaTransfer<Kind>>())
      .template set_native_fn_attr<&InferPtxStatement>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  if constexpr (Kind == TmaTransfer::Store || Kind == TmaTransfer::Scatter4)
    def.signature(sig::arg<PointerExpr>("descriptor"),
                  sig::arg<PointerExpr>("shared_pointer"),
                  sig::var_args<Expr>("coordinates_options_and_annotations"),
                  sig::var_ty_args<Type>("statement_type"));
  else if constexpr (Kind == TmaTransfer::Multicast)
    def.signature(sig::arg<PointerExpr>("descriptor"),
                  sig::arg<Expr>("barrier"),
                  sig::arg<PointerExpr>("shared_pointer"),
                  sig::arg<IntExpr>("multicast_mask"),
                  sig::var_args<Expr>("coordinates_options_and_annotations"),
                  sig::var_ty_args<Type>("statement_type"));
  else
    def.signature(sig::arg<PointerExpr>("descriptor"),
                  sig::arg<Expr>("barrier"),
                  sig::arg<PointerExpr>("shared_pointer"),
                  sig::var_args<Expr>("coordinates_options_and_annotations"),
                  sig::var_ty_args<Type>("statement_type"));
}

template <int Bits, bool Global, bool Store>
void ValidatePtxMemory(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  constexpr size_t required = Store ? 2 : 1;
  TVM_FFI_CHECK(args.size() == required ||
                    (Global && args.size() == required + 1),
                ValueError)
      << "PTX memory instruction has the wrong number of runtime operands";
  args[0].as_or_throw<PointerExpr>();
  if (args.size() > required) {
    const auto pred = args.back().as_or_throw<PrimExpr>().ty();
    TVM_FFI_CHECK(pred.IsScalar() && pred.MatchesCode(kDLBool, kDLInt, kDLUInt),
                  TypeError)
        << "PTX memory predicate must be a scalar boolean or integer";
  }
  if constexpr (Store) {
    const auto dtype = args[1].as_or_throw<PrimExpr>().ty();
    TVM_FFI_CHECK(!dtype.IsVoid() && dtype.bits() * dtype.lanes() == Bits,
                  TypeError)
        << "PTX store value width disagrees with the instruction";
    CheckCudaResult(call, InferPtxStatement(call));
  } else {
    CheckCudaResult(call, PrimType::UInt(32, Bits / 32));
  }
}

template <int Bits, bool Global, bool Store>
void DefinePtxMemory(const char *name) {
  OpDef def(name);
  def.set_validator(
         ffi::reflection::NativeFunctionView<void(const CallObj *)>::FromNative<
             &ValidatePtxMemory<Bits, Global, Store>>())
      .set_attr(tvm::op_attr::kCallEffectKind,
                Store ? CallEffectKind::kOpaque : CallEffectKind::kReadState)
      .set_attr(tirx::op_attr::kVectorizable, false);
  if constexpr (Store) {
    def.template set_native_fn_attr<&InferPtxStatement>(
        tvm::op_attr::kInferType);
    if constexpr (Global)
      def.signature(sig::arg<PointerExpr>("pointer"),
                    sig::arg<PrimExpr>("value"),
                    sig::var_args<Expr>("predicate_and_annotations"),
                    sig::var_ty_args<Type>("statement_type"));
    else
      def.signature(sig::arg<PointerExpr>("pointer"),
                    sig::arg<PrimExpr>("value"),
                    sig::var_args<tvm::Tuple>("annotation_payload"),
                    sig::var_ty_args<Type>("statement_type"));
  } else {
    def.set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(32, Bits / 32));
    if constexpr (Global)
      def.signature(sig::arg<PointerExpr>("pointer"),
                    sig::var_args<Expr>("predicate_and_annotations"),
                    sig::var_ty_args<PrimType>("result_type"));
    else
      def.signature(sig::arg<PointerExpr>("pointer"),
                    sig::var_args<tvm::Tuple>("annotation_payload"),
                    sig::var_ty_args<PrimType>("result_type"));
  }
}

void ValidateTensorMemoryTransfer(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK(args.size() == 6 || args.size() == 7, ValueError)
      << "tcgen05 transfer requires six or seven runtime operands";
  for (int i : {0, 1, 2})
    args[i].as_or_throw<IntImm>();
  CheckPtxOperand<Expr>(args[3]);
  CheckPtxOperand<IntExpr>(args[4]);
  if (args.size() == 7)
    args[5].as_or_throw<IntImm>();
  args.back().as_or_throw<PointerExpr>();
  CheckCudaResult(call, InferPtxStatement(call));
}

void DefineTensorMemoryTransfer(const char *name) {
  OpDef(name)
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateTensorMemoryTransfer>())
      .signature(
          sig::arg<IntImm>("instruction_bits"), sig::arg<IntImm>("chunks"),
          sig::arg<IntImm>("pack_or_unpack_16"), sig::arg<Expr>("tmem_column"),
          sig::arg<IntExpr>("column_offset"),
          sig::arg<Expr>("pointer_or_datapaths"),
          sig::var_args<Expr>("pointer_and_annotations"),
          sig::var_ty_args<Type>("statement_type"))
      .template set_native_fn_attr<&InferPtxStatement>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
}
} // namespace

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
  DefineCudaArithmetic<1>("tl.__exp", CallEffectKind::kOpaque);
  DefineCudaArithmetic<1>("tl.__exp10", CallEffectKind::kOpaque);
  DefineCudaArithmetic<1>("tl.__log", CallEffectKind::kOpaque);
  DefineCudaArithmetic<1>("tl.__log2", CallEffectKind::kOpaque);
  DefineCudaArithmetic<1>("tl.__log10", CallEffectKind::kOpaque);
  DefineCudaArithmetic<1>("tl.__tan", CallEffectKind::kOpaque);
  DefineCudaArithmetic<1>("tl.__cos", CallEffectKind::kOpaque);
  DefineCudaArithmetic<1>("tl.__sin", CallEffectKind::kOpaque);
  DefineCudaArithmetic<1>("tl.fast_rcp", CallEffectKind::kOpaque);
  DefineCudaArithmetic<2, false>("tl.max_nan", CallEffectKind::kPure);
  DefineCudaArithmetic<2, false>("tl.min_nan", CallEffectKind::kPure);
  DefineCudaArithmetic<2, true>("tl.ieee_add", CallEffectKind::kPure);
  DefineCudaArithmetic<2, true>("tl.ieee_sub", CallEffectKind::kPure);
  DefineCudaArithmetic<2, true>("tl.ieee_mul", CallEffectKind::kPure);
  DefineCudaArithmetic<3, true>("tl.ieee_fmaf", CallEffectKind::kPure);
  DefineCudaArithmetic<1, true>("tl.ieee_frcp", CallEffectKind::kPure);
  DefineCudaArithmetic<1, true>("tl.ieee_fsqrt", CallEffectKind::kPure);
  DefineCudaArithmetic<1, false>("tl.ieee_frsqrt", CallEffectKind::kPure);
  DefineCudaArithmetic<2, true>("tl.ieee_fdiv", CallEffectKind::kPure);
  DefineCudaArithmetic<3, false>("tl.fma", CallEffectKind::kPure, true);
  DefineCudaArithmetic<2, false>("tl.fmul", CallEffectKind::kPure, true);
  DefineCudaArithmetic<2, false>("tl.max2_nan", CallEffectKind::kPure);
  DefineCudaArithmetic<2, false>("tl.min2_nan", CallEffectKind::kPure);
  DefineTensorMap<false>("tl.create_tma_descriptor");
  DefineTensorMap<true>("tl.create_tma_im2col_descriptor");
  DefinePtxStatement("tl.prefetch_tma_descriptor",
                     sig::arg<PointerExpr>("descriptor"));
  DefineTmaTransfer<TmaTransfer::Load>("tl.tma_load");
  DefineTmaTransfer<TmaTransfer::Im2Col>("tl.tma_load_im2col");
  DefineTmaTransfer<TmaTransfer::Multicast>("tl.tma_load_multicast");
  DefineTmaTransfer<TmaTransfer::Store>("tl.tma_store");
  DefineTmaTransfer<TmaTransfer::Gather4>("tl.tma_load_gather4");
  DefineTmaTransfer<TmaTransfer::Scatter4>("tl.tma_store_scatter4");
  DefinePtxStatement("tl.ptx_fence_barrier_init");
  DefinePtxStatement("tl.ptx_arrive_cluster_barrier", sig::arg<Expr>("barrier"),
                     sig::arg<IntExpr>("cta"));
  DefinePtxStatement(
      "tl.ptx_wgmma_ss", sig::arg<StringImm>("shape"),
      sig::arg<PrimExpr>("a_is_k_major"), sig::arg<PrimExpr>("b_is_k_major"),
      sig::arg<StringImm>("a_dtype"), sig::arg<StringImm>("b_dtype"),
      sig::arg<StringImm>("c_dtype"), sig::arg<Expr>("a_descriptor"),
      sig::arg<IntExpr>("a_offset"), sig::arg<Expr>("b_descriptor"),
      sig::arg<IntExpr>("b_offset"), sig::arg<PointerExpr>("accumulator"),
      sig::arg<IntExpr>("c_offset"), sig::arg<PrimExpr>("scale_out"),
      sig::arg<IntExpr>("scale_in_a"), sig::arg<IntExpr>("scale_in_b"));
  DefinePtxStatement(
      "tl.ptx_wgmma_rs", sig::arg<StringImm>("shape"),
      sig::arg<PrimExpr>("b_is_k_major"), sig::arg<StringImm>("a_dtype"),
      sig::arg<StringImm>("b_dtype"), sig::arg<StringImm>("c_dtype"),
      sig::arg<PointerExpr>("a"), sig::arg<IntExpr>("a_offset"),
      sig::arg<Expr>("b_descriptor"), sig::arg<IntExpr>("b_offset"),
      sig::arg<PointerExpr>("accumulator"), sig::arg<IntExpr>("c_offset"),
      sig::arg<PrimExpr>("scale_out"), sig::arg<IntExpr>("scale_in_a"),
      sig::arg<IntExpr>("scale_in_b"));
  DefinePtxStatement(
      "tl.ptx_wgmma_sp_ss", sig::arg<StringImm>("shape"),
      sig::arg<PrimExpr>("a_is_k_major"), sig::arg<PrimExpr>("b_is_k_major"),
      sig::arg<StringImm>("a_dtype"), sig::arg<StringImm>("b_dtype"),
      sig::arg<StringImm>("c_dtype"), sig::arg<Expr>("a_descriptor"),
      sig::arg<IntExpr>("a_offset"), sig::arg<PointerExpr>("metadata"),
      sig::arg<IntExpr>("metadata_offset"),
      sig::arg<IntExpr>("sparse_selector"), sig::arg<Expr>("b_descriptor"),
      sig::arg<IntExpr>("b_offset"), sig::arg<PointerExpr>("accumulator"),
      sig::arg<IntExpr>("c_offset"), sig::arg<PrimExpr>("scale_out"),
      sig::arg<IntExpr>("scale_in_a"), sig::arg<IntExpr>("scale_in_b"));
  DefinePtxStatement(
      "tl.ptx_wgmma_sp_rs", sig::arg<StringImm>("shape"),
      sig::arg<PrimExpr>("b_is_k_major"), sig::arg<StringImm>("a_dtype"),
      sig::arg<StringImm>("b_dtype"), sig::arg<StringImm>("c_dtype"),
      sig::arg<PointerExpr>("a"), sig::arg<IntExpr>("a_offset"),
      sig::arg<PointerExpr>("metadata"), sig::arg<IntExpr>("metadata_offset"),
      sig::arg<IntExpr>("sparse_selector"), sig::arg<Expr>("b_descriptor"),
      sig::arg<IntExpr>("b_offset"), sig::arg<PointerExpr>("accumulator"),
      sig::arg<IntExpr>("c_offset"), sig::arg<PrimExpr>("scale_out"),
      sig::arg<IntExpr>("scale_in_a"), sig::arg<IntExpr>("scale_in_b"));
  DefinePtxStatement(
      "tl.ptx_mma_block_scale", sig::arg<StringImm>("accum_dtype"),
      sig::arg<StringImm>("shape"), sig::arg<StringImm>("a_layout"),
      sig::arg<StringImm>("b_layout"), sig::arg<StringImm>("kind"),
      sig::arg<IntImm>("scale_vector_size"), sig::arg<StringImm>("a_dtype"),
      sig::arg<StringImm>("b_dtype"), sig::arg<StringImm>("scale_type"),
      sig::arg<PointerExpr>("a"), sig::arg<IntExpr>("a_index"),
      sig::arg<PointerExpr>("b"), sig::arg<IntExpr>("b_index"),
      sig::arg<PointerExpr>("accumulator"), sig::arg<IntExpr>("c_index"),
      sig::arg<Expr>("scale_a"), sig::arg<Expr>("scale_b"),
      sig::arg<IntExpr>("scale_a_byte"), sig::arg<IntExpr>("scale_a_thread"),
      sig::arg<IntExpr>("scale_b_byte"), sig::arg<IntExpr>("scale_b_thread"));
  DefinePtxStatement(
      "tl.ptx_tcgen05_mma_ss", sig::arg<StringImm>("kind_dtype"),
      sig::arg<Expr>("a_descriptor"), sig::arg<IntExpr>("a_offset"),
      sig::arg<Expr>("b_descriptor"), sig::arg<IntExpr>("b_offset"),
      sig::arg<Expr>("c_pointer"), sig::arg<IntExpr>("c_offset"),
      sig::arg<Expr>("instruction_descriptor"), sig::arg<PrimExpr>("scale_out"),
      sig::arg<IntExpr>("mask0"), sig::arg<IntExpr>("mask1"),
      sig::arg<IntExpr>("mask2"), sig::arg<IntExpr>("mask3"),
      sig::arg<PrimExpr>("enable_ws"), sig::arg<PrimExpr>("enable_2cta"));
  DefinePtxStatement(
      "tl.ptx_tcgen05_mma_ts", sig::arg<StringImm>("kind_dtype"),
      sig::arg<Expr>("a_pointer"), sig::arg<IntExpr>("a_offset"),
      sig::arg<Expr>("b_descriptor"), sig::arg<IntExpr>("b_offset"),
      sig::arg<Expr>("c_pointer"), sig::arg<IntExpr>("c_offset"),
      sig::arg<Expr>("instruction_descriptor"), sig::arg<PrimExpr>("scale_out"),
      sig::arg<IntExpr>("mask0"), sig::arg<IntExpr>("mask1"),
      sig::arg<IntExpr>("mask2"), sig::arg<IntExpr>("mask3"),
      sig::arg<PrimExpr>("enable_2cta"));
  DefinePtxStatement(
      "tl.ptx_tcgen05_mma_blockscaled_ss", sig::arg<StringImm>("kind_dtype"),
      sig::arg<Expr>("a_descriptor"), sig::arg<IntExpr>("a_offset"),
      sig::arg<Expr>("b_descriptor"), sig::arg<IntExpr>("b_offset"),
      sig::arg<Expr>("c_pointer"), sig::arg<IntExpr>("c_offset"),
      sig::arg<Expr>("instruction_descriptor"), sig::arg<PrimExpr>("scale_out"),
      sig::arg<Expr>("sfa_pointer"), sig::arg<IntExpr>("sfa_offset"),
      sig::arg<Expr>("sfb_pointer"), sig::arg<IntExpr>("sfb_offset"),
      sig::arg<IntExpr>("reserved0"), sig::arg<IntExpr>("reserved1"),
      sig::arg<PrimExpr>("enable_2cta"));
  DefinePtxStatement(
      "tl.ptx_tcgen05_cp_warpx4", sig::arg<PointerExpr>("shared_pointer"),
      sig::arg<Expr>("tmem_pointer"), sig::arg<IntExpr>("column_offset"));
  DefinePtxStatement("tl.ptx_tcgen05_sf_warp_transpose",
                     sig::arg<PointerExpr>("shared_pointer"));
  DefinePtxStatement("tl.deallocate_tmem",
                     sig::arg<PointerExpr>("tmem_pointer"));
  DefinePtxStatement("tl.ptx_init_tensor_memory",
                     sig::arg<PointerExpr>("tmem_pointer"),
                     sig::arg<IntExpr>("columns"));
  DefinePtxStatement("tl.ptx_deallocate_tensor_memory",
                     sig::arg<Expr>("tmem_address"),
                     sig::arg<IntExpr>("columns"));
  DefinePtxStatement(
      "tl.ptx_mma_sm70", sig::arg<StringImm>("shape"),
      sig::arg<StringImm>("a_layout"), sig::arg<StringImm>("b_layout"),
      sig::arg<StringImm>("a_dtype"), sig::arg<StringImm>("b_dtype"),
      sig::arg<StringImm>("c_dtype"), sig::arg<PointerExpr>("a"),
      sig::arg<IntExpr>("a_index"), sig::arg<PointerExpr>("b"),
      sig::arg<IntExpr>("b_index"), sig::arg<PointerExpr>("accumulator"),
      sig::arg<IntExpr>("c_index"));
  DefinePtxStatement("tl.ptx_ldmatrix", sig::arg<PrimExpr>("transpose"),
                     sig::arg<IntExpr>("registers"),
                     sig::arg<PointerExpr>("shared_pointer"),
                     sig::arg<PointerExpr>("destination"));
  DefinePtxStatement("tl.ptx_cp_async_barrier_noinc",
                     sig::arg<Expr>("barrier"));
  DefinePtxStatement("tl.ptx_st_bulk_shared",
                     sig::arg<PointerExpr>("shared_pointer"),
                     sig::arg<IntImm>("bytes"), sig::arg<IntImm>("value"));
  DefinePtxStatement("tl.fence_proxy_async");
  DefinePtxStatement("tl.tcgen05_before_thread_sync");
  DefinePtxStatement("tl.tcgen05_after_thread_sync");
  DefinePtxStatement("tl.tma_store_arrive");
  DefinePtxStatement("tl.tma_store_wait", sig::arg<IntExpr>("count"),
                     sig::arg<PrimExpr>("read"));
  DefinePtxStatement("tl.set_max_nreg", sig::arg<IntExpr>("count"),
                     sig::arg<PrimExpr>("increase"));
  DefinePtxStatement("tl.warpgroup_arrive");
  DefinePtxStatement("tl.warpgroup_commit_batch");
  DefinePtxStatement("tl.warpgroup_wait", sig::arg<IntExpr>("pending_groups"));
  DefinePtxStatement("tl.warpgroup_fence_operand", sig::arg<StringImm>("dtype"),
                     sig::arg<PointerExpr>("data"), sig::arg<IntExpr>("offset"),
                     sig::arg<IntExpr>("register_count"));
  DefineThreadCoordinate<1>("tl.get_lane_idx");
  DefineThreadCoordinate<1>("tl.get_warp_idx_sync");
  DefineThreadCoordinate<1>("tl.get_warp_idx");
  DefineThreadCoordinate<2>("tl.get_warp_group_idx");
  DefineCudaIntegerQuery("tl.pack_b8x4", PrimType::Int(32),
                         sig::arg<PrimExpr>("x0"), sig::arg<PrimExpr>("x1"),
                         sig::arg<PrimExpr>("x2"), sig::arg<PrimExpr>("x3"));
  DefinePtxStatement("tl.cluster_arrive_relaxed");
  DefinePtxStatement("tl.cluster_arrive");
  DefinePtxStatement("tl.cluster_wait");
  DefinePtxStatement("tl.cluster_sync");
  DefineCudaIntegerQuery("tl.block_rank_in_cluster", PrimType::Int(32));
  DefinePtxStatement("tl.clc_try_cancel", sig::arg<PointerExpr>("result"),
                     sig::arg<PointerExpr>("barrier"));
  DefinePtxStatement("tl.clc_try_cancel_multicast",
                     sig::arg<PointerExpr>("result"),
                     sig::arg<PointerExpr>("barrier"));
  DefineCudaIntegerQuery("tl.clc_is_canceled", PrimType::Int(32),
                         sig::arg<PointerExpr>("result_pointer"));
  DefineCudaIntegerQuery("tl.clc_get_first_ctaid_x", PrimType::UInt(32),
                         sig::arg<PointerExpr>("result_pointer"));
  DefineCudaIntegerQuery("tl.clc_get_first_ctaid_y", PrimType::UInt(32),
                         sig::arg<PointerExpr>("result_pointer"));
  DefineCudaIntegerQuery("tl.clc_get_first_ctaid_z", PrimType::UInt(32),
                         sig::arg<PointerExpr>("result_pointer"));
  DefinePtxStatement("tl.named_barrier_arrive", sig::arg<IntExpr>("barrier_id"),
                     sig::arg<IntExpr>("thread_count"));
  DefinePtxStatement("tl.pdl_trigger");
  DefinePtxStatement("tl.pdl_sync");
  DefineCudaIntegerQuery("tl.tl_shuffle_elect", PrimType::Bool(),
                         sig::arg<IntExpr>("thread_extent"));
  DefinePtxStatement(
      "tl.initialize_wgmma_descriptor", sig::arg<Expr>("descriptor"),
      sig::arg<PointerExpr>("start_address"), sig::arg<IntExpr>("layout_type"),
      sig::arg<IntExpr>("leading_byte_offset"),
      sig::arg<IntExpr>("stride_byte_offset"));
  DefinePtxStatement(
      "tl.initialize_tcgen05_descriptor", sig::arg<Expr>("descriptor"),
      sig::arg<PointerExpr>("start_address"),
      sig::arg<IntExpr>("leading_byte_offset"),
      sig::arg<IntExpr>("stride_byte_offset"), sig::arg<IntExpr>("base_offset"),
      sig::arg<IntExpr>("leading_is_absolute"), sig::arg<IntExpr>("swizzle"));
  DefinePtxStatement("tl.increase_descriptor_offset",
                     sig::arg<Expr>("descriptor"), sig::arg<IntExpr>("offset"));
  DefinePtxStatement("tl.tcgen05_mma_arrive", sig::arg<Expr>("barrier"));
  DefineTensorMemoryTransfer("tl.tcgen05_ld");
  DefineTensorMemoryTransfer("tl.tcgen05_st");
  DefineCudaIntegerQuery("tl.__ffs", PrimType::Int(32),
                         sig::arg<IntExpr>("value"));
  DefineCudaIntegerQuery("tl.__fns", PrimType::UInt(32),
                         sig::arg<IntExpr>("mask"), sig::arg<IntExpr>("base"),
                         sig::arg<IntExpr>("offset"));
  DefinePtxMemory<32, true, false>("tl.ldg32");
  DefinePtxMemory<64, true, false>("tl.ldg64");
  DefinePtxMemory<128, true, false>("tl.ldg128");
  DefinePtxMemory<32, false, false>("tl.lds32");
  DefinePtxMemory<64, false, false>("tl.lds64");
  DefinePtxMemory<128, false, false>("tl.lds128");
  DefinePtxMemory<256, true, false>("tl.ldg256");
  DefinePtxMemory<32, true, true>("tl.stg32");
  DefinePtxMemory<64, true, true>("tl.stg64");
  DefinePtxMemory<128, true, true>("tl.stg128");
  DefinePtxMemory<32, false, true>("tl.sts32");
  DefinePtxMemory<64, false, true>("tl.sts64");
  DefinePtxMemory<128, false, true>("tl.sts128");
  DefinePtxMemory<256, true, true>("tl.stg256");
  DefinePtxStatement("tl.ptx_cluster_store", sig::arg<PointerExpr>("buffer"),
                     sig::arg<PrimExpr>("value"),
                     sig::arg<IntExpr>("destination_cta"),
                     sig::arg<IntExpr>("index"));
  DefinePtxStatement(
      "tl.tma_store_cluster", sig::arg<PointerExpr>("destination"),
      sig::arg<PointerExpr>("source"), sig::arg<IntExpr>("destination_cta"),
      sig::arg<IntExpr>("bytes"), sig::arg<Expr>("barrier"));
  DefineCudaIntegerQuery("tl.mvb_stage_index", PrimType::Int(32),
                         sig::arg<IntExpr>("stage"));
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

const Op &wmma_load_matrix_sync() {
  static const Op op = Op::Get("tl.wmma_load_matrix_sync");
  return op;
}

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
