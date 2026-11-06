#include "ir/vectorize.h"
#include <tvm/ffi/extra/structural_visit.h>
/*!
 * \file tl/op/builtin.cc
 * \brief Native registration of TileLang intrinsic contracts.
 */

#include "../ir/tensor_utils.h"
#include "builtin.h"
#include "builtin_registry.h"
#include "ir/annotations.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl {

TIR_DEFINE_TL_BUILTIN(lookup_param)

namespace {
Type InferAccessPointer(const CallObj *call) {
  const TensorLoad load = call->args[0].as_or_throw<TensorLoad>();
  return load->source.as_or_throw<tirx::TensorVar>().ty()->DataPtrType();
}

void ValidateAccessPointer(const CallObj *call) {
  ICHECK_EQ(call->args.size(), 3U);
  ICHECK(call->ty_args.empty());
  ICHECK(!call->attrs.has_value());
  const TensorLoad load = call->args[0].as_or_throw<TensorLoad>();
  auto tensor = load->source.as_or_throw<tirx::TensorVar>();
  ICHECK_EQ(load->indices.size(), tensor.shape().size());
  auto extent = call->args[1].as_or_throw<IntExpr>();
  ICHECK(extent.ty().IsScalar());
  auto mask = call->args[2].as_or_throw<IntImm>()->value.as<int64_t>();
  ICHECK(mask.has_value() && mask.value() >= 0 && mask.value() <= 3);
  auto pointer = call->ty.as_or_throw<PtrType>();
  ICHECK(pointer->element_type.as_or_throw<PrimType>() == tensor.dtype());
  ICHECK(pointer->scope == tensor.scope());
}

Type InferRawAccessPointer(const CallObj *call) {
  return call->args[0]->ty.as_or_throw<PtrType>();
}

void ValidateRawAccessPointer(const CallObj *call) {
  TVM_FFI_CHECK_EQ(call->args.size(), 4U, ValueError);
  TVM_FFI_CHECK(call->ty_args.empty() && !call->attrs.has_value(), ValueError);
  const PtrType pointer = call->args[0]->ty.as_or_throw<PtrType>();
  TVM_FFI_CHECK(!pointer->element_type.as_or_throw<PrimType>().IsVoid(),
                TypeError)
      << "tl.ptr_access needs a typed element pointer";
  const IntExpr offset = call->args[1].as_or_throw<IntExpr>();
  const IntExpr extent = call->args[2].as_or_throw<IntExpr>();
  TVM_FFI_CHECK(offset.ty().IsScalar() && extent.ty().IsScalar(), TypeError);
  if (auto constant = extent.as<IntImm>()) {
    TVM_FFI_CHECK_GE(constant.value()->value, 0, ValueError)
        << "tl.ptr_access extent cannot be negative";
  }
  const auto mask = call->args[3].as_or_throw<IntImm>()->value.as<int64_t>();
  TVM_FFI_CHECK(mask && *mask >= 0 && *mask <= 3, ValueError)
      << "tl.ptr_access rw_mask must be a constant in [0, 3]";
  TVM_FFI_CHECK(ffi::StructuralEqual()(call->ty, pointer), TypeError)
      << "tl.ptr_access must preserve pointer element type and scope";
}

Type InferSamePrimitive(const CallObj *call) {
  ICHECK(!call->args.empty());
  return call->args[0].as_or_throw<PrimExpr>().ty();
}

// Result hints are optional on typed native rewrites, but cannot change an
// operation's semantic result type.
void ValidatePrimitiveResult(const CallObj *call, PrimType result) {
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  if (!call->ty_args.empty())
    TVM_FFI_CHECK_EQ(call->ty_args[0].as_or_throw<PrimType>(), result,
                     TypeError)
        << "Intrinsic result type disagrees with its type argument";
  TVM_FFI_CHECK_EQ(call->ty.as_or_throw<PrimType>(), result, TypeError)
      << "Intrinsic result type disagrees with its operands";
}

Type InferWarpReduce(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 1U, ValueError)
      << "Warp reduction expects one runtime operand";
  const auto result = args[0].as_or_throw<PrimExpr>().ty();
  TVM_FFI_CHECK(!result.IsVoid(), TypeError);
  return result;
}

void ValidateWarpReduce(const CallObj *call) {
  ValidatePrimitiveResult(call, InferWarpReduce(call).as_or_throw<PrimType>());
}

void DefineWarpReduce(const char *name) {
  OpDef def(name);
  def.set_validator(ffi::reflection::NativeFunctionView<void(
                        const CallObj *)>::FromNative<&ValidateWarpReduce>())
      .signature(sig::arg<PrimExpr>("value"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false)
      .set_native_fn_attr<&InferWarpReduce>(tvm::op_attr::kInferType);
}

Type InferShuffleResult(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 4U, ValueError)
      << "Warp shuffle expects four runtime operands: mask, value, lane "
         "selector, width";
  const auto result = args[1].as_or_throw<PrimExpr>().ty();
  TVM_FFI_CHECK(!result.IsVoid(), TypeError);
  return result;
}

void ValidateShuffle(const CallObj *call) {
  ValidatePrimitiveResult(call,
                          InferShuffleResult(call).as_or_throw<PrimType>());
  const auto args = DecodeCallOperands(call);
  for (int i : {0, 2, 3})
    TVM_FFI_CHECK(args[i].as_or_throw<IntExpr>().ty().IsScalar(), TypeError)
        << "Warp shuffle mask, selector, and width must be scalar integers";
}

void DefineShuffle(const char *name) {
  OpDef def(name);
  def.set_validator(ffi::reflection::NativeFunctionView<void(
                        const CallObj *)>::FromNative<&ValidateShuffle>())
      .signature(sig::arg<IntExpr>("mask"), sig::arg<PrimExpr>("value"),
                 sig::arg<IntExpr>("lane"), sig::arg<IntExpr>("width"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false)
      .set_native_fn_attr<&InferShuffleResult>(tvm::op_attr::kInferType);
}

template <int N, bool HasMask, bool HasPredicate>
void ValidateCollective(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), N, ValueError)
      << "Collective has the wrong number of runtime operands";
  const auto result =
      Op::GetAttrMap(tvm::op_attr::kFixedReturnType)[call->op.as_or_throw<Op>()]
          .as_or_throw<PrimType>();
  ValidatePrimitiveResult(call, result);
  for (const Expr &arg : args) {
    const auto dtype = arg.as_or_throw<PrimExpr>().ty();
    TVM_FFI_CHECK(dtype.IsScalar() && !dtype.IsVoid(), TypeError)
        << "Collective operands must be scalar values";
  }
  if constexpr (HasMask) {
    const auto dtype = args[0].as_or_throw<PrimExpr>().ty();
    TVM_FFI_CHECK(dtype.MatchesCode(kDLInt, kDLUInt) && dtype.bits() == 32,
                  TypeError)
        << "Collective mask must be a 32-bit integer";
  }
  if constexpr (HasPredicate)
    TVM_FFI_CHECK(args.back().as_or_throw<PrimExpr>().ty().MatchesCode(
                      kDLInt, kDLUInt, kDLBool),
                  TypeError)
        << "Collective requires an integer or boolean predicate";
}

template <int N, bool HasMask = false, bool HasPredicate = false>
void DefineCollective(const char *name, PrimType result) {
  OpDef def(name);
  def.set_validator(
         ffi::reflection::NativeFunctionView<void(const CallObj *)>::FromNative<
             &ValidateCollective<N, HasMask, HasPredicate>>())
      .set_attr(tvm::op_attr::kFixedReturnType, result)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
  if constexpr (N == 0)
    def.signature(sig::var_args<tvm::Tuple>("annotation_payload"),
                  sig::var_ty_args<PrimType>("result_type"));
  else if constexpr (N == 1)
    def.signature(sig::arg<PrimExpr>("predicate"),
                  sig::var_args<tvm::Tuple>("annotation_payload"),
                  sig::var_ty_args<PrimType>("result_type"));
  else
    def.signature(sig::arg<IntExpr>("mask"),
                  sig::arg<PrimExpr>(HasPredicate ? "predicate" : "value"),
                  sig::var_args<tvm::Tuple>("annotation_payload"),
                  sig::var_ty_args<PrimType>("result_type"));
}

void ValidateClamp(const CallObj *call) {
  auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 3U, ValueError)
      << "tl.clamp requires x, minimum and maximum";
  const PrimType dtype = args[0].as_or_throw<PrimExpr>().ty();
  TVM_FFI_CHECK(!dtype.IsVoid(), TypeError);
  TVM_FFI_CHECK_EQ(call->ty, dtype, TypeError);
  for (const Expr &arg : args)
    TVM_FFI_CHECK_EQ(arg.as_or_throw<PrimExpr>().ty(), dtype, TypeError)
        << "tl.clamp operands must have matching types";
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  if (!call->ty_args.empty())
    TVM_FFI_CHECK_EQ(call->ty_args[0], dtype, TypeError);
}

Type InferCastResult(const CallObj *call) {
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError)
      << "tl.cast accepts one destination type";
  // Typed native rewrites may supply Call.ty instead of a type argument.
  const Type result = call->ty_args.empty() ? call->ty : call->ty_args[0];
  const auto primitive = result.as<PrimType>();
  TVM_FFI_CHECK(primitive && !primitive.value().IsVoid(), TypeError)
      << "tl.cast requires a non-void primitive destination type";
  return primitive.value();
}

void ValidateCast(const CallObj *call) {
  TVM_FFI_CHECK(call->args.size() == 1 || call->args.size() == 2, ValueError)
      << "tl.cast expects a value and optional stochastic random bits";
  const PrimType result = InferCastResult(call).as_or_throw<PrimType>();
  TVM_FFI_CHECK_EQ(call->ty, result, TypeError)
      << "tl.cast result disagrees with its destination type argument";
  const PrimType source = call->args[0].as_or_throw<PrimExpr>().ty();
  TVM_FFI_CHECK(!source.IsVoid(), TypeError)
      << "tl.cast requires a non-void primitive value";
  TVM_FFI_CHECK_EQ(source.lanes(), result.lanes(), TypeError)
      << "tl.cast preserves the number of value lanes";
  TVM_FFI_CHECK(call->attrs.has_value() &&
                    call->attrs.value().as<DictAttrsObj>(),
                TypeError)
      << "tl.cast requires dictionary attributes";
  const auto attrs = call->attrs.value().as_or_throw<DictAttrs>();
  const auto rounding = attrs.GetAttr<ffi::String>("round").value_or("");
  // Read the option even though supported dtype/saturation combinations are
  // a backend contract. This rejects incorrectly typed raw IR attributes.
  (void)attrs.GetAttr<bool>("sat").value_or(true);
  TVM_FFI_CHECK(rounding == "" || rounding == "rn" || rounding == "rz" ||
                    rounding == "rp" || rounding == "rm" || rounding == "rs",
                ValueError)
      << "tl.cast has an invalid rounding mode: " << rounding;
  if (rounding == "rs") {
    TVM_FFI_CHECK_EQ(call->args.size(), 2U, ValueError)
        << "tl.cast round='rs' requires an rbits operand";
    const PrimType rbits = call->args[1].as_or_throw<PrimExpr>().ty();
    // Scalar conversion helpers consume uint32 random bits. The original
    // operand is converted at the helper call, including integer narrowing
    // and defined floating-point to unsigned-integer conversions.
    TVM_FFI_CHECK(rbits.IsScalar() && !rbits.IsVoid(), TypeError)
        << "tl.cast rbits must be a non-void scalar primitive value";
  } else {
    TVM_FFI_CHECK_EQ(call->args.size(), 1U, ValueError)
        << "tl.cast rbits is only valid with round='rs'";
  }
}

// The shared vectorizer simplifies rewritten tensor indices. A Broadcast
// index is a proof that every lane reads the same random-bits element; retain
// one scalar read instead of passing a vector to the PTX conversion helper.
ffi::Optional<PrimExpr> UniformCastRandomBits(PrimExpr bits, const Var &lane) {
  if (const auto *broadcast = bits.as<prim::BroadcastObj>()) {
    bits = broadcast->value;
  } else if (!bits.ty().IsScalar()) {
    const auto load = bits.as<TensorLoad>();
    if (!load)
      return std::nullopt;
    ffi::Array<PrimExpr> indices;
    for (PrimExpr index : load.value()->indices) {
      if (!index.ty().IsScalar()) {
        const auto *broadcast = index.as<prim::BroadcastObj>();
        if (!broadcast)
          return std::nullopt;
        index = broadcast->value;
      }
      indices.push_back(index);
    }
    bits = TensorLoad(load.value()->source, indices, load.value()->loc);
  }
  if (!bits.ty().IsScalar() ||
      ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
          bits,
          [&](const VarObj *var) -> ffi::Expected<ffi::WalkResult> {
            return (var == lane.get())
                       ? ffi::WalkResult::Interrupt(
                             ffi::VisitInterrupt(ffi::GetRef<Var>(var)))
                       : ffi::WalkResult::Skip();
          })
          .has_value())
    return std::nullopt;
  return bits;
}

ffi::Optional<Expr> VectorizeCast(Call call, Var lane, PrimExpr,
                                  ffi::Optional<Target>,
                                  ffi::TypedFunction<Expr(Expr)> rewrite) {
  ValidateCast(call.get());
  if (call->args.size() == 2 &&
      SideEffect(call->args[1]) > CallEffectKind::kReadState) {
    // Fresh randomness must run once per scalar conversion. A vector helper
    // would share the result and change both the RNG stream and its effects.
    return std::nullopt;
  }
  const PrimExpr value = rewrite(call->args[0]).as_or_throw<PrimExpr>();
  if (value.ty().IsScalableVector())
    return std::nullopt;
  ffi::Array<Expr> operands{value};
  if (call->args.size() == 2) {
    auto bits = UniformCastRandomBits(
        rewrite(call->args[1]).as_or_throw<PrimExpr>(), lane);
    if (!bits)
      return std::nullopt;
    operands.push_back(bits.value());
  }
  const PrimType destination =
      call->ty.as_or_throw<PrimType>().WithLanes(value.ty().lanes());
  auto result = Call(destination, call->op, operands, call->attrs,
                     {destination}, call->loc);
  if (ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
          result,
          [&](const VarObj *var) -> ffi::Expected<ffi::WalkResult> {
            return (var == lane.get())
                       ? ffi::WalkResult::Interrupt(
                             ffi::VisitInterrupt(ffi::GetRef<Var>(var)))
                       : ffi::WalkResult::Skip();
          })
          .has_value())
    return std::nullopt;
  result.Validate();
  return result;
}

// Atomic values are determined by the destination address, never by a
// caller-supplied dtype. Non-returning atomics are statements with a void
// result; load/return_prev preserve the destination element dtype.
enum class AtomicContract { kReadModifyWrite, kLoad, kStore, kBitwiseOr };

template <int Lanes, bool ReturnsValue, AtomicContract Kind>
Type InferAtomicResult(const CallObj *call) {
  auto args = DecodeCallOperands(call);
  const size_t minimum = Kind == AtomicContract::kStore ? 3 : 2;
  const size_t maximum = Kind != AtomicContract::kLoad ? 3 : 2;
  TVM_FFI_CHECK(args.size() >= minimum && args.size() <= maximum, ValueError)
      << call->op << " has invalid atomic operands";
  const PtrType pointer = args[0]->ty.as_or_throw<PtrType>();
  const PrimType element = pointer->element_type.as_or_throw<PrimType>();
  TVM_FFI_CHECK(element.IsScalar() && !element.IsVoid(), TypeError)
      << "Atomic address must point to scalar primitive elements";
  if constexpr (ReturnsValue)
    return element.WithLanes(Lanes);
  return PrimType::Void();
}

template <int Lanes, bool ReturnsValue, AtomicContract Kind>
void ValidateAtomic(const CallObj *call) {
  const Type expected = InferAtomicResult<Lanes, ReturnsValue, Kind>(call);
  TVM_FFI_CHECK(ffi::StructuralEqual()(call->ty, expected), TypeError)
      << "Atomic result type must match its destination contract";
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  if (!call->ty_args.empty())
    TVM_FFI_CHECK(ffi::StructuralEqual()(call->ty_args[0], expected), TypeError)
        << "Atomic explicit result type disagrees with its destination";
  const auto args = DecodeCallOperands(call);
  const auto destination = args[0]->ty.as_or_throw<PtrType>();
  const auto element = destination->element_type.as_or_throw<PrimType>();
  if constexpr (Lanes > 1) {
    if (auto source = args[1]->ty.as<PtrType>()) {
      TVM_FFI_CHECK(ffi::StructuralEqual()((*source)->element_type, element),
                    TypeError)
          << "Vector atomic source and destination must have the same element "
             "type";
    } else {
      // The non-returning packed helper also consumes an already evaluated
      // vector. Keep its element type and lane count explicit: mixed scalar
      // contributions are converted before this call, never bit-reinterpreted.
      TVM_FFI_CHECK(!ReturnsValue, TypeError)
          << "Returning vector atomic requires a typed source pointer";
      const auto value = args[1].as_or_throw<PrimExpr>();
      TVM_FFI_CHECK(value.ty() == element.WithLanes(Lanes), TypeError)
          << "Vector atomic value must match the destination element type and "
             "lane count";
    }
  } else if constexpr (Kind != AtomicContract::kLoad) {
    const auto value = args[1].as_or_throw<PrimExpr>();
    TVM_FFI_CHECK(value.ty().IsScalar() && !value.ty().IsVoid(), TypeError)
        << "Scalar atomic value must be a scalar primitive expression";
    // Existing atomics convert the scalar contribution at the destination
    // dtype. Keep that conversion in the lowering consumer, including literals.
    if constexpr (Kind == AtomicContract::kBitwiseOr) {
      TVM_FFI_CHECK(element.MatchesCode(kDLInt, kDLUInt) &&
                        value.ty().MatchesCode(kDLInt, kDLUInt),
                    TypeError)
          << "atomic_or requires integer source and destination types";
    }
  }
  {
    const size_t order_index = Kind == AtomicContract::kLoad ? 1 : 2;
    if (args.size() > order_index) {
      const auto order =
          args[order_index].as_or_throw<IntImm>()->value.as<int64_t>();
      TVM_FFI_CHECK(order && *order >= 0 && *order <= 5, ValueError)
          << "Atomic memory_order must be an integer constant in [0, 5]";
      if constexpr (Kind == AtomicContract::kLoad)
        TVM_FFI_CHECK(*order == 0 || *order == 1 || *order == 2 || *order == 5,
                      ValueError)
            << "atomic_load cannot use release or acq_rel memory order";
      if constexpr (Kind == AtomicContract::kStore)
        TVM_FFI_CHECK(*order == 0 || *order == 3 || *order == 5, ValueError)
            << "atomic_store cannot use consume, acquire or acq_rel memory "
               "order";
    }
  }
}

template <int Lanes, bool ReturnsValue,
          AtomicContract Kind = AtomicContract::kReadModifyWrite>
void DefineAtomic(const char *name) {
  OpDef def(name);
  def.set_validator(
      ffi::reflection::NativeFunctionView<void(const CallObj *)>::FromNative<
          &ValidateAtomic<Lanes, ReturnsValue, Kind>>());
  def.signature(sig::var_args<Expr>("operands"),
                sig::var_ty_args<Type>("result_type"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_native_fn_attr<&InferAtomicResult<Lanes, ReturnsValue, Kind>>(
          tvm::op_attr::kInferType);
}

void ValidateElementCPAsync(const CallObj *call) {
  auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK(args.size() == 3 || args.size() == 4, ValueError)
      << "tl.ptx_cp_async expects three or four runtime operands";
  ValidatePrimitiveResult(call, PrimType::Void());
  // Counts remain in logical elements until vectorization chooses a final
  // byte transfer width.  In particular, this is not the bytes-form CUDA op.
  auto destination = args[0]->ty.as_or_throw<PtrType>();
  auto source = args[1]->ty.as_or_throw<PtrType>();
  auto destination_type = destination->element_type.as_or_throw<PrimType>();
  auto source_type = source->element_type.as_or_throw<PrimType>();
  TVM_FFI_CHECK(!destination_type.IsVoid() && !source_type.IsVoid(), TypeError)
      << "tl.ptx_cp_async requires typed element pointers";
  const IntExpr count = args[2].as_or_throw<IntExpr>();
  TVM_FFI_CHECK(count.ty().IsScalar(), TypeError);
  if (auto immediate = count.as<IntImm>())
    TVM_FFI_CHECK_GE(immediate.value()->value, 0, ValueError)
        << "tl.ptx_cp_async element count cannot be negative";
  if (args.size() == 4) {
    auto predicate = args[3].as_or_throw<PrimExpr>();
    TVM_FFI_CHECK(predicate.ty().MatchesCode(kDLBool), TypeError)
        << "tl.ptx_cp_async predicate must be boolean";
  }
}
Type InferRandomFloat(const CallObj *call) {
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1U, TypeError);
  const auto result = (call->ty_args.empty() ? call->ty : call->ty_args[0])
                          .as_or_throw<PrimType>();
  TVM_FFI_CHECK(result == PrimType::Float(32) || result == PrimType::Float(64),
                TypeError)
      << "Floating RNG requires float32 or float64";
  return result;
}

void ValidateRandomInit(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 4U, ValueError)
      << "RNG initialization expects four runtime operands";
  for (int i = 0; i < 3; ++i)
    TVM_FFI_CHECK(args[i].as_or_throw<IntExpr>().ty().IsScalar(), TypeError);
  args[3].as_or_throw<StringImm>();
  ValidatePrimitiveResult(call, PrimType::Void());
}

void ValidateRandomUInt(const CallObj *call) {
  TVM_FFI_CHECK(DecodeCallOperands(call).empty(), ValueError)
      << "Integer RNG expects no runtime operands";
  ValidatePrimitiveResult(call, PrimType::UInt(32));
}

void ValidateRandomFloat(const CallObj *call) {
  const auto args = DecodeCallOperands(call);
  TVM_FFI_CHECK_EQ(args.size(), 1U, ValueError)
      << "Floating RNG expects one runtime operand";
  const auto distribution = args[0].as_or_throw<StringImm>()->value;
  TVM_FFI_CHECK(distribution == "uniform" || distribution == "normal",
                ValueError)
      << "Floating RNG distribution must be uniform or normal";
  ValidatePrimitiveResult(call, InferRandomFloat(call).as_or_throw<PrimType>());
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  transform::PassContext::RegisterConfigOption<bool>(
      kDebugMergeSharedMemoryAllocations);
  transform::PassContext::RegisterConfigOption<bool>(
      kDisableSafeMemoryLegalize);
  transform::PassContext::RegisterConfigOption<bool>(kDisableThreadStorageSync);
  transform::PassContext::RegisterConfigOption<int64_t>(kConfigIndexBitwidth);
  transform::PassContext::RegisterConfigOption<bool>(
      kEnableAggressiveSharedMemoryMerge);
  transform::PassContext::RegisterConfigOption<bool>(kDisableSharedMemoryReuse);
  transform::PassContext::RegisterConfigOption<bool>(kForceLetInline);
  transform::PassContext::RegisterConfigOption<bool>(kEnableFastMath);
  transform::PassContext::RegisterConfigOption<bool>(kEnableAsyncCopy);
  transform::PassContext::RegisterConfigOption<bool>(kReducerForceBaseline);
  transform::PassContext::RegisterConfigOption<bool>(kEnableReducerPlanVerbose);
  transform::PassContext::RegisterConfigOption<ffi::String>(kLayoutCostModel);
  transform::PassContext::RegisterConfigOption<bool>(
      kEnableVectorizePlannerVerbose);
  transform::PassContext::RegisterConfigOption<bool>(
      kStorageRewriteDetectInplace);
  transform::PassContext::RegisterConfigOption<bool>(kASTPrintEnable);
  transform::PassContext::RegisterConfigOption<bool>(
      kLayoutVisualizationEnable);
  transform::PassContext::RegisterConfigOption<ffi::String>(
      kLayoutVisualizationFormats);
  transform::PassContext::RegisterConfigOption<ffi::Array<ffi::String>>(
      kDeviceCompileFlags);
  transform::PassContext::RegisterConfigOption<bool>(kEmitLineDirectives);
  transform::PassContext::RegisterConfigOption<bool>(kDisableDataRaceCheck);
  transform::PassContext::RegisterConfigOption<bool>(kDisableBufferInitCheck);
  transform::PassContext::RegisterConfigOption<bool>(kDisableLoopUnswitching);
  transform::PassContext::RegisterConfigOption<bool>(
      kLoopUnswitchingAllowNonTrivialElse);
  transform::PassContext::RegisterConfigOption<bool>(
      kIfStmtBindingInlineReplayableBinds);
  transform::PassContext::RegisterConfigOption<bool>(kDisableOutOfBoundWarning);
  transform::PassContext::RegisterConfigOption<bool>(kEnableDumpIR);
  transform::PassContext::RegisterConfigOption<ffi::String>(kDumpIRDir);
  transform::PassContext::RegisterConfigOption<bool>(kPassProfile);
  transform::PassContext::RegisterConfigOption<double>(kPassProfileThresholdMs);
}

TIR_DEFINE_TL_BUILTIN(tvm_ffi_call_with_result)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.tvm_ffi_call_with_result")
      .signature(sig::arg<Expr>("arg0"), sig::arg<Expr>("arg1"),
                 sig::arg<Expr>("arg2"), sig::arg<Expr>("arg3"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

TIR_DEFINE_TL_BUILTIN(access_ptr)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.access_ptr")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateAccessPointer>())
      .signature(sig::arg<Expr>("arg0"), sig::arg<Expr>("arg1"),
                 sig::arg<Expr>("arg2"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_native_fn_attr<&InferAccessPointer>(tvm::op_attr::kInferType);
}

TIR_DEFINE_TL_BUILTIN(ptr_access)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.ptr_access")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateRawAccessPointer>())
      .signature(sig::arg<Expr>("pointer"), sig::arg<IntExpr>("element_offset"),
                 sig::arg<IntExpr>("extent"), sig::arg<IntImm>("rw_mask"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_native_fn_attr<&InferRawAccessPointer>(tvm::op_attr::kInferType);
}

TIR_DEFINE_TL_BUILTIN(clamp)
TIR_DEFINE_TL_BUILTIN(storage_sync)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.clamp")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateClamp>())
      .signature(sig::var_args<Expr>("operands"),
                 sig::var_ty_args<Type>("result_type"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_attr(tirx::op_attr::kVectorizable, true)
      .set_native_fn_attr<&InferSamePrimitive>(tvm::op_attr::kInferType);
}

const Op &cast_op() {
  static const Op op = Op::Get("tl.cast");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.cast")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateCast>())
      .signature(sig::var_args<PrimExpr>("operands"),
                 sig::var_ty_args<PrimType>("destination_type"),
                 sig::call_attrs<DictAttrsObj>())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      // Random bits describe one conversion chunk, not a lane-wise operand.
      // Only the dedicated rule may widen the call; generic broadcasting
      // would turn this scalar operand into a vector.
      .set_attr(tirx::op_attr::kVectorizable, false)
      .set_attr(tl::kCallVectorize, tl::FCallVectorize(VectorizeCast))
      .set_native_fn_attr<&InferCastResult>(tvm::op_attr::kInferType);
}

TIR_DEFINE_TL_BUILTIN(region)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.region")
      .signature(sig::var_args<Expr>("args"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure);
}

TIR_DEFINE_TL_BUILTIN(launch_thread_idx)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.launch_thread_idx")
      .signature(sig::arg<Expr>("arg0"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

TIR_DEFINE_TL_BUILTIN(add2)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.add2")
      .signature(sig::arg<PrimExpr>("arg0"), sig::arg<PrimExpr>("arg1"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_native_fn_attr<&InferSamePrimitive>(tvm::op_attr::kInferType);
}

TIR_DEFINE_TL_BUILTIN(sub2)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.sub2")
      .signature(sig::arg<PrimExpr>("arg0"), sig::arg<PrimExpr>("arg1"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_native_fn_attr<&InferSamePrimitive>(tvm::op_attr::kInferType);
}

TIR_DEFINE_TL_BUILTIN(mul2)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.mul2")
      .signature(sig::arg<PrimExpr>("arg0"), sig::arg<PrimExpr>("arg1"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_native_fn_attr<&InferSamePrimitive>(tvm::op_attr::kInferType);
}

TIR_DEFINE_TL_BUILTIN(fma2)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.fma2")
      .signature(sig::arg<PrimExpr>("arg0"), sig::arg<PrimExpr>("arg1"),
                 sig::arg<PrimExpr>("arg2"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_native_fn_attr<&InferSamePrimitive>(tvm::op_attr::kInferType);
}

TIR_DEFINE_TL_BUILTIN(max2)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.max2")
      .signature(sig::arg<PrimExpr>("arg0"), sig::arg<PrimExpr>("arg1"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_native_fn_attr<&InferSamePrimitive>(tvm::op_attr::kInferType);
}

TIR_DEFINE_TL_BUILTIN(min2)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.min2")
      .signature(sig::arg<PrimExpr>("arg0"), sig::arg<PrimExpr>("arg1"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_native_fn_attr<&InferSamePrimitive>(tvm::op_attr::kInferType);
}

TIR_DEFINE_TL_BUILTIN(abs2)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.abs2")
      .signature(sig::arg<PrimExpr>("arg0"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_native_fn_attr<&InferSamePrimitive>(tvm::op_attr::kInferType);
}

TIR_DEFINE_TL_BUILTIN(mbarrier_wait_parity)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.mbarrier_wait_parity")
      .signature(sig::arg<Expr>("arg0"), sig::arg<Expr>("arg1"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

TIR_DEFINE_TL_BUILTIN(mbarrier_expect_tx)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.mbarrier_expect_tx")
      .signature(sig::arg<Expr>("arg0"), sig::arg<Expr>("arg1"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

TIR_DEFINE_TL_BUILTIN(ptx_stmatrix)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.ptx_stmatrix")
      .signature(sig::var_args<Expr>("args"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

TIR_DEFINE_TL_BUILTIN(ptx_cp_async)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef def("tl.ptx_cp_async");
  def.set_validator(
         ffi::reflection::NativeFunctionView<void(
             const CallObj *)>::FromNative<&ValidateElementCPAsync>())
      .signature(sig::arg<Expr>("destination"), sig::arg<Expr>("source"),
                 sig::arg<IntExpr>("element_count"),
                 sig::var_args<Expr>("predicate_and_annotations"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

TIR_DEFINE_TL_BUILTIN(annotate_producer_reg_dealloc)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.annotate_producer_reg_dealloc")
      .signature(sig::arg<Expr>("arg0"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

TIR_DEFINE_TL_BUILTIN(annotate_consumer_reg_alloc)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.annotate_consumer_reg_alloc")
      .signature(sig::arg<Expr>("arg0"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

TIR_DEFINE_TL_BUILTIN(no_set_max_nreg)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.no_set_max_nreg")
      .signature()
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

TIR_DEFINE_TL_BUILTIN(wait_wgmma)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.wait_wgmma")
      .signature(sig::arg<Expr>("arg0"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

TIR_DEFINE_TL_BUILTIN(pack_b16)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.pack_b16")
      .signature(sig::arg<Expr>("arg0"), sig::arg<Expr>("arg1"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure);
}

TIR_DEFINE_TL_BUILTIN(sync_grid)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.sync_grid")
      .signature()
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

TIR_DEFINE_TL_BUILTIN(sync_warp)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.sync_warp")
      .signature(sig::var_args<Expr>("args"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

TIR_DEFINE_TL_BUILTIN(any_sync)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineCollective<2, true, true>("tl.any_sync", PrimType::Int(32));
}

TIR_DEFINE_TL_BUILTIN(all_sync)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineCollective<2, true, true>("tl.all_sync", PrimType::Int(32));
}

TIR_DEFINE_TL_BUILTIN(ballot_sync)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineCollective<2, true, true>("tl.ballot_sync", PrimType::UInt(64));
}

TIR_DEFINE_TL_BUILTIN(ballot)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineCollective<1, false, true>("tl.ballot", PrimType::UInt(64));
}

TIR_DEFINE_TL_BUILTIN(activemask)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineCollective<0, false, false>("tl.activemask", PrimType::UInt(64));
}

TIR_DEFINE_TL_BUILTIN(syncthreads_count)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineCollective<1, false, true>("tl.syncthreads_count", PrimType::Int(32));
}

TIR_DEFINE_TL_BUILTIN(syncthreads_and)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineCollective<1, false, true>("tl.syncthreads_and", PrimType::Int(32));
}

TIR_DEFINE_TL_BUILTIN(syncthreads_or)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineCollective<1, false, true>("tl.syncthreads_or", PrimType::Int(32));
}

TIR_DEFINE_TL_BUILTIN(shfl_sync)

TVM_FFI_STATIC_INIT_BLOCK() { DefineShuffle("tl.shfl_sync"); }

TIR_DEFINE_TL_BUILTIN(shfl_xor_sync)

TVM_FFI_STATIC_INIT_BLOCK() { DefineShuffle("tl.shfl_xor_sync"); }

TIR_DEFINE_TL_BUILTIN(shfl_down_sync)

TVM_FFI_STATIC_INIT_BLOCK() { DefineShuffle("tl.shfl_down_sync"); }

TIR_DEFINE_TL_BUILTIN(shfl_up_sync)

TVM_FFI_STATIC_INIT_BLOCK() { DefineShuffle("tl.shfl_up_sync"); }

TIR_DEFINE_TL_BUILTIN(match_any_sync)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineCollective<2, true, false>("tl.match_any_sync", PrimType::UInt(32));
}

TIR_DEFINE_TL_BUILTIN(match_all_sync)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineCollective<2, true, false>("tl.match_all_sync", PrimType::UInt(32));
}

TIR_DEFINE_TL_BUILTIN(loop_break)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.loop_break")
      .signature()
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

TIR_DEFINE_TL_BUILTIN(atomic_add_elem_op)

TVM_FFI_STATIC_INIT_BLOCK() { DefineAtomic<1, false>("tl.atomic_add_elem_op"); }

TIR_DEFINE_TL_BUILTIN(atomic_add_ret_elem_op)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineAtomic<1, true>("tl.atomic_add_ret_elem_op");
}

TIR_DEFINE_TL_BUILTIN(atomic_addx2_elem_op)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineAtomic<2, false>("tl.atomic_addx2_elem_op");
}

TIR_DEFINE_TL_BUILTIN(atomic_addx2_ret_elem_op)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineAtomic<2, true>("tl.atomic_addx2_ret_elem_op");
}

TIR_DEFINE_TL_BUILTIN(atomic_addx4_elem_op)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineAtomic<4, false>("tl.atomic_addx4_elem_op");
}

TIR_DEFINE_TL_BUILTIN(atomic_addx4_ret_elem_op)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineAtomic<4, true>("tl.atomic_addx4_ret_elem_op");
}

TIR_DEFINE_TL_BUILTIN(atomic_load_elem_op)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineAtomic<1, true, AtomicContract::kLoad>("tl.atomic_load_elem_op");
}

TIR_DEFINE_TL_BUILTIN(atomic_store_elem_op)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineAtomic<1, false, AtomicContract::kStore>("tl.atomic_store_elem_op");
}

TIR_DEFINE_TL_BUILTIN(atomic_or_elem_op)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineAtomic<1, false, AtomicContract::kBitwiseOr>("tl.atomic_or_elem_op");
}

TIR_DEFINE_TL_BUILTIN(atomic_max_elem_op)

TVM_FFI_STATIC_INIT_BLOCK() { DefineAtomic<1, false>("tl.atomic_max_elem_op"); }

TIR_DEFINE_TL_BUILTIN(atomic_max_ret_elem_op)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineAtomic<1, true>("tl.atomic_max_ret_elem_op");
}

TIR_DEFINE_TL_BUILTIN(atomic_min_elem_op)

TVM_FFI_STATIC_INIT_BLOCK() { DefineAtomic<1, false>("tl.atomic_min_elem_op"); }

TIR_DEFINE_TL_BUILTIN(atomic_min_ret_elem_op)

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineAtomic<1, true>("tl.atomic_min_ret_elem_op");
}

TIR_DEFINE_TL_BUILTIN(warp_reduce_sum)

TVM_FFI_STATIC_INIT_BLOCK() { DefineWarpReduce("tl.warp_reduce_sum"); }

TIR_DEFINE_TL_BUILTIN(warp_reduce_max)

TVM_FFI_STATIC_INIT_BLOCK() { DefineWarpReduce("tl.warp_reduce_max"); }

TIR_DEFINE_TL_BUILTIN(warp_reduce_min)

TVM_FFI_STATIC_INIT_BLOCK() { DefineWarpReduce("tl.warp_reduce_min"); }

TIR_DEFINE_TL_BUILTIN(warp_reduce_bitand)

TVM_FFI_STATIC_INIT_BLOCK() { DefineWarpReduce("tl.warp_reduce_bitand"); }

TIR_DEFINE_TL_BUILTIN(warp_reduce_bitor)

TVM_FFI_STATIC_INIT_BLOCK() { DefineWarpReduce("tl.warp_reduce_bitor"); }

TIR_DEFINE_TL_BUILTIN(__ldg)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.__ldg")
      .signature(sig::var_args<Expr>("args"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure);
}

TIR_DEFINE_TL_BUILTIN(rng_init)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.rng_init")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateRandomInit>())
      .signature(sig::arg<IntExpr>("seed"), sig::arg<IntExpr>("sequence"),
                 sig::arg<IntExpr>("offset"), sig::arg<StringImm>("generator"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

TIR_DEFINE_TL_BUILTIN(rng_rand)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.rng_rand")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateRandomUInt>())
      .signature(sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(32))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

TIR_DEFINE_TL_BUILTIN(rng_rand_float)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.rng_rand_float")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateRandomFloat>())
      .signature(sig::arg<StringImm>("distribution"),
                 sig::var_args<tvm::Tuple>("annotation_payload"),
                 sig::var_ty_args<PrimType>("result_type"))
      .set_native_fn_attr<&InferRandomFloat>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tirx::op_attr::kVectorizable, false);
}

TIR_DEFINE_TL_BUILTIN(device_assert)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.device_assert")
      .signature(sig::arg<Expr>("arg0"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

TIR_DEFINE_TL_BUILTIN(device_assert_with_msg)

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.device_assert_with_msg")
      .signature(sig::arg<Expr>("arg0"), sig::arg<Expr>("arg1"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

} // namespace tvm::tl
