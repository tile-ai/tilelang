/*!
 * \file tl/op/operator.cc
 * \brief Native typed tile-operator registration and decoding.
 */

#include "operator.h"

#include <tvm/ffi/reflection/registry.h>

#include "builtin.h"

namespace tvm::tl {

namespace {

void ValidateTileOperatorCall(const CallObj *call) {
  auto decoded = DecodeTileOpArguments(call);
  const Op op = call->op.as_or_throw<Op>();
  auto builders = Op::GetAttrMap<OpBuilderFunc>("TLOpBuilder");
  ICHECK(builders.count(op)) << "No builder registered for " << op->name;
  // A builder checks operand categories, arity and static semantic constraints.
  // It must not allocate runtime resources or run target-dependent lowering.
  ICHECK(builders[op](decoded.operands, decoded.annotations).defined());
  ICHECK(call->ty.as<PrimType>().has_value() &&
         call->ty.as_or_throw<PrimType>().IsVoid())
      << op->name << " must return void";
}

} // namespace

void RegisterTileOperator(const ffi::String &name, OpBuilderFunc builder,
                          ffi::Optional<OpBlockAnnotationHandlerFunc> handler) {
  OpDef definition(name);
  definition
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateTileOperatorCall>())
      .signature(sig::var_args<Expr>("operands_and_annotations"),
                 sig::call_attrs<DictAttrsObj>())
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kUpdateState)
      .set_attr<OpBuilderFunc>("TLOpBuilder", builder);
  if (handler.has_value()) {
    definition.set_attr<OpBlockAnnotationHandlerFunc>(
        kTLOpBlockAnnotationHandler, handler.value());
  }
}

TileOperator ParseOperator(const Call &call,
                           const BlockAnnotations &block_annotations) {
  auto op = call->op.as<Op>();
  if (!op.has_value() || !Op::HasAttrMap("TLOpBuilder")) {
    return TileOperator();
  }
  auto builders = Op::GetAttrMap<OpBuilderFunc>("TLOpBuilder");
  if (!builders.count(op.value())) {
    return TileOperator();
  }
  auto decoded = DecodeTileOpArguments(call.get());
  auto result = builders[op.value()](decoded.operands, decoded.annotations);
  ICHECK(result.defined());
  if (Op::HasAttrMap(kTLOpBlockAnnotationHandler)) {
    auto handlers = Op::GetAttrMap<OpBlockAnnotationHandlerFunc>(
        kTLOpBlockAnnotationHandler);
    if (handlers.count(op.value())) {
      result = handlers[op.value()](result, block_annotations);
      ICHECK(result.defined());
    }
  }
  return result;
}

TileOperator ParseOperator(const Stmt &stmt,
                           const BlockAnnotations &block_annotations) {
  if (const auto *evaluate = stmt.as<EvaluateObj>()) {
    if (auto call = evaluate->value.as<Call>()) {
      return ParseOperator(call.value(), block_annotations);
    }
  }
  return TileOperator();
}

tirx::TensorVar GetTensorFromAccessPtr(const Expr &expr) {
  const Call call = expr.as_or_throw<Call>();
  ICHECK(call->op.same_as(tl::access_ptr()))
      << "Expected tl.access_ptr carrying a TensorLoad, got " << expr;
  ICHECK_EQ(call->args.size(), 3U);
  const TensorLoad load = call->args[0].as_or_throw<TensorLoad>();
  return load->source.as_or_throw<tirx::TensorVar>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def_packed(
      "tl.GetTileOpAccessRegions", [](ffi::PackedArgs args, ffi::Any *result) {
        TVM_FFI_CHECK(args.size() == 1 || args.size() == 2, TypeError)
            << "GetTileOpAccessRegions expects a Call and optional block "
               "annotations";
        Call call = args[0].cast<Call>();
        BlockAnnotations annotations = args.size() == 2
                                           ? args[1].cast<BlockAnnotations>()
                                           : BlockAnnotations();
        TileOperator operation = ParseOperator(call, annotations);
        TVM_FFI_CHECK(operation.defined(), ValueError)
            << "GetTileOpAccessRegions requires a registered TileLang tile "
               "operator";
        // Query the same operator and lexical annotation handler as lowering.
        // No target selection or layout inference is needed to report effects.
        AccessRegions accesses = operation->GetAccessRegions();
        *result = ffi::Map<ffi::String, ffi::Array<TensorRegion>>{
            {"reads", accesses.reads}, {"writes", accesses.writes}};
      });
}

} // namespace tvm::tl
