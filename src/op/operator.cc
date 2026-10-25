/*!
 * \file tl/op/op.cc
 *
 * Define operators usd in tile library.
 */

#include "operator.h"
#include "builtin.h"
#include "support/check.h"

#include <tvm/tirx/op/memory.h>

namespace tvm::tl {

TileOperator ParseOperator(const Call &call,
                           const BlockAnnotations &block_annotations) {
  auto operation = call->op.as<Op>();
  if (!operation || !Op::HasAttrMap("TLOpBuilder")) {
    return {};
  }
  auto builders = Op::GetAttrMap<OpBuilderFunc>("TLOpBuilder");
  if (!builders.count(operation.value())) {
    return {};
  }
  BlockAnnotations annotations;
  if (call->attrs.has_value()) {
    annotations = call->attrs.value().as_or_throw<DictAttrs>()->dict;
  }
  TileOperator result = builders[operation.value()](call->args, annotations);
  ICHECK(result.defined());
  if (Op::HasAttrMap(kTLOpBlockAnnotationHandler)) {
    auto handlers = Op::GetAttrMap<OpBlockAnnotationHandlerFunc>(
        kTLOpBlockAnnotationHandler);
    if (handlers.count(operation.value())) {
      result = handlers[operation.value()](result, block_annotations);
      ICHECK(result.defined());
    }
  }
  return result;
}

TileOperator ParseOperator(const Stmt &stmt,
                           const BlockAnnotations &block_annotations) {
  if (const auto *evaluate = stmt.as<EvaluateNode>()) {
    if (auto call = evaluate->value.as<Call>()) {
      return ParseOperator(call.value(), block_annotations);
    }
  }
  return {};
}

Var GetVarFromAccessPtr(const Expr &expression) {
  if (auto variable = expression.as<Var>()) {
    TVM_FFI_CHECK(variable.value()->ty.as<PtrTypeNode>() ||
                      variable.value()->ty.as<tirx::TensorTypeNode>(),
                  TypeError)
        << "Expected a tensor or typed pointer variable";
    return variable.value();
  }
  const auto *call = expression.as<CallNode>();
  TVM_FFI_CHECK(call, TypeError)
      << "Expected a tensor pointer expression, got " << expression;
  if (call->op.same_as(tirx::tensor_data_ptr_op())) {
    ICHECK_EQ(call->args.size(), 1U);
    return call->args[0].as_or_throw<tirx::TensorVar>();
  }
  if (call->op.same_as(tirx::address_of_op()) ||
      call->op.same_as(access_ptr())) {
    TVM_FFI_CHECK(!call->args.empty(), ValueError)
        << "A tensor pointer needs an operand";
    if (const auto *load = call->args[0].as<TensorLoadNode>()) {
      return load->source.as_or_throw<tirx::TensorVar>();
    }
    if (const auto *region = call->args[0].as<TensorRegionNode>()) {
      return region->source.as_or_throw<tirx::TensorVar>();
    }
    return GetVarFromAccessPtr(call->args[0]);
  }
  if (call->op.same_as(tirx::reinterpret_op())) {
    ICHECK_EQ(call->args.size(), 1U);
    return GetVarFromAccessPtr(call->args[0]);
  }
  if (call->op.same_as(tirx::ptr_byte_offset_op()) ||
      call->op.same_as(tirx::handle_add_byte_offset_op())) {
    ICHECK_EQ(call->args.size(), 2U);
    return GetVarFromAccessPtr(call->args[0]);
  }
  TVM_FFI_THROW(TypeError) << "Cannot identify the tensor behind pointer "
                           << expression;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::ObjectDef<TileOperatorNode>();
  ffi::reflection::GlobalDef()
      .def("tl.GetVarFromAccessPtr", GetVarFromAccessPtr)
      .def("tl.ParseOperator",
           [](ffi::ObjectRef object, BlockAnnotations annotations) {
             if (auto call = object.as<Call>()) {
               return ParseOperator(call.value(), annotations);
             }
             return ParseOperator(object.as_or_throw<Stmt>(), annotations);
           });
}

} // namespace tvm::tl
