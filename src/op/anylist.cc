/*!
 * \file tl/op/anylist.cc
 * \brief Keep boxed FFI slots at TileLang's packed-call boundary.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>

namespace tvm::tl {
namespace {

void CheckSlot(const ffi::Array<Expr> &args) {
  TVM_FFI_CHECK(args.size() >= 2, ValueError);
  TVM_FFI_CHECK(args[0]->ty.as<PtrTypeObj>(), TypeError)
      << "Any-list storage must be a pointer";
  auto index = args[1].as<IntExpr>();
  TVM_FFI_CHECK(index && index->ty().IsScalar(), TypeError)
      << "Any-list index must be a scalar integer";
}

void ValidateSlot(const CallObj *call) {
  CheckSlot(call->args);
  TVM_FFI_CHECK(!call->attrs && call->ty_args.empty(), ValueError);
  Type expected = call->op.same_as(Op::Get("tl.anylist_getitem"))
                      ? Type(PtrType::VoidPtrType())
                      : Type(PrimType::Int(32));
  TVM_FFI_CHECK(ffi::StructuralEqual()(expected, call->ty), TypeError)
      << "Any-list operation has an incorrect result type";
}

void ValidateSetter(const CallObj *call) {
  ValidateSlot(call);
  TVM_FFI_CHECK(call->args.size() >= 3, ValueError);
  TVM_FFI_CHECK(call->args[2].as<StringImmObj>(), TypeError)
      << "Any-list callback requires a literal function name";
}

void ValidateShape(const CallObj *call) {
  TVM_FFI_CHECK(ffi::StructuralEqual()(call->ty, PtrType::VoidPtrType()),
                TypeError);
  TVM_FFI_CHECK(!call->attrs && call->ty_args.empty(), ValueError);
  for (const Expr &dimension : call->args) {
    auto integer = dimension.as<IntExpr>();
    TVM_FFI_CHECK(integer && integer->ty().IsScalar(), TypeError);
  }
}

void ValidateDescriptor(const CallObj *call) {
  TVM_FFI_CHECK_EQ(call->args.size(), 8, ValueError);
  TVM_FFI_CHECK(ffi::StructuralEqual()(call->ty, PtrType::VoidPtrType()),
                TypeError);
  TVM_FFI_CHECK(!call->attrs && call->ty_args.empty(), ValueError);
  for (size_t i : {0U, 1U}) {
    TVM_FFI_CHECK(call->args[i]->ty.as<PtrTypeObj>(), TypeError);
  }
  auto strides_zero = call->args[2].as<IntImmObj>();
  TVM_FFI_CHECK(call->args[2]->ty.as<PtrTypeObj>() ||
                    (strides_zero && strides_zero->value == 0),
                TypeError);
  for (size_t i : {3U, 5U, 6U, 7U}) {
    auto integer = call->args[i].as<IntExpr>();
    TVM_FFI_CHECK(integer && integer->ty().IsScalar(), TypeError);
  }
  auto dtype = call->args[4]->ty.as<PrimType>();
  TVM_FFI_CHECK(dtype && !dtype->IsVoid(), TypeError);
}

Type InferInvoke(const CallObj *call) {
  TVM_FFI_CHECK_LE(call->ty_args.size(), 1, TypeError);
  Type result = call->ty_args.empty() ? call->ty : call->ty_args[0];
  TVM_FFI_CHECK(!result.as<MissingTypeObj>(), TypeError)
      << "Any-list invoke requires its explicit result type";
  return result;
}

void ValidateInvoke(const CallObj *call) {
  Type result = InferInvoke(call);
  TVM_FFI_CHECK(ffi::StructuralEqual()(result, call->ty), TypeError);
  TVM_FFI_CHECK(call->attrs && !call->args.empty(), ValueError);
  TVM_FFI_CHECK(call->args[0].as<StringImmObj>(), TypeError);
  auto attrs = call->attrs.value().as_or_throw<DictAttrs>();
  auto mode = attrs.GetAttr<ffi::String>("mode").value_or("");
  std::string pattern = attrs.GetAttr<ffi::String>("pattern").value_or("");
  TVM_FFI_CHECK(mode == "set" || mode == "value" || mode == "reset",
                ValueError);
  TVM_FFI_CHECK(attrs.GetAttr<bool>("cpacked").has_value(), ValueError);
  size_t count = mode == "value" ? 1 : 3;
  for (char character : pattern) {
    TVM_FFI_CHECK(character == 'v' || character == 'g', ValueError);
    count += character == 'g' ? 2 : 1;
  }
  TVM_FFI_CHECK_EQ(call->args.size(), count, ValueError);
  if (mode != "value") {
    CheckSlot({call->args[1], call->args[2]});
    TVM_FFI_CHECK(ffi::StructuralEqual()(result, PrimType::Int(32)), TypeError);
  }
  TVM_FFI_CHECK(mode != "reset" || pattern.empty(), ValueError);
  size_t cursor = mode == "value" ? 1 : 3;
  for (char character : pattern) {
    if (character == 'g') {
      CheckSlot({call->args[cursor], call->args[cursor + 1]});
      cursor += 2;
    } else {
      const Expr &value = call->args[cursor++];
      auto type = value->ty.as<PrimType>();
      TVM_FFI_CHECK(value.as<StringImmObj>() || value->ty.as<PtrTypeObj>() ||
                        (type && type->IsScalar() && !type->IsVoid()),
                    TypeError)
          << "Any-list packed arguments require a scalar, pointer, or string";
    }
  }
}

bool IsGetter(const Expr &expression) {
  auto call = expression.as<CallObj>();
  return call && call->op.same_as(Op::Get("tl.anylist_getitem"));
}

// Keep the invocation at TileLang's ABI boundary. The owning host emitter gives
// each invocation its own Any argument/return array and lexical descriptor
// storage. Ordinary packed calls, their descriptors, and workspace allocation
// retain native lowering. Nested invocations do not alias the shared Any stack.
class AnyListPreparer final : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *evaluate,
                            InplaceMode mode) final {
    Stmt result = StmtExprMutator::Mutate_(evaluate, mode)
                      .ValueOrUnchanged(ffi::GetRef<Stmt>(evaluate));
    const auto *current = result.as<EvaluateObj>();
    const auto *call = current->value.as<CallObj>();
    if (call && call->op.same_as(Op::Get("tl.anylist_invoke")) &&
        call->attrs.value()
                .as_or_throw<DictAttrs>()
                .GetAttr<ffi::String>("mode")
                .value_or("") == "value") {
      // Public call_packed historically has a nominal int32 result even when
      // a statement callback returns None. A discarded result needs no scalar
      // conversion, but still releases any owned callback result.
      return Evaluate(Call(PrimType::Void(), call->op, call->args, call->attrs,
                           {}, call->loc),
                      current->loc);
    }
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *region,
                            InplaceMode mode) final {
    auto previous = device_context_;
    if (region->op->name == "tirx.device_context") {
      TVM_FFI_CHECK_EQ(region->args.size(), 2, ValueError);
      device_context_ = region->args;
    }
    auto result = StmtExprMutator::Mutate_(region, mode);
    device_context_ = previous;
    return result;
  }

  UnchangedOr<Expr> Mutate_(const CallObj *call, InplaceMode mode) final {
    const auto *op = call->op.as<OpObj>();
    if (!op)
      return StmtExprMutator::Mutate_(call, mode);
    const std::string name = op->name;
    if (inside_argument_ &&
        (call->op.same_as(tirx::stack_make_shape_op()) ||
         call->op.same_as(tirx::stack_make_dltensor_op()))) {
      ffi::GetRef<Call>(call).Validate();
      ffi::Array<Expr> args;
      for (const Expr &argument : call->args)
        args.push_back(VisitOperand(argument));
      bool descriptor = call->op.same_as(tirx::stack_make_dltensor_op());
      if (descriptor) {
        TVM_FFI_CHECK_EQ(device_context_.size(), 2, ValueError)
            << "Any-list tensor descriptor requires a native device_context";
        for (const Expr &device : device_context_)
          args.push_back(device);
      }
      return Call(PtrType::VoidPtrType(),
                  Op::Get(descriptor ? "tl.anylist_tensor_descriptor"
                                     : "tl.anylist_shape"),
                  args, {}, {}, call->loc);
    }
    if (name == "tl.anylist_getitem") {
      TVM_FFI_THROW(ValueError)
          << "anylist_getitem is only valid as a direct packed-call argument";
    }
    if (name == "tl.anylist_invoke") {
      // This pass is idempotent before native builtin lowering.
      return StmtExprMutator::Mutate_(call, mode);
    }
    bool setter = name == "tl.anylist_setitem_call_packed" ||
                  name == "tl.anylist_setitem_call_cpacked";
    bool reset = name == "tl.anylist_resetitem";
    bool packed = call->op.same_as(tirx::call_packed_op()) ||
                  call->op.same_as(tirx::call_cpacked_op());
    bool getter_argument = false;
    if (packed) {
      for (size_t i = 1; i < call->args.size(); ++i) {
        getter_argument |= IsGetter(call->args[i]);
      }
    }
    if (!setter && !reset && !getter_argument) {
      // A nested ordinary packed call retains the shared descriptor scope.
      bool previous = inside_argument_;
      if (packed)
        inside_argument_ = false;
      auto result = StmtExprMutator::Mutate_(call, mode);
      inside_argument_ = previous;
      return result;
    }
    ffi::GetRef<Call>(call).Validate();
    bool previous = inside_argument_;
    inside_argument_ = true;
    ffi::Array<Expr> encoded;
    if (setter || reset) {
      encoded.push_back(VisitOperand(call->args[0]));
      encoded.push_back(VisitOperand(call->args[1]));
    }
    std::string pattern;
    if (!reset) {
      size_t begin = setter ? 3 : 1;
      for (size_t i = begin; i < call->args.size(); ++i) {
        const Expr &arg = call->args[i];
        if (IsGetter(arg)) {
          const auto *getter = arg.as<CallObj>();
          ffi::GetRef<Call>(getter).Validate();
          encoded.push_back(VisitOperand(getter->args[0]));
          encoded.push_back(VisitOperand(getter->args[1]));
          pattern.push_back('g');
        } else {
          encoded.push_back(VisitOperand(arg));
          pattern.push_back('v');
        }
      }
    }
    Expr function_name = reset ? Expr(StringImm("tl.anylist_reset"))
                               : call->args[setter ? 2 : 0];
    ffi::Array<Expr> packed_args{function_name};
    for (const auto &argument : encoded)
      packed_args.push_back(argument);
    bool cpacked = name == "tl.anylist_setitem_call_cpacked" ||
                   call->op.same_as(tirx::call_cpacked_op());
    DictAttrs attrs({{"mode", ffi::String(reset    ? "reset"
                                          : setter ? "set"
                                                   : "value")},
                     {"pattern", ffi::String(pattern)},
                     {"cpacked", cpacked}});
    inside_argument_ = previous;
    return Call(call->ty, Op::Get("tl.anylist_invoke"), packed_args, attrs, {},
                call->loc);
  }

private:
  bool inside_argument_ = false;
  ffi::Array<Expr> device_context_;
  Expr VisitOperand(const Expr &expression) {
    return Mutate(expression).ValueOrUnchanged(expression);
  }
};

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.anylist_shape")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateShape>())
      .signature(sig::var_args<IntExpr>("shape"))
      .set_attr(tvm::op_attr::kFixedReturnType, PtrType::VoidPtrType())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
  OpDef("tl.anylist_tensor_descriptor")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateDescriptor>())
      .signature(sig::var_args<Expr>("descriptor_fields"))
      .set_attr(tvm::op_attr::kFixedReturnType, PtrType::VoidPtrType())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
  for (const char *name : {"tl.anylist_getitem", "tl.anylist_resetitem"}) {
    bool getter = std::string(name) == "tl.anylist_getitem";
    OpDef(name)
        .set_validator(ffi::reflection::NativeFunctionView<void(
                           const CallObj *)>::FromNative<&ValidateSlot>())
        .signature(sig::arg("list_handle"), sig::arg<IntExpr>("index"))
        .set_attr(tvm::op_attr::kFixedReturnType,
                  getter ? Type(PtrType::VoidPtrType())
                         : Type(PrimType::Int(32)))
        .set_attr(tvm::op_attr::kCallEffectKind,
                  getter ? CallEffectKind::kReadState
                         : CallEffectKind::kOpaque);
  }
  for (const char *name :
       {"tl.anylist_setitem_call_packed", "tl.anylist_setitem_call_cpacked"}) {
    OpDef(name)
        .set_validator(ffi::reflection::NativeFunctionView<void(
                           const CallObj *)>::FromNative<&ValidateSetter>())
        .signature(sig::arg("list_handle"), sig::arg<IntExpr>("index"),
                   sig::arg<StringImm>("func_name"),
                   sig::var_args<Expr>("args"))
        .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Int(32))
        .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
  }
  OpDef("tl.anylist_invoke")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj *)>::FromNative<&ValidateInvoke>())
      .signature(sig::arg<StringImm>("func_name"), sig::var_args<Expr>("args"),
                 sig::var_ty_args<Type>("result_type"),
                 sig::call_attrs<DictAttrsObj>())
      .set_native_fn_attr<&InferInvoke>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
  ffi::reflection::GlobalDef().def("tl.transform.PrepareAnyList", []() {
    return tirx::transform::CreateFunctionPass(
        [](tirx::Function function, IRModule, tvm::transform::PassContext) {
          auto mutator = ffi::make_object<AnyListPreparer>();
          auto body =
              mutator->Mutate(function->body).ValueOrUnchanged(function->body);
          if (!body.same_as(function->body))
            function.CopyOnWrite()->body = body;
          return function;
        },
        0, "tl.PrepareAnyList");
  });
}

} // namespace tvm::tl
