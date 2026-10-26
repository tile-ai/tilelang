#include "compat/metadata.h"
#include "op/prim.h"
#include <tvm/backend/cuda/op/tensormap.h>
#include <tvm/backend/opencl/op/memory.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

#include "transform/tirx_passes.h"
#include <tvm/runtime/device_api.h>

namespace tvm::tl::tirx_passes {
using namespace tvm::tirx;
using tvm::tirx::transform::CreateFunctionPass;
using tvm::transform::Pass;
using tvm::transform::PassContext;
namespace {
PrimType PackedScalarType(const PrimType &type) {
  TVM_FFI_ICHECK(!type.IsVoid()) << "Cannot pass void type through packed API";
  TVM_FFI_ICHECK_EQ(type.lanes(), 1)
      << "Cannot pass vector type through packed API";
  if (type.MatchesCode(kDLBool, kDLUInt, kDLInt))
    return PrimType::Int(64);
  TVM_FFI_ICHECK(type.MatchesCode(kDLFloat, kDLBfloat));
  return PrimType::Float(64);
}

using namespace tvm::prim;

namespace {

TVM_FFI_INLINE int GetVectorBytes(const PrimType &dtype) {
  return static_cast<int>(dtype.StorageBytes());
}

} // namespace

class BuiltinLower : public tvm::tl::StmtExprMutator {
public:
  using tvm::tl::StmtExprMutator::Mutate;
  using tvm::tl::StmtExprMutator::Mutate_;
  static Function Build(Function func) {
    if (!func->body.has_value())
      return func;
    ffi::Optional<PrimExpr> device_type = std::nullopt;
    bool preserve_ffi_kernel = false;
    if (auto target = func->GetAttr(tvm::attr::kTarget)) {
      device_type = IntImm::Int32(target.value()->kind->default_device_type);
      auto host = target.value()->GetHost().value_or(target.value());
      preserve_ffi_kernel = host->kind->name == "cuda_host";
    }

    auto mutator =
        ffi::make_object<BuiltinLower>(device_type, preserve_ffi_kernel);
    func.CopyOnWrite()->body =
        mutator->VisitBodyAndRealizeAlloca(func->body.value());
    return func;
  }

  explicit BuiltinLower(ffi::Optional<PrimExpr> device_type = std::nullopt,
                        bool preserve_ffi_kernel = false)
      : device_type_(device_type), preserve_ffi_kernel_(preserve_ffi_kernel) {
    if (const auto *type = device_type_.as<IntImmNode>();
        type && type->value == kDLCPU) {
      device_id_ = IntImm::Int32(0);
    }
  }

  struct StackSizes {

    int64_t shape_stack{-1};
    uint64_t array_stack{0};
    uint64_t arg_stack{0};
  };

  struct AllocaScope {
    ffi::Optional<TensorVar> stack_shape;
    Var stack_array = Var("stack_array", PtrType::VoidPtrType());
    Var stack_ffi_any = Var("stack_ffi_any", PtrType::VoidPtrType());

    StackSizes max_sizes;
    StackSizes run_sizes;

    void UpdateMax() {
      max_sizes.shape_stack =
          std::max(max_sizes.shape_stack, run_sizes.shape_stack);
      max_sizes.array_stack =
          std::max(max_sizes.array_stack, run_sizes.array_stack);
      max_sizes.arg_stack = std::max(max_sizes.arg_stack, run_sizes.arg_stack);
    }

    void AssertMaxIsValid() const {
      TVM_FFI_ICHECK((max_sizes.shape_stack >= run_sizes.shape_stack) ||
                     (max_sizes.array_stack >= run_sizes.array_stack) ||
                     (max_sizes.arg_stack >= run_sizes.arg_stack));
    }
  };

  Stmt Build(Stmt stmt) { return this->VisitBodyAndRealizeAlloca(stmt); }

  StackSizes GetMaxStack(Stmt stmt) {
    auto precheck = ffi::make_object<BuiltinLower>();
    precheck->is_precheck_ = true;
    precheck->device_id_ = this->device_id_;
    precheck->device_type_ = this->device_type_;

    precheck->alloca_scope_.emplace_back();
    {

      auto &scope = precheck->alloca_scope_.back();
      scope.stack_shape = TensorVar(
          "stack_shape", TensorType({IntImm::Int64(0)}, PrimType::Int(64)));
    }

    precheck->Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(stmt);

    TVM_FFI_ICHECK_EQ(precheck->alloca_scope_.size(), 1);
    return precheck->alloca_scope_[0].max_sizes;
  }

  Stmt VisitBodyAndRealizeAlloca(Stmt stmt) {

    if (is_precheck_) {
      return stmt;
    }

    alloca_scope_.emplace_back();
    {

      auto &scope = alloca_scope_.back();

      scope.max_sizes = GetMaxStack(stmt);

      ffi::Array<Stmt> alloca_stmts;
      if (scope.max_sizes.arg_stack != 0) {
        alloca_stmts.push_back(
            Bind(scope.stack_ffi_any,
                 Call(scope.stack_ffi_any->ty, tirx::stack_alloca_op(),
                      {StringImm("tvm_ffi_any"),
                       IntImm::Int32(scope.max_sizes.arg_stack)})));
      }

      if (scope.max_sizes.array_stack != 0) {
        alloca_stmts.push_back(
            Bind(scope.stack_array,
                 Call(scope.stack_array->ty, tirx::stack_alloca_op(),
                      {StringImm("array"),
                       IntImm::Int32(scope.max_sizes.array_stack)})));
      }

      if (scope.max_sizes.shape_stack != -1) {
        scope.stack_shape =
            TensorVar("stack_shape",
                      TensorType({IntImm::Int64(scope.max_sizes.shape_stack)},
                                 PrimType::Int(64)));
        stmt = SeqStmt(
            {Bind(scope.stack_shape.value(),
                  Call(scope.stack_shape.value().ty(), decl_tensor_op(),
                       {Call(scope.stack_shape.value().ty()->DataPtrType(),
                             tirx::stack_alloca_op(),
                             {StringImm("shape"),
                              IntImm::Int32(scope.max_sizes.shape_stack)})},
                       {}, {scope.stack_shape.value().ty()})),
             stmt});
      }

      if (!alloca_stmts.empty()) {
        alloca_stmts.push_back(stmt);
        stmt = SeqStmt(alloca_stmts);
      }
    }

    stmt = scope_.WithNewScope([&]() -> Stmt {
      Stmt visited =
          this->Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(stmt);
      return AppendPendingFrees(visited);
    });

    TVM_FFI_ICHECK(!alloca_scope_.empty());
    alloca_scope_.pop_back();

    return stmt;
  }

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView input,
                               InplaceMode inplace_mode) final {
    if (!input.as<StmtNode>())
      return tvm::tl::StmtExprMutator::Mutate(input, inplace_mode);

    prep_seq_stack_.emplace_back(std::vector<Stmt>());

    auto scope_size = alloca_scope_.size();
    auto mutated = tvm::tl::StmtExprMutator::Mutate(input, inplace_mode);
    Stmt stmt = std::move(mutated).ValueOrUnchanged(input).as_or_throw<Stmt>();
    {

      auto &scope = alloca_scope_.back();

      TVM_FFI_ICHECK_EQ(alloca_scope_.size(), scope_size)
          << "alloca_scope_ length is different before and after recursion";
      TVM_FFI_ICHECK_EQ(scope.run_sizes.shape_stack, -1)
          << "Expect no stack_make_shape outside of CallNodes";
      TVM_FFI_ICHECK_EQ(scope.run_sizes.array_stack, 0)
          << "Expect no stack_make_dltensor outside of CallNodes";
    }

    auto prep_seq = std::move(prep_seq_stack_.back());
    prep_seq_stack_.pop_back();

    if (prep_seq.size() != 0) {
      prep_seq.push_back(stmt);
      return SeqStmt(prep_seq);
    } else {
      return stmt;
    }
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *op,
                            InplaceMode inplace_mode) final {
    if (const auto *call = op->value.as<CallNode>();
        call && call->op.same_as(alloc_tensor_op()))
      return MutateAllocTensor(op, inplace_mode);
    if (const CallNode *call = op->value.as<CallNode>()) {
      if (call->op.same_as(backend::opencl::nd_mem_alloc_with_scope_op())) {
        return MakeNdMemAllocWithScope(op, call);
      }
    }
    return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> MutateAllocTensor(const BindNode *op,
                                      InplaceMode inplace_mode) {

    Stmt stmt = tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = stmt.as<BindNode>();
    const auto *tensor_call = op->value.as<CallNode>();
    tvm::Tuple shape =
        tvm::Tuple(tensor_call->ty_args[0].as_or_throw<TensorType>()->shape);
    DLDataType dtype =
        tensor_call->ty_args[0].as_or_throw<TensorType>()->dtype->dtype;
    PrimType element_type(dtype);
    auto scope = tensor_call->ty_args[0].as_or_throw<TensorType>()->scope;
    if (element_type.IsScalableVector()) {
      return stmt;
    }
    int64_t nbytes = GetVectorBytes(element_type);
    if (const auto *dev_type = device_type_.as<IntImmNode>();
        dev_type && dev_type->value == kDLCPU) {
      if (scope == runtime::MemoryScope::Global() || scope.name() == "local") {
        auto constant_size =
            op->var->ty.as_or_throw<TensorType>()->ConstantAllocationSize();
        if (constant_size.has_value() && constant_size.value() > 0 &&
            static_cast<size_t>(constant_size.value()) * nbytes <
                runtime::kMaxStackAlloca) {
          return stmt;
        }
      }
    }
    PrimExpr total_bytes = IntImm(PrimType::UInt(64), nbytes);
    for (const Expr &extent : shape->fields) {
      total_bytes = total_bytes * extent.as_or_throw<PrimExpr>();
    }
    TVM_FFI_ICHECK(device_type_) << "Unknown device type in current IR";
    TVM_FFI_ICHECK(device_id_) << "Unknown device id in current IR";
    Stmt throw_last_error =
        Evaluate(Call(PrimType::Int(32), throw_last_error_op(), {})
                     .as_or_throw<PrimExpr>());

    Stmt alloc_nullptr_check =
        If(Call(PrimType::Bool(), isnullptr_op(),
                {op->var.as_or_throw<TensorVar>().data()})
               .as_or_throw<PrimExpr>(),
           throw_last_error);

    static const Op free_workspace_op = Op::Get("tirx.free_workspace");
    static const Op alloc_workspace_op = Op::Get("tirx.alloc_workspace");
    PrimExpr free_op =
        Call(PrimType::Int(32), free_workspace_op,
             {tl::CastValue(device_type_.value(), PrimType::Int(32)),
              tl::CastValue(device_id_.value(), PrimType::Int(32)),
              op->var.as_or_throw<TensorVar>().data()})
            .as_or_throw<PrimExpr>();
    Stmt free_stmt = If(free_op != IntImm::Int32(0), throw_last_error);

    scope_.Current().pending_frees.push_back(free_stmt);

    Stmt alloc_bind = Bind(
        op->var.as_or_throw<TensorVar>(),
        Call(op->var.as_or_throw<TensorVar>().ty(), decl_tensor_op(),
             {Call(op->var.as_or_throw<TensorVar>().ty()->DataPtrType(),
                   alloc_workspace_op,
                   {tl::CastValue(device_type_.value(), PrimType::Int(32)),
                    tl::CastValue(device_id_.value(), PrimType::Int(32)),
                    total_bytes, IntImm::Int32(element_type.code()),
                    IntImm::Int32(element_type.bits())})},
             {}, {op->var.as_or_throw<TensorVar>().ty()}, tensor_call->loc),
        op->loc);

    return SeqStmt({alloc_bind, alloc_nullptr_check});
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *op,
                            InplaceMode inplace_mode) final {
    if (op->op.same_as(device_context_op())) {
      PrimExpr device_type = this->Mutate(op->args[0], inplace_mode)
                                 .ValueOrUnchanged(op->args[0])
                                 .as_or_throw<PrimExpr>();
      PrimExpr device_id = this->Mutate(op->args[1], inplace_mode)
                               .ValueOrUnchanged(op->args[1])
                               .as_or_throw<PrimExpr>();
      auto saved_type = device_type_;
      auto saved_id = device_id_;
      device_type_ = device_type;
      device_id_ = device_id;
      Stmt body = scope_.WithNewScope([&]() -> Stmt {
        return AppendPendingFrees(
            this->Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body));
      });
      device_type_ = saved_type;
      device_id_ = saved_id;
      return body;
    }
    return scope_.WithNewScope([&]() -> Stmt {
      RegionStmt region = tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode)
                              .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                              .as_or_throw<RegionStmt>();
      if (!scope_.Current().pending_frees.empty()) {
        region.CopyOnWrite()->body = AppendPendingFrees(region->body);
      }
      return region;
    });
  }

  UnchangedOr<Stmt> Mutate_(const ForNode *op, InplaceMode inplace_mode) final {
    auto lower_control = [&](const PrimExpr &control, const char *name) {
      PrimExpr value = Mutate(control, inplace_mode).ValueOrUnchanged(control);
      if (SideEffect(value) != CallEffectKind::kPure) {
        PrimVar temporary(name, value.ty());
        prep_seq_stack_.back().push_back(Bind(temporary, value));
        return PrimExpr(temporary);
      }
      return value;
    };
    PrimExpr begin = lower_control(op->begin, "loop_begin");
    PrimExpr end = lower_control(op->end, "loop_end");
    PrimExpr step = lower_control(op->step, "loop_step");
    Stmt body = op->body;

    if (op->kind == ForKind::kParallel) {
      body = this->VisitBodyAndRealizeAlloca(op->body);
    } else {
      body = scope_.WithNewScope([&]() -> Stmt {
        Stmt visited =
            this->Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
        return AppendPendingFrees(visited);
      });
    }

    if (begin.same_as(op->begin) && end.same_as(op->end) &&
        step.same_as(op->step) && body.same_as(op->body)) {
      return ffi::Unchanged();
    } else {
      if (inplace_mode == InplaceMode::kAllow) {
        auto *n = const_cast<ForNode *>(op);
        n->begin = std::move(begin);
        n->end = std::move(end);
        n->step = step;
        n->body = std::move(body);
        return ffi::Unchanged();
      }
      auto n = ffi::make_object<ForNode>(*op);
      n->begin = std::move(begin);
      n->end = std::move(end);
      n->step = step;
      n->body = std::move(body);
      return Stmt(n);
    }
  }

  UnchangedOr<Stmt> Mutate_(const IfNode *op, InplaceMode inplace_mode) final {
    auto condition_result = this->Mutate(op->condition, inplace_mode);
    bool condition_unchanged =
        condition_result.UnchangedOrSameAs(op->condition);
    PrimExpr condition =
        std::move(condition_result).ValueOrUnchanged(op->condition);

    Stmt then_case = scope_.WithNewScope([&]() -> Stmt {
      Stmt visited = this->Mutate(op->then_case, inplace_mode)
                         .ValueOrUnchanged(op->then_case);
      return AppendPendingFrees(visited);
    });
    ffi::Optional<SeqStmt> else_case;
    if (op->else_case) {
      else_case = scope_.WithNewScope([&]() -> SeqStmt {
        Stmt visited = this->Mutate(op->else_case.value(), inplace_mode)
                           .ValueOrUnchanged(op->else_case.value());
        return AppendPendingFrees(visited);
      });
    }
    if (condition_unchanged && then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      return ffi::Unchanged();
    }
    return If(condition, then_case, else_case, op->loc);
  }

  UnchangedOr<Expr> Mutate_(const CallNode *op,
                            InplaceMode inplace_mode) final {
    if (op->op.same_as(backend::cuda::tensormap_encode_tiled_op()) &&
        !preserve_ffi_kernel_) {
      const auto *attr =
          (op->attrs.has_value()
               ? op->attrs.value().as<backend::cuda::TensorMapEncodeTiledAttr>()
               : nullptr);
      TVM_FFI_CHECK(attr && attr->rank >= 1 && attr->rank <= 5 &&
                        op->args.size() ==
                            static_cast<size_t>(4 * attr->rank + 1),
                    ValueError)
          << "Invalid tensormap_encode_tiled attributes or operands";
      ffi::Array<Expr> args{
          StringImm("runtime.cuTensorMapEncodeTiled"), op->args[0],
          StringImm(ffi::DLDataTypeToString(attr->descriptor_dtype)),
          IntImm(PrimType::Int(32), attr->rank), op->args[1]};
      for (size_t i = 2; i < op->args.size(); ++i)
        args.push_back(op->args[i]);
      for (int64_t value : {attr->interleave, attr->swizzle, attr->l2_promotion,
                            attr->oob_fill}) {
        args.push_back(IntImm(PrimType::Int(32), value));
      }
      if (attr->force_cu_dtype != -1) {
        args.push_back(IntImm(PrimType::Int(32), attr->force_cu_dtype));
      }
      Call packed(op->ty, call_packed_op(), args);
      return MakeCallPackedGeneric(packed.get(), 0, call_packed_lowered_op());
    }
    if (op->op.same_as(call_packed_op()) ||
        (op->op.same_as(call_ffi_kernel_op()) && !preserve_ffi_kernel_)) {
      return MakeCallPackedGeneric(op, 0, call_packed_lowered_op());
    } else if (op->op.same_as(call_cpacked_op())) {
      return MakeCallPackedGeneric(op, 0, call_cpacked_lowered_op());
    } else if (op->op.same_as(stack_make_shape_op())) {
      return MakeShape(op);
    } else if (op->op.same_as(stack_make_dltensor_op())) {
      return MakeArray(op);
    } else {
      return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode);
    }
  }

  StringImm GetDeviceMethodName(const char *method_name) const {
    TVM_FFI_ICHECK(device_type_)
        << "Method " << method_name << " requires the device type, "
        << "but occurred outside of a \"device_context\" region";

    auto as_int = device_type_.as<IntImmNode>();
    TVM_FFI_ICHECK(as_int)
        << "Method " << method_name
        << " requires the device type to be a DLDeviceType enum value, "
        << "but was instead the expression " << device_type_ << " with type "
        << device_type_.value()->GetTypeKey();

    ffi::String device_name =
        runtime::DLDeviceType2Str(as_int->value.as<int>().value());
    return StringImm("device_api." + device_name + "." + method_name);
  }

  Expr MakeShape(const CallNode *op) {

    TVM_FFI_ICHECK(!alloca_scope_.empty());
    auto &scope = alloca_scope_.back();
    auto &prep_seq = prep_seq_stack_.back();
    if (scope.run_sizes.shape_stack == -1) {
      scope.run_sizes.shape_stack = 0;
    }
    int64_t stack_begin = scope.run_sizes.shape_stack;
    scope.run_sizes.shape_stack += op->args.size();
    Expr expr = tvm::tl::StmtExprMutator::Mutate_(op, InplaceMode::kDisallow)
                    .ValueOrUnchanged(ffi::GetRef<Expr>(op));
    op = expr.as<CallNode>();

    for (size_t i = 0; i < op->args.size(); ++i) {
      prep_seq.emplace_back(TensorStore(
          scope.stack_shape.value(), {IntImm::Int32(stack_begin + i)},
          tl::CastValue(op->args[i].as_or_throw<PrimExpr>(),
                        PrimType::Int(64))));
    }
    PrimExpr offset = IntImm::Int32(stack_begin);
    TensorLoad load = TensorLoad(scope.stack_shape.value(), {offset});
    return Call(scope.stack_shape.value().ty()->DataPtrType(), address_of_op(),
                {load});
  }

  Expr MakeArray(const CallNode *op) {
    TVM_FFI_ICHECK(!alloca_scope_.empty());
    auto &scope = alloca_scope_.back();
    auto &prep_seq = prep_seq_stack_.back();

    size_t idx = scope.run_sizes.array_stack;
    scope.run_sizes.array_stack += 1;
    Expr expr = tvm::tl::StmtExprMutator::Mutate_(op, InplaceMode::kDisallow)
                    .ValueOrUnchanged(ffi::GetRef<Expr>(op));
    op = expr.as<CallNode>();

    prep_seq.emplace_back(
        Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                      {scope.stack_array, IntImm::Int32(idx),
                       IntImm::Int32(kDLTensorData), op->args[0]})
                     .as_or_throw<PrimExpr>()));
    prep_seq.emplace_back(
        Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                      {scope.stack_array, IntImm::Int32(idx),
                       IntImm::Int32(kDLTensorShape), op->args[1]})
                     .as_or_throw<PrimExpr>()));
    Expr strides = op->args[2];
    if (auto prim_strides = strides.as<PrimExpr>();
        prim_strides && IsZero(prim_strides.value())) {
      strides = tirx::ConstHandle(0);
    }
    prep_seq.emplace_back(
        Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                      {scope.stack_array, IntImm::Int32(idx),
                       IntImm::Int32(kDLTensorStrides), strides})
                     .as_or_throw<PrimExpr>()));
    prep_seq.emplace_back(
        Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                      {scope.stack_array, IntImm::Int32(idx),
                       IntImm::Int32(kDLTensorNDim),
                       op->args[3].as_or_throw<PrimExpr>()})
                     .as_or_throw<PrimExpr>()));
    PrimType dtype = op->args[4].as_or_throw<PrimExpr>().ty();
    prep_seq.emplace_back(Evaluate(
        Call(PrimType::Int(32), tirx::abi_field_set_op(),
             {scope.stack_array, IntImm::Int32(idx),
              IntImm::Int32(kDLTensorTypeCode),
              IntImm(PrimType::UInt(8), static_cast<int>(dtype.code()))})
            .as_or_throw<PrimExpr>()));
    prep_seq.emplace_back(
        Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                      {scope.stack_array, IntImm::Int32(idx),
                       IntImm::Int32(kDLTensorTypeBits),
                       IntImm(PrimType::UInt(8), dtype.bits())})
                     .as_or_throw<PrimExpr>()));
    prep_seq.emplace_back(
        Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                      {scope.stack_array, IntImm::Int32(idx),
                       IntImm::Int32(kDLTensorTypeLanes),
                       IntImm(PrimType::UInt(16), dtype.lanes())})
                     .as_or_throw<PrimExpr>()));

    int data_bytes = GetVectorBytes(dtype);
    PrimExpr elem_offset = op->args[5].as_or_throw<PrimExpr>();
    PrimExpr byte_offset = elem_offset;
    if (!IsZero(elem_offset)) {
      byte_offset = elem_offset * IntImm(elem_offset.ty(), data_bytes);
    } else {
      byte_offset = elem_offset;
    }
    prep_seq.emplace_back(
        Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                      {scope.stack_array, IntImm::Int32(idx),
                       IntImm::Int32(kDLTensorByteOffset),
                       tl::CastValue(byte_offset, PrimType::UInt(64))})
                     .as_or_throw<PrimExpr>()));
    TVM_FFI_ICHECK(device_type_) << "Unknown device type in current IR";
    TVM_FFI_ICHECK(device_id_) << "Unknown device id in current IR";
    prep_seq.emplace_back(
        Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                      {scope.stack_array, IntImm::Int32(idx),
                       IntImm::Int32(kDLTensorDeviceId),
                       tl::CastValue(device_id_.value(), PrimType::Int(32))})
                     .as_or_throw<PrimExpr>()));
    prep_seq.emplace_back(
        Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                      {scope.stack_array, IntImm::Int32(idx),
                       IntImm::Int32(kDLTensorDeviceType),
                       tl::CastValue(device_type_.value(), PrimType::Int(32))})
                     .as_or_throw<PrimExpr>()));
    return Call(
        PtrType::VoidPtrType(), tirx::abi_field_get_op(),
        {scope.stack_array, IntImm::Int32(idx), IntImm::Int32(kDLTensorAddr)});
  }

  void SetPackedArg(Expr arg, const Var &args_stack, size_t stack_offset,
                    std::vector<Stmt> *prep_seq) {
    int arg_type_index;
    if (arg.as<StringImmNode>()) {
      arg_type_index = ffi::TypeIndex::kTVMFFIRawStr;
      arg = tirx::reinterpret(PtrType::VoidPtrType(), std::move(arg));
    } else if (arg->ty.as<PtrTypeNode>()) {
      arg_type_index = IsArrayHandle(arg) ? ffi::TypeIndex::kTVMFFIDLTensorPtr
                                          : ffi::TypeIndex::kTVMFFIOpaquePtr;
    } else {
      PrimExpr prim_arg = arg.as_or_throw<PrimExpr>();
      PrimType arg_ty = prim_arg.ty();
      PrimType api_ty = PackedScalarType(arg_ty);
      if (arg_ty != api_ty) {
        arg = prim::Cast(api_ty, prim_arg);
      }
      if (api_ty.MatchesCode(DLDataTypeCode::kDLBool)) {
        arg_type_index = ffi::TypeIndex::kTVMFFIBool;
      } else if (api_ty.MatchesCode(DLDataTypeCode::kDLInt,
                                    DLDataTypeCode::kDLUInt)) {
        arg_type_index = ffi::TypeIndex::kTVMFFIInt;
      } else if (api_ty.code() == DLDataTypeCode::kDLFloat) {
        arg_type_index = ffi::TypeIndex::kTVMFFIFloat;
      } else {
        TVM_FFI_THROW(InternalError) << "Unsupported type: " << api_ty;
      }
    }

    if (arg_type_index == ffi::TypeIndex::kTVMFFIOpaquePtr) {
      prep_seq->emplace_back(If(
          Call(PrimType::Bool(), isnullptr_op(), {arg}).as_or_throw<PrimExpr>(),
          Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                        {args_stack, IntImm::Int32(stack_offset),
                         IntImm::Int32(kTVMFFIAnyTypeIndex),
                         IntImm::Int32(ffi::TypeIndex::kTVMFFINone)})
                       .as_or_throw<PrimExpr>()),
          SeqStmt(
              Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                            {args_stack, IntImm::Int32(stack_offset),
                             IntImm::Int32(kTVMFFIAnyTypeIndex),
                             IntImm::Int32(ffi::TypeIndex::kTVMFFIOpaquePtr)})
                           .as_or_throw<PrimExpr>()))));
    } else {
      prep_seq->emplace_back(
          Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                        {args_stack, IntImm::Int32(stack_offset),
                         IntImm::Int32(kTVMFFIAnyTypeIndex),
                         IntImm::Int32(arg_type_index)})
                       .as_or_throw<PrimExpr>()));
    }

    prep_seq->emplace_back(
        Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                      {args_stack, IntImm::Int32(stack_offset),
                       IntImm::Int32(kTVMFFIAnyZeroPadding), IntImm::Int32(0)})
                     .as_or_throw<PrimExpr>()));

    prep_seq->emplace_back(
        Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                      {args_stack, IntImm::Int32(stack_offset),
                       IntImm::Int32(kTVMFFIAnyUnionValue), arg})
                     .as_or_throw<PrimExpr>()));
  }

  Call MakeCallPackedGeneric(const CallNode *op, size_t name_offset,
                             const Op &lowered_packed_op) {
    auto &scope = alloca_scope_.back();
    auto &prep_seq = prep_seq_stack_.back();

    int64_t restore_shape_stack = scope.run_sizes.shape_stack;
    size_t restore_array_stack = scope.run_sizes.array_stack;
    size_t arg_stack_begin = scope.run_sizes.arg_stack;

    size_t args_begin = name_offset + 1;
    size_t args_end = op->args.size();

    size_t num_args = args_end - args_begin;

    scope.run_sizes.arg_stack += num_args + 1;

    Expr expr = tvm::tl::StmtExprMutator::Mutate_(op, InplaceMode::kDisallow)
                    .ValueOrUnchanged(ffi::GetRef<Expr>(op));
    op = expr.as<CallNode>();

    for (size_t i = 0; i < num_args; ++i) {
      const Expr &arg = op->args[args_begin + i];
      TVM_FFI_CHECK(arg.as<PrimExpr>() || arg->ty.as<PtrTypeNode>() ||
                        arg.as<StringImmNode>(),
                    TypeError)
          << "Packed call argument must have a primitive, pointer, or string "
             "type, but got "
          << arg->ty;
      this->SetPackedArg(arg, scope.stack_ffi_any, arg_stack_begin + i,
                         &prep_seq);
    }

    prep_seq.emplace_back(
        Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                      {scope.stack_ffi_any, IntImm::Int32(num_args),
                       IntImm::Int32(kTVMFFIAnyTypeIndex),
                       IntImm::Int32(ffi::TypeIndex::kTVMFFINone)})
                     .as_or_throw<PrimExpr>()));
    prep_seq.emplace_back(
        Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                      {scope.stack_ffi_any, IntImm::Int32(num_args),
                       IntImm::Int32(kTVMFFIAnyZeroPadding), IntImm::Int32(0)})
                     .as_or_throw<PrimExpr>()));
    prep_seq.emplace_back(
        Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                      {scope.stack_ffi_any, IntImm::Int32(num_args),
                       IntImm::Int32(kTVMFFIAnyUnionValue), IntImm::Int64(0)})
                     .as_or_throw<PrimExpr>()));

    if (is_precheck_) {
      scope.UpdateMax();
    } else {
      scope.AssertMaxIsValid();
    }
    scope.run_sizes.shape_stack = restore_shape_stack;
    scope.run_sizes.array_stack = restore_array_stack;
    scope.run_sizes.arg_stack = arg_stack_begin;
    ffi::Array<Expr> packed_args = {op->args[name_offset], scope.stack_ffi_any,
                                    IntImm::Int32(arg_stack_begin),
                                    IntImm::Int32(arg_stack_begin + num_args)};
    return Call(op->ty, lowered_packed_op, packed_args);
  }

  Stmt MakeNdMemAllocWithScope(const BindNode *let, const CallNode *call) {
    TVM_FFI_ICHECK(device_type_) << "Unknown device type in current IR";
    TVM_FFI_ICHECK(device_id_) << "Unknown device id in current IR";
    Stmt throw_last_error =
        Evaluate(Call(PrimType::Int(32), throw_last_error_op(), {})
                     .as_or_throw<PrimExpr>());

    const auto *dtype_node =
        let->var->ty.as<PtrTypeNode>()->element_type.as<PrimTypeNode>();
    TVM_FFI_ICHECK(dtype_node);
    PrimType dtype = ffi::GetRef<PrimType>(dtype_node);

    ffi::Array<Expr> args = {
        GetDeviceMethodName("alloc_nd"),
        device_type_.value(),
        device_id_.value(),
        IntImm::Int32(dtype.code()),
        IntImm::Int32(dtype.bits()),
    };

    for (size_t i = 0; i < call->args.size(); ++i) {
      args.push_back(call->args[i]);
    }

    Call call_packed = Call(let->var->ty, call_packed_op(), args);
    Stmt null_check =
        If(Call(PrimType::Bool(), isnullptr_op(), ffi::Array<Expr>{let->var})
               .as_or_throw<PrimExpr>(),
           throw_last_error);

    Expr storage_scope = call->args[0];
    Call free_op = Call(PrimType::Int(32), call_packed_op(),
                        {GetDeviceMethodName("free_nd"), device_type_.value(),
                         device_id_.value(), storage_scope, let->var});
    Stmt free_stmt = If(free_op.as_or_throw<PrimExpr>() != IntImm::Int32(0),
                        throw_last_error);

    free_stmt = tvm::tl::StmtExprMutator::Mutate(ffi::AnyView(free_stmt),
                                                 InplaceMode::kDisallow)
                    .ValueOrUnchanged(free_stmt)
                    .as_or_throw<Stmt>();
    scope_.Current().pending_frees.push_back(free_stmt);

    Stmt input = SeqStmt({Bind(let->var, call_packed), null_check});
    return tvm::tl::StmtExprMutator::Mutate(ffi::AnyView(input),
                                            InplaceMode::kDisallow)
        .ValueOrUnchanged(input)
        .as_or_throw<Stmt>();
  }

private:
  bool IsArrayHandle(const Expr &arg) {

    if (const CallNode *buf = arg.as<CallNode>()) {
      if (buf->op.same_as(abi_field_get_op()) &&
          buf->args[2].as<IntImmNode>()->value == kDLTensorAddr) {
        return true;
      }
    }
    return false;
  }

  struct ScopeLevel {
    std::vector<Stmt> pending_frees;
  };

  Stmt AppendPendingFrees(Stmt body) {
    auto &frees = scope_.Current().pending_frees;
    if (frees.empty())
      return body;
    ffi::Array<Stmt> stmts;
    stmts.push_back(body);
    for (auto it = frees.rbegin(); it != frees.rend(); ++it) {
      stmts.push_back(*it);
    }
    frees.clear();
    return SeqStmt(stmts);
  }

  std::vector<std::vector<Stmt>> prep_seq_stack_;
  ffi::Optional<PrimExpr> device_type_{std::nullopt};
  ffi::Optional<PrimExpr> device_id_{std::nullopt};

  bool preserve_ffi_kernel_{false};
  bool is_precheck_{false};

  std::vector<AllocaScope> alloca_scope_;

  ScopeStack<ScopeLevel> scope_;
};

} // namespace

Pass LowerTVMBuiltin() {
  auto pass_func = [](Function func, IRModule m, PassContext ctx) {
    bool is_host = func->HasNonzeroAttr(tvm::tirx::attr::kIsHostFunc.name);
    if (!is_host) {
      if (auto target = func->GetAttr(tvm::attr::kTarget)) {
        is_host = target.value()->HasKey("cpu");
      }
    }
    if (is_host) {
      func = BuiltinLower::Build(func);
      VLOG(2) << "LowerTVMBuiltin: " << func;
    }
    return func;
  };
  return CreateFunctionPass(pass_func, 0, "tl.LowerTVMBuiltin");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl._tirx_passes.LowerTVMBuiltin",
                                   [] { return LowerTVMBuiltin(); });
}
} // namespace tvm::tl::tirx_passes
