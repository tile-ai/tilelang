#include "compat/visit.h"
#include "cuda/op/builtin.h"
#include "cuda/runtime.h"
#include "op/utils.h"

#include <tvm/backend/cuda/op/tensormap.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>

namespace tvm::tl {
namespace {

using namespace ffi;
using namespace tirx;

bool IsDescriptor(const CallNode *call) {
  return call->op.same_as(create_tma_descriptor()) ||
         call->op.same_as(create_tma_im2col_descriptor());
}

int64_t ConstantArgument(const Expr &value, const char *name) {
  auto result = GetConstInt(value);
  TVM_FFI_CHECK(result.has_value(), ValueError)
      << "TMA " << name << " must be a constant integer";
  return *result;
}

Call EncodeTiled(const Call &call, const Var &descriptor) {
  TVM_FFI_CHECK_GE(call->args.size(), 3U, ValueError)
      << "Malformed TMA descriptor";
  int64_t rank = ConstantArgument(call->args[1], "rank");
  TVM_FFI_CHECK(rank >= 1 && rank <= 5 &&
                    call->args.size() == static_cast<size_t>(4 * rank + 7),
                ValueError)
      << "TMA descriptor requires rank 1..5 and 4 * rank + 7 operands";
  const auto *pointer = call->args[2]->ty.as<PtrTypeNode>();
  TVM_FFI_CHECK(pointer && pointer->element_type.as<PrimTypeNode>(), TypeError)
      << "TMA descriptor requires a typed data pointer";
  auto attributes = make_object<tvm::backend::cuda::TensorMapEncodeTiledAttr>();
  PrimType descriptor_type = pointer->element_type.as_or_throw<PrimType>();
  if (IsTensorFloat32(descriptor_type))
    descriptor_type = PrimType::Float(32);
  attributes->descriptor_dtype = descriptor_type->dtype;
  attributes->rank = rank;
  int64_t descriptor_encoding = ConstantArgument(call->args[0], "dtype");
  attributes->force_cu_dtype =
      !descriptor_type.MatchesCode(kDLFloat4_e2m1fn) &&
              descriptor_encoding == to_CUtensorMapDataType(descriptor_type)
          ? -1
          : descriptor_encoding;
  attributes->interleave =
      ConstantArgument(call->args[3 + 4 * rank], "interleave");
  attributes->swizzle = ConstantArgument(call->args[4 + 4 * rank], "swizzle");
  attributes->l2_promotion =
      ConstantArgument(call->args[5 + 4 * rank], "l2 promotion");
  attributes->oob_fill =
      ConstantArgument(call->args[6 + 4 * rank], "out-of-bounds fill");
  Array<Expr> arguments{descriptor, call->args[2]};
  for (int64_t axis = 0; axis < rank; ++axis) {
    arguments.push_back(call->args[3 + axis]);
  }
  for (int64_t axis = 1; axis < rank; ++axis) {
    arguments.push_back(call->args[3 + rank + axis]);
  }
  for (int64_t axis = 0; axis < 2 * rank; ++axis) {
    arguments.push_back(call->args[3 + 2 * rank + axis]);
  }
  return Call(PrimType::Int(32),
              tvm::backend::cuda::tensormap_encode_tiled_op(), arguments,
              Attrs(attributes), {}, call->loc);
}

class DeviceDescriptorLowerer : public tirx::StmtExprMutator {
public:
  DeviceDescriptorLowerer(const Array<Var> &host_variables,
                          bool disable_shuffle_elect)
      : disable_shuffle_elect_(disable_shuffle_elect) {
    for (const Var &variable : host_variables)
      host_variables_.insert(variable.get());
  }

  RegionStmt Lower(const RegionStmt &region) {
    SeqStmt body = Mutate(region->body)
                       .ValueOrUnchanged(region->body)
                       .as_or_throw<SeqStmt>();
    if (!prefetch_.empty()) {
      PrimExpr condition =
          Call(PrimType::Bool(), tl_shuffle_elect(), {PrimExpr(0)})
              .as_or_throw<PrimExpr>();
      if (disable_shuffle_elect_) {
        PrimExpr thread =
            Call(PrimType::Int(32), Op::Get("tirx.cuda.thread_idx"),
                 {StringImm("x")})
                .as_or_throw<PrimExpr>();
        condition = thread == 0;
      }
      body = SeqStmt({If(condition, SeqStmt(prefetch_)), body});
    }
    RegionStmt result = region;
    result.CopyOnWrite()->body = body;
    return result;
  }

  const Array<Stmt> &initializers() const { return initializers_; }

private:
  UnchangedOr<Stmt> Mutate_(const BindNode *node,
                            InplaceMode inplace_mode) final {
    const auto *call = node->value.as<CallNode>();
    auto tensor = node->var.as<TensorVar>();
    if (call && call->op.same_as(decl_tensor_op()) && tensor &&
        IsGlobalBuffer(*tensor)) {
      bool host_evaluable = true;
      for (const Var &variable : UndefinedVars(node->value)) {
        host_evaluable &= host_variables_.count(variable.get()) != 0;
      }
      PostOrderVisit(node->value, [&](const ObjectRef &object) {
        if (object.as<TensorLoadNode>())
          host_evaluable = false;
        if (const auto *operation = object.as<CallNode>()) {
          host_evaluable &= operation->op.same_as(decl_tensor_op()) ||
                            operation->op.same_as(tensor_data_ptr_op()) ||
                            operation->op.same_as(ptr_byte_offset_op()) ||
                            operation->op.same_as(reinterpret_op());
        }
      });
      if (host_evaluable) {
        initializers_.push_back(GetRef<Stmt>(node));
        host_variables_.insert(node->var.get());
        return SeqStmt({});
      }
    }
    return StmtExprMutator::Mutate_(node, inplace_mode);
  }

  UnchangedOr<Expr> Mutate_(const CallNode *node,
                            InplaceMode inplace_mode) final {
    Expr original = GetRef<Expr>(node);
    Expr visited =
        StmtExprMutator::Mutate_(node, inplace_mode).ValueOrUnchanged(original);
    Call call = visited.as_or_throw<Call>();
    if (!IsDescriptor(call.get()))
      return visited;
    for (const Var &variable : UndefinedVars(call)) {
      TVM_FFI_CHECK(host_variables_.count(variable.get()), ValueError)
          << "TMA descriptor depends on device-bound variable "
          << variable->name;
    }
    if (auto found = descriptors_.find(call); found != descriptors_.end())
      return found->second;
    Var descriptor("tma_descriptor_" + std::to_string(descriptors_.size()),
                   PtrType(TensorMapType()));
    descriptors_.emplace(call, descriptor);
    initializers_.push_back(
        Bind(descriptor, Call(descriptor->ty, stack_alloca_op(),
                              {StringImm("tensormap"), PrimExpr(1)})));
    if (call->op.same_as(create_tma_descriptor())) {
      initializers_.push_back(Evaluate(EncodeTiled(call, descriptor)));
    } else {
#if CUDA_MAJOR_VERSION >= 12
      Array<Expr> arguments{StringImm(tvm_tensormap_create_im2col), descriptor};
      for (const Expr &argument : call->args)
        arguments.push_back(argument);
      initializers_.push_back(
          Evaluate(Call(PrimType::Int(32), call_packed_op(), arguments)));
#else
      TVM_FFI_THROW(ValueError)
          << "TMA im2col descriptors require CUDA 12 or newer";
#endif
    }
    prefetch_.push_back(Evaluate(Call(
        TupleType(Array<Type>{}), prefetch_tma_descriptor(), {descriptor})));
    return descriptor;
  }

  bool disable_shuffle_elect_;
  std::unordered_set<const VarNode *> host_variables_;
  std::unordered_map<Call, Var, StructuralHash, StructuralEqual> descriptors_;
  Array<Stmt> initializers_;
  Array<Stmt> prefetch_;
};

class HopperLowerer : public tirx::StmtExprMutator {
public:
  explicit HopperLowerer(bool disable_shuffle_elect)
      : disable_shuffle_elect_(disable_shuffle_elect) {}

private:
  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                            InplaceMode inplace_mode) final {
    const auto &name = node->op->name;
    if (name != "tirx.device_entry" && name != "tirx.launch_thread") {
      return StmtExprMutator::Mutate_(node, inplace_mode);
    }
    RegionStmt region = GetRef<RegionStmt>(node);
    auto lowerer = make_object<DeviceDescriptorLowerer>(UndefinedVars(region),
                                                        disable_shuffle_elect_);
    RegionStmt lowered = lowerer->Lower(region);
    Array<Stmt> statements = lowerer->initializers();
    if (statements.empty())
      return lowered;
    statements.push_back(lowered);
    return SeqStmt(statements);
  }

  UnchangedOr<Expr> Mutate_(const CallNode *node,
                            InplaceMode inplace_mode) final {
    TVM_FFI_CHECK(!IsDescriptor(node), ValueError)
        << "TMA descriptor construction requires an enclosing device region";
    return StmtExprMutator::Mutate_(node, inplace_mode);
  }

  bool disable_shuffle_elect_;
};

tirx::Function LowerHopper(tirx::Function function,
                           bool disable_shuffle_elect) {
  if (!function->body.has_value())
    return function;
  auto lowerer = make_object<HopperLowerer>(disable_shuffle_elect);
  SeqStmt body = function->body.value();
  body = lowerer->Mutate(body).ValueOrUnchanged(body).as_or_throw<SeqStmt>();
  if (auto settings = function->GetAttr<Map<String, Array<PrimExpr>>>(
          "l2_persistent_map")) {
    std::unordered_map<std::string, TensorVar> tensors;
    for (const Var &parameter : function->params) {
      if (auto tensor = parameter.as<TensorVar>())
        tensors.emplace(parameter->name, tensor.value());
    }
    Array<Stmt> statements;
    for (const auto &[name, arguments] : settings.value()) {
      auto found = tensors.find(name);
      TVM_FFI_CHECK(found != tensors.end() && arguments.size() >= 2, ValueError)
          << "Invalid L2 persistence configuration for tensor " << name;
      const TensorVar &tensor = found->second;
      Array<PrimExpr> origin(tensor.shape().size(), PrimExpr(0));
      Expr pointer = Call(tensor.ty()->DataPtrType(), address_of_op(),
                          {TensorLoad(tensor, origin)});
      statements.push_back(
          Evaluate(Call(PrimType::Int(32), call_packed_op(),
                        {StringImm(tvm_cuda_stream_set_access_policy_window),
                         pointer, arguments[1], arguments[0]})));
    }
    if (!statements.empty()) {
      statements.push_back(body);
      statements.push_back(Evaluate(
          Call(PrimType::Int(32), call_packed_op(),
               {StringImm(tvm_cuda_stream_reset_access_policy_window)})));
      body = SeqStmt(statements);
    }
  }
  function.CopyOnWrite()->body = body;
  return function;
}

} // namespace

tvm::transform::Pass LowerHopperIntrin() {
  auto pass =
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext context) -> ffi::Optional<tirx::Function> {
    return LowerHopper(
        std::move(function),
        context->GetConfig<bool>(kDisableShuffleElect).value_or(false));
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LowerHopperIntrin");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.LowerHopperIntrin",
                                   LowerHopperIntrin);
}

} // namespace tvm::tl
