/*!
 * \file tl/transform/lower_hopper_intrin.cc
 * \brief Materialize host TMA descriptors and CUDA L2 stream policy calls.
 */

#include "op/builtin.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "cuda/op/builtin.h"
#include "cuda/runtime.h"
#include "ir/analyzer.h"
#include "ir/annotations.h"
#include "ir/tensor_utils.h"

namespace tvm::tl {
using namespace tirx;
using namespace ffi;
using namespace prim;

#if (CUDA_MAJOR_VERSION >= 12)
namespace {

bool StartsDeviceRegion(const RegionStmtObj *region) {
  return region->op.same_as(launch_thread_op()) ||
         region->op->name == "tirx.device_scope";
}

// Taking an address does not read the TensorLoad used to spell the lvalue.
// Ordinary memory reads and stateful calls cannot be evaluated on the host.
class HostExpressionChecker : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;

  Optional<VisitInterrupt> Visit_(const TensorLoadObj *) final {
    pure = false;
    return VisitInterrupt();
  }

  Optional<VisitInterrupt> Visit_(const CallObj *call) final {
    static auto effects = Op::GetAttrMap(tvm::op_attr::kCallEffectKind);
    if (effects.get(call->op, CallEffectKind::kOpaque) >
        CallEffectKind::kPure) {
      pure = false;
      return VisitInterrupt();
    }
    bool address =
        call->op.same_as(address_of_op()) || call->op.same_as(tl::access_ptr());
    for (size_t i = 0; i < call->args.size(); ++i) {
      if (address && i == 0) {
        if (auto load = call->args[i].as<TensorLoadObj>()) {
          Visit(load->source);
          Visit(load->indices);
          if (!pure)
            return VisitInterrupt();
          continue;
        }
      }
      Visit(call->args[i]);
      if (!pure)
        return VisitInterrupt();
    }
    return std::nullopt;
  }

  bool pure{true};
};

bool IsPureHostExpression(const Expr &expression) {
  auto checker = make_object<HostExpressionChecker>();
  checker->Visit(expression);
  return checker->pure;
}

Stmt PackedCall(Array<Expr> args, Location loc = UnknownLoc()) {
  return Evaluate(
      Call(PrimType::Int(32), call_packed_op(), std::move(args), {}, {}, loc),
      loc);
}

class HopperIntrinsicLowerer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  HopperIntrinsicLowerer(bool disable_shuffle_elect,
                         Map<Var, Array<Expr>> descriptor_args)
      : disable_shuffle_elect_(disable_shuffle_elect),
        descriptor_args_(std::move(descriptor_args)) {}

  const Map<Var, Array<Expr>> &descriptor_args() const {
    return descriptor_args_;
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *sequence,
                            InplaceMode mode) final {
    Array<Stmt> statements;
    for (const Stmt &statement : sequence->seq) {
      Array<Stmt> pending;
      auto *previous = pending_host_;
      pending_host_ = &pending;
      Stmt rewritten = Mutate(statement, mode).ValueOrUnchanged(statement);
      pending_host_ = previous;
      for (const Stmt &initialization : pending)
        statements.push_back(initialization);
      statements.push_back(rewritten);
    }
    return SeqStmt(statements, sequence->loc);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *region,
                            InplaceMode mode) final {
    if (!launch_ && StartsDeviceRegion(region)) {
      // A descriptor is initialized immediately before its owning launch, in
      // the same host lexical scope. Host allocations/shape bindings already
      // dominate this point; sibling branches and launches cannot reuse an
      // out-of-scope descriptor.
      LaunchState state;
      launch_ = &state;
      Stmt body = RewriteRegion(region, mode);
      launch_ = nullptr;
      state.initializations.push_back(body);
      return SeqStmt(state.initializations, region->loc);
    }
    return RewriteRegion(region, mode);
  }

  UnchangedOr<Stmt> Mutate_(const ForObj *loop, InplaceMode mode) final {
    if (launch_)
      launch_->device_definitions.insert(loop->var.get());
    return StmtExprMutator::Mutate_(loop, mode);
  }

  UnchangedOr<Stmt> Mutate_(const BindObj *binding, InplaceMode mode) final {
    Stmt rewritten = StmtExprMutator::Mutate_(binding, mode)
                         .ValueOrUnchanged(GetRef<Stmt>(binding));
    auto current = rewritten.as_or_throw<Bind>();
    if (auto descriptor = current->value.as<Var>();
        descriptor && descriptor_args_.count(descriptor.value())) {
      // Keep the descriptor's canonical typed identity through an explicit
      // user/helper alias, including Bind(old_void_ptr, create_descriptor()).
      VarRemapSet(binding->var, descriptor.value());
      VarRemapSet(current->var, descriptor.value());
      return SeqStmt({}, binding->loc);
    }
    if (launch_) {
      launch_->device_definitions.insert(binding->var.get());
      launch_->device_definitions.insert(current->var.get());
      if ((current->var->ty.as<PrimTypeObj>() ||
           current->var->ty.as<PtrTypeObj>()) &&
          IsPureHostExpression(current->value)) {
        Expr expanded = launch_->host_values->Mutate(current->value)
                            .ValueOrUnchanged(current->value);
        if (UsesOnlyHostValues(expanded))
          launch_->host_values->VarRemapSet(current->var, expanded);
      }
      if (auto tensor = current->var.as<TensorVar>();
          tensor && tensor.value().scope().name() == "global") {
        auto call = current->value.as<CallObj>();
        if (call && call->op.same_as(decl_tensor_op())) {
          Expr expanded = launch_->host_values->Mutate(current->value)
                              .ValueOrUnchanged(current->value);
          if (IsPureHostExpression(expanded) && UsesOnlyHostValues(expanded)) {
            // Preserve the entire native view type, including layout/offset
            // and dynamic shape.  The alias remains in the device body; a
            // separate host alias is emitted lazily only if a descriptor uses
            // it.  Replacing a TensorVar with an arbitrary pointer expression
            // would violate tensor_data_ptr's native operand contract.
            Var host_view(current->var->name + "_tma_host", expanded->ty,
                          current->var->loc);
            launch_->host_views.Set(host_view,
                                    Bind(host_view, expanded, binding->loc));
            launch_->host_values->VarRemapSet(current->var, host_view);
          }
        } else if (current->value.as<TensorVar>()) {
          Expr expanded = launch_->host_values->Mutate(current->value)
                              .ValueOrUnchanged(current->value);
          if (UsesOnlyHostValues(expanded))
            launch_->host_values->VarRemapSet(current->var, expanded);
        }
      }
    }
    return rewritten;
  }

  UnchangedOr<Expr> Mutate_(const VarObj *variable, InplaceMode mode) final {
    auto result = StmtExprMutator::Mutate_(variable, mode);
    bool unchanged = result.IsUnchanged();
    Expr value = std::move(result).ValueOrUnchanged(GetRef<Expr>(variable));
    if (launch_ && def_region_kind() != kTVMFFIDefRegionKindNone) {
      launch_->device_definitions.insert(variable);
      if (auto defined = value.as<Var>())
        launch_->device_definitions.insert(defined.value().get());
    }
    if (def_region_kind() == kTVMFFIDefRegionKindNone) {
      if (auto descriptor = value.as<Var>();
          descriptor && descriptor_args_.count(descriptor.value()))
        RecordPrefetch(descriptor.value());
    }
    if (unchanged)
      return Unchanged();
    return value;
  }

  UnchangedOr<Expr> Mutate_(const CallObj *call, InplaceMode mode) final {
    if (call->op.same_as(prefetch_tma_descriptor())) {
      Expr rewritten = StmtExprMutator::Mutate_(call, mode)
                           .ValueOrUnchanged(GetRef<Expr>(call));
      if (prefetch_) {
        if (auto descriptor = rewritten.as_or_throw<Call>()->args[0].as<Var>())
          prefetch_->issued.insert(descriptor.value().get());
      }
      return rewritten;
    }
    if (!call->op.same_as(create_tma_descriptor()) &&
        !call->op.same_as(create_tma_im2col_descriptor()))
      return StmtExprMutator::Mutate_(call, mode);

    Call rewritten = StmtExprMutator::Mutate_(call, mode)
                         .ValueOrUnchanged(GetRef<Expr>(call))
                         .as_or_throw<Call>();
    Array<Expr> arguments = DecodeCallOperands(rewritten.get());
    TVM_FFI_CHECK_GE(arguments.size(), 3U, ValueError)
        << "TMA descriptor needs dtype, rank and a global address";
    TVM_FFI_CHECK(arguments[2]->ty.as<PtrTypeObj>(), TypeError)
        << "TMA descriptor global address must have PtrType";
    if (launch_) {
      for (size_t i = 0; i < arguments.size(); ++i) {
        Expr expanded = launch_->host_values->Mutate(arguments[i])
                            .ValueOrUnchanged(arguments[i]);
        TVM_FFI_CHECK(IsPureHostExpression(expanded) &&
                          UsesOnlyHostValues(expanded),
                      ValueError)
            << "TMA descriptor argument " << i
            << " depends on a device-local value and cannot be initialized "
               "on the host: "
            << expanded;
        arguments.Set(i, std::move(expanded));
      }
    }
    Call key(rewritten->ty, rewritten->op, arguments, {}, rewritten->ty_args,
             rewritten->loc);
    if (launch_) {
      auto found = launch_->descriptors.find(key);
      if (found != launch_->descriptors.end()) {
        RecordPrefetch(found->second);
        return found->second;
      }
    }

    std::string prefix = "tma";
    for (const Var &variable : UndefinedVars(arguments[2])) {
      if (variable->ty.as<TensorTypeObj>() || variable->ty.as<PtrTypeObj>()) {
        prefix = variable->name;
        break;
      }
    }
    Var descriptor(prefix + "_desc_" + std::to_string(next_descriptor_++),
                   PtrType(TensorMapType()), rewritten->loc);
    Array<Expr> initialization_args{
        StringImm(rewritten->op.same_as(create_tma_descriptor())
                      ? tvm_tensormap_create_tiled
                      : tvm_tensormap_create_im2col),
        descriptor};
    for (const Expr &argument : arguments)
      initialization_args.push_back(argument);
    descriptor_args_.Set(descriptor, initialization_args);

    Call allocation(descriptor->ty, stack_alloca_op(),
                    {StringImm("tensormap"), IntImm(PrimType::Int(32), 1)}, {},
                    {}, rewritten->loc);
    Stmt initialization =
        SeqStmt({Bind(descriptor, allocation, rewritten->loc),
                 PackedCall(initialization_args, rewritten->loc)},
                rewritten->loc);
    if (launch_) {
      for (const Expr &argument : arguments) {
        for (const Var &variable : UndefinedVars(argument))
          EmitHostView(variable);
      }
      launch_->descriptors.emplace(key, descriptor);
      launch_->initializations.push_back(initialization);
    } else {
      TVM_FFI_CHECK(pending_host_ != nullptr, ValueError)
          << "TMA descriptor requires a statement evaluation context";
      pending_host_->push_back(initialization);
    }
    RecordPrefetch(descriptor);
    return descriptor;
  }

private:
  struct LaunchState {
    Array<Stmt> initializations;
    std::unordered_map<Call, Var, StructuralHash, StructuralEqual> descriptors;
    std::unordered_set<const VarObj *> device_definitions;
    ObjectPtr<StmtExprMutator> host_values{make_object<StmtExprMutator>()};
    Map<Var, Bind> host_views;
    std::unordered_set<const VarObj *> emitted_views;
  };

  struct PrefetchFrame {
    Array<Var> descriptors;
    std::unordered_set<const VarObj *> visited;
    std::unordered_set<const VarObj *> issued;
  };

  bool UsesOnlyHostValues(const Expr &expression) const {
    for (const Var &variable : UndefinedVars(expression)) {
      if (launch_->device_definitions.count(variable.get()))
        return false;
    }
    return true;
  }

  void RecordPrefetch(const Var &descriptor) {
    if (prefetch_ && prefetch_->visited.insert(descriptor.get()).second)
      prefetch_->descriptors.push_back(descriptor);
  }

  void EmitHostView(const Var &variable) {
    auto binding = launch_->host_views.Get(variable);
    if (!binding || !launch_->emitted_views.insert(variable.get()).second)
      return;
    for (const Var &dependency : UndefinedVars(binding.value()->value))
      EmitHostView(dependency);
    launch_->initializations.push_back(binding.value());
  }

  Stmt RewriteRegion(const RegionStmtObj *region, InplaceMode mode) {
    if (launch_) {
      for (const Var &parameter : region->body_params)
        launch_->device_definitions.insert(parameter.get());
      for (const Var &result : region->result_vars)
        launch_->device_definitions.insert(result.get());
    }
    bool is_thread_x =
        region->op.same_as(launch_thread_op()) &&
        region->args[0].as_or_throw<StringImm>()->value == "threadIdx.x";
    PrefetchFrame frame;
    auto *previous = prefetch_;
    if (is_thread_x)
      prefetch_ = &frame;
    Stmt rewritten = StmtExprMutator::Mutate_(region, mode)
                         .ValueOrUnchanged(GetRef<Stmt>(region));
    prefetch_ = previous;
    if (!is_thread_x || frame.descriptors.empty())
      return rewritten;

    auto current = rewritten.as_or_throw<RegionStmt>();
    PrimVar thread = current->body_params[0].as_or_throw<PrimVar>();
    PrimExpr condition = disable_shuffle_elect_
                             ? PrimExpr(EQ(thread, IntImm(thread.ty(), 0)))
                             : Call(PrimType::Bool(), tl_shuffle_elect(),
                                    {IntImm(PrimType::Int(32), 0)})
                                   .as_or_throw<PrimExpr>();
    Array<Stmt> prefetch_calls;
    for (const Var &descriptor : frame.descriptors) {
      if (frame.issued.count(descriptor.get()))
        continue;
      prefetch_calls.push_back(Evaluate(
          Call(PrimType::Void(), prefetch_tma_descriptor(), {descriptor})));
    }
    if (prefetch_calls.empty())
      return rewritten;
    SeqStmt body(
        {If(condition, SeqStmt(prefetch_calls), std::nullopt), current->body},
        current->loc);
    return RegionStmt(current->op, current->args, current->body_params,
                      current->attrs, body, current->result_vars, current->loc);
  }

  bool disable_shuffle_elect_;
  int next_descriptor_{0};
  LaunchState *launch_{nullptr};
  PrefetchFrame *prefetch_{nullptr};
  Array<Stmt> *pending_host_{nullptr};
  Map<Var, Array<Expr>> descriptor_args_;
};

class ResetL2BeforeHostReturn : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  explicit ResetL2BeforeHostReturn(Stmt reset) : reset_(std::move(reset)) {}

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *region,
                            InplaceMode mode) final {
    if (StartsDeviceRegion(region))
      return Unchanged();
    return StmtExprMutator::Mutate_(region, mode);
  }

  UnchangedOr<Stmt> Mutate_(const ReturnObj *statement, InplaceMode) final {
    return SeqStmt({reset_, GetRef<Stmt>(statement)}, statement->loc);
  }

private:
  Stmt reset_;
};

SeqStmt MaterializeL2Policy(const tirx::Function &function, SeqStmt body,
                            bool *materialized) {
  // Retain l2_persistent_map for source-wrapper adapters.  This phase marker
  // belongs to the host IR consumer, and prevents a repeated pass from nesting
  // the stream state save/reset pair.  A producer replacing the map must clear
  // this marker before invoking the host lowering again.
  if (function->GetAttr<bool>("tl.l2_policy_materialized").value_or(false))
    return body;
  auto entries =
      function->GetAttr<Map<String, Array<Expr>>>("l2_persistent_map");
  if (!entries || entries.value().empty())
    return body;
  std::unordered_map<std::string, TensorVar> parameters;
  for (const Var &parameter : function->params) {
    if (auto tensor = parameter.as<TensorVar>()) {
      TVM_FFI_CHECK(parameters.emplace(parameter->name, tensor.value()).second,
                    ValueError)
          << "L2 persistent annotations require unambiguous tensor names";
    }
  }
  Array<Stmt> prologue;
  auto analyzer = CreateTileLangAnalyzer();
  for (const auto &[name, arguments] : entries.value()) {
    auto found = parameters.find(name);
    if (found == parameters.end())
      continue;
    TVM_FFI_CHECK_GE(arguments.size(), 2U, ValueError)
        << "L2 persistent policy needs hit_ratio and size_in_bytes";
    const TensorVar &tensor = found->second;
    Array<PrimExpr> zeros;
    for (const PrimExpr &dimension : tensor.shape())
      zeros.push_back(IntImm(dimension.ty(), 0));
    PrimExpr offset = TensorElementOffset(tensor, zeros);
    if (offset.ty().bits() < 64)
      offset = cast(offset, PrimType::Int(64));
    int bits = tensor.dtype().bits() * tensor.dtype().lanes();
    PrimExpr bit_offset = offset * IntImm(offset.ty(), bits);
    PrimExpr byte_offset{ffi::UnsafeInit{}};
    if (bits % 8 == 0) {
      byte_offset = offset * IntImm(offset.ty(), bits / 8);
    } else {
      TVM_FFI_CHECK(
          analyzer->CanProveEqual(FloorMod(bit_offset, IntImm(offset.ty(), 8)),
                                  IntImm(offset.ty(), 0)),
          ValueError)
          << "L2 persistent pointer must start at a byte boundary";
      byte_offset = FloorDiv(bit_offset, IntImm(offset.ty(), 8));
    }
    Expr pointer = reinterpret(PtrType::VoidPtrType(), tensor.data());
    if (!IsZero(byte_offset))
      pointer = Call(PtrType::VoidPtrType(), handle_add_byte_offset_op(),
                     {pointer, byte_offset});
    prologue.push_back(
        PackedCall({StringImm(tvm_cuda_stream_set_access_policy_window),
                    pointer, arguments[1], arguments[0]}));
  }
  if (prologue.empty())
    return body;
  *materialized = true;
  Stmt reset =
      PackedCall({StringImm(tvm_cuda_stream_reset_access_policy_window)});
  auto returns = make_object<ResetL2BeforeHostReturn>(reset);
  body = returns->Mutate(body).ValueOrUnchanged(body);
  prologue.push_back(body);
  prologue.push_back(reset);
  return SeqStmt(prologue, body->loc);
}
} // namespace

tvm::transform::Pass LowerHopperIntrin() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext context) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto lowerer = make_object<HopperIntrinsicLowerer>(
        context->GetConfig<bool>(kDisableShuffleElect).value_or(false),
        function->GetAttr<Map<Var, Array<Expr>>>("tma_descriptor_args")
            .value_or(Map<Var, Array<Expr>>()));
    SeqStmt body = lowerer->Mutate(function->body, InplaceMode::kDisallow)
                       .ValueOrUnchanged(function->body);
    bool materialized = false;
    body = MaterializeL2Policy(function, std::move(body), &materialized);
    function.CopyOnWrite()->body = body;
    if (materialized)
      function =
          WithAttr(std::move(function), "tl.l2_policy_materialized", true);
    return WithAttr(std::move(function), "tma_descriptor_args",
                    lowerer->descriptor_args());
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LowerHopperIntrin");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.LowerHopperIntrin",
                                   LowerHopperIntrin);
}
#endif // (CUDA_MAJOR_VERSION >= 12)
} // namespace tvm::tl
