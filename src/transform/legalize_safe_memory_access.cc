#include "compat/expr.h"
#include "op/builtin.h"
#include "op/utils.h"

#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

struct Guards {
  Array<PrimExpr> conditions;
  Array<PrimExpr> independent_conditions;
  std::vector<std::pair<PrimVar, PrimExpr>> bindings;
};

struct PointerAccess {
  TensorRegion region;
  PrimExpr extent;
  PrimExpr mask;
};

bool IsAddress(const CallNode *call) {
  return call->op.same_as(access_ptr()) || call->op.same_as(address_of_op());
}

std::optional<PointerAccess> GetPointerAccess(const Expr &expression) {
  const auto *call = expression.as<CallNode>();
  if (!call || !IsAddress(call) || call->args.empty())
    return std::nullopt;
  if (call->op.same_as(access_ptr())) {
    return PointerAccess{NormalizeToBufferRegion(call->args[0]),
                         call->args[1].as_or_throw<PrimExpr>(),
                         call->args[2].as_or_throw<PrimExpr>()};
  }
  auto attributes = call->attrs.value_or(DictAttrs()).as_or_throw<DictAttrs>();
  auto region = attributes.GetAttr<TensorRegion>("tl.access_region");
  return PointerAccess{
      region.value_or(NormalizeToBufferRegion(call->args[0])),
      attributes.GetAttr<PrimExpr>("tl.access_extent").value_or(PrimExpr(1)),
      attributes.GetAttr<PrimExpr>("tl.access_mask").value_or(PrimExpr(3))};
}

bool IsAtomic(const CallNode *call) {
  const auto *operation = call->op.as<OpNode>();
  if (!operation)
    return false;
  std::string name = operation->name;
  return name.starts_with("tl.atomic_") && name.ends_with("_elem_op");
}

class SafeMemoryRewriter : public StmtExprMutatorWithAnalyzer {
public:
  SafeMemoryRewriter(const sym::Analyzer &analyzer, bool enabled, bool warnings)
      : StmtExprMutatorWithAnalyzer(analyzer), enabled_(enabled),
        warnings_(warnings) {}

  void Initialize(const tirx::Function &function) {
    MarkTensorParamShapes(function);
  }

private:
  PrimExpr Simplify(const PrimExpr &value) {
    try {
      return analyzer_->Simplify(value);
    } catch (const std::exception &) {
      return value;
    }
  }

  bool Proved(const PrimExpr &condition) {
    try {
      return analyzer_->CanProve(condition, sym::ProofStrength::kSymbolicBound);
    } catch (const std::exception &) {
      return false;
    }
  }

  PrimExpr Freeze(const PrimExpr &value, Guards *guards) {
    if (SideEffect(value) == CallEffectKind::kPure)
      return value;
    PrimVar variable("safe_index_" + std::to_string(next_index_++), value.ty(),
                     value->loc);
    guards->bindings.emplace_back(variable, value);
    return variable;
  }

  PrimExpr Lane(const PrimExpr &value, int lane) {
    if (value.ty().IsScalar())
      return value;
    if (const auto *ramp = value.as<RampNode>())
      return ramp->base + ramp->stride * PrimExpr(lane);
    if (const auto *broadcast = value.as<BroadcastNode>())
      return broadcast->value;
    return Simplify(Shuffle({value}, {IntImm::Int32(lane)}));
  }

  void AddBound(const PrimExpr &condition, bool global, const TensorVar &tensor,
                Guards *guards) {
    if (Proved(condition))
      return;
    if (global) {
      guards->conditions.push_back(Simplify(condition));
    } else if (warnings_) {
      LOG(WARNING) << "Index access may exceed tensor bounds for "
                   << tensor->name << ": " << condition;
    }
  }

  Array<PrimExpr> GuardIndices(const TensorVar &tensor,
                               const Array<PrimExpr> &indices, Guards *guards) {
    TVM_FFI_CHECK_EQ(indices.size(), tensor.shape().size(), ValueError);
    bool global = tensor.scope().name() == "global";
    bool has_mutable_index = false;
    for (const PrimExpr &index : indices)
      has_mutable_index |= SideEffect(index) != CallEffectKind::kPure;
    Guards independent;
    Array<PrimExpr> result;
    for (size_t axis = 0; axis < indices.size(); ++axis) {
      Guards *bounds =
          global && has_mutable_index &&
                  SideEffect(indices[axis]) == CallEffectKind::kPure
              ? &independent
              : guards;
      PrimExpr index = global ? Freeze(indices[axis], guards) : indices[axis];
      result.push_back(index);
      for (int lane = 0; lane < index.ty().lanes(); ++lane) {
        PrimExpr scalar = Lane(index, lane);
        PrimType bound_type = scalar.ty() == PrimType::UInt(64)
                                  ? PrimType::UInt(64)
                                  : PrimType::Int(64);
        scalar = prim::cast(scalar, bound_type);
        AddBound(scalar >= MakeConst(bound_type, 0), global, tensor, bounds);
        AddBound(scalar < prim::cast(tensor.shape()[axis], bound_type), global,
                 tensor, bounds);
      }
    }
    guards->independent_conditions = independent.conditions;
    return result;
  }

  PrimExpr Condition(const Array<PrimExpr> &conditions) {
    PrimExpr result = IntImm::Bool(true);
    for (const PrimExpr &condition : conditions)
      result = result && condition;
    return Simplify(result);
  }

  PrimExpr Condition(const Guards &guards) {
    return Condition(guards.conditions);
  }

  PrimExpr BindExpression(PrimExpr result, const Guards &guards,
                          Optional<PrimExpr> fallback = std::nullopt) {
    for (auto binding = guards.bindings.rbegin();
         binding != guards.bindings.rend(); ++binding) {
      result = Let(binding->first, binding->second, result, result->loc);
    }
    if (!guards.independent_conditions.empty())
      result =
          tvm::if_then_else(Condition(guards.independent_conditions), result,
                            fallback.value_or(MakeConst(result.ty(), 0)));
    return result;
  }

  Stmt BindStatement(const Stmt &statement, const Guards &guards) {
    Array<Stmt> statements;
    for (const auto &[variable, value] : guards.bindings)
      statements.push_back(Bind(variable, value));
    statements.push_back(statement);
    Stmt result = guards.bindings.empty() ? statement
                                          : SeqStmt(statements, statement->loc);
    if (!guards.independent_conditions.empty())
      result = If(Condition(guards.independent_conditions), SeqStmt(result),
                  std::nullopt, statement->loc);
    return result;
  }

  Optional<TensorVar> Origin(const Expr &expression) {
    if (auto tensor = expression.as<TensorVar>())
      return origins_.Get(*tensor).value_or(*tensor);
    if (const auto *load = expression.as<TensorLoadNode>())
      return Origin(load->source);
    if (const auto *region = expression.as<TensorRegionNode>())
      return Origin(region->source);
    const auto *call = expression.as<CallNode>();
    if (call && !call->args.empty() &&
        (IsAddress(call) || call->op.same_as(tensor_data_ptr_op()) ||
         call->op.same_as(decl_tensor_op())))
      return Origin(call->args[0]);
    return std::nullopt;
  }

  PrimExpr SafeValue(const TensorVar &tensor, PrimType type) {
    auto value = safe_values_.Get(tensor);
    if (!value)
      value = safe_values_.Get(Origin(tensor).value_or(tensor));
    PrimExpr result = value.value_or(MakeConst(type.WithLanes(1), 0));
    if (result.ty().lanes() != type.lanes()) {
      TVM_FFI_CHECK(result.ty().IsScalar(), ValueError)
          << "Safe value must be scalar or match the access lanes";
      result = Broadcast(result, type.lanes());
    }
    return Simplify(prim::cast(result, type));
  }

  Expr SnapshotPointer(const Expr &expression, Guards *guards) {
    auto access = GetPointerAccess(expression);
    if (!access)
      return expression;
    const auto *call = expression.as<CallNode>();
    Array<Range> ranges;
    Array<PrimExpr> indices;
    for (const Range &range : access->region->region) {
      PrimExpr begin = Freeze(range->begin, guards);
      PrimExpr extent = Freeze(range->extent, guards);
      indices.push_back(begin);
      ranges.push_back(Range::FromMinExtent(begin, extent));
    }
    TensorRegion region(access->region->source, ranges, TensorRegionType(),
                        expression->loc);
    PrimExpr extent = Freeze(access->extent, guards);
    PrimExpr mask = Freeze(access->mask, guards);
    auto arguments = call->args;
    auto attributes =
        call->attrs.value_or(DictAttrs()).as_or_throw<DictAttrs>();
    if (call->op.same_as(access_ptr())) {
      arguments.Set(0, call->args[0].as<TensorLoadNode>()
                           ? Expr(TensorLoad(region->source, indices))
                           : Expr(region));
      arguments.Set(1, extent);
      arguments.Set(2, mask);
    } else {
      arguments.Set(0, TensorLoad(region->source, indices));
      attributes.CopyOnWrite()->dict.Set("tl.access_region", region);
      attributes.CopyOnWrite()->dict.Set("tl.access_extent", extent);
      attributes.CopyOnWrite()->dict.Set("tl.access_mask", mask);
    }
    return Call(call->ty, call->op, arguments, attributes, call->ty_args,
                call->loc);
  }

  Expr GuardPointer(const Expr &expression, Guards *guards,
                    Optional<PrimExpr> required_extent = std::nullopt) {
    auto access = GetPointerAccess(expression);
    if (!access)
      return expression;
    TensorVar tensor = access->region->source.as_or_throw<TensorVar>();
    bool global = tensor.scope().name() == "global";
    const auto *call = expression.as<CallNode>();
    PrimExpr extent = global ? Freeze(access->extent, guards) : access->extent;
    Array<Range> ranges;
    Array<PrimExpr> indices;
    bool flat_base = call->args[0].as<TensorLoadNode>() != nullptr;
    PrimExpr region_size = IntImm::Int64(1);
    for (size_t axis = 0; axis < access->region->region.size(); ++axis) {
      Range range = access->region->region[axis];
      PrimExpr begin = range->begin;
      PrimExpr length = range->extent;
      if (flat_base && axis + 1 == access->region->region.size()) {
        length = tvm::max(
            length, prim::cast(required_extent.value_or(extent), length.ty()));
      }
      if (global) {
        begin = Freeze(begin, guards);
        length = Freeze(length, guards);
      }
      indices.push_back(begin);
      ranges.push_back(Range::FromMinExtent(begin, length));
      region_size = region_size * prim::cast(length, PrimType::Int(64));
      if (GetConstInt(access->mask).value_or(3) != 0) {
        PrimType bound_type = begin.ty() == PrimType::UInt(64)
                                  ? PrimType::UInt(64)
                                  : PrimType::Int(64);
        PrimExpr wide_begin = prim::cast(begin, bound_type);
        PrimExpr wide_length = prim::cast(length, bound_type);
        PrimExpr wide_shape = prim::cast(tensor.shape()[axis], bound_type);
        AddBound(wide_begin >= MakeConst(bound_type, 0), global, tensor,
                 guards);
        AddBound(wide_length >= MakeConst(bound_type, 0), global, tensor,
                 guards);
        AddBound(wide_length <= wide_shape, global, tensor, guards);
        AddBound(wide_begin <= wide_shape - wide_length, global, tensor,
                 guards);
      }
    }
    if (!flat_base && GetConstInt(access->mask).value_or(3) != 0) {
      PrimExpr required =
          prim::cast(required_extent.value_or(extent), PrimType::Int(64));
      AddBound(required >= IntImm::Int64(0), global, tensor, guards);
      AddBound(required <= region_size, global, tensor, guards);
      AddBound(prim::cast(extent, PrimType::Int(64)) <= region_size, global,
               tensor, guards);
    }
    TensorRegion region(tensor, ranges, TensorRegionType(), expression->loc);
    auto arguments = call->args;
    auto attributes =
        call->attrs.value_or(DictAttrs()).as_or_throw<DictAttrs>();
    if (call->op.same_as(access_ptr())) {
      arguments.Set(0, region);
      arguments.Set(1, extent);
    } else {
      arguments.Set(0, TensorLoad(tensor, indices));
      attributes.CopyOnWrite()->dict.Set("tl.access_region", region);
    }
    return Call(call->ty, call->op, arguments, attributes, call->ty_args,
                call->loc);
  }

  Call GuardCallPointers(const Call &call, Guards *guards) {
    Array<Expr> arguments;
    for (const Expr &argument : call->args)
      arguments.push_back(GuardPointer(argument, guards));
    return Call(call->ty, call->op, arguments, call->attrs, call->ty_args,
                call->loc);
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtNode *node, InplaceMode mode) final {
    auto previous_values = safe_values_;
    auto previous_origins = origins_;
    auto result = StmtExprMutatorWithAnalyzer::Mutate_(node, mode);
    safe_values_ = previous_values;
    origins_ = previous_origins;
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                            InplaceMode mode) final {
    auto previous_values = safe_values_;
    if (auto values = node->attrs.GetAttr<Map<TensorVar, PrimExpr>>(
            attr::kSafeValueMap)) {
      for (const auto &[tensor, value] : *values) {
        TVM_FFI_CHECK(SideEffect(value) == CallEffectKind::kPure, ValueError)
            << "Safe-value annotations must be pure";
        safe_values_.Set(tensor, value);
      }
    }
    RegionStmt result = StmtExprMutatorWithAnalyzer::Mutate_(node, mode)
                            .ValueOrUnchanged(GetRef<Stmt>(node))
                            .as_or_throw<RegionStmt>();
    safe_values_ = previous_values;
    DictAttrs attributes = result->attrs;
    attributes.CopyOnWrite()->dict.erase(attr::kSafeValueMap);
    result.CopyOnWrite()->attrs = attributes;
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *node, InplaceMode mode) final {
    if (auto tensor = node->var.as<TensorVar>()) {
      if (auto origin = Origin(node->value))
        origins_.Set(*tensor, *origin);
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(node, mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *node,
                                InplaceMode mode) final {
    TensorLoad original = GetRef<TensorLoad>(node);
    TensorLoad load = StmtExprMutator::Mutate_(node, mode)
                          .ValueOrUnchanged(original)
                          .as_or_throw<TensorLoad>();
    if (!enabled_)
      return load;
    TensorVar tensor = load->source.as_or_throw<TensorVar>();
    Guards guards;
    Array<PrimExpr> indices = GuardIndices(tensor, load->indices, &guards);
    PrimExpr result = TensorLoad(tensor, indices, load->loc);
    if (!guards.conditions.empty())
      result = tvm::if_then_else(Condition(guards), result,
                                 SafeValue(tensor, result.ty()));
    return BindExpression(result, guards, SafeValue(tensor, result.ty()));
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node,
                            InplaceMode mode) final {
    TensorStore original = GetRef<TensorStore>(node);
    TensorStore store = StmtExprMutator::Mutate_(node, mode)
                            .ValueOrUnchanged(original)
                            .as_or_throw<TensorStore>();
    if (!enabled_)
      return store;
    Guards guards;
    TensorVar tensor = store->dest.as_or_throw<TensorVar>();
    Array<PrimExpr> indices = GuardIndices(tensor, store->indices, &guards);
    Stmt result = TensorStore(tensor, indices, store->value, store->loc);
    if (!guards.conditions.empty())
      result = If(Condition(guards), SeqStmt(result), std::nullopt, store->loc);
    return BindStatement(result, guards);
  }

  UnchangedOr<Expr> Mutate_(const CallNode *node, InplaceMode mode) final {
    if (IsAddress(node)) {
      Array<Expr> arguments;
      for (size_t index = 0; index < node->args.size(); ++index) {
        if (index == 0 && node->args[index].as<TensorLoadNode>()) {
          const auto *load = node->args[index].as<TensorLoadNode>();
          Array<PrimExpr> indices;
          for (const PrimExpr &coordinate : load->indices)
            indices.push_back(
                Mutate(coordinate, mode).ValueOrUnchanged(coordinate));
          arguments.push_back(TensorLoad(load->source, indices, load->loc));
        } else {
          arguments.push_back(Mutate(node->args[index], mode)
                                  .ValueOrUnchanged(node->args[index]));
        }
      }
      return Call(node->ty, node->op, arguments, node->attrs, node->ty_args,
                  node->loc);
    }
    Expr original = GetRef<Expr>(node);
    Expr result = StmtExprMutatorWithAnalyzer::Mutate_(node, mode)
                      .ValueOrUnchanged(original);
    auto call = result.as<Call>();
    if (!enabled_ || !call || !IsAtomic(call->get()) ||
        !result.as<PrimExpr>() || result->ty == PrimType::Void() ||
        call.value()->args.empty())
      return result;
    auto destination = GetPointerAccess(call.value()->args[0]);
    if (!destination)
      return result;
    Guards guards;
    Call guarded = GuardCallPointers(*call, &guards);
    PrimExpr value = guarded.as_or_throw<PrimExpr>();
    if (!guards.conditions.empty()) {
      value = tvm::if_then_else(
          Condition(guards), value,
          SafeValue(destination->region->source.as_or_throw<TensorVar>(),
                    value.ty()));
    }
    return BindExpression(value, guards);
  }

  Stmt RewriteAsync(const Call &call, const Location &location) {
    TVM_FFI_CHECK(call->args.size() == 3 || call->args.size() == 4, ValueError);
    auto source = GetPointerAccess(call->args[1]);
    auto destination = GetPointerAccess(call->args[0]);
    if (!source || !destination)
      return Evaluate(call, location);
    TensorVar source_tensor = source->region->source.as_or_throw<TensorVar>();
    TensorVar destination_tensor =
        destination->region->source.as_or_throw<TensorVar>();
    TVM_FFI_CHECK(source_tensor.dtype() == destination_tensor.dtype(),
                  TypeError)
        << "Asynchronous copies require matching source and destination "
           "element types";
    Guards guards;
    auto arguments = call->args;
    arguments.Set(0, SnapshotPointer(arguments[0], &guards));
    arguments.Set(1, SnapshotPointer(arguments[1], &guards));
    PrimExpr count = Freeze(arguments[2].as_or_throw<PrimExpr>(), &guards);
    arguments.Set(2, count);
    PrimExpr predicate =
        arguments.size() == 4
            ? Freeze(arguments[3].as_or_throw<PrimExpr>(), &guards)
            : IntImm::Bool(true);
    if (arguments.size() == 4)
      arguments.Set(3, predicate);
    arguments.Set(1, GuardPointer(arguments[1], &guards, count));
    destination = GetPointerAccess(arguments[0]);
    PrimExpr condition = Condition(guards);
    if (guards.conditions.empty()) {
      return BindStatement(Evaluate(Call(call->ty, call->op, arguments,
                                         call->attrs, call->ty_args, call->loc),
                                    location),
                           guards);
    }
    PrimExpr fallback = SafeValue(source_tensor, destination_tensor.dtype());
    if (IsZeroBitPattern(fallback)) {
      PrimExpr combined = Simplify(predicate && condition);
      if (arguments.size() == 3)
        arguments.push_back(combined);
      else
        arguments.Set(3, combined);
      return BindStatement(Evaluate(Call(call->ty, call->op, arguments,
                                         call->attrs, call->ty_args, call->loc),
                                    location),
                           guards);
    }
    Array<PrimExpr> indices;
    for (const Range &range : destination->region->region)
      indices.push_back(range->begin);
    TVM_FFI_CHECK(!indices.empty(), ValueError)
        << "Asynchronous safe-value filling requires indexed destination "
           "storage";
    PrimExpr fill_count = count;
    PrimVar lane("safe_fill", fill_count.ty());
    indices.Set(indices.size() - 1,
                indices.back() + prim::cast(lane, indices.back().ty()));
    fallback =
        tvm::if_then_else(predicate, fallback, MakeConst(fallback.ty(), 0));
    Stmt fill = For(
        lane, MakeConst(fill_count.ty(), 0), fill_count,
        MakeConst(fill_count.ty(), 1), ForKind::kDefault,
        SeqStmt(TensorStore(destination_tensor, indices, fallback, location)),
        {}, location);
    Stmt result =
        If(condition,
           SeqStmt(Evaluate(Call(call->ty, call->op, arguments, call->attrs,
                                 call->ty_args, call->loc),
                            location)),
           SeqStmt(fill), location);
    return BindStatement(result, guards);
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateNode *node, InplaceMode mode) final {
    const auto *original_call = node->value.as<CallNode>();
    if (original_call &&
        original_call->op.same_as(Op::Get("tl.safe_value_hint"))) {
      TensorVar tensor = original_call->args[0].as_or_throw<TensorVar>();
      PrimExpr value = original_call->args[1].as_or_throw<PrimExpr>();
      TVM_FFI_CHECK(SideEffect(value) == CallEffectKind::kPure, ValueError)
          << "Safe-value hints must be pure; bind mutable or effectful values "
             "before the hint";
      safe_values_.Set(tensor, value);
      return SeqStmt({}, node->loc);
    }
    Expr expression = Mutate(node->value, mode).ValueOrUnchanged(node->value);
    auto call = expression.as<Call>();
    if (!enabled_ || !call)
      return Evaluate(expression, node->loc);
    if (call.value()->op.same_as(ptx_cp_async()))
      return RewriteAsync(*call, node->loc);
    if (IsAtomic(call->get()) || call.value()->op.same_as(call_extern_op()) ||
        call.value()->op.same_as(call_pure_extern_op())) {
      Guards guards;
      Call guarded = GuardCallPointers(*call, &guards);
      Stmt result = Evaluate(guarded, node->loc);
      if (!guards.conditions.empty())
        result =
            If(Condition(guards), SeqStmt(result), std::nullopt, node->loc);
      return BindStatement(result, guards);
    }
    return Evaluate(expression, node->loc);
  }

  bool enabled_;
  bool warnings_;
  int next_index_ = 0;
  Map<TensorVar, PrimExpr> safe_values_;
  Map<TensorVar, TensorVar> origins_;
};
} // namespace

tvm::transform::Pass LegalizeSafeMemoryAccess() {
  auto pass =
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext context) -> ffi::Optional<tirx::Function> {
    if (!function->body)
      return function;
    sym::Analyzer analyzer;
    auto rewriter = ffi::make_object<SafeMemoryRewriter>(
        analyzer,
        !context->GetConfig<bool>(kDisableSafeMemoryLegalize).value_or(false),
        !context->GetConfig<bool>(kDisableOutOfBoundWarning).value_or(true));
    rewriter->Initialize(function);
    SeqStmt body = *function->body;
    function.CopyOnWrite()->body =
        rewriter->Mutate(body).ValueOrUnchanged(body).as_or_throw<SeqStmt>();
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.LegalizeSafeMemoryAccess");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LegalizeSafeMemoryAccess",
                                   LegalizeSafeMemoryAccess);
}
} // namespace tvm::tl
