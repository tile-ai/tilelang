#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "op/prim.h"
/*!
 * \file simplify.cc
 * \brief Statement simplifier based on analyzer and remove useless parameters
 * of TL PrimFunc.
 */

#include "compat/region.h"
#include "transform/common/loop_aware_mutator.h"

#include "compat/expr.h"
#include "compat/visit.h"
#include "transform/common/warp_specialize.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/with_context.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace tvm::tl {
namespace {

struct SimplifyConfigNode : public ffi::Object {
  bool transitively_prove_inequalities;
  bool propagate_knowns_to_prove_conditional;
  bool propagate_knowns_to_simplify_expressions;
  bool convert_boolean_to_and_of_ors;
  bool apply_constraints_to_boolean_branches;
  bool enable_simplify_let_inline;

  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<SimplifyConfigNode>()
        .def_ro("transitively_prove_inequalities",
                &SimplifyConfigNode::transitively_prove_inequalities,
                refl::DefaultValue(false))
        .def_ro("propagate_knowns_to_prove_conditional",
                &SimplifyConfigNode::propagate_knowns_to_prove_conditional,
                refl::DefaultValue(false))
        .def_ro("propagate_knowns_to_simplify_expressions",
                &SimplifyConfigNode::propagate_knowns_to_simplify_expressions,
                refl::DefaultValue(false))
        .def_ro("convert_boolean_to_and_of_ors",
                &SimplifyConfigNode::convert_boolean_to_and_of_ors,
                refl::DefaultValue(false))
        .def_ro("apply_constraints_to_boolean_branches",
                &SimplifyConfigNode::apply_constraints_to_boolean_branches,
                refl::DefaultValue(false))
        .def_ro("enable_simplify_let_inline",
                &SimplifyConfigNode::enable_simplify_let_inline,
                refl::DefaultValue(true));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.transform.SimplifyConfig",
                                    SimplifyConfigNode, ffi::Object);
};

class SimplifyConfig : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SimplifyConfig, ffi::ObjectRef,
                                                SimplifyConfigNode);
};

std::unordered_set<const VarNode *>
UsedVariables(const ffi::ObjectRef &object) {
  std::unordered_set<const VarNode *> used;
  ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
      object,
      [&](const Var &variable,
          TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::WalkResult> {
        if (kind == kTVMFFIDefRegionKindNone) {
          used.insert(variable.get());
        }
        return ffi::WalkResult::Advance();
      });
  return used;
}

bool ContainsVolatileStorage(const tirx::Function &function) {
  bool found = function->attrs->dict.count("tirx.volatile");
  PostOrderVisit(function, [&](const ffi::ObjectRef &object) {
    if (const auto *call = object.as<CallNode>(); call && call->attrs) {
      if (const auto *attributes = call->attrs.value().as<DictAttrsNode>())
        found |= attributes->dict.count("tirx.volatile");
    }
    if (const auto *region = object.as<RegionStmtNode>())
      found |= region->attrs->dict.count("volatile_scope") ||
               region->attrs->dict.count("tirx.volatile");
  });
  return found;
}

class UnusedBindRemover : public tvm::tl::StmtExprMutator {
public:
  UnusedBindRemover(std::unordered_set<const VarNode *> used,
                    bool volatile_storage)
      : used_(std::move(used)), volatile_storage_(volatile_storage) {}
  bool changed{false};

private:
  ffi::UnchangedOr<Stmt> Mutate_(const BindNode *node,
                                 ffi::InplaceMode mode) final {
    const auto *call = node->value.as<CallNode>();
    bool tensor_view = call && call->op.same_as(tirx::decl_tensor_op());
    bool removable = node->var.as<PrimVar>() ||
                     node->var->ty.as<PtrTypeNode>() || tensor_view;
    auto allowed_effect = node->var.as<PrimVar>() && !volatile_storage_
                              ? CallEffectKind::kReadState
                              : CallEffectKind::kPure;
    bool scheduled =
        call && call->attrs && call->attrs.value().as<DictAttrsNode>() &&
        call->attrs.value().as<DictAttrsNode>()->dict.count(kWSOpIdKey);
    if (!scheduled && removable && !used_.count(node->var.get()) &&
        SideEffect(node->value) <= allowed_effect) {
      changed = true;
      return SeqStmt({}, node->loc);
    }
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }
  std::unordered_set<const VarNode *> used_;
  bool volatile_storage_;
};

class StmtSimplifier : public ::tvm::tl::LoopAwareMutator {
public:
  using Parent = ::tvm::tl::LoopAwareMutator;
  using Parent::Mutate;

  StmtSimplifier(sym::Analyzer analyzer, SimplifyConfig config,
                 const tirx::Function &function)
      : Parent(analyzer), config_(std::move(config)),
        propagate_now_(config_->propagate_knowns_to_simplify_expressions),
        noalias_(function->GetAttr<bool>("tirx.noalias").value_or(false)),
        volatile_storage_(ContainsVolatileStorage(function)) {
    MarkTensorParamShapes(function);
    PostOrderVisit(function, [&](const ffi::ObjectRef &object) {
      if (const auto *region = object.as<RegionStmtNode>())
        preserve_ws_schedule_ |= region->attrs->dict.count(kWSScheduleKey);
    });
    for (const Var &parameter : function->params) {
      if (parameter.as<tirx::TensorVar>()) {
        origins_.emplace(parameter, parameter);
      }
    }
  }

private:
  struct KnownValue {
    tirx::TensorVar tensor;
    ffi::Array<PrimExpr> indices;
    PrimExpr value;
  };

  PrimExpr SimplifyBoolean(const PrimExpr &expression) {
    PrimExpr simplified = tl::arith::Simplify(analyzer_, expression);
    if (tl::arith::CanProve(analyzer_, simplified)) {
      return IntImm::Bool(true);
    }
    if (tl::arith::CanProve(analyzer_, !simplified)) {
      return IntImm::Bool(false);
    }
    return simplified;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtNode *node,
                                 ffi::InplaceMode mode) final {
    return constraint_scope_.WithNewScope([&]() -> ffi::UnchangedOr<Stmt> {
      ffi::Array<Stmt> statements;
      for (const Stmt &statement : node->seq) {
        Stmt result = Mutate(statement, mode).ValueOrUnchanged(statement);
        statements.push_back(result);
        if (const auto *evaluate = result.as<EvaluateNode>()) {
          if (const auto *call = evaluate->value.as<CallNode>();
              call && call->op.same_as(prim::assume_op())) {
            PrimExpr predicate = call->args[0].as_or_throw<PrimExpr>();
            if (SideEffect(predicate) == CallEffectKind::kPure) {
              constraint_scope_.Current().Emplace(analyzer_, predicate, true);
            }
          }
        }
      }
      return SeqStmt(statements, node->loc);
    });
  }

  bool EqualIndices(const ffi::Array<PrimExpr> &first,
                    const ffi::Array<PrimExpr> &second) {
    if (first.size() != second.size()) {
      return false;
    }
    for (size_t axis = 0; axis < first.size(); ++axis) {
      if (!tl::arith::CanProve(analyzer_, first[axis] == second[axis])) {
        return false;
      }
    }
    return true;
  }

  bool MayAlias(const tirx::TensorVar &first, const tirx::TensorVar &second) {
    auto first_origin = origins_.find(first);
    auto second_origin = origins_.find(second);
    if (first_origin == origins_.end() || second_origin == origins_.end() ||
        first_origin->second.same_as(second_origin->second)) {
      return true;
    }
    return !noalias_ && !allocations_.count(first_origin->second) &&
           !allocations_.count(second_origin->second);
  }

  void Invalidate(const tirx::TensorVar &tensor,
                  const ffi::Array<PrimExpr> &indices) {
    std::erase_if(known_, [&](const KnownValue &known) {
      if (!MayAlias(tensor, known.tensor)) {
        return false;
      }
      if (tensor.same_as(known.tensor) &&
          indices.size() == known.indices.size()) {
        for (size_t axis = 0; axis < indices.size(); ++axis) {
          if (tl::arith::CanProve(analyzer_,
                                  indices[axis] != known.indices[axis])) {
            return false;
          }
        }
      }
      return true;
    });
  }

  ffi::UnchangedOr<ffi::Any> Mutate(ffi::AnyView input,
                                    ffi::InplaceMode mode) final {
    auto result = Parent::Mutate(input, mode);
    if (def_region_kind() != kTVMFFIDefRegionKindNone ||
        !input.as<PrimExpr>()) {
      return result;
    }
    auto value =
        std::move(result).ValueOrUnchanged(input).as_or_throw<PrimExpr>();
    return tl::arith::Simplify(analyzer_, value);
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const FloorDivNode *node,
                                     ffi::InplaceMode mode) final {
    PrimExpr original = ffi::GetRef<PrimExpr>(node);
    PrimExpr value = Parent::Mutate_(node, mode).ValueOrUnchanged(original);
    auto bound = tl::arith::ConstIntBoundOf(analyzer_, value);
    if (bound->min_value == bound->max_value &&
        bound->min_value != sym::ConstIntBound::kNegInf &&
        bound->max_value != sym::ConstIntBound::kPosInf) {
      return tl::MakeConst(value.ty(), bound->min_value);
    }
    return value;
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *node,
                                     ffi::InplaceMode mode) final {
    PrimExpr original = ffi::GetRef<PrimExpr>(node);
    auto load = Parent::Mutate_(node, mode)
                    .ValueOrUnchanged(original)
                    .as_or_throw<TensorLoad>();
    auto tensor = load->source.as_or_throw<tirx::TensorVar>();
    bool local_scalar = tensor.scope().name() == "local" &&
                        tensor.shape().size() == 1 && IsOne(tensor.shape()[0]);
    if (propagate_now_ || local_scalar) {
      for (auto known = known_.rbegin(); known != known_.rend(); ++known) {
        if (load->source.same_as(known->tensor) &&
            EqualIndices(load->indices, known->indices)) {
          return known->value;
        }
      }
    }
    return load;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node,
                                 ffi::InplaceMode mode) final {
    Stmt original = ffi::GetRef<Stmt>(node);
    auto store = Parent::Mutate_(node, mode)
                     .ValueOrUnchanged(original)
                     .as_or_throw<TensorStore>();
    auto tensor = store->dest.as_or_throw<tirx::TensorVar>();
    if (auto load = store->value.as<TensorLoad>()) {
      if (load.value()->source.same_as(tensor) &&
          EqualIndices(load.value()->indices, store->indices)) {
        return SeqStmt({}, store->loc);
      }
    }
    Invalidate(tensor, store->indices);
    if (SideEffect(store->value) == CallEffectKind::kPure &&
        store->value.ty().IsScalar() &&
        std::all_of(store->indices.begin(), store->indices.end(),
                    [](const PrimExpr &index) {
                      return SideEffect(index) == CallEffectKind::kPure &&
                             index.ty().IsScalar();
                    })) {
      known_.push_back({tensor, store->indices, store->value});
    }
    return store;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const BindNode *node,
                                 ffi::InplaceMode mode) final {
    Stmt original = ffi::GetRef<Stmt>(node);
    auto binding = tvm::tirx::StmtExprMutator::Mutate_(node, mode)
                       .ValueOrUnchanged(original)
                       .as_or_throw<Bind>();
    if (auto tensor = binding->var.as<tirx::TensorVar>()) {
      const auto *call = binding->value.as<CallNode>();
      if (call && call->op.same_as(tirx::alloc_tensor_op())) {
        allocations_.insert(tensor.value());
        origins_.insert_or_assign(tensor.value(), tensor.value());
      } else if (call && call->op.same_as(tirx::decl_tensor_op())) {
        const auto *pointer = call->args[0].as<CallNode>();
        if (pointer && pointer->op.same_as(tirx::tensor_data_ptr_op())) {
          if (auto source = pointer->args[0].as<Var>()) {
            if (auto origin = origins_.find(source.value());
                origin != origins_.end()) {
              origins_.insert_or_assign(tensor.value(), origin->second);
            }
          }
        }
      }
      return binding;
    }
    auto variable = binding->var.as<PrimVar>();
    auto value = binding->value.as<PrimExpr>();
    if (variable && value &&
        SideEffect(value.value()) == CallEffectKind::kPure) {
      condition_bindings_.Set(variable.value(), value.value());
      bool inline_value = value->as<IntImmNode>() ||
                          value->as<FloatImmNode>() || value->as<VarNode>() ||
                          value->ty().MatchesCode(kDLInt);
      if (config_->enable_simplify_let_inline && inline_value) {
        tl::arith::Bind(analyzer_, variable.value(), value.value());
        VarRemapSet(variable.value(), value.value());
      }
    }
    return binding;
  }

  PrimExpr Condition(const PrimExpr &condition, ffi::InplaceMode mode) {
    bool saved = propagate_now_;
    propagate_now_ |= config_->propagate_knowns_to_prove_conditional;
    PrimExpr substituted = Substitute(condition, condition_bindings_);
    PrimExpr value = Mutate(substituted, mode).ValueOrUnchanged(substituted);
    propagate_now_ = saved;
    value = tl::arith::Simplify(analyzer_, value);
    if (config_->transitively_prove_inequalities) {
      if (tl::arith::CanProve(analyzer_, value)) {
        return IntImm::Bool(true);
      }
      if (tl::arith::CanProve(analyzer_, !value)) {
        return IntImm::Bool(false);
      }
    }
    return value;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const IfNode *node,
                                 ffi::InplaceMode mode) final {
    PrimExpr condition = Condition(node->condition, mode);
    auto constant = GetConstInt(condition);
    if (constant && preserve_ws_schedule_) {
      known_.clear();
      SeqStmt then_case =
          Mutate(node->then_case, mode).ValueOrUnchanged(node->then_case);
      known_.clear();
      ffi::Optional<SeqStmt> else_case;
      if (node->else_case)
        else_case = Mutate(node->else_case.value(), mode)
                        .ValueOrUnchanged(node->else_case.value());
      known_.clear();
      return If(condition, then_case, else_case, node->loc);
    }
    if (constant) {
      ffi::Optional<SeqStmt> selected =
          *constant != 0 ? ffi::Optional<SeqStmt>(node->then_case)
                         : node->else_case;
      if (!selected) {
        return SeqStmt({}, node->loc);
      }
      auto body =
          Mutate(selected.value(), mode).ValueOrUnchanged(selected.value());
      return RegionStmt(::tvm::tl::scope_op(), {}, {}, DictAttrs(), body, {},
                        node->loc);
    }
    auto incoming = known_;
    PrimExpr predicate = condition;
    while (const auto *call = predicate.as<CallNode>()) {
      if (!call->op.same_as(prim::likely_op()))
        break;
      predicate = call->args[0].as_or_throw<PrimExpr>();
    }
    SeqStmt then_case = node->then_case;
    constraint_scope_.WithNewScope([&]() {
      if (SideEffect(condition) == CallEffectKind::kPure) {
        constraint_scope_.Current().Emplace(analyzer_, predicate);
      }
      then_case = Mutate(then_case, mode).ValueOrUnchanged(then_case);
    });
    known_ = incoming;
    ffi::Optional<SeqStmt> else_case;
    if (node->else_case) {
      constraint_scope_.WithNewScope([&]() {
        if (SideEffect(condition) == CallEffectKind::kPure) {
          constraint_scope_.Current().Emplace(
              analyzer_, tl::arith::Simplify(analyzer_, !predicate));
        }
        else_case = Mutate(node->else_case.value(), mode)
                        .ValueOrUnchanged(node->else_case.value());
      });
    }
    known_.clear();
    return If(condition, then_case, else_case, node->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *node,
                                 ffi::InplaceMode mode) final {
    if (!preserve_ws_schedule_ &&
        SideEffect(node->begin) == CallEffectKind::kPure &&
        SideEffect(node->end) == CallEffectKind::kPure &&
        SideEffect(node->step) == CallEffectKind::kPure &&
        tl::arith::CanProve(analyzer_,
                            (node->step > 0 && node->begin >= node->end) ||
                                (node->step < 0 && node->begin <= node->end))) {
      return SeqStmt({}, node->loc);
    }
    known_.clear();
    auto result =
        !preserve_ws_schedule_ && tl::arith::CanProve(analyzer_, node->step > 0)
            ? Parent::Mutate_(node, mode)
            : tvm::tirx::StmtExprMutator::Mutate_(node, mode);
    known_.clear();
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const WhileNode *node,
                                 ffi::InplaceMode mode) final {
    known_.clear();
    auto result = Parent::Mutate_(node, mode);
    known_.clear();
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    const auto *operation = node->op.as<OpNode>();
    bool transparent = node->op.same_as(::tvm::tl::scope_op()) ||
                       (operation && operation->name == "tl.scope");
    if (!transparent) {
      known_.clear();
    }
    Stmt result =
        Parent::Mutate_(node, mode).ValueOrUnchanged(ffi::GetRef<Stmt>(node));
    if (transparent) {
      std::unordered_set<const VarNode *> scoped_definitions;
      ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
          result,
          [&](const Var &variable,
              TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::WalkResult> {
            if (kind != kTVMFFIDefRegionKindNone)
              scoped_definitions.insert(variable.get());
            return ffi::WalkResult::Advance();
          });
      std::erase_if(known_, [&](const KnownValue &known) {
        ffi::Array<Expr> operands{known.tensor, known.value};
        for (const PrimExpr &index : known.indices)
          operands.push_back(index);
        auto variables = UsedVariables(operands);
        return std::any_of(variables.begin(), variables.end(),
                           [&](const VarNode *variable) {
                             return scoped_definitions.count(variable);
                           });
      });
    } else {
      known_.clear();
    }
    return result;
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallNode *node,
                                 ffi::InplaceMode mode) final {
    if (node->op.same_as(prim::if_then_else_op())) {
      PrimExpr condition =
          Condition(node->args[0].as_or_throw<PrimExpr>(), mode);
      if (auto constant = GetConstInt(condition)) {
        Expr branch = node->args[*constant != 0 ? 1 : 2];
        return Mutate(branch, mode).ValueOrUnchanged(branch);
      }
    }
    auto result = Parent::Mutate_(node, mode);
    static auto effects = Op::GetAttrMap(tvm::op_attr::kCallEffectKind);
    auto effect = effects.get(node->op, CallEffectKind::kOpaque);
    if (effect >= CallEffectKind::kUpdateState &&
        effect != CallEffectKind::kEmbedInfo) {
      known_.clear();
    }
    return result;
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const AndNode *node,
                                     ffi::InplaceMode mode) final {
    PrimExpr left = Mutate(node->a, mode).ValueOrUnchanged(node->a);
    PrimExpr right = node->b;
    if (config_->apply_constraints_to_boolean_branches &&
        SideEffect(left) == CallEffectKind::kPure &&
        SideEffect(right) == CallEffectKind::kPure) {
      With<tl::arith::ConstraintContext> constraint(analyzer_, left);
      right = Mutate(right, mode).ValueOrUnchanged(right);
      right = SimplifyBoolean(right);
    } else {
      right = Mutate(right, mode).ValueOrUnchanged(right);
    }
    if (config_->apply_constraints_to_boolean_branches &&
        SideEffect(left) == CallEffectKind::kPure &&
        SideEffect(right) == CallEffectKind::kPure) {
      With<tl::arith::ConstraintContext> constraint(analyzer_, right);
      left = SimplifyBoolean(left);
    }
    return left && right;
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const OrNode *node,
                                     ffi::InplaceMode mode) final {
    PrimExpr left = Mutate(node->a, mode).ValueOrUnchanged(node->a);
    PrimExpr right = node->b;
    bool pure = SideEffect(left) == CallEffectKind::kPure &&
                SideEffect(right) == CallEffectKind::kPure;
    if (config_->apply_constraints_to_boolean_branches && pure) {
      With<tl::arith::ConstraintContext> constraint(
          analyzer_, tl::arith::Simplify(analyzer_, !left));
      right = Mutate(right, mode).ValueOrUnchanged(right);
      right = SimplifyBoolean(right);
    } else {
      right = Mutate(right, mode).ValueOrUnchanged(right);
    }
    if (config_->apply_constraints_to_boolean_branches && pure) {
      With<tl::arith::ConstraintContext> constraint(
          analyzer_, tl::arith::Simplify(analyzer_, !right));
      left = SimplifyBoolean(left);
    }
    bool stable_reads = !volatile_storage_ &&
                        SideEffect(left) <= CallEffectKind::kReadState &&
                        SideEffect(right) <= CallEffectKind::kReadState;
    if (config_->convert_boolean_to_and_of_ors && (pure || stable_reads)) {
      if (const auto *conjunction = left.as<AndNode>()) {
        return (conjunction->a || right) && (conjunction->b || right);
      }
      if (const auto *conjunction = right.as<AndNode>()) {
        return (left || conjunction->a) && (left || conjunction->b);
      }
    }
    return left || right;
  }

  SimplifyConfig config_;
  bool propagate_now_;
  bool noalias_;
  bool volatile_storage_;
  bool preserve_ws_schedule_{false};
  ffi::Map<PrimVar, PrimExpr> condition_bindings_;
  std::vector<KnownValue> known_;
  std::unordered_map<Var, Var> origins_;
  std::unordered_set<Var> allocations_;
};

} // namespace

namespace transform {

tvm::transform::Pass Simplify(bool simplify_arguments = true) {
  auto pass = [simplify_arguments](tirx::Function function, IRModule,
                                   tvm::transform::PassContext context)
      -> ffi::Optional<tirx::Function> {
    if (!function->body) {
      return function;
    }
    sym::Analyzer analyzer;
    auto config = context->GetConfigOrDefault<SimplifyConfig>("tl.Simplify");
    auto simplifier =
        ffi::make_object<StmtSimplifier>(analyzer, config, function);
    SeqStmt body = function->body.value();
    function.CopyOnWrite()->body =
        simplifier->Mutate(body).ValueOrUnchanged(body);
    while (true) {
      auto remover = ffi::make_object<UnusedBindRemover>(
          UsedVariables(function), ContainsVolatileStorage(function));
      body = function->body.value();
      function.CopyOnWrite()->body =
          remover->Mutate(body).ValueOrUnchanged(body);
      if (!remover->changed) {
        break;
      }
    }
    if (simplify_arguments) {
      auto used = UsedVariables(function);
      ffi::Array<Var> parameters;
      std::vector<int> new_indices(function->params.size(), -1);
      size_t old_index = 0;
      for (const Var &parameter : function->params) {
        if (!parameter.as<tirx::TensorVar>() || used.count(parameter.get())) {
          new_indices[old_index] = static_cast<int>(parameters.size());
          parameters.push_back(parameter);
        }
        ++old_index;
      }
      for (const char *key :
           {"tl.readonly_param_indices", "tl.non_restrict_param_indices"}) {
        if (auto indices = function->GetAttr<ffi::Array<IntImm>>(key)) {
          ffi::Array<IntImm> remapped;
          for (const IntImm &index : indices.value()) {
            TVM_FFI_CHECK(index->value >= 0 &&
                              index->value <
                                  static_cast<int64_t>(new_indices.size()),
                          ValueError)
                << key << " contains an invalid parameter index";
            if (int replacement =
                    new_indices[index->value.as<int64_t>().value()];
                replacement >= 0) {
              remapped.push_back(IntImm::Int32(replacement));
            }
          }
          function = WithAttr(std::move(function), key, remapped);
        }
      }
      function.CopyOnWrite()->params = parameters;
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.Simplify");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  SimplifyConfigNode::RegisterReflection();
  tvm::transform::PassContext::RegisterConfigOption<SimplifyConfig>(
      "tl.Simplify");
  ffi::reflection::GlobalDef().def("tl.transform.Simplify", Simplify);
}

} // namespace transform
} // namespace tvm::tl
