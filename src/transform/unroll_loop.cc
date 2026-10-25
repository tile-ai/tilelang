/*!
 *  Loop unrolling as in Halide pipeline.
 * \file unroll_loop.cc
 */
// Unrolls the loop as in Halide pipeline.
#include "compat/expr.h"
#include "compat/visit.h"
#include "op/builtin.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace tvm::tl {

namespace {

struct UnrollLoopConfigNode : public ffi::Object {
  int auto_max_step;
  int auto_max_depth;
  int auto_max_extent;
  bool explicit_unroll;
  bool unroll_local_access;

  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<UnrollLoopConfigNode>()
        .def_ro("auto_max_step", &UnrollLoopConfigNode::auto_max_step,
                refl::DefaultValue(0))
        .def_ro("auto_max_depth", &UnrollLoopConfigNode::auto_max_depth,
                refl::DefaultValue(8))
        .def_ro("auto_max_extent", &UnrollLoopConfigNode::auto_max_extent,
                refl::DefaultValue(0))
        .def_ro("explicit_unroll", &UnrollLoopConfigNode::explicit_unroll,
                refl::DefaultValue(false))
        .def_ro("unroll_local_access",
                &UnrollLoopConfigNode::unroll_local_access,
                refl::DefaultValue(false));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.transform.UnrollLoopConfig",
                                    UnrollLoopConfigNode, ffi::Object);
};

class UnrollLoopConfig : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(UnrollLoopConfig,
                                                ffi::ObjectRef,
                                                UnrollLoopConfigNode);
};

class TargetedLoopControlDetector : public tirx::StmtExprVisitor {
public:
  bool found{false};

private:
  ffi::Optional<VisitInterrupt> Visit_(const ForNode *) final {
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const WhileNode *) final {
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const BreakNode *) final {
    found = true;
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const ContinueNode *) final {
    found = true;
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const CallNode *node) final {
    if (node->op.same_as(loop_break())) {
      found = true;
    }
    return StmtExprVisitor::Visit_(node);
  }
};

class LoopUnroller : public tirx::StmtExprMutator {
public:
  explicit LoopUnroller(UnrollLoopConfig config)
      : config_(std::move(config)), auto_max_step_(config_->auto_max_step),
        explicit_unroll_(config_->explicit_unroll) {
    TVM_FFI_CHECK_GE(config_->auto_max_step, 0, ValueError);
    TVM_FFI_CHECK_GE(config_->auto_max_depth, 0, ValueError);
    TVM_FFI_CHECK_GE(config_->auto_max_extent, 0, ValueError);
  }

private:
  void SetPolicy(const ffi::Map<ffi::String, ffi::Any> &attributes) {
    if (auto value = attributes.Get("pragma_auto_unroll_max_step")) {
      auto limit = GetIntAnnotation(value.value());
      TVM_FFI_CHECK(limit && *limit >= 0 &&
                        *limit <= std::numeric_limits<int>::max(),
                    ValueError)
          << "pragma_auto_unroll_max_step must be a nonnegative integer";
      auto_max_step_ = *limit;
    }
    if (auto value = attributes.Get("pragma_unroll_explicit")) {
      auto explicit_value = GetIntAnnotation(value.value());
      TVM_FFI_CHECK(explicit_value.has_value(), ValueError)
          << "pragma_unroll_explicit must be a constant boolean";
      explicit_unroll_ = *explicit_value != 0;
    }
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    int saved_limit = auto_max_step_;
    bool saved_explicit = explicit_unroll_;
    SetPolicy(node->attrs->dict);
    Stmt original = ffi::GetRef<Stmt>(node);
    auto region = StmtExprMutator::Mutate_(node, mode)
                      .ValueOrUnchanged(original)
                      .as_or_throw<RegionStmt>();
    auto_max_step_ = saved_limit;
    explicit_unroll_ = saved_explicit;
    auto attributes = region->attrs->dict;
    attributes.erase("pragma_auto_unroll_max_step");
    attributes.erase("pragma_unroll_explicit");
    region.CopyOnWrite()->attrs = DictAttrs(attributes);
    return region;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *node,
                                 ffi::InplaceMode mode) final {
    int saved_limit = auto_max_step_;
    bool saved_explicit = explicit_unroll_;
    SetPolicy(node->annotations);
    Stmt original = ffi::GetRef<Stmt>(node);
    auto loop = StmtExprMutator::Mutate_(node, mode)
                    .ValueOrUnchanged(original)
                    .as_or_throw<For>();
    int max_step = auto_max_step_;
    bool expand = explicit_unroll_;
    auto_max_step_ = saved_limit;
    explicit_unroll_ = saved_explicit;
    auto control = ffi::make_object<TargetedLoopControlDetector>();
    control->Visit(loop->body);
    if (control->found && loop->kind != ForKind::kUnrolled) {
      ++normal_loop_depth_;
      return loop;
    }
    int trip_count = GetTripCount(loop.get());
    bool eligible =
        loop->kind == ForKind::kDefault || loop->kind == ForKind::kUnrolled;
    bool automatic = loop->kind == ForKind::kDefault && trip_count >= 0 &&
                     normal_loop_depth_ == 0 &&
                     unroll_depth_ <= config_->auto_max_depth &&
                     (step_count_ <= max_step / std::max(trip_count, 1) ||
                      trip_count <= config_->auto_max_extent);
    if (loop->kind == ForKind::kUnrolled) {
      TVM_FFI_CHECK(!expand || trip_count >= 0, ValueError)
          << "Cannot explicitly unroll a loop with a non-constant "
             "positive-step trip count";
      automatic = true;
    }
    if (eligible && trip_count > 0 && config_->unroll_local_access &&
        local_indices_.count(loop->var)) {
      automatic = true;
    }
    if (automatic) {
      step_count_ = std::min<int64_t>(std::numeric_limits<int>::max(),
                                      step_count_ * std::max(trip_count, 1));
      ++unroll_depth_;
    } else {
      ++normal_loop_depth_;
    }
    if (eligible &&
        ((automatic && expand) || (trip_count >= 0 && trip_count <= 1 &&
                                   config_->auto_max_extent == 1))) {
      return Unroll(loop.get(), trip_count);
    }
    if (automatic && loop->kind != ForKind::kUnrolled) {
      loop.CopyOnWrite()->kind = ForKind::kUnrolled;
    }
    return loop;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const WhileNode *node,
                                 ffi::InplaceMode mode) final {
    auto result = StmtExprMutator::Mutate_(node, mode);
    ++normal_loop_depth_;
    return result;
  }

  void MarkLocalIndices(const Expr &source,
                        const ffi::Array<PrimExpr> &indices) {
    if (!config_->unroll_local_access) {
      return;
    }
    auto tensor = source.as<tirx::TensorVar>();
    if (!tensor) {
      return;
    }
    std::string scope = tensor->scope().name();
    if (scope != "local" && scope != "warp" && scope.rfind("local.", 0) != 0) {
      return;
    }
    for (const PrimExpr &index : indices) {
      PostOrderVisit(index, [&](const ffi::ObjectRef &object) {
        if (const auto *variable = object.as<VarNode>()) {
          local_indices_.insert(ffi::GetRef<Var>(variable));
        }
      });
    }
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *node,
                                     ffi::InplaceMode mode) final {
    MarkLocalIndices(node->source, node->indices);
    return StmtExprMutator::Mutate_(node, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node,
                                 ffi::InplaceMode mode) final {
    ++step_count_;
    MarkLocalIndices(node->dest, node->indices);
    return StmtExprMutator::Mutate_(node, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const EvaluateNode *node,
                                 ffi::InplaceMode mode) final {
    ++step_count_;
    return StmtExprMutator::Mutate_(node, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const BindNode *node,
                                 ffi::InplaceMode mode) final {
    ++step_count_;
    return StmtExprMutator::Mutate_(node, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtNode *node,
                                 ffi::InplaceMode mode) final {
    ffi::Array<Stmt> statements;
    bool changed = false;
    for (const Stmt &statement : node->seq) {
      int64_t saved_steps = step_count_;
      int saved_normal_depth = normal_loop_depth_;
      int saved_unroll_depth = unroll_depth_;
      step_count_ = normal_loop_depth_ = unroll_depth_ = 0;
      auto result = Mutate(statement, mode);
      changed |= !result.UnchangedOrSameAs(statement);
      statements.push_back(std::move(result).ValueOrUnchanged(statement));
      step_count_ = std::min<int64_t>(std::numeric_limits<int>::max(),
                                      step_count_ + saved_steps);
      normal_loop_depth_ = std::max(normal_loop_depth_, saved_normal_depth);
      unroll_depth_ = std::max(unroll_depth_, saved_unroll_depth);
    }
    if (!changed) {
      return ffi::Unchanged();
    }
    return SeqStmt(statements, node->loc);
  }

  int GetTripCount(const ForNode *node) {
    auto step = GetConstInt(analyzer_->Simplify(node->step));
    if (!step || *step <= 0 ||
        SideEffect(node->begin) != CallEffectKind::kPure ||
        SideEffect(node->end) != CallEffectKind::kPure) {
      return -1;
    }
    auto extent = GetConstInt(analyzer_->Simplify(node->end - node->begin));
    if (!extent) {
      return -1;
    }
    if (*extent <= 0) {
      return 0;
    }
    int64_t count = 1 + (*extent - 1) / *step;
    return count <= std::numeric_limits<int>::max() ? static_cast<int>(count)
                                                    : -1;
  }

  Stmt Unroll(const ForNode *node, int trip_count) {
    auto detector = ffi::make_object<TargetedLoopControlDetector>();
    detector->Visit(node->body);
    TVM_FFI_CHECK(!detector->found, ValueError)
        << "A loop containing break or continue targeting that loop cannot be "
           "fully expanded. "
           "Use T.unroll(..., explicit=False) or T.serial(...) to preserve the "
           "loop.";
    TVM_FFI_CHECK_GE(trip_count, 0, ValueError);
    ffi::Array<Stmt> expanded;
    for (int iteration = 0; iteration < trip_count; ++iteration) {
      PrimExpr value = analyzer_->Simplify(
          node->begin + MakeConst(node->var.ty(), iteration) * node->step);
      Stmt body = Substitute(node->body,
                             ffi::Map<PrimVar, PrimExpr>{{node->var, value}});
      auto renewed = tirx::RenewDef(tirx::Function({}, SeqStmt(body)));
      expanded.push_back(RegionStmt(tirx::scope_op(), {}, {}, DictAttrs(),
                                    renewed->body.value(), {}, node->loc));
    }
    return SeqStmt(expanded, node->loc);
  }

  UnrollLoopConfig config_;
  int auto_max_step_;
  bool explicit_unroll_;
  int normal_loop_depth_{0};
  int unroll_depth_{0};
  int64_t step_count_{0};
  std::unordered_set<Var> local_indices_;
  sym::Analyzer analyzer_;
};

} // namespace

namespace transform {

tvm::transform::Pass UnrollLoop() {
  auto pass =
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext context) -> ffi::Optional<tirx::Function> {
    auto config =
        context->GetConfigOrDefault<UnrollLoopConfig>("tl.UnrollLoop");
    if (function->body) {
      auto unroller = ffi::make_object<LoopUnroller>(config);
      SeqStmt body = function->body.value();
      function.CopyOnWrite()->body =
          unroller->Mutate(body).ValueOrUnchanged(body);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.UnrollLoop");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  UnrollLoopConfigNode::RegisterReflection();
  tvm::transform::PassContext::RegisterConfigOption<UnrollLoopConfig>(
      "tl.UnrollLoop");
  ffi::reflection::GlobalDef().def("tl.transform.UnrollLoop", UnrollLoop);
}

} // namespace transform
} // namespace tvm::tl
