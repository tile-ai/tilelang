/*!
 * \file unroll_loop.cc
 * \brief Expand native loops while retaining TileLang's explicit-unroll policy.
 */

#include "ir/annotations.h"
#include "ir/stmt_functor.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <unordered_set>

#include "../ir/analyzer.h"
#include "../ir/annotations.h"
#include "../op/builtin.h"
#include "ir/loop_math.h"

namespace tvm::tl {

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

namespace {
using namespace tirx;

// Native Break and Continue target the nearest For/While. A control statement
// in a nested loop must not prevent expansion of its enclosing loop.
class TargetedLoopControl : public StmtExprVisitor {
public:
  bool found{false};
  ffi::Optional<VisitInterrupt> Visit_(const BreakObj *) final {
    found = true;
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const ContinueObj *) final {
    found = true;
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const ForObj *) final {
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const WhileObj *) final {
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    if (op->op.same_as(tl::loop_break()))
      found = true;
    return StmtExprVisitor::Visit_(op);
  }
};

int64_t AnnotationInteger(ffi::AnyView value) {
  if (const auto *literal = value.as<IntImmObj>())
    return literal->value.as<int64_t>().value();
  if (auto boolean = value.as<bool>())
    return *boolean;
  return value.cast<int64_t>();
}

class LoopUnroller : public LayoutMapMutator {
public:
  using LayoutMapMutator::Mutate;
  using LayoutMapMutator::Mutate_;

  explicit LoopUnroller(UnrollLoopConfig config)
      : config_(std::move(config)), max_step_(config_->auto_max_step),
        explicit_(config_->explicit_unroll) {}

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    if (op->op->name != "tl.attr")
      return LayoutMapMutator::Mutate_(op, mode);
    auto fields = ReadAnnotations(op);
    auto key = fields.at("key").cast<ffi::String>();
    if (key != "pragma_auto_unroll_max_step" && key != "pragma_unroll_explicit")
      return LayoutMapMutator::Mutate_(op, mode);
    int64_t previous_max = max_step_;
    bool previous_explicit = explicit_;
    auto value = AnnotationInteger(fields.at("value"));
    if (key == "pragma_auto_unroll_max_step")
      max_step_ = value;
    else
      explicit_ = value != 0;
    SeqStmt body = Mutate(op->body).ValueOrUnchanged(op->body);
    max_step_ = previous_max;
    explicit_ = previous_explicit;
    TVM_FFI_CHECK(op->body_params.empty() && op->result_vars.empty(),
                  ValueError)
        << "An unroll policy region cannot define values";
    return body;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    // A For annotation controls this loop only. A surrounding pragma region
    // controls its lexical body. In particular, an explicit outer unroll must
    // not turn a nested dynamic T.unroll into a forced constant expansion.
    auto rewritten = LayoutMapMutator::Mutate_(op, mode)
                         .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                         .as_or_throw<For>();
    op = rewritten.get();
    int64_t previous_max = max_step_;
    bool previous_explicit = explicit_;
    for (const auto &[key, value] : op->annotations) {
      if (key == "pragma_auto_unroll_max_step" ||
          key == "tirx.auto_unroll_max_step")
        max_step_ = AnnotationInteger(value);
      else if (key == "pragma_unroll_explicit" || key == "tirx.unroll_explicit")
        explicit_ = AnnotationInteger(value) != 0;
    }
    auto result = RewriteLoop(std::move(rewritten));
    max_step_ = previous_max;
    explicit_ = previous_explicit;
    return result;
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *op,
                                     InplaceMode mode) final {
    MarkLocal(op->source, op->indices);
    return LayoutMapMutator::Mutate_(op, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreObj *op,
                                 InplaceMode mode) final {
    AddSteps(1);
    MarkLocal(op->dest, op->indices);
    return LayoutMapMutator::Mutate_(op, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const EvaluateObj *op,
                                 InplaceMode mode) final {
    AddSteps(1);
    return LayoutMapMutator::Mutate_(op, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    AddSteps(1);
    return LayoutMapMutator::Mutate_(op, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const BreakObj *, InplaceMode) final {
    AddSteps(1);
    return ffi::Unchanged();
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ContinueObj *, InplaceMode) final {
    AddSteps(1);
    return ffi::Unchanged();
  }

  ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode) final {
    ffi::Array<Stmt> statements;
    bool changed = false;
    for (const Stmt &statement : op->seq) {
      int64_t prior_steps = step_count_;
      int prior_normal = normal_depth_, prior_unroll = unroll_depth_;
      step_count_ = normal_depth_ = unroll_depth_ = 0;
      auto result = Mutate(statement);
      changed |= !result.UnchangedOrSameAs(statement);
      Stmt rewritten = std::move(result).ValueOrUnchanged(statement);
      // SeqStmt groups statements; For/If/Region define scopes. Expose the
      // expanded scheduling units while keeping their definitions independent.
      if (auto nested = rewritten.as<SeqStmt>()) {
        for (const Stmt &child : nested.value()->seq)
          statements.push_back(child);
        changed = true;
      } else {
        statements.push_back(rewritten);
      }
      AddSteps(prior_steps);
      normal_depth_ = std::max(prior_normal, normal_depth_);
      unroll_depth_ = std::max(prior_unroll, unroll_depth_);
    }
    return changed ? ffi::UnchangedOr<Stmt>(SeqStmt(statements, op->loc))
                   : ffi::UnchangedOr<Stmt>(ffi::Unchanged());
  }

private:
  void AddSteps(int64_t amount) {
    constexpr int64_t limit = std::numeric_limits<int>::max();
    step_count_ = std::min(limit, step_count_ + amount);
  }

  void MarkLocal(const Expr &value, const ffi::Array<PrimExpr> &indices) {
    if (!config_->unroll_local_access)
      return;
    auto tensor = value.as<TensorVar>();
    if (!tensor)
      return;
    auto scope = tensor.value().scope();
    if (scope != runtime::MemoryScope::Get("local") &&
        scope != runtime::MemoryScope::Get("warp"))
      return;
    for (const PrimExpr &index : indices)
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
          index, [&](const VarObj *var) -> ffi::Expected<ffi::WalkResult> {
            local_indices_.insert(var);
            return ffi::WalkResult::Advance();
          });
  }

  Stmt Expand(const ForObj *loop, int count) {
    auto control = ffi::make_object<TargetedLoopControl>();
    control->Visit(loop->body);
    TVM_FFI_CHECK(!control->found, ValueError)
        << "[TileLang Semantic Check] A loop containing T.loop_break() or "
           "continue that targets that loop cannot be fully expanded. Use "
           "T.unroll(..., explicit=False) or T.serial(...) to preserve the "
           "loop.";
    ffi::Array<Stmt> statements;
    for (int index = 0; index < count; ++index) {
      // Only local definitions are renewed; free parameters keep their
      // identity. Native hooks rebuild types and uses, including
      // shape/layout/region vars.
      auto copy = tl::RenewDef(tirx::Function({}, loop->body));
      auto substitution = ffi::make_object<LayoutMapMutator>();
      PrimExpr iteration = UnrolledLoopValue(loop, index, analyzer_);
      substitution->VarRemapSet(loop->var, analyzer_->Simplify(iteration));
      SeqStmt body =
          substitution->Mutate(copy->body).ValueOrUnchanged(copy->body);
      for (const Stmt &statement : body->seq)
        statements.push_back(statement);
    }
    return SeqStmt(statements, loop->loc);
  }

  ffi::UnchangedOr<Stmt> RewriteLoop(For rewritten) {
    const auto *op = rewritten.get();
    int count = ConstantUnrollTripCount(op, analyzer_);
    bool automatic = op->kind == ForKind::kDefault && count >= 0 &&
                     normal_depth_ == 0 &&
                     unroll_depth_ <= config_->auto_max_depth;
    automatic =
        automatic && (static_cast<int64_t>(count) * step_count_ <= max_step_ ||
                      count <= config_->auto_max_extent);
    bool unroll = automatic || op->kind == ForKind::kUnrolled ||
                  (count > 0 && config_->unroll_local_access &&
                   local_indices_.count(op->var.get()));
    if (op->kind == ForKind::kUnrolled && explicit_)
      TVM_FFI_CHECK_GE(count, 0, ValueError)
          << "Cannot unroll non-constant loop or an iteration that overflows "
             "its index type";
    if (unroll) {
      step_count_ = count < 0
                        ? std::numeric_limits<int>::max()
                        : std::min<int64_t>(std::numeric_limits<int>::max(),
                                            count * step_count_);
      ++unroll_depth_;
    } else {
      ++normal_depth_;
    }
    if ((unroll && explicit_ && count >= 0) ||
        (config_->auto_max_extent == 1 && count >= 0 && count <= 1))
      return Expand(op, count);
    if (unroll && op->kind != ForKind::kUnrolled)
      rewritten.CopyOnWrite()->kind = ForKind::kUnrolled;
    return rewritten;
  }

  UnrollLoopConfig config_;
  int64_t max_step_;
  bool explicit_;
  int64_t step_count_{0};
  int normal_depth_{0}, unroll_depth_{0};
  std::unordered_set<const VarObj *> local_indices_;
  sym::Analyzer analyzer_{CreateTileLangAnalyzer()};
};
} // namespace

namespace transform {
tvm::transform::Pass UnrollLoop() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext context) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto unroller = ffi::make_object<LoopUnroller>(
        context->GetConfigOrDefault<UnrollLoopConfig>("tl.UnrollLoop"));
    auto body = function->body;
    function.CopyOnWrite()->body =
        unroller->Mutate(body).ValueOrUnchanged(body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.UnrollLoop");
}
} // namespace transform

TVM_FFI_STATIC_INIT_BLOCK() {
  UnrollLoopConfigNode::RegisterReflection();
  tvm::transform::PassContext::RegisterConfigOption<UnrollLoopConfig>(
      "tl.UnrollLoop");
  ffi::reflection::GlobalDef().def("tl.transform.UnrollLoop",
                                   transform::UnrollLoop);
}

} // namespace tvm::tl
