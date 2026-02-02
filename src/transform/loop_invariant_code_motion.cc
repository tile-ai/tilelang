/*!
 * \file loop_invariant_code_motion.cc
 * \brief Loop Invariant Code Motion (LICM) with Common Subexpression Extraction
 *
 * This pass performs two optimizations in a unified manner:
 *
 * 1. **LetStmt Hoisting**: Existing LetStmt nodes whose values are
 * loop-invariant are moved outside the loop.
 *
 * 2. **Invariant Subexpression Extraction**: Loop-invariant subexpressions that
 *    appear multiple times within the loop body are extracted into new
 * variables and hoisted outside the loop.
 *
 * Example transformation:
 *
 *   // Before
 *   for (i = 0; i < n; i++) {
 *       A[(threadIdx.x >> 2) * 32 + i] = ...
 *       B[(threadIdx.x >> 2) * 32 + j] = ...
 *   }
 *
 *   // After
 *   cse_var = (threadIdx.x >> 2) * 32
 *   for (i = 0; i < n; i++) {
 *       A[cse_var + i] = ...
 *       B[cse_var + j] = ...
 *   }
 *
 * Design principles:
 *   - Two-phase approach: Analysis then Transformation
 *   - Conservative safety checks to preserve semantics
 *   - Complexity-aware extraction (larger expressions first)
 *   - Configurable minimum occurrence threshold
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "common/buffer_analysis.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;
using namespace ffi;

/*!
 * \brief Configuration node for Loop Invariant Code Motion pass
 */
struct LICMConfigNode : public AttrsNodeReflAdapter<LICMConfigNode> {
  /*! \brief Minimum occurrences for CSE-style extraction */
  int min_occurrences_for_cse{};
  /*! \brief Minimum complexity for CSE extraction */
  int min_complexity_for_cse{};
  /*! \brief Minimum complexity for LICM-style extraction (even if appears once)
   */
  int min_complexity_for_licm{};

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LICMConfigNode>()
        .def_ro(
            "min_occurrences_for_cse", &LICMConfigNode::min_occurrences_for_cse,
            "Minimum number of occurrences required for CSE-style extraction. "
            "Expressions appearing >= this many times will be extracted.",
            refl::DefaultValue(2))
        .def_ro("min_complexity_for_cse",
                &LICMConfigNode::min_complexity_for_cse,
                "Minimum expression complexity (node count) required for CSE "
                "extraction. "
                "Only expressions with complexity >= this threshold will be "
                "considered.",
                refl::DefaultValue(2))
        .def_ro(
            "min_complexity_for_licm", &LICMConfigNode::min_complexity_for_licm,
            "Minimum complexity for LICM-style extraction (even if appears "
            "once). "
            "Expressions with complexity >= this threshold will be hoisted "
            "regardless "
            "of occurrence count. Handles cases like `blockIdx.y * 131072`.",
            refl::DefaultValue(3));
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.transform.LICMConfig", LICMConfigNode,
                                    BaseAttrsNode);
};

/*!
 * \brief Managed reference to LICMConfigNode
 */
class LICMConfig : public Attrs {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LICMConfig, Attrs,
                                                LICMConfigNode);
};

TVM_FFI_STATIC_INIT_BLOCK() { LICMConfigNode::RegisterReflection(); }

TVM_REGISTER_PASS_CONFIG_OPTION("tl.LoopInvariantCodeMotion", LICMConfig);

/*! \brief Calculate the complexity of an expression (number of nodes) */
class ExprComplexityCalculator : public ExprVisitor {
public:
  size_t complexity = 0;

  static size_t Calculate(const PrimExpr &expr) {
    ExprComplexityCalculator calc;
    calc.VisitExpr(expr);
    return calc.complexity;
  }

protected:
  void VisitExpr(const PrimExpr &expr) final {
    complexity++;
    ExprVisitor::VisitExpr(expr);
  }
};

/*!
 * \brief Determine if an expression is loop-invariant
 *
 * An expression is loop-invariant if:
 *   1. It does not reference the loop variable
 *   2. It does not read buffers that are written within the loop
 *   3. It has no side effects (pure computation)
 *   4. It does not contain function calls (conservative)
 */
class LoopInvarianceChecker {
public:
  LoopInvarianceChecker(const Var &loop_var,
                        const std::unordered_set<const VarNode *> &written)
      : loop_var_(loop_var), written_buffers_(written) {}

  bool IsInvariant(const PrimExpr &expr) const {
    // Check 1: Must not use loop variable
    if (UsesVar(expr,
                [this](const VarNode *v) { return v == loop_var_.get(); })) {
      return false;
    }

    // Check 2: Must not read written buffers
    if (BufferReadChecker::Check(expr, written_buffers_)) {
      return false;
    }

    // Check 3: Must be pure (no side effects beyond reading)
    if (SideEffect(expr) > CallEffectKind::kReadState) {
      return false;
    }

    return true;
  }

private:
  const Var &loop_var_;
  const std::unordered_set<const VarNode *> &written_buffers_;
};

/*!
 * \brief Check if a CallNode is a pure built-in operation.
 * Pure operations like shift_right, bitwise_and, etc. are safe to extract.
 */
bool IsPureCall(const CallNode *op) {
  if (!op->op.as<OpNode>()) {
    return false;
  }
  // Use SideEffect analysis - pure calls have effect <= kPure
  return SideEffect(ffi::GetRef<PrimExpr>(op)) <= CallEffectKind::kPure;
}

/*!
 * \brief Predicate: Is this expression eligible for extraction?
 *
 * We exclude:
 *   - Constants and variables (too simple)
 *   - BufferLoad (may have aliasing issues)
 *   - Impure function calls (may have side effects)
 *   - Ramp/Broadcast (TVM internal constraints)
 */
bool IsEligibleForExtraction(const PrimExpr &expr) {
  // Exclude trivial expressions
  if (expr.as<IntImmNode>() || expr.as<FloatImmNode>() ||
      expr.as<StringImmNode>() || expr.as<VarNode>()) {
    return false;
  }

  // Exclude BufferLoad (aliasing issues)
  if (expr.as<BufferLoadNode>()) {
    return false;
  }

  // Exclude Ramp/Broadcast (TVM internal constraints)
  if (expr.as<RampNode>() || expr.as<BroadcastNode>()) {
    return false;
  }

  // For CallNode, only exclude impure calls
  if (const auto *call = expr.as<CallNode>()) {
    if (!IsPureCall(call)) {
      return false;
    }
  }

  return true;
}

/*!
 * \brief Check if expression contains any forbidden subexpressions
 *
 * Forbidden: BufferLoad, impure CallNode
 * Allowed: pure CallNode (shift_right, bitwise_and, etc.)
 */
class ForbiddenExprChecker : public ExprVisitor {
public:
  bool has_forbidden = false;

  static bool Check(const PrimExpr &expr) {
    ForbiddenExprChecker checker;
    checker(expr);
    return checker.has_forbidden;
  }

protected:
  void VisitExpr_(const CallNode *op) final {
    // Only forbid impure calls
    if (!IsPureCall(op)) {
      has_forbidden = true;
      return;
    }
    // Continue checking children for pure calls
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final { has_forbidden = true; }
};

/*!
 * \brief Collect all subexpressions and their occurrence counts
 *
 * Uses structural hashing for expression comparison.
 */
class SubexpressionCollector : public StmtExprVisitor {
public:
  // Map from expression to occurrence count
  // Using StructuralHash and ExprDeepEqual for proper expression comparison
  std::unordered_map<PrimExpr, size_t, StructuralHash, ExprDeepEqual>
      expr_counts;

  static std::unordered_map<PrimExpr, size_t, StructuralHash, ExprDeepEqual>
  Collect(const Stmt &stmt) {
    SubexpressionCollector collector;
    collector(stmt);
    return std::move(collector.expr_counts);
  }

protected:
  void VisitExpr(const PrimExpr &expr) final {
    // Only count eligible expressions that don't contain forbidden nodes
    if (IsEligibleForExtraction(expr) && !ForbiddenExprChecker::Check(expr)) {
      expr_counts[expr]++;
    }
    // Always recurse to find subexpressions
    StmtExprVisitor::VisitExpr(expr);
  }

  // Don't descend into nested loops
  void VisitStmt_(const ForNode *op) final {
    // Only visit loop bounds, not body (nested loop has its own scope)
    VisitExpr(op->min);
    VisitExpr(op->extent);
  }
};

/*! \brief Replace all occurrences of a target expression with a variable */
class ExpressionReplacer : public StmtExprMutator {
public:
  static Stmt Replace(const Stmt &stmt, const PrimExpr &target,
                      const Var &replacement) {
    ExpressionReplacer replacer(target, replacement);
    return replacer(stmt);
  }

private:
  const PrimExpr &target_;
  const Var &replacement_;
  ExprDeepEqual expr_equal_;

  ExpressionReplacer(const PrimExpr &target, const Var &replacement)
      : target_(target), replacement_(replacement) {}

  PrimExpr VisitExpr(const PrimExpr &expr) final {
    // Check if this expression matches the target
    if (expr_equal_(expr, target_)) {
      return replacement_;
    }
    // Otherwise, recurse
    return StmtExprMutator::VisitExpr(expr);
  }
};

/*! \brief Collect LetStmt nodes within a loop body (not in nested loops) */
class LetStmtCollector : public StmtVisitor {
public:
  std::vector<const LetStmtNode *> let_stmts;
  std::unordered_map<const VarNode *, const LetStmtNode *> var_to_let;

  static std::pair<std::vector<const LetStmtNode *>,
                   std::unordered_map<const VarNode *, const LetStmtNode *>>
  Collect(const Stmt &stmt) {
    LetStmtCollector collector;
    collector(stmt);
    return {std::move(collector.let_stmts), std::move(collector.var_to_let)};
  }

protected:
  void VisitStmt_(const LetStmtNode *op) final {
    let_stmts.push_back(op);
    var_to_let[op->var.get()] = op;
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *) final {
    // Don't descend into nested loops
  }
};

/*!
 * \brief Remove specified LetStmt nodes from a statement
 */
class LetStmtRemover : public StmtExprMutator {
public:
  std::unordered_set<const LetStmtNode *> to_remove;

  static Stmt Remove(const Stmt &stmt,
                     const std::unordered_set<const LetStmtNode *> &targets) {
    LetStmtRemover remover;
    remover.to_remove = targets;
    return remover(stmt);
  }

protected:
  Stmt VisitStmt_(const LetStmtNode *op) final {
    if (to_remove.count(op)) {
      return VisitStmt(op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

/*!
 * \brief Main Loop Invariant Code Motion transformer.
 * For each loop: (1) recursively processes nested loops (bottom-up),
 * (2) analyzes the loop body to find invariant expressions,
 * (3) hoists existing invariant LetStmts,
 * (4) extracts and hoists repeated invariant subexpressions.
 */
class LICMTransformer : public StmtExprMutator {
public:
  explicit LICMTransformer(LICMConfig config) : config_(std::move(config)) {}

  Stmt VisitStmt_(const ForNode *op) final {
    // Phase 0: Recursively process nested loops first (bottom-up)
    Stmt body = VisitStmt(op->body);

    // Collect information about the loop
    auto written_buffers = WrittenBufferCollector::Collect(body);
    LoopInvarianceChecker invariance_checker(op->loop_var, written_buffers);

    // Phase 1: Hoist existing invariant LetStmts
    auto [let_stmts, var_to_let] = LetStmtCollector::Collect(body);
    auto [hoisted_lets, new_body] =
        HoistLetStmts(body, let_stmts, var_to_let, op->loop_var,
                      written_buffers, invariance_checker);

    // Phase 2: Extract and hoist repeated invariant subexpressions
    auto [extracted_lets, final_body] = ExtractInvariantSubexpressions(
        new_body, op->loop_var, written_buffers, invariance_checker);

    // Build the new loop
    Stmt new_loop;
    if (final_body.same_as(op->body)) {
      new_loop = ffi::GetRef<Stmt>(op);
    } else {
      new_loop = For(op->loop_var, op->min, op->extent, op->kind, final_body,
                     op->thread_binding, op->annotations, op->step);
    }

    // Wrap with hoisted LetStmts (in reverse order for correct scoping)
    Stmt result = new_loop;

    // First wrap with extracted subexpressions
    for (auto it = extracted_lets.rbegin(); it != extracted_lets.rend(); ++it) {
      result = LetStmt(it->first, it->second, result);
    }

    // Then wrap with hoisted existing LetStmts
    for (auto it = hoisted_lets.rbegin(); it != hoisted_lets.rend(); ++it) {
      result = LetStmt((*it)->var, (*it)->value, result);
    }

    return result;
  }

private:
  LICMConfig config_;
  int var_counter_ = 0;

  /*!
   * \brief Generate a unique variable name for extracted expressions
   */
  Var GenerateVar(DataType dtype) {
    std::string name = "cse_var_" + std::to_string(var_counter_++);
    return Var(name, dtype);
  }

  /*!
   * \brief Check if a LetStmt's value is loop-invariant considering
   * dependencies
   */
  bool IsLetValueInvariant(
      const LetStmtNode *let, const Var &loop_var,
      const std::unordered_set<const VarNode *> &written_buffers,
      const std::unordered_set<const VarNode *> &invariant_vars,
      const std::unordered_map<const VarNode *, const LetStmtNode *>
          &var_to_let) {

    const PrimExpr &expr = let->value;

    // Check basic invariance
    if (UsesVar(expr, [&loop_var](const VarNode *v) {
          return v == loop_var.get();
        })) {
      return false;
    }

    if (BufferReadChecker::Check(expr, written_buffers)) {
      return false;
    }

    if (SideEffect(expr) > CallEffectKind::kReadState) {
      return false;
    }

    // Check that all referenced variables defined in the loop are also
    // invariant
    bool uses_variant_var = false;
    PostOrderVisit(expr, [&](const ObjectRef &obj) {
      if (uses_variant_var)
        return;
      if (const auto *var = obj.as<VarNode>()) {
        if (var == loop_var.get()) {
          uses_variant_var = true;
          return;
        }
        auto it = var_to_let.find(var);
        if (it != var_to_let.end() && !invariant_vars.count(var)) {
          uses_variant_var = true;
        }
      }
    });

    return !uses_variant_var;
  }

  /*!
   * \brief Phase 1: Hoist existing invariant LetStmts
   */
  std::pair<std::vector<const LetStmtNode *>, Stmt>
  HoistLetStmts(const Stmt &body,
                const std::vector<const LetStmtNode *> &let_stmts,
                const std::unordered_map<const VarNode *, const LetStmtNode *>
                    &var_to_let,
                const Var &loop_var,
                const std::unordered_set<const VarNode *> &written_buffers,
                const LoopInvarianceChecker &checker) {

    std::unordered_set<const VarNode *> invariant_vars;
    std::vector<const LetStmtNode *> hoisted_lets;

    // Iteratively find invariant LetStmts (handles dependencies)
    bool changed = true;
    while (changed) {
      changed = false;
      for (const auto *let : let_stmts) {
        if (invariant_vars.count(let->var.get()))
          continue;

        if (IsLetValueInvariant(let, loop_var, written_buffers, invariant_vars,
                                var_to_let)) {
          invariant_vars.insert(let->var.get());
          hoisted_lets.push_back(let);
          changed = true;
        }
      }
    }

    if (hoisted_lets.empty()) {
      return {hoisted_lets, body};
    }

    // Remove hoisted LetStmts from the loop body
    std::unordered_set<const LetStmtNode *> to_remove(hoisted_lets.begin(),
                                                      hoisted_lets.end());
    Stmt new_body = LetStmtRemover::Remove(body, to_remove);

    return {hoisted_lets, new_body};
  }

  /*!
   * \brief Phase 2: Extract and hoist repeated invariant subexpressions
   */
  std::pair<std::vector<std::pair<Var, PrimExpr>>, Stmt>
  ExtractInvariantSubexpressions(
      const Stmt &body, const Var &loop_var,
      const std::unordered_set<const VarNode *> &written_buffers,
      const LoopInvarianceChecker &checker) {

    std::vector<std::pair<Var, PrimExpr>> extracted;
    Stmt current_body = body;

    // Collect subexpressions and their counts
    auto expr_counts = SubexpressionCollector::Collect(current_body);

    // Build candidates with two criteria:
    // 1. CSE: expressions appearing multiple times (count >= threshold)
    // 2. LICM: expressions appearing once but with high complexity
    std::vector<std::pair<PrimExpr, size_t>> candidates;
    for (const auto &[expr, count] : expr_counts) {
      if (!checker.IsInvariant(expr)) {
        continue;
      }
      size_t complexity = ExprComplexityCalculator::Calculate(expr);

      // CSE criterion: appears multiple times
      bool cse_eligible =
          (static_cast<int>(count) >= config_->min_occurrences_for_cse &&
           static_cast<int>(complexity) >= config_->min_complexity_for_cse);

      // LICM criterion: complex enough to warrant hoisting even if appears once
      bool licm_eligible =
          (static_cast<int>(complexity) >= config_->min_complexity_for_licm);

      if (cse_eligible || licm_eligible) {
        candidates.emplace_back(expr, count);
      }
    }

    if (candidates.empty()) {
      return {extracted, body};
    }

    // Sort by complexity (descending) - extract larger expressions first
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const auto &a, const auto &b) {
                       return ExprComplexityCalculator::Calculate(a.first) >
                              ExprComplexityCalculator::Calculate(b.first);
                     });

    // Extract each candidate
    for (const auto &[expr, count] : candidates) {
      // Re-check if expression still exists after previous extractions
      auto current_counts = SubexpressionCollector::Collect(current_body);
      auto it = current_counts.find(expr);
      if (it == current_counts.end()) {
        continue;
      }

      // Create a new variable and replace all occurrences
      Var new_var = GenerateVar(expr.dtype());
      current_body = ExpressionReplacer::Replace(current_body, expr, new_var);
      extracted.emplace_back(new_var, expr);
    }

    return {extracted, current_body};
  }
};

using namespace tir::transform;

tvm::transform::Pass LoopInvariantCodeMotion() {
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    // Check if LICM is enabled (default: false)
    bool enabled =
        ctx->GetConfig<Bool>(kEnableLoopInvariantCodeMotion, Bool(false)).value();
    if (!enabled) {
      return f;
    }

    auto config = ctx->GetConfig<LICMConfig>("tl.LoopInvariantCodeMotion")
                      .value_or(AttrsWithDefaultValues<LICMConfig>());
    LICMTransformer transformer(config);
    f.CopyOnWrite()->body = transformer(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LoopInvariantCodeMotion", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LoopInvariantCodeMotion",
                        LoopInvariantCodeMotion);
}

} // namespace tl
} // namespace tvm
