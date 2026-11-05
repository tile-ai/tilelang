/*!
 * \file tl/ir/analyzer.h
 * \brief TileLang's policy for best-effort native symbolic proofs.
 */

#ifndef TILELANG_IR_ANALYZER_H_
#define TILELANG_IR_ANALYZER_H_

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/sym/analyzer.h>
#include <tvm/sym/constraint.h>

namespace tvm::tl {

// Use the upstream expression rules and its instance-local DEFAULT fallback.
// An analyzer starts without premises; callers supply the facts in their scope.
inline sym::Analyzer CreateTileLangAnalyzer() {
  sym::Analyzer analyzer;
  analyzer->SetZ3DefaultFallback(true);
  return analyzer;
}

/*! \brief Append a durable premise to this analyzer's explicitly owned context.
 * A Var identity can be bound only once, as required by the upstream analyzer.
 * Traversal-owned facts should instead be appended to its active collector.
 */
inline void AppendAnalyzerConstraint(sym::AnalyzerObj *analyzer,
                                     const sym::Constraint &constraint) {
  auto context = analyzer->GetConstraintContext();
  context->Append(constraint);
  analyzer->SyncConstraintContext(context);
}

// A standalone analysis may deliberately recompute a synthetic iterator's
// domain. Replace its one binding in the premise stream, then let the native
// analyzer rebuild derived facts. Never mutate a captured constraint in place.
inline void SetAnalyzerBinding(sym::AnalyzerObj *analyzer, const Var &var,
                               const sym::Constraint &binding,
                               bool replace_existing = false) {
  auto previous = analyzer->GetConstraintContext();
  sym::ConstraintContext context;
  bool found = false;
  previous->ForEach([&](const sym::Constraint &entry) {
    const Var *bound = nullptr;
    if (const auto *value = entry.as<sym::BindingConstraintObj>())
      bound = &value->var;
    if (const auto *range = entry.as<sym::RangeConstraintObj>())
      bound = &range->var;
    if (bound && bound->same_as(var)) {
      TVM_FFI_CHECK(replace_existing || ffi::StructuralEqual()(entry, binding),
                    ValueError)
          << "TileLang cannot bind a Var to two different premises";
      context->Append(binding);
      found = true;
    } else {
      context->Append(entry);
    }
  });
  if (found) {
    if (replace_existing)
      analyzer->SyncConstraintContext(context);
  } else {
    previous->Append(binding);
    analyzer->SyncConstraintContext(previous);
  }
}

inline void BindAnalyzer(sym::AnalyzerObj *analyzer, const Var &var,
                         const PrimExpr &value, bool replace_existing = false) {
  SetAnalyzerBinding(analyzer, var, sym::BindingConstraint(var, value),
                     replace_existing);
}

inline void BindAnalyzer(sym::AnalyzerObj *analyzer, const Var &var,
                         const Range &range, bool replace_existing = false) {
  SetAnalyzerBinding(analyzer, var, sym::RangeConstraint(var, range),
                     replace_existing);
}

template <typename Value>
void BindAnalyzer(const sym::Analyzer &analyzer, const Var &var,
                  const Value &value, bool replace_existing = false) {
  BindAnalyzer(analyzer.get(), var, value, replace_existing);
}

// Temporary facts in a traversal belong to its collector. Syncing only the
// analyzer would be undone by the next GetSyncedAnalyzer() query.
template <typename Traversal, typename F>
auto WithTraversalConstraints(Traversal *traversal,
                              const ffi::Array<sym::Constraint> &constraints,
                              F &&f) -> decltype(f()) {
  using Result = decltype(f());
  return traversal
      ->WithConstraintScope([&]() -> ffi::Expected<Result> {
        for (const auto &constraint : constraints)
          traversal->AppendConstraint(constraint);
        return f();
      })
      .value();
}

/*! \brief Use temporary premises and restore the previously accepted context.
 * This holds a native analyzer handle, not a second object or solver system.
 * Context entries are immutable after capture. As with upstream Sync, callers
 * must discard an analyzer when premise ingestion throws.
 */
class ScopedAnalyzerConstraints {
public:
  ScopedAnalyzerConstraints(const sym::Analyzer &analyzer,
                            const ffi::Array<sym::Constraint> &constraints)
      : analyzer_(analyzer), saved_(analyzer->GetConstraintContext()) {
    auto context = saved_->Snapshot();
    for (const auto &constraint : constraints)
      context->Append(constraint);
    analyzer_->SyncConstraintContext(context);
  }

  ScopedAnalyzerConstraints(sym::AnalyzerObj *analyzer,
                            const ffi::Array<sym::Constraint> &constraints)
      : ScopedAnalyzerConstraints(ffi::GetRef<sym::Analyzer>(analyzer),
                                  constraints) {}

  ScopedAnalyzerConstraints(const sym::Analyzer &analyzer,
                            const PrimExpr &condition,
                            bool assume_always = false)
      : ScopedAnalyzerConstraints(
            analyzer, ffi::Array<sym::Constraint>{sym::PredicateConstraint(
                          condition, assume_always)}) {}

  ScopedAnalyzerConstraints(sym::AnalyzerObj *analyzer,
                            const PrimExpr &condition,
                            bool assume_always = false)
      : ScopedAnalyzerConstraints(ffi::GetRef<sym::Analyzer>(analyzer),
                                  condition, assume_always) {}

  ScopedAnalyzerConstraints(const ScopedAnalyzerConstraints &) = delete;
  ScopedAnalyzerConstraints &
  operator=(const ScopedAnalyzerConstraints &) = delete;

  ~ScopedAnalyzerConstraints() { analyzer_->SyncConstraintContext(saved_); }

private:
  sym::Analyzer analyzer_;
  sym::ConstraintContext saved_;
};

} // namespace tvm::tl

#endif // TILELANG_IR_ANALYZER_H_
