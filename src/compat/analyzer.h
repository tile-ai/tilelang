#ifndef TILELANG_COMPAT_ANALYZER_H_
#define TILELANG_COMPAT_ANALYZER_H_

#include <tvm/ir/with_context.h>
#include <tvm/sym/analyzer.h>
#include <tvm/sym/int_set.h>

namespace tvm::tl::arith {

struct AnalyzerRef {
  AnalyzerRef(sym::AnalyzerObj *analyzer) : value(analyzer) {}
  AnalyzerRef(const sym::Analyzer &analyzer) : value(analyzer.get()) {}
  sym::AnalyzerObj *value;
};

PrimExpr Simplify(AnalyzerRef analyzer, const PrimExpr &expression,
                  int steps = 2);
PrimExpr RewriteSimplify(AnalyzerRef analyzer, const PrimExpr &expression);
PrimExpr CanonicalSimplify(AnalyzerRef analyzer, const PrimExpr &expression);
sym::ConstIntBound
ConstIntBoundOf(AnalyzerRef analyzer, const PrimExpr &expression,
                sym::ConstIntBoundAnalyzer::BoundMapType *bounds = nullptr);
sym::ModularSet ModularSetOf(AnalyzerRef analyzer, const PrimExpr &expression);
sym::IntSet IntSetOf(AnalyzerRef analyzer, const PrimExpr &expression);
sym::IntSet IntSetOf(AnalyzerRef analyzer, const PrimExpr &expression,
                     const ffi::Map<Var, sym::IntSet> &domains);
bool CanProve(AnalyzerRef analyzer, const PrimExpr &condition,
              sym::ProofStrength strength = sym::ProofStrength::kDefault);
bool CanProveEqual(AnalyzerRef analyzer, const PrimExpr &first,
                   const PrimExpr &second);
bool CanProveGreaterEqual(AnalyzerRef analyzer, const PrimExpr &expression,
                          int64_t lower_bound);
bool CanProveLess(AnalyzerRef analyzer, const PrimExpr &expression,
                  int64_t upper_bound);
bool CanProveLessEqualThanSymbolicShapeValue(AnalyzerRef analyzer,
                                             const PrimExpr &expression,
                                             const PrimExpr &shape);
bool Z3CanProve(AnalyzerRef analyzer, const PrimExpr &condition);
ffi::String Z3GetModel(AnalyzerRef analyzer, const PrimExpr &condition);
void Bind(AnalyzerRef analyzer, const Var &variable, const PrimExpr &value,
          bool allow_override = false);
void Bind(AnalyzerRef analyzer, const Var &variable, const Range &range,
          bool allow_override = false);
void Bind(AnalyzerRef analyzer, const ffi::Map<Var, Range> &ranges,
          bool allow_override = false);
PrimExpr NormalizeConstraint(AnalyzerRef analyzer, const PrimExpr &condition);

class ConstraintContext {
  friend class tvm::With<ConstraintContext>;
  ConstraintContext(AnalyzerRef analyzer, PrimExpr condition,
                    bool is_assume = false)
      : analyzer_(ffi::GetRef<sym::Analyzer>(analyzer.value)),
        condition_(std::move(condition)), is_assume_(is_assume) {}
  void EnterWithScope() {
    native_.reset(new tvm::With<sym::ConstraintContext>(
        analyzer_, NormalizeConstraint(analyzer_, condition_), is_assume_));
  }
  void ExitWithScope() { delete native_.release(); }
  sym::Analyzer analyzer_;
  PrimExpr condition_;
  bool is_assume_;
  std::unique_ptr<tvm::With<sym::ConstraintContext>> native_;
};

} // namespace tvm::tl::arith

#endif
