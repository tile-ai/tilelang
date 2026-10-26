#ifndef TILELANG_COMPAT_METADATA_H_
#define TILELANG_COMPAT_METADATA_H_

#include "compat/analyzer.h"
#include "compat/context.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>

#include <functional>
#include <unordered_map>

namespace tvm::tl {

bool CallAttrsMayContainExpr(const CallNode *call);
ffi::AnyView ExpressionMetadata(ffi::AnyView value);
ffi::Any MutateMetadata(ffi::AnyView value,
                        const std::function<ffi::Any(ffi::AnyView)> &mutate);
ffi::Any
RewriteMetadataFields(ffi::Any value,
                      const std::function<ffi::Any(ffi::AnyView)> &mutate);
ffi::Optional<ffi::VisitInterrupt> VisitMetadata(
    ffi::AnyView value,
    const std::function<ffi::Optional<ffi::VisitInterrupt>(ffi::AnyView)>
        &visit);

class MetadataDeferral {
public:
  MetadataDeferral(std::unordered_map<const ffi::Object *, size_t> *deferred,
                   ffi::AnyView metadata)
      : deferred_(deferred), object_(metadata.as<ffi::Object>()) {
    if (object_)
      ++(*deferred_)[object_];
  }
  ~MetadataDeferral() {
    if (object_ && --deferred_->at(object_) == 0)
      deferred_->erase(object_);
  }

private:
  std::unordered_map<const ffi::Object *, size_t> *deferred_;
  const ffi::Object *object_;
};

template <typename Parent> class MetadataMutator : public Parent {
public:
  MetadataMutator() = default;
  using Parent::Mutate;
  using Parent::Mutate_;
  using Parent::Parent;

  ffi::UnchangedOr<ffi::Any>
  Mutate(ffi::AnyView input,
         ffi::InplaceMode mode = ffi::InplaceMode::kDisallow) override {
    if (auto attrs = input.as<DictAttrs>()) {
      if (deferred_.count(attrs->get()))
        return ffi::Unchanged();
      auto updated = MutateMetadata(input, [this](ffi::AnyView value) {
        return this->Mutate(value, ffi::InplaceMode::kDisallow)
            .ValueOrUnchanged(value);
      });
      if (updated.same_as(input))
        return ffi::Unchanged();
      return updated;
    }
    ffi::Any rewritten;
    {
      MetadataDeferral defer(&deferred_, ExpressionMetadata(input));
      rewritten = Parent::Mutate(input, mode).ValueOrUnchanged(input);
    }
    rewritten =
        RewriteMetadataFields(std::move(rewritten), [this](ffi::AnyView value) {
          return this->Mutate(value, ffi::InplaceMode::kDisallow)
              .ValueOrUnchanged(value);
        });
    if (rewritten.same_as(input))
      return ffi::Unchanged();
    return rewritten;
  }

protected:
  explicit MetadataMutator(const typename Parent::VTable *table)
      : Parent(table) {}

private:
  std::unordered_map<const ffi::Object *, size_t> deferred_;
};

template <typename Parent> class MetadataVisitor : public Parent {
public:
  MetadataVisitor() = default;
  using Parent::Parent;
  using Parent::Visit;
  using Parent::Visit_;

  ffi::Optional<ffi::VisitInterrupt> Visit(ffi::AnyView input) override {
    if (auto attrs = input.as<DictAttrs>()) {
      if (deferred_.count(attrs->get()))
        return std::nullopt;
      return VisitMetadata(
          input, [this](ffi::AnyView value) { return this->Visit(value); });
    }
    if (!TraverseMetadataAutomatically(input))
      return Parent::Visit(input);
    {
      MetadataDeferral defer(&deferred_, ExpressionMetadata(input));
      if (auto interrupt = Parent::Visit(input))
        return interrupt;
    }
    return VisitMetadata(ExpressionMetadata(input), [this](ffi::AnyView value) {
      return this->Visit(value);
    });
  }

protected:
  explicit MetadataVisitor(const typename Parent::VTable *table)
      : Parent(table) {}
  virtual bool TraverseMetadataAutomatically(ffi::AnyView) const {
    return true;
  }

private:
  std::unordered_map<const ffi::Object *, size_t> deferred_;
};

class StmtExprMutator : public MetadataMutator<tirx::StmtExprMutator> {
public:
  using MetadataMutator::MetadataMutator;
  StmtExprMutator() = default;
};

class StmtExprVisitor : public MetadataVisitor<tirx::StmtExprVisitor> {
public:
  using MetadataVisitor::MetadataVisitor;
  StmtExprVisitor() = default;
};

class ExprMutator : public MetadataMutator<tvm::ExprMutator> {
public:
  using MetadataMutator::MetadataMutator;
  ExprMutator() = default;
};

class ExprVisitor : public MetadataVisitor<tvm::ExprVisitor> {
public:
  using MetadataVisitor::MetadataVisitor;
  ExprVisitor() = default;
};

class StmtExprVisitorWithAnalyzer : public StmtExprVisitor {
public:
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(StmtExprVisitorWithAnalyzer,
                                                StmtExprVisitor)

  PrimExpr Simplify(const PrimExpr &expr) {
    return tl::arith::Simplify(analyzer_, expr);
  }

  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit_(const ForNode *op);
  ffi::Optional<VisitInterrupt> Visit_(const BindNode *op);
  ffi::Optional<VisitInterrupt> Visit_(const IfNode *op);
  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtNode *op);
  ffi::Optional<VisitInterrupt> Visit_(const AssertStmtNode *op);
  ffi::Optional<VisitInterrupt> Visit_(const CallNode *op);
  ffi::Optional<VisitInterrupt> Visit_(const prim::LetNode *op);

protected:
  static void InitVTable(VTable *vtable);
  explicit StmtExprVisitorWithAnalyzer(const VTable *vtable)
      : StmtExprVisitor(vtable) {}

  sym::Analyzer analyzer_;

  ContextStack<WithGroup<tl::arith::ConstraintContext>> constraint_scope_;

  PrimExpr ExtractRealCondition(PrimExpr condition) const;
};

class StmtExprMutatorWithAnalyzer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit StmtExprMutatorWithAnalyzer(const sym::Analyzer &analyzer)
      : StmtExprMutatorWithAnalyzer(analyzer.get()) {}
  explicit StmtExprMutatorWithAnalyzer(sym::AnalyzerObj *analyzer)
      : StmtExprMutatorWithAnalyzer(analyzer, GlobalVTable()) {}

  UnchangedOr<Stmt> Mutate_(const ForNode *op,
                            InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const BindNode *op,
                            InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const IfNode *op,
                            InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *op,
                            InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const AssertStmtNode *op,
                            InplaceMode inplace_mode) override;
  UnchangedOr<Expr> Mutate_(const CallNode *op,
                            InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LetNode *op,
                                InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::SelectNode *op,
                                InplaceMode inplace_mode) override;

protected:
  static void InitVTable(VTable *vtable);
  StmtExprMutatorWithAnalyzer(sym::AnalyzerObj *analyzer, const VTable *vtable)
      : StmtExprMutator(vtable), analyzer_(analyzer) {}
  static const VTable *GlobalVTable();

  void MarkTensorParamShapes(const tirx::Function &func);

  ffi::Array<PrimExpr>
  IterMapSimplifyWithContext(const ffi::Array<PrimExpr> &indices,
                             bool non_trivial_only);

  sym::AnalyzerObj *analyzer_;

  ContextStack<WithGroup<tl::arith::ConstraintContext>> constraint_scope_;

  ffi::Map<PrimVar, Range> iter_vars_;

  ffi::Array<PrimExpr> iter_predicates_;

  template <typename FLambda>
  void WithRecordIterPredicate(PrimExpr condition, FLambda callback) {
    std::unordered_set<const VarNode *> iter_var_nodes;
    for (const auto &[var, domain] : iter_vars_) {
      iter_var_nodes.insert(var.get());
    }
    auto f_use_itervar = [&iter_var_nodes](const tvm::VarNode *variable) {
      return iter_var_nodes.count(variable);
    };
    auto walkfn = [&](const tvm::Var &var) -> ffi::Expected<ffi::WalkResult> {
      return f_use_itervar(var.get())
                 ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                 : ffi::WalkResult::Advance();
    };

    if (ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(condition, walkfn)
            .has_value()) {
      iter_predicates_.push_back(condition);
      callback();
      iter_predicates_.pop_back();
    } else {
      callback();
    }
  }
};

tirx::Function RenewDef(tirx::Function function);
Stmt ConvertSSA(Stmt statement);
bool CheckWellFormed(const tirx::Function &function, bool assert_mode = true);
bool CheckWellFormed(const IRModule &module, bool assert_mode = true);

struct StructuralEqual {
  bool operator()(ffi::AnyView first, ffi::AnyView second,
                  bool map_free_vars = false) const;
};

struct StructuralHash {
  int64_t operator()(ffi::AnyView value, bool map_free_vars = false) const;
};

} // namespace tvm::tl

#endif
