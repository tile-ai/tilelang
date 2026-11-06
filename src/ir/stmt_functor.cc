/*!
 * \file ir/stmt_functor.cc
 * \brief Traverse TileLang layout declarations with the native remap state.
 */
#include "stmt_functor.h"
#include "layout/layout.h"
#include "tirx/ir/data_type_rewriter.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/analysis.h>

namespace tvm::tl {

ffi::Expected<ffi::Optional<ffi::VisitInterrupt>>
VisitLayoutMapCall(const CallObj *call,
                   ffi::StructuralVisitorObj *visitor) try {
  // FFI Map's default traversal only visits values. Tensor keys are IR uses.
  for (const auto &[tensor, layout] : ReadLayoutMapAnnotation(call)) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(tensor));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(layout));
  }
  return std::nullopt;
} catch (const ffi::Error &error) {
  return ffi::Unexpected(error);
}

ffi::Expected<ffi::UnchangedOr<ffi::Any>>
MutateLayoutMapCall(const CallObj *call, ffi::StructuralMutatorObj *mutator,
                    ffi::InplaceMode mode) try {
  ffi::Map<tirx::TensorVar, Layout> rewritten;
  bool changed = false;
  for (const auto &[tensor, layout] : ReadLayoutMapAnnotation(call)) {
    // Values in this map can be shared with another annotation. Do not mutate
    // them in place while the original Call is still observable.
    auto new_tensor =
        mutator->MutateExpected(tensor, ffi::InplaceMode::kDisallow)
            .value()
            .ValueOrUnchanged(ffi::Any(tensor))
            .cast<tirx::TensorVar>();
    auto new_layout =
        mutator->MutateExpected(layout, ffi::InplaceMode::kDisallow)
            .value()
            .ValueOrUnchanged(ffi::Any(layout))
            .cast<Layout>();
    TVM_FFI_CHECK(!rewritten.count(new_tensor), ValueError)
        << "Variable substitution merges layout_map keys; alias compatibility "
           "must be checked by the owning pass";
    rewritten.Set(new_tensor, new_layout);
    changed |= !new_tensor.same_as(tensor) || !new_layout.same_as(layout);
  }
  if (!changed)
    return ffi::Unchanged();
  auto attrs = call->attrs.value().as_or_throw<DictAttrs>()->dict;
  attrs.Set("layout_map", rewritten);
  return ffi::Any(Call(call->ty, call->op, call->args, DictAttrs(attrs),
                       call->ty_args, call->loc));
} catch (const ffi::Error &error) {
  return ffi::Unexpected(error);
}

tirx::Function RenewDef(tirx::Function function) {
  ffi::Map<Var, bool> definitions;
  ffi::StructuralVisit(
      function,
      [&](const Call &call, ffi::StructuralVisitorObj *visitor) {
        return call->op.same_as(layout_map_op())
                   ? VisitLayoutMapCall(call.get(), visitor)
                   : visitor->DefaultVisitExpected(call);
      },
      [&](const Var &var, ffi::StructuralVisitorObj *visitor) {
        if (visitor->def_region_kind() != kTVMFFIDefRegionKindNone)
          definitions.Set(var, true);
        return visitor->DefaultVisitExpected(var);
      });
  return ffi::StructuralMutate(
             function,
             [](const Call &call, ffi::StructuralMutatorObj *mutator,
                ffi::InplaceMode mode) {
               return call->op.same_as(layout_map_op())
                          ? MutateLayoutMapCall(call.get(), mutator, mode)
                          : mutator->DefaultMutateExpected(call, mode);
             },
             [&](const Var &var, ffi::StructuralMutatorObj *mutator,
                 ffi::InplaceMode mode)
                 -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
               TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
                   ffi::Any, existing, mutator->VarRemapGetExpected(var));
               if (existing != nullptr)
                 return existing;
               TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
                   ffi::UnchangedOr<ffi::Any>, result,
                   mutator->DefaultMutateExpected(var, mode));
               Var mapped = std::move(result)
                                .ValueOrUnchanged(ffi::Any(var))
                                .cast<Var>();
               if (definitions.count(var) && mapped.same_as(var) &&
                   mutator->def_region_kind() != kTVMFFIDefRegionKindNone) {
                 Var fresh(var->name, mapped->ty, var->loc);
                 if (auto status = mutator->VarRemapSetExpected(var, fresh);
                     status.is_err())
                   return ffi::Unexpected(std::move(status).error());
                 return ffi::Any(fresh);
               }
               if (mapped.same_as(var))
                 return ffi::Unchanged();
               return ffi::Any(mapped);
             })
      .cast<tirx::Function>();
}

namespace {
class LayoutSubstitute : public tirx::DataTypeLegalizer {
public:
  LayoutSubstitute(
      const Stmt &statement,
      const ffi::TypedFunction<ffi::Optional<Expr>(const Var &)> &mapping) {
    ffi::StructuralVisit(
        statement,
        [&](const Call &call, ffi::StructuralVisitorObj *visitor) {
          return call->op.same_as(layout_map_op())
                     ? VisitLayoutMapCall(call.get(), visitor)
                     : visitor->DefaultVisitExpected(call);
        },
        [&](const Var &var, ffi::StructuralVisitorObj *visitor) {
          if (auto value = mapping(var))
            VarRemapSet(var, *value);
          return visitor->DefaultVisitExpected(var);
        });
  }
  using tirx::DataTypeLegalizer::Mutate_;
  ffi::UnchangedOr<Expr> Mutate_(const CallObj *call, InplaceMode mode) final {
    if (!call->op.same_as(layout_map_op()))
      return tirx::DataTypeLegalizer::Mutate_(call, mode);
    auto result = MutateLayoutMapCall(call, this, mode).value();
    if (result.IsUnchanged())
      return ffi::Unchanged();
    return std::move(result).ValueUnchecked().cast<Expr>();
  }
  ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *op,
                                     InplaceMode) final {
    auto result = tirx::StmtExprMutator::Mutate_(op, InplaceMode::kDisallow);
    if (result.UnchangedOrSameAs(ffi::GetRef<PrimExpr>(op)))
      return ffi::Unchanged();
    auto load = std::move(result).ValueUnchecked().as_or_throw<TensorLoad>();
    if (auto tensor = load->source.as<tirx::TensorVar>())
      return TensorLoad(*tensor, load->indices, load->loc);
    return load;
  }
};
} // namespace

Stmt SubstituteWithDataTypeLegalization(
    Stmt statement,
    ffi::TypedFunction<ffi::Optional<PrimExpr>(const Var &)> mapping) {
  auto adapt = [mapping](const Var &var) -> ffi::Optional<Expr> {
    if (auto value = mapping(var))
      return *value;
    return std::nullopt;
  };
  return ffi::make_object<LayoutSubstitute>(statement, adapt)
      ->Mutate(statement, InplaceMode::kDisallow)
      .ValueOrUnchanged(statement);
}

void CheckLayoutMapScopes(const tirx::Function &function) {
  class ScopeProjection : public tirx::StmtExprMutator {
  public:
    bool found = false;
    using tirx::StmtExprMutator::Mutate_;
    ffi::UnchangedOr<Expr> Mutate_(const CallObj *call,
                                   InplaceMode mode) final {
      if (!call->op.same_as(layout_map_op()))
        return tirx::StmtExprMutator::Mutate_(call, mode);
      found = true;
      ffi::Array<Expr> uses;
      for (const auto &[tensor, layout] : ReadLayoutMapAnnotation(call)) {
        uses.push_back(tensor);
        // The Layout's coordinate definitions are local, while its free
        // captures must be in scope at this exact annotation location.
        for (const Var &var : UndefinedVars(layout))
          uses.push_back(var);
      }
      return tvm::Tuple(uses, call->loc);
    }
  };
  auto projection = ffi::make_object<ScopeProjection>();
  auto view = projection->Mutate(function, InplaceMode::kDisallow)
                  .ValueOrUnchanged(function)
                  .as_or_throw<tirx::Function>();
  if (!projection->found)
    return;
  try {
    tirx::CheckWellFormed(view);
  } catch (const ffi::Error &error) {
    TVM_FFI_THROW(ValueError)
        << "Invalid layout_map scope: " << error.message();
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.RenewDef", RenewDef)
      .def("tl.CheckLayoutMapScopes", CheckLayoutMapScopes)
      .def("tl.SubstituteLayoutMap",
           [](Stmt statement, ffi::Map<Var, Expr> values) {
             auto mapping = [values](const Var &var) -> ffi::Optional<Expr> {
               if (auto value = values.Get(var))
                 return *value;
               return std::nullopt;
             };
             return ffi::make_object<LayoutSubstitute>(statement, mapping)
                 ->Mutate(statement, InplaceMode::kDisallow)
                 .ValueOrUnchanged(statement);
           });
}
} // namespace tvm::tl
