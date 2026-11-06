/*!
 * \file ascend/transform/lower_tile_op.cc
 * \brief Native Ascend lowering and retirement of validated MX slot handles.
 */
#include "ascend/ir/block.h"
#include "ascend/op/utils.h"
#include "backend/common/target_utils.h"
#include "ir/annotations.h"
#include "ir/lexical_scope.h"
#include "layout/expr_utils.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/sym/analyzer.h>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <unordered_set>

namespace tvm::tl::transform {
tvm::transform::Pass LowerTileOp();
}

namespace tvm::tl::ascend {
namespace {
using namespace tirx;
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::Optional;
using ffi::String;
using ffi::UnchangedOr;

constexpr const char *kPhysicalVersion = "tl.buffer_version";

bool IsSFVariable(const Var &var) {
  if (auto tensor = var.as<TensorVar>())
    return IsL0SFBuffer(*tensor);
  if (auto pointer = var->ty.as<PtrType>()) {
    const auto &scope = pointer.value()->scope;
    return scope.name() == "shared.l0a.sf" || scope.name() == "shared.l0b.sf";
  }
  return false;
}

class TensorDefinitions : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;
  Map<Var, Expr> definitions;
  Array<TensorVar> sf_handles;

  Optional<VisitInterrupt> Visit_(const BindObj *op) final {
    definitions.Set(op->var, op->value);
    if (IsSFVariable(op->var)) {
      auto tensor = op->var.as<TensorVar>();
      const auto *call = op->value.as<CallObj>();
      TVM_FFI_CHECK(tensor && call && call->op.same_as(alloc_tensor_op()),
                    ValueError)
          << "L0 SF handles require a native alloc_tensor binding: "
          << op->var->name;
      sf_handles.push_back(*tensor);
    }
    return StmtExprVisitor::Visit_(op);
  }
};

Function ValidateSFOwnership(Function function) {
  if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return function;
  auto target = function->GetAttr<Target>(tvm::attr::kTarget);
  TVM_FFI_CHECK(target && TargetIsAscend(*target), ValueError)
      << "AscendLowerTileOp requires an Ascend target";
  for (const Var &parameter : function->params)
    TVM_FFI_CHECK(!IsSFVariable(parameter), ValueError)
        << "L0 SF handles are allocation-bound slots, not function parameters";

  auto definitions = ffi::make_object<TensorDefinitions>();
  definitions->Visit(function->body);
  auto bindings = CollectL0SFBindings(function->body);
  if (definitions->sf_handles.empty() && bindings.empty())
    return function;

  // Erasing a logical handle is valid only after its definitions and all
  // explicit binding operands have been checked in their native scopes.
  CheckWellFormed(function);
  sym::Analyzer analyzer;
  for (const auto &[sf_var, data_var] : bindings) {
    auto sf = sf_var.as_or_throw<TensorVar>();
    auto data = data_var.as_or_throw<TensorVar>();
    auto definition = definitions->definitions.Get(sf);
    const auto *allocation =
        definition ? definition.value().as<CallObj>() : nullptr;
    TVM_FFI_CHECK(allocation && allocation->op.same_as(alloc_tensor_op()),
                  ValueError)
        << "L0 SF binding has no matching handle allocation: " << sf->name;
    bool data_available = definitions->definitions.count(data);
    for (const Var &parameter : function->params)
      data_available |= parameter.same_as(data);
    TVM_FFI_CHECK(data_available, ValueError)
        << "L0 SF binding refers to an unavailable data allocation: "
        << data->name;
    TVM_FFI_CHECK(sf.dtype().lanes() == 1 &&
                      (sf.dtype().bits() == 8 || sf.dtype().bits() == 16),
                  ValueError)
        << "L0 SF storage must contain 8-bit scales or 16-bit scale pairs";
    TVM_FFI_CHECK(sf.shape().size() >= 2 &&
                      sf.shape().size() == data.shape().size(),
                  ValueError)
        << "L0 SF and data allocations must preserve the same leading axes";
    for (size_t i = 0; i + 2 < sf.shape().size(); ++i)
      TVM_FFI_CHECK(analyzer->CanProveEqual(sf.shape()[i], data.shape()[i]),
                    ValueError)
          << "L0 SF and data allocations have different leading extents";
  }
  for (const TensorVar &sf : definitions->sf_handles)
    TVM_FFI_CHECK(bindings.count(sf), ValueError)
        << "L0 SF handle has no allocation binding: " << sf->name;
  return function;
}

Map<String, Any> StringMap(const Any &value) {
  Map<String, Any> result;
  for (const auto &[key, item] : value.cast<Map<Any, Any>>())
    result.Set(key.cast<String>(), item);
  return result;
}

DictAttrs UpdateSchema(const DictAttrs &attrs,
                       const EncodedAnnotations &encoded) {
  auto fields = attrs->dict;
  fields.Set("schema_version", 1);
  fields.Set("annotation_schema", encoded.schema);
  return DictAttrs(fields);
}

// Strip only the binding field whose consumers have just run. Preserve all
// other metadata for its own lowering/finalization and keep lexical scopes.
class SFBindingRetirement : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  SFBindingRetirement(L0SFBindings bindings, const Stmt &body)
      : bindings_(std::move(bindings)) {
    // SF handles share their data allocation's version axis. Validate the
    // metadata before retiring only the logical handles, leaving physical
    // data entries for RewriteAscendBufferVersionLayout.
    std::unordered_set<const VarObj *> members;
    for (const auto &[sf, data] : bindings_) {
      members.insert(sf.get());
      members.insert(data.get());
    }
    Map<Var, int64_t> versions;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        body, [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
          if (!IsAnnotationCall(call))
            return ffi::WalkResult::Advance();

          auto fields = ReadAnnotations(call);
          auto value = fields.Get(kPhysicalVersion);
          if (!value)
            return ffi::WalkResult::Advance();
          for (const auto &[key, count_value] : value->cast<Map<Any, Any>>()) {
            Var storage = key.cast<Var>();
            if (!members.count(storage.get()))
              continue;
            auto count = count_value.try_cast<int64_t>();
            if (!count) {
              auto expression = count_value.try_cast<PrimExpr>();
              if (expression && expression->ty().IsScalar() &&
                  !expression->ty().MatchesCode(kDLBool))
                count = TryConstInt(*expression);
            }
            // The FFI integer conversion also accepts a bare boolean.
            TVM_FFI_CHECK(count_value.type_index() !=
                                  ffi::TypeIndex::kTVMFFIBool &&
                              count && *count > 0 &&
                              *count <= std::numeric_limits<int>::max(),
                          ValueError)
                << "L0 SF physical version count must be a positive int32 "
                   "constant";
            if (auto previous = versions.Get(storage))
              TVM_FFI_CHECK_EQ(*previous, *count, ValueError)
                  << "Conflicting L0 SF physical version counts for "
                  << storage->name;
            versions.Set(storage, *count);
          }
          return ffi::WalkResult::Advance();
        });
    sym::Analyzer analyzer;
    for (const auto &[sf, data] : bindings_) {
      auto count = versions.Get(sf);
      if (!count)
        continue;
      if (auto data_count = versions.Get(data))
        TVM_FFI_CHECK_EQ(*count, *data_count, ValueError)
            << "Bound L0 data/SF physical version counts disagree";
      if (*count > 1) {
        const auto tensor = sf.as_or_throw<TensorVar>();
        TVM_FFI_CHECK(
            tensor.shape().size() > 2 &&
                analyzer->CanProveEqual(tensor.shape()[0],
                                        IntImm(tensor.shape()[0].ty(), *count)),
            ValueError)
            << "L0 SF physical version count does not match its leading axis";
      }
    }
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    const auto *call = op->value.as<CallObj>();
    if (!IsAnnotationCall(call))
      return StmtExprMutator::Mutate_(op, mode);
    auto metadata = ReadAnnotations(call);
    bool changed = metadata.count(kL0SFBindings);
    metadata.erase(kL0SFBindings);
    if (auto value = metadata.Get(kPhysicalVersion)) {
      auto versions = value->cast<Map<Any, Any>>();
      auto original = versions;
      for (const auto &[key, count] : original) {
        if (auto storage = key.try_cast<Var>();
            storage && bindings_.count(*storage)) {
          versions.erase(key);
          changed = true;
        }
      }
      if (versions.empty())
        metadata.erase(kPhysicalVersion);
      else
        metadata.Set(kPhysicalVersion, versions);
    }
    if (!changed)
      return StmtExprMutator::Mutate_(op, mode);
    if (metadata.empty())
      return SeqStmt(Array<Stmt>{}, op->loc);
    auto encoded = EncodeAnnotations(metadata);
    return Evaluate(
        Call(
            call->ty, call->op, {encoded.payload},
            UpdateSchema(call->attrs.value().as_or_throw<DictAttrs>(), encoded),
            call->ty_args, call->loc),
        op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    auto rewritten = StmtExprMutator::Mutate_(op, mode)
                         .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                         .as_or_throw<RegionStmt>();
    if (IsAscendBlock(rewritten.get())) {
      auto metadata = AscendBlockAnnotations(rewritten.get());
      if (!metadata.count(kL0SFBindings))
        return rewritten;
      metadata.erase(kL0SFBindings);
      return MakeAscendBlock(AscendBlockName(rewritten.get()), metadata,
                             rewritten->body, rewritten->loc);
    }
    if (op->op->name != "tl.attr")
      return rewritten;
    auto fields = ReadAnnotations(rewritten.get());
    String key = fields.at("key").cast<String>();
    if (key == kL0SFBindings) {
      TVM_FFI_CHECK(op->body_params.empty() && op->result_vars.empty(),
                    ValueError)
          << "An L0 SF binding annotation cannot define region parameters";
      return RegionStmt(tl::lexical_scope_op(), {}, {}, DictAttrs(),
                        rewritten->body, {}, op->loc);
    }
    if (key != "tl.kernel")
      return rewritten;
    auto metadata = StringMap(fields.at("value"));
    if (!metadata.count(kL0SFBindings))
      return rewritten;
    metadata.erase(kL0SFBindings);
    fields.Set("value", metadata);
    auto encoded = EncodeAnnotations(fields);
    auto node = rewritten.CopyOnWrite();
    node->args = {encoded.payload};
    node->attrs = UpdateSchema(node->attrs, encoded);
    return rewritten;
  }

private:
  L0SFBindings bindings_;
};

// A native visitor distinguishes Var definitions from uses, including inside
// expression payloads. Any surviving SF use means a consumer has not lowered;
// do not hide it by dropping its definition.
class SFUseVerifier : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;

  Optional<VisitInterrupt> Visit_(const VarObj *op) final {
    TVM_FFI_CHECK(def_region_kind() != kTVMFFIDefRegionKindNone ||
                      !IsSFVariable(ffi::GetRef<Var>(op)),
                  ValueError)
        << "L0 SF handle remains in an unlowered consumer: " << op->name;
    return StmtExprVisitor::Visit_(op);
  }

  Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    if (const auto *intrinsic = op->op.as<OpObj>())
      TVM_FFI_CHECK(intrinsic->name.find("tl.tileop.") != 0, ValueError)
          << "AscendLowerTileOp left an unlowered operation: "
          << intrinsic->name;
    return StmtExprVisitor::Visit_(op);
  }
};

class SFAllocationRetirement : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    if (IsSFVariable(op->var))
      return SeqStmt(Array<Stmt>{}, op->loc);
    return StmtExprMutator::Mutate_(op, mode);
  }
};

Function RetireSFHandles(Function function) {
  if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return function;
  auto definitions = ffi::make_object<TensorDefinitions>();
  definitions->Visit(function->body);
  auto bindings = CollectL0SFBindings(function->body);
  for (const TensorVar &sf : definitions->sf_handles)
    TVM_FFI_CHECK(bindings.count(sf), ValueError)
        << "Lowering introduced an L0 SF handle without an allocation binding: "
        << sf->name;

  auto verifier = ffi::make_object<SFUseVerifier>();
  if (definitions->sf_handles.empty() && bindings.empty()) {
    verifier->Visit(function->body);
    return function;
  }
  auto metadata =
      ffi::make_object<SFBindingRetirement>(bindings, function->body);
  SeqStmt body =
      metadata->Mutate(function->body).ValueOrUnchanged(function->body);
  verifier->Visit(body);
  auto allocations = ffi::make_object<SFAllocationRetirement>();
  body = allocations->Mutate(body).ValueOrUnchanged(body);
  function.CopyOnWrite()->body = body;
  CheckWellFormed(function);
  return function;
}

} // namespace
} // namespace tvm::tl::ascend

namespace tvm::tl::transform {

tvm::transform::Pass AscendLowerTileOp() {
  auto validate = tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule, tvm::transform::PassContext) {
        return ascend::ValidateSFOwnership(std::move(function));
      },
      0, "tl.ValidateAscendSFOwnership");
  auto retire = tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule, tvm::transform::PassContext) {
        return ascend::RetireSFHandles(std::move(function));
      },
      0, "tl.RetireAscendSFHandles");
  return tvm::transform::Sequential({validate, LowerTileOp(), retire},
                                    "tl.AscendLowerTileOp");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.AscendLowerTileOp",
                                   AscendLowerTileOp);
}

} // namespace tvm::tl::transform
