#include "compat/metadata.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/tirx/transform.h>

#include "transform/tirx_passes.h"

#include <unordered_set>

namespace tvm::tl {

bool CallAttrsMayContainExpr(const CallNode *call) {
  if (!call->attrs.has_value())
    return false;
  const auto *operation = call->op.as<OpNode>();
  if (!operation)
    return false;
  static const std::unordered_set<std::string> native{
      "tirx.call_extern",     "tirx.call_pure_extern",
      "tirx.reinterpret",     "tirx.address_of",
      "tirx.ptr_byte_offset", "tirx.handle_add_byte_offset",
      "tirx.alloc_tensor",    "tirx.decl_tensor",
      "tirx.tensor_data_ptr"};
  if (native.count(operation->name))
    return true;
  static auto attributes = Op::GetAttrMap<bool>("tl.attrs_may_contain_expr");
  return attributes.get(call->op, false);
}

ffi::AnyView ExpressionMetadata(ffi::AnyView value) {
  if (const auto *call = value.as<CallNode>()) {
    if (tl::CallAttrsMayContainExpr(call))
      return call->attrs;
  } else if (const auto *loop = value.as<ForNode>()) {
    return loop->annotations;
  } else if (const auto *region = value.as<RegionStmtNode>()) {
    return region->attrs;
  } else if (const auto *function = value.as<tirx::FunctionNode>()) {
    return function->attrs;
  }
  return nullptr;
}

ffi::Any MutateMetadata(ffi::AnyView value,
                        const std::function<ffi::Any(ffi::AnyView)> &mutate) {
  if (auto attrs = value.as<DictAttrs>()) {
    auto mapped = MutateMetadata(attrs.value()->dict, mutate);
    if (mapped.same_as(attrs.value()->dict))
      return value;
    return DictAttrs(mapped.cast<ffi::Map<ffi::String, ffi::Any>>());
  }
  if (auto map = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
    ffi::Map<ffi::Any, ffi::Any> rewritten;
    bool changed = false;
    for (const auto &[key, element] : *map) {
      auto mapped_key = MutateMetadata(key, mutate);
      auto mapped_element = MutateMetadata(element, mutate);
      changed |= !mapped_key.same_as(key) || !mapped_element.same_as(element);
      rewritten.Set(mapped_key, mapped_element);
    }
    return changed ? ffi::Any(rewritten) : ffi::Any(value);
  }
  if (auto array = value.as<ffi::Array<ffi::Any>>()) {
    ffi::Array<ffi::Any> rewritten;
    bool changed = false;
    for (const auto &element : *array) {
      auto mapped = MutateMetadata(element, mutate);
      changed |= !mapped.same_as(element);
      rewritten.push_back(mapped);
    }
    return changed ? ffi::Any(rewritten) : ffi::Any(value);
  }
  return mutate(value);
}

ffi::Any
RewriteMetadataFields(ffi::Any value,
                      const std::function<ffi::Any(ffi::AnyView)> &mutate) {
  auto metadata = ExpressionMetadata(value);
  if (metadata == nullptr)
    return value;
  auto mapped = MutateMetadata(metadata, mutate);
  if (mapped.same_as(metadata))
    return value;
  if (auto call = value.as<Call>()) {
    call->CopyOnWrite()->attrs = mapped.cast<Attrs>();
    return *call;
  }
  if (auto loop = value.as<For>()) {
    loop->CopyOnWrite()->annotations =
        mapped.cast<ffi::Map<ffi::String, ffi::Any>>();
    return *loop;
  }
  if (auto region = value.as<RegionStmt>()) {
    region->CopyOnWrite()->attrs = mapped.cast<DictAttrs>();
    return *region;
  }
  auto function = value.cast<tirx::Function>();
  function.CopyOnWrite()->attrs = mapped.cast<DictAttrs>();
  return function;
}

ffi::Optional<ffi::VisitInterrupt> VisitMetadata(
    ffi::AnyView value,
    const std::function<ffi::Optional<ffi::VisitInterrupt>(ffi::AnyView)>
        &visit) {
  if (value == nullptr)
    return std::nullopt;
  if (auto attrs = value.as<DictAttrs>())
    return VisitMetadata(attrs.value()->dict, visit);
  if (auto map = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
    for (const auto &[key, element] : *map) {
      if (auto interrupt = VisitMetadata(key, visit))
        return interrupt;
      if (auto interrupt = VisitMetadata(element, visit))
        return interrupt;
    }
    return std::nullopt;
  }
  if (auto array = value.as<ffi::Array<ffi::Any>>()) {
    for (const auto &element : *array)
      if (auto interrupt = VisitMetadata(element, visit))
        return interrupt;
    return std::nullopt;
  }
  return visit(value);
}

namespace {

class DefinitionRenewer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  ffi::UnchangedOr<Expr> Mutate_(const VarNode *node,
                                 ffi::InplaceMode mode) final {
    auto original = ffi::GetRef<Var>(node);
    if (def_region_kind() != kTVMFFIDefRegionKindSimple) {
      auto mapped = VarRemapGet(ffi::AnyView(node));
      if (mapped.type_index() != ffi::TypeIndex::kTVMFFINone)
        return std::move(mapped).as_or_throw<ffi::UnchangedOr<Expr>>();
      if (def_region_kind() == kTVMFFIDefRegionKindNone)
        return ffi::Unchanged();
    }
    auto mutate_type = [&] {
      return Mutate(node->ty, mode)
          .ValueOrUnchanged(node->ty)
          .as_or_throw<Type>();
    };
    auto type = def_region_kind() == kTVMFFIDefRegionKindSimple
                    ? WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_type)
                    : mutate_type();
    Var renewed(node->name, type, node->loc);
    VarRemapSet(original, renewed);
    return renewed;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *node,
                                 ffi::InplaceMode mode) final {
    auto variable = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
      return Mutate(node->var, mode)
          .ValueOrUnchanged(node->var)
          .as_or_throw<PrimVar>();
    });
    auto begin = Mutate(node->begin, mode).ValueOrUnchanged(node->begin);
    auto end = Mutate(node->end, mode).ValueOrUnchanged(node->end);
    auto step = Mutate(node->step, mode).ValueOrUnchanged(node->step);
    auto body = Mutate(node->body, mode).ValueOrUnchanged(node->body);
    return For(variable, begin, end, step, node->kind, body, node->annotations,
               node->loc);
  }
};

Stmt ComparisonMetadata(DictAttrs attrs, bool had_body) {
  return Evaluate(Call(PrimType::Void(), Op::Get("tl.structural_metadata"), {},
                       DictAttrs({{"value", attrs}, {"had_body", had_body}})));
}

class ComparisonProjector : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;

  ffi::UnchangedOr<ffi::Any> Mutate(ffi::AnyView input,
                                    ffi::InplaceMode mode) final {
    auto mapped = StmtExprMutator::Mutate(input, mode).ValueOrUnchanged(input);
    if (auto function = mapped.as<tirx::Function>()) {
      ffi::Array<Stmt> body;
      if (function.value()->body)
        body = function.value()->body.value()->seq;
      body.push_back(ComparisonMetadata(function.value()->attrs,
                                        function.value()->body.has_value()));
      function->CopyOnWrite()->body = SeqStmt(body);
      function->CopyOnWrite()->attrs = DictAttrs();
      return ffi::Any(*function);
    }
    if (auto region = mapped.as<RegionStmt>()) {
      auto body = region.value()->body->seq;
      body.push_back(ComparisonMetadata(region.value()->attrs, true));
      region->CopyOnWrite()->body = SeqStmt(body);
      region->CopyOnWrite()->attrs = DictAttrs();
      return ffi::Any(*region);
    }
    return mapped.same_as(input) ? ffi::UnchangedOr<ffi::Any>(ffi::Unchanged())
                                 : ffi::UnchangedOr<ffi::Any>(mapped);
  }
};

ffi::Any ComparisonView(ffi::AnyView value) {
  return ffi::make_object<ComparisonProjector>()
      ->Mutate(value, ffi::InplaceMode::kDisallow)
      .ValueOrUnchanged(value);
}

} // namespace

tirx::Function RenewDef(tirx::Function function) {
  return ffi::make_object<DefinitionRenewer>()
      ->Mutate(function, ffi::InplaceMode::kDisallow)
      .ValueOrUnchanged(function)
      .as_or_throw<tirx::Function>();
}

Stmt ConvertSSA(Stmt statement) {
  return tl::RenewDef(tirx::Function({}, SeqStmt({statement})))->body.value();
}

tvm::transform::Pass tirx_passes::ConvertSSA() {
  return tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule, tvm::transform::PassContext) {
        return tl::RenewDef(function);
      },
      0, "tl.ConvertSSA");
}

bool StructuralEqual::operator()(ffi::AnyView first, ffi::AnyView second,
                                 bool map_free_vars) const {
  return ffi::StructuralEqual::Equal(ComparisonView(first),
                                     ComparisonView(second), map_free_vars);
}

int64_t StructuralHash::operator()(ffi::AnyView value,
                                   bool map_free_vars) const {
  return ffi::StructuralHash::Hash(ComparisonView(value), map_free_vars);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.structural_metadata")
      .signature(sig::call_attrs<DictAttrsNode>())
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure);
  ffi::reflection::GlobalDef()
      .def("tl.ir.RenewDef", RenewDef)
      .def("tl._tirx_passes.ConvertSSA", tirx_passes::ConvertSSA)
      .def("tl.ir.StructuralEqual",
           [](ffi::Any first, ffi::Any second, bool map_free_vars) {
             return tl::StructuralEqual()(first, second, map_free_vars);
           })
      .def("tl.ir.StructuralHash",
           [](ffi::Any value, bool map_free_vars) {
             return tl::StructuralHash()(value, map_free_vars);
           })
      .def("tl.ir.AttrsMayContainExpr",
           [](Call call) { return tl::CallAttrsMayContainExpr(call.get()); });
}

} // namespace tvm::tl
