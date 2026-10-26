#include "compat/metadata.h"
/*
 * Hoist tl.non_restrict_params block annotation(s) to PrimFunc attribute.
 *
 * Previously, we only looked at the root block. This version recursively
 * scans all blocks, unions any tl.non_restrict_params entries it finds,
 * merges with any existing PrimFunc-level attribute, then writes the
 * deduplicated result back to the PrimFunc attrs. This makes annotation
 * placement within the function body flexible for frontends.
 */
#include "compat/expr.h"
#include "op/builtin.h"
#include "op/operator.h"

#include <tvm/tirx/transform.h>

#include <set>

namespace tvm::tl {
namespace {

class NonRestrictCollector : public tvm::tl::StmtExprMutator {
public:
  explicit NonRestrictCollector(ffi::Array<Var> parameters)
      : parameters_(std::move(parameters)) {}

  void Collect(const ffi::Any &annotation) {
    for (const ffi::Any &item : annotation.cast<ffi::Array<ffi::Any>>()) {
      if (auto index = GetIntAnnotation(item)) {
        TVM_FFI_CHECK(*index >= 0 &&
                          *index < static_cast<int64_t>(parameters_.size()),
                      ValueError)
            << "Non-restrict parameter index is out of range";
        indices_.insert(*index);
        continue;
      }
      auto value = item.cast<Expr>();
      Var variable = GetVarFromAccessPtr(value);
      if (auto index = FindParameter(variable)) {
        indices_.insert(index.value());
        continue;
      }
      std::string name = Normalize(variable->name);
      ffi::Optional<int> matched;
      for (size_t index = 0; index < parameters_.size(); ++index) {
        if (Normalize(parameters_[index]->name) == name) {
          TVM_FFI_CHECK(!matched, ValueError)
              << "Ambiguous non-restrict parameter name: " << name;
          matched = static_cast<int>(index);
        }
      }
      TVM_FFI_CHECK(matched.has_value(), ValueError)
          << "Non-restrict annotation must refer to a function parameter: "
          << name;
      indices_.insert(matched.value());
    }
  }

  ffi::Array<IntImm> Result() const {
    ffi::Array<IntImm> result;
    for (int index : indices_) {
      result.push_back(IntImm::Int32(index));
    }
    return result;
  }

private:
  static std::string Normalize(std::string name) {
    if (name.size() > 7 && name.ends_with("_handle")) {
      name.resize(name.size() - 7);
    }
    return name;
  }

  ffi::Optional<int> FindParameter(const Var &variable) const {
    for (size_t index = 0; index < parameters_.size(); ++index) {
      if (parameters_[index].same_as(variable)) {
        return static_cast<int>(index);
      }
    }
    return std::nullopt;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    Stmt original = ffi::GetRef<Stmt>(node);
    auto region = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                      .ValueOrUnchanged(original)
                      .as_or_throw<RegionStmt>();
    if (auto annotation = region->attrs->dict.Get(attr::kNonRestrictParams)) {
      Collect(annotation.value());
      auto attributes = region->attrs->dict;
      attributes.erase(attr::kNonRestrictParams);
      region.CopyOnWrite()->attrs = DictAttrs(attributes);
    }
    return region;
  }

  ffi::Array<Var> parameters_;
  std::set<int> indices_;
};

} // namespace

namespace transform {

tvm::transform::Pass HoistNonRestrictParams() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    auto collector = ffi::make_object<NonRestrictCollector>(function->params);
    if (auto existing =
            function->attrs->dict.Get("tl.non_restrict_param_indices")) {
      collector->Collect(existing.value());
    }
    if (auto legacy = function->attrs->dict.Get(attr::kNonRestrictParams)) {
      collector->Collect(legacy.value());
      function = WithoutAttr(std::move(function), attr::kNonRestrictParams);
    }
    if (function->body) {
      SeqStmt body = function->body.value();
      function.CopyOnWrite()->body =
          collector->Mutate(body).ValueOrUnchanged(body);
    }
    auto indices = collector->Result();
    if (!indices.empty()) {
      function = WithAttr(std::move(function), "tl.non_restrict_param_indices",
                          indices);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.HoistNonRestrictParams");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.HoistNonRestrictParams",
                                   HoistNonRestrictParams);
}

} // namespace transform
} // namespace tvm::tl
