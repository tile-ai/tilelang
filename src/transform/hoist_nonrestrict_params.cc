/*!
 * \file hoist_nonrestrict_params.cc
 * \brief Carry alias exceptions through native view lowering and ABI splitting.
 */
#include "../ir/alias_analysis.h"
#include "../ir/annotations.h"
#include "../op/native_utils.h"
#include "ir/lexical_scope.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_set>

namespace tvm::tl {
namespace {
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;
constexpr const char *kNonRestrict = "tl.non_restrict_params";

Map<String, Any> Metadata(const CallObj *call) {
  auto decoded = DecodeAnnotations(call->args[0].as_or_throw<tvm::Tuple>(),
                                   call->attrs.value()
                                       .as_or_throw<DictAttrs>()
                                       .GetAttr<Array<Any>>("annotation_schema")
                                       .value());
  Map<String, Any> values;
  for (const auto &[key, value] : decoded.cast<Map<Any, Any>>())
    values.Set(key.cast<String>(), value);
  return values;
}

Stmt Annotation(const Map<String, Any> &values, Location loc) {
  auto encoded = EncodeAnnotations(values);
  return Evaluate(Call(PrimType::Void(), Op::Get("tl.annotation"),
                       {encoded.payload},
                       DictAttrs({{"schema_version", 1},
                                  {"annotation_schema", encoded.schema}}),
                       {}, loc),
                  loc);
}

class AliasExceptions {
public:
  explicit AliasExceptions(const tirx::Function &function) : roots_(function) {
    if (auto direct = function->GetAttr<Array<Any>>(kNonRestrict))
      Add(direct.value());
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return;
    VisitNativePostOrder(function->body, [&](const ffi::ObjectRef &object) {
      if (const auto *call = object.as<CallObj>();
          call && call->op.same_as(Op::Get("tl.annotation"))) {
        auto values = Metadata(call);
        if (auto marked = values.Get(kNonRestrict))
          Add(marked.value());
      } else if (const auto *region = object.as<RegionStmtObj>();
                 region && region->op->name == "tl.attr") {
        auto fields = DecodeAttributeRegion(region);
        auto key = fields.at("key").cast<String>();
        if (key == kNonRestrict)
          Add(fields.at("value"));
        else if (key == "tl.kernel") {
          auto values = fields.at("value").cast<Map<String, Any>>();
          if (auto marked = values.Get(kNonRestrict))
            Add(marked.value());
        }
      }
    });
  }

  Array<Var> Select(const Array<Var> &candidates) const {
    Array<Var> result;
    for (const Var &var : candidates) {
      if (!var->ty.as<PtrTypeObj>() && !var->ty.as<tirx::TensorTypeObj>())
        continue;
      auto root = roots_.Root(var);
      if (unknown_ || (root && marked_.count(root.value().get())))
        result.push_back(var);
    }
    return result;
  }

  bool HasExceptions() const { return unknown_ || !marked_.empty(); }

private:
  void Add(const Any &values) {
    for (const Any &value : values.cast<Array<Any>>()) {
      auto expression = value.cast<Expr>();
      TVM_FFI_CHECK(expression->ty.as<PtrTypeObj>() ||
                        expression->ty.as<tirx::TensorTypeObj>(),
                    TypeError)
          << "non-restrict annotations require native tensor or pointer values";
      auto root = roots_.Root(expression);
      if (root)
        marked_.insert(root.value().get());
      else
        unknown_ = true;
    }
  }
  PhysicalRootAnalysis roots_;
  std::unordered_set<const VarObj *> marked_;
  bool unknown_{false};
};

class AliasAnnotationEraser : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  ffi::UnchangedOr<Stmt> Mutate_(const EvaluateObj *op,
                                 InplaceMode mode) override {
    const auto *call = op->value.as<CallObj>();
    if (!call || !call->op.same_as(Op::Get("tl.annotation")))
      return StmtExprMutator::Mutate_(op, mode);
    auto values = Metadata(call);
    if (!values.count(kNonRestrict))
      return StmtExprMutator::Mutate_(op, mode);
    values.erase(kNonRestrict);
    if (values.empty())
      return SeqStmt(Array<Stmt>{}, op->loc);
    return Annotation(values, op->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) override {
    if (op->op->name != "tl.attr")
      return StmtExprMutator::Mutate_(op, mode);
    auto fields = DecodeAttributeRegion(op);
    auto key = fields.at("key").cast<String>();
    if (key == kNonRestrict) {
      auto body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
      return RegionStmt(tl::lexical_scope_op(), {}, {}, DictAttrs(), body, {},
                        op->loc);
    }
    if (key == "tl.kernel") {
      auto values = fields.at("value").cast<Map<String, Any>>();
      if (values.count(kNonRestrict)) {
        values.erase(kNonRestrict);
        fields.Set("value", values);
        auto encoded = EncodeAnnotations(fields);
        auto body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
        return RegionStmt(op->op, {encoded.payload}, op->body_params,
                          DictAttrs({{"schema_version", 1},
                                     {"annotation_schema", encoded.schema}}),
                          body, op->result_vars, op->loc);
      }
    }
    return StmtExprMutator::Mutate_(op, mode);
  }
};

class DeviceAliasStager : public AliasAnnotationEraser {
public:
  using AliasAnnotationEraser::Mutate;
  using AliasAnnotationEraser::Mutate_;
  explicit DeviceAliasStager(const AliasExceptions *aliases)
      : aliases_(aliases) {}

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    bool entry = !in_device_ && (op->op->name == "tirx.device_scope" ||
                                 op->op->name == "tirx.device_entry" ||
                                 op->op.same_as(tirx::launch_thread_op()));
    if (!entry)
      return AliasAnnotationEraser::Mutate_(op, mode);
    auto selected =
        aliases_->Select(tvm::UndefinedVars(op->body, op->body_params));
    in_device_ = true;
    auto body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
    in_device_ = false;
    if (!selected.empty())
      body = SeqStmt(
          {Annotation(Map<String, Any>{{kNonRestrict, selected}}, op->loc),
           body},
          op->loc);
    return RegionStmt(op->op, op->args, op->body_params, op->attrs, body,
                      op->result_vars, op->loc);
  }

private:
  const AliasExceptions *aliases_;
  bool in_device_{false};
};

tirx::Function Normalize(tirx::Function function) {
  if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return function;
  if (auto values = function->GetAttr<Array<Any>>(kNonRestrict)) {
    function.CopyOnWrite()->body =
        SeqStmt({Annotation(Map<String, Any>{{kNonRestrict, values.value()}},
                            function->loc),
                 function->body},
                function->loc);
    function = WithoutAttr(function, kNonRestrict);
  }
  return function;
}
} // namespace

tirx::Function StageNonRestrictForSplit(tirx::Function function) {
  if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return function;
  AliasExceptions aliases(function);
  auto stager = ffi::make_object<DeviceAliasStager>(&aliases);
  auto body = function->body;
  function.CopyOnWrite()->body = stager->Mutate(body).ValueOrUnchanged(body);
  if (aliases.HasExceptions())
    function = WithAttr(function, "tl.host_may_alias", true);
  // These fields describe the old signature. The post-split hoister recovers
  // exceptions from actual new pointer parameters and regenerates indices.
  function = WithoutAttr(function, kNonRestrict);
  return WithoutAttr(function, "tl.may_alias_param_indices");
}

namespace transform {
tvm::transform::Pass NormalizeAliasAnnotations() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) { return Normalize(function); };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.NormalizeAliasAnnotations");
}

tvm::transform::Pass HoistNonRestrictParams(bool finalize) {
  auto pass = [finalize](tirx::Function function, IRModule,
                         tvm::transform::PassContext) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    AliasExceptions aliases(function);
    auto selected = aliases.Select(function->params);
    function = WithoutAttr(function, kNonRestrict);
    function = WithoutAttr(function, "tl.may_alias_param_indices");
    if (!selected.empty())
      function = WithAttr(function, kNonRestrict, selected);
    if (finalize) {
      auto eraser = ffi::make_object<AliasAnnotationEraser>();
      auto body = function->body;
      function.CopyOnWrite()->body =
          eraser->Mutate(body).ValueOrUnchanged(body);
      Array<int64_t> indices;
      for (size_t i = 0; i < function->params.size(); ++i) {
        const auto &parameter = function->params[i];
        if (!parameter->ty.as<PtrTypeObj>())
          continue;
        for (const auto &exception : selected) {
          if (parameter.same_as(exception)) {
            indices.push_back(i);
            break;
          }
        }
      }
      if (!indices.empty())
        function = WithAttr(function, "tl.may_alias_param_indices", indices);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.HoistNonRestrictParams");
}
} // namespace transform

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.transform.NormalizeAliasAnnotations",
           transform::NormalizeAliasAnnotations)
      .def("tl.transform.HoistNonRestrictParams",
           transform::HoistNonRestrictParams);
}
} // namespace tvm::tl
