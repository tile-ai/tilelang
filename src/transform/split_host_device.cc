#include "compat/metadata.h"
#include "op/prim.h"
#include "transform/tirx_passes.h"

#include "common/assume.h"
#include "common/attr.h"
#include "compat/expr.h"
#include "compat/visit.h"
#include "cuda/runtime.h"
#include "op/builtin.h"

#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {

tvm::transform::Pass AnnotateDeviceRegions();

namespace {

using VariableSet = std::unordered_set<const VarNode *>;

bool IsHostEvaluable(const PrimExpr &condition, const VariableSet &parameters) {
  bool valid = condition.ty() == PrimType::Bool();
  PostOrderVisit(condition, [&](const ffi::ObjectRef &object) {
    if (const auto *variable = object.as<VarNode>()) {
      valid &= variable->ty.as<PrimTypeNode>() && parameters.count(variable);
    } else if (object.as<CallNode>() || object.as<TensorLoadNode>()) {
      valid = false;
    }
  });
  return valid;
}

AssertStmt MakeRuntimeCheck(const RegionStmtNode *node) {
  auto condition =
      node->attrs.GetAttr<PrimExpr>(attr::kAssumeRequiresRuntimeCheck).value();
  auto message = node->attrs.GetAttr<ffi::String>("tl.assume_message")
                     .value_or("Assumption failed");
  return AssertStmt(condition, StringImm("RuntimeError"), {StringImm(message)},
                    node->loc);
}

class DeviceBoundaryCollector : public tvm::tl::StmtExprMutator {
public:
  explicit DeviceBoundaryCollector(const VariableSet &parameters)
      : parameters_(parameters) {}

  ffi::Array<Stmt> runtime_checks;
  ffi::Map<ffi::String, ffi::Any> function_attrs;
  ffi::Optional<Call> l2_window;

private:
  ffi::UnchangedOr<Stmt> Mutate_(const EvaluateNode *node,
                                 ffi::InplaceMode mode) final {
    const auto *call = node->value.as<CallNode>();
    const auto *operation = call ? call->op.as<OpNode>() : nullptr;
    if (!operation || operation->name != "tl.l2_persistent_hint") {
      return tvm::tl::StmtExprMutator::Mutate_(node, mode);
    }
    TVM_FFI_CHECK(conditional_depth_ == 0, ValueError)
        << "L2 persistence must be configured unconditionally at kernel entry";
    auto tensor = call->args[0].as_or_throw<tirx::TensorVar>();
    for (const Var &variable : UndefinedVars(call->args)) {
      TVM_FFI_CHECK(parameters_.count(variable.get()), ValueError)
          << "L2 persistence must reference host-visible global storage and "
             "shapes";
    }
    TVM_FFI_CHECK(tensor.scope().name() == "global", ValueError)
        << "L2 persistence requires global tensor storage";
    if (l2_window) {
      TVM_FFI_CHECK(
          l2_window.value()->args[0].same_as(tensor) &&
              tl::StructuralEqual()(l2_window.value()->args[1], call->args[1]),
          ValueError)
          << "One CUDA kernel launch supports one L2 access-policy window";
    } else {
      l2_window = ffi::GetRef<Call>(call);
    }
    return SeqStmt({});
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *node,
                                 ffi::InplaceMode mode) final {
    ++conditional_depth_;
    auto result = tvm::tl::StmtExprMutator::Mutate_(node, mode);
    --conditional_depth_;
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const WhileNode *node,
                                 ffi::InplaceMode mode) final {
    ++conditional_depth_;
    auto result = tvm::tl::StmtExprMutator::Mutate_(node, mode);
    --conditional_depth_;
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const IfNode *node,
                                 ffi::InplaceMode mode) final {
    ++conditional_depth_;
    auto result = tvm::tl::StmtExprMutator::Mutate_(node, mode);
    --conditional_depth_;
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    auto attributes = node->attrs->dict;
    if (auto dimensions = attributes.Get("cluster_dims")) {
      if (auto previous = function_attrs.Get("cluster_dims")) {
        TVM_FFI_CHECK(
            tl::StructuralEqual()(previous.value(), dimensions.value()),
            ValueError)
            << "Conflicting cluster_dims annotations in one kernel";
      }
      function_attrs.Set("cluster_dims", dimensions.value());
      attributes.erase("cluster_dims");
    }
    if (auto condition =
            node->attrs.GetAttr<PrimExpr>(attr::kAssumeRequiresRuntimeCheck)) {
      if (conditional_depth_ == 0 &&
          IsHostEvaluable(condition.value(), parameters_)) {
        runtime_checks.push_back(MakeRuntimeCheck(node));
      }
      attributes.erase(attr::kAssumeRequiresRuntimeCheck);
      attributes.erase("tl.assume_message");
    }
    for (const char *key :
         {attr::kCodeBlockSource, attr::kCodeBlockEntryName}) {
      if (auto value = attributes.Get(key)) {
        auto source = value.value().cast<ffi::String>();
        if (auto previous = function_attrs.Get(key)) {
          TVM_FFI_CHECK(previous.value().cast<ffi::String>() == source,
                        ValueError)
              << "Conflicting " << key << " annotations in one kernel";
        }
        function_attrs.Set(key, source);
        attributes.erase(key);
      }
    }
    SeqStmt body = Mutate(node->body, mode).ValueOrUnchanged(node->body);
    return RegionStmt(node->op, node->args, node->body_params,
                      DictAttrs(attributes), body, node->result_vars,
                      node->loc);
  }

  const VariableSet &parameters_;
  int conditional_depth_{0};
};

class SourceKernelSignatureRewriter : public tvm::tl::StmtExprMutator {
public:
  SourceKernelSignatureRewriter(ffi::String entry, ffi::Array<Expr> arguments)
      : entry_(std::move(entry)), arguments_(std::move(arguments)) {}

private:
  ffi::UnchangedOr<Expr> Mutate_(const CallNode *node,
                                 ffi::InplaceMode mode) final {
    if (node->op.same_as(tirx::call_extern_op()) && !node->args.empty()) {
      if (auto name = node->args[0].as<StringImm>();
          name && name.value()->value == entry_) {
        ffi::Array<Expr> arguments{StringImm(entry_)};
        for (const Expr &argument : arguments_)
          arguments.push_back(argument);
        return Call(node->ty, node->op, arguments, node->attrs, node->ty_args,
                    node->loc);
      }
    }
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  ffi::String entry_;
  ffi::Array<Expr> arguments_;
};

class HostBoundaryPreparer : public tvm::tl::StmtExprMutator {
public:
  explicit HostBoundaryPreparer(const tirx::Function &function) {
    for (const Var &parameter : function->params) {
      parameters_.insert(parameter.get());
      if (auto tensor = parameter.as<tirx::TensorVar>())
        source_arguments_.push_back(tensor.value().data());
      else
        source_arguments_.push_back(parameter);
      for (const Var &variable : UndefinedVars(parameter->ty))
        parameters_.insert(variable.get());
    }
    for (const Var &variable : UndefinedVars(function->params)) {
      parameters_.insert(variable.get());
      source_arguments_.push_back(variable);
    }
    function_attrs_.Set("tirx.noalias",
                        function->GetAttr<bool>("tirx.noalias").value_or(true));
    auto launch_params = function->GetAttr(tirx::attr::kKernelLaunchParams)
                             .value_or(ffi::Array<ffi::String>{});
    if (function->GetAttr<bool>("use_cooperative_groups").value_or(false)) {
      bool present = false;
      for (const auto &parameter : launch_params)
        present |= parameter == "tirx.use_cooperative_launch";
      if (!present)
        launch_params.push_back("tirx.use_cooperative_launch");
    }
    if (function->GetAttr<bool>(attr::kHasGridSync).value_or(false)) {
      function_attrs_.Set("tirx.noalias", false);
      function_attrs_.Set(attr::kHasGridSync, true);
      bool present = false;
      for (const auto &parameter : launch_params)
        present |= parameter == "tirx.use_programtic_dependent_launch";
      if (!present)
        launch_params.push_back("tirx.use_programtic_dependent_launch");
    }
    if (!launch_params.empty())
      function_attrs_.Set("tirx.kernel_launch_params", launch_params);
    for (const char *key :
         {"cluster_dims", kSmemAlignmentMap, attr::kCodeBlockSource,
          attr::kCodeBlockEntryName}) {
      if (auto value = function->attrs->dict.Get(key)) {
        function_attrs_.Set(key, value.value());
      }
    }
  }

private:
  ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtNode *node,
                                 ffi::InplaceMode mode) final {
    auto previous = host_assumptions_;
    auto previous_parameters = parameters_;
    ffi::Array<Stmt> body;
    for (const Stmt &statement : node->seq) {
      body.push_back(Mutate(statement, mode).ValueOrUnchanged(statement));
      if (const auto *binding = statement.as<BindNode>()) {
        parameters_.insert(binding->var.get());
      }
      if (auto condition = GetAssumeExprInEvaluateForm(statement)) {
        if (IsHostEvaluable(condition.value(), parameters_)) {
          host_assumptions_.push_back(statement);
        }
      }
    }
    host_assumptions_ = std::move(previous);
    parameters_ = std::move(previous_parameters);
    return SeqStmt(body, node->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *node,
                                 ffi::InplaceMode mode) final {
    auto previous_parameters = parameters_;
    parameters_.insert(node->var.get());
    auto result = tvm::tl::StmtExprMutator::Mutate_(node, mode);
    parameters_ = std::move(previous_parameters);
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    if (node->op->name == "tirx.device_scope") {
      auto collector = ffi::make_object<DeviceBoundaryCollector>(parameters_);
      SeqStmt body =
          collector->Mutate(node->body, mode).ValueOrUnchanged(node->body);
      VariableSet captures;
      for (const Var &variable : UndefinedVars(body)) {
        captures.insert(variable.get());
      }
      ffi::Array<Stmt> assumptions;
      for (const Stmt &assumption : host_assumptions_) {
        bool captured = true;
        for (const Var &variable : UndefinedVars(assumption)) {
          captured &= captures.count(variable.get());
        }
        if (captured)
          assumptions.push_back(assumption);
      }
      assumptions.push_back(body);
      auto attributes = node->attrs->dict;
      auto function_attrs = function_attrs_;
      if (auto existing = node->attrs.GetAttr<DictAttrs>("function_attrs")) {
        for (const auto &[key, value] : existing.value()->dict)
          function_attrs.Set(key, value);
      }
      for (const auto &[key, value] : collector->function_attrs)
        function_attrs.Set(key, value);
      if (auto entry = function_attrs.Get(attr::kCodeBlockEntryName)) {
        auto signature = ffi::make_object<SourceKernelSignatureRewriter>(
            entry.value().cast<ffi::String>(), source_arguments_);
        assumptions.Set(assumptions.size() - 1,
                        signature->Mutate(body).ValueOrUnchanged(body));
      }
      attributes.Set("function_attrs", DictAttrs(function_attrs));
      auto statements = collector->runtime_checks;
      ffi::Optional<Var> l2_stream;
      if (collector->l2_window) {
        auto target = node->attrs.GetAttr<Target>("target");
        TVM_FFI_CHECK(target && target.value()->kind->name == "cuda",
                      ValueError)
            << "L2 persistence requires a CUDA device boundary";
        const auto &window = collector->l2_window.value();
        auto tensor = window->args[0].as_or_throw<tirx::TensorVar>();
        PrimExpr elements = IntImm::Int64(1);
        if (tensor.ty()->layout) {
          elements = tl::CastValue(
              tensor.ty()->layout.value().GetSpan(ffi::String("m")),
              PrimType::Int(64));
        } else {
          for (const PrimExpr &extent : tensor.shape()) {
            elements = elements * tl::CastValue(extent, PrimType::Int(64));
          }
        }
        int bits = tensor.dtype().bits() * tensor.dtype().lanes();
        PrimExpr bytes =
            prim::FloorDiv(elements * IntImm::Int64(bits) + IntImm::Int64(7),
                           IntImm::Int64(8));
        l2_stream = Var("l2_stream", PtrType::VoidPtrType());
        statements.push_back(
            Bind(l2_stream.value(),
                 Call(PtrType::VoidPtrType(), tirx::call_packed_op(),
                      {StringImm("runtime.get_cuda_stream")})));
        Expr pointer = Call(std::nullopt, tirx::tensor_data_ptr_op(), {tensor});
        statements.push_back(Evaluate(
            Call(PrimType::Int(32), tirx::call_packed_op(),
                 {StringImm(tvm_cuda_stream_set_access_policy_window), pointer,
                  bytes, window->args[1], l2_stream.value()})));
      }
      statements.push_back(RegionStmt(
          node->op, node->args, node->body_params, DictAttrs(attributes),
          SeqStmt(assumptions), node->result_vars, node->loc));
      if (l2_stream) {
        statements.push_back(Evaluate(
            Call(PrimType::Int(32), tirx::call_packed_op(),
                 {StringImm(tvm_cuda_stream_reset_access_policy_window),
                  l2_stream.value()})));
      }
      return SeqStmt(statements, node->loc);
    }
    Stmt original = ffi::GetRef<Stmt>(node);
    auto result = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                      .ValueOrUnchanged(original);
    if (node->attrs->dict.count(attr::kAssumeRequiresRuntimeCheck)) {
      auto region = result.as_or_throw<RegionStmt>();
      auto attributes = region->attrs->dict;
      attributes.erase(attr::kAssumeRequiresRuntimeCheck);
      attributes.erase("tl.assume_message");
      region.CopyOnWrite()->attrs = DictAttrs(attributes);
      return SeqStmt({MakeRuntimeCheck(node), region}, node->loc);
    }
    return result;
  }

  VariableSet parameters_;
  ffi::Array<Expr> source_arguments_;
  ffi::Map<ffi::String, ffi::Any> function_attrs_;
  ffi::Array<Stmt> host_assumptions_;
};

VariableSet ParameterSet(const tirx::Function &function, const char *key) {
  VariableSet result;
  if (auto annotation = function->GetAttr<ffi::Array<IntImm>>(key)) {
    for (const IntImm &value : annotation.value()) {
      auto index = value->value.as<int64_t>();
      TVM_FFI_CHECK(index && *index >= 0 &&
                        *index < static_cast<int64_t>(function->params.size()),
                    ValueError)
          << "Invalid " << key;
      result.insert(function->params[*index].get());
    }
  }
  return result;
}

class SourceKernelSymbolRewriter : public tvm::tl::StmtExprMutator {
public:
  explicit SourceKernelSymbolRewriter(
      ffi::Map<GlobalVar, GlobalVar> replacements)
      : replacements_(std::move(replacements)) {}

private:
  ffi::UnchangedOr<Expr> Mutate_(const GlobalVarNode *node,
                                 ffi::InplaceMode mode) final {
    if (auto replacement = replacements_.Get(ffi::GetRef<GlobalVar>(node)))
      return replacement.value();
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  ffi::Map<GlobalVar, GlobalVar> replacements_;
};

IRModule NameSourceKernels(IRModule module) {
  std::unordered_set<std::string> names;
  ffi::Map<GlobalVar, GlobalVar> replacements;
  for (const auto &[global, base] : module->functions) {
    auto function = base.as<tirx::Function>();
    ffi::String name = global->name_hint;
    if (function) {
      if (auto entry = function.value()->GetAttr<ffi::String>(
              attr::kCodeBlockEntryName)) {
        if (function.value()
                ->GetAttr<bool>("tirx.is_global_func")
                .value_or(false))
          name = entry.value();
      }
    }
    TVM_FFI_CHECK(names.insert(std::string(name)).second, ValueError)
        << "Duplicate external kernel symbol: " << name;
    if (name != global->name_hint)
      replacements.Set(global, GlobalVar(name, global->loc));
  }
  if (replacements.empty())
    return module;
  auto rewriter = ffi::make_object<SourceKernelSymbolRewriter>(replacements);
  ffi::Map<GlobalVar, BaseFunc> functions;
  for (const auto &[global, base] : module->functions) {
    BaseFunc updated = base;
    if (auto function = base.as<tirx::Function>()) {
      auto rewritten = function.value();
      if (rewritten->body) {
        auto body = rewritten->body.value();
        rewritten.CopyOnWrite()->body =
            rewriter->Mutate(body).ValueOrUnchanged(body);
      }
      updated = rewritten;
    }
    functions.Set(replacements.Get(global).value_or(global), updated);
  }
  return IRModule(functions, module->attrs, module->global_infos);
}

void PropagateParameterMetadata(
    const tirx::Function &host, const IRModule &module,
    const std::unordered_set<const GlobalVarNode *> &originals,
    IRModule updates) {
  auto non_restrict = ParameterSet(host, "tl.non_restrict_param_indices");
  auto read_only = ParameterSet(host, "tl.readonly_param_indices");
  std::unordered_map<const VarNode *, VariableSet> origins;
  for (const Var &parameter : host->params)
    origins[parameter.get()] = {parameter.get()};
  auto roots = [&](const Expr &expression) {
    VariableSet result;
    for (const Var &variable : UndefinedVars(expression)) {
      if (!variable->ty.as<PtrTypeNode>() &&
          !variable->ty.as<tirx::TensorTypeNode>())
        continue;
      if (auto found = origins.find(variable.get()); found != origins.end()) {
        result.insert(found->second.begin(), found->second.end());
      }
    }
    return result;
  };
  if (!host->body)
    return;
  PostOrderVisit(host->body.value(), [&](const ffi::ObjectRef &object) {
    if (const auto *bind = object.as<BindNode>()) {
      origins[bind->var.get()] = roots(bind->value);
    }
    const auto *call = object.as<CallNode>();
    const auto *symbol = call ? call->op.as<GlobalVarNode>() : nullptr;
    if (!symbol || originals.count(symbol))
      return;
    GlobalVar global = ffi::GetRef<GlobalVar>(symbol);
    auto function = module->Lookup(global).as_or_throw<tirx::Function>();
    TVM_FFI_CHECK_EQ(call->args.size(), function->params.size(), ValueError);
    ffi::Array<IntImm> non_restrict_indices;
    ffi::Array<IntImm> readonly_indices;
    for (size_t index = 0; index < call->args.size(); ++index) {
      auto parameters = roots(call->args[index]);
      bool may_alias = false;
      bool immutable = !parameters.empty();
      for (const auto *parameter : parameters) {
        may_alias |= non_restrict.count(parameter);
        immutable &= read_only.count(parameter);
      }
      if (may_alias)
        non_restrict_indices.push_back(IntImm::Int32(index));
      if (immutable)
        readonly_indices.push_back(IntImm::Int32(index));
    }
    function = WithAttr(std::move(function), "tl.non_restrict_param_indices",
                        non_restrict_indices);
    function = WithAttr(std::move(function), "tl.readonly_param_indices",
                        readonly_indices);
    if (!non_restrict_indices.empty()) {
      function = WithAttr(std::move(function), "tirx.noalias", false);
    }
    updates->Add(global, function, true);
  });
}

} // namespace

namespace transform {

tvm::transform::Pass SplitHostDevice() {
  auto prepare = tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
        if (function->body) {
          auto preparer = ffi::make_object<HostBoundaryPreparer>(function);
          SeqStmt body = function->body.value();
          function.CopyOnWrite()->body =
              preparer->Mutate(body).ValueOrUnchanged(body);
        }
        return function;
      },
      0, "tl.PrepareHostDeviceBoundary");
  auto pass = [prepare](IRModule module, tvm::transform::PassContext) {
    std::unordered_set<const GlobalVarNode *> original_functions;
    for (const auto &[global, function] : module->functions)
      original_functions.insert(global.get());
    module = AnnotateDeviceRegions()(module);
    module = prepare(module);
    module = tirx_passes::OutlineDeviceRegions()(module);
    module = NameSourceKernels(module);
    IRModule updates;
    for (const auto &[global, function] : module->functions) {
      if (original_functions.count(global.get())) {
        if (auto host = function.as<tirx::Function>()) {
          PropagateParameterMetadata(host.value(), module, original_functions,
                                     updates);
        }
      }
    }
    module.CopyOnWrite()->Update(updates);
    return module;
  };
  return tvm::transform::CreateModulePass(pass, 0, "tl.SplitHostDevice");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.SplitHostDevice",
                                   SplitHostDevice);
}

} // namespace transform
} // namespace tvm::tl
