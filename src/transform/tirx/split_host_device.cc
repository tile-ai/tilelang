#include "compat/analyzer.h"
#include "compat/metadata.h"
#include <tvm/backend/cuda/attr.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/function.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/stmt.h>
#include <tvm/ir/transform.h>
#include <tvm/ir/unique_name_supply.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/sym/analyzer.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>

#include "tirx_compat/metadata.h"
#include "tirx_compat/transform.h"

#include "transform/tirx_passes.h"

namespace tvm::tl::tirx_passes {
using namespace tvm::tirx;
using tvm::tirx::transform::CreateFunctionPass;
using tvm::transform::Pass;
using tvm::transform::PassContext;
namespace {

class DeviceRegionAnnotater : public tvm::tl::StmtExprMutator {
public:
  using tvm::tl::StmtExprMutator::Mutate;
  using tvm::tl::StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView input,
                               InplaceMode inplace_mode) override {
    if (input.as<ExprNode>())
      return ffi::Unchanged();
    return tvm::tl::StmtExprMutator::Mutate(input, inplace_mode);
  }
  explicit DeviceRegionAnnotater(Target device_target)
      : device_target_(device_target) {}

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *op,
                            InplaceMode inplace_mode) final {
    static const Op device_scope = Op::Get("tirx.device_scope");
    if (op->op.same_as(device_scope)) {
      if (op->attrs->dict.count(tvm::attr::kTarget.name))
        return ffi::Unchanged();
      auto attrs = op->attrs->dict;
      attrs.Set(tvm::attr::kTarget.name, device_target_);
      return RegionStmt(op->op, op->args, op->body_params, DictAttrs(attrs),
                        op->body, op->result_vars, op->loc);
    }
    if (op->op.same_as(tirx::launch_thread_op())) {
      return RegionStmt(device_scope, {}, {},
                        DictAttrs({{tvm::attr::kTarget.name, device_target_}}),
                        ffi::GetRef<Stmt>(op));
    }
    return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode);
  }

private:
  Target device_target_;
};

Function AnnotateDeviceRegionsForSplit(Function func) {
  auto opt_target = func->GetAttr(tvm::attr::kTarget);
  TVM_FFI_ICHECK(opt_target) << "SplitHostDevice: Require the target attribute";
  Target target = opt_target.value();

  if (target->GetHost()) {
    auto mutator =
        ffi::make_object<DeviceRegionAnnotater>(target.WithoutHost());
    auto body_result =
        mutator->Mutate(func->body, func.unique() ? InplaceMode::kAllow
                                                  : InplaceMode::kDisallow);
    bool body_unchanged = body_result.UnchangedOrSameAs(func->body);
    auto body = std::move(body_result).ValueOrUnchanged(func->body);
    if (!body_unchanged) {
      func.CopyOnWrite()->body = body;
    }
  }
  return func;
}

Stmt MakeCudaKernelLaunch(const GlobalVar &symbol, Function *func,
                          ffi::Array<Expr> args, const RegionStmtNode *region);

class HostDeviceSplitter : public tvm::tl::StmtExprMutator {
public:
  using tvm::tl::StmtExprMutator::Mutate;
  using tvm::tl::StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView input,
                               InplaceMode inplace_mode) override {
    if (input.as<ExprNode>())
      return ffi::Unchanged();
    return tvm::tl::StmtExprMutator::Mutate(input, inplace_mode);
  }
  explicit HostDeviceSplitter(IRModule *device_mod,
                              std::function<GlobalVar()> var_supply,
                              Function cur_func)
      : device_mod_(device_mod), var_supply_(var_supply), cur_func_(cur_func) {}

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *op,
                            InplaceMode inplace_mode) final {
    static const Op device_scope = Op::Get("tirx.device_scope");
    if (op->op.same_as(device_scope)) {
      auto target = op->attrs->dict.Get(tvm::attr::kTarget.name);
      if (!target)
        return Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
      Target device_target = target.value().as_or_throw<Target>();
      return SplitDeviceFunc(op->body, device_target.WithoutHost(), op);
    }
    return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode);
  }

private:
  class KernelBodyRewriter : public tvm::tl::StmtExprMutator {
  public:
    using tvm::tl::StmtExprMutator::Mutate_;
    UnchangedOr<Stmt> Mutate_(const RegionStmtNode *op,
                              InplaceMode inplace_mode) final {
      static const Op device_scope = Op::Get("tirx.device_scope");
      if (op->op.same_as(device_scope)) {
        return Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
      }
      return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode);
    }
  };

  Stmt SplitDeviceFunc(Stmt body, Target device_target,
                       const RegionStmtNode *region) {
    auto [params, tensors_to_declare] =
        [&]() -> std::tuple<ffi::Array<Var>, ffi::Array<TensorVar>> {
      ffi::Array<Var> undefined = UndefinedVars(body);
      ffi::Array<TensorVar> tensors;
      for (const Var &var : undefined) {
        if (auto tensor = var.as<TensorVar>())
          tensors.push_back(tensor.value());
      }

      std::vector<Var> params{undefined.begin(), undefined.end()};
      std::sort(params.begin(), params.end(), [](const Var &a, const Var &b) {
        auto sort_key = [](const Var &var) {
          bool is_handle = var->ty.as<PtrTypeNode>() != nullptr ||
                           var->ty.as<TensorTypeNode>() != nullptr;
          return std::tuple{
              !is_handle,
              var->name,
          };
        };
        return sort_key(a) < sort_key(b);
      });
      return {params, tensors};
    }();

    ffi::Array<Var> kernel_params;
    ffi::Array<Expr> call_args;
    ffi::Map<Var, Var> tensor_data_params;
    auto kernel_rewriter = ffi::make_object<KernelBodyRewriter>();
    for (const Var &param : params) {
      if (param->ty.as<TensorTypeNode>()) {
        TensorVar tensor = param.as_or_throw<TensorVar>();
        TensorVar kernel_tensor(tensor->name, tensor.ty(), tensor->loc);
        Var data_param(tensor->name + "_ptr", tensor.ty()->DataPtrType());
        kernel_params.push_back(data_param);
        call_args.push_back(tensor.data());
        tensor_data_params.Set(param, data_param);
        kernel_rewriter->VarRemapSet(param, kernel_tensor);
      } else {
        kernel_params.push_back(param);
        call_args.push_back(param);
      }
    }
    body = kernel_rewriter->Mutate(body).ValueOrUnchanged(body);

    bool can_propagate_errors = [&]() {
      auto kind = device_target->GetTargetDeviceType();
      return kind == kDLCPU || kind == kDLExtDev || kind == kDLHexagon;
    }();
    IntImm success(PrimType::Int(32), 0);
    Type kernel_ret_type = Type::Missing();
    if (can_propagate_errors) {
      kernel_ret_type = PrimType::Int(32);
      body = SeqStmt({body, Return(success)});
    } else {
      kernel_ret_type = VoidType();
    }

    for (TensorVar buf : tensors_to_declare) {
      auto data_param = tensor_data_params.Get(buf);
      auto kernel_tensor = kernel_rewriter->VarRemapGet(buf);
      TVM_FFI_ICHECK(data_param.has_value())
          << "Undefined tensor " << buf->name
          << " was not captured as a kernel parameter";
      TVM_FFI_ICHECK(kernel_tensor != nullptr);
      body = SeqStmt({Bind(kernel_tensor.as_or_throw<TensorVar>(),
                           Call(kernel_tensor.as_or_throw<TensorVar>().ty(),
                                decl_tensor_op(), {data_param.value()}, {},
                                {kernel_tensor.as_or_throw<TensorVar>().ty()})),
                      std::move(body)});
    }
    Function device_func(kernel_params, SeqStmt(body), kernel_ret_type,
                         DictAttrs(), cur_func_->loc);
    device_func = WithAttrs(std::move(device_func),
                            {{tvm::attr::kTarget.name, device_target},
                             {tvm::tirx::attr::kNoAlias.name, true},
                             {tvm::tirx::attr::kIsGlobalFunc.name, true}});
    if (auto attributes = region->attrs.GetAttr<DictAttrs>("function_attrs")) {
      TVM_FFI_CHECK(UndefinedVars(attributes.value()).empty(), ValueError)
          << "Outlined function attributes must not capture local variables";
      for (const auto &[key, value] : attributes.value()->dict) {
        TVM_FFI_CHECK(key != tvm::attr::kTarget.name &&
                          key != tvm::attr::kCallingConv.name &&
                          key != tvm::attr::kGlobalSymbol.name,
                      ValueError)
            << "Device scope cannot override outlined function attribute "
            << key;
        device_func = WithAttr(std::move(device_func), key, value);
      }
    }
    if (auto config = region->attrs->dict.Get("backend_config")) {
      device_func =
          WithAttr(std::move(device_func), "backend_config", config.value());
    } else if (auto config =
                   cur_func_->GetAttr<ffi::String>("backend_config")) {
      device_func =
          WithAttr(std::move(device_func), "backend_config", config.value());
    }
    if (auto script_namespace =
            cur_func_->GetAttr(tvm::attr::kScriptNamespace)) {
      device_func =
          WithAttr(std::move(device_func), tvm::attr::kScriptNamespace.name,
                   script_namespace.value());
    }
    if (auto launch_params = cur_func_->GetAttr<ffi::Array<ffi::String>>(
            tvm::tirx::attr::kKernelLaunchParams.name)) {
      device_func = WithAttr(std::move(device_func),
                             tvm::tirx::attr::kKernelLaunchParams.name,
                             launch_params.value());
    }
    GlobalVar kernel_symbol_global = var_supply_();
    if (region->attrs->dict.count(
            tvm::backend::cuda::attr::kLaunchFields.name)) {
      Stmt launch = MakeCudaKernelLaunch(kernel_symbol_global, &device_func,
                                         call_args, region);
      (*device_mod_)->Add(kernel_symbol_global, device_func);
      return launch;
    }
    (*device_mod_)->Add(kernel_symbol_global, device_func);
    if (can_propagate_errors) {
      Var kernel_error_code("kernel_error_code", success.ty());
      Call kernel_call(success.ty(), kernel_symbol_global, call_args);
      AssertStmt assert_success(kernel_error_code.as_or_throw<PrimExpr>() ==
                                    success,
                                StringImm("RuntimeError"),
                                {StringImm("Error executing compute kernel")});
      return SeqStmt(ffi::Array<Stmt>{
          Bind(kernel_error_code, kernel_call.as_or_throw<PrimExpr>()),
          assert_success});

    } else {
      return Evaluate(Call(kernel_ret_type, kernel_symbol_global, call_args));
    }
  }

  IRModule *device_mod_;

  std::function<GlobalVar()> var_supply_;

  Function cur_func_;
};

Function SplitHostDevice(Function func, IRModule *device_mod,
                         std::function<GlobalVar()> var_supply) {
  auto splitter =
      ffi::make_object<HostDeviceSplitter>(device_mod, var_supply, func);

  auto body_result = splitter->Mutate(
      func->body, func.unique() ? InplaceMode::kAllow : InplaceMode::kDisallow);
  if (!body_result.UnchangedOrSameAs(func->body)) {
    func.CopyOnWrite()->body = std::move(body_result).ValueUnchecked();
  }

  return func;
}

namespace {

struct KernelInfo {
  explicit KernelInfo(Target target) : target(std::move(target)) {}

  Target target;

  ffi::String global_symbol;

  ffi::Array<Var> params;

  ffi::Array<ffi::String> launch_params;

  ffi::Array<PrimExpr> launch_args;
  ffi::Optional<PrimExpr> dynamic_smem_requirement;
};

class DeviceInfoCollector : public tvm::tl::StmtExprVisitor {
public:
  explicit DeviceInfoCollector(Target target) : info_(std::move(target)) {}

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprNode>())
      return std::nullopt;
    return tvm::tl::StmtExprVisitor::Visit(value);
  }
  static KernelInfo Collect(const GlobalVar &gvar, const Function &func,
                            bool allow_placeholder = false) {
    auto collector = ffi::make_object<DeviceInfoCollector>(
        func->GetAttr(tvm::attr::kTarget).value().WithoutHost());
    collector->info_.params = func->params;
    if (func->GetAttr<ffi::Array<ffi::String>>(
            tvm::backend::cuda::attr::kLaunchFields)) {
      collector->info_.global_symbol =
          func->GetAttr(tvm::attr::kGlobalSymbol).value_or(gvar->name_hint);
      return collector->info_;
    }

    if (auto requested = func->GetAttr(tvm::tirx::attr::kKernelLaunchParams)) {
      for (const ffi::String &tag : requested.value()) {
        if (tag == tvm::runtime::launch_param::kUseProgramaticDependentLaunch) {
          collector->use_programmatic_dependent_launch_ = true;
        } else if (tag == tvm::runtime::launch_param::kUseCooperativeLaunch) {
          collector->use_cooperative_launch_ = true;
        }
      }
    }

    collector->Visit(func->body);

    if (collector->use_programmatic_dependent_launch_) {
      collector->info_.launch_params.push_back(
          tvm::runtime::launch_param::kUseProgramaticDependentLaunch);
    }
    if (collector->use_cooperative_launch_) {
      collector->info_.launch_params.push_back(
          tvm::runtime::launch_param::kUseCooperativeLaunch);
    }

    if (!collector->dyn_shmem_size.has_value() &&
        collector->inferred_shmem_size_.has_value()) {
      const auto *inferred =
          collector->inferred_shmem_size_.value().as<IntImmNode>();
      TVM_FFI_ICHECK(allow_placeholder || !(inferred && inferred->value == 0))
          << "Function " << gvar->name_hint
          << " allocates dynamic shared memory with a placeholder extent but "
             "does not declare "
             "its size; use LaunchConfig.dynamic_smem_bytes or "
             "SMEMPool.commit().";
      collector->dyn_shmem_size = collector->inferred_shmem_size_;
    }
    if (collector->dyn_shmem_size) {
      collector->info_.launch_params.push_back(
          tvm::runtime::launch_param::kUseDynamicSharedMemoryTag);
    }

    collector->info_.global_symbol =
        func->GetAttr(tvm::attr::kGlobalSymbol).value_or(gvar->name_hint);

    for (const ffi::String &param : collector->info_.launch_params) {
      if (param == tvm::runtime::launch_param::kUseProgramaticDependentLaunch ||
          param == tvm::runtime::launch_param::kUseCooperativeLaunch ||
          param == tvm::runtime::launch_param::kUseRequiredBlockDimension) {
        continue;
      }
      collector->info_.launch_args.push_back(collector->GetArgument(param));
    }

    collector->info_.dynamic_smem_requirement = collector->dyn_shmem_size;
    return collector->info_;
  }

private:
  PrimExpr GetArgument(const ffi::String &launch_param) const {
    if (launch_param ==
        tvm::runtime::launch_param::kUseDynamicSharedMemoryTag) {
      TVM_FFI_ICHECK(dyn_shmem_size.has_value())
          << "Compute kernel requires launch parameter \"" << launch_param
          << "\", but Function has no dynamic shared memory requirement.";
      return dyn_shmem_size.value();
    }

    auto extent = thread_extent.Get(launch_param);
    TVM_FFI_ICHECK(extent) << "Compute kernel requires launch parameter \""
                           << launch_param
                           << "\", but Function does not contain a launch "
                              "region defining this axis";
    return extent.value();
  }

  ffi::Optional<VisitInterrupt> Visit_(const BindNode *op) final {
    if (const auto *call = op->value.as<CallNode>();
        call && call->op.same_as(alloc_tensor_op()))
      return DispatchAllocTensor(op, call);

    auto prim_value = op->value.as<PrimExpr>();
    if (!prim_value) {
      return tvm::tl::StmtExprVisitor::Visit_(op);
    }
    auto f_substitute =
        [this](const Var &var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = bind_map_.Get(var))
        return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    PrimExpr value = bind_map_.size()
                         ? ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
                               prim_value.value(), f_substitute)
                               .as_or_throw<PrimExpr>()
                         : prim_value.value();
    bind_map_.Set(op->var, value);
    return tvm::tl::StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtNode *op) final {
    if (op->op.same_as(tirx::launch_thread_op()) &&
        std::string(op->args[0].as_or_throw<StringImm>()->value)
                .rfind("vthread", 0) != 0) {
      ffi::String thread_tag = op->args[0].as_or_throw<StringImm>()->value;
      auto f_substitute =
          [this](const Var &var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (auto repl = bind_map_.Get(var))
          return ffi::Any(*std::move(repl));
        return ffi::Unchanged();
      };
      PrimExpr value = op->args[1].as_or_throw<PrimExpr>();
      if (bind_map_.size()) {
        value =
            ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(value, f_substitute)
                .as_or_throw<PrimExpr>();
      }
      if (auto previous = thread_extent.Get(thread_tag)) {
        sym::Analyzer analyzer;
        TVM_FFI_CHECK(
            tl::arith::CanProveEqual(analyzer, previous.value(), value),
            ValueError)
            << "Incompatible launch extents for " << thread_tag;
      } else {
        info_.launch_params.push_back(thread_tag);
        thread_extent.Set(thread_tag, value);
      }
    }
    auto outer_bindings = bind_map_;
    auto result = tvm::tl::StmtExprVisitor::Visit_(op);
    bind_map_ = std::move(outer_bindings);
    return result;
  }

  ffi::Optional<VisitInterrupt> DispatchAllocTensor(const BindNode *op,
                                                    const CallNode *call) {
    static const auto scope_shared_dyn =
        runtime::MemoryScope::Get("shared.dyn");

    runtime::MemoryScope scope =
        call->ty_args[0].as_or_throw<TensorType>()->scope;
    auto storage_scope = scope;
    if (storage_scope == scope_shared_dyn) {
      TVM_FFI_ICHECK(!saw_dyn_shared_alloc_)
          << "Only one dynamic shared memory allocation is allowed.";
      saw_dyn_shared_alloc_ = true;

      tvm::Tuple shape =
          tvm::Tuple(call->ty_args[0].as_or_throw<TensorType>()->shape);
      DLDataType dtype =
          call->ty_args[0].as_or_throw<TensorType>()->dtype->dtype;
      PrimType element_type(dtype);
      TVM_FFI_ICHECK_GT(shape->fields.size(), 0);
      PrimExpr dyn_size = IntImm::Int64(1);
      for (const auto &extent : shape->fields) {
        dyn_size *= extent.as_or_throw<PrimExpr>();
      }
      int64_t element_bits = element_type.bits() * element_type.lanes();
      dyn_size = prim::FloorDiv(dyn_size * IntImm::Int64(element_bits) +
                                    IntImm::Int64(7),
                                IntImm::Int64(8));
      if (bind_map_.size()) {
        auto f_substitute =
            [this](
                const Var &var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
          if (auto repl = bind_map_.Get(var))
            return ffi::Any(*std::move(repl));
          return ffi::Unchanged();
        };
        dyn_size = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(dyn_size,
                                                                 f_substitute)
                       .as_or_throw<PrimExpr>();
      }
      inferred_shmem_size_ = dyn_size;
    }
    return tvm::tl::StmtExprVisitor::Visit_(op);
  }

  KernelInfo info_;

  ffi::Map<ffi::String, PrimExpr> thread_extent;

  ffi::Optional<PrimExpr> dyn_shmem_size{std::nullopt};

  bool saw_dyn_shared_alloc_{false};

  ffi::Optional<PrimExpr> inferred_shmem_size_{std::nullopt};

  bool use_programmatic_dependent_launch_{false};
  bool use_cooperative_launch_{false};

  ffi::Map<Var, PrimExpr> bind_map_;
};

class KernelReturnNormalizer : public tvm::tl::StmtExprMutator {
public:
  using tvm::tl::StmtExprMutator::Mutate;
  using tvm::tl::StmtExprMutator::Mutate_;
  static Stmt Apply(const Stmt &stmt) {
    auto mutator = ffi::make_object<KernelReturnNormalizer>();
    return mutator->Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(stmt);
  }

private:
  UnchangedOr<Stmt> Mutate_(const ReturnNode *op,
                            InplaceMode inplace_mode) override {
    if (const auto *value = op->value.as<TupleNode>();
        value && value->fields.empty()) {
      return ffi::Unchanged();
    }
    auto as_int = op->value.as<IntImmNode>();
    TVM_FFI_ICHECK(as_int && as_int->value == 0)
        << "Device kernel may only contain a successful return, return () or "
           "return 0";
    return Return(tvm::Tuple(ffi::Array<Expr>{}), op->loc);
  }
};

class GlobalVarCallCollector : public tvm::tl::StmtExprVisitor {
public:
  static std::unordered_set<const GlobalVarNode *>
  Collect(const IRModule &mod) {
    auto collector = ffi::make_object<GlobalVarCallCollector>();
    for (const auto &[gvar, base_func] : mod->functions) {
      if (auto function = base_func.as<Function>()) {
        collector->Visit(function.value()->body);
      }
    }
    return collector->called_gvars_;
  }

private:
  using Parent = tvm::tl::StmtExprVisitor;

  ffi::Optional<VisitInterrupt> Visit_(const CallNode *op) final {
    if (auto *gvar = op->op.as<GlobalVarNode>()) {
      called_gvars_.insert(gvar);
    }
    return Parent::Visit_(op);
  }

  std::unordered_set<const GlobalVarNode *> called_gvars_;
};

} // namespace

Stmt MakeCudaKernelLaunch(const GlobalVar &symbol, Function *func,
                          ffi::Array<Expr> args, const RegionStmtNode *region) {
  auto fields =
      region->attrs->dict.at(tvm::backend::cuda::attr::kLaunchFields.name)
          .as_or_throw<ffi::Array<ffi::String>>();
  ffi::Array<Expr> values = region->args;
  auto kernel_attrs =
      region->attrs->dict.at(tvm::backend::cuda::attr::kKernelAttrs.name)
          .as_or_throw<ffi::Map<ffi::String, int64_t>>();
  auto info = DeviceInfoCollector::Collect(symbol, *func, true);
  ffi::Array<Stmt> host_stmts;
  auto required_bytes = info.dynamic_smem_requirement;
  if (auto pool_bytes = region->attrs->dict.Get(
          tvm::backend::cuda::attr::kSmemRequired.name)) {
    PrimExpr pool = IntImm::Int64(pool_bytes->cast<int64_t>());
    required_bytes =
        required_bytes ? prim::Max(required_bytes.value(), pool) : pool;
  }
  if (auto required = required_bytes) {
    sym::Analyzer analyzer;
    auto bytes = tl::arith::Simplify(analyzer, required.value());
    int index = -1;
    for (size_t i = 0; i < fields.size(); ++i) {
      if (fields[i] == "dynamic_smem_bytes")
        index = i;
    }
    if (index < 0) {
      TVM_FFI_CHECK(!prim::IsZero(bytes) ||
                        region->attrs->dict.count(
                            tvm::backend::cuda::attr::kSmemRequired.name),
                    ValueError)
          << "A shared.dyn placeholder requires "
             "LaunchConfig.dynamic_smem_bytes or "
             "SMEMPool.commit()";
      fields.push_back("dynamic_smem_bytes");
      values.push_back(bytes);
    } else {
      auto available = values[index].as_or_throw<PrimExpr>();
      auto condition = tl::arith::Simplify(analyzer, available >= bytes);
      TVM_FFI_CHECK(!prim::IsZero(condition), ValueError)
          << "LaunchConfig.dynamic_smem_bytes is smaller than the kernel "
             "allocation";
      if (!prim::IsOne(condition)) {
        host_stmts.push_back(
            AssertStmt(condition, StringImm("ValueError"),
                       {StringImm("LaunchConfig.dynamic_smem_bytes is smaller "
                                  "than the kernel allocation")}));
      }
    }
  }

  ffi::Array<PrimExpr> dimensions;
  for (const char *prefix : {"block.", "cluster."}) {
    for (char axis : {'x', 'y', 'z'}) {
      PrimExpr dimension = IntImm::Int32(1);
      for (size_t i = 0; i < fields.size(); ++i) {
        if (fields[i] == std::string(prefix) + axis) {
          sym::Analyzer analyzer;
          auto value =
              tl::arith::Simplify(analyzer, values[i].as_or_throw<PrimExpr>());

          dimension = value.as<IntImmNode>() ? value : IntImm::Int32(0);
        }
      }
      dimensions.push_back(dimension);
    }
  }
  *func = WithAttrs(
      std::move(*func),
      {{tvm::attr::kCallingConv.name, tvm::CallingConv::kDeviceKernelLaunch},
       {tvm::attr::kGlobalSymbol.name, symbol->name_hint},
       {tvm::backend::cuda::attr::kLaunchFields.name, fields},
       {tvm::backend::cuda::attr::kKernelAttrs.name, kernel_attrs},
       {tvm::backend::cuda::attr::kLaunchDimensions.name, dimensions}});
  auto attrs = ffi::make_object<CallFFIKernelAttr>();
  attrs->launch_fields = fields;
  attrs->kernel_attrs = kernel_attrs;
  attrs->num_kernel_args = args.size();
  ffi::Array<Expr> call_args{StringImm(symbol->name_hint)};
  call_args.insert(call_args.end(), args.begin(), args.end());
  call_args.insert(call_args.end(), values.begin(), values.end());
  host_stmts.push_back(Evaluate(
      Call(PrimType::Int(32), call_ffi_kernel_op(), call_args, Attrs(attrs))));
  return SeqStmt(host_stmts);
}

class DeviceKernelMutator : public tvm::tl::StmtExprMutator {
public:
  using tvm::tl::StmtExprMutator::Mutate;
  using tvm::tl::StmtExprMutator::Mutate_;
  using Parent = tvm::tl::StmtExprMutator;

  explicit DeviceKernelMutator(
      std::unordered_map<const GlobalVarNode *, KernelInfo> device_info_map)
      : device_info_map_(std::move(device_info_map)) {}

  Function RewriteKernelLaunchSite(const GlobalVar &gvar, Function func) {
    TVM_FFI_ICHECK(!current_target_.has_value());

    auto full_target = func->GetAttr(tvm::attr::kTarget).value();
    current_target_ = full_target.WithoutHost();
    if (full_target->GetHost().has_value()) {
      current_caller_host_target_ = full_target->GetHost().value();
    } else {
      current_caller_host_target_ = std::nullopt;
    }

    auto body_result = Mutate(func->body, InplaceMode::kDisallow);
    bool body_unchanged = body_result.UnchangedOrSameAs(func->body);
    auto body = std::move(body_result).ValueOrUnchanged(func->body);
    if (!body_unchanged) {
      func.CopyOnWrite()->body = body;
    }

    current_target_ = std::nullopt;
    current_caller_host_target_ = std::nullopt;
    return func;
  }

  Function UpdateKernelAttributes(const GlobalVar &gvar, Function func) const {
    bool is_kernel_launch = device_kernel_launch_.count(gvar.get());
    bool is_call_extern = extern_function_call_.count(gvar.get());
    TVM_FFI_ICHECK(!is_kernel_launch || !is_call_extern)
        << "Function " << gvar << " has multiple callees, "
        << "and would need to be lowered into a call_extern at some call "
           "sites, "
        << "and a device kernel launch at others.  "
        << "This case is not yet supported.";

    if (is_kernel_launch || is_call_extern) {
      func =
          WithAttr(std::move(func), tvm::tirx::attr::kIsGlobalFunc.name, true);
    }

    if (is_kernel_launch) {
      const auto &info = device_info_map_.at(gvar.get());

      {
        auto write_ptr = func.CopyOnWrite();
        write_ptr->ret_type = VoidType();
        write_ptr->body =
            KernelReturnNormalizer::Apply(write_ptr->body.value());
      }

      func = WithAttrs(
          std::move(func),
          {{tvm::attr::kCallingConv.name,
            tvm::CallingConv::kDeviceKernelLaunch},
           {tvm::tirx::attr::kKernelLaunchParams.name, info.launch_params},
           {tvm::attr::kGlobalSymbol.name, info.global_symbol}});

    } else if (is_call_extern && !func->GetAttr(tvm::attr::kGlobalSymbol)) {
      func = WithAttr(func, tvm::attr::kGlobalSymbol.name, gvar->name_hint);
    }

    return func;
  }

private:
  UnchangedOr<Expr> Mutate_(const CallNode *op,
                            InplaceMode inplace_mode) override {
    auto node = Parent::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Expr>(op))
                    .as_or_throw<Call>();

    auto *gvar = op->op.as<GlobalVarNode>();
    if (!gvar)
      return node;

    auto it = device_info_map_.find(gvar);
    TVM_FFI_ICHECK(it != device_info_map_.end())
        << "CallNode attempted subroutine call to " << gvar->name_hint
        << ", but " << gvar->name_hint << " did not appear within the IRModule";
    const KernelInfo &dev_info = it->second;

    auto callee_target = dev_info.target;

    bool callee_is_kernel = dev_info.launch_params.size() > 0;
    bool caller_is_host = current_caller_host_target_.has_value();

    auto caller_target = caller_is_host ? current_caller_host_target_.value()
                                        : current_target_.value();

    bool force_kernel_launch = callee_is_kernel && caller_is_host;

    if (!caller_is_host && !callee_is_kernel &&
        caller_target->kind->name == "cuda" &&
        callee_target->kind->name == "cuda") {
      return node;
    }

    if (!force_kernel_launch) {
      bool same_target = caller_target->str() == callee_target->str();
      if (same_target) {

        return node;
      }

      bool same_device_type = caller_target->GetTargetDeviceType() ==
                              callee_target->GetTargetDeviceType();
      if (same_device_type) {

        extern_function_call_.insert(gvar);
        ffi::Array<Expr> args;
        args.push_back(StringImm(dev_info.global_symbol));
        for (const Expr &arg : node->args) {
          args.push_back(arg);
        }
        Type ret_ty = IsVoidType(node->ty) ? PrimType::Void() : node->ty;
        return Call(ret_ty, call_extern_op(), args);
      }
    }

    TVM_FFI_ICHECK(dev_info.launch_params.defined())
        << "CallNode attempted kernel launch to " << gvar->name_hint
        << " on target " << dev_info.target << ", but subroutine "
        << gvar->name_hint
        << " did not have the tvm::tirx::attr::kKernelLaunchParams.name "
           "attribute "
        << "required for cross-target kernel launch";

    const ffi::Array<Expr> &args = node->args;
    ffi::Map<Var, PrimExpr> param_map = [&]() {
      ffi::Map<Var, PrimExpr> param_map;
      TVM_FFI_ICHECK_EQ(args.size(), dev_info.params.size())
          << "Function " << gvar->name_hint << " accepts "
          << dev_info.params.size()
          << " arguments as input, but is called using " << args.size()
          << " arguments";
      for (size_t i = 0; i < args.size(); i++) {
        if (auto prim_arg = args[i].as<PrimExpr>()) {
          param_map.Set(dev_info.params[i], prim_arg.value());
        }
      }
      return param_map;
    }();

    device_kernel_launch_.insert(gvar);

    ffi::Array<Expr> call_args;
    call_args.push_back(StringImm(dev_info.global_symbol));
    for (const Expr &arg : args) {
      call_args.push_back(arg);
    }
    auto f_substitute =
        [&param_map](
            const Var &var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = param_map.Get(var))
        return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    for (const auto &launch_arg : dev_info.launch_args) {
      call_args.push_back(ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
                              launch_arg, f_substitute)
                              .as_or_throw<PrimExpr>());
    }

    PrimType node_ty = IsVoidType(node->ty) ? PrimType::Void()
                                            : node->ty.as_or_throw<PrimType>();
    PrimType ret_ty = node_ty.IsVoid() ? PrimType::Int(32) : node_ty;

    auto attrs = ffi::make_object<CallFFIKernelAttr>();
    attrs->launch_params = dev_info.launch_params;
    return Call(ret_ty, call_ffi_kernel_op(), call_args, Attrs(attrs))
        .as_or_throw<PrimExpr>();
  }

  ffi::Optional<Target> current_target_;

  ffi::Optional<Target> current_caller_host_target_;
  std::unordered_map<const GlobalVarNode *, KernelInfo> device_info_map_;
  std::unordered_set<const GlobalVarNode *> device_kernel_launch_;
  std::unordered_set<const GlobalVarNode *> extern_function_call_;
};

IRModule LowerDeviceKernelLaunches(IRModule mod) {
  auto mutator = [&mod]() {
    std::unordered_set<const GlobalVarNode *> called_gvars =
        GlobalVarCallCollector::Collect(mod);
    std::unordered_map<const GlobalVarNode *, KernelInfo> device_info_map;
    for (const auto &[gvar, base_func] : mod->functions) {
      if (called_gvars.count(gvar.get())) {
        if (auto function = base_func.as<Function>()) {
          device_info_map.emplace(
              gvar.get(), DeviceInfoCollector::Collect(gvar, function.value()));
        }
      }
    }
    return ffi::make_object<DeviceKernelMutator>(std::move(device_info_map));
  }();

  {
    IRModule updates;
    for (const auto &[gvar, base_func] : mod->functions) {
      if (auto *ptr = base_func.as<FunctionNode>()) {
        auto function =
            mutator->RewriteKernelLaunchSite(gvar, ffi::GetRef<Function>(ptr));
        if (!function.same_as(base_func)) {
          updates->Add(gvar, function);
        }
      }
    }

    if (updates->functions.size()) {
      mod.CopyOnWrite()->Update(updates);
    }
  }

  {
    IRModule updates;
    for (const auto &[gvar, base_func] : mod->functions) {
      if (auto *ptr = base_func.as<FunctionNode>()) {
        auto function =
            mutator->UpdateKernelAttributes(gvar, ffi::GetRef<Function>(ptr));
        if (!function.same_as(base_func)) {
          updates->Add(gvar, function);
        }
      }
    }

    if (updates->functions.size()) {
      mod.CopyOnWrite()->Update(updates);
    }
  }

  return mod;
}

} // namespace

Pass OutlineDeviceRegions() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    UniqueNameSupply global_names(
        mod->functions.begin(), mod->functions.end(),
        [](const auto &kv) { return kv.first->name_hint; });

    IRModule device_mod = IRModule(ffi::Map<GlobalVar, BaseFunc>({}));
    IRModule updates = IRModule(ffi::Map<GlobalVar, BaseFunc>({}));

    for (const auto &[gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<Function>()) {
        Function func = opt.value();
        func = AnnotateDeviceRegionsForSplit(std::move(func));

        auto global_symbol = func->GetAttr(tvm::attr::kGlobalSymbol);
        auto name_prefix = global_symbol.value_or(gvar->name_hint);
        auto kernel_name = name_prefix + "_kernel";
        auto var_supply = [&global_names, &kernel_name]() -> GlobalVar {
          return GlobalVar(global_names->FreshName(kernel_name, false));
        };

        func = SplitHostDevice(std::move(func), &device_mod, var_supply);
        if (!func.same_as(base_func)) {
          updates->Add(gvar, func);
        }
      }
    }

    mod->Update(updates);
    mod->Update(device_mod);
    return tl::tirx_passes::ConvertSSA()(mod);
  };

  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.OutlineDeviceRegions");
}

Pass LowerDeviceKernelLaunch() {
  auto pass_func = [](IRModule mod, PassContext) {
    return LowerDeviceKernelLaunches(std::move(mod));
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.LowerDeviceKernelLaunch");
}

Pass SplitHostDevice() {
  return tvm::transform::Sequential(
      {OutlineDeviceRegions(), LowerDeviceKernelLaunch()},
      "tl.SplitHostDevice");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl._tirx_passes.OutlineDeviceRegions", OutlineDeviceRegions)
      .def("tl._tirx_passes.LowerDeviceKernelLaunch", LowerDeviceKernelLaunch)
      .def("tl._tirx_passes.SplitHostDevice", [] { return SplitHostDevice(); });
}

} // namespace tvm::tl::tirx_passes
