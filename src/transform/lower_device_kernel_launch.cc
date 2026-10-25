#include "compat/expr.h"

#include <tvm/backend/cuda/attr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>

namespace tvm::tl {
namespace {

struct ClusterLaunch {
  int64_t num_parameters;
  ffi::Array<PrimExpr> dimensions;
};

using ClusterMap = std::unordered_map<std::string, ClusterLaunch>;

const ffi::Array<ffi::String> &ClusterTags() {
  static const ffi::Array<ffi::String> tags{
      "clusterCtaIdx.x", "clusterCtaIdx.y", "clusterCtaIdx.z"};
  return tags;
}

const ffi::Array<ffi::String> &ClusterFields() {
  static const ffi::Array<ffi::String> fields{"cluster.x", "cluster.y",
                                              "cluster.z"};
  return fields;
}

bool HasCluster(const ffi::Array<ffi::String> &fields, bool modern) {
  auto cluster = modern ? ClusterFields() : ClusterTags();
  size_t count = 0;
  for (const auto &field : fields) {
    for (const auto &expected : cluster)
      count += field == expected;
  }
  TVM_FFI_CHECK(count == 0 || count == 3, ValueError)
      << "Incomplete CUDA cluster launch dimensions";
  return count == 3;
}

class ClusterLaunchLowerer : public tirx::StmtExprMutator {
public:
  explicit ClusterLaunchLowerer(const ClusterMap &clusters)
      : clusters_(clusters) {}

private:
  ffi::UnchangedOr<Expr> Mutate_(const CallNode *node,
                                 ffi::InplaceMode mode) final {
    Expr original = ffi::GetRef<Expr>(node);
    auto call = StmtExprMutator::Mutate_(node, mode)
                    .ValueOrUnchanged(original)
                    .as_or_throw<Call>();
    if (!call->op.same_as(tirx::call_ffi_kernel_op()))
      return call;
    const auto *symbol = call->args[0].as<StringImmNode>();
    if (!symbol)
      return call;
    auto found = clusters_.find(std::string(symbol->value));
    if (found == clusters_.end())
      return call;
    const auto &cluster = found->second;
    const auto *attributes = call->attrs.value().as<tirx::CallFFIKernelAttr>();
    TVM_FFI_CHECK(attributes != nullptr, TypeError)
        << "Missing native kernel launch metadata";
    bool modern = !attributes->launch_fields.empty();
    if (HasCluster(modern ? attributes->launch_fields
                          : attributes->launch_params,
                   modern)) {
      return call;
    }
    TVM_FFI_CHECK_GE(call->args.size(),
                     static_cast<size_t>(cluster.num_parameters + 1),
                     ValueError);
    auto metadata = ffi::make_object<tirx::CallFFIKernelAttr>(*attributes);
    ffi::Array<Expr> arguments;
    if (modern) {
      arguments = call->args;
      for (const auto &dimension : cluster.dimensions)
        arguments.push_back(dimension);
      for (const auto &field : ClusterFields())
        metadata->launch_fields.push_back(field);
    } else {
      arguments = ffi::Array<Expr>(
          call->args.begin(), call->args.begin() + cluster.num_parameters + 1);
      for (const auto &dimension : cluster.dimensions)
        arguments.push_back(dimension);
      arguments.insert(arguments.end(),
                       call->args.begin() + cluster.num_parameters + 1,
                       call->args.end());
      metadata->launch_params = ClusterTags();
      for (const auto &tag : attributes->launch_params)
        metadata->launch_params.push_back(tag);
    }
    return Call(call->ty, call->op, arguments, Attrs(metadata), call->ty_args,
                call->loc);
  }

  const ClusterMap &clusters_;
};

} // namespace

namespace transform {

tvm::transform::Pass LowerDeviceKernelLaunch() {
  auto pass = [](IRModule module, tvm::transform::PassContext) {
    module = tirx::transform::LowerDeviceKernelLaunch()(module);
    ClusterMap clusters;
    IRModule updates;
    for (const auto &[global, base] : module->functions) {
      auto optional = base.as<tirx::Function>();
      if (!optional)
        continue;
      auto function = optional.value();
      auto target = function->GetAttr<Target>("target");
      auto requested = function->GetAttr<ffi::Array<ffi::Any>>("cluster_dims");
      if (!requested || !target || target.value()->kind->name != "cuda" ||
          target.value()->GetHost()) {
        continue;
      }
      TVM_FFI_CHECK_EQ(requested.value().size(), 3, ValueError)
          << "CUDA clusters need three dimensions";
      ffi::Array<PrimExpr> dimensions;
      for (const auto &value : requested.value()) {
        auto dimension = GetIntAnnotation(value);
        TVM_FFI_CHECK(dimension && *dimension > 0 && *dimension <= INT32_MAX,
                      ValueError)
            << "CUDA cluster dimensions must be positive int32 constants";
        dimensions.push_back(IntImm::Int32(*dimension));
      }
      std::string symbol = function->GetAttr<ffi::String>("global_symbol")
                               .value_or(global->name_hint);
      clusters.emplace(
          symbol, ClusterLaunch{static_cast<int64_t>(function->params.size()),
                                dimensions});
      if (auto fields =
              function->GetAttr(tvm::backend::cuda::attr::kLaunchFields)) {
        if (!HasCluster(fields.value(), true)) {
          auto expanded = fields.value();
          for (const auto &field : ClusterFields())
            expanded.push_back(field);
          function =
              WithAttr(std::move(function),
                       tvm::backend::cuda::attr::kLaunchFields.name, expanded);
        }
      } else {
        auto tags = function->GetAttr(tirx::attr::kKernelLaunchParams)
                        .value_or(ffi::Array<ffi::String>{});
        if (!HasCluster(tags, false)) {
          auto expanded = ClusterTags();
          for (const auto &tag : tags)
            expanded.push_back(tag);
          function = WithAttr(std::move(function),
                              tirx::attr::kKernelLaunchParams.name, expanded);
        }
      }
      updates->Add(global, function);
    }
    module.CopyOnWrite()->Update(updates);
    updates = IRModule();
    for (const auto &[global, base] : module->functions) {
      if (auto optional = base.as<tirx::Function>()) {
        auto function = optional.value();
        if (!function->body)
          continue;
        auto lowerer = ffi::make_object<ClusterLaunchLowerer>(clusters);
        SeqStmt body = function->body.value();
        function.CopyOnWrite()->body =
            lowerer->Mutate(body).ValueOrUnchanged(body);
        updates->Add(global, function);
      }
    }
    module.CopyOnWrite()->Update(updates);
    return module;
  };
  return tvm::transform::CreateModulePass(pass, 0,
                                          "tl.LowerDeviceKernelLaunch");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerDeviceKernelLaunch",
                                   LowerDeviceKernelLaunch);
}

} // namespace transform
} // namespace tvm::tl
