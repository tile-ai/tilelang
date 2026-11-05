/*!
 * \file upstream_pass_bridge.cc
 * \brief Explicit internal boundaries of the compiler assembled with TileLang.
 */
#include "upstream_pass_bridge.h"
#include "size_var_defaults.h"

#include <tvm/backend/cuda/attr.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/unique_name_supply.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <functional>
#include <unordered_map>
#include <unordered_set>

// These are existing, non-anonymous functions in the *same compiler*, not a
// separately linked C++ ABI. Their declarations are intentionally concentrated
// here. The build's provider manifest pins the upstream source and compiler.
//
// Upstream: df86f98793888a9f4f92fc0d138345f58258e64f
// src/tirx/transform/split_host_device.cc:
//   AnnotateDeviceRegionsForSplit, SplitHostDevice, LowerDeviceKernelLaunches
// src/tirx/transform/lower_thread_allreduce.cc:
//   detail::LowerThreadAllreduce
//
// Rebase review must check these signatures and stage invariants together with
// the native public SplitHostDevice composition. No symbol is found via dlsym,
// exported as a new upstream API, or copied from the implementation bodies.
namespace tvm::tirx {
Function AnnotateDeviceRegionsForSplit(Function function);
Function SplitHostDevice(Function function, IRModule *device_module,
                         std::function<GlobalVar()> var_supply);
IRModule LowerDeviceKernelLaunches(IRModule module);
namespace detail {
Function LowerThreadAllreduce(Function function);
}
} // namespace tvm::tirx

namespace tvm::tl::upstream_pass_bridge {
namespace {

class PendingLaunchStager final : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *region,
                            InplaceMode mode) final {
    if (region->op->name != "tirx.device_scope" ||
        !region->attrs->dict.count(backend::cuda::attr::kLaunchFields.name))
      return StmtExprMutator::Mutate_(region, mode);
    auto attributes = region->attrs->dict;
    attributes.erase(backend::cuda::attr::kLaunchFields.name);
    attributes.erase(backend::cuda::attr::kKernelAttrs.name);
    attributes.erase(backend::cuda::attr::kSmemRequired.name);
    attributes.erase("tl.derive_launch_geometry");
    // Config operands are host uses, not captures of the outlined kernel.
    // Removing them only from the inner native boundary prevents the native
    // extractor from finalizing call_ffi_kernel before TileLang device passes.
    RegionStmt native(region->op, {}, region->body_params,
                      DictAttrs(attributes), region->body, region->result_vars,
                      region->loc);
    return RegionStmt(PendingCudaLaunchOp(), region->args, {}, region->attrs,
                      SeqStmt(native), {}, region->loc);
  }
};

ffi::Array<Var> PendingBodyParams(const CallObj *) { return {}; }

void ValidatePendingLaunch(const RegionStmtObj *region) {
  TVM_FFI_CHECK(region->body_params.empty() && region->result_vars.empty(),
                ValueError)
      << "Pending CUDA launch has no body parameters or results";
  auto fields =
      region->attrs->dict.Get(backend::cuda::attr::kLaunchFields.name);
  TVM_FFI_CHECK(fields &&
                    fields->as_or_throw<ffi::Array<ffi::String>>().size() ==
                        region->args.size(),
                ValueError)
      << "Pending CUDA fields must describe every host configuration operand";
}

// The pinned native pass uses a GlobalVar's hint for newly created call_extern
// nodes, while the callee may export a different global_symbol. Repair only
// those newly materialized names, retaining native target dispatch and callee
// attributes. Existing explicit extern names are immutable StringImm leaves;
// keep strong references so nested call rewrites cannot confuse their identity.
class NativeExternSymbolResolver final : public tirx::StmtExprMutator {
public:
  explicit NativeExternSymbolResolver(const IRModule &module) {
    for (const auto &[symbol, function] : module->functions) {
      auto exported = function->GetAttr(tvm::attr::kGlobalSymbol);
      if (exported && *exported != symbol->name_hint)
        exported_names_.emplace(symbol->name_hint, *exported);
    }
    ffi::StructuralVisit(
        module,
        [&](const CallObj *call, ffi::StructuralVisitorObj *visitor)
            -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
          if (call->op.same_as(tirx::call_extern_op()) && !call->args.empty()) {
            original_names_.insert(call->args[0].get());
            retained_names_.push_back(call->args[0]);
          }
          return visitor->DefaultVisitExpected(call);
        });
  }

  using StmtExprMutator::Mutate_;
  UnchangedOr<Expr> Mutate_(const CallObj *node, InplaceMode mode) final {
    auto call = StmtExprMutator::Mutate_(node, mode)
                    .ValueOrUnchanged(GetRef<Expr>(node))
                    .as_or_throw<Call>();
    if (!call->op.same_as(tirx::call_extern_op()) || call->args.empty() ||
        original_names_.count(call->args[0].get()))
      return call;
    const auto *name = call->args[0].as<StringImmObj>();
    if (!name)
      return call;
    auto exported = exported_names_.find(name->value);
    if (exported == exported_names_.end())
      return call;
    auto arguments = call->args;
    arguments.Set(0, StringImm(exported->second, name->loc));
    return Call(call->ty, call->op, arguments, call->attrs, call->ty_args,
                call->loc);
  }

private:
  std::unordered_map<std::string, ffi::String> exported_names_;
  std::unordered_set<const ExprObj *> original_names_;
  ffi::Array<Expr> retained_names_;
};
} // namespace

const Op &PendingCudaLaunchOp() {
  static const Op op = Op::Get("tl.pending_cuda_launch");
  return op;
}

tvm::transform::Pass OutlineDeviceRegions() {
  auto outline = [](IRModule module, tvm::transform::PassContext) {
    // Match the native name supply across the complete module, including
    // helper functions that are not TIRx Functions.
    UniqueNameSupply names(
        module->functions.begin(), module->functions.end(),
        [](const auto &entry) { return entry.first->name_hint; });
    IRModule devices(ffi::Map<GlobalVar, BaseFunc>{});
    auto functions = module->functions;
    for (const auto &[symbol, base] : module->functions) {
      if (auto function = base.as<tirx::Function>()) {
        if ((*function)->GetAttr(tvm::attr::kDeclaration).value_or(false))
          continue;
        auto staged = tirx::AnnotateDeviceRegionsForSplit(function.value());
        auto pending = ffi::make_object<PendingLaunchStager>();
        auto body = staged->body;
        staged.CopyOnWrite()->body =
            pending->Mutate(body).ValueOrUnchanged(body);
        auto prefix = staged->GetAttr(tvm::attr::kGlobalSymbol)
                          .value_or(symbol->name_hint) +
                      "_kernel";
        ffi::Array<GlobalVar> extracted;
        auto supply = [&names, &prefix, &extracted]() {
          GlobalVar kernel_symbol(names->FreshName(prefix, false));
          extracted.push_back(kernel_symbol);
          return kernel_symbol;
        };
        // Outlining moves the source body, including operand-backed size
        // defaults, into a device function. The packed host still needs the
        // same defaults to decode a non-injective shape such as max(n, 0).
        // Retain them while host parameter identities are unchanged, before
        // ConvertSSA jointly renews definitions, types and carrier operands.
        auto size_defaults = CollectSizeVarDefaults(staged);
        const bool may_alias =
            !staged->GetAttr(tirx::attr::kNoAlias).value_or(true);
        auto tile_ops_lowered = staged->GetAttr<bool>("tl.tile_ops_lowered");
        Location source_location = staged->loc;
        const SeqStmt pre_split_body = staged->body;
        auto host = tirx::SplitHostDevice(std::move(staged), &devices, supply);
        if (!host->body.same_as(pre_split_body))
          host = WithSizeVarDefaults(std::move(host), size_defaults);
        functions.Set(symbol, host);
        // Native extraction gives each new kernel noalias=true. An explicit
        // source opt-out must survive; only this source's newly supplied
        // symbols inherit it, never another host's kernels or existing funcs.
        for (const GlobalVar &kernel_symbol : extracted) {
          auto kernel =
              devices->Lookup(kernel_symbol).as_or_throw<tirx::Function>();
          if (may_alias)
            kernel =
                WithAttr(std::move(kernel), tirx::attr::kNoAlias.name, false);
          // The native splitter does not know TileLang's lowering stages.
          // FinalizeAnnotations still needs this completed-stage fact when
          // consuming loop hints in this source's extracted device body.
          if (tile_ops_lowered)
            kernel = WithAttr(std::move(kernel), "tl.tile_ops_lowered",
                              tile_ops_lowered.value());
          if (!source_location.as<UnknownLocObj>())
            kernel.CopyOnWrite()->loc = source_location;
          devices->Update(kernel_symbol, kernel);
        }
      }
    }
    for (const auto &[symbol, function] : devices->functions)
      functions.Set(symbol, function);
    // Preserve module metadata and the input module. ConvertSSA is required
    // after outlining, but launch lowering must consume the later final body.
    module = IRModule(functions, module->attrs, module->global_infos);
    return tirx::transform::ConvertSSA()(std::move(module));
  };
  return tvm::transform::CreateModulePass(outline, 0,
                                          "tl.NativeOutlineDeviceRegions");
}

tvm::transform::Pass LowerDeviceKernelLaunch() {
  return tvm::transform::CreateModulePass(
      [](IRModule module, tvm::transform::PassContext) {
        auto symbols = ffi::make_object<NativeExternSymbolResolver>(module);
        module = tirx::LowerDeviceKernelLaunches(std::move(module));
        auto functions = module->functions;
        for (const auto &[symbol, base] : module->functions) {
          if (auto function = base.as<tirx::Function>()) {
            auto body = (*function)->body;
            auto resolved = symbols->Mutate(body, InplaceMode::kDisallow)
                                .ValueOrUnchanged(body);
            if (!resolved.same_as(body)) {
              function->CopyOnWrite()->body = resolved;
              functions.Set(symbol, *function);
            }
          }
        }
        return IRModule(functions, module->attrs, module->global_infos);
      },
      0, "tl.NativeLowerDeviceKernelLaunch");
}

tirx::Function LowerThreadAllreduce(tirx::Function function) {
  return tirx::detail::LowerThreadAllreduce(std::move(function));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.pending_cuda_launch",
        "Retain host configuration until device lowering is complete.")
      .signature(sig::var_args<Expr>("configuration"),
                 sig::call_attrs<DictAttrsObj>())
      .set_native_fn_attr<&PendingBodyParams>(op_attr::kRegionGetBodyParams)
      .set_native_fn_attr<&ValidatePendingLaunch>(op_attr::kRegionValidate);
}

} // namespace tvm::tl::upstream_pass_bridge
