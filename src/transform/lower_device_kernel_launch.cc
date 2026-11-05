/*!
 * \file lower_device_kernel_launch.cc
 * \brief Finalize TileLang device resources before native launch ABI lowering.
 */
#include "common/storage_size.h"
#include "ir/analyzer.h"
#include "op/native_utils.h"

#include <tvm/backend/cuda/attr.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "upstream_pass_bridge.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {
namespace {
using namespace tirx;
using namespace ffi;

// CUDA's packed runtime accepts int32 scalars, but has no bool converter.
// Legalize the actual device signature and both native call spellings together.
// The public host signature remains intact for the original boolean checks.
class CudaBooleanCallRewriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  std::unordered_map<const GlobalVarObj *, std::vector<size_t>> by_global;
  std::unordered_map<std::string, std::vector<size_t>> by_symbol;

  UnchangedOr<Expr> Mutate_(const CallObj *call, InplaceMode mode) final {
    auto mapped = StmtExprMutator::Mutate_(call, mode);
    Call current = mapped.IsUnchanged() ? GetRef<Call>(call)
                                        : AnyView(mapped).cast<Call>();
    const std::vector<size_t> *positions = nullptr;
    size_t offset = 0;
    if (auto global = current->op.as<GlobalVar>()) {
      if (auto found = by_global.find(global->get()); found != by_global.end())
        positions = &found->second;
    } else if (current->op.same_as(call_ffi_kernel_op()) &&
               !current->args.empty()) {
      if (auto name = current->args[0].as<StringImm>()) {
        if (auto found = by_symbol.find((*name)->value);
            found != by_symbol.end()) {
          positions = &found->second;
          offset = 1;
        }
      }
    }
    if (!positions)
      return mapped;
    auto arguments = current->args;
    for (size_t position : *positions) {
      size_t index = position + offset;
      TVM_FFI_CHECK_LT(index, arguments.size(), ValueError)
          << "CUDA boolean ABI argument is missing";
      auto value = arguments[index].as_or_throw<PrimExpr>();
      TVM_FFI_CHECK(value.ty() == PrimType::Bool(), TypeError)
          << "CUDA boolean ABI requires a boolean caller operand";
      arguments.Set(index, prim::Cast(PrimType::Int(32), value));
    }
    return Call(current->ty, current->op, arguments, current->attrs,
                current->ty_args, current->loc);
  }
};

tvm::transform::Pass LowerCudaBooleanParameters() {
  return tvm::transform::CreateModulePass(
      [](IRModule module, tvm::transform::PassContext) {
        auto calls = make_object<CudaBooleanCallRewriter>();
        auto functions = module->functions;
        for (const auto &[global, base] : module->functions) {
          auto optional = base.as<tirx::Function>();
          if (!optional)
            continue;
          const auto function = *optional;
          auto target = function->GetAttr(tvm::attr::kTarget);
          // Run after native launch classification. A global CUDA helper
          // without a launch keeps its original boolean interface; a real
          // kernel need not have carried is_global_func before lowering.
          if (!target || target.value()->GetHost() ||
              target.value()->GetTargetDeviceType() != kDLCUDA ||
              function->GetAttr(tvm::attr::kCallingConv, CallingConv::kDefault)
                      .value() != CallingConv::kDeviceKernelLaunch)
            continue;
          // Raw CUDA text owns its external bool8 signature. Its runtime
          // adapter uses an explicit normalized int32 conversion holder.
          if (function->GetAttr<String>("code_block_source"))
            continue;
          auto parameters = function->params;
          Map<Var, PrimExpr> replacements;
          std::vector<size_t> positions;
          for (size_t index = 0; index < parameters.size(); ++index) {
            auto parameter = parameters[index].as<PrimVar>();
            if (!parameter || parameter->ty() != PrimType::Bool())
              continue;
            PrimVar storage(std::string((*parameter)->name) + "_abi",
                            PrimType::Int(32), (*parameter)->loc);
            parameters.Set(index, storage);
            replacements.Set(*parameter, prim::Cast(PrimType::Bool(), storage));
            positions.push_back(index);
          }
          if (positions.empty())
            continue;
          auto replace =
              [&](const Var &variable) -> Expected<UnchangedOr<Any>> {
            if (auto replacement = replacements.Get(variable))
              return Any(*replacement);
            return Unchanged();
          };
          // Structural hooks rebuild changed parameter types and their uses
          // together. Function attributes are metadata to those hooks, so
          // remap their explicit variable references separately.
          auto attributes =
              StructuralMap<WalkOrder::kPreOrder>(function->attrs, replace)
                  .as_or_throw<DictAttrs>();
          auto changed =
              tirx::Function(parameters, function->body, function->ret_type,
                             attributes, function->loc);
          changed = StructuralMap<WalkOrder::kPreOrder>(changed, replace)
                        .as_or_throw<tirx::Function>();
          functions.Set(global, changed);
          calls->by_global.emplace(global.get(), positions);
          if (auto symbol = function->GetAttr(tvm::attr::kGlobalSymbol))
            calls->by_symbol.emplace(*symbol, positions);
        }
        if (calls->by_global.empty())
          return module;
        const auto device_updates = functions;
        for (const auto &[global, base] : device_updates) {
          if (auto function = base.as<tirx::Function>()) {
            auto body = (*function)->body;
            auto rewritten = calls->Mutate(body, InplaceMode::kDisallow)
                                 .ValueOrUnchanged(body);
            if (!rewritten.same_as(body)) {
              auto updated = *function;
              updated.CopyOnWrite()->body = rewritten;
              functions.Set(global, updated);
            }
          }
        }
        return IRModule(functions, module->attrs, module->global_infos);
      },
      0, "tl.LowerCudaBooleanParameters");
}

// These attributes are consumed by TileLang's runtime adapters and codegen.
// Keep their expressions in the final device-parameter coordinate system;
// native launch lowering substitutes those parameters at each host call site.
class LaunchMetadata : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;
  explicit LaunchMetadata(const tirx::Function &function)
      : function_(function), analyzer_(CreateTileLangAnalyzer()) {
    target_ = function->GetAttr<Target>(tvm::attr::kTarget).value();
    for (const Var &parameter : function->params)
      parameters_.insert(parameter.get());
    VisitNativePostOrder(function->body, [&](const ObjectRef &object) {
      if (const auto *bind = object.as<BindObj>()) {
        if (auto value = bind->value.as<PrimExpr>())
          definitions_.Set(bind->var, value.value());
      }
    });
  }

  Optional<VisitInterrupt> Visit_(const RegionStmtObj *region) final {
    if (region->op.same_as(launch_thread_op())) {
      String tag = region->args[0].as_or_throw<StringImm>()->value;
      const bool grid_only =
          target_->GetAttr<bool>("launch_grid_only").value_or(false);
      if (tag.find("vthread") != 0 &&
          (!grid_only || tag.find("blockIdx.") == 0)) {
        auto extent = Expand(region->args[1].as_or_throw<PrimExpr>());
        if (auto previous = extents.Get(tag)) {
          TVM_FFI_CHECK(analyzer_->CanProveEqual(previous.value(), extent),
                        ValueError)
              << "Incompatible launch extents for " << tag;
        } else {
          extents.Set(tag, extent);
        }
      }
    }
    return StmtExprVisitor::Visit_(region);
  }

  Optional<VisitInterrupt> Visit_(const BindObj *binding) final {
    const auto *call = binding->value.as<CallObj>();
    if (call && call->op.same_as(alloc_tensor_op())) {
      auto tensor = binding->var.as_or_throw<TensorVar>();
      if (tensor.scope().name() == "shared.dyn") {
        TVM_FFI_CHECK(!dynamic_bytes, ValueError)
            << "LowerDeviceKernelLaunch requires one final shared.dyn "
               "allocation; "
               "run MergeSharedMemoryAllocations first";
        dynamic_bytes = Expand(GetBufferStorageSizeBytes(tensor));
      }
    }
    return StmtExprVisitor::Visit_(binding);
  }

  PrimExpr Expand(PrimExpr expression) const {
    for (size_t count = 0; count <= definitions_.size(); ++count) {
      bool changed = false;
      expression = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
                       expression,
                       [&](const Var &variable) -> Expected<UnchangedOr<Any>> {
                         if (parameters_.count(variable.get()))
                           return Unchanged();
                         if (auto value = definitions_.Get(variable)) {
                           changed = true;
                           return Any(value.value());
                         }
                         return Unchanged();
                       })
                       .as_or_throw<PrimExpr>();
      if (!changed)
        break;
      TVM_FFI_CHECK(count < definitions_.size(), ValueError)
          << "Cyclic scalar definitions in launch metadata";
    }
    for (const Var &variable : UndefinedVars(expression)) {
      TVM_FFI_CHECK(parameters_.count(variable.get()), ValueError)
          << "Launch metadata refers to a device-local variable: " << variable;
    }
    VisitNativePostOrder(expression, [](const ObjectRef &object) {
      TVM_FFI_CHECK(!object.as<TensorLoadObj>(), ValueError)
          << "Launch metadata cannot read device memory";
      if (const auto *call = object.as<CallObj>()) {
        TVM_FFI_CHECK(tvm::SideEffect(GetRef<Call>(call)) <=
                          CallEffectKind::kPure,
                      ValueError)
            << "Launch metadata cannot evaluate device-dependent calls";
      }
    });
    return analyzer_->Simplify(expression);
  }

  Map<String, PrimExpr> extents;
  Optional<PrimExpr> dynamic_bytes;

private:
  tirx::Function function_;
  Target target_{UnsafeInit{}};
  sym::Analyzer analyzer_;
  Map<Var, PrimExpr> definitions_;
  std::unordered_set<const VarObj *> parameters_;
};

// Explicit native launch operands remain on the host until every device pass
// has completed. Derive TileLang grid/block geometry in this final callee's
// parameter coordinates, then substitute the actual operands at each call.
// Cluster dimensions and user-supplied streams/events are already host values.
class ConfiguredLaunchMetadata : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  explicit ConfiguredLaunchMetadata(IRModule module)
      : module_(std::move(module)), analyzer_(CreateTileLangAnalyzer()) {}

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *statement,
                            InplaceMode mode) final {
    if (!statement->op.same_as(upstream_pass_bridge::PendingCudaLaunchOp()))
      return StmtExprMutator::Mutate_(statement, mode);
    Stmt body = statement->body;
    while (const auto *sequence = body.as<SeqStmtObj>()) {
      TVM_FFI_CHECK_EQ(sequence->seq.size(), 1, ValueError)
          << "Pending CUDA launch must contain one outlined call";
      body = sequence->seq[0];
    }
    const auto *evaluate = body.as<EvaluateObj>();
    const auto *call = evaluate ? evaluate->value.as<CallObj>() : nullptr;
    TVM_FFI_CHECK(call && call->op.as<GlobalVarObj>(), ValueError)
        << "Pending CUDA launch must contain one outlined GlobalVar call";
    GlobalVar symbol = call->op.as_or_throw<GlobalVar>();
    auto function = device_updates.Get(symbol).value_or(
        module_->Lookup(symbol).as_or_throw<tirx::Function>());
    // The native configured-launch lowerer currently emits the CUDA ABI.
    // Other backends use ordinary GlobalVar calls, which this visitor leaves
    // unchanged. Reject an explicitly configured non-CUDA callee before that
    // lowerer could attach CUDA metadata to it.
    auto target = function->GetAttr<Target>(tvm::attr::kTarget);
    TVM_FFI_CHECK(target && target.value()->GetTargetDeviceType() == kDLCUDA,
                  ValueError)
        << "Explicit native CUDA launch configuration requires a CUDA device "
           "kernel";
    auto kernel_args = call->args;
    TVM_FFI_CHECK_EQ(kernel_args.size(), function->params.size(), ValueError)
        << "Native launch arguments must match the final device parameters";
    auto attrs = statement->attrs->dict;
    auto fields =
        attrs.at(backend::cuda::attr::kLaunchFields.name).cast<Array<String>>();
    auto values = statement->args;
    TVM_FFI_CHECK_EQ(fields.size(), values.size(), ValueError)
        << "CUDA launch fields must describe every configuration operand";
    std::unordered_map<std::string, size_t> positions;
    for (size_t i = 0; i < fields.size(); ++i) {
      TVM_FFI_CHECK(positions.emplace(fields[i], i).second, ValueError)
          << "Duplicate CUDA launch field: " << fields[i];
    }
    auto set = [&](String name, Expr value) {
      if (auto found = positions.find(name); found != positions.end()) {
        values.Set(found->second, value);
      } else {
        positions.emplace(name, fields.size());
        fields.push_back(name);
        values.push_back(value);
      }
    };
    auto require_flag = [&](String name) {
      if (auto found = positions.find(name); found != positions.end()) {
        auto value =
            analyzer_->Simplify(values[found->second].as_or_throw<PrimExpr>());
        TVM_FFI_CHECK(prim::IsOne(value), ValueError)
            << "The device kernel requires CUDA launch field " << name
            << " = 1";
      } else {
        set(name, IntImm::Int32(1));
      }
    };

    auto metadata = make_object<LaunchMetadata>(function);
    metadata->Visit(function->body);
    Map<Var, Expr> arguments;
    for (size_t i = 0; i < kernel_args.size(); ++i)
      arguments.Set(function->params[i], kernel_args[i]);
    auto in_caller = [&](PrimExpr expression) {
      return analyzer_->Simplify(
          ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
              expression,
              [&](const Var &variable) -> Expected<UnchangedOr<Any>> {
                if (auto argument = arguments.Get(variable))
                  return Any(*argument);
                return Unchanged();
              })
              .as_or_throw<PrimExpr>());
    };

    if (attrs.Get("tl.derive_launch_geometry")
            .value_or(Any(false))
            .cast<bool>()) {
      const auto &extents = metadata->extents;
      TVM_FFI_CHECK(!extents.empty(), ValueError)
          << "TileLang cluster launch requires final physical thread extents";
      for (const auto &prefix : {std::pair{"blockIdx.", "grid."},
                                 std::pair{"threadIdx.", "block."}}) {
        for (char axis : {'x', 'y', 'z'}) {
          PrimExpr extent = IntImm::Int32(1);
          if (auto value = extents.Get(std::string(prefix.first) + axis)) {
            extent = in_caller(*value);
          }
          set(std::string(prefix.second) + axis, analyzer_->Simplify(extent));
        }
      }
      attrs.erase("tl.derive_launch_geometry");
    }

    // These requirements are discovered after outlining. A configured launch
    // uses ordinary scalar fields for flags; the flag-only legacy tags must
    // never coexist with this ABI in runtime FunctionInfo.
    if (function->GetAttr<int64_t>("has_cuda_pdl_trigger").value_or(0) ||
        function->GetAttr<int64_t>("has_cuda_pdl_sync").value_or(0))
      require_flag("programmatic_stream_serialization");
    if (function->GetAttr<int64_t>("use_cooperative_groups").value_or(0))
      require_flag("cooperative");
    auto kernel_attrs = attrs.Get(backend::cuda::attr::kKernelAttrs.name)
                            .value_or(Any(Map<String, int64_t>{}))
                            .cast<Map<String, int64_t>>();
    if (auto requested = function->GetAttr(tirx::attr::kKernelLaunchParams)) {
      for (const String &tag : requested.value()) {
        if (tag == "tirx.use_programtic_dependent_launch")
          require_flag("programmatic_stream_serialization");
        else if (tag == "tirx.use_cooperative_launch")
          require_flag("cooperative");
        else if (tag == "tirx.use_required_block_dimension")
          kernel_attrs.Set("required_block_size", 1);
        else
          TVM_FFI_THROW(ValueError)
              << "Cannot mix native CUDA launch fields with legacy tag " << tag;
      }
    }
    Array<Stmt> prefix;
    Optional<PrimExpr> required_bytes;
    if (metadata->dynamic_bytes)
      required_bytes = in_caller(*metadata->dynamic_bytes);
    if (auto pool = attrs.Get(backend::cuda::attr::kSmemRequired.name)) {
      PrimExpr bytes = IntImm::Int64(pool->cast<int64_t>());
      required_bytes =
          required_bytes ? prim::Max(*required_bytes, bytes) : bytes;
    }
    if (required_bytes) {
      auto bytes = analyzer_->Simplify(*required_bytes);
      auto supplied = positions.find("dynamic_smem_bytes");
      if (supplied == positions.end()) {
        TVM_FFI_CHECK(!prim::IsZero(bytes) ||
                          attrs.count(backend::cuda::attr::kSmemRequired.name),
                      ValueError)
            << "A shared.dyn placeholder requires explicit dynamic_smem_bytes";
        set("dynamic_smem_bytes", bytes);
      } else {
        auto available = values[supplied->second].as_or_throw<PrimExpr>();
        auto condition = analyzer_->Simplify(available >= bytes);
        TVM_FFI_CHECK(!prim::IsZero(condition), ValueError)
            << "dynamic_smem_bytes is smaller than the final kernel allocation";
        if (!prim::IsOne(condition))
          prefix.push_back(
              AssertStmt(condition, StringImm("ValueError"),
                         {StringImm("dynamic_smem_bytes is smaller than the "
                                    "final kernel allocation")},
                         statement->loc));
      }
    }

    // Only compile-time block/cluster dimensions belong to the device. All
    // dynamic geometry and stream/event handles remain host Call operands.
    Array<PrimExpr> dimensions;
    for (const char *axis_prefix : {"block.", "cluster."}) {
      for (char axis : {'x', 'y', 'z'}) {
        PrimExpr dimension = IntImm::Int32(1);
        if (auto found = positions.find(std::string(axis_prefix) + axis);
            found != positions.end()) {
          auto value = analyzer_->Simplify(
              values[found->second].as_or_throw<PrimExpr>());
          dimension = value.as<IntImmObj>() ? value : IntImm::Int32(0);
        }
        dimensions.push_back(dimension);
      }
    }
    if (auto previous = device_updates.Get(symbol)) {
      auto prior_dimensions =
          (*previous)->GetAttr(backend::cuda::attr::kLaunchDimensions).value();
      TVM_FFI_CHECK(
          (*previous)->GetAttr(backend::cuda::attr::kLaunchFields).value() ==
                  fields &&
              (*previous)->GetAttr(backend::cuda::attr::kKernelAttrs).value() ==
                  kernel_attrs,
          ValueError)
          << "Incompatible CUDA launch schemas for one device function";
      for (size_t i = 0; i < dimensions.size(); ++i)
        if (!analyzer_->CanProveEqual(dimensions[i], prior_dimensions[i]))
          dimensions.Set(i, IntImm::Int32(0));
    }
    String name =
        function->GetAttr(tvm::attr::kGlobalSymbol).value_or(symbol->name_hint);
    function = WithAttrs(
        function,
        {{tvm::attr::kGlobalSymbol.name, name},
         {tvm::attr::kCallingConv.name, tvm::CallingConv::kDeviceKernelLaunch},
         {tirx::attr::kIsGlobalFunc.name, true},
         {backend::cuda::attr::kLaunchFields.name, fields},
         {backend::cuda::attr::kKernelAttrs.name, kernel_attrs},
         {backend::cuda::attr::kLaunchDimensions.name, dimensions}});
    function = WithoutAttr(function, tirx::attr::kKernelLaunchParams.name);
    device_updates.Set(symbol, function);
    auto launch = make_object<CallFFIKernelAttr>();
    launch->launch_fields = fields;
    launch->kernel_attrs = kernel_attrs;
    launch->num_kernel_args = kernel_args.size();
    Array<Expr> operands{StringImm(name)};
    operands.insert(operands.end(), kernel_args.begin(), kernel_args.end());
    operands.insert(operands.end(), values.begin(), values.end());
    prefix.push_back(Evaluate(Call(PrimType::Int(32), call_ffi_kernel_op(),
                                   operands, Attrs(launch), {}, statement->loc),
                              statement->loc));
    return SeqStmt(prefix, statement->loc);
  }

  Map<GlobalVar, tirx::Function> device_updates;

private:
  IRModule module_;
  sym::Analyzer analyzer_;
};
// Native outlining supplies the final argument order. Adapt its legacy CUDA
// schema to explicit fields so zero geometry reaches the runtime validator,
// rather than being interpreted as an omitted legacy dimension. Other device
// backends keep their schemas. All backends use TileLang's packed storage size.
class LegacyLaunchRewriter final : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  explicit LegacyLaunchRewriter(const IRModule &module) {
    for (const auto &[symbol, base] : module->functions) {
      auto function = base.as<tirx::Function>();
      if (!function)
        continue;
      auto target = (*function)->GetAttr(tvm::attr::kTarget);
      if (!target || (*target)->GetHost() ||
          (*target)->GetTargetDeviceType() == kDLCPU)
        continue;
      auto metadata = make_object<LaunchMetadata>(*function);
      metadata->Visit((*function)->body);
      if (metadata->dynamic_bytes ||
          ((*target)->GetTargetDeviceType() == kDLCUDA &&
           (*function)->GetAttr(tirx::attr::kKernelLaunchParams))) {
        String name = (*function)
                          ->GetAttr(tvm::attr::kGlobalSymbol)
                          .value_or(symbol->name_hint);
        TVM_FFI_CHECK(entries_
                          .emplace(name, Entry{symbol, *function,
                                               metadata->dynamic_bytes})
                          .second,
                      ValueError)
            << "Duplicate device symbol in launch metadata: " << name;
      }
    }
  }

  UnchangedOr<Expr> Mutate_(const CallObj *node, InplaceMode mode) final {
    const auto *attrs = node->attrs.as<CallFFIKernelAttr>();
    if (!node->op.same_as(call_ffi_kernel_op()) || !attrs ||
        !attrs->launch_fields.empty() || node->args.empty())
      return StmtExprMutator::Mutate_(node, mode);
    const auto *name = node->args[0].as<StringImmObj>();
    auto entry = name ? entries_.find(name->value) : entries_.end();
    if (entry == entries_.end())
      return StmtExprMutator::Mutate_(node, mode);
    const auto &function = entry->second.function;
    Array<Expr> operands = node->args;
    size_t position = 1 + function->params.size();
    TVM_FFI_CHECK_LE(position, operands.size(), ValueError);
    Map<Var, Expr> parameters;
    for (size_t i = 0; i < function->params.size(); ++i)
      parameters.Set(function->params[i], operands[1 + i]);
    for (const String &tag : attrs->launch_params) {
      if (tag == "tirx.use_programtic_dependent_launch" ||
          tag == "tirx.use_cooperative_launch" ||
          tag == "tirx.use_required_block_dimension")
        continue;
      TVM_FFI_CHECK_LT(position, operands.size(), ValueError);
      if (tag == "tirx.use_dyn_shared_memory" && entry->second.bytes) {
        auto bytes =
            ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
                *entry->second.bytes,
                [&](const Var &variable) -> Expected<UnchangedOr<Any>> {
                  if (auto argument = parameters.Get(variable))
                    return Any(*argument);
                  return Unchanged();
                })
                .as_or_throw<PrimExpr>();
        operands.Set(position, CreateTileLangAnalyzer()->Simplify(bytes));
      }
      ++position;
    }
    TVM_FFI_CHECK_EQ(position, operands.size(), ValueError)
        << "Legacy launch tags do not match the operand suffix";
    if (function->GetAttr(tvm::attr::kTarget).value()->GetTargetDeviceType() ==
        kDLCUDA) {
      auto tags = function->GetAttr(tirx::attr::kKernelLaunchParams).value();
      TVM_FFI_CHECK(tags == attrs->launch_params, ValueError)
          << "CUDA call and device function must share the launch schema";
      auto launch = make_object<CallFFIKernelAttr>();
      launch->num_kernel_args = function->params.size();
      launch->kernel_attrs =
          function->GetAttr(backend::cuda::attr::kKernelAttrs)
              .value_or(attrs->kernel_attrs);
      Array<Expr> arguments(operands.begin(),
                            operands.begin() + 1 + function->params.size());
      position = arguments.size();
      for (const String &tag : tags) {
        if (tag == "tirx.use_required_block_dimension") {
          launch->kernel_attrs.Set("required_block_size", 1);
          continue;
        }
        String field;
        Expr value{UnsafeInit{}};
        if (tag == "tirx.use_programtic_dependent_launch" ||
            tag == "tirx.use_cooperative_launch") {
          field = tag == "tirx.use_cooperative_launch"
                      ? "cooperative"
                      : "programmatic_stream_serialization";
          value = IntImm::Int32(1);
        } else {
          field = CudaField(tag);
          value = operands[position++];
        }
        launch->launch_fields.push_back(field);
        arguments.push_back(value);
      }
      auto device = WithAttr(function, backend::cuda::attr::kLaunchFields.name,
                             launch->launch_fields);
      if (!launch->kernel_attrs.empty())
        device = WithAttr(device, backend::cuda::attr::kKernelAttrs.name,
                          launch->kernel_attrs);
      device = WithoutAttr(device, tirx::attr::kKernelLaunchParams.name);
      device_updates.Set(entry->second.symbol, device);
      return Call(node->ty, node->op, arguments, Attrs(launch), node->ty_args,
                  node->loc);
    }
    return Call(node->ty, node->op, operands, node->attrs, node->ty_args,
                node->loc);
  }

  Map<GlobalVar, tirx::Function> device_updates;

private:
  static String CudaField(const String &tag) {
    if (tag == "tirx.use_dyn_shared_memory")
      return "dynamic_smem_bytes";
    for (const auto &prefix :
         {std::pair{"blockIdx.", "grid."}, std::pair{"threadIdx.", "block."},
          std::pair{"clusterCtaIdx.", "cluster."},
          std::pair{"preferredClusterCtaIdx.", "preferred_cluster."}}) {
      for (char axis : {'x', 'y', 'z'})
        if (tag == std::string(prefix.first) + axis)
          return std::string(prefix.second) + axis;
    }
    TVM_FFI_THROW(ValueError) << "Unsupported CUDA launch tag: " << tag;
  }

  struct Entry {
    GlobalVar symbol;
    tirx::Function function;
    Optional<PrimExpr> bytes;
  };
  std::unordered_map<std::string, Entry> entries_;
};
} // namespace

namespace transform {
tvm::transform::Pass LowerDeviceKernelLaunch() {
  auto record = tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule, tvm::transform::PassContext) {
        if (function->GetAttr(tvm::attr::kDeclaration).value_or(false) ||
            !function->GetAttr<Target>(tvm::attr::kTarget))
          return function;
        auto metadata = make_object<LaunchMetadata>(function);
        metadata->Visit(function->body);
        // Runtime launch data stays in the final body and Call operands.
        // No closed-metadata codec or parameter-index attribute is required.
        Array<String> launch_flags;
        if (auto flags = function->GetAttr(tirx::attr::kKernelLaunchParams)) {
          // Repeated lowering must preserve existing value-bearing tags too;
          // finalized calls no longer pass through the native GlobalVar
          // rewriter.
          launch_flags = flags.value();
        }
        auto add_flag = [&](String flag) {
          for (const String &existing : launch_flags)
            if (existing == flag)
              return;
          launch_flags.push_back(flag);
        };
        // Re-running the pass on finalized native calls must not reintroduce
        // the mutually exclusive legacy schema.
        if (!function->GetAttr(backend::cuda::attr::kLaunchFields)) {
          if (function->GetAttr<int64_t>("has_cuda_pdl_trigger").value_or(0) ||
              function->GetAttr<int64_t>("has_cuda_pdl_sync").value_or(0))
            add_flag("tirx.use_programtic_dependent_launch");
          if (function->GetAttr<int64_t>("use_cooperative_groups").value_or(0))
            add_flag("tirx.use_cooperative_launch");
        }
        if (!launch_flags.empty())
          function = WithAttr(function, tirx::attr::kKernelLaunchParams.name,
                              launch_flags);
        return function;
      },
      0, "tl.RecordDeviceLaunchMetadata");
  auto configure = tvm::transform::CreateModulePass(
      [](IRModule module, tvm::transform::PassContext) {
        auto mutator = make_object<ConfiguredLaunchMetadata>(module);
        IRModule result(module->functions, module->attrs, module->global_infos);
        for (const auto &[symbol, base] : module->functions) {
          auto function = base.as<tirx::Function>();
          if (!function || function.value()
                               ->GetAttr(tvm::attr::kDeclaration)
                               .value_or(false))
            continue;
          auto current = function.value();
          SeqStmt original = current->body;
          auto body = mutator->Mutate(original).ValueOrUnchanged(original);
          if (!body.same_as(original)) {
            current.CopyOnWrite()->body = body;
            result->Add(symbol, current, true);
          }
        }
        for (const auto &[symbol, function] : mutator->device_updates)
          result->Add(symbol, function, true);
        return result;
      },
      0, "tl.ConfigureFinalDeviceLaunch");
  auto finalize_launch = tvm::transform::CreateModulePass(
      [](IRModule module, tvm::transform::PassContext) {
        auto mutator = make_object<LegacyLaunchRewriter>(module);
        auto functions = module->functions;
        for (const auto &[symbol, base] : module->functions) {
          if (auto function = base.as<tirx::Function>()) {
            auto current = *function;
            auto original = current->body;
            auto rewritten =
                mutator->Mutate(original).ValueOrUnchanged(original);
            if (!rewritten.same_as(original)) {
              current.CopyOnWrite()->body = rewritten;
              functions.Set(symbol, current);
            }
          }
        }
        for (const auto &[symbol, function] : mutator->device_updates)
          functions.Set(symbol, function);
        return IRModule(functions, module->attrs, module->global_infos);
      },
      0, "tl.FinalizeDeviceLaunchABI");
  return tvm::transform::Sequential(
      {record, configure, upstream_pass_bridge::LowerDeviceKernelLaunch(),
       finalize_launch, LowerCudaBooleanParameters()},
      "tl.LowerDeviceKernelLaunch");
}
} // namespace transform
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerDeviceKernelLaunch",
                                   transform::LowerDeviceKernelLaunch);
}
} // namespace tvm::tl
