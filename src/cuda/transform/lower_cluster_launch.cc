/*!
 * \file lower_cluster_launch.cc
 * \brief Move per-kernel CUDA cluster requirements into native launch operands.
 */

#include "ir/annotations.h"
#include <tvm/ffi/extra/structural_visit.h>

#include <tvm/backend/cuda/attr.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <limits>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

// Geometry belongs to the final launch_thread regions, which can still change
// during device preparation. This marker has no expression operands. It asks
// LowerDeviceKernelLaunch to derive grid/block from the final outlined callee.
constexpr const char *kDeriveLaunchGeometry = "tl.derive_launch_geometry";

int64_t ConstantInteger(const Any &value, const char *name) {
  if (auto number = value.as<int64_t>())
    return *number;
  if (const auto *number = value.as<IntImmObj>()) {
    if (auto result = number->value.as<int64_t>())
      return *result;
  }
  TVM_FFI_THROW(ValueError) << name << " must contain constant integers";
}

class ClusterLaunchLowerer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  explicit ClusterLaunchLowerer(const tirx::Function &function)
      : owners_finished_(
            function->GetAttr<bool>("tl.shared_tmem_lowered").value_or(false)) {
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *statement,
                            InplaceMode mode) final {
    const auto *call = statement->value.as<CallObj>();
    if (!IsAnnotationCall(call))
      return StmtExprMutator::Mutate_(statement, mode);
    auto metadata = ReadAnnotations(call);
    if (!ConsumeClusterMetadata(metadata))
      return StmtExprMutator::Mutate_(statement, mode);
    // Pipeline lowering can materialize kernel hints as annotation statements.
    // Keep every unrelated field at its original position and lexical scope.
    return MakeAnnotation(metadata, statement->loc);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *region,
                            InplaceMode mode) final {
    const bool explicit_scope = region->op->name == "tirx.device_scope" ||
                                region->op->name == "tirx.device_entry";
    const bool starts_kernel =
        !kernel_ && (explicit_scope || region->op.same_as(launch_thread_op()));
    if (starts_kernel) {
      KernelState state;
      kernel_ = &state;
      Stmt result = StmtExprMutator::Mutate_(region, mode)
                        .ValueOrUnchanged(GetRef<Stmt>(region));
      kernel_ = nullptr;
      if (!state.cluster)
        return result;

      // A completion attribute alone must not hide new allocations or tile ops
      // inserted after their owners ran.
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
          result,
          [](const ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
            if (const auto *type = object.as<TensorTypeObj>()) {
              TVM_FFI_CHECK(type->scope.name() != "shared.tmem", ValueError)
                  << "LowerClusterLaunch requires LowerSharedTmem to consume "
                     "all "
                     "shared.tmem tensors first";
            }
            if (const auto *call = object.as<CallObj>()) {
              if (const auto *op = call->op.as<OpObj>()) {
                TVM_FFI_CHECK(op->name.find("tl.tileop.") != 0, ValueError)
                    << "LowerClusterLaunch requires LowerTileOp first";
              }
            }
            return ffi::WalkResult::Advance();
          });

      Array<String> fields;
      Array<Expr> values;
      Map<String, Any> attrs;
      if (explicit_scope) {
        const auto *rewritten = result.as<RegionStmtObj>();
        TVM_FFI_ICHECK(rewritten);
        values = rewritten->args;
        attrs = rewritten->attrs->dict;
        if (auto existing = attrs.Get(backend::cuda::attr::kLaunchFields.name))
          fields = existing->cast<Array<String>>();
      }
      TVM_FFI_CHECK_EQ(fields.size(), values.size(), ValueError)
          << "CUDA launch fields must describe every configuration operand";
      if (fields.empty())
        attrs.Set(kDeriveLaunchGeometry, true);
      for (size_t axis = 0; axis < 3; ++axis) {
        String name = std::string("cluster.") + char('x' + axis);
        int found = -1;
        for (size_t i = 0; i < fields.size(); ++i) {
          if (fields[i] == name) {
            TVM_FFI_CHECK_EQ(found, -1, ValueError)
                << "Duplicate CUDA launch field: " << name;
            found = static_cast<int>(i);
          }
        }
        if (found < 0) {
          fields.push_back(name);
          values.push_back(state.cluster.value()[axis]);
        } else {
          const auto *value = values[found].as<IntImmObj>();
          TVM_FFI_CHECK(
              value && value->value ==
                           state.cluster.value()[axis].as<IntImmObj>()->value,
              ValueError)
              << "Native launch configuration disagrees with cluster_dims";
        }
      }
      attrs.Set(backend::cuda::attr::kLaunchFields.name, fields);
      if (!attrs.count(backend::cuda::attr::kKernelAttrs.name))
        attrs.Set(backend::cuda::attr::kKernelAttrs.name,
                  Map<String, int64_t>{});
      if (explicit_scope) {
        const auto *rewritten = result.as<RegionStmtObj>();
        return RegionStmt(rewritten->op, values, rewritten->body_params,
                          DictAttrs(attrs), rewritten->body,
                          rewritten->result_vars, rewritten->loc);
      }
      return RegionStmt(Op::Get("tirx.device_scope"), values, {},
                        DictAttrs(attrs), SeqStmt(result), {}, region->loc);
    }
    if (region->op->name != "tl.attr")
      return StmtExprMutator::Mutate_(region, mode);
    auto fields = ReadAnnotations(region);
    if (fields.at("key").cast<String>() != "tl.kernel")
      return StmtExprMutator::Mutate_(region, mode);
    auto metadata = fields.at("value").cast<Map<String, Any>>();
    if (!ConsumeClusterMetadata(metadata))
      return StmtExprMutator::Mutate_(region, mode);
    fields.Set("value", metadata);
    auto encoded = EncodeAnnotations(fields);
    SeqStmt body = Mutate(region->body, mode).ValueOrUnchanged(region->body);
    // Retain the metadata region and its lexical allocation scope until the
    // normal finalizer processes all the unrelated kernel annotations.
    return RegionStmt(region->op, {encoded.payload}, region->body_params,
                      DictAttrs({{"schema_version", 1},
                                 {"annotation_schema", encoded.schema}}),
                      body, region->result_vars, region->loc);
  }

private:
  struct KernelState {
    Optional<Array<PrimExpr>> cluster;
  };
  bool ConsumeClusterMetadata(Map<String, Any> &metadata) {
    auto cluster = metadata.Get("cluster_dims");
    auto two_cta = metadata.Get("use_2cta");
    if (!cluster && !two_cta)
      return false;
    TVM_FFI_CHECK(kernel_, ValueError)
        << "Cluster metadata requires an enclosing physical kernel launch";
    TVM_FFI_CHECK(owners_finished_, ValueError)
        << "LowerClusterLaunch must run after LowerTileOp and LowerSharedTmem";
    TVM_FFI_CHECK(cluster, ValueError)
        << "use_2cta requires cluster_dims on its own kernel";
    auto dimensions = cluster->cast<Array<Any>>();
    TVM_FFI_CHECK_EQ(dimensions.size(), 3, ValueError)
        << "cluster_dims requires three dimensions";
    Array<PrimExpr> normalized;
    for (const Any &dimension : dimensions) {
      int64_t value = ConstantInteger(dimension, "cluster_dims");
      TVM_FFI_CHECK(value > 0 && value <= std::numeric_limits<uint32_t>::max(),
                    ValueError)
          << "cluster_dims must contain positive uint32 dimensions";
      normalized.push_back(IntImm::Int64(value));
    }
    if (two_cta) {
      int64_t enabled = two_cta->as<bool>()
                            ? (*two_cta->as<bool>() ? 1 : 0)
                            : ConstantInteger(*two_cta, "use_2cta");
      TVM_FFI_CHECK(enabled == 0 || enabled == 1, ValueError)
          << "use_2cta must be a boolean flag";
      if (enabled) {
        int64_t x = ConstantInteger(dimensions[0], "cluster_dims");
        int64_t y = ConstantInteger(dimensions[1], "cluster_dims");
        int64_t z = ConstantInteger(dimensions[2], "cluster_dims");
        TVM_FFI_CHECK(z == 1 && ((x == 2 && y == 1) || (x == 1 && y == 2)),
                      ValueError)
            << "use_2cta requires cluster_dims (2,1,1) or (1,2,1)";
      }
      metadata.erase("use_2cta");
    }
    if (kernel_->cluster) {
      TVM_FFI_CHECK(StructuralEqual()(kernel_->cluster.value(), normalized),
                    ValueError)
          << "Different cluster_dims cannot share one physical kernel";
    } else {
      kernel_->cluster = normalized;
    }
    metadata.erase("cluster_dims");
    return true;
  }

  KernelState *kernel_{nullptr};
  bool owners_finished_;
};
} // namespace

tvm::transform::Pass LowerClusterLaunch() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    auto target = function->GetAttr<Target>(tvm::attr::kTarget);
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false) || !target ||
        target.value()->kind->name != "cuda")
      return function;
    auto lowerer = make_object<ClusterLaunchLowerer>(function);
    SeqStmt original = function->body;
    auto body = lowerer->Mutate(original).ValueOrUnchanged(original);
    if (!body.same_as(original))
      function.CopyOnWrite()->body = body;
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LowerClusterLaunch");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.LowerClusterLaunch",
                                   LowerClusterLaunch);
}
} // namespace tvm::tl
