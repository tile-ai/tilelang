#include "common/attr.h"
#include "compat/expr.h"
#include "compat/metadata.h"
#include "op/builtin.h"
#include "op/prim.h"

#include <tvm/ir/analysis.h>
#include <tvm/target/target.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <unordered_set>

namespace tvm::tl {
namespace {

using namespace ffi;

const CallNode *ThreadPlaceholder(const Stmt &statement) {
  const auto *bind = statement.as<BindNode>();
  if (bind == nullptr)
    return nullptr;
  const auto *call = bind->value.as<CallNode>();
  return call && call->op.same_as(launch_thread_idx()) ? call : nullptr;
}

const RegionStmtNode *KernelScope(const SeqStmt &body) {
  if (body->seq.size() != 1)
    return nullptr;
  const auto *region = body->seq[0].as<RegionStmtNode>();
  return region && IsDeviceMainBlock(region) ? region : nullptr;
}

class KernelLaunchMaterializer : public tvm::tl::StmtExprMutator {
public:
  KernelLaunchMaterializer(bool grid_binding, bool thread_binding,
                           Optional<Array<PrimExpr>> default_threads,
                           Array<String> unsupported, Array<String> extra_tags,
                           String target_name)
      : grid_binding_(grid_binding), thread_binding_(thread_binding),
        default_threads_(std::move(default_threads)),
        unsupported_(std::move(unsupported)),
        extra_tags_(std::move(extra_tags)),
        target_name_(std::move(target_name)) {}

private:
  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                            InplaceMode inplace_mode) final {
    if (IsDeviceMainBlock(node)) {
      for (const String &key : unsupported_) {
        TVM_FFI_CHECK(!node->attrs->dict.count(key), ValueError)
            << "T.Kernel launch annotation `" << key
            << "` is not supported on target `" << target_name_ << "`";
      }
    }
    if (node->op->name != "tl.launch_grid") {
      return tvm::tl::StmtExprMutator::Mutate_(node, inplace_mode);
    }
    TVM_FFI_CHECK(node->args.size() == 2 && node->body_params.size() == 1 &&
                      node->result_vars.empty(),
                  ValueError)
        << "Malformed launch dimension";
    String tag = node->args[0].as_or_throw<StringImm>()->value;
    bool block_tag =
        tag == "blockIdx.x" || tag == "blockIdx.y" || tag == "blockIdx.z";
    if (!block_tag && std::find(extra_tags_.begin(), extra_tags_.end(), tag) ==
                          extra_tags_.end()) {
      return tvm::tl::StmtExprMutator::Mutate_(node, inplace_mode);
    }
    SeqStmt body =
        Mutate(node->body).ValueOrUnchanged(node->body).as_or_throw<SeqStmt>();
    PrimVar variable = node->body_params[0].as_or_throw<PrimVar>();
    PrimExpr extent =
        tl::CastValue(node->args[1].as_or_throw<PrimExpr>(), variable.ty());
    if (auto value = GetConstInt(extent)) {
      if (grid_binding_ && block_tag) {
        TVM_FFI_CHECK_GT(*value, 0, ValueError)
            << "Kernel grid extents must be positive";
      } else {
        TVM_FFI_CHECK_GE(*value, 0, ValueError)
            << "Kernel grid extents must be nonnegative";
      }
      if (*value == 0)
        return SeqStmt({});
    }
    if (grid_binding_) {
      return RegionStmt(tirx::launch_thread_op(), {StringImm(tag), extent},
                        {variable}, DictAttrs(), body, {}, node->loc);
    }
    return For(variable, tl::MakeConst(variable.ty(), 0), extent,
               tl::MakeConst(variable.ty(), 1), ForKind::kDefault, body,
               node->attrs->dict, node->loc);
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtNode *node,
                            InplaceMode inplace_mode) final {
    if (node->seq.empty() || ThreadPlaceholder(node->seq[0]) == nullptr) {
      return tvm::tl::StmtExprMutator::Mutate_(node, inplace_mode);
    }
    Array<Var> variables;
    std::vector<int> axes;
    std::unordered_set<int> seen;
    size_t prefix = 0;
    for (; prefix < node->seq.size(); ++prefix) {
      const CallNode *call = ThreadPlaceholder(node->seq[prefix]);
      if (call == nullptr)
        break;
      TVM_FFI_CHECK_EQ(call->args.size(), 1U, ValueError)
          << "Malformed thread placeholder";
      auto axis = GetConstInt(call->args[0]);
      TVM_FFI_CHECK(axis && *axis >= 0 && *axis < 3 &&
                        seen.insert(*axis).second,
                    ValueError)
          << "Thread placeholders require distinct constant axes in [0, 3)";
      variables.push_back(node->seq[prefix].as<BindNode>()->var);
      axes.push_back(*axis);
    }
    SeqStmt body(Array<Stmt>(node->seq.begin() + prefix, node->seq.end()),
                 node->loc);
    const RegionStmtNode *kernel = KernelScope(body);
    TVM_FFI_CHECK(kernel != nullptr, ValueError)
        << "Thread placeholders must immediately precede a kernel scope";
    auto requested =
        kernel->attrs.GetAttr<Array<PrimExpr>>(attr::kLaunchThreads);
    body = Mutate(body).ValueOrUnchanged(body).as_or_throw<SeqStmt>();
    if (!thread_binding_) {
      std::unordered_set<const VarNode *> used;
      for (const Var &variable : UndefinedVars(body))
        used.insert(variable.get());
      for (const Var &variable : variables) {
        TVM_FFI_CHECK(!used.count(variable.get()), ValueError)
            << "T.Kernel thread index `" << variable->name
            << "` is used on target `" << target_name_
            << "`, which has no SIMT thread bindings";
      }
      return body;
    }
    if (!requested.has_value())
      requested = default_threads_;
    TVM_FFI_CHECK(requested.has_value(), ValueError)
        << "T.Kernel did not specify threads= and target `" << target_name_
        << "` provides no default thread-block size";
    Array<PrimExpr> extents = requested.value();
    TVM_FFI_CHECK_LE(extents.size(), 3U, ValueError)
        << "Kernel threads require at most three axes";
    while (extents.size() < 3)
      extents.push_back(PrimExpr(1));
    const char *tags[] = {"threadIdx.x", "threadIdx.y", "threadIdx.z"};
    for (size_t position = variables.size(); position-- > 0;) {
      PrimVar variable = variables[position].as_or_throw<PrimVar>();
      PrimExpr extent = tl::CastValue(extents[axes[position]], variable.ty());
      auto constant = GetConstInt(extent);
      TVM_FFI_CHECK(constant && *constant > 0, ValueError)
          << "Kernel thread extents must be positive constants";
      body = SeqStmt(RegionStmt(tirx::launch_thread_op(),
                                {StringImm(tags[axes[position]]), extent},
                                {variable}, DictAttrs(), body, {}, node->loc));
    }
    return body;
  }

  bool grid_binding_;
  bool thread_binding_;
  Optional<Array<PrimExpr>> default_threads_;
  Array<String> unsupported_;
  Array<String> extra_tags_;
  String target_name_;
};

} // namespace

tvm::transform::Pass MaterializeKernelLaunch(
    bool lower_grid_binding, bool lower_thread_binding,
    ffi::Optional<ffi::Array<PrimExpr>> default_threads,
    ffi::Optional<ffi::Array<ffi::String>> unsupported_annotations,
    ffi::Optional<ffi::Array<ffi::String>> launch_dim_tags) {
  auto pass =
      [=](tirx::Function function, IRModule,
          tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    if (!function->body.has_value())
      return function;
    auto target = function->GetAttr<Target>("target");
    auto lowerer = ffi::make_object<KernelLaunchMaterializer>(
        lower_grid_binding, lower_thread_binding, default_threads,
        unsupported_annotations.value_or(ffi::Array<ffi::String>{}),
        launch_dim_tags.value_or(ffi::Array<ffi::String>{}),
        target.has_value() ? target.value()->kind->name
                           : ffi::String("<unbound>"));
    SeqStmt body = function->body.value();
    function.CopyOnWrite()->body =
        lowerer->Mutate(body).ValueOrUnchanged(body).as_or_throw<SeqStmt>();
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.MaterializeKernelLaunch");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.MaterializeKernelLaunch",
                                   MaterializeKernelLaunch);
}

} // namespace tvm::tl
