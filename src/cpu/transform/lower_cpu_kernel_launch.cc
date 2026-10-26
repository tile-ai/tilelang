#include "compat/expr.h"
#include "compat/metadata.h"
#include "compat/visit.h"
#include "cpu/op/builtin.h"
#include "transform/common/attr.h"

#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

#include <vector>

namespace tvm::tl {
namespace {

bool IsGridBinding(const RegionStmtNode *region) {
  if (!region || region->op->name != "tl.launch_grid" ||
      region->args.size() != 2) {
    return false;
  }
  const auto *tag = region->args[0].as<StringImmNode>();
  return tag && std::string(tag->value).starts_with("blockIdx.");
}

class CPULaunchAnnotationExtractor : public tvm::tl::StmtExprMutator {
public:
  ffi::Optional<PrimExpr> num_threads;
  ffi::Array<Stmt> allocations;
  std::unordered_set<const VarNode *> defined_inside;

private:
  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *, ffi::InplaceMode) final {
    return ffi::Unchanged();
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode) final {
    if (!IsDeviceMainBlock(node)) {
      return ffi::Unchanged();
    }
    auto annotation = node->attrs->dict.Get(attr::kCPUNumThreads);
    if (annotation) {
      auto count = GetIntAnnotation(annotation.value());
      TVM_FFI_CHECK(count && *count > 0 && *count <= INT32_MAX, ValueError)
          << attr::kCPUNumThreads << " must be a positive int32 constant";
      num_threads = IntImm::Int32(*count);
    }
    auto attributes = node->attrs->dict;
    attributes.erase(attr::kCPUNumThreads);
    ffi::Array<Stmt> statements;
    for (const Stmt &statement : node->body->seq) {
      const auto *binding = statement.as<BindNode>();
      const auto *call = binding ? binding->value.as<CallNode>() : nullptr;
      bool hoist = call && call->op.same_as(tirx::alloc_tensor_op());
      if (hoist) {
        for (const Var &variable : UndefinedVars(binding->value))
          hoist &= !defined_inside.count(variable.get());
      }
      if (hoist) {
        allocations.push_back(statement);
      } else {
        statements.push_back(statement);
        if (binding)
          defined_inside.insert(binding->var.get());
      }
    }
    return RegionStmt(node->op, node->args, node->body_params,
                      DictAttrs(attributes), SeqStmt(statements),
                      node->result_vars, node->loc);
  }
};

class CPUKernelLaunchLowerer : public tvm::tl::StmtExprMutator {
public:
  explicit CPUKernelLaunchLowerer(bool parallel) : parallel_(parallel) {}

private:
  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    if (!IsGridBinding(node)) {
      Stmt original = ffi::GetRef<Stmt>(node);
      Stmt body = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                      .ValueOrUnchanged(original);
      if (IsDeviceMainBlock(node)) {
        auto extractor = ffi::make_object<CPULaunchAnnotationExtractor>();
        body = extractor->Mutate(body).ValueOrUnchanged(body);
        if (!extractor->allocations.empty()) {
          extractor->allocations.push_back(body);
          body = SeqStmt(extractor->allocations);
        }
      }
      return body;
    }
    std::vector<RegionStmt> dimensions;
    const RegionStmtNode *current = node;
    while (IsGridBinding(current)) {
      dimensions.push_back(ffi::GetRef<RegionStmt>(current));
      current = current->body->seq.size() == 1
                    ? current->body->seq[0].as<RegionStmtNode>()
                    : nullptr;
    }
    SeqStmt body = dimensions.back()->body;
    auto extractor = ffi::make_object<CPULaunchAnnotationExtractor>();
    for (const RegionStmt &dimension : dimensions)
      for (const Var &variable : dimension->body_params)
        extractor->defined_inside.insert(variable.get());
    body = extractor->Mutate(body).ValueOrUnchanged(body);
    body = Mutate(body).ValueOrUnchanged(body);
    for (size_t position = dimensions.size(); position-- > 0;) {
      RegionStmt region = dimensions[position];
      auto attributes = region->attrs->dict;
      if (parallel_ && !attributes.count(attr::kCPUGridDim)) {
        attributes.Set(attr::kCPUGridDim, IntImm::Int32(position));
      }
      if (position == 0 && extractor->num_threads &&
          !attributes.count(attr::kCPUNumThreads)) {
        attributes.Set(attr::kCPUNumThreads, extractor->num_threads.value());
      }
      body = SeqStmt(RegionStmt(region->op, region->args, region->body_params,
                                DictAttrs(attributes), body,
                                region->result_vars, region->loc));
    }
    if (!extractor->allocations.empty()) {
      extractor->allocations.push_back(body);
      return SeqStmt(extractor->allocations);
    }
    return body;
  }

  bool parallel_;
};

} // namespace

namespace transform {

tvm::transform::Pass LowerCPUKernelLaunch() {
  auto pass =
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext context) -> ffi::Optional<tirx::Function> {
    if (function->body) {
      bool parallel = context->GetConfig<bool>(kCPUParallel).value_or(false);
      auto lowerer = ffi::make_object<CPUKernelLaunchLowerer>(parallel);
      SeqStmt body = function->body.value();
      function.CopyOnWrite()->body =
          lowerer->Mutate(body).ValueOrUnchanged(body);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.LowerCPUKernelLaunch");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cpu.transform.LowerCPUKernelLaunch",
                                   LowerCPUKernelLaunch);
}

} // namespace transform
} // namespace tvm::tl
