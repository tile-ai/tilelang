/*!
 * \brief Hoist global buffer allocations to the top of the block (host side).
 * \file hoist_global_buffer_allocations.cc
 */

#include "compat/visit.h"

#include <tvm/ir/analysis.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

namespace tvm::tl {
namespace {

class DeviceAllocationCollector : public tirx::StmtExprMutator {
public:
  explicit DeviceAllocationCollector(const RegionStmt &region) {
    ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
        region,
        [&](const Var &variable,
            TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::WalkResult> {
          if (kind != kTVMFFIDefRegionKindNone) {
            definitions_.insert(variable.get());
          }
          return ffi::WalkResult::Advance();
        });
  }

  ffi::Array<Stmt> allocations;

private:
  ffi::UnchangedOr<Stmt> Mutate_(const BindNode *node,
                                 ffi::InplaceMode mode) final {
    auto tensor = node->var.as<tirx::TensorVar>();
    const auto *call = node->value.as<CallNode>();
    if (!tensor || tensor->scope().name() != "global" || !call ||
        !call->op.same_as(tirx::alloc_tensor_op())) {
      return StmtExprMutator::Mutate_(node, mode);
    }
    for (const Var &variable : UndefinedVars(node->value)) {
      TVM_FFI_CHECK(!definitions_.count(variable.get()), ValueError)
          << "Cannot hoist global tensor " << tensor.value()->name
          << " whose allocation depends on device-local variable "
          << variable->name;
    }
    allocations.push_back(ffi::GetRef<Stmt>(node));
    return SeqStmt({}, node->loc);
  }

  std::unordered_set<const VarNode *> definitions_;
};

class GlobalAllocationHoister : public tirx::StmtExprMutator {
private:
  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    bool device_region = node->op.same_as(tirx::launch_thread_op()) ||
                         node->op->name == "tirx.device_scope" ||
                         node->op->name == "tl.launch_grid";
    if (!device_region) {
      return StmtExprMutator::Mutate_(node, mode);
    }
    auto collector = ffi::make_object<DeviceAllocationCollector>(
        ffi::GetRef<RegionStmt>(node));
    Stmt original = ffi::GetRef<Stmt>(node);
    Stmt body = collector->Mutate(original, mode).ValueOrUnchanged(original);
    if (collector->allocations.empty()) {
      return body;
    }
    auto statements = collector->allocations;
    statements.push_back(body);
    return SeqStmt(statements, node->loc);
  }
};

} // namespace

namespace transform {

tvm::transform::Pass HoistGlobalBufferAllocations() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    if (function->body) {
      auto hoister = ffi::make_object<GlobalAllocationHoister>();
      SeqStmt body = function->body.value();
      function.CopyOnWrite()->body =
          hoister->Mutate(body).ValueOrUnchanged(body);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.HoistGlobalBufferAllocations");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.HoistGlobalBufferAllocations",
                                   HoistGlobalBufferAllocations);
}

} // namespace transform
} // namespace tvm::tl
