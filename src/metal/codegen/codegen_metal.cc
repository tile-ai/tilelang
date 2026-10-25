#include "codegen_metal.h"

#include "backend/common/codegen/native_codegen.h"

#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm::codegen {
namespace {
class MetalNativeLowerer : public tirx::StmtExprMutator {
  ffi::UnchangedOr<Expr> Mutate_(const CallNode *node,
                                 ffi::InplaceMode mode) final {
    Call call = StmtExprMutator::Mutate_(node, mode)
                    .ValueOrUnchanged(GetRef<Expr>(node))
                    .as_or_throw<Call>();
    if (const auto *operation = call->op.as<OpNode>()) {
      if (operation->name == "tl.atomic_add_elem_op" ||
          operation->name == "tl.atomic_add_ret_elem_op") {
        TVM_FFI_CHECK(call->args.size() == 2 || call->args.size() == 3,
                      ValueError);
        if (call->args.size() == 3) {
          const auto *order = call->args[2].as<IntImmNode>();
          TVM_FFI_CHECK(order && order->value == 0, ValueError)
              << "Metal atomic add supports only relaxed memory ordering";
          call.CopyOnWrite()->args = {call->args[0], call->args[1]};
        }
        call.CopyOnWrite()->op = tirx::atomic_add_op();
        call.CopyOnWrite()->ty = call->args[1]->ty;
      }
    }
    return call;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    RegionStmt region = StmtExprMutator::Mutate_(node, mode)
                            .ValueOrUnchanged(GetRef<Stmt>(node))
                            .as_or_throw<RegionStmt>();
    if (auto pattern = region->attrs->dict.Get("threadblock_swizzle_pattern")) {
      auto attributes = region->attrs->dict;
      attributes.Set("metal.threadblock_swizzle_pattern", *pattern);
      attributes.erase("threadblock_swizzle_pattern");
      region.CopyOnWrite()->attrs = DictAttrs(attributes);
    }
    return region;
  }
};

IRModule LowerMetalBuiltins(IRModule module) {
  for (const auto &[global, base] : module->functions) {
    auto function = base.as_or_throw<tirx::Function>();
    if (!function->body)
      continue;
    auto body = *function->body;
    function.CopyOnWrite()->body =
        MetalNativeLowerer().Mutate(body).ValueOrUnchanged(body);
    module.CopyOnWrite()->Update(global, function);
  }
  return module;
}
} // namespace

ffi::Module BuildTileLangMetal(IRModule module, Target target) {
  return tl::BuildNativeCodegen(LowerMetalBuiltins(std::move(module)),
                                std::move(target),
                                "tilelang_callback_metal_postproc");
}

ffi::Module BuildTileLangMetalWithoutCompile(IRModule module, Target target) {
  return tl::BuildNativeCodegen(LowerMetalBuiltins(std::move(module)),
                                std::move(target),
                                "tilelang_callback_metal_postproc", true);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("target.build.tilelang_metal", BuildTileLangMetal)
      .def("target.build.tilelang_metal_without_compile",
           BuildTileLangMetalWithoutCompile);
}
} // namespace tvm::codegen
