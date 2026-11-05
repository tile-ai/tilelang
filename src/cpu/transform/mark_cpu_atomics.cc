/*!
 * \file mark_cpu_atomics.cc
 * \brief Keep CPU grids serial when atomics will become serial RMW operations.
 */

#include "cpu/op/builtin.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
class AtomicCallFinder : public tirx::StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  bool found = false;

  ffi::Optional<ffi::VisitInterrupt> Visit_(const CallObj *call) final {
    if (auto op = call->op.as<Op>()) {
      const auto &name = op.value()->name;
      if (name.find("tl.atomic") == 0 || name.find("tl.tileop.atomic") == 0) {
        found = true;
        return ffi::VisitInterrupt();
      }
      if (name == "tl.annotation")
        return std::nullopt;
    }
    return StmtExprVisitor::Visit_(call);
  }
};
} // namespace

tvm::transform::Pass MarkCPUAtomics() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    if (!(!function->GetAttr(tvm::attr::kDeclaration).value_or(false)))
      return function;
    auto finder = ffi::make_object<AtomicCallFinder>();
    finder->Visit(function->body);
    if (finder->found) {
      auto attrs = function->attrs->dict;
      attrs.Set(attr::kCPUHadAtomics, true);
      function.CopyOnWrite()->attrs = DictAttrs(attrs);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.MarkCPUAtomics");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cpu.transform.MarkCPUAtomics",
                                   MarkCPUAtomics);
}
} // namespace tvm::tl
