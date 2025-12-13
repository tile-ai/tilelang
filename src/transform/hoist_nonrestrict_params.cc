/*
 * Hoist tl.non_restrict_params block annotation to PrimFunc attribute.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/function.h>

#include "../op/builtin.h"

namespace tvm {
namespace tl {
using namespace tvm::tir;

class RootBlockFinder : public StmtVisitor {
 public:
  const BlockNode* Find(const Stmt& stmt) {
    VisitStmt(stmt);
    return root_;
  }

 private:
  void VisitStmt_(const BlockNode* op) final {
    if (root_ == nullptr) root_ = op;
    // Do not recurse further; we only care about the first (root) block
  }
  const BlockNode* root_{nullptr};
};

static PrimFunc HoistNonRestrictParams(PrimFunc f) {
  if (!f.defined()) return f;
  RootBlockFinder finder;
  const BlockNode* root = finder.Find(f->body);
  if (root == nullptr) return f;

  auto it = root->annotations.find(attr::kNonRestrictParams);
  if (it == root->annotations.end()) return f;

  // Expect an Array<Var> of buffer data Vars
  if (it->second.as<ArrayNode>() == nullptr) return f;

  return WithAttr(std::move(f), attr::kNonRestrictParams, it->second);
}

namespace transform {

tvm::transform::Pass HoistNonRestrictParams() {
  auto pass_func = [](PrimFunc f, const IRModule&, const tvm::transform::PassContext&) {
    return HoistNonRestrictParams(std::move(f));
  };
  return tvm::tir::transform::CreatePrimFuncPass(pass_func, 0, "tl.HoistNonRestrictParams", {});
}

}  // namespace transform

}  // namespace tl
}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.HoistNonRestrictParams",
                        tvm::tl::transform::HoistNonRestrictParams);
}
