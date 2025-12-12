/*!
 * \file annotate_read_only_params.cc
 * \brief Annotate PrimFunc parameters that are read-only (never written).
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <unordered_set>

namespace tvm {
namespace tl {
using namespace tir;
using namespace ffi;

/*!
 * \brief A simple visitor that marks handle parameters as written when they
 *        appear on the LHS of a BufferStore or in a tvm_access_ptr with write flag.
 */
class ReadWriteMarker : public StmtExprVisitor {
public:
  explicit ReadWriteMarker(const Array<Var>& handle_params) {
    for (const Var& v : handle_params) {
      handle_params_.insert(v.get());
    }
  }

  const std::unordered_set<const VarNode*>& written() const { return written_; }

  void VisitStmt_(const BufferStoreNode* op) final {
    const VarNode* data = op->buffer->data.get();
    if (handle_params_.count(data)) {
      written_.insert(data);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode* op) final {
    // Detect tvm_access_ptr writes
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      if (op->args.size() == 5U) {
        if (const VarNode* buf = op->args[1].as<VarNode>()) {
          if (const IntImmNode* flag = op->args[4].as<IntImmNode>()) {
            if ((flag->value & 2) != 0) {
              if (handle_params_.count(buf)) {
                written_.insert(buf);
              }
            }
          }
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

private:
  std::unordered_set<const VarNode*> handle_params_;
  std::unordered_set<const VarNode*> written_;
};

/*!
 * \brief Annotate PrimFunc with indices of read-only handle parameters.
 *
 * Adds an Array<Integer> attribute "tl.readonly_param_indices" that lists
 * parameter indices which correspond to handle parameters that are never
 * written inside the function body. This can be used by codegen to emit
 * `const` qualifiers to enable read-only caching (e.g., __ldg on CUDA).
 */
static tir::PrimFunc MarkReadOnlyParams(tir::PrimFunc f) {
  // Collect handle parameters (pointer-like, e.g., buffers in global memory)
  Array<Var> handle_params;
  for (const Var& v : f->params) {
    if (v->dtype.is_handle()) {
      handle_params.push_back(v);
    }
  }
  if (handle_params.empty()) {
    return f;
  }

  ReadWriteMarker marker(handle_params);
  marker(f->body);

  // Determine read-only parameter indices among all params
  Array<Integer> readonly_indices;
  for (size_t i = 0; i < f->params.size(); ++i) {
    const Var& v = f->params[i];
    if (!v->dtype.is_handle()) continue;
    if (marker.written().count(v.get()) == 0) {
      readonly_indices.push_back(Integer(static_cast<int>(i)));
    }
  }

  if (!readonly_indices.empty()) {
    Map<String, Any> attrs;
    attrs.Set(String("tl.readonly_param_indices"), readonly_indices);
    f = WithAttrs(std::move(f), attrs);
  }
  return f;
}

namespace transform {
using namespace tir::transform;

Pass AnnotateReadOnlyParams() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    return MarkReadOnlyParams(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AnnotateReadOnlyParams", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnnotateReadOnlyParams", AnnotateReadOnlyParams);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm
