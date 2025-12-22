/*!
 * \file annotate_read_only_params.cc
 * \brief Annotate PrimFunc parameters that are read-only (never written).
 */

#include <string>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <unordered_set>

namespace tvm {
namespace tl {
using namespace tir;
using namespace ffi;

/*!
 * \brief A simple visitor that marks handle parameters as written when they
 *        appear on the LHS of a BufferStore or in a tvm_access_ptr with write
 * flag.
 */
class ReadWriteMarker : public StmtExprVisitor {
public:
  explicit ReadWriteMarker(
      const std::unordered_set<const VarNode *> &param_or_data_vars)
      : param_or_data_vars_(param_or_data_vars) {}

  const std::unordered_set<const VarNode *> &written() const {
    return written_;
  }

  // Try to resolve the underlying buffer data Var from a pointer-like
  // argument. Supports:
  //  - address_of(BufferLoad(...)) -> returns buffer->data
  //  - BufferLoad(...)             -> returns buffer->data
  // Otherwise returns nullptr.
  const VarNode *ResolveDataVarFromPtrArg(const PrimExpr &arg) const {
    if (const auto *call = arg.as<CallNode>()) {
      if (call->op.same_as(builtin::address_of())) {
        if (call->args.size() == 1U) {
          if (const auto *load = call->args[0].as<BufferLoadNode>()) {
            return load->buffer->data.get();
          }
        }
      }
    } else if (const auto *load = arg.as<BufferLoadNode>()) {
      return load->buffer->data.get();
    }
    return nullptr;
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    const VarNode *data = op->buffer->data.get();
    if (param_or_data_vars_.count(data)) {
      written_.insert(data);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    // Detect tvm_access_ptr writes. Be conservative if rw_mask is non-constant.
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      if (op->args.size() == 5U) {
        if (const VarNode *buf = op->args[1].as<VarNode>()) {
          const IntImmNode *flag = op->args[4].as<IntImmNode>();
          bool maybe_write = true; // default conservative
          if (flag) {
            maybe_write = (flag->value & 2) != 0; // write bit set
          }
          if (maybe_write && param_or_data_vars_.count(buf)) {
            written_.insert(buf);
          }
        }
      }
    } else {
      // Generic fallback: mark buffers that appear as
      // address_of(BufferLoad(...)) in call arguments as written. This matches
      // patterns like
      //   tl.tma_store(address_of(smem[..]), address_of(gmem[..]), ...)
      //   call_extern("AtomicAdd*", address_of(gmem[..]), ...)
      // and avoids over-marking plain BufferLoad used for reads.
      for (const PrimExpr &a : op->args) {
        if (const auto *c = a.as<CallNode>()) {
          if (c->op.same_as(builtin::address_of()) && c->args.size() == 1U) {
            if (const auto *bl = c->args[0].as<BufferLoadNode>()) {
              const VarNode *data = bl->buffer->data.get();
              if (param_or_data_vars_.count(data)) {
                written_.insert(data);
              }
            }
          }
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

private:
  std::unordered_set<const VarNode *> param_or_data_vars_;
  std::unordered_set<const VarNode *> written_;
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
  // Gather handle params and their corresponding buffer data vars (aliases).
  std::unordered_set<const VarNode *> param_or_data_vars;
  // Map back from data var to parameter index for result attribution.
  std::unordered_map<const VarNode *, size_t> data_var_to_param_idx;

  for (size_t i = 0; i < f->params.size(); ++i) {
    const Var &p = f->params[i];
    if (!p->dtype.is_handle())
      continue;
    param_or_data_vars.insert(p.get());
    // If there is a buffer_map entry for this param, include its data var too.
    if (auto opt = f->buffer_map.Get(p)) {
      const VarNode *data = opt.value()->data.get();
      param_or_data_vars.insert(data);
      data_var_to_param_idx[data] = i;
    }
  }
  if (param_or_data_vars.empty())
    return f;

  ReadWriteMarker marker(param_or_data_vars);
  marker(f->body);

  // Determine read-only parameter indices among all params (handle only)
  Array<Integer> readonly_indices;
  for (size_t i = 0; i < f->params.size(); ++i) {
    const Var &v = f->params[i];
    if (!v->dtype.is_handle())
      continue;

    bool is_written = false;
    // Direct param var written?
    if (marker.written().count(v.get())) {
      is_written = true;
    } else {
      // Or any aliased data var written?
      if (auto opt = f->buffer_map.Get(v)) {
        if (marker.written().count(opt.value()->data.get())) {
          is_written = true;
        }
      }
    }

    if (!is_written) {
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
  refl::GlobalDef().def("tl.transform.AnnotateReadOnlyParams",
                        AnnotateReadOnlyParams);
}

} // namespace transform
} // namespace tl
} // namespace tvm
