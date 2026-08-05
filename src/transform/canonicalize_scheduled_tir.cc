/*!
 * \file canonicalize_scheduled_tir.cc
 * \brief Canonicalize scheduled TIR into a single tilelang_root block while
 * preserving target-neutral kernel launch loops.
 */

#include "support/check.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tir/transforms/ir_utils.h"

#include "layout_reducer.h"

namespace tvm {
namespace tl {
using namespace tirx;
using namespace tvm::s_tir;

/*!
 * \brief Canonicalize scheduled TIR into the backend-neutral kernel form.
 */
class ScheduledTIRCanonicalizer : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt body) {
    ScheduledTIRCanonicalizer canonicalizer;
    return canonicalizer(std::move(body));
  }

private:
  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    // We have convert blocks into opaque blocks in previous passes.
    ICHECK(op->iter_values.empty())
        << "CanonicalizeScheduledTIR requires opaque blocks. Please call "
           "ConvertBlocksToOpaque first.";
    // Step 1. Visit the body
    block_level_++;
    SBlock new_block = Downcast<SBlock>(this->VisitStmt(op->block));
    block_level_--;
    PrimExpr predicate = this->VisitExpr(op->predicate);
    // Step 2. Transform the `predicate` to if-then-else
    Stmt body = new_block->body;
    if (!is_one(predicate) && block_level_ != 0) {
      body = IfThenElse(predicate, std::move(body));
    }

    for (size_t i = 0; i < new_block->alloc_buffers.size(); ++i) {
      allocated_buffers_.insert(new_block->alloc_buffers[i]);
    }

    // Step 4. Handle annotations, block annotations are not preserved by
    // default.
    std::vector<std::pair<std::string, PrimExpr>> pragma_attrs;
    HandleAnnotations(new_block->annotations, &pragma_attrs, /*is_block=*/true);
    for (auto it = pragma_attrs.rbegin(); it != pragma_attrs.rend(); ++it) {
      body = AttrStmt(Integer(0), it->first, it->second, std::move(body));
    }

    if (block_level_ == 0) {
      auto p_block = new_block.CopyOnWrite();
      p_block->name_hint = "tilelang_root";
      p_block->alloc_buffers = ffi::Array<Buffer>(allocated_buffers_.begin(),
                                                  allocated_buffers_.end());
      p_block->body = std::move(body);
      // Merge preserved block annotations (e.g. reducer_info) into root block.
      for (const auto &kv : root_annotations_) {
        p_block->annotations.Set(kv.first, kv.second);
      }
      Stmt block_realize = SBlockRealize(
          ffi::Array<PrimExpr>(), std::move(predicate), std::move(new_block));

      std::sort(thread_bindings_.begin(), thread_bindings_.end(),
                [](const auto &t1, const auto &t2) {
                  return t1.thread_binding->thread_tag <
                         t2.thread_binding->thread_tag;
                });

      for (auto it = thread_bindings_.rbegin(); it != thread_bindings_.rend();
           ++it) {
        block_realize = For(it->loop_var, it->min, it->extent,
                            ForKind::kThreadBinding, std::move(block_realize),
                            it->thread_binding, it->annotations, it->step);
      }
      SBlock root_block =
          SBlock(ffi::Array<IterVar>(), {}, {}, {}, block_realize);
      return SBlockRealize(ffi::Array<PrimExpr>(), const_true(),
                           std::move(root_block));
    }

    return body;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Step 1. Update unit loop info.
    PrimExpr min = this->VisitExpr(op->min);
    PrimExpr extent = this->VisitExpr(op->extent);
    if (is_one(extent) && op->annotations.empty()) {
      // handling unit loop
      unit_loop_vars_[op->loop_var] = min;
    }

    // Step 2. Visit recursively
    Stmt body = this->VisitStmt(op->body);

    // Step 3. Handle annotations
    std::vector<std::pair<std::string, PrimExpr>> pragma_attrs;
    ffi::Map<ffi::String, ffi::Any> new_annotations =
        HandleAnnotations(op->annotations, &pragma_attrs, /*is_block=*/false);
    // Step 4. Create new For loop accordingly
    if (op->kind == ForKind::kThreadBinding) {
      // Case 1. Thread binding
      ICHECK(op->thread_binding.defined());
      thread_bindings_.push_back({std::move(min), std::move(extent),
                                  op->loop_var, op->thread_binding.value(),
                                  std::move(new_annotations), op->step});
    } else if (is_one(extent) && op->annotations.empty() &&
               !op->annotations.count(
                   ::tvm::s_tir::attr::irregular_loop_mark)) {
      // Case 2. Unit loop
      return body;
    } else {
      // Case 3. An ordinary loop
      body = For(op->loop_var, std::move(min), std::move(extent), op->kind,
                 std::move(body), std::nullopt, new_annotations, op->step);
    }
    // Step 5. Insert nested attrs
    for (auto it = pragma_attrs.rbegin(); it != pragma_attrs.rend(); ++it) {
      body = AttrStmt(op->loop_var, it->first, it->second, std::move(body));
    }
    return body;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = ffi::GetRef<Var>(op);
    auto it = unit_loop_vars_.find(var);
    if (it == unit_loop_vars_.end()) {
      return var;

    } else {
      PrimExpr expr = it->second;
      if (expr.dtype() != var.dtype()) {
        expr = tvm::cast(var.dtype(), std::move(expr));
      }
      return expr;
    }
  }

  /*! \brief Convert attr value from annotation map into PrimExpr. */
  PrimExpr ConvertAttrValue(const ffi::String &key, const Any &obj) {
    if (obj == nullptr) {
      return PrimExpr();
    } else if (auto expr = obj.try_cast<PrimExpr>()) {
      return expr.value();
    } else if (auto str = obj.try_cast<ffi::String>()) {
      return std::move(StringImm(str.value()));
    } else {
      ICHECK(false) << "Illegal attribute of key " << key << ", value type "
                    << obj.GetTypeKey() << " not supported";
      return PrimExpr();
    }
  }

  /*!
   * \brief Helper to handle annotation dict.
   * (1) if the attr key is prefixed by `pragma_`, move to ordered kv list. They
   * are lowered to `AttrStmt` by legacy TE schedule convention.
   * (2) the non-pragma loop annotations are preserved
   * (3) the non-pragma block annotations are dropped
   * \return New annotation dict with preserved keys. Also update pragma attr
   * pairs ordered by key.
   */
  ffi::Map<ffi::String, ffi::Any>
  HandleAnnotations(const ffi::Map<ffi::String, ffi::Any> &annotations,
                    std::vector<std::pair<std::string, PrimExpr>> *pragma_attrs,
                    bool is_block) {
    ffi::Map<ffi::String, ffi::Any> preserved_annotations;
    pragma_attrs->clear();
    for (const auto &kv : annotations) {
      const ffi::String &key = kv.first;
      if (::tvm::tirx::attr::IsPragmaKey(key)) {
        pragma_attrs->emplace_back(key, ConvertAttrValue(key, kv.second));
      } else if (!is_block) {
        // the loop annotation is preserved
        preserved_annotations.Set(key, kv.second);
      } else if (key == ::tvm::tl::attr::kReducerInfo) {
        // Preserve reducer_info so LayoutReducer can find it on the root block.
        root_annotations_.Set(key, kv.second);
      }
    }
    std::sort(
        pragma_attrs->begin(), pragma_attrs->end(),
        [](const auto &p1, const auto &p2) { return p1.first < p2.first; });
    return preserved_annotations;
  }

  /*! \brief Record the loop_var and loop start value of unit loops, whose
   * extent is one. */
  std::unordered_map<Var, PrimExpr> unit_loop_vars_;

  std::unordered_set<Buffer, tvm::ffi::ObjectPtrHash, tvm::ffi::ObjectPtrEqual>
      allocated_buffers_;

  /*! \brief Block annotations (e.g. reducer_info) to propagate to the root
   * block. */
  ffi::Map<ffi::String, ffi::Any> root_annotations_;

  struct ThreadBindingLoop {
    PrimExpr min;
    PrimExpr extent;
    Var loop_var;
    IterVar thread_binding;
    ffi::Map<ffi::String, ffi::Any> annotations;
    ffi::Optional<PrimExpr> step;
  };

  std::vector<ThreadBindingLoop> thread_bindings_;
  int block_level_ = 0;
};

PrimFunc CanonicalizeScheduledTIR(PrimFunc f) {
  auto fptr = f.CopyOnWrite();
  fptr->body = ScheduledTIRCanonicalizer::Rewrite(std::move(fptr->body));
  return f;
}
using namespace tirx::transform;

namespace transform {
Pass CanonicalizeScheduledTIR() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return ::tvm::tl::CanonicalizeScheduledTIR(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.CanonicalizeScheduledTIR", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.CanonicalizeScheduledTIR",
                        CanonicalizeScheduledTIR);
}
} // namespace transform

} // namespace tl
} // namespace tvm
