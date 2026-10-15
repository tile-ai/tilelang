/*!
 * \file rewrite_fp4_to_fp4x2.cc
 * \brief Rewrite packed 4-bit float buffers (float4_e2m1fn, lanes=1) into the
 * 1-byte packed-pair form (float4_e2m1fnx2, lanes=2) before Ascend codegen.
 *
 * On Ascend a `float4_e2m1fn` element is 4 bits (sub-byte), so it has no valid
 * C element type and pointer arithmetic in fp4-element units over-addresses by
 * 2x (each fp4 index is emitted as a 1-byte step). We uniformly retype genuine
 * fp4 storage (UB allocations and fp4 kernel params) to `float4_e2m1fnx2`
 * (bits=4 x lanes=2 = 8 bits = one byte holding two codes) and halve every fp4
 * element count / offset / index, so `buf[i]` addresses byte i as intended.
 *
 * Only data vars that are *genuinely* allocated / typed as fp4 are rewritten;
 * fp4 `T.view`s layered over non-fp4 (e.g. int8) storage keep their existing
 * byte-based addressing and are left untouched.
 */

#include "support/check.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cast.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

#include "op/builtin.h"
#include "runtime/thread_storage_scope.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

// A genuine (unpacked) fp4 scalar: type_code float4_e2m1fn, bits=4, lanes=1.
inline bool IsFp4Scalar(const DataType &dt) {
  return dt.is_float4_e2m1fn() && dt.lanes() == 1;
}

// The 1-byte packed-pair form: same type_code, bits=4, lanes=2.
inline DataType Fp4x2() { return DataType::Float4E2M1FN(2); }

// Collect the data Vars of buffers that are genuinely allocated / typed as fp4
// scalar: AllocBuffer/DeclBuffer with fp4 dtype and PrimFunc params whose
// pointer element type is fp4. These are the only vars whose accesses we halve.
class Fp4StorageCollector : public StmtExprVisitor {
public:
  std::unordered_set<const VarNode *> fp4_vars;

  void Collect(const PrimFunc &f) {
    for (const Var &p : f->params) {
      if (const auto *ptr = p->type_annotation.as<PointerTypeNode>()) {
        if (const auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          if (IsFp4Scalar(prim->dtype))
            fp4_vars.insert(p.get());
        }
      }
    }
    for (const auto &kv : f->buffer_map) {
      if (IsFp4Scalar(kv.second->dtype))
        fp4_vars.insert(kv.second->data.get());
    }
    VisitStmt(f->body);
  }

private:
  void VisitStmt_(const AllocBufferNode *op) final {
    if (IsFp4Scalar(op->buffer->dtype))
      fp4_vars.insert(op->buffer->data.get());
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const DeclBufferNode *op) final {
    if (IsFp4Scalar(op->buffer->dtype))
      fp4_vars.insert(op->buffer->data.get());
    StmtExprVisitor::VisitStmt_(op);
  }
};

class Fp4ToFp4x2Mutator : public StmtExprMutator {
public:
  explicit Fp4ToFp4x2Mutator(std::unordered_set<const VarNode *> fp4_vars)
      : fp4_vars_(std::move(fp4_vars)) {}

private:
  std::unordered_set<const VarNode *> fp4_vars_;

  bool IsFp4Var(const VarNode *v) const { return fp4_vars_.count(v) > 0; }

  // Halve an fp4 element count / offset, requiring it to be even (fp4 storage
  // on Ascend is always accessed in whole bytes = pairs of codes).
  static PrimExpr HalveEven(const PrimExpr &e, const char *what) {
    if (const auto *imm = e.as<IntImmNode>()) {
      ICHECK_EQ(imm->value % 2, 0)
          << "RewriteFp4ToFp4x2: " << what << " must be even (fp4 is packed 2 "
          << "codes/byte), got " << imm->value;
    }
    return FloorDiv(e, make_const(e.dtype(), 2));
  }

  // Retype an fp4-scalar buffer to fp4x2, halving the innermost dim, strides,
  // and elem_offset so the buffer now counts in packed-pair (byte) units.
  Buffer RemapBuffer(const Buffer &buffer) {
    if (!IsFp4Scalar(buffer->dtype))
      return buffer;
    Buffer buf = buffer;
    auto *w = buf.CopyOnWrite();
    w->dtype = Fp4x2();
    if (!w->shape.empty()) {
      Array<PrimExpr> shape = w->shape;
      int last = static_cast<int>(shape.size()) - 1;
      shape.Set(last, HalveEven(shape[last], "fp4 buffer last-dim shape"));
      w->shape = shape;
    }
    if (!w->strides.empty()) {
      Array<PrimExpr> strides = w->strides;
      for (int i = 0; i < static_cast<int>(strides.size()); ++i) {
        if (!is_one(strides[i]))
          strides.Set(i, HalveEven(strides[i], "fp4 buffer stride"));
      }
      w->strides = strides;
    }
    if (w->elem_offset.defined() && !is_zero(w->elem_offset))
      w->elem_offset = HalveEven(w->elem_offset, "fp4 buffer elem_offset");
    return buf;
  }

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    auto node = Downcast<AllocBuffer>(StmtExprMutator::VisitStmt_(op));
    Buffer nb = RemapBuffer(node->buffer);
    if (!nb.same_as(node->buffer))
      node.CopyOnWrite()->buffer = nb;
    return std::move(node);
  }

  Stmt VisitStmt_(const DeclBufferNode *op) final {
    auto node = Downcast<DeclBuffer>(StmtExprMutator::VisitStmt_(op));
    Buffer nb = RemapBuffer(node->buffer);
    if (!nb.same_as(node->buffer))
      node.CopyOnWrite()->buffer = nb;
    return std::move(node);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    if (IsFp4Scalar(node->buffer->dtype)) {
      ICHECK_EQ(node->indices.size(), 1U)
          << "RewriteFp4ToFp4x2 expects flat (1-D) fp4 buffers; run after "
             "FlattenBuffer";
      auto *w = node.CopyOnWrite();
      w->buffer = RemapBuffer(node->buffer);
      w->indices = {HalveEven(node->indices[0], "fp4 index")};
    }
    return std::move(node);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (IsFp4Scalar(node->buffer->dtype)) {
      ICHECK_EQ(node->indices.size(), 1U)
          << "RewriteFp4ToFp4x2 expects flat (1-D) fp4 buffers; run after "
             "FlattenBuffer";
      auto *w = node.CopyOnWrite();
      w->buffer = RemapBuffer(node->buffer);
      w->indices = {HalveEven(node->indices[0], "fp4 index")};
    }
    return std::move(node);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      // tvm_access_ptr(type_annotation, data_var, offset, extent, rw_mask)
      ICHECK_EQ(op->args.size(), 5U);
      const auto *data_var = op->args[1].as<VarNode>();
      if (data_var && IsFp4Var(data_var) && IsFp4Scalar(op->args[0].dtype())) {
        // Retype the annotation to the packed pair (fp4x2) and halve the
        // offset/extent into packed-pair (byte) units. The Ascend override of
        // tirx.tvm_access_ptr (intrin_rule_ascend.cc) addresses fp4x2 as a
        // scalar byte, so this stays a plain byte pointer.
        PrimExpr offset = VisitExpr(op->args[2]);
        PrimExpr extent = VisitExpr(op->args[3]);
        return Call(op->dtype, op->op,
                    {tirx::TypeAnnotation(Fp4x2()), op->args[1],
                     HalveEven(offset, "fp4 access_ptr offset"),
                     HalveEven(extent, "fp4 access_ptr extent"), op->args[4]});
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }
};

PrimFunc RewriteFp4ToFp4x2PrimFunc(PrimFunc f) {
  if (!f.defined() || !f->body.defined())
    return f;

  Fp4StorageCollector collector;
  collector.Collect(f);
  if (collector.fp4_vars.empty())
    return f;

  Fp4ToFp4x2Mutator mutator(collector.fp4_vars);
  auto *n = f.CopyOnWrite();
  n->body = mutator(std::move(n->body));
  return f;
}

} // namespace

using namespace tirx::transform;

namespace transform {

tvm::transform::Pass RewriteFp4ToFp4x2() {
  auto pass_func = [](PrimFunc f, const IRModule &m, PassContext ctx) {
    return RewriteFp4ToFp4x2PrimFunc(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.RewriteFp4ToFp4x2", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.RewriteFp4ToFp4x2", RewriteFp4ToFp4x2);
}

} // namespace transform

} // namespace tl
} // namespace tvm
