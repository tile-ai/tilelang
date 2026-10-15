/*!
 * \file lower_cpu_atomics.cc
 * \brief Lower `tl.atomic_*_elem_op` intrinsics to plain serial
 *        read-modify-write for CPU targets.
 *
 * CPU execution is serial, so an atomic update degenerates to a plain
 * read-modify-write: `dst = dst <op> value`, with `return_prev` yielding the
 * value stored before the update. `memory_order` arguments are accepted and
 * ignored (there is no memory ordering semantics without concurrency); when
 * CPU gains thread-level parallelism this pass (together with
 * src/cpu/op/atomic_rmw.h for the tile-region path) is the single place to
 * switch to `__atomic_*` / `std::atomic_ref`.
 *
 * The pass runs on the CPU pipeline right after LowerTileOp (tile-region
 * atomics are already lowered to plain RMW loops by src/cpu/op/atomic_add.cc
 * and atomic_reduce.cc) and before LegalizeVectorizedLoop /
 * LegalizeSafeMemoryAccess / LowerAccessPtr, so every downstream pass and
 * both CPU codegens only ever see plain BufferLoad/BufferStore. This is also
 * why no CPU codegen changes are needed: the `c` codegen rejects unknown
 * intrinsics at compile time with an `Unresolved call` InternalError
 * (3rdparty/tvm/src/target/source/codegen_c.cc), and the `llvm` codegen
 * (vendored TVM) rejects them as unknown intrinsics.
 */

#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

/// Kind of an atomic read-modify-write elem op.
enum class AtomicRMWKind { kAdd, kMax, kMin, kOr, kStore, kAddVec };

struct AtomicRMWOp {
  AtomicRMWKind kind;
  int lanes;         ///< 1 for scalar ops, 2/4 for kAddVec
  bool returns_prev; ///< `_ret` variants return the pre-update value
};

/// Match all 12 side-effect atomic elem ops (everything but atomic_load).
std::optional<AtomicRMWOp> MatchAtomicRMWOp(const ObjectRef &op) {
  if (op.same_as(atomic_add_elem_op()))
    return AtomicRMWOp{AtomicRMWKind::kAdd, 1, false};
  if (op.same_as(atomic_add_ret_elem_op()))
    return AtomicRMWOp{AtomicRMWKind::kAdd, 1, true};
  if (op.same_as(atomic_max_elem_op()))
    return AtomicRMWOp{AtomicRMWKind::kMax, 1, false};
  if (op.same_as(atomic_max_ret_elem_op()))
    return AtomicRMWOp{AtomicRMWKind::kMax, 1, true};
  if (op.same_as(atomic_min_elem_op()))
    return AtomicRMWOp{AtomicRMWKind::kMin, 1, false};
  if (op.same_as(atomic_min_ret_elem_op()))
    return AtomicRMWOp{AtomicRMWKind::kMin, 1, true};
  if (op.same_as(atomic_or_elem_op()))
    return AtomicRMWOp{AtomicRMWKind::kOr, 1, false};
  if (op.same_as(atomic_store_elem_op()))
    return AtomicRMWOp{AtomicRMWKind::kStore, 1, false};
  if (op.same_as(atomic_addx2_elem_op()))
    return AtomicRMWOp{AtomicRMWKind::kAddVec, 2, false};
  if (op.same_as(atomic_addx4_elem_op()))
    return AtomicRMWOp{AtomicRMWKind::kAddVec, 4, false};
  if (op.same_as(atomic_addx2_ret_elem_op()))
    return AtomicRMWOp{AtomicRMWKind::kAddVec, 2, true};
  if (op.same_as(atomic_addx4_ret_elem_op()))
    return AtomicRMWOp{AtomicRMWKind::kAddVec, 4, true};
  return std::nullopt;
}

/// Buffer and element indices recovered from a `tl.access_ptr` argument.
struct AtomicAccess {
  Buffer buffer;
  Array<PrimExpr> indices;
};

PrimExpr MakeRMWCombine(AtomicRMWKind kind, const PrimExpr &old,
                        const PrimExpr &value) {
  switch (kind) {
  case AtomicRMWKind::kAdd:
  case AtomicRMWKind::kAddVec:
    return old + value;
  case AtomicRMWKind::kMax:
    return Max(old, value);
  case AtomicRMWKind::kMin:
    return Min(old, value);
  case AtomicRMWKind::kOr:
    return old | value;
  case AtomicRMWKind::kStore:
    return value;
  }
  LOG(FATAL) << "Unreachable atomic RMW kind";
  return PrimExpr();
}

PrimExpr CastIfNeeded(DataType dtype, const PrimExpr &value) {
  return value->dtype == dtype ? value : Cast(dtype, value);
}

class CPUAtomicRewriter : public StmtExprMutator {
public:
  CPUAtomicRewriter() = default;

  static PrimFunc Rewrite(PrimFunc func) {
    if (!func.defined() || !func->body.defined()) {
      return func;
    }
    CPUAtomicRewriter rewriter;
    PrimFuncNode *n = func.CopyOnWrite();
    n->body = rewriter(std::move(n->body));
    return func;
  }

  // A `return_prev` rewrite records (Bind of the old value, RMW store) pairs
  // while visiting expressions; each statement is wrapped so that the pairs
  // execute before the original statement:
  //   SeqStmt(Bind(old, BufferLoad(dst, idx)),
  //           BufferStore(dst, old <op> v, idx),
  //           <stmt using old>)
  // Bind has no body: the variable is visible in the subsequent statements
  // of the enclosing SeqStmt, so the old value is read before the store.
  Stmt VisitStmt(const Stmt &stmt) final {
    std::vector<Stmt> outer_prefix = std::move(pending_prefix_);
    pending_prefix_.clear();

    Stmt result = StmtExprMutator::VisitStmt(stmt);

    if (!pending_prefix_.empty()) {
      pending_prefix_.push_back(result);
      result = SeqStmt::Flatten(pending_prefix_);
    }

    pending_prefix_ = std::move(outer_prefix);
    return result;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    // Statement position: the result (if any) is discarded, so `_ret`
    // variants lower to the same plain RMW as their base op.
    if (const auto *call = op->value.as<CallNode>()) {
      if (std::optional<AtomicRMWOp> rmw = MatchAtomicRMWOp(call->op)) {
        return LowerStmtAtomic(*call, *rmw);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(atomic_load_elem_op())) {
      // Pure read on a serial target.
      AtomicAccess access = ParseAtomicAccessPtr(op->args[0], "atomic_load");
      return BufferLoad(access.buffer, MutateIndices(access.indices));
    }
    if (std::optional<AtomicRMWOp> rmw = MatchAtomicRMWOp(op->op)) {
      ICHECK(rmw->returns_prev)
          << "CPU atomic lowering: " << op->op
          << " has no return value and cannot appear in expression position; "
             "use it as a standalone statement.";
      ICHECK(rmw->lanes == 1)
          << "CPU atomic lowering: atomic_addx" << rmw->lanes
          << " with return_prev=True is not supported on CPU (the previous "
             "vector value cannot be represented without TIR vector "
             "construction); drop return_prev or use scalar atomic_add.";
      return RewriteRetAtomic(*op, *rmw);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

private:
  AtomicAccess ParseAtomicAccessPtr(const PrimExpr &ptr, const char *role) {
    const auto *call = ptr.as<CallNode>();
    ICHECK(call != nullptr && call->op.same_as(tl::access_ptr()))
        << "CPU atomic lowering expects the " << role
        << " argument to be a tl.access_ptr call, but got: " << ptr;
    const auto *load = call->args[0].as<BufferLoadNode>();
    ICHECK(load != nullptr)
        << "CPU atomic lowering expects the tl.access_ptr base of " << role
        << " to be a BufferLoad, but got: " << call->args[0];
    return {load->buffer, load->indices};
  }

  Array<PrimExpr> MutateIndices(const Array<PrimExpr> &indices) {
    Array<PrimExpr> mutated;
    mutated.reserve(indices.size());
    for (const PrimExpr &index : indices) {
      mutated.push_back(VisitExpr(index));
    }
    return mutated;
  }

  // Indices of the k-th element of a vectorized (x2/x4) access: the vector
  // occupies consecutive elements along the last index (a slice `b[i, j:j+2]`
  // reaches here as its scalar minimum `j`, so plain `+ k` is the expansion).
  Array<PrimExpr> ElemIndices(Array<PrimExpr> indices, int k) {
    if (k == 0) {
      return indices;
    }
    PrimExpr last = indices.back();
    if (const auto *ramp = last.as<RampNode>()) {
      indices.Set(indices.size() - 1,
                  ramp->base + make_const(ramp->base->dtype, k) * ramp->stride);
    } else {
      indices.Set(indices.size() - 1, last + make_const(last->dtype, k));
    }
    return indices;
  }

  // Expression position `_ret`: bind the pre-update value to a fresh var
  // (executed first), perform the RMW store, and evaluate to the old value.
  PrimExpr RewriteRetAtomic(const CallNode &call, const AtomicRMWOp &rmw) {
    AtomicAccess dst = ParseAtomicAccessPtr(call.args[0], "atomic dst");
    Array<PrimExpr> indices = MutateIndices(dst.indices);
    PrimExpr value = CastIfNeeded(dst.buffer->dtype, VisitExpr(call.args[1]));

    BufferLoad old_load(dst.buffer, indices);
    Var old_var("atomic_old_" + std::to_string(atomic_old_counter_++),
                old_load->dtype);
    pending_prefix_.push_back(Bind(old_var, old_load));
    pending_prefix_.push_back(BufferStore(
        dst.buffer, MakeRMWCombine(rmw.kind, old_var, value), indices));
    return old_var;
  }

  Stmt LowerStmtAtomic(const CallNode &call, const AtomicRMWOp &rmw) {
    AtomicAccess dst = ParseAtomicAccessPtr(call.args[0], "atomic dst");

    if (rmw.kind == AtomicRMWKind::kAddVec) {
      // args = (dst access_ptr, value access_ptr): expand to per-element RMW.
      AtomicAccess src = ParseAtomicAccessPtr(call.args[1], "atomic_addx");
      std::vector<Stmt> stores;
      for (int k = 0; k < rmw.lanes; ++k) {
        Array<PrimExpr> dst_indices =
            MutateIndices(ElemIndices(dst.indices, k));
        Array<PrimExpr> src_indices =
            MutateIndices(ElemIndices(src.indices, k));
        PrimExpr elem = CastIfNeeded(dst.buffer->dtype,
                                     BufferLoad(src.buffer, src_indices));
        stores.push_back(BufferStore(dst.buffer,
                                     BufferLoad(dst.buffer, dst_indices) + elem,
                                     dst_indices));
      }
      return SeqStmt::Flatten(stores);
    }

    // Scalar forms: args = (ptr, value[, memory_order]); the optional
    // trailing memory_order id is intentionally ignored on a serial target.
    Array<PrimExpr> indices = MutateIndices(dst.indices);
    PrimExpr value = CastIfNeeded(dst.buffer->dtype, VisitExpr(call.args[1]));
    return BufferStore(
        dst.buffer,
        MakeRMWCombine(rmw.kind, BufferLoad(dst.buffer, indices), value),
        indices);
  }

  int atomic_old_counter_ = 0;
  // Bind/Store prefix statements recorded by `return_prev` rewrites, spliced
  // before the enclosing statement by VisitStmt.
  std::vector<Stmt> pending_prefix_;
};

} // namespace

namespace transform {

tvm::transform::Pass LowerCPUAtomics() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    return CPUAtomicRewriter::Rewrite(std::move(f));
  };
  return tvm::tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                                  "tl.LowerCPUAtomics", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.cpu.transform.LowerCPUAtomics", LowerCPUAtomics);
}

} // namespace transform

} // namespace tl
} // namespace tvm
