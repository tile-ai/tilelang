#ifndef TVM_TL_CPU_OP_ATOMIC_RMW_H_
#define TVM_TL_CPU_OP_ATOMIC_RMW_H_
#include "op/prim.h"

#include "backend/common/op/atomic_reduce.h"

namespace tvm::tl::cpu {

enum class AtomicCombine { kAdd, kMax, kMin };

inline PrimExpr MakeAtomicCombine(AtomicCombine combine,
                                  const PrimExpr &old_value,
                                  const PrimExpr &value) {
  switch (combine) {
  case AtomicCombine::kAdd:
    return old_value + value;
  case AtomicCombine::kMax:
    return Max(old_value, value);
  case AtomicCombine::kMin:
    return Min(old_value, value);
  }
  TVM_FFI_THROW(ValueError) << "Unknown atomic combine kind";
  TVM_FFI_UNREACHABLE();
}

inline Stmt LowerAtomicRMW(const AtomicOpBaseNode &op, AtomicCombine combine,
                           const LowerArgs &lower_args, const char *op_name) {
  TVM_FFI_CHECK(!op.UseTMA(), ValueError)
      << "CPU " << op_name << " does not support TMA";
  TVM_FFI_CHECK(IsGlobalBuffer(op.dst) || IsLocalBuffer(op.dst, true),
                ValueError)
      << "CPU " << op_name << " requires a global or local destination tensor";
  sym::Analyzer analyzer;
  auto iteration =
      backend::atomic_reduce::MakeIteration(op, analyzer, &lower_args);
  TensorVar destination =
      backend::atomic_reduce::RemapTensor(op.dst, &lower_args);
  PrimExpr value = MakeAtomicCombine(
      combine, TensorLoad(destination, iteration.destination_indices),
      iteration.value);
  Stmt body = TensorStore(destination, iteration.destination_indices, value);
  if (!IsOne(iteration.predicate)) {
    body = If(iteration.predicate, SeqStmt(body));
  }
  for (size_t axis = iteration.loops.size(); axis-- > 0;) {
    const auto &loop = iteration.loops[axis];
    body =
        For(loop->var, tl::MakeConst(loop->var.ty(), 0), loop->dom->extent,
            tl::MakeConst(loop->var.ty(), 1), ForKind::kDefault, SeqStmt(body));
  }
  if (!iteration.loops.empty()) {
    body = PragmaUnrollLoop(body.as_or_throw<For>());
  }
  return body;
}

} // namespace tvm::tl::cpu

#endif
