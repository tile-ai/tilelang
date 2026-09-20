/*!
 * \file tl/cpu/op/atomic_rmw.h
 * \brief Shared serial read-modify-write lowering for CPU atomic tile ops.
 *
 * CPU execution is serial, so an atomic update degenerates to a plain
 * read-modify-write on the destination buffer. This helper builds a kSerial
 * loop nest over the destination region whose body is a plain
 * BufferLoad/combine/BufferStore, mirroring the src/cpu/op/reduce.cc
 * conventions (no SIMD, no thread-level parallelism; correctness first).
 *
 * `memory_order` annotations are accepted but ignored: without concurrency
 * there is no memory ordering semantics. When CPU gains thread-level
 * parallelism this helper (and tl.transform.LowerCPUAtomics for the scalar
 * path) is the single place to switch to `__atomic_*` / `std::atomic_ref`.
 */

#ifndef TVM_TL_CPU_OP_ATOMIC_RMW_H_
#define TVM_TL_CPU_OP_ATOMIC_RMW_H_

#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>

#include <string>

#include "op/atomic_reduce.h"
#include "op/utils.h"
#include "support/check.h"
#include "transform/loop_partition.h"

namespace tvm {
namespace tl {
namespace cpu {

using namespace tirx;

/// Combine operator of an atomic update: `dst = dst <combine> value`.
enum class AtomicCombine { kAdd, kMax, kMin };

inline PrimExpr MakeAtomicCombine(AtomicCombine combine, const PrimExpr &old,
                                  const PrimExpr &value) {
  switch (combine) {
  case AtomicCombine::kAdd:
    return old + value;
  case AtomicCombine::kMax:
    return Max(old, value);
  case AtomicCombine::kMin:
    return Min(old, value);
  }
  LOG(FATAL) << "Unreachable atomic combine kind";
  return PrimExpr();
}

/*!
 * \brief Lower an atomic tile op (add/max/min) to a serial RMW loop nest.
 * \param op The atomic op (AtomicAddNode/AtomicMaxNode/AtomicMinNode).
 * \param combine The combine operator matching `op.GetElemOp()`.
 * \param lower_args Lowering context (target, buffer_remap).
 * \param op_name Name used in diagnostics (e.g. "atomic_add").
 */
inline Stmt LowerAtomicRMW(const AtomicOpBaseNode &op, AtomicCombine combine,
                           const LowerArgs &lower_args, const char *op_name) {
  // TMA (cp.reduce) is a CUDA sm90+ feature with no CPU equivalent.
  if (op.annotations.Get("use_tma")) {
    LOG(FATAL) << "CPU " << op_name
               << " does not support use_tma=True: TMA (cp.reduce) is a "
                  "CUDA sm90+ feature. Target was: "
               << lower_args.target->str();
  }

  // buffer_remap resolution (mirror src/cpu/op/reduce.cc).
  auto get_buffer = [&](const Buffer &buffer) {
    auto it = lower_args.buffer_remap.find(buffer);
    return it == lower_args.buffer_remap.end() ? buffer : (*it).second;
  };
  Buffer dst_buffer = get_buffer(op.dst);

  // The dst of a CPU atomic is a global output tensor or a local buffer;
  // shared/fragment scopes do not exist on CPU.
  if (!IsGlobalBuffer(op.dst) && !IsLocalBuffer(op.dst, /*allow_var=*/true)) {
    LOG(FATAL) << "CPU " << op_name
               << " only supports global/local dst buffers, got dst scope `"
               << op.dst.scope() << "`.";
  }

  // One loop var per non-unit dst extent, in order; the k-th non-unit dst
  // dim pairs with the k-th non-unit src dim. This is the same convention as
  // the GPU MakeIterVars/MakeIndices (backend/common/op/atomic_reduce.h),
  // and matches the frontend's legalize_pairwise_extents, which may leave
  // src/dst ranges with different ndim but an equal count of non-unit dims.
  Array<Var> loop_vars;
  Array<PrimExpr> loop_extents;
  Array<PrimExpr> dst_indices;
  for (size_t k = 0; k < op.dst_range.size(); ++k) {
    const Range &range = op.dst_range[k];
    if (is_one(range->extent)) {
      dst_indices.push_back(range->min);
    } else {
      Var var("i" + std::to_string(loop_vars.size()), range->extent->dtype);
      loop_vars.push_back(var);
      loop_extents.push_back(range->extent);
      dst_indices.push_back(range->min + var);
    }
  }

  PrimExpr value = op.src_value;
  if (!op.src_value.defined()) {
    Buffer src_buffer = get_buffer(op.src);
    Array<PrimExpr> src_indices;
    size_t var_idx = 0;
    for (size_t k = 0; k < op.src_range.size(); ++k) {
      const Range &range = op.src_range[k];
      if (is_one(range->extent)) {
        src_indices.push_back(range->min);
      } else {
        ICHECK_LT(var_idx, loop_vars.size())
            << "CPU " << op_name
            << ": src region has more non-unit extents than dst region "
               "(src ndim="
            << op.src_range.size() << ", dst ndim=" << op.dst_range.size()
            << ").";
        src_indices.push_back(range->min + loop_vars[var_idx]);
        ++var_idx;
      }
    }
    ICHECK_EQ(var_idx, loop_vars.size())
        << "CPU " << op_name
        << ": src and dst regions must have the same number of non-unit "
           "extents after frontend extent legalization, got "
        << var_idx << " vs " << loop_vars.size() << ".";
    value = BufferLoad(src_buffer, src_indices);
  }
  if (value->dtype != dst_buffer->dtype) {
    value = Cast(dst_buffer->dtype, value);
  }

  // Serial read-modify-write: plain BufferStore, no atomic intrinsic.
  Stmt body = BufferStore(
      dst_buffer,
      MakeAtomicCombine(combine, BufferLoad(dst_buffer, dst_indices), value),
      dst_indices);

  // kSerial loops wrapped with PragmaUnrollLoop, matching the fill.cc /
  // reduce.cc CPU op convention (transpose.cc does not use it); under the
  // default UnrollLoopConfig (explicit_unroll=false) the tag is a no-op
  // marker.
  for (int i = static_cast<int>(loop_vars.size()) - 1; i >= 0; --i) {
    body = For(loop_vars[i], 0, loop_extents[i], ForKind::kSerial, body,
               std::nullopt);
  }
  if (!loop_vars.empty()) {
    body = PragmaUnrollLoop(Downcast<For>(body));
  }
  return body;
}

} // namespace cpu
} // namespace tl
} // namespace tvm

#endif // TVM_TL_CPU_OP_ATOMIC_RMW_H_
