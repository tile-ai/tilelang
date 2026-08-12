/*!
 * \file tl/cpu/op/reduce.cc
 * \brief CPU implementation for tl.reduce lowering (serial, local buffers).
 *
 * First version is purely serial: the reduce dimension is accumulated in a
 * kSerial loop with no SIMD horizontal reduction and no thread-level
 * parallelism. Correctness and test coverage first; performance work
 * (SIMD / multi-thread reduce) is tracked in cpu_ops.md enhancement track.
 *
 * Scope is limited to local -> local/local.var buffers, which is the only
 * path the frontend emits on CPU (reduce_op.py:69-81 emits ReduceOp directly
 * only for local src with local/local.var dst; shared/fragment paths go
 * through alloc_fragment, which CPU does not have).
 */

#include "op/reduce.h"

#include <tvm/runtime/logging.h>

#include "backend/common/op/reduce.h"
#include "backend/common/target_utils.h"
#include "op/utils.h"
#include "support/check.h"
#include "transform/loop_partition.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace cpu {

struct Reduce {
  static Stmt Lower(const ReduceOpNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    (void)analyzer;

    // 1. nan_propagate guard: CPU codegen has no __hmax_nan/__hmin_nan
    //    equivalent. Only meaningful for fp16/bf16 max/min/absmax; other
    //    dtypes ignore the flag.
    if (op.nan_propagate &&
        (op.dst->dtype.is_float16() || op.dst->dtype.is_bfloat16())) {
      LOG(FATAL) << "CPU reduce does not support nan_propagate=True for "
                    "float16/bfloat16 max/min/absmax: the CPU codegen has no "
                    "__hmax_nan/__hmin_nan equivalent intrinsics (CUDA-only). "
                    "Target was: "
                 << lower_args.target->str();
    }

    // 2. buffer_remap resolution (mirror backend/common/op/reduce.h:753-756).
    auto get_buffer = [&](const Buffer &buffer) {
      auto it = lower_args.buffer_remap.find(buffer);
      return it == lower_args.buffer_remap.end() ? buffer : (*it).second;
    };
    Buffer src_buffer = get_buffer(op.src);
    Buffer dst_buffer = get_buffer(op.dst);

    // 3. Scope guard: only local src and local/local.var dst are accepted.
    //    The frontend only emits this combination on CPU; give a readable
    //    error for anything else instead of silently producing wrong code.
    if (!IsLocalBuffer(op.src) || !IsLocalBuffer(op.dst, /*allow_var=*/true)) {
      LOG(FATAL) << "CPU reduce only supports local src and "
                    "local/local.var dst buffers, got src scope `"
                 << op.src.scope() << "` and dst scope `" << op.dst.scope()
                 << "`.";
    }

    // 4. Constraint checks.
    //    batch is a GPU AllReduce barrier batching concept; CPU has no
    //    thread-level reduction so batch > 1 is rejected.
    ICHECK_EQ(op.batch, 1)
        << "CPU reduce: batch > 1 is a GPU AllReduce barrier concept and "
           "is not supported on CPU (no thread-level reduction).";
    int src_dim = static_cast<int>(op.src->shape.size());
    int dst_dim = static_cast<int>(op.dst->shape.size());
    // dim should already be legalized to >= 0 by the frontend
    // (_legalize_dim in reduce_op.py:12-15), but defend in depth.
    ICHECK_GE(op.dim, 0) << "CPU reduce: dim must be non-negative";
    ICHECK_LT(op.dim, src_dim) << "CPU reduce: dim " << op.dim
                               << " out of range for src ndim " << src_dim;
    ICHECK(dst_dim == src_dim - 1 || dst_dim == src_dim)
        << "CPU reduce: dst ndim must be src_ndim-1 (no keepdim) or "
           "src_ndim (keepdim), got src_ndim="
        << src_dim << ", dst_ndim=" << dst_dim;
    ICHECK(src_buffer->dtype == dst_buffer->dtype)
        << "CPU reduce: src and dst dtypes must match, got "
        << src_buffer->dtype << " vs " << dst_buffer->dtype;

    // 5. Index construction (mirror backend LowerLocal reduce.h:651-671).
    Array<Var> dst_vars;
    Array<PrimExpr> dst_indices;
    for (int i = 0; i < dst_dim; ++i) {
      Var var("i" + std::to_string(i));
      dst_vars.push_back(var);
      dst_indices.push_back(var);
    }

    auto make_src_indices = [&](PrimExpr reduce_index) {
      Array<PrimExpr> indices;
      for (int i = 0; i < src_dim; ++i) {
        if (i == op.dim) {
          indices.push_back(reduce_index);
        } else if (dst_dim == src_dim) {
          // keepdim: dst has a size-1 axis at op.dim
          indices.push_back(dst_vars[i]);
        } else {
          indices.push_back(dst_vars[i < op.dim ? i : i - 1]);
        }
      }
      return indices;
    };

    Array<Stmt> stmts;

    // 6. Optional init using the type/dtype-correct identity value
    //    (backend::reduce::MakeInitValue covers all 8 types x dtype split).
    if (op.clear) {
      stmts.push_back(BufferStore(
          dst_buffer, backend::reduce::MakeInitValue(op), dst_indices));
    }

    // 7. Inner reduce loop along op.dim. kSerial (NOT kUnrolled) to avoid
    //    code-size blowup for large reduce extents compounded by JIT -O0.
    //    clear=False accumulation falls out naturally: init is skipped and
    //    MakeReduce reads the existing dst value as the accumulator.
    Var rv("rv");
    Stmt reduce_body =
        BufferStore(dst_buffer,
                    backend::reduce::MakeReduce(
                        op, /*vsize=*/1, BufferLoad(dst_buffer, dst_indices),
                        BufferLoad(src_buffer, make_src_indices(rv))),
                    dst_indices);
    stmts.push_back(For(rv, 0, op.src->shape[op.dim], ForKind::kSerial,
                        reduce_body, std::nullopt));

    // 8. Outer dst-dim loops. Built as kSerial then wrapped with
    //    PragmaUnrollLoop to match the fill.cc CPU op convention.
    //    PragmaUnrollLoop (LoopPramaUnroller, loop_partition.cc)
    //    only retags the outermost kSerial For it meets and returns without
    //    recursing into the body, so the inner kSerial reduce loop above is
    //    left untouched. Under the default UnrollLoopConfig
    //    (explicit_unroll=false) this is a no-op (no loop body replication);
    //    the tag is an intent marker consistent with other CPU ops.
    Stmt body = SeqStmt::Flatten(stmts);
    for (int i = dst_dim - 1; i >= 0; --i) {
      body = For(dst_vars[i], 0, op.dst->shape[i], ForKind::kSerial, body,
                 std::nullopt);
    }
    if (dst_dim >= 1) {
      body = PragmaUnrollLoop(Downcast<For>(body));
    }
    return body;
  }
};

} // namespace cpu

namespace {

bool MatchCPUReduceTarget(Target target) { return TargetIsCPU(target); }

bool RegisterCPUReduce() {
  RegisterReduceImpl(ReduceImpl{
      "cpu.Reduce",
      MatchCPUReduceTarget,
      cpu::Reduce::Lower,
  });
  return true;
}

const bool cpu_reduce_registered = RegisterCPUReduce();

} // namespace

} // namespace tl
} // namespace tvm
