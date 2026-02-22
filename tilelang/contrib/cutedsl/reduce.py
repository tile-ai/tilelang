"""
Reduce operations for CuTeDSL backend.
Based on tl_templates/cuda/reduce.h
"""

from __future__ import annotations

__all__ = [
    "min",
    "max",
    "SumOp",
    "MaxOp",
    "MinOp",
    "BitAndOp",
    "BitOrOp",
    "BitXorOp",
    "bar_sync",
    "bar_sync_ptx",
    "CumSum1D",
    "CumSum2D",
    "NamedBarrier",
    "AllReduce",
]

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32, Float32
from cutlass.base_dsl.typing import Numeric
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import arith, nvvm
from cutlass.cute.arch.nvvm_wrappers import shuffle_sync_op


def _is_int_type(val):
    """Check if a value is an integer Numeric type."""
    if isinstance(val, Int32):
        return True
    if isinstance(val, Numeric) and hasattr(val, "mlir_type"):
        from cutlass._mlir import ir as mlir_ir

        return isinstance(val.mlir_type, mlir_ir.IntegerType)
    if isinstance(val, int) and not isinstance(val, bool):
        return True
    # Check for signless integer ArithValue (from DSL expressions)
    if hasattr(val, "ir_value"):
        try:
            from cutlass._mlir import ir as mlir_ir

            ir_val = val.ir_value()
            if hasattr(ir_val, "type") and isinstance(ir_val.type, mlir_ir.IntegerType):
                return True
        except Exception:
            pass
    return False


@dsl_user_op
def _fmin(a, b, c=None, *, loc=None, ip=None):
    return Float32(
        nvvm.fmin(
            T.f32(),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _imin(a, b, *, loc=None, ip=None):
    return Int32(
        arith.minsi(
            Int32(a).ir_value(loc=loc, ip=ip),
            Int32(b).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fmax(a, b, c=None, *, loc=None, ip=None):
    return Float32(
        nvvm.fmax(
            T.f32(),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _imax(a, b, *, loc=None, ip=None):
    return Int32(
        arith.maxsi(
            Int32(a).ir_value(loc=loc, ip=ip),
            Int32(b).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


def min(a, b, c=None):
    """Type-aware min: uses arith.minsi for integers, nvvm.fmin for floats.
    Falls back to integer path if float conversion fails (signless int types)."""
    if _is_int_type(a) and _is_int_type(b):
        return _imin(a, b)
    try:
        return _fmin(a, b, c)
    except Exception:
        # Float32 conversion may fail for signless integer types
        return _imin(a, b)


def max(a, b, c=None):
    """Type-aware max: uses arith.maxsi for integers, nvvm.fmax for floats.
    Falls back to integer path if float conversion fails (signless int types)."""
    if _is_int_type(a) and _is_int_type(b):
        return _imax(a, b)
    try:
        return _fmax(a, b, c)
    except Exception:
        # Float32 conversion may fail for signless integer types
        return _imax(a, b)


class SumOp:
    """Sum reduction operator"""

    @staticmethod
    def __call__(x, y):
        return x + y


class MaxOp:
    """Max reduction operator"""

    @staticmethod
    def __call__(x, y):
        return max(x, y)


class MinOp:
    """Min reduction operator"""

    @staticmethod
    def __call__(x, y):
        # Use cutlass.min which is JIT-friendly
        return min(x, y)


class BitAndOp:
    """Bitwise AND reduction operator"""

    @staticmethod
    def __call__(x, y):
        return x & y


class BitOrOp:
    """Bitwise OR reduction operator"""

    @staticmethod
    def __call__(x, y):
        return x | y


class BitXorOp:
    """Bitwise XOR reduction operator"""

    @staticmethod
    def __call__(x, y):
        return x ^ y


def bar_sync(barrier_id, number_of_threads):
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=number_of_threads)


def bar_sync_ptx(barrier_id, number_of_threads):
    from cutlass._mlir.dialects import llvm

    llvm.inline_asm(
        None,
        [Int32(barrier_id).ir_value(), Int32(number_of_threads).ir_value()],
        "bar.sync $0, $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# Import shuffle functions from warp module
from .warp import __shfl_up_sync, __shfl_down_sync


def _warp_prefix_sum_forward(val, lane, MASK=0xFFFFFFFF):
    """
    Warp-level inclusive prefix sum (forward).
    Uses shfl.up to propagate values from lower lanes.
    """
    # Unrolled loop for SEG=32: off = 1, 2, 4, 8, 16
    n = __shfl_up_sync(MASK, val, 1)
    val = cutlass.select_(lane >= 1, val + n, val)
    n = __shfl_up_sync(MASK, val, 2)
    val = cutlass.select_(lane >= 2, val + n, val)
    n = __shfl_up_sync(MASK, val, 4)
    val = cutlass.select_(lane >= 4, val + n, val)
    n = __shfl_up_sync(MASK, val, 8)
    val = cutlass.select_(lane >= 8, val + n, val)
    n = __shfl_up_sync(MASK, val, 16)
    val = cutlass.select_(lane >= 16, val + n, val)
    return val


def _warp_prefix_sum_reverse(val, lane, MASK=0xFFFFFFFF):
    """
    Warp-level inclusive prefix sum (reverse).
    Uses shfl.down to propagate values from higher lanes.
    """
    SEG = 32
    # Unrolled loop for SEG=32: off = 1, 2, 4, 8, 16
    n = __shfl_down_sync(MASK, val, 1)
    val = cutlass.select_(lane < SEG - 1, val + n, val)
    n = __shfl_down_sync(MASK, val, 2)
    val = cutlass.select_(lane < SEG - 2, val + n, val)
    n = __shfl_down_sync(MASK, val, 4)
    val = cutlass.select_(lane < SEG - 4, val + n, val)
    n = __shfl_down_sync(MASK, val, 8)
    val = cutlass.select_(lane < SEG - 8, val + n, val)
    n = __shfl_down_sync(MASK, val, 16)
    val = cutlass.select_(lane < SEG - 16, val + n, val)
    return val


class CumSum1D:
    """
    1D cumulative sum operation.
    Based on tl::CumSum1D from reduce.h

    Template params:
        threads: Number of threads
        reverse: Whether to cumsum in reverse order
    """

    def __init__(self, threads: cutlass.Constexpr[int], reverse: cutlass.Constexpr[bool]):
        self.threads = threads
        self.reverse = reverse
        self.SEG = 32  # Warp size

    @cute.jit
    def run(self, src: cute.Pointer, dst: cute.Pointer, N):
        """
        Perform 1D cumulative sum.

        Args:
            src: Source pointer
            dst: Destination pointer
            N: Number of elements (must be compile-time constant or small)
        """
        MASK = 0xFFFFFFFF
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % self.SEG

        src_tensor = cute.make_tensor(src, (N,))
        dst_tensor = cute.make_tensor(dst, (N,))

        # Load value (0 if out of bounds)
        val = Float32(0.0)
        if tidx < N:
            val = src_tensor[tidx]

        # Warp-level prefix sum
        if self.reverse:
            val = _warp_prefix_sum_reverse(val, lane, MASK)
        else:
            val = _warp_prefix_sum_forward(val, lane, MASK)

        # Store result - only valid threads write
        if tidx < N:
            dst_tensor[tidx] = val


class CumSum2D:
    """
    2D cumulative sum operation.
    Based on tl::CumSum2D from reduce.h

    Template params:
        threads: Number of threads (must be power of 2, 32-1024)
        dim: Axis along which to cumsum (0 or 1)
        reverse: Whether to cumsum in reverse order
    """

    def __init__(self, threads: cutlass.Constexpr[int], dim: cutlass.Constexpr[int], reverse: cutlass.Constexpr[bool]):
        self.threads = threads
        self.dim = dim
        self.reverse = reverse
        self.SEG = 32  # Warp size
        self.TILE_H = threads // 32

    @cute.jit
    def run(self, src: cute.Pointer, dst: cute.Pointer, H, W):
        """
        Perform 2D cumulative sum.

        Args:
            src: Source pointer
            dst: Destination pointer
            H: Number of rows
            W: Number of columns (should be <= 32 for single-segment case)
        """
        MASK = 0xFFFFFFFF
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % self.SEG
        row = tidx // self.SEG

        src_tensor = cute.make_tensor(src, (H * W,))
        dst_tensor = cute.make_tensor(dst, (H * W,))

        # For 2D cumsum along dim=1 (row-wise cumsum):
        # Each warp handles one row, lane id is the column index
        # For dim=0 (column-wise), interpretation is swapped

        if self.dim == 1:
            # Row-wise cumsum: each warp processes one row
            # row = which row this warp handles
            # lane = column index within the row
            col = lane
            # Linear index into the flattened buffer
            idx = row * W + col

            # Load value (0 if out of bounds)
            val = Float32(0.0)
            if row < H and col < W:
                val = src_tensor[idx]

            # Warp-level prefix sum along the row
            if self.reverse:
                val = _warp_prefix_sum_reverse(val, lane, MASK)
            else:
                val = _warp_prefix_sum_forward(val, lane, MASK)

            # Store result - only valid threads write
            if row < H and col < W:
                dst_tensor[idx] = val
        else:
            # Column-wise cumsum (dim=0): each warp processes one column
            # Each lane maps to a row index, so H must be <= 32 (warp size).
            assert H <= 32, (
                f"CumSum2D dim=0 only supports H <= 32 (got H={H}). Use dim=1 for row-wise cumsum or implement multi-warp column iteration."
            )
            col = row  # warp index becomes column index
            row_in_col = lane  # lane becomes row index within column
            idx = row_in_col * W + col

            # Load value (0 if out of bounds)
            val = Float32(0.0)
            if row_in_col < H and col < W:
                val = src_tensor[idx]

            if self.reverse:
                val = _warp_prefix_sum_reverse(val, lane, MASK)
            else:
                val = _warp_prefix_sum_forward(val, lane, MASK)

            # Store result - only valid threads write
            if row_in_col < H and col < W:
                dst_tensor[idx] = val


class NamedBarrier:
    """Named barrier policy for AllReduce, uses bar.sync instead of __syncthreads.
    Based on tl::NamedBarrier<all_threads> from reduce.h"""

    def __init__(self, all_threads):
        self.all_threads = all_threads


def AllReduce(reducer, threads, scale, thread_offset, all_threads=None):
    """
    AllReduce operation implementing warp/block-level reduction.
    Based on tl::AllReduce from reduce.h

    Args:
        reducer: Reducer operator class (SumOp, MaxOp, etc.)
        threads: Number of threads participating in reduction
        scale: Reduction scale factor
        thread_offset: Thread ID offset
        all_threads: Total number of threads in block

    Returns:
        A callable object with run() and run_hopper() methods
    """

    # Detect NamedBarrier: extract all_threads and use bar.sync path
    use_named_barrier = isinstance(all_threads, NamedBarrier)
    if use_named_barrier:
        barrier_threads = all_threads.all_threads
    else:
        barrier_threads = all_threads

    class AllReduceInstance:
        def __init__(
            self,
            reducer,
            threads,
            scale,
            thread_offset: cutlass.Constexpr[int],
            all_threads: cutlass.Constexpr[int],
            use_named_barrier: cutlass.Constexpr[bool],
        ):
            self.reducer = reducer
            self.threads = threads
            self.scale = scale
            self.thread_offset = thread_offset
            self.all_threads = all_threads if all_threads is not None else threads
            self.use_named_barrier = use_named_barrier

        def run(self, x, red_buf: cute.Pointer = None):
            """
            Perform all-reduce across threads.
            Based on tl::AllReduce<...>::run from reduce.h
            When NamedBarrier is used, delegates to run_hopper.
            """
            if self.use_named_barrier:
                return self.run_hopper(x, red_buf)

            offset = self.threads // 2

            if offset >= 32:
                # Use shared memory for large thread counts
                cute.arch.sync_threads()
                tidx, _, _ = cute.arch.thread_idx()
                cute.make_tensor(red_buf + tidx - self.thread_offset, (1,))[0] = x
                cute.arch.sync_threads()
                x = self.reducer()(x, cute.make_tensor(red_buf + ((tidx - self.thread_offset) ^ offset), (1,))[0])
            else:
                # Use warp shuffle for small thread counts
                # Use the pre-existing shuffle_sync_op with butterfly (XOR) mode
                other = shuffle_sync_op(x, offset, mask=0xFFFFFFFF, mask_and_clamp=0x1F, kind=nvvm.ShflKind.bfly)
                x = self.reducer()(x, other)

            return (
                x
                if offset == self.scale
                else AllReduce(self.reducer, offset, self.scale, self.thread_offset, self.all_threads).run(x, red_buf)
            )

        def run_hopper(self, x, red_buf: cute.Pointer = None):
            """
            Perform all-reduce on Hopper architecture using bar.sync.
            Based on tl::AllReduce<...>::run_hopper from reduce.h
            """
            offset = self.threads // 2
            tidx, _, _ = cute.arch.thread_idx()
            if offset >= 32:
                # Use inlined asm for bar.sync to avoid instruction reordering
                bar_sync_ptx(1, self.all_threads)
                cute.make_tensor(red_buf + tidx - self.thread_offset, (1,))[0] = x
                bar_sync_ptx(2, self.all_threads)
                x = self.reducer()(x, cute.make_tensor(red_buf + ((tidx - self.thread_offset) ^ offset), (1,))[0])
            else:
                # Use warp shuffle for small thread counts
                # Use the pre-existing shuffle_sync_op with butterfly (XOR) mode
                other = shuffle_sync_op(x, offset, mask=0xFFFFFFFF, mask_and_clamp=0x1F, kind=nvvm.ShflKind.bfly)
                x = self.reducer()(x, other)

            return (
                x
                if offset == self.scale
                else AllReduce(self.reducer, offset, self.scale, self.thread_offset, self.all_threads).run_hopper(x, red_buf)
            )

    return AllReduceInstance(reducer, threads, scale, thread_offset, barrier_threads, use_named_barrier)
