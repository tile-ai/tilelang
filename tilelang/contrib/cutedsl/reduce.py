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
    "CumMax1D",
    "CumMax2D",
    "NamedBarrier",
    "AllReduce",
]

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32
from cutlass._mlir.dialects import nvvm
from cutlass.cute.arch.nvvm_wrappers import shuffle_sync_op


def min(a, b, c=None):
    """Type-preserving min for scalar CuTeDSL values."""
    result = cutlass.min(a, b)
    if c is not None:
        result = cutlass.min(result, c)
    return result


def max(a, b, c=None):
    """Type-preserving max for scalar CuTeDSL values."""
    result = cutlass.max(a, b)
    if c is not None:
        result = cutlass.max(result, c)
    return result


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
from .warp import __shfl_sync, __shfl_up_sync, __shfl_down_sync


def _warp_prefix_sum_forward(val, lane, MASK=0xFFFFFFFF):
    """
    Warp-level inclusive prefix sum (forward).
    Uses shfl.up to propagate values from lower lanes.
    """
    # Unrolled loop for WARP_SIZE=32: off = 1, 2, 4, 8, 16
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
    WARP_SIZE = 32
    # Unrolled loop for WARP_SIZE=32: off = 1, 2, 4, 8, 16
    n = __shfl_down_sync(MASK, val, 1)
    val = cutlass.select_(lane < WARP_SIZE - 1, val + n, val)
    n = __shfl_down_sync(MASK, val, 2)
    val = cutlass.select_(lane < WARP_SIZE - 2, val + n, val)
    n = __shfl_down_sync(MASK, val, 4)
    val = cutlass.select_(lane < WARP_SIZE - 4, val + n, val)
    n = __shfl_down_sync(MASK, val, 8)
    val = cutlass.select_(lane < WARP_SIZE - 8, val + n, val)
    n = __shfl_down_sync(MASK, val, 16)
    val = cutlass.select_(lane < WARP_SIZE - 16, val + n, val)
    return val


def _warp_prefix_max_forward(val, lane, MASK=0xFFFFFFFF):
    """Warp-level inclusive prefix max (forward)."""
    n = __shfl_up_sync(MASK, val, 1)
    val = cutlass.select_(lane >= 1, max(val, n), val)
    n = __shfl_up_sync(MASK, val, 2)
    val = cutlass.select_(lane >= 2, max(val, n), val)
    n = __shfl_up_sync(MASK, val, 4)
    val = cutlass.select_(lane >= 4, max(val, n), val)
    n = __shfl_up_sync(MASK, val, 8)
    val = cutlass.select_(lane >= 8, max(val, n), val)
    n = __shfl_up_sync(MASK, val, 16)
    val = cutlass.select_(lane >= 16, max(val, n), val)
    return val


def _warp_prefix_max_reverse(val, lane, active, MASK=0xFFFFFFFF):
    """Warp-level inclusive prefix max (reverse)."""
    n = __shfl_down_sync(MASK, val, 1)
    val = cutlass.select_(lane + 1 < active, max(val, n), val)
    n = __shfl_down_sync(MASK, val, 2)
    val = cutlass.select_(lane + 2 < active, max(val, n), val)
    n = __shfl_down_sync(MASK, val, 4)
    val = cutlass.select_(lane + 4 < active, max(val, n), val)
    n = __shfl_down_sync(MASK, val, 8)
    val = cutlass.select_(lane + 8 < active, max(val, n), val)
    n = __shfl_down_sync(MASK, val, 16)
    val = cutlass.select_(lane + 16 < active, max(val, n), val)
    return val


@cute.jit
def _scan_line_sum(src_tensor, dst_tensor, base_offset, extent, stride, lane, reverse, MASK=0xFFFFFFFF):
    """Inclusive sum scan over one strided line using one warp."""
    WARP_SIZE = 32
    carry = src_tensor.element_type(0)
    has_carry = False
    num_segments = (extent + WARP_SIZE - 1) // WARP_SIZE

    if reverse:
        for seg_offset in range(num_segments):
            seg = num_segments - 1 - seg_offset
            base = seg * WARP_SIZE
            active = min(extent - base, WARP_SIZE)
            val = src_tensor.element_type(0)
            if lane < active:
                val = src_tensor[base_offset + (base + lane) * stride]

            val = _warp_prefix_sum_reverse(val, lane, MASK)
            if has_carry and lane < active:
                val = val + carry
            if lane < active:
                dst_tensor[base_offset + (base + lane) * stride] = val

            carry = __shfl_sync(MASK, val, 0)
            has_carry = True
    else:
        for seg in range(num_segments):
            base = seg * WARP_SIZE
            active = min(extent - base, WARP_SIZE)
            val = src_tensor.element_type(0)
            if lane < active:
                val = src_tensor[base_offset + (base + lane) * stride]

            val = _warp_prefix_sum_forward(val, lane, MASK)
            if has_carry and lane < active:
                val = val + carry
            if lane < active:
                dst_tensor[base_offset + (base + lane) * stride] = val

            carry = __shfl_sync(MASK, val, active - 1)
            has_carry = True


@cute.jit
def _scan_line_max(src_tensor, dst_tensor, base_offset, extent, stride, lane, reverse, MASK=0xFFFFFFFF):
    """Inclusive max scan over one strided line using one warp."""
    WARP_SIZE = 32
    carry = src_tensor.element_type(0)
    has_carry = False
    num_segments = (extent + WARP_SIZE - 1) // WARP_SIZE

    if reverse:
        for seg_offset in range(num_segments):
            seg = num_segments - 1 - seg_offset
            base = seg * WARP_SIZE
            active = min(extent - base, WARP_SIZE)
            val = src_tensor.element_type(0)
            if lane < active:
                val = src_tensor[base_offset + (base + lane) * stride]

            val = _warp_prefix_max_reverse(val, lane, active, MASK)
            if has_carry and lane < active:
                val = max(val, carry)
            if lane < active:
                dst_tensor[base_offset + (base + lane) * stride] = val

            carry = __shfl_sync(MASK, val, 0)
            has_carry = True
    else:
        for seg in range(num_segments):
            base = seg * WARP_SIZE
            active = min(extent - base, WARP_SIZE)
            val = src_tensor.element_type(0)
            if lane < active:
                val = src_tensor[base_offset + (base + lane) * stride]

            val = _warp_prefix_max_forward(val, lane, MASK)
            if has_carry and lane < active:
                val = max(val, carry)
            if lane < active:
                dst_tensor[base_offset + (base + lane) * stride] = val

            carry = __shfl_sync(MASK, val, active - 1)
            has_carry = True


class CumSum1D:
    """
    1D cumulative sum operation.
    Based on tl::CumSum1D from scan.h

    Template params:
        threads: Number of threads
        reverse: Whether to cumsum in reverse order
    """

    def __init__(self, threads: cutlass.Constexpr[int], reverse: cutlass.Constexpr[bool]):
        self.threads = threads
        self.reverse = reverse
        self.WARP_SIZE = 32

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
        lane = tidx % self.WARP_SIZE

        src_tensor = cute.make_tensor(src, (N,))
        dst_tensor = cute.make_tensor(dst, (N,))

        if tidx < self.WARP_SIZE:
            _scan_line_sum(src_tensor, dst_tensor, 0, N, 1, lane, self.reverse, MASK)

    @cute.jit
    def run_auto(self, src: cute.Pointer, dst: cute.Pointer, N):
        self.run(src, dst, N)


class CumSum2D:
    """
    2D cumulative sum operation.
    Based on tl::CumSum2D from scan.h

    Template params:
        threads: Number of threads (must be power of 2, 32-1024)
        dim: Axis along which to cumsum (0 or 1)
        reverse: Whether to cumsum in reverse order
    """

    def __init__(self, threads: cutlass.Constexpr[int], dim: cutlass.Constexpr[int], reverse: cutlass.Constexpr[bool]):
        self.threads = threads
        self.dim = dim
        self.reverse = reverse
        self.WARP_SIZE = 32
        self.TILE_H = threads // self.WARP_SIZE

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
        lane = tidx % self.WARP_SIZE
        item = tidx // self.WARP_SIZE
        tile = self.threads // self.WARP_SIZE

        src_tensor = cute.make_tensor(src, (H * W,))
        dst_tensor = cute.make_tensor(dst, (H * W,))

        if self.dim == 1:
            num_blocks = (H + tile - 1) // tile
            for block in cutlass.range_constexpr(num_blocks):
                row = block * tile + item
                if row < H:
                    _scan_line_sum(src_tensor, dst_tensor, row * W, W, 1, lane, self.reverse, MASK)
        else:
            num_blocks = (W + tile - 1) // tile
            for block in cutlass.range_constexpr(num_blocks):
                col = block * tile + item
                if col < W:
                    _scan_line_sum(src_tensor, dst_tensor, col, H, W, lane, self.reverse, MASK)

    @cute.jit
    def run_auto(self, src: cute.Pointer, dst: cute.Pointer, H, W):
        self.run(src, dst, H, W)


class CumMax1D:
    """
    1D cumulative maximum operation.
    Based on tl::CumMax1D from scan.h
    """

    def __init__(self, threads: cutlass.Constexpr[int], reverse: cutlass.Constexpr[bool]):
        self.threads = threads
        self.reverse = reverse
        self.WARP_SIZE = 32

    @cute.jit
    def run(self, src: cute.Pointer, dst: cute.Pointer, N):
        MASK = 0xFFFFFFFF
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % self.WARP_SIZE

        src_tensor = cute.make_tensor(src, (N,))
        dst_tensor = cute.make_tensor(dst, (N,))

        if tidx < self.WARP_SIZE:
            _scan_line_max(src_tensor, dst_tensor, 0, N, 1, lane, self.reverse, MASK)

    @cute.jit
    def run_auto(self, src: cute.Pointer, dst: cute.Pointer, N):
        self.run(src, dst, N)


class CumMax2D:
    """
    2D cumulative maximum operation.
    Based on tl::CumMax2D from scan.h
    """

    def __init__(self, threads: cutlass.Constexpr[int], dim: cutlass.Constexpr[int], reverse: cutlass.Constexpr[bool]):
        self.threads = threads
        self.dim = dim
        self.reverse = reverse
        self.WARP_SIZE = 32

    @cute.jit
    def run(self, src: cute.Pointer, dst: cute.Pointer, H, W):
        MASK = 0xFFFFFFFF
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % self.WARP_SIZE
        item = tidx // self.WARP_SIZE
        tile = self.threads // self.WARP_SIZE

        src_tensor = cute.make_tensor(src, (H * W,))
        dst_tensor = cute.make_tensor(dst, (H * W,))

        if self.dim == 1:
            num_blocks = (H + tile - 1) // tile
            for block in cutlass.range_constexpr(num_blocks):
                row = block * tile + item
                if row < H:
                    _scan_line_max(src_tensor, dst_tensor, row * W, W, 1, lane, self.reverse, MASK)
        else:
            num_blocks = (W + tile - 1) // tile
            for block in cutlass.range_constexpr(num_blocks):
                col = block * tile + item
                if col < W:
                    _scan_line_max(src_tensor, dst_tensor, col, H, W, lane, self.reverse, MASK)

    @cute.jit
    def run_auto(self, src: cute.Pointer, dst: cute.Pointer, H, W):
        self.run(src, dst, H, W)


class NamedBarrier:
    """Named barrier policy for AllReduce, uses bar.sync instead of __syncthreads.
    Based on tl::NamedBarrier<all_threads> from reduce.h"""

    def __init__(self, all_threads):
        self.all_threads = all_threads


def AllReduce(reducer, threads, scale, thread_offset, all_threads=None, batch_size=1, workspace_stride=0):
    """
    AllReduce operation implementing warp/block-level reduction.
    Based on tl::AllReduce from reduce.h

    Args:
        reducer: Reducer operator class (SumOp, MaxOp, etc.)
        threads: Number of threads participating in reduction
        scale: Reduction scale factor
        thread_offset: Thread ID offset
        all_threads: Total number of threads in block (or NamedBarrier instance)
        batch_size: Number of elements per thread to reduce in parallel (default 1)
        workspace_stride: Stride between batch channels in shared memory (default 0)

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
            batch_size: cutlass.Constexpr[int],
            workspace_stride: cutlass.Constexpr[int],
        ):
            self.reducer = reducer
            self.threads = threads
            self.scale = scale
            self.thread_offset = thread_offset
            self.all_threads = all_threads if all_threads is not None else threads
            self.use_named_barrier = use_named_barrier
            self.batch_size = batch_size
            self.workspace_stride = workspace_stride

        def run(self, x, red_buf: cute.Pointer = None):
            """
            Perform all-reduce across threads.
            Based on tl::AllReduce<...>::run from reduce.h
            When NamedBarrier is used, delegates to run_hopper.
            Supports both scalar (x is a value) and batched (x is a pointer) modes.
            """
            if self.use_named_barrier:
                return self.run_hopper(x, red_buf)

            offset = self.threads // 2

            if offset >= 32:
                cute.arch.sync_threads()
                tidx, _, _ = cute.arch.thread_idx()
                if self.batch_size > 1:
                    x_tensor = cute.make_tensor(x, (self.batch_size,))
                    for i in range(self.batch_size):
                        cute.make_tensor(red_buf + (tidx - self.thread_offset) + i * self.workspace_stride, (1,))[0] = x_tensor[i]
                    cute.arch.sync_threads()
                    for i in range(self.batch_size):
                        x_tensor[i] = self.reducer()(
                            x_tensor[i],
                            cute.make_tensor(red_buf + ((tidx - self.thread_offset) ^ offset) + i * self.workspace_stride, (1,))[0],
                        )
                else:
                    cute.make_tensor(red_buf + tidx - self.thread_offset, (1,))[0] = x
                    cute.arch.sync_threads()
                    x = self.reducer()(x, cute.make_tensor(red_buf + ((tidx - self.thread_offset) ^ offset), (1,))[0])
            else:
                if self.batch_size > 1:
                    x_tensor = cute.make_tensor(x, (self.batch_size,))
                    for i in range(self.batch_size):
                        other = shuffle_sync_op(x_tensor[i], offset, mask=0xFFFFFFFF, mask_and_clamp=0x1F, kind=nvvm.ShflKind.bfly)
                        x_tensor[i] = self.reducer()(x_tensor[i], other)
                else:
                    other = shuffle_sync_op(x, offset, mask=0xFFFFFFFF, mask_and_clamp=0x1F, kind=nvvm.ShflKind.bfly)
                    x = self.reducer()(x, other)

            if offset == self.scale:
                return x
            else:
                return AllReduce(
                    self.reducer, offset, self.scale, self.thread_offset, self.all_threads, self.batch_size, self.workspace_stride
                ).run(x, red_buf)

        def run_hopper(self, x, red_buf: cute.Pointer = None):
            """
            Perform all-reduce on Hopper architecture using bar.sync.
            Based on tl::AllReduce<...>::run_hopper from reduce.h
            Supports both scalar and batched modes.
            """
            offset = self.threads // 2
            tidx, _, _ = cute.arch.thread_idx()
            if offset >= 32:
                bar_sync_ptx(1, self.all_threads)
                if self.batch_size > 1:
                    x_tensor = cute.make_tensor(x, (self.batch_size,))
                    for i in range(self.batch_size):
                        cute.make_tensor(red_buf + (tidx - self.thread_offset) + i * self.workspace_stride, (1,))[0] = x_tensor[i]
                    bar_sync_ptx(2, self.all_threads)
                    for i in range(self.batch_size):
                        x_tensor[i] = self.reducer()(
                            x_tensor[i],
                            cute.make_tensor(red_buf + ((tidx - self.thread_offset) ^ offset) + i * self.workspace_stride, (1,))[0],
                        )
                else:
                    cute.make_tensor(red_buf + tidx - self.thread_offset, (1,))[0] = x
                    bar_sync_ptx(2, self.all_threads)
                    x = self.reducer()(x, cute.make_tensor(red_buf + ((tidx - self.thread_offset) ^ offset), (1,))[0])
            else:
                if self.batch_size > 1:
                    x_tensor = cute.make_tensor(x, (self.batch_size,))
                    for i in range(self.batch_size):
                        other = shuffle_sync_op(x_tensor[i], offset, mask=0xFFFFFFFF, mask_and_clamp=0x1F, kind=nvvm.ShflKind.bfly)
                        x_tensor[i] = self.reducer()(x_tensor[i], other)
                else:
                    other = shuffle_sync_op(x, offset, mask=0xFFFFFFFF, mask_and_clamp=0x1F, kind=nvvm.ShflKind.bfly)
                    x = self.reducer()(x, other)

            if offset == self.scale:
                return x
            else:
                return AllReduce(
                    self.reducer, offset, self.scale, self.thread_offset, self.all_threads, self.batch_size, self.workspace_stride
                ).run_hopper(x, red_buf)

    return AllReduceInstance(reducer, threads, scale, thread_offset, barrier_threads, use_named_barrier, batch_size, workspace_stride)
