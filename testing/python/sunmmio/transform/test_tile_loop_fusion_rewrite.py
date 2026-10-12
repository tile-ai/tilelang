import tilelang as tl
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.layout import make_zz_layout
from tilelang.tileview import make_tileview
from tilelang.utils.target import SUNMMIO_TARGET_DESC
import textwrap

IRModule = tvm.IRModule


def apply_tiles_lowering(mod):
    return tl.transform.LowerTilesLoop()(mod)


def apply_sunmmio_tiles_lowering(mod):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.AddWrapperForSingleBufStore()(mod)
        mod = tl.transform.LegalizeNegativeIndex()(mod)
        mod = tl.transform.InjectAssumes()(mod)
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.InferSramScope()(mod)
        mod = tl.transform.LayoutReducer()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)
        mod = tl.transform.LowerTilesLoop()(mod)
    return mod


def apply_sunmmio_tile_loop_fusion(mod):
    mod = apply_sunmmio_tiles_lowering(mod)
    return tl.transform.SunmmioTileLoopFusion()(mod)


def single_tile_scope_kernel(block_m=32, block_n=32, tile_size=(8, 32), dtype="float16"):
    @T.prim_func
    def main(A: T.Tensor((block_m, block_n), dtype), B: T.Tensor((block_m, block_n), dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((block_m, block_n), dtype)
            B_shared = T.alloc_shared((block_m, block_n), dtype)

            T.annotate_layout(
                {
                    A_shared: make_zz_layout(A_shared),
                    B_shared: make_zz_layout(B_shared),
                }
            )
            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, tile_size, (-2, -1)),
                    B_shared: make_tileview(B_shared, tile_size, (-2, -1)),
                }
            )

            T.copy(A[0:block_m, 0:block_n], A_shared)
            for i, j in T.Tiles([block_m, block_n], parallel=True):
                B_shared[i, j] = A_shared[i, j]
            T.copy(B_shared, B[0:block_m, 0:block_n])

    return main


def two_consecutive_tile_scopes_kernel(block_m=32, block_n=32, tile_size=(8, 32), dtype="float16"):
    @T.prim_func
    def main(A: T.Tensor((block_m, block_n), dtype), B: T.Tensor((block_m, block_n), dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((block_m, block_n), dtype)
            Tmp_shared = T.alloc_shared((block_m, block_n), dtype)
            B_shared = T.alloc_shared((block_m, block_n), dtype)

            T.annotate_layout(
                {
                    A_shared: make_zz_layout(A_shared),
                    Tmp_shared: make_zz_layout(Tmp_shared),
                    B_shared: make_zz_layout(B_shared),
                }
            )
            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, tile_size, (-2, -1)),
                    Tmp_shared: make_tileview(Tmp_shared, tile_size, (-2, -1)),
                    B_shared: make_tileview(B_shared, tile_size, (-2, -1)),
                }
            )

            T.copy(A[0:block_m, 0:block_n], A_shared)
            for i, j in T.Tiles([block_m, block_n], parallel=True):
                Tmp_shared[i, j] = A_shared[i, j]
            for i, j in T.Tiles([block_m, block_n], parallel=True):
                B_shared[i, j] = Tmp_shared[i, j]
            T.copy(B_shared, B[0:block_m, 0:block_n])

    return main


def flash_attention_online_softmax_tiled_kernel(block_m=32, block_n=32, dim=32, dtype="float32", accum_dtype="float32"):
    scale = 1.0

    @T.prim_func
    def main(
        AccSIn: T.Tensor((block_m, block_n), accum_dtype),
        AccOIn: T.Tensor((block_m, dim), accum_dtype),
        ScoresMaxIn: T.Tensor((block_m,), accum_dtype),
        LogsumIn: T.Tensor((block_m,), accum_dtype),
        AccSCastOut: T.Tensor((block_m, block_n), accum_dtype),
        AccOOut: T.Tensor((block_m, dim), accum_dtype),
        ScoresMaxOut: T.Tensor((block_m,), accum_dtype),
        LogsumOut: T.Tensor((block_m,), accum_dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            acc_s = T.alloc_shared((block_m, block_n), accum_dtype, scope="shared.rsram")
            acc_s_cast = T.alloc_shared((block_m, block_n), accum_dtype, scope="shared.rsram")
            acc_o = T.alloc_shared((block_m, dim), accum_dtype, scope="shared.rsram")
            scores_max = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            scores_max_prev = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            scores_scale = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            scores_sum = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            logsum = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")

            T.copy(AccSIn[0:block_m, 0:block_n], acc_s)
            T.copy(AccOIn[0:block_m, 0:dim], acc_o)
            T.copy(ScoresMaxIn[0:block_m], scores_max)
            T.copy(LogsumIn[0:block_m], logsum)

            for i in T.Tiles([block_m], parallel=True):
                scores_max_prev[i] = scores_max[i]

            for i in T.Tiles([block_m], parallel=True):
                scores_max[i] = -T.infinity(accum_dtype)

            T.reduce(acc_s, scores_max, "max", dim=1, clear=False)

            for i in T.Tiles([block_m], parallel=True):
                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

            for i in T.Tiles([block_m], parallel=True):
                scores_scale[i] = T.exp2(scores_max_prev[i] * T.float32(scale) - scores_max[i] * T.float32(scale))

            for i, j in T.Tiles([block_m, block_n], parallel=True):
                acc_s[i, j] = T.exp2(acc_s[i, j] * T.float32(scale) - scores_max[i] * T.float32(scale))

            T.reduce(acc_s, scores_sum, "sum", dim=1, clear=True)

            for i in T.Tiles([block_m], parallel=True):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

            for i, j in T.Tiles([block_m, block_n], parallel=True):
                acc_s_cast[i, j] = acc_s[i, j]

            for i, j in T.Tiles([block_m, dim], parallel=True):
                acc_o[i, j] *= scores_scale[i]

            T.copy(acc_s_cast, AccSCastOut[0:block_m, 0:block_n])
            T.copy(acc_o, AccOOut[0:block_m, 0:dim])
            T.copy(scores_max, ScoresMaxOut[0:block_m])
            T.copy(logsum, LogsumOut[0:block_m])

    return main


def rms_norm_tiled_kernel(block_m=32, block_n=32, dtype="float32", accum_dtype="float32"):
    eps = 1e-6

    @T.prim_func
    def main(AIn: T.Tensor((block_m, block_n), dtype), AOut: T.Tensor((block_m, block_n), dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            a_shared = T.alloc_shared((block_m, block_n), dtype, scope="shared.rsram")
            a_square = T.alloc_shared((block_m, block_n), accum_dtype, scope="shared.rsram")
            a_out = T.alloc_shared((block_m, block_n), dtype, scope="shared.rsram")
            row_sum = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            row_scale = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")

            T.copy(AIn[0:block_m, 0:block_n], a_shared)

            for i in T.Tiles([block_m], parallel=True):
                row_sum[i] = T.float32(0)

            for i, j in T.Tiles([block_m, block_n], parallel=True):
                a_square[i, j] = a_shared[i, j] * a_shared[i, j]

            T.reduce(a_square, row_sum, "sum", dim=1, clear=False)

            for i in T.Tiles([block_m], parallel=True):
                row_scale[i] = T.rsqrt(row_sum[i] / T.float32(block_n) + T.float32(eps))

            for i, j in T.Tiles([block_m, block_n], parallel=True):
                a_out[i, j] = a_shared[i, j] * row_scale[i]

            T.copy(a_out, AOut[0:block_m, 0:block_n])

    return main


def attr_wrapped_two_region_lowered_kernel(dtype="float16"):
    @T.prim_func
    def main():
        with T.block("root"):
            T.reads()
            T.writes()
            A_shared = T.alloc_buffer((32, 32), dtype, scope="shared.rsram")
            Tmp_shared = T.alloc_buffer((32, 32), dtype, scope="shared.rsram")
            B_shared = T.alloc_buffer((32, 32), dtype, scope="shared.rsram")

            with T.attr("wrapper_scope", "unit_test", 1):
                for i in T.serial(
                    4,
                    annotations={
                        "tile.domain": [T.int32(32), T.int32(32)],
                        "tile.execution_axis": T.int32(0),
                        "tile.execution_domain_axes": [T.int32(0), T.int32(1)],
                        "tile.scope_entry": T.int32(1),
                        "tile.tile_size": [T.int32(8), T.int32(32)],
                    },
                ):
                    for j in T.serial(1, annotations={"tile.execution_axis": T.int32(1)}):
                        for ki in T.serial(8, annotations={"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}):
                            for kj in T.vectorized(32, annotations={"tile.interior": T.int32(1), "tile.interior_axis": T.int32(1)}):
                                Tmp_shared[i * 8 + ki, j * 32 + kj] = A_shared[i * 8 + ki, j * 32 + kj]

            with T.attr("wrapper_scope", "unit_test", 1):
                for i in T.serial(
                    4,
                    annotations={
                        "tile.domain": [T.int32(32), T.int32(32)],
                        "tile.execution_axis": T.int32(0),
                        "tile.execution_domain_axes": [T.int32(0), T.int32(1)],
                        "tile.scope_entry": T.int32(1),
                        "tile.tile_size": [T.int32(8), T.int32(32)],
                    },
                ):
                    for j in T.serial(1, annotations={"tile.execution_axis": T.int32(1)}):
                        for ki in T.serial(8, annotations={"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}):
                            for kj in T.vectorized(32, annotations={"tile.interior": T.int32(1), "tile.interior_axis": T.int32(1)}):
                                B_shared[i * 8 + ki, j * 32 + kj] = Tmp_shared[i * 8 + ki, j * 32 + kj]

    return main


def let_wrapped_two_region_lowered_kernel(dtype="float16"):
    source = f"""
# from tvm.script import tir as T
@T.prim_func
def main():
    A_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")
    Tmp_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")
    B_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")
    x0: T.int32 = 7
    for i in T.serial(4, annotations={{"tile.domain": [T.int32(32), T.int32(32)], "tile.execution_axis": T.int32(0), "tile.execution_domain_axes": [T.int32(0), T.int32(1)], "tile.scope_entry": T.int32(1), "tile.tile_size": [T.int32(8), T.int32(32)]}}):
        for j in T.serial(1, annotations={{"tile.execution_axis": T.int32(1)}}):
            for ki in T.serial(8, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}}):
                for kj in T.vectorized(32, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(1)}}):
                    Tmp_shared[i * 8 + ki, j * 32 + kj] = A_shared[i * 8 + ki, j * 32 + kj] + T.Cast("{dtype}", x0)
    x1: T.int32 = 11
    for i in T.serial(4, annotations={{"tile.domain": [T.int32(32), T.int32(32)], "tile.execution_axis": T.int32(0), "tile.execution_domain_axes": [T.int32(0), T.int32(1)], "tile.scope_entry": T.int32(1), "tile.tile_size": [T.int32(8), T.int32(32)]}}):
        for j in T.serial(1, annotations={{"tile.execution_axis": T.int32(1)}}):
            for ki in T.serial(8, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}}):
                for kj in T.vectorized(32, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(1)}}):
                    B_shared[i * 8 + ki, j * 32 + kj] = Tmp_shared[i * 8 + ki, j * 32 + kj] + T.Cast("{dtype}", x1)
"""
    return tvm.script.from_source(source)


def mixed_plain_and_let_wrapped_two_region_lowered_kernel(dtype="float16"):
    source = f"""
# from tvm.script import tir as T
@T.prim_func
def main():
    A_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")
    Tmp_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")
    B_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")
    for i in T.serial(4, annotations={{"tile.domain": [T.int32(32), T.int32(32)], "tile.execution_axis": T.int32(0), "tile.execution_domain_axes": [T.int32(0), T.int32(1)], "tile.scope_entry": T.int32(1), "tile.tile_size": [T.int32(8), T.int32(32)]}}):
        for j in T.serial(1, annotations={{"tile.execution_axis": T.int32(1)}}):
            for ki in T.serial(8, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}}):
                for kj in T.vectorized(32, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(1)}}):
                    Tmp_shared[i * 8 + ki, j * 32 + kj] = A_shared[i * 8 + ki, j * 32 + kj]
    x1: T.int32 = 7
    for i in T.serial(4, annotations={{"tile.domain": [T.int32(32), T.int32(32)], "tile.execution_axis": T.int32(0), "tile.execution_domain_axes": [T.int32(0), T.int32(1)], "tile.scope_entry": T.int32(1), "tile.tile_size": [T.int32(8), T.int32(32)]}}):
        for j in T.serial(1, annotations={{"tile.execution_axis": T.int32(1)}}):
            for ki in T.serial(8, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}}):
                for kj in T.vectorized(32, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(1)}}):
                    B_shared[i * 8 + ki, j * 32 + kj] = Tmp_shared[i * 8 + ki, j * 32 + kj] + T.Cast("{dtype}", x1)
"""
    return tvm.script.from_source(source)


def _make_manual_lowered_primfunc(buffer_decls, stmt_snippets):
    body = "\n".join([*buffer_decls, *stmt_snippets])
    source = "# from tvm.script import tir as T\n@T.prim_func\ndef main():\n"
    source += textwrap.indent(body, "    ")
    return tvm.script.from_source(source)


def _lowered_2d_tile_region(dst, expr, *, block_m=32, block_n=32, tile_size=(8, 32)):
    tile_m, tile_n = tile_size
    outer_m = block_m // tile_m
    outer_n = block_n // tile_n
    return textwrap.dedent(
        f"""
        for i in T.serial(
            {outer_m},
            annotations={{
                "tile.domain": [T.int32({block_m}), T.int32({block_n})],
                "tile.execution_axis": T.int32(0),
                "tile.execution_domain_axes": [T.int32(0), T.int32(1)],
                "tile.scope_entry": T.int32(1),
                "tile.tile_size": [T.int32({tile_m}), T.int32({tile_n})],
            }},
        ):
            for j in T.serial({outer_n}, annotations={{"tile.execution_axis": T.int32(1)}}):
                for ki in T.serial({tile_m}, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}}):
                    for kj in T.vectorized({tile_n}, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(1)}}):
                        {dst}[i * {tile_m} + ki, j * {tile_n} + kj] = {expr}
        """
    ).strip()


def _lowered_1d_tile_region(dst, expr, *, length=64, tile_size=64):
    outer = length // tile_size
    return textwrap.dedent(
        f"""
        for i in T.serial(
            {outer},
            annotations={{
                "tile.domain": [T.int32({length})],
                "tile.execution_axis": T.int32(0),
                "tile.execution_domain_axes": [T.int32(0)],
                "tile.scope_entry": T.int32(1),
                "tile.tile_size": [T.int32({tile_size})],
            }},
        ):
            for ki in T.vectorized({tile_size}, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}}):
                {dst}[i * {tile_size} + ki] = {expr}
        """
    ).strip()


def _lowered_row_reduction_region(src, dst, *, block_m=64, block_n=64, tile_size=(4, 32)):
    tile_m, tile_n = tile_size
    outer_m = block_m // tile_m
    outer_n = block_n // tile_n
    return textwrap.dedent(
        f"""
        for i in T.serial(
            {outer_m},
            annotations={{
                "tile.domain": [T.int32({block_m}), T.int32({block_n})],
                "tile.execution_axis": T.int32(0),
                "tile.execution_domain_axes": [T.int32(0), T.int32(1)],
                "tile.scope_entry": T.int32(1),
                "tile.tile_size": [T.int32({tile_m}), T.int32({tile_n})],
            }},
        ):
            for j in T.serial({outer_n}, annotations={{"tile.execution_axis": T.int32(1)}}):
                with T.block("reduce_tile_op"):
                    T.reads()
                    T.writes()
                    dst_buffer_acc = T.alloc_buffer(({tile_m}, {tile_n}), "float32", scope="shared.rsram")
                    if j == 0:
                        for ki in T.serial({tile_m}, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}}):
                            for kj in T.vectorized({tile_n}, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(1)}}):
                                dst_buffer_acc[ki, kj] = T.float32("-inf")
                    for ki in T.serial({tile_m}, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}}):
                        for kj in T.vectorized({tile_n}, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(1)}}):
                            dst_buffer_acc[ki, kj] = T.max(
                                dst_buffer_acc[ki, kj],
                                {src}[i * {tile_m} + ki, j * {tile_n} + kj],
                            )
                    if j == {outer_n - 1}:
                        for ki in T.serial({tile_m}, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}}):
                            {dst}[i * {tile_m} + ki] = dst_buffer_acc[ki, 0]
        """
    ).strip()


def _lowered_row_write_region(src, dst, *, block_m=64, block_n=64, tile_size=(4, 32)):
    tile_m, tile_n = tile_size
    outer_m = block_m // tile_m
    outer_n = block_n // tile_n
    return textwrap.dedent(
        f"""
        for i in T.serial(
            {outer_m},
            annotations={{
                "tile.domain": [T.int32({block_m}), T.int32({block_n})],
                "tile.execution_axis": T.int32(0),
                "tile.execution_domain_axes": [T.int32(0), T.int32(1)],
                "tile.scope_entry": T.int32(1),
                "tile.tile_size": [T.int32({tile_m}), T.int32({tile_n})],
            }},
        ):
            for j in T.serial({outer_n}, annotations={{"tile.execution_axis": T.int32(1)}}):
                for ki in T.serial({tile_m}, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}}):
                    {dst}[i * {tile_m} + ki] = {src}[i * {tile_m} + ki, j * {tile_n}]
        """
    ).strip()


def reduction_consumer_lowered_kernel(block_n=64):
    block_m = 64
    dtype = "float32"
    tile_size = (4, 32)
    return _make_manual_lowered_primfunc(
        [
            f'a_shared = T.alloc_buffer(({block_m}, {block_n}), "{dtype}", scope="shared.rsram")',
            f'values = T.alloc_buffer(({block_m}, {block_n}), "{dtype}", scope="shared.rsram")',
            f'row_max = T.alloc_buffer(({block_m},), "{dtype}", scope="shared.rsram")',
            f'b_shared = T.alloc_buffer(({block_m}, {block_n}), "{dtype}", scope="shared.rsram")',
        ],
        [
            _lowered_2d_tile_region(
                "values",
                "a_shared[i * 4 + ki, j * 32 + kj]",
                block_m=block_m,
                block_n=block_n,
                tile_size=tile_size,
            ),
            _lowered_row_reduction_region("values", "row_max", block_m=block_m, block_n=block_n, tile_size=tile_size),
            _lowered_2d_tile_region(
                "b_shared",
                "values[i * 4 + ki, j * 32 + kj] - row_max[i * 4 + ki]",
                block_m=block_m,
                block_n=block_n,
                tile_size=tile_size,
            ),
        ],
    )


def war_reduction_lowered_kernel():
    block_m = 64
    block_n = 64
    dtype = "float32"
    tile_size = (4, 32)
    return _make_manual_lowered_primfunc(
        [
            f'a_shared = T.alloc_buffer(({block_m}, {block_n}), "{dtype}", scope="shared.rsram")',
            f'tile_values = T.alloc_buffer(({block_m}, {block_n}), "{dtype}", scope="shared.rsram")',
            f'row_values = T.alloc_buffer(({block_m},), "{dtype}", scope="shared.rsram")',
        ],
        [
            _lowered_2d_tile_region(
                "tile_values",
                "a_shared[i * 4 + ki, j * 32 + kj] + row_values[i * 4 + ki]",
                block_m=block_m,
                block_n=block_n,
                tile_size=tile_size,
            ),
            _lowered_row_reduction_region(
                "tile_values",
                "row_values",
                block_m=block_m,
                block_n=block_n,
                tile_size=tile_size,
            ),
        ],
    )


def unsafe_war_lowered_kernel():
    block_m = 64
    block_n = 64
    dtype = "float32"
    tile_size = (4, 32)
    return _make_manual_lowered_primfunc(
        [
            f'a_shared = T.alloc_buffer(({block_m}, {block_n}), "{dtype}", scope="shared.rsram")',
            f'tile_values = T.alloc_buffer(({block_m}, {block_n}), "{dtype}", scope="shared.rsram")',
            f'row_values = T.alloc_buffer(({block_m},), "{dtype}", scope="shared.rsram")',
        ],
        [
            _lowered_2d_tile_region(
                "tile_values",
                "a_shared[i * 4 + ki, j * 32 + kj] + row_values[i * 4 + ki]",
                block_m=block_m,
                block_n=block_n,
                tile_size=tile_size,
            ),
            _lowered_row_write_region(
                "tile_values",
                "row_values",
                block_m=block_m,
                block_n=block_n,
                tile_size=tile_size,
            ),
        ],
    )


def _lowered_predicated_2d_tile_region(dst, src, *, block_m=4, block_n=4, tile_size=(4, 32)):
    tile_m, tile_n = tile_size
    outer_m = (block_m + tile_m - 1) // tile_m
    outer_n = (block_n + tile_n - 1) // tile_n
    row = f"i * {tile_m} + ki"
    col = f"j * {tile_n} + kj"
    predicate = f"{col} < {block_n}"
    return textwrap.dedent(
        f"""
        for i in T.serial(
            {outer_m},
            annotations={{
                "tile.domain": [T.int32({block_m}), T.int32({block_n})],
                "tile.execution_axis": T.int32(0),
                "tile.execution_domain_axes": [T.int32(0), T.int32(1)],
                "tile.scope_entry": T.int32(1),
                "tile.tile_size": [T.int32({tile_m}), T.int32({tile_n})],
            }},
        ):
            for j in T.serial({outer_n}, annotations={{"tile.execution_axis": T.int32(1)}}):
                for ki in T.serial({tile_m}, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}}):
                    for kj in T.vectorized({tile_n}, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(1)}}):
                        {dst}.vstore([{row}, {col}], {src}.vload([{row}, {col}], predicate={predicate}), predicate={predicate})
        """
    ).strip()


def predicated_two_region_lowered_kernel(dtype="float32"):
    return _make_manual_lowered_primfunc(
        [
            f'A_shared = T.alloc_buffer((4, 4), "{dtype}", scope="shared.rsram")',
            f'Tmp_shared = T.alloc_buffer((4, 4), "{dtype}", scope="shared.rsram")',
            f'B_shared = T.alloc_buffer((4, 4), "{dtype}", scope="shared.rsram")',
        ],
        [
            _lowered_predicated_2d_tile_region("Tmp_shared", "A_shared"),
            _lowered_predicated_2d_tile_region("B_shared", "Tmp_shared"),
        ],
    )


def _lowered_row_summary_region(dst, expr, *, block_m=32, block_n=32, tile_size=(8, 32)):
    tile_m, tile_n = tile_size
    outer_m = block_m // tile_m
    outer_n = block_n // tile_n
    return textwrap.dedent(
        f"""
        for i in T.serial(
            {outer_m},
            annotations={{
                "tile.domain": [T.int32({block_m}), T.int32({block_n})],
                "tile.execution_axis": T.int32(0),
                "tile.execution_domain_axes": [T.int32(0), T.int32(1)],
                "tile.scope_entry": T.int32(1),
                "tile.tile_size": [T.int32({tile_m}), T.int32({tile_n})],
            }},
        ):
            for j in T.serial({outer_n}, annotations={{"tile.execution_axis": T.int32(1)}}):
                {dst}[i] = {expr}
        """
    ).strip()


def independent_two_region_lowered_kernel(dtype="float16"):
    return _make_manual_lowered_primfunc(
        [
            f'A_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'Tmp_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'C_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
        ],
        [
            _lowered_2d_tile_region("Tmp_shared", "A_shared[i * 8 + ki, j * 32 + kj]"),
            _lowered_2d_tile_region("C_shared", "A_shared[i * 8 + ki, j * 32 + kj]"),
        ],
    )


def incompatible_prefix_two_region_lowered_kernel(dtype="float16"):
    return _make_manual_lowered_primfunc(
        [
            f'A_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'Tmp_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'B_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
        ],
        [
            _lowered_2d_tile_region("Tmp_shared", "A_shared[i * 8 + ki, j * 32 + kj]", tile_size=(8, 32)),
            _lowered_2d_tile_region("B_shared", "Tmp_shared[i * 4 + ki, j * 32 + kj]", tile_size=(4, 32)),
        ],
    )


def mixed_rank_two_region_lowered_kernel(dtype="float32"):
    return _make_manual_lowered_primfunc(
        [
            f'A_shared = T.alloc_buffer((64, 128), "{dtype}", scope="shared.rsram")',
            f'B_shared = T.alloc_buffer((64,), "{dtype}", scope="shared.rsram")',
        ],
        [
            _lowered_1d_tile_region("B_shared", f'T.Cast("{dtype}", 2)'),
            _lowered_2d_tile_region(
                "A_shared",
                "A_shared[i * 64 + ki, j * 128 + kj] + B_shared[i * 64 + ki]",
                block_m=64,
                block_n=128,
                tile_size=(64, 128),
            ),
        ],
    )


def interrupted_two_region_lowered_kernel(dtype="float16"):
    return _make_manual_lowered_primfunc(
        [
            f'A_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'Tmp_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'B_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'fence = T.alloc_buffer((1,), "{dtype}", scope="shared.rsram")',
        ],
        [
            _lowered_2d_tile_region("Tmp_shared", "A_shared[i * 8 + ki, j * 32 + kj]"),
            f'fence[0] = T.Cast("{dtype}", 0)',
            _lowered_2d_tile_region("B_shared", "Tmp_shared[i * 8 + ki, j * 32 + kj]"),
        ],
    )


def three_region_chain_lowered_kernel(dtype="float16"):
    return _make_manual_lowered_primfunc(
        [
            f'A_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'Tmp0_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'Tmp1_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'B_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
        ],
        [
            _lowered_2d_tile_region("Tmp0_shared", "A_shared[i * 8 + ki, j * 32 + kj]"),
            _lowered_2d_tile_region("Tmp1_shared", "Tmp0_shared[i * 8 + ki, j * 32 + kj]"),
            _lowered_2d_tile_region("B_shared", "Tmp1_shared[i * 8 + ki, j * 32 + kj]"),
        ],
    )


def war_two_region_lowered_kernel(dtype="float16"):
    return _make_manual_lowered_primfunc(
        [
            f'A_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'Tmp_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'B_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
        ],
        [
            _lowered_2d_tile_region("B_shared", "Tmp_shared[i * 8 + ki, j * 32 + kj]"),
            _lowered_2d_tile_region("Tmp_shared", "A_shared[i * 8 + ki, j * 32 + kj]"),
        ],
    )


def partial_depth_two_region_lowered_kernel(dtype="float16"):
    return _make_manual_lowered_primfunc(
        [
            f'A_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'row_sum = T.alloc_buffer((4,), "{dtype}", scope="shared.rsram")',
            f'B_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
        ],
        [
            _lowered_row_summary_region("row_sum", "A_shared[i * 8, 0]"),
            _lowered_2d_tile_region("B_shared", "A_shared[i * 8 + ki, j * 32 + kj] + row_sum[i]"),
        ],
    )


def two_disjoint_groups_lowered_kernel(dtype="float16"):
    return _make_manual_lowered_primfunc(
        [
            f'A_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'Tmp0_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'B_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'Tmp1_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'C_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'fence = T.alloc_buffer((1,), "{dtype}", scope="shared.rsram")',
        ],
        [
            _lowered_2d_tile_region("Tmp0_shared", "A_shared[i * 8 + ki, j * 32 + kj]"),
            _lowered_2d_tile_region("B_shared", "Tmp0_shared[i * 8 + ki, j * 32 + kj]"),
            f'fence[0] = T.Cast("{dtype}", 0)',
            _lowered_2d_tile_region("Tmp1_shared", "A_shared[i * 8 + ki, j * 32 + kj]"),
            _lowered_2d_tile_region("C_shared", "Tmp1_shared[i * 8 + ki, j * 32 + kj]"),
        ],
    )


def loop_var_let_wrapped_two_region_lowered_kernel(dtype="float16"):
    return _make_manual_lowered_primfunc(
        [
            f'A_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'Tmp_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
            f'B_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")',
        ],
        [
            _lowered_2d_tile_region("Tmp_shared", "A_shared[i * 8 + ki, j * 32 + kj]"),
            textwrap.dedent(
                f"""
                for i in T.serial(
                    4,
                    annotations={{
                        "tile.domain": [T.int32(32), T.int32(32)],
                        "tile.execution_axis": T.int32(0),
                        "tile.execution_domain_axes": [T.int32(0), T.int32(1)],
                        "tile.scope_entry": T.int32(1),
                        "tile.tile_size": [T.int32(8), T.int32(32)],
                    }},
                ):
                    for j in T.serial(1, annotations={{"tile.execution_axis": T.int32(1)}}):
                        x1: T.int32 = i * 8 + j * 32
                        for ki in T.serial(8, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}}):
                            for kj in T.vectorized(32, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(1)}}):
                                B_shared[i * 8 + ki, j * 32 + kj] = Tmp_shared[i * 8 + ki, j * 32 + kj] + T.Cast("{dtype}", x1)
                """
            ).strip(),
        ],
    )


def _get_block_by_name(stmt, name):
    found = None

    def visit(node):
        nonlocal found
        if isinstance(node, tvm.tir.Block) and node.name_hint == name:
            found = node

    tvm.tir.stmt_functor.post_order_visit(stmt, visit)
    assert found is not None, f"Expected block `{name}` in rewritten TIR"
    return found


def _as_seq(stmt):
    if isinstance(stmt, tvm.tir.SeqStmt):
        return list(stmt.seq)
    return [stmt]


def _root_seq(mod):
    root = _get_block_by_name(mod["main"].body, "root")
    return _as_seq(root.body)


def _tilelang_root_seq(mod):
    tilelang_root = _get_block_by_name(mod["main"].body, "tilelang_root")
    return _as_seq(tilelang_root.body)


def _for_annotations(loop):
    return dict(loop.annotations) if loop.annotations else {}


def _is_scope_entry_loop(stmt, *, tile_size=None, extent=None):
    if not isinstance(stmt, tvm.tir.For):
        return False
    annotations = _for_annotations(stmt)
    if "tile.scope_entry" not in annotations:
        return False
    if tile_size is not None and list(annotations.get("tile.tile_size", [])) != list(tile_size):
        return False
    if extent is not None and int(stmt.extent) != extent:  # noqa: SIM103
        return False
    return True


def _find_scope_entry_loops(stmts, *, tile_size=None, extent=None):
    return [stmt for stmt in stmts if _is_scope_entry_loop(stmt, tile_size=tile_size, extent=extent)]


def _expect_single_match(items, predicate, description):
    matches = [item for item in items if predicate(item)]
    assert len(matches) == 1, f"Expected exactly one {description}, but found {len(matches)}"
    return matches[0]


def _expect_loop_body_seq(scope_loop):
    body = scope_loop.body
    if isinstance(body, tvm.tir.For) and _for_annotations(body).get("tile.execution_axis") == 1:
        return _as_seq(body.body)
    return _as_seq(body)


def _unwrap_block_realize(stmt):
    if isinstance(stmt, tvm.tir.BlockRealize):
        return stmt.block
    return stmt


def _is_reduce_block(stmt):
    block = _unwrap_block_realize(stmt)
    return isinstance(block, tvm.tir.Block) and block.name_hint == "reduce_tile_op"


def _expect_reduce_block(stmt):
    block = _unwrap_block_realize(stmt)
    assert isinstance(block, tvm.tir.Block)
    assert block.name_hint == "reduce_tile_op"
    return block


def _collect_buffer_accesses(stmt):
    reads = []
    writes = []

    def visit(node):
        if isinstance(node, tvm.tir.BufferLoad):
            reads.append(node.buffer.name)
        elif isinstance(node, tvm.tir.BufferStore):
            writes.append(node.buffer.name)

    tvm.tir.stmt_functor.post_order_visit(stmt, visit)
    return reads, writes


def _single_write_name(stmt):
    _, writes = _collect_buffer_accesses(stmt)
    write_names = sorted(set(writes))
    assert len(write_names) == 1, f"Expected a single written buffer, but found {write_names}"
    return write_names[0]


def _leaf_write_names(stmts):
    return [_single_write_name(stmt) for stmt in stmts]


def _collect_var_names(node):
    names = set()

    def visit(obj):
        if isinstance(obj, tvm.tir.Var):
            names.add(obj.name)

    tvm.tir.stmt_functor.post_order_visit(node, visit)
    return names


def _semantic_leaf_tag(stmt):
    if isinstance(stmt, tvm.tir.LetStmt):
        return _semantic_leaf_tag(stmt.body)
    if isinstance(stmt, tvm.tir.AttrStmt):
        return _semantic_leaf_tag(stmt.body)
    if isinstance(stmt, tvm.tir.BlockRealize):
        return _semantic_leaf_tag(stmt.block)
    if isinstance(stmt, tvm.tir.Block):
        assert stmt.name_hint == "reduce_tile_op"
        allocs = {buf.name for buf in stmt.alloc_buffers}
        if "scores_sum_acc" in allocs:
            return "reduce_scores_sum"
        if "scores_max_acc" in allocs:
            return "reduce_scores_max"
        if "row_sum_acc" in allocs:
            return "reduce_row_sum"
        if "dst_buffer_acc" in allocs:
            if "dst_buffer_res" in allocs:
                return "reduce_row_sum"
            return "reduce_scores_sum"
        raise AssertionError(f"Unknown reduction block alloc buffers: {sorted(allocs)}")

    reads, writes = _collect_buffer_accesses(stmt)
    write_set = set(writes)
    if write_set == {"Tmp_shared"}:
        return "Tmp_shared"
    if write_set == {"B_shared"}:
        return "B_shared"
    if write_set == {"scores_max_prev"}:
        return "scores_max_prev"
    if write_set == {"scores_max"}:
        return "scores_max"
    if write_set == {"dst_buffer"}:
        if not reads:
            return "scores_max"
        if "scores_max_prev" in reads:
            return "scores_max"
        if "src_buffer" in reads:
            return "reduce_row_sum"
    if write_set == {"scores_scale"}:
        return "scores_scale"
    if write_set == {"acc_s"}:
        return "acc_s"
    if write_set == {"src_buffer"}:
        if "a_shared" in reads:
            return "a_square"
        return "acc_s"
    if write_set == {"acc_s_cast"}:
        return "acc_s_cast"
    if write_set == {"a_square"}:
        return "a_square"
    if write_set == {"acc_o"}:
        return "acc_o"
    if write_set == {"row_sum"}:
        return "row_sum_init"
    if write_set == {"row_scale"}:
        return "row_scale"
    if write_set == {"logsum"}:
        return "logsum"
    if write_set == {"a_out"}:
        return "a_out"
    raise AssertionError(f"Unknown semantic leaf writes={sorted(write_set)} reads={sorted(set(reads))}")


def _semantic_tags(stmts):
    return [_semantic_leaf_tag(stmt) for stmt in stmts]


def test_sunmmio_tile_loop_fusion_is_noop_on_single_lowered_tile_scope():
    mod = IRModule.from_expr(single_tile_scope_kernel().with_attr("global_symbol", "main"))
    mod = apply_tiles_lowering(mod)

    before = mod.script()
    after = tl.transform.SunmmioTileLoopFusion()(mod)

    assert after.script() == before


def test_sunmmio_tile_loop_fusion_rewrites_consecutive_tile_scopes():
    mod = IRModule.from_expr(two_consecutive_tile_scopes_kernel().with_attr("global_symbol", "main"))
    mod = apply_sunmmio_tile_loop_fusion(mod)

    stmts = _tilelang_root_seq(mod)
    fused_loop = _expect_single_match(
        _find_scope_entry_loops(stmts, tile_size=[8, 32], extent=4),
        lambda loop: _semantic_tags(_expect_loop_body_seq(loop)) == ["Tmp_shared", "B_shared"],
        "fused [8, 32] tile shell with Tmp_shared then B_shared semantics",
    )

    assert _semantic_tags(_expect_loop_body_seq(fused_loop)) == ["Tmp_shared", "B_shared"]


def test_sunmmio_tile_loop_fusion_rewrites_buffer_access_predicates():
    mod = IRModule.from_expr(predicated_two_region_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    scope_loops = _find_scope_entry_loops(_root_seq(mod), tile_size=[4, 32], extent=1)
    assert len(scope_loops) == 1, "Expected the two predicated regions to share one fused tile shell"

    predicated_accesses = []

    def collect_predicated_accesses(node):
        if isinstance(node, (tvm.tir.BufferLoad, tvm.tir.BufferStore)) and node.predicate is not None:
            predicated_accesses.append(node)

    tvm.tir.stmt_functor.post_order_visit(scope_loops[0], collect_predicated_accesses)

    assert len(predicated_accesses) == 4
    for access in predicated_accesses:
        tvm.ir.assert_structural_equal(access.predicate, access.indices[1] < 4)


def test_sunmmio_tile_loop_fusion_keeps_independent_readers_in_source_order():
    mod = IRModule.from_expr(independent_two_region_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    root_stmts = _root_seq(mod)
    scope_loops = _find_scope_entry_loops(root_stmts)

    if len(scope_loops) == 1:
        assert _leaf_write_names(_expect_loop_body_seq(scope_loops[0])) == ["Tmp_shared", "C_shared"]
    else:
        assert len(scope_loops) == 2
        assert [_single_write_name(loop) for loop in scope_loops] == ["Tmp_shared", "C_shared"]


def test_sunmmio_tile_loop_fusion_does_not_merge_incompatible_execution_prefixes():
    mod = IRModule.from_expr(incompatible_prefix_two_region_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    root_stmts = _root_seq(mod)
    scope_loops = _find_scope_entry_loops(root_stmts)

    assert len(scope_loops) == 2
    assert [_single_write_name(loop) for loop in scope_loops] == ["Tmp_shared", "B_shared"]
    assert [[int(x) for x in _for_annotations(loop)["tile.tile_size"]] for loop in scope_loops] == [[8, 32], [4, 32]]
    assert [int(loop.extent) for loop in scope_loops] == [4, 8]


def test_sunmmio_tile_loop_fusion_does_not_merge_different_execution_ranks():
    mod = IRModule.from_expr(mixed_rank_two_region_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    scope_loops = _find_scope_entry_loops(_root_seq(mod))

    assert len(scope_loops) == 2
    assert [_single_write_name(loop) for loop in scope_loops] == ["B_shared", "A_shared"]
    assert [[int(x) for x in _for_annotations(loop)["tile.tile_size"]] for loop in scope_loops] == [
        [64],
        [64, 128],
    ]


def test_sunmmio_tile_loop_fusion_does_not_cross_non_tile_statement():
    mod = IRModule.from_expr(interrupted_two_region_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    root_stmts = _root_seq(mod)

    assert len(root_stmts) == 3
    assert _is_scope_entry_loop(root_stmts[0], tile_size=[8, 32], extent=4)
    assert _single_write_name(root_stmts[0]) == "Tmp_shared"
    assert isinstance(root_stmts[1], tvm.tir.BufferStore)
    assert root_stmts[1].buffer.name == "fence"
    assert _is_scope_entry_loop(root_stmts[2], tile_size=[8, 32], extent=4)
    assert _single_write_name(root_stmts[2]) == "B_shared"


def test_sunmmio_tile_loop_fusion_rewrites_three_region_chain():
    mod = IRModule.from_expr(three_region_chain_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    root_stmts = _root_seq(mod)
    fused_loop = _expect_single_match(
        _find_scope_entry_loops(root_stmts, tile_size=[8, 32], extent=4),
        lambda loop: _leaf_write_names(_expect_loop_body_seq(loop)) == ["Tmp0_shared", "Tmp1_shared", "B_shared"],
        "three-region fused tile shell",
    )

    assert _leaf_write_names(_expect_loop_body_seq(fused_loop)) == ["Tmp0_shared", "Tmp1_shared", "B_shared"]


def test_sunmmio_tile_loop_fusion_preserves_war_order():
    mod = IRModule.from_expr(war_two_region_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    root_stmts = _root_seq(mod)
    scope_loops = _find_scope_entry_loops(root_stmts)

    if len(scope_loops) == 1:
        assert _leaf_write_names(_expect_loop_body_seq(scope_loops[0])) == ["B_shared", "Tmp_shared"]
    else:
        assert len(scope_loops) == 2
        assert [_single_write_name(loop) for loop in scope_loops] == ["B_shared", "Tmp_shared"]


def test_sunmmio_tile_loop_fusion_can_share_only_the_outer_loop_prefix():
    mod = IRModule.from_expr(partial_depth_two_region_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    root_stmts = _root_seq(mod)
    fused_loop = _expect_single_match(
        _find_scope_entry_loops(root_stmts, tile_size=[8, 32], extent=4),
        lambda loop: _leaf_write_names(_expect_loop_body_seq(loop)) == ["row_sum", "B_shared"],
        "depth-1 fused outer loop prefix",
    )

    fused_body = _expect_loop_body_seq(fused_loop)
    assert _leaf_write_names(fused_body) == ["row_sum", "B_shared"]
    assert all(isinstance(stmt, tvm.tir.For) for stmt in fused_body)
    assert all(_for_annotations(stmt).get("tile.execution_axis") == 1 for stmt in fused_body)


def test_sunmmio_tile_loop_fusion_waits_for_conditional_raw_reduction_write():
    mod = IRModule.from_expr(reduction_consumer_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    outer_scope = _expect_single_match(
        _find_scope_entry_loops(_root_seq(mod)),
        lambda loop: "b_shared" in _collect_buffer_accesses(loop)[1],
        "outer row shell containing the reduction consumer",
    )
    inner_scopes = [
        stmt
        for stmt in _as_seq(outer_scope.body)
        if isinstance(stmt, tvm.tir.For) and _for_annotations(stmt).get("tile.execution_axis") == 1
    ]

    assert len(inner_scopes) == 2
    assert "b_shared" not in _collect_buffer_accesses(inner_scopes[0])[1]
    assert "b_shared" in _collect_buffer_accesses(inner_scopes[1])[1]


def test_sunmmio_tile_loop_fusion_keeps_conditional_war_reduction_write_fused():
    mod = IRModule.from_expr(war_reduction_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    scope_loops = _find_scope_entry_loops(_root_seq(mod))
    assert len(scope_loops) == 1

    inner_scopes = [
        stmt
        for stmt in _as_seq(scope_loops[0].body)
        if isinstance(stmt, tvm.tir.For) and _for_annotations(stmt).get("tile.execution_axis") == 1
    ]
    assert len(inner_scopes) == 1

    fused_body = _as_seq(inner_scopes[0].body)
    assert len(fused_body) == 2
    assert _single_write_name(fused_body[0]) == "tile_values"
    reduction_block = _expect_reduce_block(fused_body[1])

    row_write_conditions = []

    def collect_row_write_condition(node):
        if not isinstance(node, tvm.tir.IfThenElse):
            return
        if "row_values" in _collect_buffer_accesses(node.then_case)[1]:
            row_write_conditions.append(node.condition)

    tvm.tir.stmt_functor.post_order_visit(reduction_block.body, collect_row_write_condition)
    assert len(row_write_conditions) == 1
    assert tvm.ir.structural_equal(
        row_write_conditions[0],
        inner_scopes[0].loop_var == inner_scopes[0].extent - 1,
    )

    reads, writes = _collect_buffer_accesses(inner_scopes[0])
    assert "row_values" in reads
    assert "row_values" in writes
    assert "tile_values" in reads
    assert "tile_values" in writes


def test_sunmmio_tile_loop_fusion_splits_loop_carried_war_write():
    mod = IRModule.from_expr(unsafe_war_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    scope_loops = _find_scope_entry_loops(_root_seq(mod))
    assert len(scope_loops) == 2
    assert [_single_write_name(loop) for loop in scope_loops] == [
        "tile_values",
        "row_values",
    ]


def test_sunmmio_tile_loop_fusion_keeps_unit_trip_reduction_consumer_fused():
    mod = IRModule.from_expr(reduction_consumer_lowered_kernel(block_n=32).with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    outer_scope = _expect_single_match(
        _find_scope_entry_loops(_root_seq(mod)),
        lambda loop: "b_shared" in _collect_buffer_accesses(loop)[1],
        "outer row shell containing the unit-trip reduction consumer",
    )
    inner_scopes = [
        stmt
        for stmt in _as_seq(outer_scope.body)
        if isinstance(stmt, tvm.tir.For) and _for_annotations(stmt).get("tile.execution_axis") == 1
    ]

    assert len(inner_scopes) == 1
    assert "b_shared" in _collect_buffer_accesses(inner_scopes[0])[1]


def test_sunmmio_tile_loop_fusion_rewrites_multiple_planning_groups_independently():
    mod = IRModule.from_expr(two_disjoint_groups_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    root_stmts = _root_seq(mod)

    assert len(root_stmts) == 3
    assert _is_scope_entry_loop(root_stmts[0], tile_size=[8, 32], extent=4)
    assert _leaf_write_names(_expect_loop_body_seq(root_stmts[0])) == ["Tmp0_shared", "B_shared"]
    assert isinstance(root_stmts[1], tvm.tir.BufferStore)
    assert root_stmts[1].buffer.name == "fence"
    assert _is_scope_entry_loop(root_stmts[2], tile_size=[8, 32], extent=4)
    assert _leaf_write_names(_expect_loop_body_seq(root_stmts[2])) == ["Tmp1_shared", "C_shared"]


def test_sunmmio_tile_loop_fusion_rewrite_preserves_loop_var_dependent_local_let():
    mod = IRModule.from_expr(loop_var_let_wrapped_two_region_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    root_stmts = _root_seq(mod)
    fused_loop = _expect_single_match(
        _find_scope_entry_loops(root_stmts, tile_size=[8, 32], extent=4),
        lambda loop: _semantic_tags(_expect_loop_body_seq(loop)) == ["Tmp_shared", "B_shared"],
        "fused shell with loop-var-dependent local LetStmt leaf",
    )

    fused_body = _expect_loop_body_seq(fused_loop)
    assert _semantic_leaf_tag(fused_body[0]) == "Tmp_shared"
    assert isinstance(fused_body[1], tvm.tir.LetStmt)
    assert fused_body[1].var.name == "x1"
    assert _semantic_leaf_tag(fused_body[1]) == "B_shared"
    assert {"i", "j"}.issubset(_collect_var_names(fused_body[1].value))


def test_sunmmio_tile_loop_fusion_rewrites_flash_attention_window():
    mod = IRModule.from_expr(flash_attention_online_softmax_tiled_kernel().with_attr("global_symbol", "main"))
    mod = apply_sunmmio_tile_loop_fusion(mod)

    stmts = _tilelang_root_seq(mod)
    fused_row = _expect_single_match(
        _find_scope_entry_loops(stmts, tile_size=[32], extent=1),
        lambda loop: _semantic_tags(_expect_loop_body_seq(loop)) == ["scores_max", "scores_scale"],
        "fused flash row shell realizing scores_max then scores_scale",
    )
    fused_tile = _expect_single_match(
        _find_scope_entry_loops(stmts, tile_size=[4, 32], extent=8),
        lambda loop: _semantic_tags(_expect_loop_body_seq(loop)) == ["acc_s", "acc_s_cast", "reduce_scores_sum"],
        "fused flash tile shell realizing acc_s, acc_s_cast, then reduce_scores_sum",
    )

    assert _semantic_tags(_expect_loop_body_seq(fused_row)) == ["scores_max", "scores_scale"]
    assert _semantic_tags(_expect_loop_body_seq(fused_tile)) == ["acc_s", "acc_s_cast", "reduce_scores_sum"]


def test_sunmmio_tile_loop_fusion_rewrite_preserves_flash_reduction_local_scratch():
    mod = IRModule.from_expr(flash_attention_online_softmax_tiled_kernel().with_attr("global_symbol", "main"))
    mod = apply_sunmmio_tile_loop_fusion(mod)

    stmts = _tilelang_root_seq(mod)

    top_level_reduce_block = _expect_single_match(
        stmts,
        _is_reduce_block,
        "top-level flash reduce_max block",
    )
    assert isinstance(top_level_reduce_block, tvm.tir.BlockRealize)
    top_level_reduce_block = _expect_reduce_block(top_level_reduce_block)

    fused_tile = _expect_single_match(
        _find_scope_entry_loops(stmts, tile_size=[4, 32], extent=8),
        lambda loop: _semantic_tags(_expect_loop_body_seq(loop)) == ["acc_s", "acc_s_cast", "reduce_scores_sum"],
        "flash fused tile shell with local reduce_scores_sum block",
    )
    assert isinstance(_expect_loop_body_seq(fused_tile)[2], tvm.tir.BlockRealize)
    fused_reduce = _expect_reduce_block(_expect_loop_body_seq(fused_tile)[2])
    fused_reduce_body = _as_seq(fused_reduce.body)

    assert isinstance(fused_reduce_body[0], tvm.tir.IfThenElse)
    assert isinstance(fused_reduce_body[1], tvm.tir.For)
    assert isinstance(fused_reduce_body[2], tvm.tir.IfThenElse)

    top_level_allocs = [buf.name for buf in top_level_reduce_block.alloc_buffers]
    fused_allocs = [buf.name for buf in fused_reduce.alloc_buffers]

    assert top_level_allocs.count("dst_buffer_acc") == 1
    assert top_level_allocs.count("dst_buffer_res") == 1
    assert fused_allocs.count("dst_buffer_acc") == 1


def test_sunmmio_tile_loop_fusion_rewrite_hoists_common_attr_wrapper():
    mod = IRModule.from_expr(attr_wrapped_two_region_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    root_stmts = _root_seq(mod)

    attr_stmt = _expect_single_match(
        root_stmts,
        lambda stmt: isinstance(stmt, tvm.tir.AttrStmt),
        "hoisted AttrStmt wrapper",
    )
    assert int(attr_stmt.value) == 1
    fused_loop = _expect_single_match(
        _as_seq(attr_stmt.body),
        lambda stmt: _is_scope_entry_loop(stmt, tile_size=[8, 32], extent=4),
        "fused child under hoisted AttrStmt",
    )
    assert _semantic_tags(_expect_loop_body_seq(fused_loop)) == ["Tmp_shared", "B_shared"]


def test_sunmmio_tile_loop_fusion_rewrite_preserves_local_let_wrappers():
    mod = IRModule.from_expr(let_wrapped_two_region_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    root_stmts = _root_seq(mod)

    let_stmt = _expect_single_match(
        root_stmts,
        lambda stmt: isinstance(stmt, tvm.tir.LetStmt),
        "outer local LetStmt wrapper",
    )
    assert let_stmt.var.name == "x0"
    assert int(let_stmt.value) == 7
    fused_loop = let_stmt.body
    assert _is_scope_entry_loop(fused_loop, tile_size=[8, 32], extent=4)
    fused_body = _expect_loop_body_seq(fused_loop)
    assert _semantic_leaf_tag(fused_body[0]) == "Tmp_shared"
    assert isinstance(fused_body[1], tvm.tir.LetStmt)
    assert fused_body[1].var.name == "x1"
    assert int(fused_body[1].value) == 11
    assert _semantic_leaf_tag(fused_body[1]) == "B_shared"


def test_sunmmio_tile_loop_fusion_rewrite_preserves_local_let_in_mixed_cluster():
    mod = IRModule.from_expr(mixed_plain_and_let_wrapped_two_region_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    root_stmts = _root_seq(mod)

    fused_loop = _expect_single_match(
        root_stmts,
        lambda stmt: _is_scope_entry_loop(stmt, tile_size=[8, 32], extent=4),
        "mixed fused shell",
    )
    fused_body = _expect_loop_body_seq(fused_loop)
    assert _semantic_leaf_tag(fused_body[0]) == "Tmp_shared"
    assert isinstance(fused_body[1], tvm.tir.LetStmt)
    assert fused_body[1].var.name == "x1"
    assert int(fused_body[1].value) == 7
    assert _semantic_leaf_tag(fused_body[1]) == "B_shared"


def test_sunmmio_tile_loop_fusion_rewrites_rmsnorm_window_structurally():
    mod = IRModule.from_expr(rms_norm_tiled_kernel().with_attr("global_symbol", "main"))
    mod = apply_sunmmio_tile_loop_fusion(mod)

    stmts = _tilelang_root_seq(mod)

    fused_tile = _expect_single_match(
        _find_scope_entry_loops(stmts, tile_size=[4, 32], extent=8),
        lambda loop: _semantic_tags(_expect_loop_body_seq(loop)) == ["a_square", "reduce_row_sum"],
        "fused RMSNorm tile shell realizing a_square then reduce_row_sum",
    )
    fused_tile_body = _expect_loop_body_seq(fused_tile)
    assert _semantic_tags(fused_tile_body) == ["a_square", "reduce_row_sum"]
    reduce_block = _expect_reduce_block(fused_tile_body[1])
    reduce_block_body = _as_seq(reduce_block.body)
    assert len(reduce_block_body) == 3
    assert isinstance(reduce_block_body[0], tvm.tir.IfThenElse)
    assert isinstance(reduce_block_body[1], tvm.tir.For)
    assert isinstance(reduce_block_body[2], tvm.tir.IfThenElse)
