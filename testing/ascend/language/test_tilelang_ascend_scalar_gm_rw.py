"""Vector-core work + junk scalar GM read/write in the same kernel (.asc check).

Exercises the core-mask assignment path for scalar tasks that read
straight from GM and store straight back to GM, running alongside real
vector-core (MTE / SimdVF) work. Modeled on the ``_scatter_verified``
scatter kernel: a persistent loop copies a row through UB (vector core)
while a handful of scalar loads/stores gate and scatter values directly
in GM. Auto-schedule must pin those scalar tasks to a concrete core;
this test asserts on the generated ``.asc`` rather than running on NPU.
"""

import re
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.language import simd as S
from tilelang.engine.lower import lower

NUM_CORES = 32
ROW = 64  # elements per row copied through UB (vector-core work)
VL = 64  # float32 lanes per 2048-bit vector register


def make_kernel(n_rows: int):
    num_vregs = ROW // VL

    @T.prim_func
    def main(
        gate: T.Tensor[(n_rows,), T.bool],  # scalar GM read (junk gate)
        dest_idx: T.Tensor[(n_rows,), T.int32],  # scalar GM read (junk scatter dest)
        src_val: T.Tensor[(n_rows,), T.float32],  # scalar GM read
        rows: T.Tensor[(n_rows, ROW), T.float32],  # vector-core MTE source
        scattered: T.Tensor[(n_rows,), T.float32],  # scalar GM write (scatter)
        out_rows: T.Tensor[(n_rows, ROW), T.float32],  # vector-core MTE dest
    ):
        with T.Kernel(NUM_CORES) as core_id:
            row_ub = T.alloc_shared((ROW,), T.float32)

            for w in T.Persistent([n_rows], NUM_CORES, core_id, num_stages=1):
                # VECTOR CORE: copy a row GM->UB, +1.0 via SimdVF, UB->GM.
                T.copy(rows[w, :], row_ub)
                with T.SimdVF():
                    full = S.pset(32, "PAT_ALL")
                    one = S.vdup(T.float32(1), "float32", full)
                    for r in T.serial(num_vregs):
                        x = S.vld(row_ub[r * VL])
                        S.vsts(row_ub[r * VL], S.vadd(x, one, full), full)
                T.copy(row_ub, out_rows[w, :])

                # JUNK SCALAR: read GM directly, gate, scatter straight to GM.
                g = T.alloc_var(T.bool, gate[w])
                if g:
                    dest = T.alloc_var(T.int32, dest_idx[w])
                    if 0 <= dest < n_rows:
                        scattered[dest] = src_val[w] + T.float32(1)

    return main


def test_scalar_gm_rw_codegen():
    artifact = lower(make_kernel(256), target="ascend")
    source = artifact.kernel_source
    print(source)

    # Vector-core work must survive: MTE copies both directions + a SimdVF helper.
    assert "copy_gm_to_ubuf" in source, "missing GM->UB vector-core copy"
    assert "copy_ubuf_to_gm" in source, "missing UB->GM vector-core copy"
    assert re.search(r"simd_vf_\d+\s*\(", source), "missing SimdVF helper call"

    # Regression: the junk scalar tasks used to be tagged broadcast, which made
    # auto-schedule treat them as a separate (Cube/AIC) core and emit spurious
    # cross-core AIC<->AIV handshakes. There is no matmul here, so the kernel
    # must stay a single pure-AIV kernel with NO Cube side and NO cross-core
    # sync at all.
    assert "__global__ __vector__" in source, "expected a single pure-AIV kernel"
    assert "__mix__" not in source, "no Cube/AIC side should be generated for a scalar-only + vector kernel"
    assert "get_subblockid" not in source, "no AIC/AIV sub-block split should appear"
    assert "CrossCore" not in source, "broadcast scalar tasks must not trigger phantom cross-core AIC<->AIV sync"


if __name__ == "__main__":
    tilelang.testing.main()
