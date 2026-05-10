import re

from tilelang import tvm as tvm
import tilelang
import tilelang.testing
from tvm.script import tir as T


def _run_merge_pass(func: tvm.tir.PrimFunc, config: dict | None = None, aggressive: bool = False):
    mod = tvm.IRModule({"main": func})
    with tvm.transform.PassContext(config=config or {}):
        mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=aggressive)(mod)
    return mod["main"]


def _load_prim_func_from_source(source: str, file_name: str, func_name: str = "before"):
    from pathlib import Path

    Path(file_name).write_text(source)
    ns = {}
    exec(compile(source, file_name, "exec"), ns)
    return ns[func_name]


def _buffer_plus_offsets_in_script(script: str, buffer_name: str) -> set[int]:
    pattern = rf"{re.escape(buffer_name)}\[[^\]\n]*\+\s*(\d+)\]"
    return {int(match.group(1)) for match in re.finditer(pattern, script)}


def _buffer_constant_offsets_in_script(script: str, buffer_name: str) -> set[int]:
    pattern = rf"{re.escape(buffer_name)}\[(\d+)\]"
    return {int(match.group(1)) for match in re.finditer(pattern, script)}


def _tvm_access_ptr_offsets(script: str) -> list[int]:
    pattern = r'T\.tvm_access_ptr\(T\.type_annotation\("[^"]+"\), [^,]+, (\d+), \d+, [123]\)'
    return [int(match.group(1)) for match in re.finditer(pattern, script)]


def _access_ptr_offsets(script: str, buffer_name: str) -> list[int]:
    pattern = rf"T\.access_ptr\({re.escape(buffer_name)}\[(\d+)\], \d+, [12]\)"
    return [int(match.group(1)) for match in re.finditer(pattern, script)]


@tilelang.testing.requires_cuda
def test_merge_dynamic_shared_reuses_non_overlapping_buffers():
    @T.prim_func(private=True)
    def before(A: T.Buffer((128,), "float16")):
        T.launch_thread("blockIdx.x", 1)
        X = T.allocate([128], "float16", "shared.dyn")
        Y = T.allocate([128], "float16", "shared.dyn")
        Z = T.allocate([128], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 128)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Xb = T.Buffer((128,), "float16", data=X, scope="shared.dyn")
        Yb = T.Buffer((128,), "float16", data=Y, scope="shared.dyn")
        Zb = T.Buffer((128,), "float16", data=Z, scope="shared.dyn")
        Xb[tx] = A[tx]
        Yb[tx] = Xb[tx]
        A[tx] = Yb[tx]
        Zb[tx] = A[tx]
        A[tx] = Zb[tx]

    @T.prim_func(private=True)
    def expected(A: T.Buffer((128,), "float16")):
        T.launch_thread("blockIdx.x", 1)
        buf_dyn_shmem = T.allocate([512], "uint8", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 128)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Xb = T.Buffer((128,), "float16", data=buf_dyn_shmem, scope="shared.dyn")
        Xb[tx] = A[tx]
        Yb = T.Buffer((128,), "float16", data=buf_dyn_shmem, scope="shared.dyn")
        Yb[tx + 128] = Xb[tx]
        A[tx] = Yb[tx + 128]
        Zb = T.Buffer((128,), "float16", data=buf_dyn_shmem, scope="shared.dyn")
        Zb[tx] = A[tx]
        A[tx] = Zb[tx]

    after = _run_merge_pass(before)
    tvm.ir.assert_structural_equal(after, expected)


@tilelang.testing.requires_cuda
def test_merge_static_shared_requires_flag_and_aggressive_tightens_reuse():
    @T.prim_func(private=True)
    def before(A: T.Buffer((128,), "float16")):
        T.launch_thread("blockIdx.x", 1)
        X = T.allocate([128], "float16", "shared")
        Y = T.allocate([128], "float16", "shared")
        tx = T.launch_thread("threadIdx.x", 128)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        if tx < 64:
            Xb = T.Buffer((128,), "float16", data=X, scope="shared")
            Xb[tx] = A[tx]
            A[tx] = Xb[tx]
        else:
            Yb = T.Buffer((128,), "float16", data=Y, scope="shared")
            Yb[tx - 64] = A[tx - 64]
            A[tx - 64] = Yb[tx - 64]

    no_merge = _run_merge_pass(before)
    merged = _run_merge_pass(before, config={tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True})
    merged_aggr = _run_merge_pass(
        before,
        config={tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True},
        aggressive=True,
    )

    no_merge_s = no_merge.script()
    merged_s = merged.script()
    merged_aggr_s = merged_aggr.script()

    assert 'T.allocate([128], "float16", "shared")' in no_merge_s
    assert 'T.allocate([256], "uint8", "shared")' in merged_s
    assert "tx - 64 + 128" not in merged_s
    assert 'T.allocate([256], "uint8", "shared")' in merged_aggr_s
    assert "tx - 64 + 128" not in merged_aggr_s


@tilelang.testing.requires_cuda
def test_merge_dynamic_shared_rewrites_cp_async_case_after_flatten():
    src = """
import tilelang.language as T

@T.prim_func
def before(A: T.Tensor((16,), T.uint8), B: T.Tensor((16,), T.uint8)):
    bx = T.launch_thread("blockIdx.x", 1)
    S0 = T.allocate([16], "uint8", "shared.dyn")
    S1 = T.allocate([16], "uint8", "shared.dyn")
    tx = T.launch_thread("threadIdx.x", 32)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    S0b = T.Buffer((16,), "uint8", data=S0, scope="shared.dyn")
    T.ptx_cp_async(T.access_ptr(S0b[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
    T.ptx_commit_group()
    T.ptx_wait_group(0)
    S1b = T.Buffer((16,), "uint8", data=S1, scope="shared.dyn")
    S1b[tx // 2] = S0b[tx // 2]
    B[tx // 2] = S1b[tx // 2]
"""

    func = _load_prim_func_from_source(src, "/tmp/merge_cp_async_case.py").with_attr("global_symbol", "main")
    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    after = tilelang.transform.MergeSharedMemoryAllocations()(mod)["main"]
    after_s = after.script()

    assert 'buf_dyn_shmem = T.allocate([32], "uint8", "shared.dyn")' in after_s
    assert "T.ptx_cp_async(T.access_ptr(S0b[0], 16, 2), T.access_ptr(A_1[0], 16, 1), 16)" in after_s
    assert "S1b[v_1 // 2 + 16] = S0b[v_1 // 2]" in after_s
    assert "B_1[v_1 // 2] = S1b[v_1 // 2 + 16]" in after_s


@tilelang.testing.requires_cuda
def test_merge_dynamic_shared_lowbit_style_scratch_and_long_buffer_do_not_reuse_yet():
    @T.prim_func(private=True)
    def before(A: T.Buffer((128,), "float16")):
        T.launch_thread("blockIdx.x", 1)
        Scratch = T.allocate([16384], "float16", "shared.dyn")
        Long = T.allocate([8192], "float16", "shared.dyn")
        Meta = T.allocate([4096], "uint8", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 128)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        ScratchB = T.Buffer((16384,), "float16", data=Scratch, scope="shared.dyn")
        LongB = T.Buffer((8192,), "float16", data=Long, scope="shared.dyn")
        MetaB = T.Buffer((4096,), "uint8", data=Meta, scope="shared.dyn")
        ScratchB[tx] = A[tx]
        LongB[tx + 16384] = ScratchB[tx]
        if tx < 32:
            MetaB[tx] = T.uint8(0)
        A[tx] = LongB[tx + 16384]

    baseline = _run_merge_pass(before)
    aggressive = _run_merge_pass(before, aggressive=True)

    baseline_s = baseline.script()
    aggressive_s = aggressive.script()

    assert 'T.allocate([49152], "uint8", "shared.dyn")' in baseline_s
    assert "LongB[tx + 16384 + 16384] = ScratchB[tx]" in baseline_s
    assert "MetaB[tx] = T.uint8(0)" in baseline_s
    assert 'T.allocate([49152], "uint8", "shared.dyn")' in aggressive_s
    assert "LongB[tx + 16384 + 16384] = ScratchB[tx]" in aggressive_s


@tilelang.testing.requires_cuda
def test_branch_exclusive_dynamic_buffers_only_shrink_under_aggressive_merge():
    @T.prim_func(private=True)
    def before(A: T.Buffer((128,), "float16")):
        T.launch_thread("blockIdx.x", 1)
        X = T.allocate([16384], "float16", "shared.dyn")
        Y = T.allocate([16384], "float16", "shared.dyn")
        Z = T.allocate([16384], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 128)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Xb = T.Buffer((16384,), "float16", data=X, scope="shared.dyn")
        Yb = T.Buffer((16384,), "float16", data=Y, scope="shared.dyn")
        Zb = T.Buffer((16384,), "float16", data=Z, scope="shared.dyn")
        if tx < 64:
            Xb[tx] = A[tx]
            A[tx] = Xb[tx]
        else:
            Yb[tx] = A[tx]
            A[tx] = Yb[tx]
        Zb[tx] = A[tx]
        A[tx] = Zb[tx]

    baseline = _run_merge_pass(before)
    aggressive = _run_merge_pass(before, aggressive=True)

    baseline_s = baseline.script()
    aggressive_s = aggressive.script()

    assert 'T.allocate([32768], "uint8", "shared.dyn")' in baseline_s
    assert 'T.allocate([32768], "uint8", "shared.dyn")' in aggressive_s
    assert "Yb[tx] = A[tx]" in baseline_s
    assert "Yb[tx] = A[tx]" in aggressive_s


@tilelang.testing.requires_cuda
def test_phase_boundary_sync_allows_dynamic_buffer_reuse():
    @T.prim_func(private=True)
    def before(
        A: T.Buffer((4,), "float16"),
        B: T.Buffer((4,), "float16"),
        C: T.Buffer((4,), "float16"),
    ):
        T.launch_thread("blockIdx.x", 1)
        X = T.allocate([16], "float16", "shared.dyn")
        Y = T.allocate([16], "float16", "shared.dyn")
        T.launch_thread("threadIdx.x", 1)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Xb = T.Buffer((16,), "float16", data=X, scope="shared.dyn")
        Yb = T.Buffer((16,), "float16", data=Y, scope="shared.dyn")
        Xb[0] = A[0]
        B[0] = Xb[0]
        T.evaluate(T.ptx_commit_group())
        T.evaluate(T.ptx_wait_group(0))
        T.tvm_storage_sync("shared.dyn")
        Yb[0] = A[1]
        C[0] = Yb[0]

    after = _run_merge_pass(before)
    after_s = after.script()

    assert 'T.allocate([32], "uint8", "shared.dyn")' in after_s
    assert "T.ptx_commit_group()" in after_s
    assert "T.ptx_wait_group(0)" in after_s
    assert 'T.tvm_storage_sync("shared.dyn")' in after_s
    assert "Yb[0] = A[1]" in after_s


@tilelang.testing.requires_cuda
def test_lowbit_like_staged_kv_phases_share_single_dynamic_arena():
    @T.prim_func(private=True)
    def before(
        A: T.Buffer((8,), "float16"),
        B: T.Buffer((8,), "float16"),
        C: T.Buffer((8,), "float16"),
        D: T.Buffer((8,), "float16"),
    ):
        T.launch_thread("blockIdx.x", 1)
        KMeta = T.allocate([32], "uint8", "shared.dyn")
        KData = T.allocate([16], "float16", "shared.dyn")
        VMeta = T.allocate([32], "uint8", "shared.dyn")
        VData = T.allocate([16], "float16", "shared.dyn")
        T.launch_thread("threadIdx.x", 1)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        KMetaB = T.Buffer((32,), "uint8", data=KMeta, scope="shared.dyn")
        KDataB = T.Buffer((16,), "float16", data=KData, scope="shared.dyn")
        VMetaB = T.Buffer((32,), "uint8", data=VMeta, scope="shared.dyn")
        VDataB = T.Buffer((16,), "float16", data=VData, scope="shared.dyn")
        KMetaB[0] = T.uint8(1)
        T.evaluate(T.ptx_commit_group())
        T.evaluate(T.ptx_wait_group(0))
        T.tvm_storage_sync("shared.dyn")
        KDataB[0] = A[0]
        B[0] = KDataB[0]
        T.evaluate(T.ptx_commit_group())
        T.evaluate(T.ptx_wait_group(0))
        T.tvm_storage_sync("shared.dyn")
        VMetaB[0] = T.uint8(2)
        T.evaluate(T.ptx_commit_group())
        T.evaluate(T.ptx_wait_group(0))
        T.tvm_storage_sync("shared.dyn")
        VDataB[0] = C[0]
        D[0] = VDataB[0]

    after = _run_merge_pass(before)
    after_s = after.script()

    assert 'T.allocate([32], "uint8", "shared.dyn")' in after_s
    assert "KMetaB[0] = T.uint8(1)" in after_s
    assert "KDataB[0] = A[0]" in after_s
    assert "VMetaB[0] = T.uint8(2)" in after_s
    assert "VDataB[0] = C[0]" in after_s
    assert after_s.count("T.ptx_wait_group(0)") == 3
    assert after_s.count('T.tvm_storage_sync("shared.dyn")') == 3


@tilelang.testing.requires_cuda
def test_repeated_phase_with_explicit_sync_is_not_yet_split_for_reuse():
    @T.prim_func(private=True)
    def before(
        A: T.Buffer((8,), "float16"),
        B: T.Buffer((8,), "float16"),
        C: T.Buffer((8,), "float16"),
        D: T.Buffer((8,), "float16"),
    ):
        T.launch_thread("blockIdx.x", 1)
        X = T.allocate([16], "float16", "shared.dyn")
        Y = T.allocate([16], "float16", "shared.dyn")
        T.launch_thread("threadIdx.x", 1)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Xb = T.Buffer((16,), "float16", data=X, scope="shared.dyn")
        Yb = T.Buffer((16,), "float16", data=Y, scope="shared.dyn")
        Xb[0] = A[0]
        B[0] = Xb[0]
        T.evaluate(T.ptx_commit_group())
        T.evaluate(T.ptx_wait_group(0))
        T.tvm_storage_sync("shared.dyn")
        Yb[0] = A[1]
        C[0] = Yb[0]
        T.evaluate(T.ptx_commit_group())
        T.evaluate(T.ptx_wait_group(0))
        T.tvm_storage_sync("shared.dyn")
        Xb[0] = A[2]
        D[0] = Xb[0]

    after = _run_merge_pass(before)
    after_s = after.script()

    assert 'T.allocate([64], "uint8", "shared.dyn")' in after_s
    assert "Yb[16] = A[1]" in after_s
    assert "C[0] = Yb[16]" in after_s
    assert after_s.count("T.ptx_wait_group(0)") == 2
    assert after_s.count('T.tvm_storage_sync("shared.dyn")') == 2


@tilelang.testing.requires_cuda
def test_repeated_phased_dynamic_buffers_can_eventually_share_one_slot():
    @T.prim_func(private=True)
    def before(
        A: T.Buffer((4,), "float16"),
        B: T.Buffer((4,), "float16"),
        C: T.Buffer((4,), "float16"),
        D: T.Buffer((4,), "float16"),
    ):
        T.launch_thread("blockIdx.x", 1)
        X = T.allocate([16], "float16", "shared.dyn")
        Y = T.allocate([16], "float16", "shared.dyn")
        T.launch_thread("threadIdx.x", 1)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Xb = T.Buffer((16,), "float16", data=X, scope="shared.dyn")
        Yb = T.Buffer((16,), "float16", data=Y, scope="shared.dyn")
        for i in T.serial(0, 4):
            Xb[0] = A[i]
            B[i] = Xb[0]
            Yb[0] = A[i]
            C[i] = Yb[0]
        for i in T.serial(0, 4):
            Xb[0] = A[i]
            B[i] = Xb[0]
            Yb[0] = A[i]
            D[i] = Yb[0]

    aggressive = _run_merge_pass(before, aggressive=True)
    aggressive_s = aggressive.script()

    assert 'T.allocate([64], "uint8", "shared.dyn")' in aggressive_s
    assert "C[i] = Yb[16]" in aggressive_s or "C[i] = Yb[0]" in aggressive_s
    assert "D[i] = Yb[16]" in aggressive_s or "D[i] = Yb[0]" in aggressive_s


@tilelang.testing.requires_cuda
def test_branch_alternative_write_and_zero_fill_keep_single_dynamic_offset():
    @T.prim_func(private=True)
    def before(
        A: T.Buffer((8,), "float16"),
        B: T.Buffer((8,), "float16"),
        C: T.Buffer((8,), "float16"),
        D: T.Buffer((8,), "float16"),
        cond: T.int32,
    ):
        T.launch_thread("blockIdx.x", 1)
        Base = T.allocate([32], "float16", "shared.dyn")
        Norm = T.allocate([8], "float16", "shared.dyn")
        Alt = T.allocate([8], "float16", "shared.dyn")
        T.launch_thread("threadIdx.x", 1)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        BaseB = T.Buffer((32,), "float16", data=Base, scope="shared.dyn")
        NormB = T.Buffer((8,), "float16", data=Norm, scope="shared.dyn")
        AltB = T.Buffer((8,), "float16", data=Alt, scope="shared.dyn")
        BaseB[0] = A[0]
        if cond > 0:
            for i in range(8):
                NormB[i] = A[i]
        else:
            for i in range(8):
                NormB[i] = T.float16(0)
        for i in range(8):
            AltB[i] = C[i]
        for i in range(8):
            B[i] = NormB[i]
            D[i] = AltB[i]
        C[0] = BaseB[0]

    after = _run_merge_pass(before, aggressive=True)
    after_s = after.script()

    norm_offsets = _buffer_plus_offsets_in_script(after_s, "NormB")
    alt_offsets = _buffer_plus_offsets_in_script(after_s, "AltB")

    assert 'T.allocate([96], "uint8", "shared.dyn")' in after_s
    assert len(norm_offsets) == 1
    assert len(alt_offsets) == 1
    assert norm_offsets != alt_offsets


@tilelang.testing.requires_cuda
def test_nested_branch_alternatives_keep_single_offset_for_postdominating_read():
    @T.prim_func(private=True)
    def before(
        A: T.Buffer((8,), "float16"),
        B: T.Buffer((8,), "float16"),
        C: T.Buffer((8,), "float16"),
        D: T.Buffer((8,), "float16"),
        E: T.Buffer((8,), "float16"),
        outer_cond: T.int32,
        inner_cond: T.int32,
    ):
        T.launch_thread("blockIdx.x", 1)
        Base = T.allocate([32], "float16", "shared.dyn")
        Norm = T.allocate([8], "float16", "shared.dyn")
        Alt = T.allocate([8], "float16", "shared.dyn")
        T.launch_thread("threadIdx.x", 1)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        BaseB = T.Buffer((32,), "float16", data=Base, scope="shared.dyn")
        NormB = T.Buffer((8,), "float16", data=Norm, scope="shared.dyn")
        AltB = T.Buffer((8,), "float16", data=Alt, scope="shared.dyn")
        BaseB[0] = A[0]
        if outer_cond > 0:
            if inner_cond > 0:
                for i in range(8):
                    NormB[i] = A[i]
            else:
                for i in range(8):
                    NormB[i] = B[i]
        else:
            for i in range(8):
                NormB[i] = T.float16(0)
        for i in range(8):
            AltB[i] = C[i]
        for i in range(8):
            D[i] = NormB[i]
            E[i] = AltB[i]
        C[0] = BaseB[0]

    after = _run_merge_pass(before, aggressive=True)
    after_s = after.script()

    norm_offsets = _buffer_plus_offsets_in_script(after_s, "NormB")
    alt_offsets = _buffer_plus_offsets_in_script(after_s, "AltB")

    assert 'T.allocate([96], "uint8", "shared.dyn")' in after_s
    assert len(norm_offsets) == 1
    assert len(alt_offsets) == 1
    assert norm_offsets != alt_offsets


@tilelang.testing.requires_cuda
def test_branch_alternative_fix_still_allows_later_slot_reuse():
    @T.prim_func(private=True)
    def before(
        A: T.Buffer((8,), "float16"),
        B: T.Buffer((8,), "float16"),
        C: T.Buffer((8,), "float16"),
        D: T.Buffer((8,), "float16"),
        E: T.Buffer((8,), "float16"),
        F: T.Buffer((8,), "float16"),
        cond: T.int32,
    ):
        T.launch_thread("blockIdx.x", 1)
        Base = T.allocate([32], "float16", "shared.dyn")
        Norm = T.allocate([8], "float16", "shared.dyn")
        Alt = T.allocate([8], "float16", "shared.dyn")
        Tail = T.allocate([8], "float16", "shared.dyn")
        T.launch_thread("threadIdx.x", 1)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        BaseB = T.Buffer((32,), "float16", data=Base, scope="shared.dyn")
        NormB = T.Buffer((8,), "float16", data=Norm, scope="shared.dyn")
        AltB = T.Buffer((8,), "float16", data=Alt, scope="shared.dyn")
        TailB = T.Buffer((8,), "float16", data=Tail, scope="shared.dyn")
        BaseB[0] = A[0]
        if cond > 0:
            for i in range(8):
                NormB[i] = A[i]
        else:
            for i in range(8):
                NormB[i] = T.float16(0)
        for i in range(8):
            AltB[i] = C[i]
        for i in range(8):
            B[i] = NormB[i]
            D[i] = AltB[i]
        for i in range(8):
            TailB[i] = E[i]
            F[i] = TailB[i]
        C[0] = BaseB[0]

    after = _run_merge_pass(before, aggressive=True)
    after_s = after.script()

    norm_offsets = _buffer_plus_offsets_in_script(after_s, "NormB")
    alt_offsets = _buffer_plus_offsets_in_script(after_s, "AltB")
    tail_offsets = _buffer_plus_offsets_in_script(after_s, "TailB")

    assert 'T.allocate([96], "uint8", "shared.dyn")' in after_s
    assert len(norm_offsets) == 1
    assert len(alt_offsets) == 1
    assert len(tail_offsets) == 1
    assert tail_offsets in (norm_offsets, alt_offsets)


@tilelang.testing.requires_cuda
def test_block_buffer_regions_are_rewritten_with_distinct_offsets():
    src = """
from tvm.script import tir as T

@T.prim_func
def before(A: T.Buffer((16,), "float16"), B: T.Buffer((16,), "float16")):
    bx = T.launch_thread("blockIdx.x", 1)
    Base = T.allocate([32], "float16", "shared.dyn")
    S0 = T.allocate([16], "float16", "shared.dyn")
    S1 = T.allocate([16], "float16", "shared.dyn")
    tx = T.launch_thread("threadIdx.x", 16)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    BaseB = T.Buffer((32,), "float16", data=Base, scope="shared.dyn")
    S0b = T.Buffer((16,), "float16", data=S0, scope="shared.dyn")
    S1b = T.Buffer((16,), "float16", data=S1, scope="shared.dyn")
    BaseB[0] = A[0]
    with T.block("stage0"):
        v = T.axis.spatial(16, tx)
        T.reads(A[v])
        T.writes(S0b[v])
        S0b[v] = A[v]
    with T.block("stage1"):
        v = T.axis.spatial(16, tx)
        T.reads(S0b[v])
        T.writes(S1b[v], B[v])
        S1b[v] = S0b[v]
        B[v] = S1b[v]
    B[0] = BaseB[0]
"""

    func = _load_prim_func_from_source(src, "/tmp/merge_block_region_case.py").with_attr("global_symbol", "main")
    after = _run_merge_pass(func, aggressive=True)
    after_s = after.script()

    assert 'buf_dyn_shmem = T.allocate([128], "uint8", "shared.dyn")' in after_s
    assert "T.writes(S0b[v + 32])" in after_s
    assert "T.reads(S0b[v + 32])" in after_s
    assert "T.writes(S1b[v + 48], B[v])" in after_s
    assert "S1b[v + 48] = S0b[v + 32]" in after_s


@tilelang.testing.requires_cuda
def test_loop_carried_buffer_stays_disjoint_from_repeated_synced_stages():
    @T.prim_func(private=True)
    def before(
        A: T.Buffer((4,), "float16"),
        B: T.Buffer((4,), "float16"),
        C: T.Buffer((4,), "float16"),
        D: T.Buffer((4,), "float16"),
    ):
        T.launch_thread("blockIdx.x", 1)
        Carry = T.allocate([16], "float16", "shared.dyn")
        Stage0 = T.allocate([16], "float16", "shared.dyn")
        Stage1 = T.allocate([16], "float16", "shared.dyn")
        T.launch_thread("threadIdx.x", 1)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        CarryB = T.Buffer((16,), "float16", data=Carry, scope="shared.dyn")
        Stage0B = T.Buffer((16,), "float16", data=Stage0, scope="shared.dyn")
        Stage1B = T.Buffer((16,), "float16", data=Stage1, scope="shared.dyn")
        CarryB[0] = A[0]
        for i in T.serial(0, 4):
            Stage0B[0] = B[i]
            T.evaluate(T.ptx_commit_group())
            T.evaluate(T.ptx_wait_group(0))
            T.tvm_storage_sync("shared.dyn")
            Stage1B[0] = Stage0B[0]
            C[i] = Stage1B[0]
        D[0] = CarryB[0]

    baseline = _run_merge_pass(before)
    aggressive = _run_merge_pass(before, aggressive=True)

    baseline_s = baseline.script()
    aggressive_s = aggressive.script()

    assert 'T.allocate([96], "uint8", "shared.dyn")' in baseline_s
    assert 'T.allocate([96], "uint8", "shared.dyn")' in aggressive_s
    assert "Stage0B[16] = B[i]" in baseline_s
    assert "Stage1B[32] = Stage0B[16]" in baseline_s
    assert "D[0] = CarryB[0]" in baseline_s
    assert "Stage0B[16] = B[i]" in aggressive_s
    assert "Stage1B[32] = Stage0B[16]" in aggressive_s
    assert "D[0] = CarryB[0]" in aggressive_s


@tilelang.testing.requires_cuda
def test_partial_subregion_live_ranges_do_not_overmerge_whole_buffers():
    @T.prim_func(private=True)
    def before(A: T.Buffer((32,), "float16"), B: T.Buffer((32,), "float16")):
        T.launch_thread("blockIdx.x", 1)
        X = T.allocate([32], "float16", "shared.dyn")
        Y = T.allocate([32], "float16", "shared.dyn")
        T.launch_thread("threadIdx.x", 32)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Xb = T.Buffer((32,), "float16", data=X, scope="shared.dyn")
        Yb = T.Buffer((32,), "float16", data=Y, scope="shared.dyn")
        for i in range(16):
            Xb[i] = A[i]
        for i in range(16, 32):
            Yb[i] = A[i]
        for i in range(16):
            B[i] = Xb[i]
        for i in range(16, 32):
            B[i] = Yb[i]

    baseline = _run_merge_pass(before)
    aggressive = _run_merge_pass(before, aggressive=True)

    baseline_s = baseline.script()
    aggressive_s = aggressive.script()

    assert 'T.allocate([128], "uint8", "shared.dyn")' in baseline_s
    assert "Yb[i + 32] = A[i]" in baseline_s
    assert "B[i] = Yb[i + 32]" in baseline_s
    assert 'T.allocate([128], "uint8", "shared.dyn")' in aggressive_s
    assert "Yb[i + 32] = A[i]" in aggressive_s
    assert "B[i] = Yb[i + 32]" in aggressive_s


@tilelang.testing.requires_cuda
def test_tvm_access_ptr_offsets_follow_merged_buffer_layout():
    src = """
from tvm.script import tir as T

@T.prim_func
def before(A: T.Buffer((16,), "float16"), B: T.Buffer((16,), "float16")):
    bx = T.launch_thread("blockIdx.x", 1)
    Base = T.allocate([32], "float16", "shared.dyn")
    S0 = T.allocate([16], "float16", "shared.dyn")
    S1 = T.allocate([16], "float16", "shared.dyn")
    tx = T.launch_thread("threadIdx.x", 1)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    BaseB = T.Buffer((32,), "float16", data=Base, scope="shared.dyn")
    S0b = T.Buffer((16,), "float16", data=S0, scope="shared.dyn")
    S1b = T.Buffer((16,), "float16", data=S1, scope="shared.dyn")
    BaseB[0] = A[0]
    T.evaluate(T.call_extern("handle", "opaque_write", T.tvm_access_ptr(T.type_annotation("float16"), S0b.data, 0, 16, 2)))
    T.evaluate(T.call_extern("handle", "opaque_read", T.tvm_access_ptr(T.type_annotation("float16"), S0b.data, 0, 16, 1)))
    T.evaluate(T.call_extern("handle", "opaque_write", T.tvm_access_ptr(T.type_annotation("float16"), S1b.data, 0, 16, 2)))
    B[0] = BaseB[0]
"""

    func = _load_prim_func_from_source(src, "/tmp/merge_tvm_access_ptr_case.py").with_attr("global_symbol", "main")
    after = _run_merge_pass(func, aggressive=True)
    after_s = after.script()
    ptr_offsets = _tvm_access_ptr_offsets(after_s)

    assert 'buf_dyn_shmem = T.allocate([96], "uint8", "shared.dyn")' in after_s
    assert ptr_offsets == [32, 32, 32]
    assert "B[0] = BaseB[0]" in after_s


@tilelang.testing.requires_cuda
def test_cp_async_branch_alternatives_share_single_destination_offset_after_sync():
    src = """
import tilelang.language as T

@T.prim_func
def before(A: T.Tensor((32,), T.uint8), B: T.Tensor((32,), T.uint8), C: T.Tensor((16,), T.uint8), cond: T.int32):
    bx = T.launch_thread("blockIdx.x", 1)
    Base = T.allocate([16], "uint8", "shared.dyn")
    S0 = T.allocate([16], "uint8", "shared.dyn")
    S1 = T.allocate([16], "uint8", "shared.dyn")
    tx = T.launch_thread("threadIdx.x", 1)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    BaseB = T.Buffer((16,), "uint8", data=Base, scope="shared.dyn")
    S0b = T.Buffer((16,), "uint8", data=S0, scope="shared.dyn")
    S1b = T.Buffer((16,), "uint8", data=S1, scope="shared.dyn")
    BaseB[0] = C[0]
    if cond > 0:
        T.ptx_cp_async(T.access_ptr(S0b[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
    else:
        T.ptx_cp_async(T.access_ptr(S0b[0], "w", 16), T.access_ptr(B[0], "r", 16), 16)
    T.ptx_commit_group()
    T.ptx_wait_group(0)
    T.tvm_storage_sync("shared.dyn")
    S1b[0] = S0b[0]
    C[1] = S1b[0]
    C[2] = BaseB[0]
"""

    func = _load_prim_func_from_source(src, "/tmp/merge_cp_async_branch_sync_case.py").with_attr("global_symbol", "main")
    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    after = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=True)(mod)["main"]
    after_s = after.script()
    cp_offsets = _access_ptr_offsets(after_s, "S0b")

    assert 'buf_dyn_shmem = T.allocate([48], "uint8", "shared.dyn")' in after_s
    assert cp_offsets == [16, 16]
    assert "S1b[32] = S0b[16]" in after_s
    assert "C_1[2] = BaseB[0]" in after_s


@tilelang.testing.requires_cuda
def test_nested_loop_and_if_alternatives_preserve_single_phase_slot():
    src = """
from tvm.script import tir as T

@T.prim_func
def before(A: T.Buffer((8,), "float16"), B: T.Buffer((8,), "float16"), C: T.Buffer((8,), "float16"), D: T.Buffer((8,), "float16"), outer_cond: T.int32, inner_cond: T.int32):
    bx = T.launch_thread("blockIdx.x", 1)
    Carry = T.allocate([16], "float16", "shared.dyn")
    Phase = T.allocate([16], "float16", "shared.dyn")
    Tmp = T.allocate([16], "float16", "shared.dyn")
    tx = T.launch_thread("threadIdx.x", 1)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    CarryB = T.Buffer((16,), "float16", data=Carry, scope="shared.dyn")
    PhaseB = T.Buffer((16,), "float16", data=Phase, scope="shared.dyn")
    TmpB = T.Buffer((16,), "float16", data=Tmp, scope="shared.dyn")
    CarryB[0] = A[0]
    for i in T.serial(0, 2):
        if outer_cond > 0:
            if inner_cond > 0:
                PhaseB[0] = A[i]
            else:
                PhaseB[0] = B[i]
        else:
            PhaseB[0] = T.float16(0)
        T.evaluate(T.ptx_commit_group())
        T.evaluate(T.ptx_wait_group(0))
        T.tvm_storage_sync("shared.dyn")
        TmpB[0] = PhaseB[0]
        C[i] = TmpB[0]
    D[0] = CarryB[0]
"""

    func = _load_prim_func_from_source(src, "/tmp/merge_nested_loop_if_mix_case.py").with_attr("global_symbol", "main")
    after = _run_merge_pass(func, aggressive=True)
    after_s = after.script()
    phase_offsets = _buffer_constant_offsets_in_script(after_s, "PhaseB")
    tmp_offsets = _buffer_constant_offsets_in_script(after_s, "TmpB")

    assert 'T.allocate([96], "uint8", "shared.dyn")' in after_s
    assert phase_offsets == {16}
    assert tmp_offsets == {32}
    assert "PhaseB[16] = A[i]" in after_s
    assert "PhaseB[16] = B[i]" in after_s
    assert "PhaseB[16] = T.float16(0.0)" in after_s
    assert "TmpB[32] = PhaseB[16]" in after_s
    assert "D[0] = CarryB[0]" in after_s


@tilelang.testing.requires_cuda
def test_merge_dynamic_shared_grows_arena_from_tail_when_freed_block_too_small():
    """Regression for upstream commit b4c913be (#2106): when the only free
    slot reusable by a new buffer sits at the arena tail and is smaller than
    the requested size, the arena must grow by the deficit rather than push
    the new allocation past the tail. The legacy best-fit free-list grew the
    arena by the full need (X(128) + Y(64) + Z(192) = 384 bytes); the
    interval-based LinearScanPack reuses the freed Y slot and only grows the
    tail to fit Z, producing a 320-byte arena (X(128) + Z(192) packed)."""

    @T.prim_func(private=True)
    def before(A: T.Buffer((128,), "float16")):
        T.launch_thread("blockIdx.x", 1)
        X = T.allocate([64], "float16", "shared.dyn")  # 128 bytes, live throughout
        Y = T.allocate([32], "float16", "shared.dyn")  #  64 bytes, dies before Z
        Z = T.allocate([96], "float16", "shared.dyn")  # 192 bytes, lives after Y
        tx = T.launch_thread("threadIdx.x", 64)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Xb = T.Buffer((64,), "float16", data=X, scope="shared.dyn")
        Yb = T.Buffer((32,), "float16", data=Y, scope="shared.dyn")
        Zb = T.Buffer((96,), "float16", data=Z, scope="shared.dyn")

        # Phase 1: X and Y co-live; Z not yet generated.
        Xb[tx] = A[tx]
        if tx < 32:
            Yb[tx] = A[tx]
            A[tx] = Yb[tx]  # last use of Y -> Y becomes free
        # Phase 2: Z generated; X still live, Y already dead -> Z must reuse
        # Y's offset and merely grow the tail.
        if tx < 96:
            Zb[tx] = T.float16(0.0)
        if tx < 64:
            Zb[tx] = Xb[tx]
        if tx < 96:
            A[tx] = Zb[tx]
        A[tx] = Xb[tx]  # last use of X (Z still live -> co-living with X)

    after = _run_merge_pass(before, aggressive=True)
    after_s = after.script()

    # Peak live = X(128B) + Z(192B) = 320B. Y must reuse the offset that Z
    # also occupies (relative to X) since Y dies before Z is born.
    assert 'T.allocate([320], "uint8", "shared.dyn")' in after_s, after_s
    # Y and Z must both start at offset 64 fp16 elements (= 128 bytes, after
    # X). The legacy free-list implementation produced 96 fp16 elements
    # (= 192 bytes) for Z (= bump past the freed Y slot), yielding a
    # 384-byte arena and a "+ 96" expression in the rewritten script.
    y_offsets = _buffer_plus_offsets_in_script(after_s, "Yb")
    z_offsets = _buffer_plus_offsets_in_script(after_s, "Zb")
    assert y_offsets == {64}, after_s
    assert z_offsets == {64}, after_s
    assert "+ 96" not in after_s, after_s


@tilelang.testing.requires_cuda
def test_merged_branch_kernel_generates_correct_results():
    """End-to-end correctness check on a merged branch-alternative kernel.

    Confirms that the arena packing produced by MergeSharedMemoryAllocations
    (64 bytes for 3 buffers that would naively consume 192 bytes) does not
    alter numerical results relative to a bit-exact reference.
    """

    @T.prim_func(private=True)
    def before(
        data: T.Buffer((2,), "float16"),
        out: T.Buffer((1,), "float16"),
        cond: T.int32,
    ):
        T.launch_thread("blockIdx.x", 1)
        A = T.allocate([32], "float16", "shared.dyn")
        B = T.allocate([32], "float16", "shared.dyn")
        C = T.allocate([32], "float16", "shared.dyn")
        T.launch_thread("threadIdx.x", 1)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Ab = T.Buffer((32,), "float16", data=A, scope="shared.dyn")
        Bb = T.Buffer((32,), "float16", data=B, scope="shared.dyn")
        Cb = T.Buffer((32,), "float16", data=C, scope="shared.dyn")
        if cond == 0:
            Ab[0] = data[0]
            out[0] = Ab[0]
        else:
            Bb[0] = data[1]
            out[0] = Bb[0]
        T.tvm_storage_sync("shared.dyn")
        Cb[0] = T.float16(0.0)

    after = _run_merge_pass(before)
    after_s = after.script()

    assert 'T.allocate([64], "uint8", "shared.dyn")' in after_s, after_s

    kernel = tilelang.compile(
        after.with_attr("global_symbol", "test_branch_correctness"),
        target="cuda",
        out_idx=[1],
    )

    import torch  # noqa: E402

    data_t = torch.tensor([1.0, 2.0], dtype=torch.float16, device="cuda")

    out_t = kernel(data_t, 0)
    torch.testing.assert_close(out_t, torch.tensor([1.0], dtype=torch.float16, device="cuda"))

    out_t = kernel(data_t, 1)
    torch.testing.assert_close(out_t, torch.tensor([2.0], dtype=torch.float16, device="cuda"))


@tilelang.testing.requires_cuda
def test_merged_sync_phased_buffers_generate_correct_results():
    """End-to-end correctness on a sync-separated two-buffer kernel."""

    @T.prim_func(private=True)
    def before(
        A: T.Buffer((8,), "float16"),
        B: T.Buffer((8,), "float16"),
        out: T.Buffer((8,), "float16"),
    ):
        T.launch_thread("blockIdx.x", 1)
        X = T.allocate([16], "float16", "shared.dyn")
        Y = T.allocate([16], "float16", "shared.dyn")
        T.launch_thread("threadIdx.x", 1)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Xb = T.Buffer((16,), "float16", data=X, scope="shared.dyn")
        Yb = T.Buffer((16,), "float16", data=Y, scope="shared.dyn")
        Xb[0] = A[0]
        out[0] = Xb[0]
        T.tvm_storage_sync("shared.dyn")
        Yb[0] = B[0]
        out[1] = Yb[0]

    after = _run_merge_pass(before)
    after_s = after.script()

    assert 'T.allocate([32], "uint8", "shared.dyn")' in after_s, after_s

    kernel = tilelang.compile(
        after.with_attr("global_symbol", "test_sync_phased_correctness"),
        target="cuda",
        out_idx=[-1],
    )

    import torch  # noqa: E402

    a_t = torch.ones(8, dtype=torch.float16, device="cuda")
    b_t = torch.full((8,), 2.0, dtype=torch.float16, device="cuda")

    result = kernel(a_t, b_t)
    torch.testing.assert_close(
        result,
        torch.tensor([1.0, 2.0] + [0.0] * 6, dtype=torch.float16, device="cuda"),
    )


def test_dynamic_size_buffer_fallback_does_not_incorrectly_alias():
    """Dynamic-size (symbolic) shared.dyn buffers must not be aliased with
    constant-size buffers when their liveness intervals overlap.  The pass
    deliberately keeps dynamic buffers on the legacy interval path, so the
    result must stay safe: the arena should be at least the sum of the
    constant-size buffer bytes (the dynamic contribution is not checked here
    because T.var introduces a symbolic extent that IR-level assertions
    cannot directly capture).
    """

    @T.prim_func(private=True, check_well_formed=False)
    def before(
        N: T.int64,
        A: T.Buffer((16,), "float16"),
        out: T.Buffer((16,), "float16"),
    ):
        T.launch_thread("blockIdx.x", 1)
        dyn = T.allocate([N], "float16", "shared.dyn")
        fix = T.allocate([16], "float16", "shared.dyn")
        T.launch_thread("threadIdx.x", 1)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        D = T.Buffer((N,), "float16", data=dyn, scope="shared.dyn")
        F = T.Buffer((16,), "float16", data=fix, scope="shared.dyn")
        D[0] = A[0]
        out[0] = D[0]
        T.tvm_storage_sync("shared.dyn")
        F[0] = A[1]
        out[1] = F[0]

    after = _run_merge_pass(before)
    after_s = after.script()

    # Both buffers should be mapped onto the same buf_dyn_shmem (they are
    # sync-separated, so aliasing is safe even with dynamic size).
    assert after_s.count("buf_dyn_shmem") >= 3, after_s
    # D starts at offset 0 (no D[x + N]) — it uses the base of buf_dyn_shmem.
    assert "D[0]" in after_s
    # F starts at offset 0 as well (shared slot).
    assert "F[0]" in after_s
    # Sync separates the two live ranges; arena should contain both.
    assert "D[0] = A[0]" in after_s
    assert "F[0] = A[1]" in after_s


@tilelang.testing.requires_cuda
def test_general_cond_branch_alternatives_share_offset_with_postdom_buffer():
    """Mutually exclusive buffers under a runtime-cond if/else, plus a
    post-dominator buffer after a sync, must collapse to a single 64-byte arena.

    Upstream's linear-scan plan does not see ``if cond == 0: A else: B`` as
    mutually exclusive (it only handles ``threadIdx``-style branches via the
    aggressive path), so it places ``A`` at offset 0 and ``B`` at offset 32,
    yielding a 128-byte arena. The per-epoch / branch-alternative analysis in
    this pass collapses ``A`` and ``B`` onto offset 0, then reuses that same
    offset for ``C`` after the sync, giving a 64-byte arena (50% smaller).
    """

    @T.prim_func(private=True)
    def before(
        A_in: T.Buffer((8,), "float16"),
        out: T.Buffer((8,), "float16"),
        cond: T.int32,
    ):
        T.launch_thread("blockIdx.x", 1)
        A = T.allocate([32], "float16", "shared.dyn")
        B = T.allocate([32], "float16", "shared.dyn")
        C = T.allocate([32], "float16", "shared.dyn")
        T.launch_thread("threadIdx.x", 1)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Ab = T.Buffer((32,), "float16", data=A, scope="shared.dyn")
        Bb = T.Buffer((32,), "float16", data=B, scope="shared.dyn")
        Cb = T.Buffer((32,), "float16", data=C, scope="shared.dyn")
        if cond == 0:
            Ab[0] = A_in[0]
            out[0] = Ab[0]
        else:
            Bb[0] = A_in[1]
            out[0] = Bb[0]
        T.tvm_storage_sync("shared.dyn")
        Cb[0] = A_in[2]
        out[1] = Cb[0]

    after = _run_merge_pass(before)
    after_s = after.script()

    # Arena = max(A, B, C) = 32 fp16 elems = 64 bytes (vs 128 on upstream).
    assert 'T.allocate([64], "uint8", "shared.dyn")' in after_s, after_s
    # All three buffers must start at offset 0 (no '+ N' indices).
    assert "Ab[0]" in after_s
    assert "Bb[0]" in after_s
    assert "Cb[0]" in after_s
    assert "Bb[32]" not in after_s, after_s
