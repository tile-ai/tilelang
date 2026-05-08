import re

from tilelang import tvm as tvm
import tilelang
import tilelang.testing
import pytest
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
    pattern = rf'T\.access_ptr\({re.escape(buffer_name)}\[(\d+)\], \d+, [12]\)'
    return [int(match.group(1)) for match in re.finditer(pattern, script)]


@tilelang.testing.requires_cuda
def test_merge_dynamic_shared_reuses_non_overlapping_buffers():
    @T.prim_func(private=True)
    def before(A: T.Buffer((128,), "float16")):
        bx = T.launch_thread("blockIdx.x", 1)
        X = T.allocate([128], "float16", "shared.dyn")
        Y = T.allocate([128], "float16", "shared.dyn")
        Z = T.allocate([128], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
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
        bx = T.launch_thread("blockIdx.x", 1)
        buf_dyn_shmem = T.allocate([512], "uint8", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
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
        bx = T.launch_thread("blockIdx.x", 1)
        X = T.allocate([128], "float16", "shared")
        Y = T.allocate([128], "float16", "shared")
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
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
    assert 'tx - 64 + 128' not in merged_s
    assert 'T.allocate([256], "uint8", "shared")' in merged_aggr_s
    assert 'tx - 64 + 128' not in merged_aggr_s


@tilelang.testing.requires_cuda
def test_merge_dynamic_shared_rewrites_cp_async_case_after_flatten():
    src = '''
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
'''

    func = _load_prim_func_from_source(src, "/tmp/merge_cp_async_case.py").with_attr("global_symbol", "main")
    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    after = tilelang.transform.MergeSharedMemoryAllocations()(mod)["main"]
    after_s = after.script()

    assert 'buf_dyn_shmem = T.allocate([32], "uint8", "shared.dyn")' in after_s
    assert 'T.ptx_cp_async(T.access_ptr(S0b[0], 16, 2), T.access_ptr(A_1[0], 16, 1), 16)' in after_s
    assert 'S1b[v_1 // 2 + 16] = S0b[v_1 // 2]' in after_s
    assert 'B_1[v_1 // 2] = S1b[v_1 // 2 + 16]' in after_s


@tilelang.testing.requires_cuda
def test_merge_dynamic_shared_lowbit_style_scratch_and_long_buffer_do_not_reuse_yet():
    @T.prim_func(private=True)
    def before(A: T.Buffer((128,), "float16")):
        bx = T.launch_thread("blockIdx.x", 1)
        Scratch = T.allocate([16384], "float16", "shared.dyn")
        Long = T.allocate([8192], "float16", "shared.dyn")
        Meta = T.allocate([4096], "uint8", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
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
    assert 'LongB[tx + 16384 + 16384] = ScratchB[tx]' in baseline_s
    assert 'MetaB[tx] = T.uint8(0)' in baseline_s
    assert 'T.allocate([49152], "uint8", "shared.dyn")' in aggressive_s
    assert 'LongB[tx + 16384 + 16384] = ScratchB[tx]' in aggressive_s


@tilelang.testing.requires_cuda
def test_branch_exclusive_dynamic_buffers_only_shrink_under_aggressive_merge():
    @T.prim_func(private=True)
    def before(A: T.Buffer((128,), "float16")):
        bx = T.launch_thread("blockIdx.x", 1)
        X = T.allocate([16384], "float16", "shared.dyn")
        Y = T.allocate([16384], "float16", "shared.dyn")
        Z = T.allocate([16384], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
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
    assert 'Yb[tx] = A[tx]' in baseline_s
    assert 'Yb[tx] = A[tx]' in aggressive_s


@tilelang.testing.requires_cuda
def test_phase_boundary_sync_allows_dynamic_buffer_reuse():
    @T.prim_func(private=True)
    def before(
        A: T.Buffer((4,), "float16"),
        B: T.Buffer((4,), "float16"),
        C: T.Buffer((4,), "float16"),
    ):
        bx = T.launch_thread("blockIdx.x", 1)
        X = T.allocate([16], "float16", "shared.dyn")
        Y = T.allocate([16], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 1)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
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
    assert 'T.ptx_commit_group()' in after_s
    assert 'T.ptx_wait_group(0)' in after_s
    assert 'T.tvm_storage_sync("shared.dyn")' in after_s
    assert 'Yb[0] = A[1]' in after_s


@tilelang.testing.requires_cuda
def test_lowbit_like_staged_kv_phases_share_single_dynamic_arena():
    @T.prim_func(private=True)
    def before(
        A: T.Buffer((8,), "float16"),
        B: T.Buffer((8,), "float16"),
        C: T.Buffer((8,), "float16"),
        D: T.Buffer((8,), "float16"),
    ):
        bx = T.launch_thread("blockIdx.x", 1)
        KMeta = T.allocate([32], "uint8", "shared.dyn")
        KData = T.allocate([16], "float16", "shared.dyn")
        VMeta = T.allocate([32], "uint8", "shared.dyn")
        VData = T.allocate([16], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 1)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
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
    assert 'KMetaB[0] = T.uint8(1)' in after_s
    assert 'KDataB[0] = A[0]' in after_s
    assert 'VMetaB[0] = T.uint8(2)' in after_s
    assert 'VDataB[0] = C[0]' in after_s
    assert after_s.count('T.ptx_wait_group(0)') == 3
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
        bx = T.launch_thread("blockIdx.x", 1)
        X = T.allocate([16], "float16", "shared.dyn")
        Y = T.allocate([16], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 1)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
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
    assert 'Yb[16] = A[1]' in after_s
    assert 'C[0] = Yb[16]' in after_s
    assert after_s.count('T.ptx_wait_group(0)') == 2
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
        bx = T.launch_thread("blockIdx.x", 1)
        X = T.allocate([16], "float16", "shared.dyn")
        Y = T.allocate([16], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 1)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
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
    assert 'C[i] = Yb[16]' in aggressive_s or 'C[i] = Yb[0]' in aggressive_s
    assert 'D[i] = Yb[16]' in aggressive_s or 'D[i] = Yb[0]' in aggressive_s


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
        bx = T.launch_thread("blockIdx.x", 1)
        Base = T.allocate([32], "float16", "shared.dyn")
        Norm = T.allocate([8], "float16", "shared.dyn")
        Alt = T.allocate([8], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 1)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
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
        bx = T.launch_thread("blockIdx.x", 1)
        Base = T.allocate([32], "float16", "shared.dyn")
        Norm = T.allocate([8], "float16", "shared.dyn")
        Alt = T.allocate([8], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 1)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
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
        bx = T.launch_thread("blockIdx.x", 1)
        Base = T.allocate([32], "float16", "shared.dyn")
        Norm = T.allocate([8], "float16", "shared.dyn")
        Alt = T.allocate([8], "float16", "shared.dyn")
        Tail = T.allocate([8], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 1)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
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
    assert tail_offsets == norm_offsets or tail_offsets == alt_offsets


@tilelang.testing.requires_cuda
def test_block_buffer_regions_are_rewritten_with_distinct_offsets():
    src = '''
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
'''

    func = _load_prim_func_from_source(src, "/tmp/merge_block_region_case.py").with_attr("global_symbol", "main")
    after = _run_merge_pass(func, aggressive=True)
    after_s = after.script()

    assert 'buf_dyn_shmem = T.allocate([128], "uint8", "shared.dyn")' in after_s
    assert 'T.writes(S0b[v + 32])' in after_s
    assert 'T.reads(S0b[v + 32])' in after_s
    assert 'T.writes(S1b[v + 48], B[v])' in after_s
    assert 'S1b[v + 48] = S0b[v + 32]' in after_s


@tilelang.testing.requires_cuda
def test_loop_carried_buffer_stays_disjoint_from_repeated_synced_stages():
    @T.prim_func(private=True)
    def before(
        A: T.Buffer((4,), "float16"),
        B: T.Buffer((4,), "float16"),
        C: T.Buffer((4,), "float16"),
        D: T.Buffer((4,), "float16"),
    ):
        bx = T.launch_thread("blockIdx.x", 1)
        Carry = T.allocate([16], "float16", "shared.dyn")
        Stage0 = T.allocate([16], "float16", "shared.dyn")
        Stage1 = T.allocate([16], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 1)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
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
    assert 'Stage0B[16] = B[i]' in baseline_s
    assert 'Stage1B[32] = Stage0B[16]' in baseline_s
    assert 'D[0] = CarryB[0]' in baseline_s
    assert 'Stage0B[16] = B[i]' in aggressive_s
    assert 'Stage1B[32] = Stage0B[16]' in aggressive_s
    assert 'D[0] = CarryB[0]' in aggressive_s


@tilelang.testing.requires_cuda
def test_partial_subregion_live_ranges_do_not_overmerge_whole_buffers():
    @T.prim_func(private=True)
    def before(A: T.Buffer((32,), "float16"), B: T.Buffer((32,), "float16")):
        bx = T.launch_thread("blockIdx.x", 1)
        X = T.allocate([32], "float16", "shared.dyn")
        Y = T.allocate([32], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 32)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
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
    assert 'Yb[i + 32] = A[i]' in baseline_s
    assert 'B[i] = Yb[i + 32]' in baseline_s
    assert 'T.allocate([128], "uint8", "shared.dyn")' in aggressive_s
    assert 'Yb[i + 32] = A[i]' in aggressive_s
    assert 'B[i] = Yb[i + 32]' in aggressive_s


@tilelang.testing.requires_cuda
def test_tvm_access_ptr_offsets_follow_merged_buffer_layout():
    src = '''
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
'''

    func = _load_prim_func_from_source(src, "/tmp/merge_tvm_access_ptr_case.py").with_attr("global_symbol", "main")
    after = _run_merge_pass(func, aggressive=True)
    after_s = after.script()
    ptr_offsets = _tvm_access_ptr_offsets(after_s)

    assert 'buf_dyn_shmem = T.allocate([96], "uint8", "shared.dyn")' in after_s
    assert ptr_offsets == [32, 32, 32]
    assert 'B[0] = BaseB[0]' in after_s


@tilelang.testing.requires_cuda
def test_cp_async_branch_alternatives_share_single_destination_offset_after_sync():
    src = '''
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
'''

    func = _load_prim_func_from_source(src, "/tmp/merge_cp_async_branch_sync_case.py").with_attr("global_symbol", "main")
    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    after = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=True)(mod)["main"]
    after_s = after.script()
    cp_offsets = _access_ptr_offsets(after_s, "S0b")

    assert 'buf_dyn_shmem = T.allocate([48], "uint8", "shared.dyn")' in after_s
    assert cp_offsets == [16, 16]
    assert 'S1b[32] = S0b[16]' in after_s
    assert 'C_1[2] = BaseB[0]' in after_s


@tilelang.testing.requires_cuda
def test_nested_loop_and_if_alternatives_preserve_single_phase_slot():
    src = '''
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
'''

    func = _load_prim_func_from_source(src, "/tmp/merge_nested_loop_if_mix_case.py").with_attr("global_symbol", "main")
    after = _run_merge_pass(func, aggressive=True)
    after_s = after.script()
    phase_offsets = _buffer_constant_offsets_in_script(after_s, "PhaseB")
    tmp_offsets = _buffer_constant_offsets_in_script(after_s, "TmpB")

    assert 'T.allocate([96], "uint8", "shared.dyn")' in after_s
    assert phase_offsets == {16}
    assert tmp_offsets == {32}
    assert 'PhaseB[16] = A[i]' in after_s
    assert 'PhaseB[16] = B[i]' in after_s
    assert 'PhaseB[16] = T.float16(0.0)' in after_s
    assert 'TmpB[32] = PhaseB[16]' in after_s
    assert 'D[0] = CarryB[0]' in after_s
