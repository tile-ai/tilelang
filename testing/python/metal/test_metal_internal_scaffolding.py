"""Internal-only Metal scaffolding/source-boundary/runtime probes.

These tests intentionally exercise private helper modules under ``tilelang.tileop``
without adding public language aliases such as ``T.rt`` or ``T.rv``.

Coverage notes:
- Runtime-validates packed uint8 fp8/fp4/e8m0 decode on MPS against a CPU
  reference using synthetic tensors only.
- Runtime-validates a GDN/attention-style KKT 8x8 score tile on MPS against
  ``torch.matmul`` using synthetic tensors only.
- Runtime-validates a small packed fp8 activation x packed fp4 weight matmul
  with e8m0 scale bytes on MPS.
- Runtime-validates a small GDN/attention-style staged W/U tile on MPS using
  internal RegisterTile helpers plus scalar local.var state.
- Runtime-validates component-scale packed quant matmul (`M=16,N=32,K=64`) and
  GDN/attention-style staged KKT/gate/WU probes
  (`chunk=16,key_dim=16,value_dim=16`) using synthetic tensors only.
- The internal RegisterTile/simdgroup helper is runtime-validated on MPS for
  one 8x8 fp32 MMA and remains source-boundary checked for PR #1869 tokens.
- Native Metal fp8/fp4 storage remains fail-closed/source-boundary only.
- Optional timing hooks are disabled by default and require
  ``TILELANG_RUN_METAL_SMALL_BENCH=1``, ``TILELANG_RUN_METAL_SCALED_BENCH=1``,
  or ``TILELANG_RUN_METAL_COMPONENT_BENCH=1``.
"""

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang import tvm as tvm
import tilelang.language as T
from tilelang.tileop import metal_gdn, metal_quant, metal_simdgroup as metal_sg


_FORBIDDEN_EXTERNAL_TOKENS = (
    "cooperative",
    "mpp",
    "mpsgraph",
    "warpgroup",
    "cp.async",
    "tcgen",
    "tma",
    "tl.ptx",
    "tl.cuda",
)


def _lower_source(func) -> str:
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(func, target="metal")
    assert artifact.kernel_source is not None
    return artifact.kernel_source


def _assert_clean_metal_source(src: str) -> None:
    lowered = src.lower()
    for token in _FORBIDDEN_EXTERNAL_TOKENS:
        assert token not in lowered, f"unexpected token {token!r} in generated Metal source:\n{src}"


def _make_register_tile_probe():
    @T.prim_func
    def register_tile_probe(
        A: T.Tensor((8, 8), T.float32),
        B: T.Tensor((8, 8), T.float32),
        C: T.Tensor((8, 8), T.float32),
    ):
        with T.Kernel(1, threads=32):
            A_rt = metal_sg.alloc_rt(T.float32, 1, 1)
            B_rt = metal_sg.alloc_rt(T.float32, 1, 1)
            C_rt = metal_sg.alloc_rt(T.float32, 1, 1)
            metal_sg.fill_rt(C_rt, T.float32(0.0))
            metal_sg.load_global_to_rt(A_rt, T.float32, A.data, 0, 64, 8)
            metal_sg.load_global_to_rt(B_rt, T.float32, B.data, 0, 64, 8)
            metal_sg.mma_ab(C_rt, A_rt, B_rt)
            metal_sg.store_rt(C_rt, T.float32, C.data, 0, 64, 8)

    return register_tile_probe


def _make_row_vector_probe():
    @T.prim_func
    def row_vector_probe(
        A: T.Tensor((8, 8), T.float32),
        B: T.Tensor((8, 8), T.float32),
        stats: T.Tensor((8, 2), T.float32),
        normalized: T.Tensor((8, 8), T.float32),
    ):
        with T.Kernel(1, threads=32):
            lane = T.get_thread_binding()
            A_rt = metal_sg.alloc_rt(T.float32, 1, 1)
            Bt_rt = metal_sg.alloc_rt(T.float32, 1, 1, layout=metal_sg.TileLayout.TRANSPOSED)
            C_rt = metal_sg.alloc_rt(T.float32, 1, 1)
            C_shared = T.alloc_shared((8, 8), T.float32)
            row_max_shared = T.alloc_shared((8,), T.float32)
            row_sum_shared = T.alloc_shared((8,), T.float32)
            row_max = metal_sg.RowVector(row_max_shared, 8, T.float32)
            row_sum = metal_sg.RowVector(row_sum_shared, 8, T.float32)

            metal_sg.fill_rt(C_rt, T.float32(0.0))
            metal_sg.load_global_to_rt(A_rt, T.float32, A.data, 0, 64, 8)
            metal_sg.load_global_to_rt(Bt_rt, T.float32, B.data, 0, 64, 8, transpose=True)
            metal_sg.mma_abt(C_rt, A_rt, Bt_rt)
            metal_sg.materialize_rt_to_shared(C_rt, T.float32, C_shared.data, 0, 64, 8)
            T.sync_threads()

            metal_sg.row_max(C_shared, row_max, rows=8, cols=8, clear=True)
            metal_sg.row_sum(C_shared, row_sum, rows=8, cols=8, clear=True)
            metal_sg.div_row(C_shared, row_sum, rows=8, cols=8)
            T.sync_threads()

            for linear in T.serial(lane, 8 * 10, step=32):
                row = linear // 10
                col = linear - row * 10
                if col == 0:
                    stats[row, 0] = row_max.values[row]
                elif col == 1:
                    stats[row, 1] = row_sum.values[row]
                else:
                    normalized[row, col - 2] = C_shared[row, col - 2]

    return row_vector_probe


def _make_deepseek_packed_quant_probe():
    @T.prim_func
    def deepseek_packed_quant_probe(
        q8: T.Tensor((16,), T.uint8),
        q4: T.Tensor((8,), T.uint8),
        e8m0_scale: T.Tensor((16,), T.uint8),
        out: T.Tensor((16,), T.float32),
    ):
        with T.Kernel(1, threads=32):
            lane = T.get_thread_binding()
            for i in T.serial(lane, 16, step=32):
                nibble_index = i - (i // 2) * 2
                decoded_fp8 = metal_quant.fp8_e4m3fn_to_float(q8[i])
                decoded_fp4 = metal_quant.fp4_e2m1fn_to_float(q4[i // 2], nibble_index)
                scale = metal_quant.e8m0_to_float(e8m0_scale[i])
                out[i] = decoded_fp8 * scale + decoded_fp4

    return deepseek_packed_quant_probe


def _make_deepseek_packed_quant_matmul_probe():
    @T.prim_func
    def deepseek_packed_quant_matmul_probe(
        q8_act: T.Tensor((8, 16), T.uint8),
        q4_weight: T.Tensor((8, 8), T.uint8),
        act_scale: T.Tensor((8, 16), T.uint8),
        weight_scale: T.Tensor((8, 16), T.uint8),
        out: T.Tensor((8, 8), T.float32),
    ):
        with T.Kernel(1, threads=32):
            lane = T.get_thread_binding()
            for linear in T.serial(lane, 64, step=32):
                m = linear // 8
                n = linear - m * 8
                acc = T.alloc_var(T.float32)
                acc = 0.0
                for k in T.serial(16):
                    nibble_index = k - (k // 2) * 2
                    decoded_act = metal_quant.fp8_e4m3fn_to_float(q8_act[m, k])
                    decoded_weight = metal_quant.fp4_e2m1fn_to_float(q4_weight[n, k // 2], nibble_index)
                    scale = metal_quant.e8m0_to_float(act_scale[m, k]) * metal_quant.e8m0_to_float(
                        weight_scale[n, k]
                    )
                    acc += decoded_act * decoded_weight * scale
                out[m, n] = acc

    return deepseek_packed_quant_matmul_probe


def _make_deepseek_component_quant_matmul_probe():
    @T.prim_func
    def deepseek_component_quant_matmul_probe(
        q8_act: T.Tensor((16, 64), T.uint8),
        q4_weight: T.Tensor((32, 32), T.uint8),
        act_scale: T.Tensor((16, 64), T.uint8),
        weight_scale: T.Tensor((32, 64), T.uint8),
        out: T.Tensor((16, 32), T.float32),
    ):
        with T.Kernel(1, threads=32):
            lane = T.get_thread_binding()
            for linear in T.serial(lane, 16 * 32, step=32):
                m = linear // 32
                n = linear - m * 32
                acc = T.alloc_var(T.float32)
                acc = 0.0
                for k in T.serial(64):
                    nibble_index = k - (k // 2) * 2
                    decoded_act = metal_quant.fp8_e4m3fn_to_float(q8_act[m, k])
                    decoded_weight = metal_quant.fp4_e2m1fn_to_float(q4_weight[n, k // 2], nibble_index)
                    scale = metal_quant.e8m0_to_float(act_scale[m, k]) * metal_quant.e8m0_to_float(
                        weight_scale[n, k]
                    )
                    acc += decoded_act * decoded_weight * scale
                out[m, n] = acc

    return deepseek_component_quant_matmul_probe


def _make_flashqla_gdn_kkt_probe():
    @T.prim_func
    def flashqla_gdn_kkt_probe(
        row_k: T.Tensor((8, 8), T.float32),
        col_k: T.Tensor((8, 8), T.float32),
        scores: T.Tensor((8, 8), T.float32),
    ):
        with T.Kernel(1, threads=32):
            lane = T.get_thread_binding()
            kkt_bias = T.alloc_var(T.float32)
            kkt_bias = 0.0
            row_shared = T.alloc_shared((8, 8), T.float32)
            col_shared = T.alloc_shared((8, 8), T.float32)
            score_shared = T.alloc_shared((8, 8), T.float32)
            for idx in T.serial(lane, 64, step=32):
                r = idx // 8
                c = idx - r * 8
                row_shared[r, c] = row_k[r, c]
                col_shared[r, c] = col_k[r, c]
            T.sync_threads()
            metal_gdn.kkt_score_tile(row_shared.data, col_shared.data, score_shared.data, block=8, key_dim=8)
            T.sync_threads()
            for idx in T.serial(lane, 64, step=32):
                r = idx // 8
                c = idx - r * 8
                scores[r, c] = score_shared[r, c] + kkt_bias

    return flashqla_gdn_kkt_probe


def _make_flashqla_gdn_wu_probe():
    @T.prim_func
    def flashqla_gdn_wu_probe(
        a: T.Tensor((8, 8), T.float32),
        k: T.Tensor((8, 8), T.float32),
        v: T.Tensor((8, 8), T.float32),
        beta: T.Tensor((8,), T.float32),
        g_cum: T.Tensor((8,), T.float32),
        w: T.Tensor((8, 8), T.float32),
        u: T.Tensor((8, 8), T.float32),
    ):
        with T.Kernel(1, threads=32):
            lane = T.get_thread_binding()
            gate_state = T.alloc_var(T.float32)
            gate_state = 1.0
            a_shared = T.alloc_shared((8, 8), T.float32)
            k_scaled_shared = T.alloc_shared((8, 8), T.float32)
            v_scaled_shared = T.alloc_shared((8, 8), T.float32)
            w_acc = metal_sg.alloc_rt(T.float32, 1, 1)
            u_acc = metal_sg.alloc_rt(T.float32, 1, 1)
            metal_sg.fill_rt(w_acc, T.float32(0.0))
            metal_sg.fill_rt(u_acc, T.float32(0.0))
            for idx in T.serial(lane, 64, step=32):
                r = idx // 8
                c = idx - r * 8
                a_shared[r, c] = a[r, c]
                k_scaled_shared[r, c] = k[r, c] * beta[r] * T.exp(g_cum[r]) * gate_state
                v_scaled_shared[r, c] = v[r, c] * beta[r] * gate_state
            T.sync_threads()
            metal_gdn.wu_score_tiles(a_shared.data, k_scaled_shared.data, v_scaled_shared.data, w_acc, u_acc, block=8)
            metal_sg.store_rt(w_acc, T.float32, w.data, 0, 64, 8)
            metal_sg.store_rt(u_acc, T.float32, u.data, 0, 64, 8)

    return flashqla_gdn_wu_probe


def _make_flashqla_gdn_component_probe():
    @T.prim_func
    def flashqla_gdn_component_probe(
        k: T.Tensor((16, 16), T.float32),
        v: T.Tensor((16, 16), T.float32),
        beta: T.Tensor((16,), T.float32),
        g_cum: T.Tensor((16,), T.float32),
        a_pre: T.Tensor((16, 16), T.float32),
        w: T.Tensor((16, 16), T.float32),
        u: T.Tensor((16, 16), T.float32),
    ):
        with T.Kernel(1, threads=32):
            lane = T.get_thread_binding()
            gate_state = T.alloc_var(T.float32)
            gate_state = 1.0
            row_shared = T.alloc_shared((8, 16), T.float32)
            col_shared = T.alloc_shared((8, 16), T.float32)
            score_shared = T.alloc_shared((8, 8), T.float32)
            a_shared = T.alloc_shared((16, 16), T.float32)
            k_scaled_shared = T.alloc_shared((16, 16), T.float32)
            v_scaled_shared = T.alloc_shared((16, 16), T.float32)
            w_acc = metal_sg.alloc_rt(T.float32, 1, 1)
            u_acc = metal_sg.alloc_rt(T.float32, 1, 1)

            for idx in T.serial(lane, 16 * 16, step=32):
                r = idx // 16
                c = idx - r * 16
                k_scaled_shared[r, c] = k[r, c] * beta[r] * T.exp(g_cum[r]) * gate_state
                v_scaled_shared[r, c] = v[r, c] * beta[r] * gate_state
                a_shared[r, c] = 0.0
            T.sync_threads()

            for row_block in T.unroll(2, explicit=True):
                for col_block in T.unroll(2, explicit=True):
                    for idx in T.serial(lane, 8 * 16, step=32):
                        r = idx // 16
                        c = idx - r * 16
                        row_shared[r, c] = k[row_block * 8 + r, c]
                        col_shared[r, c] = k[col_block * 8 + r, c]
                    T.sync_threads()
                    metal_gdn.kkt_score_tile_accum(
                        row_shared.data,
                        col_shared.data,
                        score_shared.data,
                        block=8,
                        key_dim=16,
                        key_offset=0,
                        clear=True,
                    )
                    metal_gdn.kkt_score_tile_accum(
                        row_shared.data,
                        col_shared.data,
                        score_shared.data,
                        block=8,
                        key_dim=16,
                        key_offset=8,
                        clear=False,
                    )
                    T.sync_threads()
                    for idx in T.serial(lane, 8 * 8, step=32):
                        local_row = idx // 8
                        local_col = idx - local_row * 8
                        c = row_block * 8 + local_row
                        d = col_block * 8 + local_col
                        gated = T.alloc_var(T.float32)
                        gated = 0.0
                        if d < c:
                            gated = score_shared[local_row, local_col] * T.exp(g_cum[c] - g_cum[d]) * gate_state
                        a_pre[c, d] = gated
                        a_shared[c, d] = gated
                    T.sync_threads()

            for row_block in T.unroll(2, explicit=True):
                for col_block in T.unroll(2, explicit=True):
                    metal_sg.fill_rt(w_acc, T.float32(0.0))
                    metal_sg.fill_rt(u_acc, T.float32(0.0))
                    for d_block in T.unroll(2, explicit=True):
                        metal_gdn.wu_score_tiles_strided(
                            a_shared.data,
                            k_scaled_shared.data,
                            v_scaled_shared.data,
                            w_acc,
                            u_acc,
                            a_offset=row_block * 8 * 16 + d_block * 8,
                            k_offset=d_block * 8 * 16 + col_block * 8,
                            v_offset=d_block * 8 * 16 + col_block * 8,
                            a_stride=16,
                            kv_stride=16,
                            block=8,
                        )
                    metal_sg.store_rt(
                        w_acc,
                        T.float32,
                        w.data,
                        row_block * 8 * 16 + col_block * 8,
                        16 * 16,
                        16,
                    )
                    metal_sg.store_rt(
                        u_acc,
                        T.float32,
                        u.data,
                        row_block * 8 * 16 + col_block * 8,
                        16 * 16,
                        16,
                    )

    return flashqla_gdn_component_probe


def test_internal_register_tile_helper_emits_pr_simdgroup_tokens_only():
    src = _lower_source(_make_register_tile_probe())
    _assert_clean_metal_source(src)
    assert "simdgroup_multiply_accumulate" in src
    assert "simdgroup_load" in src
    assert "simdgroup_store" in src
    assert "simdgroup_float8x8" in src
    assert "C_tmp" not in src


def test_row_vector_remains_materialized_not_scalar_indexed_simdgroup_fragment():
    src = _lower_source(_make_row_vector_probe())
    _assert_clean_metal_source(src)
    assert "simdgroup_multiply_accumulate" in src
    assert "threadgroup float" in src
    assert "simdgroup_float8x8" in src
    # RowVector reductions are over materialized threadgroup storage, not scalar
    # indexing into opaque simdgroup_matrix fragments.
    assert "rt_fragment[0][" not in src
    assert "rt_fragment_1[0][" not in src
    assert "rt_fragment_2[0][" not in src


def test_no_public_register_tile_or_row_vector_language_aliases():
    assert not hasattr(T, "rt")
    assert not hasattr(T, "rv")
    assert not hasattr(T, "RegisterTile")
    assert not hasattr(T, "RowVector")


def test_deepseek_packed_quant_probe_uses_uint8_boundary_not_native_fp8_fp4_storage():
    src = _lower_source(_make_deepseek_packed_quant_probe())
    _assert_clean_metal_source(src)
    lowered = src.lower()
    assert "device uchar" in lowered
    assert "float8" not in lowered
    assert "float4" not in lowered
    assert "simdgroup_multiply_accumulate" not in lowered
    assert metal_quant.use_large_simdgroup_tile(64, 512, mixed_fp4_weight=True)
    assert not metal_quant.use_large_simdgroup_tile(64, 256, mixed_fp4_weight=True)


def test_flashqla_gdn_kkt_probe_combines_local_var_state_and_simdgroup_boundary():
    src = _lower_source(_make_flashqla_gdn_kkt_probe())
    _assert_clean_metal_source(src)
    assert "simdgroup_multiply_accumulate" in src
    assert "simdgroup_load" in src
    assert "simdgroup_store" in src
    assert "threadgroup float" in src
    assert "local.var" not in src
    assert "float kkt_bias = 0.000000e+00f;" in src


def test_scaled_packed_quant_and_gdn_probes_source_boundary_tokens():
    deepseek_src = _lower_source(_make_deepseek_packed_quant_matmul_probe())
    _assert_clean_metal_source(deepseek_src)
    deepseek_lowered = deepseek_src.lower()
    assert deepseek_lowered.count("device uchar") >= 4
    assert "float8" not in deepseek_lowered
    assert "float4" not in deepseek_lowered
    assert "simdgroup_multiply_accumulate" not in deepseek_lowered

    gdn_src = _lower_source(_make_flashqla_gdn_wu_probe())
    _assert_clean_metal_source(gdn_src)
    assert "simdgroup_multiply_accumulate" in gdn_src
    assert gdn_src.count("simdgroup_load") >= 3
    assert gdn_src.count("simdgroup_store") >= 2
    assert "threadgroup float" in gdn_src
    assert "local.var" not in gdn_src
    assert "float gate_state" in gdn_src
    assert "gate_state = 1.000000e+00f;" in gdn_src


def test_component_packed_quant_and_gdn_probes_source_boundary_tokens():
    deepseek_src = _lower_source(_make_deepseek_component_quant_matmul_probe())
    _assert_clean_metal_source(deepseek_src)
    deepseek_lowered = deepseek_src.lower()
    assert deepseek_lowered.count("device uchar") >= 4
    assert "float8" not in deepseek_lowered
    assert "float4" not in deepseek_lowered
    assert "simdgroup_multiply_accumulate" not in deepseek_lowered

    gdn_src = _lower_source(_make_flashqla_gdn_component_probe())
    _assert_clean_metal_source(gdn_src)
    assert gdn_src.count("simdgroup_multiply_accumulate") >= 12
    assert gdn_src.count("simdgroup_load") >= 18
    assert gdn_src.count("simdgroup_store") >= 12
    assert "threadgroup float" in gdn_src
    assert "local.var" not in gdn_src
    assert "float gate_state" in gdn_src
    assert "gate_state = 1.000000e+00f;" in gdn_src


def _run_native_dtype_probe(tmp_path: Path, dtype_name: str) -> subprocess.CompletedProcess[str]:
    script = tmp_path / f"probe_{dtype_name}.py"
    script.write_text(
        textwrap.dedent(
            f'''
            import tilelang
            import tilelang.language as T

            @tilelang.jit(out_idx=[-1])
            def bad_kernel(M):
                @T.prim_func
                def main(A: T.Tensor((M,), T.float32), B: T.Tensor((M,), T.{dtype_name})):
                    with T.Kernel(T.ceildiv(M, 32), threads=32) as bx:
                        for i in T.Parallel(32):
                            B[bx * 32 + i] = A[bx * 32 + i]
                return main

            bad_kernel(32)
            '''
        )
    )
    env = os.environ.copy()
    repo_root = str(Path.cwd())
    env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("dtype_name", ["float8_e4m3fn", "float4_e2m1fn"])
def test_native_fp8_fp4_metal_storage_fail_closed_in_subprocess(tmp_path, dtype_name):
    result = _run_native_dtype_probe(tmp_path, dtype_name)
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert f"Cannot convert type {dtype_name} to Metal type" in combined


def _fp8_e4m3fn_to_float_cpu(bits: int) -> float:
    abs_bits = bits & 0x7F
    sign = (bits >> 7) & 1
    exp_bits = (bits >> 3) & 0xF
    mant_bits = bits & 0x7
    if exp_bits == 0:
        value = mant_bits / 512.0
    else:
        value = (1.0 + mant_bits / 8.0) * (2.0 ** (exp_bits - 7))
    if abs_bits == 0x7F:
        value = 0.0
    return -value if sign else value


def _fp4_e2m1fn_to_float_cpu(bits: int, nibble_index: int) -> float:
    nibble = (bits >> (nibble_index * 4)) & 0xF
    sign = (nibble >> 3) & 1
    mag = nibble & 0x7
    value = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)[mag]
    return -value if sign else value


def _e8m0_to_float_cpu(bits: int) -> float:
    return 0.0 if bits == 255 else 2.0 ** (bits - 127)


def _deepseek_synthetic_inputs():
    q8 = torch.tensor(
        [0, 1, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 120],
        dtype=torch.uint8,
    )
    q4 = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=torch.uint8)
    e8m0_scale = torch.tensor(
        [127, 128, 126, 129, 125, 130, 124, 131, 127, 128, 126, 129, 125, 130, 124, 255],
        dtype=torch.uint8,
    )
    return q8, q4, e8m0_scale


def _deepseek_decode_ref(q8: torch.Tensor, q4: torch.Tensor, e8m0_scale: torch.Tensor) -> torch.Tensor:
    values = []
    for i in range(16):
        decoded_fp8 = _fp8_e4m3fn_to_float_cpu(int(q8[i]))
        decoded_fp4 = _fp4_e2m1fn_to_float_cpu(int(q4[i // 2]), i % 2)
        scale = _e8m0_to_float_cpu(int(e8m0_scale[i]))
        values.append(decoded_fp8 * scale + decoded_fp4)
    return torch.tensor(values, dtype=torch.float32)


def _deepseek_matmul_synthetic_inputs():
    q8_act = ((torch.arange(128, dtype=torch.int16).reshape(8, 16) * 3 + 5) % 121).to(torch.uint8)
    q4_weight = ((torch.arange(64, dtype=torch.int16).reshape(8, 8) * 5 + 1) % 256).to(torch.uint8)
    act_scale = (126 + (torch.arange(128, dtype=torch.int16).reshape(8, 16) % 5)).to(torch.uint8)
    weight_scale = (125 + ((torch.arange(128, dtype=torch.int16).reshape(8, 16) * 2) % 5)).to(torch.uint8)
    return q8_act, q4_weight, act_scale, weight_scale


def _deepseek_matmul_ref(
    q8_act: torch.Tensor,
    q4_weight: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty((8, 8), dtype=torch.float32)
    for m in range(8):
        for n in range(8):
            acc = 0.0
            for k in range(16):
                decoded_act = _fp8_e4m3fn_to_float_cpu(int(q8_act[m, k]))
                decoded_weight = _fp4_e2m1fn_to_float_cpu(int(q4_weight[n, k // 2]), k % 2)
                scale = _e8m0_to_float_cpu(int(act_scale[m, k])) * _e8m0_to_float_cpu(int(weight_scale[n, k]))
                acc += decoded_act * decoded_weight * scale
            out[m, n] = acc
    return out


def _deepseek_component_matmul_synthetic_inputs():
    q8_act = ((torch.arange(16 * 64, dtype=torch.int16).reshape(16, 64) * 7 + 11) % 121).to(torch.uint8)
    q4_weight = ((torch.arange(32 * 32, dtype=torch.int16).reshape(32, 32) * 13 + 3) % 256).to(torch.uint8)
    act_scale = (124 + ((torch.arange(16 * 64, dtype=torch.int16).reshape(16, 64) * 3) % 7)).to(torch.uint8)
    weight_scale = (123 + ((torch.arange(32 * 64, dtype=torch.int16).reshape(32, 64) * 5) % 7)).to(torch.uint8)
    return q8_act, q4_weight, act_scale, weight_scale


def _deepseek_component_matmul_ref(
    q8_act: torch.Tensor,
    q4_weight: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    m_size, k_size = q8_act.shape
    n_size = q4_weight.shape[0]
    out = torch.empty((m_size, n_size), dtype=torch.float32)
    for m in range(m_size):
        for n in range(n_size):
            acc = 0.0
            for k in range(k_size):
                decoded_act = _fp8_e4m3fn_to_float_cpu(int(q8_act[m, k]))
                decoded_weight = _fp4_e2m1fn_to_float_cpu(int(q4_weight[n, k // 2]), k % 2)
                scale = _e8m0_to_float_cpu(int(act_scale[m, k])) * _e8m0_to_float_cpu(int(weight_scale[n, k]))
                acc += decoded_act * decoded_weight * scale
            out[m, n] = acc
    return out


def _flashqla_gdn_wu_synthetic_inputs():
    a = torch.tril((torch.arange(64, dtype=torch.float32).reshape(8, 8) - 9.0) / 23.0)
    k = (torch.arange(64, dtype=torch.float32).reshape(8, 8).flip(0) - 5.0) / 17.0
    v = (torch.arange(64, dtype=torch.float32).reshape(8, 8).flip(1) + 3.0) / 19.0
    beta = torch.linspace(0.25, 1.125, 8, dtype=torch.float32)
    g_cum = torch.linspace(-0.375, 0.5, 8, dtype=torch.float32)
    return a, k, v, beta, g_cum


def _flashqla_gdn_wu_ref(
    a: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_scaled = k * (beta * torch.exp(g_cum)).unsqueeze(1)
    v_scaled = v * beta.unsqueeze(1)
    return a @ k_scaled, a @ v_scaled


def _flashqla_gdn_component_synthetic_inputs():
    k = (torch.arange(16 * 16, dtype=torch.float32).reshape(16, 16) - 31.0) / 37.0
    v = (torch.arange(16 * 16, dtype=torch.float32).reshape(16, 16).flip(1) + 7.0) / 41.0
    beta = torch.linspace(0.125, 1.0625, 16, dtype=torch.float32)
    g_cum = torch.linspace(-0.5, 0.625, 16, dtype=torch.float32)
    return k, v, beta, g_cum


def _flashqla_gdn_component_ref(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = k @ k.T
    row_idx = torch.arange(16).view(16, 1)
    col_idx = torch.arange(16).view(1, 16)
    causal = col_idx < row_idx
    gated = scores * torch.exp(g_cum.view(16, 1) - g_cum.view(1, 16))
    a_pre = torch.where(causal, gated, torch.zeros_like(gated))
    k_scaled = k * (beta * torch.exp(g_cum)).unsqueeze(1)
    v_scaled = v * beta.unsqueeze(1)
    return a_pre, a_pre @ k_scaled, a_pre @ v_scaled


@tilelang.testing.requires_metal
def test_deepseek_packed_decode_runtime_mps_matches_cpu_reference():
    kernel = tilelang.compile(_make_deepseek_packed_quant_probe(), target="metal")
    q8, q4, e8m0_scale = _deepseek_synthetic_inputs()
    out = torch.empty(16, dtype=torch.float32, device="mps")

    kernel(q8.to("mps"), q4.to("mps"), e8m0_scale.to("mps"), out)
    torch.mps.synchronize()

    ref = _deepseek_decode_ref(q8, q4, e8m0_scale)
    assert torch.allclose(out.cpu(), ref, atol=1e-6, rtol=1e-6)


@tilelang.testing.requires_metal
def test_deepseek_packed_quant_matmul_runtime_mps_matches_cpu_reference():
    kernel = tilelang.compile(_make_deepseek_packed_quant_matmul_probe(), target="metal")
    q8_act, q4_weight, act_scale, weight_scale = _deepseek_matmul_synthetic_inputs()
    out = torch.empty((8, 8), dtype=torch.float32, device="mps")

    kernel(q8_act.to("mps"), q4_weight.to("mps"), act_scale.to("mps"), weight_scale.to("mps"), out)
    torch.mps.synchronize()

    ref = _deepseek_matmul_ref(q8_act, q4_weight, act_scale, weight_scale)
    assert torch.allclose(out.cpu(), ref, atol=1e-4, rtol=1e-5)


@tilelang.testing.requires_metal
def test_deepseek_component_quant_matmul_runtime_mps_matches_cpu_reference():
    kernel = tilelang.compile(_make_deepseek_component_quant_matmul_probe(), target="metal")
    q8_act, q4_weight, act_scale, weight_scale = _deepseek_component_matmul_synthetic_inputs()
    out = torch.empty((16, 32), dtype=torch.float32, device="mps")

    kernel(q8_act.to("mps"), q4_weight.to("mps"), act_scale.to("mps"), weight_scale.to("mps"), out)
    torch.mps.synchronize()

    ref = _deepseek_component_matmul_ref(q8_act, q4_weight, act_scale, weight_scale)
    assert torch.allclose(out.cpu(), ref, atol=1e-3, rtol=1e-5)


@tilelang.testing.requires_metal
def test_flashqla_gdn_kkt_runtime_mps_matches_torch_reference():
    kernel = tilelang.compile(_make_flashqla_gdn_kkt_probe(), target="metal")
    row_k = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 17.0
    col_k = (torch.arange(64, dtype=torch.float32).reshape(8, 8).flip(1) - 10.0) / 19.0
    scores = torch.empty((8, 8), dtype=torch.float32, device="mps")

    kernel(row_k.to("mps"), col_k.to("mps"), scores)
    torch.mps.synchronize()

    ref = row_k @ col_k.T
    assert torch.allclose(scores.cpu(), ref, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_metal
def test_register_tile_runtime_mps_matches_torch_reference():
    kernel = tilelang.compile(_make_register_tile_probe(), target="metal")
    a = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 13.0
    b = (torch.arange(64, dtype=torch.float32).reshape(8, 8) - 20.0) / 11.0
    c = torch.empty((8, 8), dtype=torch.float32, device="mps")

    kernel(a.to("mps"), b.to("mps"), c)
    torch.mps.synchronize()

    assert torch.allclose(c.cpu(), a @ b, atol=1e-6, rtol=1e-6)


@tilelang.testing.requires_metal
def test_flashqla_gdn_staged_wu_runtime_mps_matches_torch_reference():
    kernel = tilelang.compile(_make_flashqla_gdn_wu_probe(), target="metal")
    a, k, v, beta, g_cum = _flashqla_gdn_wu_synthetic_inputs()
    w = torch.empty((8, 8), dtype=torch.float32, device="mps")
    u = torch.empty((8, 8), dtype=torch.float32, device="mps")

    kernel(a.to("mps"), k.to("mps"), v.to("mps"), beta.to("mps"), g_cum.to("mps"), w, u)
    torch.mps.synchronize()

    ref_w, ref_u = _flashqla_gdn_wu_ref(a, k, v, beta, g_cum)
    assert torch.allclose(w.cpu(), ref_w, atol=1e-5, rtol=1e-5)
    assert torch.allclose(u.cpu(), ref_u, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_metal
def test_flashqla_gdn_component_runtime_mps_matches_torch_reference():
    kernel = tilelang.compile(_make_flashqla_gdn_component_probe(), target="metal")
    k, v, beta, g_cum = _flashqla_gdn_component_synthetic_inputs()
    a_pre = torch.empty((16, 16), dtype=torch.float32, device="mps")
    w = torch.empty((16, 16), dtype=torch.float32, device="mps")
    u = torch.empty((16, 16), dtype=torch.float32, device="mps")

    kernel(k.to("mps"), v.to("mps"), beta.to("mps"), g_cum.to("mps"), a_pre, w, u)
    torch.mps.synchronize()

    ref_a, ref_w, ref_u = _flashqla_gdn_component_ref(k, v, beta, g_cum)
    assert torch.allclose(a_pre.cpu(), ref_a, atol=1e-4, rtol=1e-5)
    assert torch.allclose(w.cpu(), ref_w, atol=1e-4, rtol=1e-5)
    assert torch.allclose(u.cpu(), ref_u, atol=1e-4, rtol=1e-5)


@tilelang.testing.requires_metal
def test_small_synthetic_runtime_benchmarks_opt_in():
    if os.environ.get("TILELANG_RUN_METAL_SMALL_BENCH") != "1":
        pytest.skip("set TILELANG_RUN_METAL_SMALL_BENCH=1 to run small Metal benchmark hooks")

    deepseek_kernel = tilelang.compile(_make_deepseek_packed_quant_probe(), target="metal")
    gdn_kernel = tilelang.compile(_make_flashqla_gdn_kkt_probe(), target="metal")
    q8, q4, e8m0_scale = _deepseek_synthetic_inputs()
    q8_mps, q4_mps, e8m0_mps = q8.to("mps"), q4.to("mps"), e8m0_scale.to("mps")
    decode_out = torch.empty(16, dtype=torch.float32, device="mps")
    # Prepare MPS inputs via CPU expressions; mixing torch MPS arithmetic with
    # TVM's Metal command encoder in the same tiny benchmark can leave an active
    # encoder and trip Metal's command-buffer assertion.
    row_k = (torch.arange(64, dtype=torch.float32).reshape(8, 8) / 17.0).to("mps")
    col_k = ((torch.arange(64, dtype=torch.float32).reshape(8, 8).flip(1) - 10.0) / 19.0).to("mps")
    scores = torch.empty((8, 8), dtype=torch.float32, device="mps")
    torch.mps.synchronize()

    def bench(name, fn, iterations: int = 20):
        # The current TVM/Metal runtime can trip Metal's single command-encoder
        # assertion if tiny kernels are launched back-to-back without a flush.
        # Keep this hook safe/rerunnable by timing synchronized iterations.
        for _ in range(3):
            fn()
            torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
            torch.mps.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / iterations
        print(f"metal_small_bench {name}: {elapsed_ms:.4f} ms/iter over {iterations} iterations")
        assert elapsed_ms >= 0.0

    bench("deepseek_packed_decode_16", lambda: deepseek_kernel(q8_mps, q4_mps, e8m0_mps, decode_out))
    bench("flashqla_gdn_kkt_8x8", lambda: gdn_kernel(row_k, col_k, scores))


@tilelang.testing.requires_metal
def test_scaled_synthetic_runtime_benchmarks_opt_in():
    if os.environ.get("TILELANG_RUN_METAL_SCALED_BENCH") != "1":
        pytest.skip("set TILELANG_RUN_METAL_SCALED_BENCH=1 to run scaled Metal benchmark hooks")

    deepseek_kernel = tilelang.compile(_make_deepseek_packed_quant_matmul_probe(), target="metal")
    gdn_kernel = tilelang.compile(_make_flashqla_gdn_wu_probe(), target="metal")
    q8_act, q4_weight, act_scale, weight_scale = _deepseek_matmul_synthetic_inputs()
    q8_mps = q8_act.to("mps")
    q4_mps = q4_weight.to("mps")
    act_scale_mps = act_scale.to("mps")
    weight_scale_mps = weight_scale.to("mps")
    matmul_out = torch.empty((8, 8), dtype=torch.float32, device="mps")
    a, k, v, beta, g_cum = _flashqla_gdn_wu_synthetic_inputs()
    a_mps, k_mps, v_mps = a.to("mps"), k.to("mps"), v.to("mps")
    beta_mps, g_cum_mps = beta.to("mps"), g_cum.to("mps")
    w = torch.empty((8, 8), dtype=torch.float32, device="mps")
    u = torch.empty((8, 8), dtype=torch.float32, device="mps")
    torch.mps.synchronize()

    def bench(name, fn, iterations: int = 20):
        for _ in range(3):
            fn()
            torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
            torch.mps.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / iterations
        print(f"metal_scaled_bench {name}: {elapsed_ms:.4f} ms/iter over {iterations} iterations")
        assert elapsed_ms >= 0.0

    bench(
        "deepseek_packed_quant_matmul_m8n8k16",
        lambda: deepseek_kernel(q8_mps, q4_mps, act_scale_mps, weight_scale_mps, matmul_out),
    )
    bench("flashqla_gdn_wu_8x8", lambda: gdn_kernel(a_mps, k_mps, v_mps, beta_mps, g_cum_mps, w, u))


@tilelang.testing.requires_metal
def test_component_synthetic_runtime_benchmarks_opt_in():
    if os.environ.get("TILELANG_RUN_METAL_COMPONENT_BENCH") != "1":
        pytest.skip("set TILELANG_RUN_METAL_COMPONENT_BENCH=1 to run component Metal benchmark hooks")

    deepseek_kernel = tilelang.compile(_make_deepseek_component_quant_matmul_probe(), target="metal")
    gdn_kernel = tilelang.compile(_make_flashqla_gdn_component_probe(), target="metal")
    q8_act, q4_weight, act_scale, weight_scale = _deepseek_component_matmul_synthetic_inputs()
    q8_mps = q8_act.to("mps")
    q4_mps = q4_weight.to("mps")
    act_scale_mps = act_scale.to("mps")
    weight_scale_mps = weight_scale.to("mps")
    matmul_out = torch.empty((16, 32), dtype=torch.float32, device="mps")
    k, v, beta, g_cum = _flashqla_gdn_component_synthetic_inputs()
    k_mps, v_mps = k.to("mps"), v.to("mps")
    beta_mps, g_cum_mps = beta.to("mps"), g_cum.to("mps")
    a_pre = torch.empty((16, 16), dtype=torch.float32, device="mps")
    w = torch.empty((16, 16), dtype=torch.float32, device="mps")
    u = torch.empty((16, 16), dtype=torch.float32, device="mps")
    torch.mps.synchronize()

    def bench(name, fn, iterations: int = 10):
        for _ in range(2):
            fn()
            torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
            torch.mps.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / iterations
        print(f"metal_component_bench {name}: {elapsed_ms:.4f} ms/iter over {iterations} iterations")
        assert elapsed_ms >= 0.0

    bench(
        "deepseek_packed_quant_matmul_m16n32k64",
        lambda: deepseek_kernel(q8_mps, q4_mps, act_scale_mps, weight_scale_mps, matmul_out),
    )
    bench("flashqla_gdn_component_chunk16_k16_v16", lambda: gdn_kernel(k_mps, v_mps, beta_mps, g_cum_mps, a_pre, w, u))
