import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _kernel(K, fragment_a, policy, compact, fragment_scales=False, transpose_a=False):
    @T.prim_func
    def main(
        A: T.Tensor((K, 128) if transpose_a else (128, K), T.float4_e2m1fn),
        B: T.Tensor((128, K), T.float4_e2m1fn),
        SA: T.Tensor((128, K // 64), T.uint32),
        SB: T.Tensor((128, K // 64), T.uint32),
        O: T.Tensor((128, 128), T.float32),
    ):
        with T.Kernel(1, threads=256):
            a = (
                T.alloc_fragment((K, 128) if transpose_a else (128, K), T.float4_e2m1fn)
                if fragment_a
                else T.alloc_shared((128, K), T.float4_e2m1fn)
            )
            b = T.alloc_shared((128, K), T.float4_e2m1fn)
            sa = T.alloc_fragment((128, K // 64), T.uint32) if fragment_scales else T.alloc_shared((128, K // 64), T.uint32)
            sb = T.alloc_fragment((128, K // 64), T.uint32) if fragment_scales else T.alloc_shared((128, K // 64), T.uint32)
            c = T.alloc_fragment((128, 128), T.float32)
            T.copy(A, a)
            T.copy(B, b)
            T.copy(SA, sa)
            T.copy(SB, sb)
            T.gemm_blockscaled(
                a,
                b,
                c,
                sa,
                sb,
                transpose_A=transpose_a,
                transpose_B=True,
                clear_accum=True,
                policy=policy,
                k_start=0,
                sf_a_granularity_k=16,
                sf_b_granularity_k=16,
                sf_layout="blockscaled_chunk_kmajor" if compact else "rowmajor",
            )
            T.copy(c, O)

    return main


def _input(K, compact, transpose=False):
    lookup = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
    nib = torch.randint(0, 16, (128, K), device="cuda", dtype=torch.uint8)
    storage = nib.T.contiguous() if transpose else nib
    packed = (storage[:, 0::2] | (storage[:, 1::2] << 4)).contiguous()
    scale = (2.0 ** torch.randint(-2, 3, (128, K // 16), device="cuda")).to(torch.float8_e4m3fn)
    decoded = lookup[nib.long()] * scale.float().repeat_interleave(16, dim=1)
    words = scale.view(torch.uint32)
    if compact:
        row = torch.arange(128, device="cuda")[:, None]
        kb = torch.arange(K // 64, device="cuda")[None, :]
        index = kb * 128 + row % 32 * 4 + row // 32
        result = torch.empty(words.numel(), device="cuda", dtype=torch.int32)
        result[index.flatten()] = words.view(torch.int32).flatten()
        words = result.view(torch.uint32).reshape(words.shape)
    return packed, words, decoded


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("K", [64, 128, 256])
@pytest.mark.parametrize("policy", [T.GemmWarpPolicy.FullRow, T.GemmWarpPolicy.FullCol, T.GemmWarpPolicy.Square])
@pytest.mark.parametrize(
    "fragment_a, compact, fragment_scales",
    [(True, False, False), (False, True, False), (True, False, True), (False, False, True)],
)
def test_sm120_blockscaled_fragment_and_odd_warps(K, policy, fragment_a, compact, fragment_scales):
    torch.manual_seed(42)
    a, sa, af = _input(K, compact)
    b, sb, bf = _input(K, compact)
    kernel = tilelang.compile(
        _kernel(K, fragment_a, policy, compact, fragment_scales),
        out_idx=[4],
        execution_backend="nvrtc",
        target={"kind": "cuda", "arch": "sm_120a"},
    )
    out = kernel(a, b, sa, sb, stream=torch.cuda.current_stream().cuda_stream)
    torch.testing.assert_close(out, af @ bf.T, atol=0, rtol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("K", [128, 256])
def test_sm120_blockscaled_transposed_fragment_a(K):
    a, sa, af = _input(K, False, transpose=True)
    b, sb, bf = _input(K, False)
    kernel = tilelang.compile(
        _kernel(K, True, T.GemmWarpPolicy.FullRow, False, True, transpose_a=True),
        out_idx=[4],
        execution_backend="nvrtc",
        target={"kind": "cuda", "arch": "sm_120a"},
    )
    out = kernel(a, b, sa, sb, stream=torch.cuda.current_stream().cuda_stream)
    torch.testing.assert_close(out, af @ bf.T, atol=0, rtol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("fragment_a, fragment_scales", [(True, False), (False, True)])
def test_sm120_blockscaled_compact_fragment_diagnostic(fragment_a, fragment_scales):
    target = tilelang.tvm.target.Target({"kind": "cuda", "arch": "sm_120a"})
    program = _kernel(128, fragment_a, T.GemmWarpPolicy.FullRow, True, fragment_scales)
    with target, pytest.raises(Exception, match="currently require.*sf_layout='rowmajor'"):
        tilelang.lower(program, target=target, enable_device_compile=False)
