import itertools
import math
from einops import rearrange
import tilelang
from tilelang import language as T
import torch
from tilelang.autotuner import autotune
from tilelang import tvm
from utils import cal_cu_seqlen_ke_for_q, cal_cu_seqlen_ks_for_q

from typing import Tuple


def ceil_to_ue8m0(x: torch.Tensor):
    assert x.view(-1).amax().item() > 0
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))

def per_custom_dims_cast_to_fp8(x: torch.Tensor, dims: Tuple[int], use_ue8m0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()

def print_red_warning(message):
    print(f"\033[31mWARNING: {message}\033[0m")


def calc_sim(x, y, name="tensor"):
    x, y = x.data.double(), y.data.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        print_red_warning(f"{name} all zero")
        return 1
    sim = 2 * (x * y).sum() / denominator
    return sim


def assert_similar(x, y, eps=1e-8, name="tensor", raise_assert=True):
    x_mask = torch.isfinite(x)
    y_mask = torch.isfinite(y)
    if not torch.all(x_mask == y_mask):
        print_red_warning(f"{name} Error: isfinite mask mismatch")
        if raise_assert:
            assert False
    if not torch.isclose(
        x.masked_fill(x_mask, 0),
        y.masked_fill(y_mask, 0),
        rtol=0,
        atol=0,
        equal_nan=True,
    ).all():
        print_red_warning(f"{name} Error: nonfinite value mismatch")
        if raise_assert:
            assert False
    x = x.masked_fill(~x_mask, 0)
    y = y.masked_fill(~y_mask, 0)
    sim = calc_sim(x, y, name)
    diff = 1.0 - sim
    if not (0 <= diff <= eps):
        print_red_warning(f"{name} Error: {diff}")
        if raise_assert:
            assert False
    return diff


def get_configs():
    iter_params = dict(
        block_N=[32, 64, 128],
        num_stages=[0, 1, 2],
        threads=[128, 256],
        block_Q=[1, 2, 4],
    )
    return [
        {k: v for k, v in zip(iter_params, values)}
        for values in itertools.product(*iter_params.values())
    ]


class SupplyProg:
    def __init__(self):
        self.tensors_dict = {}

    def get_key(self, shape, dtype) -> str:
        return f"{shape}-{dtype}"

    def supply_prog(self, params):
        shapes = [p.shape for p in params]
        dtypes = [p.dtype for p in params]
        tensor_list = []
        for shape, dtype in zip(shapes, dtypes):
            key = self.get_key(shape, dtype)
            if key not in self.tensors_dict:
                self.tensors_dict[key] = torch.randn(shape, dtype=dtype, device="cuda")
                tensor_list.append(self.tensors_dict[key])
            else:
                tensor_list.append(self.tensors_dict[key])
        return tensor_list


supply_prog = SupplyProg()


@tilelang.jit
def mqa_attn_return_logits(
    heads,
    index_dim,
    block_N=256,
    num_stages=3,
    threads=512,
    block_Q=None,
):
    if block_Q is None:
        block_Q = 128 // heads
    dtype = "float8_e4m3"
    accum_dtype = "float"
    index_dtype = "int32"

    seq_len = tvm.te.var("seq_len")
    seq_len_kv = tvm.te.var("seq_len_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    index_k_scale_shape = [seq_len_kv]
    logits_shape = [seq_len, seq_len_kv]

    @T.prim_func
    def mqa_attn_return_logits_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        IndexKScale: T.Tensor(index_k_scale_shape, accum_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:

            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            index_k_scale_fragment = T.alloc_fragment([block_N], accum_dtype)
            s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.alloc_fragment([block_N, block_Q, heads], accum_dtype)
            logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_k_s_min = T.alloc_local([1], index_dtype)
            cu_k_e_max = T.alloc_local([1], index_dtype)

            T.no_set_max_nreg()

            cu_k_s_min[0] = 2147483647
            cu_k_e_max[0] = -2147483648

            for bq_i in T.serial(block_Q):
                cu_k_s_min[0] = T.min(
                    cu_k_s_min[0], T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv)
                )
            for bq_i in T.serial(block_Q):
                cu_k_e_max[0] = T.max(
                    cu_k_e_max[0], T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv)
                )

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for nbn_i in T.Pipelined(
                T.ceildiv(cu_k_e_max[0] - cu_k_s_min[0], block_N), num_stages=num_stages
            ):
                T.copy(IndexK[cu_k_s_min[0] + nbn_i * block_N, 0], index_k_shared)
                T.copy(IndexKScale[cu_k_s_min[0] + nbn_i * block_N], index_k_scale_fragment)

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (
                        T.max(s[bn_i, bq_i * heads + h_i], 0) * weights[bq_i, h_i]
                    )  * index_k_scale_fragment[bn_i]

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for bq_i, bn_i in T.Parallel(block_Q, block_N):
                    Logits[seq_len_i + bq_i, cu_k_s_min[0] + nbn_i * block_N + bn_i] = (
                        logits[bn_i, bq_i]
                    )
    return mqa_attn_return_logits_kernel


@tilelang.jit
def clean_logits_(
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len = tvm.te.var("seq_len")
    seq_len_kv = tvm.te.var("seq_len_kv")

    dtype = "float"
    indices_dtype = "int32"

    @T.prim_func
    def clean_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = T.alloc_local([1], indices_dtype)
            cu_k_e = T.alloc_local([1], indices_dtype)
            cu_k_s[0] = CuSeqLenKS[bx]
            cu_k_e[0] = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx < cu_k_s[0] or idx >= cu_k_e[0]:
                        Logits[bx, idx] = -T.infinity(dtype)

    return clean_logits_kernel


def mqa_attn_return_logits_interface(
    q, kv, kv_scales, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True
):
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]

    clean_logits_kernel = clean_logits_()

    mqa_attn_return_logits_kernel = mqa_attn_return_logits(
        heads=heads, index_dim=index_dim
    )
    logits = torch.empty([seq_len, seq_len_kv], device=q.device, dtype=torch.float32)
    mqa_attn_return_logits_kernel(
        q.view(seq_len * heads, index_dim),
        kv,
        kv_scales,
        logits,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    if clean_logits:
        clean_logits_kernel(logits, cu_seqlen_ks, cu_seqlen_ke)
    return logits


def ref_fp8_mqa_logits(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                       cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor):
    k = kv
    q = q.float()
    k = k.float()

    seq_len_kv = kv.shape[0]
    mask_lo = torch.arange(0, seq_len_kv, device='cuda')[None, :] >= cu_seqlen_ks[:, None]
    mask_hi = torch.arange(0, seq_len_kv, device='cuda')[None, :] < cu_seqlen_ke[:, None]
    mask = mask_lo & mask_hi

    score = torch.einsum('mhd,nd->hmn', q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float('-inf'))

    cost = mask.sum()
    return logits, cost

if __name__ == "__main__":
    torch.manual_seed(0)
    S, SKV, H, HKV, D, kv_stride = 4096, 8192, 32, 1, 64, 1
    q = torch.randn(S, H, D, device="cuda", dtype=torch.bfloat16).to(
        torch.bfloat16
    )
    kv = torch.randn(SKV, D, device="cuda", dtype=torch.bfloat16).to(
        torch.bfloat16
    )
    weights = torch.randn(S, H, device="cuda", dtype=torch.float32)
    p = (torch.randn(S, SKV, device="cuda", dtype=torch.float32) * 4).softmax(dim=-1)

    def generate_random_cu_seqlens(
        per_cp_seqlen, cp_size=4, cp_rank=3, kv_stride=1, average_q_len=512
    ):
        total_seqlen = per_cp_seqlen * cp_size

        cu_seqlens = torch.randint(
            0, average_q_len * 2, (total_seqlen // average_q_len * 2,)
        ).cuda()
        last_seq_id = torch.where(cu_seqlens.cumsum(0) >= total_seqlen)[0][0]
        cu_seqlens = cu_seqlens[:last_seq_id]

        if cu_seqlens.sum() < total_seqlen:
            cu_seqlens = torch.cat(
                [cu_seqlens, torch.tensor([total_seqlen - cu_seqlens.sum()]).cuda()]
            )

        total_seqlen_k = (cu_seqlens // kv_stride).sum()

        cu_seqlens_cumsum = torch.cumsum(cu_seqlens, dim=0)
        cu_seqlens_k_cumsum = torch.cumsum(cu_seqlens // kv_stride, dim=0)
        cu_seqlens_qs = torch.cat([torch.tensor([0]).cuda(), cu_seqlens_cumsum[:-1]])
        cu_seqlens_ks = torch.cat([torch.tensor([0]).cuda(), cu_seqlens_k_cumsum[:-1]])
        cu_seqlens_qe = cu_seqlens_cumsum.clone()
        cu_seqlens_ke = cu_seqlens_k_cumsum.clone()

        cu_seqlens_ks_for_each_q = cal_cu_seqlen_ks_for_q(
            cu_seqlens_qs=cu_seqlens_qs,
            cu_seqlens_qe=cu_seqlens_qe,
            cu_seqlens_ks=cu_seqlens_ks,
            seq_len=total_seqlen,
        )
        cu_seqlens_ke_for_each_q = cal_cu_seqlen_ke_for_q(
            cu_seqlens_qs=cu_seqlens_qs,
            cu_seqlens_qe=cu_seqlens_qe,
            cu_seqlens_ks=cu_seqlens_ks,
            cu_seqlens_ke=cu_seqlens_ke,
            q_start_idxs=torch.zeros_like(cu_seqlens_qs),
            seq_len=total_seqlen,
            kv_stride=kv_stride,
        )

        assert per_cp_seqlen % 2 == 0
        per_chunk_seqlen = per_cp_seqlen // 2
        slice_short = slice(
            cp_rank * per_chunk_seqlen, (cp_rank + 1) * per_chunk_seqlen
        )
        slice_long = slice(
            total_seqlen - (cp_rank + 1) * per_chunk_seqlen,
            total_seqlen - cp_rank * per_chunk_seqlen,
        )
        ks = torch.cat(
            [
                cu_seqlens_ks_for_each_q[slice_short],
                cu_seqlens_ks_for_each_q[slice_long],
            ]
        )
        ke = torch.cat(
            [
                cu_seqlens_ke_for_each_q[slice_short],
                cu_seqlens_ke_for_each_q[slice_long],
            ]
        )
        assert len(ks) == len(ke) == per_cp_seqlen
        return ks, ke

    ks, ke = generate_random_cu_seqlens(
        per_cp_seqlen=S, cp_size=4, cp_rank=3, kv_stride=kv_stride, average_q_len=2048
    )

    logits_ref, cost_ref = ref_fp8_mqa_logits(
        q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
    )
    
    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_scales = per_custom_dims_cast_to_fp8(kv, (0, ), False)

    logits_tl = mqa_attn_return_logits_interface(
        q=q_fp8, kv=kv_fp8, kv_scales=kv_scales, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
    )
    diff = assert_similar(
        logits_ref, logits_tl, eps=1e-14, name="logits", raise_assert=False
    )

    original_diff = None
    for i in range(10):
        logits_tl = mqa_attn_return_logits_interface(
            q=q_fp8, kv=kv_fp8, kv_scales=kv_scales, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
        )
        diff = assert_similar(
            logits_ref, logits_tl, eps=1e-14, name="logits", raise_assert=False
        )
        if original_diff is None:
            original_diff = diff
        else:
            assert original_diff == diff

    from tilelang.profiler import do_bench  


    def logits_fn():
        return mqa_attn_return_logits_interface(
            q=q_fp8, kv=kv_fp8, kv_scales=kv_scales, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
        )

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        logits_fn()

    print(
        prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=50)
    )

    logits_ms = do_bench(logits_fn, warmup=100, rep=100)
    logits_flops = 2 * cost_ref * H * D
    logits_tflops = logits_flops / (logits_ms * 1e-3) / 1e12
    print(f"logits_tflops: {logits_tflops}, logits_ms: {logits_ms}")

    print(f"cost_ref: {cost_ref}")

    torch.cuda.profiler.start()
    logits_fn()
    torch.cuda.profiler.stop()
