# ruff: noqa
import itertools
import tilelang
from tilelang import language as T
import torch
from utils import generate_random_cu_seqlens, per_custom_dims_cast_to_fp8


def display_error_message(msg):
    print(f"\033[31mWARNING: {msg}\033[0m")


def compute_correlation(a, b, label="tensor"):
    a, b = a.data.double(), b.data.double()
    norm_sum = (a * a + b * b).sum()
    if norm_sum == 0:
        display_error_message(f"{label} all zero")
        return 1
    correlation = 2 * (a * b).sum() / norm_sum
    return correlation


def validate_tensor_match(a, b, tolerance=1e-8, tensor_name="tensor", should_raise=True):
    a_finite = torch.isfinite(a)
    b_finite = torch.isfinite(b)
    if not torch.all(a_finite == b_finite):
        display_error_message(f"{tensor_name} Error: isfinite mask mismatch")
        if should_raise:
            assert False
    if not torch.isclose(
        a.masked_fill(a_finite, 0),
        b.masked_fill(b_finite, 0),
        rtol=0,
        atol=0,
        equal_nan=True,
    ).all():
        display_error_message(f"{tensor_name} Error: nonfinite value mismatch")
        if should_raise:
            assert False
    a = a.masked_fill(~a_finite, 0)
    b = b.masked_fill(~b_finite, 0)
    correlation = compute_correlation(a, b, tensor_name)
    difference = 1.0 - correlation
    if not (0 <= difference <= tolerance):
        display_error_message(f"{tensor_name} Error: {difference}")
        if should_raise:
            assert False
    return difference


def get_configs():
    iter_params = dict(
        block_N=[32, 64, 128],
        num_stages=[0, 1, 2],
        threads=[128, 256],
        block_Q=[1, 2, 4],
    )
    return [{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())]


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


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
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
    dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

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
            s_reshaped = T.reshape(s, (block_N, block_Q, heads))
            logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_k_s_min = T.alloc_var(index_dtype)
            cu_k_e_max = T.alloc_var(index_dtype)

            cu_k_s_min = 2147483647
            cu_k_e_max = -2147483648

            for bq_i in T.serial(block_Q):
                cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv))
            for bq_i in T.serial(block_Q):
                cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv))

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for nbn_i in T.Pipelined(T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages):
                T.copy(IndexK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)
                T.copy(IndexKScale[cu_k_s_min + nbn_i * block_N], index_k_scale_fragment)

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]) * index_k_scale_fragment[
                        bn_i
                    ]

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for bq_i, bn_i in T.Parallel(block_Q, block_N):
                    Logits[seq_len_i + bq_i, cu_k_s_min + nbn_i * block_N + bn_i] = logits[bn_i, bq_i]

    return mqa_attn_return_logits_kernel


@tilelang.jit
def clean_logits_(
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def clean_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx < cu_k_s or idx >= cu_k_e:
                        Logits[bx, idx] = -T.infinity(dtype)

    return clean_logits_kernel


def mqa_attn_return_logits_interface(q, kv, kv_scales, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True):
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]

    clean_logits_kernel = clean_logits_()

    mqa_attn_return_logits_kernel = mqa_attn_return_logits(heads=heads, index_dim=index_dim)
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


def ref_fp8_mqa_logits(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor, cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor):
    k = kv
    q = q.float()
    k = k.float()

    seq_len_kv = kv.shape[0]
    mask_lo = torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    mask_hi = torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    cost = mask.sum()
    return logits, cost


def test_fp8_lighting_indexer(S=4096, SKV=8192, H=32, HKV=1, D=64, kv_stride=1):
    # initial random seed to make the performance reproducible
    torch.manual_seed(0)
    q = torch.randn(S, H, D, device="cuda", dtype=torch.bfloat16).to(torch.bfloat16)
    kv = torch.randn(SKV, D, device="cuda", dtype=torch.bfloat16).to(torch.bfloat16)
    weights = torch.randn(S, H, device="cuda", dtype=torch.float32)
    p = (torch.randn(S, SKV, device="cuda", dtype=torch.float32) * 4).softmax(dim=-1)

    ks, ke = generate_random_cu_seqlens(per_cp_seqlen=S, cp_size=4, cp_rank=3, kv_stride=kv_stride, average_q_len=2048)

    logits_ref, cost_ref = ref_fp8_mqa_logits(q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke)

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_scales = per_custom_dims_cast_to_fp8(kv, (0,), False)

    logits_tl = mqa_attn_return_logits_interface(q=q_fp8, kv=kv_fp8, kv_scales=kv_scales, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke)
    diff = validate_tensor_match(logits_ref, logits_tl, tolerance=1e-14, tensor_name="logits", should_raise=False)

    print(f"diff: {diff}")

    from tilelang.profiler import do_bench

    def logits_fn():
        return mqa_attn_return_logits_interface(q=q_fp8, kv=kv_fp8, kv_scales=kv_scales, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        logits_fn()

    print(prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=50))

    logits_ms = do_bench(logits_fn, warmup=100, rep=100)
    logits_flops = 2 * cost_ref * H * D
    logits_tflops = logits_flops / (logits_ms * 1e-3) / 1e12
    print(f"logits_tflops: {logits_tflops}, logits_ms: {logits_ms}")
    print(f"cost_ref: {cost_ref}")


def run_regression_perf(S=4096, SKV=8192, H=32, HKV=1, D=64, kv_stride=1):
    torch.manual_seed(0)
    q = torch.randn(S, H, D, device="cuda", dtype=torch.bfloat16).to(torch.bfloat16)
    kv = torch.randn(SKV, D, device="cuda", dtype=torch.bfloat16).to(torch.bfloat16)
    weights = torch.randn(S, H, device="cuda", dtype=torch.float32)
    p = (torch.randn(S, SKV, device="cuda", dtype=torch.float32) * 4).softmax(dim=-1)

    ks, ke = generate_random_cu_seqlens(per_cp_seqlen=S, cp_size=4, cp_rank=3, kv_stride=kv_stride, average_q_len=2048)

    logits_ref, cost_ref = ref_fp8_mqa_logits(q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke)

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_scales = per_custom_dims_cast_to_fp8(kv, (0,), False)

    logits_tl = mqa_attn_return_logits_interface(q=q_fp8, kv=kv_fp8, kv_scales=kv_scales, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke)
    diff = validate_tensor_match(logits_ref, logits_tl, tolerance=1e-14, tensor_name="logits", should_raise=False)

    from tilelang.profiler import do_bench

    def logits_fn():
        return mqa_attn_return_logits_interface(q=q_fp8, kv=kv_fp8, kv_scales=kv_scales, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        logits_fn()

    print(prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=50))

    return do_bench(logits_fn, warmup=100, rep=100, backend="cupti")


if __name__ == "__main__":
    test_fp8_lighting_indexer()
