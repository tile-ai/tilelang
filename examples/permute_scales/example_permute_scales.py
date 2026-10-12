"""TileLang implementation of the MXFP4 MoE scale permutation."""

from __future__ import annotations

import torch

import tilelang
import tilelang.language as T


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}
if hasattr(tilelang.PassConfigKey, "TL_ENABLE_MAGIC_DIV"):
    # Regression drivers run this source with both the old and new installed
    # TileLang. Keep the old baseline runnable when the pass key is absent.
    PASS_CONFIGS[tilelang.PassConfigKey.TL_ENABLE_MAGIC_DIV] = True


@tilelang.jit(pass_configs=PASS_CONFIGS)
def permute_moe_mxfp4_scales(in_scales, out_scales, use_full_perm: int, use_quad_shuffle: int, block: int = 256):
    num_experts = T.dynamic("num_experts")
    size_n = T.dynamic("size_n")
    num_groups = T.dynamic("num_groups")
    in_scales: T.Tensor[(num_experts, size_n, num_groups), T.uint8]
    out_scales: T.Tensor[(num_experts, num_groups, size_n), T.uint8]
    total = num_experts * num_groups * size_n

    with T.Kernel(T.ceildiv(total, block), threads=block) as bx:
        for i in T.Parallel(block):
            idx = bx * block + i
            if idx < total:
                e = idx // (num_groups * size_n)
                f = idx - e * num_groups * size_n
                g = f // size_n
                n = f - g * size_n
                if use_quad_shuffle:
                    q = f % 4
                    v = (f - q) + (q % 2) * 2 + q // 2
                else:
                    v = f
                if use_full_perm:
                    p = v % 64
                    u = (v - p) + (p % 8) * 8 + p // 8
                else:
                    p = v % 32
                    u = (v - p) + 2 * (p // 8) + ((p % 8) // 2) * 8 + (p % 8) % 2
                out_scales[e, g, n] = in_scales[e, u % size_n, u // size_n]


def ref_permute(scales: torch.Tensor, size_n: int, use_full_perm: bool, use_quad_shuffle: bool) -> torch.Tensor:
    num_experts, _, num_groups = scales.shape
    f = torch.arange(num_groups * size_n, device=scales.device)
    v = f
    if use_quad_shuffle:
        q = f % 4
        v = (f - q) + (q % 2) * 2 + q // 2

    tile = 64 if use_full_perm else 32
    p = v % tile
    if use_full_perm:
        u = (v - p) + (p % 8) * 8 + p // 8
    else:
        u = (v - p) + 2 * (p // 8) + ((p % 8) // 2) * 8 + (p % 8) % 2

    src = scales.reshape(num_experts, -1)[:, (u % size_n) * num_groups + u // size_n]
    return src.view(num_experts, num_groups, size_n).contiguous()


def _flags(size_k: int, group_size: int, is_a8: bool) -> tuple[bool, bool]:
    use_full_perm = group_size < size_k and group_size != -1 and not is_a8
    use_quad_shuffle = not is_a8
    return use_full_perm, use_quad_shuffle


def run_regression_perf(
    num_experts: int,
    size_n: int,
    size_k: int,
    group_size: int,
    is_a8: bool = False,
) -> float:
    num_groups = 1 if group_size == -1 else size_k // group_size
    use_full_perm, use_quad_shuffle = _flags(size_k, group_size, is_a8)

    generator = torch.Generator(device="cuda").manual_seed(42)
    scales = torch.randint(
        0,
        256,
        (num_experts, size_n, num_groups),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )

    output = torch.empty((num_experts, num_groups, size_n), dtype=torch.uint8, device="cuda")
    permute_moe_mxfp4_scales(scales, output, int(use_full_perm), int(use_quad_shuffle))
    torch.cuda.synchronize()
    reference = ref_permute(scales, size_n, use_full_perm, use_quad_shuffle)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)

    from tilelang.profiler import do_bench

    return do_bench(
        lambda: permute_moe_mxfp4_scales(scales, output, int(use_full_perm), int(use_quad_shuffle)),
        backend="cupti",
    )


def main() -> None:
    latency = run_regression_perf(
        num_experts=8,
        size_n=4096,
        size_k=8192,
        group_size=128,
    )
    print(f"Latency: {latency:.4f} ms")


if __name__ == "__main__":
    main()
