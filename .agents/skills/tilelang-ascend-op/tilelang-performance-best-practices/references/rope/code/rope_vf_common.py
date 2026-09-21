import tilelang
from tilelang import language as T

STATUS = "EXECUTABLE_BASELINE"


@tilelang.jit(out_idx=-1)
def rope_half_split(
    tokens: int,
    heads: int,
    dim: int,
    position_offset: int = 0,
    num_cores: int | None = None,
):
    """PTO RoPE for the half-split layout [x0..xh, y0..yh]."""

    if min(tokens, heads, dim) <= 0 or dim % 2 or position_offset < 0:
        raise ValueError("tokens/heads/dim must be positive, dim must be even, and position_offset must be non-negative")
    half = dim // 2
    tasks = tokens * heads
    if isinstance(num_cores, bool) or not isinstance(num_cores, int) or num_cores <= 0:
        raise ValueError("num_cores must be a positive integer from target hardware discovery")
    active_cores = min(num_cores, tasks)

    @T.prim_func
    def kernel(
        x: T.Buffer((tokens, heads, dim), "bfloat16"),
        cos: T.Buffer((tokens + position_offset, half), "float32"),
        sin: T.Buffer((tokens + position_offset, half), "float32"),
        out: T.Buffer((tokens, heads, dim), "bfloat16"),
    ):
        with T.Kernel(active_cores) as core_id:
            x_ub = T.alloc_shared((dim,), "bfloat16")
            out_ub = T.alloc_shared((dim,), "bfloat16")
            cos_ub = T.alloc_shared((half,), "float32")
            sin_ub = T.alloc_shared((half,), "float32")

            for task_iter in T.serial((tasks + active_cores - 1) // active_cores):
                task = task_iter * active_cores + core_id
                if task < tasks:
                    token = task // heads
                    head = task % heads
                    T.copy(x[token, head, :], x_ub)
                    T.copy(cos[token + position_offset, :], cos_ub)
                    T.copy(sin[token + position_offset, :], sin_ub)

                    with T.SimtVF(threads=128):
                        for i in T.Parallel(dim):
                            if i < half:
                                out_ub[i] = T.cast(
                                    T.cast(x_ub[i], "float32") * cos_ub[i]
                                    - T.cast(x_ub[i + half], "float32") * sin_ub[i],
                                    "bfloat16",
                                )
                                out_ub[i + half] = T.cast(
                                    T.cast(x_ub[i + half], "float32") * cos_ub[i]
                                    + T.cast(x_ub[i], "float32") * sin_ub[i],
                                    "bfloat16",
                                )

                    T.copy(out_ub, out[token, head, :])

    return kernel


def reference(x, cos, sin, position_offset=0):
    x_fp32 = x.float()
    half = x.shape[-1] // 2
    left, right = x_fp32[..., :half], x_fp32[..., half:]
    c = cos[position_offset : position_offset + x.shape[0]].float().unsqueeze(1)
    s = sin[position_offset : position_offset + x.shape[0]].float().unsqueeze(1)
    import torch

    return torch.cat((left * c - right * s, right * c + left * s), dim=-1).to(x.dtype)


def build(tokens: int, heads: int, dim: int, position_offset: int = 0, num_cores: int | None = None):
    if dim % 2:
        raise ValueError("RoPE head dimension must be even")
    return rope_half_split(tokens, heads, dim, position_offset, num_cores)


__all__ = ["build", "reference", "rope_half_split"]
