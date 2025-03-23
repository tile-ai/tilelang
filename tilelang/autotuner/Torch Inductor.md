# Torch Inductor介绍
**搜集一下大家的表达式是什么样子的？triton实现如果要tune的话需要哪些参数**
什么是torchinductor？ 
怎么加速的？
为什么可以加速？
用triton写了哪些算子？
这些算子在LLM中出现的位置是什么？
在LLM整体计算中的占比是多少？
triton实现带来的收益是在什么地方（效率上的提高还是易用性的提高？）triton实现的kernel有哪些参数对计算效率的影响很大？将这些参数整理出来。

TorchInductor是pytorch新的编译器后端，通过`torch.compile（backend=inductor）`就可以启用。
在inductor的ir中有`TritonTemplateKernel` 这个类用于将triton代码融入到inductor中。
为什么需要triton呢？首先，在inductor中是可以通过`ExternalKernel`这个类直接使用pytorch eager mode中已经优化好的kernel，但这有一个很明显的问题就是，已经优化好的kernel是独立的，每个kernel被设计出来只能做好1件事情。比如gemm后想接relu，使用kernel库会将gemm的结果先写回内存，算relu时再从内存中读取，也就是没办法进行fuse。

对于在inductor中集成triton，可以引用杨军大佬的概述
>Inductor在Triton的集成方面相较JAX/XLA做得会更加全面且务实。Inductor一共包含三种使用Triton的方式：
(1) 针对非计算密集算子，基于Inductor IR，实现了相对通用的[Codegen](https://link.zhihu.com/?target=https%3A//github.com/pytorch/pytorch/blob/ab148da66cb9433effac90c7bd4930a961481d19/torch/_inductor/codegen/triton.py%23L997)的支持。
(2) 针对GEMM，基于Jinja2，通过模式匹配的方式实现了[半定制的codegen](https://link.zhihu.com/?target=https%3A//github.com/pytorch/pytorch/blob/ab148da66cb9433effac90c7bd4930a961481d19/torch/_inductor/kernel/mm.py%23L27)
(3) 针对Conv，调用[pre-baked Triton kernel](https://link.zhihu.com/?target=https%3A//github.com/pytorch/pytorch/blob/ab148da66cb9433effac90c7bd4930a961481d19/torch/_inductor/triton_ops/conv.py%23L14)，没有提供定制能力。
总的来说，我会觉得Inductor的这个集成，在实现复杂度和最终收益的拿捏上做得还是不错的，把资源用到了比较关键的地方，也没有想一下子解决所有hard problem导致整个实施无法推进，是一个让我映象深刻的工作。

`ATen`是pytorch官方手动优化kernel library
# Inductor中triton算子及其parameters
## conv
`conv2d_template`,`conv3d_template`中需要调优的参数（3d多1维）
- `BLOCK_M`,`BLOCK_N`,`BLOCK_K`
- `KERNEL_H`, `KERNEL_W`：卷积核的高度和宽度
- `STRIDE` ：卷积在高度和宽度方向的步长
- `PADDING_H` 和 `PADDING_W`：padding大小
- `GROUPS`：控制分组卷积，如果>1，则表示使用分组卷积。
- `UNROLL`：是否展开该卷积操作。
- `ALLOW_TF32`：是否允许使用TF32
- `num_stages`：控制计算并行度
- `num_warps`：每个阶段使用warp的数量
conv2d:
```python
LOOP_BODY_2D = """
        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)
"""

conv2d_template = TritonTemplate(
    name="convolution2d",
    grid=conv2d_grid,
    source=r"""
{{def_kernel("X", "W")}}
    # Tensor dimensions
    BATCH = {{size("X", 0)}}
    IN_C = {{size("X", 1)}}
    IN_H = {{size("X", 2)}}
    IN_W = {{size("X", 3)}}
    OUT_C = {{size(None, 1)}}
    OUT_H = {{size(None, 2)}}
    OUT_W = {{size(None, 3)}}

    # Strides:
    stride_xn = {{stride("X", 0)}}
    stride_xc = {{stride("X", 1)}}
    stride_xh = {{stride("X", 2)}}
    stride_xw = {{stride("X", 3)}}
    stride_wc_out = {{stride("W", 0)}}
    stride_wc_in = {{stride("W", 1)}}
    stride_wh = {{stride("W", 2)}}
    stride_ww = {{stride("W", 3)}}

    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

{% if GROUPS == 1 %}
    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C
{% else %}
    group = tl.program_id(2)
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS
{% endif %}

    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

{% if UNROLL %}
{% for i in range(KERNEL_H) %}
{% for j in range(KERNEL_W) %}
    i = {{i}}
    j = {{j}}
    for k in range(0, GROUP_IN_C, BLOCK_K):
        """
    + LOOP_BODY_2D
    + """
{% endfor %}
{% endfor %}
{% else %}
    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W
        """
    + LOOP_BODY_2D
    + """
{% endif %}

    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    {{store_output(("idx_n", "idx_c", "idx_h", "idx_w"), "acc", "mask")}}
""",
)
```

conv3d
```python
LOOP_BODY_3D = """
        idx_x_d = d - PADDING_D + idx_y_d * STRIDE_D
        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_d * stride_xd)[:, None]
            + (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_d >= 0)[:, None]
            & (idx_x_d < IN_D)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] +
            (d * stride_wd) + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)
"""

conv3d_template = TritonTemplate(
    name="convolution3d",
    grid=conv3d_grid,
    source=r"""
{{def_kernel("X", "W")}}
    # Tensor dimensions
    BATCH = {{size("X", 0)}}
    IN_C = {{size("X", 1)}}
    IN_D = {{size("X", 2)}}
    IN_H = {{size("X", 3)}}
    IN_W = {{size("X", 4)}}
    OUT_C = {{size(None, 1)}}
    OUT_D = {{size(None, 2)}}
    OUT_H = {{size(None, 3)}}
    OUT_W = {{size(None, 4)}}

    # Strides:
    stride_xn = {{stride("X", 0)}}
    stride_xc = {{stride("X", 1)}}
    stride_xd = {{stride("X", 2)}}
    stride_xh = {{stride("X", 3)}}
    stride_xw = {{stride("X", 4)}}
    stride_wc_out = {{stride("W", 0)}}
    stride_wc_in = {{stride("W", 1)}}
    stride_wd = {{stride("W", 2)}}
    stride_wh = {{stride("W", 3)}}
    stride_ww = {{stride("W", 4)}}

    ndhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = ndhw % OUT_W
    ndh = ndhw // OUT_W
    idx_y_h = ndh % OUT_H
    nd = ndh // OUT_H
    idx_y_d = nd % OUT_D
    idx_n = nd // OUT_D
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

{% if GROUPS == 1 %}
    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C
{% else %}
    group = tl.program_id(2)
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS
{% endif %}

    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

{% if UNROLL %}
{% for d in range(KERNEL_D) %}
{% for i in range(KERNEL_H) %}
{% for j in range(KERNEL_W) %}
    d = {{d}}
    i = {{i}}
    j = {{j}}
    for k in range(0, GROUP_IN_C, BLOCK_K):
        """
    + LOOP_BODY_3D
    + """
{% endfor %}
{% endfor %}
{% endfor %}
{% else %}
    # Could be simplified, but slightly slower:
    # for d in range(KERNEL_D):
    #   for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for dijk in range(KERNEL_D * KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (dijk % BLOCK_K_COUNT) * BLOCK_K
        dij = dijk // BLOCK_K_COUNT
        j = dij % KERNEL_W
        di = dij // KERNEL_W
        i = di % KERNEL_H
        d = di // KERNEL_H
        """
    + LOOP_BODY_3D
    + """
{% endif %}

    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_d < OUT_D)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_d = idx_y_d[:, None]
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    {{store_output(("idx_n", "idx_c", "idx_d", "idx_h", "idx_w"), "acc", "mask")}}
""",
)
```

## mm_plus_mm
- `BLOCK_M`,`BLOCK_N`,`BLOCK_K`
- `GROUP_M`：矩阵分块的分组数
- `stride`（`stride_am`, `stride_ak`, `stride_bk`, `stride_bn`, `stride_cm`, `stride_ck`, `stride_dk`, `stride_dn`）：在每个矩阵不同维度上的步长
- `ALLOW_TF32`
- `EVEN_K`：控制是否在计算中使用偶数K。如果为True，则会加载完整的K维数据，如果为False，则使用mask来加载部分数据
- `num_stages`
- `num_warps`

```python
mm_plus_mm_template = TritonTemplate(
    name="mm_plus_mm",
    grid=mm_grid,
    debug=False,
    source=r"""
{{def_kernel("A", "B", "C", "D")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K1 = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    # K2 = {{size("C", 1)}}
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}
    stride_cm = {{stride("C", 0)}}
    stride_ck = {{stride("C", 1)}}
    stride_dk = {{stride("D", 0)}}
    stride_dn = {{stride("D", 1)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    if (((stride_am == 1 and stride_ak == M) or (stride_am == K1 and stride_ak == 1))
        and ((stride_cm == 1 and stride_ck == M) or (stride_cm == K1 and stride_ck == 1))):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M

    if (((stride_bk == 1 and stride_bn == K1) or (stride_bk == N and stride_bn == 1))
        and ((stride_dk == 1 and stride_dn == K1) or (stride_dk == N and stride_dn == 1))):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    C = C + (ram[:, None] * stride_cm + rk[None, :] * stride_ck)
    D = D + (rk[:, None] * stride_dk + rbn[None, :] * stride_dn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k1 in range(K1, 0, -BLOCK_K):
        # First matmul with A @ B
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k1, other=0.)
            b = tl.load(B, mask=rk[:, None] < k1, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    for k2 in range(K1, 0, -BLOCK_K):

        # Second matmul with C @ D
        if EVEN_K:
            c = tl.load(C)
            d = tl.load(D)
        else:
            c = tl.load(C, mask=rk[None, :] < k2, other=0.)
            d = tl.load(D, mask=rk[:, None] < k2, other=0.)
        acc += tl.dot(c, d, allow_tf32=ALLOW_TF32)
        C += BLOCK_K * stride_ck
        D += BLOCK_K * stride_dk


    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
)
```

## bmm
- `BLOCK_M`,`BLOCK_N`,`BLOCK_K`
- `GROUP_M`
- `stride_am, stride_ak, stride_bk, stride_bn`
- `EVEN_K`
- `iqx_q`：用于处理批处理纬度
- `ALLOW_TF32`
- `num_stages`
- `num_warps`

```python
bmm_template = TritonTemplate(
    name="bmm",
    grid=bmm_grid,
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", -2)}}
    N = {{size("B", -1)}}
    K = {{size("A", -1)}}

    stride_aq = {{stride("A", 0)}}
    stride_am = {{stride("A", 1)}}
    stride_ak = {{stride("A", 2)}}

    stride_bq = {{stride("B", 0)}}
    stride_bk = {{stride("B", 1)}}
    stride_bn = {{stride("B", 2)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_q", "idx_m", "idx_n"), "acc", "mask")}}
""",
)
```

## mm_scaled
**scaled_mm_device_tma_template**
- `BLOCK_M`,`BLOCK_N`,`BLOCK_K`
- `GROUP_M`
- `SCALING_ROWWISE` ：控制是否使用行级别的缩放（row-wise scaling）。如果设置为 True，则每一行的矩阵会应用不同的缩放因子；否则使用标量缩放
- `TMA_SIZE`,`TMA_DESCRIPTOR_SIZE`：
- `NUM_SMS`
- `USE_FAST_ACCUM`：控制是否使用快速累加方式
- `ACC_TYPE`
- `B_PROLOGUE_CAST_TYPE`：控制矩阵 B 在计算前是否进行类型转换，影响数据转换的效率。
- `scale_a 和 scale_b`
- `workspace`：控制分配给TMA操作的内存空间，用于存储中间的计算结果
```python
load_scales = r"""
@triton.jit
def load_scales(a_scale_ptr, b_scale_ptr, SCALING_ROWWISE: tl.constexpr):
    if SCALING_ROWWISE:
        # For row-wise scaling, we'll return the pointers
        return a_scale_ptr, b_scale_ptr
    else:
        # For per-tensor scaling, we'll load the scalar values
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr)
        return a_scale, b_scale
"""


apply_scaling = r"""
@triton.jit
def apply_scaling(
    accumulator,
    a_scale,
    b_scale,
    SCALING_ROWWISE: tl.constexpr,
    offs_cm,
    offs_cn,
    M,
    N,
    stride_a_scale_m,
    stride_b_scale_n,
):
    if SCALING_ROWWISE:
        # For row-wise scaling, we need to load the scales for each row/column
        a_scales = tl.load(
            a_scale + (offs_cm * stride_a_scale_m),
            mask=offs_cm < M,
            other=0.0,
        )
        b_scales = tl.load(
            b_scale + (offs_cn * stride_b_scale_n),
            mask=offs_cn < N,
            other=0.0,
        )
        acc_scale = a_scales[:, None] * b_scales[None, :]
    else:
        # For per-tensor scaling, we can directly use the loaded scalar values
        acc_scale = a_scale * b_scale

    return accumulator * acc_scale
"""


device_tma = r"""
{{def_kernel("A", "B", "A_inverse_scale", "B_inverse_scale")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return

    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}

    if SCALING_ROWWISE:
        stride_a_scale_m = 1
        stride_b_scale_n = 1
    else:
        stride_a_scale_m = 0
        stride_b_scale_n = 0

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    workspace_base = ws_ptr + start_pid * 2 * TMA_SIZE
    a_desc_ptr = workspace_base
    b_desc_ptr = workspace_base + TMA_SIZE

    triton.language.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=a_desc_ptr,
        global_address=A,
        load_size=[BLOCK_M, BLOCK_K],
        global_size=[M, K],
        element_ty=A.dtype.element_ty,
    )
    triton.language.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=b_desc_ptr,
        global_address=B,
        load_size=[BLOCK_N, BLOCK_K],
        global_size=[N, K],
        element_ty=B.dtype.element_ty,
    )

    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(a_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(b_desc_ptr)

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_M * num_pid_n
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    a_scale, b_scale = load_scales(A_inverse_scale, B_inverse_scale, SCALING_ROWWISE)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N

        offs_k = ki * BLOCK_K

        a = tl._experimental_descriptor_load(
            a_desc_ptr, [offs_am, offs_k], [BLOCK_M, BLOCK_K],  A.dtype.element_ty
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr, [offs_bn, offs_k], [BLOCK_N, BLOCK_K],  B.dtype.element_ty
        )
        if USE_FAST_ACCUM:
            accumulator = tl.dot(a, b.T, accumulator)
        else:
            accumulator += tl.dot(a, b.T)

        if ki == k_tiles - 1:
            # Apply inverse scaling
            offs_cm = offs_am + tl.arange(0, BLOCK_M)
            offs_cn = offs_bn + tl.arange(0, BLOCK_N)
            # Apply scaling
            accumulator = apply_scaling(
                accumulator,
                a_scale,
                b_scale,
                SCALING_ROWWISE,
                offs_cm,
                offs_cn,
                M,
                N,
                stride_a_scale_m,
                stride_b_scale_n,
            )

            idx_m = offs_cm[:, None]
            idx_n = offs_cn[None, :]
            mask = (idx_m < M) & (idx_n < N)
            # inductor generates a suffix
            {{store_output(("idx_m", "idx_n"), "accumulator", "mask", indent_width=12)}}
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
"""


scaled_mm_device_tma_template = TritonTemplate(
    name="scaled_mm_device_tma",
    grid=persistent_mm_grid,
    source=device_tma + load_scales + apply_scaling,
)
```

**scaled_mm_template**
- `BLOCK_M`,`BLOCK_N`,`BLOCK_K`
- `GROUP_M`，`GROUP_SIZE`
- `SCALING_ROWWISE`
- `EVEN_K`
- `USE_FAST_ACCUM`：
- `B_PROLOGUE_CAST_TYPE`
- `ACC_TYPE`
```python
scaled_mm_template = TritonTemplate(
    name="scaled_mm",
    grid=mm_grid,
    source=r"""
{{def_kernel("A", "B", "A_inverse_scale", "B_inverse_scale")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        if USE_FAST_ACCUM:
            acc = tl.dot(a, b, acc, out_dtype=ACC_TYPE)
        else:
            acc += tl.dot(a, b, out_dtype=ACC_TYPE)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    if SCALING_ROWWISE:
        inv_a_scale_row = tl.load(A_inverse_scale + rm, mask=rm < M)
        inv_b_scale_row = tl.load(B_inverse_scale + rn, mask=rn < N)
        inv_scale_row = inv_a_scale_row[:, None] * inv_b_scale_row[None, :]
        acc *= inv_scale_row
    else:
        # for tensor-wise scaling, the scales are scalars
        inv_a_scale = tl.load(A_inverse_scale)
        inv_b_scale = tl.load(B_inverse_scale)
        inv_scale = inv_a_scale * inv_b_scale
        acc *= inv_scale

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
)
```

**scaled_mm_bias_template**
```python
scaled_mm_bias_template = TritonTemplate(
    name="scaled_mm_bias",
    grid=mm_grid,
    source=r"""
{{def_kernel("A", "B", "A_inverse_scale", "B_inverse_scale", "bias_ptr")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        if USE_FAST_ACCUM:
            acc = tl.dot(a, b, acc, out_dtype=ACC_TYPE)
        else:
            acc += tl.dot(a, b, out_dtype=ACC_TYPE)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    if SCALING_ROWWISE:
        inv_a_scale_row = tl.load(A_inverse_scale + rm, mask=rm < M)
        inv_b_scale_row = tl.load(B_inverse_scale + rn, mask=rn < N)
        inv_scale_row = inv_a_scale_row[:, None] * inv_b_scale_row[None, :]
        acc *= inv_scale_row
    else:
        # for tensor-wise scaling, the scales are scalars
        inv_a_scale = tl.load(A_inverse_scale)
        inv_b_scale = tl.load(B_inverse_scale)
        inv_scale = inv_a_scale * inv_b_scale
        acc *= inv_scale

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # bias
    bias = tl.load(bias_ptr + rn, mask=rn < N)
    acc += bias

    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
)
```

## unpack_mixed_mm.py
`uint4x2_mixed_mm_template`
- `BLOCK_M`,`BLOCK_N`,`BLOCK_K`
- `GROUP_M`
- `EVEN_K`
- `B_PROLOGUE_CAST_TYPE`
- `b_shifts`：用于处理矩阵 B 数据的位移操作，在对 B 进行位移时，控制每个元素的移位量
- `b_subs`：控制在对 B 进行位移操作后的元素偏移，用于优化数据访问和计算
- `stride("A", 0)`, `stride("A", 1)`, `stride("B", 0)`, `stride("B", 1)`
```python
uint4x2_mixed_mm_template = TritonTemplate(
    name="uint4x2_mixed_mm",
    grid=mm_grid,
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None]//2 * stride_bk + rbn[None, :] * stride_bn)
    b_shifts = 4*(rk%2)
    b_subs = 8*(1-(rk%2))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        b = ((b >> b_shifts[:, None]) & 0xF) - 8
        b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K//2 * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
)
```

## b2b gemm
在`/_inductor/fx_passes`中藏了一个`b2b_gemm`的triton template
b2b gemm计算`A @ B @ C`，根据乘法结合律有两个template
`b2b_gemm_left_template`
- `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_O`, `BLOCK_SIZE_P`
- `num_stages`
- `num_warps`
- `EVEN_K`
- `ACC_TYPE`
- `B_PROLOGUE_CAST_TYPE`：B矩阵预处理类型转换

**b2b_gemm_left_template**
```python
b2b_gemm_left_template = TritonTemplate(
    name="b2b_gemm_left",
    grid=b2b_gemm_grid,
    debug=False,
    source=r"""
{{def_kernel("A", "B", "C")}}


    # B2B_GEMM_LEFT_TRITON_ENTRANCE

    # dynamic shapes
    M = {{size("A", 0)}}
    N = {{size("A", 1)}}
    O = {{size("C", 0)}}
    P = {{size("C", 1)}}

    # dynamic strides
    stride_am = {{stride("A", 0)}}
    stride_an = {{stride("A", 1)}}
    stride_bn = {{stride("B", 0)}}
    stride_bo = {{stride("B", 1)}}
    stride_co = {{stride("C", 0)}}
    stride_cp = {{stride("C", 1)}}

    # output block counts
    num_m_block = tl.cdiv(M, BLOCK_SIZE_M)
    num_p_block = tl.cdiv(P, BLOCK_SIZE_P)

    # internal block counts
    num_n_block = tl.cdiv(N, BLOCK_SIZE_N)
    num_o_block = tl.cdiv(O, BLOCK_SIZE_O)

    # output block ids
    pid = tl.program_id(axis=0)
    m_block_id = pid // num_p_block
    p_block_id = pid % num_p_block

    # accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_P), dtype=tl.float32)

    # main loop
    offs_m = (m_block_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_p = (p_block_id * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P))
    # (subgraph(A @ B) @ C)
    offs_o = tl.arange(0, BLOCK_SIZE_O)
    for _ in range(num_o_block):
        c_mask = (offs_o[:, None] < O) & (offs_p[None, :] < P)
        c_ptrs = C + (offs_o[:, None] * stride_co + offs_p[None, :] * stride_cp)
        c = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_O * BLOCK_SIZE_P
        acc_ab = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_O), dtype=tl.float32)
        offs_n = tl.arange(0, BLOCK_SIZE_N)
        for __ in range(num_n_block):
            a_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            a_ptrs = A + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_M * BLOCK_SIZE_N
            b_mask = (offs_n[:, None] < N) & (offs_o[None, :] < O)
            b_ptrs = B + (offs_n[:, None] * stride_bn + offs_o[None, :] * stride_bo)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_N * BLOCK_SIZE_O
            acc_ab += tl.dot(a, b, out_dtype=tl.float32)
            offs_n += BLOCK_SIZE_N
        # apply the subgraph
        {{ modification(
            subgraph_number=0,
            output_name="post_subgraph_acc_ab",
            inner_mm="acc_ab"
        ) | indent_except_first(2) }}
        acc += tl.dot(post_subgraph_acc_ab, c, out_dtype=tl.float32)
        offs_o += BLOCK_SIZE_O

    # type conversion
    acc = acc.to(tl.float16)

    # store preparation
    idx_m = offs_m[:, None]
    idx_p = offs_p[None, :]
    out_mask = (idx_m < M) & (idx_p < P)

    {{store_output(("idx_m", "idx_p"), "acc", "out_mask")}}
""",
)
```

**b2b_gemm_right_template**
```python
b2b_gemm_right_template = TritonTemplate(
    name="b2b_gemm_right",
    grid=b2b_gemm_grid,
    debug=False,
    source=r"""
{{def_kernel("A", "B", "C")}}


    # B2B_GEMM_RIGHT_TRITON_ENTRANCE

    # dynamic shapes
    M = {{size("A", 0)}}
    N = {{size("A", 1)}}
    O = {{size("C", 0)}}
    P = {{size("C", 1)}}

    # dynamic strides
    stride_am = {{stride("A", 0)}}
    stride_an = {{stride("A", 1)}}
    stride_bn = {{stride("B", 0)}}
    stride_bo = {{stride("B", 1)}}
    stride_co = {{stride("C", 0)}}
    stride_cp = {{stride("C", 1)}}

    # output block counts
    num_m_block = tl.cdiv(M, BLOCK_SIZE_M)
    num_p_block = tl.cdiv(P, BLOCK_SIZE_P)

    # internal block counts
    num_n_block = tl.cdiv(N, BLOCK_SIZE_N)
    num_o_block = tl.cdiv(O, BLOCK_SIZE_O)

    # output block ids
    pid = tl.program_id(axis=0)
    m_block_id = pid // num_p_block
    p_block_id = pid % num_p_block

    # accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_P), dtype=tl.float32)

    # main loop (two cases)
    offs_m = (m_block_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_p = (p_block_id * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P))
    # (A @ subgraph(B @ C))
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    for _ in range(num_n_block):
        a_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        a_ptrs = A + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_M * BLOCK_SIZE_N
        acc_bc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_P), dtype=tl.float32)
        offs_o = tl.arange(0, BLOCK_SIZE_O)
        for __ in range(num_o_block):
            b_mask = (offs_n[:, None] < N) & (offs_o[None, :] < O)
            b_ptrs = B + (offs_n[:, None] * stride_bn + offs_o[None, :] * stride_bo)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_N * BLOCK_SIZE_O
            c_mask = (offs_o[:, None] < O) & (offs_p[None, :] < P)
            c_ptrs = C + (offs_o[:, None] * stride_co + offs_p[None, :] * stride_cp)
            c = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_O * BLOCK_SIZE_P
            acc_bc += tl.dot(b, c, out_dtype=tl.float32)
            offs_o += BLOCK_SIZE_O
        # apply the subgraph
        {{ modification(
            subgraph_number=0,
            output_name="post_subgraph_acc_bc",
            inner_mm="acc_bc"
        ) | indent_except_first(2) }}
        acc += tl.dot(a, post_subgraph_acc_bc, out_dtype=tl.float32)
        offs_n += BLOCK_SIZE_N

    # type conversion
    acc = acc.to(tl.float16)

    # store preparation
    idx_m = offs_m[:, None]
    idx_p = offs_p[None, :]
    out_mask = (idx_m < M) & (idx_p < P)

    {{store_output(("idx_m", "idx_p"), "acc", "out_mask")}}
""",
)

```

## flex attention
- `BLOCK_M`,`BLOCK_N`,`BLOCK_K`
- `num_warps`
- `num_stages`
- `SPARSE_Q_BLOCK_SIZE`,`SPARSE_KV_BLOCK_SIZE`：控制Q、KV块的大小
- `SM_SCALE`：控制QK的预缩放因子
- `IS_DIVISIBLE`：表示输入Q和KV的序列长度能否被`BLOCK_M`和`BLOCK_N`整除，影响tile
- `GQA_SHARED_HEADS`：在GQA模型下，每个KV head共享多少个query head
- `HAS_FULL_BLOCKS`：是否启用全块的处理方法。
- `PRESCALE_QK`：是否对QK进行预缩放
- `KV_NUM_BLKS`：每个query对应多少个KV blocks
- `KV_IDX`：每个query的索引
- `FULL_KV_NUM_BLKS`：每个query没有被mask的kv blocks num
- `FULL_KV_IDX`：每个query没有被mask的kv block索引
```python
# Inner Triton functions shared by flex_attention & split-k decoding kernels.
compute_next_offset_func = r"""
@triton.jit
def get_offset_for_next_block(
    loop_iter, col_indices, total_blocks,
    SPARSE_BLOCK, SPARSE_BLOCK_MULTIPLE, BLOCK,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr
):
    if BLOCKS_ARE_CONTIGUOUS:
        return BLOCK
    cur_block_idx = loop_iter // SPARSE_BLOCK_MULTIPLE
    cur_block = tl.load(col_indices + cur_block_idx, eviction_policy="evict_last")
    next_block = tl.load(col_indices + cur_block_idx + 1, eviction_policy="evict_last", mask=cur_block_idx + 1 < total_blocks)
    needs_jump = (loop_iter + 1) % SPARSE_BLOCK_MULTIPLE == 0
    jump_to_block = (next_block - cur_block ) * SPARSE_BLOCK - (SPARSE_BLOCK_MULTIPLE - 1) * BLOCK
    offset = jump_to_block * needs_jump + (1 - needs_jump) * BLOCK
    return offset
"""

get_bounded_indices_func = r"""
@triton.jit
def get_bounded_indices(indices, max_len=None):
    return indices % max_len if max_len is not None else indices
"""


load_checked_block = r"""
@triton.jit
def load_checked_block(block_ptr, IS_DIVISIBLE: tl.constexpr, SAFE_HEAD_DIM: tl.constexpr):
  if IS_DIVISIBLE and SAFE_HEAD_DIM:
    return tl.load(block_ptr)
  elif IS_DIVISIBLE and not SAFE_HEAD_DIM:
    return tl.load(block_ptr, boundary_check=(1,), padding_option="zero")
  elif not IS_DIVISIBLE and SAFE_HEAD_DIM:
      return tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
  else:
      return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")
"""

load_checked_2d = r"""
@triton.jit
def load_checked_2d(
    ptr,
    offs_m,
    offs_n,
    stride_m,
    stride_n,
    IS_DIVISIBLE_M: tl.constexpr,
    IS_DIVISIBLE_N: tl.constexpr,
    M_LEN: tl.constexpr,
    N_DIM: tl.constexpr,
):
    # Calculate final pointer if strides are provided
    if stride_m is not None and stride_n is not None:
        ptr = ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

    # Handle all masking cases
    if not IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN) & (offs_n[None, :] < N_DIM), other=0.0)
    elif IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_n[None, :] < N_DIM), other=0.0)
    elif not IS_DIVISIBLE_M and IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN), other=0.0)
    else:  # Both divisible
        return tl.load(ptr)
"""

compute_flex_attention = r"""
{{def_kernel("Q", "K", "V", "LSE", "KV_NUM_BLKS", "KV_IDX", "FULL_KV_NUM_BLKS", "FULL_KV_IDX")}}
    # Sub notation for this kernel:
    #
    # Q: Query, K: Key, V: Value
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    #
    # The following FULL_* and PARTIAL_* is defined in the block sparse mask grid, rather than the thread block grid.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    # FULL_KV_NUM_BLKS: The number of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_KV_IDX: The indices of fully unmasked KV blocks (so we don't need masking) for each query.
    #
    # OUTPUT_LOGSUMEXP: We only need to store the logsumexp if we require grad
    #
    # (Modifiable) Performance tuning options
    # BLOCK_M: The thread block size across the seqlen dim of Q.
    # BLOCK_N: Iterate over BLOCK_N across the seqlen dim of K/V in each thread block.

    # The below are kernel options that can be applied for certain score_mods,
    # or involve a numerics vs. perf tradeoff
    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d) and change of base. Has
    # about 20% more numerical error, but slightly faster.
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check
    # BLOCKS_ARE_CONTIGUOUS: Is it guaranteed that all blocks in the mask are
    # contiguous? If so, we don't need to do an indirect jump for every block

    tl.static_assert(SPARSE_Q_BLOCK_SIZE >= BLOCK_M and SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0)
    tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)

    # Define strides of inputs
    stride_qz, stride_qh, stride_qm, stride_qk = {{stride("Q")}}
    stride_kz, stride_kh, stride_kn, stride_kk = {{stride("K")}}
    stride_vz, stride_vh, stride_vn, stride_vk = {{stride("V")}}

    ZQ = {{size("Q", 0)}}
    HQ = {{size("Q", 1)}}
    Q_LEN = {{size("Q", 2)}}
    ZKV = {{size("K", 0)}}
    KV_LEN = {{size("K", 2)}}

    MATMUL_PRECISION = Q.dtype.element_ty

    q_start = tl.program_id(0)
    off_zq = tl.program_id(1) // HQ
    off_hq = tl.program_id(1) % HQ

    # We support two cases for batch dimension. a) (ZKV == ZQ) where off_zkv = off_zq.
    # b) (ZKV == 1 and ZQ > 1) where KV is broadcasted along the batch dimension and off_zkv=0.
    off_zkv = off_zq % ZKV
    off_hkv = off_hq // GQA_SHARED_HEADS
    off_g = off_hq % GQA_SHARED_HEADS

    q_offset = off_zq * stride_qz + off_hq * stride_qh
    k_offset = off_zkv * stride_kz + off_hkv * stride_kh
    v_offset = off_zkv * stride_vz + off_hkv * stride_vh

    Q = Q + q_offset
    K = K + k_offset
    V = V + v_offset

    SPARSE_Z = {{size("KV_NUM_BLKS", 0)}}
    SPARSE_HQ = {{size("KV_NUM_BLKS", 1)}}

    sparse_idx_z = off_zq % SPARSE_Z
    sparse_idx_hq = off_hq % SPARSE_HQ

    SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M)
    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)

    stride_kv_num_blks_h = {{stride("KV_NUM_BLKS", 1)}}
    stride_kv_idx_h = {{stride("KV_IDX", 1)}}
    stride_kv_idx_m = {{stride("KV_IDX", 2)}}

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, V_HEAD_DIM_ROUNDED], dtype=tl.float32)

    offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)

    # KV_IDX and KV_NUM_BLKS are always contiguous.
    sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq
    sparse_kv_num_blks_offset = sparse_hz_offset * stride_kv_num_blks_h + q_start // SPARSE_Q_MULTIPLE
    sparse_kv_idx_offset = sparse_hz_offset * stride_kv_idx_h + (q_start // SPARSE_Q_MULTIPLE) * stride_kv_idx_m  # noqa: B950

    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(Q_LEN, QK_HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(q_start * BLOCK_M, 0),
        block_shape=(BLOCK_M, QK_HEAD_DIM_ROUNDED),
        order=(1, 0)
    )
    q = load_checked_block(Q_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)
    # ~~~~~~~~~~~~~~ normal blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We don't know anything "special" about these blocks, so we need to apply
    # both score_mod and mask_mod to it
    kv_indices = KV_IDX + sparse_kv_idx_offset
    kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
    kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
    block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))

    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(QK_HEAD_DIM, KV_LEN),
        strides=(stride_kk, stride_kn),
        offsets=(0, kv_start),
        block_shape=(QK_HEAD_DIM_ROUNDED, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(KV_LEN, V_HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(kv_start, 0),
        block_shape=(BLOCK_N, V_HEAD_DIM_ROUNDED),
        order=(1, 0)
    )
    offs_n = kv_start + tl.arange(0, BLOCK_N)

    acc, l_i, m_i = forward_inner(
        {{gen_argdefs()}},
        q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
        acc, l_i, m_i,
        off_zq, off_hq, offs_m[:, None], offs_n[None, :],
        kv_indices, kv_num_blocks,
        0, block_n_end,
        MATMUL_PRECISION,
        IS_FULL_BLOCKS=False,
    )

    # ~~~~~~~~~~~~~~ "full" blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We know these blocks are guaranteed to be "full", so we don't need to
    # apply mask_mod to them - only score_mod
    if HAS_FULL_BLOCKS:
        # FULL_KV_IDX and FULL_KV_NUM_BLKS are always contiguous.
        kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
        kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
        kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
        block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))

        K_block_ptr = tl.make_block_ptr(
            base=K,
            shape=(QK_HEAD_DIM, KV_LEN),
            strides=(stride_kk, stride_kn),
            offsets=(0, kv_start),
            block_shape=(QK_HEAD_DIM_ROUNDED, BLOCK_N),
            order=(0, 1)
        )
        V_block_ptr = tl.make_block_ptr(
            base=V,
            shape=(KV_LEN, V_HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(kv_start, 0),
            block_shape=(BLOCK_N, V_HEAD_DIM_ROUNDED),
            order=(1, 0)
        )
        offs_n = kv_start + tl.arange(0, BLOCK_N)

        acc, l_i, m_i = forward_inner(
            {{gen_argdefs()}},
            q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
            acc, l_i, m_i,
            off_zq, off_hq, offs_m[:, None], offs_n[None, :],
            kv_indices, kv_num_blocks,
            0, block_n_end,
            MATMUL_PRECISION,
            IS_FULL_BLOCKS=True,
        )


    # [Note] Handle fully masked out rows:
    # Li will be the sum(e^(-inf)) == 0.0 for masked out rows, mi will be -inf.
    # We set Li to 1.0 which will result in lse/out = 0.0 | after the log(li) + mi(0.0) step
    l_i = tl.where(l_i == 0.0, 1, l_i)

    acc = acc / l_i[:, None]
    idx_zq = tl.program_id(1) // HQ
    idx_hq = tl.program_id(1) % HQ
    idx_m = offs_m[:, None]
    idx_d = tl.arange(0, V_HEAD_DIM_ROUNDED)[None, :]

    mask = (idx_m < Q_LEN) & (idx_d < V_HEAD_DIM)

    {{store_output(("idx_zq", "idx_hq", "idx_m", "idx_d"), "acc", "mask")}}

    if OUTPUT_LOGSUMEXP:
        off_hz = tl.program_id(1)
        l_ptrs = LSE + off_hz * Q_LEN + offs_m
        lse = m_i + tl.math.log2(l_i)
        if IS_DIVISIBLE:
            tl.store(l_ptrs, lse)
        else:
            tl.store(l_ptrs, lse, mask=offs_m < Q_LEN)
 """


compute_forward_inner = r"""
@triton.jit
def forward_inner(
    {{gen_argdefs()}},
    q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets used as inputs to score_mod & mask_mod
    # of size [BLOCK_M, BLOCK_N] or scalar.
    off_z, off_h, offs_m, offs_n,
    # blocksparse data
    kv_indices, kv_num_blocks,
    # start kv and end kv block
    block_n_start, block_n_end,
    MATMUL_PRECISION,
    IS_FULL_BLOCKS,
):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
    {{gen_defines() | indent_except_first(1)}}

    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
    RCP_LN2: tl.constexpr = 1.44269504

    if PRESCALE_QK:
        q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

    # loop over k, v and update accumulator until block_n_end
    for start_n in range(block_n_start, block_n_end):
        if IS_DIVISIBLE:
            acc, l_i, m_i = forward_block_mn(
                {{gen_argdefs()}},
                q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS,
            )
        else:
            # Benchmark shows even we applied mod & mask to each block for non divisible seqlen,
            # it's on par or slightly faster than only applying to the last block in fwd.
            # However, we choose different strategy for bwd, where we only apply mod & mask
            # to the last block because it's faster a lot.
            acc, l_i, m_i = forward_block_mn(
                {{gen_argdefs()}},
                q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )

        # update pointers
        offset = get_offset_for_next_block(
            start_n, kv_indices, kv_num_blocks,
            SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N, BLOCKS_ARE_CONTIGUOUS
        )

        V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, offset))

        offs_n = offs_n + offset

    return acc, l_i, m_i

"""


compute_forward_block_mn = r"""
@triton.jit
def forward_block_mn(
    {{gen_argdefs()}},
    q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets
    off_z, off_h, offs_m, offs_n,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,

):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
    {{gen_defines() | indent_except_first(1)}}

    # -- load k --
    k = load_checked_block(K_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)
    # -- compute qk ---
    qk = tl.dot(q, k, input_precision=FLOAT32_PRECISION) # TODO: use cuda matmul when q_len <= 2.
    if not PRESCALE_QK:
        qk *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    # If this is the last block of a non divisible seqlen, we still need to load [BLOCK_M, BLOCK_N] elements,
    # which is larger than the actual number of elements. To avoid access memory out of bound,
    # we need to mask out the elements that are out of Q_LEN & KV_LEN.
    m = get_bounded_indices(offs_m, Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n, KV_LEN if CHECK_BLOCK_BOUNDARY else None)

    {{ modification(
        subgraph_number=0,
        output_name="post_mod_scores",
        score="qk",
        b="off_z",
        h="off_h",
        m="m",
        n="n",
        out="qk"
    ) | indent_except_first(1) }}

    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        post_mod_scores = tl.where(offs_n < KV_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        {{ modification(
            subgraph_number=1,
            output_name="mask_mod_output",
            score="qk",
            b="off_z",
            h="off_h",
            m="m",
            n="n",
        ) | indent_except_first(2) }}

        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n < KV_LEN, mask_mod_output, False)
        # apply mask for partially unmasked blocks
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))

    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # -- compute scaling constant ---
    m_ij = tl.maximum(m_i, tl.max(post_mod_scores, 1))
    if not ROWS_GUARANTEED_SAFE:
        masked_out_rows = (m_ij == float("-inf"))
        m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
    else:
        m_ij_masked = m_ij

    alpha = tl.math.exp2(m_i - m_ij_masked)
    p = tl.math.exp2(post_mod_scores - m_ij_masked[:, None])

    # NB: l_i update is pulled up here since it's a bit faster
    # NB: For headdim=256, it's faster to move it back down to after m_i =
    # m_ij
    l_i = l_i * alpha + tl.sum(p, 1)
    # # -- scale and update acc --
    acc = acc * alpha[:, None]
    v = load_checked_block(V_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)
    acc = tl.dot(p.to(MATMUL_PRECISION), v, acc, input_precision=FLOAT32_PRECISION)

    # -- update m_i
    m_i = m_ij

    return acc, l_i, m_i

"""


flex_attention_template = TritonTemplate(
    name="flex_attention",
    grid=flex_attention_grid,
    source=compute_flex_attention
    + compute_forward_inner
    + compute_next_offset_func
    + compute_forward_block_mn
    + load_checked_block
    + get_bounded_indices_func,
)
```

## flex decoding
- `BLOCK_M`,`BLOCK_N`,`BLOCK_K`
- `num_warps`
- `num_stages`
- `SPARSE_KV_BLOCK_SIZE`
- `SPLIT_KV`：将kv分成多少块
- `TILE_KV`：划分以后kv的length
- `GQA_SHARED_HEADS`：
- `SAFE_M_BOUNDARY`,`SAFE_N_BOUNDARY`
- `KV_NUM_BLKS`
- `KV_IDX`
- `PRESCALE_QK`
```python
flex_decoding_template = TritonTemplate(
    name="flex_decoding",
    grid=flex_decoding_grid,
    source=r"""
    {{def_kernel("Q", "K", "V", "M", "L", "KV_NUM_BLKS", "KV_IDX", "FULL_KV_NUM_BLKS", "FULL_KV_IDX")}}
    # Sub notation for this kernel:
    # Q: Query, K: Key, V: Value
    # reduction buffers: M rowmax across local KV split, L local sumexp across local KV split
    # M: Number of queries, N: Number of keys/values
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # BLOCK_M, QK_HEAD_DIM: M, and D dimemsion are always assigned to the same block
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head t: Number of kv splits
    # (Modifiable) Config options:
    # SPLIT_KV: number of blocks K & V are split into
    # TILE_KV: length of each local KV split
    # BLOCK_M: block size that Q is padded along seqlen dim.
    # BLOCK_N: block size of K & V along N dimension.
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    #
    # change of base out of the loop
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check
    # SAFE_M_BOUNDARY: Is Q seqlen a multiple of BLOCK_M? If so, we can skip an extra boundary check for loading query.
    # SAFE_N_BOUNDARY: Is KV seqlen a multiple of BLOCK_N? If so, we can skip an extra boundary check for loading key/value.

    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d) and change of base.
    #
    # SPARSE_KV_BLOCK_SIZE: sparse mask block size along KV seqlen dim.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    #
    #
    # Output: ACC output accumulated across local KV split.

    tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)

    # Define Q Strides
    stride_qz, stride_qh, stride_qg, stride_qm, stride_qk = {{stride("Q")}}
    stride_kz, stride_kh, stride_kn, stride_kk = {{stride("K")}}
    stride_vz, stride_vh, stride_vn, stride_vk = {{stride("V")}}
    stride_mz, stride_mt, stride_mh, stride_mm = {{stride("M")}}
    stride_lz, stride_lt, stride_lh, stride_lm = {{stride("L")}}


    Z = {{size("Q", 0)}}
    ZKV = {{size("K", 0)}}
    HKV = {{size("Q", 1)}}
    G: tl.constexpr = GQA_SHARED_HEADS
    HQ = HKV * G
    Q_LEN = {{size("Q", 3)}}
    KV_LEN = {{size("K", 2)}}

    MATMUL_PRECISION = Q.dtype.element_ty

    # Make sure each split is a multiple of BLOCK_N
    TILE_KV_OG = tl.cdiv(KV_LEN, SPLIT_KV)
    TILE_KV = tl.cdiv(TILE_KV_OG, BLOCK_N) * BLOCK_N
    TILE_KV_MULTIPLE: tl.constexpr = (TILE_KV // BLOCK_N)

    off_z = tl.program_id(0) // HKV
    off_zkv = off_z % ZKV
    off_hkv = tl.program_id(0) % HKV
    off_t = tl.program_id(1)

    q_offset = off_z * stride_qz + off_hkv * stride_qh
    k_offset = off_zkv * stride_kz + off_hkv * stride_kh
    v_offset = off_zkv * stride_vz + off_hkv * stride_vh

    SPARSE_Z = {{size("KV_NUM_BLKS", 0)}}
    SPARSE_HQ = {{size("KV_NUM_BLKS", 1)}}

    sparse_idx_z = off_z % SPARSE_Z
    # TODO: support masks not broadcasted along the head dimension.
    tl.device_assert(SPARSE_HQ == 1)
    sparse_idx_h = 0

    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
    SPARSE_KV_BLOCK_CNT = tl.cdiv(KV_LEN, SPARSE_KV_BLOCK_SIZE)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, V_HEAD_DIM_ROUNDED], dtype=tl.float32)

    # initialize offsets
    tl.device_assert(BLOCK_M % G == 0)
    BLOCK_M_PER_HQ: tl.constexpr = BLOCK_M // G
    off_g = tl.arange(0, G)                                                 # [G]
    offs_g = tl.ravel(tl.broadcast_to(off_g[:, None], [G, BLOCK_M_PER_HQ])) # [BLOCK_M]
    offs_hq = offs_g + off_hkv * G
    off_m = tl.arange(0, BLOCK_M_PER_HQ)                                    # [BLOCK_M_PER_HQ]
    offs_m = tl.ravel(tl.broadcast_to(off_m[None, :], [G, BLOCK_M_PER_HQ])) # [BLOCK_M]
    offs_d = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_vd = tl.arange(0, V_HEAD_DIM_ROUNDED)

    # Get HZ offsets for KV_NUM_BLKS and KV_IDX
    stride_block_z, stride_block_h, stride_block_row = {{stride("KV_NUM_BLKS")}}
    sparse_block_hz_offset = sparse_idx_z * stride_block_z + sparse_idx_h * stride_block_h
    stride_kv_z, stride_kv_h, stride_kv_row, stride_kv_col = {{stride("KV_IDX")}}
    sparse_idx_hz_offset = sparse_idx_z * stride_kv_z + sparse_idx_h * stride_kv_h

    # Calculate KV blocks that belong this CTA.
    block_n_start = off_t * TILE_KV_MULTIPLE                        # n_offset inside sparse block
    block_n_end = block_n_start + TILE_KV_MULTIPLE                  # end BLOCK_N

    q_range = stride_qg * off_g[:, None, None] + stride_qm * off_m[None, :, None] + stride_qk * offs_d[None, None, :]

    if not SAFE_M_BOUNDARY and not SAFE_HEAD_DIM:
        q = tl.load(Q + q_offset + q_range, mask=(offs_d[None, None, :] < QK_HEAD_DIM) & (off_m[None, :, None] < Q_LEN))
    elif SAFE_M_BOUNDARY and not SAFE_HEAD_DIM:
        q = tl.load(Q + q_offset + q_range, mask=offs_d[None, None, :] < QK_HEAD_DIM)
    elif not SAFE_M_BOUNDARY and SAFE_HEAD_DIM:
        q = tl.load(Q + q_offset + q_range, mask=off_m[None, :, None] < Q_LEN)
    else:
        q = tl.load(Q + q_offset + q_range)

    q = tl.reshape(q, [BLOCK_M, QK_HEAD_DIM_ROUNDED])


    # ~~~~~~~~~~~~~~ normal blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Apply both score_mod and mask_mod

    # find first kv block we are loading and the number of blocks we are loading
    # Offset the kv_indices tensor by the correct batch and head
    kv_indices = KV_IDX + sparse_idx_hz_offset
    kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_block_hz_offset)
    indices_idx = block_n_start // SPARSE_KV_MULTIPLE
    off_n_block_in_sparse = block_n_start % SPARSE_KV_MULTIPLE
    off_n = tl.load(kv_indices + indices_idx) * SPARSE_KV_BLOCK_SIZE + off_n_block_in_sparse * BLOCK_N
    # first kv block we're loading

    # last valid block according to sparse mask
    block_n_last_valid = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))

    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(QK_HEAD_DIM, KV_LEN),                # (d, N)
        strides=(stride_kk, stride_kn),
        offsets=(0, off_n),
        block_shape=(QK_HEAD_DIM_ROUNDED, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(KV_LEN, V_HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(off_n, 0),
        block_shape=(BLOCK_N, V_HEAD_DIM_ROUNDED),
        order=(1, 0)
    )
    offs_n = tl.arange(0, BLOCK_N) + off_n

    acc, l_i, m_i = forward_inner(
        {{gen_argdefs()}},
        q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
        # accumulatd values
        acc, l_i, m_i,
        #offsets
        off_z, offs_hq[:, None], offs_m[:, None], offs_n[None, :],
        #block sparse data
        kv_indices, kv_num_blocks,
        block_n_start, block_n_end if block_n_end <= block_n_last_valid else block_n_last_valid,
        MATMUL_PRECISION,
        IS_FULL_BLOCKS=False,
    )


    # ~~~~~~~~~~~~~~ "full" blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We know these blocks are guaranteed to be "full", so we don't need to
    # apply mask_mod to them - only score_mod
    if HAS_FULL_BLOCKS:
        kv_indices = FULL_KV_IDX + sparse_idx_hz_offset
        kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_block_hz_offset)
        # Assign full block in a reverse order for off_t. Prioritize the last CTA.
        block_n_start = (SPLIT_KV - off_t - 1) * TILE_KV_MULTIPLE
        block_n_end = block_n_start + TILE_KV_MULTIPLE
        indices_idx = block_n_start // SPARSE_KV_MULTIPLE
        off_n_block_in_sparse = block_n_start % SPARSE_KV_MULTIPLE
        off_n = tl.load(kv_indices + indices_idx) * SPARSE_KV_BLOCK_SIZE + off_n_block_in_sparse * BLOCK_N

        # last valid block according to sparse mask
        block_n_last_valid = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))

        K_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(QK_HEAD_DIM, KV_LEN),                # (d, N)
            strides=(stride_kk, stride_kn),
            offsets=(0, off_n),
            block_shape=(QK_HEAD_DIM_ROUNDED, BLOCK_N),
            order=(0, 1)
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + v_offset,
            shape=(KV_LEN, V_HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(off_n, 0),
            block_shape=(BLOCK_N, V_HEAD_DIM_ROUNDED),
            order=(1, 0)
        )
        offs_n = tl.arange(0, BLOCK_N) + off_n

        acc, l_i, m_i = forward_inner(
            {{gen_argdefs()}},
            q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
            # accumulatd values
            acc, l_i, m_i,
            #offsets
            off_z, offs_hq[:, None], offs_m[:, None], offs_n[None, :],
            #block sparse data
            kv_indices, kv_num_blocks,
            block_n_start, block_n_end if block_n_end <= block_n_last_valid else block_n_last_valid,
            MATMUL_PRECISION,
            IS_FULL_BLOCKS=True,
        )

    m_offset = off_t * stride_mt + off_z * stride_mz
    l_offset = off_t * stride_lt + off_z * stride_lz

    M_block_ptr = tl.make_block_ptr(
        base=M + m_offset,
        shape=(G, Q_LEN),                   # (G, M)
        strides=(stride_mh, stride_mm),
        offsets=(off_hkv*G, 0),
        block_shape=(G, BLOCK_M_PER_HQ),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        base=L + l_offset,
        shape=(G, Q_LEN),                   # (G, M)
        strides=(stride_lh, stride_lm),
        offsets=(off_hkv*G, 0),
        block_shape=(G, BLOCK_M_PER_HQ),
        order=(1, 0)
    )

    # Store output, logsumexp and rowmax for cross CTA reduction. (all in float32, even when input data are in fp16)
    m_i = m_i.reshape(G, BLOCK_M_PER_HQ)
    l_i = l_i.reshape(G, BLOCK_M_PER_HQ)
    if SAFE_M_BOUNDARY:
        tl.store(M_block_ptr, m_i)
        tl.store(L_block_ptr, l_i)
    else:
        tl.store(M_block_ptr, m_i, boundary_check=(1,))
        tl.store(L_block_ptr, l_i, boundary_check=(1,))

    # -- store output
    idx_z = off_z
    idx_t = off_t
    idx_hq = off_hkv*G + off_g[:, None, None]
    idx_m = off_m[None, :, None]
    idx_d = offs_vd[None, None, :]

    mask = (idx_m < Q_LEN) & (idx_d < V_HEAD_DIM)
    acc = acc.reshape(G, BLOCK_M_PER_HQ, V_HEAD_DIM)
    {{store_output(("idx_z", "idx_t", "idx_hq", "idx_m", "idx_d"), "acc", "mask")}}
 """
    + compute_forward_inner
    + get_bounded_indices_func
    + load_checked_block
    + load_checked_2d
    + compute_next_offset_func
    + compute_forward_block_mn, #
)
```

## pointwise（elementwise）
elementwise的算子没有固定的kernel，像是直接从inductor的ir直接到triton ir
在`_inductor/runtime/triton_heuristics.py`
- `size_hints`：包含tensor shape的dict
- `triton_meta`：内核设备信息
- `tile_hint`：内核策略，例如`TileHint.SQUARE`
- `min_elem_per_thread`：每个thread应该处理的最少元素
- `inductor_meta`：控制tune过程的数据，包括是否启用自动调优、最大调优配置是什么
```python
def pointwise(
    size_hints,
    triton_meta,
    tile_hint=None,
    filename=None,
    min_elem_per_thread=0,
    inductor_meta=None,
):
    """
    Construct @triton.heuristics() based on size_hints.
    """
    inductor_meta = {} if inductor_meta is None else inductor_meta
    assert not inductor_meta.get("no_x_dim")

    numel = functools.reduce(operator.mul, size_hints.values())
    bs = max(256, min(numel // 128, 1024))

    hinted_configs = autotune_hints_to_configs(
        inductor_meta.get("autotune_hints", OrderedSet()),
        size_hints,
        bs,
        triton_meta["device"],
    )

    triton_config_with_settings = functools.partial(
        triton_config, min_elem_per_thread=min_elem_per_thread
    )

    configs = None
    if len(size_hints) == 1:
        if disable_pointwise_autotuning(inductor_meta) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            configs = [triton_config_with_settings(size_hints, bs)]
        else:
            configs = [
                triton_config_with_settings(size_hints, bs, num_elements_per_warp=256),
                triton_config_with_settings(
                    size_hints, bs // 2, num_elements_per_warp=64
                ),
                *hinted_configs,
            ]
    if len(size_hints) == 2:
        if (
            disable_pointwise_autotuning(inductor_meta) or tile_hint == TileHint.SQUARE
        ) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            configs = [triton_config_with_settings(size_hints, 32, 32)]
        else:
            configs = [
                triton_config_with_settings(size_hints, 32, 32),
                triton_config_with_settings(size_hints, 64, 64),  # ~8% better for fp16
                triton_config_with_settings(size_hints, 256, 16),
                triton_config_with_settings(size_hints, 16, 256),
                triton_config_with_settings(size_hints, bs, 1),
                triton_config_with_settings(size_hints, 1, bs),
                *hinted_configs,
            ]
    if len(size_hints) == 3:
        if disable_pointwise_autotuning(inductor_meta):
            configs = [triton_config_with_settings(size_hints, 16, 16, 16)]
        else:
            configs = [
                triton_config_with_settings(size_hints, 16, 16, 16),
                triton_config_with_settings(size_hints, 64, 8, 8),
                triton_config_with_settings(size_hints, 8, 64, 8),
                triton_config_with_settings(size_hints, 8, 8, 64),
                triton_config_with_settings(size_hints, bs, 1, 1),
                triton_config_with_settings(size_hints, 1, bs, 1),
                triton_config_with_settings(size_hints, 1, 1, bs),
                *hinted_configs,
            ]

    if not configs:
        raise NotImplementedError(f"size_hints: {size_hints}")
    return cached_autotune(
        size_hints,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.POINTWISE,
        filename=filename,
    )
```