import math

import tilelang
from tilelang import language as T
from tilelang.language import simd as S

from tile_kernels.quant.common import *
from tile_kernels.quant.common import get_packed_ue8m0_pack_factor
from tile_kernels.config import get_num_vec_cores


@tilelang.jit
def get_per_channel_cast_with_psum_kernel_asc(
    hidden: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    num_experts: int,
    token_alignment: int,
):
    num_per_tokens = out_config.sf_block[0]
    assert num_per_tokens in (32, 128)
    if in_config.with_sf:
        num_per_channels = in_config.sf_block[1]
        assert num_per_channels in (32, 128), 'Ascend rescale supports num_per_channels in (32, 128) only'
    else:
        assert in_config.dtype == T.bfloat16, 'per_channel_cast supports bf16 or e4m3 (rescale) input only'

    quant_max = 448.0
    assert out_config.clamp_min_value >= quant_max * (2 ** (-126)), 'Ascend scale exponent fast path requires clamp to normal number'

    # num_per_tokens=128 packed 路径用 block_k=256（而非 1024），使 block_m=128，
    # groups_per_tile=1；num_per_tokens=32 保持 block_k=1024 高性能路径不受影响
    use_large_block_k = in_config.with_sf and in_config.use_packed_ue8m0 and num_per_tokens != 128
    # num_per_channels=128 non-tma non-packed: dequant_ub=(num_per_tokens, block_k) float32 超出共享内存
    # (128×256×4=128KB > 112640B)，减小 block_k 到 128
    if not use_large_block_k and in_config.with_sf and num_per_tokens == 128 and not in_config.use_packed_ue8m0:
        max_block_k = 128
    else:
        max_block_k = 1024 if use_large_block_k else 256
    block_k = math.gcd(max_block_k, hidden)

    pack_factor = get_packed_ue8m0_pack_factor()  # 上提：use_per_expert 条件和 packed output 分支需要 pack_factor
    assert pack_factor == 2

    # [P1] 扩展 use_per_expert 条件：原条件仅 unpacked output 时触发，现增加 packed output 分支。
    # packed output 时 pack_row 需配对 2 个 SF 行，当 (alignment // npt) % pack_factor != 0
    # 时 alignment-tiled 路径的 groups_per_tile*nbg 为奇数无法配对，改用 per-expert 路径。
    use_per_expert = (
        token_alignment % num_per_tokens != 0
        or (
            out_config.use_packed_ue8m0
            and (token_alignment // num_per_tokens) % pack_factor != 0
        )
    )
    if use_per_expert:
        if out_config.use_packed_ue8m0:
            # [P1] packed output per-expert：每次处理 npt*pack_factor=256 行（2 个 SF block），
            # groups_per_tile=2 使 quantize_block 产生 2 个 SF 行供 pack_row 配对
            block_m = num_per_tokens * pack_factor
            groups_per_tile = pack_factor
        else:
            # unpacked output per-expert：每次处理 npt 行（1 个 SF block）
            block_m = num_per_tokens
            groups_per_tile = 1
        num_blocks_per_group = 1
        tile_m = block_m
    else:
        block_m = min(128, 32768 // block_k)
        assert block_m >= num_per_tokens and block_m % num_per_tokens == 0, (
            f'Unsupported Ascend psum tile: block_m={block_m}, num_per_tokens={num_per_tokens}'
        )

        # Adjust block_m downward so that token_alignment is a multiple of block_m.
        # This makes tile_m = token_alignment, ensuring each Persistent iteration
        # covers exactly one alignment block and never spans two experts.
        # Example: alignment=224, block_m=128 → 224%128≠0 → reduce to 32 (224%32=0).
        if token_alignment % block_m != 0:
            for candidate in range(min(block_m, token_alignment), 0, -1):
                if token_alignment % candidate == 0 and candidate % num_per_tokens == 0:
                    block_m = candidate
                    break
            assert token_alignment % block_m == 0, (
                f'Cannot find block_m dividing token_alignment={token_alignment} '
                f'with num_per_tokens={num_per_tokens}'
            )
        groups_per_tile = block_m // num_per_tokens
        # nbg = token_alignment // block_m: each Persistent iter processes one
        # alignment block (tile_m = nbg * block_m = token_alignment), preventing
        # tiles from spanning expert boundaries.
        num_blocks_per_group = token_alignment // block_m
        tile_m = block_m * num_blocks_per_group  # == token_alignment
        assert tile_m == token_alignment

    assert block_k % 128 == 0
    num_hidden_tiles = hidden // block_k

    if in_config.with_sf:
        num_packed_per_channels = num_per_channels * pack_factor if in_config.use_packed_ue8m0 else num_per_channels
    if out_config.use_packed_ue8m0:
        assert (groups_per_tile * num_blocks_per_group) % pack_factor == 0, (
            f'packed ue8m0 requires groups_per_tile*nbg % pack_factor == 0: '
            f'groups_per_tile={groups_per_tile}, nbg={num_blocks_per_group}'
        )
        num_packed_rows = groups_per_tile * num_blocks_per_group // pack_factor

    assert num_experts > 0

    # [P1+P2] per-expert 路径强制单缓冲：跨迭代标量依赖（expert_id/token_offset）需要
    # 同步 UB 读写，双缓冲会读到未同步的陈旧数据。
    # non-per-expert + non-packed f32 dequant: block_k=128 时双缓冲超出 UB，降为单缓冲
    num_stages = 1 if (
        use_per_expert
        or (in_config.with_sf and num_per_tokens == 128 and not in_config.use_packed_ue8m0)
    ) else 2
    num_cores = get_num_vec_cores()

    num_tokens = T.dynamic('num_tokens')
    sf_stride = T.dynamic('sf_stride')
    out_sf_shape_m = T.dynamic('out_sf_shape_m')
    sf_shape = (out_sf_shape_m, hidden * (pack_factor if out_config.use_packed_ue8m0 else 1))
    x_sf_shape = get_sf_shape((num_tokens, hidden), in_config)
    packed_col_major_input = in_config.use_tma_aligned_col_major_sf and in_config.use_packed_ue8m0
    if packed_col_major_input:
        x_sf_shape = (x_sf_shape[0], x_sf_shape[1] // 2)
    x_sf_dtype = T.int16 if packed_col_major_input else in_config.sf_dtype

    @T.macro
    def compute_scale(amax):
        # [P2] 用局部变量 clamped 替代重绑定宏参数 amax。
        # 直接 amax = S.vmax(amax, ...) 会触发 "Immutable value re-bound" 警告，
        # 编译器可能忽略 clamp 操作，导致 amax=0 时 scale_raw=0 → 指数下溢 → 输出 nan。
        clamped = S.vmax(amax, S.vdup(out_config.clamp_min_value, T.float32))
        if not out_config.round_sf:
            scale = S.vdiv(clamped, S.vdup(quant_max, T.float32))
            sf_inv = S.vdiv(S.vdup(quant_max, T.float32), clamped)
            return scale, sf_inv
        scale_raw = S.vmul(clamped, S.vdup(1.0 / quant_max, T.float32))
        scale_bits = T.reinterpret(scale_raw, 'uint32x64')
        scale_exponent = S.vadds(S.vshrs(S.vsub(scale_bits, S.vdup(1, T.uint32)), 23), 1)
        inverse_exponent = S.vsub(S.vdup(254, T.uint32), scale_exponent)
        inverse = T.reinterpret(S.vshls(inverse_exponent, 23), 'float32x64')
        if out_config.use_packed_ue8m0:
            return scale_exponent, inverse
        scale = T.reinterpret(S.vshls(scale_exponent, 23), 'float32x64')
        return scale, inverse

    @T.macro
    def quantize_block_bf16(x_ub, o_ub, sf_dst, sf_input, dequant_ub, dst_base=0, num_rows=None):
        # [P2] num_rows: per-expert 路径传入 valid_rows（运行时），仅对 valid_rows 行计算 amax。
        # 旧路径不传（None），rows=block_m（整个 tile 的有效行数，所有行均有效）。
        # 守卫 row_base + row < rows 需要全局行号语义，故 rows 必须是 tile 级而非 group 级。
        rows = block_m if num_rows is None else num_rows
        with T.SimdVF():
            sf_dist = 'PK4_B32' if out_config.use_packed_ue8m0 else 'NORM_B32'
            zero_bf16 = S.vdup(0.0, T.bfloat16)
            abs_mask = T.reinterpret(S.vdup(0x7FFF, T.uint16), 'bfloat16x128')
            amax_values = S.alloc_local((2,), T.bfloat16)
            inverse = S.alloc_local((2,), T.float32)
            if in_config.with_sf:
                lane_channel = T.reinterpret(S.vshrs(S.vci(0, T.int16), 5), 'uint16x128')  # 0x32 1x32 2x32 3x32
                # 适配 NPU num_per_channels（每个缩放因子覆盖的通道数）=128 的场景：num_per_channels=32 两行 packed SF 用 vsel 按 lane 组对选择；num_per_channels=128 单行 packed，用 chunk*8 选字节
                if in_config.use_tma_aligned_col_major_sf and num_per_channels != 128:
                    use_pair1 = S.vcmps(lane_channel, 2, op='ge')  # 0x64 1x64
                    byte_shift = T.reinterpret(S.vshls(S.vand(lane_channel, S.vdup(1, T.uint16)), 3), 'int16x128')  # 0x32 8x32 0x32 8x32
                    byte_mask = S.vdup(0x00FF, T.int16)
                elif in_config.use_tma_aligned_col_major_sf:
                    byte_mask = S.vdup(0x00FF, T.int16)
            for group in T.serial(groups_per_tile):
                row_base = group * num_per_tokens
                for chunk in T.serial(block_k // 128):
                    col = chunk * 128
                    # [P2] 用 alloc_local 替代 alloc_var：S.alloc_var 创建不可变 SSA 变量，
                    # 当 T.serial 的循环次数为运行时变量（per-expert 路径 valid_rows）时，
                    # 循环无法展开，amax = S.vmax(amax, ...) 重绑定被忽略，amax 始终为 0。
                    # alloc_local 创建可变本地内存，通过数组索引 amax[0] 读写确保累加生效。
                    amax = S.alloc_local((1,), T.bfloat16)
                    amax[0] = S.vdup(0.0, T.bfloat16)
                    if in_config.with_sf:
                        # 适配 NPU num_per_channels=128：每 chunk 对应 1 个 SF 组，直接用 chunk 索引；num_per_channels=32 用 lane_channel 派生
                        if num_per_channels == 128:
                            scale_index = S.vdup(chunk, T.uint16)
                        else:
                            scale_index = S.vadds(lane_channel, chunk * 4)
                        # [P2] 用 num_per_tokens（编译期常量）替代 rows（运行时变量）作为循环次数，
                        # 强制编译器展开循环。配合 if 守卫只处理有效行。
                        # [P1] 守卫改为 row_base + row：packed output 时 groups_per_tile=2，
                        # group 1 的 row_base=128，需用全局行号判断有效性而非组内行号。
                        for row in T.serial(num_per_tokens):
                            if row_base + row < rows:
                                raw0 = S.vcvt(S.vld(x_ub[row_base + row, col], dist='UNPK4_B8'), T.float32)
                                raw1 = S.vcvt(S.vld(x_ub[row_base + row, col + 64], dist='UNPK4_B8'), T.float32)
                                _, raw_bf16 = S.vdintlv(T.reinterpret(raw0, 'bfloat16x128'), T.reinterpret(raw1, 'bfloat16x128'))
                                if in_config.use_tma_aligned_col_major_sf:
                                    sf_row = row_base + row
                                    if num_per_channels == 128:
                                        packed = S.vld(sf_input[0, sf_row], dist='BRC_B16')
                                        byte_shift = S.vdup(chunk * 8, T.int16)
                                        exponents = S.vand(S.vshr(packed, byte_shift), byte_mask)
                                    else:
                                        packed0 = S.vld(sf_input[chunk * 2, sf_row], dist='BRC_B16')
                                        packed1 = S.vld(sf_input[chunk * 2 + 1, sf_row], dist='BRC_B16')
                                        packed = S.vsel(packed1, packed0, use_pair1)
                                        exponents = S.vand(S.vshr(packed, byte_shift), byte_mask)
                                    scale_bits = S.vshls(exponents, 7)
                                else:
                                    exponents = T.reinterpret(S.vld(sf_input[row_base + row, 0], dist='UNPK_B8'), 'uint16x128')
                                    scale_bits = S.vshls(S.vselr(exponents, scale_index), 7)
                                values = S.vmul(raw_bf16, T.reinterpret(scale_bits, 'bfloat16x128'))
                                S.vsts(dequant_ub[row, 0], values, dist='NORM_B16')
                                amax[0] = S.vmax(amax[0], S.vand(values, abs_mask))
                    else:
                        # [P2] 非 SF 路径：同样用 num_per_tokens + if 守卫
                        for row in T.serial(num_per_tokens):
                            if row_base + row < rows:
                                values = S.vld(x_ub[row_base + row, col])
                                amax[0] = S.vmax(amax[0], S.vand(values, abs_mask))
                    # [P2] 通过 amax[0] 索引访问可变本地内存，确保 interleave 读到累加后的值
                    amax_values[0], amax_values[1] = S.vintlv(zero_bf16, amax[0])
                    for half in T.unroll(2, explicit=True):
                        scale, inverse[half] = compute_scale(T.reinterpret(amax_values[half], 'float32x64'))
                        S.vsts(sf_dst[dst_base + group, col + half * 64], scale, dist=sf_dist)
                    # [P2] 量化循环同样用 num_per_tokens + if 守卫
                    for half in T.unroll(2, explicit=True):
                        for row in T.serial(num_per_tokens):
                            if row_base + row < rows:
                                value_ub, value_row, value_col = (dequant_ub, row, 0) if in_config.with_sf else (x_ub, row_base + row, col)
                                values = S.vcvt(S.vld(value_ub[value_row, value_col + half * 64], dist='UNPK_B16'), T.float32)
                                quantized = S.vmul(values, inverse[half])
                                S.vsts(o_ub[row_base + row, col + half * 64], S.vcvt(quantized, T.float8_e4m3fn), dist='PK4_B32')

    def dequant_f32_scaled(x_ub, row, col, scale_vec):
        values = S.vcvt(S.vld(x_ub[row, col], dist='UNPK4_B8'), T.float32)
        return S.vmul(values, scale_vec)

    @T.macro
    def quantize_block_f32(x_ub, o_ub, sf_dst, sf_ub, dequant_ub, dst_base=0, num_rows=None):
        # [P2] num_rows: per-expert 路径传入 valid_rows（运行时），仅对 valid_rows 行计算 amax。
        # 旧路径不传（None），rows=block_m（整个 tile 的有效行数，所有行均有效）。
        # 守卫 row_base + row < rows 需要全局行号语义，故 rows 必须是 tile 级而非 group 级。
        rows = block_m if num_rows is None else num_rows
        with T.SimdVF():
            sf_dist = 'PK4_B32' if out_config.use_packed_ue8m0 else 'NORM_B32'
            abs_mask = T.reinterpret(S.vdup(0x7FFFFFFF, T.uint32), 'float32x64')
            num_subvectors = block_k // 64
            amax_values = S.alloc_local((num_subvectors,), T.float32)
            input_scale_values = S.alloc_local((num_subvectors,), T.float32)
            scale_values = S.alloc_local((num_subvectors,), T.uint32 if out_config.use_packed_ue8m0 else T.float32)
            if in_config.use_tma_aligned_col_major_sf:
                # lane 0-31 属组 2k，lane 32-63 属组 2k+1；vsel 按 lane>=32 选择对应 SF
                sf_lane_ge32 = S.vcmps(S.vci(0, T.int32), 32, op='ge')
            for g in T.serial(groups_per_tile):
                row_base = g * num_per_tokens
                for j in T.unroll(num_subvectors, explicit=True):
                    amax_values[j] = S.vdup(0.0, T.float32)
                # [P2] 用 num_per_tokens + if 守卫替代 T.serial(rows)，
                # 确保编译器展开循环，amax_values 累加正确。
                # [P1] 守卫改为 row_base + r：支持 groups_per_tile>1 时按 group 限制有效行。
                for r in T.serial(num_per_tokens):
                    if row_base + r < rows:
                        if in_config.use_tma_aligned_col_major_sf:
                            if num_per_channels == 128:
                                for subvector in T.unroll(num_subvectors, explicit=True):
                                    sf_group = subvector // 2
                                    input_scale_values[subvector] = S.vld(sf_ub[sf_group, row_base + r], dist='BRC_B32')
                            else:
                                for subvector in T.unroll(num_subvectors, explicit=True):
                                    lo = S.vld(sf_ub[subvector * 2, row_base + r], dist='BRC_B32')
                                    hi = S.vld(sf_ub[subvector * 2 + 1, row_base + r], dist='BRC_B32')
                                    input_scale_values[subvector] = S.vsel(hi, lo, sf_lane_ge32)
                        else:
                            if num_per_channels == 128:
                                num_groups = block_k // num_per_channels
                                subvectors_per_group = num_subvectors // num_groups
                                for sv in T.unroll(num_subvectors, explicit=True):
                                    input_scale_values[sv] = S.vld(sf_ub[row_base + r, sv // subvectors_per_group], dist='BRC_B32')
                            else:
                                expanded = S.vld(sf_ub[row_base + r, 0], dist='E2B_B32')
                                low, high = S.vintlv(expanded, expanded)
                                input_scale_values[0], input_scale_values[1] = S.vintlv(low, low)
                                if block_k == 256:
                                    input_scale_values[2], input_scale_values[3] = S.vintlv(high, high)
                        for subvector in T.unroll(num_subvectors, explicit=True):
                            col = subvector * 64
                            values = dequant_f32_scaled(x_ub, row_base + r, col, input_scale_values[subvector])
                            S.vsts(dequant_ub[r, col], values)
                            amax_values[subvector] = S.vmax(amax_values[subvector], S.vand(values, abs_mask))
                if out_config.round_sf:
                    for pair in T.unroll(block_k // 128, explicit=True):
                        for half in T.unroll(2, explicit=True):
                            subvector = pair * 2 + half
                            scale_values[subvector], amax_values[subvector] = compute_scale(amax_values[subvector])
                            S.vsts(sf_dst[dst_base + g, subvector * 64], scale_values[subvector], dist=sf_dist)
                        # [P2] 量化循环（round_sf 路径）：num_per_tokens + if 守卫
                        for r in T.serial(num_per_tokens):
                            if row_base + r < rows:
                                for half in T.unroll(2, explicit=True):
                                    subvector = pair * 2 + half
                                    quantized = S.vmul(S.vld(dequant_ub[r, subvector * 64]), amax_values[subvector])
                                    S.vsts(o_ub[row_base + r, subvector * 64], S.vcvt(quantized, T.float8_e4m3fn), dist='PK4_B32')
                else:
                    for subvector in T.unroll(num_subvectors, explicit=True):
                        scale, inverse = compute_scale(amax_values[subvector])
                        S.vsts(sf_dst[dst_base + g, subvector * 64], scale)
                        for r in T.serial(num_per_tokens):
                            if row_base + r < rows:
                                q = S.vmul(S.vld(dequant_ub[r, subvector * 64]), inverse)
                                S.vsts(o_ub[row_base + r, subvector * 64], S.vcvt(q, T.float8_e4m3fn), dist='PK4_B32')

    @T.macro
    def pack_row(exp_src, pk_dst, exp_row, dst_row):
        with T.SimdVF():
            for base in T.serial(0, block_k, 256):
                lo, hi = S.vintlv(S.vld(exp_src[exp_row, base]), S.vld(exp_src[exp_row + 1, base]))
                S.vsts(pk_dst[dst_row, base * 2], lo, dist='NORM_B8')
                if block_k - base >= 256:
                    S.vsts(pk_dst[dst_row, base * 2 + 256], hi, dist='NORM_B8')

    @T.prim_func
    def per_channel_cast_with_psum_kernel(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],
        x_sf_invs: T.Tensor[x_sf_shape, x_sf_dtype],
        psum_num_tokens_per_expert: T.Tensor[(num_experts,), T.int32],
    ):
        with T.Kernel(num_cores) as core_id:
            psum_ub = T.alloc_shared((num_experts,), T.int32)
            sf_offsets_ub = T.alloc_shared((num_experts + 1,), T.int32)
            T.copy(psum_num_tokens_per_expert, psum_ub)

            # CUDA computes the same compact offsets with expert_combo + cumsum.
            # Ascend keeps the existing scalar/UB style and evaluates that prefix
            # sum directly because num_experts is statically bounded by 128.
            sf_offsets_ub[0] = 0
            previous_end = T.alloc_var(T.int32, init=0)
            for expert in T.serial(num_experts):
                expert_start = (previous_end + token_alignment - 1) // token_alignment * token_alignment
                expert_count = psum_ub[expert] - expert_start
                expert_sf_blocks = T.alloc_var(T.int32, init=T.ceildiv(expert_count, num_per_tokens))
                if out_config.use_packed_ue8m0:
                    expert_sf_blocks = T.ceildiv(expert_sf_blocks, pack_factor)
                sf_offsets_ub[expert + 1] = sf_offsets_ub[expert] + expert_sf_blocks
                previous_end = psum_ub[expert]

            x_ub = T.alloc_shared((block_m, block_k), in_config.dtype)
            out_ub = T.alloc_shared((block_m, block_k), out_config.dtype)
            versions = {x_ub: num_stages, out_ub: num_stages}

            if in_config.with_sf:
                if in_config.use_tma_aligned_col_major_sf:
                    if in_config.use_packed_ue8m0:
                        # ceil_div: block_k 可能 < num_per_channels*pack_factor（num_per_tokens=128 block_k=128 时），整除会得 0 行
                        sf_in_ub = T.alloc_shared((ceil_div(block_k, num_per_channels * pack_factor), block_m), T.int16)
                    else:
                        sf_in_ub = T.alloc_shared((block_k // num_per_channels, block_m), in_config.sf_dtype)
                elif in_config.use_packed_ue8m0:
                    sf_in_ub = T.alloc_shared((block_m, max(block_k // num_per_channels, 64)), in_config.sf_dtype)
                else:
                    sf_in_ub = T.alloc_shared((block_m, max(block_k // num_per_channels, 8)), in_config.sf_dtype)
                dequant_dtype = T.bfloat16 if in_config.use_packed_ue8m0 else T.float32
                dequant_columns = 128 if in_config.use_packed_ue8m0 else block_k
                dequant_ub = T.alloc_shared((num_per_tokens, dequant_columns), dequant_dtype)
                versions[sf_in_ub] = num_stages

            if out_config.use_packed_ue8m0:
                # sf_out_ub 扩容为 groups_per_tile * nbg 行：nbg>1 时同一迭代内多个
                # block 的 SF 需各自独立的缓存行
                sf_out_ub = T.alloc_shared((groups_per_tile * num_blocks_per_group, max(block_k, 256)), T.uint8)
                pk_ub = T.alloc_shared((num_packed_rows, block_k * pack_factor), T.uint8)
                versions[sf_out_ub] = num_stages
                versions[pk_ub] = num_stages
            else:
                # sf_out_ub 扩容为 groups_per_tile * nbg 行：nbg>1 时需容纳所有 block 的 SF
                sf_out_ub = T.alloc_shared((groups_per_tile * num_blocks_per_group, block_k), T.float32)
                versions[sf_out_ub] = num_stages
            T.annotate_buffer_versions(versions)

            expert_id = T.alloc_var(T.int32, init=0)
            if use_per_expert:
                # Per-expert num_per_tokens 对齐单遍路径：每个 Persistent 迭代处理一个 SF block。
                # sf_block_id 全局单调递增，expert cursor 与 sf_offsets_ub 比较；
                # token_offset = expert_start + block_in_expert * num_per_tokens 相对 expert_start
                # 天然与 SF block 边界对齐，SF 计算与量化在同一遍完成。
                for iter_id, tile_id in T.Persistent(
                    [out_sf_shape_m, num_hidden_tiles],
                    num_cores,
                    core_id,
                    group_size=1,
                    num_stages=num_stages,
                ):
                    sf_block_id = iter_id
                    while expert_id < num_experts and sf_block_id >= sf_offsets_ub[expert_id + 1]:
                        expert_id += 1

                    expert_start = T.alloc_var(T.int32, init=0)
                    if expert_id > 0:
                        expert_start = (psum_ub[expert_id - 1] + token_alignment - 1) // token_alignment * token_alignment
                    block_in_expert = T.alloc_var(T.int32, init=0)
                    expert_end = T.alloc_var(T.int32, init=0)
                    token_offset = T.alloc_var(T.int32, init=0)
                    valid_rows = T.alloc_var(T.int32, init=0)

                    col_offset = tile_id * block_k
                    if expert_id < num_experts:
                        block_in_expert = sf_block_id - sf_offsets_ub[expert_id]
                        expert_end = psum_ub[expert_id]
                        # [P1] 用 block_m 替代 num_per_tokens：packed output 时 block_m=256，
                        # 每次迭代处理 2 个 SF block（256 行），unpacked 时 block_m=128 不变
                        token_offset = expert_start + block_in_expert * block_m
                        # [P1] valid_rows 上限改为 block_m：packed output 时允许 256 行
                        valid_rows = T.min(T.max(expert_end - token_offset, 0), block_m)

                        if valid_rows > 0:
                            T.copy(x[token_offset, col_offset], x_ub, l2_cache_ctrl='NOTALLOC_KEEP')
                            if in_config.with_sf:
                                if in_config.use_tma_aligned_col_major_sf:
                                    T.copy(
                                        x_sf_invs[col_offset // num_packed_per_channels, token_offset],
                                        sf_in_ub[: ceil_div(block_k, num_packed_per_channels), :block_m],
                                    )
                                else:
                                    T.copy(x_sf_invs[token_offset, col_offset // num_per_channels], sf_in_ub[:, : block_k // num_per_channels])
                                if in_config.use_packed_ue8m0:
                                    quantize_block_bf16(x_ub, out_ub, sf_out_ub, sf_in_ub, dequant_ub, num_rows=valid_rows)
                                else:
                                    quantize_block_f32(x_ub, out_ub, sf_out_ub, sf_in_ub, dequant_ub, num_rows=valid_rows)
                            else:
                                quantize_block_bf16(x_ub, out_ub, sf_out_ub, None, None, num_rows=valid_rows)
                            T.copy(out_ub[:valid_rows, :], out[token_offset, col_offset])

                            # [P1] per-expert packed output 分支：quantize_block 产生
                            # groups_per_tile=2 个 SF 行，pack_row 配对为 1 个 packed 行。
                            # unpacked output 路径保持原有逻辑：直接 copy SF 行。
                            if out_config.use_packed_ue8m0:
                                for p in T.serial(num_packed_rows):
                                    pack_row(sf_out_ub, pk_ub, 2 * p, p)
                                T.copy(pk_ub[:num_packed_rows, :], out_sf[sf_block_id, col_offset * pack_factor])
                            else:
                                valid_sf_blocks = T.ceildiv(valid_rows, num_per_tokens)
                                T.copy(sf_out_ub[:valid_sf_blocks, :], out_sf[sf_block_id, col_offset])
            else:
                for iter_id, tile_id in T.Persistent(
                    [T.ceildiv(num_tokens, tile_m), num_hidden_tiles],
                    num_cores,
                    core_id,
                    group_size=1,
                    num_stages=num_stages,
                ):
                    token_offset = iter_id * tile_m
                    # Same monotonic expert cursor used by swiglu_forward_asc: each
                    # core observes nondecreasing token tiles in a Persistent loop.
                    while expert_id < num_experts and token_offset >= (psum_ub[expert_id] + token_alignment - 1) // token_alignment * token_alignment:
                        expert_id += 1

                    expert_start = T.alloc_var(T.int32, init=0)
                    if expert_id > 0:
                        expert_start = (psum_ub[expert_id - 1] + token_alignment - 1) // token_alignment * token_alignment
                    expert_end = T.alloc_var(T.int32, init=psum_ub[expert_id])
                    valid_tokens = T.alloc_var(T.int32, init=T.min(T.max(expert_end - token_offset, 0), tile_m))

                    col_offset = tile_id * block_k
                    for bg in T.serial(num_blocks_per_group):
                        tile_row = iter_id * num_blocks_per_group + bg
                        if num_blocks_per_group == 1 or tile_row * block_m < num_tokens:
                            T.copy(x[tile_row * block_m, col_offset], x_ub, l2_cache_ctrl='NOTALLOC_KEEP')
                            if in_config.with_sf:
                                if in_config.use_tma_aligned_col_major_sf:
                                    T.copy(
                                        x_sf_invs[col_offset // num_packed_per_channels, tile_row * block_m],
                                        sf_in_ub[: ceil_div(block_k, num_packed_per_channels), :block_m],  # ceil_div: 防止 block_k < num_per_channels*pack_factor 时拷贝 0 行
                                    )
                                else:
                                    T.copy(x_sf_invs[tile_row * block_m, col_offset // num_per_channels], sf_in_ub[:, : block_k // num_per_channels])
                                if in_config.use_packed_ue8m0:
                                    quantize_block_bf16(x_ub, out_ub, sf_out_ub, sf_in_ub, dequant_ub, dst_base=bg * groups_per_tile)
                                else:
                                    quantize_block_f32(x_ub, out_ub, sf_out_ub, sf_in_ub, dequant_ub, dst_base=bg * groups_per_tile)
                            else:
                                quantize_block_bf16(x_ub, out_ub, sf_out_ub, None, None, dst_base=bg * groups_per_tile)
                            T.copy(out_ub, out[tile_row * block_m, col_offset])

                    if valid_tokens > 0:
                        sf_block_offset = (token_offset - expert_start) // num_per_tokens
                        valid_sf_blocks = T.ceildiv(valid_tokens, num_per_tokens)
                        if not out_config.use_packed_ue8m0:
                            sf_base = sf_offsets_ub[expert_id] + sf_block_offset
                            T.copy(sf_out_ub[:valid_sf_blocks, :], out_sf[sf_base, col_offset])
                        else:
                            for p in T.serial(num_packed_rows):
                                pack_row(sf_out_ub, pk_ub, 2 * p, p)
                            sf_base = sf_offsets_ub[expert_id] + sf_block_offset // pack_factor
                            valid_packed_rows = T.ceildiv(valid_sf_blocks, pack_factor)
                            T.copy(pk_ub[:valid_packed_rows, :], out_sf[sf_base, col_offset * pack_factor])

    return per_channel_cast_with_psum_kernel
