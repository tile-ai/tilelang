"""Output tiling and AIV transforms shared by the BF16 and FP8 kernel families."""

import tilelang.ascend.language as T

from ..config import Major
from .common import MK_ALIGNMENT, align_up, clamp, matrix_region


@T.macro
def apply_alpha(values, output, alpha):
    """Scale in FP32, then pack BF16 in place within each MAD tile's UB slot."""
    with T.SimdVF():
        # Expose the linear offset directly; equivalent row/column indexing can
        # leave division and remainder in generated code (e.g. MAD_N=48).
        flat_values = T.reshape(values, (values.shape[0] * values.shape[1],))
        flat_output = T.reshape(output, (output.shape[0] * output.shape[1],))
        mask = T.simd.pset(32)
        for i in T.serial(values.shape[0] * values.shape[1] // 64):
            value = T.simd.vld(flat_values[i * 64])
            scaled = T.simd.vmuls(value, alpha, mask)
            if output.dtype == "bfloat16":
                converted = T.simd.vcvt(scaled, "bfloat16")
                T.simd.vsts(flat_output[i * 64], converted, mask, dist="PK_B32")
            else:
                T.simd.vsts(flat_output[i * 64], scaled, mask)


@T.macro
def add_bias(values, output, bias, n_mad):
    with T.SimdVF():
        mask = T.simd.pset(32)
        # Bias is invariant across rows. Keep its few vectors in registers;
        # explicit column unrolling also removes the inner address loop.
        bias_values = T.simd.alloc_local((values.shape[1] // 64,), "float32")
        values_ptr = T.simd.make_ubuf_ptr(T.access_ptr(values[0, 0], "r", extent=values.shape[0] * values.shape[1]), "float32")
        for col in T.Unroll(values.shape[1] // 64, explicit=True):
            if bias.dtype == "bfloat16":
                bias_values[col] = T.simd.vcvt(T.simd.vld(bias[0, n_mad + col * 64], "UNPK_B16"), "float32")
            else:
                bias_values[col] = T.simd.vld(bias[0, n_mad + col * 64])
        for row in T.serial(values.shape[0]):
            for col in T.Unroll(values.shape[1] // 64, explicit=True):
                value, values_ptr = T.simd.vld(values_ptr, post_inc=64)
                result = T.simd.vadd(value, bias_values[col], mask)
                if output.dtype == "bfloat16":
                    T.simd.vsts(output[row, col * 64], T.simd.vcvt(result, "bfloat16"), mask, dist="PK_B32")
                else:
                    T.simd.vsts(output[row, col * 64], result, mask)


@T.macro
def zero_empty_groups(gm_d, output, grouped_layout, block_idx, sid, config):
    """Reuse one zero MAD tile for the empty groups skipped by the scheduler."""
    MAD_M, MAD_N = config.mad_m, config.mad_n
    num_n_tiles = T.ceildiv(gm_d.shape[2], MAD_N)
    num_tiles = T.ceildiv(gm_d.shape[1], MAD_M) * num_n_tiles
    aiv_idx = block_idx * 2 + sid
    # One unconditional UB epoch keeps AIC/AIV slots aligned and shares a
    # single SIMD-to-MTE3 handoff across all empty groups.
    for _ in T.serial(1):
        with T.SimdVF():
            elements_per_word = 2 if output.dtype == "bfloat16" else 1
            words = T.view(output, (output.shape[0] * output.shape[1] // elements_per_word,), dtype="uint32")
            zero = T.simd.vdup(T.uint32(0), "uint32")
            mask = T.simd.pset(32)
            for i in range(MAD_M * MAD_N // elements_per_word // 64):
                T.simd.vsts(words[i * 64], zero, mask)
        k_begin = T.alloc_var("int32", init=0)
        for group in range(grouped_layout.shape[0]):
            k_end = grouped_layout[group]
            if k_begin == k_end:
                for tile in T.serial(aiv_idx, num_tiles, config.num_blocks * 2):
                    m_idx = tile // num_n_tiles * MAD_M
                    n_idx = tile % num_n_tiles * MAD_N
                    rows = T.min(gm_d.shape[1] - m_idx, MAD_M)
                    cols = T.min(gm_d.shape[2] - n_idx, MAD_N)
                    T.copy(
                        output[0:rows, 0:cols],
                        gm_d[group, m_idx : m_idx + rows, n_idx : n_idx + cols],
                        l2_cache_ctrl=config.l2_ctrl_store_cd,
                    )
            k_begin = align_up(k_end, MK_ALIGNMENT)


@T.macro
def quantize_fp8(values, output, scales):
    """BF16-round, quantize per K32, and pack two UE8M0 exponents per int16.

    Keep the reference's two passes: all scales are written before values are
    packed in place, so output never overwrites a later reduction's input.
    """
    with T.SimdVF():
        rows, cols = values.shape[0], values.shape[1]
        output_bytes = T.view(output, output.shape, dtype="uint8")
        mask16 = T.simd.pset(16)
        mask64 = T.simd.pset(16, "PAT_VL64")
        mask32 = T.simd.pset(16, "PAT_VL32")
        mask_f32 = T.simd.pset(32)
        mask_fp8 = T.simd.pset(8, "PAT_VL128")
        mask_hi = T.simd.vnot(mask32, mask64)
        indices = T.reinterpret(T.simd.vci(0, "int16"), "uint16x128")
        for m_base in T.serial(T.ceildiv(rows, 128)):
            for row in T.serial(T.min(rows - m_base * 128, 128)):
                row_mask = T.simd.vcmps(indices, T.uint16(row), mask16, op="eq")
                for group in T.serial(cols // 64):
                    value = T.simd.vld(values[m_base * 128 + row, group * 64])
                    bits = T.simd.vand(T.reinterpret(value, "uint16x128"), T.simd.vdup(T.uint16(0x7FFF), "uint16"), mask64)
                    lo = T.simd.vcmax(bits, mask32)
                    hi = T.simd.vcmax(bits, mask_hi)
                    exponents, unused = T.simd.vintlv(lo, hi)
                    exponents = T.simd.vmaxs(exponents, T.uint16(14433), mask16)
                    exponents = T.simd.vadds(exponents, T.uint16(65536 - 993), mask16)
                    exponents = T.simd.vshrs(exponents, 7, mask16)
                    packed = T.reinterpret(T.simd.vpack(exponents), "uint16x128")
                    broadcast = T.simd.vdupv(packed)
                    T.simd.vsts(scales[group, m_base * 128], T.reinterpret(broadcast, "int16x128"), row_mask, "NORM_B16")
        lo_mask = T.simd.vcmps(T.simd.vand(indices, T.simd.vdup(T.uint16(32), "uint16")), T.uint16(0), op="eq")
        T.simd.mem_bar("VST_VLD")
        for row in T.serial(rows):
            for group in T.serial(cols // 128):
                sf0 = T.reinterpret(T.simd.vld(scales[group * 2, row], "BRC_B16"), "uint16x128")
                sf1 = T.reinterpret(T.simd.vld(scales[group * 2 + 1, row], "BRC_B16"), "uint16x128")
                packed = T.simd.vsel(sf0, sf1, mask64)
                exp_hi = T.simd.vshrs(packed, 8)
                exp_lo = T.simd.vand(packed, T.simd.vdup(T.uint16(255), "uint16"))
                exponent = T.simd.vsel(exp_lo, exp_hi, lo_mask)
                inverse = T.simd.vshls(T.simd.vsub(T.simd.vdup(T.uint16(254), "uint16"), exponent), 7)
                value = T.simd.vmul(T.simd.vld(values[row, group * 128]), T.reinterpret(inverse, "bfloat16x128"))
                zero = T.simd.vdup(T.bfloat16(0), "bfloat16")
                f32lo, f32hi = T.simd.vintlv(zero, value)
                fp8lo = T.simd.vcvt(T.reinterpret(f32lo, "float32x64"), "float8_e4m3fn", mask_f32)
                fp8hi = T.simd.vcvt(T.reinterpret(f32hi, "float32x64"), "float8_e4m3fn", mask_f32)
                compact, unused = T.simd.vdintlv(T.reinterpret(fp8lo, "uint16x128"), T.reinterpret(fp8hi, "uint16x128"))
                # Keep the packed bytes unsigned so Bisheng can fuse the pack
                # and store. Reinterpreting as FP8 prevents that combination.
                T.simd.vsts(output_bytes[row, group * 128], T.simd.vpack(compact), mask_fp8, "NORM_B8")


@T.macro
def stochastic_round_bf16(values, n_idx, num_rows, num_cols):
    """Match DeepGEMM's quartet hash, including zero tail columns and NaNs.

    Hash the original FP32 bits before canonicalizing NaNs. Four neighboring
    values share a hash; its two mixed words provide four 16-bit thresholds.
    The packed BF16 output overwrites the consumed prefix of the FP32 buffer.
    """
    with T.SimdVF():
        cols = values.shape[1]
        bits = T.view(values, (values.shape[0] * cols,), dtype="uint32")
        mask = T.simd.pset(32)
        mask16 = T.simd.pset(16)
        lanes = T.reinterpret(T.simd.vci(0, "int32"), "uint32x64")
        nan_bits = T.simd.vdup(T.uint32(0x7FC00000), "uint32")
        column = T.simd.alloc_local((1,), "uint32")
        h = T.simd.alloc_local((1,), "uint32")
        partial = T.simd.alloc_local((1,), "uint32")
        column[0] = lanes
        for tile in T.serial(T.ceildiv(num_rows * cols, 256)):
            if cols & (cols - 1) == 0:
                column[0] = T.simd.vadds(lanes, T.uint32((tile * 64) & (cols // 4 - 1)), mask)
                if cols < 256:
                    column[0] = T.simd.vand(column[0], T.simd.vdup(T.uint32(cols // 4 - 1), "uint32"), mask)
            else:
                for _wrap in T.Unroll(T.ceildiv(255, cols), explicit=True):
                    valid = T.simd.vcmps(column[0], T.uint32(cols // 4), mask, op="ge")
                    wrapped = T.simd.vadds(column[0], T.uint32(2**32 - cols // 4), mask)
                    column[0] = T.simd.vsel(wrapped, column[0], valid)
            even0, odd0 = T.simd.vld2(bits[tile * 256], "DINTLV_B32")
            even1, odd1 = T.simd.vld2(bits[tile * 256 + 128], "DINTLV_B32")
            value0, value2 = T.simd.vdintlv(even0, even1)
            value1, value3 = T.simd.vdintlv(odd0, odd1)
            column_hash = T.simd.vadds(column[0], T.uint32(n_idx // 4), mask)
            valid0 = T.simd.vcmps(column[0], T.uint32((num_cols + 3) // 4), mask)
            valid1 = T.simd.vcmps(column[0], T.uint32((num_cols + 2) // 4), mask)
            valid2 = T.simd.vcmps(column[0], T.uint32((num_cols + 1) // 4), mask)
            valid3 = T.simd.vcmps(column[0], T.uint32(num_cols // 4), mask)
            h[0] = T.simd.vmuls(value0, T.uint32(0x5671D42B), valid0)
            partial[0] = T.simd.vmuls(value2, T.uint32(0xACE1B8A5), valid2)
            coefficient1 = T.simd.vdup(T.uint32(0x9995E499), "uint32", valid1)
            coefficient3 = T.simd.vdup(T.uint32(0xE153538D), "uint32", valid3)
            T.simd.vmula(h[0], value1, coefficient1, mask)
            T.simd.vmula(partial[0], value3, coefficient3, mask)
            h[0] = T.simd.vadd(T.simd.vadd(h[0], partial[0], mask), column_hash, mask)
            value0 = T.simd.vsel(
                nan_bits, value0, T.simd.vcmp(T.reinterpret(value0, "float32x64"), T.reinterpret(value0, "float32x64"), op="ne")
            )
            value1 = T.simd.vsel(
                nan_bits, value1, T.simd.vcmp(T.reinterpret(value1, "float32x64"), T.reinterpret(value1, "float32x64"), op="ne")
            )
            value2 = T.simd.vsel(
                nan_bits, value2, T.simd.vcmp(T.reinterpret(value2, "float32x64"), T.reinterpret(value2, "float32x64"), op="ne")
            )
            value3 = T.simd.vsel(
                nan_bits, value3, T.simd.vcmp(T.reinterpret(value3, "float32x64"), T.reinterpret(value3, "float32x64"), op="ne")
            )
            high0 = T.simd.vsel(T.reinterpret(T.simd.vshrs(value0, 16), "uint16x128"), T.reinterpret(value1, "uint16x128"), mask)
            high1 = T.simd.vsel(T.reinterpret(T.simd.vshrs(value2, 16), "uint16x128"), T.reinterpret(value3, "uint16x128"), mask)
            low0 = T.simd.vnot(
                T.simd.vsel(T.reinterpret(value0, "uint16x128"), T.reinterpret(T.simd.vshls(value1, 16), "uint16x128"), mask), mask16
            )
            low1 = T.simd.vnot(
                T.simd.vsel(T.reinterpret(value2, "uint16x128"), T.reinterpret(T.simd.vshls(value3, 16), "uint16x128"), mask), mask16
            )
            h[0] = T.simd.vmuls(T.simd.vxor(h[0], T.simd.vshrs(h[0], 23)), T.uint32(0x7FEB352D))
            h[0] = T.simd.vxor(h[0], T.simd.vshrs(h[0], 16))
            hash0 = T.simd.vmuls(h[0], T.uint32(0x846CA68B))
            hash1 = T.simd.vmuls(h[0], T.uint32(0xD35A2D97))
            hash0 = T.reinterpret(T.simd.vxor(hash0, T.simd.vshrs(hash0, 11)), "uint16x128")
            hash1 = T.reinterpret(T.simd.vxor(hash1, T.simd.vshrs(hash1, 11)), "uint16x128")
            round0 = T.simd.vcmp(hash0, low0, mask16, op="gt")
            round1 = T.simd.vcmp(hash1, low1, mask16, op="gt")
            high0 = T.simd.vsel(T.simd.vadds(high0, T.uint16(1), round0), high0, round0)
            high1 = T.simd.vsel(T.simd.vadds(high1, T.uint16(1), round1), high1, round1)
            T.simd.vst2(bits[tile * 128], T.reinterpret(high0, "uint32x64"), T.reinterpret(high1, "uint32x64"))
            if cols & (cols - 1) != 0:
                column[0] = T.simd.vadds(column[0], T.uint32(64), mask)


def make_store_output(config, major_a, major_b, *, blockscaled=False, epilogue_kind="identity"):
    """Drain all MAD tiles in a block; GM stores exclude FP8 load/fixpipe padding."""
    MAD_M, MAD_N = config.mad_m, config.mad_n
    num_mad_m_tiles = config.block_m // MAD_M
    num_mad_n_tiles = config.block_n // MAD_N

    @T.macro
    def process_mad_block(values, output, scales, bias, alpha, n_idx, n_mad, rows, cols):
        if epilogue_kind == "alpha":
            apply_alpha(values, output, alpha)
        elif epilogue_kind == "bias":
            add_bias(values, output, bias, n_mad)
        elif epilogue_kind == "fp8":
            quantize_fp8(values, output, scales)
        elif epilogue_kind == "stochastic":
            stochastic_round_bf16(values, n_idx, rows, cols)

    @T.macro
    def store_output(
        l0c,
        epilogue_ub,
        output_ub,
        gm_d,
        m_idx,
        n_idx,
        actual_m,
        actual_n,
        epilogue_idx: T.Ref,
        alpha=1.0,
        batch_idx=0,
        sid=0,
        gm_sfd=None,
        sf_ub=None,
        gm_bias=None,
        bias_ub=None,
    ):
        # Preserve MAD-tile drain order, including direct L0C-to-GM stores.
        with T.Stage(0):
            if epilogue_kind == "bias":
                T.copy(gm_bias[0:1, n_idx : n_idx + actual_n], bias_ub[0:1, 0:actual_n])
            for m_mad_tile_idx in T.Unroll(num_mad_m_tiles, explicit=True):
                m_mad_idx = m_mad_tile_idx * MAD_M
                if m_mad_idx < actual_m:
                    eff_mad_m = clamp(actual_m - m_mad_idx, 0, MAD_M)
                    load_mad_m = align_up(eff_mad_m, 32) if blockscaled and major_a == Major.MN else eff_mad_m
                    for n_mad_tile_idx in T.Unroll(num_mad_n_tiles, explicit=True):
                        n_mad_idx = n_mad_tile_idx * MAD_N
                        if n_mad_idx < actual_n:
                            eff_mad_n = clamp(actual_n - n_mad_idx, 0, MAD_N)
                            load_mad_n = align_up(eff_mad_n, 32) if blockscaled and major_b == Major.MN else eff_mad_n
                            if config.direct_store:
                                T.copy(
                                    l0c[m_mad_tile_idx, n_mad_tile_idx, 0:eff_mad_m, 0:eff_mad_n],
                                    matrix_region(gm_d, m_idx + m_mad_idx, n_idx + n_mad_idx, eff_mad_m, eff_mad_n, batch_idx),
                                    unit_flag_ctrl=3,
                                    l2_cache_ctrl=config.l2_ctrl_store_cd,
                                )
                            else:
                                # A complete MAD tile belongs to one AIV. The compiler
                                # projects its copy and private UB epoch onto that AIV.
                                if sid == (epilogue_idx & 1):
                                    for _ in T.serial(1):
                                        T.assume_no_conflict(gm_d)
                                        T.copy(
                                            l0c[m_mad_tile_idx, n_mad_tile_idx, 0:load_mad_m, 0:load_mad_n],
                                            epilogue_ub[0:load_mad_m, 0:load_mad_n],
                                            unit_flag_ctrl=3,
                                        )
                                        process_mad_block(
                                            epilogue_ub,
                                            output_ub,
                                            sf_ub,
                                            bias_ub,
                                            alpha,
                                            n_idx + n_mad_idx,
                                            n_mad_idx,
                                            eff_mad_m,
                                            eff_mad_n,
                                        )
                                        T.copy(
                                            output_ub[0:eff_mad_m, 0:eff_mad_n],
                                            matrix_region(gm_d, m_idx + m_mad_idx, n_idx + n_mad_idx, eff_mad_m, eff_mad_n, batch_idx),
                                            l2_cache_ctrl=config.l2_ctrl_store_cd,
                                        )
                                        if epilogue_kind == "fp8":
                                            sf_col = (batch_idx * gm_d.shape[-1] + (n_idx + n_mad_idx)) // 64
                                            T.assume_no_conflict(gm_sfd)
                                            T.copy(
                                                sf_ub[0 : eff_mad_n // 64, 0:eff_mad_m],
                                                gm_sfd[
                                                    sf_col : sf_col + eff_mad_n // 64,
                                                    m_idx + m_mad_idx : m_idx + m_mad_idx + eff_mad_m,
                                                ],
                                                l2_cache_ctrl=config.l2_ctrl_store_cd,
                                            )
                            epilogue_idx += 1

    return store_output
