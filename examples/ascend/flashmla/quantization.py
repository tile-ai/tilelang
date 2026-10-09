"""SIMD dequantization of the interleaved V4.1 KV records from PR #229."""

import tilelang.ascend.language as T


def rolled(n):
    return T.unroll(n, annotations={"pragma_unroll_factor": 1})


@T.macro
def dequantize(raw, scales, nz, slot, fp4):
    with T.SimdVF():
        m16 = T.simd.pset(16)
        m8 = T.simd.pset(8)
        arange = T.reinterpret("uint16x128", T.simd.vci(0, "int16"))
        if fp4:
            row_indices = T.simd.vshls(T.simd.vshrs(arange, 5), 8)
            scale_indices = T.simd.vadd(arange, row_indices)
            for row in rolled(8):
                packed_bits = T.simd.vgather2(raw[slot, row * 4, 256], scale_indices, m16)
                packed = T.reinterpret("float8_e4m3fnx256", packed_bits)
                even = T.simd.vcvt(packed, "float32", m16, part=0)
                odd = T.simd.vcvt(packed, "float32", m16, part=2)
                even_bits = T.simd.vshrs(T.reinterpret("uint32x64", even), 16)
                bits = T.simd.vor(even_bits, T.reinterpret("uint32x64", odd))
                T.simd.vsts(scales[row * 128], T.reinterpret("bfloat16x128", bits), m16)
        else:
            row_indices = T.simd.vmuls(T.simd.vshrs(arange, 4), 528)
            scale_indices = T.simd.vadd(arange, row_indices)
            for row in rolled(4):
                packed = T.simd.vgather2(raw[slot, row * 8, 512], scale_indices, m16)
                clamped = T.simd.vmins(packed, 0xE0)
                bits = T.simd.vshls(clamped, 7)
                T.simd.vsts(scales[row * 128], T.reinterpret("bfloat16x128", bits), m16)
        T.simd.mem_bar("VST_VLD")
        for row in rolled(32):
            sf = T.alloc_local((4,), "bfloat16x128")
            if not fp4:
                sf01 = T.simd.vld(scales[row * 16], "E2B_B16")
                sf23 = T.simd.vld(scales[row * 16 + 8], "E2B_B16")
                s0, s1 = T.simd.vintlv(sf01, sf01)
                s2, s3 = T.simd.vintlv(sf23, sf23)
                sf[0] = s0
                sf[1] = s1
                sf[2] = s2
                sf[3] = s3
            for chunk in T.unroll(4):
                if fp4:
                    scale = T.simd.vld(scales[row * 32 + chunk * 8], "E2B_B16")
                    bits = T.simd.vld(raw[slot, row, chunk * 64], "UNPK4_B8")
                    packed = T.reinterpret("float4_e2m1fnx512", bits)
                    value = T.simd.vcvt(packed, "bfloat16", m16, part=0)
                    dequant = T.simd.vmul(value, scale, m16)
                    T.simd.vsstb(dequant, nz[chunk * 4224 + row * 16], 33 << 16, m16)
                else:
                    bits = T.simd.vld(raw[slot, row, chunk * 128], "UNPK_B8")
                    packed = T.reinterpret("float8_e4m3fnx256", bits)
                    even = T.simd.vcvt(packed, "float32", m8, part=0)
                    odd = T.simd.vcvt(packed, "float32", m8, part=2)
                    even_bits = T.simd.vshrs(T.reinterpret("uint32x64", even), 16)
                    data_bits = T.simd.vor(even_bits, T.reinterpret("uint32x64", odd))
                    value = T.reinterpret("bfloat16x128", data_bits)
                    dequant = T.simd.vmul(value, sf[chunk], m16)
                    T.simd.vsstb(dequant, nz[chunk * 4224 + row * 16], 33 << 16, m16)
