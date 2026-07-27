# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Literal
from tilelang import language as T

decode_i4_to_f16 = """
template <typename T1, typename T2, bool isSigned = false>
__device__ void decode_i4b_to_f16(T1 *_i4s, T2 *B_local_decode, const int N = 8)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x000f000f;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64086408 : 0x64006400;
    uint const i4s = *reinterpret_cast<uint *>(_i4s);
#pragma unroll
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
    }
}

template <typename T1, typename T2>
__device__ void decode_i4s_to_f16(T1 *_i4s, T2 *B_local_decode, const int N = 8)
{
    decode_i4b_to_f16<T1, T2, true>(_i4s, B_local_decode, N);
}

template <typename T1, typename T2>
__device__ void decode_i4u_to_f16(T1 *_i4u, T2 *B_local_decode, const int N = 8)
{
    decode_i4b_to_f16<T1, T2, false>(_i4u, B_local_decode, N);
}
"""

decode_i4_to_f16_scale = """
template <typename T1, typename T2, typename T3, bool isSigned = false, bool withScaling = false>
__device__ void decode_i4b_to_f16_scale(T1 *_i4s, T2 *B_local_decode, const int N = 8, const T3 *scale = nullptr)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x000f000f;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    // Minus 7 to scale the value to signed
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64086408 : 0x64006400;
    uint const i4s = *reinterpret_cast<uint *>(_i4s);
    T3 const scale_r = *scale;
    uint const packed_scales = __pack_half2(scale_r, scale_r);

#pragma unroll
    // decode 2 elems at one time.
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales), "r"(0));
    }
}

template <typename T1, typename T2, typename T3>
__device__ void decode_i4s_to_f16_scale(T1 *_i4s, T2 *B_local_decode, T3 *scale = nullptr, const int N = 8)
{
    decode_i4b_to_f16_scale<T1, T2, T3, true, true>(_i4s, B_local_decode, N, scale);
}

template <typename T1, typename T2, typename T3>
__device__ void decode_i4u_to_f16_scale(T1 *_i4u, T2 *B_local_decode, T3 *scale = nullptr, const int N = 8)
{
    decode_i4b_to_f16_scale<T1, T2, T3, false, true>(_i4u, B_local_decode, N, scale);
}

"""

decode_i4_to_f16_scale_offset = """
template <typename T1, typename T2, typename T3, bool isSigned = false, bool withScaling = false>
__device__ void decode_i4b_to_f16_scale_offset(T1 *_i4s, T2 *B_local_decode, const int N = 8, const T3 *scale = nullptr, const int offset = 0)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x000f000f;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    // Minus 7 to scale the value to signed
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64086408 : 0x64006400;
    uint const i4s = *reinterpret_cast<uint *>(_i4s);
    T3 const scale_l = *scale;
    T3 const scale_r = *(scale + offset);
    uint const packed_scales_l = __pack_half2(scale_l, scale_l);
    uint const packed_scales_r = __pack_half2(scale_r, scale_r);

#pragma unroll
    // decode 2 elems at one time.
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
    }
    #pragma unroll
    for (int i = 0; i < (N / 4); i++)
    {
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales_l), "r"(0));
    }
#pragma unroll
    for (int i = (N / 4); i < (N / 2); i++)
    {
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales_r), "r"(0));
    }
}

template <typename T1, typename T2, typename T3>
__device__ void decode_i4s_to_f16_scale_offset(T1 *_i4s, T2 *B_local_decode, T3 *scale = nullptr, const int offset = 0, const int N = 8)
{
    decode_i4b_to_f16_scale_offset<T1, T2, T3, true, true>(_i4s, B_local_decode, N, scale, offset);
}

template <typename T1, typename T2, typename T3>
__device__ void decode_i4u_to_f16_scale_offset(T1 *_i4u, T2 *B_local_decode, T3 *scale = nullptr, const int offset = 0, const int N = 8)
{
    decode_i4b_to_f16_scale_offset<T1, T2, T3, false, true>(_i4u, B_local_decode, N, scale, offset);
}

"""

decode_i4_to_f16_scale_zeros_original = """
template <typename T1, typename T2, typename T3, typename T4, bool isSigned = false>
__device__ void decode_i4b_to_f16_zeros_original(T1 *_i4s, T2 *B_local_decode, const int N = 8, const T3 *scale = nullptr, const T4 *zeros = nullptr)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x000f000f;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    // Minus 7 to scale the value to signed
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64086408 : 0x64006400;
    uint const i4s = *reinterpret_cast<uint *>(_i4s);
    T3 const scale_r = *scale;
    uint const packed_scales = __pack_half2(scale_r, scale_r);
    // input zeros maybe int32(qzeros) or half format
    T4 const zero_r = *zeros;
    uint const packed_zeros = __pack_half2(zero_r, zero_r);


#pragma unroll
    // decode 2 elems at one time.
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));

        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));

        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_zeros));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales), "r"(0));
    }
}

template <typename T1, typename T2, typename T3, typename T4>
__device__ void decode_i4u_to_f16_scale_zeros_original(T1 *_i4u, T2 *B_local_decode, T3 *scale = nullptr, T4 *zeros = nullptr, const int N = 8)
{
    decode_i4b_to_f16_zeros_original<T1, T2, T3, T4, false>(_i4u, B_local_decode, N, scale, zeros);
}
"""

decode_i4_to_f16_scale_zeros_original_offset = """
template <typename T1, typename T2, typename T3, typename T4, bool isSigned = false>
__device__ void decode_i4b_to_f16_zeros_original_offset(T1 *_i4s, T2 *B_local_decode, const int N = 8, const T3 *scale = nullptr, const T4 *zeros = nullptr, const int offset = 0)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x000f000f;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    // Minus 7 to scale the value to signed
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64086408 : 0x64006400;
    uint const i4s = *reinterpret_cast<uint *>(_i4s);
    T3 const scale_l = *scale;
    T3 const scale_r = *(scale + offset);
    uint const packed_scales_l = __pack_half2(scale_l, scale_l);
    uint const packed_scales_r = __pack_half2(scale_r, scale_r);
    // input zeros maybe int32(qzeros) or half format
    T3 const zeros_l = *zeros;
    T3 const zeros_r = *(zeros + offset);
    uint const packed_zeros_l = __pack_half2(zeros_l, zeros_l);
    uint const packed_zeros_r = __pack_half2(zeros_r, zeros_r);

#pragma unroll
    // decode 2 elems at one time.
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));

        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
    }

#pragma unroll
    for (int i = 0; i < (N / 4); i++)
    {
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_zeros_l));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales_l), "r"(0));
    }
#pragma unroll
    for (int i = (N / 4); i < (N / 2); i++)
    {
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_zeros_r));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales_r), "r"(0));
    }
}

template <typename T1, typename T2, typename T3, typename T4>
__device__ void decode_i4u_to_f16_scale_zeros_original_offset(T1 *_i4u, T2 *B_local_decode, T3 *scale = nullptr, T4 *zeros = nullptr, const int offset = 0, const int N = 8)
{
    decode_i4b_to_f16_zeros_original_offset<T1, T2, T3, T4, false>(_i4u, B_local_decode, N, scale, zeros, offset);
}
"""

decode_i4_to_f16_scale_zeros_rescale = """
template <typename T1, typename T2, typename T3, typename T4, bool isSigned = false>
__device__ void decode_i4b_to_f16_scale_zeros_rescale(T1 *_i4s, T2 *B_local_decode, const int N = 8, const T3 *scale = nullptr, const T4 *zeros = nullptr)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x000f000f;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    // Minus 7 to scale the value to signed
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64086408 : 0x64006400;
    uint const i4s = *reinterpret_cast<uint *>(_i4s);
    T3 const scale_r = *scale;
    uint const packed_scales = __pack_half2(scale_r, scale_r);
    T4 const zero_r = *zeros;
    uint const packed_zeros = 0x80008000 | __pack_half2(zero_r, zero_r);

#pragma unroll
    // decode 2 elems at one time.
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));

        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));

        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales), "r"(packed_zeros));
    }
}

template <typename T1, typename T2, typename T3, typename T4>
__device__ void decode_i4u_to_f16_scale_zeros_rescale(T1 *_i4u, T2 *B_local_decode, T3 *scale = nullptr, T4 *zeros = nullptr, const int N = 8)
{
    decode_i4b_to_f16_scale_zeros_rescale<T1, T2, T3, T4, false>(_i4u, B_local_decode, N, scale, zeros);
}

"""

decode_i4_to_f16_scale_zeros_rescale_offset = """
template <typename T1, typename T2, typename T3, typename T4, bool isSigned = false>
__device__ void decode_i4b_to_f16_scale_zeros_rescale_offset(T1 *_i4s, T2 *B_local_decode, const int N = 8, const T3 *scale = nullptr, const T4 *zeros = nullptr, const int offset = 0)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x000f000f;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    // Minus 7 to scale the value to signed
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64086408 : 0x64006400;
    uint const i4s = *reinterpret_cast<uint *>(_i4s);
    T3 const scale_l = *scale;
    T3 const scale_r = *(scale + offset);
    uint const packed_scales_l = __pack_half2(scale_l, scale_l);
    uint const packed_scales_r = __pack_half2(scale_r, scale_r);
    // input zeros maybe int32(qzeros) or half format
    T3 const zeros_l = *zeros;
    T3 const zeros_r = *(zeros + offset);
    uint const packed_zeros_l = 0x80008000 | __pack_half2(zeros_l, zeros_l);
    uint const packed_zeros_r = 0x80008000 | __pack_half2(zeros_r, zeros_r);

#pragma unroll
    // decode 2 elems at one time.
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));

        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
    }
#pragma unroll
    for (int i = 0; i < (N / 4); i++)
    {
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales_l), "r"(packed_zeros_l));
    }
#pragma unroll
    for (int i = (N / 4); i < (N / 2); i++)
    {
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales_r), "r"(packed_zeros_r));
    }
}

template <typename T1, typename T2, typename T3, typename T4>
__device__ void decode_i4u_to_f16_scale_zeros_rescale_offset(T1 *_i4u, T2 *B_local_decode, T3 *scale = nullptr, T4 *zeros = nullptr, const int offset = 0, const int N = 8)
{
    decode_i4b_to_f16_scale_zeros_rescale_offset<T1, T2, T3, T4, false>(_i4u, B_local_decode, N, scale, zeros, offset);
}

"""

decode_i4_to_f16_scale_zeros_quantized = """
template <typename T1, typename T2, typename T3, typename T4, bool isSigned = false>
__device__ void decode_i4b_to_f16_scale_zeros_quantized(T1 *_i4s, T2 *B_local_decode, const int N = 8, const T3 *scale = nullptr, const T4 *zeros = nullptr)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x000f000f;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    // Minus 7 to scale the value to signed
    uint const i4s = *reinterpret_cast<uint *>(_i4s);
    T3 const scale_r = *scale;
    uint const packed_scales = __pack_half2(scale_r, scale_r);
    // input zeros maybe int32(qzeros) or half format
    int16_t const zero_r = *((int16_t*)zeros);
    uint median_num = ((0xe400 | zero_r) << 16) | (0xe400 | zero_r);

#pragma unroll
    // decode 2 elems at one time.
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));

        asm volatile("add.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(median_num));

        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales), "r"(0));
    }
}

template <typename storage_dtype, typename target_dtype, typename scale_dtype, typename zero_dtype>
__device__ void decode_i4u_to_f16_scale_zeros_quantized(storage_dtype *_i4u, target_dtype *B_local_decode, scale_dtype *scale = nullptr, zero_dtype *zeros = nullptr, const int N = 8)
{
    decode_i4b_to_f16_scale_zeros_quantized<storage_dtype, target_dtype, scale_dtype, zero_dtype, false>(_i4u, B_local_decode, N, scale, zeros);
}
"""

decode_i4_to_f16_scale_zeros_quantized_offset = """
template <typename T1, typename T2, typename T3, bool isSigned = false>
__device__ void decode_i4b_to_f16_scale_zeros_quantized_offset(T1 *_i4s, T2 *B_local_decode, const int N = 8, const T3 *scale = nullptr, const T1 *qzeros = nullptr, const int scale_offset = 0, const int qzeros_offset = 0, const int group_offset = 0)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x000f000f;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    // Minus 7 to scale the value to signed
    uint const i4s = *reinterpret_cast<uint *>(_i4s);

    T3 const scale_l = *scale;
    T3 const scale_r = *(scale + scale_offset);
    uint const packed_scales_l = __pack_half2(scale_l, scale_l);
    uint const packed_scales_r = __pack_half2(scale_r, scale_r);

    const int num_elems_per_storage_dtype = sizeof(T1) * 8 / 4;

    T1 const qzeros_l = *qzeros;
    T1 const qzeros_r = *(qzeros + qzeros_offset);
    int16_t const zero_l = (qzeros_l >> (group_offset * 4) & 0xf);
    int16_t const zero_r = (qzeros_r >> (group_offset * 4) & 0xf);

    uint median_num_l = ((0xe400 | zero_l) << 16) | (0xe400 | zero_l);
    uint median_num_r = ((0xe400 | zero_r) << 16) | (0xe400 | zero_r);

#pragma unroll
    // decode 2 elems at one time.
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
    }
    #pragma unroll
    for (int i = 0; i < (N / 4); i++)
    {
        asm volatile("add.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(median_num_l));

        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales_l), "r"(0));
    }
#pragma unroll
    for (int i = (N / 4); i < (N / 2); i++)
    {
        asm volatile("add.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(median_num_r));

        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales_r), "r"(0));
    }
}

template <typename storage_dtype, typename target_dtype, typename scale_dtype>
__device__ void decode_i4u_to_f16_scale_zeros_quantized_offset(storage_dtype *_i4u, target_dtype *B_local_decode, scale_dtype *scale = nullptr, storage_dtype *qzeros = nullptr, const int scale_offset = 0, const int zero_offset = 0, const int group_offset = 0, const int N = 8)
{
    decode_i4b_to_f16_scale_zeros_quantized_offset<storage_dtype, target_dtype, scale_dtype, false>(_i4u, B_local_decode, N, scale, qzeros, scale_offset, zero_offset, group_offset);
}
"""

decode_i2_to_f16 = """
template <typename T1, typename T2, bool isSigned = false>
__device__ void decode_i2b_to_f16(T1 *_i2s, T2 *B_local_decode, const int N = 8)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x00030003;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64026402 : 0x64006400;
    int16_t const i2s_i16 = *reinterpret_cast<int16_t *>(_i2s);
    // decode 2 elems at one time.
    // interleave {e15,e13,e11,e9,e7,e5,e3,e1,e14,e12,e10,e8,e6,e4,e2,e0}
    // only decode for {x,x,x,x,e7,e5,e3,e1,x,x,x,x,e6,e4,e2,e0}
    // otherwise the pointer of _i2s should be moved to
    int i2s = (i2s_i16 & 0x00ff);
    i2s |= ((i2s_i16 & 0xff00) << 8);

#pragma unroll
    for (int i = 0; i < (N / 2); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
    }
}

template <typename T1, typename T2>
__device__ void decode_i2s_to_f16(T1 *_i2s, T2 *B_local_decode, const int N = 8)
{
    decode_i2b_to_f16<T1, T2, true>(_i2s, B_local_decode, N);
}

template <typename T1, typename T2>
__device__ void decode_i2u_to_f16(T1 *_i2u, T2 *B_local_decode, const int N = 8)
{
    decode_i2b_to_f16<T1, T2, false>(_i2u, B_local_decode, N);
}
"""

decode_i2_to_f16_scale = """
template <typename T1, typename T2, typename T3, bool isSigned = false>
__device__ void decode_i2b_to_f16_scale(T1 *_i2s, T2 *B_local_decode, T3 *scale = nullptr, const int N = 8)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x00030003;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64026402 : 0x64006400;
    int16_t const i2s_i16 = *reinterpret_cast<int16_t *>(_i2s);
    // decode 2 elems at one time.
    // interleave {e15,e13,e11,e9,e7,e5,e3,e1,e14,e12,e10,e8,e6,e4,e2,e0}
    // only decode for {x,x,x,x,e7,e5,e3,e1,x,x,x,x,e6,e4,e2,e0}
    // otherwise the pointer of _i2s should be moved to
    int i2s = (i2s_i16 & 0x00ff);
    i2s |= ((i2s_i16 & 0xff00) << 8);

#pragma unroll
    for (int i = 0; i < (N / 2); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(__pack_half2(*scale, *scale)), "r"(0));
    }
}

template <typename T1, typename T2, typename T3>
__device__ void decode_i2s_to_f16_scale(T1 *_i2s, T2 *B_local_decode, T3 *scale, const int N = 8)
{
    decode_i2b_to_f16_scale<T1, T2, T3, true>(_i2s, B_local_decode, scale, N);
}

template <typename T1, typename T2, typename T3>
__device__ void decode_i2u_to_f16_scale(T1 *_i2u, T2 *B_local_decode,  T3 *scale, const int N = 8)
{
    decode_i2b_to_f16_scale<T1, T2, T3, false>(_i2u, B_local_decode, scale, N);
}
"""

decode_i2_to_f16_scale_zeros_original_offset = """
template <typename T1, typename T2, typename T3, bool isSigned = false>
__device__ void decode_i2b_to_f16_scale_zeros_original_offset(T1 *_i2s, T2 *B_local_decode, T3 *scale = nullptr, T3 *zeros = nullptr, const int offset = 0, const int N = 8)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x00030003;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64026402 : 0x64006400;
    int16_t const i2s_i16 = *reinterpret_cast<int16_t *>(_i2s);
    // decode 2 elems at one time.
    // interleave {e15,e13,e11,e9,e7,e5,e3,e1,e14,e12,e10,e8,e6,e4,e2,e0}
    // only decode for {x,x,x,x,e7,e5,e3,e1,x,x,x,x,e6,e4,e2,e0}
    // otherwise the pointer of _i2s should be moved to
    int i2s = (i2s_i16 & 0x00ff);
    i2s |= ((i2s_i16 & 0xff00) << 8);

    T3 const zeros_l = *zeros;
    T3 const zeros_r = *(zeros + offset);
    uint const packed_zeros_l = __pack_half2(zeros_l, zeros_l);
    uint const packed_zeros_r = __pack_half2(zeros_r, zeros_r);

    T3 const scale_l = *scale;
    T3 const scale_r = *(scale + offset);
    uint const packed_scales_l = __pack_half2(scale_l, scale_l);
    uint const packed_scales_r = __pack_half2(scale_r, scale_r);

#pragma unroll
    for (int i = 0; i < (N / 2); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
    }
    #pragma unroll
    for (int i = 0; i < (N / 4); i++)
    {
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_zeros_l));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales_l), "r"(0));
    }
#pragma unroll
    for (int i = (N / 4); i < (N / 2); i++)
    {
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_zeros_r));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales_r), "r"(0));
    }
}

template <typename T1, typename T2, typename T3>
__device__ void decode_i2u_to_f16_scale_zeros_original_offset(T1 *_i2u, T2 *B_local_decode,  T3 *scale, T3 *zeros, const int offset = 0, const int N = 8)
{
    decode_i2b_to_f16_scale_zeros_original<T1, T2, T3, false>(_i2u, B_local_decode, scale, zeros, offset, N);
}
"""

decode_i2_to_f16_scale_zeros_original = """
template <typename T1, typename T2, typename T3, bool isSigned = false>
__device__ void decode_i2b_to_f16_scale_zeros_original(T1 *_i2s, T2 *B_local_decode, T3 *scale = nullptr, T3 *zeros = nullptr, const int N = 8)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x00030003;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64026402 : 0x64006400;
    int16_t const i2s_i16 = *reinterpret_cast<int16_t *>(_i2s);
    // decode 2 elems at one time.
    // interleave {e15,e13,e11,e9,e7,e5,e3,e1,e14,e12,e10,e8,e6,e4,e2,e0}
    // only decode for {x,x,x,x,e7,e5,e3,e1,x,x,x,x,e6,e4,e2,e0}
    // otherwise the pointer of _i2s should be moved to
    int i2s = (i2s_i16 & 0x00ff);
    i2s |= ((i2s_i16 & 0xff00) << 8);

#pragma unroll
    for (int i = 0; i < (N / 2); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(__pack_half2(*zeros, *zeros)));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(__pack_half2(*scale, *scale)), "r"(0));
    }
}

template <typename T1, typename T2, typename T3>
__device__ void decode_i2u_to_f16_scale_zeros_original(T1 *_i2u, T2 *B_local_decode,  T3 *scale, T3 *zeros, const int N = 8)
{
    decode_i2b_to_f16_scale_zeros_original<T1, T2, T3, false>(_i2u, B_local_decode, scale, zeros, N);
}
"""

decode_i2_to_f16_scale_zeros_rescale = """
template <typename T1, typename T2, typename T3, bool isSigned = false>
__device__ void decode_i2b_to_f16_scale_zeros_rescale(T1 *_i2s, T2 *B_local_decode, T3 *scale = nullptr, T3 *zeros = nullptr, const int N = 8)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x00030003;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64026402 : 0x64006400;
    int16_t const i2s_i16 = *reinterpret_cast<int16_t *>(_i2s);
    // decode 2 elems at one time.
    // interleave {e15,e13,e11,e9,e7,e5,e3,e1,e14,e12,e10,e8,e6,e4,e2,e0}
    // only decode for {x,x,x,x,e7,e5,e3,e1,x,x,x,x,e6,e4,e2,e0}
    // otherwise the pointer of _i2s should be moved to
    int i2s = (i2s_i16 & 0x00ff);
    i2s |= ((i2s_i16 & 0xff00) << 8);

#pragma unroll
    for (int i = 0; i < (N / 2); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(__pack_half2(*scale, *scale)), "r"(0));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(__pack_half2(*zeros, *zeros)));
    }
}

template <typename T1, typename T2, typename T3>
__device__ void decode_i2u_to_f16_scale_zeros_rescale(T1 *_i2u, T2 *B_local_decode,  T3 *scale, T3 *zeros, const int N = 8)
{
    decode_i2b_to_f16_scale_zeros_rescale<T1, T2, T3, false>(_i2u, B_local_decode, scale, zeros, N);
}
"""

decode_i2_to_f16_scale_zeros_quantized = """
template <typename T1, typename T2, typename T3, typename T4, bool isSigned = false>
__device__ void decode_i2b_to_f16_scale_zeros_quantized(T1 *_i2s, T2 *B_local_decode, const int N = 8, T3 *scale = nullptr, T4 *zeros = nullptr)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x00030003;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64016401 : 0x64006400;
    int16_t const i2s_i16 = *reinterpret_cast<int16_t *>(_i2s);
    T3 const scale_r = *scale;
    uint const packed_scales = __pack_half2(scale_r, scale_r);
    int16_t const zero_r = *((int16_t*)zeros);
    uint median_num = ((0xe400 | zero_r) << 16) | (0xe400 | zero_r);

    // decode 2 elems at one time.
    // interleave {e15,e13,e11,e9,e7,e5,e3,e1,e14,e12,e10,e8,e6,e4,e2,e0}
    // only decode for {x,x,x,x,e7,e5,e3,e1,x,x,x,x,e6,e4,e2,e0}
    // otherwise the pointer of _i2s should be moved to
    int i2s = (i2s_i16 & 0x00ff);
    i2s |= ((i2s_i16 & 0xff00) << 8);

#pragma unroll
    for (int i = 0; i < (N / 2); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("add.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(median_num));

        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales), "r"(0));
    }
}
template <typename T1, typename T2, typename T3, typename T4>
__device__ void decode_i2u_to_f16_scale_zeros_quantized(T1 *_i2u, T2 *B_local_decode, T3 *scale = nullptr, T4 *zeros = nullptr, const int N = 8)
{
    decode_i2b_to_f16_scale_zeros_quantized<T1, T2, T3, T4, false>(_i2u, B_local_decode, N, scale, zeros);
}
"""

decode_i1_to_f16 = """
/*
Kind 0: original
Kind 1: rescale
Kind 2: quantized
# documents for zeros_mode:
# original: target = (dequantize_weight - zero_point) * scale
# rescale: target = dequantize_weight * scale - zero_point
# quantized: target = (dequantize_weight - dequantize_zeros) * scale
# Notice: only support "original" and "rescale" now
zeros_mode: Literal["original", "rescale", "quantized"] = "original"
*/
template <typename T1, typename T2, bool isSigned = false, bool withScaling = false, bool withZeros = false, int ZerosKind = 1>
__device__ void decode_i1b_to_f16(T1 *_i1s, T2 *B_local_decode, const int N = 8, half *scale = nullptr, half *zeros = nullptr)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x00010001;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64006400 : 0x64006400;
    static constexpr uint TRANSFORM_SUBTRACT = 0xbc00bc00; // for signed int 2x - 1
    // interleave {e31,e29,e27,e25,e23,e21,e19,e17,e15,e13,e11,e9,e7,e5,e3,e1,e30,e28,e26,e24,e22,e20,e18,e16,e14,e12,e10,e8,e6,e4,e2,e0}
    // only decode e7,e5,e3,e1,e8,e6,e4,e2,e0
    int8_t const i1s_i16 = *reinterpret_cast<int8_t *>(_i1s);
    int i1s = (i1s_i16 & 0x0f);
    i1s |= ((i1s_i16 & 0xf0) << 12);
#pragma unroll
    // decode 2 elems at one time.
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i1s >> (1 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
        if constexpr (isSigned)
        {
            asm volatile("add.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(h[i]));
            asm volatile("add.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(TRANSFORM_SUBTRACT));
        }
        if constexpr (withZeros && ZerosKind == 0)
        {
            asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(__pack_half2(*zeros, *zeros)));
        }
        if constexpr (withScaling)
        {
            asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(__pack_half2(*scale, *scale)), "r"(0));
        }
        if constexpr (withZeros && ZerosKind == 1)
        {
            asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(__pack_half2(*zeros, *zeros)));
        }
    }
}

template <typename T1, typename T2>
__device__ void decode_i1s_to_f16(T1 *_i1s, T2 *B_local_decode, const int N = 8)
{
    decode_i1b_to_f16<T1, T2, true>(_i1s, B_local_decode, N);
}

template <typename T1, typename T2>
__device__ void decode_i1u_to_f16(T1 *_i1u, T2 *B_local_decode, const int N = 8)
{
    decode_i1b_to_f16<T1, T2, false>(_i1u, B_local_decode, N);
}
"""

decode_i1_to_f16_scale = """
template <typename T1, typename T2, typename T3>
__device__ void decode_i1u_to_f16_scale(T1 *_i1s, T2 *B_local_decode, T3 *scale = nullptr, const int N = 8)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x00010001;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = 0x64006400;
    // interleave {e31,e29,e27,e25,e23,e21,e19,e17,e15,e13,e11,e9,e7,e5,e3,e1,e30,e28,e26,e24,e22,e20,e18,e16,e14,e12,e10,e8,e6,e4,e2,e0}
    // only decode e7,e5,e3,e1,e8,e6,e4,e2,e0
    int8_t const i1s_i16 = *reinterpret_cast<int8_t *>(_i1s);
    int i1s = (i1s_i16 & 0x0f);
    i1s |= ((i1s_i16 & 0xf0) << 12);
    T3 const scale_r = *scale;
    uint const packed_scales = __pack_half2(scale_r, scale_r);
#pragma unroll
    // decode 2 elems at one time.
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i1s >> (1 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales), "r"(0));
    }
}

template <typename T1, typename T2, typename T3>
__device__ void decode_i1s_to_f16_scale(T1 *_i1s, T2 *B_local_decode, T3 *scale = nullptr, const int N = 8)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x00010001;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = 0x64006400;
    static constexpr uint TRANSFORM_SUBTRACT = 0xbc00bc00; // for signed int 2x - 1
    // interleave {e31,e29,e27,e25,e23,e21,e19,e17,e15,e13,e11,e9,e7,e5,e3,e1,e30,e28,e26,e24,e22,e20,e18,e16,e14,e12,e10,e8,e6,e4,e2,e0}
    // only decode e7,e5,e3,e1,e8,e6,e4,e2,e0

    int8_t const i1s_i16 = *reinterpret_cast<int8_t *>(_i1s);
    int i1s = (i1s_i16 & 0x0f);
    i1s |= ((i1s_i16 & 0xf0) << 12);
    T3 const scale_r = *scale;
    uint const packed_scales = __pack_half2(scale_r, scale_r);
#pragma unroll
    // decode 2 elems at one time.
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i1s >> (1 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
        asm volatile("add.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(h[i]));
        asm volatile("add.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(TRANSFORM_SUBTRACT));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales), "r"(0));
    }
}
"""

decode_i1_to_f16_scale_zeros_original = """
template <typename T1, typename T2, typename T3, typename T4, bool isSigned = false>
__device__ void decode_i1b_to_f16_zeros_original(T1 *_i1s, T2 *B_local_decode, const int N = 8, T3 *scale = nullptr, T4 *zeros = nullptr)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x00010001;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = 0x64006400;
    // interleave {e31,e29,e27,e25,e23,e21,e19,e17,e15,e13,e11,e9,e7,e5,e3,e1,e30,e28,e26,e24,e22,e20,e18,e16,e14,e12,e10,e8,e6,e4,e2,e0}
    // only decode e7,e5,e3,e1,e8,e6,e4,e2,e0
    int8_t const i1s_i16 = *reinterpret_cast<int8_t *>(_i1s);
    int i1s = (i1s_i16 & 0x0f);
    i1s |= ((i1s_i16 & 0xf0) << 12);
    T3 const scale_r = *scale;
    uint const packed_scales = __pack_half2(scale_r, scale_r);
    // input zeros maybe int32(qzeros) or half format
    T4 const zero_r = *zeros;
    uint const packed_zeros = __pack_half2(zero_r, zero_r);

#pragma unroll
    // decode 2 elems at one time.
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i1s >> (1 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_zeros));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales), "r"(0));
    }
}
template <typename T1, typename T2, typename T3, typename T4>
__device__ void decode_i1u_to_f16_scale_zeros_original(T1 *_i1u, T2 *B_local_decode, T3 *scale = nullptr, T4 *zeros = nullptr, const int N = 8)
{
    decode_i1b_to_f16_zeros_original<T1, T2, T3, T4, false>(_i1u, B_local_decode, N, scale, zeros);
}
"""

decode_i1_to_f16_scale_zeros_rescale = """
template <typename T1, typename T2, typename T3, typename T4, bool isSigned = false>
__device__ void decode_i1b_to_f16_scale_zeros_rescale(T1 *_i1s, T2 *B_local_decode, const int N = 8, T3 *scale = nullptr, T4 *zeros = nullptr)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x00010001;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = 0x64006400;
    // interleave {e31,e29,e27,e25,e23,e21,e19,e17,e15,e13,e11,e9,e7,e5,e3,e1,e30,e28,e26,e24,e22,e20,e18,e16,e14,e12,e10,e8,e6,e4,e2,e0}
    // only decode e7,e5,e3,e1,e8,e6,e4,e2,e0
    int8_t const i1s_i16 = *reinterpret_cast<int8_t *>(_i1s);
    int i1s = (i1s_i16 & 0x0f);
    i1s |= ((i1s_i16 & 0xf0) << 12);
    T3 const scale_r = *scale;
    uint const packed_scales = __pack_half2(scale_r, scale_r);
    T4 const zero_r = *zeros;
    uint const packed_zeros = 0x80008000 | __pack_half2(zero_r, zero_r);

#pragma unroll
    // decode 2 elems at one time.
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i1s >> (1 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales), "r"(packed_zeros));
    }
}

template <typename T1, typename T2, typename T3, typename T4>
__device__ void decode_i1u_to_f16_scale_zeros_rescale(T1 *_i4u, T2 *B_local_decode, T3 *scale = nullptr, T4 *zeros = nullptr, const int N = 8)
{
    decode_i1b_to_f16_scale_zeros_rescale<T1, T2, T3, T4, false>(_i4u, B_local_decode, N, scale, zeros);
}
"""

decode_i1s_to_i8s = """template <typename T1, typename T2>
__device__ void decode_i1s_to_i8s(T1 *_i1b, T2 *_i8s, const int N = 16)
{
    int i8s[4];
    // vector load
    *reinterpret_cast<int4 *>(i8s) = *reinterpret_cast<int4 *>(_i8s);
    int16_t i1b_i16 = *reinterpret_cast<int16_t *>(_i1b);
    // permutate: {e0,e4,e8,e12,e2,e6,e10,e14,e1,e5,e9,e13,e3,e7,e11,e15}
    // into: {e0,e4,e8,e12,x,x,x,x,e1,e5,e9,x,x,x,x,e13,e2,e6,e10,e14,e1,e5,e9,e13,e3,e7,e11,e15,x,x,x,x}
    int i1b = (i1b_i16 & 0x0f0f);
    i1b |= ((i1b_i16 & 0xf0f0) << 12);
    // i1b        {0..,e15,e14,e13,e12,e11,e10,e9,e8,e7,e6,e5,e4,e3,e2,e1,e0}
    // interleave {0..,e15,e13,e11,e9,e7,e5,e3,e1,e14,e12,e10,e8,e6,e4,e2,e0}
    // First, we extract the i1b and construct an intermediate fp16 number.
    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa; // 0b11101010
    static constexpr uint BOTTOM_MASK = 0x01010101;      // 0x1 -> 0b01 select 0,1
    static constexpr uint I8s_MAGIC_NUM = 0x00000000;
    static constexpr uint TRANSFORM_SUBTRACT = 0xffffffff; // for signed int 2x - 1

    for (int i = 0; i < N / 4; i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(i8s[i])
                     : "r"(i1b >> i), "n"(BOTTOM_MASK), "n"(I8s_MAGIC_NUM), "n"(immLut));
        i8s[i] = __vadd4(i8s[i], i8s[i]);
        i8s[i] = __vadd4(i8s[i], TRANSFORM_SUBTRACT);
    }
    *reinterpret_cast<int4 *>(_i8s) = *reinterpret_cast<int4 *>(i8s);
}

template <typename T1, typename T2>
__device__ void decode_i1u_to_i8s(T1 *_i1b, T2 *_i8s, const int N = 16)
{
    int *i8s = reinterpret_cast<int *>(_i8s);
    int16_t i1b_i16 = *reinterpret_cast<int16_t *>(_i1b);
    // permutate: {e0,e4,e8,e12,e2,e6,e10,e14,e1,e5,e9,e13,e3,e7,e11,e15}
    // into: {e0,e4,e8,e12,x,x,x,x,e1,e5,e9,x,x,x,x,e13,e2,e6,e10,e14,e1,e5,e9,e13,e3,e7,e11,e15,x,x,x,x}
    int i1b = (i1b_i16 & 0x0f0f);
    i1b |= ((i1b_i16 & 0xf0f0) << 12);
    // i1b        {0..,e15,e14,e13,e12,e11,e10,e9,e8,e7,e6,e5,e4,e3,e2,e1,e0}
    // interleave {0..,e15,e13,e11,e9,e7,e5,e3,e1,e14,e12,e10,e8,e6,e4,e2,e0}
    // First, we extract the i1b and construct an intermediate fp16 number.
    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa; // 0b11101010
    static constexpr uint BOTTOM_MASK = 0x01010101;      // 0x1 -> 0b01 select 0,1
    static constexpr uint I8s_MAGIC_NUM = 0x00000000;
    static constexpr uint MEDIAN_NUM = 0x00000000;

    for (int i = 0; i < N / 4; i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(i8s[i])
                     : "r"(i1b >> i), "n"(BOTTOM_MASK), "n"(I8s_MAGIC_NUM), "n"(immLut));
    }
}

"""

decode_i2s_to_i8s = """template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1 *_i2b, T2 *_i8s, const int N = 16)
{
    // convert 8 int2b_t to 8 int8b_t -> 2 int32
    uint *i8s = reinterpret_cast<uint *>(_i8s);

    // i2b = {e7,e6,e5,e4,e3,e2,e1,e0}
    // also require interleave {e7,e3,e6,e2,e5,e1,e4,e0}
    uint const i2b = *reinterpret_cast<uint *>(_i2b);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa; // 0b11101010
    static constexpr uint BOTTOM_MASK = 0x03030303;      // 0xf -> 0b11 select 0,3
    static constexpr uint I8s_MAGIC_NUM = 0x00000000;    // 1024
    static constexpr uint MEDIAN_NUM = 0x02020202;
#pragma unroll
    for (int i = 0; i < (N / 4); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(i8s[i])
                     : "r"(i2b >> (2 * i)), "n"(BOTTOM_MASK), "n"(I8s_MAGIC_NUM), "n"(immLut));
        i8s[i] = __vsub4(i8s[i], MEDIAN_NUM);
    }
}
template <typename T1, typename T2>
__device__ void decode_i2u_to_i8s(T1 *_i2b, T2 *_i8s, const int N = 16)
{
    // convert 8 int2b_t to 8 int8b_t -> 2 int32
    uint *i8s = reinterpret_cast<uint *>(_i8s);

    // i2b = {e7,e6,e5,e4,e3,e2,e1,e0}
    // also require interleave {e7,e3,e6,e2,e5,e1,e4,e0}
    uint const i2b = *reinterpret_cast<uint *>(_i2b);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa; // 0b11101010
    static constexpr uint BOTTOM_MASK = 0x03030303;      // 0xf -> 0b11 select 0,3
    static constexpr uint I8s_MAGIC_NUM = 0x00000000;    // 1024

#pragma unroll
    for (int i = 0; i < (N / 4); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(i8s[i])
                     : "r"(i2b >> (2 * i)), "n"(BOTTOM_MASK), "n"(I8s_MAGIC_NUM), "n"(immLut));
    }
}
"""

decode_i4s_to_i8s = """template <typename T1, typename T2>
__device__ void decode_i4s_to_i8s(T1 *_i4b, T2 *_i8s, const int N = 16)
{
    uint *i8s = reinterpret_cast<uint *>(_i8s);
    uint *i4b = reinterpret_cast<uint *>(_i4b);
    // First, we extract the i4s and construct an intermediate i8 number.
    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x0f0f0f0f;          // 0xf -> 0b1111 select 0,4,8,12
    static constexpr uint I4b_TO_I8s_MAGIC_NUM = 0x00000000; // 0
    static constexpr uint MEDIAN_NUM = 0x07070707;
#pragma unroll
    for (int i = 0; i < (N / 8); i++)
    {
        // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(i8s[i])
                     : "r"(i4b[0] >> (4 * i)), "n"(BOTTOM_MASK), "n"(I4b_TO_I8s_MAGIC_NUM), "n"(immLut));

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(i8s[i + 2])
                     : "r"(i4b[1] >> (4 * i)), "n"(BOTTOM_MASK), "n"(I4b_TO_I8s_MAGIC_NUM), "n"(immLut));
        i8s[i] = __vsubss4(i8s[i], MEDIAN_NUM);
        i8s[i + 2] = __vsubss4(i8s[i + 2], MEDIAN_NUM);
    }
}

template <typename T1, typename T2>
__device__ void decode_i4u_to_i8s(T1 *_i4b, T2 *_i8s, const int N = 16)
{
    uint *i8s = reinterpret_cast<uint *>(_i8s);
    uint *i4b = reinterpret_cast<uint *>(_i4b);
    // First, we extract the i4s and construct an intermediate i8 number.
    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x0f0f0f0f;          // 0xf -> 0b1111 select 0,4,8,12
    static constexpr uint I4b_TO_I8s_MAGIC_NUM = 0x00000000; // 0
#pragma unroll
    for (int i = 0; i < (N / 8); i++)
    {
        // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(i8s[i])
                     : "r"(i4b[0] >> (4 * i)), "n"(BOTTOM_MASK), "n"(I4b_TO_I8s_MAGIC_NUM), "n"(immLut));

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(i8s[i + 2])
                     : "r"(i4b[1] >> (4 * i)), "n"(BOTTOM_MASK), "n"(I4b_TO_I8s_MAGIC_NUM), "n"(immLut));
    }
}
"""

decode_i2s_to_i4s = r"""
template <typename T1, typename T2, bool isSigned>
__device__ void decode_i2b_to_i4s(T1 *_i2b, T2 *_i4s, const int N = 16)
{
    uint *i4s = reinterpret_cast<uint *>(_i4s);
    uint *i2b = reinterpret_cast<uint *>(_i2b);
    // First, we extract the i4s and construct an intermediate i8 number.
    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x33333333;          // 0xf -> 0b1111 select 0,2,4,6,8,10,12
    static constexpr uint I4b_TO_I8s_MAGIC_NUM = 0x00000000; // 0
    static constexpr uint MEDIAN_NUM = isSigned ? 0x33333333 : 0x00000000;

#pragma unroll
    for (int i = 0; i < (N / 8); i++)
    {
        // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(i4s[i])
                     : "r"(i2b[i / 2] >> (2 * (i % 2))), "n"(BOTTOM_MASK), "n"(I4b_TO_I8s_MAGIC_NUM), "n"(immLut));
        if constexpr (isSigned)
        {
            // TODO(lei): uint4 sub should be enhanced.
            // 0x03 0x03 0x03 0x03
            // i4s[i] = (((i4s[i] << 1) | i4s[i]) << 1) | i4s[i];
        }
    }
}

template <typename T1, typename T2>
__device__ void decode_i2s_to_i4s(T1 *_i4s, T2 *B_local_decode, const int N = 16)
{
    decode_i2b_to_i4s<T1, T2, true>(_i4s, B_local_decode, N);
}

template <typename T1, typename T2>
__device__ void decode_i2u_to_i4s(T1 *_i4u, T2 *B_local_decode, const int N = 16)
{
    decode_i2b_to_i4s<T1, T2, false>(_i4u, B_local_decode, N);
}
"""


def get_lop3_intrin_group(
    out_dtype: Literal[T.float16, T.int8, T.int4],
    source_format: Literal[T.int, T.uint] = T.uint,
    source_bit: int = 4,
    storage_dtype: Literal[T.int32, T.int8] = T.int8,
    with_scaling: bool = False,
    with_zeros: bool = False,
    zeros_mode: Literal["original", "rescale", "quantized"] = "original",
    storage_scope: str = "local",
) -> dict[str, str]:
    """
    This function is used to get the intrinsic group of the LOP3 operation to avoid the overhead of fast decoding.
    LOP3 is a type of logic operation that takes three inputs. The intrinsic group refers to the set of
    intrinsic operations that can be performed on these inputs. This function retrieves and returns this group.

    Parameters
    ----------
    in_dtype : Literal[T.int8]
        The data type of the input. It should be "int8".

    out_dtype : Literal[T.float16, T.int8, T.int4]
        The data type of the output. It can be either "float16" or "int8" or "int4".

    storage_nbit : int, optional
        The number of bits used for storage. By default, it is 4.

    with_scale : bool, optional
        A boolean parameter that indicates whether scaling should be applied. By default, it is False.

    with_zeros : bool, optional
        A boolean parameter that indicates whether zeros should be used. By default, it is False.

    zeros_mode : Literal["original", "rescale", "quantized"], optional
        The mode of zeros. It can be either "original", "rescale", or "quantized". By default, it is "original".

    storage_scope : Literal["local", "warp"], optional
        The scope of the storage. It can be either "local" or "warp". By default, it is "local".

    Returns
    -------
    Dict[str, str]
        A dictionary mapping the names of the intrinsics to their corresponding implementations.
    """
    out_dtype, source_format, storage_dtype = T.dtype(out_dtype), T.dtype(source_format), T.dtype(storage_dtype)
    assert out_dtype in [T.float16, T.int8, T.int4], f"Invalid out_dtype: {out_dtype}. Expected 'float16' or 'int8' or 'int4' ."

    dtype_mapping = {T.float16: "f16", T.int4: "i4", T.int8: "i8", T.int32: "i32"}
    target_dtype = dtype_mapping[out_dtype]

    if source_format not in [T.int, T.uint]:
        raise ValueError(f"Invalid source_format. Expected 'int' or 'uint', but got {source_format}, {type(source_format)}.")
    if with_zeros and source_format == T.int:
        raise ValueError(f"Zeros are not supported for signed integers, but got {source_format}")

    import_c_map = {
        "i4_to_f16": decode_i4_to_f16,
        "i2_to_f16": decode_i2_to_f16,
        "i1_to_f16": decode_i1_to_f16,
        "i4_to_f16_scale": decode_i4_to_f16_scale,
        "i4_to_f16_scale_offset": decode_i4_to_f16_scale_offset,
        "i2_to_f16_scale": decode_i2_to_f16_scale,
        "i1_to_f16_scale": decode_i1_to_f16_scale,
        "i4_to_f16_scale_zeros_original": decode_i4_to_f16_scale_zeros_original,
        "i4_to_f16_scale_zeros_original_offset": decode_i4_to_f16_scale_zeros_original_offset,
        "i2_to_f16_scale_zeros_original": decode_i2_to_f16_scale_zeros_original,
        "i1_to_f16_scale_zeros_original": decode_i1_to_f16_scale_zeros_original,
        "i4_to_f16_scale_zeros_rescale": decode_i4_to_f16_scale_zeros_rescale,
        "i4_to_f16_scale_zeros_rescale_offset": decode_i4_to_f16_scale_zeros_rescale_offset,
        "i2_to_f16_scale_zeros_rescale": decode_i2_to_f16_scale_zeros_rescale,
        "i1_to_f16_scale_zeros_rescale": decode_i1_to_f16_scale_zeros_rescale,
        "i4_to_f16_scale_zeros_quantized": decode_i4_to_f16_scale_zeros_quantized,
        "i2_to_f16_scale_zeros_quantized": decode_i2_to_f16_scale_zeros_quantized,
        "i4_to_f16_scale_zeros_quantized_offset": decode_i4_to_f16_scale_zeros_quantized_offset,
        "i1_to_i8": decode_i1s_to_i8s,
        "i2_to_i8": decode_i2s_to_i8s,
        "i4_to_i8": decode_i4s_to_i8s,
        "i2_to_i4": decode_i2s_to_i4s,
    }
    key = f"i{source_bit}_to_{target_dtype}"
    if with_scaling:
        key += "_scale"
    if with_zeros:
        key += f"_zeros_{zeros_mode}"

    is_ladder_stage3 = (storage_scope == "warp") and with_scaling
    if is_ladder_stage3:
        key += "_offset"

    if out_dtype == T.float16:
        d4f = "f16"
    elif out_dtype == T.int8:
        d4f = "i8s"
    elif out_dtype == T.int4:
        d4f = "i4s"
    else:
        raise ValueError(f"Unsupported target dtype: {target_dtype}")
    source_symbol = "u" if source_format == T.uint else "s"
    func_name = f"decode_i{source_bit}{source_symbol}_to_{d4f}"
    if with_scaling:
        func_name += "_scale"
    if with_zeros:
        func_name += f"_zeros_{zeros_mode}"
    if is_ladder_stage3:
        func_name += "_offset"

    return {
        "func_name": func_name,
        "c_source": import_c_map[key],
    }
