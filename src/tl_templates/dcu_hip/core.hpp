#ifdef __HIPCC__
#define CK_TILE_HOST inline __host__
#define CK_TILE_DEVICE inline __device__
#define CK_TILE_HOST_DEVICE inline __host__ __device__
#define CK_TILE_DEVICE_EXTERN __device__
#define CK_TILE_HOST_DEVICE_EXTERN __host__ __device__
#else
#define CK_TILE_HOST inline
#define CK_TILE_DEVICE inline
#define CK_TILE_HOST_DEVICE inline
#define CK_TILE_DEVICE_EXTERN
#define CK_TILE_HOST_DEVICE_EXTERN
#endif

#ifndef __HIP_DEVICE_COMPILE__ // for host code
#define CK_TILE_BUFFER_RESOURCE_3RD_DWORD 0xffffffff
#elif defined(__gfx803__) || defined(__gfx900__) || defined(__gfx906__) || \
    defined(__gfx9__) // for GPU code
#define CK_TILE_BUFFER_RESOURCE_3RD_DWORD 0x00020000
#elif defined(__gfx103__) // for GPU code
#define CK_TILE_BUFFER_RESOURCE_3RD_DWORD 0x31014000
#elif defined(__gfx11__) || defined(__gfx12__) // for GPU code
#define CK_TILE_BUFFER_RESOURCE_3RD_DWORD 0x31004000
#else
#define CK_TILE_BUFFER_RESOURCE_3RD_DWORD 0x31004000
#endif

namespace ck_tile{
using int32x4_t  = int32_t __attribute__((ext_vector_type(4)));
template <typename T>
CK_TILE_HOST_DEVICE constexpr T max(T x)
{
    return x;
}

template <typename T>
CK_TILE_HOST constexpr T max(T x, T y)
{
    return x > y ? x : y;
}

template <typename T>
CK_TILE_DEVICE constexpr T max(T x, T y)
{
    return x > y ? x : y;
}

template <>
CK_TILE_DEVICE float max(float x, float y)
{
    return __builtin_fmaxf(x, y); // can resultin v_max3_f32
}

template <>
CK_TILE_DEVICE double max(double x, double y)
{
    return __builtin_fmax(x, y); // maybe still v_max3_f32
}


template <typename X, typename... Ys>
CK_TILE_HOST_DEVICE constexpr auto max(X x, Ys... ys)
{
    static_assert(sizeof...(Ys) > 0, "not enough argument");
    return max(x, max(ys...));
}

template <typename T>
CK_TILE_HOST_DEVICE constexpr T min(T x)
{
    return x;
}

template <typename T>
CK_TILE_HOST constexpr T min(T x, T y)
{
    return x < y ? x : y;
}

template <typename T>
CK_TILE_DEVICE constexpr T min(T x, T y)
{
    return x < y ? x : y;
}

template <>
CK_TILE_DEVICE float min(float x, float y)
{
    return __builtin_fminf(x, y);
}

template <>
CK_TILE_DEVICE double min(double x, double y)
{
    return __builtin_fmin(x, y);
}


template <typename X, typename... Ys>
CK_TILE_HOST_DEVICE constexpr auto min(X x, Ys... ys)
{
    static_assert(sizeof...(Ys) > 0, "not enough argument");
    return min(x, min(ys...));
}
}

