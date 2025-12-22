#pragma once

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#endif

#include <cuda/atomic>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>

using cutlass::bfloat16_t;
using cutlass::half_t;

#define TL_DEVICE __forceinline__ __device__
#define TL_NOT_IMPLEMENTED()                                                   \
  {                                                                            \
    printf("%s not implemented\n", __PRETTY_FUNCTION__);                       \
    asm volatile("brkpt;\n");                                                  \
  }
template <typename T> struct normalize_atomic_type {
  using type = T;
};

template <> struct normalize_atomic_type<half_t> {
  using type = half;
};

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ > 750))
template <> struct normalize_atomic_type<bfloat16_t> {
  using type = __nv_bfloat16;
};
#endif

template <typename T1, typename T2> TL_DEVICE T1 cuda_cast(T2 val) {
  return T1(val);
}

template <> TL_DEVICE half cuda_cast<half, float>(float val) {
  return __float2half(val);
}

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ > 750))
template <> TL_DEVICE __nv_bfloat16 cuda_cast<__nv_bfloat16, float>(float val) {
  return __float2bfloat16(val);
}
#endif

template <typename T1, typename T2>
TL_DEVICE void AtomicMax(T1 *ref, T2 val,
                         int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    // There is no implementation of atomicMax for half and bf16 in cuda.
    // We simulate this process by atomicCAS loop.
    unsigned short *address_as_ushort =
        reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_ushort = *reinterpret_cast<unsigned short *>(&val);
    unsigned short old_val_ushort = *address_as_ushort;
    while (val > *reinterpret_cast<T1 *>(&old_val_ushort)) {
      unsigned short assumed_val_ushort = old_val_ushort;
      old_val_ushort =
          atomicCAS(address_as_ushort, assumed_val_ushort, val_as_ushort);
      if (assumed_val_ushort == old_val_ushort) {
        break;
      }
    }
  } else {
#if CUDART_VERSION >= 11080
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    aref.fetch_max(cuda_cast<NT1>(val), cuda::memory_order(memory_order));
#else
    TL_NOT_IMPLEMENTED();
#endif
  }
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMaxRet(T1 *ref, T2 val,
                          int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    unsigned short *address_as_ushort =
        reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_ushort = *reinterpret_cast<unsigned short *>(&val);
    unsigned short old_val_ushort = *address_as_ushort;
    while (val > *reinterpret_cast<T1 *>(&old_val_ushort)) {
      unsigned short assumed_val_ushort = old_val_ushort;
      old_val_ushort =
          atomicCAS(address_as_ushort, assumed_val_ushort, val_as_ushort);
      if (assumed_val_ushort == old_val_ushort) {
        break;
      }
    }
    return static_cast<T1>(*reinterpret_cast<T1 *>(&old_val_ushort));
  } else {
#if CUDART_VERSION >= 11080
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    return static_cast<T1>(
        aref.fetch_max(cuda_cast<NT1>(val), cuda::memory_order(memory_order)));
#else
    TL_NOT_IMPLEMENTED();
#endif
  }
}

template <typename T1, typename T2>
TL_DEVICE void AtomicMin(T1 *ref, T2 val,
                         int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    // There is no implementation of atomicMin for half and bf16 in cuda.
    // We simulate this process by atomicCAS loop.
    unsigned short *address_as_ushort =
        reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_ushort = *reinterpret_cast<unsigned short *>(&val);
    unsigned short old_val_ushort = *address_as_ushort;
    while (val < *reinterpret_cast<T1 *>(&old_val_ushort)) {
      unsigned short assumed_val_ushort = old_val_ushort;
      old_val_ushort =
          atomicCAS(address_as_ushort, assumed_val_ushort, val_as_ushort);
      if (assumed_val_ushort == old_val_ushort) {
        break;
      }
    }
  } else {
#if CUDART_VERSION >= 11080
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    aref.fetch_min(cuda_cast<NT1>(val), cuda::memory_order(memory_order));
#else
    TL_NOT_IMPLEMENTED();
#endif
  }
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMinRet(T1 *ref, T2 val,
                          int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    unsigned short *address_as_ushort =
        reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_ushort = *reinterpret_cast<unsigned short *>(&val);
    unsigned short old_val_ushort = *address_as_ushort;
    while (val < *reinterpret_cast<T1 *>(&old_val_ushort)) {
      unsigned short assumed_val_ushort = old_val_ushort;
      old_val_ushort =
          atomicCAS(address_as_ushort, assumed_val_ushort, val_as_ushort);
      if (assumed_val_ushort == old_val_ushort) {
        break;
      }
    }
    return static_cast<T1>(*reinterpret_cast<T1 *>(&old_val_ushort));
  } else {
#if CUDART_VERSION >= 11080
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    return static_cast<T1>(
        aref.fetch_min(cuda_cast<NT1>(val), cuda::memory_order(memory_order)));
#else
    TL_NOT_IMPLEMENTED();
#endif
  }
}

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ > 890))
template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1 *address, T2 val,
                         int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    if (memory_order == int(cuda::memory_order_relaxed)) {
      atomicAdd(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val));
    } else {
      // Since atomic ref do not support memory order, we need to inline ptx
      // code here for each situation
      if constexpr (std::is_same_v<NT1, half>) {
        // fp16
        __half ret_val;
        unsigned short ret_val_cast =
            *reinterpret_cast<unsigned short *>(&ret_val);
        unsigned long long ref_address =
            reinterpret_cast<unsigned long long>(address);
        unsigned short val_cast = *reinterpret_cast<unsigned short *>(&val);
        if (memory_order == int(cuda::memory_order_release) ||
            memory_order == int(cuda::memory_order_consume)) {
          asm volatile("atom.release.gpu.global.add.noftz.f16 %0, [%1], %2;"
                       : "=h"(ret_val_cast)
                       : "l"(ref_address), "h"(val_cast)
                       : "memory");
        } else if (memory_order == int(cuda::memory_order_acquire)) {
          asm volatile("atom.acquire.gpu.global.add.noftz.f16 %0, [%1], %2;"
                       : "=h"(ret_val_cast)
                       : "l"(ref_address), "h"(val_cast)
                       : "memory");
        } else if (memory_order == int(cuda::memory_order_acq_rel) ||
                   memory_order == int(cuda::memory_order_seq_cst)) {
          asm volatile("atom.acq_rel.gpu.global.add.noftz.f16 %0, [%1], %2;"
                       : "=h"(ret_val_cast)
                       : "l"(ref_address), "h"(val_cast)
                       : "memory");
        }
      } else if constexpr (std::is_same_v<NT1, __nv_bfloat16>) {
        // bf16
        __nv_bfloat16 ret_val;
        unsigned short ret_val_cast =
            *reinterpret_cast<unsigned short *>(&ret_val);
        unsigned long long ref_address =
            reinterpret_cast<unsigned long long>(address);
        unsigned short val_cast = *reinterpret_cast<unsigned short *>(&val);
        if (memory_order == int(cuda::memory_order_release) ||
            memory_order == int(cuda::memory_order_consume)) {
          asm volatile("atom.release.gpu.global.add.noftz.bf16 %0, [%1], %2;"
                       : "=h"(ret_val_cast)
                       : "l"(ref_address), "h"(val_cast)
                       : "memory");
        } else if (memory_order == int(cuda::memory_order_acquire)) {
          asm volatile("atom.acquire.gpu.global.add.noftz.bf16 %0, [%1], %2;"
                       : "=h"(ret_val_cast)
                       : "l"(ref_address), "h"(val_cast)
                       : "memory");
        } else if (memory_order == int(cuda::memory_order_acq_rel) ||
                   memory_order == int(cuda::memory_order_seq_cst)) {
          asm volatile("atom.acq_rel.gpu.global.add.noftz.bf16 %0, [%1], %2;"
                       : "=h"(ret_val_cast)
                       : "l"(ref_address), "h"(val_cast)
                       : "memory");
        }
      }
    }
  } else {
    atomicAdd(reinterpret_cast<NT1 *>(address), cuda_cast<NT1>(val));
  }
}
#else
template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1 *address, T2 val,
                         int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  (void)memory_order;
  atomicAdd(reinterpret_cast<NT1 *>(address), cuda_cast<NT1>(val));
}
#endif

template <typename T1, typename T2>
TL_DEVICE T1 AtomicAddRet(T1 *address, T2 val,
                          int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    if (memory_order == int(cuda::memory_order_relaxed)) {
      return static_cast<T1>(
          atomicAdd(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val)));
    } else {
      if constexpr (std::is_same_v<NT1, half>) {
        // fp16
        __half ret_val;
        unsigned short ret_val_cast =
            *reinterpret_cast<unsigned short *>(&ret_val);
        unsigned long long ref_address =
            reinterpret_cast<unsigned long long>(address);
        unsigned short val_cast = *reinterpret_cast<unsigned short *>(&val);
        if (memory_order == int(cuda::memory_order_release) ||
            memory_order == int(cuda::memory_order_consume)) {
          asm volatile("atom.release.gpu.global.add.noftz.f16 %0, [%1], %2;"
                       : "=h"(ret_val_cast)
                       : "l"(ref_address), "h"(val_cast)
                       : "memory");
        } else if (memory_order == int(cuda::memory_order_acquire)) {
          asm volatile("atom.acquire.gpu.global.add.noftz.f16 %0, [%1], %2;"
                       : "=h"(ret_val_cast)
                       : "l"(ref_address), "h"(val_cast)
                       : "memory");
        } else if (memory_order == int(cuda::memory_order_acq_rel) ||
                   memory_order == int(cuda::memory_order_seq_cst)) {
          asm volatile("atom.acq_rel.gpu.global.add.noftz.f16 %0, [%1], %2;"
                       : "=h"(ret_val_cast)
                       : "l"(ref_address), "h"(val_cast)
                       : "memory");
        }
        return static_cast<T1>(*reinterpret_cast<__half *>(&ret_val_cast));
      } else if constexpr (std::is_same_v<NT1, __nv_bfloat16>) {
        // bf16
        __nv_bfloat16 ret_val;
        unsigned short ret_val_cast =
            *reinterpret_cast<unsigned short *>(&ret_val);
        unsigned long long ref_address =
            reinterpret_cast<unsigned long long>(address);
        unsigned short val_cast = *reinterpret_cast<unsigned short *>(&val);
        if (memory_order == int(cuda::memory_order_release) ||
            memory_order == int(cuda::memory_order_consume)) {
          asm volatile("atom.release.gpu.global.add.noftz.bf16 %0, [%1], %2;"
                       : "=h"(ret_val_cast)
                       : "l"(ref_address), "h"(val_cast)
                       : "memory");
        } else if (memory_order == int(cuda::memory_order_acquire)) {
          asm volatile("atom.acquire.gpu.global.add.noftz.bf16 %0, [%1], %2;"
                       : "=h"(ret_val_cast)
                       : "l"(ref_address), "h"(val_cast)
                       : "memory");
        } else if (memory_order == int(cuda::memory_order_acq_rel) ||
                   memory_order == int(cuda::memory_order_seq_cst)) {
          asm volatile("atom.acq_rel.gpu.global.add.noftz.bf16 %0, [%1], %2;"
                       : "=h"(ret_val_cast)
                       : "l"(ref_address), "h"(val_cast)
                       : "memory");
        }
        return static_cast<T1>(
            *reinterpret_cast<__nv_bfloat16 *>(&ret_val_cast));
      }
    }
  } else {
#if CUDART_VERSION >= 11080
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    return static_cast<T1>(
        aref.fetch_add(cuda_cast<NT1>(val), cuda::memory_order(memory_order)));
#else
    TL_NOT_IMPLEMENTED();
#endif
  }
}

// TODO add memory_order for vectorized atomic add
TL_DEVICE void AtomicAddx2(half_t *ref, half_t *val,
                           int memory_order = int(cuda::memory_order_relaxed)) {
  if (memory_order == int(cuda::memory_order_relaxed)) {
    atomicAdd(reinterpret_cast<half2 *>(ref),
              static_cast<half2>(*reinterpret_cast<half2 *>(val)));
  } else {
    // Since atomicAdd does not support memory order, atomic_ref does not
    // support vectorized atomic operation we can only inline ptx code here
    // Note: Vectorized atomic operations only support global space
    // Note: for 16-bit value, we need to reinterpret_cast the value to unsigned
    // short and use "h" register in assembly
    __half2 add_val = *reinterpret_cast<__half2 *>(val);
    unsigned short add_val_x_cast =
        *reinterpret_cast<unsigned short *>(&add_val.x);
    unsigned short add_val_y_cast =
        *reinterpret_cast<unsigned short *>(&add_val.y);
    unsigned long long ref_addr = reinterpret_cast<unsigned long long>(ref);
    __half ret_val_x, ret_val_y;
    unsigned short ret_val_x_cast =
        *reinterpret_cast<unsigned short *>(&ret_val_x);
    unsigned short ret_val_y_cast =
        *reinterpret_cast<unsigned short *>(&ret_val_y);
    if (memory_order == int(cuda::memory_order_release) ||
        memory_order == int(cuda::memory_order_consume)) {
      asm volatile(
          "atom.release.gpu.global.add.noftz.v2.f16 {%0,%1}, [%2], {%3,%4};"
          : "=h"(ret_val_x_cast), "=h"(ret_val_y_cast)
          : "l"(ref_addr), "h"(add_val_x_cast), "h"(add_val_y_cast)
          : "memory");
    } else if (memory_order == int(cuda::memory_order_acquire)) {
      asm volatile(
          "atom.acquire.gpu.global.add.noftz.v2.f16 {%0,%1}, [%2], {%3,%4};"
          : "=h"(ret_val_x_cast), "=h"(ret_val_y_cast)
          : "l"(ref_addr), "h"(add_val_x_cast), "h"(add_val_y_cast)
          : "memory");
    } else if (memory_order == int(cuda::memory_order_acq_rel) ||
               memory_order == int(cuda::memory_order_seq_cst)) {
      asm volatile(
          "atom.acq_rel.gpu.global.add.noftz.v2.f16 {%0,%1}, [%2], {%3,%4};"
          : "=h"(ret_val_x_cast), "=h"(ret_val_y_cast)
          : "l"(ref_addr), "h"(add_val_x_cast), "h"(add_val_y_cast)
          : "memory");
    }
  }
}

TL_DEVICE half2
AtomicAddx2Ret(half_t *ref, half_t *val,
               int memory_order = int(cuda::memory_order_relaxed)) {
  if (memory_order == int(cuda::memory_order_relaxed)) {
    return atomicAdd(reinterpret_cast<half2 *>(ref),
                     static_cast<half2>(*reinterpret_cast<half2 *>(val)));
  } else {
    __half2 add_val = *reinterpret_cast<__half2 *>(val);
    unsigned short add_val_x_cast =
        *reinterpret_cast<unsigned short *>(&add_val.x);
    unsigned short add_val_y_cast =
        *reinterpret_cast<unsigned short *>(&add_val.y);
    unsigned long long ref_addr = reinterpret_cast<unsigned long long>(ref);
    __half ret_val_x, ret_val_y;
    unsigned short ret_val_x_cast =
        *reinterpret_cast<unsigned short *>(&ret_val_x);
    unsigned short ret_val_y_cast =
        *reinterpret_cast<unsigned short *>(&ret_val_y);
    if (memory_order == int(cuda::memory_order_release) ||
        memory_order == int(cuda::memory_order_consume)) {
      asm volatile(
          "atom.release.gpu.global.add.noftz.v2.f16 {%0,%1}, [%2], {%3,%4};"
          : "=h"(ret_val_x_cast), "=h"(ret_val_y_cast)
          : "l"(ref_addr), "h"(add_val_x_cast), "h"(add_val_y_cast)
          : "memory");
    } else if (memory_order == int(cuda::memory_order_acquire)) {
      asm volatile(
          "atom.acquire.gpu.global.add.noftz.v2.f16 {%0,%1}, [%2], {%3,%4};"
          : "=h"(ret_val_x_cast), "=h"(ret_val_y_cast)
          : "l"(ref_addr), "h"(add_val_x_cast), "h"(add_val_y_cast)
          : "memory");
    } else if (memory_order == int(cuda::memory_order_acq_rel) ||
               memory_order == int(cuda::memory_order_seq_cst)) {
      asm volatile(
          "atom.acq_rel.gpu.global.add.noftz.v2.f16 {%0,%1}, [%2], {%3,%4};"
          : "=h"(ret_val_x_cast), "=h"(ret_val_y_cast)
          : "l"(ref_addr), "h"(add_val_x_cast), "h"(add_val_y_cast)
          : "memory");
    }
    return half2(*reinterpret_cast<__half *>(&ret_val_x_cast),
                 *reinterpret_cast<__half *>(&ret_val_y_cast));
  }
}

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ > 750))
TL_DEVICE void AtomicAddx2(bfloat16_t *ref, bfloat16_t *val,
                           int memory_order = int(cuda::memory_order_relaxed)) {
  if (memory_order == int(cuda::memory_order_relaxed)) {
    atomicAdd(
        reinterpret_cast<__nv_bfloat162 *>(ref),
        static_cast<__nv_bfloat162>(*reinterpret_cast<__nv_bfloat162 *>(val)));
  } else {
    __nv_bfloat162 add_val = *reinterpret_cast<__nv_bfloat162 *>(val);
    unsigned short add_val_x_cast =
        *reinterpret_cast<unsigned short *>(&add_val.x);
    unsigned short add_val_y_cast =
        *reinterpret_cast<unsigned short *>(&add_val.y);
    unsigned long long ref_addr = reinterpret_cast<unsigned long long>(ref);
    __nv_bfloat162 ret_val;
    unsigned short ret_val_x_cast =
        *reinterpret_cast<unsigned short *>(&ret_val.x);
    unsigned short ret_val_y_cast =
        *reinterpret_cast<unsigned short *>(&ret_val.y);
    if (memory_order == int(cuda::memory_order_release) ||
        memory_order == int(cuda::memory_order_consume)) {
      asm volatile("atom.release.gpu.global.add.v2.bf16 {%0,%1}, [%2], {%3,%4};"
                   : "=h"(ret_val_x_cast), "=h"(ret_val_y_cast)
                   : "l"(ref_addr), "h"(add_val_x_cast), "h"(add_val_y_cast)
                   : "memory");
    } else if (memory_order == int(cuda::memory_order_acquire)) {
      asm volatile("atom.acquire.gpu.global.add.v2.bf16 {%0,%1}, [%2], {%3,%4};"
                   : "=h"(ret_val_x_cast), "=h"(ret_val_y_cast)
                   : "l"(ref_addr), "h"(add_val_x_cast), "h"(add_val_y_cast)
                   : "memory");
    } else if (memory_order == int(cuda::memory_order_acq_rel) ||
               memory_order == int(cuda::memory_order_seq_cst)) {
      asm volatile("atom.acq_rel.gpu.global.add.v2.bf16 {%0,%1}, [%2], {%3,%4};"
                   : "=h"(ret_val_x_cast), "=h"(ret_val_y_cast)
                   : "l"(ref_addr), "h"(add_val_x_cast), "h"(add_val_y_cast)
                   : "memory");
    }
  }
}

TL_DEVICE __nv_bfloat162
AtomicAddx2Ret(bfloat16_t *ref, bfloat16_t *val,
               int memory_order = int(cuda::memory_order_relaxed)) {
  if (memory_order == int(cuda::memory_order_relaxed)) {
    return atomicAdd(
        reinterpret_cast<__nv_bfloat162 *>(ref),
        static_cast<__nv_bfloat162>(*reinterpret_cast<__nv_bfloat162 *>(val)));
  } else {
    __nv_bfloat162 add_val = *reinterpret_cast<__nv_bfloat162 *>(val);
    unsigned short add_val_x_cast =
        *reinterpret_cast<unsigned short *>(&add_val.x);
    unsigned short add_val_y_cast =
        *reinterpret_cast<unsigned short *>(&add_val.y);
    unsigned long long ref_addr = reinterpret_cast<unsigned long long>(ref);
    __nv_bfloat162 ret_val;
    unsigned short ret_val_x_cast =
        *reinterpret_cast<unsigned short *>(&ret_val.x);
    unsigned short ret_val_y_cast =
        *reinterpret_cast<unsigned short *>(&ret_val.y);
    if (memory_order == int(cuda::memory_order_release) ||
        memory_order == int(cuda::memory_order_consume)) {
      asm volatile("atom.release.gpu.global.add.v2.bf16 {%0,%1}, [%2], {%3,%4};"
                   : "=h"(ret_val_x_cast), "=h"(ret_val_y_cast)
                   : "l"(ref_addr), "h"(add_val_x_cast), "h"(add_val_y_cast)
                   : "memory");
    } else if (memory_order == int(cuda::memory_order_acquire)) {
      asm volatile("atom.acquire.gpu.global.add.v2.bf16 {%0,%1}, [%2], {%3,%4};"
                   : "=h"(ret_val_x_cast), "=h"(ret_val_y_cast)
                   : "l"(ref_addr), "h"(add_val_x_cast), "h"(add_val_y_cast)
                   : "memory");
    } else if (memory_order == int(cuda::memory_order_acq_rel) ||
               memory_order == int(cuda::memory_order_seq_cst)) {
      asm volatile("atom.acq_rel.gpu.global.add.v2.bf16 {%0,%1}, [%2], {%3,%4};"
                   : "=h"(ret_val_x_cast), "=h"(ret_val_y_cast)
                   : "l"(ref_addr), "h"(add_val_x_cast), "h"(add_val_y_cast)
                   : "memory");
    }
    return __nv_bfloat162(*reinterpret_cast<__nv_bfloat16 *>(&ret_val_x_cast),
                          *reinterpret_cast<__nv_bfloat16 *>(&ret_val_y_cast));
  }
}
#endif

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 900))
TL_DEVICE void AtomicAddx2(float *ref, float *val,
                           int memory_order = int(cuda::memory_order_relaxed)) {
  if (memory_order == int(cuda::memory_order_relaxed)) {
    atomicAdd(reinterpret_cast<float2 *>(ref),
              static_cast<float2>(*reinterpret_cast<float2 *>(val)));
  } else {
    float2 add_val = *reinterpret_cast<float2 *>(val);
    unsigned long long ref_addr = reinterpret_cast<unsigned long long>(ref);
    float2 ret_val;
    if (memory_order == int(cuda::memory_order_release) ||
        memory_order == int(cuda::memory_order_consume)) {
      asm volatile("atom.release.gpu.global.add.v2.f32 {%0,%1}, [%2], {%3,%4};"
                   : "=f"(ret_val.x), "=f"(ret_val.y)
                   : "l"(ref_addr), "f"(add_val.x), "f"(add_val.y)
                   : "memory");
    } else if (memory_order == int(cuda::memory_order_acquire)) {
      asm volatile("atom.acquire.gpu.global.add.v2.f32 {%0,%1}, [%2], {%3,%4};"
                   : "=f"(ret_val.x), "=f"(ret_val.y)
                   : "l"(ref_addr), "f"(add_val.x), "f"(add_val.y)
                   : "memory");
    } else if (memory_order == int(cuda::memory_order_acq_rel) ||
               memory_order == int(cuda::memory_order_seq_cst)) {
      asm volatile("atom.acq_rel.gpu.global.add.v2.f32 {%0,%1}, [%2], {%3,%4};"
                   : "=f"(ret_val.x), "=f"(ret_val.y)
                   : "l"(ref_addr), "f"(add_val.x), "f"(add_val.y)
                   : "memory");
    }
  }
}

TL_DEVICE float2
AtomicAddx2Ret(float *ref, float *val,
               int memory_order = int(cuda::memory_order_relaxed)) {
  if (memory_order == int(cuda::memory_order_relaxed)) {
    return atomicAdd(reinterpret_cast<float2 *>(ref),
                     static_cast<float2>(*reinterpret_cast<float2 *>(val)));
  } else {
    float2 add_val = *reinterpret_cast<float2 *>(val);
    unsigned long long ref_addr = reinterpret_cast<unsigned long long>(ref);
    float2 ret_val;
    if (memory_order == int(cuda::memory_order_release) ||
        memory_order == int(cuda::memory_order_consume)) {
      asm volatile("atom.release.gpu.global.add.v2.f32 {%0,%1}, [%2], {%3,%4};"
                   : "=f"(ret_val.x), "=f"(ret_val.y)
                   : "l"(ref_addr), "f"(add_val.x), "f"(add_val.y)
                   : "memory");
    } else if (memory_order == int(cuda::memory_order_acquire)) {
      asm volatile("atom.acquire.gpu.global.add.v2.f32 {%0,%1}, [%2], {%3,%4};"
                   : "=f"(ret_val.x), "=f"(ret_val.y)
                   : "l"(ref_addr), "f"(add_val.x), "f"(add_val.y)
                   : "memory");
    } else if (memory_order == int(cuda::memory_order_acq_rel) ||
               memory_order == int(cuda::memory_order_seq_cst)) {
      asm volatile("atom.acq_rel.gpu.global.add.v2.f32 {%0,%1}, [%2], {%3,%4};"
                   : "=f"(ret_val.x), "=f"(ret_val.y)
                   : "l"(ref_addr), "f"(add_val.x), "f"(add_val.y)
                   : "memory");
    }
    return ret_val;
  }
}

TL_DEVICE void AtomicAddx4(float *ref, float *val,
                           int memory_order = int(cuda::memory_order_relaxed)) {
  if (memory_order == int(cuda::memory_order_relaxed)) {
    atomicAdd(reinterpret_cast<float4 *>(ref),
              static_cast<float4>(*reinterpret_cast<float4 *>(val)));
  } else {
    // Since atomicAdd does not support memory order, atomic_ref does not
    // support vectorized atomic operation we can only inline ptx code here
    // Note: Vectorized atomic operations only support global space
    float4 add_val = *reinterpret_cast<float4 *>(val);
    unsigned long long ref_addr = reinterpret_cast<unsigned long long>(ref);
    float4 ret_val;
    if (memory_order == int(cuda::memory_order_release) ||
        memory_order == int(cuda::memory_order_consume)) {
      asm volatile("atom.release.gpu.global.add.v4.f32 {%0,%1,%2,%3}, [%4], "
                   "{%5,%6,%7,%8};"
                   : "=f"(ret_val.x), "=f"(ret_val.y), "=f"(ret_val.z),
                     "=f"(ret_val.w)
                   : "l"(ref_addr), "f"(add_val.x), "f"(add_val.y),
                     "f"(add_val.z), "f"(add_val.w)
                   : "memory");
    } else if (memory_order == int(cuda::memory_order_acquire)) {
      asm volatile("atom.acquire.gpu.global.add.v4.f32 {%0,%1,%2,%3}, [%4], "
                   "{%5,%6,%7,%8};"
                   : "=f"(ret_val.x), "=f"(ret_val.y), "=f"(ret_val.z),
                     "=f"(ret_val.w)
                   : "l"(ref_addr), "f"(add_val.x), "f"(add_val.y),
                     "f"(add_val.z), "f"(add_val.w)
                   : "memory");
    } else if (memory_order == int(cuda::memory_order_acq_rel) ||
               memory_order == int(cuda::memory_order_seq_cst)) {
      asm volatile("atom.acq_rel.gpu.global.add.v4.f32 {%0,%1,%2,%3}, [%4], "
                   "{%5,%6,%7,%8};"
                   : "=f"(ret_val.x), "=f"(ret_val.y), "=f"(ret_val.z),
                     "=f"(ret_val.w)
                   : "l"(ref_addr), "f"(add_val.x), "f"(add_val.y),
                     "f"(add_val.z), "f"(add_val.w)
                   : "memory");
    }
  }
}

TL_DEVICE float4
AtomicAddx4Ret(float *ref, float *val,
               int memory_order = int(cuda::memory_order_relaxed)) {
  if (memory_order == int(cuda::memory_order_relaxed)) {
    return atomicAdd(reinterpret_cast<float4 *>(ref),
                     static_cast<float4>(*reinterpret_cast<float4 *>(val)));
  } else {
    float4 add_val = *reinterpret_cast<float4 *>(val);
    unsigned long long ref_addr = reinterpret_cast<unsigned long long>(ref);
    float4 ret_val;
    if (memory_order == int(cuda::memory_order_release) ||
        memory_order == int(cuda::memory_order_consume)) {
      asm volatile("atom.global.gpu.release.add.v4.f32 {%0,%1,%2,%3}, [%4], "
                   "{%5,%6,%7,%8};"
                   : "=f"(ret_val.x), "=f"(ret_val.y), "=f"(ret_val.z),
                     "=f"(ret_val.w)
                   : "l"(ref_addr), "f"(add_val.x), "f"(add_val.y),
                     "f"(add_val.z), "f"(add_val.w)
                   : "memory");
    } else if (memory_order == int(cuda::memory_order_acquire)) {
      asm volatile("atom.global.gpu.acquire.add.v4.f32 {%0,%1,%2,%3}, [%4], "
                   "{%5,%6,%7,%8};"
                   : "=f"(ret_val.x), "=f"(ret_val.y), "=f"(ret_val.z),
                     "=f"(ret_val.w)
                   : "l"(ref_addr), "f"(add_val.x), "f"(add_val.y),
                     "f"(add_val.z), "f"(add_val.w)
                   : "memory");
    } else if (memory_order == int(cuda::memory_order_acq_rel) ||
               memory_order == int(cuda::memory_order_seq_cst)) {
      asm volatile("atom.global.gpu.acq_rel.add.v4.f32 {%0,%1,%2,%3}, [%4], "
                   "{%5,%6,%7,%8};"
                   : "=f"(ret_val.x), "=f"(ret_val.y), "=f"(ret_val.z),
                     "=f"(ret_val.w)
                   : "l"(ref_addr), "f"(add_val.x), "f"(add_val.y),
                     "f"(add_val.z), "f"(add_val.w)
                   : "memory");
    }
    return ret_val;
  }
}
#else
TL_DEVICE void AtomicAddx2(float *ref, float *val,
                           int memory_order = int(cuda::memory_order_relaxed)) {
  (void)memory_order;
  float2 add_val = *reinterpret_cast<float2 *>(val);
  atomicAdd(ref + 0, add_val.x);
  atomicAdd(ref + 1, add_val.y);
}

TL_DEVICE float2
AtomicAddx2Ret(float *ref, float *val,
               int memory_order = int(cuda::memory_order_relaxed)) {
  (void)memory_order;
  float2 add_val = *reinterpret_cast<float2 *>(val);
  float2 ret;
  ret.x = atomicAdd(ref + 0, add_val.x);
  ret.y = atomicAdd(ref + 1, add_val.y);
  return ret;
}

TL_DEVICE void AtomicAddx4(float *ref, float *val,
                           int memory_order = int(cuda::memory_order_relaxed)) {
  (void)memory_order;
  float4 add_val = *reinterpret_cast<float4 *>(val);
  atomicAdd(ref + 0, add_val.x);
  atomicAdd(ref + 1, add_val.y);
  atomicAdd(ref + 2, add_val.z);
  atomicAdd(ref + 3, add_val.w);
}

TL_DEVICE float4
AtomicAddx4Ret(float *ref, float *val,
               int memory_order = int(cuda::memory_order_relaxed)) {
  (void)memory_order;
  float4 add_val = *reinterpret_cast<float4 *>(val);
  float4 ret;
  ret.x = atomicAdd(ref + 0, add_val.x);
  ret.y = atomicAdd(ref + 1, add_val.y);
  ret.z = atomicAdd(ref + 2, add_val.z);
  ret.w = atomicAdd(ref + 3, add_val.w);
  return ret;
}
#endif

template <typename T> TL_DEVICE T AtomicLoad(T *ref, int memory_order) {
#if CUDART_VERSION >= 11080
  cuda::atomic_ref<T, cuda::thread_scope_device> aref(*ref);
  return aref.load(cuda::memory_order(memory_order));
#else
  TL_NOT_IMPLEMENTED();
#endif
}

template <typename T1, typename T2>
TL_DEVICE void AtomicStore(T1 *ref, T2 value, int memory_order) {
  using NT1 = typename normalize_atomic_type<T1>::type;
#if CUDART_VERSION >= 11080
  cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*ref);
  aref.store(cuda_cast<NT1>(value), cuda::memory_order(memory_order));
#else
  TL_NOT_IMPLEMENTED();
#endif
}
