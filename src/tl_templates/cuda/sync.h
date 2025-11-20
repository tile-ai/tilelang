#pragma once

#include "common.h"

#define IS_MASTER_THREAD()                                                     \
  (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
#define IS_MASTER_BLOCK()                                                      \
  (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)

#define BARRIER_MAGIC 0x80000000

namespace tl {

// Triggers a GPU trap for debugging
TL_DEVICE void trap() { asm("trap;\n"); }

// CTA-level memory fence
TL_DEVICE void memory_fence_cta() {
  asm volatile("fence.acq_rel.cta;\n" ::: "memory");
}

// GPU-level memory fence
TL_DEVICE void memory_fence_gpu() {
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
}

// System-level memory fence
TL_DEVICE void memory_fence_sys() {
  asm volatile("fence.acq_rel.sys;\n" ::: "memory");
}

// GPU-level load with acquire semantics
TL_DEVICE uint32_t ld_acquire_gpu_u32(const uint32_t *ptr) {
  uint32_t ret;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(ret) : "l"(ptr));
  return ret;
}

// GPU-level atomic add with release semantics
TL_DEVICE uint32_t atomic_add_release_gpu_u32(const uint32_t *ptr,
                                              uint32_t value) {
  uint32_t ret;
  asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;\n"
               : "=r"(ret)
               : "l"(ptr), "r"(value));
  return ret;
}

// System-level atomic load with acquire semantics
TL_DEVICE int atomic_load_acquire_sys_s32(const int *ptr) {
  int ret;
  asm volatile("atom.load.acquire.sys.global.s32 %0, [%1];\n"
               : "=r"(ret)
               : "l"(ptr));
  return ret;
}

TL_DEVICE int ld_volatile_global_s32(const int *ptr) {
  int ret;
  asm volatile("ld.volatile.global.s32 %0, [%1];\n" : "=r"(ret) : "l"(ptr));
  return ret;
}

TL_DEVICE int ld_acquire(const int *ptr) {
  int ret = 0;
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(ret) : "l"(ptr));
  return ret;
}

// Initialize a GPU barrier
template <const uint32_t kExpected>
TL_DEVICE void init_barrier_gpu(uint32_t *barrier) {
  if (IS_MASTER_BLOCK() && IS_MASTER_THREAD()) {
    *barrier = BARRIER_MAGIC - kExpected;
  }
  memory_fence_gpu(); // TODO: Is fence or sync needed here?
}

// Arrive at a GPU barrier (atomic increment)
TL_DEVICE void arrive_barrier_gpu(uint32_t *barrier) {
  memory_fence_gpu();
  if (IS_MASTER_THREAD()) {
    atomic_add_release_gpu_u32(barrier, 1);
  }
}

// Wait at a GPU barrier until all expected blocks have arrived
TL_DEVICE void wait_barrier_gpu(uint32_t *barrier) {
  if (IS_MASTER_THREAD()) {
    uint32_t arrive = ld_acquire_gpu_u32(barrier);
    while (!(arrive & BARRIER_MAGIC)) {
      arrive = ld_acquire_gpu_u32(barrier);
    }
  }
  __syncthreads();
}

// Synchronize at a GPU barrier (arrive + wait)
TL_DEVICE void sync_barrier_gpu(uint32_t *barrier) {
  // memory_fence_gpu();
  __syncthreads();
  if (IS_MASTER_THREAD()) {
    atomic_add_release_gpu_u32(barrier, 1);
    uint32_t arrive = ld_acquire_gpu_u32(barrier);
    while (arrive < BARRIER_MAGIC) {
      arrive = ld_acquire_gpu_u32(barrier);
    }
  }
  __syncthreads();
}

// cooperative groups version of GPU barrier arrive
TL_DEVICE unsigned int sync_grids_arrive(uint32_t *barrier) {
  unsigned int oldArrive = 0;

  __syncthreads();

  if (IS_MASTER_THREAD()) {
    unsigned int expected = gridDim.x * gridDim.y * gridDim.z;
    unsigned int nb = 1;
    if (IS_MASTER_BLOCK()) {
      nb = 0x80000000 - (expected - 1);
    }
    asm volatile("atom.add.release.gpu.u32 %0,[%1],%2;"
                 : "=r"(oldArrive)
                 : "l"((unsigned int *)barrier), "r"(nb)
                 : "memory");
  }

  return oldArrive;
}

// cooperative groups version of GPU barrier arrive
TL_DEVICE void sync_grids_wait(unsigned int oldArrive, uint32_t *barrier) {
  if (IS_MASTER_THREAD()) {
    unsigned int current_arrive;
    do {
      asm volatile("ld.acquire.gpu.u32 %0,[%1];"
                   : "=r"(current_arrive)
                   : "l"((unsigned int *)barrier)
                   : "memory");
    } while (!(((oldArrive ^ current_arrive) & 0x80000000) != 0));
  }
  __syncthreads();
}

TL_DEVICE void sync_grid(uint32_t *barrier) {
  unsigned int token = sync_grids_arrive(barrier);
  sync_grids_wait(token, barrier);
}

// Synchronize all blocks at a system-level barrier
// TODO(wt): Add sync-only option and timeout handling

TL_DEVICE void barrier_all_blocks_sys(int offset, int rank, int num_ranks) {
// Macro to compute the barrier pointer for a given target rank
#define BARRIER_PTR(tgt_rank)                                                  \
  (reinterpret_cast<int32_t *>(get_remote_base_ptr(tgt_rank) + offset))

  memory_fence_sys();
  __syncthreads();

  int tid = threadIdx.x;
  if (tid < num_ranks) {
    atomicAdd_system(BARRIER_PTR(rank) + tid, 1);
    atomicAdd_system(BARRIER_PTR(tid) + rank, -1);
  }

  while (true) {
    int value =
        tid < num_ranks ? ld_volatile_global_s32(BARRIER_PTR(rank) + tid) : 0;
    if (__all_sync(0xffffffff, value <= 0)) {
      break;
    }
  }
  __syncthreads();

#undef BARRIER_PTR
}

template <typename T> TL_DEVICE void wait_eq(void *barrier, T val = 1) {
  T *flag_ptr = reinterpret_cast<T *>(barrier);
// Spin-loop
#pragma unroll 1
  while (ld_acquire(flag_ptr) != val) {
  }
}

template <typename P, typename T>
TL_DEVICE void st_release_gpu(P ptr, T value) {
  static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
  static_assert(std::is_pointer_v<P> || std::is_same_v<P, uint64_t>);
  T *ptr_ = reinterpret_cast<T *>(ptr);

  if constexpr (sizeof(T) == 2) {
    asm volatile("st.release.gpu.global.b16 [%0], %1;"
                 :
                 : "l"(ptr_), "h"(value)
                 : "memory");
  } else if constexpr (sizeof(T) == 4) {
    if constexpr (std::is_floating_point_v<T>) {
      asm volatile("st.release.gpu.global.b32 [%0], %1;"
                   :
                   : "l"(ptr_), "f"(value)
                   : "memory");
    } else {
      asm volatile("st.release.gpu.global.b32 [%0], %1;"
                   :
                   : "l"(ptr_), "r"(value)
                   : "memory");
    }
  } else {
    if constexpr (std::is_floating_point_v<T>) {
      asm volatile("st.release.gpu.global.b64 [%0], %1;"
                   :
                   : "l"(ptr_), "d"(value)
                   : "memory");
    } else {
      asm volatile("st.release.gpu.global.b64 [%0], %1;"
                   :
                   : "l"(ptr_), "l"(value)
                   : "memory");
    }
  }
}

template <typename P, typename T>
TL_DEVICE void st_relaxed_gpu(P ptr, T value) {
  static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
  static_assert(std::is_pointer_v<P> || std::is_same_v<P, uint64_t>);
  T *ptr_ = reinterpret_cast<T *>(ptr);

  if constexpr (sizeof(T) == 2) {
    asm volatile("st.relaxed.gpu.global.b16 [%0], %1;"
                 :
                 : "l"(ptr_), "h"(value)
                 : "memory");
  } else if constexpr (sizeof(T) == 4) {
    if constexpr (std::is_floating_point_v<T>) {
      asm volatile("st.relaxed.gpu.global.b32 [%0], %1;"
                   :
                   : "l"(ptr_), "f"(value)
                   : "memory");
    } else {
      asm volatile("st.relaxed.gpu.global.b32 [%0], %1;"
                   :
                   : "l"(ptr_), "r"(value)
                   : "memory");
    }
  } else {
    if constexpr (std::is_floating_point_v<T>) {
      asm volatile("st.relaxed.gpu.global.b64 [%0], %1;"
                   :
                   : "l"(ptr_), "d"(value)
                   : "memory");
    } else {
      asm volatile("st.relaxed.gpu.global.b64 [%0], %1;"
                   :
                   : "l"(ptr_), "l"(value)
                   : "memory");
    }
  }
}

template <typename P, typename T>
TL_DEVICE void st_release_sys(P ptr, T value) {
  static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
  static_assert(std::is_pointer_v<P> || std::is_same_v<P, uint64_t>);
  T *ptr_ = reinterpret_cast<T *>(ptr);

  if constexpr (sizeof(T) == 2) {
    asm volatile("st.release.sys.global.b16 [%0], %1;"
                 :
                 : "l"(ptr_), "h"(value)
                 : "memory");
  } else if constexpr (sizeof(T) == 4) {
    if constexpr (std::is_floating_point_v<T>) {
      asm volatile("st.release.sys.global.b32 [%0], %1;"
                   :
                   : "l"(ptr_), "f"(value)
                   : "memory");
    } else {
      asm volatile("st.release.sys.global.b32 [%0], %1;"
                   :
                   : "l"(ptr_), "r"(value)
                   : "memory");
    }
  } else {
    if constexpr (std::is_floating_point_v<T>) {
      asm volatile("st.release.sys.global.b64 [%0], %1;"
                   :
                   : "l"(ptr_), "d"(value)
                   : "memory");
    } else {
      asm volatile("st.release.sys.global.b64 [%0], %1;"
                   :
                   : "l"(ptr_), "l"(value)
                   : "memory");
    }
  }
}

template <typename P, typename T>
TL_DEVICE void st_relaxed_sys(P ptr, T value) {
  static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
  static_assert(std::is_pointer_v<P> || std::is_same_v<P, uint64_t>);
  T *ptr_ = reinterpret_cast<T *>(ptr);

  if constexpr (sizeof(T) == 2) {
    asm volatile("st.relaxed.sys.global.b16 [%0], %1;"
                 :
                 : "l"(ptr_), "h"(value)
                 : "memory");
  } else if constexpr (sizeof(T) == 4) {
    if constexpr (std::is_floating_point_v<T>) {
      asm volatile("st.relaxed.sys.global.b32 [%0], %1;"
                   :
                   : "l"(ptr_), "f"(value)
                   : "memory");
    } else {
      asm volatile("st.relaxed.sys.global.b32 [%0], %1;"
                   :
                   : "l"(ptr_), "r"(value)
                   : "memory");
    }
  } else {
    if constexpr (std::is_floating_point_v<T>) {
      asm volatile("st.relaxed.sys.global.b64 [%0], %1;"
                   :
                   : "l"(ptr_), "d"(value)
                   : "memory");
    } else {
      asm volatile("st.relaxed.sys.global.b64 [%0], %1;"
                   :
                   : "l"(ptr_), "l"(value)
                   : "memory");
    }
  }
}

} // namespace tl
