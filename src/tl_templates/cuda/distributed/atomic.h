#pragma once

#include <cstdint>

#include "../common.h"

namespace tl {

TL_DEVICE uint32_t ptx_atom_add_relaxed_gpu(const uint32_t *ptr,
                                            uint32_t value) {
  uint32_t ret;
  asm volatile("atom.add.relaxed.gpu.global.u32 %0, [%1], %2;\n"
               : "=r"(ret)
               : "l"(ptr), "r"(value));
  return ret;
}

TL_DEVICE uint32_t ptx_atom_add_acquire_gpu(const uint32_t *ptr,
                                            uint32_t value) {
  uint32_t ret;
  asm volatile("atom.add.acquire.gpu.global.u32 %0, [%1], %2;\n"
               : "=r"(ret)
               : "l"(ptr), "r"(value));
  return ret;
}

TL_DEVICE uint32_t ptx_atom_add_release_gpu(const uint32_t *ptr,
                                            uint32_t value) {
  uint32_t ret;
  asm volatile("atom.add.release.gpu.global.u32 %0, [%1], %2;\n"
               : "=r"(ret)
               : "l"(ptr), "r"(value));
  return ret;
}

TL_DEVICE uint32_t ptx_atom_add_acq_rel_gpu(const uint32_t *ptr,
                                            uint32_t value) {
  uint32_t ret;
  asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;\n"
               : "=r"(ret)
               : "l"(ptr), "r"(value));
  return ret;
}

TL_DEVICE uint32_t ptx_atom_add_relaxed_sys(const uint32_t *ptr,
                                            uint32_t value) {
  uint32_t ret;
  asm volatile("atom.add.relaxed.sys.global.u32 %0, [%1], %2;\n"
               : "=r"(ret)
               : "l"(ptr), "r"(value));
  return ret;
}

TL_DEVICE uint32_t ptx_atom_add_acquire_sys(const uint32_t *ptr,
                                            uint32_t value) {
  uint32_t ret;
  asm volatile("atom.add.acquire.sys.global.u32 %0, [%1], %2;\n"
               : "=r"(ret)
               : "l"(ptr), "r"(value));
  return ret;
}

TL_DEVICE uint32_t ptx_atom_add_release_sys(const uint32_t *ptr,
                                            uint32_t value) {
  uint32_t ret;
  asm volatile("atom.add.release.sys.global.u32 %0, [%1], %2;\n"
               : "=r"(ret)
               : "l"(ptr), "r"(value));
  return ret;
}

TL_DEVICE uint32_t ptx_atom_add_acq_rel_sys(const uint32_t *ptr,
                                            uint32_t value) {
  uint32_t ret;
  asm volatile("atom.add.acq_rel.sys.global.u32 %0, [%1], %2;\n"
               : "=r"(ret)
               : "l"(ptr), "r"(value));
  return ret;
}

} // namespace tl
