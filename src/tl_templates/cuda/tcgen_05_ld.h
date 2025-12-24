#pragma once

#include <cstdint>
#ifndef __CUDACC_RTC__
#include <cuda.h>
#endif

#include "common.h"

namespace tl {

// 32 data path lanes, 32-bit pattern, repeated N times
template <bool Pack16> class tmem_ld_32dp32bNx;

template <> class tmem_ld_32dp32bNx<false> {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &src_addr, uint32_t *dst_ptr) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 128,
                  "N must be a power of 2 and lies between 1 ~ 128");

    if constexpr (N == 1) {
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32"
                   "{%0},"
                   "[%1];\n"
                   : "=r"(dst_ptr[0])
                   : "r"(src_addr));
    } else if constexpr (N == 2) {
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32"
                   "{%0, %1},"
                   "[%2];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1])
                   : "r"(src_addr));
    } else if constexpr (N == 4) {
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32"
                   "{%0, %1, %2, %3},"
                   "[%4];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                     "=r"(dst_ptr[3])
                   : "r"(src_addr));
    } else if constexpr (N == 8) {
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32"
                   "{%0, %1, %2, %3, %4, %5, %6, %7},"
                   "[%8];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                     "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
                     "=r"(dst_ptr[6]), "=r"(dst_ptr[7])
                   : "r"(src_addr));
    } else if constexpr (N == 16) {
      asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.x16.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
          "%14, %15},"
          "[%16];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15])
          : "r"(src_addr));
    } else if constexpr (N == 32) {
      asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.x32.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
          "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
          "%26, %27, %28, %29, %30, %31},"
          "[%32];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31])
          : "r"(src_addr));
    } else if constexpr (N == 64) {
      asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.x64.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
          "%28, "
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, "
          "%42, "
          "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
          "%56, "
          "%57, %58, %59, %60, %61, %62, %63},"
          "[%64];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63])
          : "r"(src_addr));
    } else if constexpr (N == 128) {
      asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.x128.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
          "%28, "
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, "
          "%42, "
          "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
          "%56, "
          "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, "
          "%70, "
          "%71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, "
          "%84, "
          "%85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, "
          "%98, "
          "%99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, "
          "%110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
          "%121, %122, %123, %124, %125, %126, %127},"
          "[%128];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63]), "=r"(dst_ptr[64]), "=r"(dst_ptr[65]),
            "=r"(dst_ptr[66]), "=r"(dst_ptr[67]), "=r"(dst_ptr[68]),
            "=r"(dst_ptr[69]), "=r"(dst_ptr[70]), "=r"(dst_ptr[71]),
            "=r"(dst_ptr[72]), "=r"(dst_ptr[73]), "=r"(dst_ptr[74]),
            "=r"(dst_ptr[75]), "=r"(dst_ptr[76]), "=r"(dst_ptr[77]),
            "=r"(dst_ptr[78]), "=r"(dst_ptr[79]), "=r"(dst_ptr[80]),
            "=r"(dst_ptr[81]), "=r"(dst_ptr[82]), "=r"(dst_ptr[83]),
            "=r"(dst_ptr[84]), "=r"(dst_ptr[85]), "=r"(dst_ptr[86]),
            "=r"(dst_ptr[87]), "=r"(dst_ptr[88]), "=r"(dst_ptr[89]),
            "=r"(dst_ptr[90]), "=r"(dst_ptr[91]), "=r"(dst_ptr[92]),
            "=r"(dst_ptr[93]), "=r"(dst_ptr[94]), "=r"(dst_ptr[95]),
            "=r"(dst_ptr[96]), "=r"(dst_ptr[97]), "=r"(dst_ptr[98]),
            "=r"(dst_ptr[99]), "=r"(dst_ptr[100]), "=r"(dst_ptr[101]),
            "=r"(dst_ptr[102]), "=r"(dst_ptr[103]), "=r"(dst_ptr[104]),
            "=r"(dst_ptr[105]), "=r"(dst_ptr[106]), "=r"(dst_ptr[107]),
            "=r"(dst_ptr[108]), "=r"(dst_ptr[109]), "=r"(dst_ptr[110]),
            "=r"(dst_ptr[111]), "=r"(dst_ptr[112]), "=r"(dst_ptr[113]),
            "=r"(dst_ptr[114]), "=r"(dst_ptr[115]), "=r"(dst_ptr[116]),
            "=r"(dst_ptr[117]), "=r"(dst_ptr[118]), "=r"(dst_ptr[119]),
            "=r"(dst_ptr[120]), "=r"(dst_ptr[121]), "=r"(dst_ptr[122]),
            "=r"(dst_ptr[123]), "=r"(dst_ptr[124]), "=r"(dst_ptr[125]),
            "=r"(dst_ptr[126]), "=r"(dst_ptr[127])
          : "r"(src_addr));
    } else {
      asm volatile("trap");
    }
  }
};
template <> class tmem_ld_32dp32bNx<true> {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &src_addr, uint32_t *dst_ptr) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 128,
                  "N must be a power of 2 and lies between 1 ~ 128");

    if constexpr (N == 1) {
      asm volatile("tcgen05.ld.sync.aligned.32x32b.pack::16b.x1.b32"
                   "{%0},"
                   "[%1];\n"
                   : "=r"(dst_ptr[0])
                   : "r"(src_addr));
    } else if constexpr (N == 2) {
      asm volatile("tcgen05.ld.sync.aligned.32x32b.pack::16b.x2.b32"
                   "{%0, %1},"
                   "[%2];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1])
                   : "r"(src_addr));
    } else if constexpr (N == 4) {
      asm volatile("tcgen05.ld.sync.aligned.32x32b.pack::16b.x4.b32"
                   "{%0, %1, %2, %3},"
                   "[%4];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                     "=r"(dst_ptr[3])
                   : "r"(src_addr));
    } else if constexpr (N == 8) {
      asm volatile("tcgen05.ld.sync.aligned.32x32b.pack::16b.x8.b32"
                   "{%0, %1, %2, %3, %4, %5, %6, %7},"
                   "[%8];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                     "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
                     "=r"(dst_ptr[6]), "=r"(dst_ptr[7])
                   : "r"(src_addr));
    } else if constexpr (N == 16) {
      asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.pack::16b.x16.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
          "%14, %15},"
          "[%16];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15])
          : "r"(src_addr));
    } else if constexpr (N == 32) {
      asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.pack::16b.x32.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
          "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
          "%26, %27, %28, %29, %30, %31},"
          "[%32];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31])
          : "r"(src_addr));
    } else if constexpr (N == 64) {
      asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.pack::16b.x64.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
          "%28, "
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, "
          "%42, "
          "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
          "%56, "
          "%57, %58, %59, %60, %61, %62, %63},"
          "[%64];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63])
          : "r"(src_addr));
    } else if constexpr (N == 128) {
      asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.pack::16b.x128.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
          "%28, "
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, "
          "%42, "
          "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
          "%56, "
          "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, "
          "%70, "
          "%71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, "
          "%84, "
          "%85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, "
          "%98, "
          "%99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, "
          "%110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
          "%121, %122, %123, %124, %125, %126, %127},"
          "[%128];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63]), "=r"(dst_ptr[64]), "=r"(dst_ptr[65]),
            "=r"(dst_ptr[66]), "=r"(dst_ptr[67]), "=r"(dst_ptr[68]),
            "=r"(dst_ptr[69]), "=r"(dst_ptr[70]), "=r"(dst_ptr[71]),
            "=r"(dst_ptr[72]), "=r"(dst_ptr[73]), "=r"(dst_ptr[74]),
            "=r"(dst_ptr[75]), "=r"(dst_ptr[76]), "=r"(dst_ptr[77]),
            "=r"(dst_ptr[78]), "=r"(dst_ptr[79]), "=r"(dst_ptr[80]),
            "=r"(dst_ptr[81]), "=r"(dst_ptr[82]), "=r"(dst_ptr[83]),
            "=r"(dst_ptr[84]), "=r"(dst_ptr[85]), "=r"(dst_ptr[86]),
            "=r"(dst_ptr[87]), "=r"(dst_ptr[88]), "=r"(dst_ptr[89]),
            "=r"(dst_ptr[90]), "=r"(dst_ptr[91]), "=r"(dst_ptr[92]),
            "=r"(dst_ptr[93]), "=r"(dst_ptr[94]), "=r"(dst_ptr[95]),
            "=r"(dst_ptr[96]), "=r"(dst_ptr[97]), "=r"(dst_ptr[98]),
            "=r"(dst_ptr[99]), "=r"(dst_ptr[100]), "=r"(dst_ptr[101]),
            "=r"(dst_ptr[102]), "=r"(dst_ptr[103]), "=r"(dst_ptr[104]),
            "=r"(dst_ptr[105]), "=r"(dst_ptr[106]), "=r"(dst_ptr[107]),
            "=r"(dst_ptr[108]), "=r"(dst_ptr[109]), "=r"(dst_ptr[110]),
            "=r"(dst_ptr[111]), "=r"(dst_ptr[112]), "=r"(dst_ptr[113]),
            "=r"(dst_ptr[114]), "=r"(dst_ptr[115]), "=r"(dst_ptr[116]),
            "=r"(dst_ptr[117]), "=r"(dst_ptr[118]), "=r"(dst_ptr[119]),
            "=r"(dst_ptr[120]), "=r"(dst_ptr[121]), "=r"(dst_ptr[122]),
            "=r"(dst_ptr[123]), "=r"(dst_ptr[124]), "=r"(dst_ptr[125]),
            "=r"(dst_ptr[126]), "=r"(dst_ptr[127])
          : "r"(src_addr));
    } else {
      asm volatile("trap");
    }
  }
};

// 16 data path lanes, 64-bit pattern, repeated N times
template <bool Pack16> class tmem_ld_16dp64bNx;
template <> class tmem_ld_16dp64bNx<false> {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &src_addr, uint32_t *dst_ptr) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 128,
                  "N must be a power of 2 and lies between 1 ~ 128");

    if constexpr (N == 1) {
      asm volatile("tcgen05.ld.sync.aligned.16x64b.x1.b32"
                   "{%0},"
                   "[%1];\n"
                   : "=r"(dst_ptr[0])
                   : "r"(src_addr));
    } else if constexpr (N == 2) {
      asm volatile("tcgen05.ld.sync.aligned.16x64b.x2.b32"
                   "{%0, %1},"
                   "[%2];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1])
                   : "r"(src_addr));
    } else if constexpr (N == 4) {
      asm volatile("tcgen05.ld.sync.aligned.16x64b.x4.b32"
                   "{%0, %1, %2, %3},"
                   "[%4];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                     "=r"(dst_ptr[3])
                   : "r"(src_addr));
    } else if constexpr (N == 8) {
      asm volatile("tcgen05.ld.sync.aligned.16x64b.x8.b32"
                   "{%0, %1, %2, %3, %4, %5, %6, %7},"
                   "[%8];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                     "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
                     "=r"(dst_ptr[6]), "=r"(dst_ptr[7])
                   : "r"(src_addr));
    } else if constexpr (N == 16) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x64b.x16.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
          "%14, %15},"
          "[%16];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15])
          : "r"(src_addr));
    } else if constexpr (N == 32) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x64b.x32.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
          "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
          "%26, %27, %28, %29, %30, %31},"
          "[%32];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31])
          : "r"(src_addr));
    } else if constexpr (N == 64) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x64b.x64.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
          "%28, "
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, "
          "%42, "
          "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
          "%56, "
          "%57, %58, %59, %60, %61, %62, %63},"
          "[%64];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63])
          : "r"(src_addr));
    } else if constexpr (N == 128) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x64b.x128.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
          "%28, "
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, "
          "%42, "
          "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
          "%56, "
          "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, "
          "%70, "
          "%71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, "
          "%84, "
          "%85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, "
          "%98, "
          "%99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, "
          "%110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
          "%121, %122, %123, %124, %125, %126, %127},"
          "[%128];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63]), "=r"(dst_ptr[64]), "=r"(dst_ptr[65]),
            "=r"(dst_ptr[66]), "=r"(dst_ptr[67]), "=r"(dst_ptr[68]),
            "=r"(dst_ptr[69]), "=r"(dst_ptr[70]), "=r"(dst_ptr[71]),
            "=r"(dst_ptr[72]), "=r"(dst_ptr[73]), "=r"(dst_ptr[74]),
            "=r"(dst_ptr[75]), "=r"(dst_ptr[76]), "=r"(dst_ptr[77]),
            "=r"(dst_ptr[78]), "=r"(dst_ptr[79]), "=r"(dst_ptr[80]),
            "=r"(dst_ptr[81]), "=r"(dst_ptr[82]), "=r"(dst_ptr[83]),
            "=r"(dst_ptr[84]), "=r"(dst_ptr[85]), "=r"(dst_ptr[86]),
            "=r"(dst_ptr[87]), "=r"(dst_ptr[88]), "=r"(dst_ptr[89]),
            "=r"(dst_ptr[90]), "=r"(dst_ptr[91]), "=r"(dst_ptr[92]),
            "=r"(dst_ptr[93]), "=r"(dst_ptr[94]), "=r"(dst_ptr[95]),
            "=r"(dst_ptr[96]), "=r"(dst_ptr[97]), "=r"(dst_ptr[98]),
            "=r"(dst_ptr[99]), "=r"(dst_ptr[100]), "=r"(dst_ptr[101]),
            "=r"(dst_ptr[102]), "=r"(dst_ptr[103]), "=r"(dst_ptr[104]),
            "=r"(dst_ptr[105]), "=r"(dst_ptr[106]), "=r"(dst_ptr[107]),
            "=r"(dst_ptr[108]), "=r"(dst_ptr[109]), "=r"(dst_ptr[110]),
            "=r"(dst_ptr[111]), "=r"(dst_ptr[112]), "=r"(dst_ptr[113]),
            "=r"(dst_ptr[114]), "=r"(dst_ptr[115]), "=r"(dst_ptr[116]),
            "=r"(dst_ptr[117]), "=r"(dst_ptr[118]), "=r"(dst_ptr[119]),
            "=r"(dst_ptr[120]), "=r"(dst_ptr[121]), "=r"(dst_ptr[122]),
            "=r"(dst_ptr[123]), "=r"(dst_ptr[124]), "=r"(dst_ptr[125]),
            "=r"(dst_ptr[126]), "=r"(dst_ptr[127])
          : "r"(src_addr));
    } else {
      asm volatile("trap");
    }
  }
};
template <> class tmem_ld_16dp64bNx<true> {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &src_addr, uint32_t *dst_ptr) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 128,
                  "N must be a power of 2 and lies between 1 ~ 128");

    if constexpr (N == 1) {
      asm volatile("tcgen05.ld.sync.aligned.16x64b.pack::16b.x1.b32"
                   "{%0},"
                   "[%1];\n"
                   : "=r"(dst_ptr[0])
                   : "r"(src_addr));
    } else if constexpr (N == 2) {
      asm volatile("tcgen05.ld.sync.aligned.16x64b.pack::16b.x2.b32"
                   "{%0, %1},"
                   "[%2];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1])
                   : "r"(src_addr));
    } else if constexpr (N == 4) {
      asm volatile("tcgen05.ld.sync.aligned.16x64b.pack::16b.x4.b32"
                   "{%0, %1, %2, %3},"
                   "[%4];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                     "=r"(dst_ptr[3])
                   : "r"(src_addr));
    } else if constexpr (N == 8) {
      asm volatile("tcgen05.ld.sync.aligned.16x64b.pack::16b.x8.b32"
                   "{%0, %1, %2, %3, %4, %5, %6, %7},"
                   "[%8];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                     "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
                     "=r"(dst_ptr[6]), "=r"(dst_ptr[7])
                   : "r"(src_addr));
    } else if constexpr (N == 16) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x64b.pack::16b.x16.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
          "%14, %15},"
          "[%16];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15])
          : "r"(src_addr));
    } else if constexpr (N == 32) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x64b.pack::16b.x32.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
          "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
          "%26, %27, %28, %29, %30, %31},"
          "[%32];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31])
          : "r"(src_addr));
    } else if constexpr (N == 64) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x64b.pack::16b.x64.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
          "%28, "
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, "
          "%42, "
          "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
          "%56, "
          "%57, %58, %59, %60, %61, %62, %63},"
          "[%64];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63])
          : "r"(src_addr));
    } else if constexpr (N == 128) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x64b.pack::16b.x128.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
          "%28, "
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, "
          "%42, "
          "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
          "%56, "
          "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, "
          "%70, "
          "%71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, "
          "%84, "
          "%85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, "
          "%98, "
          "%99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, "
          "%110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
          "%121, %122, %123, %124, %125, %126, %127},"
          "[%128];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63]), "=r"(dst_ptr[64]), "=r"(dst_ptr[65]),
            "=r"(dst_ptr[66]), "=r"(dst_ptr[67]), "=r"(dst_ptr[68]),
            "=r"(dst_ptr[69]), "=r"(dst_ptr[70]), "=r"(dst_ptr[71]),
            "=r"(dst_ptr[72]), "=r"(dst_ptr[73]), "=r"(dst_ptr[74]),
            "=r"(dst_ptr[75]), "=r"(dst_ptr[76]), "=r"(dst_ptr[77]),
            "=r"(dst_ptr[78]), "=r"(dst_ptr[79]), "=r"(dst_ptr[80]),
            "=r"(dst_ptr[81]), "=r"(dst_ptr[82]), "=r"(dst_ptr[83]),
            "=r"(dst_ptr[84]), "=r"(dst_ptr[85]), "=r"(dst_ptr[86]),
            "=r"(dst_ptr[87]), "=r"(dst_ptr[88]), "=r"(dst_ptr[89]),
            "=r"(dst_ptr[90]), "=r"(dst_ptr[91]), "=r"(dst_ptr[92]),
            "=r"(dst_ptr[93]), "=r"(dst_ptr[94]), "=r"(dst_ptr[95]),
            "=r"(dst_ptr[96]), "=r"(dst_ptr[97]), "=r"(dst_ptr[98]),
            "=r"(dst_ptr[99]), "=r"(dst_ptr[100]), "=r"(dst_ptr[101]),
            "=r"(dst_ptr[102]), "=r"(dst_ptr[103]), "=r"(dst_ptr[104]),
            "=r"(dst_ptr[105]), "=r"(dst_ptr[106]), "=r"(dst_ptr[107]),
            "=r"(dst_ptr[108]), "=r"(dst_ptr[109]), "=r"(dst_ptr[110]),
            "=r"(dst_ptr[111]), "=r"(dst_ptr[112]), "=r"(dst_ptr[113]),
            "=r"(dst_ptr[114]), "=r"(dst_ptr[115]), "=r"(dst_ptr[116]),
            "=r"(dst_ptr[117]), "=r"(dst_ptr[118]), "=r"(dst_ptr[119]),
            "=r"(dst_ptr[120]), "=r"(dst_ptr[121]), "=r"(dst_ptr[122]),
            "=r"(dst_ptr[123]), "=r"(dst_ptr[124]), "=r"(dst_ptr[125]),
            "=r"(dst_ptr[126]), "=r"(dst_ptr[127])
          : "r"(src_addr));
    } else {
      asm volatile("trap");
    }
  }
};

// 16 data path lanes, 128-bit pattern, repeated N times
template <bool Pack16> class tmem_ld_16dp128bNx;
template <> class tmem_ld_16dp128bNx<false> {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &src_addr, uint32_t *dst_ptr) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 64,
                  "N must be a power of 2 and lies between 1 ~ 64");

    if constexpr (N == 1) {
      asm volatile("tcgen05.ld.sync.aligned.16x128b.x1.b32"
                   "{%0, %1},"
                   "[%2];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1])
                   : "r"(src_addr));
    } else if constexpr (N == 2) {
      asm volatile("tcgen05.ld.sync.aligned.16x128b.x2.b32"
                   "{%0, %1, %2, %3},"
                   "[%4];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                     "=r"(dst_ptr[3])
                   : "r"(src_addr));
    } else if constexpr (N == 4) {
      asm volatile("tcgen05.ld.sync.aligned.16x128b.x4.b32"
                   "{%0, %1, %2, %3, %4, %5, %6, %7},"
                   "[%8];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                     "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
                     "=r"(dst_ptr[6]), "=r"(dst_ptr[7])
                   : "r"(src_addr));
    } else if constexpr (N == 8) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x128b.x8.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
          "%14, %15},"
          "[%16];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15])
          : "r"(src_addr));
    } else if constexpr (N == 16) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x128b.x16.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
          "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
          "%26, %27, %28, %29, %30, %31},"
          "[%32];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31])
          : "r"(src_addr));
    } else if constexpr (N == 32) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x128b.x32.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
          "%28, "
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, "
          "%42, "
          "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
          "%56, "
          "%57, %58, %59, %60, %61, %62, %63},"
          "[%64];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63])
          : "r"(src_addr));
    } else if constexpr (N == 64) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x128b.x64.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
          "%28, "
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, "
          "%42, "
          "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
          "%56, "
          "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, "
          "%70, "
          "%71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, "
          "%84, "
          "%85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, "
          "%98, "
          "%99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, "
          "%110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
          "%121, %122, %123, %124, %125, %126, %127},"
          "[%128];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63]), "=r"(dst_ptr[64]), "=r"(dst_ptr[65]),
            "=r"(dst_ptr[66]), "=r"(dst_ptr[67]), "=r"(dst_ptr[68]),
            "=r"(dst_ptr[69]), "=r"(dst_ptr[70]), "=r"(dst_ptr[71]),
            "=r"(dst_ptr[72]), "=r"(dst_ptr[73]), "=r"(dst_ptr[74]),
            "=r"(dst_ptr[75]), "=r"(dst_ptr[76]), "=r"(dst_ptr[77]),
            "=r"(dst_ptr[78]), "=r"(dst_ptr[79]), "=r"(dst_ptr[80]),
            "=r"(dst_ptr[81]), "=r"(dst_ptr[82]), "=r"(dst_ptr[83]),
            "=r"(dst_ptr[84]), "=r"(dst_ptr[85]), "=r"(dst_ptr[86]),
            "=r"(dst_ptr[87]), "=r"(dst_ptr[88]), "=r"(dst_ptr[89]),
            "=r"(dst_ptr[90]), "=r"(dst_ptr[91]), "=r"(dst_ptr[92]),
            "=r"(dst_ptr[93]), "=r"(dst_ptr[94]), "=r"(dst_ptr[95]),
            "=r"(dst_ptr[96]), "=r"(dst_ptr[97]), "=r"(dst_ptr[98]),
            "=r"(dst_ptr[99]), "=r"(dst_ptr[100]), "=r"(dst_ptr[101]),
            "=r"(dst_ptr[102]), "=r"(dst_ptr[103]), "=r"(dst_ptr[104]),
            "=r"(dst_ptr[105]), "=r"(dst_ptr[106]), "=r"(dst_ptr[107]),
            "=r"(dst_ptr[108]), "=r"(dst_ptr[109]), "=r"(dst_ptr[110]),
            "=r"(dst_ptr[111]), "=r"(dst_ptr[112]), "=r"(dst_ptr[113]),
            "=r"(dst_ptr[114]), "=r"(dst_ptr[115]), "=r"(dst_ptr[116]),
            "=r"(dst_ptr[117]), "=r"(dst_ptr[118]), "=r"(dst_ptr[119]),
            "=r"(dst_ptr[120]), "=r"(dst_ptr[121]), "=r"(dst_ptr[122]),
            "=r"(dst_ptr[123]), "=r"(dst_ptr[124]), "=r"(dst_ptr[125]),
            "=r"(dst_ptr[126]), "=r"(dst_ptr[127])
          : "r"(src_addr));
    } else {
      asm volatile("trap");
    }
  }
};
template <> class tmem_ld_16dp128bNx<true> {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &src_addr, uint32_t *dst_ptr) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 64,
                  "N must be a power of 2 and lies between 1 ~ 64");

    if constexpr (N == 1) {
      asm volatile("tcgen05.ld.sync.aligned.16x128b.pack::16b.x1.b32"
                   "{%0, %1},"
                   "[%2];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1])
                   : "r"(src_addr));
    } else if constexpr (N == 2) {
      asm volatile("tcgen05.ld.sync.aligned.16x128b.pack::16b.x2.b32"
                   "{%0, %1, %2, %3},"
                   "[%4];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                     "=r"(dst_ptr[3])
                   : "r"(src_addr));
    } else if constexpr (N == 4) {
      asm volatile("tcgen05.ld.sync.aligned.16x128b.pack::16b.x4.b32"
                   "{%0, %1, %2, %3, %4, %5, %6, %7},"
                   "[%8];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                     "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
                     "=r"(dst_ptr[6]), "=r"(dst_ptr[7])
                   : "r"(src_addr));
    } else if constexpr (N == 8) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x128b.pack::16b.x8.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
          "%14, %15},"
          "[%16];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15])
          : "r"(src_addr));
    } else if constexpr (N == 16) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x128b.pack::16b.x16.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
          "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
          "%26, %27, %28, %29, %30, %31},"
          "[%32];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31])
          : "r"(src_addr));
    } else if constexpr (N == 32) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x128b.pack::16b.x32.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
          "%28, "
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, "
          "%42, "
          "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
          "%56, "
          "%57, %58, %59, %60, %61, %62, %63},"
          "[%64];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63])
          : "r"(src_addr));
    } else if constexpr (N == 64) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x128b.pack::16b.x64.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
          "%28, "
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, "
          "%42, "
          "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
          "%56, "
          "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, "
          "%70, "
          "%71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, "
          "%84, "
          "%85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, "
          "%98, "
          "%99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, "
          "%110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
          "%121, %122, %123, %124, %125, %126, %127},"
          "[%128];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63]), "=r"(dst_ptr[64]), "=r"(dst_ptr[65]),
            "=r"(dst_ptr[66]), "=r"(dst_ptr[67]), "=r"(dst_ptr[68]),
            "=r"(dst_ptr[69]), "=r"(dst_ptr[70]), "=r"(dst_ptr[71]),
            "=r"(dst_ptr[72]), "=r"(dst_ptr[73]), "=r"(dst_ptr[74]),
            "=r"(dst_ptr[75]), "=r"(dst_ptr[76]), "=r"(dst_ptr[77]),
            "=r"(dst_ptr[78]), "=r"(dst_ptr[79]), "=r"(dst_ptr[80]),
            "=r"(dst_ptr[81]), "=r"(dst_ptr[82]), "=r"(dst_ptr[83]),
            "=r"(dst_ptr[84]), "=r"(dst_ptr[85]), "=r"(dst_ptr[86]),
            "=r"(dst_ptr[87]), "=r"(dst_ptr[88]), "=r"(dst_ptr[89]),
            "=r"(dst_ptr[90]), "=r"(dst_ptr[91]), "=r"(dst_ptr[92]),
            "=r"(dst_ptr[93]), "=r"(dst_ptr[94]), "=r"(dst_ptr[95]),
            "=r"(dst_ptr[96]), "=r"(dst_ptr[97]), "=r"(dst_ptr[98]),
            "=r"(dst_ptr[99]), "=r"(dst_ptr[100]), "=r"(dst_ptr[101]),
            "=r"(dst_ptr[102]), "=r"(dst_ptr[103]), "=r"(dst_ptr[104]),
            "=r"(dst_ptr[105]), "=r"(dst_ptr[106]), "=r"(dst_ptr[107]),
            "=r"(dst_ptr[108]), "=r"(dst_ptr[109]), "=r"(dst_ptr[110]),
            "=r"(dst_ptr[111]), "=r"(dst_ptr[112]), "=r"(dst_ptr[113]),
            "=r"(dst_ptr[114]), "=r"(dst_ptr[115]), "=r"(dst_ptr[116]),
            "=r"(dst_ptr[117]), "=r"(dst_ptr[118]), "=r"(dst_ptr[119]),
            "=r"(dst_ptr[120]), "=r"(dst_ptr[121]), "=r"(dst_ptr[122]),
            "=r"(dst_ptr[123]), "=r"(dst_ptr[124]), "=r"(dst_ptr[125]),
            "=r"(dst_ptr[126]), "=r"(dst_ptr[127])
          : "r"(src_addr));
    } else {
      asm volatile("trap");
    }
  }
};

// 16 data path lanes, 256-bit pattern, repeated N times
template <bool Pack16> class tmem_ld_16dp256bNx;
template <> class tmem_ld_16dp256bNx<false> {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &src_addr, uint32_t *dst_ptr) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 32,
                  "N must be a power of 2 and lies between 1 ~ 32");

    if constexpr (N == 1) {
      asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32"
                   "{%0, %1, %2, %3},"
                   "[%4];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                     "=r"(dst_ptr[3])
                   : "r"(src_addr));
    } else if constexpr (N == 2) {
      asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32"
                   "{%0, %1, %2, %3, %4, %5, %6, %7},"
                   "[%8];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                     "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
                     "=r"(dst_ptr[6]), "=r"(dst_ptr[7])
                   : "r"(src_addr));
    } else if constexpr (N == 4) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x256b.x4.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
          "%14, %15},"
          "[%16];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15])
          : "r"(src_addr));
    } else if constexpr (N == 8) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x256b.x8.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
          "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
          "%26, %27, %28, %29, %30, %31},"
          "[%32];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31])
          : "r"(src_addr));
    } else if constexpr (N == 16) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x256b.x16.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
          "%28, "
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, "
          "%42, "
          "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
          "%56, "
          "%57, %58, %59, %60, %61, %62, %63},"
          "[%64];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63])
          : "r"(src_addr));
    } else if constexpr (N == 32) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x256b.x32.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
          "%28, "
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, "
          "%42, "
          "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
          "%56, "
          "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, "
          "%70, "
          "%71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, "
          "%84, "
          "%85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, "
          "%98, "
          "%99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, "
          "%110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
          "%121, %122, %123, %124, %125, %126, %127},"
          "[%128];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63]), "=r"(dst_ptr[64]), "=r"(dst_ptr[65]),
            "=r"(dst_ptr[66]), "=r"(dst_ptr[67]), "=r"(dst_ptr[68]),
            "=r"(dst_ptr[69]), "=r"(dst_ptr[70]), "=r"(dst_ptr[71]),
            "=r"(dst_ptr[72]), "=r"(dst_ptr[73]), "=r"(dst_ptr[74]),
            "=r"(dst_ptr[75]), "=r"(dst_ptr[76]), "=r"(dst_ptr[77]),
            "=r"(dst_ptr[78]), "=r"(dst_ptr[79]), "=r"(dst_ptr[80]),
            "=r"(dst_ptr[81]), "=r"(dst_ptr[82]), "=r"(dst_ptr[83]),
            "=r"(dst_ptr[84]), "=r"(dst_ptr[85]), "=r"(dst_ptr[86]),
            "=r"(dst_ptr[87]), "=r"(dst_ptr[88]), "=r"(dst_ptr[89]),
            "=r"(dst_ptr[90]), "=r"(dst_ptr[91]), "=r"(dst_ptr[92]),
            "=r"(dst_ptr[93]), "=r"(dst_ptr[94]), "=r"(dst_ptr[95]),
            "=r"(dst_ptr[96]), "=r"(dst_ptr[97]), "=r"(dst_ptr[98]),
            "=r"(dst_ptr[99]), "=r"(dst_ptr[100]), "=r"(dst_ptr[101]),
            "=r"(dst_ptr[102]), "=r"(dst_ptr[103]), "=r"(dst_ptr[104]),
            "=r"(dst_ptr[105]), "=r"(dst_ptr[106]), "=r"(dst_ptr[107]),
            "=r"(dst_ptr[108]), "=r"(dst_ptr[109]), "=r"(dst_ptr[110]),
            "=r"(dst_ptr[111]), "=r"(dst_ptr[112]), "=r"(dst_ptr[113]),
            "=r"(dst_ptr[114]), "=r"(dst_ptr[115]), "=r"(dst_ptr[116]),
            "=r"(dst_ptr[117]), "=r"(dst_ptr[118]), "=r"(dst_ptr[119]),
            "=r"(dst_ptr[120]), "=r"(dst_ptr[121]), "=r"(dst_ptr[122]),
            "=r"(dst_ptr[123]), "=r"(dst_ptr[124]), "=r"(dst_ptr[125]),
            "=r"(dst_ptr[126]), "=r"(dst_ptr[127])
          : "r"(src_addr));
    } else {
      asm volatile("trap");
    }
  }
};
template <> class tmem_ld_16dp256bNx<true> {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &src_addr, uint32_t *dst_ptr) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 32,
                  "N must be a power of 2 and lies between 1 ~ 32");

    if constexpr (N == 1) {
      asm volatile("tcgen05.ld.sync.aligned.16x256b.pack::16b.x1.b32"
                   "{%0, %1, %2, %3},"
                   "[%4];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                     "=r"(dst_ptr[3])
                   : "r"(src_addr));
    } else if constexpr (N == 2) {
      asm volatile("tcgen05.ld.sync.aligned.16x256b.pack::16b.x2.b32"
                   "{%0, %1, %2, %3, %4, %5, %6, %7},"
                   "[%8];\n"
                   : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                     "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
                     "=r"(dst_ptr[6]), "=r"(dst_ptr[7])
                   : "r"(src_addr));
    } else if constexpr (N == 4) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x256b.pack::16b.x4.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
          "%14, %15},"
          "[%16];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15])
          : "r"(src_addr));
    } else if constexpr (N == 8) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x256b.pack::16b.x8.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
          "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
          "%26, %27, %28, %29, %30, %31},"
          "[%32];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31])
          : "r"(src_addr));
    } else if constexpr (N == 16) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x256b.pack::16b.x16.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
          "%28, "
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, "
          "%42, "
          "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
          "%56, "
          "%57, %58, %59, %60, %61, %62, %63},"
          "[%64];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63])
          : "r"(src_addr));
    } else if constexpr (N == 32) {
      asm volatile(
          "tcgen05.ld.sync.aligned.16x256b.pack::16b.x32.b32"
          "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
          "%28, "
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, "
          "%42, "
          "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
          "%56, "
          "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, "
          "%70, "
          "%71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, "
          "%84, "
          "%85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, "
          "%98, "
          "%99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, "
          "%110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
          "%121, %122, %123, %124, %125, %126, %127},"
          "[%128];\n"
          : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63]), "=r"(dst_ptr[64]), "=r"(dst_ptr[65]),
            "=r"(dst_ptr[66]), "=r"(dst_ptr[67]), "=r"(dst_ptr[68]),
            "=r"(dst_ptr[69]), "=r"(dst_ptr[70]), "=r"(dst_ptr[71]),
            "=r"(dst_ptr[72]), "=r"(dst_ptr[73]), "=r"(dst_ptr[74]),
            "=r"(dst_ptr[75]), "=r"(dst_ptr[76]), "=r"(dst_ptr[77]),
            "=r"(dst_ptr[78]), "=r"(dst_ptr[79]), "=r"(dst_ptr[80]),
            "=r"(dst_ptr[81]), "=r"(dst_ptr[82]), "=r"(dst_ptr[83]),
            "=r"(dst_ptr[84]), "=r"(dst_ptr[85]), "=r"(dst_ptr[86]),
            "=r"(dst_ptr[87]), "=r"(dst_ptr[88]), "=r"(dst_ptr[89]),
            "=r"(dst_ptr[90]), "=r"(dst_ptr[91]), "=r"(dst_ptr[92]),
            "=r"(dst_ptr[93]), "=r"(dst_ptr[94]), "=r"(dst_ptr[95]),
            "=r"(dst_ptr[96]), "=r"(dst_ptr[97]), "=r"(dst_ptr[98]),
            "=r"(dst_ptr[99]), "=r"(dst_ptr[100]), "=r"(dst_ptr[101]),
            "=r"(dst_ptr[102]), "=r"(dst_ptr[103]), "=r"(dst_ptr[104]),
            "=r"(dst_ptr[105]), "=r"(dst_ptr[106]), "=r"(dst_ptr[107]),
            "=r"(dst_ptr[108]), "=r"(dst_ptr[109]), "=r"(dst_ptr[110]),
            "=r"(dst_ptr[111]), "=r"(dst_ptr[112]), "=r"(dst_ptr[113]),
            "=r"(dst_ptr[114]), "=r"(dst_ptr[115]), "=r"(dst_ptr[116]),
            "=r"(dst_ptr[117]), "=r"(dst_ptr[118]), "=r"(dst_ptr[119]),
            "=r"(dst_ptr[120]), "=r"(dst_ptr[121]), "=r"(dst_ptr[122]),
            "=r"(dst_ptr[123]), "=r"(dst_ptr[124]), "=r"(dst_ptr[125]),
            "=r"(dst_ptr[126]), "=r"(dst_ptr[127])
          : "r"(src_addr));
    } else {
      asm volatile("trap");
    }
  }
};

// 32 data path lanes, 64-bit pattern, repeated N times
// (conducted with 2x16dp64bNx)
template <bool Pack16 = false> class tmem_ld_32dp64bNx {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &src_addr, uint32_t *dst_ptr) {
    tmem_ld_16dp64bNx<Pack16>::copy<N>(src_addr, dst_ptr);
    tmem_ld_16dp64bNx<Pack16>::copy<N>(src_addr + (16 << 16), dst_ptr + N);
  }
};

// 32 data path lanes, 128-bit pattern, repeated N times
template <bool Pack16 = false> class tmem_ld_32dp128bNx {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &src_addr, uint32_t *dst_ptr) {
    tmem_ld_16dp128bNx<Pack16>::copy<N>(src_addr, dst_ptr);
    tmem_ld_16dp128bNx<Pack16>::copy<N>(src_addr + (16 << 16), dst_ptr + N * 2);
  }
};

// 32 data path lanes, 256-bit pattern, repeated N times
template <bool Pack16 = false> class tmem_ld_32dp256bNx {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &src_addr, uint32_t *dst_ptr) {
    tmem_ld_16dp256bNx<Pack16>::copy<N>(src_addr, dst_ptr);
    tmem_ld_16dp256bNx<Pack16>::copy<N>(src_addr + (16 << 16), dst_ptr + N * 4);
  }
};

} // namespace tl
