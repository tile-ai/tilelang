#pragma once

#include <cstdint>
#ifndef __CUDACC_RTC__
#include <cuda.h>
#endif

#include "common.h"

namespace tl {

// 32 data path lanes, 32b-bit pattern, repeated N times (store)
template <bool Unpack16> class tmem_st_32dp32bNx;

template <> class tmem_st_32dp32bNx<false> {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &dst_addr, uint32_t const *src_ptr) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 128,
                  "N must be a power of 2 and lies between 1 ~ 128");

    if constexpr (N == 1) {
      asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32"
                   "[%0],"
                   "{%1};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]));
    } else if constexpr (N == 2) {
      asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32"
                   "[%0],"
                   "{%1, %2};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]));
    } else if constexpr (N == 4) {
      asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32"
                   "[%0],"
                   "{%1, %2, %3, %4};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]), "r"(src_ptr[3]));
    } else if constexpr (N == 8) {
      asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32"
                   "[%0],"
                   "{%1, %2, %3, %4, %5, %6, %7, %8};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]));
    } else if constexpr (N == 16) {
      asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32"
                   "[%0],"
                   "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]));
    } else if constexpr (N == 32) {
      asm volatile(
          "tcgen05.st.sync.aligned.32x32b.x32.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]));
    } else if constexpr (N == 64) {
      asm volatile(
          "tcgen05.st.sync.aligned.32x32b.x64.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41,"
          "%42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54,"
          "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]),
                     "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]),
                     "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]),
                     "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]),
                     "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]),
                     "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]),
                     "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]),
                     "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]),
                     "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]),
                     "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]),
                     "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]),
                     "r"(src_ptr[62]), "r"(src_ptr[63]));
    } else if constexpr (N == 128) {
      asm volatile(
          "tcgen05.st.sync.aligned.32x32b.x128.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41,"
          "%42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54,"
          "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67,"
          "%68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80,"
          "%81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93,"
          "%94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105,"
          "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116,"
          "%117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127,"
          "%128};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]),
                     "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]),
                     "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]),
                     "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]),
                     "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]),
                     "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]),
                     "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]),
                     "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]),
                     "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]),
                     "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]),
                     "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]),
                     "r"(src_ptr[62]), "r"(src_ptr[63]), "r"(src_ptr[64]),
                     "r"(src_ptr[65]), "r"(src_ptr[66]), "r"(src_ptr[67]),
                     "r"(src_ptr[68]), "r"(src_ptr[69]), "r"(src_ptr[70]),
                     "r"(src_ptr[71]), "r"(src_ptr[72]), "r"(src_ptr[73]),
                     "r"(src_ptr[74]), "r"(src_ptr[75]), "r"(src_ptr[76]),
                     "r"(src_ptr[77]), "r"(src_ptr[78]), "r"(src_ptr[79]),
                     "r"(src_ptr[80]), "r"(src_ptr[81]), "r"(src_ptr[82]),
                     "r"(src_ptr[83]), "r"(src_ptr[84]), "r"(src_ptr[85]),
                     "r"(src_ptr[86]), "r"(src_ptr[87]), "r"(src_ptr[88]),
                     "r"(src_ptr[89]), "r"(src_ptr[90]), "r"(src_ptr[91]),
                     "r"(src_ptr[92]), "r"(src_ptr[93]), "r"(src_ptr[94]),
                     "r"(src_ptr[95]), "r"(src_ptr[96]), "r"(src_ptr[97]),
                     "r"(src_ptr[98]), "r"(src_ptr[99]), "r"(src_ptr[100]),
                     "r"(src_ptr[101]), "r"(src_ptr[102]), "r"(src_ptr[103]),
                     "r"(src_ptr[104]), "r"(src_ptr[105]), "r"(src_ptr[106]),
                     "r"(src_ptr[107]), "r"(src_ptr[108]), "r"(src_ptr[109]),
                     "r"(src_ptr[110]), "r"(src_ptr[111]), "r"(src_ptr[112]),
                     "r"(src_ptr[113]), "r"(src_ptr[114]), "r"(src_ptr[115]),
                     "r"(src_ptr[116]), "r"(src_ptr[117]), "r"(src_ptr[118]),
                     "r"(src_ptr[119]), "r"(src_ptr[120]), "r"(src_ptr[121]),
                     "r"(src_ptr[122]), "r"(src_ptr[123]), "r"(src_ptr[124]),
                     "r"(src_ptr[125]), "r"(src_ptr[126]), "r"(src_ptr[127]));
    } else {
      asm volatile("trap");
    }
  }
};
template <> class tmem_st_32dp32bNx<true> {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &dst_addr, uint32_t const *src_ptr) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 128,
                  "N must be a power of 2 and lies between 1 ~ 128");

    if constexpr (N == 1) {
      asm volatile("tcgen05.st.sync.aligned.32x32b.unpack::16b.x1.b32"
                   "[%0],"
                   "{%1};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]));
    } else if constexpr (N == 2) {
      asm volatile("tcgen05.st.sync.aligned.32x32b.unpack::16b.x2.b32"
                   "[%0],"
                   "{%1, %2};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]));
    } else if constexpr (N == 4) {
      asm volatile("tcgen05.st.sync.aligned.32x32b.unpack::16b.x4.b32"
                   "[%0],"
                   "{%1, %2, %3, %4};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]), "r"(src_ptr[3]));
    } else if constexpr (N == 8) {
      asm volatile("tcgen05.st.sync.aligned.32x32b.unpack::16b.x8.b32"
                   "[%0],"
                   "{%1, %2, %3, %4, %5, %6, %7, %8};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]));
    } else if constexpr (N == 16) {
      asm volatile("tcgen05.st.sync.aligned.32x32b.unpack::16b.x16.b32"
                   "[%0],"
                   "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]));
    } else if constexpr (N == 32) {
      asm volatile(
          "tcgen05.st.sync.aligned.32x32b.unpack::16b.x32.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]));
    } else if constexpr (N == 64) {
      asm volatile(
          "tcgen05.st.sync.aligned.32x32b.unpack::16b.x64.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41,"
          "%42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54,"
          "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]),
                     "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]),
                     "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]),
                     "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]),
                     "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]),
                     "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]),
                     "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]),
                     "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]),
                     "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]),
                     "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]),
                     "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]),
                     "r"(src_ptr[62]), "r"(src_ptr[63]));
    } else if constexpr (N == 128) {
      asm volatile(
          "tcgen05.st.sync.aligned.32x32b.unpack::16b.x128.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41,"
          "%42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54,"
          "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67,"
          "%68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80,"
          "%81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93,"
          "%94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105,"
          "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116,"
          "%117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127,"
          "%128};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]),
                     "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]),
                     "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]),
                     "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]),
                     "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]),
                     "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]),
                     "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]),
                     "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]),
                     "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]),
                     "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]),
                     "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]),
                     "r"(src_ptr[62]), "r"(src_ptr[63]), "r"(src_ptr[64]),
                     "r"(src_ptr[65]), "r"(src_ptr[66]), "r"(src_ptr[67]),
                     "r"(src_ptr[68]), "r"(src_ptr[69]), "r"(src_ptr[70]),
                     "r"(src_ptr[71]), "r"(src_ptr[72]), "r"(src_ptr[73]),
                     "r"(src_ptr[74]), "r"(src_ptr[75]), "r"(src_ptr[76]),
                     "r"(src_ptr[77]), "r"(src_ptr[78]), "r"(src_ptr[79]),
                     "r"(src_ptr[80]), "r"(src_ptr[81]), "r"(src_ptr[82]),
                     "r"(src_ptr[83]), "r"(src_ptr[84]), "r"(src_ptr[85]),
                     "r"(src_ptr[86]), "r"(src_ptr[87]), "r"(src_ptr[88]),
                     "r"(src_ptr[89]), "r"(src_ptr[90]), "r"(src_ptr[91]),
                     "r"(src_ptr[92]), "r"(src_ptr[93]), "r"(src_ptr[94]),
                     "r"(src_ptr[95]), "r"(src_ptr[96]), "r"(src_ptr[97]),
                     "r"(src_ptr[98]), "r"(src_ptr[99]), "r"(src_ptr[100]),
                     "r"(src_ptr[101]), "r"(src_ptr[102]), "r"(src_ptr[103]),
                     "r"(src_ptr[104]), "r"(src_ptr[105]), "r"(src_ptr[106]),
                     "r"(src_ptr[107]), "r"(src_ptr[108]), "r"(src_ptr[109]),
                     "r"(src_ptr[110]), "r"(src_ptr[111]), "r"(src_ptr[112]),
                     "r"(src_ptr[113]), "r"(src_ptr[114]), "r"(src_ptr[115]),
                     "r"(src_ptr[116]), "r"(src_ptr[117]), "r"(src_ptr[118]),
                     "r"(src_ptr[119]), "r"(src_ptr[120]), "r"(src_ptr[121]),
                     "r"(src_ptr[122]), "r"(src_ptr[123]), "r"(src_ptr[124]),
                     "r"(src_ptr[125]), "r"(src_ptr[126]), "r"(src_ptr[127]));
    } else {
      asm volatile("trap");
    }
  }
};

// 16 data path lanes, 64b-bit pattern, repeated N times (store)
template <bool Unpack16> class tmem_st_16dp64bNx;

template <> class tmem_st_16dp64bNx<false> {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &dst_addr, uint32_t const *src_ptr) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 128,
                  "N must be a power of 2 and lies between 1 ~ 128");

    if constexpr (N == 1) {
      asm volatile("tcgen05.st.sync.aligned.16x64b.x1.b32"
                   "[%0],"
                   "{%1};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]));
    } else if constexpr (N == 2) {
      asm volatile("tcgen05.st.sync.aligned.16x64b.x2.b32"
                   "[%0],"
                   "{%1, %2};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]));
    } else if constexpr (N == 4) {
      asm volatile("tcgen05.st.sync.aligned.16x64b.x4.b32"
                   "[%0],"
                   "{%1, %2, %3, %4};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]), "r"(src_ptr[3]));
    } else if constexpr (N == 8) {
      asm volatile("tcgen05.st.sync.aligned.16x64b.x8.b32"
                   "[%0],"
                   "{%1, %2, %3, %4, %5, %6, %7, %8};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]));
    } else if constexpr (N == 16) {
      asm volatile("tcgen05.st.sync.aligned.16x64b.x16.b32"
                   "[%0],"
                   "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]));
    } else if constexpr (N == 32) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x64b.x32.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]));
    } else if constexpr (N == 64) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x64b.x64.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41,"
          "%42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54,"
          "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]),
                     "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]),
                     "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]),
                     "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]),
                     "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]),
                     "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]),
                     "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]),
                     "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]),
                     "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]),
                     "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]),
                     "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]),
                     "r"(src_ptr[62]), "r"(src_ptr[63]));
    } else if constexpr (N == 128) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x64b.x128.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41,"
          "%42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54,"
          "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67,"
          "%68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80,"
          "%81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93,"
          "%94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105,"
          "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116,"
          "%117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127,"
          "%128};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]),
                     "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]),
                     "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]),
                     "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]),
                     "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]),
                     "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]),
                     "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]),
                     "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]),
                     "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]),
                     "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]),
                     "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]),
                     "r"(src_ptr[62]), "r"(src_ptr[63]), "r"(src_ptr[64]),
                     "r"(src_ptr[65]), "r"(src_ptr[66]), "r"(src_ptr[67]),
                     "r"(src_ptr[68]), "r"(src_ptr[69]), "r"(src_ptr[70]),
                     "r"(src_ptr[71]), "r"(src_ptr[72]), "r"(src_ptr[73]),
                     "r"(src_ptr[74]), "r"(src_ptr[75]), "r"(src_ptr[76]),
                     "r"(src_ptr[77]), "r"(src_ptr[78]), "r"(src_ptr[79]),
                     "r"(src_ptr[80]), "r"(src_ptr[81]), "r"(src_ptr[82]),
                     "r"(src_ptr[83]), "r"(src_ptr[84]), "r"(src_ptr[85]),
                     "r"(src_ptr[86]), "r"(src_ptr[87]), "r"(src_ptr[88]),
                     "r"(src_ptr[89]), "r"(src_ptr[90]), "r"(src_ptr[91]),
                     "r"(src_ptr[92]), "r"(src_ptr[93]), "r"(src_ptr[94]),
                     "r"(src_ptr[95]), "r"(src_ptr[96]), "r"(src_ptr[97]),
                     "r"(src_ptr[98]), "r"(src_ptr[99]), "r"(src_ptr[100]),
                     "r"(src_ptr[101]), "r"(src_ptr[102]), "r"(src_ptr[103]),
                     "r"(src_ptr[104]), "r"(src_ptr[105]), "r"(src_ptr[106]),
                     "r"(src_ptr[107]), "r"(src_ptr[108]), "r"(src_ptr[109]),
                     "r"(src_ptr[110]), "r"(src_ptr[111]), "r"(src_ptr[112]),
                     "r"(src_ptr[113]), "r"(src_ptr[114]), "r"(src_ptr[115]),
                     "r"(src_ptr[116]), "r"(src_ptr[117]), "r"(src_ptr[118]),
                     "r"(src_ptr[119]), "r"(src_ptr[120]), "r"(src_ptr[121]),
                     "r"(src_ptr[122]), "r"(src_ptr[123]), "r"(src_ptr[124]),
                     "r"(src_ptr[125]), "r"(src_ptr[126]), "r"(src_ptr[127]));
    } else {
      asm volatile("trap");
    }
  }
};
template <> class tmem_st_16dp64bNx<true> {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &dst_addr, uint32_t const *src_ptr) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 128,
                  "N must be a power of 2 and lies between 1 ~ 128");

    if constexpr (N == 1) {
      asm volatile("tcgen05.st.sync.aligned.16x64b.unpack::16b.x1.b32"
                   "[%0],"
                   "{%1};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]));
    } else if constexpr (N == 2) {
      asm volatile("tcgen05.st.sync.aligned.16x64b.unpack::16b.x2.b32"
                   "[%0],"
                   "{%1, %2};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]));
    } else if constexpr (N == 4) {
      asm volatile("tcgen05.st.sync.aligned.16x64b.unpack::16b.x4.b32"
                   "[%0],"
                   "{%1, %2, %3, %4};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]), "r"(src_ptr[3]));
    } else if constexpr (N == 8) {
      asm volatile("tcgen05.st.sync.aligned.16x64b.unpack::16b.x8.b32"
                   "[%0],"
                   "{%1, %2, %3, %4, %5, %6, %7, %8};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]));
    } else if constexpr (N == 16) {
      asm volatile("tcgen05.st.sync.aligned.16x64b.unpack::16b.x16.b32"
                   "[%0],"
                   "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]));
    } else if constexpr (N == 32) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x64b.unpack::16b.x32.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]));
    } else if constexpr (N == 64) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x64b.unpack::16b.x64.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41,"
          "%42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54,"
          "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]),
                     "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]),
                     "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]),
                     "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]),
                     "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]),
                     "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]),
                     "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]),
                     "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]),
                     "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]),
                     "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]),
                     "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]),
                     "r"(src_ptr[62]), "r"(src_ptr[63]));
    } else if constexpr (N == 128) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x64b.unpack::16b.x128.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41,"
          "%42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54,"
          "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67,"
          "%68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80,"
          "%81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93,"
          "%94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105,"
          "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116,"
          "%117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127,"
          "%128};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]),
                     "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]),
                     "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]),
                     "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]),
                     "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]),
                     "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]),
                     "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]),
                     "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]),
                     "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]),
                     "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]),
                     "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]),
                     "r"(src_ptr[62]), "r"(src_ptr[63]), "r"(src_ptr[64]),
                     "r"(src_ptr[65]), "r"(src_ptr[66]), "r"(src_ptr[67]),
                     "r"(src_ptr[68]), "r"(src_ptr[69]), "r"(src_ptr[70]),
                     "r"(src_ptr[71]), "r"(src_ptr[72]), "r"(src_ptr[73]),
                     "r"(src_ptr[74]), "r"(src_ptr[75]), "r"(src_ptr[76]),
                     "r"(src_ptr[77]), "r"(src_ptr[78]), "r"(src_ptr[79]),
                     "r"(src_ptr[80]), "r"(src_ptr[81]), "r"(src_ptr[82]),
                     "r"(src_ptr[83]), "r"(src_ptr[84]), "r"(src_ptr[85]),
                     "r"(src_ptr[86]), "r"(src_ptr[87]), "r"(src_ptr[88]),
                     "r"(src_ptr[89]), "r"(src_ptr[90]), "r"(src_ptr[91]),
                     "r"(src_ptr[92]), "r"(src_ptr[93]), "r"(src_ptr[94]),
                     "r"(src_ptr[95]), "r"(src_ptr[96]), "r"(src_ptr[97]),
                     "r"(src_ptr[98]), "r"(src_ptr[99]), "r"(src_ptr[100]),
                     "r"(src_ptr[101]), "r"(src_ptr[102]), "r"(src_ptr[103]),
                     "r"(src_ptr[104]), "r"(src_ptr[105]), "r"(src_ptr[106]),
                     "r"(src_ptr[107]), "r"(src_ptr[108]), "r"(src_ptr[109]),
                     "r"(src_ptr[110]), "r"(src_ptr[111]), "r"(src_ptr[112]),
                     "r"(src_ptr[113]), "r"(src_ptr[114]), "r"(src_ptr[115]),
                     "r"(src_ptr[116]), "r"(src_ptr[117]), "r"(src_ptr[118]),
                     "r"(src_ptr[119]), "r"(src_ptr[120]), "r"(src_ptr[121]),
                     "r"(src_ptr[122]), "r"(src_ptr[123]), "r"(src_ptr[124]),
                     "r"(src_ptr[125]), "r"(src_ptr[126]), "r"(src_ptr[127]));
    } else {
      asm volatile("trap");
    }
  }
};

// 16 data path lanes, 128b-bit pattern, repeated N times (store)
template <bool Unpack16> class tmem_st_16dp128bNx;

template <> class tmem_st_16dp128bNx<false> {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &dst_addr, uint32_t const *src_ptr) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 64,
                  "N must be a power of 2 and lies between 1 ~ 64");

    if constexpr (N == 1) {
      asm volatile("tcgen05.st.sync.aligned.16x128b.x1.b32"
                   "[%0],"
                   "{%1, %2};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]));
    } else if constexpr (N == 2) {
      asm volatile("tcgen05.st.sync.aligned.16x128b.x2.b32"
                   "[%0],"
                   "{%1, %2, %3, %4};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]), "r"(src_ptr[3]));
    } else if constexpr (N == 4) {
      asm volatile("tcgen05.st.sync.aligned.16x128b.x4.b32"
                   "[%0],"
                   "{%1, %2, %3, %4, %5, %6, %7, %8};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]));
    } else if constexpr (N == 8) {
      asm volatile("tcgen05.st.sync.aligned.16x128b.x8.b32"
                   "[%0],"
                   "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]));
    } else if constexpr (N == 16) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x128b.x16.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]));
    } else if constexpr (N == 32) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x128b.x32.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41,"
          "%42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54,"
          "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]),
                     "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]),
                     "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]),
                     "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]),
                     "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]),
                     "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]),
                     "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]),
                     "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]),
                     "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]),
                     "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]),
                     "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]),
                     "r"(src_ptr[62]), "r"(src_ptr[63]));
    } else if constexpr (N == 64) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x128b.x64.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41,"
          "%42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54,"
          "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67,"
          "%68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80,"
          "%81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93,"
          "%94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105,"
          "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116,"
          "%117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127,"
          "%128};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]),
                     "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]),
                     "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]),
                     "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]),
                     "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]),
                     "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]),
                     "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]),
                     "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]),
                     "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]),
                     "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]),
                     "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]),
                     "r"(src_ptr[62]), "r"(src_ptr[63]), "r"(src_ptr[64]),
                     "r"(src_ptr[65]), "r"(src_ptr[66]), "r"(src_ptr[67]),
                     "r"(src_ptr[68]), "r"(src_ptr[69]), "r"(src_ptr[70]),
                     "r"(src_ptr[71]), "r"(src_ptr[72]), "r"(src_ptr[73]),
                     "r"(src_ptr[74]), "r"(src_ptr[75]), "r"(src_ptr[76]),
                     "r"(src_ptr[77]), "r"(src_ptr[78]), "r"(src_ptr[79]),
                     "r"(src_ptr[80]), "r"(src_ptr[81]), "r"(src_ptr[82]),
                     "r"(src_ptr[83]), "r"(src_ptr[84]), "r"(src_ptr[85]),
                     "r"(src_ptr[86]), "r"(src_ptr[87]), "r"(src_ptr[88]),
                     "r"(src_ptr[89]), "r"(src_ptr[90]), "r"(src_ptr[91]),
                     "r"(src_ptr[92]), "r"(src_ptr[93]), "r"(src_ptr[94]),
                     "r"(src_ptr[95]), "r"(src_ptr[96]), "r"(src_ptr[97]),
                     "r"(src_ptr[98]), "r"(src_ptr[99]), "r"(src_ptr[100]),
                     "r"(src_ptr[101]), "r"(src_ptr[102]), "r"(src_ptr[103]),
                     "r"(src_ptr[104]), "r"(src_ptr[105]), "r"(src_ptr[106]),
                     "r"(src_ptr[107]), "r"(src_ptr[108]), "r"(src_ptr[109]),
                     "r"(src_ptr[110]), "r"(src_ptr[111]), "r"(src_ptr[112]),
                     "r"(src_ptr[113]), "r"(src_ptr[114]), "r"(src_ptr[115]),
                     "r"(src_ptr[116]), "r"(src_ptr[117]), "r"(src_ptr[118]),
                     "r"(src_ptr[119]), "r"(src_ptr[120]), "r"(src_ptr[121]),
                     "r"(src_ptr[122]), "r"(src_ptr[123]), "r"(src_ptr[124]),
                     "r"(src_ptr[125]), "r"(src_ptr[126]), "r"(src_ptr[127]));
    } else {
      asm volatile("trap");
    }
  }
};
template <> class tmem_st_16dp128bNx<true> {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &dst_addr, uint32_t const *src_ptr) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 64,
                  "N must be a power of 2 and lies between 1 ~ 64");

    if constexpr (N == 1) {
      asm volatile("tcgen05.st.sync.aligned.16x128b.unpack::16b.x1.b32"
                   "[%0],"
                   "{%1, %2};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]));
    } else if constexpr (N == 2) {
      asm volatile("tcgen05.st.sync.aligned.16x128b.unpack::16b.x2.b32"
                   "[%0],"
                   "{%1, %2, %3, %4};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]), "r"(src_ptr[3]));
    } else if constexpr (N == 4) {
      asm volatile("tcgen05.st.sync.aligned.16x128b.unpack::16b.x4.b32"
                   "[%0],"
                   "{%1, %2, %3, %4, %5, %6, %7, %8};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]));
    } else if constexpr (N == 8) {
      asm volatile("tcgen05.st.sync.aligned.16x128b.unpack::16b.x8.b32"
                   "[%0],"
                   "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]));
    } else if constexpr (N == 16) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x128b.unpack::16b.x16.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]));
    } else if constexpr (N == 32) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x128b.unpack::16b.x32.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41,"
          "%42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54,"
          "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]),
                     "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]),
                     "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]),
                     "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]),
                     "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]),
                     "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]),
                     "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]),
                     "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]),
                     "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]),
                     "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]),
                     "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]),
                     "r"(src_ptr[62]), "r"(src_ptr[63]));
    } else if constexpr (N == 64) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x128b.unpack::16b.x64.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41,"
          "%42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54,"
          "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67,"
          "%68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80,"
          "%81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93,"
          "%94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105,"
          "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116,"
          "%117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127,"
          "%128};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]),
                     "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]),
                     "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]),
                     "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]),
                     "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]),
                     "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]),
                     "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]),
                     "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]),
                     "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]),
                     "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]),
                     "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]),
                     "r"(src_ptr[62]), "r"(src_ptr[63]), "r"(src_ptr[64]),
                     "r"(src_ptr[65]), "r"(src_ptr[66]), "r"(src_ptr[67]),
                     "r"(src_ptr[68]), "r"(src_ptr[69]), "r"(src_ptr[70]),
                     "r"(src_ptr[71]), "r"(src_ptr[72]), "r"(src_ptr[73]),
                     "r"(src_ptr[74]), "r"(src_ptr[75]), "r"(src_ptr[76]),
                     "r"(src_ptr[77]), "r"(src_ptr[78]), "r"(src_ptr[79]),
                     "r"(src_ptr[80]), "r"(src_ptr[81]), "r"(src_ptr[82]),
                     "r"(src_ptr[83]), "r"(src_ptr[84]), "r"(src_ptr[85]),
                     "r"(src_ptr[86]), "r"(src_ptr[87]), "r"(src_ptr[88]),
                     "r"(src_ptr[89]), "r"(src_ptr[90]), "r"(src_ptr[91]),
                     "r"(src_ptr[92]), "r"(src_ptr[93]), "r"(src_ptr[94]),
                     "r"(src_ptr[95]), "r"(src_ptr[96]), "r"(src_ptr[97]),
                     "r"(src_ptr[98]), "r"(src_ptr[99]), "r"(src_ptr[100]),
                     "r"(src_ptr[101]), "r"(src_ptr[102]), "r"(src_ptr[103]),
                     "r"(src_ptr[104]), "r"(src_ptr[105]), "r"(src_ptr[106]),
                     "r"(src_ptr[107]), "r"(src_ptr[108]), "r"(src_ptr[109]),
                     "r"(src_ptr[110]), "r"(src_ptr[111]), "r"(src_ptr[112]),
                     "r"(src_ptr[113]), "r"(src_ptr[114]), "r"(src_ptr[115]),
                     "r"(src_ptr[116]), "r"(src_ptr[117]), "r"(src_ptr[118]),
                     "r"(src_ptr[119]), "r"(src_ptr[120]), "r"(src_ptr[121]),
                     "r"(src_ptr[122]), "r"(src_ptr[123]), "r"(src_ptr[124]),
                     "r"(src_ptr[125]), "r"(src_ptr[126]), "r"(src_ptr[127]));
    } else {
      asm volatile("trap");
    }
  }
};

// 16 data path lanes, 256b-bit pattern, repeated N times (store)
template <bool Unpack16> class tmem_st_16dp256bNx;

template <> class tmem_st_16dp256bNx<false> {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &dst_addr, uint32_t const *src_ptr) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 32,
                  "N must be a power of 2 and lies between 1 ~ 32");

    if constexpr (N == 1) {
      asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32"
                   "[%0],"
                   "{%1, %2, %3, %4};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]), "r"(src_ptr[3]));
    } else if constexpr (N == 2) {
      asm volatile("tcgen05.st.sync.aligned.16x256b.x2.b32"
                   "[%0],"
                   "{%1, %2, %3, %4, %5, %6, %7, %8};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]));
    } else if constexpr (N == 4) {
      asm volatile("tcgen05.st.sync.aligned.16x256b.x4.b32"
                   "[%0],"
                   "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]));
    } else if constexpr (N == 8) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x256b.x8.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]));
    } else if constexpr (N == 16) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x256b.x16.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41,"
          "%42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54,"
          "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]),
                     "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]),
                     "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]),
                     "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]),
                     "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]),
                     "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]),
                     "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]),
                     "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]),
                     "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]),
                     "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]),
                     "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]),
                     "r"(src_ptr[62]), "r"(src_ptr[63]));
    } else if constexpr (N == 32) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x256b.x32.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41,"
          "%42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54,"
          "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67,"
          "%68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80,"
          "%81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93,"
          "%94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105,"
          "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116,"
          "%117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127,"
          "%128};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]),
                     "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]),
                     "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]),
                     "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]),
                     "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]),
                     "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]),
                     "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]),
                     "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]),
                     "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]),
                     "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]),
                     "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]),
                     "r"(src_ptr[62]), "r"(src_ptr[63]), "r"(src_ptr[64]),
                     "r"(src_ptr[65]), "r"(src_ptr[66]), "r"(src_ptr[67]),
                     "r"(src_ptr[68]), "r"(src_ptr[69]), "r"(src_ptr[70]),
                     "r"(src_ptr[71]), "r"(src_ptr[72]), "r"(src_ptr[73]),
                     "r"(src_ptr[74]), "r"(src_ptr[75]), "r"(src_ptr[76]),
                     "r"(src_ptr[77]), "r"(src_ptr[78]), "r"(src_ptr[79]),
                     "r"(src_ptr[80]), "r"(src_ptr[81]), "r"(src_ptr[82]),
                     "r"(src_ptr[83]), "r"(src_ptr[84]), "r"(src_ptr[85]),
                     "r"(src_ptr[86]), "r"(src_ptr[87]), "r"(src_ptr[88]),
                     "r"(src_ptr[89]), "r"(src_ptr[90]), "r"(src_ptr[91]),
                     "r"(src_ptr[92]), "r"(src_ptr[93]), "r"(src_ptr[94]),
                     "r"(src_ptr[95]), "r"(src_ptr[96]), "r"(src_ptr[97]),
                     "r"(src_ptr[98]), "r"(src_ptr[99]), "r"(src_ptr[100]),
                     "r"(src_ptr[101]), "r"(src_ptr[102]), "r"(src_ptr[103]),
                     "r"(src_ptr[104]), "r"(src_ptr[105]), "r"(src_ptr[106]),
                     "r"(src_ptr[107]), "r"(src_ptr[108]), "r"(src_ptr[109]),
                     "r"(src_ptr[110]), "r"(src_ptr[111]), "r"(src_ptr[112]),
                     "r"(src_ptr[113]), "r"(src_ptr[114]), "r"(src_ptr[115]),
                     "r"(src_ptr[116]), "r"(src_ptr[117]), "r"(src_ptr[118]),
                     "r"(src_ptr[119]), "r"(src_ptr[120]), "r"(src_ptr[121]),
                     "r"(src_ptr[122]), "r"(src_ptr[123]), "r"(src_ptr[124]),
                     "r"(src_ptr[125]), "r"(src_ptr[126]), "r"(src_ptr[127]));
    } else {
      asm volatile("trap");
    }
  }
};
template <> class tmem_st_16dp256bNx<true> {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &dst_addr, uint32_t const *src_ptr) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 32,
                  "N must be a power of 2 and lies between 1 ~ 32");

    if constexpr (N == 1) {
      asm volatile("tcgen05.st.sync.aligned.16x256b.unpack::16b.x1.b32"
                   "[%0],"
                   "{%1, %2, %3, %4};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]), "r"(src_ptr[3]));
    } else if constexpr (N == 2) {
      asm volatile("tcgen05.st.sync.aligned.16x256b.unpack::16b.x2.b32"
                   "[%0],"
                   "{%1, %2, %3, %4, %5, %6, %7, %8};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]));
    } else if constexpr (N == 4) {
      asm volatile("tcgen05.st.sync.aligned.16x256b.unpack::16b.x4.b32"
                   "[%0],"
                   "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]));
    } else if constexpr (N == 8) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x256b.unpack::16b.x8.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]));
    } else if constexpr (N == 16) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x256b.unpack::16b.x16.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41,"
          "%42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54,"
          "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]),
                     "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]),
                     "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]),
                     "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]),
                     "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]),
                     "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]),
                     "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]),
                     "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]),
                     "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]),
                     "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]),
                     "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]),
                     "r"(src_ptr[62]), "r"(src_ptr[63]));
    } else if constexpr (N == 32) {
      asm volatile(
          "tcgen05.st.sync.aligned.16x256b.unpack::16b.x32.b32"
          "[%0],"
          "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
          "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28,"
          "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41,"
          "%42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54,"
          "%55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67,"
          "%68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80,"
          "%81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93,"
          "%94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105,"
          "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116,"
          "%117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127,"
          "%128};\n"
                   :
                   : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]),
                     "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]),
                     "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]),
                     "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]),
                     "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]),
                     "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]),
                     "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]),
                     "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]),
                     "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]),
                     "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]),
                     "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]),
                     "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]),
                     "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]),
                     "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]),
                     "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]),
                     "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]),
                     "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]),
                     "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]),
                     "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]),
                     "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]),
                     "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]),
                     "r"(src_ptr[62]), "r"(src_ptr[63]), "r"(src_ptr[64]),
                     "r"(src_ptr[65]), "r"(src_ptr[66]), "r"(src_ptr[67]),
                     "r"(src_ptr[68]), "r"(src_ptr[69]), "r"(src_ptr[70]),
                     "r"(src_ptr[71]), "r"(src_ptr[72]), "r"(src_ptr[73]),
                     "r"(src_ptr[74]), "r"(src_ptr[75]), "r"(src_ptr[76]),
                     "r"(src_ptr[77]), "r"(src_ptr[78]), "r"(src_ptr[79]),
                     "r"(src_ptr[80]), "r"(src_ptr[81]), "r"(src_ptr[82]),
                     "r"(src_ptr[83]), "r"(src_ptr[84]), "r"(src_ptr[85]),
                     "r"(src_ptr[86]), "r"(src_ptr[87]), "r"(src_ptr[88]),
                     "r"(src_ptr[89]), "r"(src_ptr[90]), "r"(src_ptr[91]),
                     "r"(src_ptr[92]), "r"(src_ptr[93]), "r"(src_ptr[94]),
                     "r"(src_ptr[95]), "r"(src_ptr[96]), "r"(src_ptr[97]),
                     "r"(src_ptr[98]), "r"(src_ptr[99]), "r"(src_ptr[100]),
                     "r"(src_ptr[101]), "r"(src_ptr[102]), "r"(src_ptr[103]),
                     "r"(src_ptr[104]), "r"(src_ptr[105]), "r"(src_ptr[106]),
                     "r"(src_ptr[107]), "r"(src_ptr[108]), "r"(src_ptr[109]),
                     "r"(src_ptr[110]), "r"(src_ptr[111]), "r"(src_ptr[112]),
                     "r"(src_ptr[113]), "r"(src_ptr[114]), "r"(src_ptr[115]),
                     "r"(src_ptr[116]), "r"(src_ptr[117]), "r"(src_ptr[118]),
                     "r"(src_ptr[119]), "r"(src_ptr[120]), "r"(src_ptr[121]),
                     "r"(src_ptr[122]), "r"(src_ptr[123]), "r"(src_ptr[124]),
                     "r"(src_ptr[125]), "r"(src_ptr[126]), "r"(src_ptr[127]));
    } else {
      asm volatile("trap");
    }
  }
};

// 32 data path lanes, composite via 2x16dp (store)
template <bool Unpack16 = false> class tmem_st_32dp64bNx {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &dst_addr, uint32_t const *src_ptr) {
    tmem_st_16dp64bNx<Unpack16>::template copy<N>(dst_addr, src_ptr);
    tmem_st_16dp64bNx<Unpack16>::template copy<N>(dst_addr + (16 << 16), src_ptr + N);
  }
};

// 32 data path lanes, composite via 2x16dp (store)
template <bool Unpack16 = false> class tmem_st_32dp128bNx {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &dst_addr, uint32_t const *src_ptr) {
    tmem_st_16dp128bNx<Unpack16>::template copy<N>(dst_addr, src_ptr);
    tmem_st_16dp128bNx<Unpack16>::template copy<N>(dst_addr + (16 << 16), src_ptr + N * 2);
  }
};

// 32 data path lanes, composite via 2x16dp (store)
template <bool Unpack16 = false> class tmem_st_32dp256bNx {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &dst_addr, uint32_t const *src_ptr) {
    tmem_st_16dp256bNx<Unpack16>::template copy<N>(dst_addr, src_ptr);
    tmem_st_16dp256bNx<Unpack16>::template copy<N>(dst_addr + (16 << 16), src_ptr + N * 4);
  }
};

} // namespace tl
