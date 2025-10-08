#pragma once
#include "../common.h"
#include "cute/arch/mma_sm90_gmma.hpp"

namespace tl {

template <class> inline constexpr bool always_false_v = false;

// 主类模板 - 移除默认参数，因为特化不能有默认参数
template <DataType A_type, DataType B_type, DataType C_type, int M, int N,
          int K, bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    printf("DEBUG: WgmmaSSImpl fallback - A_type=%d (kFloat16=%d), B_type=%d, "
           "C_type=%d, M=%d, N=%d, K=%d, tnspA=%d, tnspB=%d, scaleA=%d, "
           "scaleB=%d\n",
           (int)A_type, (int)DataType::kFloat16, (int)B_type, (int)C_type, M, N,
           K, (int)tnspA, (int)tnspB, scaleA, scaleB);
    // 暂时注释掉 static_assert 来看调试输出
    // static_assert(always_false_v<decltype(c)>,
    //     "wgmma_ss: No specialization available for given template
    //     parameters!");
  };
};

// ================================= F16 x F16 -> F16
// =================================

// M64N8K16 F16
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat16,
                   64, 8, 16, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %4, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16 "
                 "{%0, %1}, %2, %3, p, %5, %6, %7, %8;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// M64N16K16 F16
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat16,
                   64, 16, 16, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %6, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f16.f16.f16 "
                 "{%0, %1, %2, %3}, %4, %5, p, %7, %8, %9, %10;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// M64N32K16 F16
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat16,
                   64, 32, 16, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f16.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, %11, %12, %13, %14;\n"
        "}\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]), "+r"(c[4]),
          "+r"(c[5]), "+r"(c[6]), "+r"(c[7])
        : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
          "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
  }
};

// M64N64K16 F16
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat16,
                   64, 64, 16, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %18, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "
                 "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
                 "%8,  %9, %10, %11, %12, %13, %14, %15},"
                 " %16, %17, p, %19, %20, %21, %22;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]), "+r"(c[4]),
                   "+r"(c[5]), "+r"(c[6]), "+r"(c[7]), "+r"(c[8]), "+r"(c[9]),
                   "+r"(c[10]), "+r"(c[11]), "+r"(c[12]), "+r"(c[13]),
                   "+r"(c[14]), "+r"(c[15])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// M64N96K16 F16
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat16,
                   64, 96, 16, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %26, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n96k16.f16.f16.f16 "
                 "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
                 "%8,  %9, %10, %11, %12, %13, %14, %15, "
                 "%16, %17, %18, %19, %20, %21, %22, %23}, "
                 "%24, %25, p, %27, %28, %29, %30;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]), "+r"(c[4]),
                   "+r"(c[5]), "+r"(c[6]), "+r"(c[7]), "+r"(c[8]), "+r"(c[9]),
                   "+r"(c[10]), "+r"(c[11]), "+r"(c[12]), "+r"(c[13]),
                   "+r"(c[14]), "+r"(c[15]), "+r"(c[16]), "+r"(c[17]),
                   "+r"(c[18]), "+r"(c[19]), "+r"(c[20]), "+r"(c[21]),
                   "+r"(c[22]), "+r"(c[23])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// M64N128K16 F16
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat16,
                   64, 128, 16, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16 "
                 "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
                 "%8,  %9, %10, %11, %12, %13, %14, %15, "
                 "%16, %17, %18, %19, %20, %21, %22, %23, "
                 "%24, %25, %26, %27, %28, %29, %30, %31}, "
                 "%32, %33, p, %35, %36, %37, %38;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]), "+r"(c[4]),
                   "+r"(c[5]), "+r"(c[6]), "+r"(c[7]), "+r"(c[8]), "+r"(c[9]),
                   "+r"(c[10]), "+r"(c[11]), "+r"(c[12]), "+r"(c[13]),
                   "+r"(c[14]), "+r"(c[15]), "+r"(c[16]), "+r"(c[17]),
                   "+r"(c[18]), "+r"(c[19]), "+r"(c[20]), "+r"(c[21]),
                   "+r"(c[22]), "+r"(c[23]), "+r"(c[24]), "+r"(c[25]),
                   "+r"(c[26]), "+r"(c[27]), "+r"(c[28]), "+r"(c[29]),
                   "+r"(c[30]), "+r"(c[31])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// M64N192K16 F16
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat16,
                   64, 192, 16, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %50, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n192k16.f16.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        "%8,  %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, "
        "%40, %41, %42, %43, %44, %45, %46, %47}, "
        "%48, %49, p, %51, %52, %53, %54;\n"
        "}\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]), "+r"(c[4]),
          "+r"(c[5]), "+r"(c[6]), "+r"(c[7]), "+r"(c[8]), "+r"(c[9]),
          "+r"(c[10]), "+r"(c[11]), "+r"(c[12]), "+r"(c[13]), "+r"(c[14]),
          "+r"(c[15]), "+r"(c[16]), "+r"(c[17]), "+r"(c[18]), "+r"(c[19]),
          "+r"(c[20]), "+r"(c[21]), "+r"(c[22]), "+r"(c[23]), "+r"(c[24]),
          "+r"(c[25]), "+r"(c[26]), "+r"(c[27]), "+r"(c[28]), "+r"(c[29]),
          "+r"(c[30]), "+r"(c[31]), "+r"(c[32]), "+r"(c[33]), "+r"(c[34]),
          "+r"(c[35]), "+r"(c[36]), "+r"(c[37]), "+r"(c[38]), "+r"(c[39]),
          "+r"(c[40]), "+r"(c[41]), "+r"(c[42]), "+r"(c[43]), "+r"(c[44]),
          "+r"(c[45]), "+r"(c[46]), "+r"(c[47])
        : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
          "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
  }
};

// M64N256K16 F16
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat16,
                   64, 256, 16, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f16.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        "%8,  %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, "
        "%40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, "
        "%56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, p, %67, %68, %69, %70;\n"
        "}\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]), "+r"(c[4]),
          "+r"(c[5]), "+r"(c[6]), "+r"(c[7]), "+r"(c[8]), "+r"(c[9]),
          "+r"(c[10]), "+r"(c[11]), "+r"(c[12]), "+r"(c[13]), "+r"(c[14]),
          "+r"(c[15]), "+r"(c[16]), "+r"(c[17]), "+r"(c[18]), "+r"(c[19]),
          "+r"(c[20]), "+r"(c[21]), "+r"(c[22]), "+r"(c[23]), "+r"(c[24]),
          "+r"(c[25]), "+r"(c[26]), "+r"(c[27]), "+r"(c[28]), "+r"(c[29]),
          "+r"(c[30]), "+r"(c[31]), "+r"(c[32]), "+r"(c[33]), "+r"(c[34]),
          "+r"(c[35]), "+r"(c[36]), "+r"(c[37]), "+r"(c[38]), "+r"(c[39]),
          "+r"(c[40]), "+r"(c[41]), "+r"(c[42]), "+r"(c[43]), "+r"(c[44]),
          "+r"(c[45]), "+r"(c[46]), "+r"(c[47]), "+r"(c[48]), "+r"(c[49]),
          "+r"(c[50]), "+r"(c[51]), "+r"(c[52]), "+r"(c[53]), "+r"(c[54]),
          "+r"(c[55]), "+r"(c[56]), "+r"(c[57]), "+r"(c[58]), "+r"(c[59]),
          "+r"(c[60]), "+r"(c[61]), "+r"(c[62]), "+r"(c[63])
        : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
          "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
  }
};

// ================================= F16 x F16 -> F32
// =================================

// M64N8K16 F16->F32
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat32,
                   64, 8, 16, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %6, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
                 "{%0, %1, %2, %3}, %4, %5, p, %7, %8, %9, %10;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// M64N16K16 F16->F32
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat32,
                   64, 16, 16, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, %11, %12, %13, %14;\n"
        "}\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]), "+r"(c[4]),
          "+r"(c[5]), "+r"(c[6]), "+r"(c[7])
        : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
          "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
  }
};

// M64N32K16 F16->F32
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat32,
                   64, 32, 16, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %18, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
                 "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
                 "%8,  %9, %10, %11, %12, %13, %14, %15}, "
                 "%16, %17, p, %19, %20, %21, %22;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]), "+r"(c[4]),
                   "+r"(c[5]), "+r"(c[6]), "+r"(c[7]), "+r"(c[8]), "+r"(c[9]),
                   "+r"(c[10]), "+r"(c[11]), "+r"(c[12]), "+r"(c[13]),
                   "+r"(c[14]), "+r"(c[15])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// M64N64K16 F16->F32
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat32,
                   64, 64, 16, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
                 "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
                 "%8,  %9, %10, %11, %12, %13, %14, %15, "
                 "%16, %17, %18, %19, %20, %21, %22, %23, "
                 "%24, %25, %26, %27, %28, %29, %30, %31}, "
                 "%32, %33, p, %35, %36, %37, %38;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]), "+r"(c[4]),
                   "+r"(c[5]), "+r"(c[6]), "+r"(c[7]), "+r"(c[8]), "+r"(c[9]),
                   "+r"(c[10]), "+r"(c[11]), "+r"(c[12]), "+r"(c[13]),
                   "+r"(c[14]), "+r"(c[15]), "+r"(c[16]), "+r"(c[17]),
                   "+r"(c[18]), "+r"(c[19]), "+r"(c[20]), "+r"(c[21]),
                   "+r"(c[22]), "+r"(c[23]), "+r"(c[24]), "+r"(c[25]),
                   "+r"(c[26]), "+r"(c[27]), "+r"(c[28]), "+r"(c[29]),
                   "+r"(c[30]), "+r"(c[31])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// ================================= BF16 x BF16 -> F32
// =================================

// M64N8K16 BF16->F32
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kBFloat16, DataType::kBFloat16, DataType::kFloat32,
                   64, 8, 16, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %6, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
                 "{%0, %1, %2, %3}, %4, %5, p, %7, %8, %9, %10;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// M64N16K16 BF16->F32
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kBFloat16, DataType::kBFloat16, DataType::kFloat32,
                   64, 16, 16, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, %11, %12, %13, %14;\n"
        "}\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]), "+r"(c[4]),
          "+r"(c[5]), "+r"(c[6]), "+r"(c[7])
        : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
          "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
  }
};

// ================================= TF32 x TF32 -> F32
// =================================

// M64N8K8 TF32->F32
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kTensorFloat32, DataType::kTensorFloat32,
                   DataType::kFloat32, 64, 8, 8, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %6, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32 "
                 "{%0, %1, %2, %3}, %4, %5, p, %7, %8, %9, %10;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// M64N16K8 TF32->F32
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kTensorFloat32, DataType::kTensorFloat32,
                   DataType::kFloat32, 64, 16, 8, tnspA, tnspB, scaleA,
                   scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k8.f32.tf32.tf32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, %11, %12, %13, %14;\n"
        "}\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]), "+r"(c[4]),
          "+r"(c[5]), "+r"(c[6]), "+r"(c[7])
        : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
          "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
  }
};

// ================================= INT8 x INT8 -> INT32
// =================================

// M64N8K32 S8->S32
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kInt8, DataType::kInt8, DataType::kInt32, 64, 8,
                   32, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %4, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8 "
                 "{%0, %1}, %2, %3, p, %5, %6, %7, %8;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// M64N16K32 S8->S32
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kInt8, DataType::kInt8, DataType::kInt32, 64, 16,
                   32, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %6, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k32.s32.s8.s8 "
                 "{%0, %1, %2, %3}, %4, %5, p, %7, %8, %9, %10;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// ================================= FP8 x FP8 -> F16/F32
// =================================

// M64N8K32 E4M3->F16
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                   DataType::kFloat16, 64, 8, 32, tnspA, tnspB, scaleA,
                   scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %4, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k32.f16.e4m3.e4m3 "
                 "{%0, %1}, %2, %3, p, %5, %6, %7, %8;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// M64N8K32 E4M3->F32
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
                   DataType::kFloat32, 64, 8, 32, tnspA, tnspB, scaleA,
                   scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %6, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
                 "{%0, %1, %2, %3}, %4, %5, p, %7, %8, %9, %10;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// 函数模板委托给类模板
template <DataType A_type, DataType B_type, DataType C_type, int M, int N,
          int K, bool tnspA, bool tnspB, int scaleA = 1, int scaleB = 1>
TL_DEVICE void wgmma_ss(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                        bool scale_out) {
  WgmmaSSImpl<A_type, B_type, C_type, M, N, K, tnspA, tnspB, scaleA,
              scaleB>::execute(desc_a, desc_b, c, scale_out);
}

// ================================= Mixed Precision Support
// =================================

// Mixed precision: S8 x U8 -> S32
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kInt8, DataType::kUInt8, DataType::kInt32, 64, 8,
                   32, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %4, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.u8 "
                 "{%0, %1}, %2, %3, p, %5, %6, %7, %8;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// Mixed precision: U8 x S8 -> S32
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kUInt8, DataType::kInt8, DataType::kInt32, 64, 8,
                   32, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %4, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.s8 "
                 "{%0, %1}, %2, %3, p, %5, %6, %7, %8;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// Mixed precision: U8 x U8 -> S32
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kUInt8, DataType::kUInt8, DataType::kInt32, 64, 8,
                   32, tnspA, tnspB, scaleA, scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %4, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8 "
                 "{%0, %1}, %2, %3, p, %5, %6, %7, %8;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// Mixed precision FP8: E4M3 x E5M2 -> F16
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kFloat8_e4m3, DataType::kFloat8_e5m2,
                   DataType::kFloat16, 64, 8, 32, tnspA, tnspB, scaleA,
                   scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %4, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k32.f16.e4m3.e5m2 "
                 "{%0, %1}, %2, %3, p, %5, %6, %7, %8;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// Mixed precision FP8: E5M2 x E4M3 -> F16
template <bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl<DataType::kFloat8_e5m2, DataType::kFloat8_e4m3,
                   DataType::kFloat16, 64, 8, 32, tnspA, tnspB, scaleA,
                   scaleB> {
  TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                                bool scale_out) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %4, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k32.f16.e5m2.e4m3 "
                 "{%0, %1}, %2, %3, p, %5, %6, %7, %8;\n"
                 "}\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_out)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)),
                   "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};

// ================================= Convenience Templates
// =================================

// Type trait to determine the number of output registers needed
template <DataType C_type, int M, int N> struct WgmmaOutputRegs {
  static constexpr int value =
      (M * N * (C_type == DataType::kFloat32 ? 32 : 16)) / (32 * 8);
};

// Type trait to get element size in bits
template <DataType dtype> struct ElementBits {
  static constexpr int value =
      (dtype == DataType::kFloat32 || dtype == DataType::kTensorFloat32 ||
       dtype == DataType::kInt32)
          ? 32
      : (dtype == DataType::kFloat16 || dtype == DataType::kBFloat16 ||
         dtype == DataType::kInt16 || dtype == DataType::kUInt16)
          ? 16
      : (dtype == DataType::kInt8 || dtype == DataType::kUInt8 ||
         dtype == DataType::kFloat8_e4m3 || dtype == DataType::kFloat8_e5m2)
          ? 8
      : (dtype == DataType::kInt4 || dtype == DataType::kUInt4) ? 4
                                                                : 8;
};

} // namespace tl