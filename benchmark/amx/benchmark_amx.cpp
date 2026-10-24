// Build from the repository root with a recent GCC or Clang:
// c++ -std=c++17 -O3 -mavx512f -mavx512bw -mavx512vbmi -mamx-tile \
//   -mamx-bf16 -mamx-int8 -Isrc benchmark/amx/benchmark_amx.cpp \
//   -o benchmark_amx
#include <tl_templates/cpu/gemm.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

struct Shape {
  int m, n, k;
};

static std::uint16_t to_bf16(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>(bits >> 16);
}
static float from_bf16(std::uint16_t value) {
  std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

template <typename Function>
static double benchmark(Function &&function, double operations) {
  constexpr double target_seconds = 0.2;
  int iterations = 1;
  double seconds = 0.0;
  do {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
      function();
    seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    if (seconds < target_seconds)
      iterations *= 2;
  } while (seconds < target_seconds);
  return operations * iterations / seconds / 1e9;
}

static bool run_bf16(const Shape &shape, bool validate = true) {
  const int M = shape.m, N = shape.n, K = shape.k;
  std::vector<std::uint16_t> a(M * K), b(K * N);
  std::vector<float> actual(M * N), expected(M * N);
  for (int i = 0; i < M * K; ++i)
    a[i] = to_bf16(static_cast<float>((i % 17) - 8) / 16.0f);
  for (int i = 0; i < K * N; ++i)
    b[i] = to_bf16(static_cast<float>((i % 13) - 6) / 16.0f);
  if (validate)
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k)
          expected[i * N + j] +=
              from_bf16(a[i * K + k]) * from_bf16(b[k * N + j]);

  const auto kernel = [&] {
    tl_amx_bf16_gemm(reinterpret_cast<float *>(a.data()),
                     reinterpret_cast<float *>(b.data()), actual.data(), M, N,
                     K, K, N, N, 0, 0, true);
  };
  kernel();
  for (int i = 0; validate && i < M * N; ++i) {
    if (std::fabs(actual[i] - expected[i]) > 1e-4f) {
      std::fprintf(stderr, "BF16 mismatch at %d: got=%f ref=%f\n", i, actual[i],
                   expected[i]);
      return false;
    }
  }
  std::printf("BF16 %4dx%4dx%4d  %8.2f GFLOP/s\n", M, N, K,
              benchmark(kernel, 2.0 * M * N * K));
  return true;
}

static bool run_int8(const Shape &shape, bool validate = true) {
  const int M = shape.m, N = shape.n, K = shape.k;
  std::vector<std::int8_t> a(M * K), b(K * N);
  std::vector<std::int32_t> actual(M * N), expected(M * N);
  for (int i = 0; i < M * K; ++i)
    a[i] = static_cast<std::int8_t>((i % 15) - 7);
  for (int i = 0; i < K * N; ++i)
    b[i] = static_cast<std::int8_t>((i % 13) - 6);
  if (validate)
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k)
          expected[i * N + j] +=
              static_cast<std::int32_t>(a[i * K + k]) * b[k * N + j];

  const auto kernel = [&] {
    tl_amx_int8_gemm_ss(a.data(), b.data(), actual.data(), M, N, K, K, N, N, 0,
                        0, true, 0, 0);
  };
  kernel();
  if (validate && actual != expected) {
    const auto mismatch =
        std::mismatch(actual.begin(), actual.end(), expected.begin());
    const auto index = mismatch.first - actual.begin();
    std::fprintf(stderr, "INT8 mismatch at %td: got=%d ref=%d\n", index,
                 actual[index], expected[index]);
    return false;
  }
  std::printf("INT8  %4dx%4dx%4d  %8.2f GOP/s\n", M, N, K,
              benchmark(kernel, 2.0 * M * N * K));
  return true;
}

int main(int argc, char **argv) {
  if (argc == 5) {
    const Shape shape = {std::atoi(argv[2]), std::atoi(argv[3]),
                         std::atoi(argv[4])};
    if (std::string_view(argv[1]) == "bf16")
      return run_bf16(shape, false) ? 0 : 1;
    if (std::string_view(argv[1]) == "int8")
      return run_int8(shape, false) ? 0 : 1;
  }
  const Shape shapes[] = {{16, 16, 32},     {32, 32, 64},    {64, 64, 64},
                          {64, 64, 256},    {128, 128, 128}, {256, 256, 256},
                          {128, 1024, 4096}};
  for (const auto &shape : shapes)
    if (!run_bf16(shape))
      return 1;
  for (const auto &shape : shapes)
    if (!run_int8(shape))
      return 1;
  return 0;
}
