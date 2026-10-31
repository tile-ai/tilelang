// Build from the repository root with a recent GCC or Clang:
// c++ -std=c++17 -O3 -mavx512f -mavx512bw -mavx512vbmi -mamx-tile \
//   -mamx-bf16 -mamx-int8 -Isrc benchmark/amx/benchmark_amx.cpp \
//   -o benchmark_amx
// Add -fopenmp to benchmark the bf16-parallel and int8-parallel operations.
#include <tl_templates/cpu/gemm.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>

struct Shape {
  int m, n, k;
};

static bool parse_positive_int(std::string_view text, int &value) {
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size() &&
         value > 0;
}

static bool valid_shape(const Shape &shape) {
  const auto product_fits = [](int lhs, int rhs) {
    return lhs <= std::numeric_limits<int>::max() / rhs;
  };
  return product_fits(shape.m, shape.k) && product_fits(shape.k, shape.n) &&
         product_fits(shape.m, shape.n);
}

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

static bool run_bf16(const Shape &shape, bool validate = true,
                     bool parallel = false) {
  const int M = shape.m, N = shape.n, K = shape.k;
  tl::cpu::AlignedScratch<std::uint16_t> a_storage, b_storage;
  tl::cpu::AlignedScratch<float> actual_storage, expected_storage;
  auto *a = a_storage.Resize(M * K);
  auto *b = b_storage.Resize(K * N);
  auto *actual = actual_storage.Resize(M * N);
  auto *expected = expected_storage.Resize(M * N);
  std::fill_n(actual, M * N, 0.0f);
  std::fill_n(expected, M * N, 0.0f);
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
    if (parallel)
      tl_amx_bf16_gemm_parallel(reinterpret_cast<float *>(a),
                                reinterpret_cast<float *>(b), actual, M, N, K,
                                K, N, N, 0, 0, true);
    else
      tl_amx_bf16_gemm(reinterpret_cast<float *>(a),
                       reinterpret_cast<float *>(b), actual, M, N, K, K, N, N,
                       0, 0, true);
  };
  kernel();
  for (int i = 0; validate && i < M * N; ++i) {
    if (std::fabs(actual[i] - expected[i]) > 1e-4f) {
      std::fprintf(stderr, "BF16 mismatch at %d: got=%f ref=%f\n", i, actual[i],
                   expected[i]);
      return false;
    }
  }
  std::printf("BF16%s %4dx%4dx%4d  %8.2f GFLOP/s\n",
              parallel ? " parallel" : "         ", M, N, K,
              benchmark(kernel, 2.0 * M * N * K));
  return true;
}

static bool run_int8(const Shape &shape, bool validate = true,
                     bool parallel = false) {
  const int M = shape.m, N = shape.n, K = shape.k;
  tl::cpu::AlignedScratch<std::int8_t> a_storage, b_storage;
  tl::cpu::AlignedScratch<std::int32_t> actual_storage, expected_storage;
  auto *a = a_storage.Resize(M * K);
  auto *b = b_storage.Resize(K * N);
  auto *actual = actual_storage.Resize(M * N);
  auto *expected = expected_storage.Resize(M * N);
  std::fill_n(actual, M * N, 0);
  std::fill_n(expected, M * N, 0);
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
    if (parallel)
      tl_amx_int8_gemm_ss_parallel(a, b, actual, M, N, K, K, N, N, 0, 0, true,
                                   0, 0);
    else
      tl_amx_int8_gemm_ss(a, b, actual, M, N, K, K, N, N, 0, 0, true, 0, 0);
  };
  kernel();
  if (validate && !std::equal(actual, actual + M * N, expected)) {
    const auto mismatch = std::mismatch(actual, actual + M * N, expected);
    const auto index = mismatch.first - actual;
    std::fprintf(stderr, "INT8 mismatch at %td: got=%d ref=%d\n", index,
                 actual[index], expected[index]);
    return false;
  }
  std::printf("INT8%s %4dx%4dx%4d  %8.2f GOP/s\n",
              parallel ? " parallel" : "         ", M, N, K,
              benchmark(kernel, 2.0 * M * N * K));
  return true;
}

int main(int argc, char **argv) {
  if (argc == 5) {
    Shape shape;
    if (!parse_positive_int(argv[2], shape.m) ||
        !parse_positive_int(argv[3], shape.n) ||
        !parse_positive_int(argv[4], shape.k) || !valid_shape(shape)) {
      std::fprintf(stderr, "M, N, and K must be positive integers whose "
                           "pairwise products fit in int\n");
      return 2;
    }
    const auto operation = std::string_view(argv[1]);
    if (operation == "bf16" || operation == "bf16-parallel")
      return run_bf16(shape, false, operation == "bf16-parallel") ? 0 : 1;
    if (operation == "int8" || operation == "int8-parallel")
      return run_int8(shape, false, operation == "int8-parallel") ? 0 : 1;
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
