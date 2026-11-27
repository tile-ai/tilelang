// Build from the repository root with a recent GCC or Clang:
// c++ -std=c++17 -O3 -pthread -mamx-tile -mamx-bf16 -mamx-int8 -Isrc \
//   testing/cpp/cpu/amx_smoke.cpp -o amx_smoke
#include <tl_templates/cpu/gemm.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

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

int main() {
  constexpr int M = 17, N = 19, K = 67;
  std::vector<std::uint16_t> a_bf16(M * K), b_bf16(K * N);
  std::vector<float> c_bf16(M * N), ref_bf16(M * N);
  for (int i = 0; i < M * K; ++i)
    a_bf16[i] = to_bf16(static_cast<float>((i % 13) - 6) / 8.0f);
  for (int i = 0; i < K * N; ++i)
    b_bf16[i] = to_bf16(static_cast<float>((i % 11) - 5) / 8.0f);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < K; ++k)
        ref_bf16[i * N + j] +=
            from_bf16(a_bf16[i * K + k]) * from_bf16(b_bf16[k * N + j]);
  std::puts("running AMX BF16 smoke test");
  std::fflush(stdout);
  tl_amx_bf16_gemm(reinterpret_cast<float *>(a_bf16.data()),
                   reinterpret_cast<float *>(b_bf16.data()), c_bf16.data(), M,
                   N, K, K, N, N, 0, 0, true);
  std::puts("AMX BF16 helper returned");
  for (int i = 0; i < M * N; ++i)
    if (std::fabs(c_bf16[i] - ref_bf16[i]) > 1e-4f) {
      std::fprintf(stderr, "BF16 mismatch at %d: got=%f ref=%f\n", i, c_bf16[i],
                   ref_bf16[i]);
      return 1;
    }

  std::vector<std::int8_t> a_int8(M * K), b_int8(K * N);
  std::vector<std::int32_t> c_int8(M * N), ref_int8(M * N);
  for (int i = 0; i < M * K; ++i)
    a_int8[i] = static_cast<std::int8_t>((i % 13) - 6);
  for (int i = 0; i < K * N; ++i)
    b_int8[i] = static_cast<std::int8_t>((i % 11) - 5);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < K; ++k)
        ref_int8[i * N + j] += static_cast<int>(a_int8[i * K + k]) *
                               static_cast<int>(b_int8[k * N + j]);
  std::puts("running AMX INT8 smoke test");
  std::fflush(stdout);
  tl_amx_int8_gemm_ss(a_int8.data(), b_int8.data(), c_int8.data(), M, N, K, K,
                      N, N, 0, 0, true, 0, 0);
  for (int i = 0; i < M * N; ++i)
    if (c_int8[i] != ref_int8[i]) {
      std::fprintf(stderr, "INT8 mismatch at %d: got=%d ref=%d\n", i, c_int8[i],
                   ref_int8[i]);
      return 2;
    }

  std::atomic<int> thread_failures{0};
  std::vector<std::thread> threads;
  for (int thread_id = 0; thread_id < 4; ++thread_id) {
    threads.emplace_back([&] {
      std::vector<float> threaded_c(M * N);
      tl_amx_bf16_gemm(reinterpret_cast<float *>(a_bf16.data()),
                       reinterpret_cast<float *>(b_bf16.data()),
                       threaded_c.data(), M, N, K, K, N, N, 0, 0, true);
      for (int i = 0; i < M * N; ++i) {
        if (std::fabs(threaded_c[i] - ref_bf16[i]) > 1e-4f) {
          ++thread_failures;
          break;
        }
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  if (thread_failures != 0) {
    std::fprintf(stderr, "AMX failed in %d worker threads\n",
                 thread_failures.load());
    return 3;
  }

  std::puts("AMX BF16, INT8, and per-thread permission smoke tests passed");
  return 0;
}
