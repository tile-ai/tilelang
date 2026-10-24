#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__) && defined(__x86_64__)
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#if defined(__AMX_TILE__) && defined(__AMX_BF16__) && defined(__AMX_INT8__) && \
    defined(__x86_64__)
#include <immintrin.h>

namespace tl {
namespace cpu {

struct alignas(64) AMXTileConfig {
  std::uint8_t palette_id;
  std::uint8_t start_row;
  std::uint8_t reserved0[14];
  std::uint16_t colsb[8];
  std::uint8_t reserved1[16];
  std::uint8_t rows[8];
  std::uint8_t reserved2[8];
};
static_assert(sizeof(AMXTileConfig) == 64, "invalid AMX tile config size");

template <typename T> class AlignedScratch {
public:
  AlignedScratch() = default;
  AlignedScratch(const AlignedScratch &) = delete;
  AlignedScratch &operator=(const AlignedScratch &) = delete;
  ~AlignedScratch() { std::free(data_); }

  T *Resize(std::size_t count) {
    if (count > capacity_) {
      std::free(data_);
      const std::size_t bytes = (count * sizeof(T) + 63) & ~std::size_t(63);
      data_ = static_cast<T *>(std::aligned_alloc(64, bytes));
      if (!data_)
        std::abort();
      capacity_ = bytes / sizeof(T);
    }
    return data_;
  }

private:
  T *data_{nullptr};
  std::size_t capacity_{0};
};

inline void EnsureAMXPermission() {
#if defined(__linux__)
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif
  static thread_local bool permission_granted = false;
  if (!permission_granted) {
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) {
      std::perror("TileLang: ARCH_REQ_XCOMP_PERM(XTILEDATA)");
      std::abort();
    }
    permission_granted = true;
  }
#endif
}

inline AMXTileConfig MakeTileConfig2x2(int m0_rows, int m1_rows, int n0_cols,
                                       int n1_cols) {
  AMXTileConfig config{};
  config.palette_id = 1;
  const int m_rows[2] = {m0_rows, m1_rows};
  const int n_cols[2] = {n0_cols, n1_cols};
  for (int mi = 0; mi < 2; ++mi) {
    for (int ni = 0; ni < 2; ++ni) {
      const int tile = mi * 2 + ni;
      const bool active = m_rows[mi] && n_cols[ni];
      config.rows[tile] = static_cast<std::uint8_t>(active ? m_rows[mi] : 0);
      config.colsb[tile] = static_cast<std::uint16_t>(
          active ? n_cols[ni] * sizeof(std::uint32_t) : 0);
    }
    const int tile = 4 + mi;
    config.rows[tile] = static_cast<std::uint8_t>(m_rows[mi]);
    config.colsb[tile] = m_rows[mi] ? 64 : 0;
  }
  for (int ni = 0; ni < 2; ++ni) {
    const int tile = 6 + ni;
    config.rows[tile] = n_cols[ni] ? 16 : 0;
    config.colsb[tile] =
        static_cast<std::uint16_t>(n_cols[ni] * sizeof(std::uint32_t));
  }
  return config;
}

inline AMXTileConfig MakeTileConfig(int k_bytes) {
  AMXTileConfig config{};
  config.palette_id = 1;
  config.rows[0] = 16;
  config.colsb[0] = 64;
  config.rows[1] = 16;
  config.colsb[1] = static_cast<std::uint16_t>(k_bytes);
  config.rows[2] = 16;
  config.colsb[2] = static_cast<std::uint16_t>(k_bytes);
  return config;
}

} // namespace cpu
} // namespace tl

extern "C" inline void
tl_amx_bf16_gemm(float *a_raw, float *b_raw, float *c_raw, std::int32_t M,
                 std::int32_t N, std::int32_t K, std::int32_t lda,
                 std::int32_t ldb, std::int32_t ldc, std::int32_t trans_a,
                 std::int32_t trans_b, bool clear_accum) {
  tl::cpu::EnsureAMXPermission();
  const auto *a = reinterpret_cast<const std::uint16_t *>(a_raw);
  const auto *b = reinterpret_cast<const std::uint16_t *>(b_raw);
  auto *c = c_raw;
  alignas(64) std::uint16_t a_pack[2][16][32];
  const int k_blocks = (K + 31) / 32;
  const int n_tiles = (N + 15) / 16;
  alignas(64) static constexpr std::uint16_t interleave_bf16_indices[32] = {
      0, 32, 1, 33, 2,  34, 3,  35, 4,  36, 5,  37, 6,  38, 7,  39,
      8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47};
  const auto interleave_bf16 = _mm512_load_si512(interleave_bf16_indices);
  if (!trans_a && !trans_b && M % 32 == 0 && N % 32 == 0 && K % 32 == 0) {
    constexpr int n_block = 1024;
    static thread_local tl::cpu::AlignedScratch<std::uint32_t>
        packed_b_block_scratch;
    auto *packed_b_block = packed_b_block_scratch.Resize(
        static_cast<std::size_t>(n_block / 32) * k_blocks * 2 * 16 * 16);
    static thread_local tl::cpu::AlignedScratch<std::uint16_t> packed_a_scratch;
    std::uint16_t *packed_a = nullptr;
    const bool pack_a = N >= 128;
    if (pack_a) {
      packed_a = packed_a_scratch.Resize(static_cast<std::size_t>(M) * K);
      for (int mt = 0; mt < M / 16; ++mt) {
        for (int kb_index = 0; kb_index < k_blocks; ++kb_index) {
          const int k0 = kb_index * 32;
          auto *panel =
              packed_a +
              (static_cast<std::size_t>(mt) * k_blocks + kb_index) * 16 * 32;
          for (int i = 0; i < 16; ++i)
            _mm512_storeu_si512(
                panel + i * 32,
                _mm512_loadu_si512(a + (mt * 16 + i) * lda + k0));
        }
      }
    }
    const auto config = tl::cpu::MakeTileConfig2x2(16, 16, 16, 16);
    _tile_loadconfig(&config);
    for (int n_block_base = 0; n_block_base < N; n_block_base += n_block) {
      const int n_pairs =
          (N - n_block_base < n_block ? N - n_block_base : n_block) / 32;
      for (int kb_index = 0; kb_index < k_blocks; ++kb_index) {
        const int k0 = kb_index * 32;
        for (int kp = 0; kp < 16; ++kp) {
          for (int pair = 0; pair < n_pairs; ++pair) {
            const int n0 = n_block_base + pair * 32;
            auto *panel0 =
                packed_b_block +
                (static_cast<std::size_t>(pair) * k_blocks + kb_index) * 2 *
                    16 * 16;
            auto *panel1 = panel0 + 16 * 16;
            const auto row0 = _mm512_loadu_si512(b + (k0 + kp * 2) * ldb + n0);
            const auto row1 =
                _mm512_loadu_si512(b + (k0 + kp * 2 + 1) * ldb + n0);
            const auto row0_hi =
                _mm512_zextsi256_si512(_mm512_extracti64x4_epi64(row0, 1));
            const auto row1_hi =
                _mm512_zextsi256_si512(_mm512_extracti64x4_epi64(row1, 1));
            _mm512_storeu_si512(
                panel0 + kp * 16,
                _mm512_permutex2var_epi16(row0, interleave_bf16, row1));
            _mm512_storeu_si512(
                panel1 + kp * 16,
                _mm512_permutex2var_epi16(row0_hi, interleave_bf16, row1_hi));
          }
        }
      }
      for (int pair = 0; pair < n_pairs; ++pair) {
        const int n0 = n_block_base + pair * 32;
        for (int m0 = 0; m0 < M; m0 += 32) {
          if (clear_accum) {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);
          } else {
            _tile_loadd(0, c + m0 * ldc + n0, ldc * sizeof(float));
            _tile_loadd(1, c + m0 * ldc + n0 + 16, ldc * sizeof(float));
            _tile_loadd(2, c + (m0 + 16) * ldc + n0, ldc * sizeof(float));
            _tile_loadd(3, c + (m0 + 16) * ldc + n0 + 16, ldc * sizeof(float));
          }
          for (int kb_index = 0; kb_index < k_blocks; ++kb_index) {
            const auto *panel0 =
                packed_b_block +
                (static_cast<std::size_t>(pair) * k_blocks + kb_index) * 2 *
                    16 * 16;
            const auto *panel1 = panel0 + 16 * 16;
            if (pack_a) {
              const auto *a_panel0 =
                  packed_a +
                  (static_cast<std::size_t>(m0 / 16) * k_blocks + kb_index) *
                      16 * 32;
              const auto *a_panel1 =
                  a_panel0 + static_cast<std::size_t>(k_blocks) * 16 * 32;
              _tile_loadd(4, a_panel0, 64);
              _tile_loadd(5, a_panel1, 64);
            } else {
              const int k0 = kb_index * 32;
              _tile_loadd(4, a + m0 * lda + k0, lda * sizeof(std::uint16_t));
              _tile_loadd(5, a + (m0 + 16) * lda + k0,
                          lda * sizeof(std::uint16_t));
            }
            _tile_loadd(6, panel0, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(2, 5, 6);
            _tile_loadd(7, panel1, 64);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(3, 5, 7);
          }
          _tile_stored(0, c + m0 * ldc + n0, ldc * sizeof(float));
          _tile_stored(1, c + m0 * ldc + n0 + 16, ldc * sizeof(float));
          _tile_stored(2, c + (m0 + 16) * ldc + n0, ldc * sizeof(float));
          _tile_stored(3, c + (m0 + 16) * ldc + n0 + 16, ldc * sizeof(float));
        }
      }
    }
    _tile_release();
    return;
  }
  static thread_local tl::cpu::AlignedScratch<std::uint32_t> packed_b_scratch;
  auto *packed_b = packed_b_scratch.Resize(static_cast<std::size_t>(n_tiles) *
                                           k_blocks * 16 * 16);
  int first_generic_nt = 0;
  if (!trans_b && K % 32 == 0) {
    const int paired_tiles = (N / 32) * 2;
    for (int nt = 0; nt < paired_tiles; nt += 2) {
      const int n_base = nt * 16;
      for (int kb_index = 0; kb_index < k_blocks; ++kb_index) {
        const int k0 = kb_index * 32;
        auto *panel0 =
            packed_b +
            (static_cast<std::size_t>(nt) * k_blocks + kb_index) * 16 * 16;
        auto *panel1 = panel0 + static_cast<std::size_t>(k_blocks) * 16 * 16;
        for (int kp = 0; kp < 16; ++kp) {
          const auto row0 =
              _mm512_loadu_si512(b + (k0 + kp * 2) * ldb + n_base);
          const auto row1 =
              _mm512_loadu_si512(b + (k0 + kp * 2 + 1) * ldb + n_base);
          const auto row0_hi =
              _mm512_zextsi256_si512(_mm512_extracti64x4_epi64(row0, 1));
          const auto row1_hi =
              _mm512_zextsi256_si512(_mm512_extracti64x4_epi64(row1, 1));
          _mm512_storeu_si512(
              panel0 + kp * 16,
              _mm512_permutex2var_epi16(row0, interleave_bf16, row1));
          _mm512_storeu_si512(
              panel1 + kp * 16,
              _mm512_permutex2var_epi16(row0_hi, interleave_bf16, row1_hi));
        }
      }
    }
    first_generic_nt = paired_tiles;
  }
  for (int nt = first_generic_nt; nt < n_tiles; ++nt) {
    const int n_base = nt * 16;
    const int nb = N - n_base < 16 ? N - n_base : 16;
    for (int kb_index = 0; kb_index < k_blocks; ++kb_index) {
      const int k0 = kb_index * 32;
      const int kb = K - k0 < 32 ? K - k0 : 32;
      auto *panel =
          packed_b +
          (static_cast<std::size_t>(nt) * k_blocks + kb_index) * 16 * 16;
      if (!trans_b && nb == 16 && kb == 32) {
        for (int kp = 0; kp < 16; ++kp) {
          const auto lo16 = _mm512_zextsi256_si512(
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                  b + (k0 + kp * 2) * ldb + n_base)));
          const auto hi16 = _mm512_zextsi256_si512(
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                  b + (k0 + kp * 2 + 1) * ldb + n_base)));
          _mm512_storeu_si512(
              panel + kp * 16,
              _mm512_permutex2var_epi16(lo16, interleave_bf16, hi16));
        }
        continue;
      }
      std::fill_n(panel, 16 * 16, 0);
      for (int kp = 0; kp < (kb + 1) / 2; ++kp) {
        const int k = kp * 2;
        for (int j = 0; j < nb; ++j) {
          const std::uint16_t lo = trans_b ? b[(n_base + j) * ldb + (k0 + k)]
                                           : b[(k0 + k) * ldb + (n_base + j)];
          const std::uint16_t hi =
              k + 1 < kb ? (trans_b ? b[(n_base + j) * ldb + (k0 + k + 1)]
                                    : b[(k0 + k + 1) * ldb + (n_base + j)])
                         : 0;
          panel[kp * 16 + j] = static_cast<std::uint32_t>(lo) |
                               (static_cast<std::uint32_t>(hi) << 16);
        }
      }
    }
  }

  int configured_mb0 = -1, configured_mb1 = -1;
  int configured_nb0 = -1, configured_nb1 = -1;
  for (int m0 = 0; m0 < M; m0 += 32) {
    const int mb0 = M - m0 < 16 ? M - m0 : 16;
    const int mb1 = M - m0 - 16 < 16 ? M - m0 - 16 : 16;
    const int safe_mb1 = mb1 > 0 ? mb1 : 0;
    for (int n0 = 0; n0 < N; n0 += 32) {
      const int nb0 = N - n0 < 16 ? N - n0 : 16;
      const int nb1 = N - n0 - 16 < 16 ? N - n0 - 16 : 16;
      const int safe_nb1 = nb1 > 0 ? nb1 : 0;
      if (mb0 != configured_mb0 || safe_mb1 != configured_mb1 ||
          nb0 != configured_nb0 || safe_nb1 != configured_nb1) {
        const auto config =
            tl::cpu::MakeTileConfig2x2(mb0, safe_mb1, nb0, safe_nb1);
        _tile_loadconfig(&config);
        configured_mb0 = mb0;
        configured_mb1 = safe_mb1;
        configured_nb0 = nb0;
        configured_nb1 = safe_nb1;
      }
      if (clear_accum) {
        _tile_zero(0);
        if (safe_nb1)
          _tile_zero(1);
        if (safe_mb1) {
          _tile_zero(2);
          if (safe_nb1)
            _tile_zero(3);
        }
      } else {
        _tile_loadd(0, c + m0 * ldc + n0, ldc * sizeof(float));
        if (safe_nb1)
          _tile_loadd(1, c + m0 * ldc + n0 + 16, ldc * sizeof(float));
        if (safe_mb1) {
          _tile_loadd(2, c + (m0 + 16) * ldc + n0, ldc * sizeof(float));
          if (safe_nb1)
            _tile_loadd(3, c + (m0 + 16) * ldc + n0 + 16, ldc * sizeof(float));
        }
      }

      for (int k0 = 0; k0 < K; k0 += 32) {
        const int kb = K - k0 < 32 ? K - k0 : 32;
        for (int mi = 0; mi < (safe_mb1 ? 2 : 1); ++mi) {
          const int mb = mi ? safe_mb1 : mb0;
          const int m_base = m0 + mi * 16;
          if (!trans_a && kb == 32) {
            if (mi == 0)
              _tile_loadd(4, a + m_base * lda + k0,
                          lda * sizeof(std::uint16_t));
            else
              _tile_loadd(5, a + m_base * lda + k0,
                          lda * sizeof(std::uint16_t));
            continue;
          }
          std::memset(a_pack[mi], 0, sizeof(a_pack[mi]));
          for (int i = 0; i < mb; ++i)
            for (int k = 0; k < kb; ++k)
              a_pack[mi][i][k] = trans_a ? a[(k0 + k) * lda + (m_base + i)]
                                         : a[(m_base + i) * lda + (k0 + k)];
          if (mi == 0)
            _tile_loadd(4, a_pack[0], 64);
          else
            _tile_loadd(5, a_pack[1], 64);
        }

        const int kb_index = k0 / 32;
        const int nt = n0 / 16;
        const auto *b_panel0 =
            packed_b +
            (static_cast<std::size_t>(nt) * k_blocks + kb_index) * 16 * 16;
        _tile_loadd(6, b_panel0, 64);
        if (safe_nb1) {
          const auto *b_panel1 =
              packed_b +
              (static_cast<std::size_t>(nt + 1) * k_blocks + kb_index) * 16 *
                  16;
          _tile_loadd(7, b_panel1, 64);
        }
        _tile_dpbf16ps(0, 4, 6);
        if (safe_nb1)
          _tile_dpbf16ps(1, 4, 7);
        if (safe_mb1) {
          _tile_dpbf16ps(2, 5, 6);
          if (safe_nb1)
            _tile_dpbf16ps(3, 5, 7);
        }
      }
      _tile_stored(0, c + m0 * ldc + n0, ldc * sizeof(float));
      if (safe_nb1)
        _tile_stored(1, c + m0 * ldc + n0 + 16, ldc * sizeof(float));
      if (safe_mb1) {
        _tile_stored(2, c + (m0 + 16) * ldc + n0, ldc * sizeof(float));
        if (safe_nb1)
          _tile_stored(3, c + (m0 + 16) * ldc + n0 + 16, ldc * sizeof(float));
      }
    }
  }
  _tile_release();
}

inline void tl_amx_int8_gemm_impl(const void *a_raw, const void *b_raw,
                                  void *c_raw, std::int32_t M, std::int32_t N,
                                  std::int32_t K, std::int32_t lda,
                                  std::int32_t ldb, std::int32_t ldc,
                                  std::int32_t trans_a, std::int32_t trans_b,
                                  bool clear_accum, std::int32_t a_unsigned,
                                  std::int32_t b_unsigned) {
  tl::cpu::EnsureAMXPermission();
  const auto *a = reinterpret_cast<const std::uint8_t *>(a_raw);
  const auto *b = reinterpret_cast<const std::uint8_t *>(b_raw);
  auto *c = reinterpret_cast<std::int32_t *>(c_raw);
  alignas(64) std::uint8_t a_pack[2][16][64];
  const int k_blocks = (K + 63) / 64;
  const int n_tiles = (N + 15) / 16;
  alignas(64) static constexpr std::uint8_t interleave_int8_indices[64] = {
      0,  16, 32, 48, 1,  17, 33, 49, 2,  18, 34, 50, 3,  19, 35, 51,
      4,  20, 36, 52, 5,  21, 37, 53, 6,  22, 38, 54, 7,  23, 39, 55,
      8,  24, 40, 56, 9,  25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
      12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63};
  const auto interleave_int8 = _mm512_load_si512(interleave_int8_indices);
#define TL_AMX_INT8_DOT(C, A, B)                                               \
  do {                                                                         \
    if (a_unsigned && b_unsigned)                                              \
      _tile_dpbuud(C, A, B);                                                   \
    else if (a_unsigned)                                                       \
      _tile_dpbusd(C, A, B);                                                   \
    else if (b_unsigned)                                                       \
      _tile_dpbsud(C, A, B);                                                   \
    else                                                                       \
      _tile_dpbssd(C, A, B);                                                   \
  } while (false)
  if (!trans_a && !trans_b && M % 32 == 0 && N % 32 == 0 && K % 64 == 0) {
    constexpr int n_block = 1024;
    static thread_local tl::cpu::AlignedScratch<std::uint32_t>
        packed_b_block_scratch;
    auto *packed_b_block = packed_b_block_scratch.Resize(
        static_cast<std::size_t>(n_block / 32) * k_blocks * 2 * 16 * 16);
    static thread_local tl::cpu::AlignedScratch<std::uint8_t> packed_a_scratch;
    std::uint8_t *packed_a = nullptr;
    const bool pack_a = N >= 128;
    if (pack_a) {
      packed_a = packed_a_scratch.Resize(static_cast<std::size_t>(M) * K);
      for (int mt = 0; mt < M / 16; ++mt) {
        for (int kb_index = 0; kb_index < k_blocks; ++kb_index) {
          const int k0 = kb_index * 64;
          auto *panel =
              packed_a +
              (static_cast<std::size_t>(mt) * k_blocks + kb_index) * 16 * 64;
          for (int i = 0; i < 16; ++i)
            _mm512_storeu_si512(
                panel + i * 64,
                _mm512_loadu_si512(a + (mt * 16 + i) * lda + k0));
        }
      }
    }
    const auto config = tl::cpu::MakeTileConfig2x2(16, 16, 16, 16);
    _tile_loadconfig(&config);
    for (int n_block_base = 0; n_block_base < N; n_block_base += n_block) {
      const int n_pairs =
          (N - n_block_base < n_block ? N - n_block_base : n_block) / 32;
      for (int kb_index = 0; kb_index < k_blocks; ++kb_index) {
        const int k0 = kb_index * 64;
        for (int kp = 0; kp < 16; ++kp) {
          for (int pair = 0; pair < n_pairs; ++pair) {
            const int n0 = n_block_base + pair * 32;
            auto *panel0 =
                packed_b_block +
                (static_cast<std::size_t>(pair) * k_blocks + kb_index) * 2 *
                    16 * 16;
            auto *panel1 = panel0 + 16 * 16;
            const auto b0 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                    b + (k0 + kp * 4) * ldb + n0));
            const auto b1 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                    b + (k0 + kp * 4 + 1) * ldb + n0));
            const auto b2 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                    b + (k0 + kp * 4 + 2) * ldb + n0));
            const auto b3 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                    b + (k0 + kp * 4 + 3) * ldb + n0));
            auto rows0 = _mm512_castsi128_si512(_mm256_castsi256_si128(b0));
            rows0 = _mm512_inserti32x4(rows0, _mm256_castsi256_si128(b1), 1);
            rows0 = _mm512_inserti32x4(rows0, _mm256_castsi256_si128(b2), 2);
            rows0 = _mm512_inserti32x4(rows0, _mm256_castsi256_si128(b3), 3);
            auto rows1 =
                _mm512_castsi128_si512(_mm256_extracti128_si256(b0, 1));
            rows1 =
                _mm512_inserti32x4(rows1, _mm256_extracti128_si256(b1, 1), 1);
            rows1 =
                _mm512_inserti32x4(rows1, _mm256_extracti128_si256(b2, 1), 2);
            rows1 =
                _mm512_inserti32x4(rows1, _mm256_extracti128_si256(b3, 1), 3);
            _mm512_storeu_si512(panel0 + kp * 16, _mm512_permutexvar_epi8(
                                                      interleave_int8, rows0));
            _mm512_storeu_si512(panel1 + kp * 16, _mm512_permutexvar_epi8(
                                                      interleave_int8, rows1));
          }
        }
      }
      for (int pair = 0; pair < n_pairs; ++pair) {
        const int n0 = n_block_base + pair * 32;
        for (int m0 = 0; m0 < M; m0 += 32) {
          if (clear_accum) {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);
          } else {
            _tile_loadd(0, c + m0 * ldc + n0, ldc * sizeof(std::int32_t));
            _tile_loadd(1, c + m0 * ldc + n0 + 16, ldc * sizeof(std::int32_t));
            _tile_loadd(2, c + (m0 + 16) * ldc + n0,
                        ldc * sizeof(std::int32_t));
            _tile_loadd(3, c + (m0 + 16) * ldc + n0 + 16,
                        ldc * sizeof(std::int32_t));
          }
          for (int kb_index = 0; kb_index < k_blocks; ++kb_index) {
            const auto *panel0 =
                packed_b_block +
                (static_cast<std::size_t>(pair) * k_blocks + kb_index) * 2 *
                    16 * 16;
            const auto *panel1 = panel0 + 16 * 16;
            if (pack_a) {
              const auto *a_panel0 =
                  packed_a +
                  (static_cast<std::size_t>(m0 / 16) * k_blocks + kb_index) *
                      16 * 64;
              const auto *a_panel1 =
                  a_panel0 + static_cast<std::size_t>(k_blocks) * 16 * 64;
              _tile_loadd(4, a_panel0, 64);
              _tile_loadd(5, a_panel1, 64);
            } else {
              const int k0 = kb_index * 64;
              _tile_loadd(4, a + m0 * lda + k0, lda);
              _tile_loadd(5, a + (m0 + 16) * lda + k0, lda);
            }
            _tile_loadd(6, panel0, 64);
            _tile_loadd(7, panel1, 64);
            TL_AMX_INT8_DOT(0, 4, 6);
            TL_AMX_INT8_DOT(1, 4, 7);
            TL_AMX_INT8_DOT(2, 5, 6);
            TL_AMX_INT8_DOT(3, 5, 7);
          }
          _tile_stored(0, c + m0 * ldc + n0, ldc * sizeof(std::int32_t));
          _tile_stored(1, c + m0 * ldc + n0 + 16, ldc * sizeof(std::int32_t));
          _tile_stored(2, c + (m0 + 16) * ldc + n0, ldc * sizeof(std::int32_t));
          _tile_stored(3, c + (m0 + 16) * ldc + n0 + 16,
                       ldc * sizeof(std::int32_t));
        }
      }
    }
    _tile_release();
#undef TL_AMX_INT8_DOT
    return;
  }
  static thread_local tl::cpu::AlignedScratch<std::uint32_t> packed_b_scratch;
  auto *packed_b = packed_b_scratch.Resize(static_cast<std::size_t>(n_tiles) *
                                           k_blocks * 16 * 16);
  for (int nt = 0; nt < n_tiles; ++nt) {
    const int n_base = nt * 16;
    const int nb = N - n_base < 16 ? N - n_base : 16;
    for (int kb_index = 0; kb_index < k_blocks; ++kb_index) {
      const int k0 = kb_index * 64;
      const int kb = K - k0 < 64 ? K - k0 : 64;
      auto *panel =
          packed_b +
          (static_cast<std::size_t>(nt) * k_blocks + kb_index) * 16 * 16;
      if (!trans_b && nb == 16 && kb == 64) {
        for (int kp = 0; kp < 16; ++kp) {
          const auto b0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
              b + (k0 + kp * 4) * ldb + n_base));
          const auto b1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
              b + (k0 + kp * 4 + 1) * ldb + n_base));
          const auto b2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
              b + (k0 + kp * 4 + 2) * ldb + n_base));
          const auto b3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
              b + (k0 + kp * 4 + 3) * ldb + n_base));
          auto rows = _mm512_castsi128_si512(b0);
          rows = _mm512_inserti32x4(rows, b1, 1);
          rows = _mm512_inserti32x4(rows, b2, 2);
          rows = _mm512_inserti32x4(rows, b3, 3);
          _mm512_storeu_si512(panel + kp * 16,
                              _mm512_permutexvar_epi8(interleave_int8, rows));
        }
        continue;
      }
      std::fill_n(panel, 16 * 16, 0);
      for (int kp = 0; kp < (kb + 3) / 4; ++kp) {
        for (int j = 0; j < nb; ++j) {
          std::uint32_t packed = 0;
          for (int q = 0; q < 4 && kp * 4 + q < kb; ++q) {
            const int k = kp * 4 + q;
            const auto value = trans_b ? b[(n_base + j) * ldb + (k0 + k)]
                                       : b[(k0 + k) * ldb + (n_base + j)];
            packed |= static_cast<std::uint32_t>(value) << (q * 8);
          }
          panel[kp * 16 + j] = packed;
        }
      }
    }
  }

  int configured_mb0 = -1, configured_mb1 = -1;
  int configured_nb0 = -1, configured_nb1 = -1;
  for (int m0 = 0; m0 < M; m0 += 32) {
    const int mb0 = M - m0 < 16 ? M - m0 : 16;
    const int mb1 = M - m0 - 16 < 16 ? M - m0 - 16 : 16;
    const int safe_mb1 = mb1 > 0 ? mb1 : 0;
    for (int n0 = 0; n0 < N; n0 += 32) {
      const int nb0 = N - n0 < 16 ? N - n0 : 16;
      const int nb1 = N - n0 - 16 < 16 ? N - n0 - 16 : 16;
      const int safe_nb1 = nb1 > 0 ? nb1 : 0;
      if (mb0 != configured_mb0 || safe_mb1 != configured_mb1 ||
          nb0 != configured_nb0 || safe_nb1 != configured_nb1) {
        const auto config =
            tl::cpu::MakeTileConfig2x2(mb0, safe_mb1, nb0, safe_nb1);
        _tile_loadconfig(&config);
        configured_mb0 = mb0;
        configured_mb1 = safe_mb1;
        configured_nb0 = nb0;
        configured_nb1 = safe_nb1;
      }
      if (clear_accum) {
        _tile_zero(0);
        if (safe_nb1)
          _tile_zero(1);
        if (safe_mb1) {
          _tile_zero(2);
          if (safe_nb1)
            _tile_zero(3);
        }
      } else {
        _tile_loadd(0, c + m0 * ldc + n0, ldc * sizeof(std::int32_t));
        if (safe_nb1)
          _tile_loadd(1, c + m0 * ldc + n0 + 16, ldc * sizeof(std::int32_t));
        if (safe_mb1) {
          _tile_loadd(2, c + (m0 + 16) * ldc + n0, ldc * sizeof(std::int32_t));
          if (safe_nb1)
            _tile_loadd(3, c + (m0 + 16) * ldc + n0 + 16,
                        ldc * sizeof(std::int32_t));
        }
      }

      for (int k0 = 0; k0 < K; k0 += 64) {
        const int kb = K - k0 < 64 ? K - k0 : 64;
        for (int mi = 0; mi < (safe_mb1 ? 2 : 1); ++mi) {
          const int mb = mi ? safe_mb1 : mb0;
          const int m_base = m0 + mi * 16;
          if (!trans_a && kb == 64) {
            if (mi == 0)
              _tile_loadd(4, a + m_base * lda + k0, lda);
            else
              _tile_loadd(5, a + m_base * lda + k0, lda);
            continue;
          }
          std::memset(a_pack[mi], 0, sizeof(a_pack[mi]));
          for (int i = 0; i < mb; ++i)
            for (int k = 0; k < kb; ++k)
              a_pack[mi][i][k] = trans_a ? a[(k0 + k) * lda + (m_base + i)]
                                         : a[(m_base + i) * lda + (k0 + k)];
          if (mi == 0)
            _tile_loadd(4, a_pack[0], 64);
          else
            _tile_loadd(5, a_pack[1], 64);
        }

        const int kb_index = k0 / 64;
        const int nt = n0 / 16;
        const auto *b_panel0 =
            packed_b +
            (static_cast<std::size_t>(nt) * k_blocks + kb_index) * 16 * 16;
        _tile_loadd(6, b_panel0, 64);
        if (safe_nb1) {
          const auto *b_panel1 =
              packed_b +
              (static_cast<std::size_t>(nt + 1) * k_blocks + kb_index) * 16 *
                  16;
          _tile_loadd(7, b_panel1, 64);
        }
#define TL_AMX_INT8_DOT(C, A, B)                                               \
  do {                                                                         \
    if (a_unsigned && b_unsigned)                                              \
      _tile_dpbuud(C, A, B);                                                   \
    else if (a_unsigned)                                                       \
      _tile_dpbusd(C, A, B);                                                   \
    else if (b_unsigned)                                                       \
      _tile_dpbsud(C, A, B);                                                   \
    else                                                                       \
      _tile_dpbssd(C, A, B);                                                   \
  } while (false)
        TL_AMX_INT8_DOT(0, 4, 6);
        if (safe_nb1)
          TL_AMX_INT8_DOT(1, 4, 7);
        if (safe_mb1) {
          TL_AMX_INT8_DOT(2, 5, 6);
          if (safe_nb1)
            TL_AMX_INT8_DOT(3, 5, 7);
        }
#undef TL_AMX_INT8_DOT
      }
      _tile_stored(0, c + m0 * ldc + n0, ldc * sizeof(std::int32_t));
      if (safe_nb1)
        _tile_stored(1, c + m0 * ldc + n0 + 16, ldc * sizeof(std::int32_t));
      if (safe_mb1) {
        _tile_stored(2, c + (m0 + 16) * ldc + n0, ldc * sizeof(std::int32_t));
        if (safe_nb1)
          _tile_stored(3, c + (m0 + 16) * ldc + n0 + 16,
                       ldc * sizeof(std::int32_t));
      }
    }
  }
  _tile_release();
}

#define TL_AMX_INT8_WRAPPER(SUFFIX, A_TYPE, B_TYPE, A_UNSIGNED, B_UNSIGNED)    \
  extern "C" inline void tl_amx_int8_gemm_##SUFFIX(                            \
      A_TYPE *a, B_TYPE *b, std::int32_t *c, std::int32_t M, std::int32_t N,   \
      std::int32_t K, std::int32_t lda, std::int32_t ldb, std::int32_t ldc,    \
      std::int32_t trans_a, std::int32_t trans_b, bool clear_accum,            \
      std::int32_t, std::int32_t) {                                            \
    tl_amx_int8_gemm_impl(a, b, c, M, N, K, lda, ldb, ldc, trans_a, trans_b,   \
                          clear_accum, A_UNSIGNED, B_UNSIGNED);                \
  }

TL_AMX_INT8_WRAPPER(ss, std::int8_t, std::int8_t, 0, 0)
TL_AMX_INT8_WRAPPER(su, std::int8_t, std::uint8_t, 0, 1)
TL_AMX_INT8_WRAPPER(us, std::uint8_t, std::int8_t, 1, 0)
TL_AMX_INT8_WRAPPER(uu, std::uint8_t, std::uint8_t, 1, 1)

#undef TL_AMX_INT8_WRAPPER

#endif
