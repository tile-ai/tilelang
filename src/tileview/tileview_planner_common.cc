/*!
 * \file tileview/tileview_planner_common.cc
 * \brief Shared legality and enumeration helpers for TileView planning.
 */

#include "tileview_planner_common.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>

#include <tvm/runtime/logging.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

TrailingTilePattern
MakeTrailingTilePatternImpl(const Array<PrimExpr> &buffer_shape,
                            const std::vector<int> &tile_shape) {
  int buffer_rank = static_cast<int>(buffer_shape.size());
  int tile_rank = static_cast<int>(tile_shape.size());

  TrailingTilePattern pattern;
  pattern.mapped_dims.reserve(tile_rank);
  pattern.tile_shape = tile_shape;
  for (int i = 0; i < tile_rank; ++i) {
    pattern.mapped_dims.push_back(buffer_rank - tile_rank + i);
  }
  return pattern;
}

void AppendRank1Pattern(std::vector<TrailingTilePattern> *patterns,
                        const Buffer &buffer, int tile_width) {
  int buffer_rank = static_cast<int>(buffer->shape.size());
  if (buffer_rank < 1) {
    return;
  }

  patterns->push_back(MakeTrailingTilePatternImpl(buffer->shape, {tile_width}));
}

void AppendRank2Pattern(std::vector<TrailingTilePattern> *patterns,
                        const Buffer &buffer, int tile_height, int tile_width) {
  int buffer_rank = static_cast<int>(buffer->shape.size());
  if (buffer_rank < 2) {
    return;
  }

  patterns->push_back(
      MakeTrailingTilePatternImpl(buffer->shape, {tile_height, tile_width}));
}

bool TileChunkRsramAligned(const Buffer &buffer,
                           const SunmmioTileProcessorConfig &config,
                           int tile_height, int tile_width) {
  if (config.rsram_align_bytes <= 0 || buffer.scope() != kSunmmioScopeRSRAM) {
    return true;
  }
  return IsTileChunkRsramAligned(tile_height, tile_width,
                                 GetElementBits(buffer),
                                 config.rsram_align_bytes);
}

} // namespace

int64_t GetStaticIntValue(const PrimExpr &expr, int64_t fallback) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    return imm->value;
  }
  return fallback;
}

std::vector<ContiguousStep>
GetBufferContiguousSteps(const Buffer &buffer,
                         const Map<Buffer, Layout> &layout_map) {
  if (layout_map.count(buffer)) {
    auto steps = ComputeContiguousTileSteps(layout_map[buffer]);
    if (!steps.empty())
      return steps;
  }
  // Row-major fallback: trailing dim is contiguous, then second-to-last.
  int buffer_rank = static_cast<int>(buffer->shape.size());
  std::vector<ContiguousStep> steps;
  if (buffer_rank >= 1) {
    int64_t w = GetStaticIntValue(buffer->shape[buffer_rank - 1]);
    if (w > 0) {
      steps.push_back({buffer_rank - 1, static_cast<int>(w)});
      if (buffer_rank >= 2) {
        int64_t h = GetStaticIntValue(buffer->shape[buffer_rank - 2]);
        if (h > 0) {
          steps.push_back({buffer_rank - 2, static_cast<int>(h)});
        }
      }
    }
  }
  return steps;
}

int GetElementBits(const Buffer &buffer) {
  ICHECK_EQ(buffer->dtype.lanes(), 1)
      << "TileView planning expects scalar element dtypes, but buffer "
      << buffer->name << " uses lanes=" << buffer->dtype.lanes() << ".";
  return buffer->dtype.bits();
}

int GetCapacityElems(const Buffer &buffer,
                     const SunmmioTileProcessorConfig &config) {
  int element_bits = GetElementBits(buffer);
  ICHECK_GT(element_bits, 0)
      << "TileView planning requires a positive element bit-width for buffer "
      << buffer->name << ".";
  ICHECK_EQ(config.register_bits % element_bits, 0)
      << "Sunmmio tile register size " << config.register_bits
      << " is not divisible by element bit-width " << element_bits
      << " for buffer " << buffer->name << ".";
  return config.register_bits / element_bits;
}

bool IsTileChunkRsramAligned(int tile_height, int tile_width, int element_bits,
                             int rsram_align_bytes) {
  ICHECK_GT(tile_height, 0);
  ICHECK_GT(tile_width, 0);
  ICHECK_GT(element_bits, 0);
  ICHECK_GT(rsram_align_bytes, 0);

  // Every legal trailing tile is a contiguous chunk of h*w elements whose
  // base offsets are multiples of the chunk size. Bit arithmetic keeps
  // sub-byte dtypes exact; the widened product avoids signed overflow for
  // large but valid static int extents.
  using WideUInt = unsigned __int128;
  WideUInt chunk_bits = static_cast<WideUInt>(tile_height) *
                        static_cast<WideUInt>(tile_width) *
                        static_cast<WideUInt>(element_bits);
  WideUInt alignment_bits =
      static_cast<WideUInt>(rsram_align_bytes) * static_cast<WideUInt>(8);
  return chunk_bits % alignment_bits == 0;
}

namespace {

struct TrailingTileEnvelope {
  enum class Kind { kZZBlock, kRowMajorCovered };
  Kind kind;
  int max_height;
  int max_width;
};

struct IntPairHash {
  size_t operator()(const std::pair<int, int> &value) const {
    size_t lhs = std::hash<int>{}(value.first);
    size_t rhs = std::hash<int>{}(value.second);
    return lhs ^ (rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2));
  }
};

bool ConsumeContiguousExtent(const std::vector<ContiguousStep> &steps,
                             size_t *position, int dim, int expected_extent) {
  ICHECK(position);
  ICHECK_GT(expected_extent, 0);
  int64_t extent = 1;
  while (*position < steps.size() && steps[*position].dim == dim &&
         extent < expected_extent) {
    ICHECK_GT(steps[*position].extent, 0);
    if (extent >
        std::numeric_limits<int64_t>::max() / steps[*position].extent) {
      return false;
    }
    extent *= steps[*position].extent;
    ++*position;
  }
  return extent == expected_extent;
}

std::optional<TrailingTileEnvelope>
GetTilesExecutionEnvelope(const Buffer &buffer, int exec_rank,
                          const Map<Buffer, Layout> &layout_map) {
  if (!layout_map.count(buffer)) {
    return std::nullopt;
  }

  Layout layout = layout_map[buffer];
  const auto *cute = layout.as<CuteLayoutNode>();
  if (!cute) {
    return std::nullopt;
  }

  int buffer_rank = static_cast<int>(buffer->shape.size());
  if (buffer_rank < 1 || (exec_rank == 2 && buffer_rank < 2)) {
    return std::nullopt;
  }
  int width_dim = buffer_rank - 1;
  int height_dim = buffer_rank - 2;
  std::vector<ContiguousStep> steps = ComputeContiguousTileSteps(layout);

  if (auto block = sunmmio::GetZZBlockShape(layout)) {
    size_t position = 0;
    if (!ConsumeContiguousExtent(steps, &position, width_dim, block->width)) {
      return std::nullopt;
    }
    if (exec_rank == 2 &&
        !ConsumeContiguousExtent(steps, &position, height_dim, block->height)) {
      return std::nullopt;
    }
    return TrailingTileEnvelope{TrailingTileEnvelope::Kind::kZZBlock,
                                exec_rank == 2 ? block->height : 1,
                                block->width};
  }

  // Only single-level CuteLayouts are admitted as row-major. This excludes
  // ZN and other blocked layouts even if they expose a short contiguous
  // prefix that happens to begin on a trailing dimension.
  for (const Integer &levels : cute->GetDimLevels()) {
    if (levels.IntValue() != 1) {
      return std::nullopt;
    }
  }

  Array<PrimExpr> covered_shape = cute->GetCoveredShape();
  int64_t max_width = GetStaticIntValue(covered_shape[width_dim]);
  int64_t max_height =
      exec_rank == 2 ? GetStaticIntValue(covered_shape[height_dim]) : 1;
  if (max_width <= 0 || max_height <= 0 ||
      max_width > std::numeric_limits<int>::max() ||
      max_height > std::numeric_limits<int>::max()) {
    return std::nullopt;
  }

  size_t position = 0;
  if (!ConsumeContiguousExtent(steps, &position, width_dim,
                               static_cast<int>(max_width)) ||
      (exec_rank == 2 &&
       !ConsumeContiguousExtent(steps, &position, height_dim,
                                static_cast<int>(max_height)))) {
    return std::nullopt;
  }

  return TrailingTileEnvelope{TrailingTileEnvelope::Kind::kRowMajorCovered,
                              static_cast<int>(max_height),
                              static_cast<int>(max_width)};
}

std::vector<int> EnumeratePositiveDivisors(int extent) {
  ICHECK_GT(extent, 0);
  std::vector<int> divisors;
  for (int factor = 1; factor <= extent / factor; ++factor) {
    if (extent % factor != 0) {
      continue;
    }
    divisors.push_back(factor);
    if (factor != extent / factor) {
      divisors.push_back(extent / factor);
    }
  }
  std::sort(divisors.begin(), divisors.end());
  return divisors;
}

std::vector<TrailingTilePattern>
EnumerateLayoutBoundedPatterns(const Buffer &buffer, int exec_rank,
                               const SunmmioTileProcessorConfig &config,
                               const TrailingTileEnvelope &envelope,
                               AlignmentMode alignment_mode) {
  std::vector<TrailingTilePattern> patterns;
  std::unordered_set<int> emitted_rank1;
  std::unordered_set<std::pair<int, int>, IntPairHash> emitted_rank2;
  bool check_alignment = alignment_mode == AlignmentMode::kStrict;

  for (int width : EnumeratePositiveDivisors(envelope.max_width)) {
    if (check_alignment && !TileChunkRsramAligned(buffer, config, 1, width)) {
      continue;
    }
    if (emitted_rank1.insert(width).second) {
      AppendRank1Pattern(&patterns, buffer, width);
    }
    if (exec_rank == 2 && emitted_rank2.emplace(1, width).second) {
      AppendRank2Pattern(&patterns, buffer, 1, width);
    }
  }

  if (exec_rank == 2) {
    for (int height : EnumeratePositiveDivisors(envelope.max_height)) {
      if (check_alignment &&
          !TileChunkRsramAligned(buffer, config, height, envelope.max_width)) {
        continue;
      }
      if (emitted_rank2.emplace(height, envelope.max_width).second) {
        AppendRank2Pattern(&patterns, buffer, height, envelope.max_width);
      }
    }
  }
  return patterns;
}

bool ContainsTileShape(const std::vector<TrailingTilePattern> &patterns,
                       const std::vector<int> &tile_shape) {
  return std::any_of(patterns.begin(), patterns.end(),
                     [&](const TrailingTilePattern &pattern) {
                       return pattern.tile_shape == tile_shape;
                     });
}

} // namespace

bool CanProveDivisible(arith::Analyzer *analyzer, const PrimExpr &value,
                       int factor) {
  PrimExpr remainder = analyzer->Simplify(floormod(value, Integer(factor)));
  return analyzer->CanProve(remainder == make_zero(remainder.dtype()));
}

void RequireDivisible(arith::Analyzer *analyzer, const PrimExpr &value,
                      int factor, const PrimExpr &index, const Buffer &buffer) {
  ICHECK(CanProveDivisible(analyzer, value, factor))
      << "Tile access offset " << value << " is not divisible by tile size "
      << factor << " in index " << index << " for buffer " << buffer->name
      << ".";
}

int NormalizeMappedDim(const PrimExpr &expr, int ndim) {
  const auto *imm = expr.as<IntImmNode>();
  ICHECK(imm) << "TileView index_map entries must be IntImm, but got " << expr;
  int mapped_dim = static_cast<int>(imm->value);
  if (mapped_dim < 0) {
    mapped_dim += ndim;
  }
  ICHECK(mapped_dim >= 0 && mapped_dim < ndim)
      << "TileView index_map entry " << expr << " is out of bounds for rank "
      << ndim << ".";
  return mapped_dim;
}

bool HasTrailingIndexMap(const TileView &tv, int exec_rank) {
  if (static_cast<int>(tv->TileDim()) != exec_rank) {
    return false;
  }

  int buf_ndim = static_cast<int>(tv->BufferShape().size());
  int first_exec_dim = buf_ndim - exec_rank;
  for (int i = 0; i < exec_rank; ++i) {
    int mapped_dim = NormalizeMappedDim(tv->IndexMap()[i], buf_ndim);
    if (mapped_dim != first_exec_dim + i) {
      return false;
    }
  }
  return true;
}

int64_t TileElements(const std::vector<int> &tile_shape) {
  int64_t elems = 1;
  for (int extent : tile_shape) {
    ICHECK_GT(extent, 0)
        << "TileView planning requires positive tile extents, but got "
        << extent << ".";
    ICHECK_LE(static_cast<int64_t>(extent),
              std::numeric_limits<int64_t>::max() / elems)
        << "TileView element count overflows int64.";
    elems *= extent;
  }
  return elems;
}

int64_t TileElements(const Array<PrimExpr> &tile_shape) {
  int64_t elems = 1;
  for (const PrimExpr &extent : tile_shape) {
    int64_t value = GetStaticIntValue(extent);
    ICHECK_GT(value, 0)
        << "TileView planning requires a positive static tile extent, but got "
        << extent << ".";
    ICHECK_LE(value, std::numeric_limits<int64_t>::max() / elems)
        << "TileView element count overflows int64.";
    elems *= value;
  }
  return elems;
}

Array<PrimExpr> MakeTileShapeExpr(const std::vector<int> &tile_shape) {
  Array<PrimExpr> tile_shape_expr;
  tile_shape_expr.reserve(tile_shape.size());
  for (int extent : tile_shape) {
    tile_shape_expr.push_back(Integer(extent));
  }
  return tile_shape_expr;
}

Array<PrimExpr> MakeCanonicalIndexMap(int buffer_rank, int tile_rank) {
  Array<PrimExpr> index_map;
  for (int i = 0; i < tile_rank; ++i) {
    index_map.push_back(Integer(i - tile_rank));
  }
  return index_map;
}

TileView MakeTrailingTileView(const Array<PrimExpr> &buffer_shape,
                              const TrailingTilePattern &pattern) {
  return makeTileView(
      buffer_shape, MakeTileShapeExpr(pattern.tile_shape),
      MakeCanonicalIndexMap(static_cast<int>(buffer_shape.size()),
                            static_cast<int>(pattern.tile_shape.size())));
}

std::vector<TrailingTilePattern> EnumerateInferredTrailingTilePatterns(
    const Buffer &buffer, int exec_rank, const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer,
    TilePatternOptions options) {
  ICHECK(exec_rank == 1 || exec_rank == 2)
      << "TileView planning expects execution rank 1 or 2, but got "
      << exec_rank << ".";

  std::vector<TrailingTilePattern> patterns;
  int buffer_rank = static_cast<int>(buffer->shape.size());
  if (buffer_rank < 1) {
    return patterns;
  }

  if (options.extent_policy == TileExtentPolicy::kTilesExecutionLayoutBounded ||
      options.extent_policy == TileExtentPolicy::kReductionLayoutBounded) {
    if (auto envelope =
            GetTilesExecutionEnvelope(buffer, exec_rank, layout_map)) {
      return EnumerateLayoutBoundedPatterns(buffer, exec_rank, config,
                                            *envelope, options.alignment_mode);
    }
  }

  int capacity_elems = GetCapacityElems(buffer, config);
  auto steps = GetBufferContiguousSteps(buffer, layout_map);

  // Identify the trailing buffer dimensions (width = last, height =
  // second-to-last).
  int width_dim = buffer_rank - 1;
  int height_dim = buffer_rank - 2;

  // RSRAM access alignment: the contiguous tile chunk must be a multiple of
  // rsram_align_bytes (see TileChunkRsramAligned). Relaxed mode skips it.
  bool check_alignment = options.alignment_mode == AlignmentMode::kStrict;

  // Prefix-partial step walk.
  //
  // The contiguous step sequence defines a linear memory ordering.  A legal
  // tile of shape (h, w) must correspond to a contiguous sub-sequence in that
  // ordering.  At step k, the only new tiles are those formed by fully
  // consuming steps [0..k-1] and *partially* consuming step k:
  //
  //   - If step k is a width step:  h = forced_h, w = forced_w * partial
  //   - If step k is a height step: h = forced_h * partial, w = forced_w
  //
  // where `forced_w` / `forced_h` are the accumulated width/height extents
  // from all fully consumed prior steps.  `partial` ranges from 1 to the
  // step's extent (capped by register capacity).
  //
  // This model correctly handles:
  //   - Row-major: single W step of full row width → all sub-widths valid
  //   - ZZ block: W then H step → inner-block (h,w) combos, then outer steps
  //   - ZN: H-first ordering → only narrow width tiles
  //
  // Dedup sets avoid emitting the same (h,w) at multiple step boundaries.

  std::unordered_set<int> emitted_rank1;
  std::unordered_set<std::pair<int, int>, IntPairHash> emitted_rank2;

  int forced_w = 1;        // min width forced by fully consumed W steps
  int forced_h = 1;        // min height forced by fully consumed H steps
  bool rank1_done = false; // rank-1 only from leading W steps
  bool had_width_step = false;

  for (const auto &step : steps) {
    if (step.dim == width_dim) {
      had_width_step = true;
      for (int partial = 1; partial <= step.extent; ++partial) {
        int w = forced_w * partial;
        int h = forced_h;
        if (h * w > capacity_elems)
          break;
        if (check_alignment && !TileChunkRsramAligned(buffer, config, h, w))
          continue;

        // Rank-1: only from leading width steps (before any non-width step).
        if (!rank1_done && h == 1 && !emitted_rank1.count(w)) {
          emitted_rank1.insert(w);
          AppendRank1Pattern(&patterns, buffer, w);
        }

        // Rank-2.
        if (exec_rank == 2 && buffer_rank >= 2) {
          std::pair<int, int> key{h, w};
          if (!emitted_rank2.count(key)) {
            emitted_rank2.insert(key);
            AppendRank2Pattern(&patterns, buffer, h, w);
          }
        }
      }
      forced_w *= step.extent;
    } else if (step.dim == height_dim) {
      rank1_done = true;
      for (int partial = 1; partial <= step.extent; ++partial) {
        int h = forced_h * partial;
        int w = forced_w;
        if (h * w > capacity_elems)
          break;
        if (check_alignment && !TileChunkRsramAligned(buffer, config, h, w))
          continue;

        if (exec_rank == 2 && buffer_rank >= 2) {
          std::pair<int, int> key{h, w};
          if (!emitted_rank2.count(key)) {
            emitted_rank2.insert(key);
            AppendRank2Pattern(&patterns, buffer, h, w);
          }
        }
      }
      forced_h *= step.extent;
    } else {
      // Non-trailing dim step — cannot tile into this dimension,
      // so subsequent steps are unreachable for trailing tiles.
      break;
    }
  }

  // If no steps contributed to width (empty steps or no width-dim steps),
  // fall back to brute-force enumeration up to capacity (preserves behavior
  // for buffers with no CuteLayout and fully dynamic shapes).
  if (!had_width_step) {
    for (int w = 1; w <= capacity_elems; ++w) {
      AppendRank1Pattern(&patterns, buffer, w);
    }
    if (exec_rank == 2 && buffer_rank >= 2) {
      int64_t row_width = GetStaticIntValue(buffer->shape[width_dim]);
      if (row_width > 0) {
        int max_w = std::min(capacity_elems, static_cast<int>(row_width));
        for (int w = 1; w <= max_w; ++w) {
          AppendRank2Pattern(&patterns, buffer, 1, w);
        }
        if (row_width <= capacity_elems) {
          int max_h = capacity_elems / static_cast<int>(row_width);
          for (int h = 2; h <= max_h; ++h) {
            AppendRank2Pattern(&patterns, buffer, h,
                               static_cast<int>(row_width));
          }
        }
      }
    }
  }

  return patterns;
}

TrailingTilePattern ValidateManualTrailingTileView(
    const Buffer &buffer, const TileView &manual_tv, int exec_rank,
    const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config, arith::Analyzer *analyzer,
    TilePatternOptions options, const char *usage) {
  int tv_rank = static_cast<int>(manual_tv->TileDim());
  int buffer_rank = static_cast<int>(buffer->shape.size());
  ICHECK_EQ(static_cast<int>(manual_tv->BufferShape().size()), buffer_rank)
      << usage << " rank mismatch for buffer " << buffer->name << ".";

  if (exec_rank == 1) {
    ICHECK_EQ(tv_rank, 1) << usage << " is incompatible with buffer "
                          << buffer->name << ", which requires a " << tv_rank
                          << "D TileView.";
  } else {
    ICHECK(tv_rank == 1 || tv_rank == 2)
        << usage << " currently supports only 1D or trailing-2D execution "
        << "TileViews for buffer " << buffer->name << ", but got rank "
        << tv_rank << ".";
  }

  ICHECK(HasTrailingIndexMap(manual_tv, tv_rank))
      << usage << " must target trailing " << tv_rank << " buffer dims for "
      << buffer->name << ".";

  std::optional<TrailingTileEnvelope> execution_envelope;
  if (options.extent_policy == TileExtentPolicy::kTilesExecutionLayoutBounded ||
      options.extent_policy == TileExtentPolicy::kReductionLayoutBounded) {
    execution_envelope =
        GetTilesExecutionEnvelope(buffer, exec_rank, layout_map);
  }
  bool register_bounded = !execution_envelope.has_value();
  int capacity_elems = GetCapacityElems(buffer, config);
  int width_dim = buffer_rank - 1;
  int width = GetStaticIntValue(manual_tv->TileShape()[tv_rank - 1]);
  ICHECK_GT(width, 0) << usage
                      << " must use a positive static tile width for buffer "
                      << buffer->name << ".";
  if (register_bounded) {
    ICHECK_LE(width, capacity_elems)
        << usage << " width " << width
        << " exceeds the Sunmmio register capacity of " << capacity_elems
        << " elements for buffer " << buffer->name << ".";
  }

  // Prefix-step legality: the manual tile must be achievable by walking
  // the contiguous step sequence.  At each step the only new tiles are
  // formed by fully consuming prior steps and partially consuming the
  // current one — the same model used by enumeration.
  auto steps = GetBufferContiguousSteps(buffer, layout_map);
  int height_dim = buffer_rank - 2;

  if (tv_rank == 1) {
    // Rank-1: width must come from leading width steps only (before any
    // non-width step).
    if (!steps.empty()) {
      bool valid = false;
      int fw = 1;
      for (const auto &step : steps) {
        if (step.dim != width_dim)
          break;
        if (width >= fw && width <= fw * step.extent && width % fw == 0) {
          valid = true;
          break;
        }
        fw *= step.extent;
      }
      ICHECK(valid)
          << usage << " width " << width
          << " is not contiguous in the layout step structure for buffer "
          << buffer->name << ".";
    }
    ICHECK(TileChunkRsramAligned(buffer, config, 1, width))
        << usage << " width " << width << " elements is not a multiple of the "
        << config.rsram_align_bytes << "-byte RSRAM alignment for buffer "
        << buffer->name << ".";
    if (execution_envelope.has_value()) {
      ICHECK(ContainsTileShape(EnumerateLayoutBoundedPatterns(
                                   buffer, exec_rank, config,
                                   *execution_envelope, options.alignment_mode),
                               {width}))
          << usage << " width " << width
          << " is outside the layout-bounded execution envelope for buffer "
          << buffer->name << ".";
    }
    return MakeTrailingTilePatternImpl(buffer->shape, {width});
  }

  // Rank-2 from here.
  int height = GetStaticIntValue(manual_tv->TileShape()[0]);
  ICHECK_GT(height, 0) << usage
                       << " must use a positive static tile height for buffer "
                       << buffer->name << ".";
  if (register_bounded) {
    ICHECK_LE(TileElements(std::vector<int>{height, width}), capacity_elems)
        << usage << " shape (" << height << ", " << width
        << ") exceeds the Sunmmio register capacity of " << capacity_elems
        << " elements for buffer " << buffer->name << ".";
  }
  ICHECK(TileChunkRsramAligned(buffer, config, height, width))
      << usage << " shape (" << height << ", " << width
      << ") is not a multiple of the " << config.rsram_align_bytes
      << "-byte RSRAM alignment for buffer " << buffer->name << ".";

  if (execution_envelope.has_value()) {
    ICHECK(ContainsTileShape(EnumerateLayoutBoundedPatterns(
                                 buffer, exec_rank, config, *execution_envelope,
                                 options.alignment_mode),
                             {height, width}))
        << usage << " shape (" << height << ", " << width
        << ") is outside the layout-bounded execution envelope for buffer "
        << buffer->name << ".";
  }

  if (!steps.empty()) {
    // Prefix-step legality for rank-2: walk steps, checking if the
    // manual (height, width) matches some partial consumption.
    bool valid = false;
    int fw = 1, fh = 1;
    for (const auto &step : steps) {
      if (step.dim == width_dim) {
        // At a width step: tile must have h == forced_h and
        // w == forced_w * partial for some 1 <= partial <= extent.
        if (height == fh && width >= fw && width <= fw * step.extent &&
            width % fw == 0) {
          valid = true;
          break;
        }
        fw *= step.extent;
      } else if (step.dim == height_dim) {
        // At a height step: tile must have w == forced_w and
        // h == forced_h * partial for some 1 <= partial <= extent.
        if (width == fw && height >= fh && height <= fh * step.extent &&
            height % fh == 0) {
          valid = true;
          break;
        }
        fh *= step.extent;
      } else {
        break;
      }
    }
    ICHECK(valid)
        << usage << " shape (" << height << ", " << width
        << ") is not contiguous in the layout step structure for buffer "
        << buffer->name << ".";
  } else {
    // Row-major fallback for buffers with no layout info (dynamic shapes).
    int64_t row_width = GetStaticIntValue(buffer->shape[width_dim]);
    ICHECK_GT(row_width, 0)
        << usage << " requires a static trailing row width for buffer "
        << buffer->name << ".";
    ICHECK_LE(width, row_width)
        << usage << " width " << width << " exceeds trailing buffer dimension "
        << row_width << " for buffer " << buffer->name << ".";
    if (height > 1) {
      ICHECK_EQ(width, row_width)
          << usage << " multi-row width must match trailing buffer dimension "
          << row_width << " for buffer " << buffer->name << ".";
    }
  }

  return MakeTrailingTilePatternImpl(buffer->shape, {height, width});
}

} // namespace tl
} // namespace tvm
