#include "compat/analyzer.h"
#include "cuda/op/copy.h"
#include "layout/utils.h"
#include "op/prim.h"
#include "op/utils.h"

#include <limits>
#include <sstream>
#include <tvm/tirx/op/memory.h>

namespace tvm::tl::cuda {

namespace {

int64_t TMABytesFromElements(int64_t elements, PrimType type) {
  TVM_FFI_CHECK(elements >= 0 && type.bits() > 0 &&
                    elements <=
                        (std::numeric_limits<int64_t>::max() - 7) / type.bits(),
                ValueError)
      << "TMA byte count exceeds int64";
  return (elements * type.bits() + 7) / 8;
}

PrimExpr TMAGlobalBytesFromElements(PrimExpr elements, PrimType type) {
  int bits = IsFP4UnpackedType(type) ? 4 : type.bits();
  PrimExpr wide = Cast(PrimType::Int(64), elements);
  return FloorDiv(wide * tl::MakeConst(PrimType::Int(64), bits) +
                      tl::MakeConst(PrimType::Int(64), 7),
                  tl::MakeConst(PrimType::Int(64), 8));
}

struct BulkCopyTile {
  // Shape of the copied tile.
  Array<int64_t> tile_shape;
  // tile -> logical shared.
  cute::Layout tile_to_shared_logical;
  // tile -> logical global, using ScaledBasis for modes.
  cute::Layout tile_to_global_mode;
  // logical shared offset.
  cute::IntTuple shared_logical_offset;
  // logical global coords.
  cute::IntTuple global_logical_coords;
};

// The shared tile and the global tile should be of the exact same size. And we
// derive the mappings of the tile to the logical shared and global tensors.
BulkCopyTile ComputeBulkCopyTile(const Array<PrimExpr> &shared_shape,
                                 const Array<Range> &shared_range,
                                 const Array<Range> &global_range) {
  // The tile should have constant shape.
  Array<int64_t> shared_range_extents =
      shared_range.Map([](const Range &range) {
        auto extent = GetConstInt(range->extent);
        ICHECK(extent) << "extent of shared_range: " << range
                       << " is not constant";
        return *extent;
      });
  Array<int64_t> global_range_extents =
      global_range.Map([](const Range &range) {
        auto extent = GetConstInt(range->extent);
        ICHECK(extent) << "extent of global_range: " << range
                       << " is not constant";
        return *extent;
      });

  // Find out the corresponding shared modes and global modes.
  Array<int64_t> tile_shape;
  Array<cute::IntTuple> tile_to_shared_mode_stride, tile_to_global_mode_stride;
  int64_t shared_cur = 0, global_cur = 0;
  while (shared_cur < shared_range_extents.size() &&
         global_cur < global_range_extents.size()) {
    int64_t s_extent = shared_range_extents[shared_cur];
    int64_t g_extent = global_range_extents[global_cur];
    if (s_extent <= 1) {
      shared_cur++;
      continue;
    }
    if (g_extent <= 1) {
      global_cur++;
      continue;
    }
    ICHECK_EQ(s_extent, g_extent)
        << "Shared tile and global tile shape mismatch: tile_shape_shared["
        << shared_cur << "] = " << s_extent << " != " << g_extent
        << " = tile_shape_global[" << global_cur << "]";
    tile_shape.push_back(s_extent);
    tile_to_shared_mode_stride.push_back(cute::E({shared_cur}));
    tile_to_global_mode_stride.push_back(cute::E({global_cur}));
    shared_cur++;
    global_cur++;
  }
  for (; shared_cur < shared_range_extents.size(); shared_cur++) {
    ICHECK_EQ(shared_range_extents[shared_cur], 1)
        << "Shared tile and global tile shape mismatch: "
        << shared_range_extents << " vs " << global_range_extents;
  }
  for (; global_cur < global_range_extents.size(); global_cur++) {
    ICHECK_EQ(global_range_extents[global_cur], 1)
        << "Shared tile and global tile shape mismatch: "
        << shared_range_extents << " vs " << global_range_extents;
  }

  // tile -> logical shared mode
  auto tile_to_shared_mode =
      cute::Layout(tile_shape, std::move(tile_to_shared_mode_stride));
  // logical shared mode -> logical shared
  auto shared_mode_to_shared_logical =
      cute::MakeColumnMajorLayout(shared_shape);
  // tile -> logical shared mode -> logical shared
  auto tile_to_shared_logical =
      cute::Composition(shared_mode_to_shared_logical, tile_to_shared_mode);

  auto shared_coords = shared_range.Map(
      [](const Range &domain) { return cute::IntTuple(domain->begin); });
  auto shared_logical_offset =
      shared_mode_to_shared_logical(cute::IntTupleTuple(shared_coords));

  auto global_logical_coords = global_range.Map(
      [](const Range &domain) { return cute::IntTuple(domain->begin); });

  return {
      tile_shape,
      std::move(tile_to_shared_logical),
      cute::Layout(tile_shape, std::move(tile_to_global_mode_stride)),
      std::move(shared_logical_offset),
      cute::IntTupleTuple(std::move(global_logical_coords)),
  };
}

} // namespace

PrimExpr TMABulkCopyPlan::SharedOffset(std::optional<PrimExpr> rest_idx) const {
  PrimType index_type = PrimType::Int(64);
  PrimExpr offset = tl::CastValue(
      cute::AsConstOrPrimExpr(shared_offset, index_type), index_type);
  if (rest_idx.has_value()) {
    offset = offset + tl::CastValue(cute::AsConstOrPrimExpr(
                                        rest_to_smem(*rest_idx), index_type),
                                    index_type);
  }
  return offset;
}

Expr TMABulkCopyPlan::SharedPointer(std::optional<PrimExpr> rest_idx,
                                    int access_mask) const {
  PrimExpr bit_offset =
      Cast(PrimType::Int(64), SharedOffset(rest_idx)) *
      tl::MakeConst(PrimType::Int(64), shared_tensor.dtype().bits());
  sym::Analyzer analyzer;
  PrimExpr byte_bits = tl::MakeConst(PrimType::Int(64), 8);
  TVM_FFI_CHECK(
      tl::arith::CanProve(analyzer, FloorMod(bit_offset, byte_bits) ==
                                        tl::MakeConst(PrimType::Int(64), 0)),
      ValueError)
      << "TMA shared offsets must be byte aligned";
  DictAttrs attrs(ffi::Map<ffi::String, ffi::Any>{
      {attr::kAccessRegion, NormalizeToBufferRegion(shared_tensor)},
      {attr::kAccessExtent, tl::MakeConst(PrimType::Int(64), box_size)},
      {attr::kAccessMask, PrimExpr(access_mask)}});
  return Call(shared_tensor.ty()->DataPtrType(), tirx::ptr_byte_offset_op(),
              {shared_tensor.data(),
               tl::arith::Simplify(analyzer, FloorDiv(bit_offset, byte_bits))},
              attrs);
}

Array<PrimExpr>
TMABulkCopyPlan::TmaCoords(std::optional<PrimExpr> rest_idx) const {
  Array<PrimExpr> out;
  out.reserve(desc.rank);
  PrimType coordinate_type = PrimType::Int(32);
  cute::IntTuple steps =
      cute::IntTupleTuple(Array<cute::IntTuple>(desc.rank, 0));
  if (rest_idx.has_value()) {
    steps = steps + rest_to_tma_mode(*rest_idx);
  }
  for (int64_t axis = 0; axis < static_cast<int64_t>(desc.rank); axis++) {
    PrimExpr coordinate = tl::CastValue(
        cute::AsConstOrPrimExpr(tma_coords[axis], coordinate_type),
        coordinate_type);
    if (rest_idx.has_value()) {
      coordinate =
          coordinate +
          tl::CastValue(cute::AsConstOrPrimExpr(steps[axis], coordinate_type),
                        coordinate_type);
    }
    out.push_back(coordinate);
  }
  return out;
}

Stmt TMABulkCopyPlan::EmitInstructions(
    const std::function<Stmt(std::optional<PrimExpr>)> &make_copy) const {
  if (rest_size > 1) {
    TVM_FFI_CHECK_LE(rest_size, std::numeric_limits<int32_t>::max(), ValueError)
        << "Too many TMA instructions";
    PrimVar loop_var("tma_box", PrimType::Int(32));
    return For(loop_var, PrimExpr(0), tl::MakeConst(loop_var.ty(), rest_size),
               PrimExpr(1), ForKind::kUnrolled, SeqStmt(make_copy(loop_var)));
  }
  return make_copy(std::nullopt);
}

TMABulkCopyAnalysis
AnalyzeTMABulkCopy(const LowerArgs &lower_args, const TensorVar &global_tensor,
                   TensorVar shared_tensor, const Array<Range> &global_range,
                   const Array<Range> &shared_range,
                   const std::vector<int64_t> &box_dim_caps) {
  auto unsupported = [&](const std::string &reason) -> TMABulkCopyAnalysis {
    DLOG(WARNING) << "TMA bulk copy between " << global_tensor->name << " and "
                  << shared_tensor->name << " unsupported: " << reason;
    return {std::nullopt, reason};
  };

  if (lower_args.layout_map.count(global_tensor)) {
    return unsupported("global tensor " + std::string(global_tensor->name) +
                       " has a non-trivial layout");
  }

  Array<PrimExpr> global_shape = global_tensor.shape();
  ICHECK(box_dim_caps.empty() || box_dim_caps.size() == global_shape.size());
  auto global_pattern = tirx::MatchShapeStridePattern(global_tensor.ty());
  if (!global_pattern.has_value()) {
    return unsupported(
        "global tensor does not have shape-aligned memory strides");
  }
  Array<PrimExpr> global_stride = global_pattern.value().get<1>();
  if (global_range.size() != global_shape.size() ||
      shared_range.size() != shared_tensor.shape().size()) {
    return unsupported("tensor and region ranks differ");
  }
  for (const auto &ranges : {global_range, shared_range}) {
    for (const Range &range : ranges) {
      auto extent = GetConstInt(range->extent);
      if (!extent || *extent <= 0) {
        return unsupported("TMA tile extents must be positive constants");
      }
    }
  }

  // Check if tile shapes match, and compute the tile layouts.
  // E.g., tile_shape = (64,512)
  //       tile_to_shared_logical = (64,512):(1,64)
  //       tile_to_global_mode = (64,512):(1@1,1@2)
  auto [tile_shape, tile_to_shared_logical, tile_to_global_mode,
        shared_logical_offset, global_logical_coords] =
      ComputeBulkCopyTile(shared_tensor.shape(), shared_range, global_range);

  TMABulkCopyPlan plan;
  TMADesc &desc = plan.desc;
  Array<PrimExpr> origin(global_shape.size(), PrimExpr(0));
  desc.global_addr =
      Call(global_tensor.ty()->DataPtrType(), tirx::address_of_op(),
           {TensorLoad(global_tensor, origin)});

  Array<PrimExpr> shared_shape = shared_tensor.shape();
  // logical shared -> physical shared, in TileLang convention
  Layout shared_layout;
  if (lower_args.layout_map.count(shared_tensor)) {
    if (!shared_tensor.ty()->HasDefaultLayout()) {
      return unsupported(
          "shared tensor cannot own both native and TileLang layouts");
    }
    shared_layout = lower_args.layout_map.at(shared_tensor);
    ICHECK(lower_args.buffer_remap.count(shared_tensor))
        << "shared_tensor: " << shared_tensor->name
        << " not found in buffer_remap";
    shared_tensor = lower_args.buffer_remap.at(shared_tensor);
  } else {
    Array<PrimExpr> coordinates;
    for (size_t axis = 0; axis < shared_shape.size(); ++axis) {
      coordinates.push_back(InputPlaceholder(axis));
    }
    shared_layout =
        Layout(shared_shape, shared_tensor.ty()->ElemOffset(coordinates));
  }

  // Convert the TileLang layout to a TensorMap-compatible CuTe layout.
  TMASharedLayoutAnalysis layout_analysis =
      AnalyzeTMASharedLayout(shared_layout, shared_tensor.dtype());
  if (!layout_analysis.encoding.has_value()) {
    return unsupported("shared layout cannot be encoded for TMA: " +
                       layout_analysis.reason);
  }
  const TMASharedLayoutEncoding &layout_encoding =
      layout_analysis.encoding.value();
  // The box loop below may widen the mode; desc.swizzle and the alignment
  // requirement are recorded once the mode is final.
  SwizzleMode swizzle_mode = layout_encoding.swizzle_mode;
  int elem_bits = shared_tensor.dtype().bits();

  // logical shared -> physical shared (without swizzle)
  // E.g., smem_plain = (64,64,8):(64,1,4096)
  auto smem_plain = layout_encoding.composed->layout;

  // tile -> logical shared -> physical shared (without siwzzle)
  // E.g., tile_to_smem_plain = (64,64,8):(64,1,4096)
  auto tile_to_smem_plain =
      cute::Composition(smem_plain, tile_to_shared_logical);

  // physical shared (without swizzle) -> tile
  // E.g., smem_plain_to_tile = (64,64,8):(64,1,4096)
  auto smem_plain_to_tile = cute::RightInverse(tile_to_smem_plain);
  // right_inverse only inverts the bijective stride chain, so a gapped or
  // non-injective SMEM layout would silently cover fewer elements than the
  // tile.
  int64_t inv_size = cute::AsConst(cute::Size(smem_plain_to_tile));
  int64_t tile_size = cute::AsConst(cute::Size(tile_to_smem_plain));
  ICHECK_EQ(inv_size, tile_size)
      << "plain SMEM layout is not a bijection (gapped or non-injective): "
      << "RightInverse covers " << inv_size << " of " << tile_size
      << " elements; try to use annotate_layout to make your SMEM tile "
         "contiguous";

  // physical shared (without swizzle) -> tile -> logical global mode
  // E.g, physical_shared_to_global_mode = (64,64,8):(1@2,1@1,64@2)
  // Merged runs may not exceed the largest per-dim cap; the box loop below
  // enforces the exact cap of whichever dim a run belongs to.
  int64_t coalesce_cap = kTmaMaxBoxDim;
  for (int64_t box_cap : box_dim_caps)
    coalesce_cap = std::max(coalesce_cap, box_cap);
  auto physical_shared_to_global_mode = cute::Coalesce(
      cute::Composition(tile_to_global_mode, smem_plain_to_tile), coalesce_cap);

  // Truncate, because we do not want to start in the middle of a global mode.
  // E.g., smem_rank = 2
  int64_t smem_rank = 0;
  while (smem_rank < cute::Rank(physical_shared_to_global_mode)) {
    auto st =
        cute::BasisValue(physical_shared_to_global_mode->stride[smem_rank]);
    if (!cute::IsConst(st) || AsConst(st) != 1) {
      break;
    }
    smem_rank++;
  }

  if (smem_rank == 0) {
    return unsupported("non-contiguous SMEM innermost stride");
  }

  // physical shared (without swizzle) -> logical global mode (no > 1 stride)
  // Ideally the shape of this is the exact box_dim. But we have
  // several hardware constraints.
  // E.g., tile_gbasis = (64, 64):(1@2, 1@1)
  auto tile_gbasis = cute::Take(physical_shared_to_global_mode, smem_rank);

  // logical global mode -> TMA mode
  Array<cute::IntTuple> gmode_to_tma_mode_shape(global_shape.size(),
                                                cute::IntTuple(int64_t(1)));
  Array<cute::IntTuple> gmode_to_tma_mode_stride(global_shape.size(),
                                                 cute::E({}));
  // TMA mode -> logical global mode
  std::vector<int64_t> tma_mode_to_gmode(global_shape.size());
  // Some of the modes are not included in tile_gbasis and we need to track
  // the modes, and later add the missed modes.
  std::vector<bool> visited_global_modes(global_shape.size(), false);
  // We have some hardware restrictions on tile_gbasis. But we can shrink it to
  // meet some requirements if we need to. This is the modified tile_gbasis.
  Array<cute::IntTuple> box_shape, box_stride;
  int64_t box_elems = 1; // running product of the accepted box dims
  for (int64_t axis = 0; axis < smem_rank; axis++) {
    // Collect the global mode information.
    auto basis = cute::BasisPath(tile_gbasis->stride[axis]);
    ICHECK(basis.size() == 1);
    auto mode = basis[0];
    ICHECK(!visited_global_modes[mode]);

    int64_t box_dim = cute::AsConst(tile_gbasis->shape[axis]);
    int64_t cap = box_dim_caps.empty() ? kTmaMaxBoxDim : box_dim_caps[mode];
    if (axis == 0 && !swizzle_mode.IsNone()) {
      // A TensorMap requires the innermost box bytes to fit within the
      // swizzle span. Recovery reports the narrowest pattern observable in
      // the buffer; a wider mode describes the same layout iff its extra XOR
      // source bits lie beyond the buffer (then it also recovers the same
      // plain layout, so the pairing above stays valid). Widen one mode at a
      // time while the contiguous run wants more than the span; the recovery
      // proof gates each step. b_bits is recast-invariant, so the byte-space
      // width binds the element-space recovery directly.
      int64_t run_bytes =
          TMABytesFromElements(std::min(box_dim, cap), shared_tensor.dtype());
      while (swizzle_mode.ByteWidth() < 128 &&
             run_bytes > swizzle_mode.ByteWidth()) {
        int b_bits = swizzle_mode.BBits() + 1;
        if (!cute::ComposedLayoutFromTileLang(shared_layout, b_bits)
                 .has_value()) {
          break;
        }
        swizzle_mode = SwizzleMode::FromBBits(b_bits);
      }
      int64_t span_elems =
          std::max<int64_t>(1, swizzle_mode.ByteWidth() * 8 / elem_bits);
      cap = std::min(cap, span_elems);
    }
    bool truncate_box = false;
    if (box_dim > cap) {
      // Cap the mode at its largest clean divisor. The leftover and every
      // slower mode fall to the rest loop: the capped mode becomes the
      // box'extent outermost, so the shared prefix stays contiguous. Rest
      // instructions step the shared tile by the box size, so the divisor must
      // also keep the box a whole kTmaSmemAddrAlign multiple.
      int64_t inner = -1;
      for (int64_t divisor = cap; divisor > 1; --divisor) {
        if (box_dim % divisor == 0 &&
            TMABytesFromElements(box_elems * divisor, shared_tensor.dtype()) %
                    kTmaSmemAddrAlign ==
                0) {
          inner = divisor;
          break;
        }
      }
      if (inner == -1) {
        std::ostringstream oss;
        oss << "TMA box dim " << box_dim << " (mode " << axis
            << ") exceeds the cap " << cap << " and cannot be cleanly split";
        return unsupported(oss.str());
      }
      box_dim = inner;
      truncate_box = true;
    }
    // The innermost box dim must be a whole 16-byte multiple: TMA transfers the
    // innermost box as 16-byte vectors.
    if (axis == 0 &&
        TMABytesFromElements(box_dim, shared_tensor.dtype()) % 16 != 0) {
      std::ostringstream oss;
      oss << "TMA innermost box dim " << box_dim
          << " (=" << TMABytesFromElements(box_dim, shared_tensor.dtype())
          << " bytes) is not a whole 16-byte multiple";
      return unsupported(oss.str());
    }
    box_shape.push_back(cute::IntTuple(box_dim));
    box_stride.push_back(tile_gbasis->stride[axis]);
    visited_global_modes[mode] = true;
    gmode_to_tma_mode_stride.Set(mode, cute::E({axis}));
    tma_mode_to_gmode[axis] = mode;
    if (truncate_box)
      break;
  }

  desc.swizzle = swizzle_mode.CanonicalOrdinal();
  // The TMA unit applies the descriptor'extent swizzle pattern to absolute
  // shared-memory addresses, so the buffer base must sit on a swizzle-pattern
  // repeat boundary or the data lands with a shifted phase (silently wrong
  // results, no fault). Report the requirement implied by the chosen swizzle
  // mode so MergeSharedMemoryAllocations can align the buffer accordingly.
  // With an aligned base, per-instruction phases are exactly the ones the
  // recovered global swizzle encodes, so rest instructions may start at any
  // in-buffer position.
  RequireTMASmemAlignment(lower_args, shared_tensor, swizzle_mode);

  // Adopt the validated/shrunk box dims (a no-op when nothing was shrunk).
  smem_rank = static_cast<int64_t>(box_shape.size());
  tile_gbasis = cute::Layout(cute::IntTupleTuple(std::move(box_shape)),
                             cute::IntTupleTuple(std::move(box_stride)));

  // The TMA rank with all the modes.
  // E.g., tma_rank = 3
  int64_t tma_rank = smem_rank;
  // Basically tile_gbasis, but with all the global modes.
  auto tma_gbasis_shape = cute::TupleFields(cute::Wrap(tile_gbasis->shape));
  auto tma_gbasis_stride = cute::TupleFields(cute::Wrap(tile_gbasis->stride));
  for (int64_t axis = static_cast<int64_t>(global_shape.size()) - 1; axis >= 0;
       axis--) {
    if (visited_global_modes[axis])
      continue;
    gmode_to_tma_mode_stride.Set(axis, cute::E({tma_rank}));
    tma_mode_to_gmode[tma_rank] = axis;
    tma_gbasis_shape.push_back(1);
    tma_gbasis_stride.push_back(cute::E({axis}));
    tma_rank++;
  }

  ICHECK_EQ(tma_rank, global_shape.size());
  desc.rank = tma_rank;
  ICHECK(desc.rank >= 1 && desc.rank <= 5) << desc.rank;

  // logical global mode -> global mode in TMA'extent view
  // E.g., (1,1,1):(1@2,1@1,1@0)
  auto gmode_to_tma_mode = cute::Layout(std::move(gmode_to_tma_mode_shape),
                                        std::move(gmode_to_tma_mode_stride));

  // physical shared (without swizzle) -> logical global mode, all global modes
  // E.g., tma_gbasis = (64,64.1):(1@2,1@1,1@0)
  auto tma_gbasis =
      cute::Layout(std::move(tma_gbasis_shape), std::move(tma_gbasis_stride));
  ICHECK_EQ(cute::Rank(tma_gbasis), tma_rank);

  // The size in the box.
  // E.g., box_size = 4096
  plan.box_size = cute::AsConst(cute::Size(tile_gbasis)); // also tma_gbasis
  // The size out of the box.
  // E.g., rest_size = 8
  plan.rest_size =
      cute::AsConst(cute::Size(tile_to_shared_logical)) / plan.box_size;

  // Rest instructions step the shared tile by the box size; the divisor
  // search above keeps split boxes aligned, but a box formed from unpaired
  // slower modes may still violate the instruction address alignment.
  int64_t box_bytes =
      TMABytesFromElements(plan.box_size, shared_tensor.dtype());
  if (plan.rest_size > 1 && box_bytes % kTmaSmemAddrAlign != 0) {
    std::ostringstream oss;
    oss << "TMA rest instructions step the shared tile by " << box_bytes
        << " bytes, which is not a whole " << kTmaSmemAddrAlign
        << "-byte multiple";
    return unsupported(oss.str());
  }

  // Physical shared offset of the tile base.
  plan.shared_offset = smem_plain(shared_logical_offset);

  // Rest index -> physical shared
  // E.g., 8:4096
  plan.rest_to_smem =
      cute::Coalesce(cute::Layout(plan.rest_size, plan.box_size));
  // Rest index -> physical shared -> logical global mode
  // E.g., 8:64@2
  auto rest_to_gmode =
      cute::Composition(physical_shared_to_global_mode, plan.rest_to_smem);
  // Rest index -> logical global mode -> global mode in TMA'extent view
  // E.g., 8:64@0
  plan.rest_to_tma_mode =
      cute::Coalesce(cute::Composition(gmode_to_tma_mode, rest_to_gmode));

  desc.global_shape = Array<PrimExpr>();
  desc.global_stride = Array<PrimExpr>();
  desc.smem_box = Array<PrimExpr>();
  desc.smem_stride = Array<PrimExpr>(tma_rank, PrimExpr(1));

  for (int64_t axis = 0; axis < tma_rank; axis++) {
    desc.smem_box.push_back(
        cute::AsConstOrPrimExpr(tma_gbasis->shape[axis], PrimType::Int(32)));

    desc.global_shape.push_back(global_shape[tma_mode_to_gmode[axis]]);

    PrimExpr elem_stride = global_stride[tma_mode_to_gmode[axis]];
    if (axis == 0 && !IsOne(elem_stride)) {
      std::ostringstream oss;
      oss << "TMA innermost global stride " << elem_stride
          << " != 1 element (transposed/non-contiguous box)";
      return unsupported(oss.str());
    }
    PrimExpr byte_stride =
        TMAGlobalBytesFromElements(elem_stride, global_tensor.dtype());
    if (axis >= 1) {
      if (auto extent = GetConstInt(byte_stride)) {
        if (*extent % 16 != 0 || *extent >= (1LL << 40)) {
          std::ostringstream oss;
          oss << "TMA global stride " << byte_stride
              << " must be a 16-byte multiple below 2^40";
          return unsupported(oss.str());
        }
      }
    }
    desc.global_stride.push_back(byte_stride);
  }

  // TMA coords
  {
    Array<cute::IntTuple> modes;
    modes.reserve(tma_rank);
    for (int64_t axis = 0; axis < tma_rank; axis++)
      modes.push_back(global_logical_coords[tma_mode_to_gmode[axis]]);
    plan.tma_coords = cute::IntTupleTuple(std::move(modes));
  }

  plan.shared_tensor = shared_tensor;
  return {std::move(plan), ""};
}

} // namespace tvm::tl::cuda
