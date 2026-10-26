#include "compat/analyzer.h"
/*!
 * \file ascend/layout/ascend_layouts.cc
 * \brief Ascend fractal/NZ layout constructors and analysis (moved out of the
 *  generic src/layout/ tree).
 */

#include "support/check.h"
#include <algorithm>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/memory.h>

#include "ascend/layout/ascend_layouts.h"
#include "layout/layout.h"
#include "layout/printer.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

// Expand a 2-D base layout to the rank of \p buffer by prepending the leading
// (batch) dimensions of the buffer shape. Local copy of the swizzle helper so
// this file does not depend on gemm_layouts.cc internals.
static Layout ExpandLayout2D(const Layout &base, const TensorVar &buffer) {
  Array<PrimExpr> leading_shape;
  leading_shape.reserve(buffer.shape().size() - 2);
  for (size_t i = 0; i + 2 < buffer.shape().size(); ++i) {
    leading_shape.push_back(buffer.shape()[i]);
  }
  return base->Expand(leading_shape);
}

// Ascend fractal layout for Cube operands. A logical [.., rows, cols] tile is
// physically stored as [.., cols / C0, rows / 16, 16, C0], matching the MTE2
// ND2NZ convention where the contiguous axis (cols) is grouped by C0 and is the
// outer physical major. Within a fractal the row lane remains stride C0 and the
// column lane remains stride 1. MajorK means cols is the reduce-K axis; MajorMN
// means cols is the output M/N axis. The legacy NZ API remains an alias of
// MajorK while the frontend migrates to explicit major names.
static constexpr int kAscendRowFractal = 16;

namespace {

// Canonical Ascend layouts retain their semantic kind and logical axis roles.
// Small extents can simplify different forward maps to the same expressions,
// so these properties cannot always be reconstructed from the map alone.
class AscendFractalLayoutNode : public LayoutNode {
public:
  AscendFractalLayoutNode() = default;
  AscendFractalLayoutNode(Array<PrimExpr> input_size,
                          Array<PrimExpr> forward_index, AscendFractalKind kind)
      : LayoutNode(std::move(input_size), std::move(forward_index)),
        kind_(static_cast<int>(kind)) {}

  AscendFractalKind Kind() const {
    return static_cast<AscendFractalKind>(kind_);
  }
  Layout Expand(const Array<PrimExpr> &leading_shape) const final;

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<AscendFractalLayoutNode>().def_ro(
        "kind", &AscendFractalLayoutNode::kind_);
    refl::TypeAttrDef<AscendFractalLayoutNode>().attr(
        script::printer::type_attr::kDocTranslate.name,
        script::printer::FDocTranslate::FromNative<&DocTranslate>());
  }

  static ffi::Optional<script::printer::ExprDoc>
  DocTranslate(script::printer::DocTranslatorObj *translator,
               ffi::AnyView value, const ffi::Object *) {
    return TranslateLayoutFields(translator, value,
                                 "AscendFractalLayoutFromIndexMap",
                                 {"input_size", "index_map", "kind"});
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.AscendFractalLayout",
                                    AscendFractalLayoutNode, LayoutNode);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;

private:
  int kind_{-1};
};

Layout MakeTaggedAscendFractalLayout(Array<PrimExpr> input_size,
                                     Array<PrimExpr> forward_index,
                                     AscendFractalKind kind) {
  ObjectPtr<LayoutNode> node = make_object<AscendFractalLayoutNode>(
      std::move(input_size), std::move(forward_index), kind);
  return Layout(std::move(node));
}

std::pair<LogicalAxis, LogicalAxis> GetAxesForKind(AscendFractalKind kind) {
  ICHECK(kind == AscendFractalKind::kMajorK ||
         kind == AscendFractalKind::kMajorMN ||
         kind == AscendFractalKind::kL0C || kind == AscendFractalKind::kSF);
  if (kind == AscendFractalKind::kMajorMN) {
    return {LogicalAxis::kRow, LogicalAxis::kCol};
  }
  return {LogicalAxis::kCol, LogicalAxis::kRow};
}

Layout
AscendFractalLayoutNode::Expand(const Array<PrimExpr> &leading_shape) const {
  Layout expanded = LayoutNode::Expand(leading_shape);
  if (leading_shape.empty()) {
    return expanded;
  }
  return MakeTaggedAscendFractalLayout(expanded->InputShape(),
                                       expanded->GetForwardIndex(), Kind());
}

} // namespace

bool IsAscendFractalLayout(const Layout &layout) {
  return layout.defined() && layout.as<AscendFractalLayoutNode>() != nullptr;
}

// Column fractal C0 = 32 bytes / element_size, expressed from element bits.
int AscendC0(int element_bits) {
  ICHECK(element_bits > 0 && 256 % element_bits == 0)
      << "Ascend NZ layout: unsupported element bits=" << element_bits;
  return 256 / element_bits;
}

static Layout MakeAscendMajorKLayout2D(PrimExpr rows, PrimExpr cols,
                                       int element_bits) {
  PrimExpr c0 = IntImm::Int32(AscendC0(element_bits));
  PrimExpr row_frac = IntImm::Int32(kAscendRowFractal);
  PrimVar i = InputPlaceholder(0); // row
  PrimVar j = InputPlaceholder(1); // col
  PrimExpr ro = FloorDiv(i, row_frac);
  PrimExpr ri = FloorMod(i, row_frac);
  PrimExpr co = FloorDiv(j, c0);
  PrimExpr ci = FloorMod(j, c0);
  return MakeTaggedAscendFractalLayout(Array<PrimExpr>{rows, cols},
                                       {co, ro, ri, ci},
                                       AscendFractalKind::kMajorK);
}

// MajorMN fractal: the contiguous output M/N axis is the rows axis (dim -2)
// instead of the cols axis, so the C0 grouping falls on rows and the 16-fractal
// grouping falls on cols. The forward map is the "swapped" form
// (i, j) -> [i / C0, j / 16, j % 16, i % C0], recognized by
// TryExtractAscendFractalLayout as c0_axis == kRow.
static Layout MakeAscendMajorMNLayout2D(PrimExpr rows, PrimExpr cols,
                                        int element_bits) {
  PrimExpr c0 = IntImm::Int32(AscendC0(element_bits));
  PrimExpr row_frac = IntImm::Int32(kAscendRowFractal);
  PrimVar i = InputPlaceholder(0); // row (contiguous M/N axis)
  PrimVar j = InputPlaceholder(1); // col (reduce-K axis)
  PrimExpr co = FloorDiv(i, c0);
  PrimExpr ci = FloorMod(i, c0);
  PrimExpr ro = FloorDiv(j, row_frac);
  PrimExpr ri = FloorMod(j, row_frac);
  return MakeTaggedAscendFractalLayout(Array<PrimExpr>{rows, cols},
                                       {co, ro, ri, ci},
                                       AscendFractalKind::kMajorMN);
}

static void CheckStaticDivisible(const PrimExpr &value, int divisor,
                                 const std::string &what) {
  if (divisor <= 1) {
    return;
  }
  sym::Analyzer analyzer;
  PrimExpr simplified = tl::arith::Simplify(analyzer, value);
  auto int_value = GetConstInt(simplified);
  ICHECK(int_value.has_value())
      << what << " requires a static extent to verify divisibility by "
      << divisor << ", got " << value;
  ICHECK(*int_value % divisor == 0)
      << what << " requires K-axis extent to be divisible by " << divisor
      << ", got " << *int_value
      << ". For blockscaled GEMM, GM->L1 copy supports auto padding for "
         "tail K values, so allocate the L1/L0 K axis to a multiple of "
      << divisor << ".";
}

Layout MakeAscendMajorKLayout(const TensorVar &buffer, int k_align) {
  ICHECK(buffer.defined() && buffer.shape().size() >= 2)
      << "MakeAscendMajorKLayout expects rank >= 2 buffer";
  size_t n = buffer.shape().size();
  PrimExpr rows = buffer.shape()[n - 2];
  PrimExpr cols = buffer.shape()[n - 1];
  CheckStaticDivisible(cols, k_align,
                       "Ascend MajorK layout for " + buffer->name);
  auto base = MakeAscendMajorKLayout2D(rows, cols, buffer.dtype().bits());
  return ExpandLayout2D(base, buffer);
}

Layout MakeAscendMajorMNLayout(const TensorVar &buffer, int k_align) {
  // For a [.., K, MN] logical tile the contiguous rows axis (dim -2) is the
  // output M/N axis, so C0 groups rows and the 16-fractal groups cols. This is
  // the swapped counterpart of MajorK and yields c0_axis == kRow.
  ICHECK(buffer.defined() && buffer.shape().size() >= 2)
      << "MakeAscendMajorMNLayout expects rank >= 2 buffer";
  size_t n = buffer.shape().size();
  PrimExpr rows = buffer.shape()[n - 2];
  PrimExpr cols = buffer.shape()[n - 1];
  CheckStaticDivisible(rows, k_align,
                       "Ascend MajorMN layout for " + buffer->name);
  auto base = MakeAscendMajorMNLayout2D(rows, cols, buffer.dtype().bits());
  return ExpandLayout2D(base, buffer);
}

Layout MakeAscendNZLayout(const TensorVar &buffer) {
  return MakeAscendMajorKLayout(buffer);
}

Layout MakeAscendCompactNZLayout(const TensorVar &buffer) {
  if (!buffer.defined() || buffer.shape().size() < 2) {
    TVM_FFI_THROW(ValueError)
        << "Compact Ascend NZ layout expects a rank >= 2 buffer";
  }

  int c0 = AscendC0(buffer.dtype().bits());
  const PrimExpr &cols = buffer.shape().back();
  auto cols_value = GetConstInt(cols);
  if (!cols_value.has_value()) {
    TVM_FFI_THROW(ValueError)
        << "Compact Ascend NZ layout requires a static column extent, got "
        << cols;
  }
  if (*cols_value % c0 != 0) {
    TVM_FFI_THROW(ValueError)
        << "Compact Ascend NZ layout requires columns divisible by C0=" << c0
        << ", got " << *cols_value;
  }

  Array<LayoutAxis> iter_vars;
  iter_vars.reserve(buffer.shape().size());
  for (size_t i = 0; i < buffer.shape().size(); ++i) {
    iter_vars.push_back(
        MakeLayoutAxis("i" + std::to_string(i), buffer.shape()[i]));
  }

  PrimExpr stage = IntImm::Int32(0);
  for (size_t i = 0; i + 2 < buffer.shape().size(); ++i) {
    stage = stage * buffer.shape()[i] + iter_vars[i]->var;
  }
  const PrimExpr &row = iter_vars[iter_vars.size() - 2]->var;
  const PrimExpr &col = iter_vars.back()->var;
  PrimExpr c0_expr = IntImm::Int32(c0);
  PrimExpr c0_groups = FloorDiv(cols, c0_expr);
  return Layout(iter_vars, {stage * c0_groups + FloorDiv(col, c0_expr), row,
                            FloorMod(col, c0_expr)});
}

bool IsAscendCompactNZLayout(const Layout &layout, const TensorVar &buffer) {
  if (!layout.defined() || !buffer.defined() || buffer.shape().size() < 2) {
    return false;
  }
  int bits = buffer.dtype().bits();
  if (bits <= 0 || 256 % bits != 0)
    return false;
  auto cols = GetConstInt(buffer.shape().back());
  if (!cols.has_value() || *cols % AscendC0(bits) != 0)
    return false;
  Layout expected = MakeAscendCompactNZLayout(buffer);
  return layout->IsEqual(expected.get());
}

static Layout MakeAscendL0CLayout2D(PrimExpr rows, PrimExpr cols) {
  PrimExpr frac = IntImm::Int32(kAscendRowFractal);
  PrimVar i = InputPlaceholder(0); // M row
  PrimVar j = InputPlaceholder(1); // N col
  PrimExpr mo = FloorDiv(i, frac);
  PrimExpr mi = FloorMod(i, frac);
  PrimExpr no = FloorDiv(j, frac);
  PrimExpr ni = FloorMod(j, frac);
  return MakeTaggedAscendFractalLayout(
      Array<PrimExpr>{rows, cols}, {no, mo, mi, ni}, AscendFractalKind::kL0C);
}

Layout MakeAscendL0CLayout(const TensorVar &buffer) {
  ICHECK(buffer.defined()) << "L0C layout expects a defined buffer";
  ICHECK(buffer.shape().size() >= 2)
      << "L0C layout expects rank >= 2 buffer, got rank="
      << buffer.shape().size();
  size_t ndim = buffer.shape().size();
  PrimExpr rows = buffer.shape()[ndim - 2];
  PrimExpr cols = buffer.shape()[ndim - 1];
  auto base = MakeAscendL0CLayout2D(rows, cols);
  return ExpandLayout2D(base, buffer);
}

static Layout MakeAscendSFLayout2D(PrimExpr mn, PrimExpr k, int element_bits) {
  PrimExpr mn_frac = IntImm::Int32(kAscendRowFractal);
  ICHECK(element_bits > 0 && 16 % element_bits == 0)
      << "SF layout expects element bits to divide 16, got " << element_bits;
  PrimExpr pack = IntImm::Int32(16 / element_bits);
  PrimVar i = InputPlaceholder(0); // M/N axis
  PrimVar j = InputPlaceholder(1); // K axis in scale elements
  PrimExpr mo = FloorDiv(i, mn_frac);
  PrimExpr mi = FloorMod(i, mn_frac);
  PrimExpr ko = FloorDiv(j, pack);
  PrimExpr pi = FloorMod(j, pack);
  return MakeTaggedAscendFractalLayout(Array<PrimExpr>{mn, k}, {mo, ko, mi, pi},
                                       AscendFractalKind::kSF);
}

Layout MakeAscendSFLayout(const TensorVar &buffer) {
  ICHECK(buffer.defined()) << "SF layout expects a defined buffer";
  ICHECK(buffer.shape().size() >= 2)
      << "SF layout expects rank >= 2 buffer with trailing [MN, K], got rank="
      << buffer.shape().size();
  size_t ndim = buffer.shape().size();
  PrimExpr mn = buffer.shape()[ndim - 2];
  PrimExpr k = buffer.shape()[ndim - 1];
  ICHECK(buffer.dtype().bits() > 0 && 16 % buffer.dtype().bits() == 0)
      << "SF layout expects storage dtype whose bit width divides 16, got "
      << buffer.dtype();
  if (buffer.dtype().bits() != 16) {
    sym::Analyzer analyzer;
    PrimExpr k_extent = tl::arith::Simplify(analyzer, k);
    auto k_value = GetConstInt(k_extent);
    ICHECK(k_value.has_value())
        << "SF_K layout for non-uint16 storage requires a static K-axis "
           "extent to verify 2-byte alignment, got "
        << k;
    ICHECK((*k_value * buffer.dtype().bits()) % 16 == 0)
        << "SF_K layout for " << buffer->name << " with dtype "
        << buffer.dtype()
        << " requires the K-axis extent to be 2-byte aligned. Got K="
        << *k_value << ". Pad the K-axis with identity e8m0 scale value 127.";
  }
  auto base = MakeAscendSFLayout2D(mn, k, buffer.dtype().bits());
  return ExpandLayout2D(base, buffer);
}

// ── Ascend fractal layout analysis ──────────────────────────────────

bool TryExtractAscendFractalLayout(const Layout &layout, const TensorVar &buf,
                                   AscendFractalLayoutInfo *out) {
  if (!layout.defined() || !buf.defined() || buf.shape().size() < 2)
    return false;

  auto I = [](int64_t v) { return IntImm::Int32(v); };
  size_t n = buf.shape().size();
  PrimExpr rows = buf.shape()[n - 2];
  PrimExpr cols = buf.shape()[n - 1];
  PrimExpr row_frac = I(16);

  auto ceildiv_expr = [](const PrimExpr &v, const PrimExpr &f) {
    return FloorDiv(v + f - IntImm::Int32(1), f);
  };
  const auto *tagged = layout.as<AscendFractalLayoutNode>();
  ICHECK(tagged != nullptr)
      << "Expected a tagged canonical Ascend fractal layout for buffer "
      << buf->name << ", but got layout type " << layout->GetTypeKey()
      << ". Construct the layout with MakeAscendMajorKLayout, "
         "MakeAscendMajorMNLayout, MakeAscendL0CLayout, or "
         "MakeAscendSFLayout.";

  AscendFractalKind kind = tagged->Kind();
  auto [c0_axis, row16_axis] = GetAxesForKind(kind);
  PrimExpr c0 = 0;
  if (kind == AscendFractalKind::kL0C) {
    c0 = row_frac;
  } else if (kind == AscendFractalKind::kSF) {
    int element_bits = buf.dtype().bits();
    ICHECK(element_bits > 0 && 16 % element_bits == 0)
        << "Tagged Ascend SF layout expects element bits to divide 16, got "
        << element_bits << " for buffer " << buf->name;
    c0 = I(16 / element_bits);
  } else {
    c0 = I(AscendC0(buf.dtype().bits()));
  }
  auto axis_extent = [&](LogicalAxis axis) {
    return axis == LogicalAxis::kRow ? rows : cols;
  };
  PrimExpr c0_outer = ceildiv_expr(axis_extent(c0_axis), c0);
  PrimExpr row16_outer = ceildiv_expr(axis_extent(row16_axis), row_frac);
  PrimExpr outer0 = kind == AscendFractalKind::kSF ? row16_outer : c0_outer;
  PrimExpr outer1 = kind == AscendFractalKind::kSF ? c0_outer : row16_outer;
  *out = {kind, c0, row_frac, rows, cols, outer0, outer1, c0_axis, row16_axis};
  return true;
}

bool TryExtractAscendFractalRegion(const Layout &layout,
                                   const Array<Range> &logical_region,
                                   AscendFractalRegionInfo *out) {
  if (!layout.defined() || logical_region.size() != layout->InputDim())
    return false;
  Array<Range> phys = layout->MapRegionBounds(logical_region);
  if (phys.size() < 4)
    return false;
  size_t n = phys.size();
  *out = {phys[n - 4], phys[n - 3], phys[n - 2], phys[n - 1]};
  return true;
}

// Ascend ND (row-major) identity layout for GM/UB operands. Forward is the
// identity [i0, .., i_{n-1}] -> [i0, .., i_{n-1}], so OutputShape ==
// InputShape. Used only for copy-parameter derivation in lowering;
// intentionally NOT anchored into layout_map/buffer_remap (an identity remap is
// a no-op, and a flattening layout would alias GM/UB buffers to 1D).
Layout makeAscendNDLayout(const TensorVar &buffer) {
  int ndim = static_cast<int>(buffer.shape().size());
  Array<LayoutAxis> iter_vars;
  Array<PrimExpr> forward_index;
  for (int i = 0; i < ndim; ++i) {
    auto iv = MakeLayoutAxis("i" + std::to_string(i), buffer.shape()[i]);
    iter_vars.push_back(iv);
    forward_index.push_back(iv->var);
  }
  return Layout(iter_vars, forward_index);
}

// ── MakeStridedSlice ──────────────────────────────────────────────────
// Given a physical (remapped) buffer and a layout, compute a 0-origin
// strided buffer view for the given logical sub-region.
//
// Steps:
// 1. MapRegionBounds(logical_region) → physical region (one Range per output
// dim)
// 2. Compute row-major strides from phys_buf shape (or use existing strides)
// 3. offset = sum(phys_region[i].min * strides[i])
// 4. shape  = [phys_region[i].extent for each dim]
// 5. Return TensorVar(same data, shape, strides, offset)
Call MakeStridedSlice(const TensorVar &tensor, const Layout &layout,
                      const Array<Range> &logical_region) {
  Array<Range> physical = layout->MapRegionBounds(logical_region);
  TVM_FFI_CHECK(physical.size() == tensor.shape().size(), ValueError)
      << "Slice rank must match the physical tensor rank";
  auto tile =
      tensor.ty()
          ->layout.value_or(tirx::TileLayout::DefaultLayout(tensor.shape()))
          .as<tirx::TileLayout>();
  TVM_FFI_CHECK(tile && tile.value()->replica.empty() &&
                    tile.value()->shard.size() == physical.size(),
                ValueError)
      << "Strided slicing requires an unreplicated physical tile layout";
  Array<PrimExpr> shape;
  Array<tirx::Iter> dimensions;
  auto origins = tile.value()->offset;
  for (size_t axis = 0; axis < physical.size(); ++axis) {
    const auto &dimension = tile.value()->shard[axis];
    TVM_FFI_CHECK(dimension->axis->IsMemoryAxis(), ValueError)
        << "A physical slice cannot contain thread axes";
    shape.push_back(physical[axis]->extent);
    dimensions.push_back(
        tirx::Iter(physical[axis]->extent, dimension->stride, dimension->axis));
    PrimExpr origin = origins.Get(dimension->axis).value_or(PrimExpr(0));
    origins.Set(dimension->axis,
                origin + physical[axis]->begin * dimension->stride);
  }
  TensorType type(shape, tensor.dtype(), tensor.scope(), 0,
                  tirx::TileLayout(dimensions, {}, origins));
  return Call(type, decl_tensor_op(),
              {Call(std::nullopt, tensor_data_ptr_op(), {tensor})},
              std::nullopt, {type});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  AscendFractalLayoutNode::RegisterReflection();
  refl::GlobalDef()
      .def("tl.AscendFractalLayoutFromIndexMap",
           [](Array<PrimExpr> shape, IndexMap index, int kind) {
             TVM_FFI_CHECK(
                 kind >= static_cast<int>(AscendFractalKind::kMajorK) &&
                     kind <= static_cast<int>(AscendFractalKind::kMajorMN),
                 ValueError)
                 << "Unknown Ascend fractal layout kind";
             return Layout(make_object<AscendFractalLayoutNode>(
                 shape, MapLayoutCoordinates(index, shape.size(), false),
                 static_cast<AscendFractalKind>(kind)));
           })
      .def("tl.make_ascend_major_k_layout",
           [](const TensorVar &buffer, int k_align) {
             return MakeAscendMajorKLayout(buffer, k_align);
           })
      .def("tl.make_ascend_major_mn_layout",
           [](const TensorVar &buffer, int k_align) {
             return MakeAscendMajorMNLayout(buffer, k_align);
           })
      .def("tl.make_ascend_l0c_layout",
           [](const TensorVar &buffer) { return MakeAscendL0CLayout(buffer); })
      .def("tl.make_ascend_sf_layout",
           [](const TensorVar &buffer) { return MakeAscendSFLayout(buffer); })
      .def("tl.make_ascend_nz_layout",
           [](const TensorVar &buffer) { return MakeAscendNZLayout(buffer); })
      .def("tl.make_ascend_compact_nz_layout",
           [](const TensorVar &buffer) {
             return MakeAscendCompactNZLayout(buffer);
           })
      .def("tl.try_extract_fractal_layout",
           [](Layout layout,
              const TensorVar &buf) -> Optional<Map<String, Any>> {
             AscendFractalLayoutInfo info;
             if (!TryExtractAscendFractalLayout(layout, buf, &info))
               return std::nullopt;
             int c0_axis_int = static_cast<int>(info.c0_axis);
             int row16_axis_int = static_cast<int>(info.row16_axis);
             int kind_int = static_cast<int>(info.kind);
             return Map<String, Any>{
                 {"kind", IntImm::Int32(kind_int)},
                 {"c0", info.c0},
                 {"row_frac", info.row_frac},
                 {"rows", info.rows},
                 {"cols", info.cols},
                 {"outer0", info.outer0},
                 {"outer1", info.outer1},
                 {"c0_axis", IntImm::Int32(c0_axis_int)},
                 {"row16_axis", IntImm::Int32(row16_axis_int)},
             };
           })
      .def("tl.make_strided_slice", [](const TensorVar &phys_buf, Layout layout,
                                       Array<Range> logical_region) {
        return MakeStridedSlice(phys_buf, layout, logical_region);
      });
}

} // namespace tl
} // namespace tvm
