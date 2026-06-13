/*!
 * \file cute_layout.h
 * \brief CuTe-style layout IR types for TileLang.
 */

#ifndef TVM_TL_LAYOUT_CUTE_LAYOUT_H_
#define TVM_TL_LAYOUT_CUTE_LAYOUT_H_

#include "support/check.h"
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/base.h>
#include <tvm/tirx/buffer.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

// Forward declaration: the TileLang layout (defined in layout.h).
class Layout;

namespace cute {

using namespace ffi;
using namespace tirx;

// A generic CUTLASS/CuTe-style XOR swizzle, described by three bit-field
// parameters (b_bits, m_base, s_shift):
//
// 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
//                               ^--^ m_base  (least-significant bits kept
//                               fixed)
//                  ^-^       ^-^     b_bits   (number of mask bits)
//                    ^---------^     s_shift  (distance YYY is shifted onto
//                    ZZZ)
//
// apply(x) = x ^ ((x & yyy_msk) >> s_shift)
class SwizzleNode : public Object {
public:
  int b_bits{0};
  int m_base{0};
  int s_shift{0};

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.cute.Swizzle", SwizzleNode, Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SwizzleNode>()
        .def_ro("b_bits", &SwizzleNode::b_bits)
        .def_ro("m_base", &SwizzleNode::m_base)
        .def_ro("s_shift", &SwizzleNode::s_shift);
  }

  // Whether the layout actually applies an XOR swizzle. An identity layout has
  // b_bits == 0 and is not swizzled.
  bool IsSwizzled() const { return b_bits > 0; }

  // Apply the swizzle to a physical offset: ZZZ ^= YYY.
  int64_t Apply(int64_t offset) const;
};

class Swizzle : public ObjectRef {
public:
  TVM_DLL Swizzle(int b_bits, int m_base, int s_shift);

  // A non-swizzle.
  static Swizzle Identity() { return Swizzle(0, 0, 0); }

  // Reinterpret the swizzle when the underlying buffer is viewed as a different
  // element type.
  TVM_DLL Swizzle Recast(int old_bits, int new_bits) const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Swizzle, ObjectRef, SwizzleNode);
};

// A hierarchical CuTe IntTuple.
// A tuple can have multiple children.
// A leaf can encode a constant (int64_t), a dynamic value (PrimExpr), or a
// scaled basis (for describing hierarchical strides, 1@0 etc.).
class IntTupleNode : public Object {
public:
  static constexpr uint32_t _type_child_slots = 4;
  TVM_FFI_DECLARE_OBJECT_INFO("tl.cute.IntTuple", IntTupleNode, Object);
};

class IntTuple : public ObjectRef {
public:
  // The i-th child. A leaf has rank 1 and only index 0 is allowed.
  TVM_DLL IntTuple operator[](int64_t index) const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntTuple, ObjectRef, IntTupleNode);
};

// A constant integer.
class IntTupleConstNode : public IntTupleNode {
public:
  int64_t value{0};

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.cute.IntTupleConst", IntTupleConstNode,
                                    IntTupleNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IntTupleConstNode>().def_ro("value",
                                                &IntTupleConstNode::value);
  }
};

class IntTupleConst : public IntTuple {
public:
  TVM_DLL explicit IntTupleConst(int64_t value);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntTupleConst, IntTuple,
                                             IntTupleConstNode);
};

// A dynamic (runtime-valued) integer, carrying a PrimExpr.
class IntTuplePrimExprNode : public IntTupleNode {
public:
  PrimExpr value;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.cute.IntTuplePrimExpr",
                                    IntTuplePrimExprNode, IntTupleNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IntTuplePrimExprNode>().def_ro(
        "value", &IntTuplePrimExprNode::value);
  }
};

class IntTuplePrimExpr : public IntTuple {
public:
  TVM_DLL explicit IntTuplePrimExpr(PrimExpr value);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntTuplePrimExpr, IntTuple,
                                             IntTuplePrimExprNode);
};

// A CuTe ScaledBasis: value * E<basis...>.
class IntTupleScaledBasisNode : public IntTupleNode {
public:
  IntTuple value;
  Array<int64_t> basis;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.cute.IntTupleScaledBasis",
                                    IntTupleScaledBasisNode, IntTupleNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IntTupleScaledBasisNode>()
        .def_ro("value", &IntTupleScaledBasisNode::value)
        .def_ro("basis", &IntTupleScaledBasisNode::basis);
  }
};

class IntTupleScaledBasis : public IntTuple {
public:
  // Pre: `value` is a scalar leaf (IsConst or IsPrimExpr), not a tuple.
  TVM_DLL IntTupleScaledBasis(IntTuple value, Array<int64_t> basis);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntTupleScaledBasis, IntTuple,
                                             IntTupleScaledBasisNode);
};

// A tuple.
class IntTupleTupleNode : public IntTupleNode {
public:
  Array<IntTuple> fields;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.cute.IntTupleTuple", IntTupleTupleNode,
                                    IntTupleNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IntTupleTupleNode>().def_ro("fields",
                                                &IntTupleTupleNode::fields);
  }
};

class IntTupleTuple : public IntTuple {
public:
  TVM_DLL explicit IntTupleTuple(Array<IntTuple> fields);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntTupleTuple, IntTuple,
                                             IntTupleTupleNode);
};

inline bool IsConst(const IntTuple &t) {
  return t.as<IntTupleConstNode>() != nullptr;
}
inline bool IsPrimExpr(const IntTuple &t) {
  return t.as<IntTuplePrimExprNode>() != nullptr;
}
inline bool IsScaledBasis(const IntTuple &t) {
  return t.as<IntTupleScaledBasisNode>() != nullptr;
}
inline bool IsTuple(const IntTuple &t) {
  return t.as<IntTupleTupleNode>() != nullptr;
}

inline int64_t AsConst(const IntTuple &t) {
  const auto *c = t.as<IntTupleConstNode>();
  ICHECK(c != nullptr) << "AsConst on a non-constant leaf";
  return c->value;
}
inline PrimExpr AsPrimExpr(const IntTuple &t) {
  const auto *e = t.as<IntTuplePrimExprNode>();
  ICHECK(e != nullptr) << "AsPrimExpr on a non-PrimExpr leaf";
  return e->value;
}
inline PrimExpr AsConstOrPrimExpr(const IntTuple &t, const DataType &dtype) {
  if (const auto *c = t.as<IntTupleConstNode>())
    return IntImm(dtype, c->value);
  const auto *e = t.as<IntTuplePrimExprNode>();
  ICHECK(e != nullptr) << "AsConstOrPrimExpr on a non-leaf";
  return e->value;
}
inline IntTuple BasisValue(const IntTuple &t) {
  if (const auto *b = t.as<IntTupleScaledBasisNode>())
    return b->value;
  return t;
}
inline Array<int64_t> BasisPath(const IntTuple &t) {
  if (const auto *b = t.as<IntTupleScaledBasisNode>())
    return b->basis;
  return Array<int64_t>();
}

// Select a child by a basis (ScaledBasis indexing).
TVM_DLL IntTuple BasisGet(const Array<int64_t> &path, const IntTuple &t);

// The rank, or number of children, of an IntTuple. For a leaf, the rank is 1.
TVM_DLL int64_t Rank(const IntTuple &t);

/// Product of all leaf values.
/// @pre every leaf IsConst.
TVM_DLL int64_t Product(const IntTuple &t);

/// Flatten a tree into a IntTupleTuple of leaves.
/// A leaf flattens to itself.
/// @post depth <= 1.
TVM_DLL IntTuple Flatten(const IntTuple &t);

// Higher-order function mapping each leaf by `f`.
template <class F> IntTuple TransformLeaf(const IntTuple &t, F &&f) {
  if (!IsTuple(t))
    return f(t);
  const auto *a = t.as<IntTupleTupleNode>();
  Array<IntTuple> out;
  out.reserve(a->fields.size());
  for (const auto &child : a->fields)
    out.push_back(TransformLeaf(child, f));
  return IntTupleTuple(out);
}

// Higher-order function reducing all the leaves by `f`.
template <class Acc, class F>
Acc FoldLeaves(const IntTuple &t, Acc init, F &&f) {
  if (!IsTuple(t))
    return f(std::move(init), t);
  const auto *a = t.as<IntTupleTupleNode>();
  for (const auto &child : a->fields)
    init = FoldLeaves(child, std::move(init), f);
  return init;
}

// A hierarchical CuTe layout.
// `shape` and `stride` are congruent IntTuples (same tree structure).
// NOTE: aligned with CuTe, this layout is COLUMN-MAJOR.
// Evaluating maps a single linear coordinate to a hierarchical coordinate, then
// dotted with the strides.
class LayoutNode : public Object {
public:
  IntTuple shape;
  IntTuple stride;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.cute.Layout", LayoutNode, Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LayoutNode>()
        .def_ro("shape", &LayoutNode::shape)
        .def_ro("stride", &LayoutNode::stride);
  }

  /// Map a single linear coordinate to a physical index.
  /// @pre the shape and stride must be constant.
  int64_t operator()(int64_t coord) const;

  // Symbolic evaluation: like operator()(int64_t) but the coordinate (and
  // result) are dynamic integers (PrimExprs).
  PrimExpr operator()(const PrimExpr &coord) const;
};

class Layout : public ObjectRef {
public:
  // Convenience constructor: static shape, static stride.
  TVM_DLL Layout(Array<int64_t> shape, Array<int64_t> stride);
  // Convenience constructor: static shape, dynamic stride.
  TVM_DLL Layout(Array<int64_t> shape, Array<IntTuple> stride);
  // Convenience constructor: dynamic shape, dynamic stride.
  TVM_DLL Layout(Array<IntTuple> shape, Array<IntTuple> stride);
  // Canonical constructor.
  TVM_DLL Layout(IntTuple shape, IntTuple stride);

  // Sub-layout.
  TVM_DLL Layout operator[](int64_t index) const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Layout, ObjectRef, LayoutNode);
};

/// Same with IntTuple.
/// @return the rank of the shape (and stride).
TVM_DLL int64_t Rank(const Layout &layout);
/// Same with IntTuple.
/// @return the flattened shape and stride.
TVM_DLL Layout Flatten(const Layout &layout);
/// The domain size of the shape.
/// @return the product of the shape.
/// @pre the shape must be static.
TVM_DLL int64_t Size(const Layout &layout);

/// Flatten, and merge adjacent modes that can be composed into one.
/// A fully collapsed layout is returned as CuTe's scalar mode (1):(0)
/// @return result(i) == layout(i) for all i < size(result).
/// @post depth <= 1, no need to flatten again.
TVM_DLL Layout Coalesce(const Layout &layout);

/// Right-inverse of a layout.
/// @return layout(result(i)) == i for all i < size(result).
TVM_DLL Layout RightInverse(const Layout &layout);

/// Composition lhs o rhs.
/// @return result(c) == lhs(rhs(c)) for all c in the domain of rhs.
/// @post size(result) == size(rhs).
/// @note a ScaledBasis rhs stride (value * E<path>) reroutes into the matching
/// lhs axis.
TVM_DLL Layout Composition(const Layout &lhs, const Layout &rhs);

/// Same with Coalesce, but caps each merged run at 256 (the TMA box per-mode
/// limit), so a large contiguous run stays split instead of overflowing.
/// This is tailored for TMA descriptor's box_dims field.
/// @post depth <= 1.
TVM_DLL Layout Coalesce256(const Layout &layout);

// Column major layout over `shape`.
TVM_DLL Layout MakeColumnMajorLayout(const std::vector<int64_t> &shape);

// Row major layout over `shape`.
TVM_DLL Layout MakeRowMajorLayout(const std::vector<int64_t> &shape);

// Identity layout over `shape`: per-axis unit ScaledBases E<k>.
TVM_DLL Layout MakeIdentityLayout(const std::vector<int64_t> &shape);

// A CuTe ComposedLayout of the form Swizzle o offset o Layout.
// Evaluating at x yields swizzle.apply(offset + layout(x)).
class ComposedLayoutNode : public Object {
public:
  Swizzle swizzle;
  int64_t offset{0};
  Layout layout;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.cute.ComposedLayout",
                                    ComposedLayoutNode, Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ComposedLayoutNode>()
        .def_ro("swizzle", &ComposedLayoutNode::swizzle)
        .def_ro("offset", &ComposedLayoutNode::offset)
        .def_ro("layout", &ComposedLayoutNode::layout);
  }

  /// Map a single linear coordinate: Sw(offset + layout(coord)).
  /// @pre: the layout is constant.
  int64_t operator()(int64_t coord) const {
    return swizzle->Apply(offset + layout->operator()(coord));
  }
};

class ComposedLayout : public ObjectRef {
public:
  TVM_DLL ComposedLayout(Swizzle swizzle, int64_t offset, Layout layout);

  // Recast into a different element width: the swizzle's m_base shifts (see
  // Swizzle::Recast) and the strides/offset scale by old_bits/new_bits.
  TVM_DLL ComposedLayout Recast(int old_bits, int new_bits) const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ComposedLayout, ObjectRef,
                                             ComposedLayoutNode);
};

/// Recover an affine TileLang layout as a cute::Layout.
/// @return result(coords) == layout(coords)
/// @return nullopt if the address is not purely affine (e.g. it has a swizzle,
///         a non-zero base offset, or a non-constant extent).
Optional<Layout> LayoutFromTileLang(const tvm::tl::Layout &layout);

/// Recover a possibly swizzled and offset TileLang layout as a
/// cute::ComposedLayout.
/// @return result(coords) == layout(coords)
/// @return nullopt if the recovery fails, e.g. the TileLang layout contains
///         dynamic values, or the analyzer is unable to prove the equivalence
///         between a cute::ComposedLayout and the TileLang layout.
/// @note Addresses are element-offset positions; recast with
///       `.Recast(dtype.bits(), 8)` for the byte-address uses.
Optional<ComposedLayout>
ComposedLayoutFromTileLang(const tvm::tl::Layout &layout);

// The faithful-CuTe decomposition of a bulk copy, porting CuTe's
// construct_tma_gbasis (copy_traits_sm90_tma.hpp). All members are
// cute::Layouts with `scale @ axis` ScaledBasis strides; consumers recover
// concrete geometry by Composition (the basis routes each mode into a global
// axis, i.e. CuTe's gtensor.compose(sidx2gmode) / basis_get) rather than by
// parsing modes.
//
//  - box: the single TMA descriptor box (CuTe's tma_gbasis). shape = boxDim
//    (IntTupleConst, each <= 256, mode 0 unit-scale); stride = `1 @ axis` per
//    mode (SMEM-vector order, 1:1 with distinct global axes).
//    Composition(gmem, box) -> per-mode global strides; Composition(gextent,
//    box) -> globalDim.
//  - rest_gmem: the iteration space CuTe truncates away (modes past smem_rank,
//    plus any >256 box overflow), strides `scale @ axis`, e.g. (8):(64@1). One
//    TMA instruction per coordinate; Composition(unit_axis, rest_gmem) gives
//    that axis's gmem-coord step.
//  - rest_smem: congruent to rest_gmem with plain SMEM element strides, e.g.
//    (8):(4096); rest_smem(i) is the SMEM offset of the i-th instruction.
//
// NOTE: with no rest, rest_gmem/rest_smem are the empty layout () -> one
// instruction. This never permutes modes and never folds a non-contiguous
// (scale != 1) global mode into the box, so per-box out-of-bounds is exact.
struct TmaTile {
  Layout box;
  Layout rest_gmem;
  Layout rest_smem;
};

/// Derive the faithful-CuTe TMA decomposition, porting construct_tma_gbasis:
///   inv  = RightInverse(smem_plain)                       // SMEM idx -> coord
///   full = Coalesce(Composition(identity(tile_shape), inv))  //
///   sidx2gmode_full smem_rank = first mode with scale != 1; box =
///   take<0,smem_rank>(full) box dims sized via Coalesce256(Composition(gmem,
///   box))   // coalesce_256
/// The leading scale-1 run is the box; the rest (and any >256 box overflow)
/// goes to rest_gmem/rest_smem. `gmem` carries the global extents (shape) and
/// element strides (stride), which may be dynamic (IntTuplePrimExpr).
/// @pre smem_plain is a bijection onto [0, size) (ICHECKed).
/// @return nullopt when not TMA-expressible (innermost mode not unit-scale, a
/// mode with no axis, an unsplittable >256 mode, or box rank > 5).
std::optional<TmaTile> DeriveTmaTile(const Layout &gmem,
                                     const Layout &smem_plain,
                                     const std::vector<int64_t> &tile_shape);

} // namespace cute
} // namespace tl
} // namespace tvm

#endif // TVM_TL_LAYOUT_CUTE_LAYOUT_H_
