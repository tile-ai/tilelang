/*!
 * \file cute_layout.h
 * \brief CuTe-style layout IR types for TileLang.
 */

#ifndef TVM_TL_LAYOUT_CUTE_LAYOUT_H_
#define TVM_TL_LAYOUT_CUTE_LAYOUT_H_

#include "axis.h"
#include "compat/expr.h"
#include "support/base.h"
#include "support/check.h"
#include "swizzle_mode.h"
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/expr.h>

#include <cstdint>
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

  /// Whether this swizzle can be lowered to TMA instructions.
  /// TMA only supports unswizzled, 32B, 64B, and 128B swizzling.
  /// @pre this should be recast to byte addresses.
  bool IsTMACompatible() const {
    if (!IsSwizzled()) {
      return true;
    }
    // 128B: S<3, 4, 3>
    // 64B:  S<2, 4, 3>
    // 32B:  S<1, 4, 3>
    return m_base == 4 && s_shift == 3 && b_bits >= 1 && b_bits <= 3;
  }

  // The span of one swizzle period: 2^(m_base + b_bits). For the TMA atoms
  // <b,4,3> this is the CUtensorMap swizzle size (32/64/128 B for b = 1/2/3).
  int64_t Granularity() const { return int64_t(1) << (m_base + b_bits); }

  /// The canonical swizzle mode supported by the hardware.
  /// @pre this should be in byte-address space, so M=4 and S=3 if there is
  /// swizzling present.
  SwizzleMode ToSwizzleMode() const {
    if (b_bits == 0) {
      return SwizzleMode::None();
    }
    ICHECK(m_base == 4 && s_shift == 3 && b_bits >= 1 && b_bits <= 3)
        << "Not a canonical GMMA swizzle atom Sw<" << b_bits << "," << m_base
        << "," << s_shift << ">; expected Sw<{1,2,3},4,3>";
    // b_bits 1->32B(ord 1), 2->64B(ord 2), 3->128B(ord 3).
    return SwizzleMode::FromOrdinal(b_bits);
  }

  // Apply the swizzle to a physical offset: ZZZ ^= YYY.
  int64_t Apply(int64_t offset) const;
};

class Swizzle : public ObjectRef {
public:
  TILELANG_DLL Swizzle(int b_bits, int m_base, int s_shift);

  // A non-swizzle.
  static Swizzle Identity() { return Swizzle(0, 0, 0); }

  // Reinterpret the swizzle when the underlying buffer is viewed as a different
  // element type.
  TILELANG_DLL Swizzle Recast(int old_bits, int new_bits) const;

  // Parse the CuTe spelling "Sw<b,m,s>" (the inverse of Print).
  TILELANG_DLL static Swizzle Parse(const ffi::String &text);

  // Print in the CuTe spelling "Sw<b,m,s>".
  TILELANG_DLL void Print(std::ostream &os) const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Swizzle, ObjectRef, SwizzleNode);
};

// A hierarchical CuTe IntTuple.
// A tuple can have multiple children.
// A leaf can encode a constant (int64_t), a dynamic value (PrimExpr), or a
// scaled basis (for describing hierarchical strides, 1@0 etc.).
class IntTupleNode : public Object {
public:
  static constexpr uint32_t _type_child_slots = 4;
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("tl.cute.IntTuple", IntTupleNode, Object);
};

class IntTuple : public ObjectRef {
public:
  // Creates a static integer.
  TILELANG_DLL IntTuple(int64_t value);

  // Creates a either static or dynamic integer, based on the PrimExpr's value.
  TILELANG_DLL IntTuple(PrimExpr value);

  // The i-th child. A leaf has rank 1 and only index 0 is allowed.
  TILELANG_DLL IntTuple operator[](int64_t index) const;

  // Element-wise sum.
  TILELANG_DLL friend IntTuple operator+(const IntTuple &a, const IntTuple &b);
  IntTuple &operator+=(const IntTuple &other) {
    *this = *this + other;
    return *this;
  }

  // The product. The product of two tuples is dot-product.
  TILELANG_DLL friend IntTuple operator*(const IntTuple &a, const IntTuple &b);
  IntTuple &operator*=(const IntTuple &other) {
    *this = *this * other;
    return *this;
  }

  // Parse the CuTe spelling (the inverse of Print): nested "(a,b,...)"
  // tuples, integer leaves, and scaled bases "v@m@..." with the mode path
  // innermost-first (E<1,0> is spelled 1@0@1).
  TILELANG_DLL static IntTuple Parse(const ffi::String &text);

  // Print in the CuTe spelling.
  TILELANG_DLL void Print(std::ostream &os) const;

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
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntTupleConst, IntTuple,
                                             IntTupleConstNode);
};

// A dynamic (runtime-valued) integer, carrying a PrimExpr.
class IntTuplePrimExprNode : public IntTupleNode {
public:
  explicit IntTuplePrimExprNode(PrimExpr expression)
      : value(std::move(expression)) {}
  explicit IntTuplePrimExprNode(ffi::UnsafeInit tag) : value(tag) {}
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
  TILELANG_DLL IntTupleScaledBasis(IntTuple value, Array<int64_t> basis);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntTupleScaledBasis, IntTuple,
                                             IntTupleScaledBasisNode);
};

inline IntTupleScaledBasis E(Array<int64_t> basis) {
  return IntTupleScaledBasis(IntTuple(1), std::move(basis));
}

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
  TILELANG_DLL explicit IntTupleTuple(Array<IntTuple> fields);
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
inline PrimExpr AsConstOrPrimExpr(const IntTuple &t, const PrimType &dtype) {
  if (const auto *c = t.as<IntTupleConstNode>())
    return IntImm(dtype, c->value);
  const auto *e = t.as<IntTuplePrimExprNode>();
  ICHECK(e != nullptr) << "AsConstOrPrimExpr on a non-leaf: " << t;
  return e->value;
}
inline IntTupleScaledBasis AsScaledBasis(const IntTuple &t) {
  const auto *b = t.as<IntTupleScaledBasisNode>();
  ICHECK(b != nullptr) << "AsScaledBasis on a non-ScaledBasis leaf";
  return GetRef<IntTupleScaledBasis>(b);
}
inline IntTuple BasisValue(const IntTuple &t) {
  return AsScaledBasis(t)->value;
}
inline Array<int64_t> BasisPath(const IntTuple &t) {
  return AsScaledBasis(t)->basis;
}
inline IntTupleTuple AsTuple(const IntTuple &t) {
  const auto *a = t.as<IntTupleTupleNode>();
  ICHECK(a != nullptr) << "AsTuple on a leaf";
  return GetRef<IntTupleTuple>(a);
}
inline Array<IntTuple> TupleFields(const IntTuple &t) {
  return AsTuple(t)->fields;
}

// Select a child by a basis (ScaledBasis indexing).
TILELANG_DLL IntTuple BasisGet(const Array<int64_t> &path, const IntTuple &t);

// The rank, or number of children, of an IntTuple. For a leaf, the rank is 1.
TILELANG_DLL int64_t Rank(const IntTuple &t);

/// Product of all leaf values.
TILELANG_DLL IntTuple Product(const IntTuple &t);

/// Flatten a tree into a IntTupleTuple of leaves.
/// A leaf flattens to itself.
/// @post depth <= 1.
TILELANG_DLL IntTuple Flatten(const IntTuple &t);

/// Wrap an IntTuple into a tuple.
/// @post IsTuple(result).
TILELANG_DLL IntTupleTuple Wrap(const IntTuple &t);

// Checks if two IntTuple's are congruent: whether their tree structure
// matches -- same nesting and same rank at every branch.
TILELANG_DLL bool Congruent(const IntTuple &a, const IntTuple &b);

/// CuTe shape compatibility (`a <= b`): `a` and `b` have the same total size,
/// and every coordinate into `a` is also a valid coordinate into `b`.
TILELANG_DLL bool Compatible(const IntTuple &a, const IntTuple &b);

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
// NOTE: Aligned with CuTe, this layout is COLUMN-MAJOR.
//       That is, the first mode is fastest-changing. Pay attention to this when
//       you are calling operator().
class LayoutNode : public Object {
public:
  IntTuple shape;
  IntTuple stride;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.cute.Layout", LayoutNode, Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LayoutNode>()
        .def_ro("shape", &LayoutNode::shape)
        .def_ro("stride", &LayoutNode::stride);
  }
};

class Layout : public ObjectRef {
public:
  // Convenience constructor: static shape, static stride.
  TILELANG_DLL Layout(Array<int64_t> shape, Array<int64_t> stride);
  // Convenience constructor: static shape, dynamic stride.
  TILELANG_DLL Layout(Array<int64_t> shape, Array<IntTuple> stride);
  // Convenience constructor: dynamic shape, dynamic stride.
  TILELANG_DLL Layout(Array<IntTuple> shape, Array<IntTuple> stride);
  // Canonical constructor.
  TILELANG_DLL Layout(IntTuple shape, IntTuple stride);

  // Sub-layout.
  TILELANG_DLL Layout operator[](int64_t index) const;

  /// Map a single coordinate to a physical index.
  /// @return a const or an expression; when stride has scaled basis, the result
  ///         is a tuple.
  TILELANG_DLL IntTuple operator()(const IntTuple &coord) const;

  // Compose with a column-major layout of this shape.
  TILELANG_DLL Layout WithShape(const IntTuple &shape) const;

  // Parse the CuTe spelling "shape:stride" (the inverse of Print).
  TILELANG_DLL static Layout Parse(const ffi::String &text);

  // Print in the CuTe spelling "shape:stride".
  TILELANG_DLL void Print(std::ostream &os) const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Layout, ObjectRef, LayoutNode);
};

/// Same with IntTuple.
/// @return the rank of the shape (and stride).
TILELANG_DLL int64_t Rank(const Layout &layout);

/// Same with IntTuple.
/// @return the flattened shape and stride.
TILELANG_DLL Layout Flatten(const Layout &layout);

/// The domain size of the shape.
/// @return Product(layout->shape).
TILELANG_DLL IntTuple Size(const Layout &layout);

/// Flatten, and merge adjacent modes that can be composed into one.
/// A fully collapsed layout is returned as CuTe's scalar mode (1):(0)
/// @return result(i) == layout(i) for all i < size(result).
/// @post depth <= 1, no need to flatten again.
TILELANG_DLL Layout Coalesce(const Layout &layout);

/// Right-inverse of a layout.
/// @return layout(result(i)) == i for all i < size(result).
TILELANG_DLL Layout RightInverse(const Layout &layout);

/// Left-inverse of an injective layout (CuTe left_inverse):
/// @return result(layout(i)) == i for all i < size(layout).
/// Like CuTe, ICHECKs when the sorted strides do not form the divisibility
/// chain the inverse requires (an injective layout with irregular gaps).
TILELANG_DLL Layout LeftInverse(const Layout &layout);

/// Composition lhs o rhs.
/// @return result(c) == lhs(rhs(c)) for all c in the domain of rhs.
/// @post size(result) == size(rhs).
/// @note a ScaledBasis rhs stride (value * E<path>) reroutes into the matching
/// lhs axis.
TILELANG_DLL Layout Composition(const Layout &lhs, const Layout &rhs);

/// Same with Coalesce, but caps each merged run at `max_extent`. A run wider
/// than the cap stays split instead of fusing into one oversized mode.
/// @post depth <= 1.
TILELANG_DLL Layout Coalesce(const Layout &layout, int64_t max_extent);

/// Take a given number of modes from a layout.
TILELANG_DLL Layout Take(const Layout &layout, int64_t n);

/// Drop stride-0 and size-1 modes, then coalesce.
TILELANG_DLL Layout Filter(const Layout &layout);

/// Codomain shape of a layout (CuTe coshape): the per-axis extents
/// 1 + inner_product(shape - 1, |stride|), e.g.
/// coshape((128,32):(1@0,1@1)) == (128,32).
TILELANG_DLL IntTuple Coshape(const Layout &layout);

/// Codomain size of a layout (CuTe cosize): size(coshape(layout)).
/// @post for all `0 <= i < size(layout)`, layout(i) < result.
TILELANG_DLL IntTuple Cosize(const Layout &layout);

/// Complement of a layout.
TILELANG_DLL Layout Complement(const Layout &layout, int64_t cotarget);

/// Complement against the layout's own cosize.
TILELANG_DLL Layout Complement(const Layout &layout);

/// Logical divide of `layout` by `tiler`.
/// Splits each tiled mode into (tile, rest).
/// @pre `tiler` and `Size(layout)` must be static.
TILELANG_DLL Layout LogicalDivide(const Layout &layout, const Layout &tiler);

/// Logical product of a block layout and a tiler layout.
/// Repeats `block` according to `tiler`, returning `(block, rest)`.
/// @pre `Size(block)` and `Cosize(tiler)` must be static.
TILELANG_DLL Layout LogicalProduct(const Layout &block, const Layout &tiler);

/// Tiled product convenience form: keep `block` as the first mode and unpack
/// the actual repeated mode of the logical product after it.
TILELANG_DLL Layout TiledProduct(const Layout &block, const Layout &tiler);

/// Blocked product: repeat `block` over `tiler` and zip the per-mode pairs, so
/// mode i of the result is ((block_i, rest_i)) and accepts a flat coordinate
/// covering block extent x tiler extent along that mode.
/// @post Rank(result) == max(Rank(block), Rank(tiler)).
TILELANG_DLL Layout BlockedProduct(const Layout &block, const Layout &tiler);

// Column major layout over `shape`.  Size-1 modes take stride 0 and leave
// the running product untouched (CuTe compact_col_major, stride.hpp:302).
template <typename Container>
inline Layout MakeColumnMajorLayout(const Container &shape) {
  int64_t n = shape.size();
  Array<IntTuple> sh, st;
  sh.reserve(n);
  st.reserve(n);
  IntTuple stride = 1;
  for (int64_t k = 0; k < n; ++k) {
    sh.push_back(shape[k]);
    if (IsConst(IntTuple(shape[k])) && AsConst(IntTuple(shape[k])) == 1) {
      st.push_back(IntTuple(int64_t(0)));
      continue;
    }
    st.push_back(stride);
    stride *= shape[k];
  }
  return Layout(std::move(sh), std::move(st));
}

// Row major layout over `shape`.  Size-1 modes take stride 0 (CuTe
// compact_row_major).
template <typename Container>
inline Layout MakeRowMajorLayout(const Container &shape) {
  int64_t n = shape.size();
  Array<IntTuple> sh(n, IntTuple()), st(n, IntTuple());
  IntTuple stride = 1;
  for (int64_t k = n - 1; k >= 0; --k) {
    sh.Set(k, shape[k]);
    if (IsConst(IntTuple(shape[k])) && AsConst(IntTuple(shape[k])) == 1) {
      st.Set(k, IntTuple(int64_t(0)));
      continue;
    }
    st.Set(k, stride);
    stride *= shape[k];
  }
  return Layout(std::move(sh), std::move(st));
}

// Identity layout over `shape`: per-axis unit ScaledBases E<k>.
template <typename Container>
inline Layout MakeIdentityLayout(const Container &shape) {
  // Each mode k carries the unit basis E<k>, so the layout maps a coordinate to
  // itself, tagged by which axis it came from (CuTe make_identity_layout).
  int64_t n = shape.size();
  Array<IntTuple> sh, st;
  for (int64_t k = 0; k < n; ++k) {
    sh.push_back(shape[k]);
    st.push_back(E({k}));
  }
  return Layout(sh, st);
}

// Column major layout over `shape`.
TILELANG_DLL Layout MakeColumnMajorLayout(const IntTuple &shape);

// Row major layout over `shape`.
TILELANG_DLL Layout MakeRowMajorLayout(const IntTuple &shape);

// Identity layout over `shape`: per-axis unit ScaledBases.
TILELANG_DLL Layout MakeIdentityLayout(const IntTuple &shape);

// Concatenate layouts.
TILELANG_DLL Layout MakeLayout(const Array<Layout> &layouts);

// A CuTe ComposedLayout of the form Swizzle o offset o Layout.
// Evaluating at x yields swizzle.apply(offset + layout(x)).
class ComposedLayoutNode : public Object {
public:
  Swizzle swizzle;
  int64_t offset{0};
  Layout layout;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
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
    return swizzle->Apply(offset + AsConst(layout(IntTuple(coord))));
  }
};

class ComposedLayout : public ObjectRef {
public:
  TILELANG_DLL ComposedLayout(Swizzle swizzle, int64_t offset, Layout layout);

  // Recast into a different element width: the swizzle's m_base shifts (see
  // Swizzle::Recast) and the strides/offset scale by old_bits/new_bits.
  TILELANG_DLL ComposedLayout Recast(int old_bits, int new_bits) const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ComposedLayout, ObjectRef,
                                             ComposedLayoutNode);
};

/// Recover an affine TileLang layout as a cute::Layout.
/// @return result(coords) == serialize(layout(coords))
/// @return nullopt if the address is not purely affine (e.g. it has a swizzle,
///         a non-zero base offset, or a non-constant extent).
Optional<Layout> LayoutFromTileLang(const tvm::tl::Layout &layout);

/// Recover a hierarchical affine TileLang layout as a cute::Layout.
/// Each mode in the output indices is multiplied with a scaled basis.
/// @return result(coords) == layout(coords)
/// @return nullopt if any of the output modes is not purely affine, or
///         different output modes are dependent.
Optional<Layout> LayoutFromTileLangHierarchical(const tvm::tl::Layout &layout);

/// Recover a possibly swizzled and offset TileLang layout as a
/// cute::ComposedLayout.
/// @return result(coords) == layout(coords)
/// @return nullopt if the recovery fails, e.g. the TileLang layout contains
///         dynamic values, or the analyzer is unable to prove the equivalence
///         between a cute::ComposedLayout and the TileLang layout.
/// @note Addresses are element-offset positions; recast with
///       `.Recast(dtype.bits(), 8)` for the byte-address uses.
Optional<ComposedLayout>
ComposedLayoutFromTileLang(const tvm::tl::Layout &layout,
                           std::optional<int> least_b_bits = std::nullopt);

/// Restrict the layout to a sublayout defined by a region.
/// @return result.sublayout(coords_in_region) + result.offset
//            == layout(range.min + coords_in_region)
ffi::Tuple<IntTuple, Layout> Restrict(const Layout &layout,
                                      const Array<Range> &range);

} // namespace cute
} // namespace tl
} // namespace tvm

#endif // TVM_TL_LAYOUT_CUTE_LAYOUT_H_
