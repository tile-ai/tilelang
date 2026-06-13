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
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("tl.cute.IntTuple", IntTupleNode, Object);
};

class IntTuple : public ObjectRef {
public:
  // Creates a static integer.
  TVM_DLL IntTuple(int64_t value);

  // Creates a either static or dynamic integer, based on the PrimExpr's value.
  TVM_DLL IntTuple(PrimExpr value);

  // The i-th child. A leaf has rank 1 and only index 0 is allowed.
  TVM_DLL IntTuple operator[](int64_t index) const;

  // Element-wise sum.
  TVM_DLL friend IntTuple operator+(const IntTuple &a, const IntTuple &b);
  IntTuple &operator+=(const IntTuple &other) {
    *this = *this + other;
    return *this;
  }

  // The product. The product of two tuples is dot-product.
  TVM_DLL friend IntTuple operator*(const IntTuple &a, const IntTuple &b);
  IntTuple &operator*=(const IntTuple &other) {
    *this = *this * other;
    return *this;
  }

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
  TVM_DLL IntTupleScaledBasis(IntTuple value, Array<int64_t> basis);
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
TVM_DLL IntTuple BasisGet(const Array<int64_t> &path, const IntTuple &t);

// The rank, or number of children, of an IntTuple. For a leaf, the rank is 1.
TVM_DLL int64_t Rank(const IntTuple &t);

/// Product of all leaf values.
TVM_DLL IntTuple Product(const IntTuple &t);

/// Flatten a tree into a IntTupleTuple of leaves.
/// A leaf flattens to itself.
/// @post depth <= 1.
TVM_DLL IntTuple Flatten(const IntTuple &t);

/// Wrap an IntTuple into a tuple.
/// @post IsTuple(result).
TVM_DLL IntTupleTuple Wrap(const IntTuple &t);

// Checks if two IntTuple's are congruent: whether their tree structure
// matches -- same nesting and same rank at every branch.
TVM_DLL bool Congruent(const IntTuple &a, const IntTuple &b);

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
  TVM_DLL Layout(Array<int64_t> shape, Array<int64_t> stride);
  // Convenience constructor: static shape, dynamic stride.
  TVM_DLL Layout(Array<int64_t> shape, Array<IntTuple> stride);
  // Convenience constructor: dynamic shape, dynamic stride.
  TVM_DLL Layout(Array<IntTuple> shape, Array<IntTuple> stride);
  // Canonical constructor.
  TVM_DLL Layout(IntTuple shape, IntTuple stride);

  // Sub-layout.
  TVM_DLL Layout operator[](int64_t index) const;

  /// Map a single coordinate to a physical index.
  /// @return a const or an expression; when stride has scaled basis, the result
  ///         is a tuple.
  TVM_DLL IntTuple operator()(const IntTuple &coord) const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Layout, ObjectRef, LayoutNode);
};

/// Same with IntTuple.
/// @return the rank of the shape (and stride).
TVM_DLL int64_t Rank(const Layout &layout);

/// Same with IntTuple.
/// @return the flattened shape and stride.
TVM_DLL Layout Flatten(const Layout &layout);

/// The domain size of the shape.
/// @return Product(layout->shape).
TVM_DLL IntTuple Size(const Layout &layout);

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

/// Same with Coalesce, but caps each merged run at `max_extent`. A run wider
/// than the cap stays split instead of fusing into one oversized mode.
/// @post depth <= 1.
TVM_DLL Layout Coalesce(const Layout &layout, int64_t max_extent);

/// Take a given number of modes from a layout.
TVM_DLL Layout Take(const Layout &layout, int64_t n);

// Column major layout over `shape`.
template <typename Container>
inline Layout MakeColumnMajorLayout(const Container &shape) {
  int64_t n = shape.size();
  Array<IntTuple> sh, st;
  sh.reserve(n);
  st.reserve(n);
  IntTuple stride = 1;
  for (int64_t k = 0; k < n; ++k) {
    sh.push_back(shape[k]);
    st.push_back(stride);
    stride *= shape[k];
  }
  return Layout(std::move(sh), std::move(st));
}

// Row major layout over `shape`.
template <typename Container>
inline Layout MakeRowMajorLayout(const Container &shape) {
  int64_t n = shape.size();
  Array<IntTuple> sh(n, IntTuple()), st(n, IntTuple());
  IntTuple stride = 1;
  for (int64_t k = n - 1; k >= 0; --k) {
    sh.Set(k, shape[k]);
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
TVM_DLL Layout MakeColumnMajorLayout(const IntTuple &shape);

// Row major layout over `shape`.
TVM_DLL Layout MakeRowMajorLayout(const IntTuple &shape);

// Identity layout over `shape`: per-axis unit ScaledBases.
TVM_DLL Layout MakeIdentityLayout(const IntTuple &shape);

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

} // namespace cute
} // namespace tl
} // namespace tvm

#endif // TVM_TL_LAYOUT_CUTE_LAYOUT_H_
