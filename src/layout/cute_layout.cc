/*!
 * \file cute_layout.cc
 * \brief CuTe-style layout IR types and TileLang-to-CuTe layout recovery.
 */

#include "cute_layout.h"
#include "layout.h"
#include "utils.h"

#include "support/check.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace tl {
namespace cute {

using namespace tirx;

namespace {

int CountTrailingZeros(uint64_t value) {
  ICHECK_NE(value, 0);
  int count = 0;
  while ((value & 1) == 0) {
    value >>= 1;
    ++count;
  }
  return count;
}

int CountLeadingZeros(uint64_t value) {
  ICHECK_NE(value, 0);
  int count = 0;
  for (uint64_t mask = uint64_t(1) << 63; (value & mask) == 0; mask >>= 1)
    ++count;
  return count;
}

int CountSetBits(uint64_t value) {
  int count = 0;
  while (value != 0) {
    value &= value - 1;
    ++count;
  }
  return count;
}

// Return log2(v) if v is a positive power of two (log2(1) == 0), else -1.
int Log2Exact(int64_t v) {
  if (v <= 0 || (v & (v - 1)) != 0)
    return -1;
  return CountTrailingZeros(static_cast<uint64_t>(v));
}

} // namespace

int64_t SwizzleNode::Apply(int64_t offset) const {
  if (b_bits <= 0)
    return offset;
  uint64_t mask = (static_cast<uint64_t>(1) << b_bits) - 1;
  uint64_t yyy = mask << (m_base + s_shift);
  uint64_t x = static_cast<uint64_t>(offset);
  return static_cast<int64_t>(x ^ ((x & yyy) >> s_shift));
}

Swizzle::Swizzle(int b_bits, int m_base, int s_shift) {
  auto node = make_object<SwizzleNode>();
  node->b_bits = b_bits;
  node->m_base = m_base;
  node->s_shift = s_shift;
  data_ = std::move(node);
}

Swizzle Swizzle::Recast(int old_bits, int new_bits) const {
  const SwizzleNode *n = get();
  ICHECK(n != nullptr) << "Recast on a null Swizzle";
  if (!n->IsSwizzled())
    return Swizzle::Identity();
  int old_log = Log2Exact(old_bits);
  int new_log = Log2Exact(new_bits);
  ICHECK(old_log >= 0 && new_log >= 0)
      << "Recast expects power-of-two element sizes, got " << old_bits << " -> "
      << new_bits;
  // Mirror CuTe recast_layout(Swizzle<B,M,S>): new>=old => upcast<new/old>
  // (NewM = M - log2N; clamp via B if negative); new<old => downcast<old/new>
  // (M + log2N).
  if (new_bits >= old_bits) {
    int new_m = n->m_base - (new_log - old_log);
    if (new_m >= 0)
      return Swizzle(n->b_bits, new_m, n->s_shift);
    return Swizzle(std::max(n->b_bits + new_m, 0), 0, n->s_shift);
  }
  return Swizzle(n->b_bits, n->m_base + (old_log - new_log), n->s_shift);
}

Swizzle Swizzle::Parse(const ffi::String &text) {
  std::string s(text);
  int b_bits, m_base, s_shift;
  ICHECK_EQ(std::sscanf(s.c_str(), "Sw<%d,%d,%d>", &b_bits, &m_base, &s_shift),
            3)
      << "Swizzle text must be 'Sw<b,m,s>', got '" << s << "'";
  return Swizzle(b_bits, m_base, s_shift);
}

void Swizzle::Print(std::ostream &os) const {
  os << "Sw<" << (*this)->b_bits << "," << (*this)->m_base << ","
     << (*this)->s_shift << ">";
}

IntTuple::IntTuple(int64_t value) {
  auto node = make_object<IntTupleConstNode>();
  node->value = value;
  data_ = std::move(node);
}

IntTuple::IntTuple(PrimExpr value) {
  if (const auto *imm = value.as<IntImmNode>()) {
    auto node = make_object<IntTupleConstNode>();
    node->value = imm->value;
    data_ = std::move(node);
  } else {
    auto node = make_object<IntTuplePrimExprNode>();
    node->value = std::move(value);
    data_ = std::move(node);
  }
}

IntTuple IntTuple::operator[](int64_t index) const {
  if (const auto *a = this->as<IntTupleTupleNode>()) {
    int64_t n = a->fields.size();
    ICHECK(index >= 0 && index < n)
        << "IntTuple index " << index << " out of range " << n;
    return a->fields[index];
  }
  // A scalar leaf has rank 1 and [0] returns itself.
  ICHECK(index == 0) << "Leaf index out of range: " << index << " vs 0";
  return *this;
}

namespace {

// Recursive-descent parser for the CuTe IntTuple spelling emitted by
// IntTuple::Print: nested "(a,b,...)" tuples, decimal integer leaves, and
// scaled bases "v@m@..." with the mode path innermost-first (E<1,0> is
// spelled 1@0@1), exactly inverting the printer. Whitespace is free.
class IntTupleParser {
public:
  explicit IntTupleParser(const std::string &text) : text_(text) {}

  IntTuple Parse() {
    IntTuple result = ParseTerm();
    SkipSpace();
    ICHECK_EQ(pos_, text_.size()) << "Trailing characters in IntTuple text: '"
                                  << text_.substr(pos_) << "'";
    return result;
  }

  IntTuple ParseTerm() {
    SkipSpace();
    IntTuple value;
    if (Peek() == '(') {
      ++pos_;
      Array<IntTuple> fields;
      SkipSpace();
      while (Peek() != ')') {
        fields.push_back(ParseTerm());
        SkipSpace();
        if (Peek() == ',') {
          ++pos_;
        } else {
          ICHECK_EQ(Peek(), ')') << "Expected ',' or ')' at position " << pos_
                                 << " of '" << text_ << "'";
        }
        SkipSpace();
      }
      ++pos_;
      value = IntTupleTuple(std::move(fields));
    } else {
      value = ParseInt();
    }
    // A scaled basis: value@m@... with the path innermost-first.
    SkipSpace();
    if (Peek() != '@')
      return value;
    std::vector<int64_t> path;
    while (Peek() == '@') {
      ++pos_;
      path.push_back(ParseInt());
      SkipSpace();
    }
    // The printed path is reversed (innermost-first); restore outer-first.
    return value * E(Array<int64_t>(path.rbegin(), path.rend()));
  }

private:
  char Peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }

  void SkipSpace() {
    while (pos_ < text_.size() && std::isspace(text_[pos_]))
      ++pos_;
  }

  int64_t ParseInt() {
    SkipSpace();
    size_t start = pos_;
    if (Peek() == '-')
      ++pos_;
    while (pos_ < text_.size() && std::isdigit(text_[pos_]))
      ++pos_;
    ICHECK_GT(pos_, start + (text_[start] == '-' ? 1 : 0))
        << "Expected an integer at position " << start << " of '" << text_
        << "'";
    return std::stoll(text_.substr(start, pos_ - start));
  }

  const std::string &text_;
  size_t pos_{0};
};

} // namespace

IntTuple IntTuple::Parse(const ffi::String &text) {
  std::string s(text);
  return IntTupleParser(s).Parse();
}

void IntTuple::Print(std::ostream &os) const {
  if (const auto *c = as<IntTupleConstNode>()) {
    os << c->value;
    return;
  }
  if (const auto *e = as<IntTuplePrimExprNode>()) {
    os << e->value;
    return;
  }
  if (const auto *b = as<IntTupleScaledBasisNode>()) {
    // CuTe scaled-basis spelling value@m@...: the mode path prints
    // innermost-first (CuTe's E<...> print order), e.g. E<1,0> -> 1@0@1.
    b->value.Print(os);
    for (auto it = b->basis.rbegin(); it != b->basis.rend(); ++it)
      os << "@" << *it;
    return;
  }
  const auto *a = as<IntTupleTupleNode>();
  ICHECK(a != nullptr) << "IntTuple::Print on an unknown IntTuple kind";
  os << "(";
  for (size_t i = 0; i < a->fields.size(); ++i) {
    if (i > 0)
      os << ",";
    a->fields[i].Print(os);
  }
  os << ")";
}

namespace {

// A leaf that is the constant 0 -- the additive identity (CuTe's C<0>).
bool IsZeroLeaf(const IntTuple &t) { return IsConst(t) && AsConst(t) == 0; }

// Sum two scalar leaves. As with MulLeaf, TVM's operator+ folds constants, so
// the result is an IntImm exactly when both operands are constant. A
// ScaledBasis leaf is not summable here -- the basis identity is preserved by
// operator+'s recursion, not by adding raw basis values.
IntTuple AddLeaf(const IntTuple &a, const IntTuple &b) {
  ICHECK(!IsScaledBasis(a) && !IsScaledBasis(b))
      << "Cannot add ScaledBasis leaves";
  ICHECK(!IsTuple(a) && !IsTuple(b)) << "Can only add leaves";
  if (IsConst(a) && IsConst(b))
    return IntTuple(AsConst(a) + AsConst(b));
  std::optional<DataType> dtype;
  if (IsPrimExpr(a))
    dtype = AsPrimExpr(a)->dtype;
  if (IsPrimExpr(b)) {
    auto dt = AsPrimExpr(b)->dtype;
    if (dtype.has_value()) {
      ICHECK_EQ(*dtype, dt) << "dtype mismatch: " << *dtype << " vs " << dt
                            << " between operands " << a << " and " << b;
    } else {
      dtype = dt;
    }
  }
  ICHECK(dtype.has_value()) << "At least one leaf must be a PrimExpr";
  return IntTuple(AsConstOrPrimExpr(a, *dtype) + AsConstOrPrimExpr(b, *dtype));
}

// Multiply two plain (non-basis) scalar leaves. TVM's operator* already folds
// constants and applies the 0/1 identities (arith::TryConstFold<Mul>), so the
// product is an IntImm exactly when constant -> pick the leaf kind from that.
IntTuple MulLeaf(const IntTuple &a, const IntTuple &b) {
  ICHECK(!IsScaledBasis(a) && !IsScaledBasis(b));
  ICHECK(!IsTuple(a) && !IsTuple(b)) << "Can only multiply leaves";
  if (IsConst(a) && IsConst(b)) {
    return IntTuple(AsConst(a) * AsConst(b));
  }
  std::optional<DataType> dtype;
  if (IsPrimExpr(a)) {
    dtype = AsPrimExpr(a)->dtype;
  }
  if (IsPrimExpr(b)) {
    auto dt = AsPrimExpr(b)->dtype;
    if (dtype.has_value()) {
      ICHECK_EQ(*dtype, dt) << "dtype mismatch: " << *dtype << " vs " << dt
                            << " between operands " << a << " and " << b;
    } else {
      dtype = dt;
    }
  }
  ICHECK(dtype.has_value()) << "At least one leaf must be a PrimExpr";
  PrimExpr p = AsConstOrPrimExpr(a, *dtype) * AsConstOrPrimExpr(b, *dtype);
  return IntTuple(p);
}

// Floored divide / modulo of two scalar leaves (const folds to a const leaf,
// else a PrimExpr leaf). Coordinates are non-negative, so floor matches the
// FloorDiv/FloorMod the dynamic path uses.
IntTuple DivLeaf(const IntTuple &a, const IntTuple &b) {
  ICHECK(!IsTuple(a) && !IsScaledBasis(a)) << "DivLeaf on a non-scalar leaf";
  if (IsConst(a) && IsConst(b)) {
    int64_t x = AsConst(a), y = AsConst(b);
    ICHECK(y != 0) << "division by zero";
    int64_t q = x / y, r = x % y; // round toward -inf (floor)
    return IntTuple(q - ((r != 0) && ((r < 0) != (y < 0)) ? 1 : 0));
  }
  DataType dt = IsPrimExpr(a) ? AsPrimExpr(a)->dtype : AsPrimExpr(b)->dtype;
  return IntTuple(FloorDiv(AsConstOrPrimExpr(a, dt), AsConstOrPrimExpr(b, dt)));
}

IntTuple ModLeaf(const IntTuple &a, const IntTuple &b) {
  ICHECK(!IsTuple(a) && !IsScaledBasis(a)) << "ModLeaf on a non-scalar leaf";
  if (IsConst(a) && IsConst(b)) {
    int64_t x = AsConst(a), y = AsConst(b);
    ICHECK(y != 0) << "modulo by zero";
    int64_t r = x % y;
    return IntTuple((r != 0 && (r < 0) != (y < 0)) ? r + y : r); // floored
  }
  DataType dt = IsPrimExpr(a) ? AsPrimExpr(a)->dtype : AsPrimExpr(b)->dtype;
  return IntTuple(FloorMod(AsConstOrPrimExpr(a, dt), AsConstOrPrimExpr(b, dt)));
}

// Compact column-major stride congruent to `shape` (CuTe compact_col_major):
// thread a running product `acc` through the leaves depth-first, mode 0
// fastest, keeping the nested structure -- each leaf takes the current `acc`
// and advances it by the leaf's extent. e.g. (2,(2,2)) -> (1,(2,4)). Dynamic
// extents thread through as PrimExpr products.
IntTuple CompactColMajor(const IntTuple &shape, IntTuple &acc) {
  if (IsTuple(shape)) {
    Array<IntTuple> out;
    for (int64_t i = 0, n = Rank(shape); i < n; ++i)
      out.push_back(CompactColMajor(shape[i], acc));
    return IntTupleTuple(out);
  }
  // A static size-1 mode takes stride 0 (CuTe compact, stride.hpp:302) and
  // leaves the running product untouched.
  if (IsConst(shape) && AsConst(shape) == 1)
    return IntTuple(int64_t(0));
  IntTuple stride = acc;
  acc = MulLeaf(acc, shape);
  return stride;
}

IntTuple CompactColMajor(const IntTuple &shape) {
  IntTuple acc = IntTuple(int64_t(1));
  return CompactColMajor(shape, acc);
}

// Compact row-major stride congruent to `shape` (CuTe compact_row_major): like
// CompactColMajor but the running product threads through the leaves in reverse
// (last mode fastest). e.g. (2,(2,2)) -> (4,(2,1)).
IntTuple CompactRowMajor(const IntTuple &shape, IntTuple &acc) {
  if (IsTuple(shape)) {
    int64_t n = Rank(shape);
    Array<IntTuple> out(n, IntTuple());
    for (int64_t i = n - 1; i >= 0; --i)
      out.Set(i, CompactRowMajor(shape[i], acc));
    return IntTupleTuple(out);
  }
  // A static size-1 mode takes stride 0 (CuTe compact, stride.hpp:302).
  if (IsConst(shape) && AsConst(shape) == 1)
    return IntTuple(int64_t(0));
  IntTuple stride = acc;
  acc = MulLeaf(acc, shape);
  return stride;
}

IntTuple CompactRowMajor(const IntTuple &shape) {
  IntTuple acc = IntTuple(int64_t(1));
  return CompactRowMajor(shape, acc);
}

// Identity layout over `shape` (CuTe make_identity_layout): each leaf at tree
// path (p0,p1,...) carries the unit basis E<p0,p1,...>, so the layout maps a
// coordinate to itself tagged by its full nested axis path.
IntTuple IdentityStride(const IntTuple &shape, std::vector<int64_t> &path) {
  if (IsTuple(shape)) {
    Array<IntTuple> out;
    for (int64_t i = 0, n = Rank(shape); i < n; ++i) {
      path.push_back(i);
      out.push_back(IdentityStride(shape[i], path));
      path.pop_back();
    }
    return IntTupleTuple(out);
  }
  return IntTupleScaledBasis(IntTuple(int64_t(1)),
                             Array<int64_t>(path.begin(), path.end()));
}

// Expand a ScaledBasis leaf v@(n0,n1,...) into its ArithmeticTuple form (CuTe
// as_arithmetic_tuple): v sits at nested position (n0,n1,...) with 0 in every
// other slot, e.g. 1@0 -> (1), 1@1 -> (0,1), 2@(1,0) -> (0,(2)). This is what
// lets two ScaledBasis terms on different axes add into one coordinate tuple.
IntTuple AsArithmeticTuple(const IntTuple &sb) {
  IntTuple result = BasisValue(sb);
  Array<int64_t> path = BasisPath(sb);
  for (int64_t i = static_cast<int64_t>(path.size()) - 1; i >= 0; --i) {
    Array<IntTuple> fields;
    for (int64_t j = 0; j < path[i]; ++j)
      fields.push_back(IntTuple(int64_t(0)));
    fields.push_back(result);
    result = IntTupleTuple(fields);
  }
  return result;
}

} // namespace

IntTuple operator+(const IntTuple &a, const IntTuple &b) {
  // CuTe ArithmeticTuple addition. A zero leaf is the additive identity, so it
  // adds to anything (including a tuple of a different rank).
  if (IsZeroLeaf(a))
    return b;
  if (IsZeroLeaf(b))
    return a;
  if (IsTuple(a) || IsTuple(b)) {
    // A ScaledBasis on either side expands to its ArithmeticTuple form so it
    // can add to the tuple (CuTe as_arithmetic_tuple); a plain non-zero leaf
    // has no slot and is ill-formed.
    if (IsScaledBasis(a))
      return AsArithmeticTuple(a) + b;
    if (IsScaledBasis(b))
      return a + AsArithmeticTuple(b);
    ICHECK(IsTuple(a) && IsTuple(b))
        << "Cannot add a non-zero leaf to a tuple: " << a << " + " << b;
    // Zero-pad the shorter to the longer's rank, then add component-wise.
    int64_t ra = Rank(a), rb = Rank(b), r = std::max(ra, rb);
    Array<IntTuple> out;
    out.reserve(r);
    for (int64_t i = 0; i < r; i++) {
      IntTuple ai = i < ra ? a[i] : IntTuple(int64_t(0));
      IntTuple bi = i < rb ? b[i] : IntTuple(int64_t(0));
      out.push_back(ai + bi);
    }
    return IntTupleTuple(std::move(out));
  }
  // ScaledBasis leaves expand to their ArithmeticTuple form and add as tuples
  // (CuTe as_arithmetic_tuple): both same-axis and distinct-axis sums become
  // coordinate tuples -- e.g. 1@0 + 2@0 -> (1)+(2) -> (3), and 1@0 + 1@1 ->
  // (1)+(0,1) -> (1,1).
  bool ba = IsScaledBasis(a), bb = IsScaledBasis(b);
  if (ba && bb)
    return AsArithmeticTuple(a) + AsArithmeticTuple(b);
  // A lone ScaledBasis added to a plain leaf: only the zero identity (handled
  // above) is well-formed; a non-zero plain leaf has no axis to land on.
  ICHECK(!ba && !bb) << "Cannot add a ScaledBasis and a non-basis leaf: " << a
                     << " + " << b;
  return AddLeaf(a, b);
}

IntTuple operator*(const IntTuple &a, const IntTuple &b) {
  bool ba = IsScaledBasis(a), bb = IsScaledBasis(b);
  ICHECK(!(ba && bb)) << "Two ScaledBasis leaves cannot multiply";
  // A ScaledBasis multiplies into its scale (CuTe arithmetic_tuple.hpp
  // operator*: a * E<Ns...>{v} == E<Ns...>{a * v}) with no special case for
  // zero -- 0 * E<1> is the basis term 0@1, not the plain additive identity
  // (only ArithmeticTuple ADDITION has C<0> shortcuts).  The recursion
  // distributes a tuple peer over its leaves ((2,3) * E<1> == (2@1, 3@1)),
  // an extension beyond CuTe (which only defines scalar peers).
  if (ba || bb) {
    const IntTuple &basis = ba ? a : b;
    const IntTuple &peer = ba ? b : a;
    Array<int64_t> path = BasisPath(basis);
    IntTuple scale = BasisValue(basis);
    return TransformLeaf(peer, [&](const IntTuple &leaf) -> IntTuple {
      return IntTuple(IntTupleScaledBasis(MulLeaf(scale, leaf), path));
    });
  }
  if (IsTuple(a) || IsTuple(b)) {
    int64_t ra = Rank(a), rb = Rank(b);
    ICHECK_EQ(ra, rb) << "Must have the same rank";
    Array<IntTuple> out;
    for (int64_t i = 0; i < ra; i++) {
      out.push_back(a[i] * b[i]);
    }
    return IntTupleTuple(std::move(out));
  }
  return MulLeaf(a, b);
}

IntTupleScaledBasis::IntTupleScaledBasis(IntTuple value, Array<int64_t> basis) {
  ICHECK(!IsTuple(value)) << "ScaledBasis value must be a scalar leaf "
                             "(const/primexpr), not a tuple";
  auto node = make_object<IntTupleScaledBasisNode>();
  node->value = std::move(value);
  node->basis = std::move(basis);
  data_ = std::move(node);
}

IntTupleTuple::IntTupleTuple(Array<IntTuple> fields) {
  auto node = make_object<IntTupleTupleNode>();
  node->fields = std::move(fields);
  data_ = std::move(node);
}

IntTuple BasisGet(const Array<int64_t> &path, const IntTuple &t) {
  IntTuple result = t;
  for (auto i : path) {
    result = result[i];
  }
  return result;
}

int64_t Rank(const IntTuple &t) {
  const auto *a = t.as<IntTupleTupleNode>();
  return a ? static_cast<int64_t>(a->fields.size()) : 1;
}

IntTuple Product(const IntTuple &t) {
  if (!IsTuple(t))
    return t;
  IntTuple acc = IntTuple(1);
  for (int64_t i = 0, n = Rank(t); i < n; ++i)
    acc = MulLeaf(acc, Product(t[i]));
  return acc;
}

IntTuple Flatten(const IntTuple &t) {
  if (!IsTuple(t))
    return t;
  Array<IntTuple> out;
  FoldLeaves(t, 0, [&](int, const IntTuple &s) {
    out.push_back(s);
    return 0;
  });
  return IntTupleTuple(out);
}

IntTupleTuple Wrap(const IntTuple &t) {
  return IsTuple(t) ? AsTuple(t) : IntTupleTuple({t});
}

bool Congruent(const IntTuple &a, const IntTuple &b) {
  if (IsTuple(a) != IsTuple(b))
    return false;
  if (!IsTuple(a))
    return true;
  int64_t n = Rank(a);
  if (Rank(b) != n)
    return false;
  for (int64_t i = 0; i < n; ++i)
    if (!Congruent(a[i], b[i]))
      return false;
  return true;
}

bool Compatible(const IntTuple &a, const IntTuple &b) {
  if (IsTuple(a) && IsTuple(b)) {
    int64_t n = Rank(a);
    if (Rank(b) != n)
      return false;
    for (int64_t i = 0; i < n; ++i) {
      if (!Compatible(a[i], b[i]))
        return false;
    }
    return true;
  }
  if (IsTuple(a))
    return false;

  // Shapes have integral leaves.  A ScaledBasis is meaningful in a stride,
  // but it is not a shape terminal and cannot participate in compatibility.
  if ((!IsConst(a) && !IsPrimExpr(a)) ||
      FoldLeaves(b, false, [](bool invalid, const IntTuple &leaf) {
        return invalid || (!IsConst(leaf) && !IsPrimExpr(leaf));
      })) {
    return false;
  }

  IntTuple b_size = Product(b);
  if (IsConst(a) && IsConst(b_size))
    return AsConst(a) == AsConst(b_size);

  // CuTe compares integral terminals by value, not by expression identity.
  // Use the dynamic terminal's dtype for a constant peer and let Analyzer
  // prove algebraically equivalent expressions (for example n + n == 2*n).
  DataType dtype =
      IsPrimExpr(a) ? AsPrimExpr(a)->dtype : AsPrimExpr(b_size)->dtype;
  if (IsPrimExpr(a) && IsPrimExpr(b_size) &&
      AsPrimExpr(a)->dtype != AsPrimExpr(b_size)->dtype) {
    return false;
  }
  arith::Analyzer analyzer;
  return analyzer.CanProveEqual(AsConstOrPrimExpr(a, dtype),
                                AsConstOrPrimExpr(b_size, dtype));
}

Layout::Layout(Array<int64_t> shape, Array<int64_t> stride) {
  ICHECK_EQ(shape.size(), stride.size());
  Array<IntTuple> shape_fields, stride_fields;
  for (size_t k = 0; k < shape.size(); ++k) {
    shape_fields.push_back(IntTuple(shape[k]));
    stride_fields.push_back(IntTuple(stride[k]));
  }
  auto node = make_object<LayoutNode>();
  node->shape = IntTupleTuple(shape_fields);
  node->stride = IntTupleTuple(stride_fields);
  data_ = std::move(node);
}

Layout::Layout(Array<int64_t> shape, Array<IntTuple> stride) {
  ICHECK_EQ(shape.size(), stride.size());
  Array<IntTuple> shape_fields;
  for (size_t k = 0; k < shape.size(); ++k) {
    shape_fields.push_back(IntTuple(shape[k]));
    ICHECK(!IsTuple(stride[k]));
  }
  auto node = make_object<LayoutNode>();
  node->shape = IntTupleTuple(shape_fields);
  node->stride = IntTupleTuple(stride);
  data_ = std::move(node);
}

Layout::Layout(Array<IntTuple> shape, Array<IntTuple> stride) {
  ICHECK(Congruent(IntTupleTuple(shape), IntTupleTuple(stride)))
      << "Layout shape and stride must be congruent: " << IntTupleTuple(shape)
      << " vs " << IntTupleTuple(stride);
  auto node = make_object<LayoutNode>();
  node->shape = IntTupleTuple(shape);
  node->stride = IntTupleTuple(stride);
  data_ = std::move(node);
}

Layout::Layout(IntTuple shape, IntTuple stride) {
  ICHECK(Congruent(shape, stride))
      << "Layout shape and stride must be congruent: " << shape << " vs "
      << stride;
  auto node = make_object<LayoutNode>();
  node->shape = std::move(shape);
  node->stride = std::move(stride);
  data_ = std::move(node);
}

Layout Layout::operator[](int64_t index) const {
  return Layout((*this)->shape[index], (*this)->stride[index]);
}

namespace {

// CuTe crd2idx(coord, shape, stride): map a coordinate within <shape, stride>
// to its codomain value. The result is a scalar leaf for plain strides and a
// coordinate IntTuple when a stride is a ScaledBasis (operator+ accumulates the
// per-axis contributions). Neither shape nor stride need be constant -- every
// step goes through the dynamic-aware leaf helpers.
//
// Three cases (mirroring CuTe stride.hpp):
//   op((c,C),(s,S),(d,D)) = op(c,s,d) + op((C),(S),(D))   [tuple coord]
//   op(c,(s,S),(d,D))     = op(c%prod(s),s,d) + op(c/prod(s),(S),(D))  [itt]
//   op(c,s,d)             = c * d                          [scalar]
IntTuple Crd2Idx(const IntTuple &coord, const IntTuple &shape,
                 const IntTuple &stride) {
  if (IsTuple(coord)) {
    // Coord, shape, stride are all tuples: per-mode, summed (crd2idx_ttt).
    ICHECK(IsTuple(shape) && IsTuple(stride)) << "crd2idx rank mismatch";
    int64_t r = Rank(coord);
    ICHECK(Rank(shape) == r && Rank(stride) == r) << "crd2idx rank mismatch";
    IntTuple acc = IntTuple(int64_t(0));
    for (int64_t i = 0; i < r; ++i)
      acc = acc + Crd2Idx(coord[i], shape[i], stride[i]);
    return acc;
  }
  if (IsTuple(shape)) {
    // Scalar coord, tuple shape/stride: split the coord across the modes by
    // divmod, skipping the mod on the LAST mode so the layout extends linearly
    // past its domain there (crd2idx_itt). For in-domain coords this matches a
    // full mod, since the final remainder is already smaller than its extent.
    ICHECK(IsTuple(stride)) << "crd2idx rank mismatch";
    int64_t r = Rank(shape);
    ICHECK(Rank(stride) == r) << "crd2idx rank mismatch";
    IntTuple acc = IntTuple(int64_t(0));
    IntTuple rem = coord;
    for (int64_t i = 0; i < r; ++i) {
      IntTuple prod = Product(shape[i]);
      IntTuple crd = i + 1 < r ? ModLeaf(rem, prod) : rem;
      acc = acc + Crd2Idx(crd, shape[i], stride[i]);
      if (i + 1 < r)
        rem = DivLeaf(rem, prod);
    }
    return acc;
  }
  // Scalar coord, scalar shape, scalar stride: c * d. operator* keeps a
  // zero product on its basis (0@i, like CuTe), which is what pads the
  // coordinate sum to full rank: E<0>{c} + E<1>{0} == (c, 0).
  return coord * stride;
}

} // namespace

IntTuple Layout::operator()(const IntTuple &coord) const {
  return Crd2Idx(coord, (*this)->shape, (*this)->stride);
}

int64_t Rank(const Layout &layout) { return Rank(layout->shape); }

Layout Flatten(const Layout &layout) {
  return Layout(Flatten(layout->shape), Flatten(layout->stride));
}

IntTuple Size(const Layout &layout) { return Product(layout->shape); }

namespace {

// Core of coalesce (CuTe bw_coalesce): merge adjacent modes that describe the
// same affine run. `fsh`/`fst` are the flat (leaf) modes fastest-first; the
// scan runs from the slowest mode down to mode 0, accumulating into a running
// "front" mode (front_sh, front_st) that the caller seeds with the slowest
// mode. For each next-faster mode we either:
//   - drop it if it has extent 1 (contributes nothing);
//   - adopt it if the front is currently extent 1 (placeholder);
//   - fuse it into the front when the front continues it contiguously, i.e.
//     mode.shape * mode.stride == front.stride, and the fused extent stays
//     within max_extent (the TMA box derivation caps this at 256); or
//   - otherwise emit the front and start a new one.
// A layout that collapses entirely (all extent-1) becomes the scalar (1):(0).
Layout BwCoalesce(const IntTuple &fsh, const IntTuple &fst, IntTuple front_sh,
                  IntTuple front_st, int64_t max_extent) {
  std::vector<IntTuple> res_sh, res_st;
  auto prepend = [&](IntTuple s, IntTuple d) {
    res_sh.insert(res_sh.begin(), s);
    res_st.insert(res_st.begin(), d);
  };
  for (int64_t i = Rank(fsh) - 2; i >= 0; --i) {
    IntTuple os = fsh[i], od = fst[i];
    if (IsConst(os) && AsConst(os) == 1)
      continue; // extent-1 mode: contributes nothing.
    if (IsConst(front_sh) && AsConst(front_sh) == 1) {
      front_sh = os; // front is a placeholder: adopt this mode.
      front_st = od;
      continue;
    }
    // Contiguity test: this mode and the front fuse iff stepping past this mode
    // (shape*stride) lands exactly on the front's stride.
    bool mergeable = IsConst(os) && IsConst(front_sh) &&
                     StructuralEqual()(os * od, front_st) &&
                     AsConst(os) * AsConst(front_sh) <= max_extent;
    if (mergeable) {
      front_sh = IntTuple(AsConst(os) * AsConst(front_sh));
      front_st = od;
    } else {
      prepend(front_sh, front_st);
      front_sh = os;
      front_st = od;
    }
  }
  prepend(front_sh, front_st);
  if (res_sh.size() == 1 && IsConst(res_sh[0]) && AsConst(res_sh[0]) == 1)
    return Layout(IntTuple(int64_t(1)), IntTuple(int64_t(0)));
  // CuTe bw_coalesce returns Layout<NewShape,NewStride> with a SCALAR NewShape
  // when a single mode survives (it only builds a tuple via prepend on >=2
  // modes). Mirror that: a one-mode result is scalar-shaped, not a 1-tuple, so
  // logical_divide/complement stay flat and congruent matches CuTe.
  if (res_sh.size() == 1)
    return Layout(res_sh[0], res_st[0]);
  return Layout(Array<IntTuple>(res_sh.begin(), res_sh.end()),
                Array<IntTuple>(res_st.begin(), res_st.end()));
}

// Simplify a layout by fusing contiguous modes (CuTe coalesce). Result is flat
// (depth <= 1) and equal to the input as a function on [0, size). `max_extent`
// caps each fused run (the TMA box derivation passes 256 to keep box modes
// within the descriptor's per-mode boxDim limit).
Layout CoalesceImpl(const Layout &layout, int64_t max_extent) {
  IntTuple fsh = Flatten(layout->shape), fst = Flatten(layout->stride);
  int64_t R = Rank(fsh);
  return BwCoalesce(fsh, fst, fsh[R - 1], fst[R - 1], max_extent);
}

} // namespace

Layout Coalesce(const Layout &layout) {
  ICHECK(layout.defined()) << "Coalesce on a null Layout";
  constexpr int64_t kInf = std::numeric_limits<int64_t>::max();
  return CoalesceImpl(layout, kInf);
}

Layout RightInverse(const Layout &layout) {
  ICHECK(layout.defined()) << "RightInverse on a null Layout";
  // Build the layout `r` with layout(r(i)) == i for i < size(r) (CuTe
  // right_inverse). After coalescing, sort the modes by stride and walk them
  // looking for the chain 1, s0, s0*s1, ... where each mode's stride equals the
  // running product of previous extents. Such a chain is exactly the part of
  // the layout that is a bijection onto a contiguous prefix; the inverse sends
  // each chained mode back to its position (the input-side prefix product).
  // Modes that break the chain -- including any with non-constant strides,
  // which have no comparable order -- are dropped, so the result inverts only
  // the bijective prefix. Requires a concrete stride order, hence const strides
  // only.
  Layout c = Coalesce(layout); // flat; no Flatten needed below.
  IntTuple m_sh = c->shape, m_st = c->stride;
  int64_t r = Rank(m_sh);

  // preprod[k] = product of extents before mode k (its coordinate's weight).
  std::vector<int64_t> sh(r), preprod(r);
  int64_t acc = 1;
  for (int64_t k = 0; k < r; ++k) {
    sh[k] = AsConst(m_sh[k]);
    preprod[k] = acc;
    acc *= sh[k];
  }

  std::vector<int64_t> order;
  for (int64_t k = 0; k < r; ++k)
    if (IsConst(m_st[k]))
      order.push_back(k);
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    return AsConst(m_st[a]) < AsConst(m_st[b]);
  });

  Array<IntTuple> out_sh, out_st;
  int64_t current = 1; // next stride the chain must hit to stay contiguous.
  for (int64_t k : order) {
    int64_t st = AsConst(m_st[k]);
    ICHECK(st >= 0) << "RightInverse requires non-negative strides, got " << st;
    if (st == current) {
      out_sh.push_back(IntTuple(sh[k]));
      out_st.push_back(IntTuple(preprod[k]));
      current = sh[k] * st;
    }
  }
  // Empty chain (nothing invertible) is the scalar (1):(0).
  if (out_sh.empty())
    return Coalesce(Layout(Array<int64_t>{1}, Array<int64_t>{0}));
  return Coalesce(Layout(out_sh, out_st));
}

Layout LeftInverse(const Layout &layout) {
  // CuTe left_inverse (layout.hpp:1324): coalesce, sort the modes by stride,
  // and divide successive strides.  Each emitted mode's shape is the ratio
  // stride / size-so-far -- a serialized gap folds INTO the mode covering it
  // -- and its stride is the sorted mode's position in the flattened domain
  // (the exclusive prefix product), with a leading stride-0 slot absorbing
  // everything below the smallest stride.  Stride-0 (broadcast) modes are
  // skipped; the largest-stride mode contributes its full extent last.
  Layout flat = Coalesce(layout);
  IntTuple fsh = Flatten(flat->shape), fst = Flatten(flat->stride);
  int64_t r = Rank(fsh);
  std::vector<int64_t> order, preprod(r, 1);
  int64_t acc = 1;
  for (int64_t i = 0; i < r; ++i) {
    ICHECK(IsConst(fsh[i]) && IsConst(fst[i]))
        << "LeftInverse requires static shapes and strides, got " << flat;
    preprod[i] = acc;
    acc *= AsConst(fsh[i]);
    order.push_back(i);
  }
  // stable_sort: ties (several stride-0 broadcast modes) keep their mode
  // order, so the result is a pure function of the input on every platform
  // (std::sort's tie order is implementation-defined).
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    return AsConst(fst[a]) < AsConst(fst[b]);
  });
  Array<IntTuple> out_sh;
  Array<IntTuple> out_st{IntTuple(int64_t(0))};
  int64_t covered = 1; // product of the emitted ratios == last stride
  for (int64_t k = 0; k < r; ++k) {
    int64_t d = AsConst(fst[order[k]]);
    if (d == 0)
      continue;
    ICHECK(d % covered == 0)
        << "LeftInverse divisibility condition: stride " << d
        << " is not a multiple of the covered extent " << covered;
    out_sh.push_back(d / covered);
    out_st.push_back(preprod[order[k]]);
    covered = d;
  }
  out_sh.push_back(fsh[order[r - 1]]);
  return Coalesce(Layout(IntTupleTuple(out_sh), IntTupleTuple(out_st)));
}

namespace {

int64_t Signum(int64_t v) { return (v > 0) - (v < 0); }
int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Like Coalesce, but additionally keeps the layout's behavior correct for
// coordinates PAST its domain (CuTe coalesce_x). Plain Coalesce can drop a
// trailing extent-1 mode whose stride still matters out of range -- e.g.
// (4,1):(1,0) coalesces to 4:1, losing the trailing slot. Seeding the scan with
// a shape-2 placeholder instead preserves that slot, yielding (4,2):(1,0).
// Composition needs this because the rhs codomain can index the lhs beyond its
// nominal size.
Layout CoalesceX(const Layout &layout) {
  IntTuple fsh = Flatten(layout->shape), fst = Flatten(layout->stride);
  int64_t R = Rank(fsh);
  IntTuple seed_sh = fsh[R - 1];
  if (IsConst(seed_sh) && AsConst(seed_sh) == 1)
    seed_sh = IntTuple(int64_t(2));
  constexpr int64_t kInf = std::numeric_limits<int64_t>::max();
  return BwCoalesce(fsh, fst, seed_sh, fst[R - 1], kInf);
}

// Reconstruct a tree shape from a set of basis paths. Each path is a sequence
// of child indices (a ScaledBasis E<i,j,...>); this groups the paths by their
// first index to rebuild the nesting they came from. A leaf IntTuple(0) marks a
// position reached by a plain (path-free) stride; a tuple marks a position with
// further structure. See Coprofile for what the result is used for.
IntTuple BuildProfile(const std::vector<std::vector<int64_t>> &paths) {
  int64_t max_idx = -1;
  for (const auto &p : paths)
    if (!p.empty())
      max_idx = std::max(max_idx, p[0]);
  if (max_idx < 0)
    return IntTuple(int64_t(0)); // no structure remaining: a plain slot.
  Array<IntTuple> fields;
  for (int64_t j = 0; j <= max_idx; ++j) {
    std::vector<std::vector<int64_t>> sub;
    for (const auto &p : paths)
      if (!p.empty() && p[0] == j)
        sub.emplace_back(p.begin() + 1, p.end());
    fields.push_back(sub.empty() ? IntTuple(int64_t(0)) : BuildProfile(sub));
  }
  return IntTupleTuple(fields);
}

// Describe the structure of a layout's codomain as a tree of placeholders (CuTe
// coprofile). The shape (a tree of 0-leaves) tells composition how finely to
// coalesce the lhs: composition coalesces the lhs "at the terminals" of this
// profile, so a coarser profile fuses more lhs modes together. A layout with
// only plain integer strides has an unstructured (scalar) codomain, so the
// profile is a single leaf -- the lhs is fully coalesced. ScaledBasis strides
// address distinct codomain axes by path, so the profile is a tree indexed by
// those paths and the lhs is coalesced per-axis instead, which keeps each lhs
// mode addressable by its basis path during composition.
IntTuple Coprofile(const IntTuple &stride) {
  std::vector<std::vector<int64_t>> paths;
  bool any_basis = false;
  FoldLeaves(stride, 0, [&](int, const IntTuple &s) {
    if (IsScaledBasis(s)) {
      any_basis = true;
      Array<int64_t> p = BasisPath(s);
      paths.emplace_back(p.begin(), p.end());
    } else {
      paths.emplace_back();
    }
    return 0;
  });
  if (!any_basis)
    return IntTuple(int64_t(0));
  return BuildProfile(paths);
}

// Coalesce a layout only as far as `profile` allows (CuTe coalesce_x at the
// profile terminals): a leaf profile fully coalesces this (sub)layout, while a
// tuple profile recurses into the corresponding lhs modes and leaves the rest
// uncoalesced. Any lhs modes beyond the profile's rank are kept as-is.
Layout CoalesceXProfile(const Layout &layout, const IntTuple &profile) {
  if (!IsTuple(profile))
    return CoalesceX(layout);
  int64_t rl = Rank(layout->shape), rp = Rank(profile);
  int64_t r = std::min(rl, rp);
  Array<IntTuple> out_sh, out_st;
  for (int64_t i = 0; i < r; ++i) {
    Layout sub = CoalesceXProfile(layout[i], profile[i]);
    out_sh.push_back(sub->shape);
    out_st.push_back(sub->stride);
  }
  for (int64_t i = r; i < rl; ++i) { // lhs modes the profile doesn't reach.
    out_sh.push_back(layout->shape[i]);
    out_st.push_back(layout->stride[i]);
  }
  return Layout(IntTupleTuple(out_sh), IntTupleTuple(out_st));
}

// Collapse rank-1 tuples down to their single element (CuTe unwrap): ((x)) ->
// x, stopping at the first leaf or rank>1 tuple.
IntTuple Unwrap(const IntTuple &t) {
  if (IsTuple(t) && Rank(t) == 1)
    return Unwrap(t[0]);
  return t;
}

// Compose one rhs mode through the lhs (CuTe composition_impl): returns the
// layout r such that r(c) == lhs(rhs_mode(c)), with a shape tree congruent to
// the rhs mode. The lhs is kept hierarchical (not flattened) so that a
// ScaledBasis rhs stride can address a specific nested lhs mode by its path.
// Five cases, in priority order:
//   1. rhs is a tuple: compose each child independently (composition
//   distributes
//      over the rhs modes) and keep the rhs's tuple shape.
//   2. rhs stride is a ScaledBasis E<path>*k: that codomain axis is lhs mode
//      `path`, so recurse into that lhs mode with scalar rhs stride k.
//   3. rhs stride 0: the rhs mode lands on a single lhs element; pass through.
//   4. lhs is a single mode: the composed stride is just rhs_stride *
//   lhs_stride.
//   5. lhs is a tuple: peel the rhs mode across the lhs modes (the divide loop
//      below).
// rhs_stride is always a constant int or a ScaledBasis here; lhs shapes are
// constant ints, lhs strides any leaf (const/dynamic/basis).
Layout CompositionImpl(const IntTuple &lhs_shape, const IntTuple &lhs_stride,
                       const IntTuple &rhs_shape, const IntTuple &rhs_stride) {
  // 1. rhs tuple: compose each child, preserving rhs's shape.
  if (IsTuple(rhs_shape)) {
    int64_t r = Rank(rhs_shape);
    Array<IntTuple> out_sh, out_st;
    for (int64_t i = 0; i < r; ++i) {
      Layout sub =
          CompositionImpl(lhs_shape, lhs_stride, rhs_shape[i], rhs_stride[i]);
      out_sh.push_back(sub->shape);
      out_st.push_back(sub->stride);
    }
    return Layout(IntTupleTuple(out_sh), IntTupleTuple(out_st));
  }
  // 2. rhs stride E<path>*k: route into the lhs mode named by `path`.
  if (IsScaledBasis(rhs_stride)) {
    Array<int64_t> path = BasisPath(rhs_stride);
    return CompositionImpl(BasisGet(path, lhs_shape),
                           BasisGet(path, lhs_stride), rhs_shape,
                           BasisValue(rhs_stride));
  }
  // 3. rhs stride 0: the whole rhs mode maps to one address.
  if (IsConst(rhs_stride) && AsConst(rhs_stride) == 0)
    return Layout(rhs_shape, rhs_stride);
  // 4. single lhs mode: stride composes by multiplication.
  if (!IsTuple(lhs_shape))
    return Layout(rhs_shape, rhs_stride * lhs_stride);

  // 5. lhs tuple, scalar rhs mode. Walk the lhs modes fastest-first, splitting
  //    the rhs mode across them: each lhs mode i consumes the part of the rhs
  //    extent that lands within it (new_shape) at stride
  //    rest_stride*lhs.stride, and `rest_*` carries the leftover extent/stride
  //    into the next lhs mode. The divisibility checks are CuTe's precondition
  //    that composition is well-defined (the rhs mode must tile cleanly across
  //    the lhs modes).
  ICHECK(IsConst(rhs_stride))
      << "Composition RHS stride must be a constant int or a ScaledBasis";
  int64_t R = Rank(lhs_shape);
  Array<IntTuple> res_shape, res_stride;
  int64_t rest_shape = AsConst(rhs_shape);
  int64_t rest_stride = AsConst(rhs_stride);
  for (int64_t i = 0; i < R - 1; ++i) {
    int64_t curr_shape = AsConst(lhs_shape[i]);
    ICHECK(rest_stride % curr_shape == 0 || rest_stride < curr_shape)
        << "Composition stride divisibility: rest_stride=" << rest_stride
        << ", curr_shape=" << curr_shape;
    // How much of this lhs mode the current stride spans, and the stride left
    // over for the next mode.
    int64_t next_shape = CeilDiv(curr_shape, std::abs(rest_stride));
    int64_t next_stride =
        CeilDiv(std::abs(rest_stride), curr_shape) * Signum(rest_stride);
    if (next_shape == 1 || rest_shape == 1) {
      rest_stride = next_stride; // nothing lands in this mode; carry on.
    } else {
      int64_t new_shape = std::min(next_shape, rest_shape);
      ICHECK(rest_shape % new_shape == 0)
          << "Composition shape divisibility: rest_shape=" << rest_shape
          << ", new_shape=" << new_shape;
      res_shape.push_back(IntTuple(new_shape));
      res_stride.push_back(IntTuple(rest_stride) * lhs_stride[i]);
      rest_shape /= new_shape;
      rest_stride = next_stride;
    }
  }
  // The remaining extent rides the slowest lhs mode's stride.
  IntTuple last = lhs_stride[R - 1];
  if (res_shape.empty())
    return Layout(IntTuple(rest_shape), IntTuple(rest_stride) * last);
  if (rest_shape == 1) // exactly tiled: drop the trivial trailing mode.
    return Layout(Unwrap(IntTupleTuple(res_shape)),
                  Unwrap(IntTupleTuple(res_stride)));
  res_shape.push_back(IntTuple(rest_shape));
  res_stride.push_back(IntTuple(rest_stride) * last);
  return Layout(IntTupleTuple(res_shape), IntTupleTuple(res_stride));
}

} // namespace

Layout Composition(const Layout &lhs, const Layout &rhs) {
  ICHECK(lhs.defined() && rhs.defined()) << "Composition on a null Layout";
  // result(c) == lhs(rhs(c)). The lhs is first coalesced to the granularity the
  // rhs actually addresses (Coprofile): plain rhs strides let the lhs be fully
  // coalesced, so CompositionImpl's divide loop sees flat lhs modes;
  // ScaledBasis rhs strides keep the lhs split per-axis so each basis path
  // still resolves to its own lhs mode. CompositionImpl then composes the
  // (possibly hierarchical) trees directly.
  Layout flat_lhs = CoalesceXProfile(lhs, Coprofile(rhs->stride));
  return CompositionImpl(flat_lhs->shape, flat_lhs->stride, rhs->shape,
                         rhs->stride);
}

Layout Coalesce(const Layout &layout, int64_t max_extent) {
  ICHECK(layout.defined()) << "Coalesce on a null Layout";
  return CoalesceImpl(layout, max_extent);
}

Layout Take(const Layout &layout, int64_t n) {
  ICHECK(layout.defined());
  int64_t r = Rank(layout);
  ICHECK(0 <= n && n <= r) << "Take(n) out of range: n=" << n << ", rank=" << r;
  Array<IntTuple> shape, stride;
  for (int64_t i = 0; i < n; ++i) {
    shape.push_back(layout->shape[i]);
    stride.push_back(layout->stride[i]);
  }
  return Layout(std::move(shape), std::move(stride));
}

Layout Filter(const Layout &layout) {
  // CuTe filter = coalesce(filter_zeros): a constant-0-stride mode becomes
  // size-1 (then dropped by coalesce). Dynamic/ScaledBasis modes pass through.
  IntTuple fsh = Flatten(layout->shape), fst = Flatten(layout->stride);
  Array<IntTuple> out_sh, out_st;
  for (int64_t i = 0, r = Rank(fsh); i < r; ++i) {
    out_sh.push_back(IsZeroLeaf(fst[i]) ? IntTuple(int64_t(1)) : fsh[i]);
    out_st.push_back(fst[i]);
  }
  return Coalesce(Layout(out_sh, out_st));
}

IntTuple Coshape(const Layout &layout) {
  // CuTe coshape (layout.hpp:649):
  //   m1_shapes   = transform_leaf(shape,  s => s - 1)
  //   abs_strides = transform_leaf(stride, abs)
  //   co_coord    = as_arithmetic_tuple(inner_product(m1_shapes, abs_strides))
  //   coshape     = transform_leaf(co_coord, c => c + 1)
  // The inner product's ArithmeticTuple sum is our operator+, so ScaledBasis
  // strides accumulate per coordinate axis: coshape((128,32):(1@0,1@1)) ==
  // (128,32); plain strides sum into one scalar extent.
  IntTuple m1_shapes = TransformLeaf(layout->shape, [](const IntTuple &s) {
    return IntTuple(AddLeaf(s, IntTuple(int64_t(-1))));
  });
  IntTuple abs_strides =
      TransformLeaf(layout->stride, [](const IntTuple &d) -> IntTuple {
        if (IsScaledBasis(d)) {
          IntTuple v = BasisValue(d);
          IntTuple abs_v = IsConst(v) ? IntTuple(std::abs(AsConst(v)))
                                      : IntTuple(tvm::abs(AsPrimExpr(v)));
          return IntTupleScaledBasis(abs_v, BasisPath(d));
        }
        return IsConst(d) ? IntTuple(std::abs(AsConst(d)))
                          : IntTuple(tvm::abs(AsPrimExpr(d)));
      });
  IntTuple co_coord =
      FoldLeaves(m1_shapes * abs_strides, IntTuple(int64_t(0)),
                 [](IntTuple acc, const IntTuple &term) { return acc + term; });
  return TransformLeaf(co_coord, [](const IntTuple &c) {
    return IntTuple(AddLeaf(c, IntTuple(int64_t(1))));
  });
}

IntTuple Cosize(const Layout &layout) {
  // CuTe cosize = size(coshape(layout)).
  return Product(Coshape(layout));
}

namespace {

// ceil_div of two scalar leaves; constant when both are, else a PrimExpr.
IntTuple CeilDivLeaf(const IntTuple &a, const IntTuple &b) {
  if (IsConst(a) && IsConst(b))
    return IntTuple(CeilDiv(AsConst(a), AsConst(b)));
  DataType dt = IsPrimExpr(a) ? AsPrimExpr(a)->dtype : AsPrimExpr(b)->dtype;
  return IntTuple(
      tvm::floordiv(AsConstOrPrimExpr(a, dt) + AsConstOrPrimExpr(b, dt) - 1,
                    AsConstOrPrimExpr(b, dt)));
}

// Static port of CuTe detail::complement (layout.hpp:1176-1227). `shapes`/
// `strides` are the FILTERED flat modes (no stride-0, no size-1); `cotarget` is
// the codomain size to fill. Sort-and-fold: repeatedly take the smallest
// remaining stride, emit the gap below it as a mode, advance past it.
Layout ComplementStatic(std::vector<int64_t> shapes,
                        std::vector<int64_t> strides, int64_t cotarget) {
  int64_t R = static_cast<int64_t>(shapes.size());
  if (R == 0) // Fully-trivial layout: complement is the whole [0, cotarget).
    return Coalesce(Layout(Array<int64_t>{cotarget}, Array<int64_t>{1}));
  Array<int64_t> out_sh, out_st;
  int64_t run = 1; // running stride
  for (int64_t i = 0; i < R - 1; ++i) {
    auto min_it = std::min_element(strides.begin(), strides.end());
    int64_t min_idx = min_it - strides.begin(), min_stride = *min_it;
    int64_t new_shape = min_stride / run;
    ICHECK(new_shape != 0) << "Non-injective layout detected in complement";
    out_sh.push_back(new_shape);
    out_st.push_back(run);
    run = min_stride * shapes[min_idx];
    shapes.erase(shapes.begin() + min_idx);
    strides.erase(strides.begin() + min_idx);
  }
  // Last shape mode, then the rest mode filling up to cotarget.
  int64_t new_shape = strides[0] / run;
  ICHECK(new_shape != 0) << "Non-injective layout detected in complement";
  out_sh.push_back(new_shape);
  out_st.push_back(run);
  int64_t last_stride = strides[0] * shapes[0];
  out_sh.push_back(CeilDiv(cotarget, last_stride));
  out_st.push_back(last_stride);
  return Coalesce(Layout(out_sh, out_st));
}

// CuTe detail::complement for a single filtered rank-1 mode (s):(d),
// specialized to R==1 (fold body runs zero times), with possibly-dynamic
// shape/stride:
//   coalesce( (d, ceil_div(cotarget, d*s)) : (1, d*s) ).
Layout ComplementRank1(const IntTuple &shape, const IntTuple &stride,
                       int64_t cotarget) {
  ICHECK(!IsTuple(stride) && !IsScaledBasis(stride))
      << "Complement requires a scalar stride, got " << stride;
  IntTuple ds = MulLeaf(stride, shape);
  return Coalesce(
      Layout(Array<IntTuple>{stride, CeilDivLeaf(IntTuple(cotarget), ds)},
             Array<IntTuple>{IntTuple(int64_t(1)), ds}));
}

} // namespace

Layout Complement(const Layout &layout, int64_t cotarget) {
  Layout f = Filter(layout); // flat (Coalesce output).
  IntTuple fsh = Flatten(f->shape), fst = Flatten(f->stride);
  // Drop the residual (1):(0) Coalesce/Filter leaves for an empty layout.
  std::vector<IntTuple> sh, st;
  bool all_const = true;
  for (int64_t i = 0, r = Rank(fsh); i < r; ++i) {
    if (IsConst(fsh[i]) && AsConst(fsh[i]) == 1 && IsZeroLeaf(fst[i]))
      continue;
    sh.push_back(fsh[i]);
    st.push_back(fst[i]);
    all_const &= IsConst(fsh[i]) && IsConst(fst[i]);
  }
  int64_t R = static_cast<int64_t>(sh.size());
  if (all_const) {
    std::vector<int64_t> ci_sh, ci_st;
    for (int64_t i = 0; i < R; ++i) {
      ci_sh.push_back(AsConst(sh[i]));
      ci_st.push_back(AsConst(st[i]));
    }
    return ComplementStatic(std::move(ci_sh), std::move(ci_st), cotarget);
  }
  // Dynamic strides can't be stride-ordered, so CuTe only supports rank-1
  // (layout.hpp:1187-1189: static_assert(R == 1 || is_static<Stride>)).
  ICHECK(R == 1) << "Dynamic-stride complement only for rank-1 layouts "
                    "(mirrors CuTe static_assert); got rank "
                 << R;
  return ComplementRank1(sh[0], st[0], cotarget);
}

Layout Complement(const Layout &layout) {
  IntTuple cs = Cosize(Filter(layout));
  ICHECK(IsConst(cs)) << "single-argument Complement needs a static cosize "
                         "(its codomain target), got "
                      << cs;
  return Complement(layout, AsConst(cs));
}

Layout LogicalDivide(const Layout &layout, const Layout &tiler) {
  // CuTe 2-arg logical_divide (layout.hpp:1559-1562):
  //   composition(layout, make_layout(tiler, complement(tiler,
  //   size(coalesce(layout))))).
  // Result is rank-2: mode 0 the tile, mode 1 the rest. The cotarget
  // size(coalesce(layout)) must be static; `layout`'s strides may be dynamic.
  IntTuple sz = Size(Coalesce(layout));
  ICHECK(IsConst(sz)) << "LogicalDivide needs a static layout size (the "
                         "complement cotarget), got "
                      << sz;
  Layout comp = Complement(tiler, AsConst(sz));
  return Composition(layout, MakeLayout({tiler, comp}));
}

Layout LogicalProduct(const Layout &block, const Layout &tiler) {
  // CuTe 2-arg logical_product (layout.hpp:1653-1656):
  //   make_layout(block,
  //               composition(complement(block,
  //                                      size(block) * cosize(tiler)),
  //                           tiler)).
  // Like the C++ CuTe operation, this is the rank-2 primitive.  Higher-level
  // zipped/tiled products only rearrange its profile.
  IntTuple block_size = Size(block);
  IntTuple tiler_cosize = Cosize(tiler);
  ICHECK(IsConst(block_size) && IsConst(tiler_cosize))
      << "LogicalProduct needs a static block size and tiler cosize, got "
      << block_size << " and " << tiler_cosize;
  int64_t cotarget = AsConst(block_size) * AsConst(tiler_cosize);
  Layout rest = Composition(Complement(block, cotarget), tiler);
  return MakeLayout({block, rest});
}

Layout TiledProduct(const Layout &block, const Layout &tiler) {
  // A Layout tiler is atomic to CuTe's zip2_by, so zipped_product is the same
  // rank-2 result as logical_product.  tiled_product unpacks the actual second
  // mode, whose profile need only be compatible with the tiler: complement can
  // split one tiler mode into several residual modes.  Slicing with
  // repeat<R1>(_) leaves a rank-1 second mode untouched (repeat<1>(_) is _),
  // so only rank >= 2 unpacks.
  Layout product = LogicalProduct(block, tiler);
  Layout rest = product[1];
  if (Rank(rest) == 1)
    return product;
  Array<Layout> modes{product[0]};
  for (int64_t i = 0, r = Rank(rest); i < r; ++i) {
    modes.push_back(rest[i]);
  }
  return MakeLayout(modes);
}

Layout BlockedProduct(const Layout &block, const Layout &tiler) {
  // CuTe blocked_product (layout.hpp:1726-1742):
  //   R = max(rank(block), rank(tiler))
  //   zip(get<0>(r), get<1>(r)) with r = logical_product(append<R>(block),
  //                                                      append<R>(tiler)).
  // Mode i of the result is ((block_i, rest_i)), so a flat per-mode coordinate
  // splits into (within-block, block-index) by the ordinary idx2crd rule.
  Layout scalar(1, 0);
  int64_t rb = Rank(block), rt = Rank(tiler);
  int64_t r = std::max(rb, rt);
  Array<Layout> block_modes, tiler_modes;
  for (int64_t i = 0; i < r; ++i) {
    block_modes.push_back(i < rb ? block[i] : scalar);
    tiler_modes.push_back(i < rt ? tiler[i] : scalar);
  }
  Layout product =
      LogicalProduct(MakeLayout(block_modes), MakeLayout(tiler_modes));
  Layout padded = product[0], rest = product[1];
  // Composition against the padded tiler keeps one rest mode per tiler mode.
  ICHECK_EQ(Rank(rest), r) << "BlockedProduct: rest profile " << rest->shape
                           << " does not match the tiler rank " << r;
  Array<Layout> modes;
  for (int64_t i = 0; i < r; ++i) {
    modes.push_back(MakeLayout({padded[i], rest[i]}));
  }
  return MakeLayout(modes);
}

Layout MakeColumnMajorLayout(const IntTuple &shape) {
  return Layout(shape, CompactColMajor(shape));
}

Layout MakeRowMajorLayout(const IntTuple &shape) {
  return Layout(shape, CompactRowMajor(shape));
}

Layout MakeIdentityLayout(const IntTuple &shape) {
  std::vector<int64_t> path;
  return Layout(shape, IdentityStride(shape, path));
}

Layout MakeLayout(const Array<Layout> &layouts) {
  Array<IntTuple> sh, st;
  for (const Layout &l : layouts) {
    sh.push_back(l->shape);
    st.push_back(l->stride);
  }
  return Layout(IntTupleTuple(sh), IntTupleTuple(st));
}

Layout Layout::WithShape(const IntTuple &shape) const {
  return Composition(*this, MakeColumnMajorLayout(shape));
}

Layout Layout::Parse(const ffi::String &text) {
  // The printed spelling "shape:stride"; the shape holds no ':' so the first
  // one splits.
  std::string s(text);
  size_t colon = s.find(':');
  ICHECK_NE(colon, std::string::npos)
      << "Layout text must be 'shape:stride', got '" << s << "'";
  return Layout(IntTuple::Parse(s.substr(0, colon)),
                IntTuple::Parse(s.substr(colon + 1)));
}

void Layout::Print(std::ostream &os) const {
  (*this)->shape.Print(os);
  os << ":";
  (*this)->stride.Print(os);
}

ComposedLayout::ComposedLayout(Swizzle swizzle, int64_t offset, Layout layout) {
  auto node = make_object<ComposedLayoutNode>();
  node->swizzle = std::move(swizzle);
  node->offset = offset;
  node->layout = std::move(layout);
  data_ = std::move(node);
}

namespace {

// CuTe upcast<N>/downcast<N> on one scalar mode (layout.hpp:1806-1855).
// upcast/downcast scale the unit-stride mode's SHAPE, not its stride (uniform
// stride scaling would give a stride-1 mode a fractional stride). `up` selects
// upcast vs downcast; factor==1 is identity. Returns (shape, stride).
std::pair<int64_t, int64_t> RecastMode(int64_t factor, bool up, int64_t shape,
                                       int64_t stride) {
  if (factor == 1 || (up && stride == 0))
    return {shape, stride};
  if (up) {
    int64_t a = std::abs(stride);
    return {CeilDiv(shape, CeilDiv(factor, a)),
            Signum(stride) * CeilDiv(a, factor)};
  }
  if (stride == 1 || stride == -1)
    return {shape * factor, stride};
  return {shape, stride * factor};
}

// Apply RecastMode over a (shape, stride) tree, preserving the hierarchy
// (CuTe upcast/downcast recurse via transform_layout).
std::pair<IntTuple, IntTuple>
RecastTree(int64_t factor, bool up, const IntTuple &sh, const IntTuple &st) {
  if (IsTuple(sh)) {
    ICHECK(IsTuple(st) && Rank(sh) == Rank(st))
        << "Recast: shape/stride tree mismatch";
    Array<IntTuple> nsh, nst;
    for (int64_t i = 0, n = Rank(sh); i < n; ++i) {
      auto [csh, cst] = RecastTree(factor, up, sh[i], st[i]);
      nsh.push_back(csh);
      nst.push_back(cst);
    }
    return {IntTupleTuple(nsh), IntTupleTuple(nst)};
  }
  ICHECK(IsConst(sh) && IsConst(st)) << "Recast requires plain integer leaves";
  auto [os, od] = RecastMode(factor, up, AsConst(sh), AsConst(st));
  return {IntTuple(os), IntTuple(od)};
}

} // namespace

ComposedLayout ComposedLayout::Recast(int old_bits, int new_bits) const {
  const ComposedLayoutNode *n = get();
  ICHECK(n != nullptr) << "Recast on a null ComposedLayout";
  int old_log = Log2Exact(old_bits);
  int new_log = Log2Exact(new_bits);
  ICHECK(old_log >= 0 && new_log >= 0)
      << "Recast expects power-of-two element sizes, got " << old_bits << " -> "
      << new_bits;
  // Recasting old_bits -> new_bits is CuTe recast_layout: new>old =>
  // upcast<new/old>, new<old => downcast<old/new>. See RecastMode for why the
  // unit-stride mode scales its shape. The offset is a plain address.
  bool up = new_bits >= old_bits;
  int64_t factor = up ? (int64_t(1) << (new_log - old_log))
                      : (int64_t(1) << (old_log - new_log));
  auto [out_sh, out_st] =
      RecastTree(factor, up, n->layout->shape, n->layout->stride);
  int64_t new_offset;
  if (factor == 1) {
    new_offset = n->offset;
  } else if (up) {
    ICHECK(n->offset % factor == 0)
        << "Recast cannot scale offset " << n->offset << " up by " << factor;
    new_offset = n->offset / factor;
  } else {
    new_offset = n->offset * factor;
  }
  return ComposedLayout(n->swizzle.Recast(old_bits, new_bits), new_offset,
                        Layout(out_sh, out_st));
}

namespace {

// When processing shared tensor layout, all the shapes and strides are int32_t.
// Also, we need to make sure not to mix with int64_t, because otherwise there
// would be a lot of conversion nodes that prevent the analyzer from cancelling
// out the terms.
PrimExpr MakeI32(int x) { return IntImm(DataType::Int(32), x); }

// Probes a TileLang layout's row-major linearized physical address A(x).
// The constant input shape, output strides, forward-index expressions, and
// input placeholders are parsed once in the constructor. operator() folds
// concrete integer coordinates to a constant address; Symbolic() substitutes
// fresh variables for the equivalence proof. valid() is false when any extent
// is non-constant.
class AddrProbe {
public:
  explicit AddrProbe(const tvm::tl::Layout &layout) {
    ICHECK(layout.defined());
    for (const auto &e : layout->InputShape()) {
      auto c = as_const_int(e);
      ICHECK(c) << "InputShape extent " << e
                << " of a shared tensor layout must be constant";
      shape_.push_back(*c);
    }
    std::vector<int32_t> out_sizes;
    for (const auto &e : layout->OutputShape()) {
      auto c = as_const_int(e);
      ICHECK(c) << "OutputShape extent " << e
                << " of a shared tensor layout must be constant";
      out_sizes.push_back(*c);
    }
    out_strides_.assign(out_sizes.size(), 0);
    int32_t acc = 1;
    for (int64_t d = static_cast<int64_t>(out_sizes.size()) - 1; d >= 0; --d) {
      out_strides_[d] = acc;
      acc *= out_sizes[d];
    }
    forward_index_ = layout->GetForwardIndex();
    ICHECK_EQ(forward_index_.size(), out_strides_.size());
    ICHECK_EQ(layout->InputDim(), shape_.size());
    for (size_t k = 0; k < shape_.size(); ++k)
      placeholders_.push_back(InputPlaceholder(k));
  }

  const std::vector<int32_t> &shape() const { return shape_; }

  // Concrete address A(coords); nullopt if it does not fold to a constant.
  std::optional<int32_t> operator()(const std::vector<int32_t> &coords) const {
    Map<Var, PrimExpr> vmap;
    for (size_t k = 0; k < coords.size(); ++k)
      vmap.Set(placeholders_[k], MakeI32(coords[k]));
    int32_t addr = 0;
    for (size_t d = 0; d < forward_index_.size(); ++d) {
      PrimExpr e = Substitute(forward_index_[d], vmap);
      std::optional<int64_t> val = EvaluateConstantInteger(e);
      if (!val)
        return std::nullopt;
      addr += static_cast<int32_t>(*val) * out_strides_[d];
    }
    return addr;
  }

  // Symbolic address A(vars). Built in int32 (the forward index's native
  // dtype): int64 casts in the spine are opaque to the simplifier's div/mod
  // rules and would defeat the equivalence proof.
  PrimExpr Symbolic(const std::vector<Var> &vars) const {
    Map<Var, PrimExpr> vmap;
    for (size_t k = 0; k < vars.size(); ++k)
      vmap.Set(placeholders_[k], vars[k]);
    PrimExpr a = MakeI32(0);
    for (size_t d = 0; d < forward_index_.size(); ++d) {
      a = a + Substitute(forward_index_[d], vmap) * MakeI32(out_strides_[d]);
    }
    return a;
  }

private:
  std::vector<int32_t> shape_;
  std::vector<int32_t> out_strides_;
  Array<PrimExpr> forward_index_;
  std::vector<Var> placeholders_;
};

PrimExpr Bit(const PrimExpr &e, int k) {
  PrimExpr shifted = k > 0 ? FloorDiv(e, IntImm(e->dtype, int64_t(1) << k)) : e;
  return FloorMod(shifted, IntImm(e->dtype, 2));
}

// Rewrite the bitwise builtins a swizzle is built from into plain integer
// arithmetic (+, -, *, FloorDiv, FloorMod) that the arith machinery can reason
// about. Each rewrite is an exact identity:
//   - shift_left(x, k)  == x * 2^k                       (k a constant)
//   - shift_right(x, k) == FloorDiv(x, 2^k)              (k a constant, x >= 0)
//   - bitwise_and(x, C) == sum_{bit k of C set} bit_k(x) * 2^k   (C constant)
//   - bitwise_xor(x, y) == hi(wide) + sum_{k<w} ((bit_k(x)+bit_k(y))%2)*2^k
// The xor expansion relies on single-bit parities not carrying, so it expands
// only the low `w` bits, where w is the bit width of the NARROWER operand (the
// one whose value is bounded small). Bits at or above w are zero in the narrow
// operand, so there x^y == bit-of-wide unchanged; their sum is kept as the
// single symbolic term hi(wide) = FloorDiv(wide, 2^w) * 2^w. This is what lets
// a swizzle applied over an address carrying a large base offset reduce: the
// offset stays intact in hi(wide) instead of being shattered into ~log2(offset)
// parity terms that the prover cannot reconverge. Every other node recurses
// through ExprMutator's default child-rebuilding visitors.
class BitOpLowerer : public ExprMutator {
public:
  explicit BitOpLowerer(arith::Analyzer *ana) : ana_(ana) {}
  using ExprMutator::operator();

protected:
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->args.size() == 2 && op->op.same_as(builtin::shift_left())) {
      if (auto k = as_const_int(op->args[1])) {
        ICHECK(*k >= 0 && *k < 64);
        return VisitExpr(op->args[0]) * IntImm(op->dtype, int64_t(1) << *k);
      }
    }
    if (op->args.size() == 2 && op->op.same_as(builtin::shift_right())) {
      if (auto k = as_const_int(op->args[1])) {
        ICHECK(*k >= 0 && *k < 64);
        return FloorDiv(VisitExpr(op->args[0]),
                        IntImm(op->dtype, int64_t(1) << *k));
      }
    }
    if (op->args.size() == 2 && op->op.same_as(builtin::bitwise_and())) {
      auto ca = as_const_int(op->args[0]), cb = as_const_int(op->args[1]);
      if (ca || cb) {
        int64_t mask = ca ? *ca : *cb;
        PrimExpr x = VisitExpr(ca ? op->args[1] : op->args[0]);
        if (mask >= 0) {
          PrimExpr out = MakeI32(0);
          for (int k = 0; (mask >> k) != 0; ++k)
            if ((mask >> k) & 1)
              out = out + Bit(x, k) * MakeI32(int64_t(1) << k);
          return out;
        }
      }
    }
    if (op->args.size() == 2 && op->op.same_as(builtin::bitwise_xor())) {
      PrimExpr x = VisitExpr(op->args[0]), y = VisitExpr(op->args[1]);
      // Expand over the NARROWER operand's bit width: bits above it are zero
      // there, so x^y == the wider operand unchanged in that range. Bounding by
      // the min keeps a large base offset on the wide side out of the per-bit
      // expansion (it rides along in hi() below) instead of exploding into
      // ~log2(offset) parity terms the prover cannot reconverge.
      int64_t bx = ana_->const_int_bound(x)->max_value;
      int64_t by = ana_->const_int_bound(y)->max_value;
      if (bx < 0 || by < 0)
        return bitwise_xor(x, y); // unbounded operand: cannot size the window.
      int64_t mn = std::min(bx, by);
      if (mn < (int64_t(1) << 20)) {
        int width = std::max<int>(
            1, 64 - CountLeadingZeros(static_cast<uint64_t>(mn) | 1));
        // wide = the operand whose high bits (>= width) pass through unchanged.
        PrimExpr wide = bx >= by ? x : y;
        PrimExpr hi = FloorDiv(wide, MakeI32(int64_t(1) << width)) *
                      MakeI32(int64_t(1) << width);
        PrimExpr out = hi;
        for (int k = 0; k < width; ++k)
          out = out + FloorMod(Bit(x, k) + Bit(y, k), MakeI32(2)) *
                          MakeI32(int64_t(1) << k);
        return out;
      }
      return bitwise_xor(x, y);
    }
    return ExprMutator::VisitExpr_(op);
  }

private:
  arith::Analyzer *ana_;
};

PrimExpr LowerBitOps(const PrimExpr &e, arith::Analyzer *ana) {
  return BitOpLowerer(ana)(e);
}

// Canonicalize every parity node FloorMod(x, 2) in `e`. Working mod 2:
// +/- coincide, even-coefficient terms vanish, and a nested (y % 2) inside a
// parity sum equals y — so the parity's terms flatten into a bag. Each leaf
// term is keyed by Simplify(term % 2) (the simplifier's canonical single-bit
// form, e.g. (i // 2) % 2 == i % 4 // 2), keys appearing an even number of
// times cancel, and the survivors are rebuilt in sorted order. Two equal
// parities thus become structurally identical, which lets the surrounding
// Simplify cancel them — the step the rewrite simplifier cannot do by itself.
// Only FloorMod(_, 2) needs custom handling; every other node recurses through
// ExprMutator's default child-rebuilding visitors.
class ParityCanonicalizer : public ExprMutator {
public:
  explicit ParityCanonicalizer(arith::Analyzer *ana) : ana_(ana) {}
  using ExprMutator::operator();

protected:
  PrimExpr VisitExpr_(const FloorModNode *op) final {
    if (!is_two(op->b))
      return ExprMutator::VisitExpr_(op);
    std::vector<PrimExpr> terms;
    int parity = 0;
    flatten(op->a, &terms, &parity);
    std::stable_sort(terms.begin(), terms.end(),
                     [](const PrimExpr &a, const PrimExpr &b) {
                       return StructuralHash()(a) < StructuralHash()(b);
                     });
    PrimExpr sum = MakeI32(parity);
    for (size_t i = 0; i < terms.size();) {
      if (i + 1 < terms.size() && StructuralEqual()(terms[i], terms[i + 1])) {
        i += 2; // t + t == 0 (mod 2)
      } else {
        sum = sum + terms[i];
        ++i;
      }
    }
    return FloorMod(sum, MakeI32(2));
  }

private:
  static bool is_two(const PrimExpr &x) {
    auto c = as_const_int(x);
    return c && *c == 2;
  }

  // Flatten `x` (interpreted mod 2) into leaf terms + a constant parity.
  void flatten(const PrimExpr &x, std::vector<PrimExpr> *terms, int *parity) {
    if (const auto *op = x.as<AddNode>()) {
      flatten(op->a, terms, parity);
      flatten(op->b, terms, parity);
      return;
    }
    if (const auto *op = x.as<SubNode>()) { // -t == t (mod 2)
      flatten(op->a, terms, parity);
      flatten(op->b, terms, parity);
      return;
    }
    if (const auto *imm = x.as<IntImmNode>()) {
      *parity ^= static_cast<int>(imm->value & 1);
      return;
    }
    if (const auto *op = x.as<MulNode>()) {
      const auto *ca = op->a.as<IntImmNode>();
      const auto *cb = op->b.as<IntImmNode>();
      if (ca || cb) {
        int64_t c = ca ? ca->value : cb->value;
        if (c % 2 == 0)
          return; // even coefficient: vanishes mod 2.
        flatten(ca ? op->b : op->a, terms, parity);
        return;
      }
    }
    if (const auto *op = x.as<FloorModNode>()) {
      if (is_two(op->b)) { // (y % 2) == y (mod 2)
        flatten(op->a, terms, parity);
        return;
      }
    }
    // Leaf: canonicalize via the simplifier's single-bit form (recursing into
    // any nested parities first). If that form is itself a parity of a sum,
    // keep flattening through it.
    PrimExpr key = ana_->Simplify(FloorMod(VisitExpr(x), 2));
    if (const auto *km = key.as<FloorModNode>()) {
      if (is_two(km->b) && (km->a.as<AddNode>() || km->a.as<SubNode>() ||
                            km->a.as<MulNode>() || km->a.as<IntImmNode>())) {
        flatten(km->a, terms, parity);
        return;
      }
    }
    if (const auto *imm = key.as<IntImmNode>()) {
      *parity ^= static_cast<int>(imm->value & 1);
      return;
    }
    terms->push_back(key);
  }

  arith::Analyzer *ana_;
};

PrimExpr ParityCanon(const PrimExpr &e, arith::Analyzer *ana) {
  return ParityCanonicalizer(ana)(e);
}

// Decide E == 0 for all assignments of the analyzer's range-bound variables,
// by iterating Simplify -> LowerBitOps -> ParityCanon to a fixpoint. Sound:
// every step is an exact identity, so a 0 result is a proof; a non-0 fixpoint
// means "not proven" and the caller must reject. Complete enough in practice
// that neither sampling nor an SMT solver is needed.
bool ProveZero(PrimExpr e, arith::Analyzer *ana, int max_rounds = 8) {
  PrimExpr d = ana->Simplify(e);
  size_t last_hash = 0;
  for (int round = 0; round < max_rounds; ++round) {
    if (is_zero(d))
      return true;
    size_t h = StructuralHash()(d);
    if (round > 0 && h == last_hash)
      return false; // fixpoint reached without proving 0.
    last_hash = h;
    d = ana->Simplify(ParityCanon(LowerBitOps(d, ana), ana));
  }
  return is_zero(d);
}

// Symbolically prove that the probed layout address equals the recovered
// composed layout, A(x) == Sw(offset + plain(linearize(x))), for ALL logical
// coordinates x. With mid(Q) = the swizzle's XOR of Q's target and source bit
// fields written as per-bit parities, and tgt(Q) = Q's target field, the
// equality is the single integer identity
//
//   A - Q - (mid(Q) - tgt(Q)) * 2^m_base == 0,
//
// which ProveZero decides with the parity-aware rewriting. All proof
// arithmetic is int32; layouts that could overflow it are rejected.
bool ProveEquivalent(const AddrProbe &probe, const Swizzle &swizzle,
                     int64_t offset, const Layout &plain) {
  const Layout cl = Flatten(plain);
  const std::vector<int32_t> &shape = probe.shape();
  // RecoverPlainLayout always builds `plain` flat with constant integer leaves.
  std::vector<int32_t> cl_shape, cl_stride;
  for (int64_t k = 0; k < Rank(cl->shape); ++k)
    cl_shape.push_back(AsConst(cl->shape[k]));
  for (int64_t k = 0; k < Rank(cl->stride); ++k)
    cl_stride.push_back(AsConst(cl->stride[k]));

  int b = swizzle->b_bits, m = swizzle->m_base, s = swizzle->s_shift;
  if (b > 0 && m + s + b >= 31)
    return false;

  // Analyzer with each coordinate range-bound: x_k in [0, shape[k]).
  arith::Analyzer ana;
  std::vector<Var> vars;
  for (size_t k = 0; k < shape.size(); ++k) {
    Var v("c" + std::to_string(k), DataType::Int(32));
    vars.push_back(v);
    if (shape[k] > 0)
      ana.Bind(v, Range(MakeI32(0), MakeI32(shape[k])));
  }

  PrimExpr A = probe.Symbolic(vars);

  // Q(x) = offset + plain(linearize(x)): linearize x column-major over the
  // input shape, then evaluate the plain layout's CuTe idx2crd symbolically.
  PrimExpr coord = MakeI32(0);
  int64_t place = 1;
  for (size_t k = 0; k < vars.size(); ++k) {
    coord = coord + vars[k] * MakeI32(place);
    place *= std::max<int64_t>(shape[k], 1);
  }
  PrimExpr Q = MakeI32(offset);
  {
    PrimExpr rem = coord;
    for (size_t i = 0; i < cl_shape.size(); ++i) {
      int64_t ext = cl_shape[i];
      PrimExpr crd = ext > 0 ? FloorMod(rem, MakeI32(ext)) : PrimExpr(rem);
      Q = Q + crd * MakeI32(cl_stride[i]);
      if (ext > 0)
        rem = FloorDiv(rem, MakeI32(ext));
    }
  }

  if (b <= 0)
    return ProveZero(A - Q, &ana);

  PrimExpr mid = MakeI32(0);
  for (int p = 0; p < b; ++p)
    mid = mid + FloorMod(Bit(Q, m + p) + Bit(Q, m + s + p), MakeI32(2)) *
                    MakeI32(int64_t(1) << p);
  PrimExpr tgt =
      FloorMod(FloorDiv(Q, MakeI32(int64_t(1) << m)), MakeI32(int64_t(1) << b));
  return ProveZero(A - Q - (mid - tgt) * MakeI32(int64_t(1) << m), &ana);
}

} // namespace

namespace {

// Recover the unswizzled affine cute::Layout underlying a probed address, given
// the swizzle and base offset, returning nullopt unless the recovery provably
// reproduces the probe for every coordinate. Shared by the two entry points
// below (one detects the swizzle first, the other assumes none).
//
// The swizzle is an involution, so undoing it gives the plain address
// P(x) = Sw(A(x)) - offset. Probing P at one-hot coordinates reads off the
// strides: each input dim is broken into one mode per power-of-two bit plus a
// non-power-of-2 remainder, which captures a dim even when it is split across
// non-contiguous regions of the address space. Each mode's stride is P at the
// coordinate that activates only that mode.
Optional<Layout> RecoverPlainLayout(const AddrProbe &A, const Swizzle &swizzle,
                                    int64_t offset) {
  const std::vector<int32_t> &shape = A.shape();
  int64_t n = shape.size();
  auto P = [&](const std::vector<int32_t> &x) -> std::optional<int64_t> {
    std::optional<int32_t> a = A(x);
    if (!a)
      return std::nullopt;
    return swizzle->Apply(*a) - offset;
  };
  Array<int64_t> mode_shape, mode_stride;
  for (int64_t k = 0; k < n; ++k) {
    int64_t rem = shape[k];
    if (rem <= 1) {
      mode_shape.push_back(1); // singleton dim: extent-1 mode, stride 0.
      mode_stride.push_back(0);
      continue;
    }
    // One mode per power-of-two bit, then a non-power-of-2 remainder. Modes are
    // emitted dim-by-dim in increasing place order, so the recovered layout's
    // single-coordinate domain is the column-major linearization of `shape`.
    for (int32_t place = 1; rem > 1;) {
      int64_t ext = rem % 2 == 0 ? 2 : rem;
      std::vector<int32_t> x(n, 0);
      x[k] = place;
      std::optional<int64_t> d = P(x);
      if (!d)
        return std::nullopt;
      // A stride-0 mode is a valid broadcast (e.g. (8,8):(0,1)); the final
      // ProveEquivalent check, not injectivity, is what gates correctness.
      mode_shape.push_back(ext);
      mode_stride.push_back(*d);
      rem /= ext;
      place *= ext;
    }
  }
  Layout plain = Coalesce(Layout(mode_shape, mode_stride));

  // The per-mode probe only samples one-hot coordinates; confirm the recovered
  // layout matches the probe everywhere before trusting it.
  if (!ProveEquivalent(A, swizzle, offset, plain))
    return std::nullopt;
  return plain;
}

// Build an IntTuple congruent to a flat TileLang input shape.
IntTuple ShapeTuple(const std::vector<int32_t> &input_shape) {
  Array<IntTuple> dims;
  for (int32_t s : input_shape)
    dims.push_back(IntTuple(static_cast<int64_t>(s)));
  return IntTupleTuple(dims);
}

} // namespace

// Recover a TileLang layout that has no swizzle and zero offset as a plain
// cute::Layout, returning nullopt if it is actually swizzled/offset (the
// equivalence proof then fails). Same recovery as the swizzled case with the
// detection step skipped.
Optional<Layout> LayoutFromTileLang(const tvm::tl::Layout &layout) {
  AddrProbe A(layout);
  if (A.shape().empty())
    return std::nullopt;
  Optional<Layout> plain =
      RecoverPlainLayout(A, Swizzle::Identity(), /*offset=*/0);
  if (!plain.defined())
    return std::nullopt;
  // Return a layout congruent to the input TileLang shape (not the flat form).
  return plain.value().WithShape(ShapeTuple(A.shape()));
}

// Recover a multi-output TileLang layout while keeping its codomain hierarchy:
// each output coordinate axis is recovered as its own plain affine layout and
// rerouted onto the basis axis E<i>, so the result maps a logical coordinate to
// the full output coordinate tuple (the inverse of Python's
// Layout.to_tilelang). LayoutFromTileLang instead serializes all output axes
// into one flat address.
Optional<Layout> LayoutFromTileLangHierarchical(const tvm::tl::Layout &layout) {
  ICHECK(layout.defined());
  Array<PrimExpr> input_shape = layout->InputShape();
  Array<PrimExpr> forward_index = layout->GetForwardIndex();
  int64_t rank = static_cast<int64_t>(forward_index.size());
  if (rank == 0)
    return std::nullopt;

  // Recover each output axis independently as a plain affine layout.
  std::vector<Layout> projected;
  projected.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    Optional<Layout> axis =
        LayoutFromTileLang(tvm::tl::Layout(input_shape, {forward_index[i]}));
    if (!axis.defined())
      return std::nullopt;
    projected.push_back(axis.value());
  }

  // Conform all axes to one common shape profile. WithShape is an exact
  // composition, so refining a profile never changes the function. Threading
  // the progressively refined profile forward hands the last axis every
  // axis's split points; conforming each earlier axis to that finest profile
  // then makes all shapes identical.
  std::vector<Layout> scaled(projected);
  for (int64_t i = 1; i < rank; ++i)
    scaled[i] = projected[i].WithShape(scaled[i - 1]->shape);
  for (int64_t i = 0; i + 1 < rank; ++i)
    scaled[i] = projected[i].WithShape(scaled[rank - 1]->shape);
  const IntTuple shape = scaled[rank - 1]->shape;
  for (int64_t i = 0; i + 1 < rank; ++i)
    ICHECK(StructuralEqual()(scaled[i]->shape, shape))
        << "Hierarchical recovery produced incompatible axis profiles: "
        << scaled[i]->shape << " vs " << shape;

  // Reroute each axis onto its basis and sum, keeping broadcast (stride-0)
  // leaves the plain additive identity -- official operator* would tag them
  // 0@i, and 0@i + 0@j sums to a coordinate tuple, breaking shape/stride
  // congruence for leaves no axis drives. A leaf driven by exactly one axis
  // keeps a single ScaledBasis stride; a leaf driven by two axes sums to a
  // coordinate tuple instead, breaking congruence -- the output axes are
  // dependent and no cute::Layout represents that.
  IntTuple stride = 0;
  for (int64_t i = 0; i < rank; ++i) {
    stride += TransformLeaf(scaled[i]->stride, [&](const IntTuple &leaf) {
      return IsZeroLeaf(leaf) ? leaf : IntTuple(leaf * E({i}));
    });
  }
  if (!Congruent(shape, stride))
    return std::nullopt;
  return Layout(shape, stride);
}

// Recover a swizzled affine layout, A(x) = Sw(offset + plain(x)), from a
// TileLang layout, as a Swizzle composed with a plain cute::Layout. Sizes and
// strides need not be powers of two.
//
// First detect the XOR swizzle by GF(2) bit-incidence. Probing the address at
// each power-of-two bit-atom of a power-of-two-sized input dim and XOR-ing
// against A(0) isolates that bit's image column. An ordinary (identity) bit
// gives a single-bit column; a swizzled bit gives a two-bit column {lo,
// lo+s_shift} whose low bit lo also appears as some identity column -- that
// cross-check distinguishes a real swizzle from a two-bit column that a
// non-power-of-2 plain stride happens to produce. The swizzle parameters are
// read off the lowest contiguous run of such columns that share one s_shift.
//
// Then recover the plain layout and prove equivalence via RecoverPlainLayout.
Optional<ComposedLayout>
ComposedLayoutFromTileLang(const tvm::tl::Layout &layout) {
  AddrProbe A(layout);
  if (A.shape().empty())
    return std::nullopt;
  const std::vector<int32_t> &shape = A.shape();
  int64_t n = shape.size();

  std::vector<int32_t> zero(n, 0);
  std::optional<int32_t> A0 = A(zero);
  if (!A0)
    return std::nullopt;

  // Collect the image column of every power-of-two bit-atom. Probe bit p of dim
  // k (one-hot 1 << p) for p < ctz(shape[k]): only then does 2^(p+1) divide the
  // extent, so bit p toggles cleanly within the dim and the one-hot isolates a
  // single mode. A non-pow2 extent still contributes its low bits (192 = 64*3
  // -> p < 6); an odd extent contributes none.
  std::vector<uint64_t> cols;
  uint64_t weight1 = 0; // bit positions that are some atom's identity image.
  for (int64_t k = 0; k < n; ++k) {
    ICHECK_GT(shape[k], 0); // ctz is undefined at 0; extents are >= 1.
    int max_p = CountTrailingZeros(static_cast<uint32_t>(shape[k]));
    for (int p = 0; p < max_p; ++p) {
      std::vector<int32_t> x(n, 0);
      x[k] = int32_t(1) << p;
      std::optional<int32_t> a = A(x);
      if (!a)
        return std::nullopt;
      uint64_t col = static_cast<uint64_t>(*a ^ *A0);
      cols.push_back(col);
      if (CountSetBits(col) == 1)
        weight1 |= col;
    }
  }
  std::map<int, int> s_at; // swizzle target bit -> s_shift.
  for (uint64_t col : cols) {
    if (CountSetBits(col) != 2)
      continue;
    int lo = CountTrailingZeros(col);
    int hi = 63 - CountLeadingZeros(col);
    if (!(weight1 & (uint64_t(1) << lo)))
      continue; // two-bit column from a plain stride, not a swizzle.
    if (s_at.count(lo))
      return std::nullopt; // two sources hitting one target: not a Swizzle.
    s_at.emplace(lo, hi - lo);
  }
  Swizzle swizzle = Swizzle::Identity();
  if (!s_at.empty()) {
    // Take the lowest contiguous run of target bits that share one s_shift.
    int m_base = s_at.begin()->first;
    int s_shift = s_at.begin()->second;
    int b_bits = 0;
    for (auto it = s_at.find(m_base + b_bits);
         it != s_at.end() && it->second == s_shift;
         it = s_at.find(m_base + b_bits))
      ++b_bits;
    if (s_shift < b_bits)
      return std::nullopt; // source and target bit regions must not overlap.
    swizzle = Swizzle(b_bits, m_base, s_shift);
  }

  // The base offset is Sw(A(0)), since Sw is an involution; recover the plain
  // layout under that swizzle and offset.
  int64_t offset = swizzle->Apply(*A0);
  Optional<Layout> plain = RecoverPlainLayout(A, swizzle, offset);
  if (!plain.defined())
    return std::nullopt;
  // Return a layout congruent to the input TileLang shape (not the flat form),
  // so callers can index modes positionally (e.g. logical_divide per mode).
  return ComposedLayout(swizzle, offset,
                        plain.value().WithShape(ShapeTuple(A.shape())));
}

// Restrict an affine cute::Layout to the sub-tile selected by `range` (one
// Range per top-level mode, e.g. a BufferRegion's per-axis ranges). The layout
// may be hierarchical (a swizzle-split decoded layout): each mode is reshaped
// to its logical extent via CuTe `with_shape`, which flattens the split sub-
// modes back to the logical slice. Because the layout is affine, evaluating it
// at the region origin `mins` gives a single base offset, so:
//   layout(mins + c) == layout(mins) + sublayout(c)  for all c in the sub-tile.
// Returns (offset, sublayout). `mins` may be dynamic (e.g. a sliced operand
// `B[:, j*64:...]` inside a loop), so the offset is an IntTuple, not an int.
Tuple<IntTuple, Layout> Restrict(const Layout &layout,
                                 const Array<Range> &range) {
  int64_t r = Rank(layout->shape);
  ICHECK_EQ(static_cast<int64_t>(range.size()), r)
      << "Restrict: region rank " << range.size() << " != layout rank " << r;

  Array<IntTuple> min_modes;
  Array<Layout> sub_modes;
  for (int64_t i = 0; i < r; ++i) {
    min_modes.push_back(range[i]->min);
    // Skip statically-extent-1 modes (e.g. a pipeline stage pinned to one
    // element): they'd leave size-1 modes that break the GMMA canonical divide.
    const auto *ext_imm = range[i]->extent.as<IntImmNode>();
    if (ext_imm && ext_imm->value == 1)
      continue;
    sub_modes.push_back(layout[i].WithShape(range[i]->extent));
  }
  IntTuple offset = layout(IntTupleTuple(min_modes));
  if (sub_modes.empty()) // everything pinned to one element: scalar (1):(0).
    return Tuple<IntTuple, Layout>(offset, Layout(1, 0));
  return Tuple<IntTuple, Layout>(offset, MakeLayout(sub_modes));
}

namespace {

// Stream any Print-able object into a String (for the __ffi_repr__ hooks).
template <class T> ffi::String PrintToString(const T &value) {
  std::ostringstream os;
  value.Print(os);
  return os.str();
}

std::string FormatComposedLayout(const ComposedLayoutNode *c) {
  std::ostringstream os;
  c->swizzle.Print(os);
  os << " o " << c->offset << " o ";
  c->layout.Print(os);
  return os.str();
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  SwizzleNode::RegisterReflection();
  IntTupleConstNode::RegisterReflection();
  IntTuplePrimExprNode::RegisterReflection();
  IntTupleScaledBasisNode::RegisterReflection();
  IntTupleTupleNode::RegisterReflection();
  LayoutNode::RegisterReflection();
  ComposedLayoutNode::RegisterReflection();
}

// CuTe-notation printing: register an __ffi_repr__ hook per concrete type so
// ffi.ReprPrint (and therefore Python's str()/repr()) renders these objects in
// CuTe spelling. The hook is keyed by exact type index (no ancestor lookup), so
// every concrete IntTuple kind is registered individually.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::TypeAttrDef<SwizzleNode>().def(
      refl::type_attr::kRepr,
      [](Swizzle s, ffi::Function) -> ffi::String { return PrintToString(s); });
  refl::TypeAttrDef<IntTupleConstNode>().def(
      refl::type_attr::kRepr, [](IntTuple t, ffi::Function) -> ffi::String {
        return PrintToString(t);
      });
  refl::TypeAttrDef<IntTuplePrimExprNode>().def(
      refl::type_attr::kRepr, [](IntTuple t, ffi::Function) -> ffi::String {
        return PrintToString(t);
      });
  refl::TypeAttrDef<IntTupleScaledBasisNode>().def(
      refl::type_attr::kRepr, [](IntTuple t, ffi::Function) -> ffi::String {
        return PrintToString(t);
      });
  refl::TypeAttrDef<IntTupleTupleNode>().def(
      refl::type_attr::kRepr, [](IntTuple t, ffi::Function) -> ffi::String {
        return PrintToString(t);
      });
  refl::TypeAttrDef<LayoutNode>().def(
      refl::type_attr::kRepr,
      [](Layout l, ffi::Function) -> ffi::String { return PrintToString(l); });
  refl::TypeAttrDef<ComposedLayoutNode>().def(
      refl::type_attr::kRepr,
      [](ComposedLayout c, ffi::Function) -> ffi::String {
        return FormatComposedLayout(c.get());
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // All CuTe-layout FFI lives under the `tl.cute.*` namespace so the Python
  // side initializes a single module (tilelang.layout.cute). The Python layer
  // reads a layout's hierarchical shape/stride as IntTuple objects
  // (layout_shape / layout_stride) and walks them itself, mirroring CuTeDSL's
  // tuple-valued Layout.shape / Layout.stride -- there is no flat-leaf accessor
  // here.
  refl::GlobalDef()
      // -- Swizzle --------------------------------------------------------
      .def("tl.cute.swizzle_is_swizzled",
           [](const Swizzle &swizzle) { return swizzle->IsSwizzled(); })
      .def("tl.cute.swizzle_to_swizzle_mode",
           [](const Swizzle &swizzle) { return swizzle->ToSwizzleMode(); })
      .def("tl.cute.swizzle_recast",
           [](const Swizzle &swizzle, int old_bits, int new_bits) {
             return swizzle.Recast(old_bits, new_bits);
           })
      .def("tl.cute.swizzle_parse",
           [](const ffi::String &text) { return Swizzle::Parse(text); })
      // -- IntTuple leaf builders ----------------------------------------
      .def("tl.cute.make_int_const",
           [](int64_t value) { return IntTuple(value); })
      .def("tl.cute.make_int_expr",
           [](PrimExpr value) { return IntTuple(std::move(value)); })
      .def("tl.cute.make_scaled_basis",
           [](IntTuple value, Array<int64_t> basis) {
             return IntTupleScaledBasis(std::move(value), std::move(basis));
           })
      // Build an IntTupleTuple branch from already-built children (for
      // hierarchical shape/stride trees).
      .def("tl.cute.make_int_tuple_tuple",
           [](Array<IntTuple> fields) {
             return IntTuple(IntTupleTuple(std::move(fields)));
           })
      // -- IntTuple arithmetic -------------------------------------------
      .def("tl.cute.int_tuple_add",
           [](const IntTuple &a, const IntTuple &b) { return a + b; })
      .def("tl.cute.int_tuple_mul",
           [](const IntTuple &a, const IntTuple &b) { return a * b; })
      .def("tl.cute.int_tuple_parse",
           [](const ffi::String &text) { return IntTuple::Parse(text); })
      .def("tl.cute.product", [](const IntTuple &t) { return Product(t); })
      // -- Layout constructor + accessors --------------------------------
      // Build a (possibly hierarchical / dynamic) layout from congruent
      // shape/stride IntTuple trees.
      .def("tl.cute.make_layout",
           [](IntTuple shape, IntTuple stride) {
             return Layout(std::move(shape), std::move(stride));
           })
      .def("tl.cute.make_layout_concat",
           [](const Array<Layout> &layouts) { return MakeLayout(layouts); })
      .def("tl.cute.with_shape",
           [](const Layout &layout, const IntTuple &shape) {
             return layout.WithShape(shape);
           })
      .def("tl.cute.layout_get",
           [](const Layout &layout, int64_t index) { return layout[index]; })
      // coord is an IntTuple: a scalar leaf for a single linear index, or a
      // hierarchical coordinate (CuTe crd2idx).
      .def("tl.cute.layout_eval",
           [](const Layout &layout, const IntTuple &coord) {
             return layout(coord);
           })
      .def("tl.cute.layout_parse",
           [](const ffi::String &text) { return Layout::Parse(text); })
      .def("tl.cute.layout_shape",
           [](const Layout &layout) { return layout->shape; })
      .def("tl.cute.layout_stride",
           [](const Layout &layout) { return layout->stride; })
      // -- Layout algebra (header order) ---------------------------------
      .def("tl.cute.layout_rank",
           [](const Layout &layout) { return Rank(layout); })
      .def("tl.cute.flatten",
           [](const Layout &layout) { return Flatten(layout); })
      .def("tl.cute.layout_size",
           [](const Layout &layout) { return Size(layout); })
      .def("tl.cute.coalesce",
           [](const Layout &layout) { return Coalesce(layout); })
      .def("tl.cute.right_inverse",
           [](const Layout &layout) { return RightInverse(layout); })
      .def("tl.cute.left_inverse",
           [](const Layout &layout) { return LeftInverse(layout); })
      .def("tl.cute.composition",
           [](const Layout &lhs, const Layout &rhs) {
             return Composition(lhs, rhs);
           })
      .def("tl.cute.coalesce_max",
           [](const Layout &layout, int64_t max_extent) {
             return Coalesce(layout, max_extent);
           })
      .def("tl.cute.filter",
           [](const Layout &layout) { return Filter(layout); })
      .def("tl.cute.congruent",
           [](const IntTuple &a, const IntTuple &b) { return Congruent(a, b); })
      .def(
          "tl.cute.compatible",
          [](const IntTuple &a, const IntTuple &b) { return Compatible(a, b); })
      .def("tl.cute.coshape",
           [](const Layout &layout) { return Coshape(layout); })
      .def("tl.cute.cosize",
           [](const Layout &layout) { return Cosize(layout); })
      .def("tl.cute.complement",
           [](const Layout &layout, int64_t cotarget) {
             return Complement(layout, cotarget);
           })
      .def("tl.cute.logical_divide",
           [](const Layout &layout, const Layout &tiler) {
             return LogicalDivide(layout, tiler);
           })
      .def("tl.cute.logical_product",
           [](const Layout &block, const Layout &tiler) {
             return LogicalProduct(block, tiler);
           })
      .def("tl.cute.tiled_product",
           [](const Layout &block, const Layout &tiler) {
             return TiledProduct(block, tiler);
           })
      .def("tl.cute.blocked_product",
           [](const Layout &block, const Layout &tiler) {
             return BlockedProduct(block, tiler);
           })
      // -- Make*Layout ----------------------------------------------------
      .def("tl.cute.make_column_major_layout",
           [](const IntTuple &shape) { return MakeColumnMajorLayout(shape); })
      .def("tl.cute.make_row_major_layout",
           [](const IntTuple &shape) { return MakeRowMajorLayout(shape); })
      .def("tl.cute.make_identity_layout",
           [](const IntTuple &shape) { return MakeIdentityLayout(shape); })
      // -- ComposedLayout -------------------------------------------------
      .def("tl.cute.composed_layout_recast",
           [](const ComposedLayout &layout, int old_bits, int new_bits) {
             return layout.Recast(old_bits, new_bits);
           })
      // -- TileLang -> CuTe recovery -------------------------------------
      .def("tl.cute.layout_from_tilelang",
           [](const tvm::tl::Layout &layout) {
             return LayoutFromTileLang(layout);
           })
      .def("tl.cute.layout_from_tilelang_hierarchical",
           [](const tvm::tl::Layout &layout) {
             return LayoutFromTileLangHierarchical(layout);
           })
      .def("tl.cute.composed_layout_from_tilelang",
           [](const tvm::tl::Layout &layout) {
             return ComposedLayoutFromTileLang(layout);
           })
      .def("tl.cute.restrict",
           [](const Layout &layout, const Array<Range> &range) {
             return Restrict(layout, range);
           });
}

} // namespace cute
} // namespace tl
} // namespace tvm
