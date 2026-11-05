/*!
 * \file layout/layout.cc
 *
 */

#include "layout.h"
#include "ir/analyzer.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "support/check.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/runtime/logging.h>

#include "expr_utils.h"
#include "iter_map_utils.h"
#include <tvm/ir/prim/op.h>

#include "utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;
using namespace ffi;

namespace {

int LayoutBoundBits(const PrimExpr &expression) {
  const auto *type = expression.ty().as<PrimTypeObj>();
  return type && type->dtype.bits > 32 ? 64 : 32;
}

PrimExpr LayoutBoundLiteral(int64_t value, int minimum_bits = 32) {
  if (minimum_bits <= 32 && value >= std::numeric_limits<int32_t>::min() &&
      value <= std::numeric_limits<int32_t>::max())
    return IntImm::Int32(value);
  return IntImm::Int64(value);
}

// A swizzle combines nonnegative coordinate bits. The native analyzer does
// not bound XOR/OR yet; add those layout facts only to a private query clone.
// The real expression and the caller's accepted constraints remain unchanged.
sym::ConstIntBound LayoutCoordinateBound(const PrimExpr &expression,
                                         const sym::Analyzer &analyzer) {
  sym::Analyzer bounded = analyzer->Clone();
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      expression,
      [&](const ffi::ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
        const PrimExpr *left = nullptr;
        const PrimExpr *right = nullptr;
        if (const auto *op = node.as<prim::BitwiseXorObj>()) {
          left = &op->a;
          right = &op->b;
        } else if (const auto *op = node.as<prim::BitwiseOrObj>()) {
          left = &op->a;
          right = &op->b;
        } else {
          return ffi::WalkResult::Advance();
        }
        auto a = bounded->const_int_bound(*left);
        auto b = bounded->const_int_bound(*right);
        if (a->min_value < 0 || b->min_value < 0 ||
            a->max_value == sym::ConstIntBound::kPosInf ||
            b->max_value == sym::ConstIntBound::kPosInf)
          return ffi::WalkResult::Advance();
        // If both operands use only the low k bits, so does their XOR/OR.
        // Fill those k bits without overflowing a signed host integer.
        uint64_t mask =
            static_cast<uint64_t>(std::max(a->max_value, b->max_value));
        for (int shift = 1; shift < 64; shift *= 2)
          mask |= mask >> shift;
        if (mask >= static_cast<uint64_t>(sym::ConstIntBound::kPosInf))
          return ffi::WalkResult::Advance();
        PrimExpr value = node.as_or_throw<PrimExpr>();
        PrimType type = value.ty().as_or_throw<PrimType>();
        if (!type.IsScalar())
          return ffi::WalkResult::Advance();
        AppendAnalyzerConstraint(
            bounded.get(),
            sym::PredicateConstraint(
                And(value >= IntImm(type, 0),
                    value <= IntImm(type, static_cast<int64_t>(mask)))));
        return ffi::WalkResult::Advance();
      });
  return bounded->const_int_bound(expression);
}

// Native index analysis may treat integer arithmetic as mathematical and a
// narrowing cast as monotone. Neither gives a safe allocation bound when a
// coordinate wraps. Relax only such subexpressions to independent values of
// their result type, for this bound query; never rewrite the actual layout.
std::pair<PrimExpr, bool> RelaxLayoutCoordinate(const PrimExpr &expression,
                                                const sym::Analyzer &analyzer) {
  bool relaxed = false;
  auto value =
      ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
          expression,
          [&](const PrimExpr &current) -> PrimExpr {
            auto type = current.ty().as<PrimType>();
            if (!type || !type->IsScalar() ||
                !type->MatchesCode(kDLInt, kDLUInt))
              return current;

            PrimExpr mathematical = current;
            const PrimExpr *left = nullptr;
            const PrimExpr *right = nullptr;
            if (const auto *op = current.as<prim::CastObj>()) {
              auto source_type = op->value.ty().as<PrimType>();
              if (!source_type || !source_type->IsScalar() ||
                  !source_type->MatchesCode(kDLInt, kDLUInt))
                return current;
              // Bound the value before conversion, not the already narrowed
              // bound.
              mathematical = op->value;
            } else if (const auto *op = current.as<prim::AddObj>()) {
              left = &op->a;
              right = &op->b;
            } else if (const auto *op = current.as<prim::SubObj>()) {
              left = &op->a;
              right = &op->b;
            } else if (const auto *op = current.as<prim::MulObj>()) {
              left = &op->a;
              right = &op->b;
            } else {
              return current;
            }

            bool fits = type->bits() > 0 && type->bits() <= 64;
            if (left) {
              // In particular, unsigned subtraction must be queried in a signed
              // type; its native result bound may already hide an underflow.
              auto a = LayoutCoordinateBound(*left, analyzer);
              auto b = LayoutCoordinateBound(*right, analyzer);
              fits = fits && a->min_value != sym::ConstIntBound::kNegInf &&
                     a->max_value != sym::ConstIntBound::kPosInf &&
                     b->min_value != sym::ConstIntBound::kNegInf &&
                     b->max_value != sym::ConstIntBound::kPosInf;
              if (fits) {
                PrimExpr a_wide = cast(*left, PrimType::Int(64));
                PrimExpr b_wide = cast(*right, PrimType::Int(64));
                if (current.as<prim::AddObj>())
                  mathematical = prim::Add(a_wide, b_wide);
                else if (current.as<prim::SubObj>())
                  mathematical = prim::Sub(a_wide, b_wide);
                else
                  mathematical = prim::Mul(a_wide, b_wide);
              }
            }
            if (fits) {
              auto bound = LayoutCoordinateBound(mathematical, analyzer);
              const bool is_unsigned = type->code() == kDLUInt;
              const int magnitude_bits = type->bits() - (is_unsigned ? 0 : 1);
              const int64_t upper = magnitude_bits >= 63
                                        ? std::numeric_limits<int64_t>::max()
                                        : (int64_t{1} << magnitude_bits) - 1;
              const int64_t lower = is_unsigned ? 0 : -upper - 1;
              fits = bound->min_value != sym::ConstIntBound::kNegInf &&
                     bound->max_value != sym::ConstIntBound::kPosInf &&
                     bound->min_value >= lower && bound->max_value <= upper;
            }
            if (fits)
              return current;
            relaxed = true;
            // A fresh Var has the full native dtype domain. Do not equate it
            // with the original expression, or the unsound narrowing facts
            // return.
            return PrimVar("layout_bound", *type);
          })
          .as_or_throw<PrimExpr>();
  return {value, relaxed};
}

Array<PrimVar> CreateReshapeVars(const Array<PrimExpr> &shape,
                                 sym::AnalyzerObj *analyzer) {
  Array<PrimVar> vars;
  vars.reserve(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    auto var = PrimVar(std::string("n_") + std::to_string(i), shape[i].ty());
    BindAnalyzer(analyzer, var, Range(0, shape[i]));
    vars.push_back(var);
  }
  return vars;
}

PrimExpr ComputeFlatIndex(const Array<PrimExpr> &shape,
                          const Array<PrimVar> &vars) {
  PrimExpr flat_index = IntImm::Int32(0);
  for (size_t i = 0; i < shape.size(); ++i) {
    PrimExpr stride = IntImm::Int32(1);
    for (size_t j = i + 1; j < shape.size(); ++j) {
      stride = stride * shape[j];
    }
    flat_index = flat_index + vars[i] * stride;
  }
  return flat_index;
}

Array<PrimExpr> RecoverOriginalIndices(const Array<PrimExpr> &shape,
                                       const PrimExpr &flat_index) {
  Array<PrimExpr> original_indices;
  PrimExpr remaining = flat_index;
  for (size_t i = 0; i < shape.size(); ++i) {
    PrimExpr stride = IntImm::Int32(1);
    for (size_t j = i + 1; j < shape.size(); ++j) {
      stride = stride * shape[j];
    }
    original_indices.push_back(floordiv(remaining, stride));
    remaining = floormod(remaining, stride);
  }
  return original_indices;
}

Array<PrimExpr> SubstituteForwardIndex(const Array<PrimExpr> &forward_index,
                                       const Array<PrimExpr> &input_shape,
                                       const Array<PrimExpr> &original_indices,
                                       sym::AnalyzerObj *analyzer) {
  Array<PrimExpr> new_forward_index;
  for (const auto &fwd_expr : forward_index) {
    PrimExpr substituted = fwd_expr;
    for (size_t i = 0; i < input_shape.size(); ++i) {
      substituted = SubstituteLayoutVars(
          substituted, {{InputPlaceholder(i), original_indices[i]}});
    }
    new_forward_index.push_back(analyzer->Simplify(substituted));
  }
  return new_forward_index;
}

PrimExpr SubstituteReshapedExpr(const PrimExpr &expr,
                                const Array<PrimExpr> &input_shape,
                                const Array<PrimExpr> &original_indices,
                                sym::AnalyzerObj *analyzer) {
  PrimExpr substituted = expr;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    substituted = SubstituteLayoutVars(
        substituted, {{InputPlaceholder(i), original_indices[i]}});
  }
  return analyzer->Simplify(substituted);
}

Array<PrimExpr> RestoreInputPlaceholders(const Array<PrimExpr> &forward_index,
                                         const Array<PrimVar> &vars) {
  Array<PrimExpr> restored = forward_index;
  for (size_t i = 0; i < vars.size(); ++i) {
    restored = SubstituteLayoutVars(restored, {{vars[i], InputPlaceholder(i)}});
  }
  return restored;
}

PrimExpr RestoreInputPlaceholders(const PrimExpr &expr,
                                  const Array<PrimVar> &vars) {
  PrimExpr restored = expr;
  for (size_t i = 0; i < vars.size(); ++i) {
    restored = SubstituteLayoutVars(restored, {{vars[i], InputPlaceholder(i)}});
  }
  return restored;
}

/*!
 * \brief Derive the layout of a reinterpreting view (same storage, new
 *        element width).
 *
 * Layout outputs are measured in the buffer's own elements, so aliases of
 * one storage are compatible iff they induce the same storage-bit map,
 * bit(k) = flat_output(k) * elem_bits. Deriving a view's layout is unit
 * conversion through that map, and the directions are not symmetric:
 *  - Narrowing by a factor f: old element q becomes new elements
 *    f*q .. f*q+f-1; always representable, as last_output * f + lane.
 *  - Widening by f: representable only when every aligned group of f old
 *    elements sits contiguously in one aligned slot, which is proven here;
 *    a layout that interleaves storage below the new element width (e.g.
 *    the pack::16b TMEM form) has no compatible view layout, a hard error.
 *
 * Returns an undefined Layout when the widths match: the generic flat-index
 * reshape is exact then, and only then.
 */
Layout TryReinterpretReshape(const LayoutNode *layout_node,
                             const Array<PrimExpr> &shape,
                             sym::AnalyzerObj *analyzer,
                             const PrimExpr &old_elem_bits_expr,
                             const PrimExpr &new_elem_bits_expr) {
  auto old_elem_bits = TryConstInt(old_elem_bits_expr);
  auto new_elem_bits = TryConstInt(new_elem_bits_expr);
  if (!old_elem_bits || !new_elem_bits || *old_elem_bits == *new_elem_bits) {
    return Layout();
  }
  ICHECK_GT(*old_elem_bits, 0);
  ICHECK_GT(*new_elem_bits, 0);

  const Array<PrimExpr> &input_shape = layout_node->InputShape();
  Array<PrimVar> new_vars = CreateReshapeVars(shape, analyzer);
  PrimExpr flat_index = ComputeFlatIndex(shape, new_vars);

  if (*old_elem_bits > *new_elem_bits) {
    ICHECK_EQ(*old_elem_bits % *new_elem_bits, 0)
        << "cannot reinterpret " << *old_elem_bits << "-bit elements as "
        << *new_elem_bits << "-bit elements";
    int64_t pack_factor = *old_elem_bits / *new_elem_bits;
    PrimExpr old_flat_index = floordiv(flat_index, IntImm::Int32(pack_factor));
    PrimExpr lane_in_pack = floormod(flat_index, IntImm::Int32(pack_factor));

    Array<PrimExpr> original_indices =
        RecoverOriginalIndices(input_shape, old_flat_index);
    Array<PrimExpr> new_forward_index =
        SubstituteForwardIndex(layout_node->GetForwardIndex(), input_shape,
                               original_indices, analyzer);
    ICHECK(!new_forward_index.empty());
    PrimExpr last = new_forward_index.back();
    new_forward_index.Set(
        new_forward_index.size() - 1,
        analyzer->Simplify(last * IntImm::Int32(pack_factor) + lane_in_pack));
    new_forward_index = RestoreInputPlaceholders(new_forward_index, new_vars);
    return Layout(shape, new_forward_index);
  } else { // *old_elem_bits < *new_elem_bits
    ICHECK_EQ(*new_elem_bits % *old_elem_bits, 0)
        << "cannot reinterpret " << *old_elem_bits << "-bit elements as "
        << *new_elem_bits << "-bit elements";
    int64_t pack_factor = *new_elem_bits / *old_elem_bits;

    PrimVar lane("_lane", flat_index.ty());
    BindAnalyzer(analyzer, lane, Range(0, IntImm::Int32(pack_factor)));
    Array<PrimExpr> base = SubstituteForwardIndex(
        layout_node->GetForwardIndex(), input_shape,
        RecoverOriginalIndices(input_shape,
                               flat_index * IntImm::Int32(pack_factor)),
        analyzer);
    Array<PrimExpr> probe = SubstituteForwardIndex(
        layout_node->GetForwardIndex(), input_shape,
        RecoverOriginalIndices(input_shape,
                               flat_index * IntImm::Int32(pack_factor) + lane),
        analyzer);
    ICHECK(!base.empty());
    PrimExpr slot = floordiv(base.back(), IntImm::Int32(pack_factor));
    bool compatible = analyzer->CanProveEqual(
        probe.back(), slot * IntImm::Int32(pack_factor) + lane);
    for (size_t i = 0; compatible && i + 1 < base.size(); ++i) {
      compatible = analyzer->CanProveEqual(probe[i], base[i]);
    }
    ICHECK(compatible)
        << "cannot reinterpret layout " << layout_node->DebugOutput() << " as "
        << *new_elem_bits << "-bit elements: aliasing buffers must induce the "
        << "same storage layout, and this one interleaves storage below the "
        << "new element width";
    Array<PrimExpr> new_forward_index = base;
    new_forward_index.Set(new_forward_index.size() - 1,
                          analyzer->Simplify(slot));
    new_forward_index = RestoreInputPlaceholders(new_forward_index, new_vars);
    return Layout(shape, new_forward_index);
  }
}

/*!
 * \brief The Fragment counterpart of TryReinterpretReshape above: the same
 *        unit conversion over the per-thread value index, with the thread
 *        mapping required to be invariant inside each widened group.
 */
Fragment TryReinterpretReshape(const FragmentNode *fragment_node,
                               const Array<PrimExpr> &shape,
                               sym::AnalyzerObj *analyzer,
                               const PrimExpr &old_elem_bits_expr,
                               const PrimExpr &new_elem_bits_expr) {
  auto old_elem_bits = TryConstInt(old_elem_bits_expr);
  auto new_elem_bits = TryConstInt(new_elem_bits_expr);
  if (!old_elem_bits || !new_elem_bits || *old_elem_bits == *new_elem_bits) {
    return Fragment();
  }
  ICHECK_GT(*old_elem_bits, 0);
  ICHECK_GT(*new_elem_bits, 0);

  const Array<PrimExpr> &input_shape = fragment_node->InputShape();
  Array<PrimVar> new_vars = CreateReshapeVars(shape, analyzer);
  PrimExpr flat_index = ComputeFlatIndex(shape, new_vars);

  auto make_fragment = [&](Array<PrimExpr> forward_index,
                           PrimExpr forward_thread) {
    forward_index = RestoreInputPlaceholders(forward_index, new_vars);
    forward_thread = RestoreInputPlaceholders(forward_thread, new_vars);
    Fragment reshaped(shape, forward_index, forward_thread,
                      fragment_node->ReplicateExtent(), std::nullopt);
    if (fragment_node->ThreadRange().has_value()) {
      reshaped = reshaped->BindThreadRange(fragment_node->ThreadRange());
    }
    return reshaped;
  };

  if (*old_elem_bits > *new_elem_bits) {
    ICHECK_EQ(*old_elem_bits % *new_elem_bits, 0)
        << "cannot reinterpret " << *old_elem_bits << "-bit elements as "
        << *new_elem_bits << "-bit elements";
    int64_t pack_factor = *old_elem_bits / *new_elem_bits;
    PrimExpr old_flat_index = floordiv(flat_index, IntImm::Int32(pack_factor));
    PrimExpr lane_in_pack = floormod(flat_index, IntImm::Int32(pack_factor));

    Array<PrimExpr> original_indices =
        RecoverOriginalIndices(input_shape, old_flat_index);
    Array<PrimExpr> new_forward_index =
        SubstituteForwardIndex(fragment_node->GetForwardIndex(), input_shape,
                               original_indices, analyzer);
    ICHECK(!new_forward_index.empty());
    PrimExpr last = new_forward_index.back();
    new_forward_index.Set(
        new_forward_index.size() - 1,
        analyzer->Simplify(last * IntImm::Int32(pack_factor) + lane_in_pack));
    PrimExpr new_forward_thread =
        SubstituteReshapedExpr(fragment_node->GetForwardThread(), input_shape,
                               original_indices, analyzer);
    return make_fragment(new_forward_index, new_forward_thread);
  } else { // *old_elem_bits < *new_elem_bits
    ICHECK_EQ(*new_elem_bits % *old_elem_bits, 0)
        << "cannot reinterpret " << *old_elem_bits << "-bit elements as "
        << *new_elem_bits << "-bit elements";
    int64_t pack_factor = *new_elem_bits / *old_elem_bits;
    Array<PrimExpr> original_indices = RecoverOriginalIndices(
        input_shape, flat_index * IntImm::Int32(pack_factor));

    PrimVar lane("_lane", flat_index.ty());
    BindAnalyzer(analyzer, lane, Range(0, IntImm::Int32(pack_factor)));
    Array<PrimExpr> probe_indices = RecoverOriginalIndices(
        input_shape, flat_index * IntImm::Int32(pack_factor) + lane);
    Array<PrimExpr> base =
        SubstituteForwardIndex(fragment_node->GetForwardIndex(), input_shape,
                               original_indices, analyzer);
    Array<PrimExpr> probe = SubstituteForwardIndex(
        fragment_node->GetForwardIndex(), input_shape, probe_indices, analyzer);
    PrimExpr thread_base =
        SubstituteReshapedExpr(fragment_node->GetForwardThread(), input_shape,
                               original_indices, analyzer);
    PrimExpr thread_probe =
        SubstituteReshapedExpr(fragment_node->GetForwardThread(), input_shape,
                               probe_indices, analyzer);
    ICHECK(!base.empty());
    PrimExpr slot = floordiv(base.back(), IntImm::Int32(pack_factor));
    bool compatible =
        analyzer->CanProveEqual(probe.back(),
                                slot * IntImm::Int32(pack_factor) + lane) &&
        analyzer->CanProveEqual(thread_probe, thread_base);
    for (size_t i = 0; compatible && i + 1 < base.size(); ++i) {
      compatible = analyzer->CanProveEqual(probe[i], base[i]);
    }
    ICHECK(compatible)
        << "cannot reinterpret fragment " << fragment_node->DebugOutput()
        << " as " << *new_elem_bits << "-bit elements: aliasing buffers must "
        << "induce the same storage layout, and this one interleaves storage "
        << "or threads below the new element width";
    Array<PrimExpr> new_forward_index = base;
    new_forward_index.Set(new_forward_index.size() - 1,
                          analyzer->Simplify(slot));
    return make_fragment(new_forward_index, thread_base);
  }
}

} // namespace

static constexpr size_t kMaxPlaceholders = 16;

static PrimVar getPlaceholder(const std::string &s) {
  // Pre-allocate all possible placeholders so the map is immutable after init.
  // C++11 guarantees thread-safe initialization of function-local statics,
  // so concurrent reads are safe without a mutex.
  static const std::unordered_map<std::string, PrimVar> map = []() {
    std::unordered_map<std::string, PrimVar> m;
    m.reserve(kMaxPlaceholders + 1);
    m.emplace("_rep", PrimVar("_rep"));
    for (size_t i = 0; i < kMaxPlaceholders; ++i) {
      std::string key{'_', char('i' + i)};
      m.emplace(key, PrimVar(key));
    }
    return m;
  }();
  auto it = map.find(s);
  ICHECK(it != map.end()) << "Unknown placeholder: " << s;
  return it->second;
}

PrimVar ReplicationPlaceholder() { return getPlaceholder("_rep"); }
PrimVar InputPlaceholder(size_t idx) {
  return getPlaceholder(std::string{'_', char('i' + idx)});
}

Map<PrimVar, Range> LayoutNode::GetVarMap() const {
  Map<PrimVar, Range> map;
  for (size_t i = 0; i < InputDim(); i++) {
    map.Set(InputPlaceholder(i), {0, input_size_[i]});
  }
  return map;
}

Map<PrimVar, Range> FragmentNode::GetVarMap() const {
  auto map = LayoutNode::GetVarMap();
  map.Set(ReplicationPlaceholder(), {0, ReplicateExtent()});
  return map;
}

namespace {

constexpr int64_t kMaxExactLayoutDomain = 1 << 18;

struct IntTupleHash {
  size_t operator()(const std::vector<int64_t> &values) const {
    size_t seed = values.size();
    for (int64_t value : values) {
      seed ^=
          std::hash<int64_t>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

std::string FormatIntTuple(const std::vector<int64_t> &values) {
  std::ostringstream os;
  os << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << values[i];
  }
  os << ']';
  return os.str();
}

enum class ExactInjectivityStatus { kInjective, kNonInjective, kUnknown };

struct ExactInjectivityResult {
  ExactInjectivityStatus status;
  std::string detail;
};

ExactInjectivityResult
CheckStaticInjectivity(const Array<PrimExpr> &forward_indices,
                       const Map<PrimVar, Range> &input_iters) {
  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  std::vector<PrimVar> vars;
  std::vector<int64_t> mins;
  std::vector<int64_t> extents;
  int64_t domain_size = 1;

  for (const auto &[var, range] : input_iters) {
    PrimExpr simplified_min = analyzer->Simplify(range->begin);
    PrimExpr simplified_extent = analyzer->Simplify(range->extent);
    auto min = TryConstInt(simplified_min);
    auto extent = TryConstInt(simplified_extent);
    if (!min || !extent) {
      return {ExactInjectivityStatus::kUnknown,
              "the logical input domain is symbolic"};
    }
    if (*extent < 0) {
      return {ExactInjectivityStatus::kUnknown,
              "the logical input domain has a negative extent"};
    }
    if (*extent == 0) {
      return {ExactInjectivityStatus::kInjective, ""};
    }
    if (*extent > kMaxExactLayoutDomain / domain_size) {
      return {ExactInjectivityStatus::kUnknown,
              "the logical input domain exceeds the exact-check limit of " +
                  std::to_string(kMaxExactLayoutDomain) + " points"};
    }
    domain_size *= *extent;
    vars.push_back(var);
    mins.push_back(*min);
    extents.push_back(*extent);
  }

  using Coordinate = std::vector<int64_t>;
  std::unordered_map<Coordinate, Coordinate, IntTupleHash> first_preimage;
  first_preimage.reserve(static_cast<size_t>(domain_size));

  for (int64_t linear = 0; linear < domain_size; ++linear) {
    int64_t residual = linear;
    Coordinate input(vars.size());
    Map<PrimVar, PrimExpr> substitution;
    for (size_t rev = vars.size(); rev > 0; --rev) {
      size_t i = rev - 1;
      input[i] = mins[i] + residual % extents[i];
      residual /= extents[i];
      substitution.Set(vars[i], IntImm(vars[i].ty(), input[i]));
    }

    Coordinate output;
    output.reserve(forward_indices.size());
    for (const PrimExpr &index : forward_indices) {
      PrimExpr value =
          analyzer->Simplify(SubstituteLayoutVars(index, substitution));
      std::optional<int64_t> constant = EvaluateConstantInteger(value);
      if (!constant) {
        return {ExactInjectivityStatus::kUnknown,
                "the forward map could not be evaluated exactly at logical "
                "coordinates " +
                    FormatIntTuple(input)};
      }
      output.push_back(*constant);
    }

    auto [it, inserted] = first_preimage.emplace(output, input);
    if (!inserted) {
      return {ExactInjectivityStatus::kNonInjective,
              "logical coordinates " + FormatIntTuple(it->second) + " and " +
                  FormatIntTuple(input) + " both map to physical coordinates " +
                  FormatIntTuple(output)};
    }
  }
  return {ExactInjectivityStatus::kInjective, ""};
}

enum class InverseRoundTripStatus { kValid, kInvalid, kUnknown };

struct InverseRoundTripResult {
  InverseRoundTripStatus status;
  std::string detail;
};

bool GetStaticDomain(const Array<PrimExpr> &shape,
                     std::vector<int64_t> *extents, int64_t *domain_size) {
  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  *domain_size = 1;
  extents->clear();
  extents->reserve(shape.size());
  for (const PrimExpr &dim : shape) {
    auto extent = TryConstInt(analyzer->Simplify(dim));
    if (!extent || *extent < 0) {
      return false;
    }
    if (*extent == 0) {
      *domain_size = 0;
    } else if (*domain_size != 0) {
      if (*extent > kMaxExactLayoutDomain / *domain_size) {
        return false;
      }
      *domain_size *= *extent;
    }
    extents->push_back(*extent);
  }
  return true;
}

std::vector<int64_t> UnflattenCoordinate(int64_t linear,
                                         const std::vector<int64_t> &extents) {
  std::vector<int64_t> coordinate(extents.size());
  for (size_t rev = extents.size(); rev > 0; --rev) {
    size_t i = rev - 1;
    coordinate[i] = linear % extents[i];
    linear /= extents[i];
  }
  return coordinate;
}

std::optional<std::vector<int64_t>>
EvaluateStaticMap(const Array<PrimExpr> &indices,
                  const std::vector<int64_t> &input) {
  Map<PrimVar, PrimExpr> substitution;
  for (size_t i = 0; i < input.size(); ++i) {
    PrimVar placeholder = InputPlaceholder(i);
    substitution.Set(placeholder,
                     IntImm(placeholder.ty(), static_cast<int64_t>(input[i])));
  }

  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  std::vector<int64_t> output;
  output.reserve(indices.size());
  for (const PrimExpr &index : indices) {
    PrimExpr value =
        analyzer->Simplify(SubstituteLayoutVars(index, substitution));
    std::optional<int64_t> constant = EvaluateConstantInteger(value);
    if (!constant) {
      return std::nullopt;
    }
    output.push_back(*constant);
  }
  return output;
}

bool CanProveInverseRoundTrip(const Array<PrimExpr> &input_shape,
                              const Array<PrimExpr> &output_shape,
                              const Array<PrimExpr> &forward_indices,
                              const Array<PrimExpr> &backward_indices) {
  sym::Analyzer analyzer = CreateTileLangAnalyzer();

  Array<PrimVar> logical_vars;
  Map<PrimVar, PrimExpr> logical_substitution;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    PrimVar var("__layout_logical_" + std::to_string(i), input_shape[i].ty());
    BindAnalyzer(analyzer, var, Range(0, input_shape[i]));
    logical_vars.push_back(var);
    logical_substitution.Set(InputPlaceholder(i), var);
  }
  Array<PrimExpr> physical_from_logical =
      SubstituteLayoutVars(forward_indices, logical_substitution);
  for (size_t i = 0; i < physical_from_logical.size(); ++i) {
    if (!analyzer->CanProve(physical_from_logical[i] >= 0) ||
        !analyzer->CanProve(physical_from_logical[i] < output_shape[i]))
      return false;
  }
  Map<PrimVar, PrimExpr> inverse_substitution;
  for (size_t i = 0; i < physical_from_logical.size(); ++i) {
    inverse_substitution.Set(InputPlaceholder(i), physical_from_logical[i]);
  }
  Array<PrimExpr> recovered_logical =
      SubstituteLayoutVars(backward_indices, inverse_substitution);
  for (size_t i = 0; i < logical_vars.size(); ++i) {
    if (!analyzer->CanProveEqual(recovered_logical[i], logical_vars[i])) {
      return false;
    }
  }

  Array<PrimVar> physical_vars;
  Map<PrimVar, PrimExpr> physical_substitution;
  for (size_t i = 0; i < output_shape.size(); ++i) {
    PrimVar var("__layout_physical_" + std::to_string(i), output_shape[i].ty());
    BindAnalyzer(analyzer, var, Range(0, output_shape[i]));
    physical_vars.push_back(var);
    physical_substitution.Set(InputPlaceholder(i), var);
  }
  Array<PrimExpr> logical_from_physical =
      SubstituteLayoutVars(backward_indices, physical_substitution);
  for (size_t i = 0; i < logical_from_physical.size(); ++i) {
    if (!analyzer->CanProve(logical_from_physical[i] >= 0) ||
        !analyzer->CanProve(logical_from_physical[i] < input_shape[i]))
      return false;
  }
  Map<PrimVar, PrimExpr> forward_substitution;
  for (size_t i = 0; i < logical_from_physical.size(); ++i) {
    forward_substitution.Set(InputPlaceholder(i), logical_from_physical[i]);
  }
  Array<PrimExpr> recovered_physical =
      SubstituteLayoutVars(forward_indices, forward_substitution);
  for (size_t i = 0; i < physical_vars.size(); ++i) {
    if (!analyzer->CanProveEqual(recovered_physical[i], physical_vars[i])) {
      return false;
    }
  }
  return true;
}

InverseRoundTripResult
CheckStaticInverseRoundTrip(const Array<PrimExpr> &input_shape,
                            const Array<PrimExpr> &output_shape,
                            const Array<PrimExpr> &forward_indices,
                            const Array<PrimExpr> &backward_indices) {
  if (CanProveInverseRoundTrip(input_shape, output_shape, forward_indices,
                               backward_indices)) {
    return {InverseRoundTripStatus::kValid, ""};
  }

  std::vector<int64_t> input_extents;
  std::vector<int64_t> output_extents;
  int64_t input_domain_size = 0;
  int64_t output_domain_size = 0;
  if (!GetStaticDomain(input_shape, &input_extents, &input_domain_size) ||
      !GetStaticDomain(output_shape, &output_extents, &output_domain_size)) {
    return {InverseRoundTripStatus::kUnknown,
            "the layout domain is symbolic or exceeds the exact-check limit"};
  }
  if (input_domain_size != output_domain_size) {
    return {InverseRoundTripStatus::kInvalid,
            "the logical and physical domains have different sizes"};
  }

  for (int64_t linear = 0; linear < input_domain_size; ++linear) {
    std::vector<int64_t> logical = UnflattenCoordinate(linear, input_extents);
    auto physical = EvaluateStaticMap(forward_indices, logical);
    if (!physical) {
      return {InverseRoundTripStatus::kUnknown,
              "the forward map could not be evaluated exactly at logical "
              "coordinate " +
                  FormatIntTuple(logical)};
    }
    for (size_t i = 0; i < physical->size(); ++i) {
      if ((*physical)[i] < 0 || (*physical)[i] >= output_extents[i])
        return {InverseRoundTripStatus::kInvalid,
                "the forward map leaves its physical domain at logical "
                "coordinate " +
                    FormatIntTuple(logical)};
    }
    auto recovered = EvaluateStaticMap(backward_indices, *physical);
    if (!recovered) {
      return {InverseRoundTripStatus::kUnknown,
              "the inverse map could not be evaluated exactly at physical "
              "coordinate " +
                  FormatIntTuple(*physical)};
    }
    if (*recovered != logical) {
      return {InverseRoundTripStatus::kInvalid,
              "logical coordinate " + FormatIntTuple(logical) +
                  " maps to physical coordinate " + FormatIntTuple(*physical) +
                  ", but the inferred inverse maps it to " +
                  FormatIntTuple(*recovered)};
    }
  }

  for (int64_t linear = 0; linear < output_domain_size; ++linear) {
    std::vector<int64_t> physical = UnflattenCoordinate(linear, output_extents);
    auto logical = EvaluateStaticMap(backward_indices, physical);
    if (!logical) {
      return {InverseRoundTripStatus::kUnknown,
              "the inverse map could not be evaluated exactly at physical "
              "coordinate " +
                  FormatIntTuple(physical)};
    }
    for (size_t i = 0; i < logical->size(); ++i) {
      if ((*logical)[i] < 0 || (*logical)[i] >= input_extents[i])
        return {InverseRoundTripStatus::kInvalid,
                "the inverse map leaves its logical domain at physical "
                "coordinate " +
                    FormatIntTuple(physical)};
    }
    auto recovered = EvaluateStaticMap(forward_indices, *logical);
    if (!recovered) {
      return {InverseRoundTripStatus::kUnknown,
              "the forward map could not be evaluated exactly at logical "
              "coordinate " +
                  FormatIntTuple(*logical)};
    }
    if (*recovered != physical) {
      return {InverseRoundTripStatus::kInvalid,
              "physical coordinate " + FormatIntTuple(physical) +
                  " maps to logical coordinate " + FormatIntTuple(*logical) +
                  ", but the forward map maps it back to " +
                  FormatIntTuple(*recovered)};
    }
  }
  return {InverseRoundTripStatus::kValid, ""};
}

Optional<Layout>
TryCompactStaticInverse(const Array<PrimExpr> &input_shape,
                        const Array<PrimExpr> &output_shape,
                        const Array<PrimExpr> &forward_indices) {
  // Native IterMap may not fuse an inner axis that straddles a radix boundary,
  // e.g. (i * 6 + j) // 4. Use one compact coordinate for the same finite
  // domain and let the native bijective checker invert that expression. Do not
  // turn ragged physical domains into unguarded inverses or recurse for a
  // single axis.
  std::vector<int64_t> input_extents, output_extents;
  int64_t input_size = 0, output_size = 0;
  if (input_shape.size() < 2 ||
      !GetStaticDomain(input_shape, &input_extents, &input_size) ||
      !GetStaticDomain(output_shape, &output_extents, &output_size) ||
      input_size <= 0 || input_size != output_size)
    return std::nullopt;

  auto analyzer = CreateTileLangAnalyzer();
  PrimVar flat("__layout_flat", PrimType::Int(32));
  BindAnalyzer(analyzer, flat, Range(0, IntImm::Int32(input_size)));
  auto coordinates = RecoverOriginalIndices(input_shape, flat);
  Map<PrimVar, PrimExpr> substitution;
  for (size_t i = 0; i < coordinates.size(); ++i)
    substitution.Set(InputPlaceholder(i), coordinates[i]);
  auto flat_indices = SubstituteLayoutVars(forward_indices, substitution)
                          .Map([&](const PrimExpr &expression) {
                            return analyzer->Simplify(expression);
                          });
  flat_indices =
      SubstituteLayoutVars(flat_indices, {{flat, InputPlaceholder(0)}});

  try {
    Layout compact({IntImm::Int32(input_size)}, flat_indices);
    Layout inverse = compact->Inverse();
    auto backward =
        RecoverOriginalIndices(input_shape, inverse->GetForwardIndex()[0]);
    auto checked = CheckStaticInverseRoundTrip(input_shape, output_shape,
                                               forward_indices, backward);
    if (checked.status == InverseRoundTripStatus::kValid)
      return Layout(output_shape, backward);
  } catch (const NormalizeIterException &) {
    // Preserve the original diagnostic if compact coordinates cannot prove
    // the same bijection; a failed attempt never changes the source layout.
  }
  return std::nullopt;
}

bool CanProveInjective(const Array<PrimExpr> &forward_indices,
                       const Map<PrimVar, Range> &input_iters) {
  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  Map<PrimVar, PrimExpr> other_substitution;
  PrimExpr same_input = IntImm::Bool(true);
  for (const auto &[var, range] : input_iters) {
    PrimVar other(var->name + "<OTHER>", var.ty());
    BindAnalyzer(analyzer, var, range);
    BindAnalyzer(analyzer, other, range);
    other_substitution.Set(var, other);
    same_input = And(same_input, EQ(var, other));
  }

  PrimExpr same_output = IntImm::Bool(true);
  for (const PrimExpr &index : forward_indices) {
    same_output =
        And(same_output,
            EQ(index, SubstituteLayoutVars(index, other_substitution)));
  }
  ScopedAnalyzerConstraints constraint(analyzer, same_output);
  return analyzer->CanProve(same_input);
}

// A left inverse proves injectivity even when symbolic padding prevents a
// bijective IterMap. NoCheck only constructs the candidate: every logical
// coordinate must still be recovered on the original input domain.
bool CanProveLeftInverse(const Array<PrimExpr> &forward_indices,
                         const Map<PrimVar, Range> &input_iters) {
  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  // Match the lossless coordinate view used by InverseWithLevel. A generated
  // wide partition index contains direct casts of int32 placeholders, which
  // native IterMap does not recognize as iterators. The view preserves their
  // exact domains; narrowing or mixed-width uses are left untouched.
  auto view = MakeLayoutIterMapView(forward_indices, input_iters);
  PrimExpr nonempty = IntImm::Bool(true);
  for (const auto &[var, range] : view.ranges) {
    BindAnalyzer(analyzer, var, range);
    nonempty = And(nonempty, range->extent > 0);
  }
  // Empty input domains are trivially injective. For a nonempty domain all
  // extents are positive, including symbolic strides in a flattened index.
  ScopedAnalyzerConstraints constraint(analyzer.get(), nonempty);
  auto iter_map = sym::DetectIterMap(view.indices, view.ranges, 1,
                                     sym::IterMapLevel::NoCheck, analyzer);
  if (!iter_map->errors.empty()) {
    return false;
  }
  Map<Var, PrimExpr> inverse;
  try {
    inverse = sym::InverseAffineIterMap(iter_map->indices, view.indices);
  } catch (const Error &) {
    // NoCheck can produce overlapping iter sums that the inverse builder
    // cannot invert. Such a candidate supplies no proof of injectivity.
    return false;
  }
  for (const auto &[var, range] : view.ranges) {
    auto it = inverse.find(var);
    PrimExpr recovered = it == inverse.end() ? range->begin : (*it).second;
    if (!analyzer->CanProveEqual(recovered, var)) {
      return false;
    }
  }
  return true;
}

sym::IterMapResult MakeInjectivityError(const std::string &message) {
  sym::IterMapResult result;
  result->errors.push_back(message);
  return result;
}

sym::IterMapResult
DetectInjectiveMapping(const Array<PrimExpr> &forward_indices,
                       const Map<PrimVar, Range> &input_iters,
                       bool require_padding_guard) {
  if (!require_padding_guard) {
    sym::Analyzer analyzer = CreateTileLangAnalyzer();
    sym::IterMapResult iter_map =
        sym::DetectIterMap(forward_indices, input_iters, 1,
                           sym::IterMapLevel::Bijective, analyzer);
    if (iter_map->errors.empty()) {
      return iter_map;
    }
  }

  ExactInjectivityResult exact =
      CheckStaticInjectivity(forward_indices, input_iters);
  if (exact.status == ExactInjectivityStatus::kInjective) {
    return sym::IterMapResult();
  }
  if (exact.status == ExactInjectivityStatus::kNonInjective) {
    return MakeInjectivityError(exact.detail);
  }
  // The two symbolic proofs have disjoint blind spots, so neither subsumes
  // the other. CanProveInjective propagates equalities across outputs but
  // cannot invert floordiv/floormod with a symbolic divisor: it proves
  // (i, i + j) yet fails the padded partition ((i*n+j)%128, (i*n+j)//128).
  // CanProveLeftInverse decodes a mixed-radix IterMap digit by digit, which
  // handles that partition but has no candidate inverse for (i, i + j),
  // whose second output is not a radix chain. Keep both.
  if (CanProveInjective(forward_indices, input_iters) ||
      CanProveLeftInverse(forward_indices, input_iters)) {
    return sym::IterMapResult();
  }
  return MakeInjectivityError("injectivity could not be proven: " +
                              exact.detail);
}

} // namespace

LayoutNode::LayoutNode(Array<PrimExpr> input_size,
                       Array<PrimExpr> forward_index) {
  input_size_ = input_size;
  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  UpdateAnalyzer(analyzer.get());
  forward_index_ = forward_index.Map(
      [&](const PrimExpr &e) { return analyzer->Simplify(e); });
}

Layout::Layout(Array<LayoutIterVar> forward_var,
               Array<PrimExpr> forward_index) {
  Map<PrimVar, PrimExpr> vmap;
  Array<PrimExpr> input_size;
  for (size_t i = 0; i < forward_var.size(); i++) {
    vmap.Set(forward_var[i]->var, InputPlaceholder(i));
    ICHECK(IsZero(forward_var[i]->dom->begin));
    input_size.push_back(forward_var[i]->dom->extent);
  }
  forward_index = forward_index.Map(
      [&](const PrimExpr &e) { return SubstituteLayoutVars(e, vmap); });
  auto n = make_object<LayoutNode>(input_size, forward_index);
  data_ = std::move(n);
}

Layout::Layout(Array<PrimExpr> input_size, Array<PrimExpr> forward_index) {
  auto n = make_object<LayoutNode>(input_size, forward_index);
  data_ = std::move(n);
}

void LayoutNode::RegisterReflection() {
  namespace refl = reflection;
  refl::ObjectDef<LayoutNode>()
      .def_ro("input_size", &LayoutNode::input_size_)
      .def_ro("forward_index", &LayoutNode::forward_index_)
      .def("_DebugOutput", &LayoutNode::DebugOutput);
}

void LayoutNode::UpdateAnalyzer(sym::AnalyzerObj *analyzer) const {
  for (const auto &[var, dom] : GetVarMap()) {
    BindAnalyzer(analyzer, var, dom);
  }
}

Array<PrimExpr> LayoutNode::GetForwardVars() const {
  Array<PrimExpr> vars;
  for (size_t i = 0; i < InputDim(); i++) {
    vars.push_back(InputPlaceholder(i));
  }
  return vars;
}

Array<PrimExpr> LayoutNode::OutputShape() const {
  Array<PrimExpr> ret(OutputDim(), 1);
  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  UpdateAnalyzer(analyzer.get());
  // Mutable loads are not durable analyzer premises. The layout's domains
  // still define this one bound query, so pass them explicitly instead of
  // letting an unbound input placeholder escape into an allocation extent.
  Map<Var, sym::IntSet> domains;
  for (const auto &[var, range] : GetVarMap())
    domains.Set(var, sym::IntSet::FromRange(range));
  for (size_t i = 0; i < ret.size(); i++) {
    // Bound the coordinate before adding one. Adding one in a narrow integer
    // type can wrap at its maximum even though the allocation extent is valid.
    auto [coordinate, relaxed] =
        RelaxLayoutCoordinate(forward_index_[i], analyzer);
    PrimExpr maximum = relaxed ? prim::pos_inf()
                               : analyzer->int_set(coordinate, domains).max();
    // A declared wide shape also determines the width of subsequent address
    // arithmetic. Keep that contract even when one bound fits in int32.
    int bound_bits = LayoutBoundBits(forward_index_[i]);
    for (const PrimExpr &input : input_size_)
      bound_bits = std::max(bound_bits, LayoutBoundBits(input));
    if (prim::is_pos_inf(maximum) || prim::is_neg_inf(maximum)) {
      // Unsupported or relaxed coordinates cannot use the native interval
      // endpoints. Bound the query expression, including any full dtype domain.
      auto cib = LayoutCoordinateBound(coordinate, analyzer);
      TVM_FFI_CHECK(cib->min_value >= 0 &&
                        cib->max_value != sym::ConstIntBound::kPosInf,
                    ValueError)
          << "Cannot infer a safe output extent for layout coordinate "
          << forward_index_[i];
      // Physical coordinates keep their origin: max-min+1 would discard
      // leading storage. An input extent (or 1 for a new axis) is not a bound
      // on an arbitrary output coordinate.
      maximum = LayoutBoundLiteral(cib->max_value, bound_bits);
    }
    if (const auto *constant = maximum.as<IntImmObj>()) {
      auto value = constant->value.as<int64_t>();
      TVM_FFI_CHECK(value && *value < std::numeric_limits<int64_t>::max(),
                    ValueError)
          << "Layout output extent does not fit int64";
      ret.Set(i,
              LayoutBoundLiteral(
                  *value + 1, std::max(bound_bits, LayoutBoundBits(maximum))));
    } else {
      int bits = std::max(bound_bits, LayoutBoundBits(maximum));
      if (bits <= 32 && analyzer->const_int_bound(maximum)->max_value >=
                            std::numeric_limits<int32_t>::max())
        bits = 64;
      // The coordinate can fit its type while the exclusive capacity cannot.
      // Cast before the increment, including for a symbolic maximum.
      PrimType capacity_type = PrimType::Int(bits);
      ret.Set(i, analyzer->Simplify(cast(maximum, capacity_type) +
                                    IntImm(capacity_type, 1)));
    }
  }
  return ret;
}

Array<Range> LayoutNode::MapRegionBounds(const Array<Range> &region) const {
  ICHECK_EQ(region.size(), InputDim())
      << "MapRegionBounds: region rank (" << region.size()
      << ") != layout input rank (" << InputDim() << ")";

  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  // Bind each InputPlaceholder to the sub-range [min, min + extent)
  // instead of the full extent [0, input_size) that OutputShape uses.
  Map<Var, sym::IntSet> domains;
  for (size_t i = 0; i < InputDim(); i++) {
    PrimExpr min = region[i]->begin;
    PrimExpr ext = region[i]->extent;
    Range range(min, min + ext);
    BindAnalyzer(analyzer, InputPlaceholder(i), range);
    domains.Set(InputPlaceholder(i), sym::IntSet::FromRange(range));
  }

  Array<Range> result;
  result.reserve(OutputDim());
  for (size_t i = 0; i < OutputDim(); i++) {
    auto [coordinate, relaxed] =
        RelaxLayoutCoordinate(forward_index_[i], analyzer);
    PrimExpr lo = prim::neg_inf();
    PrimExpr hi = prim::pos_inf();
    if (!relaxed) {
      auto ist = analyzer->int_set(coordinate, domains);
      lo = ist.min();
      hi = ist.max();
    }
    if (prim::is_neg_inf(lo) || prim::is_pos_inf(hi)) {
      // Use the relaxed expression for both endpoints. Narrowing only the old
      // maximum would miss coordinates that wrap back to zero.
      auto cib = LayoutCoordinateBound(coordinate, analyzer);
      if (cib->min_value != sym::ConstIntBound::kNegInf &&
          cib->max_value != sym::ConstIntBound::kPosInf) {
        int bits = LayoutBoundBits(forward_index_[i]);
        for (const Range &input : region)
          bits = std::max({bits, LayoutBoundBits(input->begin),
                           LayoutBoundBits(input->extent)});
        lo = LayoutBoundLiteral(cib->min_value, bits);
        hi = LayoutBoundLiteral(cib->max_value, bits);
      } else {
        // A full-layout bound only covers regions inside its logical domain.
        for (size_t axis = 0; axis < region.size(); ++axis) {
          TVM_FFI_CHECK(analyzer->CanProve(region[axis]->begin >= 0) &&
                            analyzer->CanProve(region[axis]->begin +
                                                   region[axis]->extent <=
                                               input_size_[axis]),
                        ValueError)
              << "Cannot bound a layout region outside its input domain";
        }
        Array<PrimExpr> out_shape = OutputShape();
        lo = IntImm::Int32(0);
        hi = out_shape[i] - 1;
      }
    }
    PrimExpr end = hi;
    if (const auto *constant = hi.as<IntImmObj>()) {
      auto value = constant->value.as<int64_t>();
      TVM_FFI_CHECK(value && *value < std::numeric_limits<int64_t>::max(),
                    ValueError)
          << "Layout region bound does not fit int64";
      const int64_t end_value = *value + 1;
      int bits = std::max(LayoutBoundBits(lo), LayoutBoundBits(hi));
      if (const auto *lower = lo.as<IntImmObj>()) {
        auto begin_value = lower->value.as<int64_t>();
        TVM_FFI_CHECK(begin_value && end_value >= *begin_value, ValueError)
            << "Layout region bounds do not fit int64";
        // Both endpoints may fit int32 while their distance does not. Do the
        // subtraction in unsigned C++ arithmetic first, then select one
        // signed type that contains the origin, exclusive end, and capacity.
        const uint64_t extent = static_cast<uint64_t>(end_value) -
                                static_cast<uint64_t>(*begin_value);
        TVM_FFI_CHECK(extent <= static_cast<uint64_t>(
                                    std::numeric_limits<int64_t>::max()),
                      ValueError)
            << "Layout region extent does not fit int64";
        if (*begin_value < std::numeric_limits<int32_t>::min() ||
            end_value > std::numeric_limits<int32_t>::max() ||
            extent > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
          bits = 64;
        result.push_back(Range::FromMinExtent(
            LayoutBoundLiteral(*begin_value, bits),
            LayoutBoundLiteral(static_cast<int64_t>(extent), bits)));
        continue;
      }
    }
    int bits = std::max(LayoutBoundBits(lo), LayoutBoundBits(hi));
    if (bits <= 32) {
      auto lower = analyzer->const_int_bound(lo);
      auto upper = analyzer->const_int_bound(hi);
      // Once both bounds are within int32 the C++ difference fits int64.
      // Otherwise promote without evaluating an overflowing difference.
      if (lower->min_value < std::numeric_limits<int32_t>::min() ||
          upper->max_value >= std::numeric_limits<int32_t>::max() ||
          upper->max_value + 1 - lower->min_value >
              std::numeric_limits<int32_t>::max())
        bits = 64;
    }
    PrimType capacity_type = PrimType::Int(bits);
    lo = cast(lo, capacity_type);
    end =
        analyzer->Simplify(cast(hi, capacity_type) + IntImm(capacity_type, 1));
    result.push_back(Range(lo, end));
  }
  return result;
}

PrimExpr LayoutNode::GetLinearizedForwardIndex() const {
  Array<PrimExpr> output_shape = OutputShape();
  ICHECK_EQ(output_shape.size(), forward_index_.size());

  PrimExpr linearized_index = IntImm::Int32(0);
  for (size_t i = 0; i < forward_index_.size(); ++i) {
    linearized_index = linearized_index * output_shape[i] + forward_index_[i];
  }
  return linearized_index;
}

Array<PrimExpr> LayoutNode::Forward(const Array<PrimExpr> &vars) const {
  if (vars.empty())
    return forward_index_;
  ICHECK_GE(vars.size(), InputDim());

  // Take the last InputDim() elements for transformation
  Array<PrimExpr> transform_vars;
  for (size_t i = vars.size() - InputDim(); i < vars.size(); i++) {
    transform_vars.push_back(vars[i]);
  }

  Map<PrimVar, PrimExpr> vmap;
  for (size_t i = 0; i < InputDim(); i++) {
    vmap.Set(InputPlaceholder(i), transform_vars[i]);
  }

  Array<PrimExpr> transformed = forward_index_.Map(
      [&](const PrimExpr &e) { return SubstituteLayoutVars(e, vmap); });
  // Concatenate with the remaining elements from vars
  Array<PrimExpr> result;
  for (size_t i = 0; i < vars.size() - InputDim(); i++) {
    result.push_back(vars[i]);
  }
  for (const auto &expr : transformed) {
    result.push_back(expr);
  }

  return result;
}

Layout LayoutNode::Repeat(int dim, int factor) const {
  if (factor < 1) {
    TVM_FFI_THROW(ValueError) << "factor must be >= 1, got " << factor;
  }
  if (factor == 1) {
    return GetRef<Layout>(this);
  }

  const int ndim = static_cast<int>(InputDim());
  if (ndim <= 0) {
    TVM_FFI_THROW(ValueError) << "Cannot repeat a 0-dim layout";
  }
  int normalized_dim = dim;
  if (normalized_dim < 0) {
    normalized_dim += ndim;
  }
  if (normalized_dim < 0 || normalized_dim >= ndim) {
    TVM_FFI_THROW(ValueError)
        << "dim out of range: dim=" << dim << ", ndim=" << ndim;
  }

  Array<PrimExpr> new_input_size = input_size_;
  PrimExpr extent_dim = input_size_[normalized_dim];
  new_input_size.Set(normalized_dim, extent_dim * IntImm::Int32(factor));

  Map<PrimVar, PrimExpr> vmap;
  vmap.Set(InputPlaceholder(normalized_dim),
           FloorMod(InputPlaceholder(normalized_dim), extent_dim));

  Array<PrimExpr> new_forward_index;
  new_forward_index.reserve(OutputDim() + 1);
  new_forward_index.push_back(
      FloorDiv(InputPlaceholder(normalized_dim), extent_dim));
  for (const auto &e : forward_index_) {
    new_forward_index.push_back(SubstituteLayoutVars(e, vmap));
  }

  return Layout(new_input_size, new_forward_index);
}

Layout LayoutNode::Expand(const Array<PrimExpr> &leading_shape) const {
  if (leading_shape.empty()) {
    return GetRef<Layout>(this);
  }

  for (size_t i = 0; i < leading_shape.size(); ++i) {
    if (auto imm = leading_shape[i].as<IntImm>()) {
      if ((*imm)->value <= 0) {
        TVM_FFI_THROW(ValueError)
            << "leading_shape[" << i << "] must be > 0, got " << (*imm)->value;
      }
    }
  }

  const size_t offset = leading_shape.size();

  Array<PrimExpr> new_input_size;
  new_input_size.reserve(offset + InputDim());
  for (const auto &s : leading_shape) {
    new_input_size.push_back(s);
  }
  for (const auto &s : input_size_) {
    new_input_size.push_back(s);
  }

  Map<PrimVar, PrimExpr> vmap;
  for (size_t i = 0; i < InputDim(); ++i) {
    vmap.Set(InputPlaceholder(i), InputPlaceholder(i + offset));
  }

  Array<PrimExpr> new_forward_index;
  new_forward_index.reserve(offset + OutputDim());
  for (size_t i = 0; i < offset; ++i) {
    new_forward_index.push_back(InputPlaceholder(i));
  }
  for (const auto &e : forward_index_) {
    new_forward_index.push_back(SubstituteLayoutVars(e, vmap));
  }

  return Layout(new_input_size, new_forward_index);
}

Fragment FragmentNode::Repeat(const Array<PrimExpr> &repeats,
                              bool repeat_on_thread,
                              bool lower_dim_first) const {
  ICHECK_EQ(repeats.size(), InputDim());
  Array<PrimExpr> new_input_size;
  Map<PrimVar, PrimExpr> vmap;
  for (size_t i = 0; i < InputDim(); i++) {
    new_input_size.push_back(input_size_[i] * repeats[i]);
    vmap.Set(InputPlaceholder(i),
             FloorMod(InputPlaceholder(i), InputShape()[i]));
  }

  PrimExpr repeats_index = 0, repeat_stride = 1;
  if (lower_dim_first) {
    for (int i = InputDim() - 1; i >= 0; i--) {
      repeats_index +=
          repeat_stride * FloorDiv(InputPlaceholder(i), InputShape()[i]);
      repeat_stride *= repeats[i];
    }
  } else {
    for (size_t i = 0; i < InputDim(); i++) {
      repeats_index +=
          repeat_stride * FloorDiv(InputPlaceholder(i), InputShape()[i]);
      repeat_stride *= repeats[i];
    }
  }

  if (repeat_on_thread) {
    PrimExpr thread_size = ThreadExtent();
    auto new_forward_index = forward_index_.Map(
        [&](const PrimExpr &e) { return SubstituteLayoutVars(e, vmap); });
    auto new_forward_thread = SubstituteLayoutVars(forward_thread_, vmap) +
                              thread_size * repeats_index;
    return Fragment(new_input_size, new_forward_index, new_forward_thread,
                    replicate_size_, std::nullopt);
  } else {
    ICHECK(OutputDim() == 1);
    PrimExpr frag_len = OutputShape()[0];
    Array<PrimExpr> new_forward_index = {
        SubstituteLayoutVars(forward_index_[0], vmap) +
        frag_len * repeats_index};
    PrimExpr new_forward_thread = SubstituteLayoutVars(forward_thread_, vmap);
    return Fragment(new_input_size, new_forward_index, new_forward_thread,
                    replicate_size_, std::nullopt);
  }
}

Fragment FragmentNode::Replicate(int repeats) const {
  ICHECK(repeats >= 1);
  Map<PrimVar, PrimExpr> vmap;
  vmap.Set(ReplicationPlaceholder(),
           FloorMod(ReplicationPlaceholder(), ReplicateExtent()));
  PrimExpr new_forward_thread =
      SubstituteLayoutVars(forward_thread_, vmap) +
      ThreadExtent() * FloorDiv(ReplicationPlaceholder(), ReplicateExtent());
  return Fragment(input_size_, forward_index_, new_forward_thread,
                  ReplicateExtent() * repeats, std::nullopt);
}

Fragment FragmentNode::DeReplicate() const {
  ICHECK(OutputDim() == 1);
  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  UpdateAnalyzer(analyzer.get());
  int factor = 1;
  auto rep_size = TryConstInt(ReplicateExtent());
  auto idx_size = TryConstInt(OutputShape()[0]);
  if (rep_size && idx_size) {
    factor = std::gcd(*rep_size, *idx_size);
  }
  if (factor == 1)
    return GetRef<Fragment>(this);

  Map<PrimVar, PrimExpr> vmap;
  vmap.Set(ReplicationPlaceholder(), ReplicationPlaceholder() * factor +
                                         FloorMod(forward_index_[0], factor));
  PrimExpr new_forward_thread = SubstituteLayoutVars(forward_thread_, vmap);
  Array<PrimExpr> new_forward_index = {FloorDiv(forward_index_[0], factor)};
  return Fragment(input_size_, new_forward_index, new_forward_thread,
                  int(*rep_size) / factor, std::nullopt)
      ->BindThreadRange(Range(0, ThreadExtent()));
}

Fragment
FragmentNode::BindThreadRange(ffi::Optional<Range> thread_range) const {
  auto n = make_object<FragmentNode>(*this);
  n->thread_range_ = thread_range;
  return Fragment(n);
}

std::pair<Layout, sym::IterMapLevel>
LayoutNode::InverseWithLevel(bool require_padding_guard) const {
  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  auto collect_symbolic = [&](const Array<PrimExpr> &shape) {
    Array<PrimExpr> symbolic_dims;
    for (const auto &dim : shape) {
      if (!TryConstInt(dim)) {
        symbolic_dims.push_back(dim);
      }
    }
    return symbolic_dims;
  };
  Array<PrimExpr> symbolic_dims = collect_symbolic(input_size_);
  Array<PrimExpr> output_shape = OutputShape();
  symbolic_dims.insert(symbolic_dims.end(), output_shape.begin(),
                       output_shape.end());
  symbolic_dims = collect_symbolic(symbolic_dims);
  bool is_static_shape = symbolic_dims.empty();
  auto level = (is_static_shape && !require_padding_guard)
                   ? sym::IterMapLevel::Bijective
                   : sym::IterMapLevel::NoCheck;
  if (!is_static_shape) {
    // Runtime guards keep dynamic tails safe, so we allow NoCheck here and
    // warn.
    DLOG(WARNING) << "Layout::Inverse on symbolic layout, falling back to "
                     "NoCheck; symbolic dims: "
                  << symbolic_dims;
  }
  auto iter_map_view = MakeLayoutIterMapView(forward_index_, GetVarMap());
  sym::IterMapResult res = sym::DetectIterMap(
      iter_map_view.indices, iter_map_view.ranges, 1, level, analyzer);
  if (!res->errors.empty()) {
    if (is_static_shape) {
      if (auto inverse = TryCompactStaticInverse(input_size_, output_shape,
                                                 forward_index_))
        return {*inverse, sym::IterMapLevel::Bijective};
    }
    std::ostringstream msg;
    msg << "Layout " << DebugOutput() << " has errors: " << res->errors;
    throw NormalizeIterException(msg.str());
  }

  auto outputs_shape = OutputShape();
  Array<PrimExpr> outputs;
  for (size_t i = 0; i < OutputDim(); i++) {
    outputs.push_back(InputPlaceholder(i));
  }

  auto inv = sym::InverseAffineIterMap(res->indices, outputs);

  Array<PrimExpr> backward_index;
  for (size_t i = 0; i < InputDim(); i++) {
    PrimVar original = InputPlaceholder(i);
    PrimVar coordinate =
        iter_map_view.coordinates.Get(original).value_or(original);
    if (inv.find(coordinate) != inv.end()) {
      backward_index.push_back(inv[coordinate]);
    } else {
      backward_index.push_back(0);
    }
  }

  if (level == sym::IterMapLevel::Bijective) {
    InverseRoundTripResult round_trip = CheckStaticInverseRoundTrip(
        input_size_, outputs_shape, forward_index_, backward_index);
    if (round_trip.status == InverseRoundTripStatus::kInvalid) {
      std::ostringstream msg;
      msg << "Layout " << DebugOutput()
          << " has a non-round-tripping inverse: " << round_trip.detail
          << ". Refusing to use an inferred inverse that changes coordinates.";
      throw NormalizeIterException(msg.str());
    }
  }

  return {Layout(outputs_shape, backward_index), level};
}

Layout LayoutNode::Reshape(const Array<PrimExpr> &shape,
                           sym::AnalyzerObj *analyzer,
                           const PrimExpr rescale_num,
                           const PrimExpr rescale_den) const {

  // Fast path: if shape is the same, return the original layout
  if (StructuralEqual()(InputShape(), shape)) {
    return GetRef<Layout>(this);
  }

  // Step 1. Prove the product relation holds under rescale:
  //   prod(InputShape) * rescale_num == prod(shape) * rescale_den
  PrimExpr input_shape_product = IntImm::Int32(1);
  for (const auto &dim : InputShape()) {
    input_shape_product *= dim;
  }
  PrimExpr shape_product = IntImm::Int32(1);
  for (const auto &dim : shape) {
    shape_product *= dim;
  }

  // Use provided analyzer if present, otherwise a local fallback to avoid
  // potential null dereference paths flagged by static analysis.
  sym::Analyzer fallback_analyzer = CreateTileLangAnalyzer();
  sym::AnalyzerObj *az = analyzer ? analyzer : fallback_analyzer.get();
  ICHECK(az->CanProveEqual(input_shape_product * rescale_num,
                           shape_product * rescale_den))
      << "InputShape() = " << InputShape() << " shape = " << shape
      << ", rescale_num = " << rescale_num << ", rescale_den = " << rescale_den;

  // The generic reshape below only reasons about a flat logical element
  // index, which is exact only when the element width is unchanged.
  // Reinterpreting views are handled first.
  if (auto reinterpreted =
          TryReinterpretReshape(this, shape, az, rescale_num, rescale_den);
      reinterpreted.defined()) {
    return reinterpreted;
  }

  // Step 2. Create new forward indices by reshaping
  Array<PrimVar> new_vars = CreateReshapeVars(shape, az);
  // Step 3. Compute the flat index from new shape indices
  // flat_index = k0 * (s1 * s2 * ...) + k1 * (s2 * s3 * ...) + ... + kn
  PrimExpr flat_index = ComputeFlatIndex(shape, new_vars);
  // Convert new flat index (in units of new elements) to the old flat index
  // (in units of old elements) using the rational rescale factor.
  // old_flat = floor((flat_index * rescale_den) / rescale_num)
  PrimExpr old_flat_index = floordiv(flat_index * rescale_den, rescale_num);
  Array<PrimExpr> original_indices =
      RecoverOriginalIndices(InputShape(), old_flat_index);
  // Step 5. Substitute original indices into forward_index_
  Array<PrimExpr> new_forward_index = SubstituteForwardIndex(
      forward_index_, InputShape(), original_indices, az);
  new_forward_index = RestoreInputPlaceholders(new_forward_index, new_vars);
  return Layout(shape, new_forward_index);
}

Layout FragmentNode::Reshape(const Array<PrimExpr> &shape,
                             sym::AnalyzerObj *analyzer,
                             const PrimExpr rescale_num,
                             const PrimExpr rescale_den) const {

  // Fast path: identical input shape, return self
  if (StructuralEqual()(InputShape(), shape)) {
    return GetRef<Fragment>(this);
  }

  // 1) Prove total number of elements remains the same
  PrimExpr input_prod = IntImm::Int32(1);
  for (const auto &d : InputShape())
    input_prod *= d;
  PrimExpr shape_prod = IntImm::Int32(1);
  for (const auto &d : shape)
    shape_prod *= d;

  // Use provided analyzer if present, otherwise a local fallback.
  sym::Analyzer fallback_analyzer = CreateTileLangAnalyzer();
  sym::AnalyzerObj *az = analyzer ? analyzer : fallback_analyzer.get();
  ICHECK(az->CanProveEqual(input_prod * rescale_num, shape_prod * rescale_den))
      << "InputShape() = " << InputShape() << " shape = " << shape
      << ", rescale_num = " << rescale_num << ", rescale_den = " << rescale_den
      << " input fragment layout is = " << DebugOutput();

  // Fragments need the same special handling as plain layouts so that
  // reinterpreting views keep a stable thread/data mapping through reshape.
  if (auto reinterpreted =
          TryReinterpretReshape(this, shape, az, rescale_num, rescale_den);
      reinterpreted.defined()) {
    return reinterpreted;
  }

  // 2) Build flat index from new-shape indices
  Array<PrimVar> new_vars = CreateReshapeVars(shape, az);
  PrimExpr flat = ComputeFlatIndex(shape, new_vars);
  // Convert to old flat index units using the rational rescale factor.
  // old_flat = floor((flat * rescale_den) / rescale_num)
  PrimExpr old_flat = floordiv(flat * rescale_den, rescale_num);
  // 3) Recover original indices from flat index
  Array<PrimExpr> orig_indices = RecoverOriginalIndices(InputShape(), old_flat);
  // 4) Substitute old placeholders with expressions of new indices
  Array<PrimExpr> new_forward_index =
      SubstituteForwardIndex(forward_index_, InputShape(), orig_indices, az);
  PrimExpr new_forward_thread =
      SubstituteReshapedExpr(forward_thread_, InputShape(), orig_indices, az);
  new_forward_index = RestoreInputPlaceholders(new_forward_index, new_vars);
  new_forward_thread = RestoreInputPlaceholders(new_forward_thread, new_vars);
  Fragment reshaped(shape, new_forward_index, new_forward_thread,
                    ReplicateExtent(), std::nullopt);
  if (thread_range_.has_value()) {
    reshaped = reshaped->BindThreadRange(thread_range_);
  }
  return reshaped;
}

Layout LayoutNode::Inverse() const {
  auto inverse_result = InverseWithLevel();
  return std::move(inverse_result.first);
}

PrimExpr infer_fragment_index(const Map<PrimVar, Range> &input_iters,
                              const PrimExpr &forward_thread) {
  // we build iter_vars from input_iters, but set _rep to range [0, 1)
  // to make it not contribute to the index of the forward_idx
  Array<LayoutIterVar> iter_vars;
  auto analyzer = CreateTileLangAnalyzer();
  for (const auto &[var, range_] : input_iters) {
    Range range = range_;
    if (var.same_as(ReplicationPlaceholder())) {
      range = Range(0, 1);
    }
    iter_vars.push_back(LayoutIterVar(range, var));
    BindAnalyzer(analyzer, var, range);
  }

  // The thread layout uses every replica, while local storage is inferred
  // for replica zero. Simplify with that exact domain before asking the
  // native IterMap normalizer to expose the used digits. Reusing the caller's
  // full-replica bounds can leave an unrepresentable coordinate expression
  // in IterSum.base, incorrectly treating every input digit as unused.
  PrimExpr local_thread = analyzer->Simplify(SubstituteLayoutVars(
      forward_thread, {{ReplicationPlaceholder(), IntImm::Int32(0)}}));
  Array<sym::IterSplitExpr> splits =
      DivideUnusedIterators({local_thread}, iter_vars, analyzer.get());
  return MakeFlattenedExpression(splits);
}

FragmentNode::FragmentNode(Array<PrimExpr> input_size,
                           Array<PrimExpr> forward_index,
                           PrimExpr forward_thread, PrimExpr replicate_size) {
  input_size_ = input_size;
  replicate_size_ = replicate_size;
  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  UpdateAnalyzer(analyzer.get());
  forward_thread_ = analyzer->Simplify(forward_thread);
  if (forward_index.empty()) {
    forward_index = {infer_fragment_index(GetVarMap(), forward_thread_)};
  }
  forward_index_ = forward_index.Map(
      [&](const PrimExpr &e) { return analyzer->Simplify(e); });
}

Fragment::Fragment(Array<LayoutIterVar> forward_var,
                   Array<PrimExpr> forward_index, PrimExpr forward_thread,
                   LayoutIterVar thread_replicate) {
  Map<PrimVar, PrimExpr> vmap;
  Array<PrimExpr> input_size;
  PrimExpr replicate_size = 1;
  for (size_t i = 0; i < forward_var.size(); i++) {
    vmap.Set(forward_var[i]->var, InputPlaceholder(i));
    ICHECK(IsZero(forward_var[i]->dom->begin));
    input_size.push_back(forward_var[i]->dom->extent);
  }
  if (thread_replicate.defined()) {
    ICHECK(IsZero(thread_replicate->dom->begin));
    replicate_size = thread_replicate->dom->extent;
    vmap.Set(thread_replicate->var, ReplicationPlaceholder());
  }
  forward_index = forward_index.Map(
      [&](const PrimExpr &e) { return SubstituteLayoutVars(e, vmap); });
  forward_thread = SubstituteLayoutVars(forward_thread, vmap);

  auto n = make_object<FragmentNode>(input_size, forward_index, forward_thread,
                                     replicate_size);
  data_ = std::move(n);
}

Fragment::Fragment(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
                   PrimExpr forward_thread, PrimExpr replicate_size,
                   Optional<PrimVar> replicate_var) {
  if (replicate_var.has_value()) {
    forward_thread = SubstituteLayoutVars(
        forward_thread, {{replicate_var.value(), ReplicationPlaceholder()}});
  }
  auto n = make_object<FragmentNode>(input_size, forward_index, forward_thread,
                                     replicate_size);
  data_ = std::move(n);
}

Fragment Fragment::FullyReplicated(Array<PrimExpr> shape,
                                   PrimExpr thread_extent) {
  return Fragment(shape, {}, ReplicationPlaceholder(), thread_extent,
                  std::nullopt)
      ->BindThreadRange(Range(0, thread_extent));
}

PartialFragmentNode::PartialFragmentNode(Array<PrimExpr> input_size,
                                         Array<PrimExpr> forward_index,
                                         PrimExpr forward_thread,
                                         PrimExpr replicate_size,
                                         PrimExpr combine_size,
                                         Optional<Range> thread_range)
    : FragmentNode(std::move(input_size), std::move(forward_index),
                   std::move(forward_thread), std::move(replicate_size)) {
  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  combine_size_ = analyzer->Simplify(combine_size);
  // The low-bits convention only makes sense when the combine width evenly
  // tiles the replication coordinate.
  auto rep = TryConstInt(replicate_size_);
  auto comb = TryConstInt(combine_size_);
  if (rep.has_value() && comb.has_value()) {
    ICHECK(*comb >= 1 && *rep % *comb == 0)
        << "PartialFragment: combine width " << *comb
        << " must evenly divide the replication extent " << *rep;
  }
  // BindThreadRange copies into a plain FragmentNode and would drop the
  // partial kind, so the range is fixed at construction.
  if (thread_range.has_value()) {
    thread_range_ = thread_range.value();
  }
}

PartialFragment::PartialFragment(Array<PrimExpr> input_size,
                                 Array<PrimExpr> forward_index,
                                 PrimExpr forward_thread,
                                 PrimExpr replicate_size, PrimExpr combine_size,
                                 Optional<PrimVar> replicate_var,
                                 Optional<Range> thread_range) {
  if (replicate_var.has_value()) {
    forward_thread = SubstituteLayoutVars(
        forward_thread, {{replicate_var.value(), ReplicationPlaceholder()}});
  }
  data_ = make_object<PartialFragmentNode>(input_size, forward_index,
                                           forward_thread, replicate_size,
                                           combine_size, thread_range);
}

PartialFragment PartialFragment::FromFragment(const Fragment &fragment) {
  // Annotation semantics: every declared replica is an addend lane; copy
  // groups only ever come from loop replication, which a user-constructed
  // partial does not carry.
  const FragmentNode *node = fragment.get();
  ICHECK(node != nullptr) << "PartialFragment::FromFragment: null fragment";
  return PartialFragment::FromInduced(fragment, node->ReplicateExtent());
}

PartialFragment PartialFragment::FromInduced(const Fragment &fragment,
                                             PrimExpr combine_size) {
  const FragmentNode *node = fragment.get();
  ICHECK(node != nullptr) << "PartialFragment::FromInduced: null fragment";
  Optional<Range> thread_range;
  if (node->ThreadRange().has_value()) {
    thread_range = node->ThreadRange();
  }
  return PartialFragment(node->InputShape(), node->GetForwardIndex(),
                         node->GetForwardThread(), node->ReplicateExtent(),
                         combine_size, std::nullopt, thread_range);
}

PartialFragment PartialFragment::FullyReplicated(Array<PrimExpr> shape,
                                                 PrimExpr thread_extent,
                                                 Optional<Range> thread_range) {
  if (!thread_range.has_value()) {
    thread_range = Range(0, thread_extent);
  }
  return PartialFragment(shape, {}, ReplicationPlaceholder(), thread_extent,
                         /*combine_size=*/thread_extent, std::nullopt,
                         thread_range);
}

Fragment PartialFragment::AsPostCollective() const {
  const PartialFragmentNode *node = get();
  ICHECK(node != nullptr) << "PartialFragment::AsPostCollective: null partial";
  Fragment ret(node->InputShape(), node->GetForwardIndex(),
               node->GetForwardThread(), node->ReplicateExtent(), std::nullopt);
  if (node->ThreadRange().has_value()) {
    ret = ret->BindThreadRange(node->ThreadRange());
  }
  return ret;
}

std::string PartialFragmentNode::DebugOutput() const {
  std::stringstream ss;
  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  ss << "PartialFragment(" << InputShape() << " -> " << OutputShape()
     << ", replicate: " << ReplicateExtent() << " (combine " << combine_size_
     << " x copy "
     << analyzer->Simplify(FloorDiv(ReplicateExtent(), combine_size_)) << ")"
     << ", thread: " << ThreadExtent()
     << ", forward_thread: " << forward_thread_
     << ", forward_index: " << GetForwardIndex();
  if (thread_range_.has_value()) {
    ss << ", thread_range: " << thread_range_;
  }
  ss << ")";
  return ss.str();
}

bool PartialFragmentNode::IsEqual(const LayoutNode *other,
                                  bool skip_index) const {
  if (other == nullptr || !other->IsInstance<PartialFragmentNode>()) {
    return false;
  }
  const auto *partial = static_cast<const PartialFragmentNode *>(other);
  // Same storage algebra with a different combine decomposition is a
  // DIFFERENT physical plan (e.g. 8 addend lanes x 16 copy groups vs the
  // FullParticipant 128): the decomposition is part of the identity.
  if (!StructuralEqual()(combine_size_, partial->combine_size_)) {
    return false;
  }
  return FragmentNode::IsEqual(partial, skip_index);
}

std::vector<std::pair<int, int>> PartialFragmentNode::CombineSteps() const {
  // Split `_rep = lo + combine_size * hi` and collect the thread-expression
  // splits sourced from `lo`: each one is a butterfly of `extent` lanes at
  // thread stride `scale` — the complete communication spec of the finalize
  // collective. Mirrors backend CollectThreadReduceSteps, which cannot be
  // used here for layering reasons.
  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  PrimExpr rep_extent = ReplicateExtent();
  PrimExpr copy_extent =
      analyzer->Simplify(FloorDiv(rep_extent, combine_size_));
  PrimVar lo("_rep_lo", ReplicationPlaceholder().ty());
  PrimVar hi("_rep_hi", ReplicationPlaceholder().ty());
  PrimExpr thread = SubstituteLayoutVars(
      forward_thread_, {{ReplicationPlaceholder(), lo + combine_size_ * hi}});
  Map<PrimVar, Range> vmap;
  for (size_t i = 0; i < InputDim(); ++i) {
    vmap.Set(
        InputPlaceholder(i),
        Range::FromMinExtent(MakeConst(PrimType::Int(32), 0), InputShape()[i]));
  }
  vmap.Set(
      lo, Range::FromMinExtent(MakeConst(PrimType::Int(32), 0), combine_size_));
  vmap.Set(hi,
           Range::FromMinExtent(MakeConst(PrimType::Int(32), 0), copy_extent));
  sym::IterSumExpr sum = sym::NormalizeToIterSum(thread, vmap, analyzer);
  std::vector<std::pair<int, int>> steps;
  for (const auto &split : sum->args) {
    auto mark = split->source->source.as<PrimVar>();
    if (!mark || !mark.value().same_as(lo)) {
      continue;
    }
    auto scale = TryConstInt(split->scale);
    auto extent = TryConstInt(split->extent);
    ICHECK(scale.has_value() && extent.has_value())
        << "PartialFragment::CombineSteps requires constant extents: "
        << DebugOutput();
    if (*extent == 1) {
      continue;
    }
    // (reducing_threads, scale), matching AnalyzeReducerUpdateSite's
    // convention: reducing_threads = extent * scale.
    steps.emplace_back(static_cast<int>(*extent * *scale),
                       static_cast<int>(*scale));
  }
  return steps;
}

// which means the forward_thread is rep_var -> lambda i, rep: rep
bool FragmentNode::IsCompletedReplicated() const {
  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  return ExprDeepEqual()(analyzer->Simplify(forward_thread_),
                         ReplicationPlaceholder());
}

sym::IterMapResult
LayoutNode::DetectInjective(bool require_padding_guard) const {
  return DetectInjectiveMapping(forward_index_, GetVarMap(),
                                require_padding_guard);
}

sym::IterMapResult
FragmentNode::DetectInjective(bool require_padding_guard) const {
  // A fragment's physical identity includes its owning thread as well as its
  // per-thread indices. The replication coordinate is part of GetVarMap().
  Array<PrimExpr> indices{forward_thread_};
  indices.insert(indices.end(), forward_index_.begin(), forward_index_.end());
  return DetectInjectiveMapping(indices, GetVarMap(), require_padding_guard);
}

PrimExpr FragmentNode::ThreadExtent() const {
  Array<PrimExpr> ret(OutputDim(), 1);
  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  UpdateAnalyzer(analyzer.get());
  Map<Var, sym::IntSet> domains;
  for (const auto &[var, range] : GetVarMap())
    domains.Set(var, sym::IntSet::FromRange(range));
  auto ist = analyzer->int_set(forward_thread_ + 1, domains);
  return ist.max();
}

Array<PrimExpr> FragmentNode::GetForwardVars() const {
  Array<PrimExpr> vars;
  if (*TryConstInt(ReplicateExtent()) > 1) {
    vars.push_back(ReplicationPlaceholder());
  }
  for (size_t i = 0; i < InputDim(); i++) {
    vars.push_back(InputPlaceholder(i));
  }
  return vars;
}

PrimExpr FragmentNode::ForwardThread(const Array<PrimExpr> &vars,
                                     const Optional<PrimExpr> &rep_var) const {
  Map<PrimVar, PrimExpr> vmap;
  ICHECK_EQ(vars.size(), InputDim());
  for (size_t i = 0; i < InputDim(); i++) {
    vmap.Set(InputPlaceholder(i), vars[i]);
  }
  if (rep_var.has_value())
    vmap.Set(ReplicationPlaceholder(), rep_var.value());

  return SubstituteLayoutVars(forward_thread_, vmap);
}

Layout FragmentNode::Inverse() const {
  auto result = InverseWithLevel();
  return std::move(result.first);
}

std::pair<Layout, sym::IterMapLevel>
FragmentNode::InverseWithLevel(bool require_padding_guard) const {
  auto input_size_copy = input_size_;
  input_size_copy.push_back(ReplicateExtent());
  auto forward_index_copy = forward_index_;
  forward_index_copy.push_back(SubstituteLayoutVars(
      forward_thread_,
      {{ReplicationPlaceholder(), InputPlaceholder(InputDim())}}));
  auto fwd = Layout(input_size_copy, forward_index_copy);
  return fwd->InverseWithLevel(require_padding_guard);
}

Fragment FragmentNode::CondenseReplicateVar() const {
  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  auto input_iters = GetVarMap();
  input_iters.Set(ReplicationPlaceholder(), {0, ReplicateExtent()});
  PrimExpr new_forward_thread{0};
  LayoutIterVar new_thread_replicate;
  std::tie(new_forward_thread, new_thread_replicate) =
      CompressIterator(forward_thread_, ToIterVars(input_iters),
                       ReplicationPlaceholder(), analyzer.get());
  return Fragment(input_size_, forward_index_, new_forward_thread,
                  new_thread_replicate->dom->extent, new_thread_replicate->var);
}

std::pair<Fragment, PrimExpr>
CondenseReplicateVarKeepingBoundary(const Fragment &fragment,
                                    const PrimExpr &combine_size) {
  sym::Analyzer analyzer = CreateTileLangAnalyzer();
  const FragmentNode *node = fragment.get();
  ICHECK(node != nullptr);
  PrimExpr rep_extent = node->ReplicateExtent();
  PrimExpr copy_extent = analyzer->Simplify(FloorDiv(rep_extent, combine_size));
  PrimVar lo("_rep_lo", ReplicationPlaceholder().ty());
  PrimVar hi("_rep_hi", ReplicationPlaceholder().ty());
  PrimExpr thread = SubstituteLayoutVars(
      node->GetForwardThread(),
      {{ReplicationPlaceholder(), lo + combine_size * hi}});
  Map<PrimVar, Range> vmap;
  for (size_t i = 0; i < node->InputDim(); ++i) {
    vmap.Set(InputPlaceholder(i),
             Range::FromMinExtent(MakeConst(PrimType::Int(32), 0),
                                  node->InputShape()[i]));
  }
  vmap.Set(lo,
           Range::FromMinExtent(MakeConst(PrimType::Int(32), 0), combine_size));
  vmap.Set(hi,
           Range::FromMinExtent(MakeConst(PrimType::Int(32), 0), copy_extent));
  // The projection may have collapsed the two halves into a single
  // expression over `_rep` (e.g. `(lo + W*hi) // c` from a per-thread
  // serial run under loop replication). Re-simplify with the ranges bound
  // so the halves separate again; otherwise NormalizeToIterSum fuses
  // `lo + W*hi` into one iterator and CompressIterator, which attributes
  // splits per source var, cannot compress either half.
  for (const auto &[var, range] : vmap) {
    BindAnalyzer(analyzer, var, range);
  }
  thread = analyzer->Simplify(thread);
  // Compress the combine half first, then the copy half; each keeps only
  // the sub-iterators that actually reach the thread expression.
  auto [expr_lo, lo_iv] =
      CompressIterator(thread, ToIterVars(vmap), lo, analyzer.get());
  vmap.erase(lo);
  vmap.Set(lo_iv->var, lo_iv->dom);
  auto [expr_hi, hi_iv] =
      CompressIterator(expr_lo, ToIterVars(vmap), hi, analyzer.get());
  PrimExpr w = analyzer->Simplify(lo_iv->dom->extent);
  PrimExpr c = analyzer->Simplify(hi_iv->dom->extent);
  PrimExpr new_thread = SubstituteLayoutVars(
      expr_hi, {{lo_iv->var, FloorMod(ReplicationPlaceholder(), w)},
                {hi_iv->var, FloorDiv(ReplicationPlaceholder(), w)}});
  Fragment out(node->InputShape(), node->GetForwardIndex(), new_thread,
               analyzer->Simplify(w * c), std::nullopt);
  if (node->ThreadRange().has_value()) {
    out = out->BindThreadRange(node->ThreadRange());
  }
  return {out, w};
}

std::string LayoutNode::DebugOutput() const {
  std::stringstream ss;
  ss << "Layout(" << InputShape() << " -> " << OutputShape()
     << ", transform: " << GetForwardVars() << " -> " << GetForwardIndex()
     << ")";
  return ss.str();
}

std::string FragmentNode::DebugOutput() const {
  std::stringstream ss;
  ss << "Fragment(" << InputShape() << " -> " << OutputShape()
     << ", replicate: " << ReplicateExtent() << ", thread: " << ThreadExtent()
     << ", forward_thread: " << forward_thread_
     << ", forward_index: " << GetForwardIndex();
  if (thread_range_.has_value()) {
    ss << ", thread_range: " << thread_range_;
  }
  ss << ")";
  return ss.str();
}

bool LayoutNode::IsEqual(const LayoutNode *other, bool skip_index) const {
  bool ret = StructuralEqual()(this->InputShape(), other->InputShape());
  ret &= StructuralEqual()(this->OutputShape(), other->OutputShape());
  if (!ret) {
    return false;
  }
  if (!skip_index) {
    // Create common variables for comparison. Using Forward with common
    // variables ensures we compare the actual mapping rather than AST
    // structure, since InputPlaceholder may compare equal in StructuralEqual.
    Array<PrimExpr> common_vars;
    for (size_t i = 0; i < this->InputDim(); i++) {
      common_vars.push_back(PrimVar("_cmp_v" + std::to_string(i)));
    }

    auto this_forward = this->Forward(common_vars);
    auto other_forward = other->Forward(common_vars);

    if (!StructuralEqual()(this_forward, other_forward)) {
      return false;
    }
  }
  return ret;
}

bool FragmentNode::IsEqual(const LayoutNode *other, bool skip_index) const {
  if (other != nullptr && other->IsInstance<PartialFragmentNode>()) {
    return false;
  }
  return LayoutNode::IsEqual(other, skip_index);
}

bool FragmentNode::IsEqual(const FragmentNode *other, bool skip_index) const {
  // This overload is used by Fragment_is_equal and by C++ callers holding
  // Fragment references. It must enforce partial semantics too: overload
  // resolution otherwise bypasses PartialFragmentNode's LayoutNode override.
  bool lhs_partial = IsInstance<PartialFragmentNode>();
  bool rhs_partial =
      other != nullptr && other->IsInstance<PartialFragmentNode>();
  if (other == nullptr || lhs_partial != rhs_partial) {
    return false;
  }
  if (lhs_partial) {
    const auto *lhs = static_cast<const PartialFragmentNode *>(this);
    const auto *rhs = static_cast<const PartialFragmentNode *>(other);
    if (!StructuralEqual()(lhs->CombineSize(), rhs->CombineSize()) ||
        !StructuralEqual()(lhs->InputShape(), rhs->InputShape())) {
      return false;
    }
  }
  // Fragment Layout Comparison can skip the index comparison
  // when the output shape is the same, as we can do
  // a[i, j] = b[j, i] in register level.

  bool ret = StructuralEqual()(this->InputShape(), other->InputShape());
  if (!ret) {
    // may be broadcast case
    return true;
  }
  if (this->thread_range_.has_value() && other->thread_range_.has_value()) {
    ret &= StructuralEqual()(this->thread_range_, other->thread_range_);
  }
  ret &= StructuralEqual()(this->OutputShape(), other->OutputShape());
  ret &= StructuralEqual()(this->ReplicateExtent(), other->ReplicateExtent());
  ret &= StructuralEqual()(this->ThreadExtent(), other->ThreadExtent());
  if (!ret) {
    return false;
  }
  if (!skip_index) {
    // Create common variables for comparison. Using Forward/ForwardThread with
    // common variables ensures we compare the actual mapping rather than AST
    // structure, since InputPlaceholder may compare equal in StructuralEqual.
    Array<PrimExpr> common_vars;
    for (size_t i = 0; i < this->InputDim(); i++) {
      common_vars.push_back(PrimVar("_cmp_v" + std::to_string(i)));
    }
    PrimVar common_rep("_cmp_rep");

    auto this_forward = this->Forward(common_vars);
    auto other_forward = other->Forward(common_vars);

    if (!StructuralEqual()(this_forward, other_forward)) {
      return false;
    }

    // Also compare forward_thread mapping.
    auto this_thread = this->ForwardThread(common_vars, common_rep);
    auto other_thread = other->ForwardThread(common_vars, common_rep);
    if (!StructuralEqual()(this_thread, other_thread)) {
      return false;
    }
  }
  return ret;
}

void FragmentNode::RegisterReflection() {
  namespace refl = reflection;
  refl::ObjectDef<FragmentNode>()
      .def_ro("forward_thread", &FragmentNode::forward_thread_)
      .def_ro("replicate_size", &FragmentNode::replicate_size_)
      .def_ro("thread_range", &FragmentNode::thread_range_)
      .def("_DebugOutput", &FragmentNode::DebugOutput);
}

void PartialFragmentNode::RegisterReflection() {
  namespace refl = reflection;
  // Base fields and _DebugOutput are inherited from FragmentNode's
  // reflection (DebugOutput dispatches virtually).
  refl::ObjectDef<PartialFragmentNode>().def_ro(
      "combine_size", &PartialFragmentNode::combine_size_);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  LayoutIterVarNode::RegisterReflection();
  refl::GlobalDef()
      .def("tl.LayoutIterVar",
           [](Range dom, PrimVar var) { return LayoutIterVar(dom, var); })
      .def_packed("tl.Layout",
                  [](PackedArgs args, Any *rv) {
                    *rv = Layout(args[0].cast<Array<LayoutIterVar>>(),
                                 args[1].cast<Array<PrimExpr>>());
                  })
      .def("tl.Layout_input_shape",
           [](Layout layout) { return layout->InputShape(); })
      .def("tl.Layout_output_shape",
           [](Layout layout) { return layout->OutputShape(); })
      .def("tl.Layout_inverse", [](Layout layout) { return layout->Inverse(); })
      .def("tl.Layout_reshape",
           [](Layout layout, Array<PrimExpr> shape, PrimExpr rescale_num,
              PrimExpr rescale_den) {
             return layout->Reshape(shape, nullptr, rescale_num, rescale_den);
           })
      .def("tl.Layout_index",
           [](Layout layout) { return layout->GetForwardIndex(); })
      .def("tl.Layout_linearized_index",
           [](Layout layout) { return layout->GetLinearizedForwardIndex(); })
      .def("tl.Layout_map_indices", MapLayoutIndices)
      .def("tl.Layout_forward_vars",
           [](Layout layout) { return layout->GetForwardVars(); })
      .def("tl.Layout_repeat",
           [](Layout layout, int dim, int factor) {
             return layout->Repeat(dim, factor);
           })
      .def("tl.Layout_expand",
           [](Layout layout, Array<PrimExpr> leading_shape) {
             return layout->Expand(leading_shape);
           })
      .def("tl.Layout_is_equal",
           [](Layout layout, Layout other) {
             const LayoutNode *other_node = other.as<LayoutNode>();
             return layout->IsEqual(other_node);
           })
      .def("tl.Layout_map_region_bounds",
           [](Layout layout, Array<Range> region) {
             return layout->MapRegionBounds(region);
           })
      .def_packed("tl.Fragment",
                  [](PackedArgs args, Any *rv) {
                    *rv = Fragment(
                        /*forward_var=*/args[0].cast<Array<LayoutIterVar>>(),
                        /*forward_index=*/args[1].cast<Array<PrimExpr>>(),
                        /*forward_thread=*/args[2].cast<PrimExpr>(),
                        /*thread_replicate=*/args[3].cast<LayoutIterVar>());
                  })
      .def("tl.Fragment_is_equal",
           [](Fragment fragment, Fragment other) {
             const FragmentNode *other_node = other.as<FragmentNode>();
             return fragment->IsEqual(other_node);
           })
      .def("tl.Fragment_thread_size",
           [](Fragment fragment) { return fragment->ThreadExtent(); })
      .def("tl.Fragment_thread",
           [](Fragment fragment) { return fragment->GetForwardThread(); })
      .def("tl.Fragment_repeat",
           [](Fragment fragment, Array<PrimExpr> repeats, bool repeat_on_thread,
              bool lower_dim_first) {
             return fragment->Repeat(repeats, repeat_on_thread,
                                     lower_dim_first);
           })
      .def("tl.Fragment_replicate",
           [](Fragment fragment, int repeats) {
             return fragment->Replicate(repeats);
           })
      .def("tl.Fragment_condense_rep_var",
           [](Fragment fragment) { return fragment->CondenseReplicateVar(); })
      .def("tl.make_swizzled_layout",
           [](const TensorVar &buffer, bool k_inner, bool allow_pad) {
             return MakeSwizzledLayout(buffer, k_inner, allow_pad);
           })
      .def("tl.make_volta_swizzled_layout",
           [](const TensorVar &buffer, bool is_a, bool k_inner) {
             return MakeVoltaSwizzledLayout(buffer, is_a, k_inner);
           })
      .def("tl.make_wgmma_swizzled_layout",
           [](const TensorVar &buffer, int continuity, bool k_inner) {
             return MakeWgmmaSwizzledLayout(buffer, continuity, k_inner);
           })
      .def("tl.make_tcgen05mma_swizzled_layout",
           [](const TensorVar &buffer, int continuity, bool k_inner) {
             return MakeTcgen05MmaSwizzledLayout(buffer, continuity, k_inner);
           })
      .def("tl.make_full_bank_swizzled_layout",
           [](const TensorVar &buffer) {
             return MakeFullBankSwizzleLayout(buffer);
           })
      .def("tl.make_half_bank_swizzled_layout",
           [](const TensorVar &buffer) {
             return MakeHalfBankSwizzleLayout(buffer);
           })
      .def("tl.make_quarter_bank_swizzled_layout",
           [](const TensorVar &buffer) {
             return MakeQuarterBankSwizzleLayout(buffer);
           })
      .def("tl.make_linear_layout",
           [](Array<PrimExpr> shape) { return MakeLinearLayout(shape); })
      .def("tl.make_gemm_fragment_8x8", []() { return MakeGemmFragment8x8(); })
      .def("tl.make_gemm_fragment_8x8_transposed",
           []() { return MakeGemmFragment8x8Transposed(); })
      .def("tl.make_fully_replicated_layout_fragment",
           [](Array<PrimExpr> shape, PrimExpr thread_extent) {
             return Fragment::FullyReplicated(shape, thread_extent);
           })
      .def_packed("tl.PartialFragment",
                  [](PackedArgs args, Any *rv) {
                    // Same argument shape as tl.Fragment plus a trailing
                    // combine width; the LayoutIterVar-form Fragment ctor does
                    // the placeholder substitution, then the result is
                    // reinterpreted as per-thread partials. `combine` is the
                    // number of addend lanes in the low bits of the
                    // replication coordinate; when absent, every declared
                    // replica is an addend lane.
                    Fragment fragment(
                        /*forward_var=*/args[0].cast<Array<LayoutIterVar>>(),
                        /*forward_index=*/args[1].cast<Array<PrimExpr>>(),
                        /*forward_thread=*/args[2].cast<PrimExpr>(),
                        /*thread_replicate=*/args[3].cast<LayoutIterVar>());
                    if (args.size() <= 4) {
                      *rv = PartialFragment::FromFragment(fragment);
                      return;
                    }
                    int64_t combine = args[4].cast<int64_t>();
                    auto rep = TryConstInt(fragment->ReplicateExtent());
                    if (combine < 1 ||
                        (rep.has_value() && *rep % combine != 0)) {
                      TVM_FFI_THROW(ValueError)
                          << "PartialFragment: combine (" << combine
                          << ") must be >= 1 and evenly divide replicate ("
                          << fragment->ReplicateExtent() << ")";
                    }
                    *rv = PartialFragment::FromInduced(
                        fragment, IntImm(PrimType::Int(32), combine));
                  })
      .def("tl.PartialFragment_from_fragment",
           [](Fragment fragment) {
             return PartialFragment::FromFragment(fragment);
           })
      .def("tl.PartialFragment_as_post_collective",
           [](PartialFragment partial) { return partial.AsPostCollective(); })
      .def("tl.PartialFragment_combine_steps",
           [](PartialFragment partial) {
             Array<Array<IntImm>> result;
             for (const auto &[reducing_threads, scale] :
                  partial->CombineSteps()) {
               result.push_back({IntImm(PrimType::Int(32), reducing_threads),
                                 IntImm(PrimType::Int(32), scale)});
             }
             return result;
           })
      .def("tl.make_fully_replicated_partial_fragment",
           [](Array<PrimExpr> shape, PrimExpr thread_extent) {
             return PartialFragment::FullyReplicated(shape, thread_extent);
           });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  LayoutNode::RegisterReflection();
  FragmentNode::RegisterReflection();
  PartialFragmentNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
