#include "compat/analyzer.h"
/*!
 * \file layout/layout.cc
 *
 */

#include "layout.h"

#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "support/check.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/runtime/logging.h>

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/stmt_functor.h>

#include "printer.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

Array<PrimVar> CreateReshapeVars(const Array<PrimExpr> &shape,
                                 sym::Analyzer analyzer) {
  Array<PrimVar> vars;
  vars.reserve(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    auto var = PrimVar(std::string("n_") + std::to_string(i), shape[i].ty());
    tl::arith::Bind(analyzer, var, Range(0, shape[i]));
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
                                       sym::Analyzer analyzer) {
  Array<PrimExpr> new_forward_index;
  for (const auto &fwd_expr : forward_index) {
    PrimExpr substituted = fwd_expr;
    for (size_t i = 0; i < input_shape.size(); ++i) {
      substituted =
          Substitute(substituted, {{InputPlaceholder(i), original_indices[i]}});
    }
    new_forward_index.push_back(tl::arith::Simplify(analyzer, substituted));
  }
  return new_forward_index;
}

PrimExpr SubstituteReshapedExpr(const PrimExpr &expr,
                                const Array<PrimExpr> &input_shape,
                                const Array<PrimExpr> &original_indices,
                                sym::Analyzer analyzer) {
  PrimExpr substituted = expr;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    substituted =
        Substitute(substituted, {{InputPlaceholder(i), original_indices[i]}});
  }
  return tl::arith::Simplify(analyzer, substituted);
}

Array<PrimExpr> RestoreInputPlaceholders(const Array<PrimExpr> &forward_index,
                                         const Array<PrimVar> &vars) {
  Array<PrimExpr> restored = forward_index;
  for (size_t i = 0; i < vars.size(); ++i) {
    restored = Substitute(restored, {{vars[i], InputPlaceholder(i)}});
  }
  return restored;
}

PrimExpr RestoreInputPlaceholders(const PrimExpr &expr,
                                  const Array<PrimVar> &vars) {
  PrimExpr restored = expr;
  for (size_t i = 0; i < vars.size(); ++i) {
    restored = Substitute(restored, {{vars[i], InputPlaceholder(i)}});
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
                             sym::Analyzer analyzer,
                             const PrimExpr &old_elem_bits_expr,
                             const PrimExpr &new_elem_bits_expr) {
  auto old_elem_bits = GetConstInt(old_elem_bits_expr);
  auto new_elem_bits = GetConstInt(new_elem_bits_expr);
  if (!old_elem_bits.has_value() || !new_elem_bits.has_value() ||
      *old_elem_bits == *new_elem_bits) {
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
        tl::arith::Simplify(analyzer,
                            last * IntImm::Int32(pack_factor) + lane_in_pack));
    new_forward_index = RestoreInputPlaceholders(new_forward_index, new_vars);
    return Layout(shape, new_forward_index);
  } else { // *old_elem_bits < *new_elem_bits
    ICHECK_EQ(*new_elem_bits % *old_elem_bits, 0)
        << "cannot reinterpret " << *old_elem_bits << "-bit elements as "
        << *new_elem_bits << "-bit elements";
    int64_t pack_factor = *new_elem_bits / *old_elem_bits;

    PrimVar lane("_lane", flat_index.ty());
    tl::arith::Bind(analyzer, lane, Range(0, IntImm::Int32(pack_factor)));
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
    bool compatible = tl::arith::CanProveEqual(
        analyzer, probe.back(), slot * IntImm::Int32(pack_factor) + lane);
    for (size_t i = 0; compatible && i + 1 < base.size(); ++i) {
      compatible = tl::arith::CanProveEqual(analyzer, probe[i], base[i]);
    }
    ICHECK(compatible)
        << "cannot reinterpret layout " << layout_node->DebugOutput() << " as "
        << *new_elem_bits << "-bit elements: aliasing buffers must induce the "
        << "same storage layout, and this one interleaves storage below the "
        << "new element width";
    Array<PrimExpr> new_forward_index = base;
    new_forward_index.Set(new_forward_index.size() - 1,
                          tl::arith::Simplify(analyzer, slot));
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
                               sym::Analyzer analyzer,
                               const PrimExpr &old_elem_bits_expr,
                               const PrimExpr &new_elem_bits_expr) {
  auto old_elem_bits = GetConstInt(old_elem_bits_expr);
  auto new_elem_bits = GetConstInt(new_elem_bits_expr);
  if (!old_elem_bits.has_value() || !new_elem_bits.has_value() ||
      *old_elem_bits == *new_elem_bits) {
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
        tl::arith::Simplify(analyzer,
                            last * IntImm::Int32(pack_factor) + lane_in_pack));
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
    tl::arith::Bind(analyzer, lane, Range(0, IntImm::Int32(pack_factor)));
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
        tl::arith::CanProveEqual(analyzer, probe.back(),
                                 slot * IntImm::Int32(pack_factor) + lane) &&
        tl::arith::CanProveEqual(analyzer, thread_probe, thread_base);
    for (size_t i = 0; compatible && i + 1 < base.size(); ++i) {
      compatible = tl::arith::CanProveEqual(analyzer, probe[i], base[i]);
    }
    ICHECK(compatible)
        << "cannot reinterpret fragment " << fragment_node->DebugOutput()
        << " as " << *new_elem_bits << "-bit elements: aliasing buffers must "
        << "induce the same storage layout, and this one interleaves storage "
        << "or threads below the new element width";
    Array<PrimExpr> new_forward_index = base;
    new_forward_index.Set(new_forward_index.size() - 1,
                          tl::arith::Simplify(analyzer, slot));
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
  sym::Analyzer analyzer;
  std::vector<PrimVar> vars;
  std::vector<int64_t> mins;
  std::vector<int64_t> extents;
  int64_t domain_size = 1;

  for (const auto &[var, range] : input_iters) {
    PrimExpr simplified_min = tl::arith::Simplify(analyzer, range->begin);
    PrimExpr simplified_extent = tl::arith::Simplify(analyzer, range->extent);
    auto min = GetConstInt(simplified_min);
    auto extent = GetConstInt(simplified_extent);
    if (!min.has_value() || !extent.has_value()) {
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
          tl::arith::Simplify(analyzer, Substitute(index, substitution));
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
  sym::Analyzer analyzer;
  *domain_size = 1;
  extents->clear();
  extents->reserve(shape.size());
  for (const PrimExpr &dim : shape) {
    auto extent = GetConstInt(tl::arith::Simplify(analyzer, dim));
    if (!extent.has_value() || *extent < 0) {
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

  sym::Analyzer analyzer;
  std::vector<int64_t> output;
  output.reserve(indices.size());
  for (const PrimExpr &index : indices) {
    PrimExpr value =
        tl::arith::Simplify(analyzer, Substitute(index, substitution));
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
  sym::Analyzer analyzer;

  Array<PrimVar> logical_vars;
  Map<PrimVar, PrimExpr> logical_substitution;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    PrimVar var("__layout_logical_" + std::to_string(i), input_shape[i].ty());
    tl::arith::Bind(analyzer, var, Range(0, input_shape[i]));
    logical_vars.push_back(var);
    logical_substitution.Set(InputPlaceholder(i), var);
  }
  Array<PrimExpr> physical_from_logical =
      Substitute(forward_indices, logical_substitution);
  Map<PrimVar, PrimExpr> inverse_substitution;
  for (size_t i = 0; i < physical_from_logical.size(); ++i) {
    inverse_substitution.Set(InputPlaceholder(i), physical_from_logical[i]);
  }
  Array<PrimExpr> recovered_logical =
      Substitute(backward_indices, inverse_substitution);
  for (size_t i = 0; i < logical_vars.size(); ++i) {
    if (!tl::arith::CanProveEqual(analyzer, recovered_logical[i],
                                  logical_vars[i])) {
      return false;
    }
  }

  Array<PrimVar> physical_vars;
  Map<PrimVar, PrimExpr> physical_substitution;
  for (size_t i = 0; i < output_shape.size(); ++i) {
    PrimVar var("__layout_physical_" + std::to_string(i), output_shape[i].ty());
    tl::arith::Bind(analyzer, var, Range(0, output_shape[i]));
    physical_vars.push_back(var);
    physical_substitution.Set(InputPlaceholder(i), var);
  }
  Array<PrimExpr> logical_from_physical =
      Substitute(backward_indices, physical_substitution);
  Map<PrimVar, PrimExpr> forward_substitution;
  for (size_t i = 0; i < logical_from_physical.size(); ++i) {
    forward_substitution.Set(InputPlaceholder(i), logical_from_physical[i]);
  }
  Array<PrimExpr> recovered_physical =
      Substitute(forward_indices, forward_substitution);
  for (size_t i = 0; i < physical_vars.size(); ++i) {
    if (!tl::arith::CanProveEqual(analyzer, recovered_physical[i],
                                  physical_vars[i])) {
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

bool CanProveInjective(const Array<PrimExpr> &forward_indices,
                       const Map<PrimVar, Range> &input_iters) {
  sym::Analyzer analyzer;
  Map<PrimVar, PrimExpr> other_substitution;
  PrimExpr same_input = IntImm::Bool(true);
  for (const auto &[var, range] : input_iters) {
    PrimVar other(var->name + "<OTHER>", var.ty());
    tl::arith::Bind(analyzer, var, range);
    tl::arith::Bind(analyzer, other, range);
    other_substitution.Set(var, other);
    same_input = And(same_input, EQ(var, other));
  }

  PrimExpr same_output = IntImm::Bool(true);
  for (const PrimExpr &index : forward_indices) {
    same_output =
        And(same_output, EQ(index, Substitute(index, other_substitution)));
  }
  With<tl::arith::ConstraintContext> constraint(analyzer, same_output);
  return tl::arith::CanProve(analyzer, same_input);
}

// A left inverse proves injectivity even when symbolic padding prevents a
// bijective IterMap. NoCheck only constructs the candidate: every logical
// coordinate must still be recovered on the original input domain.
bool CanProveLeftInverse(const Array<PrimExpr> &forward_indices,
                         const Map<PrimVar, Range> &input_iters) {
  sym::Analyzer analyzer;
  PrimExpr nonempty = IntImm::Bool(true);
  for (const auto &[var, range] : input_iters) {
    tl::arith::Bind(analyzer, var, range);
    nonempty = And(nonempty, range->extent > 0);
  }
  // Empty input domains are trivially injective. For a nonempty domain all
  // extents are positive, including symbolic strides in a flattened index.
  With<tl::arith::ConstraintContext> constraint(analyzer, nonempty);
  auto iter_map = sym::DetectIterMap(forward_indices, input_iters, 1,
                                     sym::IterMapLevel::NoCheck, analyzer);
  if (!iter_map->errors.empty()) {
    return false;
  }
  Map<tvm::Var, PrimExpr> inverse;
  try {
    inverse = sym::InverseAffineIterMap(iter_map->indices, forward_indices);
  } catch (const Error &) {
    // NoCheck can produce overlapping iter sums that the inverse builder
    // cannot invert. Such a candidate supplies no proof of injectivity.
    return false;
  }
  for (const auto &[var, range] : input_iters) {
    auto it = inverse.find(var);
    PrimExpr recovered = it == inverse.end() ? range->begin : (*it).second;
    recovered = tl::arith::Simplify(analyzer, recovered);
    recovered = sym::IterMapSimplify({recovered}, input_iters, 1,
                                     sym::IterMapLevel::NoCheck, analyzer)[0];
    if (!tl::arith::CanProveEqual(analyzer, recovered, var)) {
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
    sym::Analyzer analyzer;
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

namespace {

IndexMap BindLayoutCoordinates(const Array<PrimExpr> &expressions,
                               size_t input_dimensions, bool include_replica) {
  Array<PrimVar> variables;
  Map<PrimVar, PrimExpr> replacements;
  for (size_t index = 0; index < input_dimensions; ++index) {
    PrimVar variable("index_" + std::to_string(index),
                     InputPlaceholder(index).ty());
    replacements.Set(InputPlaceholder(index), variable);
    variables.push_back(variable);
  }
  if (include_replica) {
    PrimVar replica("replica", ReplicationPlaceholder().ty());
    replacements.Set(ReplicationPlaceholder(), replica);
    variables.push_back(replica);
  }
  return IndexMap(variables, Substitute(expressions, replacements));
}

Array<PrimExpr> CanonicalLayoutCoordinates(const IndexMap &mapping,
                                           size_t input_dimensions) {
  ICHECK(mapping->initial_indices.size() == input_dimensions ||
         mapping->initial_indices.size() == input_dimensions + 1);
  Map<PrimVar, PrimExpr> replacements;
  for (size_t index = 0; index < input_dimensions; ++index) {
    replacements.Set(mapping->initial_indices[index], InputPlaceholder(index));
  }
  if (mapping->initial_indices.size() == input_dimensions + 1) {
    replacements.Set(mapping->initial_indices.back(), ReplicationPlaceholder());
  }
  return Substitute(mapping->final_indices, replacements);
}

} // namespace

Array<PrimExpr> LayoutNode::GetForwardIndex() const {
  return CanonicalLayoutCoordinates(index_map_, InputDim());
}

PrimExpr FragmentNode::GetForwardThread() const {
  return CanonicalLayoutCoordinates(thread_map_, InputDim())[0];
}

LayoutNode::LayoutNode(Array<PrimExpr> input_size,
                       Array<PrimExpr> forward_index)
    : index_map_({}, {}), input_size_(std::move(input_size)) {
  sym::Analyzer analyzer;
  input_size_ = input_size_.Map([&](const PrimExpr &extent) {
    return tl::arith::Simplify(analyzer, extent);
  });
  UpdateAnalyzer(analyzer);
  auto simplified = forward_index.Map([&](const PrimExpr &expression) {
    return tl::arith::Simplify(analyzer, expression);
  });
  index_map_ = BindLayoutCoordinates(simplified, InputDim(), false);
}

Layout::Layout(Array<LayoutAxis> forward_var, Array<PrimExpr> forward_index) {
  Map<PrimVar, PrimExpr> vmap;
  Array<PrimExpr> input_size;
  for (size_t i = 0; i < forward_var.size(); i++) {
    vmap.Set(forward_var[i]->var, InputPlaceholder(i));
    ICHECK(IsZero(forward_var[i]->dom->begin));
    input_size.push_back(forward_var[i]->dom->extent);
  }
  forward_index =
      forward_index.Map([&](const PrimExpr &e) { return Substitute(e, vmap); });
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
      .def_ro("index_map", &LayoutNode::index_map_)
      .def("_DebugOutput", &LayoutNode::DebugOutput);
}

void LayoutNode::UpdateAnalyzer(sym::Analyzer analyzer) const {
  for (const auto &[var, dom] : GetVarMap()) {
    tl::arith::Bind(analyzer, var, dom);
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
  sym::Analyzer analyzer;
  UpdateAnalyzer(analyzer);
  for (size_t i = 0; i < ret.size(); i++) {
    auto ist = tl::arith::IntSetOf(analyzer, GetForwardIndex()[i] + 1);
    if (prim::is_neg_inf(ist.min()) && prim::is_pos_inf(ist.max())) {
      // Analyzer couldn't form an IntervalSet (e.g. bitwise ops).
      // Fall back to ConstIntBound to derive a safe extent.
      auto cib = tl::arith::ConstIntBoundOf(analyzer, GetForwardIndex()[i]);
      if (cib->min_value != sym::ConstIntBound::kNegInf &&
          cib->max_value != sym::ConstIntBound::kPosInf &&
          cib->min_value >= 0) {
        ret.Set(i, IntImm::Int64(cib->max_value + 1));
      } else {
        TVM_FFI_THROW(ValueError) << "Cannot bound layout output axis " << i
                                  << ": " << GetForwardIndex()[i];
      }
    } else {
      ret.Set(i, tl::arith::Simplify(analyzer, ist.max()));
    }
  }
  return ret;
}

Array<Range> LayoutNode::MapRegionBounds(const Array<Range> &region) const {
  ICHECK_EQ(region.size(), InputDim())
      << "MapRegionBounds: region rank (" << region.size()
      << ") != layout input rank (" << InputDim() << ")";

  sym::Analyzer analyzer;
  // Bind each InputPlaceholder to the sub-range [min, min + extent)
  // instead of the full extent [0, input_size) that OutputShape uses.
  for (size_t i = 0; i < InputDim(); i++) {
    PrimExpr min = region[i]->begin;
    PrimExpr ext = region[i]->extent;
    tl::arith::Bind(analyzer, InputPlaceholder(i), Range(min, min + ext));
  }

  Array<Range> result;
  result.reserve(OutputDim());
  for (size_t i = 0; i < OutputDim(); i++) {
    auto ist = tl::arith::IntSetOf(analyzer, GetForwardIndex()[i]);
    PrimExpr lo = ist.min();
    PrimExpr hi = ist.max();
    if (prim::is_neg_inf(lo) && prim::is_pos_inf(hi)) {
      // Analyzer couldn't form an IntervalSet (e.g. bitwise ops).
      // Fall back to ConstIntBound, mirroring OutputShape's strategy.
      auto cib = tl::arith::ConstIntBoundOf(analyzer, GetForwardIndex()[i]);
      if (cib->min_value != sym::ConstIntBound::kNegInf &&
          cib->max_value != sym::ConstIntBound::kPosInf &&
          cib->min_value >= 0) {
        lo = IntImm::Int32(cib->min_value);
        hi = IntImm::Int32(cib->max_value);
      } else {
        // Last-resort: use the full output dimension extent.
        Array<PrimExpr> out_shape = OutputShape();
        lo = IntImm::Int32(0);
        hi = out_shape[i];
      }
    }
    result.push_back(Range(lo, hi + 1));
  }
  return result;
}

PrimExpr LayoutNode::GetLinearizedForwardIndex() const {
  Array<PrimExpr> output_shape = OutputShape();
  ICHECK_EQ(output_shape.size(), GetForwardIndex().size());

  PrimExpr linearized_index = IntImm::Int32(0);
  for (size_t i = 0; i < GetForwardIndex().size(); ++i) {
    linearized_index =
        linearized_index * output_shape[i] + GetForwardIndex()[i];
  }
  return linearized_index;
}

Array<PrimExpr> LayoutNode::Forward(const Array<PrimExpr> &vars) const {
  if (vars.empty())
    return GetForwardIndex();
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

  Array<PrimExpr> transformed = GetForwardIndex().Map(
      [&](const PrimExpr &e) { return Substitute(e, vmap); });
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
  for (const auto &e : GetForwardIndex()) {
    new_forward_index.push_back(Substitute(e, vmap));
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
  for (const auto &e : GetForwardIndex()) {
    new_forward_index.push_back(Substitute(e, vmap));
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
    auto new_forward_index = GetForwardIndex().Map(
        [&](const PrimExpr &e) { return Substitute(e, vmap); });
    auto new_forward_thread =
        Substitute(GetForwardThread(), vmap) + thread_size * repeats_index;
    return Fragment(new_input_size, new_forward_index, new_forward_thread,
                    replicate_size_, std::nullopt);
  } else {
    ICHECK(OutputDim() == 1);
    PrimExpr frag_len = OutputShape()[0];
    Array<PrimExpr> new_forward_index = {
        Substitute(GetForwardIndex()[0], vmap) + frag_len * repeats_index};
    PrimExpr new_forward_thread = Substitute(GetForwardThread(), vmap);
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
      Substitute(GetForwardThread(), vmap) +
      ThreadExtent() * FloorDiv(ReplicationPlaceholder(), ReplicateExtent());
  return Fragment(input_size_, GetForwardIndex(), new_forward_thread,
                  ReplicateExtent() * repeats, std::nullopt);
}

Fragment FragmentNode::DeReplicate() const {
  ICHECK(OutputDim() == 1);
  sym::Analyzer analyzer;
  UpdateAnalyzer(analyzer);
  int factor = 1;
  auto rep_size = GetConstInt(ReplicateExtent());
  auto idx_size = GetConstInt(OutputShape()[0]);
  if (rep_size && idx_size) {
    factor = std::gcd(*rep_size, *idx_size);
  }
  if (factor == 1)
    return GetRef<Fragment>(this);

  Map<PrimVar, PrimExpr> vmap;
  vmap.Set(ReplicationPlaceholder(),
           ReplicationPlaceholder() * factor +
               FloorMod(GetForwardIndex()[0], factor));
  PrimExpr new_forward_thread = Substitute(GetForwardThread(), vmap);
  Array<PrimExpr> new_forward_index = {FloorDiv(GetForwardIndex()[0], factor)};
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

namespace {
Optional<Array<PrimExpr>>
InvertFlattenedLayout(const LayoutNode &layout,
                      const Array<PrimExpr> &outputs) {
  sym::Analyzer analyzer;
  std::vector<int64_t> shape;
  int64_t elements = 1;
  for (const PrimExpr &dimension : layout.InputShape()) {
    auto extent = GetConstInt(tl::arith::Simplify(analyzer, dimension));
    if (!extent || *extent <= 0 ||
        *extent > std::numeric_limits<int32_t>::max() / elements)
      return std::nullopt;
    shape.push_back(*extent);
    elements *= *extent;
  }
  PrimVar flattened("layout_flat", PrimType::Int(32));
  Range domain(0, static_cast<int32_t>(elements));
  tl::arith::Bind(analyzer, flattened, domain);
  Map<PrimVar, PrimExpr> substitutions;
  int64_t stride = elements;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    stride /= shape[axis];
    substitutions.Set(
        InputPlaceholder(axis),
        FloorMod(FloorDiv(flattened, static_cast<int32_t>(stride)),
                 static_cast<int32_t>(shape[axis])));
  }
  auto forward = layout.GetForwardIndex().Map([&](const PrimExpr &index) {
    return tl::arith::Simplify(analyzer, Substitute(index, substitutions));
  });
  auto mapping = sym::DetectIterMap(forward, {{flattened, domain}}, 1,
                                    sym::IterMapLevel::Bijective, analyzer);
  if (!mapping->errors.empty())
    return std::nullopt;
  auto inverse = sym::InverseAffineIterMap(mapping->indices, outputs);
  PrimExpr flat_inverse = inverse.Get(flattened).value_or(PrimExpr(0));
  Array<PrimExpr> result;
  for (size_t axis = 0; axis < shape.size(); ++axis)
    result.push_back(Substitute(substitutions[InputPlaceholder(axis)],
                                {{flattened, flat_inverse}}));
  return result;
}
} // namespace

std::pair<Layout, sym::IterMapLevel>
LayoutNode::InverseWithLevel(bool require_padding_guard) const {
  sym::Analyzer analyzer;
  auto collect_symbolic = [&](const Array<PrimExpr> &shape) {
    Array<PrimExpr> symbolic_dims;
    for (const auto &dim : shape) {
      if (!GetConstInt(tl::arith::Simplify(analyzer, dim))) {
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
  auto domains = GetVarMap();
  for (const auto &[variable, domain] : domains)
    tl::arith::Bind(analyzer, variable, domain);
  auto forward_indices = GetForwardIndex().Map([&](const PrimExpr &index) {
    return tl::arith::Simplify(analyzer, index);
  });
  sym::IterMapResult res =
      sym::DetectIterMap(forward_indices, domains, 1, level, analyzer);

  auto outputs_shape = OutputShape();
  Array<PrimExpr> outputs;
  for (size_t i = 0; i < OutputDim(); i++) {
    outputs.push_back(InputPlaceholder(i));
  }

  Array<PrimExpr> backward_index;
  if (!res->errors.empty()) {
    auto flattened = level == sym::IterMapLevel::Bijective
                         ? InvertFlattenedLayout(*this, outputs)
                         : std::nullopt;
    if (!flattened) {
      std::ostringstream msg;
      msg << "Layout " << DebugOutput() << " has errors: " << res->errors;
      throw NormalizeIterException(msg.str());
    }
    backward_index = *flattened;
  } else {
    auto inv = sym::InverseAffineIterMap(res->indices, outputs);
    for (size_t axis = 0; axis < InputDim(); ++axis)
      backward_index.push_back(
          inv.Get(InputPlaceholder(axis)).value_or(PrimExpr(0)));
  }

  if (level == sym::IterMapLevel::Bijective) {
    InverseRoundTripResult round_trip = CheckStaticInverseRoundTrip(
        input_size_, outputs_shape, GetForwardIndex(), backward_index);
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

Layout LayoutNode::Reshape(const Array<PrimExpr> &shape, sym::Analyzer analyzer,
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
  sym::Analyzer az = analyzer;
  ICHECK(tl::arith::CanProveEqual(az, input_shape_product * rescale_num,
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
  // Step 5. Substitute original indices into GetForwardIndex()
  Array<PrimExpr> new_forward_index = SubstituteForwardIndex(
      GetForwardIndex(), InputShape(), original_indices, az);
  new_forward_index = RestoreInputPlaceholders(new_forward_index, new_vars);
  return Layout(shape, new_forward_index);
}

Layout FragmentNode::Reshape(const Array<PrimExpr> &shape,
                             sym::Analyzer analyzer, const PrimExpr rescale_num,
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
  sym::Analyzer az = analyzer;
  ICHECK(tl::arith::CanProveEqual(az, input_prod * rescale_num,
                                  shape_prod * rescale_den))
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
      SubstituteForwardIndex(GetForwardIndex(), InputShape(), orig_indices, az);
  PrimExpr new_forward_thread = SubstituteReshapedExpr(
      GetForwardThread(), InputShape(), orig_indices, az);
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
                              const PrimExpr &forward_thread,
                              sym::Analyzer analyzer) {
  // we build iter_vars from input_iters, but set _rep to range [0, 1)
  // to make it not contribute to the index of the forward_idx
  Array<LayoutAxis> iter_vars;
  for (const auto &[var, range_] : input_iters) {
    Range range = range_;
    if (var.same_as(ReplicationPlaceholder())) {
      range = Range(0, 1);
    }
    iter_vars.push_back(LayoutAxis(range, var));
  }

  Array<sym::IterSplitExpr> splits = DivideUnusedIterators(
      {tl::arith::Simplify(
          analyzer, Substitute(forward_thread,
                               {{ReplicationPlaceholder(), PrimExpr(0)}}))},
      iter_vars, analyzer);
  return MakeFlattenedExpression(splits);
}

FragmentNode::FragmentNode(Array<PrimExpr> input_size,
                           Array<PrimExpr> forward_index,
                           PrimExpr forward_thread, PrimExpr replicate_size)
    : thread_map_({}, {}), replicate_size_(std::move(replicate_size)) {
  input_size_ = std::move(input_size);
  sym::Analyzer analyzer;
  input_size_ = input_size_.Map([&](const PrimExpr &extent) {
    return tl::arith::Simplify(analyzer, extent);
  });
  replicate_size_ = tl::arith::Simplify(analyzer, replicate_size_);
  UpdateAnalyzer(analyzer);
  forward_thread = tl::arith::Simplify(analyzer, forward_thread);
  if (forward_index.empty()) {
    forward_index = {
        infer_fragment_index(GetVarMap(), forward_thread, analyzer)};
  }
  auto simplified = forward_index.Map([&](const PrimExpr &expression) {
    return tl::arith::Simplify(analyzer, expression);
  });
  index_map_ = BindLayoutCoordinates(simplified, InputDim(), true);
  thread_map_ = BindLayoutCoordinates({forward_thread}, InputDim(), true);
}

Fragment::Fragment(Array<LayoutAxis> forward_var, Array<PrimExpr> forward_index,
                   PrimExpr forward_thread,
                   Optional<LayoutAxis> thread_replicate) {
  Map<PrimVar, PrimExpr> vmap;
  Array<PrimExpr> input_size;
  PrimExpr replicate_size = 1;
  for (size_t i = 0; i < forward_var.size(); i++) {
    vmap.Set(forward_var[i]->var, InputPlaceholder(i));
    ICHECK(IsZero(forward_var[i]->dom->begin));
    input_size.push_back(forward_var[i]->dom->extent);
  }
  if (thread_replicate.has_value()) {
    ICHECK(IsZero(thread_replicate.value()->dom->begin));
    replicate_size = thread_replicate.value()->dom->extent;
    vmap.Set(thread_replicate.value()->var, ReplicationPlaceholder());
  }
  forward_index =
      forward_index.Map([&](const PrimExpr &e) { return Substitute(e, vmap); });
  forward_thread = Substitute(forward_thread, vmap);

  auto n = make_object<FragmentNode>(input_size, forward_index, forward_thread,
                                     replicate_size);
  data_ = std::move(n);
}

Fragment::Fragment(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
                   PrimExpr forward_thread, PrimExpr replicate_size,
                   Optional<PrimVar> replicate_var) {
  if (replicate_var.has_value()) {
    forward_thread = Substitute(
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
                   std::move(forward_thread), std::move(replicate_size)),
      combine_size_(combine_size) {
  sym::Analyzer analyzer;
  combine_size_ = tl::arith::Simplify(analyzer, combine_size);
  // The low-bits convention only makes sense when the combine width evenly
  // tiles the replication coordinate.
  auto rep = GetConstInt(replicate_size_);
  auto comb = GetConstInt(combine_size_);
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
    forward_thread = Substitute(
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
  sym::Analyzer analyzer;
  ss << "PartialFragment(" << InputShape() << " -> " << OutputShape()
     << ", replicate: " << ReplicateExtent() << " (combine " << combine_size_
     << " x copy "
     << tl::arith::Simplify(analyzer,
                            FloorDiv(ReplicateExtent(), combine_size_))
     << ")"
     << ", thread: " << ThreadExtent()
     << ", forward_thread: " << GetForwardThread()
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
  sym::Analyzer analyzer;
  PrimExpr rep_extent = ReplicateExtent();
  PrimExpr copy_extent =
      tl::arith::Simplify(analyzer, FloorDiv(rep_extent, combine_size_));
  PrimVar lo("_rep_lo", ReplicationPlaceholder().ty());
  PrimVar hi("_rep_hi", ReplicationPlaceholder().ty());
  PrimExpr thread =
      Substitute(GetForwardThread(),
                 {{ReplicationPlaceholder(), lo + combine_size_ * hi}});
  Map<PrimVar, Range> vmap;
  for (size_t i = 0; i < InputDim(); ++i) {
    vmap.Set(InputPlaceholder(i),
             Range::FromMinExtent(IntImm::Int32(0), InputShape()[i]));
  }
  vmap.Set(lo, Range::FromMinExtent(IntImm::Int32(0), combine_size_));
  vmap.Set(hi, Range::FromMinExtent(IntImm::Int32(0), copy_extent));
  sym::IterSumExpr sum = sym::NormalizeToIterSum(thread, vmap, analyzer);
  std::vector<std::pair<int, int>> steps;
  for (const auto &split : sum->args) {
    auto mark = split->source->source.as<PrimVar>();
    if (!mark || !mark.value().same_as(lo)) {
      continue;
    }
    auto scale = GetConstInt(split->scale);
    auto extent = GetConstInt(split->extent);
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
  sym::Analyzer analyzer;
  return ExprDeepEqual()(tl::arith::Simplify(analyzer, GetForwardThread()),
                         ReplicationPlaceholder());
}

sym::IterMapResult
LayoutNode::DetectInjective(bool require_padding_guard) const {
  return DetectInjectiveMapping(GetForwardIndex(), GetVarMap(),
                                require_padding_guard);
}

sym::IterMapResult
FragmentNode::DetectInjective(bool require_padding_guard) const {
  // A fragment's physical identity includes its owning thread as well as its
  // per-thread indices. The replication coordinate is part of GetVarMap().
  Array<PrimExpr> indices{GetForwardThread()};
  auto forward_indices = GetForwardIndex();
  indices.insert(indices.end(), forward_indices.begin(), forward_indices.end());
  return DetectInjectiveMapping(indices, GetVarMap(), require_padding_guard);
}

PrimExpr FragmentNode::ThreadExtent() const {
  Array<PrimExpr> ret(OutputDim(), 1);
  sym::Analyzer analyzer;
  UpdateAnalyzer(analyzer);
  auto ist = tl::arith::IntSetOf(analyzer, GetForwardThread() + 1);
  return ist.max();
}

Array<PrimExpr> FragmentNode::GetForwardVars() const {
  Array<PrimExpr> vars;
  if (*GetConstInt(ReplicateExtent()) > 1) {
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

  return Substitute(GetForwardThread(), vmap);
}

Layout FragmentNode::Inverse() const {
  auto result = InverseWithLevel();
  return std::move(result.first);
}

std::pair<Layout, sym::IterMapLevel>
FragmentNode::InverseWithLevel(bool require_padding_guard) const {
  auto input_size_copy = input_size_;
  input_size_copy.push_back(ReplicateExtent());
  auto forward_index_copy = GetForwardIndex();
  forward_index_copy.push_back(
      Substitute(GetForwardThread(),
                 {{ReplicationPlaceholder(), InputPlaceholder(InputDim())}}));
  auto fwd = Layout(input_size_copy, forward_index_copy);
  return fwd->InverseWithLevel(require_padding_guard);
}

Fragment FragmentNode::CondenseReplicateVar() const {
  sym::Analyzer analyzer;
  auto input_iters = GetVarMap();
  input_iters.Set(ReplicationPlaceholder(), {0, ReplicateExtent()});
  auto [new_forward_thread, new_thread_replicate] =
      CompressIterator(GetForwardThread(), ToLayoutAxes(input_iters),
                       ReplicationPlaceholder(), analyzer);
  return Fragment(input_size_, GetForwardIndex(), new_forward_thread,
                  new_thread_replicate->dom->extent, new_thread_replicate->var);
}

std::pair<Fragment, PrimExpr>
CondenseReplicateVarKeepingBoundary(const Fragment &fragment,
                                    const PrimExpr &combine_size) {
  sym::Analyzer analyzer;
  const FragmentNode *node = fragment.get();
  ICHECK(node != nullptr);
  PrimExpr rep_extent = node->ReplicateExtent();
  PrimExpr copy_extent =
      tl::arith::Simplify(analyzer, FloorDiv(rep_extent, combine_size));
  PrimVar lo("_rep_lo", ReplicationPlaceholder().ty());
  PrimVar hi("_rep_hi", ReplicationPlaceholder().ty());
  PrimExpr thread =
      Substitute(node->GetForwardThread(),
                 {{ReplicationPlaceholder(), lo + combine_size * hi}});
  Map<PrimVar, Range> vmap;
  for (size_t i = 0; i < node->InputDim(); ++i) {
    vmap.Set(InputPlaceholder(i),
             Range::FromMinExtent(IntImm::Int32(0), node->InputShape()[i]));
  }
  vmap.Set(lo, Range::FromMinExtent(IntImm::Int32(0), combine_size));
  vmap.Set(hi, Range::FromMinExtent(IntImm::Int32(0), copy_extent));
  // The projection may have collapsed the two halves into a single
  // expression over `_rep` (e.g. `(lo + W*hi) // c` from a per-thread
  // serial run under loop replication). Re-simplify with the ranges bound
  // so the halves separate again; otherwise NormalizeToIterSum fuses
  // `lo + W*hi` into one iterator and CompressIterator, which attributes
  // splits per source var, cannot compress either half.
  for (const auto &[var, range] : vmap) {
    tl::arith::Bind(analyzer, var, range);
  }
  thread = tl::arith::Simplify(analyzer, thread);
  // Compress the combine half first, then the copy half; each keeps only
  // the sub-iterators that actually reach the thread expression.
  auto [expr_lo, lo_iv] =
      CompressIterator(thread, ToLayoutAxes(vmap), lo, analyzer);
  vmap.erase(lo);
  vmap.Set(lo_iv->var, lo_iv->dom);
  auto [expr_hi, hi_iv] =
      CompressIterator(expr_lo, ToLayoutAxes(vmap), hi, analyzer);
  PrimExpr w = tl::arith::Simplify(analyzer, lo_iv->dom->extent);
  PrimExpr c = tl::arith::Simplify(analyzer, hi_iv->dom->extent);
  PrimExpr new_thread = Substitute(
      expr_hi, {{lo_iv->var, FloorMod(ReplicationPlaceholder(), w)},
                {hi_iv->var, FloorDiv(ReplicationPlaceholder(), w)}});
  Fragment out(node->InputShape(), node->GetForwardIndex(), new_thread,
               tl::arith::Simplify(analyzer, w * c), std::nullopt);
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
     << ", forward_thread: " << GetForwardThread()
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
      .def_ro("thread_map", &FragmentNode::thread_map_)
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
  refl::GlobalDef()
      .def("tl.LayoutAxis",
           [](Range domain, PrimVar variable) {
             return LayoutAxis(std::move(domain), std::move(variable));
           })
      .def_packed("tl.Layout",
                  [](PackedArgs args, Any *rv) {
                    *rv = Layout(args[0].cast<Array<LayoutAxis>>(),
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
             return layout->Reshape(shape, sym::Analyzer(), rescale_num,
                                    rescale_den);
           })
      .def("tl.Layout_index",
           [](Layout layout) { return layout->GetForwardIndex(); })
      .def("tl.Layout_linearized_index",
           [](Layout layout) { return layout->GetLinearizedForwardIndex(); })
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
      .def_packed(
          "tl.Fragment",
          [](PackedArgs args, Any *rv) {
            *rv = Fragment(
                /*forward_var=*/args[0].cast<Array<LayoutAxis>>(),
                /*forward_index=*/args[1].cast<Array<PrimExpr>>(),
                /*forward_thread=*/args[2].cast<PrimExpr>(),
                /*thread_replicate=*/args[3].cast<Optional<LayoutAxis>>());
          })
      .def("tl.Fragment_is_equal",
           [](Fragment fragment, Fragment other) {
             if (fragment.as<PartialFragmentNode>() ||
                 other.as<PartialFragmentNode>()) {
               return fragment->IsEqual(
                   static_cast<const LayoutNode *>(other.get()));
             }
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
           [](const tirx::TensorVar &buffer, bool k_inner, bool allow_pad) {
             return MakeSwizzledLayout(buffer, k_inner, allow_pad);
           })
      .def("tl.make_volta_swizzled_layout",
           [](const tirx::TensorVar &buffer, bool is_a, bool k_inner) {
             return MakeVoltaSwizzledLayout(buffer, is_a, k_inner);
           })
      .def("tl.make_wgmma_swizzled_layout",
           [](const tirx::TensorVar &buffer, int continuity, bool k_inner) {
             return MakeWgmmaSwizzledLayout(buffer, continuity, k_inner);
           })
      .def("tl.make_tcgen05mma_swizzled_layout",
           [](const tirx::TensorVar &buffer, int continuity, bool k_inner) {
             return MakeTcgen05MmaSwizzledLayout(buffer, continuity, k_inner);
           })
      .def("tl.make_full_bank_swizzled_layout",
           [](const tirx::TensorVar &buffer) {
             return MakeFullBankSwizzleLayout(buffer);
           })
      .def("tl.make_half_bank_swizzled_layout",
           [](const tirx::TensorVar &buffer) {
             return MakeHalfBankSwizzleLayout(buffer);
           })
      .def("tl.make_quarter_bank_swizzled_layout",
           [](const tirx::TensorVar &buffer) {
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
      .def_packed(
          "tl.PartialFragment",
          [](PackedArgs args, Any *rv) {
            // Same argument shape as tl.Fragment plus a trailing
            // combine width; the LayoutAxis-form Fragment ctor does the
            // placeholder substitution, then the result is
            // reinterpreted as per-thread partials. `combine` is the
            // number of addend lanes in the low bits of the
            // replication coordinate; when absent, every declared
            // replica is an addend lane.
            Fragment fragment(
                /*forward_var=*/args[0].cast<Array<LayoutAxis>>(),
                /*forward_index=*/args[1].cast<Array<PrimExpr>>(),
                /*forward_thread=*/args[2].cast<PrimExpr>(),
                /*thread_replicate=*/args[3].cast<Optional<LayoutAxis>>());
            if (args.size() <= 4) {
              *rv = PartialFragment::FromFragment(fragment);
              return;
            }
            int64_t combine = args[4].cast<int64_t>();
            auto rep = GetConstInt(fragment->ReplicateExtent());
            if (combine < 1 || (rep.has_value() && *rep % combine != 0)) {
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
  LayoutAxisNode::RegisterReflection();
  FragmentNode::RegisterReflection();
  PartialFragmentNode::RegisterReflection();
}

namespace {

ffi::Optional<script::printer::ExprDoc>
LayoutDocTranslate(script::printer::DocTranslatorObj *translator,
                   ffi::AnyView value, const ffi::Object *) {
  return TranslateLayoutFields(translator, value, "LayoutFromIndexMap",
                               {"input_size", "index_map"});
}

ffi::Optional<script::printer::ExprDoc>
FragmentDocTranslate(script::printer::DocTranslatorObj *translator,
                     ffi::AnyView value, const ffi::Object *) {
  return TranslateLayoutFields(translator, value, "FragmentFromIndexMaps",
                               {"input_size", "index_map", "thread_map",
                                "replicate_size", "thread_range"});
}

ffi::Optional<script::printer::ExprDoc>
PartialFragmentDocTranslate(script::printer::DocTranslatorObj *translator,
                            ffi::AnyView value, const ffi::Object *) {
  return TranslateLayoutFields(
      translator, value, "PartialFragmentFromIndexMaps",
      {"input_size", "index_map", "thread_map", "replicate_size",
       "combine_size", "thread_range"});
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  using script::printer::FDocTranslate;
  using script::printer::type_attr::kDocTranslate;
  reflection::TypeAttrDef<LayoutNode>().attr(
      kDocTranslate.name, FDocTranslate::FromNative<&LayoutDocTranslate>());
  reflection::TypeAttrDef<FragmentNode>().attr(
      kDocTranslate.name, FDocTranslate::FromNative<&FragmentDocTranslate>());
  reflection::TypeAttrDef<PartialFragmentNode>().attr(
      kDocTranslate.name,
      FDocTranslate::FromNative<&PartialFragmentDocTranslate>());
  reflection::GlobalDef()
      .def("tl.LayoutFromIndexMap",
           [](Array<PrimExpr> shape, IndexMap index) {
             return Layout(shape,
                           MapLayoutCoordinates(index, shape.size(), false));
           })
      .def("tl.FragmentFromIndexMaps",
           [](Array<PrimExpr> shape, IndexMap index, IndexMap thread,
              PrimExpr replicas, Optional<Range> thread_range) {
             auto thread_indices =
                 MapLayoutCoordinates(thread, shape.size(), true);
             TVM_FFI_CHECK_EQ(thread_indices.size(), 1U, ValueError)
                 << "Fragment thread map must have one output";
             Fragment result(shape,
                             MapLayoutCoordinates(index, shape.size(), true),
                             thread_indices[0], replicas, std::nullopt);
             return result->BindThreadRange(thread_range);
           })
      .def("tl.PartialFragmentFromIndexMaps",
           [](Array<PrimExpr> shape, IndexMap index, IndexMap thread,
              PrimExpr replicas, PrimExpr combine,
              Optional<Range> thread_range) {
             auto thread_indices =
                 MapLayoutCoordinates(thread, shape.size(), true);
             TVM_FFI_CHECK_EQ(thread_indices.size(), 1U, ValueError)
                 << "Fragment thread map must have one output";
             return PartialFragment(
                 shape, MapLayoutCoordinates(index, shape.size(), true),
                 thread_indices[0], replicas, combine, std::nullopt,
                 thread_range);
           });
}

} // namespace tl
} // namespace tvm
