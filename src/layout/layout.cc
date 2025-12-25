/*!
 * \file layout/layout.cc
 *
 */

#include "layout.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <tvm/arith/pattern.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "arith/pattern_match.h"
#include "tvm/node/functor.h"
#include "tvm/node/repr_printer.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

static Var getPlaceholder(const std::string &s) {
  static std::unordered_map<std::string, Var> map;
  if (map.find(s) == map.end()) {
    map[s] = Var(s);
  }
  return map[s];
}

Var ReplicationPlaceholder() { return getPlaceholder("_rep"); }
Var InputPlaceholder(size_t idx) {
  return getPlaceholder(std::string{'_', char('i' + idx)});
}

Map<Var, Range> LayoutNode::getVarMap() const {
  Map<Var, Range> map;
  for (size_t i = 0; i < InputDim(); i++) {
    map.Set(InputPlaceholder(i), {0, input_size_[i]});
  }
  return map;
}

Map<Var, Range> FragmentNode::getVarMap() const {
  auto map = LayoutNode::getVarMap();
  map.Set(ReplicationPlaceholder(), {0, ReplicateExtent()});
  return map;
}

LayoutNode::LayoutNode(Array<PrimExpr> input_size,
                       Array<PrimExpr> forward_index) {
  input_size_ = input_size;
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  forward_index_ = forward_index.Map(
      [&](const PrimExpr &e) { return analyzer.Simplify(e); });
}

Layout::Layout(Array<IterVar> forward_var, Array<PrimExpr> forward_index) {
  Map<Var, PrimExpr> vmap;
  Array<PrimExpr> input_size;
  for (size_t i = 0; i < forward_var.size(); i++) {
    vmap.Set(forward_var[i]->var, InputPlaceholder(i));
    CHECK(is_zero(forward_var[i]->dom->min));
    input_size.push_back(forward_var[i]->dom->extent);
  }
  forward_index =
      forward_index.Map([&](const PrimExpr &e) { return Substitute(e, vmap); });
  auto n = tvm::ffi::make_object<LayoutNode>(input_size, forward_index);
  data_ = std::move(n);
}

Layout::Layout(Array<PrimExpr> input_size, Array<PrimExpr> forward_index) {
  auto n = tvm::ffi::make_object<LayoutNode>(input_size, forward_index);
  data_ = std::move(n);
}

void LayoutNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<LayoutNode>()
      .def_ro("input_size", &LayoutNode::input_size_)
      .def_ro("forward_index", &LayoutNode::forward_index_)
      .def("_DebugOutput", &LayoutNode::DebugOutput);
}

void LayoutNode::UpdateAnalyzer(arith::Analyzer *analyzer) const {
  for (const auto &[var, dom] : getVarMap()) {
    analyzer->Bind(var, dom);
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
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  for (size_t i = 0; i < ret.size(); i++) {
    auto ist = analyzer.int_set(forward_index_[i] + 1);
    if (arith::is_neg_inf(ist.min()) && arith::is_pos_inf(ist.max())) {
      // Analyzer couldn't form an IntervalSet (e.g. bitwise ops).
      // Fall back to ConstIntBound to derive a safe extent.
      auto cib = analyzer.const_int_bound(forward_index_[i]);
      if (cib->min_value != arith::ConstIntBound::kNegInf &&
          cib->max_value != arith::ConstIntBound::kPosInf &&
          cib->min_value >= 0) {
        // extent = max - min + 1, using 64-bit integer literal
        ret.Set(i, Integer(cib->max_value - cib->min_value + 1));
      } else {
        // Last-resort conservative fallback to avoid OOB/crash
        // Prefer to keep dimension from known input_size_ if available.
        if (i < input_size_.size()) {
          ret.Set(i, input_size_[i]);
        } else {
          ret.Set(i, Integer(1));
        }
      }
    } else {
      ret.Set(i, ist.max());
    }
  }
  return ret;
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

  Map<Var, PrimExpr> vmap;
  for (size_t i = 0; i < InputDim(); i++) {
    vmap.Set(InputPlaceholder(i), transform_vars[i]);
  }

  Array<PrimExpr> transformed = forward_index_.Map(
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

Fragment FragmentNode::Repeat(const Array<PrimExpr> &repeats,
                              bool repeat_on_thread,
                              bool lower_dim_first) const {
  ICHECK_EQ(repeats.size(), InputDim());
  Array<PrimExpr> new_input_size;
  Map<Var, PrimExpr> vmap;
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
        [&](const PrimExpr &e) { return Substitute(e, vmap); });
    auto new_forward_thread =
        Substitute(forward_thread_, vmap) + thread_size * repeats_index;
    return Fragment(new_input_size, new_forward_index, new_forward_thread,
                    replicate_size_, std::nullopt);
  } else {
    ICHECK(OutputDim() == 1);
    PrimExpr frag_len = OutputShape()[0];
    Array<PrimExpr> new_forward_index = {Substitute(forward_index_[0], vmap) +
                                         frag_len * repeats_index};
    PrimExpr new_forward_thread = Substitute(forward_thread_, vmap);
    return Fragment(new_input_size, new_forward_index, new_forward_thread,
                    replicate_size_, std::nullopt);
  }
}

Fragment FragmentNode::Replicate(int repeats) const {
  ICHECK(repeats >= 1);
  Map<Var, PrimExpr> vmap;
  vmap.Set(ReplicationPlaceholder(),
           FloorMod(ReplicationPlaceholder(), ReplicateExtent()));
  PrimExpr new_forward_thread =
      Substitute(forward_thread_, vmap) +
      ThreadExtent() * FloorDiv(ReplicationPlaceholder(), ReplicateExtent());
  return Fragment(input_size_, forward_index_, new_forward_thread,
                  ReplicateExtent() * repeats, std::nullopt);
}

Fragment FragmentNode::DeReplicate() const {
  ICHECK(OutputDim() == 1);
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  int factor = 1;
  auto rep_size = as_const_int(ReplicateExtent());
  auto idx_size = as_const_int(OutputShape()[0]);
  if (rep_size && idx_size) {
    factor = arith::ZeroAwareGCD(*rep_size, *idx_size);
  }
  if (factor == 1)
    return tvm::ffi::GetRef<Fragment>(this);

  Map<Var, PrimExpr> vmap;
  vmap.Set(ReplicationPlaceholder(), ReplicationPlaceholder() * factor +
                                         FloorMod(forward_index_[0], factor));
  PrimExpr new_forward_thread = Substitute(forward_thread_, vmap);
  Array<PrimExpr> new_forward_index = {FloorDiv(forward_index_[0], factor)};
  return Fragment(input_size_, new_forward_index, new_forward_thread,
                  int(*rep_size) / factor, std::nullopt);
}

Fragment FragmentNode::BindThreadRange(Range thread_range) const {
  auto n = tvm::ffi::make_object<FragmentNode>(*this);
  n->thread_range_ = thread_range;
  return Fragment(n);
}

std::pair<Layout, arith::IterMapLevel> LayoutNode::InverseWithLevel() const {
  arith::Analyzer analyzer;
  auto collect_symbolic = [&](const Array<PrimExpr> &shape) {
    Array<PrimExpr> symbolic_dims;
    for (const auto &dim : shape) {
      if (!as_const_int(dim)) {
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
  auto level = is_static_shape ? arith::IterMapLevel::Bijective
                               : arith::IterMapLevel::NoCheck;
  if (!is_static_shape) {
    // Runtime guards keep dynamic tails safe, so we allow NoCheck here and
    // warn.
    DLOG(WARNING) << "Layout::Inverse on symbolic layout, falling back to "
                     "NoCheck; symbolic dims: "
                  << symbolic_dims;
  }
  arith::IterMapResult res =
      arith::DetectIterMap(forward_index_, getVarMap(), 1, level, &analyzer);
  if (!res->errors.empty()) {
    std::ostringstream msg;
    msg << "Layout " << DebugOutput() << " has errors: " << res->errors;
    throw NormalizeIterException(msg.str());
  }

  auto outputs_shape = OutputShape();
  Array<PrimExpr> outputs;
  for (size_t i = 0; i < OutputDim(); i++) {
    outputs.push_back(InputPlaceholder(i));
  }

  auto inv = arith::InverseAffineIterMap(res->indices, outputs);

  Array<PrimExpr> backward_index;
  for (size_t i = 0; i < InputDim(); i++) {
    if (inv.find(InputPlaceholder(i)) != inv.end()) {
      backward_index.push_back(inv[InputPlaceholder(i)]);
    } else {
      backward_index.push_back(0);
    }
  }

  return {Layout(outputs_shape, backward_index), level};
}

Layout LayoutNode::Reshape(const Array<PrimExpr> &shape,
                           arith::Analyzer *analyzer,
                           const PrimExpr rescale_num,
                           const PrimExpr rescale_den) const {

  // Fast path: if shape is the same, return the original layout
  if (StructuralEqual()(InputShape(), shape)) {
    return ffi::GetRef<Layout>(this);
  }

  // Step 1. Prove the product relation holds under rescale:
  //   prod(InputShape) * rescale_num == prod(shape) * rescale_den
  PrimExpr input_shape_product = Integer(1);
  for (const auto &dim : InputShape()) {
    input_shape_product *= dim;
  }
  PrimExpr shape_product = Integer(1);
  for (const auto &dim : shape) {
    shape_product *= dim;
  }

  // Use provided analyzer if present, otherwise a local fallback to avoid
  // potential null dereference paths flagged by static analysis.
  arith::Analyzer fallback_analyzer;
  arith::Analyzer *az = analyzer ? analyzer : &fallback_analyzer;
  ICHECK(az->CanProveEqual(input_shape_product * rescale_num,
                           shape_product * rescale_den))
      << "InputShape() = " << InputShape() << " shape = " << shape
      << ", rescale_num = " << rescale_num << ", rescale_den = " << rescale_den;

  // Step 2. Create new forward indices by reshaping
  // For each dimension in the new shape, we create a placeholder variable
  Array<Var> new_vars;
  new_vars.reserve(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    auto var = Var(std::string("n_") + std::to_string(i), shape[i].dtype());
    az->Bind(var, Range(0, shape[i]));
    new_vars.push_back(var);
  }
  // Step 3. Compute the flat index from new shape indices
  // flat_index = k0 * (s1 * s2 * ...) + k1 * (s2 * s3 * ...) + ... + kn
  PrimExpr flat_index = Integer(0);
  for (size_t i = 0; i < shape.size(); ++i) {
    PrimExpr stride = Integer(1);
    for (size_t j = i + 1; j < shape.size(); ++j) {
      stride = stride * shape[j];
    }
    flat_index = flat_index + new_vars[i] * stride;
  }
  // Convert new flat index (in units of new elements) to the old flat index
  // (in units of old elements) using the rational rescale factor.
  // old_flat = floor((flat_index * rescale_den) / rescale_num)
  PrimExpr old_flat_index = floordiv(flat_index * rescale_den, rescale_num);
  // Step 4. Convert flat index back to original shape indices
  // For original shape [s0, s1, ..., sm]:
  // i0 = flat_index // (s1 * s2 * ... * sm)
  // i1 = (flat_index % (s1 * s2 * ... * sm)) // (s2 * s3 * ... * sm)
  // ...
  Array<PrimExpr> original_indices;
  PrimExpr remaining = old_flat_index;
  for (size_t i = 0; i < InputShape().size(); ++i) {
    PrimExpr stride = Integer(1);
    for (size_t j = i + 1; j < InputShape().size(); ++j) {
      stride = stride * InputShape()[j];
    }
    original_indices.push_back(floordiv(remaining, stride));
    remaining = floormod(remaining, stride);
  }
  // Step 5. Substitute original indices into forward_index_
  Array<PrimExpr> new_forward_index;
  for (const auto &fwd_expr : forward_index_) {
    PrimExpr substituted = fwd_expr;
    // Replace each InputPlaceholder(i) with original_indices[i]
    for (size_t i = 0; i < InputShape().size(); ++i) {
      substituted =
          Substitute(substituted, {{InputPlaceholder(i), original_indices[i]}});
    }
    new_forward_index.push_back(az->Simplify(substituted));
  }
  for (size_t i = 0; i < new_vars.size(); ++i) {
    new_forward_index =
        Substitute(new_forward_index, {{new_vars[i], InputPlaceholder(i)}});
  }
  return Layout(shape, new_forward_index);
}

Layout FragmentNode::Reshape(const Array<PrimExpr> &shape,
                             arith::Analyzer *analyzer,
                             const PrimExpr rescale_num,
                             const PrimExpr rescale_den) const {

  // Fast path: identical input shape, return self
  if (StructuralEqual()(InputShape(), shape)) {
    return ffi::GetRef<Fragment>(this);
  }

  // 1) Prove total number of elements remains the same
  PrimExpr input_prod = Integer(1);
  for (const auto &d : InputShape())
    input_prod *= d;
  PrimExpr shape_prod = Integer(1);
  for (const auto &d : shape)
    shape_prod *= d;

  // Use provided analyzer if present, otherwise a local fallback.
  arith::Analyzer fallback_analyzer;
  arith::Analyzer *az = analyzer ? analyzer : &fallback_analyzer;
  ICHECK(az->CanProveEqual(input_prod * rescale_num, shape_prod * rescale_den))
      << "InputShape() = " << InputShape() << " shape = " << shape
      << ", rescale_num = " << rescale_num << ", rescale_den = " << rescale_den
      << " input fragment layout is = " << DebugOutput();

  // 2) Build flat index from new-shape indices
  Array<Var> new_vars;
  new_vars.reserve(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    // Cannot use InputPlaceholder(i) here, because it would cause name capture
    // (variable capture) with InputPlaceholder(i) in upper scopes. Therefore,
    // we must create a fresh variable here to avoid confusion when
    // substituting.
    auto var = Var(std::string("n_") + std::to_string(i), shape[i].dtype());
    az->Bind(var, Range(0, shape[i]));
    new_vars.push_back(var);
  }

  PrimExpr flat = Integer(0);
  for (size_t i = 0; i < shape.size(); ++i) {
    PrimExpr stride = Integer(1);
    for (size_t j = i + 1; j < shape.size(); ++j)
      stride = stride * shape[j];
    flat = flat + new_vars[i] * stride;
  }
  // Convert to old flat index units using the rational rescale factor.
  // old_flat = floor((flat * rescale_den) / rescale_num)
  PrimExpr old_flat = floordiv(flat * rescale_den, rescale_num);
  // 3) Recover original indices from flat index
  Array<PrimExpr> orig_indices;
  PrimExpr remain = old_flat;
  for (size_t i = 0; i < InputShape().size(); ++i) {
    PrimExpr stride = Integer(1);
    for (size_t j = i + 1; j < InputShape().size(); ++j)
      stride = stride * InputShape()[j];
    orig_indices.push_back(floordiv(remain, stride));
    remain = floormod(remain, stride);
  }
  // 4) Substitute old placeholders with expressions of new indices
  Array<PrimExpr> new_forward_index;
  for (const auto &e : forward_index_) {
    PrimExpr cur = e;
    for (size_t i = 0; i < InputShape().size(); ++i) {
      cur = Substitute(cur, {{InputPlaceholder(i), orig_indices[i]}});
    }
    cur = az->Simplify(cur);
    new_forward_index.push_back(cur);
  }
  PrimExpr new_forward_thread = forward_thread_;
  for (size_t i = 0; i < InputShape().size(); ++i) {
    new_forward_thread = Substitute(new_forward_thread,
                                    {{InputPlaceholder(i), orig_indices[i]}});
  }
  new_forward_thread = az->Simplify(new_forward_thread);
  for (size_t i = 0; i < new_vars.size(); ++i) {
    auto var = new_vars[i];
    new_forward_index =
        Substitute(new_forward_index, {{var, InputPlaceholder(i)}});
    new_forward_thread =
        Substitute(new_forward_thread, {{var, InputPlaceholder(i)}});
  }
  Fragment reshaped(shape, new_forward_index, new_forward_thread,
                    ReplicateExtent(), std::nullopt);
  if (thread_range_.defined()) {
    reshaped = reshaped->BindThreadRange(thread_range_);
  }
  return reshaped;
}

Layout LayoutNode::Inverse() const {
  auto inverse_result = InverseWithLevel();
  return std::move(inverse_result.first);
}

PrimExpr infer_fragment_index(const Map<Var, Range> &input_iters,
                              const PrimExpr &forward_thread,
                              arith::Analyzer *analyzer) {
  Array<arith::IterSplitExpr> splits = DivideUnusedIterators(
      {forward_thread}, ToIterVars(input_iters), analyzer);

  Array<arith::IterSplitExpr> split_without_rep;
  for (const auto &split : splits) {
    CHECK(split->source->source.as<Var>());
    if (split->source->source.as<Var>().value().same_as(
            ReplicationPlaceholder()))
      continue;
    split_without_rep.push_back(split);
  }
  return MakeFlattenedExpression(split_without_rep);
}

FragmentNode::FragmentNode(Array<PrimExpr> input_size,
                           Array<PrimExpr> forward_index,
                           PrimExpr forward_thread, PrimExpr replicate_size) {
  input_size_ = input_size;
  replicate_size_ = replicate_size;
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  forward_thread_ = analyzer.Simplify(forward_thread);
  if (forward_index.empty()) {
    forward_index = {
        infer_fragment_index(getVarMap(), forward_thread_, &analyzer)};
  }
  forward_index_ = forward_index.Map(
      [&](const PrimExpr &e) { return analyzer.Simplify(e); });
}

Fragment::Fragment(Array<IterVar> forward_var, Array<PrimExpr> forward_index,
                   PrimExpr forward_thread, IterVar thread_replicate) {
  Map<Var, PrimExpr> vmap;
  Array<PrimExpr> input_size;
  PrimExpr replicate_size = 1;
  for (size_t i = 0; i < forward_var.size(); i++) {
    vmap.Set(forward_var[i]->var, InputPlaceholder(i));
    CHECK(is_zero(forward_var[i]->dom->min));
    input_size.push_back(forward_var[i]->dom->extent);
  }
  if (thread_replicate.defined()) {
    ICHECK(is_zero(thread_replicate->dom->min));
    replicate_size = thread_replicate->dom->extent;
    vmap.Set(thread_replicate->var, ReplicationPlaceholder());
  }
  forward_index =
      forward_index.Map([&](const PrimExpr &e) { return Substitute(e, vmap); });
  forward_thread = Substitute(forward_thread, vmap);

  auto n = tvm::ffi::make_object<FragmentNode>(input_size, forward_index,
                                               forward_thread, replicate_size);
  data_ = std::move(n);
}

Fragment::Fragment(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
                   PrimExpr forward_thread, PrimExpr replicate_size,
                   Optional<Var> replicate_var) {
  if (replicate_var.defined()) {
    forward_thread = Substitute(
        forward_thread, {{replicate_var.value(), ReplicationPlaceholder()}});
  }
  auto n = tvm::ffi::make_object<FragmentNode>(input_size, forward_index,
                                               forward_thread, replicate_size);
  data_ = std::move(n);
}

Fragment Fragment::FullyReplicated(Array<PrimExpr> shape,
                                   PrimExpr thread_extent) {
  return Fragment(shape, {}, ReplicationPlaceholder(), thread_extent,
                  std::nullopt);
}

// which means the forward_thread is rep_var -> lambda i, rep: rep
bool FragmentNode::IsCompletedReplicated() const {
  arith::Analyzer analyzer;
  return ExprDeepEqual()(analyzer.Simplify(forward_thread_),
                         ReplicationPlaceholder());
}

arith::IterMapResult FragmentNode::DetectInjective() const {
  // lei:To perform injective check, we need to reverse the layout
  // and use surjective check, now we use bijective check for convenience
  // can be relaxed in future
  arith::Analyzer analyzer;
  // Build a flat indices array: [forward_thread_, forward_index_[...]]
  Array<PrimExpr> indices;
  indices.push_back(forward_thread_);
  for (const auto &e : forward_index_) {
    indices.push_back(e);
  }

  // Mirror Layout::InverseWithLevel(): if any participating shape is
  // symbolic, relax to NoCheck and rely on runtime guards elsewhere.
  auto collect_symbolic = [&](const Array<PrimExpr> &shape) {
    Array<PrimExpr> symbolic_dims;
    for (const auto &dim : shape) {
      if (!as_const_int(dim)) {
        symbolic_dims.push_back(dim);
      }
    }
    return symbolic_dims;
  };

  Array<PrimExpr> symbolic_dims = collect_symbolic(InputShape());
  Array<PrimExpr> output_shape = OutputShape();
  symbolic_dims.insert(symbolic_dims.end(), output_shape.begin(),
                       output_shape.end());
  // Also consider replicate size for fragments
  if (!as_const_int(ReplicateExtent())) {
    symbolic_dims.push_back(ReplicateExtent());
  }
  symbolic_dims = collect_symbolic(symbolic_dims);

  bool is_static_shape = symbolic_dims.empty();
  auto level = is_static_shape ? arith::IterMapLevel::Bijective
                               : arith::IterMapLevel::NoCheck;
  if (!is_static_shape) {
    DLOG(WARNING)
        << "Fragment::DetectInjective on symbolic layout, falling back to "
        << "NoCheck; symbolic dims: " << symbolic_dims;
  }

  return arith::DetectIterMap(indices, getVarMap(), 1, level, &analyzer);
}

PrimExpr FragmentNode::ThreadExtent() const {
  Array<PrimExpr> ret(OutputDim(), 1);
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  auto ist = analyzer.int_set(forward_thread_ + 1);
  return ist.max();
}

Array<PrimExpr> FragmentNode::GetForwardVars() const {
  Array<PrimExpr> vars;
  if (*as_const_int(ReplicateExtent()) > 1) {
    vars.push_back(ReplicationPlaceholder());
  }
  for (size_t i = 0; i < InputDim(); i++) {
    vars.push_back(InputPlaceholder(i));
  }
  return vars;
}

PrimExpr FragmentNode::ForwardThread(const Array<PrimExpr> &vars,
                                     const Optional<PrimExpr> &rep_var) const {
  Map<Var, PrimExpr> vmap;
  ICHECK_EQ(vars.size(), InputDim());
  for (size_t i = 0; i < InputDim(); i++) {
    vmap.Set(InputPlaceholder(i), vars[i]);
  }
  if (rep_var.defined())
    vmap.Set(ReplicationPlaceholder(), rep_var.value());

  return Substitute(forward_thread_, vmap);
}

Layout FragmentNode::Inverse() const {
  auto result = InverseWithLevel();
  return std::move(result.first);
}

std::pair<Layout, arith::IterMapLevel> FragmentNode::InverseWithLevel() const {
  auto input_size_copy = input_size_;
  input_size_copy.push_back(ReplicateExtent());
  auto forward_index_copy = forward_index_;
  forward_index_copy.push_back(
      Substitute(forward_thread_,
                 {{ReplicationPlaceholder(), InputPlaceholder(InputDim())}}));
  auto fwd = Layout(input_size_copy, forward_index_copy);
  return fwd->InverseWithLevel();
}

Fragment FragmentNode::CondenseReplicateVar() const {
  arith::Analyzer analyzer;
  auto input_iters = getVarMap();
  input_iters.Set(ReplicationPlaceholder(), {0, ReplicateExtent()});
  PrimExpr new_forward_thread;
  IterVar new_thread_replicate;
  std::tie(new_forward_thread, new_thread_replicate) =
      CompressIterator(forward_thread_, ToIterVars(input_iters),
                       ReplicationPlaceholder(), &analyzer);
  return Fragment(input_size_, forward_index_, new_forward_thread,
                  new_thread_replicate->dom->extent, new_thread_replicate->var);
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
  if (thread_range_.defined()) {
    ss << ", thread_range: " << thread_range_;
  }
  ss << ")";
  return ss.str();
}

bool LayoutNode::IsEqual(const LayoutNode *other, bool skip_index) const {
  bool ret = StructuralEqual()(this->InputShape(), other->InputShape());
  ret &= StructuralEqual()(this->OutputShape(), other->OutputShape());
  if (!skip_index) {
    ret &= StructuralEqual()(this->forward_index_, other->forward_index_);
  }
  return ret;
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
  if (this->thread_range_.defined() && other->thread_range_.defined()) {
    ret &= StructuralEqual()(this->thread_range_, other->thread_range_);
  }
  ret &= StructuralEqual()(this->OutputShape(), other->OutputShape());
  ret &= StructuralEqual()(this->ReplicateExtent(), other->ReplicateExtent());
  ret &= StructuralEqual()(this->ThreadExtent(), other->ThreadExtent());
  if (!skip_index) {
    ret &= StructuralEqual()(this->forward_index_, other->forward_index_);
  }
  return ret;
}

void FragmentNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<FragmentNode>()
      .def_ro("forward_thread", &FragmentNode::forward_thread_)
      .def_ro("replicate_size", &FragmentNode::replicate_size_)
      .def("_DebugOutput", &FragmentNode::DebugOutput);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FragmentNode>([](const ObjectRef &obj, ReprPrinter *p) {
      auto *node = static_cast<const FragmentNode *>(obj.get());
      p->stream << node->DebugOutput();
    })
    .set_dispatch<LayoutNode>([](const ObjectRef &obj, ReprPrinter *p) {
      auto *node = static_cast<const LayoutNode *>(obj.get());
      p->stream << node->DebugOutput();
    });

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("tl.Layout",
                  [](PackedArgs args, Any *rv) {
                    *rv = Layout(args[0].cast<Array<IterVar>>(),
                                 args[1].cast<Array<PrimExpr>>());
                  })
      .def("tl.Layout_input_shape",
           [](Layout layout) { return layout->InputShape(); })
      .def("tl.Layout_output_shape",
           [](Layout layout) { return layout->OutputShape(); })
      .def("tl.Layout_inverse", [](Layout layout) { return layout->Inverse(); })
      .def("tl.Layout_index",
           [](Layout layout) { return layout->GetForwardIndex(); })
      .def("tl.Layout_forward_vars",
           [](Layout layout) { return layout->GetForwardVars(); })
      .def("tl.Layout_is_equal",
           [](Layout layout, Layout other) {
             const LayoutNode *other_node = other.as<LayoutNode>();
             return layout->IsEqual(other_node);
           })
      .def_packed("tl.Fragment",
                  [](PackedArgs args, Any *rv) {
                    *rv = Fragment(
                        /*forward_var=*/args[0].cast<Array<IterVar>>(),
                        /*forward_index=*/args[1].cast<Array<PrimExpr>>(),
                        /*forward_thread=*/args[2].cast<PrimExpr>(),
                        /*thread_replicate=*/args[3].cast<IterVar>());
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
           [](int stride, int continuous, int element_size, bool k_inner,
              bool allow_pad = true) {
             if (allow_pad) {
               return makeGemmABLayout(stride, continuous, continuous,
                                       element_size, k_inner);
             } else {
               return makeGemmABLayoutHopper(stride, continuous, continuous,
                                             element_size, k_inner);
             }
           })
      .def("tl.make_volta_swizzled_layout",
           [](int stride, int mat_continuous, bool is_a, bool k_inner) {
             return makeGemmVoltaABLayout(stride, mat_continuous, is_a,
                                          k_inner);
           })
      .def("tl.make_wgmma_swizzled_layout",
           [](int stride, int mat_continuous, int continuity, int element_size,
              bool k_inner) {
             return makeGemmABLayoutHopper(stride, mat_continuous, continuity,
                                           element_size, k_inner);
           })
      .def("tl.make_tcgen05mma_swizzled_layout",
           [](int stride, int mat_continuous, int continuity, int element_size,
              bool k_inner) {
             return makeGemmABLayoutSm100(stride, mat_continuous, continuity,
                                          element_size, k_inner);
           })
      .def("tl.make_full_bank_swizzled_layout",
           [](int stride, int continuous, int element_size) {
             return makeFullBankSwizzleLayout(stride, continuous, element_size);
           })
      .def("tl.make_half_bank_swizzled_layout",
           [](int stride, int continuous, int element_size) {
             return makeHalfBankSwizzleLayout(stride, continuous, element_size);
           })
      .def("tl.make_quarter_bank_swizzled_layout",
           [](int stride, int continuous, int element_size) {
             return makeQuarterBankSwizzleLayout(stride, continuous,
                                                 element_size);
           })
      .def("tl.make_linear_layout",
           [](Array<PrimExpr> shape) { return makeLinearLayout(shape); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  LayoutNode::RegisterReflection();
  FragmentNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
