/*!
 * \file materialize_cpu_parallel_grid.cc
 * \brief Prove native CPU grid independence before enabling parallel loops.
 *
 * Tensor identity, aliases, physical layout and allocation lifetime are checked
 * together. An inconclusive proof keeps the source grid serial. C collapses a
 * perfect grid nest; LLVM parallelizes only its first non-unit dimension.
 */

#include "cpu/op/builtin.h"
#include "ir/analyzer.h"
#include "ir/annotations.h"
#include "ir/lexical_scope.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/prim/vector_expr.h>
#include <tvm/sym/analyzer.h>
#include <tvm/sym/iter_affine_map.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace prim;
using tirx::StmtExprMutator;
using tirx::StmtExprVisitor;
using tirx::TensorType;
using tirx::TensorTypeObj;
using tirx::TensorVar;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
template <class T>
using VarMap = std::unordered_map<Var, T, ObjectPtrHash, ObjectPtrEqual>;
using Ranges = Map<PrimVar, Range>;

std::optional<int64_t> ConstInt(const PrimExpr &expr) {
  if (const auto *value = expr.as<IntImmObj>())
    return value->value.as<int64_t>();
  return std::nullopt;
}

PrimExpr Extent(const ForObj *op) {
  return sym::Analyzer()->Simplify(op->end - op->begin);
}

bool HasGridAnnotation(const ForObj *op) {
  return op && op->annotations.count(attr::kCPUGridDim);
}

Map<String, Any> StripGridAnnotation(Map<String, Any> annotations) {
  annotations.erase(attr::kCPUGridDim);
  return annotations;
}

bool IsTransparent(const RegionStmtObj *op) {
  // These regions execute their body once without changing thread context.
  // Analyze through their bodies, but preserve the lexical allocation boundary
  // when rebuilding the grid. Other operators need their own analysis.
  return op->body_params.empty() && op->result_vars.empty() &&
         (op->op->name == "tl.attr" || op->op->name == "ir.scope" ||
          op->op.same_as(tl::lexical_scope_op()));
}

bool IsDefinitionCall(const CallObj *op) {
  return op && (op->op.same_as(tirx::alloc_tensor_op()) ||
                op->op.same_as(tirx::decl_tensor_op()));
}

bool IsAnnotationCall(const CallObj *op) { return IsAnnotationCall(op); }

struct StorageRef {
  Var root;
  // Constant physical byte offset from root. A dynamic/unknown offset may
  // still alias root but cannot justify a parallel access or a full reset.
  std::optional<int64_t> byte_offset;
};

/*! \brief Resolve native tensor/pointer definitions before visiting accesses.
 * Definitions alone are not memory reads or escapes. Alias declarations that
 * move with a private allocation must move as one ordered definition group.
 */
class GridAccessAnalysis : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  explicit GridAccessAnalysis(const tirx::Function &function,
                              const ForObj *focus = nullptr)
      : no_alias_(function->attrs.GetAttr<bool>("tirx.noalias").value_or(true)),
        focus_(focus) {
    for (const Var &param : function->params) {
      if (param->ty.as<TensorTypeObj>() || param->ty.as<PtrTypeObj>()) {
        storage_.emplace(param, StorageRef{param, 0});
        parameter_roots_.insert(param);
      }
    }
    if (auto values =
            function->attrs.GetAttr<Array<Var>>(attr::kNonRestrictParams))
      for (const Var &param : values.value())
        non_restrict_.insert(param);
  }

  struct AccessInfo {
    Optional<TensorVar> allocation;
    int min_depth = 0;
    bool inside = false;
    bool outside = false;
    bool opaque = false;
    bool opaque_inside = false;
    bool store_inside = false;
    bool movable = true;
    std::unordered_set<const ForObj *> nests;
    std::unordered_set<const BindObj *> definitions;
  };

  StorageRef Resolve(const Var &var) const {
    auto it = storage_.find(var);
    return it == storage_.end() ? StorageRef{var, std::nullopt} : it->second;
  }

  std::optional<StorageRef> ResolvePointer(const Expr &expr) const {
    if (auto var = expr.as<Var>()) {
      auto it = storage_.find(var.value());
      if (it != storage_.end())
        return it->second;
      if (var.value()->ty.as<PtrTypeObj>())
        return StorageRef{var.value(), 0};
      return std::nullopt;
    }
    const auto *call = expr.as<CallObj>();
    if (!call || call->args.empty())
      return std::nullopt;
    if (call->op.same_as(tirx::tensor_data_ptr_op())) {
      if (auto tensor = call->args[0].as<Var>())
        return Resolve(tensor.value());
    }
    if (call->op.same_as(tirx::reinterpret_op()))
      return ResolvePointer(call->args[0]);
    if ((call->op.same_as(tirx::ptr_byte_offset_op()) ||
         call->op.same_as(tirx::handle_add_byte_offset_op())) &&
        call->args.size() == 2) {
      auto base = ResolvePointer(call->args[0]);
      if (!base.has_value())
        return std::nullopt;
      auto extra = call->args[1].as<PrimExpr>();
      auto offset = extra.has_value() ? ConstInt(extra.value()) : std::nullopt;
      auto &old = base.value().byte_offset;
      if (!old.has_value() || !offset.has_value() ||
          (offset.value() > 0 &&
           old.value() >
               std::numeric_limits<int64_t>::max() - offset.value()) ||
          (offset.value() < 0 &&
           old.value() <
               std::numeric_limits<int64_t>::min() - offset.value())) {
        old = std::nullopt;
      } else {
        old = old.value() + offset.value();
      }
      return base;
    }
    return std::nullopt;
  }

  AccessInfo Lookup(const Var &var) const {
    auto it = info_.find(Resolve(var).root);
    return it == info_.end() ? AccessInfo{} : it->second;
  }

  const VarMap<int> &AllocDepths() const { return alloc_depths_; }
  bool FoundFocus() const { return seen_focus_; }
  bool HasAllocation(const Var &var) const {
    return alloc_depths_.count(Resolve(var).root);
  }
  Optional<Var> DefinitionRoot(const BindObj *bind) const {
    auto it = definition_roots_.find(bind);
    return it == definition_roots_.end() ? Optional<Var>{} : it->second;
  }
  bool MayAlias(const Var &left, const Var &right) const {
    Var lhs = Resolve(left).root, rhs = Resolve(right).root;
    return lhs.same_as(rhs) || !no_alias_ || !parameter_roots_.count(lhs) ||
           !parameter_roots_.count(rhs) ||
           (non_restrict_.count(lhs) && non_restrict_.count(rhs));
  }

  Optional<PrimExpr> PhysicalIndex(const TensorVar &tensor,
                                   const Array<PrimExpr> &indices) const {
    auto location = Resolve(tensor);
    if (!location.byte_offset.has_value())
      return std::nullopt;
    // ElemOffset is valid only after the type describes physical memory.
    // A memory-only strided pattern is also safe before flattening. Thread
    // ownership, replicas and arbitrary composed layouts need tile lowering.
    auto pattern = tirx::MatchShapeStridePattern(tensor.ty());
    if (!pattern.has_value())
      return std::nullopt;
    auto offsets = tensor.ty()->ElemOffset(indices);
    if (offsets.size() != 1)
      return std::nullopt;
    auto inliner = make_object<StmtExprMutator>();
    for (const auto &[variable, value] : pure_bindings_)
      inliner->VarRemapSet(variable, value);
    PrimExpr index = inliner->Mutate(offsets[0]).ValueOrUnchanged(offsets[0]);
    if (location.byte_offset.value() == 0)
      return index;
    PrimType dtype = tensor.dtype();
    if (dtype.IsScalableVector() || dtype.bits() < 8)
      return std::nullopt;
    int64_t bytes = (dtype.bits() * dtype.lanes()) / 8;
    if (bytes <= 0 || location.byte_offset.value() % bytes != 0)
      return std::nullopt;
    // Keep all iterator identities and physical offset widths explicit.
    // A scalar 64-bit cast here is lossless for the supported index dtypes.
    if (!index.ty().IsScalar() || !index.ty().MatchesCode(kDLInt) ||
        index.ty().bits() > 64)
      return std::nullopt;
    return Cast(PrimType::Int(64), index) +
           IntImm(PrimType::Int(64), location.byte_offset.value() / bytes);
  }

  Optional<VisitInterrupt> Visit_(const ForObj *op) final {
    bool enter_focus = op == focus_;
    if (enter_focus) {
      inside_focus_ = true;
      seen_focus_ = true;
    }
    bool grid = HasGridAnnotation(op) && (!focus_ || inside_focus_);
    if (grid && grid_depth_++ == 0)
      current_nest_ = op;
    auto result = StmtExprVisitor::Visit_(op);
    if (grid && --grid_depth_ == 0)
      current_nest_ = nullptr;
    if (enter_focus)
      inside_focus_ = false;
    return result;
  }

  Optional<VisitInterrupt> Visit_(const BindObj *op) final {
    if (const auto *call = op->value.as<CallObj>(); IsDefinitionCall(call)) {
      TensorVar tensor(op->var);
      StorageRef resolved{op->var, 0};
      if (call->op.same_as(tirx::alloc_tensor_op())) {
        alloc_depths_[op->var] = grid_depth_;
        info_[op->var].allocation = tensor;
        // Explicit placement and effectful shape evaluation are not duplicated.
        if (!call->args.empty())
          info_[op->var].movable = false;
        for (const auto &dimension : tensor.shape())
          if (SideEffect(dimension) != CallEffectKind::kPure)
            info_[op->var].movable = false;
      } else {
        auto base = ResolvePointer(call->args[0]);
        if (base.has_value())
          resolved = base.value();
        else
          resolved.byte_offset = std::nullopt;
      }
      storage_.insert_or_assign(op->var, resolved);
      info_[resolved.root].definitions.insert(op);
      definition_roots_.emplace(op, resolved.root);
      return std::nullopt;
    }
    if (op->var->ty.as<TensorTypeObj>() || op->var->ty.as<PtrTypeObj>()) {
      if (auto resolved = ResolvePointer(op->value)) {
        storage_.insert_or_assign(op->var, resolved.value());
        info_[resolved.value().root].definitions.insert(op);
        definition_roots_.emplace(op, resolved.value().root);
        return std::nullopt;
      }
    }
    // Native scalar Bind is an SSA definition. Expand only pure values for
    // address proofs; a load or external call must remain at its source point.
    if (op->var->ty.as<PrimTypeObj>()) {
      if (auto value = op->value.as<PrimExpr>();
          value.has_value() &&
          SideEffect(value.value()) == CallEffectKind::kPure) {
        auto inliner = make_object<StmtExprMutator>();
        for (const auto &[variable, bound] : pure_bindings_)
          inliner->VarRemapSet(variable, bound);
        pure_bindings_.emplace(
            op->var,
            inliner->Mutate(value.value()).ValueOrUnchanged(value.value()));
      }
    }
    return StmtExprVisitor::Visit_(op);
  }

  Optional<VisitInterrupt> Visit_(const TensorStoreObj *op) final {
    if (auto tensor = op->dest.as<TensorVar>()) {
      Touch(tensor.value(), true);
      for (const auto &index : op->indices)
        Visit(index);
      return Visit(op->value);
    }
    return StmtExprVisitor::Visit_(op);
  }
  Optional<VisitInterrupt> Visit_(const TensorLoadObj *op) final {
    if (auto tensor = op->source.as<TensorVar>()) {
      Touch(tensor.value(), false);
      for (const auto &index : op->indices)
        Visit(index);
      return std::nullopt;
    }
    return StmtExprVisitor::Visit_(op);
  }
  Optional<VisitInterrupt> Visit_(const VarObj *op) final {
    if (def_region_kind() == kTVMFFIDefRegionKindNone &&
        (op->ty.as<TensorTypeObj>() || op->ty.as<PtrTypeObj>()))
      TouchOpaque(GetRef<Var>(op));
    return StmtExprVisitor::Visit_(op);
  }
  Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    if (IsAnnotationCall(op))
      return std::nullopt;
    if (op->op.same_as(tirx::address_of_op()) && !op->args.empty()) {
      if (const auto *load = op->args[0].as<TensorLoadObj>())
        if (auto tensor = load->source.as<Var>())
          TouchOpaque(tensor.value());
    }
    return StmtExprVisitor::Visit_(op);
  }
  Optional<VisitInterrupt> Visit_(const RegionStmtObj *op) final {
    if (IsTransparent(op))
      return Visit(op->body);
    return StmtExprVisitor::Visit_(op);
  }

private:
  void TouchOpaque(const Var &var) {
    auto &entry = info_[Resolve(var).root];
    entry.opaque = true;
    entry.min_depth = 0;
    if (grid_depth_) {
      entry.opaque_inside = entry.inside = true;
      entry.nests.insert(current_nest_);
    }
  }
  void Touch(const Var &var, bool store) {
    auto &entry = info_[Resolve(var).root];
    if (grid_depth_ == 0) {
      entry.min_depth = 0;
      entry.outside = true;
    } else {
      entry.min_depth =
          entry.inside ? std::min(entry.min_depth, grid_depth_) : grid_depth_;
      entry.inside = true;
      entry.store_inside |= store;
      entry.nests.insert(current_nest_);
    }
  }

  VarMap<StorageRef> storage_;
  VarMap<PrimExpr> pure_bindings_;
  VarMap<AccessInfo> info_;
  VarMap<int> alloc_depths_;
  std::unordered_map<const BindObj *, Var> definition_roots_;
  VarSet parameter_roots_, non_restrict_;
  bool no_alias_;
  const ForObj *focus_;
  bool inside_focus_ = false;
  bool seen_focus_ = false;
  const ForObj *current_nest_ = nullptr;
  int grid_depth_ = 0;
};

PrimExpr ScalarizedIndex(const PrimExpr &index, Ranges *ranges) {
  if (const auto *ramp = index.as<RampObj>()) {
    PrimVar lane("affine_lane", ramp->base.ty());
    auto lanes = Cast(ramp->base.ty(), ramp->lanes);
    ranges->Set(lane, Range(IntImm(lane.ty(), 0), lanes));
    return ramp->base + lane * ramp->stride;
  }
  return index;
}

class CastVarFreshener : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit CastVarFreshener(sym::Analyzer analyzer)
      : analyzer_(std::move(analyzer)) {}
  bool valid = true;
  // Each cast keeps its originating iterator identity. Two surviving
  // representations of one iterator cannot be independent input dimensions.
  VarMap<PrimVar> origins;
  VarMap<std::unordered_map<int, PrimVar>> fresh;
  UnchangedOr<PrimExpr> Mutate_(const CastObj *op, InplaceMode mode) final {
    PrimType target = op->ty.as_or_throw<PrimType>();
    auto value = Mutate(op->value, mode).ValueOrUnchanged(op->value);
    auto candidate = value.as<PrimVar>();
    if (candidate.has_value() && target.IsScalar() &&
        target.MatchesCode(kDLInt) && candidate.value().ty().IsScalar() &&
        candidate.value().ty().MatchesCode(kDLInt)) {
      PrimVar variable = candidate.value();
      auto origin_it = origins.find(variable);
      PrimVar original =
          origin_it == origins.end() ? variable : origin_it->second;
      if (target.bits() < original.ty().bits()) {
        auto bound = analyzer_->const_int_bound(original);
        if (target.bits() >= 64) {
          if (bound->min_value == sym::ConstIntBound::kNegInf ||
              bound->max_value == sym::ConstIntBound::kPosInf) {
            valid = false;
            return Cast(target, value, op->loc);
          }
        } else {
          int64_t limit = int64_t{1} << (target.bits() - 1);
          if (bound->min_value < -limit || bound->max_value >= limit) {
            valid = false;
            return Cast(target, value, op->loc);
          }
        }
      }
      if (target == variable.ty())
        return variable;
      auto &representations = fresh[original];
      auto found = representations.find(target.bits());
      if (found == representations.end()) {
        PrimVar replacement(
            original->name + "_i" + std::to_string(target.bits()), target);
        origins.emplace(replacement, original);
        found = representations.emplace(target.bits(), replacement).first;
      }
      return found->second;
    }
    return Cast(target, value, op->loc);
  }

private:
  sym::Analyzer analyzer_;
};

class IndexVariableCollector : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  VarSet used;
  Optional<VisitInterrupt> Visit_(const VarObj *op) final {
    used.insert(GetRef<Var>(op));
    return std::nullopt;
  }
};

bool ProveInjectiveIndex(const PrimExpr &index, const Ranges &ranges) {
  if (!index.ty().IsScalar() || !index.ty().MatchesCode(kDLInt))
    return false;
  sym::Analyzer analyzer;
  for (const auto &[var, range] : ranges)
    BindAnalyzer(analyzer, var, range);
  auto freshener = make_object<CastVarFreshener>(analyzer);
  auto normalized = freshener->Mutate(index).ValueOrUnchanged(index);
  if (!freshener->valid)
    return false;
  auto variables = make_object<IndexVariableCollector>();
  variables->Visit(normalized);
  Ranges normalized_ranges;
  for (const auto &[var, range] : ranges) {
    PrimVar representative = var;
    size_t count = variables->used.count(var);
    auto found = freshener->fresh.find(var);
    if (found != freshener->fresh.end())
      for (const auto &[bits, replacement] : found->second)
        if (variables->used.count(replacement)) {
          representative = replacement;
          ++count;
        }
    if (count > 1)
      return false;
    // Keep the original wide exclusive endpoint: [0, 128) is a valid
    // int8 iterator domain, but int8(128) would become -128.
    normalized_ranges.Set(representative, range);
  }
  normalized = analyzer->Simplify(normalized);
  auto result = sym::DetectIterMap({normalized}, normalized_ranges,
                                   MakeConst(PrimType::Bool(), true),
                                   sym::IterMapLevel::Bijective, analyzer);
  return result->errors.empty();
}

class SameStorageReadFinder : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  SameStorageReadFinder(const GridAccessAnalysis &analysis, Var root)
      : analysis_(analysis), root_(std::move(root)) {}
  bool found = false;
  Optional<VisitInterrupt> Visit_(const TensorLoadObj *op) final {
    if (auto tensor = op->source.as<Var>())
      found |= analysis_.Resolve(tensor.value()).root.same_as(root_);
    return StmtExprVisitor::Visit_(op);
  }

private:
  const GridAccessAnalysis &analysis_;
  Var root_;
};

/*! \brief A reset covers the backing allocation, not merely the current view.
 * Require injective, bounded physical addresses and exact byte capacity. This
 * also rejects packed/sub-byte/padded vector allocations as proof premises.
 */
std::optional<size_t> FullRewriteScope(const TensorStoreObj *store,
                                       const TensorVar &allocation,
                                       const std::vector<For> &enclosing,
                                       const GridAccessAnalysis &analysis) {
  auto tensor = store->dest.as<TensorVar>();
  if (!tensor.has_value() || !allocation.ty()->HasDefaultLayout())
    return std::nullopt;
  PrimType alloc_type = allocation.dtype(),
           element_type = tensor.value().dtype();
  if (alloc_type.IsScalableVector() || element_type.IsScalableVector())
    return std::nullopt;
  int64_t allocation_bits = alloc_type.bits() * alloc_type.lanes();
  int64_t element_bits = element_type.bits() * element_type.lanes();
  if (alloc_type.bits() < 8 || element_type.bits() < 8 ||
      allocation_bits <= 0 || element_bits <= 0 ||
      (allocation_bits & (allocation_bits - 1)) ||
      (element_bits & (element_bits - 1)))
    return std::nullopt;
  for (const auto &dimension : allocation.shape()) {
    auto extent = ConstInt(dimension);
    if (!extent.has_value() || extent.value() <= 0 ||
        allocation_bits > std::numeric_limits<int64_t>::max() / extent.value())
      return std::nullopt;
    allocation_bits *= extent.value();
  }
  if (allocation_bits % element_bits)
    return std::nullopt;
  int64_t numel = allocation_bits / element_bits;
  auto index = analysis.PhysicalIndex(tensor.value(), store->indices);
  if (!index.has_value())
    return std::nullopt;
  int64_t lanes = 1;
  if (const auto *ramp = index.value().as<RampObj>()) {
    auto count = ConstInt(ramp->lanes);
    if (!count.has_value() || count.value() <= 0)
      return std::nullopt;
    lanes = count.value();
  } else if (index.value().as<BroadcastObj>()) {
    return std::nullopt;
  }
  for (size_t start = enclosing.size() + 1; start-- > 0;) {
    Ranges ranges;
    int64_t trip = 1;
    bool valid = true;
    for (size_t i = start; i < enclosing.size(); ++i) {
      const auto &loop = enclosing[i];
      auto extent = ConstInt(Extent(loop.get()));
      if (!extent.has_value() || extent.value() <= 0 || !IsZero(loop->begin) ||
          !IsOne(loop->step) || trip > numel / extent.value()) {
        valid = false;
        break;
      }
      ranges.Set(loop->var, Range(loop->begin, loop->end));
      trip *= extent.value();
    }
    if (!valid || numel % lanes || trip != numel / lanes)
      continue;
    auto scalar_index = ScalarizedIndex(index.value(), &ranges);
    if (!scalar_index.ty().IsScalar())
      continue;
    sym::Analyzer analyzer;
    for (const auto &[var, range] : ranges)
      BindAnalyzer(analyzer, var, range);
    auto bound = analyzer->const_int_bound(scalar_index);
    if (bound->min_value >= 0 && bound->max_value < numel &&
        ProveInjectiveIndex(scalar_index, ranges))
      return start;
  }
  return std::nullopt;
}

class IterationPrivacyChecker : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  explicit IterationPrivacyChecker(const GridAccessAnalysis &analysis)
      : analysis_(analysis) {}
  bool ReadsPrevious(const Var &var) const {
    return reads_previous_.count(analysis_.Resolve(var).root);
  }
  Optional<VisitInterrupt> Visit_(const ForObj *op) final {
    auto before = initialized_;
    loop_stack_.push_back(GetRef<For>(op));
    loop_resets_.emplace_back();
    auto result = StmtExprVisitor::Visit_(op);
    auto extent = ConstInt(Extent(op));
    if (extent.has_value() && extent.value() > 0 && IsOne(op->step))
      initialized_.insert(loop_resets_.back().begin(),
                          loop_resets_.back().end());
    else
      initialized_ = std::move(before);
    loop_resets_.pop_back();
    loop_stack_.pop_back();
    return result;
  }
  Optional<VisitInterrupt> Visit_(const IfObj *op) final {
    ++conditional_depth_;
    auto result = StmtExprVisitor::Visit_(op);
    --conditional_depth_;
    return result;
  }
  Optional<VisitInterrupt> Visit_(const WhileObj *op) final {
    ++conditional_depth_;
    auto result = StmtExprVisitor::Visit_(op);
    --conditional_depth_;
    return result;
  }
  Optional<VisitInterrupt> Visit_(const RegionStmtObj *op) final {
    if (IsTransparent(op))
      return Visit(op->body);
    ++conditional_depth_;
    auto result = StmtExprVisitor::Visit_(op);
    --conditional_depth_;
    return result;
  }
  Optional<VisitInterrupt> Visit_(const BindObj *op) final {
    if (analysis_.DefinitionRoot(op).has_value())
      return std::nullopt;
    return StmtExprVisitor::Visit_(op);
  }
  Optional<VisitInterrupt> Visit_(const TensorStoreObj *op) final {
    auto result = StmtExprVisitor::Visit_(op);
    auto tensor = op->dest.as<TensorVar>();
    if (!tensor.has_value() || conditional_depth_)
      return result;
    auto info = analysis_.Lookup(tensor.value());
    if (!info.allocation.has_value())
      return result;
    auto root = analysis_.Resolve(tensor.value()).root;
    auto finder = make_object<SameStorageReadFinder>(analysis_, root);
    finder->Visit(op->value);
    if (finder->found)
      return result;
    if (auto scope = FullRewriteScope(op, info.allocation.value(), loop_stack_,
                                      analysis_)) {
      if (scope.value() == loop_stack_.size())
        initialized_.insert(root);
      else
        loop_resets_[scope.value()].insert(root);
    }
    return result;
  }
  Optional<VisitInterrupt> Visit_(const TensorLoadObj *op) final {
    if (auto tensor = op->source.as<Var>()) {
      auto root = analysis_.Resolve(tensor.value()).root;
      if (!initialized_.count(root))
        reads_previous_.insert(root);
    }
    return StmtExprVisitor::Visit_(op);
  }

private:
  const GridAccessAnalysis &analysis_;
  VarSet initialized_, reads_previous_;
  std::vector<For> loop_stack_;
  std::vector<VarSet> loop_resets_;
  int conditional_depth_ = 0;
};

class UnsafeEffectFinder : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  bool unsafe = false;
  bool parallel = false;
  Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    static const auto effects = Op::GetAttrMap<int>("TCallEffectKind");
    auto callee = op->op.as<Op>();
    if (!IsDefinitionCall(op) && !IsAnnotationCall(op) &&
        (!callee.has_value() || !effects.count(callee.value()) ||
         effects[callee.value()] > static_cast<int>(CallEffectKind::kPure)))
      unsafe = true;
    return StmtExprVisitor::Visit_(op);
  }
  Optional<VisitInterrupt> Visit_(const ForObj *op) final {
    parallel |= op->kind == ForKind::kParallel;
    return StmtExprVisitor::Visit_(op);
  }
  Optional<VisitInterrupt> Visit_(const BreakObj *) final {
    unsafe = true;
    return std::nullopt;
  }
  Optional<VisitInterrupt> Visit_(const ContinueObj *) final {
    unsafe = true;
    return std::nullopt;
  }
  Optional<VisitInterrupt> Visit_(const ReturnObj *op) final {
    unsafe = true;
    return StmtExprVisitor::Visit_(op);
  }
  Optional<VisitInterrupt> Visit_(const RegionStmtObj *op) final {
    if (!IsTransparent(op))
      unsafe = true;
    return Visit(op->body);
  }
};

class OverlapStoreChecker : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  OverlapStoreChecker(const GridAccessAnalysis &analysis, Ranges scopes)
      : analysis_(analysis), scopes_(std::move(scopes)) {}
  bool found = false;
  std::string buffer_name, reason;

  void Finish() {
    if (found)
      return;
    sym::Analyzer analyzer;
    for (const auto &[root, access] : accesses_) {
      if (access.opaque) {
        Reject(root, "opaque pointer access");
        return;
      }
      if (!access.stores.empty()) {
        if (access.mixed_width) {
          Reject(root, "buffer views use different element widths");
          return;
        }
        for (const auto &[other, other_access] : accesses_)
          if (!root.same_as(other) && analysis_.MayAlias(root, other)) {
            Reject(root, "may alias buffer `" + std::string(other->name) + "`");
            return;
          }
      }
      Optional<PrimExpr> uniform;
      for (const auto &store : access.stores) {
        if (!store.index.has_value()) {
          Reject(root, "unanalyzable store address");
          return;
        }
        auto simplified = analyzer->Simplify(store.index.value());
        if (uniform.has_value() &&
            !StructuralEqual()(uniform.value(), simplified)) {
          Reject(root, "stores to one buffer cover each other's addresses");
          return;
        }
        uniform = simplified;
        auto ranges = store.ranges;
        if (!ProveInjectiveIndex(ScalarizedIndex(store.index.value(), &ranges),
                                 ranges)) {
          Reject(root, "write addresses collide across grid iterations");
          return;
        }
      }
      if (uniform.has_value())
        for (const auto &load : access.loads)
          if (!load.has_value() ||
              !StructuralEqual()(uniform.value(),
                                 analyzer->Simplify(load.value()))) {
            Reject(root, "cross-iteration dependency (loop-carried load)");
            return;
          }
    }
  }

  Optional<VisitInterrupt> Visit_(const ForObj *op) final {
    if (!IsOne(op->step)) {
      found = true;
      reason = "non-unit inner loop step";
    }
    scopes_.Set(op->var, Range(op->begin, op->end));
    auto result = StmtExprVisitor::Visit_(op);
    scopes_.erase(op->var);
    return result;
  }
  Optional<VisitInterrupt> Visit_(const BindObj *op) final {
    if (analysis_.DefinitionRoot(op).has_value())
      return std::nullopt;
    return StmtExprVisitor::Visit_(op);
  }
  Optional<VisitInterrupt> Visit_(const TensorStoreObj *op) final {
    if (auto tensor = op->dest.as<TensorVar>()) {
      if (!analysis_.HasAllocation(tensor.value())) {
        auto &access = Record(tensor.value());
        access.stores.push_back(
            {analysis_.PhysicalIndex(tensor.value(), op->indices), scopes_});
      }
      for (const auto &index : op->indices)
        Visit(index);
      return Visit(op->value);
    }
    found = true;
    reason = "tensor store destination is not a bound native tensor";
    return StmtExprVisitor::Visit_(op);
  }
  Optional<VisitInterrupt> Visit_(const TensorLoadObj *op) final {
    if (auto tensor = op->source.as<TensorVar>()) {
      if (!analysis_.HasAllocation(tensor.value()))
        Record(tensor.value())
            .loads.push_back(
                analysis_.PhysicalIndex(tensor.value(), op->indices));
      for (const auto &index : op->indices)
        Visit(index);
      return std::nullopt;
    }
    found = true;
    reason = "tensor load source is not a bound native tensor";
    return StmtExprVisitor::Visit_(op);
  }
  Optional<VisitInterrupt> Visit_(const VarObj *op) final {
    if (def_region_kind() == kTVMFFIDefRegionKindNone &&
        (op->ty.as<TensorTypeObj>() || op->ty.as<PtrTypeObj>()) &&
        !analysis_.HasAllocation(GetRef<Var>(op)))
      accesses_[analysis_.Resolve(GetRef<Var>(op)).root].opaque = true;
    return StmtExprVisitor::Visit_(op);
  }
  Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    if (IsAnnotationCall(op))
      return std::nullopt;
    if (op->op.same_as(tirx::address_of_op()) && !op->args.empty()) {
      if (const auto *load = op->args[0].as<TensorLoadObj>())
        if (auto tensor = load->source.as<Var>();
            tensor.has_value() && !analysis_.HasAllocation(tensor.value()))
          accesses_[analysis_.Resolve(tensor.value()).root].opaque = true;
    }
    return StmtExprVisitor::Visit_(op);
  }
  Optional<VisitInterrupt> Visit_(const RegionStmtObj *op) final {
    if (IsTransparent(op))
      return Visit(op->body);
    return StmtExprVisitor::Visit_(op);
  }

private:
  struct Store {
    Optional<PrimExpr> index;
    Ranges ranges;
  };
  struct Access {
    std::vector<Store> stores;
    std::vector<Optional<PrimExpr>> loads;
    int element_bits = 0;
    bool mixed_width = false;
    bool opaque = false;
  };
  Access &Record(const TensorVar &tensor) {
    auto &access = accesses_[analysis_.Resolve(tensor).root];
    if (tensor.dtype().IsScalableVector()) {
      access.mixed_width = true;
      return access;
    }
    int width = tensor.dtype().bits() * tensor.dtype().lanes();
    access.mixed_width |= access.element_bits && access.element_bits != width;
    access.element_bits = width;
    return access;
  }
  void Reject(const Var &root, std::string why) {
    found = true;
    buffer_name = root->name;
    reason = std::move(why);
  }
  const GridAccessAnalysis &analysis_;
  Ranges scopes_;
  VarMap<Access> accesses_;
};

class GridRewriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  GridRewriter(bool collapse, int64_t min_trip, tirx::Function function,
               bool force_serial)
      : collapse_(collapse), min_trip_(min_trip),
        function_(std::move(function)), force_serial_(force_serial) {}

  static const ForObj *TransparentNestHead(const Stmt &stmt) {
    if (const auto *loop = stmt.as<ForObj>())
      return HasGridAnnotation(loop) ? loop : nullptr;
    if (const auto *sequence = stmt.as<SeqStmtObj>();
        sequence && sequence->seq.size() == 1)
      return TransparentNestHead(sequence->seq[0]);
    if (const auto *region = stmt.as<RegionStmtObj>();
        region && IsTransparent(region))
      return TransparentNestHead(region->body);
    return nullptr;
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    Array<Stmt> out;
    available_sequences_.push_back(&out);
    for (const auto &element : op->seq) {
      const auto *head = in_parallel_ ? nullptr : TransparentNestHead(element);
      if (head)
        out.push_back(
            ConvertNestElement(element, head, CollectSinkableAllocs(head)));
      else
        out.push_back(Mutate(element, mode).ValueOrUnchanged(element));
    }
    available_sequences_.pop_back();
    return SeqStmt(std::move(out), op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    if (IsTransparent(op))
      return StmtExprMutator::Mutate_(op, mode);
    DefinitionBoundary boundary(&available_begin_, available_sequences_.size());
    return StmtExprMutator::Mutate_(op, mode);
  }

  UnchangedOr<Stmt> Mutate_(const IfObj *op, InplaceMode mode) final {
    DefinitionBoundary boundary(&available_begin_, available_sequences_.size());
    return StmtExprMutator::Mutate_(op, mode);
  }

  UnchangedOr<Stmt> Mutate_(const WhileObj *op, InplaceMode mode) final {
    DefinitionBoundary boundary(&available_begin_, available_sequences_.size());
    return StmtExprMutator::Mutate_(op, mode);
  }

  UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    if (!HasGridAnnotation(op))
      return StmtExprMutator::Mutate_(op, mode);
    if (in_parallel_) {
      LOG(WARNING) << "tl.cpu_parallel: nested grid `" << op->var->name
                   << "` stays serial";
      auto body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
      return For(op->var, op->begin, op->end, op->step, ForKind::kDefault, body,
                 StripGridAnnotation(op->annotations), op->loc);
    }
    return ConvertThenVisit(op, {});
  }

private:
  // Ancestor allocations can move through serial loops and transparent
  // metadata scopes, but not through a context-changing/conditional region.
  struct DefinitionBoundary {
    DefinitionBoundary(size_t *begin, size_t boundary)
        : begin(begin), previous(std::exchange(*begin, boundary)) {}
    ~DefinitionBoundary() { *begin = previous; }
    size_t *begin;
    size_t previous;
  };

  static std::vector<const ForObj *> CollectChain(const ForObj *head) {
    std::vector<const ForObj *> loops;
    for (const auto *current = head; HasGridAnnotation(current);) {
      loops.push_back(current);
      current = current->body->seq.size() == 1
                    ? current->body->seq[0].as<ForObj>()
                    : nullptr;
    }
    return loops;
  }
  size_t SinkIndex(const std::vector<const ForObj *> &loops) const {
    if (collapse_)
      return loops.size() - 1;
    for (size_t i = 0; i < loops.size(); ++i)
      if (ConstInt(Extent(loops[i])) != 1)
        return i;
    return 0;
  }
  static bool Sinkable(const GridAccessAnalysis::AccessInfo &info,
                       const ForObj *head, int depth) {
    return info.movable && !info.opaque && !info.outside && info.inside &&
           info.nests.size() == 1 && info.nests.count(head) &&
           info.min_depth >= depth;
  }
  const GridAccessAnalysis &Analysis(const ForObj *head) {
    auto it = analyses_.find(head);
    if (it == analyses_.end()) {
      // Ownership is relative to this candidate nest. If an outer grid was
      // kept serial, its allocation is still shared by an inner candidate.
      auto analysis = make_object<GridAccessAnalysis>(function_, head);
      analysis->Visit(function_->body);
      TVM_FFI_ICHECK(analysis->FoundFocus())
          << "CPU grid candidate must belong to the analyzed source function";
      it = analyses_.emplace(head, std::move(analysis)).first;
    }
    return *it->second;
  }
  IterationPrivacyChecker &Privacy(const ForObj *head) {
    auto it = privacy_.find(head);
    if (it == privacy_.end()) {
      auto checker = make_object<IterationPrivacyChecker>(Analysis(head));
      auto loops = CollectChain(head);
      checker->Visit(loops[SinkIndex(loops)]->body);
      it = privacy_.emplace(head, std::move(checker)).first;
    }
    return *it->second;
  }

  std::vector<Stmt> CollectSinkableAllocs(const ForObj *head) {
    const auto &analysis = Analysis(head);
    auto depth = static_cast<int>(SinkIndex(CollectChain(head))) + 1;
    VarSet roots;
    std::unordered_set<const BindObj *> available;
    for (size_t i = available_begin_; i < available_sequences_.size(); ++i)
      for (const auto &element : *available_sequences_[i])
        if (const auto *binding = element.as<BindObj>())
          available.insert(binding);
    for (size_t i = available_begin_; i < available_sequences_.size(); ++i) {
      for (const auto &element : *available_sequences_[i]) {
        const auto *binding = element.as<BindObj>();
        if (!binding)
          continue;
        const auto *call = binding->value.as<CallObj>();
        if (!call || !call->op.same_as(tirx::alloc_tensor_op()))
          continue;
        auto info = analysis.Lookup(binding->var);
        if (!Sinkable(info, head, depth) ||
            Privacy(head).ReadsPrevious(binding->var))
          continue;
        // Alias definitions may be inside the grid already. Other definitions
        // must dominate it in an available sequence and move with their root.
        bool closed = true;
        for (const auto *definition : info.definitions) {
          if (available.count(definition))
            continue;
          // A definition nested inside the grid remains dominated after
          // sinking.
          auto finder = make_object<DefinitionFinder>(definition);
          finder->Visit(head->body);
          if (!finder->found) {
            closed = false;
            break;
          }
        }
        if (closed)
          roots.insert(binding->var);
      }
    }
    std::vector<Stmt> sunk;
    // Preserve definition order from the outermost available sequence inward.
    // Only these new output arrays are edited; the source function used by
    // the access and reset analyses remains immutable.
    for (size_t i = available_begin_; i < available_sequences_.size(); ++i) {
      Array<Stmt> kept;
      for (const auto &element : *available_sequences_[i]) {
        const auto *binding = element.as<BindObj>();
        auto root =
            binding ? analysis.DefinitionRoot(binding) : Optional<Var>{};
        if (root.has_value() && roots.count(root.value()))
          sunk.push_back(element);
        else
          kept.push_back(element);
      }
      *available_sequences_[i] = std::move(kept);
    }
    return sunk;
  }

  class DefinitionFinder : public StmtExprVisitor {
  public:
    using StmtExprVisitor::Visit;
    using StmtExprVisitor::Visit_;
    explicit DefinitionFinder(const BindObj *target) : target_(target) {}
    bool found = false;
    Optional<VisitInterrupt> Visit_(const BindObj *op) final {
      found |= op == target_;
      return StmtExprVisitor::Visit_(op);
    }

  private:
    const BindObj *target_;
  };

  Stmt ConvertNestElement(const Stmt &element, const ForObj *head,
                          std::vector<Stmt> sunk) {
    if (const auto *sequence = element.as<SeqStmtObj>())
      return SeqStmt(
          ConvertNestElement(sequence->seq[0], head, std::move(sunk)),
          sequence->loc);
    if (const auto *region = element.as<RegionStmtObj>())
      return RegionStmt(
          region->op, region->args, region->body_params, region->attrs,
          SeqStmt(ConvertNestElement(region->body, head, std::move(sunk))),
          region->result_vars, region->loc);
    return ConvertThenVisit(head, std::move(sunk));
  }
  Stmt ConvertThenVisit(const ForObj *head, std::vector<Stmt> sunk) {
    auto [result, parallelized] = ConvertGridNest(head, sunk);
    auto before = in_parallel_;
    in_parallel_ |= parallelized;
    result = Mutate(result).ValueOrUnchanged(result);
    in_parallel_ = before;
    return result;
  }
  Stmt Rebuild(const std::vector<const ForObj *> &loops,
               const std::vector<Stmt> &sunk, bool parallelize, size_t sink) {
    SeqStmt body = loops.back()->body;
    for (size_t i = loops.size(); i-- > 0;) {
      if (parallelize && i == sink && !sunk.empty()) {
        Array<Stmt> inner(sunk.begin(), sunk.end());
        inner.push_back(body);
        body = SeqStmt(std::move(inner));
      }
      const auto *loop = loops[i];
      auto kind = parallelize && (collapse_ || i == sink) ? ForKind::kParallel
                                                          : ForKind::kDefault;
      body =
          SeqStmt(For(loop->var, loop->begin, loop->end, loop->step, kind, body,
                      StripGridAnnotation(loop->annotations), loop->loc));
    }
    if (!parallelize && !sunk.empty()) {
      Array<Stmt> outer(sunk.begin(), sunk.end());
      outer.push_back(body);
      return SeqStmt(std::move(outer));
    }
    return body;
  }
  std::pair<Stmt, bool> ConvertGridNest(const ForObj *head,
                                        const std::vector<Stmt> &sunk) {
    const auto &analysis = Analysis(head);
    auto loops = CollectChain(head);
    size_t sink = SinkIndex(loops);
    int depth = static_cast<int>(sink) + 1;
    auto serial = [&]() -> std::pair<Stmt, bool> {
      return {Rebuild(loops, sunk, false, sink), false};
    };
    if (force_serial_)
      return serial();
    int64_t trip = 1;
    bool dynamic = false;
    for (const auto *loop : loops) {
      if (!IsOne(loop->step))
        return serial();
      auto extent = ConstInt(Extent(loop));
      if (!extent.has_value()) {
        dynamic = true;
        continue;
      }
      if (extent.value() <= 0)
        return serial();
      trip = trip > std::numeric_limits<int64_t>::max() / extent.value()
                 ? std::numeric_limits<int64_t>::max()
                 : trip * extent.value();
    }
    if ((dynamic && min_trip_ > 0) || (!dynamic && trip < min_trip_))
      return serial();
    auto effects = make_object<UnsafeEffectFinder>();
    effects->Visit(loops[sink]->body);
    if (!collapse_ && (!IsZero(loops[sink]->begin) || effects->parallel))
      return serial();
    if (effects->unsafe) {
      LOG(WARNING)
          << "tl.cpu_parallel: grid `" << head->var->name
          << "` contains effects that cannot be proven safe; it stays serial";
      return serial();
    }
    Ranges scopes;
    for (const auto *loop : loops)
      scopes.Set(loop->var, Range(loop->begin, loop->end));
    auto overlap = make_object<OverlapStoreChecker>(analysis, scopes);
    overlap->Visit(loops.back()->body);
    overlap->Finish();
    if (overlap->found) {
      LOG(WARNING) << "tl.cpu_parallel: buffer `" << overlap->buffer_name
                   << "` cannot be proven race-free (" << overlap->reason
                   << "); grid `" << head->var->name << "` stays serial";
      return serial();
    }
    for (const auto &[root, allocation_depth] : analysis.AllocDepths()) {
      if (allocation_depth >= depth)
        continue;
      auto info = analysis.Lookup(root);
      if (!info.nests.count(head))
        continue;
      bool moved = std::any_of(sunk.begin(), sunk.end(), [&](const Stmt &stmt) {
        const auto *binding = stmt.as<BindObj>();
        return binding && binding->var.same_as(root);
      });
      if (Sinkable(info, head, depth) && !Privacy(head).ReadsPrevious(root) &&
          moved)
        continue;
      if (info.store_inside || info.opaque_inside) {
        LOG(WARNING) << "tl.cpu_parallel: allocation `" << root->name
                     << "` carries state or cannot be privatized; grid `"
                     << head->var->name << "` stays serial";
        return serial();
      }
    }
    return {Rebuild(loops, sunk, true, sink), true};
  }

  bool collapse_;
  int64_t min_trip_;
  tirx::Function function_;
  std::unordered_map<const ForObj *, ObjectPtr<GridAccessAnalysis>> analyses_;
  bool force_serial_;
  bool in_parallel_ = false;
  std::vector<Array<Stmt> *> available_sequences_;
  size_t available_begin_ = 0;
  std::unordered_map<const ForObj *, ObjectPtr<IterationPrivacyChecker>>
      privacy_;
};
} // namespace

tvm::transform::Pass MaterializeCPUParallelGrid() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext context) {
    auto target = function->attrs.GetAttr<Target>("target");
    if (!(!function->GetAttr(tvm::attr::kDeclaration).value_or(false)) ||
        !target.has_value())
      return function;
    auto kind = target.value()->kind->name;
    if (kind != "c" && kind != "llvm")
      return function;
    auto minimum = context->GetConfig<int64_t>(kCPUParallelMinTrip).value_or(0);
    auto force_serial =
        function->attrs.GetAttr<bool>(attr::kCPUHadAtomics).value_or(false);
    if (force_serial)
      LOG(WARNING) << "tl.cpu_parallel: atomic kernels use serial "
                      "read-modify-write; grid stays serial";
    auto rewriter =
        make_object<GridRewriter>(kind == "c", minimum, function, force_serial);
    auto body = function->body;
    auto result = rewriter->Mutate(body).ValueOrUnchanged(body);
    if (!result.same_as(body))
      function.CopyOnWrite()->body = result;
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.MaterializeCPUParallelGrid");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.cpu.transform.MaterializeCPUParallelGrid",
      MaterializeCPUParallelGrid);
}
} // namespace tvm::tl
