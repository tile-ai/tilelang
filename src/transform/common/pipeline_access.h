/*!
 * \file transform/common/pipeline_access.h
 * \brief Native storage and access contracts shared by pipeline passes.
 */
#ifndef TILELANG_TRANSFORM_COMMON_PIPELINE_ACCESS_H_
#define TILELANG_TRANSFORM_COMMON_PIPELINE_ACCESS_H_

#include "backend/common/target_utils.h"
#include "ir/alias_analysis.h"
#include "ir/annotations.h"
#include "ir/tensor_utils.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/utils.h"
#include "pipeline_utils.h"
#include <optional>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <unordered_set>
#include <utility>

namespace tvm::tl {

// Allocation identity is collected before storage reuse. decl_tensor is a
// view, never an allocation, and an unknown external call is never declared
// independent. Different views of one root remain conservative even if their
// scopes or element types differ.
class PipelineMemoryAnalysis {
public:
  explicit PipelineMemoryAnalysis(const tirx::Function &function)
      : roots_(function),
        no_alias_(function->GetAttr<bool>("tirx.noalias").value_or(false)) {
    for (const Var &parameter : function->params)
      if (auto root = roots_.Root(parameter))
        parameters_.insert(*root);
    if (auto marked =
            function->GetAttr<ffi::Array<ffi::Any>>("tl.non_restrict_params"))
      AddNonRestrict(marked.value());
    if (auto indices = function->GetAttr<ffi::Array<int64_t>>(
            "tl.may_alias_param_indices"))
      for (int64_t index : indices.value()) {
        TVM_FFI_CHECK(index >= 0 &&
                          static_cast<size_t>(index) < function->params.size(),
                      ValueError);
        AddNonRestrict(ffi::Array<ffi::Any>{function->params[index]});
      }
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        function->body,
        [&](const ffi::ObjectRef &obj) -> ffi::Expected<ffi::WalkResult> {
          if (const auto *bind = obj.as<BindObj>()) {
            definitions_.Set(bind->var, bind->value);
            if (const auto *call = bind->value.as<CallObj>();
                call && call->op.same_as(tirx::alloc_tensor_op()))
              allocations_.insert(bind->var);
          }
          if (const auto *call = obj.as<CallObj>();
              call && IsAnnotationCall(call)) {
            auto values = ReadAnnotations(call);
            if (auto marked = values.Get("tl.non_restrict_params"))
              AddNonRestrict(*marked);
          }
          if (const auto *region = obj.as<RegionStmtObj>();
              region && region->op->name == "tl.attr") {
            auto fields = ReadAnnotations(region);
            auto encoded_key = fields.Get("key");
            if (!encoded_key || !encoded_key->as<ffi::String>())
              return ffi::WalkResult::Advance();
            auto key = encoded_key->cast<ffi::String>();
            if (key == "tl.non_restrict_params")
              AddNonRestrict(fields.at("value"));
            else if (key == "tl.kernel") {
              auto values =
                  fields.at("value").cast<ffi::Map<ffi::String, ffi::Any>>();
              if (auto marked = values.Get("tl.non_restrict_params"))
                AddNonRestrict(*marked);
            }
          }
          return ffi::WalkResult::Advance();
        });
  }

  ffi::Optional<tirx::TensorVar> Tensor(const Expr &expression) const {
    std::unordered_set<const VarObj *> active;
    return Tensor(expression, &active);
  }

  ffi::Optional<Var> Root(const Expr &expression) const {
    std::unordered_set<const VarObj *> active;
    return Root(expression, &active);
  }

  // Rewritten stage-local aliases have fresh identities. Keep their explicit
  // value definitions alongside the immutable input facts so access regions
  // continue to refer to their backing allocation after versioning.
  void ObserveDefinitions(const Stmt &statement) const {
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        statement,
        [&](const ffi::ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
          if (const auto *bind = object.as<BindObj>();
              bind && !definitions_.count(bind->var))
            generated_definitions_.Set(bind->var, bind->value);
          return ffi::WalkResult::Advance();
        });
  }

  void RecordGeneratedDefinition(const Var &var, const Expr &value) const {
    TVM_FFI_CHECK(!definitions_.count(var), ValueError)
        << "A pipeline-generated definition must have a fresh identity";
    generated_definitions_.Set(var, value);
  }

  void RecordGeneratedAlias(const Var &var, const Expr &value,
                            const tirx::TensorVar &root) const {
    RecordGeneratedDefinition(var, value);
    generated_alias_roots_.Set(var, root);
  }

  TensorRegion CanonicalRegion(const TensorRegion &region) const {
    auto root = Root(region->source);
    if (root && !root.value().same_as(region->source)) {
      if (auto tensor = root.value().as<tirx::TensorVar>())
        return FullTensorRegion(*tensor);
    }
    return region;
  }

  bool MayAlias(const tirx::TensorVar &lhs, const tirx::TensorVar &rhs) const {
    if (lhs.same_as(rhs))
      return true;
    auto left = Root(lhs), right = Root(rhs);
    if (!left || !right)
      return true;
    if (left.value().same_as(right.value()))
      return true;
    // alloc_tensor denotes new storage. A function argument cannot point at
    // this invocation's future allocation. This guarantee does not apply to
    // decl_tensor, capture or opaque allocation callbacks.
    if (allocations_.count(*left) || allocations_.count(*right))
      return false;
    if (no_alias_ && !unknown_non_restrict_ && parameters_.count(*left) &&
        parameters_.count(*right) && !non_restrict_.count(*left) &&
        !non_restrict_.count(*right))
      return false;
    auto space = [](const tirx::TensorVar &tensor) {
      std::string scope = tensor.scope().name();
      if (scope.starts_with("shared"))
        return std::string("shared");
      if (scope.starts_with("local"))
        return std::string("local");
      return scope.empty() ? std::string("global") : scope;
    };
    const auto a = space(lhs), b = space(rhs);
    if ((a == "global" || a == "shared" || a == "local") &&
        (b == "global" || b == "shared" || b == "local") && a != b)
      return false;
    return true;
  }

  bool MayConflict(const TensorRegion &lhs, const TensorRegion &rhs,
                   const ffi::Map<Var, sym::IntSet> &threads) const {
    if (!MayAlias(RegionTensor(lhs), RegionTensor(rhs)))
      return false;
    // Distinct views may have different strides, offsets and element widths.
    // The common helper compares physical offsets only for the same tensor;
    // retaining a dependency is preferable to comparing unlike coordinates.
    return PipelineRegionsMayConflict(lhs, rhs, threads);
  }

  const ffi::Map<Var, Expr> &definitions() const { return definitions_; }

private:
  ffi::Optional<Var> Root(const Expr &expression,
                          std::unordered_set<const VarObj *> *active) const {
    if (auto var = expression.as<Var>()) {
      if (!var->get()->ty.as<PtrTypeObj>() &&
          !var->get()->ty.as<tirx::TensorTypeObj>())
        return std::nullopt;
      if (!active->insert(var->get()).second)
        return std::nullopt;
      if (auto alias = generated_alias_roots_.Get(*var)) {
        auto root = Root(*alias, active);
        active->erase(var->get());
        return root;
      }
      auto definition = generated_definitions_.Get(*var);
      if (!definition)
        definition = definitions_.Get(*var);
      ffi::Optional<Var> root = var;
      if (definition) {
        const auto *call = definition->as<CallObj>();
        if (!call || !call->op.same_as(tirx::alloc_tensor_op()))
          root = Root(*definition, active);
      }
      active->erase(var->get());
      return root;
    }
    if (const auto *load = expression.as<TensorLoadObj>())
      return Root(load->source, active);
    if (const auto *region = expression.as<TensorRegionObj>())
      return Root(region->source, active);
    if (const auto *call = expression.as<CallObj>()) {
      const auto *op = call->op.as<OpObj>();
      if (op && !call->args.empty() &&
          (op == tirx::decl_tensor_op().get() ||
           op == tirx::tensor_data_ptr_op().get() ||
           op == tirx::address_of_op().get() ||
           op == tirx::reinterpret_op().get() ||
           op == tirx::ptr_byte_offset_op().get() ||
           op == tirx::handle_add_byte_offset_op().get() ||
           op == tl::access_ptr().get() || op == tl::ptr_access().get()))
        return Root(call->args[0], active);
    }
    return std::nullopt;
  }

  void AddNonRestrict(const ffi::Any &values) {
    for (const ffi::Any &value : values.cast<ffi::Array<ffi::Any>>()) {
      if (auto root = Root(value.cast<Expr>()))
        non_restrict_.insert(*root);
      else
        unknown_non_restrict_ = true;
    }
  }

  ffi::Optional<tirx::TensorVar>
  Tensor(const Expr &expression,
         std::unordered_set<const VarObj *> *active) const {
    if (auto tensor = expression.as<tirx::TensorVar>())
      return tensor;
    if (auto var = expression.as<Var>()) {
      if (!active->insert(var->get()).second)
        return std::nullopt;
      auto definition = generated_definitions_.Get(*var);
      if (!definition)
        definition = definitions_.Get(*var);
      auto result = definition ? Tensor(*definition, active) : std::nullopt;
      active->erase(var->get());
      return result;
    }
    if (const auto *load = expression.as<TensorLoadObj>())
      return Tensor(load->source, active);
    if (const auto *region = expression.as<TensorRegionObj>())
      return Tensor(region->source, active);
    if (const auto *call = expression.as<CallObj>()) {
      const auto *op = call->op.as<OpObj>();
      if (op && !call->args.empty() &&
          (op == tirx::decl_tensor_op().get() ||
           op == tirx::tensor_data_ptr_op().get() ||
           op == tirx::address_of_op().get() ||
           op == tirx::reinterpret_op().get() ||
           op == tirx::ptr_byte_offset_op().get() ||
           op == tirx::handle_add_byte_offset_op().get() ||
           op == tl::access_ptr().get() || op == tl::ptr_access().get()))
        return Tensor(call->args[0], active);
    }
    return std::nullopt;
  }

  PhysicalRootAnalysis roots_;
  ffi::Map<Var, Expr> definitions_;
  mutable ffi::Map<Var, Expr> generated_definitions_;
  mutable ffi::Map<Var, tirx::TensorVar> generated_alias_roots_;
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> allocations_;
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> parameters_,
      non_restrict_;
  bool no_alias_{false}, unknown_non_restrict_{false};
};

// Region operands of a tile operation are addresses. Their bounds can still
// contain real loads, so traverse address expressions while recording effects
// from the operation's declared read/write regions, not from region spelling.
class PipelineAccessCollector : public tirx::StmtExprVisitor {
public:
  using tirx::StmtExprVisitor::Visit;
  using tirx::StmtExprVisitor::Visit_;
  using Result = ffi::Optional<VisitInterrupt>;

  PipelineAccessCollector(const PipelineMemoryAnalysis *memory, Target target)
      : memory_(memory), target_(std::move(target)) {}

  const ffi::Array<TensorRegion> &GetReads() const { return reads_; }
  const ffi::Array<TensorRegion> &GetWrites() const { return writes_; }
  bool GetGlobalCopyPattern() const { return global_copy_; }
  bool GetTmaCopyPattern() const { return tma_copy_; }
  bool HasNonCopyTileOp() const { return non_copy_; }
  bool HasUnknownAccess() const { return unknown_access_; }
  // An imprecise index still has a tensor identity and can be rewritten by
  // explicit software pipelining. Opaque effects/addresses cannot.
  bool HasUnknownAddress() const { return unknown_address_; }

private:
  static bool IsGlobal(const tirx::TensorVar &tensor) {
    return IsGlobalBuffer(tensor) || tensor.scope().name().empty();
  }

  ffi::Optional<Range> AnalysisRange(const PrimExpr &index) const {
    auto integer_scalar = [](const PrimType &type) {
      return type.IsScalar() && type.MatchesCode(kDLInt, kDLUInt);
    };
    sym::Analyzer analyzer;
    using Bounds = std::pair<ffi::BigInt, ffi::BigInt>;
    auto bounds = [&](const PrimExpr &value) -> std::optional<Bounds> {
      if (const auto *literal = value.as<IntImmObj>())
        return Bounds{literal->value, literal->value};
      auto bound = analyzer->const_int_bound(value);
      if (bound->min_value <= sym::ConstIntBound::kNegInf ||
          bound->max_value >= sym::ConstIntBound::kPosInf)
        return std::nullopt;
      return Bounds{ffi::BigInt(bound->min_value),
                    ffi::BigInt(bound->max_value)};
    };

    auto type_bounds = [](const PrimType &type) -> Bounds {
      if (type.MatchesCode(kDLUInt))
        return {ffi::BigInt(0), (ffi::BigInt(1) << type.bits()) - 1};
      const ffi::BigInt limit = ffi::BigInt(1) << (type.bits() - 1);
      return {-limit, limit - 1};
    };
    using LaneBounds = std::pair<PrimExpr, PrimExpr>;
    auto lane_bounds = [&](auto &&self,
                           const PrimExpr &value) -> std::optional<LaneBounds> {
      if (integer_scalar(value.ty()))
        return LaneBounds{value, value};
      if (value.ty().IsScalableVector() ||
          !value.ty().MatchesCode(kDLInt, kDLUInt))
        return std::nullopt;
      if (const auto *broadcast = value.as<prim::BroadcastObj>())
        return self(self, broadcast->value);
      if (const auto *cast = value.as<prim::CastObj>()) {
        auto source = self(self, cast->value);
        if (!source)
          return std::nullopt;
        auto low = bounds(source->first), high = bounds(source->second);
        PrimType scalar_type = value.ty().WithLanes(1);
        auto domain = type_bounds(scalar_type);
        // A narrowing or sign-changing cast is monotone only over values
        // representable by the result type. Casting just the two endpoints
        // can otherwise omit every lane between a pair of wrap boundaries.
        if (!low || !high || low->first < domain.first ||
            high->second > domain.second)
          return std::nullopt;
        return LaneBounds{prim::cast(source->first, scalar_type),
                          prim::cast(source->second, scalar_type)};
      }
      const auto *ramp = value.as<prim::RampObj>();
      if (!ramp || !integer_scalar(ramp->base.ty()))
        return std::nullopt;
      const auto *stride = ramp->stride.as<IntImmObj>();
      const auto *lanes = ramp->lanes.as<IntImmObj>();
      if (!stride || !lanes || lanes->value <= 0)
        return std::nullopt;
      auto base = bounds(ramp->base);
      auto domain = type_bounds(ramp->base.ty());
      const ffi::BigInt span = stride->value * (lanes->value - 1);
      if (!base || span < domain.first || span > domain.second)
        return std::nullopt;
      const ffi::BigInt first =
          base->first + (span < 0 ? span : ffi::BigInt(0));
      const ffi::BigInt last =
          base->second + (span > 0 ? span : ffi::BigInt(0));
      if (first < domain.first || last > domain.second)
        return std::nullopt;
      PrimExpr endpoint = ramp->base + IntImm(ramp->base.ty(), span);
      return span < 0 ? LaneBounds{endpoint, ramp->base}
                      : LaneBounds{ramp->base, endpoint};
    };
    auto lanes = lane_bounds(lane_bounds, index);
    if (!lanes)
      return std::nullopt;
    PrimExpr lower = lanes->first, upper = lanes->second;

    // The inclusive lane bounds use the original integer semantics, after
    // checking that Ramp/cast cannot wrap. Only the analysis interval is
    // widened: int8 {0,127} needs extent 128, and uint8 {255} needs end 256.
    auto low = bounds(lower), high = bounds(upper);
    const ffi::BigInt limit = ffi::BigInt(1) << 63;
    // Prove representability before constructing either addition or
    // subtraction. In particular no signed/unsigned 64-bit cast or overflowing
    // endpoint may be used as evidence that two accesses are disjoint.
    if (!low || !high || low->first < -limit || low->second >= limit ||
        high->first < -limit || high->second >= limit - 1 ||
        high->second - low->first + 1 >= limit)
      return std::nullopt;
    lower = prim::cast(lower, PrimType::Int(64));
    upper = prim::cast(upper, PrimType::Int(64));
    return Range::FromMinExtent(
        analyzer->Simplify(lower),
        analyzer->Simplify(upper + IntImm::Int64(1) - lower));
  }

  TensorRegion AccessRegion(const Expr &tensor,
                            const ffi::Array<PrimExpr> &indices, Location loc) {
    ffi::Array<Range> ranges;
    auto type = tensor->ty.as_or_throw<tirx::TensorType>();
    for (size_t axis = 0; axis < indices.size(); ++axis) {
      // A vector access covers all lanes, including descending ramps and
      // broadcasts. Range endpoints are scalar expressions, never the vector
      // itself with a scalar one forced into its type.
      if (auto range = AnalysisRange(indices[axis])) {
        ranges.push_back(*range);
        continue;
      }
      // Wrapping, opaque or unrepresentable indices cannot prove disjointness.
      // Retain the tensor identity and conservatively cover its full axis.
      unknown_access_ = true;
      const PrimExpr &extent = type->shape[axis];
      ranges.push_back(Range::FromMinExtent(IntImm(extent.ty(), 0), extent));
    }
    return TensorRegion(tensor, ranges, TensorRegionType(), loc);
  }

  Result Visit_(const BindObj *op) final {
    const bool previous = address_only_;
    address_only_ =
        op->var->ty.as<PtrTypeObj>() || op->var->ty.as<tirx::TensorTypeObj>();
    Visit(op->value);
    Visit(op->var->ty);
    address_only_ = previous;
    return std::nullopt;
  }

  Result Visit_(const TensorStoreObj *op) final {
    writes_.push_back(AccessRegion(op->dest, op->indices, op->loc));
    const bool saved_read = global_read_;
    global_read_ = false;
    Visit(op->value);
    if (global_read_ && IsSharedBuffer(op->dest.as_or_throw<tirx::TensorVar>()))
      global_copy_ = true;
    for (const PrimExpr &index : op->indices)
      Visit(index);
    global_read_ = saved_read;
    return std::nullopt;
  }

  Result Visit_(const TensorLoadObj *op) final {
    reads_.push_back(AccessRegion(op->source, op->indices, op->loc));
    if (!condition_ && IsGlobal(op->source.as_or_throw<tirx::TensorVar>()))
      global_read_ = true;
    const bool saved_read = global_read_;
    for (const PrimExpr &index : op->indices)
      Visit(index);
    global_read_ = saved_read;
    return std::nullopt;
  }

  void AccessPointer(const Expr &base, int mask,
                     bool performed_access = false) {
    if (!address_only_ || performed_access) {
      if (auto tensor = memory_->Tensor(base)) {
        auto region = FullTensorRegion(*tensor);
        if (mask & kAccessRead)
          reads_.push_back(region);
        if (mask & kAccessWrite)
          writes_.push_back(region);
      } else {
        unknown_access_ = unknown_address_ = true;
      }
    }
  }

  // An address expression is not itself a read of the addressed element.
  // Computing that address can still execute loads, including byte offsets
  // and tensor shape/layout expressions. Keep those dependencies explicit.
  void VisitAddress(const Expr &base) {
    const bool previous = address_only_;
    address_only_ = true;
    if (const auto *load = base.as<TensorLoadObj>()) {
      Visit(load->source->ty);
      for (const PrimExpr &index : load->indices)
        Visit(index);
    } else {
      Visit(base);
    }
    address_only_ = previous;
  }

  Result Visit_(const CallObj *op) final {
    const auto *named_op = op->op.as<OpObj>();
    if (named_op &&
        (IsAnnotationCall(op) || named_op == tirx::alloc_tensor_op().get()))
      return std::nullopt;
    if (auto tile = ParseOperator(ffi::GetRef<Call>(op)); tile.defined()) {
      AccessRegions access = tile->GetAccessRegions();
      for (const auto &region : access.reads)
        reads_.push_back(region);
      for (const auto &region : access.writes)
        writes_.push_back(region);
      if (const auto *copy = tile.as<CopyNode>()) {
        global_copy_ |= IsGlobal(copy->src) && IsSharedBuffer(copy->dst);
      } else if (const auto *im2col = tile.as<Im2ColOpNode>()) {
        const bool copy =
            IsGlobal(im2col->src_) && IsSharedBuffer(im2col->dst_);
        global_copy_ |= copy;
        tma_copy_ |= copy && TargetIsHopper(target_);
      } else {
        non_copy_ = true;
      }
      return tirx::StmtExprVisitor::Visit_(op);
    }
    if (named_op && named_op == tl::access_ptr().get()) {
      AccessPointer(op->args[0], GetConservativeAccessMask(
                                     op->args[2].as_or_throw<PrimExpr>()));
      VisitAddress(op->args[0]);
      Visit(op->args[1]);
      Visit(op->args[2]);
      return std::nullopt;
    }
    if (named_op && named_op == tl::ptr_access().get()) {
      AccessPointer(op->args[0], GetConservativeAccessMask(
                                     op->args[3].as_or_throw<PrimExpr>()));
      VisitAddress(op->args[0]);
      for (size_t i = 1; i < op->args.size(); ++i)
        Visit(op->args[i]);
      return std::nullopt;
    }
    if (named_op && named_op == tirx::address_of_op().get()) {
      AccessPointer(op->args[0], kAccessReadWrite);
      VisitAddress(op->args[0]);
      return std::nullopt;
    }
    if (named_op && named_op == prim::if_then_else_op().get()) {
      const bool previous = condition_;
      condition_ = true;
      Visit(op->args[0]);
      condition_ = previous;
      for (size_t i = 1; i < op->args.size(); ++i)
        Visit(op->args[i]);
      return std::nullopt;
    }
    const auto effects = Op::GetAttrMap(tvm::op_attr::kCallEffectKind);
    auto effect = named_op && effects.count(ffi::GetRef<Op>(named_op))
                      ? effects[ffi::GetRef<Op>(named_op)]
                      : CallEffectKind::kOpaque;
    if (effect >= CallEffectKind::kReadState) {
      bool described_access = false;
      for (const Expr &arg : op->args) {
        if (!arg->ty.as<PtrTypeObj>() && !arg->ty.as<tirx::TensorTypeObj>())
          continue;
        described_access = true;
        // An explicit access descriptor owns its mask. Looking through it
        // here would turn a write-only stmatrix into a fictitious read.
        const auto *descriptor = arg.as<CallObj>();
        const auto *name = descriptor ? descriptor->op.as<OpObj>() : nullptr;
        if (name &&
            (name == tl::access_ptr().get() || name == tl::ptr_access().get()))
          continue;
        AccessPointer(arg,
                      effect == CallEffectKind::kReadState ? kAccessRead
                                                           : kAccessReadWrite,
                      true);
      }
      // Opaque external calls can access closure/global state even without
      // explicit pointers. Only known pipeline synchronization controls have
      // a state contract independent of their argument tensor regions.
      std::string name = named_op ? std::string(named_op->name) : "";
      static const std::unordered_set<std::string> synchronization_ops = {
          "tl.ptx_commit_group",
          "tl.ptx_wait_group",
          "tl.ptx_arrive_barrier",
          "tl.ptx_cp_async_barrier_noinc",
          "tl.ptx_fence_barrier_init",
          "tl.mbarrier_init",
          "tl.mbarrier_arrive",
          "tl.mbarrier_expect_tx",
          "tl.mbarrier_arrive_expect_tx",
          "tl.mbarrier_wait_parity",
          "tl.fence_proxy_async",
          "tl.tcgen05_before_thread_sync",
          "tl.tcgen05_after_thread_sync",
          "tl.tcgen05_mma_arrive",
          "tl.warpgroup_arrive",
          "tl.warpgroup_commit_batch",
          "tl.warpgroup_wait",
          "tl.wait_wgmma",
          "tl.tma_store_arrive",
          "tl.tma_store_wait",
          "tirx.gpu_storage_sync",
          "tl.storage_sync"};
      const bool control = synchronization_ops.count(name);
      // Only these registered hardware memory primitives confine their
      // storage effects to explicit operands. kOpaque on an arbitrary op,
      // including call_extern with pointer arguments, makes no such promise.
      static const std::unordered_set<std::string> argument_memory_ops = {
          "tl.ptx_cp_async",
          "tl.ptx_ldmatrix",
          "ptx.ldmatrix",
          "ptx.stmatrix",
          "tl.ptx_stmatrix",
          "tl.ptx_mma_sm70",
          "tl.ptx_st_bulk_shared",
          "tl.ptx_cluster_store",
          "tl.atomic_add",
          "tl.atomic_addx2",
          "tl.atomic_addx4",
          "tl.atomic_max",
          "tl.atomic_min",
          "tl.atomic_or",
          "tl.atomic_load",
          "tl.atomic_store",
          "tirx.atomic_add",
          "tirx.memcpy",
          "tl.atomic_add_elem_op",
          "tl.atomic_add_ret_elem_op",
          "tl.atomic_addx2_elem_op",
          "tl.atomic_addx2_ret_elem_op",
          "tl.atomic_addx4_elem_op",
          "tl.atomic_addx4_ret_elem_op",
          "tl.atomic_max_elem_op",
          "tl.atomic_max_ret_elem_op",
          "tl.atomic_min_elem_op",
          "tl.atomic_min_ret_elem_op",
          "tl.atomic_or_elem_op",
          "tl.atomic_load_elem_op",
          "tl.atomic_store_elem_op"};
      if ((!control && !argument_memory_ops.count(name)) ||
          (!described_access && !control))
        unknown_access_ = unknown_address_ = true;
    }
    return tirx::StmtExprVisitor::Visit_(op);
  }

  Result Visit_(const IfObj *op) final {
    const bool previous = condition_;
    condition_ = true;
    Visit(op->condition);
    condition_ = previous;
    Visit(op->then_case);
    if (op->else_case)
      Visit(*op->else_case);
    return std::nullopt;
  }

  const PipelineMemoryAnalysis *memory_;
  Target target_;
  ffi::Array<TensorRegion> reads_, writes_;
  bool global_read_{false}, global_copy_{false}, tma_copy_{false};
  bool non_copy_{false}, condition_{false}, unknown_access_{false};
  bool unknown_address_{false};
  bool address_only_{false};
};

} // namespace tvm::tl
#endif // TILELANG_TRANSFORM_COMMON_PIPELINE_ACCESS_H_
