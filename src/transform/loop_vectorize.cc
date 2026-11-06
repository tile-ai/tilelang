/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tl/transform/loop_vectorize.cc
 * \brief Native vector-width planning for TileLang memory and tile operations.
 */
#include "loop_vectorize.h"
#include "../config.h"
#include "../ir/analyzer.h"
#include "../op/builtin.h"
#include "../op/utils.h"
#include "backend/common/target_utils.h"
#include "common/loop_vectorization_utils.h"
#include "common/pipeline_access.h"
#include "ir/analyzer.h"
#include "ir/loop_math.h"
#include "ir/stmt_functor.h"
#include "layout/expr_utils.h"
#include "op/builtin.h"
#include "op/reducer.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <optional>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <vector>

namespace tvm::tl {
using namespace tirx;
using namespace prim;
using namespace ffi;

namespace {
class VectorizeAccessCollector : public PipelineAccessCollector {
public:
  using PipelineAccessCollector::PipelineAccessCollector;
  using PipelineAccessCollector::Visit;
  Optional<VisitInterrupt> Visit(ffi::AnyView value) final {
    if (auto call = value.as<Call>();
        call && call.value()->op.same_as(prim::assume_op())) {
      // The native assume is kEmbedInfo, not a memory-writing instruction.
      // Its condition still contributes every real read or opaque effect.
      return Visit(call.value()->args[0]);
    }
    return PipelineAccessCollector::Visit(value);
  }
};
} // namespace

struct VectorizeMemoryScope::Impl {
  explicit Impl(const tirx::Function &function) : memory(function) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return;
    Target target =
        function->GetAttr<Target>(tvm::attr::kTarget).value_or(Target("llvm"));
    auto access = make_object<VectorizeAccessCollector>(&memory, target);
    access->Visit(function->body);
    writes = access->GetWrites();
    unknown = access->HasUnknownAddress();
    // Volatile aliases are observations even in the absence of a write.
    // Like the owning simplifier, conservatively disable memory reuse for
    // the whole function when any supported volatile annotation is present.
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        function->body,
        [&](const ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
          if (const auto *bind = object.as<BindObj>();
              bind && bind->var.as<PrimVar>()) {
            if (auto value = bind->value.as<PrimExpr>()) {
              PrimExpr expanded = tirx::SubstituteWithDataTypeLegalization(
                  *value, [&](const Var &var) -> ffi::Optional<PrimExpr> {
                    return read_bindings.Get(var);
                  });
              if (SideEffect(expanded) > CallEffectKind::kPure)
                read_bindings.Set(bind->var, expanded);
            }
          }
          if (const auto *region = object.as<RegionStmtObj>();
              region && region->op->name == "tl.attr") {
            auto fields = ReadAnnotations(region);
            if (auto key = fields.Get("key"); key && key->as<String>())
              unknown |= key->cast<String>() == "volatile_scope";
          }
          if (const auto *call = object.as<CallObj>(); call && call->attrs) {
            if (auto attrs = call->attrs.value().as<DictAttrs>())
              unknown |= (*attrs)->dict.count("tirx.volatile") ||
                         (*attrs)->dict.count("volatile_scope");
          }
          return ffi::WalkResult::Advance();
        });
  }

  bool Stable(const PrimExpr &expression) const {
    if (unknown)
      return false;
    bool safe = true;
    auto effects = Op::GetAttrMap(tvm::op_attr::kCallEffectKind);
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        expression,
        [&](const ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
          if (const auto *load = object.as<TensorLoadObj>()) {
            const auto tensor = load->source.as<TensorVar>();
            if (!tensor || !memory.Root(*tensor)) {
              safe = false;
              return ffi::WalkResult::Advance();
            }
            for (const auto &write : writes)
              if (memory.MayAlias(*tensor, RegionTensor(write))) {
                safe = false;
                return ffi::WalkResult::Advance();
              }
          }
          if (const auto *call = object.as<CallObj>()) {
            const auto op = call->op.as<Op>();
            // Read-state calls are not plain tensor reads: clocks, volatile
            // intrinsics and external callbacks have no reuse contract here.
            safe &= op && effects.count(*op) &&
                    effects[*op] <= CallEffectKind::kPure;
          }
          return ffi::WalkResult::Advance();
        });
    return safe;
  }

  PipelineMemoryAnalysis memory;
  // Only proof inputs are expanded. The emitted program retains the actual
  // Bind and the vectorizer's fresh use mapping at its lexical position.
  Map<Var, PrimExpr> read_bindings;
  Array<TensorRegion> writes;
  bool unknown{false};
};

thread_local const VectorizeMemoryScope::Impl *VectorizeMemoryScope::current_ =
    nullptr;

VectorizeMemoryScope::VectorizeMemoryScope(const tirx::Function &function)
    : impl_(std::make_unique<Impl>(function)), previous_(current_) {
  current_ = impl_.get();
}

VectorizeMemoryScope::~VectorizeMemoryScope() { current_ = previous_; }

bool IsStableVectorizeIndex(const PrimExpr &expression) {
  const auto *context = VectorizeMemoryScope::current_;
  // A scalar Var is not evidence that its defining load was immutable.
  // Follow read-state Bind chains even when this final pass is called without
  // the earlier width planner. Otherwise a volatile/local aliased row load
  // becomes a fictitious pure address just by naming its result.
  PrimExpr expanded = context
                          ? tirx::SubstituteWithDataTypeLegalization(
                                expression,
                                [&](const Var &var) -> ffi::Optional<PrimExpr> {
                                  return context->read_bindings.Get(var);
                                })
                          : expression;
  auto effect = SideEffect(expanded);
  if (effect <= CallEffectKind::kPure)
    return true;
  return effect <= CallEffectKind::kReadState && context &&
         context->Stable(expanded);
}

Optional<TensorLoad> ExtractBufferLoadForAtomic(const Expr &expr) {
  if (auto load = expr.as<TensorLoad>())
    return load;
  if (const auto *call = expr.as<CallObj>()) {
    if ((call->op.same_as(tl::access_ptr()) ||
         call->op.same_as(tirx::address_of_op())) &&
        !call->args.empty())
      return call->args[0].as<TensorLoad>();
  }
  return std::nullopt;
}

int GetMaxAtomicVectorSize(PrimType dtype, const String &scope,
                           const Target &target) {
  if (!dtype.IsScalar())
    return 1;
  // TileLang's existing half2/bfloat162 atomic templates also cover shared
  // memory and targets preceding Hopper. Only float2/float4 require SM90.
  if (dtype == PrimType::Float(16) || dtype == PrimType::BFloat(16))
    return 2;
  // CUDA's packed float atomics require SM90+ and global float32 storage.
  if (TargetIsCuda(target) && TargetHasSMVersionGE(target, 90) &&
      (scope.empty() || scope == "global") && dtype == PrimType::Float(32))
    return 4;
  return 1;
}

int MaxVectorLoadBits(const Target &target, bool global_only) {
  if (auto bits = target->GetAttr<int64_t>("max_vector_bits")) {
    ICHECK_GT(*bits, 0);
    return static_cast<int>(*bits);
  }
  return TargetSupportVectorize256(target) &&
                 !tl_config::Vectorize256Disabled() && global_only
             ? 256
             : 128;
}

bool CanProveIndependent(const PrimExpr &expr, Var var,
                         sym::AnalyzerObj *analyzer) {
  if (!ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
           expr,
           [&](const VarObj *node) -> ffi::Expected<ffi::WalkResult> {
             return (node == var.get())
                        ? ffi::WalkResult::Interrupt(
                              ffi::VisitInterrupt(ffi::GetRef<Var>(node)))
                        : ffi::WalkResult::Skip();
           })
           .has_value())
    return true;
  PrimVar fresh(var->name + "_independent", var->ty.as_or_throw<PrimType>());
  PrimExpr other = tirx::SubstituteWithDataTypeLegalization(
      expr,
      [values = ffi::Map<Var, PrimExpr>{{var, fresh}}](const Var &var)
          -> ffi::Optional<PrimExpr> { return values.Get(var); });
  return analyzer->CanProveEqual(expr, other);
}

bool IsExprInvariantInVectorBoundary(const PrimExpr &expr, const Var &var,
                                     const PrimExpr &extent, int width,
                                     sym::AnalyzerObj *analyzer,
                                     const PrimExpr &begin) {
  ICHECK_GT(width, 0);
  auto local = analyzer->Clone();
  PrimType index_type = var->ty.as_or_throw<PrimType>();
  PrimVar ordinal("ordinal", index_type);
  BindAnalyzer(
      local, ordinal,
      Range::FromMinExtent(IntImm(index_type, 0), cast(extent, index_type)));
  // Expand dominating scalar Bind values before changing the loop variable.
  // Otherwise an alias bound to the original induction variable retains that
  // variable while its direct uses are replaced by ordinal/lane expressions.
  PrimExpr relative = tirx::SubstituteWithDataTypeLegalization(
      local->Simplify(expr),
      [values =
           ffi::Map<Var, PrimExpr>{{var, cast(begin, index_type) + ordinal}}](
          const Var &var) -> ffi::Optional<PrimExpr> {
        return values.Get(var);
      });
  PrimExpr aligned =
      floordiv(ordinal, IntImm(index_type, width)) * IntImm(index_type, width);
  return local->CanProveEqual(
      relative, tirx::SubstituteWithDataTypeLegalization(
                    relative,
                    [values = ffi::Map<Var, PrimExpr>{{ordinal, aligned}}](
                        const Var &var) -> ffi::Optional<PrimExpr> {
                      return values.Get(var);
                    }));
}

bool IndicesCanVectorize(const PrimExpr &expr, const Var &var,
                         const PrimExpr &extent, int width,
                         sym::AnalyzerObj *analyzer, bool broadcast,
                         const PrimExpr &begin) {
  ICHECK_GT(width, 0);
  if (width == 1)
    return true;
  auto local = analyzer->Clone();
  if (!local->CanProveEqual(floormod(extent, IntImm(extent.ty(), width)), 0))
    return false;
  if (broadcast && IsExprInvariantInVectorBoundary(expr, var, extent, width,
                                                   local.get(), begin))
    return true;
  PrimType index_type = var->ty.as_or_throw<PrimType>();
  PrimVar lane("lane", index_type), group("group", index_type);
  PrimExpr group_width = IntImm(index_type, width);
  BindAnalyzer(local, lane, Range(IntImm(index_type, 0), group_width));
  BindAnalyzer(
      local, group,
      Range(IntImm(index_type, 0),
            cast(floordiv(extent, IntImm(extent.ty(), width)), index_type)));
  PrimExpr shifted = tirx::SubstituteWithDataTypeLegalization(
      local->Simplify(expr),
      [values = ffi::Map<Var, PrimExpr>{{var, cast(begin, index_type) +
                                                  group * group_width + lane}}](
          const Var &var) -> ffi::Optional<PrimExpr> {
        return values.Get(var);
      });
  PrimExpr vector = local->Simplify(
      VectorizeNativeExpr(local->Simplify(shifted), lane, group_width));
  if (const auto *ramp = vector.as<RampObj>()) {
    // Broadcast eligibility does not relax the alignment of a contiguous
    // memory vector. Matrix loads and ordinary vector loads still require
    // every Ramp group to start on its vector boundary.
    return IsOne(ramp->stride) &&
           local->CanProveEqual(
               floormod(ramp->base, IntImm(ramp->base.ty(), width)), 0);
  }
  return broadcast && vector.ty().IsScalar();
}

namespace {
// A query uses either the pending TileLang placement or the committed native
// tensor layout. It never fabricates row-major strides for a strided view.
Optional<PrimExpr> AddressOffset(const TensorVar &tensor,
                                 const Array<PrimExpr> &indices,
                                 const LayoutMap &layouts) {
  if (auto layout = layouts.Get(tensor)) {
    auto output = (*layout)->Forward(indices);
    auto shape = (*layout)->OutputShape();
    if (shape.size() != output.size())
      return std::nullopt;
    PrimType index_type = tensor.ty()->DefaultIndexType();
    PrimExpr offset = IntImm(index_type, 0);
    for (size_t i = 0; i < shape.size(); ++i)
      offset = offset * shape[i] + output[i];
    Array<PrimExpr> zeros;
    for (const auto &extent : tensor.shape())
      zeros.push_back(IntImm(extent.ty(), 0));
    auto bases = tensor.ty()->ElemOffset(zeros);
    if (bases.size() != 1)
      return std::nullopt;
    return offset + bases[0];
  }
  auto offsets = tensor.ty()->ElemOffset(indices);
  return offsets.size() == 1 ? Optional<PrimExpr>(offsets[0]) : std::nullopt;
}

struct AtomicAddress {
  TensorVar tensor;
  PrimExpr bit_offset;
};

// A raw pointer has no implicit alignment or shape. Retain provenance only
// through explicit tensor projections and pointer operations; never fabricate
// a TensorType for an unrelated PtrType variable or external pointer result.
std::optional<AtomicAddress> ResolveAtomicAddress(const Expr &expression,
                                                  const LayoutMap &layouts,
                                                  sym::AnalyzerObj *analyzer,
                                                  const Var &lane) {
  if (const auto *load = expression.as<TensorLoadObj>()) {
    auto tensor = load->source.as<TensorVar>();
    if (!tensor || !tensor->dtype().IsScalar() || tensor->dtype().bits() % 8)
      return std::nullopt;
    // Only address calculations, not the pointee load, are evaluated here.
    // Inspect them before layout application or simplification can erase an
    // opaque call whose value is algebraically invariant across lanes.
    for (const auto &index : load->indices)
      if (!IsStableVectorizeIndex(index))
        return std::nullopt;
    auto offset = AddressOffset(*tensor, load->indices, layouts);
    if (!offset || !IsStableVectorizeIndex(*offset))
      return std::nullopt;
    return AtomicAddress{*tensor,
                         cast(*offset, PrimType::Int(64)) *
                             IntImm(PrimType::Int(64), tensor->dtype().bits())};
  }
  const auto *call = expression.as<CallObj>();
  if (!call)
    return std::nullopt;
  const auto op = call->op.as<Op>();
  if (!op)
    return std::nullopt;
  if (call->op.same_as(tl::access_ptr()) ||
      call->op.same_as(tl::ptr_access())) {
    const size_t extent_index = call->op.same_as(tl::access_ptr()) ? 1 : 2;
    if (call->args.size() != extent_index + 2)
      return std::nullopt;
    const auto extent = call->args[extent_index].as<PrimExpr>();
    const auto mask = call->args[extent_index + 1].as<IntImm>();
    if (!extent || !mask || (*mask)->value != 3 ||
        SideEffect(*extent) > CallEffectKind::kPure ||
        ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
            *extent,
            [&](const VarObj *var) -> ffi::Expected<ffi::WalkResult> {
              return (var == lane.get())
                         ? ffi::WalkResult::Interrupt(
                               ffi::VisitInterrupt(ffi::GetRef<Var>(var)))
                         : ffi::WalkResult::Skip();
            })
            .has_value() ||
        !analyzer->CanProve(*extent >= IntImm(extent->ty(), 1)))
      return std::nullopt;
    if (call->op.same_as(tl::access_ptr()))
      return ResolveAtomicAddress(call->args[0], layouts, analyzer, lane);
  }
  if (call->op.same_as(tirx::address_of_op())) {
    if (call->args.size() != 1)
      return std::nullopt;
    return ResolveAtomicAddress(call->args[0], layouts, analyzer, lane);
  }
  if (call->op.same_as(tirx::tensor_data_ptr_op())) {
    if (call->args.size() != 1)
      return std::nullopt;
    const auto tensor = call->args[0].as<TensorVar>();
    if (!tensor)
      return std::nullopt;
    return AtomicAddress{*tensor, IntImm(PrimType::Int(64), 0)};
  }
  if (call->op.same_as(tirx::reinterpret_op())) {
    if (call->args.size() != 1)
      return std::nullopt;
    const auto source = call->args[0]->ty.as<PtrType>();
    const auto target = call->ty.as<PtrType>();
    if (!source || !target || (*source)->scope != (*target)->scope)
      return std::nullopt;
    return ResolveAtomicAddress(call->args[0], layouts, analyzer, lane);
  }
  if (call->op.same_as(tl::ptr_access()) ||
      call->op.same_as(tirx::ptr_byte_offset_op())) {
    if (call->args.size() < 2)
      return std::nullopt;
    auto base = ResolveAtomicAddress(call->args[0], layouts, analyzer, lane);
    const auto offset = call->args[1].as<PrimExpr>();
    if (!base || !offset || !IsStableVectorizeIndex(*offset))
      return std::nullopt;
    int bits = 8;
    if (call->op.same_as(tl::ptr_access())) {
      const auto pointer = call->ty.as<PtrType>();
      const auto element = pointer ? (*pointer)->element_type.as<PrimType>()
                                   : Optional<PrimType>();
      if (!element || !element.value().IsScalar() ||
          element.value().bits() <= 0 || element.value().bits() % 8)
        return std::nullopt;
      bits = element.value().bits();
    }
    base->bit_offset = base->bit_offset + cast(*offset, PrimType::Int(64)) *
                                              IntImm(PrimType::Int(64), bits);
    return base;
  }
  return std::nullopt;
}

bool ContainsNestedLoop(const ForObj *loop) {
  bool found = false;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      loop->body,
      [&](const ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
        found |= object.as<ForObj>() != nullptr;
        return ffi::WalkResult::Advance();
      });
  return found;
}

bool HasMultipleStatements(const Stmt &stmt) {
  bool multiple = false;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      stmt, [&](const SeqStmtObj *seq) -> ffi::Expected<ffi::WalkResult> {
        int count = 0;
        for (const Stmt &child : seq->seq)
          count += child.as<BindObj>() == nullptr;
        multiple |= count > 1;

        return ffi::WalkResult::Advance();
      });
  return multiple;
}

bool GlobalOnlyMemory(const Stmt &stmt) {
  bool global = false, shared = false;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      stmt, [&](const ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
        Optional<TensorVar> tensor;
        if (const auto *load = object.as<TensorLoadObj>())
          tensor = load->source.as<TensorVar>();
        if (const auto *store = object.as<TensorStoreObj>())
          tensor = store->dest.as<TensorVar>();
        if (tensor) {
          global |= IsGlobalBuffer(*tensor);
          shared |= IsSharedBuffer(*tensor);
        }
        return ffi::WalkResult::Advance();
      });
  return global && !shared;
}

enum class ConstraintKind {
  kMemory,
  kLocal,
  kBroadcastLoad,
  kCall,
  kCast,
  kScalar
};
struct Constraint {
  ConstraintKind kind;
  int width;
  Optional<PrimExpr> offset;
  For loop;
  sym::Analyzer context;
};

class VectorizePlanner : public StmtExprVisitor {
public:
  VectorizePlanner(sym::Analyzer analyzer, const LayoutMap &layouts,
                   Target target)
      : context_(std::move(analyzer)), layouts_(layouts),
        target_(std::move(target)) {}
  using StmtExprVisitor::Visit_;
  int Plan(const For &loop) {
    bits_ = MaxVectorLoadBits(target_, GlobalOnlyMemory(loop));
    extent_width_ = bits_;
    Visit(loop);
    bool multi = HasMultipleStatements(loop);
    int memory = bits_, local = bits_, call = bits_, cast_width = bits_;
    bool has_memory = false;
    for (const Constraint &item : constraints_) {
      if (item.kind == ConstraintKind::kScalar)
        return 1;
      if (item.kind == ConstraintKind::kMemory) {
        memory = std::gcd(memory, item.width);
        has_memory = true;
      } else if (item.kind == ConstraintKind::kLocal ||
                 item.kind == ConstraintKind::kBroadcastLoad) {
        local = std::gcd(local, item.width);
      } else if (item.kind == ConstraintKind::kCast) {
        cast_width = std::gcd(cast_width, item.width);
      } else
        call = std::gcd(call, item.width);
    }
    int width = has_memory && !multi ? std::gcd(memory, call)
                                     : std::gcd(std::gcd(memory, local),
                                                std::gcd(call, cast_width));
    for (const Constraint &item : constraints_) {
      if (width > item.width && item.offset &&
          (item.kind == ConstraintKind::kLocal ||
           item.kind == ConstraintKind::kBroadcastLoad) &&
          !IndicesCanVectorize(*item.offset, item.loop->var,
                               LoopExtent(item.loop.get()), width,
                               item.context.get(), true, item.loop->begin))
        width = std::gcd(width, item.width);
    }
    width = std::max(1, std::gcd(width, extent_width_));
    if (tl_config::VectorizePlannerVerboseEnabled())
      std::cerr << "TileLang native vector plan: width=" << width
                << ", constraints=" << constraints_.size() << "\n";
    return width;
  }

  Optional<VisitInterrupt> Visit_(const ForObj *node) final {
    auto saved_context = context_;
    auto saved_loop = loop_;
    auto saved_bindings = scalar_bindings_;
    context_ = context_->Clone();
    loop_ = GetRef<For>(node);
    // Values defined outside this loop are captured once for its execution.
    // Only local Bind expressions need expansion when proving lane sharing.
    scalar_bindings_ = {};
    BindAnalyzer(context_, node->var, Range(node->begin, node->end), true);
    if (!ContainsNestedLoop(node)) {
      auto extent =
          IsOne(node->step)
              ? TryConstInt(context_->Simplify(node->end - node->begin))
              : std::nullopt;
      if (!extent || *extent <= 0)
        extent_width_ = 1;
      else
        extent_width_ = std::gcd<int64_t>(extent_width_, *extent);
    }
    auto result = StmtExprVisitor::Visit_(node);
    context_ = saved_context;
    loop_ = saved_loop;
    scalar_bindings_ = saved_bindings;
    return result;
  }
  Optional<VisitInterrupt> Visit_(const SeqStmtObj *node) final {
    auto saved = context_;
    auto saved_bindings = scalar_bindings_;
    context_ = context_->Clone();
    auto result = StmtExprVisitor::Visit_(node);
    context_ = saved;
    scalar_bindings_ = saved_bindings;
    return result;
  }
  Optional<VisitInterrupt> Visit_(const BindObj *node) final {
    auto result = Visit(node->value);
    if (result)
      return result;
    if (auto var = node->var.as<PrimVar>()) {
      if (auto value = node->value.as<PrimExpr>()) {
        scalar_bindings_.Set(
            *var, tirx::SubstituteWithDataTypeLegalization(
                      *value, [&](const Var &var) -> ffi::Optional<PrimExpr> {
                        return scalar_bindings_.Get(var);
                      }));
        if (IsStableVectorizeIndex(*value))
          BindAnalyzer(context_, *var, *value, true);
      }
    }
    return std::nullopt;
  }
  Optional<VisitInterrupt> Visit_(const TensorLoadObj *node) final {
    if (auto tensor = node->source.as<TensorVar>())
      Memory(*tensor, node->indices, false);
    return StmtExprVisitor::Visit_(node);
  }
  Optional<VisitInterrupt> Visit_(const TensorStoreObj *node) final {
    if (auto tensor = node->dest.as<TensorVar>())
      Memory(*tensor, node->indices, true);
    return StmtExprVisitor::Visit_(node);
  }
  Optional<VisitInterrupt> Visit_(const IfObj *node) final {
    Uniform(node->condition);
    return StmtExprVisitor::Visit_(node);
  }
  Optional<VisitInterrupt> Visit_(const SelectObj *node) final {
    if (!target_->GetAttr<bool>("supports_vector_predicate", false).value())
      Uniform(node->condition);
    return StmtExprVisitor::Visit_(node);
  }
  Optional<VisitInterrupt> Visit_(const CastObj *node) final {
    int source_bits = node->value.ty().bits();
    int target_bits = node->ty.as_or_throw<PrimType>().bits();
    Add(ConstraintKind::kCast,
        std::max(1, bits_ / std::max(source_bits, target_bits)));
    return StmtExprVisitor::Visit_(node);
  }
  Optional<VisitInterrupt> Visit_(const CallObj *node) final {
    auto op = node->op.as<Op>();
    String name = op ? (*op)->name : "";
    if (node->op.same_as(prim::if_then_else_op())) {
      Uniform(node->args[0].as_or_throw<PrimExpr>());
    } else if (node->op.same_as(tl::cast_op())) {
      // The value operand varies across lanes. Only stochastic rounding bits
      // are shared by one packed conversion; retain the ordinary cast width
      // bound for both the source and result types.
      PrimType source = node->args[0]->ty.as_or_throw<PrimType>();
      PrimType result = node->ty.as_or_throw<PrimType>();
      Add(ConstraintKind::kCast,
          std::max(1, bits_ / std::max(source.bits(), result.bits())));
      if (node->args.size() == 2) {
        PrimExpr rbits = tirx::SubstituteWithDataTypeLegalization(
            node->args[1].as_or_throw<PrimExpr>(),
            [&](const Var &var) -> ffi::Optional<PrimExpr> {
              return scalar_bindings_.Get(var);
            });
        // Check effects before simplification, including effects hidden by a
        // local SSA definition. An RNG call must still execute for each lane.
        if (SideEffect(rbits) > CallEffectKind::kReadState)
          Add(ConstraintKind::kScalar, 1);
        else
          Uniform(rbits);
      }
    } else if (node->op.same_as(tl::rng_rand_float())) {
      auto dtype = node->ty.as_or_throw<PrimType>();
      Add(ConstraintKind::kCall,
          std::max(1, std::min(4, bits_ / (dtype.bits() * dtype.lanes()))));
    } else if (name == "tl.atomic_add_elem" ||
               node->op.same_as(tl::atomic_add_elem_op())) {
      auto pointer = node->args[0]->ty.as<PtrType>();
      auto element = pointer ? (*pointer)->element_type.as<PrimType>()
                             : Optional<PrimType>();
      auto replace = make_object<LayoutMapMutator>();
      for (const auto &[var, value] : scalar_bindings_)
        replace->VarRemapSet(var, value);
      Expr pointer_value =
          replace->Mutate(node->args[0]).ValueOrUnchanged(node->args[0]);
      auto address = loop_ ? ResolveAtomicAddress(pointer_value, layouts_,
                                                  context_.get(), (*loop_)->var)
                           : std::optional<AtomicAddress>();
      if (!address || !element || !element.value().IsScalar() ||
          element.value().bits() <= 0 || element.value().bits() % 8)
        Add(ConstraintKind::kScalar, 1);
      else {
        PrimExpr bits = IntImm(PrimType::Int(64), element.value().bits());
        PrimExpr offset = floordiv(address->bit_offset, bits);
        bool integral =
            context_->CanProveEqual(floormod(address->bit_offset, bits), 0);
        int width =
            GetMaxAtomicVectorSize(*element, (*pointer)->scope.name(), target_);
        while (width > 1 &&
               (!integral ||
                address->tensor.ty()->data_alignment %
                        (width * element.value().bits() / 8) !=
                    0 ||
                !IndicesCanVectorize(offset, (*loop_)->var,
                                     LoopExtent(loop_.value().get()), width,
                                     context_.get(), false, (*loop_)->begin)))
          width /= 2;
        Add(ConstraintKind::kCall, width);
      }
      // The destination is a pointer footprint, not a load of its pointee.
      // Its address/extent/mask have just been checked by the owning rule.
      // Generic pointer-call visitation would require a lane-varying raw
      // element offset to be uniform and discard otherwise valid plans.
      for (size_t i = 1; i < node->args.size(); ++i) {
        if (i >= 2) {
          if (SideEffect(node->args[i]) > CallEffectKind::kPure)
            Add(ConstraintKind::kScalar, 1);
          if (auto value = node->args[i].as<PrimExpr>())
            Uniform(*value);
        }
        if (auto interrupted = Visit(node->args[i]))
          return interrupted;
      }
      return std::nullopt;
    } else if (node->op.same_as(tl::ptx_cp_async()) ||
               name == "tirx.ptx_cp_async" || name == "tirx.gpu_cp_async") {
      int64_t count = GetIntegerArgument(node->args[2]);
      int64_t bits_per_call = count * 8;
      if (node->op.same_as(tl::ptx_cp_async())) {
        auto dst = node->args[0]->ty.as<PtrType>();
        auto src = node->args[1]->ty.as<PtrType>();
        if (!dst || !src)
          bits_per_call = 0;
        else {
          auto d = dst->operator->()->element_type.as_or_throw<PrimType>();
          auto s = src->operator->()->element_type.as_or_throw<PrimType>();
          ICHECK(d.bits() * d.lanes() == s.bits() * s.lanes());
          bits_per_call = count * d.bits() * d.lanes();
        }
      }
      int width = 1;
      if (bits_per_call > 0)
        for (int bytes : {4, 8, 16})
          if ((bytes * 8) % bits_per_call == 0)
            width = std::max<int64_t>(width, (bytes * 8) / bits_per_call);
      Add(ConstraintKind::kCall, width);
    } else if (node->op.same_as(tl::access_ptr()) ||
               node->op.same_as(tl::reducer_update())) {
      // These explicit operands describe the memory touched by the later
      // lowering. Preserve contribution loads and tensor identity in planning.
    } else {
      bool vectorizable = op && Op::HasAttrMap("TVectorizable") &&
                          Op::GetAttrMap<bool>("TVectorizable").get(*op, false);
      if (!vectorizable) {
        for (const Expr &arg : node->args) {
          if (auto scalar = arg.as<PrimExpr>())
            Uniform(*scalar);
        }
      }
    }
    return StmtExprVisitor::Visit_(node);
  }

private:
  void Add(ConstraintKind kind, int width,
           Optional<PrimExpr> offset = std::nullopt) {
    if (!loop_)
      return;
    constraints_.push_back(
        {kind, std::max(1, width), offset, *loop_, context_->Clone()});
  }
  void Uniform(const PrimExpr &expr) {
    if (!loop_ || !IsOne((*loop_)->step))
      return;
    int width = extent_width_;
    while (width > 1 &&
           !IsExprInvariantInVectorBoundary(
               expr, (*loop_)->var, LoopExtent(loop_.value().get()), width,
               context_.get(), (*loop_)->begin))
      width /= 2;
    Add(ConstraintKind::kCall, width);
  }
  void Memory(const TensorVar &tensor, const Array<PrimExpr> &indices,
              bool store) {
    if (!loop_ || !IsOne((*loop_)->step))
      return;
    auto expanded = indices.Map([&](const PrimExpr &index) {
      return tirx::SubstituteWithDataTypeLegalization(
          index, [&](const Var &var) -> ffi::Optional<PrimExpr> {
            return scalar_bindings_.Get(var);
          });
    });
    for (const auto &index : expanded) {
      if (!IsStableVectorizeIndex(index)) {
        Add(ConstraintKind::kScalar, 1);
        return;
      }
    }
    auto offset = AddressOffset(tensor, expanded, layouts_);
    if (!offset || !IsStableVectorizeIndex(*offset)) {
      Add(ConstraintKind::kScalar, 1);
      return;
    }
    offset = context_->Simplify(*offset);
    int bits = tensor.dtype().bits() * tensor.dtype().lanes();
    int bound = std::max(1, bits_ / std::max(1, bits));
    int width = std::max(1, std::gcd(extent_width_, bound));
    PrimExpr extent = LoopExtent(loop_.value().get());
    bool independent =
        CanProveIndependent(*offset, (*loop_)->var, context_.get());
    bool broadcast = IsExprInvariantInVectorBoundary(
        *offset, (*loop_)->var, extent, width, context_.get(), (*loop_)->begin);
    if (store && (independent || broadcast)) {
      Add(ConstraintKind::kScalar, 1, offset);
      return;
    }
    while (width > 1 &&
           !IndicesCanVectorize(*offset, (*loop_)->var, extent, width,
                                context_.get(), !store, (*loop_)->begin))
      width /= 2;
    ConstraintKind kind =
        IsLocalBuffer(tensor, true) || IsFragmentBuffer(tensor)
            ? ConstraintKind::kLocal
        : broadcast && !store ? ConstraintKind::kBroadcastLoad
                              : ConstraintKind::kMemory;
    Add(kind, width, offset);
  }
  sym::Analyzer context_;
  ffi::Map<Var, PrimExpr> scalar_bindings_;
  LayoutMap layouts_;
  Target target_;
  Optional<For> loop_;
  int bits_ = 128;
  int extent_width_ = 128;
  std::vector<Constraint> constraints_;
};

class VectorizeRewriter : public LayoutMapMutator {
public:
  VectorizeRewriter(int width, sym::Analyzer analyzer)
      : width_(width), analyzer_(std::move(analyzer)) {}
  using LayoutMapMutator::Mutate_;
  UnchangedOr<Stmt> Mutate_(const ForObj *node, InplaceMode mode) final {
    For loop = LayoutMapMutator::Mutate_(node, mode)
                   .ValueOrUnchanged(GetRef<Stmt>(node))
                   .as_or_throw<For>();
    if (ContainsNestedLoop(node) || width_ == 1 || !IsOne(node->step)) {
      if (loop->kind == ForKind::kParallel)
        loop.CopyOnWrite()->kind = ForKind::kDefault;
      return loop;
    }
    auto extent = TryConstInt(analyzer_->Simplify(LoopExtent(node)));
    ICHECK(extent && *extent > 0 && *extent % width_ == 0);
    PrimType dtype = node->var.ty();
    PrimVar lane("vec", dtype), group(node->var->name, dtype);
    PrimExpr zero = IntImm(dtype, 0), one = IntImm(dtype, 1),
             width = IntImm(dtype, width_);
    PrimExpr replacement = node->begin + lane;
    if (*extent != width_)
      replacement = node->begin + group * width + lane;
    Stmt body = tl::SubstituteWithDataTypeLegalization(
        loop->body,
        [values = ffi::Map<Var, PrimExpr>{{node->var, replacement}}](
            const Var &var) -> ffi::Optional<PrimExpr> {
          return values.Get(var);
        });
    For vector(lane, zero, width, one, ForKind::kVectorized, SeqStmt(body), {},
               node->loc);
    if (*extent == width_) {
      vector.CopyOnWrite()->annotations = node->annotations;
      return vector;
    }
    ForKind outer =
        node->kind == ForKind::kParallel ? ForKind::kDefault : node->kind;
    return For(group, zero, IntImm(dtype, *extent / width_), one, outer,
               SeqStmt(vector), node->annotations, node->loc);
  }

private:
  int width_;
  sym::Analyzer analyzer_;
};
} // namespace

int GetVectorizeSize(const For &loop, const LayoutMap &layouts,
                     Optional<Target> target) {
  auto analyzer = CreateTileLangAnalyzer();
  return GetVectorizeSize(loop, analyzer.get(), layouts, std::move(target));
}
int GetVectorizeSize(const For &loop, const sym::AnalyzerObj *analyzer,
                     const LayoutMap &layouts, Optional<Target> target) {
  auto context = analyzer ? analyzer->Clone() : CreateTileLangAnalyzer();
  Target effective_target =
      target.has_value() ? target.value() : Target::Current(false).value();
  return make_object<VectorizePlanner>(context, layouts, effective_target)
      ->Plan(loop);
}
For VectorizeLoop(const For &loop, const LayoutMap &layouts, int hint,
                  Optional<Target> target) {
  auto analyzer = CreateTileLangAnalyzer();
  return VectorizeLoop(loop, analyzer.get(), layouts, hint, std::move(target));
}
For VectorizeLoop(const For &loop, const sym::AnalyzerObj *analyzer,
                  const LayoutMap &layouts, int hint, Optional<Target> target) {
  if (hint <= 0)
    hint = GetVectorizeSize(loop, analyzer, layouts, std::move(target));
  auto context = analyzer ? analyzer->Clone() : CreateTileLangAnalyzer();
  auto rewriter = make_object<VectorizeRewriter>(hint, context);
  return rewriter->Mutate(loop).ValueOrUnchanged(loop).as_or_throw<For>();
}
} // namespace tvm::tl
