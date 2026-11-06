#include "cuda/op/builtin.h"
#include "ir/stmt_functor.h"
#include "ir/vectorize.h"
#include "op/builtin.h"
#include <tvm/ffi/extra/structural_visit.h>
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file vectorize_loop.cc
 * \brief TileLang intrinsic rules for the shared native loop vectorizer.
 */
#include "../../3rdparty/tvm-tirx/src/tirx/ir/data_type_rewriter.h"
#include "../ir/analyzer.h"
#include "../ir/annotations.h"
#include "../ir/expr_utils.h"
#include "../ir/stmt_functor.h"
#include "../ir/vectorize.h"
#include "../op/builtin.h"
#include "common/pipeline_access.h"
#include "ir/annotations.h"
#include "loop_vectorize.h"
#include <optional>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
using namespace ffi;
using namespace prim;
using namespace tirx;

namespace {
bool IsElementwisePrimOp(const Op &op) {
  return op->name.starts_with("prim.") &&
         Op::GetAttrMap(tirx::op_attr::kVectorizable).get(op, false) &&
         Op::GetAttrMap(tvm::op_attr::kCallEffectKind)
                 .get(op, CallEffectKind::kOpaque) <= CallEffectKind::kPure;
}

bool UsesLane(ffi::AnyView value, const Var &lane) {
  return ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
             value,
             [&](const VarObj *var) -> ffi::Expected<ffi::WalkResult> {
               return (var == lane.get())
                          ? ffi::WalkResult::Interrupt(
                                ffi::VisitInterrupt(ffi::GetRef<Var>(var)))
                          : ffi::WalkResult::Skip();
             })
      .has_value();
}

// Only an owning intrinsic rule may discard the lane from a pointer, after
// proving that all scalar accesses form one combined transaction.
Expr AtLaneZero(Expr expression, const Var &lane) {
  auto replace = make_object<tirx::StmtExprMutator>();
  replace->VarRemapSet(lane, IntImm(lane->ty.as_or_throw<PrimType>(), 0));
  return replace->Mutate(expression, InplaceMode::kDisallow)
      .ValueOrUnchanged(expression);
}

struct Address {
  Expr root;
  PrimExpr bit_offset;
};

std::optional<Address> MemoryAddress(const Expr &value) {
  if (const auto *load = value.as<TensorLoadObj>()) {
    const auto tensor = load->source.as<TensorVar>();
    if (!tensor)
      return std::nullopt;
    for (const auto &index : load->indices)
      if (!IsStableVectorizeIndex(index))
        return std::nullopt;
    const auto offsets = tensor.value().ty()->ElemOffset(load->indices);
    if (offsets.size() != 1 || !IsStableVectorizeIndex(offsets[0]))
      return std::nullopt;
    const PrimType element = tensor.value().dtype();
    if (element.IsScalableVector() || element.bits() <= 0)
      return std::nullopt;
    return Address{load->source, cast(offsets[0], PrimType::Int(64)) *
                                     IntImm(PrimType::Int(64),
                                            element.bits() * element.lanes())};
  }
  if (const auto *call = value.as<CallObj>()) {
    const auto op = call->op.as<Op>();
    if (!op)
      return std::nullopt;
    if (call->op.same_as(tl::access_ptr()) ||
        call->op.same_as(tirx::address_of_op()))
      return MemoryAddress(call->args[0]);
    if (call->op.same_as(tirx::tensor_data_ptr_op()))
      return Address{call->args[0], IntImm(PrimType::Int(64), 0)};
    if (call->op.same_as(tirx::reinterpret_op()) && call->args.size() == 1) {
      const auto source = call->args[0]->ty.as<PtrType>();
      const auto target = call->ty.as<PtrType>();
      if (source && target && (*source)->scope == (*target)->scope)
        return MemoryAddress(call->args[0]);
      return std::nullopt;
    }
    if (call->op.same_as(tl::ptr_access()) ||
        call->op.same_as(tirx::ptr_byte_offset_op())) {
      auto base = MemoryAddress(call->args[0]);
      if (!base)
        return std::nullopt;
      PrimExpr offset = call->args[1].as_or_throw<PrimExpr>();
      if (!IsStableVectorizeIndex(offset))
        return std::nullopt;
      int bits = 8;
      if (call->op.same_as(tl::ptr_access())) {
        const auto type = call->ty.as_or_throw<PtrType>();
        const PrimType element = type->element_type.as_or_throw<PrimType>();
        if (!element.IsScalar() || element.bits() <= 0)
          return std::nullopt;
        bits = element.bits();
      }
      PrimType index_type = base->bit_offset.ty();
      return Address{base->root,
                     base->bit_offset +
                         cast(offset, index_type) * IntImm(index_type, bits)};
    }
  }
  if (value.as<VarObj>() && value->ty.as<PtrTypeObj>())
    return Address{value, IntImm(PrimType::Int(64), 0)};
  return std::nullopt;
}

bool AtomicAccessContract(const Expr &pointer, const Var &lane,
                          sym::AnalyzerObj *analyzer) {
  if (const auto *load = pointer.as<TensorLoadObj>()) {
    for (const auto &index : load->indices)
      if (!IsStableVectorizeIndex(index))
        return false;
    return true;
  }
  const auto *call = pointer.as<CallObj>();
  if (!call || !call->op.as<Op>())
    return false;
  if (call->op.same_as(tl::access_ptr()) ||
      call->op.same_as(tl::ptr_access())) {
    const size_t extent_index = call->op.same_as(tl::access_ptr()) ? 1 : 2;
    if (call->args.size() != extent_index + 2)
      return false;
    const auto extent = call->args[extent_index].as<PrimExpr>();
    const auto mask = call->args[extent_index + 1].as<IntImm>();
    if (!extent || !mask || (*mask)->value != 3 || UsesLane(*extent, lane) ||
        SideEffect(*extent) > CallEffectKind::kPure ||
        !analyzer->CanProve(*extent >= IntImm(extent->ty(), 1)))
      return false;
    return AtomicAccessContract(call->args[0], lane, analyzer);
  }
  if (call->op.same_as(tirx::tensor_data_ptr_op()))
    return call->args.size() == 1 && call->args[0]->ty.as<TensorTypeObj>();
  if (call->op.same_as(tirx::address_of_op()) ||
      call->op.same_as(tirx::ptr_byte_offset_op()) ||
      call->op.same_as(tirx::reinterpret_op()))
    return !call->args.empty() &&
           AtomicAccessContract(call->args[0], lane, analyzer);
  return false;
}

bool IsContiguous(const Expr &pointer, const Var &lane, int lanes,
                  int bits_per_lane, sym::AnalyzerObj *analyzer,
                  bool require_allocation_alignment = true,
                  Optional<PrimExpr> begin = std::nullopt) {
  const auto address = MemoryAddress(pointer);
  if (!address || UsesLane(address->root, lane))
    return false;
  const auto allocation = address->root->ty.as<TensorType>();
  if (!allocation ||
      (require_allocation_alignment &&
       allocation.value()->data_alignment * 8 % (bits_per_lane * lanes) != 0))
    return false;
  const PrimExpr first =
      begin.value_or(IntImm(lane->ty.as_or_throw<PrimType>(), 0));
  const PrimExpr base =
      analyzer->Simplify(tirx::SubstituteWithDataTypeLegalization(
          address->bit_offset,
          [values = ffi::Map<Var, PrimExpr>{{lane, first}}](const Var &var)
              -> ffi::Optional<PrimExpr> { return values.Get(var); }));
  const PrimType index_type = base.ty();
  // The native tensor type guarantees base-allocation alignment. Preserve the
  // layout/view offset and require the access to start on its vector boundary.
  // Repeated, reversed and strided addresses must retain their scalar calls.
  if (!analyzer->CanProveEqual(
          floormod(base, IntImm(index_type, bits_per_lane * lanes)), 0))
    return false;
  for (int i = 1; i < lanes; ++i) {
    auto offset = tirx::SubstituteWithDataTypeLegalization(
        address->bit_offset,
        [values =
             ffi::Map<Var, PrimExpr>{
                 {lane, first + IntImm(lane->ty.as_or_throw<PrimType>(), i)}}](
            const Var &var) -> ffi::Optional<PrimExpr> {
          return values.Get(var);
        });
    if (!analyzer->CanProveEqual(offset,
                                 base + IntImm(index_type, i * bits_per_lane)))
      return false;
  }
  return true;
}

Expr BasePointer(const Expr &pointer, const Var &lane, int elements) {
  Expr result = AtLaneZero(pointer, lane);
  if (const auto *call = result.as<CallObj>()) {
    auto op = call->op.as<Op>();
    if (op && (op.value().same_as(tl::access_ptr()) ||
               op.value().same_as(tl::ptr_access()))) {
      auto args = call->args;
      const size_t extent_index = op.value().same_as(tl::access_ptr()) ? 1 : 2;
      PrimType index_type = args[extent_index].as_or_throw<PrimExpr>().ty();
      args.Set(extent_index, IntImm(index_type, elements));
      return Call(call->ty, call->op, args, call->attrs, call->ty_args,
                  call->loc);
    }
  }
  return result;
}

Optional<Expr> VectorizeCPAsync(Call call, Var lane, PrimExpr lane_count,
                                Optional<Target>,
                                TypedFunction<Expr(Expr)> rewrite) {
  const auto width = lane_count.as<IntImm>();
  if (!width || width.value()->value <= 1 || width.value()->value > 128)
    return std::nullopt;
  const int lanes = width.value()->value.as<int>().value();
  const auto operands = DecodeCallOperands(call.get());
  if (operands.size() != 3 && operands.size() != 4)
    return std::nullopt;
  const auto count = operands[2].as<IntImm>();
  if (!count || count.value()->value <= 0 || count.value()->value > 16)
    return std::nullopt;
  const int per_lane = count.value()->value.as<int>().value();
  const bool bytes_form =
      call->op.as_or_throw<Op>().same_as(tl::ptx_cp_async_bytes());
  const auto dst = operands[0]->ty.as_or_throw<PtrType>();
  const auto src = operands[1]->ty.as_or_throw<PtrType>();
  const PrimType dst_element = dst->element_type.as_or_throw<PrimType>();
  const PrimType src_element = src->element_type.as_or_throw<PrimType>();
  if (!dst_element.IsScalar() || !src_element.IsScalar() ||
      dst_element.bits() <= 0 || src_element.bits() <= 0)
    return std::nullopt;
  if (!bytes_form && dst_element.bits() != src_element.bits())
    return std::nullopt;
  const int scalar_bits = per_lane * (bytes_form ? 8 : dst_element.bits());
  const int total_bits = scalar_bits * lanes;
  if (total_bits != 32 && total_bits != 64 && total_bits != 128)
    return std::nullopt;
  if (scalar_bits % dst_element.bits() || scalar_bits % src_element.bits())
    return std::nullopt;
  // A dynamic metadata payload must retain its per-call meaning.
  if (call->args.size() > operands.size() && UsesLane(call->args.back(), lane))
    return std::nullopt;
  auto analyzer = CreateTileLangAnalyzer();
  // Explicit sub-instruction counts are defined by their complete vector
  // group. They are not independently executable scalar transactions. Keep
  // the original low-level final-instruction alignment obligation local to
  // this form; never strengthen TensorType alignment or automatic memory
  // accesses. Already legal scalar calls still need proof before widening.
  const bool require_allocation_alignment = scalar_bits >= 32;
  if (!IsContiguous(operands[0], lane, lanes, scalar_bits, analyzer.get(),
                    require_allocation_alignment) ||
      !IsContiguous(operands[1], lane, lanes, scalar_bits, analyzer.get(),
                    require_allocation_alignment))
    return std::nullopt;
  auto args = call->args;
  if (operands.size() == 4) {
    auto predicate = rewrite(operands[3]).as_or_throw<PrimExpr>();
    if (!predicate.ty().IsScalar())
      return std::nullopt;
    args.Set(3, predicate);
  }
  args.Set(0, BasePointer(operands[0], lane, total_bits / dst_element.bits()));
  args.Set(1, BasePointer(operands[1], lane, total_bits / src_element.bits()));
  args.Set(2, IntImm(count.value().ty(), per_lane * lanes));
  return Call(call->ty, call->op, args, call->attrs, call->ty_args, call->loc);
}

Optional<Expr> VectorizeAtomicAdd(Call call, Var lane, PrimExpr lane_count,
                                  Optional<Target> target,
                                  TypedFunction<Expr(Expr)> rewrite) {
  const auto width = lane_count.as<IntImm>();
  if (!width || (width.value()->value != 2 && width.value()->value != 4) ||
      !target)
    return std::nullopt;
  const int lanes = width.value()->value.as<int>().value();
  const auto operands = DecodeCallOperands(call.get());
  const auto destination = operands[0]->ty.as_or_throw<PtrType>();
  const auto element = destination->element_type.as_or_throw<PrimType>();
  if (lanes >
      GetMaxAtomicVectorSize(element, destination->scope.name(), *target))
    return std::nullopt;
  const auto source = operands[1].as<TensorLoad>();
  const bool bound_value = operands[1].as<PrimVar>().has_value();
  // A vectorized Bind already evaluates its contribution at the original
  // lexical position. Preserve that value instead of inventing a pointer or
  // duplicating its defining load. Other computed/broadcast expressions keep
  // the existing scalar fallback until they have an owning packing contract.
  if (!source && !bound_value)
    return std::nullopt;
  for (size_t index = 2; index < call->args.size(); ++index)
    if (UsesLane(call->args[index], lane) ||
        SideEffect(call->args[index]) > CallEffectKind::kPure)
      return std::nullopt;
  auto analyzer = CreateTileLangAnalyzer();
  if (!AtomicAccessContract(operands[0], lane, analyzer.get()) ||
      !IsContiguous(operands[0], lane, lanes, element.bits(), analyzer.get()))
    return std::nullopt;
  if (source && !IsContiguous(*source, lane, lanes, source.value().ty().bits(),
                              analyzer.get()))
    return std::nullopt;
  auto args = call->args;
  // The owning vectorizer must also remap any local scalar Bind used in the
  // base. A changed vector-valued address asks it to retain the scalar loop;
  // returning an old bound Var would leave a dangling definition instead.
  args.Set(0, rewrite(BasePointer(operands[0], lane, lanes)));
  if (source && source.value().ty() == element) {
    auto source_base = AtLaneZero(*source, lane).as_or_throw<TensorLoad>();
    const auto source_type = source_base->source->ty.as_or_throw<TensorType>();
    args.Set(1, rewrite(Call(source_type->DataPtrType(), tirx::address_of_op(),
                             {source_base}, {}, {}, source_base->loc)));
  } else {
    const auto value = rewrite(operands[1]).as<PrimExpr>();
    if (!value || !value.value().ty().IsFixedLengthVector() ||
        value.value().ty().lanes() != lanes)
      return std::nullopt;
    // Match the scalar AtomicAdd conversion, using the vector's own type.
    // CUDA's packed helpers accept both typed pointers and packed values.
    args.Set(1, cast(*value, element.WithLanes(lanes)));
  }
  const Op op = lanes == 2 ? atomic_addx2_elem_op() : atomic_addx4_elem_op();
  return Call(call->ty, op, args, call->attrs, call->ty_args, call->loc);
}

Optional<Expr> VectorizeClamp(Call call, Var, PrimExpr, Optional<Target>,
                              TypedFunction<Expr(Expr)> rewrite) {
  // This rule owns only the three pure elementwise operands. In particular,
  // it does not infer vector semantics for an arbitrary tl.* intrinsic.
  const auto type = call->ty.as<PrimType>();
  if (!type || !type->IsScalar() || type->IsVoid() || call->args.size() != 3 ||
      call->ty_args.size() > 1)
    return std::nullopt;
  Array<Expr> arguments;
  int lanes = 1;
  for (const auto &argument : call->args) {
    const auto value = rewrite(argument).as<PrimExpr>();
    if (!value || value->ty().IsScalableVector() ||
        value->ty().WithLanes(1) != *type)
      return std::nullopt;
    lanes = std::max(lanes, value->ty().lanes());
    arguments.push_back(*value);
  }
  const PrimType result_type = type->WithLanes(lanes);
  for (size_t i = 0; i < arguments.size(); ++i) {
    const auto value = arguments[i].as_or_throw<PrimExpr>();
    if (value.ty().IsScalar() && lanes != 1)
      arguments.Set(i, Broadcast(value, IntImm(PrimType::Int(32), lanes)));
    else if (value.ty() != result_type)
      return std::nullopt;
  }
  Array<Type> result_arguments;
  if (!call->ty_args.empty())
    result_arguments.push_back(result_type);
  return Call(result_type, call->op, arguments, call->attrs, result_arguments,
              call->loc);
}

// A predicate shared by lanes may be evaluated once only when its reads are
// stable and its arithmetic cannot trap. Positive divisors exclude both zero
// division and signed-min / -1 overflow; unknown divisors remain scalar.
bool SafePackingPredicate(const PrimExpr &condition,
                          sym::AnalyzerObj *analyzer) {
  if (condition.ty() != PrimType::Bool() || !IsStableVectorizeIndex(condition))
    return false;
  bool safe = true;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      condition,
      [&](const ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
        const PrimExpr *divisor = nullptr;
        if (const auto *op = object.as<DivObj>())
          divisor = &op->b;
        else if (const auto *op = object.as<ModObj>())
          divisor = &op->b;
        else if (const auto *op = object.as<FloorDivObj>())
          divisor = &op->b;
        else if (const auto *op = object.as<FloorModObj>())
          divisor = &op->b;
        if (divisor)
          safe &= analyzer->CanProve(*divisor > IntImm(divisor->ty(), 0));
        return ffi::WalkResult::Advance();
      });
  return safe;
}

// Admission concerns the original whole loop. Dtype legalization below is a
// constructor, not a proof that reordering scalar iterations is legal.
class PackingAnalysis : public StmtExprVisitor {
public:
  PackingAnalysis(Var lane, int lanes, const PipelineMemoryAnalysis *memory,
                  PrimExpr begin, const sym::Analyzer &analyzer)
      : lane_(std::move(lane)), lanes_(lanes), memory_(memory), begin_(begin),
        analyzer_(analyzer->Clone()) {
    BindAnalyzer(analyzer_, lane_,
                 Range::FromMinExtent(begin_, IntImm(begin_.ty(), lanes_)));
  }

  bool Run(const Stmt &body) {
    Visit(body);
    if (!allowed_ || (!owning_call_ && accesses_.empty()) ||
        effectful_calls_ > 1 || (effectful_calls_ && nonlocal_store_))
      return false;
    for (size_t i = 0; i < accesses_.size(); ++i) {
      const auto &a = accesses_[i];
      if (a.write && !IsContiguous(a.expression, lane_, lanes_, a.bits,
                                   analyzer_.get(), false, begin_))
        return false;
      for (size_t j = 0; j < i; ++j) {
        const auto &b = accesses_[j];
        if ((!a.write && !b.write) || !memory_->MayAlias(a.tensor, b.tensor))
          continue;
        // Equal per-lane accesses to the very same view are independent.
        // Distinct possibly-overlapping arguments and different views need
        // stronger provenance than equal-looking logical indices.
        if (!a.tensor.same_as(b.tensor) ||
            !((a.bits == b.bits &&
               analyzer_->CanProveEqual(a.offset, b.offset)) ||
              SeparateTransactions(a.expression, b.expression, a.bits, b.bits)))
          return false;
      }
    }
    return true;
  }

  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;

  const Map<PrimExpr, PrimExpr> &UniformLeadingIndices() const {
    return uniform_leading_indices_;
  }

  Optional<VisitInterrupt> Visit(ffi::AnyView value) final {
    if (!allowed_)
      return std::nullopt;
    if (auto stmt = value.as<Stmt>()) {
      if (!stmt->as<SeqStmtObj>() && !stmt->as<BindObj>() &&
          !stmt->as<TensorStoreObj>() && !stmt->as<EvaluateObj>() &&
          !stmt->as<RegionStmtObj>() && !stmt->as<IfObj>()) {
        allowed_ = false;
        return std::nullopt;
      }
    }
    if (auto expr = value.as<PrimExpr>()) {
      const auto *load = expr->as<TensorLoadObj>();
      bool vector_element =
          load && expr->ty().IsFixedLengthVector() &&
          load->source.as<TensorVar>() &&
          load->source.as_or_throw<TensorVar>().dtype() == expr->ty();
      if ((!expr->ty().IsScalar() && !expr->ty().IsVoid() && !vector_element) ||
          expr->as<prim::LetObj>()) {
        allowed_ = false;
        return std::nullopt;
      }
    }
    return StmtExprVisitor::Visit(value);
  }

  Optional<VisitInterrupt> Visit_(const BindObj *node) final {
    if (node->value->ty.as<TupleType>())
      return Visit(node->value);
    auto value = node->value.as<PrimExpr>();
    if (!value || !node->var.as<PrimVar>()) {
      allowed_ = false;
      return std::nullopt;
    }
    Visit(*value);
    bindings_.Set(node->var,
                  tirx::SubstituteWithDataTypeLegalization(
                      *value, [&](const Var &var) -> ffi::Optional<PrimExpr> {
                        return bindings_.Get(var);
                      }));
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const TensorLoadObj *node) final {
    const auto type = node->ty.as_or_throw<PrimType>();
    AddAccess(GetRef<Expr>(node), type.bits() * type.lanes(), false);
    CheckLeadingIndices(node->indices);
    return Visit(node->indices);
  }

  Optional<VisitInterrupt> Visit_(const TensorStoreObj *node) final {
    // Private temporaries may be packed before one atomic transaction. An
    // externally visible store must retain its scalar position relative to
    // that transaction, including acquire/release ordering. Check the owning
    // allocation, rather than trusting a view's declared storage scope.
    auto root = memory_->Root(node->dest);
    auto tensor = root ? root.value().as<TensorVar>() : Optional<TensorVar>();
    auto definition =
        root ? memory_->definitions().Get(*root) : Optional<Expr>();
    const auto *allocation =
        definition ? definition.value().as<CallObj>() : nullptr;
    nonlocal_store_ |= !tensor || tensor.value().scope().name() != "local" ||
                       !allocation ||
                       !allocation->op.same_as(tirx::alloc_tensor_op());
    TensorLoad address(node->dest, node->indices, node->loc);
    AddAccess(address, address.ty().bits() * address.ty().lanes(), true);
    CheckLeadingIndices(node->indices);
    Visit(node->indices);
    return Visit(node->value);
  }

  Optional<VisitInterrupt> Visit_(const IfObj *node) final {
    if (!UniformCondition(node->condition)) {
      allowed_ = false;
      return std::nullopt;
    }
    Visit(node->condition);
    const auto saved = bindings_;
    Visit(node->then_case);
    bindings_ = saved;
    if (node->else_case)
      Visit(*node->else_case);
    bindings_ = saved;
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const RegionStmtObj *node) final {
    // TileLang's annotation tuple is a typed lexical payload. Other regions
    // have their own control-flow, binding and execution contracts.
    if (node->op->name != "tl.attr" || !node->body_params.empty() ||
        !node->result_vars.empty()) {
      allowed_ = false;
      return std::nullopt;
    }
    auto fields = ReadAnnotations(node);
    if (auto key = fields.Get("key");
        key && key->as<String>() && key->cast<String>() == "volatile_scope") {
      allowed_ = false;
      return std::nullopt;
    }
    auto key = fields.Get("key");
    if (key && key->as<String>() &&
        key->cast<String>() == "tl.assume_requires_runtime_check") {
      auto predicate = fields.Get("node");
      auto condition =
          predicate ? predicate->as<PrimExpr>() : Optional<PrimExpr>();
      auto message = fields.Get("value");
      // This marker duplicates the same scalar assumption below. Its bound
      // variables are freshened per lane, just like the optimizer fact.
      // Other metadata schemas keep their ordinary admission checks.
      allowed_ &= fields.size() == 3 && message && message->as<String>() &&
                  condition &&
                  SideEffect(*condition) <= CallEffectKind::kPure &&
                  SafePackingPredicate(Expand(*condition), analyzer_.get());
    } else {
      Visit(node->args);
    }
    auto saved = bindings_;
    Visit(node->body);
    bindings_ = saved;
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const CallObj *node) final {
    auto op = node->op.as<Op>();
    auto rules = Op::GetAttrMap(kCallVectorize);
    if (!op) {
      allowed_ = false;
      return std::nullopt;
    }
    if (op->same_as(prim::assume_op())) {
      auto condition = node->args.size() == 1 ? node->args[0].as<PrimExpr>()
                                              : Optional<PrimExpr>();
      // An optimizer assumption has no per-lane memory effect. Its scalar
      // conjunction and checked annotation are reconstructed together below.
      allowed_ &= condition &&
                  SideEffect(*condition) <= CallEffectKind::kPure &&
                  SafePackingPredicate(Expand(*condition), analyzer_.get());
      return std::nullopt;
    }
    if (op->same_as(prim::if_then_else_op())) {
      // Preserve lazy evaluation: a condition shared by every lane may guard
      // a vector arm, but a per-lane or memory-reading condition stays scalar.
      if (node->args.size() != 3 ||
          !UniformCondition(node->args[0].as_or_throw<PrimExpr>())) {
        allowed_ = false;
        return std::nullopt;
      }
      for (const auto &arg : node->args)
        Visit(arg);
      return std::nullopt;
    }
    if (IsElementwisePrimOp(*op) && node->ty_args.empty()) {
      owning_call_ = true;
      for (const auto &arg : node->args) {
        allowed_ &= arg.as<PrimExpr>().has_value();
        Visit(arg);
      }
      return std::nullopt;
    }
    if (!rules.count(*op)) {
      // An unknown pure scalar call may remain uniform, but never infer that
      // it can accept vector operands or that a stateful call can run once.
      auto scalar = GetRef<Expr>(node).as<PrimExpr>();
      allowed_ &= scalar && !UsesLane(Expand(*scalar), lane_) &&
                  SideEffect(GetRef<Expr>(node)) <= CallEffectKind::kPure;
      return std::nullopt;
    }
    owning_call_ = true;
    if ((*op).same_as(tl::cast_op()) || (*op).same_as(tl::clamp())) {
      for (const auto &arg : node->args)
        Visit(arg);
      return std::nullopt;
    }
    ++effectful_calls_;
    auto operands = DecodeCallOperands(node);
    if ((*op).same_as(tl::ptx_cp_async()) ||
        (*op).same_as(tl::ptx_cp_async_bytes())) {
      const auto count = operands[2].as<IntImm>();
      if (!count || count.value()->value <= 0 || count.value()->value > 16) {
        allowed_ = false;
        return std::nullopt;
      }
      int unit = (*op).same_as(tl::ptx_cp_async_bytes())
                     ? 8
                     : operands[0]
                           ->ty.as_or_throw<PtrType>()
                           ->element_type.as_or_throw<PrimType>()
                           .bits();
      int bits = count.value()->value.as<int>().value() * unit;
      AddAccess(operands[0], bits, true);
      AddAccess(operands[1], bits, false);
      for (size_t i = 3; i < node->args.size(); ++i)
        allowed_ &= !UsesLane(node->args[i], lane_) &&
                    SideEffect(node->args[i]) <= CallEffectKind::kPure;
    } else if ((*op).same_as(tl::atomic_add_elem_op())) {
      AddAccess(operands[0],
                operands[0]
                    ->ty.as_or_throw<PtrType>()
                    ->element_type.as_or_throw<PrimType>()
                    .bits(),
                true);
      Visit(operands[1]);
      for (size_t i = 2; i < node->args.size(); ++i)
        allowed_ &= !UsesLane(node->args[i], lane_) &&
                    SideEffect(node->args[i]) <= CallEffectKind::kPure;
    } else {
      allowed_ = false;
    }
    return std::nullopt;
  }

private:
  bool UniformCondition(const PrimExpr &condition) {
    const PrimExpr expanded = Expand(condition);
    return !UsesLane(expanded, lane_) &&
           SafePackingPredicate(expanded, analyzer_.get());
  }

  bool SeparateTransactions(const Expr &lhs, const Expr &rhs, int lhs_bits,
                            int rhs_bits) {
    // Only the same typed view is compared by the caller. Two contiguous
    // transactions may be reordered when their complete byte/bit ranges are
    // disjoint; comparing just lane zero would miss loop-carried dependencies.
    if (!IsContiguous(lhs, lane_, lanes_, lhs_bits, analyzer_.get(), false,
                      begin_) ||
        !IsContiguous(rhs, lane_, lanes_, rhs_bits, analyzer_.get(), false,
                      begin_))
      return false;
    auto left = MemoryAddress(lhs), right = MemoryAddress(rhs);
    if (!left || !right)
      return false;
    PrimExpr a = analyzer_->Simplify(tirx::SubstituteWithDataTypeLegalization(
        left->bit_offset,
        [values = ffi::Map<Var, PrimExpr>{{lane_, begin_}}](const Var &var)
            -> ffi::Optional<PrimExpr> { return values.Get(var); }));
    PrimExpr b = analyzer_->Simplify(tirx::SubstituteWithDataTypeLegalization(
        right->bit_offset,
        [values = ffi::Map<Var, PrimExpr>{{lane_, begin_}}](const Var &var)
            -> ffi::Optional<PrimExpr> { return values.Get(var); }));
    auto safe_end = [&](const PrimExpr &base, int span) {
      auto bound = analyzer_->const_int_bound(base);
      return bound->min_value != sym::ConstIntBound::kNegInf &&
             bound->max_value != sym::ConstIntBound::kPosInf &&
             bound->max_value < sym::ConstIntBound::kPosInf - span;
    };
    int a_span = lhs_bits * lanes_, b_span = rhs_bits * lanes_;
    if (!safe_end(a, a_span) || !safe_end(b, b_span))
      return false;
    return analyzer_->CanProve(a + IntImm(a.ty(), a_span) <= b) ||
           analyzer_->CanProve(b + IntImm(b.ty(), b_span) <= a);
  }

  PrimExpr Expand(const PrimExpr &value) {
    return tirx::SubstituteWithDataTypeLegalization(
        value, [&](const Var &var) -> ffi::Optional<PrimExpr> {
          return bindings_.Get(var);
        });
  }
  void CheckLeadingIndices(const Array<PrimExpr> &indices) {
    if (indices.empty()) {
      allowed_ = false;
      return;
    }
    for (size_t axis = 0; axis + 1 < indices.size(); ++axis) {
      PrimExpr index = Expand(indices[axis]);
      if (!IsStableVectorizeIndex(index)) {
        allowed_ = false;
        return;
      }
      PrimExpr base =
          analyzer_->Simplify(tirx::SubstituteWithDataTypeLegalization(
              index,
              [values = ffi::Map<Var, PrimExpr>{{lane_, begin_}}](
                  const Var &var) -> ffi::Optional<PrimExpr> {
                return values.Get(var);
              }));
      for (int lane = 1; lane < lanes_; ++lane) {
        PrimExpr other = tirx::SubstituteWithDataTypeLegalization(
            index,
            [values =
                 ffi::Map<Var, PrimExpr>{
                     {lane_, begin_ + IntImm(begin_.ty(), lane)}}](
                const Var &var) -> ffi::Optional<PrimExpr> {
              return values.Get(var);
            });
        if (!analyzer_->CanProveEqual(base, other)) {
          allowed_ = false;
          return;
        }
      }
      uniform_leading_indices_.Set(indices[axis], base);
    }
  }
  void AddAccess(Expr expression, int bits, bool write) {
    auto remap = make_object<StmtExprMutator>();
    for (const auto &[var, value] : bindings_)
      remap->VarRemapSet(var, value);
    expression = remap->Mutate(expression).ValueOrUnchanged(expression);
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        expression,
        [&](const ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
          const auto *call = object.as<CallObj>();
          const auto *op = call ? call->op.as<OpObj>() : nullptr;
          if (!op ||
              (op != tl::access_ptr().get() && op != tl::ptr_access().get()))
            return ffi::WalkResult::Advance();
          const size_t extent = op == tl::access_ptr().get() ? 1 : 2;
          for (size_t i = extent; i < call->args.size(); ++i)
            allowed_ &= !UsesLane(call->args[i], lane_) &&
                        SideEffect(call->args[i]) <= CallEffectKind::kPure;
          return ffi::WalkResult::Advance();
        });
    const auto address = MemoryAddress(expression);
    auto tensor =
        address ? address->root.as<TensorVar>() : Optional<TensorVar>();
    if (!address || !tensor || !memory_->Root(*tensor) || bits <= 0) {
      allowed_ = false;
      return;
    }
    accesses_.push_back(
        {expression, *tensor, address->bit_offset, bits, write});
  }
  struct Access {
    Expr expression;
    TensorVar tensor;
    PrimExpr offset;
    int bits;
    bool write;
  };
  Var lane_;
  int lanes_;
  const PipelineMemoryAnalysis *memory_;
  PrimExpr begin_;
  sym::Analyzer analyzer_;
  Map<Var, PrimExpr> bindings_;
  Map<PrimExpr, PrimExpr> uniform_leading_indices_;
  std::vector<Access> accesses_;
  bool allowed_{true}, owning_call_{false};
  int effectful_calls_{0};
  bool nonlocal_store_{false};
};

// Only expressions proved equal at every lane may be made uniform. This
// happens after whole-loop admission and before constructing vector types.
class UniformLeadingIndexRewriter final : public StmtExprMutator {
public:
  explicit UniformLeadingIndexRewriter(Map<PrimExpr, PrimExpr> replacements)
      : replacements_(std::move(replacements)) {}
  using StmtExprMutator::Mutate;
  UnchangedOr<Any> Mutate(AnyView value, InplaceMode mode) final {
    if (auto expr = value.as<PrimExpr>())
      if (auto replacement = replacements_.Get(*expr))
        return *replacement;
    return StmtExprMutator::Mutate(value, mode);
  }

private:
  Map<PrimExpr, PrimExpr> replacements_;
};

// Keep arithmetic reconstruction in the native dtype legalizer. TileLang only
// owns its call contracts, scalar Bind remapping and tensor lane-axis handling.
class PackingRewriter : public tirx::DataTypeLegalizer {
public:
  PackingRewriter(PrimVar lane, int lanes, Optional<Target> target,
                  const sym::Analyzer &analyzer)
      : lane_(lane), lanes_(IntImm(lane.ty(), lanes)), target_(target),
        analyzer_(analyzer->Clone()) {
    VarRemapSet(lane_,
                Ramp(IntImm(lane.ty(), 0), IntImm(lane.ty(), 1), lanes_));
  }
  using DataTypeLegalizer::Mutate;
  using DataTypeLegalizer::Mutate_;
  bool valid() const { return valid_; }

  UnchangedOr<PrimExpr> Mutate_(const SelectObj *node, InplaceMode mode) final {
    PrimExpr condition =
        Mutate(node->condition, mode).ValueOrUnchanged(node->condition);
    PrimExpr on_true =
        Mutate(node->true_value, mode).ValueOrUnchanged(node->true_value);
    PrimExpr on_false =
        Mutate(node->false_value, mode).ValueOrUnchanged(node->false_value);
    if (!valid_)
      return ffi::Unchanged();
    int width = std::max(
        {condition.ty().lanes(), on_true.ty().lanes(), on_false.ty().lanes()});
    // A uniform condition can select vector arms. A vector condition needs
    // vector arms too, even when both original scalar values were uniform.
    auto broadcast = [&](PrimExpr value) {
      return value.ty().IsScalar() && width > 1
                 ? PrimExpr(Broadcast(value, IntImm(lane_.ty(), width)))
                 : value;
    };
    on_true = broadcast(on_true);
    on_false = broadcast(on_false);
    if (on_true.ty() != on_false.ty() || on_true.ty().lanes() != width ||
        (!condition.ty().IsScalar() && condition.ty().lanes() != width)) {
      valid_ = false;
      return ffi::Unchanged();
    }
    return Select(condition, on_true, on_false, node->loc);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::NotObj *node,
                                InplaceMode mode) final {
    PrimExpr value = Mutate(node->a, mode).ValueOrUnchanged(node->a);
    if (!valid_)
      return ffi::Unchanged();
    // Structural mutation preserves the old scalar type. Rebuild the logical
    // operator so a packed operand produces the same number of boolean lanes;
    // otherwise TensorStore would broadcast one ill-typed Not across the
    // vector.
    return prim::Not(value, node->loc);
  }

  UnchangedOr<Expr> Mutate_(const TupleObj *node, InplaceMode mode) final {
    auto fields = node->fields.Map([&](const Expr &field) {
      return Mutate(field, InplaceMode::kDisallow).ValueOrUnchanged(field);
    });
    return tvm::Tuple(fields, node->loc);
  }

  UnchangedOr<Expr> Mutate_(const TupleGetItemObj *node,
                            InplaceMode mode) final {
    Expr value = Mutate(node->tuple, InplaceMode::kDisallow)
                     .ValueOrUnchanged(node->tuple);
    return TupleGetItem(value, node->index, node->loc);
  }

  UnchangedOr<Stmt> Mutate_(const BindObj *node, InplaceMode mode) final {
    if (!node->value->ty.as<TupleType>())
      return DataTypeLegalizer::Mutate_(node, mode);
    Expr value = Mutate(node->value, InplaceMode::kDisallow)
                     .ValueOrUnchanged(node->value);
    Var var = node->var;
    if (!StructuralEqual()(value->ty, var->ty)) {
      var = Var(node->var->name, value->ty, node->var->loc);
      VarRemapSet(node->var, var);
    }
    return Bind(var, value, node->loc);
  }

  UnchangedOr<PrimExpr> Mutate_(const CastObj *node, InplaceMode mode) final {
    PrimExpr value = Mutate(node->value, mode).ValueOrUnchanged(node->value);
    if (!valid_)
      return ffi::Unchanged();
    return Cast(node->ty.as_or_throw<PrimType>().WithLanes(value.ty().lanes()),
                value, node->loc);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *node,
                                InplaceMode mode) final {
    auto indices = RewriteIndices(node->indices);
    if (!valid_)
      return ffi::Unchanged();
    return TensorLoad(node->source, indices, node->loc);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj *node,
                            InplaceMode mode) final {
    auto indices = RewriteIndices(node->indices);
    auto value = Mutate(node->value, mode).ValueOrUnchanged(node->value);
    if (!valid_)
      return ffi::Unchanged();
    int width = TensorLoad(node->dest, indices).ty().lanes();
    if (value.ty().IsScalar() && width > 1)
      value = Broadcast(value, IntImm(lane_.ty(), width));
    return TensorStore(node->dest, indices, value, node->loc);
  }

  UnchangedOr<Expr> Mutate_(const CallObj *node, InplaceMode mode) final {
    Call call = GetRef<Call>(node);
    auto op = node->op.as<Op>();
    auto rules = Op::GetAttrMap(kCallVectorize);
    if (op && op->same_as(prim::assume_op())) {
      PrimExpr condition =
          AllLaneConditions(node->args[0].as_or_throw<PrimExpr>());
      return prim::assume(condition, node->loc);
    }
    if (op && op->same_as(prim::if_then_else_op())) {
      PrimExpr condition = Mutate(node->args[0], InplaceMode::kDisallow)
                               .ValueOrUnchanged(node->args[0])
                               .as_or_throw<PrimExpr>();
      PrimExpr on_true = Mutate(node->args[1], InplaceMode::kDisallow)
                             .ValueOrUnchanged(node->args[1])
                             .as_or_throw<PrimExpr>();
      PrimExpr on_false = Mutate(node->args[2], InplaceMode::kDisallow)
                              .ValueOrUnchanged(node->args[2])
                              .as_or_throw<PrimExpr>();
      int width = std::max(on_true.ty().lanes(), on_false.ty().lanes());
      if (on_true.ty().IsScalar() && width > 1)
        on_true = Broadcast(on_true, IntImm(lane_.ty(), width));
      if (on_false.ty().IsScalar() && width > 1)
        on_false = Broadcast(on_false, IntImm(lane_.ty(), width));
      if (!valid_ || !condition.ty().IsScalar() ||
          on_true.ty() != on_false.ty()) {
        valid_ = false;
        return ffi::Unchanged();
      }
      return Call(on_true.ty(), node->op, {condition, on_true, on_false},
                  node->attrs, {}, node->loc);
    }
    if (op && rules.count(*op)) {
      auto replacement =
          rules[*op](call, lane_, lanes_, target_, [this](Expr value) {
            return Mutate(value, InplaceMode::kDisallow)
                .ValueOrUnchanged(value);
          });
      if (replacement && valid_)
        return *replacement;
      valid_ = false;
      return ffi::Unchanged();
    }
    if (op && IsElementwisePrimOp(*op) && node->ty_args.empty()) {
      Array<Expr> args;
      int width = 1;
      for (const auto &arg : node->args) {
        auto value = Mutate(arg, InplaceMode::kDisallow)
                         .ValueOrUnchanged(arg)
                         .as_or_throw<PrimExpr>();
        width = std::max(width, value.ty().lanes());
        args.push_back(value);
      }
      if (!valid_)
        return ffi::Unchanged();
      for (size_t i = 0; i < args.size(); ++i) {
        auto value = args[i].as_or_throw<PrimExpr>();
        if (value.ty().IsScalar() && width > 1)
          args.Set(i, Broadcast(value, IntImm(lane_.ty(), width)));
      }
      return Call(node->ty.as_or_throw<PrimType>().WithLanes(width), node->op,
                  args, node->attrs, {}, node->loc);
    }
    Expr result = DataTypeLegalizer::Mutate_(node, InplaceMode::kDisallow)
                      .ValueOrUnchanged(call);
    const auto *rewritten = result.as<CallObj>();
    for (const Expr &arg : rewritten->args) {
      if (auto scalar = arg.as<PrimExpr>())
        valid_ &= scalar->ty().IsScalar();
    }
    valid_ &= !UsesLane(result, lane_);
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *node, InplaceMode mode) final {
    if (node->op->name != "tl.attr")
      return DataTypeLegalizer::Mutate_(node, mode);
    auto fields = ReadAnnotations(node);
    auto key = fields.Get("key");
    if (!key || !key->as<String>() ||
        key->cast<String>() != "tl.assume_requires_runtime_check")
      return DataTypeLegalizer::Mutate_(node, mode);
    PrimExpr condition = AllLaneConditions(fields.at("node").cast<PrimExpr>());
    fields.Set("node", condition);
    auto encoded = EncodeAnnotations(fields);
    SeqStmt body = Mutate(node->body, mode).ValueOrUnchanged(node->body);
    return RegionStmt(node->op, {encoded.payload}, node->body_params,
                      DictAttrs({{"schema_version", 1},
                                 {"annotation_schema", encoded.schema}}),
                      body, node->result_vars, node->loc);
  }

private:
  PrimExpr AllLaneConditions(const PrimExpr &condition) {
    PrimExpr result = IntImm(PrimType::Bool(), 1);
    const int lanes = lanes_.as_or_throw<IntImm>()->value.as<int>().value();
    for (int i = 0; i < lanes; ++i) {
      PrimExpr scalar = tirx::SubstituteWithDataTypeLegalization(
          RenewExpressionDefs(condition),
          [values = ffi::Map<Var, PrimExpr>{{lane_, IntImm(lane_.ty(), i)}}](
              const Var &var) -> ffi::Optional<PrimExpr> {
            return values.Get(var);
          });
      scalar = Mutate(scalar, InplaceMode::kDisallow).ValueOrUnchanged(scalar);
      if (scalar.ty() != PrimType::Bool()) {
        valid_ = false;
        return IntImm(PrimType::Bool(), 1);
      }
      // Do not simplify a runtime-check predicate using the assumption that
      // check is intended to validate. Every original lane remains explicit.
      result = And(result, scalar, condition->loc);
    }
    return result;
  }
  Array<PrimExpr> RewriteIndices(const Array<PrimExpr> &original) {
    Array<PrimExpr> indices;
    for (const auto &index : original) {
      PrimExpr value =
          Mutate(index, InplaceMode::kDisallow).ValueOrUnchanged(index);
      if (valid_) {
        value = analyzer_->Simplify(value);
        if (const auto *broadcast = value.as<BroadcastObj>())
          value = broadcast->value;
      }
      indices.push_back(value);
    }
    return indices;
  }
  PrimVar lane_;
  PrimExpr lanes_;
  Optional<Target> target_;
  sym::Analyzer analyzer_;
  bool valid_{true};
};

class IntrinsicLoopPacker : public StmtExprMutatorWithAnalyzer {
public:
  explicit IntrinsicLoopPacker(const tirx::Function &function)
      : memory_(function),
        target_(function->GetAttr<Target>(tvm::attr::kTarget)) {
    MarkTensorParamShapes(function);
  }
  using StmtExprMutatorWithAnalyzer::Mutate_;
  UnchangedOr<Stmt> Mutate_(const ForObj *node, InplaceMode mode) final {
    auto analyzer_ = GetAnalyzer();
    if (node->kind != ForKind::kVectorized || !IsOne(node->step))
      return StmtExprMutatorWithAnalyzer::Mutate_(node, mode);
    PrimType index_type = node->var.ty();
    // The Ramp lane introduced below has the native signed 32/64-bit index
    // contract. A modular narrow/unsigned extent must never turn an empty
    // scalar loop into an executed vector transaction.
    if (!index_type.IsScalar() || !index_type.MatchesCode(kDLInt) ||
        (index_type.bits() != 32 && index_type.bits() != 64) ||
        !IsStableVectorizeIndex(node->end) ||
        !analyzer_->CanProve(node->end > node->begin))
      return StmtExprMutatorWithAnalyzer::Mutate_(node, mode);
    auto extent = analyzer_->Simplify(node->end - node->begin).as<IntImm>();
    if (!extent || extent.value()->value <= 1 || extent.value()->value > 128 ||
        !IsStableVectorizeIndex(node->begin))
      return StmtExprMutatorWithAnalyzer::Mutate_(node, mode);
    int width = extent.value()->value.as<int>().value();
    auto plan = make_object<PackingAnalysis>(node->var, width, &memory_,
                                             node->begin, analyzer_);
    if (!plan->Run(node->body))
      return StmtExprMutatorWithAnalyzer::Mutate_(node, mode);
    // Effects were inspected before substitution can simplify index algebra.
    PrimVar lane(node->var->name, node->var.ty());
    Stmt body =
        make_object<UniformLeadingIndexRewriter>(plan->UniformLeadingIndices())
            ->Mutate(node->body, InplaceMode::kDisallow)
            .ValueOrUnchanged(node->body);
    body = tl::SubstituteWithDataTypeLegalization(
        body,
        [values = ffi::Map<Var, PrimExpr>{{node->var, node->begin + lane}}](
            const Var &var) -> ffi::Optional<PrimExpr> {
          return values.Get(var);
        });
    auto rewriter =
        make_object<PackingRewriter>(lane, width, target_, analyzer_);
    Stmt rewritten =
        rewriter->Mutate(body, InplaceMode::kDisallow).ValueOrUnchanged(body);
    if (!rewriter->valid())
      return StmtExprMutatorWithAnalyzer::Mutate_(node, mode);
    return rewritten;
  }

private:
  PipelineMemoryAnalysis memory_;
  Optional<Target> target_;
};
} // namespace

tvm::transform::Pass VectorizeLoop(bool enable_vectorize) {
  auto native = tirx::transform::VectorizeLoop(enable_vectorize);
  return tirx::transform::CreateFunctionPass(
      [native, enable_vectorize](tirx::Function function, IRModule,
                                 tvm::transform::PassContext context) {
        VectorizeMemoryScope memory_scope(function);
        if (enable_vectorize) {
          Stmt packed = make_object<IntrinsicLoopPacker>(function)
                            ->Mutate(function->body, InplaceMode::kDisallow)
                            .ValueOrUnchanged(function->body);
          function.CopyOnWrite()->body = SeqStmt(packed);
        }
        GlobalVar global("__tl_vectorize_function");
        IRModule module({{global, function}});
        return native(module, context)
            ->Lookup(global)
            .as_or_throw<tirx::Function>();
      },
      0, "tl.VectorizeLoop");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.clamp").set_attr(kCallVectorize, FCallVectorize(VectorizeClamp));
  OpDef("tl.ptx_cp_async")
      .set_attr(kCallVectorize, FCallVectorize(VectorizeCPAsync));
  OpDef("tl.ptx_cp_async_bytes")
      .set_attr(kCallVectorize, FCallVectorize(VectorizeCPAsync));
  OpDef("tl.atomic_add_elem_op")
      .set_attr(kCallVectorize, FCallVectorize(VectorizeAtomicAdd));
  reflection::GlobalDef().def("tl.transform.VectorizeLoop", [](bool enable) {
    return tvm::tl::VectorizeLoop(enable);
  });
}
} // namespace tvm::tl
