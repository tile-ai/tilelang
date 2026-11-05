#include "ir/loop_math.h"
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
 * \file thread_storage_sync.cc
 * \brief Plan barriers over native tensor storage and lexical thread domains.
 */
#include "common/collector.h"
#include "common/constr_visitor.h"
#include "ir/analyzer.h"
#include "ir/tensor_utils.h"
#include "op/builtin.h"
#include "thread_storage_sync.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/sym/int_set.h>
#include <tvm/target/target.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace prim;
using namespace tirx;
using Threads = std::vector<NativeThreadBinding>;

String OpName(const CallObj *call) {
  auto op = call->op.as<Op>();
  return op ? op.value()->name : String();
}

bool IsShared(const String &scope) {
  return scope == "shared" || scope == "shared.dyn";
}

bool IsStorageSync(const CallObj *call) {
  return call && (call->op.same_as(gpu_storage_sync_op()) ||
                  OpName(call) == "tl.storage_sync");
}

bool IsFullSharedStorageSync(const CallObj *call) {
  return call && call->op.same_as(gpu_storage_sync_op()) &&
         call->args.size() == 1 &&
         IsShared(call->args[0].as_or_throw<StringImm>()->value);
}

Stmt MakeSync(const String &scope, const Location &loc = UnknownLoc()) {
  return Evaluate(Call(PrimType::Void(), gpu_storage_sync_op(),
                       {StringImm(scope)}, {}, {}, loc),
                  loc);
}

Map<Any, Any> AttrValues(const RegionStmtObj *region) {
  return DecodeAnnotations(
             region->args[0].as_or_throw<tvm::Tuple>(),
             region->attrs.GetAttr<Array<Any>>("annotation_schema").value())
      .cast<Map<Any, Any>>();
}

const NativeThreadBinding *FindThread(const Threads &threads,
                                      const String &tag) {
  for (auto it = threads.rbegin(); it != threads.rend(); ++it)
    if (it->tag == tag)
      return &*it;
  return nullptr;
}

PrimExpr LinearThreadId(const Threads &threads) {
  PrimExpr result = IntImm(PrimType::Int(64), 0);
  PrimExpr stride = IntImm(PrimType::Int(64), 1);
  for (const char *tag : {"threadIdx.x", "threadIdx.y", "threadIdx.z"}) {
    if (const auto *thread = FindThread(threads, tag)) {
      result =
          result +
          Cast(PrimType::Int(64), thread->var - thread->dom->begin) * stride;
      stride = stride * Cast(PrimType::Int(64), thread->dom->extent);
    }
  }
  return result;
}

// Record even an unentered loop body in the rewrite stream, without analyzing
// its memory effects. Shared nodes there must not shift later occurrences.
class StatementOccurrenceRecorder : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  explicit StatementOccurrenceRecorder(
      std::vector<const StmtObj *> *occurrences)
      : occurrences_(occurrences) {}

  Optional<VisitInterrupt> Visit(AnyView value) final {
    if (const auto *statement = value.as<StmtObj>())
      occurrences_->push_back(statement);
    return StmtExprVisitor::Visit(value);
  }

private:
  std::vector<const StmtObj *> *occurrences_;
};

// All native statements retain a SeqStmt body. Rewrite the selected lexical
// occurrence, including when the same immutable statement has several uses.
class StatementRewriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  String scope;
  std::unordered_set<size_t> insert_before;
  std::vector<const StmtObj *> occurrences;

  SeqStmt Rewrite(const SeqStmt &body) {
    auto result = Mutate(body).ValueOrUnchanged(body);
    TVM_FFI_CHECK_EQ(next_, occurrences.size(), InternalError)
        << "Barrier insertion did not consume every statement occurrence";
    return result;
  }

  UnchangedOr<Any> Mutate(AnyView value, InplaceMode mode) final {
    const auto *stmt = value.as<StmtObj>();
    size_t occurrence = next_;
    if (stmt) {
      TVM_FFI_CHECK_LT(next_, occurrences.size(), InternalError)
          << "Barrier insertion found an unrecorded statement occurrence";
      TVM_FFI_CHECK_EQ(occurrences[next_++], stmt, InternalError)
          << "Barrier insertion changed lexical statement order";
    }
    auto result = StmtExprMutator::Mutate(value, mode);
    if (stmt && insert_before.count(occurrence)) {
      Stmt updated = result.IsUnchanged()
                         ? GetRef<Stmt>(stmt)
                         : std::move(result).ValueUnchecked().cast<Stmt>();
      return Any(SeqStmt({MakeSync(scope, stmt->loc), updated}, stmt->loc));
    }
    return result;
  }

private:
  size_t next_{0};
};

// cp.async.wait_group does not provide inter-thread visibility by itself. An
// earlier barrier cannot cover its completion. Keep the fence in the same
// lexical sequence as the wait, so a surrounding conditional is not confused
// with an unconditional wait or with an unconditional existing barrier.
class SyncAfterWaitGroup : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit SyncAfterWaitGroup(String scope) : scope_(std::move(scope)) {}

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    Array<Stmt> visited;
    for (const Stmt &stmt : op->seq)
      visited.push_back(Mutate(stmt, mode).ValueOrUnchanged(stmt));
    Array<Stmt> result;
    for (size_t i = 0; i < visited.size(); ++i) {
      result.push_back(visited[i]);
      const auto *evaluate = visited[i].as<EvaluateObj>();
      const auto *call = evaluate ? evaluate->value.as<CallObj>() : nullptr;
      if (!call || (OpName(call) != "tl.ptx_wait_group" &&
                    OpName(call) != "ptx.cp_async_wait_group"))
        continue;
      const auto *next =
          i + 1 < visited.size() ? visited[i + 1].as<EvaluateObj>() : nullptr;
      const auto *barrier = next ? next->value.as<CallObj>() : nullptr;
      if (!(IsStorageSync(barrier) && barrier->args.size() == 1 &&
            barrier->args[0].as_or_throw<StringImm>()->value == scope_))
        result.push_back(MakeSync(scope_, visited[i]->loc));
    }
    return SeqStmt(result, op->loc);
  }

private:
  String scope_;
};

struct ConditionProperty {
  bool runtime{false};
  bool block_uniform{true};
  bool requires_hoist{false};
  void Merge(const ConditionProperty &other) {
    runtime |= other.runtime;
    block_uniform &= other.block_uniform;
    requires_hoist |= other.requires_hoist;
  }
};
using ConditionProperties =
    std::unordered_map<const VarObj *, ConditionProperty>;

class ConditionChecker : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  ConditionChecker(const Threads &threads,
                   const ConditionProperties &properties)
      : threads_(threads), properties_(properties) {}
  ConditionProperty property;

  Optional<VisitInterrupt> Visit_(const VarObj *op) final {
    for (const auto &thread : threads_) {
      if (thread.var.get() == op) {
        if (thread.IsBlockThread())
          property.block_uniform = false;
        else
          property.runtime = true;
        return std::nullopt;
      }
    }
    if (auto it = properties_.find(op); it != properties_.end())
      property.Merge(it->second);
    else
      property.runtime = true;
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const TensorLoadObj *op) final {
    property.runtime = true;
    // Match the source policy: addresses determine uniformity; a local
    // fragment may contain the same block-uniform value in every thread.
    for (const PrimExpr &index : op->indices)
      Visit(index);
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    if (OpName(op) == "tl.annotation")
      return std::nullopt;
    if (SideEffect(GetRef<Expr>(op)) > CallEffectKind::kPure)
      property.runtime = true;
    return StmtExprVisitor::Visit_(op);
  }

private:
  const Threads &threads_;
  const ConditionProperties &properties_;
};

ConditionProperty GetConditionProperty(const Expr &expr, const Threads &threads,
                                       const ConditionProperties &properties) {
  auto checker = make_object<ConditionChecker>(threads, properties);
  checker->Visit(expr);
  return checker->property;
}

// Agreement is proved outside the condition's guard. Entering the guard first
// would make both copies true and allow every divergent barrier vacuously.
bool UniformCondition(const PrimExpr &condition, const ConstrSet &premises,
                      const Threads &threads, int warp_size = 0) {
  Map<Var, PrimExpr> first, second;
  for (const auto &thread : threads) {
    if (!thread.IsBlockThread())
      continue;
    first.Set(thread.var, PrimVar(thread.var->name + "<T1>", thread.var.ty()));
    second.Set(thread.var, PrimVar(thread.var->name + "<T2>", thread.var.ty()));
  }
  if (first.empty())
    return true;
  auto left = premises.RenameFrom("<T1>", first, std::nullopt, false);
  auto right = premises.RenameFrom("<T2>", second, std::nullopt, false);
  auto analyzer = CreateTileLangAnalyzer();
  auto context =
      left.ToConstraints().Merge(right.ToConstraints()).Populate(analyzer);
  auto lhs = SubstituteConstraintExpr(condition, first);
  auto rhs = SubstituteConstraintExpr(condition, second);
  PrimExpr agreement = Or(And(lhs, rhs), And(Not(lhs), Not(rhs)));
  if (warp_size) {
    auto linear = LinearThreadId(threads);
    auto first_warp = FloorDiv(SubstituteConstraintExpr(linear, first),
                               IntImm(linear.ty(), warp_size));
    auto second_warp = FloorDiv(SubstituteConstraintExpr(linear, second),
                                IntImm(linear.ty(), warp_size));
    agreement = Or(first_warp != second_warp, agreement);
  }
  return analyzer->CanProve(agreement);
}

// Resolve definitions, not type identity, into an allocation and a bit offset.
// Two views can have different layouts/dtypes and still share this address.
// An unresolved pointer is never manufactured into a fresh independent root.
struct StorageAddress {
  Optional<Var> root;
  Optional<PrimExpr> bit_offset;
  String scope;
};

Optional<PrimExpr> WidenIndex(const PrimExpr &index) {
  auto type = index.ty();
  if (!type.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) ||
      type.IsScalableVector() ||
      (type.MatchesCode(DLDataTypeCode::kDLUInt) && type.bits() >= 64))
    return std::nullopt;
  return Cast(PrimType::Int(64, type.lanes()), index);
}

// Express one arbitrary vector lane with scalar IR of the original element
// types. A cast must remain inside this expression: narrowing just the extrema
// of a Ramp can lose wrapped lanes and incorrectly prove two accesses disjoint.
// Unsupported vector nodes retain a possible conflict instead of being treated
// as scalar points by IntSet::Vector.
std::optional<PrimExpr> ScalarLaneValue(const PrimExpr &value,
                                        const PrimVar &lane, int lanes) {
  if (value.ty().IsScalar())
    return value;
  if (!value.ty().IsFixedLengthVector() || value.ty().lanes() != lanes)
    return std::nullopt;
  if (const auto *ramp = value.as<RampObj>()) {
    if (!ramp->base.ty().IsScalar() || !ramp->stride.ty().IsScalar())
      return std::nullopt;
    return Add(ramp->base,
               Mul(Cast(ramp->stride.ty(), lane), ramp->stride, ramp->loc),
               ramp->loc);
  }
  if (const auto *broadcast = value.as<BroadcastObj>())
    return broadcast->value;
  if (const auto *cast = value.as<CastObj>()) {
    auto input = ScalarLaneValue(cast->value, lane, lanes);
    if (!input)
      return std::nullopt;
    return Cast(value.ty().WithLanes(1), *input, cast->loc);
  }
#define TL_SCALAR_LANE_BINARY(Name)                                            \
  if (const auto *op = value.as<Name##Obj>()) {                                \
    auto a = ScalarLaneValue(op->a, lane, lanes);                              \
    auto b = ScalarLaneValue(op->b, lane, lanes);                              \
    if (!a || !b)                                                              \
      return std::nullopt;                                                     \
    return Name(*a, *b, op->loc);                                              \
  }
  TL_SCALAR_LANE_BINARY(Add)
  TL_SCALAR_LANE_BINARY(Sub)
  TL_SCALAR_LANE_BINARY(Mul)
  TL_SCALAR_LANE_BINARY(Div)
  TL_SCALAR_LANE_BINARY(Mod)
  TL_SCALAR_LANE_BINARY(FloorDiv)
  TL_SCALAR_LANE_BINARY(FloorMod)
  TL_SCALAR_LANE_BINARY(Min)
  TL_SCALAR_LANE_BINARY(Max)
  TL_SCALAR_LANE_BINARY(EQ)
  TL_SCALAR_LANE_BINARY(NE)
  TL_SCALAR_LANE_BINARY(LT)
  TL_SCALAR_LANE_BINARY(LE)
  TL_SCALAR_LANE_BINARY(GT)
  TL_SCALAR_LANE_BINARY(GE)
  TL_SCALAR_LANE_BINARY(And)
  TL_SCALAR_LANE_BINARY(Or)
  TL_SCALAR_LANE_BINARY(LShift)
  TL_SCALAR_LANE_BINARY(RShift)
  TL_SCALAR_LANE_BINARY(BitwiseAnd)
  TL_SCALAR_LANE_BINARY(BitwiseOr)
  TL_SCALAR_LANE_BINARY(BitwiseXor)
#undef TL_SCALAR_LANE_BINARY
  if (const auto *op = value.as<NotObj>()) {
    auto input = ScalarLaneValue(op->a, lane, lanes);
    if (input)
      return Not(*input, op->loc);
    return std::nullopt;
  }
  if (const auto *op = value.as<BitwiseNotObj>()) {
    auto input = ScalarLaneValue(op->a, lane, lanes);
    if (input)
      return BitwiseNot(*input, op->loc);
    return std::nullopt;
  }
  if (const auto *op = value.as<SelectObj>()) {
    auto condition = ScalarLaneValue(op->condition, lane, lanes);
    auto on_true = ScalarLaneValue(op->true_value, lane, lanes);
    auto on_false = ScalarLaneValue(op->false_value, lane, lanes);
    if (condition && on_true && on_false)
      return Select(*condition, *on_true, *on_false, op->loc);
    return std::nullopt;
  }
  if (const auto *op = value.as<CallObj>();
      op && OpName(op) == "prim.if_then_else" && op->args.size() == 3) {
    auto condition =
        ScalarLaneValue(op->args[0].as_or_throw<PrimExpr>(), lane, lanes);
    auto on_true =
        ScalarLaneValue(op->args[1].as_or_throw<PrimExpr>(), lane, lanes);
    auto on_false =
        ScalarLaneValue(op->args[2].as_or_throw<PrimExpr>(), lane, lanes);
    if (condition && on_true && on_false)
      return tvm::if_then_else(*condition, *on_true, *on_false, op->loc);
  }
  return std::nullopt;
}

// The scalar proof uses mathematical arithmetic for narrow/unsigned lanes.
// Establish that each intermediate is representable before using that proof.
// Bounds of the original operation already account for wrapping, so they cannot
// establish this: int16(lane) * 32767 has a type-bounded result even when it
// wraps. Bound the operation on widened operands instead. Signed int32/int64
// expressions keep the target's existing index contract.
class LaneArithmeticGuard : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  explicit LaneArithmeticGuard(sym::Analyzer analyzer)
      : analyzer_(std::move(analyzer)) {}

  bool safe{true};

  Optional<VisitInterrupt> Visit(AnyView value) final {
    if (!safe)
      return std::nullopt;
    auto result = StmtExprVisitor::Visit(value);
    auto expression = value.as<PrimExpr>();
    if (!safe || !expression)
      return result;
    const PrimExpr &expr = *expression;
    if (const auto *cast = expr.as<CastObj>()) {
      PrimType from = cast->value.ty();
      PrimType to = expr.ty();
      if (from.IsScalar() && to.IsScalar() &&
          from.MatchesCode(kDLInt, kDLUInt) &&
          to.MatchesCode(kDLInt, kDLUInt)) {
        // Native Ramp is wide, so an explicit per-lane narrowing carries the
        // original finite-width semantics. The scalar solver may discard that
        // cast; admit it only when its input range proves it lossless.
        if ((from.code() == to.code() && to.bits() >= from.bits()) ||
            (from.MatchesCode(kDLUInt) && to.MatchesCode(kDLInt) &&
             to.bits() > from.bits()))
          return result;
        auto bounds = analyzer_->const_int_bound(cast->value);
        int64_t minimum = to.MatchesCode(kDLUInt) ? 0
                          : to.bits() >= 64
                              ? std::numeric_limits<int64_t>::min()
                              : -(int64_t{1} << (to.bits() - 1));
        // UInt63's upper endpoint is INT64_MAX. Compute the endpoint in
        // unsigned arithmetic so forming it does not overflow signed int64.
        int64_t maximum =
            to.bits() >= 64
                ? std::numeric_limits<int64_t>::max()
                : static_cast<int64_t>(
                      (uint64_t{1}
                       << (to.bits() - (to.MatchesCode(kDLUInt) ? 0 : 1))) -
                      uint64_t{1});
        safe = bounds->min_value != sym::ConstIntBound::kNegInf &&
               bounds->max_value != sym::ConstIntBound::kPosInf &&
               bounds->min_value >= minimum && bounds->max_value <= maximum;
      }
      return result;
    }
    bool arithmetic = expr.as<AddObj>() || expr.as<SubObj>() ||
                      expr.as<MulObj>() || expr.as<LShiftObj>() ||
                      expr.as<DivObj>() || expr.as<FloorDivObj>();
    PrimType type = expr.ty();
    if (!arithmetic || !type.IsScalar() || !type.MatchesCode(kDLInt, kDLUInt) ||
        (type.MatchesCode(kDLInt) && type.bits() >= 32))
      return result;
    if (type.bits() >= 63) {
      safe = false;
      return result;
    }
    int64_t minimum =
        type.MatchesCode(kDLUInt) ? 0 : -(int64_t{1} << (type.bits() - 1));
    int64_t maximum =
        (int64_t{1} << (type.bits() - (type.MatchesCode(kDLUInt) ? 0 : 1))) - 1;
    std::optional<PrimExpr> mathematical;
#define TL_WIDEN_LANE_ARITHMETIC(Name)                                         \
  if (const auto *op = expr.as<Name##Obj>()) {                                 \
    mathematical = Name(Cast(PrimType::Int(64), op->a),                        \
                        Cast(PrimType::Int(64), op->b), op->loc);              \
  }
    TL_WIDEN_LANE_ARITHMETIC(Add)
    TL_WIDEN_LANE_ARITHMETIC(Sub)
    TL_WIDEN_LANE_ARITHMETIC(Mul)
    TL_WIDEN_LANE_ARITHMETIC(LShift)
    TL_WIDEN_LANE_ARITHMETIC(Div)
    TL_WIDEN_LANE_ARITHMETIC(FloorDiv)
#undef TL_WIDEN_LANE_ARITHMETIC
    TVM_FFI_ICHECK(mathematical.has_value());
    auto bounds = analyzer_->const_int_bound(*mathematical);
    // A uint32 product can exceed the widened signed range. Infinity sentinels
    // do not establish representability; wider unsigned operations exit above.
    safe = bounds->min_value != sym::ConstIntBound::kNegInf &&
           bounds->max_value != sym::ConstIntBound::kPosInf &&
           bounds->min_value >= minimum && bounds->max_value <= maximum;
    return result;
  }

private:
  sym::Analyzer analyzer_;
};

class StorageAddresses : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  explicit StorageAddresses(const tirx::Function &function) {
    if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      Visit(function->body);
  }

  Optional<VisitInterrupt> Visit_(const BindObj *op) final {
    if (op->var->ty.as<TensorTypeObj>() || op->var->ty.as<PtrTypeObj>()) {
      definitions_.Set(op->var, op->value);
      const auto *call = op->value.as<CallObj>();
      if (call && call->op.same_as(alloc_tensor_op()))
        allocations_.insert(op->var.get());
    }
    return StmtExprVisitor::Visit_(op);
  }
  Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    return OpName(op) == "tl.annotation" ? std::nullopt
                                         : StmtExprVisitor::Visit_(op);
  }

  bool Independent(const StorageAddress &left,
                   const StorageAddress &right) const {
    if (!left.root || !right.root ||
        left.root.value().same_as(right.root.value()))
      return false;
    // shared.dyn denotes one runtime allocation, including separate typed
    // decl_tensor definitions generated by memory reuse.
    if (left.scope == "shared.dyn" && right.scope == "shared.dyn")
      return false;
    return allocations_.count(left.root.value().get()) ||
           allocations_.count(right.root.value().get());
  }

  StorageAddress Resolve(const Expr &expr) const {
    std::unordered_set<const VarObj *> active;
    return Resolve(expr, &active);
  }

private:
  StorageAddress Resolve(const Expr &expr,
                         std::unordered_set<const VarObj *> *active) const {
    if (auto tensor = expr.as<TensorVar>()) {
      StorageAddress result{tensor.value(), IntImm(PrimType::Int(64), 0),
                            tensor.value().scope().name()};
      auto definition = definitions_.Get(tensor.value());
      if (!definition || allocations_.count(tensor.value().get()))
        return result;
      if (!active->insert(tensor.value().get()).second)
        return {std::nullopt, std::nullopt, result.scope};
      auto resolved = Resolve(definition.value(), active);
      active->erase(tensor.value().get());
      resolved.scope = result.scope;
      return resolved;
    }
    if (auto variable = expr.as<Var>(); variable && expr->ty.as<PtrTypeObj>()) {
      String scope = expr->ty.as_or_throw<PtrType>()->scope.name();
      auto definition = definitions_.Get(variable.value());
      if (!definition)
        return {variable.value(), IntImm(PrimType::Int(64), 0), scope};
      if (!active->insert(variable.value().get()).second)
        return {std::nullopt, std::nullopt, scope};
      auto result = Resolve(definition.value(), active);
      active->erase(variable.value().get());
      result.scope = scope;
      return result;
    }
    if (const auto *load = expr.as<TensorLoadObj>()) {
      auto tensor = load->source.as_or_throw<TensorVar>();
      auto result = Resolve(tensor, active);
      if (result.bit_offset) {
        auto offsets = tensor.ty()->ElemOffset(load->indices);
        if (offsets.size() == 1) {
          if (auto offset = WidenIndex(offsets[0])) {
            int bits = tensor.dtype().bits() * tensor.dtype().lanes();
            result.bit_offset =
                result.bit_offset.value() + offset.value() * bits;
            return result;
          }
        }
        result.bit_offset = std::nullopt;
      }
      return result;
    }
    if (const auto *call = expr.as<CallObj>()) {
      String name = OpName(call);
      if (name == "tirx.decl_tensor" || name == "tirx.tensor_data_ptr" ||
          name == "tirx.address_of" || name == "tirx.reinterpret" ||
          name == "tl.access_ptr")
        return Resolve(call->args[0], active);
      if (name == "tirx.ptr_byte_offset" ||
          name == "tirx.handle_add_byte_offset" || name == "tl.ptr_access") {
        auto result = Resolve(call->args[0], active);
        auto offset = WidenIndex(call->args[1].as_or_throw<PrimExpr>());
        int bits = 8;
        if (name == "tl.ptr_access") {
          auto element = call->args[0]
                             ->ty.as_or_throw<PtrType>()
                             ->element_type.as<PrimType>();
          if (!element)
            offset = std::nullopt;
          else
            bits = element.value().bits() * element.value().lanes();
        }
        if (result.bit_offset && offset)
          result.bit_offset = result.bit_offset.value() + offset.value() * bits;
        else
          result.bit_offset = std::nullopt;
        return result;
      }
    }
    String scope;
    if (auto type = expr->ty.as<PtrType>())
      scope = type.value()->scope.name();
    return {std::nullopt, std::nullopt, scope};
  }
  Map<Var, Expr> definitions_;
  std::unordered_set<const VarObj *> allocations_;
};

struct AccessEntry {
  enum Kind { kRead, kWrite, kSync } kind;
  Threads threads;
  ConstrSet constraints;
  StorageAddress address;
  Optional<PrimExpr> width_bits;
  std::vector<Var> relaxed_vars;
  bool async_copy{false};
  bool atomic{false};
};

struct StmtEntry {
  size_t occurrence{};
  std::vector<AccessEntry> access;
};

class ThreadSyncPlanner : public ConstrVisitor {
public:
  using ConstrVisitor::Visit;
  explicit ThreadSyncPlanner(const tirx::Function &function, String scope,
                             int warp_size)
      : addresses_(make_object<StorageAddresses>(function)),
        scope_name_(std::move(scope)), warp_size_(warp_size) {
    scopes_.emplace_back();
  }
  std::unordered_set<size_t> syncs;
  std::vector<const StmtObj *> occurrences;

  Result Visit(AnyView value) final {
    size_t previous = current_occurrence_;
    if (const auto *statement = value.as<StmtObj>()) {
      current_occurrence_ = occurrences.size();
      occurrences.push_back(statement);
    }
    auto result = ConstrVisitor::Visit(value);
    current_occurrence_ = previous;
    return result;
  }

  Result Visit_(const TensorLoadObj *op) final {
    Record(GetRef<TensorLoad>(op), TensorWidth(op->source), AccessEntry::kRead);
    for (const PrimExpr &index : op->indices)
      Visit(index);
    return std::nullopt;
  }

  Result Visit_(const TensorStoreObj *op) final {
    Start(op);
    auto tensor = op->dest.as_or_throw<TensorVar>();
    auto base = TensorLoad(tensor, op->indices, op->loc);
    Record(base, TensorWidth(tensor), AccessEntry::kWrite);
    Visit(op->value);
    for (const PrimExpr &index : op->indices)
      Visit(index);
    Finish();
    return std::nullopt;
  }

  Result Visit_(const EvaluateObj *op) final {
    Start(op);
    Visit(op->value);
    Finish();
    return std::nullopt;
  }

  Result Visit_(const BindObj *op) final {
    Start(op);
    Visit(op->value);
    Finish();
    if (op->value.as<PrimExpr>())
      properties_.insert_or_assign(
          op->var.get(),
          GetConditionProperty(op->value, threads_, properties_));
    return std::nullopt;
  }

  Result Visit_(const AssertStmtObj *op) final {
    Start(op);
    Visit(op->condition);
    Finish();
    return std::nullopt;
  }

  Result Visit_(const ReturnObj *op) final {
    Start(op);
    auto result = ConstrVisitor::Visit_(op);
    Finish();
    return result;
  }

  Result Visit_(const SeqStmtObj *op) final {
    auto saved = properties_;
    auto result = ConstrVisitor::Visit_(op);
    properties_ = std::move(saved);
    return result;
  }

  Result Visit_(const RegionStmtObj *op) final {
    auto saved = properties_;
    size_t old_threads = threads_.size();
    auto bindings = GetRegionThreadBindings(op);
    threads_.insert(threads_.end(), bindings.begin(), bindings.end());
    bool was_device = in_device_;
    bool entry = !bindings.empty() && !was_device;
    if (entry) {
      in_device_ = true;
      scopes_.emplace_back();
    }
    auto result = ConstrVisitor::Visit_(op);
    if (entry) {
      Summarize(scopes_.back(), nullptr);
      scopes_.pop_back();
    }
    in_device_ = was_device;
    threads_.erase(threads_.begin() + old_threads, threads_.end());
    properties_ = std::move(saved);
    return result;
  }

  Result Visit_(const ForObj *op) final {
    size_t occurrence = Start(op);
    Visit(op->begin);
    Visit(op->end);
    Visit(op->step);
    auto header = std::move(current_.access);
    collecting_ = false;
    auto analyzer = CreateTileLangAnalyzer();
    auto context = GetConstrSet().Populate(analyzer);
    if (SideEffect(op->begin) <= CallEffectKind::kPure &&
        SideEffect(op->end) <= CallEffectKind::kPure &&
        analyzer->CanProve(op->begin >= op->end)) {
      // Native For always tests var < end, including a negative step. Its
      // header still evaluates, but an unentered body has no memory effects.
      if (!header.empty())
        scopes_.back().push_back({occurrence, std::move(header)});
      auto skipped = make_object<StatementOccurrenceRecorder>(&occurrences);
      skipped->Visit(op->body);
      return std::nullopt;
    }
    scopes_.emplace_back();
    {
      auto scope = Scope();
      if (CanUseLoopBounds(op->begin, op->end, op->step, analyzer.get())) {
        constr_stack_.emplace_back(op->var, Range(op->begin, op->end));
        constr_stack_.emplace_back(op->end > op->begin);
      } else {
        // Negative, zero or wrapping updates do not justify a lower bound.
        // Keep the definition for per-iteration renaming and only record
        // predicates that hold under the native loop's actual execution.
        constr_stack_.emplace_back(op->var, op->var.as_or_throw<PrimExpr>());
        if (SideEffect(op->end) <= CallEffectKind::kPure)
          constr_stack_.emplace_back(op->var < op->end);
        if (SideEffect(op->begin) <= CallEffectKind::kPure &&
            SideEffect(op->end) <= CallEffectKind::kPure)
          constr_stack_.emplace_back(op->begin < op->end);
      }
      Visit(op->body);
    }
    auto access = Summarize(scopes_.back(), op);
    scopes_.pop_back();
    for (auto &entry : access)
      if (entry.kind != AccessEntry::kSync)
        entry.relaxed_vars.push_back(op->var);
    header.insert(header.end(), access.begin(), access.end());
    if (!header.empty())
      scopes_.back().push_back({occurrence, std::move(header)});
    return std::nullopt;
  }

  Result Visit_(const IfObj *op) final;
  Result Visit_(const WhileObj *op) final;
  Result Visit_(const CallObj *op) final;

private:
  static Optional<PrimExpr> TensorWidth(const Expr &tensor) {
    auto type = tensor->ty.as_or_throw<TensorType>()->dtype;
    if (type.IsScalableVector())
      return std::nullopt;
    return IntImm(PrimType::Int(64), type.bits() * type.lanes());
  }

  size_t Start(const Object *stmt) {
    TVM_FFI_ICHECK(current_.access.empty());
    TVM_FFI_ICHECK_LT(current_occurrence_, occurrences.size());
    TVM_FFI_ICHECK_EQ(occurrences[current_occurrence_], stmt);
    current_.occurrence = current_occurrence_;
    collecting_ = true;
    return current_occurrence_;
  }
  void Finish() {
    if (!current_.access.empty())
      scopes_.back().push_back(std::move(current_));
    current_ = StmtEntry();
    collecting_ = false;
  }

  void Record(const Expr &pointer, Optional<PrimExpr> width,
              AccessEntry::Kind kind) {
    if (!collecting_ || !in_device_)
      return;
    auto address = addresses_->Resolve(pointer);
    if (address.scope != scope_name_)
      return;
    if (width && prim::IsZero(width.value()))
      return;
    current_.access.push_back({kind,
                               threads_,
                               GetConstrSet(),
                               std::move(address),
                               width,
                               {},
                               async_depth_ > 0,
                               atomic_depth_ > 0});
  }

  void RecordSync(const String &scope) {
    if (!collecting_ || !in_device_ || scope != scope_name_)
      return;
    current_.access.push_back({AccessEntry::kSync,
                               threads_,
                               GetConstrSet(),
                               {std::nullopt, std::nullopt, scope},
                               std::nullopt});
  }

  void VisitPointer(const Expr &pointer, Optional<PrimExpr> width, int mask);
  std::vector<AccessEntry> Summarize(const std::vector<StmtEntry> &seq,
                                     const ForObj *loop);
  std::vector<AccessEntry> Exposed(const std::vector<StmtEntry> &seq) const;
  bool Conflict(const AccessEntry &prev, const AccessEntry &curr,
                const ForObj *loop) const;
  bool Conflict(const std::vector<AccessEntry> &entries,
                const AccessEntry &current, const ForObj *loop) const {
    for (const auto &entry : entries)
      if (Conflict(entry, current, loop))
        return true;
    return false;
  }

  ObjectPtr<StorageAddresses> addresses_;
  String scope_name_;
  int warp_size_;
  bool in_device_{false};
  bool collecting_{false};
  int async_depth_{0};
  int atomic_depth_{0};
  Threads threads_;
  ConditionProperties properties_;
  StmtEntry current_;
  size_t current_occurrence_{std::numeric_limits<size_t>::max()};
  std::vector<std::vector<StmtEntry>> scopes_;
};

void ThreadSyncPlanner::VisitPointer(const Expr &pointer,
                                     Optional<PrimExpr> width, int mask) {
  if (mask & 1)
    Record(pointer, width, AccessEntry::kRead);
  if (mask & 2)
    Record(pointer, width, AccessEntry::kWrite);
  // A TensorLoad used as an address is not itself a read. Its index expressions
  // may read memory and are evaluated normally. Flat Bind definitions have
  // already been evaluated and must not be revisited at every pointer use.
  std::function<void(const Expr &)> visit_address = [&](const Expr &expr) {
    if (const auto *load = expr.as<TensorLoadObj>()) {
      for (const PrimExpr &index : load->indices)
        Visit(index);
    } else if (const auto *call = expr.as<CallObj>()) {
      const auto name = OpName(call);
      if (name == "tl.access_ptr" || name == "tl.ptr_access" ||
          name == "tirx.address_of" || name == "tirx.tensor_data_ptr" ||
          name == "tirx.ptr_byte_offset" || name == "tirx.reinterpret") {
        visit_address(call->args[0]);
        for (size_t i = 1; i < call->args.size(); ++i)
          Visit(call->args[i]);
      } else {
        for (const Expr &arg : call->args)
          Visit(arg);
      }
    }
  };
  visit_address(pointer);
}

ThreadSyncPlanner::Result ThreadSyncPlanner::Visit_(const CallObj *op) {
  String name = OpName(op);
  if (name == "tl.annotation" || name == "tirx.alloc_tensor")
    return std::nullopt;
  if (IsStorageSync(op)) {
    RecordSync(op->args[0].as_or_throw<StringImm>()->value);
    for (size_t i = 1; i < op->args.size(); ++i)
      Visit(op->args[i]);
    return std::nullopt;
  }
  if (name == "tl.sync_grid" || name == "tl.syncthreads_count" ||
      name == "tl.syncthreads_and" || name == "tl.syncthreads_or") {
    for (const Expr &arg : op->args)
      Visit(arg);
    RecordSync(scope_name_);
    return std::nullopt;
  }

  auto pointer_width = [](const Expr &pointer,
                          const PrimExpr &extent) -> Optional<PrimExpr> {
    auto type = pointer->ty.as<PtrType>();
    auto count = WidenIndex(extent);
    if (!type || !count)
      return std::nullopt;
    auto element = type.value()->element_type.as<PrimType>();
    if (!element || element.value().IsScalableVector() ||
        element.value().IsVoid())
      return std::nullopt;
    return count.value() * (element.value().bits() * element.value().lanes());
  };

  if (name == "tl.access_ptr" || name == "tl.ptr_access") {
    size_t extent_index = name == "tl.access_ptr" ? 1 : 2;
    int mask = GetConservativeAccessMask(
        op->args[extent_index + 1].as_or_throw<PrimExpr>());
    auto width = pointer_width(GetRef<Expr>(op),
                               op->args[extent_index].as_or_throw<PrimExpr>());
    VisitPointer(GetRef<Expr>(op), width, mask);
    return std::nullopt;
  }
  if (name == "tirx.address_of" || name == "tirx.tensor_data_ptr" ||
      name == "tirx.ptr_byte_offset" || name == "tirx.decl_tensor" ||
      (name == "tirx.reinterpret" && op->ty.as<PtrTypeObj>())) {
    // Pointer creation carries no read/write effect. Opaque consumers of an
    // unannotated pointer are handled below with a conservative access mask.
    VisitPointer(GetRef<Expr>(op), std::nullopt, 0);
    return std::nullopt;
  }
  if (name.find("tl.atomic_") == 0 || name == "tirx.atomic_add") {
    if (op->args.empty())
      return std::nullopt;
    int mask = name == "tl.atomic_load_elem_op"    ? 1
               : name == "tl.atomic_store_elem_op" ? 2
                                                   : 3;
    int count = name.find("addx4") != String::npos   ? 4
                : name.find("addx2") != String::npos ? 2
                                                     : 1;
    ++atomic_depth_;
    VisitPointer(op->args[0],
                 pointer_width(op->args[0], IntImm(PrimType::Int(64), count)),
                 mask);
    --atomic_depth_;
    for (size_t i = 1; i < op->args.size(); ++i)
      Visit(op->args[i]);
    return std::nullopt;
  }
  if (name == "tl.ptx_cp_async" || name == "tl.ptx_cp_async_bytes") {
    auto count = op->args[2].as_or_throw<PrimExpr>();
    Optional<PrimExpr> width;
    if (name == "tl.ptx_cp_async_bytes") {
      if (auto bytes = WidenIndex(count))
        width = bytes.value() * 8;
    } else {
      width = pointer_width(op->args[0], count);
    }
    ++async_depth_;
    VisitPointer(op->args[0], width, 2);
    VisitPointer(op->args[1], width, 1);
    --async_depth_;
    for (size_t i = 2; i < op->args.size(); ++i)
      Visit(op->args[i]);
    return std::nullopt;
  }
  bool async = name == "tl.tma_load" || name == "tl.tma_load_im2col" ||
               name == "tl.tma_load_multicast" || name == "tl.tma_load_gather4";
  if (async)
    ++async_depth_;
  bool opaque = SideEffect(GetRef<Expr>(op)) > CallEffectKind::kPure;
  for (const Expr &arg : op->args) {
    const auto *nested = arg.as<CallObj>();
    if (opaque && (arg->ty.as<PtrTypeObj>() || arg->ty.as<TensorTypeObj>()) &&
        !(nested && (OpName(nested) == "tl.access_ptr" ||
                     OpName(nested) == "tl.ptr_access"))) {
      VisitPointer(arg, std::nullopt, 3);
    } else {
      Visit(arg);
    }
  }
  if (async)
    --async_depth_;
  return std::nullopt;
}

ThreadSyncPlanner::Result ThreadSyncPlanner::Visit_(const IfObj *op) {
  size_t occurrence = Start(op);
  Visit(op->condition);
  auto condition_access = std::move(current_.access);
  current_ = StmtEntry();
  collecting_ = false;
  const auto before = syncs;
  std::vector<StmtEntry> then_entries;
  std::vector<StmtEntry> else_entries;
  {
    auto guard = MakeGuard(op->condition);
    scopes_.emplace_back();
    Visit(op->then_case);
    then_entries = std::move(scopes_.back());
    scopes_.pop_back();
    Summarize(then_entries, nullptr);
  }
  const auto after_then = syncs;
  if (op->else_case) {
    auto guard = MakeGuard(Not(op->condition));
    scopes_.emplace_back();
    Visit(op->else_case.value());
    else_entries = std::move(scopes_.back());
    scopes_.pop_back();
    Summarize(else_entries, nullptr);
  }
  std::vector<size_t> then_syncs, else_syncs;
  for (size_t sync : syncs) {
    if (!before.count(sync) && after_then.count(sync))
      then_syncs.push_back(sync);
    else if (!after_then.count(sync))
      else_syncs.push_back(sync);
  }
  auto should_hoist = [&](const PrimExpr &condition, const char *branch) {
    ConditionProperty property =
        GetConditionProperty(condition, threads_, properties_);
    auto analyzer = CreateTileLangAnalyzer();
    auto context = GetConstrSet().Populate(analyzer);
    if (const auto *tx = FindThread(threads_, "threadIdx.x")) {
      auto extent = analyzer->Simplify(tx->dom->extent).as<IntImmObj>();
      if (extent) {
        ScopedAnalyzerConstraints guard(analyzer, condition);
        if (analyzer->z3_prover.CountSatisfyingValues(
                tx->var, extent->value.as<int64_t>().value(), warp_size_) < 0)
          property.requires_hoist = true;
      } else {
        property.requires_hoist = true;
      }
    }
    bool proven_uniform = (property.runtime || property.requires_hoist) &&
                          UniformCondition(condition, GetConstrSet(), threads_);
    bool block_uniform = property.block_uniform || proven_uniform;
    bool hoist = (property.runtime && !block_uniform) ||
                 (property.requires_hoist && !proven_uniform);
    if (!hoist && !block_uniform &&
        !UniformCondition(condition, GetConstrSet(), threads_, warp_size_)) {
      TVM_FFI_THROW(InternalError)
          << "[ThreadSync] Cannot lower a required shared-memory sync inside "
             "the "
          << branch
          << " branch: its participating threads are not warp-uniform. "
          << "A partial barrier requires every non-exited thread in each "
             "participating warp "
          << "to reach the barrier. Condition: " << condition;
    }
    return hoist;
  };
  bool hoist_then = !then_syncs.empty() && should_hoist(op->condition, "then");
  bool hoist_else =
      !else_syncs.empty() && should_hoist(Not(op->condition), "else");
  if (hoist_then || hoist_else) {
    LOG(WARNING) << "[ThreadSync] Hoisting sync out of an if whose condition "
                    "is not safe for an in-if "
                 << "sync. This is not a fix: both ends of the conflict are "
                    "inside the branch, so the "
                 << "hoisted barrier no longer separates them and the race "
                    "remains. Constraining the "
                 << "condition -- a T.assume on the parameters it reads -- "
                    "keeps the barrier in place "
                 << "instead. Condition: " << op->condition;
    if (hoist_then)
      for (size_t sync : then_syncs)
        syncs.erase(sync);
    if (hoist_else)
      for (size_t sync : else_syncs)
        syncs.erase(sync);
    syncs.insert(occurrence);
  }
  // A conditional barrier does not fence accesses of threads that skip that
  // branch. Keep both branches' unsynchronized boundary accesses visible to
  // the enclosing sequence; do not promote the branch-local kSync marker.
  auto append = [&](const std::vector<StmtEntry> &entries) {
    for (const auto &entry : entries)
      for (const auto &access : entry.access)
        if (access.kind != AccessEntry::kSync)
          condition_access.push_back(access);
  };
  append(then_entries);
  append(else_entries);
  if (!condition_access.empty())
    scopes_.back().push_back({occurrence, std::move(condition_access)});
  return std::nullopt;
}

ThreadSyncPlanner::Result ThreadSyncPlanner::Visit_(const WhileObj *op) {
  size_t occurrence = Start(op);
  Visit(op->condition);
  auto condition_access = std::move(current_.access);
  current_ = StmtEntry();
  collecting_ = false;
  scopes_.emplace_back();
  {
    auto guard = MakeGuard(op->condition);
    Visit(op->body);
  }
  auto entries = std::move(scopes_.back());
  scopes_.pop_back();
  Summarize(entries, nullptr);
  // A while-loop has no affine successor map. Test exposed end/start accesses
  // conservatively with independent body-local Bind identities.
  auto exposed = Exposed(entries);
  for (const auto &entry : entries) {
    bool needs_sync = false;
    for (const auto &access : entry.access) {
      if (access.kind == AccessEntry::kSync)
        break;
      for (const auto &previous : exposed)
        if ((previous.kind == AccessEntry::kWrite ||
             access.kind == AccessEntry::kWrite) &&
            previous.kind != AccessEntry::kSync &&
            Conflict(previous, access, nullptr))
          needs_sync = true;
    }
    if (needs_sync) {
      syncs.insert(entry.occurrence);
      break;
    }
  }
  for (const auto &entry : entries)
    for (const auto &access : entry.access)
      if (access.kind != AccessEntry::kSync)
        condition_access.push_back(access);
  if (!condition_access.empty())
    scopes_.back().push_back({occurrence, std::move(condition_access)});
  return std::nullopt;
}

std::vector<AccessEntry>
ThreadSyncPlanner::Exposed(const std::vector<StmtEntry> &seq) const {
  bool seen_sync = false;
  std::vector<AccessEntry> head, tail;
  AccessEntry fence{AccessEntry::kSync,
                    threads_,
                    GetConstrSet(),
                    {std::nullopt, std::nullopt, scope_name_},
                    std::nullopt};
  auto synchronize = [&]() {
    if (!seen_sync)
      head.push_back(fence);
    else
      tail.clear();
    seen_sync = true;
  };
  for (const auto &statement : seq) {
    if (syncs.count(statement.occurrence))
      synchronize();
    for (const auto &access : statement.access) {
      if (access.kind == AccessEntry::kSync)
        synchronize();
      else
        (seen_sync ? tail : head).push_back(access);
    }
  }
  head.insert(head.end(), tail.begin(), tail.end());
  return head;
}

std::vector<AccessEntry>
ThreadSyncPlanner::Summarize(const std::vector<StmtEntry> &seq,
                             const ForObj *loop) {
  std::vector<AccessEntry> reads, writes;
  auto clear = [&]() {
    reads.clear();
    writes.clear();
  };
  for (const auto &statement : seq) {
    if (syncs.count(statement.occurrence))
      clear();
    bool barrier = false;
    for (const auto &access : statement.access) {
      if (access.kind == AccessEntry::kSync) {
        clear();
      } else if ((access.kind == AccessEntry::kRead &&
                  Conflict(writes, access, nullptr)) ||
                 (access.kind == AccessEntry::kWrite &&
                  (Conflict(reads, access, nullptr) ||
                   Conflict(writes, access, nullptr)))) {
        barrier = true;
        break;
      }
    }
    if (barrier) {
      syncs.insert(statement.occurrence);
      clear();
    }
    for (const auto &access : statement.access) {
      if (access.kind == AccessEntry::kSync)
        clear();
      else
        (access.kind == AccessEntry::kRead ? reads : writes).push_back(access);
    }
  }
  if (loop) {
    for (const auto &statement : seq) {
      if (syncs.count(statement.occurrence) ||
          (reads.empty() && writes.empty()))
        break;
      bool barrier = false;
      for (const auto &access : statement.access) {
        if (access.kind == AccessEntry::kSync) {
          clear();
        } else if ((access.kind == AccessEntry::kRead &&
                    Conflict(writes, access, loop)) ||
                   (access.kind == AccessEntry::kWrite &&
                    (Conflict(reads, access, loop) ||
                     Conflict(writes, access, loop)))) {
          barrier = true;
          break;
        }
      }
      if (barrier) {
        // A pre-loop barrier cannot order iteration i against i+step, including
        // write-only WAW. Place the required fence on the real carried edge.
        syncs.insert(statement.occurrence);
        break;
      }
    }
  }
  return Exposed(seq);
}

bool ThreadSyncPlanner::Conflict(const AccessEntry &previous,
                                 const AccessEntry &current,
                                 const ForObj *loop) const {
  if (previous.kind == AccessEntry::kWrite &&
      current.kind == AccessEntry::kWrite && previous.async_copy &&
      current.async_copy)
    return false;
  if (previous.atomic && current.atomic)
    return false;
  if (addresses_->Independent(previous.address, current.address))
    return false;
  if (!previous.address.root || !current.address.root ||
      (!previous.address.root.value().same_as(current.address.root.value()) &&
       previous.address.scope != "shared.dyn"))
    return true;
  if (!previous.address.bit_offset || !current.address.bit_offset ||
      !previous.width_bits || !current.width_bits)
    return true;

  auto analyzer = CreateTileLangAnalyzer();
  ConstrSet left = previous.constraints, right = current.constraints;
  Map<Var, PrimExpr> left_sub, right_sub;
  PrimExpr left_start = previous.address.bit_offset.value();
  PrimExpr right_start = current.address.bit_offset.value();
  PrimExpr left_width = previous.width_bits.value();
  PrimExpr right_width = current.width_bits.value();

  // Compare different hardware threads for RAW, WAR, and WAW alike. Serial
  // program order already orders accesses of one thread, including iterations.
  PrimExpr different_threads = IntImm(PrimType::Bool(), 0);
  for (const char *tag : {"threadIdx.x", "threadIdx.y", "threadIdx.z"}) {
    const auto *prev_thread = FindThread(previous.threads, tag);
    const auto *curr_thread = FindThread(current.threads, tag);
    if (!prev_thread && !curr_thread)
      continue;
    if (!prev_thread || !curr_thread)
      return true;
    PrimVar a(prev_thread->var->name + "<PREV>", prev_thread->var.ty());
    PrimVar b(curr_thread->var->name + "<CURR>", curr_thread->var.ty());
    left_sub.Set(prev_thread->var, a);
    right_sub.Set(curr_thread->var, b);
    different_threads = Or(different_threads, Cast(PrimType::Int(64), a) !=
                                                  Cast(PrimType::Int(64), b));
  }
  if (IsZero(different_threads))
    return false;

  // Iterations of a fully traversed inner loop are independent in a summary.
  // Keeping that induction variable shared would hide cross-iteration hazards
  // when an enclosing sequence compares two whole loops.
  for (const Var &var : previous.relaxed_vars)
    left_sub.Set(var, PrimVar(var->name + "<PREV_LOOP>", var->ty));
  for (const Var &var : current.relaxed_vars)
    right_sub.Set(var, PrimVar(var->name + "<CURR_LOOP>", var->ty));

  left = left.RenameFrom("<PREV>", left_sub, std::nullopt, false);
  right = right.RenameFrom("<CURR>", right_sub, std::nullopt, false);
  if (loop) {
    auto step = loop->step.as<IntImmObj>();
    auto value = step ? step->value.as<int64_t>() : std::nullopt;
    auto loop_analyzer = CreateTileLangAnalyzer();
    auto loop_context = previous.constraints.Populate(loop_analyzer);
    // The shifted-iteration proof below requires monotonic, non-wrapping
    // induction. Otherwise retain the possible loop-carried hazard.
    if (!value || *value <= 0 ||
        !CanUseLoopBounds(loop->begin, loop->end, loop->step,
                          loop_analyzer.get()))
      return true;
    Map<Var, PrimExpr> shift{{loop->var, loop->var + loop->step}};
    right = right.ToConstraints().Substitute(shift);
    right_start = SubstituteConstraintExpr(right_start, shift);
    right_width = SubstituteConstraintExpr(right_width, shift);
    left.AddConstr(loop->var + loop->step < loop->end);
  }
  auto context =
      left.ToConstraints().Merge(right.ToConstraints()).Populate(analyzer);
  // Unit thread axes cannot distinguish hardware threads. Reduce their
  // comparisons using the active domains before the solver journals this
  // premise, avoiding irrelevant disjunctions in the bounded conflict proof.
  ScopedAnalyzerConstraints thread_guard(analyzer,
                                         analyzer->Simplify(different_threads));

  auto fresh_left = make_object<FreshenMutableReads>();
  auto fresh_right = make_object<FreshenMutableReads>();
  left_start = (*fresh_left)(SubstituteConstraintExpr(left_start, left_sub));
  right_start =
      (*fresh_right)(SubstituteConstraintExpr(right_start, right_sub));
  left_width = SubstituteConstraintExpr(left_width, left_sub);
  right_width = SubstituteConstraintExpr(right_width, right_sub);

  auto interval = [&](PrimExpr index, PrimExpr width, const char *lane_name)
      -> std::optional<std::pair<PrimExpr, PrimExpr>> {
    if (!width.ty().IsScalar())
      return std::nullopt;
    if (!index.ty().IsScalar()) {
      if (!index.ty().IsFixedLengthVector())
        return std::nullopt;
      int lanes = index.ty().lanes();
      PrimVar lane(lane_name, PrimType::Int(32));
      auto scalar = ScalarLaneValue(index, lane, lanes);
      if (!scalar)
        return std::nullopt;
      BindAnalyzer(
          analyzer, lane,
          Range::FromMinExtent(IntImm(lane.ty(), 0), IntImm(lane.ty(), lanes)));
      index = *scalar;
    }
    auto guard = make_object<LaneArithmeticGuard>(analyzer);
    guard->Visit(index);
    if (!guard->safe)
      return std::nullopt;
    index = analyzer->Simplify(index);
    PrimExpr end = index + width - 1;
    // Sub-byte stores use byte/word RMW in existing emitters. Use a
    // conservative word footprint instead of treating neighbouring packed
    // elements as independently writable. No whole-byte access is rounded this
    // way.
    if (auto bits = width.as<IntImmObj>();
        bits && bits->value.as<int64_t>().value() < 8) {
      index = FloorDiv(index, IntImm(index.ty(), 32)) * 32;
      end = index + 31;
    }
    return std::pair(index, end);
  };
  auto a = interval(left_start, left_width, "prev_lane");
  auto b = interval(right_start, right_width, "curr_lane");
  if (!a || !b)
    return true;
  return !analyzer->CanProve(Or(a->second < b->first, b->second < a->first));
}

// Rewrite explicit and inferred full-block barriers only after the planner has
// fixed their placement. Replaying the lexical constraint stream here avoids
// stale Analyzer facts from sibling guards, and retains flat Bind definitions.
struct PartialBarrierOccurrence {
  const EvaluateObj *statement;
  Optional<Stmt> replacement;
};

class PartialBarrierLowering : public ConstrVisitor {
public:
  explicit PartialBarrierLowering(int warp_size) : warp_size_(warp_size) {}
  std::vector<PartialBarrierOccurrence> occurrences;

  Result Visit_(const SeqStmtObj *op) final {
    auto saved = properties_;
    auto result = ConstrVisitor::Visit_(op);
    properties_ = std::move(saved);
    return result;
  }
  Result Visit_(const BindObj *op) final {
    if (op->value.as<PrimExpr>())
      properties_.insert_or_assign(
          op->var.get(),
          GetConditionProperty(op->value, threads_, properties_));
    return ConstrVisitor::Visit_(op);
  }
  Result Visit_(const RegionStmtObj *op) final {
    auto old_threads = threads_.size();
    auto bindings = GetRegionThreadBindings(op);
    threads_.insert(threads_.end(), bindings.begin(), bindings.end());
    int previous = runtime_divergence_;
    if (op->op->name == "tl.attr") {
      auto values = AttrValues(op);
      if (values.at("key").cast<String>() == "tl.assume")
        EnterFilter(values.at("node").cast<PrimExpr>());
    }
    auto result = ConstrVisitor::Visit_(op);
    runtime_divergence_ = previous;
    threads_.erase(threads_.begin() + old_threads, threads_.end());
    return result;
  }
  Result Visit_(const IfObj *op) final {
    int previous = runtime_divergence_;
    EnterFilter(op->condition);
    auto result = ConstrVisitor::Visit_(op);
    runtime_divergence_ = previous;
    return result;
  }
  Result Visit_(const WhileObj *op) final {
    int previous = runtime_divergence_;
    EnterFilter(op->condition);
    auto result = ConstrVisitor::Visit_(op);
    runtime_divergence_ = previous;
    return result;
  }
  Result Visit_(const EvaluateObj *op) final {
    const auto *call = op->value.as<CallObj>();
    if (!IsFullSharedStorageSync(call))
      return ConstrVisitor::Visit_(op);
    // A shared immutable statement can occur in distinct lexical domains.
    // Record unchanged visits as well, so a later partial or empty domain
    // cannot overwrite the result for an earlier full-domain occurrence.
    occurrences.push_back({op, std::nullopt});
    if (threads_.empty())
      return std::nullopt;
    TVM_FFI_CHECK_EQ(runtime_divergence_, 0, InternalError)
        << "[ThreadSync] Cannot prove a thread rectangle with "
           "runtime-dependent participation";

    bool thread_constraint = false;
    for (const Constr &constraint : constr_stack_) {
      if (constraint.kind == Constr::kConstr &&
          !GetConditionProperty(constraint.value, threads_, properties_)
               .block_uniform)
        thread_constraint = true;
    }
    if (!thread_constraint)
      return std::nullopt;

    auto analyzer = CreateTileLangAnalyzer();
    auto context = GetConstrSet().Populate(analyzer);
    std::array<std::pair<int64_t, int64_t>, 3> bounds{{{0, 0}, {0, 0}, {0, 0}}};
    std::array<int64_t, 3> extents{{1, 1, 1}};
    int64_t count = 1, full_count = 1;
    for (int axis = 0; axis < 3; ++axis) {
      String tag = std::string("threadIdx.") + "xyz"[axis];
      const auto *thread = FindThread(threads_, tag);
      if (!thread)
        continue;
      auto extent_expr = analyzer->Simplify(thread->dom->extent);
      const auto *extent = extent_expr.as<IntImmObj>();
      TVM_FFI_CHECK(extent, InternalError)
          << "[ThreadSync] A partial barrier requires a constant thread "
             "rectangle, got "
          << thread->dom->extent;
      extents[axis] = extent->value.as<int64_t>().value();
      TVM_FFI_CHECK_GT(extents[axis], 0, InternalError);
      TVM_FFI_CHECK_LE(full_count,
                       std::numeric_limits<int32_t>::max() / extents[axis],
                       InternalError);
      full_count *= extents[axis];
      auto bound = analyzer->const_int_bound(thread->var);
      int64_t low = bound->min_value, high = bound->max_value;
      if (low > high ||
          analyzer->z3_prover.CanProve(IntImm(PrimType::Bool(), 0))) {
        occurrences.back().replacement = SeqStmt(Array<Stmt>{}, op->loc);
        return std::nullopt;
      }
      TVM_FFI_CHECK(low >= 0 && high < extents[axis], InternalError)
          << "[ThreadSync] Cannot prove a bounded thread rectangle";
      int64_t width = high - low + 1;
      int64_t values =
          analyzer->z3_prover.CountSatisfyingValues(thread->var, width);
      TVM_FFI_CHECK_GT(values, 0, InternalError)
          << "[ThreadSync] Cannot prove a thread rectangle";
      if (values != width) {
        // Count enumerates a projection. A matching contiguous containing
        // interval proves no holes; Count alone may have hit a resource limit.
        int64_t upper_search = high - values + 1;
        while (low < upper_search) {
          int64_t middle = low + (upper_search - low + 1) / 2;
          if (analyzer->z3_prover.CanProve(thread->var >=
                                           IntImm(thread->var.ty(), middle)))
            low = middle;
          else
            upper_search = middle - 1;
        }
        high = low + values - 1;
        TVM_FFI_CHECK(analyzer->z3_prover.CanProve(
                          thread->var <= IntImm(thread->var.ty(), high)),
                      InternalError)
            << "[ThreadSync] Cannot prove a thread rectangle";
      }
      bounds[axis] = {low, high};
      count *= values;
    }

    // Per-axis projections are insufficient: tx == ty has full projections,
    // but it is a diagonal rather than a rectangle. Check the cardinality of
    // the injective linear thread ID against the Cartesian product as well.
    PrimVar linear("partial_sync_linear_id", PrimType::Int(64));
    BindAnalyzer(analyzer, linear, LinearThreadId(threads_));
    TVM_FFI_CHECK_EQ(analyzer->z3_prover.CountSatisfyingValues(linear, count),
                     count, InternalError)
        << "[ThreadSync] Cannot prove a joint thread rectangle";
    if (count == full_count)
      return std::nullopt;

    TVM_FFI_CHECK_EQ(count % warp_size_, 0, InternalError)
        << "[ThreadSync] A partial barrier requires a warp-multiple thread "
           "count; got "
        << count;
    // Count alone would accept [16, 80). CUDA counts complete hardware warps;
    // certify the actual linearized ownership, not only a multiple of 32.
    for (int64_t warp = 0; warp < full_count; warp += warp_size_) {
      int active = 0;
      int present =
          static_cast<int>(std::min<int64_t>(warp_size_, full_count - warp));
      for (int lane = 0; lane < present; ++lane) {
        int64_t index = warp + lane;
        bool selected = true;
        for (int axis = 0; axis < 3; ++axis) {
          int64_t coordinate = index % extents[axis];
          index /= extents[axis];
          selected &= bounds[axis].first <= coordinate &&
                      coordinate <= bounds[axis].second;
        }
        active += selected;
      }
      TVM_FFI_CHECK(active == 0 || active == present, InternalError)
          << "[ThreadSync] A partial barrier's participating threads are not "
             "warp-uniform";
    }
    ThreadBoundKey key{bounds[0].first,  bounds[0].second, bounds[1].first,
                       bounds[1].second, bounds[2].first,  bounds[2].second};
    auto [it, inserted] = ids_.emplace(
        key, ids_.size() +
                 static_cast<size_t>(ReservedNamedBarriers::kFirstUsedBarrier));
    (void)inserted;
    TVM_FFI_CHECK_LT(it->second, 16U, InternalError)
        << "[ThreadSync] Too many partial-sync barriers";
    occurrences.back().replacement =
        Evaluate(Call(PrimType::Void(), Op::Get("tl.storage_sync"),
                      {call->args[0], IntImm(PrimType::Int(32), it->second),
                       IntImm(PrimType::Int(32), count)},
                      {}, {}, call->loc),
                 op->loc);
    return std::nullopt;
  }

private:
  void EnterFilter(const PrimExpr &condition) {
    const auto property =
        GetConditionProperty(condition, threads_, properties_);
    if (property.runtime && !property.block_uniform &&
        !UniformCondition(condition, GetConstrSet(), threads_))
      ++runtime_divergence_;
  }
  int warp_size_;
  int runtime_divergence_{0};
  Threads threads_;
  ConditionProperties properties_;
  std::unordered_map<ThreadBoundKey, size_t> ids_;
};

class PartialBarrierRewriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  explicit PartialBarrierRewriter(
      std::vector<PartialBarrierOccurrence> occurrences)
      : occurrences_(std::move(occurrences)) {}

  SeqStmt Rewrite(const SeqStmt &body) {
    auto result = Mutate(body).ValueOrUnchanged(body);
    TVM_FFI_CHECK_EQ(next_, occurrences_.size(), InternalError)
        << "Partial barrier traversal did not consume every occurrence";
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    if (!IsFullSharedStorageSync(op->value.as<CallObj>()))
      return StmtExprMutator::Mutate_(op, mode);
    TVM_FFI_CHECK_LT(next_, occurrences_.size(), InternalError)
        << "Partial barrier traversal found an unrecorded occurrence";
    const auto &occurrence = occurrences_[next_++];
    TVM_FFI_CHECK_EQ(occurrence.statement, op, InternalError)
        << "Partial barrier traversal changed lexical occurrence order";
    if (occurrence.replacement)
      return occurrence.replacement.value();
    return Unchanged();
  }

private:
  std::vector<PartialBarrierOccurrence> occurrences_;
  size_t next_{0};
};

Type InferNamedStorageSync(const CallObj *call) {
  TVM_FFI_CHECK(call->args.size() == 2 || call->args.size() == 3, ValueError)
      << "tl.storage_sync expects (scope, barrier_id[, thread_count])";
  const auto scope = call->args[0].as_or_throw<StringImm>()->value;
  TVM_FFI_CHECK(IsShared(scope), ValueError)
      << "A named storage barrier requires shared or shared.dyn scope";
  if (const auto *id = call->args[1].as<IntImmObj>()) {
    auto value = id->value.as<int64_t>().value();
    TVM_FFI_CHECK(value >= 0 && value < 16, ValueError)
        << "A named storage barrier ID must be in [0, 15]";
  }
  if (call->args.size() == 3) {
    if (const auto *count = call->args[2].as<IntImmObj>())
      TVM_FFI_CHECK_GT(count->value.as<int64_t>().value(), 0, ValueError)
          << "A named storage barrier needs a positive thread count";
  }
  return PrimType::Void();
}

SeqStmt PlanThreadStorageSyncImpl(const tirx::Function &function, SeqStmt body,
                                  const String &storage_scope, int warp_size,
                                  bool insert_cuda_wait_sync,
                                  bool lower_cuda_partial_sync) {
  if (storage_scope == "global")
    return body;
  // Verify that callers name a registered native scope.
  runtime::MemoryScope::Get(storage_scope);
  TVM_FFI_CHECK_GT(warp_size, 0, ValueError);
  if (insert_cuda_wait_sync && storage_scope == "shared") {
    auto waits = make_object<SyncAfterWaitGroup>(storage_scope);
    body = waits->Mutate(body).ValueOrUnchanged(body);
  }
  auto planner =
      make_object<ThreadSyncPlanner>(function, storage_scope, warp_size);
  planner->Visit(body);
  auto insert = make_object<StatementRewriter>();
  insert->scope = storage_scope;
  insert->insert_before = std::move(planner->syncs);
  insert->occurrences = std::move(planner->occurrences);
  body = insert->Rewrite(body);
  if (lower_cuda_partial_sync) {
    auto partial = make_object<PartialBarrierLowering>(warp_size);
    partial->Visit(body);
    auto rewrite =
        make_object<PartialBarrierRewriter>(std::move(partial->occurrences));
    body = rewrite->Rewrite(body);
  }
  return body;
}

tirx::Function TileLangThreadSync(tirx::Function function,
                                  const String &storage_scope) {
  if (function->GetAttr(tvm::attr::kDeclaration).value_or(false) ||
      storage_scope == "global")
    return function;
  int warp_size = 32;
  if (auto target = function->attrs.GetAttr<Target>(tvm::attr::kTarget))
    warp_size =
        target.value()->GetAttr<int64_t>("thread_warp_size").value_or(32);
  auto body = PlanThreadStorageSync(function, function->body, storage_scope,
                                    warp_size, true, true);
  function.CopyOnWrite()->body = std::move(body);
  return function;
}

tvm::transform::Pass ThreadSync(const String &scope) {
  auto pass = [scope](tirx::Function function, IRModule,
                      tvm::transform::PassContext context) {
    if (context->GetConfig<bool>(kDisableThreadStorageSync).value_or(false))
      return function;
    return TileLangThreadSync(std::move(function), scope);
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.ThreadSync");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.storage_sync", "TileLang's named shared-memory barrier")
      .signature(sig::arg<StringImm>("scope"),
                 sig::var_args<IntExpr>("barrier_values"))
      .set_native_fn_attr<&InferNamedStorageSync>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
  reflection::GlobalDef().def("tl.transform.ThreadSync", ThreadSync);
}

} // namespace

SeqStmt PlanThreadStorageSync(const tirx::Function &function,
                              const SeqStmt &body,
                              const ffi::String &storage_scope, int warp_size,
                              bool insert_cuda_wait_sync,
                              bool lower_cuda_partial_sync) {
  return PlanThreadStorageSyncImpl(function, body, storage_scope, warp_size,
                                   insert_cuda_wait_sync,
                                   lower_cuda_partial_sync);
}

} // namespace tvm::tl
