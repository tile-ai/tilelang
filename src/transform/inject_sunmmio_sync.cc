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
 * \file inject_sunmmio_sync.cc
 * \brief Inject hardware-unit synchronization and collective launch/completion
 * barriers.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "../op/comm.h"
#include "../op/utils.h"
#include "../target/sunmmio_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

namespace {

PrimExpr I64Imm(int64_t value) { return IntImm(DataType::Int(64), value); }

PrimExpr AsI64(PrimExpr value) {
  if (const auto *imm = value.as<IntImmNode>()) {
    return I64Imm(imm->value);
  }
  if (value.dtype() == DataType::Int(64)) {
    return value;
  }
  return Cast(DataType::Int(64), value);
}

PrimExpr CoreBitMask(PrimExpr core_id) {
  if (const auto *imm = core_id.as<IntImmNode>()) {
    ICHECK_GE(imm->value, 0);
    ICHECK_LT(imm->value, 64)
        << "barrier mask currently supports core ids in [0, 64)";
    return I64Imm(static_cast<int64_t>(uint64_t{1} << imm->value));
  }
  return I64Imm(1) << AsI64(core_id);
}

bool SamePrimExpr(const PrimExpr &lhs, const PrimExpr &rhs) {
  return StructuralEqual()(lhs, rhs);
}

struct BarrierMaskInfo {
  PrimExpr expr;
  std::vector<int64_t> candidates;
};

void AddUniqueInt64(std::vector<int64_t> *values, int64_t value) {
  if (std::find(values->begin(), values->end(), value) == values->end()) {
    values->push_back(value);
  }
}

uint64_t UnsignedMask(int64_t value) { return static_cast<uint64_t>(value); }

std::optional<int64_t> FloorDivInt64(int64_t lhs, int64_t rhs) {
  if (rhs == 0) {
    return std::nullopt;
  }
  int64_t quotient = lhs / rhs;
  int64_t remainder = lhs % rhs;
  if (remainder != 0 && ((remainder > 0) != (rhs > 0))) {
    --quotient;
  }
  return quotient;
}

std::optional<int64_t> EvalInt64(PrimExpr expr, arith::Analyzer *analyzer) {
  if (analyzer) {
    expr = analyzer->Simplify(expr);
  }
  if (const auto *imm = expr.as<IntImmNode>()) {
    return static_cast<int64_t>(imm->value);
  }
  if (const auto *op = expr.as<CastNode>()) {
    return EvalInt64(op->value, analyzer);
  }

  auto eval_binary = [&](const PrimExpr &a, const PrimExpr &b,
                         auto fn) -> std::optional<int64_t> {
    std::optional<int64_t> lhs = EvalInt64(a, analyzer);
    std::optional<int64_t> rhs = EvalInt64(b, analyzer);
    if (!lhs || !rhs) {
      return std::nullopt;
    }
    return fn(*lhs, *rhs);
  };

  if (const auto *op = expr.as<AddNode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a + b; });
  }
  if (const auto *op = expr.as<SubNode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a - b; });
  }
  if (const auto *op = expr.as<MulNode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a * b; });
  }
  if (const auto *op = expr.as<DivNode>()) {
    return eval_binary(op->a, op->b, [](int64_t a, int64_t b) {
      return b == 0 ? std::optional<int64_t>() : std::optional<int64_t>(a / b);
    });
  }
  if (const auto *op = expr.as<ModNode>()) {
    return eval_binary(op->a, op->b, [](int64_t a, int64_t b) {
      return b == 0 ? std::optional<int64_t>() : std::optional<int64_t>(a % b);
    });
  }
  if (const auto *op = expr.as<FloorDivNode>()) {
    return eval_binary(
        op->a, op->b, [](int64_t a, int64_t b) { return FloorDivInt64(a, b); });
  }
  if (const auto *op = expr.as<FloorModNode>()) {
    return eval_binary(op->a, op->b, [](int64_t a, int64_t b) {
      std::optional<int64_t> div = FloorDivInt64(a, b);
      return div ? std::optional<int64_t>(a - *div * b)
                 : std::optional<int64_t>();
    });
  }
  if (const auto *op = expr.as<EQNode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a == b ? 1 : 0; });
  }
  if (const auto *op = expr.as<NENode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a != b ? 1 : 0; });
  }
  if (const auto *op = expr.as<LTNode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a < b ? 1 : 0; });
  }
  if (const auto *op = expr.as<LENode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a <= b ? 1 : 0; });
  }
  if (const auto *op = expr.as<GTNode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a > b ? 1 : 0; });
  }
  if (const auto *op = expr.as<GENode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a >= b ? 1 : 0; });
  }
  if (const auto *op = expr.as<SelectNode>()) {
    std::optional<int64_t> cond = EvalInt64(op->condition, analyzer);
    return cond ? EvalInt64(*cond != 0 ? op->true_value : op->false_value,
                            analyzer)
                : std::optional<int64_t>();
  }
  if (const auto *call = expr.as<CallNode>()) {
    const auto *op = call->op.as<OpNode>();
    if (!op || call->args.size() != 2) {
      return std::nullopt;
    }
    if (op->name == "tir.bitwise_or") {
      return eval_binary(
          call->args[0], call->args[1], [](int64_t a, int64_t b) {
            return static_cast<int64_t>(UnsignedMask(a) | UnsignedMask(b));
          });
    }
    if (op->name == "tir.bitwise_and") {
      return eval_binary(
          call->args[0], call->args[1], [](int64_t a, int64_t b) {
            return static_cast<int64_t>(UnsignedMask(a) & UnsignedMask(b));
          });
    }
    if (op->name == "tir.bitwise_xor") {
      return eval_binary(
          call->args[0], call->args[1], [](int64_t a, int64_t b) {
            return static_cast<int64_t>(UnsignedMask(a) ^ UnsignedMask(b));
          });
    }
    if (op->name == "tir.shift_left") {
      return eval_binary(call->args[0], call->args[1],
                         [](int64_t a, int64_t b) {
                           if (b < 0 || b >= 64) {
                             return std::optional<int64_t>();
                           }
                           return std::optional<int64_t>(
                               static_cast<int64_t>(UnsignedMask(a) << b));
                         });
    }
  }
  return std::nullopt;
}

int CountMaskBits(uint64_t mask) {
  return static_cast<int>(__builtin_popcountll(mask));
}

bool IsMaskWithinMesh(uint64_t mask, int total_cores) {
  if (total_cores == 64) {
    return true;
  }
  return (mask & ~((uint64_t{1} << total_cores) - 1)) == 0;
}

bool MaskAlignedWithDirection(uint64_t mask, int direction, int mesh_nrow,
                              int mesh_ncol) {
  int total_cores = mesh_nrow * mesh_ncol;
  if (mask == 0 || !IsMaskWithinMesh(mask, total_cores)) {
    return false;
  }
  int min_participants =
      direction == 0 ? std::min(mesh_ncol, 2) : std::min(mesh_nrow, 2);
  if (CountMaskBits(mask) < min_participants) {
    return false;
  }

  int ref_row = -1;
  int ref_col = -1;
  for (int core = 0; core < total_cores; ++core) {
    if ((mask & (uint64_t{1} << core)) == 0) {
      continue;
    }
    int row = core / mesh_ncol;
    int col = core % mesh_ncol;
    if (ref_row < 0) {
      ref_row = row;
      ref_col = col;
    } else if ((direction == 0 && row != ref_row) ||
               (direction == 1 && col != ref_col)) {
      return false;
    }
  }
  return true;
}

class VarCollector : public ExprVisitor {
public:
  void VisitExpr_(const VarNode *op) final {
    Var var = ffi::GetRef<Var>(op);
    for (const Var &existing : vars) {
      if (existing.same_as(var)) {
        return;
      }
    }
    vars.push_back(var);
  }

  std::vector<Var> vars;
};

bool ExprUsesAnyVar(const PrimExpr &expr, const std::vector<Var> &vars) {
  VarCollector collector;
  collector(expr);
  for (const Var &used : collector.vars) {
    for (const Var &candidate : vars) {
      if (used.same_as(candidate)) {
        return true;
      }
    }
  }
  return false;
}

std::vector<int64_t> EnumerateMaskCandidates(PrimExpr expr, int direction,
                                             int mesh_nrow, int mesh_ncol,
                                             arith::Analyzer *analyzer) {
  VarCollector collector;
  collector(expr);
  if (collector.vars.empty()) {
    std::optional<int64_t> value = EvalInt64(expr, analyzer);
    if (value && MaskAlignedWithDirection(UnsignedMask(*value), direction,
                                          mesh_nrow, mesh_ncol)) {
      return {*value};
    }
    return {};
  }
  if (collector.vars.size() > 2) {
    return {};
  }

  int total_cores = mesh_nrow * mesh_ncol;
  int64_t num_cases = 1;
  for (size_t i = 0; i < collector.vars.size(); ++i) {
    num_cases *= total_cores;
  }

  std::vector<int64_t> candidates;
  for (int64_t case_id = 0; case_id < num_cases; ++case_id) {
    Map<Var, PrimExpr> var_map;
    int64_t case_value = case_id;
    for (const Var &var : collector.vars) {
      int core = static_cast<int>(case_value % total_cores);
      case_value /= total_cores;
      var_map.Set(var, IntImm(var.dtype(), core));
    }
    PrimExpr candidate_expr = Substitute(expr, var_map);
    if (analyzer) {
      candidate_expr = analyzer->Simplify(candidate_expr);
    }
    std::optional<int64_t> value = EvalInt64(candidate_expr, analyzer);
    if (!value) {
      return {};
    }
    if (MaskAlignedWithDirection(UnsignedMask(*value), direction, mesh_nrow,
                                 mesh_ncol)) {
      AddUniqueInt64(&candidates, *value);
    }
  }
  return candidates;
}

Array<PrimExpr> MakeBarrierArgs(const BarrierMaskInfo &info) {
  Array<PrimExpr> args;
  args.push_back(info.expr);
  for (int64_t mask : info.candidates) {
    args.push_back(I64Imm(mask));
  }
  return args;
}

Array<PrimExpr> MakeBarrierInitArgs(const BarrierMaskInfo &info) {
  if (info.candidates.empty()) {
    return MakeBarrierArgs(info);
  }
  Array<PrimExpr> args;
  args.push_back(I64Imm(-1));
  for (int64_t mask : info.candidates) {
    args.push_back(I64Imm(mask));
  }
  return args;
}

BarrierMaskInfo BarrierMaskInfoFromArgs(const Array<PrimExpr> &args) {
  ICHECK_GE(args.size(), 1U) << "barrier call requires participant_mask";
  BarrierMaskInfo info;
  info.expr = args[0];
  for (size_t i = 1; i < args.size(); ++i) {
    const auto *imm = args[i].as<IntImmNode>();
    ICHECK(imm) << "barrier candidate masks must be IntImm";
    AddUniqueInt64(&info.candidates, static_cast<int64_t>(imm->value));
  }
  return info;
}

bool SameBarrierMaskInfo(const BarrierMaskInfo &lhs,
                         const BarrierMaskInfo &rhs) {
  return SamePrimExpr(lhs.expr, rhs.expr) && lhs.candidates == rhs.candidates;
}

void AddUniqueBarrierMaskInfo(std::vector<BarrierMaskInfo> *values,
                              const BarrierMaskInfo &value) {
  for (const BarrierMaskInfo &existing : *values) {
    if (SameBarrierMaskInfo(existing, value)) {
      return;
    }
  }
  values->push_back(value);
}

bool BroadcastCallHasSrcCore(const CallNode *call) {
  ICHECK_GE(call->args.size(), static_cast<size_t>(kBroadcastArgCount))
      << "broadcast_() call is missing its fixed argument prefix";
  size_t fixed_and_source_args = call->args.size();
  if (ParseSunmmioOdmaUnitExpr(call->args.back())) {
    --fixed_and_source_args;
  }
  ICHECK(fixed_and_source_args == static_cast<size_t>(kBroadcastArgCount) ||
         fixed_and_source_args == static_cast<size_t>(kBroadcastArgCount + 1))
      << "broadcast_() expects fixed args plus optional src_core";
  return fixed_and_source_args == static_cast<size_t>(kBroadcastArgCount + 1);
}

PrimExpr GetBroadcastSrcCore(const CallNode *call) {
  ICHECK(BroadcastCallHasSrcCore(call))
      << "broadcast_() call does not carry optional src_core";
  size_t index = call->args.size() - 1;
  if (ParseSunmmioOdmaUnitExpr(call->args.back())) {
    --index;
  }
  return call->args[index];
}

bool RegionIntersect(const Region &lhs, const Region &rhs) {
  if (lhs.size() != rhs.size()) {
    return true;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    arith::IntSet lhs_set = arith::IntSet::FromRange(lhs[i]);
    arith::IntSet rhs_set = arith::IntSet::FromRange(rhs[i]);
    if (arith::Intersect({lhs_set, rhs_set}).IsNothing()) {
      return false;
    }
  }
  return true;
}

bool IsRsramBuffer(const Buffer &buffer) {
  return buffer.scope() == kSunmmioScopeRSRAM || buffer.scope() == "local";
}

class BufferLoadCollector : public ExprVisitor {
public:
  void VisitExpr_(const BufferLoadNode *op) final {
    if (IsRsramBuffer(op->buffer)) {
      Region region;
      for (const PrimExpr &index : op->indices) {
        region.push_back(Range::FromMinExtent(index, 1));
      }
      loads.emplace_back(op->buffer, std::move(region));
    }
    ExprVisitor::VisitExpr_(op);
  }

  std::vector<std::pair<Buffer, Region>> loads;
};

enum class AccessProducerKind {
  kAsync,
  kTileLoad,
  kTileStore,
};

struct UnitAccess {
  Buffer buffer;
  Region region;
  bool write{false};
  AccessProducerKind producer_kind{AccessProducerKind::kAsync};
  SunmmioSyncUnits units{kSunmmioSyncNone};
};

bool AccessRegionsMayOverlap(const UnitAccess &lhs, const UnitAccess &rhs) {
  if (!lhs.buffer->data.same_as(rhs.buffer->data)) {
    return false;
  }
  if (lhs.buffer.same_as(rhs.buffer) ||
      StructuralEqual()(lhs.buffer, rhs.buffer)) {
    return RegionIntersect(lhs.region, rhs.region);
  }
  // Different views of the same storage may use unrelated shapes, strides, or
  // element offsets. Treat them as aliases until their physical address ranges
  // can be proven disjoint.
  return true;
}

bool NeedsUnitSync(const UnitAccess &producer, const UnitAccess &consumer) {
  if (producer.producer_kind == AccessProducerKind::kAsync) {
    return true;
  }
  if (consumer.producer_kind != AccessProducerKind::kAsync) {
    return false;
  }
  if (producer.producer_kind == AccessProducerKind::kTileStore) {
    return true;
  }
  return producer.producer_kind == AccessProducerKind::kTileLoad &&
         consumer.write;
}

class InjectUnitSyncRewriter : public StmtMutator {
public:
  Stmt operator()(Stmt body) {
    Stmt rewritten = VisitStmt(body);
    SunmmioSyncUnits pending_links = PendingUnits() & kLinkUnits;
    CompleteUnits(pending_links);
    return AppendSync(std::move(rewritten), pending_links);
  }

private:
  static constexpr SunmmioSyncUnits kTileMemoryUnits =
      kSunmmioSyncVector | kSunmmioSyncRsram;
  static constexpr SunmmioSyncUnits kLinkUnits =
      kSunmmioSyncHlink | kSunmmioSyncVlink;

  static Stmt MakeSync(SunmmioSyncUnits units) {
    ICHECK_NE(units, kSunmmioSyncNone);
    return Evaluate(Call(DataType::Handle(), sunmmio_sync(),
                         {IntImm(DataType::Int(32), units)}));
  }

  static Stmt PrependSync(Stmt stmt, SunmmioSyncUnits units) {
    if (units == kSunmmioSyncNone) {
      return stmt;
    }
    return SeqStmt::Flatten(Array<Stmt>{MakeSync(units), std::move(stmt)});
  }

  static Stmt AppendSync(Stmt stmt, SunmmioSyncUnits units) {
    if (units == kSunmmioSyncNone) {
      return stmt;
    }
    return SeqStmt::Flatten(Array<Stmt>{std::move(stmt), MakeSync(units)});
  }

  static UnitAccess MakeRegionAccess(const BufferRegion &region, bool write,
                                     AccessProducerKind kind,
                                     SunmmioSyncUnits units) {
    return {region->buffer, region->region, write, kind, units};
  }

  static SunmmioSyncUnits OdmaUnits(const CallNode *call) {
    std::optional<SunmmioOdmaUnit> unit = GetSunmmioOdmaUnit(call);
    if (!unit) {
      // ResolveSunmmioUnit runs before this pass in the production pipeline.
      // Standalone pass use remains conservative when the marker is absent.
      return kSunmmioSyncOdma0 | kSunmmioSyncOdma1;
    }
    return *unit == SunmmioOdmaUnit::kOdma0 ? kSunmmioSyncOdma0
                                            : kSunmmioSyncOdma1;
  }

  static bool GetAsyncAccesses(const CallNode *call,
                               std::vector<UnitAccess> *accesses,
                               std::vector<UnitAccess> *argument_loads,
                               bool *completed_inline) {
    *completed_inline = false;
    if (call->op.same_as(dma_copy()) ||
        call->op.same_as(sunmmio_layout_transform()) ||
        call->op.same_as(sunmmio_transpose())) {
      ICHECK_GE(call->args.size(), 2U);
      CollectRegionArgumentLoads(call, {0, 1}, argument_loads);
      SunmmioSyncUnits units = OdmaUnits(call);
      accesses->push_back(
          MakeRegionAccess(NormalizeToBufferRegion(call->args[0]), false,
                           AccessProducerKind::kAsync, units));
      accesses->push_back(
          MakeRegionAccess(NormalizeToBufferRegion(call->args[1]), true,
                           AccessProducerKind::kAsync, units));
      return true;
    }
    if (call->op.same_as(mma_sunmmio())) {
      ICHECK_GE(call->args.size(), 3U);
      CollectRegionArgumentLoads(call, {0, 1, 2}, argument_loads);
      for (int index : {0, 1, 2}) {
        accesses->push_back(
            MakeRegionAccess(NormalizeToBufferRegion(call->args[index]), false,
                             AccessProducerKind::kAsync, kSunmmioSyncTc));
      }
      accesses->push_back(
          MakeRegionAccess(NormalizeToBufferRegion(call->args[2]), true,
                           AccessProducerKind::kAsync, kSunmmioSyncTc));
      return true;
    }
    if (call->op.same_as(broadcast_())) {
      ICHECK_GE(call->args.size(), static_cast<size_t>(kBroadcastArgCount));
      CollectRegionArgumentLoads(call,
                                 {static_cast<size_t>(kBroadcastArgSrc),
                                  static_cast<size_t>(kBroadcastArgDst)},
                                 argument_loads);
      const auto *direction =
          call->args[kBroadcastArgDirection].as<IntImmNode>();
      ICHECK(direction && (direction->value == 0 || direction->value == 1))
          << "tl.broadcast_ requires a constant row/column direction";
      SunmmioSyncUnits units =
          direction->value == 0 ? kSunmmioSyncHlink : kSunmmioSyncVlink;
      accesses->push_back(MakeRegionAccess(
          NormalizeToBufferRegion(call->args[kBroadcastArgSrc]), false,
          AccessProducerKind::kAsync, units));
      accesses->push_back(MakeRegionAccess(
          NormalizeToBufferRegion(call->args[kBroadcastArgDst]), true,
          AccessProducerKind::kAsync, units));
      // SUVM codegen keeps multicast and its link sync in the same source-core
      // branch. Unguarded multicasts remain pending until the post-collective
      // barrier, which completes the link before all participant cores leave.
      *completed_inline = BroadcastCallHasSrcCore(call);
      return true;
    }
    return false;
  }

  static bool GetTileCompoundAccesses(const CallNode *call,
                                      std::vector<UnitAccess> *accesses,
                                      std::vector<UnitAccess> *argument_loads) {
    bool is_pack = call->op.same_as(mx_pack());
    bool is_unpack = call->op.same_as(mx_unpack());
    if (!is_pack && !is_unpack) {
      return false;
    }

    ICHECK_EQ(call->args.size(), 3U);
    CollectRegionArgumentLoads(call, {0, 1, 2}, argument_loads);
    // MX lowering expands to tile loads followed by tile stores. Preserve those
    // access kinds so a later tokenless async operation drains vector/RSRAM.
    for (size_t i = 0; i < call->args.size(); ++i) {
      bool write = is_pack ? i == 2 : i != 0;
      accesses->push_back(
          MakeRegionAccess(NormalizeToBufferRegion(call->args[i]), write,
                           write ? AccessProducerKind::kTileStore
                                 : AccessProducerKind::kTileLoad,
                           kTileMemoryUnits));
    }
    return true;
  }

  static void CollectTileLoads(const PrimExpr &expr,
                               std::vector<UnitAccess> *accesses) {
    BufferLoadCollector collector;
    collector(expr);
    for (const auto &[buffer, region] : collector.loads) {
      accesses->push_back({buffer, region, false, AccessProducerKind::kTileLoad,
                           kTileMemoryUnits});
    }
  }

  static void
  CollectRegionArgumentLoads(const CallNode *call,
                             std::initializer_list<size_t> region_args,
                             std::vector<UnitAccess> *accesses) {
    for (size_t i = 0; i < call->args.size(); ++i) {
      if (std::find(region_args.begin(), region_args.end(), i) ==
          region_args.end()) {
        CollectTileLoads(call->args[i], accesses);
        continue;
      }
      BufferRegion region = NormalizeToBufferRegion(call->args[i]);
      for (const Range &range : region->region) {
        CollectTileLoads(range->min, accesses);
        CollectTileLoads(range->extent, accesses);
      }
    }
  }

  static void CollectRangeLoads(const Region &region,
                                std::vector<UnitAccess> *accesses) {
    for (const Range &range : region) {
      CollectTileLoads(range->min, accesses);
      CollectTileLoads(range->extent, accesses);
    }
  }

  SunmmioSyncUnits RequiredSync(const std::vector<UnitAccess> &accesses) const {
    SunmmioSyncUnits required = kSunmmioSyncNone;
    for (const UnitAccess &consumer : accesses) {
      for (const UnitAccess &producer : pending_) {
        if (!AccessRegionsMayOverlap(producer, consumer) ||
            !NeedsUnitSync(producer, consumer)) {
          continue;
        }
        required |= producer.units;
      }
    }
    return required;
  }

  void CompleteUnits(SunmmioSyncUnits completed) {
    pending_.erase(std::remove_if(pending_.begin(), pending_.end(),
                                  [&](const UnitAccess &access) {
                                    return (completed & access.units) ==
                                           access.units;
                                  }),
                   pending_.end());
  }

  void RecordAccesses(const std::vector<UnitAccess> &accesses) {
    pending_.insert(pending_.end(), accesses.begin(), accesses.end());
  }

  SunmmioSyncUnits ProcessAccesses(const std::vector<UnitAccess> &accesses,
                                   bool completed_inline = false) {
    SunmmioSyncUnits required = RequiredSync(accesses);
    CompleteUnits(required);
    if (!completed_inline) {
      RecordAccesses(accesses);
    }
    return required;
  }

  SunmmioSyncUnits
  SynchronizeBeforeLoads(const std::vector<UnitAccess> &accesses) {
    SunmmioSyncUnits required = RequiredSync(accesses);
    CompleteUnits(required);
    return required;
  }

  SunmmioSyncUnits DrainPending() {
    SunmmioSyncUnits units = PendingUnits();
    CompleteUnits(units);
    return units;
  }

  SunmmioSyncUnits PendingUnits() const {
    SunmmioSyncUnits units = kSunmmioSyncNone;
    for (const UnitAccess &access : pending_) {
      units |= access.units;
    }
    return units;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> sequence;
    for (const Stmt &stmt : op->seq) {
      Stmt rewritten = VisitStmt(stmt);
      if (const auto *nested = rewritten.as<SeqStmtNode>()) {
        for (const Stmt &child : nested->seq) {
          sequence.push_back(child);
        }
      } else {
        sequence.push_back(rewritten);
      }
    }
    return SeqStmt::Flatten(sequence);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call && call->op.same_as(sunmmio_sync())) {
      ICHECK_EQ(call->args.size(), 1U);
      const auto *units = call->args[0].as<IntImmNode>();
      ICHECK(units) << "tl.sunmmio_sync expects a constant unit mask";
      CompleteUnits(static_cast<SunmmioSyncUnits>(units->value));
      return StmtMutator::VisitStmt_(op);
    }

    if (call && call->op.same_as(barrier_arrive_and_wait())) {
      SunmmioSyncUnits completed_links = PendingUnits() & kLinkUnits;
      CompleteUnits(completed_links);
      return PrependSync(StmtMutator::VisitStmt_(op), completed_links);
    }

    std::vector<UnitAccess> accesses;
    std::vector<UnitAccess> argument_loads;
    bool completed_inline = false;
    if (call &&
        GetAsyncAccesses(call, &accesses, &argument_loads, &completed_inline)) {
      SunmmioSyncUnits required = SynchronizeBeforeLoads(argument_loads);
      required |= ProcessAccesses(accesses, completed_inline);
      // Argument expressions are evaluated before the async submission. Keep
      // their memory accesses pending for later statements without introducing
      // a same-statement dependency that could only be synchronized too early.
      RecordAccesses(argument_loads);
      return PrependSync(StmtMutator::VisitStmt_(op), required);
    }

    if (call && GetTileCompoundAccesses(call, &accesses, &argument_loads)) {
      SunmmioSyncUnits required = SynchronizeBeforeLoads(argument_loads);
      required |= ProcessAccesses(accesses);
      RecordAccesses(argument_loads);
      return PrependSync(StmtMutator::VisitStmt_(op), required);
    }

    CollectTileLoads(op->value, &accesses);
    SunmmioSyncUnits required = ProcessAccesses(accesses);
    return PrependSync(StmtMutator::VisitStmt_(op), required);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    std::vector<UnitAccess> accesses;
    for (const PrimExpr &index : op->indices) {
      CollectTileLoads(index, &accesses);
    }
    CollectTileLoads(op->value, &accesses);
    if (op->predicate.defined()) {
      CollectTileLoads(op->predicate.value(), &accesses);
    }
    if (IsRsramBuffer(op->buffer)) {
      Region region;
      for (const PrimExpr &index : op->indices) {
        region.push_back(Range::FromMinExtent(index, 1));
      }
      accesses.push_back({op->buffer, std::move(region), true,
                          AccessProducerKind::kTileStore, kTileMemoryUnits});
    }
    SunmmioSyncUnits required = ProcessAccesses(accesses);
    return PrependSync(StmtMutator::VisitStmt_(op), required);
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    std::vector<UnitAccess> accesses;
    CollectTileLoads(op->value, &accesses);
    SunmmioSyncUnits required = ProcessAccesses(accesses);
    return PrependSync(StmtMutator::VisitStmt_(op), required);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    std::vector<UnitAccess> accesses;
    CollectTileLoads(op->value, &accesses);
    SunmmioSyncUnits required = ProcessAccesses(accesses);
    return PrependSync(StmtMutator::VisitStmt_(op), required);
  }

  Stmt VisitStmt_(const AssertStmtNode *op) final {
    std::vector<UnitAccess> accesses;
    CollectTileLoads(op->condition, &accesses);
    SunmmioSyncUnits required = ProcessAccesses(accesses);
    return PrependSync(StmtMutator::VisitStmt_(op), required);
  }

  Stmt VisitStmt_(const AllocateNode *op) final {
    std::vector<UnitAccess> accesses;
    for (const PrimExpr &extent : op->extents) {
      CollectTileLoads(extent, &accesses);
    }
    CollectTileLoads(op->condition, &accesses);
    SunmmioSyncUnits required = ProcessAccesses(accesses);
    return PrependSync(StmtMutator::VisitStmt_(op), required);
  }

  Stmt VisitStmt_(const BufferRealizeNode *op) final {
    std::vector<UnitAccess> accesses;
    CollectRangeLoads(op->bounds, &accesses);
    CollectTileLoads(op->condition, &accesses);
    SunmmioSyncUnits required = ProcessAccesses(accesses);
    return PrependSync(StmtMutator::VisitStmt_(op), required);
  }

  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    std::vector<UnitAccess> accesses;
    for (const PrimExpr &value : op->iter_values) {
      CollectTileLoads(value, &accesses);
    }
    CollectTileLoads(op->predicate, &accesses);
    SunmmioSyncUnits required = ProcessAccesses(accesses);
    return PrependSync(StmtMutator::VisitStmt_(op), required);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    // A unit sync is safe on a path that did not submit work. Draining at
    // structured-control-flow boundaries avoids recreating token SSA while
    // preserving correctness for path-dependent pending work.
    SunmmioSyncUnits prefix = DrainPending();

    pending_.clear();
    std::vector<UnitAccess> condition_loads;
    CollectTileLoads(op->condition, &condition_loads);
    RecordAccesses(condition_loads);
    std::vector<UnitAccess> branch_entry = pending_;

    Stmt then_case = VisitStmt(op->then_case);
    then_case = AppendSync(std::move(then_case), DrainPending());

    Stmt else_case;
    if (op->else_case.defined()) {
      pending_ = branch_entry;
      else_case = VisitStmt(op->else_case.value());
      else_case = AppendSync(std::move(else_case), DrainPending());
    } else {
      pending_ = branch_entry;
      SunmmioSyncUnits false_path_units = DrainPending();
      if (false_path_units != kSunmmioSyncNone) {
        else_case = MakeSync(false_path_units);
      }
    }
    pending_.clear();
    Stmt rewritten = IfThenElse(op->condition, then_case, else_case, op->span);
    return PrependSync(std::move(rewritten), prefix);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Conservatively drain at entry and at each backedge. Straight-line code
    // remains dependency-driven; loop overlap can be recovered later with a
    // dedicated unit-aware loop-carried analysis.
    SunmmioSyncUnits prefix = DrainPending();
    pending_.clear();
    std::vector<UnitAccess> header_loads;
    CollectTileLoads(op->min, &header_loads);
    CollectTileLoads(op->extent, &header_loads);
    if (op->step.has_value()) {
      CollectTileLoads(op->step.value(), &header_loads);
    }
    RecordAccesses(header_loads);
    SunmmioSyncUnits header_units = PendingUnits();
    Stmt body = VisitStmt(op->body);
    body = AppendSync(std::move(body), DrainPending());
    pending_.clear();
    Stmt loop = For(op->loop_var, op->min, op->extent, op->kind, body,
                    op->thread_binding, op->annotations, op->step, op->span);
    loop = AppendSync(std::move(loop), header_units);
    return PrependSync(std::move(loop), prefix);
  }

  Stmt VisitStmt_(const WhileNode *op) final {
    SunmmioSyncUnits prefix = DrainPending();
    pending_.clear();
    std::vector<UnitAccess> condition_loads;
    CollectTileLoads(op->condition, &condition_loads);
    RecordAccesses(condition_loads);
    SunmmioSyncUnits condition_units = PendingUnits();
    Stmt body = VisitStmt(op->body);
    body = AppendSync(std::move(body), DrainPending());
    pending_.clear();
    Stmt loop = While(op->condition, body, op->span);
    loop = AppendSync(std::move(loop), condition_units);
    return PrependSync(std::move(loop), prefix);
  }

  std::vector<UnitAccess> pending_;
};

class InjectBroadcastBarriersRewriter : public StmtMutator {
public:
  InjectBroadcastBarriersRewriter(const Target &target,
                                  arith::Analyzer *analyzer)
      : analyzer_(analyzer) {
    SunmmioMeshConfig mesh = GetSunmmioMeshConfig(target);
    mesh_nrow_ = mesh.nrow;
    mesh_ncol_ = mesh.ncol;
    ICHECK_LE(mesh_nrow_ * mesh_ncol_, 64)
        << "tl.broadcast_ barrier mask supports at most 64 cores";
  }

private:
  PrimExpr LocalMaskBitSet(PrimExpr local_mask, int local_index) const {
    PrimExpr bit = I64Imm(static_cast<int64_t>(uint64_t{1} << local_index));
    return (AsI64(local_mask) & bit) != I64Imm(0);
  }

  std::optional<int64_t>
  TryExpandBroadcastLocalMaskImm(const PrimExpr &local_mask, int direction,
                                 const PrimExpr &src_core) const {
    std::optional<int64_t> local_value = EvalInt64(local_mask, analyzer_);
    std::optional<int64_t> src_value = EvalInt64(src_core, analyzer_);
    if (!local_value || !src_value) {
      return std::nullopt;
    }

    int total_cores = mesh_nrow_ * mesh_ncol_;
    ICHECK_GE(*src_value, 0);
    ICHECK_LT(*src_value, total_cores);
    int src_row = static_cast<int>(*src_value) / mesh_ncol_;
    int src_col = static_cast<int>(*src_value) % mesh_ncol_;
    int axis_len = direction == 0 ? mesh_ncol_ : mesh_nrow_;
    uint64_t valid_local_mask =
        axis_len == 64 ? ~uint64_t{0} : ((uint64_t{1} << axis_len) - 1);
    uint64_t local = UnsignedMask(*local_value);
    ICHECK_EQ(local & ~valid_local_mask, 0U)
        << "tl.broadcast_ direction-local mask has bits outside the active "
           "mesh axis";

    uint64_t global = 0;
    if (direction == 0) {
      for (int col = 0; col < mesh_ncol_; ++col) {
        if ((local & (uint64_t{1} << col)) != 0) {
          global |= uint64_t{1} << (src_row * mesh_ncol_ + col);
        }
      }
    } else {
      for (int row = 0; row < mesh_nrow_; ++row) {
        if ((local & (uint64_t{1} << row)) != 0) {
          global |= uint64_t{1} << (row * mesh_ncol_ + src_col);
        }
      }
    }
    return static_cast<int64_t>(global);
  }

  PrimExpr ExpandBroadcastLocalMask(const PrimExpr &local_mask, int direction,
                                    const PrimExpr &src_core) const {
    if (std::optional<int64_t> imm =
            TryExpandBroadcastLocalMaskImm(local_mask, direction, src_core)) {
      return I64Imm(*imm);
    }

    PrimExpr src_core_i64 = AsI64(src_core);
    PrimExpr ncol = I64Imm(mesh_ncol_);
    PrimExpr src_row = floordiv(src_core_i64, ncol);
    PrimExpr src_col = floormod(src_core_i64, ncol);
    PrimExpr global_mask = I64Imm(0);
    if (direction == 0) {
      for (int col = 0; col < mesh_ncol_; ++col) {
        PrimExpr bit = CoreBitMask(src_row * ncol + I64Imm(col));
        global_mask = Select(LocalMaskBitSet(local_mask, col),
                             AsI64(global_mask) | AsI64(bit), global_mask);
      }
    } else {
      ICHECK_EQ(direction, 1)
          << "tl.broadcast_ local mask expansion only supports direction 0/1";
      for (int row = 0; row < mesh_nrow_; ++row) {
        PrimExpr bit = CoreBitMask(I64Imm(row * mesh_ncol_) + src_col);
        global_mask = Select(LocalMaskBitSet(local_mask, row),
                             AsI64(global_mask) | AsI64(bit), global_mask);
      }
    }
    return analyzer_ ? analyzer_->Simplify(global_mask) : global_mask;
  }

  PrimExpr GetBroadcastBarrierSrcCore(const CallNode *call) const {
    if (BroadcastCallHasSrcCore(call)) {
      return GetBroadcastSrcCore(call);
    }
    ICHECK(current_kernel_core_id_.defined())
        << "tl.broadcast_ without optional src_core requires an enclosing "
           "blockIdx.x binding";
    return current_kernel_core_id_;
  }

  BarrierMaskInfo GetBroadcastBarrierMask(const CallNode *call) const {
    const auto *direction_imm =
        call->args[kBroadcastArgDirection].as<IntImmNode>();
    ICHECK(direction_imm &&
           (direction_imm->value == 0 || direction_imm->value == 1))
        << "tl.broadcast_ barrier mask expansion only supports row or column "
           "broadcasts";
    int direction = static_cast<int>(direction_imm->value);
    PrimExpr src_core = GetBroadcastBarrierSrcCore(call);
    PrimExpr write_mask = ExpandBroadcastLocalMask(
        call->args[kBroadcastArgMask], direction, src_core);

    BarrierMaskInfo info;
    info.expr = AsI64(CoreBitMask(src_core)) | AsI64(write_mask);
    if (analyzer_) {
      info.expr = analyzer_->Simplify(info.expr);
    }
    if (!info.expr.as<IntImmNode>()) {
      info.candidates = EnumerateMaskCandidates(
          info.expr, direction, mesh_nrow_, mesh_ncol_, analyzer_);
      ICHECK(!info.candidates.empty())
          << "Could not derive static candidate masks for dynamic "
             "tl.broadcast_ barrier mask";
    }
    return info;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call || !call->op.same_as(broadcast_())) {
      return StmtMutator::VisitStmt_(op);
    }
    BarrierMaskInfo participant_mask = GetBroadcastBarrierMask(call);
    Array<Stmt> stmts;
    stmts.push_back(Evaluate(Call(DataType::Handle(), barrier_arrive_and_wait(),
                                  MakeBarrierArgs(participant_mask))));
    stmts.push_back(StmtMutator::VisitStmt_(op));
    stmts.push_back(Evaluate(Call(DataType::Handle(), barrier_arrive_and_wait(),
                                  MakeBarrierArgs(participant_mask))));
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    PrimExpr old_kernel_core_id = current_kernel_core_id_;
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "blockIdx.x") {
        current_kernel_core_id_ = iv->var;
      }
    }
    Stmt stmt = StmtMutator::VisitStmt_(op);
    current_kernel_core_id_ = old_kernel_core_id;
    return stmt;
  }

  int mesh_nrow_{0};
  int mesh_ncol_{0};
  arith::Analyzer *analyzer_;
  PrimExpr current_kernel_core_id_;
};

class InitReusableBarriersRewriter : public StmtMutator {
public:
  Stmt operator()(Stmt body) {
    Stmt rewritten = VisitStmt(body);
    return HasThreadExtent(rewritten) ? rewritten
                                      : PrependBarrierInits(rewritten, {});
  }

private:
  class ThreadExtentFinder : public StmtVisitor {
  public:
    void VisitStmt_(const AttrStmtNode *op) final {
      if (op->attr_key == tir::attr::thread_extent) {
        found = true;
        return;
      }
      StmtVisitor::VisitStmt_(op);
    }

    bool found{false};
  };

  class BarrierMaskCollector : public StmtExprVisitor {
  public:
    explicit BarrierMaskCollector(std::vector<Var> scoped_vars)
        : scoped_vars_(std::move(scoped_vars)) {}

    void VisitStmt_(const ForNode *op) final {
      scoped_vars_.push_back(op->loop_var);
      StmtExprVisitor::VisitStmt_(op);
      scoped_vars_.pop_back();
    }

    void VisitStmt_(const LetStmtNode *op) final {
      scoped_vars_.push_back(op->var);
      StmtExprVisitor::VisitStmt_(op);
      scoped_vars_.pop_back();
    }

    void VisitStmt_(const EvaluateNode *op) final {
      if (const auto *call = op->value.as<CallNode>()) {
        if (call->op.same_as(barrier_arrive_and_wait()) &&
            !call->args.empty()) {
          BarrierMaskInfo info = BarrierMaskInfoFromArgs(call->args);
          ICHECK(!info.candidates.empty() ||
                 !ExprUsesAnyVar(info.expr, scoped_vars_))
              << "dynamic barrier mask depends on a local control-flow "
                 "variable and cannot be initialized in the enclosing entry "
                 "block required by suvm.barrier.init";
          AddUniqueBarrierMaskInfo(&masks, info);
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }

    std::vector<BarrierMaskInfo> masks;

  private:
    std::vector<Var> scoped_vars_;
  };

  static bool HasThreadExtent(const Stmt &body) {
    ThreadExtentFinder finder;
    finder(body);
    return finder.found;
  }

  static Stmt PrependBarrierInits(const Stmt &body,
                                  const std::vector<Var> &scoped_vars) {
    BarrierMaskCollector collector(scoped_vars);
    collector(body);
    if (collector.masks.empty()) {
      return body;
    }

    Array<Stmt> stmts;
    for (const BarrierMaskInfo &mask : collector.masks) {
      stmts.push_back(Evaluate(
          Call(DataType::Handle(), barrier_init(), MakeBarrierInitArgs(mask))));
    }
    if (const auto *seq = body.as<SeqStmtNode>()) {
      for (const Stmt &stmt : seq->seq) {
        stmts.push_back(stmt);
      }
    } else {
      stmts.push_back(body);
    }
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != tir::attr::thread_extent) {
      return StmtMutator::VisitStmt_(op);
    }
    Stmt body = StmtMutator::VisitStmt(op->body);
    if (HasThreadExtent(body)) {
      return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
    }
    return AttrStmt(op->node, op->attr_key, op->value,
                    PrependBarrierInits(body, scoped_vars_), op->span);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    scoped_vars_.push_back(op->loop_var);
    Stmt body = StmtMutator::VisitStmt(op->body);
    scoped_vars_.pop_back();
    if (body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    }
    return For(op->loop_var, op->min, op->extent, op->kind, body,
               op->thread_binding, op->annotations, op->step, op->span);
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    scoped_vars_.push_back(op->var);
    Stmt body = StmtMutator::VisitStmt(op->body);
    scoped_vars_.pop_back();
    if (body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    }
    return LetStmt(op->var, op->value, body, op->span);
  }

  std::vector<Var> scoped_vars_;
};

PrimFunc RewriteSunmmioSync(PrimFunc func, arith::Analyzer *analyzer) {
  Target target = func->GetAttr<Target>(tvm::attr::kTarget).value();
  InjectBroadcastBarriersRewriter inject_barriers(target, analyzer);
  func.CopyOnWrite()->body = inject_barriers(func->body);
  InjectUnitSyncRewriter inject_unit_sync;
  func.CopyOnWrite()->body = inject_unit_sync(func->body);
  InitReusableBarriersRewriter init_barriers;
  func.CopyOnWrite()->body = init_barriers(func->body);
  return func;
}

} // namespace

tvm::transform::Pass InjectSunmmioSync() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const PassContext &) -> PrimFunc {
    if (!func->HasNonzeroAttr(tir::attr::kIsGlobalFunc)) {
      return func;
    }
    arith::Analyzer analyzer;
    return RewriteSunmmioSync(std::move(func), &analyzer);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectSunmmioSync", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectSunmmioSync", InjectSunmmioSync);
}

} // namespace tl
} // namespace tvm
