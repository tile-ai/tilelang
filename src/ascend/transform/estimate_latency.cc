/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
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
 * \file estimate_latency.cc
 * \brief Ascend task latency and initiation-interval estimation pass.
 */

#include "ascend/transform/source_scopes.h"
#include "ir/analyzer.h"
#include "layout/expr_utils.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tvm/ffi/extra/base.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "ascend/op/ascend_mte_plan.h"
#include "ascend/op/builtin.h"
#include "ascend/op/copy.h"
#include "ascend/op/nd2nz.h"
#include "ascend/op/utils.h"
#include "ascend/transform/attr.h"
#include "ascend/transform/auto_schedule/ir_structure.h"
#include "ascend/transform/auto_schedule/kernel_rewriter.h"
#include "ascend/transform/auto_schedule/task_analysis.h"
#include "ascend/transform/auto_schedule/task_annotations.h"
#include "ascend/transform/estimate_latency.h"
#include "ir/annotations.h"
#include "op/copy.h"
#include "op/gemm.h"
#include "op/gemm_blockscaled.h"

#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;
using namespace ffi;
using namespace tirx::transform;
using namespace ascend;
using ffi::GetRef;
using ffi::String;

namespace {

// Task cost model.

struct PerCoreTaskCandidate {
  Stmt marker;
  std::shared_ptr<TaskNode> task;
};

enum class CopyGeometry {
  kUnknown,
  kM,
  kN,
};

struct MteGeometry {
  CopyGeometry kind{CopyGeometry::kUnknown};
  int64_t contiguous_bytes_lower_bound{0};
};

using Hf32ModeMap =
    std::unordered_map<Call, uint8_t, ObjectPtrHash, ObjectPtrEqual>;

// Track the mode reaching each GEMM before estimating individual Task markers.
// Keep branch conditions in the state: MaterializeScheduleUnits can split a
// guarded mode setter and GEMM into separate, identically guarded tasks.
// Loop backedges join possible modes; an ambiguous mode uses the FP32 cost.
class Hf32ModeAnalyzer : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  using Result = ffi::Optional<ffi::VisitInterrupt>;
  static constexpr uint8_t kDisabled = 1;
  static constexpr uint8_t kEnabled = 2;

  static Hf32ModeMap Analyze(const Stmt &body) {
    auto analyzer = make_object<Hf32ModeAnalyzer>();
    analyzer->Visit(body);
    return std::move(analyzer->modes_);
  }

private:
  static PrimExpr Mode(uint8_t modes) {
    return IntImm(PrimType::Int(32), modes);
  }

  uint8_t PossibleModes() {
    if (analyzer_->CanProve(state_ == kDisabled))
      return kDisabled;
    if (analyzer_->CanProve(state_ == kEnabled))
      return kEnabled;
    return kDisabled | kEnabled;
  }

  Result Visit_(const CallObj *op) final {
    if (op->op.same_as(tl::ascend_set_hf32_mode())) {
      const auto args = DecodeCallOperands(op);
      ICHECK_EQ(args.size(), 1);
      const auto *mode = args[0].as<IntImmObj>();
      ICHECK(mode && mode->value.as<int64_t>().value() >= 0 &&
             mode->value.as<int64_t>().value() <= 2)
          << "HF32 mode must be a constant in {0, 1, 2}";
      state_ =
          Mode(mode->value.as<int64_t>().value() == 0 ? kDisabled : kEnabled);
    } else if (op->op.same_as(Op::Get("tl.tileop.gemm")) ||
               op->op.same_as(Op::Get("tl.tileop.gemm_blockscaled"))) {
      modes_[GetRef<Call>(op)] |= PossibleModes();
    }
    StmtExprVisitor::Visit_(op);
    return std::nullopt;
  }

  Result Visit_(const BindObj *op) final {
    Expr value = IsScheduledBind(op)
                     ? ReadScheduledBinding(op->value.as<CallObj>()).value
                     : op->value;
    Visit(value);
    if (auto primitive = value.as<PrimExpr>();
        primitive && primitive->ty().IsScalar()) {
      auto freshener = make_object<FreshenMutableReads>();
      BindAnalyzer(analyzer_, op->var, (*freshener)(*primitive), true);
    }
    return std::nullopt;
  }

  void VisitConditional(const PrimExpr &condition, const Stmt &then_case,
                        const Optional<SeqStmt> &else_case = std::nullopt) {
    Visit(condition);
    PrimExpr guard =
        analyzer_->Simplify((*make_object<FreshenMutableReads>())(condition));
    if (analyzer_->CanProve(guard)) {
      Visit(then_case);
      return;
    }
    if (analyzer_->CanProve(Not(guard))) {
      if (else_case)
        Visit(else_case.value());
      return;
    }
    PrimExpr incoming = state_;
    PrimExpr then_out = state_;
    {
      ScopedAnalyzerConstraints context(analyzer_, guard);
      state_ = analyzer_->Simplify(incoming);
      Visit(then_case);
      then_out = state_;
    }
    {
      ScopedAnalyzerConstraints context(analyzer_, Not(guard));
      state_ = analyzer_->Simplify(incoming);
      if (else_case)
        Visit(else_case.value());
    }
    state_ = analyzer_->Simplify(Select(guard, then_out, state_));
  }

  Result Visit_(const IfObj *op) final {
    VisitConditional(op->condition, op->then_case, op->else_case);
    return std::nullopt;
  }

  void AnalyzeLoop(const Stmt &body, bool must_execute) {
    // Forget iteration-local predicates at the backedge: the same loop Var
    // and Bind nodes represent different values on the next iteration.
    uint8_t incoming = PossibleModes();
    uint8_t header = incoming;
    // There are only two mode bits, so the ascending state chain converges
    // after at most three visits, including the final unchanged state.
    for (int i = 0; i < 3; ++i) {
      state_ = Mode(header);
      Visit(body);
      uint8_t body_out = PossibleModes();
      uint8_t next = incoming | body_out;
      if (next == header) {
        state_ = Mode(must_execute ? body_out : header);
        return;
      }
      header = next;
    }
    LOG(FATAL) << "HF32 mode analysis failed to converge";
  }

  Result Visit_(const ForObj *op) final {
    Visit(op->begin);
    Visit(op->end);
    Visit(op->step);
    auto freshener = make_object<FreshenMutableReads>();
    PrimExpr begin = (*freshener)(op->begin), end = (*freshener)(op->end);
    PrimExpr step = (*freshener)(op->step);
    // The native loop always tests var < end, even for a negative step.
    if (analyzer_->CanProve(begin >= end))
      return std::nullopt;
    PrimExpr distance =
        cast(end, PrimType::Int(64)) - cast(begin, PrimType::Int(64));
    if (analyzer_->CanProve(step > 0) &&
        analyzer_->CanProve(distance <= cast(step, PrimType::Int(64)))) {
      Visit(op->body);
      return std::nullopt;
    }
    AnalyzeLoop(op->body, analyzer_->CanProve(begin < end));
    return std::nullopt;
  }

  Result Visit_(const WhileObj *op) final {
    Visit(op->condition);
    AnalyzeLoop(op->body, false);
    return std::nullopt;
  }

  Result Visit_(const RegionStmtObj *op) final {
    // Block init/predicate logic is explicit native control flow in body.
    // The metadata payload is not an executed setter or GEMM.
    if (!IsAscendBlock(op) && !IsSemanticTask(op) && !IsScheduleUnit(op) &&
        !IsStageScope(op) && !ReadSourceAttribute(op))
      for (const Expr &arg : op->args)
        Visit(arg);
    return Visit(op->body);
    return std::nullopt;
  }

  sym::Analyzer analyzer_;
  PrimExpr state_{Mode(kDisabled)};
  Hf32ModeMap modes_;
};

// Estimator-only operation details. They deliberately do not live on
// AutoSchedule's TaskNode: the scheduler consumes the resulting latency/II,
// not the copy descriptors or Cube shapes used to derive them.
struct TaskCostFeatures {
  struct CubeShape {
    int64_t m;
    int64_t n;
    int64_t k;
    PrimType input_dtype;
    bool hf32;
    bool blockscaled;
  };

  struct CopyInfo {
    String src_scope;
    String dst_scope;
    // One physical AIV's payload. Unrewritten dual copies are normalized to
    // the same representation while collecting their full/half regions.
    int64_t bytes;
    int dual_dst_ctl;
    // Physical MTE access pattern inferred from coalesced endpoint layouts.
    CopyGeometry geometry;
    // Proven lower bound for one AIV's coalesced innermost row. Zero means
    // that a dynamic tail could be narrower at runtime.
    int64_t contiguous_bytes_lower_bound;
    // ND2NZ post-copy traverses physical NZ D-groups rather than logical ND
    // rows, so it does not use raw N-strided row-width bandwidths.
    bool is_nd2nz_post_copy;
  };

  std::vector<CubeShape> cube_shapes;
  std::vector<CopyInfo> copy_infos;
};

// Internal cost model for Ascend AutoSchedule tasks. The public transform pass
// is implemented separately in estimate_latency.cc.
class TaskCostEstimator {
public:
  explicit TaskCostEstimator(Hf32ModeMap hf32_modes)
      : hf32_modes_(std::move(hf32_modes)) {}

  void SetAivLanes(int lanes) { aiv_lanes_ = lanes; }

  // Estimate latency and initiation interval for one Ascend task.
  void Estimate(TaskNode *task);

  // Estimate one canonical Task marker from its statement structure.
  // Sequential statements add, mutually-exclusive branches take their
  // maximum. Unsynchronized sequencing exposes each preceding statement's II
  // and keeps the latest completion time across all statements; serial loops
  // apply the same rule across iterations. Parallel loops and VF blocks remain
  // atomic and use Estimate(TaskNode*) directly.
  void EstimateComposedTask(TaskNode *task);

  // Estimate a PerCoreTask from canonical Task candidate markers.
  // Candidate bodies are serial, candidates within one inter-core phase run in
  // parallel across cores, and cross-core waits delimit serial phases.
  void EstimatePerCoreTask(TaskNode *per_core_task, const Stmt &body,
                           const std::vector<PerCoreTaskCandidate> &candidates);

private:
  struct SerialTaskCost {
    int64_t latency{0};
    int64_t ii{0};
  };

  static int64_t CheckedSerialTaskAdd(int64_t lhs, int64_t rhs,
                                      const char *field) {
    ICHECK_GE(lhs, 0) << "Serial task " << field << " must be non-negative";
    ICHECK_GE(rhs, 0) << "Serial task " << field << " must be non-negative";
    ICHECK_LE(lhs, std::numeric_limits<int64_t>::max() - rhs)
        << "Serial task " << field << " overflow while aggregating operations";
    return lhs + rhs;
  }

  static int64_t CheckedSerialTaskMultiply(int64_t value, int64_t factor,
                                           const char *field) {
    ICHECK_GE(value, 0) << "Serial task " << field << " must be non-negative";
    ICHECK_GE(factor, 0) << "Serial task trip count must be non-negative";
    if (value == 0 || factor == 0)
      return 0;
    ICHECK_LE(value, std::numeric_limits<int64_t>::max() / factor)
        << "Serial task " << field << " overflow while multiplying by loop "
        << "trip count " << factor;
    return value * factor;
  }

  SerialTaskCost EstimateStmtCost(const Stmt &stmt, const ConstrSet &outer_ctx);
  SerialTaskCost EstimateLeafTaskCost(const Stmt &stmt,
                                      const ConstrSet &outer_ctx);

  AscendLatencyParams params_;
  Hf32ModeMap hf32_modes_;
  int aiv_lanes_{2};

  // ====================================================================
  // Every divisor is guarded so an unknown bandwidth/throughput contributes
  // zero rather than causing a divide-by-zero.
  //
  // The estimator is structured by physical resource:
  //   1. Memory latency: walk read/write regions, classify (src,dst) scope
  //      pair to a physical path, charge `base + bytes/bandwidth`.
  //   2. Compute latency:
  //        Cube   -> (M*K*N) / cube_throughput_<dtype> + base
  //        Vector -> max(op-count via OperationCounter,
  //                      bytes / valu_bandwidth)
  //        Scalar -> one cycle per atomic task
  //   3. II by pipe: take the max of the bandwidth- and throughput-limited
  //      initiation intervals for each active physical resource.
  //   4. Total latency = memory_latency + compute_latency
  //      under sequential assumption; pipeline overlap is handled by the
  //      scheduler (Z3) via the II we emit.
  // ====================================================================

  static int64_t safe_div_ceil(int64_t numer, int64_t denom) {
    return denom > 0 ? (numer + denom - 1) / denom : 0;
  }

  static int64_t ScaleBytesByLanes(int64_t bytes, int64_t lanes) {
    if (bytes <= 0 || lanes <= 0)
      return 0;
    if (bytes > std::numeric_limits<int64_t>::max() / lanes)
      return std::numeric_limits<int64_t>::max();
    return bytes * lanes;
  }

  // Read `tl.vf_latency` annotation from SIMT_VF / SIMD_VF blocks within
  // the task. Returns 0 when no annotation is present.
  static int64_t ReadVFLatencyFromBlocks(const TaskNode *task) {
    struct VFLatencyReader : public StmtExprVisitor {
      using StmtExprVisitor::Visit;
      using StmtExprVisitor::Visit_;
      using Result = ffi::Optional<ffi::VisitInterrupt>;
      Result Visit_(const RegionStmtObj *op) override {
        if (auto name = BlockNameAtSourceBoundary(op);
            name && (*name == "SIMT_VF" || *name == "SIMD_VF")) {
          ScheduleUnitMetadata annotations;
          if (IsAscendBlock(op)) {
            annotations = AscendBlockAnnotations(op);
          } else {
            auto fields =
                ReadSourceAttribute(op)->value.cast<ScheduleUnitMetadata>();
            if (auto value = fields.Get("annotations"))
              annotations = value->cast<ScheduleUnitMetadata>();
          }
          if (auto value = annotations.Get("tl.vf_latency")) {
            auto amount = value->try_cast<int64_t>();
            if (!amount)
              amount = TryConstInt(value->cast<PrimExpr>());
            if (amount)
              latency = std::max(latency, *amount);
          }
        }
        return Visit(op->body);
        return std::nullopt;
      }
      int64_t latency{0};
    };
    auto reader = make_object<VFLatencyReader>();
    reader->Visit(task->stmt);
    return reader->latency;
  }

  // Classification of (scope -> scope) for an MTE-style copy. Matches the
  // PipeMask inferred from the task body, but lets
  // EstimateAscend pick the *specific* physical port (e.g. L1->L0A vs
  // L1->L0B) from the region's buffer.
  enum class AscendPath {
    kUnknown,
    kGmToL1,    // MTE2 (AIC)
    kGmToUb,    // MTE2 (AIV)
    kUbToGm,    // MTE3
    kUbToL1,    // MTE3 (cross-core)
    kL1ToL0a,   // MTE1
    kL1ToL0b,   // MTE1
    kL1ToL0Sf,  // MTE1, shared by A/B scale-factor loads
    kL1ToBt,    // MTE1
    kL1ToFpBuf, // MTE1
    kL0cToUb,   // Fixpipe
    kL0cToGm,   // Fixpipe
  };

  // Pick the bandwidth (bytes/cycle) for a physical path. Returns 0 when
  // the path or the parameter is unknown — `safe_div` then collapses that
  // term to zero in the formulas below.
  int64_t BandwidthForPath(AscendPath path) const {
    switch (path) {
    case AscendPath::kGmToL1:
      return params_.aic_mte2_to_l1_bandwidth;
    case AscendPath::kGmToUb:
      return params_.mte2_gm_to_ub_bandwidth;
    case AscendPath::kUbToGm:
      return params_.mte3_ub_to_gm_bandwidth;
    case AscendPath::kUbToL1:
      return params_.mte3_ub_to_l1_bandwidth;
    case AscendPath::kL1ToL0a:
      return params_.l1_to_l0a_bandwidth;
    case AscendPath::kL1ToL0b:
      return params_.l1_to_l0b_bandwidth;
    case AscendPath::kL1ToL0Sf:
      return params_.l1_to_l0_sf_bandwidth;
    case AscendPath::kL1ToBt:
      return params_.l1_to_bt_bandwidth;
    case AscendPath::kL1ToFpBuf:
      return params_.l1_to_fp_buf_bandwidth;
    case AscendPath::kL0cToUb:
      return params_.fixpipe_bandwidth;
    case AscendPath::kL0cToGm:
      return params_.fixpipe_bandwidth;
    default:
      return 0;
    }
  }

  static bool IsDualCopy(const TaskCostFeatures::CopyInfo &info) {
    return info.dual_dst_ctl == 1 || info.dual_dst_ctl == 2;
  }

  static bool IsAivMtePath(AscendPath path) {
    return path == AscendPath::kGmToUb || path == AscendPath::kUbToGm ||
           path == AscendPath::kUbToL1;
  }

  static int64_t ScaleBandwidth(int64_t bytes, int64_t numerator,
                                int64_t denominator) {
    if (bytes <= 0 || numerator <= 0 || denominator <= 0)
      return 0;
    if (bytes > std::numeric_limits<int64_t>::max() / numerator)
      return std::numeric_limits<int64_t>::max() / denominator;
    return std::max<int64_t>(bytes * numerator / denominator, 1);
  }

  // A strided N-split row of `width` bytes travels in whole
  // `transaction_bytes`-sized MTE transactions. The row rate peaks
  // at whole transaction multiples and scales with the occupied fraction
  // between them:
  //   rate = peak * width / (transaction_bytes * ceil(width / transaction)).
  static int64_t TransactionBandwidth(int64_t width, int64_t transaction_bytes,
                                      int64_t peak_bandwidth) {
    if (width <= 0 || transaction_bytes <= 0 || peak_bandwidth <= 0)
      return 0;
    int64_t transactions = (width + transaction_bytes - 1) / transaction_bytes;
    if (width > std::numeric_limits<int64_t>::max() / peak_bandwidth)
      return std::numeric_limits<int64_t>::max();
    return std::max<int64_t>(
        width * peak_bandwidth / (transaction_bytes * transactions), 1);
  }

  // AIV MTE copies use one physical model regardless of whether their TIR
  // originated from T.copy or T.dual_copy. M-like regions coalesce; N-like
  // regions retain a row stride whose proven contiguous width selects the
  // transaction-limited rate.
  int64_t BandwidthForCopy(AscendPath path,
                           const TaskCostFeatures::CopyInfo &info) const {
    int64_t width = info.contiguous_bytes_lower_bound;
    switch (path) {
    case AscendPath::kGmToUb: {
      int64_t shared_bandwidth = params_.mte2_gm_to_ub_bandwidth;
      if (info.geometry == CopyGeometry::kM)
        return shared_bandwidth;
      if (width <= 0)
        return params_.mte2_gm_to_ub_n_unknown_bandwidth;
      if (width < params_.mte2_gm_to_ub_n_transaction_bytes)
        return params_.mte2_gm_to_ub_n_plateau_bandwidth;
      if (width >= 2 * params_.mte2_gm_to_ub_n_transaction_bytes)
        return std::min(shared_bandwidth,
                        params_.mte2_gm_to_ub_n_wide_bandwidth);
      return std::min(
          shared_bandwidth,
          TransactionBandwidth(width, params_.mte2_gm_to_ub_n_transaction_bytes,
                               params_.mte2_gm_to_ub_n_peak_bandwidth));
    }
    case AscendPath::kUbToGm: {
      int64_t shared_bandwidth = params_.mte3_ub_to_gm_bandwidth;
      if (info.geometry == CopyGeometry::kM)
        return shared_bandwidth;
      if (width <= 0)
        return params_.mte3_ub_to_gm_n_unknown_bandwidth;
      if (width < params_.mte3_ub_to_gm_n_transaction_bytes)
        return params_.mte3_ub_to_gm_n_plateau_bandwidth;
      if (width >= 4 * params_.mte3_ub_to_gm_n_transaction_bytes)
        return shared_bandwidth;
      if (width >= 2 * params_.mte3_ub_to_gm_n_transaction_bytes)
        return std::min(shared_bandwidth,
                        params_.mte3_ub_to_gm_n_wide_bandwidth);
      return std::min(
          shared_bandwidth,
          TransactionBandwidth(width, params_.mte3_ub_to_gm_n_transaction_bytes,
                               params_.mte3_ub_to_gm_n_peak_bandwidth));
    }
    case AscendPath::kUbToL1:
      if (info.is_nd2nz_post_copy || info.geometry == CopyGeometry::kM)
        return params_.mte3_ub_to_l1_bandwidth;
      if (width >= params_.mte3_ub_to_l1_raw_n_full_width_bytes)
        return std::min(params_.mte3_ub_to_l1_bandwidth, width);
      if (width <= 0)
        return params_.mte3_ub_to_l1_raw_n_unknown_bandwidth;
      return std::min(
          params_.mte3_ub_to_l1_bandwidth,
          ScaleBandwidth(width, params_.mte3_ub_to_l1_raw_n_width_numerator,
                         params_.mte3_ub_to_l1_raw_n_width_denominator));
    case AscendPath::kL0cToUb: {
      if (!IsDualCopy(info))
        return 0;
      int64_t aiv_bandwidth =
          width > 0 ? ScaleBandwidth(width, params_.fixpipe_dual_width_scale, 1)
                    : params_.fixpipe_dual_unknown_width_bandwidth;
      return std::min(params_.fixpipe_dual_bandwidth, aiv_bandwidth);
    }
    default:
      return 0;
    }
  }

  int64_t BaseLatencyForCopy(AscendPath path,
                             const TaskCostFeatures::CopyInfo &info) const {
    switch (path) {
    case AscendPath::kGmToUb:
      return info.geometry == CopyGeometry::kM
                 ? params_.mte2_gm_to_ub_base_latency
                 : params_.mte2_gm_to_ub_n_base_latency;
    case AscendPath::kUbToGm:
      if (info.geometry != CopyGeometry::kM &&
          (info.contiguous_bytes_lower_bound <= 0 ||
           info.contiguous_bytes_lower_bound < 128)) {
        return params_.mte3_ub_to_gm_narrow_base_latency;
      }
      return params_.mte3_ub_to_gm_base_latency;
    case AscendPath::kUbToL1:
      return params_.mte3_ub_to_l1_base_latency;
    case AscendPath::kL0cToUb:
      return IsDualCopy(info) ? params_.fixpipe_dual_base_latency : 0;
    default:
      return 0;
    }
  }

  // Latency (cycles) for moving `bytes` over `path`.
  int64_t MemLatencyForPath(AscendPath path, int64_t bytes,
                            int64_t bandwidth_override = 0,
                            int64_t base_latency_override = 0) const {
    if (bandwidth_override > 0) {
      int64_t base = base_latency_override > 0 ? base_latency_override
                                               : BaseLatencyForPath(path);
      return base + safe_div_ceil(bytes, bandwidth_override);
    }
    return BaseLatencyForPath(path) +
           safe_div_ceil(bytes, BandwidthForPath(path));
  }

  int64_t BaseLatencyForPath(AscendPath path) const {
    switch (path) {
    case AscendPath::kGmToL1:
      return params_.mte2_gm_to_l1_base_latency;
    case AscendPath::kGmToUb:
      return params_.mte2_gm_to_ub_base_latency;
    case AscendPath::kUbToGm:
      return params_.mte3_ub_to_gm_base_latency;
    case AscendPath::kUbToL1:
      return params_.mte3_ub_to_l1_base_latency;
    case AscendPath::kL1ToL0a:
    case AscendPath::kL1ToL0b:
    case AscendPath::kL1ToBt:
    case AscendPath::kL1ToFpBuf:
      return params_.mte1_base_latency;
    case AscendPath::kL1ToL0Sf:
      return params_.mte1_sf_base_latency;
    case AscendPath::kL0cToUb:
      return params_.fixpipe_base_latency;
    case AscendPath::kL0cToGm:
      return params_.fixpipe_l0c_to_gm_base_latency;
    default:
      return 0;
    }
  }

  struct AivMteCost {
    int64_t completion_latency;
    int64_t ii_bytes;
    int64_t bandwidth;
  };

  AivMteCost EstimateAivMte(AscendPath path,
                            const TaskCostFeatures::CopyInfo &info) const {
    int64_t normalized_bytes = ScaleBytesByLanes(info.bytes, aiv_lanes_);
    int64_t bandwidth = BandwidthForCopy(path, info);
    int64_t completion_latency =
        params_.mte_descriptor_cycles +
        MemLatencyForPath(path, normalized_bytes, bandwidth,
                          BaseLatencyForCopy(path, info));
    return {completion_latency, normalized_bytes, bandwidth};
  }

  // Best-effort classification of a single region into an AscendPath from
  // its buffer scope. For reads, `path = ? -> dst_of_pipe`; for writes,
  // `path = src_of_pipe -> ?`. The task's pipe mask narrows it further.
  AscendPath ClassifyRegion(const String &scope, uint16_t pipe_mask,
                            bool is_read) const {
    bool has_mte1 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE1);
    bool has_mte2 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE2);
    bool has_mte3 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE3);
    bool has_fix = pipe_mask & static_cast<uint16_t>(ResourcePipe::kFixpipe);

    std::string s = scope.operator std::string();
    if (has_mte2) {
      if (s == "global" && !is_read)
        return AscendPath::kUnknown;
      if (s == "global" && is_read)
        return AscendPath::kGmToUb; // default
      if (s == "shared.l1")
        return AscendPath::kGmToL1;
      if (s == "shared" || s == "shared.dyn")
        return AscendPath::kGmToUb;
    }
    if (has_mte3) {
      if (s == "global")
        return AscendPath::kUbToGm;
      if (s == "shared.l1")
        return AscendPath::kUbToL1;
    }
    if (has_mte1) {
      if (s == "shared.l0a")
        return AscendPath::kL1ToL0a;
      if (s == "shared.l0b")
        return AscendPath::kL1ToL0b;
      if (s == "shared.l0a.sf" || s == "shared.l0b.sf")
        return AscendPath::kL1ToL0Sf;
      if (s == "shared.bt")
        return AscendPath::kL1ToBt;
    }
    if (has_fix) {
      if (s == "shared.l0c" && is_read) {
        // L0C is the source; dest scope comes from the *other* region.
        return AscendPath::kL0cToUb; // refined by paired region below
      }
      if (s == "global")
        return AscendPath::kL0cToGm;
      if (s == "shared" || s == "shared.dyn")
        return AscendPath::kL0cToUb;
    }
    return AscendPath::kUnknown;
  }

  AscendPath ClassifyCopy(const String &src_scope,
                          const String &dst_scope) const {
    std::string src = src_scope.operator std::string();
    std::string dst = dst_scope.operator std::string();
    if (src == "global" && dst == "shared.l1")
      return AscendPath::kGmToL1;
    if (src == "global" && (dst == "shared" || dst == "shared.dyn"))
      return AscendPath::kGmToUb;
    if ((src == "shared" || src == "shared.dyn") && dst == "global")
      return AscendPath::kUbToGm;
    if ((src == "shared" || src == "shared.dyn") && dst == "shared.l1")
      return AscendPath::kUbToL1;
    if (src == "shared.l1" && dst == "shared.l0a")
      return AscendPath::kL1ToL0a;
    if (src == "shared.l1" && dst == "shared.l0b")
      return AscendPath::kL1ToL0b;
    if ((src == "shared.l1" || src == "shared.l1.dyn") &&
        (dst == "shared.l0a.sf" || dst == "shared.l0b.sf"))
      return AscendPath::kL1ToL0Sf;
    if (src == "shared.l1" && dst == "shared.bt")
      return AscendPath::kL1ToBt;
    if (src == "shared.l1" && dst == "shared.fp")
      return AscendPath::kL1ToFpBuf;
    if (src == "shared.l0c" && (dst == "shared" || dst == "shared.dyn"))
      return AscendPath::kL0cToUb;
    if (src == "shared.l0c" && dst == "global")
      return AscendPath::kL0cToGm;
    return AscendPath::kUnknown;
  }

  // MAD consumes logical regions, but executes complete compute groups.
  // This geometry is independent of the padded L0 allocation and of MTE1's
  // transpose groups. Full FP32 on A5 has K parallelism 1; HF32 has 8.
  int64_t CubeOperations(const TaskCostFeatures::CubeShape &shape) const {
    int64_t k_group = 256 / shape.input_dtype.bits();
    if (shape.input_dtype.MatchesCode(kDLFloat) &&
        shape.input_dtype.bits() == 32)
      k_group = shape.hf32 ? 8 : 1;
    auto align = [](int64_t extent, int64_t group) {
      return ((extent + group - 1) / group) * group;
    };
    int64_t m = align(shape.m, 16);
    int64_t n = align(shape.n, 16);
    if (shape.blockscaled) {
      // MX MAD's independent-issue cost has a minimum larger M/N extent.
      return 2 * std::min(m, n) * std::max({m, n, params_.cube_mx_min_mn}) *
             align(shape.k, k_group);
    }
    return 2 * m * n * align(shape.k, k_group);
  }

  // Throughput in operations/cycle, matching the rounded operation count.
  int64_t CubeThroughputFor(PrimType dtype, bool hf32 = false,
                            bool blockscaled = false) const {
    if (blockscaled && dtype.MatchesElementType(kDLFloat4_e2m1fn, 4))
      return params_.cube_throughput_mxfp4;
    if (dtype.MatchesElementType(kDLFloat, 16))
      return params_.cube_throughput_fp16 ? params_.cube_throughput_fp16
                                          : params_.cube_fallback_throughput;
    if (dtype.MatchesElementType(kDLBfloat, 16))
      return params_.cube_throughput_bf16 ? params_.cube_throughput_bf16
                                          : params_.cube_fallback_throughput;
    if (dtype.MatchesCode(kDLFloat) && dtype.bits() == 32) {
      int64_t throughput =
          hf32 ? params_.cube_throughput_hf32 : params_.cube_throughput_fp32;
      ICHECK_GT(throughput, 0) << "FP32/HF32 Cube throughput must be positive";
      return throughput;
    }
    if (dtype.MatchesCode(kDLFloat8_e3m4, kDLFloat8_e4m3, kDLFloat8_e4m3b11fnuz,
                          kDLFloat8_e4m3fn, kDLFloat8_e4m3fnuz, kDLFloat8_e5m2,
                          kDLFloat8_e5m2fnuz, kDLFloat8_e8m0fnu) ||
        (dtype.MatchesCode(kDLFloat) && dtype.bits() == 8))
      return params_.cube_throughput_fp8 ? params_.cube_throughput_fp8
                                         : params_.cube_fallback_throughput;
    if (dtype.MatchesCode(kDLInt) && dtype.bits() == 8)
      return params_.cube_throughput_int8 ? params_.cube_throughput_int8
                                          : params_.cube_fallback_throughput;
    return params_.cube_fallback_throughput;
  }

  // Ascend latency / II estimation entry point.
  void EstimateAscend(TaskNode *task, const TaskCostFeatures &cost_features,
                      sym::AnalyzerObj *analyzer) {
    uint16_t pipe_mask = task->GetPipeMask();
    bool has_mte1 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE1);
    bool has_mte2 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE2);
    bool has_mte3 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE3);
    bool has_fix = pipe_mask & static_cast<uint16_t>(ResourcePipe::kFixpipe);
    bool has_cube = pipe_mask & static_cast<uint16_t>(ResourcePipe::kCube);
    bool has_vec = pipe_mask & static_cast<uint16_t>(ResourcePipe::kVector);
    bool has_scl = pipe_mask & static_cast<uint16_t>(ResourcePipe::kScalar);
    bool has_dma = has_mte1 || has_mte2 || has_mte3 || has_fix;
    int64_t measured_vf_latency = ReadVFLatencyFromBlocks(task);

    // -----------------------------------------------------------------
    // 1. Memory latency: per-physical-path, base + bytes / bandwidth.
    //    Bytes are aggregated per path for the II step.
    // -----------------------------------------------------------------
    struct PathIICost {
      int64_t bytes{0};
      int64_t override_cycles{0};
    };
    std::unordered_map<int, PathIICost> ii_cost_per_path;
    int64_t memory_latency = 0;

    auto add_memory_access = [&](AscendPath path, int64_t completion_latency,
                                 int64_t ii_bytes,
                                 int64_t ii_bandwidth_override) {
      memory_latency += completion_latency;
      PathIICost &cost = ii_cost_per_path[static_cast<int>(path)];
      if (path == AscendPath::kGmToL1 && ii_bytes > 0) {
        // The small-packet floor is paid by every descriptor, so charge it
        // before descriptors on the same MTE2 path are accumulated.
        cost.override_cycles +=
            std::max(safe_div_ceil(ii_bytes, BandwidthForPath(path)),
                     params_.mte2_gm_to_l1_min_ii);
      } else if ((path == AscendPath::kGmToUb || path == AscendPath::kUbToGm) &&
                 ii_bytes > 0) {
        int64_t min_ii = path == AscendPath::kGmToUb
                             ? params_.mte2_gm_to_ub_min_ii
                             : params_.mte3_ub_to_gm_min_ii;
        int64_t bandwidth = ii_bandwidth_override > 0 ? ii_bandwidth_override
                                                      : BandwidthForPath(path);
        cost.override_cycles +=
            std::max(min_ii, safe_div_ceil(ii_bytes, bandwidth));
      } else if ((path == AscendPath::kL1ToL0a ||
                  path == AscendPath::kL1ToL0b) &&
                 ii_bytes > 0) {
        cost.override_cycles += params_.mte1_issue_overhead +
                                safe_div_ceil(ii_bytes, BandwidthForPath(path));
      } else if (path == AscendPath::kL1ToL0Sf && ii_bytes > 0) {
        cost.override_cycles += params_.mte1_sf_issue_overhead +
                                safe_div_ceil(ii_bytes, BandwidthForPath(path));
      } else if ((path == AscendPath::kL0cToUb ||
                  path == AscendPath::kL0cToGm) &&
                 ii_bytes > 0) {
        // A dual copy can be limited by either the shared FixPipe endpoint
        // or its per-AIV row width. Charge descriptor overhead only on the
        // endpoint: a narrow row already keeps it occupied for longer.
        int64_t endpoint_bandwidth =
            path == AscendPath::kL0cToUb && ii_bandwidth_override > 0
                ? params_.fixpipe_dual_bandwidth
                : params_.fixpipe_bandwidth;
        int64_t endpoint_ii = params_.fixpipe_issue_overhead +
                              safe_div_ceil(ii_bytes, endpoint_bandwidth);
        int64_t geometry_ii =
            ii_bandwidth_override > 0
                ? safe_div_ceil(ii_bytes, ii_bandwidth_override)
                : 0;
        cost.override_cycles += std::max(endpoint_ii, geometry_ii);
      } else if (ii_bandwidth_override > 0) {
        cost.override_cycles += safe_div_ceil(ii_bytes, ii_bandwidth_override);
      } else {
        cost.bytes += ii_bytes;
      }
    };

    if (!cost_features.copy_infos.empty()) {
      for (const TaskCostFeatures::CopyInfo &info : cost_features.copy_infos) {
        AscendPath path = ClassifyCopy(info.src_scope, info.dst_scope);
        int64_t completion_latency = 0;
        int64_t ii_bytes = info.bytes;
        int64_t ii_bandwidth = 0;
        if (IsAivMtePath(path)) {
          AivMteCost cost = EstimateAivMte(path, info);
          completion_latency = cost.completion_latency;
          ii_bytes = cost.ii_bytes;
          ii_bandwidth = cost.bandwidth;
        } else {
          int64_t bandwidth = BandwidthForCopy(path, info);
          int64_t copy_bytes = path == AscendPath::kL0cToUb && IsDualCopy(info)
                                   ? ScaleBytesByLanes(info.bytes, 2)
                                   : info.bytes;
          completion_latency =
              params_.mte_descriptor_cycles +
              MemLatencyForPath(path, copy_bytes, bandwidth,
                                BaseLatencyForCopy(path, info));
          if (IsDualCopy(info)) {
            ii_bytes = copy_bytes;
            ii_bandwidth = bandwidth;
          }
        }
        add_memory_access(path, completion_latency, ii_bytes, ii_bandwidth);
      }
    } else {
      auto visit_region = [&](const TensorRegion &region, bool is_read) {
        const TensorVar buffer = RegionTensor(region);
        String scope = buffer.scope().name();
        AscendPath path = ClassifyRegion(scope, pipe_mask, is_read);
        int64_t bytes = CalculateAccessBytes(region, analyzer);
        if (IsAivMtePath(path)) {
          TaskCostFeatures::CopyInfo info{String(),
                                          String(),
                                          bytes,
                                          /*dual_dst_ctl=*/0,
                                          /*geometry=*/CopyGeometry::kM,
                                          /*contiguous_bytes_lower_bound=*/0,
                                          /*is_nd2nz_post_copy=*/false};
          AivMteCost cost = EstimateAivMte(path, info);
          add_memory_access(path, cost.completion_latency, cost.ii_bytes,
                            cost.bandwidth);
        } else {
          int64_t completion_latency =
              params_.mte_descriptor_cycles + MemLatencyForPath(path, bytes);
          add_memory_access(path, completion_latency, bytes, 0);
        }
      };
      for (const auto &r : task->GetReadRegions())
        visit_region(r, /*is_read=*/true);
      for (const auto &r : task->GetWriteRegions())
        visit_region(r, /*is_read=*/false);
    }

    // -----------------------------------------------------------------
    // 2. Compute latency.
    // -----------------------------------------------------------------
    int64_t compute_latency = 0;
    int64_t vector_latency = 0;

    if (has_vec) {
      if (measured_vf_latency > 0) {
        // User-annotated latency — use it directly, skip model estimation.
        vector_latency = measured_vf_latency;
      } else {
        // Op-count path (TIR visitor).
        auto counter = make_object<OperationCounter>(&params_);
        counter->Visit(task->stmt);
        int64_t op_count_latency = counter->GetEstimatedLatency();

        // Bandwidth-bound path: total bytes the VALU consumes / valu_bandwidth.
        int64_t valu_bytes = 0;
        for (const auto &r : task->GetReadRegions()) {
          String scope = RegionTensor(r).scope().name();
          if (scope == "local" || scope == "local.fragment" ||
              scope == "shared" || scope == "shared.dyn") {
            valu_bytes += CalculateAccessBytes(r, analyzer);
          }
        }
        int64_t bw_latency = safe_div_ceil(valu_bytes, params_.valu_bandwidth);

        vector_latency = std::max(op_count_latency, bw_latency);
      }
      compute_latency = std::max(compute_latency, vector_latency);
    }

    if (has_cube) {
      // Charge completion overhead once for this leaf's Cube instruction
      // stream, separately from its total throughput cost. Sequential tasks
      // and loop iterations are composed by EstimateStmtCost below.
      // Use recorded GEMM shapes, or fall back to one unit MMAD.
      int64_t matmul_latency = params_.cube_base_latency;
      if (!cost_features.cube_shapes.empty()) {
        int64_t base_latency = params_.cube_base_latency;
        for (const TaskCostFeatures::CubeShape &shape :
             cost_features.cube_shapes) {
          int64_t ops = CubeOperations(shape);
          int64_t throughput = CubeThroughputFor(shape.input_dtype, shape.hf32,
                                                 shape.blockscaled);
          if (shape.blockscaled) {
            base_latency = std::max(base_latency, params_.cube_mx_base_latency);
          } else if (shape.input_dtype.MatchesCode(kDLFloat) &&
                     shape.input_dtype.bits() == 32) {
            base_latency =
                std::max(base_latency, params_.cube_fp32_base_latency);
          }
          matmul_latency += safe_div_ceil(ops, throughput);
        }
        matmul_latency += base_latency - params_.cube_base_latency;
      } else if (params_.cube_unit_m > 0 && params_.cube_unit_k > 0 &&
                 params_.cube_unit_n > 0) {
        // Single unit mad fallback.
        int64_t unit_ops =
            params_.cube_unit_m * params_.cube_unit_k * params_.cube_unit_n;
        int64_t throughput = CubeThroughputFor(PrimType::Float(16));
        matmul_latency =
            params_.cube_base_latency + safe_div_ceil(unit_ops, throughput);
      }
      compute_latency = std::max(compute_latency, matmul_latency);
    }

    if (has_scl && !has_vec) {
      // Pure scalar tasks (set/wait_flag bodies, address compute). Charge
      // a flat one-cycle cost. A task carries exactly one statement.
      compute_latency = std::max<int64_t>(compute_latency, 1);
    }

    int64_t total_latency = memory_latency + compute_latency;

    // -----------------------------------------------------------------
    // 4. II: per-pipe analysis. Take the max across all active pipes.
    // -----------------------------------------------------------------
    int64_t ii = 1;
    int64_t memory_ii = 1;
    for (const auto &kv : ii_cost_per_path) {
      AscendPath path = static_cast<AscendPath>(kv.first);
      const PathIICost &cost = kv.second;
      int64_t bw = BandwidthForPath(path);
      int64_t path_ii = cost.override_cycles + safe_div_ceil(cost.bytes, bw);
      memory_ii = std::max(memory_ii, path_ii);
    }

    int64_t cube_ii = params_.cube_min_ii;
    if (has_cube) {
      if (!cost_features.cube_shapes.empty()) {
        for (const TaskCostFeatures::CubeShape &shape :
             cost_features.cube_shapes) {
          int64_t ops = CubeOperations(shape);
          int64_t throughput = CubeThroughputFor(shape.input_dtype, shape.hf32,
                                                 shape.blockscaled);
          cube_ii = std::max(cube_ii, safe_div_ceil(ops, throughput));
        }
      }
      cube_ii = std::max(cube_ii, params_.cube_pipeline_depth);
    }

    if (has_dma && !has_cube && !has_vec) {
      // DMA-only: II = max bandwidth-limited II across active paths.
      ii = std::max(ii, memory_ii);
    } else if (has_cube && !has_dma && !has_vec) {
      // Cube-only: II is bounded by pipeline depth and MAC throughput.
      ii = std::max(ii, cube_ii);
    } else if (has_vec && !has_dma && !has_cube) {
      // Vector tasks occupy the pipe for their full completion latency.
      ii = std::max(ii, vector_latency);
    } else {
      // Mixed / fallback: keep task granularity, but bound II by the slowest
      // active pipe instead of forcing it to the full sequential latency.
      if (has_dma)
        ii = std::max(ii, memory_ii);
      if (has_cube)
        ii = std::max(ii, cube_ii);
      if (has_vec)
        ii = std::max(ii, vector_latency);
    }

    task->SetLatency(total_latency);
    task->SetII(ii);
  }

  // VALU operation counter.
  class OperationCounter : public StmtExprVisitor {
  public:
    using StmtExprVisitor::Visit;
    using StmtExprVisitor::Visit_;
    using Result = ffi::Optional<ffi::VisitInterrupt>;
    // Loop dimension information
    struct LoopDimension {
      int64_t trip_count;
      int depth;
    };

    int64_t total_latency = 0;
    const AscendLatencyParams *params = nullptr;

    // Track loop dimensions for loop-invariant detection
    std::vector<LoopDimension> loop_stack;
    std::unordered_map<const VarObj *, int> var_to_depth;

    explicit OperationCounter(const AscendLatencyParams *params)
        : params(params) {}

    // Operator to visit a statement
    void operator()(const Stmt &stmt) { Visit(stmt); }

    // Get estimated latency
    int64_t GetEstimatedLatency() const { return total_latency; }

    // Helper function to update latency for an operation based on throughput
    void
    update_operation(int64_t throughput,
                     const std::unordered_set<const VarObj *> &contained_vars) {
      int64_t effective_factor =
          CalculateEffectiveParallelFactor(contained_vars);

      if (throughput > 0) {
        int64_t cycles_for_operations =
            (effective_factor + throughput - 1) / throughput;
        total_latency += std::max<int64_t>(cycles_for_operations, 1);
      } else {
        total_latency += 1;
      }
    }

    // Analyze which loop variables are contained in an expression
    std::unordered_set<const VarObj *>
    AnalyzeContainedLoopVars(const Expr &expr) {
      class VarCollector : public ExprVisitor {
      public:
        using ExprVisitor::Visit;
        using ExprVisitor::Visit_;
        using Result = ffi::Optional<ffi::VisitInterrupt>;
        const std::unordered_map<const VarObj *, int> &var_to_depth;
        std::unordered_set<const VarObj *> collected_vars;
        explicit VarCollector(
            const std::unordered_map<const VarObj *, int> &depths)
            : var_to_depth(depths) {}
        Result Visit_(const VarObj *var) final {
          if (var_to_depth.count(var))
            collected_vars.insert(var);
          return std::nullopt;
        }
      };

      auto collector = make_object<VarCollector>(var_to_depth);
      collector->Visit(expr);
      return collector->collected_vars;
    }

    // Calculate effective parallel factor based on contained loop variables
    int64_t CalculateEffectiveParallelFactor(
        const std::unordered_set<const VarObj *> &contained_vars) {
      if (contained_vars.empty()) {
        return 1; // Completely loop-invariant
      }

      // Find the maximum depth of contained loop variables
      int max_depth = -1;
      for (const VarObj *var : contained_vars) {
        auto it = var_to_depth.find(var);
        if (it != var_to_depth.end()) {
          max_depth = std::max(max_depth, it->second);
        }
      }

      // Calculate effective parallel factor: product of trip counts for loops
      // with depth <= max_depth
      int64_t effective_factor = 1;
      for (const auto &loop : loop_stack) {
        if (loop.depth <= max_depth) {
          effective_factor *= loop.trip_count;
        }
      }

      return effective_factor;
    }

    Result Visit_(const AddObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->add_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    Result Visit_(const SubObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->sub_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    Result Visit_(const MulObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->mul_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    Result Visit_(const DivObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->div_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    Result Visit_(const ModObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->mod_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    Result Visit_(const FloorDivObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->div_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    Result Visit_(const FloorModObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->mod_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    Result Visit_(const LTObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    Result Visit_(const LEObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    Result Visit_(const GTObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    Result Visit_(const GEObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    Result Visit_(const EQObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    Result Visit_(const NEObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    Result Visit_(const AndObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->logic_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    Result Visit_(const OrObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->logic_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    Result Visit_(const NotObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->logic_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    Result Visit_(const MinObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->min_max_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    Result Visit_(const MaxObj *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));
      update_operation(params->min_max_throughput, contained_vars);
      StmtExprVisitor::Visit_(op);
      return std::nullopt;
    }

    int64_t EstimateNd2NzScatterLatency(const CallObj *op) const {
      // InsertNd2Nz validates this template contract before EstimateLatency.
      const auto args = DecodeCallOperands(op);
      ICHECK_EQ(args.size(), 7U);
      auto rows = TryConstInt(args[2].as_or_throw<PrimExpr>());
      auto cols = TryConstInt(args[3].as_or_throw<PrimExpr>());
      const auto *dst_dtype = args[6].as<StringImmObj>();
      const auto *src_dtype = args[5].as<StringImmObj>();
      ICHECK(rows && cols && src_dtype && dst_dtype)
          << "ND->NZ scatter requires static shape and dtype arguments: "
          << GetRef<Call>(op);
      ICHECK_GT(*rows, 0);
      ICHECK_GT(*cols, 0);
      ICHECK(src_dtype->value == "float" || src_dtype->value == "half" ||
             src_dtype->value == "bfloat16_t");

      const AscendLatencyParams::Nd2NzLatencyParams *profile =
          &params->nd2nz_same_dtype;
      if (src_dtype->value != dst_dtype->value) {
        if (src_dtype->value == "bfloat16_t" && dst_dtype->value == "float") {
          profile = &params->nd2nz_bf16_to_f32;
        } else {
          ICHECK(src_dtype->value == "float" &&
                 dst_dtype->value == "bfloat16_t")
              << "Unsupported ND->NZ scatter dtype conversion: "
              << src_dtype->value << " -> " << dst_dtype->value;
          profile = &params->nd2nz_f32_to_bf16;
        }
      }

      int64_t vector_elements = src_dtype->value == "float" ? 64 : 128;
      int64_t passes = *cols / vector_elements + (*cols % vector_elements != 0);
      int64_t row_iterations =
          CheckedSerialTaskMultiply(*rows, passes, "ND->NZ row iterations");
      int64_t row_cycles_x4 = CheckedSerialTaskMultiply(
          row_iterations, profile->row_cycles_x4, "ND->NZ row cycles");
      int64_t cycles = CheckedSerialTaskAdd(
          profile->setup_cycles,
          CheckedSerialTaskMultiply(passes, profile->pass_cycles,
                                    "ND->NZ pass cycles"),
          "ND->NZ setup cycles");
      cycles = CheckedSerialTaskAdd(
          cycles, row_cycles_x4 / 4 + (row_cycles_x4 % 4 != 0),
          "ND->NZ scatter latency");
      return std::max(profile->min_cycles, cycles);
    }

    Result Visit_(const CallObj *op) final {
      if (IsAnnotationCall(op))
        return std::nullopt;
      if (op->op.same_as(tl::ascend_nd2nz_scatter())) {
        // This opaque helper's loops, conversions and stores are absent from
        // TIR. Count the full template, including repeated calls inside a VF.
        // Access-pointer descriptors are not extra vector operations.
        int64_t cycles = EstimateNd2NzScatterLatency(op);
        for (const auto &loop : loop_stack) {
          cycles = CheckedSerialTaskMultiply(cycles, loop.trip_count,
                                             "ND->NZ loop latency");
        }
        total_latency =
            CheckedSerialTaskAdd(total_latency, cycles, "ND->NZ task latency");
        return std::nullopt;
      }
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<Expr>(op));

      // Check for special math functions by name
      if (op->op.as<OpObj>()) {
        auto op_node = op->op.as<OpObj>();
        std::string op_name = op_node->name;

        // Check for special math functions
        if (op_name == "exp2" || op_name == "log2" || op_name == "exp" ||
            op_name == "log" || op_name == "sin" || op_name == "cos" ||
            op_name == "tan" || op_name == "asin" || op_name == "acos" ||
            op_name == "atan" || op_name == "sinh" || op_name == "cosh" ||
            op_name == "tanh" || op_name == "sqrt" || op_name == "rsqrt" ||
            op_name == "pow" || op_name == "erf" || op_name == "sigmoid") {
          // Special math functions have lower throughput
          update_operation(params->special_func_throughput, contained_vars);
        } else if (op_name.find("copy") != std::string::npos ||
                   op_name.find("gemm") != std::string::npos) {
          // Data movement and Cube latency are modeled separately.
        } else {
          update_operation(params->default_operation_throughput,
                           contained_vars);
        }
      } else {
        update_operation(params->default_operation_throughput, contained_vars);
      }
      // Visit arguments
      // Only executable operands contribute operations; metadata is
      // compile-time.
      const auto *callee = op->op.as<OpObj>();
      auto args = callee && callee->name.find("tl.tileop.") == 0
                      ? DecodeTileOpArguments(op).operands
                      : DecodeCallOperands(op);
      for (const Expr &arg : args)
        Visit(arg);
      return std::nullopt;
    }

    Result Visit_(const ForObj *op) final {
      int64_t trip_count = GetSerialTripCount(GetRef<For>(op));
      LoopDimension dimension{trip_count, static_cast<int>(loop_stack.size())};
      loop_stack.push_back(dimension);
      var_to_depth[op->var.get()] = dimension.depth;
      if (trip_count > 0)
        Visit(op->body);
      var_to_depth.erase(op->var.get());
      loop_stack.pop_back();
      return std::nullopt;
    }

    Result Visit_(const EvaluateObj *op) final {
      if (op->value.defined()) {
        StmtExprVisitor::Visit(op->value);
      }
      return std::nullopt;
    }

    Result Visit_(const TensorStoreObj *op) final {
      // Count operations in indices
      for (const auto &index : op->indices) {
        StmtExprVisitor::Visit(index);
      }
      // Count operations in value
      StmtExprVisitor::Visit(op->value);
      return std::nullopt;
    }

    Result Visit_(const SeqStmtObj *op) final {
      for (const auto &child : op->seq) {
        StmtExprVisitor::Visit(child);
      }
      return std::nullopt;
    }

    Result Visit_(const RegionStmtObj *op) final {
      StmtExprVisitor::Visit(op->body);

      return std::nullopt;
    }

    Result Visit_(const BindObj *op) final {
      // Let binding: the value expression is evaluated once
      StmtExprVisitor::Visit(op->value);

      return std::nullopt;
    }

    Result Visit_(const IfObj *op) final {
      Visit(op->condition);

      // For if-then-else branches, we need to take the maximum latency of both
      // paths Save current latency
      int64_t old_latency = total_latency;

      // Count latency in then branch
      StmtExprVisitor::Visit(op->then_case);
      int64_t then_latency = total_latency;

      // Restore latency and count else branch
      total_latency = old_latency;
      if (op->else_case) {
        StmtExprVisitor::Visit(op->else_case.value());
      }
      int64_t else_latency = total_latency;

      // Take the maximum of both branches
      total_latency = old_latency + std::max(then_latency - old_latency,
                                             else_latency - old_latency);

      return std::nullopt;
    }
  };

  // Helper function to calculate total bytes accessed in a region.
  int64_t CalculateAccessBytes(const TensorRegion &region,
                               sym::AnalyzerObj *analyzer) {
    RegionElementUpperBound estimate = EstimateRegionElementUpperBound(
        RegionTensor(region), region->region, analyzer);
    return ElementsToBytes(estimate.elements, RegionTensor(region).dtype());
  }
};

class CostFeatureAnalyzer : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  using Result = ffi::Optional<ffi::VisitInterrupt>;
  static TaskCostFeatures Analyze(const Stmt &stmt,
                                  sym::AnalyzerObj *arith_analyzer,
                                  const Hf32ModeMap &hf32_modes) {
    ICHECK(arith_analyzer != nullptr);
    auto analyzer =
        make_object<CostFeatureAnalyzer>(arith_analyzer, hf32_modes);
    analyzer->Visit(stmt);
    return analyzer->features_;
  }

public:
  CostFeatureAnalyzer(sym::AnalyzerObj *arith_analyzer,
                      const Hf32ModeMap &hf32_modes)
      : arith_analyzer_(arith_analyzer), hf32_modes_(hf32_modes) {}

  static int64_t SafeHalf(int64_t value) {
    if (value <= 0)
      return 0;
    return value / 2 + value % 2;
  }

  static int64_t SafeMultiply(int64_t lhs, int64_t rhs) {
    if (lhs <= 0 || rhs <= 0)
      return 0;
    return lhs > std::numeric_limits<int64_t>::max() / rhs
               ? std::numeric_limits<int64_t>::max()
               : lhs * rhs;
  }

  static int SplitAxis(const Array<Range> &ranges, int dual_dst_ctl) {
    if (ranges.empty())
      return -1;
    if (ranges.size() == 1)
      return 0;
    return static_cast<int>(ranges.size()) - 2 + (dual_dst_ctl == 1 ? 0 : 1);
  }

  static Array<Range> ReplaceExtent(const Array<Range> &ranges, int axis,
                                    const PrimExpr &extent) {
    Array<Range> result;
    result.reserve(ranges.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
      result.push_back(static_cast<int>(i) == axis
                           ? Range::FromMinExtent(ranges[i]->begin, extent)
                           : ranges[i]);
    }
    return result;
  }

  MteGeometry InferMteGeometry(const AscendCopyNode *copy) const {
    Array<Range> src_ranges = copy->src_range;
    Array<Range> dst_ranges = copy->dst_range;
    if (src_ranges.empty() || dst_ranges.empty() ||
        src_ranges.size() != copy->src.shape().size() ||
        dst_ranges.size() != copy->dst.shape().size()) {
      return {};
    }

    src_ranges = NormalizeEmptyUnitAxesForMTE(src_ranges, arith_analyzer_);
    dst_ranges = NormalizeEmptyUnitAxesForMTE(dst_ranges, arith_analyzer_);

    int dual_dst_ctl = copy->dual_dst_ctl;
    if (dual_dst_ctl == 1 || dual_dst_ctl == 2) {
      int src_axis = SplitAxis(src_ranges, dual_dst_ctl);
      int dst_axis = SplitAxis(dst_ranges, dual_dst_ctl);
      if (src_axis < 0 || dst_axis < 0)
        return {};
      bool source_is_half = copy->annotations.Get("double").has_value();
      if (source_is_half) {
        dst_ranges =
            ReplaceExtent(dst_ranges, dst_axis, src_ranges[src_axis]->extent);
      } else {
        src_ranges =
            ReplaceExtent(src_ranges, src_axis, dst_ranges[dst_axis]->extent);
      }
    }

    int src_bits = copy->src.dtype().bits() * copy->src.dtype().lanes();
    int dst_bits = copy->dst.dtype().bits() * copy->dst.dtype().lanes();
    StridedLayout src_layout =
        StridedLayout::FromBufferRange(copy->src, src_ranges, src_bits)
            .Coalesce(arith_analyzer_);
    StridedLayout dst_layout =
        StridedLayout::FromBufferRange(copy->dst, dst_ranges, dst_bits)
            .Coalesce(arith_analyzer_);
    if (src_layout.modes.empty() || dst_layout.modes.empty() ||
        src_layout.modes.size() > 2 || dst_layout.modes.size() > 2) {
      return {};
    }

    CopyGeometry kind =
        src_layout.modes.size() == 1 && dst_layout.modes.size() == 1
            ? CopyGeometry::kM
            : CopyGeometry::kN;
    bool is_fixpipe_dual = (dual_dst_ctl == 1 || dual_dst_ctl == 2) &&
                           IsL0CBuffer(copy->src) &&
                           IsSharedBuffer(copy->dst) &&
                           src_ranges.size() >= 2 && dst_ranges.size() >= 2;
    PrimExpr src_row_size = src_layout.modes[0].size;
    PrimExpr dst_row_size = dst_layout.modes[0].size;
    if (is_fixpipe_dual) {
      StridedLayout src_mte_layout =
          NormalizeTrailingMTE2DLayout(copy->src, src_ranges, arith_analyzer_,
                                       "Ascend L0C->UB latency source");
      StridedLayout dst_mte_layout =
          NormalizeMTE2DLayout(copy->dst, dst_ranges, arith_analyzer_,
                               "Ascend L0C->UB latency destination");
      src_row_size = src_mte_layout.modes[0].size;
      dst_row_size = dst_mte_layout.modes[0].size;
    } else if (src_layout.modes.size() == 1 && dst_layout.modes.size() == 2) {
      // Match PlanMTECopy: split the contiguous side at the strided side's
      // physical row boundary. Its coalesced size is the whole transfer,
      // whose dynamic row count must not make a fixed row width unknown.
      src_row_size = dst_row_size;
    } else if (src_layout.modes.size() == 2 && dst_layout.modes.size() == 1) {
      dst_row_size = src_row_size;
    }
    int64_t src_row_elements = EstimateExprPositiveLowerBound(src_row_size);
    int64_t dst_row_elements = EstimateExprPositiveLowerBound(dst_row_size);
    if (src_row_elements <= 0 || dst_row_elements <= 0)
      return {kind, 0};
    int64_t src_row_bytes =
        ElementsToBytes(src_row_elements, copy->src.dtype());
    int64_t dst_row_bytes =
        ElementsToBytes(dst_row_elements, copy->dst.dtype());
    return {kind, std::min(src_row_bytes, dst_row_bytes)};
  }

  int64_t EstimateExprUpperBound(const PrimExpr &expr) const {
    if (auto value = TryConstInt(expr))
      return std::max<int64_t>(*value, 0);
    sym::ConstIntBound bound = arith_analyzer_->const_int_bound(expr);
    if (bound->max_value >= 0 &&
        bound->max_value != sym::ConstIntBound::kPosInf) {
      return bound->max_value;
    }
    return 1;
  }

  int64_t EstimateExprPositiveLowerBound(const PrimExpr &expr) const {
    if (auto value = TryConstInt(expr))
      return std::max<int64_t>(*value, 0);
    sym::ConstIntBound bound = arith_analyzer_->const_int_bound(expr);
    return bound->min_value > 0 ? bound->min_value : 0;
  }

  int64_t EstimateGemmExtent(const PrimExpr &extent, int64_t fallback) const {
    sym::ConstIntBound bound = arith_analyzer_->const_int_bound(extent);
    if (bound->max_value >= 0 &&
        bound->max_value != sym::ConstIntBound::kPosInf)
      return std::min(fallback, bound->max_value);
    return fallback;
  }

  int64_t CalculateCopyBytes(const AscendCopyNode *copy) const {
    RegionElementUpperBound src_estimate = EstimateRegionElementUpperBound(
        copy->src, copy->src_range, arith_analyzer_);
    RegionElementUpperBound dst_estimate = EstimateRegionElementUpperBound(
        copy->dst, copy->dst_range, arith_analyzer_);

    if (copy->dual_dst_ctl == 1 || copy->dual_dst_ctl == 2) {
      // The frontend represents one side as a full logical region and the
      // other as one AIV's half region. Normalize both forms to one AIV's
      // physical payload. `double` identifies the half-source form. Prefer
      // whichever reliable side is available for a dynamic tail instead of
      // accepting the fallback 1.
      bool source_is_half = copy->annotations.Get("double").has_value();
      int64_t src_elements = source_is_half ? src_estimate.elements
                                            : SafeHalf(src_estimate.elements);
      int64_t dst_elements = source_is_half ? SafeHalf(dst_estimate.elements)
                                            : dst_estimate.elements;
      int64_t elements = 1;
      if (src_estimate.reliable && dst_estimate.reliable) {
        elements = std::min(src_elements, dst_elements);
      } else if (src_estimate.reliable) {
        elements = src_elements;
      } else if (dst_estimate.reliable) {
        elements = dst_elements;
      }
      return ElementsToBytes(elements, copy->dst.dtype());
    }

    int64_t elements = src_estimate.elements;
    PrimExpr src_elements = RegionElementCountExpr(copy->src_range);
    PrimExpr dst_elements = RegionElementCountExpr(copy->dst_range);
    if (arith_analyzer_->CanProveEqual(src_elements, dst_elements)) {
      if (src_estimate.reliable && dst_estimate.reliable) {
        elements = std::min(src_estimate.elements, dst_estimate.elements);
      } else if (dst_estimate.reliable) {
        elements = dst_estimate.elements;
      }
    }

    return ElementsToBytes(elements, copy->dst.dtype());
  }

  void RecordNd2NzPostCopy(const CallObj *op) {
    const auto args = DecodeCallOperands(op);
    ICHECK_EQ(args.size(), 7)
        << "tl.ascend_nd2nz_post_copy expects 7 arguments";
    int64_t element_bytes = Nd2NzElementBytes(args[6]);
    int64_t rows = EstimateExprUpperBound(args[2].as_or_throw<PrimExpr>());
    int64_t cols = EstimateExprUpperBound(args[3].as_or_throw<PrimExpr>());
    int64_t bytes = SafeMultiply(SafeMultiply(rows, cols), element_bytes);

    features_.copy_infos.push_back({String("shared.dyn"), String("shared.l1"),
                                    bytes, /*dual_dst_ctl=*/0,
                                    /*geometry=*/CopyGeometry::kM,
                                    /*contiguous_bytes_lower_bound=*/0,
                                    /*is_nd2nz_post_copy=*/true});
  }

  Result Visit_(const CallObj *op) final {
    static const auto gemm_op = Op::Get("tl.tileop.gemm");
    static const auto gemm_blockscaled_op =
        Op::Get("tl.tileop.gemm_blockscaled");
    if (IsAscendCopyCall(op)) {
      auto decoded = DecodeTileOpArguments(op);
      AscendCopy copy_obj(decoded.operands, decoded.annotations);
      const AscendCopyNode *copy = copy_obj.get();
      MteGeometry geometry = InferMteGeometry(copy);
      features_.copy_infos.push_back(
          {copy->src.scope().name(), copy->dst.scope().name(),
           CalculateCopyBytes(copy), copy->dual_dst_ctl, geometry.kind,
           geometry.contiguous_bytes_lower_bound,
           /*is_nd2nz_post_copy=*/false});
    } else if (op->op.same_as(tl::ascend_nd2nz_post_copy())) {
      RecordNd2NzPostCopy(op);
    } else if (op->op.same_as(gemm_op) || op->op.same_as(gemm_blockscaled_op)) {
      auto decoded = DecodeTileOpArguments(op);
      int64_t m = *TryConstInt(decoded.operands[5].as_or_throw<PrimExpr>());
      int64_t n = *TryConstInt(decoded.operands[6].as_or_throw<PrimExpr>());
      int64_t k = *TryConstInt(decoded.operands[7].as_or_throw<PrimExpr>());
      // Both spellings share the leading 13 dense slots. Keep the MX identity
      // so its compute cost does not fall back to the dense GEMM model.
      bool blockscaled = op->op.same_as(gemm_blockscaled_op);
      Gemm gemm = blockscaled
                      ? GemmBlockScaled(decoded.operands, decoded.annotations)
                      : Gemm(decoded.operands, decoded.annotations);
      if (IsL0ABuffer(gemm->a_) && IsL0BBuffer(gemm->b_)) {
        // The serialized ints are static tile metadata. A dynamic L0 tail
        // may have a tighter region bound than its padded allocation.
        const auto &c_ranges = gemm->cRegion_->region;
        const auto &a_ranges = gemm->aRegion_->region;
        m = EstimateGemmExtent(c_ranges[c_ranges.size() - 2]->extent, m);
        n = EstimateGemmExtent(c_ranges[c_ranges.size() - 1]->extent, n);
        k = EstimateGemmExtent(
            a_ranges[a_ranges.size() - (gemm->transA_ ? 2 : 1)]->extent, k);
      }
      auto mode = hf32_modes_.find(GetRef<Call>(op));
      bool hf32 = mode != hf32_modes_.end() &&
                  mode->second == Hf32ModeAnalyzer::kEnabled;
      if (m > 0 && n > 0 && k > 0)
        features_.cube_shapes.push_back(
            {m, n, k, gemm->a_.dtype(), hf32, blockscaled});
    }
    StmtExprVisitor::Visit_(op);

    return std::nullopt;
  }

  // Resource analysis treats blocks as atomic scheduler operations. Preserve
  // the same boundary for cost features; VF cost is modeled from the block.
  Result Visit_(const RegionStmtObj *op) final {
    if (BlockNameAtSourceBoundary(op))
      return std::nullopt;
    return Visit(op->body);

    return std::nullopt;
  }

  sym::AnalyzerObj *arith_analyzer_;
  const Hf32ModeMap &hf32_modes_;
  TaskCostFeatures features_;
};

struct PerCoreTaskCost {
  int64_t latency{0};
  int64_t ii{0};
};

int64_t CheckedAdd(int64_t lhs, int64_t rhs, const char *field) {
  ICHECK_GE(lhs, 0) << "PerCoreTask " << field << " must be non-negative";
  ICHECK_GE(rhs, 0) << "PerCoreTask " << field << " must be non-negative";
  ICHECK_LE(lhs, std::numeric_limits<int64_t>::max() - rhs)
      << "PerCoreTask " << field << " overflow while aggregating phases";
  return lhs + rhs;
}

int64_t CheckedMultiply(int64_t value, int64_t factor, const char *field) {
  ICHECK_GE(value, 0) << "PerCoreTask " << field << " must be non-negative";
  ICHECK_GE(factor, 0) << "PerCoreTask trip count must be non-negative";
  if (value == 0 || factor == 0)
    return 0;
  ICHECK_LE(value, std::numeric_limits<int64_t>::max() / factor)
      << "PerCoreTask " << field << " overflow while multiplying by trip count "
      << factor;
  return value * factor;
}

PerCoreTaskCost MaxCost(const PerCoreTaskCost &lhs,
                        const PerCoreTaskCost &rhs) {
  return {std::max(lhs.latency, rhs.latency), std::max(lhs.ii, rhs.ii)};
}

PerCoreTaskCost AddCost(const PerCoreTaskCost &lhs,
                        const PerCoreTaskCost &rhs) {
  return {CheckedAdd(lhs.latency, rhs.latency, "latency"),
          CheckedAdd(lhs.ii, rhs.ii, "II")};
}

PerCoreTaskCost MultiplyCost(const PerCoreTaskCost &cost, int64_t factor) {
  return {CheckedMultiply(cost.latency, factor, "latency"),
          CheckedMultiply(cost.ii, factor, "II")};
}

struct PhaseSummary {
  bool has_boundary{false};
  PerCoreTaskCost prefix;
  PerCoreTaskCost middle;
  PerCoreTaskCost suffix;
  size_t candidate_markers{0};
  size_t boundary_markers{0};
};

void AppendSummary(PhaseSummary *dst, const PhaseSummary &src) {
  dst->candidate_markers += src.candidate_markers;
  dst->boundary_markers += src.boundary_markers;
  if (!src.has_boundary) {
    if (dst->has_boundary) {
      dst->suffix = MaxCost(dst->suffix, src.prefix);
    } else {
      dst->prefix = MaxCost(dst->prefix, src.prefix);
    }
    return;
  }

  if (!dst->has_boundary) {
    dst->has_boundary = true;
    dst->prefix = MaxCost(dst->prefix, src.prefix);
    dst->middle = src.middle;
    dst->suffix = src.suffix;
    return;
  }

  dst->middle = AddCost(dst->middle, MaxCost(dst->suffix, src.prefix));
  dst->middle = AddCost(dst->middle, src.middle);
  dst->suffix = src.suffix;
}

PhaseSummary RepeatSummary(const PhaseSummary &body, int64_t trip_count) {
  if (trip_count <= 0) {
    PhaseSummary result;
    // Marker counts describe the static statement tree, not the number of
    // dynamic executions.  Preserve them even when the loop has no work so
    // validation does not misreport a missing Task or wait marker.
    result.candidate_markers = body.candidate_markers;
    result.boundary_markers = body.boundary_markers;
    return result;
  }
  PhaseSummary result = body;
  if (trip_count == 1)
    return result;

  if (!body.has_boundary) {
    // Without an inter-core wait there is no proof that different loop
    // iterations are mutually exclusive across cores. Keep the serial-loop
    // estimate conservative.
    result.prefix = MultiplyCost(body.prefix, trip_count);
    return result;
  }

  // One physical core executes its suffix from iteration i before its prefix
  // from iteration i + 1.  They are serial parts of the same compound task;
  // only distinct Task candidates within one phase may overlap and
  // use MaxCost.
  PerCoreTaskCost between_iterations = AddCost(body.suffix, body.prefix);
  result.middle = AddCost(MultiplyCost(body.middle, trip_count),
                          MultiplyCost(between_iterations, trip_count - 1));
  return result;
}

using CandidateCostMap =
    std::unordered_map<Stmt, PerCoreTaskCost, ObjectPtrHash, ObjectPtrEqual>;

class PhaseAnalyzer {
public:
  struct Result {
    PerCoreTaskCost cost;
    bool has_boundary{false};
  };

  static Result Analyze(const Stmt &stmt,
                        const CandidateCostMap &candidate_costs,
                        const ConstrSet &outer_ctx) {
    PhaseAnalyzer analyzer(candidate_costs);
    auto context = outer_ctx.Populate(analyzer.arith_analyzer_);
    PhaseSummary summary = analyzer.AnalyzeStmt(stmt);
    ICHECK_EQ(summary.candidate_markers, candidate_costs.size())
        << "Failed to match every T.Task marker while estimating "
           "PerCoreTask latency";
    ICHECK_EQ(summary.boundary_markers, analyzer.CountWaitMarkers(stmt))
        << "Failed to match every inter-core wait while estimating PerCoreTask "
           "latency";

    PerCoreTaskCost cost = summary.prefix;
    if (summary.has_boundary) {
      cost = AddCost(cost, summary.middle);
      cost = AddCost(cost, summary.suffix);
    }
    return {cost, summary.has_boundary};
  }

private:
  explicit PhaseAnalyzer(const CandidateCostMap &candidate_costs)
      : candidate_costs_(candidate_costs) {}

  static bool IsInterCoreWait(const EvaluateObj *evaluate) {
    const auto *call = evaluate->value.as<CallObj>();
    return call && call->op.same_as(tl::ascend_cross_core_wait_flag());
  }

  size_t CountWaitMarkers(const Stmt &stmt) const {
    class Counter : public StmtExprVisitor {
    public:
      using StmtExprVisitor::Visit;
      using StmtExprVisitor::Visit_;
      using Result = ffi::Optional<ffi::VisitInterrupt>;
      size_t count{0};
      Result Visit_(const CallObj *op) final {
        if (op->op.same_as(tl::ascend_cross_core_wait_flag()))
          ++count;
        if (IsAnnotationCall(op))
          return std::nullopt;
        return StmtExprVisitor::Visit_(op);

        return std::nullopt;
      }
      Result Visit_(const RegionStmtObj *op) final {
        return Visit(op->body);
        return std::nullopt;
      }
    };
    auto counter = make_object<Counter>();
    counter->Visit(stmt);
    return counter->count;
  }

  PhaseSummary AnalyzeStmt(const Stmt &stmt) const {
    if (auto candidate = candidate_costs_.find(stmt);
        candidate != candidate_costs_.end()) {
      PhaseSummary result;
      result.prefix = candidate->second;
      result.candidate_markers = 1;
      return result;
    }
    if (const auto *seq = stmt.as<SeqStmtObj>()) {
      PhaseSummary result;
      for (const Stmt &child : seq->seq)
        AppendSummary(&result, AnalyzeStmt(child));
      return result;
    }
    if (const auto *region = stmt.as<RegionStmtObj>())
      return AnalyzeStmt(region->body);

    if (const auto *evaluate = stmt.as<EvaluateObj>()) {
      if (IsInterCoreWait(evaluate)) {
        PhaseSummary result;
        result.has_boundary = true;
        result.boundary_markers = 1;
        return result;
      }
      return {};
    }
    if (const auto *loop = stmt.as<ForObj>()) {
      ICHECK(loop->kind == ForKind::kDefault ||
             loop->kind == ForKind::kUnrolled)
          << "Control flow selecting T.Task candidates must use a "
             "serial or unrolled loop";
      return RepeatSummary(
          AnalyzeStmt(loop->body),
          GetSerialTripCount(GetRef<For>(loop), arith_analyzer_.get()));
    }
    if (const auto *while_loop = stmt.as<WhileObj>()) {
      // Match ControlNode::GetTripCount's dynamic-loop fallback.
      return RepeatSummary(AnalyzeStmt(while_loop->body),
                           kDynamicSerialTripCountFallback);
    }
    if (const auto *branch = stmt.as<IfObj>()) {
      PhaseSummary then_summary = AnalyzeStmt(branch->then_case);
      PhaseSummary else_summary = branch->else_case
                                      ? AnalyzeStmt(branch->else_case.value())
                                      : PhaseSummary{};
      ICHECK(!then_summary.has_boundary && !else_summary.has_boundary)
          << "Inter-core synchronization inside T.PerCoreTask must be outside "
             "conditional control flow";
      PhaseSummary result;
      result.prefix = MaxCost(then_summary.prefix, else_summary.prefix);
      result.candidate_markers =
          then_summary.candidate_markers + else_summary.candidate_markers;
      result.boundary_markers =
          then_summary.boundary_markers + else_summary.boundary_markers;
      return result;
    }

    return {};
  }

  const CandidateCostMap &candidate_costs_;
  sym::Analyzer arith_analyzer_;
};

bool IsComposedLatencyStmt(const Stmt &stmt) {
  if (const auto *region = stmt.as<RegionStmtObj>();
      region && BlockNameAtSourceBoundary(region))
    return false;
  if (stmt.as<SeqStmtObj>() || stmt.as<RegionStmtObj>() || stmt.as<IfObj>() ||
      stmt.as<WhileObj>()) {
    return true;
  }
  if (const auto *loop = stmt.as<ForObj>()) {
    return loop->kind == ForKind::kDefault || loop->kind == ForKind::kUnrolled;
  }
  return false;
}

void TaskCostEstimator::Estimate(TaskNode *task) {
  ICHECK(task != nullptr);
  sym::Analyzer analyzer;
  auto context = task->outer_ctx.Populate(analyzer);
  TaskCostFeatures cost_features =
      CostFeatureAnalyzer::Analyze(task->stmt, analyzer.get(), hf32_modes_);
  EstimateAscend(task, cost_features, analyzer.get());
}

void TaskCostEstimator::EstimateComposedTask(TaskNode *task) {
  if (!IsComposedLatencyStmt(task->stmt)) {
    Estimate(task);
    return;
  }
  SerialTaskCost cost = EstimateStmtCost(task->stmt, task->outer_ctx);
  task->SetLatency(cost.latency);
  task->SetII(std::max<int64_t>(cost.ii, 1));
}

TaskCostEstimator::SerialTaskCost
TaskCostEstimator::EstimateLeafTaskCost(const Stmt &stmt,
                                        const ConstrSet &outer_ctx) {
  TaskNode leaf_task;
  leaf_task.stmt = stmt;
  leaf_task.outer_ctx = outer_ctx;
  ApplyTaskResourceUsage(AnalyzeTaskResourceUsage(stmt), &leaf_task);
  ApplyTaskAccesses(AnalyzeTaskAccesses(stmt), &leaf_task);
  Estimate(&leaf_task);
  return {leaf_task.GetLatency(), leaf_task.GetII()};
}

TaskCostEstimator::SerialTaskCost
TaskCostEstimator::EstimateStmtCost(const Stmt &stmt,
                                    const ConstrSet &outer_ctx) {
  if (IsTaskMetadataStatement(stmt) || tirx::IsNoOp(stmt))
    return {};

  auto add_cost = [&](const SerialTaskCost &lhs, const SerialTaskCost &rhs) {
    if (lhs.ii == 0)
      return rhs;
    if (rhs.ii == 0)
      return lhs;
    return SerialTaskCost{
        std::max(lhs.latency,
                 CheckedSerialTaskAdd(lhs.ii, rhs.latency, "latency")),
        CheckedSerialTaskAdd(lhs.ii, rhs.ii, "II")};
  };
  auto multiply_cost = [&](const SerialTaskCost &cost, int64_t factor) {
    if (factor <= 0 || cost.ii == 0)
      return SerialTaskCost{};
    return SerialTaskCost{
        CheckedSerialTaskAdd(
            CheckedSerialTaskMultiply(cost.ii, factor - 1, "latency"),
            cost.latency, "latency"),
        CheckedSerialTaskMultiply(cost.ii, factor, "II")};
  };

  if (const auto *seq = stmt.as<SeqStmtObj>()) {
    SerialTaskCost result;
    ConstrSet child_ctx = outer_ctx;
    for (const Stmt &child : seq->seq) {
      result = add_cost(result, EstimateStmtCost(child, child_ctx));
      if (auto bind = GetFlatTaskBind(child);
          bind && bind->value.as<PrimExpr>())
        child_ctx.AddConstr(bind->var, bind->value.as_or_throw<PrimExpr>());
    }
    return result;
  }
  if (const auto *region = stmt.as<RegionStmtObj>()) {
    if (BlockNameAtSourceBoundary(region))
      return EstimateLeafTaskCost(stmt, outer_ctx);
    ConstrSet body_ctx = outer_ctx;
    if (auto attr = ReadSourceAttribute(region);
        attr && IsScheduleGuardAttribute(attr->key)) {
      if (auto condition = attr->node.try_cast<PrimExpr>();
          condition && SideEffect(*condition) <= CallEffectKind::kPure)
        body_ctx.AddConstr(*condition, true);
    }
    return EstimateStmtCost(region->body, body_ctx);
  }
  if (const auto *loop = stmt.as<ForObj>()) {
    if (loop->kind != ForKind::kDefault && loop->kind != ForKind::kUnrolled)
      return EstimateLeafTaskCost(stmt, outer_ctx);

    sym::Analyzer analyzer;
    auto context = outer_ctx.Populate(analyzer);
    int64_t trip_count = GetSerialTripCount(GetRef<For>(loop), analyzer.get());
    ConstrSet body_ctx = outer_ctx;
    if (analyzer->CanProve(loop->step > 0)) {
      body_ctx.AddConstr(loop->var, Range(loop->begin, loop->end));
    }
    body_ctx.AddConstr(loop->var < loop->end);
    body_ctx.AddConstr(loop->begin < loop->end);
    return multiply_cost(EstimateStmtCost(loop->body, body_ctx), trip_count);
  }
  if (const auto *while_loop = stmt.as<WhileObj>()) {
    ConstrSet body_ctx = outer_ctx;
    body_ctx.AddConstr(while_loop->condition);
    return multiply_cost(EstimateStmtCost(while_loop->body, body_ctx),
                         kDynamicSerialTripCountFallback);
  }
  if (const auto *branch = stmt.as<IfObj>()) {
    ConstrSet then_ctx = outer_ctx;
    then_ctx.AddConstr(branch->condition);
    SerialTaskCost then_cost = EstimateStmtCost(branch->then_case, then_ctx);
    SerialTaskCost else_cost;
    if (branch->else_case) {
      ConstrSet else_ctx = outer_ctx;
      else_ctx.AddConstr(Not(branch->condition));
      else_cost = EstimateStmtCost(branch->else_case.value(), else_ctx);
    }
    return {std::max(then_cost.latency, else_cost.latency),
            std::max(then_cost.ii, else_cost.ii)};
  }
  return EstimateLeafTaskCost(stmt, outer_ctx);
}

void TaskCostEstimator::EstimatePerCoreTask(
    TaskNode *per_core_task, const Stmt &body,
    const std::vector<PerCoreTaskCandidate> &candidates) {
  CandidateCostMap candidate_costs;
  for (const PerCoreTaskCandidate &candidate : candidates) {
    ICHECK(candidate.task);
    ICHECK(HasTaskBufferAccess(candidate.task.get()))
        << "T.Task must contain at least one dependency-bearing "
           "statement";
    bool inserted = candidate_costs
                        .emplace(candidate.marker,
                                 PerCoreTaskCost{candidate.task->GetLatency(),
                                                 candidate.task->GetII()})
                        .second;
    ICHECK(inserted) << "Duplicate T.Task marker identity";
  }

  PhaseAnalyzer::Result phase_cost =
      PhaseAnalyzer::Analyze(body, candidate_costs, per_core_task->outer_ctx);
  per_core_task->SetLatency(phase_cost.cost.latency);
  // Mode-0 inter-core flags are explicit, unversioned protocol state. Keep
  // dynamic protocol invocations non-overlapping until a versioned flag
  // contract is introduced.
  per_core_task->SetII(std::max<int64_t>(
      phase_cost.has_boundary ? phase_cost.cost.latency : phase_cost.cost.ii,
      1));
}

// EstimateLatency pass.
using TaskCostMap =
    std::unordered_map<Stmt, TaskCost, ObjectPtrHash, ObjectPtrEqual>;

class TaskCostCollector : public TaskAwareConstrVisitor {
public:
  using TaskAwareConstrVisitor::Visit;
  using TaskAwareConstrVisitor::Visit_;
  using Result = ffi::Optional<ffi::VisitInterrupt>;
  static TaskCostMap Collect(const Stmt &body, const ConstrSet &outer_ctx) {
    auto collector = make_object<TaskCostCollector>(body, outer_ctx);
    collector->Visit(body);
    return std::move(collector->costs_);
  }

public:
  TaskCostCollector(const Stmt &body, const ConstrSet &outer_ctx)
      : estimator_(Hf32ModeAnalyzer::Analyze(body)) {
    constr_stack_ = outer_ctx.constrs_;
  }

  bool InScheduleBody() const { return schedule_body_depth_ > 0; }

  std::shared_ptr<TaskNode> NewTask(const Stmt &stmt) const {
    auto task = std::make_shared<TaskNode>();
    task->stmt = stmt;
    task->outer_ctx = GetConstrSet();
    return task;
  }

  void RecordCost(const Stmt &key, const std::shared_ptr<TaskNode> &task,
                  const TaskMetadata &metadata) {
    TaskCost estimated{task->GetLatency(), task->GetII()};
    TaskCost cost = ResolveTaskCost(estimated, metadata);
    task->SetLatency(cost.latency);
    task->SetII(cost.ii);
    costs_[key] = cost;
  }

  void EstimateMarkedTask(const Stmt &marker, const Stmt &body,
                          const Any &node) {
    TaskMetadata metadata = ParseTaskMetadata(node);
    // Projected tasks describe one AIV's traffic; a collective task
    // describes the same operation on both AIVs.
    auto attributes = task_metadata_detail::CopyTaskMetadata(node);
    estimator_.SetAivLanes(attributes.count("subcore") ? 1 : 2);
    auto task = NewTask(body);
    ApplyTaskResourceUsage(AnalyzeTaskResourceUsage(body), task.get());
    ApplyTaskAccesses(AnalyzeTaskAccesses(body), task.get());
    estimator_.EstimateComposedTask(task.get());
    RecordCost(marker, task, metadata);
    if (per_core_task_depth_ > 0 && HasTaskBufferAccess(task.get()))
      per_core_task_candidates_.push_back({marker, task});
  }

  void EstimatePerCoreTask(const RegionStmtObj *op) {
    ICHECK_EQ(per_core_task_depth_, 0)
        << "Nested T.PerCoreTask regions are not supported";
    TaskMetadata metadata = ParseTaskMetadata(TaskAnnotations(op));

    std::vector<PerCoreTaskCandidate> saved_candidates;
    saved_candidates.swap(per_core_task_candidates_);
    ++per_core_task_depth_;
    Visit(op->body);
    --per_core_task_depth_;
    std::vector<PerCoreTaskCandidate> candidates;
    candidates.swap(per_core_task_candidates_);
    per_core_task_candidates_.swap(saved_candidates);

    ICHECK(!candidates.empty())
        << "T.PerCoreTask must contain at least one normalized "
           "T.Task "
           "candidate before EstimateLatency";
    auto task = NewTask(op->body);
    TaskResourceUsage resource_usage = AnalyzeTaskResourceUsage(op->body);
    uint16_t pipe = resource_usage.pipe_mask;
    ICHECK_NE(pipe, 0) << "Cannot infer an Ascend pipe for T.PerCoreTask";
    ICHECK_EQ(pipe & (pipe - 1), 0)
        << "T.PerCoreTask requires exactly one inferred hardware pipe, but "
           "its body uses mask="
        << pipe;
    for (const PerCoreTaskCandidate &candidate : candidates) {
      uint16_t candidate_pipe = candidate.task->GetPipeMask();
      ICHECK(candidate_pipe != 0 &&
             (candidate_pipe & (candidate_pipe - 1)) == 0)
          << "Each dependency-bearing statement in T.PerCoreTask "
             "must use "
             "exactly one hardware pipe, but statement mask="
          << candidate_pipe << ", statement=" << candidate.task->stmt;
      ICHECK_EQ(candidate_pipe, pipe)
          << "T.PerCoreTask candidate pipe does not match its "
             "inferred pipe. Inferred mask="
          << pipe << ", statement mask=" << candidate_pipe
          << ", statement=" << candidate.task->stmt;
    }
    ApplyTaskResourceUsage(resource_usage, task.get());
    estimator_.EstimatePerCoreTask(task.get(), op->body, candidates);
    RecordCost(GetRef<Stmt>(op), task, metadata);
  }

  Result Visit_(const ForObj *op) final {
    if (!InScheduleBody()) {
      ConstrVisitor::Visit_(op);
      return std::nullopt;
    }
    if (op->kind == ForKind::kDefault || op->kind == ForKind::kUnrolled) {
      ConstrVisitor::Visit_(op);
      return std::nullopt;
    }
    FatalUnmarkedTask("EstimateLatency", "parallel loop");

    return std::nullopt;
  }

  Result Visit_(const EvaluateObj *op) final {
    if (IsTaskMetadataStatement(GetRef<Stmt>(op)) ||
        tirx::IsNoOp(GetRef<Stmt>(op)))
      return std::nullopt;
    FatalUnmarkedTask("EstimateLatency", "Evaluate");
    return std::nullopt;
  }

  Result Visit_(const TensorStoreObj *op) final {
    if (!InScheduleBody()) {
      ConstrVisitor::Visit_(op);
      return std::nullopt;
    }
    FatalUnmarkedTask("EstimateLatency", "BufferStore");

    return std::nullopt;
  }

  Result Visit_(const BindObj *op) final {
    if (IsScheduledBind(op)) {
      auto binding = ReadScheduledBinding(op->value.as<CallObj>());
      EstimateMarkedTask(GetRef<Stmt>(op),
                         Bind(op->var, binding.value, op->loc), binding.task);
      return std::nullopt;
    }
    FatalUnmarkedTask("EstimateLatency", "Bind");
    return std::nullopt;
  }

  Result Visit_(const WhileObj *op) final {
    if (!InScheduleBody()) {
      ConstrVisitor::Visit_(op);
      return std::nullopt;
    }
    FatalUnmarkedTask("EstimateLatency", "While");

    return std::nullopt;
  }

  Result Visit_(const RegionStmtObj *op) final {
    if (IsAscendBlock(op))
      FatalUnmarkedTask("EstimateLatency", "block");
    if (IsTask(op)) {
      EstimateMarkedTask(GetRef<Stmt>(op), op->body, TaskAnnotations(op));
      return std::nullopt;
    }
    if (IsPerCoreTask(op)) {
      EstimatePerCoreTask(op);
      return std::nullopt;
    }
    if (SourceBlockName(op))
      FatalUnmarkedTask("EstimateLatency", "block");
    return ConstrVisitor::Visit_(op);

    return std::nullopt;
  }

  TaskCostEstimator estimator_;
  TaskCostMap costs_;
  int schedule_body_depth_{1};
  int per_core_task_depth_{0};
  std::vector<PerCoreTaskCandidate> per_core_task_candidates_;
};

class TaskAnnotationRewriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit TaskAnnotationRewriter(const TaskCostMap &costs) : costs_(costs) {}
  UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    auto found = costs_.find(GetRef<Stmt>(op));
    if (found == costs_.end())
      return StmtExprMutator::Mutate_(op, mode);
    auto binding = ReadScheduledBinding(op->value.as<CallObj>());
    TaskMetadata metadata;
    metadata.latency = found->second.latency;
    metadata.ii = found->second.ii;
    return WithScheduledBindingMetadata(
        GetRef<Bind>(op), binding.stage,
        MergeTaskMetadata(metadata, binding.task).cast<ScheduleUnitMetadata>());
  }
  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    auto found = costs_.find(GetRef<Stmt>(op));
    if (found == costs_.end())
      return StmtExprMutator::Mutate_(op, mode);
    ICHECK(IsSemanticTask(op));
    RegionStmt result = GetRef<RegionStmt>(op);
    if (IsPerCoreTask(op))
      result.CopyOnWrite()->body = Mutate(op->body).ValueOrUnchanged(op->body);
    TaskMetadata metadata;
    metadata.latency = found->second.latency;
    metadata.ii = found->second.ii;
    return WithTaskAnnotations(
        result, MergeTaskMetadata(metadata, TaskAnnotations(op)));
  }

private:
  const TaskCostMap &costs_;
};

SeqStmt RunEstimateLatency(const SeqStmt &body, const ConstrSet &outer_ctx) {
  TaskCostMap costs = TaskCostCollector::Collect(body, outer_ctx);
  ICHECK(!costs.empty())
      << "EstimateLatency found no schedulable tasks inside tilelang_root";
  auto rewriter = make_object<TaskAnnotationRewriter>(costs);
  return rewriter->Mutate(body).ValueOrUnchanged(body);
}

} // namespace

tvm::transform::Pass EstimateLatency() {
  auto pass_func = [](tirx::Function func, const IRModule &,
                      const PassContext &) {
    return RewriteTilelangKernels(
        std::move(func), "EstimateLatency",
        [](const TilelangKernelContext &context) {
          RegionStmt root = context.root;
          root.CopyOnWrite()->body =
              RunEstimateLatency(root->body, context.outer_ctx);
          return root;
        },
        /*require_kernel=*/false);
  };
  return CreateFunctionPass(pass_func, 0, "tl.EstimateLatency");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.EstimateLatency", EstimateLatency);
}

} // namespace tl
} // namespace tvm
