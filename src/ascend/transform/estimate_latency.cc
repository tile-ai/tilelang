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
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "./auto_schedule/ir_structure.h"
#include "./auto_schedule/kernel_rewriter.h"
#include "./auto_schedule/task_analysis.h"
#include "./auto_schedule/task_annotations.h"
#include "./estimate_latency.h"
#include "ascend/op/ascend_mte_plan.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/gemm.h"
#include "transform/common/attr.h"
#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;
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

// Estimator-only operation details. They deliberately do not live on
// AutoSchedule's TaskNode: the scheduler consumes the resulting latency/II,
// not the copy descriptors or Cube shapes used to derive them.
struct TaskCostFeatures {
  struct CubeShape {
    int64_t m;
    int64_t n;
    int64_t k;
    DataType input_dtype;
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
    // rows, so it does not use raw N-strided row-width fits.
    bool is_nd2nz_post_copy;
  };

  std::vector<CubeShape> cube_shapes;
  std::vector<CopyInfo> copy_infos;
};

// Internal cost model for Ascend AutoSchedule tasks. The public transform pass
// is implemented separately in estimate_latency.cc.
class TaskCostEstimator {
public:
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
    struct VFLatencyReader : public StmtVisitor {
      void VisitStmt_(const SBlockNode *op) override {
        if (op->name_hint == "SIMT_VF" || op->name_hint == "SIMD_VF") {
          auto it = op->annotations.find("tl.vf_latency");
          if (it != op->annotations.end()) {
            if (auto *imm = (*it).second.as<IntImmNode>()) {
              latency = std::max(latency, imm->value);
            }
          }
        }
        StmtVisitor::VisitStmt_(op);
      }
      int64_t latency{0};
    };

    VFLatencyReader reader;
    reader(task->stmt);
    return reader.latency;
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

  int64_t AivMteIIBytes(AscendPath path, int64_t per_aiv_bytes,
                        int64_t normalized_bytes) const {
    // Tiny GM packets are descriptor-limited. Larger packets and every
    // UB->L1 transfer use the common two-AIV calibration unit.
    switch (path) {
    case AscendPath::kGmToUb:
      return normalized_bytes > params_.mte2_gm_to_ub_packet_floor_bytes
                 ? normalized_bytes
                 : per_aiv_bytes;
    case AscendPath::kUbToGm:
      return normalized_bytes > params_.mte3_ub_to_gm_packet_floor_bytes
                 ? normalized_bytes
                 : per_aiv_bytes;
    case AscendPath::kUbToL1:
      return normalized_bytes;
    default:
      return per_aiv_bytes;
    }
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
  // `transaction_bytes`-sized MTE transactions. The measured row rate peaks
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
    int64_t normalized_bytes = ScaleBytesByLanes(info.bytes, 2);
    int64_t bandwidth = BandwidthForCopy(path, info);
    int64_t completion_latency =
        params_.mte_descriptor_cycles +
        MemLatencyForPath(path, normalized_bytes, bandwidth,
                          BaseLatencyForCopy(path, info));
    return {completion_latency,
            AivMteIIBytes(path, info.bytes, normalized_bytes), bandwidth};
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

  // Per-dtype Cube throughput (MACs/cycle).
  int64_t CubeThroughputFor(DataType dtype) const {
    if (dtype.is_float16())
      return params_.cube_throughput_fp16 ? params_.cube_throughput_fp16
                                          : params_.cube_fallback_throughput;
    if (dtype.is_bfloat16())
      return params_.cube_throughput_bf16 ? params_.cube_throughput_bf16
                                          : params_.cube_fallback_throughput;
    if (dtype.is_float() && dtype.bits() == 32)
      return params_.cube_throughput_fp32 ? params_.cube_throughput_fp32
                                          : params_.cube_fallback_throughput;
    if (dtype.is_float8() || (dtype.is_float() && dtype.bits() == 8))
      return params_.cube_throughput_fp8 ? params_.cube_throughput_fp8
                                         : params_.cube_fallback_throughput;
    if (dtype.is_int() && dtype.bits() == 8)
      return params_.cube_throughput_int8 ? params_.cube_throughput_int8
                                          : params_.cube_fallback_throughput;
    return params_.cube_fallback_throughput;
  }

  // Ascend latency / II estimation entry point.
  void EstimateAscend(TaskNode *task, const TaskCostFeatures &cost_features,
                      arith::Analyzer *analyzer) {
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
      auto visit_region = [&](const BufferRegion &region, bool is_read) {
        const Buffer &buffer = region->buffer;
        String scope = buffer.scope();
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
        OperationCounter counter(&params_);
        counter(task->stmt);
        int64_t op_count_latency = counter.GetEstimatedLatency();

        // Bandwidth-bound path: total bytes the VALU consumes / valu_bandwidth.
        int64_t valu_bytes = 0;
        for (const auto &r : task->GetReadRegions()) {
          String scope = r->buffer.scope();
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
      // matmul latency = base + total_MACs / per-dtype throughput.
      // Use recorded GEMM shapes, or fall back to one unit MMAD.
      int64_t matmul_latency = params_.cube_base_latency;
      if (!cost_features.cube_shapes.empty()) {
        for (const TaskCostFeatures::CubeShape &shape :
             cost_features.cube_shapes) {
          int64_t ops = shape.m * shape.n * shape.k * 2;
          int64_t throughput = CubeThroughputFor(shape.input_dtype);
          matmul_latency += safe_div_ceil(ops, throughput);
        }
      } else if (params_.cube_unit_m > 0 && params_.cube_unit_k > 0 &&
                 params_.cube_unit_n > 0) {
        // Single unit mad fallback.
        int64_t unit_ops =
            params_.cube_unit_m * params_.cube_unit_k * params_.cube_unit_n;
        int64_t throughput = CubeThroughputFor(DataType::Float(16));
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
          int64_t ops = shape.m * shape.n * shape.k * 2;
          int64_t throughput = CubeThroughputFor(shape.input_dtype);
          cube_ii = std::max(cube_ii, safe_div_ceil(ops, throughput));
        }
      }
      cube_ii = std::max(cube_ii, params_.cube_pipeline_depth);
    }

    int64_t vector_ii = 1;
    if (has_vec) {
      if (measured_vf_latency > 0) {
        vector_ii = measured_vf_latency;
      } else {
        int64_t denom =
            params_.vector_queue_depth > 0 ? params_.vector_queue_depth : 1;
        vector_ii = std::max(vector_ii, safe_div_ceil(vector_latency, denom));
      }
    }

    if (has_dma && !has_cube && !has_vec) {
      // DMA-only: II = max bandwidth-limited II across active paths.
      ii = std::max(ii, memory_ii);
    } else if (has_cube && !has_dma && !has_vec) {
      // Cube-only: II is bounded by pipeline depth and MAC throughput.
      ii = std::max(ii, cube_ii);
    } else if (has_vec && !has_dma && !has_cube) {
      // Vector-only: II = compute_latency / vector_queue_depth (fallback to
      // compute_latency when queue depth is unknown).
      ii = std::max(ii, vector_ii);
    } else {
      // Mixed / fallback: keep task granularity, but bound II by the slowest
      // active pipe instead of forcing it to the full sequential latency.
      if (has_dma)
        ii = std::max(ii, memory_ii);
      if (has_cube)
        ii = std::max(ii, cube_ii);
      if (has_vec)
        ii = std::max(ii, vector_ii);
    }

    task->SetLatency(total_latency);
    task->SetII(ii);
  }

  // VALU operation counter.
  class OperationCounter : public StmtExprVisitor {
  public:
    // Loop dimension information
    struct LoopDimension {
      int64_t trip_count;
      int depth;
    };

    int64_t total_latency = 0;
    const AscendLatencyParams *params = nullptr;

    // Track loop dimensions for loop-invariant detection
    std::vector<LoopDimension> loop_stack;
    std::unordered_map<const VarNode *, int> var_to_depth;

    explicit OperationCounter(const AscendLatencyParams *params)
        : params(params) {}

    // Operator to visit a statement
    void operator()(const Stmt &stmt) { VisitStmt(stmt); }

    // Get estimated latency
    int64_t GetEstimatedLatency() const { return total_latency; }

    // Helper function to update latency for an operation based on throughput
    void update_operation(
        int64_t throughput,
        const std::unordered_set<const VarNode *> &contained_vars) {
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
    std::unordered_set<const VarNode *>
    AnalyzeContainedLoopVars(const PrimExpr &expr) {
      class VarCollector : public ExprVisitor {
      public:
        const std::unordered_map<const VarNode *, int> &var_to_depth;
        std::unordered_set<const VarNode *> collected_vars;

        VarCollector(
            const std::unordered_map<const VarNode *, int> &var_to_depth)
            : var_to_depth(var_to_depth) {}

        void VisitExpr_(const VarNode *op) final {
          if (var_to_depth.count(op)) {
            collected_vars.insert(op);
          }
          ExprVisitor::VisitExpr_(op);
        }
      };

      VarCollector collector(var_to_depth);
      collector(expr);
      return collector.collected_vars;
    }

    // Calculate effective parallel factor based on contained loop variables
    int64_t CalculateEffectiveParallelFactor(
        const std::unordered_set<const VarNode *> &contained_vars) {
      if (contained_vars.empty()) {
        return 1; // Completely loop-invariant
      }

      // Find the maximum depth of contained loop variables
      int max_depth = -1;
      for (const VarNode *var : contained_vars) {
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

    void VisitExpr_(const AddNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->add_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const SubNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->sub_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const MulNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->mul_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const DivNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->div_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const ModNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->mod_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const FloorDivNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->div_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const FloorModNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->mod_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const LTNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const LENode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const GTNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const GENode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const EQNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const NENode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const AndNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->logic_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const OrNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->logic_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const NotNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->logic_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const MinNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->min_max_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const MaxNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->min_max_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const CallNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));

      // Check for special math functions by name
      if (op->op.as<OpNode>()) {
        auto op_node = op->op.as<OpNode>();
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
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const ForNode *op) final {
      // Calculate trip count for the loop
      int64_t trip_count = 1;
      PrimExpr loop_extent = op->extent;
      PrimExpr loop_step = op->step.has_value() ? op->step.value()
                                                : IntImm(DataType::Int(32), 1);

      // Try to get constant values
      if (const int64_t *extent_ptr = as_const_int(loop_extent)) {
        if (const int64_t *step_ptr = as_const_int(loop_step)) {
          int64_t extent = *extent_ptr;
          int64_t step = *step_ptr;
          if (step > 0) {
            // ceil(extent / step) = (extent + step - 1) / step
            trip_count = (extent + step - 1) / step;
          } else {
            trip_count = extent; // Invalid step, use extent
          }
        } else {
          trip_count = 100; // Non-constant step, use default
        }
      } else {
        trip_count = 100; // Non-constant extent, use default
      }

      // Create loop dimension information
      LoopDimension loop_dim{.trip_count = trip_count,
                             .depth = static_cast<int>(loop_stack.size())};

      // Push loop onto stack and update mapping
      loop_stack.push_back(loop_dim);
      var_to_depth[op->loop_var.get()] = loop_dim.depth;

      // Visit loop body
      StmtExprVisitor::VisitStmt_(op);

      // Pop loop from stack
      var_to_depth.erase(op->loop_var.get());
      loop_stack.pop_back();
    }

    void VisitStmt_(const EvaluateNode *op) final {
      if (op->value.defined()) {
        StmtExprVisitor::VisitExpr(op->value);
      }
    }

    void VisitStmt_(const BufferStoreNode *op) final {
      // Count operations in indices
      for (const auto &index : op->indices) {
        StmtExprVisitor::VisitExpr(index);
      }
      // Count operations in value
      StmtExprVisitor::VisitExpr(op->value);
    }

    void VisitStmt_(const SeqStmtNode *op) final {
      for (const auto &child : op->seq) {
        StmtExprVisitor::VisitStmt(child);
      }
    }

    void VisitStmt_(const AttrStmtNode *op) final {
      StmtExprVisitor::VisitStmt(op->body);
    }

    void VisitStmt_(const BindNode *op) final {
      // Let binding: the value expression is evaluated once
      StmtExprVisitor::VisitExpr(op->value);
    }

    void VisitStmt_(const IfThenElseNode *op) final {
      VisitExpr(op->condition);

      // For if-then-else branches, we need to take the maximum latency of both
      // paths Save current latency
      int64_t old_latency = total_latency;

      // Count latency in then branch
      StmtExprVisitor::VisitStmt(op->then_case);
      int64_t then_latency = total_latency;

      // Restore latency and count else branch
      total_latency = old_latency;
      if (op->else_case) {
        StmtExprVisitor::VisitStmt(op->else_case.value());
      }
      int64_t else_latency = total_latency;

      // Take the maximum of both branches
      total_latency = old_latency + std::max(then_latency - old_latency,
                                             else_latency - old_latency);
    }

    void VisitStmt_(const SBlockNode *op) final {
      StmtExprVisitor::VisitStmt(op->body);
    }
  };

  // Helper function to calculate total bytes accessed in a region.
  int64_t CalculateAccessBytes(const BufferRegion &region,
                               arith::Analyzer *analyzer) {
    RegionElementUpperBound estimate = EstimateRegionElementUpperBound(
        region->buffer, region->region, analyzer);
    return ElementsToBytes(estimate.elements, DataType(region->buffer->dtype));
  }
};

class CostFeatureAnalyzer : public StmtExprVisitor {
public:
  static TaskCostFeatures Analyze(const Stmt &stmt,
                                  arith::Analyzer *arith_analyzer) {
    ICHECK(arith_analyzer != nullptr);
    CostFeatureAnalyzer analyzer(arith_analyzer);
    analyzer(stmt);
    return analyzer.features_;
  }

private:
  explicit CostFeatureAnalyzer(arith::Analyzer *arith_analyzer)
      : arith_analyzer_(arith_analyzer) {}

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
                           ? Range::FromMinExtent(ranges[i]->min, extent)
                           : ranges[i]);
    }
    return result;
  }

  MteGeometry InferMteGeometry(const CopyNode *copy) const {
    Array<Range> src_ranges = copy->src_range;
    Array<Range> dst_ranges = copy->dst_range;
    if (src_ranges.empty() || dst_ranges.empty() ||
        src_ranges.size() != copy->src->shape.size() ||
        dst_ranges.size() != copy->dst->shape.size()) {
      return {};
    }

    src_ranges = NormalizeEmptyUnitAxesForMTE(src_ranges, arith_analyzer_);
    dst_ranges = NormalizeEmptyUnitAxesForMTE(dst_ranges, arith_analyzer_);

    int dual_dst_ctl = copy->GetDualDstCtl();
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

    int src_bits = copy->src->dtype.bits() * copy->src->dtype.lanes();
    int dst_bits = copy->dst->dtype.bits() * copy->dst->dtype.lanes();
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
    }
    int64_t src_row_elements = EstimateExprPositiveLowerBound(src_row_size);
    int64_t dst_row_elements = EstimateExprPositiveLowerBound(dst_row_size);
    if (src_row_elements <= 0 || dst_row_elements <= 0)
      return {kind, 0};
    int64_t src_row_bytes =
        ElementsToBytes(src_row_elements, DataType(copy->src->dtype));
    int64_t dst_row_bytes =
        ElementsToBytes(dst_row_elements, DataType(copy->dst->dtype));
    return {kind, std::min(src_row_bytes, dst_row_bytes)};
  }

  int64_t EstimateExprUpperBound(const PrimExpr &expr) const {
    if (const int64_t *value = as_const_int(expr))
      return std::max<int64_t>(*value, 0);
    arith::ConstIntBound bound = arith_analyzer_->const_int_bound(expr);
    if (bound->max_value >= 0 &&
        bound->max_value != arith::ConstIntBound::kPosInf) {
      return bound->max_value;
    }
    return 1;
  }

  int64_t EstimateExprPositiveLowerBound(const PrimExpr &expr) const {
    if (const int64_t *value = as_const_int(expr))
      return std::max<int64_t>(*value, 0);
    arith::ConstIntBound bound = arith_analyzer_->const_int_bound(expr);
    return bound->min_value > 0 ? bound->min_value : 0;
  }

  int64_t CalculateCopyBytes(const CopyNode *copy) const {
    RegionElementUpperBound src_estimate = EstimateRegionElementUpperBound(
        copy->src, copy->src_range, arith_analyzer_);
    RegionElementUpperBound dst_estimate = EstimateRegionElementUpperBound(
        copy->dst, copy->dst_range, arith_analyzer_);

    if (copy->GetDualDstCtl() == 1 || copy->GetDualDstCtl() == 2) {
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
      return ElementsToBytes(elements, DataType(copy->dst->dtype));
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

    return ElementsToBytes(elements, DataType(copy->dst->dtype));
  }

  void RecordNd2NzPostCopy(const CallNode *op) {
    ICHECK_EQ(op->args.size(), 6)
        << "tl.ascend_nd2nz_post_copy expects 6 arguments";
    const auto *dtype = op->args[5].as<StringImmNode>();
    ICHECK(dtype) << "tl.ascend_nd2nz_post_copy dtype must be a StringImm";
    int64_t element_bytes = dtype->value == "float" ? 4 : 2;
    int64_t rows = EstimateExprUpperBound(op->args[2]);
    int64_t cols = EstimateExprUpperBound(op->args[3]);
    int64_t bytes = SafeMultiply(SafeMultiply(rows, cols), element_bytes);

    features_.copy_infos.push_back({String("shared.dyn"), String("shared.l1"),
                                    bytes, /*dual_dst_ctl=*/0,
                                    /*geometry=*/CopyGeometry::kM,
                                    /*contiguous_bytes_lower_bound=*/0,
                                    /*is_nd2nz_post_copy=*/true});
  }

  void VisitExpr_(const CallNode *op) final {
    static const auto copy_op = Op::Get("tl.tileop.copy");
    static const auto gemm_op = Op::Get("tl.tileop.gemm");
    if (op->op.same_as(copy_op)) {
      Copy copy_obj(op->args, op->annotations);
      const CopyNode *copy = copy_obj.get();
      MteGeometry geometry = InferMteGeometry(copy);
      features_.copy_infos.push_back({copy->src.scope(), copy->dst.scope(),
                                      CalculateCopyBytes(copy),
                                      copy->GetDualDstCtl(), geometry.kind,
                                      geometry.contiguous_bytes_lower_bound,
                                      /*is_nd2nz_post_copy=*/false});
    } else if (op->op.same_as(tl::ascend_nd2nz_post_copy())) {
      RecordNd2NzPostCopy(op);
    } else if (op->op.same_as(gemm_op)) {
      int64_t m = op->args[5].as<IntImmNode>()->value;
      int64_t n = op->args[6].as<IntImmNode>()->value;
      int64_t k = op->args[7].as<IntImmNode>()->value;
      Gemm gemm(op->args, op->annotations);
      if (m > 0 && n > 0 && k > 0)
        features_.cube_shapes.push_back({m, n, k, gemm->a_->dtype});
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  // Resource analysis treats blocks as atomic scheduler operations. Preserve
  // the same boundary for cost features; VF cost is modeled from the block.
  void VisitStmt_(const SBlockNode *) final {}

  arith::Analyzer *arith_analyzer_;
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
    outer_ctx.Populate(analyzer.arith_analyzer_);
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

  static bool IsInterCoreWait(const EvaluateNode *evaluate) {
    const auto *call = evaluate->value.as<CallNode>();
    return call && call->op.same_as(tl::ascend_cross_core_wait_flag());
  }

  size_t CountWaitMarkers(const Stmt &stmt) const {
    class Counter : public StmtExprVisitor {
    public:
      size_t count{0};

    private:
      void VisitExpr_(const CallNode *op) final {
        if (op->op.same_as(tl::ascend_cross_core_wait_flag()))
          ++count;
        StmtExprVisitor::VisitExpr_(op);
      }
    } counter;
    counter(stmt);
    return counter.count;
  }

  PhaseSummary AnalyzeStmt(const Stmt &stmt) const {
    if (!stmt.defined())
      return {};
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      PhaseSummary result;
      for (const Stmt &child : seq->seq)
        AppendSummary(&result, AnalyzeStmt(child));
      return result;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == tl::attr::kAscendTask) {
        Stmt marker = GetRef<Stmt>(attr);
        auto it = candidate_costs_.find(marker);
        if (it != candidate_costs_.end()) {
          PhaseSummary result;
          result.prefix = it->second;
          result.candidate_markers = 1;
          return result;
        }
        return AnalyzeStmt(attr->body);
      }
      return AnalyzeStmt(attr->body);
    }
    if (const auto *evaluate = stmt.as<EvaluateNode>()) {
      if (IsInterCoreWait(evaluate)) {
        PhaseSummary result;
        result.has_boundary = true;
        result.boundary_markers = 1;
        return result;
      }
      return {};
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      ICHECK(loop->kind == ForKind::kSerial || loop->kind == ForKind::kUnrolled)
          << "Control flow selecting T.Task candidates must use a "
             "serial or unrolled loop";
      return RepeatSummary(
          AnalyzeStmt(loop->body),
          GetSerialTripCount(GetRef<For>(loop), &arith_analyzer_));
    }
    if (const auto *while_loop = stmt.as<WhileNode>()) {
      // Match ControlNode::GetTripCount's dynamic-loop fallback.
      return RepeatSummary(AnalyzeStmt(while_loop->body),
                           kDynamicSerialTripCountFallback);
    }
    if (const auto *branch = stmt.as<IfThenElseNode>()) {
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
    if (const auto *block = stmt.as<SBlockNode>()) {
      PhaseSummary result;
      if (block->init)
        AppendSummary(&result, AnalyzeStmt(block->init.value()));
      AppendSummary(&result, AnalyzeStmt(block->body));
      return result;
    }
    if (const auto *realize = stmt.as<SBlockRealizeNode>())
      return AnalyzeStmt(realize->block);
    return {};
  }

  const CandidateCostMap &candidate_costs_;
  mutable arith::Analyzer arith_analyzer_;
};

bool IsComposedLatencyStmt(const Stmt &stmt) {
  if (stmt.as<SeqStmtNode>() || stmt.as<AttrStmtNode>() ||
      stmt.as<IfThenElseNode>() || stmt.as<WhileNode>()) {
    return true;
  }
  if (const auto *loop = stmt.as<ForNode>()) {
    return loop->kind == ForKind::kSerial || loop->kind == ForKind::kUnrolled;
  }
  return false;
}

void TaskCostEstimator::Estimate(TaskNode *task) {
  ICHECK(task != nullptr);
  arith::Analyzer analyzer;
  task->outer_ctx.Populate(analyzer);
  TaskCostFeatures cost_features =
      CostFeatureAnalyzer::Analyze(task->stmt, &analyzer);
  EstimateAscend(task, cost_features, &analyzer);
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
  if (!stmt.defined())
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

  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    SerialTaskCost result;
    ConstrSet child_ctx = outer_ctx;
    for (const Stmt &child : seq->seq) {
      result = add_cost(result, EstimateStmtCost(child, child_ctx));
      if (const auto *bind = child.as<BindNode>())
        child_ctx.AddConstr(bind->var, bind->value);
    }
    return result;
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    ConstrSet body_ctx = outer_ctx;
    if (attr->attr_key == tirx::attr::tilelang_assume) {
      if (const auto *expr = attr->node.as<PrimExprNode>())
        body_ctx.AddConstr(GetRef<PrimExpr>(expr), /*is_assume=*/true);
    }
    return EstimateStmtCost(attr->body, body_ctx);
  }
  if (const auto *loop = stmt.as<ForNode>()) {
    if (loop->kind != ForKind::kSerial && loop->kind != ForKind::kUnrolled)
      return EstimateLeafTaskCost(stmt, outer_ctx);

    arith::Analyzer analyzer;
    outer_ctx.Populate(analyzer);
    int64_t trip_count = GetSerialTripCount(GetRef<For>(loop), &analyzer);
    ConstrSet body_ctx = outer_ctx;
    body_ctx.AddConstr(loop->loop_var,
                       Range::FromMinExtent(loop->min, loop->extent));
    body_ctx.AddConstr(loop->extent > 0);
    return multiply_cost(EstimateStmtCost(loop->body, body_ctx), trip_count);
  }
  if (const auto *while_loop = stmt.as<WhileNode>()) {
    ConstrSet body_ctx = outer_ctx;
    body_ctx.AddConstr(while_loop->condition);
    return multiply_cost(EstimateStmtCost(while_loop->body, body_ctx),
                         kDynamicSerialTripCountFallback);
  }
  if (const auto *branch = stmt.as<IfThenElseNode>()) {
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
  static TaskCostMap Collect(const Stmt &body, const ConstrSet &outer_ctx) {
    TaskCostCollector collector(outer_ctx);
    collector(body);
    return std::move(collector.costs_);
  }

private:
  explicit TaskCostCollector(const ConstrSet &outer_ctx) {
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

  void EstimateMarkedTask(const AttrStmtNode *op) {
    TaskMetadata metadata = ParseTaskMetadata(op->node);
    auto task = NewTask(op->body);
    ApplyTaskResourceUsage(AnalyzeTaskResourceUsage(op->body), task.get());
    ApplyTaskAccesses(AnalyzeTaskAccesses(op->body), task.get());
    estimator_.EstimateComposedTask(task.get());
    Stmt marker = GetRef<Stmt>(op);
    RecordCost(marker, task, metadata);
    if (per_core_task_depth_ > 0 && HasTaskBufferAccess(task.get()))
      per_core_task_candidates_.push_back({marker, task});
  }

  void EstimatePerCoreTask(const AttrStmtNode *op) {
    ICHECK_EQ(per_core_task_depth_, 0)
        << "Nested T.PerCoreTask regions are not supported";
    TaskMetadata metadata = ParseTaskMetadata(op->node);

    std::vector<PerCoreTaskCandidate> saved_candidates;
    saved_candidates.swap(per_core_task_candidates_);
    ++per_core_task_depth_;
    VisitStmt(op->body);
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

  void VisitStmt_(const ForNode *op) final {
    if (!InScheduleBody()) {
      ConstrVisitor::VisitStmt_(op);
      return;
    }
    if (op->kind == ForKind::kSerial || op->kind == ForKind::kUnrolled) {
      ConstrVisitor::VisitStmt_(op);
      return;
    }
    FatalUnmarkedTask("EstimateLatency", "parallel loop",
                      "MaterializeScheduleUnits");
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (!InScheduleBody()) {
      ConstrVisitor::VisitStmt_(op);
      return;
    }
    FatalUnmarkedTask("EstimateLatency", "Evaluate",
                      "MaterializeScheduleUnits");
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (!InScheduleBody()) {
      ConstrVisitor::VisitStmt_(op);
      return;
    }
    FatalUnmarkedTask("EstimateLatency", "BufferStore",
                      "MaterializeScheduleUnits");
  }

  void VisitStmt_(const BindNode *op) final {
    if (!InScheduleBody()) {
      ConstrVisitor::VisitStmt_(op);
      return;
    }
    FatalUnmarkedTask("EstimateLatency", "Bind", "MaterializeScheduleUnits");
  }

  void VisitStmt_(const WhileNode *op) final {
    if (!InScheduleBody()) {
      ConstrVisitor::VisitStmt_(op);
      return;
    }
    FatalUnmarkedTask("EstimateLatency", "While", "MaterializeScheduleUnits");
  }

  void VisitStmt_(const SBlockNode *op) final {
    FatalUnmarkedTask("EstimateLatency", "block", "MaterializeScheduleUnits");
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (!InScheduleBody()) {
      ConstrVisitor::VisitStmt_(op);
      return;
    }
    if (op->attr_key == tl::attr::kAscendTask) {
      EstimateMarkedTask(op);
      return;
    }
    if (op->attr_key == tl::attr::kAscendPerCoreTask) {
      EstimatePerCoreTask(op);
      return;
    }
    ConstrVisitor::VisitStmt_(op);
  }

  TaskCostEstimator estimator_;
  TaskCostMap costs_;
  int schedule_body_depth_{1};
  int per_core_task_depth_{0};
  std::vector<PerCoreTaskCandidate> per_core_task_candidates_;
};

class TaskAnnotationRewriter : public StmtMutator {
public:
  explicit TaskAnnotationRewriter(const TaskCostMap &costs) : costs_(costs) {}

  Stmt Rewrite(const Stmt &stmt) { return VisitStmt(stmt); }

private:
  Stmt VisitStmt(const Stmt &stmt) final {
    auto it = costs_.find(stmt);
    if (it == costs_.end())
      return StmtMutator::VisitStmt(stmt);

    const auto *attr = stmt.as<AttrStmtNode>();
    ICHECK(attr && (attr->attr_key == tl::attr::kAscendTask ||
                    attr->attr_key == tl::attr::kAscendPerCoreTask));
    Stmt body = attr->body;
    if (attr->attr_key == tl::attr::kAscendPerCoreTask)
      body = VisitStmt(body);
    TaskMetadata metadata;
    metadata.latency = it->second.latency;
    metadata.ii = it->second.ii;
    return AttrStmt(MergeTaskMetadata(metadata, attr->node), attr->attr_key,
                    attr->value, std::move(body), attr->span);
  }

  const TaskCostMap &costs_;
};

Stmt RunEstimateLatency(const Stmt &body, const ConstrSet &outer_ctx) {
  TaskCostMap costs = TaskCostCollector::Collect(body, outer_ctx);
  ICHECK(!costs.empty()) << "EstimateLatency found no schedulable tasks inside "
                            "tilelang_root";
  TaskAnnotationRewriter rewriter(costs);
  return rewriter.Rewrite(body);
}

} // namespace

tvm::transform::Pass EstimateLatency() {
  auto pass_func = [](PrimFunc func, const IRModule &, const PassContext &) {
    return RewriteTilelangKernels(
        std::move(func), "EstimateLatency",
        [](const TilelangKernelContext &context) {
          SBlock root = context.root;
          root.CopyOnWrite()->body =
              RunEstimateLatency(root->body, context.outer_ctx);
          return root;
        },
        /*require_kernel=*/false);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.EstimateLatency", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.EstimateLatency", EstimateLatency);
}

} // namespace tl
} // namespace tvm
