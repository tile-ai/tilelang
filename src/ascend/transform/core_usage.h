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

#pragma once

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>

#include "ascend_pipe.h"
#include "core_mask.h"

namespace tvm {
namespace tl {

// Execution ownership and launch resources are different: on 3510 an AIC
// issues L0C->UB, but the receiving AIV still requires a mixed launch.
struct AscendCoreUsage {
  CoreMask issuers{kCoreUnassigned};
  // Union of definite execution demands, including ordinary scalar accesses.
  CoreMask execution_cores{kCoreUnassigned};
  CoreMask scalar_candidates{kCoreAll};
  bool requires_mix{false};
  bool uses_ss{false};
  bool unknown_effect{false};
  // Executable work outside explicit CUBE/VECTOR regions.
  bool unplaced_work{false};

  // Structural facts are populated after operation analysis, not merged per
  // call.
  CoreMask explicit_cores{kCoreUnassigned};
  ffi::Optional<Integer> vector_count;

  void Merge(const AscendCoreUsage &other) {
    issuers |= other.issuers;
    execution_cores |= other.execution_cores;
    scalar_candidates &= other.scalar_candidates;
    requires_mix |= other.requires_mix;
    uses_ss |= other.uses_ss;
    unknown_effect |= other.unknown_effect;
    unplaced_work |= other.unplaced_work;
  }
};

inline CoreMask AscendPipeIssuers(uint16_t pipes) {
  CoreMask mask = kCoreUnassigned;
  if (pipes & (PipeMask(ResourcePipe::kCube) | PipeMask(ResourcePipe::kMTE1) |
               PipeMask(ResourcePipe::kFixpipe)))
    mask |= kCoreCube;
  if (pipes & (PipeMask(ResourcePipe::kVector) | PipeMask(ResourcePipe::kMTE3)))
    mask |= kCoreVector;
  // MTE2 and S are shared; their operation/memory determines the issuer.
  return mask;
}

inline CoreMask AscendMemoryCore(const Buffer &buffer) {
  if (IsSharedBuffer(buffer))
    return kCoreVector;
  if (IsL1Buffer(buffer) || IsL0ABuffer(buffer) || IsL0BBuffer(buffer) ||
      IsL0CBuffer(buffer) || IsL0SFBuffer(buffer))
    return kCoreCube;
  return kCoreAll;
}

inline bool IsAscendAccessMetadata(const CallNode *op) {
  return op->op.same_as(tl::access_ptr()) ||
         op->op.same_as(builtin::tvm_access_ptr()) ||
         op->op.same_as(tl::region());
}

inline bool IsAscendCoreSync(const CallNode *op) {
  return op->op.same_as(ascend_set_flag()) ||
         op->op.same_as(ascend_wait_flag()) ||
         op->op.same_as(ascend_pipe_barrier()) ||
         op->op.same_as(ascend_cross_core_set_flag()) ||
         op->op.same_as(ascend_cross_core_wait_flag()) ||
         op->op.same_as(ascend_get_buf()) || op->op.same_as(ascend_rls_buf()) ||
         op->op.same_as(builtin::tvm_storage_sync());
}

// Used only by manual kernel normalization. DMA operands describe
// address spaces, not ordinary scalar loads from the opposite execution core.
class AscendCoreUsageAnalyzer : public StmtExprVisitor {
public:
  static AscendCoreUsage Analyze(const SBlock &root) {
    AscendCoreUsageAnalyzer visitor;
    if (auto value = root->annotations.Get("vector_count"))
      visitor.vector_count_ = value.value().cast<Integer>();
    visitor(root->body);
    visitor.usage_.execution_cores |= visitor.usage_.issuers;
    visitor.usage_.explicit_cores = visitor.explicit_cores_;
    visitor.usage_.vector_count = visitor.vector_count_;
    return visitor.usage_;
  }

private:
  void Access(const Buffer &buffer) {
    CoreMask core = AscendMemoryCore(buffer);
    ICHECK((core & placement_))
        << "Scalar access conflicts with explicit "
        << (placement_ == kCoreCube ? "T.Cube" : "T.Vector")
        << " placement: " << buffer
        << ". Move UB scalar accesses into T.Vector(G) and L1/L0 scalar "
           "accesses into T.Cube(); transfer cross-core data using supported "
           "copies and the required synchronization.";
    usage_.unplaced_work |= placement_ == kCoreAll;
    usage_.scalar_candidates &= core;
    // GM/local storage permits either side; it does not require both sides.
    if (core != kCoreAll)
      usage_.execution_cores |= core;
    usage_.uses_ss |= IsSSBuffer(buffer);
    usage_.requires_mix |= IsSSBuffer(buffer);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    Access(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    Access(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint == "CUBE" || op->name_hint == "VECTOR") {
      explicit_cores_ |= op->name_hint == "CUBE" ? kCoreCube : kCoreVector;
      if (auto value = op->annotations.Get("vector_count")) {
        Integer requested = value.value().cast<Integer>();
        ICHECK(!vector_count_.has_value() ||
               vector_count_.value()->value == requested->value)
            << "Conflicting AIV counts between MixedKernel/Vector regions: "
            << vector_count_.value_or(requested) << " and " << requested
            << ". Use the same count for MixedKernel(sids=G) and all "
               "T.Vector(G) regions.";
        vector_count_ = requested;
      }
    }
    bool vf = op->name_hint == "SIMT_VF" || op->name_hint == "SIMD_VF";
    CoreMask previous = placement_;
    if ((vf || op->name_hint == "CUBE" || op->name_hint == "VECTOR")) {
      CoreMask core = op->name_hint == "CUBE" ? kCoreCube : kCoreVector;
      ICHECK(previous == kCoreAll || previous == core)
          << "T.Cube and T.Vector cannot be nested across execution cores; "
             "write them as sibling regions under the kernel, and keep each "
             "side's operations and synchronization in its own region.";
      placement_ = core;
    }
    // VF remains an atomic AIV program for resource analysis. Validation may
    // inspect its body, but must not replace that resource contract.
    AscendCoreUsage before_vf = usage_;
    {
      if (op->init.defined())
        VisitStmt(op->init.value());
      VisitStmt(op->body);
    }
    if (vf) {
      usage_ = before_vf;
      usage_.issuers |= kCoreVector;
      usage_.unplaced_work |= previous == kCoreAll;
    }
    placement_ = previous;
  }

  // The BufferLoad carrying an address is metadata, but its indices may
  // contain real scalar reads. Do not discard those expressions as metadata.
  void VisitAddress(const PrimExpr &expr) {
    if (const auto *load = expr.as<BufferLoadNode>())
      StmtExprVisitor::VisitExpr_(load);
    else
      VisitExpr(expr);
  }

  void VisitExpr_(const CallNode *op) final {
    // Analyze each call subtree once. Keep its local requirements separate
    // from preceding operations, validate them, then merge into the parent.
    AscendCoreUsage enclosing = usage_;
    usage_ = {};
    AnalyzeCall(op);
    ICHECK(usage_.scalar_candidates &&
           (usage_.issuers & usage_.scalar_candidates) == usage_.issuers)
        << "Operation has incompatible execution and scalar memory "
           "requirements: "
        << ffi::GetRef<Call>(op)
        << ". Do not directly read another core's private buffer in an "
           "operation or its indices. Use a supported data-transfer path "
           "with the required synchronization, then read on the owning core.";
    ICHECK(!usage_.unknown_effect || placement_ != kCoreAll)
        << "Cannot infer execution core of an unknown side-effecting "
           "operation; "
           "place it explicitly in T.Cube/T.Vector (pointer arguments do not "
           "prove the implementation's execution core): "
        << ffi::GetRef<Call>(op);
    ICHECK(placement_ == kCoreAll ||
           ((usage_.issuers & placement_) == usage_.issuers &&
            (usage_.scalar_candidates & placement_) != 0))
        << "Operation conflicts with explicit "
        << (placement_ == kCoreCube ? "T.Cube" : "T.Vector")
        << " placement: " << ffi::GetRef<Call>(op)
        << ". Move the operation to its issuing core's T.Cube()/T.Vector(G) "
           "region; cross-core copies belong to the issuing side, not the "
           "destination side. Keep required synchronization on each side.";
    if (placement_ == kCoreAll && !IsAscendAccessMetadata(op) &&
        (usage_.issuers || IsAscendCoreSync(op) ||
         SideEffect(ffi::GetRef<Call>(op)) > CallEffectKind::kPure))
      usage_.unplaced_work = true;
    usage_.Merge(enclosing);
  }

  void AnalyzeCall(const CallNode *op) {
    if (IsAscendAccessMetadata(op)) {
      for (size_t i = 0; i < op->args.size(); ++i) {
        if (i == 0 && !op->op.same_as(builtin::tvm_access_ptr()))
          VisitAddress(op->args[i]);
        else
          VisitExpr(op->args[i]);
      }
      return;
    }
    Call call = ffi::GetRef<Call>(op);
    if (IsAscendCopyCall(op)) {
      Copy copy(op->args, op->annotations);
      ICHECK((!(IsL1Buffer(copy->src) && IsSharedBuffer(copy->dst)) &&
              !(IsL0CBuffer(copy->src) && IsL1Buffer(copy->dst))))
          << "Ascend 3510 copy " << copy->src.scope() << " -> "
          << copy->dst.scope()
          << " has no registered DMA lowering in TileLang; element-wise "
             "fallback cannot implement this hardware transfer. Restructure "
             "the "
             "data flow using supported copy routes (for example, load the "
             "original GM data separately into L1 and UB), or add the missing "
             "DMA lowering. Changing core placement or enabling automatic "
             "scheduling does not add a missing lowering.";
      uint16_t pipe = GetAscendCopyPipeMask(copy->src, copy->dst);
      usage_.issuers |= pipe == PipeMask(ResourcePipe::kMTE2)
                            ? AscendMemoryCore(copy->dst)
                            : AscendPipeIssuers(pipe);
      usage_.requires_mix |=
          (IsSharedBuffer(copy->src) && IsL1Buffer(copy->dst)) ||
          (IsL0CBuffer(copy->src) && IsSharedBuffer(copy->dst));
      for (size_t i = 0; i < op->args.size(); ++i) {
        if (i < 2)
          VisitAddress(op->args[i]);
        else
          VisitExpr(op->args[i]);
      }
      return;
    }
    if (op->op.same_as(Op::Get("tl.tileop.fill"))) {
      Fill fill(op->args, op->annotations);
      ICHECK(IsL1Buffer(fill->dst) || IsSharedBuffer(fill->dst) ||
             IsGlobalBuffer(fill->dst) || IsLocalBuffer(fill->dst) ||
             IsLocalVarBuffer(fill->dst) || IsFragmentBuffer(fill->dst))
          << "Ascend fill has no lowering for storage scope "
          << fill->dst.scope()
          << ". Initialize data in a supported scope such as L1 or UB, then "
             "use a supported copy into the destination. For L0A/L0B, fill "
             "L1 first and copy from L1. Enabling automatic scheduling does "
             "not add a missing fill lowering.";
      Access(fill->dst);
      if (IsL1Buffer(fill->dst))
        usage_.issuers |= kCoreCube;
      VisitExpr(fill->value);
      for (const auto &range : fill->region) {
        VisitExpr(range->min);
        VisitExpr(range->extent);
      }
      return;
    }
    // Check only the local endpoint's placement, not the synchronization
    // protocol. Shared/full pipes do not constrain placement, and
    // synchronization must not establish an additional compute stream.
    if (IsAscendCoreSync(op)) {
      uint16_t pipes = GetAscendCallPipeMask(call);
      CoreMask endpoint = pipes == kAllResourcePipes ? kCoreUnassigned
                                                     : AscendPipeIssuers(pipes);
      ICHECK(placement_ == kCoreAll || (endpoint & placement_) == endpoint)
          << "Synchronization pipe conflicts with explicit "
          << (placement_ == kCoreCube ? "T.Cube" : "T.Vector")
          << " placement: " << call
          << ". Move this synchronization endpoint into the "
             "T.Cube()/T.Vector(G) "
             "region that owns its pipe; keep each cross-core endpoint on its "
             "respective core.";
      StmtExprVisitor::VisitExpr_(op);
      return;
    }
    uint16_t pipes = GetAscendCallPipeMask(call);
    usage_.issuers |= AscendPipeIssuers(pipes);
    if (pipes & PipeMask(ResourcePipe::kMTE2)) {
      if (op->op.same_as(ascend_copy_gm_to_cbuf()) ||
          op->op.same_as(ascend_fill_l1()))
        usage_.issuers |= kCoreCube;
      else if (op->op.same_as(ascend_copy_gm_to_ubuf()))
        usage_.issuers |= kCoreVector;
      else
        usage_.unknown_effect = true;
    }
    usage_.requires_mix |= op->op.same_as(ascend_copy_matrix_cc_to_ub()) ||
                           op->op.same_as(ascend_copy_ubuf_to_cbuf()) ||
                           op->op.same_as(ascend_nd2nz_post_copy());
    if (pipes && pipes != PipeMask(ResourcePipe::kScalar)) {
      StmtExprVisitor::VisitExpr_(op);
      return;
    }
    if (op->op.same_as(ascend_set_hf32_mode()) ||
        op->op.same_as(ascend_set_mmad_direction()))
      usage_.issuers |= kCoreCube;
    // Pointer arguments do not describe an external implementation's core
    // requirements. Unknown non-pure calls require explicit core placement.
    if (!pipes && SideEffect(call) > CallEffectKind::kPure)
      usage_.unknown_effect = true;
    StmtExprVisitor::VisitExpr_(op);
  }

  // Keep structural facts outside temporary per-call and VF resource snapshots.
  CoreMask explicit_cores_{kCoreUnassigned};
  ffi::Optional<Integer> vector_count_;
  AscendCoreUsage usage_;
  CoreMask placement_{kCoreAll};
};

} // namespace tl
} // namespace tvm
