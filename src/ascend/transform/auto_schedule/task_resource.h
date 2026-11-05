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
 * \file task_resource.h
 * \brief Hardware resources of native Ascend tasks, before access-footprint
 * analysis.
 */

#pragma once

#include "../ascend_pipe.h"
#include "native_task_ir.h"

#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl::ascend {

enum class HbmPort : uint16_t {
  kNone = 0,
  kAiv = 1 << 0,
  kAic = 1 << 1,
  kStore = 1 << 2,
};
inline constexpr uint16_t kHbmCoreAffinityMask =
    static_cast<uint16_t>(HbmPort::kAiv) | static_cast<uint16_t>(HbmPort::kAic);

enum class SpecialRegister : uint16_t {
  kNone = 0,
  kHf32Mode = 1 << 0,
  kPadValue = 1 << 1,
  kLoopControl = 1 << 2,
  kAtomicMode = 1 << 3,
  kPipeMTE1 = 1 << 4,
  kPipeMTE2 = 1 << 5,
  kPipeMTE3 = 1 << 6,
  kPipeCube = 1 << 7,
  kPipeVector = 1 << 8,
  kPipeFixpipe = 1 << 9,
  kPipeScalar = 1 << 10,
  kMmadDirection = 1 << 11,
};

struct TaskResourceUsage {
  uint16_t pipe_mask{0};
  uint16_t hbm_mask{0};
  bool reads_pad_value{false};
  bool reads_atomic{false};
};

namespace task_resource_detail {

inline bool IsCrossCoreSyncCall(const CallObj *op) {
  return op->op.same_as(tl::ascend_cross_core_set_flag()) ||
         op->op.same_as(tl::ascend_cross_core_wait_flag());
}

inline uint16_t GetHbmResourceMask(const TensorVar &src, const TensorVar &dst) {
  if (IsGlobalBuffer(src) && IsSharedBuffer(dst))
    return static_cast<uint16_t>(HbmPort::kAiv);
  if (IsSharedBuffer(src) && IsGlobalBuffer(dst))
    return static_cast<uint16_t>(HbmPort::kAiv) |
           static_cast<uint16_t>(HbmPort::kStore);
  if (IsGlobalBuffer(src) && IsL1Buffer(dst))
    return static_cast<uint16_t>(HbmPort::kAic);
  if (IsL0CBuffer(src) && IsGlobalBuffer(dst))
    return static_cast<uint16_t>(HbmPort::kAic) |
           static_cast<uint16_t>(HbmPort::kStore);
  return 0;
}

inline SpecialRegister SpecialRegisterForCrossCorePipe(const Expr &pipe_arg) {
  const auto *pipe = pipe_arg.as<StringImmObj>();
  TVM_FFI_CHECK(pipe != nullptr, ValueError)
      << "Cross-core synchronization pipe must be a constant string, got "
      << pipe_arg;
  switch (ParseAscendPipe(pipe->value)) {
  case ResourcePipe::kScalar:
    return SpecialRegister::kPipeScalar;
  case ResourcePipe::kVector:
    return SpecialRegister::kPipeVector;
  case ResourcePipe::kCube:
    return SpecialRegister::kPipeCube;
  case ResourcePipe::kMTE1:
    return SpecialRegister::kPipeMTE1;
  case ResourcePipe::kMTE2:
    return SpecialRegister::kPipeMTE2;
  case ResourcePipe::kMTE3:
    return SpecialRegister::kPipeMTE3;
  case ResourcePipe::kFixpipe:
    return SpecialRegister::kPipeFixpipe;
  default:
    LOG(FATAL) << "Unsupported cross-core synchronization pipe " << pipe->value;
  }
  return SpecialRegister::kNone;
}

class SpecialRegisterWriteAnalyzer : public tirx::StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;
  uint16_t mask{0};

  ffi::Optional<ffi::VisitInterrupt> Visit_(const BreakObj *) final {
    if (loop_depth_ == 0)
      Add(SpecialRegister::kLoopControl);
    return std::nullopt;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const ForObj *op) final {
    ++loop_depth_;
    auto result = StmtExprVisitor::Visit_(op);
    --loop_depth_;
    return result;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const WhileObj *op) final {
    ++loop_depth_;
    auto result = StmtExprVisitor::Visit_(op);
    --loop_depth_;
    return result;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const CallObj *op) final {
    if (op->op.same_as(tl::loop_break())) {
      if (loop_depth_ == 0)
        Add(SpecialRegister::kLoopControl);
    } else if (op->op.same_as(tl::ascend_set_hf32_mode())) {
      Add(SpecialRegister::kHf32Mode);
    } else if (op->op.same_as(tl::ascend_set_mmad_direction())) {
      Add(SpecialRegister::kMmadDirection);
    } else if (op->op.same_as(tl::ascend_set_copy_pad_value())) {
      Add(SpecialRegister::kPadValue);
    } else if (op->op.same_as(tl::ascend_set_atomic()) ||
               op->op.same_as(tl::ascend_set_atomic_none())) {
      Add(SpecialRegister::kAtomicMode);
    } else if (IsCrossCoreSyncCall(op)) {
      TVM_FFI_CHECK_EQ(op->args.size(), 3U, ValueError);
      Add(SpecialRegisterForCrossCorePipe(op->args[1]));
    }
    if (IsScheduledBind(op))
      return Visit(ReadScheduledBinding(op).value);
    return StmtExprVisitor::Visit_(op);
  }

private:
  int loop_depth_{0};
  void Add(SpecialRegister value) { mask |= static_cast<uint16_t>(value); }
};

class ResourceAnalyzer : public tirx::StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;
  TaskResourceUsage usage;

  ffi::Optional<ffi::VisitInterrupt> Visit_(const CallObj *op) final {
    if (IsScheduledBind(op))
      return Visit(ReadScheduledBinding(op).value);
    usage.pipe_mask |= GetAscendTaskPipeMask(ffi::GetRef<Call>(op));
    if (IsAscendCopyCall(op)) {
      auto decoded = DecodeTileOpArguments(op);
      AscendCopy copy(decoded.operands, decoded.annotations);
      usage.hbm_mask |= GetHbmResourceMask(copy->src, copy->dst);
      usage.reads_pad_value |= copy->data_select != 0;
      usage.reads_atomic |= IsGlobalBuffer(copy->dst);
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const RegionStmtObj *op) final {
    if (auto name = NativeBlockName(op)) {
      // A VF body is one task on its own hardware pipe. Its scalar internals
      // must not turn it into a multi-pipe task during boundary validation.
      usage.pipe_mask |= PipeMask(GetAscendBlockPipe(name.value()));
      return std::nullopt;
    }
    return StmtExprVisitor::Visit_(op);
  }
};

// Boundary validation only asks whether a real access exists. The complete
// may/must footprint and alias analysis remains the scheduler's responsibility.
// TensorRegion operands are addresses, so consult each registered operator's
// access contract rather than treating every metadata tensor as a memory read.
class BufferAccessFinder : public tirx::StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;
  bool found{false};

  ffi::Optional<ffi::VisitInterrupt> Visit_(const TensorLoadObj *) final {
    found = true;
    return ffi::VisitInterrupt();
  }
  ffi::Optional<ffi::VisitInterrupt> Visit_(const TensorStoreObj *) final {
    found = true;
    return ffi::VisitInterrupt();
  }
  ffi::Optional<ffi::VisitInterrupt> Visit_(const CallObj *op) final {
    if (IsScheduledBind(op))
      return Visit(ReadScheduledBinding(op).value);
    if (auto tile = ParseOperator(ffi::GetRef<Call>(op)); tile.defined()) {
      auto regions = tile->GetAccessRegions();
      if (!regions.reads.empty() || !regions.writes.empty()) {
        found = true;
        return ffi::VisitInterrupt();
      }
    }
    return StmtExprVisitor::Visit_(op);
  }
};

} // namespace task_resource_detail

inline TaskResourceUsage AnalyzeTaskResourceUsage(const Stmt &stmt) {
  auto analyzer = ffi::make_object<task_resource_detail::ResourceAnalyzer>();
  analyzer->Visit(stmt);
  auto usage = analyzer->usage;
  constexpr uint16_t specialized =
      PipeMask(ResourcePipe::kMTE1) | PipeMask(ResourcePipe::kMTE2) |
      PipeMask(ResourcePipe::kMTE3) | PipeMask(ResourcePipe::kCube) |
      PipeMask(ResourcePipe::kVector) | PipeMask(ResourcePipe::kFixpipe);
  if ((usage.pipe_mask & specialized) == 0)
    usage.pipe_mask |= PipeMask(ResourcePipe::kScalar);
  return usage;
}

inline uint16_t AnalyzeSpecialRegisterWrites(const Stmt &stmt) {
  auto analyzer =
      ffi::make_object<task_resource_detail::SpecialRegisterWriteAnalyzer>();
  analyzer->Visit(stmt);
  return analyzer->mask;
}

inline bool HasTaskBufferAccess(const Stmt &stmt) {
  auto finder = ffi::make_object<task_resource_detail::BufferAccessFinder>();
  finder->Visit(stmt);
  return finder->found;
}

[[noreturn]] inline void FatalUnmarkedTask(const char *consumer,
                                           const std::string &node_kind) {
  LOG(FATAL) << consumer << " found an unmarked " << node_kind
             << " after internal task normalization";
  throw;
}

} // namespace tvm::tl::ascend
