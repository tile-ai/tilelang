#ifndef TILELANG_ASCEND_TRANSFORM_TASK_RESOURCES_H_
#define TILELANG_ASCEND_TRANSFORM_TASK_RESOURCES_H_

#include "ascend/transform/ascend_pipe.h"
#include "ascend/transform/native_visitor.h"
#include "op/operator.h"

namespace tvm::tl::ascend {

// HBM bandwidth-contention resource bits. These are tracked separately from
// ResourcePipe because a load and a store use different issue pipes but can
// still contend on the same HBM port.
enum class HbmPort : uint16_t {
  kNone = 0,
  kAiv = 1 << 0, // AIV HBM port: GM<->UB
  kAic = 1 << 1, // AIC HBM port: GM->L1 / L0C->GM
  // UB->GM and L0C->GM share the HBM store path even though their commands
  // issue from different cores and hardware pipes.
  kStore = 1 << 2,
};

constexpr uint16_t kHbmCoreAffinityMask =
    static_cast<uint16_t>(HbmPort::kAiv) | static_cast<uint16_t>(HbmPort::kAic);

// Special scalar "registers" written by config/control intrinsics and read by
// their consumers. Pipe-order registers model a scalar synchronization call
// that fences the hardware pipe named by its argument without making the call
// itself consume that pipe.
enum class SpecialRegister : uint16_t {
  kNone = 0,
  kHf32Mode = 1 << 0,    // set_hf32_mode      <- read by Cube (MAD) ops
  kPadValue = 1 << 1,    // set_copy_pad_value <- read by padded MTE2 copies
  kLoopControl = 1 << 2, // loop_break         <- read by every sibling (fence)
  kAtomicMode = 1 << 3,  // set_atomic         <- read by GM-store copies
  kPipeMTE1 = 1 << 4,
  kPipeMTE2 = 1 << 5,
  kPipeMTE3 = 1 << 6,
  kPipeCube = 1 << 7,
  kPipeVector = 1 << 8,
  kPipeFixpipe = 1 << 9,
  kPipeScalar = 1 << 10,
  kMmadDirection = 1 << 11, // set_mmad_direction <- read by Cube (MAD) ops
};

inline uint16_t PipeSpecialRegisterMask(uint16_t pipe_mask) {
  uint16_t mask = 0;
  auto add_pipe = [&](ResourcePipe pipe, SpecialRegister reg) {
    if (pipe_mask & static_cast<uint16_t>(pipe))
      mask |= static_cast<uint16_t>(reg);
  };
  add_pipe(ResourcePipe::kMTE1, SpecialRegister::kPipeMTE1);
  add_pipe(ResourcePipe::kMTE2, SpecialRegister::kPipeMTE2);
  add_pipe(ResourcePipe::kMTE3, SpecialRegister::kPipeMTE3);
  add_pipe(ResourcePipe::kCube, SpecialRegister::kPipeCube);
  add_pipe(ResourcePipe::kVector, SpecialRegister::kPipeVector);
  add_pipe(ResourcePipe::kFixpipe, SpecialRegister::kPipeFixpipe);
  add_pipe(ResourcePipe::kScalar, SpecialRegister::kPipeScalar);
  return mask;
}

struct TaskResourceUsage {
  uint16_t pipe_mask{0};
  uint16_t hbm_mask{0};
  bool reads_pad_value{false};
  bool reads_atomic{false};
};

namespace task_analysis_detail {

inline bool IsCrossCoreSyncCall(const CallNode *op) {
  return op->op.same_as(tl::ascend_cross_core_set_flag()) ||
         op->op.same_as(tl::ascend_cross_core_wait_flag());
}

inline constexpr uint16_t kSpecializedPipes =
    static_cast<uint16_t>(ResourcePipe::kMTE1) |
    static_cast<uint16_t>(ResourcePipe::kMTE2) |
    static_cast<uint16_t>(ResourcePipe::kMTE3) |
    static_cast<uint16_t>(ResourcePipe::kCube) |
    static_cast<uint16_t>(ResourcePipe::kVector) |
    static_cast<uint16_t>(ResourcePipe::kFixpipe);

inline uint16_t ResolvePipe(uint16_t mask) {
  if ((mask & kSpecializedPipes) == 0)
    return mask | static_cast<uint16_t>(ResourcePipe::kScalar);
  return mask;
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
  const auto *pipe = pipe_arg.as<StringImmNode>();
  ICHECK(pipe != nullptr)
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
  case ResourcePipe::kNone:
  case ResourcePipe::kAll:
    break;
  }
  LOG(FATAL) << "Unsupported cross-core synchronization pipe " << pipe->value;
  return SpecialRegister::kNone;
}

class SpecialRegisterWriteAnalyzer : public AnalysisVisitor {
public:
  static uint16_t Analyze(const Stmt &stmt) {
    auto analyzer = ffi::make_object<SpecialRegisterWriteAnalyzer>();
    analyzer->Visit(stmt);
    return analyzer->mask_;
  }

private:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::loop_break())) {
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
      ICHECK_EQ(op->args.size(), 3)
          << "ascend_cross_core_set_flag/wait_flag expects 3 arguments";
      Add(SpecialRegisterForCrossCorePipe(op->args[1]));
    }
    AnalysisVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BreakNode *) final {
    Add(SpecialRegister::kLoopControl);
  }
  void VisitStmt_(const ContinueNode *) final {
    Add(SpecialRegister::kLoopControl);
  }
  void VisitStmt_(const ReturnNode *) final {
    Add(SpecialRegister::kLoopControl);
  }

  void Add(SpecialRegister reg) { mask_ |= static_cast<uint16_t>(reg); }

  uint16_t mask_{0};
};

class ResourceAnalyzer : public AnalysisVisitor {
public:
  static TaskResourceUsage Analyze(const Stmt &stmt) {
    auto analyzer = ffi::make_object<ResourceAnalyzer>();
    analyzer->Visit(stmt);
    analyzer->usage_.pipe_mask = ResolvePipe(analyzer->usage_.pipe_mask);
    return analyzer->usage_;
  }

private:
  void VisitExpr_(const CallNode *op) final {
    usage_.pipe_mask |= GetAscendTaskPipeMask(ffi::GetRef<Call>(op));
    if (IsAscendCopyCall(op)) {
      AscendCopy copy_obj(op->args, AscendCallAnnotations(op));
      const AscendCopyNode *copy = copy_obj.get();
      usage_.hbm_mask |= GetHbmResourceMask(copy->src, copy->dst);
      if (copy->data_select != 0)
        usage_.reads_pad_value = true;
      if (IsGlobalBuffer(copy->dst))
        usage_.reads_atomic = true;
    }

    AnalysisVisitor::VisitExpr_(op);
  }

  // Task-level blocks are atomic to all task analyses.
  void VisitStmt_(const RegionStmtNode *op) final {
    auto name = op->attrs.GetAttr<ffi::String>("name_hint").value_or("");
    auto pipe = GetAscendBlockPipe(name);
    if (pipe != ResourcePipe::kUnknown) {
      usage_.pipe_mask |= PipeMask(pipe);
      return;
    }
    AnalysisVisitor::VisitStmt_(op);
  }

  TaskResourceUsage usage_;
};

} // namespace task_analysis_detail

inline TaskResourceUsage AnalyzeTaskResourceUsage(const Stmt &statement) {
  return task_analysis_detail::ResourceAnalyzer::Analyze(statement);
}

inline uint16_t AnalyzeSpecialRegisterWrites(const Stmt &statement) {
  return task_analysis_detail::SpecialRegisterWriteAnalyzer::Analyze(statement);
}

inline bool HasTaskBufferAccess(const Stmt &statement) {
  class Detector : public AnalysisVisitor {
  public:
    bool found{false};

  private:
    void VisitExpr_(const TensorLoadNode *node) final {
      found = true;
      AnalysisVisitor::VisitExpr_(node);
    }
    void VisitStmt_(const TensorStoreNode *node) final {
      found = true;
      AnalysisVisitor::VisitStmt_(node);
    }
    void VisitExpr_(const CallNode *node) final {
      auto operation = ParseOperator(ffi::GetRef<Call>(node));
      if (operation.defined()) {
        auto accesses = operation->GetAccessRegions();
        found |= !accesses.reads.empty() || !accesses.writes.empty();
      }
      AnalysisVisitor::VisitExpr_(node);
    }
  };
  auto detector = ffi::make_object<Detector>();
  detector->Visit(statement);
  return detector->found;
}

[[noreturn]] inline void FatalUnmarkedTask(const char *consumer,
                                           const char *node_kind,
                                           const char *prerequisite = nullptr) {
  if (prerequisite)
    TVM_FFI_THROW(ValueError)
        << consumer << " found an unmarked " << node_kind
        << ". Run tl.transform." << prerequisite << " first";
  TVM_FFI_THROW(ValueError) << consumer << " found an unmarked " << node_kind;
  TVM_FFI_UNREACHABLE();
}

} // namespace tvm::tl::ascend

#endif
