#pragma once
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include "op/gemm.h"
#include "transform/common/constr_visitor.h"
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../ascend_pipe.h"
#include "../buffer_version.h"
#include "../core_mask.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Any;
using ffi::String;

// Forward declarations
class IRStructure;
class TaskNode;
class ControlNode;

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

// Helper function to compare if two regions are equal
inline bool RegionsEqual(const Region &a, const Region &b) {
  if (a.size() != b.size())
    return false;

  arith::Analyzer analyzer;
  for (size_t i = 0; i < a.size(); ++i) {
    // Check if min values are equal
    if (!analyzer.CanProveEqual(a[i]->min, b[i]->min)) {
      return false;
    }
    // Check if extent values are equal
    if (!analyzer.CanProveEqual(a[i]->extent, b[i]->extent)) {
      return false;
    }
  }
  return true;
}

constexpr int64_t kDynamicSerialTripCountFallback = 100;

inline int64_t GetSerialTripCount(const For &loop,
                                  arith::Analyzer *analyzer = nullptr) {
  arith::Analyzer local_analyzer;
  if (analyzer == nullptr)
    analyzer = &local_analyzer;

  auto upper_bound_or = [&](const PrimExpr &expr, int64_t fallback) {
    if (const int64_t *value = as_const_int(expr))
      return *value;

    arith::ConstIntBound bound = analyzer->const_int_bound(expr);
    arith::Analyzer unconstrained_analyzer;
    Var unconstrained("unconstrained_trip_count", expr.dtype());
    int64_t unconstrained_max =
        unconstrained_analyzer.const_int_bound(unconstrained)->max_value;
    bool has_useful_bound = bound->max_value >= 0 &&
                            bound->max_value != arith::ConstIntBound::kPosInf &&
                            bound->max_value < unconstrained_max;
    return has_useful_bound ? bound->max_value : fallback;
  };

  PrimExpr step = loop->step.has_value() ? loop->step.value()
                                         : IntImm(DataType::Int(32), 1);
  int64_t extent =
      upper_bound_or(loop->extent, kDynamicSerialTripCountFallback);
  // A trip-count upper bound needs a lower bound for a symbolic positive
  // step.  Use one when the step is not constant; using its upper bound would
  // underestimate the number of serial iterations.
  const int64_t *constant_step = as_const_int(step);
  int64_t step_value = constant_step ? *constant_step : 1;
  if (extent <= 0)
    return 0;
  if (step_value <= 0)
    return extent;
  return extent / step_value + (extent % step_value != 0);
}

// A wrapper introduced at a node relative to its nearest surviving structural
// ancestor. Guards are scheduling metadata rather than schedulable tasks.
class Guard {
public:
  enum class Kind { kCondition, kAttribute };

  virtual ~Guard() = default;
  virtual Kind GetKind() const = 0;
  virtual std::unique_ptr<Guard> Clone() const = 0;
  virtual void SubstituteVar(const Var &old_var, const Var &new_var) = 0;
  virtual Stmt Wrap(Stmt body) const = 0;

  bool IsCondition() const { return GetKind() == Kind::kCondition; }
  bool IsAttribute() const { return GetKind() == Kind::kAttribute; }
};

class ConditionGuard final : public Guard {
public:
  explicit ConditionGuard(PrimExpr condition)
      : condition(std::move(condition)) {}

  Kind GetKind() const final { return Kind::kCondition; }
  std::unique_ptr<Guard> Clone() const final {
    return std::make_unique<ConditionGuard>(condition);
  }
  void SubstituteVar(const Var &old_var, const Var &new_var) final {
    condition = Substitute(condition, {{old_var, new_var}});
  }
  Stmt Wrap(Stmt body) const final {
    return IfThenElse(condition, std::move(body));
  }

  PrimExpr condition;
};

class AttributeGuard final : public Guard {
public:
  AttributeGuard(Any node, String key, PrimExpr value, Span span)
      : node(std::move(node)), key(std::move(key)), value(std::move(value)),
        span(std::move(span)) {}

  Kind GetKind() const final { return Kind::kAttribute; }
  std::unique_ptr<Guard> Clone() const final {
    return std::make_unique<AttributeGuard>(node, key, value, span);
  }
  void SubstituteVar(const Var &old_var, const Var &new_var) final {
    if (node.as<PrimExprNode>()) {
      node = Substitute(Downcast<PrimExpr>(node), {{old_var, new_var}});
    }
    if (value.defined()) {
      value = Substitute(value, {{old_var, new_var}});
    }
  }
  Stmt Wrap(Stmt body) const final {
    return AttrStmt(node, key, value, std::move(body), span);
  }

  Any node;
  String key;
  PrimExpr value;
  Span span;
};

using GuardList = std::vector<std::unique_ptr<Guard>>;

// Optional pass-local state owned by an IRStructure node. Concrete extra-info
// types are declared by the pass that owns them; shared IR only defines their
// lifecycle and substitution hooks.
class IRExtraInfo {
public:
  virtual ~IRExtraInfo() = default;
  virtual std::unique_ptr<IRExtraInfo> Clone() const = 0;
  virtual void SubstituteVar(const Var &, const Var &) {}
};

// Structural and dependency-analysis state shared by both local IR phases.
class IRStructure {
public:
  enum class Kind { kTask, kControl };

  explicit IRStructure(std::unique_ptr<IRExtraInfo> extra_info = nullptr)
      : extra_info_(std::move(extra_info)) {}

  virtual ~IRStructure() = default;
  virtual Kind GetKind() const = 0;
  virtual std::shared_ptr<IRStructure> Clone() const = 0;

  // The parent does not own this node; ownership remains with the parent's
  // shared_ptr fields. Root nodes have no parent.
  IRStructure *GetParent() { return parent_; }
  const IRStructure *GetParent() const { return parent_; }
  void SetParent(IRStructure *parent) {
    ICHECK(parent != this) << "An IRStructure node cannot parent itself";
    parent_ = parent;
  }

  // Return whether this node is the given ancestor or lies in its subtree.
  bool IsWithin(const IRStructure *ancestor) const {
    for (const IRStructure *node = this; node != nullptr;
         node = node->GetParent()) {
      if (node == ancestor)
        return true;
    }
    return false;
  }

  // Return the lowest common ancestor with another node. Nodes in different
  // top-level trees have no explicit common ancestor and return null.
  IRStructure *GetLowestCommonAncestor(IRStructure *other);

  // Return the structural path from an ancestor to this node, inclusive. A
  // null ancestor means the root of this node's tree.
  std::vector<IRStructure *> PathFrom(IRStructure *ancestor = nullptr);

  // Return the ancestor's direct child on the path to this node. The argument
  // must be a strict ancestor of this node.
  IRStructure *GetChildOnPathFrom(IRStructure *ancestor);

  template <typename T> T *GetExtraInfo() {
    ICHECK(extra_info_ != nullptr) << "Expected pass-local IR extra info";
    return static_cast<T *>(extra_info_.get());
  }

  // Guard deltas introduced at this node relative to its nearest
  // surviving structural ancestor.
  const GuardList &GetGuards() const { return guards_; }
  PrimExpr GetConditionGuard() const;
  bool HasGuards() const { return !guards_.empty(); }
  // True iff the node carries a real *conditional* guard
  bool HasConditions() const {
    for (const auto &guard : guards_) {
      if (guard->IsCondition())
        return true;
    }
    return false;
  }
  void SetGuards(GuardList guards) { guards_ = std::move(guards); }
  // Prepend a conditional guard (the delta stack is built outermost-last, so
  // new outer guards go to the front).
  void PrependCondition(const PrimExpr &cond) {
    guards_.insert(guards_.begin(), std::make_unique<ConditionGuard>(cond));
  }
  // Prepend an AttrStmt wrapper supported by AutoSchedule.
  void PrependAttribute(const Any &node, const String &key,
                        const PrimExpr &value, const Span &span) {
    guards_.insert(guards_.begin(),
                   std::make_unique<AttributeGuard>(node, key, value, span));
  }

  // Pipeline stage assigned by the Z3 scheduler: producer = 0, consumers are
  // positive (value = latency in stages). See lower_scheduled_tir.cc.
  int GetStage() const { return stage_; }
  void SetStage(int stage) { stage_ = stage; }

  // Task cost carried by T.Task annotations or synthesized for a ControlNode.
  // These are ordinary IR properties used by scheduling and dependency
  // analysis, rather than scratch state owned by one pass.
  void SetLatency(int64_t latency) { latency_ = latency; }
  int64_t GetLatency() const { return latency_; }
  void SetII(int64_t ii) { ii_ = ii; }
  int64_t GetII() const { return ii_; }

  // Position in the scheduled sibling list. Together with stage this determines
  // the emitted software-pipeline order inside the current parent.
  size_t GetIndex() const { return index_; }
  void SetIndex(size_t index) { index_ = index; }

  // Helper methods for safe casting
  bool IsTask() const { return GetKind() == Kind::kTask; }
  bool IsControl() const { return GetKind() == Kind::kControl; }

  // Substitute a var inside this node's own guard deltas. Call from each
  // node's SubstituteVar override so core specialization reaches both If
  // conditions and propagated AttrStmt fields.
  void SubstituteVarInGuards(const Var &old_var, const Var &new_var) {
    for (auto &guard : guards_) {
      guard->SubstituteVar(old_var, new_var);
    }
  }

  void CopyBaseMetadataFrom(const IRStructure &other) {
    guards_.clear();
    guards_.reserve(other.guards_.size());
    for (const auto &guard : other.guards_) {
      guards_.push_back(guard->Clone());
    }
    stage_ = other.stage_;
    index_ = other.index_;
    latency_ = other.latency_;
    ii_ = other.ii_;
  }

  // Resource pipe bitmask (accessible by all IR nodes)
  virtual uint16_t GetPipeMask() const = 0;
  bool UsesPipe(ResourcePipe pipe) const {
    return (GetPipeMask() & static_cast<uint16_t>(pipe)) != 0;
  }
  bool UsesOnlyPipe(ResourcePipe pipe) const {
    return GetPipeMask() == static_cast<uint16_t>(pipe);
  }

  // HBM port-contention bitmask. Composite nodes OR-aggregate descendants;
  // only TaskNode stores a settable mask.
  virtual uint16_t GetHbmMask() const = 0;

  // Memory access regions (collected during analysis)
  virtual std::vector<BufferRegion> GetReadRegions() const = 0;
  virtual std::vector<BufferRegion> GetWriteRegions() const = 0;

  // Storage identity is the Buffer data Var, so aliases with different shapes
  // and strides are treated as accesses to the same physical allocation.
  bool ReadsStorage(const Var &storage) const {
    for (const BufferRegion &region : GetReadRegions()) {
      if (region->buffer->data.same_as(storage))
        return true;
    }
    return false;
  }
  bool WritesStorage(const Var &storage) const {
    for (const BufferRegion &region : GetWriteRegions()) {
      if (region->buffer->data.same_as(storage))
        return true;
    }
    return false;
  }
  bool TouchesStorage(const Var &storage) const {
    return ReadsStorage(storage) || WritesStorage(storage);
  }

  // Variable access (used for dependency analysis)
  virtual std::vector<Var> GetReadVars() const = 0;
  virtual std::vector<Var> GetWriteVars() const = 0;

  // Substitute a variable throughout this IR node
  virtual void SubstituteVar(const Var &old_var, const Var &new_var) = 0;

  // Special scalar registers this node *writes* (bitmask of SpecialRegister).
  // Only leaf config/control intrinsics write; everything else returns 0.
  virtual uint16_t GetSpecialWriteMask() const { return 0; }

  // Special scalar registers this node *reads*:
  //   Cube (MAD)     reads the HF32-mode register (derived from the Cube pipe)
  //   padded GM->UB copies read the pad-value register (only copies that carry
  //     data_select / pad_value)
  //   loop_break is a fence: every node implicitly reads the loop-control
  //     register, so any loop_break write orders against all siblings.
  uint16_t GetSpecialReadMask() const {
    uint16_t mask = static_cast<uint16_t>(SpecialRegister::kLoopControl);
    if (UsesPipe(ResourcePipe::kCube))
      mask |= static_cast<uint16_t>(SpecialRegister::kHf32Mode);
    if (ReadsPadValueRegister())
      mask |= static_cast<uint16_t>(SpecialRegister::kPadValue);
    if (ReadsAtomicRegister())
      mask |= static_cast<uint16_t>(SpecialRegister::kAtomicMode);
    mask |= PipeSpecialRegisterMask(GetPipeMask());
    return mask;
  }

  // Whether this node contains a padded GM->UB copy.
  virtual bool ReadsPadValueRegister() const { return false; }

  // Whether this node contains a copy that stores to GM.
  virtual bool ReadsAtomicRegister() const { return false; }

  bool WritesSpecialRegister(SpecialRegister reg) const {
    return (GetSpecialWriteMask() & static_cast<uint16_t>(reg)) != 0;
  }

  // Whether this node is the loop_break marker (writes the loop-control
  // register). Used by core broadcast + sequence scans.
  bool ContainsLoopBreak() const {
    return WritesSpecialRegister(SpecialRegister::kLoopControl);
  }

protected:
  std::unique_ptr<IRExtraInfo> CloneExtraInfo() const {
    return extra_info_ ? extra_info_->Clone() : nullptr;
  }
  void SubstituteVarInExtraInfo(const Var &old_var, const Var &new_var) {
    if (extra_info_)
      extra_info_->SubstituteVar(old_var, new_var);
  }

private:
  IRStructure *parent_{nullptr};
  std::unique_ptr<IRExtraInfo> extra_info_;
  GuardList guards_;
  int stage_{0};
  size_t index_{0};
  int64_t latency_{0};
  int64_t ii_{0};
};

// Task node: one atomic scheduler node. Ordinary tasks contain one statement;
// an explicit Task marker may group a structured sequence into this one
// node while scheduling and dependency analysis keep it atomic.
class TaskNode : public IRStructure {
public:
  explicit TaskNode(std::unique_ptr<IRExtraInfo> extra_info = nullptr)
      : IRStructure(std::move(extra_info)) {}

  Stmt stmt;

  // Snapshot of the surrounding-scope ConstrSet at the moment this task was
  // built or decoded. Captures ancestor For ranges, LetStmt bindings,
  // kernel-block
  // thread_extent / virtual_thread iter ranges, tilelang_assume, and
  // IfThenElse / Select branch guards. Populated by the scheduled-TIR builder
  // or decoder when the TaskNode is constructed; consumed by
  // RegionsMayConflict so the prover sees the symbol environment each region
  // was written under.
  ConstrSet outer_ctx;

  // Return the constraints available immediately before this task's own
  // condition guards take effect. This retains enclosing control and prior
  // Bind facts without assuming the conclusion of a guard implication proof.
  ConstrSet GetContextBeforeConditionGuards() const;

  Kind GetKind() const override { return Kind::kTask; }

  // Resource pipe mask
  uint16_t GetPipeMask() const override { return pipe_mask_; }
  uint16_t GetHbmMask() const override { return hbm_mask_; }

  // Special scalar registers written by this leaf (config/control intrinsics).
  uint16_t GetSpecialWriteMask() const override { return special_write_mask_; }

  // A padded GM->UB copy (data_select / pad_value) in this leaf reads the
  // pad-value register.
  bool ReadsPadValueRegister() const override { return reads_pad_value_; }

  // A copy that stores to GM in this leaf reads the atomic-mode register.
  bool ReadsAtomicRegister() const override { return reads_atomic_; }

  // Memory access regions (collected during analysis)
  std::vector<BufferRegion> GetReadRegions() const override {
    return read_regions_;
  }
  std::vector<BufferRegion> GetWriteRegions() const override {
    return write_regions_;
  }
  std::vector<Var> GetReadVars() const override { return read_vars_; }
  std::vector<Var> GetWriteVars() const override { return write_vars_; }

  // Setters
  void SetPipeMask(uint16_t mask) { pipe_mask_ = mask; }
  void SetHbmMask(uint16_t mask) { hbm_mask_ = mask; }
  void SetSpecialWriteMask(uint16_t mask) { special_write_mask_ = mask; }
  void SetReadsPadValue(bool v) { reads_pad_value_ = v; }
  void SetReadsAtomic(bool v) { reads_atomic_ = v; }
  void MarkPerCoreTask() { is_per_core_task_ = true; }
  bool IsPerCoreTask() const { return is_per_core_task_; }
  void AddSpecialWrite(SpecialRegister reg) {
    special_write_mask_ |= static_cast<uint16_t>(reg);
  }
  void SetReadRegions(const std::vector<BufferRegion> &regions) {
    read_regions_ = regions;
  }
  void SetWriteRegions(const std::vector<BufferRegion> &regions) {
    write_regions_ = regions;
  }
  void SetReadVars(const std::vector<Var> &vars) { read_vars_ = vars; }
  void SetWriteVars(const std::vector<Var> &vars) { write_vars_ = vars; }
  void SubstituteVar(const Var &old_var, const Var &new_var) override {
    Map<Var, PrimExpr> substitution{{old_var, new_var}};
    auto substitute_vars = [&](std::vector<Var> &vars) {
      for (auto &var : vars) {
        if (var.same_as(old_var))
          var = new_var;
      }
    };
    std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
        buffer_remap;
    auto substitute_regions = [&](std::vector<BufferRegion> &regions) {
      for (auto &region : regions) {
        Buffer buffer = region->buffer;
        auto buffer_it = buffer_remap.find(buffer);
        if (buffer_it != buffer_remap.end()) {
          buffer = buffer_it->second;
        } else if (buffer->data.same_as(old_var)) {
          Buffer remapped = buffer;
          remapped.CopyOnWrite()->data = new_var;
          buffer_remap.emplace(buffer, remapped);
          buffer = std::move(remapped);
        }
        region = BufferRegion(std::move(buffer),
                              Substitute(region->region, substitution));
      }
    };
    substitute_vars(read_vars_);
    substitute_vars(write_vars_);
    substitute_regions(read_regions_);
    substitute_regions(write_regions_);
    outer_ctx = outer_ctx.Substitute(substitution);
    stmt = Substitute(stmt, substitution);
    SubstituteVarInGuards(old_var, new_var);
    SubstituteVarInExtraInfo(old_var, new_var);
  }
  void SetCoreMask(CoreMask core_mask) {
    ICHECK(IsValidCoreMask(core_mask));
    core_mask_ = core_mask;
  }
  CoreMask GetCoreMask() const { return core_mask_; }

  // Clone method
  std::shared_ptr<IRStructure> Clone() const override;

  // Helper methods to add regions (for incremental analysis)
  void AddReadRegion(const BufferRegion &region) {
    // Check for duplicate regions
    for (const auto &existing : read_regions_) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        return; // Region already exists
      }
    }
    read_regions_.push_back(region);
  }

  void AddWriteRegion(const BufferRegion &region) {
    // Check for duplicate regions
    for (const auto &existing : write_regions_) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        return; // Region already exists
      }
    }
    write_regions_.push_back(region);
  }

  void AddReadVar(const Var &var) {
    for (const auto &existing : read_vars_) {
      if (existing.same_as(var))
        return;
    }
    read_vars_.push_back(var);
  }
  void AddWriteVar(const Var &var) {
    for (const auto &existing : write_vars_) {
      if (existing.same_as(var))
        return;
    }
    write_vars_.push_back(var);
  }

private:
  // Resource pipe mask (bitmask of ResourcePipe values)
  uint16_t pipe_mask_{0};
  uint16_t hbm_mask_{0};
  // Special scalar registers written by this leaf (bitmask of SpecialRegister).
  uint16_t special_write_mask_{0};
  // Whether a padded GM->UB copy in this leaf reads the pad-value register.
  bool reads_pad_value_{false};
  bool reads_atomic_{false};
  // A T.PerCoreTask is scheduled as one logical group. When InsertSync
  // materializes synchronization, same-core syncs are moved beside each
  // inferred or explicitly marked candidate.
  bool is_per_core_task_{false};

  // Memory access regions (collected during analysis)
  std::vector<BufferRegion> read_regions_;
  std::vector<BufferRegion> write_regions_;

  std::vector<Var> read_vars_;
  std::vector<Var> write_vars_;

  CoreMask core_mask_{kCoreUnassigned};
};

// Control node: contains a For operation and an ordered list of child
// IRStructures (the loop body). The SequenceNode level was removed: the loop
// body's ordered children live directly here.
class ControlNode : public IRStructure {
public:
  explicit ControlNode(std::unique_ptr<IRExtraInfo> extra_info = nullptr)
      : IRStructure(std::move(extra_info)) {}

  For control; // The For operation
  std::shared_ptr<TaskNode> task;
  std::vector<std::shared_ptr<IRStructure>> children;

  Kind GetKind() const override { return Kind::kControl; }

  bool BodyHasLoopBreak() const {
    for (const auto &child : children) {
      if (child && child->ContainsLoopBreak())
        return true;
    }
    return false;
  }

  // Resource pipe mask (aggregate from children)
  uint16_t GetPipeMask() const override {
    uint16_t mask = 0;
    for (const auto &child : children) {
      if (child)
        mask |= child->GetPipeMask();
    }
    return mask;
  }
  uint16_t GetHbmMask() const override {
    uint16_t mask = 0;
    for (const auto &child : children) {
      if (child)
        mask |= child->GetHbmMask();
    }
    return mask;
  }

  // Special-register writes survive the control boundary and must order
  // sibling nodes outside the loop. loop_break is the sole exception: it only
  // controls the loop that directly contains it and must not fence outer
  // siblings.
  uint16_t GetSpecialWriteMask() const override {
    uint16_t mask = task ? task->GetSpecialWriteMask() : 0;
    for (const auto &child : children) {
      if (child)
        mask |= child->GetSpecialWriteMask();
    }
    return mask & static_cast<uint16_t>(
                      ~static_cast<uint16_t>(SpecialRegister::kLoopControl));
  }

  // A loop reads the pad-value register if any child (or its own task) contains
  // a padded GM->UB copy.
  bool ReadsPadValueRegister() const override {
    for (const auto &child : children) {
      if (child && child->ReadsPadValueRegister())
        return true;
    }
    return task && task->ReadsPadValueRegister();
  }

  // A loop reads the atomic register if any child (or its own task) stores to
  // GM.
  bool ReadsAtomicRegister() const override {
    for (const auto &child : children) {
      if (child && child->ReadsAtomicRegister())
        return true;
    }
    return task && task->ReadsAtomicRegister();
  }

  // Memory access regions (aggregate from children & task)
  std::vector<BufferRegion> GetReadRegions() const override {
    std::vector<BufferRegion> regions;
    for (const auto &child : children) {
      if (child) {
        auto child_regions = child->GetReadRegions();
        regions.insert(regions.end(), child_regions.begin(),
                       child_regions.end());
      }
    }
    if (task) {
      auto task_regions = task->GetReadRegions();
      regions.insert(regions.end(), task_regions.begin(), task_regions.end());
    }
    return regions;
  }
  std::vector<BufferRegion> GetWriteRegions() const override {
    std::vector<BufferRegion> regions;
    for (const auto &child : children) {
      if (child) {
        auto child_regions = child->GetWriteRegions();
        regions.insert(regions.end(), child_regions.begin(),
                       child_regions.end());
      }
    }
    if (task) {
      auto task_regions = task->GetWriteRegions();
      regions.insert(regions.end(), task_regions.begin(), task_regions.end());
    }
    return regions;
  }

  // Variable access (aggregate from children & task)
  std::vector<Var> GetReadVars() const override {
    std::vector<Var> vars;
    for (const auto &child : children) {
      if (child) {
        auto child_vars = child->GetReadVars();
        vars.insert(vars.end(), child_vars.begin(), child_vars.end());
      }
    }
    if (task) {
      auto task_vars = task->GetReadVars();
      vars.insert(vars.end(), task_vars.begin(), task_vars.end());
    }
    return vars;
  }
  std::vector<Var> GetWriteVars() const override {
    std::vector<Var> vars;
    for (const auto &child : children) {
      if (child) {
        auto child_vars = child->GetWriteVars();
        vars.insert(vars.end(), child_vars.begin(), child_vars.end());
      }
    }
    if (task) {
      auto task_vars = task->GetWriteVars();
      vars.insert(vars.end(), task_vars.begin(), task_vars.end());
    }
    return vars;
  }

  void SubstituteVar(const Var &old_var, const Var &new_var) override {
    Map<Var, PrimExpr> substitution{{old_var, new_var}};
    for (auto &child : children) {
      if (child)
        child->SubstituteVar(old_var, new_var);
    }
    if (task)
      task->SubstituteVar(old_var, new_var);
    // Also substitute in the For statement's min/extent/step
    For new_for = control;
    new_for.CopyOnWrite()->min = Substitute(control->min, substitution);
    new_for.CopyOnWrite()->extent = Substitute(control->extent, substitution);
    if (control->step.has_value()) {
      new_for.CopyOnWrite()->step =
          Substitute(control->step.value(), substitution);
    }
    control = new_for;
    loop_body_ctx_ = loop_body_ctx_.Substitute(substitution);
    SubstituteVarInGuards(old_var, new_var);
    SubstituteVarInExtraInfo(old_var, new_var);
  }

  // Clone method
  std::shared_ptr<IRStructure> Clone() const override;

  int64_t GetTripCount() const {
    arith::Analyzer analyzer;
    if (task)
      task->outer_ctx.Populate(analyzer);
    return GetSerialTripCount(control, &analyzer);
  }

  ControlNode *GetParentControl() {
    IRStructure *parent = GetParent();
    if (parent == nullptr)
      return nullptr;
    ICHECK(parent->IsControl());
    return static_cast<ControlNode *>(parent);
  }

  const ControlNode *GetParentControl() const {
    const IRStructure *parent = GetParent();
    if (parent == nullptr)
      return nullptr;
    ICHECK(parent->IsControl());
    return static_cast<const ControlNode *>(parent);
  }

  // Return the root-to-this chain of enclosing loop controls, inclusive.
  std::vector<const ControlNode *> GetAncestorControls() const {
    std::vector<const ControlNode *> controls;
    for (const ControlNode *control = this; control != nullptr;
         control = control->GetParentControl()) {
      controls.push_back(control);
    }
    std::reverse(controls.begin(), controls.end());
    return controls;
  }

  // Constraints available when proving facts about this loop body: the loop's
  // outer context, iteration range, non-empty condition, and direct top-level
  // Bind definitions. Conditional definitions retain their definition guards.
  void InitializeLoopBodyContext();
  const ConstrSet &GetLoopBodyContext() const { return loop_body_ctx_; }

  ControlNode *GetOutermostControl() {
    ControlNode *outermost = this;
    while (ControlNode *parent = outermost->GetParentControl())
      outermost = parent;
    return outermost;
  }

  bool IsMultiBufferEligible(const Var &storage) const {
    auto annotation = control->annotations.Get(kMultiBufferEligible);
    if (!annotation.has_value())
      return false;
    for (const Var &eligible : annotation.value().cast<Array<Var>>()) {
      if (eligible.same_as(storage))
        return true;
    }
    return false;
  }

  bool BodyTouchesStorage(const Var &storage) const {
    for (const auto &child : children) {
      if (child && child->TouchesStorage(storage))
        return true;
    }
    return false;
  }

private:
  ConstrSet loop_body_ctx_;
};

// Pass-local multi-buffer owner loops keyed by storage identity. Each owner
// vector follows tree order at collection time and remains valid while the
// decoded IRStructure tree is alive.
using MultiBufferOwnerMap = std::unordered_map<Var, std::vector<ControlNode *>,
                                               ObjectPtrHash, ObjectPtrEqual>;

// Compare root-to-node (stage, index) paths. Returns -2 when lhs is an
// ancestor of rhs, +2 when rhs is an ancestor of lhs, and -1/+1 for ordinary
// lexicographic order. A null node is the virtual root above every tree.
int CompareIRStructure(IRStructure *lhs, IRStructure *rhs);

// Flatten the current iteration over a loop and all of its ancestors. The
// kernel top level (null) has iteration zero.
PrimExpr CalculateIterationCount(ControlNode *loop = nullptr);

// ConstrSet-populated analyzers are proof-only. Their rewrites must never enter
// emitted IR because the context may contain freshened mutable reads.
bool GuardsEquivalent(const PrimExpr &a, const PrimExpr &b,
                      const ConstrSet &outer_ctx);
bool GuardImplies(const PrimExpr &premise, const PrimExpr &conclusion,
                  const ConstrSet &outer_ctx);
Stmt WrapWithGuard(Stmt stmt, const PrimExpr &guard);

inline bool IsRegisterRegion(const BufferRegion &region) {
  String scope = region->buffer.scope();
  return scope == "local" || scope == "local.var" || scope == "local.fragment";
}

void CollectAllTaskNodes(IRStructure *node, std::vector<TaskNode *> &all_tasks);

inline void
CollectAllTaskNodes(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                    std::vector<TaskNode *> &all_tasks) {
  for (const auto &node : nodes)
    CollectAllTaskNodes(node.get(), all_tasks);
}

// Debug logging helpers.
void PrintIRStructure(const IRStructure *node, int indent = 0);

} // namespace tl
} // namespace tvm
