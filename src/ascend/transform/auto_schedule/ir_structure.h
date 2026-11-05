#pragma once

#include "../buffer_version.h"
#include "../core_mask.h"
#include "ascend/op/utils.h"
#include "ir/loop_math.h"
#include "native_storage.h"
#include "task_resource.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm::tl::ascend {

using namespace tirx;
using namespace prim;
using ffi::Any;
using ffi::String;
using Region = ffi::Array<Range>;

class IRStructure;
class TaskNode;
class ControlNode;
class ScopeNode;

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

  auto analyzer = CreateTileLangAnalyzer();
  for (size_t i = 0; i < a.size(); ++i) {
    // Check if min values are equal
    if (!analyzer->CanProveEqual(a[i]->begin, b[i]->begin)) {
      return false;
    }
    // Check if extent values are equal
    if (!analyzer->CanProveEqual(a[i]->extent, b[i]->extent)) {
      return false;
    }
  }
  return true;
}

// Return true only when every point in `inner` is provably contained in
// `outer`. This is stricter than overlap and is suitable for checking whether
// one task's declared write footprint fully defines another task's read.
inline bool RegionCovers(const Region &outer, const Region &inner) {
  if (outer.size() != inner.size())
    return false;

  auto analyzer = CreateTileLangAnalyzer();
  for (size_t i = 0; i < outer.size(); ++i) {
    PrimExpr outer_end = outer[i]->begin + outer[i]->extent;
    PrimExpr inner_end = inner[i]->begin + inner[i]->extent;
    if (!analyzer->CanProve(outer[i]->begin <= inner[i]->begin) ||
        !analyzer->CanProve(inner_end <= outer_end)) {
      return false;
    }
  }
  return true;
}

constexpr int64_t kDynamicSerialTripCountFallback = 100;

inline int64_t GetSerialTripCount(const For &loop,
                                  const sym::AnalyzerObj *context = nullptr) {
  auto analyzer = context ? context->Clone() : CreateTileLangAnalyzer();
  // Native For always tests var < end, including negative and zero steps.
  // This API estimates scheduling cost; the fallback is not a proof that an
  // entering nonpositive-step loop terminates after a finite number of trips.
  if (SideEffect(loop->begin) > CallEffectKind::kPure ||
      SideEffect(loop->end) > CallEffectKind::kPure)
    return kDynamicSerialTripCountFallback;
  if (analyzer->CanProve(loop->begin >= loop->end))
    return 0;
  if (!CanUseLoopBounds(loop->begin, loop->end, loop->step, analyzer.get()))
    return kDynamicSerialTripCountFallback;
  // Widen before subtraction, so an int32 loop spanning zero does not wrap
  // into a negative trip count.
  PrimExpr distance =
      cast(loop->end, PrimType::Int(64)) - cast(loop->begin, PrimType::Int(64));
  auto step = TryConstInt(analyzer->Simplify(loop->step));
  auto count = TryConstInt(analyzer->Simplify(distance));
  int64_t extent;
  if (count) {
    extent = *count;
  } else {
    auto bound = analyzer->const_int_bound(distance);
    auto unconstrained = CreateTileLangAnalyzer();
    PrimVar variable("unconstrained_trip_count", distance.ty());
    if (bound->max_value < 0 ||
        bound->max_value == sym::ConstIntBound::kPosInf ||
        bound->max_value >= unconstrained->const_int_bound(variable)->max_value)
      return kDynamicSerialTripCountFallback;
    extent = bound->max_value;
  }
  if (extent <= 0)
    return kDynamicSerialTripCountFallback;
  // A dynamic positive step has a lower bound of one. Using an upper bound
  // would underestimate scheduling cost.
  uint64_t stride = step ? static_cast<uint64_t>(*step) : 1;
  uint64_t elements = static_cast<uint64_t>(extent);
  return static_cast<int64_t>(elements / stride + (elements % stride != 0));
}

// Record the loop definition before its body facts, so RenameFrom can give
// different iterations independent variables even without a finite range.
inline void AppendSerialLoopFacts(ConstrSet *context, const For &loop) {
  auto analyzer = CreateTileLangAnalyzer();
  auto outer = context->Populate(analyzer);
  bool pure_header = SideEffect(loop->begin) <= CallEffectKind::kPure &&
                     SideEffect(loop->end) <= CallEffectKind::kPure &&
                     SideEffect(loop->step) <= CallEffectKind::kPure;
  if (CanUseLoopBounds(loop->begin, loop->end, loop->step, analyzer.get())) {
    context->AddConstr(loop->var, Range(loop->begin, loop->end));
  } else {
    // A shared auxiliary symbol would equate independent producer/consumer
    // iterations after renaming. An identity bind records only the definition.
    context->AddConstr(loop->var, loop->var.as_or_throw<PrimExpr>());
    if (pure_header)
      context->AddConstr(loop->var < loop->end);
  }
  if (pure_header)
    context->AddConstr(loop->begin < loop->end);
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
    condition = SubstituteConstraintExpr(
        condition, {{old_var, new_var.as_or_throw<PrimExpr>()}});
  }
  Stmt Wrap(Stmt body) const final {
    return If(condition, NativeTaskAttribute::AsSequence(body), std::nullopt,
              body->loc);
  }

  PrimExpr condition;
};

class AttributeGuard final : public Guard {
public:
  AttributeGuard(Any node, String key, Any value, Location loc)
      : node(std::move(node)), key(std::move(key)), value(std::move(value)),
        loc(std::move(loc)) {}
  Kind GetKind() const final { return Kind::kAttribute; }
  std::unique_ptr<Guard> Clone() const final {
    return std::make_unique<AttributeGuard>(node, key, value, loc);
  }
  void SubstituteVar(const Var &old_var, const Var &new_var) final {
    auto attribute = MakeNativeAttribute(node, key, value, SeqStmt({}), loc);
    auto remapped =
        ReadTaskAttribute(RemapTaskStatement(attribute, {{old_var, new_var}}))
            .value();
    node = remapped.node;
    value = remapped.value;
  }
  Stmt Wrap(Stmt body) const final {
    return MakeNativeAttribute(node, key, value,
                               NativeTaskAttribute::AsSequence(body), loc);
  }
  Any node;
  String key;
  Any value;
  Location loc;
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
  enum class Kind { kTask, kControl, kScope };

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

  ControlNode *GetParentControl();
  const ControlNode *GetParentControl() const;

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
  // Return whether condition or attribute-node guards access this storage.
  bool GuardsTouchStorage(const Var &storage) const;
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
  void PrependAttribute(const Any &node, const String &key, const Any &value,
                        const Location &loc) {
    guards_.insert(guards_.begin(),
                   std::make_unique<AttributeGuard>(node, key, value, loc));
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
  bool IsScope() const { return GetKind() == Kind::kScope; }

  using ChildList = std::vector<std::shared_ptr<IRStructure>>;
  virtual std::vector<ChildList *> ChildLists() { return {}; }
  virtual std::vector<const ChildList *> ChildLists() const { return {}; }
  virtual TaskNode *HeaderTask() { return nullptr; }
  virtual const TaskNode *HeaderTask() const { return nullptr; }

  void SetStorageContext(std::shared_ptr<const NativeStorageContext> value) {
    storage_context_ = std::move(value);
  }
  const NativeStorageContext &StorageContext() const {
    static const NativeStorageContext empty;
    return storage_context_ ? *storage_context_ : empty;
  }
  Var StorageIdentity(const TensorVar &tensor) const {
    return StorageContext().Identity(tensor);
  }

  // Substitute a var inside this node's own guard deltas. Call from each
  // node's SubstituteVar override so core specialization reaches both If
  // conditions and propagated AttrStmt fields.
  void SubstituteVarInGuards(const Var &old_var, const Var &new_var) {
    for (auto &guard : guards_) {
      guard->SubstituteVar(old_var, new_var);
    }
    // Cloned/specialized tasks share the original context until their own IR
    // changes. Keep pointer origins consistent with the rewritten statements
    // without changing provenance in sibling tasks or the original tree.
    if (storage_context_)
      storage_context_ = std::make_shared<NativeStorageContext>(
          storage_context_->Substitute(old_var, new_var));
  }

  void CopyBaseMetadataFrom(const IRStructure &other) {
    guards_.clear();
    guards_.reserve(other.guards_.size());
    for (const auto &guard : other.guards_) {
      guards_.push_back(guard->Clone());
    }
    storage_context_ = other.storage_context_;
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

  // Logical access regions. Physical footprints are derived from current IR.
  virtual std::vector<TensorRegion> GetReadRegions() const = 0;
  virtual std::vector<TensorRegion> GetWriteRegions() const = 0;

  // Storage identity is the Buffer data Var, so aliases with different shapes
  // and strides are treated as accesses to the same physical allocation.
  bool ReadsStorage(const Var &storage) const {
    for (const TensorRegion &region : GetReadRegions()) {
      if (StorageIdentity(RegionTensor(region)).same_as(storage))
        return true;
    }
    return false;
  }
  bool WritesStorage(const Var &storage) const {
    for (const TensorRegion &region : GetWriteRegions()) {
      if (StorageIdentity(RegionTensor(region)).same_as(storage))
        return true;
    }
    return false;
  }
  bool TouchesStorage(const Var &storage) const {
    return ReadsStorage(storage) || WritesStorage(storage);
  }

  // Return unique on-chip storage identities touched by this node. Composite
  // nodes include their complete subtree through GetRead/WriteRegions().
  // MX scale-factor handles count as storage here so the multi-buffer
  // eligibility planner claims an owner loop for them; their version count
  // is never chosen independently — AutoSchedule copies it from the bound
  // data tile after selection.
  std::vector<Var> GetOnChipStorages() const {
    std::vector<Var> storages;
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> seen;
    auto collect = [&](const std::vector<TensorRegion> &regions) {
      for (const TensorRegion &region : regions) {
        if ((IsAscendOnChipBuffer(RegionTensor(region)) ||
             IsL0SFBuffer(RegionTensor(region))) &&
            seen.insert(StorageIdentity(RegionTensor(region))).second) {
          storages.push_back(StorageIdentity(RegionTensor(region)));
        }
      }
    };
    collect(GetReadRegions());
    collect(GetWriteRegions());
    return storages;
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
  //   Cube (MAD) reads HF32 mode and MAD direction (derived from the Cube pipe)
  //   padded GM->UB copies read the pad-value register (only copies that carry
  //     data_select / pad_value)
  //   loop_break is a fence: every node implicitly reads the loop-control
  //     register, so any loop_break write orders against all siblings.
  uint16_t GetSpecialReadMask() const {
    uint16_t mask = static_cast<uint16_t>(SpecialRegister::kLoopControl);
    if (UsesPipe(ResourcePipe::kCube)) {
      mask |= static_cast<uint16_t>(SpecialRegister::kHf32Mode) |
              static_cast<uint16_t>(SpecialRegister::kMmadDirection);
    }
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
  std::shared_ptr<const NativeStorageContext> storage_context_;
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

  Stmt stmt{ffi::UnsafeInit{}};

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

  // Logical access regions. Physical footprints are derived from current IR.
  std::vector<TensorRegion> GetReadRegions() const override {
    return read_regions_;
  }
  std::vector<TensorRegion> GetWriteRegions() const override {
    return write_regions_;
  }
  std::vector<Var> GetReadVars() const override { return read_vars_; }
  std::vector<Var> GetWriteVars() const override { return write_vars_; }
  const std::vector<Var> &GetOpaqueAccessVars() const {
    return opaque_access_vars_;
  }

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
  void SetReadRegions(const std::vector<TensorRegion> &regions) {
    read_regions_ = regions;
  }
  void SetWriteRegions(const std::vector<TensorRegion> &regions) {
    write_regions_ = regions;
  }
  const Map<Var, Optional<PrimExpr>> &GetL0Pitches() const {
    return l0_pitches_;
  }
  Optional<PrimExpr> GetL0Pitch(const Var &storage) const {
    auto it = l0_pitches_.find(storage);
    return it == l0_pitches_.end() ? Optional<PrimExpr>() : (*it).second;
  }
  void SetL0Pitches(const Map<Var, Optional<PrimExpr>> &pitches) {
    l0_pitches_ = pitches;
  }
  void SetReadVars(const std::vector<Var> &vars) { read_vars_ = vars; }
  void SetWriteVars(const std::vector<Var> &vars) { write_vars_ = vars; }
  void SetOpaqueAccessVars(const std::vector<Var> &vars) {
    opaque_access_vars_ = vars;
  }
  void AddOpaqueAccessVar(const Var &var) {
    if (std::none_of(
            opaque_access_vars_.begin(), opaque_access_vars_.end(),
            [&](const Var &existing) { return existing.same_as(var); }))
      opaque_access_vars_.push_back(var);
  }
  void SubstituteVar(const Var &old_var, const Var &new_var) override {
    ffi::Map<Var, Expr> substitution{{old_var, new_var}};
    auto substitute_vars = [&](std::vector<Var> &vars) {
      for (auto &var : vars)
        if (var.same_as(old_var))
          var = new_var;
    };
    substitute_vars(read_vars_);
    substitute_vars(write_vars_);
    substitute_vars(opaque_access_vars_);
    for (auto &region : read_regions_)
      region = RemapTaskRegion(region, substitution);
    for (auto &region : write_regions_)
      region = RemapTaskRegion(region, substitution);
    if (auto primitive = new_var.as<PrimExpr>()) {
      ffi::Map<Var, PrimExpr> values{{old_var, primitive.value()}};
      Map<Var, Optional<PrimExpr>> pitches;
      for (const auto &[storage, pitch] : l0_pitches_)
        pitches.Set(
            storage.same_as(old_var) ? new_var : storage,
            pitch ? Optional<PrimExpr>(SubstituteConstraintExpr(*pitch, values))
                  : std::nullopt);
      l0_pitches_ = std::move(pitches);
      outer_ctx = outer_ctx.Substitute(values);
    }
    stmt = RemapTaskStatement(stmt, substitution);
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
  void AddReadRegion(const TensorRegion &region) {
    // Check for duplicate regions
    for (const auto &existing : read_regions_) {
      if (existing->source.same_as(region->source) &&
          RegionsEqual(existing->region, region->region)) {
        return; // Region already exists
      }
    }
    read_regions_.push_back(region);
  }

  void AddWriteRegion(const TensorRegion &region) {
    // Check for duplicate regions
    for (const auto &existing : write_regions_) {
      if (existing->source.same_as(region->source) &&
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

  // Logical access regions. Physical footprints are derived from current IR.
  std::vector<TensorRegion> read_regions_;
  std::vector<TensorRegion> write_regions_;
  Map<Var, Optional<PrimExpr>> l0_pitches_;
  std::vector<Var> read_vars_;
  std::vector<Var> write_vars_;

  // Memory references with no precise region. These retain dependencies even
  // when the pointer came from an opaque native expression.
  std::vector<Var> opaque_access_vars_;

  CoreMask core_mask_{kCoreUnassigned};
};

// Control node: contains a For operation and an ordered list of child
// IRStructures (the loop body). The SequenceNode level was removed: the loop
// body's ordered children live directly here.
class ControlNode : public IRStructure {
public:
  explicit ControlNode(std::unique_ptr<IRExtraInfo> extra_info = nullptr)
      : IRStructure(std::move(extra_info)) {}

  For control{ffi::UnsafeInit{}}; // The For operation
  std::shared_ptr<TaskNode> task;
  std::vector<std::shared_ptr<IRStructure>> children;

  Kind GetKind() const override { return Kind::kControl; }
  std::vector<ChildList *> ChildLists() final { return {&children}; }
  std::vector<const ChildList *> ChildLists() const final {
    return {&children};
  }
  TaskNode *HeaderTask() final { return task.get(); }
  const TaskNode *HeaderTask() const final { return task.get(); }

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
  std::vector<TensorRegion> GetReadRegions() const override {
    std::vector<TensorRegion> regions;
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
  std::vector<TensorRegion> GetWriteRegions() const override {
    std::vector<TensorRegion> regions;
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
    Map<Var, PrimExpr> substitution{{old_var, new_var.as_or_throw<PrimExpr>()}};
    for (auto &child : children)
      if (child)
        child->SubstituteVar(old_var, new_var);
    if (task)
      task->SubstituteVar(old_var, new_var);
    For new_for = control;
    auto *writer = new_for.CopyOnWrite();
    writer->begin = SubstituteConstraintExpr(control->begin, substitution);
    writer->end = SubstituteConstraintExpr(control->end, substitution);
    writer->step = SubstituteConstraintExpr(control->step, substitution);
    control = new_for;
    loop_body_ctx_ = loop_body_ctx_.Substitute(substitution);
    SubstituteVarInGuards(old_var, new_var);
    SubstituteVarInExtraInfo(old_var, new_var);
  }

  // Clone method
  std::shared_ptr<IRStructure> Clone() const override;

  int64_t GetTripCount() const {
    auto analyzer = CreateTileLangAnalyzer();
    auto context = task ? task->outer_ctx.Populate(analyzer) : nullptr;
    return GetSerialTripCount(control, analyzer.get());
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
    auto annotation =
        ReadLoopAnnotations(control.get()).Get(kMultiBufferEligible);
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

// A lexical If, assumption/loop-metadata region, or metadata statement.
// Children stay inside the exact original scope, so mutable conditions execute
// once and branch-local definitions can never leak into sibling sequences.
class ScopeNode : public IRStructure {
public:
  explicit ScopeNode(std::unique_ptr<IRExtraInfo> extra = nullptr)
      : IRStructure(std::move(extra)) {}
  Stmt shell{ffi::UnsafeInit{}};
  // The lexical context before this scope/header. Metadata statements have no
  // schedulable header task, but their typed operands still have a definition
  // and execution domain that analysis must preserve.
  ConstrSet outer_ctx;
  Optional<PrimExpr> condition_snapshot;
  std::shared_ptr<TaskNode> task;
  ChildList children;
  ChildList else_children;
  bool has_else{false};
  Kind GetKind() const final { return Kind::kScope; }
  std::shared_ptr<IRStructure> Clone() const final;
  std::vector<ChildList *> ChildLists() final {
    return has_else ? std::vector<ChildList *>{&children, &else_children}
                    : std::vector<ChildList *>{&children};
  }
  std::vector<const ChildList *> ChildLists() const final {
    return has_else ? std::vector<const ChildList *>{&children, &else_children}
                    : std::vector<const ChildList *>{&children};
  }
  TaskNode *HeaderTask() final { return task.get(); }
  const TaskNode *HeaderTask() const final { return task.get(); }

  template <typename F> void Each(F visit) const {
    if (task)
      visit(task.get());
    for (const auto *list : ChildLists())
      for (const auto &child : *list)
        visit(child.get());
  }
  uint16_t GetPipeMask() const final {
    uint16_t result = 0;
    Each([&](const IRStructure *node) { result |= node->GetPipeMask(); });
    return result;
  }
  uint16_t GetHbmMask() const final {
    uint16_t result = 0;
    Each([&](const IRStructure *node) { result |= node->GetHbmMask(); });
    return result;
  }
  uint16_t GetSpecialWriteMask() const final {
    uint16_t result = 0;
    Each([&](const IRStructure *node) {
      result |= node->GetSpecialWriteMask();
    });
    return result;
  }
  bool ReadsPadValueRegister() const final {
    bool result = false;
    Each([&](const IRStructure *node) {
      result |= node->ReadsPadValueRegister();
    });
    return result;
  }
  bool ReadsAtomicRegister() const final {
    bool result = false;
    Each([&](const IRStructure *node) {
      result |= node->ReadsAtomicRegister();
    });
    return result;
  }
  std::vector<TensorRegion> GetReadRegions() const final {
    std::vector<TensorRegion> result;
    Each([&](const IRStructure *node) {
      auto values = node->GetReadRegions();
      result.insert(result.end(), values.begin(), values.end());
    });
    return result;
  }
  std::vector<TensorRegion> GetWriteRegions() const final {
    std::vector<TensorRegion> result;
    Each([&](const IRStructure *node) {
      auto values = node->GetWriteRegions();
      result.insert(result.end(), values.begin(), values.end());
    });
    return result;
  }
  std::vector<Var> GetReadVars() const final {
    std::vector<Var> result;
    Each([&](const IRStructure *node) {
      auto values = node->GetReadVars();
      result.insert(result.end(), values.begin(), values.end());
    });
    return result;
  }
  std::vector<Var> GetWriteVars() const final {
    std::vector<Var> result;
    Each([&](const IRStructure *node) {
      auto values = node->GetWriteVars();
      result.insert(result.end(), values.begin(), values.end());
    });
    return result;
  }
  void SubstituteVar(const Var &old_var, const Var &new_var) final {
    shell = RemapTaskStatement(shell, {{old_var, new_var}});
    if (auto primitive = new_var.as<PrimExpr>()) {
      outer_ctx = outer_ctx.Substitute({{old_var, primitive.value()}});
      if (condition_snapshot)
        condition_snapshot = SubstituteConstraintExpr(
            *condition_snapshot, {{old_var, primitive.value()}});
    }
    if (task)
      task->SubstituteVar(old_var, new_var);
    for (auto *list : ChildLists())
      for (auto &child : *list)
        child->SubstituteVar(old_var, new_var);
    SubstituteVarInGuards(old_var, new_var);
    SubstituteVarInExtraInfo(old_var, new_var);
  }
};

inline ControlNode *IRStructure::GetParentControl() {
  IRStructure *parent = GetParent();
  if (parent == nullptr)
    return nullptr;
  while (parent && !parent->IsControl())
    parent = parent->GetParent();
  return static_cast<ControlNode *>(parent);
}

inline const ControlNode *IRStructure::GetParentControl() const {
  const IRStructure *parent = GetParent();
  if (parent == nullptr)
    return nullptr;
  while (parent && !parent->IsControl())
    parent = parent->GetParent();
  return static_cast<const ControlNode *>(parent);
}

// Pass-local multi-buffer owner loops keyed by storage identity. Each owner
// vector follows tree order at collection time and remains valid while the
// decoded IRStructure tree is alive.
using MultiBufferOwnerMap = std::unordered_map<Var, std::vector<ControlNode *>,
                                               ObjectPtrHash, ObjectPtrEqual>;

// Compare root-to-node (stage, index) paths. Returns -2 when lhs is an
// ancestor of rhs, +2 when rhs is an ancestor of lhs, and -1/+1 for ordinary
// lexicographic order. A null node is the virtual root above every tree.
int CompareIRStructure(IRStructure *lhs, IRStructure *rhs);

struct SerialInductionInfo {
  bool no_wrap{false};
  // Inclusive machine-coordinate bounds when representable in int64.
  std::optional<std::pair<int64_t, int64_t>> bounds;
  bool int64_delta{false};
};

// Header range facts or a mandatory tail Break may prove that every executed
// induction update stays within its machine type. This is not a trip count.
SerialInductionInfo AnalyzeSerialInduction(const ControlNode *loop);

// Flatten the current iteration over a loop and all of its ancestors. The
// kernel top level (null) has iteration zero.
// Lexical mixed-radix coordinate, not a count of completed iterations. Inner
// positive For ranges provide nominal spans; early Break may leave holes.
// A single signed nonzero-step loop uses its local (var - begin) / step
// ordinal only with no-wrap evidence. Unknown machine-coordinate or int64
// arithmetic bounds return nullopt; the strict wrapper diagnoses that case.
Optional<PrimExpr> TryCalculateIterationCount(ControlNode *loop = nullptr);
PrimExpr CalculateIterationCount(ControlNode *loop = nullptr);

// ConstrSet-populated analyzers are proof-only. Their rewrites must never enter
// emitted IR because the context may contain freshened mutable reads.
bool GuardsEquivalent(const PrimExpr &a, const PrimExpr &b,
                      const ConstrSet &outer_ctx);
bool GuardImplies(const PrimExpr &premise, const PrimExpr &conclusion,
                  const ConstrSet &outer_ctx);
Stmt WrapWithGuard(Stmt stmt, const PrimExpr &guard);

inline bool IsRegisterRegion(const TensorRegion &region) {
  String scope = RegionTensor(region).scope().name();
  return scope == "local" || scope == "local.var" || scope == "local.fragment";
}

void CollectAllTaskNodes(IRStructure *node, std::vector<TaskNode *> &all_tasks);
void CollectTaskNodesAndHeaders(IRStructure *node,
                                std::vector<TaskNode *> &tasks);

inline void
CollectAllTaskNodes(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                    std::vector<TaskNode *> &all_tasks) {
  for (const auto &node : nodes)
    CollectAllTaskNodes(node.get(), all_tasks);
}

inline void CollectTaskNodesAndHeaders(
    const std::vector<std::shared_ptr<IRStructure>> &nodes,
    std::vector<TaskNode *> &tasks) {
  for (const auto &node : nodes)
    CollectTaskNodesAndHeaders(node.get(), tasks);
}

// Debug logging helpers.
void PrintIRStructure(const IRStructure *node, int indent = 0);

} // namespace tvm::tl::ascend
