/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file normalize_pto_alloc_markers.cc
 * \brief NormalizePTOAllocMarkers turns frontend VMI allocation-kind metadata
 * into scoped AttrStmt carriers; ImportPTOAllocMarkers attaches the kind to the
 * final AllocBuffer after block lowering.
 */

#include <tvm/ir/cast.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ascend/target_utils.h"
#include "ascend/transform/attr.h"
#include "ascend/transform/vmi_alloc.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

/*! \brief Context label used by marker normalization diagnostics. */
constexpr const char *kNormalizeContext = "NormalizePTOAllocMarkers";
/*! \brief Context label used by marker import diagnostics. */
constexpr const char *kImportContext = "ImportPTOAllocMarkers";
/*! \brief Ascend VF scope attribute emitted by the frontend SIMD_VF frame. */
constexpr const char *kSimdVfScope = "tl.simdvf_scope";
/*! \brief SBlock name of the Ascend vector-function region. */
constexpr const char *kSimdVfBlock = "SIMD_VF";

/*! \brief Whether the target attribute selects the PTO pipeline. */
bool HasPtoTarget(const PrimFunc &func) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  return target.defined() && TargetIsPTO(target.value());
}

/*! \brief Wrap a body in deterministic nested allocation-kind carriers. */
Stmt InsertCarriers(Stmt body,
                    const std::vector<std::pair<Var, String>> &carriers) {
  // The first allocation in alloc_buffers order is the outermost marker.
  for (auto it = carriers.rbegin(); it != carriers.rend(); ++it) {
    body = AttrStmt(it->first, tl::attr::kVmiAllocKind, StringImm(it->second),
                    std::move(body));
  }
  return body;
}

/*!
 * \brief Insert carriers into the body of a marked SIMD_VF allocation owner.
 *
 * The existing outer `tl.simdvf_scope` statement must stay outermost because
 * later PTO lowering depends on it.  Carriers therefore go inside its body,
 * not before the scope statement.
 */
Stmt InsertInsideVfScope(Stmt body,
                         const std::vector<std::pair<Var, String>> &carriers) {
  const auto *scope = body.as<AttrStmtNode>();
  ICHECK(scope != nullptr && scope->attr_key == kSimdVfScope)
      << kNormalizeContext << " found a `" << kSimdVfBlock
      << "` allocation owner whose outermost statement is not `" << kSimdVfScope
      << "`; refusing to move VMI allocation markers outside "
      << "the existing VF scope";
  return AttrStmt(scope->node, scope->attr_key, scope->value,
                  InsertCarriers(scope->body, carriers), scope->span);
}

// Validate carriers that exist before normalization.  Collection proves that
// each carrier names an active allocation, but identity alone does not prove
// placement: a carrier at the root or in another SIMD_VF must not be accepted
// in place of the marker for the actual owner.  An existing carrier is a
// representation of the marker only when it is inside the region that
// normalization would emit it into for the block that owns the allocation.
class ExistingCarrierPlacementValidator : public StmtVisitor {
public:
  ExistingCarrierPlacementValidator(const vmi::AllocMarkers &markers,
                                    const char *context)
      : markers_(markers), context_(context) {}

  void Validate(const Stmt &stmt) {
    if (markers_.carrier_datas.empty()) {
      return;
    }
    tirx::PostOrderVisit(stmt, [this](const ffi::ObjectRef &node) {
      const auto *block = node.as<SBlockNode>();
      if (block == nullptr) {
        return;
      }
      for (const Buffer &buffer : block->alloc_buffers) {
        if (markers_.carrier_datas.count(buffer->data) == 0) {
          continue;
        }
        const StmtNode *region = OwnerRegion(block, buffer->name);
        owner_regions_[buffer->data.get()].push_back(region);
        region_nodes_.insert(region);
      }
    });
    VisitStmt(stmt);
  }

protected:
  void VisitStmt(const Stmt &stmt) final {
    bool region_pushed = region_nodes_.count(stmt.get()) != 0;
    if (region_pushed) {
      active_regions_.push_back(stmt.get());
    }
    bool vf_pushed = false;
    if (const auto *block = stmt.as<SBlockNode>()) {
      vf_pushed = block->name_hint == kSimdVfBlock;
    } else if (const auto *attr = stmt.as<AttrStmtNode>()) {
      vf_pushed = attr->attr_key == kSimdVfScope;
    }
    if (vf_pushed) {
      ++vf_depth_;
    }
    StmtVisitor::VisitStmt(stmt);
    if (vf_pushed) {
      --vf_depth_;
    }
    if (region_pushed) {
      active_regions_.pop_back();
    }
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tl::attr::kVmiAllocKind) {
      ffi::String kind;
      Var data = vmi::DecodeAllocKindCarrier(op, &kind, context_);
      ValidateCarrier(data);
    }
    StmtVisitor::VisitStmt_(op);
  }

private:
  const StmtNode *OwnerRegion(const SBlockNode *block, const String &name) {
    if (block->name_hint != kSimdVfBlock) {
      return block->body.get();
    }
    const auto *scope = block->body.as<AttrStmtNode>();
    ICHECK(scope != nullptr && scope->attr_key == kSimdVfScope)
        << context_ << " found a `" << kSimdVfBlock
        << "` allocation owner for `" << name
        << "` whose outermost statement is not `" << kSimdVfScope
        << "`; refusing to move VMI allocation markers outside the existing VF "
           "scope";
    return scope->body.get();
  }

  void ValidateCarrier(const Var &data) {
    auto owners = owner_regions_.find(data.get());
    if (owners != owner_regions_.end()) {
      const bool inside_owner = std::any_of(
          owners->second.begin(), owners->second.end(),
          [this](const StmtNode *region) {
            return std::find(active_regions_.begin(), active_regions_.end(),
                             region) != active_regions_.end();
          });
      ICHECK(inside_owner)
          << context_ << " found an existing `" << tl::attr::kVmiAllocKind
          << "` carrier for data Var `" << data->name_hint
          << "` outside the owning marker region of its allocation; a "
             "misplaced carrier must not suppress the correct marker";
      return;
    }
    ICHECK(vf_depth_ > 0)
        << context_ << " found an existing `" << tl::attr::kVmiAllocKind
        << "` carrier for data Var `" << data->name_hint << "` outside a `"
        << kSimdVfBlock
        << "` block; VMI allocation markers must stay inside T.SimdVF()";
  }

  const vmi::AllocMarkers &markers_;
  const char *context_;
  std::unordered_map<const VarNode *, std::vector<const StmtNode *>>
      owner_regions_;
  std::unordered_set<const StmtNode *> region_nodes_;
  std::vector<const StmtNode *> active_regions_;
  int vf_depth_{0};
};

// Consume frontend block maps and emit scoped AttrStmt carriers on the block
// that actually owns each marked allocation.  The kind table is keyed by data
// Var identity, so no name, shape, or allocation-order heuristic can redirect
// a marker.  Allocations, data Vars, and statement order stay unchanged, and
// carriers exist only inside a SIMD_VF region; an allocation owned outside one
// is rejected instead of being relocated or classified by inference.
class AllocMarkerCarrierEmitter : public StmtExprMutator {
public:
  explicit AllocMarkerCarrierEmitter(const vmi::AllocMarkers &markers)
      : markers_(markers) {}

  Stmt VisitStmt_(const SBlockNode *op) final {
    const bool block_inside_vf = inside_vf_ || op->name_hint == kSimdVfBlock;
    bool previous_inside_vf = inside_vf_;
    inside_vf_ = block_inside_vf;
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    inside_vf_ = previous_inside_vf;
    SBlock block = Downcast<SBlock>(stmt);

    // Collect one carrier per marked allocation, in alloc_buffers order.  A
    // data Var that already has a carrier or a final annotation is skipped so
    // re-running the pass neither duplicates markers nor rewrites the body.
    std::vector<std::pair<Var, String>> carriers;
    std::unordered_set<const VarNode *> emitted;
    for (const Buffer &buffer : block->alloc_buffers) {
      auto kind = markers_.kinds.find(buffer->data);
      if (kind == markers_.kinds.end()) {
        continue;
      }
      ICHECK(block_inside_vf)
          << kNormalizeContext << " `" << tl::attr::kVmiAllocKind
          << "` marks allocation `" << buffer->name
          << "` that is not owned inside a `" << kSimdVfBlock
          << "` block; VMI allocations must be created inside T.SimdVF()";
      if (markers_.carrier_datas.count(buffer->data) != 0 ||
          markers_.annotation_datas.count(buffer->data) != 0) {
        continue;
      }
      if (emitted.insert(buffer->data.get()).second) {
        carriers.emplace_back(buffer->data, kind->second);
      }
    }

    auto map_entry = block->annotations.find(tl::attr::kVmiAllocKind);
    const bool has_block_map = map_entry != block->annotations.end();
    if (carriers.empty() && !has_block_map) {
      return stmt;
    }

    Stmt new_body = block->body;
    if (!carriers.empty()) {
      new_body = block->name_hint == kSimdVfBlock
                     ? InsertInsideVfScope(block->body, carriers)
                     : InsertCarriers(block->body, carriers);
    }
    Map<String, Any> annotations = block->annotations;
    if (has_block_map) {
      // Consumed and residual entries both disappear with the block map;
      // no allocation is revived for a residual entry.
      annotations.erase(tl::attr::kVmiAllocKind);
    }
    auto writer = block.CopyOnWrite();
    writer->body = std::move(new_body);
    if (has_block_map) {
      writer->annotations = std::move(annotations);
    }
    return Stmt(block);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    // A bare VF scope can appear without its named SBlock wrapper.  Its body
    // still counts as SIMD_VF ownership for markers that were not represented
    // by a block map.
    if (op->attr_key != kSimdVfScope) {
      return StmtExprMutator::VisitStmt_(op);
    }
    bool previous_inside_vf = inside_vf_;
    inside_vf_ = true;
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    inside_vf_ = previous_inside_vf;
    return stmt;
  }

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    AllocBuffer alloc = Downcast<AllocBuffer>(stmt);
    auto kind = markers_.kinds.find(alloc->buffer->data);
    if (kind == markers_.kinds.end()) {
      return stmt;
    }
    ICHECK(inside_vf_)
        << kNormalizeContext << " `" << tl::attr::kVmiAllocKind
        << "` marks statement allocation `" << alloc->buffer->name
        << "` outside a `" << kSimdVfBlock
        << "` block; VMI allocations must be created inside T.SimdVF()";
    // A statement allocation is already in its final form.  Preserve an
    // existing annotation and otherwise attach the kind directly instead of
    // introducing a carrier that would have to wrap the statement.
    auto existing = vmi::FindAllocKind(alloc->annotations, kNormalizeContext);
    if (existing.has_value()) {
      ICHECK(existing.value() == kind->second)
          << kNormalizeContext << " found conflicting `"
          << tl::attr::kVmiAllocKind << "` values for one allocation: `"
          << existing.value() << "` vs `" << kind->second << "`";
      return stmt;
    }
    Map<String, Any> annotations = alloc->annotations;
    annotations.Set(tl::attr::kVmiAllocKind, kind->second);
    auto writer = alloc.CopyOnWrite();
    writer->annotations = std::move(annotations);
    return Stmt(alloc);
  }

private:
  const vmi::AllocMarkers &markers_;
  bool inside_vf_{false};
};

// Remove the temporary carriers and attach each kind to the AllocBuffer whose
// data Var identity matches.  The collection/validation pass has already
// proved every carrier names an active allocation and that no two
// representations disagree, so this pass only performs the identity match.
class AllocMarkerImporter : public StmtExprMutator {
public:
  explicit AllocMarkerImporter(const vmi::AllocKindMap &kinds)
      : kinds_(kinds) {}

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != tl::attr::kVmiAllocKind) {
      return StmtExprMutator::VisitStmt_(op);
    }
    ffi::String kind;
    vmi::DecodeAllocKindCarrier(op, &kind, kImportContext);
    // The allocation may be materialized before the VF scope marker, so the
    // match is performed by data Var identity over the whole body rather than
    // by lexical containment inside this carrier.
    return VisitStmt(op->body);
  }

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    AllocBuffer alloc = Downcast<AllocBuffer>(stmt);
    auto kind = kinds_.find(alloc->buffer->data);
    if (kind == kinds_.end()) {
      return stmt;
    }
    auto existing = vmi::FindAllocKind(alloc->annotations, kImportContext);
    if (existing.has_value()) {
      ICHECK(existing.value() == kind->second)
          << kImportContext << " found conflicting `" << tl::attr::kVmiAllocKind
          << "` values for one allocation: `" << existing.value() << "` vs `"
          << kind->second << "`";
      return stmt;
    }
    Map<String, Any> annotations = alloc->annotations;
    annotations.Set(tl::attr::kVmiAllocKind, kind->second);
    auto writer = alloc.CopyOnWrite();
    writer->annotations = std::move(annotations);
    return Stmt(alloc);
  }

private:
  const vmi::AllocKindMap &kinds_;
};

Stmt NormalizePTOAllocMarkers(const Stmt &stmt) {
  vmi::AllocMarkers markers;
  vmi::CollectAllocMarkers(stmt, &markers, kNormalizeContext);
  // Validate the placement of carriers that already exist before treating
  // them as existing markers.  This runs after collection so an orphan carrier
  // still gets the more specific identity diagnostic.
  ExistingCarrierPlacementValidator(markers, kNormalizeContext).Validate(stmt);
  // `markers.kinds` holds only data Vars with an active allocation, so an
  // all-residual input reaches the emitter with an empty table on purpose:
  // its stale block maps still have to be removed.  A function without any
  // marker is returned unchanged by the mutator.
  return AllocMarkerCarrierEmitter(markers)(stmt);
}

Stmt ImportPTOAllocMarkers(const Stmt &stmt) {
  vmi::AllocMarkers markers;
  vmi::CollectAllocMarkers(stmt, &markers, kImportContext);
  if (markers.carrier_datas.empty()) {
    // Nothing was emitted for this function; leave final annotations and all
    // unrelated IR untouched.
    return stmt;
  }
  return AllocMarkerImporter(markers.kinds)(stmt);
}

} // namespace

namespace transform {

using namespace tirx::transform;

Pass NormalizePTOAllocMarkers() {
  auto pass_func = [](PrimFunc func, const IRModule &, const PassContext &) {
    if (!HasPtoTarget(func)) {
      return func;
    }
    Stmt body = tl::NormalizePTOAllocMarkers(func->body);
    if (body.same_as(func->body)) {
      return func;
    }
    auto *node = func.CopyOnWrite();
    node->body = std::move(body);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.NormalizePTOAllocMarkers", {});
}

Pass ImportPTOAllocMarkers() {
  auto pass_func = [](PrimFunc func, const IRModule &, const PassContext &) {
    if (!HasPtoTarget(func)) {
      return func;
    }
    Stmt body = tl::ImportPTOAllocMarkers(func->body);
    if (body.same_as(func->body)) {
      return func;
    }
    auto *node = func.CopyOnWrite();
    node->body = std::move(body);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ImportPTOAllocMarkers", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.NormalizePTOAllocMarkers",
                        NormalizePTOAllocMarkers);
  refl::GlobalDef().def("tl.transform.ImportPTOAllocMarkers",
                        ImportPTOAllocMarkers);
}

} // namespace transform

} // namespace tl
} // namespace tvm
