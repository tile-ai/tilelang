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
 * \brief NormalizePTOAllocMarkers: rebuild VMI allocation-kind metadata on
 * the allocations that actually own it.
 */

#include <tvm/ir/cast.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "ascend/target_utils.h"
#include "transform/common/attr.h"
#include "transform/common/vmi_alloc.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

/*! \brief Context label used in every diagnostic of this pass. */
constexpr const char *kContext = "NormalizePTOAllocMarkers";

// Compare by data Var identity and kind value: a freshly built map that
// encodes the same entries as the recorded one leaves the IR untouched, so
// re-running the pass on its own output keeps PrimFunc identity.
bool SameAllocKinds(const vmi::AllocKindMap &lhs,
                    const vmi::AllocKindMap &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (const auto &[data, kind] : lhs) {
    auto it = rhs.find(data);
    if (it == rhs.end() || !(it->second == kind)) {
      return false;
    }
  }
  return true;
}

// Rebuild each SBlock's `tl.vmi.alloc_kind` map from its own alloc_buffers
// and attach the kind string to marked statement allocations.  The kinds
// table comes from the shared collector and is keyed by data Var identity,
// so an entry migrates from whatever block recorded it to every block that
// actually holds the allocation, and to the statement allocation itself.
// Nothing else moves: allocations stay where they are, Buffer.data Vars keep
// their identity, and unmarked allocations gain no new restriction.
class AllocMarkerOwnerRewriter : public StmtExprMutator {
public:
  explicit AllocMarkerOwnerRewriter(const vmi::AllocKindMap &kinds)
      : kinds_(kinds) {}

  Stmt VisitStmt_(const SBlockNode *op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    SBlock block = Downcast<SBlock>(stmt);
    // The local map contains only this block's own marked allocations;
    // entries recorded on an outer block disappear with the rebuild.
    vmi::AllocKindMap owned;
    for (const Buffer &buffer : block->alloc_buffers) {
      auto kind = kinds_.find(buffer->data);
      if (kind != kinds_.end()) {
        owned.emplace(buffer->data, kind->second);
      }
    }
    auto annotation = block->annotations.find(tl::attr::kVmiAllocKind);
    if (annotation == block->annotations.end()) {
      if (owned.empty()) {
        return stmt;
      }
    } else {
      vmi::AllocKindMap existing;
      vmi::CollectAllocKindMap((*annotation).second, &existing, kContext);
      // Only a non-empty recorded map equal to the rebuilt one keeps the
      // original block.  An empty map is never a final state: a block that
      // owns no marked allocation must not carry the key at all.
      if (!existing.empty() && SameAllocKinds(existing, owned)) {
        return stmt;
      }
    }
    Map<String, Any> annotations = block->annotations;
    if (owned.empty()) {
      // Residual entries whose allocation no longer exists are dropped with
      // the consumed map; no allocation is revived for them.
      annotations.erase(tl::attr::kVmiAllocKind);
    } else {
      Map<Var, String> encoded;
      for (const auto &[data, kind] : owned) {
        encoded.Set(data, kind);
      }
      annotations.Set(tl::attr::kVmiAllocKind, encoded);
    }
    auto writer = block.CopyOnWrite();
    writer->annotations = std::move(annotations);
    return Stmt(block);
  }

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    AllocBuffer alloc = Downcast<AllocBuffer>(stmt);
    auto kind = kinds_.find(alloc->buffer->data);
    if (kind == kinds_.end()) {
      return stmt;
    }
    // A statement allocation carries its kind itself; the kind never moves
    // back onto an enclosing block map.  Conflicting values were rejected
    // by the shared collector, so an existing annotation is preserved.
    auto existing = vmi::FindAllocKind(alloc->annotations, kContext);
    if (existing.has_value()) {
      ICHECK(existing.value() == kind->second)
          << kContext << " found conflicting `" << tl::attr::kVmiAllocKind
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
  vmi::CollectAllocMarkers(stmt, &markers, kContext);
  // `markers.kinds` holds only data Vars with an active allocation, so an
  // all-residual input reaches the rewriter with an empty table on purpose:
  // its stale maps still have to be removed.  A function without any marker
  // rewrites nothing and returns the input unchanged.
  return AllocMarkerOwnerRewriter(markers.kinds)(stmt);
}

} // namespace

namespace transform {

using namespace tirx::transform;

Pass NormalizePTOAllocMarkers() {
  auto pass_func = [](PrimFunc func, const IRModule &, const PassContext &) {
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value())) {
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

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.NormalizePTOAllocMarkers",
                        NormalizePTOAllocMarkers);
}

} // namespace transform

} // namespace tl
} // namespace tvm
