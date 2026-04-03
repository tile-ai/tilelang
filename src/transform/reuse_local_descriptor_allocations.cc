/*!
 * \file reuse_local_descriptor_allocations.cc
 * \brief Pool lexically-disjoint local descriptor allocations.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
namespace refl = tvm::ffi::reflection;

namespace {

bool IsLocalDescriptorScope(const Var &buffer_var) {
  std::string scope = GetPtrStorageScope(buffer_var);
  return scope.rfind("local.descriptor.", 0) == 0;
}

bool IsReusableDescriptorAllocate(const AllocateNode *op) {
  return IsLocalDescriptorScope(op->buffer_var) && is_one(op->condition) &&
         op->annotations.empty() && op->ConstantAllocationSize() > 0;
}

std::string MakeDescriptorSignature(const AllocateNode *op) {
  const DataType &dtype = op->dtype;
  return GetPtrStorageScope(op->buffer_var) + "|" +
         std::to_string(dtype.code()) + ":" + std::to_string(dtype.bits()) +
         ":" + std::to_string(dtype.lanes()) + "|" +
         std::to_string(op->ConstantAllocationSize());
}

struct AllocSite {
  Var var;
  DataType dtype;
  ffi::Array<PrimExpr> extents;
  ffi::Map<ffi::String, ffi::Any> annotations;
  std::string signature;
};

class DescriptorAllocCollector : public StmtExprVisitor {
public:
  static std::vector<AllocSite> Collect(const Stmt &stmt) {
    DescriptorAllocCollector collector;
    collector(stmt);
    return std::move(collector.allocs_);
  }

private:
  void VisitStmt_(const AllocateNode *op) final {
    if (IsReusableDescriptorAllocate(op)) {
      allocs_.push_back(AllocSite{
          op->buffer_var, op->dtype, op->extents, op->annotations,
          MakeDescriptorSignature(op)});
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  std::vector<AllocSite> allocs_;
};

class DescriptorVarRemapper : public StmtExprMutator {
public:
  DescriptorVarRemapper(
      std::unordered_map<const VarNode *, Var> var_remap,
      std::unordered_set<const VarNode *> removed_allocs)
      : var_remap_(std::move(var_remap)),
        removed_allocs_(std::move(removed_allocs)) {}

private:
  PrimExpr VisitExpr_(const VarNode *op) final {
    if (auto it = var_remap_.find(op); it != var_remap_.end()) {
      return it->second;
    }
    return tvm::ffi::GetRef<Var>(op);
  }

  Stmt VisitStmt_(const AllocateNode *op) final {
    if (removed_allocs_.count(op->buffer_var.get())) {
      return VisitStmt(op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const DeclBufferNode *op) final {
    auto node = Downcast<DeclBuffer>(StmtExprMutator::VisitStmt_(op));
    Buffer new_buffer = RemapBuffer(node->buffer);
    if (!new_buffer.same_as(node->buffer)) {
      node.CopyOnWrite()->buffer = new_buffer;
    }
    return std::move(node);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    Buffer new_buffer = RemapBuffer(node->buffer);
    if (!new_buffer.same_as(node->buffer)) {
      node.CopyOnWrite()->buffer = new_buffer;
    }
    return std::move(node);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    Buffer new_buffer = RemapBuffer(node->buffer);
    if (!new_buffer.same_as(node->buffer)) {
      node.CopyOnWrite()->buffer = new_buffer;
    }
    return std::move(node);
  }

  Buffer RemapBuffer(Buffer buffer) const {
    if (auto it = var_remap_.find(buffer->data.get()); it != var_remap_.end()) {
      Buffer new_buffer = buffer;
      new_buffer.CopyOnWrite()->data = it->second;
      return new_buffer;
    }
    return buffer;
  }

  std::unordered_map<const VarNode *, Var> var_remap_;
  std::unordered_set<const VarNode *> removed_allocs_;
};

class ReuseLocalDescriptorAllocationsMutator : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc func) {
    auto fptr = func.CopyOnWrite();
    ReuseLocalDescriptorAllocationsMutator rewriter;
    fptr->body = rewriter(std::move(fptr->body));
    return func;
  }

private:
  struct PoolSlot {
    AllocSite canonical;
    int use_count{0};
  };

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    ffi::Array<Stmt> visited_children;
    visited_children.reserve(op->seq.size());
    for (const Stmt &stmt : op->seq) {
      visited_children.push_back(VisitStmt(stmt));
    }

    std::unordered_map<std::string, std::vector<int>> signature_slots;
    std::unordered_map<const VarNode *, int> alloc_to_slot;
    std::vector<PoolSlot> slots;

    for (const Stmt &stmt : visited_children) {
      std::unordered_map<std::string, int> local_slot_index;
      for (const AllocSite &alloc : DescriptorAllocCollector::Collect(stmt)) {
        int ordinal = local_slot_index[alloc.signature]++;
        std::vector<int> &sig_slots = signature_slots[alloc.signature];
        if (static_cast<int>(sig_slots.size()) <= ordinal) {
          sig_slots.push_back(static_cast<int>(slots.size()));
          slots.push_back(PoolSlot{alloc, 0});
        }
        int slot_idx = sig_slots[ordinal];
        alloc_to_slot[alloc.var.get()] = slot_idx;
        ++slots[slot_idx].use_count;
      }
    }

    std::unordered_map<const VarNode *, Var> var_remap;
    std::unordered_set<const VarNode *> removed_allocs;
    std::vector<AllocSite> hoisted_allocs;
    hoisted_allocs.reserve(slots.size());

    for (const PoolSlot &slot : slots) {
      if (slot.use_count <= 1) {
        continue;
      }
      removed_allocs.insert(slot.canonical.var.get());
      hoisted_allocs.push_back(slot.canonical);
    }

    if (hoisted_allocs.empty()) {
      return visited_children.size() == 1 ? visited_children[0]
                                          : SeqStmt(visited_children);
    }

    for (const auto &[var, slot_idx] : alloc_to_slot) {
      if (slots[slot_idx].use_count <= 1) {
        continue;
      }
      removed_allocs.insert(var);
      const Var &canonical_var = slots[slot_idx].canonical.var;
      if (var != canonical_var.get()) {
        var_remap[var] = canonical_var;
      }
    }

    DescriptorVarRemapper rewriter(std::move(var_remap),
                                   std::move(removed_allocs));
    ffi::Array<Stmt> rewritten_children;
    rewritten_children.reserve(visited_children.size());
    for (const Stmt &stmt : visited_children) {
      rewritten_children.push_back(rewriter(stmt));
    }

    Stmt body = rewritten_children.size() == 1 ? rewritten_children[0]
                                               : SeqStmt(rewritten_children);
    for (auto it = hoisted_allocs.rbegin(); it != hoisted_allocs.rend(); ++it) {
      body = Allocate(it->var, it->dtype, it->extents, const_true(),
                      std::move(body), it->annotations);
    }
    return body;
  }
};

} // namespace

tir::transform::Pass ReuseLocalDescriptorAllocations() {
  auto pass_func = [](PrimFunc func, IRModule mod,
                      tvm::transform::PassContext ctx) {
    return ReuseLocalDescriptorAllocationsMutator::Rewrite(std::move(func));
  };
  return tir::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.ReuseLocalDescriptorAllocations", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  refl::GlobalDef().def("tl.transform.ReuseLocalDescriptorAllocations",
                        ReuseLocalDescriptorAllocations);
}

} // namespace tl
} // namespace tvm
