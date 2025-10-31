/*!
 * \file warp_specialized_pipeline.cc
 * \brief Warp specialized Pipeline for cuda GPU (sm90+)
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <functional>
#include <unordered_set>
#include <utility>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

enum class Role : uint8_t { kConsumer, kProducer, kBoth };

class WarpSpecializedRoleMarker_ : public StmtVisitor {
public:
  WarpSpecializedRoleMarker_(Map<Var, Buffer> buffer_data_to_buffer)
      : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)) {}

  Role GetRole(const StmtNode *stmt) const {
    auto it = map_.find(stmt);
    ICHECK(it != map_.end());
    return it->second;
  }

  Role GetRole(const Stmt &stmt) const { return GetRole(stmt.get()); }

  void VisitStmt_(const EvaluateNode *op) final {
    Role role = Role::kConsumer;
    if (auto call = op->value.as<CallNode>()) {
      if (call->op.same_as(tma_load()) || call->op.same_as(tma_load_im2col())) {
        role = Role::kProducer;
        has_bulk_copy_ = true;
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    bool is_shared_store =
        op->buffer.scope() == "shared.dyn" || op->buffer.scope() == "shared";
    if (!is_shared_store) {
      SetRole(op, Role::kConsumer);
      return;
    }

    // Check reads from global
    Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{}, /*name_hint=*/"",
                /*body*/ tvm::ffi::GetRef<Stmt>(op));
    auto access = GetBlockReadWriteRegion(block, buffer_data_to_buffer_);
    auto reads = access[0];
    Role role = Role::kProducer;
    for (auto read : reads) {
      if (read->buffer.scope() != "global") {
        role = Role::kConsumer;
        break;
      }
    }
    if (role == Role::kProducer)
      has_simt_copy_ = true;
    SetRole(op, role);
  }

  void VisitStmt_(const SeqStmtNode *op) final {
    StmtVisitor::VisitStmt_(op);
    auto role = GetRole(op->seq[0]);
    for (auto stmt : op->seq) {
      if (role != GetRole(stmt)) {
        role = Role::kBoth;
        break;
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    StmtVisitor::VisitStmt_(op);
    auto role = GetRole(op->then_case);
    if (op->else_case.defined()) {
      auto role_else = GetRole(op->else_case.value());
      if (role != role_else)
        role = Role::kBoth;
    }
    SetRole(op, role);
  }

  void VisitStmt_(const BlockRealizeNode *op) final {
    StmtVisitor::VisitStmt_(op);
    SetRole(op, GetRole(op->block));
  }

  template <class NodeType> void HandleBodyStmt(const NodeType *op) {
    StmtVisitor::VisitStmt_(op);
    SetRole(op, GetRole(op->body));
  }

  void VisitStmt_(const ForNode *op) final { HandleBodyStmt(op); }
  void VisitStmt_(const LetStmtNode *op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AttrStmtNode *op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AssertStmtNode *op) final { HandleBodyStmt(op); }
  void VisitStmt_(const BlockNode *op) final { HandleBodyStmt(op); }

  bool HasProducer() { return has_simt_copy_ || has_bulk_copy_; }

  bool HasSimtCopy() { return has_simt_copy_; }

private:
  void SetRole(const StmtNode *stmt, Role role) { map_[stmt] = role; }
  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_map<const StmtNode *, Role> map_;
  bool has_simt_copy_ = false;
  bool has_bulk_copy_ = false;
};

class MultiVersionBufferRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    auto rewriter = MultiVersionBufferRewriter();
    rewriter.buffer_lca_ = DetectBufferAccessLCA(f);
    for (auto [buffer, _] : rewriter.buffer_lca_) {
      Var buffer_var = buffer->data;
      rewriter.buffer_data_to_buffer_.Set(buffer_var, buffer);
    }
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  MultiVersionBufferRewriter() = default;

  Array<Buffer> GetVersionedBuffers(const Array<Stmt> &seq_stmt,
                                    const Array<Buffer> &scoped_buffers) {
    Array<Stmt> pipeline_stmts;
    std::function<void(const Stmt &)> collect_stmts = [&](const Stmt &stmt) {
      if (const auto *seq = stmt.as<SeqStmtNode>()) {
        for (const Stmt &s : seq->seq) {
          collect_stmts(s);
        }
        return;
      }
      if (const auto *let = stmt.as<LetStmtNode>()) {
        collect_stmts(let->body);
        return;
      }
      if (const auto *attr = stmt.as<AttrStmtNode>()) {
        collect_stmts(attr->body);
        return;
      }
      if (const auto *block_realize = stmt.as<BlockRealizeNode>()) {
        collect_stmts(block_realize->block->body);
        return;
      }
      if (const auto *block = stmt.as<BlockNode>()) {
        collect_stmts(block->body);
        return;
      }
      pipeline_stmts.push_back(stmt);
    };
    for (const Stmt &stmt : seq_stmt) {
      collect_stmts(stmt);
    }

    std::vector<Role> roles;
    Array<Array<BufferRegion>> reads, writes;
    auto marker = WarpSpecializedRoleMarker_(buffer_data_to_buffer_);
    for (const Stmt &stmt : pipeline_stmts) {
      marker(stmt);
      Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
                  /*name_hint=*/"", /*body*/ stmt);
      auto access = GetBlockAccessRegion(block, buffer_data_to_buffer_);
      reads.push_back(access[0]);
      writes.push_back(access[1]);
      roles.push_back(marker.GetRole(stmt));
    }

    std::unordered_set<const BufferNode *> consumer_used, producer_used;
    std::unordered_map<const BufferNode *, size_t> first_write_index;
    std::unordered_map<const BufferNode *, size_t> last_read_index;
    auto is_copy_stage = [&](size_t idx) {
      bool has_shared_write = false;
      for (const BufferRegion &wr : writes[idx]) {
        auto scope = wr->buffer.scope();
        if (scope == "shared" || scope == "shared.dyn") {
          has_shared_write = true;
          break;
        }
      }
      if (!has_shared_write)
        return false;
      for (const BufferRegion &rd : reads[idx]) {
        if (rd->buffer.scope() == "global") {
          return true;
        }
      }
      return false;
    };
    for (size_t i = 0; i < pipeline_stmts.size(); i++) {
      bool copy_stage = is_copy_stage(i);
      bool is_producer = roles[i] == Role::kProducer ||
                         (roles[i] == Role::kBoth && copy_stage);
      bool is_consumer = roles[i] == Role::kConsumer ||
                         (roles[i] == Role::kBoth && !copy_stage);
      if (is_producer) {
        for (BufferRegion br : writes[i]) {
          producer_used.insert(br->buffer.get());
        }
      }
      if (is_consumer) {
        for (BufferRegion br : reads[i]) {
          consumer_used.insert(br->buffer.get());
        }
      }
      for (BufferRegion br : writes[i]) {
        const BufferNode *buf = br->buffer.get();
        if (!first_write_index.count(buf)) {
          first_write_index[buf] = i;
        }
      }
      for (BufferRegion br : reads[i]) {
        last_read_index[br->buffer.get()] = i;
      }
    }
    Array<Buffer> versioned_buffers;
    for (Buffer buffer : scoped_buffers) {
      if (consumer_used.count(buffer.get()) &&
          producer_used.count(buffer.get())) {
        versioned_buffers.push_back(buffer);
        continue;
      }
      // Fallback: if we saw a write before a later read, the buffer spans
      // multiple stages even if role classification missed one side.
      auto it_w = first_write_index.find(buffer.get());
      auto it_r = last_read_index.find(buffer.get());
      if (it_w != first_write_index.end() && it_r != last_read_index.end() &&
          it_w->second < it_r->second) {
        if (!is_copy_stage(it_w->second))
          continue;
        versioned_buffers.push_back(buffer);
      }
    }
    return versioned_buffers;
  }

  static Buffer RewriteAllocBuffer(const Buffer &buffer, int num_versions) {
    ObjectPtr<BufferNode> new_buffer =
        tvm::ffi::make_object<BufferNode>(*(buffer.get()));
    new_buffer->shape.insert(new_buffer->shape.begin(), PrimExpr(num_versions));
    if (!new_buffer->strides.empty()) {
      ICHECK(new_buffer->strides.size() + 1 == new_buffer->shape.size());
      PrimExpr stride_0 = new_buffer->strides[0] * new_buffer->shape[1];
      new_buffer->strides.insert(new_buffer->strides.begin(), stride_0);
    }
    return Buffer(new_buffer);
  }

  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    BlockRealize block_realize =
        Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(op));
    Block block = block_realize->block;
    Array<Buffer> alloc_buffers;
    for (auto buffer : block->alloc_buffers) {
      if (buffer_remap_.count(buffer)) {
        Buffer new_buffer = buffer_remap_[buffer];
        alloc_buffers.push_back(new_buffer);
      } else {
        alloc_buffers.push_back(buffer);
      }
    }
    block.CopyOnWrite()->alloc_buffers = std::move(alloc_buffers);
    // Record the updated alloc list to recover buffers whose LCA is the block.
    block_alloc_buffers_[op->block.get()] = block->alloc_buffers;
    block_realize.CopyOnWrite()->block = block;
    return block_realize;
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    stmt_stack_.push_back(op);
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    stmt_stack_.pop_back();
    return stmt;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    stmt_stack_.push_back(op);
    loop_stack_.emplace_back(op->loop_var, op->extent);
    auto num_stages_anno = op->annotations.Get("num_stages");
    if (!num_stages_anno) {
      auto for_node = StmtExprMutator::VisitStmt_(op);
      loop_stack_.pop_back();
      stmt_stack_.pop_back();
      return for_node;
    }

    ICHECK(num_stages_anno->as<IntImmNode>());
    int num_stages = static_cast<int>(num_stages_anno->as<IntImmNode>()->value);

    Stmt pipeline_body_root{nullptr};
    if (const auto *realize = op->body.as<BlockRealizeNode>()) {
      const auto &block = realize->block;
      for (const auto &buffer : block->alloc_buffers) {
        ICHECK(buffer->IsInstance<BufferNode>());
        buffer_data_to_buffer_.Set(buffer->data, buffer);
      }
      pipeline_body_root = block->body;
    } else {
      pipeline_body_root = op->body;
    }

    const SeqStmtNode *pipeline_body_seq = nullptr;
    {
      // Traverse trivial wrappers (let/if) to find the actual SeqStmt body.
      Stmt current = pipeline_body_root;
      while (true) {
        if (const auto *seq_stmt = current.as<SeqStmtNode>()) {
          pipeline_body_seq = seq_stmt;
          break;
        }
        if (const auto *if_then_else = current.as<IfThenElseNode>()) {
          ICHECK(!if_then_else->else_case.defined())
              << "MultiVersionBuffer: Can't handle the body of the loop "
                 "because the IfThenElse node has an else branch";
          current = if_then_else->then_case;
          continue;
        }
        if (const auto *let_stmt = current.as<LetStmtNode>()) {
          current = let_stmt->body;
          continue;
        }
        LOG(FATAL)
            << "MultiVersionBuffer: Can't handle the body of the loop because "
            << "it is not a SeqStmt, IfThenElse without else, "
            << "or LetStmt wrapping them, but got " << current->GetTypeKey();
      }
    }
    ICHECK(pipeline_body_seq != nullptr);

    Array<Buffer> scoped_buffers;
    std::unordered_set<const BufferNode *> seen;
    for (auto [buffer, stmt] : buffer_lca_) {
      if (!stmt.defined())
        continue;
      const StmtNode *lca = stmt.value().get();
      bool in_scope = false;
      for (const StmtNode *ancestor : stmt_stack_) {
        if (ancestor == lca) {
          in_scope = true;
          break;
        }
      }
      if (!in_scope)
        continue;
      // Only double-buffer shared allocations; locals do not need versioning.
      auto scope = buffer.scope();
      if (!(scope == "shared" || scope == "shared.dyn"))
        continue;
      if (seen.insert(buffer.get()).second) {
        scoped_buffers.push_back(buffer);
      }
    }
    for (auto it = stmt_stack_.rbegin(); it != stmt_stack_.rend(); ++it) {
      if (!(*it)->IsInstance<BlockNode>())
        continue;
      const auto *block = static_cast<const BlockNode *>(*it);
      auto map_it = block_alloc_buffers_.find(block);
      if (map_it == block_alloc_buffers_.end())
        continue;
      for (const Buffer &buffer : map_it->second) {
        auto scope = buffer.scope();
        if (!(scope == "shared" || scope == "shared.dyn"))
          continue;
        if (seen.insert(buffer.get()).second) {
          scoped_buffers.push_back(buffer);
        }
      }
    }

    Array<Buffer> versioned_buffers =
        GetVersionedBuffers(pipeline_body_seq->seq, scoped_buffers);

    for (auto buffer : versioned_buffers) {
      Var buffer_var = buffer->data;
      Buffer new_buffer = RewriteAllocBuffer(buffer, num_stages);
      buffer_remap_.Set(buffer, new_buffer);
    }
    PrimExpr linear_index = loop_stack_[0].first;
    for (size_t i = 1; i < loop_stack_.size(); ++i) {
      linear_index =
          linear_index * loop_stack_[i].second + loop_stack_[i].first;
    }
    version_index_ = FloorMod(linear_index, num_stages);
    auto for_node = StmtExprMutator::VisitStmt_(op);
    loop_stack_.pop_back();
    stmt_stack_.pop_back();

    return for_node;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = buffer_remap_.find(load->buffer);
    if (it == buffer_remap_.end()) {
      return std::move(load);
    }
    const Buffer &new_buffer = (*it).second;
    auto *n = load.CopyOnWrite();
    n->buffer = new_buffer;
    n->indices.insert(n->indices.begin(), version_index_);
    return std::move(load);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = buffer_remap_.find(store->buffer);
    if (it == buffer_remap_.end()) {
      return std::move(store);
    }
    const Buffer &new_buffer = (*it).second;
    auto *n = store.CopyOnWrite();
    n->buffer = new_buffer;
    n->indices.insert(n->indices.begin(), version_index_);
    return std::move(store);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (call->op.same_as(builtin::tvm_access_ptr())) {
      return RewriteBufferAccess(call, {1});
    }
    return call;
  }

  PrimExpr RewriteBufferAccess(const Call &call,
                               const std::vector<int> &arg_indices) {
    auto product = [](const Array<PrimExpr> &input) {
      return foldl(
          [](PrimExpr a, PrimExpr b, Span span) {
            return mul(std::move(a), std::move(b), std::move(span));
          },
          make_const(DataType::Int(32), 1), input);
    };
    Array<PrimExpr> new_args = call->args;
    for (int i : arg_indices) {
      auto buffer_var = Downcast<Var>(call->args[i]);
      if (!buffer_data_to_buffer_.count(buffer_var))
        continue;
      const Buffer &buffer = buffer_data_to_buffer_[buffer_var];
      auto it = buffer_remap_.find(buffer);
      if (it != buffer_remap_.end()) {
        const Buffer &new_buffer = (*it).second;
        const PrimExpr &old_index = call->args[i + 1];
        PrimExpr offset;
        if (new_buffer->strides.empty()) {
          offset = product(buffer->shape);
        } else {
          offset = new_buffer->strides[0];
        }
        PrimExpr new_index = old_index + version_index_ * offset;
        new_args.Set(i + 1, new_index);
      }
    }
    return Call(call->dtype, call->op, new_args, call->span);
  }

  PrimExpr version_index_;
  std::vector<std::pair<Var, PrimExpr>> loop_stack_;
  // Track ancestor statements to query whether an LCA is inside the current
  // loop.
  std::vector<const StmtNode *> stmt_stack_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Optional<Stmt>> buffer_lca_;
  Map<Buffer, Buffer> buffer_remap_;
  // Remember each block's alloc list so the loop can see buffers defined in
  // parents.
  std::unordered_map<const BlockNode *, Array<Buffer>> block_alloc_buffers_;
};

using namespace tir::transform;

tvm::transform::Pass MultiVersionBuffer() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return MultiVersionBufferRewriter::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MultiVersionBuffer", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MultiVersionBuffer", MultiVersionBuffer);
}

} // namespace tl
} // namespace tvm
