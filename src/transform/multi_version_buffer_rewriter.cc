/*!
 * \file warp_specialized_pipeline.cc
 * \brief Warp specialized Pipeline for cuda GPU (sm90+)
 */

#include <tvm/arith/analyzer.h>
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
#include "../op/utils.h"

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
    ICHECK(it != map_.end())
        << " Cannot find role for stmt: " << stmt->GetTypeKey();
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
    if (!IsSharedBuffer(op->buffer)) {
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
      if (!IsGlobalBuffer(read->buffer)) {
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
  void VisitStmt_(const AllocateNode *op) final { HandleBodyStmt(op); }
  void VisitStmt_(const DeclBufferNode *op) final { HandleBodyStmt(op); }

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
  static PrimFunc Substitute(PrimFunc &f, bool barrier_only = false) {
    auto rewriter = MultiVersionBufferRewriter(barrier_only);
    rewriter.buffer_lca_ = DetectBufferAccessLCA(f);
    for (auto [buffer, _] : rewriter.buffer_lca_) {
      Var buffer_var = buffer->data;
      rewriter.buffer_data_to_buffer_.Set(buffer_var, buffer);
    }
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  explicit MultiVersionBufferRewriter(bool barrier_only = false)
      : barrier_only_(barrier_only) {}

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
        if (IsSharedBuffer(wr->buffer)) {
          has_shared_write = true;
          break;
        }
      }
      if (!has_shared_write)
        return false;
      for (const BufferRegion &rd : reads[idx]) {
        if (IsGlobalBuffer(rd->buffer)) {
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
    if (buffer.scope() == "shared.barrier") {
      // Barrier buffers: expand first dimension to keep 1D shape.
      // (1,) -> (num_versions,) so lower_shared_barrier.cc still works.
      new_buffer->shape.Set(0, PrimExpr(num_versions) * new_buffer->shape[0]);
    } else {
      new_buffer->shape.insert(new_buffer->shape.begin(),
                               PrimExpr(num_versions));
      if (!new_buffer->strides.empty()) {
        ICHECK(new_buffer->strides.size() + 1 == new_buffer->shape.size());
        PrimExpr stride_0 = new_buffer->strides[0] * new_buffer->shape[1];
        new_buffer->strides.insert(new_buffer->strides.begin(), stride_0);
      }
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

    // Update barrier_init annotation: replicate arrive counts for versioned
    // barrier buffers so lower_shared_barrier sees the correct count.
    if (block->annotations.count("barrier_init")) {
      auto barrier_init_map = Downcast<Map<Var, Array<PrimExpr>>>(
          block->annotations.Get("barrier_init").value());
      Map<Var, Array<PrimExpr>> new_init;
      bool changed = false;
      for (auto [data_var, counts] : barrier_init_map) {
        auto buf_it = buffer_data_to_buffer_.find(data_var);
        if (buf_it != buffer_data_to_buffer_.end()) {
          Buffer old_buf = (*buf_it).second;
          auto remap_it = buffer_remap_.find(old_buf);
          if (remap_it != buffer_remap_.end()) {
            Buffer new_buf = (*remap_it).second;
            int new_size =
                static_cast<int>(Downcast<IntImm>(new_buf->shape[0])->value);
            Array<PrimExpr> new_counts;
            new_counts.reserve(new_size);
            for (int v = 0; v < new_size;
                 v += static_cast<int>(counts.size())) {
              for (auto c : counts)
                new_counts.push_back(c);
            }
            new_init.Set(data_var, new_counts);
            changed = true;
            continue;
          }
        }
        new_init.Set(data_var, counts);
      }
      if (changed) {
        auto ann = block->annotations;
        ann.Set("barrier_init", new_init);
        block.CopyOnWrite()->annotations = std::move(ann);
      }
    }

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
      // Only double-buffer shared/barrier allocations; locals do not need
      // versioning.
      if (!IsSharedBuffer(buffer) && buffer.scope() != "shared.barrier")
        continue;
      if (seen.insert(buffer.get()).second) {
        scoped_buffers.push_back(buffer);
      }
    }
    for (auto it = stmt_stack_.rbegin(); it != stmt_stack_.rend(); ++it) {
      if (!(*it)->IsInstance<BlockNode>())
        continue;
      const auto *block = static_cast<const BlockNode *>(*it);
      // Try cached alloc list first; fall back to the original IR node
      // (the cache may not be populated yet during the recursive visit).
      auto map_it = block_alloc_buffers_.find(block);
      const Array<Buffer> &buffers = map_it != block_alloc_buffers_.end()
                                         ? map_it->second
                                         : block->alloc_buffers;
      for (const Buffer &buffer : buffers) {
        if (!IsSharedBuffer(buffer) && buffer.scope() != "shared.barrier")
          continue;
        if (seen.insert(buffer.get()).second) {
          scoped_buffers.push_back(buffer);
        }
      }
    }

    Array<Buffer> versioned_buffers =
        GetVersionedBuffers(pipeline_body_seq->seq, scoped_buffers);

    // Barrier buffers always get versioned in pipelined loops —
    // they don't fit the producer/consumer analysis above.
    {
      std::unordered_set<const BufferNode *> already;
      for (auto b : versioned_buffers)
        already.insert(b.get());
      for (auto buffer : scoped_buffers) {
        if (buffer.scope() == "shared.barrier" &&
            !already.count(buffer.get())) {
          versioned_buffers.push_back(buffer);
        }
      }
    }

    // In barrier_only mode, only version barrier buffers.
    // Data buffer versioning is left to InjectSoftwarePipeline.
    if (barrier_only_) {
      Array<Buffer> filtered;
      for (auto buffer : versioned_buffers) {
        if (buffer.scope() == "shared.barrier") {
          filtered.push_back(buffer);
        }
      }
      versioned_buffers = filtered;
    }

    for (auto buffer : versioned_buffers) {
      Var buffer_var = buffer->data;
      Buffer new_buffer = RewriteAllocBuffer(buffer, num_stages);
      buffer_remap_.Set(buffer, new_buffer);
      // Ensure the data var is discoverable so the barrier_init annotation
      // update in VisitStmt_(BlockRealizeNode*) can find the remapped buffer.
      if (!buffer_data_to_buffer_.count(buffer_var)) {
        buffer_data_to_buffer_.Set(buffer_var, buffer);
      }
    }
    PrimExpr linear_index = loop_stack_[0].first;
    for (size_t i = 1; i < loop_stack_.size(); ++i) {
      linear_index =
          linear_index * loop_stack_[i].second + loop_stack_[i].first;
    }
    version_index_ = FloorMod(linear_index, num_stages);
    // Parity cycles every num_stages iterations for mbarrier phase tracking.
    parity_cycle_ = FloorMod(FloorDiv(linear_index, num_stages), 2);
    // Store the pipelined loop variable and its min value so we can compute
    // the initial-phase offset of each mbarrier_wait_parity expression.
    pipeline_loop_var_ = op->loop_var;
    pipeline_loop_min_ = op->min;
    auto for_node = StmtExprMutator::VisitStmt_(op);
    parity_cycle_ = PrimExpr(); // reset
    pipeline_loop_var_ = Var();
    pipeline_loop_min_ = PrimExpr();
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
    Buffer old_buffer = load->buffer;
    const Buffer &new_buffer = (*it).second;
    auto *n = load.CopyOnWrite();
    n->buffer = new_buffer;
    if (old_buffer.scope() == "shared.barrier") {
      // Barrier: offset into expanded 1D array
      n->indices.Set(0, version_index_ * old_buffer->shape[0] + n->indices[0]);
    } else {
      n->indices.insert(n->indices.begin(), version_index_);
    }
    return std::move(load);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = buffer_remap_.find(store->buffer);
    if (it == buffer_remap_.end()) {
      return std::move(store);
    }
    Buffer old_buffer = store->buffer;
    const Buffer &new_buffer = (*it).second;
    auto *n = store.CopyOnWrite();
    n->buffer = new_buffer;
    if (old_buffer.scope() == "shared.barrier") {
      n->indices.Set(0, version_index_ * old_buffer->shape[0] + n->indices[0]);
    } else {
      n->indices.insert(n->indices.begin(), version_index_);
    }
    return std::move(store);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (call->op.same_as(builtin::tvm_access_ptr())) {
      return RewriteBufferAccess(call, {1});
    }
    // Rewrite parity for mbarrier_wait_parity on versioned barrier buffers.
    // The user writes single-barrier parity (e.g. k % 2 or (k+1) % 2).
    // After multi-versioning, each barrier is reused every num_stages
    // iterations, so the base parity becomes (k // num_stages) % 2.
    // However, different barriers may have different initial-phase offsets
    // (e.g. back-pressure barriers use (k+1)%2 so the first iteration
    // passes immediately). We detect this offset by evaluating the original
    // parity at the loop's initial value and preserving it.
    if (call->op.same_as(mbarrier_wait_parity()) && parity_cycle_.defined()) {
      if (auto load = call->args[0].as<BufferLoadNode>()) {
        if (load->buffer.scope() == "shared.barrier") {
          PrimExpr new_parity = parity_cycle_;
          if (pipeline_loop_var_.defined()) {
            arith::Analyzer analyzer;
            auto subst = [&](const Var &v) -> Optional<PrimExpr> {
              if (v.same_as(pipeline_loop_var_))
                return pipeline_loop_min_;
              return Optional<PrimExpr>();
            };
            PrimExpr init_orig =
                analyzer.Simplify(tir::Substitute(call->args[1], subst));
            PrimExpr init_cycle =
                analyzer.Simplify(tir::Substitute(parity_cycle_, subst));
            PrimExpr offset =
                analyzer.Simplify(FloorMod(init_orig - init_cycle, 2));
            if (auto *imm = offset.as<IntImmNode>()) {
              if (imm->value % 2 != 0) {
                new_parity = FloorMod(parity_cycle_ + 1, 2);
              }
            }
          }
          Array<PrimExpr> new_args = call->args;
          new_args.Set(1, new_parity);
          return Call(call->dtype, call->op, new_args, call->annotations);
        }
      }
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
    return Call(call->dtype, call->op, new_args, call->annotations, call->span);
  }

  bool barrier_only_;
  PrimExpr version_index_;
  PrimExpr parity_cycle_; // (k / num_stages) % 2 for mbarrier parity rewriting
  Var pipeline_loop_var_; // loop variable of the pipelined loop
  PrimExpr pipeline_loop_min_; // min value of the pipelined loop
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

tvm::transform::Pass MultiVersionBuffer(bool barrier_only) {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return MultiVersionBufferRewriter::Substitute(f, barrier_only);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MultiVersionBuffer", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MultiVersionBuffer", MultiVersionBuffer);
}

} // namespace tl
} // namespace tvm
