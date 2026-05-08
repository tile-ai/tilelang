/*!
 * \file warp_specialized_pipeline.cc
 * \brief Warp specialized Pipeline for cuda GPU (sm90+)
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../layout/layout.h"
#include "../op/builtin.h"
#include "../op/operator.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "common/pipeline_utils.h"
#include "multi_version_buffer_rewriter.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

bool ShapesEqual(const Array<PrimExpr> &lhs, const Array<PrimExpr> &rhs,
                 arith::Analyzer *analyzer) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!analyzer->CanProveEqual(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

Layout ExpandAnnotatedLayoutForMultiVersionedBuffer(const Layout &layout,
                                                    const Buffer &old_buffer,
                                                    const Buffer &new_buffer) {
  if (!layout.defined() ||
      new_buffer->shape.size() <= old_buffer->shape.size()) {
    return Layout();
  }

  arith::Analyzer analyzer;
  if (!ShapesEqual(layout->InputShape(), old_buffer->shape, &analyzer)) {
    return Layout();
  }

  size_t leading_ndim = new_buffer->shape.size() - old_buffer->shape.size();
  Array<PrimExpr> trailing_shape;
  Array<PrimExpr> leading_shape;
  for (size_t i = 0; i < leading_ndim; ++i) {
    leading_shape.push_back(new_buffer->shape[i]);
  }
  for (size_t i = 0; i < old_buffer->shape.size(); ++i) {
    trailing_shape.push_back(new_buffer->shape[leading_ndim + i]);
  }
  if (!ShapesEqual(trailing_shape, old_buffer->shape, &analyzer)) {
    return Layout();
  }

  return layout->Expand(leading_shape);
}

bool UpdateExpandedLayoutMapForRemappedAllocs(
    const std::vector<std::pair<Buffer, Buffer>> &remapped_allocs,
    Map<String, Any> *annotations) {
  if (remapped_allocs.empty() || !annotations->count(attr::kLayoutMap)) {
    return false;
  }

  auto layout_map_ref = annotations->Get(attr::kLayoutMap);
  if (!layout_map_ref.has_value()) {
    return false;
  }
  auto layout_map = layout_map_ref.value().as<Map<Var, Layout>>();
  if (!layout_map.has_value()) {
    return false;
  }

  Map<Var, Layout> updated_layout_map = layout_map.value();
  std::unordered_set<const VarNode *> visited;
  bool changed = false;
  for (const auto &[old_buffer, new_buffer] : remapped_allocs) {
    if (!visited.insert(old_buffer->data.get()).second ||
        !updated_layout_map.count(old_buffer->data)) {
      continue;
    }
    Layout layout = updated_layout_map[old_buffer->data];
    Layout expanded = ExpandAnnotatedLayoutForMultiVersionedBuffer(
        layout, old_buffer, new_buffer);
    if (!expanded.defined()) {
      continue;
    }
    updated_layout_map.Set(old_buffer->data, expanded);
    changed = true;
  }

  if (changed) {
    annotations->Set(attr::kLayoutMap, updated_layout_map);
  }
  return changed;
}

} // namespace

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
  static PrimFunc Substitute(PrimFunc f) {
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
  explicit MultiVersionBufferRewriter() = default;

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
      if (const auto *if_then_else = stmt.as<IfThenElseNode>()) {
        collect_stmts(if_then_else->then_case);
        if (if_then_else->else_case.defined()) {
          collect_stmts(if_then_else->else_case.value());
        }
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
      Array<BufferRegion> stmt_reads = access[0];
      Array<BufferRegion> stmt_writes = access[1];

      // Supplement with tile-op analysis.
      // GetBlockAccessRegion misses buffer references that are encoded as
      // tl.tileop.region Call args or as plain BufferLoad args whose
      // semantic role (read vs write) is only known to the tile-op.
      // Let the tile-op report its own access regions, and fall back to
      // RegionOp scanning for any ops that still do not expose them.
      if (auto *eval = stmt.as<EvaluateNode>()) {
        if (auto *call = eval->value.as<CallNode>()) {
          auto tile_op = ParseOperator(ffi::GetRef<Call>(call));
          if (tile_op.defined()) {
            AccessRegions access = tile_op->GetAccessRegions();
            if (!access.reads.empty() || !access.writes.empty()) {
              stmt_reads.insert(stmt_reads.end(), access.reads.begin(),
                                access.reads.end());
              stmt_writes.insert(stmt_writes.end(), access.writes.begin(),
                                 access.writes.end());
            } else {
              // Fallback: scan RegionOp-encoded args.
              for (const auto &arg : call->args) {
                if (auto *region_call = arg.as<CallNode>()) {
                  if (region_call->op.same_as(RegionOp::Get())) {
                    auto region_op =
                        ParseOperator(ffi::GetRef<Call>(region_call));
                    if (auto *rn = region_op.as<RegionOpNode>()) {
                      int mask = rn->GetAccessMask();
                      auto br = BufferRegion(rn->GetBuffer(), rn->GetRanges());
                      if (mask & 1) { // read
                        stmt_reads.push_back(br);
                      }
                      if (mask & 2) { // write
                        stmt_writes.push_back(br);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      reads.push_back(stmt_reads);
      writes.push_back(stmt_writes);
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

  Array<Stmt> GetPipelineTopLevelStmts(const Stmt &pipeline_body) const {
    Stmt current = pipeline_body;
    while (true) {
      if (const auto *realize = current.as<BlockRealizeNode>()) {
        current = realize->block->body;
        continue;
      }
      if (const auto *block = current.as<BlockNode>()) {
        current = block->body;
        continue;
      }
      break;
    }
    if (const auto *seq = current.as<SeqStmtNode>()) {
      return seq->seq;
    }
    return {current};
  }

  Array<Buffer> CollectScopedBuffers() const {
    Array<Buffer> scoped_buffers;
    std::unordered_set<const BufferNode *> seen;
    for (auto [buffer, stmt] : buffer_lca_) {
      if (!stmt.defined()) {
        continue;
      }
      const StmtNode *lca = stmt.value().get();
      bool in_scope = false;
      for (const StmtNode *ancestor : stmt_stack_) {
        if (ancestor == lca) {
          in_scope = true;
          break;
        }
      }
      if (!in_scope) {
        continue;
      }
      if (!IsSharedBuffer(buffer) && buffer.scope() != "shared.barrier") {
        continue;
      }
      if (seen.insert(buffer.get()).second) {
        scoped_buffers.push_back(buffer);
      }
    }
    for (auto it = stmt_stack_.rbegin(); it != stmt_stack_.rend(); ++it) {
      if (!(*it)->IsInstance<BlockNode>()) {
        continue;
      }
      const auto *block = static_cast<const BlockNode *>(*it);
      auto map_it = block_alloc_buffers_.find(block);
      const Array<Buffer> &buffers = map_it != block_alloc_buffers_.end()
                                         ? map_it->second
                                         : block->alloc_buffers;
      for (const Buffer &buffer : buffers) {
        if (!IsSharedBuffer(buffer) && buffer.scope() != "shared.barrier") {
          continue;
        }
        if (seen.insert(buffer.get()).second) {
          scoped_buffers.push_back(buffer);
        }
      }
    }
    return scoped_buffers;
  }

  Array<Buffer> SelectVersionedBuffers(const Stmt &pipeline_body,
                                       int num_stages) {
    Array<Buffer> scoped_buffers = CollectScopedBuffers();
    Array<Buffer> versioned_buffers = GetVersionedBuffers(
        GetPipelineTopLevelStmts(pipeline_body), scoped_buffers);

    std::unordered_set<const BufferNode *> already;
    for (const Buffer &buffer : versioned_buffers) {
      already.insert(buffer.get());
    }
    for (const Buffer &buffer : scoped_buffers) {
      if (buffer.scope() == "shared.barrier" && !already.count(buffer.get())) {
        versioned_buffers.push_back(buffer);
      }
    }

    if (num_stages <= 1) {
      Array<Buffer> filtered;
      for (const Buffer &buffer : versioned_buffers) {
        if (buffer.scope() == "shared.barrier") {
          filtered.push_back(buffer);
        }
      }
      versioned_buffers = filtered;
    }
    return versioned_buffers;
  }

  void EnsureVersionedBuffers(const Array<Buffer> &versioned_buffers,
                              int num_stages) {
    for (const Buffer &buffer : versioned_buffers) {
      if (buffer_remap_.count(buffer)) {
        continue;
      }
      Var buffer_var = buffer->data;
      Buffer new_buffer = RewriteAllocBuffer(buffer, num_stages);
      buffer_remap_.Set(buffer, new_buffer);
      if (!buffer_data_to_buffer_.count(buffer_var)) {
        buffer_data_to_buffer_.Set(buffer_var, buffer);
      }
    }
  }

  PrimExpr CurrentVersionIndex() const {
    if (!explicit_version_index_stack_.empty()) {
      return explicit_version_index_stack_.back();
    }
    return version_index_;
  }

  PrimExpr CurrentParityCycle() const {
    if (!explicit_parity_cycle_stack_.empty()) {
      return explicit_parity_cycle_stack_.back();
    }
    return parity_cycle_;
  }

  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    BlockRealize block_realize =
        Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(op));
    Block block = block_realize->block;
    Array<Buffer> alloc_buffers;
    std::vector<std::pair<Buffer, Buffer>> remapped_allocs;
    for (auto buffer : block->alloc_buffers) {
      if (buffer_remap_.count(buffer)) {
        Buffer new_buffer = buffer_remap_[buffer];
        alloc_buffers.push_back(new_buffer);
        remapped_allocs.emplace_back(buffer, new_buffer);
      } else {
        alloc_buffers.push_back(buffer);
      }
    }
    block.CopyOnWrite()->alloc_buffers = std::move(alloc_buffers);

    if (!remapped_allocs.empty()) {
      auto ann = block->annotations;
      if (UpdateExpandedLayoutMapForRemappedAllocs(remapped_allocs, &ann)) {
        block.CopyOnWrite()->annotations = std::move(ann);
      }
    }

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

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    stmt_stack_.push_back(op);

    bool pushed_explicit_version = false;
    bool pushed_explicit_parity = false;
    if (op->attr_key == kPipelineMVBContextNumStages) {
      if (const int64_t *imm = as_const_int(op->value)) {
        int num_stages = static_cast<int>(*imm);
        EnsureVersionedBuffers(SelectVersionedBuffers(op->body, num_stages),
                               num_stages);
      }
    } else if (op->attr_key == kPipelineMVBStageExpr) {
      explicit_version_index_stack_.push_back(op->value);
      pushed_explicit_version = true;
    } else if (op->attr_key == kPipelineMVBParityExpr) {
      explicit_parity_cycle_stack_.push_back(op->value);
      pushed_explicit_parity = true;
    }

    Stmt body = this->VisitStmt(op->body);

    if (pushed_explicit_version) {
      explicit_version_index_stack_.pop_back();
    }
    if (pushed_explicit_parity) {
      explicit_parity_cycle_stack_.pop_back();
    }
    stmt_stack_.pop_back();

    if (op->attr_key == kPipelineMVBStageExpr ||
        op->attr_key == kPipelineMVBParityExpr ||
        op->attr_key == kPipelineMVBContextNumStages) {
      return body;
    }
    return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    stmt_stack_.push_back(op);
    loop_stack_.emplace_back(op->loop_var, op->extent);
    Optional<Integer> num_stages_anno = GetPipelineNumStages(op);
    if (!num_stages_anno) {
      auto for_node = StmtExprMutator::VisitStmt_(op);
      loop_stack_.pop_back();
      stmt_stack_.pop_back();
      return for_node;
    }

    int num_stages = num_stages_anno.value()->value;
    EnsureVersionedBuffers(SelectVersionedBuffers(op->body, num_stages),
                           num_stages);

    PrimExpr linear_index = loop_stack_[0].first;
    for (size_t i = 1; i < loop_stack_.size(); ++i) {
      linear_index =
          linear_index * loop_stack_[i].second + loop_stack_[i].first;
    }
    PrimExpr old_version_index = version_index_;
    PrimExpr old_parity_cycle = parity_cycle_;
    Var old_pipeline_loop_var = pipeline_loop_var_;
    PrimExpr old_pipeline_loop_min = pipeline_loop_min_;
    version_index_ = FloorMod(linear_index, num_stages);
    // Parity cycles every num_stages iterations for mbarrier phase tracking.
    parity_cycle_ = FloorMod(FloorDiv(linear_index, num_stages), 2);
    // Store the pipelined loop variable and its min value so we can compute
    // the initial-phase offset of each mbarrier_wait_parity expression.
    pipeline_loop_var_ = op->loop_var;
    pipeline_loop_min_ = op->min;
    auto for_node = StmtExprMutator::VisitStmt_(op);
    version_index_ = old_version_index;
    parity_cycle_ = old_parity_cycle;
    pipeline_loop_var_ = old_pipeline_loop_var;
    pipeline_loop_min_ = old_pipeline_loop_min;
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
    PrimExpr version_index = CurrentVersionIndex();
    ICHECK(version_index.defined())
        << "Versioned buffer load escaped pipeline stage context";
    auto *n = load.CopyOnWrite();
    n->buffer = new_buffer;
    if (old_buffer.scope() == "shared.barrier") {
      // Barrier: offset into expanded 1D array
      n->indices.Set(0, version_index * old_buffer->shape[0] + n->indices[0]);
    } else {
      n->indices.insert(n->indices.begin(), version_index);
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
    PrimExpr version_index = CurrentVersionIndex();
    ICHECK(version_index.defined())
        << "Versioned buffer store escaped pipeline stage context";
    auto *n = store.CopyOnWrite();
    n->buffer = new_buffer;
    if (old_buffer.scope() == "shared.barrier") {
      n->indices.Set(0, version_index * old_buffer->shape[0] + n->indices[0]);
    } else {
      n->indices.insert(n->indices.begin(), version_index);
    }
    return std::move(store);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (call->op.same_as(builtin::tvm_access_ptr())) {
      return RewriteBufferAccess(call, {1});
    }
    // Rewrite tl.tileop.region Calls for versioned buffers.
    // The region encoding is:
    //   region(BufferLoad(buf, [min_0, ..., min_N]), access_mask, ext_0, ...,
    //   ext_N)
    // After the recursive visit, VisitExpr_(BufferLoadNode*) prepends a
    // version_index to the BufferLoad indices, yielding [version_index,
    // min_0, ..., min_N].  We must also insert a matching extent (1) for the
    // new leading dimension so that RegionOp's ndim == indices.size()
    // invariant is preserved.
    //
    // Detection: if the BufferLoad has more indices than the number of extent
    // args (args.size() - 2), a version index was prepended.
    if (call->op.same_as(RegionOp::Get()) && call->args.size() >= 2) {
      if (auto load = call->args[0].as<BufferLoadNode>()) {
        size_t num_extents =
            call->args.size() - 2; // args = [load, mask, ext...]
        if (load->indices.size() == num_extents + 1) {
          // Version index was prepended.  Insert a unit extent to match.
          Array<PrimExpr> new_args;
          new_args.push_back(call->args[0]); // rewritten BufferLoad
          new_args.push_back(call->args[1]); // access_mask
          new_args.push_back(IntImm(DataType::Int(32), 1)); // stage extent
          for (size_t i = 2; i < call->args.size(); ++i) {
            new_args.push_back(call->args[i]);
          }
          return Call(call->dtype, call->op, new_args, call->annotations);
        }
      }
    }
    // Rewrite parity for mbarrier_wait_parity on versioned barrier buffers.
    // The user writes single-barrier parity (e.g. k % 2 or (k+1) % 2).
    // After multi-versioning, each barrier is reused every num_stages
    // iterations, so the base parity becomes (k // num_stages) % 2.
    // However, different barriers may have different initial-phase offsets
    // (e.g. back-pressure barriers use (k+1)%2 so the first iteration
    // passes immediately). We detect this offset by evaluating the original
    // parity at the loop's initial value and preserving it.
    PrimExpr parity_cycle = CurrentParityCycle();
    if (call->op.same_as(mbarrier_wait_parity()) && parity_cycle.defined()) {
      if (auto load = call->args[0].as<BufferLoadNode>()) {
        if (load->buffer.scope() == "shared.barrier") {
          PrimExpr new_parity = parity_cycle;
          arith::Analyzer analyzer;
          PrimExpr init_orig = call->args[1];
          PrimExpr init_cycle = parity_cycle;
          if (!explicit_parity_cycle_stack_.empty()) {
            PrimExpr version_index = CurrentVersionIndex();
            ICHECK(version_index.defined())
                << "Explicit parity rewrite requires a version index";
            init_cycle = version_index;
          }
          if (pipeline_loop_var_.defined()) {
            auto subst = [&](const Var &v) -> Optional<PrimExpr> {
              if (v.same_as(pipeline_loop_var_))
                return pipeline_loop_min_;
              return Optional<PrimExpr>();
            };
            init_orig = analyzer.Simplify(tir::Substitute(init_orig, subst));
            init_cycle = analyzer.Simplify(tir::Substitute(init_cycle, subst));
          }
          PrimExpr offset =
              analyzer.Simplify(FloorMod(init_orig - init_cycle, 2));
          if (const int64_t *imm = as_const_int(offset)) {
            if (*imm % 2 != 0) {
              new_parity = FloorMod(parity_cycle + 1, 2);
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
        PrimExpr version_index = CurrentVersionIndex();
        ICHECK(version_index.defined())
            << "Versioned access_ptr escaped pipeline stage context";
        const Buffer &new_buffer = (*it).second;
        const PrimExpr &old_index = call->args[i + 1];
        PrimExpr offset;
        if (new_buffer->strides.empty()) {
          offset = product(buffer->shape);
        } else {
          offset = new_buffer->strides[0];
        }
        PrimExpr new_index = old_index + version_index * offset;
        new_args.Set(i + 1, new_index);
      }
    }
    return Call(call->dtype, call->op, new_args, call->annotations, call->span);
  }

  PrimExpr version_index_;
  PrimExpr parity_cycle_; // (k / num_stages) % 2 for mbarrier parity rewriting
  Var pipeline_loop_var_; // loop variable of the pipelined loop
  PrimExpr pipeline_loop_min_; // min value of the pipelined loop
  std::vector<std::pair<Var, PrimExpr>> loop_stack_;
  // Track ancestor statements to query whether an LCA is inside the current
  // loop.
  std::vector<const StmtNode *> stmt_stack_;
  std::vector<PrimExpr> explicit_version_index_stack_;
  std::vector<PrimExpr> explicit_parity_cycle_stack_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Optional<Stmt>> buffer_lca_;
  Map<Buffer, Buffer> buffer_remap_;
  // Remember each block's alloc list so the loop can see buffers defined in
  // parents.
  std::unordered_map<const BlockNode *, Array<Buffer>> block_alloc_buffers_;
};

PrimFunc ApplyMultiVersionBufferRewriter(PrimFunc f) {
  return MultiVersionBufferRewriter::Substitute(std::move(f));
}

} // namespace tl
} // namespace tvm
