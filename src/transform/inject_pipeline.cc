/*!
 * \file inject_pipeline.cc
 * \brief Transform annotated loops into pipelined one that parallelize
 * producers and consumers.
 */
#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/pipeline_utils.h"
#include "pipeline/barrier.h"
#include "pipeline/helpers.h"
#include "pipeline/rewriter.h"
#include "tir/schedule/utils.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {
using namespace tirx;
using namespace ffi;
using tirx::GetSBlockReadWriteRegion;
namespace software_pipeline {

class PipelineInjector : private StmtExprMutator {
public:
  static Stmt Inject(const PrimFunc &func) {
    auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    PipelineInjector injector(global_symbol, target);
    for (const auto &kv : func->buffer_map) {
      const Buffer &buffer = kv.second;
      injector.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    return injector(func->body);
  }

private:
  explicit PipelineInjector(Optional<String> global_symbol,
                            Optional<Target> target)
      : global_symbol_(std::move(global_symbol)), target_(std::move(target)) {}

  /*!
   * \brief Check the pipeline satisfies the following conditions:
   * 1. No conflicting order: The order of each statement should be unique.
   * 2. Reordering of statements doesn't break buffer access dependencies.
   * Specifically, for dependency (e.g. read-after-write) from statement A to
   * statement B, it requires: case 1: stage(A) < stage(B) case 2: stage(A) ==
   * stage(B) and order(A) < order(B)
   */
  void ValidatePipelineBody(const PipelineInfo &pipeline_info,
                            const Array<SBlock> &original_order) {
    std::unordered_set<int> used_orders;
    for (const SBlock &block : original_order) {
      const auto &stmt_info = pipeline_info.at(block);
      int order = stmt_info.order;
      ICHECK(!used_orders.count(order))
          << "ValueError: Two statements in the software pipeline cannot have "
             "the same order";
      used_orders.insert(order);
    }

    std::unordered_map<SBlock, Array<SBlock>, ObjectPtrHash, ObjectPtrEqual>
        dep_src2dst;
    BuildDependencyGraph(original_order, &dep_src2dst, nullptr);

    for (const auto &pair : dep_src2dst) {
      const SBlock &src = pair.first;
      const auto &src_info = pipeline_info.at(src);
      const Array<SBlock> &dsts = pair.second;
      for (const SBlock &dst : dsts) {
        const auto &dst_info = pipeline_info.at(dst);
        ICHECK_LE(src_info.stage, dst_info.stage)
            << "ValueError: statement " << dst << " in stage " << dst_info.stage
            << " cannot depends on statement " << src << " in a later stage "
            << src_info.stage;
        if (src_info.stage == dst_info.stage) {
          ICHECK_LT(src_info.order, dst_info.order)
              << "ValueError: two statements with buffer "
                 "access dependency in the same stage of the "
                 "software pipeline cannot be reordered";
        }
      }
    }
  }

  void ValidateScheduledBindDependencies(const PipelineInfo &pipeline_info,
                                         const Array<SBlock> &scheduled_order) {
    std::unordered_map<Var, SBlock, ObjectPtrHash, ObjectPtrEqual>
        bind_producers;
    for (const SBlock &block : scheduled_order) {
      if (const auto *bind = block->body.as<BindNode>()) {
        bind_producers.emplace(bind->var, block);
      }
    }
    if (bind_producers.empty()) {
      return;
    }

    for (const SBlock &consumer : scheduled_order) {
      Array<Var> undefined_vars = UndefinedVars(consumer->body, Array<Var>{});
      for (const Var &var : undefined_vars) {
        auto it = bind_producers.find(var);
        if (it == bind_producers.end() || it->second.same_as(consumer)) {
          continue;
        }

        const PipelineAnnotation &producer_info = pipeline_info.at(it->second);
        const PipelineAnnotation &consumer_info = pipeline_info.at(consumer);
        ICHECK_EQ(producer_info.stage, consumer_info.stage)
            << "ValueError: scheduled scalar Bind '" << var
            << "' is used from a different pipeline stage. Scalar Bind "
               "statements that cannot be replayed must be scheduled in the "
               "same stage as their consumers.";
        ICHECK_LT(producer_info.order, consumer_info.order)
            << "ValueError: scheduled scalar Bind '" << var
            << "' must be ordered before every consumer in the same pipeline "
               "stage.";
      }
    }
  }

  bool HasOverlappableStages(const PipelineInfo &pipeline_info) const {
    std::optional<int> first_stage;
    for (const auto &pair : pipeline_info) {
      int stage = pair.second.stage;
      if (!first_stage.has_value()) {
        first_stage = stage;
      } else if (stage != first_stage.value()) {
        return true;
      }
    }
    return false;
  }

  struct PipelineScheduleUnit {
    SBlock block;
    Array<Buffer> nested_local_allocs;
  };

  struct PipelineSchedule {
    Array<SBlock> original_order;
    Array<Buffer> nested_local_allocs;
  };

  PipelineScheduleUnit MakePipelineScheduleUnit(const Stmt &stmt) {
    PipelineScheduleUnit unit;
    if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
      if (is_one(realize->predicate) &&
          realize->block->body->IsInstance<SeqStmtNode>()) {
        const SBlock &nested_block = realize->block;
        ICHECK(nested_block->match_buffers.empty())
            << "match_buffer should have been lowered before "
               "InjectSoftwarePipeline";
        for (const Buffer &buffer : nested_block->alloc_buffers) {
          buffer_data_to_buffer_.Set(buffer->data, buffer);
          allocated_buffers_.insert(buffer);
          unit.nested_local_allocs.push_back(buffer);
        }
      }
    }
    unit.block = MakeBlock(stmt, buffer_data_to_buffer_);
    return unit;
  }

  PipelineSchedule BuildPipelineSchedule(const Array<Stmt> &stmts) {
    PipelineSchedule schedule;
    for (const Stmt &stmt : stmts) {
      PipelineScheduleUnit unit = MakePipelineScheduleUnit(stmt);
      schedule.original_order.push_back(unit.block);
      schedule.nested_local_allocs.insert(schedule.nested_local_allocs.end(),
                                          unit.nested_local_allocs.begin(),
                                          unit.nested_local_allocs.end());
    }
    return schedule;
  }

  Array<Stmt> StripPipelineDeclarationStmts(const Array<Stmt> &pipeline_body,
                                            Array<Buffer> *block_local_allocs,
                                            Array<Buffer> *flat_local_allocs) {
    ICHECK(block_local_allocs != nullptr);
    ICHECK(flat_local_allocs != nullptr);
    Array<Stmt> stage_stmts;
    bool filtered = false;
    for (const Stmt &child : pipeline_body) {
      if (IsPipelineDeclarationStmt(child)) {
        if (const auto *alloc = child.as<AllocBufferNode>()) {
          const Buffer &buffer = alloc->buffer;
          buffer_data_to_buffer_.Set(buffer->data, buffer);
          allocated_buffers_.insert(buffer);
          block_local_allocs->push_back(buffer);
          flat_local_allocs->push_back(buffer);
        } else {
          const auto *decl = child.as<DeclBufferNode>();
          ICHECK(decl != nullptr);
          const Buffer &buffer = decl->buffer;
          buffer_data_to_buffer_.Set(buffer->data, buffer);
        }
        filtered = true;
        continue;
      }
      stage_stmts.push_back(child);
    }
    if (!filtered) {
      return pipeline_body;
    }
    ICHECK(!stage_stmts.empty())
        << "ValueError: The body of the software pipeline has no stages "
           "after removing buffer declarations";
    return stage_stmts;
  }

  Map<String, Any>
  StripPipelineAnnotations(const Map<String, Any> &annotations) const {
    Map<String, Any> preserved_annotations;
    for (const auto &kv : annotations) {
      const String &key = kv.first;
      if (key != s_tir::attr::software_pipeline_stage &&
          key != s_tir::attr::software_pipeline_order &&
          key != s_tir::attr::software_pipeline_async_stages &&
          key != kPipelineAsyncProducers &&
          key != kPipelineAsyncProducerGroups && key != kPipelineTmaCopies &&
          key != kPipelineReplayableScalarBinds && key != "num_stages" &&
          key != "tl_pipelined_num_stages") {
        preserved_annotations.Set(key, kv.second);
      }
    }
    return preserved_annotations;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    struct ScopedAllocation {
      Buffer buffer;
      bool existed;
    };

    Array<Stmt> seq;
    bool changed = false;
    std::vector<std::pair<Var, Optional<Buffer>>> old_bindings;
    std::vector<ScopedAllocation> old_allocated;
    std::vector<std::pair<size_t, size_t>> flat_alloc_indices;

    auto register_buffer = [&](const Buffer &buffer,
                               bool is_allocation) -> std::optional<size_t> {
      old_bindings.emplace_back(buffer->data,
                                buffer_data_to_buffer_.Get(buffer->data));
      buffer_data_to_buffer_.Set(buffer->data, buffer);
      if (is_allocation) {
        old_allocated.push_back({buffer, allocated_buffers_.count(buffer) > 0});
        allocated_buffers_.insert(buffer);
        return old_allocated.size() - 1;
      }
      return std::nullopt;
    };

    auto apply_pending_flat_alloc_remaps = [&]() {
      for (auto &[stmt_index, alloc_state_index] : flat_alloc_indices) {
        const Buffer &old_buffer = old_allocated[alloc_state_index].buffer;
        if (auto remapped = pending_buffer_remap_.Get(old_buffer)) {
          const auto *alloc = seq[stmt_index].as<AllocBufferNode>();
          ICHECK(alloc != nullptr);
          Buffer new_buffer = remapped.value();
          seq.Set(stmt_index,
                  AllocBuffer(new_buffer, alloc->annotations, alloc->span));
          buffer_data_to_buffer_.Set(old_buffer->data, new_buffer);
          if (!old_allocated[alloc_state_index].existed) {
            allocated_buffers_.erase(old_buffer);
            allocated_buffers_.insert(new_buffer);
          }
          pending_layout_remapped_allocs_.emplace_back(old_buffer, new_buffer);
          old_allocated[alloc_state_index].buffer = new_buffer;
          pending_buffer_remap_.erase(old_buffer);
          changed = true;
        }
      }
    };

    for (const Stmt &child : op->seq) {
      Stmt new_child = VisitStmt(child);
      changed = changed || !new_child.same_as(child);
      seq.push_back(new_child);
      apply_pending_flat_alloc_remaps();

      if (const auto *alloc = new_child.as<AllocBufferNode>()) {
        std::optional<size_t> alloc_state_index =
            register_buffer(alloc->buffer, true);
        ICHECK(alloc_state_index.has_value());
        flat_alloc_indices.emplace_back(seq.size() - 1,
                                        alloc_state_index.value());
      } else if (const auto *decl = new_child.as<DeclBufferNode>()) {
        register_buffer(decl->buffer, false);
      }
    }
    apply_pending_flat_alloc_remaps();

    for (auto it = old_allocated.rbegin(); it != old_allocated.rend(); ++it) {
      if (!it->existed) {
        allocated_buffers_.erase(it->buffer);
      }
    }
    for (auto it = old_bindings.rbegin(); it != old_bindings.rend(); ++it) {
      if (it->second.defined()) {
        buffer_data_to_buffer_.Set(it->first, it->second.value());
      } else {
        buffer_data_to_buffer_.erase(it->first);
      }
    }

    if (!changed) {
      return GetRef<Stmt>(op);
    }
    return SeqStmt(seq, op->span);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Step 1: Recursively rewrite the children first.
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (!HasPipelineAnnotation(op)) {
      return for_node;
    }
    // Step 2: Find the body and buffer allocations of the pipeline.
    Stmt pipeline_body_root = for_node->body;
    Array<Buffer> pipeline_allocs;
    Array<Buffer> block_local_allocs; // flat allocations inside pipeline body
    Array<Buffer> flat_local_allocs;

    Array<Stmt> pipeline_body_stmts = NormalizePipelineBody(pipeline_body_root);

    // PipelinePlanning emits stage/order annotations only for executable
    // pipeline statements. Flat TIRX keeps loop-local AllocBuffer/DeclBuffer as
    // standalone statements in the loop body, so strip them from the stage
    // stream before blockizing and consuming annotations. The declarations are
    // still registered as local allocations so RewritePipeline can
    // multi-version and reattach them.
    pipeline_body_stmts = StripPipelineDeclarationStmts(
        pipeline_body_stmts, &block_local_allocs, &flat_local_allocs);

    PipelineInfo pipeline_info;
    PipelineSchedule schedule = BuildPipelineSchedule(pipeline_body_stmts);
    Array<SBlock> original_order = schedule.original_order;

    // Collect all buffers that are actually used in the pipeline loop body.
    // This includes buffers allocated in outer blocks (like logits_smem) that
    // are used inside the pipeline loop.
    pipeline_allocs =
        CollectUsedPipelineBuffers(MakePipelineBody(pipeline_body_stmts),
                                   buffer_data_to_buffer_, allocated_buffers_);

    Optional<Array<Integer>> replayable_bind_mask;
    if (auto replayable_bind_anno =
            op->annotations.Get(kPipelineReplayableScalarBinds)) {
      auto mask = Downcast<Array<Integer>>(replayable_bind_anno.value());
      if (mask.size() == original_order.size()) {
        bool valid_mask = true;
        for (size_t i = 0; i < original_order.size(); ++i) {
          if (!is_zero(mask[i]) &&
              original_order[i]->body.as<BindNode>() == nullptr) {
            valid_mask = false;
            break;
          }
        }
        if (valid_mask) {
          replayable_bind_mask = std::move(mask);
        }
      }
    }
    BufferSet pipeline_write_buffers;
    if (!replayable_bind_mask.defined()) {
      pipeline_write_buffers = CollectPipelineWriteBuffers(original_order);
    }
    Array<SBlock> scalar_binding_blocks;
    Array<SBlock> scheduled_order;
    std::vector<char> is_replayable_bind;
    is_replayable_bind.reserve(original_order.size());
    for (size_t i = 0; i < original_order.size(); ++i) {
      const SBlock &block = original_order[i];
      bool replayable =
          replayable_bind_mask.defined()
              ? !is_zero(replayable_bind_mask.value()[i])
              : IsReplayableScalarBindBlock(block, pipeline_write_buffers);
      is_replayable_bind.push_back(replayable ? 1 : 0);
      if (replayable) {
        scalar_binding_blocks.push_back(block);
      } else {
        scheduled_order.push_back(block);
      }
    }
    ICHECK(!scheduled_order.empty())
        << "ValueError: The body of the software pipeline has no schedulable "
           "statements after removing replayable scalar Bind statements";

    auto pipeline_stages = Downcast<Array<Integer>>(
        op->annotations.at(s_tir::attr::software_pipeline_stage));
    auto pipeline_orders = Downcast<Array<Integer>>(
        op->annotations.at(s_tir::attr::software_pipeline_order));
    ICHECK_EQ(pipeline_stages.size(), pipeline_orders.size())
        << "PrimFunc " << global_symbol_
        << " has software_pipeline_stage annotation " << pipeline_stages
        << " and software_pipeline_order annotation " << pipeline_orders
        << " with different sizes";

    bool annotations_include_replayable_binds = false;
    if (pipeline_stages.size() == scheduled_order.size()) {
      annotations_include_replayable_binds = false;
    } else if (pipeline_stages.size() == original_order.size()) {
      annotations_include_replayable_binds = true;
    } else {
      LOG(FATAL) << "PrimFunc " << global_symbol_
                 << " has schedulable pipeline order "
                 << scheduled_order.Map(
                        [](const auto &block) { return block->name_hint; })
                 << " and original order "
                 << original_order.Map(
                        [](const auto &block) { return block->name_hint; })
                 << ", but pipeline annotation is " << pipeline_stages
                 << " with different size";
    }

    std::vector<size_t> scheduled_annotation_indices;
    scheduled_annotation_indices.reserve(scheduled_order.size());
    if (annotations_include_replayable_binds) {
      size_t scheduled_index = 0;
      for (size_t i = 0; i < original_order.size(); ++i) {
        if (is_replayable_bind[i]) {
          continue;
        }
        ICHECK(scheduled_index < scheduled_order.size());
        ICHECK(scheduled_order[scheduled_index].same_as(original_order[i]));
        scheduled_annotation_indices.push_back(i);
        ++scheduled_index;
      }
    } else {
      for (size_t i = 0; i < scheduled_order.size(); ++i) {
        scheduled_annotation_indices.push_back(i);
      }
    }

    auto expected_annotation_size = annotations_include_replayable_binds
                                        ? original_order.size()
                                        : scheduled_order.size();

    std::unordered_set<int> pipeline_async_stages;
    if (auto async_annot =
            op->annotations.Get(s_tir::attr::software_pipeline_async_stages)) {
      for (const Integer &stage :
           Downcast<Array<Integer>>(async_annot.value())) {
        pipeline_async_stages.insert(static_cast<int>(stage.IntValue()));
      }
    }
    Optional<Array<Integer>> pipeline_async_producers;
    if (auto async_producers_anno =
            op->annotations.Get(kPipelineAsyncProducers)) {
      auto async_flags = Downcast<Array<Integer>>(async_producers_anno.value());
      ICHECK_EQ(async_flags.size(), expected_annotation_size)
          << "PrimFunc " << global_symbol_ << " has schedulable order "
          << scheduled_order.Map(
                 [](const auto &block) { return block->name_hint; })
          << ", but async producer annotation is " << async_flags
          << " with different size";
      pipeline_async_producers = async_flags;
    }
    Optional<Array<Integer>> pipeline_async_producer_groups;
    if (auto async_groups_anno =
            op->annotations.Get(kPipelineAsyncProducerGroups)) {
      auto async_group_ids =
          Downcast<Array<Integer>>(async_groups_anno.value());
      ICHECK_EQ(async_group_ids.size(), expected_annotation_size)
          << "PrimFunc " << global_symbol_ << " has schedulable order "
          << scheduled_order.Map(
                 [](const auto &block) { return block->name_hint; })
          << ", but async producer group annotation is " << async_group_ids
          << " with different size";
      pipeline_async_producer_groups = async_group_ids;
    }

    for (size_t i = 0; i < scheduled_order.size(); i++) {
      size_t annotation_index = scheduled_annotation_indices[i];
      int stage =
          static_cast<int>(pipeline_stages[annotation_index].IntValue());
      bool is_async_candidate =
          pipeline_async_producers
              ? !is_zero(pipeline_async_producers.value()[annotation_index])
              : (pipeline_async_stages.count(stage) > 0);
      // Stages that already carry pipeline async control attrs keep that
      // ownership; the injector only annotates plain producer stages.
      bool is_async = is_async_candidate && !ContainsPipelineAsyncControlAttrs(
                                                scheduled_order[i]->body);
      PipelineAnnotation stage_order{
          stage,
          /*order=*/
          static_cast<int>(pipeline_orders[annotation_index].IntValue()),
          /*async=*/is_async,
          /*async_group_id=*/
          pipeline_async_producer_groups
              ? static_cast<int>(
                    pipeline_async_producer_groups.value()[annotation_index]
                        .IntValue())
              : -1};
      pipeline_info.emplace(scheduled_order[i], stage_order);
    }

    if (annotations_include_replayable_binds) {
      for (const SBlock &binding_block : scalar_binding_blocks) {
        const auto *bind = binding_block->body.as<BindNode>();
        ICHECK(bind != nullptr);
        bool seen_consumer = false;
        bool multiple_consumers = false;
        PipelineAnnotation first_consumer;
        for (const SBlock &consumer : scheduled_order) {
          Array<Var> undefined_vars =
              UndefinedVars(consumer->body, Array<Var>{});
          bool uses_binding = false;
          for (const Var &var : undefined_vars) {
            if (var.same_as(bind->var)) {
              uses_binding = true;
              break;
            }
          }
          if (!uses_binding) {
            continue;
          }
          const PipelineAnnotation &anno = pipeline_info.at(consumer);
          if (!seen_consumer) {
            first_consumer = anno;
            seen_consumer = true;
          } else if (first_consumer.stage != anno.stage ||
                     first_consumer.order != anno.order) {
            multiple_consumers = true;
            break;
          }
        }
        if (multiple_consumers) {
          LOG(WARNING)
              << "Scalar Bind '" << bind->var
              << "' is used by multiple pipeline stages; its annotation is "
                 "ignored and the bind is replayed at each use.";
        }
      }
    }

    ValidateScheduledBindDependencies(pipeline_info, scheduled_order);
    ValidatePipelineBody(pipeline_info, scheduled_order);

    if (!HasOverlappableStages(pipeline_info)) {
      for (const auto &buffer : flat_local_allocs) {
        buffer_data_to_buffer_.erase(buffer->data);
        allocated_buffers_.erase(buffer);
      }
      return For(for_node->loop_var, for_node->min, for_node->extent,
                 for_node->kind, for_node->body, for_node->thread_binding,
                 StripPipelineAnnotations(for_node->annotations),
                 for_node->step, for_node->span);
    }

    // Step 3.5: Pipeline-level TMA barrier management.
    // When TMA copies are present (without warp specialization), rewrite
    // them to use tl.tileop.tma_copy with shared pipeline barriers and insert
    // mbarrier_wait_parity before the first consumer stage.
    // Creates pipeline_mbar[pipeline_depth] at final size so LowerTileOp
    // uses the provided barrier instead of allocating separate per-copy ones.
    Buffer pipeline_barrier_buf;
    {
      int max_stage = 0;
      for (const auto &pair : pipeline_info) {
        max_stage = std::max(max_stage, pair.second.stage);
      }
      // Use the actual pipeline depth (number of buffer copies) for barrier
      // sizing, not the SW pipeline stage count (max_stage + 1).
      // Even for pipeline_depth=1 we create a shared barrier so that
      // LowerTileOp uses it instead of allocating separate per-copy barriers.
      Optional<Integer> pipelined_num_stages = GetPipelineNumStages(op);
      int pipeline_depth =
          pipelined_num_stages.defined()
              ? static_cast<int>(pipelined_num_stages.value().IntValue())
              : max_stage + 1;
      // Clamp to at least 1 so we always allocate at least one barrier slot.
      pipeline_depth = std::max(pipeline_depth, 1);
      if (max_stage > 0) {
        if (auto tma_copies_anno = op->annotations.Get(kPipelineTmaCopies)) {
          auto raw_tma_copies =
              Downcast<Array<Integer>>(tma_copies_anno.value());
          Array<Integer> tma_copies;
          if (raw_tma_copies.size() == scheduled_order.size()) {
            tma_copies = raw_tma_copies;
          } else if (raw_tma_copies.size() == original_order.size()) {
            for (size_t annotation_index : scheduled_annotation_indices) {
              tma_copies.push_back(raw_tma_copies[annotation_index]);
            }
          }
          if (tma_copies.size() == scheduled_order.size()) {
            bool has_tma_copy =
                std::any_of(tma_copies.begin(), tma_copies.end(),
                            [](const Integer &tc) { return !is_zero(tc); });
            if (has_tma_copy) {
              pipeline_barrier_buf = RewritePipelineTmaBarriers(
                  scheduled_order, pipeline_info, tma_copies,
                  buffer_data_to_buffer_, allocated_buffers_,
                  block_local_allocs, for_node->loop_var, for_node->min,
                  pipeline_depth);
            }
          }
        }
      }
    }

    // Step 4: Rewrite the pipeline body.
    // local_allocs contains buffers allocated in the pipeline block itself.
    // pipeline_allocs contains all buffers that need multi-versioning,
    // including buffers from outer blocks.
    // Step 4.5: Expand all barrier buffers for pipelining.
    // This handles both ISP-created pipeline_mbar AND user-written
    // T.alloc_barrier, so that no late standalone barrier-only fixup is needed.
    // Must run BEFORE local_allocs is copied from block_local_allocs.
    {
      Optional<Integer> pipelined_ns = GetPipelineNumStages(op);
      int barrier_depth = 1;
      if (pipelined_ns.defined()) {
        barrier_depth = static_cast<int>(pipelined_ns.value().IntValue());
      } else if (op->annotations.count("num_stages")) {
        barrier_depth = static_cast<int>(
            Downcast<Integer>(op->annotations.Get("num_stages").value())
                .IntValue());
      }
      Map<Buffer, Buffer> barrier_remap = ExpandPipelineBarriers(
          scheduled_order, pipeline_info, buffer_data_to_buffer_,
          allocated_buffers_, block_local_allocs, pipeline_allocs,
          for_node->loop_var, for_node->min, barrier_depth);
      // Register expanded barriers for outer block alloc_buffers update.
      for (const auto &[old_buf, new_buf] : barrier_remap) {
        pending_buffer_remap_.Set(old_buf, new_buf);
      }
    }

    Array<Buffer> local_allocs = block_local_allocs;
    local_allocs.insert(local_allocs.end(),
                        schedule.nested_local_allocs.begin(),
                        schedule.nested_local_allocs.end());

    PipelineRewriteResult rewrite_result = RewritePipeline(
        buffer_data_to_buffer_, pipeline_allocs, local_allocs, for_node,
        pipeline_info, scalar_binding_blocks, target_);
    Stmt pipeline = rewrite_result.pipeline;
    subtree_modified_ = true;

    auto unwrap_outer_attrs = [](Stmt stmt) {
      std::vector<AttrStmt> attrs;
      while (const auto *attr = stmt.as<AttrStmtNode>()) {
        attrs.push_back(Downcast<AttrStmt>(stmt));
        stmt = attr->body;
      }
      return std::make_pair(attrs, stmt);
    };
    auto rewrap_outer_attrs = [](Stmt stmt,
                                 const std::vector<AttrStmt> &attrs) {
      for (auto it = attrs.rbegin(); it != attrs.rend(); ++it) {
        stmt = AttrStmt((*it)->node, (*it)->attr_key, (*it)->value, stmt,
                        (*it)->span);
      }
      return stmt;
    };

    // Update barrier_init annotations for expanded barrier buffers.
    // For pipeline_mbar (ISP-created): add new entry with arrive_count=1 per
    // slot. For user barriers (T.alloc_barrier): replicate existing arrive
    // counts across the expanded slots.
    {
      auto [outer_attrs, inner_stmt] = unwrap_outer_attrs(pipeline);
      SBlockRealize br = Downcast<SBlockRealize>(inner_stmt);
      SBlock block = br->block;
      SBlockNode *bn = block.CopyOnWrite();

      Map<Var, Array<PrimExpr>> barrier_init_map;
      if (bn->annotations.count("barrier_init")) {
        barrier_init_map = Downcast<Map<Var, Array<PrimExpr>>>(
            bn->annotations.Get("barrier_init").value());
      }
      bool changed = false;

      // Handle ISP-created pipeline barrier (needs new entry).
      if (pipeline_barrier_buf.defined()) {
        // After ExpandPipelineBarriers, pipeline_mbar has been expanded.
        // Look up the expanded buffer via buffer_data_to_buffer_.
        Buffer expanded_buf =
            buffer_data_to_buffer_[pipeline_barrier_buf->data];
        int expanded_slots = Downcast<IntImm>(expanded_buf->shape[0])->value;
        Array<PrimExpr> counts;
        for (int s = 0; s < expanded_slots; ++s) {
          counts.push_back(IntImm(DataType::Int(32), 1));
        }
        barrier_init_map.Set(expanded_buf->data, counts);
        changed = true;
      }

      // Replicate existing barrier_init entries for expanded barriers.
      Map<Var, Array<PrimExpr>> updated_init;
      for (const auto &[var, counts] : barrier_init_map) {
        Buffer buf = buffer_data_to_buffer_[var];
        int buf_size = Downcast<IntImm>(buf->shape[0])->value;
        int orig_size = static_cast<int>(counts.size());
        if (buf_size > orig_size && orig_size > 0 &&
            buf_size % orig_size == 0) {
          // Replicate pattern to match expanded size.
          Array<PrimExpr> new_counts;
          for (int v = 0; v < buf_size; v += orig_size) {
            for (const auto &c : counts) {
              new_counts.push_back(c);
            }
          }
          updated_init.Set(var, new_counts);
          changed = true;
        } else {
          updated_init.Set(var, counts);
        }
      }

      if (changed) {
        bn->annotations.Set("barrier_init", updated_init);
        pipeline = rewrap_outer_attrs(
            SBlockRealize(br->iter_values, br->predicate, block, br->span),
            outer_attrs);
      }
    }

    // Store the buffer remapping for updating outer block alloc_buffers
    for (const auto &kv : rewrite_result.buffer_remap) {
      pending_buffer_remap_.Set(kv.first, kv.second);
    }
    pipeline = LowerAsyncCommitWaitAttrs(pipeline);

    return pipeline;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
      allocated_buffers_.insert(buffer);
    }

    bool outer_flag = subtree_modified_;
    size_t layout_remap_mark = pending_layout_remapped_allocs_.size();
    subtree_modified_ = false;
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    bool children_modified = subtree_modified_;
    // Propagate to parent: if this subtree was modified, parent should know.
    subtree_modified_ = outer_flag || children_modified;

    // Update alloc_buffers with any pending buffer remaps from pipeline
    // rewriting. This handles buffers allocated in this block but
    // multi-versioned during pipeline rewriting of inner loops.
    bool allocs_changed = false;
    bool layout_changed = false;
    Array<Buffer> new_alloc_buffers;
    std::vector<std::pair<Buffer, Buffer>> remapped_allocs;
    for (const auto &buffer : block->alloc_buffers) {
      if (auto remapped = pending_buffer_remap_.Get(buffer)) {
        new_alloc_buffers.push_back(remapped.value());
        remapped_allocs.emplace_back(buffer, remapped.value());
        pending_buffer_remap_.erase(buffer);
        allocs_changed = true;
      } else {
        new_alloc_buffers.push_back(buffer);
      }
    }

    if (!remapped_allocs.empty()) {
      auto ann = block->annotations;
      if (UpdateExpandedLayoutMapForRemappedAllocs(remapped_allocs, &ann)) {
        block.CopyOnWrite()->annotations = std::move(ann);
        layout_changed = true;
      }
    }
    if (pending_layout_remapped_allocs_.size() > layout_remap_mark) {
      std::vector<std::pair<Buffer, Buffer>> flat_remapped_allocs(
          pending_layout_remapped_allocs_.begin() + layout_remap_mark,
          pending_layout_remapped_allocs_.end());
      auto ann = block->annotations;
      if (UpdateExpandedLayoutMapForRemappedAllocs(flat_remapped_allocs,
                                                   &ann)) {
        block.CopyOnWrite()->annotations = std::move(ann);
        pending_layout_remapped_allocs_.erase(
            pending_layout_remapped_allocs_.begin() + layout_remap_mark,
            pending_layout_remapped_allocs_.end());
        layout_changed = true;
      }
    }

    // Replicate barrier_init counts for any expanded barrier buffers.
    if (allocs_changed && block->annotations.count("barrier_init")) {
      Map<Var, Array<PrimExpr>> init_map = Downcast<Map<Var, Array<PrimExpr>>>(
          block->annotations.Get("barrier_init").value());
      Map<Var, Array<PrimExpr>> new_init;
      bool init_changed = false;
      for (const auto &[var, counts] : init_map) {
        // Find the buffer for this var — it may have been remapped.
        Buffer buf;
        for (const auto &ab : new_alloc_buffers) {
          if (ab->data.same_as(var)) {
            buf = ab;
            break;
          }
        }
        if (buf.defined()) {
          int buf_size = Downcast<IntImm>(buf->shape[0])->value;
          int orig_size = static_cast<int>(counts.size());
          if (buf_size > orig_size && orig_size > 0 &&
              buf_size % orig_size == 0) {
            Array<PrimExpr> new_counts;
            for (int v = 0; v < buf_size; v += orig_size) {
              for (const auto &c : counts)
                new_counts.push_back(c);
            }
            new_init.Set(var, new_counts);
            init_changed = true;
            continue;
          }
        }
        new_init.Set(var, counts);
      }
      if (init_changed) {
        SBlockNode *bn = block.CopyOnWrite();
        bn->annotations.Set("barrier_init", new_init);
        bn->alloc_buffers = new_alloc_buffers;
        allocs_changed = false; // already applied
      }
    }

    bool modified = children_modified || allocs_changed || layout_changed;
    if (modified) {
      // Recalculate reads/writes only when the block was actually
      // modified by pipeline rewriting.  Unconditional recalculation
      // can embed references to block-local buffers (e.g. local.var)
      // into the block's own read/write annotations, which misleads
      // downstream LCA analysis and causes those buffers to be
      // promoted to kernel parameters.
      //
      // After recalculation:
      // 1. Drop BufferRegions whose buffer is allocated in this block.
      // 2. Widen to full-region any BufferRegion whose index
      //    expressions reference a data var of any buffer allocated
      //    in this block or any nested block. This prevents
      //    downstream LCA analysis from seeing those vars at the
      //    outer scope and promoting them to kernel parameters.
      BufferSet local_bufs;
      VarSet local_data_vars;
      for (const auto &buf : block->alloc_buffers) {
        local_bufs.insert(buf);
        local_data_vars.insert(buf->data);
      }
      // Also collect data vars from all nested blocks.
      PostOrderVisit(block->body, [&](const ObjectRef &obj) {
        if (auto *inner = obj.as<SBlockNode>()) {
          for (const auto &buf : inner->alloc_buffers) {
            local_data_vars.insert(buf->data);
          }
        }
      });
      auto region_uses_local_var = [&](const BufferRegion &br) -> bool {
        for (const auto &range : br->region) {
          bool found = false;
          PostOrderVisit(range->min, [&](const ObjectRef &obj) {
            if (found)
              return;
            if (auto *load = obj.as<BufferLoadNode>()) {
              if (local_data_vars.count(load->buffer->data)) {
                found = true;
              }
            }
            if (auto *var = obj.as<VarNode>()) {
              if (local_data_vars.count(GetRef<Var>(var))) {
                found = true;
              }
            }
          });
          if (found)
            return true;
          PostOrderVisit(range->extent, [&](const ObjectRef &obj) {
            if (found)
              return;
            if (auto *load = obj.as<BufferLoadNode>()) {
              if (local_data_vars.count(load->buffer->data)) {
                found = true;
              }
            }
            if (auto *var = obj.as<VarNode>()) {
              if (local_data_vars.count(GetRef<Var>(var))) {
                found = true;
              }
            }
          });
          if (found)
            return true;
        }
        return false;
      };
      Array<Array<BufferRegion>> access =
          GetSBlockReadWriteRegion(block, buffer_data_to_buffer_);
      auto sanitize = [&](const Array<BufferRegion> &regions) {
        Array<BufferRegion> out;
        for (const auto &br : regions) {
          if (local_bufs.count(br->buffer)) {
            continue; // drop block-local buffer
          }
          if (region_uses_local_var(br)) {
            out.push_back(BufferRegion::FullRegion(br->buffer));
          } else {
            out.push_back(br);
          }
        }
        return out;
      };
      SBlockNode *n = block.CopyOnWrite();
      n->reads = sanitize(access[0]);
      n->writes = sanitize(access[1]);
      n->alloc_buffers = std::move(new_alloc_buffers);
    }

    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
      allocated_buffers_.erase(buffer);
    }
    return block;
  }

  bool HasPipelineAnnotation(const ForNode *op) const {
    auto it1 = op->annotations.find(s_tir::attr::software_pipeline_stage);
    auto it2 = op->annotations.find(s_tir::attr::software_pipeline_order);
    bool has_stage = it1 != op->annotations.end();
    bool has_order = it2 != op->annotations.end();
    if (has_stage && has_order) {
      return true;
    }
    if (has_stage) {
      LOG(FATAL)
          << "ValueError: Stage of the software pipeline is not defined.";
    }
    if (has_order) {
      LOG(FATAL)
          << "ValueError: Order of the software pipeline is not defined.";
    }
    return false;
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> allocated_buffers_;
  Map<Buffer, Buffer> pending_buffer_remap_;
  std::vector<std::pair<Buffer, Buffer>> pending_layout_remapped_allocs_;
  Optional<Target> target_;
  Optional<String> global_symbol_;
  // Track whether any pipeline was actually injected in the current
  // subtree.  Used to avoid unnecessary reads/writes recalculation
  // on blocks whose descendants were not modified.
  bool subtree_modified_ = false;
};

Stmt InjectPipeline(const PrimFunc &func) {
  return PipelineInjector::Inject(func);
}

} // namespace software_pipeline

/*!
 * \brief Transform annotated loops into pipelined one that parallelize
 * producers and consumers. \return The IR transform pass.
 */
tirx::transform::Pass InjectSoftwarePipeline() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto *fptr = f.CopyOnWrite();
    fptr->body = software_pipeline::InjectPipeline(f);
    fptr->body = ConvertSSA(std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectSoftwarePipeline", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.InjectSoftwarePipeline",
                        InjectSoftwarePipeline);
}

} // namespace tl
} // namespace tvm
