/*!
 *  \file lower_shared_tmem.cc
 *  \brief Convert shared.tmem buffers to plain shared + ptx init, and do
 *         coordinate translation (from logical address to physical address)
 *
 *  Logical buffers are not one-to-one with tcgen05.alloc calls: PlanTmemArenas
 *  packs several into one allocation when that saves columns, and shifts every
 *  address a packed buffer forms by where it starts in the allocation.
 */
#include "cuda/target_utils.h"
#include "op/builtin.h"
#include "support/check.h"
#include "tvm/ir/type.h"
#include <algorithm>
#include <string>
#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

/*!
 * \brief Collect TMEM buffers explicitly deallocated on fallthrough paths.
 *
 * A "fallthrough path" is one that reaches the end of the statement without
 * hitting thread_return().  Buffers deallocated on every such path already
 * have an explicit dealloc, so we can skip the auto-dealloc at block end.
 *
 * \return {buffers deallocated on fallthrough, whether the stmt can
 * fallthrough}
 */
static std::pair<VarSet, bool> CollectFallthroughDeallocs(const Stmt &stmt) {
  if (!stmt.defined())
    return {{}, true};

  // Unwrap transparent wrapper nodes
  if (stmt.as<BindNode>())
    return {{}, true};
  if (auto *n = stmt.as<AttrStmtNode>())
    return CollectFallthroughDeallocs(n->body);
  if (auto *n = stmt.as<SBlockNode>())
    return CollectFallthroughDeallocs(n->body);
  if (auto *n = stmt.as<SBlockRealizeNode>())
    return CollectFallthroughDeallocs(n->block->body);
  if (auto *n = stmt.as<ForNode>())
    return CollectFallthroughDeallocs(n->body);

  // Sequential: accumulate deallocs; stop if any child doesn't fallthrough
  if (auto *seq = stmt.as<SeqStmtNode>()) {
    VarSet deallocs;
    for (const auto &child : seq->seq) {
      auto [d, ft] = CollectFallthroughDeallocs(child);
      if (!ft)
        return {{}, false};
      deallocs.insert(d.begin(), d.end());
    }
    return {std::move(deallocs), true};
  }

  // Branch: collect deallocs only from branches that can fallthrough
  if (auto *iff = stmt.as<IfThenElseNode>()) {
    auto [then_d, then_ft] = CollectFallthroughDeallocs(iff->then_case);
    auto [else_d, else_ft] =
        iff->else_case.defined()
            ? CollectFallthroughDeallocs(iff->else_case.value())
            : std::pair<VarSet, bool>{{}, true};
    VarSet deallocs;
    if (then_ft)
      deallocs.insert(then_d.begin(), then_d.end());
    if (else_ft)
      deallocs.insert(else_d.begin(), else_d.end());
    return {std::move(deallocs), then_ft || else_ft};
  }

  // Leaf: detect deallocate_tmem and thread_return
  if (auto *eval = stmt.as<EvaluateNode>()) {
    if (auto *call = eval->value.as<CallNode>()) {
      if (call->op.same_as(tl::deallocate_tmem())) {
        ICHECK_EQ(call->args.size(), 1U);
        auto *buf = call->args[0].as<VarNode>();
        ICHECK(buf) << "tl.deallocate_tmem expects a buffer data Var";
        return {{GetRef<Var>(buf)}, true};
      }
      if (call->op.same_as(builtin::thread_return())) {
        return {{}, false};
      }
    }
  }

  return {{}, true};
}

/*!
 * \brief Collect every TMEM buffer named by a tl.deallocate_tmem call.
 *
 * Unlike CollectFallthroughDeallocs this ignores control flow: a buffer whose
 * lifetime the kernel manages by hand -- even on a path that ends in
 * thread_return() -- must own its physical allocation outright, because
 * releasing it would release whatever is packed alongside it.
 */
static VarSet CollectExplicitDeallocs(const Stmt &stmt) {
  VarSet deallocs;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    const auto *call = node.as<CallNode>();
    if (call == nullptr || !call->op.same_as(tl::deallocate_tmem()))
      return;
    ICHECK_EQ(call->args.size(), 1U);
    const auto *buffer_data = call->args[0].as<VarNode>();
    ICHECK(buffer_data) << "tl.deallocate_tmem expects a buffer data Var";
    deallocs.insert(GetRef<Var>(buffer_data));
  });
  return deallocs;
}

// A CTA's tensor memory is 128 datapaths (rows) of 512 32-bit columns.
static constexpr int kTmemNumDatapaths = 128;
static constexpr int kTmemNumColumns = 512;
// tcgen05.alloc hands out columns in powers of two, 32 at a minimum.
static constexpr int kTmemMinAllocColumns = 32;
// tcgen05.cp.32x128b.warpx4 writes four b32 columns per instruction, which is
// also the granularity tcgen05.mma.blockscaled reads its scale factors at.
static constexpr int kTmemScaleFactorAlignment = 4;

/*! \brief Round a column count up to a count tcgen05.alloc accepts. */
static int TmemAllocationSize(int num_cols) {
  int num_cols_allocated = kTmemMinAllocColumns;
  for (; num_cols_allocated < num_cols; num_cols_allocated *= 2) {
  }
  return num_cols_allocated;
}

/*!
 * \brief The column boundary a logical buffer may start on inside an arena.
 *
 * A buffer at least kTmemMinAllocColumns wide keeps the 32-column alignment
 * tcgen05.alloc would have given it on its own, so sharing an allocation can
 * never leave an accumulator -- or a TMEM-resident A operand -- less aligned
 * than it is when allocated alone.  Narrower buffers are block-scale factors,
 * written by tcgen05.cp.32x128b.warpx4 and read by tcgen05.mma.blockscaled;
 * four columns is that instruction pair's granularity, and the boundary CUTLASS
 * (cutlass::detail::find_tmem_tensor_col_offset) and DeepGEMM place the very
 * same operands on.
 */
static int TmemSubAllocationAlignment(int num_cols) {
  return num_cols >= kTmemMinAllocColumns ? kTmemMinAllocColumns
                                          : kTmemScaleFactorAlignment;
}

/*! \brief Where one logical TMEM buffer sits inside a physical allocation. */
struct TmemPlacement {
  int arena_index = -1;
  /*! \brief b32 columns from the allocation's base address to this buffer. */
  int col_offset = 0;
};

/*! \brief One tcgen05.alloc, shared by one or more logical TMEM buffers. */
struct TmemArena {
  /*! \brief Indices of the logical buffers placed here, in placement order. */
  std::vector<int> members;
  /*! \brief Columns handed out so far, including alignment padding. */
  int num_cols_used = 0;
  /*! \brief Columns requested from tcgen05.alloc: a power of two, >= 32. */
  int num_cols_allocated = 0;
  /*!
   * \brief Whether more buffers may join this allocation.
   *
   * Closed for a buffer with an explicit tl.deallocate_tmem, whose lifetime
   * ends before the block does.
   */
  bool open = true;
};

/*!
 * \brief Pack logical TMEM buffers into as few tcgen05.alloc calls as possible.
 *
 * Rounding each buffer up to a power of two on its own wastes whole columns: a
 * 384-column accumulator next to three 4-column scale-factor buffers needs 396
 * columns, but as four allocations asks for 512 + 32 + 32 + 32 = 608 of the 512
 * a CTA has, so the kernel cannot be expressed at all.  CUTLASS and DeepGEMM
 * share one allocation between these very operands by hand.
 *
 * Strategy: widest buffer first, each joining an existing arena only when that
 * *strictly* lowers the total column count and otherwise opening its own.  Two
 * properties follow:
 *
 *  - a placement costs at most the buffer's standalone allocation, so packing
 *    can never grow a kernel's TMEM footprint;
 *  - a kernel whose buffers already fit lowers exactly as it did before.
 *
 * Widest-first also keeps wide buffers on 32-column boundaries and lets narrow
 * ones fill padding the power-of-two rounding has already paid for.
 *
 * TODO: this assumes every packable buffer is live for the whole block, and
 * excludes a buffer with an explicit tl.deallocate_tmem instead of modelling
 * its lifetime.  A liveness analysis would let buffers whose live ranges are
 * disjoint reuse the same columns outright, rather than only fill each other's
 * power-of-two padding, and would let a hand-released buffer pass its columns
 * on to one that starts later.
 *
 * \param num_cols b32 columns each logical buffer needs.
 * \param packable Whether a buffer may share an allocation, see
 * TmemArena::open.
 * \return The arenas, plus one placement per logical buffer.
 */
static std::pair<std::vector<TmemArena>, std::vector<TmemPlacement>>
PlanTmemArenas(const std::vector<int> &num_cols,
               const std::vector<bool> &packable) {
  ICHECK_EQ(num_cols.size(), packable.size());

  std::vector<int> order(num_cols.size());
  for (size_t i = 0; i < order.size(); ++i) {
    order[i] = static_cast<int>(i);
  }
  // Stable, so equally wide buffers keep their declaration order: the plan must
  // not depend on sort or hash implementation details.
  std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    return num_cols[lhs] > num_cols[rhs];
  });

  std::vector<TmemArena> arenas;
  std::vector<TmemPlacement> placements(num_cols.size());

  for (int i : order) {
    int alignment = TmemSubAllocationAlignment(num_cols[i]);
    int best_arena = -1;
    int best_offset = 0;
    // A fresh allocation is the baseline an existing arena has to beat.
    int best_delta = TmemAllocationSize(num_cols[i]);
    if (packable[i]) {
      for (size_t a = 0; a < arenas.size(); ++a) {
        if (!arenas[a].open)
          continue;
        int offset =
            (arenas[a].num_cols_used + alignment - 1) / alignment * alignment;
        if (offset + num_cols[i] > kTmemNumColumns)
          continue;
        int delta = TmemAllocationSize(offset + num_cols[i]) -
                    arenas[a].num_cols_allocated;
        if (delta < best_delta) {
          best_delta = delta;
          best_arena = static_cast<int>(a);
          best_offset = offset;
        }
      }
    }
    if (best_arena < 0) {
      TmemArena arena;
      arena.open = packable[i];
      arenas.push_back(arena);
      best_arena = static_cast<int>(arenas.size()) - 1;
      best_offset = 0;
    }
    TmemArena &arena = arenas[best_arena];
    arena.members.push_back(i);
    arena.num_cols_used = best_offset + num_cols[i];
    arena.num_cols_allocated = TmemAllocationSize(arena.num_cols_used);
    placements[i] = TmemPlacement{best_arena, best_offset};
  }
  return {std::move(arenas), std::move(placements)};
}

class SharedTmemRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt body, Target target) {
    SharedTmemRewriter rewriter;
    rewriter.target_ = std::move(target);
    return rewriter(body);
  }

private:
  static int GetValueBitWidth(DataType dtype) {
    int value_bits = dtype.bits() * dtype.lanes();
    ICHECK_GT(value_bits, 0) << "Invalid TMEM value dtype " << dtype;
    return value_bits;
  }

  static PrimExpr ValueColumnOffsetToB32Column(PrimExpr value_col,
                                               DataType dtype) {
    arith::Analyzer analyzer;
    DataType index_dtype = value_col->dtype;
    PrimExpr value_bits = IntImm(index_dtype, GetValueBitWidth(dtype));
    PrimExpr bits_per_column = IntImm(index_dtype, 32);
    PrimExpr bit_offset = analyzer.Simplify(value_col * value_bits);
    ICHECK(analyzer.CanProveEqual(FloorMod(bit_offset, bits_per_column),
                                  IntImm(index_dtype, 0)))
        << "TMEM address must start on a b32 column, but value-column offset "
        << value_col << " of dtype " << dtype << " has bit offset "
        << bit_offset;
    return analyzer.Simplify(FloorDiv(bit_offset, bits_per_column));
  }

  /*!
   * \brief The number of 32-bit TMEM columns a buffer occupies.
   *
   * LowerTileOp has already materialized the buffer's inferred layout, so the
   * buffer shape is the physical (datapath, value-column) footprint.  This is
   * what a buffer *needs*; PlanTmemArenas decides how many columns are actually
   * allocated for it, alone or shared.
   */
  int GetNumB32ColsRequired(const Buffer &buffer) const {
    ICHECK_EQ(buffer->shape.size(), 2U);

    arith::Analyzer analyzer;
    int num_rows_required =
        analyzer.const_int_bound(buffer->shape[0])->max_value;
    ICHECK(num_rows_required <= kTmemNumDatapaths)
        << "The number of rows required for tmem buffer " << buffer->name
        << " is " << num_rows_required << ", which exceeds the maximum of "
        << kTmemNumDatapaths << " rows";

    int num_value_cols_required =
        analyzer.const_int_bound(buffer->shape[1])->max_value;
    // Layout column coordinates count values of buffer->dtype; PTX TMEM
    // allocation counts 32-bit columns.  Round up so a final partially
    // occupied b32 column is included.
    int num_cols_required =
        (num_value_cols_required * GetValueBitWidth(buffer->dtype) + 31) / 32;
    ICHECK(num_cols_required <= kTmemNumColumns)
        << "The number of columns required for tmem buffer " << buffer->name
        << " is " << num_cols_required << ", which exceeds the maximum of "
        << kTmemNumColumns << " columns";
    return num_cols_required;
  }

  /*! \brief The b32 column offset of a TMEM buffer inside its arena. */
  int GetArenaColOffset(const Var &buffer_data) const {
    auto it = tmem_col_offsets_.find(buffer_data);
    return it == tmem_col_offsets_.end() ? 0 : it->second;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock block = GetRef<SBlock>(op);
    if (op->annotations.count(attr::kLayoutMap)) {
      auto layout_map = op->annotations.Get(attr::kLayoutMap);
      ICHECK(layout_map) << "layout map is not defined";
      layout_map_ = layout_map->as<Map<Buffer, Layout>>().value();
    }

    // Record the mapping from buffer data var to buffer for later lookup
    for (auto buffer : op->alloc_buffers) {
      buffer_map_.insert({buffer->data, buffer});
    }
    for (auto match_buffer : op->match_buffers) {
      buffer_map_.insert({match_buffer->buffer->data, match_buffer->buffer});
    }

    // The TMEM buffers this block introduces, in declaration order.  Arena
    // column offsets follow this order, so it must not come from a hash table.
    Array<Buffer> tmem_buffers;
    auto collect_tmem_buffer = [&](const Buffer &buffer) {
      const auto *ptr_type =
          buffer->data->type_annotation.as<PointerTypeNode>();
      ICHECK(ptr_type) << "LowerSharedTmem requires buffer " << buffer->name
                       << "'s data Var to have a PointerType annotation";
      if (ptr_type->storage_scope != "shared.tmem")
        return;
      // Already lowered together with an enclosing block.
      if (var_remap_.count(buffer->data))
        return;
      tmem_buffers.push_back(buffer);
    };
    for (auto buffer : op->alloc_buffers) {
      collect_tmem_buffer(buffer);
    }
    for (auto match_buffer : op->match_buffers) {
      collect_tmem_buffer(match_buffer->buffer);
    }

    if (tmem_buffers.empty()) {
      return StmtExprMutator::VisitStmt_(op);
    }

    ICHECK(thread_var_.defined()) << "thread_var_ is not defined";

    auto [fallthrough_deallocs, _] = CollectFallthroughDeallocs(op->body);
    VarSet explicit_deallocs = CollectExplicitDeallocs(op->body);

    for (auto buffer : tmem_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }

    // If block has use_2cta attr, add use_2cta: 1 to tmem alloc/dealloc call
    // annotations.
    Map<String, ObjectRef> tmem_call_ann;
    if (op->annotations.count("use_2cta")) {
      PrimExpr val = Downcast<PrimExpr>(op->annotations["use_2cta"]);
      // Bool in TVM is a subclass of IntImm, so only check IntImm.
      if (const auto *i = val.as<IntImmNode>()) {
        if (i->value != 0) {
          tmem_call_ann.Set("use_2cta", IntImm(DataType::Int(32), 1));
        }
      }
    }

    /*
    Replace the tmem buffers with one shared address word per physical
    allocation, and allocate each of those once:
        tmem_buf0 = T.alloc_buffer((128, 384), "float32", scope="shared.tmem")
        tmem_buf1 = T.alloc_buffer((128, 4), "uint32", scope="shared.tmem")

    into:
        tmem_buf0 = T.alloc_buffer((1,), "uint32", scope="shared")

        if tx // 32 == 0:
          T.ptx_init_tensor_memory(tmem_buf0[0], 512)

    where tmem_buf1 shares tmem_buf0's allocation and is addressed as
    tmem_buf0[0] + 384.  See PlanTmemArenas for which buffers get packed.
    */
    // 1. plan the physical allocations
    std::vector<int> num_cols_required;
    std::vector<bool> packable;
    for (const Buffer &buffer : tmem_buffers) {
      num_cols_required.push_back(GetNumB32ColsRequired(buffer));
      packable.push_back(!explicit_deallocs.count(buffer->data));
    }
    auto [arenas, placements] = PlanTmemArenas(num_cols_required, packable);

    // 2. one shared address word per allocation, named after the arena's first
    // (widest) member so that a block with a single TMEM buffer -- or with
    // buffers packing cannot help -- lowers exactly as it did before.
    std::vector<Buffer> arena_buffers;
    std::vector<Var> arena_base_vars;
    for (const TmemArena &arena : arenas) {
      const Buffer &base_buffer = tmem_buffers[arena.members.front()];
      Var arena_data(base_buffer->data->name_hint,
                     PointerType(PrimType(tmem_dtype_), "shared"));
      Buffer arena_buffer(arena_data, tmem_dtype_, Array<PrimExpr>({1}),
                          Array<PrimExpr>({1}), PrimExpr(0), base_buffer->name,
                          base_buffer->data_alignment,
                          base_buffer->offset_factor, base_buffer->buffer_type);
      arena_buffers.push_back(arena_buffer);
      Var arena_base(base_buffer->name + "_base", tmem_dtype_);
      arena_base_vars.push_back(arena_base);
      buffer_data_to_buffer_.Set(arena_data, arena_buffer);
      for (int member : arena.members) {
        const Buffer &buffer = tmem_buffers[member];
        var_remap_.Set(buffer->data, arena_data);
        buffer_remap_.Set(buffer, arena_buffer);
        tmem_col_offsets_[buffer->data] = placements[member].col_offset;
        tmem_num_cols_allocated_[buffer->data] = arena.num_cols_allocated;
        tmem_call_annotations_[buffer->data] = tmem_call_ann;
        tmem_base_var_remap_.Set(buffer->data, arena_base);
      }
    }

    // 3. swap the tmem buffers for their arenas' address words
    Array<Buffer> alloc_buffers;
    std::unordered_set<const BufferNode *> declared_arenas;
    for (const Buffer &buffer : op->alloc_buffers) {
      auto it = buffer_remap_.find(buffer);
      if (it == buffer_remap_.end()) {
        alloc_buffers.push_back(buffer);
        continue;
      }
      // Buffers sharing an allocation share its word: declare it once, where
      // the first of them used to be.
      Buffer arena_buffer = (*it).second;
      if (declared_arenas.insert(arena_buffer.get()).second) {
        alloc_buffers.push_back(arena_buffer);
      }
    }
    block.CopyOnWrite()->alloc_buffers = alloc_buffers;

    // 4. create one init & dealloc call per allocation
    std::vector<Stmt> init_mtmem_calls_;
    std::vector<Stmt> dealloc_tmem_calls_;
    int num_cols_total = 0;
    for (size_t i = 0; i < arenas.size(); ++i) {
      const TmemArena &arena = arenas[i];
      int num_cols_allocated = arena.num_cols_allocated;
      num_cols_total += num_cols_allocated;

      auto arena_access = arena_buffers[i].access_ptr(1, DataType::Handle(), 1,
                                                      PrimExpr(0), PrimExpr(1));
      auto alloc_call =
          Call(DataType::Handle(), tl::ptx_init_tensor_memory(),
               {arena_access, PrimExpr(num_cols_allocated)}, tmem_call_ann);
      init_mtmem_calls_.push_back(Evaluate(alloc_call));
      // A buffer that releases itself on every fallthrough path replaces the
      // dealloc at block end.  Packing leaves such buffers alone, so an
      // allocation released by hand never holds anything else.
      bool released_by_hand =
          arena.members.size() == 1 &&
          fallthrough_deallocs.count(tmem_buffers[arena.members.front()]->data);
      if (!released_by_hand) {
        auto dealloc_call =
            Call(DataType::Handle(), tl::ptx_deallocate_tensor_memory(),
                 {arena_access, PrimExpr(num_cols_allocated)}, tmem_call_ann);
        dealloc_tmem_calls_.push_back(Evaluate(dealloc_call));
      }
    }
    // Without an explicit tl.deallocate_tmem every allocation in the block is
    // live at once, so an over-budget kernel can be reported here instead of
    // failing inside tcgen05.alloc on device.
    if (explicit_deallocs.empty()) {
      ICHECK(num_cols_total <= kTmemNumColumns)
          << "This block allocates " << num_cols_total
          << " TMEM columns across " << arenas.size()
          << " tcgen05.alloc calls, but a CTA only has " << kTmemNumColumns
          << " columns; shrink the TMEM footprint or release a buffer earlier "
             "with T.deallocate_tmem";
    }
    // PTX forbids a tcgen05.alloc from asking for more columns than one issued
    // before it in the same CTA, so allocate the widest first.
    auto compare_by_num_cols_desc = [&](const Stmt &a, const Stmt &b) {
      auto call_a = a.as<EvaluateNode>()->value.as<CallNode>();
      auto call_b = b.as<EvaluateNode>()->value.as<CallNode>();
      auto num_cols_a = call_a->args[1].as<IntImmNode>()->value;
      auto num_cols_b = call_b->args[1].as<IntImmNode>()->value;
      return num_cols_a > num_cols_b;
    };
    std::sort(init_mtmem_calls_.begin(), init_mtmem_calls_.end(),
              compare_by_num_cols_desc);

    Array<Stmt> new_body;
    ICHECK(target_.defined()) << "LowerSharedTmem requires a bound target";
    auto warp_size = TargetCudaGetWarpSize(target_);
    auto thread_var_div_warp_size =
        FloorDiv(thread_var_->var, IntImm(thread_var_->var->dtype, warp_size));
    new_body.push_back(IfThenElse(EQ(thread_var_div_warp_size, 0),
                                  init_mtmem_calls_.size() > 1
                                      ? SeqStmt(init_mtmem_calls_)
                                      : init_mtmem_calls_.back(),
                                  Stmt()));
    new_body.push_back(
        Evaluate(Call(DataType::Handle(), builtin::tvm_storage_sync(),
                      {StringImm("shared")})));
    // Cache each arena's shared address word for ordinary TMEM address
    // expressions; raw PTX intrinsics retain their shared-pointer contract.
    for (size_t i = 0; i < arena_buffers.size(); ++i) {
      new_body.push_back(
          Bind(arena_base_vars[i], BufferLoad(arena_buffers[i], {Integer(0)})));
    }
    new_body.push_back(block->body);
    if (!dealloc_tmem_calls_.empty()) {
      if (tmem_call_ann.find("use_2cta") != tmem_call_ann.end()) {
        new_body.push_back(
            Evaluate(Call(DataType::Handle(), tl::cluster_sync(), {})));
      }
      new_body.push_back(IfThenElse(EQ(thread_var_div_warp_size, 0),
                                    dealloc_tmem_calls_.size() > 1
                                        ? SeqStmt(dealloc_tmem_calls_)
                                        : dealloc_tmem_calls_.back(),
                                    Stmt()));
    }

    auto block_ptr = block.CopyOnWrite();
    block_ptr->annotations.erase(attr::kLayoutMap);
    block_ptr->body = SeqStmt(new_body);

    return StmtExprMutator::VisitStmt_(block.get());
  }

  PrimExpr GetTmemOffset(const Buffer &buffer, const Array<PrimExpr> &indices) {
    // LowerTileOp MATERIALIZED the buffer's inferred layout: it rewrote the
    // allocation to the layout's physical shape and pushed every address
    // token through Layout::Forward.  Whatever the fragment's structure --
    // interleaved datapaths (PTX Layout F), weight-stationary N folding
    // (Layouts E/G), 2SM shards, leading batch modes -- `indices` here are
    // always the resulting physical (datapath, value-column) pair; this pass
    // only encodes them into the hardware word.  A TMEM buffer that skipped
    // layout inference would still be logical, which the rank check below
    // (and the 2-D allocation check above) rejects.
    ICHECK_EQ(indices.size(), 2U)
        << "TMEM address for " << buffer->name
        << " must be a (datapath, value-column) coordinate";
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-addressing
    PrimExpr result = indices[0] << 16 |
                      ValueColumnOffsetToB32Column(indices[1], buffer->dtype);
    return result;
  }

  /*!
   * \brief GetTmemOffset, shifted to where the buffer starts in its arena.
   *
   * A TMEM address is datapath<<16 | b32_column, so adding a column offset to
   * the encoded coordinate lands in the same datapath of a later column -- the
   * arena's base address plus this is the buffer's own base address.
   */
  PrimExpr GetArenaTmemOffset(const Buffer &buffer,
                              const Array<PrimExpr> &indices) {
    PrimExpr offset = GetTmemOffset(buffer, indices);
    int col_offset = GetArenaColOffset(buffer->data);
    if (col_offset == 0) {
      return offset;
    }
    return offset + IntImm(offset.dtype(), col_offset);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    // Translate tmem[datapath, value_col] to tmem[0] + tmem_offset
    // Where
    // - (datapath, value_col) is the physical address in the tmem buffer
    // - tmem[0] is the base address of the allocation holding the buffer
    // - tmem_offset = datapath<<16 | b32_col, with b32_col converted from
    //   the typed value-column coordinate and shifted by the buffer's column
    //   offset when it shares that allocation with others
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto buffer = load->buffer;
    auto indices = load->indices;

    if (buffer_remap_.count(buffer)) {
      return tmem_base_var_remap_.at(buffer->data) +
             GetArenaTmemOffset(buffer, indices);
    } else if (var_remap_.count(buffer->data)) {
      return tmem_base_var_remap_.at(buffer->data) +
             GetArenaTmemOffset(buffer, indices);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto buffer = store->buffer;
    ICHECK(buffer.scope() != "shared.tmem")
        << "We should never directly store data into tmem!";
    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::deallocate_tmem())) {
      ICHECK_EQ(op->args.size(), 1U);
      Var buffer_data = Downcast<Var>(op->args[0]);
      auto num_cols_it = tmem_num_cols_allocated_.find(buffer_data);
      ICHECK(num_cols_it != tmem_num_cols_allocated_.end())
          << "tl.deallocate_tmem expects a TMEM buffer allocated in the same "
             "or an enclosing block";
      ICHECK(buffer_data_to_buffer_.count(buffer_data))
          << "TMEM buffer for tl.deallocate_tmem is not tracked";
      Buffer old_buffer = buffer_data_to_buffer_.at(buffer_data);
      ICHECK(buffer_remap_.count(old_buffer))
          << "TMEM buffer for tl.deallocate_tmem has not been remapped";
      Buffer new_buffer = buffer_remap_[old_buffer];
      auto new_buffer_access = new_buffer.access_ptr(1, DataType::Handle(), 1,
                                                     PrimExpr(0), PrimExpr(1));

      Map<String, ObjectRef> ann;
      auto ann_it = tmem_call_annotations_.find(buffer_data);
      if (ann_it != tmem_call_annotations_.end()) {
        ann = ann_it->second;
      }
      return Call(DataType::Handle(), tl::ptx_deallocate_tensor_memory(),
                  {new_buffer_access, PrimExpr(num_cols_it->second)}, ann);
    }
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5U);
      Var buffer_data = Downcast<Var>(op->args[1]);
      if (!var_remap_.count(buffer_data)) {
        return StmtExprMutator::VisitExpr_(op);
      }
      // A pointer to the address word carries no column offset, so a buffer
      // sharing an allocation cannot be reached this way.  See
      // VisitExpr_(VarNode) for the same argument.
      CheckAddressableWithoutColOffset(buffer_data);
      Var new_data = var_remap_[buffer_data];
      return Call(
          op->dtype, op->op,
          {op->args[0], new_data, op->args[2], op->args[3], op->args[4]});
    }
    if (HasPackedTmemOperand(op)) {
      ICHECK(TakesTmemBaseOffsetPairs(op->op))
          << op->op
          << " names a TMEM buffer that shares a physical allocation, but only "
             "the tl.ptx_tcgen05_* intrinsics pass tensor memory as a "
             "(base, offset) pair this pass can shift";
      return RebaseTmemOperands(op);
    }
    auto expr = StmtExprMutator::VisitExpr_(op);
    return expr;
  }

  /*!
   * \brief Whether a call names a TMEM buffer that starts inside an allocation.
   *
   * Calls that only name buffers at column 0 need no rewriting, so they keep
   * the exact expression -- source span included -- that they had before.
   */
  bool HasPackedTmemOperand(const CallNode *op) const {
    for (const PrimExpr &arg : op->args) {
      const auto *var = arg.as<VarNode>();
      if (var != nullptr && GetArenaColOffset(GetRef<Var>(var)) != 0) {
        return true;
      }
    }
    return false;
  }

  /*!
   * \brief Whether an op takes its tensor-memory operands as (base, offset).
   *
   * Every tl.ptx_tcgen05_* intrinsic passes a tensor-memory operand as the
   * buffer's base-address Var immediately followed by that operand's address
   * offset: see ptx_tcgen05_mma_ss, _mma_ts, _mma_blockscaled_ss and
   * cp_warpx4 in codegen_cuda.cc, which all emit
   * `*(uint32_t*)base + offset`.  Packing folds a buffer's column offset into
   * the second half of each such pair.
   */
  static bool TakesTmemBaseOffsetPairs(const RelaxExpr &op) {
    const auto *op_node = op.as<OpNode>();
    if (op_node == nullptr) {
      return false;
    }
    const std::string name = op_node->name;
    return name.rfind("tl.ptx_tcgen05_", 0) == 0;
  }

  /*!
   * \brief Add the arena column offsets to one tl.ptx_tcgen05_* call's
   * operands.
   */
  PrimExpr RebaseTmemOperands(const CallNode *op) {
    Array<PrimExpr> args;
    for (size_t i = 0; i < op->args.size(); ++i) {
      const auto *var = op->args[i].as<VarNode>();
      if (var == nullptr || !var_remap_.count(GetRef<Var>(var))) {
        args.push_back(VisitExpr(op->args[i]));
        continue;
      }
      Var buffer_data = GetRef<Var>(var);
      ICHECK_LT(i + 1, op->args.size())
          << op->op << " passes TMEM buffer " << buffer_data
          << " without the address offset that has to follow it";
      PrimExpr offset = VisitExpr(op->args[i + 1]);
      ICHECK(offset.dtype().is_int() || offset.dtype().is_uint())
          << op->op << " expects an integer TMEM address offset after buffer "
          << buffer_data << ", got " << offset << " of type " << offset.dtype();
      int col_offset = GetArenaColOffset(buffer_data);
      args.push_back(var_remap_[buffer_data]);
      args.push_back(col_offset == 0
                         ? offset
                         : offset + IntImm(offset.dtype(), col_offset));
      // The offset argument was consumed together with its base.
      ++i;
    }
    return Call(op->dtype, op->op, args, op->annotations, op->span);
  }

  /*!
   * \brief Reject a base address that no column offset can be attached to.
   *
   * A buffer that starts partway into a shared allocation is only addressable
   * through a construct this pass knows how to shift.  Reaching one of the
   * remaining paths would silently address the start of the allocation --
   * another buffer's data -- so report it instead.
   */
  void CheckAddressableWithoutColOffset(const Var &buffer_data) const {
    int col_offset = GetArenaColOffset(buffer_data);
    ICHECK_EQ(col_offset, 0)
        << "TMEM buffer " << buffer_data
        << " shares a physical TMEM allocation and starts at column "
        << col_offset
        << " of it, but its base address reaches an operation LowerSharedTmem "
           "cannot offset; teach RebaseTmemOperands about that operation";
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = GetRef<Var>(op);
    if (var_remap_.count(var)) {
      // The base address escaped the (base, offset) rewrite above, so nothing
      // downstream adds this buffer's column offset.
      CheckAddressableWithoutColOffset(var);
      return var_remap_[var];
    }
    return var;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_var_ = iv;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // Datatypes for tmem
  const DataType tmem_dtype_ = DataType::UInt(32);
  // This is a workaround for cpu backend,
  // we need to define a thread_var for the serial loop.
  IterVar thread_var_;
  Target target_;
  Map<Var, Var> var_remap_;
  Map<Var, Var> tmem_base_var_remap_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Buffer> buffer_remap_;
  // Mapping from data Var of a Buffer to Buffer, for lookup
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual>
      tmem_num_cols_allocated_;
  // b32 columns between a TMEM buffer and the base of the allocation it shares
  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> tmem_col_offsets_;
  std::unordered_map<Var, Map<String, ObjectRef>, ObjectPtrHash, ObjectPtrEqual>
      tmem_call_annotations_;
  Map<Buffer, Layout> layout_map_;
};

PrimFunc LowerSharedTmem(PrimFunc f) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target.defined()) << "LowerSharedTmem: Require the target attribute";
  f.CopyOnWrite()->body = SharedTmemRewriter::Rewrite(f->body, target.value());
  return f;
}

namespace transform {
using namespace tirx::transform;

tvm::transform::Pass LowerSharedTmem() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return tl::LowerSharedTmem(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerSharedTmem", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.cuda.transform.LowerSharedTmem", LowerSharedTmem);
}

} // namespace transform
} // namespace tl
} // namespace tvm
