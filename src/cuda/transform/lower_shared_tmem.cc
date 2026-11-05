/*!
 * \file lower_shared_tmem.cc
 * \brief Lower native physical TMEM tensors to shared address words.
 *
 * LowerTileOp has already materialized (datapath, value-column) coordinates.
 * This pass allocates b32 column arenas and preserves both address-token and
 * base/offset intrinsic contracts, including aliases and explicit release.
 */
#include "cuda/op/builtin.h"
#include "cuda/target_utils.h"
#include "ir/analyzer.h"
#include "op/native_intrinsic.h"
#include "transform/common/native_layout_utils.h"

#include "ir/stmt_functor.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;
using namespace prim;
using VarSet = std::unordered_set<const VarObj *>;

bool IsTmem(const TensorVar &tensor) {
  return tensor.scope().name() == "shared.tmem";
}

// Layout and safe-value maps described the logical tensor before LowerTileOp
// and LegalizeSafeMemoryAccess. They cannot be attached to its uint32 address
// word: that would describe the wrong rank, dtype and storage object.
Map<Any, Any> WithoutTmemEntries(const Any &value, bool *changed) {
  Map<Any, Any> result;
  for (const auto &[key, item] : value.cast<Map<Any, Any>>()) {
    if (auto tensor = key.as<TensorVar>(); tensor && IsTmem(*tensor))
      *changed = true;
    else
      result.Set(key, item);
  }
  return result;
}

bool ConsumeTmemMetadata(Map<String, Any> *metadata) {
  bool changed = false;
  for (const char *key : {"layout_map", "safe_value_map"}) {
    if (auto value = metadata->Get(key)) {
      bool removed = false;
      auto remaining = WithoutTmemEntries(*value, &removed);
      if (removed) {
        changed = true;
        if (remaining.empty())
          metadata->erase(key);
        else
          metadata->Set(key, remaining);
      }
    }
  }
  return changed;
}

class SourceBindings : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;
  Optional<VisitInterrupt> Visit_(const BindObj *binding) final {
    bindings.Set(binding->var, binding->value);
    return StmtExprVisitor::Visit_(binding);
  }
  Map<Var, Expr> bindings;
};

// Recover only pointer/tensor provenance, never an integer obtained by reading
// an address word. The latter is already an encoded hardware address.
class TmemProvenance {
public:
  explicit TmemProvenance(Map<Var, Expr> bindings)
      : bindings_(std::move(bindings)) {}

  std::optional<TensorVar> Root(const Expr &value) const {
    std::unordered_set<const VarObj *> seen;
    return Root(value, &seen);
  }

  std::pair<VarSet, bool> Fallthrough(const Stmt &statement) const {
    if (const auto *sequence = statement.as<SeqStmtObj>()) {
      VarSet result;
      for (const Stmt &child : sequence->seq) {
        auto [released, fallthrough] = Fallthrough(child);
        if (!fallthrough)
          return {{}, false};
        result.insert(released.begin(), released.end());
      }
      return {std::move(result), true};
    }
    if (const auto *branch = statement.as<IfObj>()) {
      auto [then_released, then_fallthrough] = Fallthrough(branch->then_case);
      auto [else_released, else_fallthrough] =
          branch->else_case ? Fallthrough(branch->else_case.value())
                            : std::pair<VarSet, bool>{{}, true};
      VarSet result;
      if (then_fallthrough)
        result.insert(then_released.begin(), then_released.end());
      if (else_fallthrough)
        result.insert(else_released.begin(), else_released.end());
      // Explicit deallocation is warp-managed. Preserve the source contract:
      // an explicit release on a surviving path (e.g. tx < 32) suppresses the
      // automatic first-warp tail. Do not add a duplicate release on that warp.
      return {std::move(result), then_fallthrough || else_fallthrough};
    }
    if (const auto *loop = statement.as<ForObj>())
      return Fallthrough(loop->body);
    if (const auto *region = statement.as<RegionStmtObj>())
      return Fallthrough(region->body);
    if (const auto *evaluate = statement.as<EvaluateObj>()) {
      if (const auto *call = evaluate->value.as<CallObj>()) {
        if (call->op.same_as(gpu_thread_return_op()))
          return {{}, false};
        if (call->op.same_as(deallocate_tmem())) {
          auto args = NativeIntrinsicOperands(call);
          TVM_FFI_CHECK_EQ(args.size(), 1U, ValueError);
          auto tensor = Root(args[0]);
          TVM_FFI_CHECK(tensor, ValueError)
              << "tl.deallocate_tmem requires native TMEM backing";
          return {{tensor.value().get()}, true};
        }
      }
    }
    return {{}, true};
  }

  VarSet Explicit(const Stmt &statement) const {
    VarSet result;
    VisitNativePostOrder(statement, [&](const ObjectRef &object) {
      if (const auto *call = object.as<CallObj>();
          call && call->op.same_as(deallocate_tmem())) {
        auto args = NativeIntrinsicOperands(call);
        TVM_FFI_CHECK_EQ(args.size(), 1U, ValueError);
        auto tensor = Root(args[0]);
        TVM_FFI_CHECK(tensor, ValueError)
            << "tl.deallocate_tmem requires native TMEM backing";
        result.insert(tensor.value().get());
      }
    });
    return result;
  }

  const Map<Var, Expr> &bindings() const { return bindings_; }

private:
  std::optional<TensorVar>
  Root(const Expr &value, std::unordered_set<const VarObj *> *seen) const {
    if (auto variable = value.as<Var>()) {
      if (!seen->insert(variable->get()).second)
        return std::nullopt;
      auto binding = bindings_.Get(*variable);
      if (auto tensor = variable->as<TensorVar>(); tensor && IsTmem(*tensor)) {
        if (!binding)
          return tensor;
        if (const auto *call = binding->as<CallObj>()) {
          if (call->op.same_as(alloc_tensor_op()))
            return tensor;
          if (call->op.same_as(decl_tensor_op()))
            return Root(call->args[0], seen);
        }
      }
      return binding ? Root(*binding, seen) : std::nullopt;
    }
    if (const auto *load = value.as<TensorLoadObj>())
      return Root(load->source, seen);
    if (const auto *call = value.as<CallObj>()) {
      if (call->op.same_as(tensor_data_ptr_op()) ||
          call->op.same_as(reinterpret_op()) ||
          call->op.same_as(ptr_byte_offset_op()) ||
          call->op.same_as(address_of_op()) ||
          call->op.same_as(Op::Get("tl.access_ptr")) ||
          call->op.same_as(Op::Get("tl.ptr_access")))
        return Root(call->args[0], seen);
    }
    return std::nullopt;
  }
  Map<Var, Expr> bindings_;
};

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
  TVM_FFI_CHECK_EQ(num_cols.size(), packable.size(), ValueError);

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

struct TmemReference {
  TensorVar tensor;
  TensorVar word;
  PrimExpr offset;
  int columns;
  bool two_cta;
};

class SharedTmemLowerer : public StmtExprMutatorWithAnalyzer {
public:
  using StmtExprMutatorWithAnalyzer::Mutate_;
  SharedTmemLowerer(sym::AnalyzerObj *analyzer, Target target,
                    Map<Var, Expr> bindings)
      : StmtExprMutatorWithAnalyzer(analyzer), target_(std::move(target)),
        provenance_(std::move(bindings)) {}

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *region,
                            InplaceMode mode) final {
    auto saved_thread = thread_;
    bool saved_two_cta = two_cta_;
    if (region->op.same_as(launch_thread_op()) &&
        region->args[0].as_or_throw<StringImm>()->value == "threadIdx.x")
      thread_ = region->body_params[0].as_or_throw<PrimVar>();
    if (region->op->name == "tl.attr") {
      auto fields = ReadNativeAnnotations(region);
      const String key = fields.at("key").cast<String>();
      if (key == "layout_map" || key == "safe_value_map") {
        bool removed = false;
        auto remaining = WithoutTmemEntries(fields.at("value"), &removed);
        if (removed) {
          SeqStmt body =
              Mutate(region->body, mode).ValueOrUnchanged(region->body);
          if (remaining.empty())
            return body;
          return MakeNativeAttribute(fields.at("node"), key, remaining, body,
                                     region->loc);
        }
      }
      if (key == "tl.kernel") {
        auto metadata = fields.at("value").cast<Map<String, Any>>();
        two_cta_ = ReadFlag(metadata.Get("use_2cta"));
        if (ConsumeTmemMetadata(&metadata)) {
          SeqStmt body =
              Mutate(region->body, mode).ValueOrUnchanged(region->body);
          thread_ = saved_thread;
          two_cta_ = saved_two_cta;
          return MakeNativeAttribute(fields.at("node"), key, metadata, body,
                                     region->loc);
        }
      } else if (key == "use_2cta") {
        two_cta_ = ReadFlag(fields.at("value"));
      }
    }
    auto result = StmtExprMutatorWithAnalyzer::Mutate_(region, mode);
    thread_ = saved_thread;
    two_cta_ = saved_two_cta;
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *sequence,
                            InplaceMode mode) final {
    const auto saved_references = references_;
    auto explicit_releases = provenance_.Explicit(GetRef<Stmt>(sequence));
    auto [tail_releases, falls_through] =
        provenance_.Fallthrough(GetRef<Stmt>(sequence));
    (void)falls_through;
    std::vector<TensorVar> tensors;
    size_t first_allocation = sequence->seq.size();
    for (size_t i = 0; i < sequence->seq.size(); ++i) {
      const auto *binding = sequence->seq[i].as<BindObj>();
      if (!binding)
        continue;
      auto tensor = binding->var.as<TensorVar>();
      const auto *call = binding->value.as<CallObj>();
      if (tensor && IsTmem(*tensor) && call &&
          call->op.same_as(alloc_tensor_op())) {
        TVM_FFI_CHECK(call->args.empty(), ValueError)
            << "A native TMEM allocation takes no operands";
        tensors.push_back(*tensor);
        first_allocation = std::min(first_allocation, i);
      } else if (auto variable = binding->var.as<PrimVar>()) {
        if (auto expression = binding->value.as<PrimExpr>();
            expression && SideEffect(*expression) <= CallEffectKind::kPure)
          BindAnalyzer(GetSyncedAnalyzer(), *variable, *expression, true);
      }
    }
    std::vector<int> required;
    std::vector<bool> packable;
    for (const TensorVar &tensor : tensors) {
      required.push_back(RequiredColumns(tensor));
      packable.push_back(!explicit_releases.count(tensor.get()) &&
                         !HasOpaqueAllocationAttrs(tensor));
    }
    auto [arenas, placements] = PlanTmemArenas(required, packable);
    Array<Stmt> definitions;
    std::vector<Stmt> initializers;
    Array<Stmt> finalizers;
    int total_columns = 0;
    for (size_t i = 0; i < arenas.size(); ++i) {
      const TmemArena &arena = arenas[i];
      const TensorVar &first = tensors[arena.members.front()];
      int alignment = first.ty()->data_alignment;
      for (int member : arena.members)
        alignment = std::max(alignment, tensors[member].ty()->data_alignment);
      TensorVar word(first->name,
                     TensorType({IntImm(PrimType::Int(32), 1)},
                                PrimType::UInt(32),
                                runtime::MemoryScope("shared"), alignment),
                     first->loc);
      auto allocation_attrs = AllocationAttrs(first)->dict;
      for (int member : arena.members)
        if (NeedsVolatileWord(tensors[member]))
          allocation_attrs.Set(tirx::attr::kVolatile.name, true);
      definitions.push_back(
          Bind(word,
               Call(word->ty, alloc_tensor_op(), {},
                    DictAttrs(allocation_attrs), {word->ty}, first->loc),
               first->loc));
      for (int member : arena.members) {
        references_.insert_or_assign(
            tensors[member].get(),
            TmemReference{
                tensors[member], word,
                IntImm(PrimType::UInt(32), placements[member].col_offset),
                arena.num_cols_allocated, two_cta_});
      }
      initializers.push_back(Evaluate(
          AllocationCall(ptx_init_tensor_memory(), word,
                         arena.num_cols_allocated, two_cta_, first->loc),
          first->loc));
      const bool explicit_tail =
          arena.members.size() == 1 && tail_releases.count(first.get());
      if (!explicit_tail)
        finalizers.push_back(Evaluate(
            AllocationCall(ptx_deallocate_tensor_memory(), word,
                           arena.num_cols_allocated, two_cta_, first->loc),
            first->loc));
      total_columns += arena.num_cols_allocated;
    }
    if (explicit_releases.empty())
      TVM_FFI_CHECK_LE(total_columns, kTmemNumColumns, ValueError)
          << "This scope allocates " << total_columns << " TMEM columns across "
          << arenas.size()
          << " tcgen05.alloc calls, but a CTA only has 512 columns";
    std::stable_sort(
        initializers.begin(), initializers.end(),
        [](const Stmt &a, const Stmt &b) {
          const auto aa = NativeIntrinsicOperands(
              a.as_or_throw<Evaluate>()->value.as_or_throw<Call>().get());
          const auto bb = NativeIntrinsicOperands(
              b.as_or_throw<Evaluate>()->value.as_or_throw<Call>().get());
          return aa[1].as_or_throw<IntImm>()->value >
                 bb[1].as_or_throw<IntImm>()->value;
        });
    Array<Stmt> output;
    for (size_t i = 0; i < sequence->seq.size(); ++i) {
      if (i == first_allocation && !initializers.empty()) {
        for (const Stmt &definition : definitions)
          output.push_back(definition);
        output.push_back(
            FirstWarp(Array<Stmt>(initializers.begin(), initializers.end()),
                      sequence->loc));
        output.push_back(
            Evaluate(Call(PrimType::Void(), gpu_storage_sync_op(),
                          {StringImm("shared")}, {}, {}, sequence->loc),
                     sequence->loc));
      }
      output.push_back(
          Mutate(sequence->seq[i], mode).ValueOrUnchanged(sequence->seq[i]));
    }
    if (!finalizers.empty()) {
      if (two_cta_)
        output.push_back(Evaluate(
            Call(PrimType::Void(), cluster_sync(), {}, {}, {}, sequence->loc),
            sequence->loc));
      output.push_back(FirstWarp(finalizers, sequence->loc));
    }
    references_ = saved_references;
    return SeqStmt(output, sequence->loc);
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *evaluate,
                            InplaceMode mode) final {
    const auto *call = evaluate->value.as<CallObj>();
    if (call && call->op.same_as(Op::Get("tl.annotation"))) {
      auto metadata = ReadNativeAnnotations(call);
      if (ConsumeTmemMetadata(&metadata)) {
        if (metadata.empty())
          return SeqStmt({}, evaluate->loc);
        auto replacement = MakeNativeAnnotation(metadata, evaluate->loc)
                               .as_or_throw<Evaluate>();
        return StmtExprMutatorWithAnalyzer::Mutate_(replacement.get(), mode)
            .ValueOrUnchanged(replacement);
      }
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(evaluate, mode);
  }

  UnchangedOr<Stmt> Mutate_(const BindObj *binding, InplaceMode mode) final {
    if (auto tensor = binding->var.as<TensorVar>(); tensor && IsTmem(*tensor)) {
      const auto *call = binding->value.as<CallObj>();
      if (call && call->op.same_as(alloc_tensor_op())) {
        TVM_FFI_CHECK(references_.count(tensor.value().get()), ValueError);
        return SeqStmt({}, binding->loc);
      }
      Expr source = call && call->op.same_as(decl_tensor_op()) ? call->args[0]
                                                               : binding->value;
      auto alias = ResolvePointer(source);
      TVM_FFI_CHECK(alias, ValueError)
          << "TMEM view must refer to an allocated native TMEM tensor";
      alias->tensor = *tensor;
      references_.insert_or_assign(tensor.value().get(), *alias);
      return SeqStmt({}, binding->loc);
    }
    if (binding->var->ty.as<PtrTypeObj>() && ResolvePointer(binding->value))
      return SeqStmt({}, binding->loc);
    return StmtExprMutatorWithAnalyzer::Mutate_(binding, mode);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj *store,
                            InplaceMode mode) final {
    if (auto tensor = store->dest.as<TensorVar>(); tensor && IsTmem(*tensor))
      TVM_FFI_THROW(ValueError) << "TMEM stores must be lowered to tcgen05 "
                                   "instructions before LowerSharedTmem";
    return StmtExprMutatorWithAnalyzer::Mutate_(store, mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *load,
                                InplaceMode mode) final {
    auto tensor = load->source.as<TensorVar>();
    if (tensor) {
      auto found = references_.find(tensor.value().get());
      if (found != references_.end()) {
        Array<PrimExpr> indices;
        for (const PrimExpr &value : load->indices)
          indices.push_back(Mutate(value, mode).ValueOrUnchanged(value));
        PrimExpr offset = GetSyncedAnalyzer()->Simplify(
            found->second.offset + EncodeCoordinates(*tensor, indices));
        return GetSyncedAnalyzer()->Simplify(
            TensorLoad(found->second.word, {IntImm(PrimType::Int(32), 0)},
                       load->loc) +
            offset);
      }
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(load, mode);
  }

  UnchangedOr<Expr> Mutate_(const CallObj *call, InplaceMode mode) final {
    if (call->op.same_as(deallocate_tmem())) {
      auto args = NativeIntrinsicOperands(call);
      TVM_FFI_CHECK_EQ(args.size(), 1U, ValueError);
      auto reference = ResolvePointer(args[0]);
      TVM_FFI_CHECK(reference, ValueError)
          << "tl.deallocate_tmem expects a TMEM allocation in the same or "
             "enclosing scope";
      return AllocationCall(ptx_deallocate_tensor_memory(), reference->word,
                            reference->columns, reference->two_cta, call->loc);
    }
    auto op = call->op.as<Op>();
    if (op && op.value()->name.find("tl.ptx_tcgen05_") == 0) {
      auto operands = NativeIntrinsicOperands(call);
      Array<Expr> rewritten;
      bool changed = false;
      for (size_t i = 0; i < operands.size(); ++i) {
        if (auto reference = ResolvePointer(operands[i])) {
          TVM_FFI_CHECK_LT(i + 1, operands.size(), ValueError)
              << op.value()->name << " must pass a TMEM (base, offset) pair";
          PrimExpr offset = operands[++i].as_or_throw<IntExpr>();
          offset = Mutate(offset, mode).ValueOrUnchanged(offset);
          rewritten.push_back(WordPointer(reference->word));
          rewritten.push_back(GetSyncedAnalyzer()->Simplify(
              offset + cast(reference->offset, offset.ty())));
          changed = true;
        } else {
          Expr value = Mutate(operands[i], mode).ValueOrUnchanged(operands[i]);
          changed |= !value.same_as(operands[i]);
          rewritten.push_back(value);
        }
      }
      if (!changed)
        return Unchanged();
      for (size_t i = operands.size(); i < call->args.size(); ++i)
        rewritten.push_back(
            Mutate(call->args[i], mode).ValueOrUnchanged(call->args[i]));
      return Call(call->ty, call->op, rewritten, call->attrs, call->ty_args,
                  call->loc);
    }
    if (auto reference = ResolvePointer(GetRef<Expr>(call)))
      return UnshiftedPointer(*reference);
    return StmtExprMutatorWithAnalyzer::Mutate_(call, mode);
  }

  UnchangedOr<Expr> Mutate_(const VarObj *variable, InplaceMode mode) final {
    if (def_region_kind() == kTVMFFIDefRegionKindNone) {
      if (auto tensor = GetRef<Var>(variable).as<TensorVar>();
          tensor && IsTmem(*tensor))
        TVM_FFI_THROW(ValueError)
            << "TMEM TensorVar must be consumed by a memory-address intrinsic "
               "after layout lowering";
      if (auto reference = ResolvePointer(GetRef<Expr>(variable)))
        return UnshiftedPointer(*reference);
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(variable, mode);
  }

private:
  static bool ReadFlag(std::optional<Any> value) {
    if (!value)
      return false;
    if (auto boolean = value->as<bool>())
      return *boolean;
    if (auto integer = value->as<int64_t>())
      return *integer != 0;
    if (const auto *integer = value->as<IntImmObj>())
      return integer->value != 0;
    TVM_FFI_THROW(ValueError) << "use_2cta requires a constant integer or bool";
  }

  DictAttrs AllocationAttrs(const TensorVar &tensor) const {
    auto call = provenance_.bindings().at(tensor).as_or_throw<Call>();
    auto attrs = call->attrs ? call->attrs.value().as_or_throw<DictAttrs>()
                             : DictAttrs();
    if (!attrs->dict.count("tl.source_block_alloc"))
      return attrs;
    const auto origin = attrs->dict.at("tl.source_block_alloc").as<IntImm>();
    TVM_FFI_CHECK(origin && origin.value()->value == 1, ValueError)
        << "tl.source_block_alloc must be the source allocation marker";
    // The frontend origin controls CPU allocation placement. TMEM's native
    // lexical allocation and explicit release already determine its arena
    // lifetime, so the lowered shared address word must not inherit it.
    auto values = attrs->dict;
    values.erase("tl.source_block_alloc");
    return DictAttrs(values);
  }

  bool HasOpaqueAllocationAttrs(const TensorVar &tensor) const {
    for (const auto &[key, value] : AllocationAttrs(tensor)->dict)
      if (key != tirx::attr::kVolatile.name)
        return true;
    return false;
  }

  bool NeedsVolatileWord(const TensorVar &tensor) const {
    for (const auto &[variable, expression] : provenance_.bindings()) {
      const auto *call = expression.as<CallObj>();
      if (!call || !call->attrs ||
          (!call->op.same_as(alloc_tensor_op()) &&
           !call->op.same_as(decl_tensor_op())))
        continue;
      auto attrs = call->attrs.value().as_or_throw<DictAttrs>();
      if (!attrs.GetAttr(tirx::attr::kVolatile).value_or(false))
        continue;
      auto root = provenance_.Root(variable);
      if (root && root->same_as(tensor))
        return true;
    }
    return false;
  }

  int ValueBits(const TensorVar &tensor) const {
    int bits = tensor.dtype().bits() * tensor.dtype().lanes();
    TVM_FFI_CHECK_GT(bits, 0, ValueError);
    return bits;
  }

  int RequiredColumns(const TensorVar &tensor) const {
    TVM_FFI_CHECK_EQ(tensor.shape().size(), 2U, ValueError)
        << "Native TMEM must have a materialized (datapath, value-column) "
           "shape";
    const auto rows =
        GetSyncedAnalyzer()->const_int_bound(tensor.shape()[0])->max_value;
    const auto values =
        GetSyncedAnalyzer()->const_int_bound(tensor.shape()[1])->max_value;
    TVM_FFI_CHECK(rows > 0 && rows <= kTmemNumDatapaths, ValueError)
        << "TMEM datapath extent must be provably in [1, 128]";
    const int bits = ValueBits(tensor);
    TVM_FFI_CHECK(values > 0 && values <= kTmemNumColumns * 32 / bits,
                  ValueError)
        << "TMEM allocation exceeds 512 b32 columns or has an unbounded "
           "value-column extent";
    return static_cast<int>((values * bits + 31) / 32);
  }

  PrimExpr EncodeCoordinates(const TensorVar &tensor,
                             const Array<PrimExpr> &indices) const {
    TVM_FFI_CHECK_EQ(indices.size(), 2U, ValueError)
        << "TMEM address requires a physical (datapath, value-column) "
           "coordinate";
    const auto i64 = PrimType::Int(64);
    const auto u32 = PrimType::UInt(32);
    PrimExpr bits = GetSyncedAnalyzer()->Simplify(
        cast(indices[1], i64) * IntImm(i64, ValueBits(tensor)));
    TVM_FFI_CHECK(GetSyncedAnalyzer()->CanProveEqual(
                      floormod(bits, IntImm(i64, 32)), IntImm(i64, 0)),
                  ValueError)
        << "TMEM address must start on a b32 column";
    return GetSyncedAnalyzer()->Simplify(
        (cast(indices[0], u32) << IntImm(u32, 16)) |
        cast(floordiv(bits, IntImm(i64, 32)), u32));
  }

  std::optional<TmemReference> ResolvePointer(const Expr &value) {
    std::unordered_set<const VarObj *> seen;
    return ResolvePointer(value, &seen);
  }

  std::optional<TmemReference>
  ResolvePointer(const Expr &value, std::unordered_set<const VarObj *> *seen) {
    if (auto variable = value.as<Var>()) {
      auto found = references_.find(variable->get());
      if (found != references_.end())
        return found->second;
      if (!seen->insert(variable->get()).second)
        return std::nullopt;
      auto binding = provenance_.bindings().Get(*variable);
      return binding ? ResolvePointer(*binding, seen) : std::nullopt;
    }
    const auto *call = value.as<CallObj>();
    if (!call)
      return std::nullopt;
    if (call->op.same_as(tensor_data_ptr_op()) ||
        call->op.same_as(reinterpret_op()))
      return ResolvePointer(call->args[0], seen);
    if (call->op.same_as(decl_tensor_op()))
      return ResolvePointer(call->args[0], seen);
    if (call->op.same_as(address_of_op()) ||
        call->op.same_as(Op::Get("tl.access_ptr"))) {
      const auto *load = call->args[0].as<TensorLoadObj>();
      if (!load)
        return std::nullopt;
      auto reference = ResolvePointer(load->source, seen);
      if (reference)
        reference->offset = GetSyncedAnalyzer()->Simplify(
            reference->offset +
            EncodeCoordinates(reference->tensor, load->indices));
      return reference;
    }
    if (call->op.same_as(ptr_byte_offset_op()) ||
        call->op.same_as(Op::Get("tl.ptr_access"))) {
      auto reference = ResolvePointer(call->args[0], seen);
      if (!reference)
        return std::nullopt;
      auto offset = call->args[1].as_or_throw<IntExpr>();
      const auto i64 = PrimType::Int(64);
      int bits_per_unit = 8;
      if (!call->op.same_as(ptr_byte_offset_op())) {
        auto element = call->args[0]
                           ->ty.as_or_throw<PtrType>()
                           ->element_type.as_or_throw<PrimType>();
        bits_per_unit = element.bits() * element.lanes();
      }
      PrimExpr bit_offset = cast(offset, i64) * IntImm(i64, bits_per_unit);
      PrimExpr value_bits = IntImm(i64, ValueBits(reference->tensor));
      TVM_FFI_CHECK(GetSyncedAnalyzer()->CanProveEqual(
                        floormod(bit_offset, value_bits), IntImm(i64, 0)),
                    ValueError)
          << "TMEM pointer offset cannot split a physical tensor element";
      PrimExpr elements = floordiv(bit_offset, value_bits);
      PrimExpr width = cast(reference->tensor.shape()[1], i64);
      reference->offset = GetSyncedAnalyzer()->Simplify(
          reference->offset +
          EncodeCoordinates(reference->tensor, {floordiv(elements, width),
                                                floormod(elements, width)}));
      return reference;
    }
    return std::nullopt;
  }

  Expr UnshiftedPointer(const TmemReference &reference) const {
    TVM_FFI_CHECK(GetSyncedAnalyzer()->CanProveEqual(
                      reference.offset, IntImm(PrimType::UInt(32), 0)),
                  ValueError)
        << "Packed or offset TMEM base requires a supported (base, offset) "
           "intrinsic";
    return WordPointer(reference.word);
  }

  static Expr WordPointer(const TensorVar &word) {
    return Call(word.ty()->DataPtrType(), tensor_data_ptr_op(), {word}, {}, {},
                word->loc);
  }

  static Call AllocationCall(Op op, const TensorVar &word, int columns,
                             bool two_cta, Location loc) {
    Array<Expr> args{WordPointer(word), IntImm(PrimType::Int(32), columns)};
    if (two_cta)
      return MakeAnnotatedCall(PrimType::Void(), op, args,
                               {{"use_2cta", IntImm(PrimType::Int(32), 1)}},
                               loc);
    return Call(PrimType::Void(), op, args, {}, {}, loc);
  }

  Stmt FirstWarp(const Array<Stmt> &statements, Location loc) const {
    TVM_FFI_CHECK(thread_, ValueError)
        << "LowerSharedTmem requires an enclosing threadIdx.x region";
    const int warp_size = TargetCudaGetWarpSize(target_);
    auto zero = IntImm(thread_.value().ty(), 0);
    PrimExpr selected =
        floordiv(*thread_, IntImm(thread_.value().ty(), warp_size)) == zero;
    return If(selected, SeqStmt(statements, loc), std::nullopt, loc);
  }

  Target target_;
  TmemProvenance provenance_;
  Optional<PrimVar> thread_;
  bool two_cta_{false};
  std::unordered_map<const VarObj *, TmemReference> references_;
};
} // namespace

namespace transform {
tvm::transform::Pass LowerSharedTmem() {
  auto rewrite = [](tirx::Function function, IRModule,
                    tvm::transform::PassContext) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto target = function->GetAttr<Target>(tvm::attr::kTarget);
    TVM_FFI_CHECK(target, ValueError)
        << "LowerSharedTmem requires a bound target";
    auto collector = make_object<SourceBindings>();
    collector->Visit(function->body);
    auto analyzer = CreateTileLangAnalyzer();
    auto lowerer = make_object<SharedTmemLowerer>(
        analyzer.get(), target.value(), collector->bindings);
    SeqStmt body = function->body;
    function.CopyOnWrite()->body = lowerer->Mutate(body).ValueOrUnchanged(body);
    // Cluster launch lowering may consume use_2cta only after this owner has
    // used it to choose the one- or two-CTA allocation/release instructions.
    return WithAttr(function, "tl.shared_tmem_lowered", true);
  };
  return tirx::transform::CreateFunctionPass(rewrite, 0, "tl.LowerSharedTmem");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.LowerSharedTmem",
                                   LowerSharedTmem);
}
} // namespace transform
} // namespace tvm::tl
