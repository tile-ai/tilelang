#include "compat/visit.h"
#include "cuda/op/builtin.h"
#include "cuda/target_utils.h"
#include "op/utils.h"
#include "transform/common/constr_visitor.h"

#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

struct TmemPlacement {
  int arena;
  int offset;
};

struct TmemArena {
  std::vector<size_t> members;
  int used = 0;
  int allocated = 32;
  bool open = true;
};

int AllocationColumns(int columns) {
  int result = 32;
  while (result < columns)
    result *= 2;
  return result;
}

std::pair<std::vector<TmemArena>, std::vector<TmemPlacement>>
PlanArenas(const std::vector<int> &columns, const std::vector<bool> &packable) {
  std::vector<size_t> order(columns.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
    return columns[left] > columns[right];
  });
  std::vector<TmemArena> arenas;
  std::vector<TmemPlacement> placements(columns.size());
  for (size_t index : order) {
    int alignment = columns[index] >= 32 ? 32 : 4;
    int selected = -1;
    int selected_offset = 0;
    int delta = AllocationColumns(columns[index]);
    if (packable[index]) {
      for (size_t candidate = 0; candidate < arenas.size(); ++candidate) {
        const TmemArena &arena = arenas[candidate];
        int offset = (arena.used + alignment - 1) / alignment * alignment;
        if (!arena.open || offset + columns[index] > 512)
          continue;
        int growth =
            AllocationColumns(offset + columns[index]) - arena.allocated;
        if (growth < delta) {
          selected = candidate;
          selected_offset = offset;
          delta = growth;
        }
      }
    }
    if (selected < 0) {
      selected = arenas.size();
      arenas.push_back({{}, 0, 32, packable[index]});
    }
    TmemArena &arena = arenas[selected];
    arena.members.push_back(index);
    arena.used = selected_offset + columns[index];
    arena.allocated = AllocationColumns(arena.used);
    placements[index] = {selected, selected_offset};
  }
  return {std::move(arenas), std::move(placements)};
}

using VarSet = std::unordered_set<const VarNode *>;

std::pair<VarSet, bool> FallthroughReleases(const Stmt &statement,
                                            const VarSet &required,
                                            VarSet released = {},
                                            int loop_depth = 0) {
  if (statement.as<ReturnNode>()) {
    for (const VarNode *tensor : required)
      TVM_FFI_CHECK(released.count(tensor), ValueError)
          << "Every early-return path must explicitly deallocate all live TMEM "
             "allocations";
    return {{}, false};
  }
  if (statement.as<BreakNode>() || statement.as<ContinueNode>()) {
    TVM_FFI_CHECK(loop_depth > 0, ValueError)
        << "Break or continue cannot escape a live TMEM allocation scope";
    return {{}, false};
  }
  if (const auto *evaluation = statement.as<EvaluateNode>()) {
    const auto *call = evaluation->value.as<CallNode>();
    if (call && call->op.same_as(deallocate_tmem()))
      released.insert(GetVarFromAccessPtr(call->args[0]).get());
    return {released, true};
  }
  if (const auto *sequence = statement.as<SeqStmtNode>()) {
    for (const Stmt &child : sequence->seq) {
      auto [next, falls] =
          FallthroughReleases(child, required, released, loop_depth);
      if (!falls)
        return {{}, false};
      released = std::move(next);
    }
    return {released, true};
  }
  if (const auto *region = statement.as<RegionStmtNode>())
    return FallthroughReleases(region->body, required, released, loop_depth);
  if (const auto *loop = statement.as<ForNode>())
    FallthroughReleases(loop->body, required, released, loop_depth + 1);
  if (const auto *loop = statement.as<WhileNode>())
    FallthroughReleases(loop->body, required, released, loop_depth + 1);
  if (const auto *conditional = statement.as<IfNode>()) {
    auto [left, left_falls] = FallthroughReleases(
        conditional->then_case, required, released, loop_depth);
    auto [right, right_falls] =
        conditional->else_case
            ? FallthroughReleases(*conditional->else_case, required, released,
                                  loop_depth)
            : std::pair<VarSet, bool>{released, true};
    if (!left_falls)
      return {right, right_falls};
    if (!right_falls)
      return {left, left_falls};
    VarSet both;
    for (const VarNode *variable : left)
      if (right.count(variable))
        both.insert(variable);
    return {both, true};
  }
  return {released, true};
}

struct TmemAddress {
  TensorVar word;
  TensorVar arena;
  int columns;
  DictAttrs attributes;
  Optional<TensorVar> live;
};

class SharedTmemRewriter : public StmtExprMutatorWithAnalyzer {
public:
  SharedTmemRewriter(sym::AnalyzerObj *analyzer, Target target)
      : StmtExprMutatorWithAnalyzer(analyzer), target_(std::move(target)) {}

private:
  using Base = StmtExprMutatorWithAnalyzer;

  PrimExpr ThreadIndex() const {
    PrimExpr result = 0;
    PrimExpr stride = 1;
    for (size_t dimension = 0; dimension < threads_.size(); ++dimension) {
      if (threads_[dimension]) {
        result = result + threads_[dimension].value() * stride;
        stride = stride * thread_extents_[dimension];
      }
    }
    return result;
  }

  Stmt Sync(Location location) const {
    return Evaluate(Call(std::nullopt, gpu_storage_sync_op(),
                         {StringImm("shared")}, std::nullopt, {}, location),
                    location);
  }

  Stmt AllocateWord(const TensorVar &tensor, Location location) const {
    return Bind(tensor,
                Call(tensor.ty(), alloc_tensor_op(), {}, DictAttrs(),
                     {tensor.ty()}, location),
                location);
  }

  Stmt Release(const TmemAddress &address, Location location) const {
    Array<Stmt> statements{
        Evaluate(Call(std::nullopt, ptx_deallocate_tensor_memory(),
                      {address.arena.data(), PrimExpr(address.columns)},
                      address.attributes, {}, location),
                 location)};
    if (address.live)
      statements.push_back(TensorStore(*address.live, {PrimExpr(0)},
                                       IntImm::Bool(false), location));
    Stmt result = SeqStmt(statements, location);
    if (address.live)
      result = If(TensorLoad(*address.live, {PrimExpr(0)}), result,
                  std::nullopt, location);
    return If(FloorDiv(ThreadIndex(), 32) == 0, result, std::nullopt, location);
  }

  bool VariesByThread(const PrimExpr &expression) const {
    PrimExpr simplified =
        analyzer_->Simplify(Substitute(expression, scalar_values_));
    if (SideEffect(simplified) != CallEffectKind::kPure)
      return true;
    for (const Var &variable : tvm::UndefinedVars(simplified))
      for (const auto &thread : threads_)
        if (thread && variable.same_as(*thread))
          return true;
    return false;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                            InplaceMode mode) final {
    auto saved_threads = threads_;
    auto saved_extents = thread_extents_;
    DictAttrs saved_attributes = attributes_;
    if (node->op.same_as(launch_thread_op())) {
      auto tag = node->args[0].as_or_throw<StringImm>()->value;
      if (tag.find("threadIdx.") == 0) {
        int dimension = tag.data()[tag.size() - 1] - 'x';
        TVM_FFI_CHECK(dimension >= 0 && dimension < 3, ValueError)
            << "Invalid thread axis";
        threads_[dimension] = node->body_params[0].as_or_throw<PrimVar>();
        thread_extents_[dimension] = node->args[1].as_or_throw<PrimExpr>();
      }
    }
    if (auto use_2cta = node->attrs->dict.Get("use_2cta")) {
      auto enabled = GetIntAnnotation(*use_2cta);
      TVM_FFI_CHECK(enabled && (*enabled == 0 || *enabled == 1), ValueError)
          << "use_2cta must be boolean";
      attributes_ = DictAttrs({{"use_2cta", IntImm::Bool(*enabled != 0)}});
    }
    RegionStmt region = GetRef<RegionStmt>(node);
    auto attributes = region->attrs->dict;
    if (auto layouts =
            region->attrs.GetAttr<Map<TensorVar, Layout>>("layout_map")) {
      auto retained = *layouts;
      for (const auto &[tensor, layout] : *layouts)
        if (IsTmemBuffer(tensor))
          retained.erase(tensor);
      if (retained.empty())
        attributes.erase("layout_map");
      else
        attributes.Set("layout_map", retained);
      region.CopyOnWrite()->attrs = DictAttrs(attributes);
    }
    Stmt result = Base::Mutate_(region.get(), mode).ValueOrUnchanged(region);
    threads_ = saved_threads;
    thread_extents_ = saved_extents;
    attributes_ = saved_attributes;
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const IfNode *node, InplaceMode mode) final {
    bool saved = divergent_;
    divergent_ |= VariesByThread(node->condition);
    PrimExpr saved_path = path_;
    PrimExpr condition =
        Mutate(node->condition, mode).ValueOrUnchanged(node->condition);
    PrimExpr expanded = Substitute(condition, scalar_values_);
    path_ = saved_path && expanded;
    SeqStmt then_case =
        Mutate(node->then_case, mode).ValueOrUnchanged(node->then_case);
    Optional<SeqStmt> else_case;
    if (node->else_case) {
      path_ = saved_path && !expanded;
      else_case =
          Mutate(*node->else_case, mode).ValueOrUnchanged(*node->else_case);
    }
    path_ = saved_path;
    divergent_ = saved;
    return If(condition, then_case, else_case, node->loc);
  }

  UnchangedOr<Stmt> Mutate_(const WhileNode *node, InplaceMode mode) final {
    bool saved = divergent_;
    divergent_ |= VariesByThread(node->condition);
    auto result = Base::Mutate_(node, mode);
    divergent_ = saved;
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *node, InplaceMode mode) final {
    if (auto scalar = node->var.as<PrimVar>()) {
      if (auto value = node->value.as<PrimExpr>())
        scalar_values_.Set(*scalar, Substitute(*value, scalar_values_));
    }
    if (auto tensor = node->var.as<TensorVar>();
        tensor && IsTmemBuffer(*tensor)) {
      const auto *declaration = node->value.as<CallNode>();
      TVM_FFI_CHECK(declaration && declaration->op.same_as(decl_tensor_op()) &&
                        declaration->args.size() == 1,
                    ValueError)
          << "TMEM tensor definitions require a direct allocation or a "
             "borrowed view";
      Expr pointer = declaration->args[0];
      while (const auto *cast = pointer.as<CallNode>()) {
        if (!cast->op.same_as(reinterpret_op()))
          break;
        pointer = cast->args[0];
      }
      const auto *projection = pointer.as<CallNode>();
      TVM_FFI_CHECK(projection && projection->op.same_as(tensor_data_ptr_op()),
                    ValueError)
          << "A borrowed TMEM view must preserve its source storage origin";
      TensorVar source = projection->args[0].as_or_throw<TensorVar>();
      auto allocation = addresses_.find(source.get());
      TVM_FFI_CHECK(allocation != addresses_.end(), ValueError)
          << "A borrowed TMEM view requires a live enclosing TMEM allocation";
      TVM_FFI_CHECK(tensor.value().shape().size() == 2 &&
                        tensor.value().ty()->HasDefaultLayout(
                            GetRef<sym::Analyzer>(analyzer_)),
                    ValueError)
          << "TMEM views require a materialized physical datapath and "
             "value-column layout";
      int view_bits =
          tensor.value().dtype().bits() * tensor.value().dtype().lanes();
      int source_bits = source.dtype().bits() * source.dtype().lanes();
      TVM_FFI_CHECK(
          analyzer_->CanProve(tensor.value().shape()[0] <= source.shape()[0]) &&
              analyzer_->CanProve(tensor.value().shape()[1] * view_bits <=
                                  source.shape()[1] * source_bits),
          ValueError)
          << "A borrowed TMEM view cannot exceed its source physical storage";
      addresses_.insert_or_assign(tensor.value().get(), allocation->second);
      view_bindings_.push_back(tensor.value().get());
      return SeqStmt({}, node->loc);
    }
    return Base::Mutate_(node, mode);
  }

  void RestoreViews(size_t count) {
    while (view_bindings_.size() > count) {
      addresses_.erase(view_bindings_.back());
      view_bindings_.pop_back();
    }
  }

  UnchangedOr<Stmt> Mutate_(const ForNode *node, InplaceMode mode) final {
    bool saved = divergent_;
    divergent_ |= VariesByThread(node->begin) || VariesByThread(node->end) ||
                  VariesByThread(node->step);
    auto result = Base::Mutate_(node, mode);
    divergent_ = saved;
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtNode *node, InplaceMode mode) final {
    size_t view_count = view_bindings_.size();
    Array<TensorVar> tensors;
    std::vector<int> columns;
    VarSet bindings;
    for (const Stmt &statement : node->seq) {
      const auto *binding = statement.as<BindNode>();
      auto tensor = binding ? binding->var.as<TensorVar>() : std::nullopt;
      const auto *call = binding ? binding->value.as<CallNode>() : nullptr;
      if (!tensor || !IsTmemBuffer(*tensor) || !call ||
          !call->op.same_as(alloc_tensor_op()))
        continue;
      TVM_FFI_CHECK(!divergent_, ValueError)
          << "TMEM allocations require a block-uniform lexical scope";
      TVM_FFI_CHECK(tensor.value().shape().size() == 2 &&
                        tensor.value().ty()->HasDefaultLayout(
                            GetRef<sym::Analyzer>(analyzer_)),
                    ValueError)
          << "TMEM allocation must have its physical (datapath, value-column) "
             "layout materialized";
      auto rows = GetConstInt(analyzer_->Simplify(tensor.value().shape()[0]));
      auto values = GetConstInt(analyzer_->Simplify(tensor.value().shape()[1]));
      int bits = tensor.value().dtype().bits() * tensor.value().dtype().lanes();
      TVM_FFI_CHECK(bits > 0, ValueError)
          << "TMEM allocation requires a positive element bit width";
      TVM_FFI_CHECK(rows && values && *rows > 0 && *rows <= 128 &&
                        *values > 0 && *values <= 512 * 32 / bits,
                    ValueError)
          << "TMEM allocation exceeds 128 datapaths or 512 b32 columns";
      tensors.push_back(*tensor);
      columns.push_back((*values * bits + 31) / 32);
      bindings.insert(tensor.value().get());
    }
    if (tensors.empty()) {
      auto result = Base::Mutate_(node, mode);
      RestoreViews(view_count);
      return result;
    }
    TVM_FFI_CHECK(TargetHasSMVersionGE(target_, 100) && threads_[0], ValueError)
        << "TMEM allocation requires a Blackwell target and a thread launch";
    PrimExpr thread_count =
        thread_extents_[0] * thread_extents_[1] * thread_extents_[2];
    TVM_FFI_CHECK(analyzer_->CanProve(thread_count >= 32), ValueError)
        << "TMEM allocation requires a complete warp";
    VarSet explicit_releases;
    bool returns = false;
    PostOrderVisit(GetRef<SeqStmt>(node), [&](const ObjectRef &object) {
      returns |= object.as<ReturnNode>() != nullptr;
      if (const auto *call = object.as<CallNode>();
          call && call->op.same_as(deallocate_tmem()))
        explicit_releases.insert(GetVarFromAccessPtr(call->args[0]).get());
    });
    auto [fallthrough, falls] =
        FallthroughReleases(GetRef<Stmt>(node), bindings);
    std::vector<bool> packable;
    for (const TensorVar &tensor : tensors) {
      packable.push_back(!explicit_releases.count(tensor.get()));
      TVM_FFI_CHECK(!returns || explicit_releases.count(tensor.get()),
                    ValueError)
          << "Early-return TMEM scopes require explicit deallocation of each "
             "arena";
    }
    auto [arenas, placements] = PlanArenas(columns, packable);
    int total_columns = 0;
    for (const TmemArena &arena : arenas)
      total_columns += arena.allocated;
    TVM_FFI_CHECK(total_columns <= 512, ValueError)
        << "This lexical scope requires " << total_columns
        << " TMEM columns, exceeding 512";
    std::vector<TensorVar> words;
    Array<Stmt> prefix;
    for (const TensorVar &tensor : tensors) {
      TensorVar word("tmem_address_" + tensor->name,
                     TensorType({PrimExpr(1)}, PrimType::UInt(32),
                                runtime::MemoryScope::Get("shared")));
      words.push_back(word);
      prefix.push_back(AllocateWord(word, node->loc));
    }
    std::vector<TmemAddress> releases;
    for (size_t index = 0; index < tensors.size(); ++index) {
      const TmemArena &arena = arenas[placements[index].arena];
      Optional<TensorVar> live;
      if (explicit_releases.count(tensors[index].get()) &&
          !fallthrough.count(tensors[index].get())) {
        live = TensorVar("tmem_live_" + tensors[index]->name,
                         TensorType({PrimExpr(1)}, PrimType::Bool(),
                                    runtime::MemoryScope::Get("local")));
        prefix.push_back(AllocateWord(*live, node->loc));
        prefix.push_back(
            TensorStore(*live, {PrimExpr(0)}, IntImm::Bool(true), node->loc));
      }
      TmemAddress address{words[index], words[arena.members.front()],
                          arena.allocated, attributes_, live};
      addresses_.insert_or_assign(tensors[index].get(), address);
      if (index == arena.members.front() &&
          !fallthrough.count(tensors[index].get()))
        releases.push_back(address);
    }
    std::vector<size_t> arena_order(arenas.size());
    std::iota(arena_order.begin(), arena_order.end(), 0);
    std::stable_sort(arena_order.begin(), arena_order.end(),
                     [&](size_t left, size_t right) {
                       return arenas[left].allocated > arenas[right].allocated;
                     });
    Array<Stmt> initialize;
    for (size_t index : arena_order) {
      const TmemArena &arena = arenas[index];
      initialize.push_back(Evaluate(
          Call(std::nullopt, ptx_init_tensor_memory(),
               {words[arena.members.front()].data(), PrimExpr(arena.allocated)},
               attributes_, {}, node->loc),
          node->loc));
    }
    prefix.push_back(If(FloorDiv(ThreadIndex(), 32) == 0, SeqStmt(initialize),
                        std::nullopt, node->loc));
    prefix.push_back(Sync(node->loc));
    Array<Stmt> shifted;
    for (size_t index = 0; index < tensors.size(); ++index) {
      const TmemArena &arena = arenas[placements[index].arena];
      if (index != arena.members.front())
        shifted.push_back(TensorStore(
            words[index], {PrimExpr(0)},
            TensorLoad(words[arena.members.front()], {PrimExpr(0)}) +
                IntImm(PrimType::UInt(32), placements[index].offset),
            node->loc));
    }
    if (!shifted.empty()) {
      prefix.push_back(
          If(ThreadIndex() == 0, SeqStmt(shifted), std::nullopt, node->loc));
      prefix.push_back(Sync(node->loc));
    }
    Array<Stmt> body;
    for (const Stmt &statement : node->seq) {
      const auto *binding = statement.as<BindNode>();
      if (!binding || !bindings.count(binding->var.get()))
        body.push_back(statement);
    }
    SeqStmt rewritten = SeqStmt(body, node->loc);
    rewritten = Base::Mutate_(rewritten.get(), mode)
                    .ValueOrUnchanged(rewritten)
                    .as_or_throw<SeqStmt>();
    prefix.insert(prefix.end(), rewritten->seq.begin(), rewritten->seq.end());
    if (!releases.empty() && falls) {
      if (!returns)
        prefix.push_back(Sync(node->loc));
      if (auto use_2cta = attributes_->dict.Get("use_2cta");
          use_2cta && GetIntAnnotation(*use_2cta).value_or(0))
        prefix.push_back(Evaluate(Call(std::nullopt, cluster_sync(), {})));
      for (const TmemAddress &address : releases)
        prefix.push_back(Release(address, node->loc));
    }
    for (const TensorVar &tensor : tensors)
      addresses_.erase(tensor.get());
    RestoreViews(view_count);
    return SeqStmt(prefix, node->loc);
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateNode *node, InplaceMode mode) final {
    const auto *call = node->value.as<CallNode>();
    if (call && call->op.same_as(deallocate_tmem())) {
      const VarNode *tensor = GetVarFromAccessPtr(call->args[0]).get();
      TVM_FFI_CHECK(std::find(view_bindings_.begin(), view_bindings_.end(),
                              tensor) == view_bindings_.end(),
                    ValueError)
          << "Borrowed TMEM views cannot be deallocated; release the owning "
             "tensor";
      auto found = addresses_.find(tensor);
      TVM_FFI_CHECK(found != addresses_.end(), ValueError)
          << "deallocate_tmem requires a live enclosing TMEM allocation";
      sym::Analyzer proof;
      Optional<PrimExpr> first_lane;
      for (int lane = 0; lane < 32; ++lane) {
        Map<PrimVar, PrimExpr> replacements;
        PrimExpr remaining = lane;
        for (size_t dimension = 0; dimension < threads_.size(); ++dimension) {
          if (threads_[dimension]) {
            replacements.Set(*threads_[dimension],
                             FloorMod(remaining, thread_extents_[dimension]));
            remaining = FloorDiv(remaining, thread_extents_[dimension]);
          }
        }
        PrimExpr active = proof->Simplify(Substitute(path_, replacements));
        if (!first_lane)
          first_lane = active;
        TVM_FFI_CHECK(!proof->CanProve(!active) &&
                          proof->CanProve(active == *first_lane),
                      ValueError)
            << "Explicit TMEM deallocation must execute on the complete first "
               "warp";
      }
      return Release(found->second, node->loc);
    }
    return Base::Mutate_(node, mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *node,
                                InplaceMode mode) final {
    auto tensor = node->source.as<TensorVar>();
    auto found =
        tensor ? addresses_.find(tensor.value().get()) : addresses_.end();
    if (found == addresses_.end())
      return Base::Mutate_(node, mode);
    TVM_FFI_CHECK(node->indices.size() == 2, ValueError)
        << "TMEM address needs datapath and value-column coordinates";
    PrimExpr row =
        Mutate(node->indices[0], mode).ValueOrUnchanged(node->indices[0]);
    PrimExpr column =
        Mutate(node->indices[1], mode).ValueOrUnchanged(node->indices[1]);
    int bits = tensor.value().dtype().bits() * tensor.value().dtype().lanes();
    PrimExpr bit_offset = column * MakeConst(column.ty(), bits);
    TVM_FFI_CHECK(analyzer_->CanProve(floormod(bit_offset, 32) == 0),
                  ValueError)
        << "TMEM address must begin on a b32 column";
    PrimExpr offset = BitwiseOr(
        LShift(Cast(PrimType::UInt(32), row), IntImm(PrimType::UInt(32), 16)),
        Cast(PrimType::UInt(32), floordiv(bit_offset, 32)));
    return TensorLoad(found->second.word, {PrimExpr(0)}) + offset;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node,
                            InplaceMode mode) final {
    auto tensor = node->dest.as<TensorVar>();
    TVM_FFI_CHECK(!tensor || !IsTmemBuffer(*tensor), ValueError)
        << "Direct TMEM stores must be lowered to tcgen05 instructions";
    return Base::Mutate_(node, mode);
  }

  UnchangedOr<Expr> Mutate_(const CallNode *node, InplaceMode mode) final {
    if (node->op.same_as(tensor_data_ptr_op())) {
      auto tensor = node->args[0].as<TensorVar>();
      if (tensor)
        if (auto found = addresses_.find(tensor.value().get());
            found != addresses_.end())
          return found->second.word.data();
    }
    return Base::Mutate_(node, mode);
  }

  UnchangedOr<Expr> Mutate_(const VarNode *node, InplaceMode mode) final {
    if (auto found = addresses_.find(node); found != addresses_.end())
      return found->second.word.data();
    return Base::Mutate_(node, mode);
  }

  Target target_;
  std::array<Optional<PrimVar>, 3> threads_;
  std::array<PrimExpr, 3> thread_extents_{PrimExpr(1), PrimExpr(1),
                                          PrimExpr(1)};
  DictAttrs attributes_;
  Map<PrimVar, PrimExpr> scalar_values_;
  PrimExpr path_ = IntImm::Bool(true);
  bool divergent_ = false;
  std::unordered_map<const VarNode *, TmemAddress> addresses_;
  std::vector<const VarNode *> view_bindings_;
};
} // namespace

tvm::transform::Pass LowerSharedTmem() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    if (function->body) {
      auto target = function->GetAttr<Target>("target");
      TVM_FFI_CHECK(target, ValueError) << "LowerSharedTmem requires a target";
      sym::Analyzer analyzer;
      auto rewriter =
          ffi::make_object<SharedTmemRewriter>(analyzer.get(), *target);
      SeqStmt body = *function->body;
      function.CopyOnWrite()->body =
          rewriter->Mutate(body).ValueOrUnchanged(body);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.cuda.LowerSharedTmem");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.LowerSharedTmem",
                                   LowerSharedTmem);
}
} // namespace tvm::tl
