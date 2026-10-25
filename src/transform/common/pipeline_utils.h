#ifndef TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_

#include "memory_access.h"

#include <tvm/tirx/op/region.h>

#include <algorithm>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {
using namespace tirx;

inline constexpr const char *kPipelineTmaCopies =
    "software_pipeline_tma_copies";
inline constexpr const char *kPipelineAsyncProducers =
    "software_pipeline_async_producers";
inline constexpr const char *kPipelineAsyncProducerGroups =
    "software_pipeline_async_producer_groups";
inline constexpr const char *kPipelineReplayableScalarBinds =
    "software_pipeline_replayable_scalar_binds";
inline constexpr const char *kPipelineStage = "software_pipeline_stage";
inline constexpr const char *kPipelineOrder = "software_pipeline_order";

inline std::vector<int> PipelineIntegerArray(const ffi::Any &value,
                                             const char *name) {
  std::vector<int> result;
  for (const ffi::Any &element : value.cast<ffi::Array<ffi::Any>>()) {
    auto integer = GetIntAnnotation(element);
    TVM_FFI_CHECK(integer && *integer >= 0 && *integer <= 1024, ValueError)
        << name << " must contain integers in [0, 1024]";
    result.push_back(static_cast<int>(*integer));
  }
  return result;
}

inline ffi::Array<IntImm> PipelineIntegerArray(const std::vector<int> &values) {
  ffi::Array<IntImm> result;
  for (int value : values)
    result.push_back(IntImm(PrimType::Int(32), value));
  return result;
}

inline std::optional<int> GetPipelineNumStages(const ForNode *loop) {
  for (const char *key : {"num_stages", "tl_pipelined_num_stages"}) {
    if (auto value = loop->annotations.Get(key)) {
      auto stages = GetIntAnnotation(*value);
      TVM_FFI_CHECK(stages && *stages >= 1 && *stages <= 1024, ValueError)
          << key << " must be an integer in [1, 1024]";
      return static_cast<int>(*stages);
    }
  }
  if (auto value = loop->annotations.Get(kPipelineStage)) {
    auto stages = PipelineIntegerArray(*value, kPipelineStage);
    if (!stages.empty())
      return *std::max_element(stages.begin(), stages.end()) + 1;
  }
  return std::nullopt;
}

inline bool IsPipelineDeclarationStmt(const Stmt &statement) {
  const auto *binding = statement.as<BindNode>();
  const auto *call = binding ? binding->value.as<CallNode>() : nullptr;
  return binding && binding->var->ty.as<TensorTypeNode>() && call &&
         call->op.same_as(alloc_tensor_op());
}

inline ffi::Array<Stmt> NormalizePipelineBody(const Stmt &body) {
  if (const auto *sequence = body.as<SeqStmtNode>())
    return sequence->seq;
  return {body};
}

inline Stmt MakePipelineBody(const ffi::Array<Stmt> &statements) {
  return SeqStmt(statements);
}

struct PipelineWriteSpan {
  const VarNode *storage;
  PrimExpr begin;
  PrimExpr bytes;
};

struct PipelineStatement {
  Stmt statement;
  std::unordered_map<const VarNode *, int> memory;
  std::unordered_set<const VarNode *> reads;
  std::unordered_set<const VarNode *> definitions;
  std::unordered_set<const VarNode *> overwrites;
  std::vector<MemoryAccess> accesses;
  std::vector<PipelineWriteSpan> write_spans;
};

inline bool FullTensorRegion(const TensorRegion &region,
                             sym::Analyzer analyzer) {
  auto tensor = region->source.as<TensorVar>();
  if (!tensor || region->region.size() != tensor.value().shape().size())
    return false;
  for (size_t axis = 0; axis < region->region.size(); ++axis)
    if (!analyzer->CanProve(region->region[axis]->begin == 0) ||
        !analyzer->CanProve(region->region[axis]->extent ==
                            tensor.value().shape()[axis]))
      return false;
  return true;
}

inline std::unordered_set<const VarNode *>
PipelineOverwrites(const Stmt &statement, ConstrSet constraints = {},
                   std::vector<PipelineWriteSpan> *spans = nullptr) {
  std::unordered_set<const VarNode *> result;
  sym::Analyzer analyzer;
  auto guard = constraints.Populate(analyzer);
  if (const auto *sequence = statement.as<SeqStmtNode>()) {
    for (const Stmt &child : sequence->seq) {
      auto writes = PipelineOverwrites(child, constraints, spans);
      result.insert(writes.begin(), writes.end());
      if (const auto *binding = child.as<BindNode>()) {
        auto variable = binding->var.as<PrimVar>();
        auto value = binding->value.as<PrimExpr>();
        if (variable && value && SideEffect(*value) == CallEffectKind::kPure)
          constraints.AddConstr(*variable, *value);
      }
    }
    return result;
  }
  if (const auto *region = statement.as<RegionStmtNode>();
      region &&
      (region->op.same_as(scope_op()) || region->op->name == "tl.scope") &&
      region->body_params.empty() && region->result_vars.empty())
    return PipelineOverwrites(region->body, constraints, spans);
  if (const auto *loop = statement.as<ForNode>()) {
    if (analyzer->CanProve(loop->step == 1) &&
        analyzer->CanProve(loop->end > loop->begin)) {
      constraints.AddConstr(loop->var, Range(loop->begin, loop->end));
      return PipelineOverwrites(loop->body, constraints, spans);
    }
    return result;
  }
  if (const auto *evaluation = statement.as<EvaluateNode>()) {
    if (auto call = evaluation->value.as<Call>()) {
      auto operation = ParseOperator(*call);
      if (operation.defined()) {
        for (const TensorRegion &region :
             operation->GetAccessRegions().writes) {
          TensorVar tensor = region->source.as_or_throw<TensorVar>();
          if (FullTensorRegion(region, analyzer)) {
            result.insert(tensor.get());
            if (spans)
              spans->push_back({tensor.get(), IntImm::Int64(0),
                                GetBufferStorageSizeBytes(tensor)});
          } else {
            auto domain = constraints;
            ffi::Array<PrimExpr> indices;
            bool nonempty = true;
            for (const Range &range : region->region) {
              nonempty &= analyzer->CanProve(range->extent > 0);
              PrimVar coordinate("pipeline_write_coordinate",
                                 range->begin.ty());
              domain.AddConstr(coordinate, Range(0, range->extent));
              indices.push_back(range->begin + coordinate);
            }
            if (nonempty) {
              auto writes = PipelineOverwrites(
                  TensorStore(tensor, indices, MakeConst(tensor.dtype(), 0)),
                  domain, spans);
              result.insert(writes.begin(), writes.end());
            }
          }
        }
      }
    }
  }
  if (const auto *store = statement.as<TensorStoreNode>()) {
    TensorVar tensor = store->dest.as_or_throw<TensorVar>();
    bool full = store->indices.size() == tensor.shape().size();
    for (size_t axis = 0; full && axis < store->indices.size(); ++axis) {
      if (const auto *ramp = store->indices[axis].as<RampNode>())
        full &= analyzer->CanProve(ramp->base == 0) &&
                analyzer->CanProve(ramp->stride == 1) &&
                analyzer->CanProve(ramp->lanes == tensor.shape()[axis]);
      else
        full &= analyzer->CanProve(store->indices[axis] == 0) &&
                analyzer->CanProve(tensor.shape()[axis] == 1);
    }
    if (full) {
      result.insert(tensor.get());
      if (spans)
        spans->push_back({tensor.get(), IntImm::Int64(0),
                          GetBufferStorageSizeBytes(tensor)});
      return result;
    }
    if (!tensor.dtype().IsScalar() || tensor.dtype().bits() % 8 != 0)
      return result;
    for (const PrimExpr &index : store->indices)
      if (!index.ty().IsScalar())
        return result;
    MemoryAliases aliases;
    MemoryLocation location = aliases.Element(tensor, store->indices);
    if (!location.precise)
      return result;
    ffi::Map<PrimVar, PrimExpr> definitions, minima;
    ffi::Map<PrimVar, Range> ranges;
    for (const Constr &constraint : constraints.constrs_) {
      if (constraint.kind == Constr::kBindValue)
        definitions.Set(constraint.var,
                        Substitute(constraint.value, definitions));
      if (constraint.kind == Constr::kBindRange)
        ranges.Set(constraint.var, constraint.range);
    }
    PrimExpr offset =
        analyzer->Simplify(Substitute(location.offset, definitions));
    struct Axis {
      PrimVar variable;
      PrimExpr minimum;
      int64_t extent;
      int64_t stride;
    };
    std::vector<Axis> axes;
    for (const Var &variable : tvm::UndefinedVars(offset)) {
      auto primitive = variable.as<PrimVar>();
      auto range = primitive ? ranges.Get(*primitive) : std::nullopt;
      if (!range)
        continue;
      auto extent = GetConstInt(analyzer->Simplify(range.value()->extent));
      if (!extent || *extent <= 0)
        return result;
      minima.Set(*primitive, range.value()->begin);
      if (*extent == 1)
        continue;
      PrimExpr delta = analyzer->Simplify(
          Substitute(offset, {{*primitive, range.value()->begin + 1}}) -
          Substitute(offset, {{*primitive, range.value()->begin}}));
      auto stride = GetConstInt(delta);
      if (!stride || *stride <= 0)
        return result;
      axes.push_back({*primitive, range.value()->begin, *extent, *stride});
    }
    PrimExpr base = analyzer->Simplify(Substitute(offset, minima));
    PrimExpr reconstructed = base;
    for (const Axis &axis : axes)
      reconstructed = reconstructed +
                      Cast(PrimType::Int(64), axis.variable - axis.minimum) *
                          IntImm::Int64(axis.stride);
    if (!analyzer->CanProveEqual(offset, reconstructed))
      return result;
    std::sort(axes.begin(), axes.end(),
              [](const Axis &left, const Axis &right) {
                return left.stride < right.stride;
              });
    PrimExpr covered = IntImm::Int64(tensor.dtype().StorageBytes());
    for (const Axis &axis : axes) {
      if (!analyzer->CanProveEqual(covered, IntImm::Int64(axis.stride)))
        return result;
      covered = covered * IntImm::Int64(axis.extent);
    }
    if (spans)
      spans->push_back({tensor.get(), base, covered});
    if (analyzer->CanProveEqual(base, 0) &&
        analyzer->CanProveEqual(covered, GetBufferStorageSizeBytes(tensor)))
      result.insert(tensor.get());
  }
  return result;
}

struct PipelineBody {
  std::vector<PipelineStatement> statements;
  ffi::Array<Bind> declarations;
  std::vector<size_t> source_indices;
};

inline PipelineBody
AnalyzePipelineBody(const SeqStmt &body, MemoryAliases aliases = {},
                    const ConstrSet &constraints = {},
                    const std::vector<int> &replay_mask = {}) {
  PipelineBody result;
  TVM_FFI_CHECK(replay_mask.empty() || replay_mask.size() == body->seq.size(),
                ValueError)
      << "Pipeline scalar replay mask must match the source statements";
  ffi::Map<PrimVar, PrimExpr> replay;
  for (size_t index = 0; index < body->seq.size(); ++index) {
    Stmt statement = Substitute(body->seq[index], replay);
    if (IsPipelineDeclarationStmt(statement)) {
      Bind binding = statement.as_or_throw<Bind>();
      aliases.Bind(binding.get());
      result.declarations.push_back(binding);
      continue;
    }
    if (const auto *binding = statement.as<BindNode>()) {
      auto variable = binding->var.as<PrimVar>();
      auto value = binding->value.as<PrimExpr>();
      if (variable && value &&
          (SideEffect(*value) == CallEffectKind::kPure ||
           (!replay_mask.empty() && replay_mask[index]))) {
        replay.Set(*variable, *value);
        continue;
      }
    }
    auto collector = ffi::make_object<MemoryAccessCollector>(aliases);
    collector->constr_stack_ = constraints.constrs_;
    collector->Visit(statement);
    TVM_FFI_CHECK(!collector->exits, ValueError)
        << "Software-pipelined bodies cannot contain break, continue, or "
           "return";
    PipelineStatement item{statement, {}, {}, {}, {}, collector->accesses, {}};
    item.overwrites =
        PipelineOverwrites(statement, constraints, &item.write_spans);
    for (const MemoryAccess &access : item.accesses)
      if (access.location.allocation)
        item.memory[access.location.allocation.value().get()] |= access.mask;
    for (const Var &variable : tvm::UndefinedVars(statement))
      item.reads.insert(variable.get());
    if (const auto *binding = statement.as<BindNode>())
      item.definitions.insert(binding->var.get());
    result.source_indices.push_back(index);
    result.statements.push_back(std::move(item));
  }
  return result;
}

inline bool PipelineMemoryOverlap(const MemoryAccess &left,
                                  const MemoryAccess &right) {
  if (!((left.mask | right.mask) & kAccessWrite))
    return false;
  if (!left.location.allocation || !right.location.allocation)
    return true;
  if (!left.location.allocation.value().same_as(*right.location.allocation))
    return false;
  auto bounds = [](const MemoryAccess &access)
      -> std::optional<std::pair<int64_t, int64_t>> {
    if (!access.location.precise)
      return std::nullopt;
    sym::Analyzer analyzer;
    auto guard = access.constraints.Populate(analyzer);
    auto begin = analyzer->const_int_bound(access.location.offset);
    auto end = analyzer->const_int_bound(access.location.offset + access.bytes);
    if (begin->min_value == sym::ConstIntBound::kNegInf ||
        end->max_value == sym::ConstIntBound::kPosInf)
      return std::nullopt;
    return std::pair{begin->min_value, end->max_value};
  };
  auto left_bounds = bounds(left);
  auto right_bounds = bounds(right);
  return !left_bounds || !right_bounds ||
         (left_bounds->first < right_bounds->second &&
          right_bounds->first < left_bounds->second);
}

inline bool PipelineDependency(const PipelineStatement &before,
                               const PipelineStatement &after) {
  for (const VarNode *definition : before.definitions)
    if (after.reads.count(definition))
      return true;
  for (const MemoryAccess &left : before.accesses) {
    for (const MemoryAccess &right : after.accesses) {
      if (PipelineMemoryOverlap(left, right))
        return true;
    }
  }
  return false;
}

inline bool PipelineNeedsVersion(const PipelineBody &body,
                                 const VarNode *storage,
                                 const std::vector<int> &stages) {
  for (size_t before = 0; before < body.statements.size(); ++before) {
    for (size_t after = before + 1; after < body.statements.size(); ++after) {
      if (stages[before] == stages[after])
        continue;
      for (const MemoryAccess &left : body.statements[before].accesses) {
        if (!left.location.allocation ||
            left.location.allocation.value().get() != storage)
          continue;
        for (const MemoryAccess &right : body.statements[after].accesses) {
          if (PipelineMemoryOverlap(left, right))
            return true;
        }
      }
    }
  }
  return false;
}

inline ffi::Optional<Range>
PipelinePartialVersionRange(const PipelineBody &body, const VarNode *storage) {
  auto tensor = ffi::GetRef<Var>(storage).as<TensorVar>();
  if (!tensor || tensor.value().shape().size() != 1 ||
      !tensor.value().dtype().IsScalar() ||
      tensor.value().dtype().bits() % 8 != 0)
    return std::nullopt;
  auto elements = GetConstInt(tensor.value().shape()[0]);
  if (!elements || *elements <= 0)
    return std::nullopt;
  int64_t element_bytes = tensor.value().dtype().StorageBytes();
  sym::Analyzer analyzer;
  PrimVar coordinate("pipeline_coordinate", PrimType::Int(64));
  analyzer->Bind(coordinate, Range(0, IntImm::Int64(*elements)));
  MemoryAliases aliases;
  auto location = aliases.Element(*tensor, {coordinate});
  if (!location.precise ||
      !analyzer->CanProveEqual(location.offset,
                               coordinate * IntImm::Int64(element_bytes)))
    return std::nullopt;
  std::vector<std::pair<int64_t, int64_t>> spans;
  for (const PipelineStatement &item : body.statements) {
    auto access = item.memory.find(storage);
    if (access != item.memory.end() && (access->second & kAccessRead))
      break;
    for (const PipelineWriteSpan &span : item.write_spans) {
      if (span.storage != storage)
        continue;
      auto begin = GetConstInt(analyzer->Simplify(span.begin));
      auto end = GetConstInt(analyzer->Simplify(span.begin + span.bytes));
      if (!begin || !end || *begin < 0 || *end <= *begin ||
          *begin % element_bytes || *end % element_bytes ||
          *end / element_bytes > *elements)
        return std::nullopt;
      spans.emplace_back(*begin, *end);
    }
  }
  if (spans.empty())
    return std::nullopt;
  std::sort(spans.begin(), spans.end());
  int64_t begin = spans[0].first;
  int64_t end = spans[0].second;
  for (const auto &span : spans) {
    if (span.first > end)
      return std::nullopt;
    end = std::max(end, span.second);
  }
  if (begin == 0 && end / element_bytes == *elements)
    return std::nullopt;
  for (const PipelineStatement &item : body.statements) {
    for (const MemoryAccess &access : item.accesses) {
      if (!(access.mask & kAccessWrite) || !access.location.allocation ||
          access.location.allocation.value().get() != storage)
        continue;
      if (!access.location.precise)
        return std::nullopt;
      sym::Analyzer bounds;
      auto guard = access.constraints.Populate(bounds);
      if (!bounds->CanProve(access.location.offset >=
                            MakeConst(access.location.offset.ty(), begin)) ||
          !bounds->CanProve(access.location.offset + access.bytes <=
                            MakeConst(access.location.offset.ty(), end)))
        return std::nullopt;
    }
  }
  return Range(MakeConst(tensor.value().shape()[0].ty(), begin / element_bytes),
               MakeConst(tensor.value().shape()[0].ty(), end / element_bytes));
}

inline bool PipelineVersionable(const PipelineBody &body,
                                const VarNode *storage) {
  auto tensor = ffi::GetRef<Var>(storage).as<TensorVar>();
  if (!tensor || (tensor.value().scope().name().find("shared") != 0 &&
                  tensor.value().scope().name().find("local") != 0))
    return false;
  sym::Analyzer analyzer;
  std::vector<PipelineWriteSpan> written;
  for (const PipelineStatement &item : body.statements) {
    auto found = item.memory.find(storage);
    if (found == item.memory.end())
      continue;
    if (found->second & kAccessRead)
      return PipelinePartialVersionRange(body, storage).has_value();
    if (item.overwrites.count(storage))
      return true;
    for (const PipelineWriteSpan &span : item.write_spans)
      if (span.storage == storage)
        written.push_back(span);
    PrimExpr covered = IntImm::Int64(0);
    for (size_t attempt = 0; attempt < written.size(); ++attempt) {
      bool changed = false;
      for (const PipelineWriteSpan &span : written) {
        PrimExpr end = analyzer->Simplify(span.begin + span.bytes);
        if (analyzer->CanProve(span.begin <= covered) &&
            analyzer->CanProve(end > covered)) {
          covered = end;
          changed = true;
        }
      }
      if (!changed)
        break;
    }
    if (analyzer->CanProve(covered >= GetBufferStorageSizeBytes(*tensor)))
      return true;
  }
  return PipelinePartialVersionRange(body, storage).has_value();
}

inline void ValidatePipelineSchedule(const PipelineBody &body,
                                     const std::vector<int> &stages,
                                     const std::vector<int> &order) {
  TVM_FFI_CHECK(stages.size() == body.statements.size() &&
                    order.size() == stages.size(),
                ValueError)
      << "Pipeline stage and order arrays must have one entry per executable "
         "statement";
  std::unordered_set<int> distinct(order.begin(), order.end());
  TVM_FFI_CHECK(distinct.size() == order.size(), ValueError)
      << "Pipeline orders must be unique";
  for (size_t before = 0; before < stages.size(); ++before)
    for (size_t after = before + 1; after < stages.size(); ++after)
      if (PipelineDependency(body.statements[before], body.statements[after]))
        TVM_FFI_CHECK(stages[before] < stages[after] ||
                          (stages[before] == stages[after] &&
                           order[before] < order[after]),
                      ValueError)
            << "Pipeline schedule reverses a memory or scalar dependency "
               "between statements "
            << before << " and " << after;
  std::unordered_map<const VarNode *, std::pair<int, int>> bounds;
  std::unordered_set<const VarNode *> written;
  for (size_t index = 0; index < stages.size(); ++index)
    for (const auto &[storage, mask] : body.statements[index].memory) {
      auto [found, inserted] =
          bounds.try_emplace(storage, stages[index], stages[index]);
      found->second.first = std::min(found->second.first, stages[index]);
      found->second.second = std::max(found->second.second, stages[index]);
      if (mask & kAccessWrite)
        written.insert(storage);
    }
  for (const auto &[storage, interval] : bounds)
    TVM_FFI_CHECK(interval.first == interval.second ||
                      !written.count(storage) ||
                      !PipelineNeedsVersion(body, storage, stages) ||
                      PipelineVersionable(body, storage),
                  ValueError)
        << "Pipeline crosses iteration-carried state in " << storage->name
        << "; all accesses must share a stage or start with a full overwrite";
}

inline tirx::Layout ExpandPipelineLayout(const tirx::Layout &layout,
                                         PrimExpr depth,
                                         int64_t alignment = 1) {
  if (auto tile = layout.as<tirx::TileLayout>()) {
    PrimExpr stride =
        FloorDiv(Cast(PrimType::Int(64), layout.GetSpan(ffi::String("m"))) +
                     IntImm::Int64(alignment - 1),
                 IntImm::Int64(alignment)) *
        IntImm::Int64(alignment);
    ffi::Array<tirx::Iter> dimensions{
        tirx::Iter(depth, stride, tirx::Axis::Get("m"))};
    dimensions.insert(dimensions.end(), tile.value()->shard.begin(),
                      tile.value()->shard.end());
    return tirx::TileLayout(dimensions, tile.value()->replica,
                            tile.value()->offset);
  }
  auto composed = layout.as_or_throw<tirx::ComposeLayout>();
  int swizzle_bits =
      composed->per_element + composed->swizzle_len + composed->atom_len;
  TVM_FFI_CHECK(swizzle_bits >= 0 && swizzle_bits < 62, ValueError)
      << "Pipeline swizzle period exceeds the supported address range";
  alignment = std::lcm(alignment, int64_t{1} << swizzle_bits);
  auto tile = ExpandPipelineLayout(composed->tile_layout, depth, alignment)
                  .as_or_throw<tirx::TileLayout>();
  return tirx::ComposeLayout(composed->per_element, composed->swizzle_len,
                             composed->atom_len, tile, composed->swizzle_inner);
}

inline TensorVar ExpandPipelineTensor(const TensorVar &tensor, int depth) {
  ffi::Array<PrimExpr> shape{PrimExpr(depth)};
  shape.insert(shape.end(), tensor.shape().begin(), tensor.shape().end());
  int64_t element_bits = IsPacked4BitStorage(tensor.dtype())
                             ? tensor.dtype().bits() * tensor.dtype().lanes()
                             : tensor.dtype().StorageBytes() * 8;
  int64_t alignment_bits = std::max(1, tensor.ty()->data_alignment) * 8;
  int64_t alignment = std::lcm(element_bits, alignment_bits) / element_bits;
  auto layout =
      ExpandPipelineLayout(tensor.ty()->layout.value_or(
                               tirx::TileLayout::DefaultLayout(tensor.shape())),
                           PrimExpr(depth), alignment);
  return TensorVar(tensor->name + "_pipeline",
                   TensorType(shape, tensor.dtype(), tensor.scope(),
                              tensor.ty()->data_alignment, layout, tensor->loc),
                   tensor->loc);
}
} // namespace tvm::tl

#endif
