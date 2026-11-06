/*!
 * \file merge_shared_memory_allocations.cc
 * \brief Plan native shared arenas while retaining typed tensor/view
 * identities.
 */
#include "common/storage_size.h"
#include "cuda/op/builtin.h"
#include "ir/alias_analysis.h"
#include "ir/analyzer.h"
#include "ir/annotations.h"
#include "ir/lexical_scope.h"
#include "op/builtin.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/stmt_functor.h>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {
namespace {
using namespace tirx;
using namespace ffi;
using namespace prim;
using Indices = std::unordered_set<size_t>;

struct Allocation {
  Bind binding;
  TensorVar tensor;
  PrimExpr size{UnsafeInit{}};
  PrimExpr offset{UnsafeInit{}};
  int alignment{1};
  int definition{-1};
  int first{std::numeric_limits<int>::max()};
  int last{-1};
  bool asynchronous{false};
};

bool CanHoistSharedShape(AnyView object) {
  bool safe = true;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      object, [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
        if (node.as<TensorLoadObj>() || node.as<CallObj>())
          safe = false;
        auto check_divisor = [&](const PrimExpr &divisor) {
          const auto *constant = divisor.as<IntImmObj>();
          if (!constant || constant->value == 0 || constant->value == -1)
            safe = false;
        };
        if (const auto *value = node.as<DivObj>())
          check_divisor(value->b);
        if (const auto *value = node.as<ModObj>())
          check_divisor(value->b);
        if (const auto *value = node.as<FloorDivObj>())
          check_divisor(value->b);
        if (const auto *value = node.as<FloorModObj>())
          check_divisor(value->b);
        return ffi::WalkResult::Advance();
      });
  return safe;
}

class ShapeExpander : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  ShapeExpander(Map<Var, Expr> definitions,
                std::unordered_set<const VarObj *> parameters)
      : definitions_(std::move(definitions)),
        parameters_(std::move(parameters)) {}

  UnchangedOr<Expr> Mutate_(const VarObj *op, InplaceMode mode) final {
    if (parameters_.count(op))
      return Unchanged();
    auto definition = definitions_.Get(GetRef<Var>(op));
    TVM_FFI_CHECK(definition && op->ty.as<PrimTypeObj>() &&
                      active_.insert(op).second,
                  ValueError)
        << "Shared allocation size depends on device-local variable "
        << op->name;
    TVM_FFI_CHECK(CanHoistSharedShape(*definition), ValueError)
        << "Shared allocation size cannot hoist memory reads or potentially "
           "trapping expressions";
    Expr result = Mutate(*definition, mode).ValueOrUnchanged(*definition);
    active_.erase(op);
    return result;
  }

private:
  Map<Var, Expr> definitions_;
  std::unordered_set<const VarObj *> parameters_, active_;
};

// Native pointer aliases refer to TensorVars rather than a separate Buffer.data
// identity. Resolve them transitively; descriptor initialization additionally
// makes subsequent uses of the descriptor keep its shared operands alive.
class SharedUses {
public:
  SharedUses(const tirx::Function &function,
             const std::vector<Allocation> &allocations)
      : physical_(function) {
    for (size_t i = 0; i < allocations.size(); ++i)
      indices_.emplace(allocations[i].tensor.get(), i);
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        function->body,
        [&](const ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
          if (const auto *binding = object.as<BindObj>()) {
            // A pointer may pass through an integer/handle representation
            // before becoming a typed view again. Keep those SSA edges as well.
            definitions_.Set(binding->var, binding->value);
          }
          if (const auto *region = object.as<RegionStmtObj>()) {
            for (const Var &parameter : region->body_params) {
              definitions_.Set(parameter, tvm::Tuple(region->args));
            }
          }
          const auto *call = object.as<CallObj>();
          if (!call)
            return ffi::WalkResult::Advance();
          const auto *operation = call->op.as<OpObj>();
          if (operation != tl::initialize_wgmma_descriptor().get() &&
              operation != tl::initialize_tcgen05_descriptor().get())
            return ffi::WalkResult::Advance();
          auto args = DecodeCallOperands(call);
          if (args.empty())
            return ffi::WalkResult::Advance();
          auto root = physical_.Root(args[0]);
          if (root) {
            auto inputs = descriptor_inputs_.Get(*root).value_or(Array<Expr>{});
            inputs.insert(inputs.end(), args.begin(), args.end());
            descriptor_inputs_.Set(*root, inputs);
          }
          return ffi::WalkResult::Advance();
        });
  }

  Indices Get(AnyView object) const {
    Indices result;
    std::unordered_set<const VarObj *> active;
    for (const Var &variable : UndefinedVars(object))
      Add(variable, &result, &active);
    return result;
  }

private:
  void Add(const Var &variable, Indices *result,
           std::unordered_set<const VarObj *> *active) const {
    auto root = physical_.Root(variable);
    if (!root) {
      // Selects, region arguments and pointer-producing calls can have more
      // than one physical source. Retain every possible backing allocation.
      if (!active->insert(variable.get()).second)
        return;
      if (auto value = definitions_.Get(variable))
        for (const Var &dependency : UndefinedVars(*value))
          Add(dependency, result, active);
      return;
    }
    if (!active->insert(root.value().get()).second)
      return;
    auto index = indices_.find(root.value().get());
    if (index != indices_.end())
      result->insert(index->second);
    auto inputs = descriptor_inputs_.Get(*root);
    if (inputs) {
      for (const Expr &input : *inputs)
        for (const Var &dependency : UndefinedVars(input))
          Add(dependency, result, active);
    }
  }
  PhysicalRootAnalysis physical_;
  std::unordered_map<const VarObj *, size_t> indices_;
  Map<Var, Array<Expr>> descriptor_inputs_;
  Map<Var, Expr> definitions_;
};

// A compound statement is a lifetime boundary, not an ordering between GPU
// threads. Both arms of an If may execute concurrently in different warps.
// Loop-carried allocations remain live over the entire loop. Aggressive mode
// can shorten only allocations defined afresh within a loop iteration.
class SharedLiveness {
public:
  SharedLiveness(const tirx::Function &function,
                 std::vector<Allocation> *allocations, bool aggressive)
      : uses_(function, *allocations), allocations_(*allocations),
        aggressive_(aggressive) {
    for (size_t i = 0; i < allocations_.size(); ++i)
      definitions_.emplace(allocations_[i].tensor.get(), i);
    Visit(function->body);
    for (auto &allocation : allocations_) {
      if (allocation.first == std::numeric_limits<int>::max()) {
        allocation.first = allocation.definition;
        allocation.last = allocation.first + 1;
      }
      // An asynchronous instruction can retain an address after its source
      // statement. Until its matching completion protocol is proved, reserve
      // the affected storage through kernel completion, including descriptors.
      if (allocation.asynchronous)
        allocation.last = tick_ + 1;
    }
  }

private:
  static void Append(Indices *to, const Indices &from) {
    to->insert(from.begin(), from.end());
  }
  Indices Touch(AnyView object, int point,
                const CallObj *tracked_tma_store = nullptr) {
    Indices touched = uses_.Get(object);
    bool asynchronous = false;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        object, [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
          if (const auto *call = node.as<CallObj>()) {
            const auto *operation = call->op.as<OpObj>();
            const std::string name =
                operation ? std::string(operation->name) : std::string();
            asynchronous |= call != tracked_tma_store &&
                            (name.find("cp_async") != std::string::npos ||
                             name.find("tma_load") != std::string::npos ||
                             name.find("tma_store") != std::string::npos ||
                             name.find("wgmma") != std::string::npos ||
                             name.find("tcgen05") != std::string::npos ||
                             tvm::SideEffect(GetRef<Call>(call)) ==
                                 CallEffectKind::kOpaque);
          }
          if (node.as<ReturnObj>())
            asynchronous = true;
          return ffi::WalkResult::Advance();
        });
    for (size_t index : touched) {
      auto &allocation = allocations_[index];
      allocation.first = std::min(allocation.first, point);
      allocation.last = std::max(allocation.last, point + 1);
      allocation.asynchronous |= asynchronous;
    }
    return touched;
  }
  void Widen(const Indices &indices, int first, int last, bool loop) {
    for (size_t index : indices) {
      auto &allocation = allocations_[index];
      if (loop && aggressive_ && allocation.definition >= first)
        continue;
      allocation.first = std::min(allocation.first, first);
      allocation.last = std::max(allocation.last, last);
    }
  }
  Indices VisitSequence(const SeqStmtObj *sequence) {
    Indices touched, uncommitted, committed;
    auto retain_unfinished = [&]() {
      for (size_t index : uncommitted)
        allocations_[index].asynchronous = true;
      for (size_t index : committed)
        allocations_[index].asynchronous = true;
      uncommitted.clear();
      committed.clear();
    };
    // A bulk store, commit and wait must execute in this same straight-line
    // sequence. Do not match a conditional/loop-local wait to an outer store,
    // or let a sibling branch's wait release another thread's pending work.
    for (const Stmt &child : sequence->seq) {
      const auto *evaluate = child.as<EvaluateObj>();
      const auto *call = evaluate ? evaluate->value.as<CallObj>() : nullptr;
      const auto *operation = call ? call->op.as<OpObj>() : nullptr;
      if (operation == tl::tma_store().get()) {
        Indices inputs = Touch(child, tick_++, call);
        Append(&touched, inputs);
        Append(&uncommitted, inputs);
      } else if (operation == tl::tma_store_arrive().get()) {
        Append(&touched, Visit(child));
        Append(&committed, uncommitted);
        uncommitted.clear();
      } else if (operation == tl::tma_store_wait().get()) {
        Append(&touched, Visit(child));
        auto args = DecodeCallOperands(call);
        const auto *count = args.empty() ? nullptr : args[0].as<IntImmObj>();
        if (count && count->value == 0) {
          // Both wait_group 0 and wait_group.read 0 finish reading the shared
          // source. A nonzero/symbolic count is kept conservatively pending;
          // an uncommitted store is never covered by any wait_group.
          for (size_t index : committed)
            allocations_[index].last =
                std::max(allocations_[index].last, tick_);
          Append(&touched, committed);
          committed.clear();
        }
      } else {
        const bool straight_line =
            child.as<BindObj>() || child.as<TensorStoreObj>() ||
            (evaluate && (!call || tvm::SideEffect(GetRef<Call>(call)) !=
                                       CallEffectKind::kOpaque));
        if (!straight_line)
          retain_unfinished();
        Append(&touched, Visit(child));
      }
    }
    retain_unfinished();
    return touched;
  }
  Indices Visit(const Stmt &statement) {
    const int begin = tick_++;
    Indices touched;
    if (const auto *sequence = statement.as<SeqStmtObj>()) {
      touched = VisitSequence(sequence);
    } else if (const auto *loop = statement.as<ForObj>()) {
      Append(&touched, Touch(loop->begin, begin));
      Append(&touched, Touch(loop->end, begin));
      Append(&touched, Touch(loop->step, begin));
      Append(&touched, Visit(loop->body));
      Widen(touched, begin, ++tick_, true);
    } else if (const auto *loop = statement.as<WhileObj>()) {
      Append(&touched, Touch(loop->condition, begin));
      Append(&touched, Visit(loop->body));
      Widen(touched, begin, ++tick_, true);
    } else if (const auto *branch = statement.as<IfObj>()) {
      Append(&touched, Touch(branch->condition, begin));
      Append(&touched, Visit(branch->then_case));
      if (branch->else_case)
        Append(&touched, Visit(*branch->else_case));
      Widen(touched, begin, ++tick_, false);
    } else if (const auto *region = statement.as<RegionStmtObj>()) {
      Append(&touched, Touch(region->args, begin));
      Append(&touched, Visit(region->body));
      // Ordinary thread binders and metadata regions do not reorder their
      // sequential body. Region operators with other execution models can.
      const std::string name = region->op->name;
      bool sequential = region->op.same_as(launch_thread_op()) ||
                        IsLexicalScope(region) || name == "tl.attr";
      if (name == "tl.attr") {
        auto metadata = ReadAnnotations(region);
        auto key = metadata.find("key");
        if (key != metadata.end() &&
            (*key).second.cast<String>() == "kWarpSpecializationScope")
          sequential = false;
      }
      if (!sequential)
        Widen(touched, begin, ++tick_, false);
    } else if (const auto *binding = statement.as<BindObj>()) {
      auto found = definitions_.find(binding->var.get());
      if (found != definitions_.end())
        allocations_[found->second].definition = begin;
      Append(&touched, Touch(binding->value, begin));
      Append(&touched, Touch(binding->var->ty, begin));
    } else {
      Append(&touched, Touch(statement, begin));
    }
    return touched;
  }

  SharedUses uses_;
  std::vector<Allocation> &allocations_;
  std::unordered_map<const VarObj *, size_t> definitions_;
  bool aggressive_;
  int tick_{0};
};

int64_t AlignConstant(int64_t value, int alignment) {
  TVM_FFI_CHECK(value >= 0 && value <= std::numeric_limits<int64_t>::max() -
                                           (alignment - 1),
                ValueError)
      << "Shared arena size overflows int64";
  return (value + alignment - 1) / alignment * alignment;
}

PrimExpr AlignExpression(PrimExpr value, int alignment) {
  if (const auto *constant = value.as<IntImmObj>())
    return IntImm(
        value.ty(),
        AlignConstant(static_cast<int64_t>(constant->value), alignment));
  PrimExpr a = IntImm(value.ty(), alignment);
  return FloorDiv(value + IntImm(value.ty(), alignment - 1), a) * a;
}

bool Overlap(const Allocation &a, const Allocation &b, bool disable_reuse) {
  return disable_reuse || (a.first < b.last && b.first < a.last);
}

PrimExpr Pack(std::vector<Allocation> *allocations, bool disable_reuse,
              int alignment, const sym::Analyzer &analyzer) {
  std::vector<size_t> order(allocations->size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
    const auto &a = allocations->at(lhs), &b = allocations->at(rhs);
    if (a.first != b.first)
      return a.first < b.first;
    return a.tensor->name < b.tensor->name;
  });
  std::vector<size_t> placed;
  int64_t constant_end = 0;
  for (size_t index : order) {
    auto &allocation = allocations->at(index);
    const auto *size = allocation.size.as<IntImmObj>();
    if (!size)
      continue;
    TVM_FFI_CHECK_GE(size->value, 0, ValueError);
    int64_t offset = 0;
    bool retry = true;
    while (retry) {
      retry = false;
      offset = AlignConstant(offset, allocation.alignment);
      TVM_FFI_CHECK_LE(size->value,
                       std::numeric_limits<int64_t>::max() - offset,
                       ValueError);
      for (size_t previous : placed) {
        const auto &other = allocations->at(previous);
        if (!Overlap(allocation, other, disable_reuse))
          continue;
        const int64_t other_begin =
            static_cast<int64_t>(other.offset.as_or_throw<IntImm>()->value);
        const int64_t other_end =
            other_begin +
            static_cast<int64_t>(other.size.as_or_throw<IntImm>()->value);
        if (offset < other_end && other_begin < offset + size->value) {
          offset = other_end;
          retry = true;
          break;
        }
      }
    }
    allocation.offset = IntImm(PrimType::Int(64), offset);
    constant_end =
        std::max(constant_end, offset + static_cast<int64_t>(size->value));
    placed.push_back(index);
  }
  PrimExpr cursor = IntImm(PrimType::Int(64), constant_end);
  for (size_t index : order) {
    auto &allocation = allocations->at(index);
    if (allocation.size.as<IntImmObj>())
      continue;
    cursor = analyzer->Simplify(AlignExpression(cursor, allocation.alignment));
    allocation.offset = cursor;
    cursor = analyzer->Simplify(cursor + allocation.size);
  }
  return analyzer->Simplify(AlignExpression(cursor, alignment));
}

class ArenaRewriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  Map<Var, Expr> replacements;
  UnchangedOr<Stmt> Mutate_(const BindObj *binding, InplaceMode mode) final {
    auto value = replacements.Get(binding->var);
    if (!value)
      return StmtExprMutator::Mutate_(binding, mode);
    // The tensor definition stays at its lexical position with exactly the
    // original immutable type. The raw arena alone is hoisted; alias views,
    // strides, offsets, packed dtypes and all existing uses remain intact.
    return Bind(binding->var, *value, binding->loc);
  }
};

tirx::Function Merge(tirx::Function function, bool merge_static,
                     bool aggressive, int align_bytes, bool disable_reuse,
                     bool verbose) {
  if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return function;
  if (auto target = function->GetAttr<Target>(tvm::attr::kTarget);
      target && target.value()->kind->name == "webgpu") {
    // WGSL workgroup storage consists of named, typed arrays. Keep those
    // definitions instead of introducing unrepresentable C pointer aliases.
    return function;
  }
  TVM_FFI_CHECK(align_bytes > 0 && !(align_bytes & (align_bytes - 1)) &&
                    align_bytes <= (1 << 27),
                ValueError)
      << "Shared arena alignment must be a power of two in [1, 2^27]";
  std::vector<Allocation> all;
  Map<Var, Expr> definitions;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      function->body,
      [&](const BindObj *binding) -> ffi::Expected<ffi::WalkResult> {
        definitions.Set(binding->var, binding->value);
        auto tensor = binding->var.as<TensorVar>();
        const auto *call = binding->value.as<CallObj>();
        if (!tensor || !call || !call->op.same_as(alloc_tensor_op()))
          return ffi::WalkResult::Advance();
        const std::string scope = tensor->scope().name();
        if (scope != "shared.dyn" && !(merge_static && scope == "shared"))
          return ffi::WalkResult::Advance();
        all.push_back({GetRef<Bind>(binding), *tensor});
        return ffi::WalkResult::Advance();
      });
  if (all.empty())
    return function;
  SharedLiveness liveness(function, &all, aggressive);
  auto analyzer = CreateTileLangAnalyzer();
  std::unordered_set<const VarObj *> parameters;
  for (const Var &parameter : function->params) {
    parameters.insert(parameter.get());
    for (const Var &symbol : UndefinedVars(parameter->ty))
      parameters.insert(symbol.get());
  }
  auto expander =
      make_object<ShapeExpander>(definitions, std::move(parameters));
  Map<String, int64_t> required;
  if (auto attrs = function->GetAttr<Map<String, Any>>(kSmemAlignmentMap)) {
    for (const auto &[name, value] : *attrs) {
      int64_t alignment =
          value.as<IntImmObj>()
              ? static_cast<int64_t>(value.as<IntImmObj>()->value)
              : value.cast<int64_t>();
      TVM_FFI_CHECK(alignment > 0 && !(alignment & (alignment - 1)) &&
                        alignment <= (1 << 27),
                    ValueError);
      required.Set(name, alignment);
    }
  }
  SharedUses uses(function, all);
  std::unordered_set<size_t> swizzle_sensitive;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      function->body,
      [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
        const auto *operation = call->op.as<OpObj>();
        if (operation == tl::tma_load().get() ||
            operation == tl::tma_store().get() ||
            operation == tl::initialize_wgmma_descriptor().get() ||
            operation == tl::initialize_tcgen05_descriptor().get()) {
          auto touched = uses.Get(GetRef<Call>(call));
          swizzle_sensitive.insert(touched.begin(), touched.end());
        }

        return ffi::WalkResult::Advance();
      });
  auto rewriter = make_object<ArenaRewriter>();
  Array<Stmt> prefix;
  for (const std::string &scope :
       {std::string("shared.dyn"), std::string("shared")}) {
    std::vector<Allocation> group;
    int maximum_alignment = align_bytes;
    size_t count = 0;
    for (const auto &allocation : all)
      count += allocation.tensor.scope().name() == scope;
    if (count <= 1)
      continue;
    for (size_t index = 0; index < all.size(); ++index) {
      Allocation allocation = all[index];
      if (allocation.tensor.scope().name() != scope)
        continue;
      const auto *call = allocation.binding->value.as<CallObj>();
      bool empty_placement =
          call->args.empty() ||
          (call->args.size() == 1 && call->args[0].as<tvm::TupleObj>() &&
           call->args[0].as<tvm::TupleObj>()->fields.empty());
      TVM_FFI_CHECK(empty_placement, ValueError)
          << "Shared arena merge requires placement coordinates to be lowered "
             "first";
      bool supported_annotations = true;
      if (call->attrs) {
        for (const auto &[key, unused] :
             call->attrs.value().as_or_throw<DictAttrs>()->dict)
          supported_annotations &= key == tirx::attr::kVolatile.name;
      }
      TVM_FFI_CHECK(supported_annotations, ValueError)
          << "Shared arena merge requires allocation annotations to be "
             "consumed first: "
          << allocation.tensor;
      TVM_FFI_CHECK(allocation.tensor.shape().size() <= 1 &&
                        allocation.tensor.ty()->HasDefaultLayout(),
                    ValueError)
          << "MergeSharedMemoryAllocations requires flattened compact "
             "allocations";
      allocation.alignment =
          std::max(align_bytes, allocation.tensor.ty()->data_alignment);
      if (auto requirement = required.Get(allocation.tensor->name))
        allocation.alignment =
            std::max(allocation.alignment, static_cast<int>(*requirement));
      else if (swizzle_sensitive.count(index))
        allocation.alignment = std::max(allocation.alignment, 1024);
      maximum_alignment = std::max(maximum_alignment, allocation.alignment);
      TVM_FFI_CHECK(CanHoistSharedShape(allocation.tensor.ty()), ValueError)
          << "Shared allocation shape contains memory reads or unsafe "
             "evaluation";
      auto expanded_type =
          expander->Mutate(allocation.tensor.ty(), InplaceMode::kDisallow)
              .ValueOrUnchanged(allocation.tensor.ty());
      allocation.size = analyzer->Simplify(
          GetBufferStorageSizeBytes(expanded_type.cast<TensorType>()));
      group.push_back(std::move(allocation));
    }
    PrimExpr size = Pack(&group, disable_reuse, maximum_alignment, analyzer);
    // C-family arrays cannot have zero length. Empty logical tensors still
    // have no legal accesses, but their shared backing must remain well-formed.
    size = analyzer->Simplify(max(size, IntImm(size.ty(), 1)));
    TensorType type({size}, PrimType::UInt(8), runtime::MemoryScope::Get(scope),
                    maximum_alignment);
    TensorVar arena(scope == "shared.dyn" ? "buf_dyn_shmem" : "buf_shmem",
                    type);
    prefix.push_back(
        Bind(arena, Call(type, alloc_tensor_op(), {}, {}, {type})));
    for (const auto &allocation : group) {
      Expr pointer =
          Call(allocation.tensor.ty()->DataPtrType(), ptr_byte_offset_op(),
               {arena.data(), allocation.offset});
      rewriter->replacements.Set(
          allocation.tensor,
          Call(allocation.tensor.ty(), decl_tensor_op(), {pointer},
               allocation.binding->value.as_or_throw<Call>()->attrs,
               {allocation.tensor.ty()}, allocation.binding->loc));
      if (verbose)
        LOG(INFO) << "Shared arena " << scope << ": " << allocation.tensor->name
                  << " offset=" << allocation.offset
                  << " bytes=" << allocation.size
                  << " alignment=" << allocation.alignment << " lifetime=["
                  << allocation.first << "," << allocation.last << ")";
    }
  }
  if (prefix.empty())
    return function;
  SeqStmt body = function->body;
  body = rewriter->Mutate(body, InplaceMode::kDisallow).ValueOrUnchanged(body);
  for (const Stmt &statement : body->seq)
    prefix.push_back(statement);
  function.CopyOnWrite()->body = SeqStmt(prefix, body->loc);
  return function;
}
} // namespace

namespace transform {
tvm::transform::Pass
MergeSharedMemoryAllocations(bool enable_aggressive_merge = false,
                             int align_bytes = 16, bool disable_reuse = false) {
  auto pass = [=](tirx::Function function, IRModule,
                  tvm::transform::PassContext context) {
    return Merge(
        std::move(function),
        context->GetConfig<bool>("tirx.merge_static_smem").value_or(false),
        enable_aggressive_merge, align_bytes, disable_reuse,
        context->GetConfig<bool>(kDebugMergeSharedMemoryAllocations)
            .value_or(false));
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.MergeSharedMemoryAllocations");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.MergeSharedMemoryAllocations",
                                   MergeSharedMemoryAllocations);
}
} // namespace transform
} // namespace tvm::tl
