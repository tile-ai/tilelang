#include "compat/visit.h"
#include "cpu/op/builtin.h"
#include "transform/common/constr_visitor.h"
#include "transform/common/memory_access.h"

#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <unordered_set>
#include <vector>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

struct GridAccess {
  MemoryLocation location;
  PrimExpr offset;
  int bytes;
  bool write;
  ConstrSet constraints;
};

class GridAccessCollector : public ConstrVisitor {
public:
  explicit GridAccessCollector(MemoryAliases aliases)
      : aliases(std::move(aliases)) {}
  bool safe = true;
  std::string reason = "access independence cannot be proven safe";
  MemoryAliases aliases;
  std::vector<GridAccess> accesses;
  std::unordered_set<const VarNode *> private_storage;

private:
  using Result = Optional<VisitInterrupt>;

  bool BoundedMath(const PrimExpr &expression, const ConstrSet &constraints) {
    sym::Analyzer analyzer;
    auto context = constraints.Populate(analyzer);
    bool bounded = true;
    PostOrderVisit(analyzer->Simplify(expression), [&](const ObjectRef &node) {
      if (!node.as<AddNode>() && !node.as<SubNode>() && !node.as<MulNode>())
        return;
      PrimExpr value = node.as_or_throw<PrimExpr>();
      PrimType type = value.ty();
      if (!type.IsScalar() ||
          !type.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
        bounded = false;
        return;
      }
      auto bound = analyzer->const_int_bound(value);
      bounded &= bound->min_value > sym::ConstIntBound::kNegInf &&
                 bound->max_value < sym::ConstIntBound::kPosInf &&
                 ffi::BigInt(bound->min_value) >=
                     min_value(type).as_or_throw<IntImm>()->value &&
                 ffi::BigInt(bound->max_value) <=
                     max_value(type).as_or_throw<IntImm>()->value;
    });
    return bounded;
  }

  void Record(const Expr &storage, const Array<PrimExpr> &indices, bool write) {
    auto tensor = storage.as<TensorVar>();
    if (!tensor) {
      safe = false;
      reason = "memory access does not have a native tensor type";
      return;
    }
    MemoryLocation root = aliases.Resolve(storage);
    if (root.allocation && private_storage.count(root.allocation.value().get()))
      return;
    Array<PrimExpr> coordinates;
    ConstrSet constraints = GetConstrSet();
    for (Constr &constraint : constraints.constrs_) {
      FreshenMutableReads freshen;
      if (constraint.kind == Constr::kConstr)
        constraint.value = freshen(constraint.value);
      else
        constraint = constraint.FreshenReads();
    }
    Optional<PrimVar> lane;
    int lanes = 1;
    for (const PrimExpr &index : indices) {
      if (const auto *ramp = index.as<RampNode>()) {
        auto extent = GetConstInt(ramp->lanes);
        if (!extent || *extent <= 0 || *extent > INT32_MAX) {
          safe = false;
          return;
        }
        int count = *extent;
        if (lane && count != lanes) {
          safe = false;
          return;
        }
        if (!lane) {
          lane = PrimVar("access_lane", ramp->base.ty());
          lanes = count;
          constraints.AddConstr(
              *lane, Range::FromMinExtent(MakeConst(lane.value().ty(), 0),
                                          MakeConst(lane.value().ty(), count)));
        }
        coordinates.push_back(ramp->base + *lane * ramp->stride);
      } else if (index.ty().IsScalar()) {
        coordinates.push_back(index);
      } else {
        safe = false;
        return;
      }
    }
    if (coordinates.size() != tensor.value().shape().size()) {
      safe = false;
      return;
    }
    MemoryLocation location = aliases.Element(storage, coordinates);
    int bits = tensor.value().dtype().StorageBytes() * 8;
    if (!location.precise || !location.allocation) {
      safe = false;
      reason = "memory access has an unresolved storage alias";
      return;
    }
    if (!BoundedMath(location.offset, constraints)) {
      safe = false;
      reason = "integer overflow in memory addressing cannot be excluded";
      return;
    }
    FreshenMutableReads freshen;
    sym::Analyzer analyzer;
    auto context = constraints.Populate(analyzer);
    accesses.push_back({location, freshen(analyzer->Simplify(location.offset)),
                        bits / 8, write, constraints});
  }

  Result Visit_(const TensorLoadNode *node) final {
    Record(node->source, node->indices, false);
    for (const PrimExpr &index : node->indices)
      Visit(index);
    return std::nullopt;
  }

  Result Visit_(const TensorStoreNode *node) final {
    Visit(node->value);
    for (const PrimExpr &index : node->indices)
      Visit(index);
    Record(node->dest, node->indices, true);
    return std::nullopt;
  }

  Result Visit_(const BindNode *node) final {
    aliases.Bind(node);
    if (const auto *call = node->value.as<CallNode>();
        call && call->op.same_as(alloc_tensor_op())) {
      private_storage.insert(node->var.get());
      for (const Expr &argument : call->args)
        Visit(argument);
      return std::nullopt;
    }
    if (node->var->ty.as<PtrTypeNode>() || node->var->ty.as<TensorTypeNode>()) {
      MemoryLocation location = aliases.Resolve(node->var);
      if (!location.precise || !location.allocation ||
          SideEffect(node->value) != CallEffectKind::kPure) {
        safe = false;
        reason = "pointer binding does not have a pure resolved alias";
      }
      return std::nullopt;
    }
    return ConstrVisitor::Visit_(node);
  }

  Result Visit_(const VarNode *node) final {
    if (node->ty.as<TensorTypeNode>() || node->ty.as<PtrTypeNode>()) {
      safe = false;
      reason = "tensor or pointer escapes a tracked memory access";
    }
    return std::nullopt;
  }

  Result Visit_(const CallNode *node) final {
    if (SideEffect(GetRef<Expr>(node)) > CallEffectKind::kReadState) {
      safe = false;
      reason = "effects that cannot be proven safe";
    }
    return ConstrVisitor::Visit_(node);
  }

  Result Visit_(const WhileNode *node) final {
    safe = false;
    return ConstrVisitor::Visit_(node);
  }

  Result Visit_(const ReturnNode *) final {
    safe = false;
    return std::nullopt;
  }
  Result Visit_(const BreakNode *) final {
    safe = false;
    return std::nullopt;
  }
  Result Visit_(const ContinueNode *) final {
    safe = false;
    return std::nullopt;
  }
};

class CPUGridMaterializer : public StmtExprMutator {
public:
  CPUGridMaterializer(tirx::Function function, bool collapse, int64_t min_trip)
      : function_(function), collapse_(collapse), min_trip_(min_trip) {
    if (auto indices =
            function->GetAttr<Array<IntImm>>("tl.non_restrict_param_indices")) {
      for (const IntImm &index : *indices) {
        int position = index->value.as<int>().value();
        if (position >= 0 &&
            position < static_cast<int>(function->params.size())) {
          non_restrict_.insert(function->params[position].get());
        }
      }
    }
    if (auto parameters =
            function->GetAttr<Array<Var>>(attr::kNonRestrictParams)) {
      for (const Var &parameter : *parameters)
        non_restrict_.insert(parameter.get());
    }
    for (const Var &parameter : function->params)
      parameters_.insert(parameter.get());
  }

private:
  bool Independent(const ForNode *loop,
                   const std::vector<const ForNode *> &loops, size_t depth) {
    auto collector = make_object<GridAccessCollector>(aliases_);
    for (size_t outer = 0; outer < depth; ++outer) {
      collector->constr_stack_.emplace_back(
          loops[outer]->var, Range(loops[outer]->begin, loops[outer]->end));
    }
    collector->Visit(GetRef<For>(loop));
    if (!collector->safe) {
      reason_ = collector->reason;
      return false;
    }
    bool no_alias = function_->GetAttr(tirx::attr::kNoAlias).value_or(true);
    for (size_t first = 0; first < collector->accesses.size(); ++first) {
      const GridAccess &left = collector->accesses[first];
      for (size_t second = first; second < collector->accesses.size();
           ++second) {
        const GridAccess &right = collector->accesses[second];
        if (!left.write && !right.write)
          continue;
        if (!left.location.allocation.value().same_as(
                *right.location.allocation)) {
          if (collector->aliases.Disjoint(left.location, right.location))
            continue;
          const VarNode *left_root = left.location.allocation.value().get();
          const VarNode *right_root = right.location.allocation.value().get();
          if (no_alias && parameters_.count(left_root) &&
              parameters_.count(right_root) &&
              (!non_restrict_.count(left_root) ||
               !non_restrict_.count(right_root)))
            continue;
          reason_ = "parameters may alias";
          return false;
        }
        Map<PrimVar, PrimExpr> replacements;
        ConstrSet other = right.constraints.RenameFrom("_other_iteration",
                                                       replacements, loop->var);
        if (!replacements.count(loop->var))
          return false;
        ConstrSet constraints = left.constraints.Merge(other).Normalized();
        PrimExpr other_offset = Substitute(right.offset, replacements);
        constraints.AddConstr(left.offset <
                              other_offset + IntImm::Int64(right.bytes));
        constraints.AddConstr(other_offset <
                              left.offset + IntImm::Int64(left.bytes));
        sym::Analyzer analyzer;
        if (analyzer->z3_prover.IsEnabled()) {
          analyzer->z3_prover.SetRLimit(1000000);
        }
        auto context = constraints.Populate(analyzer);
        if (!analyzer->CanProve(loop->var == replacements.at(loop->var),
                                sym::ProofStrength::kSymbolicBound)) {
          reason_ = left.bytes != right.bytes
                        ? "overlapping accesses with different element widths"
                        : "access independence cannot be proven safe";
          return false;
        }
      }
    }
    return true;
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *node, InplaceMode mode) final {
    aliases_.Bind(node);
    return StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Stmt> Mutate_(const ForNode *node, InplaceMode mode) final {
    if (!node->annotations.count(attr::kCPUGridDim)) {
      return StmtExprMutator::Mutate_(node, mode);
    }
    std::vector<const ForNode *> loops{node};
    while (loops.back()->body->seq.size() == 1) {
      auto child = loops.back()->body->seq[0].as<ForNode>();
      if (!child || !child->annotations.count(attr::kCPUGridDim))
        break;
      loops.push_back(child);
    }
    bool parallel =
        !in_parallel_ &&
        !function_->GetAttr<bool>(attr::kCPUHadAtomics).value_or(false);
    int64_t trip_count = 1;
    for (const ForNode *loop : loops) {
      auto step = GetConstInt(loop->step);
      auto begin = GetConstInt(loop->begin);
      auto end = GetConstInt(loop->end);
      if (!step || *step <= 0)
        parallel = false;
      if (!begin || !end || !step || *step <= 0 || *end <= *begin ||
          (*end - *begin - 1) / *step + 1 >
              std::numeric_limits<int64_t>::max() / trip_count) {
        if (min_trip_ > 0)
          parallel = false;
      } else {
        trip_count *= (*end - *begin - 1) / *step + 1;
      }
    }
    if (trip_count < min_trip_)
      parallel = false;
    std::vector<bool> selected(loops.size(), false);
    bool selected_one = false;
    for (size_t depth = 0; depth < loops.size() && parallel; ++depth) {
      const ForNode *loop = loops[depth];
      sym::Analyzer analyzer;
      if (analyzer->CanProve(loop->end - loop->begin <= loop->step))
        continue;
      if (!collapse_ && selected_one)
        break;
      if (!Independent(loop, loops, depth)) {
        parallel = false;
        break;
      }
      selected[depth] = true;
      selected_one = true;
    }
    if (!parallel && !in_parallel_) {
      LOG(WARNING) << "tl.cpu_parallel: grid loop `" << node->var->name
                   << "` stays serial: " << reason_;
    }
    bool previous = in_parallel_;
    in_parallel_ |= parallel && selected_one;
    SeqStmt body =
        Mutate(loops.back()->body, mode).ValueOrUnchanged(loops.back()->body);
    in_parallel_ = previous;
    Optional<Any> thread_count =
        loops.front()->annotations.Get(attr::kCPUNumThreads);
    size_t first_parallel = 0;
    while (first_parallel < selected.size() && !selected[first_parallel])
      ++first_parallel;
    for (size_t depth = loops.size(); depth-- > 0;) {
      const ForNode *loop = loops[depth];
      auto annotations = loop->annotations;
      annotations.erase(attr::kCPUGridDim);
      if (parallel && depth == first_parallel && thread_count) {
        annotations.Set("codegen.num_threads", *thread_count);
      }
      if (auto count = annotations.Get(attr::kCPUNumThreads)) {
        annotations.erase(attr::kCPUNumThreads);
      }
      body = SeqStmt(For(loop->var, loop->begin, loop->end, loop->step,
                         parallel && selected[depth] ? ForKind::kParallel
                                                     : ForKind::kDefault,
                         body, annotations, loop->loc));
    }
    return body;
  }

  tirx::Function function_;
  bool collapse_;
  int64_t min_trip_;
  bool in_parallel_ = false;
  MemoryAliases aliases_;
  std::string reason_ =
      "allocation privacy or trip count cannot be proven safe";
  std::unordered_set<const VarNode *> non_restrict_;
  std::unordered_set<const VarNode *> parameters_;
};
} // namespace

namespace transform {
tvm::transform::Pass PlanAndUpdateBufferAllocationLocation();

tvm::transform::Pass MaterializeCPUParallelGrid() {
  auto materialize = tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext context) -> Optional<tirx::Function> {
        auto target = function->GetAttr(tvm::attr::kTarget);
        if (!function->body || !target ||
            (target.value()->kind->name != "c" &&
             target.value()->kind->name != "llvm"))
          return function;
        int64_t min_trip =
            context->GetConfig<int64_t>(kCPUParallelMinTrip).value_or(0);
        TVM_FFI_CHECK(min_trip >= 0, ValueError)
            << kCPUParallelMinTrip << " must be nonnegative";
        auto rewriter = make_object<CPUGridMaterializer>(
            function, target.value()->kind->name == "c", min_trip);
        SeqStmt body = *function->body;
        function.CopyOnWrite()->body =
            rewriter->Mutate(body).ValueOrUnchanged(body);
        return function;
      },
      0, "tl.MaterializeCPUParallelGrid");
  return tvm::transform::Sequential(
      {PlanAndUpdateBufferAllocationLocation(), materialize},
      "tl.MaterializeCPUParallelGrid");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.cpu.transform.MaterializeCPUParallelGrid",
                              MaterializeCPUParallelGrid);
}
} // namespace transform
} // namespace tvm::tl
