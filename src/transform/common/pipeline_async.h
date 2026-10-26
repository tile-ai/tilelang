#include "compat/metadata.h"
#ifndef TVM_TL_TRANSFORM_COMMON_PIPELINE_ASYNC_H_
#define TVM_TL_TRANSFORM_COMMON_PIPELINE_ASYNC_H_

#include "cuda/op/copy.h"
#include "cuda/target_utils.h"
#include "pipeline_utils.h"

#include <tvm/ir/transform.h>

#include <map>
#include <set>

namespace tvm::tl {

inline constexpr const char *kPipelineCPAsyncScope = "tl.pipeline_cp_async";

inline bool PipelineExplicitCPAsync(const PipelineStatement &statement) {
  bool result = false;
  PostOrderVisit(statement.statement, [&](const ffi::ObjectRef &object) {
    if (const auto *call = object.as<CallNode>()) {
      result |= call->op.same_as(ptx_cp_async()) ||
                call->op.same_as(Op::Get("tl.tileop.async_copy"));
      if (auto operation = ParseOperator(ffi::GetRef<Call>(call));
          const auto *copy = operation.as<CopyNode>())
        if (auto flag = copy->annotations.Get("is_async_copy"))
          result |= GetIntAnnotation(*flag).value_or(0) != 0;
    }
  });
  return result;
}

inline bool PipelineAsyncCandidate(const PipelineStatement &statement,
                                   const Target &target) {
  if (!TargetCudaHasAsyncCopy(target))
    return false;
  bool automatic = tvm::transform::PassContext::Current()
                       ->GetConfig<bool>(kEnableAsyncCopy)
                       .value_or(true);
  int candidates = 0;
  PostOrderVisit(statement.statement, [&](const ffi::ObjectRef &object) {
    if (const auto *call = object.as<CallNode>()) {
      auto operation = ParseOperator(ffi::GetRef<Call>(call));
      if (const auto *copy = operation.as<CopyNode>())
        candidates += cuda::IsPipelineManagedCPAsyncCopy(*copy, target);
      else if (call->op.same_as(ptx_cp_async()))
        ++candidates;
    }
    if (const auto *store = object.as<TensorStoreNode>(); store && automatic) {
      auto destination = store->dest.as<TensorVar>();
      const auto *load = store->value.as<TensorLoadNode>();
      auto source = load ? load->source.as<TensorVar>() : std::nullopt;
      candidates += source && destination && IsGlobalBuffer(*source) &&
                    IsSharedBuffer(*destination) &&
                    source.value().dtype() == destination.value().dtype() &&
                    source.value().dtype().bits() % 8 == 0;
    }
  });
  bool shared_write = false;
  bool shared_read = false;
  bool safe = true;
  for (const MemoryAccess &access : statement.accesses) {
    bool shared = access.location.scope == "shared" ||
                  access.location.scope == "shared.dyn";
    shared_write |= shared && (access.mask & kAccessWrite);
    shared_read |= shared && (access.mask & kAccessRead);
    if (shared && (access.mask & kAccessRead))
      safe &= access.location.allocation &&
              !(statement.memory.at(access.location.allocation.value().get()) &
                kAccessWrite);
    safe &=
        !(access.location.scope == "global" && (access.mask & kAccessWrite));
  }
  safe &= !shared_read || candidates == 1;
  TVM_FFI_CHECK(safe || !PipelineExplicitCPAsync(statement), ValueError)
      << "Explicit asynchronous pipeline copies require a separate statement "
         "whose address reads do not alias its destination";
  return candidates > 0 && shared_write && safe;
}

struct PipelineAsyncPlan {
  std::vector<int> producers;
  std::vector<int> groups;
  std::vector<int> group_last;
  std::vector<int> group_stage;
  std::vector<size_t> execution;

  int WaitBefore(size_t consumer, const PipelineBody &body,
                 const std::vector<int> &stages,
                 const std::vector<int> &order) const {
    int committed = 0;
    for (int last : group_last)
      committed += last < order[consumer];
    int wait = -1;
    for (size_t producer = 0; producer < producers.size(); ++producer) {
      if (!producers[producer] || producer == consumer ||
          !(stages[producer] < stages[consumer] ||
            (stages[producer] == stages[consumer] &&
             order[producer] < order[consumer])) ||
          !PipelineDependency(body.statements[producer],
                              body.statements[consumer]))
        continue;
      int pending = (stages[consumer] - stages[producer]) *
                        static_cast<int>(group_last.size()) +
                    committed - groups[producer] - 1;
      TVM_FFI_CHECK(pending >= 0, ValueError)
          << "An asynchronous pipeline consumer precedes its commit group";
      pending = std::min(pending, 7);
      wait = wait < 0 ? pending : std::min(wait, pending);
    }
    return wait;
  }
};

inline std::vector<int> PipelineGroupArray(const ffi::Any &value) {
  std::vector<int> result;
  for (const ffi::Any &element : value.cast<ffi::Array<ffi::Any>>()) {
    auto integer = GetIntAnnotation(element);
    TVM_FFI_CHECK(integer && *integer >= -1 && *integer <= 1024, ValueError)
        << "Pipeline asynchronous group IDs must be integers in [-1, 1024]";
    result.push_back(static_cast<int>(*integer));
  }
  return result;
}

inline PipelineAsyncPlan
PlanPipelineAsync(const PipelineBody &body, const std::vector<int> &stages,
                  const std::vector<int> &order,
                  const ffi::Map<ffi::String, ffi::Any> &attrs,
                  const ffi::Optional<Target> &target) {
  PipelineAsyncPlan result;
  size_t count = body.statements.size();
  result.producers.resize(count, 0);
  result.groups.resize(count, -1);
  result.execution.resize(count);
  std::iota(result.execution.begin(), result.execution.end(), 0);
  std::sort(
      result.execution.begin(), result.execution.end(),
      [&](size_t left, size_t right) { return order[left] < order[right]; });
  auto explicit_mask = attrs.Get(kPipelineAsyncProducers);
  auto explicit_groups = attrs.Get(kPipelineAsyncProducerGroups);
  auto explicit_stages = attrs.Get("software_pipeline_async_stages");
  if (explicit_mask) {
    result.producers =
        PipelineIntegerArray(*explicit_mask, kPipelineAsyncProducers);
    TVM_FFI_CHECK(result.producers.size() == count, ValueError)
        << "Pipeline asynchronous producer mask must match the scheduled "
           "statements";
    for (int producer : result.producers)
      TVM_FFI_CHECK(producer <= 1, ValueError)
          << "Pipeline asynchronous producer mask must be boolean";
  }
  if (explicit_groups) {
    result.groups = PipelineGroupArray(*explicit_groups);
    TVM_FFI_CHECK(result.groups.size() == count, ValueError)
        << "Pipeline asynchronous groups must match the scheduled statements";
  }
  std::set<int> selected_stages;
  if (explicit_stages) {
    auto values = PipelineIntegerArray(*explicit_stages,
                                       "software_pipeline_async_stages");
    selected_stages.insert(values.begin(), values.end());
  }
  bool requested = explicit_mask || explicit_groups || explicit_stages;
  if (requested) {
    TVM_FFI_CHECK(target && TargetCudaHasAsyncCopy(*target), ValueError)
        << "Asynchronous software pipelines require a CUDA target with "
           "cp.async support";
  }
  if (!target || !TargetCudaHasAsyncCopy(*target))
    return result;
  for (size_t index = 0; index < count; ++index) {
    bool eligible = PipelineAsyncCandidate(body.statements[index], *target);
    if (!explicit_mask)
      result.producers[index] =
          eligible &&
          (!explicit_stages || selected_stages.count(stages[index]));
    else if (!eligible)
      result.producers[index] = 0;
    TVM_FFI_CHECK(!eligible ||
                      !PipelineExplicitCPAsync(body.statements[index]) ||
                      result.producers[index],
                  ValueError)
        << "Explicit asynchronous copies must belong to a managed pipeline "
           "group";
    if (!result.producers[index])
      result.groups[index] = -1;
  }
  if (std::any_of(result.producers.begin(), result.producers.end(),
                  [](int producer) { return producer != 0; }))
    for (const PipelineStatement &statement : body.statements)
      PostOrderVisit(statement.statement, [](const ffi::ObjectRef &object) {
        const auto *call = object.as<CallNode>();
        const auto *operation = call ? call->op.as<OpNode>() : nullptr;
        if (operation)
          TVM_FFI_CHECK(operation->name != "tirx.ptx.cp_async_commit_group" &&
                            operation->name != "tirx.ptx.cp_async_wait_group" &&
                            operation->name != "tirx.ptx.cp_async_wait_all",
                        ValueError)
              << "Manual cp.async commit/wait cannot be mixed with managed "
                 "software pipeline groups";
      });
  std::map<int, int> normalized;
  int previous = -1;
  for (size_t index : result.execution) {
    if (!result.producers[index]) {
      previous = -1;
      continue;
    }
    int group = -1;
    if (explicit_groups) {
      int supplied = result.groups[index];
      TVM_FFI_CHECK(supplied >= 0, ValueError)
          << "Asynchronous producers require nonnegative group IDs";
      auto found = normalized.find(supplied);
      if (found != normalized.end()) {
        group = found->second;
        TVM_FFI_CHECK(previous == group, ValueError)
            << "An asynchronous commit group cannot straddle a consumer";
      } else {
        group = static_cast<int>(result.group_last.size());
        normalized.emplace(supplied, group);
      }
    } else if (previous >= 0 && result.group_stage[previous] == stages[index]) {
      group = previous;
      for (size_t producer = 0; producer < count; ++producer)
        if (order[producer] < order[index] &&
            result.groups[producer] == group &&
            PipelineDependency(body.statements[producer],
                               body.statements[index]))
          group = -1;
    }
    if (group < 0)
      group = static_cast<int>(result.group_last.size());
    if (group == static_cast<int>(result.group_last.size())) {
      result.group_last.push_back(order[index]);
      result.group_stage.push_back(stages[index]);
    } else {
      TVM_FFI_CHECK(result.group_stage[group] == stages[index], ValueError)
          << "An asynchronous commit group must belong to one pipeline stage";
      result.group_last[group] = order[index];
    }
    result.groups[index] = group;
    previous = group;
  }
  return result;
}

inline Stmt PipelineAsyncCommit() {
  return Evaluate(
      Call(std::nullopt, Op::Get("tirx.ptx.cp_async_commit_group"),
           {StringImm("async"), StringImm("commit_group"), StringImm("")}));
}

inline Stmt PipelineAsyncWait(int pending) {
  return Evaluate(Call(std::nullopt, Op::Get("tirx.ptx.cp_async_wait_group"),
                       {PrimExpr(pending), StringImm("async"),
                        StringImm("wait_group"), StringImm("")}));
}

class PipelineAsyncMarker : public tvm::tl::StmtExprMutator {
public:
  explicit PipelineAsyncMarker(Target target) : target_(std::move(target)) {}

private:
  UnchangedOr<Expr> Mutate_(const CallNode *node, InplaceMode mode) final {
    Call result = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                      .ValueOrUnchanged(ffi::GetRef<Call>(node))
                      .as_or_throw<Call>();
    auto operation = ParseOperator(result);
    if (const auto *copy = operation.as<CopyNode>();
        copy && cuda::IsPipelineManagedCPAsyncCopy(*copy, target_)) {
      DictAttrs attrs = result->attrs
                            ? result->attrs.value().as_or_throw<DictAttrs>()
                            : DictAttrs();
      attrs.CopyOnWrite()->dict.Set(attr::kAsyncCopyNoImplicitCommitWait,
                                    IntImm::Bool(true));
      result.CopyOnWrite()->attrs = attrs;
    }
    return result;
  }

  Target target_;
};

} // namespace tvm::tl

#endif
