#include "compat/analyzer.h"
#include "compat/expr.h"
#include "compat/metadata.h"
#include "op/builtin.h"
#include "op/gemm.h"
#include "op/reducer.h"
#include "span_utils.h"

#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

struct InitializationEvent {
  TensorVar tensor;
  const Object *owner;
  Location location;
  bool write;
  bool gemm;
};

class InitializationAccessCollector : public tvm::tl::StmtExprVisitor {
public:
  std::vector<InitializationEvent> events;

private:
  Optional<VisitInterrupt> Visit_(const ForNode *node) final {
    loops_.push_back(node);
    auto result = tvm::tl::StmtExprVisitor::Visit_(node);
    loops_.pop_back();
    return result;
  }

  Optional<VisitInterrupt> Visit_(const IfNode *node) final {
    ++guard_depth_;
    auto result = tvm::tl::StmtExprVisitor::Visit_(node);
    --guard_depth_;
    return result;
  }

  bool ClearsAtLoopEntry(const GemmNode *gemm) const {
    if (!gemm || guard_depth_ != 0 || loops_.empty())
      return false;
    Map<PrimVar, PrimExpr> beginnings;
    for (const ForNode *loop : loops_)
      beginnings.Set(loop->var, Substitute(loop->begin, beginnings));
    return tl::arith::CanProve(sym::Analyzer(),
                               Substitute(gemm->clearAccum_, beginnings));
  }

  std::vector<TensorVar> Origins(const Expr &value) {
    if (const auto *variable = value.as<VarNode>()) {
      if (auto found = aliases_.find(variable); found != aliases_.end())
        return found->second;
      if (auto tensor = value.as<TensorVar>())
        return {*tensor};
    }
    if (const auto *load = value.as<TensorLoadNode>())
      return Origins(load->source);
    if (const auto *region = value.as<TensorRegionNode>())
      return Origins(region->source);
    if (const auto *call = value.as<CallNode>()) {
      if (!call->args.empty() && (call->op.same_as(access_ptr()) ||
                                  call->op.same_as(address_of_op()) ||
                                  call->op.same_as(tensor_data_ptr_op()) ||
                                  call->op.same_as(decl_tensor_op()))) {
        return Origins(call->args[0]);
      }
      std::vector<TensorVar> result;
      for (const Expr &argument : call->args) {
        if (argument->ty.as<PtrTypeNode>() ||
            argument->ty.as<TensorTypeNode>() ||
            argument.as<TensorRegionNode>()) {
          auto nested = Origins(argument);
          result.insert(result.end(), nested.begin(), nested.end());
        }
      }
      return result;
    }
    return {};
  }

  void Record(const Expr &tensor, const Object *owner, const Location &location,
              bool write, bool gemm = false) {
    for (const TensorVar &origin : Origins(tensor)) {
      events.push_back({origin, owner, location, write, gemm});
    }
  }

  Optional<VisitInterrupt> Visit_(const BindNode *node) final {
    Visit(node->value);
    if (node->var->ty.as<TensorTypeNode>() || node->var->ty.as<PtrTypeNode>()) {
      auto origins = Origins(node->value);
      if (!origins.empty())
        aliases_[node->var.get()] = std::move(origins);
    }
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const TensorStoreNode *node) final {
    auto previous = active_owner_;
    active_owner_ = node;
    for (const PrimExpr &index : node->indices)
      Visit(index);
    Visit(node->value);
    active_owner_ = previous;
    Record(node->dest, node, node->loc, true);
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const TensorLoadNode *node) final {
    Record(node->source, active_owner_ ? active_owner_ : node, node->loc,
           false);
    for (const PrimExpr &index : node->indices)
      Visit(index);
    return std::nullopt;
  }

  void AddressIndices(const Expr &value) {
    if (const auto *load = value.as<TensorLoadNode>()) {
      for (const PrimExpr &index : load->indices)
        Visit(index);
    } else if (const auto *region = value.as<TensorRegionNode>()) {
      for (const Range &range : region->region) {
        Visit(range->begin);
        Visit(range->extent);
      }
    }
  }

  Optional<VisitInterrupt> Visit_(const CallNode *node) final {
    if (node->op.same_as(Op::Get("tl.layout_hint")) ||
        node->op.same_as(Op::Get("tl.safe_value_hint")))
      return std::nullopt;
    if (node->op.same_as(access_ptr()) || node->op.same_as(address_of_op()) ||
        node->op.same_as(tensor_data_ptr_op())) {
      if (!node->args.empty())
        AddressIndices(node->args[0]);
      for (size_t index = 1; index < node->args.size(); ++index)
        Visit(node->args[index]);
      return std::nullopt;
    }
    TileOperator operation;
    try {
      operation = ParseOperator(GetRef<Call>(node));
    } catch (const std::exception &) {
    }
    if (operation.defined()) {
      auto access = operation->GetAccessRegions();
      const auto *gemm = operation.as<GemmNode>();
      bool clears_at_entry = ClearsAtLoopEntry(gemm);
      for (const TensorRegion &region : access.reads) {
        if (!clears_at_entry || !region->source.same_as(gemm->c_))
          Record(region->source, node, node->loc, false, gemm != nullptr);
      }
      for (const Expr &argument : node->args) {
        if (!argument.as<TensorRegionNode>() && !argument.as<TensorVar>()) {
          const auto *descriptor = argument.as<CallNode>();
          if (!descriptor || !descriptor->op.same_as(region()))
            Visit(argument);
        }
      }
      for (const TensorRegion &region : access.writes)
        Record(region->source, node, node->loc, true);
      return std::nullopt;
    }
    if (node->op.same_as(reducer_update())) {
      auto update = ParseReducerUpdate(node);
      Record(update.reducer, node, node->loc, false);
      for (const PrimExpr &index : update.indices)
        Visit(index);
      Visit(update.value);
      Record(update.reducer, node, node->loc, true);
      return std::nullopt;
    }
    bool may_write = true;
    if (auto operation = node->op.as<Op>()) {
      static auto effects = Op::GetAttrMap(tvm::op_attr::kCallEffectKind);
      if (effects.count(*operation))
        may_write = effects[*operation] >= CallEffectKind::kUpdateState;
    }
    for (const Expr &argument : node->args) {
      Visit(argument);
      bool writable = may_write;
      if (const auto *pointer = argument.as<CallNode>()) {
        if (pointer->op.same_as(access_ptr()) && pointer->args.size() == 3) {
          writable = writable &&
                     (GetConstInt(pointer->args[2]).value_or(3) & kAccessWrite);
        } else if (pointer->op.same_as(address_of_op()) && pointer->attrs) {
          auto mask =
              pointer->attrs.value().as_or_throw<DictAttrs>().GetAttr<PrimExpr>(
                  attr::kAccessMask);
          if (mask)
            writable =
                writable && (GetConstInt(*mask).value_or(3) & kAccessWrite);
        }
      }
      if (writable && (argument->ty.as<PtrTypeNode>() ||
                       argument->ty.as<TensorTypeNode>() ||
                       argument.as<TensorRegionNode>()))
        Record(argument, node, node->loc, true);
    }
    return std::nullopt;
  }

  std::unordered_map<const VarNode *, std::vector<TensorVar>> aliases_;
  const Object *active_owner_ = nullptr;
  std::vector<const ForNode *> loops_;
  int guard_depth_ = 0;
};

void VerifyInitialization(const tirx::Function &function) {
  if (!function->body)
    return;
  auto collector = ffi::make_object<InitializationAccessCollector>();
  collector->Visit(*function->body);
  std::unordered_map<const VarNode *, std::unordered_set<const Object *>>
      writers;
  for (const auto &event : collector->events) {
    if (event.write)
      writers[event.tensor.get()].insert(event.owner);
  }
  std::unordered_set<const VarNode *> written;
  std::unordered_set<const VarNode *> reported;
  for (const Var &parameter : function->params)
    written.insert(parameter.get());
  std::ostringstream report;
  for (const auto &event : collector->events) {
    if (event.write) {
      written.insert(event.tensor.get());
      continue;
    }
    std::string scope = event.tensor.scope().name();
    if (scope == "global" || scope == "shared.barrier" ||
        scope == "shared.cluster_barrier")
      continue;
    bool initialized = written.count(event.tensor.get());
    if (scope.starts_with("shared")) {
      initialized = false;
      for (const Object *writer : writers[event.tensor.get()]) {
        if (writer != event.owner)
          initialized = true;
      }
      for (const Var &parameter : function->params) {
        if (parameter.same_as(event.tensor))
          initialized = true;
      }
    }
    if (!initialized && reported.insert(event.tensor.get()).second) {
      report << "  `" << event.tensor->name
             << "` is read before anything writes it"
             << SpanHintSuffix({event.location, event.tensor->loc}) << "\n";
      if (event.gemm)
        report << "  GEMM accumulates into C; initialize it or use "
                  "clear_accum=True.\n";
    }
  }
  if (!reported.empty()) {
    LOG(WARNING)
        << "Buffer read before initialization: " << reported.size()
        << " buffer(s)\n"
        << report.str()
        << "Initialize storage with T.clear, T.fill, or T.copy. "
        << "Disable this diagnostic with tl.disable_buffer_init_check.";
  }
}
} // namespace

tvm::transform::Pass VerifyBufferInit() {
  return tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext context) -> Optional<tirx::Function> {
        if (!context->GetConfig<bool>(kDisableBufferInitCheck).value_or(false))
          VerifyInitialization(function);
        return function;
      },
      0, "tl.VerifyBufferInit");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.VerifyBufferInit",
                                   VerifyBufferInit);
}
} // namespace tvm::tl
