/*!
 * \file cuda/transform/ws_native.h
 * \brief Native kernel, metadata and definition boundaries used by WS passes.
 */
#pragma once

#include "ir/annotations.h"
#include "op/native_utils.h"
#include "transform/common/native_layout_utils.h"
#include "transform/common/pipeline_access.h"
#include "transform/common/pipeline_tensor.h"
#include "transform/common/warp_specialize.h"
#include <functional>
#include <tvm/tirx/transform.h>
#include <utility>
#include <vector>

namespace tvm::tl {
using namespace tirx;
using namespace prim;
using namespace ffi;

// Buffer slots and barrier parity only observe the iteration number modulo
// 2 * stages. Widen before subtracting loop origins, including narrow signed
// loops, then keep each mixed-radix step bounded. Unsigned 64-bit subtraction
// represents every nonnegative distance between two same-type loop endpoints;
// the reduced multiply/add cannot overflow for positive int32 stage counts.
inline PrimExpr
WSPipelineIterationModulo(const std::vector<const ForObj *> &loops,
                          int stages) {
  TVM_FFI_CHECK_GT(stages, 0, ValueError);
  PrimType type = PrimType::UInt(64);
  PrimExpr period = IntImm(type, int64_t{2} * stages);
  PrimExpr linear = IntImm(type, 0);
  for (const auto *loop : loops) {
    TVM_FFI_CHECK(IsOne(loop->step), ValueError)
        << "ProducerConsumerWarpSpecialized requires unit-step loop bounds";
    PrimExpr extent = cast(loop->end, type) - cast(loop->begin, type);
    PrimExpr index = cast(loop->var, type) - cast(loop->begin, type);
    linear = floormod(
        linear * floormod(extent, period) + floormod(index, period), period);
  }
  return linear;
}

// Analysis-only facts. All IR state remains in native Bind/Call/Region
// operands; this is not an alternative Buffer/Block object model or registry.
struct NativeWSKernel {
  SeqStmt body{Array<Stmt>{}};
  PrimExpr threads{ffi::UnsafeInit{}};
  Array<TensorVar> allocations;
  Map<TensorVar, Bind> definitions;
  Array<Bind> views;
  Map<String, Any> annotations;
  std::vector<RegionStmt> wrappers;
  const PipelineMemoryAnalysis *memory{nullptr};
  Location loc{UnknownLoc()};
};

template <typename T>
inline T WSSubstitute(const T &object, const Map<Var, Expr> &values) {
  auto mutator = make_object<StmtExprMutator>();
  for (const auto &[var, value] : values)
    mutator->VarRemapSet(var, value);
  return mutator->Mutate(object).ValueOrUnchanged(object);
}

inline TensorVar WSLocalTensor(Array<PrimExpr> shape, PrimType dtype,
                               String name, const std::string &scope) {
  return TensorVar(name,
                   TensorType(shape, dtype, runtime::MemoryScope::Get(scope)));
}

inline SeqStmt WSSequence(const Array<Stmt> &statements,
                          Location loc = UnknownLoc()) {
  Array<Stmt> flat;
  std::function<void(const Stmt &)> append = [&](const Stmt &statement) {
    if (const auto *seq = statement.as<SeqStmtObj>())
      for (const auto &part : seq->seq)
        append(part);
    else
      flat.push_back(statement);
  };
  for (const auto &statement : statements)
    append(statement);
  return SeqStmt(flat, loc);
}

inline bool IsWSAllocation(const BindObj *bind) {
  const auto *call = bind->value.as<CallObj>();
  return bind->var.as<TensorVar>() && call &&
         call->op.same_as(alloc_tensor_op());
}

inline Optional<Map<String, Any>> WSAttribute(const RegionStmtObj *region) {
  if (region->op->name != "tl.attr")
    return std::nullopt;
  return ReadNativeAnnotations(region);
}

// Scalar Bind defines a variable for the remainder of its SeqStmt. Wrapping
// it in a result-less region would hide that definition from later users.
// A scheduling marker therefore identifies the following statement without
// adding an IR scope. It is erased by MaterializeWSSchedule.
inline constexpr const char *kWSNextOpIdKey = "tl.ws_next_op_id";

inline Optional<String> WSNextOpId(const Stmt &statement) {
  const auto *evaluate = statement.as<EvaluateObj>();
  const auto *call = evaluate ? evaluate->value.as<CallObj>() : nullptr;
  const auto *op = call ? call->op.as<OpObj>() : nullptr;
  if (!op || op->name != "tl.annotation")
    return std::nullopt;
  auto annotations = ReadNativeAnnotations(call);
  auto id = annotations.Get(kWSNextOpIdKey);
  return id ? Optional<String>(id->cast<String>()) : std::nullopt;
}

inline Stmt WSMarkNext(String id, Location loc) {
  return MakeNativeAnnotation({{kWSNextOpIdKey, std::move(id)}},
                              std::move(loc));
}

inline Stmt WSOnlyStatement(const SeqStmt &body) {
  TVM_FFI_CHECK_EQ(body.size(), 1U, ValueError)
      << "Expected one statement in a warp-specialization wrapper";
  return body[0];
}

inline void MergeWSAnnotations(Map<String, Any> *into,
                               const Map<String, Any> &from) {
  for (const auto &[key, value] : from)
    into->Set(key, value);
}

inline NativeWSKernel ReadNativeWSKernel(SeqStmt body,
                                         const PipelineMemoryAnalysis *memory) {
  NativeWSKernel kernel;
  kernel.loc = body->loc;
  kernel.memory = memory;
  while (body.size() == 1) {
    const auto *region = body[0].as<RegionStmtObj>();
    if (!region)
      break;
    bool peel = false;
    if (region->op.same_as(launch_thread_op())) {
      auto tag = region->args[0].as_or_throw<StringImm>()->value;
      peel = tag == "threadIdx.y" || tag == "threadIdx.z";
    } else if (auto data = WSAttribute(region)) {
      peel = data.value().at("key").cast<String>() == "tl.kernel";
      if (peel)
        MergeWSAnnotations(&kernel.annotations,
                           data.value().at("value").cast<Map<String, Any>>());
    }
    if (!peel)
      break;
    kernel.wrappers.push_back(GetRef<RegionStmt>(region));
    body = region->body;
  }
  // Kernel attributes may nest after allocations (for example swizzle and
  // minimum-block annotations). Read schedule/layout metadata through those
  // transparent scopes, keeping the wrappers for the materializer to rebuild.
  // A named operation or a defining region is not a metadata boundary.
  std::function<SeqStmt(const SeqStmt &)> extract = [&](const SeqStmt &body) {
    Array<Stmt> executable;
    for (const Stmt &statement : body->seq) {
      if (const auto *bind = statement.as<BindObj>();
          bind && IsWSAllocation(bind)) {
        auto tensor = bind->var.as_or_throw<TensorVar>();
        kernel.allocations.push_back(tensor);
        kernel.definitions.Set(tensor, GetRef<Bind>(bind));
        continue;
      }
      if (const auto *bind = statement.as<BindObj>()) {
        const auto *call = bind->value.as<CallObj>();
        auto root = memory->Root(bind->var);
        if (call && call->op.same_as(decl_tensor_op()) && root &&
            root.value().as<TensorVar>() &&
            kernel.definitions.count(root.value().as_or_throw<TensorVar>()) &&
            SideEffect(bind->value) <= CallEffectKind::kPure) {
          // Views are explicit native definitions but correspond to no runtime
          // operation in the source DSL. Preserve their one evaluation before
          // role partitioning and shift the captured pointer at each access.
          kernel.views.push_back(GetRef<Bind>(bind));
          continue;
        }
      }
      if (const auto *evaluate = statement.as<EvaluateObj>();
          evaluate && !WSNextOpId(statement)) {
        if (const auto *call = evaluate->value.as<CallObj>();
            call && call->op.same_as(Op::Get("tl.annotation"))) {
          MergeWSAnnotations(&kernel.annotations, ReadNativeAnnotations(call));
          continue;
        }
      }
      if (const auto *region = statement.as<RegionStmtObj>()) {
        auto attribute = WSAttribute(region);
        if (attribute && region->body_params.empty() &&
            region->result_vars.empty() &&
            attribute.value().at("key").cast<String>() != kWSOpIdKey) {
          executable.push_back(RegionStmt(
              region->op, region->args, region->body_params, region->attrs,
              extract(region->body), region->result_vars, region->loc));
          continue;
        }
      }
      executable.push_back(statement);
    }
    return SeqStmt(executable, body->loc);
  };
  kernel.body = extract(body);
  return kernel;
}

inline Bind WSAllocate(const TensorVar &tensor,
                       Optional<Attrs> attrs = std::nullopt) {
  return Bind(tensor,
              Call(tensor.ty(), alloc_tensor_op(), {}, attrs, {tensor.ty()},
                   tensor->loc),
              tensor->loc);
}

inline SeqStmt
RebuildNativeWSKernel(const NativeWSKernel &kernel, Stmt body,
                      const Map<TensorVar, TensorVar> &replacements = {},
                      const Array<TensorVar> &extra_allocations = {}) {
  Array<Stmt> statements;
  for (const TensorVar &old : kernel.allocations) {
    auto replacement = replacements.Get(old);
    if (replacement) {
      const auto *call = kernel.definitions.at(old)->value.as<CallObj>();
      statements.push_back(WSAllocate(*replacement, call->attrs));
    } else
      statements.push_back(kernel.definitions.at(old));
  }
  for (const auto &tensor : extra_allocations)
    statements.push_back(WSAllocate(tensor));
  Map<Var, Expr> native_replacements;
  for (const auto &[old, tensor] : replacements)
    native_replacements.Set(old, tensor);
  for (const auto &view : kernel.views) {
    Expr value = WSSubstitute(view->value, native_replacements);
    statements.push_back(Bind(view->var, value, view->loc));
  }
  if (!kernel.annotations.empty())
    statements.push_back(MakeNativeAnnotation(kernel.annotations, kernel.loc));
  statements.push_back(body);
  SeqStmt result = WSSequence(statements, kernel.loc);
  for (auto it = kernel.wrappers.rbegin(); it != kernel.wrappers.rend(); ++it) {
    auto region = *it;
    result = SeqStmt(RegionStmt(region->op, region->args, region->body_params,
                                region->attrs, result, region->result_vars,
                                region->loc));
  }
  return result;
}

struct NativeWSResult {
  SeqStmt body;
  Optional<int> threads;
};

using NativeWSRewrite =
    std::function<Optional<NativeWSResult>(NativeWSKernel, PrimVar, Target)>;

// MaterializeKernelLaunch has already established native thread definitions.
// Rewrite one x launch at a time; y/z definitions and kernel metadata stay in
// their lexical regions. A declined automatic schedule returns its exact input.
inline tirx::Function RewriteNativeWSKernels(tirx::Function function,
                                             const NativeWSRewrite &rewrite) {
  if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return function;
  class Rewriter : public StmtExprMutator {
  public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;
    Rewriter(tirx::Function function, NativeWSRewrite rewrite)
        : target_(function->GetAttr<Target>(tvm::attr::kTarget).value()),
          memory_(function), rewrite_(std::move(rewrite)) {}
    UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
      if (!op->op.same_as(launch_thread_op()) ||
          op->args[0].as_or_throw<StringImm>()->value != "threadIdx.x")
        return StmtExprMutator::Mutate_(op, mode);
      auto kernel = ReadNativeWSKernel(op->body, &memory_);
      kernel.threads = op->args[1].as_or_throw<PrimExpr>();
      auto result =
          rewrite_(std::move(kernel), op->body_params[0].as_or_throw<PrimVar>(),
                   target_);
      if (!result)
        return ffi::Unchanged();
      auto args = op->args;
      if (result.value().threads)
        args.Set(1, IntImm(op->args[1]->ty.as_or_throw<PrimType>(),
                           result.value().threads.value()));
      return RegionStmt(op->op, args, op->body_params, op->attrs,
                        result.value().body, op->result_vars, op->loc);
    }

  private:
    Target target_;
    PipelineMemoryAnalysis memory_;
    NativeWSRewrite rewrite_;
  };
  TVM_FFI_CHECK(function->GetAttr<Target>(tvm::attr::kTarget), ValueError)
      << "Warp specialization requires a bound target";
  auto mutator = make_object<Rewriter>(function, rewrite);
  auto body = function->body;
  auto result = mutator->Mutate(body).ValueOrUnchanged(body);
  if (!result.same_as(body))
    function.CopyOnWrite()->body = result;
  return function;
}

} // namespace tvm::tl
