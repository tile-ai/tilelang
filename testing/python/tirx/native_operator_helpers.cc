#include "backend/common/op/atomic_reduce.h"
#include "backend/common/op/reduce_value.h"
#include "compat/analyzer.h"
#include "op/copy.h"
#include "op/fill.h"
#include "op/parallel.h"
#include "op/reducer.h"
#include "transform/common/loop_aware_visitor.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/common/memory_access.h"

namespace tvm::tl {

class ExitTestContext {
  friend class tvm::With<ExitTestContext>;
  ExitTestContext(ffi::Array<ffi::String> *events, int identifier, bool fail)
      : events_(events), identifier_(identifier), fail_(fail) {}
  void EnterWithScope() {}
  void ExitWithScope() {
    events_->push_back(std::to_string(identifier_));
    if (fail_)
      throw std::runtime_error("exit " + std::to_string(identifier_));
  }
  ffi::Array<ffi::String> *events_;
  int identifier_;
  bool fail_;
};

ffi::Array<ffi::String> ContextExitBehavior(bool unwind, bool nested) {
  ffi::Array<ffi::String> events;
  try {
    ContextStack<WithGroup<ExitTestContext>> scopes;
    scopes.Current().Emplace(&events, 1, true);
    auto body = [&] {
      scopes.Current().Emplace(&events, 2, true);
      scopes.Current().Emplace(&events, 3, false);
      if (unwind)
        throw std::runtime_error("body");
    };
    if (nested)
      scopes.WithNewScope(body);
    else
      body();
  } catch (const std::runtime_error &error) {
    events.push_back(error.what());
  }
  return events;
}

class LoopProofCollector : public LoopAwareVisitor {
public:
  ffi::Array<int64_t> proofs;

  ffi::Optional<ffi::VisitInterrupt> Visit_(const EvaluateNode *node) final {
    proofs.push_back(
        tl::arith::CanProve(analyzer_, node->value.as_or_throw<PrimExpr>()));
    return std::nullopt;
  }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.testing.ContextExitBehavior", ContextExitBehavior)
      .def("tl.testing.CollectLoopProofs",
           [](Stmt body) {
             auto visitor = ffi::make_object<LoopProofCollector>();
             visitor->Visit(body);
             return visitor->proofs;
           })
      .def("tl.testing.SummarizeMemoryAccess",
           [](Stmt body, Stmt statement) {
             auto aliases = ffi::make_object<MemoryAccessCollector>();
             aliases->Visit(body);
             auto collector =
                 ffi::make_object<MemoryAccessCollector>(aliases->aliases);
             collector->Visit(statement);
             ffi::Array<ffi::Map<ffi::String, ffi::Any>> result;
             for (const auto &access : collector->accesses) {
               result.push_back({{"storage", access.location.allocation},
                                 {"scope", access.location.scope},
                                 {"offset", access.location.offset},
                                 {"bytes", access.bytes},
                                 {"mask", access.mask},
                                 {"atomic", access.atomic},
                                 {"asynchronous", access.asynchronous},
                                 {"precise", access.location.precise}});
             }
             return result;
           })
      .def("tl.testing.LowerAtomicSIMT",
           [](TileOperator operation, Target target, Range threads,
              PrimExpr thread, LayoutMap layouts) {
             const auto *atomic =
                 dynamic_cast<const AtomicOpBaseNode *>(operation.get());
             ICHECK(atomic != nullptr);
             sym::Analyzer analyzer;
             if (auto variable = thread.as<PrimVar>()) {
               tl::arith::Bind(analyzer, variable.value(), threads);
             }
             return backend::AtomicReduce::Lower(
                 *atomic, {target, threads, thread, layouts, {}}, analyzer);
           })
      .def("tl.testing.FuseParallelLoops", ParallelLoopFuser::Fuse)
      .def("tl.testing.ReduceIdentity",
           [](ReduceOp operation, int width) {
             return backend::reduce::MakeInitValue(*operation.operator->(),
                                                   width);
           })
      .def("tl.testing.ReduceCombine",
           [](ReduceOp operation, int width, PrimExpr accumulator,
              PrimExpr value) {
             return backend::reduce::MakeReduce(*operation.operator->(), width,
                                                accumulator, value);
           })
      .def("tl.testing.ReduceCodegen",
           [](ReduceOp operation, int width) {
             return backend::reduce::MakeCodegenReducer(*operation.operator->(),
                                                        width)
                 .value_or("");
           })
      .def("tl.testing.LowerReduction",
           [](ReduceOp operation, Target target, Range threads, PrimExpr thread,
              LayoutMap layouts, ffi::Map<TensorVar, TensorVar> remap) {
             sym::Analyzer analyzer;
             if (auto variable = thread.as<PrimVar>()) {
               tl::arith::Bind(analyzer, variable.value(), threads);
             }
             ffi::Array<Stmt> statements;
             LowerArgs arguments{target, threads, thread, layouts, remap};
             arguments.add_workspace = [&](int size, PrimType type) -> Expr {
               TensorVar workspace(
                   "reduce_workspace_" + std::to_string(statements.size()),
                   TensorType({MakeConst(PrimType::Int(32), size)}, type,
                              runtime::MemoryScope("shared")));
               statements.push_back(
                   Bind(workspace, Call(workspace.ty(), tirx::alloc_tensor_op(),
                                        {}, std::nullopt, {workspace.ty()})));
               return workspace.data();
             };
             statements.push_back(operation->Lower(arguments, analyzer));
             return SeqStmt(statements);
           })
      .def("tl.testing.LowerNormalCopy",
           [](Copy operation, Target target, Range threads, PrimExpr thread,
              LayoutMap layouts) {
             sym::Analyzer analyzer;
             if (auto variable = thread.as<PrimVar>()) {
               tl::arith::Bind(analyzer, variable.value(), threads);
             }
             return LowerNormalCopy(
                 *operation.operator->(),
                 LowerArgs{target, threads, thread, layouts, {}}, analyzer);
           })
      .def("tl.testing.InferSIMTCopy",
           [](Copy operation, Target target, Range threads, LayoutMap layouts,
              int level) {
             return operation->InferSIMTLayout(
                 {target, threads, layouts, sym::Analyzer(), {}, {}},
                 static_cast<InferLevel>(level));
           })
      .def("tl.testing.MakeParallel", [](For loop) { return ParallelOp(loop); })
      .def("tl.testing.ParallelState",
           [](ParallelOp operation, PrimExpr thread) {
             return ffi::Array<ffi::Any>{
                 operation->GetLoopLayout(), operation->GetPredicate(thread),
                 operation->LoopLayoutRequiresPaddingGuard()};
           })
      .def("tl.testing.InferOperator",
           [](TileOperator operation, Target target, Range threads,
              LayoutMap layouts, ffi::Map<TensorVar, TensorVar> remap,
              LayoutMap strict_layouts, int level,
              ffi::Map<Var, Expr> bindings) {
             LayoutInferArgs arguments{target,          threads,       layouts,
                                       sym::Analyzer(), remap,         bindings,
                                       false,           strict_layouts};
             return operation->InferLayout(arguments,
                                           static_cast<InferLevel>(level));
           })
      .def("tl.testing.LowerOperator",
           [](TileOperator operation, Target target, Range threads,
              PrimExpr thread, LayoutMap layouts,
              ffi::Map<TensorVar, TensorVar> remap) {
             sym::Analyzer analyzer;
             if (auto variable = thread.as<PrimVar>()) {
               tl::arith::Bind(analyzer, variable.value(), threads);
             }
             LowerArgs arguments{target, threads, thread, layouts, remap};
             return operation->Lower(arguments, analyzer);
           })
      .def("tl.testing.GetOperatorAccessRegions",
           [](TileOperator operation) {
             AccessRegions access = operation->GetAccessRegions();
             return ffi::Array<ffi::Any>{access.reads, access.writes};
           })
      .def("tl.testing.CloneOperator",
           [](TileOperator operation) { return operation->Clone(); })
      .def("tl.testing.GetReadBeforeWriteRegions",
           [](TileOperator operation) {
             return operation->GetReadBeforeWriteRegions();
           })
      .def("tl.testing.AnalyzeReducerSite",
           [](Fragment layout, ffi::Array<PrimVar> variables,
              ffi::Array<PrimExpr> indices, PrimExpr value,
              ffi::Array<PrimExpr> shape, Range threads) {
             auto extent = GetConstInt(threads->extent);
             auto begin = GetConstInt(threads->begin);
             ICHECK(extent && begin);
             ReducerSiteAnalysis analysis = AnalyzeReducerUpdateSite(
                 {layout, variables, indices, value}, shape, *extent, *begin,
                 sym::Analyzer());
             ffi::Array<ffi::Array<int>> steps;
             for (const auto &[count, scale] : analysis.steps) {
               steps.push_back({count, scale});
             }
             return ffi::Map<ffi::String, ffi::Any>{
                 {"eligible", analysis.narrow_eligible},
                 {"reason", analysis.reason},
                 {"combine_size", analysis.combine_size},
                 {"steps", steps},
                 {"induced", analysis.induced}};
           })
      .def("tl.testing.ReducerIdentity",
           [](ffi::String operation, PrimType type) {
             return ReducerV2Identity(ParseReducerV2OpType(operation), type);
           })
      .def("tl.testing.ReducerCombine",
           [](ffi::String operation, PrimExpr left, PrimExpr right) {
             return ReducerV2Combine(ParseReducerV2OpType(operation), left,
                                     right);
           })
      .def("tl.testing.ValueIsReplicaSafe", ValueIsReplicaSafe);
}

} // namespace tvm::tl
