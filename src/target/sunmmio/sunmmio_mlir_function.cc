#include "sunmmio_mlir_function.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Verifier.h"
#include "npuir/Dialect/SUVM/IR/Attributes.h"
#include "npuir/Dialect/SUVM/IR/Dialect.h"
#include "npuir/Dialect/SUVM/IR/Ops.h"

namespace tvm {
namespace codegen {

SunmmioMlirFunction::SunmmioMlirFunction(SunmmioMlirContext &ctx)
    : ctx_(ctx), type_(ctx) {}

void SunmmioMlirFunction::BeginModule() {
  ctx_.mlir_ctx.getOrLoadDialect<mlir::suvm::SUVMDialect>();
  ctx_.mlir_ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx_.mlir_ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx_.mlir_ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx_.mlir_ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();

  ctx_.module = mlir::ModuleOp::create(ctx_.builder.getUnknownLoc());
  ctx_.module->getOperation()->setAttr(
      "suvm.device_arch", mlir::suvm::DeviceArchAttr::get(
                              &ctx_.mlir_ctx, mlir::suvm::DeviceArch::a4e));
  ctx_.builder.setInsertionPointToEnd(ctx_.module->getBody());
  current_func_ = mlir::func::FuncOp();
  ctx_.ClearFunctionState();
}

void SunmmioMlirFunction::EndModule() {
  if (failed(mlir::verify(*ctx_.module))) {
    LOG(FATAL) << "SunMMIO MLIR module verification failed";
  }
}

void SunmmioMlirFunction::BeginFunction(const std::string &name,
                                        const std::vector<BuilderArg> &args) {
  mlir::SmallVector<mlir::Type, 8> arg_types;
  arg_types.reserve(args.size());
  for (const BuilderArg &arg : args) {
    arg_types.push_back(type_.MapType(arg.type));
  }
  mlir::FunctionType func_type =
      ctx_.builder.getFunctionType(arg_types, mlir::TypeRange{});
  mlir::func::FuncOp func =
      mlir::func::FuncOp::create(type_.Loc(), name, func_type);
  ctx_.module->push_back(func);
  current_func_ = func;

  ctx_.ClearFunctionState();
  ctx_.PushMLIRValueScope();

  mlir::Block *entry = func.addEntryBlock();
  ctx_.builder.setInsertionPointToStart(entry);
  for (int i = 0, e = static_cast<int>(args.size()); i < e; ++i) {
    ctx_.BindMLIRValue(args[i].name, entry->getArgument(i));
  }
}

void SunmmioMlirFunction::EndFunction() {
  current_func_ = mlir::func::FuncOp();
  ctx_.ClearFunctionState();
  ctx_.builder.setInsertionPointToEnd(ctx_.module->getBody());
}

void SunmmioMlirFunction::EmitReturn() {
  mlir::Block *block = ctx_.builder.getInsertionBlock();
  if (block != nullptr && !block->empty() &&
      block->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    return;
  }
  if (ctx_.pending_sync_units != SunmmioMlirContext::kNoSyncUnits) {
    SunmmioMlirContext::SyncUnitMask units_mask = ctx_.pending_sync_units;
    mlir::suvm::SyncUnits units = SunmmioMlirContext::ToSyncUnits(units_mask);
    mlir::suvm::SyncOp::create(
        ctx_.builder, type_.MakeDebugLoc("function_exit_sync"),
        mlir::suvm::SyncUnitsAttr::get(&ctx_.mlir_ctx, units));
    ctx_.CompletePendingSyncUnits(units_mask);
  }
  mlir::func::ReturnOp::create(ctx_.builder, type_.Loc());
}

void SunmmioMlirFunction::BeginFor(
    const std::string &iv, const SunMMIOValue &lb, const SunMMIOValue &ub,
    const SunMMIOValue &step,
    const ffi::Map<ffi::String, ffi::Any> &annotations,
    const std::vector<SunMMIOValue> &live_out_values) {
  mlir::Value lb_v =
      type_.EnsureIndex(type_.ResolveValue(lb, ctx_.builder.getIndexType()));
  mlir::Value ub_v =
      type_.EnsureIndex(type_.ResolveValue(ub, ctx_.builder.getIndexType()));
  mlir::Value step_v =
      type_.EnsureIndex(type_.ResolveValue(step, ctx_.builder.getIndexType()));

  mlir::SmallVector<mlir::Value, 8> init_args;
  init_args.reserve(live_out_values.size());
  for (const SunMMIOValue &value : live_out_values) {
    mlir::Value init = ctx_.LookupMLIRValue(value.value);
    if (!init) {
      init = type_.ResolveValue(value, type_.MapType(value.type));
    }
    init_args.push_back(init);
  }

  mlir::scf::ForOp for_op = mlir::scf::ForOp::create(
      ctx_.builder, type_.Loc(), lb_v, ub_v, step_v, init_args);

  SunmmioMlirContext::ForFrame frame;
  frame.op = for_op;
  frame.annotations = annotations;
  frame.entry_pending_sync_units = ctx_.pending_sync_units;
  frame.live_out_value_names.reserve(live_out_values.size());
  for (const SunMMIOValue &value : live_out_values) {
    frame.live_out_value_names.push_back(value.value);
  }
  frame.iter_values.assign(for_op.getRegionIterArgs().begin(),
                           for_op.getRegionIterArgs().end());
  frame.produced_values.assign(frame.iter_values.size(), mlir::Value());
  ctx_.for_stack.push_back(std::move(frame));
  ctx_.control_flow_stack.push_back(SunmmioMlirContext::ControlNode{
      SunmmioMlirContext::ControlKind::kFor,
      static_cast<int>(ctx_.for_stack.size()) - 1});

  ctx_.PushMLIRValueScope();
  ctx_.BindMLIRValue(iv, for_op.getInductionVar());
  SunmmioMlirContext::ForFrame &active_frame = ctx_.for_stack.back();
  for (int i = 0, e = static_cast<int>(active_frame.iter_values.size()); i < e;
       ++i) {
    ctx_.BindMLIRValue(active_frame.live_out_value_names[i],
                       active_frame.iter_values[i]);
  }
  ctx_.builder.setInsertionPointToStart(for_op.getBody());
}

void SunmmioMlirFunction::EndFor() {
  ICHECK(!ctx_.for_stack.empty())
      << "EndFor called without a matching BeginFor";
  ICHECK(!ctx_.control_flow_stack.empty() &&
         ctx_.control_flow_stack.back().kind ==
             SunmmioMlirContext::ControlKind::kFor &&
         ctx_.control_flow_stack.back().index ==
             static_cast<int>(ctx_.for_stack.size()) - 1)
      << "EndFor control_flow_stack mismatch";
  ctx_.control_flow_stack.pop_back();

  SunmmioMlirContext::ForFrame frame = std::move(ctx_.for_stack.back());
  ctx_.for_stack.pop_back();
  SunmmioMlirContext::SyncUnitMask body_pending_sync_units =
      ctx_.pending_sync_units;
  mlir::SmallVector<mlir::Value, 8> yielded;
  yielded.reserve(frame.iter_values.size());
  for (int i = 0, e = static_cast<int>(frame.iter_values.size()); i < e; ++i) {
    yielded.push_back(frame.produced_values[i] ? frame.produced_values[i]
                                               : frame.iter_values[i]);
  }

  mlir::Block *body = frame.op.getBody();
  mlir::scf::YieldOp yield_op =
      body->mightHaveTerminator()
          ? mlir::dyn_cast<mlir::scf::YieldOp>(body->back())
          : mlir::scf::YieldOp();
  if (yield_op) {
    yield_op.getOperation()->setOperands(yielded);
  } else {
    ctx_.builder.setInsertionPointToEnd(body);
    mlir::scf::YieldOp::create(ctx_.builder, type_.Loc(), yielded);
  }

  ctx_.builder.setInsertionPointAfter(frame.op);
  ctx_.PopMLIRValueScope();
  ctx_.pending_sync_units = SunmmioMlirContext::MergeSyncUnits(
      frame.entry_pending_sync_units, body_pending_sync_units);
  for (int i = 0, e = static_cast<int>(frame.live_out_value_names.size());
       i < e; ++i) {
    ctx_.BindMLIRValue(frame.live_out_value_names[i], frame.op.getResult(i));
  }
}

void SunmmioMlirFunction::BeginWhile(
    const std::vector<SunMMIOValue> &live_out_values) {
  mlir::SmallVector<mlir::Value, 8> init_args;
  init_args.reserve(live_out_values.size());
  for (const SunMMIOValue &value : live_out_values) {
    mlir::Value init = ctx_.LookupMLIRValue(value.value);
    if (!init) {
      init = type_.ResolveValue(value, type_.MapType(value.type));
    }
    init_args.push_back(init);
  }

  mlir::SmallVector<mlir::Type, 8> result_types;
  for (mlir::Value init_arg : init_args) {
    result_types.push_back(init_arg.getType());
  }
  auto empty_builder = [](mlir::OpBuilder &, mlir::Location, mlir::ValueRange) {
  };
  mlir::scf::WhileOp while_op =
      mlir::scf::WhileOp::create(ctx_.builder, type_.Loc(), result_types,
                                 init_args, empty_builder, empty_builder);

  SunmmioMlirContext::WhileFrame frame;
  frame.op = while_op;
  frame.entry_pending_sync_units = ctx_.pending_sync_units;
  frame.live_out_value_names.reserve(live_out_values.size());
  for (const SunMMIOValue &value : live_out_values) {
    frame.live_out_value_names.push_back(value.value);
  }
  frame.before_values.assign(while_op.getBeforeArguments().begin(),
                             while_op.getBeforeArguments().end());
  frame.iter_values.assign(while_op.getAfterArguments().begin(),
                           while_op.getAfterArguments().end());
  frame.produced_values.assign(frame.iter_values.size(), mlir::Value());
  ctx_.while_stack.push_back(std::move(frame));
  ctx_.control_flow_stack.push_back(SunmmioMlirContext::ControlNode{
      SunmmioMlirContext::ControlKind::kWhile,
      static_cast<int>(ctx_.while_stack.size()) - 1});

  ctx_.PushMLIRValueScope();
  SunmmioMlirContext::WhileFrame &active_frame = ctx_.while_stack.back();
  for (int i = 0, e = static_cast<int>(active_frame.before_values.size());
       i < e; ++i) {
    ctx_.BindMLIRValue(active_frame.live_out_value_names[i],
                       active_frame.before_values[i]);
  }
  ctx_.builder.setInsertionPointToStart(while_op.getBeforeBody());
}

void SunmmioMlirFunction::BeginWhileBody(const SunMMIOValue &cond) {
  ICHECK(!ctx_.while_stack.empty())
      << "BeginWhileBody called without a matching BeginWhile";
  SunmmioMlirContext::WhileFrame &frame = ctx_.while_stack.back();
  frame.in_body = true;
  frame.condition_pending_sync_units = ctx_.pending_sync_units;
  mlir::Value cond_v =
      type_.EnsureI1(type_.ResolveValue(cond, ctx_.builder.getI1Type()));

  mlir::Block *before_body = frame.op.getBeforeBody();
  mlir::scf::ConditionOp condition_op =
      before_body->mightHaveTerminator()
          ? mlir::dyn_cast<mlir::scf::ConditionOp>(before_body->back())
          : mlir::scf::ConditionOp();
  if (condition_op) {
    mlir::SmallVector<mlir::Value, 8> operands;
    operands.push_back(cond_v);
    operands.append(frame.before_values.begin(), frame.before_values.end());
    condition_op.getOperation()->setOperands(operands);
  } else {
    ctx_.builder.setInsertionPointToEnd(before_body);
    mlir::scf::ConditionOp::create(ctx_.builder, type_.Loc(), cond_v,
                                   frame.before_values);
  }

  ctx_.PopMLIRValueScope();
  ctx_.PushMLIRValueScope();
  for (int i = 0, e = static_cast<int>(frame.iter_values.size()); i < e; ++i) {
    ctx_.BindMLIRValue(frame.live_out_value_names[i], frame.iter_values[i]);
  }
  ctx_.builder.setInsertionPointToStart(frame.op.getAfterBody());
}

void SunmmioMlirFunction::EndWhile() {
  ICHECK(!ctx_.while_stack.empty())
      << "EndWhile called without a matching BeginWhile";
  ICHECK(!ctx_.control_flow_stack.empty() &&
         ctx_.control_flow_stack.back().kind ==
             SunmmioMlirContext::ControlKind::kWhile &&
         ctx_.control_flow_stack.back().index ==
             static_cast<int>(ctx_.while_stack.size()) - 1)
      << "EndWhile control_flow_stack mismatch";
  ctx_.control_flow_stack.pop_back();

  SunmmioMlirContext::WhileFrame frame = std::move(ctx_.while_stack.back());
  ctx_.while_stack.pop_back();
  SunmmioMlirContext::SyncUnitMask body_pending_sync_units =
      ctx_.pending_sync_units;
  mlir::SmallVector<mlir::Value, 8> yielded;
  yielded.reserve(frame.iter_values.size());
  for (int i = 0, e = static_cast<int>(frame.iter_values.size()); i < e; ++i) {
    yielded.push_back(frame.produced_values[i] ? frame.produced_values[i]
                                               : frame.iter_values[i]);
  }

  mlir::Block *after_body = frame.op.getAfterBody();
  mlir::scf::YieldOp yield_op =
      after_body->mightHaveTerminator()
          ? mlir::dyn_cast<mlir::scf::YieldOp>(after_body->back())
          : mlir::scf::YieldOp();
  if (yield_op) {
    yield_op.getOperation()->setOperands(yielded);
  } else {
    ctx_.builder.setInsertionPointToEnd(after_body);
    mlir::scf::YieldOp::create(ctx_.builder, type_.Loc(), yielded);
  }

  ctx_.builder.setInsertionPointAfter(frame.op);
  ctx_.PopMLIRValueScope();
  ctx_.pending_sync_units = SunmmioMlirContext::MergeSyncUnits(
      frame.condition_pending_sync_units, body_pending_sync_units);
  for (int i = 0, e = static_cast<int>(frame.live_out_value_names.size());
       i < e; ++i) {
    ctx_.BindMLIRValue(frame.live_out_value_names[i], frame.op.getResult(i));
  }
}

void SunmmioMlirFunction::BeginIf(
    const SunMMIOValue &cond,
    const std::vector<SunMMIOValue> &live_out_values) {
  mlir::Value cond_v =
      type_.EnsureI1(type_.ResolveValue(cond, ctx_.builder.getI1Type()));
  mlir::SmallVector<mlir::Type, 8> result_types;
  mlir::SmallVector<mlir::Value, 8> base_values;
  for (const SunMMIOValue &value : live_out_values) {
    result_types.push_back(type_.MapType(value.type));
    mlir::Value base = ctx_.LookupMLIRValue(value.value);
    if (!base) {
      base = type_.ResolveValue(value, type_.MapType(value.type));
    }
    base_values.push_back(base);
  }

  mlir::scf::IfOp if_op =
      mlir::scf::IfOp::create(ctx_.builder, type_.Loc(), result_types, cond_v,
                              /*withElseRegion=*/true);
  SunmmioMlirContext::IfFrame frame;
  frame.op = if_op;
  frame.entry_pending_sync_units = ctx_.pending_sync_units;
  frame.live_out_value_names.reserve(live_out_values.size());
  for (const SunMMIOValue &value : live_out_values) {
    frame.live_out_value_names.push_back(value.value);
  }
  frame.base_values.assign(base_values.begin(), base_values.end());
  frame.produced_values = frame.base_values;
  ctx_.if_stack.push_back(std::move(frame));
  ctx_.control_flow_stack.push_back(SunmmioMlirContext::ControlNode{
      SunmmioMlirContext::ControlKind::kIf,
      static_cast<int>(ctx_.if_stack.size()) - 1});

  ctx_.PushMLIRValueScope();
  ctx_.builder.setInsertionPointToStart(&if_op.getThenRegion().front());
  SunmmioMlirContext::IfFrame &active_frame = ctx_.if_stack.back();
  for (int i = 0, e = static_cast<int>(active_frame.base_values.size()); i < e;
       ++i) {
    ctx_.BindMLIRValue(active_frame.live_out_value_names[i],
                       active_frame.base_values[i]);
  }
}

void SunmmioMlirFunction::BeginElse() {
  ICHECK(!ctx_.if_stack.empty())
      << "BeginElse called without a matching BeginIf";
  SunmmioMlirContext::IfFrame &frame = ctx_.if_stack.back();
  ICHECK(!frame.in_else) << "BeginElse called twice for the same scf.if";
  frame.then_yield_values = frame.produced_values;
  frame.then_pending_sync_units = ctx_.pending_sync_units;
  frame.in_else = true;
  ctx_.pending_sync_units = frame.entry_pending_sync_units;

  ctx_.PopMLIRValueScope();
  ctx_.PushMLIRValueScope();
  ctx_.builder.setInsertionPointToStart(&frame.op.getElseRegion().front());
  frame.produced_values = frame.base_values;
  for (int i = 0, e = static_cast<int>(frame.base_values.size()); i < e; ++i) {
    ctx_.BindMLIRValue(frame.live_out_value_names[i], frame.base_values[i]);
  }
}

void SunmmioMlirFunction::EndIf() {
  ICHECK(!ctx_.if_stack.empty()) << "EndIf called without a matching BeginIf";
  ICHECK(!ctx_.control_flow_stack.empty() &&
         ctx_.control_flow_stack.back().kind ==
             SunmmioMlirContext::ControlKind::kIf &&
         ctx_.control_flow_stack.back().index ==
             static_cast<int>(ctx_.if_stack.size()) - 1)
      << "EndIf control_flow_stack mismatch";
  ctx_.control_flow_stack.pop_back();

  SunmmioMlirContext::IfFrame frame = std::move(ctx_.if_stack.back());
  ctx_.if_stack.pop_back();
  SunmmioMlirContext::SyncUnitMask then_pending_sync_units =
      frame.in_else ? frame.then_pending_sync_units : ctx_.pending_sync_units;
  SunmmioMlirContext::SyncUnitMask else_pending_sync_units =
      frame.in_else ? ctx_.pending_sync_units : frame.entry_pending_sync_units;
  mlir::SmallVector<mlir::Value, 8> then_yield;
  mlir::SmallVector<mlir::Value, 8> else_yield;
  if (frame.in_else) {
    then_yield.append(frame.then_yield_values.begin(),
                      frame.then_yield_values.end());
    else_yield.append(frame.produced_values.begin(),
                      frame.produced_values.end());
  } else {
    then_yield.append(frame.produced_values.begin(),
                      frame.produced_values.end());
    else_yield.append(frame.base_values.begin(), frame.base_values.end());
  }

  auto set_yield = [&](mlir::Region &region,
                       const mlir::SmallVector<mlir::Value, 8> &values) {
    mlir::Block &body = region.front();
    mlir::scf::YieldOp yield_op =
        !body.empty() ? mlir::dyn_cast<mlir::scf::YieldOp>(body.back())
                      : mlir::scf::YieldOp();
    if (yield_op) {
      yield_op.getOperation()->setOperands(values);
    } else {
      ctx_.builder.setInsertionPointToEnd(&body);
      mlir::scf::YieldOp::create(ctx_.builder, type_.Loc(), values);
    }
  };
  set_yield(frame.op.getThenRegion(), then_yield);
  set_yield(frame.op.getElseRegion(), else_yield);

  ctx_.builder.setInsertionPointAfter(frame.op);
  ctx_.PopMLIRValueScope();
  ctx_.pending_sync_units = SunmmioMlirContext::MergeSyncUnits(
      then_pending_sync_units, else_pending_sync_units);
  for (int i = 0, e = static_cast<int>(frame.live_out_value_names.size());
       i < e; ++i) {
    ctx_.BindMLIRValue(frame.live_out_value_names[i], frame.op.getResult(i));
  }
}

void SunmmioMlirFunction::EmitAssert(const SunMMIOValue &cond,
                                     const std::string &msg_text) {
  mlir::Value cond_v =
      type_.EnsureI1(type_.ResolveValue(cond, ctx_.builder.getI1Type()));
  mlir::cf::AssertOp::create(ctx_.builder, type_.Loc(), cond_v,
                             ctx_.builder.getStringAttr(msg_text));
}

} // namespace codegen
} // namespace tvm
