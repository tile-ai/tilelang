/*!
 * \file tl/cpu/transform/lower_cpu_any_all.cc
 * \brief Lower logical any_of/all_of intrinsics to serial CPU loops.
 */

#include "op/builtin.h"
#include "support/check.h"
#include "transform/common/access_ptr_utils.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

enum class LogicalKind { kAny, kAll };

std::optional<LogicalKind> MatchLogicalOp(const ObjectRef &op) {
  if (op.same_as(Op::Get("tl.any_of"))) {
    return LogicalKind::kAny;
  }
  if (op.same_as(Op::Get("tl.all_of"))) {
    return LogicalKind::kAll;
  }
  return std::nullopt;
}

PrimExpr MakeTruthValue(const PrimExpr &value) {
  if (value->dtype == DataType::Bool()) {
    return value;
  }
  PrimExpr nonzero = value != make_zero(value->dtype);
  if (value->dtype.is_float() || value->dtype.is_bfloat16()) {
    return Or(nonzero, Call(DataType::Bool(), builtin::isnan(), {value}));
  }
  return nonzero;
}

bool IsDenseBuffer(const Buffer &buffer) {
  if (buffer->strides.empty()) {
    return true;
  }
  if (buffer->strides.size() != buffer->shape.size()) {
    return false;
  }
  for (size_t i = 0; i < buffer->strides.size(); ++i) {
    PrimExpr expected = make_const(buffer->strides[i]->dtype, 1);
    for (size_t j = i + 1; j < buffer->shape.size(); ++j) {
      expected = expected * buffer->shape[j];
    }
    if (!StructuralEqual()(buffer->strides[i], expected)) {
      return false;
    }
  }
  return true;
}

class CPUAnyAllRewriter : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc func) {
    if (!func.defined() || !func->body.defined()) {
      return func;
    }
    CPUAnyAllRewriter rewriter;
    func.CopyOnWrite()->body = rewriter(std::move(func->body));
    return func;
  }

  Stmt VisitStmt(const Stmt &stmt) final {
    std::vector<Stmt> outer_prefix = std::move(pending_prefix_);
    pending_prefix_.clear();

    Stmt result = StmtExprMutator::VisitStmt(stmt);
    if (!pending_prefix_.empty()) {
      pending_prefix_.push_back(result);
      result = SeqStmt::Flatten(pending_prefix_);
    }

    pending_prefix_ = std::move(outer_prefix);
    return result;
  }

  Stmt VisitStmt_(const WhileNode *op) final {
    in_while_condition_ = true;
    PrimExpr condition = VisitExpr(op->condition);
    in_while_condition_ = false;

    std::vector<Stmt> condition_prefix = std::move(pending_condition_prefix_);
    pending_condition_prefix_.clear();
    Stmt body = VisitStmt(op->body);
    if (condition_prefix.empty()) {
      return While(condition, body, op->span);
    }

    condition_prefix.push_back(IfThenElse(
        Not(condition),
        Evaluate(Call(DataType::Void(), builtin::break_loop(), {}))));
    condition_prefix.push_back(body);
    return While(Bool(true), SeqStmt::Flatten(condition_prefix), op->span);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    std::optional<LogicalKind> kind = MatchLogicalOp(op->op);
    if (!kind) {
      return StmtExprMutator::VisitExpr_(op);
    }

    ICHECK_EQ(op->args.size(), 2U)
        << "CPU any_of/all_of expects (access_ptr, extent)";
    const auto *access = op->args[0].as<CallNode>();
    ICHECK(access != nullptr && access->op.same_as(tl::access_ptr()))
        << "CPU any_of/all_of expects a tl.access_ptr source, got "
        << op->args[0];
    ICHECK_EQ(access->args.size(), 3U)
        << "tl.access_ptr expects (BufferLoad, extent, rw_mask)";

    auto visit_expr = [this](const PrimExpr &expr) {
      return this->VisitExpr(expr);
    };
    BufferLoad base_load =
        detail::VisitAccessPtrBase(access->args[0], visit_expr);
    Buffer buffer = base_load->buffer;
    int ndim = static_cast<int>(buffer->shape.size());
    ICHECK_GT(ndim, 0) << "CPU any_of/all_of requires a non-scalar buffer";
    ICHECK_EQ(ndim, static_cast<int>(base_load->indices.size()))
        << "CPU any_of/all_of source indices must match buffer rank";
    ICHECK(IsDenseBuffer(buffer))
        << "CPU any_of/all_of currently requires a densely packed buffer, "
           "got explicit non-dense strides "
        << buffer->strides;
    ICHECK(buffer->axis_separators.empty())
        << "CPU any_of/all_of does not support axis separators";
    ICHECK_EQ(buffer->dtype.lanes(), 1)
        << "CPU any_of/all_of requires scalar elements";

    PrimExpr elems = VisitExpr(op->args[1]);
    ICHECK(elems->dtype.is_int() || elems->dtype.is_uint())
        << "CPU any_of/all_of extent must be an integer, got " << elems->dtype;

    bool is_any = kind.value() == LogicalKind::kAny;
    if (is_zero(elems)) {
      return make_const(DataType::Bool(), !is_any);
    }

    int result_id = result_counter_++;
    Buffer result =
        decl_buffer({1}, DataType::Bool(),
                    "logical_result_" + std::to_string(result_id), "local");
    Buffer index =
        decl_buffer({1}, elems->dtype,
                    "logical_index_" + std::to_string(result_id), "local");
    pending_prefix_.push_back(AllocBuffer(result));
    pending_prefix_.push_back(
        BufferStore(result, make_const(DataType::Bool(), !is_any), {0}));
    pending_prefix_.push_back(AllocBuffer(index));
    pending_prefix_.push_back(BufferStore(index, make_zero(elems->dtype), {0}));

    Array<PrimExpr> indices = base_load->indices;
    for (int d = ndim - 1; d >= 0; --d) {
      PrimExpr axis_index = BufferLoad(index, {0});
      for (int inner = d + 1; inner < ndim; ++inner) {
        axis_index = FloorDiv(axis_index, buffer->shape[inner]);
      }
      axis_index = FloorMod(axis_index, buffer->shape[d]);
      indices.Set(d, base_load->indices[d] + axis_index);
    }

    PrimExpr value = MakeTruthValue(BufferLoad(buffer, indices));
    PrimExpr stop_condition = is_any ? value : Not(value);
    PrimExpr continue_condition = BufferLoad(result, {0});
    if (is_any) {
      continue_condition = Not(continue_condition);
    }
    auto make_scan = [&]() {
      Stmt stop =
          BufferStore(result, make_const(DataType::Bool(), is_any), {0});
      Stmt advance = BufferStore(
          index, BufferLoad(index, {0}) + make_const(elems->dtype, 1), {0});
      return While(And(BufferLoad(index, {0}) < elems, continue_condition),
                   SeqStmt({IfThenElse(stop_condition, stop), advance}));
    };
    if (in_while_condition_) {
      pending_condition_prefix_.push_back(
          BufferStore(result, make_const(DataType::Bool(), !is_any), {0}));
      pending_condition_prefix_.push_back(
          BufferStore(index, make_zero(elems->dtype), {0}));
      pending_condition_prefix_.push_back(make_scan());
    } else {
      pending_prefix_.push_back(make_scan());
    }
    return BufferLoad(result, {0});
  }

private:
  int result_counter_ = 0;
  bool in_while_condition_ = false;
  std::vector<Stmt> pending_prefix_;
  std::vector<Stmt> pending_condition_prefix_;
};

} // namespace

namespace transform {

tvm::transform::Pass LowerCPUAnyAll() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    return CPUAnyAllRewriter::Rewrite(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0, "tl.LowerCPUAnyAll",
                                             {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.cpu.transform.LowerCPUAnyAll", LowerCPUAnyAll);
}

} // namespace transform

} // namespace tl
} // namespace tvm
