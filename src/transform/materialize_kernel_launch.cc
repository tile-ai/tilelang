/*!
 * \file materialize_kernel_launch.cc
 * \brief Lower typed TileLang kernel regions to CPU loops or device launches.
 */

#include "ir/analyzer.h"
#include "ir/annotations.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/target/target.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <utility>

namespace tvm::tl {
namespace {
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::Optional;
using ffi::String;
using namespace prim;

Array<PrimExpr> ReadLaunchTuple(const RegionStmtObj *region, size_t index) {
  return region->args[index].as_or_throw<Tuple>()->fields.Map(
      [](const Expr &value) { return value.as_or_throw<PrimExpr>(); });
}

class KernelLaunchMaterializer : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  KernelLaunchMaterializer(bool grid, bool threads,
                           Optional<Array<PrimExpr>> defaults,
                           Array<String> unsupported, Array<String> extra_tags,
                           String target)
      : grid_(grid), threads_(threads), defaults_(std::move(defaults)),
        unsupported_(std::move(unsupported)),
        extra_tags_(std::move(extra_tags)), target_(std::move(target)) {}

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    if (!grid_ && op->op.same_as(tirx::launch_thread_op())) {
      const String &tag = op->args[0].as_or_throw<StringImm>()->value;
      if (tag.find("blockIdx.") == 0 ||
          std::find(extra_tags_.begin(), extra_tags_.end(), tag) !=
              extra_tags_.end()) {
        // Explicit T.thread_binding grids use the native launch operation
        // directly, without a tl.launch_kernel wrapper.  Non-SIMT consumers
        // still need the same serial grid and the original coordinate identity.
        // An enclosing typed annotation region retains its lexical metadata.
        PrimVar var = op->body_params[0].as_or_throw<PrimVar>();
        auto body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
        return For(var, IntImm(var.ty(), 0),
                   cast(op->args[1].as_or_throw<PrimExpr>(), var.ty()),
                   IntImm(var.ty(), 1), ForKind::kDefault, body, {}, op->loc);
      }
    }
    if (op->op->name != "tl.launch_kernel")
      return StmtExprMutator::Mutate_(op, mode);
    TVM_FFI_CHECK_EQ(op->args.size(), 5, ValueError);
    auto begins = ReadLaunchTuple(op, 0);
    auto ends = ReadLaunchTuple(op, 1);
    auto steps = ReadLaunchTuple(op, 2);
    auto grid_tags =
        op->attrs.GetAttr<Array<String>>("grid_tags")
            .value_or(Array<String>{"blockIdx.x", "blockIdx.y", "blockIdx.z"});
    TVM_FFI_CHECK_GE(grid_tags.size(), ends.size(), ValueError);
    size_t handled_dims = 0;
    while (handled_dims < ends.size()) {
      const String &tag = grid_tags[handled_dims];
      if (tag.find("blockIdx.") != 0 &&
          std::find(extra_tags_.begin(), extra_tags_.end(), tag) ==
              extra_tags_.end()) {
        break;
      }
      ++handled_dims;
    }
    if (handled_dims < ends.size()) {
      // Backend-specific dimensions are opt-in. Retain the undeclared tail
      // as a typed launch region, including its original bounds, definitions,
      // metadata and thread operands. A subsequent pass with the corresponding
      // launch_dim_tags can consume it. In particular, do not reinterpret an
      // unknown device dimension as a serial CPU loop or drop its tag.
      if (handled_dims == 0)
        return ffi::Unchanged{};
      auto args = op->args;
      for (size_t k = 0; k < 3; ++k) {
        const auto &fields = op->args[k].as_or_throw<Tuple>()->fields;
        args.Set(
            k, Tuple(Array<Expr>(fields.begin() + handled_dims, fields.end())));
      }
      auto attrs = op->attrs->dict;
      attrs.Set("grid_tags", Array<String>(grid_tags.begin() + handled_dims,
                                           grid_tags.end()));
      attrs.Set("grid_dim_offset",
                op->attrs.GetAttr<int64_t>("grid_dim_offset").value_or(0) +
                    static_cast<int64_t>(handled_dims));
      SeqStmt body(RegionStmt(op->op, args,
                              Array<Var>(op->body_params.begin() + handled_dims,
                                         op->body_params.end()),
                              DictAttrs(attrs), op->body, op->result_vars,
                              op->loc));
      return RebuildGridNest(op, begins, ends, steps, grid_tags, handled_dims,
                             body);
    }
    auto hints = op->args[4].as_or_throw<Tuple>();
    auto schema = op->attrs.GetAttr<Array<Any>>("hint_schema")
                      .value_or(Array<Any>{"map"});
    auto metadata = DecodeAnnotations(hints, schema).cast<Map<Any, Any>>();
    for (const String &key : unsupported_) {
      TVM_FFI_CHECK(!metadata.count(key), ValueError)
          << "T.Kernel launch annotation `" << key
          << "` is not supported on target `" << target_ << "`";
    }
    SeqStmt body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
    if (!threads_) {
      // The old CPU/Ascend contract rejected uses, rather than pretending a
      // SIMT kernel executes correctly with each thread index set to zero.
      for (const Var &used : UndefinedVars(body)) {
        for (size_t i = ends.size(); i < op->body_params.size(); ++i) {
          TVM_FFI_CHECK(!used.same_as(op->body_params[i]), ValueError)
              << "T.Kernel body references thread index `" << used->name
              << "`, but target `" << target_ << "` has no SIMT threads";
        }
      }
    }
    // Keep the source annotation owner even when the launch has no hints.
    // Later consumers need it to distinguish whole-block sblock_attr from
    // ordinary sequence metadata. Payloads remain explicit native uses, and
    // LowerSourceAnnotations replaces the owner with an ordinary lexical scope.
    {
      Array<Any> scope_schema{
          "map", Array<Any>{Array<Any>{"str", "node"}, Array<Any>{"none"}},
          Array<Any>{Array<Any>{"str", "key"}, Array<Any>{"str", "tl.kernel"}},
          Array<Any>{Array<Any>{"str", "value"}, schema}};
      body =
          SeqStmt(RegionStmt(Op::Get("tl.attr"), {hints}, {},
                             DictAttrs({{"schema_version", 1},
                                        {"annotation_schema", scope_schema}}),
                             body, {}, op->loc));
    }
    if (threads_) {
      bool specified =
          op->attrs.GetAttr<bool>("threads_was_specified", false).value();
      auto extents = specified ? ReadLaunchTuple(op, 3)
                               : defaults_.value_or(Array<PrimExpr>{});
      TVM_FFI_CHECK(!extents.empty() && extents.size() <= 3, ValueError)
          << "T.Kernel needs threads= or a backend default thread-block size";
      while (extents.size() < 3)
        extents.push_back(IntImm(PrimType::Int(32), 1));
      auto tags = op->attrs.GetAttr<Array<String>>("thread_tags")
                      .value_or(Array<String>{"threadIdx.x", "threadIdx.y",
                                              "threadIdx.z"});
      for (int i = 2; i >= 0; --i) {
        PrimVar var = op->body_params[ends.size() + i].as_or_throw<PrimVar>();
        PrimExpr extent = cast(extents[i], var.ty());
        if (auto imm = extent.as<IntImmObj>()) {
          TVM_FFI_CHECK_GT(imm->value, 0, ValueError)
              << "Thread extent must be positive";
        }
        body = SeqStmt(RegionStmt(tirx::launch_thread_op(),
                                  {StringImm(tags[i]), extent}, {var},
                                  DictAttrs(), body, {}, op->loc));
      }
    }
    return RebuildGridNest(op, begins, ends, steps, grid_tags, ends.size(),
                           body);
  }

private:
  Stmt RebuildGridNest(const RegionStmtObj *op, const Array<PrimExpr> &begins,
                       const Array<PrimExpr> &ends,
                       const Array<PrimExpr> &steps, const Array<String> &tags,
                       size_t count, SeqStmt body) {
    for (size_t i = count; i-- > 0;) {
      auto var = op->body_params[i].as_or_throw<PrimVar>();
      if (grid_) {
        const String &tag = tags[i];
        // Frontend grids are zero-based, but an explicitly transformed range
        // keeps its original coordinate through a separate bound index.
        PrimExpr zero = IntImm(var.ty(), 0);
        auto analyzer = CreateTileLangAnalyzer();
        PrimExpr extent = ends[i];
        PrimVar physical = var;
        if (!IsZero(begins[i]) || !IsOne(steps[i])) {
          TVM_FFI_CHECK(analyzer->CanProve(steps[i] > zero), ValueError)
              << "Device Kernel grid step must be provably positive";
          extent = analyzer->Simplify(
              max(ceildiv(ends[i] - begins[i], steps[i]), zero));
          physical = PrimVar(var->name + "_launch", var.ty());
          body = SeqStmt(
              {Bind(var, begins[i] + physical * steps[i], op->loc), body},
              op->loc);
        }
        body = SeqStmt(RegionStmt(tirx::launch_thread_op(),
                                  {StringImm(tag), extent}, {physical},
                                  DictAttrs(), body, {}, op->loc));
      } else {
        body = SeqStmt(
            For(var, begins[i], ends[i], steps[i], ForKind::kDefault, body,
                {{"tl.cpu_grid_dim",
                  op->attrs.GetAttr<int64_t>("grid_dim_offset").value_or(0) +
                      static_cast<int64_t>(i)}},
                op->loc));
      }
    }
    return body;
  }

  bool grid_;
  bool threads_;
  Optional<Array<PrimExpr>> defaults_;
  Array<String> unsupported_;
  Array<String> extra_tags_;
  String target_;
};
} // namespace

tvm::transform::Pass
MaterializeKernelLaunch(bool lower_grid_binding, bool lower_thread_binding,
                        Optional<Array<PrimExpr>> default_threads,
                        Optional<Array<String>> unsupported_annotations,
                        Optional<Array<String>> launch_dim_tags) {
  auto pass = [=](tirx::Function function, IRModule,
                  tvm::transform::PassContext) {
    if (!(!function->GetAttr(tvm::attr::kDeclaration).value_or(false)))
      return function;
    String target_name = "<unbound>";
    if (auto target = function->GetAttr<Target>(tvm::attr::kTarget)) {
      target_name = target.value()->kind->name;
    }
    auto mutator = ffi::make_object<KernelLaunchMaterializer>(
        lower_grid_binding, lower_thread_binding, default_threads,
        unsupported_annotations.value_or(Array<String>{}),
        launch_dim_tags.value_or(Array<String>{}), target_name);
    auto body =
        mutator->Mutate(function->body).ValueOrUnchanged(function->body);
    if (!body.same_as(function->body))
      function.CopyOnWrite()->body = body;
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.MaterializeKernelLaunch");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.MaterializeKernelLaunch",
                                   MaterializeKernelLaunch);
}
} // namespace tvm::tl
