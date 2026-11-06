/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file lower_tensor_spec.cc
 * \brief Materialize source tensor constraints before layout and ABI lowering.
 */

#include "../ir/analyzer.h"
#include "../ir/annotations.h"
#include "ir/stmt_functor.h"
#include "tensor_descriptor.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>
#include <vector>

namespace tvm::tl {
namespace {

using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;
using tirx::TensorVar;

using TensorSpecs = Map<TensorVar, Map<Any, Any>>;

// Local declarations carry their typed metadata after the Bind, so collect
// it before rewriting owned storage. Parameters remain unchanged for the ABI.
class TensorSpecCollector : public tirx::StmtExprVisitor {
public:
  TensorSpecs specs;
  std::unordered_set<const VarObj *> explicit_stride_parameters;
  Map<Var, Expr> definitions;
  Array<TensorVar> annotated_layouts;

  ffi::Optional<VisitInterrupt> Visit_(const BindObj *op) final {
    definitions.Set(op->var, op->value);
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const EvaluateObj *op) final {
    const auto *call = op->value.as<CallObj>();
    if (call && IsAnnotationCall(call)) {
      RecordFields(ffi::Any(ReadAnnotations(call)).cast<Map<Any, Any>>());
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj *op) final {
    if (op->op.same_as(Op::Get("tl.attr"))) {
      auto fields =
          DecodeAnnotations(
              op->args[0].as_or_throw<tvm::Tuple>(),
              op->attrs.GetAttr<Array<Any>>("annotation_schema").value())
              .cast<Map<Any, Any>>();
      if (fields.count(String("key")) &&
          fields.at(String("key")).cast<String>() == "tl.tensor_spec") {
        Record(fields.at(String("value")));
      } else if (fields.count(String("key")) &&
                 fields.at(String("key")).cast<String>() == "layout_map") {
        RecordLayouts(fields.at(String("value")));
      } else if (fields.count(String("key")) &&
                 fields.at(String("key")).cast<String>() == "tl.kernel") {
        RecordFields(fields.at(String("value")).cast<Map<Any, Any>>());
      }
    } else if (op->op.same_as(Op::Get("tl.launch_kernel"))) {
      auto schema = op->attrs.GetAttr<Array<Any>>("hint_schema");
      if (schema && op->args.size() >= 5) {
        RecordFields(DecodeAnnotations(op->args[4].as_or_throw<tvm::Tuple>(),
                                       schema.value())
                         .cast<Map<Any, Any>>());
      }
    }
    return StmtExprVisitor::Visit_(op);
  }

private:
  void RecordFields(const Map<Any, Any> &fields) {
    if (fields.count(String("tl.tensor_spec"))) {
      Record(fields.at(String("tl.tensor_spec")));
    }
    if (fields.count(String("layout_map"))) {
      RecordLayouts(fields.at(String("layout_map")));
    }
  }

  void RecordLayouts(const Any &value) {
    for (const auto &[key, layout] : value.cast<Map<Any, Any>>()) {
      if (auto tensor = key.as<TensorVar>())
        annotated_layouts.push_back(*tensor);
    }
  }

  void Record(const Any &value) {
    for (const auto &[key, entry] : value.cast<Map<Any, Any>>()) {
      auto tensor = key.cast<TensorVar>();
      auto spec = entry.cast<Map<Any, Any>>();
      if (spec.count(String("compact")) &&
          !spec.at(String("compact")).cast<bool>()) {
        explicit_stride_parameters.insert(tensor.get());
      }
      if (!spec.at(String("axis_separators")).cast<Array<Any>>().empty()) {
        specs.Set(tensor, spec);
      }
    }
  }
};

class TensorSpecLowerer : public tl::LayoutMapMutator {
public:
  explicit TensorSpecLowerer(const TensorSpecCollector &collector)
      : specs_(collector.specs), definitions_(collector.definitions) {
    for (const auto &tensor : collector.annotated_layouts) {
      annotated_storage_.insert(SourceDataIdentity(tensor).get());
    }
  }
  using LayoutMapMutator::Mutate;
  using LayoutMapMutator::Mutate_;

  ffi::UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    auto value = LayoutMapMutator::Mutate_(op, mode)
                     .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                     .as_or_throw<Bind>();
    const auto *call = value->value.as<CallObj>();
    auto source = op->var.as<TensorVar>();
    if (!source || !call || !call->op.same_as(tirx::alloc_tensor_op()) ||
        !specs_.count(source.value())) {
      return value;
    }
    const auto &spec = specs_.at(source.value());
    TensorVar tensor = value->var.as_or_throw<TensorVar>();
    if (HasAnnotatedLayout(source.value())) {
      // A TileLang layout explicitly replaces source strides/separators in
      // the original lowering. Keep one logical allocation for its inference
      // and physical remapping, rather than introducing a larger alias first.
      TensorVar logical =
          LogicalTensor(tensor, CompactStrides(tensor), spec, tensor->loc);
      VarRemapSet(source.value(), logical);
      materialized_.insert(source.value().get());
      materialized_.insert(logical.get());
      return Bind(logical,
                  Call(logical.ty(), tirx::alloc_tensor_op(), call->args,
                       call->attrs, {logical.ty()}, call->loc),
                  value->loc);
    }
    auto geometry = SegmentedGeometry(tensor, spec);
    auto physical_type =
        ffi::make_object<tirx::TensorTypeObj>(*tensor.ty().get());
    physical_type->shape = {geometry.capacity};
    physical_type->layout = std::nullopt;
    TensorVar storage(tensor->name + "_storage",
                      tirx::TensorType(physical_type), tensor->loc);
    TensorVar logical =
        LogicalTensor(tensor, geometry.strides, spec, tensor->loc);
    // Allocation and address contract must change together. An alias over an
    // unchanged logical-sized allocation would under-allocate strided segments.
    Bind allocation(storage,
                    Call(storage.ty(), tirx::alloc_tensor_op(), call->args,
                         call->attrs, {storage.ty()}, call->loc),
                    value->loc);
    Bind view(logical,
              Call(logical.ty(), tirx::decl_tensor_op(), {storage.data()}, {},
                   {logical.ty()}, call->loc),
              value->loc);
    VarRemapSet(source.value(), logical);
    materialized_.insert(source.value().get());
    materialized_.insert(logical.get());
    return SeqStmt({allocation, view}, value->loc);
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *op,
                                     InplaceMode mode) final {
    auto value = LayoutMapMutator::Mutate_(op, mode)
                     .ValueOrUnchanged(ffi::GetRef<TensorLoad>(op))
                     .as_or_throw<TensorLoad>();
    if (auto tensor = value->source.as<TensorVar>();
        tensor && broadcasts_.count(tensor.value().get())) {
      return TensorLoad(value->source,
                        BroadcastIndices(tensor.value(), value->indices),
                        value->loc);
    }
    return value;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreObj *op,
                                 InplaceMode mode) final {
    auto value = LayoutMapMutator::Mutate_(op, mode)
                     .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                     .as_or_throw<TensorStore>();
    if (auto tensor = value->dest.as<TensorVar>();
        tensor && broadcasts_.count(tensor.value().get())) {
      return TensorStore(value->dest,
                         BroadcastIndices(tensor.value(), value->indices),
                         value->value, value->loc);
    }
    return value;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    if (!op->op.same_as(Op::Get("tl.attr"))) {
      return LayoutMapMutator::Mutate_(op, mode);
    }
    auto fields =
        DecodeAnnotations(
            op->args[0].as_or_throw<tvm::Tuple>(),
            op->attrs.GetAttr<Array<Any>>("annotation_schema").value())
            .cast<Map<Any, Any>>();
    if (!fields.count(String("key")) ||
        fields.at(String("key")).cast<String>() != "tl.tensor_spec") {
      return LayoutMapMutator::Mutate_(op, mode);
    }
    Array<Stmt> prefix = LowerSpecs(fields.at(String("value")), op->loc);
    prefix.push_back(Mutate(op->body, mode).ValueOrUnchanged(op->body));
    return SeqStmt(prefix, op->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const EvaluateObj *op,
                                 InplaceMode mode) final {
    const auto *call = op->value.as<CallObj>();
    if (!call || !IsAnnotationCall(call)) {
      return LayoutMapMutator::Mutate_(op, mode);
    }
    if (call->op.same_as(layout_map_op()))
      return LayoutMapMutator::Mutate_(op, mode);
    DictAttrs attrs = call->attrs.value().as_or_throw<DictAttrs>();
    auto schema = attrs.GetAttr<Array<Any>>("annotation_schema").value();
    auto fields =
        DecodeAnnotations(call->args[0].as_or_throw<tvm::Tuple>(), schema)
            .cast<Map<Any, Any>>();
    if (!fields.count(String("tl.tensor_spec"))) {
      return LayoutMapMutator::Mutate_(op, mode);
    }
    Array<Stmt> prefix =
        LowerSpecs(fields.at(String("tl.tensor_spec")), op->loc);
    if (fields.size() > 1) {
      // Retain every other annotation and its original operand indices.  The
      // operands remain explicit IR uses until that annotation's own consumer.
      Array<Any> remaining{String("map")};
      for (size_t i = 1; i < schema.size(); ++i) {
        auto pair = schema[i].cast<Array<Any>>();
        auto key = pair[0].cast<Array<Any>>();
        if (key.size() == 2 && key[0].cast<String>() == "str" &&
            key[1].cast<String>() == "tl.tensor_spec") {
          continue;
        }
        remaining.push_back(schema[i]);
      }
      auto dict = attrs->dict;
      dict.Set("annotation_schema", remaining);
      Call retained(call->ty, call->op, call->args, DictAttrs(dict),
                    call->ty_args, call->loc);
      prefix.push_back(
          LayoutMapMutator::Mutate_(Evaluate(retained, op->loc).get(), mode)
              .ValueOrUnchanged(Evaluate(retained, op->loc)));
    }
    return SeqStmt(prefix, op->loc);
  }

private:
  Expr SourceDataIdentity(Expr value) const {
    // Follow only exact data projections and declarations, matching source
    // Buffer.data identity. Byte-offset/reinterpret calls are not conflated
    // with their allocation merely because they physically alias it.
    std::unordered_set<const VarObj *> visited;
    while (true) {
      if (auto var = value.as<Var>()) {
        if (!visited.insert(var.value().get()).second)
          return *var;
        auto definition = definitions_.Get(var.value());
        if (!definition)
          return *var;
        if (definition.value().as<Var>()) {
          value = definition.value();
          continue;
        }
        if (const auto *call = definition.value().as<CallObj>();
            call && call->op.same_as(tirx::decl_tensor_op())) {
          value = call->args[0];
          continue;
        }
        return *var;
      }
      const auto *call = value.as<CallObj>();
      if (call && call->op.same_as(tirx::tensor_data_ptr_op())) {
        value = call->args[0];
        continue;
      }
      // An opaque pointer expression only aliases the same expression object.
      // Retain its real identity; a temporary Var would leave a dangling key.
      return value;
    }
  }

  bool HasAnnotatedLayout(const TensorVar &tensor) const {
    return annotated_storage_.count(SourceDataIdentity(tensor).get());
  }

  Array<PrimExpr> CompactStrides(const TensorVar &tensor) {
    PrimExpr stride = IntImm(tensor.ty()->DefaultIndexType(), 1);
    Array<PrimExpr> strides(tensor.shape().size(), stride);
    for (size_t i = tensor.shape().size(); i > 0; --i) {
      strides.Set(i - 1, analyzer_->Simplify(stride));
      stride = stride * tensor.shape()[i - 1];
    }
    return strides;
  }

  struct SegmentGeometry {
    Array<PrimExpr> strides;
    PrimExpr capacity;
  };

  SegmentGeometry SegmentedGeometry(const TensorVar &tensor,
                                    const Map<Any, Any> &spec) {
    auto pattern = tirx::MatchShapeStridePattern(tensor.ty());
    TVM_FFI_CHECK(pattern.has_value(), ValueError)
        << "Tensor annotation requires a shape-aligned strided layout: "
        << tensor->name;
    const auto &shape = tensor.shape();
    const auto &strides = pattern.value().get<1>();
    const auto &offset = pattern.value().get<2>();
    auto separators = spec.at(String("axis_separators")).cast<Array<Any>>();
    TVM_FFI_ICHECK(!separators.empty());
    // The old descriptor has one scalar offset, while segmentation produces
    // several physical coordinates. It accepts only a literal zero origin.
    TVM_FFI_ICHECK(prim::IsZero(offset))
        << "If element offsets are defined, there must be one element offset "
           "for each output index";
    std::vector<size_t> seps;
    for (const auto &separator : separators) {
      int64_t index = separator.cast<int64_t>();
      TVM_FFI_CHECK_GE(index, 0, ValueError);
      TVM_FFI_CHECK_LE(index, static_cast<int64_t>(shape.size()), ValueError);
      TVM_FFI_CHECK(seps.empty() || seps.back() <= static_cast<size_t>(index),
                    ValueError)
          << "axis_separators must be nondecreasing";
      seps.push_back(static_cast<size_t>(index));
    }
    bool compact = spec.count(String("compact"))
                       ? spec.at(String("compact")).cast<bool>()
                       : !tensor.ty()->layout.has_value();
    const PrimType index_type = tensor.ty()->DefaultIndexType();
    PrimExpr one = IntImm(index_type, 1);
    Array<PrimExpr> physical_shape(seps.size() + 1, one);
    std::vector<size_t> groups;
    size_t group = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
      // Deliberately advance once, preserving leading/trailing and repeated
      // separator behavior. Sorting, deduplication or a while loop changes it.
      if (group < seps.size() && i == seps[group])
        ++group;
      groups.push_back(group);
      if (compact)
        physical_shape.Set(group, physical_shape[group] * shape[i]);
    }
    if (!compact) {
      TVM_FFI_CHECK(!shape.empty(), ValueError);
      physical_shape.Set(0, strides[0] * shape[0]);
      for (size_t i = 0; i < seps.size(); ++i) {
        // The old implementation indexes outside shape for this combination.
        // Diagnose it instead of depending on an out-of-bounds source read.
        TVM_FFI_CHECK_LT(seps[i], shape.size(), ValueError)
            << "An explicit-stride axis separator at tensor rank has no "
               "physical storage extent";
        physical_shape.Set(i + 1, strides[seps[i]] * shape[seps[i]]);
      }
    } else if (physical_shape.size() == shape.size()) {
      // GetFlattenedBuffer returns the original no-stride descriptor in this
      // case, including duplicate separators.
      physical_shape = shape;
    }
    Array<PrimExpr> multipliers(physical_shape.size(), one);
    PrimExpr capacity = one;
    for (size_t i = physical_shape.size(); i > 0; --i) {
      multipliers.Set(i - 1, capacity);
      capacity = capacity * physical_shape[i - 1];
    }
    Array<PrimExpr> mapped_strides(shape.size(), one);
    PrimExpr inner = one;
    for (size_t i = shape.size(); i > 0; --i) {
      size_t axis = i - 1;
      if (i == shape.size() || groups[axis] != groups[i])
        inner = one;
      mapped_strides.Set(axis,
                         analyzer_->Simplify((compact ? inner : strides[axis]) *
                                             multipliers[groups[axis]]));
      if (compact)
        inner = inner * shape[axis];
    }
    // Some accepted repeated-separator descriptors address beyond their
    // reported physical shape. Do not reproduce an under-sized owned buffer.
    PrimExpr span = one;
    PrimExpr empty = prim::MakeConst(PrimType::Bool(), false);
    for (size_t i = 0; i < shape.size(); ++i) {
      span = span + (shape[i] - 1) * mapped_strides[i];
      empty = tvm::logical_or(empty, shape[i] == 0);
    }
    capacity = analyzer_->Simplify(tvm::if_then_else(
        empty, IntImm(index_type, 0), tvm::max(capacity, span)));
    return {mapped_strides, capacity};
  }

  TensorVar LogicalTensor(const TensorVar &tensor,
                          const Array<PrimExpr> &strides,
                          const Map<Any, Any> &spec, const Location &loc) {
    auto pattern = tirx::MatchShapeStridePattern(tensor.ty());
    TVM_FFI_ICHECK(pattern.has_value());
    bool broadcast = spec.at(String("buffer_type")).cast<String>() == "auto";
    Array<tirx::Iter> logical_iters;
    for (size_t i = 0; i < tensor.shape().size(); ++i) {
      PrimExpr stride = strides[i];
      if (broadcast) {
        stride = analyzer_->Simplify(tvm::if_then_else(
            tensor.shape()[i] == 1, IntImm(stride.ty(), 0), stride));
      }
      logical_iters.push_back(
          tirx::Iter(tensor.shape()[i], stride, tirx::Axis::Get("m")));
    }
    auto type = ffi::make_object<tirx::TensorTypeObj>(*tensor.ty().get());
    type->layout = tirx::TileLayout(
        logical_iters, {}, {{tirx::Axis::Get("m"), pattern.value().get<2>()}});
    TensorVar logical(tensor->name + (broadcast ? "_broadcast" : "_segmented"),
                      tirx::TensorType(type), loc);
    if (broadcast)
      broadcasts_.insert(logical.get());
    return logical;
  }

  Array<PrimExpr> BroadcastIndices(const TensorVar &tensor,
                                   const Array<PrimExpr> &indices) {
    TVM_FFI_CHECK_EQ(indices.size(), tensor.shape().size(), ValueError);
    Array<PrimExpr> result;
    for (size_t i = 0; i < indices.size(); ++i) {
      // Layout application may linearize coordinates before applying strides.
      // Normalize a broadcast coordinate first, so an out-of-range coordinate
      // in a unit dimension cannot carry into its neighboring dimension.
      result.push_back(analyzer_->Simplify(
          tvm::if_then_else(tensor.shape()[i] == 1,
                            prim::MakeConst(indices[i].ty(), 0), indices[i])));
    }
    return result;
  }

  Array<Stmt> LowerSpecs(const Any &specs, const Location &loc) {
    Array<Stmt> result;
    for (const auto &[key, value] : specs.cast<Map<Any, Any>>()) {
      TensorVar tensor = key.cast<TensorVar>();
      auto spec = value.cast<Map<Any, Any>>();
      auto pattern = tirx::MatchShapeStridePattern(tensor.ty());
      TVM_FFI_CHECK(pattern.has_value(), ValueError)
          << "Tensor annotation requires a shape-aligned strided layout: "
          << tensor->name;
      PrimExpr offset = pattern.value().get<2>();
      int64_t factor = spec.at(String("offset_factor")).cast<int64_t>();
      TVM_FFI_CHECK_GT(factor, 0, ValueError)
          << "offset_factor must be positive after normalization";
      if (factor > 1) {
        PrimExpr condition = analyzer_->Simplify(
            tvm::truncmod(offset, IntImm(offset.ty(), factor)) ==
            IntImm(offset.ty(), 0));
        TVM_FFI_CHECK(!prim::IsZero(condition), ValueError)
            << "Tensor " << tensor->name
            << " violates offset_factor=" << factor;
        if (!prim::IsOne(condition)) {
          result.push_back(
              AssertStmt(condition, StringImm("ValueError"),
                         {StringImm("Tensor " + tensor->name +
                                    " element offset must be divisible by " +
                                    std::to_string(factor))},
                         loc));
        }
      }
      if (materialized_.count(tensor.get()))
        continue;
      auto separators = spec.at(String("axis_separators")).cast<Array<Any>>();
      if (separators.empty() &&
          spec.at(String("buffer_type")).cast<String>() != "auto") {
        continue;
      }
      auto strides = separators.empty() ? pattern.value().get<1>()
                     : HasAnnotatedLayout(tensor)
                         ? CompactStrides(tensor)
                         : SegmentedGeometry(tensor, spec).strides;
      TensorVar logical = LogicalTensor(tensor, strides, spec, loc);
      // The original parameter retains physical strides for the packed ABI.
      // tensor_data_ptr denotes its allocation base; the view owns the offset,
      // so it is applied exactly once by normal layout lowering.
      result.push_back(Bind(logical,
                            Call(logical.ty(), tirx::decl_tensor_op(),
                                 {tensor.data()}, {}, {logical.ty()}, loc),
                            loc));
      VarRemapSet(tensor, logical);
    }
    return result;
  }

  sym::Analyzer analyzer_ = CreateTileLangAnalyzer();
  TensorSpecs specs_;
  Map<Var, Expr> definitions_;
  std::unordered_set<const ExprObj *> annotated_storage_;
  std::unordered_set<const VarObj *> materialized_;
  std::unordered_set<const VarObj *> broadcasts_;
};

} // namespace

tvm::transform::Pass LowerTensorSpec() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    CheckLayoutMapScopes(function);
    if (!(!function->GetAttr(tvm::attr::kDeclaration).value_or(false)))
      return function;
    function = LowerTensorDescriptorParameters(std::move(function));
    auto collector = ffi::make_object<TensorSpecCollector>();
    collector->Visit(function->body);
    auto lowerer = ffi::make_object<TensorSpecLowerer>(*collector);
    SeqStmt body =
        lowerer->Mutate(function->body).ValueOrUnchanged(function->body);
    if (!body.same_as(function->body))
      function.CopyOnWrite()->body = body;
    // Declaration provenance cannot be recovered from address equivalence:
    // StridedTensor((1, n), (n, 1)) still checks the first descriptor stride.
    // Carry only constant flags while these passes preserve parameter order;
    // LowerTensorLayouts consumes them before host/device outlining.
    auto flags = function->GetAttr<Array<Any>>("tl.explicit_stride_params")
                     .value_or(Array<Any>(function->params.size(), Any(false)));
    TVM_FFI_CHECK_EQ(flags.size(), function->params.size(), ValueError)
        << "Explicit stride parameter metadata must match the signature";
    bool has_explicit = false;
    for (size_t i = 0; i < function->params.size(); ++i) {
      bool explicit_stride =
          flags[i].cast<bool>() || collector->explicit_stride_parameters.count(
                                       function->params[i].get());
      flags.Set(i, explicit_stride);
      has_explicit |= explicit_stride;
    }
    if (has_explicit) {
      function = WithAttr(std::move(function), "tl.explicit_stride_params",
                          std::move(flags));
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LowerTensorSpec");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerTensorSpec",
                                   LowerTensorSpec);
}

} // namespace tvm::tl
