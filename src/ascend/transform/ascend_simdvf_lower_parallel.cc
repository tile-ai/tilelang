// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_simdvf_lower_parallel.cc
 * \brief Lower native SIMD_VF parallel loops and tile ops to register SSA.
 */
#include "ascend/ir/block.h"
#include "ir/analyzer.h"
#include "ir/annotations.h"
#include "ir/loop_math.h"
#include "ir/tensor_utils.h"
#include "op/reducer.h"
#include "op/utils.h"
#include <cmath>
#include <limits>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;
using namespace prim;

class SimdVFLowerer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit SimdVFLowerer(const tirx::Function &function) {
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        function->body,
        [&](const BindObj *bind) -> ffi::Expected<ffi::WalkResult> {
          definitions_.Set(bind->var, bind->value);
          return ffi::WalkResult::Advance();
        });
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *region,
                            InplaceMode mode) final {
    auto kind = ascend::GetVFKind(region);
    bool previous = inside_;
    if (kind != ascend::VFKind::kNone)
      inside_ = kind == ascend::VFKind::kSimd;
    auto result = StmtExprMutator::Mutate_(region, mode);
    inside_ = previous;
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const ForObj *loop, InplaceMode mode) final {
    if (!inside_ || loop->kind != ForKind::kParallel)
      return StmtExprMutator::Mutate_(loop, mode);
    RecordNarrowRange(loop);
    auto leaf = Peel(loop->body);
    const auto *inner = leaf.as<ForObj>();
    if (inner && inner->kind == ForKind::kParallel) {
      RecordNarrowRange(inner);
      auto nested = Peel(inner->body);
      const auto *third = nested.as<ForObj>();
      TVM_FFI_CHECK(!third || third->kind != ForKind::kParallel, ValueError)
          << "SIMD_VF supports at most two parallel dimensions";
      if (!StoreBody(inner->body))
        return StmtExprMutator::Mutate_(loop, mode);
      const auto *store = FirstStore(inner->body);
      SeqStmt leaf_body({RemoveInnerLoop(loop->body, inner)});
      if (StoreStride(store, inner->var) == 1 && prim::IsOne(inner->step))
        return LowerParallel(inner, loop, leaf_body);
      if (StoreStride(store, loop->var) == 1 && prim::IsOne(loop->step))
        return LowerParallel(loop, inner, leaf_body);
      TVM_FFI_THROW(ValueError)
          << "SIMD_VF 2D parallel requires a unit-step contiguous dimension";
    }
    if (!StoreBody(loop->body))
      return StmtExprMutator::Mutate_(loop, mode);
    return LowerParallel(loop, nullptr, loop->body);
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *evaluate,
                            InplaceMode mode) final {
    const auto *call = evaluate->value.as<CallObj>();
    const auto *op = call ? call->op.as<OpObj>() : nullptr;
    if (!inside_ || !op)
      return StmtExprMutator::Mutate_(evaluate, mode);
    if (op->name == "tl.tileop.fill")
      return LowerFill(call);
    if (op->name == "tl.tileop.copy" || op->name == "tl.tileop.ascend_copy")
      return LowerCopy(call);
    if (op->name == "tl.tileop.reduce")
      return LowerReduce(call);
    return StmtExprMutator::Mutate_(evaluate, mode);
  }

private:
  static int64_t VRegSize(PrimType dtype) {
    TVM_FFI_CHECK(dtype.lanes() == 1 && dtype.bits() >= 8 &&
                      dtype.bits() <= 64 && 2048 % dtype.bits() == 0,
                  TypeError)
        << "SIMD_VF parallel lowering needs an 8/16/32/64-bit scalar element "
           "type, got "
        << dtype;
    return 2048 / dtype.bits();
  }
  static PrimExpr I32(int64_t value) {
    return IntImm(PrimType::Int(32), value);
  }
  static Expr Text(const char *text) { return StringImm(text); }
  Var Register(PrimType dtype) {
    return Var("vreg_" + std::to_string(next_register_++),
               dtype.WithLanes(VRegSize(dtype)));
  }
  Call Intrinsic(const char *name, Type type, Array<Expr> operands,
                 Location loc = UnknownLoc(),
                 Map<String, Any> annotations = {}) {
    auto op = Op::Get(String("tl.simd.") + name);
    return annotations.empty()
               ? Call(type, op, operands, {}, {}, loc)
               : MakeAnnotatedCall(type, op, operands, annotations, loc);
  }
  Bind Mask(const Var &mask, PrimType dtype, Location loc) {
    return Bind(
        mask,
        Intrinsic("pset", mask->ty, {I32(dtype.bits()), Text("PAT_ALL")}, loc),
        loc);
  }
  static For Serial(PrimVar variable, PrimExpr end, SeqStmt body,
                    Location loc) {
    return For(variable, IntImm(variable.ty(), 0), std::move(end),
               IntImm(variable.ty(), 1), ForKind::kDefault, std::move(body), {},
               loc);
  }

  // Look through metadata while retaining it during reconstruction. Never
  // treat a computational region or a conditional as a transparent wrapper.
  static Stmt Peel(Stmt stmt) {
    while (true) {
      if (const auto *seq = stmt.as<SeqStmtObj>();
          seq && seq->seq.size() == 1) {
        stmt = seq->seq[0];
        continue;
      }
      if (const auto *region = stmt.as<RegionStmtObj>();
          region &&
          (region->op->name == "tl.attr" || ascend::IsAscendBlock(region)) &&
          region->body_params.empty() && region->result_vars.empty() &&
          ascend::GetVFKind(region) == ascend::VFKind::kNone) {
        stmt = region->body;
        continue;
      }
      return stmt;
    }
  }
  static Stmt RemoveInnerLoop(const Stmt &stmt, const ForObj *inner) {
    if (stmt.get() == inner)
      return inner->body;
    if (const auto *sequence = stmt.as<SeqStmtObj>()) {
      TVM_FFI_CHECK_EQ(sequence->seq.size(), 1U, ValueError);
      return SeqStmt({RemoveInnerLoop(sequence->seq[0], inner)}, stmt->loc);
    }
    auto region = stmt.as_or_throw<RegionStmt>();
    return RegionStmt(region->op, region->args, region->body_params,
                      region->attrs,
                      SeqStmt({RemoveInnerLoop(region->body, inner)}),
                      region->result_vars, region->loc);
  }
  static bool StoreBody(const Stmt &stmt) {
    if (stmt.as<TensorStoreObj>())
      return true;
    if (const auto *sequence = stmt.as<SeqStmtObj>()) {
      if (sequence->seq.empty())
        return false;
      for (const Stmt &child : sequence->seq)
        if (!StoreBody(child))
          return false;
      return true;
    }
    if (const auto *region = stmt.as<RegionStmtObj>())
      return (region->op->name == "tl.attr" || ascend::IsAscendBlock(region)) &&
             region->body_params.empty() && region->result_vars.empty() &&
             ascend::GetVFKind(region) == ascend::VFKind::kNone &&
             StoreBody(region->body);
    return false;
  }
  static const TensorStoreObj *FirstStore(const Stmt &stmt) {
    if (const auto *store = stmt.as<TensorStoreObj>())
      return store;
    if (const auto *sequence = stmt.as<SeqStmtObj>())
      return FirstStore(sequence->seq[0]);
    return FirstStore(stmt.as_or_throw<RegionStmt>()->body);
  }

  PrimExpr Substitute(const PrimExpr &expr) {
    auto mutator = make_object<StmtExprMutator>();
    for (const auto &[var, value] : substitutions_)
      mutator->VarRemapSet(var, value);
    return analyzer_->Simplify(mutator->Mutate(expr).ValueOrUnchanged(expr));
  }
  bool Uses(const PrimExpr &expr, const PrimVar &variable) {
    return ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
               expr,
               [&](const VarObj *var) -> ffi::Expected<ffi::WalkResult> {
                 return (var == variable.get())
                            ? ffi::WalkResult::Interrupt(
                                  ffi::VisitInterrupt(ffi::GetRef<Var>(var)))
                            : ffi::WalkResult::Skip();
               })
        .has_value();
  }
  void RecordNarrowRange(const ForObj *loop) {
    if (loop->var.ty().bits() > 16 || !prim::IsOne(loop->step))
      return;
    const auto *begin = loop->begin.as<IntImmObj>();
    const auto *end = loop->end.as<IntImmObj>();
    if (begin && end) {
      auto lo = begin->value.as<int64_t>(), hi = end->value.as<int64_t>();
      if (lo && hi && *hi > *lo && *hi - *lo <= 4096)
        narrow_ranges_[loop->var.get()] = {*lo, *hi};
    }
  }
  int64_t Stride(const PrimExpr &offset, const PrimVar &variable) {
    if (!Uses(offset, variable))
      return 0;
    // Compare physical offsets in a wide type. An int8 delta of +1 must not
    // disguise a memory discontinuity at the int8 wraparound boundary.
    auto wide_offset = prim::Cast(PrimType::Int(64), offset);
    auto substitute = make_object<StmtExprMutator>();
    substitute->VarRemapSet(variable, variable + IntImm(variable.ty(), 1));
    auto next = substitute->Mutate(wide_offset).ValueOrUnchanged(wide_offset);
    auto difference = analyzer_->Simplify(next - wide_offset);
    const auto *constant = difference.as<IntImmObj>();
    if (constant)
      return constant->value.as<int64_t>().value_or(-1);
    // Narrow machine indices can have casts whose generic difference is not
    // affine. For a small complete constant domain, establish every adjacent
    // physical difference. This never extends a proof beyond the loop domain.
    auto range = narrow_ranges_.find(variable.get());
    if (range == narrow_ranges_.end() ||
        range->second.second - range->second.first < 2)
      return -1;
    Optional<int64_t> stride;
    Optional<PrimExpr> previous;
    for (int64_t i = range->second.first; i < range->second.second; ++i) {
      auto point = make_object<StmtExprMutator>();
      point->VarRemapSet(variable, IntImm(variable.ty(), i));
      auto current = analyzer_->Simplify(
          point->Mutate(wide_offset).ValueOrUnchanged(wide_offset));
      if (previous) {
        auto delta = analyzer_->Simplify(current - *previous);
        const auto *literal = delta.as<IntImmObj>();
        auto value = literal ? literal->value.as<int64_t>() : std::nullopt;
        if (!value || (stride && *stride != *value))
          return -1;
        stride = *value;
      }
      previous = current;
    }
    return stride.value_or(-1);
  }
  int64_t StoreStride(const TensorStoreObj *store, const PrimVar &variable) {
    auto tensor = store->dest.as_or_throw<TensorVar>();
    return Stride(TensorElementOffset(tensor, store->indices), variable);
  }
  Expr Address(const TensorVar &tensor, Array<PrimExpr> indices, int access,
               int64_t elements, Location loc) {
    for (size_t i = 0; i < indices.size(); ++i)
      indices.Set(i, Substitute(indices[i]));
    auto index_type = indices.empty() ? PrimType::Int(32) : indices[0].ty();
    auto extent_type = index_type.bits() < 32 ? PrimType::Int(32) : index_type;
    return MakeTensorAccessPtr(TensorLoad(tensor, indices, loc),
                               IntImm(extent_type, elements), access);
  }

  Var Decompose(const PrimExpr &expression, PrimType dtype, const Var &mask,
                Array<Stmt> *statements) {
    if (const auto *load = expression.as<TensorLoadObj>()) {
      auto tensor = load->source.as_or_throw<TensorVar>();
      TVM_FFI_CHECK(tensor.dtype() == dtype &&
                        load->ty.as_or_throw<PrimType>() == dtype,
                    TypeError)
          << "SIMD_VF element expression needs one uniform scalar dtype";
      int64_t stride =
          Stride(TensorElementOffset(tensor, load->indices), *vector_var_);
      TVM_FFI_CHECK(stride == 0 || stride == 1, ValueError)
          << "SIMD_VF vector load requires unit physical stride or a scalar "
             "broadcast";
      Var value = Register(dtype);
      auto distribution = stride == 1
                              ? String("NORM")
                              : String("BRC_B" + std::to_string(dtype.bits()));
      statements->push_back(
          Bind(value,
               Intrinsic("vld", value->ty,
                         {Address(tensor, load->indices, 1,
                                  stride ? vector_size_ : 1, load->loc),
                          StringImm(distribution)},
                         load->loc),
               load->loc));
      return value;
    }
    auto binary = [&](const PrimExpr &left, const PrimExpr &right,
                      const char *name) {
      Var lhs = Decompose(left, dtype, mask, statements);
      Var rhs = Decompose(right, dtype, mask, statements);
      Var output = Register(dtype);
      statements->push_back(Bind(
          output,
          Intrinsic(name, output->ty, {lhs, rhs, mask, Text("MODE_ZEROING")},
                    expression->loc),
          expression->loc));
      return output;
    };
    if (const auto *op = expression.as<prim::AddObj>())
      return binary(op->a, op->b, "vadd");
    if (const auto *op = expression.as<prim::SubObj>()) {
      const auto *zero = op->a.as<FloatImmObj>();
      if (prim::IsZero(op->a) || (zero && zero->value == 0.0)) {
        Var input = Decompose(op->b, dtype, mask, statements);
        Var output = Register(dtype);
        statements->push_back(Bind(
            output,
            Intrinsic("vneg", output->ty, {input, mask, Text("MODE_ZEROING")},
                      expression->loc),
            expression->loc));
        return output;
      }
      return binary(op->a, op->b, "vsub");
    }
    if (const auto *op = expression.as<prim::MulObj>())
      return binary(op->a, op->b, "vmul");
    if (const auto *op = expression.as<prim::DivObj>())
      return binary(op->a, op->b, "vdiv");
    if (const auto *op = expression.as<prim::MaxObj>())
      return binary(op->a, op->b, "vmax");
    if (const auto *op = expression.as<prim::MinObj>())
      return binary(op->a, op->b, "vmin");
    if (const auto *call = expression.as<CallObj>()) {
      const auto *op = call->op.as<OpObj>();
      std::string name = op ? std::string(op->name) : "";
      const char *instruction = name == "prim.exp" ? "vexp"
                                : name == "prim.log" || name == "prim.ln"
                                    ? "vln"
                                : name == "prim.sqrt" ? "vsqrt"
                                : name == "prim.fabs" ? "vabs"
                                                      : nullptr;
      auto arguments = DecodeCallOperands(call);
      if (instruction && arguments.size() == 1) {
        auto input = Decompose(arguments[0].as_or_throw<PrimExpr>(), dtype,
                               mask, statements);
        auto output = Register(dtype);
        Map<String, Any> annotations;
        if (arguments.size() != call->args.size())
          annotations = DecodeTileOpArguments(call).annotations;
        else if (call->attrs) {
          if (auto attrs = call->attrs.value().as<DictAttrs>()) {
            annotations = attrs.value()->dict;
            annotations.erase("tl.source_annotation_keys");
          }
        }
        statements->push_back(
            Bind(output,
                 Intrinsic(instruction, output->ty,
                           {input, mask, Text("MODE_ZEROING")}, call->loc,
                           annotations),
                 call->loc));
        return output;
      }
    }
    bool scalar = expression.as<IntImmObj>() || expression.as<FloatImmObj>() ||
                  expression.as<VarObj>();
    if (const auto *cast = expression.as<prim::CastObj>()) {
      bool has_load = false;
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
          Evaluate(cast->value),
          [&](const ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
            has_load |= node.as<TensorLoadObj>() != nullptr;
            return ffi::WalkResult::Advance();
          });
      scalar =
          !ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
               cast->value,
               [&](const VarObj *var) -> ffi::Expected<ffi::WalkResult> {
                 return (vector_var_ && var == vector_var_.value().get())
                            ? ffi::WalkResult::Interrupt(
                                  ffi::VisitInterrupt(ffi::GetRef<Var>(var)))
                            : ffi::WalkResult::Skip();
               })
               .has_value() &&
          !has_load;
    }
    if (scalar && !Uses(expression, *vector_var_)) {
      auto value = Substitute(expression);
      if (value.ty() != dtype)
        value = prim::Cast(dtype, value);
      Var output = Register(dtype);
      statements->push_back(
          Bind(output,
               Intrinsic("vdup", output->ty,
                         {value, mask, Text("MODE_ZEROING")}, expression->loc),
               expression->loc));
      return output;
    }
    TVM_FFI_THROW(ValueError)
        << "Unsupported SIMD_VF vector expression: " << expression;
    return Var(ffi::UnsafeInit{});
  }

  Stmt LowerBody(const Stmt &stmt, PrimType dtype, const Var &mask) {
    if (const auto *store = stmt.as<TensorStoreObj>()) {
      auto destination = store->dest.as_or_throw<TensorVar>();
      TVM_FFI_CHECK(destination.dtype() == dtype &&
                        StoreStride(store, *vector_var_) == 1,
                    ValueError)
          << "SIMD_VF stores need one uniform dtype and contiguous physical "
             "vector dimension";
      Array<Stmt> sequence;
      Var value = Decompose(store->value, dtype, mask, &sequence);
      sequence.push_back(
          Evaluate(Intrinsic("vsts", PrimType::Void(),
                             {Address(destination, store->indices, 2,
                                      vector_size_, store->loc),
                              value, mask, Text("NORM_B32")},
                             store->loc),
                   store->loc));
      return SeqStmt(sequence, store->loc);
    }
    if (const auto *sequence = stmt.as<SeqStmtObj>()) {
      Array<Stmt> result;
      for (const Stmt &child : sequence->seq)
        result.push_back(LowerBody(child, dtype, mask));
      return SeqStmt(result, stmt->loc);
    }
    auto region = stmt.as_or_throw<RegionStmt>();
    Array<Expr> arguments;
    auto substitution = make_object<StmtExprMutator>();
    for (const auto &[var, value] : substitutions_)
      substitution->VarRemapSet(var, value);
    for (const Expr &arg : region->args)
      arguments.push_back(substitution->Mutate(arg).ValueOrUnchanged(arg));
    return RegionStmt(region->op, arguments, {}, region->attrs,
                      SeqStmt({LowerBody(region->body, dtype, mask)}), {},
                      region->loc);
  }

  Stmt LowerParallel(const ForObj *vector_loop, const ForObj *serial_loop,
                     SeqStmt leaf_body) {
    TVM_FFI_CHECK(prim::IsOne(vector_loop->step), ValueError)
        << "SIMD_VF vector dimension must have unit step";
    int vector_count = ConstantUnrollTripCount(vector_loop, analyzer_);
    int serial_count =
        serial_loop ? ConstantUnrollTripCount(serial_loop, analyzer_) : 1;
    TVM_FFI_CHECK(vector_count >= 0 && serial_count >= 0, ValueError)
        << "SIMD_VF parallel ranges need finite constant iteration counts";
    auto dtype =
        FirstStore(leaf_body)->dest->ty.as_or_throw<TensorType>()->dtype;
    vector_size_ = VRegSize(dtype);
    TVM_FFI_CHECK_EQ(vector_count % vector_size_, 0, ValueError)
        << "SIMD_VF parallel extent must be divisible by VReg size "
        << vector_size_;
    if (vector_count == 0 || serial_count == 0)
      return SeqStmt({}, vector_loop->loc);
    auto previous_vars = substitutions_;
    auto previous_vector = vector_var_;
    vector_var_ = vector_loop->var;
    // Counting complete registers needs a type which can represent VRegSize.
    // Keep the source loop's index type at the final substitution boundary.
    auto counter_type = [](PrimType type) {
      return type.bits() < 32 ? PrimType::Int(32) : type;
    };
    PrimVar chunk("vchunk", counter_type(vector_loop->var.ty()));
    PrimExpr vector_index = prim::Cast(chunk.ty(), vector_loop->begin) +
                            chunk * IntImm(chunk.ty(), vector_size_);
    substitutions_.Set(vector_loop->var,
                       prim::Cast(vector_loop->var.ty(), vector_index));
    PrimVar serial("vserial", serial_loop ? counter_type(serial_loop->var.ty())
                                          : PrimType::Int(32));
    if (serial_loop) {
      PrimExpr serial_index =
          prim::Cast(serial.ty(), serial_loop->begin) +
          serial * prim::Cast(serial.ty(), serial_loop->step);
      substitutions_.Set(serial_loop->var,
                         prim::Cast(serial_loop->var.ty(), serial_index));
    }
    Var mask("mask", PrimType::Bool(256));
    SeqStmt body({Mask(mask, dtype, vector_loop->loc),
                  LowerBody(leaf_body, dtype, mask)});
    Stmt result = For(chunk, IntImm(chunk.ty(), 0),
                      IntImm(chunk.ty(), vector_count / vector_size_),
                      IntImm(chunk.ty(), 1), ForKind::kDefault, body,
                      vector_loop->annotations, vector_loop->loc);
    if (serial_loop)
      result =
          For(serial, IntImm(serial.ty(), 0), IntImm(serial.ty(), serial_count),
              IntImm(serial.ty(), 1), ForKind::kDefault, SeqStmt({result}),
              serial_loop->annotations, serial_loop->loc);
    substitutions_ = previous_vars;
    vector_var_ = previous_vector;
    return result;
  }

  std::vector<int64_t> Extents(const TensorRegion &region, int64_t *total) {
    std::vector<int64_t> extents;
    *total = 1;
    for (const Range &range : region->region) {
      const auto *extent = range->extent.as<IntImmObj>();
      auto value = extent ? extent->value.as<int64_t>() : std::nullopt;
      TVM_FFI_CHECK(
          value && *value >= 0 &&
              (*value == 0 ||
               *total <= std::numeric_limits<int64_t>::max() / *value),
          ValueError)
          << "SIMD_VF tile regions need non-negative constant extents without "
             "overflow";
      extents.push_back(*value);
      *total *= *value;
    }
    return extents;
  }
  bool Contiguous(const TensorVar &tensor,
                  const std::vector<int64_t> &extents) {
    auto pattern = tirx::MatchShapeStridePattern(tensor.ty());
    if (!pattern || extents.size() != tensor.shape().size())
      return false;
    auto [shape, strides, origin] = pattern.value();
    int64_t expected = 1;
    for (size_t i = extents.size(); i-- > 0;) {
      if (extents[i] == 1)
        continue;
      if (!analyzer_->CanProveEqual(strides[i],
                                    IntImm(strides[i].ty(), expected)))
        return false;
      expected *= extents[i];
    }
    return true;
  }
  void FullRegion(const TensorRegion &region) {
    auto tensor = RegionTensor(region);
    TVM_FFI_CHECK_EQ(region->region.size(), tensor.shape().size(), ValueError);
    for (size_t i = 0; i < tensor.shape().size(); ++i)
      TVM_FFI_CHECK(prim::IsZero(region->region[i]->begin) &&
                        analyzer_->CanProveEqual(region->region[i]->extent,
                                                 tensor.shape()[i]),
                    ValueError)
          << "SIMD_VF copy/reduce requires full-buffer regions";
  }
  Array<PrimExpr> RegionIndices(const TensorRegion &region,
                                const std::vector<int64_t> &extents,
                                PrimExpr flat) {
    Array<PrimExpr> indices;
    for (size_t axis = extents.size(); axis-- > 0;) {
      PrimExpr index = flat;
      if (axis != 0) {
        index = floormod(flat, IntImm(flat.ty(), extents[axis]));
        flat = floordiv(flat, IntImm(flat.ty(), extents[axis]));
      }
      auto begin = region->region[axis]->begin;
      if (index.ty() != begin.ty())
        index = prim::Cast(begin.ty(), index);
      indices.insert(indices.begin(), begin + index);
    }
    return indices;
  }

  Optional<PrimExpr> ByteOrigin(const Expr &value,
                                std::unordered_set<const VarObj *> *active) {
    if (auto variable = value.as<Var>()) {
      if (!active->insert(variable.value().get()).second)
        return std::nullopt;
      auto definition = definitions_.Get(*variable);
      Optional<PrimExpr> result =
          definition ? ByteOrigin(*definition, active)
                     : Optional<PrimExpr>(IntImm(PrimType::Int(64), 0));
      active->erase(variable.value().get());
      return result;
    }
    if (const auto *call = value.as<CallObj>()) {
      if (call->op.same_as(tirx::alloc_tensor_op()))
        return IntImm(PrimType::Int(64), 0);
      if (call->op.same_as(tirx::decl_tensor_op()) ||
          call->op.same_as(tirx::tensor_data_ptr_op()) ||
          call->op.same_as(tirx::reinterpret_op()))
        return ByteOrigin(call->args[0], active);
      if (call->op.same_as(tirx::ptr_byte_offset_op())) {
        auto base = ByteOrigin(call->args[0], active);
        if (base)
          return base.value() +
                 prim::Cast(PrimType::Int(64),
                            call->args[1].as_or_throw<PrimExpr>());
      }
    }
    return std::nullopt;
  }
  void FillAlignment(const TensorRegion &region) {
    auto tensor = RegionTensor(region);
    Array<PrimExpr> starts;
    for (const Range &range : region->region)
      starts.push_back(range->begin);
    auto offset = TensorElementOffset(tensor, starts);
    std::unordered_set<const VarObj *> active;
    auto origin = ByteOrigin(tensor, &active);
    TVM_FFI_CHECK(origin, ValueError)
        << "SIMD_VF fill cannot establish its backing pointer's alignment";
    auto bytes = prim::Cast(PrimType::Int(64), offset) *
                     IntImm(PrimType::Int(64), tensor.dtype().bits() / 8) +
                 *origin;
    TVM_FFI_CHECK(
        analyzer_->CanProveEqual(floormod(bytes, IntImm(bytes.ty(), 32)),
                                 IntImm(bytes.ty(), 0)),
        ValueError)
        << "T.fill inside SIMD_VF requires a 32-byte-aligned region base, got "
        << bytes;
  }

  Stmt LowerFill(const CallObj *call) {
    auto decoded = DecodeTileOpArguments(call);
    TVM_FFI_CHECK_EQ(decoded.operands.size(), 2U, ValueError);
    auto region = NormalizeToTensorRegion(decoded.operands[0]);
    auto tensor = RegionTensor(region);
    auto dtype = tensor.dtype();
    auto size = VRegSize(dtype);
    int64_t total;
    auto extents = Extents(region, &total);
    TVM_FFI_CHECK(!extents.empty() && total > 0 && total % size == 0 &&
                      Contiguous(tensor, extents),
                  ValueError)
        << "T.fill inside SIMD_VF needs a contiguous region divisible by VReg "
           "size "
        << size;
    FillAlignment(region);
    auto value = decoded.operands[1].as_or_throw<PrimExpr>();
    if (value.ty() != dtype)
      value = prim::Cast(dtype, value);
    Var mask("mask", PrimType::Bool(256));
    Var value_register = Register(dtype);
    PrimVar chunk("vchunk", PrimType::Int(32));
    auto address =
        Address(tensor, RegionIndices(region, extents, chunk * I32(size)), 2,
                size, call->loc);
    auto store =
        Evaluate(Intrinsic("vsts", PrimType::Void(),
                           {address, value_register, mask, Text("NORM_B32")},
                           call->loc, decoded.annotations),
                 call->loc);
    return SeqStmt(
        {Mask(mask, dtype, call->loc),
         Bind(value_register,
              Intrinsic("vdup", value_register->ty,
                        {value, mask, Text("MODE_ZEROING")}, call->loc),
              call->loc),
         Serial(chunk, I32(total / size), SeqStmt({store}), call->loc)},
        call->loc);
  }

  Stmt LowerCopy(const CallObj *call) {
    auto decoded = DecodeTileOpArguments(call);
    TVM_FFI_CHECK_EQ(decoded.operands.size(), 2U, ValueError);
    auto source = NormalizeToTensorRegion(decoded.operands[0]);
    auto destination = NormalizeToTensorRegion(decoded.operands[1]);
    FullRegion(source);
    FullRegion(destination);
    auto src = RegionTensor(source), dst = RegionTensor(destination);
    auto from = src.dtype(), to = dst.dtype();
    bool convert = from != to;
    TVM_FFI_CHECK(!convert || (from == PrimType::Float(32) &&
                               (to == PrimType::Float(16) ||
                                to == PrimType(kDLBfloat, 16))),
                  TypeError)
        << "SIMD_VF cast copy supports float32 to float16/bfloat16";
    for (const char *key :
         {"transpose", "nd2nz", "dual_dst_ctl", "data_select"})
      if (auto value = decoded.annotations.Get(key))
        TVM_FFI_CHECK(!AnnotationAsBool(*value), ValueError)
            << "SIMD_VF copy does not accept DMA transform " << key;
    int64_t src_total, dst_total;
    auto src_extents = Extents(source, &src_total);
    auto dst_extents = Extents(destination, &dst_total);
    auto size = VRegSize(from);
    TVM_FFI_CHECK(src_total == dst_total && src_total % size == 0 &&
                      Contiguous(src, src_extents) &&
                      Contiguous(dst, dst_extents),
                  ValueError)
        << "SIMD_VF copy needs equal contiguous regions divisible by VReg size "
        << size;
    Var mask("mask", PrimType::Bool(256));
    PrimVar chunk("vchunk", PrimType::Int(32));
    Var loaded = Register(from);
    Array<Stmt> body;
    body.push_back(Bind(
        loaded,
        Intrinsic(
            "vld", loaded->ty,
            {Address(src, RegionIndices(source, src_extents, chunk * I32(size)),
                     1, size, call->loc),
             Text("NORM")},
            call->loc),
        call->loc));
    Var value = loaded;
    if (convert) {
      value = Register(to);
      body.push_back(
          Bind(value,
               Intrinsic("vcvt", value->ty,
                         {loaded, mask, Text("ROUND_R"), Text("RS_ENABLE"),
                          Text("PART_EVEN"), Text("MODE_ZEROING")},
                         call->loc, decoded.annotations),
               call->loc));
    }
    // PART_EVEN f32->f16/bf16 conversion populates the low half of each
    // 32-bit slot. PK_B32 stores those halves contiguously, matching
    // T.simd.vcvt
    // + vsts in the existing cast kernels; NORM_B32 would double the footprint.
    body.push_back(Evaluate(
        Intrinsic(
            "vsts", PrimType::Void(),
            {Address(dst,
                     RegionIndices(destination, dst_extents, chunk * I32(size)),
                     2, size, call->loc),
             value, mask, Text(convert ? "PK_B32" : "NORM_B32")},
            call->loc),
        call->loc));
    return SeqStmt(
        {Mask(mask, convert ? to : from, call->loc),
         Serial(chunk, I32(src_total / size), SeqStmt(body), call->loc)},
        call->loc);
  }

  Stmt LowerReduce(const CallObj *call) {
    auto decoded = DecodeTileOpArguments(call);
    TVM_FFI_CHECK_EQ(decoded.operands.size(), 5U, ValueError);
    auto source = NormalizeToTensorRegion(decoded.operands[0]);
    auto destination = NormalizeToTensorRegion(decoded.operands[1]);
    FullRegion(source);
    FullRegion(destination);
    auto src = RegionTensor(source), dst = RegionTensor(destination);
    auto type = src.dtype();
    int64_t src_total, dst_total;
    auto src_shape = Extents(source, &src_total);
    auto dst_shape = Extents(destination, &dst_total);
    const auto *dimension = decoded.operands[3].as<IntImmObj>();
    TVM_FFI_CHECK(dimension, ValueError);
    int dim = dimension->value.as<int>().value();
    TVM_FFI_CHECK(src_shape.size() == 2 && (dim == 0 || dim == 1) &&
                      (dst_shape.size() == 1 || dst_shape.size() == 2) &&
                      src.dtype() == dst.dtype(),
                  ValueError)
        << "SIMD_VF reduce requires a 2D source and matching 1D/keepdims "
           "output";
    TVM_FFI_CHECK_EQ(dst_total, src_shape[1 - dim], ValueError);
    if (dst_shape.size() == 2)
      TVM_FFI_CHECK(dst_shape[dim] == 1 &&
                        dst_shape[1 - dim] == src_shape[1 - dim],
                    ValueError);
    auto operation = decoded.operands[2].as_or_throw<StringImm>()->value;
    bool absolute = operation == "abssum" || operation == "absmax";
    bool sum = operation == "sum" || operation == "abssum";
    bool maximum = operation == "max" || operation == "absmax";
    bool minimum = operation == "min";
    bool bitwise =
        operation == "bitand" || operation == "bitor" || operation == "bitxor";
    TVM_FFI_CHECK(sum || maximum || minimum || bitwise, ValueError)
        << "Unsupported SIMD_VF reduction " << operation;
    TVM_FFI_CHECK(!bitwise || type.MatchesCode(kDLInt, kDLUInt), TypeError)
        << "Bitwise SIMD_VF reduction requires integer data";
    auto clear_flag = decoded.operands[4].as_or_throw<IntImm>();
    TVM_FFI_CHECK(clear_flag->value == 0 || clear_flag->value == 1, ValueError)
        << "SIMD_VF reduce clear must be a boolean literal";
    bool clear = clear_flag->value == 1;
    const char *element_op = sum                     ? "vadd"
                             : maximum               ? "vmax"
                             : minimum               ? "vmin"
                             : operation == "bitand" ? "vand"
                             : operation == "bitor"  ? "vor"
                                                     : "vxor";
    const char *horizontal_op = sum ? "vcadd" : maximum ? "vcmax" : "vcmin";
    int64_t size = VRegSize(type);
    int64_t rows = src_shape[0], columns = src_shape[1];
    if (dst_total == 0 || (src_shape[dim] == 0 && !clear))
      return SeqStmt({}, call->loc);
    if (src_shape[dim] == 0) {
      // The identity is part of the reducer contract, even when no source
      // element is read. Use the shared native helper instead of inventing
      // finite float sentinels or reading row zero of an empty tensor.
      auto combine = ParseReducerV2OpType(sum       ? String("sum")
                                          : maximum ? String("max")
                                          : minimum ? String("min")
                                                    : operation);
      auto identity = operation == "absmax" ? MakeConst(type, 0)
                                            : ReducerV2Identity(combine, type);
      Var mask("mask", PrimType::Bool(256));
      Var value = Register(type);
      PrimVar point("vpoint", PrimType::Int(32));
      auto store = Evaluate(
          Intrinsic("vsts", PrimType::Void(),
                    {Address(dst, RegionIndices(destination, dst_shape, point),
                             2, 1, call->loc),
                     value, mask, Text("ONEPT_B32")},
                    call->loc),
          call->loc);
      return SeqStmt(
          {Mask(mask, type, call->loc),
           Bind(value,
                Intrinsic("vdup", value->ty,
                          {identity, mask, Text("MODE_ZEROING")}, call->loc),
                call->loc),
           Serial(point, I32(dst_total), SeqStmt({store}), call->loc)},
          call->loc);
    }
    TVM_FFI_CHECK(rows > 0 && columns > 0 && columns % size == 0 &&
                      Contiguous(src, src_shape) && Contiguous(dst, dst_shape),
                  ValueError)
        << "SIMD_VF reduce needs contiguous rows whose column extent is "
           "divisible by VReg size";
    Var mask("mask", PrimType::Bool(256));
    PrimVar repeat(dim == 1 ? "vrow" : "vchunk", PrimType::Int(32));
    Array<Stmt> body;
    auto load = [&](const TensorVar &tensor, Array<PrimExpr> indices,
                    bool broadcast) {
      Var value = Register(type);
      String distribution = broadcast
                                ? String("BRC_B" + std::to_string(type.bits()))
                                : String("NORM");
      body.push_back(Bind(value,
                          Intrinsic("vld", value->ty,
                                    {Address(tensor, indices, 1,
                                             broadcast ? 1 : size, call->loc),
                                     StringImm(distribution)},
                                    call->loc),
                          call->loc));
      return value;
    };
    auto unary = [&](Var input, const char *op) {
      Call operation(std::nullopt, Op::Get(String("tl.simd.") + op),
                     {input, mask, Text("MODE_ZEROING")}, std::nullopt, {},
                     call->loc);
      Var output = Register(operation->ty.as_or_throw<PrimType>().WithLanes(1));
      body.push_back(Bind(output, operation, call->loc));
      return output;
    };
    auto combine = [&](Var left, Var right) {
      Var output = Register(type);
      body.push_back(
          Bind(output,
               Intrinsic(element_op, output->ty,
                         {left, right, mask, Text("MODE_ZEROING")}, call->loc),
               call->loc));
      return output;
    };
    auto destination_indices = [&](PrimExpr index) -> Array<PrimExpr> {
      return dst_shape.size() == 1 ? Array<PrimExpr>{index}
             : dim == 0            ? Array<PrimExpr>{I32(0), index}
                                   : Array<PrimExpr>{index, I32(0)};
    };
    Optional<Var> accumulator;
    if (dim == 1) {
      // A horizontal bitwise instruction is not part of the Ascend intrinsic
      // surface. Broadcast one scalar at a time and fold in register SSA;
      // unlike mapping every non-sum reduction to max this preserves the op.
      int64_t step = bitwise ? 1 : size;
      for (int64_t column = 0; column < columns; column += step) {
        Var value = load(src, {repeat, I32(column)}, bitwise);
        if (absolute)
          value = unary(value, "vabs");
        if (!bitwise) {
          value = unary(value, horizontal_op);
          if (sum && type.bits() == 16 && type.MatchesCode(kDLInt, kDLUInt)) {
            // Integer vcadd returns 32-bit lanes. Pack their low halves before
            // the existing 16-bit fold/store to retain modulo arithmetic.
            Var packed = Register(type);
            body.push_back(Bind(packed,
                                Intrinsic("vpack", packed->ty,
                                          {value, Text("LOWER")}, call->loc),
                                call->loc));
            value = packed;
          }
        }
        accumulator = accumulator ? combine(*accumulator, value) : value;
      }
    } else {
      // Vector lanes enumerate OUTPUT columns; the scalar fold enumerates
      // REDUCED rows. Their extents are independent, including non-square
      // inputs.
      for (int64_t row = 0; row < rows; ++row) {
        Var value = load(src, {I32(row), repeat * I32(size)}, false);
        if (absolute)
          value = unary(value, "vabs");
        accumulator = accumulator ? combine(*accumulator, value) : value;
      }
    }
    PrimExpr output_index = dim == 1 ? PrimExpr(repeat) : repeat * I32(size);
    if (!clear) {
      Var previous = load(dst, destination_indices(output_index), dim == 1);
      accumulator = combine(*accumulator, previous);
    }
    body.push_back(
        Evaluate(Intrinsic("vsts", PrimType::Void(),
                           {Address(dst, destination_indices(output_index), 2,
                                    dim == 1 ? 1 : size, call->loc),
                            *accumulator, mask,
                            Text(dim == 1 ? "ONEPT_B32" : "NORM_B32")},
                           call->loc),
                 call->loc));
    return SeqStmt({Mask(mask, type, call->loc),
                    Serial(repeat, I32(dim == 1 ? rows : columns / size),
                           SeqStmt(body), call->loc)},
                   call->loc);
  }

  bool inside_{false};
  int next_register_{0};
  int64_t vector_size_{0};
  Optional<PrimVar> vector_var_;
  Map<PrimVar, PrimExpr> substitutions_;
  Map<Var, Expr> definitions_;
  std::unordered_map<const VarObj *, std::pair<int64_t, int64_t>>
      narrow_ranges_;
  sym::Analyzer analyzer_{CreateTileLangAnalyzer()};
};
} // namespace

tvm::transform::Pass AscendSimdVFLowerParallel() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false)) {
      auto lowerer = ffi::make_object<SimdVFLowerer>(function);
      auto body = function->body;
      function.CopyOnWrite()->body =
          lowerer->Mutate(body).ValueOrUnchanged(body);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.AscendSimdVFLowerParallel");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.AscendSimdVFLowerParallel",
                                   AscendSimdVFLowerParallel);
}
} // namespace tvm::tl
