/*!
 * \file legalize_safe_memory_access.cc
 * \brief Preserve TileLang bounds and padding contracts on native tensor IR.
 */

#include "../ir/analyzer.h"
#include "../ir/annotations.h"
#include "../op/native_intrinsic.h"
#include "../op/native_utils.h"
#include "../op/utils.h"
#include "common/constr_visitor.h"
#include "ir/lexical_scope.h"

#include "ir/stmt_functor.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/prim/vector_expr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/transform.h>

#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace tvm::tl {
namespace {
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;
using tirx::TensorVar;

// Bounds need a scalar predicate for every physical lane. Rebuild arithmetic
// from scalar operands so its native result type is inferred, not copied from
// the vector expression. Unknown vector operations remain an explicit extract.
PrimExpr ScalarLane(const PrimExpr &value, int lane) {
  if (value.ty().IsScalar())
    return value;
  if (const auto *op = value.as<prim::RampObj>())
    return op->base + op->stride * IntImm(op->stride.ty(), lane);
  if (const auto *op = value.as<prim::BroadcastObj>())
    return op->value;
  if (const auto *op = value.as<prim::CastObj>())
    return prim::cast(ScalarLane(op->value, lane),
                      op->ty.as_or_throw<PrimType>().WithLanes(1));
  if (const auto *op = value.as<prim::ShuffleObj>()) {
    TVM_FFI_CHECK_LT(static_cast<size_t>(lane), op->indices.size(), ValueError);
    auto index = op->indices[lane].as<IntImmObj>();
    TVM_FFI_CHECK(index, ValueError)
        << "Safe-memory shuffle index must be constant";
    int64_t selected = static_cast<int64_t>(index->value);
    for (const PrimExpr &input : op->vectors) {
      if (selected < input.ty().lanes())
        return ScalarLane(input, selected);
      selected -= input.ty().lanes();
    }
    TVM_FFI_THROW(ValueError) << "Safe-memory shuffle index out of range";
  }
#define TL_SCALAR_BINARY(Name)                                                 \
  if (const auto *op = value.as<prim::Name##Obj>())                            \
  return prim::Name(ScalarLane(op->a, lane), ScalarLane(op->b, lane), op->loc)
  TL_SCALAR_BINARY(Add);
  TL_SCALAR_BINARY(Sub);
  TL_SCALAR_BINARY(Mul);
  TL_SCALAR_BINARY(Div);
  TL_SCALAR_BINARY(Mod);
  TL_SCALAR_BINARY(FloorDiv);
  TL_SCALAR_BINARY(FloorMod);
  TL_SCALAR_BINARY(Min);
  TL_SCALAR_BINARY(Max);
  TL_SCALAR_BINARY(EQ);
  TL_SCALAR_BINARY(NE);
  TL_SCALAR_BINARY(LT);
  TL_SCALAR_BINARY(LE);
  TL_SCALAR_BINARY(GT);
  TL_SCALAR_BINARY(GE);
  TL_SCALAR_BINARY(And);
  TL_SCALAR_BINARY(Or);
#undef TL_SCALAR_BINARY
  if (const auto *op = value.as<prim::NotObj>())
    return prim::Not(ScalarLane(op->a, lane), op->loc);
  if (const auto *op = value.as<prim::SelectObj>())
    return prim::Select(ScalarLane(op->condition, lane),
                        ScalarLane(op->true_value, lane),
                        ScalarLane(op->false_value, lane), op->loc);
  return prim::Shuffle::ExtractElement(value, lane, value->loc);
}

std::string OpName(const CallObj *call) {
  if (const auto *op = call->op.as<OpObj>())
    return op->name;
  return "";
}

bool IsAtomic(const CallObj *call) {
  const std::string name = OpName(call);
  return name.rfind("tl.atomic", 0) == 0;
}

bool IsAsyncCopy(const CallObj *call) {
  const std::string name = OpName(call);
  return name == "tl.ptx_cp_async" || name == "tl.ptx_cp_async_bytes" ||
         name == "tirx.ptx_cp_async" || name == "cuda.ptx_cp_async";
}

struct Access {
  TensorLoad base;
  int mask;
};

std::optional<Access> TensorAccess(const Expr &expression) {
  const auto *call = expression.as<CallObj>();
  if (!call || OpName(call) != "tl.access_ptr")
    return std::nullopt;
  TVM_FFI_CHECK_EQ(call->args.size(), 3U, ValueError);
  auto load = call->args[0].as_or_throw<TensorLoad>();
  const auto *mask = call->args[2].as<IntImmObj>();
  TVM_FFI_CHECK(mask, ValueError) << "access_ptr rw_mask must be constant";
  TVM_FFI_CHECK(mask->value >= 0 && mask->value <= 3, ValueError);
  return Access{load, static_cast<int>(static_cast<int64_t>(mask->value))};
}

class SafeMemoryRewriter : public tl::StmtExprMutatorWithAnalyzer {
public:
  explicit SafeMemoryRewriter(const sym::Analyzer &analyzer)
      : StmtExprMutatorWithAnalyzer(analyzer) {}
  using StmtExprMutatorWithAnalyzer::Mutate;
  using StmtExprMutatorWithAnalyzer::Mutate_;

  SeqStmt Rewrite(const tirx::Function &function) {
    MarkTensorParamShapes(function);
    // A volatile load is a new observation even in the absence of a store.
    // Conservatively disable memory facts for the whole function when any
    // supported volatile annotation is present, including aliases of storage.
    VisitNativePostOrder(function->body, [&](const ffi::ObjectRef &object) {
      if (const auto *region = object.as<RegionStmtObj>();
          region && OpNameForRegion(region) == "tl.attr") {
        auto fields = DecodeAttributeRegion(region);
        has_volatile_memory_ |=
            fields.at("key").cast<String>() == "volatile_scope";
      }
      if (const auto *call = object.as<CallObj>(); call && call->attrs) {
        if (auto attrs = call->attrs.value().as<DictAttrs>()) {
          has_volatile_memory_ |= (*attrs)->dict.count("tirx.volatile") ||
                                  (*attrs)->dict.count("volatile_scope");
        }
      }
    });
    return Mutate(function->body).ValueOrUnchanged(function->body);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    const auto saved = safe_values_;
    auto result = StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    safe_values_ = saved;
    memory_assumptions_.clear();
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const IfObj *op, InplaceMode mode) final {
    // A branch can change the observed storage. Facts established in one
    // branch must not be available in its sibling or after the join.
    memory_assumptions_.clear();
    auto result = StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    memory_assumptions_.clear();
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    memory_assumptions_.clear();
    auto result = StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    memory_assumptions_.clear();
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const WhileObj *op, InplaceMode mode) final {
    memory_assumptions_.clear();
    auto result = StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    memory_assumptions_.clear();
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    // Keep safe-value inheritance across typed views of the same data pointer.
    if (op->var.as<TensorVar>()) {
      if (const auto *call = op->value.as<CallObj>();
          call && (call->op.same_as(tirx::decl_tensor_op()) ||
                   OpName(call) == "tirx.view")) {
        if (!call->args.empty()) {
          if (auto identity = PointerIdentity(call->args[0]))
            aliases_.Set(op->var, identity.value());
        }
      }
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    if (OpNameForRegion(op) != "tl.attr") {
      memory_assumptions_.clear();
      auto result = StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
      memory_assumptions_.clear();
      return result;
    }
    auto fields = DecodeAttributeRegion(op);
    const auto saved = safe_values_;
    const String key = fields.at("key").cast<String>();
    if (key == "tl.kernel" || key == "tl.sblock") {
      const auto metadata = CollectSourceBlockAnnotations(op->body);
      if (auto values = metadata.Get("safe_value_map"))
        ReadSafeMap(*values);
    }
    bool consumed = false;
    bool updated_kernel = false;
    if (key == "safe_value_map") {
      ReadSafeMap(fields.at("value"));
      consumed = true;
    } else if (key == "tl.kernel") {
      auto values = fields.at("value").cast<Map<Any, Any>>();
      if (values.count(String("safe_value_map"))) {
        ReadSafeMap(values.at(String("safe_value_map")));
        values.erase(String("safe_value_map"));
        fields.Set("value", values);
        updated_kernel = true;
      }
    }
    SeqStmt body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
    safe_values_ = saved;
    if (consumed)
      return RegionStmt(tl::lexical_scope_op(), {}, {}, DictAttrs(), body, {},
                        op->loc);
    if (updated_kernel) {
      auto encoded = EncodeAnnotations(fields);
      return RegionStmt(op->op, {encoded.payload}, op->body_params,
                        DictAttrs({{"schema_version", 1},
                                   {"annotation_schema", encoded.schema}}),
                        body, op->result_vars, op->loc);
    }
    return RegionStmt(op->op, op->args, op->body_params, op->attrs, body,
                      op->result_vars, op->loc);
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    // A foreign/read-state callback can observe or change the index storage.
    // Plain TensorLoad nodes are handled separately and retain their facts
    // until a store; an assume itself only establishes the new observation.
    if (!op->op.same_as(prim::assume_op()) &&
        SideEffect(ffi::GetRef<Expr>(op)) > CallEffectKind::kPure)
      memory_assumptions_.clear();
    if (OpName(op) == "tl.access_ptr") {
      // This TensorLoad denotes an address: do not turn it into a conditional
      // value load. Only its index expressions are evaluated by the caller.
      const auto base = op->args[0].as_or_throw<TensorLoad>();
      Array<PrimExpr> indices = base->indices.Map([&](const PrimExpr &index) {
        return Mutate(index, mode).ValueOrUnchanged(index);
      });
      Array<Expr> args{TensorLoad(base->source, indices, base->loc)};
      for (size_t i = 1; i < op->args.size(); ++i)
        args.push_back(Mutate(op->args[i], mode).ValueOrUnchanged(op->args[i]));
      return Call(op->ty, op->op, args, op->attrs, op->ty_args, op->loc);
    }
    Expr changed =
        StmtExprMutatorWithAnalyzer::Mutate_(op, mode).ValueOrUnchanged(
            ffi::GetRef<Call>(op));
    const auto *call = changed.as<CallObj>();
    if (!call || !IsAtomic(call) || !changed.as<PrimExpr>())
      return changed;
    Array<PrimExpr> conditions = CallConditions(call);
    if (conditions.empty() || call->args.empty())
      return changed;
    auto pointer = TensorAccess(call->args[0]);
    if (!pointer)
      return changed;
    PrimExpr value = changed.as_or_throw<PrimExpr>();
    return GuardValue(
        value,
        SafeValue(pointer->base->source.as_or_throw<TensorVar>(), value.ty()),
        conditions);
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *op,
                                     InplaceMode mode) final {
    auto load = StmtExprMutatorWithAnalyzer::Mutate_(op, mode)
                    .ValueOrUnchanged(ffi::GetRef<TensorLoad>(op))
                    .as_or_throw<TensorLoad>();
    const auto tensor = load->source.as_or_throw<TensorVar>();
    auto conditions = Bounds(tensor, load->indices);
    if (conditions.empty())
      return load;
    return GuardValue(load, SafeValue(tensor, load.ty()), conditions);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreObj *op,
                                 InplaceMode mode) final {
    auto store = StmtExprMutatorWithAnalyzer::Mutate_(op, mode)
                     .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                     .as_or_throw<TensorStore>();
    Stmt guarded = GuardStatement(
        store, Bounds(store->dest.as_or_throw<TensorVar>(), store->indices));
    // Do not assume distinct TensorVars cannot alias. The RHS and indices
    // were evaluated before this write, but all later reads may see it.
    memory_assumptions_.clear();
    return guarded;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const EvaluateObj *op,
                                 InplaceMode mode) final {
    const auto *call = op->value.as<CallObj>();
    if (!call)
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    if (call->op.same_as(prim::assume_op())) {
      auto result = StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
      Stmt statement =
          std::move(result).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
      const auto *value = statement.as<EvaluateObj>();
      const auto *fact = value ? value->value.as<CallObj>() : nullptr;
      if (fact && fact->op.same_as(prim::assume_op()) &&
          !has_volatile_memory_) {
        PrimExpr condition = fact->args[0].as_or_throw<PrimExpr>();
        bool memory_read = false;
        bool stable_reads = true;
        VisitNativePostOrder(condition, [&](const ffi::ObjectRef &object) {
          memory_read |= object.as<TensorLoadObj>() != nullptr;
          if (auto nested = object.as<Call>())
            stable_reads &= SideEffect(*nested) <= CallEffectKind::kPure;
        });
        if (memory_read && stable_reads && condition.ty().IsScalar())
          memory_assumptions_.push_back(condition);
      }
      return statement;
    }
    if (OpName(call) == "tl.annotation") {
      auto attrs = call->attrs.value().as_or_throw<DictAttrs>();
      auto values = DecodeAnnotations(
                        call->args[0].as_or_throw<tvm::Tuple>(),
                        attrs.GetAttr<Array<Any>>("annotation_schema").value())
                        .cast<Map<Any, Any>>();
      if (!values.count(String("safe_value_map")))
        return ffi::Unchanged();
      if (!values.count(String("tl.source_block_annotation")))
        ReadSafeMap(values.at(String("safe_value_map")));
      values.erase(String("safe_value_map"));
      if (values.empty())
        return SeqStmt(Array<Stmt>{}, op->loc);
      auto encoded = EncodeAnnotations(values);
      auto dict = attrs->dict;
      dict.Set("annotation_schema", encoded.schema);
      return Evaluate(Call(call->ty, call->op, {encoded.payload},
                           DictAttrs(dict), call->ty_args, call->loc),
                      op->loc);
    }
    if (IsAsyncCopy(call)) {
      memory_assumptions_.clear();
      return RewriteAsyncCopy(op, call);
    }
    const std::string name = OpName(call);
    if (IsAtomic(call) || name == "tirx.call_extern" ||
        name == "ir.call_extern" || name == "prim.call_extern") {
      // These statement paths deliberately bypass Mutate_(CallObj). They
      // still invalidate observations, including through raw pointer aliases.
      memory_assumptions_.clear();
      auto conditions = CallConditions(call);
      // Preserve external/atomic argument evaluation behind the guard; unlike
      // a scalar load, a pointer argument cannot be replaced by a safe value.
      return GuardStatement(ffi::GetRef<Stmt>(op), conditions);
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }

private:
  static String OpNameForRegion(const RegionStmtObj *region) {
    return region->op->name;
  }

  // A tensor access retains the logical address until native layout lowering.
  // Check every element of a transfer, not just the address's first lane.
  // Scalarized fallbacks must denote the same consecutive physical addresses
  // as cp.async; a non-contiguous view is not evidence for that contract.
  Array<Array<PrimExpr>> TransferIndices(const Access &access, int64_t count) {
    const TensorVar tensor = access.base->source.as_or_throw<TensorVar>();
    Array<PrimExpr> base;
    for (const PrimExpr &index : access.base->indices)
      base.push_back(ScalarLane(index, 0));
    TVM_FFI_CHECK(!base.empty() || count == 1, ValueError)
        << "A scalar tensor cannot describe a multi-element cp.async transfer";
    Array<Array<PrimExpr>> result{base};
    auto origin = tensor.ty()->ElemOffset(base);
    for (int64_t element = 1; element < count; ++element) {
      auto indices = base;
      auto last = indices.back();
      // Widen before adding the transfer displacement. A small signed index
      // must not wrap back into bounds at its maximum representable value.
      if ((last.ty().MatchesCode(kDLInt) || last.ty().MatchesCode(kDLUInt)) &&
          last.ty().bits() < 64)
        last = prim::cast(last, last.ty().WithBits(64));
      indices.Set(indices.size() - 1, last + IntImm(last.ty(), element));
      auto address = tensor.ty()->ElemOffset(indices);
      TVM_FFI_CHECK_EQ(origin.size(), 1U, ValueError)
          << "cp.async transfer requires one physical memory axis";
      TVM_FFI_CHECK_EQ(address.size(), 1U, ValueError);
      TVM_FFI_CHECK(Prove(address[0] == prim::cast(origin[0], address[0].ty()) +
                                            IntImm(address[0].ty(), element)),
                    ValueError)
          << "cp.async safe-memory access requires consecutive tensor "
             "elements; "
             "physical pointer operations must use an explicit raw pointer "
             "contract";
      result.push_back(std::move(indices));
    }
    return result;
  }

  Array<PrimExpr> TransferBounds(const Access &access,
                                 const Array<PrimExpr> &indices,
                                 int64_t element) {
    auto conditions =
        Bounds(access.base->source.as_or_throw<TensorVar>(), indices, true);
    if (element > 0 && !access.base->indices.empty()) {
      const PrimExpr base = ScalarLane(access.base->indices.back(), 0);
      if (base.ty().bits() == 64) {
        PrimExpr no_overflow =
            base <= prim::max_value(base.ty()) - IntImm(base.ty(), element);
        if (!Prove(no_overflow)) {
          // Evaluate the addition and its bounds only after representability
          // is established. Neither a wrapped uint64 index nor poison from
          // a signed overflow may turn an invalid tail into a valid element.
          return {if_then_else(no_overflow, Combine(conditions),
                               IntImm::Bool(false))};
        }
      }
    }
    return conditions;
  }

  Stmt RewriteAsyncCopy(const EvaluateObj *statement, const CallObj *call) {
    const auto operands = NativeIntrinsicOperands(call);
    TVM_FFI_CHECK(operands.size() == 3 || operands.size() == 4, ValueError);
    const auto src = TensorAccess(operands[1]);
    const auto dst = TensorAccess(operands[0]);
    // Raw pointers intentionally have no logical tensor extent. As in the
    // baseline, their producer is responsible for a valid physical transfer.
    if (!src || !dst)
      return ffi::GetRef<Stmt>(statement);
    TVM_FFI_CHECK((src->mask & 1) && (dst->mask & 2), ValueError)
        << "cp.async requires readable source and writable destination access "
           "contracts";
    const auto *amount = operands[2].as<IntImmObj>();
    TVM_FFI_CHECK(amount, ValueError)
        << "cp.async transfer size must be constant";
    int64_t count = static_cast<int64_t>(amount->value);
    const TensorVar source = src->base->source.as_or_throw<TensorVar>();
    const TensorVar destination = dst->base->source.as_or_throw<TensorVar>();
    PrimType source_type = source.dtype(), dest_type = destination.dtype();
    const int source_bits = source_type.bits() * source_type.lanes();
    const int dest_bits = dest_type.bits() * dest_type.lanes();
    TVM_FFI_CHECK_EQ(source_bits, dest_bits, TypeError)
        << "cp.async tensor access needs equally sized source and destination "
           "elements";
    if (OpName(call) != "tl.ptx_cp_async") {
      TVM_FFI_CHECK(source_bits > 0 && count > 0 && count <= 16 &&
                        count * 8 % source_bits == 0,
                    ValueError)
          << "cp.async byte count must describe whole tensor elements";
      count = count * 8 / source_bits;
    }
    // The largest legal PTX transfer is 128 bits (128 elements for int1).
    // Smaller pre-vectorization calls are allowed to be widened by that pass.
    TVM_FFI_CHECK(count > 0 && count <= 128, ValueError)
        << "cp.async element count must be in [1, 128]";
    const auto source_indices = TransferIndices(src.value(), count);
    const auto dest_indices = TransferIndices(dst.value(), count);
    Array<Array<PrimExpr>> source_bounds, dest_bounds;
    Array<PrimExpr> all_bounds;
    for (int64_t element = 0; element < count; ++element) {
      auto reads =
          TransferBounds(src.value(), source_indices[element], element);
      auto writes = TransferBounds(dst.value(), dest_indices[element], element);
      for (const auto &condition : reads)
        all_bounds.push_back(condition);
      for (const auto &condition : writes)
        all_bounds.push_back(condition);
      source_bounds.push_back(std::move(reads));
      dest_bounds.push_back(std::move(writes));
    }
    if (all_bounds.empty())
      return ffi::GetRef<Stmt>(statement);
    PrimExpr predicate = operands.size() == 4
                             ? operands[3].as_or_throw<PrimExpr>()
                             : IntImm::Bool(true);
    PrimExpr safe = SafeValue(source, source_type);
    // Only a single element can be zero-filled by one predicate without
    // dropping a valid prefix of a partially in-bounds transfer.
    if (count == 1 && IsZeroBitPattern(safe)) {
      auto condition = predicate && Combine(source_bounds[0]);
      Stmt copy = Evaluate(
          WithOperands(call, {operands[0], operands[1], operands[2],
                              GetSyncedAnalyzer()->Simplify(condition)}),
          statement->loc);
      return GuardStatement(copy, dest_bounds[0]);
    }
    Array<Stmt> fallback;
    for (int64_t element = 0; element < count; ++element) {
      PrimExpr value = GuardValue(
          TensorLoad(source, source_indices[element], statement->loc), safe,
          source_bounds[element]);
      // An explicit false cp.async predicate requests all-zero bytes, even
      // when safe_value_map specifies non-zero padding for an invalid source.
      value =
          if_then_else(predicate, value, MakeSafeValueConstant(source_type, 0));
      if (source_type != dest_type)
        value = Call(dest_type, tirx::reinterpret_op(), {value}, {}, {},
                     statement->loc)
                    .as_or_throw<PrimExpr>();
      fallback.push_back(
          GuardStatement(TensorStore(destination, dest_indices[element], value,
                                     statement->loc),
                         dest_bounds[element]));
    }
    return If(Combine(all_bounds), SeqStmt(ffi::GetRef<Stmt>(statement)),
              SeqStmt(fallback), statement->loc);
  }

  Call WithOperands(const CallObj *call, Array<Expr> operands) {
    auto attrs = call->attrs;
    if (attrs && attrs.value().as<DictAttrsObj>()) {
      auto dict = attrs.value().as_or_throw<DictAttrs>();
      if (dict->dict.count("operand_count")) {
        auto fields = dict->dict;
        fields.Set("operand_count", static_cast<int64_t>(operands.size()));
        operands.push_back(call->args.back());
        attrs = DictAttrs(fields);
      }
    }
    return Call(call->ty, call->op, std::move(operands), attrs, call->ty_args,
                call->loc);
  }

  ffi::Optional<Var> PointerIdentity(const Expr &expression) {
    if (auto variable = expression.as<Var>())
      return variable;
    const auto *call = expression.as<CallObj>();
    if (call && call->op.same_as(tirx::tensor_data_ptr_op()))
      return call->args[0].as<Var>();
    return std::nullopt;
  }

  void ReadSafeMap(const Any &map) {
    for (const auto &[key, raw] : map.cast<Map<Any, Any>>()) {
      Expr expression = key.cast<Expr>();
      auto identity = PointerIdentity(expression);
      TVM_FFI_CHECK(identity.has_value(), TypeError)
          << "safe_value_map keys must identify a native tensor or its data "
             "pointer";
      Type type = identity.value()->ty;
      auto tensor = identity.value().as<TensorVar>();
      TVM_FFI_CHECK(tensor.has_value(), TypeError)
          << "safe value needs a tensor element type";
      PrimExpr value{ffi::UnsafeInit{}};
      if (auto expression = raw.as<PrimExpr>()) {
        value = expression.value();
      } else {
        if (auto integer = raw.as<int64_t>())
          value =
              MakeSafeValueConstant(tensor.value().dtype(), integer.value());
        else if (auto real = raw.as<double>())
          value = MakeSafeValueConstant(tensor.value().dtype(), real.value());
        else
          TVM_FFI_THROW(TypeError) << "safe_value_map values must be primitive "
                                      "expressions or numbers";
      }
      safe_values_.Set(identity.value(), value);
    }
  }

  PrimExpr SafeValue(const TensorVar &tensor, PrimType type) {
    ffi::Optional<PrimExpr> found = safe_values_.Get(tensor);
    Var identity = tensor;
    // Alias chains follow definitions; a cycle is invalid native SSA and is
    // separately rejected by the IR verifier.
    while (!found && aliases_.count(identity)) {
      identity = aliases_.at(identity);
      found = safe_values_.Get(identity);
    }
    // A vector index loads several tensor elements. The result can therefore
    // have more lanes than the tensor's element dtype. Default padding is a
    // zero at that result type, not a vector that can be broadcast as a scalar.
    if (!found)
      return MakeSafeValueConstant(type, 0);
    PrimExpr value = found.value();
    if (value.ty().lanes() != type.lanes()) {
      if (value.ty().IsScalar() && type.IsFixedLengthVector()) {
        value = prim::Broadcast(value, type.lanes());
      } else {
        TVM_FFI_CHECK(value.ty().IsFixedLengthVector() &&
                          type.IsFixedLengthVector() &&
                          type.lanes() % value.ty().lanes() == 0,
                      TypeError)
            << "Safe padding must be a scalar or a whole vector tensor element";
        // Explicit per-element vector padding repeats in element-major lane
        // order for a multi-element load, preserving nonuniform lane values.
        Array<PrimExpr> elements;
        for (int i = 0; i < type.lanes() / value.ty().lanes(); ++i)
          elements.push_back(value);
        value = prim::Shuffle::Concat(elements, value->loc);
      }
    }
    return GetSyncedAnalyzer()->Simplify(prim::cast(value, type));
  }

  bool Prove(const PrimExpr &condition) {
    try {
      if (memory_assumptions_.empty())
        return GetSyncedAnalyzer()->CanProve(condition);
      // The native analyzer deliberately ignores mutable-memory predicates.
      // Within this query, the observations retained since the last possible
      // write denote stable values. Share only identical reads in that one
      // snapshot, without rewriting the IR or binding future loads to them.
      auto snapshot = ffi::make_object<FreshenMutableReads>(
          FreshenMutableReads::Mode::kSnapshot);
      ffi::Array<sym::Constraint> observations;
      for (const PrimExpr &fact : memory_assumptions_)
        observations.push_back(
            sym::PredicateConstraint((*snapshot)(fact), true));
      PrimExpr query = (*snapshot)(condition);
      return WithTraversalConstraints(this, observations, [&] {
        return GetSyncedAnalyzer()->CanProve(query);
      });
    } catch (const std::exception &) {
      return false;
    }
  }

  Array<PrimExpr> Bounds(const TensorVar &tensor,
                         const Array<PrimExpr> &indices,
                         bool transfer = false) {
    TVM_FFI_CHECK_EQ(indices.size(), tensor.shape().size(), ValueError);
    Array<PrimExpr> conditions;
    const bool global = tensor.scope().name() == "global";
    const bool warn = !tvm::transform::PassContext::Current()
                           ->GetConfig<bool>("tl.disable_out_of_bound_warning")
                           .value_or(true);
    for (size_t axis = 0; axis < indices.size(); ++axis) {
      const PrimExpr &index = indices[axis];
      // Preserve the baseline policy for explicitly constant indices. TileLang
      // automatic guards target symbolic accesses, not hand-written constants.
      bool symbolic = false;
      VisitNativePostOrder(index, [&](const ffi::ObjectRef &node) {
        if (node.as<VarObj>() || node.as<TensorLoadObj>())
          symbolic = true;
      });
      if (!symbolic && !transfer)
        continue;
      TVM_FFI_CHECK(index.ty().IsScalar() || index.ty().IsFixedLengthVector(),
                    TypeError);
      for (int lane = 0; lane < index.ty().lanes(); ++lane) {
        PrimExpr scalar =
            GetSyncedAnalyzer()->Simplify(ScalarLane(index, lane));
        for (const PrimExpr &condition :
             Array<PrimExpr>{scalar < tensor.shape()[axis], scalar >= 0}) {
          if (Prove(condition))
            continue;
          if (global || transfer)
            conditions.push_back(condition);
          else if (warn)
            LOG(WARNING) << "Index access may exceed tensor bounds: "
                         << tensor->name << ": " << condition;
        }
      }
    }
    return conditions;
  }

  Array<PrimExpr> CallConditions(const CallObj *call) {
    Array<PrimExpr> result;
    for (const Expr &arg : NativeIntrinsicOperands(call)) {
      if (auto access = TensorAccess(arg)) {
        if (access->mask & 3) {
          for (const auto &cond :
               Bounds(access->base->source.as_or_throw<TensorVar>(),
                      access->base->indices))
            result.push_back(cond);
        }
        continue;
      }
      VisitNativePostOrder(arg, [&](const ffi::ObjectRef &node) {
        if (const auto *load = node.as<TensorLoadObj>())
          for (const auto &cond :
               Bounds(load->source.as_or_throw<TensorVar>(), load->indices))
            result.push_back(cond);
      });
    }
    return result;
  }

  bool PureTotal(const Array<PrimExpr> &conditions) {
    for (const PrimExpr &condition : conditions) {
      if (SideEffect(condition) > CallEffectKind::kPure)
        return false;
      bool total = true;
      VisitNativePostOrder(condition, [&](const ffi::ObjectRef &node) {
        if (node.as<CallObj>() || node.as<TensorLoadObj>() ||
            node.as<prim::DivObj>() || node.as<prim::ModObj>() ||
            node.as<prim::FloorDivObj>() || node.as<prim::FloorModObj>())
          total = false;
      });
      if (!total)
        return false;
    }
    return true;
  }

  PrimExpr Combine(const Array<PrimExpr> &conditions) {
    PrimExpr condition = IntImm::Bool(true);
    for (const PrimExpr &part : conditions)
      condition = condition && part;
    return GetSyncedAnalyzer()->Simplify(condition);
  }

  PrimExpr GuardValue(PrimExpr value, const PrimExpr &safe,
                      const Array<PrimExpr> &conditions) {
    if (conditions.empty())
      return value;
    if (PureTotal(conditions))
      return if_then_else(Combine(conditions), value, safe);
    for (const PrimExpr &condition : conditions)
      value = if_then_else(condition, value, safe);
    return value;
  }

  Stmt GuardStatement(Stmt statement, const Array<PrimExpr> &conditions) {
    if (conditions.empty())
      return statement;
    if (PureTotal(conditions))
      return If(Combine(conditions), SeqStmt(statement), std::nullopt,
                statement->loc);
    for (const PrimExpr &condition : conditions)
      statement =
          If(condition, SeqStmt(statement), std::nullopt, statement->loc);
    return statement;
  }

  Map<Var, PrimExpr> safe_values_;
  Map<Var, Var> aliases_;
  std::vector<PrimExpr> memory_assumptions_;
  bool has_volatile_memory_{false};
};
} // namespace

tvm::transform::Pass LegalizeSafeMemoryAccess() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext context) {
    if (!(!function->GetAttr(tvm::attr::kDeclaration).value_or(false)) ||
        context->GetConfig<bool>("tl.disable_safe_memory_legalize")
            .value_or(false))
      return function;
    auto analyzer = CreateTileLangAnalyzer();
    auto mutator = ffi::make_object<SafeMemoryRewriter>(analyzer);
    function.CopyOnWrite()->body = mutator->Rewrite(function);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.LegalizeSafeMemoryAccess");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LegalizeSafeMemoryAccess",
                                   LegalizeSafeMemoryAccess);
}
} // namespace tvm::tl
