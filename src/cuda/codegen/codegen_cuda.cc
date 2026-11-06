/*!
 * \file target/codegen.cc
 */

#include "codegen_cuda.h"
#include "backend/common/codegen/codegen_ir_utils.h"
#include "backend/common/codegen/codegen_utils.h"
#include "cuda/op/builtin.h"
#include "ir/lexical_scope.h"
#include "ir/stmt_functor.h"
#include "layout/layout.h"
#include "op/builtin.h"
#include "support/check.h"
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/prim/pattern_match.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cuda/codegen/ptx.h"
#include "cuda/target_utils.h"

namespace tvm {
namespace codegen {
using namespace tvm::tl::codegen;
using namespace ffi;

namespace {

// A lexical region at the end of an existing C scope has the same lifetime
// endpoint. Retain preceding statements and native thread bindings; thread
// regions do not themselves introduce C braces. Never unwrap a region that
// has a following sibling or binds/returns values, or mutate the input IR.
SeqStmt UnwrapRedundantLexicalScope(SeqStmt body) {
  while (!body->seq.empty()) {
    const auto *region = body->seq.back().as<RegionStmtObj>();
    if (!region)
      break;
    if (region->op.same_as(tirx::launch_thread_op())) {
      SeqStmt inner = UnwrapRedundantLexicalScope(region->body);
      if (inner.same_as(region->body))
        break;
      RegionStmt launch = GetRef<RegionStmt>(region);
      launch.CopyOnWrite()->body = std::move(inner);
      Array<Stmt> statements = body->seq;
      statements.Set(statements.size() - 1, launch);
      return SeqStmt(std::move(statements), body->loc);
    }
    if (!region->body_params.empty() || !region->result_vars.empty())
      break;
    bool lexical = tl::IsLexicalScope(region) && region->args.empty();
    if (region->op->name == "tl.attr")
      lexical = DecodeTileLangAttribute(region).key == "lexical_alloc_scope";
    if (!lexical)
      break;
    Array<Stmt> statements;
    for (size_t i = 0; i + 1 < body->seq.size(); ++i)
      statements.push_back(body->seq[i]);
    for (const Stmt &statement : region->body->seq)
      statements.push_back(statement);
    body = SeqStmt(std::move(statements), body->loc);
  }
  return body;
}

bool IsUnpackedFP4SharedPointer(const Type &type) {
  const auto *pointer = type.as<PtrTypeObj>();
  if (!pointer)
    return false;
  const auto dtype = pointer->element_type.as<PrimType>();
  if (!dtype || !tl::IsFloat4E2M1Unpacked(*dtype) || !dtype->IsScalar())
    return false;
  const std::string scope = pointer->scope.name();
  return scope == "shared" || scope == "shared.dyn";
}

bool IsValidCPAsyncTransferBytes(int64_t bytes) {
  return bytes == 4 || bytes == 8 || bytes == 16;
}

bool IsProvablyDivisible(const PrimExpr &expr, int64_t divisor) {
  ICHECK_GT(divisor, 0);
  sym::Analyzer analyzer;
  sym::ModularSet modular_set = analyzer->modular_set(expr);
  return modular_set->coeff % divisor == 0 && modular_set->base % divisor == 0;
}

// A memory transaction can carry more 64-bit indices than CUDA has vector
// register types. Its scalar fallback must not first materialize int64x8 (or
// wider). Project only known lane-wise, pure expressions and retain every
// arithmetic/cast type; this is not a proof of contiguous or aligned memory.
std::optional<PrimExpr> ScalarMemoryIndexLane(const PrimExpr &index, int lane) {
  const PrimType type = index.ty();
  if (type.IsScalableVector() || lane < 0 || lane >= type.lanes())
    return std::nullopt;
  if (type.IsScalar())
    return index;
  if (const auto *op = index.as<prim::BroadcastObj>())
    return op->value;
  if (const auto *op = index.as<prim::RampObj>()) {
    PrimExpr offset =
        prim::Mul(op->stride, IntImm(op->stride.ty(), lane), op->loc);
    return prim::Add(op->base, offset, op->loc);
  }
  if (const auto *op = index.as<prim::CastObj>()) {
    auto value = ScalarMemoryIndexLane(op->value, lane);
    return value ? std::optional<PrimExpr>(
                       prim::Cast(type.WithLanes(1), *value, op->loc))
                 : std::nullopt;
  }
  if (const auto *op = index.as<prim::ShuffleObj>()) {
    const auto *selected = op->indices[lane].as<IntImmObj>();
    auto offset = selected ? selected->value.as<int>() : std::nullopt;
    if (!offset || *offset < 0)
      return std::nullopt;
    for (const PrimExpr &source : op->vectors) {
      if (source.ty().IsScalableVector())
        return std::nullopt;
      if (*offset < source.ty().lanes())
        return ScalarMemoryIndexLane(source, *offset);
      *offset -= source.ty().lanes();
    }
    return std::nullopt;
  }
  // C promotes narrow scalar arithmetic to int, while a packed vector lane
  // truncates at each operation. Do not introduce a widening across that
  // boundary; explicit Cast nodes above remain supported and retain it.
  if (!type.MatchesCode(kDLInt, kDLUInt) || type.bits() < 32)
    return std::nullopt;
#define TL_SCALAR_MEMORY_INDEX(Name)                                           \
  if (const auto *op = index.as<prim::Name##Obj>()) {                          \
    auto lhs = ScalarMemoryIndexLane(op->a, lane);                             \
    auto rhs = ScalarMemoryIndexLane(op->b, lane);                             \
    if (lhs && rhs)                                                            \
      return prim::Name(*lhs, *rhs, op->loc);                                  \
    return std::nullopt;                                                       \
  }
  TL_SCALAR_MEMORY_INDEX(Add);
  TL_SCALAR_MEMORY_INDEX(Sub);
  TL_SCALAR_MEMORY_INDEX(Mul);
#undef TL_SCALAR_MEMORY_INDEX
  return std::nullopt;
}

std::optional<std::vector<PrimExpr>>
ScalarWideMemoryIndices(const PrimExpr &index, int value_lanes) {
  const PrimType type = index.ty();
  if (!type.IsFixedLengthVector() || !type.MatchesCode(kDLInt, kDLUInt) ||
      type.bits() != 64 || type.lanes() <= 4 || type.lanes() != value_lanes ||
      SideEffect(index) != CallEffectKind::kPure)
    return std::nullopt;
  std::vector<PrimExpr> lanes;
  lanes.reserve(value_lanes);
  for (int lane = 0; lane < value_lanes; ++lane) {
    auto value = ScalarMemoryIndexLane(index, lane);
    if (!value)
      return std::nullopt;
    lanes.push_back(*value);
  }
  return lanes;
}

bool PackedIndexValueFits(const PrimType &type, const ffi::BigInt &value) {
  if (!type.MatchesCode(kDLInt, kDLUInt) || type.bits() <= 0)
    return false;
  const bool is_signed = type.MatchesCode(kDLInt);
  ffi::BigInt limit = ffi::BigInt(1) << (type.bits() - is_signed);
  return value < limit && (is_signed ? value >= -limit : value >= 0);
}

// Resolve only literal integer lanes and literal-valued dominating SSA binds.
// Check every arithmetic/cast result in its own type: widening a vector after
// an overflowing narrow operation must not turn it into mathematical indices.
std::optional<std::vector<ffi::BigInt>>
PackedIndexConstants(const PrimExpr &value, sym::AnalyzerObj *analyzer,
                     bool *invalid) {
  const PrimType type = value.ty();
  if (!type.MatchesCode(kDLInt, kDLUInt) || type.IsScalableVector() ||
      type.lanes() > 32)
    return std::nullopt;
  std::optional<std::vector<ffi::BigInt>> result;
  if (const auto *literal = value.as<IntImmObj>()) {
    result = std::vector<ffi::BigInt>{literal->value};
  } else if (value.as<VarObj>()) {
    const PrimExpr bound = analyzer->Simplify(value);
    if (const auto *literal = bound.as<IntImmObj>())
      result = std::vector<ffi::BigInt>{literal->value};
  } else if (const auto *broadcast = value.as<prim::BroadcastObj>()) {
    auto scalar = PackedIndexConstants(broadcast->value, analyzer, invalid);
    if (scalar && scalar->size() == 1)
      result = std::vector<ffi::BigInt>(type.lanes(), scalar->front());
  } else if (const auto *ramp = value.as<prim::RampObj>()) {
    auto base = PackedIndexConstants(ramp->base, analyzer, invalid);
    auto stride = PackedIndexConstants(ramp->stride, analyzer, invalid);
    if (base && stride && base->size() == 1 && stride->size() == 1) {
      result.emplace();
      for (int lane = 0; lane < type.lanes(); ++lane)
        result->push_back(base->front() + lane * stride->front());
    }
  } else if (const auto *cast = value.as<prim::CastObj>()) {
    result = PackedIndexConstants(cast->value, analyzer, invalid);
  } else if (const auto *shuffle = value.as<prim::ShuffleObj>()) {
    std::vector<ffi::BigInt> source;
    bool known = true;
    for (const PrimExpr &vector : shuffle->vectors) {
      auto values = PackedIndexConstants(vector, analyzer, invalid);
      if (values)
        source.insert(source.end(), values->begin(), values->end());
      else
        known = false;
    }
    if (known) {
      result.emplace();
      for (const PrimExpr &index : shuffle->indices) {
        const auto *literal = index.as<IntImmObj>();
        auto lane = literal ? literal->value.as<int>() : std::nullopt;
        if (!lane || *lane < 0 || static_cast<size_t>(*lane) >= source.size())
          return std::nullopt;
        result->push_back(source[*lane]);
      }
    }
  } else {
    PrimExpr lhs = value, rhs = value;
    enum { kNone, kAdd, kSub, kMul } operation = kNone;
    if (const auto *add = value.as<prim::AddObj>()) {
      lhs = add->a;
      rhs = add->b;
      operation = kAdd;
    } else if (const auto *sub = value.as<prim::SubObj>()) {
      lhs = sub->a;
      rhs = sub->b;
      operation = kSub;
    } else if (const auto *mul = value.as<prim::MulObj>()) {
      lhs = mul->a;
      rhs = mul->b;
      operation = kMul;
    }
    if (operation != kNone) {
      auto a = PackedIndexConstants(lhs, analyzer, invalid);
      auto b = PackedIndexConstants(rhs, analyzer, invalid);
      if (a && b && a->size() == b->size()) {
        result.emplace();
        for (size_t i = 0; i < a->size(); ++i) {
          if (operation == kAdd)
            result->push_back((*a)[i] + (*b)[i]);
          else if (operation == kSub)
            result->push_back((*a)[i] - (*b)[i]);
          else
            result->push_back((*a)[i] * (*b)[i]);
        }
      }
    }
  }
  if (result)
    for (const ffi::BigInt &lane : *result)
      if (!PackedIndexValueFits(type, lane)) {
        *invalid = true;
        return std::nullopt;
      }
  return result;
}

class PackedIndexConstantFolder : public tirx::StmtExprMutator {
public:
  explicit PackedIndexConstantFolder(sym::AnalyzerObj *analyzer)
      : analyzer_(analyzer) {}
  using StmtExprMutator::Mutate_;

  UnchangedOr<PrimExpr> Mutate_(const prim::ShuffleObj *node,
                                InplaceMode mode) final {
    return Fold(StmtExprMutator::Mutate_(node, mode)
                    .ValueOrUnchanged(GetRef<PrimExpr>(node)));
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::CastObj *node,
                                InplaceMode mode) final {
    return Fold(StmtExprMutator::Mutate_(node, mode)
                    .ValueOrUnchanged(GetRef<PrimExpr>(node)));
  }

private:
  PrimExpr Fold(PrimExpr value) {
    if (value.ty().IsScalar())
      return value;
    bool invalid = false;
    auto lanes = PackedIndexConstants(value, analyzer_, &invalid);
    if (invalid || !lanes || lanes->size() < 2)
      return value;
    ffi::BigInt stride = (*lanes)[1] - (*lanes)[0];
    const PrimType scalar = value.ty().WithLanes(1);
    if (!PackedIndexValueFits(scalar, stride))
      return value;
    for (size_t i = 2; i < lanes->size(); ++i)
      if ((*lanes)[i] != lanes->front() + ffi::BigInt(i) * stride)
        return value;
    if (stride == 0)
      return prim::Broadcast(IntImm(scalar, lanes->front()),
                             IntImm::Int32(lanes->size()), value->loc);
    // Native Ramp only represents 32- and 64-bit index arithmetic. Keep
    // narrow intermediates intact until the complete address is checked.
    if (scalar.bits() != 32 && scalar.bits() != 64)
      return value;
    return prim::Ramp(IntImm(scalar, lanes->front()), IntImm(scalar, stride),
                      IntImm::Int32(lanes->size()), value->loc);
  }
  sym::AnalyzerObj *analyzer_;
};

// Vector legalization represents literal lane vectors with Shuffle, often
// followed by a widening Cast and scalar Broadcast binds. Recover their ramp
// contract only for memory emission. This local copy leaves the input module
// and global simplifier unchanged. Non-subbyte accesses use this only for
// exact literal lanes; stores retain their alignment/unit-stride proof.
class PackedMemoryIndexNormalizer : public tl::StmtExprMutatorWithAnalyzer {
public:
  explicit PackedMemoryIndexNormalizer(sym::AnalyzerObj *analyzer)
      : StmtExprMutatorWithAnalyzer(analyzer) {}
  using StmtExprMutatorWithAnalyzer::Mutate_;

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *node,
                                InplaceMode mode) final {
    auto load = StmtExprMutatorWithAnalyzer::Mutate_(node, mode)
                    .ValueOrUnchanged(GetRef<PrimExpr>(node))
                    .as_or_throw<TensorLoad>();
    if (IsPackedAccess(load->source, load.ty(), load->indices.size())) {
      PrimExpr index = Normalize(load->indices[0], load.ty());
      if (!index.same_as(load->indices[0]))
        load.CopyOnWrite()->indices.Set(0, index);
    }
    return load;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj *node,
                            InplaceMode mode) final {
    auto store = StmtExprMutatorWithAnalyzer::Mutate_(node, mode)
                     .ValueOrUnchanged(GetRef<Stmt>(node))
                     .as_or_throw<TensorStore>();
    if (IsPackedAccess(store->dest, store->value.ty(), store->indices.size())) {
      PrimExpr index = Normalize(store->indices[0], store->value.ty());
      if (!index.same_as(store->indices[0]))
        store.CopyOnWrite()->indices.Set(0, index);
    }
    return store;
  }

private:
  static bool IsPackedAccess(const Expr &expr, const PrimType &value_type,
                             size_t index_count) {
    const auto tensor = expr.as<TensorVar>();
    if (!tensor || index_count != 1)
      return false;
    const PrimType element = tensor->dtype();
    return element.IsScalar() && !value_type.IsScalar() &&
           value_type.WithLanes(1) == element;
  }

  PrimExpr Normalize(const PrimExpr &index, const PrimType &value_type) {
    if (SideEffect(index) > CallEffectKind::kPure)
      return index;
    bool invalid = false;
    auto constants = PackedIndexConstants(index, GetSyncedAnalyzer(), &invalid);
    if (invalid)
      return index;
    const PrimType element = value_type.WithLanes(1);
    if (!constants && element != PrimType::Int(4) &&
        element != PrimType::UInt(4))
      return index;
    PrimExpr candidate = index;
    if (constants && index.ty().bits() < 32 && !index.ty().IsScalar()) {
      // Every intermediate cast and arithmetic result was checked in its
      // original finite type above. These exact addresses can now use a
      // native int32 carrier; uint8/uint16 values also fit it losslessly.
      Array<PrimExpr> values, lanes;
      for (size_t i = 0; i < constants->size(); ++i) {
        values.push_back(IntImm(PrimType::Int(32), (*constants)[i]));
        lanes.push_back(IntImm::Int32(i));
      }
      candidate = prim::Shuffle(values, lanes, index->loc);
    }
    auto folder =
        ffi::make_object<PackedIndexConstantFolder>(GetSyncedAnalyzer());
    PrimExpr folded = folder->Mutate(candidate, InplaceMode::kDisallow)
                          .ValueOrUnchanged(candidate);
    bool unresolved_vector_cast = false;
    ffi::StructuralVisit(
        folded,
        [&](const prim::CastObj *cast, ffi::StructuralVisitorObj *visitor)
            -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
          unresolved_vector_cast |= !cast->value.ty().IsScalar();
          return visitor->DefaultVisitExpected(cast);
        });
    if (unresolved_vector_cast)
      return index;
    PrimExpr normalized = GetSyncedAnalyzer()->Simplify(folded);
    PackedIndexConstants(normalized, GetSyncedAnalyzer(), &invalid);
    return !invalid && normalized.as<prim::RampObj>() ? normalized : index;
  }
};

std::optional<PrimType> GetAccessPtrElementType(const Expr &expr) {
  if (const auto *ptr = expr->ty.as<PtrTypeObj>()) {
    if (auto type = ptr->element_type.as<PrimType>())
      return type.value();
  }
  const auto *ptr_call = expr.as<CallObj>();
  if (ptr_call == nullptr) {
    return std::nullopt;
  }
  if (ptr_call->op.same_as(tirx::address_of_op())) {
    const auto *buffer_load = ptr_call->args[0].as<TensorLoadObj>();
    ICHECK(buffer_load) << "address_of arg must be BufferLoad";
    return buffer_load->source->ty.as_or_throw<TensorType>()->dtype;
  }
  if (ptr_call->op.same_as(tl::access_ptr())) {
    ICHECK_EQ(ptr_call->args.size(), 3U)
        << "tl.access_ptr expects 3 args: (BufferLoad, extent, rw_mask)";
    const auto *buffer_load = ptr_call->args[0].as<TensorLoadObj>();
    ICHECK(buffer_load) << "tl.access_ptr arg0 must be BufferLoad";
    return buffer_load->source->ty.as_or_throw<TensorType>()->dtype;
  }
  return std::nullopt;
}

int GetTileLangCPAsyncTransferBytes(const CallObj *op) {
  auto call = GetRef<Call>(op);
  ICHECK(call->args.size() == 3 || call->args.size() == 4)
      << call
      << " expects 3 or 4 arguments (dst_access_ptr, "
         "src_access_ptr, num_elems, [predicate])";
  const auto *num_elems_imm = call->args[2].as<IntImmObj>();
  ICHECK(num_elems_imm) << call << " num_elems must be IntImm, but got "
                        << call->args[2];
  int64_t num_elems = CodegenInt(num_elems_imm);
  ICHECK_GT(num_elems, 0) << call << " num_elems must be positive";

  auto dst_elem_type = GetAccessPtrElementType(call->args[0]);
  auto src_elem_type = GetAccessPtrElementType(call->args[1]);
  ICHECK(dst_elem_type.has_value() && src_elem_type.has_value())
      << call
      << " expects address_of, tl.access_ptr, or "
         "tvm_access_ptr operands";

  int64_t dst_total_bits =
      num_elems * dst_elem_type.value().bits() * dst_elem_type.value().lanes();
  int64_t src_total_bits =
      num_elems * src_elem_type.value().bits() * src_elem_type.value().lanes();
  ICHECK_EQ(dst_total_bits, src_total_bits)
      << call << " requires src/dst transfer widths to match, but got "
      << dst_total_bits << " vs " << src_total_bits << " bits";
  ICHECK_EQ(dst_total_bits % 8, 0)
      << call << " requires byte-aligned transfers, but got " << dst_total_bits
      << " bits";

  int64_t total_bytes = dst_total_bits / 8;
  ICHECK(IsValidCPAsyncTransferBytes(total_bytes))
      << call << " requires a final PTX byte width in {4, 8, 16}, but got "
      << total_bytes;
  return static_cast<int>(total_bytes);
}

bool CanEmitPackedX2Math(const PrimType &t) {
  int lanes = t.lanes();
  if (lanes < 2 || lanes % 2 != 0) {
    return false;
  }

  if (t.MatchesElementType(kDLBfloat, 16) ||
      t.MatchesElementType(kDLFloat, 16)) {
    return true;
  }

  if (t.MatchesCode(kDLFloat) && t.bits() == 32) {
    auto cur_target = Target::Current(/*allow_not_defined=*/true);
    return cur_target.has_value() &&
           tl::TargetHasSMVersionGE(cur_target.value(), 100);
  }

  return false;
}

const char *GetPackedBinaryOpName(const std::string &op) {
  static constexpr std::pair<const char *, const char *> kNames[] = {
      {"+", "add2"},          {"-", "sub2"},   {"*", "mul2"},
      {"min", "min2"},        {"max", "max2"}, {"min_nan", "min2_nan"},
      {"max_nan", "max2_nan"}};
  for (const auto &[symbol, name] : kNames) {
    if (op == symbol) {
      return name;
    }
  }
  return nullptr;
}

const char *GetPackedComparisonSuffix(const std::string &op) {
  // C++/TIR != is true for unordered (NaN) operands.
  static constexpr std::pair<const char *, const char *> kSuffixes[] = {
      {"==", "eq"}, {"!=", "neu"}, {"<", "lt"},
      {"<=", "le"}, {">", "gt"},   {">=", "ge"}};
  for (const auto &[symbol, suffix] : kSuffixes) {
    if (op == symbol) {
      return suffix;
    }
  }
  return nullptr;
}

const char *GetPackedMathIntrinsicName(const Call &call) {
  static const std::pair<Op, const char *> kNames[] = {
      {tl::add2(), "add2"},         {tl::sub2(), "sub2"},
      {tl::mul2(), "mul2"},         {tl::fma2(), "fma2"},
      {tl::max2(), "max2"},         {tl::min2(), "min2"},
      {tl::max2_nan(), "max2_nan"}, {tl::min2_nan(), "min2_nan"},
      {tl::abs2(), "abs2"}};
  for (const auto &[intrinsic, name] : kNames) {
    if (call->op.same_as(intrinsic)) {
      return name;
    }
  }
  return nullptr;
}

} // namespace

struct CUDAMath {
  std::string operator()(const PrimType &t, std::string name) const {
    if (t.MatchesCode(kDLFloat)) {
      switch (t.bits()) {
      case 64:
        return name;
      case 32:
        return name + 'f';
      case 16: {
        if (name == "fabs") {
          return "__habs";
        } else if (name == "round") {
          return "hrint";
        } else {
          return "h" + name;
        }
      }
      default:
        return "";
      }
    } else if (t.MatchesElementType(kDLBfloat, 16)) {
      if (name == "fabs") {
        return "__habs";
      } else if (name == "round") {
        return "hrint";
      } else {
        return "h" + name;
      }
    } else if (t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt)) {
      switch (t.bits()) {
      case 32:
        return "__" + name;
      case 64:
        return "__" + name + "ll";
      default:
        return "";
      }
    }
    return "";
  }
};

struct CUDAFastMath : public CUDAMath {
  std::string operator()(const PrimType &t, std::string name) const {
    if (t.MatchesCode(kDLFloat) && t.bits() == 32) {
      return "__" + name + 'f';
    } else {
      return CUDAMath::operator()(t, name);
    }
    return "";
  }
};

struct CUDAFastMathTan : public CUDAMath {
  std::string operator()(const PrimType &t, std::string name) const {
    if (t.MatchesCode(kDLFloat)) {
      switch (t.bits()) {
      case 64:
        return name;
      // `__tanf` seems to produce some values too deviant from numpy tan
      // version. So, let's use just `tanf` instead.
      case 32:
        return name + 'f';
      case 16:
        return 'h' + name;
      default:
        return "";
      }
    }
    return "";
  }
};

struct CUDAIEEEMath {
  std::string operator()(const PrimType &t, std::string name,
                         std::string rounding_mode) const {
    // float32: all ops with all rounding modes are supported.
    //   Pattern: __<name>_<rm>   (e.g., __fadd_rn, __fmul_rz)
    if (t.MatchesCode(kDLFloat) && t.bits() == 32) {
      return "__" + name + "_" + rounding_mode;
    }

    // float64: strip the leading 'f' from <name> and prefix with 'd'.
    //   Pattern: __d<stem>_<rm>  (e.g., fadd -> __dadd_rn)
    if (t.MatchesCode(kDLFloat) && t.bits() == 64) {
      // Special case: fp64 FMA is __fma_rn, not __dmaf_rn or __dfmaf_rn.
      if (name == "fmaf") {
        return "__fma_" + rounding_mode;
      }
      // fp64 has no reciprocal-square-root intrinsic — must be composed.
      if (name == "frsqrt") {
        TVM_FFI_THROW(InternalError)
            << "IEEE frsqrt is not supported for float64 in CUDA. "
            << "Use ieee_fsqrt followed by ieee_frcp as a workaround.";
        return "";
      }
      // fadd -> dadd, fsub -> dsub, fmul -> dmul, frcp -> drcp,
      // fsqrt -> dsqrt, fdiv -> ddiv
      std::string base = name;
      if (!base.empty() && base[0] == 'f') {
        base = base.substr(1);
      }
      return "__d" + base + "_" + rounding_mode;
    }

    // float16/bfloat16: half-precision intrinsics only exist for
    // round-to-nearest-even.
    if (t.MatchesElementType(kDLFloat, 16) ||
        t.MatchesElementType(kDLBfloat, 16)) {
      if (rounding_mode != "rn") {
        TVM_FFI_THROW(InternalError)
            << "IEEE " << name << " with rounding mode '" << rounding_mode
            << "' is not supported for float16/bfloat16 in CUDA. "
            << "Only rounding mode 'rn' is available for half precision.";
        return "";
      }
      // fmaf -> __hfma (not __hmaf)
      if (name == "fmaf") {
        return "__hfma";
      }
      // Unary ops: hsqrt, hrcp, hrsqrt
      if (name == "fsqrt") {
        return "hsqrt";
      }
      if (name == "frcp") {
        return "hrcp";
      }
      if (name == "frsqrt") {
        return "hrsqrt";
      }
      // Binary ops: __hadd_rn, __hsub_rn, __hmul_rn, __hdiv
      if (name == "fdiv") {
        return "__hdiv";
      }
      std::string base = name;
      if (!base.empty() && base[0] == 'f') {
        base = base.substr(1);
      }
      return "__h" + base + "_rn";
    }

    TVM_FFI_THROW(InternalError)
        << "IEEE " << name << " is not supported for dtype " << t;
    return "";
  }
};

static std::string GetTileLangFP8Type(PrimType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.IsScalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "_2";
  } else if (lanes == 4) {
    vec = "_4";
  } else if (lanes == 8) {
    vec = "_8";
  } else if (lanes == 16) {
    vec = "_16";
  } else if (lanes == 32) {
    vec = "_32";
  } else {
    TVM_FFI_THROW(InternalError)
        << "Only support scalar and vector types of width (2, 4, 8, 16, 32) "
           "for FP8";
  }
  if (type.MatchesElementType(kDLFloat8_e4m3, 8) ||
      type.MatchesElementType(kDLFloat8_e4m3fn, 8)) {
    stream << "fp8_e4" << vec << "_t";
  } else if (type.MatchesElementType(kDLFloat8_e5m2, 8)) {
    stream << "fp8_e5" << vec << "_t";
  } else if (type.MatchesElementType(kDLFloat8_e8m0fnu, 8)) {
    stream << "fp8_e8" << vec << "_t";
  } else {
    TVM_FFI_THROW(InternalError)
        << "Unsupported FP8 type in CUDA codegen but got " << type;
  }
  return stream.str();
}

std::string GetTileLangFP6Type(PrimType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.IsScalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "x2";
  } else if (lanes == 4) {
    vec = "x4";
  } else if (lanes == 8) {
    vec = "x8";
  } else if (lanes == 16) {
    vec = "x16";
  } else {
    TVM_FFI_THROW(InternalError)
        << "Only support scalar and vector types of width (2, 4) for FP6";
  }
  stream << "__nv_fp6";
  std::string suffix;
  if (type.code() == kDLFloat6_e2m3fn) {
    suffix = "_e2m3";
  } else if (type.code() == kDLFloat6_e3m2fn) {
    suffix = "_e3m2";
  } else {
    TVM_FFI_THROW(InternalError) << "Unsupported FP6 type in CUDA codegen";
  }
  stream << vec << suffix;
  return stream.str();
}

std::string GetTileLangFP4Type(PrimType type) {
  if (tl::IsFloat4E2M1Unpacked(type)) {
    TVM_FFI_THROW(InternalError)
        << "float4_e2m1_unpacked is a SMEM/TMA storage tag and has no "
           "CUDA register type";
  }

  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.IsScalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "_2";
  } else if (lanes == 4) {
    vec = "_4";
  } else if (lanes == 8) {
    vec = "_8";
  } else if (lanes == 16) {
    vec = "_16";
  } else if (lanes == 32) {
    vec = "_32";
  } else if (lanes == 64) {
    vec = "_64";
  } else {
    TVM_FFI_THROW(InternalError)
        << "Only support scalar and vector types of width (2, 4, 8, 16, "
           "32, 64) for FP4";
  }

  std::string suffix;
  if (type.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
    suffix = "_e2";
  } else {
    TVM_FFI_THROW(InternalError) << "Unsupported FP4 type in CUDA codegen";
  }

  stream << "fp4" << suffix << vec << "_t";
  return stream.str();
}

CodeGenTileLangCUDA::CodeGenTileLangCUDA() {
  restrict_keyword_ = "__restrict__";
}

void CodeGenTileLangCUDA::ReserveKeywordsAsUnique_() {
  CodeGenC::ReserveKeywordsAsUnique();
  name_supply_->ReserveName("max");
  name_supply_->ReserveName("min");
  name_supply_->ReserveName("isfinite");
  name_supply_->ReserveName("isinf");
  name_supply_->ReserveName("isnan");

  // skip single precision mathematical functions
  name_supply_->ReserveName("acosf");
  name_supply_->ReserveName("acoshf");
  name_supply_->ReserveName("asinf");
  name_supply_->ReserveName("asinhf");
  name_supply_->ReserveName("atan2f");
  name_supply_->ReserveName("atanf");
  name_supply_->ReserveName("atanhf");
  name_supply_->ReserveName("cbrtf");
  name_supply_->ReserveName("ceilf");
  name_supply_->ReserveName("copysignf");
  name_supply_->ReserveName("cosf");
  name_supply_->ReserveName("coshf");
  name_supply_->ReserveName("cospif");
  name_supply_->ReserveName("cyl_bessel_i0f");
  name_supply_->ReserveName("cyl_bessel_i1f");
  name_supply_->ReserveName("erfcf");
  name_supply_->ReserveName("erfcinvf");
  name_supply_->ReserveName("erfcxf");
  name_supply_->ReserveName("erff");
  name_supply_->ReserveName("erfinvf");
  name_supply_->ReserveName("exp10f");
  name_supply_->ReserveName("exp2f");
  name_supply_->ReserveName("expf");
  name_supply_->ReserveName("expm1f");
  name_supply_->ReserveName("fabsf");
  name_supply_->ReserveName("fdimf");
  name_supply_->ReserveName("fdividef");
  name_supply_->ReserveName("floorf");
  name_supply_->ReserveName("fmaf");
  name_supply_->ReserveName("fmaxf");
  name_supply_->ReserveName("fminf");
  name_supply_->ReserveName("fmodf");
  name_supply_->ReserveName("frexpf");
  name_supply_->ReserveName("hypotf");
  name_supply_->ReserveName("ilogbf");
  name_supply_->ReserveName("j0f");
  name_supply_->ReserveName("j1f");
  name_supply_->ReserveName("jnf");
  name_supply_->ReserveName("ldexpf");
  name_supply_->ReserveName("lgammaf");
  name_supply_->ReserveName("llrintf");
  name_supply_->ReserveName("llroundf");
  name_supply_->ReserveName("log10f");
  name_supply_->ReserveName("log1pf");
  name_supply_->ReserveName("log2f");
  name_supply_->ReserveName("logbf");
  name_supply_->ReserveName("logf");
  name_supply_->ReserveName("lrintf");
  name_supply_->ReserveName("lroundf");
  name_supply_->ReserveName("modff");
  name_supply_->ReserveName("nanf");
  name_supply_->ReserveName("nearbyintf");
  name_supply_->ReserveName("nextafterf");
  name_supply_->ReserveName("norm3df");
  name_supply_->ReserveName("norm4df");
  name_supply_->ReserveName("normcdff");
  name_supply_->ReserveName("normcdfinvf");
  name_supply_->ReserveName("normf");
  name_supply_->ReserveName("powf");
  name_supply_->ReserveName("rcbrtf");
  name_supply_->ReserveName("remainderf");
  name_supply_->ReserveName("remquof");
  name_supply_->ReserveName("rhypotf");
  name_supply_->ReserveName("rintf");
  name_supply_->ReserveName("rnorm3df");
  name_supply_->ReserveName("rnorm4df");
  name_supply_->ReserveName("rnormf");
  name_supply_->ReserveName("roundf");
  name_supply_->ReserveName("rsqrtf");
  name_supply_->ReserveName("scalblnf");
  name_supply_->ReserveName("scalbnf");
  name_supply_->ReserveName("signbit");
  name_supply_->ReserveName("sincosf");
  name_supply_->ReserveName("sincospif");
  name_supply_->ReserveName("sinf");
  name_supply_->ReserveName("sinhf");
  name_supply_->ReserveName("sinpif");
  name_supply_->ReserveName("sqrtf");
  name_supply_->ReserveName("tanf");
  name_supply_->ReserveName("tanhf");
  name_supply_->ReserveName("tgammaf");
  name_supply_->ReserveName("truncf");
  name_supply_->ReserveName("y0f");
  name_supply_->ReserveName("y1f");
  name_supply_->ReserveName("ynf");

  // skip double precision mathematical functions
  name_supply_->ReserveName("acos");
  name_supply_->ReserveName("acosh");
  name_supply_->ReserveName("asin");
  name_supply_->ReserveName("asinh");
  name_supply_->ReserveName("atan");
  name_supply_->ReserveName("atan2");
  name_supply_->ReserveName("atanh");
  name_supply_->ReserveName("cbrt");
  name_supply_->ReserveName("ceil");
  name_supply_->ReserveName("copysign");
  name_supply_->ReserveName("cos");
  name_supply_->ReserveName("cosh");
  name_supply_->ReserveName("cospi");
  name_supply_->ReserveName("cyl_bessel_i0");
  name_supply_->ReserveName("cyl_bessel_i1");
  name_supply_->ReserveName("erf");
  name_supply_->ReserveName("erfc");
  name_supply_->ReserveName("erfcinv");
  name_supply_->ReserveName("erfcx");
  name_supply_->ReserveName("erfinv");
  name_supply_->ReserveName("exp");
  name_supply_->ReserveName("exp10");
  name_supply_->ReserveName("exp2");
  name_supply_->ReserveName("expm1");
  name_supply_->ReserveName("fabs");
  name_supply_->ReserveName("fdim");
  name_supply_->ReserveName("floor");
  name_supply_->ReserveName("fma");
  name_supply_->ReserveName("fmax");
  name_supply_->ReserveName("fmin");
  name_supply_->ReserveName("fmod");
  name_supply_->ReserveName("frexp");
  name_supply_->ReserveName("hypot");
  name_supply_->ReserveName("ilogb");
  name_supply_->ReserveName("j0");
  name_supply_->ReserveName("j1");
  name_supply_->ReserveName("jn");
  name_supply_->ReserveName("ldexp");
  name_supply_->ReserveName("lgamma");
  name_supply_->ReserveName("llrint");
  name_supply_->ReserveName("llround");
  name_supply_->ReserveName("log");
  name_supply_->ReserveName("log10");
  name_supply_->ReserveName("log1p");
  name_supply_->ReserveName("log2");
  name_supply_->ReserveName("logb");
  name_supply_->ReserveName("lrint");
  name_supply_->ReserveName("lround");
  name_supply_->ReserveName("modf");
  name_supply_->ReserveName("nan");
  name_supply_->ReserveName("nearbyint");
  name_supply_->ReserveName("nextafter");
  name_supply_->ReserveName("norm");
  name_supply_->ReserveName("norm3d");
  name_supply_->ReserveName("norm4d");
  name_supply_->ReserveName("normcdf");
  name_supply_->ReserveName("normcdfinv");
  name_supply_->ReserveName("pow");
  name_supply_->ReserveName("rcbrt");
  name_supply_->ReserveName("remainder");
  name_supply_->ReserveName("remquo");
  name_supply_->ReserveName("rhypot");
  name_supply_->ReserveName("rint");
  name_supply_->ReserveName("rnorm");
  name_supply_->ReserveName("rnorm3d");
  name_supply_->ReserveName("rnorm4d");
  name_supply_->ReserveName("round");
  name_supply_->ReserveName("rsqrt");
  name_supply_->ReserveName("scalbln");
  name_supply_->ReserveName("scalbn");
  name_supply_->ReserveName("signbit");
  name_supply_->ReserveName("sin");
  name_supply_->ReserveName("sincos");
  name_supply_->ReserveName("sincospi");
  name_supply_->ReserveName("sinh");
  name_supply_->ReserveName("sinpi");
  name_supply_->ReserveName("sqrt");
  name_supply_->ReserveName("tan");
  name_supply_->ReserveName("tanh");
  name_supply_->ReserveName("tgamma");
  name_supply_->ReserveName("trunc");
  name_supply_->ReserveName("y0");
  name_supply_->ReserveName("y1");
  name_supply_->ReserveName("yn");
}

void CodeGenTileLangCUDA::PrintFuncPrefix(std::ostream &os) {
  os << "extern \"C\" __global__ ";
}

class LaunchConfigExtractor : public tirx::StmtExprVisitor {
public:
  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj *op) final {
    if (op->op.same_as(tirx::launch_thread_op())) {
      auto tag = op->args[0].as_or_throw<StringImm>()->value;
      PrimExpr extent = op->args[1].as_or_throw<PrimExpr>();
      auto [it, added] = extents.emplace(tag, extent);
      TVM_FFI_CHECK(added || analyzer->CanProveEqual(it->second, extent),
                    ValueError)
          << "Conflicting launch extents for " << tag;
      if (tag == "threadIdx.x")
        threadIdx_x_ext = extent;
      if (tag == "threadIdx.y")
        threadIdx_y_ext = extent;
      if (tag == "threadIdx.z")
        threadIdx_z_ext = extent;
    } else if (op->op->name == "tl.attr") {
      auto attr = DecodeTileLangAttribute(op);
      if (attr.key == "tl.min_blocks_per_sm") {
        min_blocks_per_sm = attr.value.as<IntImmObj>()
                                ? CodegenInt(attr.value.as<IntImmObj>())
                                : attr.value.cast<int64_t>();
        TVM_FFI_CHECK_GT(min_blocks_per_sm, 0, ValueError);
      }
    }
    return StmtExprVisitor::Visit_(op);
  }
  PrimExpr threadIdx_x_ext = IntImm::Int32(1);
  PrimExpr threadIdx_y_ext = IntImm::Int32(1);
  PrimExpr threadIdx_z_ext = IntImm::Int32(1);
  int64_t min_blocks_per_sm = 1;

private:
  sym::Analyzer analyzer;
  std::unordered_map<std::string, PrimExpr> extents;
};

class ClusterInfoExtractor {
public:
  std::optional<std::tuple<int64_t, int64_t, int64_t>>
  extract(const tirx::Function &f) {
    auto dims = f->GetAttr<Array<PrimExpr>>("cluster_dims");
    if (!dims) {
      if (auto native = f->GetAttr<Array<PrimExpr>>("cuda.launch_dimensions")) {
        TVM_FFI_ICHECK_EQ(native.value().size(), 6);
        dims = Array<PrimExpr>{native.value()[3], native.value()[4],
                               native.value()[5]};
      }
    }
    if (!dims)
      return std::nullopt;
    TVM_FFI_CHECK_EQ(dims.value().size(), 3, ValueError);
    auto x = CodegenInt(dims.value()[0]);
    auto y = CodegenInt(dims.value()[1]);
    auto z = CodegenInt(dims.value()[2]);
    TVM_FFI_CHECK(x > 0 && y > 0 && z > 0, ValueError);
    if (x == 1 && y == 1 && z == 1)
      return std::nullopt;
    return std::make_tuple(x, y, z);
  }
};

void CodeGenTileLangCUDA::PrintExtraAttrs(const tirx::Function &f,
                                          std::ostream &os) {
  auto extractor = ffi::make_object<LaunchConfigExtractor>();
  extractor->Visit(f->body);
  if (auto dims = f->GetAttr<Array<PrimExpr>>("cuda.launch_dimensions")) {
    TVM_FFI_ICHECK_EQ(dims.value().size(), 6);
    extractor->threadIdx_x_ext = dims.value()[0];
    extractor->threadIdx_y_ext = dims.value()[1];
    extractor->threadIdx_z_ext = dims.value()[2];
  }
  if (auto attrs = f->GetAttr<Map<String, int64_t>>("cuda.kernel_attrs")) {
    extractor->min_blocks_per_sm = attrs.value()
                                       .Get("min_blocks_per_sm")
                                       .value_or(extractor->min_blocks_per_sm);
  }
  sym::Analyzer analyzer;
  PrimExpr threads = analyzer->Simplify(extractor->threadIdx_x_ext *
                                        extractor->threadIdx_y_ext *
                                        extractor->threadIdx_z_ext);
  if (const auto *count = threads.as<IntImmObj>(); count && count->value > 1) {
    os << " __launch_bounds__(" << count->value << ", "
       << extractor->min_blocks_per_sm << ")";
  }
}

void CodeGenTileLangCUDA::EmitPreamble_(std::ostream &os) const {
  // CUDA 13's cuda.h declares CUtensorMap with `alignas(128)`, which MSVC
  // rejects as a function parameter alignment (C2719: "formal parameter
  // requires <N>-byte alignment"). MSVC's hard limit for parameter alignment
  // is 64. NVCC still feeds the kernel signature through MSVC's host
  // frontend on Windows, so any kernel taking `__grid_constant__ const
  // CUtensorMap` by value fails to compile.
  //
  // Older MSVC frontends reject such a parameter, so cap alignas(N) at 64 just
  // for the duration of <cuda.h>'s parsing there. Newer MSVC versions accept
  // the 128-byte aligned parameter, and keeping that alignment avoids
  // CUDA_ERROR_MISALIGNED_ADDRESS at launch time.
  os << "#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940\n";
  os << "#define _tl_orig_alignas alignas\n";
  os << "#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)\n";
  os << "#include <cuda.h>\n";
  os << "#undef alignas\n";
  os << "#define alignas _tl_orig_alignas\n";
  os << "#endif\n";

  if (need_mma_h_) {
    os << "#include <mma.h>\n";
  }
  if (need_mma_instruction_h_) {
    os << "#include <tl_templates/cuda/instruction/mma.h>\n";
  }
  if (need_mma_block_scale_instruction_h_) {
    os << "#include <tl_templates/cuda/instruction/mma_block_scale.h>\n";
  }
  if (need_wgmma_instruction_h_) {
    os << "#include <tl_templates/cuda/instruction/wgmma.h>\n";
  }
  if (need_wgmma_sp_instruction_h_) {
    os << "#include <tl_templates/cuda/instruction/wgmma_sp.h>\n";
  }
  if (need_tcgen05mma_instruction_h_) {
    os << "#include <tl_templates/cuda/instruction/tcgen05mma.h>\n";
  }
  if (need_mma_sm70_instruction_h_) {
    os << "#include <tl_templates/cuda/instruction/mma_sm70.h>\n";
  }
  if (need_mma_sp_instruction_h_) {
    os << "#include <tl_templates/cuda/instruction/mma_sp.h>\n";
  }
  if (need_tcgen05_common_h_) {
    os << "#include <tl_templates/cuda/tcgen_05.h>\n";
  }
  if (need_intrin_h_) {
    os << "#include <tl_templates/cuda/intrin.h>\n";
  }
  if (need_atomic_h_) {
    os << "#include <tl_templates/cuda/atomic.h>\n";
  }
  if (need_math_h_) {
    os << "#include <tl_templates/cuda/math.h>\n";
  }
  if (need_barrier_h_) {
    os << "#include <tl_templates/cuda/barrier.h>\n";
  }
  if (need_copy_h_) {
    os << "#include <tl_templates/cuda/copy.h>\n";
  }
  if (need_copy_sm90_h_ || need_copy_sm100_h_) {
    os << "#include <tl_templates/cuda/copy_sm90.h>\n";
  }
  if (need_copy_sm100_h_) {
    os << "#include <tl_templates/cuda/copy_sm100.h>\n";
  }
  if (enable_fp8_) {
    os << "#include <tl_templates/cuda/cuda_fp8.h>\n";
  }
  if (enable_fp6_) {
    os << "#include <cuda_fp6.h>\n";
  }
  if (enable_fp4_) {
    os << "#include <tl_templates/cuda/cuda_fp4.h>\n";
  }

  if (need_math_constants_h_) {
    os << "#include <math_constants.h>\n";
  }

  if (need_cooperative_groups_) {
    os << "#include <cooperative_groups.h>\n";
  }

  if (need_cluster_h_) {
    os << "#include <tl_templates/cuda/cluster.h>\n";
  }

  if (need_curand_kernel_h_) {
    os << "#include <curand_kernel.h>\n";
  }

  os << "#include <tl_templates/cuda/reduce.h>\n";
  os << "#include <tl_templates/cuda/scan.h>\n";
  os << "#include <tl_templates/cuda/ldsm.h>\n";
  os << "#include <tl_templates/cuda/threadblock_swizzle.h>\n";
  os << "#include <tl_templates/cuda/debug.h>\n";
  os << "#ifdef ENABLE_BF16\n";
  os << "#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>\n";
  os << "#endif\n";
  os << "\n";
}

void CodeGenTileLangCUDA::Dispatch_(const ForObj *op) {
  if (op->kind == ForKind::kUnrolled) {
    PrintIndent();
    stream << "#pragma unroll";
    // Native loop annotations remain on For. A loop-local hint takes
    // precedence over the surrounding attribute carrier, if both exist.
    if (auto factor = op->annotations.Get("pragma_unroll_factor")) {
      const auto *literal = factor->as<IntImmObj>();
      stream << " "
             << (literal ? CodegenInt(literal) : factor->cast<int64_t>());
    } else if (auto it = unroll_factor.find(op->var.get());
               it != unroll_factor.end()) {
      stream << " " << PrintExpr(it->second);
    }
    stream << "\n";
  }
  For loop = GetRef<For>(op);
  auto *node = loop.CopyOnWrite();
  node->body = UnwrapRedundantLexicalScope(node->body);
  sym::Analyzer analyzer;
  node->begin = analyzer->Simplify(node->begin);
  node->end = analyzer->Simplify(node->end);
  node->step = analyzer->Simplify(node->step);
  CodeGenC::Dispatch_(loop.get());
}

void CodeGenTileLangCUDA::BindThreadIndex(const PrimVar &var,
                                          const ffi::String &thread_tag) {
  ICHECK(!var_idmap_.count(var.get()));
  var_idmap_[var.get()] = CastFromTo(thread_tag, PrimType::UInt(32), var.ty());
}

void CodeGenTileLangCUDA::PrintType(const Type &t, std::ostream &os) {
  // This tag describes one byte of shared storage, not a C arithmetic type.
  // Retain it in the IR while using the same byte pointer as its allocation.
  if (IsUnpackedFP4SharedPointer(t)) {
    os << "uint8_t*";
    return;
  }
  CodeGenTileLangCBase::PrintType(t, os);
}

void CodeGenTileLangCUDA::PrintType(const PrimType &t,
                                    std::ostream &os) { // NOLINT(*)
  if (t.IsVoid()) {
    os << "void";
    return;
  }
  int lanes = t.lanes();
  if (t.MatchesCode(kDLOpaqueHandle)) {
    ICHECK(t.IsScalar()) << "do not yet support vector types";
    os << "void*";
    return;
  }

  bool fail = false;
  if (t.MatchesCode(kDLFloat)) {
    switch (t.bits()) {
    case 16:
      enable_fp16_ = true;
      if (t.IsScalar()) {
        os << "half_t";
      } else if (lanes <= 8) {
        // Emit CUDA code to access fp16 vector elements.
        //
        // half4 is stored as uint2
        //
        // h4.x is emitted as *(half2*)(&(u2.x)).x
        // h4.y is emitted as *(half2*)(&(u2.x)).y
        // h4.z is emitted as *(half2*)(&(u2.y)).x
        // h4.w is emitted as *(half2*)(&(u2.y)).y
        //
        ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
        os << "uint" << lanes / 2;
      } else if (lanes <= 16) {
        ICHECK_EQ(lanes % 4, 0) << "only support (mod 4 = 0) lanes for half "
                                   "type of more than 8 lanes";
        os << "ulonglong" << lanes / 4;
      } else {
        fail = true;
      }
      break;
    case 32:
      if (lanes <= 4) {
        os << "float";
      } else if (lanes <= 8) {
        // Emit CUDA code to access fp32 vector elements for 4 < lanes <= 8.
        //
        // float8 is stored as ulonglong4
        //
        // f8.v1 is emitted as *(float2*)(&(ul4.x)).x
        // f8.v2 is emitted as *(float2*)(&(ul4.x)).y
        //
        ICHECK_EQ(lanes % 2, 0)
            << "only support even lane for float type with lanes > 4";
        os << "ulonglong" << lanes / 2;
      } else {
        fail = true;
      }
      break;
    case 64:
      os << "double";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && (t.IsScalar() || t.bits() == 16))
      return;
    if (!fail && (lanes > 4 && lanes <= 8 && t.bits() == 32))
      return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.MatchesElementType(kDLBfloat, 16)) {
    enable_bf16_ = true;
    if (t.IsScalar()) {
      os << "bfloat16_t";
    } else if (lanes <= 8) {
      ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
      os << "uint" << lanes / 2;
    } else if (lanes <= 16) {
      ICHECK_EQ(lanes % 4, 0) << "only support (mod 4 = 0) lanes for half type "
                                 "of more than 8 lanes";
      os << "ulonglong" << lanes / 4;
    } else {
      fail = true;
    }
    if (!fail)
      return;
  } else if (CodegenIsFloat8(t)) {
    enable_fp8_ = true;
    os << GetTileLangFP8Type(t);
    return;
  } else if (CodegenIsFloat6(t)) {
    enable_fp6_ = true;
    if (t.lanes() <= 4) {
      os << GetTileLangFP6Type(t);
      return;
    }
    fail = true;
  } else if (tl::IsFloat4E2M1Unpacked(t)) {
    TVM_FFI_THROW(InternalError)
        << "float4_e2m1_unpacked is a SMEM/TMA storage tag and must be "
           "lowered through shared-memory allocation";
    return;
  } else if (t.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
    enable_fp4_ = true;
    if (t.lanes() <= 64) {
      os << GetTileLangFP4Type(t);
    } else {
      fail = true;
    }
    return;
  } else if (tl::IsTensorFloat32(t)) {
    if (t.IsScalar()) {
      os << "tfloat32_t";
    } else if (lanes <= 4) {
      os << "float" << lanes;
    } else if (lanes <= 8) {
      ICHECK_EQ(lanes % 2, 0)
          << "only support even lane for tfloat32 type with lanes > 4";
      os << "ulonglong" << lanes / 2;
    } else {
      fail = true;
    }
    if (!fail)
      return;
  } else if (t == PrimType::Bool()) {
    os << "bool";
    return;
  } else if (CodegenIsVectorBool(t)) {
    // CUDA does not support bool vectors.
    // Use ushort vectors to represent instead.
    int n = t.lanes();
    if (n <= 4) {
      os << "ushort" << n;
      return;
    }
  } else if (t.MatchesCode(kDLUInt) || t.MatchesCode(kDLInt)) {
    if (t.MatchesCode(kDLUInt)) {
      os << "u";
    }
    switch (t.bits()) {
    case 1: {
      if (t.IsScalar()) {
        os << "int";
        return;
      } else if (t.lanes() == 8) {
        os << "int8_t";
        return;
      } else if (t.lanes() == 16) {
        os << "int16_t";
        return;
      } else if (t.lanes() == 32) {
        os << "int";
        return;
      } else {
        TVM_FFI_THROW(InternalError)
            << "Cannot convert type " << t << " to CUDA type!";
      }
    }
    case 4: {
      if (t.IsScalar()) {
        enable_int8_ = true;
        if (!t.MatchesCode(kDLUInt)) {
          os << "signed char";
        } else {
          os << "char";
        }
        return;
      } else if (t.lanes() == 2) {
        // Two packed 4-bit lanes occupy exactly one byte.
        enable_int8_ = true;
        os << "int8_t";
        return;
      } else if (t.lanes() == 4) {
        os << "int16_t";
        return;
      } else if (t.lanes() == 8) {
        // directly 8 4-bit int in integer.
        os << "int";
        return;
      } else if (t.lanes() == 16) {
        os << "int2";
        return;
      } else if (t.lanes() == 32) {
        os << "int4";
        return;
      } else if (t.lanes() == 64) {
        // 256 bits; CUDA has no (u)int8 vector type, use (u)longlong4 like
        // int8x32.
        os << "longlong4";
        return;
      } else {
        TVM_FFI_THROW(InternalError)
            << "Cannot convert type " << t << " to CUDA type!";
      }
    }
    case 8: {
      if (t.lanes() == 4) {
        // directly 4 8 bit int in integer.
        enable_int8_ = true;

        // We use int for int8x4 instead of char4 because using char4 is
        // likely to produce extra instructions to pack four int8 elements
        // into 32-bit data.
        os << "int";
        return;
      } else if (t.lanes() == 8) {
        enable_int8_ = true;
        os << "int2";
        return;
      } else if (t.lanes() == 16) {
        enable_int8_ = true;
        os << "int4";
        return;
      } else if (t.lanes() == 32) {
        enable_int8_ = true;
        os << "longlong4";
        return;
      } else if (!t.MatchesCode(kDLUInt) && t.IsScalar()) {
        os << "signed char";
        break;
      } else {
        os << "char";
        break;
      }
    }
    case 16: {
      if (t.IsScalar()) {
        os << "short";
      } else if (t.lanes() <= 4) {
        os << "short" << lanes;
      } else if (t.lanes() <= 8) {
        // Emit CUDA code to access int16 vector elements.
        //
        // short4 is stored as int2
        //
        // s4.x is emitted as *(short2*)(&(i2.x)).x
        // s4.y is emitted as *(short2*)(&(i2.x)).y
        // s4.z is emitted as *(short2*)(&(i2.y)).x
        // s4.w is emitted as *(short2*)(&(i2.y)).y
        //
        ICHECK_EQ(t.lanes() % 2, 0)
            << "only support even lane for shorT type with lanes > 4";
        os << "int" << t.lanes() / 2;
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 32: {
      if (t.IsScalar()) {
        os << "int";
      } else if (t.lanes() <= 4) {
        os << "int" << t.lanes();
      } else if (t.lanes() <= 8) {
        // Emit CUDA code to access int32 vector elements for 4 < lanes <= 8.
        //
        // int8 is stored as longlong4
        //
        // i8.v1 is emitted as *(int2*)(&(l4.x)).x
        // i8.v2 is emitted as *(int2*)(&(l4.x)).y
        //
        ICHECK_EQ(lanes % 2, 0)
            << "only support even lane for int32 type with lanes > 4";
        os << "longlong" << lanes / 2;
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 64: {
      if (t.IsScalar()) {
        os << "int64_t";
      } else if (t.lanes() == 2) {
        os << "longlong2";
      } else if (t.lanes() == 3) {
        os << "longlong3";
      } else if (t.lanes() == 4) {
        os << "longlong4";
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1) {
      return;
    }
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  }
  TVM_FFI_THROW(InternalError)
      << "Cannot convert type " << t << " to CUDA type";
}

void CodeGenTileLangCUDA::PrintVecConstructor(const PrimType &t,
                                              std::ostream &os) { // NOLINT(*)
  if ((t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt)) && t.bits() == 4 &&
      t.lanes() == 2) {
    os << (t.MatchesCode(kDLUInt) ? "tl_pack_uint4x2" : "tl_pack_int4x2");
    return;
  }
  // fp4/fp8 vector structs have no per-lane-count constructor; a variadic
  // packer builds them from one scalar element per lane.
  if (t.MatchesElementType(kDLFloat4_e2m1fn, 4) && t.lanes() >= 2) {
    os << "tl::make_fp4_vec<";
    PrintType(t, os);
    os << ">";
    return;
  }
  if (CodegenIsFloat8(t) && t.lanes() >= 2) {
    os << "tl::make_vec<";
    PrintType(t, os);
    os << ">";
    return;
  }
  // CUDA's built-in vector types are aggregates, not C++ constructors.  The
  // shared native C emitter only prints the type name here.
  os << "make_";
  PrintType(t, os);
}

void CodeGenTileLangCUDA::EmitPackedX2Call(const std::string &tl_func,
                                           const PrimType &t,
                                           const std::vector<PrimExpr> &args,
                                           std::ostream &os) {
  // Decompose into lanes/2 independent x2 packed operations.
  //
  // Vector type → CUDA struct mapping:
  //   bf16/fp16 x2..x8  -> uint1..uint4  (one packed x2 pair per field)
  //   bf16/fp16 x12/x16 -> ulonglong3/4 (two packed x2 pairs per field)
  //   f32x2  -> float2 {.x, .y}
  //   f32x4  -> float4 {.x,.y,.z,.w}
  //   f32x6/x8 -> ulonglong3/4 (one float2 pair per field)
  //
  // For bf16/fp16: each 32-bit field already packs a pair of elements,
  //   so we apply tl::*2 on each field directly for <= 8 lanes. For
  //   12/16 lanes, each 64-bit field stores two x2 pairs.
  // For f32: float4 stores pairs at {x,z}; ulonglong3/4 stores one
  //   float2 pair per field at {x,y,z,w}.
  int lanes = t.lanes();
  bool is_bf16x2 = t.MatchesElementType(kDLBfloat, 16);
  bool is_fp16x2 = t.MatchesElementType(kDLFloat, 16);
  int num_pairs = lanes / 2;
  static const char access[] = {'x', 'y', 'z', 'w'};

  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(t, stream);
  stream << ' ' << sret << ";\n";
  int ssa_scope = BeginScope();
  {
    std::vector<std::string> packed_vecs;
    packed_vecs.reserve(args.size());
    for (const PrimExpr &arg : args) {
      packed_vecs.push_back(
          SSAGetID(PrintExpr(arg), arg.ty(),
                   tvm::SideEffect(arg) == CallEffectKind::kPure));
    }

    if (is_bf16x2 || is_fp16x2) {
      std::string native_type = is_bf16x2 ? "__nv_bfloat162" : "__half2";
      auto make_half_pair = [&](const std::string &vec_name,
                                const std::string &field, int pair_offset) {
        std::string pair = "tl::from_uint1<";
        pair += native_type;
        pair += ">(";
        if (lanes <= 8) {
          pair += "*(uint1*)(&(";
          pair += vec_name;
          pair += ".";
          pair += field;
          pair += "))";
        } else {
          pair += "*(((uint1*)(&(";
          pair += vec_name;
          pair += ".";
          pair += field;
          pair += "))) + ";
          pair += std::to_string(pair_offset);
          pair += ")";
        }
        pair += ")";
        return pair;
      };
      for (int p = 0; p < num_pairs; ++p) {
        int field_idx = lanes <= 8 ? p : (p / 2);
        ICHECK_LT(field_idx, 4);
        int pair_offset = lanes <= 8 ? 0 : (p % 2);
        std::string field(1, access[field_idx]);
        std::vector<std::string> pair_args;
        pair_args.reserve(packed_vecs.size());
        for (const auto &vec_name : packed_vecs) {
          pair_args.push_back(make_half_pair(vec_name, field, pair_offset));
        }
        this->PrintIndent();
        if (lanes <= 8) {
          stream << "*(uint1*)(&(" << sret << "." << field
                 << ")) = tl::to_uint1(tl::" << tl_func << "(";
        } else {
          stream << "*(((uint1*)(&(" << sret << "." << field << "))) + "
                 << pair_offset << ") = tl::to_uint1(tl::" << tl_func << "(";
        }
        stream << pair_args[0];
        for (size_t i = 1; i < pair_args.size(); ++i) {
          stream << ", " << pair_args[i];
        }
        stream << "));\n";
      }
    } else {
      // f32: apply tl::*2 on each consecutive pair of float fields,
      // reinterpreted as float2.
      auto make_float_pair = [&](const std::string &vec_name,
                                 const std::string &field) {
        return "*(float2*)(&(" + vec_name + "." + field + "))";
      };
      for (int p = 0; p < num_pairs; ++p) {
        int field_idx = lanes <= 4 ? (p * 2) : p;
        ICHECK_LT(field_idx, 4);
        std::string field(1, access[field_idx]);
        std::vector<std::string> pair_args;
        pair_args.reserve(packed_vecs.size());
        for (const auto &vec_name : packed_vecs) {
          pair_args.push_back(make_float_pair(vec_name, field));
        }
        this->PrintIndent();
        stream << "*(float2*)(&(" << sret << "." << field
               << ")) = tl::" << tl_func << "(" << pair_args[0];
        for (size_t i = 1; i < pair_args.size(); ++i) {
          stream << ", " << pair_args[i];
        }
        stream << ");\n";
      }
    }
  }
  EndScope(ssa_scope);
  os << sret;
}

void CodeGenTileLangCUDA::EmitPerLaneScalarCall(
    const std::string &func_name, const PrimType &t,
    const std::vector<PrimExpr> &args, std::ostream &os) {
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(t, stream);
  stream << ' ' << sret << ";\n";
  int ssa_scope = BeginScope();
  {
    std::vector<std::string> vec_ids;
    vec_ids.reserve(args.size());
    for (const PrimExpr &arg : args) {
      vec_ids.push_back(
          SSAGetID(PrintExpr(arg), arg.ty(),
                   tvm::SideEffect(arg) == CallEffectKind::kPure));
    }
    for (int i = 0; i < t.lanes(); ++i) {
      std::ostringstream value_temp;
      value_temp << func_name << "(";
      for (size_t j = 0; j < vec_ids.size(); ++j) {
        if (j != 0) {
          value_temp << ", ";
        }
        PrintVecElemLoad(vec_ids[j], args[j]->ty.as_or_throw<PrimType>(), i,
                         value_temp);
      }
      value_temp << ")";
      PrintVecElemStore(sret, t, i, value_temp.str());
    }
  }
  EndScope(ssa_scope);
  os << sret;
}

void CodeGenTileLangCUDA::PrintVecBinaryOp(const std::string &op,
                                           const PrimType &t, PrimExpr lhs,
                                           PrimExpr rhs,
                                           std::ostream &os) { // NOLINT(*)
  if (TryEmitPackedBinaryOp_(op, t, lhs, rhs, os)) {
    return;
  }

  // Declare the result.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(t, stream);
  stream << ' ' << sret << ";\n";
  int ssa_scope = BeginScope();
  {
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.ty(),
                                tvm::SideEffect(lhs) == CallEffectKind::kPure);
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.ty(),
                                tvm::SideEffect(rhs) == CallEffectKind::kPure);
    if (!TryEmitPackedComparison_(op, t, lhs.ty(), rhs.ty(), vlhs, vrhs,
                                  sret)) {
      EmitScalarizedBinaryOp_(op, t, lhs.ty(), rhs.ty(), vlhs, vrhs, sret);
    }
  }
  EndScope(ssa_scope);
  os << sret;
}

bool CodeGenTileLangCUDA::TryEmitPackedBinaryOp_(const std::string &op,
                                                 PrimType result_type,
                                                 const PrimExpr &lhs,
                                                 const PrimExpr &rhs,
                                                 std::ostream &os) {
  // Fast-path for packed x2 arithmetic (float32x2, bfloat16x2, float16x2).
  //
  // For float32x2: PTX `.f32x2` instructions are available on SM100+.
  // For bfloat16x2 / float16x2: native packed half-precision instructions
  // (__hadd2, __hsub2, etc.) are available on SM80+ (bf16) / SM53+ (fp16).
  // The tl::*2 C++ helpers have compile-time arch guards with scalar
  // fallbacks, so we can emit them unconditionally for 16-bit types.
  //
  // When lanes > 2 and is even, we decompose the vector operation into
  // lanes/2 independent x2 packed operations on consecutive pairs.
  if (!CanEmitPackedX2Math(result_type)) {
    return false;
  }
  const char *packed_name = GetPackedBinaryOpName(op);
  if (packed_name == nullptr) {
    return false;
  }
  std::string tl_func = packed_name;
  std::vector<PrimExpr> packed_args{lhs, rhs};
  if (op == "+") {
    // Fuse packed mul+add here instead of relying on NVCC to recover
    // packed FMA from tl::mul2/tl::add2 (or the underlying __fmul2 /
    // __fadd2-style helpers). Once the pairwise ops are emitted as
    // separate calls, NVCC does not reliably contract them back to fma2.
    auto try_fuse_mul_add = [&](const PrimExpr &maybe_mul,
                                const PrimExpr &addend) -> bool {
      const prim::MulObj *mul = maybe_mul.as<prim::MulObj>();
      if (mul == nullptr || mul->ty.as_or_throw<PrimType>() != result_type ||
          mul->a.ty() != result_type || mul->b.ty() != result_type ||
          addend.ty() != result_type) {
        return false;
      }
      tl_func = "fma2";
      packed_args = {mul->a, mul->b, addend};
      return true;
    };
    if (!try_fuse_mul_add(lhs, rhs)) {
      try_fuse_mul_add(rhs, lhs);
    }
  }
  EmitPackedX2Call(tl_func, result_type, packed_args, os);
  return true;
}

bool CodeGenTileLangCUDA::TryEmitPackedComparison_(
    const std::string &op, PrimType result_type, PrimType lhs_type,
    PrimType rhs_type, const std::string &lhs, const std::string &rhs,
    const std::string &result) {
  // Comparisons return bool vectors, not the input floating-point dtype.
  // Keep their packed path separate from CanEmitPackedX2Math(result_type).
  // CUDA bool carriers currently support at most four lanes.
  const char *comparison = GetPackedComparisonSuffix(op);
  int lanes = result_type.lanes();
  if (comparison == nullptr || !CodegenIsVectorBool(result_type) ||
      (lanes != 2 && lanes != 4) || lhs_type != rhs_type ||
      lhs_type.lanes() != lanes ||
      !(lhs_type.MatchesElementType(kDLFloat, 16) ||
        lhs_type.MatchesElementType(kDLBfloat, 16))) {
    return false;
  }

  // The mask intrinsics arrived in CUDA 12. BF16 packed set requires SM90;
  // retain the scalar path for older toolkits/architectures without PTX.
  stream << "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= "
         << (lhs_type.MatchesElementType(kDLBfloat, 16) ? 900 : 530)
         << ") && defined(CUDART_VERSION) && (CUDART_VERSION >= 12000)\n";
  std::string native_type =
      lhs_type.MatchesElementType(kDLBfloat, 16) ? "__nv_bfloat162" : "__half2";
  for (int pair = 0; pair < lanes / 2; ++pair) {
    std::string field = pair == 0 ? ".x" : ".y";
    std::string mask = name_supply_->FreshName("cmp_mask");
    PrintIndent();
    stream << "uint " << mask << " = __h" << comparison << "2_mask("
           << "tl::from_uint1<" << native_type << ">(make_uint1(" << lhs
           << field << ")), tl::from_uint1<" << native_type << ">(make_uint1("
           << rhs << field << ")));\n";
    // Each true mask lane is 0xffff. Normalize to 0/1 so casts and boolean
    // arithmetic retain the same semantics as the scalar comparison path.
    PrintVecElemStore(result, result_type, pair * 2, "(" + mask + " & 1u)");
    PrintVecElemStore(result, result_type, pair * 2 + 1,
                      "((" + mask + " >> 16) & 1u)");
  }
  stream << "#else\n";
  EmitScalarizedBinaryOp_(op, result_type, lhs_type, rhs_type, lhs, rhs,
                          result);
  stream << "#endif\n";
  return true;
}

void CodeGenTileLangCUDA::EmitScalarizedBinaryOp_(
    const std::string &op, PrimType result_type, PrimType lhs_type,
    PrimType rhs_type, const std::string &lhs, const std::string &rhs,
    const std::string &result) {
  for (int lane = 0; lane < result_type.lanes(); ++lane) {
    std::ostringstream value;
    if (isalpha(op[0])) {
      value << op << "(";
      PrintVecElemLoad(lhs, lhs_type, lane, value);
      value << ", ";
      PrintVecElemLoad(rhs, rhs_type, lane, value);
      value << ")";
    } else {
      value << "(";
      PrintVecElemLoad(lhs, lhs_type, lane, value);
      value << op;
      PrintVecElemLoad(rhs, rhs_type, lane, value);
      value << ")";
    }
    PrintVecElemStore(result, result_type, lane, value.str());
  }
}

void CodeGenTileLangCUDA::PrintVecElemLoad(const std::string &vec,
                                           const PrimType &t, int i,
                                           std::ostream &os) { // NOLINT(*)
  if (t.IsScalar()) {
    os << vec;
    return;
  }

  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < 256 / t.bits())
      << "i: " << i << " t: " << t << " t.bits(): " << t.bits()
      << " t.lanes(): " << t.lanes();
  bool is_packed_int4 =
      t.bits() == 4 && (t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt));
  if (is_packed_int4) {
    std::ostringstream packed_byte;
    packed_byte << "((const unsigned char*)(&(" << vec << ")))[" << i / 2
                << "]";
    int shift = i % 2 * 4;
    if (t.MatchesCode(kDLUInt)) {
      os << "((static_cast<unsigned int>(" << packed_byte.str() << ") >> "
         << shift << ") & 0x0fu)";
    } else {
      os << "((static_cast<int>((static_cast<unsigned int>("
         << packed_byte.str() << ") >> " << shift << ") & 0x0fu) ^ 8) - 8)";
    }
  } else if (t.bits() == 8 &&
             (t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt))) {
    std::string type_name = t.MatchesCode(kDLInt) ? "char" : "unsigned char";
    if (t.lanes() == 2 || t.lanes() == 3) {
      os << vec << "." << access[i % t.lanes()];
    } else if (t.lanes() <= 16) {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      os << "((" << type_name << ")(" << ac << " >> " << i % 4 * 8 << "))";
    } else {
      ICHECK(t.lanes() == 32);
      std::string ac = vec + "." + access[i / 8];
      os << "((" << type_name << ")(" << ac << " >> " << i % 8 * 8 << "))";
    }
  } else if (t.MatchesElementType(kDLFloat, 16)) {
    if (t.lanes() <= 8) {
      os << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->"
         << access[i % 2];
    } else {
      os << "(((half2*)(&(" << vec << "." << access[i / 4] << "))) + "
         << (i / 2 % 2) << ")->" << access[i % 2];
    }
  } else if (t.MatchesElementType(kDLBfloat, 16)) {
    if (t.lanes() <= 8) {
      os << "((nv_bfloat162*)(&(" << vec << "." << access[i / 2] << ")))->"
         << access[i % 2];
    } else {
      os << "(((nv_bfloat162*)(&(" << vec << "." << access[i / 4] << "))) + "
         << (i / 2 % 2) << ")->" << access[i % 2];
    }
  } else if (CodegenIsFloat8(t)) {
    os << vec;
    // fp8_e5_32_t
    if (t.lanes() >= 32)
      os << "." << access[i / 16];
    // fp8_e5_16_t
    if (t.lanes() >= 16)
      os << "." << access[(i % 16) / 8];
    // fp8_e5_8_t
    if (t.lanes() >= 8)
      os << "." << access[(i % 8) / 4];
    // fp8_e5_4_t or fp8_e5_2_t
    os << "." << access[i % 4];
  } else if (t.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
    os << vec;
    // fp4_e2_64_t
    if (t.lanes() >= 64)
      os << "." << access[i / 32];
    // fp4_e2_32_t
    if (t.lanes() >= 32)
      os << "." << access[(i % 32) / 16];
    // fp4_e2_16_t
    if (t.lanes() >= 16)
      os << "." << access[(i % 16) / 8];
    // fp4_e2_8_t
    if (t.lanes() >= 8)
      os << "." << access[(i % 8) / 4];
    // fp4_e2_4_t -> fp4_e2_2_t member
    if (t.lanes() >= 4)
      os << "." << access[(i % 4) / 2];
    // fp4_e2_2_t -> method call x() or y()
    os << "." << access[i % 2] << "()";
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.MatchesCode(kDLInt)) {
        type_name = "short";
      } else if (t.MatchesCode(kDLUInt)) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.MatchesCode(kDLInt)) {
        type_name = "int";
      } else if (t.MatchesCode(kDLUInt)) {
        type_name = "uint";
      } else if (t.MatchesCode(kDLFloat)) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    os << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2]
       << ")))->" << access[i % 2];
  } else {
    os << vec << "." << access[i];
  }
}

void CodeGenTileLangCUDA::PrintVecElemStore(const std::string &vec,
                                            const PrimType &t, int i,
                                            const std::string &value) {
  this->PrintIndent();
  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < 256 / t.bits());
  bool is_packed_int4 =
      t.bits() == 4 && (t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt));
  if (is_packed_int4) {
    std::ostringstream packed_byte;
    packed_byte << "((unsigned char*)(&(" << vec << ")))[" << i / 2 << "]";
    stream << packed_byte.str() << " = ";
    // Packed vector temporaries are initialized in ascending lane order.
    // Each even lane starts a new byte; each odd lane preserves the low
    // nibble written immediately before it.
    if (i % 2 != 0) {
      stream << "(" << packed_byte.str() << " & 0x0fu) | ";
    }
    stream << "((static_cast<unsigned int>(" << value << ") & 0x0fu) << "
           << i % 2 * 4 << ");\n";
  } else if (t.bits() == 8 &&
             (t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt))) {
    if (t.lanes() == 2 || t.lanes() == 3) {
      stream << vec << '.' << access[i % t.lanes()] << "=" << "(" << value
             << ");\n";
    } else if (t.lanes() <= 16) {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      stream << ac << "=";
      // Do not read the first undef lane.
      if (i % 4 != 0) {
        stream << ac << " & ~(0x000000ff << " << i % 4 * 8 << ") |";
      }
      stream << "(" << value << " << " << i % 4 * 8 << ");\n";
    } else {
      ICHECK(t.lanes() == 32);
      std::string ac = vec + "." + access[i / 8];
      stream << ac << "=";
      // Do not read the first undef lane.
      if (i % 8 != 0) {
        stream << ac << " & ~(static_cast<unsigned long long>(0x000000ffu) << "
               << i % 8 * 8 << ") |";
      }
      stream << "((static_cast<unsigned long long>(" << value
             << ") & 0xffULL) << " << i % 8 * 8 << ");\n";
    }
  } else if (t.MatchesElementType(kDLFloat, 16)) {
    if (t.lanes() <= 8) {
      stream << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->"
             << access[i % 2] << " = " << value << ";\n";
    } else {
      stream << "(((half2*)(&(" << vec << "." << access[i / 4] << "))) + "
             << (i / 2 % 2) << ")->" << access[i % 2] << " = " << value
             << ";\n";
    }
  } else if (t.MatchesElementType(kDLBfloat, 16)) {
    if (t.lanes() <= 8) {
      stream << "((nv_bfloat162*)(&(" << vec << "." << access[i / 2] << ")))->"
             << access[i % 2] << " = " << value << ";\n";
    } else {
      stream << "(((nv_bfloat162*)(&(" << vec << "." << access[i / 4]
             << "))) + " << (i / 2 % 2) << ")->" << access[i % 2] << " = "
             << value << ";\n";
    }
  } else if (CodegenIsFloat8(t)) {
    stream << vec;
    // fp8_e5_32_t
    if (t.lanes() >= 32)
      stream << "." << access[i / 16];
    // fp8_e5_16_t
    if (t.lanes() >= 16)
      stream << "." << access[(i % 16) / 8];
    // fp8_e5_8_t
    if (t.lanes() >= 8)
      stream << "." << access[(i % 8) / 4];
    // fp8_e5_4_t or fp8_e5_2_t
    stream << "." << access[i % 4] << " = " << value << ";\n";
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.MatchesCode(kDLInt)) {
        type_name = "short";
      } else if (t.MatchesCode(kDLUInt)) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.MatchesCode(kDLInt)) {
        type_name = "int";
      } else if (t.MatchesCode(kDLUInt)) {
        type_name = "uint";
      } else if (t.MatchesCode(kDLFloat)) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    stream << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2]
           << ")))->" << access[i % 2] << " = " << value << ";\n";
  } else if (t.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
    stream << vec;
    // fp4_e2_64_t
    if (t.lanes() >= 64)
      stream << "." << access[i / 32];
    // fp4_e2_32_t
    if (t.lanes() >= 32)
      stream << "." << access[(i % 32) / 16];
    // fp4_e2_16_t
    if (t.lanes() >= 16)
      stream << "." << access[(i % 16) / 8];
    // fp4_e2_8_t
    if (t.lanes() >= 8)
      stream << "." << access[(i % 8) / 4];
    // fp4_e2_4_t -> fp4_e2_2_t member
    if (t.lanes() >= 4)
      stream << "." << access[(i % 4) / 2];
    // fp4_e2_2_t -> set_x() or set_y()
    stream << ".set_" << access[i % 2] << "(" << value << ");\n";
  } else {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  }
}

void CodeGenTileLangCUDA::PrintStorageSync(const CallObj *op) {
  auto args = op->args;
  const std::string &sync = args[0].as<StringImmObj>()->value;
  if (sync == "warp") {
    // DO nothing.
  } else if (sync == "shared" || sync == "shared.dyn") {
    this->PrintIndent();
    if (args.size() == 1) {
      this->stream << "__syncthreads();\n";
    } else if (args.size() == 2) {
      ICHECK(args[1]->ty.as_or_throw<PrimType>().MatchesCode(kDLInt))
          << "storage_sync barrier_id must be integer type, got "
          << args[1]->ty.as_or_throw<PrimType>();
      this->stream << "tl::__sync_thread_partial(" << PrintExpr(args[1])
                   << ");\n";
    } else if (args.size() == 3) {
      ICHECK(args[1]->ty.as_or_throw<PrimType>().MatchesCode(kDLInt))
          << "storage_sync barrier_id must be integer type, got "
          << args[1]->ty.as_or_throw<PrimType>();
      ICHECK(args[2]->ty.as_or_throw<PrimType>().MatchesCode(kDLInt))
          << "storage_sync thread_count must be integer type, got "
          << args[2]->ty.as_or_throw<PrimType>();
      this->stream << "tl::__sync_thread_partial(" << PrintExpr(args[1]) << ", "
                   << PrintExpr(args[2]) << ");\n";
    } else {
      TVM_FFI_THROW(InternalError)
          << "Invalid number of arguments for storage sync: " << args.size();
    }
  } else if (sync == "cluster") {
    need_cluster_h_ = true;
    this->PrintIndent();
    this->stream << "tl::cluster_sync();\n";
  } else if (sync == "global") {
    TVM_FFI_THROW(InternalError)
        << "Global storage sync is no longer supported";
  }
}

void CodeGenTileLangCUDA::PrintStorageScope(
    const runtime::MemoryScope &memory_scope,
    std::ostream &os) { // NOLINT(*)
  const std::string scope = memory_scope.name();
  ICHECK_NE(scope, "global")
      << "Cannot allocate global memory when targeting CUDA. You must pass "
         "all global arrays as input instead";
  if (scope == "shared" || scope == "shared.barrier" ||
      scope == "shared.cluster_barrier") {
    os << "__shared__ __align__(" << barrier_alignment_bytes_ << ") ";
  } else if (scope == "shared.dyn") {
    os << "extern __shared__ __align__(1024) ";
  }
}

std::string CodeGenTileLangCUDA::CastFromTo(std::string value,
                                            const PrimType &from,
                                            const PrimType &target) {
  if (from == target)
    return value;
  std::ostringstream os;
  os << "((";
  this->PrintType(target, os);
  os << ")";
  if (from.MatchesElementType(kDLFloat, 16) &&
      (target.MatchesCode(kDLInt) || target.MatchesCode(kDLUInt)) &&
      target.bits() == 8) {
    os << "(";
    if (target.MatchesCode(kDLUInt)) {
      os << "u";
    }
    os << "int)";
  }
  if ((from.MatchesElementType(kDLFloat, 16) ||
       from.MatchesElementType(kDLBfloat, 16)) &&
      CodegenIsFloat8(target)) {
    os << "(float)";
  }
  os << value << ")";
  return os.str();
}

void CodeGenTileLangCUDA::Dispatch_(const prim::CastObj *op, std::ostream &os) {
  const PrimType target_ty = op->ty.as_or_throw<PrimType>();
  // Only plain casts may discard this exact f32 detour. An explicit tl.cast
  // carries an annotation boundary even when its sat flag equals the default.
  if (const auto *inner = op->value.as<prim::CastObj>();
      inner && op->value.ty().MatchesElementType(kDLFloat, 32) &&
      inner->value.ty().MatchesElementType(kDLFloat4_e2m1fn, 4) &&
      (target_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
       target_ty.MatchesElementType(kDLFloat8_e4m3fn, 8))) {
    os << PrintExpr(prim::Cast(target_ty, inner->value));
    return;
  }
  PrintCast(op->value, target_ty, "", true, std::nullopt, os);
}

void CodeGenTileLangCUDA::PrintCast(const PrimExpr &value,
                                    const PrimType &target_ty,
                                    const std::string &cast_round,
                                    bool cast_sat,
                                    Optional<PrimExpr> cast_rbits,
                                    std::ostream &os) {
  auto native_cast = prim::Cast(target_ty, value);
  const auto *op = native_cast.get();
  PrimType from_ty = value.ty();
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());
  // Scalar fp8 <-> half via the __tl_cvt helpers; the default cast detours
  // through fp32.
  if (from_ty.IsScalar() && cast_round.empty() &&
      target_ty.MatchesElementType(kDLFloat, 16) &&
      tl::IsCudaVectorizableFP8(from_ty)) {
    bool is_e4m3 = from_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
                   from_ty.MatchesElementType(kDLFloat8_e4m3fn, 8);
    std::string interp = is_e4m3 ? "__NV_E4M3" : "__NV_E5M2";
    os << "half_t(__tl_cvt_fp8_to_half((" << PrintExpr(op->value) << ").raw(), "
       << interp << "))";
    return;
  }

  if (from_ty.IsScalar() && cast_round.empty() &&
      from_ty.MatchesElementType(kDLFloat, 16) &&
      tl::IsCudaVectorizableFP8(target_ty)) {
    bool is_e4m3 = target_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
                   target_ty.MatchesElementType(kDLFloat8_e4m3fn, 8);
    std::string interp = is_e4m3 ? "__NV_E4M3" : "__NV_E5M2";
    this->PrintType(target_ty, os);
    os << "::bitcast(__tl_cvt_half_to_fp8((" << PrintExpr(op->value)
       << ").to_half(), " << interp << "))";
    return;
  }

  // Scalar fp8 <-> bfloat16: same idea, via the native/PTX helpers.
  if (from_ty.IsScalar() && cast_round.empty() &&
      target_ty.MatchesElementType(kDLBfloat, 16) &&
      tl::IsCudaVectorizableFP8(from_ty)) {
    bool is_e4m3 = from_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
                   from_ty.MatchesElementType(kDLFloat8_e4m3fn, 8);
    std::string interp = is_e4m3 ? "__NV_E4M3" : "__NV_E5M2";
    os << "bfloat16_t(__tl_cvt_fp8_to_bfloat16((" << PrintExpr(op->value)
       << ").raw(), " << interp << "))";
    return;
  }

  if (from_ty.IsScalar() && cast_round.empty() &&
      from_ty.MatchesElementType(kDLBfloat, 16) &&
      tl::IsCudaVectorizableFP8(target_ty)) {
    bool is_e4m3 = target_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
                   target_ty.MatchesElementType(kDLFloat8_e4m3fn, 8);
    std::string interp = is_e4m3 ? "__NV_E4M3" : "__NV_E5M2";
    this->PrintType(target_ty, os);
    os << "::bitcast(__tl_cvt_bfloat16_to_fp8((" << PrintExpr(op->value)
       << ").to_nv_bfloat16(), " << interp << "))";
    return;
  }

  // Scalar fp4 (E2M1) -> fp8 (E4M3): exact bit transcode.
  if (from_ty.IsScalar() && cast_round.empty() &&
      from_ty.MatchesElementType(kDLFloat4_e2m1fn, 4) &&
      (target_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
       target_ty.MatchesElementType(kDLFloat8_e4m3fn, 8))) {
    this->PrintType(target_ty, os);
    os << "::bitcast(__tl_cvt_e2m1_to_e4m3((" << PrintExpr(op->value)
       << ").__x))";
    return;
  }

  // The FP4 wrapper's uint8 constructor preserves storage bits. An IR numeric
  // integer cast must instead use its float constructor; an int literal is
  // otherwise ambiguous, and a uint8 operand would silently become raw bits.
  bool integer_to_fp4 = target_ty.MatchesElementType(kDLFloat4_e2m1fn, 4) &&
                        from_ty.MatchesCode(kDLInt, kDLUInt, kDLBool);
  if (from_ty.IsScalar() && cast_round.empty() && integer_to_fp4) {
    this->PrintType(target_ty, os);
    os << "(static_cast<float>(" << PrintExpr(op->value) << "))";
    return;
  }

  // Emit simple C-style type conversion for scalar casts without custom
  // rounding.
  if (from_ty.IsScalar() && cast_round.empty())
    return CodeGenC::Dispatch_(op, os);

  // We could emit make_float4 like calls, but the emitted code looks
  // too compact to read. Emit this as vectorized unary ops.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(target_ty, stream);
  stream << ' ' << sret << ";\n";
  std::string src =
      SSAGetID(PrintExpr(op->value), from_ty,
               tvm::SideEffect(op->value) == CallEffectKind::kPure);

  int lanes = from_ty.lanes();

  auto PrintVectorizedCast =
      [&](const std::string &cast_func, const std::string &src_type,
          const std::string &dst_type, const std::string &extra_args = "",
          bool src_needs_reinterpret = false,
          bool dst_needs_reinterpret = false, int chunk_lanes = 2) {
        int num_chunks = lanes / chunk_lanes;
        std::string src_cast = src_needs_reinterpret
                                   ? "reinterpret_cast<" + src_type + "*>"
                                   : "(" + src_type + "*)";
        std::string dst_cast = dst_needs_reinterpret
                                   ? "reinterpret_cast<" + dst_type + "*>"
                                   : "(" + dst_type + "*)";

        for (int i = 0; i < num_chunks; i++) {
          PrintIndent();
          stream << "(" << dst_cast << "(&" << sret << "))[" << i
                 << "] = " << cast_func << "((" << src_cast << "(&" << src
                 << "))[" << i << "]" << extra_args << ");\n";
        }
        os << sret;
      };

  // A list of casting functions that are supported by TileLang templates.
  // To add a new type conversion, you should do the following things:
  // 1. Add the new conversion function in tl_templates. (__tl_cvt_xx)
  // 2. Add a new if statement like the one below.
  // 3. In src/cuda/target_utils.cc, allow this vectorizable cast.

  // Validate supported dtype/rounding combinations before conversion branches
  // can silently ignore the requested mode.
  auto round_is_supported = [&]() {
    if (cast_round.empty()) {
      return true;
    }
    if (cast_round != "rs" || !cast_rbits.has_value()) {
      return false;
    }
    if (!(from_ty.MatchesCode(kDLFloat) && from_ty.bits() == 32)) {
      return false;
    }
    // E8M0 is deliberately absent: its helpers round towards +infinity
    // unconditionally, so they cannot honour a requested mode.
    return target_ty.MatchesElementType(kDLFloat, 16) ||
           target_ty.MatchesElementType(kDLBfloat, 16) ||
           tl::IsCudaVectorizableFP8(target_ty) ||
           target_ty.MatchesElementType(kDLFloat4_e2m1fn, 4);
  };
  if (!round_is_supported()) {
    TVM_FFI_THROW(InternalError)
        << "round '" << cast_round << "' is not supported for a cast from "
        << from_ty << " to " << target_ty
        << ". Supported combinations: round='rs' with an rbits operand, "
           "from float32 to float16, bfloat16, float8_e4m3*, "
           "float8_e5m2 or float4_e2m1fn; or any dtype pair without a "
           "round annotation.";
  }

  // Handle conversion from float16 to float32
  if (from_ty.MatchesElementType(kDLFloat, 16) &&
      target_ty.MatchesCode(kDLFloat) && target_ty.bits() == 32) {
    // Use __half22float2 for vectorized conversion (half2 -> float2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__half22float2", "half2", "float2");
      return;
    }
  }

  // Handle conversion from float32 to float16
  if (from_ty.MatchesCode(kDLFloat) && from_ty.bits() == 32 &&
      target_ty.MatchesElementType(kDLFloat, 16)) {
    if (cast_round == "rs" && cast_rbits.has_value()) {
      ICHECK(cast_sat)
          << "sat=false is not supported for stochastic rounding f32 -> fp16";
      std::string extra_args = ", " + PrintExpr(cast_rbits.value());
      if (lanes == 1) {
        PrintIndent();
        stream << "*reinterpret_cast<unsigned short*>(&" << sret
               << ") = tl::__tl_cvt_f32x1_to_f16x1_rs_sat(" << src << extra_args
               << ");\n";
        os << sret;
        return;
      }
      if (lanes == 2 || lanes == 4 || lanes == 8) {
        PrintVectorizedCast("tl::__tl_cvt_f32x2_to_f16x2_rs_sat", "float2",
                            "half2", extra_args);
        return;
      }
    }
    if (!cast_round.empty()) {
      TVM_FFI_THROW(InternalError)
          << "Unsupported rounding mode '" << cast_round
          << "' for f32 -> FP16 cast. Only packed 'rs' stochastic "
             "rounding is supported.";
    }
    // Use __float22half2_rn for vectorized conversion (float2 -> half2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__float22half2_rn", "float2", "half2");
      return;
    }
  }

  // Handle conversion from bfloat16 to float32
  if (from_ty.MatchesElementType(kDLBfloat, 16) &&
      target_ty.MatchesCode(kDLFloat) && target_ty.bits() == 32) {
    // Expand each bf16 lane directly into the high 16 bits of an fp32 lane.
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("tl::__tl_cvt_bf16x2_to_f32x2", "__nv_bfloat162",
                          "float2", "", true, false);
      return;
    }
  }

  // Handle conversion from float32 to bfloat16
  if (from_ty.MatchesCode(kDLFloat) && from_ty.bits() == 32 &&
      target_ty.MatchesElementType(kDLBfloat, 16)) {
    if (cast_round == "rs" && cast_rbits.has_value()) {
      ICHECK(cast_sat)
          << "sat=false is not supported for stochastic rounding f32 -> bf16";
      std::string extra_args = ", " + PrintExpr(cast_rbits.value());
      if (lanes == 1) {
        PrintIndent();
        stream << "*reinterpret_cast<unsigned short*>(&" << sret
               << ") = tl::__tl_cvt_f32x1_to_bf16x1_rs_sat(" << src
               << extra_args << ");\n";
        os << sret;
        return;
      }
      if (lanes == 2 || lanes == 4 || lanes == 8) {
        PrintVectorizedCast("tl::__tl_cvt_f32x2_to_bf16x2_rs_sat", "float2",
                            "__nv_bfloat162", extra_args, false, true);
        return;
      }
    }
    if (!cast_round.empty()) {
      TVM_FFI_THROW(InternalError)
          << "Unsupported rounding mode '" << cast_round
          << "' for f32 -> BF16 cast. Only packed 'rs' stochastic "
             "rounding is supported.";
    }
    // Use __float22bfloat162_rn for vectorized conversion (float2 -> bfloat162)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__float22bfloat162_rn", "float2", "__nv_bfloat162",
                          "", false, true);
      return;
    }
  }

  // Handle conversion from float32 to float8 (E4M3/E5M2)
  if (from_ty.MatchesCode(kDLFloat) && from_ty.bits() == 32 &&
      tl::IsCudaVectorizableFP8(target_ty)) {
    bool target_type_is_e4m3 =
        target_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
        target_ty.MatchesElementType(kDLFloat8_e4m3fn, 8);
    std::string type_suffix = target_type_is_e4m3 ? "__NV_E4M3" : "__NV_E5M2";

    bool has_rbits = cast_rbits.has_value();

    if (cast_round == "rs" && has_rbits) {
      // PTX: cvt.rs.satfinite.{e4m3x4,e5m2x4}.f32 d, {a,b,e,f}, rbits
      // x4 is the native PTX width; x1/x2 zero-pad the unused lanes.
      ICHECK(cast_sat)
          << "sat=false is not supported for stochastic rounding f32 -> fp8";
      std::string fp8_type_x4 = target_type_is_e4m3 ? "e4m3x4" : "e5m2x4";
      std::string fp8_type_x2 = target_type_is_e4m3 ? "e4m3x2" : "e5m2x2";
      std::string fp8_type_x1 = target_type_is_e4m3 ? "e4m3x1" : "e5m2x1";
      std::string rbits_str = PrintExpr(cast_rbits.value());

      if (lanes == 1) {
        // x1: pad 3 zeros, take byte 0
        std::string func_name = "__tl_cvt_f32x1_to_" + fp8_type_x1 + "_rs_sat";
        PrintIndent();
        stream << "*reinterpret_cast<__nv_fp8_storage_t*>(&" << sret
               << ") = " << func_name << "(" << src << ", " << rbits_str
               << ");\n";
        os << sret;
        return;
      }

      if (lanes == 2) {
        // x2: pad 2 zeros, take low 2 bytes
        std::string func_name = "__tl_cvt_f32x2_to_" + fp8_type_x2 + "_rs_sat";
        std::string src_cast = "(float2*)";
        std::string dst_cast = "reinterpret_cast<__nv_fp8x2_storage_t*>";
        PrintIndent();
        stream << "(" << dst_cast << "(&" << sret << "))[0] = " << func_name
               << "((" << src_cast << "(&" << src << "))[0], " << rbits_str
               << ");\n";
        os << sret;
        return;
      }

      if (lanes == 4 || lanes == 8) {
        // x4: native; chunk lanes==8 into two float4s
        std::string func_name = "__tl_cvt_f32x4_to_" + fp8_type_x4 + "_rs_sat";
        int num_chunks = lanes / 4;
        std::string src_cast = "(float4*)";
        std::string dst_cast = "reinterpret_cast<__nv_fp8x4_storage_t*>";
        for (int i = 0; i < num_chunks; i++) {
          PrintIndent();
          stream << "(" << dst_cast << "(&" << sret << "))[" << i
                 << "] = " << func_name << "(" << "(" << src_cast << "(&" << src
                 << "))[" << i << "], " << rbits_str << ");\n";
        }
        os << sret;
        return;
      }
    }

    if (!cast_round.empty()) {
      TVM_FFI_THROW(InternalError)
          << "Unsupported rounding mode '" << cast_round
          << "' for f32 -> FP8 cast. "
          << "Only 'rs' (stochastic with rbits) is supported.";
    }

    // Use __nv_cvt_float2_to_fp8x2 for vectorized conversion (float2 -> fp8x2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      std::string extra_args = ", __NV_SATFINITE, " + type_suffix;
      PrintVectorizedCast("__nv_cvt_float2_to_fp8x2", "float2",
                          "__nv_fp8x2_storage_t", extra_args, false, true);
      return;
    }
  }

  // Handle conversion from float16 to float8 (E4M3/E5M2)
  if (from_ty.MatchesElementType(kDLFloat, 16) &&
      tl::IsCudaVectorizableFP8(target_ty)) {
    bool target_type_is_e4m3 =
        target_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
        target_ty.MatchesElementType(kDLFloat8_e4m3fn, 8);
    std::string type_suffix = target_type_is_e4m3 ? "__NV_E4M3" : "__NV_E5M2";
    // Use __tl_cvt_half2_to_fp8x2 for vectorized conversion (half2 -> fp8x2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_half2_to_fp8x2", "half2",
                          "__nv_fp8x2_storage_t", ", " + type_suffix, false,
                          true);
      return;
    }
  }

  // Handle conversion from bfloat16 to float8 (E4M3/E5M2)
  if (from_ty.MatchesElementType(kDLBfloat, 16) &&
      tl::IsCudaVectorizableFP8(target_ty)) {
    bool target_type_is_e4m3 =
        target_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
        target_ty.MatchesElementType(kDLFloat8_e4m3fn, 8);
    std::string type_suffix = target_type_is_e4m3 ? "__NV_E4M3" : "__NV_E5M2";
    // Use __tl_cvt_bfloat162_to_fp8x2 for vectorized conversion (bfloat162 ->
    // fp8x2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_bfloat162_to_fp8x2", "__nv_bfloat162",
                          "__nv_fp8x2_storage_t", ", " + type_suffix, true,
                          true);
      return;
    }
  }

  // Handle conversion from float8 (E4M3/E5M2) to float32
  if (tl::IsCudaVectorizableFP8(from_ty) && target_ty.MatchesCode(kDLFloat) &&
      target_ty.bits() == 32) {
    bool from_type_is_e4m3 = from_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
                             from_ty.MatchesElementType(kDLFloat8_e4m3fn, 8);
    std::string type_suffix = from_type_is_e4m3 ? "__NV_E4M3" : "__NV_E5M2";

    // Use __tl_cvt_fp8x2_to_float2 for vectorized conversion (fp8x2 -> float2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_fp8x2_to_float2", "__nv_fp8x2_storage_t",
                          "float2", ", " + type_suffix, true, false);
      return;
    }
  }

  // Handle conversion from float8 (E4M3/E5M2) to float16
  if (tl::IsCudaVectorizableFP8(from_ty) &&
      target_ty.MatchesElementType(kDLFloat, 16)) {
    bool from_type_is_e4m3 = from_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
                             from_ty.MatchesElementType(kDLFloat8_e4m3fn, 8);
    std::string type_suffix = from_type_is_e4m3 ? "__NV_E4M3" : "__NV_E5M2";
    // Use __tl_cvt_fp8x2_to_half2 for vectorized conversion (fp8x2 -> half2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_fp8x2_to_half2", "__nv_fp8x2_storage_t",
                          "half2", ", " + type_suffix, true, false);
      return;
    }
  }

  // Handle conversion from float8 (E4M3/E5M2) to bfloat16
  if (tl::IsCudaVectorizableFP8(from_ty) &&
      target_ty.MatchesElementType(kDLBfloat, 16)) {
    bool from_type_is_e4m3 = from_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
                             from_ty.MatchesElementType(kDLFloat8_e4m3fn, 8);
    // PTX cvt encodes the fp8 type in the mnemonic, so pick the helper by name.
    std::string cast_func = from_type_is_e4m3 ? "__tl_cvt_e4m3x2_to_bfloat162"
                                              : "__tl_cvt_e5m2x2_to_bfloat162";
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast(cast_func, "__nv_fp8x2_storage_t", "__nv_bfloat162",
                          "", true, false);
      return;
    }
  }

  // Handle conversion from float8 (E8M0) to bfloat16
  if (from_ty.MatchesElementType(kDLFloat8_e8m0fnu, 8) &&
      target_ty.MatchesElementType(kDLBfloat, 16)) {
    // Use __tl_cvt_e8m0x2_to_bfloat162 for vectorized conversion (fp8_e8m0x2 ->
    // bfloat162)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_e8m0x2_to_bfloat162",
                          "__nv_fp8x2_storage_t", "__nv_bfloat162", "", true,
                          false);
      return;
    }
  }

  // Handle conversion from bfloat16 to float8 (E8M0)
  if (from_ty.MatchesElementType(kDLBfloat, 16) &&
      target_ty.MatchesElementType(kDLFloat8_e8m0fnu, 8)) {
    // Use __tl_cvt_bfloat162_to_e8m0x2 for vectorized conversion (bfloat162 ->
    // fp8_e8m0x2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_bfloat162_to_e8m0x2", "__nv_bfloat162",
                          "__nv_fp8x2_storage_t", "", false, true);
      return;
    }
  }

  // Handle conversion from float32 to float8 (E8M0)
  if (from_ty.MatchesCode(kDLFloat) && from_ty.bits() == 32 &&
      target_ty.MatchesElementType(kDLFloat8_e8m0fnu, 8)) {
    // Use __tl_cvt_float2_to_e8m0x2 for vectorized conversion (float2 ->
    // fp8_e8m0x2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_float2_to_e8m0x2", "float2",
                          "__nv_fp8x2_storage_t", "", false, true);
      return;
    }
  }

  // Handle conversion from double to float8 (E8M0)
  if (from_ty.MatchesCode(kDLFloat) && from_ty.bits() == 64 &&
      target_ty.MatchesElementType(kDLFloat8_e8m0fnu, 8)) {
    // Use __tl_cvt_double2_to_e8m0x2 for vectorized conversion (double2 ->
    // fp8_e8m0x2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_double2_to_e8m0x2", "double2",
                          "__nv_fp8x2_storage_t", "", false, true);
      return;
    }
  }

  // Handle conversion from float16 to float4 (E2M1)
  if (from_ty.MatchesElementType(kDLFloat, 16) &&
      target_ty.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
    // Use __tl_cvt_half2_to_fp4x2 for vectorized conversion (half2 -> fp4x2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_half2_to_fp4x2", "half2", "uint8_t", "",
                          false, true);
      return;
    }
  }

  // Handle conversion from float32 to float4 (E2M1)
  if (from_ty.MatchesCode(kDLFloat) && from_ty.bits() == 32 &&
      target_ty.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
    bool has_rbits = cast_rbits.has_value();

    if (cast_round == "rs" && has_rbits) {
      // PTX: cvt.rs.satfinite.e2m1x4.f32 d, {a, b, e, f}, rbits
      // x4 is the native PTX width; x1/x2 zero-pad the unused lanes.
      ICHECK(cast_sat)
          << "sat=false is not supported for stochastic rounding f32 -> fp4";
      std::string rbits_str = PrintExpr(cast_rbits.value());

      if (lanes == 1) {
        // x1: pad 3 zeros, take low nibble
        PrintIndent();
        stream << "*reinterpret_cast<__nv_fp4_storage_t*>(&" << sret
               << ") = " << "__tl_cvt_f32x1_to_e2m1x1_rs_sat(" << src << ", "
               << rbits_str << ");\n";
        os << sret;
        return;
      }

      if (lanes == 2) {
        // x2: pad 2 zeros, take low byte
        PrintIndent();
        stream << "*reinterpret_cast<__nv_fp4x2_storage_t*>(&" << sret
               << ") = " << "__tl_cvt_f32x2_to_e2m1x2_rs_sat(((float2*)(&"
               << src << "))[0], " << rbits_str << ");\n";
        os << sret;
        return;
      }

      if (lanes == 4 || lanes == 8) {
        // x4: native; chunk lanes==8 into two float4s
        int num_chunks = lanes / 4;
        for (int i = 0; i < num_chunks; i++) {
          PrintIndent();
          stream << "reinterpret_cast<__nv_fp4x4_storage_t*>(&" << sret << ")["
                 << i << "] = __tl_cvt_f32x4_to_e2m1x4_rs_sat("
                 << "((float4*)(&" << src << "))[" << i << "], " << rbits_str
                 << ");\n";
        }
        os << sret;
        return;
      }
    }

    if (!cast_round.empty()) {
      TVM_FFI_THROW(InternalError)
          << "Unsupported rounding mode '" << cast_round
          << "' for f32 -> FP4 cast. "
          << "PTX ISA only supports 'rs' (e2m1x4 stochastic).";
    }

    // Use __tl_cvt_float2_to_fp4x2 for vectorized conversion (float2 -> fp4x2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_float2_to_fp4x2", "float2", "uint8_t", "",
                          false, true);
      return;
    }
  }

  // Handle conversion from float4 (E2M1) to float16
  if (from_ty.MatchesElementType(kDLFloat4_e2m1fn, 4) &&
      target_ty.MatchesElementType(kDLFloat, 16)) {
    // Use __tl_cvt_fp4x2_to_half2 for vectorized conversion (fp4x2 -> half2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_fp4x2_to_half2", "uint8_t", "half2", "",
                          true, false);
      return;
    }
  }

  // Handle conversion from float4 (E2M1) to float32
  if (from_ty.MatchesElementType(kDLFloat4_e2m1fn, 4) &&
      target_ty.MatchesCode(kDLFloat) && target_ty.bits() == 32) {
    // Use __tl_cvt_fp4x2_to_float2 for vectorized conversion (fp4x2 -> float2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_fp4x2_to_float2", "uint8_t", "float2", "",
                          true, false);
      return;
    }
  }

  // Handle conversion from double to float4 (E2M1)
  if (from_ty.MatchesCode(kDLFloat) && from_ty.bits() == 64 &&
      target_ty.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
    // Use __tl_cvt_double2_to_fp4x2 for vectorized conversion (double2 ->
    // fp4x2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_double2_to_fp4x2", "double2", "uint8_t", "",
                          false, true);
      return;
    }
  }

  // Handle conversion from float4 (E2M1) to double
  if (from_ty.MatchesElementType(kDLFloat4_e2m1fn, 4) &&
      target_ty.MatchesCode(kDLFloat) && target_ty.bits() == 64) {
    // Use __tl_cvt_fp4x2_to_double2 for vectorized conversion (fp4x2 ->
    // double2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_fp4x2_to_double2", "uint8_t", "double2", "",
                          true, false);
      return;
    }
  }

  // Handle conversion from bfloat16 to float4 (E2M1)
  if (from_ty.MatchesElementType(kDLBfloat, 16) &&
      target_ty.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
    // Use __tl_cvt_bfloat162_to_fp4x2 for vectorized conversion (bfloat162 ->
    // fp4x2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_bfloat162_to_fp4x2", "__nv_bfloat162",
                          "uint8_t", "", false, true);
      return;
    }
  }

  // Handle conversion from float4 (E2M1) to bfloat16
  if (from_ty.MatchesElementType(kDLFloat4_e2m1fn, 4) &&
      target_ty.MatchesElementType(kDLBfloat, 16)) {
    // Use __tl_cvt_fp4x2_to_bfloat162 for vectorized conversion (fp4x2 ->
    // bfloat162)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_fp4x2_to_bfloat162", "uint8_t",
                          "__nv_bfloat162", "", true, false);
      return;
    }
  }

  // Handle conversion from float4 (E2M1) to float8 (E4M3)
  if (from_ty.MatchesElementType(kDLFloat4_e2m1fn, 4) &&
      (target_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
       target_ty.MatchesElementType(kDLFloat8_e4m3fn, 8))) {
    // Exact bit transcode; __tl_cvt_e2m1x4_to_e4m3x4 handles four lanes per
    // __byte_perm pair, so chunk by four wherever the width allows.
    if (lanes == 2) {
      PrintVectorizedCast("__tl_cvt_e2m1x2_to_e4m3x2", "__nv_fp4x2_storage_t",
                          "__nv_fp8x2_storage_t", "", true, true);
      return;
    }
    if (lanes == 4 || lanes == 8 || lanes == 16 || lanes == 32) {
      PrintVectorizedCast("__tl_cvt_e2m1x4_to_e4m3x4", "__nv_fp4x4_storage_t",
                          "__nv_fp8x4_storage_t", "", true, true,
                          /*chunk_lanes=*/4);
      return;
    }
  }

  // Reaching here with a non-empty `round` means this dtype pair has no
  // PTX `cvt.<round>` lowering; fail loudly instead of silently dropping it.
  if (!cast_round.empty()) {
    TVM_FFI_THROW(InternalError)
        << "round '" << cast_round << "' is not supported for cast from "
        << from_ty << " to " << target_ty
        << " (only supported f32 packed stochastic conversions are "
           "available)";
  }

  // Fallback: elementwise cast.
  // fp16<->bf16 elements load as native __half/__nv_bfloat16, where a direct
  // `(half_t)(__nv_bfloat16)` (or reverse) is an ambiguous conversion; route
  // through float to disambiguate.
  bool cross_half = (from_ty.MatchesElementType(kDLFloat, 16) &&
                     target_ty.MatchesElementType(kDLBfloat, 16)) ||
                    (from_ty.MatchesElementType(kDLBfloat, 16) &&
                     target_ty.MatchesElementType(kDLFloat, 16));
  for (int i = 0, lanes = from_ty.lanes(); i < lanes; ++i) {
    std::ostringstream val;
    val << "(";
    PrintType(target_ty.WithLanes(1), val);
    val << ")(";
    if (cross_half || integer_to_fp4)
      val << "(float)(";
    PrintVecElemLoad(src, from_ty, i, val);
    if (cross_half || integer_to_fp4)
      val << ")";
    val << ")";
    PrintVecElemStore(sret, target_ty, i, val.str());
  }

  os << sret;
}

void CodeGenTileLangCUDA::Dispatch_(const prim::MinObj *op, std::ostream &os) {
  // TODO(wt): Consider vectorized reduction and impl for other dtypes
  PrimType t = op->ty.as_or_throw<PrimType>();

  // Standard min/max functions don't support bfloat16 or float16
  if (t.MatchesElementType(kDLBfloat, 16) && t.IsScalar()) {
    os << "cutlass::bfloat16_t(__hmin(" << "(" << PrintExpr(op->a)
       << ").to_nv_bfloat16(), " << "(" << PrintExpr(op->b)
       << ").to_nv_bfloat16()))";
    return;
  }

  if (t.MatchesElementType(kDLFloat, 16) && t.IsScalar()) {
    os << "cutlass::half_t(__hmin(" << "(" << PrintExpr(op->a)
       << ").to_half(), " << "(" << PrintExpr(op->b) << ").to_half()))";
    return;
  }

  // For float32 and float64 scalar, use standard min functions
  if (t.MatchesCode(kDLFloat) && t.IsScalar()) {
    if (t.bits() == 32 || t.bits() == 64) {
      os << "min(" << PrintExpr(op->a) << ", " << PrintExpr(op->b) << ")";
      return;
    }
  }

  // For all other scalar types (int, uint), use default implementation
  CodeGenC::Dispatch_(op, os);
}

void CodeGenTileLangCUDA::Dispatch_(const prim::MaxObj *op, std::ostream &os) {
  // TODO(wt): Consider vectorized reduction and impl for other dtypes
  PrimType t = op->ty.as_or_throw<PrimType>();

  // Standard min/max functions don't support bfloat16 or float16
  if (t.MatchesElementType(kDLBfloat, 16) && t.IsScalar()) {
    os << "cutlass::bfloat16_t(__hmax(" << "(" << PrintExpr(op->a)
       << ").to_nv_bfloat16(), " << "(" << PrintExpr(op->b)
       << ").to_nv_bfloat16()))";
    return;
  }

  if (t.MatchesElementType(kDLFloat, 16) && t.IsScalar()) {
    os << "cutlass::half_t(__hmax(" << "(" << PrintExpr(op->a)
       << ").to_half(), " << "(" << PrintExpr(op->b) << ").to_half()))";
    return;
  }

  // For float32 and float64 scalar, use standard max functions
  if (t.MatchesCode(kDLFloat) && t.IsScalar()) {
    if (t.bits() == 32 || t.bits() == 64) {
      os << "max(" << PrintExpr(op->a) << ", " << PrintExpr(op->b) << ")";
      return;
    }
  }

  // For all other scalar types (int, uint), use default implementation
  CodeGenC::Dispatch_(op, os);
}

void CodeGenTileLangCUDA::PrintCallExtern(Type ret_type, String global_symbol,
                                          const Array<Expr> &args,
                                          bool skip_first_arg,
                                          std::ostream &os) { // NOLINT(*)
  if (!ret_type.as<PrimTypeObj>()) {
    CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg,
                              os);
    return;
  }
  PrimType ret_dtype = ret_type.as_or_throw<PrimType>();
  if (ret_dtype.IsFixedLengthVector()) {
    //
    // Emit an unsupported vector call
    //
    // v = intrin_f((float4*)A[0], (float4*)B[0])
    //
    // as
    //
    // float4 __ret;
    // {
    //   float4 __arg0 = ((float4*)A)[0];
    //   float4 __arg1 = ((float4*)B)[0];
    //   __ret.x = intrin_f(__arg0.x, __arg1.x);
    //   __ret.y = intrin_f(__arg0.y, __arg1.y);
    //   __ret.z = intrin_f(__arg0.z, __arg1.z);
    //   __ret.w = intrin_f(__arg0.w, __arg1.w);
    // }
    // v = __ret;
    //
    // Declare the result vector.
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(ret_dtype, stream);
    stream << ' ' << sret << ";\n";
    {
      // Load arguments.
      std::vector<std::string> sargs;
      size_t arg_begin = static_cast<size_t>(skip_first_arg);
      for (size_t i = arg_begin; i < args.size(); ++i) {
        std::string val =
            SSAGetID(PrintExpr(args[i]), args[i]->ty.as_or_throw<PrimType>(),
                     tvm::SideEffect(args[i]) == CallEffectKind::kPure);
        sargs.push_back(std::move(val));
      }

      // Emit a scalar call for each lane.
      for (int i = 0; i < ret_dtype.lanes(); ++i) {
        std::ostringstream scall;
        scall << global_symbol << "(";
        for (size_t j = 0; j < sargs.size(); ++j) {
          if (j > 0)
            scall << ", ";
          PrintVecElemLoad(sargs[j],
                           args[arg_begin + j]->ty.as_or_throw<PrimType>(), i,
                           scall);
        }
        scall << ")";
        PrintVecElemStore(sret, ret_dtype, i, scall.str());
      }
    }
    os << sret;
  } else {
    CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg,
                              os);
  }
}

// Print a reference expression to a buffer.
std::string CodeGenTileLangCUDA::GetBufferRef(const PrimType &t,
                                              const VarObj *buffer,
                                              PrimExpr index) {
  const VarObj *buffer_var = buffer;
  std::ostringstream os;
  std::string vid = GetVarID(buffer_var);
  // Local scalar FP4 allocations use a distinct fp4x2 backing array. Every
  // reference, including address_of, must name that physical allocation.
  auto it = fp4_packed_buffers_.find(ffi::GetRef<Var>(buffer));
  if (it != fp4_packed_buffers_.end()) {
    vid = it->second;
  }
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var).name();
  }
  if (scope.empty()) {
    scope = CodegenStorageScope(ffi::GetRef<Var>(buffer));
  }
  const bool is_vol = IsVolatile(buffer_var);

  auto ptr_cast = [this, is_vol, scope](PrimType pointed_to) {
    std::ostringstream ptr_os;
    ptr_os << "(";
    if (is_vol) {
      ptr_os << "volatile ";
    }
    if (!scope.empty() && IsScopePartOfType()) {
      PrintStorageScope(runtime::MemoryScope::Get(scope), ptr_os);
    }
    if (tl::IsFloat4E2M1Unpacked(pointed_to) &&
        (scope == "shared" || scope == "shared.dyn")) {
      ptr_os << "uint8_t";
    } else {
      PrintType(pointed_to, ptr_os);
    }
    ptr_os << "*)";
    return ptr_os.str();
  };

  PrimType buffer_element_dtype = CodegenTensorDType(buffer);

  std::string buffer_str = vid;
  if (!HandleTypeMatch(buffer_var, buffer_element_dtype) || is_vol) {
    std::stringstream temp;
    temp << "(" << ptr_cast(buffer_element_dtype) << vid << ")";
    buffer_str = temp.str();
  }
  if (scope == "local.var" || scope.find("local.descriptor") == 0) {
    os << vid;
    return os.str();
  }
  std::string index_str = PrintExpr(index);
  bool is_compact_scalar_fp4 =
      buffer_element_dtype.lanes() == 1 &&
      buffer_element_dtype.MatchesElementType(kDLFloat4_e2m1fn, 4);
  if ((t.bits() == 4 && !CodegenIsFloat4(t)) ||
      (t.bits() == 1 && t.MatchesCode(kDLInt))) {
    // Scalar int4/uint4 storage is byte-packed (2 logical elements per byte).
    // Vector int4 loads/stores reinterpret the underlying packed bytes as the
    // requested vector type, so their index still advances by the vector lane
    // count. Scalar int1 keeps the existing int32 backing.
    int div_factor = t.lanes();
    if (t.lanes() == 1) {
      div_factor = (t.bits() == 4) ? 2 : (32 / t.bits());
    }
    index_str =
        PrintExpr(sym::Analyzer()->Simplify(truncdiv(index, div_factor)));
    os << "*((" << ptr_cast(t) << vid << ")" << " + " << index_str << ")";
  } else if (t == buffer_element_dtype) {
    int div_factor = 1;
    if (is_compact_scalar_fp4) {
      div_factor = 2;
    }
    index_str =
        PrintExpr(sym::Analyzer()->Simplify(truncdiv(index, div_factor)));
    os << buffer_str << "[" << index_str << "]";
  } else {
    // Fix fp4 pointer arithmetic: fp4 elements are 4-bit packed 2 per byte.
    // fp4* + n incorrectly advances n bytes (skipping 2n elements).
    int div_factor = 1;
    if (is_compact_scalar_fp4) {
      div_factor = 2;
    }
    index_str =
        PrintExpr(sym::Analyzer()->Simplify(truncdiv(index, div_factor)));
    os << "*" << ptr_cast(t) << "(" << buffer_str << " + " << index_str << ")";
  }

  return os.str();
}

std::string CodeGenTileLangCUDA::GetVecLoad(const PrimType &t,
                                            const VarObj *buffer,
                                            PrimExpr base) {
  std::string scope = CodegenStorageScope(ffi::GetRef<Var>(buffer));
  if (auto it = alloc_storage_scope_.find(buffer);
      it != alloc_storage_scope_.end()) {
    scope = it->second.name();
  }
  const int bits = t.bits() * t.lanes();
  // As in the original CUDA emitter, vector accesses require aligned storage.
  if (scope != "global" || bits <= 128) {
    return CodeGenC::GetVecLoad(t, buffer, base);
  }
  ICHECK_EQ(bits, 256) << "Unsupported vector load size: " << bits;
  need_copy_sm100_h_ = true;
  return "tl::load_global_256(&(" + GetBufferRef(t, buffer, base) + "))";
}

void CodeGenTileLangCUDA::PrintVecStore(const VarObj *buffer, const PrimType &t,
                                        PrimExpr base,
                                        const std::string &value) {
  std::string scope = CodegenStorageScope(ffi::GetRef<Var>(buffer));
  if (auto it = alloc_storage_scope_.find(buffer);
      it != alloc_storage_scope_.end()) {
    scope = it->second.name();
  }
  const int bits = t.bits() * t.lanes();
  if (scope != "global" || bits <= 128) {
    CodeGenC::PrintVecStore(buffer, t, base, value);
    return;
  }
  ICHECK_EQ(bits, 256) << "Unsupported vector store size: " << bits;
  need_copy_sm100_h_ = true;
  const std::string ref = GetBufferRef(t, buffer, base);
  PrintIndent();
  stream << "tl::store_global_256(&(" << ref << "), " << value << ");\n";
}

/**
 * @brief Emit CUDA/TensorLib-specific code for a call expression.
 *
 * This visitor handles CallObj intrinsics and builtins that require emitting
 * CUDA/TL-specific code (inline PTX/ASM sequences, TensorLanguage runtime
 * calls, WMMA/TMA helpers, barriers, cp.async primitives, index-map based
 * stores, reinterpret/packing helpers, and various mma/ldmatrix patterns). The
 * function writes the generated code to the provided output stream and falls
 * back to the C codegen for unrecognized calls.
 *
 * The method recognizes and emits code for (non-exhaustive): cp.async and its
 * commit/wait variants, tma_load/store and im2col variants, ptX
 * ldmatrix/stmatrix helpers, mbarrier APIs, cooperative grid sync, WMMA/legacy
 * MMA intrinsics (fill/load/store/mma/bmma/ptx_mma/ptx_mma_sp), low-level PTX
 * asm helpers (ldg32, cp_async bulk/init/arrive/wait barriers), reinterpret
 * paths for special small-float encodings (e.g., float4 e2m1fn)
 * and related external calls, and other TL runtime calls.
 *
 * Side effects:
 * - Emits to `os` and the internal codegen output stream.
 * - May set internal feature flags (e.g., need_cooperative_groups_,
 * need_mma_h_, need_cast_smem_ptr_to_int_, enable_sparse_gemm_).
 * - May open/close SSA scopes and mutate internal variable mappings.
 * - May call LOG(FATAL) / ICHECK on invalid or unsupported argument
 *   patterns.
 *
 * @param op The call node to generate code for; the function inspects op->op
 *           and op->args to determine the appropriate emission.
 * @param os  Output stream to receive expression-level output when the caller
 *            expects an expression result (some paths write directly to the
 *            member stream instead).
 */
std::string CodeGenTileLangCUDA::BarrierReference(const Expr &barrier) {
  if (const auto *id = barrier.as<IntImmObj>()) {
    int64_t index = CodegenInt(id);
    TVM_FFI_CHECK(barrier_count_ > 0 && !legacy_barrier_name_.empty(),
                  ValueError)
        << "Legacy barrier ID requires create_barriers in the same kernel";
    TVM_FFI_CHECK(index >= 0 && index < barrier_count_, ValueError)
        << "Legacy barrier ID out of bounds: " << index << " for "
        << barrier_count_ << " barriers";
    return legacy_barrier_name_ + "[" + std::to_string(index) + "]";
  }
  if (const auto *load = barrier.as<TensorLoadObj>()) {
    auto tensor = load->source.as_or_throw<TensorVar>();
    TVM_FFI_CHECK(tensor.dtype() == PrimType::UInt(64) &&
                      (tensor.scope().name() == "shared.barrier" ||
                       tensor.scope().name() == "shared.cluster_barrier"),
                  ValueError)
        << "Barrier element must be a uint64 shared barrier tensor load";
    return PrintExpr(barrier);
  }
  if (const auto *ptr = barrier->ty.as<PtrTypeObj>()) {
    TVM_FFI_CHECK(ptr->scope.name().find("shared") == 0, ValueError)
        << "Barrier pointer must carry shared-memory scope";
    return "(*reinterpret_cast<Barrier*>(" + PrintExpr(barrier) + "))";
  }
  TVM_FFI_THROW(TypeError) << "Barrier must be a constant legacy ID, a shared "
                              "barrier element, or a shared pointer";
}

void CodeGenTileLangCUDA::Dispatch_(const CallObj *op, std::ostream &os) {
  static const Op cuda_bmma_sync_op = Op::Get("tirx.cuda.bmma_sync");
  static const Op cuda_mma_fill_op = Op::Get("tirx.cuda.mma_fill");
  static const Op cuda_mma_store_op = Op::Get("tirx.cuda.mma_store");
  static const Op gpu_fill_fragment_op = Op::Get("tirx.gpu_fill_fragment");
  static const Op gpu_load_matrix_sync_op =
      Op::Get("tirx.gpu_load_matrix_sync");
  static const Op gpu_mma_sync_op = Op::Get("tirx.gpu_mma_sync");
  static const Op gpu_store_matrix_sync_op =
      Op::Get("tirx.gpu_store_matrix_sync");
  static const Op ptx_legacy_ldmatrix_op = Op::Get("tirx.ptx_legacy.ldmatrix");
  static const Op ptx_legacy_mma_op = Op::Get("tirx.ptx_legacy.mma");
  const auto args = op->args;
  if (IsUnpackedFP4SharedPointer(op->ty)) {
    if (op->op.same_as(tirx::ptr_byte_offset_op())) {
      TVM_FFI_CHECK_EQ(args.size(), 2U, ValueError);
      // The operand already counts bytes. In particular, unpacked FP4 must
      // not inherit the packed global FP4 division by two.
      os << "((uint8_t*)(((char*)" << PrintExpr(args[0]) << ") + "
         << PrintExpr(args[1]) << "))";
      return;
    }
    if (op->op.same_as(tirx::reinterpret_op())) {
      TVM_FFI_CHECK_EQ(args.size(), 1U, ValueError);
      os << "((uint8_t*)" << PrintExpr(args[0]) << ")";
      return;
    }
  }
  if (op->op.same_as(tl::ptr_access())) {
    TVM_FFI_CHECK_EQ(args.size(), 4, ValueError);
    TVM_FFI_CHECK(args[0]->ty.as<PtrTypeObj>(), TypeError);
    os << "(" << PrintExpr(args[0]) << " + " << PrintExpr(args[1]) << ")";
    return;
  }

  const auto *attrs = op->attrs.as<DictAttrsObj>();
  if (op->op.same_as(tl::cast_op())) {
    auto attrs = op->attrs.value().as_or_throw<DictAttrs>();
    auto rounding = attrs.GetAttr<String>("round").value_or("");
    auto saturate = attrs.GetAttr<bool>("sat").value_or(true);
    Optional<PrimExpr> random_bits =
        args.size() == 2 ? Optional<PrimExpr>(args[1].as_or_throw<PrimExpr>())
                         : std::nullopt;
    PrintCast(args[0].as_or_throw<PrimExpr>(), op->ty.as_or_throw<PrimType>(),
              rounding, saturate, random_bits, os);
    return;
  }

  auto print_extern_call_stmt = [&](std::string name, size_t start = 0,
                                    size_t end = 0) {
    // Cache context into a private ss, otherwise the let node may generate
    // within the function call arguments.
    std::ostringstream ss;

    for (size_t i = start; i < args.size() - end; i++) {
      if (i > start)
        ss << ", ";
      ss << this->PrintExpr(args[i]);
    }

    this->PrintIndent();
    this->stream << name << "(";
    this->stream << ss.str();
    this->stream << ");\n";
  };
  auto print_extern_call_expr = [&](std::ostream &os, std::string name,
                                    size_t start = 0, size_t end = 0) {
    os << name << "(";
    for (size_t i = start; i < args.size() - end; i++) {
      if (i > start)
        os << ", ";
      os << this->PrintExpr(args[i]);
    }
    os << ")";
  };
  if (op->op.same_as(tl::max_nan()) || op->op.same_as(tl::min_nan())) {
    ICHECK_EQ(args.size(), 2);
    const bool is_max = op->op.same_as(tl::max_nan());
    const PrimType t = op->ty.as_or_throw<PrimType>();
    const char *f16_intrin = is_max ? "__hmax_nan" : "__hmin_nan";
    const char *fallback = is_max ? "tl::fast_max" : "tl::fast_min";

    if (t.MatchesElementType(kDLBfloat, 16) && t.IsScalar()) {
      os << "cutlass::bfloat16_t(" << f16_intrin << "(" << "("
         << PrintExpr(args[0]) << ").to_nv_bfloat16(), " << "("
         << PrintExpr(args[1]) << ").to_nv_bfloat16()))";
      return;
    }
    if (t.MatchesElementType(kDLFloat, 16) && t.IsScalar()) {
      os << "cutlass::half_t(" << f16_intrin << "(" << "(" << PrintExpr(args[0])
         << ").to_half(), " << "(" << PrintExpr(args[1]) << ").to_half()))";
      return;
    }
    os << fallback << "(" << PrintExpr(args[0]) << ", " << PrintExpr(args[1])
       << ")";
    return;
  }
  if (op->op.same_as(tl::ptx_cp_async_bytes())) {
    need_copy_h_ = true;
    // args[0] = dst_access_ptr, args[1] = src_access_ptr, args[2] = bytes,
    // args[3] = predicate (optional)
    ICHECK(args.size() == 3 || args.size() == 4)
        << "ptx_cp_async expects 3 or 4 arguments (dst_access_ptr, "
           "src_access_ptr, bytes, [predicate])";

    std::string dst = this->PrintExpr(args[0]);
    std::string src = this->PrintExpr(args[1]);
    std::string size = this->PrintExpr(args[2]);

    this->PrintIndent();
    if (args.size() == 3) {
      // Non-predicated version
      this->stream << "tl::cp_async_gs<" << size << ">(" << dst << ", " << src
                   << ");\n";
    } else {
      // Predicated version
      std::string condition = this->PrintExpr(args[3]);
      this->stream << "tl::cp_async_gs_conditional<" << size << ">(" << dst
                   << ", " << src << ", " << condition << ");\n";
    }
  } else if (op->op.same_as(tl::ptx_cp_async())) {
    need_copy_h_ = true;
    // TileLang version: args[0] = dst_access_ptr, args[1] = src_access_ptr,
    // args[2] = num_elems, args[3] = predicate (optional)
    int total_bytes = GetTileLangCPAsyncTransferBytes(op);

    std::string dst = this->PrintExpr(args[0]);
    std::string src = this->PrintExpr(args[1]);
    std::string size = std::to_string(total_bytes);

    this->PrintIndent();
    if (args.size() == 3) {
      // Non-predicated version
      this->stream << "tl::cp_async_gs<" << size << ">(" << dst << ", " << src
                   << ");\n";
    } else {
      // Predicated version
      std::string condition = this->PrintExpr(args[3]);
      this->stream << "tl::cp_async_gs_conditional<" << size << ">(" << dst
                   << ", " << src << ", " << condition << ");\n";
    }
  } else if (op->op.same_as(tl::ptx_commit_group())) {
    need_copy_h_ = true;
    print_extern_call_stmt("tl::cp_async_commit");
  } else if (op->op.same_as(tl::ptx_wait_group())) {
    need_copy_h_ = true;
    int n = CodegenInt(args[0]);
    std::string func_name = "tl::cp_async_wait<" + std::to_string(n) + ">";
    print_extern_call_stmt(func_name, 1);
  } else if (op->op.same_as(tl::create_barriers())) {
    need_barrier_h_ = true;
    this->PrintIndent();
    TVM_FFI_CHECK_EQ(barrier_count_, -1, ValueError)
        << "Only one legacy create_barriers is allowed per kernel";
    barrier_count_ = CodegenInt(args[0]);
    TVM_FFI_CHECK_GT(barrier_count_, 0, ValueError);
    legacy_barrier_name_ = name_supply_->FreshName("mbarrier");
    auto mbarrier_storage_name =
        name_supply_->FreshName(legacy_barrier_name_ + "_mem");
    this->stream << "__shared__ __align__(" << barrier_alignment_bytes_
                 << ") uint64_t " << mbarrier_storage_name << "["
                 << barrier_count_ << "];\n";
    this->PrintIndent();
    this->stream << "auto " << legacy_barrier_name_ << " = reinterpret_cast<"
                 << mbarrier_dtype_ << "*>(" << mbarrier_storage_name << ");\n";
  } else if (op->op.same_as(tl::ptx_arrive_barrier())) {
    need_barrier_h_ = true;
    ICHECK_EQ(args.size(), 1);
    this->PrintIndent();
    auto mbarrier_obj = BarrierReference(args[0]);
    this->stream << mbarrier_obj << ".arrive();\n";
  } else if (op->op.same_as(tl::ptx_arrive_cluster_barrier())) {
    need_barrier_h_ = true;
    ICHECK_EQ(args.size(), 2);
    this->PrintIndent();
    auto mbarrier_obj = BarrierReference(args[0]);
    auto cta_id = this->PrintExpr(args[1]);
    if (args[1].as<IntImmObj>()) {
      cta_id += "u"; // Ensure cta_id as u32
    }
    this->stream << mbarrier_obj << ".arrive(" << cta_id << ");\n";
  } else if (op->op.same_as(tl::ptx_init_barrier_thread_count())) {
    need_barrier_h_ = true;
    ICHECK_EQ(args.size(), 2);
    this->PrintIndent();
    auto mbarrier_obj = BarrierReference(args[0]);
    auto arrive_count = this->PrintExpr(args[1]);
    this->stream << mbarrier_obj << ".init(" << arrive_count << ");\n";
  } else if (op->op.same_as(tl::ptx_arrive_barrier_expect_tx())) {
    need_barrier_h_ = true;
    if (args.size() == 2) {
      this->PrintIndent();
      auto mbarrier_obj = BarrierReference(args[0]);
      auto transaction_bytes = this->PrintExpr(args[1]);
      this->stream << mbarrier_obj << ".arrive_and_expect_tx("
                   << transaction_bytes << ");\n";
    } else if (args.size() == 4) {
      this->PrintIndent();
      auto mbarrier_obj = BarrierReference(args[0]);
      auto transaction_bytes = this->PrintExpr(args[1]);
      auto cta_id = this->PrintExpr(args[2]);
      auto pred = this->PrintExpr(args[3]);
      this->stream << mbarrier_obj << ".arrive_and_expect_tx("
                   << transaction_bytes << ", " << cta_id << ", " << pred
                   << ");\n";
    } else {
      TVM_FFI_THROW(InternalError)
          << "Invalid parameter  for tl::arrive_barrier_expect_tx "
          << args.size();
    }
  } else if (op->op.same_as(tl::ptx_cp_async_barrier())) {
    need_barrier_h_ = true;
    PrintIndent();
    stream << "tl::mbarrier_cp_async_arrive(" << BarrierReference(args[0])
           << ");\n";
  } else if (op->op.same_as(tl::ptx_fence_barrier_init())) {
    need_barrier_h_ = true;
    print_extern_call_stmt("tl::fence_barrier_init");
  } else if (op->op.same_as(tl::ptx_cp_async_barrier_noinc())) {
    need_barrier_h_ = true;
    PrintIndent();
    stream << "tl::mbarrier_cp_async_arrive_noinc(" << BarrierReference(args[0])
           << ");\n";
  } else if (op->op.same_as(tl::mbarrier_expect_tx())) {
    need_barrier_h_ = true;
    ICHECK_EQ(args.size(), 2);
    this->PrintIndent();
    auto mbarrier_obj = BarrierReference(args[0]);
    auto transaction_bytes = this->PrintExpr(args[1]);
    this->stream << mbarrier_obj << ".expect_transaction(" << transaction_bytes
                 << ");\n";
  } else if (op->op.same_as(tl::mbarrier_wait_parity())) {
    need_barrier_h_ = true;
    ICHECK_EQ(args.size(), 2);
    this->PrintIndent();
    auto mbarrier_obj = BarrierReference(args[0]);
    auto phase = this->PrintExpr(args[1]);
    this->stream << mbarrier_obj << ".wait(" << phase << ");\n";
  } else if (op->op.same_as(tl::ptx_init_tensor_memory())) {
    need_tcgen05_common_h_ = true;
    std::ostringstream ss;
    ss << "tl::tmem_allocate";
    if (attrs && CodegenBool(attrs->dict.Get("use_2cta").value_or(false))) {
      ss << "<true>";
    }
    print_extern_call_stmt(ss.str());
  } else if (op->op.same_as(tl::ptx_deallocate_tensor_memory())) {
    need_tcgen05_common_h_ = true;
    std::ostringstream ss;
    ss << "tl::tmem_deallocate";
    if (attrs && CodegenBool(attrs->dict.Get("use_2cta").value_or(false))) {
      ss << "<true>";
    }
    print_extern_call_stmt(ss.str());
  } else if (op->op.same_as(tl::no_set_max_nreg())) {
    return;
  } else if (op->op.same_as(tl::prefetch_tma_descriptor())) {
    need_copy_sm90_h_ = true;
    print_extern_call_stmt("tl::prefetch_tma_descriptor");
  } else if (op->op.same_as(tl::tma_load())) {
    std::ostringstream ss;
    ICHECK_GE(args.size(), 2);
    auto eviction_policy =
        this->eviction_policy_names_[CodegenInt(args[args.size() - 1])];
    // Simplify the code by using the default eviction policy
    if (attrs && CodegenBool(attrs->dict.Get("use_2cta").value_or(false))) {
      need_copy_sm100_h_ = true;
      if (eviction_policy != "EVICT_NORMAL") {
        ss << "tl::tma_load_2sm<tl::CacheHintSm100::" << eviction_policy
           << ">(";
      } else {
        ss << "tl::tma_load_2sm(";
      }
    } else {
      need_copy_sm90_h_ = true;
      if (eviction_policy != "EVICT_NORMAL") {
        ss << "tl::tma_load<tl::CacheHintSm90::" << eviction_policy << ">(";
      } else {
        ss << "tl::tma_load(";
      }
    }
    auto desc = args[0];
    ss << this->PrintExpr(desc) << ", ";
    ss << this->PrintExpr(args[1]) << ", ";
    for (size_t i = 2; i < args.size() - 1; i++) {
      if (i > 2)
        ss << ", ";
      ss << this->PrintExpr(args[i]);
    }
    ss << ");\n";
    this->PrintIndent();
    this->stream << ss.str();
  } else if (op->op.same_as(tl::tma_load_multicast())) {
    need_copy_sm90_h_ = true;
    std::ostringstream ss;
    ICHECK_GE(args.size(), 5) << "tma_load_multicast requires at least 5 args";
    auto eviction_policy =
        this->eviction_policy_names_[CodegenInt(args[args.size() - 1])];
    if (eviction_policy != "EVICT_NORMAL") {
      ss << "tl::tma_load_multicast<tl::CacheHintSm90::" << eviction_policy
         << ">(";
    } else {
      ss << "tl::tma_load_multicast(";
    }
    ss << this->PrintExpr(args[0]) << ", ";
    ss << this->PrintExpr(args[1]) << ", ";
    ss << this->PrintExpr(args[2]) << ", ";
    ss << "(uint16_t)(" << this->PrintExpr(args[3]) << ")";
    for (size_t i = 4; i < args.size() - 1; i++) {
      ss << ", " << this->PrintExpr(args[i]);
    }
    ss << ");\n";
    this->PrintIndent();
    this->stream << ss.str();
  } else if (op->op.same_as(tl::tma_load_im2col())) {
    need_copy_sm90_h_ = true;
    std::stringstream ss;
    auto eviction_policy =
        this->eviction_policy_names_[CodegenInt(args[args.size() - 1])];
    if (eviction_policy != "EVICT_NORMAL") {
      ss << "tl::tma_load_im2col<tl::CacheHintSm90::" << eviction_policy << ">";
    } else {
      ss << "tl::tma_load_im2col";
    }
    print_extern_call_stmt(ss.str(), 0, 1);
  } else if (op->op.same_as(tl::tma_store())) {
    need_copy_sm90_h_ = true;
    std::stringstream ss;
    auto need_reduce = CodegenInt(args[args.size() - 2]);
    if (need_reduce) {
      print_extern_call_stmt("tl::tma_store_add", 0, 2);
      return;
    }
    auto eviction_policy =
        this->eviction_policy_names_[CodegenInt(args[args.size() - 1])];
    if (eviction_policy != "EVICT_NORMAL") {
      ss << "tl::tma_store<tl::CacheHintSm90::" << eviction_policy << ">";
    } else {
      ss << "tl::tma_store";
    }
    print_extern_call_stmt(ss.str(), 0, 2);
  } else if (op->op.same_as(tl::tma_load_gather4())) {
    need_copy_sm100_h_ = true;
    std::ostringstream ss;
    ICHECK_EQ(args.size(), 9u)
        << "tma_load_gather4 expects 9 args (desc, mbar, smem, col, "
           "r0..r3, eviction_policy), got "
        << args.size();
    auto eviction_policy =
        this->eviction_policy_names_[CodegenInt(args[args.size() - 1])];
    if (eviction_policy != "EVICT_NORMAL") {
      ss << "tl::tma_load_gather4<tl::CacheHintSm90::" << eviction_policy
         << ">";
    } else {
      ss << "tl::tma_load_gather4";
    }
    print_extern_call_stmt(ss.str(), 0, 1);
  } else if (op->op.same_as(tl::tma_store_scatter4())) {
    need_copy_sm100_h_ = true;
    std::ostringstream ss;
    ICHECK_EQ(args.size(), 8u)
        << "tma_store_scatter4 expects 8 args (desc, smem, col, r0..r3, "
           "eviction_policy), got "
        << args.size();
    auto eviction_policy =
        this->eviction_policy_names_[CodegenInt(args[args.size() - 1])];
    if (eviction_policy != "EVICT_NORMAL") {
      ss << "tl::tma_store_scatter4<tl::CacheHintSm90::" << eviction_policy
         << ">";
    } else {
      ss << "tl::tma_store_scatter4";
    }
    print_extern_call_stmt(ss.str(), 0, 1);
  } else if (op->op.same_as(tl::ptx_ldmatrix())) {
    ICHECK_EQ(args.size(), 4U);
    int trans = CodegenInt(args[0]);
    int num = CodegenInt(args[1]);
    std::string func_name = "tl::ptx_ldmatrix_x" + std::to_string(num);
    if (trans == 1)
      func_name += "_trans";
    print_extern_call_stmt(func_name, 2);
  } else if (op->op.same_as(tl::ptx_stmatrix())) {
    int trans = CodegenInt(args[0]);
    int num = CodegenInt(args[1]);
    std::string shape = "m8n8";
    bool is_shape_encoded = args.size() >= 4 && args.back().as<StringImmObj>();
    if (is_shape_encoded) {
      shape = (args.back()).as_or_throw<StringImm>()->value;
    }
    std::string func_name =
        "tl::ptx_stmatrix_" + shape + "_x" + std::to_string(num);
    if (trans == 1)
      func_name += "_trans";
    print_extern_call_stmt(func_name, 2, is_shape_encoded ? 1 : 0);
  } else if (op->op.same_as(tl::fence_proxy_async())) {
    need_barrier_h_ = true;
    print_extern_call_stmt("tl::fence_proxy_async");
  } else if (op->op.same_as(tl::tma_store_arrive())) {
    need_barrier_h_ = true;
    print_extern_call_stmt("tl::tma_store_arrive");
  } else if (op->op.same_as(tl::tma_store_wait())) {
    need_barrier_h_ = true;
    int count = CodegenInt(args[0]);
    bool read = args.size() == 1 || CodegenInt(args[1]) != 0;
    this->PrintIndent();
    this->stream << "tl::tma_store_wait<" << count << ", "
                 << (read ? "true" : "false") << ">();\n";
  } else if (op->op.same_as(tl::warpgroup_arrive())) {
    need_intrin_h_ = true;
    print_extern_call_stmt("tl::warpgroup_arrive");
  } else if (op->op.same_as(tl::warpgroup_commit_batch())) {
    need_intrin_h_ = true;
    print_extern_call_stmt("tl::warpgroup_commit_batch");
  } else if (op->op.same_as(tl::warpgroup_wait())) {
    need_intrin_h_ = true;
    this->PrintIndent();
    int num_mma = CodegenInt(args[0]);
    this->stream << "tl::warpgroup_wait<" << std::to_string(num_mma)
                 << ">();\n";
  } else if (op->op.same_as(tl::warpgroup_fence_operand())) {
    need_intrin_h_ = true;
    ICHECK_EQ(args.size(), 4U);
    std::string dtype = (args[0]).as_or_throw<StringImm>()->value;
    std::string data_ptr = this->PrintExpr(args[1]);
    std::string offset = this->PrintExpr(args[2]);
    std::string num_regs = this->PrintExpr(args[3]);
    auto dtype_enum = tl::codegen::ptx::DTypeFromString(dtype);
    std::string cast_type = "uint32_t";
    if (dtype_enum == tl::codegen::ptx::DataType::kFloat32 ||
        dtype_enum == tl::codegen::ptx::DataType::kTensorFloat32) {
      cast_type = "float";
    }
    this->PrintIndent();
    this->stream << "tl::warpgroup_fence_operand(reinterpret_cast<" << cast_type
                 << "*>(" << data_ptr << " + " << offset << "), " << num_regs
                 << ");\n";
  } else if (op->op.same_as(tl::set_max_nreg())) {
    need_intrin_h_ = true;
    this->PrintIndent();
    int nreg = CodegenInt(args[0]);
    int is_inc = CodegenInt(args[1]);
    std::string func_name =
        is_inc ? "tl::warpgroup_reg_alloc" : "tl::warpgroup_reg_dealloc";
    this->stream << func_name << "<" << std::to_string(nreg) << ">();\n";
  } else if (op->op.same_as(tl::annotate_producer_reg_dealloc()) ||
             op->op.same_as(tl::annotate_consumer_reg_alloc())) {
    return;
  } else if (op->op.same_as(tl::wait_wgmma())) {
    need_intrin_h_ = true;
    this->PrintIndent();
    int num_mma = CodegenInt(args[0]);
    this->stream << "tl::wait_wgmma<" << std::to_string(num_mma) << ">();\n";
  } else if (op->op.same_as(tl::pack_b16())) {
    os << "__pack_half2(" << this->PrintExpr(args[0]) << ", "
       << this->PrintExpr(args[1]) << ")";
  } else if (op->op.same_as(tl::pack_b8x4())) {
    print_extern_call_expr(os, "pack_b8x4");
  } else if (op->op.same_as(tl::sync_grid())) {
    this->need_cooperative_groups_ = true;
    this->PrintIndent();
    this->stream << "cooperative_groups::this_grid().sync();\n";
  } else if (op->op.same_as(tl::sync_warp())) {
    this->PrintIndent();
    this->stream << "__syncwarp(";
    if (!args.empty()) {
      this->stream << this->PrintExpr(args[0]);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::named_barrier_arrive())) {
    ICHECK_EQ(args.size(), 2U)
        << "tl.named_barrier_arrive expects <barrier_id, thread_count>.";
    this->PrintIndent();
    this->stream << "tl::__named_barrier_arrive(" << this->PrintExpr(args[0])
                 << ", " << this->PrintExpr(args[1]) << ");\n";
  } else if (op->op.same_as(tl::pdl_trigger())) {
    this->PrintIndent();
    this->stream << "cudaTriggerProgrammaticLaunchCompletion();\n";
  } else if (op->op.same_as(tl::pdl_sync())) {
    this->PrintIndent();
    this->stream << "cudaGridDependencySynchronize();\n";
  } else if (op->op.same_as(tl::cluster_arrive_relaxed())) {
    need_cluster_h_ = true;
    this->PrintIndent();
    this->stream << "tl::cluster_arrive_relaxed();\n";
  } else if (op->op.same_as(tl::cluster_arrive())) {
    need_cluster_h_ = true;
    this->PrintIndent();
    this->stream << "tl::cluster_arrive();\n";
  } else if (op->op.same_as(tl::cluster_wait())) {
    need_cluster_h_ = true;
    this->PrintIndent();
    this->stream << "tl::cluster_wait();\n";
  } else if (op->op.same_as(tl::cluster_sync())) {
    need_cluster_h_ = true;
    this->PrintIndent();
    this->stream << "tl::cluster_sync();\n";
  } else if (op->op.same_as(tl::block_rank_in_cluster())) {
    need_cluster_h_ = true;
    os << "tl::block_rank_in_cluster()";
  } else if (op->op.same_as(tl::clc_try_cancel())) {
    need_cluster_h_ = true;
    print_extern_call_stmt("tl::clc_try_cancel");
  } else if (op->op.same_as(tl::clc_try_cancel_multicast())) {
    need_cluster_h_ = true;
    print_extern_call_stmt("tl::clc_try_cancel_multicast");
  } else if (op->op.same_as(tl::clc_is_canceled())) {
    need_cluster_h_ = true;
    os << "tl::clc_is_canceled(" << this->PrintExpr(args[0]) << ")";
  } else if (op->op.same_as(tl::clc_get_first_ctaid_x())) {
    need_cluster_h_ = true;
    os << "tl::clc_get_first_ctaid_x(" << this->PrintExpr(args[0]) << ")";
  } else if (op->op.same_as(tl::clc_get_first_ctaid_y())) {
    need_cluster_h_ = true;
    os << "tl::clc_get_first_ctaid_y(" << this->PrintExpr(args[0]) << ")";
  } else if (op->op.same_as(tl::clc_get_first_ctaid_z())) {
    need_cluster_h_ = true;
    os << "tl::clc_get_first_ctaid_z(" << this->PrintExpr(args[0]) << ")";
  } else if (op->op.same_as(tl::loop_break())) {
    this->PrintIndent();
    this->stream << "break;\n";
  } else if (op->op.same_as(gpu_fill_fragment_op)) {
    need_mma_h_ = true;
    ICHECK_EQ(args.size(), 6U);
    os << "nvcuda::wmma::fill_fragment(";
    this->PrintExpr(args[0], os);
    os << "[";
    this->PrintExpr(args[4], os);
    os << "], ";
    this->PrintExpr(args[5], os);
    os << ")";
  } else if (op->op.same_as(tl::wmma_load_matrix_sync()) ||
             op->op.same_as(gpu_load_matrix_sync_op)) {
    need_mma_h_ = true;
    ICHECK_EQ(args.size(), 8U);
    os << "nvcuda::wmma::load_matrix_sync(";
    this->PrintExpr(args[0], os);
    os << "[";
    this->PrintExpr(args[4], os);
    os << "], ";
    this->PrintExpr(args[5], os);
    os << ", ";
    this->PrintExpr(args[6], os);
    os << ")";
  } else if (op->op.same_as(gpu_store_matrix_sync_op)) {
    need_mma_h_ = true;
    ICHECK_EQ(args.size(), 8U);
    os << "nvcuda::wmma::store_matrix_sync(";
    this->PrintExpr(args[5], os);
    os << ", ";
    this->PrintExpr(args[0], os);
    os << "[";
    this->PrintExpr(args[4], os);
    os << "], ";
    this->PrintExpr(args[6], os);
    if (const StringImmObj *str = args[7].as<StringImmObj>()) {
      os << ", nvcuda::wmma::mem_" << str->value;
    } else {
      TVM_FFI_THROW(InternalError) << "Invalid parameters";
    }
    os << ")";
  } else if (op->op.same_as(gpu_mma_sync_op)) {
    need_mma_h_ = true;
    ICHECK_EQ(args.size(), 8U);
    os << "nvcuda::wmma::mma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(args[i * 2], os);
      os << "[";
      this->PrintExpr(args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (op->op.same_as(cuda_bmma_sync_op)) {
    need_mma_h_ = true;
    ICHECK_EQ(args.size(), 8U);
    os << "nvcuda::wmma::bmma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(args[i * 2], os);
      os << "[";
      this->PrintExpr(args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (op->op.same_as(ptx_legacy_mma_op)) {
    // arg 0: shape: mXnXkX
    // arg 1: A layout: row/col
    // arg 2: B layout: row/col
    // arg 3: A precision: fp16, fp64, ...
    // arg 4: B precision: fp16, fp64, ...
    // arg 5: C precision: fp32, fp64, ...
    // arg 6: A multiplicand
    // arg 7: A multiplicand index
    // arg 8: B multiplicand
    // arg 9: B multiplicand index
    // arg 10: C accumulator
    // arg 11: C accumulator index
    // arg 12: saturate
    // arg 13: (optional) 1-bit operator (xor or and)
    ICHECK(args.size() == 13U || args.size() == 14U);
    std::string shape = (args[0]).as_or_throw<StringImm>()->value;
    std::string A_layout = (args[1]).as_or_throw<StringImm>()->value;
    std::string B_layout = (args[2]).as_or_throw<StringImm>()->value;
    std::string A_dtype = (args[3]).as_or_throw<StringImm>()->value;
    std::string B_dtype = (args[4]).as_or_throw<StringImm>()->value;
    std::string C_dtype = (args[5]).as_or_throw<StringImm>()->value;
    std::string a_ref = this->PrintExpr(args[6]);
    std::string b_ref = this->PrintExpr(args[8]);
    std::string c_ref = this->PrintExpr(args[10]);
    std::string c_bias = this->PrintExpr(args[11]);
    auto dtype_a_enum = tl::codegen::ptx::DTypeFromString(A_dtype);
    auto dtype_b_enum = tl::codegen::ptx::DTypeFromString(B_dtype);
    auto dtype_c_enum = tl::codegen::ptx::DTypeFromString(C_dtype);
    PrimExpr a_bias_expr = args[7].as_or_throw<PrimExpr>();
    PrimExpr b_bias_expr = args[9].as_or_throw<PrimExpr>();
    if (dtype_a_enum == tl::codegen::ptx::DataType::kInt4 ||
        dtype_a_enum == tl::codegen::ptx::DataType::kUInt4) {
      a_bias_expr = sym::Analyzer()->Simplify(truncdiv(a_bias_expr, 2));
    }
    if (dtype_b_enum == tl::codegen::ptx::DataType::kInt4 ||
        dtype_b_enum == tl::codegen::ptx::DataType::kUInt4) {
      b_bias_expr = sym::Analyzer()->Simplify(truncdiv(b_bias_expr, 2));
    }
    std::string a_bias = this->PrintExpr(a_bias_expr);
    std::string b_bias = this->PrintExpr(b_bias_expr);
    auto [m, n, k] = tl::codegen::ptx::ParseMMAShape(shape);

    need_mma_instruction_h_ = true;
    this->PrintIndent();
    std::string mma_call =
        "tl::mma_sync<(AType), (BType), (CType), (M), (N), (K), (TransA), "
        "(TransB)>(reinterpret_cast<(CRegType)*>((C_ptr) + (C_offset)), "
        "reinterpret_cast<const (ARegType)*>((A_ptr) + (A_offset)), "
        "reinterpret_cast<const (BRegType)*>((B_ptr) + (B_offset)));\n";
    tl::codegen::Replacer replacer;

    std::string AType = tl::codegen::ptx::DTypeEnumToString(dtype_a_enum);
    std::string BType = tl::codegen::ptx::DTypeEnumToString(dtype_b_enum);
    std::string ARegType = tl::codegen::GetMMARegisterType(dtype_a_enum);
    if (ARegType == "float") {
      ARegType = "uint32_t";
    }
    std::string BRegType = tl::codegen::GetMMARegisterType(dtype_b_enum);
    if (BRegType == "float") {
      BRegType = "uint32_t";
    }
    replacer.register_rule("(AType)", AType);
    replacer.register_rule("(BType)", BType);
    replacer.register_rule("(CType)",
                           tl::codegen::ptx::DTypeEnumToString(dtype_c_enum));
    replacer.register_rule("(M)", std::to_string(m));
    replacer.register_rule("(N)", std::to_string(n));
    replacer.register_rule("(K)", std::to_string(k));
    replacer.register_rule("(TransA)", A_layout == "row" ? "false" : "true");
    replacer.register_rule("(TransB)", B_layout == "row" ? "false" : "true");
    replacer.register_rule("(ARegType)", ARegType);
    replacer.register_rule("(BRegType)", BRegType);
    replacer.register_rule("(CRegType)",
                           tl::codegen::GetMMARegisterType(dtype_c_enum));
    replacer.register_rule("(A_ptr)", a_ref);
    replacer.register_rule("(A_offset)", a_bias);
    replacer.register_rule("(B_ptr)", b_ref);
    replacer.register_rule("(B_offset)", b_bias);
    replacer.register_rule("(C_ptr)", c_ref);
    replacer.register_rule("(C_offset)", c_bias);
    this->stream << replacer.rewrite(mma_call);
  } else if (op->op.same_as(tl::tma_store_cluster())) {
    need_copy_sm90_h_ = true;
    ICHECK_EQ(args.size(), 5U) << "tma_store_cluster requires 5 args";
    PrimExpr size_arg = args[3].as_or_throw<PrimExpr>();
    while (const auto *cast_node = size_arg.as<prim::CastObj>()) {
      size_arg = cast_node->value;
    }
    if (const auto *size_imm = size_arg.as<IntImmObj>()) {
      ICHECK_EQ(size_imm->value % 16, 0)
          << "tma_store_cluster transfer size (" << size_imm->value
          << " bytes) must be a multiple of 16 bytes as required by "
             "cp.async.bulk";
    }
    this->PrintIndent();
    this->stream << "tl::tma_store_cluster(";
    this->stream << this->PrintExpr(args[0]) << ", ";
    this->stream << this->PrintExpr(args[1]) << ", ";
    this->stream << "(int)(" << this->PrintExpr(args[2]) << "), ";
    this->stream << "(uint32_t)(" << this->PrintExpr(args[3]) << "), ";
    this->stream << this->PrintExpr(args[4]) << ");\n";

  } else if (op->op.same_as(tl::ptx_cluster_store())) {
    ICHECK_EQ(args.size(), 4U);
    std::string buffer_var = this->PrintExpr(args[0]);
    std::string value = this->PrintExpr(args[1]);
    std::string dst_block = this->PrintExpr(args[2]);
    std::string index = this->PrintExpr(args[3]);

    this->need_cooperative_groups_ = true;
    this->PrintIndent();
    this->stream << "{\n";
    int cluster_scope = this->BeginScope();
    this->PrintIndent();
    this->stream << "namespace cg = cooperative_groups;\n";
    this->PrintIndent();
    this->stream << "cg::cluster_group cluster = cg::this_cluster();\n";
    this->PrintIndent();
    this->stream << "auto* dst_ptr = cluster.map_shared_rank(&" << buffer_var
                 << "[" << index << "], " << dst_block << ");\n";
    this->PrintIndent();
    this->stream << "*dst_ptr = " << value << ";\n";
    this->EndScope(cluster_scope);
    this->PrintIndent();
    this->stream << "}\n";

  } else if (op->op.same_as(tl::ptx_mma_sm70())) {
    // arg 0: shape: mXnXkX
    // arg 1: A layout: row/col
    // arg 2: B layout: row/col
    // arg 3: A precision: fp16
    // arg 4: B precision: fp16
    // arg 5: C precision: fp16, fp32
    // arg 6: A multiplicand
    // arg 7: A multiplicand index
    // arg 8: B multiplicand
    // arg 9: B multiplicand index
    // arg 10: C accumulator
    // arg 11: C accumulator index
    // arg 12: saturate
    ICHECK_EQ(args.size(), 12U);
    std::string shape = (args[0]).as_or_throw<StringImm>()->value;
    std::string A_layout = (args[1]).as_or_throw<StringImm>()->value;
    std::string B_layout = (args[2]).as_or_throw<StringImm>()->value;
    std::string A_dtype = (args[3]).as_or_throw<StringImm>()->value;
    std::string B_dtype = (args[4]).as_or_throw<StringImm>()->value;
    std::string C_dtype = (args[5]).as_or_throw<StringImm>()->value;
    std::string a_ref = this->PrintExpr(args[6]);
    std::string a_bias = this->PrintExpr(args[7]);
    std::string b_ref = this->PrintExpr(args[8]);
    std::string b_bias = this->PrintExpr(args[9]);
    std::string c_ref = this->PrintExpr(args[10]);
    std::string c_bias = this->PrintExpr(args[11]);
    auto dtype_a_enum = tl::codegen::ptx::DTypeFromString(A_dtype);
    auto dtype_b_enum = tl::codegen::ptx::DTypeFromString(B_dtype);
    auto dtype_c_enum = tl::codegen::ptx::DTypeFromString(C_dtype);
    auto [m, n, k] = tl::codegen::ptx::ParseMMAShape(shape);

    need_mma_sm70_instruction_h_ = true;
    this->PrintIndent();
    std::string mma_call =
        "tl::mma_sync_sm70<(AType), (BType), (CType), (M), (N), (K), (TransA), "
        "(TransB)>(reinterpret_cast<(CRegType)*>((C_ptr) + (C_offset)), "
        "reinterpret_cast<const (ARegType)*>((A_ptr) + (A_offset)), "
        "reinterpret_cast<const (BRegType)*>((B_ptr) + (B_offset)));\n";
    tl::codegen::Replacer replacer;

    replacer.register_rule("(AType)",
                           tl::codegen::ptx::DTypeEnumToString(dtype_a_enum));
    replacer.register_rule("(BType)",
                           tl::codegen::ptx::DTypeEnumToString(dtype_b_enum));
    replacer.register_rule("(CType)",
                           tl::codegen::ptx::DTypeEnumToString(dtype_c_enum));
    replacer.register_rule("(M)", std::to_string(m));
    replacer.register_rule("(N)", std::to_string(n));
    replacer.register_rule("(K)", std::to_string(k));
    replacer.register_rule("(TransA)", A_layout == "row" ? "false" : "true");
    replacer.register_rule("(TransB)", B_layout == "row" ? "false" : "true");
    replacer.register_rule("(ARegType)",
                           tl::codegen::GetMMARegisterType(dtype_a_enum));
    replacer.register_rule("(BRegType)",
                           tl::codegen::GetMMARegisterType(dtype_b_enum));
    replacer.register_rule("(CRegType)",
                           tl::codegen::GetMMARegisterType(dtype_c_enum));
    replacer.register_rule("(A_ptr)", a_ref);
    replacer.register_rule("(A_offset)", a_bias);
    replacer.register_rule("(B_ptr)", b_ref);
    replacer.register_rule("(B_offset)", b_bias);
    replacer.register_rule("(C_ptr)", c_ref);
    replacer.register_rule("(C_offset)", c_bias);
    this->stream << replacer.rewrite(mma_call);
  } else if (op->op.same_as(tl::ptx_mma_block_scale())) {
    need_mma_block_scale_instruction_h_ = true;
    // arg 0: accum_dtype
    // arg 1: shape: mXnXkX
    // arg 2: A layout: row/col
    // arg 3: B layout: row/col
    // arg 4: kind: mxf4nvf4
    // arg 5: scale_vec_size: 4
    // arg 6: A dtype: e2m1
    // arg 7: B dtype: e2m1
    // arg 8: scale_type: ue4m3
    // arg 9: A_data pointer
    // arg 10: A_offset
    // arg 11: B_data pointer
    // arg 12: B_offset
    // arg 13: C_data pointer
    // arg 14: C_offset
    // arg 15: scale_a_data
    // arg 16: scale_b_data
    // arg 17: scale_a_byte_id
    // arg 18: scale_a_thread_id
    // arg 19: scale_b_byte_id
    // arg 20: scale_b_thread_id
    ICHECK_EQ(args.size(), 21U);
    std::string accum_dtype = (args[0]).as_or_throw<StringImm>()->value;
    std::string shape = (args[1]).as_or_throw<StringImm>()->value;
    std::string A_layout = (args[2]).as_or_throw<StringImm>()->value;
    std::string B_layout = (args[3]).as_or_throw<StringImm>()->value;
    std::string kind = (args[4]).as_or_throw<StringImm>()->value;
    int scale_vec_size = static_cast<int>(CodegenInt(args[5]));
    std::string A_dtype = (args[6]).as_or_throw<StringImm>()->value;
    std::string B_dtype = (args[7]).as_or_throw<StringImm>()->value;
    std::string scale_type = (args[8]).as_or_throw<StringImm>()->value;
    std::string a_ref = this->PrintExpr(args[9]);
    std::string a_offset = this->PrintExpr(args[10]);
    std::string b_ref = this->PrintExpr(args[11]);
    std::string b_offset = this->PrintExpr(args[12]);
    std::string c_ref = this->PrintExpr(args[13]);
    std::string c_offset = this->PrintExpr(args[14]);
    std::string scale_a = this->PrintExpr(args[15]);
    std::string scale_b = this->PrintExpr(args[16]);
    std::string scale_a_byte_id = this->PrintExpr(args[17]);
    std::string scale_a_thread_id = this->PrintExpr(args[18]);
    std::string scale_b_byte_id = this->PrintExpr(args[19]);
    std::string scale_b_thread_id = this->PrintExpr(args[20]);

    bool supported_mxf4nvf4_4x_ue4m3 =
        accum_dtype == "float32" && shape == "m16n8k64" && A_layout == "row" &&
        B_layout == "col" && kind == "mxf4nvf4" && scale_vec_size == 4 &&
        A_dtype == "e2m1" && B_dtype == "e2m1" && scale_type == "ue4m3";
    ICHECK(supported_mxf4nvf4_4x_ue4m3)
        << "Unsupported ptx_mma_block_scale configuration: accum_dtype="
        << accum_dtype << ", shape=" << shape << ", A_layout=" << A_layout
        << ", B_layout=" << B_layout << ", kind=" << kind
        << ", scale_vec_size=" << scale_vec_size << ", A_dtype=" << A_dtype
        << ", B_dtype=" << B_dtype << ", scale_type=" << scale_type
        << ". Currently supported: f32 m16n8k64 row.col kind::mxf4nvf4 "
           "scale_vec::4X e2m1.e2m1 f32 ue4m3.";

    auto resolve_fp4_packed_buffer = [&](const Expr &var_expr, std::string &ref,
                                         std::string &offset) {
      const VarObj *var = var_expr.as<VarObj>();
      if (const auto *data = var_expr.as<CallObj>();
          data && data->op.same_as(tirx::tensor_data_ptr_op())) {
        // Native Tensor.data projects a pointer instead of returning the
        // tensor Var itself. Keep the allocation's identity when selecting
        // its packed backing and converting FP4 element offsets to bytes.
        ICHECK_EQ(data->args.size(), 1U);
        var = data->args[0].as<VarObj>();
      }
      if (var) {
        auto it = fp4_packed_buffers_.find(GetRef<Var>(var));
        if (it != fp4_packed_buffers_.end()) {
          ref = it->second;
          offset = "(" + offset + ") / 2";
        }
      }
    };
    resolve_fp4_packed_buffer(args[9], a_ref, a_offset);
    resolve_fp4_packed_buffer(args[11], b_ref, b_offset);

    this->PrintIndent();

    std::string mma_kind_enum = "tl::SM120MmaBlockScaledKind::kMxf4nvf4";
    std::string scale_type_enum = "tl::SM120MmaScaleType::kUE4M3";
    std::string mma_call =
        "tl::sm120_mma_sync_blockscaled<(MMA_KIND), (SCALE_VEC_SIZE), "
        "(SCALE_TYPE)>("
        "reinterpret_cast<float*>((C_ptr) + (C_offset)), "
        "reinterpret_cast<const uint32_t*>((A_ptr) + (A_offset)), "
        "reinterpret_cast<const uint32_t*>((B_ptr) + (B_offset)), "
        "reinterpret_cast<const float*>((C_ptr) + (C_offset)), "
        "(*reinterpret_cast<const uint32_t*>((scale_a))), "
        "(*reinterpret_cast<const uint32_t*>((scale_b))), "
        "static_cast<uint16_t>((scale_a_byte_id)), "
        "static_cast<uint16_t>((scale_a_thread_id)), "
        "static_cast<uint16_t>((scale_b_byte_id)), "
        "static_cast<uint16_t>((scale_b_thread_id)));\n";

    tl::codegen::Replacer replacer;
    replacer.register_rule("(MMA_KIND)", mma_kind_enum);
    replacer.register_rule("(SCALE_VEC_SIZE)", std::to_string(scale_vec_size));
    replacer.register_rule("(SCALE_TYPE)", scale_type_enum);
    replacer.register_rule("(A_ptr)", a_ref);
    replacer.register_rule("(A_offset)", a_offset);
    replacer.register_rule("(B_ptr)", b_ref);
    replacer.register_rule("(B_offset)", b_offset);
    replacer.register_rule("(C_ptr)", c_ref);
    replacer.register_rule("(C_offset)", c_offset);
    replacer.register_rule("(scale_a)", scale_a);
    replacer.register_rule("(scale_b)", scale_b);
    replacer.register_rule("(scale_a_byte_id)", scale_a_byte_id);
    replacer.register_rule("(scale_a_thread_id)", scale_a_thread_id);
    replacer.register_rule("(scale_b_byte_id)", scale_b_byte_id);
    replacer.register_rule("(scale_b_thread_id)", scale_b_thread_id);
    this->stream << replacer.rewrite(mma_call);
  } else if (op->op.same_as(tl::ptx_mma_sp())) {
    // arg 0: shape: mXnXkX
    // arg 1: A layout: row/col
    // arg 2: B layout: row/col
    // arg 3: A precision: fp16, fp32, ...
    // arg 4: B precision: fp16, fp32, ...
    // arg 5: C precision: fp16, fp32, ...
    // arg 6: A multiplicand pointer
    // arg 7: A multiplicand index
    // arg 8: B multiplicand pointer
    // arg 9: B multiplicand index
    // arg 10: C accumulator pointer
    // arg 11: C accumulator index
    // arg 12: metadata
    // arg 13: metadata index
    // arg 14: sparse_selector
    // arg 15: saturate
    ICHECK_EQ(args.size(), 16U);
    std::string shape = (args[0]).as_or_throw<StringImm>()->value;
    std::string A_layout = (args[1]).as_or_throw<StringImm>()->value;
    std::string B_layout = (args[2]).as_or_throw<StringImm>()->value;
    std::string A_dtype = (args[3]).as_or_throw<StringImm>()->value;
    std::string B_dtype = (args[4]).as_or_throw<StringImm>()->value;
    std::string C_dtype = (args[5]).as_or_throw<StringImm>()->value;
    std::string a_ref = this->PrintExpr(args[6]);
    std::string a_offset = this->PrintExpr(args[7]);
    std::string b_ref = this->PrintExpr(args[8]);
    std::string b_offset = this->PrintExpr(args[9]);
    std::string c_ref = this->PrintExpr(args[10]);
    std::string c_offset = this->PrintExpr(args[11]);
    std::string e_ref = this->PrintExpr(args[12]);
    std::string e_offset = this->PrintExpr(args[13]);
    int64_t sparse_selector = CodegenInt(args[14]);

    auto dtype_a_enum = tl::codegen::ptx::DTypeFromString(A_dtype);
    auto dtype_b_enum = tl::codegen::ptx::DTypeFromString(B_dtype);
    auto dtype_c_enum = tl::codegen::ptx::DTypeFromString(C_dtype);
    auto [m, n, k] = tl::codegen::ptx::ParseMMAShape(shape);

    need_mma_sp_instruction_h_ = true;
    this->PrintIndent();

    std::string mma_call =
        "tl::mma_sp_sync<(AType), (BType), (CType), (M), (N), (K), (TransA), "
        "(TransB), (SparseSel)>("
        "reinterpret_cast<(CRegType)*>((C_ptr) + (C_offset)), "
        "reinterpret_cast<const (ARegType)*>((A_ptr) + (A_offset)), "
        "reinterpret_cast<const (BRegType)*>((B_ptr) + (B_offset)), "
        "reinterpret_cast<const uint32_t*>((E_ptr) + (E_offset)));\n";
    tl::codegen::Replacer replacer;

    // TF32 workaround: float32 A/B in TF32 context maps to kTensorFloat32.
    std::string AType = tl::codegen::ptx::DTypeEnumToString(dtype_a_enum);
    if (AType == "tl::DataType::kFloat32") {
      AType = "tl::DataType::kTensorFloat32";
    }
    std::string BType = tl::codegen::ptx::DTypeEnumToString(dtype_b_enum);
    if (BType == "tl::DataType::kFloat32") {
      BType = "tl::DataType::kTensorFloat32";
    }
    std::string ARegType = tl::codegen::GetMMARegisterType(dtype_a_enum);
    if (ARegType == "float") {
      ARegType = "uint32_t";
    }
    std::string BRegType = tl::codegen::GetMMARegisterType(dtype_b_enum);
    if (BRegType == "float") {
      BRegType = "uint32_t";
    }

    replacer.register_rule("(AType)", AType);
    replacer.register_rule("(BType)", BType);
    replacer.register_rule("(CType)",
                           tl::codegen::ptx::DTypeEnumToString(dtype_c_enum));
    replacer.register_rule("(M)", std::to_string(m));
    replacer.register_rule("(N)", std::to_string(n));
    replacer.register_rule("(K)", std::to_string(k));
    replacer.register_rule("(TransA)", A_layout == "row" ? "false" : "true");
    replacer.register_rule("(TransB)", B_layout == "row" ? "false" : "true");
    replacer.register_rule("(SparseSel)", sparse_selector == 0
                                              ? "SM80::MMA::SparseSel::Zero"
                                              : "SM80::MMA::SparseSel::One");
    replacer.register_rule("(ARegType)", ARegType);
    replacer.register_rule("(BRegType)", BRegType);
    replacer.register_rule("(CRegType)",
                           tl::codegen::GetMMARegisterType(dtype_c_enum));
    replacer.register_rule("(A_ptr)", a_ref);
    replacer.register_rule("(A_offset)", a_offset);
    replacer.register_rule("(B_ptr)", b_ref);
    replacer.register_rule("(B_offset)", b_offset);
    replacer.register_rule("(C_ptr)", c_ref);
    replacer.register_rule("(C_offset)", c_offset);
    replacer.register_rule("(E_ptr)", e_ref);
    replacer.register_rule("(E_offset)", e_offset);
    this->stream << replacer.rewrite(mma_call);
  } else if (op->op.same_as(tl::ptx_wgmma_ss())) {
    // arg 0: dtype
    // arg 1: shape
    // arg 2: A_layout
    // arg 3: B_layout
    // arg 4: A_dtype
    // arg 5: B_dtype
    // arg 6: C_dtype
    // arg 7: multiplicand_a
    // arg 8: multiplicand_b
    // arg 9: accumulator
    // arg 10: saturate
    ICHECK_EQ(args.size(), 15U) << "ptx_wgmma_ss args is " << args;
    std::string shape = (args[0]).as_or_throw<StringImm>()->value;
    bool a_is_k_major = CodegenBool(args[1]);
    bool b_is_k_major = CodegenBool(args[2]);
    std::string A_dtype = (args[3]).as_or_throw<StringImm>()->value;
    std::string B_dtype = (args[4]).as_or_throw<StringImm>()->value;
    std::string C_dtype = (args[5]).as_or_throw<StringImm>()->value;
    std::string a_desc = this->PrintExpr(args[6]);
    std::string A_offset = this->PrintExpr(args[7]);
    std::string b_desc = this->PrintExpr(args[8]);
    std::string B_offset = this->PrintExpr(args[9]);
    std::string c_ref = this->PrintExpr(args[10]);
    std::string c_offset = this->PrintExpr(args[11]);
    std::string scale_out = this->PrintExpr(args[12]);
    bool scale_in_a = CodegenBool(args[13]);
    bool scale_in_b = CodegenBool(args[14]);

    const bool a_is_shared = true;
    this->PrintIndent();
    auto [m, n, k] = tl::codegen::ptx::ParseMMAShape(shape);
    need_wgmma_instruction_h_ = true;
    std::string wgmma_asm_code =
        "tl::wgmma_ss<(AType), (BType), (CType), (M), (N), (K), (tnspA), "
        "(tnspB), (scaleA), (scaleB)>(uint64_t((desc_a) + (A_offset)), "
        "uint64_t((desc_b) + (B_offset)), ((uint32_t*)((C))), (scale_out));\n";
    // replace patterns
    tl::codegen::Replacer replacer;

    std::string AType = tl::codegen::ptx::DTypeEnumToString(A_dtype);
    std::string BType = tl::codegen::ptx::DTypeEnumToString(B_dtype);

    replacer.register_rule("(AType)", AType);
    replacer.register_rule("(BType)", BType);
    replacer.register_rule("(CType)",
                           tl::codegen::ptx::DTypeEnumToString(C_dtype));
    replacer.register_rule("(M)", std::to_string(m));
    replacer.register_rule("(N)", std::to_string(n));
    replacer.register_rule("(K)", std::to_string(k));
    replacer.register_rule("(tnspA)", a_is_k_major ? "false" : "true");
    replacer.register_rule("(tnspB)", b_is_k_major ? "false" : "true");
    replacer.register_rule("(scaleA)", scale_in_a ? "1" : "-1");
    replacer.register_rule("(scaleB)", scale_in_b ? "1" : "-1");
    replacer.register_rule("(desc_a)", a_desc);
    replacer.register_rule("(A_offset)", A_offset);
    replacer.register_rule("(desc_b)", b_desc);
    replacer.register_rule("(B_offset)", B_offset);
    replacer.register_rule("(C)", c_ref + " + " + c_offset);
    replacer.register_rule("(scale_out)", scale_out);
    wgmma_asm_code = replacer.rewrite(wgmma_asm_code);
    this->stream << wgmma_asm_code;
  } else if (op->op.same_as(tl::ptx_wgmma_rs())) {
    // arg 0: shape
    // arg 1: B_layout
    // arg 2: A_dtype
    // arg 3: B_dtype
    // arg 4: C_dtype
    // arg 5: multiplicand_a
    // arg 6: multiplicand_a offset
    // arg 7: multiplicand_b descriptor
    // arg 8: multiplicand_b offset
    // arg 9: accumulator
    // arg 10: accumulator offset
    // arg 11: scale_out
    // arg 12: scale_in_a
    // arg 13: scale_in_b
    ICHECK_EQ(args.size(), 14U) << "ptx_wgmma_rs args is " << args;
    std::string shape = (args[0]).as_or_throw<StringImm>()->value;
    bool b_is_k_major = CodegenBool(args[1]);
    std::string A_dtype = (args[2]).as_or_throw<StringImm>()->value;
    std::string B_dtype = (args[3]).as_or_throw<StringImm>()->value;
    std::string C_dtype = (args[4]).as_or_throw<StringImm>()->value;
    std::string a_ref = this->PrintExpr(args[5]);
    std::string A_offset = this->PrintExpr(args[6]);
    std::string b_desc = this->PrintExpr(args[7]);
    std::string B_offset = this->PrintExpr(args[8]);
    std::string c_ref = this->PrintExpr(args[9]);
    std::string c_offset = this->PrintExpr(args[10]);
    std::string scale_out = this->PrintExpr(args[11]);
    bool scale_in_a = CodegenBool(args[12]);
    bool scale_in_b = CodegenBool(args[13]);

    auto dtype_a_enum = tl::codegen::ptx::DTypeFromString(A_dtype);
    auto dtype_b_enum = tl::codegen::ptx::DTypeFromString(B_dtype);
    auto dtype_c_enum = tl::codegen::ptx::DTypeFromString(C_dtype);
    auto [m, n, k] = tl::codegen::ptx::ParseMMAShape(shape);

    need_wgmma_instruction_h_ = true;
    this->PrintIndent();
    std::string wgmma_call =
        "tl::wgmma_rs<(AType), (BType), (CType), (M), (N), (K), (tnspA), "
        "(tnspB), (scaleA), (scaleB)>(reinterpret_cast<const "
        "uint32_t*>((A_ptr) + (A_offset)), "
        "uint64_t((desc_b) + (B_offset)), "
        "reinterpret_cast<uint32_t*>((C_ptr) + (C_offset)), "
        "(scale_out));\n";

    tl::codegen::Replacer replacer;
    std::string AType = tl::codegen::ptx::DTypeEnumToString(A_dtype);
    std::string BType = tl::codegen::ptx::DTypeEnumToString(B_dtype);

    replacer.register_rule("(AType)", AType);
    replacer.register_rule("(BType)", BType);
    replacer.register_rule("(CType)",
                           tl::codegen::ptx::DTypeEnumToString(dtype_c_enum));
    replacer.register_rule("(M)", std::to_string(m));
    replacer.register_rule("(N)", std::to_string(n));
    replacer.register_rule("(K)", std::to_string(k));
    replacer.register_rule("(tnspA)", "false");
    replacer.register_rule("(tnspB)", b_is_k_major ? "false" : "true");
    replacer.register_rule("(scaleA)", scale_in_a ? "1" : "-1");
    replacer.register_rule("(scaleB)", scale_in_b ? "1" : "-1");
    replacer.register_rule("(A_ptr)", a_ref);
    replacer.register_rule("(A_offset)", A_offset);
    replacer.register_rule("(desc_b)", b_desc);
    replacer.register_rule("(B_offset)", B_offset);
    replacer.register_rule("(C_ptr)", c_ref);
    replacer.register_rule("(C_offset)", c_offset);
    replacer.register_rule("(scale_out)", scale_out);
    wgmma_call = replacer.rewrite(wgmma_call);
    this->stream << wgmma_call;
  } else if (op->op.same_as(tl::ptx_wgmma_sp_ss())) {
    ICHECK_EQ(args.size(), 18U) << "ptx_wgmma_sp_ss args is " << args;
    std::string wgmma_prefix = (args[0]).as_or_throw<StringImm>()->value;
    bool a_is_k_major = CodegenBool(args[1]);
    bool b_is_k_major = CodegenBool(args[2]);
    std::string A_dtype = (args[3]).as_or_throw<StringImm>()->value;
    std::string B_dtype = (args[4]).as_or_throw<StringImm>()->value;
    std::string C_dtype = (args[5]).as_or_throw<StringImm>()->value;
    std::string a_desc = this->PrintExpr(args[6]);
    std::string A_offset = this->PrintExpr(args[7]);
    std::string e_data = this->PrintExpr(args[8]);
    std::string E_offset = this->PrintExpr(args[9]);
    std::string sparse_selector = this->PrintExpr(args[10]);
    std::string b_desc = this->PrintExpr(args[11]);
    std::string B_offset = this->PrintExpr(args[12]);
    std::string c_data = this->PrintExpr(args[13]);
    std::string C_offset = this->PrintExpr(args[14]);
    std::string scale_out = this->PrintExpr(args[15]);
    bool scale_in_a = CodegenBool(args[16]);
    bool scale_in_b = CodegenBool(args[17]);

    this->PrintIndent();
    auto [m, n, k] = tl::codegen::ptx::ParseMMAShape(wgmma_prefix);
    need_wgmma_sp_instruction_h_ = true;
    int sparse_sel_val_ss = CodegenInt(args[10]);
    std::string spsel_ss = (sparse_sel_val_ss == 0)
                               ? "cute::SM90::GMMA::SparseSel::Zero"
                               : "cute::SM90::GMMA::SparseSel::One";
    std::string wgmma_sp_asm_code =
        "tl::wgmma_sp_ss<(AType), (BType), (CType), (M), (N), (K), (tnspA), "
        "(tnspB), (scaleA), (scaleB), (spsel)>(uint64_t((desc_a) + "
        "(A_offset)), "
        "uint64_t((desc_b) + (B_offset)), "
        "reinterpret_cast<uint32_t*>((C_data)) "
        "+ (C_offset), (scale_out), *reinterpret_cast<uint32_t*>((e_data) + "
        "(E_offset)));\n";

    tl::codegen::Replacer replacer;
    std::string AType = tl::codegen::ptx::DTypeEnumToString(A_dtype);
    if (AType == "tl::DataType::kFloat32") {
      AType = "tl::DataType::kTensorFloat32";
    }
    std::string BType = tl::codegen::ptx::DTypeEnumToString(B_dtype);
    if (BType == "tl::DataType::kFloat32") {
      BType = "tl::DataType::kTensorFloat32";
    }

    replacer.register_rule("(AType)", AType);
    replacer.register_rule("(BType)", BType);
    replacer.register_rule("(CType)",
                           tl::codegen::ptx::DTypeEnumToString(C_dtype));
    replacer.register_rule("(M)", std::to_string(m));
    replacer.register_rule("(N)", std::to_string(n));
    replacer.register_rule("(K)", std::to_string(k));
    replacer.register_rule("(tnspA)", a_is_k_major ? "false" : "true");
    replacer.register_rule("(tnspB)", b_is_k_major ? "false" : "true");
    replacer.register_rule("(scaleA)", scale_in_a ? "1" : "-1");
    replacer.register_rule("(scaleB)", scale_in_b ? "1" : "-1");
    replacer.register_rule("(spsel)", spsel_ss);
    replacer.register_rule("(desc_a)", a_desc);
    replacer.register_rule("(A_offset)", A_offset);
    replacer.register_rule("(e_data)", e_data);
    replacer.register_rule("(E_offset)", E_offset);
    replacer.register_rule("(desc_b)", b_desc);
    replacer.register_rule("(B_offset)", B_offset);
    replacer.register_rule("(C_data)", c_data);
    replacer.register_rule("(C_offset)", C_offset);
    replacer.register_rule("(scale_out)", scale_out);
    wgmma_sp_asm_code = replacer.rewrite(wgmma_sp_asm_code);
    this->stream << wgmma_sp_asm_code;
  } else if (op->op.same_as(tl::ptx_wgmma_sp_rs())) {
    ICHECK_EQ(args.size(), 17U) << "ptx_wgmma_sp_rs args is " << args;
    std::string wgmma_prefix = (args[0]).as_or_throw<StringImm>()->value;
    bool b_is_k_major = CodegenBool(args[1]);
    std::string A_dtype = (args[2]).as_or_throw<StringImm>()->value;
    std::string B_dtype = (args[3]).as_or_throw<StringImm>()->value;
    std::string C_dtype = (args[4]).as_or_throw<StringImm>()->value;
    std::string a_ref = this->PrintExpr(args[5]);
    std::string A_offset = this->PrintExpr(args[6]);
    std::string e_data = this->PrintExpr(args[7]);
    std::string E_offset = this->PrintExpr(args[8]);
    std::string sparse_selector = this->PrintExpr(args[9]);
    std::string b_desc = this->PrintExpr(args[10]);
    std::string B_offset = this->PrintExpr(args[11]);
    std::string c_data = this->PrintExpr(args[12]);
    std::string C_offset = this->PrintExpr(args[13]);
    std::string scale_out = this->PrintExpr(args[14]);
    bool scale_in_a = CodegenBool(args[15]);
    bool scale_in_b = CodegenBool(args[16]);

    auto [m, n, k] = tl::codegen::ptx::ParseMMAShape(wgmma_prefix);
    need_wgmma_sp_instruction_h_ = true;
    this->PrintIndent();
    int sparse_sel_val_rs = CodegenInt(args[9]);
    std::string spsel_rs = (sparse_sel_val_rs == 0)
                               ? "cute::SM90::GMMA::SparseSel::Zero"
                               : "cute::SM90::GMMA::SparseSel::One";
    std::string wgmma_sp_rs_asm_code =
        "tl::wgmma_sp_rs<(AType), (BType), (CType), (M), (N), (K), false, "
        "(tnspB), (scaleA), (scaleB), (spsel)>(reinterpret_cast<const "
        "uint32_t*>((A_ptr) + "
        "(A_offset)), uint64_t((desc_b) + (B_offset)), "
        "reinterpret_cast<uint32_t*>((C_data)) + (C_offset), (scale_out), "
        "*reinterpret_cast<uint32_t*>((e_data) + (E_offset)));\n";

    tl::codegen::Replacer replacer;
    std::string AType = tl::codegen::ptx::DTypeEnumToString(A_dtype);
    if (AType == "tl::DataType::kFloat32") {
      AType = "tl::DataType::kTensorFloat32";
    }
    std::string BType = tl::codegen::ptx::DTypeEnumToString(B_dtype);
    if (BType == "tl::DataType::kFloat32") {
      BType = "tl::DataType::kTensorFloat32";
    }

    replacer.register_rule("(AType)", AType);
    replacer.register_rule("(BType)", BType);
    replacer.register_rule("(CType)",
                           tl::codegen::ptx::DTypeEnumToString(C_dtype));
    replacer.register_rule("(M)", std::to_string(m));
    replacer.register_rule("(N)", std::to_string(n));
    replacer.register_rule("(K)", std::to_string(k));
    replacer.register_rule("(tnspB)", b_is_k_major ? "false" : "true");
    replacer.register_rule("(scaleA)", scale_in_a ? "1" : "-1");
    replacer.register_rule("(scaleB)", scale_in_b ? "1" : "-1");
    replacer.register_rule("(spsel)", spsel_rs);
    replacer.register_rule("(A_ptr)", a_ref);
    replacer.register_rule("(A_offset)", A_offset);
    replacer.register_rule("(e_data)", e_data);
    replacer.register_rule("(E_offset)", E_offset);
    replacer.register_rule("(desc_b)", b_desc);
    replacer.register_rule("(B_offset)", B_offset);
    replacer.register_rule("(C_data)", c_data);
    replacer.register_rule("(C_offset)", C_offset);
    replacer.register_rule("(scale_out)", scale_out);
    wgmma_sp_rs_asm_code = replacer.rewrite(wgmma_sp_rs_asm_code);
    this->stream << wgmma_sp_rs_asm_code;
  } else if (op->op.same_as(tl::ptx_tcgen05_mma_ss())) {
    ICHECK_EQ(args.size(), 15U) << "ptx_tcgen05_mma_ss args is " << args;
    std::string kind_dtype = (args[0]).as_or_throw<StringImm>()->value;
    std::string a_desc = this->PrintExpr(args[1]);
    std::string A_offset = this->PrintExpr(args[2]);
    std::string b_desc = this->PrintExpr(args[3]);
    std::string B_offset = this->PrintExpr(args[4]);
    std::string c_ref = this->PrintExpr(args[5]);
    std::string c_offset = this->PrintExpr(args[6]);
    PrimExpr desc_expr = args[7].as_or_throw<PrimExpr>();
    std::string scale_out = this->PrintExpr(args[8]);
    std::string mask0 = this->PrintExpr(args[9]);
    std::string mask1 = this->PrintExpr(args[10]);
    std::string mask2 = this->PrintExpr(args[11]);
    std::string mask3 = this->PrintExpr(args[12]);
    bool enable_ws = CodegenBool(args[13]);
    bool enable_2cta = CodegenBool(args[14]);

    std::string use_2cta_suffix;
    if (enable_ws) {
      ICHECK(!enable_2cta)
          << "enable_ws and enable_2cta cannot be true at the same time";
    } else {
      use_2cta_suffix = std::string(", ") + (enable_2cta ? "true" : "false");
    }
    auto dtype_enum = tl::codegen::ptx::DTypeFromString(kind_dtype);
    std::string ab_type_str = tl::codegen::ptx::DTypeEnumToString(dtype_enum);

    need_tcgen05mma_instruction_h_ = true;
    this->PrintIndent();
    std::string tcgen05_call =
        "tl::(tcgen05_name)<(ABType)(USE_2CTA_SUFFIX)>(uint64_t((desc_a) + "
        "(A_offset)), "
        "uint64_t((desc_b) + (B_offset)), (*reinterpret_cast<uint32_t*>((C))) "
        "+ (C_offset), "
        "(scale_out), static_cast<uint32_t>((desc_val)), (mask0), (mask1), "
        "(mask2), (mask3));\n";
    tl::codegen::Replacer replacer;
    replacer.register_rule("(ABType)", ab_type_str);
    replacer.register_rule("(USE_2CTA_SUFFIX)", use_2cta_suffix);
    replacer.register_rule("(desc_a)", a_desc);
    replacer.register_rule("(A_offset)", A_offset);
    replacer.register_rule("(desc_b)", b_desc);
    replacer.register_rule("(B_offset)", B_offset);
    replacer.register_rule("(C)", c_ref);
    replacer.register_rule("(C_offset)", c_offset);
    replacer.register_rule("(tcgen05_name)",
                           enable_ws ? "tcgen05mma_ws_ss" : "tcgen05mma_ss");
    replacer.register_rule("(scale_out)", scale_out);
    replacer.register_rule("(desc_val)", this->PrintExpr(desc_expr));
    replacer.register_rule("(mask0)", mask0);
    replacer.register_rule("(mask1)", mask1);
    replacer.register_rule("(mask2)", mask2);
    replacer.register_rule("(mask3)", mask3);
    tcgen05_call = replacer.rewrite(tcgen05_call);
    this->stream << tcgen05_call;
  } else if (op->op.same_as(tl::ptx_tcgen05_mma_ts())) {
    // TS: A from TMEM, B from SMEM (desc)
    ICHECK_EQ(args.size(), 14U) << "ptx_tcgen05_mma_ts args is " << args;
    std::string kind_dtype = (args[0]).as_or_throw<StringImm>()->value;
    std::string a_ref = this->PrintExpr(args[1]);
    std::string A_offset = this->PrintExpr(args[2]);
    std::string b_desc = this->PrintExpr(args[3]);
    std::string B_offset = this->PrintExpr(args[4]);
    std::string c_ref = this->PrintExpr(args[5]);
    std::string c_offset = this->PrintExpr(args[6]);
    PrimExpr desc_expr = args[7].as_or_throw<PrimExpr>();
    std::string scale_out = this->PrintExpr(args[8]);
    std::string mask0 = this->PrintExpr(args[9]);
    std::string mask1 = this->PrintExpr(args[10]);
    std::string mask2 = this->PrintExpr(args[11]);
    std::string mask3 = this->PrintExpr(args[12]);
    bool enable_2cta = CodegenBool(args[13]);

    auto dtype_enum = tl::codegen::ptx::DTypeFromString(kind_dtype);
    std::string use_2cta_suffix =
        std::string(", ") + (enable_2cta ? "true" : "false");

    need_tcgen05mma_instruction_h_ = true;
    this->PrintIndent();
    std::string tcgen05_call =
        "tl::tcgen05mma_ts<(ABType)(USE_2CTA_SUFFIX)>( "
        "(*reinterpret_cast<uint32_t*>((A))) + "
        "(A_offset), "
        "uint64_t((desc_b) + (B_offset)), (*reinterpret_cast<uint32_t*>((C))) "
        "+ (C_offset), "
        "(scale_out), static_cast<uint32_t>((desc_val)), (mask0), (mask1), "
        "(mask2), (mask3));\n";
    tl::codegen::Replacer replacer;
    replacer.register_rule("(ABType)",
                           tl::codegen::ptx::DTypeEnumToString(dtype_enum));
    replacer.register_rule("(USE_2CTA_SUFFIX)", use_2cta_suffix);
    replacer.register_rule("(A)", a_ref);
    replacer.register_rule("(A_offset)", A_offset);
    replacer.register_rule("(desc_b)", b_desc);
    replacer.register_rule("(B_offset)", B_offset);
    replacer.register_rule("(C)", c_ref);
    replacer.register_rule("(C_offset)", c_offset);
    replacer.register_rule("(scale_out)", scale_out);
    replacer.register_rule("(desc_val)", this->PrintExpr(desc_expr));
    replacer.register_rule("(mask0)", mask0);
    replacer.register_rule("(mask1)", mask1);
    replacer.register_rule("(mask2)", mask2);
    replacer.register_rule("(mask3)", mask3);
    tcgen05_call = replacer.rewrite(tcgen05_call);
    this->stream << tcgen05_call;
  } else if (op->op.same_as(tl::ptx_tcgen05_mma_blockscaled_ss())) {
    ICHECK_EQ(args.size(), 16U)
        << "ptx_tcgen05_mma_blockscaled_ss expects 16 arguments";
    std::string kind_dtype = (args[0]).as_or_throw<StringImm>()->value;
    std::string a_desc = this->PrintExpr(args[1]);
    std::string A_offset = this->PrintExpr(args[2]);
    std::string b_desc = this->PrintExpr(args[3]);
    std::string B_offset = this->PrintExpr(args[4]);
    std::string c_ref = this->PrintExpr(args[5]);
    std::string c_offset = this->PrintExpr(args[6]);
    PrimExpr desc_expr = args[7].as_or_throw<PrimExpr>();
    std::string scale_out = this->PrintExpr(args[8]);
    std::string sfa_ref = this->PrintExpr(args[9]);
    std::string sfa_offset = this->PrintExpr(args[10]);
    std::string sfb_ref = this->PrintExpr(args[11]);
    std::string sfb_offset = this->PrintExpr(args[12]);
    // args[13], [14] reserved for future mask/flags
    bool enable_2cta = CodegenBool(args[15]);

    auto dtype_enum = tl::codegen::ptx::DTypeFromString(kind_dtype);

    need_tcgen05mma_instruction_h_ = true;
    this->PrintIndent();
    std::string tcgen05_call =
        "tl::(tcgen05_name)<(ABType), (USE_2CTA)>(uint64_t((desc_a) + "
        "(A_offset)), "
        "uint64_t((desc_b) + (B_offset)), (*reinterpret_cast<uint32_t*>((C))) "
        "+ (C_offset), "
        "(scale_out), static_cast<uint32_t>((desc_val)), "
        "(*reinterpret_cast<uint32_t*>((SFA))) + (SFA_offset), "
        "(*reinterpret_cast<uint32_t*>((SFB))) + (SFB_offset));\n";
    tl::codegen::Replacer replacer;
    replacer.register_rule("(ABType)",
                           tl::codegen::ptx::DTypeEnumToString(dtype_enum));
    replacer.register_rule("(USE_2CTA)", enable_2cta ? "true" : "false");
    replacer.register_rule("(desc_a)", a_desc);
    replacer.register_rule("(A_offset)", A_offset);
    replacer.register_rule("(desc_b)", b_desc);
    replacer.register_rule("(B_offset)", B_offset);
    replacer.register_rule("(C)", c_ref);
    replacer.register_rule("(C_offset)", c_offset);
    replacer.register_rule("(tcgen05_name)", "tcgen05mma_blockscaled_ss");
    replacer.register_rule("(scale_out)", scale_out);
    replacer.register_rule("(desc_val)", this->PrintExpr(desc_expr));
    replacer.register_rule("(SFA)", sfa_ref);
    replacer.register_rule("(SFA_offset)", sfa_offset);
    replacer.register_rule("(SFB)", sfb_ref);
    replacer.register_rule("(SFB_offset)", sfb_offset);
    tcgen05_call = replacer.rewrite(tcgen05_call);
    this->stream << tcgen05_call;
  } else if (op->op.same_as(tl::ptx_tcgen05_cp_warpx4())) {
    ICHECK_EQ(args.size(), 3U) << "ptx_tcgen05_cp_warpx4 expects 3 arguments";
    need_tcgen05_common_h_ = true;
    // arg[0] = smem pointer, arg[1] = tmem data pointer, arg[2] = tmem column
    // offset
    std::string smem_ptr = this->PrintExpr(args[0]);
    std::string tmem_ptr = this->PrintExpr(args[1]);
    std::string tmem_col_offset = this->PrintExpr(args[2]);
    bool use_2cta =
        attrs && CodegenBool(attrs->dict.Get("use_2cta").value_or(false));
    this->PrintIndent();
    this->stream << "tl::tcgen05_cp<" << (use_2cta ? "true" : "false") << ">("
                 << "tl::make_sf_smem_desc(reinterpret_cast<void*>(" << smem_ptr
                 << ")), " << "(*reinterpret_cast<uint32_t*>(" << tmem_ptr
                 << ")) + " << tmem_col_offset << ");\n";
  } else if (op->op.same_as(tl::ptx_tcgen05_sf_warp_transpose())) {
    ICHECK_EQ(args.size(), 1U)
        << "ptx_tcgen05_sf_warp_transpose expects 1 argument";
    need_tcgen05_common_h_ = true;
    std::string smem_ptr = this->PrintExpr(args[0]);
    this->PrintIndent();
    this->stream << "tl::tcgen05_sf_warp_transpose(reinterpret_cast<uint32_t*>("
                 << smem_ptr << "));\n";
  } else if (op->op.same_as(tl::tcgen05_ld())) {
    ICHECK(args.size() == 6U || args.size() == 7U)
        << "tcgen05_ld expects 6 or 7 arguments";
    need_copy_sm100_h_ = true;
    int inst_bits = CodegenInt(args[0]);
    int chunks = CodegenInt(args[1]);
    bool pack16 = CodegenBool(args[2]);
    std::string tmem_start_col = this->PrintExpr(args[3]);
    std::string col_offset = this->PrintExpr(args[4]);
    // The datapath count is optional: a hand-written call from before it
    // existed means the sub-partition-filling (32dp) wrappers.
    bool has_datapaths = args.size() == 7U;
    int datapaths = has_datapaths ? CodegenInt(args[5]) : 32;
    std::string dst_ptr = this->PrintExpr(args[has_datapaths ? 6 : 5]);
    this->PrintIndent();
    this->stream << "tl::tcgen05_ld_" << datapaths << "dp" << inst_bits
                 << "bNx<" << chunks << ", " << (pack16 ? "true" : "false")
                 << ">(" << tmem_start_col << ", " << col_offset << ", "
                 << dst_ptr << ");\n";
  } else if (op->op.same_as(tl::tcgen05_st())) {
    ICHECK(args.size() == 6U || args.size() == 7U)
        << "tcgen05_st expects 6 or 7 arguments";
    need_copy_sm100_h_ = true;
    int inst_bits = CodegenInt(args[0]);
    int chunks = CodegenInt(args[1]);
    bool unpack16 = CodegenBool(args[2]);
    std::string tmem_start_col = this->PrintExpr(args[3]);
    std::string col_offset = this->PrintExpr(args[4]);
    // The datapath count is optional: a hand-written call from before it
    // existed means the sub-partition-filling (32dp) wrappers.
    bool has_datapaths = args.size() == 7U;
    int datapaths = has_datapaths ? CodegenInt(args[5]) : 32;
    std::string src_ptr = this->PrintExpr(args[has_datapaths ? 6 : 5]);
    this->PrintIndent();
    this->stream << "tl::tcgen05_st_" << datapaths << "dp" << inst_bits
                 << "bNx<" << chunks << ", " << (unpack16 ? "true" : "false")
                 << ">(" << tmem_start_col << ", " << col_offset << ", "
                 << src_ptr << ");\n";
  } else if (op->op.same_as(tl::tcgen05_mma_arrive())) {
    ICHECK_EQ(args.size(), 1U) << "tcgen05_mma_arrive expects 1 argument";
    need_tcgen05_common_h_ = true;
    std::ostringstream ss;
    ss << "tl::tcgen05_mma_arrive";
    if (attrs && CodegenBool(attrs->dict.Get("use_2cta").value_or(false))) {
      ss << "<true>";
    }
    print_extern_call_stmt(ss.str());
  } else if (op->op.same_as(tl::tcgen05_before_thread_sync())) {
    ICHECK_EQ(args.size(), 0U)
        << "tcgen05_before_thread_sync expects no arguments";
    need_tcgen05_common_h_ = true;
    print_extern_call_stmt("tl::tcgen05_before_thread_sync");
  } else if (op->op.same_as(tl::tcgen05_after_thread_sync())) {
    ICHECK_EQ(args.size(), 0U)
        << "tcgen05_after_thread_sync expects no arguments";
    need_tcgen05_common_h_ = true;
    print_extern_call_stmt("tl::tcgen05_after_thread_sync");
  } else if (op->op.same_as(ptx_legacy_ldmatrix_op)) {
    // arg 0: whether the matrix is loaded in column major format or not.
    // arg 1: number of matrices to load.
    // arg 2: The data type in the matrix, .b16 is the only accepted data type.
    // arg 3: pointer to local buffer.
    // arg 4: The offset of the element to store in the local buffer.
    // arg 5: pointer to the shared memory buffer to load.
    // arg 6: The offset of the start element of the row to load in shared
    // memory.
    ICHECK_EQ(args.size(), 7U);
    bool trans = CodegenBool(args[0]);
    int num = CodegenInt(args[1]);
    std::string type = (args[2]).as_or_throw<StringImm>()->value;
    std::string local_ptr = this->PrintExpr(args[3]);
    const PrimType dtype = op->ty.as_or_throw<PrimType>();
    bool is_packed_int4 = dtype.bits() == 4 && (dtype.MatchesCode(kDLInt) ||
                                                dtype.MatchesCode(kDLUInt));
    PrimExpr local_elem_offset_expr = args[4].as_or_throw<PrimExpr>();
    if (is_packed_int4) {
      local_elem_offset_expr =
          sym::Analyzer()->Simplify(truncdiv(local_elem_offset_expr, 2));
    }
    std::string local_elem_offset = this->PrintExpr(local_elem_offset_expr);
    std::string smem_ptr = this->PrintExpr(args[5]);

    if (trans && dtype.bits() == 8) {
      // Since ldmatrix assumes that a matrix element is 16 bit, it cannot
      // properly transpose an int8 matrix.
      std::string smem_stride = this->PrintExpr(args[6]);
      ICHECK(num == 4);
      os << "for (int i = 0; i < 16; ++i) {\n";
      os << local_ptr << "[" + local_elem_offset + " + i] = " << smem_ptr
         << "[(i % 8) / 4 * " + smem_stride +
                " * 16 + (threadIdx.x % 4) * 4 * " + smem_stride +
                "+ (i % 4) * " + smem_stride +
                " + threadIdx.x / 4 + (i / 8) * 8];\n";
      os << "}\n";
    } else {
      PrimExpr smem_elem_offset_expr = args[6].as_or_throw<PrimExpr>();
      if (is_packed_int4) {
        smem_elem_offset_expr =
            sym::Analyzer()->Simplify(truncdiv(smem_elem_offset_expr, 2));
      }
      std::string smem_elem_offset = this->PrintExpr(smem_elem_offset_expr);
      std::string func_name = "tl::ptx_ldmatrix_x" + std::to_string(num);
      if (trans == 1)
        func_name += "_trans";
      this->PrintIndent();
      this->stream << func_name << "(" << smem_ptr << " + " << smem_elem_offset
                   << ", " << local_ptr << " + " << local_elem_offset << ");\n";
    }
  } else if (op->op.same_as(cuda_mma_store_op)) {
    int m = CodegenInt(args[0]);
    int n = CodegenInt(args[1]);
    std::string dst = this->PrintExpr(args[2]);
    std::string src = this->PrintExpr(args[3]);
    std::string src_offset = this->PrintExpr(args[4]);
    PrimExpr stride = args[5].as_or_throw<PrimExpr>();

    ICHECK(m == 16 && n == 16)
        << "Only m == 16 && n == 16 case supported for now";

    // Each thread in a warp holds a certain number of elements of an MMA
    // output. For example, if we compute a 16x16 tile using MMA, each thread
    // holds 8 elements in its registers. So conceptually, a warp memory is
    // organized as a 32x8 block. A map from a 16x16 tile to a 32x8 block of
    // memory is specified by the index map below.

    // To store the 32x8 output back to a 16x16 tile in shared or global memory,
    // we invert this map to determine the output location for each 8 element.

    const auto index_map_func = ffi::Function::GetGlobal(
        "tir.index_map.shared_16x16_to_mma_32x8_layout");

    PrimVar i("mma_row", PrimType::Int(32)), j("mma_column", PrimType::Int(32));
    Array<PrimExpr> mapped;
    if (index_map_func) {
      mapped = (*index_map_func)(i, j).cast<Array<PrimExpr>>();
    } else {
      mapped = {4 * floormod(i, 8) + floordiv(floormod(j, 8), 2),
                4 * floordiv(j, 8) + floordiv(i, 8) * 2 + floormod(j, 2)};
    }
    tl::Layout layout(
        Array<tl::LayoutIterVar>{tl::LayoutIterVar(Range(0, m), i),
                                 tl::LayoutIterVar(Range(0, n), j)},
        mapped);
    auto inverse = layout->Inverse();
    auto indices = inverse->GetForwardIndex();
    auto lower_div_mod =
        [](const PrimExpr &expr) -> ffi::UnchangedOr<ffi::Any> {
      if (const auto *div = expr.as<prim::FloorDivObj>())
        return ffi::Any(prim::Div(div->a, div->b));
      if (const auto *mod = expr.as<prim::FloorModObj>())
        return ffi::Any(prim::Mod(mod->a, mod->b));
      return ffi::Unchanged();
    };
    auto dst_ind = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
                       indices[0] * stride + indices[1], lower_div_mod)
                       .as_or_throw<PrimExpr>();
    auto coordinates = inverse->GetForwardVars();
    var_idmap_[coordinates[0].as_or_throw<PrimVar>().get()] = "threadIdx.x";
    var_idmap_[coordinates[1].as_or_throw<PrimVar>().get()] = "local_id";
    if (op->ty.as_or_throw<PrimType>().bits() == 16) {
      os << "for (int local_id = 0; local_id < 8; local_id+=2) {\n";
      os << "*((uint *)&" << dst << "[" + this->PrintExpr(dst_ind) + "])"
         << " = " << "*((uint *)&" << src << "[" << src_offset
         << " + local_id]);\n";
      os << "}\n";
    } else {
      os << "for (int local_id = 0; local_id < 8; ++local_id) {\n";
      os << dst << "[" + this->PrintExpr(dst_ind) + "]" << " = " << src << "["
         << src_offset << " + local_id];\n";
      os << "}\n";
    }

  } else if (op->op.same_as(cuda_mma_fill_op)) {
    std::string num_elem = this->PrintExpr(args[0]);
    std::string dst = this->PrintExpr(args[1]);
    std::string dst_offset = this->PrintExpr(args[2]);

    os << "for (int i = 0; i < " << num_elem << "; ++i) {\n";
    os << dst << "[" << dst_offset << " + i] = 0.0;";
    os << "}\n";
  } else if (op->op.same_as(tl::ptx_cp_async_bulk())) {
    need_cast_smem_ptr_to_int_ = true;
    std::string dst = this->PrintExpr(args[0]);
    std::string dst_offset = this->PrintExpr(args[1]);
    std::string src = this->PrintExpr(args[2]);
    std::string src_offset = this->PrintExpr(args[3]);
    std::string size = this->PrintExpr(args[4]);
    TVM_FFI_CHECK(args[5].as<IntImmObj>(), TypeError)
        << "tl.ptx_cp_async_bulk retains the legacy constant barrier-ID ABI";
    std::string barrier = BarrierReference(args[5]);
    this->stream << PrintCpAsyncBulkAsm(dst, dst_offset, src, src_offset, size,
                                        barrier);
  } else if (op->op.same_as(tl::ptx_st_bulk_shared())) {
    need_cast_smem_ptr_to_int_ = true;
    std::string smem = this->PrintExpr(args[0]);
    int64_t bytes = CodegenInt(args[1]);
    int64_t init_val = CodegenInt(args[2]);
    this->stream << "tl::st_bulk_shared<" << bytes << ", " << init_val
                 << ">((void*)(" << smem << "));\n";
  } else if (op->op.same_as(tl::ptx_wait_barrier())) {
    need_cast_smem_ptr_to_int_ = true;
    std::string barrier = BarrierReference(args[0]);
    this->stream << PrintWaitBarrierAsm(barrier);
  } else if (op->op.same_as(tl::ptx_ldg32())) {
    /*
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        // " @p ld.global.nc.f32 %0, [%1];}\n"t
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
        : "=f"(reg)
        : "l"(addr), "r"((int)guard)
    );
    */

    // get local
    std::string reg = this->PrintExpr(args[0]);
    // get guard
    std::string guard = this->PrintExpr(args[1]);
    const TensorLoadObj *addr_buffer = args[2].as<TensorLoadObj>();
    std::string global_addr = this->PrintExpr(addr_buffer->indices[0]);
    std::string global_buffer = this->PrintExpr(addr_buffer->source);
    std::string local_addr = this->PrintExpr(args[3]);
    this->stream << "asm volatile (\n";
    this->stream << "\"{.reg .pred p;\\n\"\n";
    this->stream << "\" setp.ne.b32 p, %2, 0;\\n\"\n";
    this->stream << "\" @!p mov.b32 %0, 0;\\n\"\n";
    this->stream << "\" @p ld.global.nc.f32 %0, [%1];}\\n\"\n";
    // stream << "\" @p ld.global.nc.L2::128B.f32 %0, [%1];}\\n\"\n" ;
    stream << ": \"=f\"(" << reg << "[" << local_addr << "]" << ")\n";
    stream << ": \"l\"((void*)(" << global_buffer << "+" << global_addr
           << ")), \"r\"((int)" << guard << ")\n";
    stream << ");\n";
  } else if (op->op.same_as(tl::__ldg())) {
    // Explicit read-only cached load. Preferred form: __ldg(BufferLoad(...)).
    // Fallback form: __ldg(buffer, index)
    const TensorLoadObj *bl = nullptr;
    if (!args.empty()) {
      bl = args[0].as<TensorLoadObj>();
    }
    if (bl == nullptr) {
      TVM_FFI_THROW(InternalError)
          << "T.__ldg expects a BufferLoad as the first argument.";
    }
    const VarObj *buffer = bl->source.as_or_throw<TensorVar>().get();
    ICHECK_EQ(bl->indices.size(), 1)
        << "T.__ldg currently supports flattened 1D buffer accesses.";
    PrimExpr base = bl->indices[0];
    // Emit __ldg(&buffer_ref)
    auto buffer_ref =
        this->GetBufferRef(op->ty.as_or_throw<PrimType>(), buffer, base);
    os << "__ldg(&(" << buffer_ref << "))";
  } else if (op->op.same_as(tl::__ffs())) {
    ICHECK_EQ(args.size(), 1U) << "T.__ffs expects one argument.";
    PrimType arg_dtype = args[0]->ty.as_or_throw<PrimType>();
    ICHECK(arg_dtype.MatchesCode(kDLInt) || arg_dtype.MatchesCode(kDLUInt))
        << "T.__ffs expects an integer argument, but got " << arg_dtype;
    ICHECK(arg_dtype.bits() == 32 || arg_dtype.bits() == 64)
        << "T.__ffs expects a 32-bit or 64-bit integer argument, but got "
        << arg_dtype;
    os << (arg_dtype.bits() == 64 ? "__ffsll(" : "__ffs(") << PrintExpr(args[0])
       << ")";
  } else if (op->op.same_as(tl::__fns())) {
    ICHECK_EQ(args.size(), 3U)
        << "T.__fns expects three arguments: mask, base, offset.";
    PrimType mask_dtype = args[0]->ty.as_or_throw<PrimType>();
    ICHECK(mask_dtype.MatchesCode(kDLInt) || mask_dtype.MatchesCode(kDLUInt))
        << "T.__fns expects an integer mask argument, but got " << mask_dtype;
    ICHECK(mask_dtype.bits() == 32)
        << "T.__fns expects a 32-bit integer mask argument, but got "
        << mask_dtype;
    os << "__fns(" << PrintExpr(args[0]) << ", " << PrintExpr(args[1]) << ", "
       << PrintExpr(args[2]) << ")";
  } else if (op->op.same_as(tl::ldg32())) {
    need_copy_h_ = true;
    // Explicit 32-bit global memory load: load_global_32(ptr) or
    // load_global_32_conditional(ptr, pred)
    ICHECK(!args.empty()) << "T.ldg32 expects a pointer argument.";
    if (args.size() > 1) {
      os << "tl::load_global_32_conditional(";
      this->PrintExpr(args[0], os);
      os << ", ";
      this->PrintExpr(args[1], os);
    } else {
      os << "tl::load_global_32(";
      this->PrintExpr(args[0], os);
    }
    os << ")";
  } else if (op->op.same_as(tl::ldg64())) {
    need_copy_h_ = true;
    // Explicit 64-bit global memory load: load_global_64(ptr) or
    // load_global_64_conditional(ptr, pred)
    ICHECK(!args.empty()) << "T.ldg64 expects a pointer argument.";
    if (args.size() > 1) {
      os << "tl::load_global_64_conditional(";
      this->PrintExpr(args[0], os);
      os << ", ";
      this->PrintExpr(args[1], os);
    } else {
      os << "tl::load_global_64(";
      this->PrintExpr(args[0], os);
    }
    os << ")";
  } else if (op->op.same_as(tl::ldg128())) {
    need_copy_h_ = true;
    // Explicit 128-bit global memory load: load_global_128(ptr) or
    // load_global_128_conditional(ptr, pred)
    ICHECK(!args.empty()) << "T.ldg128 expects a pointer argument.";
    if (args.size() > 1) {
      os << "tl::load_global_128_conditional(";
      this->PrintExpr(args[0], os);
      os << ", ";
      this->PrintExpr(args[1], os);
    } else {
      os << "tl::load_global_128(";
      this->PrintExpr(args[0], os);
    }
    os << ")";
  } else if (op->op.same_as(tl::lds32())) {
    need_copy_h_ = true;
    ICHECK_EQ(args.size(), 1U) << "T.lds32 expects a pointer argument.";
    os << "tl::load_shared_32(";
    this->PrintExpr(args[0], os);
    os << ")";
  } else if (op->op.same_as(tl::lds64())) {
    need_copy_h_ = true;
    ICHECK_EQ(args.size(), 1U) << "T.lds64 expects a pointer argument.";
    os << "tl::load_shared_64(";
    this->PrintExpr(args[0], os);
    os << ")";
  } else if (op->op.same_as(tl::lds128())) {
    need_copy_h_ = true;
    ICHECK_EQ(args.size(), 1U) << "T.lds128 expects a pointer argument.";
    os << "tl::load_shared_128(";
    this->PrintExpr(args[0], os);
    os << ")";
  } else if (op->op.same_as(tl::ldg256())) {
    need_copy_sm100_h_ = true;
    // Explicit 256-bit global memory load: load_global_256(ptr) or
    // load_global_256_conditional(ptr, pred). The builtin's result dtype is
    // uint32x8 (ulonglong4), while the pointer argument carries the buffer's
    // element type, so route through the exact ulonglong4 overload instead of
    // letting the generic template deduce (and return) the element type.
    ICHECK(!args.empty()) << "T.ldg256 expects a pointer argument.";
    if (args.size() > 1) {
      os << "tl::load_global_256_conditional((const ulonglong4*)(";
      this->PrintExpr(args[0], os);
      os << "), ";
      this->PrintExpr(args[1], os);
    } else {
      os << "tl::load_global_256((const ulonglong4*)(";
      this->PrintExpr(args[0], os);
      os << ")";
    }
    os << ")";
  } else if (op->op.same_as(tl::stg32())) {
    need_copy_h_ = true;
    // Explicit 32-bit global memory store: store_global_32(ptr, value) or
    // store_global_32_conditional(ptr, value, pred)
    ICHECK(args.size() >= 2) << "T.stg32 expects pointer and value arguments.";
    if (args.size() > 2) {
      os << "tl::store_global_32_conditional(";
      this->PrintExpr(args[0], os);
      os << ", ";
      this->PrintExpr(args[1], os);
      os << ", ";
      this->PrintExpr(args[2], os);
    } else {
      os << "tl::store_global_32(";
      this->PrintExpr(args[0], os);
      os << ", ";
      this->PrintExpr(args[1], os);
    }
    os << ")";
  } else if (op->op.same_as(tl::stg64())) {
    need_copy_h_ = true;
    // Explicit 64-bit global memory store: store_global_64(ptr, value) or
    // store_global_64_conditional(ptr, value, pred)
    ICHECK(args.size() >= 2) << "T.stg64 expects pointer and value arguments.";
    if (args.size() > 2) {
      os << "tl::store_global_64_conditional(";
      this->PrintExpr(args[0], os);
      os << ", ";
      this->PrintExpr(args[1], os);
      os << ", ";
      this->PrintExpr(args[2], os);
    } else {
      os << "tl::store_global_64(";
      this->PrintExpr(args[0], os);
      os << ", ";
      this->PrintExpr(args[1], os);
    }
    os << ")";
  } else if (op->op.same_as(tl::stg128())) {
    need_copy_h_ = true;
    // Explicit 128-bit global memory store: store_global_128(ptr, value) or
    // store_global_128_conditional(ptr, value, pred)
    ICHECK(args.size() >= 2) << "T.stg128 expects pointer and value arguments.";
    if (args.size() > 2) {
      os << "tl::store_global_128_conditional(";
      this->PrintExpr(args[0], os);
      os << ", ";
      this->PrintExpr(args[1], os);
      os << ", ";
      this->PrintExpr(args[2], os);
    } else {
      os << "tl::store_global_128(";
      this->PrintExpr(args[0], os);
      os << ", ";
      this->PrintExpr(args[1], os);
    }
    os << ")";
  } else if (op->op.same_as(tl::sts32())) {
    need_copy_h_ = true;
    ICHECK_EQ(args.size(), 2U)
        << "T.sts32 expects pointer and value arguments.";
    os << "tl::store_shared_32(";
    this->PrintExpr(args[0], os);
    os << ", ";
    this->PrintExpr(args[1], os);
    os << ")";
  } else if (op->op.same_as(tl::sts64())) {
    need_copy_h_ = true;
    ICHECK_EQ(args.size(), 2U)
        << "T.sts64 expects pointer and value arguments.";
    os << "tl::store_shared_64(";
    this->PrintExpr(args[0], os);
    os << ", ";
    this->PrintExpr(args[1], os);
    os << ")";
  } else if (op->op.same_as(tl::sts128())) {
    need_copy_h_ = true;
    ICHECK_EQ(args.size(), 2U)
        << "T.sts128 expects pointer and value arguments.";
    os << "tl::store_shared_128(";
    this->PrintExpr(args[0], os);
    os << ", ";
    this->PrintExpr(args[1], os);
    os << ")";
  } else if (op->op.same_as(tl::stg256())) {
    need_copy_sm100_h_ = true;
    // Explicit 256-bit global memory store: store_global_256(ptr, value) or
    // store_global_256_conditional(ptr, value, pred)
    ICHECK(args.size() >= 2) << "T.stg256 expects pointer and value arguments.";
    if (args.size() > 2) {
      os << "tl::store_global_256_conditional(";
      this->PrintExpr(args[0], os);
      os << ", ";
      this->PrintExpr(args[1], os);
      os << ", ";
      this->PrintExpr(args[2], os);
    } else {
      os << "tl::store_global_256(";
      this->PrintExpr(args[0], os);
      os << ", ";
      this->PrintExpr(args[1], os);
    }
    os << ")";
  } else if (op->op.same_as(tirx::reinterpret_op())) {
    TVM_FFI_ICHECK_EQ(args.size(), 1U);
    if (op->ty.as<PtrTypeObj>() || args[0]->ty.as<PtrTypeObj>()) {
      CodeGenC::Dispatch_(op, os);
      return;
    }
    PrimType tgt_dtype = op->ty.as_or_throw<PrimType>();
    PrimType src_dtype = args[0]->ty.as_or_throw<PrimType>();
    PrimExpr value = args[0].as_or_throw<PrimExpr>();

    // Handle packed float4_e2m1fn reinterpret. The unpacked SMEM dtype is not a
    // general register conversion dtype.
    if (!src_dtype.MatchesElementType(kDLFloat4_e2m1fn, 4) &&
        !tgt_dtype.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
      ICHECK_EQ(tgt_dtype.lanes() * tgt_dtype.bits(),
                src_dtype.lanes() * src_dtype.bits())
          << "reinterpret expects source and target to have the same number of "
             "bits";

      std::string src_val = PrintExpr(value);
      std::string rhs = SSAGetID(
          src_val, src_dtype, tvm::SideEffect(value) == CallEffectKind::kPure);

      // If SSAGetID returns the expression itself (happens when MarkConst was
      // called for constants like -CUDART_INF_F), we need to create a temp
      // variable because we cannot take the address of an rvalue.
      if (rhs == src_val) {
        rhs = name_supply_->FreshName("_reinterpret_tmp");
        PrintIndent();
        PrintType(src_dtype, stream);
        stream << " " << rhs << " = " << src_val << ";\n";
      }

      os << "(*(";
      this->PrintType(tgt_dtype, os);
      os << " *)(&(" << rhs << ")))";
      return;
    }
    if (src_dtype == tgt_dtype || tgt_dtype.lanes() * tgt_dtype.bits() ==
                                      src_dtype.lanes() * src_dtype.bits()) {
      return CodeGenC::Dispatch_(op, os);
    }
    ICHECK_EQ(tgt_dtype.lanes(), src_dtype.lanes())
        << "E2M1 float4 reinterpret expects source and target to have the same "
           "number of lanes. "
        << "Source dtype: " << src_dtype << ", Target dtype: " << tgt_dtype;
    ICHECK_EQ(tgt_dtype.StorageBytes(), src_dtype.StorageBytes())
        << "E2M1 float4 reinterpret expects source and target to have the same "
           "number of bytes. "
        << "Source dtype: " << src_dtype << ", Target dtype: " << tgt_dtype;

    int lanes = tgt_dtype.lanes();

    int ssa_scope = BeginScope();
    if (lanes == 1) {
      // The case of lane=1 is same as the normal reinterpret,
      // except that we allow the src and dst dtype to have different number of
      // bits.
      std::string src_val = PrintExpr(value);
      std::string rhs = SSAGetID(
          src_val, src_dtype, tvm::SideEffect(value) == CallEffectKind::kPure);
      // If SSAGetID returns the expression itself (constant), create temp var
      if (rhs == src_val) {
        rhs = name_supply_->FreshName("_reinterpret_tmp");
        PrintIndent();
        PrintType(src_dtype, stream);
        stream << " " << rhs << " = " << src_val << ";\n";
      }
      os << "(*(";
      this->PrintType(tgt_dtype, os);
      os << " *)(&(" << rhs << ")))";
    } else if (lanes == 2) {
      if (tgt_dtype.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
        // We view the source as an uint16, and then extract bits of two fp4
        // numbers, and finally reinterpret the result as fp4x2.
        value = tirx::reinterpret(PrimType::UInt(16), value);
        PrimVar temp_var("temp_var", PrimType::UInt(16));
        value =
            prim::Let(temp_var, value,
                      prim::Cast(PrimType::UInt(8),
                                 (temp_var & IntImm(PrimType::UInt(16), 0xF)) |
                                     ((temp_var >> 4) &
                                      IntImm(PrimType::UInt(16), 0xF0))));
      } else {
        value = prim::Cast(PrimType::UInt(16),
                           tirx::reinterpret(PrimType::UInt(8), value));
        PrimVar temp_var("temp_var", PrimType::UInt(16));
        value =
            prim::Let(temp_var, value,
                      (temp_var & IntImm(PrimType::UInt(16), 0xF)) |
                          ((temp_var & IntImm(PrimType::UInt(16), 0xF0)) << 4));
      }
      os << PrintExpr(tirx::reinterpret(tgt_dtype, value));
    } else if (lanes == 4) {
      if (tgt_dtype.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
        // We view the source as an uint32, and then extract bits of four fp4
        // numbers, and finally reinterpret the result as fp4x4.
        value = tirx::reinterpret(PrimType::UInt(32), value);
        PrimVar temp_var("temp_var", PrimType::UInt(32));
        value = prim::Let(
            temp_var, value,
            prim::Cast(
                PrimType::UInt(16),
                (temp_var & IntImm(PrimType::UInt(32), 0xF)) |
                    ((temp_var >> 4) & IntImm(PrimType::UInt(32), 0xF0)) |
                    ((temp_var >> 8) & IntImm(PrimType::UInt(32), 0xF00)) |
                    ((temp_var >> 12) & IntImm(PrimType::UInt(32), 0xF000))));
      } else {
        value = prim::Cast(PrimType::UInt(32),
                           tirx::reinterpret(PrimType::UInt(16), value));
        PrimVar temp_var("temp_var", PrimType::UInt(32));
        value = prim::Let(
            temp_var, value,
            (temp_var & IntImm(PrimType::UInt(32), 0xF)) |
                ((temp_var & IntImm(PrimType::UInt(32), 0xF0)) << 4) |
                ((temp_var & IntImm(PrimType::UInt(32), 0xF00)) << 8) |
                ((temp_var & IntImm(PrimType::UInt(32), 0xF000)) << 12));
      }
      os << PrintExpr(tirx::reinterpret(tgt_dtype, value));
    } else {
      TVM_FFI_THROW(InternalError)
          << "Invalid number of lanes for float4_e2m1fn reinterpret: " << lanes;
    }
    EndScope(ssa_scope);
  } else if (op->op.same_as(tirx::gpu_thread_return_op())) {
    os << "return";
  } else if (op->op.same_as(tl::any_sync())) {
    ICHECK_EQ(args.size(), 2U) << "tl.any_sync expects <mask, predicate>.";
    os << "__any_sync(" << PrintExpr(args[0]) << ", " << PrintExpr(args[1])
       << ")";
  } else if (op->op.same_as(tl::all_sync())) {
    ICHECK_EQ(args.size(), 2U) << "tl.all_sync expects <mask, predicate>.";
    os << "__all_sync(" << PrintExpr(args[0]) << ", " << PrintExpr(args[1])
       << ")";
  } else if (op->op.same_as(tl::ballot_sync())) {
    ICHECK_EQ(args.size(), 2U) << "tl.ballot_sync expects <mask, predicate>.";
    // __ballot_sync returns unsigned int (32 bits); zero-extend to uint64.
    os << "((unsigned long long)__ballot_sync(" << PrintExpr(args[0]) << ", "
       << PrintExpr(args[1]) << "))";
  } else if (op->op.same_as(tl::ballot())) {
    ICHECK_EQ(args.size(), 1U) << "tl.ballot expects <predicate>.";
    os << "((unsigned long long)__ballot_sync(0xFFFFFFFFu, "
       << PrintExpr(args[0]) << "))";
  } else if (op->op.same_as(tl::activemask())) {
    ICHECK(args.empty()) << "tl.activemask takes no arguments.";
    os << "((unsigned long long)__activemask())";
  } else if (op->op.same_as(tl::syncthreads_count())) {
    ICHECK_EQ(args.size(), 1U) << "tl.syncthreads_count expects <predicate>.";
    os << "__syncthreads_count(" << PrintExpr(args[0]) << ")";
  } else if (op->op.same_as(tl::syncthreads_and())) {
    ICHECK_EQ(args.size(), 1U) << "tl.syncthreads_and expects <predicate>.";
    os << "__syncthreads_and(" << PrintExpr(args[0]) << ")";
  } else if (op->op.same_as(tl::syncthreads_or())) {
    ICHECK_EQ(args.size(), 1U) << "tl.syncthreads_or expects <predicate>.";
    os << "__syncthreads_or(" << PrintExpr(args[0]) << ")";
  } else if (op->op.same_as(tl::shfl_sync())) {
    ICHECK_EQ(args.size(), 4U)
        << "tl.shfl_sync expects <mask, value, src_lane, width>.";
    os << "__shfl_sync(" << PrintExpr(args[0]) << ", " << PrintExpr(args[1])
       << ", " << PrintExpr(args[2]) << ", " << PrintExpr(args[3]) << ")";
  } else if (op->op.same_as(tl::shfl_xor_sync())) {
    ICHECK_EQ(args.size(), 4U)
        << "tl.shfl_xor_sync expects <mask, value, lane_mask, width>.";
    os << "__shfl_xor_sync(" << PrintExpr(args[0]) << ", " << PrintExpr(args[1])
       << ", " << PrintExpr(args[2]) << ", " << PrintExpr(args[3]) << ")";
  } else if (op->op.same_as(tl::shfl_down_sync())) {
    ICHECK_EQ(args.size(), 4U)
        << "tl.shfl_down_sync expects <mask, value, delta, width>.";
    os << "__shfl_down_sync(" << PrintExpr(args[0]) << ", "
       << PrintExpr(args[1]) << ", " << PrintExpr(args[2]) << ", "
       << PrintExpr(args[3]) << ")";
  } else if (op->op.same_as(tl::shfl_up_sync())) {
    ICHECK_EQ(args.size(), 4U)
        << "tl.shfl_up_sync expects <mask, value, delta, width>.";
    os << "__shfl_up_sync(" << PrintExpr(args[0]) << ", " << PrintExpr(args[1])
       << ", " << PrintExpr(args[2]) << ", " << PrintExpr(args[3]) << ")";
  } else if (op->op.same_as(tl::match_any_sync())) {
    ICHECK_EQ(args.size(), 2U) << "tl.match_any_sync expects <mask, value>.";
    os << "__match_any_sync(" << PrintExpr(args[0]) << ", "
       << PrintExpr(args[1]) << ")";
  } else if (op->op.same_as(tl::match_all_sync())) {
    ICHECK_EQ(args.size(), 2U) << "tl.match_all_sync expects <mask, value>.";
    // __match_all_sync writes a `pred` flag through its third argument. We
    // hide the out-parameter behind an immediately-invoked lambda and
    // discard pred (the returned mask already encodes whether all lanes
    // agreed: a non-zero result implies pred == 1).
    os << "([&]() -> unsigned { int _tl_pred = 0; return __match_all_sync("
       << PrintExpr(args[0]) << ", " << PrintExpr(args[1])
       << ", &_tl_pred); }())";
  } else if (op->op.same_as(tl::get_lane_idx())) {
    need_intrin_h_ = true;
    ICHECK_LE(args.size(), 1)
        << "tl.get_lane_idx expects at most one argument <warp_size>.";
    os << "tl::get_lane_idx(";
    if (!args.empty()) {
      os << PrintExpr(args[0]);
    }
    os << ")";
  } else if (op->op.same_as(tl::get_warp_idx_sync())) {
    need_intrin_h_ = true;
    ICHECK_LE(args.size(), 1)
        << "tl.get_warp_idx_sync expects at most one argument <warp_size>.";
    os << "tl::get_warp_idx_sync(";
    if (!args.empty()) {
      os << PrintExpr(args[0]);
    }
    os << ")";
  } else if (op->op.same_as(tl::get_warp_idx())) {
    need_intrin_h_ = true;
    ICHECK_LE(args.size(), 1)
        << "tl.get_warp_idx expects at most one argument <warp_size>.";
    os << "tl::get_warp_idx(";
    if (!args.empty()) {
      os << PrintExpr(args[0]);
    }
    os << ")";
  } else if (op->op.same_as(tl::get_warp_group_idx())) {
    need_intrin_h_ = true;
    ICHECK_LE(args.size(), 2)
        << "tl.get_warp_group_idx expects <warp_size, warps_per_group>.";
    os << "tl::get_warp_group_idx(";
    for (size_t i = 0; i < args.size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << PrintExpr(args[i]);
    }
    os << ")";
  } else if (op->op.same_as(tl::tl_shuffle_elect())) {
    need_intrin_h_ = true;
    os << "tl::tl_shuffle_elect<" << PrintExpr(args[0]) << ">()";
  } else if (op->op.same_as(tl::initialize_wgmma_descriptor())) {
    ICHECK(args.size() == 5)
        << "tl_initialize_wgmma_descriptor expects 5 arguments but got "
        << args.size();
    auto descriptor = args[0];
    auto start_address = args[1];
    auto layout_type = args[2];
    auto leading_byte_offset = args[3];
    auto stride_byte_offset = args[4];
    os << "tl::initialize_wgmma_descriptor<" << PrintExpr(layout_type) << ", "
       << PrintExpr(leading_byte_offset) << ", "
       << PrintExpr(stride_byte_offset) << ">(" << PrintExpr(descriptor) << ", "
       << PrintExpr(start_address) << ")";
  } else if (op->op.same_as(tl::initialize_tcgen05_descriptor())) {
    ICHECK(args.size() == 7)
        << "tl_initialize_tcgen05_descriptor expects 7 arguments but got "
        << args.size();
    auto descriptor = args[0];
    auto start_address = args[1];
    auto leading_byte_offset = args[2];
    auto stride_byte_offset = args[3];
    auto base_offset = args[4];
    auto leading_abs = args[5];
    auto swizzle_mode = args[6];
    os << "tl::initialize_tcgen05_descriptor(" << PrintExpr(descriptor) << ", "
       << PrintExpr(start_address) << ", " << PrintExpr(leading_byte_offset)
       << ", " << PrintExpr(stride_byte_offset) << ", "
       << PrintExpr(base_offset) << ", " << PrintExpr(leading_abs) << ", "
       << PrintExpr(swizzle_mode) << ")";
  } else if (op->op.same_as(tl::increase_descriptor_offset())) {
    ICHECK(args.size() == 2)
        << "tl_increase_descriptor_offset expects 2 arguments but got "
        << args.size();
    auto descriptor = args[0];
    auto offset = args[1];
    os << "tl::increase_descriptor_offset<int>(" << PrintExpr(descriptor)
       << ", " << PrintExpr(offset) << ")";
  } else if (HandleLateIntrinsicCall(op, os)) {
    // Handled by a helper to keep MSVC's parser away from the giant tail chain.
  } else {
    CodeGenC::Dispatch_(op, os);
  }
}

bool CodeGenTileLangCUDA::TryEmitFastMathCall_(const Call &call,
                                               std::ostream &os) {
  struct FastMathIntrinsic {
    Op op;
    const char *name;
    bool needs_math_header;
  };
  static const FastMathIntrinsic kIntrinsics[] = {
      {tl::__exp(), "exp", true},      {tl::__exp10(), "exp10", false},
      {tl::__log(), "log", true},      {tl::__log2(), "log2", false},
      {tl::__log10(), "log10", false}, {tl::__tan(), "tan", false},
      {tl::__cos(), "cos", true},      {tl::__sin(), "sin", true}};
  for (const auto &intrinsic : kIntrinsics) {
    if (call->op.same_as(intrinsic.op)) {
      CUDAFastMath math_func;
      std::string func_name =
          math_func(call->ty.as_or_throw<PrimType>(), intrinsic.name);
      need_math_h_ = need_math_h_ || intrinsic.needs_math_header;
      os << func_name << "(" << PrintExpr(call->args[0]) << ")";
      return true;
    }
  }
  return false;
}

bool CodeGenTileLangCUDA::TryEmitIEEEMathCall_(const Call &call,
                                               std::ostream &os) {
  if (call->op.same_as(tl::ieee_frsqrt())) {
    CUDAIEEEMath math_func;
    std::string func_name =
        math_func(call->ty.as_or_throw<PrimType>(), "frsqrt", "rn");
    os << func_name << "(" << PrintExpr(call->args[0]) << ")";
    return true;
  }

  struct IEEEMathIntrinsic {
    Op op;
    const char *name;
    int operand_count;
  };
  static const IEEEMathIntrinsic kIntrinsics[] = {
      {tl::ieee_add(), "fadd", 2},  {tl::ieee_sub(), "fsub", 2},
      {tl::ieee_mul(), "fmul", 2},  {tl::ieee_fmaf(), "fmaf", 3},
      {tl::ieee_frcp(), "frcp", 1}, {tl::ieee_fsqrt(), "fsqrt", 1},
      {tl::ieee_fdiv(), "fdiv", 2}};
  for (const auto &intrinsic : kIntrinsics) {
    if (call->op.same_as(intrinsic.op)) {
      CUDAIEEEMath math_func;
      std::string rounding_mode =
          (call->args[intrinsic.operand_count]).as_or_throw<StringImm>()->value;
      std::string func_name = math_func(call->ty.as_or_throw<PrimType>(),
                                        intrinsic.name, rounding_mode);
      os << func_name << "(";
      for (int index = 0; index < intrinsic.operand_count; ++index) {
        if (index != 0) {
          os << ", ";
        }
        os << PrintExpr(call->args[index]);
      }
      os << ")";
      return true;
    }
  }
  return false;
}

bool CodeGenTileLangCUDA::TryEmitPackedMathCall_(const Call &call,
                                                 std::ostream &os) {
  const char *packed_name = GetPackedMathIntrinsicName(call);
  if (packed_name == nullptr) {
    return false;
  }

  // Packed x2 element-wise math intrinsics.
  //
  // For float32x2 the CUDA type is float2 and C++ overload resolution
  // works directly.  For bfloat16x2 / float16x2 the CUDA type is uint1
  // (both map to the same 32-bit struct), so we must cast arguments to
  // the correct native type (__nv_bfloat162 or __half2) and cast the
  // result back to uint1 to avoid the ambiguous uint1 bridge overload.
  const PrimType dtype = call->ty.as_or_throw<PrimType>();
  std::string op_name = packed_name;
  std::vector<PrimExpr> packed_args;
  for (const auto &arg : call->args)
    packed_args.push_back(arg.as_or_throw<PrimExpr>());
  if (call->op.same_as(tl::add2()) && call->args.size() == 2) {
    // Keep explicit packed helper trees on the same fused path for the
    // same reason as PrintVecBinaryOp: NVCC will not reliably rewrite
    // tl::mul2(...) + tl::add2(...) back into packed fma2 on its own.
    auto try_fuse_mul_add = [&](const PrimExpr &mul_expr,
                                const PrimExpr &addend) -> bool {
      const CallObj *mul_call = mul_expr.as<CallObj>();
      if (mul_call == nullptr || !mul_call->op.same_as(tl::mul2()) ||
          mul_call->args.size() != 2 || mul_expr.ty() != dtype ||
          addend.ty() != dtype) {
        return false;
      }
      op_name = "fma2";
      packed_args = {mul_call->args[0].as_or_throw<PrimExpr>(),
                     mul_call->args[1].as_or_throw<PrimExpr>(), addend};
      return true;
    };
    if (!try_fuse_mul_add(call->args[0].as_or_throw<PrimExpr>(),
                          call->args[1].as_or_throw<PrimExpr>())) {
      try_fuse_mul_add(call->args[1].as_or_throw<PrimExpr>(),
                       call->args[0].as_or_throw<PrimExpr>());
    }
  }

  bool need_cast = dtype.MatchesElementType(kDLBfloat, 16) ||
                   dtype.MatchesElementType(kDLFloat, 16);
  std::string native_type;
  if (dtype.MatchesElementType(kDLBfloat, 16)) {
    native_type = "__nv_bfloat162";
  } else if (dtype.MatchesElementType(kDLFloat, 16)) {
    native_type = "__half2";
  }

  // Helper lambda to print a casted argument expression.
  auto print_arg = [&](const PrimExpr &arg) -> std::string {
    std::string arg_str = PrintExpr(arg);
    if (need_cast) {
      return "tl::from_uint1<" + native_type + ">(" + arg_str + ")";
    }
    return arg_str;
  };

  if (need_cast) {
    os << "tl::to_uint1(tl::" << op_name << "(";
  } else {
    os << "tl::" << op_name << "(";
  }

  os << print_arg(packed_args[0]);
  for (size_t index = 1; index < packed_args.size(); ++index) {
    os << ", " << print_arg(packed_args[index]);
  }
  os << ")";

  if (need_cast) {
    os << ")";
  }
  return true;
}

bool CodeGenTileLangCUDA::HandleLateIntrinsicCall(const CallObj *op,
                                                  std::ostream &os) {
  Call call = GetRef<Call>(op);
  const auto operands = op->args;
  if (TryEmitFastMathCall_(call, os) || TryEmitIEEEMathCall_(call, os) ||
      TryEmitPackedMathCall_(call, os)) {
    return true;
  }
  if (op->op.same_as(tl::clamp())) {
    ICHECK_EQ(operands.size(), 3);
    need_math_h_ = true;
    PrimType dtype = op->ty.as_or_throw<PrimType>();
    std::vector<PrimExpr> args;
    for (const auto &arg : op->args)
      args.push_back(arg.as_or_throw<PrimExpr>());
    for (const PrimExpr &arg : args) {
      ICHECK_EQ(arg.ty(), dtype)
          << "tl.clamp operands must have matching types";
    }
    if (!dtype.IsScalar()) {
      // The vector emitters cache printed expressions. Bind effectful inputs
      // before using them so identical calls (or loads around an atomic) are
      // still evaluated independently, once per argument.
      for (PrimExpr &arg : args) {
        if (SideEffect(arg) > CallEffectKind::kPure) {
          PrimVar value("clamp_arg", dtype);
          arg = prim::Let(value, arg, value);
        }
      }
    }
    if (dtype.IsScalar()) {
      os << "tl::clamp(" << PrintExpr(args[0]) << ", " << PrintExpr(args[1])
         << ", " << PrintExpr(args[2]) << ")";
    } else if ((dtype.MatchesElementType(kDLFloat, 16) ||
                dtype.MatchesElementType(kDLBfloat, 16)) &&
               CanEmitPackedX2Math(dtype)) {
      EmitPackedX2Call("clamp2", dtype, args, os);
    } else {
      EmitPerLaneScalarCall("tl::clamp", dtype, args, os);
    }
    return true;
  } else if (op->op.same_as(tl::fma()) || op->op.same_as(tl::fmul())) {
    // Round-to-nearest multiply / fused multiply-add with a guaranteed
    // instruction boundary (never re-contracted or split by NVCC). Scalar
    // forms reuse the IEEE intrinsic names with the fixed "rn" mode; even
    // vector widths lower to packed x2 helpers where the target supports
    // them (see CanEmitPackedX2Math) and fall back to per-lane scalar
    // calls elsewhere.
    bool is_fma = op->op.same_as(tl::fma());
    std::vector<PrimExpr> args;
    for (const auto &arg : op->args)
      args.push_back(arg.as_or_throw<PrimExpr>());
    PrimType t = op->ty.as_or_throw<PrimType>();
    if (!t.IsScalar() && CanEmitPackedX2Math(t)) {
      EmitPackedX2Call(is_fma ? "fma2" : "mul2", t, args, os);
      return true;
    }
    CUDAIEEEMath math_func;
    std::string func_name =
        math_func(t.WithLanes(1), is_fma ? "fmaf" : "fmul", "rn");
    if (t.IsScalar()) {
      os << func_name << "(" << PrintExpr(operands[0]) << ", "
         << PrintExpr(operands[1]);
      if (is_fma) {
        os << ", " << PrintExpr(operands[2]);
      }
      os << ")";
    } else {
      EmitPerLaneScalarCall(func_name, t, args, os);
    }
    return true;
  } else if (op->op.same_as(tl::fast_rcp())) {
    need_math_h_ = true;
    const PrimType dtype = op->ty.as_or_throw<PrimType>();
    ICHECK(dtype.MatchesCode(kDLFloat) && dtype.bits() == 32 &&
           dtype.lanes() == 1)
        << "tl.fast_rcp currently supports scalar float32 only";
    os << "tl::fast_rcp(" << PrintExpr(operands[0]) << ")";
    return true;
  } else if (op->op.same_as(tl::rng_init())) {
    this->need_curand_kernel_h_ = true;
    auto it = rng_state_name_map_.find(op);
    ICHECK(it != rng_state_name_map_.end())
        << "tl.rng_init call was not registered by the AddFunction pre-scan";
    this->curand_random_generator_state = it->second;
    this->curand_random_generator_state_type =
        operands[3].as<StringImmObj>()->value;
    this->PrintIndent();
    this->stream << "curand_init(" << PrintExpr(operands[0]) << ", "
                 << PrintExpr(operands[1]) << ", " << PrintExpr(operands[2])
                 << ", &" << this->curand_random_generator_state << ");\n";
    // The state var is declared at function scope (see AddFunction) so it
    // stays visible to rng_rand / rng_rand_float even when passes split the
    // enclosing block across __syncthreads() barriers.
    return true;
  } else if (op->op.same_as(tl::rng_rand())) {
    this->need_curand_kernel_h_ = true;
    os << "curand(&" << this->curand_random_generator_state << ")";
    return true;
  } else if (op->op.same_as(tl::rng_rand_float())) {
    this->need_curand_kernel_h_ = true;
    os << "curand_" << operands[0].as<StringImmObj>()->value;
    if (op->ty.as_or_throw<PrimType>().bits() == 64) {
      os << "_double";
    }
    os << "(&" << this->curand_random_generator_state << ")";
    return true;
  } else if (op->op.same_as(tl::warp_reduce_sum())) {
    os << "tl::warp_reduce_sum(" << PrintExpr(operands[0]) << ")";
    return true;
  } else if (op->op.same_as(tl::warp_reduce_max())) {
    os << "tl::warp_reduce_max(" << PrintExpr(operands[0]) << ")";
    return true;
  } else if (op->op.same_as(tl::warp_reduce_min())) {
    os << "tl::warp_reduce_min(" << PrintExpr(operands[0]) << ")";
    return true;
  } else if (op->op.same_as(tl::warp_reduce_bitand())) {
    os << "tl::warp_reduce_bitand(" << PrintExpr(operands[0]) << ")";
    return true;
  } else if (op->op.same_as(tl::warp_reduce_bitor())) {
    os << "tl::warp_reduce_bitor(" << PrintExpr(operands[0]) << ")";
    return true;
  } else if (op->op.same_as(tl::atomic_add_elem_op())) {
    need_atomic_h_ = true;
    // atomic_add_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr(operands[0]);
    std::string src_value = PrintExpr(operands[1]);
    this->PrintIndent();
    this->stream << "AtomicAdd(" << dst_ptr << ", " << src_value;
    if (operands.size() > 2) {
      this->stream << ", " << PrintExpr(operands[2]);
    }
    this->stream << ");\n";
    return true;
  } else if (op->op.same_as(tl::atomic_add_ret_elem_op())) {
    need_atomic_h_ = true;
    // atomic_add_ret_elem_op(dst_ptr, src_value[, memory_order]) -> returns
    // prev value
    os << "AtomicAddRet(" << PrintExpr(operands[0]) << ", "
       << PrintExpr(operands[1]);
    if (operands.size() > 2) {
      os << ", " << PrintExpr(operands[2]);
    }
    os << ")";
    return true;
  } else if (op->op.same_as(tl::atomic_addx2_elem_op())) {
    need_atomic_h_ = true;
    // atomic_addx2_elem_op(dst_ptr, src_ptr[, memory_order])
    std::string dst_ptr = PrintExpr(operands[0]);
    std::string src_ptr = PrintExpr(operands[1]);
    this->PrintIndent();
    this->stream << "AtomicAddx2(" << dst_ptr << ", " << src_ptr;
    if (operands.size() > 2) {
      this->stream << ", " << PrintExpr(operands[2]);
    }
    this->stream << ");\n";
    return true;
  } else if (op->op.same_as(tl::atomic_addx2_ret_elem_op())) {
    need_atomic_h_ = true;
    // fp16/bf16 x2 is stored as uint1 but AtomicAddx2Ret returns half2/bf162;
    // bridge with to_uint1. float32 is native float2, emitted bare.
    const PrimType dtype = op->ty.as_or_throw<PrimType>();
    bool need_cast = dtype.MatchesElementType(kDLFloat, 16) ||
                     dtype.MatchesElementType(kDLBfloat, 16);
    if (need_cast) {
      os << "tl::to_uint1(";
    }
    os << "AtomicAddx2Ret(" << PrintExpr(operands[0]) << ", "
       << PrintExpr(operands[1]);
    if (operands.size() > 2) {
      os << ", " << PrintExpr(operands[2]);
    }
    os << ")";
    if (need_cast) {
      os << ")";
    }
    return true;
  } else if (op->op.same_as(tl::atomic_addx4_elem_op())) {
    need_atomic_h_ = true;
    // atomic_addx4_elem_op(dst_ptr, src_ptr[, memory_order])
    std::string dst_ptr = PrintExpr(operands[0]);
    std::string src_ptr = PrintExpr(operands[1]);
    this->PrintIndent();
    this->stream << "AtomicAddx4(" << dst_ptr << ", " << src_ptr;
    if (operands.size() > 2) {
      this->stream << ", " << PrintExpr(operands[2]);
    }
    this->stream << ");\n";
    return true;
  } else if (op->op.same_as(tl::atomic_addx4_ret_elem_op())) {
    need_atomic_h_ = true;
    // AtomicAddx4Ret already returns the store type (float4 / uint2), so bare.
    os << "AtomicAddx4Ret(" << PrintExpr(operands[0]) << ", "
       << PrintExpr(operands[1]);
    if (operands.size() > 2) {
      os << ", " << PrintExpr(operands[2]);
    }
    os << ")";
    return true;
  } else if (op->op.same_as(tl::atomic_load_elem_op())) {
    need_atomic_h_ = true;
    // atomic_load_elem_op(src_ptr, memory_order) -> returns loaded value
    os << "AtomicLoad(" << PrintExpr(operands[0]) << ", "
       << PrintExpr(operands[1]) << ")";
    return true;
  } else if (op->op.same_as(tl::atomic_store_elem_op())) {
    need_atomic_h_ = true;
    // atomic_store_elem_op(dst_ptr, value, memory_order)
    std::string dst_ptr = PrintExpr(operands[0]);
    std::string value = PrintExpr(operands[1]);
    std::string memory_order = PrintExpr(operands[2]);
    this->PrintIndent();
    this->stream << "AtomicStore(" << dst_ptr << ", " << value << ", "
                 << memory_order << ");\n";
    return true;
  } else if (op->op.same_as(tl::atomic_or_elem_op())) {
    need_atomic_h_ = true;
    // atomic_or_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr(operands[0]);
    std::string src_value = PrintExpr(operands[1]);
    this->PrintIndent();
    this->stream << "AtomicOr(" << dst_ptr << ", " << src_value;
    if (operands.size() > 2) {
      this->stream << ", " << PrintExpr(operands[2]);
    }
    this->stream << ");\n";
    return true;
  } else if (op->op.same_as(tl::atomic_max_elem_op())) {
    need_atomic_h_ = true;
    // atomic_max_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr(operands[0]);
    std::string src_value = PrintExpr(operands[1]);
    this->PrintIndent();
    this->stream << "AtomicMax(" << dst_ptr << ", " << src_value;
    if (operands.size() > 2) {
      this->stream << ", " << PrintExpr(operands[2]);
    }
    this->stream << ");\n";
    return true;
  } else if (op->op.same_as(tl::atomic_max_ret_elem_op())) {
    need_atomic_h_ = true;
    // atomic_max_ret_elem_op(dst_ptr, src_value[, memory_order]) -> returns
    // prev value
    os << "AtomicMaxRet(" << PrintExpr(operands[0]) << ", "
       << PrintExpr(operands[1]);
    if (operands.size() > 2) {
      os << ", " << PrintExpr(operands[2]);
    }
    os << ")";
    return true;
  } else if (op->op.same_as(tl::atomic_min_elem_op())) {
    need_atomic_h_ = true;
    // atomic_min_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr(operands[0]);
    std::string src_value = PrintExpr(operands[1]);
    this->PrintIndent();
    this->stream << "AtomicMin(" << dst_ptr << ", " << src_value;
    if (operands.size() > 2) {
      this->stream << ", " << PrintExpr(operands[2]);
    }
    this->stream << ");\n";
    return true;
  } else if (op->op.same_as(tl::atomic_min_ret_elem_op())) {
    need_atomic_h_ = true;
    // atomic_min_ret_elem_op(dst_ptr, src_value[, memory_order]) -> returns
    // prev value
    os << "AtomicMinRet(" << PrintExpr(operands[0]) << ", "
       << PrintExpr(operands[1]);
    if (operands.size() > 2) {
      os << ", " << PrintExpr(operands[2]);
    }
    os << ")";
    return true;
  }
  return false;
}

void CodeGenTileLangCUDA::Dispatch_(const RegionStmtObj *op) {
  if (op->op->name != "tl.attr") {
    CodeGenC::Dispatch_(op);
    return;
  }
  auto attr = DecodeTileLangAttribute(op);
  if (attr.key == "fragment_shape") {
    fragment_shapes[attr.node.cast<Var>().get()] =
        CodegenAnnotationString(attr.value);
  } else if (attr.key == "fragment_layout") {
    fragment_layouts[attr.node.cast<Var>().get()] =
        CodegenAnnotationString(attr.value);
  } else if (attr.key == "threadblock_swizzle_pattern") {
    std::string func_name;
    int64_t panel_size;
    if (auto tuple = attr.value.as<tvm::Tuple>()) {
      TVM_FFI_CHECK_EQ(tuple.value()->fields.size(), 2, ValueError);
      func_name = tuple.value()->fields[0].as_or_throw<StringImm>()->value;
      panel_size = CodegenInt(tuple.value()->fields[1]);
    } else {
      auto values = attr.value.cast<Array<Any>>();
      TVM_FFI_CHECK_EQ(values.size(), 2, ValueError);
      func_name = CodegenAnnotationString(values[0]);
      panel_size = values[1].as<IntImmObj>()
                       ? CodegenInt(values[1].as<IntImmObj>())
                       : values[1].cast<int64_t>();
    }
    TVM_FFI_CHECK_GT(panel_size, 0, ValueError);
    TVM_FFI_CHECK(func_name == "rasterization2DRow" ||
                      func_name == "rasterization2DColumn",
                  ValueError)
        << "threadblock swizzle pattern `" << func_name
        << "` is not supported by CUDA";
    PrintIndent();
    if (cluster_dims) {
      auto [x, y, z] = *cluster_dims;
      TVM_FFI_CHECK(y == 1 && z == 1 && panel_size % x == 0, ValueError)
          << "CUDA swizzle requires one-dimensional clusters and panel_size "
             "divisible by clusterDim.x";
      stream << "const dim3 blockIdx = tl::" << func_name << "WithCluster<"
             << panel_size / x << ", " << x << ">();\n";
    } else {
      stream << "const dim3 blockIdx = tl::" << func_name << "<" << panel_size
             << ">();\n";
    }
  } else if (attr.key == "pragma_unroll_factor") {
    unroll_factor.insert_or_assign(attr.node.cast<Var>().get(),
                                   attr.value.cast<IntImm>());
  } else if (attr.key == "volatile_scope") {
    MarkVolatile(attr.node.cast<Var>().get());
  }
  PrintStmt(op->body);
}

void CodeGenTileLangCUDA::Dispatch_(const BindObj *op) {
  if (IsUnpackedFP4SharedPointer(op->var->ty)) {
    RegisterHandleTypeFromPointer(op->var, &op->value);
    const std::string value = PrintExpr(op->value);
    if (print_ssa_form_) {
      TVM_FFI_ICHECK(!var_idmap_.count(op->var.get()));
      var_idmap_[op->var.get()] = value;
    } else {
      PrintIndent();
      if (IsVolatile(op->var.get()))
        stream << "volatile ";
      PrintType(op->var->ty, stream);
      stream << ' ' << AllocVarID(op->var.get()) << " = " << value << ";\n";
    }
    return;
  }
  const auto *allocation = op->value.as<CallObj>();
  if (!allocation || !allocation->op.same_as(tirx::alloc_tensor_op())) {
    CodeGenC::Dispatch_(op);
    return;
  }
  auto tensor = op->var.as_or_throw<TensorVar>();
  const VarObj *buffer = tensor.get();
  const auto *attrs = allocation->attrs.as<DictAttrsObj>();
  std::string vid = AllocVarID(buffer);
  PrintIndent();
  std::string scope = tensor.scope().name();
  auto memory_scope = tensor.scope();
  alloc_storage_scope_[buffer] = memory_scope;
  PrimType alloc_dtype = tensor.dtype();
  bool is_float4_unpacked_shared = tl::IsFloat4E2M1Unpacked(alloc_dtype) &&
                                   (scope == "shared" || scope == "shared.dyn");
  if (scope.find("wmma.") == 0) {
    if (auto shape = attrs ? attrs->dict.Get("tl.wmma_shape") : std::nullopt)
      fragment_shapes[buffer] = CodegenAnnotationString(*shape);
    if (auto layout = attrs ? attrs->dict.Get("tl.wmma_layout") : std::nullopt)
      fragment_layouts[buffer] = CodegenAnnotationString(*layout);
    if (scope == "wmma.matrix_a" || scope == "wmma.matrix_b") {
      ICHECK(
          alloc_dtype == PrimType::Float(16) ||
          alloc_dtype == PrimType::Int(8) || alloc_dtype == PrimType::UInt(8) ||
          alloc_dtype == PrimType::Int(4) || alloc_dtype == PrimType::UInt(4) ||
          alloc_dtype == PrimType::Int(1) ||
          alloc_dtype == PrimType::BFloat(16))
          << "Matrix_a and matrix_b only support half or char or unsigned char "
          << "or uint4 or int4 or int1 type for now";
    } else {
      ICHECK(alloc_dtype == PrimType::Float(16) ||
             alloc_dtype == PrimType::Float(32) ||
             alloc_dtype == PrimType::Int(32))
          << "Accumulator only support half, float and int type for now";
    }
    PrintWmmaScope(scope, alloc_dtype, buffer, stream);
  } else if (scope == "local.descriptor.wgmma") {
    stream << "tl::GmmaDescriptor " << vid << ";\n";
  } else if (scope == "local.descriptor.tcgen05_smem") {
    stream << "tl::Tcgen05SMemDescriptor " << vid << ";\n";
  } else if (scope == "local.descriptor.tcgen05_instr") {
    stream << "tl::Tcgen05InstrDescriptor " << vid << ";\n";
  } else {
    bool is_fp4_scalar_local =
        alloc_dtype.MatchesElementType(kDLFloat4_e2m1fn, 4) &&
        alloc_dtype.IsScalar() && scope == "local";
    bool is_int4_scalar_local =
        (alloc_dtype == PrimType::Int(4) || alloc_dtype == PrimType::UInt(4)) &&
        alloc_dtype.IsScalar() && scope == "local";
    if (!is_fp4_scalar_local && !is_int4_scalar_local) {
      if (scope == "shared" || scope == "shared.dyn") {
        // Storage planners may combine views in one arena. Its native type
        // carries the strongest constituent alignment; an emitter must not
        // discard that guarantee when declaring the backing allocation.
        const int minimum =
            scope == "shared.dyn" ? 1024 : barrier_alignment_bytes_;
        const int alignment = std::max(minimum, tensor.ty()->data_alignment);
        TVM_FFI_CHECK((alignment & (alignment - 1)) == 0, ValueError)
            << "CUDA shared-memory alignment must be a power of two";
        if (scope == "shared.dyn")
          stream << "extern ";
        stream << "__shared__ __align__(" << alignment << ") ";
      } else {
        PrintStorageScope(memory_scope, stream);
      }
      if (is_float4_unpacked_shared) {
        stream << "uint8_t";
      } else {
        PrintType(alloc_dtype, stream);
      }
    }
  }

  if (scope == "shared.dyn") {
    stream << ' ' << vid << "[];\n";
  } else {
    size_t constant_size = CodegenAllocationElements(tensor);
    if (scope.find("wmma.") == 0) {
      constant_size = GetWmmaFragmentSize(scope, buffer, constant_size);
    }
    bool is_byte_packed_4bit =
        alloc_dtype == PrimType::Int(4) || alloc_dtype == PrimType::UInt(4) ||
        (alloc_dtype.MatchesElementType(kDLFloat4_e2m1fn, 4) &&
         alloc_dtype.IsScalar());
    if (is_byte_packed_4bit && scope == "shared") {
      constant_size = (constant_size + 1) / 2;
    } else if (alloc_dtype == PrimType::Int(1) && scope == "shared") {
      constant_size = constant_size / 32;
    }
    if (scope == "shared") {
      stream << ' ' << vid << '[' << constant_size << "];\n";
    } else if (scope == "shared.barrier" || scope == "shared.cluster_barrier") {
      auto v_id_mem = vid + "_mem";
      stream << ' ' << v_id_mem << "[" << constant_size << "];\n";
      PrintIndent();
      stream << "auto " << vid << " = reinterpret_cast<" << mbarrier_dtype_
             << "*>(" << v_id_mem << ");\n";
    } else if (scope == "local") {
      if (alloc_dtype == PrimType::Int(4) || alloc_dtype == PrimType::UInt(4)) {
        stream << "alignas(16) ";
        PrintType(alloc_dtype, stream);
        stream << ' ' << vid << '[' << (constant_size + 1) / 2 << "];\n";
      } else {
        if (alloc_dtype.MatchesElementType(kDLFloat4_e2m1fn, 4) &&
            alloc_dtype.IsScalar()) {
          auto vid_packed = vid + "_packed";
          stream << "fp4_e2_2_t " << vid_packed << '['
                 << (constant_size + 1) / 2 << "];\n";
          fp4_packed_buffers_[tensor] = vid_packed;
        } else {
          stream << ' ' << vid << '[' << constant_size << "];\n";
        }
      }
    } else if (scope == "local.var") {
      PrimExpr init = prim::MakeConst(alloc_dtype, 0);
      if (auto initializer =
              attrs ? attrs->dict.Get("tl.local_var_init") : std::nullopt) {
        PrimExpr user_init = initializer->cast<PrimExpr>();
        if (!user_init.ty().IsVoid() && user_init.ty() != alloc_dtype) {
          user_init = prim::Cast(alloc_dtype, user_init);
        }
        init = user_init;
      }
      stream << ' ' << vid << " = " << PrintExpr(init) << ";\n";
    } else if (scope.find("local.descriptor") != 0) {
      ICHECK(false) << "Unsupported scope: " << scope << " for buffer "
                    << tensor->name;
    }
  }

  RegisterHandleType(buffer, alloc_dtype);
  if (attrs &&
      CodegenBool(attrs->dict.Get(tirx::attr::kVolatile.name).value_or(false)))
    MarkVolatile(buffer);
}

void CodeGenTileLangCUDA::Dispatch_(const AssertStmtObj *op) {
  // Every function this codegen emits is a __global__ kernel, so a tirx
  // AssertStmt has to lower to the device-legal helper. The inherited CodeGenC
  // visitor streams a host-only TVMFFI error call plus `return -1`, neither of
  // which is valid inside a kernel.
  std::string cond = PrintExpr(op->condition);
  this->PrintIndent();
  if (op->message_parts.empty()) {
    stream << "device_assert(" << cond << ");\n";
    return;
  }
  std::string joined_msg;
  for (const auto &part : op->message_parts) {
    joined_msg += part->value;
  }
  stream << "device_assert_with_msg(" << cond << ", ";
  PrintEscapedCString(joined_msg, stream);
  stream << ");\n";
}

void CodeGenTileLangCUDA::Dispatch_(const EvaluateObj *op) {
  if (op->value.as<IntImmObj>())
    return;
  const CallObj *call = op->value.as<CallObj>();
  if (call && (call->op.same_as(tl::device_assert()))) {
    std::string cond = PrintExpr(call->args[0]);
    this->PrintIndent();
    stream << "device_assert(" << cond << ");\n";
  } else if (call && call->op.same_as(tl::device_assert_with_msg())) {
    std::string cond = PrintExpr(call->args[0]);
    std::string msg_expr = PrintExpr(call->args[1]);
    this->PrintIndent();
    stream << "device_assert_with_msg(" << cond << ", " << msg_expr << ");\n";
  } else {
    CodeGenC::Dispatch_(op);
  }
}

void CodeGenTileLangCUDA::Dispatch_(const ReturnObj *op) {
  const auto *value = op->value.as<IntImmObj>();
  TVM_FFI_CHECK(value && value->value == 0, ValueError)
      << "CUDA kernel return must be the native successful early return, "
         "return 0";
  PrintIndent();
  stream << "return;\n";
}

void CodeGenTileLangCUDA::Dispatch_(const prim::RampObj *op, std::ostream &os) {
  const PrimType dtype = op->ty.as_or_throw<PrimType>();
  int lanes = static_cast<int>(CodegenInt(op->lanes));
  // TODO(chaofan): Comment the ramp lanes limit for now since we have
  // LegalizeVectorizedLoop to automatically legalize vectorized loop whose
  // width exceeds the limit. But we should add check here for safety in the
  // future. The check should be aligned to certain bit width like 128bits or
  // 256bits.

  // ICHECK_LE(lanes, 8) << "Translate Ramp Node " << GetRef<Ramp>(op)
  //                    << "error: " << lanes << " exceeds max ramp lanes 8.";
  bool is_packed_int4x2 =
      (dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt)) &&
      dtype.bits() == 4 && dtype.lanes() == 2;
  os << "(";
  if (is_packed_int4x2) {
    PrintVecConstructor(dtype, os);
  } else {
    os << "make_";
    PrintType(dtype, os);
  }
  os << "(";
  for (int i = 0; i < lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")" << "+(" << PrintExpr(op->stride)
       << "*" << i << ")";
    if (i != lanes - 1)
      os << ", ";
  }
  os << "))";
}

void CodeGenTileLangCUDA::Dispatch_(const TensorLoadObj *op,
                                    std::ostream &os) { // NOLINT(*)
  ICHECK_EQ(op->indices.size(), 1)
      << "Load from non-flat memory not supported.";

  PrimType value_dtype = op->ty.as_or_throw<PrimType>();
  PrimExpr index = op->indices[0];
  Var buffer_var = op->source.as_or_throw<TensorVar>();
  PrimType element_dtype = op->source.as_or_throw<TensorVar>().dtype();

  const bool is_packed_int4_buffer = (element_dtype == PrimType::Int(4) ||
                                      element_dtype == PrimType::UInt(4)) &&
                                     element_dtype.IsScalar();
  const bool is_packed_int4_vector = is_packed_int4_buffer &&
                                     value_dtype.lanes() > 1 &&
                                     value_dtype.WithLanes(1) == element_dtype;
  const bool is_packed_int4x2 =
      is_packed_int4_vector && value_dtype.lanes() == 2;

  std::string vid = GetVarID(buffer_var.get());
  auto print_packed_int4_load = [&](const std::string &idx_str,
                                    std::ostream &stream) {
    if (element_dtype.MatchesCode(kDLUInt)) {
      stream << "tl_uint4_packed_load((const unsigned char*)" << vid << ", "
             << idx_str << ")";
    } else {
      stream << "tl_int4_packed_load((const signed char*)" << vid << ", "
             << idx_str << ")";
    }
  };

  if (is_packed_int4_buffer && value_dtype.IsScalar()) {
    print_packed_int4_load(PrintExpr(index), os);
    return;
  }

  // Check if this is a fp4 packed buffer access
  auto packed_it = fp4_packed_buffers_.find(buffer_var);
  if (packed_it != fp4_packed_buffers_.end() && value_dtype.IsScalar()) {
    std::string idx_str = PrintExpr(index);
    os << "tl_fp4_packed_load(" << packed_it->second << ", " << idx_str << ")";
    return;
  }
  int lanes = value_dtype.lanes();
  // declare type.
  if (value_dtype.lanes() == element_dtype.lanes()) {
    // For scalar fp4 loads from non-packed buffers, use tl_fp4_packed_load
    // to correctly extract the nibble at the given index (the /2 in
    // GetBufferRef maps two consecutive fp4 elements to the same byte, but
    // reading that byte only returns the low nibble — the odd-indexed element
    // is lost).
    if (element_dtype.MatchesElementType(kDLFloat4_e2m1fn, 4) &&
        element_dtype.lanes() == 1) {
      std::string idx_str = PrintExpr(index);
      std::string vid = GetVarID(buffer_var.get());
      os << "tl_fp4_packed_load((fp4_e2_2_t*)" << vid << ", " << idx_str << ")";
    } else {
      std::string ref = GetBufferRef(value_dtype, buffer_var.get(), index);
      HandleVolatileLoads(ref, op, os);
    }
  } else {
    bool can_vector_load = false;
    PVar<PrimExpr> base;
    int ramp_lanes = value_dtype.lanes() / element_dtype.lanes();
    if (tvm::ramp(base, 1, ramp_lanes).Match(index)) {
      can_vector_load = true;

      // A direct packed int4/uint4 vector load reinterprets the underlying
      // bytes as one vector carrier, so its logical base must be aligned to
      // the full vector lane count.
      if (is_packed_int4_vector) {
        can_vector_load = IsProvablyDivisible(base.Eval(), value_dtype.lanes());
      }
    }

    if (can_vector_load) {
      std::string ref = GetVecLoad(value_dtype, buffer_var.get(), base.Eval());
      HandleVolatileLoads(ref, op, os);
    } else {
      std::vector<std::string> scalar_indices;
      if (auto indices = ScalarWideMemoryIndices(index, lanes)) {
        for (const PrimExpr &lane : *indices)
          scalar_indices.push_back(SSAGetID(PrintExpr(lane), lane.ty(), true));
      }
      if (is_packed_int4_vector && !is_packed_int4x2) {
        // A packed vector load that cannot use a directly aligned carrier is
        // still safe to scalarize: loads do not introduce byte-level
        // read-modify-write races. Load each logical nibble and assemble the
        // carrier in a temporary.
        std::string result = name_supply_->FreshName("_");
        this->PrintIndent();
        this->PrintType(value_dtype, stream);
        stream << ' ' << result << ";\n";
        int ssa_scope = BeginScope();
        const prim::RampObj *ramp = index.as<prim::RampObj>();
        std::string sindex;
        if (ramp == nullptr && scalar_indices.empty()) {
          sindex = SSAGetID(PrintExpr(index), index.ty(),
                            tvm::SideEffect(index) == CallEffectKind::kPure);
        }
        for (int i = 0; i < lanes; ++i) {
          std::ostringstream lane_index;
          if (ramp != nullptr) {
            PrimExpr lane =
                sym::Analyzer()->Simplify(ramp->base + ramp->stride * i);
            lane_index << PrintExpr(lane);
          } else if (!scalar_indices.empty()) {
            lane_index << scalar_indices[i];
          } else {
            PrintVecElemLoad(sindex, index.ty(), i, lane_index);
          }
          std::ostringstream lane_value;
          print_packed_int4_load(lane_index.str(), lane_value);
          PrintVecElemStore(result, value_dtype, i, lane_value.str());
        }
        EndScope(ssa_scope);
        os << result;
        return;
      }
      std::ostringstream svalue_expr;
      std::string sindex;
      if (scalar_indices.empty())
        sindex = SSAGetID(PrintExpr(index), index.ty(),
                          tvm::SideEffect(index) == CallEffectKind::kPure);
      PrimType elem_type = value_dtype.WithLanes(1);
      for (int i = 0; i < lanes; ++i) {
        std::ostringstream value_temp;

        if (is_packed_int4x2) {
          std::ostringstream lane_index;
          PrintVecElemLoad(sindex, index.ty(), i, lane_index);
          print_packed_int4_load(lane_index.str(), value_temp);
        } else {
          const bool volatile_bf16 =
              elem_type.MatchesElementType(kDLBfloat, 16) &&
              IsVolatile(buffer_var.get());
          if (!HandleTypeMatch(buffer_var.get(), elem_type) || volatile_bf16) {
            value_temp << "((";
            if (volatile_bf16)
              value_temp << "volatile ";
            if (IsScopePartOfType()) {
              auto it = alloc_storage_scope_.find(buffer_var.get());
              if (it != alloc_storage_scope_.end()) {
                PrintStorageScope(it->second, value_temp);
              }
            }
            PrintType(elem_type, value_temp);
            value_temp << "*)" << vid << ')';
          } else {
            value_temp << vid;
          }
          value_temp << '[';
          if (!scalar_indices.empty())
            value_temp << scalar_indices[i];
          else
            PrintVecElemLoad(sindex, index.ty(), i, value_temp);
          value_temp << ']';
          if (volatile_bf16) {
            const std::string reference = value_temp.str();
            value_temp.str("");
            value_temp << "bfloat16_t::bitcast((" << reference << ").storage)";
          }
        }

        PrintVecElemLoadExpr(value_dtype, i, value_temp.str(), svalue_expr);
      }
      os << svalue_expr.str();
    }
  }
}

void CodeGenTileLangCUDA::Dispatch_(const TensorStoreObj *op) {
  ICHECK_EQ(op->indices.size(), 1) << "Store to non-flat memory not supported.";

  PrimType value_dtype = op->value.ty();
  PrimType element_dtype = op->dest.as_or_throw<TensorVar>().dtype();
  PrimExpr index_expr = op->indices[0];
  Var buffer_var = op->dest.as_or_throw<TensorVar>();

  const bool is_packed_int4_buffer = (element_dtype == PrimType::Int(4) ||
                                      element_dtype == PrimType::UInt(4)) &&
                                     element_dtype.IsScalar();
  const bool is_packed_int4_vector = is_packed_int4_buffer &&
                                     value_dtype.lanes() > 1 &&
                                     value_dtype.WithLanes(1) == element_dtype;

  if (is_packed_int4_buffer && value_dtype.IsScalar()) {
    std::string idx_str = PrintExpr(index_expr);
    std::string value = this->PrintExpr(op->value);
    std::string vid = GetVarID(buffer_var.get());
    this->PrintIndent();
    if (element_dtype.MatchesCode(kDLUInt)) {
      stream << "tl_uint4_packed_store((unsigned char*)" << vid << ", "
             << idx_str << ", " << value << ");\n";
    } else {
      stream << "tl_int4_packed_store((signed char*)" << vid << ", " << idx_str
             << ", " << value << ");\n";
    }
    return;
  }

  // Check if this is a fp4 packed buffer access
  auto packed_it = fp4_packed_buffers_.find(buffer_var);
  if (packed_it != fp4_packed_buffers_.end() && value_dtype.IsScalar()) {
    std::string idx_str = PrintExpr(index_expr);
    std::string value = this->PrintExpr(op->value);
    this->PrintIndent();
    stream << "tl_fp4_packed_store(" << packed_it->second << ", " << idx_str
           << ", " << value << ");\n";
    return;
  }
  if (value_dtype.lanes() == element_dtype.lanes()) {
    // For scalar fp4 stores to non-packed buffers, use tl_fp4_packed_store
    // to correctly handle nibble-level writes. The /2 in GetBufferRef maps two
    // consecutive fp4 elements to the same byte, and a plain assignment
    // overwrites the entire byte — destroying the neighboring nibble.
    if (element_dtype.MatchesElementType(kDLFloat4_e2m1fn, 4) &&
        element_dtype.lanes() == 1) {
      std::string idx_str = PrintExpr(index_expr);
      std::string value = this->PrintExpr(op->value);
      std::string vid = GetVarID(buffer_var.get());
      this->PrintIndent();
      stream << "tl_fp4_packed_store((fp4_e2_2_t*)" << vid << ", " << idx_str
             << ", " << value << ");\n";
    } else {
      std::string value = this->PrintExpr(op->value);
      std::string ref =
          this->GetBufferRef(value_dtype, buffer_var.get(), index_expr);
      this->PrintIndent();
      if (value_dtype.MatchesElementType(kDLBfloat, 16) &&
          IsVolatile(buffer_var.get())) {
        // cutlass::bfloat16_t has no volatile object assignment. Its public
        // uint16_t storage member keeps one volatile 16-bit write and preserves
        // the value bits without a floating-point conversion.
        stream << "(" << ref << ").storage = bfloat16_t(" << value
               << ").storage;\n";
      } else {
        stream << ref << " = " << value << ";\n";
      }
    }
  } else {
    PVar<PrimExpr> base;
    int ramp_lanes = value_dtype.lanes() / element_dtype.lanes();
    bool is_unit_stride_ramp = tvm::ramp(base, 1, ramp_lanes).Match(index_expr);
    if (is_packed_int4_vector &&
        (!is_unit_stride_ramp ||
         !IsProvablyDivisible(base.Eval(), value_dtype.lanes()))) {
      TVM_FFI_THROW(InternalError)
          << "Packed int4/uint4 vector stores require a unit-stride ramp "
             "with a logical base provably divisible by the vector lane "
             "count, but got "
          << index_expr;
    }
    if (is_unit_stride_ramp) {
      std::string value = this->PrintExpr(op->value);
      this->PrintVecStore(buffer_var.get(), value_dtype, base.Eval(), value);
    } else {
      // The assignment below introduces side-effect, and the resulting value
      // cannot be reused across multiple expression, thus a new scope is needed
      int vec_scope = BeginScope();

      // store elements separately
      std::vector<std::string> scalar_indices;
      if (auto indices =
              ScalarWideMemoryIndices(index_expr, value_dtype.lanes())) {
        for (const PrimExpr &lane : *indices)
          scalar_indices.push_back(SSAGetID(PrintExpr(lane), lane.ty(), true));
      }
      std::string index;
      if (scalar_indices.empty())
        index = SSAGetID(PrintExpr(index_expr), index_expr.ty(),
                         tvm::SideEffect(index_expr) == CallEffectKind::kPure);
      std::string value =
          SSAGetID(PrintExpr(op->value), op->value.ty(),
                   tvm::SideEffect(op->value) == CallEffectKind::kPure);
      std::string vid = GetVarID(buffer_var.get());
      for (int i = 0; i < value_dtype.lanes(); ++i) {
        this->PrintIndent();
        PrimType elem_type = value_dtype.WithLanes(1);
        const bool volatile_bf16 =
            elem_type.MatchesElementType(kDLBfloat, 16) &&
            IsVolatile(buffer_var.get());
        if (!HandleTypeMatch(buffer_var.get(), elem_type) || volatile_bf16) {
          stream << "((";
          if (volatile_bf16)
            stream << "volatile ";
          if (IsScopePartOfType()) {
            auto it = alloc_storage_scope_.find(buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, stream);
            }
          }
          PrintType(elem_type, stream);
          stream << "*)" << vid << ')';
        } else {
          stream << vid;
        }
        stream << '[';
        if (!scalar_indices.empty())
          stream << scalar_indices[i];
        else
          PrintVecElemLoad(index, index_expr.ty(), i, stream);
        stream << (volatile_bf16 ? "].storage = " : "] = ");
        // Packed BF16 registers expose CUDA's native scalar type. Memory uses
        // cutlass::bfloat16_t, whose native-BF16 constructor is explicit and
        // preserves the original bits (including signed zero and NaN payloads).
        bool is_bf16 = elem_type.MatchesElementType(kDLBfloat, 16);
        if (is_bf16)
          stream << "bfloat16_t(";
        PrintVecElemLoad(value, op->value.ty(), i, stream);
        if (is_bf16)
          stream << ")";
        if (volatile_bf16)
          stream << ".storage";
        stream << ";\n";
      }
      EndScope(vec_scope);
    }
  }
}

void CodeGenTileLangCUDA::Dispatch_(const prim::SelectObj *op,
                                    std::ostream &os) {
  const PrimType dtype = op->ty.as_or_throw<PrimType>();
  // Non-vector cases.
  if (!op->condition.ty().IsFixedLengthVector()) {
    CodeGenC::Dispatch_(op, os);
    return;
  }

  // Codegen vector condition case by serializing the select op.
  TVM_FFI_ICHECK(op->false_value.ty() == dtype &&
                 op->true_value.ty() == dtype &&
                 dtype.lanes() == op->condition.ty().lanes());

  std::string r_var = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(dtype, stream);
  stream << ' ' << r_var << ";\n";
  {
    std::string c_var =
        SSAGetID(PrintExpr(op->condition), op->condition.ty(),
                 tvm::SideEffect(op->condition) == CallEffectKind::kPure);
    std::string t_var =
        SSAGetID(PrintExpr(op->true_value), dtype,
                 tvm::SideEffect(op->true_value) == CallEffectKind::kPure);
    std::string f_var =
        SSAGetID(PrintExpr(op->false_value), dtype,
                 tvm::SideEffect(op->false_value) == CallEffectKind::kPure);

    // The condition is stored as an ushort vector.
    int lanes = dtype.lanes();
    PrimType memory_ty(kDLUInt, 16, lanes);

    for (int i = 0; i < lanes; ++i) {
      std::ostringstream item;
      item << "(bool(";
      PrintVecElemLoad(c_var, memory_ty, i, item);
      item << ")?";
      PrintVecElemLoad(t_var, dtype, i, item);
      item << ':';
      PrintVecElemLoad(f_var, dtype, i, item);
      item << ')';
      PrintVecElemStore(r_var, dtype, i, item.str());
    }
  }
  os << r_var;
}

void CodeGenTileLangCUDA::Dispatch_(const prim::NotObj *op, std::ostream &os) {
  const PrimType dtype = op->ty.as_or_throw<PrimType>();
  if (!dtype.IsFixedLengthVector()) {
    CodeGenC::Dispatch_(op, os);
    return;
  }

  std::string result = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(dtype, stream);
  stream << ' ' << result << ";\n";
  int ssa_scope = BeginScope();
  {
    std::string value =
        SSAGetID(PrintExpr(op->a), op->a.ty(),
                 tvm::SideEffect(op->a) == CallEffectKind::kPure);
    for (int i = 0; i < dtype.lanes(); ++i) {
      std::ostringstream lane;
      lane << "!bool(";
      PrintVecElemLoad(value, op->a.ty(), i, lane);
      lane << ')';
      PrintVecElemStore(result, dtype, i, lane.str());
    }
  }
  EndScope(ssa_scope);
  os << result;
}

void CodeGenTileLangCUDA::Dispatch_(const prim::BitwiseNotObj *op,
                                    std::ostream &os) {
  PrimType dtype = op->ty.as_or_throw<PrimType>();
  if (!dtype.IsFixedLengthVector() ||
      !(dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt))) {
    CodeGenC::Dispatch_(op, os);
    return;
  }
  // CUDA integer vector carriers have no operator~; XOR with
  // a lane-width all-ones mask preserves both signed and unsigned bits.
  // PrintVecBinaryOp snapshots the operand once before extracting its lanes.
  PrimType scalar_type = dtype.WithLanes(1);
  PrimExpr mask = scalar_type.MatchesCode(kDLUInt)
                      ? prim::max_value(scalar_type)
                      : prim::MakeConst(scalar_type, -1);
  PrintVecBinaryOp("^", dtype, op->a, prim::Broadcast(mask, dtype.lanes()), os);
}

void CodeGenTileLangCUDA::Dispatch_(const prim::ShuffleObj *op,
                                    std::ostream &os) { // NOLINT(*)
  // For bfloat16x2 / float16x2 construction from two scalar lanes, emit a
  // proper pack intrinsic instead of the generic `uint1(a, b)` produced by
  // the base CodeGenC which is not valid CUDA.
  PrimType t = op->ty.as_or_throw<PrimType>();
  bool is_bf16x2 = t.MatchesElementType(kDLBfloat, 16) && t.lanes() == 2;
  bool is_fp16x2 = t.MatchesElementType(kDLFloat, 16) && t.lanes() == 2;
  if ((is_bf16x2 || is_fp16x2) && op->vectors.size() == 2 &&
      op->vectors[0].ty().lanes() == 1 && op->vectors[1].ty().lanes() == 1) {
    // Collect the two scalar element expressions.
    std::string e0 = PrintExpr(op->vectors[0]);
    std::string e1 = PrintExpr(op->vectors[1]);
    if (is_bf16x2) {
      enable_bf16_ = true;
      // __pack_nv_bfloat162(bfloat16_t, bfloat16_t) -> unsigned (32-bit).
      // Use aggregate initialisation of uint1 (struct { unsigned x; })
      // to avoid taking the address of a temporary.
      os << "uint1{__pack_nv_bfloat162(" << e0 << ", " << e1 << ")}";
    } else {
      enable_fp16_ = true;
      os << "uint1{tl::pack_half2(" << e0 << ", " << e1 << ")}";
    }
    return;
  }
  // Handle ExtractElement: extract a scalar lane from a bfloat16x2 / float16x2
  // vector (produced by packed reduction, etc.). The vector is stored as an
  // opaque uint1 in the lowered code, but semantically it is a packed pair.
  PrimType vec_t =
      op->vectors.size() == 1 ? op->vectors[0].ty() : PrimType::Void();
  bool vec_is_bf16x2 =
      vec_t.MatchesElementType(kDLBfloat, 16) && vec_t.lanes() == 2;
  bool vec_is_fp16x2 =
      vec_t.MatchesElementType(kDLFloat, 16) && vec_t.lanes() == 2;
  if ((vec_is_bf16x2 || vec_is_fp16x2) && op->vectors.size() == 1 &&
      op->indices.size() == 1) {
    int lane = CodegenInt(op->indices[0]);
    std::string vec = PrintExpr(op->vectors[0]);
    if (vec_is_bf16x2) {
      enable_bf16_ = true;
      os << "bfloat16_t(((nv_bfloat162*)(&(" << vec << ")))->"
         << (lane == 0 ? "x" : "y") << ")";
    } else {
      enable_fp16_ = true;
      os << "half_t(((half2*)(&(" << vec << ")))->" << (lane == 0 ? "x" : "y")
         << ")";
    }
    return;
  }

  // 32-bit int/uint vectors with lanes > 4 are typed as (u)longlong{lanes/2}
  // by PrintType. The base CodeGenC emits `make_<type>(v0, v1, ..., v_{N-1})`
  // N args for N lanes, which produces e.g. `make_ulonglong4(8 args)` and
  // nvcc rejects with "too many arguments in function call". Pack consecutive
  // pairs of 32-bit values into a single 64-bit lane to match the type's real
  // arity. Triggered e.g. by storing eight T.rng_rand() (uint32) results via
  // a 256-bit vectorized store.
  if ((t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt)) && t.bits() == 32 &&
      t.lanes() > 4 && t.lanes() % 2 == 0) {
    int lanes = t.lanes();
    // Reuse CodeGenC's concat logic by reading individual scalars from the
    // shuffle's source vectors at the indices the shuffle requested.
    std::vector<std::string> scalars;
    scalars.reserve(lanes);
    if (op->vectors.size() == static_cast<size_t>(lanes) &&
        op->indices.size() == static_cast<size_t>(lanes)) {
      // Common case: each lane is a scalar expression in vectors[i] and
      // indices[i] == 0. Print each scalar directly to avoid extra SSA copies.
      for (int i = 0; i < lanes; ++i) {
        scalars.push_back(PrintExpr(op->vectors[i]));
      }
    } else {
      // Fall back to the generic concat-then-shuffle path used by CodeGenC.
      std::vector<std::string> concat_vec;
      for (const PrimExpr &vec : op->vectors) {
        std::string vec_value = PrintExpr(vec);
        if (vec.ty().lanes() == 1) {
          concat_vec.push_back(vec_value);
        } else {
          for (int j = 0; j < vec.ty().lanes(); ++j) {
            std::ostringstream s;
            PrintVecElemLoad(vec_value, vec.ty(), j, s);
            concat_vec.push_back(s.str());
          }
        }
      }
      ICHECK_EQ(static_cast<int>(op->indices.size()), lanes);
      for (const PrimExpr &idx : op->indices) {
        ICHECK(idx->IsInstance<IntImmObj>())
            << "ShuffleObj indices must be constants at codegen time, got "
            << idx;
        int64_t k = CodegenInt(idx);
        ICHECK_LT(k, static_cast<int64_t>(concat_vec.size()));
        scalars.push_back(concat_vec[k]);
      }
    }

    const char *u64 =
        t.MatchesCode(kDLUInt) ? "unsigned long long" : "long long";
    PrintVecConstructor(t, os);
    os << '(';
    for (int i = 0; i + 1 < lanes; i += 2) {
      if (i != 0)
        os << ", ";
      // Pack lane i (lo) and lane i+1 (hi) into one 64-bit value. Widen through
      // `unsigned int` so both lanes are zero-extended.
      os << "(" << u64 << ")(((unsigned long long)(unsigned int)(" << scalars[i]
         << ")) | ((unsigned long long)(unsigned int)(" << scalars[i + 1]
         << ") << 32))";
    }
    os << ')';
    return;
  }

  // Default path for all other types.
  CodeGenC::Dispatch_(op, os);
}

void CodeGenTileLangCUDA::Dispatch_(const prim::BroadcastObj *op,
                                    std::ostream &os) { // NOLINT(*)
  const PrimType dtype = op->ty.as_or_throw<PrimType>();
  int lanes = static_cast<int>(CodegenInt(op->lanes));
  if ((dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt)) &&
      dtype.bits() == 4 && lanes == 2) {
    std::string value = PrintExpr(op->value);
    PrintVecConstructor(dtype, os);
    os << '(' << value << ", " << value << ')';
    return;
  }
  if ((dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt)) &&
      dtype.bits() == 8) {
    const auto *p = op->value.as<IntImmObj>();
    if (p && lanes == 4) {
      // make_int8x4
      int64_t v = CodegenInt(p) & 0xFF;
      v = (v << 24) | (v << 16) | (v << 8) | v;
      if (dtype.MatchesCode(kDLUInt)) {
        os << "(uint)" << v;
      } else {
        os << "(int)" << v;
      }
      return;
    }
    // Replicate the byte across the carrier type (int/int2/int4/longlong4).
    // Restricted to side-effect-free values: an impure value (e.g. a cast of
    // an rng call) must be re-evaluated per lane, which the generic fallback
    // below preserves via the per-lane packing overloads in common.h.
    if ((lanes == 4 || lanes == 8 || lanes == 16 || lanes == 32) &&
        tvm::SideEffect(op->value) <= CallEffectKind::kReadState) {
      std::string sval =
          SSAGetID(PrintExpr(op->value), op->value.ty(),
                   tvm::SideEffect(op->value) == CallEffectKind::kPure);
      os << "tl::broadcast<";
      PrintType(dtype, os);
      os << ">(" << sval << ")";
      return;
    }
  }

  if (dtype.MatchesElementType(kDLFloat, 16)) {
    std::string v = PrintExpr(op->value);
    os << "make_";
    PrintType(dtype, os);
    os << '(';
    if (lanes <= 8) {
      for (int i = 0; i < lanes / 2; ++i) {
        if (i != 0)
          os << ", ";
        os << "__pack_half2(" << v << ", " << v << ")";
      }
    } else {
      for (int i = 0; i < lanes / 4; ++i) {
        if (i != 0)
          os << ", ";
        os << "tl::pack_float16x4(" << v << ", " << v << ", " << v << ", " << v
           << ")";
      }
    }
    os << ')';
    return;
  }

  if (dtype.MatchesElementType(kDLBfloat, 16)) {
    std::string v = PrintExpr(op->value);
    os << "make_";
    PrintType(dtype, os);
    os << '(';
    if (lanes <= 8) {
      for (int i = 0; i < lanes / 2; ++i) {
        if (i != 0)
          os << ", ";
        os << "__pack_nv_bfloat162(" << v << ", " << v << ")";
      }
    } else {
      for (int i = 0; i < lanes / 4; ++i) {
        if (i != 0)
          os << ", ";
        os << "tl::pack_bfloat16x4(" << v << ", " << v << ", " << v << ", " << v
           << ")";
      }
    }
    os << ')';
    return;
  }

  if (dtype.MatchesCode(kDLFloat) && dtype.bits() == 32 && dtype.lanes() == 8) {
    std::string v = PrintExpr(op->value);
    os << "make_ulonglong4(";
    for (int i = 0; i < 4; ++i) {
      if (i != 0)
        os << ", ";
      os << "*(unsigned long long*)&make_float2(" << v << ", " << v << ")";
    }
    os << ')';
    return;
  }

  // 32-bit int/uint broadcast with lanes > 4 lowers to (u)longlong{lanes/2}
  // (each ulonglong holds two consecutive 32-bit lanes). Without packing,
  // the generic loop below would emit one arg per lane, producing e.g.
  // `make_ulonglong4(v, v, v, v, v, v, v, v)` which nvcc rejects with
  // "too many arguments". This path is hit when a vectorized store of
  // `T.rng_rand()` (uint32) is broadcast across the 8 lanes of a 256-bit
  // store. PrintExpr is invoked twice per packed slot so the underlying
  // side-effecting call (curand) executes once per uint32 lane, matching
  // the lane count.
  if ((dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt)) &&
      dtype.bits() == 32 && dtype.lanes() > 4 && dtype.lanes() % 2 == 0) {
    int lanes = dtype.lanes();
    std::string v = PrintExpr(op->value);
    const char *u64 =
        dtype.MatchesCode(kDLUInt) ? "unsigned long long" : "long long";
    os << "make_";
    PrintType(dtype, os);
    os << '(';
    for (int i = 0; i < lanes / 2; ++i) {
      if (i != 0)
        os << ", ";
      // Widen through `unsigned int` so both lanes are zero-extended.
      os << "(" << u64 << ")(((unsigned long long)(unsigned int)(" << v
         << ")) | ((unsigned long long)(unsigned int)(" << v << ") << 32))";
    }
    os << ')';
    return;
  }

  // fp4 vectors are nibble-packed structs: pack one fp4x2 byte and replicate
  // it, which covers every lane count uniformly (including 64 lanes for
  // 256-bit vectorization on sm_100+). Side-effecting values fall through to
  // the generic path, which re-evaluates the expression once per lane.
  if (dtype.MatchesElementType(kDLFloat4_e2m1fn, 4) && lanes % 2 == 0 &&
      tvm::SideEffect(op->value) <= CallEffectKind::kReadState) {
    std::string sval =
        SSAGetID(PrintExpr(op->value), op->value.ty(),
                 tvm::SideEffect(op->value) == CallEffectKind::kPure);
    if (lanes == 2) {
      os << "tl::make_fp4_vec<fp4_e2_2_t>(" << sval << ", " << sval << ")";
    } else {
      os << "tl::broadcast<";
      PrintType(dtype, os);
      os << ">(tl::make_fp4_vec<fp4_e2_2_t>(" << sval << ", " << sval << "))";
    }
    return;
  }

  // fp8 scalars are single bytes: replicate the byte across the vector.
  // Restricted to side-effect-free values -- an impure value (e.g. an rng
  // call) must be re-evaluated per lane, which the generic fallback below
  // preserves by passing one argument per lane to tl::make_vec.
  if (CodegenIsFloat8(dtype) && lanes >= 2 &&
      tvm::SideEffect(op->value) <= CallEffectKind::kReadState) {
    std::string sval =
        SSAGetID(PrintExpr(op->value), op->value.ty(),
                 tvm::SideEffect(op->value) == CallEffectKind::kPure);
    os << "tl::broadcast<";
    PrintType(dtype, os);
    os << ">(" << sval << ")";
    return;
  }

  // Note that packed 4-bit broadcast cannot trivially fallback to the
  // generic make_<Type>(v, ...) path, as it can emit malformed make_int16_t()
  // calls that cause nvcc compilation errors when lanes==4.
  if ((dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt)) &&
      dtype.bits() == 4) {
    const auto *p = op->value.as<IntImmObj>();

    // Materialize for reuse to avoid side-effects.
    std::string sval;
    if (!p) {
      sval = SSAGetID(PrintExpr(op->value), op->value.ty(),
                      tvm::SideEffect(op->value) == CallEffectKind::kPure);
    }

    auto emit_packed_field = [&](int nibbles_per_field) {
      if (p) {
        int64_t v = CodegenInt(p) & 0xF;
        int64_t packed = 0;
        for (int i = 0; i < nibbles_per_field; ++i) {
          packed |= v << (i * 4);
        }
        os << packed;
      } else {
        os << '(';
        for (int i = 0; i < nibbles_per_field; ++i) {
          if (i != 0)
            os << " | ";
          os << "((static_cast<unsigned int>(" << sval << ") & 0x0fu) << "
             << (i * 4) << ")";
        }
        os << ')';
      }
    };

    if (lanes == 4) {
      os << (dtype.MatchesCode(kDLUInt) ? "(uint16_t)" : "(int16_t)");
      emit_packed_field(4);
      return;
    } else if (lanes == 8) {
      os << (dtype.MatchesCode(kDLUInt) ? "(uint)" : "(int)");
      emit_packed_field(8);
      return;
    } else if (lanes == 16 || lanes == 32 || lanes == 64) {
      // Carrier types are (u)int2/(u)int4/(u)longlong4: replicate one packed
      // 32-bit field (8 nibbles) across the vector.
      os << "tl::broadcast<";
      PrintType(dtype, os);
      os << ">(" << (dtype.MatchesCode(kDLUInt) ? "(uint)" : "(int)");
      emit_packed_field(8);
      os << ")";
      return;
    }
  }

  if (auto call = op->value.as<CallObj>()) {
    if (this->curand_random_generator_state_type ==
        "curandStatePhilox4_32_10_t") {
      if (call->op.same_as(tl::rng_rand()) && lanes == 4) {
        os << "curand4(&" << this->curand_random_generator_state << ")";
        return;
      }
      if (call->op.same_as(tl::rng_rand_float())) {
        int bits = call->ty.as_or_throw<PrimType>().bits();
        std::string dist = call->args[0].as<StringImmObj>()->value;
        if (bits == 32) {
          if (lanes == 4) {
            os << "curand_" << dist << "4(&"
               << this->curand_random_generator_state << ")";
            return;
          } else if (lanes == 2 && dist == "normal") {
            os << "curand_normal2(&" << this->curand_random_generator_state
               << ")";
            return;
          }

        } else {
          if (lanes == 2) {
            os << "curand_" << dist << "2_double(&"
               << this->curand_random_generator_state << ")";
            return;
          }
        }
      }
    } else if (this->curand_random_generator_state_type ==
                   "curandStateMRG32k3a_t" ||
               this->curand_random_generator_state_type ==
                   "curandStateXORWOW_t") {
      if (call->op.same_as(tl::rng_rand_float())) {
        int bits = call->ty.as_or_throw<PrimType>().bits();
        std::string dist = call->args[0].as<StringImmObj>()->value;
        if (bits == 32) {
          if (lanes == 2 && dist == "normal") {
            os << "curand_normal2(&" << this->curand_random_generator_state
               << ")";
            return;
          }
        } else {
          if (lanes == 2 && dist == "normal") {
            os << "curand_normal2_double(&"
               << this->curand_random_generator_state << ")";
            return;
          }
        }
      }
    }
  }

  std::string v = PrintExpr(op->value);
  if (dtype.MatchesElementType(kDLFloat4_e2m1fn, 4) || CodegenIsFloat8(dtype)) {
    // Only side-effecting fp4/fp8 values reach here (pure ones take the
    // tl::broadcast branches above); the variadic packer takes one argument
    // per lane, so the expression is re-evaluated per lane as required.
    PrintVecConstructor(dtype, os);
  } else {
    os << "make_";
    PrintType(dtype, os);
  }
  os << '(';
  for (int i = 0; i < lanes; ++i) {
    if (i != 0)
      os << ", ";
    os << v;
  }
  os << ')';
}

inline void PrintConst(const FloatImmObj *op, std::ostream &os,
                       CodeGenTileLangCUDA *p) { // NOLINT(*)
  const PrimType dtype = op->ty.as_or_throw<PrimType>();
  // Type code is kBFloat/kFloat16
  // which is indeed CUTLASS supported types currently
  if (dtype.MatchesElementType(kDLBfloat, 16) ||
      dtype.MatchesElementType(kDLFloat, 16)) {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      if (op->value < 0) {
        temp << "-";
      }
      temp << "std::numeric_limits<";
      p->PrintType(dtype, temp);
      temp << ">::infinity()";
    } else if (std::isnan(op->value)) {
      temp << "std::numeric_limits<";
      p->PrintType(dtype, temp);
      temp << ">::quiet_NaN()";
    } else {
      p->PrintType(dtype, temp);
      temp << '(' << FlexibleHexFormat(op->value) << 'f';
      temp << "/*" << std::scientific << op->value << "*/";
      temp << ')';
    }
    p->MarkConst(temp.str());
    os << temp.str();
    return;
  }
  // Type code is kTensorFloat32
  // infinity() and quiet_NaN() for tfloat32_t are not inlined in CUTLASS
  // currently
  if (tl::IsTensorFloat32(dtype)) {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      if (op->value < 0) {
        temp << "-";
      }
      temp << "tfloat32_t::bitcast(0x7f800000)";
    } else if (std::isnan(op->value)) {
      temp << "tfloat32_t::bitcast(0x7fffffff)";
    } else {
      p->PrintType(dtype, temp);
      temp << '(' << std::hexfloat << op->value << 'f';
      temp << "/*" << std::scientific << op->value << "*/";
      temp << ')';
    }
    p->MarkConst(temp.str());
    os << temp.str();
    return;
  }
  // Type code is kFloat8_e5m2 or kE4M4Float
  if (CodegenIsFloat8(dtype) || CodegenIsFloat4(dtype)) {
    // e5m2 inf/NaN have no float-literal spelling; emit the bit pattern.
    if (dtype.MatchesElementType(kDLFloat8_e5m2, 8) && std::isinf(op->value)) {
      p->PrintType(dtype, os);
      os << "::bitcast(" << (op->value < 0 ? "0xfc" : "0x7c") << ")";
      return;
    }
    if (dtype.MatchesElementType(kDLFloat8_e5m2, 8) && std::isnan(op->value)) {
      p->PrintType(dtype, os);
      os << "::bitcast(0x7e)";
      return;
    }
    p->PrintType(dtype, os);
    os << '(' << FlexibleHexFormat(op->value) << 'f';
    os << "/*" << std::scientific << op->value << "*/";
    os << ')';
    return;
  }
  // Type code is kFloat64/kFloat32 (kFloat16 is handled above)
  switch (dtype.bits()) {
  case 64:
  case 32: {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      if (op->value < 0) {
        temp << "-";
      }
      temp << ((dtype.bits() == 32) ? "CUDART_INF_F" : "CUDART_INF");
      p->need_math_constants_h_ = true;
    } else if (std::isnan(op->value)) {
      temp << ((dtype.bits() == 32) ? "CUDART_NAN_F" : "CUDART_NAN");
      p->need_math_constants_h_ = true;
    } else {
      temp << FlexibleHexFormat(op->value);
      if (dtype.bits() == 32)
        temp << 'f';
      temp << "/*" << std::scientific << op->value << "*/";
    }
    p->MarkConst(temp.str());
    os << temp.str();
    break;
  }
  default:
    TVM_FFI_THROW(InternalError)
        << "Bad bit-width for float: " << dtype << "\n";
  }
}

void CodeGenTileLangCUDA::Dispatch_(const FloatImmObj *op,
                                    std::ostream &os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenTileLangCUDA::PrintWmmaScope(const std::string &scope,
                                         const PrimType &t,
                                         const VarObj *variable,
                                         std::ostream &os) {
  std::stringstream type;
  PrintType(t, type);
  ICHECK(fragment_shapes.count(variable))
      << "Cannot find shape of the wmma fragment " << variable->name;
  std::string shape_str = fragment_shapes.at(variable);
  if ((t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt)) && t.bits() < 8 &&
      t.lanes() == 1) {
    type.str(std::string());
    if (t.MatchesCode(kDLInt)) {
      if (t.bits() == 4) {
        type << "nvcuda::wmma::experimental::precision::s4";
      } else if (t.bits() == 1) {
        type << "nvcuda::wmma::experimental::precision::b1";
      } else {
        TVM_FFI_THROW(InternalError)
            << "Unhandled integer type for wmma fragment!";
      }
    } else if (t.MatchesCode(kDLUInt)) {
      if (t.bits() == 4) {
        type << "nvcuda::wmma::experimental::precision::u4";
      } else {
        TVM_FFI_THROW(InternalError)
            << "Unhandled integer type for wmma fragment!";
      }
    }
  }
  if (scope == "wmma.matrix_a") {
    std::string layout_str = fragment_layouts[variable];
    ICHECK_NE(layout_str, "") << "Layout must be defined for matrix_a";
    os << "nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, " << shape_str << ", "
       << type.str() << ", nvcuda::wmma::" << layout_str << ">";
  } else if (scope == "wmma.matrix_b") {
    std::string layout_str = fragment_layouts[variable];
    ICHECK_NE(layout_str, "") << "Layout must be defined for matrix_b";
    os << "nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, " << shape_str << ", "
       << type.str() << ", nvcuda::wmma::" << layout_str << ">";
  } else if (scope == "wmma.accumulator") {
    os << "nvcuda::wmma::fragment<nvcuda::wmma::accumulator, " << shape_str
       << ", " << type.str() << ">";
  }
}

int32_t CodeGenTileLangCUDA::GetWmmaFragmentSize(const std::string &scope,
                                                 const VarObj *variable,
                                                 int32_t size) {
  ICHECK(fragment_shapes.count(variable))
      << "Cannot find shape of the wmma fragment " << variable->name;
  std::string shape_str = fragment_shapes.at(variable);
  // Metadata records the native WMMA shape as "m, n, k".  Each declared
  // fragment contains the two dimensions selected by its operand role.
  std::istringstream fields(shape_str);
  int32_t m, n, k;
  char first, second;
  TVM_FFI_CHECK(static_cast<bool>(fields >> m >> first >> n >> second >> k) &&
                    first == ',' && second == ',' && m > 0 && n > 0 && k > 0,
                ValueError)
      << "Malformed WMMA fragment shape " << shape_str;
  int32_t rows = scope == "wmma.matrix_b" ? k : m;
  int32_t columns = scope == "wmma.matrix_a" ? k : n;
  return size / rows / columns;
}

void CodeGenTileLangCUDA::HandleVolatileLoads(const std::string &value,
                                              const TensorLoadObj *op,
                                              std::ostream &os) {
  const PrimType dtype = op->ty.as_or_throw<PrimType>();
  // Only the physical access is volatile, not the resulting register value.
  // CUTLASS BF16 cannot copy a volatile object: load its integer storage member
  // once and construct the same BF16 bits in a non-volatile temporary.
  if (dtype.IsScalar() && dtype.MatchesElementType(kDLBfloat, 16) &&
      IsVolatile(op->source.as_or_throw<TensorVar>().get())) {
    os << "bfloat16_t::bitcast((" << value << ").storage)";
  } else if ((dtype.MatchesElementType(kDLFloat, 16) ||
              dtype.MatchesElementType(kDLBfloat, 16)) &&
             IsVolatile(op->source.as_or_throw<TensorVar>().get())) {
    os << "(";
    PrintType(dtype, os);
    os << ")(" << value << ")";
  } else {
    os << value;
  }
}

void CodeGenTileLangCUDA::PrintVecElemLoadExpr(const PrimType &t, int i,
                                               const std::string &value,
                                               std::ostream &os) {
  ICHECK_GT(t.lanes(), 1);
  if (t.bits() == 32 && t.lanes() > 4 &&
      t.MatchesCode(kDLInt, kDLUInt, kDLFloat)) {
    // CUDA represents eight 32-bit values with four 64-bit carrier fields.
    // The built-in make_(u)longlong4 takes four carrier values, not eight
    // scalar values. Pack the original lane bits with the existing helper.
    if (i == 0) {
      os << "tl::make_vec<";
      PrintType(t, os);
      os << ">(";
    }
    os << value << (i == t.lanes() - 1 ? ")" : ",");
    return;
  }
  if ((t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt)) && t.bits() == 4 &&
      t.lanes() == 2) {
    if (i == 0) {
      PrintVecConstructor(t, os);
      os << '(';
    }
    os << value;
    os << (i == t.lanes() - 1 ? ")" : ",");
    return;
  }
  if (t.bits() == 8 && (t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt))) {
    if (!(t.lanes() == 2 || t.lanes() == 3)) {
      if (i != 0) {
        os << "|";
      }
      os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8
         << "))";
      return;
    }
  }

  if (t.MatchesElementType(kDLFloat, 16)) {
    if (i == 0) {
      os << "make_";
      PrintType(t, os);
      os << '(';
    }
    if (i % 2 == 0) {
      os << "__pack_half2(" << value;
    } else {
      os << "," << value << ")";
      if (i != t.lanes() - 1) {
        os << ",";
      } else {
        os << ")";
      }
    }
    return;
  }

  if (t.MatchesElementType(kDLBfloat, 16)) {
    if (i == 0) {
      os << "make_";
      PrintType(t, os);
      os << '(';
    }
    if (i % 2 == 0) {
      os << "__pack_nv_bfloat162(" << value;
    } else {
      os << "," << value << ")";
      if (i != t.lanes() - 1) {
        os << ",";
      } else {
        os << ")";
      }
    }
    return;
  }

  if (i == 0) {
    if (t.MatchesElementType(kDLFloat4_e2m1fn, 4) || CodegenIsFloat8(t)) {
      // Emits the variadic tl::make_(fp4_)vec packer; the make_<type>
      // constructors do not exist for these struct types.
      PrintVecConstructor(t, os);
    } else {
      os << "make_";
      PrintType(t, os);
    }
    os << "(";
  }
  os << value;
  if (i != t.lanes() - 1) {
    os << ",";
  } else {
    os << ")";
  }
  return;
}

void CodeGenTileLangCUDA::PrintFunctionSignature(const String &function_name,
                                                 const tirx::Function &func,
                                                 std::ostream &os) {
  PrintFuncPrefix(os);
  CodeGenC::PrintType(func->ret_type, os);
  PrintExtraAttrs(func, os);
  bool no_alias = func->HasNonzeroAttr(tirx::attr::kNoAlias.name);
  // NVCC rejects restrictive pointer assumptions with PDL synchronization.
  bool has_cuda_pdl_sync = func->HasNonzeroAttr("has_cuda_pdl_sync");
  std::unordered_set<const VarObj *> non_restrict;
  if (auto opt = func->GetAttr<Array<Var>>("tl.non_restrict_params")) {
    for (const Var &var : opt.value())
      non_restrict.insert(var.get());
  }
  std::unordered_set<size_t> readonly;
  if (auto indices = func->GetAttr<Array<Any>>("tl.readonly_param_indices")) {
    for (const auto &value : indices.value()) {
      int64_t index = value.as<IntImmObj>() ? CodegenInt(value.as<IntImmObj>())
                                            : value.cast<int64_t>();
      TVM_FFI_CHECK(index >= 0 &&
                        static_cast<size_t>(index) < func->params.size(),
                    ValueError)
          << "Read-only CUDA parameter index out of range";
      readonly.insert(static_cast<size_t>(index));
    }
  }
  os << ' ' << function_name << '(';
  for (size_t i = 0; i < func->params.size(); ++i) {
    Var var = func->params[i];
    if (i)
      os << ", ";
    const auto *ptr = var->ty.as<PtrTypeObj>();
    if (ptr && ptr->element_type.as<tirx::TensorMapTypeObj>()) {
      // Native tensor maps are by-value grid-constant kernel arguments.
      os << "__grid_constant__ const CUtensorMap";
    } else if (ptr && ptr->scope.name() == "grid_constant") {
      os << "__grid_constant__ const ";
      CodeGenC::PrintType(ptr->element_type, os);
    } else {
      if (ptr) {
        if (auto it = alloc_storage_scope_.find(var.get());
            it != alloc_storage_scope_.end())
          PrintStorageScope(it->second, os);
        if (readonly.count(i))
          os << "const ";
      }
      CodeGenC::PrintType(var->ty, os);
      if (ptr) {
        if (auto dtype = ptr->element_type.as<PrimType>())
          RegisterHandleType(var.get(), dtype.value());
        if (!has_cuda_pdl_sync && no_alias && !non_restrict.count(var.get()))
          PrintRestrict(var, os);
      }
    }
    os << ' ' << AllocVarID(var.get());
  }
  os << ')';
}

void CodeGenTileLangCUDA::AddFunction(const GlobalVar &gvar,
                                      const tirx::Function &func) {
  if (auto source = func->GetAttr<String>("code_block_source")) {
    auto symbol = func->GetAttr(tvm::attr::kGlobalSymbol);
    TVM_FFI_CHECK(symbol.has_value(), ValueError)
        << "CUDA source kernel requires global_symbol";
    if (auto entry = func->GetAttr<String>("code_block_entry_name")) {
      TVM_FFI_CHECK_EQ(symbol.value(), entry.value(), ValueError)
          << "T.CUDASourceCodeKernel requires global_symbol to match "
             "entry_name";
    }
    stream << source.value() << "\n\n";
    return;
  }
  CodeGenC::DeclareFunction(gvar, func);
  if (func->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return;
  sym::Analyzer index_analyzer;
  auto index_normalizer =
      ffi::make_object<PackedMemoryIndexNormalizer>(index_analyzer.get());
  const auto body = index_normalizer->Mutate(func->body, InplaceMode::kDisallow)
                        .ValueOrUnchanged(func->body);
  InitFuncState(func);
  barrier_count_ = -1;
  legacy_barrier_name_.clear();
  fragment_shapes.clear();
  fragment_layouts.clear();
  unroll_factor.clear();
  fp4_packed_buffers_.clear();
  ReserveKeywordsAsUnique_();
  auto symbol = func->GetAttr(tvm::attr::kGlobalSymbol);
  TVM_FFI_CHECK(symbol.has_value(), ValueError)
      << "CUDA kernel requires global_symbol";
  cluster_dims = ClusterInfoExtractor().extract(func);
  PrintFunctionSignature(symbol.value(), func, stream);
  stream << " {\n";
  PreFunctionBody(func);
  // CUDA RNG state is function-scoped: synchronization lowering can split the
  // lexical block containing rng_init before a later rng_rand use.
  rng_state_name_map_.clear();
  ffi::StructuralVisit(
      body,
      [this](const CallObj *call, ffi::StructuralVisitorObj *visitor)
          -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        if (call->op.same_as(tl::rng_init()) &&
            !rng_state_name_map_.count(call)) {
          TVM_FFI_CHECK_GE(call->args.size(), 4, ValueError);
          need_curand_kernel_h_ = true;
          std::string name =
              name_supply_->FreshName("__random_generator_state");
          stream << "  " << call->args[3].as_or_throw<StringImm>()->value << ' '
                 << name << ";\n";
          rng_state_name_map_.emplace(call, std::move(name));
        }
        return visitor->DefaultVisitExpected(call);
      },
      [](const TypeObj *, ffi::StructuralVisitorObj *)
          -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        return std::nullopt;
      });
  int scope = BeginScope();
  PrintStmt(UnwrapRedundantLexicalScope(body));
  EndScope(scope);
  PrintIndent();
  stream << "}\n\n";
}

} // namespace codegen
} // namespace tvm
