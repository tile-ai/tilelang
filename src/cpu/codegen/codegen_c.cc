/*!
 * \file cpu/codegen/codegen_c.cc
 */
#include "codegen_c.h"
#include "op/builtin.h"

#include <cmath>
#include <sstream>

namespace tvm::codegen {

void CodeGenTileLangC::Init(bool output_ssa, bool emit_asserts,
                            bool emit_fwd_func_decl, std::string target_str,
                            const std::unordered_set<std::string> &devices) {
  tl::CodeGenCHost::Init(output_ssa, emit_asserts, emit_fwd_func_decl,
                         target_str, devices);
  preamble_stream_ << "#include <tl_templates/cpp/common.h>\n\n";
  omp_chain_members_.clear();
}

void CodeGenTileLangC::Dispatch_(const CallObj *op, std::ostream &os) {
  if (op->op.same_as(tl::clamp())) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 3);
    os << "tl::clamp(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ", " << PrintExpr(op->args[2]) << ")";
  } else if (op->op.same_as(prim::fma_op())) {
    // Explicit FMA is fused. Do not install an FLowerIntrinsic entry for it:
    // LowerIntrin would then also start contracting ordinary a*b+c on CPUs.
    PrimType type = op->ty.as_or_throw<PrimType>();
    TVM_FFI_CHECK(type.IsScalar() && type.MatchesCode(kDLFloat) &&
                      (type.bits() == 32 || type.bits() == 64),
                  TypeError)
        << "CPU fma requires scalar float32 or float64, got " << type;
    PrintCallExtern(type, type.bits() == 32 ? "fmaf" : "fma", op->args, false,
                    os);
  } else {
    tl::CodeGenCHost::Dispatch_(op, os);
  }
}

void CodeGenTileLangC::InitFuncState(const Function &func) {
  tl::CodeGenCHost::InitFuncState(func);
  name_supply_->ReserveName("fma");
  name_supply_->ReserveName("fmaf");
}

void CodeGenTileLangC::PrintVecElemLoad(const std::string &vec,
                                        const PrimType &type, int lane,
                                        std::ostream &os) {
  TVM_FFI_ICHECK(lane >= 0 && lane < type.lanes());
  os << '(' << vec << ")[" << lane << ']';
}

void CodeGenTileLangC::PrintVecElemStore(const std::string &vec,
                                         const PrimType &type, int lane,
                                         const std::string &value) {
  PrintIndent();
  PrintVecElemLoad(vec, type, lane, stream);
  stream << " = " << value << ";\n";
}

void CodeGenTileLangC::PrintVecElemLoadExpr(const PrimType &type, int lane,
                                            const std::string &value,
                                            std::ostream &os) {
  // CPU byte vectors have the same lane-array representation as other types;
  // the shared C emitter's packed integer expression is not that storage.
  TVM_FFI_ICHECK(type.lanes() > 1 && lane >= 0 && lane < type.lanes());
  if (lane == 0) {
    PrintVecConstructor(type, os);
    os << '(';
  } else {
    os << ", ";
  }
  os << value;
  if (lane + 1 == type.lanes())
    os << ')';
}

void CodeGenTileLangC::Dispatch_(const prim::BroadcastObj *op,
                                 std::ostream &os) {
  PrintVecConstructor(op->ty.as_or_throw<PrimType>(), os);
  // The scalar constructor broadcasts one evaluation, including a memory read.
  os << '(' << PrintExpr(op->value) << ')';
}

void CodeGenTileLangC::PrintType(const PrimType &type, std::ostream &os) {
  if (type.MatchesCode(kDLBool) && type.IsFixedLengthVector()) {
    os << "vec_type<bool, " << type.lanes() << ">";
    return;
  }
  tl::CodeGenCHost::PrintType(type, os);
}

void CodeGenTileLangC::PrintVecBinaryOp(const std::string &op,
                                        const PrimType &type, PrimExpr lhs,
                                        PrimExpr rhs, std::ostream &os) {
  if (!type.MatchesCode(kDLBool))
    return tl::CodeGenCHost::PrintVecBinaryOp(op, type, lhs, rhs, os);
  // Comparisons return booleans while their operands retain signedness and
  // precision. Materialize each operand once, including an effectful load.
  std::string left = PrintExprToSSA(lhs), right = PrintExprToSSA(rhs);
  std::string result = name_supply_->FreshName("predicate");
  PrintIndent();
  PrintType(type, stream);
  stream << ' ' << result << ";\n";
  for (int lane = 0; lane < type.lanes(); ++lane) {
    std::ostringstream value;
    value << '(';
    if (lhs.ty().IsScalar())
      value << left;
    else
      PrintVecElemLoad(left, lhs.ty(), lane, value);
    value << ' ' << op << ' ';
    if (rhs.ty().IsScalar())
      value << right;
    else
      PrintVecElemLoad(right, rhs.ty(), lane, value);
    value << ')';
    PrintVecElemStore(result, type, lane, value.str());
  }
  os << result;
}

void CodeGenTileLangC::Dispatch_(const prim::NotObj *op, std::ostream &os) {
  PrimType type = op->ty.as_or_throw<PrimType>();
  if (type.IsScalar())
    return tl::CodeGenCHost::Dispatch_(op, os);
  std::string operand = PrintExprToSSA(op->a);
  std::string result = name_supply_->FreshName("predicate");
  PrintIndent();
  PrintType(type, stream);
  stream << ' ' << result << ";\n";
  for (int lane = 0; lane < type.lanes(); ++lane) {
    std::ostringstream value;
    value << "(!";
    PrintVecElemLoad(operand, type, lane, value);
    value << ')';
    PrintVecElemStore(result, type, lane, value.str());
  }
  os << result;
}

void CodeGenTileLangC::Dispatch_(const prim::SelectObj *op, std::ostream &os) {
  PrimType type = op->ty.as_or_throw<PrimType>();
  if (type.IsScalar())
    return tl::CodeGenCHost::Dispatch_(op, os);
  // Native Select evaluates its condition and both arms. Use scalar ternary
  // operators only after each whole vector operand has been evaluated once.
  std::string condition = PrintExprToSSA(op->condition);
  std::string on_true = PrintExprToSSA(op->true_value);
  std::string on_false = PrintExprToSSA(op->false_value);
  std::string result = name_supply_->FreshName("selected");
  PrintIndent();
  PrintType(type, stream);
  stream << ' ' << result << ";\n";
  for (int lane = 0; lane < type.lanes(); ++lane) {
    std::ostringstream value;
    value << '(';
    if (op->condition.ty().IsScalar())
      value << condition;
    else
      PrintVecElemLoad(condition, op->condition.ty(), lane, value);
    value << " ? ";
    PrintVecElemLoad(on_true, type, lane, value);
    value << " : ";
    PrintVecElemLoad(on_false, type, lane, value);
    value << ')';
    PrintVecElemStore(result, type, lane, value.str());
  }
  os << result;
}

void CodeGenTileLangC::PrintStorageScope(const runtime::MemoryScope &scope,
                                         std::ostream &os) {
  // CPU allocations are ordinary arrays. The scope still participates in IR
  // analysis, but does not add a C address-space or storage qualifier.
  // alloc_shared defaults to shared.dyn even when its shape is constant.
  const std::string name = scope.name();
  TVM_FFI_CHECK(name == "global" || name == "shared" || name == "shared.dyn" ||
                    name == "local" || name.rfind("local.", 0) == 0,
                ValueError)
      << "Unsupported TileLang CPU allocation scope " << name;
}

void CodeGenTileLangC::Dispatch_(const ForObj *op) {
  if (op->kind == ForKind::kParallel && omp_chain_members_.count(op) == 0) {
    auto saved_members = omp_chain_members_;
    int depth = 1;
    auto single_loop = [](const SeqStmt &body) -> const ForObj * {
      return body->seq.size() == 1 ? body->seq[0].as<ForObj>() : nullptr;
    };
    for (const ForObj *child = single_loop(op->body);
         child && child->kind == ForKind::kParallel;
         child = single_loop(child->body)) {
      omp_chain_members_.insert(child);
      ++depth;
    }
    PrintIndent();
    stream << "#pragma omp parallel for";
    if (depth > 1)
      stream << " collapse(" << depth << ")";
    if (auto num_threads = op->annotations.Get("tl.cpu_num_threads")) {
      int64_t count;
      if (const auto *literal = num_threads->as<IntImmObj>()) {
        count = literal->value.as<int64_t>().value();
      } else {
        count = num_threads->cast<int64_t>();
      }
      TVM_FFI_CHECK_GT(count, 0, ValueError)
          << "OpenMP num_threads must be positive";
      stream << " num_threads(" << count << ")";
    }
    stream << '\n';
    CodeGenC::Dispatch_(op);
    omp_chain_members_ = std::move(saved_members);
    return;
  }
  CodeGenC::Dispatch_(op);
}

void CodeGenTileLangC::Dispatch_(const FloatImmObj *op, std::ostream &os) {
  // C/C++ have no `inff` or `nanf` numeric literals. The template header owns
  // half support; finite formatting continues to use the shared emitter.
  if (std::isinf(op->value)) {
    const std::string literal = op->value < 0 ? "-INFINITY" : "INFINITY";
    MarkConst(literal);
    os << literal;
  } else if (std::isnan(op->value)) {
    MarkConst("NAN");
    os << "NAN";
  } else {
    CodeGenC::Dispatch_(op, os);
  }
}

} // namespace tvm::codegen
