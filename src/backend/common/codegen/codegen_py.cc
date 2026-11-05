/*!
 * \file codegen_py.cc
 */
#include "codegen_py.h"
#include "backend/common/codegen/codegen_utils.h"
#include "support/check.h"
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/logging.h>

#include <tvm/ir/unique_name_supply.h>
#include <tvm/sym/analyzer.h>

#include <cctype>

namespace tvm {
namespace codegen {

using namespace ffi;

void CodeGenTileLangPY::AddFunction(const GlobalVar &gvar,
                                    const tirx::Function &f) {
  RegisterFunction_(gvar, f);
  auto function_name = GetFunctionName_(gvar);

  // clear previous generated state.
  InitFuncState_(f);

  PrintFuncDecorator_(stream);
  PrintFunctionSignature_(function_name, f, stream);
  stream << ":\n";

  int func_scope = BeginScope();
  PreFunctionBody_(f);
  TVM_FFI_CHECK((!f->GetAttr(tvm::attr::kDeclaration).value_or(false)),
                ValueError)
      << "Python emission requires a function body";
  PrintSuite_(f->body);
  EndScope(func_scope);
}

std::string CodeGenTileLangPY::Finish() {
  std::ostringstream code;
  code << decl_stream.str();
  code << stream.str();
  return code.str();
}

String CodeGenTileLangPY::GetFunctionName_(const GlobalVar &gvar) {
  auto it = internal_functions_.find(gvar);
  ICHECK(it != internal_functions_.end())
      << "Attempted to find name of " << gvar
      << ", but no function with this GlobalVar has been declared";
  return it->second;
}

void CodeGenTileLangPY::RegisterFunction_(const GlobalVar &gvar,
                                          const tirx::Function &func) {
  if (internal_functions_.count(gvar)) {
    return;
  }

  auto function_name = [&]() -> String {
    if (auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
      auto name = global_symbol.value();
      ICHECK(!func_name_supply_->ContainsName(name))
          << "Function " << gvar << " must use global symbol " << name
          << ", but this name has already been used.";
      func_name_supply_->ReserveName(name);
      return name;
    } else {
      ICHECK(!func_name_supply_->ContainsName(gvar->name_hint))
          << "Function " << gvar << " must use name hint " << gvar->name_hint
          << ", but this name has already been used.";
      func_name_supply_->ReserveName(gvar->name_hint);
      return gvar->name_hint;
    }
  }();
  internal_functions_.insert({gvar, function_name});
}

void CodeGenTileLangPY::InitFuncState_(const tirx::Function &f) {
  alloc_storage_scope_.clear();
  handle_data_type_.clear();
  CodeGenSourceBase::ClearFuncState();
  ReserveKeywordsAsUnique_();
}

void CodeGenTileLangPY::PrintFunctionSignature_(const String &name,
                                                const tirx::Function &func,
                                                std::ostream &os) {
  os << "def " << name << "(";
  for (size_t i = 0; i < func->params.size(); ++i) {
    const Var &var = func->params[i];
    if (i)
      os << ", ";
    os << AllocVarID(var.get());
    if (auto ptr = var->ty.as<PtrType>()) {
      if (auto type = ptr.value()->element_type.as<PrimType>())
        RegisterHandleType_(var.get(), type.value());
      alloc_storage_scope_[var.get()] = ptr.value()->scope.name();
    } else if (auto tensor = var.as<TensorVar>()) {
      RegisterHandleType_(var.get(), tensor.value().dtype());
      alloc_storage_scope_[var.get()] = tensor.value().scope().name();
    }
  }
  os << ")";
}

void CodeGenTileLangPY::ReserveKeywordsAsUnique_() {
  // skip the first underscore, so SSA variable starts from _1
  name_supply_->ReserveName("_");
  name_supply_->ReserveName("False");
  name_supply_->ReserveName("None");
  name_supply_->ReserveName("True");
  name_supply_->ReserveName("and");
  name_supply_->ReserveName("as");
  name_supply_->ReserveName("assert");
  name_supply_->ReserveName("async");
  name_supply_->ReserveName("await");
  name_supply_->ReserveName("break");
  name_supply_->ReserveName("class");
  name_supply_->ReserveName("continue");
  name_supply_->ReserveName("def");
  name_supply_->ReserveName("del");
  name_supply_->ReserveName("elif");
  name_supply_->ReserveName("else");
  name_supply_->ReserveName("except");
  name_supply_->ReserveName("finally");
  name_supply_->ReserveName("for");
  name_supply_->ReserveName("from");
  name_supply_->ReserveName("global");
  name_supply_->ReserveName("if");
  name_supply_->ReserveName("import");
  name_supply_->ReserveName("in");
  name_supply_->ReserveName("is");
  name_supply_->ReserveName("lambda");
  name_supply_->ReserveName("nonlocal");
  name_supply_->ReserveName("not");
  name_supply_->ReserveName("or");
  name_supply_->ReserveName("pass");
  name_supply_->ReserveName("raise");
  name_supply_->ReserveName("return");
  name_supply_->ReserveName("try");
  name_supply_->ReserveName("while");
  name_supply_->ReserveName("with");
  name_supply_->ReserveName("yield");

  name_supply_->ReserveName("void");
  name_supply_->ReserveName("int");
  name_supply_->ReserveName("float");
  name_supply_->ReserveName("double");
  name_supply_->ReserveName("char");
  name_supply_->ReserveName("unsigned");
  name_supply_->ReserveName("short");
  name_supply_->ReserveName("long");

  for (const char *name : {"abs", "range", "min", "max"})
    name_supply_->ReserveName(name);
  name_supply_->ReserveName("cutlass");
  name_supply_->ReserveName("cute");
  name_supply_->ReserveName("tl");
}

void CodeGenTileLangPY::PrintSSAAssign(const std::string &target,
                                       const std::string &src, const Type &t) {
  PrintIndent();
  stream << target << " = " << RemoveOutermostParentheses(src) << "\n";
}

void CodeGenTileLangPY::PrintType(const PrimType &type,
                                  std::ostream &os) { // NOLINT(*)
  if (type.MatchesCode(kDLBool)) {
    os << "bool";
  } else if (type.MatchesCode(kDLFloat)) {
    if (type.bits() == 16 || type.bits() == 32 || type.bits() == 64) {
      os << "float";
    } else {
      LOG(FATAL) << "Cannot convert float" << type.bits() << " to Python type";
    }
  } else if (type.MatchesCode(kDLUInt)) {
    switch (type.bits()) {
    case 8:
    case 16:
    case 32:
    case 64: {
      os << "int";
      break;
    }
    case 1:
      os << "bool";
      break;
    default:
      LOG(FATAL) << "Cannot convert uint" << type.bits() << " to Python type";
    }
  } else if (type.MatchesCode(kDLInt)) {
    switch (type.bits()) {
    case 8:
    case 16:
    case 32:
    case 64: {
      os << "int";
      break;
    }
    case 1:
      os << "bool";
      break;
    default:
      LOG(FATAL) << "Cannot convert int" << type.bits() << " to Python type";
    }
  } else {
    LOG(FATAL) << "Cannot convert type " << type << " to Python type";
  }
}

void CodeGenTileLangPY::Dispatch_(const VarObj *op,
                                  std::ostream &os) { // NOLINT(*)
  os << GetVarID(op);
}

void CodeGenTileLangPY::Dispatch_(const IntImmObj *op,
                                  std::ostream &os) { // NOLINT(*)
  if (CodegenDType(op) == PrimType::Bool()) {
    os << (op->value ? "True" : "False");
  } else {
    std::ostringstream temp;
    temp << op->value;
    MarkConst(temp.str());
    os << temp.str();
  }
}

void CodeGenTileLangPY::Dispatch_(const FloatImmObj *op,
                                  std::ostream &os) { // NOLINT(*)
  switch (CodegenDType(op).bits()) {
  case 64:
  case 32: {
    std::ostringstream temp;
    temp << "float.fromhex('" << FlexibleHexFormat(op->value) << "')";
    MarkConst(temp.str());
    os << temp.str();
    break;
  }
  case 16: {
    PrintType(CodegenDType(op), os);
    os << "(float.fromhex('" << FlexibleHexFormat(op->value) << "'))";
    break;
  }
  default:
    LOG(FATAL) << "Bad bit-width for float: " << CodegenDType(op) << "\n";
  }
}

void CodeGenTileLangPY::Dispatch_(const StringImmObj *op,
                                  std::ostream &os) { // NOLINT(*)
  EscapeStringLiteral_(op->value, os);
}

void CodeGenTileLangPY::Dispatch_(const CastObj *op,
                                  std::ostream &os) { // NOLINT(*)
  std::stringstream value;
  PrintExpr_(op->value, value);
  os << CastFromTo_(value.str(), CodegenDType(op->value), CodegenDType(op));
}

void CodeGenTileLangPY::Dispatch_(const AddObj *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("+", CodegenDType(op), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const SubObj *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("-", CodegenDType(op), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const MulObj *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("*", CodegenDType(op), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const DivObj *op, std::ostream &os) {
  if (CodegenDType(op).MatchesCode(kDLInt, kDLUInt)) {
    os << "(lambda a, b: -(abs(a) // abs(b)) if (a < 0) != (b < 0) else abs(a) "
          "// abs(b))("
       << PrintExpr_(op->a) << ", " << PrintExpr_(op->b) << ")";
  } else {
    PrintBinaryExpr_("/", CodegenDType(op), op->a, op->b, os);
  }
}
void CodeGenTileLangPY::Dispatch_(const ModObj *op, std::ostream &os) {
  TVM_FFI_CHECK(CodegenDType(op).MatchesCode(kDLInt, kDLUInt), TypeError);
  os << "(lambda a, b: a - (-(abs(a) // abs(b)) if (a < 0) != (b < 0) else "
        "abs(a) // abs(b)) * b)("
     << PrintExpr_(op->a) << ", " << PrintExpr_(op->b) << ")";
}

void CodeGenTileLangPY::Dispatch_(const MinObj *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("min", CodegenDType(op), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const MaxObj *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("max", CodegenDType(op), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const EQObj *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("==", CodegenDType(op), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const NEObj *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("!=", CodegenDType(op), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const LTObj *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("<", CodegenDType(op), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const LEObj *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("<=", CodegenDType(op), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const GTObj *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_(">", CodegenDType(op), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const GEObj *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_(">=", CodegenDType(op), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const AndObj *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("and", CodegenDType(op), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const OrObj *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("or", CodegenDType(op), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const NotObj *op,
                                  std::ostream &os) { // NOLINT(*)
  os << "(not ";
  PrintExpr_(op->a, os);
  os << ")";
}

void CodeGenTileLangPY::Dispatch_(const SelectObj *op,
                                  std::ostream &os) { // NOLINT(*)
  os << "(";
  PrintExpr_(op->true_value, os);
  os << " if ";
  PrintExpr_(op->condition, os);
  os << " else ";
  PrintExpr_(op->false_value, os);
  os << ")";
}

void CodeGenTileLangPY::Dispatch_(const RampObj *op,
                                  std::ostream &os) { // NOLINT(*)
  int lanes = CodegenDType(op).lanes();
  os << "(";
  for (int i = 0; i < lanes; i++) {
    os << "(" << PrintExpr_(op->base) << ")" << "+(" << PrintExpr_(op->stride)
       << "*" << i << ")";
    if (i != lanes - 1)
      os << ", ";
  }
  os << ")";
}

void CodeGenTileLangPY::Dispatch_(const CallObj *op, std::ostream &os) {
  if (auto callee = op->op.as<Op>()) {
    if (callee.value().same_as(builtin_call_extern_) ||
        callee.value().same_as(builtin_call_pure_extern_)) {
      TVM_FFI_ICHECK(!op->args.empty());
      PrintCallExtern_(op->ty, op->args[0].as_or_throw<StringImm>()->value,
                       op->args, true, os);
    } else if (op_attr_global_symbol_.count(callee.value())) {
      PrintCallExtern_(op->ty, op_attr_global_symbol_[callee.value()], op->args,
                       false, os);
    } else if (callee.value().same_as(prim::if_then_else_op())) {
      TVM_FFI_ICHECK_EQ(op->args.size(), 3);
      os << "(" << PrintExpr_(op->args[1]) << " if " << PrintExpr_(op->args[0])
         << " else " << PrintExpr_(op->args[2]) << ")";
    } else if (callee.value().same_as(tirx::isnullptr_op())) {
      TVM_FFI_ICHECK_EQ(op->args.size(), 1);
      os << "(" << PrintExpr_(op->args[0]) << " is None)";
    } else if (callee.value().same_as(prim::isnan_op())) {
      // Evaluate the operand once, even if it is an external call.
      os << "(lambda x: x != x)(" << PrintExpr_(op->args[0]) << ")";
    } else if (callee.value().same_as(tirx::tensor_data_ptr_op())) {
      TVM_FFI_ICHECK_EQ(op->args.size(), 1);
      PrintExpr_(op->args[0], os);
    } else {
      TVM_FFI_THROW(ValueError) << "Unresolved Python backend call " << op->op;
    }
  } else if (auto callee = op->op.as<GlobalVar>()) {
    PrintCallExtern_(op->ty, GetFunctionName_(callee.value()), op->args, false,
                     os);
  } else {
    TVM_FFI_THROW(ValueError)
        << "Python backend expects an operator or GlobalVar callee";
  }
}

void CodeGenTileLangPY::Dispatch_(const TensorLoadObj *op, std::ostream &os) {
  TVM_FFI_CHECK_EQ(op->indices.size(), 1, ValueError)
      << "Python source load requires flat memory";
  TensorVar tensor = op->source.as_or_throw<TensorVar>();
  TVM_FFI_CHECK(CodegenDType(op) == tensor.dtype(), TypeError)
      << "Scalar Python source load requires the tensor element type";
  os << GetBufferRef_(tensor.dtype(), tensor.get(), op->indices[0]);
}

void CodeGenTileLangPY::Dispatch_(const TensorStoreObj *op) {
  TVM_FFI_CHECK_EQ(op->indices.size(), 1, ValueError)
      << "Python source store requires flat memory";
  TensorVar tensor = op->dest.as_or_throw<TensorVar>();
  TVM_FFI_CHECK(op->value.ty() == tensor.dtype(), TypeError)
      << "Scalar Python source store requires the tensor element type";
  const auto value = PrintExpr_(op->value);
  const auto ref = GetBufferRef_(tensor.dtype(), tensor.get(), op->indices[0]);
  PrintIndent();
  stream << ref << " = " << RemoveOutermostParentheses(value) << "\n";
}

void CodeGenTileLangPY::EmitTensorDeclaration_(const BindObj *op,
                                               const CallObj *call) {
  TensorVar tensor = op->var.as_or_throw<TensorVar>();
  TVM_FFI_CHECK_EQ(call->args.size(), 1, ValueError);
  std::string value = PrintExpr_(call->args[0]);
  RegisterHandleType_(tensor.get(), tensor.dtype());
  alloc_storage_scope_[tensor.get()] = tensor.scope().name();
  PrintIndent();
  stream << AllocVarID(tensor.get()) << " = " << value << "\n";
}

void CodeGenTileLangPY::Dispatch_(const BindObj *op) {
  if (const auto *call = op->value.as<CallObj>()) {
    if (call->op.same_as(tirx::alloc_tensor_op()))
      return EmitTensorAllocation_(op, call);
    if (call->op.same_as(tirx::decl_tensor_op()))
      return EmitTensorDeclaration_(op, call);
  }
  std::string value = PrintExpr_(op->value);
  PrintIndent();
  stream << AllocVarID(op->var.get()) << " = " << value << "\n";
}

void CodeGenTileLangPY::EmitTensorAllocation_(const BindObj *op,
                                              const CallObj *) {
  TensorVar tensor = op->var.as_or_throw<TensorVar>();
  const size_t size = CodegenAllocationElements(tensor);
  RegisterHandleType_(tensor.get(), tensor.dtype());
  alloc_storage_scope_[tensor.get()] = tensor.scope().name();
  PrintIndent();
  stream << AllocVarID(tensor.get()) << " = [None] * " << size << "\n";
}

void CodeGenTileLangPY::Dispatch_(const RegionStmtObj *op) {
  bool lexical = op->op->name == "tl.lexical_scope";
  if (lexical)
    TVM_FFI_CHECK(op->args.empty(), ValueError)
        << "A TileLang lexical scope cannot take operands";
  if (op->op->name == "tl.attr")
    lexical = DecodeTileLangAttribute(op).key == "lexical_alloc_scope";
  TVM_FFI_CHECK(lexical && op->body_params.empty() && op->result_vars.empty(),
                ValueError)
      << "Python backend cannot ignore region semantics: " << op->op;
  PrintIndent();
  stream << "if True:\n";
  int scope = BeginScope();
  PrintSuite_(op->body);
  EndScope(scope);
}

void CodeGenTileLangPY::Dispatch_(const ForObj *op) {
  PrintIndent();
  stream << "for " << AllocVarID(op->var.get()) << " in range("
         << PrintExpr_(op->begin) << ", " << PrintExpr_(op->end) << ", "
         << PrintExpr_(op->step) << "):\n";
  int scope = BeginScope();
  PrintSuite_(op->body);
  EndScope(scope);
}

void CodeGenTileLangPY::Dispatch_(const WhileObj *op) {
  std::string cond = PrintExpr_(op->condition);
  PrintIndent();
  stream << "while " << RemoveOutermostParentheses(cond) << ":\n";
  int while_scope = BeginScope();
  PrintSuite_(op->body);
  EndScope(while_scope);
}

void CodeGenTileLangPY::Dispatch_(const IfObj *op) {
  std::string cond = PrintExpr_(op->condition);
  PrintIndent();
  stream << "if " << RemoveOutermostParentheses(cond) << ":\n";
  int then_scope = BeginScope();
  PrintSuite_(op->then_case);
  EndScope(then_scope);

  if (op->else_case) {
    PrintIndent();
    stream << "else:\n";
    int else_scope = BeginScope();
    PrintSuite_(op->else_case.value());
    EndScope(else_scope);
  }
}

void CodeGenTileLangPY::Dispatch_(const SeqStmtObj *op) {
  for (Stmt stmt : op->seq) {
    PrintStmt_(stmt);
  }
}

void CodeGenTileLangPY::Dispatch_(const EvaluateObj *op) {
  if (op->value.as<IntImmObj>())
    return;
  if (const auto *call = op->value.as<CallObj>();
      call && IsCodegenOp(call, "prim.assume")) {
    TVM_FFI_CHECK_EQ(call->args.size(), 1, TypeError);
    auto condition_type = call->args[0]->ty.as<PrimType>();
    TVM_FFI_CHECK(condition_type && *condition_type == PrimType::Bool(),
                  TypeError)
        << "prim.assume requires one scalar boolean condition";
    return;
  }
  std::string value = PrintExpr_(op->value);
  if (!value.empty()) {
    PrintIndent();
    stream << value << "\n";
  }
}

void CodeGenTileLangPY::Dispatch_(const AssertStmtObj *op) {
  std::string cond = PrintExpr_(op->condition);
  PrintIndent();
  if (!op->message_parts.empty()) {
    std::string joined_msg;
    for (const auto &part : op->message_parts) {
      joined_msg += part->value;
    }
    stream << "assert " << cond << ", ";
    EscapeStringLiteral_(joined_msg, stream);
    stream << "\n";
  } else {
    stream << "assert " << cond << "\n";
  }
}

std::string CodeGenTileLangPY::CastFromTo_(const std::string &value,
                                           PrimType from, PrimType target) {
  if (from == target)
    return value;
  std::ostringstream os;
  PrintType(target, os);
  os << "(" << value << ")";
  return os.str();
}

void CodeGenTileLangPY::PrintBinaryExpr_(const std::string &opstr,
                                         PrimType dtype, PrimExpr lhs,
                                         PrimExpr rhs,
                                         std::ostream &os) { // NOLINT(*)
  ICHECK_EQ(dtype.lanes(), 1);
  if (isalpha(opstr[0]) && opstr != "and" && opstr != "or") {
    os << opstr << '(';
    PrintExpr_(lhs, os);
    os << ", ";
    PrintExpr_(rhs, os);
    os << ')';
  } else {
    os << '(';
    PrintExpr_(lhs, os);
    os << ' ' << opstr << ' ';
    PrintExpr_(rhs, os);
    os << ')';
  }
}

void CodeGenTileLangPY::PrintBinaryIntrinsic_(const CallObj *op,
                                              const char *opstr,
                                              std::ostream &os) { // NOLINT(*)
  ICHECK_EQ(CodegenDType(op).lanes(), 1);
  ICHECK_EQ(op->args.size(), 2U);
  os << '(';
  PrintExpr_(op->args[0], os);
  os << ' ' << opstr << ' ';
  PrintExpr_(op->args[1], os);
  os << ')';
}

void CodeGenTileLangPY::PrintCallExtern_(Type ret_type, String global_symbol,
                                         const Array<Expr> &args,
                                         bool skip_first_arg,
                                         std::ostream &os) { // NOLINT(*)
  os << global_symbol << "(";
  for (size_t i = static_cast<size_t>(skip_first_arg); i < args.size(); ++i) {
    PrintExpr_(args[i], os);
    if (i < args.size() - 1) {
      os << ", ";
    }
  }
  os << ")";
}

// Print a reference expression to a buffer.
std::string CodeGenTileLangPY::GetBufferRef_(PrimType type, const VarObj *var,
                                             PrimExpr index) {
  TensorVar tensor = CodegenTensor(var);
  TVM_FFI_CHECK(type == tensor.dtype() && HandleTypeMatch_(var, type),
                TypeError)
      << "Conflicting Python tensor element type";
  return GetVarID(var) + "[" + PrintExpr_(index) + "]";
}

void CodeGenTileLangPY::RegisterHandleType_(const VarObj *buf_var, PrimType t) {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) {
    handle_data_type_.insert_or_assign(buf_var, t);
  } else {
    ICHECK(it->second == t) << "conflicting buf var type";
  }
}

bool CodeGenTileLangPY::HandleTypeMatch_(const VarObj *buf_var,
                                         PrimType t) const {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end())
    return false;
  return it->second == t;
}

void CodeGenTileLangPY::EscapeStringLiteral_(const std::string &s,
                                             std::ostream &os) {
  os << '"';
  for (unsigned char c : s) {
    switch (c) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    case '\f':
      os << "\\f";
      break;
    case '\b':
      os << "\\b";
      break;
    default:
      // Handle non-printable and non-ASCII characters
      if (c < 32 || c == 127) {
        // Output as \xHH
        os << "\\x";
        const char hex[] = "0123456789abcdef";
        os << hex[(c >> 4) & 0xF];
        os << hex[c & 0xF];
      } else {
        os << c;
      }
      break;
    }
  }
  os << '"';
}

void CodeGenTileLangPY::PrintSuite_(const Stmt &body) {
  auto position = stream.tellp();
  PrintStmt_(body);
  if (stream.tellp() == position) {
    PrintIndent();
    stream << "pass\n";
  }
}

#define TILELANG_PY_BINARY(Node, Symbol)                                       \
  void CodeGenTileLangPY::Dispatch_(const Node *op, std::ostream &os) {        \
    PrintBinaryExpr_(Symbol, CodegenDType(op), op->a, op->b, os);              \
  }
TILELANG_PY_BINARY(FloorDivObj, "//")
TILELANG_PY_BINARY(FloorModObj, "%")
TILELANG_PY_BINARY(LShiftObj, "<<")
TILELANG_PY_BINARY(RShiftObj, ">>")
TILELANG_PY_BINARY(BitwiseAndObj, "&")
TILELANG_PY_BINARY(BitwiseOrObj, "|")
TILELANG_PY_BINARY(BitwiseXorObj, "^")
#undef TILELANG_PY_BINARY

void CodeGenTileLangPY::Dispatch_(const BitwiseNotObj *op, std::ostream &os) {
  os << "(~" << PrintExpr_(op->a) << ")";
}
void CodeGenTileLangPY::Dispatch_(const DataTypeImmObj *op, std::ostream &os) {
  PrintType(PrimType(op->value), os);
}
void CodeGenTileLangPY::Dispatch_(const GenericConstObj *op, std::ostream &os) {
  TVM_FFI_CHECK(op->value == nullptr, TypeError)
      << "Unsupported Python generic constant";
  os << "None";
}
void CodeGenTileLangPY::Dispatch_(const TupleObj *op, std::ostream &os) {
  os << "(";
  for (const Expr &field : op->fields)
    os << PrintExpr_(field) << ", ";
  os << ")";
}
void CodeGenTileLangPY::Dispatch_(const TupleGetItemObj *op, std::ostream &os) {
  os << "(" << PrintExpr_(op->tuple) << ")[" << op->index << "]";
}
void CodeGenTileLangPY::Dispatch_(const LetObj *op, std::ostream &os) {
  const std::string value = PrintExpr_(op->value);
  const std::string var = AllocVarID(op->var.get());
  const std::string body = PrintExpr_(op->body);
  var_idmap_.erase(op->var.get());
  os << "(lambda " << var << ": " << body << ")(" << value << ")";
}
void CodeGenTileLangPY::Dispatch_(const ReturnObj *op) {
  PrintIndent();
  stream << "return " << PrintExpr_(op->value) << "\n";
}
void CodeGenTileLangPY::Dispatch_(const BreakObj *) {
  PrintIndent();
  stream << "break\n";
}
void CodeGenTileLangPY::Dispatch_(const ContinueObj *) {
  PrintIndent();
  stream << "continue\n";
}

} // namespace codegen
} // namespace tvm
