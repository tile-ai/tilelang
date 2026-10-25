/*!
 * \file codegen_py.cc
 */
#include "codegen_py.h"
#include "backend/common/codegen/codegen_utils.h"
#include "support/check.h"
#include <tvm/runtime/logging.h>

#include <tvm/ir/unique_name_supply.h>
#include <tvm/sym/analyzer.h>

#include <cctype>

namespace tvm {
namespace codegen {

using namespace ffi;

PrimType PythonDType(const Expr &expression) {
  if (auto scalar = expression->ty.as<PrimType>())
    return scalar.value();
  if (const auto *tuple = expression->ty.as<TupleTypeNode>();
      tuple && tuple->fields.empty())
    return PrimType::Void();
  TVM_FFI_THROW(TypeError) << "Expected a primitive expression, got "
                           << expression;
}

bool PythonIsPointer(const Expr &expression) {
  return expression->ty.as<PtrTypeNode>() ||
         expression->ty.as<tirx::TensorTypeNode>();
}

std::string PythonStorageScope(const Var &variable) {
  if (auto tensor = variable.as<TensorVar>())
    return tensor.value().scope().name();
  if (const auto *pointer = variable->ty.as<PtrTypeNode>())
    return pointer->scope.name();
  return "";
}

bool PythonIsOp(const Expr &expression, const std::string &name) {
  const auto *operation = expression.as<OpNode>();
  if (!operation)
    return false;
  std::string qualified = operation->name;
  return qualified == name ||
         qualified.substr(qualified.find_last_of('.') + 1) == name;
}

PrimExpr PythonTensorOffset(const TensorVar &tensor,
                            const Array<PrimExpr> &indices) {
  TVM_FFI_CHECK(indices.size() == 1 && tensor.shape().size() == 1, ValueError)
      << "Python code generation requires FlattenTensor before emission: "
      << tensor;
  return indices[0];
}

PrimExpr PythonTensorSize(const TensorVar &tensor) {
  PrimExpr size(1);
  for (const auto &extent : tensor.shape())
    size = size * extent;
  return sym::Analyzer()->Simplify(size);
}

void CodeGenTileLangPY::PrintType(const Type &type, std::ostream &os) {
  PrintType(type.as_or_throw<PrimType>(), os);
}

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
  PrintStmt_(f->body.value());
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
    if (auto global_symbol = func->GetAttr<String>("global_symbol")) {
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

void CodeGenTileLangPY::PrintFunctionSignature_(const String &function_name,
                                                const tirx::Function &func,
                                                std::ostream &os) { // NOLINT(*)
  os << "def " << function_name << "(";
  for (size_t i = 0; i < func->params.size(); ++i) {
    Var v = func->params[i];
    if (i > 0) {
      os << ", ";
    }
    os << AllocVarID(v.get());
  }
  os << ")";

  // Register handle data type
  for (const auto &param : func->params) {
    if (auto tensor = param.as<TensorVar>()) {
      RegisterHandleType_(param.get(), tensor.value().dtype());
      alloc_storage_scope_[param.get()] = tensor.value().scope().name();
    }
    if (auto *ptr = param->ty.as<PtrTypeNode>()) {
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        RegisterHandleType_(param.get(), GetRef<PrimType>(prim));
      }
    }
  }
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

  name_supply_->ReserveName("cutlass");
  name_supply_->ReserveName("cute");
  name_supply_->ReserveName("tl");
}

void CodeGenTileLangPY::PrintSSAAssign(const std::string &target,
                                       const std::string &src, const Type &t) {
  stream << target << " = " << RemoveOutermostParentheses(src) << "\n";
}

void CodeGenTileLangPY::PrintType(PrimType type,
                                  std::ostream &os) { // NOLINT(*)
  if (type == PrimType::Bool()) {
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

void CodeGenTileLangPY::Dispatch_(const VarNode *op,
                                  std::ostream &os) { // NOLINT(*)
  os << GetVarID(op);
}

void CodeGenTileLangPY::Dispatch_(const IntImmNode *op,
                                  std::ostream &os) { // NOLINT(*)
  if (PythonDType(GetRef<Expr>(op)) == PrimType::Bool()) {
    os << (op->value.as<int64_t>().value() ? "True" : "False");
  } else {
    std::ostringstream temp;
    temp << op->value;
    MarkConst(temp.str());
    os << temp.str();
  }
}

void CodeGenTileLangPY::Dispatch_(const FloatImmNode *op,
                                  std::ostream &os) { // NOLINT(*)
  switch (PythonDType(GetRef<Expr>(op)).bits()) {
  case 64:
  case 32: {
    std::ostringstream temp;
    temp << "float.fromhex('" << FlexibleHexFormat(op->value) << "')";
    MarkConst(temp.str());
    os << temp.str();
    break;
  }
  case 16: {
    PrintType(PythonDType(GetRef<Expr>(op)), os);
    os << "(float.fromhex('" << FlexibleHexFormat(op->value) << "'))";
    break;
  }
  default:
    LOG(FATAL) << "Bad bit-width for float: " << PythonDType(GetRef<Expr>(op))
               << "\n";
  }
}

void CodeGenTileLangPY::Dispatch_(const StringImmNode *op,
                                  std::ostream &os) { // NOLINT(*)
  EscapeStringLiteral_(op->value, os);
}

void CodeGenTileLangPY::Dispatch_(const CastNode *op,
                                  std::ostream &os) { // NOLINT(*)
  std::stringstream value;
  PrintExpr_(op->value, value);
  os << CastFromTo_(value.str(), PythonDType(op->value),
                    PythonDType(GetRef<Expr>(op)));
}

void CodeGenTileLangPY::Dispatch_(const AddNode *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("+", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const SubNode *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("-", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const MulNode *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("*", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const DivNode *op,
                                  std::ostream &os) { // NOLINT(*)
  if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLInt) ||
      PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt)) {
    PrintBinaryExpr_("//", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
  } else {
    PrintBinaryExpr_("/", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
  }
}
void CodeGenTileLangPY::Dispatch_(const ModNode *op,
                                  std::ostream &os) { // NOLINT(*)
  ICHECK(PythonDType(GetRef<Expr>(op)).MatchesCode(kDLInt) ||
         PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt) ||
         PythonDType(GetRef<Expr>(op)).MatchesCode(kDLFloat))
      << "Expected floating point or integer dtype in Mod, but got "
      << PythonDType(GetRef<Expr>(op));
  PrintBinaryExpr_("%", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}

void CodeGenTileLangPY::Dispatch_(const MinNode *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("min", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const MaxNode *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("max", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const EQNode *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("==", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const NENode *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("!=", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const LTNode *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("<", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const LENode *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("<=", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const GTNode *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_(">", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const GENode *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_(">=", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const AndNode *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("and", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const OrNode *op,
                                  std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("or", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}
void CodeGenTileLangPY::Dispatch_(const NotNode *op,
                                  std::ostream &os) { // NOLINT(*)
  os << "(not ";
  PrintExpr_(op->a, os);
  os << ")";
}

void CodeGenTileLangPY::Dispatch_(const SelectNode *op,
                                  std::ostream &os) { // NOLINT(*)
  os << "(";
  PrintExpr_(op->true_value, os);
  os << " if ";
  PrintExpr_(op->condition, os);
  os << " else ";
  PrintExpr_(op->false_value, os);
  os << ")";
}

void CodeGenTileLangPY::Dispatch_(const LShiftNode *node,
                                  std::ostream &output) {
  PrintBinaryExpr_("<<", PythonDType(GetRef<Expr>(node)), node->a, node->b,
                   output);
}

void CodeGenTileLangPY::Dispatch_(const RShiftNode *node,
                                  std::ostream &output) {
  PrintBinaryExpr_(">>", PythonDType(GetRef<Expr>(node)), node->a, node->b,
                   output);
}

void CodeGenTileLangPY::Dispatch_(const BitwiseAndNode *node,
                                  std::ostream &output) {
  PrintBinaryExpr_("&", PythonDType(GetRef<Expr>(node)), node->a, node->b,
                   output);
}

void CodeGenTileLangPY::Dispatch_(const BitwiseOrNode *node,
                                  std::ostream &output) {
  PrintBinaryExpr_("|", PythonDType(GetRef<Expr>(node)), node->a, node->b,
                   output);
}

void CodeGenTileLangPY::Dispatch_(const BitwiseXorNode *node,
                                  std::ostream &output) {
  PrintBinaryExpr_("^", PythonDType(GetRef<Expr>(node)), node->a, node->b,
                   output);
}

void CodeGenTileLangPY::Dispatch_(const BitwiseNotNode *node,
                                  std::ostream &output) {
  output << "(~";
  PrintExpr_(node->a, output);
  output << ")";
}

void CodeGenTileLangPY::Dispatch_(const RampNode *op,
                                  std::ostream &os) { // NOLINT(*)
  int lanes = PythonDType(GetRef<Expr>(op)).lanes();
  os << "(";
  for (int i = 0; i < lanes; i++) {
    os << "(" << PrintExpr_(op->base) << ")"
       << "+(" << PrintExpr_(op->stride) << "*" << i << ")";
    if (i != lanes - 1)
      os << ", ";
  }
  os << ")";
}

void CodeGenTileLangPY::Dispatch_(const CallNode *op, std::ostream &os) {
  if (const auto *call_op = op->op.as<OpNode>()) {
    std::string name = call_op->name;
    if (PythonIsOp(op->op, "call_extern") ||
        PythonIsOp(op->op, "call_pure_extern")) {
      auto symbol = op->args[0].as_or_throw<StringImm>();
      PrintCallExtern_(op->ty, symbol->value, op->args, true, os);
      return;
    }
    if (op_attr_global_symbol_.count(GetRef<Op>(call_op))) {
      PrintCallExtern_(op->ty, op_attr_global_symbol_[GetRef<Op>(call_op)],
                       op->args, false, os);
      return;
    }
    if (PythonIsOp(op->op, "tensor_data_ptr")) {
      PrintExpr_(op->args[0], os);
      return;
    }
    if (PythonIsOp(op->op, "if_then_else")) {
      os << "(" << PrintExpr_(op->args[1]) << " if " << PrintExpr_(op->args[0])
         << " else " << PrintExpr_(op->args[2]) << ")";
      return;
    }
    static const std::unordered_map<std::string, std::string> binary = {
        {"prim.bitwise_and", "&"},
        {"prim.bitwise_or", "|"},
        {"prim.bitwise_xor", "^"},
        {"prim.shift_left", "<<"},
        {"prim.shift_right", ">>"}};
    auto symbol = binary.find(name);
    if (symbol != binary.end()) {
      PrintBinaryIntrinsic_(op, symbol->second.c_str(), os);
      return;
    }
    if (PythonIsOp(op->op, "bitwise_not")) {
      os << "(~" << PrintExpr_(op->args[0]) << ")";
      return;
    }
    if (PythonIsOp(op->op, "isnullptr")) {
      os << "(" << PrintExpr_(op->args[0]) << " is None)";
      return;
    }
    if (PythonIsOp(op->op, "isnan")) {
      os << "(" << PrintExpr_(op->args[0]) << " != " << PrintExpr_(op->args[0])
         << ")";
      return;
    }
    if (PythonIsOp(op->op, "assume"))
      return;
    TVM_FFI_THROW(ValueError)
        << "Unsupported Python codegen operation: " << name;
  }
  if (auto global = op->op.as<GlobalVar>()) {
    PrintCallExtern_(op->ty, GetFunctionName_(global.value()), op->args, false,
                     os);
    return;
  }
  TVM_FFI_THROW(ValueError) << "Unsupported Python callee: " << op->op;
}

void CodeGenTileLangPY::Dispatch_(const TensorLoadNode *op, std::ostream &os) {
  auto tensor = op->source.as_or_throw<TensorVar>();
  os << GetBufferRef_(PythonDType(GetRef<Expr>(op)), tensor,
                      PythonTensorOffset(tensor, op->indices));
}

void CodeGenTileLangPY::Dispatch_(const TensorStoreNode *op) {
  auto tensor = op->dest.as_or_throw<TensorVar>();
  auto value = PrintExpr_(op->value);
  auto reference = GetBufferRef_(PythonDType(op->value), tensor,
                                 PythonTensorOffset(tensor, op->indices));
  PrintIndent();
  stream << reference << " = " << RemoveOutermostParentheses(value) << "\n";
}

void CodeGenTileLangPY::Dispatch_(const BindNode *op) {
  if (const auto *call = op->value.as<CallNode>()) {
    if (call->op.same_as(tirx::alloc_tensor_op()))
      return PrintAllocTensor_(op, call);
    if (call->op.same_as(tirx::decl_tensor_op())) {
      auto tensor = op->var.as_or_throw<TensorVar>();
      auto value = PrintExpr_(call->args[0]);
      RegisterHandleType_(tensor.get(), tensor.dtype());
      alloc_storage_scope_[tensor.get()] = tensor.scope().name();
      PrintIndent();
      stream << AllocVarID(tensor.get()) << " = " << value << "\n";
      return;
    }
  }
  std::string value = PrintExpr_(op->value);
  PrintIndent();
  stream << AllocVarID(op->var.get()) << " = " << value << "\n";
}

void CodeGenTileLangPY::Dispatch_(const AssertStmtNode *op) {
  auto condition = PrintExpr_(op->condition);
  PrintIndent();
  stream << "assert " << condition;
  if (!op->message_parts.empty()) {
    std::string message;
    for (const auto &part : op->message_parts)
      message += part->value;
    stream << ", ";
    EscapeStringLiteral_(message, stream);
  }
  stream << "\n";
}

void CodeGenTileLangPY::PrintAllocTensor_(const BindNode *op,
                                          const CallNode *allocation) {
  auto tensor = op->var.as_or_throw<TensorVar>();
  auto size = PrintExpr_(PythonTensorSize(tensor));
  auto name = AllocVarID(tensor.get());
  alloc_storage_scope_[tensor.get()] = tensor.scope().name();
  RegisterHandleType_(tensor.get(), tensor.dtype());
  PrintIndent();
  stream << name << " = [None] * " << size << "\n";
}

void CodeGenTileLangPY::Dispatch_(const RegionStmtNode *op) {
  TVM_FFI_CHECK(op->op.same_as(tirx::scope_op()), ValueError)
      << "Python codegen does not support region " << op->op;
  PrintStmt_(op->body);
}

void CodeGenTileLangPY::Dispatch_(const ForNode *op) {
  auto begin = PrintExpr_(op->begin);
  auto end = PrintExpr_(op->end);
  auto step = PrintExpr_(op->step);
  PrintIndent();
  stream << "for " << AllocVarID(op->var.get()) << " in range(" << begin << ", "
         << end << ", " << step << "):\n";
  int scope = BeginScope();
  PrintStmt_(op->body);
  EndScope(scope);
}

void CodeGenTileLangPY::Dispatch_(const WhileNode *op) {
  std::string cond = PrintExpr_(op->condition);
  PrintIndent();
  stream << "while " << RemoveOutermostParentheses(cond) << ":\n";
  int while_scope = BeginScope();
  PrintStmt_(op->body);
  EndScope(while_scope);
}

void CodeGenTileLangPY::Dispatch_(const IfNode *op) {
  std::string cond = PrintExpr_(op->condition);
  PrintIndent();
  stream << "if " << RemoveOutermostParentheses(cond) << ":\n";
  int then_scope = BeginScope();
  PrintStmt_(op->then_case);
  EndScope(then_scope);

  if (op->else_case) {
    PrintIndent();
    stream << "else:\n";
    int else_scope = BeginScope();
    PrintStmt_(op->else_case.value());
    EndScope(else_scope);
  }
}

void CodeGenTileLangPY::Dispatch_(const SeqStmtNode *op) {
  if (op->seq.empty()) {
    PrintIndent();
    stream << "pass\n";
  }
  for (Stmt stmt : op->seq) {
    PrintStmt_(stmt);
  }
}

void CodeGenTileLangPY::Dispatch_(const EvaluateNode *op) {
  if (tl::GetConstInt(op->value))
    return;

  std::string vid = PrintExpr_(op->value);
  if (!vid.empty()) {
    PrintIndent();
    stream << vid << "\n";
  }
}

void CodeGenTileLangPY::Dispatch_(const ReturnNode *op) {
  auto value = PrintExpr_(op->value);
  PrintIndent();
  stream << "return " << value << "\n";
}

void CodeGenTileLangPY::Dispatch_(const BreakNode *op) {
  PrintIndent();
  stream << "break\n";
}

void CodeGenTileLangPY::Dispatch_(const ContinueNode *op) {
  PrintIndent();
  stream << "continue\n";
}

void CodeGenTileLangPY::Dispatch_(const TupleNode *op, std::ostream &os) {
  os << "(";
  for (const auto &field : op->fields)
    os << PrintExpr_(field) << ", ";
  os << ")";
}

void CodeGenTileLangPY::Dispatch_(const LetNode *op, std::ostream &os) {
  auto value = PrintExpr_(op->value);
  PrintIndent();
  stream << AllocVarID(op->var.get()) << " = " << value << "\n";
  PrintExpr_(op->body, os);
  var_idmap_.erase(op->var.get());
}

void CodeGenTileLangPY::Dispatch_(const TupleGetItemNode *op,
                                  std::ostream &os) {
  os << PrintExpr_(op->tuple) << "[" << op->index << "]";
}

void CodeGenTileLangPY::Dispatch_(const FloorDivNode *op, std::ostream &os) {
  PrintBinaryExpr_("//", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}

void CodeGenTileLangPY::Dispatch_(const FloorModNode *op, std::ostream &os) {
  PrintBinaryExpr_("%", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
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
                                         PrimType dtype, Expr lhs, Expr rhs,
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

void CodeGenTileLangPY::PrintBinaryIntrinsic_(const CallNode *op,
                                              const char *opstr,
                                              std::ostream &os) { // NOLINT(*)
  ICHECK_EQ(PythonDType(GetRef<Expr>(op)).lanes(), 1);
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
std::string CodeGenTileLangPY::GetBufferRef_(PrimType t,
                                             const TensorVar &buffer,
                                             PrimExpr index) {
  const VarNode *buffer_var = buffer.get();
  std::string vid = GetVarID(buffer_var);
  PrimType buffer_element_dtype = buffer.dtype();

  ICHECK(HandleTypeMatch_(buffer_var, buffer_element_dtype));
  ICHECK_EQ(t, buffer_element_dtype);

  std::string index_str = PrintExpr_(index);
  return vid + "[" + index_str + "]";
}

void CodeGenTileLangPY::RegisterHandleType_(const VarNode *buf_var,
                                            PrimType t) {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) {
    handle_data_type_.insert_or_assign(buf_var, t);
  } else {
    ICHECK(it->second == t) << "conflicting buf var type";
  }
}

bool CodeGenTileLangPY::HandleTypeMatch_(const VarNode *buf_var,
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

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "target.build.tilelang_python_without_compile",
      [](IRModule module, Target target) {
        CodeGenTileLangPY generator;
        for (const auto &[global, function] : module->functions)
          generator.AddFunction(global, function.as_or_throw<tirx::Function>());
        return SourceModuleCreate(generator.Finish(), "py");
      });
}

} // namespace codegen
} // namespace tvm
