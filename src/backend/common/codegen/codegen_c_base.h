/*!
 * \file backend/common/codegen/codegen_c_base.h
 * \brief Shared source emission for TileLang's native C-like code generators.
 */

#ifndef TVM_TL_BACKEND_COMMON_CODEGEN_CODEGEN_C_BASE_H_
#define TVM_TL_BACKEND_COMMON_CODEGEN_CODEGEN_C_BASE_H_

#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/location.h>
#include <tvm/ir/transform.h>

#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "backend/common/codegen/codegen_ir_utils.h"
#include "codegen/c/codegen_c.h"
#include "ir/lexical_scope.h"

namespace tvm::codegen {

inline bool TileLangEmitLineDirectivesEnabled() {
  return tvm::transform::PassContext::Current()
      ->GetConfig<bool>("tl.emit_line_directives")
      .value_or(false);
}

/*!
 * \brief Shared source ordering and source-location handling.
 *
 * Backend headers precede imported source, then declarations and definitions.
 * Native StmtFunctor::Dispatch is virtual and every recursive PrintStmt uses
 * it. Emitting a directive at each located statement preserves repeated
 * same-line mappings after generated lines; only immediately duplicate
 * directives vanish.
 */
class CodeGenTileLangCBase : public CodeGenC {
public:
  using CodeGenC::Dispatch;
  using CodeGenC::Dispatch_;

  void SetEmitLineDirectives(bool enable) { emit_line_directives_ = enable; }

  void DeclareFunction(const GlobalVar &global,
                       const Function &function) override {
    CodeGenC::DeclareFunction(global, function);
    declared_function_names_.insert(std::string(GetFunctionName(global)));
  }

  void Dispatch(const Stmt &stmt) final {
    if (emit_line_directives_)
      EmitLocationDirective(stmt->loc);
    if (const auto *binding = stmt.as<BindObj>();
        binding && HasVolatilePointerSource(binding->value)) {
      // Native decl_tensor propagates a qualifier from direct Var operands.
      // TileLang also carries pointers through byte offsets and integer
      // reinterpret bindings; keep the qualifier along those exact aliases.
      MarkVolatile(binding->var.get());
    }
    if (const auto *evaluate = stmt.as<EvaluateObj>()) {
      if (const auto *call = evaluate->value.as<CallObj>()) {
        if (call->ty.as<PtrTypeObj>() &&
            (IsCodegenOp(call, "tirx.call_extern") ||
             IsCodegenOp(call, "tirx.call_pure_extern"))) {
          // Legacy TileLang programs use a handle return annotation for
          // statement-only C helpers, including helpers that return void.
          // Discard this call's result without converting it to a pointer.
          // Argument expressions still use their normal typed emission, so
          // a nested call whose pointer result is consumed retains its cast.
          TVM_FFI_ICHECK_GE(call->args.size(), 1U);
          auto symbol = call->args[0].as_or_throw<StringImm>()->value;
          std::ostringstream expression;
          PrintCallExtern(VoidType(), symbol, call->args, true, expression);
          if (!declared_function_names_.count(std::string(symbol))) {
            ffi::Array<Type> argument_types;
            for (size_t i = 1; i < call->args.size(); ++i)
              argument_types.push_back(call->args[i]->ty);
            // Keep declaration contracts intact; discarded use does not
            // infer or change the external function's actual return type.
            GenerateForwardFunctionDeclarations(symbol, argument_types,
                                                call->ty);
          }
          PrintIndent();
          stream << expression.str() << ";\n";
          return;
        }
        if (IsCodegenOp(call, "tl.storage_sync")) {
          // Named/partial barriers are a TileLang operation. The target's
          // gpu_storage_sync only represents the full-group variant.
          PrintStorageSync(call);
          return;
        }
        if (const auto *callee = call->op.as<OpObj>();
            callee && callee->name == "prim.assume") {
          TVM_FFI_CHECK_EQ(call->args.size(), 1, TypeError);
          auto condition_type = call->args[0]->ty.as<PrimType>();
          TVM_FFI_CHECK(condition_type && *condition_type == PrimType::Bool(),
                        TypeError)
              << "prim.assume requires a scalar boolean condition";
          // The checked contract is represented by AssertStmt. This separate
          // optimizer hint has no runtime work in C-like source backends.
          return;
        }
      }
    }
    if (const auto *region = stmt.as<RegionStmtObj>();
        tl::IsLexicalScope(region)) {
      TVM_FFI_CHECK(region->args.empty() && region->body_params.empty() &&
                        region->result_vars.empty(),
                    ValueError)
          << "A TileLang lexical scope cannot bind or return values";
      // Native C codegen deliberately accepts only its own region dialect.
      // Keep TileLang's once-executed lifetime boundary in this consumer.
      PrintLexicalScope(region->body);
      return;
    }
    if (const auto *region = stmt.as<RegionStmtObj>();
        region && region->op->name == "tl.attr") {
      auto attr = DecodeTileLangAttribute(region);
      if (attr.key == "pragma_import_c") {
        TVM_FFI_ICHECK(attr.value.as<StringImmObj>() ||
                       attr.value.try_cast<ffi::String>().has_value())
            << "pragma_import_c expects a string-valued source";
        // The second newline survives a trailing backslash-newline splice.
        import_c_stream_ << CodegenAnnotationString(attr.value) << "\n\n";
        PrintStmt(region->body);
        return;
      }
      if (attr.key == "lexical_alloc_scope") {
        // A frontend cast or local temporary owns this lifetime boundary.
        // Keep the block even when it is the sole body of a loop: storage
        // planning may attach more statements to either side of the boundary.
        PrintLexicalScope(region->body);
        return;
      }
    }
    CodeGenC::Dispatch(stmt);
  }

  void InitFuncState(const Function &func) override {
    CodeGenC::InitFuncState(func);
    // Reserve extern symbols before assigning parameter/local names. The IR
    // may legally call exp2 while also holding a tensor named "exp2".
    ffi::StructuralVisit(
        func->body,
        [this](const CallObj *call, ffi::StructuralVisitorObj *visitor)
            -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
          if (auto op = call->op.as<Op>()) {
            ffi::String symbol;
            if ((op.value()->name == "tirx.call_extern" ||
                 op.value()->name == "tirx.call_pure_extern") &&
                !call->args.empty()) {
              if (const auto *name = call->args[0].as<StringImmObj>())
                symbol = name->value;
            } else if (op_attr_global_symbol_.count(op.value())) {
              symbol = op_attr_global_symbol_[op.value()];
            }
            if (!symbol.empty())
              name_supply_->ReserveName(symbol);
          }
          return visitor->DefaultVisitExpected(call);
        },
        [](const TypeObj *, ffi::StructuralVisitorObj *)
            -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
          return std::nullopt;
        });
    non_restrict_.clear();
    if (auto indices =
            func->GetAttr<ffi::Array<int64_t>>("tl.may_alias_param_indices")) {
      for (int64_t index : *indices) {
        TVM_FFI_CHECK(index >= 0 &&
                          static_cast<size_t>(index) < func->params.size(),
                      ValueError)
            << "may_alias_param_indices contains an invalid parameter index";
        const auto &parameter = func->params[index];
        TVM_FFI_CHECK(parameter->ty.as<PtrTypeObj>(), TypeError)
            << "may_alias_param_indices must name pointer parameters";
        non_restrict_.insert(parameter.get());
      }
    }
    if (auto vars = func->GetAttr<ffi::Array<Var>>("tl.non_restrict_params")) {
      for (const Var &var : vars.value())
        non_restrict_.insert(var.get());
    }
  }

  void PrintRestrict(const Var &var, std::ostream &os) override {
    if (!non_restrict_.count(var.get()))
      CodeGenC::PrintRestrict(var, os);
  }

  void PreFunctionBody(const Function &func) override {
    if (emit_line_directives_)
      EmitLocationDirective(func->loc);
  }

  std::string Finish() final {
    std::ostringstream code;
    code << preamble_stream_.str();
    EmitPreamble_(code);
    code << import_c_stream_.str();
    code << CodeGenC::Finish();
    return code.str();
  }

protected:
  std::string AllocHelperFunctionName(const std::string &hint) {
    // Local names reset for each Function. Keep generated helper names in
    // the module's declaration set as well, so separate functions cannot
    // emit helpers with the same name.
    std::string name;
    do {
      name = name_supply_->FreshName(hint);
    } while (declared_function_names_.count(name));
    declared_function_names_.insert(name);
    return name;
  }

  void PrintLexicalScope(const SeqStmt &body) {
    PrintIndent();
    stream << "{\n";
    int scope = BeginScope();
    PrintStmt(body);
    EndScope(scope);
    PrintIndent();
    stream << "}\n";
  }

  void PrintStorageSync(const CallObj *call) override {
    TVM_FFI_CHECK(!IsCodegenOp(call, "tl.storage_sync"), ValueError)
        << "This source backend does not support named/partial barriers";
    CodeGenC::PrintStorageSync(call);
  }

  virtual void EmitPreamble_(std::ostream &) const {}
  std::ostringstream preamble_stream_;

private:
  std::unordered_set<std::string> declared_function_names_;

  bool HasVolatilePointerSource(const Expr &expression) const {
    if (const auto *variable = expression.as<VarObj>())
      return IsVolatile(variable);
    const auto *call = expression.as<CallObj>();
    if (!call)
      return false;
    if (IsCodegenOp(call, "tirx.decl_tensor")) {
      // Shared arena merging preserves allocation qualifiers on each view,
      // rather than qualifying the arena and its unrelated ordinary views.
      if (auto flag =
              CodegenCallAnnotations(call).Get(tirx::attr::kVolatile.name);
          flag && CodegenBool(*flag))
        return true;
    }
    if (call->args.empty())
      return false;
    if (IsCodegenOp(call, "tirx.address_of")) {
      const auto *load = call->args[0].as<TensorLoadObj>();
      return load && HasVolatilePointerSource(load->source);
    }
    if (IsCodegenOp(call, "tirx.decl_tensor") ||
        IsCodegenOp(call, "tirx.tensor_data_ptr") ||
        IsCodegenOp(call, "tirx.ptr_byte_offset") ||
        IsCodegenOp(call, "tirx.reinterpret"))
      return HasVolatilePointerSource(call->args[0]);
    return false;
  }

  static const SourceLocObj *SourceLocation(const Location &loc) {
    // Preserve callee-first lookup without revisiting shared provenance or
    // consuming the native stack for deep macro/inlining chains.
    std::vector<Location> pending{loc};
    std::unordered_set<const LocationObj *> visited;
    while (!pending.empty()) {
      Location current = pending.back();
      pending.pop_back();
      if (!visited.insert(current.get()).second)
        continue;
      if (const auto *source = current.as<SourceLocObj>())
        return source;
      if (const auto *call = current.as<CallSiteLocObj>()) {
        pending.push_back(call->caller);
        pending.push_back(call->callee);
      }
    }
    return nullptr;
  }

  void EmitLocationDirective(const Location &loc) {
    const auto *source = SourceLocation(loc);
    if (source == nullptr || source->start_line <= 0)
      return;
    std::string file = source->source_name->name;
    if (source->start_line == last_line_ && file == last_file_ &&
        stream.tellp() == last_directive_pos_)
      return;
    stream << "#line " << source->start_line << " \"";
    for (char c : file) {
      if (c == '\\' || c == '"')
        stream << '\\';
      stream << c;
    }
    stream << "\"\n";
    last_line_ = source->start_line;
    last_file_ = file;
    last_directive_pos_ = stream.tellp();
  }

  bool emit_line_directives_{false};
  int64_t last_line_{0};
  std::string last_file_;
  std::streampos last_directive_pos_{};
  std::ostringstream import_c_stream_;
  std::unordered_set<const VarObj *> non_restrict_;
};

} // namespace tvm::codegen

#endif // TVM_TL_BACKEND_COMMON_CODEGEN_CODEGEN_C_BASE_H_
