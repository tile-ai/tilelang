/*!
 * \file backend/common/codegen/codegen_c_base.h
 * \brief Shared source emission for TileLang's C-like code generators.
 */

#ifndef TVM_TL_BACKEND_COMMON_CODEGEN_CODEGEN_C_BASE_H_
#define TVM_TL_BACKEND_COMMON_CODEGEN_CODEGEN_C_BASE_H_

#include <sstream>
#include <string>

#include "target/source/codegen_c.h"

namespace tvm {
namespace codegen {

/*!
 * \brief Common source ordering, imported C source, and optional #line
 * directives for C-like code generators.
 *
 * Backend headers precede imported source, which precedes generated
 * declarations and functions. In particular, generated helpers may call
 * imported helpers. Backends write headers to preamble_stream_ or emit them in
 * EmitPreamble_ after visiting the module; decl_stream is reserved for
 * generated declarations.
 *
 * `CodeGenC::PrintStmt` is non-virtual but forwards to the virtual
 * `StmtFunctor::VisitStmt`, and every statement recursion in the vendored and
 * TileLang codegen paths goes through it; overriding `VisitStmt` here
 * therefore intercepts every printed statement without touching vendored TVM.
 *
 * A directive is emitted for every statement that carries a valid span — no
 * deduplication: after `#line N`, each emitted line advances the logical line
 * by one, so re-emitting is required to map a second statement from the same
 * Python line back to N (dedup would drift). Statements without a span simply
 * inherit the previous directive's mapping.
 *
 * Enabled via the `tl.emit_line_directives` pass config (default off), read by
 * the backend builders before codegen.
 */
class CodeGenTileLangCBase : public CodeGenC {
public:
  void SetEmitLineDirectives(bool enable) { emit_line_directives_ = enable; }

  // Intercept every statement visit. final: leaf codegen classes must not
  // accidentally shadow the dispatch.
  void VisitStmt(const Stmt &n) final {
    if (emit_line_directives_ && n.defined() && n->span.defined()) {
      EmitSpanDirective_(n->span);
    }
    // Intercept imports before backend-specific AttrStmt dispatch. A backend
    // may delegate other attributes directly to CodeGenC::VisitStmt_.
    if (const auto *attr = n.as<AttrStmtNode>();
        attr && attr->attr_key == tirx::attr::pragma_import_c) {
      const auto *source = attr->value.as<StringImmNode>();
      TVM_FFI_ICHECK(source != nullptr)
          << "pragma_import_c expects a string-valued source";
      // The second newline survives a trailing backslash-newline splice.
      import_c_stream_ << source->value << "\n\n";
      PrintStmt(attr->body);
      return;
    }
    // Qualified call: dispatch through StmtFunctor::VisitStmt to the matching
    // VisitStmt_ callback without re-entering this override.
    CodeGenC::VisitStmt(n);
  }

  // C promotes 8/16-bit operands to int; cast scalar sub-word + - * << ~
  // results back to their dtype so chained ops wrap like TIR (issue #2961).
  void VisitExpr(const PrimExpr &n, std::ostream &os) final {
    DataType t = n.dtype();
    bool wrap = t.lanes() == 1 && (t.is_int() || t.is_uint()) &&
                (t.bits() == 8 || t.bits() == 16);
    if (wrap && !n.as<AddNode>() && !n.as<SubNode>() && !n.as<MulNode>()) {
      const auto *call = n.as<CallNode>();
      wrap = call && (call->op.same_as(tirx::builtin::shift_left()) ||
                      call->op.same_as(tirx::builtin::bitwise_not()));
    }
    if (!wrap) {
      CodeGenC::VisitExpr(n, os);
      return;
    }
    os << "((";
    PrintType(t, os);
    os << ")";
    // * and << can overflow int (65535 * 65535, negative << k); do them in
    // uint32, where wrapping is defined.
    const auto *mul = n.as<MulNode>();
    const auto *call = n.as<CallNode>();
    if (mul) {
      PrintUInt32Op_(mul->a, " * ", mul->b, os);
    } else if (call && call->op.same_as(tirx::builtin::shift_left())) {
      PrintUInt32Op_(call->args[0], " << ", call->args[1], os);
    } else {
      CodeGenC::VisitExpr(n, os);
    }
    os << ")";
  }

  // Called by AddFunction between the signature and the body; anchors the
  // function entry to the PrimFunc's own span.
  void PreFunctionBody(const PrimFunc &f) override {
    if (emit_line_directives_ && f.defined() && f->span.defined()) {
      EmitSpanDirective_(f->span);
    }
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
  /*! \brief Emit backend headers whose requirements are known after codegen. */
  virtual void EmitPreamble_(std::ostream &) const {}

  /*! \brief Backend headers collected during initialization or traversal. */
  std::ostringstream preamble_stream_;

private:
  void PrintUInt32Op_(const PrimExpr &a, const char *op, const PrimExpr &b,
                      std::ostream &os) {
    os << "((";
    PrintType(DataType::UInt(32), os);
    os << ")" << PrintExpr(a) << op << "(";
    PrintType(DataType::UInt(32), os);
    os << ")" << PrintExpr(b) << ")";
  }

  void EmitSpanDirective_(const Span &span) {
    if (!span->source_name.defined() || span->line <= 0) {
      return;
    }
    const std::string file = static_cast<std::string>(span->source_name->name);
    // Suppress a re-emission that immediately follows the same directive with
    // no generated output in between: nothing advanced the logical line, so
    // the mapping is unchanged and the extra directive would only be noise.
    // Re-emission after real output still happens (no-dedup semantics).
    if (span->line == last_line_ && file == last_file_ &&
        stream.tellp() == last_directive_pos_) {
      return;
    }
    stream << "#line " << span->line << " \"";
    for (char c : file) {
      if (c == '\\' || c == '"') {
        stream << '\\';
      }
      stream << c;
    }
    stream << "\"\n";
    last_line_ = span->line;
    last_file_ = file;
    last_directive_pos_ = stream.tellp();
  }

  /*! \brief Whether to emit #line directives (off by default). */
  bool emit_line_directives_{false};
  /*! \brief (line, file) of the last emitted directive, for noise
   * suppression of immediately repeated directives. */
  int64_t last_line_{0};
  std::string last_file_;
  /*! \brief Stream position right after the last emitted directive. */
  std::streampos last_directive_pos_{};
  /*! \brief Imported source, in visitation order and separated by blank lines.
   */
  std::ostringstream import_c_stream_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_CODEGEN_CODEGEN_C_BASE_H_
