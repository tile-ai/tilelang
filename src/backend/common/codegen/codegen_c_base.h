/*!
 * \file backend/common/codegen/codegen_c_base.h
 * \brief Shared source emission for TileLang's C-like code generators.
 */

#ifndef TVM_TL_BACKEND_COMMON_CODEGEN_CODEGEN_C_BASE_H_
#define TVM_TL_BACKEND_COMMON_CODEGEN_CODEGEN_C_BASE_H_

#include <sstream>
#include <streambuf>
#include <string>

#include "target/source/codegen_c.h"

namespace tvm {
namespace codegen {

/*!
 * \brief Shared source assembly and optional #line directives for C-like code
 * generators.
 *
 * Headers go to preamble_stream_ (or EmitPreamble_), imported source to
 * import_c_stream_, and generated code to the vendored declaration and stream
 * buffers; Finish() assembles them in that order.
 *
 * CodeGenC::PrintStmt forwards to the virtual StmtFunctor::VisitStmt for every
 * statement, so overriding VisitStmt here intercepts all emitted statements
 * without touching vendored TVM.
 *
 * The tl.emit_line_directives level maps generated code back to TIR source
 * spans via "#line N file": 0 is off, 1 re-emits per spanned statement, and 2
 * or more re-anchors every emitted line, including the several lines one
 * statement can print and a block's closing brace. Directives are not
 * deduplicated: every emitted line advances the logical line, so a repeated
 * source line must be re-emitted to map back correctly. Anchors are
 * function-local, and lines without a source span map to
 * "<tilelang-generated>".
 */
class CodeGenTileLangCBase : public CodeGenC {
public:
  /*! \brief Lowest level that emits a directive per spanned statement. */
  static constexpr int kStatementLevel = 1;
  /*! \brief Lowest level that re-anchors every emitted source line. */
  static constexpr int kPerLineLevel = 2;

  /*!
   * \brief Set the #line emission level from the pass config.
   *
   * Level 2 or more installs the stream wrapper; lower levels detach it so a
   * reused codegen object cannot keep anchoring.
   */
  void SetEmitLineDirectives(int level) {
    line_directive_level_ = level;
    if (line_directive_level_ >= kPerLineLevel) {
      line_anchoring_buf_.Install(&stream, this);
    } else {
      line_anchoring_buf_.Restore();
    }
  }

  /*! \brief Start a function: anchors never carry over from the previous one.
   *
   * A spanned PrimFunc anchors its signature, declarations, and closing brace;
   * a span-less one leaves all unsourced lines to the generated pseudo-source.
   */
  void InitFuncState(const PrimFunc &f) override {
    CodeGenC::InitFuncState(f);
    if (line_directive_level_ < kPerLineLevel) {
      return;
    }
    enclosing_span_ = Span();
    function_span_ = f.defined() ? f->span : Span();
    last_span_ = function_span_;
    in_function_ = true;
  }

  // Intercept every statement visit. final: leaf codegen classes must not
  // accidentally shadow the dispatch.
  void VisitStmt(const Stmt &n) final {
    Span saved_enclosing = enclosing_span_;
    if (line_directive_level_ >= kPerLineLevel && n.defined()) {
      // The wrapper re-reads this anchor for every line of the dispatch.
      if (n->span.defined()) {
        enclosing_span_ = n->span;
        last_span_ = n->span;
      } else if (!n.as<SeqStmtNode>() && !enclosing_span_.defined() &&
                 function_span_.defined()) {
        // Inside a sourced function a span-less statement inherits the previous
        // statement's span; sequences and unsourced functions do not.
        enclosing_span_ = last_span_.defined() ? last_span_ : function_span_;
      }
    } else if (line_directive_level_ >= kStatementLevel && n.defined() &&
               n->span.defined()) {
      EmitSpanDirective_(n->span);
    }
    VisitStmtImpl_(n);
    enclosing_span_ = saved_enclosing;
  }

  // Level 1 anchors the function entry here; level 2 anchors every line.
  void PreFunctionBody(const PrimFunc &f) override {
    if (line_directive_level_ == kStatementLevel && f.defined() &&
        f->span.defined()) {
      EmitSpanDirective_(f->span);
    }
  }

  std::string Finish() final {
    // Detach the wrapper before the collected source is read back.
    line_anchoring_buf_.Restore();
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

  /*! \brief Anchor the next line, then pass raw user source through verbatim.
   *
   * A directive inside a macro continuation or string literal would corrupt it.
   */
  void BeginRawSource() {
    if (line_directive_level_ >= kPerLineLevel) {
      line_anchoring_buf_.BeginRawSource();
    }
  }

  void EndRawSource() {
    if (line_directive_level_ >= kPerLineLevel) {
      line_anchoring_buf_.EndRawSource();
    }
  }

  /*! \brief Backend headers collected during initialization or traversal. */
  std::ostringstream preamble_stream_;

private:
  /*! \brief Pseudo-source for lines that have no user source. */
  static constexpr const char *kGeneratedSourceDirective =
      "#line 1 \"<tilelang-generated>\"\n";

  // Dispatch after anchoring; imports are intercepted before AttrStmt dispatch.
  void VisitStmtImpl_(const Stmt &n) {
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
    // VisitStmt_ callback without re-entering the override.
    CodeGenC::VisitStmt(n);
  }

  /*! \brief Directive for the next line: the statement anchor, the function
   * span, or the generated pseudo-source; empty before the first function. */
  std::string NextLineDirective_() const {
    const Span anchor =
        enclosing_span_.defined() ? enclosing_span_ : function_span_;
    if (anchor.defined()) {
      return FormatSpanDirective_(anchor);
    }
    return in_function_ ? kGeneratedSourceDirective : std::string();
  }

  /*! \brief #line directive for \p span, or empty without a source position. */
  std::string FormatSpanDirective_(const Span &span) const {
    if (!span.defined() || !span->source_name.defined() || span->line <= 0) {
      return std::string();
    }
    std::string directive = "#line " + std::to_string(span->line) + " \"";
    for (char c : static_cast<std::string>(span->source_name->name)) {
      if (c == '\\' || c == '"') {
        directive += '\\';
      }
      directive += c;
    }
    return directive + "\"\n";
  }

  /*! \brief Emit \p span's directive unless it is an immediate repeat with no
   * output in between. */
  void EmitSpanDirective_(const Span &span) {
    const std::string directive = FormatSpanDirective_(span);
    if (directive.empty()) {
      return;
    }
    // Nothing advanced the logical line, so the mapping is unchanged.
    if (directive == last_directive_ && stream.tellp() == last_directive_pos_) {
      return;
    }
    stream << directive;
    last_directive_ = directive;
    last_directive_pos_ = stream.tellp();
  }

  /*!
   * \brief Wraps stream and prefixes every emitted line with its anchor
   * directive, buffering indentation so the directive stays at column zero.
   *
   * Raw source between BeginRawSource() and EndRawSource() passes through
   * untouched.
   */
  class LineAnchoringBuf final : public std::streambuf {
  public:
    void Install(std::ostream *stream, CodeGenTileLangCBase *owner) {
      if (stream_ != nullptr) {
        return;
      }
      stream_ = stream;
      owner_ = owner;
      target_ = stream->rdbuf();
      stream->rdbuf(this);
    }

    void Restore() {
      if (stream_ == nullptr) {
        return;
      }
      FlushIndent_();
      stream_->rdbuf(target_);
      stream_ = nullptr;
    }

    ~LineAnchoringBuf() override { Restore(); }

    /*! \brief Anchor the next line once, then pass raw source through. */
    void BeginRawSource() {
      if (suspended_) {
        return;
      }
      FlushIndent_();
      WriteRaw_(owner_->NextLineDirective_());
      at_line_start_ = true;
      suspended_ = true;
    }

    void EndRawSource() { suspended_ = false; }

  protected:
    std::streamsize xsputn(const char *data, std::streamsize size) override {
      if (suspended_) {
        WriteRaw_(data, size);
        if (size > 0) {
          at_line_start_ = data[size - 1] == '\n';
        }
        return size;
      }
      for (std::streamsize i = 0; i < size; ++i) {
        Put_(data[i]);
      }
      return size;
    }

    int_type overflow(int_type ch) override {
      if (traits_type::eq_int_type(ch, traits_type::eof())) {
        return traits_type::not_eof(ch);
      }
      const char c = traits_type::to_char_type(ch);
      if (suspended_) {
        WriteRaw_(&c, 1);
        at_line_start_ = c == '\n';
        return ch;
      }
      Put_(c);
      return ch;
    }

    int sync() override { return target_ == nullptr ? 0 : target_->pubsync(); }

  private:
    void Put_(char c) {
      if (!at_line_start_) {
        WriteRaw_(&c, 1);
        at_line_start_ = c == '\n';
        return;
      }
      if (c == ' ' || c == '\t') {
        indent_.push_back(c);
        return;
      }
      if (c == '\n') {
        // Blank line: flush trailing indentation and stay at line start.
        FlushIndent_();
        WriteRaw_("\n", 1);
        return;
      }
      WriteRaw_(owner_->NextLineDirective_());
      FlushIndent_();
      WriteRaw_(&c, 1);
      at_line_start_ = false;
    }

    void FlushIndent_() {
      if (!indent_.empty()) {
        WriteRaw_(indent_);
        indent_.clear();
      }
    }

    /*! \brief Write raw bytes to the wrapped buffer, bypassing anchoring. */
    void WriteRaw_(const char *data, std::streamsize size) {
      if (target_ != nullptr && size > 0) {
        target_->sputn(data, size);
      }
    }

    void WriteRaw_(const std::string &text) {
      WriteRaw_(text.data(), static_cast<std::streamsize>(text.size()));
    }

    std::ostream *stream_{nullptr};
    std::streambuf *target_{nullptr};
    CodeGenTileLangCBase *owner_{nullptr};
    bool at_line_start_{true};
    bool suspended_{false};
    std::string indent_;
  };

  /*! \brief Emission level from the pass config; 0 disables it. */
  int line_directive_level_{0};
  /*! \brief Anchor for the statement being emitted; undefined between them. */
  Span enclosing_span_;
  /*! \brief Span of the current function; anchors its signature and braces. */
  Span function_span_;
  /*! \brief Anchor of the last sourced statement, for span-less ones. */
  Span last_span_;
  /*! \brief Whether a function is being emitted; an undefined function span
   * then marks lines that have no user source. */
  bool in_function_{false};
  /*! \brief Per-line stream wrapper, active at level 2 and above. */
  LineAnchoringBuf line_anchoring_buf_;
  /*! \brief Text and stream position of the last emitted directive. */
  std::string last_directive_;
  std::streampos last_directive_pos_{};
  /*! \brief Imported source, in visitation order. */
  std::ostringstream import_c_stream_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_CODEGEN_CODEGEN_C_BASE_H_
