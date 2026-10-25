/*!
 * \file tl/span_utils.h
 * \brief Source-location helpers for standalone TIRx IR nodes.
 *
 * The historical span helper names operate on native Location metadata.
 * Native node locations are mutable and ignored by structural equality and
 * hashing.
 */
#ifndef TVM_TL_SPAN_UTILS_H_
#define TVM_TL_SPAN_UTILS_H_

#include <tvm/ir/location.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

#include <initializer_list>
#include <sstream>
#include <string>

namespace tvm {
namespace tl {

/*!
 * \brief Format a span as "file:line:col" for error messages.
 * \return Empty string when the span or its source name is undefined.
 */
inline std::string FormatSpan(const Location &span) {
  if (const auto *source = span.as<SourceLocNode>()) {
    std::ostringstream os;
    os << source->source_name->name << ":" << source->start_line << ":"
       << source->start_column;
    return os.str();
  }
  if (const auto *call = span.as<CallSiteLocNode>()) {
    auto callee = FormatSpan(call->callee);
    return callee.empty() ? FormatSpan(call->caller) : callee;
  }
  return "";
}

/*!
 * \brief Fill in the given span on every Stmt node in the subtree whose span
 *        is undefined. Nodes that already carry a span are left untouched, so
 *        lowered code inherits the source line of the statement it was
 *        lowered from without overriding finer-grained spans.
 * \note No-op when `span` is undefined. Spans never participate in structural
 *       equality or hashing, so stamping does not change compilation results.
 */
void StampSubtreeSpans(const Stmt &stmt, const Location &span);

/*!
 * \brief Suffix for error messages pointing at a source location, e.g.
 *        "\n  --> /path/to/kernel.py:21:1". Empty when no span is available.
 */
inline std::string SpanHintSuffix(const Location &span) {
  std::string loc = FormatSpan(span);
  return loc.empty() ? "" : "\n  --> " + loc;
}

/*!
 * \brief SpanHintSuffix over multiple candidates; the first defined span wins.
 * Useful when an error involves several buffers (e.g. src/dst) and any one of
 * their declaration sites helps the user locate the problem.
 */
inline std::string SpanHintSuffix(std::initializer_list<Location> spans) {
  for (const auto &span : spans) {
    std::string hint = SpanHintSuffix(span);
    if (!hint.empty()) {
      return hint;
    }
  }
  return "";
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_SPAN_UTILS_H_
