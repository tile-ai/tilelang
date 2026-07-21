/*!
 * \file tl/span_utils.h
 * \brief Helpers to read/write source spans on tirx IR nodes and to format
 *        spans into compiler error messages.
 *
 * `StmtNode::span` / `BufferNode::span` / `PrimFuncNode::span` are mutable
 * fields that are reflected read-only to Python. These helpers write them
 * directly from C++, so spans can be injected during script parsing without
 * touching the vendored TVM fork. Spans never participate in structural
 * equality or hashing.
 */
#ifndef TVM_TL_SPAN_UTILS_H_
#define TVM_TL_SPAN_UTILS_H_

#include <tvm/ir/source_map.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

#include <initializer_list>
#include <sstream>
#include <string>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

/*!
 * \brief Format a span as "file:line:col" for error messages.
 * \return Empty string when the span or its source name is undefined.
 */
inline std::string FormatSpan(const Span &span) {
  if (!span.defined() || !span->source_name.defined()) {
    return "";
  }
  std::ostringstream os;
  os << span->source_name->name << ":" << span->line << ":" << span->column;
  return os.str();
}

/*!
 * \brief Suffix for error messages pointing at a source location, e.g.
 *        "\n  --> /path/to/kernel.py:21:1". Empty when no span is available.
 */
inline std::string SpanHintSuffix(const Span &span) {
  std::string loc = FormatSpan(span);
  return loc.empty() ? "" : "\n  --> " + loc;
}

/*!
 * \brief SpanHintSuffix over multiple candidates; the first defined span wins.
 * Useful when an error involves several buffers (e.g. src/dst) and any one of
 * their declaration sites helps the user locate the problem.
 */
inline std::string SpanHintSuffix(std::initializer_list<Span> spans) {
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
