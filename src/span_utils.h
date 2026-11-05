/*!
 * \file tl/span_utils.h
 * \brief Source-location formatting and propagation for native IR.
 */
#ifndef TVM_TL_SPAN_UTILS_H_
#define TVM_TL_SPAN_UTILS_H_

#include <initializer_list>
#include <sstream>
#include <string>
#include <tvm/ir/function.h>
#include <tvm/ir/location.h>
#include <tvm/ir/stmt.h>
#include <tvm/tirx/expr.h>

namespace tvm::tl {

inline std::string FormatSpan(const Location &loc) {
  if (const auto *source = loc.as<SourceLocObj>()) {
    std::ostringstream os;
    os << source->source_name->name << ":" << source->start_line << ":"
       << source->start_column;
    return os.str();
  }
  if (const auto *site = loc.as<CallSiteLocObj>()) {
    auto callee = FormatSpan(site->callee);
    return callee.empty() ? FormatSpan(site->caller) : callee;
  }
  return "";
}

// Location metadata is excluded from structural equality/hash. Stamp only
// UnknownLoc statement locations and preserve all finer source provenance.
void StampSubtreeSpans(const Stmt &stmt, const Location &loc);

inline std::string SpanHintSuffix(const Location &loc) {
  auto text = FormatSpan(loc);
  return text.empty() ? "" : "\n  --> " + text;
}

inline std::string SpanHintSuffix(std::initializer_list<Location> locations) {
  for (const auto &loc : locations) {
    auto hint = SpanHintSuffix(loc);
    if (!hint.empty())
      return hint;
  }
  return "";
}

} // namespace tvm::tl
#endif // TVM_TL_SPAN_UTILS_H_
