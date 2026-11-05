/*!
 * \file backend/common/codegen/codegen_utils.cc
 * \brief Shared utility functions for code generation
 */

#include "codegen_utils.h"

#include <sstream>

namespace tvm {
namespace codegen {

std::string EscapeCString(const std::string &value) {
  std::string escaped;
  for (unsigned char ch : value) {
    switch (ch) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      if (ch < 0x20 || ch >= 0x7f) {
        escaped += '\\';
        escaped += static_cast<char>('0' + ((ch >> 6) & 7));
        escaped += static_cast<char>('0' + ((ch >> 3) & 7));
        escaped += static_cast<char>('0' + (ch & 7));
      } else {
        escaped += static_cast<char>(ch);
      }
    }
  }
  return escaped;
}

bool CheckOutermostParenthesesMatch(const std::string &s) {
  if (!s.empty() && s.front() == '(' && s.back() == ')') {
    size_t len = s.size();
    int n_unmatched = 0;
    for (size_t i = 0; i < len; ++i) {
      if (s[i] == '(') {
        n_unmatched++;
      } else if (s[i] == ')') {
        n_unmatched--;
      }
      if (n_unmatched < 0) {
        return false;
      }
      if (n_unmatched == 0) {
        return i == len - 1;
      }
    }
  }
  return false;
}

std::string RemoveOutermostParentheses(const std::string &s) {
  if (CheckOutermostParenthesesMatch(s)) {
    return s.substr(1, s.size() - 2);
  } else {
    return s;
  }
}

std::string FlexibleHexFormat(double value) {
  std::ostringstream os;
  os << std::hexfloat << value;
  std::string repr = os.str();

  const size_t exponent_pos = repr.find_first_of("pP");
  if (exponent_pos == std::string::npos) {
    return repr;
  }

  const size_t dot_pos = repr.find('.');
  if (dot_pos == std::string::npos || dot_pos > exponent_pos) {
    return repr;
  }

  size_t frac_end = exponent_pos;
  while (frac_end > dot_pos + 1 && repr[frac_end - 1] == '0') {
    --frac_end;
  }

  if (frac_end == dot_pos + 1) {
    repr.erase(dot_pos, exponent_pos - dot_pos);
  } else if (frac_end < exponent_pos) {
    repr.erase(frac_end, exponent_pos - frac_end);
  }

  return repr;
}

} // namespace codegen
} // namespace tvm
