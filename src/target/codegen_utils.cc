/*!
 * \file target/codegen_utils.cc
 * \brief Shared utility functions for code generation
 */

#include "codegen_utils.h"

namespace tvm {
namespace codegen {

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

} // namespace codegen
} // namespace tvm
