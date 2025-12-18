/*!
 * \file target/codegen_utils.h
 * \brief Shared utility functions for code generation
 */

#ifndef TVM_TARGET_CODEGEN_UTILS_H_
#define TVM_TARGET_CODEGEN_UTILS_H_

#include <string>

namespace tvm {
namespace codegen {

/*!
 * \brief Check if the outermost parentheses match
 * \param s The input string
 * \return true if the first character is '(' and the last character is ')'
 *         and they form a matching pair
 */
bool CheckOutermostParenthesesMatch(const std::string &s);

/*!
 * \brief Remove outermost parentheses if they match
 * \param s The input string
 * \return The string with outermost parentheses removed if they match,
 *         otherwise return the original string
 */
std::string RemoveOutermostParentheses(const std::string &s);

} // namespace codegen
} // namespace tvm

#endif // TVM_TARGET_CODEGEN_UTILS_H_
