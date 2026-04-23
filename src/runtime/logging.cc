#include <tvm/runtime/logging.h>

#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

namespace tvm {
namespace runtime {
namespace detail {

namespace {
const char *level_strings[] = {
    ": Debug: ",   // TVM_LOG_LEVEL_DEBUG = 0
    ": ",          // TVM_LOG_LEVEL_INFO  = 1
    ": Warning: ", // TVM_LOG_LEVEL_WARNING = 2
    ": Error: ",   // TVM_LOG_LEVEL_ERROR = 3
    ": Fatal: ",   // TVM_LOG_LEVEL_FATAL = 4
};
} // namespace

void LogMessageImpl(const std::string &file, int lineno, int level,
                    const std::string &message) {
  std::time_t t = std::time(nullptr);
  std::cerr << "[" << std::put_time(std::localtime(&t), "%H:%M:%S") << "] ";
#ifdef TILELANG_RELEASE_BUILD
  // Release (wheel) builds: omit file path for a cleaner user experience.
  std::cerr << level_strings[level] << message << std::endl;
#else
  // Dev builds: include file path for debugging.
  std::cerr << file << ":" << lineno << level_strings[level] << message
            << std::endl;
#endif
}

[[noreturn]] void LogFatalImpl(const std::string &file, int lineno,
                               const std::string &message) {
  LogMessageImpl(file, lineno, TVM_LOG_LEVEL_FATAL, message);
  throw InternalError(file, lineno, message);
}

} // namespace detail
} // namespace runtime
} // namespace tvm
