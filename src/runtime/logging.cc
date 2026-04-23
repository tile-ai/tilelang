#include <tvm/runtime/logging.h>

#include <algorithm>
#include <cstdlib>
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

constexpr const char *kSrcPrefix = "/src/";
constexpr const char *kNoSlashSrcPrefix = "src/";
constexpr const char *kDefaultKeyword = "DEFAULT";

// MSVC commonly reports __FILE__ with backslashes, while TVM_LOG_DEBUG expects
// source-root-relative keys. Normalize first so Windows and Unix use the same
// lookup format.
std::string FileToVLogMapKey(const std::string &filename) {
  std::string normalized = filename;
  std::replace(normalized.begin(), normalized.end(), '\\', '/');

  size_t last_src = normalized.rfind(kSrcPrefix);
  if (last_src != std::string::npos) {
    return normalized.substr(last_src +
                             std::char_traits<char>::length(kSrcPrefix));
  }

  if (normalized.rfind(kNoSlashSrcPrefix, 0) == 0) {
    return normalized.substr(std::char_traits<char>::length(kNoSlashSrcPrefix));
  }

  return normalized;
}
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

/* static */
TvmLogDebugSettings TvmLogDebugSettings::ParseSpec(const char *opt_spec) {
  TvmLogDebugSettings settings;
  if (opt_spec == nullptr) {
    return settings;
  }

  std::string spec(opt_spec);
  if (spec.empty() || spec == "0") {
    return settings;
  }

  settings.dlog_enabled_ = true;
  if (spec == "1") {
    return settings;
  }

  std::istringstream spec_stream(spec);
  auto tell_pos = [&](const std::string &last_read) {
    int pos = static_cast<int>(spec_stream.tellg());
    if (pos == -1) {
      pos = static_cast<int>(spec.size() - last_read.size());
    }
    return pos;
  };

  while (spec_stream) {
    std::string name;
    if (!std::getline(spec_stream, name, '=')) {
      break;
    }
    if (name.empty()) {
      LOG(FATAL) << "TVM_LOG_DEBUG ill-formed at position " << tell_pos(name)
                 << ": empty filename";
    }

    name = FileToVLogMapKey(name);

    std::string level;
    if (!std::getline(spec_stream, level, ',')) {
      LOG(FATAL) << "TVM_LOG_DEBUG ill-formed at position " << tell_pos(level)
                 << ": expecting \"=<level>\" after \"" << name << "\"";
      return settings;
    }
    if (level.empty()) {
      LOG(FATAL) << "TVM_LOG_DEBUG ill-formed at position " << tell_pos(level)
                 << ": empty level after \"" << name << "\"";
      return settings;
    }

    char *end_of_level = nullptr;
    int level_val = static_cast<int>(strtol(level.c_str(), &end_of_level, 10));
    if (end_of_level != level.c_str() + level.size()) {
      LOG(FATAL) << "TVM_LOG_DEBUG ill-formed at position " << tell_pos(level)
                 << ": invalid level: \"" << level << "\"";
      return settings;
    }

    LOG(INFO) << "TVM_LOG_DEBUG enables VLOG statements in '" << name
              << "' up to level " << level;
    settings.vlog_level_map_.emplace(name, level_val);
  }

  return settings;
}

bool TvmLogDebugSettings::VerboseEnabledImpl(const std::string &filename,
                                             int level) const {
  auto itr = vlog_level_map_.find(FileToVLogMapKey(filename));
  if (itr != vlog_level_map_.end()) {
    return level <= itr->second;
  }

  itr = vlog_level_map_.find(kDefaultKeyword);
  if (itr != vlog_level_map_.end()) {
    return level <= itr->second;
  }

  return false;
}

std::string VLogContext::str() const {
  std::stringstream result;
  for (const auto *entry : context_stack_) {
    ICHECK_NOTNULL(entry);
    result << entry->str();
    result << ": ";
  }
  return result.str();
}

} // namespace detail
} // namespace runtime
} // namespace tvm
