#include <tvm/runtime/logging.h>

#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

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
constexpr const size_t kSrcPrefixLength = 5;
constexpr const char *kDefaultKeyword = "DEFAULT";

std::string FileToVLogMapKey(const std::string &filename) {
  size_t last_src =
      filename.rfind(kSrcPrefix, std::string::npos, kSrcPrefixLength);
  if (last_src == std::string::npos) {
    std::string no_slash_src{kSrcPrefix + 1};
    if (filename.substr(0, no_slash_src.size()) == no_slash_src) {
      return filename.substr(no_slash_src.size());
    }
  }
  return (last_src == std::string::npos)
             ? filename
             : filename.substr(last_src + kSrcPrefixLength);
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
    int pos = spec_stream.tellg();
    if (pos == -1) {
      pos = spec.size() - last_read.size();
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

} // namespace detail
} // namespace runtime
} // namespace tvm
