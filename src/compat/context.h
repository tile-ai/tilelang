#ifndef TILELANG_COMPAT_CONTEXT_H_
#define TILELANG_COMPAT_CONTEXT_H_

#include <tvm/ir/with_context.h>

#include <exception>
#include <memory>
#include <vector>

namespace tvm::tl {

template <typename Context> class WithGroup {
public:
  WithGroup() = default;
  WithGroup(WithGroup &&) = default;
  WithGroup &operator=(WithGroup &&other) {
    if (this != &other) {
      Clear();
      entries_ = std::move(other.entries_);
    }
    return *this;
  }
  WithGroup(const WithGroup &) = delete;
  WithGroup &operator=(const WithGroup &) = delete;

  template <typename... Arguments> void Emplace(Arguments &&...arguments) {
    entries_.reserve(entries_.size() + 1);
    entries_.emplace_back(
        new tvm::With<Context>(std::forward<Arguments>(arguments)...));
  }

  size_t size() const { return entries_.size(); }

  ~WithGroup() noexcept(false) { Clear(); }

private:
  void Clear() {
    std::exception_ptr first_exception;
    while (!entries_.empty()) {
      auto entry = entries_.back().release();
      entries_.pop_back();
      try {
        delete entry;
      } catch (...) {
        if (!std::uncaught_exceptions() && !first_exception)
          first_exception = std::current_exception();
      }
    }
    if (first_exception)
      std::rethrow_exception(first_exception);
  }

  std::vector<std::unique_ptr<tvm::With<Context>>> entries_;
};

template <typename ScopeEntry> class ContextStack {
public:
  ContextStack() { entries_.push_back(std::make_unique<ScopeEntry>()); }
  ContextStack(const ContextStack &) = delete;
  ContextStack &operator=(const ContextStack &) = delete;
  ScopeEntry &Current() { return *entries_.back(); }
  const ScopeEntry &Current() const { return *entries_.back(); }

  template <typename Callback>
  auto WithNewScope(Callback &&callback) -> decltype(callback()) {
    entries_.push_back(std::make_unique<ScopeEntry>());
    struct Guard {
      ContextStack *stack;
      ~Guard() noexcept(false) { stack->Pop(); }
    } guard{this};
    return callback();
  }

  ~ContextStack() noexcept(false) {
    std::exception_ptr first_exception;
    while (!entries_.empty()) {
      try {
        Pop();
      } catch (...) {
        if (!first_exception)
          first_exception = std::current_exception();
      }
    }
    if (first_exception)
      std::rethrow_exception(first_exception);
  }

private:
  void Pop() {
    auto entry = entries_.back().release();
    entries_.pop_back();
    try {
      delete entry;
    } catch (...) {
      if (!std::uncaught_exceptions())
        throw;
    }
  }

  std::vector<std::unique_ptr<ScopeEntry>> entries_;
};

} // namespace tvm::tl

#endif
