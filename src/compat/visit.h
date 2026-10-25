#ifndef TILELANG_COMPAT_VISIT_H_
#define TILELANG_COMPAT_VISIT_H_

#include <tvm/tirx/stmt_functor.h>

#include <functional>
#include <unordered_set>
#include <utility>

namespace tvm::tl {

class PostOrderVisitor : public tirx::StmtExprVisitor {
public:
  explicit PostOrderVisitor(
      std::function<void(const ffi::ObjectRef &)> callback)
      : callback_(std::move(callback)) {}

  ffi::Optional<ffi::VisitInterrupt> Visit(ffi::AnyView value) final {
    auto object = value.as<ffi::ObjectRef>();
    if (!object ||
        (!object.value().as<ExprNode>() && !object.value().as<StmtNode>())) {
      return StmtExprVisitor::Visit(value);
    }
    if (!visited_.insert(object.value().get()).second)
      return std::nullopt;
    auto interrupt = StmtExprVisitor::Visit(value);
    if (!interrupt)
      callback_(object.value());
    return interrupt;
  }

private:
  std::function<void(const ffi::ObjectRef &)> callback_;
  std::unordered_set<const ffi::Object *> visited_;
};

inline void
PostOrderVisit(const ffi::ObjectRef &root,
               std::function<void(const ffi::ObjectRef &)> callback) {
  ffi::make_object<PostOrderVisitor>(std::move(callback))->Visit(root);
}

} // namespace tvm::tl

#endif
