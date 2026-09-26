/* Native partial application for packed runtime functions. */
#include "support/check.h"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace ffi;

namespace {

class BoundFramePool {
public:
  BoundFramePool(Function callee, int64_t parameter_count,
                 Array<int64_t> static_indices, Array<Any> static_values,
                 Array<int64_t> dynamic_indices)
      : callee_(std::move(callee)), static_values_(std::move(static_values)),
        dynamic_indices_(std::move(dynamic_indices)),
        template_frame_(parameter_count) {
    for (int64_t i = 0; i < static_indices.size(); ++i)
      template_frame_[static_indices[i]] = static_values_[i];
  }

  void Invoke(PackedArgs dynamic_args, Any *result) {
    ICHECK(dynamic_args.size() == dynamic_indices_.size())
        << "bound function received the wrong number of dynamic arguments";

    std::vector<AnyView> frame = Acquire();
    for (int64_t i = 0; i < dynamic_indices_.size(); ++i)
      frame[dynamic_indices_[i]] = dynamic_args[i];

    try {
      callee_.CallPacked(frame.data(), static_cast<int32_t>(frame.size()),
                         result);
    } catch (...) {
      Release(std::move(frame));
      throw;
    }
    Release(std::move(frame));
  }

private:
  std::vector<AnyView> Acquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (available_.empty())
      return template_frame_;
    std::vector<AnyView> frame = std::move(available_.back());
    available_.pop_back();
    return frame;
  }

  void Release(std::vector<AnyView> frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    available_.push_back(std::move(frame));
  }

  Function callee_;
  // Own the values referenced by static AnyViews for the pool's full lifetime.
  Array<Any> static_values_;
  Array<int64_t> dynamic_indices_;
  std::vector<AnyView> template_frame_;
  std::mutex mutex_;
  std::vector<std::vector<AnyView>> available_;
};

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def_packed(
      "tilelang.runtime.bind_packed_function", [](PackedArgs args, Any *ret) {
        ICHECK(args.size() == 5)
            << "bind_packed_function expects function, parameter count, static "
               "indices, static values, and dynamic indices";
        Function callee = args[0].cast<Function>();
        int64_t parameter_count = args[1].cast<int64_t>();
        Array<int64_t> static_indices = args[2].cast<Array<int64_t>>();
        Array<Any> static_values = args[3].cast<Array<Any>>();
        Array<int64_t> dynamic_indices = args[4].cast<Array<int64_t>>();

        ICHECK(parameter_count >= 0);
        ICHECK(static_indices.size() == static_values.size());
        std::vector<bool> occupied(parameter_count, false);
        for (int64_t index : static_indices) {
          ICHECK(index >= 0 && index < parameter_count);
          ICHECK(!occupied[index]);
          occupied[index] = true;
        }
        for (int64_t index : dynamic_indices) {
          ICHECK(index >= 0 && index < parameter_count);
          ICHECK(!occupied[index]);
          occupied[index] = true;
        }
        for (bool slot : occupied)
          ICHECK(slot);

        auto pool = std::make_shared<BoundFramePool>(
            std::move(callee), parameter_count, std::move(static_indices),
            std::move(static_values), std::move(dynamic_indices));
        *ret =
            Function::FromPacked([pool](PackedArgs dynamic_args, Any *result) {
              pool->Invoke(dynamic_args, result);
            });
      });
}

} // namespace tl
} // namespace tvm
