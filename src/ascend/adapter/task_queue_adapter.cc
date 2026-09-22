/*!
 * \file task_queue_adapter.cc
 * \brief torch_npu task queue adapter built on demand by
 *        tilelang.ascend.task_queue.ensure_task_queue_adapter()
 *
 * This file lives in src/ascend/adapter/, a directory outside the CMake glob
 * list, so it is not compiled into TileLang itself. It is built on demand by
 * tilelang.ascend.task_queue.ensure_task_queue_adapter() as a separate DSO
 * with the installed torch/torch_npu toolchain. C++ standard library objects
 * stay on that side of the C ABI below.
 */

#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <string>

#include <torch/extension.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#if defined(__GNUC__) || defined(__clang__)
#define TILELANG_TASK_QUEUE_EXPORT __attribute__((visibility("default")))
#else
#define TILELANG_TASK_QUEUE_EXPORT
#endif

using TileLangLaunchFn = int (*)(void *context);
using TileLangDestroyFn = void (*)(void *context);

extern "C" TILELANG_TASK_QUEUE_EXPORT int
tilelang_torch_npu_submit(const char *op_name, TileLangLaunchFn launch,
                          TileLangDestroyFn destroy, void *context,
                          int sync) noexcept {
  try {
    if (destroy == nullptr) {
      std::fprintf(stderr,
                   "[tilelang] torch_npu task queue submit failed: destroy "
                   "callback is null\n");
      return -1;
    }
    // Takes ownership as part of the C API contract, including every error
    // path below.  torch_npu copies the std::function into its queue, so the
    // shared owner remains alive until the queued callback is released.
    std::shared_ptr<void> owner(context, destroy);
    if (op_name == nullptr || launch == nullptr || context == nullptr) {
      std::fprintf(stderr,
                   "[tilelang] torch_npu task queue submit failed: invalid "
                   "null argument\n");
      return -1;
    }

    std::function<int()> call = [owner = std::move(owner), launch]() -> int {
      return launch(owner.get());
    };
    at_npu::native::OpCommand::RunOpApiV2(std::string(op_name), call,
                                          sync != 0);
    return 0;
  } catch (const std::exception &exception) {
    std::fprintf(stderr,
                 "[tilelang] torch_npu task queue submit failed for `%s`: %s\n",
                 op_name == nullptr ? "<null>" : op_name, exception.what());
  } catch (...) {
    std::fprintf(
        stderr,
        "[tilelang] torch_npu task queue submit failed for `%s`: unknown "
        "error\n",
        op_name == nullptr ? "<null>" : op_name);
  }
  return -1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("submit_address", []() {
    return reinterpret_cast<std::uintptr_t>(&tilelang_torch_npu_submit);
  });
}
