// Optional MLX runtime. MLX's primitive API owns array and stream lifetimes.
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlx/backend/metal/device.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;
namespace {
class TileLangPrimitive : public mx::Primitive {
public:
  TileLangPrimitive(mx::Stream stream, std::string source, std::string symbol,
                    std::string key, std::vector<int> bindings,
                    std::vector<int> aliases, std::array<int, 3> grid,
                    std::array<int, 3> threads,
                    std::vector<std::vector<int64_t>> strides)
      : Primitive(stream), source_(std::move(source)),
        symbol_(std::move(symbol)), key_(std::move(key)),
        bindings_(std::move(bindings)), aliases_(std::move(aliases)),
        grid_(grid), threads_(threads), strides_(std::move(strides)) {}

  void eval_cpu(const std::vector<mx::array> &,
                std::vector<mx::array> &) override {
    throw std::runtime_error("TileLang MLX kernels require a GPU stream");
  }

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override {
    // A state destination must never be copied to repair its layout: doing so
    // would silently stop updating the caller's cache allocation.
    std::vector<size_t> spans;
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto &input = inputs[i];
      bool matches_strides = input.strides().size() == strides_[i].size();
      for (size_t axis = 0; matches_strides && axis < input.ndim(); ++axis) {
        // A singleton dimension is always indexed at zero. MLX may assign it
        // stride zero while TileLang declares the compact stride.
        if (input.shape(axis) != 1 &&
            input.strides(axis) != strides_[i][axis]) {
          matches_strides = false;
        }
      }
      if (!matches_strides) {
        throw std::invalid_argument(
            "MLX inputs must be row-contiguous or match declared static "
            "strides; state is never copied");
      }
      size_t span = 1;
      for (size_t axis = 0; axis < input.ndim(); ++axis) {
        span += (input.shape(axis) - 1) * input.strides(axis);
      }
      spans.push_back(span * input.itemsize());
    }
    // Views of the same allocation may overlap. raw_ptr identifies the Metal
    // buffer, not a CPU address range; compare offsets only within that buffer.
    for (int alias : aliases_) {
      if (alias < 0)
        continue;
      auto begin = inputs[alias].offset();
      auto end = begin + spans[alias];
      for (size_t j = 0; j < inputs.size(); ++j) {
        if (j == static_cast<size_t>(alias) ||
            inputs[alias].buffer().raw_ptr() != inputs[j].buffer().raw_ptr())
          continue;
        auto other = inputs[j].offset();
        if (begin < other + spans[j] && other < end) {
          throw std::invalid_argument("In-place state overlaps another input");
        }
      }
    }
    auto &device = mx::metal::device(stream().device);
    auto *library = device.get_library(key_, [this] { return source_; });
    auto *kernel = device.get_kernel(symbol_, library);
    size_t threads =
        static_cast<size_t>(threads_[0]) * threads_[1] * threads_[2];
    if (threads > kernel->maxTotalThreadsPerThreadgroup()) {
      throw std::invalid_argument(
          "TileLang threadgroup exceeds the Metal kernel limit");
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (aliases_[i] >= 0) {
        outputs[i].copy_shared_buffer(inputs[aliases_[i]]);
      } else {
        outputs[i].set_data(mx::allocator::malloc(outputs[i].nbytes()));
      }
    }
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    for (size_t i = 0; i < bindings_.size(); ++i) {
      int binding = bindings_[i];
      if (binding >= 0) {
        encoder.set_input_array(inputs[binding], static_cast<int>(i));
      } else {
        encoder.set_output_array(outputs[-binding - 1], static_cast<int>(i));
      }
    }
    encoder.dispatch_threadgroups(
        MTL::Size(grid_[0], grid_[1], grid_[2]),
        MTL::Size(threads_[0], threads_[1], threads_[2]));
  }

  const char *name() const override { return "TileLangMetal"; }
  // In-place calls are ordered effects, not common subexpressions.
  bool is_equivalent(const mx::Primitive &) const override { return false; }

private:
  std::string source_, symbol_, key_;
  std::vector<int> bindings_, aliases_;
  std::array<int, 3> grid_, threads_;
  std::vector<std::vector<int64_t>> strides_;
};

std::vector<mx::array>
Launch(const std::vector<mx::array> &inputs,
       const std::vector<std::vector<int>> &shape_values,
       const std::vector<mx::Dtype> &dtypes, const std::string &source,
       const std::string &symbol, const std::string &key,
       const std::vector<int> &bindings, const std::vector<int> &aliases,
       const std::array<int, 3> &grid, const std::array<int, 3> &threads,
       const std::vector<std::vector<int64_t>> &strides, mx::Stream stream) {
  if (strides.size() != inputs.size())
    throw std::invalid_argument("Invalid input stride metadata");
  std::vector<mx::Shape> shapes;
  for (const auto &shape : shape_values)
    shapes.emplace_back(shape.begin(), shape.end());
  if (shapes.size() != dtypes.size() || shapes.size() != aliases.size()) {
    throw std::invalid_argument("Inconsistent TileLang output metadata");
  }
  for (int binding : bindings) {
    if ((binding >= 0 && static_cast<size_t>(binding) >= inputs.size()) ||
        (binding < 0 && static_cast<size_t>(-static_cast<int64_t>(binding) -
                                            1) >= shapes.size())) {
      throw std::invalid_argument("Invalid TileLang buffer binding");
    }
  }
  for (size_t i = 0; i < aliases.size(); ++i) {
    int alias = aliases[i];
    if (alias < -1 ||
        (alias >= 0 && static_cast<size_t>(alias) >= inputs.size())) {
      throw std::invalid_argument("Invalid TileLang state alias");
    }
    if (alias >= 0 && (inputs[alias].shape() != shapes[i] ||
                       inputs[alias].dtype() != dtypes[i])) {
      throw std::invalid_argument(
          "State aliases must preserve shape and dtype");
    }
  }
  for (int i = 0; i < 3; ++i) {
    if (grid[i] <= 0 || threads[i] <= 0) {
      throw std::invalid_argument("Launch dimensions must be positive");
    }
  }
  auto primitive = std::make_shared<TileLangPrimitive>(
      stream, source, symbol, key, bindings, aliases, grid, threads, strides);
  return mx::array::make_arrays(shapes, dtypes, primitive, inputs);
}
} // namespace

NB_MODULE(_tilelang_mlx, m) {
  m.attr("mlx_version") = "0.32.4";
  m.def("launch", &Launch);
}
