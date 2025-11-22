#include "ts_ext_ops.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "TileScale unified CUDA/IPCs extension";

  // alloc_cuda API
  m.def("tensor_from_ptr", &tensor_from_ptr, py::arg("ptr"), py::arg("shape"),
        py::arg("dtype") = std::string("float32"), py::arg("device") = 0,
        py::arg("take_ownership") = false);

  // ipc_ext API
  m.def(
      "_create_tensor",
      [](const std::vector<int64_t> &shape, const py::object &dtype) {
        return create_tensor(shape,
                             torch::python::detail::py_object_to_dtype(dtype));
      },
      py::arg("shape"), py::arg("dtype"));

  m.def("get_device_tensor", &get_device_tensor, py::arg("tensor"));

  m.def(
      "_create_ipc_handle",
      [](uintptr_t ptr_value) {
        void *ptr = reinterpret_cast<void *>(ptr_value);
        return create_ipc_handle(ptr);
      },
      py::arg("ptr_value"));

  m.def(
      "_sync_ipc_handles",
      [](int rank, const std::vector<int> &device_ids,
         uintptr_t buffer_ptrs_gpu_addr,
         const std::vector<std::optional<py::bytearray>> &all_gathered_handles,
         const std::optional<py::bytearray> &root_unique_id_opt) {
        void **buffer_ptrs_gpu =
            reinterpret_cast<void **>(buffer_ptrs_gpu_addr);
        sync_ipc_handles(rank, device_ids, buffer_ptrs_gpu,
                         all_gathered_handles, root_unique_id_opt);
      },
      py::arg("rank"), py::arg("device_ids"), py::arg("buffer_ptrs_gpu_addr"),
      py::arg("all_gathered_handles"), py::arg("root_unique_id_opt"));
}
