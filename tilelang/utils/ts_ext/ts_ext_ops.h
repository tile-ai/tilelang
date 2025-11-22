#pragma once
#include <cstdint>
#include <optional>
#include <pybind11/pytypes.h>
#include <string>
#include <torch/types.h>
#include <vector>

torch::Tensor tensor_from_ptr(uint64_t ptr_val, std::vector<int64_t> shape,
                              const std::string &dtype = "float32",
                              int64_t device = 0, bool take_ownership = false);

torch::Tensor create_tensor(const std::vector<int64_t> &shape,
                            c10::ScalarType dtype);

torch::Tensor get_device_tensor(torch::Tensor tensor);

pybind11::bytearray create_ipc_handle(void *ptr);

void sync_ipc_handles(
    int rank, const std::vector<int> &device_ids, void **buffer_ptrs_gpu,
    const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles,
    const std::optional<pybind11::bytearray> &root_unique_id_opt);
