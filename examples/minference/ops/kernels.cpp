// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "torch/extension.h"
#include <vector>

std::vector<at::Tensor> convert_vertical_slash_indexes(
    torch::Tensor seqlens,          // [BATCH, ]
    torch::Tensor vertical_indexes, // [BATCH, N_HEADS, NNZ_V]
    torch::Tensor slash_indexes,    // [BATCH, N_HEADS, NNZ_S]
    int context_size, int block_size_M, int block_size_N);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("convert_vertical_slash_indexes", &convert_vertical_slash_indexes,
        "dynamic sparse index function");
}
