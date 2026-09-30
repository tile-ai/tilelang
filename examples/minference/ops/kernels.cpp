// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "torch/extension.h"
#include <cassert>
#include <vector>

#include <cuda_runtime_api.h>

// Keep PyTorch headers in the host translation unit: nvcc 13.4 fails
// to compile the dependent List types pulled in by torch/extension.h.
void convert_vertical_slash_indexes_64x64(
    const int *seqlens,          // [BATCH, ]
    const int *vertical_indexes, // [BATCH, N_HEADS, NNZ_V]
    const int *slash_indexes,    // [BATCH, N_HEADS, NNZ_S]
    int *block_count,            // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int *block_offset, // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    int *column_count, // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int *column_index, // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    int BATCH_SIZE, int N_HEADS, int N_ROWS, int NNZ_V, int NNZ_S);

std::vector<at::Tensor> convert_vertical_slash_indexes(
    torch::Tensor seqlens,          // [BATCH, ]
    torch::Tensor vertical_indexes, // [BATCH, N_HEADS, NNZ_V]
    torch::Tensor slash_indexes,    // [BATCH, N_HEADS, NNZ_S]
    int context_size, int block_size_M, int block_size_N) {
  assert(block_size_M == 64);
  assert(block_size_N == 64);

  cudaSetDevice(seqlens.get_device());

  int batch_size = slash_indexes.size(0);
  int num_heads = slash_indexes.size(1);
  int nnz_slash = slash_indexes.size(2);
  int nnz_vertical = vertical_indexes.size(2);
  int num_rows = (context_size + block_size_M - 1) / block_size_M;

  torch::Tensor block_count =
      torch::zeros({batch_size, num_heads, num_rows}, seqlens.options());
  torch::Tensor block_offset = torch::zeros(
      {batch_size, num_heads, num_rows, nnz_slash}, seqlens.options());
  torch::Tensor column_count =
      torch::zeros({batch_size, num_heads, num_rows}, seqlens.options());
  torch::Tensor column_index = torch::zeros(
      {batch_size, num_heads, num_rows, nnz_vertical}, seqlens.options());

  convert_vertical_slash_indexes_64x64(
      seqlens.data_ptr<int>(), vertical_indexes.data_ptr<int>(),
      slash_indexes.data_ptr<int>(), block_count.data_ptr<int>(),
      block_offset.data_ptr<int>(), column_count.data_ptr<int>(),
      column_index.data_ptr<int>(), batch_size, num_heads, num_rows,
      nnz_vertical, nnz_slash);

  return {block_count, block_offset, column_count, column_index};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("convert_vertical_slash_indexes", &convert_vertical_slash_indexes,
        "dynamic sparse index function");
}
