// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <assert.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

#include <cuda.h>

__device__ void save_blocks(int* block_offset, int range_start, int range_end, int block_size, int& block_count) {
    for (int idx = range_start; idx < range_end; idx += block_size) {
        block_offset[block_count++] = idx;
    }
}

__global__ void convert_vertical_slash_indexes_kernel(
    const int* seqlens,           // [BATCH, ]
    const int* vertical_indexes,  // [BATCH, N_HEADS, NNZ_V]
    const int* slash_indexes,     // [BATCH, N_HEADS, NNZ_S]
    int* block_count,             // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* block_offset,            // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    int* column_count,            // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* column_index,            // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    int N_HEADS,
    int N_ROWS,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int NNZ_V,
    int NNZ_S
) {
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int group_idx = blockIdx.z;

    int seqlen = seqlens[batch_idx];
    int block_idx_m = group_idx * blockDim.x + threadIdx.x;
    int start_m = block_idx_m * BLOCK_SIZE_M;
    if (start_m >= seqlen) {
        return;
    }
    int end_m = start_m + BLOCK_SIZE_M;
    vertical_indexes += (batch_idx * N_HEADS + head_idx) * NNZ_V;
    slash_indexes += (batch_idx * N_HEADS + head_idx) * NNZ_S;
    int row_offset = (batch_idx * N_HEADS + head_idx) * N_ROWS + block_idx_m;
    block_count += row_offset;
    block_offset += row_offset * NNZ_S;
    column_count += row_offset;
    column_index += row_offset * NNZ_V;

    int tmp_col_cnt = 0, tmp_blk_cnt = 0;
    int s = 0, v = 0;
    int v_idx = vertical_indexes[v++];
    int s_idx = slash_indexes[s++];
    while (s_idx >= end_m) {
        s_idx = slash_indexes[s++];
    }
    s_idx = max(end_m - s_idx, BLOCK_SIZE_M);
    int range_start = s_idx - BLOCK_SIZE_M, range_end = s_idx;
    while (1) {
        if (v_idx < range_end) {
            if (v_idx < range_start) {
                column_index[tmp_col_cnt++] = v_idx;
            }
            if (v < NNZ_V) {
                v_idx = vertical_indexes[v++];
            } else {
                v_idx = end_m + BLOCK_SIZE_M;
            }
        } else {
            if (s < NNZ_S) {
                s_idx = max(end_m - slash_indexes[s++], BLOCK_SIZE_M);
            } else {
                save_blocks(block_offset, range_start, range_end, BLOCK_SIZE_N, tmp_blk_cnt);
                break;
            }
            if (s_idx > range_end + BLOCK_SIZE_M) {
                save_blocks(block_offset, range_start, range_end, BLOCK_SIZE_N, tmp_blk_cnt);
                range_start = s_idx - BLOCK_SIZE_M;
                range_end = s_idx;
            } else if (s_idx > range_end) {
                range_end += BLOCK_SIZE_M;
            }
        }
    }

    block_count[0] = tmp_blk_cnt;
    column_count[0] = tmp_col_cnt;
}

void convert_vertical_slash_indexes_64x64(
    const int* seqlens,           // [BATCH, ]
    const int* vertical_indexes,  // [BATCH, N_HEADS, NNZ_V]
    const int* slash_indexes,     // [BATCH, N_HEADS, NNZ_S]
    int* block_count,             // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* block_offset,            // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    int* column_count,            // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* column_index,            // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    int BATCH_SIZE,
    int N_HEADS,
    int N_ROWS,
    int NNZ_V,
    int NNZ_S
) {
    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_N = 64;
    const int N_THREADS = 64;
    const dim3 dimBlock(N_THREADS);
    const dim3 dimGrid(N_HEADS, BATCH_SIZE, (N_ROWS + N_THREADS - 1) / N_THREADS);
    convert_vertical_slash_indexes_kernel<<<dimGrid, dimBlock>>>(
        seqlens, vertical_indexes, slash_indexes,
        block_count, block_offset, column_count, column_index,
        N_HEADS, N_ROWS, BLOCK_SIZE_M, BLOCK_SIZE_N, NNZ_V, NNZ_S
    );
}

std::vector<at::Tensor> convert_vertical_slash_indexes(
    torch::Tensor seqlens,           // [BATCH, ]
    torch::Tensor vertical_indexes,  // [BATCH, N_HEADS, NNZ_V]
    torch::Tensor slash_indexes,     // [BATCH, N_HEADS, NNZ_S]
    int context_size,
    int block_size_M,
    int block_size_N
) {
    assert(block_size_M == 64);
    assert(block_size_N == 64);

    cudaSetDevice(seqlens.get_device());

    int batch_size = slash_indexes.size(0);
    int num_heads = slash_indexes.size(1);
    int nnz_slash = slash_indexes.size(2);
    int nnz_vertical = vertical_indexes.size(2);
    int num_rows = (context_size + block_size_M - 1) / block_size_M;

    torch::Tensor block_count = torch::zeros({batch_size, num_heads, num_rows}, seqlens.options());
    torch::Tensor block_offset = torch::zeros({batch_size, num_heads, num_rows, nnz_slash}, seqlens.options());
    torch::Tensor column_count = torch::zeros({batch_size, num_heads, num_rows}, seqlens.options());
    torch::Tensor column_index = torch::zeros({batch_size, num_heads, num_rows, nnz_vertical}, seqlens.options());

    convert_vertical_slash_indexes_64x64(
        seqlens.data_ptr<int>(),
        vertical_indexes.data_ptr<int>(),
        slash_indexes.data_ptr<int>(),
        block_count.data_ptr<int>(),
        block_offset.data_ptr<int>(),
        column_count.data_ptr<int>(),
        column_index.data_ptr<int>(),
        batch_size,
        num_heads,
        num_rows,
        nnz_vertical,
        nnz_slash
    );

    return { block_count, block_offset, column_count, column_index };
}
