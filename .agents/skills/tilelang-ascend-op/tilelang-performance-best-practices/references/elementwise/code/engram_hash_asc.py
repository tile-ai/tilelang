import math

import tilelang
from tilelang import language as T

from tile_kernels.config import get_num_vec_cores


@tilelang.jit
def get_engram_hash_kernel_asc(
    max_ngram_size: int = 3,
    num_ngram_layers: int = 2,
    num_embed_table_per_ngram: int = 8,
):
    """Compute Engram hash indices on Ascend NPU vector cores."""
    num_tokens = T.dynamic("num_tokens")
    available_cores = get_num_vec_cores()
    cores_per_layer = available_cores // num_ngram_layers
    num_out_cols = (max_ngram_size - 1) * num_embed_table_per_ngram
    max_simt_threads = 1024
    input_token_alignment = 8 // math.gcd(max_ngram_size, 8)
    output_token_alignment = 8 // math.gcd(num_out_cols, 8)
    token_alignment = math.lcm(input_token_alignment, output_token_alignment)
    max_tile_tokens = max_simt_threads // num_out_cols
    assert max_tile_tokens >= token_alignment
    tile_tokens = max_tile_tokens // token_alignment * token_alignment
    simt_threads = tile_tokens * num_out_cols

    num_cores = cores_per_layer * num_ngram_layers

    @T.prim_func
    def engram_hash_kernel_asc(
        ngram_token_ids: T.Tensor[(num_tokens, max_ngram_size), T.int32],
        multipliers: T.Tensor[(num_ngram_layers, max_ngram_size), T.int64],
        vocab_sizes: T.Tensor[
            (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram),
            T.int32,
        ],
        offsets: T.Tensor[(num_ngram_layers, num_out_cols), T.int32],
        output: T.Tensor[(num_ngram_layers, num_tokens, num_out_cols), T.int32],
    ):
        with T.Kernel(num_cores) as core_id:
            layer_idx = core_id // cores_per_layer
            layer_core_id = core_id % cores_per_layer
            token_ids_ub = T.alloc_shared((tile_tokens, max_ngram_size), T.int32)
            multipliers_ub = T.alloc_shared((max_ngram_size,), T.int64)
            vocab_sizes_ub = T.alloc_shared((max_ngram_size - 1, num_embed_table_per_ngram), T.int32)
            offsets_ub = T.alloc_shared((num_out_cols,), T.int32)
            output_ub = T.alloc_shared((tile_tokens, num_out_cols), T.int32)

            # Layer-owned lookup tables remain resident and single-versioned.
            T.copy(multipliers[layer_idx, :], multipliers_ub)
            T.copy(vocab_sizes[layer_idx, :, :], vocab_sizes_ub)
            T.copy(offsets[layer_idx, :], offsets_ub)

            for tile_idx in T.Persistent(
                [T.ceildiv(num_tokens, tile_tokens)],
                cores_per_layer,
                layer_core_id,
                group_size=1,
                num_stages=0,
            ):
                token_start = tile_idx * tile_tokens
                valid_tokens = T.min(tile_tokens, num_tokens - token_start)

                T.copy(
                    ngram_token_ids[token_start : token_start + valid_tokens, :],
                    token_ids_ub[:valid_tokens, :],
                )

                with T.SimtVF(threads=simt_threads):
                    thread_idx = T.get_thread_binding()
                    token_in_tile = thread_idx // num_out_cols
                    out_col = thread_idx % num_out_cols
                    if token_in_tile < valid_tokens:
                        last_ngram_idx = out_col // num_embed_table_per_ngram + 1
                        out_table_idx = out_col % num_embed_table_per_ngram
                        first_token_id = T.cast(token_ids_ub[token_in_tile, 0], T.int64)
                        first_product = first_token_id * multipliers_ub[0]
                        hash_value = T.alloc_var(T.int64, init=first_product)
                        for hash_ngram_idx in T.serial(1, max_ngram_size):
                            if hash_ngram_idx <= last_ngram_idx:
                                token_id = T.cast(
                                    token_ids_ub[token_in_tile, hash_ngram_idx],
                                    T.int64,
                                )
                                product = token_id * multipliers_ub[hash_ngram_idx]
                                hash_value = T.bitwise_xor(hash_value, product)

                        vocab_size = T.cast(
                            vocab_sizes_ub[last_ngram_idx - 1, out_table_idx],
                            T.int64,
                        )
                        remainder = T.truncmod(hash_value, vocab_size)
                        hash_index = T.Select(
                            remainder < 0,
                            remainder + vocab_size,
                            remainder,
                        )
                        output_ub[token_in_tile, out_col] = T.cast(hash_index, T.int32) + offsets_ub[out_col]

                T.copy(
                    output_ub[:valid_tokens, :],
                    output[
                        layer_idx,
                        token_start : token_start + valid_tokens,
                        :,
                    ],
                )

    return engram_hash_kernel_asc
