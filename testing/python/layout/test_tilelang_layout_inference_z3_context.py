# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.autotuner.grouped_compile import compile_grouped_unit_tvm_ffi
from tilelang.autotuner.param import CompileArgs


_STATE_CACHE_CASES = (
    (266, 138, 44, 576),
    (266, 138, 44, 8),
    (145, 17, 21, 2048),
    (145, 17, 21, 5120),
)

_STATE_CACHE_CONFIG_KEYS = (
    "block_size_kvstore",
    "block_size_inference",
    "num_layers",
    "feature_dim",
)

_STATE_CACHE_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
}


def _make_state_cache_kernel(
    block_size_kvstore: int,
    block_size_inference: int,
    num_layers: int,
    feature_dim: int,
):
    cache_layer_stride = T.dynamic("cache_layer_stride")
    cache_token_stride = T.dynamic("cache_token_stride")
    tensor_layer_stride = T.dynamic("tensor_layer_stride")
    tensor_token_stride = T.dynamic("tensor_token_stride")

    block = 64
    enable_pdl_sync = block_size_inference * feature_dim >= 65536

    @T.prim_func
    def kernel(
        state_tensor: T.StridedTensor(
            (num_layers, block_size_inference, feature_dim),
            (tensor_layer_stride, tensor_token_stride, 1),
            "uint8",
        ),
        state_cache: T.StridedTensor(
            (num_layers, block_size_kvstore, feature_dim),
            (cache_layer_stride, cache_token_stride, 1),
            "uint8",
        ),
        last_hit_position: T.int32,
    ):
        T.assume(cache_layer_stride % 8 == 0)
        T.assume(cache_token_stride % 8 == 0)
        T.assume(tensor_layer_stride % 8 == 0)
        T.assume(tensor_token_stride % 8 == 0)

        start_pos = last_hit_position - block_size_inference

        with T.Kernel(num_layers, T.ceildiv(block_size_inference, block)) as (bx, by):
            if enable_pdl_sync:
                T.pdl_sync()
            for token_idx, feat_idx in T.Parallel(block, feature_dim):
                actual_token_idx = by * block + token_idx
                if actual_token_idx < block_size_inference:
                    source_token_idx = (start_pos + actual_token_idx + block_size_kvstore) % block_size_kvstore
                    target_token_idx = (start_pos + actual_token_idx + block_size_inference) % block_size_inference
                    state_tensor[bx, target_token_idx, feat_idx] = state_cache[bx, source_token_idx, feat_idx]

    return kernel


@tilelang.testing.requires_cuda
def test_layout_inference_isolated_from_previous_compile_z3_state():
    cache_was_enabled = tilelang.is_cache_enabled()
    tilelang.disable_cache()
    try:
        for config in _STATE_CACHE_CASES:
            tilelang.compile(
                _make_state_cache_kernel(*config),
                target="cuda",
                pass_configs=_STATE_CACHE_PASS_CONFIGS,
            )
    finally:
        if cache_was_enabled:
            tilelang.enable_cache()


@tilelang.testing.requires_cuda
def test_grouped_layout_inference_uses_fresh_z3_context_per_kernel():
    unit_items = [(index, dict(zip(_STATE_CACHE_CONFIG_KEYS, config))) for index, config in enumerate(_STATE_CACHE_CASES)]
    compile_args = CompileArgs(
        execution_backend="tvm_ffi",
        target=tvm.target.Target("cuda"),
        pass_configs=_STATE_CACHE_PASS_CONFIGS,
    )

    results = compile_grouped_unit_tvm_ffi(
        unit_items=unit_items,
        compile_args=compile_args,
        elaborate_func=_make_state_cache_kernel,
    )

    assert len(results) == len(unit_items)
    for _, _, kernel, error in results:
        if error is not None:
            raise error
        assert kernel is not None


if __name__ == "__main__":
    tilelang.testing.main()
