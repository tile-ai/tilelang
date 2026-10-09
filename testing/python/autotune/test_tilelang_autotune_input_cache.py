import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.autotuner import AutoTuner
from tilelang.autotuner.tuner import _BenchmarkWorkerState


def _benchmark_input_configs(kernel, supply, ref_prog):
    tuner = AutoTuner(kernel, configs=[{"variant": 0}, {"variant": 1}]).set_profile_args(
        supply_prog=supply, ref_prog=ref_prog, cache_input_tensors=True
    )
    state = _BenchmarkWorkerState()
    # Control trial order and require every trial to pass validation and timing.
    # Repeat each signature to also check that compatible inputs are reused.
    for variant in (0, 0, 1, 1):
        compiled = tilelang.compile(kernel(variant), out_idx=[2], target="cuda", execution_backend="tvm_ffi")
        latency, _ = tuner._benchmark_target(compiled, warmup=1, rep=1, early_stop_factor=2.0, benchmark_state=state)
        assert latency > 0


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("cache_inputs", [True, False])
@pytest.mark.parametrize("symbolic", [False, True])
def test_autotune_reuses_compatible_inputs(cache_inputs, symbolic):
    supplied = []

    def supply(params):
        inputs = [
            torch.ones(tuple(256 if isinstance(dim, T.Var) else int(dim) for dim in p.shape), dtype=p.torch_dtype(), device="cuda")
            for p in params
        ]
        supplied.append(inputs)
        return inputs

    def kernel(threads):
        size = T.dynamic("n") if symbolic else 256

        @T.prim_func
        def main(A: T.Tensor((size,), T.float32), B: T.Tensor((size,), T.float32)):
            with T.Kernel(T.ceildiv(size, 256), threads=threads) as bx:
                for i in T.Parallel(256):
                    B[bx * 256 + i] = A[bx * 256 + i] + 1

        return main

    result = (
        AutoTuner(kernel, configs=[{"threads": 64}, {"threads": 128}])
        .set_compile_args(out_idx=[1], target="cuda")
        .set_profile_args(supply_prog=supply, ref_prog=lambda x: x + 1, cache_input_tensors=cache_inputs)
        .run(warmup=1, rep=1)
    )
    # The reference benchmark has its own inputs.
    assert len(supplied) == (2 if cache_inputs else 3)
    x = supplied[0][0]
    torch.testing.assert_close(result.kernel(x), x + 1)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("change", ["rank", "extent", "dtype"])
def test_autotune_refreshes_incompatible_inputs(change):
    supplied = []

    def supply(params):
        inputs = [torch.ones(tuple(p.shape), dtype=p.torch_dtype(), device="cuda") for p in params]
        supplied.append(inputs)
        return inputs

    def kernel(variant):
        size = 128 if change == "extent" and variant else 256
        dtype = T.float16 if change == "dtype" and variant else T.float32
        shape = (size, 1) if change == "rank" and variant else (size,)

        @T.prim_func
        def main(A: T.Tensor(shape, dtype), B: T.Tensor(shape, dtype)):
            with T.Kernel(1, threads=128):
                T.copy(A, B)

        return main

    result = (
        AutoTuner(kernel, configs=[{"variant": 0}, {"variant": 1}])
        .set_compile_args(out_idx=[1], target="cuda")
        .set_profile_args(supply_prog=supply, ref_prog=lambda x: x, cache_input_tensors=True)
        .run(warmup=1, rep=1)
    )
    assert len(supplied) == 3
    param = result.kernel.params[0]
    x = torch.ones(tuple(param.shape), dtype=param.torch_dtype(), device="cuda")
    torch.testing.assert_close(result.kernel(x), x)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("same_name", [False, True])
def test_autotune_refreshes_conflicting_symbolic_inputs(same_name):
    supplied = []

    def supply(params):
        bindings = {}
        inputs = []
        for p in params:
            dim = p.shape[0]
            size = bindings.setdefault(dim, 256 if not bindings else 128)
            inputs.append(torch.ones((size,), dtype=p.torch_dtype(), device="cuda"))
        supplied.append(inputs)
        return inputs

    def kernel(variant):
        n = T.dynamic("n")
        other = n if variant else T.dynamic("n" if same_name else "m")

        @T.prim_func
        def main(A: T.Tensor((n,), T.float32), B: T.Tensor((other,), T.float32), C: T.Tensor((n,), T.float32)):
            with T.Kernel(T.ceildiv(n, 256), threads=128) as bx:
                for i in T.Parallel(256):
                    C[bx * 256 + i] = A[bx * 256 + i] + B[0]

        return main

    _benchmark_input_configs(kernel, supply, lambda a, b: a + b[0])
    # Initial trial, separate reference inputs, then the changed signature.
    assert len(supplied) == 3
    assert [tuple(x.shape) for x in supplied[0]] == [(256,), (128,)]
    assert [tuple(x.shape) for x in supplied[-1]] == [(256,), (256,)]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("scalar_dtypes", [(T.float32, T.int32), (T.int32, T.float32), (T.int32, T.int64)])
def test_autotune_refreshes_changed_scalar_dtype(scalar_dtypes):
    supplied = []

    def supply(params):
        inputs = [
            torch.ones((256,), dtype=params[0].torch_dtype(), device="cuda"),
            1.25 if params[1].torch_dtype().is_floating_point else 1,
        ]
        supplied.append(inputs)
        return inputs

    def kernel(variant):
        scalar_dtype = scalar_dtypes[variant]

        @T.prim_func
        def main(A: T.Tensor((256,), T.float32), value: scalar_dtype, C: T.Tensor((256,), T.float32)):
            with T.Kernel(1, threads=128):
                for i in T.Parallel(256):
                    C[i] = A[i] + T.cast(value, T.float32)

        return main

    _benchmark_input_configs(kernel, supply, lambda a, value: a + value)
    assert len(supplied) == 3


def _add_one(threads):
    @T.prim_func
    def main(A: T.Tensor((256,), T.float32), B: T.Tensor((256,), T.float32)):
        with T.Kernel(1, threads=threads):
            for i in T.Parallel(256):
                B[i] = A[i] + 1

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("change_supplier", [False, True])
def test_autotune_refreshes_inputs_between_runs(change_supplier):
    supplied = []
    expected_value = 1

    def make_supply(value):
        def supply(params):
            inputs = [torch.full(tuple(p.shape), value, dtype=p.torch_dtype(), device="cuda") for p in params]
            supplied.append(inputs)
            return inputs

        return supply

    def reference(x):
        torch.testing.assert_close(x, torch.full_like(x, expected_value))
        return x + 1

    tuner = (
        AutoTuner(_add_one, configs=[{"threads": 64}, {"threads": 128}])
        .set_compile_args(out_idx=[1], target="cuda")
        .set_profile_args(supply_prog=make_supply(1), ref_prog=reference, cache_input_tensors=True)
    )
    tuner.run(warmup=1, rep=1)
    assert len(supplied) == 2

    if change_supplier:
        expected_value = 7
        tuner.set_profile_args(supply_prog=make_supply(7), ref_prog=reference, cache_input_tensors=True)

    result = tuner.run(warmup=1, rep=1)
    # Each run supplies inputs once for both trials and once for reference timing.
    assert len(supplied) == 4
    x = supplied[-1][0]
    torch.testing.assert_close(result.kernel(x), x + 1)


@tilelang.testing.requires_cuda
def test_autotune_refreshes_inputs_when_device_changes():
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least two CUDA devices")

    supplied = []

    def supply(params):
        inputs = [torch.ones(tuple(p.shape), dtype=p.torch_dtype(), device="cuda") for p in params]
        supplied.append(inputs)
        return inputs

    def reference(x):
        assert x.device == torch.device("cuda", torch.cuda.current_device())
        return x + 1

    tuner = (
        AutoTuner(_add_one, configs=[{"threads": 64}, {"threads": 128}])
        .set_compile_args(out_idx=[1], target="cuda")
        .set_profile_args(supply_prog=supply, ref_prog=reference, cache_input_tensors=True)
    )
    first_device = torch.cuda.current_device()
    next_device = (first_device + 1) % torch.cuda.device_count()
    for run_index, device in enumerate((first_device, next_device), start=1):
        with torch.cuda.device(device):
            result = tuner.run(warmup=1, rep=1)
            assert len(supplied) == run_index * 2
            assert all(inputs[0].device == torch.device("cuda", device) for inputs in supplied[-2:])
            x = supplied[-1][0]
            torch.testing.assert_close(result.kernel(x), x + 1)


if __name__ == "__main__":
    tilelang.testing.main()
