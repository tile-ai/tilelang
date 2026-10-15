import pytest

import tilelang
import tilelang.language as T
import tilelang.testing


CUDA_SM86_TARGET = {"kind": "cuda", "arch": "sm_86"}
CUDA_SM90_TARGET = {"kind": "cuda", "arch": "sm_90"}


def _barrier_kernel():
    """A minimal kernel whose only special feature is a shared barrier."""

    @T.prim_func
    def main(C: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1) as _:
            bar = T.alloc_barrier(1)  # noqa: F841
            C[0] = 0.0

    return main


def _no_barrier_kernel():
    """The same minimal kernel without any barrier, used as a control."""

    @T.prim_func
    def main(C: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1) as _:
            C[0] = 0.0

    return main


def _compile(target, kernel_factory=_barrier_kernel):
    """Compile a freshly built kernel for ``target`` with caching disabled."""
    tilelang.disable_cache()
    try:
        return tilelang.compile(kernel_factory(), out_idx=-1, target=target)
    finally:
        tilelang.enable_cache()


@tilelang.testing.requires_cuda
def test_alloc_barrier_rejected_on_pre_hopper_target():
    """T.alloc_barrier() must be rejected with a clear error on sm_86 instead of
    emitting sm_90-only code that fails later inside nvcc."""
    with pytest.raises(ValueError) as exc_info:
        _compile(CUDA_SM86_TARGET)

    message = str(exc_info.value)
    assert "requires sm_90" in message
    assert "sm_86" in message
    # Names the offending primitives so users know what to remove.
    assert "tl_shuffle_elect" in message or "Barrier" in message


@tilelang.testing.requires_cuda
def test_no_barrier_compiles_on_pre_hopper_target():
    """The guard must not fire for pre-Hopper kernels that use no barrier."""
    _compile(CUDA_SM86_TARGET, kernel_factory=_no_barrier_kernel)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_alloc_barrier_allowed_on_hopper_target():
    """The guard must not regress sm_90, where T.alloc_barrier() is valid."""
    _compile(CUDA_SM90_TARGET)


if __name__ == "__main__":
    tilelang.testing.main()
