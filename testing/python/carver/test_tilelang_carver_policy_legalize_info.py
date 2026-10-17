"""CPU regression test for the ``_legalize_info`` fallback in TensorCorePolicy.

When the ``pipeline_stage`` / ``use_async_copy`` tags are absent the policy
falls back on ``arch.compute_capability``.  ``CUDA.__init__`` fills that with
the bare digits of ``device.compute_version`` ("80", "90"), but the fallback
compared it against ``{"sm_80", "sm_90", "sm_90a"}`` -- never true -- so every
CUDA arch silently got 1 pipeline stage and no ``cp.async`` on this path.

Now gated on ``arch.sm_version in {80, 90}`` -- the faithful translation of the
old literal set (``sm_90a`` also parses to 90), and the same membership the tag
path in ``matmul_analysis.py`` uses.  Whether sm_86/89/100+ should enable the
pipeline too is a separate behavioural question, deliberately not part of this
fix.  No GPU is needed: the policy is
built without a device, following ``test_tilelang_carver_sm_version.py``.
"""

import tilelang.testing
from tilelang.carver.arch.arch_base import TileDevice
from tilelang.carver.arch.cuda import CUDA
from tilelang.carver.roller.policy.tensorcore import TensorCorePolicy


def _cuda_arch(sm_version: int, compute_max_core: int = 108) -> CUDA:
    arch = CUDA.__new__(CUDA)
    arch.sm_version = sm_version
    arch.compute_capability = str(sm_version)
    arch.compute_max_core = compute_max_core
    arch.l2_cache_size_bytes = 40 << 20
    return arch


class _NonCudaArch(TileDevice):
    """A device that is neither CUDA nor RDNA (e.g. CDNA): the gates must not
    touch ``sm_version`` on it."""

    def __init__(self):
        self.compute_capability = "gfx942"
        self.compute_max_core = 304
        self.l2_cache_size_bytes = 256 << 20


class _Node:
    def __init__(self, tags=None, input_buffers=()):
        self._tags = tags or {}
        self.input_buffers = list(input_buffers)

    def get_tag(self, k):
        return self._tags.get(k)


class _Buffer:
    def __init__(self, shape, dtype="float16"):
        self.shape = shape
        self.dtype = dtype


def _policy(arch, node) -> TensorCorePolicy:
    policy = TensorCorePolicy.__new__(TensorCorePolicy)
    policy.arch = arch
    policy.prim_func_node = node
    policy.ordered_nodes = [node]
    return policy


def test_legalize_info_tags_win():
    policy = _policy(_cuda_arch(75), _Node({"pipeline_stage": 3, "use_async_copy": True}))
    policy._legalize_info()
    assert policy.pipeline_stage == 3
    assert policy.use_async_copy is True


def test_legalize_info_fallback_sm80_sm90():
    for sm in (80, 90):
        policy = _policy(_cuda_arch(sm), _Node())
        policy._legalize_info()
        assert policy.pipeline_stage == 2, sm
        assert policy.use_async_copy is True, sm


def test_legalize_info_fallback_other_cuda_arches():
    # Same membership as the tag path in matmul_analysis.py: only {80, 90}
    # enable the 2-stage pipeline + cp.async; everything else stays at 1/False.
    for sm in (70, 75, 86, 89, 100, 120):
        policy = _policy(_cuda_arch(sm), _Node())
        policy._legalize_info()
        assert policy.pipeline_stage == 1, sm
        assert policy.use_async_copy is False, sm


def test_legalize_info_fallback_non_cuda_arch():
    policy = _policy(_NonCudaArch(), _Node())
    policy._legalize_info()
    assert policy.pipeline_stage == 1
    assert policy.use_async_copy is False


if __name__ == "__main__":
    tilelang.testing.main()
