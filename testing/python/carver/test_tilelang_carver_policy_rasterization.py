"""CPU regression test for the arch gate in ``TensorCorePolicy.plan_rasterization``.

The gate compared ``arch.compute_capability`` -- the bare digit string
``CUDA.__init__`` derives from ``device.compute_version`` -- as a *string*:
``compute_capability < "80"``.  Lexicographically ``"100" < "80"`` and
``"120" < "80"``, so Blackwell targets never received a rasterization plan
while sm_86 / sm_89 did.  Now gated on ``arch.sm_version`` (an int).

No GPU is needed: the policy is built without a device, following
``test_tilelang_carver_sm_version.py``.
"""

import tilelang.testing
from tilelang.carver.arch.arch_base import TileDevice
from tilelang.carver.arch.cuda import CUDA
from tilelang.carver.roller.policy.tensorcore import TensorCorePolicy
from tilelang.carver.roller.rasterization import NoRasterization, Rasterization2DColumn


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
        self.l2_cache_size_bytes = 4 << 20  # MI300X L2 per XCD; must be < _BIG (64 MiB)


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


_BIG = [_Buffer([4096, 4096]), _Buffer([4096, 4096])]  # 64 MiB > l2


def test_rasterization_enabled_on_ampere_and_newer():
    for sm in (80, 86, 89, 90, 100, 120):
        plan = _policy(_cuda_arch(sm), _Node(input_buffers=_BIG)).plan_rasterization(None)
        assert isinstance(plan, Rasterization2DColumn), sm
        assert plan.panel_width_ == int(108**0.5)


def test_rasterization_disabled_pre_ampere():
    for sm in (70, 75):
        plan = _policy(_cuda_arch(sm), _Node(input_buffers=_BIG)).plan_rasterization(None)
        assert isinstance(plan, NoRasterization), sm


def test_rasterization_disabled_when_inputs_fit_l2():
    small = [_Buffer([256, 256]), _Buffer([256, 256])]
    plan = _policy(_cuda_arch(80), _Node(input_buffers=small)).plan_rasterization(None)
    assert isinstance(plan, NoRasterization)


def test_rasterization_non_cuda_arch_unchanged():
    # Non-CUDA devices were never gated by the string compare ("gfx942" > "80");
    # keep that behaviour.
    plan = _policy(_NonCudaArch(), _Node(input_buffers=_BIG)).plan_rasterization(None)
    assert isinstance(plan, Rasterization2DColumn)


if __name__ == "__main__":
    tilelang.testing.main()
