"""Ascend's boolean scheduling flag and CUDA's scheduler name must coexist."""

import pytest

import tilelang
import tilelang.testing
from tilelang.ascend.pipeline import allow_autoschedule

# These exercise Ascend's boolean scheduling flag. A build without USE_ASCEND
# does not register tl.enable_auto_schedule at all.
# These only need the Ascend backend compiled in, not an NPU attached, so they
# take the compile-only marks and keep running on a host without a device.
pytestmark = tilelang.testing.requires_ascend.marks("compile-only")


def test_ascend_auto_schedule_defaults_to_enabled():
    with tilelang.transform.PassContext() as context:
        assert allow_autoschedule(context)


@pytest.mark.parametrize("enabled", [False, True])
def test_ascend_auto_schedule_preserves_boolean_option(enabled):
    with tilelang.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: enabled}) as context:
        assert bool(allow_autoschedule(context)) is enabled


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("enabled", [False, True])
def test_cuda_warp_specialization_and_ascend_auto_schedule_options_coexist(enabled):
    with tilelang.transform.PassContext(
        config={
            tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: enabled,
            tilelang.PassConfigKey.TL_ENABLE_AUTO_WARP_SPECIALIZATION: "role_based",
        }
    ) as context:
        assert bool(allow_autoschedule(context)) is enabled
        assert context.config[tilelang.PassConfigKey.TL_ENABLE_AUTO_WARP_SPECIALIZATION] == "role_based"
        assert tilelang.ascend.transform.AutoSchedule().info.name == "tl.AutoSchedule"
        assert tilelang.cuda.transform.AutoWarpSpecialization().info.name == "tl.AutoWarpSpecialization"


if __name__ == "__main__":
    tilelang.testing.main()
