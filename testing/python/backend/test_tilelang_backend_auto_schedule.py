"""Ascend's boolean scheduling flag and CUDA's scheduler name must coexist."""

import pytest

import tilelang
import tilelang.testing
from tilelang.backend.pass_pipeline.pipeline_utils import allow_autoschedule


def test_ascend_auto_schedule_defaults_to_enabled():
    with tilelang.transform.PassContext() as context:
        assert allow_autoschedule(context)


@pytest.mark.parametrize("enabled", [False, True])
def test_ascend_auto_schedule_preserves_boolean_option(enabled):
    with tilelang.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: enabled}) as context:
        assert bool(allow_autoschedule(context)) is enabled


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("enabled", [False, True])
def test_cuda_and_ascend_auto_schedule_options_coexist(enabled):
    with tilelang.transform.PassContext(
        config={
            tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: enabled,
            tilelang.PassConfigKey.TL_CUDA_AUTO_SCHEDULE: "role_based",
        }
    ) as context:
        assert bool(allow_autoschedule(context)) is enabled
        assert context.config[tilelang.PassConfigKey.TL_CUDA_AUTO_SCHEDULE] == "role_based"
        assert tilelang.ascend.transform.AutoSchedule().info.name == "tl.AutoSchedule"
        assert tilelang.cuda.transform.AutoSchedule().info.name == "tl.cuda.AutoSchedule"


if __name__ == "__main__":
    tilelang.testing.main()
