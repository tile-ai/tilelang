from tilelang import tvm


def test_disable_fast_math_pass_config_is_removed():
    configs = tvm.transform.PassContext.list_configs()

    assert "tl.disable_fast_math" not in configs
    assert "tl.enable_fast_math" in configs
