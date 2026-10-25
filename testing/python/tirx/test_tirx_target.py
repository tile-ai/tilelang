import pytest
from tvm_tirx.target import Target

from tilelang._compat.target import normalize_target


def test_cli_options_are_typed_and_preserved():
    target = normalize_target("cuda -arch=sm_90 -max_num_threads=256 -keys=cuda,gpu")
    assert target.attrs["arch"] == "sm_90"
    assert target.attrs["max_num_threads"] == 256
    assert list(target.keys) == ["cuda", "gpu"]


def test_cli_space_separated_values():
    target = normalize_target("cuda -arch sm_90 -max_num_threads 256")
    assert target.attrs["arch"] == "sm_90"
    assert target.attrs["max_num_threads"] == 256


def test_dict_input_is_not_mutated():
    config = {"kind": "cuda", "arch": "sm_90", "host": {"kind": "c"}}
    target = normalize_target(config)
    assert config["host"] == {"kind": "c"}
    assert target.host.kind.name == "c"


def test_existing_target_keeps_identity():
    target = Target("c")
    assert normalize_target(target).same_as(target)


def test_json_and_host():
    target = normalize_target('{"kind": "cuda", "arch": "sm_90"}', host="c")
    assert target.attrs["arch"] == "sm_90"
    assert target.host.kind.name == "c"


@pytest.mark.parametrize(
    "target",
    [
        "",
        "cuda -tl.fake=true",
        "cuda -arch",
        "cuda -arch=sm_90 -arch=sm_80",
        "cuda unexpected",
        "cuda -max_num_threads=not-an-int",
    ],
)
def test_bad_options_are_not_silently_dropped(target):
    with pytest.raises(ValueError):
        normalize_target(target)
