"""HIP's public target schema preserves the frozen fork's effective defaults."""

import json
import subprocess
import sys

import pytest
import tvm_tirx
from tilelang.rocm.target import with_rocm_target_attrs

Target = tvm_tirx.target.Target
RESOURCE_KEYS = {
    "max_num_threads",
    "max_threads_per_block",
    "max_shared_memory_per_block",
    "thread_warp_size",
}


def _plain(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Target):
        return _describe(value)
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "value"):
        return _plain(value.value)
    return [_plain(item) for item in value]


def _describe(target):
    return {
        "kind": target.kind.name,
        "keys": list(target.keys),
        "attrs": _plain(target.attrs),
        "tag": str(target.tag),
        "device_type": int(target.get_target_device_type()),
        "host": _describe(target.host) if target.host is not None else None,
    }


@pytest.mark.parametrize(
    "config,normalize,resources,keys,tag,host",
    [
        ({"kind": "hip", "mcpu": "gfx90a"}, False, {}, ["hip", "gpu"], "", False),
        (
            {"kind": "hip", "mcpu": "gfx1100", "thread_warp_size": 32},
            False,
            {"thread_warp_size": 32},
            ["hip", "gpu"],
            "",
            False,
        ),
        (
            {
                "kind": "hip",
                "mcpu": "gfx950",
                "mattr": ["+xnack"],
                "max_shared_memory_per_block": 131072,
                "max_num_threads": 512,
            },
            False,
            {"max_shared_memory_per_block": 131072, "max_num_threads": 512},
            ["hip", "gpu"],
            "",
            False,
        ),
        (
            {"kind": "hip", "mcpu": "gfx1200"},
            True,
            {"thread_warp_size": 32},
            ["hip", "gpu"],
            "",
            False,
        ),
        (
            {
                "kind": "hip",
                "mcpu": "gfx942",
                "keys": ["custom", "gpu"],
                "tag": "hip-regression",
                "host": {"kind": "c"},
            },
            False,
            {},
            ["custom", "gpu"],
            "hip-regression",
            True,
        ),
    ],
)
def test_hip_target_public_attributes(config, normalize, resources, keys, tag, host):
    target = Target(config)
    if normalize:
        target = with_rocm_target_attrs(target)
    observed = _describe(target)
    assert observed["kind"] == "hip" and observed["device_type"] == 10
    assert observed["keys"] == keys and observed["tag"] == tag
    assert observed["attrs"]["mcpu"] == config["mcpu"]
    assert observed["attrs"]["mtriple"] == "amdgcn-amd-amdhsa-hcc"
    assert {key: value for key, value in observed["attrs"].items() if key in RESOURCE_KEYS} == resources
    if "mattr" in config:
        assert observed["attrs"]["mattr"] == config["mattr"]
    assert (observed["host"] is not None) is host
    if host:
        assert observed["host"]["kind"] == "c"
    assert _describe(Target(target.export())) == observed


@pytest.mark.parametrize(
    "config,message",
    [
        ("hip -mcpu=gfx1100 -thread_warp_size=32", "CLI target string form"),
        ({"kind": "hip", "mcpu": "sm_100"}, "invalid GFX version"),
        ({"kind": "hip", "mcpu": "gfx90a", "mtriple": "x86_64-linux-gnu"}, "mtriple"),
        ({"kind": "hip", "mcpu": 950}, "mcpu"),
        ({"kind": "hip", "mcpu": "gfx90a", "unknown_option": 1}, "unknown_option"),
    ],
)
def test_hip_target_rejects_invalid_configuration(config, message):
    with pytest.raises(ValueError, match=message):
        Target(config)


@pytest.mark.parametrize("normalize", [False, True])
def test_unknown_hip_architecture_preserves_normalizer_fallback(normalize):
    target = Target({"kind": "hip", "mcpu": "gfx9999"})
    if normalize:
        target = with_rocm_target_attrs(target)
    resources = {key: value for key, value in _describe(target)["attrs"].items() if key in RESOURCE_KEYS}
    assert resources == ({"thread_warp_size": 64} if normalize else {})


def test_rocm_target_retains_its_separate_defaults():
    target = Target({"kind": "rocm", "mcpu": "gfx90a"})
    assert target.attrs["max_num_threads"] == 256
    assert target.attrs["max_threads_per_block"] == 256
    assert target.attrs["max_shared_memory_per_block"] == 65536
    assert target.attrs["thread_warp_size"] == 64


def test_hip_arch_callback_does_not_introduce_resource_defaults(tmp_path):
    source = """
import json
import tilelang
import tvm_tirx
import tvm_ffi
tvm_ffi.register_global_func("tvm_callback_rocm_get_arch", lambda: "gfx942", override=True)
target = tvm_tirx.target.Target({"kind": "hip"})
assert target.kind.name == "hip"
assert list(target.keys) == ["hip", "gpu"]
assert target.attrs["mcpu"] == "gfx942"
assert target.attrs["mtriple"] == "amdgcn-amd-amdhsa-hcc"
resource_keys = {
    "max_num_threads", "max_threads_per_block",
    "max_shared_memory_per_block", "thread_warp_size",
}
assert not set(target.attrs).intersection(resource_keys)
print(json.dumps({"passed": True, "compiler": tvm_tirx.__file__}))
"""
    script = tmp_path / "hip_callback.py"
    script.write_text(source)
    result = subprocess.run(
        [sys.executable, "-B", str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1])["passed"]
