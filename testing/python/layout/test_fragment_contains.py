"""Regression tests for fragment thread ownership containment."""

import ctypes
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import tilelang
import tvm_ffi
from tilelang import tvm
from tilelang.env import TL_LIBS
from tvm_ffi import libinfo as ffi_libinfo

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="The test bridge requires a Linux shared library")


@pytest.fixture(scope="module")
def prove_contains(tmp_path_factory):
    root = Path(__file__).resolve().parents[3]
    output = tmp_path_factory.mktemp("fragment_contains") / "probe.so"
    library_dirs = list(
        dict.fromkeys([Path(tilelang._LIB_PATH).parent, *(Path(path) for path in TL_LIBS), Path(ffi_libinfo.find_libtvm_ffi()).parent])
    )
    includes = [
        root / "src",
        root / "3rdparty/tvm/include",
        root / "3rdparty/tvm/src",
        root / "3rdparty/tvm/3rdparty/tvm-ffi/include",
        root / "3rdparty/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include",
    ]
    command = shlex.split(os.environ.get("CXX", "c++"))
    command += ["-std=c++17", "-shared", "-fPIC", "-O2"]
    command += [f"-I{path}" for path in includes]
    command += [f"-L{path}" for path in library_dirs]
    command += [f"-Wl,-rpath,{path}" for path in library_dirs]
    command += [
        str(root / "testing/cpp/fragment_contains_probe.cc"),
        "-ltilelang",
        "-ltvm_compiler",
        "-ltvm_runtime",
        "-ltvm_ffi",
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    _library_handle = ctypes.CDLL(str(output), mode=ctypes.RTLD_GLOBAL)
    function = tvm.get_global_func("tl.testing.ProveFragmentContains")

    def call(small, large, small_min, large_min, small_index, large_index, check_index, small_extent=128, large_extent=128):
        return function(small, large, small_min, large_min, small_index, large_index, check_index, small_extent, large_extent)

    yield call


def replicated(count, *, stride=1, slot=0):
    return tilelang.Fragment(
        (1,),
        forward_thread_fn=lambda index, replica: replica * stride,
        forward_index_fn=lambda index: slot,
        replicate=count,
    )


def replicated_pair(count, first, second):
    return tilelang.Fragment(
        (1,),
        forward_thread_fn=lambda index, replica: replica,
        forward_index_fn=lambda index: tvm_ffi.convert([first, second]),
        replicate=count,
    )


def test_fully_replicated_requires_thread_coverage(prove_contains):
    assert not prove_contains(replicated(128), replicated(96), 0, 0, 0, 0, False)


@pytest.mark.parametrize("small_count,large_count", [(96, 128), (96, 96)])
def test_fully_replicated_accepts_covered_threads(prove_contains, small_count, large_count):
    assert prove_contains(replicated(small_count), replicated(large_count), 0, 0, 0, 0, False)


def test_fully_replicated_checks_physical_slots_when_requested(prove_contains):
    small = replicated(128, slot=0)
    large = replicated(128, slot=1)
    assert prove_contains(small, large, 0, 0, 0, 0, False)
    assert not prove_contains(small, large, 0, 0, 0, 0, True)
    assert prove_contains(small, replicated(128, slot=0), 0, 0, 0, 0, True)
    assert not prove_contains(replicated_pair(128, 0, 1), replicated_pair(128, 0, 2), 0, 0, 0, 0, True)
    assert prove_contains(replicated_pair(128, 0, 1), replicated_pair(128, 0, 1), 0, 0, 0, 0, True)


@pytest.mark.parametrize("count,expected", [(48, True), (64, False)])
def test_noncontiguous_owners_require_actual_coverage(prove_contains, count, expected):
    assert bool(prove_contains(replicated(count, stride=2), replicated(96), 0, 0, 0, 0, False)) is expected


@pytest.mark.parametrize(
    "small_min,large_min,small_count,large_count,expected",
    [(32, 32, 32, 64, True), (32, 0, 32, 128, True), (0, 32, 64, 64, False)],
)
def test_nonzero_thread_ranges_compare_physical_owners(prove_contains, small_min, large_min, small_count, large_count, expected):
    assert bool(prove_contains(replicated(small_count), replicated(large_count), small_min, large_min, 0, 0, False)) is expected


@pytest.mark.parametrize("small_count,expected", [(128, False), (64, True)])
def test_replication_cannot_cover_threads_outside_its_bound_range(prove_contains, small_count, expected):
    assert bool(prove_contains(replicated(small_count), replicated(128), 0, 0, 0, 0, False, 128, 96)) is expected


def test_fully_replicated_respects_small_bound_range(prove_contains):
    assert prove_contains(replicated(128), replicated(128), 0, 0, 0, 0, False, 96, 96)


@pytest.mark.parametrize(
    "stride,small_min,large_min,small_extent,large_extent,expected",
    [
        (1, 0, 0, 128, 96, False),
        (1, 0, 0, 64, 96, True),
        (2, 0, 0, 96, 96, True),
        (2, 0, 0, 96, 64, False),
        (1, 32, 0, 64, 96, True),
        (1, 32, 0, 96, 96, False),
        (1, 0, 32, 64, 96, False),
        (2, 32, 32, 96, 96, True),
    ],
)
def test_small_bound_range_thread_coverage(prove_contains, stride, small_min, large_min, small_extent, large_extent, expected):
    assert (
        bool(prove_contains(replicated(128, stride=stride), replicated(128), small_min, large_min, 0, 0, False, small_extent, large_extent))
        is expected
    )


def test_bounded_thread_coverage_matches_enumerated_owners(prove_contains):
    for stride in (1, 2, 3):
        for small_min in (0, 16):
            for large_min in (0, 16):
                for small_extent in (32, 64):
                    for large_extent in (32, 96):
                        small_threads = {small_min + stride * r for r in range(32) if stride * r < small_extent}
                        large_threads = set(range(large_min, large_min + large_extent))
                        expected = small_threads.issubset(large_threads)
                        actual = bool(
                            prove_contains(
                                replicated(32, stride=stride),
                                replicated(128),
                                small_min,
                                large_min,
                                0,
                                0,
                                False,
                                small_extent,
                                large_extent,
                            )
                        )
                        assert actual is expected, (stride, small_min, large_min, small_extent, large_extent)


@pytest.mark.parametrize(
    "small_min,large_min,small_count,large_count,expected",
    [(-1, -1, 96, 128, True), (-1, -1, 128, 96, False), (0, -1, 96, 128, True), (-1, 0, 96, 128, True)],
)
def test_unbound_thread_ranges_use_normalized_coordinates(prove_contains, small_min, large_min, small_count, large_count, expected):
    assert bool(prove_contains(replicated(small_count), replicated(large_count), small_min, large_min, 0, 0, False)) is expected


@pytest.mark.parametrize("large_count,expected", [(32, True), (24, False)])
def test_partially_replicated_owners_with_symbolic_logical_index(prove_contains, large_count, expected):
    small = tilelang.Fragment(
        (4,),
        forward_thread_fn=lambda index, replica: index * 8 + replica,
        forward_index_fn=lambda index: index,
        replicate=8,
    )
    large = tilelang.Fragment(
        (4,),
        forward_thread_fn=lambda index, replica: replica,
        forward_index_fn=lambda index: index,
        replicate=large_count,
    )
    assert bool(prove_contains(small, large, 0, 0, -1, -1, True)) is expected


def test_static_thread_coverage_matches_enumerated_owners(prove_contains):
    """Compare the proof with independently enumerated physical thread sets."""
    for small_count in (8, 16, 32):
        for stride in (1, 2, 3):
            for small_min in (0, 16):
                for large_min in (0, 16):
                    for large_count in (16, 48, 96):
                        for large_extent in (32, 128):
                            small_threads = {small_min + stride * r for r in range(small_count)}
                            large_threads = set(range(large_min, large_min + min(large_count, large_extent)))
                            expected = small_threads.issubset(large_threads)
                            actual = bool(
                                prove_contains(
                                    replicated(small_count, stride=stride),
                                    replicated(large_count),
                                    small_min,
                                    large_min,
                                    0,
                                    0,
                                    False,
                                    128,
                                    large_extent,
                                )
                            )
                            assert actual is expected, (small_count, stride, small_min, large_min, large_count, large_extent)
