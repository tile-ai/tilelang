"""A straddling logical axis can still cover a compact physical domain."""

import itertools

import pytest

import tilelang
from tvm_tirx import ir


@pytest.mark.parametrize("rows,groups,lanes,threads", [(2, 6, 32, 128), (3, 10, 8, 16)])
def test_straddling_radix_inverse_covers_both_domains(rows, groups, lanes, threads):
    total = rows * groups * lanes
    layout = tilelang.Layout(
        [rows, groups, lanes],
        lambda i, j, k: [(i * groups + j) // (threads // lanes), (i * groups * lanes + j * lanes + k) % threads],
    )
    inverse = layout.inverse()
    for i, j, k in itertools.product(range(rows), range(groups), range(lanes)):
        linear = (i * groups + j) * lanes + k
        actual = inverse.map_forward_index([ir.const(linear // threads), ir.const(linear % threads)])
        assert [int(value) for value in actual] == [i, j, k]
    for slot, thread in itertools.product(range(total // threads), range(threads)):
        logical = inverse.map_forward_index([ir.const(slot), ir.const(thread)])
        assert all(0 <= int(value) < extent for value, extent in zip(logical, (rows, groups, lanes)))
        physical = layout.map_forward_index(logical)
        assert [int(value) for value in physical] == [slot, thread]


def test_compaction_does_not_accept_unoccupied_physical_coordinates():
    layout = tilelang.Layout([2, 3, 64], lambda i, j, k: [i * 3 + j, (i * 192 + j * 64 + k) % 128])
    with pytest.raises(Exception, match="does not form a bijective transform|do not traverse full iter space"):
        layout.inverse()


def test_equal_domain_sizes_do_not_admit_shifted_out_of_bounds_image():
    layout = tilelang.Layout(
        [2, 6, 32],
        lambda i, j, k: [(i * 6 + j) // 4 + 1, (i * 192 + j * 32 + k) % 128 - 32],
    )
    # Both rectangular domains contain 384 points. The algebraic inverse at
    # physical (0, 0) would still escape the logical domain as (-1, 3, 0).
    assert [int(value) for value in layout.get_output_shape()] == [4, 96]
    with pytest.raises(Exception, match="bijective transform|full iter space|leaves its .* domain"):
        layout.inverse()
