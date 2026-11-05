"""Condensing a fused radix permutation must not duplicate replica owners."""

from collections import Counter

import pytest
from tvm_tirx import ir

from tilelang.layout import Fragment


def _evaluate(expr, values):
    if isinstance(expr, ir.IntImm):
        return int(expr.value)
    if isinstance(expr, ir.Var):
        return values[expr]
    operations = {
        ir.prim.Add: lambda a, b: a + b,
        ir.prim.Mul: lambda a, b: a * b,
        ir.prim.FloorDiv: lambda a, b: a // b,
        ir.prim.FloorMod: lambda a, b: a % b,
    }
    return operations[type(expr)](_evaluate(expr.a, values), _evaluate(expr.b, values))


@pytest.mark.parametrize("rows,columns,radix", [(6, 6, 4), (8, 6, 4), (3, 8, 6)])
@pytest.mark.parametrize("replica_axis", [0, 1])
def test_condense_disjoint_windows_preserves_owner_multiplicity(rows, columns, radix, replica_axis):
    extent = columns if replica_axis == 0 else rows
    replicas = rows if replica_axis == 0 else columns

    def owner(index, replica):
        flat = replica * columns + index if replica_axis == 0 else index * columns + replica
        return (flat % radix) * ((rows * columns) // radix) + flat // radix

    fragment = Fragment((extent,), forward_thread_fn=owner, forward_index_fn=lambda index: index, replicate=replicas)
    condensed = fragment.condense_rep_var()
    assert int(condensed.replicate_size) == replicas
    assert [int(value) for value in condensed.get_input_shape()] == [extent]

    # A replicated Fragment lists the replica coordinate before logical axes.
    replica_var, index_var = condensed.get_forward_vars()
    for index in range(extent):
        expected = Counter(owner(index, replica) for replica in range(replicas))
        actual = Counter(
            _evaluate(condensed.thread, {index_var: index, replica_var: replica}) for replica in range(int(condensed.replicate_size))
        )
        assert actual == expected
