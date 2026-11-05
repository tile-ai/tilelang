"""A pure division is not speculatively safe outside its zero-fill condition."""

from testing.python.cpu.test_async_width_contract import (
    CopyProgram,
    check,
    existing_native_operator_bridge,  # noqa: F401 - imported autouse pytest fixture
    integer,
)


def test_division_address_is_not_hoisted_from_zero_fill_branch(tmp_path):
    program = CopyProgram()
    source_index = 16 * (integer(1) // program.offset) + program.lane
    copy = program.copy(source_index=source_index, predicate=program.offset != 0)
    # offset=0 is a valid input: the original false branch stores +0 without
    # evaluating the address. It must not be moved into an eager call argument.
    check(program, [program.loop([copy])], tmp_path, expected_count=None)
