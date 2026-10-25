import pytest

from tilelang.cuda.intrinsics.layout.mma_sp_layout import (
    metadata_8bit_load_32x4_to_shared_16x4_layout_32bit,
    metadata_16bit_load_32x2_to_shared_16x2_layout_32bit,
)


@pytest.mark.parametrize("thread_id", range(32))
@pytest.mark.parametrize("local_id", range(4))
def test_tf32_metadata_byte_layout_matches_halfword_layout(thread_id, local_id):
    row, column = metadata_8bit_load_32x4_to_shared_16x4_layout_32bit(thread_id, local_id)
    halfword_row, halfword_column = metadata_16bit_load_32x2_to_shared_16x2_layout_32bit(thread_id, local_id // 2)
    assert 0 <= row < 16
    assert 0 <= column < 4
    assert row == halfword_row
    assert column == halfword_column * 2 + local_id % 2
