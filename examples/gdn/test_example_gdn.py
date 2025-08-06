import tilelang.testing

import example_cumsum
import example_chunk_scaled_dot_kkt
import example_wy_fast
import example_wy_fast_bwd_split
import example_chunk_o
import example_chunk_o_bwd
import example_chunk_delta_h
import example_chunk_delta_bwd


def test_example_cumsum():
    example_cumsum.main()


def test_example_chunk_scaled_dot_kkt():
    example_chunk_scaled_dot_kkt.main()


def test_example_wy_fast():
    example_wy_fast.main()


def test_example_wy_fast_bwd_split():
    example_wy_fast_bwd_split.main()


def test_example_chunk_o():
    example_chunk_o.main()


def test_example_chunk_o_bwd():
    example_chunk_o_bwd.main()


def test_example_chunk_delta_h():
    example_chunk_delta_h.main()


def test_example_chunk_delta_bwd():
    example_chunk_delta_bwd.main()


if __name__ == "__main__":
    tilelang.testing.main()
