import tilelang.testing

import cumsum
import chunk_scaled_dot_kkt
import wy_fast
import wy_fast_bwd_split
import chunk_o
import chunk_o_bwd
import chunk_delta_h
import chunk_delta_bwd

def test_example_cumsum():
    cumsum.main()

def test_example_chunk_scaled_dot_kkt():
    chunk_scaled_dot_kkt.main()

def test_example_wy_fast():
    wy_fast.main()

def test_example_wy_fast_bwd_split():
    wy_fast_bwd_split.main()

def test_example_chunk_o():
    chunk_o.main()

def test_example_chunk_o_bwd():
    chunk_o_bwd.main()

def test_example_chunk_delta_h():
    chunk_delta_h.main()

def test_example_chunk_delta_bwd():
    chunk_delta_bwd.main()


if __name__ == "__main__":
    tilelang.testing.main()
