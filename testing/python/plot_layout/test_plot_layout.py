import pytest
import tilelang.language as T
from examples.plot_layout.fragment_mma_load_a import make_mma_load_base_layout
from tilelang.tools import plot_layout

def test_mma_load_base_layout():
    # Test float16 matrix A layout
    base_layout = make_mma_load_base_layout(dtype="float16", matrix="A", transposed=False)
    assert base_layout.shape == [8, 32]  # micro_size_r=8, micro_size_s=32 for float16
    
    # Test float16 matrix B layout
    base_layout_b = make_mma_load_base_layout(dtype="float16", matrix="B", transposed=False)
    assert base_layout_b.shape == [8, 32]
    
    # Test float8 matrix A layout
    base_layout_fp8 = make_mma_load_base_layout(dtype="float8", matrix="A", transposed=False)
    assert base_layout_fp8.shape == [16, 32]  # micro_size_r=16, micro_size_s=32 for float8

def test_layout_operations():
    base_layout = make_mma_load_base_layout(dtype="float16", matrix="A", transposed=False)
    
    # Test repeat operation
    repeated_layout = base_layout.repeat([2, 1], repeat_on_thread=True)
    assert repeated_layout.shape == [16, 32]  # doubled in first dimension
    
    # Test replicate operation
    replicated_layout = base_layout.replicate(2)
    assert replicated_layout.shape == [8, 64]  # doubled in second dimension

def test_layout_plot():
    base_layout = make_mma_load_base_layout(dtype="float16", matrix="A", transposed=False)
    
    # Test that plot_layout doesn't raise any errors
    try:
        plot_layout(base_layout, name="test_layout")
    except Exception as e:
        pytest.fail(f"plot_layout raised an exception: {e}")

def test_invalid_inputs():
    # Test invalid matrix type
    with pytest.raises(AssertionError):
        make_mma_load_base_layout(dtype="float16", matrix="C", transposed=False)
    
    # Test invalid dtype
    with pytest.raises(ValueError):
        make_mma_load_base_layout(dtype="float32", matrix="A", transposed=False)
    
    # Test transposed (not supported yet)
    with pytest.raises(AssertionError):
        make_mma_load_base_layout(dtype="float16", matrix="A", transposed=True) 