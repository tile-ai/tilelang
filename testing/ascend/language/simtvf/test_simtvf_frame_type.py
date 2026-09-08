"""Tests for SimtVFFrame type after ForFrame→SimtVFFrame refactor."""

import tilelang.language as T
from tilelang.ascend.language.frame import SimtVFFrame
from tvm.tirx.script.builder.frame import TIRFrame, ForFrame


def test_simtvf_returns_simtvf_frame():
    """SimtVF() should return a SimtVFFrame instance."""
    f = T.SimtVF(128)
    assert isinstance(f, SimtVFFrame), f"Expected SimtVFFrame, got {type(f)}"


def test_simtvf_frame_is_tir_frame():
    """SimtVFFrame should be a subclass of TIRFrame."""
    assert issubclass(SimtVFFrame, TIRFrame)


def test_simtvf_frame_is_not_for_frame():
    """SimtVFFrame should NOT be a subclass of ForFrame."""
    assert not issubclass(SimtVFFrame, ForFrame)


def test_simtvf_frame_exported():
    """SimtVFFrame should be importable from tilelang.language."""
    from tilelang.language import SimtVFFrame as SF

    assert SF is SimtVFFrame
