"""Native source-parser entry point for TileLang's script dialect."""

from tvm_tirx.tirx.script import function


def prim_func(func=None, private=False, check_well_formed=True):
    return function(func, private=private, check_well_formed=check_well_formed)
