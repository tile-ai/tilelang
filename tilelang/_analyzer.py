"""TileLang's local proof policy on the upstream symbolic analyzer."""

from tvm_tirx import sym


def create_analyzer() -> sym.Analyzer:
    """Allow DEFAULT fallback without changing native rules or other passes.

    Premises remain local to the caller's ConstraintContext. In particular,
    this does not enable fallback on analyzers owned by upstream TileDispatch.
    """
    analyzer = sym.Analyzer()
    analyzer.set_z3_default_fallback(True)
    return analyzer
